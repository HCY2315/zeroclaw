//! # ModelProvider（模型供应商）抽象层
//!
//! 定义 ZeroClaw 访问各家大模型供应商（Anthropic / OpenAI / Gemini / Ollama /
//! OpenRouter / 各类兼容端点等）的行业中立抽象；各厂商 SDK 的具体实现位于
//! `zeroclaw-providers` crate。
//!
//! 核心内容：
//! - `ModelProvider` trait：供应商总接口（能力声明、普通 / 多轮 / 工具聊天、流式聊天）。
//! - 消息载体：`ChatMessage`（对话消息）、`ChatRequest`（请求载荷）、`ChatResponse`
//!   （响应：文字 + 工具调用 + 思考内容）、`ConversationMessage`（多轮会话消息）。
//! - 流式类型：`StreamChunk` / `StreamEvent`（结构化事件，含原生工具调用信号）。
//! - 能力与默认值：`ProviderCapabilities` 与 `BASELINE_*` 系列常量。
//! - 扩展思考：`NativeThinkingParams` / `ThinkingDisplay`。
//! - 工具协议：`ToolsPayload`（供应商原生格式或提示词引导文本）。

use crate::tool::ToolSpec;
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::sync::Arc;

// ── 扩展思考（extended thinking）参数 ─────────────────────────

/// 扩展思考预算 token 上限。
pub const MAX_BUDGET_TOKENS: u32 = 128_000;
/// Anthropic's documented minimum for extended-thinking `budget_tokens`.
/// Requests below this are rejected with 400 by the provider; clamping at
/// resolution time gives a clearer error site than the first API call.
///
/// 中文：Anthropic 官方规定的 extended-thinking `budget_tokens` 下限；
/// 低于该值的请求会被供应商以 400 拒绝，所以在解析阶段先夹紧，
/// 比等到第一次 API 调用才暴露错误更清晰。
pub const MIN_BUDGET_TOKENS: u32 = 1_024;

/// Parameters for native extended thinking support.
/// 中文：原生「扩展思考」参数。`budget_tokens` 为思考预算；
/// `display` 控制思考块在流式响应中以何种形式返回（见 `ThinkingDisplay`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeThinkingParams {
    pub budget_tokens: u32,
    /// Requests Anthropic's `thinking.display` beta
    /// (`thinking-display-updates-2026-08-18`), which controls whether
    /// thinking blocks come back omitted, as progress updates, or
    /// summarized. `None` leaves the field out of the request entirely,
    /// matching pre-beta behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<ThinkingDisplay>,
}

/// Anthropic's `thinking.display` request field (beta
/// `thinking-display-updates-2026-08-18`), controlling whether thinking
/// blocks come back omitted, as progress updates, or summarized.
///
/// 中文：`thinking.display` 字段的取值——思考块以哪种形式回流：
/// `Omitted`（省略，即不返回）/ `Updates`（作为流式进度更新）/ `Summarized`（摘要）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    Omitted,
    Updates,
    Summarized,
}

impl ThinkingDisplay {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Omitted => "omitted",
            Self::Updates => "updates",
            Self::Summarized => "summarized",
        }
    }
}

// ── 对话消息与上下文修剪 ─────────────────────────────────────

/// A single message in a conversation.
/// 中文：一条对话消息（`role` + `content` 极简模型）。
/// 上下文过大被压缩时，会以 `PRUNED_*` 占位消息的形式留在历史里（见下方常量）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// 上下文修剪占位符：整段「工具调用 → 结果」被折叠成一条摘要的边界标记。
/// 折叠摘要形如 `[Tool exchange: N tool call(s) — results collapsed]`。
pub const PRUNED_TOOL_EXCHANGE_SUMMARY_PREFIX: &str = "[Tool exchange:";
pub const PRUNED_TOOL_EXCHANGE_SUMMARY_SUFFIX: &str = "results collapsed]";
/// 上下文修剪的分隔占位：标记「上下文在这里继续」，防止被修剪掉的中间内容
/// 与后续内容粘连（代理循环据此判断哪些占位条目可以安全丢弃）。
pub const PRUNED_CONTEXT_SEPARATOR: &str = "[context continues]";

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
        }
    }

    pub fn pruned_tool_exchange_summary(tool_count: usize) -> String {
        format!(
            "{PRUNED_TOOL_EXCHANGE_SUMMARY_PREFIX} {tool_count} tool call(s) — {PRUNED_TOOL_EXCHANGE_SUMMARY_SUFFIX}"
        )
    }

    pub fn pruned_context_separator() -> Self {
        Self::user(PRUNED_CONTEXT_SEPARATOR)
    }

    pub fn is_pruned_tool_exchange_summary(&self) -> bool {
        self.role == "assistant"
            && self
                .content
                .starts_with(PRUNED_TOOL_EXCHANGE_SUMMARY_PREFIX)
            && self.content.contains(PRUNED_TOOL_EXCHANGE_SUMMARY_SUFFIX)
    }

    pub fn is_pruned_context_separator(&self) -> bool {
        self.role == "user" && self.content.trim() == PRUNED_CONTEXT_SEPARATOR
    }

    pub fn should_skip_internal_pruning_marker(messages: &[Self], index: usize) -> bool {
        let Some(msg) = messages.get(index) else {
            return false;
        };
        if msg.is_pruned_tool_exchange_summary() {
            return true;
        }
        msg.is_pruned_context_separator()
            && index
                .checked_sub(1)
                .and_then(|previous| messages.get(previous))
                .is_some_and(Self::is_pruned_tool_exchange_summary)
    }

    pub fn is_system(&self) -> bool {
        self.role == "system"
    }

    pub fn is_user(&self) -> bool {
        self.role == "user"
    }

    /// 中文：修正「回合次序」——删除开头孤立出现的 assistant / tool 消息，
    /// 保证非 system 消息序列以一条 `user` 消息打头，避免把残片历史发往供应商。
    pub fn sanitize_leading_turn_order(messages: &mut Vec<Self>) {
        let first_non_system = messages
            .iter()
            .position(|m| !m.is_system())
            .unwrap_or(messages.len());
        let mut drop_to = first_non_system;
        while drop_to < messages.len() && !messages[drop_to].is_user() {
            drop_to += 1;
        }
        if drop_to > first_non_system {
            messages.drain(first_non_system..drop_to);
        }
    }
}

// ── 工具调用 / Token 用量 / 响应类型 ─────────────────────────

/// A tool call requested by the LLM.
/// 中文：模型请求的一次工具调用。`id` / `name` / `arguments`（JSON 字符串）；
/// `extra_content` 为供应商特有的不透明扩展字段，必须在后续回合中原样回传
/// （如 Gemini 3 的 `thought_signature`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    /// ModelProvider-specific opaque extension fields that must round-trip
    /// unchanged on follow-up turns (e.g. Gemini 3 `thoughtSignature`
    /// carried as `extra_content.google.thought_signature`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<serde_json::Value>,
}

/// 中文：Token 用量统计。`input_tokens` 总提示长度（含缓存读写部分），
/// `cached_input_tokens` 本次命中缓存的输入子集，`cache_creation_input_tokens`
/// 本次写入缓存的输入子集（供应商按更贵的「缓存写入」费率计费）。
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Total prompt size: uncached + cached input tokens (including the
    /// cache-write subset when the provider reports it separately).
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Subset of `input_tokens` that was served from the model_provider's
    /// prompt cache (Anthropic `cache_read_input_tokens`,
    /// OpenAI `prompt_tokens_details.cached_tokens`).
    pub cached_input_tokens: Option<u64>,
    /// Subset of `input_tokens` that the model_provider wrote into its
    /// prompt cache on this request (Anthropic
    /// `cache_creation_input_tokens`, OpenAI-compatible
    /// `prompt_tokens_details.cache_creation_input_tokens`). Providers
    /// bill these at a premium over the plain input rate.
    pub cache_creation_input_tokens: Option<u64>,
}

/// An LLM response that may contain text, tool calls, or both.
/// 中文：模型响应。`text` 文字（工具调用型可能为空）、`tool_calls` 工具调用、
/// `usage` 用量、`reasoning_content` 思考内容（不透明保留、需回传给供应商）。
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Text content of the response (may be empty if only tool calls).
    pub text: Option<String>,
    /// Tool calls requested by the LLM.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage reported by the model_provider, if available.
    pub usage: Option<TokenUsage>,
    /// Raw reasoning/thinking content from thinking models (e.g. DeepSeek-R1,
    /// Kimi K2.5, GLM-4.7). Preserved as an opaque pass-through so it can be
    /// sent back in subsequent API requests — some model_providers reject tool-call
    /// history that omits this field.
    pub reasoning_content: Option<String>,
}

/// A transport-successful provider result that cannot complete a request.
///
/// The result has neither user-visible final text nor native tool calls.
/// Reasoning is intentionally not part of this contract because it is opaque
/// provider round-trip metadata rather than a final answer.
///
/// 中文：传输成功、但「语义上为空」的终态结果——既没有用户可见的最终文本，
/// 也没有原生工具调用。思考内容被刻意排除在判定之外：它只是需回传给供应商的
/// 不透明元数据，不是最终答案。用于把「空响应」表达为类型化错误而非成功返回。
#[derive(Debug)]
pub struct SemanticEmptyTerminalCompletion;

impl std::fmt::Display for SemanticEmptyTerminalCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("provider completed without final text or tool calls")
    }
}

impl std::error::Error for SemanticEmptyTerminalCompletion {}

impl ChatResponse {
    /// True when the LLM wants to invoke at least one tool.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Convenience: return text content or empty string.
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }

    /// True when this response cannot make progress or complete a turn.
    ///
    /// Reasoning content is intentionally excluded: it may need to be
    /// round-tripped to a provider, but it is not a user-visible final answer.
    /// A response containing one or more tool calls remains valid even when
    /// its text is empty.
    pub fn is_semantically_empty_terminal(&self) -> bool {
        strip_think_tags(self.text_or_empty()).is_empty() && self.tool_calls.is_empty()
    }
}

/// Remove inline `<think>...</think>` reasoning before terminal-response
/// classification or user-visible parsing.
///
/// An unclosed opening tag suppresses the remainder so partial reasoning never
/// becomes final output.
///
/// 中文：在「终态响应判定 / 用户可见解析」前剥除内联的 ` thinking... response`
/// 思考片段。注意若开头标签未闭合，则剩余内容全部被丢弃，保证半截思考
/// 永远不会被当作最终输出。
pub fn strip_think_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    loop {
        if let Some(start) = remaining.find("<think>") {
            result.push_str(&remaining[..start]);
            if let Some(end) = remaining[start..].find("</think>") {
                remaining = &remaining[start + end + "</think>".len()..];
            } else {
                break;
            }
        } else {
            result.push_str(remaining);
            break;
        }
    }
    result.trim().to_string()
}

// ── 请求 / 多轮会话类型 ──────────────────────────────────────

/// Request payload for model_provider chat calls.
/// 中文：非流式 chat 的请求载荷：消息列表 + 可选工具定义 + 可选扩展思考参数。
#[derive(Debug, Clone, Copy)]
pub struct ChatRequest<'a> {
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
    /// Native extended thinking parameters. When `Some`, providers that
    /// support extended thinking should send a dedicated thinking budget
    /// in the API request and force `temperature = 1.0`.
    pub thinking: Option<NativeThinkingParams>,
}

/// A tool result to feed back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub content: String,
    #[serde(default)]
    pub tool_name: String,
}

/// A message in a multi-turn conversation, including tool interactions.
/// 中文：多轮会话中的消息（serde tag 为 `type`）：
/// `Chat`（普通文本消息）、`AssistantToolCalls`（助手发出工具调用，保留历史保真）、
/// `ToolResults`（工具执行结果回喂给模型）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ConversationMessage {
    /// Regular chat message (system, user, assistant).
    Chat(ChatMessage),
    /// Tool calls from the assistant (stored for history fidelity).
    AssistantToolCalls {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        /// Raw reasoning content from thinking models, preserved for round-trip
        /// fidelity with model_provider APIs that require it.
        reasoning_content: Option<String>,
    },
    /// Results of tool executions, fed back to the LLM.
    ToolResults(Vec<ToolResultMessage>),
}

// ── 流式类型（chunk / event / 选项 / 错误） ───────────────────

/// A chunk of content from a streaming response.
/// 中文：流式响应中的一个内容块：`delta` 文本增量、`reasoning` 思考增量、
/// `is_final` 是否末块、`token_count` 每块 token 估算。
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Text delta for this chunk.
    pub delta: String,
    /// Reasoning/thinking delta (chain-of-thought from thinking models).
    pub reasoning: Option<String>,
    /// Whether this is the final chunk.
    pub is_final: bool,
    /// Approximate token count for this chunk (estimated).
    pub token_count: usize,
}

impl StreamChunk {
    /// Create a new non-final chunk.
    pub fn delta(text: impl Into<String>) -> Self {
        Self {
            delta: text.into(),
            reasoning: None,
            is_final: false,
            token_count: 0,
        }
    }

    /// Create a reasoning/thinking chunk.
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self {
            delta: String::new(),
            reasoning: Some(text.into()),
            is_final: false,
            token_count: 0,
        }
    }

    /// Create a final chunk.
    pub fn final_chunk() -> Self {
        Self {
            delta: String::new(),
            reasoning: None,
            is_final: true,
            token_count: 0,
        }
    }

    /// Create an error chunk.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            delta: message.into(),
            reasoning: None,
            is_final: true,
            token_count: 0,
        }
    }

    /// Estimate tokens (rough approximation: ~4 chars per token).
    pub fn with_token_estimate(mut self) -> Self {
        self.token_count = self.delta.len().div_ceil(4);
        self
    }
}

/// Structured events emitted by model_provider streaming APIs.
/// This extends plain text chunk streaming with explicit tool-call signals so
/// agent loops can preserve native tool semantics without parsing payload text.
///
/// 中文：流式 API 的结构化事件流，相比纯文本增量多了明确的工具调用信号，
/// 让代理循环无需解析正文即可保留原生工具语义：
/// - `TextDelta` / `ThinkingDelta`：文本增量与思考进度（思考进度仅用于展示，不持久化）；
/// - `ReasoningFinalized`：可回放的最终思考块（追加到 `reasoning_content` 回传）；
/// - `ToolCall` / `PreExecutedToolCall` / `PreExecutedToolResult`：工具调用信号
///   （`PreExecuted*` 为供应商已自行执行的，仅作观测、不再被调度器执行）；
/// - `Usage`（通常出现在 `Final` 之前）/ `Final`。
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text delta from the assistant.
    TextDelta(StreamChunk),
    /// Transient, human-readable thinking progress. Surfaced to the user
    /// (gated by the runtime visibility policy) and never persisted into
    /// reasoning_content.
    ThinkingDelta(String),
    /// Durable, replay-only finalized reasoning payload (signed thinking
    /// blocks in the provider's history-replay representation). Appended to
    /// `ChatResponse::reasoning_content` for the next provider request and
    /// never surfaced as user-visible progress.
    ReasoningFinalized(String),
    /// Structured tool call emitted during streaming.
    ToolCall(ToolCall),
    /// A tool call that was already executed by the model_provider (e.g. Claude Code proxy).
    /// Emitted for observability only — not re-executed by the agent's dispatcher.
    PreExecutedToolCall { name: String, args: String },
    /// The result of a pre-executed tool call.
    PreExecutedToolResult { name: String, output: String },
    /// Token usage reported by the provider, typically just before [`StreamEvent::Final`].
    /// Providers that do not surface usage in streaming responses simply omit this event.
    Usage(TokenUsage),
    /// Stream has completed.
    Final,
}

impl StreamEvent {
    pub fn from_chunk(chunk: StreamChunk) -> Self {
        if chunk.is_final {
            Self::Final
        } else {
            Self::TextDelta(chunk)
        }
    }
}

/// Options for streaming chat requests.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamOptions {
    /// Whether to enable streaming (default: true).
    pub enabled: bool,
    /// Whether to include token counts in chunks.
    pub count_tokens: bool,
}

impl StreamOptions {
    /// Create new streaming options with enabled flag.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            count_tokens: false,
        }
    }

    /// Enable token counting.
    pub fn with_token_count(mut self) -> Self {
        self.count_tokens = true;
        self
    }
}

/// Result type for streaming operations.
pub type StreamResult<T> = std::result::Result<T, StreamError>;

/// A provider safety refusal that completed at the transport layer but cannot
/// be accepted as an assistant response.
///
/// The optional usage belongs to the refusing attempt. It is carried on the
/// typed cause so reliability and turn accounting can bill that work without
/// treating it as accepted-response context usage. `category` is diagnostic
/// metadata only and must not be rendered to users.
///
/// 中文：供应商（Anthropic）因安全分类器拒答的类型化错误。
/// 每次「被拒尝试」自带的用量随类型保留，供计费 / 可靠性统计入账；
/// `attempted_candidate_index` 用于在复合供应商的故障转移域内定位
/// 已结算过的那条候选，供非流式恢复路径精确跳过。
#[derive(Debug, Clone, thiserror::Error)]
#[error("anthropic refusal: model declined this request (safety classifiers)")]
pub struct ModelRefusalError {
    /// Model requested on the refusing attempt.
    pub requested_model: String,
    /// Refusal category token, when the provider supplied one.
    pub category: Option<String>,
    /// Normalized usage billed by the refusing attempt.
    pub usage: Option<Box<TokenUsage>>,
    /// Exact reliability candidate that emitted a streamed refusal.
    ///
    /// Leaf providers leave this unset. Composite providers fill it while
    /// forwarding a stream so a non-streaming recovery can skip exactly the
    /// already-billed candidate.
    pub attempted_candidate: Option<String>,
    /// Position of that candidate in the active reliability domain.
    ///
    /// This disambiguates same-profile fallback models, which intentionally
    /// share one configured candidate/cooldown identity.
    pub attempted_candidate_index: Option<usize>,
}

/// Errors that can occur during streaming.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("JSON parse error: {0}")]
    Json(serde_json::Error),

    #[error("Invalid SSE format: {0}")]
    InvalidSse(String),

    #[error("ModelProvider error: {0}")]
    ModelProvider(String),

    #[error(transparent)]
    ModelRefusal(#[from] Box<ModelRefusalError>),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ── 能力声明 / 基线默认 / 定价模型 ───────────────────────────

/// Structured error returned when a requested capability is not supported.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "provider_capability_error model_provider={model_provider} capability={capability} message={message}"
)]
pub struct ProviderCapabilityError {
    pub model_provider: String,
    pub capability: String,
    pub message: String,
}

/// ModelProvider capabilities declaration.
/// Describes what features a model_provider supports, enabling intelligent
/// adaptation of tool calling modes and request formatting.
///
/// 中文：供应商能力声明（默认全为不支持/`false`）：原生工具调用、视觉输入、
/// 提示词缓存、原生扩展思考。调用方据此调整工具调用模式与请求格式。
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderCapabilities {
    /// Whether the model_provider supports native tool calling via API primitives.
    pub native_tool_calling: bool,
    /// Whether the model_provider supports vision / image inputs.
    pub vision: bool,
    /// Whether the model_provider supports prompt caching.
    pub prompt_caching: bool,
    /// Whether the provider supports native extended thinking.
    pub extended_thinking: bool,
}

/// ModelProvider-specific tool payload formats.
/// 中文：工具定义的供应商原生格式：Gemini `functionDeclarations`、
/// Anthropic `tools`（带 input_schema）、OpenAI `tools`（带 function）、
/// 以及 `PromptGuided`（把工具说明以文本注入 system 提示词的兜底方案）。
#[derive(Debug, Clone)]
pub enum ToolsPayload {
    /// Gemini API format (functionDeclarations).
    Gemini {
        function_declarations: Vec<serde_json::Value>,
    },
    /// Anthropic Messages API format (tools with input_schema).
    Anthropic { tools: Vec<serde_json::Value> },
    /// OpenAI Chat Completions API format (tools with function).
    OpenAI { tools: Vec<serde_json::Value> },
    /// Prompt-guided fallback (tools injected as text in system prompt).
    PromptGuided { instructions: String },
}

// ── 基线默认值（供应商不声明时使用） ──────────────────────────

/// Industry-neutral sampling temperature. OpenAI, Gemini, OpenRouter, and
/// most OpenAI-compatible endpoints document 0.7 as their typical default;
/// Anthropic and Ollama override (1.0 and 0.0 respectively).
///
/// 中文：行业中立的采样温度基线 0.7（OpenAI / Gemini / OpenRouter 等默认值；
/// Anthropic 用 1.0、Ollama 用 0.0，各自覆盖）。
pub const BASELINE_TEMPERATURE: f64 = 0.7;

/// Output-token budget roomy enough for typical agent turns. Providers
/// override per family where the model's own context window is the
/// binding constraint.
/// 中文：单轮输出 token 预算基线 4096；上下文窗口更紧的模型家族会各自覆盖。
pub const BASELINE_MAX_TOKENS: u32 = 4096;

/// HTTP timeout for cloud inference. Local model_providers (Ollama) override
/// upward since CPU/GPU-bound inference runs slower than round-tripping to
/// a hyperscaler.
/// 中文：云端推理 HTTP 超时基线 120s；本地供应商（如 Ollama）会调高，
/// 因为 CPU/GPU 推理比往返远端更慢。
pub const BASELINE_TIMEOUT_SECS: u64 = 120;

/// Wire protocol used when the model_provider doesn't declare one. Only OpenAI's
/// Codex stack uses the "responses" protocol; everything else speaks the
/// classic chat completions shape.
/// 中文：未声明时的线上协议基线 `chat_completions`；仅 OpenAI Codex 系用 `responses`。
pub const BASELINE_WIRE_API: &str = "chat_completions";

/// 中文：模型按 token 计费单价（USD/token，字符串保留小数精度）。
/// `prompt` 输入 / `completion` 输出 / `input_cache_read` 读取缓存 /
/// `input_cache_write` 写入缓存（后两者为 Kilo Gateway 专有）。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelPricing {
    /// Input/prompt tokens per-token rate (USD per token, e.g. `"0.000005"` = $5/1M tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Output/completion tokens per-token rate (USD per token, e.g. `"0.000020"` = $20/1M tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<String>,
    /// Cached input read rate — per-token charge for reading cached prompt data
    /// (USD per token, e.g. `"0.000001"` = $1/1M tokens). Kilo Gateway specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_cache_read: Option<String>,
    /// Cached input write rate — per-token charge for writing prompt data to cache
    /// (USD per token, e.g. `"0.000001"` = $1/1M tokens). Kilo Gateway specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_cache_write: Option<String>,
}

/// Model info with optional pricing — returned by `list_models_with_pricing`.
/// 中文：模型信息，可选携带定价；`context_window` 未知时必须保持 `None`
/// （禁止擅自填默认值，好让运维看到「未设置」而非假值）。
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<ModelPricing>,
    /// Maximum input window in tokens, as reported by the provider catalog.
    /// `None` when the catalog does not publish one — callers must treat that
    /// as "unknown" rather than substituting a default, so an operator can be
    /// told the window is unset instead of silently getting a stub value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
}

// ── ModelProvider trait（供应商总接口） ────────────────────────

/// 中文：供应商必须实现的总接口。方法按能力分组，绝大多数带默认实现，
/// 只有 `chat_with_system` 为必选：
/// - 能力声明：`capabilities` / `capabilities_for_model` / `supports_native_tools` /
///   `supports_vision` / `supports_streaming` 等。
/// - 聊天入口（自简到繁）：`simple_chat`（单轮）→ `chat_with_system`（带系统提示，必选）→
///   `chat_with_history`（多轮）→ `chat`（结构化；供应商标称不支持原生工具时
///   自动把工具说明以 `PromptGuided` 方式注入 system 消息）。
/// - 流式：`stream_chat` / `stream_chat_with_history` / `stream_chat_with_system`
///   （默认返回空流）。
/// - 默认值：`default_temperature` / `default_max_tokens` / `default_timeout_secs` /
///   `default_base_url` / `default_wire_api`（对应 `BASELINE_*`）。
/// - `has_stable_request_identity` 默认 `false`（承认「不稳定」），身份敏感功能
///   （如持久全量响应缓存）据此 fail-closed。
#[async_trait]
pub trait ModelProvider: Send + Sync + crate::attribution::Attributable {
    /// Whether repeated requests for `model` are dispatched to one stable
    /// provider/model identity.
    ///
    /// The default is deliberately unstable. Known leaf providers are marked
    /// stable at their construction choke point; composite and out-of-tree
    /// providers must opt in only when they can prove one concrete dispatch
    /// identity. Callers use this fact to fail closed for identity-sensitive
    /// behavior such as persistent full-response caching.
    fn has_stable_request_identity(&self, _model: &str) -> bool {
        false
    }

    /// Query model_provider capabilities.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    /// Query the effective capabilities for the model that will be dispatched.
    ///
    /// Most providers have one capability set for every model and inherit this
    /// default. Composite providers override it when the model selects a route
    /// or when failover can reach children with different capabilities.
    fn capabilities_for_model(&self, _model: &str) -> ProviderCapabilities {
        let mut capabilities = self.capabilities();
        // Preserve compatibility with providers that historically overrode the
        // convenience accessors instead of capabilities(). Composite overrides
        // should still make the model-aware value authoritative.
        capabilities.native_tool_calling = self.supports_native_tools();
        capabilities.vision = self.supports_vision();
        capabilities
    }

    /// Name the entry that forced `vision` to `false` on this provider's
    /// [`Self::capabilities_for_model`], for providers that aggregate several
    /// named entries (e.g. a primary plus configured fallbacks) into one
    /// capability set. Returns `None` when this provider is not such an
    /// aggregate, or when nothing about it limits vision for `model`.
    fn vision_limited_by(&self, _model: &str) -> Option<String> {
        None
    }

    /// Whether the selected request can reach both native-tool and text-only
    /// candidates.
    ///
    /// Ordinary providers are homogeneous and inherit `false`. Composite
    /// providers override this so callers that must select one tool protocol
    /// before dispatch can reject an incompatible strict configuration.
    fn has_mixed_native_tool_support_for_model(&self, _model: &str) -> bool {
        false
    }

    /// Family-preferred temperature default. Override per family. Documented
    /// for introspection only; never use to convert `None` into a wire value.
    fn default_temperature(&self) -> f64 {
        BASELINE_TEMPERATURE
    }

    /// Max output tokens used when the caller / config doesn't set one.
    fn default_max_tokens(&self) -> u32 {
        BASELINE_MAX_TOKENS
    }

    /// HTTP timeout (seconds) used when the caller / config doesn't set one.
    fn default_timeout_secs(&self) -> u64 {
        BASELINE_TIMEOUT_SECS
    }

    /// Canonical public API endpoint, when there is one. Returned as a
    /// string slice so model_provider impls can serve from `const &'static str`s
    /// without allocations. `None` = model_provider has no universal endpoint
    /// (local model_providers, auth-less CLIs, user-BYO endpoints).
    fn default_base_url(&self) -> Option<&str> {
        None
    }

    /// Wire protocol variant. Either `"responses"` (OpenAI Codex-style) or
    /// `"chat_completions"` (everything else). Providers override to their
    /// native format.
    fn default_wire_api(&self) -> &str {
        BASELINE_WIRE_API
    }

    /// Convert tool specifications to provider-native format.
    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        ToolsPayload::PromptGuided {
            instructions: build_tool_instructions_text(tools),
        }
    }

    /// Simple one-shot chat (single user message, no explicit system prompt).
    /// `temperature == None` means the field is omitted on the wire.
    async fn simple_chat(
        &self,
        message: &str,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        self.chat_with_system(None, message, model, temperature)
            .await
    }

    /// One-shot chat with optional system prompt. See `simple_chat` for
    /// the `temperature` contract.
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String>;

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        anyhow::bail!("live model listing is not supported for this model_provider")
    }

    /// Fetch the list of available models with pricing data for this
    /// model_provider. Default delegates to `list_models` and returns no
    /// pricing. Concrete providers that receive pricing from their `/models`
    /// endpoint override this to return enriched data.
    async fn list_models_with_pricing(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(self
            .list_models()
            .await?
            .into_iter()
            .map(|id| ModelInfo {
                id,
                pricing: None,
                context_window: None,
            })
            .collect())
    }

    /// Multi-turn conversation. See `simple_chat` for the `temperature`
    /// contract.
    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        let system = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.as_str());
        let last_user = messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        self.chat_with_system(system, last_user, model, temperature)
            .await
    }

    /// Structured chat API for agent loop callers. See `simple_chat` for
    /// the `temperature` contract.
    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ChatResponse> {
        if let Some(tools) = request.tools
            && !tools.is_empty()
            && !self.supports_native_tools()
        {
            let tool_instructions = match self.convert_tools(tools) {
                ToolsPayload::PromptGuided { instructions } => instructions,
                payload => {
                    anyhow::bail!(
                        "ModelProvider returned non-prompt-guided tools payload ({payload:?}) while supports_native_tools() is false"
                    )
                }
            };
            let mut modified_messages = request.messages.to_vec();

            if let Some(system_message) = modified_messages.iter_mut().find(|m| m.role == "system")
            {
                if !system_message.content.is_empty() {
                    system_message.content.push_str("\n\n");
                }
                system_message.content.push_str(&tool_instructions);
            } else {
                modified_messages.insert(0, ChatMessage::system(tool_instructions));
            }

            let text = self
                .chat_with_history(&modified_messages, model, temperature)
                .await?;
            let response = ChatResponse {
                text: Some(text),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            };
            return (!response.is_semantically_empty_terminal())
                .then_some(response)
                .ok_or_else(|| anyhow::Error::new(SemanticEmptyTerminalCompletion));
        }

        let text = self
            .chat_with_history(request.messages, model, temperature)
            .await?;
        let response = ChatResponse {
            text: Some(text),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        };
        (!response.is_semantically_empty_terminal())
            .then_some(response)
            .ok_or_else(|| anyhow::Error::new(SemanticEmptyTerminalCompletion))
    }

    /// Whether model_provider supports native tool calls over API.
    fn supports_native_tools(&self) -> bool {
        self.capabilities().native_tool_calling
    }

    /// Whether model_provider supports multimodal vision input.
    fn supports_vision(&self) -> bool {
        self.capabilities().vision
    }

    /// Warm up the HTTP connection pool.
    async fn warmup(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Chat with tool definitions for native function calling support.
    /// See `simple_chat` for the `temperature` contract.
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: &[serde_json::Value],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ChatResponse> {
        let text = self.chat_with_history(messages, model, temperature).await?;
        let response = ChatResponse {
            text: Some(text),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        };
        (!response.is_semantically_empty_terminal())
            .then_some(response)
            .ok_or_else(|| anyhow::Error::new(SemanticEmptyTerminalCompletion))
    }

    /// Whether model_provider supports streaming responses.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Whether model_provider can emit structured tool-call stream events.
    fn supports_streaming_tool_events(&self) -> bool {
        false
    }

    /// Streaming chat with optional system prompt. See `simple_chat` for
    /// the `temperature` contract.
    fn stream_chat_with_system(
        &self,
        _system_prompt: Option<&str>,
        _message: &str,
        _model: &str,
        _temperature: Option<f64>,
        _options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
        stream::empty().boxed()
    }

    /// Streaming chat with history. See `simple_chat` for the `temperature`
    /// contract.
    fn stream_chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
        let system = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.as_str());
        let last_user = messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        self.stream_chat_with_system(system, last_user, model, temperature, options)
    }

    /// Structured streaming chat interface. See `simple_chat` for the
    /// `temperature` contract.
    fn stream_chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        self.stream_chat_with_history(request.messages, model, temperature, options)
            .map(|chunk_result| chunk_result.map(StreamEvent::from_chunk))
            .boxed()
    }
}

/// Blanket implementation: `Arc<T>` delegates all `ModelProvider` methods to `T`.
/// This eliminates the need for manual `impl ModelProvider for Arc<MyModelProvider>`
/// boilerplate in test and production code.
///
/// 中文：对所有 `Arc<T>` 的统一定向委托实现——`Arc<MyModelProvider>` 直接当作
/// `ModelProvider` 使用，免去测试与生产中手写样板转发方法。
#[async_trait]
impl<T: ModelProvider + ?Sized> ModelProvider for Arc<T> {
    fn has_stable_request_identity(&self, model: &str) -> bool {
        self.as_ref().has_stable_request_identity(model)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.as_ref().capabilities()
    }

    fn capabilities_for_model(&self, model: &str) -> ProviderCapabilities {
        self.as_ref().capabilities_for_model(model)
    }

    fn vision_limited_by(&self, model: &str) -> Option<String> {
        self.as_ref().vision_limited_by(model)
    }

    fn has_mixed_native_tool_support_for_model(&self, model: &str) -> bool {
        self.as_ref().has_mixed_native_tool_support_for_model(model)
    }

    fn default_max_tokens(&self) -> u32 {
        self.as_ref().default_max_tokens()
    }

    fn default_temperature(&self) -> f64 {
        self.as_ref().default_temperature()
    }

    fn default_timeout_secs(&self) -> u64 {
        self.as_ref().default_timeout_secs()
    }

    fn default_base_url(&self) -> Option<&str> {
        self.as_ref().default_base_url()
    }

    fn default_wire_api(&self) -> &str {
        self.as_ref().default_wire_api()
    }

    fn convert_tools(&self, tools: &[ToolSpec]) -> ToolsPayload {
        self.as_ref().convert_tools(tools)
    }

    fn supports_native_tools(&self) -> bool {
        self.as_ref().supports_native_tools()
    }

    fn supports_vision(&self) -> bool {
        self.as_ref().supports_vision()
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        self.as_ref()
            .chat_with_system(system_prompt, message, model, temperature)
            .await
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        self.as_ref()
            .chat_with_history(messages, model, temperature)
            .await
    }

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ChatResponse> {
        self.as_ref().chat(request, model, temperature).await
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        self.as_ref().warmup().await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ChatResponse> {
        self.as_ref()
            .chat_with_tools(messages, tools, model, temperature)
            .await
    }

    fn supports_streaming(&self) -> bool {
        self.as_ref().supports_streaming()
    }

    fn supports_streaming_tool_events(&self) -> bool {
        self.as_ref().supports_streaming_tool_events()
    }

    fn stream_chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
        self.as_ref()
            .stream_chat_with_system(system_prompt, message, model, temperature, options)
    }

    fn stream_chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
        self.as_ref()
            .stream_chat_with_history(messages, model, temperature, options)
    }

    fn stream_chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        self.as_ref()
            .stream_chat(request, model, temperature, options)
    }
}

/// Build tool instructions text for prompt-guided tool calling.
/// 中文：为「提示词引导式」工具调用生成说明文本：教模型用
/// `<tool_call>{"name":..., "arguments":{...}}</tool_call>` 包裹工具调用，
/// 并把每个工具的 JSON Schema 参数拼进 system 提示词。
pub fn build_tool_instructions_text(tools: &[ToolSpec]) -> String {
    let mut instructions = String::new();

    instructions.push_str("## Tool Use Protocol\n\n");
    instructions.push_str("To use a tool, wrap a JSON object in <tool_call></tool_call> tags:\n\n");
    instructions.push_str("<tool_call>\n");
    instructions.push_str(r#"{"name": "tool_name", "arguments": {"param": "value"}}"#);
    instructions.push_str("\n</tool_call>\n\n");
    instructions.push_str("You may use multiple tool calls in a single response. ");
    instructions.push_str("After tool execution, results appear in <tool_result> tags. ");
    instructions
        .push_str("Continue reasoning with the results until you can give a final answer.\n\n");
    instructions.push_str("### Available Tools\n\n");

    for tool in tools {
        writeln!(&mut instructions, "**{}**: {}", tool.name, tool.description)
            .expect("writing to String cannot fail");

        let parameters =
            serde_json::to_string(&tool.parameters).unwrap_or_else(|_| "{}".to_string());
        writeln!(&mut instructions, "Parameters: `{parameters}`")
            .expect("writing to String cannot fail");
        instructions.push('\n');
    }

    instructions
}

// ── 单元测试 ─────────────────────────────────────────────────

#[cfg(test)]
mod capability_tests {
    //! 供应商能力声明的兼容性测试：验证模型感知能力查询 `capabilities_for_model`
    //! 会保留历史遗留 accessor（`supports_native_tools` 等）的覆盖行为。

    use super::ModelProvider;
    use crate::attribution::{Attributable, ModelProviderKind, ProviderKind, Role};
    use async_trait::async_trait;

    /// 中文：测试夹具——只覆盖 `supports_native_tools()`（遗留 accessor）与必选方法
    /// `chat_with_system`，`capabilities()` 走默认（全 false）。
    struct NativeAccessorOnlyProvider;

    impl Attributable for NativeAccessorOnlyProvider {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }

        fn alias(&self) -> &str {
            "native_accessor_only"
        }
    }

    #[async_trait]
    impl ModelProvider for NativeAccessorOnlyProvider {
        fn supports_native_tools(&self) -> bool {
            true
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    /// 中文：验证 `capabilities_for_model` 会保留遗留 accessor 的覆盖——
    /// 即使 `capabilities()` 默认返回 `native_tool_calling=false`，
    /// 只要厂商覆写了 `supports_native_tools() -> true`，模型级查询仍应给出 `true`。
    #[test]
    fn model_capabilities_preserve_native_accessor_overrides() {
        let provider = NativeAccessorOnlyProvider;

        assert!(
            !provider.capabilities().native_tool_calling,
            "the fixture must exercise the legacy accessor-only override"
        );
        assert!(
            provider
                .capabilities_for_model("requested-model")
                .native_tool_calling,
            "model-aware capability lookup must preserve legacy supports_native_tools overrides"
        );
    }
}

#[cfg(test)]
mod turn_order_tests {
    //! 会话历史「回合次序」与「语义空终态」判定测试：
    //! 覆盖 `ChatMessage::sanitize_leading_turn_order` 与
    //! `ChatResponse::is_semantically_empty_terminal` 的边界行为。

    use super::{ChatMessage, ChatResponse, ToolCall};

    /// 中文：只有思考内容、没有可见文字的响应（text 为空白 + reasoning_content 有值）
    /// 应判定为「语义空终态」——思考内容不参与判定。
    #[test]
    fn semantic_empty_terminal_ignores_reasoning_content() {
        let response = ChatResponse {
            text: Some("  \n".to_string()),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: Some("internal reasoning".to_string()),
        };

        assert!(response.is_semantically_empty_terminal());
    }

    /// 中文：剥除内联 ` thinking...response` 思考片段后没有剩余可见文字，
    /// 判定为空终态（思考标签内的内容不算最终输出）。
    #[test]
    fn semantic_empty_terminal_uses_display_text_after_think_tag_stripping() {
        let response = ChatResponse {
            text: Some("<think>internal reasoning</think>".to_string()),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        };

        assert!(response.is_semantically_empty_terminal());
    }

    /// 中文：仅含工具调用（text 为空）的响应不算空终态——工具调用本身就代表有效进展。
    #[test]
    fn semantic_empty_terminal_keeps_tool_only_response_valid() {
        let response = ChatResponse {
            text: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: None,
            reasoning_content: None,
        };

        assert!(!response.is_semantically_empty_terminal());
    }

    /// 中文：含正常最终文字的响应不算空终态。
    #[test]
    fn text_response_is_not_semantically_empty() {
        let response = ChatResponse {
            text: Some("done".to_string()),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        };

        assert!(!response.is_semantically_empty_terminal());
    }

    /// 中文：仅含工具调用同样不算空终态（与上一处语义一致，作为对照用例）。
    #[test]
    fn tool_only_response_is_not_semantically_empty() {
        let response = ChatResponse {
            text: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: None,
            reasoning_content: None,
        };

        assert!(!response.is_semantically_empty_terminal());
    }

    /// 中文：删掉开头孤立出现的 assistant 工具调用与 tool 结果，
    /// 保留到第一个 `user` 之前为止，把残片历史修整为合法回合序。
    #[test]
    fn drops_leading_assistant_tool_call_before_first_user() {
        let mut msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::assistant("[tool_call] fire"),
            ChatMessage::tool("result"),
            ChatMessage::user("actual user"),
        ];
        ChatMessage::sanitize_leading_turn_order(&mut msgs);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "actual user");
    }

    /// 中文：开头孤立的 tool 结果（没有前置 assistant 调用）同样被丢弃。
    #[test]
    fn drops_leading_orphan_tool_turn() {
        let mut msgs = vec![ChatMessage::tool("orphan result"), ChatMessage::user("hi")];
        ChatMessage::sanitize_leading_turn_order(&mut msgs);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    /// 中文：已以 `user` 打头的合法历史保持不变（修整是幂等的）。
    #[test]
    fn preserves_already_valid_history() {
        let mut msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("q"),
            ChatMessage::assistant("[tool_call] x"),
            ChatMessage::tool("r"),
            ChatMessage::assistant("done"),
        ];
        let before = msgs.clone();
        ChatMessage::sanitize_leading_turn_order(&mut msgs);
        assert_eq!(msgs.len(), before.len());
        assert_eq!(msgs[1].role, "user");
    }

    /// 中文：没有任何 `user` 回合时，所有非 system 消息都被丢弃，只保留 system。
    #[test]
    fn no_user_turn_drops_all_non_system() {
        let mut msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::assistant("[tool_call] x"),
            ChatMessage::tool("r"),
        ];
        ChatMessage::sanitize_leading_turn_order(&mut msgs);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "system");
    }

    /// 中文：空历史不做任何修改（修整是 no-op）。
    #[test]
    fn empty_history_is_noop() {
        let mut msgs: Vec<ChatMessage> = vec![];
        ChatMessage::sanitize_leading_turn_order(&mut msgs);
        assert!(msgs.is_empty());
    }
}

#[cfg(test)]
mod thinking_display_tests {
    //! `ThinkingDisplay` / `NativeThinkingParams` 的取值映射与序列化测试：
    //! 覆盖 `as_str` 映射以及 `display` 字段序列化时「有则带、无则省略」的约定。

    use super::{NativeThinkingParams, ThinkingDisplay};

    /// 中文：`ThinkingDisplay::Updates` 应映射为线上取值字符串 `"updates"`。
    #[test]
    fn as_str_maps_updates_variant() {
        assert_eq!(ThinkingDisplay::Updates.as_str(), "updates");
    }

    /// 中文：`display=Some(...)` 时，序列化结果必须包含 `"display":"updates"` 字段。
    #[test]
    fn serialization_includes_display_when_present() {
        let params = NativeThinkingParams {
            budget_tokens: 1_024,
            display: Some(ThinkingDisplay::Updates),
        };
        let json = serde_json::to_string(&params).expect("serialization should succeed");
        assert!(
            json.contains("\"display\":\"updates\""),
            "expected display field in serialized params, got: {json}"
        );
    }

    /// 中文：`display=None` 时，序列化结果不得出现 `display` 字段，
    ///   保持扩展思考 beta 之前的请求形状。
    #[test]
    fn serialization_omits_display_when_absent() {
        let params = NativeThinkingParams {
            budget_tokens: 1_024,
            display: None,
        };
        let json = serde_json::to_string(&params).expect("serialization should succeed");
        assert!(
            !json.contains("display"),
            "expected display field to be omitted, got: {json}"
        );
    }
}
