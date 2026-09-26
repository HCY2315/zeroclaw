//! 飞书真实联调示例：监听读取消息，并对每条消息回一条 echo 回复。
//!
//! 用法（先配置好 `feishu.toml`）：
//!
//! ```bash
//! cargo run -p zeroclaw-channels --features channel-lark --example feishu_echo -- feishu.toml
//! ```
//!
//! `feishu.toml` 示例：
//!
//! ```toml
//! # 允许机器人响应的用户 open_id 列表；含 "*" 表示任何人。
//! # 可在飞书开放平台 -> 应用详情页拿到 bot 的 open_id，或用 get_tenant_access_token 后
//! # 调用 /contact/v3/users/{user_id} 查询。
//! peers = ["*"]
//!
//! [lark]
//! enabled = true
//! app_id = "cli_xxxxxxxxxxxxxxxx"
//! app_secret = "你的 App Secret"
//! # 国内飞书用 true，国际版 Lark 用 false
//! use_feishu = true
//! # 接收模式：
//! #   "websocket"（默认）——长连接主动拉消息，无需公网回调地址；
//! #   "webhook"    ——需要公网可访问的地址指向本机，并在开放平台配置回调 URL。
//! receive_mode = "websocket"
//! # webhook 模式必填（开放平台 -> 事件订阅里的 Verification Token）
//! verification_token = "可选，webhook 模式必填"
//! # 群聊里是否仅响应 @机器人的消息（单聊不受影响）
//! mention_only = false
//! # 收到消息后是否先加一个 ❤ 表情回应（需机器人具备相应权限）
//! ack_reactions = false
//! ```
//!
//! 前置条件：
//! 1. 在飞书开放平台创建「企业自建应用」，开通 im 消息权限并发布版本；
//! 2. 用户需要先给机器人发过一条消息（或在事件订阅中订阅 `im.message.receive_v1`
//!    for websocket mode），机器人才能向该会话回发；
//! 3. 把上面的 `app_id` / `app_secret` 换成你应用的真实值。

use std::sync::Arc;

use zeroclaw_api::channel::{Channel, ChannelMessage, SendMessage};
use zeroclaw_channels::lark::LarkChannel;

#[derive(serde::Deserialize)]
struct DemoConfig {
    /// 允许机器人响应的用户 open_id 列表；空则等价于允许任何人（"*"）。
    #[serde(default)]
    peers: Vec<String>,
    lark: zeroclaw_config::schema::LarkConfig,
}

impl DemoConfig {
    fn peer_list(&self) -> Vec<String> {
        if self.peers.is_empty() {
            vec!["*".to_string()]
        } else {
            self.peers.clone()
        }
    }
}

/// 解析配置文件路径：
/// 1. 命令行显式传入的路径优先；
/// 2. 没传或传的 `feishu.toml` 在当前目录不存在时，
///    回退到仓库根目录下的 `crates/zeroclaw-channels/examples/feishu.toml`，
///    方便直接运行 `target/debug/examples/feishu_echo`。
fn resolve_config_path(requested: &str) -> std::path::PathBuf {
    let requested = std::path::PathBuf::from(requested);
    if requested.exists() {
        return requested;
    }
    for candidate in [
        "feishu.toml",
        "crates/zeroclaw-channels/examples/feishu.toml",
    ] {
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    requested
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 安装 rustls ring provider（与运行时 enroll 的约定一致），否则 WS 建连会 panic。
    let _ = rustls::crypto::ring::default_provider().install_default();

    // 打开 zeroclaw_log 的 stderr 输出：示例默认静默，WS 连接/注册/token/权限等
    // 记录看不到，先把它们打出来便于联调定位。注意 record! 事件的 target 是
    // `zeroclaw_log_event`（不是调用模块），所以必须过滤它才能看到通道日志。
    ::zeroclaw_log::install_global_subscriber(
        None,
        "zeroclaw_log_event=debug,zeroclaw_channels=debug",
        true,
    );

    let requested = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "feishu.toml".to_string());
    let config_path = resolve_config_path(&requested);
    let raw = std::fs::read_to_string(&config_path).map_err(|e| {
        anyhow::Error::msg(format!("读取配置文件 {} 失败: {e}", config_path.display()))
    })?;
    let demo: DemoConfig = toml::from_str(&raw)
        .map_err(|e| anyhow::Error::msg(format!("解析 {} 失败: {e}", config_path.display())))?;

    let peer_list = demo.peer_list();
    let peer_resolver: Arc<dyn Fn() -> Vec<String> + Send + Sync> =
        Arc::new(move || peer_list.clone());
    let ch = Arc::new(LarkChannel::from_config(
        &demo.lark,
        "feishu_echo",
        peer_resolver,
    ));

    println!(
        "[Feishu echo] platform={} receive_mode={:?} alias=feishu_echo",
        if demo.lark.use_feishu {
            "feishu"
        } else {
            "lark"
        },
        demo.lark.receive_mode,
    );
    println!("[Feishu echo] listening (Ctrl-C to stop)...");

    let (tx, mut rx) = tokio::sync::mpsc::channel(6400);
    let listen_ch = Arc::clone(&ch);
    zeroclaw_spawn::spawn!(async move {
        if let Err(e) = listen_ch.listen(tx).await {
            eprintln!("[Feishu echo] listen ended with error: {e}");
        }
    });

    // 下拉（读）消息：长连接/ webhook 会把每条 ChannelMessage 推进这条管道。
    while let Some(msg) = rx.recv().await {
        reply(&ch, &msg).await;
    }

    Ok(())
}

async fn reply(ch: &LarkChannel, msg: &ChannelMessage) {
    println!(
        "[收到] sender={} chat={} text={} thread_ts={:?}",
        msg.sender, msg.reply_target, msg.content, msg.thread_ts
    );

    let reply_text = format!("echo: {}", msg.content);
    // 用 reply_to 而非 SendMessage::new：这样会把入站消息的 thread_ts
    // （话题根消息 ID）复制给出站消息，send 端据它在 body 里补 root_id，
    // 回复才能落回原话题。
    let send = SendMessage::reply_to(msg, reply_text.clone());
    match Channel::send(ch, &send).await {
        Ok(()) => println!(
            "[发送] to={} text={} thread_ts={:?}",
            msg.reply_target, reply_text, msg.thread_ts
        ),
        Err(e) => eprintln!("[发送失败] to={} error={e}", msg.reply_target),
    }
}
