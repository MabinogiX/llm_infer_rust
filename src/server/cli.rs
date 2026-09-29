//! Minimal model-server CLI, with the supported mini-sglang options.

use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
};

use crate::engine::ServerArgs;
use crate::logging::LoggingConfig;

use super::ServeArgs;

pub const USAGE: &str = "Usage: sglang-rust --model-path PATH [options]\n\
Options:\n\
  --host IP                 Bind address (default 127.0.0.1)\n\
  --port PORT               Bind port (default 8000)\n\
  --tp-size N               Tensor parallel size (only 1 is supported)\n\
  --memory-ratio R          KV cache memory ratio (default 0.9)\n\
  --max-running-req N       Maximum active requests (default 256)\n\
  --max-seq-len N           Maximum sequence length (default 8192)\n\
  --page-size N             KV cache page size (default 16)\n\
  --attention-backend NAME  Attention backend (default pt)\n\
  --dtype NAME              Model dtype (currently float32)\n\
  --device NAME             Device: auto, cpu, or cuda (default auto)\n\
  --trust-remote-code       Request Hugging Face remote code\n\
  --log-dir PATH            Log directory (default logs)\n\
  --log-level LEVEL         trace/debug/info/warn/error (default info)\n\
  -h, --help                Show this help";

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<ServeArgs, String> {
    let mut args = args.into_iter();
    let mut model_path: Option<PathBuf> = None;
    let mut host: IpAddr = "127.0.0.1".parse().expect("literal IP");
    let mut port = 8000;
    let mut engine = ServerArgs::new("");
    let mut logging = LoggingConfig::default();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} 缺少参数"));
        match flag.as_str() {
            "--model-path" => model_path = Some(PathBuf::from(value()?)),
            "--host" => {
                host = value()?
                    .parse()
                    .map_err(|_| "--host 必须是 IP 地址".to_owned())?
            }
            "--port" => port = parse_value(&value()?, &flag)?,
            "--tp-size" => engine.tp_size = parse_value(&value()?, &flag)?,
            "--memory-ratio" => engine.memory_ratio = parse_value(&value()?, &flag)?,
            "--max-running-req" => engine.max_running_req = parse_value(&value()?, &flag)?,
            "--max-seq-len" => engine.max_seq_len = parse_value(&value()?, &flag)?,
            "--page-size" => engine.page_size = parse_value(&value()?, &flag)?,
            "--attention-backend" => engine.attention_backend = value()?,
            "--dtype" => {
                let dtype = value()?;
                if dtype != "auto" && dtype != "float32" {
                    return Err("当前仅支持 float32；--dtype 必须是 auto 或 float32".to_owned());
                }
                engine.dtype = dtype;
            }
            "--device" => {
                let device = value()?;
                if device != "auto" && device != "cpu" && device != "cuda" {
                    return Err("--device 必须是 auto、cpu 或 cuda".to_owned());
                }
                engine.device = device;
            }
            "--trust-remote-code" => engine.trust_remote_code = true,
            "--log-dir" => logging.directory = PathBuf::from(value()?),
            "--log-level" => {
                let level = value()?.to_ascii_lowercase();
                logging.level = Some(match level.as_str() {
                    "trace" | "debug" | "info" | "warn" | "error" => level,
                    "warning" => "warn".to_owned(),
                    _ => return Err(format!("无效的 --log-level: {level}")),
                });
            }
            _ => return Err(format!("未知参数: {flag}")),
        }
    }
    engine.model_path = model_path.ok_or("必须指定 --model-path".to_owned())?;
    if engine.tp_size != 1 {
        return Err("当前不支持 tp_size > 1".to_owned());
    }
    Ok(ServeArgs {
        engine,
        bind: SocketAddr::new(host, port),
        logging,
    })
}

fn parse_value<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, String> {
    text.parse().map_err(|_| format!("{flag} 参数无效: {text}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_server_address_and_device() {
        let args = parse_args(["--model-path", "/tmp/model", "--port", "9001"].map(str::to_owned))
            .unwrap();
        assert_eq!(args.bind.port(), 9001);
        assert_eq!(args.engine.model_path, PathBuf::from("/tmp/model"));
        let logging_args = parse_args(
            [
                "--model-path",
                "/tmp/model",
                "--log-dir",
                "/tmp/minisgl-logs",
                "--log-level",
                "DEBUG",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            logging_args.logging.directory,
            PathBuf::from("/tmp/minisgl-logs")
        );
        assert_eq!(logging_args.logging.level.as_deref(), Some("debug"));
        assert_eq!(args.engine.device, "auto");
        for device in ["cpu", "cuda"] {
            let parsed =
                parse_args(["--model-path", "/tmp/model", "--device", device].map(str::to_owned))
                    .unwrap();
            assert_eq!(parsed.engine.device, device);
        }
        assert!(
            parse_args(["--model-path", "/tmp/model", "--device", "mps"].map(str::to_owned))
                .is_err()
        );
    }
}
