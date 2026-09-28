use std::process::ExitCode;

use sglang_rust::{
    logging::init_logging,
    server::{USAGE, parse_args, serve},
};

#[tokio::main]
async fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let _log_guard = match init_logging(&args.logging) {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("日志初始化失败: {error}");
            return ExitCode::FAILURE;
        }
    };
    tracing::info!(model_path = %args.engine.model_path.display(), log_dir = %args.logging.directory.display(), "starting model server");
    match serve(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = %error, "model server stopped with an error");
            ExitCode::FAILURE
        }
    }
}
