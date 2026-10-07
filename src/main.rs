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
    let engine = &args.engine;
    tracing::info!(
        model_path = %engine.model_path.display(),
        bind = %args.bind,
        tp_size = engine.tp_size,
        attention_backend = %engine.attention_backend,
        max_seq_len = engine.max_seq_len,
        max_running_req = engine.max_running_req,
        page_size = engine.page_size,
        memory_ratio = engine.memory_ratio,
        cuda_graph_bs_limit = engine.cuda_graph_bs.unwrap_or(engine.max_running_req),
        prefill_cuda_graph_max_tokens = engine.prefill_cuda_graph_max_tokens,
        requested_dtype = %engine.dtype,
        requested_device = %engine.device,
        log_dir = %args.logging.directory.display(),
        "starting model server"
    );
    match serve(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = %error, "model server stopped with an error");
            ExitCode::FAILURE
        }
    }
}
