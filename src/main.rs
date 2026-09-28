use sglang_rust::server::{USAGE, parse_args, serve};

#[tokio::main]
async fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return;
    }
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(error) = serve(args).await {
        eprintln!("Server failed: {error}");
        std::process::exit(1);
    }
}
