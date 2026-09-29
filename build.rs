use std::{env, process::Command};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda_graph)");
    println!("cargo:rerun-if-changed=src/engine/cuda_graph_bridge.cpp");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let python = env::var("VIRTUAL_ENV")
        .map(|venv| format!("{venv}/bin/python"))
        .unwrap_or_else(|_| "python".to_owned());
    let output = Command::new(python)
        .args([
            "-c",
            "import sys, torch; from pathlib import Path; from torch.utils.cpp_extension import CUDA_HOME, include_paths; sys.exit(2) if not torch.version.cuda or CUDA_HOME is None else None; print(torch.version.cuda); print(int(torch._C._GLIBCXX_USE_CXX11_ABI)); print(Path(torch.__file__).parent / 'lib'); print(*include_paths(device_type='cuda'), sep='\\n')",
        ])
        .output();
    let Ok(output) = output else { return };
    if !output.status.success() {
        return;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    if lines.next().unwrap_or_default().is_empty() {
        return;
    }
    let abi = lines.next().unwrap_or("0");
    let lib_dir = lines.next().unwrap_or_default();
    let mut build = cc::Build::new();
    build.cpp(true).file("src/engine/cuda_graph_bridge.cpp");
    build.flag_if_supported("-std=c++20");
    build.define("_GLIBCXX_USE_CXX11_ABI", abi);
    for include in lines {
        build.include(include);
    }
    build.compile("sglang_cuda_graph_bridge");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
    println!("cargo:rustc-cfg=has_cuda_graph");
}
