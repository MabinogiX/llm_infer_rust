use std::{env, fs, path::Path, process::Command};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda_graph)");
    println!("cargo:rustc-check-cfg=cfg(has_flashinfer)");
    println!("cargo:rustc-check-cfg=cfg(has_qwen3_cuda)");
    println!("cargo:rustc-check-cfg=cfg(has_cuda_kv_store)");
    println!("cargo:rerun-if-changed=src/models/attention/kv_store_bridge.cpp");
    println!("cargo:rerun-if-changed=src/models/attention/kv_store.cu");
    println!("cargo:rerun-if-changed=src/engine/cuda_graph_bridge.cpp");
    println!("cargo:rerun-if-changed=src/engine/sampling_bridge.cpp");
    println!("cargo:rerun-if-changed=src/engine/sampling_flashinfer.cu");
    println!("cargo:rerun-if-changed=src/models/attention/flashinfer_bridge.cpp");
    println!("cargo:rerun-if-changed=src/models/attention/flashinfer_plan_run.cu");
    println!("cargo:rerun-if-changed=src/models/qwen3/fused_ops_bridge.cpp");
    println!("cargo:rerun-if-changed=src/models/qwen3/fused_ops.cu");
    println!("cargo:rerun-if-changed=third_party/sglang");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=FLASHINFER_CUDA_ARCH");
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
    let torch_includes: Vec<String> = lines.map(str::to_owned).collect();
    for include in &torch_includes {
        build.include(include);
    }
    build.compile("sglang_cuda_graph_bridge");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
    println!("cargo:rustc-cfg=has_cuda_graph");

    let arch = env::var("FLASHINFER_CUDA_ARCH").unwrap_or_else(|_| "80".to_owned());
    let flashinfer_include = env::var("VIRTUAL_ENV").ok().and_then(|venv| {
        fs::read_dir(Path::new(&venv).join("lib"))
            .ok()
            .and_then(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path().join("site-packages/flashinfer/data/include"))
                    .find(|path| path.join("flashinfer/attention/decode.cuh").exists())
            })
    });
    let mut fused_bridge = cc::Build::new();
    fused_bridge
        .cpp(true)
        .file("src/models/qwen3/fused_ops_bridge.cpp")
        .file("src/models/attention/kv_store_bridge.cpp")
        .flag_if_supported("-std=c++20")
        .warnings(false)
        .define("_GLIBCXX_USE_CXX11_ABI", abi);
    for include in &torch_includes {
        fused_bridge.include(include);
    }
    fused_bridge.compile("sglang_qwen3_fused_bridge");
    let mut fused_kernels = cc::Build::new();
    fused_kernels
        .cuda(true)
        .debug(false)
        .opt_level(3)
        .file("src/models/qwen3/fused_ops.cu")
        .file("src/models/attention/kv_store.cu")
        .include("third_party/sglang")
        .include("third_party/sglang/include")
        .flag(&format!("-DSGL_CUDA_ARCH={arch}0"))
        .flag("-std=c++20")
        .flag("--expt-relaxed-constexpr")
        .flag("-UNDEBUG")
        .flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .warnings(false);
    if let Some(include) = &flashinfer_include {
        let data = include.parent().unwrap();
        println!("cargo:rerun-if-changed={}", include.display());
        fused_kernels
            .define("SGLANG_USE_FLASHINFER_NORM", None)
            .include(include)
            .include(data.join("cccl/libcudacxx/include"))
            .include(data.join("cccl/cub"))
            .include(data.join("cccl/thrust"));
    }
    fused_kernels.compile("sglang_qwen3_fused_kernels");
    println!("cargo:rustc-cfg=has_qwen3_cuda");
    println!("cargo:rustc-cfg=has_cuda_kv_store");

    // Compile the thin native adapter against the wheel's official CUDA headers.
    let Some(flashinfer_include) = flashinfer_include else {
        return;
    };
    // Wheel upgrades must rebuild the adapter even when Rust sources are unchanged.
    println!("cargo:rerun-if-changed={}", flashinfer_include.display());
    let flashinfer_data = flashinfer_include.parent().unwrap();
    let mut flashinfer_build = cc::Build::new();
    flashinfer_build
        .cpp(true)
        .file("src/models/attention/flashinfer_bridge.cpp")
        .file("src/engine/sampling_bridge.cpp")
        .flag_if_supported("-std=c++20")
        .warnings(false)
        .define("_GLIBCXX_USE_CXX11_ABI", abi);
    for include in &torch_includes {
        flashinfer_build.include(include);
    }
    flashinfer_build.compile("sglang_flashinfer_bridge");

    let mut kernel = cc::Build::new();
    kernel
        .cuda(true)
        .debug(false)
        .opt_level(3)
        .file("src/models/attention/flashinfer_plan_run.cu")
        .include(&flashinfer_include)
        .include(flashinfer_data.join("cutlass/include"))
        .include(flashinfer_data.join("cccl/libcudacxx/include"))
        .include(flashinfer_data.join("cccl/cub"))
        .include(flashinfer_data.join("cccl/thrust"))
        .flag("-std=c++17")
        .flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .warnings(false);
    kernel.compile("sglang_flashinfer_plan_run");
    // FlashInfer sampling launches 1024 threads on Ampere. Bound registers
    // per thread so vectorized large-vocabulary kernels fit the SM register file.
    let mut sampling = cc::Build::new();
    sampling
        .cuda(true)
        .debug(false)
        .opt_level(3)
        .file("src/engine/sampling_flashinfer.cu")
        .include(&flashinfer_include)
        .include(flashinfer_data.join("cccl/libcudacxx/include"))
        .include(flashinfer_data.join("cccl/cub"))
        .include(flashinfer_data.join("cccl/thrust"))
        .flag("-std=c++17")
        .flag("-maxrregcount=64")
        .flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .warnings(false);
    sampling.compile("sglang_flashinfer_sampling");
    println!("cargo:rustc-cfg=has_flashinfer");
}
