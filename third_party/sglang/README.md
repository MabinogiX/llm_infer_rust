# SGLang device kernels

Source: SGLang commit `15b256bdb` on the user reference checkout.
License: Apache-2.0; see LICENSE.

`include/sgl_kernel/{type,vec,tile,warp,math,impl/norm}.cuh` are verbatim copies
from `python/sglang/kernels/jit/include/`. `utils.cuh` retains upstream scalar
aliases and device utilities, with DLPack constants and TVM host launchers removed.
`qknorm.cuh` and `kvcache.cuh` retain upstream parameter structs/device kernels
from `python/sglang/kernels/jit/csrc/elementwise/`; TensorView host wrappers and
their includes are removed. Device algorithms are unchanged. Our ATen adapters
provide validation, CUDA stream and launch configuration; no TVM runtime needed.
PDL is disabled on the A100. CUDA compilation requires C++20 for upstream helpers.
