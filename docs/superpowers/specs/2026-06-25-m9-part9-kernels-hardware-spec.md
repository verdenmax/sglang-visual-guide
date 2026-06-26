# M9 — Part 9 Kernels & Hardware (L38–L42) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M9. Builds on M0–M8 (L01–37 exist).

## Goal
Go below the Python layers into the **C++/CUDA kernel layer** and **multi-hardware** support — the
deep part the guide promised. How `sgl-kernel` (AOT C++/CUDA) is built and bound to Python, the
lightweight JIT-kernel path, what a real attention kernel does (memory layout / tiling / paged-KV
access), how operator fusion + CUDA graphs cooperate, and how the platform abstraction spreads SGLang
across NVIDIA/AMD/TPU/NPU/CPU. Module: `src/part9.py`, lessons L38–42.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with EXACTLY 4 visual blocks/lang (identical zh/en; no SVG;
NOT `timeline`) → one cited `.codefile` (`file ::symbol`; docstrings in `<pre>` as `#` comments) →
本课要点 card → quiz (3 MCQ + 2 open). **zh ≥3500 CJK.** Only `shell.CSS` classes. Validators 0/0.
Kernel honesty: explain intent + memory layout from cited bindings/signatures; do NOT reproduce CUDA
kernel bodies wholesale; cite the Python binding (`torch.ops.sgl_kernel.*`) + reference the csrc by path.

## Lessons

### L38 — 38-sgl-kernel-overview.html / "sgl-kernel 总览 / sgl-kernel overview"
**Scope:** the AOT native kernel package. `sgl-kernel/` is a separate C++/CUDA project (CMake) whose
`csrc/` holds the kernels (attention, gemm, moe, elementwise, allreduce, quantization, sampling…),
compiled into a `.so` and exposed to Python as `torch.ops.sgl_kernel.*`. The Python wrappers in
`sgl-kernel/python/sgl_kernel/` (e.g. `attention.py`, `gemm.py`) call those ops. AOT = compiled
ahead of time, shipped in the wheel; the layers (Part 8) call these for the hot paths. **Cited:**
`sgl-kernel/python/sgl_kernel/attention.py` (a wrapper calling `torch.ops.sgl_kernel.*`). **Read:**
`sgl-kernel/` (CMakeLists, csrc/ dir listing), `sgl-kernel/python/sgl_kernel/__init__.py`,
`sgl-kernel/python/sgl_kernel/attention.py`. **Diagrams:** a `layers` Python layer→sgl_kernel wrapper→
`torch.ops.sgl_kernel`→csrc CUDA kernel; a `table.t` csrc subdir → what kernels; a `cols` AOT vs JIT;
a `flow` build→.so→torch.ops→call. Forward-ref JIT 第 39 课, attention kernel 第 40 课.

### L39 — 39-jit-kernel.html / "JIT kernel / JIT kernels"
**Scope:** the lightweight just-in-time path. `python/sglang/jit_kernel/` compiles small CUDA kernels
ON DEMAND at runtime (load-and-cache) instead of shipping them AOT — handy for experimental ops,
arch-specific codegen, or kernels that depend on runtime shapes/flags. `load_jit(...)` compiles +
caches a module; ops like `activation.py` build a per-dtype module. Trade vs AOT: flexible / no wheel
bloat, but a first-call compile cost. **Cited:** `python/sglang/jit_kernel/utils.py ::load_jit` (or
`jit_kernel/activation.py`). **Read:** `jit_kernel/__init__.py`, `jit_kernel/utils.py` (load_jit ~206,
get_jit_cuda_arch ~412), `jit_kernel/activation.py`. **Diagrams:** a `vflow` first-call compile→cache→
reuse; a `cols` AOT (sgl-kernel) vs JIT (jit_kernel); a `table.t` when-to-use; a `flow` source→nvcc→
cached .so→call. Forward-ref sgl-kernel 第 38 课. (There is a project skill "add-jit-kernel" — cite the pattern.)

### L40 — 40-attention-kernel-dissection.html / "关键 attention kernel 剖析"
**Scope:** what a real attention kernel must do, concretely. A decode attention kernel reads the paged
KV (第 30 课) — gathering K/V blocks via the page table — computes Q·Kᵀ, softmax, and ·V, all while
TILING the work to fit registers/shared memory and maximize memory-bandwidth use (decode is
bandwidth-bound, 第 4 课). Key ideas: the KV memory LAYOUT (NHD vs vectorized), block/page-wise gather,
online softmax (FlashAttention-style, no full score matrix), the split between prefill (big GEMM-like)
and decode (skinny, gather-heavy) kernels. The backends (第 33 课) wrap these. **Cited:**
`sgl-kernel/python/sgl_kernel/attention.py ::cutlass_mla_decode` (the Python wrapper, citing the csrc).
**Read:** `sgl-kernel/python/sgl_kernel/attention.py`, `sgl-kernel/csrc/attention/` (dir/headers),
`srt/mem_cache/memory_pool.py` (the layout, recall 第 30 课). **Diagrams:** a `flow`/`vflow` Q×Kᵀ→
softmax→×V with paged-KV gather; a `cellgroup` of KV pages gathered by a page table; a `cols` prefill
vs decode kernel; a `table.t` why tiling / online softmax / layout. Forward-ref CUDA graph 第 41 课.

### L41 — 41-operator-fusion-and-cuda-graph.html / "算子融合与 CUDA Graph 配合"
**Scope:** two kernel-level speedups and how they cooperate. **Fusion**: merge several small ops into
one kernel to avoid extra HBM round-trips and launch overhead — e.g. `SiluAndMul` (gate×up in one
kernel), fused add+RMSNorm, fused qk-norm+rope. **CUDA graph cooperation** (recall 第 27 课): graphs
need static shapes + no dynamic control flow, so fused kernels with fixed shapes are graph-friendly;
ops with data-dependent behavior must stay outside the captured region or use piecewise/breakable
graphs. Together: fewer, fused, static kernels → captured once → replayed with near-zero overhead. **Cited:**
`srt/layers/activation.py ::SiluAndMul` (calls `torch.ops.sgl_kernel.silu_and_mul`) or
`srt/layers/layernorm.py` (fused_add_rmsnorm). **Read:** activation.py (SiluAndMul ~90), layernorm.py
(fused_add_rmsnorm ~514), recall 第 27 课. **Diagrams:** a `flow` unfused (op→HBM→op→HBM) vs fused (one
kernel); a `table.t` common fusions; a `cols` fusion-friendly vs graph-breaking ops; a `vflow` fuse→
static shape→capture→replay. Forward-ref CUDA graph 第 27 课, kernels 第 38 课.

### L42 — 42-multi-hardware-backends.html / "多硬件后端 / Multi-hardware backends"
**Scope:** one engine, many chips. A platform abstraction (`SRTPlatform` + per-device subclasses
`CudaSRTPlatform`, ROCm, etc.) + `hardware_backend/` (cpu/gpu/npu/xpu/musa/mlx) lets SGLang run on
NVIDIA, AMD (ROCm), Google TPU (sglang-jax), Ascend NPU, Intel XPU, CPU. The upper layers
(scheduler/model/layers) are hardware-agnostic; only the kernels + platform hooks change per chip —
the attention backend (第 33 课), the kernel package (第 38 课), and platform-specific ops are swapped.
This is the engineering basis for "single GPU to large clusters across many hardware". **Cited:**
`srt/platforms/interface.py ::SRTPlatform` (or `platforms/cuda.py ::CudaSRTPlatform`). **Read:**
`srt/platforms/interface.py` (SRTPlatform ~26), `srt/platforms/cuda.py` (CudaSRTPlatform ~65),
`srt/hardware_backend/` (dir listing). **Diagrams:** a `layers` hardware-agnostic upper layers /
platform line / per-chip kernels; a `table.t` hardware → backend/kernels; a `cols` what's portable vs
what's per-chip; a `flow` op → platform dispatch → device kernel. Forward-ref attention backend 第 33
课, sgl-kernel 第 38 课, the design theme 第 62 课. Closes Part 9.

## Wiring & DoD
- New module `src/part9.py` (`LESSON_38..42`); `registry.py` imports `part9` + 5 keys; `shell.PAGES` +
  `SUBTITLES` += 5; `quizzes.QUIZZES` += 5. Filenames as above. Part label
  "第九部分 · 内核与硬件 / Part 9 · Kernels & hardware".
- All validators 0 err / 0 warn; no-diff; index pill "共 42 课 · 9 个部分"; nav L37↔L38…L42.
- Source-accurate: cite the Python binding + reference csrc by path; never paste large CUDA bodies.
