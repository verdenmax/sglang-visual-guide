# M9 — Part 9 Kernels & Hardware (L38–L42) — Plan

> Execute with superpowers:subagent-driven-development. One subagent per lesson, model
> claude-opus-4.8, sync, high effort. Two-stage review (research + code-review, opus-4.8 max,
> long context) after exec. Edit large files incrementally.

**Goal:** add Part 9 (L38–42) to the SGLang visual guide — the deep kernel + multi-hardware part.
**Companion spec:** `docs/superpowers/specs/2026-06-25-m9-part9-kernels-hardware-spec.md`.

## Verified citations (pre-fetched — pass to subagents verbatim)
- L38 `sgl-kernel/python/sgl_kernel/attention.py ::merge_state_v2` (line 6; calls
  `torch.ops.sgl_kernel.merge_state_v2.default`) — the AOT Python→torch.ops binding pattern.
- L39 `python/sglang/jit_kernel/utils.py ::load_jit` (line 206) — runtime JIT compile+cache.
- L40 `sgl-kernel/python/sgl_kernel/attention.py ::cutlass_mla_decode` (line 29; calls
  `torch.ops.sgl_kernel.cutlass_mla_decode.default`); csrc: `sgl-kernel/csrc/attention/`
  (`cutlass_mla_kernel.cu`, `merge_attn_states.cu`).
- L41 `python/sglang/srt/layers/activation.py ::SiluAndMul` (line 90; calls `silu_and_mul` →
  `torch.ops.sgl_kernel.silu_and_mul*`). Alt: `layers/layernorm.py` `fused_add_rmsnorm` (line 327).
- L42 `python/sglang/srt/platforms/interface.py ::SRTPlatform` (line 26); `platforms/cuda.py
  ::CudaSRTPlatform` (line 65); dirs `srt/platforms/{cpu,cuda,rocm,interface}.py`,
  `srt/hardware_backend/{cpu,gpu,npu,xpu,musa,mlx}`.

## Task 0 — orchestrator wiring (do myself)
- [ ] Create `src/part9.py` with `LESSON_38..LESSON_42` placeholders
  (`{"zh": "PLACEHOLDER … " * 4, "en": "PLACEHOLDER … " * 3}`).
- [ ] `registry.py`: add `import part9`; add 5 CONTENT keys (filenames below → `part9.LESSON_3X`).
- [ ] `shell.py`: append 5 PAGES 5-tuples (anchor after L37) + 5 SUBTITLES; part label
  "第九部分 · 内核与硬件" / "Part 9 · Kernels & hardware".
- [ ] `quizzes.py`: subagents append; nothing here in Task 0.
- [ ] Build + confirm pill "共 42 课 · 9 个部分", nav L37↔L38. (Placeholders fail CJK WARN only.)

Filenames: `38-sgl-kernel-overview.html`, `39-jit-kernel.html`,
`40-attention-kernel-dissection.html`, `41-operator-fusion-and-cuda-graph.html`,
`42-multi-hardware-backends.html`.

## Tasks 1–5 — per-lesson subagents (L38…L42)
Each: replace that lesson's placeholder in `src/part9.py` + append its `quizzes.py` entry. Prompt
template (proven): exact placeholder string; verified `file ::symbol`; allowed CSS palette
(lead/analogy/macro/cols/vflow/flow/layers/cellgroup/`table.t`/codefile/cards/quiz); "real prose
≥3500 CJK zh, identical zh/en visual inventory, EXACTLY 4 diagram blocks, NO SVG, NOT timeline,
docstrings→`#` in `<pre>`, don't over-research"; VERIFY cmds
(`cd src && python3 build.py && python3 check_html.py && python3 check_links.py`). Retry on
truncation; draft+expand fallback after 3 fails.

- [ ] **T1 L38** sgl-kernel overview — cite `merge_state_v2`. Diagrams: layers (Py layer→wrapper→
  torch.ops→csrc), table.t (csrc subdir→kernels), cols (AOT vs JIT), flow (build→.so→op→call).
- [ ] **T2 L39** JIT kernel — cite `load_jit`. Diagrams: vflow (compile→cache→reuse), cols (AOT vs
  JIT), table.t (when to use), flow (src→nvcc→cached .so→call).
- [ ] **T3 L40** attention kernel dissection — cite `cutlass_mla_decode` (+ csrc path). Diagrams:
  flow (Q×Kᵀ→softmax→×V + paged gather), cellgroup (KV pages via page table), cols (prefill vs
  decode), table.t (tiling/online-softmax/layout).
- [ ] **T4 L41** operator fusion & CUDA graph — cite `SiluAndMul`. Diagrams: flow (unfused vs fused),
  table.t (common fusions), cols (fusion-friendly vs graph-breaking), vflow (fuse→static→capture→
  replay). Tie to 第 27 课.
- [ ] **T5 L42** multi-hardware backends — cite `SRTPlatform`/`CudaSRTPlatform`. Diagrams: layers
  (agnostic upper / platform / per-chip kernels), table.t (hw→backend), cols (portable vs per-chip),
  flow (op→platform dispatch→device kernel). Closes Part 9.

## Task 6 — verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py &&
  python3 check_links.py` → 0 err / 0 warn; `git diff` no-diff after rebuild.
- [ ] Commit (`verdenmax`, Co-authored-by Copilot).

## Task 7 — two-stage review + fixes
- [ ] research agent (concept/spec/completeness) + code-review agent (markup/citations/parity),
  parallel, opus-4.8 max, long context. Fix findings (forward-refs → curriculum map; real symbols
  only; zh/en parity). Re-verify + commit.

## DoD
42 lessons; Part 9 nav + pill correct; validators 0/0; no-diff; citations source-accurate
(Python binding + csrc path, no pasted CUDA bodies); spec scope met.
