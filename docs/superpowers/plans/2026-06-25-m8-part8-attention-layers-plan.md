# M8 — Part 8 Attention & Layers (L33–L37) — Plan

> One sync subagent per lesson, then two-stage review. Spec:
> `specs/2026-06-25-m8-part8-attention-layers-spec.md`.

**Goal:** Add L33–L37 (the operator layer) as `src/part8.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source:** `~/course/sglang/python/sglang/srt/layers/`.

## Common recipe (every lesson)
Model on part1–7. Each `LESSON_NN`: lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` → **EXACTLY 4 visual
blocks/lang, identical zh/en** (cols/vflow/flow/layers/cellgroup/`<table class="t">`; NOT `timeline`;
no SVG) → ONE cited `.codefile` (`file ::symbol`; `"""docstring"""` in `<pre>` → `#` comment) → 本课要点
card → quiz (3 MCQ + 2 open). **zh ≥ 3500 CJK.** Only `shell.CSS` classes. Smoke-build after each.

## Task 0 — module + wiring (orchestrator)
- [ ] Create `src/part8.py` (`LESSON_33..37` placeholders); `import part8` in `registry.py`; add 5 PAGES +
  5 SUBTITLES + 5 CONTENT keys. Part label "第八部分 · Attention 与算子层 / Part 8 · Attention & layers".
  Filenames: `33-attention-backend-abstraction.html`, `34-moe-layer.html`, `35-quantization.html`,
  `36-rope-norm-and-ops.html`, `37-logits-and-vocab-parallel.html`.

## Per-lesson tasks (read source, author per spec, add quiz, smoke-build)
- [ ] **L33** Attention backend abstraction. Cite `srt/layers/attention/base_attn_backend.py
  ::AttentionBackend` (~18; FlashInferAttnBackend ~291, TritonAttnBackend ~103; RadixAttention layer in
  `layers/radix_attention.py` ~57). Analogy: a **power-tool with swappable bits/drivers** — the model
  asks for "attention", the backend (FlashInfer/Triton/FA) is the interchangeable driver picked for your
  hardware. Diagrams: layers model→RadixAttention→AttentionBackend→{backends}; table.t backend→strengths;
  cols extend-vs-decode attention; flow one attention call (plan metadata→kernel).
- [ ] **L34** MoE layer. Cite `srt/layers/moe/fused_moe_triton/layer.py ::FusedMoE` (~136). Analogy: a
  **panel of specialists with a triage nurse (router)** — each token sees only the top-k relevant
  specialists, not the whole panel (sparse). Diagrams: flow token→router→top-k experts→combine; cols dense
  FFN vs MoE; table.t MoE terms; cellgroup tokens→different experts. Forward-ref EP 第 47 课.
- [ ] **L35** Quantization. Cite `srt/layers/quantization/fp8.py ::Fp8LinearMethod` (~321; QuantizationConfig
  in `quantization/base_config.py` ~126, LinearMethodBase ~46). Analogy: **JPEG for weights** — store fewer
  bits + a scale, accept a tiny quality loss for big size/bandwidth savings. Diagrams: vflow fp16→quantize
  (scale)→store→dequant/quantized-matmul; table.t formats (FP8/FP4/INT4/AWQ/GPTQ→bits/where); cols
  weight-only vs weight+activation; cellgroup scales. Forward-ref loading 第 25 课, kernels 第 38 课.
- [ ] **L36** RoPE/Norm/ops. Cite `srt/layers/rotary_embedding/base.py ::RotaryEmbedding` (~75; get_rope in
  `rotary_embedding/factory.py` ~63; RMSNorm in `layers/layernorm.py` ~203). Analogy: **RoPE = rotating the
  clock hands by position** so attention feels relative distance; RMSNorm = a lightweight volume-leveler.
  Diagrams: vflow q/k→apply RoPE(positions)→attention; table.t op→what/why; cols LayerNorm vs RMSNorm;
  cellgroup/flow RoPE rotating a vector by position. Mention NTK/YaRN context extension. Forward-ref 第 33 课.
- [ ] **L37** Logits & vocab parallel. Cite `srt/layers/logits_processor.py ::LogitsProcessor` (~260;
  VocabParallelEmbedding in `vocab_parallel_embedding.py` ~185, ParallelLMHead ~541). Analogy: **a giant
  dictionary split among N clerks** — each clerk scores its slice of the vocab, then they merge to the full
  score sheet (all-gather). Diagrams: layers/flow hidden→lm_head(vocab-sharded)→gather→logits→Sampler; cols
  full-vocab vs vocab-parallel; table.t what's sharded/gathered; cellgroup vocab shards across ranks.
  Forward-ref sampler 第 28 课, TP 第 46 课. Closes Part 8.

## Verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py`
  → 0 err / 0 warn; pill "共 37 课 · 8 个部分"; nav L32↔L33↔…↔L37; no-diff.
- [ ] One commit: `M8: Part 8 attention & layers — L33 attention backend, L34 MoE, L35 quantization, L36 RoPE/norm, L37 logits/vocab-parallel` (+ Co-authored-by trailer).

## Guardrails
- Cite `file ::symbol`; the attention-backend abstraction (model→RadixAttention→AttentionBackend) and the
  vocab-parallel split + gather must match the code. No `timeline`.
- zh ≥3500 CJK; zh/en identical diagram inventory. Don't touch `docs/`, earlier parts, pipeline, reference repo.
