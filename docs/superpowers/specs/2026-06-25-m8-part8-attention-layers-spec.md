# M8 — Part 8 Attention & Layers (L33–L37) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M8. Builds on M0–M7 (L01–32 exist).

## Goal
The operator layer that the model (第 26 课) is built from: the pluggable attention backend, the MoE
layer, quantization, RoPE/normalization, and logits/vocab-parallel. These are SGLang's reusable
`nn.Module`s under `srt/layers/`. Module: `src/part8.py`, lessons L33–37.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with EXACTLY 4 visual blocks/lang (identical zh/en; no SVG;
NOT `timeline`) → one cited `.codefile` (`file ::symbol`; docstrings in `<pre>` as `#` comments) →
本课要点 card → quiz (3 MCQ + 2 open). **zh ≥3500 CJK.** Only `shell.CSS` classes. Validators 0/0.

## Lessons

### L33 — 33-attention-backend-abstraction.html / "Attention 后端抽象 / Attention backend abstraction"
**Scope:** attention is a swappable strategy. The model's `RadixAttention` layer (第 7/29 课) delegates
the actual kernel to an `AttentionBackend` (an ABC): FlashInfer, Triton, FlashAttention(3), etc. The
backend plans metadata per forward (which KV pages to read, the causal mask) and runs the kernel for
EXTEND vs DECODE. Chosen by `--attention-backend` / hardware. Why an interface: new kernels/hardware
plug in without touching models. **Cited:** `srt/layers/attention/base_attn_backend.py ::AttentionBackend`
(or `srt/layers/radix_attention.py ::RadixAttention`). **Read:** attention/base_attn_backend.py
(AttentionBackend ~18), attention/flashinfer_backend.py (~291), attention/triton_backend.py (~103),
radix_attention.py (~57). **Diagrams:** a `layers` model→RadixAttention layer→AttentionBackend→{FlashInfer/
Triton/FA}; a `table.t` backend → strengths; a `cols` extend vs decode attention; a `flow` of one attention call.

### L34 — 34-moe-layer.html / "MoE 层 / The MoE layer"
**Scope:** mixture-of-experts. Instead of one big FFN, an MoE layer has N experts; a router (gate)
picks top-k experts per token; only those run (sparse compute → more params, ~same FLOPs/token). SGLang's
`FusedMoE` fuses the routing + grouped GEMM into efficient kernels; expert parallelism (第 46/47 课)
spreads experts across GPUs. DeepSeek/Mixtral/Qwen-MoE use this. **Cited:** `srt/layers/moe/
fused_moe_triton/layer.py ::FusedMoE`. **Read:** moe/fused_moe_triton/layer.py (FusedMoE ~136).
**Diagrams:** a `flow` token→router→top-k experts→combine; a `cols` dense FFN vs MoE; a `table.t` MoE
terms (expert/router/top-k/grouped GEMM); a `cellgroup` of tokens routed to different experts. Forward-ref EP 第 47 课.

### L35 — 35-quantization.html / "量化 / Quantization"
**Scope:** fewer bits per weight/activation. FP8/FP4/INT4/AWQ/GPTQ shrink the model (less HBM, less
bandwidth → faster) at a small accuracy cost. SGLang models a quant method via a `QuantizationConfig` +
a `LinearMethod` (e.g. `Fp8LinearMethod`) that replaces how a linear layer stores weights and does the
matmul (quantized kernel + scales). Weight-only vs weight+activation; per-tensor/channel/group scales;
KV-cache quant (第 8 课 ties). **Cited:** `srt/layers/quantization/fp8.py ::Fp8LinearMethod` (or
`quantization/base_config.py ::QuantizationConfig`). **Read:** quantization/base_config.py
(QuantizationConfig ~126, LinearMethodBase ~46), quantization/fp8.py (Fp8Config ~147, Fp8LinearMethod ~321).
**Diagrams:** a `vflow` fp16 weight→quantize(scale)→store→dequant/quantized-matmul; a `table.t` formats
(FP8/FP4/INT4/AWQ/GPTQ → bits/where); a `cols` weight-only vs weight+activation; a `cellgroup` of scales.
Forward-ref model loading 第 25 课, kernels 第 38 课.

### L36 — 36-rope-norm-and-ops.html / "RoPE、归一化与其它算子 / RoPE, norm & other ops"
**Scope:** the smaller-but-essential ops in a layer. **RoPE** (rotary position embedding) rotates q/k by
position so attention is relative-position aware (and supports context-length extension: NTK/YaRN/linear
scaling). **RMSNorm** (the norm Llama-family uses) — cheaper than LayerNorm, often fused. `SiluAndMul`
and other activation fusions. These are reusable `nn.Module`s the model composes (第 26 课). **Cited:**
`srt/layers/rotary_embedding/base.py ::RotaryEmbedding` (or `srt/layers/layernorm.py ::RMSNorm`). **Read:**
rotary_embedding/base.py (RotaryEmbedding ~75), rotary_embedding/factory.py (get_rope ~63), layernorm.py
(RMSNorm ~203). **Diagrams:** a `vflow` q/k → apply RoPE(positions) → attention; a `table.t` op → what/why;
a `cols` LayerNorm vs RMSNorm; a `cellgroup`/`flow` of RoPE rotating a vector by position. Forward-ref
context-length extension, attention 第 33 课.

### L37 — 37-logits-and-vocab-parallel.html / "Logits 处理与词表并行 / Logits & vocab parallel"
**Scope:** the output head. `VocabParallelEmbedding` / `ParallelLMHead` split the (huge) vocab dimension
across TP ranks — each rank computes logits for its vocab shard, then an all-gather/all-reduce assembles
the full logits. The `LogitsProcessor` takes hidden states → lm_head → logits, handling the last-token
slice (only the final position needs logits in decode) and TP gather, before the Sampler (第 28 课). Why
shard vocab: the vocab can be 128k+, so its embedding/lm_head matrices are huge. **Cited:**
`srt/layers/logits_processor.py ::LogitsProcessor` (or `vocab_parallel_embedding.py ::VocabParallelEmbedding`).
**Read:** logits_processor.py (LogitsProcessor ~260), vocab_parallel_embedding.py (VocabParallelEmbedding
~185, ParallelLMHead ~541). **Diagrams:** a `layers`/`flow` hidden→lm_head(vocab-sharded)→gather→logits→Sampler;
a `cols` full-vocab vs vocab-parallel; a `table.t` what's sharded/gathered; a `cellgroup` of vocab shards
across ranks. Forward-ref sampler 第 28 课, TP 第 46 课. Closes Part 8.

## Wiring & DoD
- New module `src/part8.py` (`LESSON_33..37`); `registry.py` imports `part8` + 5 keys; `shell.PAGES` +
  `SUBTITLES` += 5; `quizzes.QUIZZES` += 5. Filenames as above. Part label
  "第八部分 · Attention 与算子层 / Part 8 · Attention & layers".
- All validators 0 err / 0 warn; no-diff; index pill "共 37 课 · 8 个部分"; nav L32↔L33…L37.
- Source-accurate (`file ::symbol`); the backend-abstraction + vocab-parallel claims must match the code.
