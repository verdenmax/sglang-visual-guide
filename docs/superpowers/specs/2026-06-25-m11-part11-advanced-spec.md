# M11 — Part 11 Advanced / Optional (L49–L52) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec §P11, roadmap M11. Builds on M0–M10 (L01–48 exist).

## Goal
The "advanced / optional reading" part: four big capabilities beyond text LLM serving that reuse the
same engine — multimodal (VLM) serving, multi-LoRA batching, the RL rollout / weight-sync path, and
diffusion (image/video) models. Theme: SGLang's core (scheduler, paged KV, batching, kernels,
multi-hardware) is a REUSABLE substrate; each capability plugs in at a well-defined seam. Module:
`src/part11.py`, lessons L49–52 (4).

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy (`card analogy` + `<div class="tag">🔌 生活类比</div>` zh / `🔌 Analogy` en) → 🌍 macro
(`card macro`, `🌍 宏观理解` / `🌍 The big picture`) → 3–4 `<h2>` with EXACTLY 4 visual blocks/lang
(identical zh/en; from cols/flow/vflow/layers/cellgroup/`table.t`/step+num+sc; NO `<svg>`, NOT
`timeline`) → one cited `.codefile` (`<div class="cf-head"><span class="dot"></span><span
class="path">file ::symbol</span><span class="ln">…</span></div>`; docstrings in `<pre>` as `#`
comments; escape `<`/`>`/`&`) → key card (`card key`, `📌 本课要点` / `📌 Key points`) → quiz (3 MCQ +
2 open). **zh ≥3600 CJK.** Only `shell.CSS` classes. Cite REAL symbols only.

## Lessons

### L49 — 49-multimodal-vlm-serving.html / "多模态 VLM 服务 / Multimodal VLM serving"
**Scope:** serving vision-language models (images/audio/video + text). Two seams. (1) INPUT: a
per-model `BaseMultimodalProcessor` turns raw pixels/audio into model tensors and inserts PLACEHOLDER
tokens into the prompt where the media goes. (2) EMBED: at forward time `general_mm_embed_routine`
runs the vision encoder (a ViT, sometimes with its own CUDA-graph runner) to get media embeddings,
then SPLICES them into the token embedding stream at the placeholder positions (per-modality
`data_embedding_funcs` keyed by `placeholder_tokens`). After that splice the rest of the engine —
scheduler (第18课), paged KV (第30课), attention (第33课) — runs UNCHANGED, because by then it's just a
sequence of embeddings. So VLM support is mostly "encode media + weave embeddings in", not a new
engine. **Cited:** `python/sglang/srt/managers/mm_utils.py ::general_mm_embed_routine` (input_ids +
data_embedding_funcs + placeholder_tokens). **Read:** `managers/mm_utils.py`
(general_mm_embed_routine, embed_mm_inputs, get_embedding_and_mask),
`multimodal/processors/base_processor.py` (`BaseMultimodalProcessor`), `multimodal/` (vit cuda graph).
**Diagrams:** a `flow` raw image → processor → placeholder tokens in prompt → ViT encode → splice
embeddings → normal decode; a `layers`/`cellgroup` token-embedding row with image embeddings spliced
at placeholder slots; a `cols` what's VLM-specific (processor + encoder + splice) vs what's reused
(scheduler/KV/attention); a `table.t` the two seams → role. Tie 第14课 (tokenizer/processor), 第30/33课.

### L50 — 50-multi-lora-batching.html / "多 LoRA 批处理 / Multi-LoRA batching"
**Scope:** serving MANY fine-tuned adapters on ONE base model, even mixed in one batch. A LoRA adapter
is a small low-rank ΔW (`LoRAAdapter`) added to the base weights; instead of loading N full models,
SGLang keeps ONE base model + a POOL of small adapters and applies the right one per request. The hard
part is BATCHING requests that use DIFFERENT adapters together: `LoRAManager.prepare_lora_batch` gathers,
for the current `forward_batch`, which adapter each request needs and stages them (segmented/grouped
GEMM applies per-request ΔW in one kernel), bounded by `max_loras_per_batch`. Adapters are added/removed
at runtime via `load_lora_adapter` / `unload_lora_adapter` against a LoRA memory pool. This is how a
single deployment serves dozens of task-specific adapters cheaply. **Cited:**
`python/sglang/srt/lora/lora_manager.py ::LoRAManager` (prepare_lora_batch + load/unload +
max_loras_per_batch). **Read:** `lora/lora_manager.py`, `lora/lora.py` (`LoRAAdapter`),
`lora/mem_pool.py`, `lora/backend/`. **Diagrams:** a `layers`/`flow` base weight + per-request ΔW
(adapter) → output; a `cellgroup` one batch where rows use different adapters (A/B/A/C); a `table.t`
LoRAManager method → role; a `cols` N full models (naive) vs 1 base + adapter pool (SGLang). Tie 第24课
(ForwardBatch), 第25课 (weights), 第5课 (batching). Forward-ref 第51课 (RL also swaps weights).

### L51 — 51-rl-rollout-and-weight-sync.html / "RL Rollout 与权重同步"
**Scope:** SGLang as the fast GENERATION ("rollout") engine inside an RL / RLHF training loop. RL
alternates: the policy GENERATES samples (rollout — what SGLang is great at), the trainer computes a
reward + gradient and UPDATES the weights; then the rollout engine must run with the NEW weights for
the next round. Restarting the server each step is far too slow, so SGLang exposes IN-PLACE weight
update: `update_weights_from_tensor(named_tensors, load_format=...)` pushes freshly-trained weights
straight into the running model's parameters (and `update_weights_from_distributed` streams them over a
process group from the trainer ranks). To move many tensors efficiently, the
`load_format="flattened_bucket"` path packs them with a `FlattenedTensorBucket` (flatten many named
tensors into ONE buffer → one transfer/copy → reconstruct), cutting per-tensor overhead. So the same
engine is both the serving server and the RL rollout worker, just with a weight-sync hook. **Cited:**
`python/sglang/srt/entrypoints/engine.py ::update_weights_from_tensor` (named_tensors + flattened_bucket).
**Read:** `entrypoints/engine.py` (update_weights_from_tensor/_from_distributed),
`weight_sync/tensor_bucket.py` (`FlattenedTensorBucket`),
`managers/scheduler_components/weight_updater.py`. **Diagrams:** a `flow`/`vflow` RL loop: rollout
(generate) → reward+grad (trainer) → update_weights → next rollout; a `cols` restart-the-server (slow)
vs in-place update_weights (fast); a `table.t` from_tensor vs from_distributed vs flattened_bucket →
when; a `layers`/`flow` many named tensors → FlattenedTensorBucket (one buffer) → one copy → params. Tie
第13/14课 (engine entrypoints), 第25课 (weight loading), 第50课 (also swaps weights). Forward-ref design
themes 第59-63课.

### L52 — 52-diffusion-models.html / "扩散模型 / Diffusion models"
**Scope:** image/video generation (a DIFFERENT compute pattern) on the SGLang stack — the
`sglang-diffusion` sub-project (`python/sglang/multimodal_gen/`). Unlike autoregression (one token per
step), diffusion ITERATIVELY DENOISES: start from pure noise and run the model N times, each step
removing a bit of noise toward the final image/video; a scheduler/pipeline drives the denoise loop and
a `PipelineConfig` describes the model + steps. The win: SGLang REUSES its serving muscle — the
optimized sgl-kernel ops (第38课), an efficient scheduler loop, CUDA-graph capture (第27课),
quantization, multi-hardware (第42课, NVIDIA/AMD/NPU/Apple/Moore-Threads), an OpenAI-compatible API
(第15课), and even PD-style disaggregation — for diffusion. Diffusion-specific tricks like TeaCache
(cache + skip redundant denoise steps) layer on top. So "one stack, two paradigms": autoregressive LLM
and iterative diffusion share the same infra. **Cited:**
`python/sglang/multimodal_gen/configs/pipeline_configs/base.py ::PipelineConfig` (the diffusion
pipeline descriptor). **Read:** `multimodal_gen/README.md`, `multimodal_gen/configs/pipeline_configs/`
(`PipelineConfig`, `ImagePipelineConfig`), `multimodal_gen/runtime/` (scheduler/launch),
`multimodal_gen/runtime/cache/teacache.py`. **Diagrams:** a `vflow`/`flow` noise → denoise×N (the model
repeatedly) → image; a `cols` autoregressive (token-by-token) vs diffusion (iterative denoise); a
`table.t` reused SGLang piece → benefit for diffusion (kernels/scheduler/CUDA graph/multi-hw/API); a
`cellgroup`/`layers` the denoise schedule steps (with TeaCache skipping some). Tie 第38/27/42/15课.
Closes Part 11 with a short wrap-up note.

## Wiring & DoD
- New module `src/part11.py` (`LESSON_49..52`); `registry.py` imports `part11` + 4 keys; `shell.PAGES`
  + `SUBTITLES` += 4; `quizzes.QUIZZES` += 4. Part label "第十一部分 · 进阶选读" /
  "Part 11 · Advanced (optional)".
- Filenames: `49-multimodal-vlm-serving.html`, `50-multi-lora-batching.html`,
  `51-rl-rollout-and-weight-sync.html`, `52-diffusion-models.html`.
- All validators 0 err / 0 warn; no-diff; pill "共 52 课 · 11 个部分"; nav L48↔L49…L52.
- Source-accurate: cite real symbols; note L52 lives under `python/sglang/multimodal_gen/` (not srt/).
