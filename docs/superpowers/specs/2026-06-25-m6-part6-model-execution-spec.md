# M6 — Part 6 Model Execution (L24–L28) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M6. Builds on M0–M5 (L01–23 exist).

## Goal
What `run_batch` (第 18 课) actually triggers on the GPU: the `ModelRunner` + `ForwardBatch` that
organize one forward pass, how model weights are loaded, how a model is written (Llama as the worked
example), how CUDA graphs cut per-step launch overhead, and how the `Sampler` turns logits into the
next token. Module: `src/part6.py`, lessons L24–28.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with ≥3 visual blocks/lang (identical zh/en; no SVG;
NOT `timeline`) → one cited `.codefile` (`file ::symbol`, docstrings in `<pre>` as `#` comments) →
本课要点 card → quiz (3 MCQ + 2 open). **zh ≥3500 CJK.** Only `shell.CSS` classes. Validators 0/0.

## Lessons

### L24 — 24-model-runner-and-forward-batch.html / "ModelRunner 与 ForwardBatch"
**Scope:** the GPU executor. `ModelRunner` (one per TP rank, owned by TpWorker) holds the model + KV
pool and runs `forward(forward_batch)`. A `ForwardBatch` is the GPU-side view of a ScheduleBatch (第 19
课): input ids, positions, the `forward_mode` (EXTEND/DECODE), the attention metadata, and the KV pool
indices. `forward` dispatches to the model, returns logits; `sample` calls the Sampler (第 28 课).
**Cited:** `srt/model_executor/model_runner.py ::ModelRunner` (or `::ModelRunner.forward`). **Read:**
model_runner.py (ModelRunner ~343, forward ~2915, sample ~3121), forward_batch_info.py (ForwardBatch
~322, ForwardMode ~77). **Diagrams:** a `flow` ScheduleBatch→ForwardBatch→model.forward→logits→Sampler;
a `table.t` ForwardBatch fields → role; a `cols` extend-forward vs decode-forward; a `vflow` of one forward.

### L25 — 25-model-loading-and-weights.html / "模型加载与权重 / Model loading & weights"
**Scope:** getting weights onto GPUs. A `ModelLoader` (`DefaultModelLoader`) reads HF checkpoint
shards (safetensors), maps HF parameter names → SGLang module params (a `load_weights` per model),
sharding each tensor for TP (column/row parallel), and casting to the runtime dtype / quantized format.
Lazy/streamed loading to bound host RAM. **Cited:** `srt/model_loader/loader.py ::DefaultModelLoader`
(or `::DefaultModelLoader.load_model`). **Read:** model_loader/loader.py (BaseModelLoader ~329,
DefaultModelLoader ~351, load_model ~740). **Diagrams:** a `vflow` checkpoint→shards→name-map→shard-for-TP
→dtype/quant→GPU; a `table.t` loader concerns; a `cols` HF names vs SGLang params; a `layers` of the load path.

### L26 — 26-writing-a-model.html / "写一个模型 / Writing a model"
**Scope:** how a model file is structured (Llama as the example). A model is plain PyTorch `nn.Module`s
composed from SGLang's parallel layers: `LlamaAttention` (uses the attention backend + RoPE + a KV
slot), `LlamaDecoderLayer` (attn + MLP + norms), `LlamaModel` (embed + N layers), `LlamaForCausalLM`
(model + lm_head + `forward(input_ids, positions, forward_batch)` + `load_weights`). Adding a model =
write these + register it. This is why "broad model support" is cheap. **Cited:**
`srt/models/llama.py ::LlamaForCausalLM` (or `::LlamaModel`). **Read:** models/llama.py (LlamaAttention
~126, LlamaDecoderLayer ~255, LlamaModel ~338, LlamaForCausalLM ~462). **Diagrams:** a `layers` of the
model stack (embed→decoder layers→norm→lm_head); a `vflow` forward through a decoder layer; a `table.t`
the 4 classes → role; a `cols` "what you write vs what SGLang provides (parallel layers/attn backend)".

### L27 — 27-cuda-graph-capture-and-replay.html / "CUDA Graph 捕获与重放"
**Scope:** killing per-step launch overhead. Decode steps launch hundreds of tiny kernels; the CPU
launch overhead dominates. A CUDA graph CAPTURES the whole forward's kernel sequence once (per batch
size) and REPLAYS it as a single op — no per-kernel CPU launch. SGLang captures graphs for a set of
batch sizes at startup (`BaseCudaGraphRunner.capture`), PADS the real batch up to the nearest captured
size, and replays. Constraints: static shapes (hence padding + bucketed sizes), no data-dependent
control flow (hence "breakable"/piecewise graphs for some ops). Pairs with the overlap scheduler (第 21
课). **Cited:** `srt/model_executor/runner/base_cuda_graph_runner.py ::BaseCudaGraphRunner` (capture
~155). **Read:** runner/base_cuda_graph_runner.py, decode_cuda_graph_runner.py, cuda_graph_config.py.
**Diagrams:** a `cols` no-graph (per-kernel launch) vs graph (one replay); a `vflow` capture→bucket→pad
→replay; a `table.t` constraints (static shape, padding, bucketed sizes); a `cellgroup` of batch-size buckets.

### L28 — 28-sampler-and-sampling-params.html / "采样器与采样参数 / Sampler & sampling params"
**Scope:** logits → next token. The `Sampler` (an `nn.Module`) takes the model's logits and the batch's
`SamplingBatchInfo` and produces the next token id per request: apply temperature, top-k/top-p, min-p,
repetition/frequency/presence penalties, then sample (or argmax for greedy). `SamplingParams` is the
per-request knob set. Structured-output constraints (第 48 课) and logit bias hook in here. **Cited:**
`srt/layers/sampler.py ::Sampler` (forward ~93). **Read:** layers/sampler.py (Sampler ~68, forward ~93),
sampling/sampling_batch_info.py (SamplingBatchInfo ~24), sampling/sampling_params.py (SamplingParams ~64).
**Diagrams:** a `vflow` logits→penalties→temperature→top-k/p→sample; a `table.t` param → effect
(temperature/top-p/top-k/penalties); a `cols` greedy vs sampling; a `cellgroup` of a top-p truncation.
Forward-ref structured outputs 第 48 课.

## Wiring & DoD
- New module `src/part6.py` (`LESSON_24..28`); `registry.py` imports `part6` + 5 keys; `shell.PAGES` +
  `SUBTITLES` += 5; `quizzes.QUIZZES` += 5. Filenames as above. Part label
  "第六部分 · 模型执行 / Part 6 · Model execution".
- All validators 0 err / 0 warn; no-diff; index pill "共 28 课 · 6 个部分"; nav L23↔L24…L28.
- Source-accurate (`file ::symbol`); the ModelRunner→model→Sampler chain must match model_runner.py.
