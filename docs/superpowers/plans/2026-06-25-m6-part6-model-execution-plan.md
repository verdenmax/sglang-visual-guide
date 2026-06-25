# M6 — Part 6 Model Execution (L24–L28) — Plan

> One sync subagent per lesson, then two-stage review. Spec:
> `specs/2026-06-25-m6-part6-model-execution-spec.md`.

**Goal:** Add L24–L28 (model execution) as `src/part6.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source:** `~/course/sglang/python/sglang/srt/`.

## Common recipe (every lesson)
Model on part1–5. Each `LESSON_NN = {"zh": r"""…""","en": r"""…"""}`: lead → 🔌 analogy → 🌍 macro →
3–4 `<h2>` → **EXACTLY 4 visual blocks/lang, identical zh/en** (cols/vflow/flow/layers/cellgroup/
`<table class="t">`; NOT `timeline`; no SVG) → ONE cited `.codefile` (`file ::symbol`; inside `<pre>`
render any `"""docstring"""` as a `#` comment) → 本课要点 card → quiz (3 MCQ + 2 open). **zh ≥ 3500 CJK.**
Only `shell.CSS` classes. Forward-refs map to real lessons. Smoke-build after each.

## Task 0 — module + wiring (orchestrator)
- [ ] Create `src/part6.py` (`LESSON_24..28` placeholders); `import part6` in `registry.py`; add 5 PAGES +
  5 SUBTITLES + 5 CONTENT keys. Part label "第六部分 · 模型执行 / Part 6 · Model execution". Filenames:
  `24-model-runner-and-forward-batch.html`, `25-model-loading-and-weights.html`, `26-writing-a-model.html`,
  `27-cuda-graph-capture-and-replay.html`, `28-sampler-and-sampling-params.html`.

## Per-lesson tasks (read source, author per spec, add quiz, smoke-build)
- [ ] **L24** ModelRunner & ForwardBatch. Cite `srt/model_executor/model_runner.py ::ModelRunner` (~343;
  forward ~2915, sample ~3121; ForwardBatch in forward_batch_info.py ~322). Analogy: a **GPU foreman** who
  takes the work order (ForwardBatch), runs the machine (model.forward), and reads off the result (logits→Sampler).
  Diagrams: flow ScheduleBatch→ForwardBatch→forward→logits→Sampler; table.t ForwardBatch fields; cols
  extend-vs-decode forward; vflow one forward.
- [ ] **L25** Model loading & weights. Cite `srt/model_loader/loader.py ::DefaultModelLoader` (~351; load_model
  ~740). Analogy: **furnishing a house from flat-pack boxes** — read shards, match part names, cut each to fit
  each room (TP shard), assemble. Diagrams: vflow checkpoint→shards→name-map→TP-shard→dtype/quant→GPU; table.t
  loader concerns; cols HF-names-vs-SGLang-params; layers load path.
- [ ] **L26** Writing a model. Cite `srt/models/llama.py ::LlamaForCausalLM` (~462; LlamaModel ~338,
  LlamaDecoderLayer ~255, LlamaAttention ~126). Analogy: **LEGO from standard bricks** — compose a model from
  SGLang's parallel layers + attention backend; you only write the assembly. Diagrams: layers model stack
  (embed→N decoder layers→norm→lm_head); vflow forward through a decoder layer; table.t the 4 classes→role;
  cols what-you-write vs what-SGLang-provides.
- [ ] **L27** CUDA graph capture & replay. Cite `srt/model_executor/runner/base_cuda_graph_runner.py
  ::BaseCudaGraphRunner` (~103; capture ~155). Analogy: a **macro/player-piano roll** — record the whole
  keystroke sequence once, replay it in one shot instead of pressing each key. Diagrams: cols no-graph
  (per-kernel launch) vs graph (one replay); vflow capture→bucket→pad→replay; table.t constraints (static
  shape, padding, bucketed sizes); cellgroup batch-size buckets. Pairs with overlap scheduler 第 21 课.
- [ ] **L28** Sampler & sampling params. Cite `srt/layers/sampler.py ::Sampler` (~68; forward ~93;
  SamplingBatchInfo in sampling/sampling_batch_info.py ~24; SamplingParams sampling/sampling_params.py ~64).
  Analogy: a **weighted dice/lottery** over the vocabulary, with temperature reshaping the odds and top-k/p
  trimming the pool. Diagrams: vflow logits→penalties→temperature→top-k/p→sample; table.t param→effect; cols
  greedy-vs-sampling; cellgroup top-p truncation. Forward-ref structured outputs 第 48 课. Closes Part 6.

## Verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py`
  → 0 err / 0 warn; pill "共 28 课 · 6 个部分"; nav L23↔L24↔…↔L28; no-diff.
- [ ] One commit: `M6: Part 6 model execution — L24 ModelRunner/ForwardBatch, L25 loading, L26 writing a model, L27 CUDA graph, L28 sampler` (+ Co-authored-by trailer).

## Guardrails
- Cite `file ::symbol`; the ModelRunner→model.forward→Sampler chain MUST match model_runner.py. No `timeline`.
- zh ≥3500 CJK; zh/en identical diagram inventory. Don't touch `docs/`, earlier parts, pipeline, reference repo.
