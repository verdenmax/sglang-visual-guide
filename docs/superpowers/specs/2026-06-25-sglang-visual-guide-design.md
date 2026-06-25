# SGLang Visual Guide — Design Spec

**Date:** 2026-06-25
**Status:** Approved (approach + full per-lesson TOC), ready for planning
**Author:** @verdenmax (driven autonomously)

## 1. Purpose

A **visual, bilingual (中文 + English)** learning guide that explains the **internals of
[SGLang](https://github.com/sgl-project/sglang)** — the high-performance serving framework for
large-language and multimodal models. The guide takes a reader from *"what is LLM serving / what is
a KV cache"* all the way to *"how the scheduler, RadixAttention KV cache, model executor and CUDA
kernels work inside the code"* and *"how to build, benchmark, test and contribute a PR"*.

It is modeled **directly** on the sibling projects `~/course/milvus-visual-guide` and
`~/course/hermes-agent-visual-guide`: same zero-dependency Python-generator architecture, same
visual design system, same per-lesson pedagogy (life analogy → macro picture → diagrams → cited
code → worked-example trace → self-test quiz), same bilingual in-page toggle, same GitHub-Pages
delivery, same dual licensing.

### Audience & depth
- Primary: engineers who want to **understand SGLang internals** and eventually contribute, plus
  the author learning the codebase.
- Starts from first principles (autoregressive decoding, the KV cache, continuous batching, paged
  attention) so a motivated newcomer can follow.
- Builds up to deep internals: the zero-overhead scheduler, RadixAttention prefix caching, the
  model executor + CUDA graphs, the attention/MoE/quantization layers, the C++/CUDA kernel layer,
  speculative decoding, PD disaggregation, TP/PP/EP/DP parallelism, structured outputs, and the
  contribution workflow.

### Non-goals
- Not a user manual / API reference (docs.sglang.io already does that). We cite the server CLI and
  OpenAI-compatible API only as the entry point into the internals.
- Not a fork or mirror of SGLang source. We **quote small, cited snippets only** and explain them.
- Not a benchmark report or a "which GPU is fastest" comparison.

## 2. Architecture (tech) — cloned from the reference, retargeted to SGLang

Pure **Python 3, zero runtime dependencies**. Generators under `src/` emit committed static HTML.

```
sglang-visual-guide/
  src/
    part1.py .. part13.py   bilingual lesson content; one module per Part; each exports
                            LESSON_NN = {"zh": <html>, "en": <html>}
    quizzes.py              per-lesson self-test (mcq + open); deterministic option shuffle
    shell.py                CSS design system + PAGES list + bi()/esc()/head_meta()
                            + page() (lesson shell) + index_page() (TOC) + SUBTITLES
    registry.py             ordered filename -> {"zh","en"} content map (imports part1..N)
    build.py                writes index.html + lessons/NN-*.html
    build_print.py          writes print_zh.html + print_en.html (one page per lesson)
    check_html.py           structural + quality gates (CI-blocking on ERROR)
    check_links.py          internal relative-link resolver
  lessons/                  generated lesson pages (committed, must rebuild with no diff)
  index.html                generated TOC (committed)
  print_zh.html print_en.html   generated print/PDF editions (committed)
  docs/superpowers/specs/   design specs (this file + per-milestone specs)
  docs/superpowers/plans/   roadmap + per-milestone implementation plans
  .github/workflows/        ci.yml (build+validate+no-diff), deploy.yml (Pages)
  README.md  LICENSE(MIT)  LICENSE-CONTENT(CC BY 4.0)  .gitignore
```

### Single source of truth & invariants
- `shell.PAGES` is the ordered page list `(filename, title_zh, title_en, part_zh, part_en)`.
  `registry.CONTENT` must have exactly one non-empty bilingual entry per `PAGES` filename.
  `check_html.py` enforces this alignment (no orphan keys, no missing entries).
- Generated HTML is committed and must be **byte-identical** on rebuild (CI `git diff --exit-code`).
- Navigation uses **relative `href`** links so the site works via `file://` or any static host.
- Bilingual contract: every page carries `data-lang="zh"` (default); CSS hides the inactive
  language; a topbar toggle + `localStorage` switch persists the choice.

### Theme (SGLang identity)
- Accent recolored to an **SGLang violet/indigo** (`--accent` ≈ `#7c48e6`), with a matching
  dark-mode variant. Topbar mark uses ⚡ (high-performance serving). Titles: "SGLang 图解教程 /
  SGLang Visual Guide".
- All other design tokens (cards, flow/vflow/layers/cols/cellgroup/timeline/trace, code blocks,
  accordion, TOC search) are carried over unchanged — they are subject-agnostic. The accent is a
  cosmetic choice and may be adjusted later without affecting structure.

## 3. Content model — what every lesson contains

Each lesson is a self-contained bilingual page. Authoring target per lesson:

1. **Lead paragraph** (`.lead`) — one-paragraph hook: what this lesson is about and why it matters.
2. **Life analogy card** (`.card.analogy`, 🔌 类比) — map the concept onto an everyday situation.
3. **Macro card** (`.card.macro`, 🌍) — the big-picture takeaway in 2–3 sentences.
4. **2–5 `<h2>` sections** mixing prose with **≥3 visual blocks per language** chosen from:
   `flow`, `vflow`, `layers`, `cols`, `cellgroup`, `timeline`, `trace`, `table.t`, hand-drawn
   `<svg>` figures (`.fig` + `.figcap`).
5. **Cited code** via `.codefile` — *small* snippets with a `file + symbol` caption (line numbers
   drift, so cite symbols). Never paste large source; explain, don't copy.
6. **Key-points card** (`.card.key`, 本课要点 / Key points) — bulleted recap.
7. **Self-test** (`quizzes.py`) — 2–4 design-insight MCQs (with "why") + 1–2 open questions.

Borrowed from the hermes guide where it fits a serving engine: lessons that introduce a component
may add a 🎯 **design-tradeoff** note (the bottleneck it fights — latency, HBM, CPU overhead,
bandwidth) to reinforce the "life of a request" + "what problem each piece solves" throughline.

### Quality gates (enforced by `check_html.py`, adapted from reference)
- ERROR (CI-blocking): balanced `div/details/table/pre/summary`; `details==summary`; exactly one
  `<h1>`; `<title>` + meta description present; both `lang-zh` and `lang-en` blocks present; no
  unescaped `<` inside `<pre>`; prev/next nav matches `PAGES`; TOC lists every page; the
  "共 N 课 · N 个部分" pill matches `PAGES`; registry has non-empty zh+en per page; no orphan keys.
- WARN (visible, non-blocking, but we drive to zero): per-lesson **≥ 3000 CJK chars** (zh),
  **≥ 6 visual blocks** (counting both languages, i.e. ≥3/lang), an analogy card and a
  key-points card present.
- Cross-reference guard: "第 N 课" references must be within `1..MAX_LESSON`.

## 4. Source-accuracy discipline

SGLang is large and evolves fast (day-0 model support, frequent kernel/scheduler changes). Every
content milestone MUST, before writing:
1. Read the relevant `docs_new/docs/**` pages (developer_guide, advanced_features) for the topic.
2. Read the **actual source** for the components in scope (Key packages below).
3. Cross-check doc vs code; when they disagree, the **code wins** and we note it.

Cited facts use `file + symbol` (e.g. `python/sglang/srt/managers/scheduler.py ::Scheduler.run_loop`),
never bare line numbers. Verified-against-source is stated on the index page.

### Key packages (source map)
- `python/sglang/lang/` — frontend DSL: `api.py`, `interpreter.py`, `tracer.py`, `ir.py`, `backend/`
- `python/sglang/srt/entrypoints/` — `engine.py`, `http_server.py`, `openai/` `anthropic/` `ollama/`, gRPC
- `python/sglang/srt/managers/` — `tokenizer_manager.py`, `scheduler.py`, `schedule_batch.py`,
  `schedule_policy.py`, `detokenizer_manager.py`, `data_parallel_controller.py`, `tp_worker.py`, `io_struct.py`
- `python/sglang/srt/model_executor/` — ModelRunner, ForwardBatch, CUDA-graph runner
- `python/sglang/srt/model_loader/`, `python/sglang/srt/models/` — weight loading + per-model code
- `python/sglang/srt/layers/` — attention backends, MoE, quantization, sampler, rotary/norm, logits
- `python/sglang/srt/mem_cache/` — RadixAttention, req/token paged pools, HiCache; `managers/cache_controller.py`
- `python/sglang/srt/speculative/` — EAGLE / draft-verify; `disaggregation/` — PD; `distributed/` — TP/PP/EP/DP
- `python/sglang/srt/eplb/`, `constrained/`, `lora/`, `sampling/`, `multimodal/`
- `python/sglang/multimodal_gen/` — diffusion; `srt/weight_sync/`, `srt/checkpoint_engine/` — RL rollout
- `python/sglang/jit_kernel/` + top-level `sgl-kernel/` — JIT + AOT C++/CUDA kernels
- `python/sglang/srt/platforms/`, `srt/hardware_backend/` — multi-hardware; `srt/server_args.py` — config
- `test/`, `python/sglang/bench_*`, `python/sglang/profiler.py` — tests, benchmarks, profiling

## 5. Licensing & disclaimer

- **Code** (`src/` generators + validators) — **MIT**, `LICENSE`.
- **Content** (lesson prose + diagrams authored in `src/part*.py` / `src/quizzes.py` and rendered
  into the HTML) — **CC BY 4.0**, `LICENSE-CONTENT`.
- **Disclaimer** (README + index): third-party, **unofficial** educational material *about* SGLang;
  contains **no SGLang source code** beyond small, cited snippets; **SGLang itself is Apache-2.0**
  licensed by its authors (matches Milvus; phrasing carried over).

## 6. Curriculum — 13 parts, ~63 lessons

The guide follows a **"life of a request"** throughline (foundations → frontend DSL → entrypoints →
scheduler → model execution → KV cache → layers → kernels → performance innovations → advanced →
practice → design synthesis). Final per-part counts may shift by ±1 as research firms up each
milestone; `MAX_LESSON` in `check_html.py` is updated as parts land. Trimmable targets are noted.

| Part | Title (zh / en) | Lessons |
| --- | --- | --- |
| 1 | 宏观全景 / Overview | L01–03 |
| 2 | 推理前置基础 / Foundations of LLM inference | L04–08 |
| 3 | 前端语言 DSL / The frontend language | L09–12 |
| 4 | 服务入口与编排 / Entrypoints & orchestration | L13–17 |
| 5 | 调度器（心脏）/ The scheduler | L18–23 |
| 6 | 模型执行 / Model execution | L24–28 |
| 7 | KV 缓存与内存 / KV cache & memory | L29–32 |
| 8 | Attention 与算子层 / Attention & layers | L33–37 |
| 9 | 内核与硬件（深入）/ Kernels & hardware (deep) | L38–42 |
| 10 | 性能创新专题 / Performance innovations | L43–48 |
| 11 | 进阶·选读 / Advanced topics (optional) | L49–52 |
| 12 | 实战与贡献 / Practice & contributing | L53–57 |
| 13 | 设计专题·综合 / Design themes (synthesis) | L58–63 |

Full per-lesson enumeration (approved; will be carried verbatim into the roadmap):

- **P1 Overview** — L01 What is SGLang · L02 The project map (`lang/`+`srt/`+`sgl-kernel`) · L03 Life of a request (HTTP→TokenizerManager→Scheduler→ModelRunner→Sampler→Detokenizer)
- **P2 Foundations** — L04 Autoregressive decode & the KV cache (prefill vs decode) · L05 Continuous batching · L06 PagedAttention & paged KV · L07 RadixAttention & prefix caching (signature) · L08 Throughput vs latency (TTFT/ITL, the scheduling tension)
- **P3 Frontend DSL** — L09 The structured-generation language (`gen`/`select`/`fork`/`join`) · L10 Interpreter & tracer (`interpreter.py`/`tracer.py`/`ir.py`) · L11 Why it fits prefix caching (`fork` shares prefixes) · L12 Backends & OpenAI compat (`lang/backend/`, `api.py`)
- **P4 Entrypoints** — L13 Engine & HTTP server (`engine.py`/`http_server.py`) · L14 TokenizerManager (tokenize, ZMQ enqueue) · L15 OpenAI/Anthropic/Ollama compat layer · L16 IO structs & IPC (`io_struct.py`) · L17 Detokenizer & streaming (incremental decode, stop, SSE)
- **P5 Scheduler** — L18 Event-loop overview (`scheduler.py`) · L19 Req & ScheduleBatch (`schedule_batch.py`) · L20 Schedule policy (`schedule_policy.py`, LPM/priority/fairness) · L21 Zero-overhead overlap scheduler (`batch_overlap`) · L22 Chunked prefill (`prefill_delayer`) · L23 DP controller & PP scheduling (`data_parallel_controller.py`/`scheduler_pp_mixin.py`)
- **P6 Model execution** — L24 ModelRunner & ForwardBatch (`model_executor/`) · L25 Model loading & weights (`model_loader/`) · L26 Writing a model (`models/`, Llama/Qwen) · L27 CUDA graph capture & replay · L28 Sampler & sampling params (`sampling/`)
- **P7 KV cache & memory** — L29 RadixAttention implementation (`mem_cache/` radix tree) · L30 Paged memory pools (req-to-token / token-to-KV) · L31 HiCache tiering (GPU/CPU/disk, `cache_controller.py`) · L32 Eviction & hit rate (LRU, refcount)
- **P8 Attention & layers** — L33 Attention backend abstraction (FlashInfer/Triton/FA) · L34 MoE layer (routing, grouped GEMM) · L35 Quantization (FP8/FP4/INT4/AWQ/GPTQ) · L36 RoPE/Norm/other ops · L37 Logits processing & vocab parallel
- **P9 Kernels & hardware (deep)** — L38 sgl-kernel overview (AOT C++/CUDA build+binding) · L39 JIT kernels (`jit_kernel/`) · L40 Key attention kernel dissection (memory layout/tiling) · L41 Operator fusion & CUDA-graph cooperation · L42 Multi-hardware backends (`platforms/`/`hardware_backend/`, ROCm/TPU/NPU/CPU)
- **P10 Performance innovations** — L43 Speculative decoding overview (draft/verify, accept rate) · L44 EAGLE & next-gen (DFlash/Spec V2) · L45 PD disaggregation (`disaggregation/`, KV transfer) · L46 TP/PP/EP/DP parallelism (`distributed/`) · L47 Large-scale EP & EPLB (`eplb/`) · L48 Structured outputs & compressed FSM (`constrained/`, jump-forward)
- **P11 Advanced (optional)** — L49 Multimodal VLM serving (`multimodal/`) · L50 Multi-LoRA batching (`lora/`) · L51 RL rollout backend (`weight_sync/`/`checkpoint_engine/`) · L52 Diffusion models (`multimodal_gen/`)
- **P12 Practice & contributing** — L53 Build & run (`launch_server`/`server_args`) · L54 Benchmark & profiling (`bench_serving`/`profiler`) · L55 Test suite & CI (`test/`, CustomTestCase) · L56 Code conventions & opening a PR · L57 Glossary
- **P13 Design themes (synthesis)** — L58 RadixAttention as a first-class idea · L59 Zero-overhead scheduling · L60 PD disaggregation (two workloads) · L61 Speculative decoding (draft-for-parallel-verify) · L62 Everything pluggable (attention/hardware/quant/parallel) · L63 Built for throughput (batching+paging+caching)

## 7. Risks & mitigations

- **Scope/accuracy drift** — SGLang internals are deep and version-sensitive (kernels, scheduler,
  spec decoding change often). Mitigation: per-part source reading + code-wins cross-check +
  symbol-based citations; pin facts to a recorded commit per milestone.
- **Content volume** — ~63 rich bilingual lessons. Mitigation: milestone-by-milestone delivery;
  heavy lesson drafting delegated to subagents with a strict format brief + the milestone spec,
  then two-stage review (spec-compliance + quality) and validator gates before commit.
- **Kernel/C++ depth** — Part 9 reaches into CUDA/C++; risk of over-claiming. Mitigation: explain
  intent + memory layout from cited headers/signatures; never reproduce kernel bodies wholesale.
- **Pipeline regressions** — Mitigation: validators + no-diff rebuild run after every milestone;
  CI mirrors them.

## 8. Definition of done (whole project)

All 13 milestones complete; `build.py`/`build_print.py` regenerate with **no diff**;
`check_html.py` + `check_links.py` pass with **0 errors and 0 warnings**; CI + deploy workflows
green; README documents build/validate/print; dual licenses present; the index page states
"verified against source" with the pinned reference commit.
