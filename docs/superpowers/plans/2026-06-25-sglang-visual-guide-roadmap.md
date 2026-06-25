# SGLang Visual Guide — Roadmap (milestones)

Companion to `specs/2026-06-25-sglang-visual-guide-design.md`. Each milestone ships its own
**spec** (`specs/…`, when the part is non-trivial) and **plan** (`plans/…`), is fully built +
validated + reviewed, then committed. Lesson titles below are the zh / en page titles; `NN` =
global lesson number.

**Per-milestone Definition of Done:** lessons authored to the content model; `shell.PAGES`,
`shell.SUBTITLES`, `registry.CONTENT`, `quizzes.QUIZZES` updated; `build.py` + `build_print.py` +
`check_html.py` + `check_links.py` pass (0 errors, 0 warnings); rebuild produces no diff;
two-stage review passed (spec-compliance subagent + code-quality subagent); committed with sign-off.

**Source-accuracy:** before authoring each part, read the Key packages listed and the relevant
`docs_new/docs/**` pages; cite `file + symbol` (never bare line numbers); code wins over docs.
Reference commit is pinned per milestone in its plan.

---

## M0 · Scaffold & pipeline
**Deliverable:** the full Python pipeline + project chrome, proving the toolchain end-to-end by
shipping **L01** as the baseline lesson. No content beyond L01.
- `src/shell.py` (SGLang-themed CSS, PAGES seeded with L01, page/index_page, bi/esc/head_meta, SUBTITLES).
- `src/registry.py`, `src/part1.py` (L01 only), `src/quizzes.py` (L01 quiz + validator).
- `src/build.py`, `src/build_print.py`, `src/check_html.py`, `src/check_links.py`.
- `.github/workflows/ci.yml` + `deploy.yml`; `README.md`; `LICENSE`; `LICENSE-CONTENT`; `.gitignore`.
- `check_html.py` `MAX_LESSON` starts small and grows per part.
**Key reading:** SGLang `README.md`, `docs_new/docs/get-started/**`; the reference `src/` (format,
copied near-verbatim then re-themed).

## M1 · Part 1 — 宏观全景 / Overview (L02–03)
- **L02** 项目全景地图 / The project map — `lang/` (frontend DSL) vs `srt/` (runtime) vs `sgl-kernel`;
  the four process roles (TokenizerManager / Scheduler+TpWorker / DetokenizerManager); repo layout.
- **L03** 一次请求的一生 / Life of a request — generate request end-to-end at 10,000 ft:
  HTTP → TokenizerManager → (ZMQ) → Scheduler → ModelRunner → Sampler → DetokenizerManager → SSE.
**Key packages:** `python/sglang/srt/entrypoints/{engine.py,http_server.py}`,
`srt/managers/{tokenizer_manager.py,scheduler.py,detokenizer_manager.py,io_struct.py}`;
docs `get-started/quick_start`, `basic_usage/send_request`.

## M2 · Part 2 — 推理前置基础 / Foundations (L04–08)
- **L04** 自回归生成与 KV 缓存 / Autoregressive decode & the KV cache — prefill vs decode, why KV cache.
- **L05** 连续批处理 / Continuous batching — static-batch waste; request-level dynamic batching.
- **L06** PagedAttention 与分页 KV / PagedAttention & paged KV — HBM fragmentation; page/block model.
- **L07** RadixAttention 与前缀缓存 / RadixAttention & prefix caching — the radix tree, automatic reuse (signature).
- **L08** 吞吐 vs 延迟 / Throughput vs latency — TTFT/ITL, the scheduling tension, SLA framing.
**Key packages:** `srt/mem_cache/` (radix cache, pools), `srt/managers/schedule_batch.py`,
`srt/layers/attention/` (concepts only); docs `basic_usage/sampling_params`, blog references (cited as prose).

## M3 · Part 3 — 前端语言 DSL / The frontend language (L09–12)
- **L09** 结构化生成语言 / The structured-generation language — `gen`/`select`/`fork`/`join`/`image` primitives.
- **L10** 解释器与 tracer / Interpreter & tracer — how a program executes vs compiles.
- **L11** 为何契合前缀缓存 / Why it fits prefix caching — `fork` shares prefixes; parallel generation.
- **L12** 后端与 OpenAI 兼容 / Backends & OpenAI compat — local runtime vs OpenAI vs other backends.
**Key packages:** `python/sglang/lang/{api.py,interpreter.py,tracer.py,ir.py,choices.py}`,
`python/sglang/lang/backend/`; docs `references/frontend/**`.

## M4 · Part 4 — 服务入口与编排 / Entrypoints & orchestration (L13–17)
- **L13** Engine 与 HTTP Server / Engine & HTTP server — offline `Engine` vs online server; `EngineBase`.
- **L14** TokenizerManager — tokenize, build sampling params, ZMQ hand-off into the scheduler.
- **L15** OpenAI/Anthropic/Ollama 兼容层 / Compat layer — protocol adapters in `entrypoints/`.
- **L16** IO 结构与进程间通信 / IO structs & IPC — how `Req` travels across tokenizer/scheduler/detokenizer.
- **L17** Detokenizer 与流式输出 / Detokenizer & streaming — incremental decode, stop conditions, SSE.
**Key packages:** `srt/entrypoints/{engine.py,http_server.py,EngineBase.py,openai/,anthropic/,ollama/}`,
`srt/managers/{tokenizer_manager.py,detokenizer_manager.py,io_struct.py}`; docs `basic_usage/{openai_api,offline_engine_api}`.

## M5 · Part 5 — 调度器（心脏）/ The scheduler (L18–23)
- **L18** 事件循环总览 / Event-loop overview — `Scheduler.event_loop_*`: recv → schedule → forward → output.
- **L19** Req 与 ScheduleBatch / Req & ScheduleBatch — the request state machine, batch assembly, mode (prefill/decode).
- **L20** 调度策略 / Schedule policy — LPM / priority / fairness; prefill-vs-decode admission, waiting queue.
- **L21** 零开销重叠调度 / Zero-overhead overlap scheduler — overlap CPU schedule with GPU compute.
- **L22** Chunked prefill — splitting long prompts; mixing prefill/decode; the prefill delayer.
- **L23** DP 控制器与 PP 调度 / DP controller & PP scheduling — multi-process orchestration.
**Key packages:** `srt/managers/{scheduler.py,schedule_batch.py,schedule_policy.py,
data_parallel_controller.py,scheduler_pp_mixin.py,prefill_delayer.py,tp_worker.py}`,
`srt/batch_overlap/`, `srt/managers/overlap_utils.py`; docs `advanced_features/{pipeline_parallelism,hyperparameter_tuning}`.

## M6 · Part 6 — 模型执行 / Model execution (L24–28)
- **L24** ModelRunner 与 ForwardBatch / ModelRunner & ForwardBatch — how one forward pass is organized.
- **L25** 模型加载与权重 / Model loading & weights — weight mapping, sharded + quantized loading.
- **L26** 写一个模型 / Writing a model — `models/llama.py` / `qwen*.py`: layer assembly, runtime hooks.
- **L27** CUDA Graph 捕获与重放 / CUDA graph capture & replay — why, capture, padding, breakable graphs.
- **L28** 采样器与采样参数 / Sampler & sampling params — logits processing, top-k/p, penalties, structured hook.
**Key packages:** `srt/model_executor/{model_runner.py,forward_batch_info.py,cuda_graph_runner.py}`,
`srt/model_loader/`, `srt/models/{llama.py,qwen2.py}`, `srt/sampling/`; docs `advanced_features/{model_loading,piecewise_cuda_graph,breakable_cuda_graph}`.

## M7 · Part 7 — KV 缓存与内存 / KV cache & memory (L29–32)
- **L29** RadixAttention 实现 / RadixAttention implementation — radix-tree nodes, prefix match, insert/split.
- **L30** 分页内存池 / Paged memory pools — `req_to_token` + `token_to_kv` pools; alloc/free.
- **L31** HiCache 分层缓存 / HiCache tiering — GPU/CPU/disk three-tier; the cache controller.
- **L32** 缓存淘汰与命中 / Eviction & hit rate — LRU, refcount, hit-rate vs recompute tradeoff.
**Key packages:** `srt/mem_cache/{radix_cache.py,memory_pool.py,...}`, `srt/managers/cache_controller.py`;
docs `advanced_features/{hicache,hicache_design,hicache_best_practices}`.

## M8 · Part 8 — Attention 与算子层 / Attention & layers (L33–37)
- **L33** Attention 后端抽象 / Attention backend abstraction — FlashInfer / Triton / FlashAttention selection.
- **L34** MoE 层 / The MoE layer — expert routing, grouped GEMM, fused kernels.
- **L35** 量化 / Quantization — FP8/FP4/INT4/AWQ/GPTQ: weight/activation quant + kernels.
- **L36** RoPE/Norm/其它算子 / RoPE, norm & other ops — rotary embedding, normalization specifics.
- **L37** Logits 处理与词表并行 / Logits processing & vocab parallel — vocab-parallel, logits all-gather.
**Key packages:** `srt/layers/{attention/,moe/,quantization/,rotary_embedding.py,layernorm.py,
logits_processor.py,vocab_parallel_embedding.py}`; docs `advanced_features/{attention_backend,quantization}`.

## M9 · Part 9 — 内核与硬件（深入）/ Kernels & hardware, deep (L38–42)
- **L38** sgl-kernel 总览 / sgl-kernel overview — AOT C++/CUDA kernels: build system, pybind surface.
- **L39** JIT kernel / JIT kernels — the lightweight just-in-time kernel path and how to add one.
- **L40** 关键 attention kernel 剖析 / Key attention kernel dissection — memory layout, tiling, paged-KV access.
- **L41** 算子融合与 CUDA Graph 配合 / Operator fusion & CUDA-graph cooperation — fusion points, graph constraints.
- **L42** 多硬件后端 / Multi-hardware backends — ROCm / TPU-Jax / NPU / CPU dispatch.
**Key packages:** top-level `sgl-kernel/` (csrc + python binding), `python/sglang/jit_kernel/`,
`srt/layers/attention/` (backend glue), `srt/platforms/`, `srt/hardware_backend/`;
docs `developer_guide/development_jit_kernel_guide`, `hardware-platforms/**`.

## M10 · Part 10 — 性能创新专题 / Performance innovations (L43–48)
- **L43** 投机解码总览 / Speculative decoding overview — draft/verify, acceptance rate, the speedup math.
- **L44** EAGLE 与下一代 / EAGLE & next-gen — draft model, tree verification, DFlash / Spec V2.
- **L45** PD 分离 / PD disaggregation — prefill and decode as two workloads; KV transfer between them.
- **L46** TP/PP/EP/DP 并行 / TP/PP/EP/DP parallelism — partition schemes, communication primitives.
- **L47** 大规模专家并行与 EPLB / Large-scale EP & EPLB — expert load balancing & placement.
- **L48** 结构化输出与压缩 FSM / Structured outputs & compressed FSM — grammar, jump-forward decoding.
**Key packages:** `srt/speculative/`, `srt/disaggregation/`, `srt/distributed/`, `srt/eplb/`,
`srt/constrained/`; docs `advanced_features/{speculative_decoding,adaptive_speculative_decoding,
pd_disaggregation,expert_parallelism,structured_outputs}`.

## M11 · Part 11 — 进阶·选读 / Advanced topics, optional (L49–52)
- **L49** 多模态 VLM 服务 / Multimodal VLM serving — image/video encoders, multimodal batching.
- **L50** 多 LoRA 批处理 / Multi-LoRA batching — adapter switching within a batch.
- **L51** RL rollout 后端 / RL rollout backend — weight sync, checkpoint engine, use as a training rollout.
- **L52** 扩散模型 / Diffusion models — SGLang Diffusion (WAN / Qwen-Image).
**Key packages:** `srt/multimodal/`, `srt/managers/multimodal_processor.py`, `srt/lora/`,
`srt/weight_sync/`, `srt/checkpoint_engine/`, `python/sglang/multimodal_gen/`;
docs `advanced_features/{vlm_query,lora,sglang_for_rl,checkpoint_engine}`, `sglang-diffusion/**`.

## M12 · Part 12 — 实战与贡献 / Practice & contributing (L53–57)
- **L53** 本地构建与运行 / Build & run — install, `launch_server`, key `server_args` flags.
- **L54** 基准与性能剖析 / Benchmark & profiling — `bench_serving`, `bench_one_batch`, the profiler.
- **L55** 测试体系与 CI / Test suite & CI — `test/`, `CustomTestCase`, the staged CI layout.
- **L56** 代码约定与提 PR / Code conventions & opening a PR — env-var conventions, naming, the flow.
- **L57** 术语表 / Glossary — KV / TTFT / PD / EP / MLA / spec… one-line glossary + jump links.
**Key packages:** `python/sglang/{launch_server.py,bench_serving.py,bench_one_batch.py,profiler.py}`,
`srt/server_args.py`, `srt/environ.py`, `test/`; docs `developer_guide/{contribution_guide,bench_serving,benchmark_and_profiling}`. (L57 is `SOFT_EXEMPT` from CJK/diagram floors.)

## M13 · Part 13 — 设计专题·综合 / Design themes, synthesis (L58–63)
- **L58** RadixAttention：前缀复用做成一等公民 / RadixAttention as a first-class idea.
- **L59** 零开销调度：把 CPU 调度移出关键路径 / Zero-overhead scheduling.
- **L60** PD 分离：承认两类负载 / PD disaggregation as two workloads.
- **L61** 投机解码：小模型草稿换大模型并行验证 / Speculative decoding as draft-for-parallel-verify.
- **L62** 一切皆可插拔 / Everything pluggable — attention/hardware/quant/parallel as strategies.
- **L63** 为吞吐而生 / Built for throughput — batching + paging + caching working together.
**Key reading:** synthesis of earlier parts — no new packages; each lesson back-references the
lessons where the mechanism was introduced (forward-ref-safe per `check_html.py`).

## M14 · Polish
- Print/PDF editions verified (one page per lesson, details expanded); CI `print-pdf` job.
- `check_html.py` `MAX_LESSON` finalized; 0 warnings across all lessons.
- README finalized (badges, build/validate/print, structure, license, 中文说明).
- Optional: concept dependency graph on the glossary page; final no-diff + link sweep; pin the
  "verified against source" reference commit on the index page.

---

## Build order rationale
M0 proves the toolchain with one lesson before any bulk content. Parts then follow the natural
**"life of a request"** arc: understand it (1) → the foundations a request rides on (2) → the
frontend that issues it (3) → how it enters the server (4) → the scheduler that drives it (5) →
the model forward it triggers (6) → the KV cache/memory it reads & grows (7) → the attention/op
layers underneath (8) → the C++/CUDA kernels at the bottom (9) → the performance innovations that
reshape the path (10) → advanced modalities (11) → operate & contribute (12) → synthesize the
design themes (13) → polish (14). Each part depends only on the scaffold (M0) and, loosely, on
concepts introduced earlier — so cross-references always point backward or are forward-ref-safe.
