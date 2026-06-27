# Visual & Code Enrichment Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add ~2 hand-drawn SVG `.fig` figures per language, a 2nd cited `.codefile`, and more inline
examples to all 63 lessons of the SGLang visual guide (add-only), then lock it in with a `MIN_FIGURES`
validator floor.

**Architecture:** roll out part-by-part (E1 Part 1 … E13 Part 13), one subagent per lesson, each using
the shared SVG recipe below. The orchestrator pre-fetches/verifies the 2nd-codefile symbol per lesson
and picks 2 figure subjects; each part gets the two-stage review. The `check_html.py` figure floor is
flipped on LAST (E-final), so the build stays green throughout the rollout.

**Tech stack:** Python 3 (zero-dep) generators (`src/partN.py` + `shell.py` CSS + `check_html.py`);
hand-written inline SVG themed via CSS variables.

**Companion spec:** `docs/superpowers/specs/2026-06-27-visual-code-enrichment-design.md`.

---

## Shared SVG recipe (the reusable core — every lesson subagent uses this)

### The `.fig` markup (one figure)
```html
<div class="fig">
  <svg viewBox="0 0 W H" role="img" aria-label="<one-sentence localized description>">
    <!-- shapes + <text>, color ONLY via CSS vars -->
  </svg>
  <div class="figcap"><b>图 N · 标题</b> — 一句话说明。</div>   <!-- en: <b>Fig N · Title</b> — one line. -->
</div>
```
- `viewBox="0 0 W H"`: typical `W` 660–820, `H` 200–360. NO `width`/`height` attrs (responsive via CSS).
- `.fig svg text` already defaults to `fill: var(--ink)` — only set `fill` for emphasis.
- Use `class="mono"` on `<text>` for code/identifiers.

### Theming palette (CSS vars ONLY — never hardcode `#hex`)
- lines/neutral: `var(--line)`, `var(--muted)`, `var(--faint)`, `var(--panel-2)`
- primary emphasis: `var(--accent)` (stroke), `var(--accent-soft)` (fill), `var(--accent-ink)` (text)
- semantic color-coding (sparingly): `var(--blue)/--blue-soft`, `var(--amber)/--amber-soft`,
  `var(--purple)/--purple-soft`, `var(--teal)/--teal-soft`, `var(--red)/--red-soft`
- never set an svg/rect background that fills the whole canvas (the `.fig` panel provides bg).

### Worked example (theme-correct — copy this style)
```html
<div class="fig">
  <svg viewBox="0 0 720 230" role="img" aria-label="分页 KV：逻辑上连续的 KV 被切成固定大小的页，物理上分散存放">
    <text x="20" y="30" style="font-weight:700;fill:var(--accent-ink)">逻辑序列（连续）</text>
    <rect x="20" y="44" width="480" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="260" y="66" text-anchor="middle">一条请求的 KV（token 0 → N）</text>
    <text x="20" y="120" style="font-weight:700;fill:var(--accent-ink)">物理显存（分页）</text>
    <rect x="20"  y="134" width="90" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="140" y="134" width="90" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="380" y="134" width="90" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="500" y="134" width="90" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="65"  y="156" text-anchor="middle" class="mono">page 7</text>
    <text x="185" y="156" text-anchor="middle" class="mono">page 2</text>
    <text x="425" y="156" text-anchor="middle" class="mono">page 9</text>
    <text x="545" y="156" text-anchor="middle" class="mono" style="fill:var(--muted)">free</text>
    <text x="20" y="205" style="fill:var(--muted)">页表把逻辑顺序映射到分散的物理页 → 无碎片、可共享</text>
  </svg>
  <div class="figcap"><b>图 1 · 分页 KV</b> — 逻辑上连续的一条 KV 被切成固定大小的页，物理可分散；页表负责映射。</div>
</div>
```

### Per-lesson subagent prompt skeleton (orchestrator fills `<...>`)
> Edit `src/partN.py` ONLY for `LESSON_XX`, plus `quizzes.py` is NOT touched. ADD (do not remove/alter
> existing blocks):
> - In the zh block: insert **2** `<div class="fig">…</div>` figures (Chinese labels) at natural points
>   (after a relevant `<h2>`'s prose). Subjects: `<subject A>` and `<subject B>` (from the spec list).
> - In the en block: insert the SAME 2 figures with identical geometry/coords, English `<text>` +
>   `aria-label` + figcap.
> - Insert **1 more** `.codefile` citing `<verified file ::symbol>` (different from the existing one),
>   condensed faithfully (docstrings→`#`, escape `<`/`>`/`&`), in BOTH languages.
> - Add **1–2 inline concrete examples** (a real number or a `<span class="mono">…</span>` snippet) to
>   existing prose, both languages.
> Use ONLY CSS vars for SVG colors (palette above); NO `#hex`, NO `<img>`, NO JS. Keep zh/en `.fig`,
> `.codefile` counts equal. VERIFY commands below must pass.

### Per-lesson VERIFY (subagent runs; all must pass)
```bash
cd /home/verden/course/sglang-visual-guide/src && python3 build.py >/dev/null && python3 check_html.py && python3 check_links.py
python3 - <<'PY'
import re
s=open('partN.py',encoding='utf-8').read()   # N = the part
m=re.search(r'LESSON_XX = \{"zh": r"""(.*?)""", "en": r"""(.*?)"""\}', s, re.S)
zh,en=m.group(1),m.group(2)
print("fig zh/en:", zh.count('class="fig"'), en.count('class="fig"'))      # want 2 / 2
print("codefile zh/en:", zh.count('class="codefile"'), en.count('class="codefile"'))  # want 2 / 2
print("hardcoded hex in svg (want 0):", len(re.findall(r'(?:fill|stroke)\s*:\s*#', zh+en)))
print("zh CJK:", len(re.findall(r'[\u4e00-\u9fff]', zh)))
PY
```
Required: validators no ERROR; `fig zh==en==2`; `codefile zh==en==2`; hardcoded hex == 0; zh CJK still ≥3500.

---

## Task E0 — verify the recipe on ONE lesson (de-risk SVG truncation)

**Files:** `src/part1.py` (LESSON_01)

- [ ] **Step 1:** Dispatch ONE subagent for L01 using the recipe (subjects below). Confirm an SVG
  subagent can produce 2 themed figures + a 2nd codefile without truncation.
- [ ] **Step 2:** Run the per-lesson VERIFY. Expected: `fig 2/2`, `codefile 2/2`, `hex 0`, validators
  no ERROR. If it truncates 3×, use the draft+expand fallback (orchestrator writes the 2 SVG skeletons,
  subagent localizes labels + adds the codefile/examples).
- [ ] **Step 3:** Eyeball `lessons/01-what-is-sglang.html` rendered (figures present, captions read,
  no raw `#hex`). This locks the quality bar for the rest.

## Task E1 — Part 1 · Overview (L01–03)

**Files:** `src/part1.py`

Per-lesson inputs (orchestrator: verify each `::symbol` with `grep` before dispatch):
| Lesson | existing codefile | ADD 2nd codefile (verified real) | Figure A | Figure B |
| --- | --- | --- | --- | --- |
| L01 what-is-sglang | `engine.py ::Engine` | `python/sglang/srt/entrypoints/engine.py ::Engine.generate` (offline API, line 317) | serving engine vs raw `model.generate()` (what the engine adds: batching/KV/scheduler) | one prompt → many tokens streamed (autoregressive loop, real token count) |
| L02 project-map | `engine.py ::_launch_subprocesses` | `python/sglang/srt/server_args.py ::ServerArgs` (the config map, line 374) | the `python/sglang/{lang,srt}` directory map | process topology: TokenizerManager ↔ Scheduler ↔ Detokenizer over IPC |
| L03 life-of-a-request | `scheduler.py ::event_loop_normal` | `python/sglang/srt/managers/tokenizer_manager.py ::generate_request` (request entry, line 588) | the request journey timeline (HTTP→tokenize→schedule→forward→detokenize→stream) | prefill (1 big pass) vs decode (token-by-token) over time |

- [ ] **Step 1:** For each of L01–03, dispatch a subagent (recipe skeleton + that row's inputs). One
  lesson per agent, sync, claude-opus-4.8, high effort. Retry on truncation; draft+expand after 3×.
- [ ] **Step 2:** After each lands, run the per-lesson VERIFY (fig 2/2, codefile 2/2, hex 0, validators
  clean).
- [ ] **Step 3:** Full part verify:
  `cd src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py`
  → 0 err (figure floor not yet enforced; MIN_DIAGRAMS still passes); `git diff` no-diff after rebuild.
- [ ] **Step 4:** Commit:
```bash
git add -A && git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "feat(e1): enrich Part 1 (L01-03) with SVG figures + 2nd codefile + inline examples

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
- [ ] **Step 5:** Two-stage review (research concept + code-review, opus-4.8 max, long context), with
  the EXTRA checks: (a) every SVG uses ONLY CSS vars — `grep -nE '(fill|stroke):\s*#' part1.py` returns
  nothing; (b) each 2nd codefile cites a real symbol (open the source); (c) zh/en figure geometry
  matches (same coords, translated text); (d) figures are illustrative (match their figcap), not
  decorative. Fix findings → re-verify → commit.

## Tasks E2–E13 — Parts 2–13 (same recipe, one task per part)

Each part repeats E1's Steps 1–5 (subagent per lesson → per-lesson VERIFY → full part verify →
commit `feat(eN): enrich Part N …` → two-stage review → fix). Orchestrator VERIFIES every `::symbol`
with `grep` before dispatch and may swap a candidate that doesn't resolve. Per-lesson inputs:

### E2 — Part 2 · Foundations (L04–08) · `src/part2.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L04 autoregressive-and-kv-cache | `srt/mem_cache/memory_pool.py ::MHATokenToKVPool` | KV memory grows linearly with tokens (chart: MB vs seq-len) | with-cache vs no-cache compute (recompute triangle each step) |
| L05 continuous-batching | `srt/managers/schedule_batch.py ::ScheduleBatch` | static batching idle gaps vs continuous fill (GPU-util timeline) | requests join/leave the running batch across steps |
| L06 paged-attention-and-paged-kv | `srt/mem_cache/memory_pool.py ::ReqToTokenPool` | contiguous fragmentation vs paged blocks (the worked example) | page table mapping logical positions → scattered physical pages |
| L07 radixattention-and-prefix-caching | `srt/mem_cache/radix_cache.py ::TreeNode` | a radix tree of shared-prefix nodes (two requests share a path) | compute saved on a cache hit (only the new suffix is computed) |
| L08 throughput-vs-latency | `python/sglang/benchmark/serving.py ::BenchmarkMetrics` | throughput-vs-latency trade-off curve (knee of the curve) | TTFT vs TPOT marked on one request's token timeline |

### E3 — Part 3 · Frontend DSL (L09–12) · `src/part3.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L09 structured-generation-language | `python/sglang/lang/ir.py ::SglGen` | a DSL program as a dependency graph (gen/select nodes + edges) | DSL vs raw loop (what the runtime schedules for you) |
| L10 interpreter-and-tracer | `python/sglang/lang/interpreter.py ::StreamExecutor` | trace = static expansion of the program before running | interpreter executing nodes, forks running in parallel |
| L11 fork-join-and-prefix-sharing | `python/sglang/lang/interpreter.py ::ProgramState` | fork tree: branches sharing one common prefix (KV reused) | join merging branch results back |
| L12 backends-and-openai-compat | `python/sglang/lang/backend/runtime_endpoint.py ::RuntimeEndpoint` | one DSL program → local runtime vs remote OpenAI backend | the backend interface (same program, swappable target) |

### E4 — Part 4 · Entrypoints (L13–17) · `src/part4.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L13 engine-and-http-server | `srt/entrypoints/http_server.py ::generate_request` | Engine (in-process) vs HTTP server (online) topology | request → router → worker fan-out |
| L14 tokenizer-manager | `srt/managers/io_struct.py ::GenerateReqInput` | text → token ids (tokenize) with a worked short example | TokenizerManager process talking to Scheduler over ZMQ |
| L15 openai-anthropic-ollama-compat | `srt/entrypoints/openai/serving_chat.py ::OpenAIServingChat` | 3 protocols → one internal request (adapter funnel) | chat-template applied to messages (before → after tokens) |
| L16 io-structs-and-ipc | `srt/managers/io_struct.py ::BatchTokenIDOutput` | the IPC message types flowing between the 3 processes | a request/response round-trip over ZMQ sockets |
| L17 detokenizer-and-streaming | `srt/managers/detokenizer_manager.py ::DetokenizerManager` | incremental detokenize: tokens → text deltas streamed | streaming SSE frames over time (token N arrives at t_N) |

### E5 — Part 5 · Scheduler (L18–23) · `src/part5.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L18 scheduler-event-loop | `srt/managers/scheduler.py ::Scheduler.run_batch` | the event loop cycle (recv→batch→forward→emit) as a wheel | queue → running batch → finished, per iteration |
| L19 req-and-schedule-batch | `srt/managers/schedule_batch.py ::Req` | a Req's lifecycle states (waiting→running→finished) | ScheduleBatch packing several Reqs for one forward |
| L20 schedule-policy | `srt/managers/schedule_policy.py ::CacheAwarePolicy` | LPM policy ordering the queue by prefix-hit | priority/fairness vs cache-aware selection |
| L21 zero-overhead-overlap-scheduler | `srt/managers/scheduler.py ::Scheduler.event_loop_overlap` | serial CPU↔GPU gaps vs overlapped (CPU N+1 ‖ GPU N) | the overlap pipeline: step N forward while N+1 is scheduled |
| L22 chunked-prefill | `srt/managers/schedule_batch.py ::ScheduleBatch` | one long prefill split into chunks interleaved with decode | TTFT with vs without chunking (timeline) |
| L23 dp-controller-and-pp-scheduling | `srt/managers/data_parallel_controller.py ::DataParallelController` | DP: requests split across replicas | PP: layers as stages, micro-batches flowing through |

### E6 — Part 6 · Model execution (L24–28) · `src/part6.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L24 model-runner-and-forward-batch | `srt/model_executor/forward_batch_info.py ::ForwardBatch` | ModelRunner: inputs → model → logits, one forward | ForwardBatch unifying prefill+decode requests into one batch |
| L25 model-loading-and-weights | `srt/model_loader/loader.py ::DefaultModelLoader` | weights sharded across TP ranks (one matrix split) | load flow: disk shards → dtype/quant → each rank's VRAM |
| L26 writing-a-model | `srt/models/llama.py ::LlamaForCausalLM` | a decoder model's layer stack (embed→N×block→lm_head) | forward() delegating attention to `self.attn` (the seam) |
| L27 cuda-graph-capture-and-replay | `srt/model_executor/runner/base_cuda_graph_runner.py ::BaseCudaGraphRunner` | capture once → replay many (static graph) | per-step launch overhead with vs without a graph (timeline) |
| L28 sampler-and-sampling-params | `srt/sampling/sampling_params.py ::SamplingParams` | logits → temperature → top-k → top-p → token (funnel) | top-p nucleus cut on a sorted probability distribution |

### E7 — Part 7 · KV cache & memory (L29–32) · `src/part7.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L29 radixattention-implementation | `srt/mem_cache/radix_cache.py ::RadixCache.match_prefix` | match_prefix walking the radix tree to the longest hit | insert splitting a node where two sequences diverge |
| L30 paged-memory-pools | `srt/mem_cache/memory_pool.py ::MHATokenToKVPool` | the two pools: ReqToToken (index) + TokenToKV (data) | page alloc on growth, free on finish (occupancy over time) |
| L31 hicache-tiering | `srt/mem_cache/hiradix_cache.py ::HiRadixCache` | GPU / CPU / disk tiers (capacity ↑, speed ↓) | swap-in on a deeper-tier hit, evict-down on pressure |
| L32 eviction-and-hit-rate | `srt/mem_cache/radix_cache.py ::RadixCache.evict` | LRU evicting evictable leaves; locked in-use nodes survive | hit-rate → throughput (the higher the hit rate, the better) |

### E8 — Part 8 · Attention & layers (L33–37) · `src/part8.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L33 attention-backend-abstraction | `srt/layers/attention/flashinfer_backend.py ::FlashInferAttnBackend` | model → AttentionBackend ABC → FlashInfer/Triton/FA | backend chosen by hardware/model at deploy |
| L34 moe-layer | `srt/layers/moe/topk.py ::TopKConfig` | router sends each token to its top-k experts | dense (all params) vs sparse MoE (only k experts) compute |
| L35 quantization | `srt/layers/quantization/fp8.py ::Fp8Config` | bit-width memory bars: fp16 vs fp8 vs int4 (same weights) | quantize → store low-bit → dequantize on use (flow) |
| L36 rope-norm-and-ops | `srt/layers/rotary_embedding/base.py ::RotaryEmbedding` | RoPE rotating q/k by position (vectors on a circle) | RMSNorm: divide by RMS, scale by weight (before→after) |
| L37 logits-and-vocab-parallel | `srt/layers/logits_processor.py ::LogitsProcessor` | hidden × [hidden×vocab] → logits (one score per token) | vocab sharded across TP ranks → all-gather full logits |

### E9 — Part 9 · Kernels & hardware (L38–42) · `src/part9.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L38 sgl-kernel-overview | `sgl-kernel/python/sgl_kernel/elementwise.py ::silu_and_mul` | Python wrapper → `torch.ops.sgl_kernel` → csrc CUDA kernel | AOT (.so in wheel) vs JIT (compiled at runtime) |
| L39 jit-kernel | `python/sglang/jit_kernel/activation.py ::silu_and_mul` | first call: compile → cache .so → reuse (timeline) | AOT shipped vs JIT compiled-on-demand |
| L40 attention-kernel-dissection | `sgl-kernel/python/sgl_kernel/attention.py ::merge_state_v2` | tiled Q·Kᵀ → online softmax → ·V (streaming tiles) | paged-KV gather: page_table → scattered KV pages |
| L41 operator-fusion-and-cuda-graph | `srt/layers/layernorm.py ::RMSNorm` (fused_add_rmsnorm) | unfused op→HBM→op→HBM vs fused single kernel | fuse → static shape → capture → replay |
| L42 multi-hardware-backends | `srt/platforms/cuda.py ::CudaSRTPlatform` | agnostic upper layers / platform line / per-chip kernels | one engine → NVIDIA/AMD/NPU/XPU/MUSA/MLX (capability flags) |

### E10 — Part 10 · Performance innovations (L43–48) · `src/part10.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L43 speculative-decoding-overview | `srt/speculative/spec_info.py ::SpeculativeAlgorithm` | draft k → target verifies in 1 forward → accept prefix+bonus | baseline 1 tok/forward vs spec m tok/forward |
| L44 eagle-and-next-gen | `srt/speculative/eagle_worker_v2.py ::EAGLEWorkerV2` | a token tree (root → topk children → grandchildren) | chain draft vs tree draft (tree accepts more) |
| L45 pd-disaggregation | `srt/disaggregation/base/conn.py ::BaseKVReceiver` | prefill pool → KV transfer → decode pool → stream | compute-bound prefill node vs bandwidth-bound decode node |
| L46 tp-pp-ep-dp-parallelism | `srt/distributed/parallel_state.py ::GroupCoordinator.all_reduce` | TP/PP/EP/DP splitting a model four ways | what each communicates: all-reduce / all-to-all / send-recv |
| L47 large-scale-ep-and-eplb | `srt/eplb/expert_location.py ::ExpertLocationMetadata` | skewed expert load: one hot expert stalls the step | rebalance flattening per-GPU token counts |
| L48 structured-outputs-and-jump-forward | `srt/constrained/outlines_jump_forward.py ::OutlinesJumpForwardMap` | logits → FSM vocab mask → only valid tokens sampled | jump-forward emitting a forced deterministic span |

### E11 — Part 11 · Advanced (L49–52) · `src/part11.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L49 multimodal-vlm-serving | `srt/multimodal/processors/base_processor.py ::BaseMultimodalProcessor` | media → placeholder tokens → encoder → splice | a token-embedding row with image embeddings at placeholder slots |
| L50 multi-lora-batching | `srt/lora/lora.py ::LoRAAdapter` | base W + low-rank B·A = adapted weight | one batch, rows using adapters A/B/A/C (grouped GEMM) |
| L51 rl-rollout-and-weight-sync | `srt/weight_sync/tensor_bucket.py ::FlattenedTensorBucket` | RL loop: rollout → reward+grad → update → rollout | many named tensors → one bucket → one copy → live params |
| L52 diffusion-models | `python/sglang/multimodal_gen/runtime/cache/teacache.py ::TeaCacheContext` | noise → denoise × N (the DiT) → VAE → image | autoregressive (token-by-token) vs diffusion (N-step denoise) |

### E12 — Part 12 · Practice & contributing (L53–57) · `src/part12.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L53 build-and-run | `python/sglang/launch_server.py ::run_server` | CLI flags → ServerArgs → engine → HTTP endpoint | Engine (offline, in-process) vs HTTP server (online) |
| L54 benchmark-and-profiling | `python/sglang/benchmark/serving.py ::benchmark` | load gen → server → collect TTFT/TPOT → percentiles | benchmark numbers (black box) vs profiler timeline (white box) |
| L55 test-suite-and-ci | `python/sglang/test/test_utils.py ::popen_launch_server` | unit test (no server) vs e2e test (launches a server) | PR push → pre-commit → CI partitioned stages → merge |
| L56 code-conventions-and-pr | `.pre-commit-config.yaml` (SGLang repo — the configured hooks) | fork → branch → pre-commit → test → push → PR | a good PR (small, tested) vs hard-to-review PR |
| L57 glossary | *(exempt — no 2nd codefile)* | ONE optional overview SVG: the 13-part map of the guide | — (glossary is exempt from the figure floor) |

### E13 — Part 13 · Design themes (L58–63) · `src/part13.py`
| Lesson | ADD 2nd codefile (candidate) | Figure A | Figure B |
| --- | --- | --- | --- |
| L58 radixattention-as-a-first-class-idea | `srt/mem_cache/radix_cache.py ::RadixCache.insert` | the radial web: where RadixAttention recurs (DSL/cache/sched/tier) | match → reuse → compute only the new suffix |
| L59 zero-overhead-scheduling | `srt/managers/scheduler.py ::Scheduler.event_loop_overlap` | CPU bubbles hidden behind the GPU forward (timeline) | the overlap pipeline: schedule N+1 ‖ forward N ‖ process N-1 |
| L60 two-workloads-one-engine | `srt/disaggregation/base/conn.py ::KVPoll` | prefill compute-bound vs decode bandwidth-bound (arith. intensity) | co-located time-share vs disaggregated separate pools |
| L61 draft-for-parallel-verify | `srt/speculative/eagle_info.py ::EagleVerifyInput` | serial decode (k steps) vs draft → 1 parallel verify | the trade: cheap parallel draft compute ↔ saved serial steps |
| L62 everything-is-pluggable | `srt/platforms/interface.py ::SRTPlatform` | stable core + pluggable edges (radial of 8 seams) | program-to-interface → select implementation at deploy |
| L63 built-for-throughput | `srt/managers/schedule_policy.py ::CacheAwarePolicy` | the throughput-lever map (each design → throughput it buys) | the full request path annotated with the lever at each stage |

---

## Task E-final — land the validator floor + final sweep

**Files:** `src/check_html.py:42-44` and `:98-106` region

- [ ] **Step 1:** Add `fig` to `DIAGRAM_CLASSES` and a `MIN_FIGURES` floor. Edit `src/check_html.py`:
```python
# line 42 — add "fig":
DIAGRAM_CLASSES = ("layers", "vflow", "flow", "cols", "cellgroup", "timeline", "trace", "fig")
MIN_DIAGRAMS = 6  # per lesson, counting BOTH languages (>= 3 per language)
MIN_FIGURES = 4   # SVG .fig per lesson, counting BOTH languages (>= 2 per language)
MIN_CJK = 3000
```
Then inside the `if fname not in SOFT_EXEMPT:` block (after the MIN_DIAGRAMS check, ~line 106) add:
```python
        nfig = html.count('class="fig"')
        if nfig < MIN_FIGURES:
            add("ERR", fname, f"only {nfig} SVG figures (want >= {MIN_FIGURES}; >= 2 per language)")
```
- [ ] **Step 2:** Run `cd src && python3 check_html.py`. Expected: **0 errors / 0 warnings** across 63
  lessons (every non-glossary lesson now has ≥4 `.fig`). If any lesson errors, it was missed in E1–E13 —
  fix that lesson, then re-run.
- [ ] **Step 3:** Full sweep: `python3 build.py && python3 build_print.py && python3 check_html.py &&
  python3 check_links.py` → 0/0; `git diff` no-diff after rebuild; print editions include the figures.
- [ ] **Step 4:** Update README: bump the figure mention if any; confirm claims still accurate. Commit:
```bash
git add -A && git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "feat(e-final): enforce MIN_FIGURES floor; all 63 lessons enriched

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
- [ ] **Step 5:** Push: `git push origin master` (triggers CI + Pages redeploy). Verify the live site
  shows figures (`curl -s -o /dev/null -w '%{http_code}' https://verdenmax.github.io/sglang-visual-guide/lessons/06-paged-attention-and-paged-kv.html`).

## Definition of done
All 63 lessons (glossary exempt) have ≥2 `.fig` SVG per language (CSS-vars only, theme-correct) + ≥2
`.codefile` + added inline examples; `check_html.py` enforces `MIN_FIGURES`; validators 0/0; links
resolve; no-diff; print editions + live Pages updated. Existing content preserved.
