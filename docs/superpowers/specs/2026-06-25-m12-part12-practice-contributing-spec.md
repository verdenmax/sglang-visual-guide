# M12 — Part 12 Practice & Contributing (L53–L57) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec §P12, roadmap M12. Builds on M0–M11 (L01–52 exist).

## Goal
Turn the reader from "understands SGLang" into "can run, measure, test, and contribute to SGLang."
Five lessons: build & run a server, benchmark & profile it, the test suite & CI, code conventions &
opening a PR, and a whole-guide glossary. Module: `src/part12.py`, lessons L53–57 (5).

## Content model & gates
**L53–L56 are normal lessons** (same model as all prior): lead → 🔌 analogy (`card analogy`,
`🔌 生活类比` / `🔌 Analogy`) → 🌍 macro (`card macro`, `🌍 宏观理解` / `🌍 The big picture`) → 3–4 `<h2>`
with EXACTLY 4 visual blocks/lang (identical zh/en; from cols/flow/vflow/layers/cellgroup/`table.t`/
step+num+sc; NO `<svg>`, NOT `timeline`) → one cited `.codefile` (`cf-head` dot/path/ln; escape
`<`/`>`/`&`; docstrings→`#`) → key card (`card key`, `📌 本课要点` / `📌 Key points`) → quiz (3 MCQ + 2
open). **zh ≥3600 CJK.**
**L57 is the GLOSSARY** — exempt from MIN_CJK (`check_html.SOFT_EXEMPT={"57-glossary.html"}`). Format
(mirrors the reference guide): a lead `<p>` then thematic `<h2>` sections, each a `<table class="t">`
with columns 术语 / 一句话定义 / 课次 (term / one-line definition / lesson). NO analogy/macro/codefile/
quiz needed; it's a reference. Bilingual (zh + en versions, same tables).
All lessons: only `shell.CSS` classes; validators 0/0; cite REAL symbols.

## Lessons

### L53 — 53-build-and-run.html / "构建与运行 / Build & run"
**Scope:** get a server up. `python -m sglang.launch_server --model-path … [flags]` → `run_server`
parses a huge `ServerArgs` dataclass (model path, dtype/quantization, tp-size/dp-size,
mem-fraction-static, attention-backend, chunked-prefill-size, max-running-requests, …) and boots the
engine (TokenizerManager + Scheduler + DetokenizerManager, 第13课). Knowing the KEY knobs maps the
whole guide onto CLI flags: `--tp-size`/`--dp-size` (第46课), `--attention-backend` (第33课),
`--quantization` (第35课), `--mem-fraction-static` (第30/31课 KV pool), `--chunked-prefill-size`
(第22课), `--speculative-algorithm` (第43课), `--enable-eplb` (第47课). Also `Engine` (offline, in-process)
vs the HTTP server (第13课). **Cited:** `python/sglang/srt/server_args.py ::ServerArgs` (the central
config). **Read:** `python/sglang/launch_server.py` (`run_server`, `prepare_server_args`),
`srt/server_args.py` (ServerArgs fields + add_cli_args). **Diagrams:** a `flow` CLI flags → ServerArgs
→ launch engine → HTTP endpoint; a `table.t` key flag → what it controls (→ which lesson); a `cols`
Engine (offline/in-process) vs HTTP server (online); a `vflow`/`step` install → launch → /health →
/generate. Tie 第13课 (engine/server), 第15课 (OpenAI API). Forward-ref bench 第54课.

### L54 — 54-benchmark-and-profiling.html / "压测与性能分析 / Benchmark & profiling"
**Scope:** measure it. `python -m sglang.bench_serving` fires a configurable request load at a running
server and reports a `BenchmarkMetrics`: request/input/output throughput plus TTFT and TPOT/ITL with
mean / median / p90 / p95 / p99 percentiles — exactly the latency-vs-throughput trade-off of 第8课.
`bench_one_batch` measures a single batch's latency in isolation. For WHY it's slow, the torch
profiler (set `SGLANG_TORCH_PROFILER_DIR`, or `/start_profile`+`/stop_profile`) captures a kernel
timeline you open in Chrome trace / Perfetto — letting you see prefill vs decode, kernel gaps, CUDA
graph replay. Read metrics right: throughput (system view) vs per-request latency (user view), and the
percentiles matter under load. **Cited:** `python/sglang/benchmark/serving.py ::BenchmarkMetrics` (the
result schema). **Read:** `python/sglang/benchmark/serving.py` (BenchmarkMetrics, benchmark,
run_benchmark), `benchmark/one_batch_server.py`, profiler hooks. **Diagrams:** a `flow` load gen →
server → collect TTFT/TPOT → BenchmarkMetrics; a `table.t` metric → meaning (throughput/TTFT/TPOT/p99);
a `cols` benchmark (black-box numbers) vs profiler (white-box kernel timeline); a `vflow`/`step`
warm-up → run → percentiles → read the trace. Tie 第8课 (throughput vs latency), 第4课, 第27课. Forward-ref
tests 第55课.

### L55 — 55-test-suite-and-ci.html / "测试套件与 CI / Test suite & CI"
**Scope:** how SGLang is tested. The `test/` tree holds unit tests (no server) and e2e tests (launch a
server). SGLang uses Python `unittest` + `pytest` runner; tests subclass `CustomTestCase`, which wraps
`setUpClass` so `tearDownClass` ALWAYS runs even if setup fails — preventing leaked ports/processes
across the big test suite. CI registers tests under `test/registered/` and runs them in partitioned
stages on labelled GPU runners; `pre-commit` gates style before code even reaches CI. Knowing this lets
a contributor add a test for their change and run it locally (`pytest test/registered/unit/…`). **Cited:**
`python/sglang/test/test_utils.py ::CustomTestCase` (the safe-teardown base class). **Read:**
`python/sglang/test/test_utils.py` (CustomTestCase, popen_launch_server, is_in_ci),
`test/README.md`, `test/registered/`. **Diagrams:** a `cols` unit test (no server) vs e2e test (server
required); a `flow` PR push → pre-commit → CI partitioned stages → merge gate; a `table.t` test helper
→ role (CustomTestCase / popen_launch_server / run_eval); a `vflow`/`step` write test → run locally →
push → CI green. Tie 第13课 (server to launch), 第56课 (PR flow). (Honors project conventions; there is a
`write-sglang-test` skill.)

### L56 — 56-code-conventions-and-pr.html / "代码规范与提 PR / Conventions & opening a PR"
**Scope:** how to contribute. The flow: fork → clone → branch → make the change → `pre-commit install`
+ `pre-commit run --all-files` (formats + lints; re-run until clean) → add a unit test (第55课) → run it
locally → push → open a PR against `main`. Conventions: pre-commit enforces style (black/isort/etc.);
new `srt/` code should get a `test/registered/unit/` test; docs go under `docs_new/` (legacy `docs/` is
rejected by lint); follow the existing module patterns (e.g. pluggable backends, the `large-class-init`
style for Scheduler/ModelRunner). A good PR is small, tested, and explains WHY. **Cited:**
`docs/developer_guide/contribution_guide.md` (the official fork→pre-commit→test→PR steps). **Read:**
`docs/developer_guide/contribution_guide.md`, `.pre-commit-config.yaml`, `CONTRIBUTING`/`docs_new/`.
**Diagrams:** a `vflow`/`step` fork → branch → pre-commit → test → push → PR; a `table.t` step → command
/ gate; a `cols` good PR (small, tested, motivated) vs hard-to-review PR; a `flow` local checks →
CI checks → review → merge. NOTE: the cited file is a `.md` doc, not a Python symbol — caption it as
`docs/developer_guide/contribution_guide.md` and put the real command sequence (pre-commit install /
run --all-files / pytest) in the `<pre>`. Tie 第55课 (tests), 第63课-ish themes. Forward-ref glossary 第57课.

### L57 — 57-glossary.html / "术语速查表 / Glossary"
**Scope:** a whole-guide bilingual glossary. Lead `<p>` explaining it collects the key terms from
L01–56, each with a one-line definition and its lesson number for back-reference. Then thematic `<h2>`
+ `<table class="t">` (头 术语 / 一句话定义 / 课次) sections, roughly: 总览与架构 (LLM/token/autoregression/
prefill/decode/throughput/latency — L1-8); 前端 DSL (RadixAttention concept/fork-join/SGLang DSL — L9-12);
入口与分发 (Engine/TokenizerManager/IO structs/detokenizer/OpenAI API — L13-17); 调度器 (event loop/
ScheduleBatch/policy/overlap/chunked prefill/DP-PP — L18-23); 模型执行 (ModelRunner/ForwardBatch/CUDA
graph/sampler — L24-28); KV 缓存 (RadixAttention impl/paged pool/HiCache/eviction — L29-32); 注意力与算子
(attention backend/MoE/quantization/RoPE-norm/logits — L33-37); 内核与硬件 (sgl-kernel/JIT/attention
kernel/fusion/multi-hw — L38-42); 性能创新 (speculative/EAGLE/PD disaggregation/TP-PP-EP-DP/EPLB/
structured outputs — L43-48); 进阶 (VLM/LoRA/RL weight-sync/diffusion — L49-52); 实战 (ServerArgs/
BenchmarkMetrics/CustomTestCase — L53-55). **NO codefile, NO quiz** (it's a reference). zh + en tables.
**Cite** lesson numbers only (no source symbol). Closes Part 12 + the main learning track.

## Wiring & DoD
- New module `src/part12.py` (`LESSON_53..57`); `registry.py` imports `part12` + 5 keys; `shell.PAGES`
  + `SUBTITLES` += 5; `quizzes.QUIZZES` += 4 (L53–56 only; L57 has no quiz). Part label
  "第十二部分 · 实战与贡献" / "Part 12 · Practice & contributing".
- Filenames: `53-build-and-run.html`, `54-benchmark-and-profiling.html`, `55-test-suite-and-ci.html`,
  `56-code-conventions-and-pr.html`, `57-glossary.html`.
- All validators 0 err / 0 warn (L57 CJK-exempt); no-diff; pill "共 57 课 · 12 个部分"; nav L52↔L53…L57.
- Source-accurate: cite real symbols (L56 cites the .md guide); L57 cites lesson numbers only.
