# M12 — Part 12 Practice & Contributing (L53–L57) — Plan

> Execute with subagent-driven-development. L53–56: one subagent each, claude-opus-4.8, sync, high.
> L57 (glossary): hand-author (table-heavy, needs accurate term→lesson mapping). Two-stage review
> (research + code-review, opus-4.8 max, long context) after exec.

**Goal:** add Part 12 (L53–57) — build/run, benchmark/profile, test/CI, conventions/PR, glossary.
**Companion spec:** `docs/superpowers/specs/2026-06-25-m12-part12-practice-contributing-spec.md`.

## Verified citations (pre-fetched — pass to subagents verbatim)
- L53 `python/sglang/srt/server_args.py ::ServerArgs` (central config dataclass, line 374);
  `python/sglang/launch_server.py` (`run_server` line 15, `prepare_server_args`). Key flags map to
  lessons: --tp-size/--dp-size (L46), --attention-backend (L33), --quantization (L35),
  --mem-fraction-static (L30/31), --chunked-prefill-size (L22), --speculative-algorithm (L43),
  --enable-eplb (L47).
- L54 `python/sglang/benchmark/serving.py ::BenchmarkMetrics` (line 937: request/input/output
  throughput, mean/median/p90/p95/p99 TTFT + TPOT); `benchmark/one_batch_server.py`; profiler env
  `SGLANG_TORCH_PROFILER_DIR` / `/start_profile`+`/stop_profile`.
- L55 `python/sglang/test/test_utils.py ::CustomTestCase` (line 2172: wraps setUpClass so
  tearDownClass runs even on failure → no leaked ports/processes); `popen_launch_server`, `is_in_ci`;
  `test/registered/unit/`, `test/README.md`.
- L56 `docs/developer_guide/contribution_guide.md` (fork→`pre-commit install`→`pre-commit run
  --all-files`→add unit test→`pytest test/registered/unit/`→push→PR). `.pre-commit-config.yaml`.
  NOTE: cited file is a `.md` doc — caption it as the path, put the real command sequence in `<pre>`.
- L57 glossary: cite LESSON NUMBERS only (use the authoritative shell.py numbering). No source symbol.

## Authoritative lesson numbering (from src/shell.py — use for all cross-refs & the glossary)
L1 what-is, L2 project-map, L3 life-of-request, L4 autoregression/KV, L5 continuous batching, L6 paged
attention/KV, L7 RadixAttention+prefix, L8 throughput vs latency; L9 DSL, L10 interpreter/tracer, L11
fork-join, L12 backends/OpenAI-compat; L13 engine/HTTP server, L14 tokenizer manager, L15 OpenAI/
Anthropic/Ollama compat, L16 IO structs/IPC, L17 detokenizer/streaming; L18 scheduler loop, L19 Req &
ScheduleBatch, L20 schedule policy, L21 overlap scheduler, L22 chunked prefill, L23 DP controller/PP;
L24 ModelRunner & ForwardBatch, L25 model loading/weights, L26 writing a model, L27 CUDA graph, L28
sampler; L29 RadixAttention impl, L30 paged memory pools, L31 HiCache, L32 eviction/hit-rate; L33
attention backend abstraction, L34 MoE, L35 quantization, L36 RoPE/norm/ops, L37 logits/vocab-parallel;
L38 sgl-kernel, L39 JIT kernel, L40 attention kernel, L41 fusion+CUDA graph, L42 multi-hardware; L43
speculative overview, L44 EAGLE, L45 PD disaggregation, L46 TP/PP/EP/DP, L47 EPLB, L48 structured
outputs; L49 VLM, L50 multi-LoRA, L51 RL weight-sync, L52 diffusion; L53 build/run, L54 bench/profile,
L55 test/CI, L56 conventions/PR, L57 glossary.

## Task 0 — orchestrator wiring (do myself)
- [ ] Create `src/part12.py` with `LESSON_53..LESSON_57` placeholders.
- [ ] `registry.py`: `import part12`; add 5 CONTENT keys (filenames below → `part12.LESSON_5X`).
- [ ] `shell.py`: append 5 PAGES 5-tuples (anchor after L52) + 5 SUBTITLES; label
  "第十二部分 · 实战与贡献" / "Part 12 · Practice & contributing".
- [ ] Build + confirm pill "共 57 课 · 12 个部分", nav L52↔L53.

Filenames: `53-build-and-run.html`, `54-benchmark-and-profiling.html`, `55-test-suite-and-ci.html`,
`56-code-conventions-and-pr.html`, `57-glossary.html`.

## Tasks 1–4 — per-lesson subagents (L53…L56)
Each: replace placeholder in `src/part12.py` + append `quizzes.py` entry. Proven edit-first template:
exact placeholder line; verified `file ::symbol` + real condensed code (L56 = the .md path + command
block); house markup; allowed CSS palette; "≥3600 CJK zh, identical zh/en inventory, EXACTLY 4 diagram
blocks, NO SVG, NOT timeline, docstrings→`#`"; VERIFY cmds. Retry on truncation; expand fallback ×3.

- [ ] **T1 L53** build & run — ServerArgs (CLI flags → lessons).
- [ ] **T2 L54** benchmark & profiling — BenchmarkMetrics (throughput/TTFT/TPOT/percentiles + profiler).
- [ ] **T3 L55** test suite & CI — CustomTestCase (safe teardown; unit vs e2e; pre-commit→CI).
- [ ] **T4 L56** conventions & PR — contribution_guide.md (fork→pre-commit→test→PR command block).

## Task 5 — L57 glossary (hand-author myself, incrementally)
- [ ] Author `LESSON_57` in `src/part12.py`: lead `<p>` + thematic `<h2>` + `<table class="t">`
  (术语/一句话定义/课次) sections covering L01–55 terms (use the numbering above). zh + en tables, same
  rows. NO codefile, NO quiz. Build incrementally (one `<h2>` table block at a time per spec).

## Task 6 — verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py &&
  python3 check_links.py` → 0/0 (L57 CJK-exempt); no-diff. Commit (`verdenmax`, Co-authored-by Copilot).

## Task 7 — two-stage review + fixes
- [ ] research (concept/spec/completeness: ServerArgs flags, bench metrics, CustomTestCase semantics,
  PR flow accuracy, glossary term→lesson correctness) + code-review (markup/parity/escaping/wiring/
  quiz schema; verify L57 glossary structure + that it has no quiz entry), parallel, opus-4.8 max, long
  context. Fix → re-verify → commit.

## DoD
57 lessons; Part 12 nav + pill correct; validators 0/0 (L57 exempt); no-diff; citations source-accurate;
glossary lesson numbers match shell.py; spec scope met.
