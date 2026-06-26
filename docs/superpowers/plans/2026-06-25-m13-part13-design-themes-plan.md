# M13 — Part 13 Design Themes / Synthesis (L58–L63) — Plan

> Execute with subagent-driven-development. One subagent per lesson, claude-opus-4.8, sync, high.
> Two-stage review (research + code-review, opus-4.8 max, long context) after exec. FINAL content part.

**Goal:** add Part 13 (L58–63) — six synthesis/theme lessons tying the guide together.
**Companion spec:** `docs/superpowers/specs/2026-06-25-m13-part13-design-themes-spec.md`.

## Verified citations (pre-fetched — re-citing earlier symbols is intended for synthesis lessons)
- L58 `python/sglang/srt/mem_cache/radix_cache.py ::RadixCache` (line 286) — prefix tree as a
  first-class data structure.
- L59 `python/sglang/srt/managers/scheduler.py ::Scheduler` (line 295; `enable_overlap` ~340) — the
  overlap event loop.
- L60 `python/sglang/srt/disaggregation/base/conn.py ::BaseKVManager` (line 84) — the PD seam.
- L61 `python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker` (line 274) — draft+target.
- L62 `python/sglang/srt/layers/attention/base_attn_backend.py ::AttentionBackend` (line 18) — the
  archetypal pluggable ABC.
- L63 `python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy` (line 149; `CacheAwarePolicy`
  ~133) — cache-aware policy maximizing batch + prefix-hit throughput.

## Authoritative numbering for backward-refs (from shell.py — synthesis lessons ref MANY lessons)
L3 life-of-request, L4 autoregression/bandwidth, L5 continuous batching, L6 paged attention, L7
RadixAttention concept, L8 throughput vs latency, L11 fork-join, L14 tokenizer, L16 IPC, L18 scheduler
loop, L20 schedule policy, L21 overlap scheduler, L22 chunked prefill, L26 writing a model, L27 CUDA
graph, L29 RadixAttention impl, L30 paged pools, L31 HiCache, L33 attention backend, L35 quantization,
L38 sgl-kernel, L41 fusion, L42 multi-hardware, L43 speculative, L44 EAGLE, L45 PD disaggregation, L46
TP/PP/EP/DP, L48 structured outputs. (Verify every "第 N 课 / Lesson N" against this.)

## Task 0 — orchestrator wiring (do myself)
- [ ] Create `src/part13.py` with `LESSON_58..LESSON_63` placeholders.
- [ ] `registry.py`: `import part13`; add 6 CONTENT keys (filenames below → `part13.LESSON_6X`).
- [ ] `shell.py`: append 6 PAGES 5-tuples (anchor after L57) + 6 SUBTITLES; label
  "第十三部分 · 设计专题综合" / "Part 13 · Design themes (synthesis)".
- [ ] Build + confirm pill "共 63 课 · 13 个部分", nav L57↔L58.

Filenames: `58-radixattention-as-a-first-class-idea.html`, `59-zero-overhead-scheduling.html`,
`60-two-workloads-one-engine.html`, `61-draft-for-parallel-verify.html`,
`62-everything-is-pluggable.html`, `63-built-for-throughput.html`.

## Tasks 1–6 — per-lesson subagents (L58…L63)
Each: replace placeholder in `src/part13.py` + append `quizzes.py` entry. Proven edit-first template:
exact placeholder line; verified `file ::symbol` + real condensed code; house markup; allowed CSS
palette; "≥3600 CJK zh, identical zh/en inventory, EXACTLY 4 diagram blocks incl. one cross-lesson
`table.t`/`cols` web, NO SVG, NOT timeline, docstrings→`#`"; backward-ref numbers per the map above;
VERIFY cmds. Retry on truncation; expand fallback ×3.

- [ ] **T1 L58** RadixAttention first-class — RadixCache (the "where it shows up" web).
- [ ] **T2 L59** zero-overhead scheduling — Scheduler/enable_overlap (hide CPU bubbles).
- [ ] **T3 L60** two workloads one engine — BaseKVManager (prefill vs decode root tension).
- [ ] **T4 L61** draft-for-parallel-verify — BaseSpecWorker (serial→parallel trick).
- [ ] **T5 L62** everything pluggable — AttentionBackend (program-to-an-interface web).
- [ ] **T6 L63** built for throughput — SchedulePolicy (capstone map + guide-wide closing note). Closes guide.

## Task 7 — verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py &&
  python3 check_links.py` → 0/0; no-diff. Commit (`verdenmax`, Co-authored-by Copilot).

## Task 8 — two-stage review + fixes
- [ ] research (concept/spec/completeness + EVERY backward-ref lesson number correct against shell.py —
  these lessons ref many) + code-review (markup/parity/escaping/wiring/quiz schema), parallel, opus-4.8
  max, long context. Fix → re-verify → commit.

## DoD
63 lessons; Part 13 nav + pill correct; validators 0/0; no-diff; citations source-accurate; all
backward-ref numbers match shell.py; L63 closes the guide; spec scope met.
