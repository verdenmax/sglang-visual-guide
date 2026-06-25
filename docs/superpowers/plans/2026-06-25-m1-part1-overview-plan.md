# M1 — Part 1 Overview (L02–L03) — Plan

> **For agentic workers:** one implementer subagent builds both lessons, then two-stage review.
> Steps use checkbox (`- [ ]`) syntax. Spec: `specs/2026-06-25-m1-part1-overview-spec.md`.

**Goal:** Add L02 (project map) + L03 (life of a request) to complete Part 1.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source (read-only):** `~/course/sglang/`.

**Pattern to follow:** L01 in `src/part1.py` is the reference for HTML idioms, card classes, diagram
markup, bilingual parity, and the cited-`.codefile` style. Match its quality and density.

---

## Task 1 — L02 项目全景地图 / The project map

**Files:**
- Modify: `src/part1.py` (add `LESSON_02 = {"zh": ..., "en": ...}`)
- Modify: `src/quizzes.py` (add `QUIZZES["02-project-map.html"]`)
- Modify: `src/shell.py` (`PAGES` += L02 entry; `SUBTITLES` += L02 entry)
- Modify: `src/registry.py` (`CONTENT` += `"02-project-map.html": part1.LESSON_02`)

- [ ] **Step 1: Read source for accuracy.** Inspect, in `~/course/sglang/`:
  `python/sglang/` and `python/sglang/srt/` directory listings; `python/sglang/srt/entrypoints/engine.py`
  (find the subprocess-launch function — likely `_launch_subprocesses` and `Engine.__init__`);
  `python/sglang/srt/managers/scheduler.py` (top of file + `run_scheduler_process`); `srt/server_args.py`
  (skim the dataclass for the headline options). Note the real `file + symbol` you will cite.

- [ ] **Step 2: Author `part1.LESSON_02["zh"]`** to the content model (see spec "L02"):
  lead → 🔌 analogy (a big company's **org chart / floor plan**: front desk, dispatchers, workers,
  the machine room) → 🌍 macro (two halves `lang/` + `srt/`, kernels underneath; narrow core, power
  at the edges) → `<h2>` sections with **≥3 visual blocks**: (a) a `layers` package map
  DSL→entrypoints→managers→model_executor→layers→mem_cache→kernels; (b) a `table.t` 包 → 它负责什么 →
  第 N 课; (c) a `flow`/`cols` of the 3-process model (TokenizerManager ↔ Scheduler+TpWorker ↔
  DetokenizerManager) with ZMQ. One cited `.codefile` (the subprocess split). 本课要点 card.
  Forward-ref later parts with correct 第 N 课 numbers (per design spec §6 curriculum). ≥3000 CJK.
- [ ] **Step 3: Author `part1.LESSON_02["en"]`** — faithful parallel; identical diagram inventory.
- [ ] **Step 4: Add the quiz** `QUIZZES["02-project-map.html"]` (2–4 MCQ + 1–2 open), bilingual,
  design-insight (e.g. why split into processes? what does `srt` vs `lang` own? why ZMQ?).
- [ ] **Step 5: Wire** `shell.PAGES`, `shell.SUBTITLES`, `registry.CONTENT` for L02. PAGES entry:
```python
("02-project-map.html", "项目全景地图", "The project map",
 "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
```
  SUBTITLES entry: a one-line zh/en summary (e.g. "两半：前端 DSL + 运行时引擎 · 三进程 · ZMQ").
- [ ] **Step 6: Smoke-build** `cd src && python3 build.py && python3 check_html.py` — fix any
  ERROR/WARN for L02 (undefined CSS class, <3000 CJK, <6 visual blocks, missing cards) before Task 2.

## Task 2 — L03 一次请求的一生 / Life of a request

**Files:**
- Modify: `src/part1.py` (add `LESSON_03`)
- Modify: `src/quizzes.py` (add `QUIZZES["03-life-of-a-request.html"]`)
- Modify: `src/shell.py` (`PAGES` += L03; `SUBTITLES` += L03)
- Modify: `src/registry.py` (`CONTENT` += `"03-life-of-a-request.html": part1.LESSON_03`)

- [ ] **Step 1: Read source for accuracy.** In `~/course/sglang/python/sglang/srt/`:
  `entrypoints/http_server.py` (the `/generate` route handler); `managers/tokenizer_manager.py`
  (`TokenizerManager.generate_request` — how a request is tokenized, wrapped, sent over ZMQ);
  `managers/scheduler.py` (`event_loop_normal` / `event_loop_overlap`, `recv_requests`,
  `get_next_batch_to_run`, `run_batch`); `managers/io_struct.py` (the `*GenerateReqInput` /
  `*ReqOutput` structs); `managers/detokenizer_manager.py`. Record the real `file + symbol` to cite.
- [ ] **Step 2: Author `part1.LESSON_03["zh"]`** to the content model: lead → 🔌 analogy (an
  **order at a restaurant**: counter takes it → kitchen dispatcher batches → line cooks fire each
  pass → plating → served in courses) → 🌍 macro → `<h2>` sections with **≥3 visual blocks**:
  (a) a `vflow`/`trace` of the full path HTTP→Tokenizer→(ZMQ)→Scheduler→ModelRunner→Sampler→
  Detokenizer→SSE, with process boundaries marked; (b) a `flow` of the decode loop (steps 2–4
  repeat); (c) a `timeline`/`cols` prefill-vs-decode. One cited `.codefile` (the scheduler event
  loop or `generate_request`). 本课要点 card. Forward-ref deep lessons (entrypoints 第 13–17 课,
  scheduler 第 18–23 课, model exec 第 24–28 课, KV cache 第 29–32 课). ≥3000 CJK.
- [ ] **Step 3: Author `part1.LESSON_03["en"]`** — faithful parallel; identical diagram inventory.
- [ ] **Step 4: Add the quiz** `QUIZZES["03-life-of-a-request.html"]` (2–4 MCQ + 1–2 open),
  bilingual, design-insight (e.g. why is the Scheduler a separate process? what loops, prefill vs
  decode? where does batching "cut in"?).
- [ ] **Step 5: Wire** `shell.PAGES`, `shell.SUBTITLES`, `registry.CONTENT` for L03. PAGES entry:
```python
("03-life-of-a-request.html", "一次请求的一生", "Life of a request",
 "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
```
  SUBTITLES: one-line zh/en (e.g. "端到端追踪一条 generate · 预填充 + 多次解码 · 进程边界").

## Verify (both lessons)

- [ ] **Step 1:** Full build + validate:
```bash
cd ~/course/sglang-visual-guide/src
python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py
```
  Expected: `check_html.py` → "0 error(s), 0 warning(s)"; `check_links.py` passes; index pill reads
  **"共 3 课 · 1 个部分"**; nav chain L01↔L02↔L03 resolves. Fix all ERROR/WARN.
- [ ] **Step 2:** No-diff guard:
```bash
cd ~/course/sglang-visual-guide && python3 src/build.py >/dev/null && git add -A && git status --short
```
  A second `build.py` must produce no further changes.

## Commit

- [ ] One commit for the milestone:
```bash
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "M1: Part 1 overview — L02 project map + L03 life of a request

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Guardrails
- Cite `file + symbol`, never bare line numbers. Small snippets only; explain, don't dump.
- Forward-ref with **correct** 第 N 课 numbers per design spec §6 (don't number sequentially in row
  order — that was the M0 review bug; map each teaser to its real lesson).
- zh ≥ 3000 CJK (target ~4000); zh/en strict parity (identical diagram inventory).
- Use only CSS classes defined in `shell.CSS`. Default page language zh.
- The 3-process model: TokenizerManager and DetokenizerManager are separate processes from the
  Scheduler; Scheduler holds the TpWorker/ModelRunner and does the GPU work. Don't conflate them.
- Don't touch `docs/`, the reference repo, or L01. SGLang is Apache-2.0.
