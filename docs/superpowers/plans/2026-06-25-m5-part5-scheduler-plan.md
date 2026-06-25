# M5 — Part 5 The Scheduler (L18–L23) — Plan

> One sync subagent per lesson, then two-stage review. Spec:
> `specs/2026-06-25-m5-part5-scheduler-spec.md`. Steps use checkbox (`- [ ]`).

**Goal:** Add L18–L23 (the scheduler, the engine's heart) as `src/part5.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source:** `~/course/sglang/python/sglang/srt/managers/`.

## Common recipe (every lesson)
Model on part1–4 (full real lessons). Each `LESSON_NN = {"zh": r"""…""","en": r"""…"""}`: lead → 🔌
analogy → 🌍 macro → 3–4 `<h2>` → **≥3 visual blocks/lang, identical zh/en inventory** (cols/vflow/flow/
layers/cellgroup/timeline/`<table class="t">`, no SVG) → ONE cited `.codefile` (`file ::symbol`, small
faithful snippet) → 本课要点 card → quiz (3 MCQ + 2 open). **zh ≥ 3500 CJK (aim ~3800)** — core lessons,
add depth. Only `shell.CSS` classes. Forward-refs map to real lessons. After each lesson, smoke-build.

## Task 0 — module + wiring (orchestrator)
- [ ] Create `src/part5.py` (`LESSON_18..23` placeholders); `import part5` in `registry.py`; add 6 PAGES
  + 6 SUBTITLES + 6 CONTENT keys. Part label "第五部分 · 调度器 / Part 5 · The scheduler". Filenames:
  `18-scheduler-event-loop.html`, `19-req-and-schedule-batch.html`, `20-schedule-policy.html`,
  `21-zero-overhead-overlap-scheduler.html`, `22-chunked-prefill.html`, `23-dp-controller-and-pp-scheduling.html`.

## Per-lesson tasks (read source, author per spec, add quiz, smoke-build)
- [ ] **L18** event loop. Cite `srt/managers/scheduler.py ::Scheduler.event_loop_normal` (~1516). Read
  event_loop_normal, process_input_requests ~1639, run_batch ~3185. Analogy: an **air-traffic controller's
  radar sweep** — every tick: take new arrivals, decide who flies this tick, run them, handle results, repeat.
  Diagrams: vflow 5-stage loop; flow recv→schedule→forward→output; table.t stage→lesson.
- [ ] **L19** Req & ScheduleBatch. Cite `srt/managers/schedule_batch.py ::ScheduleBatch` (~1669; Req ~666,
  prepare_for_extend ~2006, prepare_for_decode ~2613). Analogy: a **boarding manifest** — each passenger
  (Req) has a state; the manifest (ScheduleBatch) is either a boarding group (extend/prefill) or a cruising
  cabin (decode). Diagrams: vflow Req lifecycle (waiting→running→finished); cols extend-vs-decode batch;
  table.t fields→role; cellgroup batch of reqs.
- [ ] **L20** schedule policy. Cite `srt/managers/schedule_policy.py ::SchedulePolicy.calc_priority` (~170;
  PrefillAdder ~425, add_one_req ~866). Analogy: a **smart queue/bouncer** — orders the line by who reuses
  the cache (LPM, 第 7 课) or arrival/priority, and admits as many as fit the budget. Diagrams: table.t
  policies (LPM/FCFS/priority→effect); vflow PrefillAdder filling under budget; cols cache-aware-vs-FCFS;
  cellgroup queue ordering.
- [ ] **L21** zero-overhead overlap (★). Cite `srt/managers/scheduler.py ::Scheduler.event_loop_overlap`
  (~1546). Read event_loop_overlap, `srt/batch_overlap/`, overlap_utils.py. Analogy: a **relay/assembly
  line** — while one worker (GPU) runs the current part, the next (CPU) is already prepping the following
  one; nobody waits. Diagrams: **timeline GPU/CPU serial-vs-overlapped** (the key picture); vflow one-step
  staggered pipeline; cols normal-vs-overlap; table.t cost/benefit. Make "GPU never idles" vivid. Tie to L01/L03.
- [ ] **L22** chunked prefill. Cite `srt/managers/schedule_policy.py ::PrefillAdder` (the budget/chunking)
  or `srt/managers/prefill_delayer.py`. Read add_one_req ~866, server_args chunked_prefill_size. Analogy:
  **eating a huge meal in bites** while still chatting — split a giant prefill into chunks across steps,
  mixed with decodes, so no step is huge. Diagrams: timeline one-big-prefill-spike vs chunked-smooth; vflow
  splitting prefill into chunks; cols without-vs-with chunking; table.t knob/effect. Forward-ref 第 8 课.
- [ ] **L23** DP controller & PP scheduling. Cite `srt/managers/data_parallel_controller.py
  ::DataParallelController` (~127; round_robin_scheduler ~606, event_loop ~648) or
  `scheduler_pp_mixin.py ::SchedulerPPMixin.event_loop_pp` (~68). Analogy: **multiple identical checkout
  lanes (DP)** vs **a single conveyor with stations (PP)**. Diagrams: layers/flow DP controller→N replicas;
  vflow/timeline PP micro-batch pipeline; table.t DP-vs-PP-vs-TP (replicated/split); cols DP-vs-PP.
  Forward-ref parallelism 第 46 课. This closes Part 5.

## Verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py`
  → 0 err / 0 warn; pill "共 23 课 · 5 个部分"; nav L17↔L18↔…↔L23; no-diff.
- [ ] One commit: `M5: Part 5 scheduler — L18 event loop, L19 Req/ScheduleBatch, L20 policy, L21 overlap scheduler, L22 chunked prefill, L23 DP/PP` (+ Co-authored-by trailer).

## Guardrails
- Cite `file ::symbol`; the event-loop and overlap mechanics MUST match scheduler.py (don't invent a
  different loop order; recv→process_input→get_next_batch_to_run→run_batch→process_batch_result).
- L21 is SGLang's signature scheduler win — get the "CPU schedules N+1 while GPU runs N, results one step
  late" pipeline exactly right; cite event_loop_overlap.
- zh ≥3500 CJK; zh/en identical diagram inventory. Don't touch `docs/`, earlier parts, pipeline, reference repo.
