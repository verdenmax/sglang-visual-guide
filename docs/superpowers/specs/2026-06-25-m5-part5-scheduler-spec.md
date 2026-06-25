# M5 — Part 5 The Scheduler (L18–L23) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M5. Builds on M0–M4 (L01–17 exist).

## Goal
The engine's **heart**. How the Scheduler subprocess drives everything: the event loop, the request
state machine + batch object, the admission/ordering policy, the zero-overhead overlap trick (SGLang's
signature scheduler optimization), chunked prefill, and multi-process DP/PP scheduling. Module:
`src/part5.py`, lessons L18–23. These are the most important runtime lessons — author with care.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with ≥3 visual blocks/lang (identical zh/en; no SVG) →
one cited `.codefile` (`file ::symbol`) → 本课要点 card → quiz (3 MCQ + 2 open). **zh ≥3500 CJK
(aim ~3800)** — these are the core lessons, give them depth. Only `shell.CSS` classes. Validators
0 err / 0 warn. Forward-refs map to real lessons.

## Lessons

### L18 — 18-scheduler-event-loop.html / "调度器事件循环 / The scheduler event loop"
**Scope:** the Scheduler runs in its own process; `event_loop_normal` is the heartbeat — an infinite
loop: `recv_requests` (drain ZMQ) → `process_input_requests` (add to waiting queue) →
`get_next_batch_to_run` (form this step's batch) → `run_batch` (GPU forward via TpWorker) →
`process_batch_result` (sample, append, detect finished, send outputs to detok) → repeat. This is where
continuous batching (第 5 课) actually happens. **Cited:** `srt/managers/scheduler.py
::Scheduler.event_loop_normal`. **Read:** scheduler.py (event_loop_normal ~1516, process_input_requests
~1639, run_batch ~3185). **Diagrams:** a `vflow` of the loop's 5 stages; a `flow` recv→schedule→
forward→output→loop; a `table.t` stage → what it does → which lesson details it.

### L19 — 19-req-and-schedule-batch.html / "Req 与 ScheduleBatch / Req & ScheduleBatch"
**Scope:** the data the loop operates on. A `Req` is one request's state machine (tokens, output,
sampling params, KV indices, finish status). A `ScheduleBatch` bundles the reqs for one forward step,
with a **mode**: an EXTEND/prefill batch (new prompts) or a DECODE batch (one step each). The batch
prepares tensors (`prepare_for_extend` / `prepare_for_decode`), tracks the KV pool allocation, and
filters finished reqs. **Cited:** `srt/managers/schedule_batch.py ::ScheduleBatch` (or `::Req`). **Read:**
schedule_batch.py (Req ~666, ScheduleBatch ~1669, prepare_for_extend ~2006, prepare_for_decode ~2613).
**Diagrams:** a `vflow`/`flow` Req lifecycle (waiting→running→finished); a `cols` extend-batch vs
decode-batch; a `table.t` Req/ScheduleBatch fields → role; a `cellgroup` of a batch's reqs.

### L20 — 20-schedule-policy.html / "调度策略 / The schedule policy"
**Scope:** who runs next. The waiting queue is ordered by a `SchedulePolicy` (`calc_priority`):
cache-aware (Longest-Prefix-Match — favor reqs that hit the radix cache, 第 7 课), FCFS, or priority;
`PrefillAdder` decides how many waiting reqs fit into the next prefill batch under the token/memory
budget (and chunks long prompts → 第 22 课). The prefill-vs-decode admission decision. **Cited:**
`srt/managers/schedule_policy.py ::SchedulePolicy.calc_priority` (or `::PrefillAdder`). **Read:**
schedule_policy.py (SchedulePolicy ~149, calc_priority ~170, PrefillAdder ~425, add_one_req ~866).
**Diagrams:** a `table.t` policies (LPM/FCFS/priority → effect); a `vflow` PrefillAdder filling a batch
under budget; a `cols` cache-aware vs FCFS; a `cellgroup` queue ordering.

### L21 — 21-zero-overhead-overlap-scheduler.html / "零开销重叠调度器" (★ signature)
**Scope:** the headline optimization. The CPU work of scheduling a step (build batch, sample, bookkeeping)
would normally make the GPU wait. `event_loop_overlap` **pipelines** them: while the GPU runs step N's
forward, the CPU already schedules step N+1, and results are processed one step late (a `result_queue` /
future). So per-step CPU overhead is hidden behind GPU compute → near-zero-overhead. The cost: one step
of extra latency + careful state management. **Cited:** `srt/managers/scheduler.py
::Scheduler.event_loop_overlap`. **Read:** scheduler.py (event_loop_overlap ~1546), `srt/batch_overlap/`,
`managers/overlap_utils.py`. **Diagrams:** a `timeline` GPU/CPU serial vs overlapped (the key picture);
a `vflow` of the one-step-staggered pipeline; a `cols` normal-vs-overlap loop; a `table.t` cost/benefit.
This lesson is referenced by L01/L03 — make the GPU-never-idles picture vivid.

### L22 — 22-chunked-prefill.html / "分块预填充 / Chunked prefill"
**Scope:** long prompts break the pipeline. A 32k-token prefill done in one step stalls everyone's decode
(latency spike). Chunked prefill splits a big prefill into fixed-size token chunks (`chunked_prefill_size`)
processed over several steps, **mixed with ongoing decodes**, so no single step is huge. Trade-off: more
steps, but smooth latency + better goodput. The `PrefillAdder` enforces the per-step token budget; the
prefill delayer can defer. **Cited:** `srt/managers/schedule_policy.py ::PrefillAdder` (the chunking/budget)
or `srt/managers/prefill_delayer.py`. **Read:** schedule_policy.py (PrefillAdder, add_one_req ~866),
`srt/server_args.py` (chunked_prefill_size), prefill_delayer.py. **Diagrams:** a `timeline` one-big-prefill
(spike) vs chunked (smooth); a `vflow` splitting a prefill into chunks across steps; a `cols`
without-vs-with chunking; a `table.t` knob/effect. Forward-ref throughput/latency 第 8 课.

### L23 — 23-dp-controller-and-pp-scheduling.html / "DP 控制器与 PP 调度"
**Scope:** scheduling across processes. Data Parallel: a `DataParallelController` fans requests across N
independent scheduler replicas (round-robin / load-based) — each replica is a full runtime. Pipeline
Parallel: `SchedulerPPMixin.event_loop_pp` runs a pipelined loop across PP stages (micro-batches in
flight across stages). How these compose with TP (deep dive in Part 10, 第 46 课). **Cited:**
`srt/managers/data_parallel_controller.py ::DataParallelController` (or `scheduler_pp_mixin.py
::SchedulerPPMixin.event_loop_pp`). **Read:** data_parallel_controller.py (DataParallelController ~127,
round_robin_scheduler ~606, event_loop ~648), scheduler_pp_mixin.py (event_loop_pp ~68). **Diagrams:** a
`layers`/`flow` DP controller → N scheduler replicas; a `vflow`/`timeline` PP micro-batch pipeline across
stages; a `table.t` DP vs PP vs TP (what's replicated/split); a `cols` DP-vs-PP. Forward-ref parallelism 第 46 课.

## Wiring & DoD
- New module `src/part5.py` (`LESSON_18..23`); `registry.py` imports `part5` + 6 keys; `shell.PAGES` +
  `SUBTITLES` += 6; `quizzes.QUIZZES` += 6. Filenames as above. Part label
  "第五部分 · 调度器 / Part 5 · The scheduler".
- All validators 0 err / 0 warn; no-diff; index pill "共 23 课 · 5 个部分"; nav L17↔L18…L23.
- Source-accurate (`file ::symbol`); the event-loop and overlap mechanics must match scheduler.py exactly.
