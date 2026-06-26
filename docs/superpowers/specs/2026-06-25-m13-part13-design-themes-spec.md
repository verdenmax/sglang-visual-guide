# M13 — Part 13 Design Themes / Synthesis (L58–L63) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec §P13, roadmap M13. Builds on M0–M12 (L01–57 exist). FINAL content part.

## Goal
Synthesis. Six reflective "theme" lessons that zoom OUT from individual components to the recurring
DESIGN PRINCIPLES that make SGLang fast and extensible — each principle threaded through many earlier
lessons. Not new mechanisms; new PERSPECTIVE. This part turns "I know the parts" into "I see why they
were designed this way." Module: `src/part13.py`, lessons L58–63 (6). L63 closes the whole guide.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy (`card analogy`, `🔌 生活类比` / `🔌 Analogy`) → 🌍 macro (`card macro`, `🌍 宏观理解` /
`🌍 The big picture`) → 3–4 `<h2>` with EXACTLY 4 visual blocks/lang (identical zh/en; from cols/flow/
vflow/layers/cellgroup/`table.t`/step; NO `<svg>`, NOT `timeline`) → one cited `.codefile` (`cf-head`
dot/path/ln; escape `<`/`>`/`&`; docstrings→`#`) → key card (`card key`, `📌 本课要点` / `📌 Key points`)
→ quiz (3 MCQ + 2 open). **zh ≥3600 CJK.** Only `shell.CSS` classes. Cite REAL symbols (re-citing a
symbol introduced earlier is fine — these are synthesis lessons; the value is the cross-lesson web).
**Theme-lesson emphasis:** at least one diagram per lesson should be a `table.t` or `cols` that maps
the principle across the lessons where it recurs (the "where you've seen this" web). Heavy use of
backward-refs by lesson number (validate against shell.py numbering).

## Lessons

### L58 — 58-radixattention-as-a-first-class-idea.html / "RadixAttention 作为一等公民"
**Theme:** prefix sharing is not a bolt-on optimization — it's a FIRST-CLASS organizing idea that
shows up at every layer. The same "share identical prefixes via a radix tree" concept appears as: the
DSL's fork/join prefix sharing (第11课), the prefix-cache concept (第7课), the radix-tree
implementation in the cache (第29课), the paged-KV pages it points at (第30课/第6课), cache-aware
SCHEDULING that prefers high-prefix-hit requests (第20课), and HiCache tiering of that cache (第31课).
SGLang elevates "the KV cache is a TREE of shared prefixes" to a core data structure the whole engine
is built around. The lesson is the synthesis of that thread. **Cited:**
`python/sglang/srt/mem_cache/radix_cache.py ::RadixCache` (the first-class data structure). **Read:**
`mem_cache/radix_cache.py` (RadixCache), recall 第7/29/20/31课. **Diagrams:** a `table.t` "where
RadixAttention shows up" (DSL fork-join 第11 / concept 第7 / impl 第29 / scheduling 第20 / tiering 第31);
a `layers`/`cellgroup` a radix tree with shared-prefix nodes; a `cols` bolt-on optimization vs
first-class data structure; a `flow` request → match longest prefix → reuse KV → only compute the new
suffix. Tie all listed lessons.

### L59 — 59-zero-overhead-scheduling.html / "零开销调度的哲学"
**Theme:** the CPU must never make the GPU wait. SGLang's scheduler is engineered so that scheduling,
tokenization, and detokenization OVERLAP with GPU compute rather than serializing before/after it. The
thread: the event loop (第18课), the zero-overhead overlap scheduler (第21课, `enable_overlap`),
chunked prefill interleaving prefill with decode (第22课), CUDA graphs removing per-step launch
overhead (第27课), and the IPC/process split (第14/16课) that lets the CPU-side work run on a different
process than the GPU-side forward. The principle: identify every CPU bubble and hide it behind GPU
work. **Cited:** `python/sglang/srt/managers/scheduler.py ::Scheduler` (the loop + `enable_overlap`).
**Read:** `managers/scheduler.py` (Scheduler, enable_overlap), `managers/overlap_utils.py`, recall
第18/21/22/27课. **Diagrams:** a `flow`/`vflow` serial (CPU then GPU then CPU) vs overlapped (CPU work
hidden under GPU forward); a `table.t` CPU bubble → how SGLang hides it (→ lesson); a `cols`
overlap-on vs overlap-off; a `layers` the pipeline of overlapped stages. Tie 第18/21/22/27/14课.

### L60 — 60-two-workloads-one-engine.html / "两种负载，一套引擎（PD 视角）"
**Theme:** prefill and decode are fundamentally DIFFERENT workloads (compute-bound vs bandwidth-bound,
第4课), and much of SGLang's design is about serving both well. The thread: the throughput/latency
trade-off they create (第8课), chunked prefill that time-shares one GPU between them (第22课), and PD
disaggregation that gives each its OWN pool + transfers KV between them (第45课). Seeing prefill-vs-
decode as the root tension explains many decisions. The synthesis: one engine, two opposite workloads,
reconciled by scheduling (co-located) or by disaggregation (separated). **Cited:**
`python/sglang/srt/disaggregation/base/conn.py ::BaseKVManager` (the seam that lets the two pools be
separated). **Read:** `disaggregation/base/conn.py`, recall 第4/8/22/45课. **Diagrams:** a `cols`
prefill (compute-bound, big parallel) vs decode (bandwidth-bound, skinny) — the root contrast; a
`table.t` design response → which lesson (chunked prefill 第22 / PD disaggregation 第45 / scheduling
第20); a `flow` request → prefill workload → KV → decode workload; a `vflow` co-located (time-share)
vs disaggregated (separate pools). Tie 第4/8/22/45课.

### L61 — 61-draft-for-parallel-verify.html / "用草稿换并行验证（投机的本质）"
**Theme:** the deep idea behind speculative decoding is "trade cheap parallel-izable work for a
sequential bottleneck." Decode is sequential and bandwidth-bound (第4课); spec decoding spends a cheap
draft to turn "generate k tokens sequentially" into "verify k tokens in ONE parallel forward" (第43课),
and EAGLE's token tree pushes the parallel-verify even further (第44课). It's the same shape as other
SGLang wins: do speculative/parallel work to dodge a serial wall (cf. CUDA-graph capturing a fixed
sequence 第27课, overlap scheduling 第59课). The principle: convert latency-bound serial steps into
throughput-bound parallel ones. **Cited:**
`python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker` (draft + target). **Read:**
`speculative/base_spec_worker.py`, recall 第43/44/4课. **Diagrams:** a `flow` sequential decode (1
token/step) vs draft→parallel-verify (k tokens/step); a `cols` the trade (cheap draft compute ↔ saved
sequential steps); a `table.t` "serial wall → parallel trick" across SGLang (spec 第43 / tree 第44 /
CUDA graph 第27 / overlap 第59); a `vflow` propose → verify-in-parallel → accept prefix + bonus. Tie
第43/44/4/27课.

### L62 — 62-everything-is-pluggable.html / "一切皆可插拔"
**Theme:** the guide's recurring motif, made explicit. SGLang keeps a stable CORE (scheduler, model
loop, memory) and pushes variability behind PLUGGABLE INTERFACES, so new hardware/algorithms slot in
without touching the engine. The web: the attention backend ABC (第33课), the platform abstraction for
hardware (第42课), quantization methods (第35课), the speculative-algorithm registry (第43课), the
parallelism axes behind one GroupCoordinator (第46课), grammar backends (第48课), KV-transfer
connectors (第45课), and "write a model" against stable layer APIs (第26课). One pattern — program to
an interface, select an implementation at deploy time — repeated everywhere. **Cited:**
`python/sglang/srt/layers/attention/base_attn_backend.py ::AttentionBackend` (the archetypal pluggable
ABC). **Read:** `layers/attention/base_attn_backend.py`, recall 第33/42/35/43/46/48/45/26课.
**Diagrams:** a `table.t` pluggable seam → its interface → examples → lesson (attention/hardware/quant/
spec/parallel/grammar/KV-transfer); a `layers` stable core vs pluggable edges; a `cols` hard-coded vs
program-to-an-interface; a `flow` request → core → (selected impl at the seam) → result. Tie all listed.

### L63 — 63-built-for-throughput.html / "为吞吐而生（全书收束）"
**Theme:** the final synthesis — almost every design choice serves ONE north star: keep the GPU busy
doing useful token work. Batching amortizes weight reads (第5课), paging eliminates KV fragmentation so
more requests fit (第6/30课), prefix caching/RadixAttention skips redundant compute (第7/29课), the
overlap scheduler removes CPU bubbles (第21/59课), CUDA graphs cut launch overhead (第27课), kernels +
fusion maximize bandwidth use (第38/41课), and quantization + parallelism fit bigger models on the
hardware (第35/46课). Read together, SGLang is a throughput machine; latency features (chunked prefill,
spec decoding) are about not sacrificing the user while maximizing the system. End with a guide-wide
CLOSING note: from "life of a request" (第3课) to here, you can now read any part as a deliberate
throughput choice. **Cited:** `python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy` (the
cache-aware policy that maximizes batch + prefix-hit throughput). **Read:** `managers/schedule_policy.py`
(SchedulePolicy, CacheAwarePolicy), recall the listed lessons. **Diagrams:** a `table.t` design choice
→ the throughput it buys (→ lesson) — the capstone map; a `flow` the full request path annotated with
the throughput lever at each stage (第3课 arc revisited); a `cols` throughput levers vs latency
safeguards; a `layers` the stack as concentric throughput optimizations. Tie the whole guide. Add a
final `card` (bare, `🏁 全书收束 / The end`) wrapping up the journey. Closes Part 13 & the guide.

## Wiring & DoD
- New module `src/part13.py` (`LESSON_58..63`); `registry.py` imports `part13` + 6 keys; `shell.PAGES`
  + `SUBTITLES` += 6; `quizzes.QUIZZES` += 6. Part label "第十三部分 · 设计专题综合" /
  "Part 13 · Design themes (synthesis)".
- Filenames: `58-radixattention-as-a-first-class-idea.html`, `59-zero-overhead-scheduling.html`,
  `60-two-workloads-one-engine.html`, `61-draft-for-parallel-verify.html`,
  `62-everything-is-pluggable.html`, `63-built-for-throughput.html`.
- All validators 0 err / 0 warn; no-diff; pill "共 63 课 · 13 个部分"; nav L57↔L58…L63.
- Source-accurate: cite real symbols (re-citing earlier symbols is fine); all backward-ref lesson
  numbers match shell.py.
