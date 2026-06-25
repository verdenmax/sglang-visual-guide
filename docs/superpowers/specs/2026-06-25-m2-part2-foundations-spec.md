# M2 — Part 2 Foundations (L04–L08) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M2. Builds on M0/M1 (L01–03 exist).

## Goal
Lay the **inference foundations** every later part assumes: how autoregressive decoding + the KV
cache work, why continuous batching and paged KV exist, what RadixAttention adds (SGLang's
signature), and the throughput-vs-latency tension that drives scheduling. Mostly conceptual, but
L05–L07 cite real SGLang code; L07 is the pivotal "aha" lesson.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 2–5 `<h2>` with ≥3 visual blocks/lang → one cited `.codefile`
(`file + symbol`) → 本课要点 card → quiz (2–4 MCQ + 1–2 open). zh ≥3000 CJK (~4000), zh/en identical
diagram inventory, only CSS classes in `shell.CSS`. Validators 0 err / 0 warn. Forward-refs map to
the real lesson (design spec §6); use the en-dash-aware range form freely (`第 29–32 课`).

## Lessons

### L04 — 04-autoregressive-and-kv-cache.html / "Autoregressive decode & the KV cache"
**Scope:** tokens generated one at a time; each step attends to all previous tokens; the KV cache
stores past K/V so step *t* is O(t) not O(t²); **prefill** (process the whole prompt, fill the
cache) vs **decode** (one token/step, append to cache); memory cost of the cache (per-token bytes ×
layers × heads). The bottleneck framing: decode is **memory-bandwidth bound**.
**Diagrams:** a `vflow`/`trace` of decode steps reusing cached K/V; a `cellgroup`/`table.t` of cache
growth; a `cols` with-vs-without cache. **Cited:** `srt/mem_cache/memory_pool.py` (the KV pool class
that physically stores K/V, e.g. the token-to-KV pool) — show the shape/role, not internals.
**Read:** `srt/mem_cache/memory_pool.py`, `srt/model_executor/forward_batch_info.py` (prefill/decode mode).

### L05 — 05-continuous-batching.html / "Continuous batching"
**Scope:** static/padded batching wastes the GPU (everyone waits for the slowest, finished slots
idle); continuous (a.k.a. in-flight / iteration-level) batching re-forms the batch **every step** —
new requests join, finished ones leave. Where SGLang does this: the scheduler picks a fresh batch
each iteration. **Cited:** `srt/managers/scheduler.py ::get_next_batch_to_run` (or `schedule_batch.py`
::ScheduleBatch). **Read:** `srt/managers/schedule_batch.py`, `scheduler.py` (`get_next_batch_to_run`).
**Diagrams:** a `timeline` static vs continuous (slots filling/leaving); a `flow` per-step re-batch;
a `table.t` static-vs-continuous tradeoffs.

### L06 — 06-paged-attention-and-paged-kv.html / "PagedAttention & paged KV"
**Scope:** contiguous per-request KV reservations fragment HBM and cap concurrency; paging splits KV
into fixed-size blocks/pages indexed by a table (like OS virtual memory), so memory is allocated on
demand and shared. In SGLang: the token-to-KV-pool allocator + a page/req-to-token mapping. **Cited:**
`srt/mem_cache/allocator/` (the paged allocator) or `memory_pool.py` (req_to_token / token_to_kv).
**Read:** `srt/mem_cache/allocator/`, `srt/mem_cache/memory_pool.py`. **Diagrams:** a `layers`/`cellgroup`
of pages + index table; a `cols` contiguous-vs-paged; a `table.t` fragmentation/over-allocation.

### L07 — 07-radixattention-and-prefix-caching.html / "RadixAttention & prefix caching" (signature)
**Scope:** the headline. Many requests share prefixes (system prompt, few-shot, chat history);
RadixAttention keeps **all KV in a radix tree** keyed by token sequence, so a shared prefix's KV is
computed once and **reused across requests** (and across turns). Insert/match/split of radix nodes;
eviction by LRU + refcount; automatic (no user hints). This is why SGLang is fast on real workloads.
**Cited:** `srt/mem_cache/radix_cache.py ::RadixCache` (match_prefix / insert) and/or
`base_prefix_cache.py`. **Read:** `srt/mem_cache/radix_cache.py`, `base_prefix_cache.py`,
`evict_policy.py`. **Diagrams:** a `fig`/SVG radix tree with a shared prefix highlighted; a `vflow`
match→reuse→insert; a `cellgroup` two requests sharing a prefix; a `table.t` hit/miss economics.
This lesson may run richer (more diagrams) — it is the part's centerpiece.

### L08 — 08-throughput-vs-latency.html / "Throughput vs latency"
**Scope:** the metrics (TTFT = time to first token, ITL/TPOT = inter-token latency, throughput =
tokens/s, goodput under SLA); why batching trades latency for throughput; prefill-heavy vs
decode-heavy; the knobs (max running requests, max tokens, chunked prefill — forward-ref). The
scheduling tension that the whole runtime balances. **Cited:** `srt/managers/schedule_policy.py`
(the policy that orders the queue) or `srt/server_args.py` (the knobs). **Read:** `schedule_policy.py`,
`server_args.py` (max_running_requests, mem_fraction, chunked_prefill_size). **Diagrams:** a `cols`
throughput-vs-latency; a `timeline` batch-size effect; a `table.t` metrics glossary.

## Wiring & DoD
- New module `src/part2.py` exporting `LESSON_04..LESSON_08`; `registry.py` imports `part2` and maps
  the 5 files; `shell.PAGES` + `SUBTITLES` += 5 Part 2 entries; `quizzes.QUIZZES` += 5.
- Filenames: `04-autoregressive-and-kv-cache.html`, `05-continuous-batching.html`,
  `06-paged-attention-and-paged-kv.html`, `07-radixattention-and-prefix-caching.html`,
  `08-throughput-vs-latency.html`. Part label "第二部分 · 推理前置基础 / Part 2 · Foundations".
- All four validators 0 err / 0 warn; no-diff rebuild; index pill "共 8 课 · 2 个部分"; nav L03↔L04…L08.
- Source-accurate (cite `file + symbol`); concepts that are general LLM-serving knowledge are fine to
  explain from first principles, but anything SGLang-specific must match the code.
