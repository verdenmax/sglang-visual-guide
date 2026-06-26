# M7 — Part 7 KV Cache & Memory (L29–L32) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M7. Builds on M0–M6 (L01–28 exist).

## Goal
The memory subsystem in depth. L07 (Part 2) introduced RadixAttention as a *concept*; this part opens
the *implementation*: the radix tree code (TreeNode/match/insert/split/lock), the paged memory pools and
allocator, HiCache's GPU↔CPU↔disk tiering, and eviction policy + hit-rate economics. Module:
`src/part7.py`, lessons L29–32. Don't repeat L07 — this is the code-level deep dive.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with EXACTLY 4 visual blocks/lang (identical zh/en; no SVG;
NOT `timeline`) → one cited `.codefile` (`file ::symbol`; docstrings in `<pre>` as `#` comments) →
本课要点 card → quiz (3 MCQ + 2 open). **zh ≥3500 CJK.** Only `shell.CSS` classes. Validators 0/0.

## Lessons

### L29 — 29-radixattention-implementation.html / "RadixAttention 实现 / RadixAttention implementation"
**Scope:** the actual `RadixCache`. A `TreeNode` holds a key (token-id run), children (keyed by first
token), the KV value (indices into the pool), a `lock_ref` (in-use count) and last-access time.
`match_prefix` walks children doing longest-prefix match, splitting a node (`_split_node`) when a match
ends mid-edge. `insert` adds the divergent suffix. `inc_lock_ref`/`dec_lock_ref` protect a running req's
prefix from eviction. (This is L07's concept turned into data structures.) **Cited:**
`srt/mem_cache/radix_cache.py ::RadixCache.match_prefix` (or `::TreeNode`). **Read:** radix_cache.py
(TreeNode ~223, RadixCache ~286, match_prefix ~360, insert ~420, evict ~568, inc/dec_lock_ref ~597/612).
**Diagrams:** a `vflow` match→split→insert; a `cellgroup`/`layers` of a TreeNode (key/children/value/lock_ref);
a `flow` of two requests sharing then diverging; a `table.t` TreeNode fields → role.

### L30 — 30-paged-memory-pools.html / "分页内存池 / Paged memory pools"
**Scope:** where KV physically lives. Two pools: `ReqToTokenPool` (per request → its token slots) and the
token→KV pool (`MHATokenToKVPool`, the actual K/V tensors per layer). A `TokenToKVPoolAllocator` hands out
free slots/pages (paged, 第 6 课). The radix tree's node values are indices INTO these pools — so cache
nodes and physical KV are decoupled (the tree is the index; the pool is the storage). Alloc on admit, free
on evict/finish. **Cited:** `srt/mem_cache/memory_pool.py ::ReqToTokenPool` (or `::MHATokenToKVPool`).
**Read:** memory_pool.py (ReqToTokenPool ~230, MHATokenToKVPool ~1069), allocator/base.py
(BaseTokenToKVPoolAllocator ~27), allocator/paged.py. **Diagrams:** a `layers` req_to_token → token_to_kv
pools; a `flow` alloc on admit → write K/V → free on evict; a `table.t` the two pools → what each maps; a
`cellgroup` of req→token→KV indirection. Tie to 第 6 课 (paging) and 第 29 课 (tree points into the pool).

### L31 — 31-hicache-tiering.html / "HiCache 分层缓存 / HiCache tiering"
**Scope:** caching beyond GPU HBM. `HiRadixCache` (a RadixCache subclass) + a `HiCacheController` extend
the prefix cache across THREE tiers: GPU HBM (hot) → CPU host memory (warm) → disk/object store (cold).
Evicted-but-valuable prefixes are written DOWN a tier instead of dropped; a later match can fetch them back
UP. Background threads do the prefetch/writeback so the GPU loop isn't blocked. Big win for long shared
prefixes that don't fit HBM. **Cited:** `srt/mem_cache/hiradix_cache.py ::HiRadixCache` (or
`srt/managers/cache_controller.py ::HiCacheController`). **Read:** hiradix_cache.py (HiRadixCache ~75),
managers/cache_controller.py (HiCacheController ~209). **Diagrams:** a `layers` three tiers
GPU/CPU/disk; a `vflow` evict→writeback-down / match→prefetch-up; a `cols` HBM-only vs HiCache; a `table.t`
tier → speed/capacity/role. Forward-ref HiCache best practices (advanced) and KV pool 第 30 课.

### L32 — 32-eviction-and-hit-rate.html / "缓存淘汰与命中 / Eviction & hit rate"
**Scope:** the economics. The tree can't grow forever; when KV slots run low the scheduler triggers
`evict`. An `EvictionStrategy` (`LRUStrategy` default, also LFU/FIFO/MRU) orders evictable LEAF nodes by
`get_priority`; nodes with `lock_ref > 0` (in use by a running req) are NEVER evictable. Hit rate is the
payoff metric — high prefix sharing (第 7/29 课) → high hit rate → less recompute → higher throughput; the
scheduler is cache-aware (第 20 课) to raise it. The recompute-vs-keep tradeoff. **Cited:**
`srt/mem_cache/evict_policy.py ::LRUStrategy` (or `radix_cache.py ::RadixCache.evict`). **Read:**
evict_policy.py (EvictionStrategy ~10, LRUStrategy ~16, get_priority), radix_cache.py (evict ~568,
inc/dec_lock_ref). **Diagrams:** a `table.t` strategies (LRU/LFU/FIFO/MRU → when); a `vflow` evict: pick
leaf by priority → skip if locked → free its KV; a `cellgroup` of a tree shrinking under pressure
(highlight locked nodes that survive); a `cols` evict-vs-keep (recompute cost vs HBM). Closes Part 7.

## Wiring & DoD
- New module `src/part7.py` (`LESSON_29..32`); `registry.py` imports `part7` + 4 keys; `shell.PAGES` +
  `SUBTITLES` += 4; `quizzes.QUIZZES` += 4. Filenames as above. Part label
  "第七部分 · KV 缓存与内存 / Part 7 · KV cache & memory".
- All validators 0 err / 0 warn; no-diff; index pill "共 32 课 · 7 个部分"; nav L28↔L29…L32.
- Source-accurate (`file ::symbol`); the radix tree + pool indirection must match the code. L29 must go
  deeper than L07 (data structures + match/split/lock), not re-explain the concept.
