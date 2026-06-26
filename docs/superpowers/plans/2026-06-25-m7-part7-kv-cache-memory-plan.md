# M7 — Part 7 KV Cache & Memory (L29–L32) — Plan

> One sync subagent per lesson, then two-stage review. Spec:
> `specs/2026-06-25-m7-part7-kv-cache-memory-spec.md`.

**Goal:** Add L29–L32 (the memory subsystem) as `src/part7.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source:** `~/course/sglang/python/sglang/srt/mem_cache/`.

## Common recipe (every lesson)
Model on part1–6. Each `LESSON_NN`: lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` → **EXACTLY 4 visual
blocks/lang, identical zh/en** (cols/vflow/flow/layers/cellgroup/`<table class="t">`; NOT `timeline`;
no SVG) → ONE cited `.codefile` (`file ::symbol`; `"""docstring"""` in `<pre>` → `#` comment) → 本课要点
card → quiz (3 MCQ + 2 open). **zh ≥ 3500 CJK.** Only `shell.CSS` classes. Smoke-build after each.

## Task 0 — module + wiring (orchestrator)
- [ ] Create `src/part7.py` (`LESSON_29..32` placeholders); `import part7` in `registry.py`; add 4 PAGES +
  4 SUBTITLES + 4 CONTENT keys. Part label "第七部分 · KV 缓存与内存 / Part 7 · KV cache & memory". Filenames:
  `29-radixattention-implementation.html`, `30-paged-memory-pools.html`, `31-hicache-tiering.html`,
  `32-eviction-and-hit-rate.html`.

## Per-lesson tasks (read source, author per spec, add quiz, smoke-build)
- [ ] **L29** RadixAttention implementation (DEEPER than L07 — data structures, not concept). Cite
  `srt/mem_cache/radix_cache.py ::RadixCache.match_prefix` (TreeNode ~223, RadixCache ~286, match_prefix ~360,
  insert ~420, evict ~568, inc/dec_lock_ref ~597/612). Analogy: a **shared trie/filesystem of folders** —
  each node is a folder holding a run of tokens + a pointer to where the KV is stored; you walk down matching,
  and split a folder when paths diverge. Diagrams: vflow match→split→insert; cellgroup/layers a TreeNode
  (key/children/value/lock_ref); flow two reqs share-then-diverge; table.t TreeNode fields→role.
- [ ] **L30** Paged memory pools. Cite `srt/mem_cache/memory_pool.py ::ReqToTokenPool` (ReqToTokenPool ~230,
  MHATokenToKVPool ~1069; allocator/base.py ::BaseTokenToKVPoolAllocator ~27). Analogy: a **coat-check +
  locker room** — req_to_token is the ticket→locker-numbers map; token_to_kv is the actual lockers holding
  the K/V; the allocator hands out free lockers. Diagrams: layers req_to_token→token_to_kv; flow alloc-on-admit
  →write→free-on-evict; table.t the two pools→what each maps; cellgroup req→token→KV indirection. Tie 第 6/29 课.
- [ ] **L31** HiCache tiering. Cite `srt/mem_cache/hiradix_cache.py ::HiRadixCache` (~75) or
  `srt/managers/cache_controller.py ::HiCacheController` (~209). Analogy: **desk → drawer → warehouse** —
  hot prefixes on the GPU desk, warm ones in the CPU drawer, cold in the disk warehouse; a clerk (controller)
  fetches up / files down in the background. Diagrams: layers three tiers GPU/CPU/disk; vflow evict→writeback-down
  / match→prefetch-up; cols HBM-only vs HiCache; table.t tier→speed/capacity/role.
- [ ] **L32** Eviction & hit rate. Cite `srt/mem_cache/evict_policy.py ::LRUStrategy` (EvictionStrategy ~10,
  LRUStrategy ~16) or `radix_cache.py ::RadixCache.evict` (~568). Analogy: a **library weeding the least-used
  books**, but never a book someone's currently reading (lock_ref>0). Diagrams: table.t strategies (LRU/LFU/
  FIFO/MRU→when); vflow evict (pick leaf by priority→skip if locked→free KV); cellgroup tree shrinking under
  pressure (locked nodes survive, `cell hl`); cols evict-vs-keep (recompute cost vs HBM). Tie hit-rate to
  cache-aware scheduling 第 20 课 + RadixAttention 第 7/29 课. Closes Part 7.

## Verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py`
  → 0 err / 0 warn; pill "共 32 课 · 7 个部分"; nav L28↔L29↔…↔L32; no-diff.
- [ ] One commit: `M7: Part 7 KV cache & memory — L29 RadixAttention impl, L30 paged pools, L31 HiCache, L32 eviction` (+ Co-authored-by trailer).

## Guardrails
- Cite `file ::symbol`; the radix tree (TreeNode/match/split/lock) + pool indirection (tree node value = pool
  index) must match the code. L29 goes DEEPER than L07 (don't re-explain the concept; show the data structures).
- zh ≥3500 CJK; zh/en identical diagram inventory; no `timeline`. Don't touch `docs/`, earlier parts, pipeline, reference repo.
