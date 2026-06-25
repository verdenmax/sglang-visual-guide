# M2 — Part 2 Foundations (L04–L08) — Plan

> **For agentic workers:** one implementer subagent builds all five lessons, then two-stage review.
> Spec: `specs/2026-06-25-m2-part2-foundations-spec.md`. Steps use checkbox (`- [ ]`).

**Goal:** Add L04–L08 (the inference foundations) as a new module `src/part2.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source (read-only):** `~/course/sglang/`.

## Common authoring recipe (applies to every lesson below)
Model the HTML on the existing `src/part1.py` lessons (same CSS classes, diagram markup, cited-code
style, bilingual parity, 本课要点 card). Each `LESSON_NN = {"zh": r"""...""", "en": r"""..."""}` MUST have:
lead → 🔌 analogy card → 🌍 macro card → 2–5 `<h2>` → **≥3 visual blocks per language** (identical
zh/en inventory; pick from `layers`/`vflow`/`flow`/`cols`/`cellgroup`/`timeline`/`trace`/`<table class="t">`/
`.fig`+SVG) → exactly ONE cited `.codefile` (`file + symbol` caption, small snippet) → 本课要点/Key
points card. **zh ≥ 3000 CJK (aim ~4000).** Use ONLY classes defined in `shell.CSS`. Forward-refs
("第 N 课") must map to the real lesson (design spec §6); ranges like `第 29–32 课` are fine.

After EACH lesson, smoke-build (`cd src && python3 build.py && python3 check_html.py`) and fix that
lesson's ERROR/WARN before starting the next, so problems don't pile up.

---

## Task 0 — Create the module + wiring skeleton
**Files:** create `src/part2.py`; modify `src/registry.py`, `src/shell.py`.
- [ ] Create `src/part2.py` with a module docstring and the five `LESSON_04..08` placeholders to be
  filled by the tasks below. Add `import part2` to `src/registry.py`.
- [ ] Add all five `PAGES` entries (after the L03 tuple) and five `SUBTITLES` entries in `shell.py`,
  and five `CONTENT` entries in `registry.py`, using these filenames + titles (Part label
  "第二部分 · 推理前置基础" / "Part 2 · Foundations"):
```python
("04-autoregressive-and-kv-cache.html", "自回归生成与 KV 缓存", "Autoregressive decode & the KV cache",
 "第二部分 · 推理前置基础", "Part 2 · Foundations"),
("05-continuous-batching.html", "连续批处理", "Continuous batching",
 "第二部分 · 推理前置基础", "Part 2 · Foundations"),
("06-paged-attention-and-paged-kv.html", "PagedAttention 与分页 KV", "PagedAttention & paged KV",
 "第二部分 · 推理前置基础", "Part 2 · Foundations"),
("07-radixattention-and-prefix-caching.html", "RadixAttention 与前缀缓存", "RadixAttention & prefix caching",
 "第二部分 · 推理前置基础", "Part 2 · Foundations"),
("08-throughput-vs-latency.html", "吞吐 vs 延迟", "Throughput vs latency",
 "第二部分 · 推理前置基础", "Part 2 · Foundations"),
```
  (Author the matching one-line zh/en `SUBTITLES` summaries yourself.)

## Task 1 — L04 自回归生成与 KV 缓存 / Autoregressive decode & the KV cache
- [ ] Read `~/course/sglang/python/sglang/srt/mem_cache/memory_pool.py` (the KV pool that stores
  K/V) and `srt/model_executor/forward_batch_info.py` (prefill vs decode mode). Cite one real symbol.
- [ ] Author `part2.LESSON_04` (zh+en) per the recipe. Analogy: doing long multiplication and
  **writing down partial results** so you never recompute them (the cache). Cover: one-token-at-a-time
  generation, attention over all past tokens, why caching K/V makes step *t* O(t) not O(t²),
  prefill-fills vs decode-appends, the per-token memory cost, decode = bandwidth-bound. Diagrams:
  `vflow`/`trace` of decode reusing cache; `table.t` cache growth; `cols` with-vs-without cache.
  Cited `.codefile`: `srt/mem_cache/memory_pool.py` (the KV pool class). Quiz: 2–4 MCQ + 1–2 open.

## Task 2 — L05 连续批处理 / Continuous batching
- [ ] Read `srt/managers/schedule_batch.py` (`ScheduleBatch`) and `srt/managers/scheduler.py`
  (`get_next_batch_to_run`). Cite one real symbol.
- [ ] Author `part2.LESSON_05`. Analogy: a **carpool/shuttle that picks up and drops off riders at
  every stop** instead of waiting until full and going only when everyone shares the destination.
  Cover: padded/static batching waste (slowest-wins, idle finished slots); continuous / iteration-level
  batching re-forms the batch each step; SGLang re-picks a batch per scheduler iteration. Diagrams:
  `timeline` static-vs-continuous; `flow` per-step re-batch; `table.t` tradeoffs. Cited `.codefile`:
  `scheduler.py ::get_next_batch_to_run` (or `ScheduleBatch`). Quiz.

## Task 3 — L06 PagedAttention 与分页 KV / PagedAttention & paged KV
- [ ] Read `srt/mem_cache/allocator/` (the paged KV allocator) and `srt/mem_cache/memory_pool.py`
  (`req_to_token` / `token_to_kv` pools). Cite one real symbol.
- [ ] Author `part2.LESSON_06`. Analogy: **OS virtual memory / a hotel giving out rooms by the
  night** — fixed-size pages allocated on demand and indexed by a table, instead of reserving a whole
  contiguous wing per guest. Cover: contiguous per-request KV fragments HBM and caps concurrency;
  paging = fixed-size blocks + index table; allocate on demand, enabling sharing (sets up RadixAttention
  next lesson). Diagrams: `layers`/`cellgroup` of pages + index table; `cols` contiguous-vs-paged;
  `table.t` fragmentation/over-allocation. Cited `.codefile`: the paged allocator or KV pool. Quiz.
  Forward-ref RadixAttention → 第 7 课.

## Task 4 — L07 RadixAttention 与前缀缓存 / RadixAttention & prefix caching  (★ centerpiece)
- [ ] Read `srt/mem_cache/radix_cache.py` (`RadixCache`: `match_prefix`, `insert`, node split),
  `srt/mem_cache/base_prefix_cache.py`, `srt/mem_cache/evict_policy.py`. Cite a real symbol
  (`radix_cache.py ::RadixCache.match_prefix` or `::RadixCache`).
- [ ] Author `part2.LESSON_07` — richer than the others (this is the part's "aha"). Analogy: a
  **shared filing tree / autocomplete trie** where everyone who starts with the same words reuses the
  already-written pages. Cover: real requests share prefixes (system prompt, few-shot, chat history);
  RadixAttention stores ALL KV in a radix tree keyed by token ids; a shared prefix's KV is computed
  once and reused across requests and turns; node match/insert/split; eviction by LRU + refcount;
  it's automatic (no user hints). Diagrams (≥3/lang, aim 4–5): a `.fig`+SVG **radix tree** with a
  shared prefix highlighted; a `vflow` match→reuse→insert→evict; a `cellgroup` two requests sharing a
  prefix; a `table.t` hit/miss economics. Cited `.codefile`: `radix_cache.py ::RadixCache`. Quiz
  (make at least one MCQ about *why a tree and not a hash map*). This lesson is referenced by L01/L03.

## Task 5 — L08 吞吐 vs 延迟 / Throughput vs latency
- [ ] Read `srt/managers/schedule_policy.py` (queue ordering) and `srt/server_args.py`
  (`max_running_requests`, `mem_fraction_static`, `chunked_prefill_size`). Cite one real symbol.
- [ ] Author `part2.LESSON_08`. Analogy: a **restaurant choosing between serving each table fast vs
  feeding the most people per hour** (latency vs throughput). Cover: TTFT, ITL/TPOT, throughput,
  goodput-under-SLA; batching trades latency for throughput; prefill-heavy vs decode-heavy; the knobs
  (max running requests / max tokens / chunked prefill → forward-ref 第 22 课). Diagrams: `cols`
  throughput-vs-latency; `timeline` batch-size effect; `table.t` metrics glossary. Cited `.codefile`:
  `schedule_policy.py` or `server_args.py`. Quiz. This closes Part 2 and motivates the scheduler (Part 5).

## Verify (all five lessons)
- [ ] Full build + validate:
```bash
cd ~/course/sglang-visual-guide/src
python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py
```
  Expected: `check_html.py` → "0 error(s), 0 warning(s)"; links resolve; index pill
  **"共 8 课 · 2 个部分"**; nav chain L03↔L04↔…↔L08 resolves. Fix every ERROR/WARN.
- [ ] No-diff guard: `cd ~/course/sglang-visual-guide && python3 src/build.py >/dev/null && git add -A && git status --short` — second build adds nothing new.

## Commit
- [ ] One commit:
```bash
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "M2: Part 2 foundations — L04 KV cache, L05 continuous batching, L06 paged KV, L07 RadixAttention, L08 throughput vs latency

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Guardrails
- Cite `file + symbol`, never bare line numbers; small snippets; explain don't dump.
- L04/L08 lean conceptual (general LLM-serving knowledge is fine to teach from first principles), but
  any SGLang-specific claim (RadixAttention tree, the pools, the scheduler picking batches) must match
  the code in `~/course/sglang/`.
- Forward-refs map to the REAL lesson (e.g. chunked prefill = 第 22 课, scheduler = 第 18–23 课,
  attention backends = 第 33 课). zh ≥ 3000 CJK; zh/en identical diagram inventory.
- Don't touch `docs/`, Part 1 lessons, the pipeline scripts, or the reference repo. SGLang is Apache-2.0.
