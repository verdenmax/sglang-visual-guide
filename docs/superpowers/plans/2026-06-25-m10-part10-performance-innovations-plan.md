# M10 — Part 10 Performance Innovations (L43–L48) — Plan

> Execute with subagent-driven-development. One subagent per lesson, model claude-opus-4.8, sync,
> high effort. Two-stage review (research + code-review, opus-4.8 max, long context) after exec.
> Edit large files incrementally.

**Goal:** add Part 10 (L43–48) — speculative decoding, EAGLE, PD disaggregation, TP/PP/EP/DP, EPLB,
structured outputs. **Companion spec:** `docs/superpowers/specs/2026-06-25-m10-part10-performance-innovations-spec.md`.

## Verified citations (pre-fetched — pass to subagents verbatim)
- L43 `python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker` — abstract `target_worker()`
  + `draft_worker()` split; `spec_info.py ::SpeculativeAlgorithm` enum = EAGLE/EAGLE3/NGRAM/STANDALONE/
  DFLASH/FROZEN_KV_MTP/NONE (pluggable family).
- L44 `python/sglang/srt/speculative/eagle_info.py ::EagleVerifyInput` — tree verify payload:
  `draft_token, custom_mask, retrieve_index, retrieve_next_token, retrieve_next_sibling, spec_steps,
  topk, draft_token_num`; worker `eagle_worker_v2.py ::EAGLEWorkerV2`.
- L45 `python/sglang/srt/disaggregation/base/conn.py ::BaseKVSender` — `init`/`send`/`poll`→`KVPoll`,
  `get_transfer_metric`→`KVTransferMetric`; peer `BaseKVReceiver`.
- L46 `python/sglang/srt/distributed/parallel_state.py ::GroupCoordinator` — `rank, ranks, world_size,
  local_rank, rank_in_group`; `get_tensor_model_parallel_world_size/rank`, `get_pipeline_model_parallel_*`.
- L47 `python/sglang/srt/eplb/eplb_manager.py ::EPLBManager` — `__init__(model_runner)`,
  `on_forward_pass_end()`, `rebalance()`; `eplb/expert_location.py`, `expert_distribution.py`.
- L48 `python/sglang/srt/constrained/base_grammar_backend.py ::BaseGrammarObject` — `accept_token(token)`,
  `allocate_vocab_mask`, `fill_vocab_mask(vocab_mask, idx)`, `rollback(k)`; jump-forward
  `constrained/outlines_jump_forward.py ::OutlinesJumpForwardMap` (`jump_forward_symbol`).

## Spec-decoding naming (L43/L44 — speculative-naming skill)
`accept_rate` (α, per-draft, excl. bonus); `accept_length` (τ, per verify step, incl. bonus);
`bonus_token` = the +1 target token; `correct_drafts` = drafts that passed; verb `accept` (not accepted).

## Task 0 — orchestrator wiring (do myself)
- [ ] Create `src/part10.py` with `LESSON_43..LESSON_48` placeholders.
- [ ] `registry.py`: `import part10`; add 6 CONTENT keys (filenames below → `part10.LESSON_4X`).
- [ ] `shell.py`: append 6 PAGES 5-tuples (anchor after L42) + 6 SUBTITLES; label
  "第十部分 · 性能创新专题" / "Part 10 · Performance innovations".
- [ ] Build + confirm pill "共 48 课 · 10 个部分", nav L42↔L43.

Filenames: `43-speculative-decoding-overview.html`, `44-eagle-and-next-gen.html`,
`45-pd-disaggregation.html`, `46-tp-pp-ep-dp-parallelism.html`, `47-large-scale-ep-and-eplb.html`,
`48-structured-outputs-and-jump-forward.html`.

## Tasks 1–6 — per-lesson subagents (L43…L48)
Each: replace that lesson's placeholder in `src/part10.py` + append its `quizzes.py` entry. Proven
prompt template (edit-first, no narration): exact placeholder line; verified `file ::symbol` + the
real condensed code; house markup (card analogy/macro/key with `<div class="tag">…</div>`; codefile
cf-head `dot`/`path`/`ln`); allowed CSS palette; "real prose ≥3600 CJK zh, identical zh/en inventory,
EXACTLY 4 diagram blocks, NO SVG, NOT timeline, docstrings→`#`, don't over-research"; VERIFY cmds.
Retry on truncation; draft+expand fallback after 3 fails. Diagram picks per the spec's per-lesson list.

- [ ] **T1 L43** spec-decode overview — BaseSpecWorker. (spec naming: accept_rate/accept_length/bonus_token)
- [ ] **T2 L44** EAGLE & next-gen — EagleVerifyInput (token tree, tree attention).
- [ ] **T3 L45** PD disaggregation — BaseKVSender (init/send/poll, KV transfer).
- [ ] **T4 L46** TP/PP/EP/DP — GroupCoordinator (one abstraction, four axes).
- [ ] **T5 L47** large-scale EP & EPLB — EPLBManager (rebalance loop).
- [ ] **T6 L48** structured outputs & jump-forward — BaseGrammarObject (mask + jump-forward). Closes Part 10.

## Task 7 — verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py &&
  python3 check_links.py` → 0/0; `git diff` no-diff. Commit (`verdenmax`, Co-authored-by Copilot).

## Task 8 — two-stage review + fixes
- [ ] research (concept/spec/completeness, esp. spec-decoding naming correctness + EAGLE tree + PD KV
  transfer + parallelism axes + EPLB + grammar/jump-forward accuracy) + code-review (markup/parity/
  escaping/wiring/quiz schema), parallel, opus-4.8 max, long context. Fix → re-verify → commit.

## DoD
48 lessons; Part 10 nav + pill correct; validators 0/0; no-diff; citations source-accurate; spec
scope met; spec-decoding identifiers follow the skill.
