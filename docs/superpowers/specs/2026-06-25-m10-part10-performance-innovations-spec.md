# M10 — Part 10 Performance Innovations (L43–L48) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec §P10, roadmap M10. Builds on M0–M9 (L01–42 exist).

## Goal
The "wow" part: the headline performance innovations that make SGLang fast at scale. Speculative
decoding (draft/verify, accept rate) and EAGLE; prefill/decode (PD) disaggregation with KV transfer;
the four parallelism axes TP/PP/EP/DP; large-scale expert parallelism with EPLB; and structured
outputs via a compressed-FSM grammar with jump-forward. Module: `src/part10.py`, lessons L43–48 (6).

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy (`card analogy` + `<div class="tag">🔌 生活类比</div>` zh / `🔌 Analogy` en) → 🌍 macro
(`card macro`, `🌍 宏观理解` / `🌍 The big picture`) → 3–4 `<h2>` with EXACTLY 4 visual blocks/lang
(identical zh/en; from cols/flow/vflow/layers/cellgroup/`table.t`/step+num+sc; NO `<svg>`, NOT
`timeline`) → one cited `.codefile` (`<div class="cf-head"><span class="dot"></span><span
class="path">file ::symbol</span><span class="ln">…</span></div>`; docstrings in `<pre>` as `#`
comments; escape `<`/`>`/`&`) → key card (`card key`, `📌 本课要点` / `📌 Key points`) → quiz (3 MCQ +
2 open). **zh ≥3600 CJK.** Only `shell.CSS` classes. Validators 0/0. Cite REAL symbols only.

**Spec-decoding naming (L43/L44, per speculative-naming skill):** use paper terms `accept_rate`
(α, per-draft acceptance prob, **excludes** bonus) and `accept_length` (τ, avg tokens per verify
step, **includes** bonus); the always-emitted "+1" target token is the `bonus_token`; drafts that
pass verification are `correct_drafts`. Verb form `accept` (never `accepted`).

## Lessons

### L43 — 43-speculative-decoding-overview.html / "投机解码总览 / Speculative decoding overview"
**Scope:** the core idea. Autoregression emits one token per forward (第4课, bandwidth-bound); spec
decoding breaks that by having a cheap DRAFT model propose k tokens, then the expensive TARGET model
VERIFY all k in ONE forward, keeping the longest correct prefix + one `bonus_token`. Net: multiple
tokens per target forward when the draft is good. Key metrics: `accept_rate` (α) and `accept_length`
(τ). SGLang makes the algorithm PLUGGABLE: `SpeculativeAlgorithm` enum (EAGLE / EAGLE3 / NGRAM /
STANDALONE / DFLASH / FROZEN_KV_MTP / NONE) + a `BaseSpecWorker` that owns a `target_worker` and a
`draft_worker`. Verification is lossless (output distribution == target's). **Cited:**
`python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker` (target_worker + draft_worker
abstract split). **Read:** `speculative/base_spec_worker.py`, `speculative/spec_info.py`
(`SpeculativeAlgorithm`, `SpecInput`). **Diagrams:** a `flow` draft k → verify in 1 target forward →
accept prefix + bonus; a `cols` baseline (1 tok/forward) vs spec (m tok/forward); a `table.t`
algorithm family (enum → idea); a `vflow`/`step` propose→verify→accept→roll-forward loop. Forward-ref
EAGLE 第44课. Tie 第28课 sampler (verification uses sampling), 第4课.

### L44 — 44-eagle-and-next-gen.html / "EAGLE 与下一代 / EAGLE & next-gen"
**Scope:** the best-known concrete algorithm. EAGLE drafts at the FEATURE level (reuses the target's
last hidden state) and proposes a TOKEN TREE, not a chain: `topk` candidates per step over
`spec_steps`, giving `draft_token_num` nodes; tree attention verifies the whole tree in one pass with
a `custom_mask` + retrieve indices (`retrieve_index`, `retrieve_next_token`, `retrieve_next_sibling`).
A tree accepts more than a chain → higher `accept_length`. EAGLE3 + the broader family (DFLASH,
FROZEN_KV_MTP, STANDALONE, NGRAM) extend this. **Cited:** `python/sglang/srt/speculative/eagle_info.py
::EagleVerifyInput` (the tree-verify payload: draft_token, custom_mask, retrieve_*). **Read:**
`speculative/eagle_info.py` (EagleVerifyInput/EagleDraftInput), `speculative/eagle_worker_v2.py`
(`EAGLEWorkerV2`). **Diagrams:** a `cellgroup`/`layers` token TREE (root → topk children → grandchildren);
a `cols` chain draft vs tree draft; a `table.t` EagleVerifyInput fields → role; a `flow` reuse hidden
→ build tree → tree-attn verify → accept subtree. Forward-ref spec overview 第43课; tie 第33课 (the
attention backend runs the tree mask), 第8课 hidden states.

### L45 — 45-pd-disaggregation.html / "PD 分离 / Prefill–decode disaggregation"
**Scope:** split the two phases onto SEPARATE machines. Prefill (compute-bound, big parallel pass) and
decode (bandwidth-bound, skinny steps) have opposite resource profiles (第4课); co-locating them makes
each interfere with the other's latency. PD disaggregation runs prefill on one pool and decode on
another, then TRANSFERS the KV cache from prefill→decode over a fast link. SGLang abstracts the
transfer: a `BaseKVSender` (prefill side: `init`→`send`→`poll`→`KVPoll`) and `BaseKVReceiver` (decode
side), with `KVTransferMetric`. Backends: Mooncake / NIXL / ascend. This is how very large
deployments hit tight TTFT + high decode throughput at once. **Cited:**
`python/sglang/srt/disaggregation/base/conn.py ::BaseKVSender` (init/send/poll transfer contract).
**Read:** `disaggregation/base/conn.py` (BaseKVSender/BaseKVReceiver/KVPoll/KVArgs), `disaggregation/`
(decode.py, prefill.py, common/). **Diagrams:** a `flow` request → prefill pool → KV transfer →
decode pool → stream; a `cols` prefill node (compute) vs decode node (bandwidth); a `table.t`
sender/receiver methods → role; a `vflow`/`step` init→send→poll(KVPoll)→received. Tie 第30课 paged KV
(what's transferred), 第4课, 第13课 router. Forward-ref EP 第46/47课.

### L46 — 46-tp-pp-ep-dp-parallelism.html / "四种并行 TP/PP/EP/DP"
**Scope:** the four ways to split a model/workload across GPUs. **TP** (tensor): shard each layer's
matrices across ranks, all-reduce per layer (第25/37课). **PP** (pipeline): split layers into stages,
micro-batch through them (第23课). **EP** (expert): shard MoE experts across ranks, all-to-all to
route (第34课). **DP** (data): replicate and split requests (第23课 dp-controller). SGLang unifies all
four behind ONE `GroupCoordinator` (a process group with `rank`/`ranks`/`world_size`/`local_rank`/
`rank_in_group`) — TP/PP/EP/DP are just different GroupCoordinator instances over different rank
layouts. They COMPOSE (e.g. TP×PP×DP). **Cited:** `python/sglang/srt/distributed/parallel_state.py
::GroupCoordinator` (the unified group abstraction). **Read:** `distributed/parallel_state.py`
(GroupCoordinator, get_tensor_model_parallel_*, get_pipeline_model_parallel_*),
`distributed/communication_op.py`. **Diagrams:** a `table.t` TP/PP/EP/DP → what's split / what's
communicated; a `layers` one model sharded by TP across 4 ranks; a `cols` split weights (TP/PP/EP) vs
split data (DP); a `flow` op → GroupCoordinator (all-reduce / all-to-all / p2p) → result. Tie
第25/34/37/23课. Forward-ref EPLB 第47课.

### L47 — 47-large-scale-ep-and-eplb.html / "大规模 EP 与 EPLB"
**Scope:** making expert parallelism balanced at scale. With hundreds of experts across many GPUs (第34
课), token routing is SKEWED — some experts are hot, some idle — so the busiest rank bottlenecks the
step. EPLB (Expert-Parallel Load Balancer) periodically measures the expert load distribution and
REBALANCES expert→GPU placement (and may replicate hot experts) to flatten the load. SGLang's
`EPLBManager` hooks `on_forward_pass_end` to gather stats and `rebalance()` to recompute placement
via `expert_location`/`expert_distribution`. This is essential for large MoE (DeepSeek-scale)
serving. **Cited:** `python/sglang/srt/eplb/eplb_manager.py ::EPLBManager` (on_forward_pass_end +
rebalance lifecycle). **Read:** `eplb/eplb_manager.py`, `eplb/expert_location.py`,
`eplb/expert_distribution.py`, `eplb/eplb_algorithms/`. **Diagrams:** a `flow` route → measure load →
rebalance placement → flatter load; a `cols` skewed (one hot expert stalls all) vs balanced; a
`table.t` EPLBManager hooks → job; a `vflow`/`step` collect stats→solve placement→update→repeat. Tie
第34课 MoE, 第46课 EP. Forward-ref design theme 第61/62课.

### L48 — 48-structured-outputs-and-jump-forward.html / "结构化输出与跳跃前进"
**Scope:** force valid JSON/regex output, fast. A grammar backend (xgrammar / outlines / llguidance)
compiles the schema into an FSM; each step it MASKS the logits (第37课) so only tokens that keep the
output grammar-valid can be sampled — guaranteeing well-formed output. `BaseGrammarObject` is the
per-request FSM: `accept_token` advances it, `allocate_vocab_mask`/`fill_vocab_mask` build the
token mask, `rollback` supports backtracking (and spec decoding 第43课). The optimization is
JUMP-FORWARD: when the FSM has only ONE possible continuation for several tokens (e.g. the fixed key
`"name":` in a JSON schema), emit them directly WITHOUT calling the model — a compressed FSM
(`OutlinesJumpForwardMap.jump_forward_symbol`) skips deterministic spans. **Cited:**
`python/sglang/srt/constrained/base_grammar_backend.py ::BaseGrammarObject` (accept_token +
fill_vocab_mask + rollback). **Read:** `constrained/base_grammar_backend.py`,
`constrained/outlines_jump_forward.py` (`OutlinesJumpForwardMap`), `constrained/xgrammar_backend.py`.
**Diagrams:** a `flow` logits → vocab mask (FSM) → sample valid token → advance FSM; a `cellgroup`/
`layers` FSM states with a deterministic span jumped; a `cols` plain decode vs jump-forward; a
`table.t` BaseGrammarObject methods → role. Tie 第37课 logits/mask, 第28课 sampler, 第43课 rollback.
Closes Part 10.

## Wiring & DoD
- New module `src/part10.py` (`LESSON_43..48`); `registry.py` imports `part10` + 6 keys; `shell.PAGES`
  + `SUBTITLES` += 6; `quizzes.QUIZZES` += 6. Part label "第十部分 · 性能创新专题" /
  "Part 10 · Performance innovations".
- Filenames: `43-speculative-decoding-overview.html`, `44-eagle-and-next-gen.html`,
  `45-pd-disaggregation.html`, `46-tp-pp-ep-dp-parallelism.html`,
  `47-large-scale-ep-and-eplb.html`, `48-structured-outputs-and-jump-forward.html`.
- All validators 0 err / 0 warn; no-diff; pill "共 48 课 · 10 个部分"; nav L42↔L43…L48.
- Source-accurate: cite real symbols; spec-decoding naming follows the skill.
