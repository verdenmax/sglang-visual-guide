# M11 — Part 11 Advanced / Optional (L49–L52) — Plan

> Execute with subagent-driven-development. One subagent per lesson, claude-opus-4.8, sync, high
> effort. Two-stage review (research + code-review, opus-4.8 max, long context) after exec.

**Goal:** add Part 11 (L49–52) — multimodal VLM, multi-LoRA, RL rollout/weight-sync, diffusion.
**Companion spec:** `docs/superpowers/specs/2026-06-25-m11-part11-advanced-spec.md`.

## Verified citations (pre-fetched — pass to subagents verbatim)
- L49 `python/sglang/srt/managers/mm_utils.py ::general_mm_embed_routine` — `(input_ids,
  data_embedding_funcs: Dict[Modality, fn], placeholder_tokens, …)`: run encoders + splice media
  embeddings at placeholder positions; `multimodal/processors/base_processor.py
  ::BaseMultimodalProcessor` (raw media → tensors + placeholders).
- L50 `python/sglang/srt/lora/lora_manager.py ::LoRAManager` — `prepare_lora_batch(forward_batch)`,
  `load_lora_adapter`/`unload_lora_adapter`, `max_loras_per_batch`; adapter = `lora/lora.py
  ::LoRAAdapter` (low-rank ΔW); `lora/mem_pool.py`.
- L51 `python/sglang/srt/entrypoints/engine.py ::update_weights_from_tensor` — `(named_tensors,
  load_format=…)`; `load_format=="flattened_bucket"` uses `weight_sync/tensor_bucket.py
  ::FlattenedTensorBucket` (flatten many named tensors into one buffer); also
  `update_weights_from_distributed`.
- L52 `python/sglang/multimodal_gen/configs/pipeline_configs/base.py ::PipelineConfig` — the diffusion
  pipeline descriptor (NOTE: under `python/sglang/multimodal_gen/`, NOT `srt/`); README: sglang-diffusion
  = image/video inference reusing sgl-kernel + scheduler loop + OpenAI API + multi-hw; cache
  `multimodal_gen/runtime/cache/teacache.py` (TeaCache skips redundant denoise steps).

## Task 0 — orchestrator wiring (do myself)
- [ ] Create `src/part11.py` with `LESSON_49..LESSON_52` placeholders.
- [ ] `registry.py`: `import part11`; add 4 CONTENT keys (filenames below → `part11.LESSON_5X`).
- [ ] `shell.py`: append 4 PAGES 5-tuples (anchor after L48) + 4 SUBTITLES; label
  "第十一部分 · 进阶选读" / "Part 11 · Advanced (optional)".
- [ ] Build + confirm pill "共 52 课 · 11 个部分", nav L48↔L49.

Filenames: `49-multimodal-vlm-serving.html`, `50-multi-lora-batching.html`,
`51-rl-rollout-and-weight-sync.html`, `52-diffusion-models.html`.

## Tasks 1–4 — per-lesson subagents (L49…L52)
Each: replace that lesson's placeholder in `src/part11.py` + append its `quizzes.py` entry. Proven
edit-first prompt template: exact placeholder line; verified `file ::symbol` + real condensed code;
house markup (card analogy/macro/key with `<div class="tag">…</div>`; codefile cf-head dot/path/ln);
allowed CSS palette; "real prose ≥3600 CJK zh, identical zh/en inventory, EXACTLY 4 diagram blocks, NO
SVG, NOT timeline, docstrings→`#`"; VERIFY cmds. Retry on truncation; draft+expand fallback after 3 fails.

- [ ] **T1 L49** multimodal VLM — general_mm_embed_routine (processor + splice).
- [ ] **T2 L50** multi-LoRA batching — LoRAManager (prepare_lora_batch / adapter pool).
- [ ] **T3 L51** RL rollout & weight sync — update_weights_from_tensor (+ FlattenedTensorBucket).
- [ ] **T4 L52** diffusion models — PipelineConfig (denoise loop, reuse SGLang infra). Closes Part 11.

## Task 5 — verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py &&
  python3 check_links.py` → 0/0; no-diff. Commit (`verdenmax`, Co-authored-by Copilot).

## Task 6 — two-stage review + fixes
- [ ] research (concept/spec/completeness: VLM splice mechanism, LoRA batching, RL weight-sync, the
  diffusion-reuses-infra claim + multimodal_gen path accuracy) + code-review (markup/parity/escaping/
  wiring/quiz schema), parallel, opus-4.8 max, long context. Fix → re-verify → commit.

## DoD
52 lessons; Part 11 nav + pill correct; validators 0/0; no-diff; citations source-accurate (incl. the
multimodal_gen path note); spec scope met.
