# M1 — Part 1 Overview (L02–L03) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M1. Builds on M0 (pipeline + L01 exist).

## Goal
Complete Part 1 (宏观全景) by adding **L02 项目全景地图 / The project map** and **L03 一次请求的一生
/ Life of a request**. These two give the reader a mental map (where code lives, who the processes
are) and a concrete end-to-end trace, so every later part has an anchor.

## Content model & gates (unchanged from design spec §3)
Each lesson: lead → 🔌 analogy → 🌍 macro → 2–5 `<h2>` with ≥3 visual blocks/lang → one cited
`.codefile` (`file + symbol`) → 本课要点 card → quiz (2–4 MCQ + 1–2 open). zh ≥3000 CJK (~4000),
zh/en identical diagram inventory, only CSS classes defined in `shell.CSS`. Validators 0 err / 0 warn.

## L02 — 02-project-map.html / "The project map"
**Scope:** the repository & process map.
- The **two halves**: `python/sglang/lang/` (frontend DSL) vs `python/sglang/srt/` (serving runtime
  "SRT = SGLang Runtime"), plus the native `sgl-kernel/` (AOT C++/CUDA) and `python/sglang/jit_kernel/`.
- The **runtime sub-packages** at a glance (what each owns): `entrypoints/`, `managers/`,
  `model_executor/`, `model_loader/`, `models/`, `layers/`, `mem_cache/`, `sampling/`,
  `speculative/`, `disaggregation/`, `distributed/`, `constrained/`, `lora/`, `multimodal/`.
- The **process model**: TokenizerManager (front) ↔ Scheduler + TpWorker (one per TP rank, holds
  ModelRunner) ↔ DetokenizerManager, wired by **ZMQ**; launched by `Engine` / `launch_server`.
- "Where do I start reading?" pointers that forward-ref later parts.
**Diagrams (≥3/lang):** a `layers` of the package map (DSL / entrypoints / managers / executor /
layers / mem_cache / kernels); a `table.t` package → responsibility → later 第 N 课; a `flow`/`cols`
of the 3-process model with ZMQ arrows.
**Cited code:** one small snippet from `srt/entrypoints/engine.py` (the subprocess launch, e.g.
`_launch_subprocesses` / `Engine.__init__`) OR `srt/managers/scheduler.py ::run_scheduler_process`,
showing the process split. Cite `file + symbol`.
**Read first:** `python/sglang/` + `python/sglang/srt/` dir listings; `srt/entrypoints/engine.py`
(subprocess launch), `srt/server_args.py` (top-level options), `README.md` (feature list).

## L03 — 03-life-of-a-request.html / "Life of a request"
**Scope:** trace one `generate` request end-to-end at 10,000 ft (zoom-ins are later parts).
- Entry: HTTP `POST /generate` (or OpenAI route) → `http_server` → `TokenizerManager.generate_request`.
- Tokenize + build a `TokenizedGenerateReqInput` (io_struct) → **ZMQ** → Scheduler.
- Scheduler loop: `recv_requests` → waiting queue → `get_next_batch_to_run` (prefill vs decode) →
  `run_batch` → `TpWorker`/`ModelRunner.forward` → logits → `Sampler` → next token.
- Stream out: result → DetokenizerManager (incremental detok) → back to TokenizerManager → SSE chunk.
- The prefill→decode loop (one prefill + many decodes); where continuous batching "cuts in".
**Diagrams (≥3/lang):** a `vflow`/`trace` of the full path with the process boundaries marked; a
`flow` decode-loop (steps 2–4 repeat); a `timeline`/`cols` prefill-vs-decode. Forward-ref the deep
lessons (entrypoints M4, scheduler M5, model exec M6, KV cache M7).
**Cited code:** one small snippet from `srt/managers/scheduler.py` (the event loop, e.g.
`Scheduler.event_loop_normal` / `event_loop_overlap`) or `tokenizer_manager.py ::generate_request`.
**Read first:** `srt/entrypoints/http_server.py` (the `/generate` handler), `srt/managers/
tokenizer_manager.py` (`generate_request`), `srt/managers/scheduler.py` (event loop +
`get_next_batch_to_run`), `srt/managers/io_struct.py` (the Req structs), `srt/managers/
detokenizer_manager.py`.

## Wiring & DoD
- `shell.PAGES` += L02, L03 (Part 1 titles); `shell.SUBTITLES` += L02, L03; `registry.CONTENT` +=
  both; `quizzes.QUIZZES` += both. Filenames `02-project-map.html`, `03-life-of-a-request.html`.
- `build.py` + `build_print.py` + `check_html.py` + `check_links.py` → 0 err / 0 warn; no-diff
  rebuild; index pill reads "共 3 课 · 1 个部分"; nav chain L01↔L02↔L03 correct.
- Source-accurate: cite `file + symbol`; code wins over docs; small snippets only.
