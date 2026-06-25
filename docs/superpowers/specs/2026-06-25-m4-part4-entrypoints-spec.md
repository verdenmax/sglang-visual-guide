# M4 — Part 4 Entrypoints & Orchestration (L13–L17) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M4. Builds on M0–M3 (L01–12 exist).

## Goal
Open the runtime: how a request enters the server and is orchestrated across processes. Covers the
`Engine`/HTTP server entry, the `TokenizerManager` front door, the OpenAI/Anthropic/Ollama compat
layer, the IPC `io_struct` messages that travel between processes, and the `DetokenizerManager` +
streaming exit. This is the "outer shell" before the scheduler (Part 5). Module: `src/part4.py`, L13–17.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with ≥3 visual blocks/lang (identical zh/en; no SVG) →
one cited `.codefile` (`file ::symbol`) → 本课要点 card → quiz (3 MCQ + 2 open). zh ≥3000 CJK,
only CSS classes in `shell.CSS`. Validators 0 err / 0 warn. Forward-refs map to real lessons.

## Lessons

### L13 — 13-engine-and-http-server.html / "Engine 与 HTTP Server / Engine & HTTP server"
**Scope:** two ways to use SGLang — **offline `Engine`** (in-process Python API, `Engine.generate`)
vs **online server** (`launch_server` starts a FastAPI app exposing `/generate` + OpenAI routes that
wraps the same Engine). `EngineBase` is the shared abstract base. The Engine's `__init__` launches the
3 subprocesses (recall L02). When to use which (batch jobs/RL rollout vs serving). **Cited:**
`srt/entrypoints/engine.py ::Engine` (or `::Engine.generate`). **Read:** `entrypoints/engine.py`
(Engine, generate ~317), `entrypoints/EngineBase.py`, `entrypoints/http_server.py` (launch_server ~2473,
the `/generate` route ~773). **Diagrams:** a `cols` offline-Engine-vs-online-server; a `layers`/`flow`
HTTP→app→Engine→subprocesses; a `table.t` entry points (Engine.generate / POST /generate / /v1/*).
Forward-ref TokenizerManager 第 14 课, scheduler 第 18 课.

### L14 — 14-tokenizer-manager.html / "TokenizerManager / TokenizerManager"
**Scope:** the front door. Runs in the main process; `generate_request` tokenizes the prompt, builds
sampling params + a `TokenizedGenerateReqInput`, assigns a request id, and sends it over **ZMQ** to the
Scheduler; it also awaits the streamed outputs coming back (an async per-request state). Handles
abort, multi-request fan-in. Why tokenize here (CPU work off the GPU process). **Cited:**
`srt/managers/tokenizer_manager.py ::TokenizerManager.generate_request`. **Read:**
`managers/tokenizer_manager.py` (generate_request ~588, _tokenize_one_request ~792). **Diagrams:** a
`vflow` text→tokenize→params→ZMQ send→await; a `flow` of the request id round-trip; a `table.t` of
what it owns. Forward-ref io_struct 第 16 课, scheduler 第 18 课.

### L15 — 15-openai-anthropic-ollama-compat.html / "OpenAI/Anthropic/Ollama 兼容层"
**Scope:** the server speaks multiple protocols. `entrypoints/openai/` implements `/v1/completions`,
`/v1/chat/completions`, `/v1/embeddings` by translating OpenAI requests → the native `GenerateReqInput`
→ TokenizerManager, and the outputs back into OpenAI response/SSE shapes. Anthropic + Ollama adapters
do the same for their schemas. So any OpenAI/Anthropic/Ollama client works unchanged (tie to L12's
"OpenAI client → SGLang server" direction). **Cited:** an OpenAI serving class under
`entrypoints/openai/` (e.g. `serving_chat.py ::OpenAIServingChat`) or the `/v1/chat/completions` route
in `http_server.py`. **Read:** `entrypoints/openai/` (serving_chat/serving_completions), `http_server.py`
(the v1 routes). **Diagrams:** a `layers` client→protocol adapter→native request→TokenizerManager; a
`table.t` route → native mapping; a `cols` native /generate vs /v1/chat/completions. Forward-ref L12, L14.

### L16 — 16-io-structs-and-ipc.html / "IO 结构与进程间通信 / IO structs & IPC"
**Scope:** the messages on the wire. `GenerateReqInput` (user-facing request) →
`TokenizedGenerateReqInput` (what crosses to the scheduler) → batch outputs `BatchTokenIDOutput` /
`BatchStrOutput` coming back. Why dataclasses + ZMQ (cheap, language-agnostic serialization between the
3 processes). The request id threads everything. This is the "type system" of the runtime's IPC.
**Cited:** `srt/managers/io_struct.py ::TokenizedGenerateReqInput` (or `GenerateReqInput`). **Read:**
`managers/io_struct.py` (GenerateReqInput ~116, TokenizedGenerateReqInput ~734, BatchStrOutput ~1209).
**Diagrams:** a `flow` GenerateReqInput→Tokenized→(ZMQ)→scheduler→BatchOutput→detokenizer; a `table.t`
struct → fields → which hop; a `cellgroup`/`layers` of the 3-process message bus. Forward-ref L14, L17, L18.

### L17 — 17-detokenizer-and-streaming.html / "Detokenizer 与流式输出 / Detokenizer & streaming"
**Scope:** the exit. `DetokenizerManager` (a subprocess) turns generated token ids back into text
**incrementally** — it tracks a per-request `sent_offset` and emits only the newly-decoded slice each
step (handles multi-byte/partial tokens correctly), then sends it back to the TokenizerManager, which
streams it to the HTTP client as **SSE**. Stop conditions (eos, stop strings, max tokens). Why detok is
its own process (CPU work, overlap). **Cited:** `srt/managers/detokenizer_manager.py
::DetokenizerManager.event_loop` (or the `sent_offset` incremental slice). **Read:**
`managers/detokenizer_manager.py` (event_loop ~161, incremental sent_offset ~71/381). **Diagrams:** a
`vflow` token ids→incremental detok→sent_offset slice→SSE; a `cellgroup` showing sent_offset advancing;
a `table.t` stop conditions. Forward-ref io_struct 第 16 课, scheduler 第 18 课.

## Wiring & DoD
- New module `src/part4.py` (`LESSON_13..17`); `registry.py` imports `part4` + 5 keys; `shell.PAGES`
  + `SUBTITLES` += 5; `quizzes.QUIZZES` += 5. Filenames as above. Part label
  "第四部分 · 服务入口与编排 / Part 4 · Entrypoints & orchestration".
- All validators 0 err / 0 warn; no-diff; index pill "共 17 课 · 4 个部分"; nav L12↔L13…L17.
- Source-accurate (`file ::symbol`); the 3-process / ZMQ story must match the real managers.
