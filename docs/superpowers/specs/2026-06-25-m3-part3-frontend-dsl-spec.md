# M3 — Part 3 Frontend DSL (L09–L12) — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** design spec, roadmap M3. Builds on M0–M2 (L01–08 exist).

## Goal
Explain the **other half** of SGLang: the frontend **Structured Generation Language** — the embedded
Python DSL that lets you write multi-call LLM programs (`gen`/`select`/`fork`/`join`), how it executes
vs traces/compiles, why its programming model maps perfectly onto RadixAttention prefix sharing, and
how it talks to backends (local runtime vs OpenAI/Anthropic). Module: `src/part3.py`, lessons L09–L12.

## Content model & gates (unchanged): per lesson
lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` with ≥3 visual blocks/lang (identical zh/en inventory;
no SVG needed) → one cited `.codefile` (`file ::symbol`) → 本课要点 card → quiz (3 MCQ + 2 open).
zh ≥3000 CJK, only CSS classes in `shell.CSS`. Validators 0 err / 0 warn. Forward-refs map to real lessons.

## Lessons

### L09 — 09-structured-generation-language.html / "结构化生成语言 / The structured-generation language"
**Scope:** SGLang the *language* (the project's origin/namesake). A `@sgl.function` decorated Python
function receives a state `s` and builds a prompt by `s += ...`; primitives: `gen(name, max_tokens,
stop, …)` (a fill-in slot), `select(name, choices)` (constrained choice), role helpers
`system/user/assistant`, `image`. Multiple `gen`s in one function = a multi-call program; `s["name"]`
reads results. Why a DSL beats hand-rolling: control flow + constrained decoding + automatic
KV reuse. **Cited:** `lang/api.py ::gen` (or `ir.py ::SglFunction`). **Read:** `lang/api.py`,
`lang/ir.py` (SglGen/SglSelect/SglFunction). **Diagrams:** a `vflow`/`flow` of a 2-call program; a
`table.t` primitive → what it does; a `cols` DSL-vs-raw-API. Forward-ref interpreter 第 10 课, prefix 第 11 课.

### L10 — 10-interpreter-and-tracer.html / "解释器与 tracer / Interpreter & tracer"
**Scope:** how a program runs. Two modes: **interpret** — `StreamExecutor` drives the program call by
call against a backend, streaming each `gen`; `ProgramState` (the `s`) holds the text/vars. And
**trace** — `trace_program` runs the function symbolically to extract structure/static prefix without
hitting the model (used for compilation/prefix extraction). **Cited:** `lang/interpreter.py
::StreamExecutor` (or `tracer.py ::trace_program`). **Read:** `lang/interpreter.py` (StreamExecutor,
ProgramState), `lang/tracer.py`. **Diagrams:** a `layers`/`vflow` interpret path (program → executor →
backend → state); a `cols` interpret-vs-trace; a `table.t` of what each mode is for.

### L11 — 11-fork-join-and-prefix-sharing.html / "fork/join 与前缀共享 / fork/join & prefix sharing"
**Scope:** the payoff. `s.fork(n)` spawns `n` parallel continuations that **share the prompt prefix so
far**; each branch generates independently; `join`/gather collects their variables back. Because all
branches share the same prefix, RadixAttention computes that prefix's KV **once** and reuses it across
branches — the DSL's structure and the runtime's prefix cache are two halves of one idea. Use cases:
parallel sampling, branch-and-evaluate, map over options. **Cited:** `lang/interpreter.py
::ProgramState.fork` (or `ProgramStateGroup.join`). **Read:** `lang/interpreter.py` (fork ~888, join
~1052), `lang/ir.py` (SglFork). **Diagrams:** a `flow`/`vflow` fork-then-join; a `cellgroup` shared
prefix across branches (tie to L07); a `cols` serial-vs-fork. Forward-ref RadixAttention 第 7 课.

### L12 — 12-backends-and-openai-compat.html / "后端与 OpenAI 兼容 / Backends & OpenAI compat"
**Scope:** the DSL is backend-agnostic. A `BaseBackend` interface; the local **SGLang runtime**
backend (`RuntimeEndpoint`) talks to the `/generate` server we built in Part 1; other backends
(OpenAI, Anthropic, …) let the same program run against hosted models (no prefix-cache benefit there).
This is the seam between Part 3 (frontend) and Parts 4+ (runtime). **Cited:** `lang/backend/
runtime_endpoint.py ::RuntimeEndpoint` (or `base_backend.py ::BaseBackend`). **Read:** `lang/backend/
base_backend.py`, `runtime_endpoint.py`, `openai.py`. **Diagrams:** a `layers` program → backend
interface → {runtime, OpenAI, …}; a `table.t` backend → capabilities (streaming, prefix cache,
constrained); a `flow` program → RuntimeEndpoint → /generate. Forward-ref entrypoints 第 13–17 课.

## Wiring & DoD
- New module `src/part3.py` (`LESSON_09..12`); `registry.py` imports `part3` + 4 keys; `shell.PAGES`
  + `SUBTITLES` += 4 Part 3 entries; `quizzes.QUIZZES` += 4.
- Filenames: `09-structured-generation-language.html`, `10-interpreter-and-tracer.html`,
  `11-fork-join-and-prefix-sharing.html`, `12-backends-and-openai-compat.html`. Part label
  "第三部分 · 前端语言 / Part 3 · The frontend language".
- All validators 0 err / 0 warn; no-diff; index pill "共 12 课 · 3 个部分"; nav L08↔L09…L12.
- Source-accurate (cite `file ::symbol`); the DSL examples must be runnable-looking SGLang code.
