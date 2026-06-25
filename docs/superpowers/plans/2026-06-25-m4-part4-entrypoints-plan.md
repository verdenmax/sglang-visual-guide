# M4 — Part 4 Entrypoints & Orchestration (L13–L17) — Plan

> One sync subagent per lesson, then two-stage review. Spec:
> `specs/2026-06-25-m4-part4-entrypoints-spec.md`. Steps use checkbox (`- [ ]`).

**Goal:** Add L13–L17 (the runtime's outer shell) as `src/part4.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source:** `~/course/sglang/python/sglang/srt/`.

## Common recipe (every lesson)
Model on part1/2/3. Each `LESSON_NN = {"zh": r"""…""","en": r"""…"""}`: lead → 🔌 analogy → 🌍 macro
→ 3–4 `<h2>` → **≥3 visual blocks/lang, identical zh/en inventory** (cols/vflow/flow/layers/cellgroup/
`<table class="t">`, no SVG) → ONE cited `.codefile` (`file ::symbol`, small faithful snippet) → 本课要点
card → quiz (3 MCQ + 2 open). zh ≥3000 CJK. Only `shell.CSS` classes. Forward-refs map to real lessons.

## Task 0 — module + wiring (orchestrator does this)
- [ ] Create `src/part4.py` (`LESSON_13..17` placeholders); `import part4` in `registry.py`; add 5
  PAGES + 5 SUBTITLES + 5 CONTENT keys. Part label "第四部分 · 服务入口与编排 / Part 4 · Entrypoints & orchestration".
  Filenames: `13-engine-and-http-server.html`, `14-tokenizer-manager.html`,
  `15-openai-anthropic-ollama-compat.html`, `16-io-structs-and-ipc.html`, `17-detokenizer-and-streaming.html`.

## Per-lesson tasks (each: read source for accuracy, author per spec, add quiz, smoke-build)
- [ ] **L13** Engine & HTTP server. Cite `srt/entrypoints/engine.py ::Engine`. Read engine.py (Engine ~182,
  generate ~317), EngineBase.py, http_server.py (launch_server ~2473, /generate ~773). Analogy: a
  **walk-in counter (offline Engine, call directly) vs a phone-order hotline (online server)** wrapping the
  same kitchen. Diagrams: cols offline-vs-online; layers/flow HTTP→app→Engine→subprocesses; table.t entry points.
- [ ] **L14** TokenizerManager. Cite `srt/managers/tokenizer_manager.py ::TokenizerManager.generate_request`.
  Read generate_request ~588, _tokenize_one_request ~792. Analogy: a **front-desk receptionist** who
  translates your words into a ticket (token ids + params), files it to dispatch (ZMQ), and waits for the reply.
  Diagrams: vflow text→tokenize→params→ZMQ→await; flow request-id round trip; table.t what it owns.
- [ ] **L15** OpenAI/Anthropic/Ollama compat. Cite an `srt/entrypoints/openai/` serving class (e.g.
  `serving_chat.py ::OpenAIServingChat`) or the `/v1/chat/completions` route in http_server.py — VERIFY the
  symbol exists before citing. Read entrypoints/openai/ + http_server.py v1 routes. Analogy: **multilingual
  translators at the door** turning any client's dialect (OpenAI/Anthropic/Ollama) into the house language
  (GenerateReqInput). Diagrams: layers client→adapter→native→TokenizerManager; table.t route→native; cols
  /generate vs /v1/chat/completions. Tie to L12 ("OpenAI client → SGLang server").
- [ ] **L16** IO structs & IPC. Cite `srt/managers/io_struct.py ::TokenizedGenerateReqInput`. Read
  GenerateReqInput ~116, TokenizedGenerateReqInput ~734, BatchStrOutput ~1209. Analogy: the **standardized
  forms/envelopes** that move between departments (processes) — each hop has its own form. Diagrams: flow
  GenerateReqInput→Tokenized→(ZMQ)→scheduler→BatchOutput→detok; table.t struct→fields→hop; layers 3-process bus.
- [ ] **L17** Detokenizer & streaming. Cite `srt/managers/detokenizer_manager.py ::DetokenizerManager.event_loop`
  (or the `sent_offset` incremental slice ~71/381). Analogy: a **simultaneous interpreter** who speaks only
  the NEW words since last time (sent_offset), not the whole sentence again. Diagrams: vflow token-ids→
  incremental-detok→sent_offset-slice→SSE; cellgroup sent_offset advancing; table.t stop conditions.

## Verify + commit
- [ ] `cd src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py`
  → 0 err / 0 warn; pill "共 17 课 · 4 个部分"; nav L12↔L13↔…↔L17; no-diff.
- [ ] One commit: `M4: Part 4 entrypoints — L13 engine/server, L14 tokenizer mgr, L15 OpenAI compat, L16 io_struct/IPC, L17 detokenizer/streaming` (+ Co-authored-by trailer).

## Guardrails
- Cite `file ::symbol`, small snippets. The 3-process (TokenizerManager / Scheduler+TpWorker /
  DetokenizerManager) + ZMQ story must match the real managers (don't invent a 2-process or in-process model).
- For L15, VERIFY the cited OpenAI serving symbol exists in `entrypoints/openai/` before using it.
- zh ≥3000 CJK; zh/en identical diagram inventory. Don't touch `docs/`, earlier parts, pipeline, or the reference repo.
