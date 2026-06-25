# M3 — Part 3 Frontend DSL (L09–L12) — Plan

> One implementer per lesson (sync subagent), then two-stage review. Spec:
> `specs/2026-06-25-m3-part3-frontend-dsl-spec.md`. Steps use checkbox (`- [ ]`).

**Goal:** Add L09–L12 (the SGLang frontend language) as a new module `src/part3.py`.

**Working dir:** `~/course/sglang-visual-guide/`. **SGLang source (read-only):** `~/course/sglang/python/sglang/lang/`.

## Common authoring recipe (every lesson)
Model on `src/part1.py` + `src/part2.py` (full real lessons). Each `LESSON_NN = {"zh": r"""…""",
"en": r"""…"""}` MUST have: lead → 🔌 analogy → 🌍 macro → 3–4 `<h2>` → **≥3 visual blocks per
language, identical zh/en inventory** (cols/vflow/flow/layers/cellgroup/`<table class="t">` — no SVG
needed) → exactly ONE cited `.codefile` (`file ::symbol`, small faithful snippet) → 本课要点/Key points
card → quiz (3 MCQ + 2 open). **zh ≥ 3000 CJK.** Use ONLY classes defined in `shell.CSS`. Forward-refs
map to the real lesson (design §6). After each lesson, `cd src && python3 build.py && python3 check_html.py`
and fix that lesson's ERROR/WARN before the next.

## Task 0 — module + wiring
- [ ] Create `src/part3.py` (docstring + `LESSON_09..12`); add `import part3` to `src/registry.py`.
- [ ] Add 4 `PAGES` entries (after L08), 4 `SUBTITLES`, 4 `CONTENT` keys. Part label
  "第三部分 · 前端语言" / "Part 3 · The frontend language". Filenames + titles:
```python
("09-structured-generation-language.html", "结构化生成语言", "The structured-generation language",
 "第三部分 · 前端语言", "Part 3 · The frontend language"),
("10-interpreter-and-tracer.html", "解释器与 tracer", "Interpreter & tracer",
 "第三部分 · 前端语言", "Part 3 · The frontend language"),
("11-fork-join-and-prefix-sharing.html", "fork/join 与前缀共享", "fork/join & prefix sharing",
 "第三部分 · 前端语言", "Part 3 · The frontend language"),
("12-backends-and-openai-compat.html", "后端与 OpenAI 兼容", "Backends & OpenAI compat",
 "第三部分 · 前端语言", "Part 3 · The frontend language"),
```

## Task 1 — L09 structured-generation-language
- [ ] Read `~/course/sglang/python/sglang/lang/api.py` (gen ~75, select ~236, system/user/assistant)
  and `lang/ir.py` (SglFunction ~141, SglGen ~451, SglSelect ~533). Cite `lang/api.py ::gen`.
- [ ] Author per spec L09 (analogy: a **fill-in-the-blank form / mail-merge template** where the model
  fills the blanks). Show a real `@sgl.function` 2-call program. Diagrams: a `vflow`/`flow` of the
  program's calls; a `table.t` primitive→meaning; a `cols` DSL-vs-raw-API. Forward-ref 第 10/11 课.

## Task 2 — L10 interpreter-and-tracer
- [ ] Read `lang/interpreter.py` (StreamExecutor ~274, ProgramState ~852) and `lang/tracer.py`
  (trace_program ~54, extract_prefix_by_tracing ~29). Cite `lang/interpreter.py ::StreamExecutor`.
- [ ] Author per spec L10 (analogy: a **recipe** either cooked step-by-step now (interpret) or
  read-through to shop the ingredient list first (trace)). Diagrams: `layers`/`vflow` interpret path;
  `cols` interpret-vs-trace; `table.t` what each mode is for.

## Task 3 — L11 fork-join-and-prefix-sharing
- [ ] Read `lang/interpreter.py` (ProgramState.fork ~888, ProgramStateGroup.join ~1052), `lang/ir.py`
  (SglFork ~552). Cite `lang/interpreter.py ::ProgramState.fork`.
- [ ] Author per spec L11 (analogy: a **choose-your-own-adventure** branching after a shared opening;
  all branches reuse the same first chapters). Diagrams: a `flow`/`vflow` fork→branches→join; a
  `cellgroup` shared prefix across branches (tie to L07 RadixAttention); a `cols` serial-vs-fork.
  Make the L07 prefix-sharing connection explicit. Forward-ref 第 7 课.

## Task 4 — L12 backends-and-openai-compat
- [ ] Read `lang/backend/base_backend.py`, `runtime_endpoint.py` (RuntimeEndpoint), `openai.py`.
  Cite `lang/backend/runtime_endpoint.py ::RuntimeEndpoint`.
- [ ] Author per spec L12 (analogy: a **universal remote** driving different TVs via one interface;
  only your own runtime gives the prefix-cache superpower). Diagrams: `layers` program→backend
  interface→{runtime, OpenAI, …}; `table.t` backend→capabilities; `flow` program→RuntimeEndpoint→/generate.
  Forward-ref entrypoints 第 13–17 课. This closes Part 3 and bridges to the runtime (Part 4).

## Verify + commit
- [ ] `cd ~/course/sglang-visual-guide/src && python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py` → 0 err / 0 warn; pill "共 12 课 · 3 个部分"; nav L08↔L09↔…↔L12; no-diff (`git add -A && git status`).
- [ ] One commit: `M3: Part 3 frontend DSL — L09 language, L10 interpreter/tracer, L11 fork/join, L12 backends` (+ Co-authored-by trailer).

## Guardrails
- Cite `file ::symbol`, small snippets; the DSL code examples should look like real SGLang programs.
- Make the L11 fork/join → RadixAttention (第 7 课) connection explicit — it's the chapter's point.
- zh ≥ 3000 CJK; zh/en identical diagram inventory; only `shell.CSS` classes. Don't touch `docs/`,
  earlier parts, the pipeline, or the reference repo. SGLang is Apache-2.0.
