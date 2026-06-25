# M0 — Scaffold & Pipeline — Plan

> **For agentic workers:** execute task-by-task; this milestone is run by one implementer subagent,
> then two-stage review. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stand up the zero-dependency Python pipeline + chrome, cloned from
`~/course/milvus-visual-guide/` and re-themed for SGLang, shipping only L01.

**Spec:** `docs/superpowers/specs/2026-06-25-m0-scaffold-spec.md`

**Reference (read-only template):** `~/course/milvus-visual-guide/` (the `REF` repo below).

**Tech Stack:** Python 3 (stdlib only), static HTML/CSS, GitHub Actions.

**Working dir:** `~/course/sglang-visual-guide/` (git repo already initialized; spec + roadmap committed).

---

## Task 1 — Copy subject-agnostic infra verbatim

**Files (create in `~/course/sglang-visual-guide/`):**
- `src/build.py`, `src/build_print.py`, `src/check_links.py`
- `.gitignore`, `LICENSE`, `LICENSE-CONTENT`
- `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`

- [ ] **Step 1:** Copy each file from `REF` to the same relative path, byte-for-byte:
```bash
cd ~/course/sglang-visual-guide
mkdir -p src .github/workflows
REF=~/course/milvus-visual-guide
cp "$REF/src/build.py" src/build.py
cp "$REF/src/build_print.py" src/build_print.py
cp "$REF/src/check_links.py" src/check_links.py
cp "$REF/.gitignore" .gitignore
cp "$REF/LICENSE" LICENSE
cp "$REF/LICENSE-CONTENT" LICENSE-CONTENT
cp "$REF/.github/workflows/ci.yml" .github/workflows/ci.yml
cp "$REF/.github/workflows/deploy.yml" .github/workflows/deploy.yml
```

- [ ] **Step 2:** Verify `LICENSE` / `LICENSE-CONTENT` already say "Copyright (c) 2026 verdenmax"
  (they do in REF — no edit needed). `ci.yml`/`deploy.yml` use generic paths (`src/`, `.`) — no
  milvus-specific names, so no edit needed. Confirm with:
```bash
grep -ri "milvus" .github/ LICENSE* .gitignore || echo "clean (no milvus references)"
```
  Expected: `clean (no milvus references)`.

## Task 2 — Adapt `src/shell.py` (re-theme + seed L01)

**Files:** Create `src/shell.py` by copying `REF/src/shell.py` then editing the theme tokens.

- [ ] **Step 1:** Copy the file: `cp "$REF/src/shell.py" src/shell.py`.

- [ ] **Step 2:** Edit the docstring (line 1): `... for the Milvus visual guide."""` →
  `... for the SGLang visual guide."""`.

- [ ] **Step 3:** Edit the favicon SVG block:
  - `"<rect width='32' height='32' rx='7' fill='#1296db'/>"` → `fill='#7c48e6'`
  - `" font-weight='800' fill='#fff' text-anchor='middle'>Mv</text></svg>"` → `>Sg</text></svg>`

- [ ] **Step 4:** In `head_meta()`: `content="#1296db"` → `content="#7c48e6"`;
  `og:site_name" content="Milvus 图解教程"` → `content="SGLang 图解教程"`.

- [ ] **Step 5:** Replace the entire `PAGES = [ ... ]` list with a single L01 entry:
```python
PAGES = [
    ("01-what-is-sglang.html", "SGLang 是什么", "What is SGLang",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
]
```

- [ ] **Step 6:** In the CSS `:root` block, change the accent palette (light mode):
  `--accent: #1296db; --accent-soft: #e2f2fb; --accent-ink: #0b6699;` →
  `--accent: #7c48e6; --accent-soft: #efe9fc; --accent-ink: #5b34b0;`
  Then find the dark-mode block (`@media (prefers-color-scheme: dark)` or a `[data-theme]` /
  `.dark` selector that redefines `--accent`) and recolor its accent trio analogously
  (a brighter violet `#a98af0`, a translucent soft, a light ink) so dark mode is not left blue.
  Grep first: `grep -n "accent" src/shell.py` and recolor EVERY `--accent*` definition that is a
  blue hex; leave non-accent tokens untouched.

- [ ] **Step 7:** In `page()` (lesson shell): `title_tag = f"{idx+1:02d} · {title_zh} / {title_en} - Milvus 图解教程"`
  → `... - SGLang 图解教程`; the `desc = f"...Milvus 图解教程..."` → `SGLang 图解教程`;
  the topbar home link `🐦 <b class="lang-zh">Milvus 图解教程</b><b class="lang-en">Milvus Visual Guide</b>`
  → `⚡ <b class="lang-zh">SGLang 图解教程</b><b class="lang-en">SGLang Visual Guide</b>`.

- [ ] **Step 8:** Replace the `SUBTITLES = { ... }` dict with a single L01 entry:
```python
SUBTITLES = {
    "01-what-is-sglang.html": ("高性能 LLM/多模态服务框架 · 前端语言 + 运行时引擎 · 为何快",
                               "high-perf LLM/multimodal serving · frontend DSL + runtime engine · why it's fast"),
}
```

- [ ] **Step 9:** In `index_page()`: retarget the title/desc/h1/disclaimer strings:
  - `title_tag = "Milvus 图解教程 · 看懂分布式向量数据库内部 / Milvus Visual Guide"` →
    `"SGLang 图解教程 · 看懂高性能 LLM 服务引擎内部 / SGLang Visual Guide"`
  - the index topbar `🐦 ... Milvus ...` → `⚡ ... SGLang ...` (both lang spans)
  - the `<h1>`: `用图解读懂整个 Milvus / Understand all of Milvus, visually` →
    `用图解读懂整个 SGLang / Understand all of SGLang, visually`
  - the "verified against source" line: `milvus-io/milvus` → `sgl-project/sglang`
  - the disclaimer line: `不含 Milvus 源码 … Milvus 由其作者以 Apache-2.0 许可发布` →
    `不含 SGLang 源码 … SGLang 由其作者以 Apache-2.0 许可发布`; English likewise (Milvus→SGLang).
  - Rewrite the intro `desc` paragraph to pitch SGLang internals (scheduler, RadixAttention,
    model executor, kernels, performance innovations) instead of the Milvus pitch.

- [ ] **Step 10:** Sanity grep — no stray Milvus/blue left in shell.py:
```bash
grep -ni "milvus\|1296db\|e2f2fb\|0b6699\|🐦\|>Mv<" src/shell.py || echo "shell.py clean"
```
  Expected: `shell.py clean`.

## Task 3 — Adapt `src/check_html.py`

**Files:** Create `src/check_html.py` from `REF/src/check_html.py`.

- [ ] **Step 1:** Copy: `cp "$REF/src/check_html.py" src/check_html.py`.
- [ ] **Step 2:** Edit `MAX_LESSON = 56` → `MAX_LESSON = 63  # final planned lesson count; cross-refs may point forward`.
- [ ] **Step 3:** Edit `SOFT_EXEMPT = {"46-glossary.html"}` → `SOFT_EXEMPT = {"57-glossary.html"}`.
- [ ] **Step 4:** Confirm nothing else is milvus-specific: `grep -ni "milvus" src/check_html.py || echo "clean"`. Expected `clean`.

## Task 4 — Create `src/quizzes.py` (framework + L01 quiz only)

**Files:** Create `src/quizzes.py`. Keep the REF framework; replace the `QUIZZES` dict body.

- [ ] **Step 1:** Copy `REF/src/quizzes.py` to `src/quizzes.py`, then KEEP unchanged: the module
  docstring, `import hashlib`, the head-string constants (`_HEAD`, `_SEE`, `_CLICK`, `_ANS`, `_SEP`,
  `_OPEN`), the `_shuffle()` helper, and the `render(fname, lang)` function (everything that is not
  the `QUIZZES = { ... }` data literal).
- [ ] **Step 2:** Replace the whole `QUIZZES = { ... }` literal with a single L01 entry (2–4 MCQs +
  1–2 open). Author bilingual, design-insight questions about *why SGLang is a serving engine /
  what the two halves do / why prefix reuse matters*. Schema (raw-HTML text context; escape `<`):
```python
QUIZZES = {
    "01-what-is-sglang.html": {
        "mcq": [
            {
                "q": {"zh": "……", "en": "……"},
                "opts": [{"zh": "正确项", "en": "correct"}, {"zh": "干扰项", "en": "distractor"}],
                "answer": 0,            # 0-based index into opts as written
                "why": {"zh": "解析……", "en": "explanation…"},
            },
        ],
        "open": [{"zh": "发散问题……", "en": "open question…"}],
    },
}
```
- [ ] **Step 3:** Verify it imports and renders without error:
```bash
cd src && python3 -c "import quizzes; print(quizzes.render('01-what-is-sglang.html','zh')[:40])"
```
  Expected: prints the start of the rendered quiz HTML (no traceback).

## Task 5 — Create `src/registry.py`

**Files:** Create `src/registry.py`.

- [ ] **Step 1:** Write it (only part1 exists at M0):
```python
"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and build_print.py
both import this so the lesson set stays in sync with shell.PAGES.

Grows one Part module per milestone (part1 .. part13).
"""
import part1

# Filename -> {"zh": ..., "en": ...}. Keep keys in sync with shell.PAGES.
CONTENT = {
    "01-what-is-sglang.html": part1.LESSON_01,
}
```

## Task 6 — Write `src/part1.py` — `LESSON_01` (the real content)

**Files:** Create `src/part1.py` exporting `LESSON_01 = {"zh": r"""…""", "en": r"""…"""}`.

**Read first (source accuracy):** SGLang `~/course/sglang/README.md`,
`~/course/sglang/docs_new/docs/get-started/**`, and skim
`~/course/sglang/python/sglang/srt/entrypoints/engine.py` (the `Engine` entry) +
`~/course/sglang/python/sglang/launch_server.py` to cite one real symbol accurately.

- [ ] **Step 1:** Author `LESSON_01["zh"]` to the content model (see spec "L01 scope"):
  1. `<p class="lead">` hook — SGLang = high-performance **serving** engine for LLM + multimodal.
  2. `<div class="card analogy">` 🔌 — a busy restaurant kitchen that **batches orders** and
     **reuses prepped ingredients** (continuous batching + prefix cache).
  3. `<div class="card macro">` 🌍 — two halves: frontend DSL (`lang/`) + runtime engine (`srt/`);
     fast via RadixAttention + zero-overhead scheduler + continuous batching.
  4. 2–4 `<h2>` sections with **≥3 visual blocks** (a `layers` of the stack
     DSL→Engine→Scheduler→ModelRunner→KV cache→kernels; a `cols` 前端 DSL vs 运行时引擎; a
     `vflow`/`flow` life-of-a-request teaser; a `table.t` 核心特性→它解决的瓶颈, each forward-ref'd
     to a later 第 N 课).
  5. One cited `.codefile` — small snippet + `file + symbol` caption (e.g.
     `python/sglang/srt/entrypoints/engine.py ::Engine`); explain, don't dump.
  6. `<div class="card key">` 本课要点 — bulleted recap.
  Target ≥3000 CJK chars (aim ~4000). Use only CSS classes defined in `shell.CSS`.
- [ ] **Step 2:** Author `LESSON_01["en"]` as a faithful parallel translation — **same diagram
  inventory** (every `layers`/`cols`/`vflow`/`table.t`/`.codefile`/cards present in both langs).
- [ ] **Step 3:** Self-check class usage: `cd src && python3 -c "import part1"` (no error).

## Task 7 — Write `README.md`

**Files:** Create `README.md` (model on `REF/README.md`, retargeted to SGLang).

- [ ] **Step 1:** Write a bilingual README: title "SGLang Visual Guide / SGLang 图解学习指南";
  one-line pitch (high-performance LLM/multimodal serving engine internals, bilingual, browser,
  zero-install); **Status: in progress (M0 — pipeline + L01 baseline)**; third-party/unofficial
  disclaimer (SGLang is Apache-2.0); a "What it covers" table of the **13 parts** (from the design
  spec §6); How to view (`cd src && python3 build.py`), How to print (`build_print.py`), Build &
  validate (all four scripts), Project structure block, dual-license note, and a 中文说明 section.
  Use placeholder badge URLs pointing at `verdenmax/sglang-visual-guide` (consistent with siblings).

## Task 8 — Build, validate, no-diff, commit

- [ ] **Step 1:** Build + validate:
```bash
cd ~/course/sglang-visual-guide/src
python3 build.py && python3 build_print.py && python3 check_html.py && python3 check_links.py
```
  Expected: build writes `index.html`, `lessons/01-what-is-sglang.html`, `print_zh.html`,
  `print_en.html`; `check_html.py` → "0 error(s), 0 warning(s)"; `check_links.py` passes.
  Fix any ERROR/WARN before proceeding (esp. undefined CSS class, unbalanced tags, <3000 CJK,
  <6 visual blocks).
- [ ] **Step 2:** No-diff rebuild guard:
```bash
cd ~/course/sglang-visual-guide && python3 src/build.py >/dev/null && git add -A && git status --short
```
  All generated files should be staged once; a second `build.py` must produce no further changes.
- [ ] **Step 3:** Commit the milestone:
```bash
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "M0: scaffold + pipeline — L01 What is SGLang baseline

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Guardrails
- Pure stdlib; no new dependencies. Relative links only.
- Cite `file + symbol`, never bare line numbers. Small snippets only; never paste large source.
- zh ≥ 3000 CJK (target ~4000); zh/en strict parity (identical diagram inventory).
- Every CSS class used must exist in `shell.CSS` (check_html enforces). Default page language zh.
- Do not touch `docs/` specs/plans or the reference repo. SGLang is Apache-2.0 — keep the disclaimer.

