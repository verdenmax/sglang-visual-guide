# M0 — Scaffold & Pipeline — Spec

**Date:** 2026-06-25 · **Status:** ready to plan
**Companion to:** `specs/2026-06-25-sglang-visual-guide-design.md`, roadmap M0.

## Goal
Stand up the **full zero-dependency Python pipeline + project chrome**, proving the toolchain
end-to-end by shipping **L01 (What is SGLang)** as the only lesson. Everything is cloned from
`~/course/milvus-visual-guide/` and re-themed; no content beyond L01.

## Source of truth
The reference repo `~/course/milvus-visual-guide/` is the template. Infra files are copied (verbatim
or near-verbatim); the design system (`shell.CSS`) is subject-agnostic and carried over unchanged
except for the accent palette and chrome strings. **Content** files (`part*.py`, the per-lesson
`quizzes.QUIZZES` body, `SUBTITLES`) are written fresh for SGLang.

## File inventory (copy / adapt / write-fresh)
| File | Action | Notes |
| --- | --- | --- |
| `src/build.py` | copy verbatim | subject-agnostic |
| `src/build_print.py` | copy verbatim | subject-agnostic |
| `src/check_links.py` | copy verbatim | subject-agnostic |
| `src/check_html.py` | adapt | `MAX_LESSON=63`; `SOFT_EXEMPT={"57-glossary.html"}`; rest unchanged |
| `src/shell.py` | adapt | retheme accent + chrome strings; `PAGES`/`SUBTITLES` seeded with L01 only |
| `src/registry.py` | write-fresh | `import part1`; `CONTENT={"01-what-is-sglang.html": part1.LESSON_01}` |
| `src/quizzes.py` | write-fresh | copy the framework (`_shuffle`, `render`, head strings); only the L01 quiz body |
| `src/part1.py` | write-fresh | `LESSON_01` bilingual content to the content model |
| `.github/workflows/ci.yml` | adapt | copy; no milvus-specific names to change (paths are generic) |
| `.github/workflows/deploy.yml` | copy verbatim | Pages deploy |
| `.gitignore` | copy verbatim | `__pycache__`, `*.pyc`, `*.pdf`, `.venv` |
| `LICENSE` | copy verbatim | MIT, "Copyright (c) 2026 verdenmax" |
| `LICENSE-CONTENT` | copy verbatim | CC BY 4.0 |
| `README.md` | write-fresh | SGLang pitch, build/validate/print, structure, dual license, 中文说明 |

## Theme tokens (SGLang identity) — exact changes in `shell.py`
- **Accent palette** (light): `--accent: #7c48e6; --accent-soft: #efe9fc; --accent-ink: #5b34b0;`
  Derive dark-mode equivalents the same way milvus does (brighter accent, translucent soft, light ink).
- **favicon**: rect `fill='#7c48e6'`; glyph text `Sg` (keep the rounded-rect SVG).
- **theme-color** meta: `#7c48e6`.
- **og:site_name**: `SGLang 图解教程`.
- **Topbar mark + name**: `⚡ SGLang 图解教程 / SGLang Visual Guide` (replace `🐦 Milvus …`).
- **Lesson `title_tag` / `desc`**: replace `Milvus 图解教程` → `SGLang 图解教程`.
- **index_page title/desc/h1/disclaimer**: retarget to SGLang; disclaimer keeps Apache-2.0 wording
  (SGLang is Apache-2.0). "Verified against the real sgl-project/sglang source; references cite
  file + symbol".

## Invariants (from design spec, enforced by validators)
- `shell.PAGES` ↔ `registry.CONTENT` ↔ `quizzes.QUIZZES` aligned (no orphan keys / missing entries).
- Generated `index.html` + `lessons/01-what-is-sglang.html` committed; rebuild is byte-identical.
- Relative `href`; bilingual `data-lang` toggle; default zh.
- `check_html.py` + `check_links.py` pass with **0 errors**; L01 also targets **0 warnings**
  (≥3000 zh CJK, ≥6 visual blocks across both langs, analogy + key-points cards).

## L01 scope — 01-what-is-sglang.html / "What is SGLang"
Definition (high-performance LLM + multimodal **serving** framework); the two halves (frontend DSL
`lang/` + serving runtime `srt/`) at 10,000 ft; why it is fast (one-line tour of RadixAttention,
zero-overhead scheduler, continuous batching — each forward-referenced to its later lesson); where
it runs (single GPU → large clusters; broad hardware). Pedagogy: lead → 🔌 analogy (a busy
restaurant kitchen that batches & reuses prep) → 🌍 macro card → 2–4 `<h2>` with ≥3 diagrams/lang
(a `layers` of the stack, a `cols` DSL-vs-runtime, a `vflow`/`flow` life-of-a-request teaser, a
`table.t` of core features→benefit) → one cited `.codefile` (e.g. `srt/entrypoints/engine.py
::Engine` or the `Runtime`/`launch_server` entry) → 本课要点 key-points card. Forward-refs allowed
(MAX_LESSON=63). Source-accurate: read SGLang `README.md` + `docs_new/docs/get-started/**`.

## Definition of done
`build.py` + `build_print.py` run; `check_html.py` + `check_links.py` → 0 err / 0 warn; rebuild
no-diff; index pill reads "共 1 课 · 1 个部分"; committed. (GitHub Pages must be enabled once in
repo Settings before deploy.yml can publish — note for later, not an M0 blocker.)
