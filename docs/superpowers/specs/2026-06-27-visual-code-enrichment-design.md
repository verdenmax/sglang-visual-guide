# SGLang Visual Guide — Visual & Code Enrichment Pass — Design

**Date:** 2026-06-27 · **Status:** approved (pending spec review)
**Author:** verdenmax (with Copilot) · **Type:** enhancement to the existing 63-lesson guide.

## Problem
The guide (63 lessons / 13 parts) ships with strong box-diagrams (`cols`/`flow`/`layers`/`cellgroup`/
`table.t`) and **exactly one** cited `.codefile` per lesson, but **zero SVG figures**. The two
reference guides this project models (`milvus-visual-guide`, `hermes-agent-visual-guide`) lean heavily
on hand-drawn inline **SVG figures** (Milvus ≈124, Hermes ≈138 — ~1–2 per language per lesson) for the
things box-diagrams cannot show: geometry, real-number charts, spatial/architecture layout, before/
after, sequence-over-time. The SGLang guide therefore feels code-light and visually thinner than its
models. (SVG was skipped during the original build only to avoid subagent output truncation.)

## Goal
Bring every lesson up to the reference guides' visual + code richness, **adding only** (never removing
existing content):
1. **~2 SVG `.fig` figures per language** per lesson.
2. **A 2nd cited `.codefile`** per lesson (a different real symbol).
3. **1–2 more concrete inline examples** (real numbers or a short snippet) in prose.
Enforced going forward by a new **figure floor** in the validator (codefile count NOT enforced).

## The SVG figure pattern (house style — already CSS-supported)
`shell.CSS` already styles `.fig` (panel + border + shadow, centered), `.fig svg` (responsive:
`max-width:100%; height:auto`), `.fig svg text { fill: var(--ink) }`, and `.fig .figcap` (caption,
`b`→accent). The required markup, per figure:

```html
<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="<one-sentence description, localized>">
    <!-- shapes + text using ONLY CSS variables for color -->
  </svg>
  <div class="figcap"><b>图N · 标题</b> — 一句话说明这张图在讲什么。</div>
</div>
```

**Theming rule (CRITICAL):** every color MUST be a CSS variable so figures render in BOTH light and
dark themes. Hardcoded hex (`#fff`, `#333`) is forbidden. Allowed palette:
- structure/lines: `var(--line)`, `var(--muted)`, `var(--faint)`, `var(--panel-2)`
- emphasis / primary: `var(--accent)`, `var(--accent-soft)` (fills), `var(--accent-ink)` (text)
- text default is `var(--ink)` (from `.fig svg text`); set explicitly for emphasis
- color-coding (use sparingly, semantically): `var(--amber)/--amber-soft`, `var(--blue)/--blue-soft`,
  `var(--purple)/--purple-soft`, `var(--teal)/--teal-soft`, `var(--red)/--red-soft`
- never set a background on the svg (the `.fig` panel provides it).

**Sizing:** `viewBox="0 0 W H"` (typical W≈680–820, H≈220–360); no width/height attributes (responsive
via CSS). Keep each SVG focused (~15–45 primitives); two small clear figures beat one cluttered one.

**Bilingual parity:** the en figure is the zh figure with the SAME geometry/coordinates and TRANSLATED
`<text>` + `aria-label` + `figcap`. So each lesson's zh block has 2 `.fig`, en block has 2 `.fig`
(identical inventory, the existing zh/en rule). `.fig` does NOT replace the existing box-diagrams — it
ADDS to them.

### What each figure should depict (per-lesson, illustrative not decorative)
Choose 2 that a box-diagram can't capture, e.g.:
- a **chart with real numbers** (e.g. throughput vs batch size; memory vs sequence length; accept-rate
  curve; p50/p99 latency bars);
- a **geometry/spatial** view (paged KV blocks in memory; a radix tree; a token tree; attention tiling;
  TP sharding of a matrix);
- a **before/after** (fragmented vs paged KV; unfused vs fused kernel; serial vs overlapped schedule);
- an **annotated architecture/sequence** (the request path; PD disaggregation transfer; the overlap
  pipeline timeline) — richer than the box `flow` because it can show timing, sizes, proportions.

## The 2nd codefile (code density)
Add one more `.codefile` per lesson citing a **different real symbol** from that lesson's source area
(same `file ::symbol` caption convention; condensed faithfully; docstrings→`#`; escape `<`/`>`/`&`).
It should illuminate a *different facet* than the existing codefile (e.g. the data structure vs the
function that uses it; the caller vs the callee; the config vs the runtime). Real symbols only — the
orchestrator pre-fetches/verifies each citation before dispatch (as in the original build).

## Inline examples (concrete prose)
Add 1–2 inline concretizations per lesson: a worked number ("vocab=150k, hidden=8192 ⇒ lm_head ≈
1.2B params"), a tiny `<span class="mono">…</span>` snippet, or a short `<pre>`-free pseudo-line, so an
abstraction lands. These raise CJK slightly (fine; floor stays 3000 soft / ~3600 target).

## Validator changes (`src/check_html.py`)
- Add `fig` to `DIAGRAM_CLASSES` (so figures also count toward the existing `MIN_DIAGRAMS=6`).
- Add `MIN_FIGURES = 4` (≥2 per language, counting both languages) gating `class="fig"`; emit a
  hard **error** (not warn) when a non-exempt lesson has fewer, so the additions can't regress.
- `SOFT_EXEMPT = {"57-glossary.html"}` is also exempt from `MIN_FIGURES` (it's a reference table; we
  add 1 optional overview SVG but don't require 4).
- Do NOT add any codefile-count gate (user decision).
- Keep `MIN_CJK` (3000 soft warn) unchanged.

## Execution (per-part enhancement pass, mirrors the original milestones)
Roll out **Part 1 → Part 13** as enhancement milestones (call them E1…E13). Per part:
1. Orchestrator pre-fetches/verifies the 2nd-codefile symbol for each lesson + decides the 2 figure
   subjects (from the list above).
2. One **subagent per lesson** (model claude-opus-4.8, sync, high effort): ADD 2 zh `.fig` + 2 en
   `.fig` (CSS-vars only), 1 more `.codefile` (verified symbol), 1–2 inline examples — WITHOUT
   touching existing blocks. Provide the exact `.fig` template + palette + a worked SVG example.
   Mitigate SVG truncation: one lesson per agent; ask for 2 FOCUSED svgs; draft+expand fallback after
   3 truncations (orchestrator drafts the SVG skeleton, subagent deepens labels/prose).
3. Verify: `build.py && build_print.py && check_html.py && check_links.py` → 0/0 (new figure floor
   passes); no-diff after rebuild; zh/en `.fig` counts equal.
4. **Two-stage review** (research concept + code-review), opus-4.8 max, long context, per part —
   adding two checks: (a) every SVG uses ONLY CSS vars (grep for `#` hex / hardcoded colors), (b) the
   2nd codefile cites a real symbol. Fix → re-verify → commit.

Order within the pass may start with a **pilot: Part 1 (L01–03)** to lock the SVG template/quality bar,
then proceed E2…E13. The validator change lands in E1 (after L01–03 have their figures, so the build
stays green).

## Out of scope (YAGNI)
- No new lessons, no re-themeing, no restructuring of existing diagrams/prose.
- No animation/JS in figures (static SVG only — matches the reference + print/PDF compatibility).
- No raster images (keeps zero-dependency + crisp print).

## Definition of done
- All 63 lessons (glossary exempt from the floor) have ≥2 `.fig` SVG per language + ≥2 `.codefile`,
  plus added inline examples; figures use only CSS vars and render in light & dark.
- `check_html.py` enforces `MIN_FIGURES`; validators 0 err/0 warn; 252+ links resolve; no-diff.
- Print editions still build and include the figures. Existing content preserved.
