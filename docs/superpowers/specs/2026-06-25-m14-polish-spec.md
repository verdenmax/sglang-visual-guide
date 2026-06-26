# M14 — Polish & Finalization — Spec + Plan

**Date:** 2026-06-25 · **Status:** ready · FINAL milestone (no new lessons).
**Companion to:** roadmap M14. Builds on M0–M13 (all 63 lessons L01–63 complete).

## Goal
The guide content is complete (63 lessons / 13 parts, validators 0/0, 252 links). M14 removes the
remaining "in progress / M0 baseline" staleness from the user-facing README and does a final
consistency sweep so the repo presents as a finished, source-verified guide.

## Findings from the final sweep (already done)
- Validators: 63 lessons + index, **0 err / 0 warn**; 252 internal links resolve; rebuild = no-diff.
- Print editions (`print_zh.html`, `print_en.html`) include all 63 lessons (build_print iterates
  `shell.PAGES` and asserts a `registry.CONTENT` entry for each); L01 + L63 confirmed present in both.
- Index shows all 13 parts (第一…第十三部分); pill "共 63 课 · 13 个部分".
- `check_html.MAX_LESSON=63` correct; no forward-refs beyond 63.
- No stale `milvus/mvg` references in `src/`; branding correct ("SGLang 图解教程 / SGLang Visual Guide",
  accent #7c48e6, favicon Sg).
- The only user-facing staleness is the **README status** (still says "in progress (M0 — pipeline +
  L01 baseline)"), plus a few "M0 ships only L01 / only part1 at M0" notes. (The per-milestone spec
  docs under `docs/superpowers/` legitimately describe their own milestone and are left as history.)
- `PLACEHOLDER` substring hits in `src/part11.py` are legitimate prose ("placeholder token"), not
  leftover scaffolding.

## Tasks (do myself, incrementally — docs polish)
1. **README — English status & notes:**
   - Replace the `> **Status:** in progress (M0 …)` block (≈L28–31) with a "complete" status: all 13
     parts / 63 lessons shipped, bilingual, validators green, source-verified.
   - Fix the part-table note (≈L60) "M0 ships only L01; the remaining lessons land milestone by
     milestone" → all 63 lessons complete.
   - Fix the project-structure note (≈L101) "(only part1 at M0)" → part1.py … part13.py all present.
2. **README — 中文 status & notes:**
   - Replace the `> **进度：** 进行中（M0 …）` block (≈L134–136) with the completed status.
   - Fix "M0 仅产出 L01，其余课程随后续里程碑逐步落地。" (≈L145) → 全部 63 课已完成。
3. **Verify** README renders as plain markdown (no build needed) and re-run the full validator suite
   once more to confirm nothing regressed; confirm no-diff after a clean rebuild.
4. **Commit** the README finalization (`verdenmax`, Co-authored-by Copilot).
5. **Light review** (self): re-read the README diff against the actual repo state; confirm every claim
   (lesson count, parts, zero-dep, license, print/PDF, build commands) is accurate.

## DoD
- README has no "in progress / M0 / baseline / only L01 / only part1" staleness; states the guide is
  complete (63 lessons / 13 parts) and source-verified.
- Validators 0/0; 252 links; no-diff after rebuild.
- Final commit pins the completed state. Guide is done.
