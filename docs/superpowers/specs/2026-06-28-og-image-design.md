# Design: og:image brand share card

Date: 2026-06-28
Status: approved (design), pending implementation

## Purpose

When a link to the SGLang Visual Guide is shared (Twitter/X, Slack, Discord,
WeChat, Facebook, etc.), it currently shows no preview image: `head_meta()` in
`src/shell.py` emits `og:title` / `og:description` / `twitter:card=summary` but
**no `og:image`**. Add a single, branded 1200×630 share card and wire the
social-meta tags so every page (index + all 63 lessons) renders a rich preview.

Scope: **one site-wide brand card** (not per-lesson). Per-lesson cards are
explicitly out of scope (possible future enhancement).

## Constraints

- **Zero-dependency site build.** CI (`.github/workflows/deploy.yml`,
  `ci.yml`) runs `python3 build.py && python3 build_print.py` on a plain
  `python-version: 3.x` with **no** Pillow/rsvg installed, then uploads the
  whole repo (`path: '.'`). Therefore the card must be a **pre-generated,
  committed PNG asset**; generation must NOT run inside `build.py`.
- **Absolute URL.** `og:image` must be an absolute URL. The site is served at a
  subpath: `https://verdenmax.github.io/sglang-visual-guide/`.
- **Reproducible + on-brand.** The guide is SVG-native; the card is authored as
  an SVG using the guide's literal palette and rendered with `rsvg-convert`
  (already used in this environment), so it matches the site visually and can be
  regenerated deterministically.

## Asset

- One file `og-cover.png`, 1200×630 (the Open Graph / Twitter
  `summary_large_image` standard size), committed at the **repo root**.
- Served at `https://verdenmax.github.io/sglang-visual-guide/og-cover.png`.

## Visual design (approved draft)

Dark brand card on the guide's dark palette:

- Background: vertical gradient `#0e1116 → #171a24`; a 8px top accent bar
  (`#7c48e6 → #a98af0`).
- Top-left: the `Sg` logo (rounded purple square, matches the favicon) +
  `SGLang Visual Guide` + `verdenmax.github.io/sglang-visual-guide`.
- Center-left: bilingual title `SGLang 图解教程` (white, ~76px, 800) + subtitle
  `看懂高性能 LLM 服务引擎内部` (accent `#a98af0`) + one-line content overview.
- Bottom-left: three stat pills — `63 lessons` (accent), `13 parts` (teal),
  `中文 / English` (neutral). **Counts are derived from `shell.PAGES`** at
  generation time (lesson count = len(PAGES); part count = distinct part labels)
  so they never drift.
- Right third: a small RadixAttention prefix-tree motif (purple root → purple
  inner nodes → teal leaves) with a `RadixAttention` label — the guide's
  signature concept.

Colors are taken from `src/shell.py` `:root` (brand `#7c48e6`, dark bg
`#0e1116`, panel `#161b22`, teal `#0d9488`, ink `#e6edf3`, muted `#9aa6b2`,
accent-ink `#c9b6f7`, teal dark `#5ed0c4`). Fonts via fontconfig:
`Noto Sans CJK SC` (CJK) with `DejaVu Sans` fallback for Latin.

## Components & data flow

```
shell.PAGES ──┐
              ▼
   src/build_og.py  ──(emits)──►  og card SVG string (literal colors,
   (manual/optional run)                 derived counts)
              │
              ▼  subprocess: rsvg-convert -w 1200 -h 630
        og-cover.png  (committed asset, served by Pages)
              ▲
              │ referenced by absolute URL
   src/shell.py head_meta() ──► og:image / twitter:image / og:url tags
              │
              ▼
   build.py (zero-dep) writes index.html + lessons/*.html with the tags
```

### `src/build_og.py` (new, separate from build.py)

- A standalone generator, run manually like `build_print.py` (NOT imported by
  `build.py`). Has a top-of-file docstring noting it requires `rsvg-convert`
  and is only needed when the card design changes.
- `card_svg()` returns the SVG string. Lesson/part counts derived from
  `shell.PAGES` (`LESSON_COUNT = len(shell.PAGES)`;
  `PART_COUNT = len({p[3] for p in shell.PAGES})`).
- `build()` writes the SVG to a temp/intermediate (`og-cover.svg`, committed for
  reproducibility) and shells out to `rsvg-convert` to produce `og-cover.png`.
  Exits non-zero with a clear message if `rsvg-convert` is missing.
- Deterministic: same PAGES → byte-identical SVG (PNG byte-stability is
  best-effort; the SVG is the source of truth and is committed).

### `src/shell.py` changes

- Add `SITE_URL = "https://verdenmax.github.io/sglang-visual-guide/"` (single
  source of truth for absolute URLs).
- Add `OG_IMAGE = SITE_URL + "og-cover.png"`.
- `head_meta(title, description, og_type="website", page_url=None)` gains an
  optional `page_url` (absolute URL of the current page). It additionally emits:
  - `<meta property="og:url" content="{page_url}">` (when provided)
  - `<meta property="og:image" content="{OG_IMAGE}">`
  - `<meta property="og:image:width" content="1200">`
  - `<meta property="og:image:height" content="630">`
  - `<meta property="og:image:alt" content="SGLang Visual Guide — 63-lesson bilingual illustrated guide">`
  - `<meta name="twitter:image" content="{OG_IMAGE}">`
  - and changes `twitter:card` from `summary` to **`summary_large_image`**.
- `page()` passes `page_url = SITE_URL + "lessons/" + filename`;
  `index_page()` passes `page_url = SITE_URL`.

## Validation (`src/check_html.py`)

Extend per-page checks to assert:

- exactly one `og:image` and one `twitter:image`, both equal to an **absolute**
  `https://…/og-cover.png` URL;
- `twitter:card` is `summary_large_image`;
- each page has an `og:url` that is absolute and ends in `.html` (lessons) or is
  the site root (index).

Add a one-time asset check (in `check_html.py main()` or a tiny standalone): the
referenced `og-cover.png` exists at repo root and is a valid PNG with size
1200×630 (read the 24-byte PNG IHDR header with stdlib `struct` — no Pillow, so
the check stays runnable in CI).

`check_links.py` already skips non-`.html` and external links, so the absolute
`og:image` URL won't be treated as a broken internal link (confirm).

## Files changed

- NEW `src/build_og.py` — card generator (SVG + rsvg-convert).
- NEW `og-cover.svg` — committed SVG source (reproducibility).
- NEW `og-cover.png` — committed 1200×630 asset (served by Pages).
- `src/shell.py` — `SITE_URL`, `OG_IMAGE`, `head_meta()` image/url tags,
  `summary_large_image`, `page()`/`index_page()` pass `page_url`.
- `src/check_html.py` — og:image / twitter:image / og:url / card guards + PNG
  IHDR asset check.
- `README.md` — note `build_og.py` + the share card (brief).

## Testing / acceptance

1. `python3 build_og.py` produces `og-cover.png` (1200×630); visually matches
   the approved draft; CJK renders (not tofu).
2. `python3 build.py && python3 build_print.py && python3 check_html.py &&
   python3 check_links.py && python3 check_citations.py` all pass 0/0.
3. Built `index.html` and a sample lesson each contain exactly one absolute
   `og:image`/`twitter:image` = `…/og-cover.png`, `twitter:card=
   summary_large_image`, and an absolute `og:url`.
4. Determinism: re-running `build.py` is byte-identical; re-running `build_og.py`
   yields a byte-identical `og-cover.svg`.
5. Guards have teeth: removing `og:image` or making it relative, or downgrading
   `twitter:card`, makes `check_html.py` error.
6. After push: validate live with a card debugger (e.g. opengraph.xyz or the
   raw tags) — `og:image` resolves to a 200 PNG; preview renders.

## Out of scope

- Per-lesson share cards / dynamic per-page images.
- Running card generation inside `build.py` or CI (keeps build zero-dep).
- Favicon changes (the existing `Sg` favicon stays).
