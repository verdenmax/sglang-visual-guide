# og:image Brand Share Card — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single branded 1200×630 share card (`og-cover.png`) and wire the social-meta tags so links to the guide render a rich preview on Twitter/Slack/Discord/WeChat/etc.

**Architecture:** The card is authored as a brand-palette SVG and rendered to PNG with `rsvg-convert` by a standalone generator (`src/build_og.py`, run manually like `build_print.py`). The committed `og-cover.png` is the served asset, so the site build (`build.py`) stays zero-dependency in CI. `shell.head_meta()` references the asset by absolute URL; `check_html.py` guards the tags and the asset.

**Tech Stack:** Python 3 stdlib only for the site build; `rsvg-convert` (librsvg) + Noto Sans CJK fonts for the (manual, optional) card generation; GitHub Pages for hosting.

---

## File Structure

- **NEW `src/build_og.py`** — standalone card generator. Emits `og-cover.svg` (committed source) and shells out to `rsvg-convert` to produce `og-cover.png`. Derives lesson/part counts from `shell.PAGES`. NOT imported by `build.py`.
- **NEW `og-cover.svg`** (repo root) — generated SVG source, committed for reproducibility.
- **NEW `og-cover.png`** (repo root) — generated 1200×630 asset, committed and served at `https://verdenmax.github.io/sglang-visual-guide/og-cover.png`.
- **MODIFY `src/shell.py`** — add `SITE_URL` + `OG_IMAGE`; extend `head_meta()` with `og:url`/`og:image*`/`twitter:image` and upgrade `twitter:card` to `summary_large_image`; thread `page_url` from `page()` and `index_page()`.
- **MODIFY `src/check_html.py`** — add `check_social_meta()` (per page) + `check_og_asset()` (PNG IHDR) guards.
- **MODIFY `README.md`** — one line documenting `build_og.py` + the share card.

---

## Task 1: Card generator + committed asset

**Files:**
- Create: `src/build_og.py`
- Generates (committed): `og-cover.svg`, `og-cover.png` (repo root)

- [ ] **Step 1: Write `src/build_og.py`**

```python
#!/usr/bin/env python3
"""Generate the og:image brand share card (og-cover.png, 1200x630).

Run manually when the card design changes:  python3 build_og.py
Requires `rsvg-convert` (librsvg) on PATH and Noto Sans CJK fonts installed.
This is intentionally NOT imported by build.py, so the site build stays
zero-dependency (the committed og-cover.png is the served asset).
"""
import os
import shutil
import subprocess
import sys

import shell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG_PATH = os.path.join(ROOT, "og-cover.svg")
PNG_PATH = os.path.join(ROOT, "og-cover.png")

LESSON_COUNT = len(shell.PAGES)
PART_COUNT = len({p[3] for p in shell.PAGES})


def card_svg():
    """Return the 1200x630 share-card SVG (literal brand colors from shell CSS)."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" font-family="Noto Sans CJK SC, DejaVu Sans, sans-serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0e1116"/>
      <stop offset="1" stop-color="#171a24"/>
    </linearGradient>
    <linearGradient id="acc" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#7c48e6"/>
      <stop offset="1" stop-color="#a98af0"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <rect x="0" y="0" width="1200" height="8" fill="url(#acc)"/>
  <rect x="80" y="74" width="76" height="76" rx="17" fill="url(#acc)"/>
  <text x="118" y="127" font-size="40" font-weight="800" fill="#ffffff" text-anchor="middle">Sg</text>
  <text x="176" y="105" font-size="27" font-weight="700" fill="#e6edf3">SGLang Visual Guide</text>
  <text x="176" y="138" font-size="22" fill="#9aa6b2">verdenmax.github.io/sglang-visual-guide</text>
  <text x="80" y="300" font-size="76" font-weight="800" fill="#ffffff">SGLang 图解教程</text>
  <text x="80" y="372" font-size="40" font-weight="700" fill="#a98af0">看懂高性能 LLM 服务引擎内部</text>
  <text x="80" y="430" font-size="27" fill="#9aa6b2">从前端 DSL、调度器、RadixAttention 到高性能算子 · 中英双语</text>
  <g font-size="25" font-weight="700">
    <rect x="80" y="486" width="168" height="52" rx="26" fill="#241a3e" stroke="#7c48e6"/>
    <text x="164" y="520" fill="#c9b6f7" text-anchor="middle">{LESSON_COUNT} lessons</text>
    <rect x="262" y="486" width="150" height="52" rx="26" fill="#0d2e2a" stroke="#0d9488"/>
    <text x="337" y="520" fill="#5ed0c4" text-anchor="middle">{PART_COUNT} parts</text>
    <rect x="426" y="486" width="196" height="52" rx="26" fill="#161b22" stroke="#2a323c"/>
    <text x="524" y="520" fill="#e6edf3" text-anchor="middle">中文 / English</text>
  </g>
  <g stroke="#3a2f5c" stroke-width="3" fill="none">
    <path d="M970 150 L870 250 M970 150 L1070 250 M870 250 L820 360 M870 250 L930 360 M1070 250 L1070 360"/>
  </g>
  <g>
    <circle cx="970" cy="150" r="30" fill="url(#acc)"/>
    <circle cx="870" cy="250" r="26" fill="#161b22" stroke="#7c48e6" stroke-width="3"/>
    <circle cx="1070" cy="250" r="26" fill="#161b22" stroke="#7c48e6" stroke-width="3"/>
    <circle cx="820" cy="360" r="22" fill="#0d2e2a" stroke="#0d9488" stroke-width="3"/>
    <circle cx="930" cy="360" r="22" fill="#0d2e2a" stroke="#0d9488" stroke-width="3"/>
    <circle cx="1070" cy="360" r="22" fill="#0d2e2a" stroke="#0d9488" stroke-width="3"/>
  </g>
  <text x="970" y="448" font-size="22" fill="#6e7a86" text-anchor="middle">RadixAttention</text>
</svg>
"""


def build():
    if not shutil.which("rsvg-convert"):
        sys.exit("build_og error: rsvg-convert not found (install librsvg). "
                 "The committed og-cover.png is the served asset; only regenerate "
                 "when the card design changes.")
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(card_svg())
    subprocess.run(
        ["rsvg-convert", "-w", "1200", "-h", "630", SVG_PATH, "-o", PNG_PATH],
        check=True,
    )
    print(f"wrote {os.path.relpath(SVG_PATH, ROOT)} and "
          f"{os.path.relpath(PNG_PATH, ROOT)} "
          f"({LESSON_COUNT} lessons / {PART_COUNT} parts)")


if __name__ == "__main__":
    build()
```

- [ ] **Step 2: Generate the asset**

Run: `cd src && python3 build_og.py`
Expected: `wrote og-cover.svg and og-cover.png (63 lessons / 13 parts)`

- [ ] **Step 3: Verify PNG dimensions + CJK rendered (not tofu)**

Run:
```bash
cd /home/verden/course/sglang-visual-guide && python3 - <<'PY'
import struct
h=open("og-cover.png","rb").read(24)
assert h[:8]==b"\x89PNG\r\n\x1a\n" and h[12:16]==b"IHDR", "not PNG"
w,ht=struct.unpack(">II",h[16:24]); print("size",w,ht); assert (w,ht)==(1200,630)
from PIL import Image
im=Image.open("og-cover.png").convert("RGB")
cnt=sum(1 for x in range(80,760) for y in range(240,300) if im.getpixel((x,y))!=(14,17,22))
print("title non-bg px:",cnt); assert cnt>5000, "CJK title not rendered"
print("OK")
PY
```
Expected: `size 1200 630`, `title non-bg px: <large>`, `OK`

- [ ] **Step 4: Eyeball the card matches the approved draft**

Open `og-cover.png` (view tool / image viewer). Confirm: dark brand card, `Sg` logo + URL, bilingual title, tagline, three pills (`63 lessons` / `13 parts` / `中文 / English`), RadixAttention tree on the right. No clipped text, no tofu glyphs.

- [ ] **Step 5: Commit**

```bash
cd /home/verden/course/sglang-visual-guide
git add src/build_og.py og-cover.svg og-cover.png
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "Add og:image card generator + asset (build_og.py, og-cover.png)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Wire social-meta tags (+ guard, TDD red→green)

**Files:**
- Modify: `src/check_html.py` (add `check_social_meta()` + calls) — write the guard first
- Modify: `src/shell.py:1-44` (`SITE_URL`, `OG_IMAGE`, `head_meta()`), `src/shell.py:642` (`page()` call), `src/shell.py:836` (`index_page()` call)

- [ ] **Step 1: Add the guard `check_social_meta()` to `src/check_html.py`**

Add this function next to the other `check_*` helpers (e.g. after `check_balance`, ~line 115):

```python
def check_social_meta(name, html):
    """og:image / twitter:image must be the exact absolute card URL, the card
    must be summary_large_image, and og:url must be present and absolute."""
    for prop in ("og:image", "twitter:image"):
        attr = "property" if prop.startswith("og:") else "name"
        n = html.count(f'<meta {attr}="{prop}" content="{shell.OG_IMAGE}">')
        if n != 1:
            add("ERR", name, f"{prop} should appear once as {shell.OG_IMAGE!r} (found {n})")
    if '<meta name="twitter:card" content="summary_large_image">' not in html:
        add("ERR", name, "twitter:card must be summary_large_image")
    m = re.search(r'<meta property="og:url" content="([^"]*)">', html)
    if not m:
        add("ERR", name, "missing og:url")
    elif not m.group(1).startswith("https://"):
        add("ERR", name, f"og:url not absolute: {m.group(1)!r}")
```

Wire it into `check_lesson()` (add near the other meta checks, ~line 208):

```python
    check_social_meta(fname, html)
```

And into `main()` for the index, right after `check_classes("index.html", idx)` (~line 288):

```python
    check_social_meta("index.html", idx)
```

- [ ] **Step 2: Run the guard — expect RED (proves teeth)**

Run: `cd src && python3 build.py >/dev/null && python3 check_html.py`
Expected: FAIL — many `og:image should appear once ... (found 0)`, `twitter:card must be summary_large_image`, `missing og:url` errors (tags not emitted yet). `shell.OG_IMAGE` must already exist for the guard to import — if `check_html.py` errors with `AttributeError: module 'shell' has no attribute 'OG_IMAGE'`, that confirms Step 3 is needed next; proceed.

> Note: `shell.OG_IMAGE` is referenced by the guard, so add the constant (Step 3) before re-running. The RED state is: constant present, tags absent → guard fires.

- [ ] **Step 3: Implement the meta tags in `src/shell.py`**

Add the constants just above `def head_meta` (~line 28):

```python
SITE_URL = "https://verdenmax.github.io/sglang-visual-guide/"
OG_IMAGE = SITE_URL + "og-cover.png"
```

Replace `head_meta` (lines 29-44) with:

```python
def head_meta(title, description, og_type="website", page_url=None):
    """SEO / social meta tags + favicon for a page <head>."""
    t = esc(title)
    d = esc(description)
    url_tag = f'<meta property="og:url" content="{page_url}">\n' if page_url else ""
    return (
        f'<meta name="description" content="{d}">\n'
        f'<meta name="theme-color" content="#7c48e6">\n'
        f'<link rel="icon" type="image/svg+xml" href="{FAVICON}">\n'
        f'<meta property="og:type" content="{og_type}">\n'
        f'<meta property="og:site_name" content="SGLang 图解教程">\n'
        f'<meta property="og:title" content="{t}">\n'
        f'<meta property="og:description" content="{d}">\n'
        f'{url_tag}'
        f'<meta property="og:image" content="{OG_IMAGE}">\n'
        f'<meta property="og:image:width" content="1200">\n'
        f'<meta property="og:image:height" content="630">\n'
        f'<meta property="og:image:alt" content="SGLang Visual Guide — {len(PAGES)}-lesson bilingual illustrated guide">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{t}">\n'
        f'<meta name="twitter:description" content="{d}">\n'
        f'<meta name="twitter:image" content="{OG_IMAGE}">'
    )
```

> `PAGES` is defined later in the file but `head_meta` is only *called* at build time (after module load), so `len(PAGES)` resolves fine.

Thread `page_url` at the two call sites. In `page()` (line 642), change:

```python
{head_meta(title_tag, desc, og_type="article", page_url=SITE_URL + "lessons/" + filename)}
```

In `index_page()` (line 836), change:

```python
{head_meta(title_tag, desc, og_type="website", page_url=SITE_URL)}
```

- [ ] **Step 4: Run the guard — expect GREEN**

Run: `cd src && python3 build.py >/dev/null && python3 build_print.py >/dev/null && python3 check_html.py && python3 check_links.py | tail -1 && python3 check_citations.py | tail -1`
Expected: `Checked 63 lessons + index - 0 error(s), 0 warning(s).`, `all … internal links resolve`, `all citations resolve`.

- [ ] **Step 5: Spot-check the emitted tags + determinism**

Run:
```bash
cd /home/verden/course/sglang-visual-guide
grep -o '<meta [^>]*og:image[^>]*>\|twitter:card[^>]*>\|og:url[^>]*>' index.html lessons/01-what-is-sglang.html | sort -u
cd src && python3 build.py >/dev/null && a=$(cat ../lessons/*.html ../index.html|sha256sum); python3 build.py >/dev/null && b=$(cat ../lessons/*.html ../index.html|sha256sum); [ "$a" = "$b" ] && echo DETERMINISTIC
```
Expected: index shows `og:url` = site root, lesson shows `og:url` = `…/lessons/01-…html`, both show `og:image` = `…/og-cover.png` and `twitter:card=summary_large_image`; `DETERMINISTIC`.

- [ ] **Step 6: Commit**

```bash
cd /home/verden/course/sglang-visual-guide
git add src/shell.py src/check_html.py lessons/ index.html print_en.html print_zh.html
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "Wire og:image/twitter:image/og:url meta + summary_large_image (+guard)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: PNG asset integrity guard (IHDR, stdlib only)

**Files:**
- Modify: `src/check_html.py` (add `import struct`; add `check_og_asset()`; call it in `main()`)

- [ ] **Step 1: Add `import struct`**

At the top of `src/check_html.py` (with the other stdlib imports, ~line 28-30), add:

```python
import struct
```

- [ ] **Step 2: Add `check_og_asset()`**

Add near the other module-level `check_*` helpers (e.g. after `check_social_meta`):

```python
def check_og_asset():
    """The committed og-cover.png must exist and be a valid 1200x630 PNG
    (read the IHDR header with stdlib struct — no Pillow, so this runs in CI)."""
    path = os.path.join(ROOT, "og-cover.png")
    if not os.path.exists(path):
        add("ERR", "og-cover.png", "asset missing (run build_og.py)")
        return
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        add("ERR", "og-cover.png", "not a valid PNG")
        return
    w, h = struct.unpack(">II", head[16:24])
    if (w, h) != (1200, 630):
        add("ERR", "og-cover.png", f"size {w}x{h}, expected 1200x630")
```

> `ROOT` is already defined in `check_html.py` (used for `lessons/`); confirm and reuse it. If `ROOT` points at `src/`, use `os.path.dirname(ROOT)` instead so the path resolves to the repo root where `og-cover.png` lives.

Call it once in `main()` (e.g. right after the index checks, near line 300):

```python
    check_og_asset()
```

- [ ] **Step 3: Run — expect GREEN (asset exists from Task 1)**

Run: `cd src && python3 check_html.py`
Expected: `Checked 63 lessons + index - 0 error(s), 0 warning(s).`

- [ ] **Step 4: Verify the guard has teeth**

Run:
```bash
cd /home/verden/course/sglang-visual-guide
mv og-cover.png /tmp/og-cover.bak.png
cd src && python3 check_html.py; echo "exit=$?"
mv /tmp/og-cover.bak.png ../og-cover.png
python3 check_html.py | tail -1
```
Expected: with the asset moved → `[ERR] og-cover.png: asset missing` and non-zero exit; after restore → `0 error(s)`.

- [ ] **Step 5: Commit**

```bash
cd /home/verden/course/sglang-visual-guide
git add src/check_html.py
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "check_html: assert og-cover.png exists + is 1200x630 (stdlib IHDR)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: README note + push + live verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the generator + card in `README.md`**

Find the build/scripts section (search for `build_print.py` or `build.py` in `README.md`) and add a sibling line. If a "Scripts" / "构建" list exists, add:

```markdown
- `src/build_og.py` — regenerate the social share card `og-cover.png` (1200×630, needs `rsvg-convert`); the committed PNG is the served `og:image`.
```

If no such list exists, add a short subsection near the build instructions:

```markdown
### Social share card

`og-cover.png` (1200×630) is the Open Graph / Twitter preview image, referenced
by absolute URL in every page's `<head>`. It is generated from a brand-palette
SVG by `python3 src/build_og.py` (requires `rsvg-convert`); the committed PNG is
the served asset, so the normal `build.py` stays dependency-free.
```

- [ ] **Step 2: Full local validation**

Run: `cd src && python3 build.py >/dev/null && python3 build_print.py >/dev/null && python3 check_html.py && python3 check_links.py | tail -1 && python3 check_citations.py | tail -1`
Expected: all pass, `0 error(s), 0 warning(s)`.

- [ ] **Step 3: Commit + push**

```bash
cd /home/verden/course/sglang-visual-guide
git add README.md
git -c user.name="verdenmax" -c user.email="verdenmax@users.noreply.github.com" \
  commit -m "README: document build_og.py + social share card

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git -c core.sshCommand="ssh -p 22" push origin master
```

- [ ] **Step 4: Verify live (after Pages deploy, ~60-90s)**

Run:
```bash
sleep 80
curl -sI "https://verdenmax.github.io/sglang-visual-guide/og-cover.png" | head -3
curl -s "https://verdenmax.github.io/sglang-visual-guide/" | grep -o 'og:image[^>]*>\|twitter:card[^>]*>'
```
Expected: PNG returns `HTTP/2 200` + `content-type: image/png`; the index serves the `og:image` (…/og-cover.png) and `twitter:card content="summary_large_image"`.

- [ ] **Step 5: Confirm the social preview renders**

Paste `https://verdenmax.github.io/sglang-visual-guide/` into a card debugger (e.g. https://www.opengraph.xyz/ or the platform's own validator) and confirm the brand card shows as a large image. (Manual; no code.)

---

## Self-Review (completed by plan author)

**Spec coverage:** asset (T1) · visual (T1, approved draft) · generator `build_og.py` separate from build.py (T1) · `SITE_URL`/`OG_IMAGE`/`og:image*`/`twitter:image`/`og:url`/`summary_large_image` (T2) · per-page + index guards (T2) · PNG IHDR asset check (T3) · README (T4) · live verification (T4). All spec sections map to a task.

**Placeholder scan:** none — every code/command step is concrete.

**Type/name consistency:** `OG_IMAGE`/`SITE_URL` defined in T2 Step 3 and referenced by the T2 guard (Step 1) and `page()`/`index_page()`; `check_social_meta` defined and wired in T2; `check_og_asset` defined and called in T3; `og-cover.png` path consistent across T1/T3/T4. `len({p[3] for p in shell.PAGES})` == `index_page`'s `len(order)` == 13.

**Known sequencing note:** T2 wires the guard before the tags exist, but the guard references `shell.OG_IMAGE`; the task instructs adding the constant (Step 3) so the RED state is "constant present, tags absent." Implementer should add the constant first if the guard errors on import, then observe RED, then add the tags for GREEN.
