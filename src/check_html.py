"""Structural / consistency regression guard for the generated HTML.

Run after build.py:
    cd src && python check_html.py

Exits non-zero on any ERROR (used by CI). WARN/INFO print but don't fail.
Checks each lesson + index:
* balanced tags (div/details/table/pre/summary + inline strong/span/em/b/code)
  and details<->summary
* a <title> + meta description; exactly one <h1> per lesson
* both languages present (lang-zh and lang-en blocks)
* no unescaped '<' inside <pre> code blocks (and none in prose either: a '<'
  not starting a real tag/comment must be written &lt;)
* cross-references "第 N 课" within 1..MAX_LESSON (forward refs allowed)
* nav prev/next chain matches shell.PAGES order
* index TOC lists every page; '共 N 课 · N 个部分' pill matches PAGES
* registry CONTENT has non-empty zh+en for every PAGES filename (no orphan keys)
* zh/en inventory parity: equal .fig and .codefile counts, equal <h2> counts
* every <div class="fig"> has a <div class="figcap">
* every inline <svg> has viewBox (no width/height attr); no hardcoded fill/
  stroke colors (var(--...) only, for dark mode); <text> does not clip the
  viewBox (width estimated from an Arial-ish per-char model + Noto safety)
* (WARN) every lesson has a key-points card and an analogy card

The "第 N 课" cross-ref check matches Chinese-Arabic digits only (e.g. "第 12 课");
English ("Lesson N") or Chinese-numeral references are not range-checked.
"""
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import shell  # noqa: E402
from registry import CONTENT  # noqa: E402

PAGES = shell.PAGES
ORDER = [p[0] for p in PAGES]
TOTAL = len(PAGES)
MAX_LESSON = 63  # final planned lesson count; cross-refs may point forward
MIN_CONTENT = 80  # min chars of zh/en source content per lesson (catch empty translations)

PRE_INLINE = ("span", "strong", "b", "em", "u", "a")
SOFT_EXEMPT = {"57-glossary.html"}

# Visual-block density (soft): containers that count as a "diagram/table".
DIAGRAM_CLASSES = ("layers", "vflow", "flow", "cols", "cellgroup", "timeline", "trace", "fig")
MIN_DIAGRAMS = 6  # per lesson, counting BOTH languages (>= 3 per language)
MIN_FIGURES = 4  # SVG .fig per lesson, counting BOTH languages (>= 2 per language)
MIN_CJK = 3000  # per-lesson zh CJK chars (soft floor; authoring target ~4000+)
SVG_OVERFLOW_MARGIN = 6  # px past viewBox before flagging a <text> clip (calibrated)

# Every class used in generated HTML must be defined in shell.CSS. Catches
# consolidation artifacts (e.g. a diagram-variant whose CSS was never merged in,
# rendering silently broken). Whitelist intentional no-style hooks.
CSS_DEFINED = set(re.findall(r"\.([A-Za-z][\w-]*)", shell.CSS))
CSS_CLASS_WHITELIST = {"prev"}  # footnav prev link is styled via `.footnav a`; needs no own rule
# Classes defined in CSS but legitimately not present as a static class="..."
# (toggled at runtime by JS). Anything else unused is reported (WARN) so dead
# CSS can't silently accumulate.
CSS_UNUSED_WHITELIST = {"hide", "show"}

issues = []

# Allowed CSS color tokens inside inline SVG fill:/stroke: (dark-mode safe).
# Hardcoded hex/rgb/named colors break dark mode; flag them.
SVG_COLOR_RE = re.compile(r"(?:fill|stroke)\s*:\s*(#[0-9a-fA-F]|rgb|\bwhite\b|\bblack\b)")
# Text width model for the SVG overflow guard. Inline SVG clips to its viewBox
# (UA overflow:hidden), so any <text> past the edge is cut off on screen. The
# shipped font stack includes "Noto Sans SC", which renders Latin ~5-10% wider
# than Arial -- so we model the WIDE case to catch what narrow fonts hide.
SVG_TEXT_DEFAULT_FS = 16.0  # <text> with no font-size uses the 16px SVG default


def _text_width(s, fs, bold):
    """Approximate rendered width (px) of an SVG <text> string. Per-char em
    fractions ~ Arial advance widths; CJK = 1em. A safety factor models the
    shipped 'Noto Sans SC' stack, which renders Latin a few % wider."""
    narrow = set("ijl.,:;'’!|()[] ")
    wide = set("mwMW")
    upper = set("ABCDEFGHJKLNOPQRSTUVXYZ")  # (M/W handled as wide)
    w = 0.0
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3000 <= o <= 0x33FF or 0xFF00 <= o <= 0xFFEF or o in (0x2014, 0x2026):
            w += 1.0          # CJK / fullwidth / em-dash / ellipsis
        elif o == 0xB7:
            w += 0.33         # middot · is narrow even in CJK fonts
        elif o > 0x2100:
            w += 1.1          # emoji / symbols (✅ ❌ → …) render wide
        elif ch in narrow:
            w += 0.28
        elif ch in wide:
            w += 0.87
        elif ch in "ftr":
            w += 0.34
        elif ch in upper or ch.isdigit():
            w += 0.64
        else:
            w += 0.50         # most lowercase
    return w * fs * (1.07 if bold else 1.04)  # bold + Noto safety


def add(sev, f, msg):
    issues.append((sev, f, msg))


def check_balance(name, html, tag):
    o = len(re.findall(rf"<{tag}[\s>]", html))
    c = len(re.findall(rf"</{tag}>", html))
    if o != c:
        add("ERR", name, f"<{tag}> unbalanced: {o} open / {c} close")


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


def check_classes(name, html):
    """Every class used in the HTML must have a `.cls` rule in shell.CSS."""
    seen = set()
    for m in re.findall(r'class="([^"]+)"', html):
        for c in m.split():
            if c in seen or c in CSS_CLASS_WHITELIST or c in CSS_DEFINED:
                continue
            seen.add(c)
            add("ERR", name, f"undefined CSS class {c!r} (no .{c} rule in shell.CSS)")


def check_source_pair(fname, zh, en):
    """Source-level zh/en parity + figure/SVG guards (regression catchers for
    the bugs that actually recurred: text overflow, missing figcaps, zh/en
    drift). Operates on the registry source so the shell chrome can't mask it."""
    exempt = fname in SOFT_EXEMPT
    # 1. Every <div class="fig"> must contain a <div class="figcap">.
    for lang, txt in (("zh", zh), ("en", en)):
        nfig, ncap = txt.count('class="fig"'), txt.count('class="figcap"')
        if nfig != ncap:
            add("ERR", fname, f"{lang}: {nfig} .fig but {ncap} .figcap (each figure needs a caption)")
    # 2. zh/en inventory parity (figures + code references must match).
    for cls in ("fig", "codefile"):
        z, e = zh.count(f'class="{cls}"'), en.count(f'class="{cls}"')
        if z != e:
            add("ERR", fname, f"zh/en .{cls} count mismatch: {z} vs {e}")
    # 3. Per-language section parity (a section present in one language only).
    if not exempt:
        z, e = zh.count("<h2"), en.count("<h2")
        if z != e:
            add("ERR", fname, f"zh/en <h2> count mismatch: {z} vs {e}")
    # 4. Per-SVG structure, theming, and text-overflow.
    for lang, txt in (("zh", zh), ("en", en)):
        for sm in re.finditer(r"<svg\b([^>]*)>(.*?)</svg>", txt, re.S):
            attrs, body = sm.group(1), sm.group(2)
            vb = re.search(r'viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"', attrs)
            if not vb:
                add("ERR", fname, f"{lang}: <svg> missing viewBox")
                continue
            if re.search(r"\b(?:width|height)\s*=", attrs):
                add("ERR", fname, f"{lang}: <svg> has width/height attr (use viewBox only)")
            if 'role="img"' not in attrs:
                add("WARN", fname, f"{lang}: <svg> missing role=\"img\"")
            if "aria-label" not in attrs:
                add("WARN", fname, f"{lang}: <svg> missing aria-label")
            if SVG_COLOR_RE.search(body):
                bad = SVG_COLOR_RE.search(body).group(0)
                add("ERR", fname, f"{lang}: hardcoded SVG color {bad!r} (use var(--...) for dark mode)")
            vbw = float(vb.group(1))
            for tm in re.finditer(r"<text\b([^>]*)>(.*?)</text>", body, re.S):
                ta = tm.group(1)
                s = re.sub(r"<[^>]+>", "", tm.group(2)).strip()
                xm = re.search(r'\bx="(-?\d+(?:\.\d+)?)"', ta)
                if not s or not xm:
                    continue
                x = float(xm.group(1))
                fsm = re.search(r"font-size:\s*(\d+(?:\.\d+)?)px", ta)
                fs = float(fsm.group(1)) if fsm else SVG_TEXT_DEFAULT_FS
                bold = "font-weight:700" in ta or "font-weight:600" in ta or "font-weight:bold" in ta
                w = _text_width(s, fs, bold)
                if 'text-anchor="middle"' in ta:
                    end, start = x + w / 2, x - w / 2
                elif 'text-anchor="end"' in ta:
                    end, start = x, x - w
                else:
                    end, start = x + w, x
                # Margin: only flag a CLEAR clip so the heuristic never false-fails
                # a legitimate edge-touching label.
                if end > vbw + SVG_OVERFLOW_MARGIN:
                    add("ERR", fname, f"{lang}: SVG text clips viewBox (end~{end:.0f}>{vbw:.0f}): {s[:42]!r}")
                elif start < -SVG_OVERFLOW_MARGIN:
                    add("ERR", fname, f"{lang}: SVG text clips left edge (start~{start:.0f}): {s[:42]!r}")


def check_lesson(fname, html):
    for tag in ("div", "details", "table", "pre", "summary",
                "strong", "span", "em", "b", "code"):
        check_balance(fname, html, tag)
    check_classes(fname, html)
    nd = len(re.findall(r"<details", html))
    ns = len(re.findall(r"<summary", html))
    if nd != ns:
        add("ERR", fname, f"details({nd}) != summary({ns})")
    h1 = len(re.findall(r"<h1", html))
    if h1 == 0:
        add("ERR", fname, "missing <h1>")
    elif h1 > 1:
        add("WARN", fname, f"{h1} <h1> (expected 1)")
    if "<title>" not in html:
        add("ERR", fname, "missing <title>")
    if 'name="description"' not in html:
        add("ERR", fname, "missing meta description")
    check_social_meta(fname, html)
    if 'class="lang-zh"' not in html:
        add("ERR", fname, "missing lang-zh content")
    if 'class="lang-en"' not in html:
        add("ERR", fname, "missing lang-en content")
    if fname not in SOFT_EXEMPT:
        if "本课要点" not in html and "Key points" not in html:
            add("WARN", fname, "no key-points card")
        if "card analogy" not in html:
            add("WARN", fname, "no analogy card")
        nvis = sum(html.count(f'class="{c}"') for c in DIAGRAM_CLASSES)
        nvis += html.count('<table class="t"')
        if nvis < MIN_DIAGRAMS:
            add("WARN", fname, f"only {nvis} visual blocks (want >= {MIN_DIAGRAMS}; add diagrams)")
        nfig = html.count('class="fig"')
        if nfig < MIN_FIGURES:
            add("ERR", fname, f"only {nfig} SVG figures (want >= {MIN_FIGURES}; >= 2 per language)")

    for pre in re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.S):
        cleaned = re.sub(r"</?(?:%s)\b[^>]*>" % "|".join(PRE_INLINE), "", pre)
        if re.search(r"<(?!/)", cleaned):
            m = re.search(r"<(?!/).{0,20}", cleaned)
            add("ERR", fname, f"unescaped '<' in <pre>: {m.group(0)!r}")
            break

    # A '<' not starting a real tag/comment is a literal that must be escaped as
    # &lt; — left bare it renders wrong AND breaks the cross-ref linkifier's
    # tokenizer. (The <pre> check above is stricter; this covers prose, li, etc.)
    bare = re.search(r"<(?![A-Za-z/!]).{0,20}", html)
    if bare:
        add("ERR", fname, f"unescaped '<' (use &lt;): {bare.group(0)!r}")

    for m in re.finditer(r"第\s*([0-9、,，~\-－–—\s]+?)\s*课", html):
        nums = [int(x) for x in re.findall(r"[0-9]+", m.group(1))]
        over = [n for n in nums if n == 0 or n > MAX_LESSON]
        if over:
            add("ERR", fname, f"course ref out of range: {m.group(0)!r} -> {over}")

    if fname in ORDER:
        idx = ORDER.index(fname)
        if idx + 1 < TOTAL and f'href="{ORDER[idx + 1]}"' not in html:
            add("ERR", fname, f"next link missing -> {ORDER[idx + 1]}")
        if idx > 0 and f'href="{ORDER[idx - 1]}"' not in html:
            add("ERR", fname, f"prev link missing -> {ORDER[idx - 1]}")


def main():
    for page in PAGES:
        fname = page[0]
        path = os.path.join(ROOT, "lessons", fname)
        if not os.path.exists(path):
            add("ERR", fname, "lesson file missing (run build.py)")
            continue
        with open(path, encoding="utf-8") as fh:
            check_lesson(fname, fh.read())

    # registry <-> PAGES alignment + non-empty bilingual source content.
    # Checking the source (not the rendered HTML) avoids being fooled by the
    # shell chrome, which always emits lang-zh/lang-en spans.
    for page in PAGES:
        fname = page[0]
        c = CONTENT.get(fname)
        if c is None:
            add("ERR", fname, "no registry CONTENT entry")
            continue
        for lang in ("zh", "en"):
            if len(c.get(lang, "").strip()) < MIN_CONTENT:
                add("ERR", fname, f"{lang} content missing or too short")
        if fname not in SOFT_EXEMPT:
            cjk = len(re.findall(r"[\u4e00-\u9fff]", c.get("zh", "")))
            if cjk < MIN_CJK:
                add("WARN", fname, f"only {cjk} CJK chars in zh (want >= {MIN_CJK})")
        check_source_pair(fname, c.get("zh", ""), c.get("en", ""))
    for fname in CONTENT:
        if fname not in ORDER:
            add("ERR", "registry", f"CONTENT key not in PAGES: {fname}")

    index_path = os.path.join(ROOT, shell.INDEX_FILE)
    with open(index_path, encoding="utf-8") as fh:
        idx = fh.read()
    check_classes("index.html", idx)
    check_social_meta("index.html", idx)
    check_og_asset()
    for page in PAGES:
        fname, tz, te = page[0], page[1], page[2]
        if fname not in idx:
            add("ERR", "index.html", f"TOC missing entry {fname}")
        if shell.esc(tz) not in idx:
            add("WARN", "index.html", f"TOC missing zh title {tz!r}")
        if shell.esc(te) not in idx:
            add("WARN", "index.html", f"TOC missing en title {te!r}")
    m = re.search(r"共 (\d+) 课 · (\d+) 个部分", idx)
    if m:
        if int(m.group(1)) != TOTAL:
            add("ERR", "index.html", f"count says {m.group(1)} but PAGES has {TOTAL}")
        nparts = len({p[3] for p in PAGES})
        if int(m.group(2)) != nparts:
            add("ERR", "index.html", f"parts says {m.group(2)} but PAGES has {nparts}")
    else:
        add("WARN", "index.html", "could not find '共 N 课 · N 个部分' pill")

    # Defined-but-unused CSS guard (WARN): catch dead rules across the whole
    # corpus (lessons + index + print), minus runtime JS-toggled classes.
    used_classes = set()
    corpus = [os.path.join(ROOT, "lessons", p[0]) for p in PAGES]
    corpus += [os.path.join(ROOT, shell.INDEX_FILE),
               os.path.join(ROOT, "print_en.html"),
               os.path.join(ROOT, "print_zh.html")]
    for p in corpus:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for attr in re.findall(r'class="([^"]+)"', fh.read()):
                    used_classes.update(attr.split())
    for cls in sorted(CSS_DEFINED - used_classes - CSS_UNUSED_WHITELIST):
        add("WARN", "shell.css", f"defined-but-unused CSS class .{cls} (dead rule?)")

    errs = [i for i in issues if i[0] == "ERR"]
    warns = [i for i in issues if i[0] == "WARN"]
    rank = {"ERR": 0, "WARN": 1, "INFO": 2}
    for sev, f, msg in sorted(issues, key=lambda x: rank[x[0]]):
        print(f"  [{sev}] {f}: {msg}")
    print(f"\nChecked {TOTAL} lessons + index - {len(errs)} error(s), {len(warns)} warning(s).")
    if errs:
        print("structural check FAILED")
        return 1
    print("structural check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
