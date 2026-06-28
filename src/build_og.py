#!/usr/bin/env python3
"""Generate the og:image brand share card (og-cover.png, 1200x630).

Run manually when the card design changes:  cd src && python3 build_og.py
Requires `rsvg-convert` (librsvg) on PATH and Noto Sans CJK fonts installed.
This is intentionally NOT imported by build.py, so the site build stays
zero-dependency (the committed og-cover.png is the served asset).
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import shell  # noqa: E402

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
    return SVG_PATH, PNG_PATH


if __name__ == "__main__":
    svg_path, png_path = build()
    print(f"wrote {os.path.relpath(svg_path, ROOT)} and "
          f"{os.path.relpath(png_path, ROOT)} "
          f"({LESSON_COUNT} lessons / {PART_COUNT} parts)")
