"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and build_print.py
both import this so the lesson set stays in sync with shell.PAGES.

Grows one Part module per milestone (part1 .. part13).
"""
import part1
import part2

# Filename -> {"zh": ..., "en": ...}. Keep keys in sync with shell.PAGES.
CONTENT = {
    "01-what-is-sglang.html": part1.LESSON_01,
    "02-project-map.html": part1.LESSON_02,
    "03-life-of-a-request.html": part1.LESSON_03,
    "04-autoregressive-and-kv-cache.html": part2.LESSON_04,
    "05-continuous-batching.html": part2.LESSON_05,
    "06-paged-attention-and-paged-kv.html": part2.LESSON_06,
    "07-radixattention-and-prefix-caching.html": part2.LESSON_07,
    "08-throughput-vs-latency.html": part2.LESSON_08,
}
