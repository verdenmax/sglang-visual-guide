"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and build_print.py
both import this so the lesson set stays in sync with shell.PAGES.

Grows one Part module per milestone (part1 .. part13).
"""
import part1
import part2
import part3
import part4

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
    "09-structured-generation-language.html": part3.LESSON_09,
    "10-interpreter-and-tracer.html": part3.LESSON_10,
    "11-fork-join-and-prefix-sharing.html": part3.LESSON_11,
    "12-backends-and-openai-compat.html": part3.LESSON_12,
    "13-engine-and-http-server.html": part4.LESSON_13,
    "14-tokenizer-manager.html": part4.LESSON_14,
    "15-openai-anthropic-ollama-compat.html": part4.LESSON_15,
    "16-io-structs-and-ipc.html": part4.LESSON_16,
    "17-detokenizer-and-streaming.html": part4.LESSON_17,
}
