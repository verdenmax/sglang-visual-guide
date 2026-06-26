"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and build_print.py
both import this so the lesson set stays in sync with shell.PAGES.

Grows one Part module per milestone (part1 .. part13).
"""
import part1
import part2
import part3
import part4
import part5
import part6
import part7
import part8
import part9
import part10
import part11
import part12
import part13

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
    "18-scheduler-event-loop.html": part5.LESSON_18,
    "19-req-and-schedule-batch.html": part5.LESSON_19,
    "20-schedule-policy.html": part5.LESSON_20,
    "21-zero-overhead-overlap-scheduler.html": part5.LESSON_21,
    "22-chunked-prefill.html": part5.LESSON_22,
    "23-dp-controller-and-pp-scheduling.html": part5.LESSON_23,
    "24-model-runner-and-forward-batch.html": part6.LESSON_24,
    "25-model-loading-and-weights.html": part6.LESSON_25,
    "26-writing-a-model.html": part6.LESSON_26,
    "27-cuda-graph-capture-and-replay.html": part6.LESSON_27,
    "28-sampler-and-sampling-params.html": part6.LESSON_28,
    "29-radixattention-implementation.html": part7.LESSON_29,
    "30-paged-memory-pools.html": part7.LESSON_30,
    "31-hicache-tiering.html": part7.LESSON_31,
    "32-eviction-and-hit-rate.html": part7.LESSON_32,
    "33-attention-backend-abstraction.html": part8.LESSON_33,
    "34-moe-layer.html": part8.LESSON_34,
    "35-quantization.html": part8.LESSON_35,
    "36-rope-norm-and-ops.html": part8.LESSON_36,
    "37-logits-and-vocab-parallel.html": part8.LESSON_37,
    "38-sgl-kernel-overview.html": part9.LESSON_38,
    "39-jit-kernel.html": part9.LESSON_39,
    "40-attention-kernel-dissection.html": part9.LESSON_40,
    "41-operator-fusion-and-cuda-graph.html": part9.LESSON_41,
    "42-multi-hardware-backends.html": part9.LESSON_42,
    "43-speculative-decoding-overview.html": part10.LESSON_43,
    "44-eagle-and-next-gen.html": part10.LESSON_44,
    "45-pd-disaggregation.html": part10.LESSON_45,
    "46-tp-pp-ep-dp-parallelism.html": part10.LESSON_46,
    "47-large-scale-ep-and-eplb.html": part10.LESSON_47,
    "48-structured-outputs-and-jump-forward.html": part10.LESSON_48,
    "49-multimodal-vlm-serving.html": part11.LESSON_49,
    "50-multi-lora-batching.html": part11.LESSON_50,
    "51-rl-rollout-and-weight-sync.html": part11.LESSON_51,
    "52-diffusion-models.html": part11.LESSON_52,
    "53-build-and-run.html": part12.LESSON_53,
    "54-benchmark-and-profiling.html": part12.LESSON_54,
    "55-test-suite-and-ci.html": part12.LESSON_55,
    "56-code-conventions-and-pr.html": part12.LESSON_56,
    "57-glossary.html": part12.LESSON_57,
    "58-radixattention-as-a-first-class-idea.html": part13.LESSON_58,
    "59-zero-overhead-scheduling.html": part13.LESSON_59,
    "60-two-workloads-one-engine.html": part13.LESSON_60,
    "61-draft-for-parallel-verify.html": part13.LESSON_61,
    "62-everything-is-pluggable.html": part13.LESSON_62,
    "63-built-for-throughput.html": part13.LESSON_63,
}
