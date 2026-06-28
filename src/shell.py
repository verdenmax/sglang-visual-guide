"""Shared HTML shell (CSS design system + navigation) for the SGLang visual guide."""

import base64
import re

# ---- favicon (inline SVG, base64) ----
_FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='#7c48e6'/>"
    "<text x='16' y='22' font-family='system-ui,sans-serif' font-size='14'"
    " font-weight='800' fill='#fff' text-anchor='middle'>Sg</text></svg>"
)
FAVICON = "data:image/svg+xml;base64," + base64.b64encode(_FAVICON_SVG.encode()).decode()


def esc(s):
    """Escape plain text for an HTML text/attribute context.

    For chrome/meta strings that are NOT meant to carry inline markup (page
    titles, descriptions). Do NOT use on lesson body content or bi() inputs,
    which may legitimately contain inline tags.
    """
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def head_meta(title, description, og_type="website"):
    """SEO / social meta tags + favicon for a page <head>."""
    t = esc(title)
    d = esc(description)
    return (
        f'<meta name="description" content="{d}">\n'
        f'<meta name="theme-color" content="#7c48e6">\n'
        f'<link rel="icon" type="image/svg+xml" href="{FAVICON}">\n'
        f'<meta property="og:type" content="{og_type}">\n'
        f'<meta property="og:site_name" content="SGLang 图解教程">\n'
        f'<meta property="og:title" content="{t}">\n'
        f'<meta property="og:description" content="{d}">\n'
        f'<meta name="twitter:card" content="summary">\n'
        f'<meta name="twitter:title" content="{t}">\n'
        f'<meta name="twitter:description" content="{d}">'
    )


# Ordered list of all pages:
# (filename, title_zh, title_en, part_zh, part_en)
# Grows one Part per milestone; M0 ships only L01 to prove the pipeline.
PAGES = [
    ("01-what-is-sglang.html", "SGLang 是什么", "What is SGLang",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
    ("02-project-map.html", "项目全景地图", "The project map",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
    ("03-life-of-a-request.html", "一次请求的一生", "Life of a request",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
    ("04-autoregressive-and-kv-cache.html", "自回归生成与 KV 缓存", "Autoregressive decode & the KV cache",
     "第二部分 · 推理前置基础", "Part 2 · Foundations"),
    ("05-continuous-batching.html", "连续批处理", "Continuous batching",
     "第二部分 · 推理前置基础", "Part 2 · Foundations"),
    ("06-paged-attention-and-paged-kv.html", "PagedAttention 与分页 KV", "PagedAttention & paged KV",
     "第二部分 · 推理前置基础", "Part 2 · Foundations"),
    ("07-radixattention-and-prefix-caching.html", "RadixAttention 与前缀缓存", "RadixAttention & prefix caching",
     "第二部分 · 推理前置基础", "Part 2 · Foundations"),
    ("08-throughput-vs-latency.html", "吞吐 vs 延迟", "Throughput vs latency",
     "第二部分 · 推理前置基础", "Part 2 · Foundations"),
    ("09-structured-generation-language.html", "结构化生成语言", "The structured-generation language",
     "第三部分 · 前端语言", "Part 3 · The frontend language"),
    ("10-interpreter-and-tracer.html", "解释器与 tracer", "Interpreter & tracer",
     "第三部分 · 前端语言", "Part 3 · The frontend language"),
    ("11-fork-join-and-prefix-sharing.html", "fork/join 与前缀共享", "fork/join & prefix sharing",
     "第三部分 · 前端语言", "Part 3 · The frontend language"),
    ("12-backends-and-openai-compat.html", "后端与 OpenAI 兼容", "Backends & OpenAI compat",
     "第三部分 · 前端语言", "Part 3 · The frontend language"),
    ("13-engine-and-http-server.html", "Engine 与 HTTP Server", "Engine & HTTP server",
     "第四部分 · 服务入口与编排", "Part 4 · Entrypoints & orchestration"),
    ("14-tokenizer-manager.html", "TokenizerManager", "TokenizerManager",
     "第四部分 · 服务入口与编排", "Part 4 · Entrypoints & orchestration"),
    ("15-openai-anthropic-ollama-compat.html", "OpenAI/Anthropic/Ollama 兼容层", "OpenAI/Anthropic/Ollama compat",
     "第四部分 · 服务入口与编排", "Part 4 · Entrypoints & orchestration"),
    ("16-io-structs-and-ipc.html", "IO 结构与进程间通信", "IO structs & IPC",
     "第四部分 · 服务入口与编排", "Part 4 · Entrypoints & orchestration"),
    ("17-detokenizer-and-streaming.html", "Detokenizer 与流式输出", "Detokenizer & streaming",
     "第四部分 · 服务入口与编排", "Part 4 · Entrypoints & orchestration"),
    ("18-scheduler-event-loop.html", "调度器事件循环", "The scheduler event loop",
     "第五部分 · 调度器", "Part 5 · The scheduler"),
    ("19-req-and-schedule-batch.html", "Req 与 ScheduleBatch", "Req & ScheduleBatch",
     "第五部分 · 调度器", "Part 5 · The scheduler"),
    ("20-schedule-policy.html", "调度策略", "The schedule policy",
     "第五部分 · 调度器", "Part 5 · The scheduler"),
    ("21-zero-overhead-overlap-scheduler.html", "零开销重叠调度器", "Zero-overhead overlap scheduler",
     "第五部分 · 调度器", "Part 5 · The scheduler"),
    ("22-chunked-prefill.html", "分块预填充", "Chunked prefill",
     "第五部分 · 调度器", "Part 5 · The scheduler"),
    ("23-dp-controller-and-pp-scheduling.html", "DP 控制器与 PP 调度", "DP controller & PP scheduling",
     "第五部分 · 调度器", "Part 5 · The scheduler"),
    ("24-model-runner-and-forward-batch.html", "ModelRunner 与 ForwardBatch", "ModelRunner & ForwardBatch",
     "第六部分 · 模型执行", "Part 6 · Model execution"),
    ("25-model-loading-and-weights.html", "模型加载与权重", "Model loading & weights",
     "第六部分 · 模型执行", "Part 6 · Model execution"),
    ("26-writing-a-model.html", "写一个模型", "Writing a model",
     "第六部分 · 模型执行", "Part 6 · Model execution"),
    ("27-cuda-graph-capture-and-replay.html", "CUDA Graph 捕获与重放", "CUDA graph capture & replay",
     "第六部分 · 模型执行", "Part 6 · Model execution"),
    ("28-sampler-and-sampling-params.html", "采样器与采样参数", "Sampler & sampling params",
     "第六部分 · 模型执行", "Part 6 · Model execution"),
    ("29-radixattention-implementation.html", "RadixAttention 实现", "RadixAttention implementation",
     "第七部分 · KV 缓存与内存", "Part 7 · KV cache & memory"),
    ("30-paged-memory-pools.html", "分页内存池", "Paged memory pools",
     "第七部分 · KV 缓存与内存", "Part 7 · KV cache & memory"),
    ("31-hicache-tiering.html", "HiCache 分层缓存", "HiCache tiering",
     "第七部分 · KV 缓存与内存", "Part 7 · KV cache & memory"),
    ("32-eviction-and-hit-rate.html", "缓存淘汰与命中", "Eviction & hit rate",
     "第七部分 · KV 缓存与内存", "Part 7 · KV cache & memory"),
    ("33-attention-backend-abstraction.html", "Attention 后端抽象", "Attention backend abstraction",
     "第八部分 · Attention 与算子层", "Part 8 · Attention & layers"),
    ("34-moe-layer.html", "MoE 层", "The MoE layer",
     "第八部分 · Attention 与算子层", "Part 8 · Attention & layers"),
    ("35-quantization.html", "量化", "Quantization",
     "第八部分 · Attention 与算子层", "Part 8 · Attention & layers"),
    ("36-rope-norm-and-ops.html", "RoPE、归一化与其它算子", "RoPE, norm & other ops",
     "第八部分 · Attention 与算子层", "Part 8 · Attention & layers"),
    ("37-logits-and-vocab-parallel.html", "Logits 处理与词表并行", "Logits & vocab parallel",
     "第八部分 · Attention 与算子层", "Part 8 · Attention & layers"),
    ("38-sgl-kernel-overview.html", "sgl-kernel 总览", "sgl-kernel overview",
     "第九部分 · 内核与硬件", "Part 9 · Kernels & hardware"),
    ("39-jit-kernel.html", "JIT kernel", "JIT kernels",
     "第九部分 · 内核与硬件", "Part 9 · Kernels & hardware"),
    ("40-attention-kernel-dissection.html", "关键 attention kernel 剖析", "Dissecting an attention kernel",
     "第九部分 · 内核与硬件", "Part 9 · Kernels & hardware"),
    ("41-operator-fusion-and-cuda-graph.html", "算子融合与 CUDA Graph 配合", "Operator fusion & CUDA graph",
     "第九部分 · 内核与硬件", "Part 9 · Kernels & hardware"),
    ("42-multi-hardware-backends.html", "多硬件后端", "Multi-hardware backends",
     "第九部分 · 内核与硬件", "Part 9 · Kernels & hardware"),
    ("43-speculative-decoding-overview.html", "投机解码总览", "Speculative decoding overview",
     "第十部分 · 性能创新专题", "Part 10 · Performance innovations"),
    ("44-eagle-and-next-gen.html", "EAGLE 与下一代", "EAGLE & next-gen",
     "第十部分 · 性能创新专题", "Part 10 · Performance innovations"),
    ("45-pd-disaggregation.html", "PD 分离", "Prefill–decode disaggregation",
     "第十部分 · 性能创新专题", "Part 10 · Performance innovations"),
    ("46-tp-pp-ep-dp-parallelism.html", "四种并行 TP/PP/EP/DP", "TP/PP/EP/DP parallelism",
     "第十部分 · 性能创新专题", "Part 10 · Performance innovations"),
    ("47-large-scale-ep-and-eplb.html", "大规模 EP 与 EPLB", "Large-scale EP & EPLB",
     "第十部分 · 性能创新专题", "Part 10 · Performance innovations"),
    ("48-structured-outputs-and-jump-forward.html", "结构化输出与跳跃前进", "Structured outputs & jump-forward",
     "第十部分 · 性能创新专题", "Part 10 · Performance innovations"),
    ("49-multimodal-vlm-serving.html", "多模态 VLM 服务", "Multimodal VLM serving",
     "第十一部分 · 进阶选读", "Part 11 · Advanced (optional)"),
    ("50-multi-lora-batching.html", "多 LoRA 批处理", "Multi-LoRA batching",
     "第十一部分 · 进阶选读", "Part 11 · Advanced (optional)"),
    ("51-rl-rollout-and-weight-sync.html", "RL Rollout 与权重同步", "RL rollout & weight sync",
     "第十一部分 · 进阶选读", "Part 11 · Advanced (optional)"),
    ("52-diffusion-models.html", "扩散模型", "Diffusion models",
     "第十一部分 · 进阶选读", "Part 11 · Advanced (optional)"),
    ("53-build-and-run.html", "构建与运行", "Build & run",
     "第十二部分 · 实战与贡献", "Part 12 · Practice & contributing"),
    ("54-benchmark-and-profiling.html", "压测与性能分析", "Benchmark & profiling",
     "第十二部分 · 实战与贡献", "Part 12 · Practice & contributing"),
    ("55-test-suite-and-ci.html", "测试套件与 CI", "Test suite & CI",
     "第十二部分 · 实战与贡献", "Part 12 · Practice & contributing"),
    ("56-code-conventions-and-pr.html", "代码规范与提 PR", "Conventions & opening a PR",
     "第十二部分 · 实战与贡献", "Part 12 · Practice & contributing"),
    ("57-glossary.html", "术语速查表", "Glossary",
     "第十二部分 · 实战与贡献", "Part 12 · Practice & contributing"),
    ("58-radixattention-as-a-first-class-idea.html", "RadixAttention 作为一等公民", "RadixAttention as a first-class idea",
     "第十三部分 · 设计专题综合", "Part 13 · Design themes (synthesis)"),
    ("59-zero-overhead-scheduling.html", "零开销调度的哲学", "Zero-overhead scheduling",
     "第十三部分 · 设计专题综合", "Part 13 · Design themes (synthesis)"),
    ("60-two-workloads-one-engine.html", "两种负载，一套引擎", "Two workloads, one engine",
     "第十三部分 · 设计专题综合", "Part 13 · Design themes (synthesis)"),
    ("61-draft-for-parallel-verify.html", "用草稿换并行验证", "Draft for parallel verify",
     "第十三部分 · 设计专题综合", "Part 13 · Design themes (synthesis)"),
    ("62-everything-is-pluggable.html", "一切皆可插拔", "Everything is pluggable",
     "第十三部分 · 设计专题综合", "Part 13 · Design themes (synthesis)"),
    ("63-built-for-throughput.html", "为吞吐而生", "Built for throughput",
     "第十三部分 · 设计专题综合", "Part 13 · Design themes (synthesis)"),
]


# --- cross-reference auto-linking ------------------------------------------
# Map lesson number -> filename, so bare "第 N 课" / "Lesson N" mentions in the
# prose become clickable links to that lesson.
_NUM_HREF = {}
for _p in PAGES:
    try:
        _NUM_HREF[int(_p[0].split("-", 1)[0])] = _p[0]
    except ValueError:
        pass

# Never linkify text inside these elements (SVG labels, aria-labels live in tag
# attributes; code/links must stay literal).
_XREF_SKIP = {"pre", "a", "svg", "code", "script", "style", "h1", "title"}
_ZH_REF = re.compile(r"第\s*(\d+)\s*课")
_EN_REF = re.compile(r"\bLesson\s+(\d+)\b")


def linkify_refs(html, current):
    """Wrap bare '第 N 课' / 'Lesson N' cross-references in <a> links to the
    target lesson. Skips self-references, out-of-range numbers, and any text
    inside tags or <pre>/<svg>/<a>/<code> blocks (so SVG text, aria-labels,
    code, and existing links are never touched)."""

    def repl(m):
        n = int(m.group(1))
        href = _NUM_HREF.get(n)
        if not href or href == current:
            return m.group(0)
        return f'<a class="xref" href="{href}">{m.group(0)}</a>'

    out, depth = [], 0
    for tok in re.split(r"(<[^>]+>)", html):
        if tok[:1] == "<":
            mm = re.match(r"</?([A-Za-z][\w-]*)", tok)
            if mm and mm.group(1).lower() in _XREF_SKIP:
                if tok[:2] == "</":
                    depth = max(0, depth - 1)
                elif not tok.endswith("/>"):
                    depth += 1
            out.append(tok)
        elif depth == 0 and tok:
            out.append(_EN_REF.sub(repl, _ZH_REF.sub(repl, tok)))
        else:
            out.append(tok)
    return "".join(out)


def bi(zh, en):
    """Inline bilingual pair; only the active language is shown (CSS-controlled)."""
    return f'<span class="lang-zh">{zh}</span><span class="lang-en">{en}</span>'


INDEX_FILE = "index.html"

CSS = r"""
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f6f7f9; --panel: #ffffff; --panel-2: #f0f2f5; --ink: #1d2129;
  --muted: #5b6470; --faint: #8a939f; --line: #e1e5ea;
  --accent: #7c48e6; --accent-soft: #efe9fc; --accent-ink: #5b34b0;
  --blue: #2563eb; --blue-soft: #e7efff; --amber: #b4690e; --amber-soft: #fdf1dd;
  --purple: #7c3aed; --purple-soft: #f0e9ff; --red: #d23f3f; --red-soft: #fbe6e6;
  --teal: #0d9488; --teal-soft: #d7f3ef;
  --code-bg: #0f172a; --code-ink: #e2e8f0; --code-line: #1e293b;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.06);
  --radius: 14px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1116; --panel: #161b22; --panel-2: #1c232c; --ink: #e6edf3;
    --muted: #9aa6b2; --faint: #6e7a86; --line: #2a323c;
    --accent: #a98af0; --accent-soft: #241a3e; --accent-ink: #c9b6f7;
    --blue: #6ea8fe; --blue-soft: #16243f; --amber: #e0a44a; --amber-soft: #33270f;
    --purple: #b794f6; --purple-soft: #271a40; --red: #f08080; --red-soft: #3a1a1a;
    --teal: #5ed0c4; --teal-soft: #0d2e2a;
    --code-bg: #0a0f1a; --code-ink: #d8e2f0; --code-line: #14202f;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
}
html { scroll-behavior: smooth; overflow-x: hidden; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
    "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  background: var(--bg); color: var(--ink); line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
.xref { color: inherit; border-bottom: 1px dotted var(--accent); }
.xref:hover { color: var(--accent); border-bottom-style: solid; }
code, .mono { font-family: "SF Mono", "JetBrains Mono", "Fira Code", ui-monospace, Menlo, Consolas, monospace; overflow-wrap: break-word; }

/* ---- top progress bar ---- */
.topbar {
  position: sticky; top: 0; z-index: 50; background: var(--panel);
  border-bottom: 1px solid var(--line); backdrop-filter: blur(8px);
}
.topbar-inner {
  max-width: 960px; margin: 0 auto; padding: .7rem 1.25rem;
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.topbar .home { font-size: .82rem; color: var(--muted); font-weight: 600; display:flex; gap:.5rem; align-items:center; }
.topbar .home b { color: var(--accent); }
.topbar .pill { font-size: .72rem; color: var(--muted); background: var(--panel-2);
  padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; }
.progress { height: 3px; background: var(--panel-2); }
.progress > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--purple)); }

.wrap { max-width: 820px; margin: 0 auto; padding: 2.4rem 1.25rem 5rem; }

/* ---- hero ---- */
.hero { margin-bottom: 2rem; }
.hero .part { font-size: .76rem; letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); font-weight: 700; margin-bottom: .55rem; }
.hero h1 { font-size: 2.05rem; line-height: 1.2; letter-spacing: -.01em; font-weight: 750; }
.hero .lead { margin-top: .9rem; font-size: 1.06rem; color: var(--muted); }

h2 { font-size: 1.32rem; margin: 2.4rem 0 .9rem; letter-spacing: -.01em;
  display: flex; align-items: center; gap: .55rem; }
h2::before { content: ""; width: 4px; height: 1.05em; background: var(--accent); border-radius: 3px; display: inline-block; }
h3 { font-size: 1.05rem; margin: 1.4rem 0 .5rem; }
p { margin: .7rem 0; }
ul, ol { margin: .6rem 0 .6rem 1.3rem; }
li { margin: .3rem 0; }
strong { color: var(--ink); font-weight: 680; }
.inline { background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px;
  padding: .08em .4em; font-size: .9em; color: var(--accent-ink); }

/* ---- callout cards ---- */
.card { border-radius: var(--radius); padding: 1.05rem 1.2rem; margin: 1.2rem 0;
  border: 1px solid var(--line); background: var(--panel); box-shadow: var(--shadow); }
.card .tag { font-size: .72rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
  display: inline-flex; align-items: center; gap: .4rem; margin-bottom: .5rem; }
.card.macro { border-left: 4px solid var(--blue); }
.card.macro .tag { color: var(--blue); }
.card.detail { border-left: 4px solid var(--purple); }
.card.detail .tag { color: var(--purple); }
.card.analogy { border-left: 4px solid var(--amber); background: var(--amber-soft); }
.card.analogy .tag { color: var(--amber); }
.card.key { border-left: 4px solid var(--accent); background: var(--accent-soft); }
.card.key .tag { color: var(--accent-ink); }

.card.spark { border-left: 4px solid #e0a000;
  background: linear-gradient(100deg, rgba(224,160,0,.12), transparent 70%); }
.card.spark .tag { color: #c98a00; }
@media (prefers-color-scheme: dark) { .card.spark .tag { color: #f0c050; } }

/* ---- code file callout ---- */
.codefile { margin: 1.2rem 0; border-radius: 12px; overflow: hidden; border: 1px solid var(--line);
  box-shadow: var(--shadow); }
.codefile .cf-head { display: flex; align-items: center; gap: .55rem; padding: .5rem .85rem;
  background: var(--panel-2); border-bottom: 1px solid var(--line); font-size: .8rem; }
.codefile .cf-head .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); flex-shrink:0; }
.codefile .cf-head .path { font-family: ui-monospace, monospace; color: var(--ink); font-weight: 600; }
.codefile .cf-head .ln { margin-left: auto; color: var(--faint); font-size: .72rem; }
.codefile pre { background: var(--code-bg); color: var(--code-ink); padding: .9rem 1rem;
  overflow-x: auto; font-size: .82rem; line-height: 1.6; }
.codefile pre .cm { color: #7d8aa3; }
.codefile pre .kw { color: #c792ea; }
.codefile pre .fn { color: #82aaff; }
.codefile pre .st { color: #c3e88d; }
.codefile pre .nb { color: #f78c6c; }


/* ---- collapsible accordion (details/summary) ---- */
.accordion { border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
  margin: .7rem 0; box-shadow: var(--shadow); overflow: hidden; }
.accordion > summary { cursor: pointer; padding: .85rem 1.1rem; font-weight: 650; font-size: .96rem;
  list-style: none; display: flex; align-items: center; gap: .6rem; user-select: none; }
.accordion > summary::-webkit-details-marker { display: none; }
.accordion > summary::after { content: "▶"; font-size: .68rem; color: var(--accent);
  margin-left: auto; transition: transform .15s ease; }
.accordion[open] > summary::after { transform: rotate(90deg); }
.accordion > summary:hover { background: var(--panel-2); }
.accordion[open] > summary { border-bottom: 1px solid var(--line); }
.accordion .hint { font-size: .72rem; color: var(--faint); font-weight: 400; }
.acc-body { padding: .9rem 1.1rem 1.1rem; }
.qa { margin: 1rem 0; }
.qa:first-child { margin-top: .3rem; }
.qa .q { font-weight: 680; font-size: .9rem; display: flex; gap: .45rem; align-items: center; margin-bottom: .3rem; }
.qa .a { color: var(--muted); font-size: .9rem; }
.qa .a strong { color: var(--ink); }

/* ---- flow diagram ---- */
.flow { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: 1.3rem 0;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1.2rem 1rem; box-shadow: var(--shadow); }
.flow .node { flex: 1 1 0; min-width: 110px; text-align: center; padding: .7rem .5rem;
  border-radius: 10px; background: var(--panel-2); border: 1px solid var(--line); }
.flow .node .nt { font-weight: 700; font-size: .92rem; }
.flow .node .nd { font-size: .76rem; color: var(--muted); margin-top: .2rem; }
.flow .node.hl { background: var(--accent-soft); border-color: var(--accent); }
.flow .arrow { align-self: center; color: var(--faint); font-size: 1.3rem; padding: 0 .35rem; }

/* vertical flow */
.vflow { margin: 1.3rem 0; }
.vflow .step { display: flex; gap: .9rem; position: relative; padding-bottom: 1.1rem; }
.vflow .step:not(:last-child)::before { content:""; position:absolute; left: 15px; top: 34px; bottom: -2px;
  width: 2px; background: var(--line); }
.vflow .num { width: 32px; height: 32px; border-radius: 50%; background: var(--accent); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; flex-shrink: 0; z-index:1; }
.vflow .sc h4 { margin: .25rem 0 .2rem; font-size: 1rem; }
.vflow .sc p { margin: .15rem 0; font-size: .92rem; color: var(--muted); }
.vflow .sc .mono { font-size: .8rem; color: var(--accent-ink); }

/* layered architecture */
.layers { margin: 1.3rem 0; display: flex; flex-direction: column; gap: .55rem; }
.layer { border-radius: 12px; padding: .85rem 1.1rem; border: 1px solid var(--line); background: var(--panel);
  box-shadow: var(--shadow); }
.layer .lh { display: flex; align-items: center; gap: .6rem; }
.layer .lh .badge { font-size: .7rem; font-weight: 700; padding: .12rem .5rem; border-radius: 999px; }
.layer .lh .name { font-weight: 700; font-family: ui-monospace, monospace; }
.layer .ld { font-size: .85rem; color: var(--muted); margin-top: .35rem; }
.layer.l-core { border-left: 4px solid var(--accent); } .layer.l-core .badge { background: var(--accent-soft); color: var(--accent-ink); }
.layer.l-main { border-left: 4px solid var(--blue); } .layer.l-main .badge { background: var(--blue-soft); color: var(--blue); }
.layer.l-part { border-left: 4px solid var(--purple); } .layer.l-part .badge { background: var(--purple-soft); color: var(--purple); }
.layer.l-app { border-left: 4px solid var(--amber); } .layer.l-app .badge { background: var(--amber-soft); color: var(--amber); }

/* two-column compare */
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.2rem 0; }
@media (max-width: 640px) { .cols { grid-template-columns: 1fr; } }
.col { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; box-shadow: var(--shadow); min-width: 0; }
.col h4 { margin: 0 0 .4rem; font-size: .95rem; }

table.t { width: 100%; border-collapse: collapse; margin: 1.1rem 0; font-size: .9rem;
  background: var(--panel); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); }
table.t th, table.t td { padding: .6rem .8rem; text-align: left; border-bottom: 1px solid var(--line); }
table.t th { background: var(--panel-2); font-size: .8rem; letter-spacing: .02em; }
table.t tr:last-child td { border-bottom: none; }
table.t td.mono, table.t td .mono { font-family: ui-monospace, monospace; font-size: .82rem; color: var(--accent-ink); }
@media (max-width: 640px) {
  /* Wide multi-column tables: scroll within their own box instead of
     forcing page-level horizontal overflow (which clipped right columns). */
  table.t { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table.t th, table.t td { padding: .5rem .6rem; }
}
.selftest { margin: 2.2rem 0 0; border-top: 2px dashed var(--line); padding-top: 1.2rem; }
.selftest > h2 { margin-top: .2rem; }
.quiz { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--blue);
  border-radius: 12px; padding: .9rem 1.1rem; margin: 1rem 0; box-shadow: var(--shadow); }
.quiz .qn { font-weight: 650; }
.quiz ol.opts { list-style: upper-alpha; margin: .55rem 0 .6rem 1.5rem; padding: 0; }
.quiz ol.opts li { margin: .3rem 0; padding-left: .15rem; }
.quiz details.accordion { margin: .5rem 0 0; }
.selftest code { font-family: ui-monospace, monospace; font-size: .9em; color: var(--accent-ink);
  background: var(--accent-soft); padding: 0 .28em; border-radius: 4px; }

/* footer nav */
.footnav { display: flex; justify-content: space-between; gap: 1rem; margin-top: 3rem;
  padding-top: 1.4rem; border-top: 1px solid var(--line); }
.footnav a { flex: 1; padding: .85rem 1.1rem; border-radius: 12px; border: 1px solid var(--line);
  background: var(--panel); box-shadow: var(--shadow); transition: .15s; }
.footnav a:hover { border-color: var(--accent); transform: translateY(-1px); }
.footnav a.next { text-align: right; }
.footnav .dir { font-size: .72rem; color: var(--faint); text-transform: uppercase; letter-spacing: .05em; }
.footnav .ttl { font-weight: 700; color: var(--ink); margin-top: .15rem; }

/* index page */
.toc { display: grid; gap: .7rem; margin-top: 1.6rem; }
.toc-part { font-size: .78rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  color: var(--accent); margin: 1.4rem 0 .2rem; }
.toc a { display: flex; align-items: center; gap: .9rem; padding: .85rem 1.05rem; border-radius: 12px;
  background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); transition: .15s; }
.toc a:hover { border-color: var(--accent); transform: translateX(3px); }
.toc .n { width: 30px; height: 30px; border-radius: 8px; background: var(--accent-soft); color: var(--accent-ink);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; flex-shrink: 0; }
.toc .tt { font-weight: 650; color: var(--ink); }
.toc .ts { font-size: .8rem; color: var(--muted); margin-left: auto; text-align: right; }
.toc-search { position: relative; margin: 1.6rem 0 -.4rem; }
.toc-search input { width: 100%; box-sizing: border-box; padding: .75rem 2.8rem .75rem 1rem;
  border-radius: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  font-size: .98rem; box-shadow: var(--shadow); }
.toc-search input:focus { outline: none; border-color: var(--accent); }
.toc-search .qcount { position: absolute; right: 1rem; top: 50%; transform: translateY(-50%);
  color: var(--faint); font-size: .8rem; pointer-events: none; }
.toc a.hide, .toc .toc-part.hide { display: none; }
.toc-empty { display: none; color: var(--muted); padding: 1rem; text-align: center; }
.toc-empty.show { display: block; }
.hero.index h1 { font-size: 2.3rem; }
.legend { display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:1rem; font-size:.8rem; color:var(--muted); }
.legend span { display:flex; align-items:center; gap:.4rem; }
.legend i { width:12px; height:12px; border-radius:3px; display:inline-block; }

/* ---- bilingual language switch ----
   Contract: <html> must carry data-lang="zh" (default) or "en".
   page()/index_page() hard-code data-lang="zh"; LANG_BOOT may switch to "en". */
html[data-lang="en"] .lang-zh { display: none !important; }
html[data-lang="zh"] .lang-en { display: none !important; }
.langtoggle { font-size:.72rem; font-weight:700; color:var(--accent-ink);
  background:var(--accent-soft); border:1px solid var(--accent); border-radius:999px;
  padding:.22rem .7rem; cursor:pointer; line-height:1.4; white-space:nowrap; }
.langtoggle:hover { background:var(--accent); color:#fff; }

/* ---- schematic: cell strips (vector rows / quant blocks / KV columns) ---- */
.cellgroup { margin: 1.2rem 0; background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.cellgroup .cg-cap { font-size: .82rem; color: var(--muted); margin-bottom: .55rem; }
.cellgroup .cg-cap b { color: var(--ink); }
.cells { display: flex; flex-wrap: wrap; gap: .35rem; align-items: center; }
.cells + .cells { margin-top: .5rem; }
.cell { min-width: 2.1rem; padding: .38rem .5rem; text-align: center; border-radius: 8px;
  background: var(--panel-2); border: 1px solid var(--line); font-size: .78rem;
  font-family: ui-monospace, monospace; white-space: nowrap; }
.cell.hl    { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); font-weight: 700; }
.cell.q     { background: var(--blue-soft); border-color: var(--blue); color: var(--blue); }
.cell.dim   { opacity: .45; }
.cells .lab { font-size: .76rem; color: var(--faint); padding: 0 .35rem; }
.cells .sep { color: var(--faint); padding: 0 .1rem; }

/* ---- schematic: timeline lanes (write vs read, step-by-step) ---- */
.timeline { margin: 1.2rem 0; display: flex; flex-direction: column; gap: .5rem;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.timeline .lane { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.timeline .lane-label { min-width: 6rem; font-size: .8rem; font-weight: 700; color: var(--muted); }
.timeline .tslot { padding: .4rem .6rem; border-radius: 8px; background: var(--panel-2);
  border: 1px solid var(--line); font-size: .78rem; text-align: center; font-family: ui-monospace, monospace; }
.timeline .tslot.span { flex: 1; min-width: 8rem; background: var(--blue-soft); border-color: var(--blue);
  color: var(--blue); font-weight: 700; }
.timeline .tslot.now { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); font-weight: 700; }

/* ---- worked-example trace: one concrete input, stepped through ---- */
.trace { margin: 1.3rem 0; background: var(--panel); border: 1px solid var(--line);
  border-left: 4px solid var(--accent); border-radius: var(--radius); padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.trace .tcap { font-size: .82rem; color: var(--muted); margin-bottom: .7rem; }
.trace .tcap b { color: var(--accent-ink); }
.trace .stations { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; }
.trace .stn { flex: 1 1 0; min-width: 116px; border: 1px solid var(--line); border-radius: 10px;
  padding: .55rem; background: var(--bg); }
.trace .stn h5 { margin: 0 0 .45rem; font-size: .8rem; color: var(--ink); }
.trace .cellrow { display: flex; gap: .3rem; align-items: center; flex-wrap: wrap; }
.trace .vc { min-width: 2.1rem; padding: .32rem .45rem; text-align: center; border-radius: 7px;
  background: var(--panel-2); border: 1px solid var(--line); font: 600 .76rem ui-monospace, monospace; white-space: nowrap; }
.trace .vc.hot  { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); }
.trace .vc.blue { background: var(--blue-soft); border-color: var(--blue); color: var(--blue); }
.trace .vc.dim  { opacity: .42; }
.trace .tlab { font-size: .68rem; color: var(--faint); margin-top: .35rem; }
.trace .op { align-self: center; color: var(--accent); font: 700 .72rem ui-monospace, monospace;
  padding: 0 .5rem; text-align: center; white-space: nowrap; }
.trace svg { max-width: 100%; height: auto; display: block; margin: .3rem auto; }
@media (max-width: 640px) { .trace .stations { flex-direction: column; } .trace .op { padding: .3rem 0; } }
/* --- hand-drawn figure (inline SVG illustrations) --- */
.fig { margin: 1.3rem 0; background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1rem 1rem .85rem; box-shadow: var(--shadow); text-align: center; }
.fig svg { max-width: 100%; height: auto; display: block; margin: 0 auto;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif; }
.fig svg text { fill: var(--ink); }
.fig .figcap { margin: .72rem auto 0; font-size: .8rem; color: var(--muted); line-height: 1.55; max-width: 46rem; }
.fig .figcap b { color: var(--accent-ink); font-weight: 700; }
"""

SEARCH_JS = """
(function(){
  var q=document.getElementById('q'); if(!q) return;
  var toc=document.querySelector('.toc');
  var empty=document.getElementById('tocempty');
  var count=document.getElementById('qcount');
  var links=[].slice.call(toc.querySelectorAll('a'));
  var heads=[].slice.call(toc.querySelectorAll('.toc-part'));
  links.forEach(function(a){ a.setAttribute('data-s',(a.textContent||'').toLowerCase()); });
  function run(){
    var t=(q.value||'').toLowerCase().trim(), n=0;
    links.forEach(function(a){
      var hit=!t||a.getAttribute('data-s').indexOf(t)>=0;
      a.classList.toggle('hide',!hit); if(hit)n++;
    });
    heads.forEach(function(h){
      var el=h.nextElementSibling, any=false;
      while(el && !el.classList.contains('toc-part')){
        if(el.tagName==='A' && !el.classList.contains('hide')){any=true;break;}
        el=el.nextElementSibling;
      }
      h.classList.toggle('hide',!any);
    });
    empty.classList.toggle('show', !!t && n===0);
    count.textContent = t ? String(n) : '';
  }
  q.addEventListener('input',run);
})();
"""

LANG_JS = """
function sgvgSetLang(l){
  l=(l==='en')?'en':'zh';
  var d=document.documentElement;
  d.dataset.lang=l; d.lang=(l==='en'?'en':'zh-CN');
  try{localStorage.setItem('sgvg-lang',l);}catch(e){}
}
function sgvgToggleLang(){
  sgvgSetLang(document.documentElement.dataset.lang==='en'?'zh':'en');
}
"""

# Runs in <head> before first paint to avoid a flash of the wrong language.
LANG_BOOT = (
    "<script>try{var l=localStorage.getItem('sgvg-lang');"
    "if(l==='en'){document.documentElement.dataset.lang='en';"
    "document.documentElement.lang='en';}}catch(e){}</script>"
)


def page(filename, content, home_href="../index.html"):
    """Wrap one lesson's bilingual content in the full HTML shell.

    ``content`` is a dict ``{"zh": html, "en": html}``. Both are emitted; CSS
    shows only the active language. Navigation uses plain relative ``href``
    links so the site works via file:// and any static server (lessons share
    one directory; home defaults to ``../index.html``).
    """
    idx = next(i for i, p in enumerate(PAGES) if p[0] == filename)
    fname, title_zh, title_en, part_zh, part_en = PAGES[idx]
    total = len(PAGES)
    pct = int((idx + 1) / total * 100)
    home = home_href

    if idx > 0:
        p = PAGES[idx - 1]
        prev_link = (
            f'<a class="prev" href="{p[0]}"><div class="dir">{bi("← 上一课", "← Prev")}</div>'
            f'<div class="ttl">{bi(esc(p[1]), esc(p[2]))}</div></a>'
        )
    else:
        prev_link = (
            f'<a class="prev" href="{home}"><div class="dir">{bi("← 返回", "← Back")}</div>'
            f'<div class="ttl">{bi("目录", "Contents")}</div></a>'
        )
    if idx + 1 < total:
        p = PAGES[idx + 1]
        next_link = (
            f'<a class="next" href="{p[0]}"><div class="dir">{bi("下一课 →", "Next →")}</div>'
            f'<div class="ttl">{bi(esc(p[1]), esc(p[2]))}</div></a>'
        )
    else:
        next_link = (
            f'<a class="next" href="{home}"><div class="dir">{bi("完成 →", "Done →")}</div>'
            f'<div class="ttl">{bi("返回目录", "Back to index")}</div></a>'
        )

    title_tag = f"{idx+1:02d} · {title_zh} / {title_en} - SGLang 图解教程"
    desc = f"{part_zh}｜{title_zh} - SGLang 图解教程（中英双语，配真实源码对应、折叠深挖与设计亮点）"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN" data-lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{LANG_BOOT}
<title>{esc(title_tag)}</title>
{head_meta(title_tag, desc, og_type="article")}
<style>{CSS}</style>
</head><body>
<div class="topbar">
  <div class="topbar-inner">
    <a class="home" href="{home}">⚡ <b class="lang-zh">SGLang 图解教程</b><b class="lang-en">SGLang Visual Guide</b></a>
    <span class="pill">{bi(esc(part_zh), esc(part_en))}</span>
    <span class="pill">{idx+1:02d} / {total:02d}</span>
    <button class="langtoggle" onclick="sgvgToggleLang()" aria-label="switch language"><span class="lang-zh">EN</span><span class="lang-en">中</span></button>
  </div>
  <div class="progress"><span style="width:{pct}%"></span></div>
</div>
<div class="wrap">
  <div class="hero">
    <div class="part">{bi(esc(part_zh), esc(part_en))}</div>
    <h1><span class="lang-zh">{esc(title_zh)}</span><span class="lang-en">{esc(title_en)}</span></h1>
  </div>
  <div class="lang-zh">{content["zh"]}</div>
  <div class="lang-en">{content["en"]}</div>
  <div class="footnav">{prev_link}{next_link}</div>
</div>
<script>{LANG_JS}</script>
</body></html>"""
    return html


# Per-lesson TOC subtitle: filename -> (zh, en). Missing entries render blank.
# Grows one Part per milestone alongside PAGES.
SUBTITLES = {
    "01-what-is-sglang.html": ("高性能 LLM/多模态服务框架 · 前端语言 + 运行时引擎 · 为何快",
                               "high-perf LLM/multimodal serving · frontend DSL + runtime engine · why it's fast"),
    "02-project-map.html": ("两半：前端 DSL + 运行时引擎 · srt 分层 · 三进程 · ZMQ",
                            "two halves: frontend DSL + runtime · srt layers · three processes · ZMQ"),
    "03-life-of-a-request.html": ("端到端追踪一条 generate · 预填充 + 多次解码 · 进程边界",
                                  "end-to-end trace of one generate · prefill + many decodes · process boundaries"),
    "04-autoregressive-and-kv-cache.html": ("逐 token 自回归 · 缓存 K/V 让每步 O(t) · 预填充填、解码追加 · 解码受带宽限",
                                            "token-by-token autoregression · cache K/V → O(t) per step · prefill fills, decode appends · decode is bandwidth-bound"),
    "05-continuous-batching.html": ("静态/填充批处理的浪费 · 迭代级连续批处理 · 调度器每步重组批次",
                                    "waste of static/padded batching · iteration-level continuous batching · scheduler re-forms the batch each step"),
    "06-paged-attention-and-paged-kv.html": ("连续预留导致碎片 · 定长分页 + 索引表 · 按需分配、为前缀复用铺路",
                                             "contiguous reservation fragments HBM · fixed-size pages + index table · allocate on demand, sets up prefix reuse"),
    "07-radixattention-and-prefix-caching.html": ("把所有 KV 存进基数树 · 共享前缀只算一次、跨请求跨轮复用 · 匹配/插入/分裂 + LRU 驱逐",
                                                  "all KV in a radix tree · shared prefixes computed once, reused across requests & turns · match/insert/split + LRU eviction"),
    "08-throughput-vs-latency.html": ("TTFT/ITL/吞吐/有效吞吐 · 批处理拿延迟换吞吐 · 旋钮与调度张力",
                                      "TTFT/ITL/throughput/goodput · batching trades latency for throughput · the knobs & the scheduling tension"),
    "09-structured-generation-language.html": ("用 gen/select/fork/join 写多调用 LLM 程序 · SGLang 的命名由来",
                                               "write multi-call LLM programs with gen/select/fork/join · SGLang's namesake"),
    "10-interpreter-and-tracer.html": ("解释执行 vs 追踪编译 · StreamExecutor 与 ProgramState",
                                       "interpret vs trace/compile · StreamExecutor & ProgramState"),
    "11-fork-join-and-prefix-sharing.html": ("fork 并行分支共享前缀 · 与 RadixAttention 是一体两面",
                                             "fork's parallel branches share the prefix · two halves of RadixAttention"),
    "12-backends-and-openai-compat.html": ("后端无关 · 本地 runtime vs OpenAI/Anthropic · 通往运行时的接缝",
                                           "backend-agnostic · local runtime vs OpenAI/Anthropic · the seam to the runtime"),
    "13-engine-and-http-server.html": ("离线 Engine vs 在线服务器 · 同一引擎两种入口",
                                       "offline Engine vs online server · one engine, two entry points"),
    "14-tokenizer-manager.html": ("前门：分词、组参数、ZMQ 入队、等回流",
                                  "the front door: tokenize, build params, ZMQ enqueue, await outputs"),
    "15-openai-anthropic-ollama-compat.html": ("多协议适配 · 把 OpenAI/Anthropic/Ollama 翻成原生请求",
                                               "multi-protocol adapters · translate OpenAI/Anthropic/Ollama to native"),
    "16-io-structs-and-ipc.html": ("进程间消息的类型系统 · GenerateReqInput→Tokenized→批输出",
                                   "the IPC type system · GenerateReqInput→Tokenized→batch outputs"),
    "17-detokenizer-and-streaming.html": ("增量反分词 · sent_offset 只发新增片段 · SSE",
                                          "incremental detok · sent_offset emits only the new slice · SSE"),
    "18-scheduler-event-loop.html": ("引擎心跳 · recv→schedule→forward→output 的主循环",
                                     "the engine's heartbeat · recv→schedule→forward→output loop"),
    "19-req-and-schedule-batch.html": ("请求状态机 Req · 一步的批 ScheduleBatch（extend/decode）",
                                       "the Req state machine · ScheduleBatch for one step (extend/decode)"),
    "20-schedule-policy.html": ("谁先跑 · 缓存感知 LPM / FCFS / 优先级 · PrefillAdder 预算",
                                "who runs next · cache-aware LPM / FCFS / priority · PrefillAdder budget"),
    "21-zero-overhead-overlap-scheduler.html": ("CPU 调度与 GPU 计算重叠 · GPU 几乎不空转（招牌）",
                                                "overlap CPU scheduling with GPU compute · GPU rarely idles (signature)"),
    "22-chunked-prefill.html": ("长 prompt 分块 · 与 decode 混批 · 抹平延迟尖峰",
                                "split long prefills into chunks · mix with decode · smooth latency spikes"),
    "23-dp-controller-and-pp-scheduling.html": ("DP 控制器分发到 N 个调度器副本 · PP 流水线跨阶段",
                                                "DP controller fans out to N scheduler replicas · PP pipelines across stages"),
    "24-model-runner-and-forward-batch.html": ("GPU 执行器 · ForwardBatch 是 ScheduleBatch 的 GPU 视图 · forward→logits→采样",
                                               "the GPU executor · ForwardBatch is the GPU view of ScheduleBatch · forward→logits→sample"),
    "25-model-loading-and-weights.html": ("读 HF 分片 · 名称映射 · 按 TP 切分 · dtype/量化 上卡",
                                          "read HF shards · map names · shard for TP · dtype/quant onto GPU"),
    "26-writing-a-model.html": ("用并行层拼模型 · Llama 为例 · 加模型很便宜",
                                "compose a model from parallel layers · Llama as the example · adding a model is cheap"),
    "27-cuda-graph-capture-and-replay.html": ("捕获整段 forward 的 kernel 序列一次性重放 · 抹掉逐核启动开销",
                                              "capture the forward's kernel sequence and replay it · kill per-kernel launch overhead"),
    "28-sampler-and-sampling-params.html": ("logits→下一个 token · 温度/top-k/p/惩罚 · SamplingParams",
                                            "logits→next token · temperature/top-k/p/penalties · SamplingParams"),
    "29-radixattention-implementation.html": ("radix 树实现 · TreeNode/match/split/lock · L07 概念落成代码",
                                              "the radix tree in code · TreeNode/match/split/lock · L07's concept as data structures"),
    "30-paged-memory-pools.html": ("KV 物理存储 · req_to_token + token_to_kv 两池 · 分配器",
                                   "where KV physically lives · req_to_token + token_to_kv pools · the allocator"),
    "31-hicache-tiering.html": ("缓存跨 GPU/CPU/磁盘三级 · 后台预取与回写",
                                "cache across GPU/CPU/disk tiers · background prefetch & writeback"),
    "32-eviction-and-hit-rate.html": ("LRU 淘汰可淘汰叶子 · 锁定的在用节点不淘汰 · 命中率即吞吐",
                                      "LRU evicts evictable leaves · locked in-use nodes survive · hit rate is throughput"),
    "33-attention-backend-abstraction.html": ("attention 是可换的策略 · FlashInfer/Triton/FA · 按硬件选",
                                              "attention as a swappable strategy · FlashInfer/Triton/FA · picked by hardware"),
    "34-moe-layer.html": ("路由 top-k 专家 · 稀疏计算 · FusedMoE 融合 routing+GEMM",
                          "route to top-k experts · sparse compute · FusedMoE fuses routing+GEMM"),
    "35-quantization.html": ("更少比特 · FP8/FP4/INT4/AWQ/GPTQ · 省显存省带宽",
                             "fewer bits · FP8/FP4/INT4/AWQ/GPTQ · save HBM & bandwidth"),
    "36-rope-norm-and-ops.html": ("RoPE 按位置旋转 q/k · RMSNorm 轻量归一 · 算子融合",
                                  "RoPE rotates q/k by position · RMSNorm lightweight norm · fused ops"),
    "37-logits-and-vocab-parallel.html": ("词表维度按 TP 切分 · 各 rank 算一片再 gather · 接采样器",
                                          "vocab split across TP ranks · each scores a shard then gather · into the Sampler"),
    "38-sgl-kernel-overview.html": ("AOT C++/CUDA 内核包 · csrc 编译成 .so · 暴露为 torch.ops.sgl_kernel.*",
                                    "AOT C++/CUDA kernel package · csrc compiled into a .so · exposed as torch.ops.sgl_kernel.*"),
    "39-jit-kernel.html": ("运行时按需编译小内核 · 编译即缓存 · 灵活但有首次编译开销",
                           "compile small kernels on demand at runtime · compile-and-cache · flexible but first-call cost"),
    "40-attention-kernel-dissection.html": ("decode 内核读分页 KV · Q·Kᵀ→softmax→·V · tiling 与在线 softmax",
                                            "decode kernel reads paged KV · Q·Kᵀ→softmax→·V · tiling & online softmax"),
    "41-operator-fusion-and-cuda-graph.html": ("融合算子省 HBM 往返 · 静态形状利于图捕获 · 二者协同近零开销",
                                               "fused ops save HBM round-trips · static shapes suit graph capture · together near-zero overhead"),
    "42-multi-hardware-backends.html": ("一套引擎多种芯片 · 平台抽象 + 各芯片内核 · 上层硬件无关",
                                        "one engine many chips · platform abstraction + per-chip kernels · upper layers hardware-agnostic"),
    "43-speculative-decoding-overview.html": ("草稿模型猜 k 个 · 目标模型一次验完 · 接受率即加速",
                                              "a draft proposes k · the target verifies in one pass · accept rate is the speedup"),
    "44-eagle-and-next-gen.html": ("复用隐藏态草拟 · 树状候选 + 树注意力 · 比链更高接受长度",
                                   "draft from reused hidden states · a token tree + tree attention · higher accept length than a chain"),
    "45-pd-disaggregation.html": ("prefill 与 decode 分机 · KV 跨机传输 · 各自吃满不互相拖累",
                                  "split prefill and decode across pools · transfer the KV across · each saturates without interfering"),
    "46-tp-pp-ep-dp-parallelism.html": ("张量/流水/专家/数据四种切法 · 统一 GroupCoordinator · 可组合",
                                        "tensor/pipeline/expert/data — four splits · one GroupCoordinator · composable"),
    "47-large-scale-ep-and-eplb.html": ("专家负载会偏斜 · EPLB 周期性重平衡放置 · 大 MoE 必备",
                                        "expert load gets skewed · EPLB periodically rebalances placement · essential for big MoE"),
    "48-structured-outputs-and-jump-forward.html": ("语法 FSM 屏蔽 logits 保证合法 · 确定段跳跃前进 · 免调模型",
                                                    "a grammar FSM masks logits to stay valid · jump-forward over fixed spans · skip the model"),
    "49-multimodal-vlm-serving.html": ("处理器把媒体变占位符 · 编码器出嵌入再拼接进序列 · 之后引擎照常跑",
                                       "the processor turns media into placeholders · the encoder's embeddings are spliced in · then the engine runs as usual"),
    "50-multi-lora-batching.html": ("一个底座 + 一池小适配器 · 同批不同请求用不同 LoRA · 分组 GEMM 一次算",
                                    "one base + a pool of small adapters · different requests in one batch use different LoRAs · grouped GEMM in one pass"),
    "51-rl-rollout-and-weight-sync.html": ("SGLang 当 RL 的 rollout 引擎 · 原地热更权重免重启 · 打包张量加速同步",
                                           "SGLang as the RL rollout engine · in-place weight updates without restart · bucket tensors to speed the sync"),
    "52-diffusion-models.html": ("扩散是迭代去噪不是自回归 · 复用 sgl-kernel/调度/CUDA Graph · 一套栈两种范式",
                                 "diffusion is iterative denoising, not autoregression · reuses sgl-kernel/scheduler/CUDA graph · one stack, two paradigms"),
    "53-build-and-run.html": ("一行命令起服务 · ServerArgs 把全书旋钮收成 CLI · Engine 离线 / HTTP 在线",
                              "one command starts a server · ServerArgs gathers the whole guide's knobs into CLI flags · Engine offline / HTTP online"),
    "54-benchmark-and-profiling.html": ("bench_serving 给吞吐与 TTFT/TPOT 分位数 · profiler 看内核时间线 · 黑盒数 + 白盒线",
                                        "bench_serving reports throughput + TTFT/TPOT percentiles · the profiler shows the kernel timeline · black-box numbers + white-box trace"),
    "55-test-suite-and-ci.html": ("单测免起服务 / e2e 起服务 · CustomTestCase 保证清理 · pre-commit 先于 CI",
                                  "unit tests need no server / e2e launches one · CustomTestCase guarantees cleanup · pre-commit gates before CI"),
    "56-code-conventions-and-pr.html": ("fork→分支→pre-commit→测试→PR · 改 srt 就补测试 · 文档进 docs_new",
                                        "fork→branch→pre-commit→test→PR · touch srt then add a test · docs go under docs_new"),
    "57-glossary.html": ("全书术语速查 · 一句话定义 + 课次回查 · 按主题归拢成表",
                         "whole-guide term lookup · one-line definitions + lesson back-refs · grouped into tables by theme"),
    "58-radixattention-as-a-first-class-idea.html": ("前缀共享不是补丁而是一等数据结构 · 从 DSL 到调度处处可见 · KV 缓存是共享前缀的树",
                                                     "prefix sharing is a first-class data structure, not a patch · it shows up from DSL to scheduling · the KV cache is a tree of shared prefixes"),
    "59-zero-overhead-scheduling.html": ("CPU 永不让 GPU 等 · 调度/分词/解码与前向重叠 · 把每个气泡藏进 GPU 工作里",
                                         "the CPU never makes the GPU wait · scheduling/tokenize/detokenize overlap the forward · hide every bubble behind GPU work"),
    "60-two-workloads-one-engine.html": ("prefill 算力受限 vs decode 带宽受限 · 一套引擎服务两种相反负载 · 同卡分时或分机分离",
                                         "prefill is compute-bound vs decode bandwidth-bound · one engine serves two opposite workloads · time-share one GPU or disaggregate"),
    "61-draft-for-parallel-verify.html": ("投机的本质：拿便宜的并行换串行瓶颈 · 草稿把串行 decode 变成一次并行验证 · 把延迟瓶颈变吞吐瓶颈",
                                          "the essence of speculation: trade cheap parallel work for a serial bottleneck · a draft turns serial decode into one parallel verify · convert a latency wall into a throughput one"),
    "62-everything-is-pluggable.html": ("稳定内核 + 可插拔接口 · 注意力/硬件/量化/并行/语法都按接口编程 · 部署时选实现",
                                        "a stable core + pluggable interfaces · attention/hardware/quant/parallel/grammar all program to an interface · pick the impl at deploy time"),
    "63-built-for-throughput.html": ("几乎每个设计都服务一个北极星：让 GPU 一直做有用功 · 批/页/缓存/重叠/图/内核齐上 · 全书收束",
                                     "almost every design serves one north star: keep the GPU doing useful work · batching/paging/caching/overlap/graphs/kernels together · the guide's finale"),
}


def index_page(lesson_prefix="lessons/"):
    """Build the bilingual index (table of contents). Always relative links."""
    order = []   # ordered list of (part_zh, part_en)
    groups = {}  # part_zh -> [(num, fname, title_zh, title_en), ...]
    for i, (fname, tz, te, pz, pe) in enumerate(PAGES):
        if pz not in groups:
            groups[pz] = []
            order.append((pz, pe))
        groups[pz].append((i + 1, fname, tz, te))

    blocks = []
    for pz, pe in order:
        blocks.append(f'<div class="toc-part">{bi(esc(pz), esc(pe))}</div>')
        for num, fname, tz, te in groups[pz]:
            sz, se = SUBTITLES.get(fname, ("", ""))
            blocks.append(
                f'<a href="{lesson_prefix}{fname}"><span class="n">{num:02d}</span>'
                f'<span class="tt"><span class="lang-zh">{esc(tz)}</span>'
                f'<span class="lang-en">{esc(te)}</span></span>'
                f'<span class="ts"><span class="lang-zh">{esc(sz)}</span>'
                f'<span class="lang-en">{esc(se)}</span></span></a>'
            )
    toc = "\n".join(blocks)
    total = len(PAGES)
    nparts = len(order)

    title_tag = "SGLang 图解教程 · 看懂高性能 LLM 服务引擎内部 / SGLang Visual Guide"
    desc = ("从零理解整个 SGLang 高性能 LLM/多模态服务引擎的中英双语图解教程：前端 DSL、运行时引擎、"
            "调度器、RadixAttention 前缀缓存、模型执行器与高性能算子，每课配真实源码对应、折叠深挖与设计亮点。")

    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{LANG_BOOT}
<title>{esc(title_tag)}</title>
{head_meta(title_tag, desc, og_type="website")}
<style>{CSS}</style>
</head><body>
<div class="topbar">
  <div class="topbar-inner">
    <span class="home">⚡ <b class="lang-zh">SGLang 图解教程</b><b class="lang-en">SGLang Visual Guide</b></span>
    <span class="pill"><span class="lang-zh">共 {total} 课 · {nparts} 个部分</span><span class="lang-en">{total} lesson{'' if total == 1 else 's'} · {nparts} part{'' if nparts == 1 else 's'}</span></span>
    <button class="langtoggle" onclick="sgvgToggleLang()" aria-label="switch language"><span class="lang-zh">EN</span><span class="lang-en">中</span></button>
  </div>
  <div class="progress"><span style="width:100%"></span></div>
</div>
<div class="wrap">
  <div class="hero index">
    <div class="part">{bi("从零开始 · 面向完全新手", "From scratch · for complete beginners")}</div>
    <h1><span class="lang-zh">用图解读懂整个 SGLang</span><span class="lang-en">Understand all of SGLang, visually</span></h1>
    <p class="lead"><span class="lang-zh">这套教程带你<strong>层层深入</strong>：先建立<strong>宏观全景</strong>与<strong>推理服务基础</strong>，再看懂<strong>前端 DSL 与运行时引擎</strong>的分工，
    然后顺着请求走一遍<strong>分词</strong>·<strong>调度</strong>·<strong>模型执行</strong>·<strong>采样</strong>四个环节，深入 <strong>RadixAttention 前缀缓存</strong>与<strong>重叠调度器</strong>，最后学会<strong>本地构建与贡献</strong>。每课配真实源码对应、图解与设计亮点。</span>
    <span class="lang-en">A layered tour: build the <strong>big picture</strong> and <strong>serving foundations</strong> first, understand how the <strong>frontend DSL</strong> and the <strong>runtime engine</strong> split the work,
    then follow a request through <strong>tokenize</strong>, <strong>schedule</strong>, <strong>model execution</strong> and <strong>sampling</strong>, dive into <strong>RadixAttention prefix caching</strong> and the <strong>overlap scheduler</strong>, and finally learn to <strong>build and contribute</strong>. Every lesson maps to real source, with diagrams and design insights.</span></p>
    <div class="legend">
      <span><i style="background:var(--blue)"></i>{bi("宏观理解", "Big picture")}</span>
      <span><i style="background:var(--purple)"></i>{bi("细节 / 源码", "Details / source")}</span>
      <span><i style="background:var(--amber)"></i>{bi("生活类比", "Analogy")}</span>
      <span><i style="background:var(--accent)"></i>{bi("关键要点", "Key points")}</span>
    </div>
    <p style="margin:.8rem 0 0;color:var(--faint);font-size:.8rem">{bi("📌 对照 sgl-project/sglang 仓库真实源码核实 · 源码引用以“文件 + 符号名”为主（行号随上游更新而变）", "📌 Verified against the real sgl-project/sglang source; references cite file + symbol (line numbers drift upstream)")}</p>
  </div>
  <div class="toc-search">
    <input id="q" type="search" placeholder="🔎 搜索课程 / Search lessons" autocomplete="off" aria-label="search">
    <span class="qcount" id="qcount"></span>
  </div>
  <div class="toc">{toc}</div>
  <div class="toc-empty" id="tocempty">{bi("没有匹配的课程，换个关键词试试。", "No matching lessons, try another keyword.")}</div>
  <p style="margin:2.4rem 0 0;color:var(--faint);font-size:.78rem;text-align:center">{bi("本项目是第三方、非官方的学习材料，不含 SGLang 源码（仅引用少量标注来源的片段）；SGLang 由其作者以 Apache-2.0 许可发布。", "Third-party, unofficial learning material; contains no SGLang source code beyond small, cited snippets. SGLang is Apache-2.0-licensed by its authors.")}</p>
</div>
<script>{LANG_JS}</script>
<script>{SEARCH_JS}</script>
</body></html>"""
