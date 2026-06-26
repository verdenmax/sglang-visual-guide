# SGLang Visual Guide / SGLang 图解学习指南

<p align="center">
  <a href="https://verdenmax.github.io/sglang-visual-guide/"><b>📖&nbsp; Read the guide online&nbsp; →&nbsp; verdenmax.github.io/sglang-visual-guide</b></a>
  <br>
  <sub>A bilingual (English&nbsp;+&nbsp;中文) visual guide to the internals of SGLang, the high-performance LLM &amp; multimodal serving engine — runs in your browser, zero install.<br>点开即读这份 SGLang 内部原理图解指南 · 中英双语 · 浏览器直读 · 无需安装</sub>
</p>

[![Read online](https://img.shields.io/badge/Read_online-Live_Demo-7c48e6?logo=githubpages&logoColor=white)](https://verdenmax.github.io/sglang-visual-guide/)
[![CI](https://github.com/verdenmax/sglang-visual-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/verdenmax/sglang-visual-guide/actions/workflows/ci.yml)
[![Deploy](https://github.com/verdenmax/sglang-visual-guide/actions/workflows/deploy.yml/badge.svg)](https://github.com/verdenmax/sglang-visual-guide/actions/workflows/deploy.yml)
[![Parts](https://img.shields.io/badge/parts-13-7c48e6)](https://verdenmax.github.io/sglang-visual-guide/)
[![Explains SGLang](https://img.shields.io/badge/explains-sglang-7c48e6?logo=github&logoColor=white)](https://github.com/sgl-project/sglang)
[![Dependencies](https://img.shields.io/badge/dependencies-0-2b8a3e)](#build--validate)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC_BY_4.0-blue.svg)](LICENSE-CONTENT)

A visual, bilingual (English + 中文) guide to the internals of
[SGLang](https://github.com/sgl-project/sglang) — the high-performance serving engine for large
language models and multimodal models — that takes you from *"what is a serving engine"* all the way
to *"how the scheduler, RadixAttention, model executor and kernels work in the code"* and *"how to
build, test and contribute a PR"*.

> **Disclaimer:** This is **third-party, unofficial** educational material *about* SGLang. It
> contains **no SGLang source code** beyond small, cited snippets; it explains SGLang by quoting
> short, attributed excerpts. SGLang itself is **Apache-2.0**-licensed by its own authors.

> **Status:** **complete — all 13 parts / 63 lessons shipped.** The full bilingual guide is built and
> committed: 63 lessons across 13 parts, every page validated (`check_html.py` + `check_links.py` pass
> with **0 errors / 0 warnings**, all internal links resolve), and every code excerpt cited to a real
> symbol in the [SGLang source](https://github.com/sgl-project/sglang). Built milestone-driven (M0–M14);
> see `docs/superpowers/plans/` for the per-milestone specs and plans.

Every lesson is self-contained, embeds both languages (toggle in the page), and uses hand-drawn
diagrams, layered architecture maps, real (cited) code, and a short self-test quiz.

---

## What it covers

The guide follows a **"life of a request"** throughline and is organized into **thirteen parts**
(foundations → frontend DSL → entrypoints → scheduler → model execution → KV cache → layers →
kernels → performance innovations → advanced → practice → design synthesis):

| Part | Topic (zh / en) | Lessons |
| --- | --- | --- |
| 1 | 宏观全景 / Overview — what SGLang is, the project map, the life of a request | L01–03 |
| 2 | 推理前置基础 / Foundations — autoregressive decode, continuous batching, paged & RadixAttention, throughput vs latency | L04–08 |
| 3 | 前端语言 DSL / The frontend language — `gen`/`select`/`fork`/`join`, interpreter & tracer, prefix-cache fit, backends | L09–12 |
| 4 | 服务入口与编排 / Entrypoints — Engine & HTTP server, TokenizerManager, OpenAI compat, IO structs & IPC, detokenizer/streaming | L13–17 |
| 5 | 调度器（心脏）/ The scheduler — event loop, Req & ScheduleBatch, policy, zero-overhead overlap, chunked prefill, DP/PP | L18–23 |
| 6 | 模型执行 / Model execution — ModelRunner & ForwardBatch, weight loading, writing a model, CUDA graphs, Sampler | L24–28 |
| 7 | KV 缓存与内存 / KV cache & memory — RadixAttention impl, paged memory pools, HiCache tiering, eviction & hit rate | L29–32 |
| 8 | Attention 与算子层 / Attention & layers — attention backends, MoE, quantization, RoPE/Norm, logits processing | L33–37 |
| 9 | 内核与硬件（深入）/ Kernels & hardware — sgl-kernel, JIT kernels, attention-kernel dissection, fusion, multi-hardware backends | L38–42 |
| 10 | 性能创新专题 / Performance innovations — speculative decoding, EAGLE/next-gen, PD disaggregation, TP/PP/EP/DP, large-scale EP, structured outputs | L43–48 |
| 11 | 进阶·选读 / Advanced (optional) — multimodal VLM, multi-LoRA, RL rollout backend, diffusion models | L49–52 |
| 12 | 实战与贡献 / Practice & contributing — build & run, benchmark & profiling, tests & CI, conventions & PRs, glossary | L53–57 |
| 13 | 设计专题·综合 / Design themes (synthesis) — RadixAttention, zero-overhead scheduling, PD disaggregation, speculative decoding, pluggability, built-for-throughput | L58–63 |

> All **63 lessons** across these 13 parts are complete and committed. Each lesson is bilingual
> (中文 / English, toggle in the page), ≥3500 CJK of prose, with 4+ diagrams per language and a
> self-test quiz (the glossary, L57, is a reference table). `MAX_LESSON` in `check_html.py` is 63.

## How to view

**Online:** published via GitHub Pages at **https://verdenmax.github.io/sglang-visual-guide/**.

**Locally** (zero dependencies, just Python 3):

```bash
cd src
python3 build.py
# then open ../index.html in a browser
```

## How to print / export a PDF

```bash
cd src
python3 build_print.py
# open ../print_zh.html (Chinese) or ../print_en.html (English) in a browser,
# then File -> Print -> Save as PDF (Ctrl/Cmd+P). Each lesson starts on a new page.
```

## Build & validate

```bash
cd src
python3 build.py          # regenerate index.html + lessons/*.html
python3 build_print.py    # regenerate print_zh.html + print_en.html
python3 check_html.py     # structural checks (0 error / 0 warning expected)
python3 check_links.py    # all internal links must resolve
```

The generated HTML is committed and kept in sync with the sources; a re-run of `build.py` should
produce no diff.

## Project structure

```
src/            generators + tooling (pure Python 3, no dependencies)
  part1.py .. part13.py   lesson content (bilingual), one module per part (all 13 present)
  quizzes.py              per-lesson self-test questions
  shell.py                page shell + the shared CSS design system
  registry.py             ordered filename -> content map
  build.py                builds index.html + lessons/*.html
  build_print.py          builds print_zh.html + print_en.html
  check_html.py           structural HTML validation
  check_links.py          internal link validation
lessons/        generated lesson pages (committed, kept in sync)
index.html      generated table of contents (committed)
print_*.html    generated print editions (committed)
docs/superpowers/   design specs and implementation plans
```

## License

Dual-licensed:

- **Code** (the Python generators and validation scripts under `src/`) — MIT, see [LICENSE](LICENSE).
- **Content** (the lesson text and diagrams rendered into `index.html`, `lessons/*.html`,
  `print_*.html`) — CC BY 4.0, see [LICENSE-CONTENT](LICENSE-CONTENT).

---

## 中文说明

这是一份 [SGLang](https://github.com/sgl-project/sglang) 内部原理的**图解、双语**学习指南。SGLang 是
面向大语言模型与多模态模型的**高性能服务引擎**；本指南从"SGLang 是什么"一路讲到"调度器、
RadixAttention、模型执行器与算子内核在代码里怎么走"以及"怎么本地构建、测试、提一个 PR"。

> **声明：** 本项目是**第三方、非官方**的学习材料，**不包含 SGLang 源码**（仅引用少量、标注来源的
> 代码片段来讲解）。SGLang 本身由其作者以 **Apache-2.0** 许可发布。

> **进度：** **已完成 —— 全部 13 个部分 / 63 课。** 整份中英双语指南已构建并提交：13 部分共 63 课，
> 每页均通过校验（`check_html.py` + `check_links.py` **0 错 0 警**，全部内链解析通过），每段代码摘录
> 都标注到 [SGLang 源码](https://github.com/sgl-project/sglang)中的真实符号。本指南按里程碑（M0–M14）
> 构建，每个里程碑的 spec 与 plan 见 `docs/superpowers/plans/`。

每一课都自成一体、内嵌中英双语（页内可切换），用手绘图、分层架构图、真实（标注来源的）代码和一段
自测题来讲清一个概念。

**十三个部分**（沿"一次请求的一生"层层递进）：① 宏观全景（L01-03）② 推理前置基础（L04-08）
③ 前端语言 DSL（L09-12）④ 服务入口与编排（L13-17）⑤ 调度器·心脏（L18-23）⑥ 模型执行（L24-28）
⑦ KV 缓存与内存（L29-32）⑧ Attention 与算子层（L33-37）⑨ 内核与硬件·深入（L38-42）
⑩ 性能创新专题（L43-48）⑪ 进阶·选读（L49-52）⑫ 实战与贡献（L53-57）⑬ 设计专题·综合（L58-63）。
全部 63 课均已完成并提交，每课内嵌中英双语、配 4+ 张图与自测题（术语表 L57 为速查表）。

**怎么看：** 在线版见 **https://verdenmax.github.io/sglang-visual-guide/**；本地零依赖，
`cd src && python3 build.py` 后用浏览器打开 `index.html`。

**怎么打印：** `cd src && python3 build_print.py`，再打开 `print_zh.html`（中文）或
`print_en.html`（英文），用 `Ctrl/Cmd+P` 导出 PDF，每课自动分页。

**许可：** 双许可 —— 代码（`src/` 下的 Python 生成器与校验脚本）用 MIT（见 LICENSE），
教学内容（课程文字与图）用 CC BY 4.0（见 LICENSE-CONTENT）。
