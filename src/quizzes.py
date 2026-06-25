"""Per-lesson bilingual self-test (自测题): design-insight multiple-choice + open prompts.

Schema per lesson::

    "NN-file.html": {
        "mcq": [
            {
                "q":   {"zh": "...", "en": "..."},
                "opts": [{"zh": "...", "en": "..."}, ...],
                "answer": 1,                      # 0-based index into opts (as written)
                "why": {"zh": "...", "en": "..."},
            },
        ],
        "open": [{"zh": "...", "en": "..."}],
    }

``render(fname, lang)`` turns it into HTML that build.py appends to the bottom of
each language's lesson body. Options are deterministically shuffled per question
(same permutation for zh and en, so the correct letter matches across languages).

Quiz text (q/opts/why) is raw HTML in a text context (like the lesson body):
write literal ``<``/``&`` as ``&lt;``/``&amp;`` (or wrap code in ``<code>``).
"""
import hashlib

_HEAD = {"zh": "🧪 自测 · 想一想为什么这么设计", "en": "🧪 Self-test - think about the design"}
_SEE = {"zh": "看答案与解析", "en": "Show answer &amp; explanation"}
_CLICK = {"zh": "点击展开", "en": "click to expand"}
_ANS = {"zh": "答案：", "en": "Answer: "}
_SEP = {"zh": "。", "en": ". "}
_OPEN = {
    "zh": "💭 发散思考（没有标准答案，动手或动脑想想）",
    "en": "💭 Open questions (no single right answer - just think or try)",
}


def _shuffle(opts, answer, seed):
    """Deterministically permute opts (stable across builds); return
    (new_opts, new_answer_index) so the correct option lands in a varied slot."""
    order = sorted(
        range(len(opts)),
        key=lambda i: hashlib.md5(f"{seed}:{i}".encode("utf-8")).hexdigest(),
    )
    return [opts[i] for i in order], order.index(answer)


QUIZZES = {
    "01-what-is-sglang.html": {
        "mcq": [
            {
                "q": {
                    "zh": "SGLang 常被称为“高性能服务引擎（serving engine）”。把它和一个普通的“模型训练脚本”相比，它的核心定位最准确的是什么？",
                    "en": "SGLang is called a high-performance serving engine. Compared with an ordinary model-training script, what most accurately describes its core role?",
                },
                "opts": [
                    {
                        "zh": "它负责把训练好的模型在线上“跑起来对外服务”：高吞吐、低延迟地接收大量并发请求并生成 token，围绕推理把批处理、缓存、调度都做到极致",
                        "en": "It puts a trained model online to serve: high-throughput, low-latency generation under heavy concurrency, pushing batching, caching and scheduling to the limit around inference",
                    },
                    {"zh": "它是一个训练框架，专注反向传播与梯度更新", "en": "It is a training framework focused on backprop and gradient updates"},
                    {"zh": "它是一个向量数据库，把文本编码成向量做相似检索", "en": "It is a vector database that encodes text for similarity search"},
                    {"zh": "它只是一个把多个模型权重合并的转换工具", "en": "It is just a tool that merges multiple model weights"},
                ],
                "answer": 0,
                "why": {
                    "zh": "服务引擎解决的是“推理上线”这一段：模型已经训练好，关键是如何在大量并发下又快又省地生成 token。SGLang 通过连续批处理、RadixAttention 前缀缓存、零开销调度器把这件事做到极致——它不做训练，也不是数据库。",
                    "en": "A serving engine owns the inference-online phase: the model is already trained; the challenge is generating tokens fast and cheaply under concurrency. SGLang pushes this via continuous batching, RadixAttention prefix caching and a zero-overhead scheduler — it does not train, and it is not a database.",
                },
            },
            {
                "q": {
                    "zh": "SGLang 由“两半”组成：前端 DSL（lang/）与运行时引擎（srt/）。这两半的分工是什么？",
                    "en": "SGLang has two halves: the frontend DSL (lang/) and the runtime engine (srt/). How do they divide the work?",
                },
                "opts": [
                    {
                        "zh": "前端 DSL 让你用 gen/fork/join 等原语“表达”复杂的多步调用与并行；运行时引擎负责把请求真正“执行”出来——分词、调度、模型前向、采样、反分词",
                        "en": "The frontend DSL lets you express multi-step, parallel LLM programs with gen/fork/join; the runtime engine actually executes requests — tokenize, schedule, model forward, sample, detokenize",
                    },
                    {"zh": "前端负责训练，运行时负责评测", "en": "The frontend trains, the runtime evaluates"},
                    {"zh": "两半都是可选的 GUI，核心逻辑在别处", "en": "Both halves are optional GUIs; the real logic is elsewhere"},
                    {"zh": "前端是 C++ 内核，运行时是 Python 脚本", "en": "The frontend is the C++ kernel, the runtime is a Python script"},
                ],
                "answer": 0,
                "why": {
                    "zh": "“前端表达、后端执行”是 SGLang 的主线：lang/ 提供 gen/fork/join 这类原语，让你像写程序一样编排 LLM 调用；srt/ 才是真正的服务引擎，把每个请求送过 TokenizerManager → Scheduler → ModelRunner → Sampler → DetokenizerManager 这条流水线。",
                    "en": "Express up front, execute in the back is SGLang's throughline: lang/ offers gen/fork/join to orchestrate LLM calls like a program; srt/ is the real engine, pushing each request through TokenizerManager → Scheduler → ModelRunner → Sampler → DetokenizerManager.",
                },
            },
            {
                "q": {
                    "zh": "RadixAttention（前缀缓存）为什么能显著提速？它复用的是什么？",
                    "en": "Why does RadixAttention (prefix caching) speed things up so much? What does it reuse?",
                },
                "opts": [
                    {
                        "zh": "复用“共享前缀”已经算好的 KV 缓存：多个请求若开头相同（同一系统提示、同一few-shot 模板），那段的注意力计算只做一次，后续请求直接命中",
                        "en": "It reuses the already-computed KV cache of shared prefixes: when many requests start the same way (same system prompt, same few-shot template), that prefix is computed once and later requests hit the cache",
                    },
                    {"zh": "复用磁盘上的模型权重文件，避免重复加载", "en": "It reuses the on-disk weight files to avoid reloading"},
                    {"zh": "复用 GPU 的电源管理状态以省电", "en": "It reuses the GPU power state to save energy"},
                    {"zh": "复用网络连接，减少 TCP 握手", "en": "It reuses network connections to cut TCP handshakes"},
                ],
                "answer": 0,
                "why": {
                    "zh": "大量真实请求共享相同开头：系统提示、few-shot 例子、对话历史。RadixAttention 用基数树把这些前缀的 KV 缓存组织起来，相同前缀只算一次注意力，命中即省——这正是“边批边复用”里“复用”的来源（第 7 课展开）。",
                    "en": "Many real requests share the same opening: system prompts, few-shot examples, chat history. RadixAttention organizes those prefixes' KV cache in a radix tree so a shared prefix's attention is computed once and reused on a hit — the reuse half of batch-and-reuse (detailed in Lesson 7).",
                },
            },
            {
                "q": {
                    "zh": "“零开销重叠调度器（zero-overhead overlap scheduler）”想解决的核心问题是什么？",
                    "en": "What core problem does the zero-overhead overlap scheduler target?",
                },
                "opts": [
                    {
                        "zh": "让 CPU 的调度/准备工作和 GPU 的前向计算“重叠”进行，把 CPU 开销藏到 GPU 忙碌的时间里，避免 GPU 因等 CPU 而空转",
                        "en": "Overlap the CPU's scheduling/prep with the GPU's forward pass, hiding CPU overhead behind GPU-busy time so the GPU never idles waiting for the CPU",
                    },
                    {"zh": "彻底取消批处理，让每个请求独占一张 GPU", "en": "Abolish batching so each request owns a whole GPU"},
                    {"zh": "把所有计算搬到 CPU，绕开 GPU", "en": "Move all compute to the CPU to bypass the GPU"},
                    {"zh": "通过降低精度到 INT1 来减少计算量", "en": "Drop precision to INT1 to cut compute"},
                ],
                "answer": 0,
                "why": {
                    "zh": "GPU 很贵也很快，最怕“空转等 CPU”。重叠调度器在 GPU 算当前批次的同时，CPU 已经在准备下一批，让两者流水线化重叠，从而把调度开销近似降为零——这是 SGLang 高吞吐的关键之一（第 21 课展开）。",
                    "en": "The GPU is expensive and fast; the worst case is idling while it waits on the CPU. The overlap scheduler prepares the next batch on the CPU while the GPU runs the current one, pipelining them so scheduling overhead is effectively zero — a key to SGLang's throughput (detailed in Lesson 21).",
                },
            },
        ],
        "open": [
            {
                "zh": "假设你要为一个客服机器人上线服务：成千上万的对话都以同一段很长的“系统提示 + 公司知识”开头，只有结尾的用户问题不同。结合连续批处理与 RadixAttention，说说为什么 SGLang 这类引擎能比“逐条独立推理”省下大量算力？哪些设计在这里发挥作用？",
                "en": "Suppose you deploy a customer-service bot: thousands of conversations all begin with the same long system prompt + company knowledge, differing only in the final user question. Using continuous batching and RadixAttention, explain why an engine like SGLang saves so much compute versus running each request independently. Which designs matter here?",
            },
            {
                "zh": "SGLang 强调“从单卡到大型集群、覆盖 NVIDIA/AMD/TPU/NPU/CPU 等多种硬件”。为什么一个服务引擎要把“可移植 + 可扩展”当成一等目标？这对它的内部分层（前端 DSL / 运行时 / 算子内核）提出了什么要求？",
                "en": "SGLang stresses running from a single GPU to large clusters across NVIDIA/AMD/TPU/NPU/CPU. Why should a serving engine treat portability + scalability as first-class goals? What does that demand of its internal layering (frontend DSL / runtime / operator kernels)?",
            },
        ],
    },
    "02-project-map.html": {
        "mcq": [
            {
                "q": {
                    "zh": "仓库分成 python/sglang/lang/ 与 python/sglang/srt/ 两半，其中 srt（SGLang RunTime）这一半负责什么？",
                    "en": "The repo splits into python/sglang/lang/ and python/sglang/srt/. What does the srt (SGLang RunTime) half own?",
                },
                "opts": [
                    {
                        "zh": "运行时引擎：真正执行请求——分词、调度、模型前向、采样、反分词，内置 RadixAttention、连续批处理、并行等，是性能的来源",
                        "en": "The runtime engine: actually executing requests — tokenize, schedule, model forward, sample, detokenize — with RadixAttention, continuous batching, parallelism; the source of performance",
                    },
                    {"zh": "前端 DSL：用 gen/fork/join 表达多步并行的 LLM 程序", "en": "The frontend DSL: expressing multi-step parallel LLM programs with gen/fork/join"},
                    {"zh": "模型训练：反向传播与梯度更新", "en": "Model training: backprop and gradient updates"},
                    {"zh": "一个独立的向量数据库，用于相似检索", "en": "A standalone vector database for similarity search"},
                ],
                "answer": 0,
                "why": {
                    "zh": "srt = SGLang RunTime，是真正的服务引擎，把每条请求送过分词→调度→前向→采样→反分词的流水线；lang/ 才是前端 DSL。两半的分工是“前端表达、后端执行”，而 srt/ 是这门教程深挖的主角。",
                    "en": "srt = SGLang RunTime, the real serving engine pushing each request through tokenize→schedule→forward→sample→detokenize; lang/ is the frontend DSL. The split is express-up-front, execute-in-back, and srt/ is the star this guide digs into.",
                },
            },
            {
                "q": {
                    "zh": "引擎跑起来时会裂成三个进程，靠 ZMQ 的 IPC 通信。下列哪一项对“谁握着 GPU”的描述是正确的？",
                    "en": "At run time the engine forks into three processes talking over ZMQ IPC. Which statement about who holds the GPU is correct?",
                },
                "opts": [
                    {
                        "zh": "Scheduler 子进程握着 GPU：它持有 TpWorker/ModelRunner 做前向，张量并行时每个 TP rank 各起一个；两个 Manager 只是一进一出的“翻译官”",
                        "en": "The Scheduler subprocess holds the GPU: it owns TpWorker/ModelRunner for the forward (one per TP rank under tensor parallelism); the two Managers are just in/out translators",
                    },
                    {"zh": "TokenizerManager 握着 GPU，Scheduler 只在 CPU 上排队", "en": "TokenizerManager holds the GPU; the Scheduler only queues on the CPU"},
                    {"zh": "DetokenizerManager 握着 GPU 做模型前向", "en": "DetokenizerManager holds the GPU and runs the model forward"},
                    {"zh": "三个进程共享同一段 GPU 显存，没有明确分工", "en": "All three processes share the same GPU memory with no clear division"},
                ],
                "answer": 0,
                "why": {
                    "zh": "TokenizerManager 在主进程（连 HTTP 服务器、Engine 也在主进程），Scheduler 与 DetokenizerManager 在子进程。真正占着 GPU 做前向的是 Scheduler 子进程里的 ModelRunner，且张量并行时每个 rank 一个 Scheduler；两个 Manager 只做文本↔token 的互转，千万别混。",
                    "en": "TokenizerManager runs in the main process (so do the HTTP server and Engine); Scheduler and DetokenizerManager are subprocesses. The one occupying the GPU is the ModelRunner inside the Scheduler subprocess, with one Scheduler per rank under TP; the two Managers only convert text↔tokens — don't conflate them.",
                },
            },
            {
                "q": {
                    "zh": "为什么 SGLang 要把分词、调度+前向、反分词拆成独立进程、用 ZMQ 通信，而不是放进一个进程的多个线程？",
                    "en": "Why does SGLang split tokenize, schedule+forward, and detokenize into separate processes over ZMQ, instead of threads in one process?",
                },
                "opts": [
                    {
                        "zh": "Python 的 GIL 让多线程难以让 CPU 分词与 GPU 前向真正并行；独立进程各有独立解释器，能物理上同时推进，ZMQ 提供异步解耦的消息队列，正是“零开销重叠”的基础",
                        "en": "Python's GIL keeps threads from truly parallelizing CPU tokenization and GPU forward; separate processes each have their own interpreter and advance physically in parallel, with ZMQ as an async-decoupled message queue — the basis of zero-overhead overlap",
                    },
                    {"zh": "因为 ZMQ 比函数调用更快，能减少 CPU 指令数", "en": "Because ZMQ is faster than a function call and cuts CPU instructions"},
                    {"zh": "为了让每个组件能用不同的编程语言编写", "en": "So each component can be written in a different programming language"},
                    {"zh": "纯粹是历史包袱，并无性能上的理由", "en": "Purely historical baggage with no performance reason"},
                ],
                "answer": 0,
                "why": {
                    "zh": "根因是 Python 的全局解释器锁：多线程很难让 CPU 活与 GPU 活真正并行。拆成独立进程后，分词、前向、反分词可以物理上同时进行，ZMQ 用每进程一个端口的消息队列做异步解耦，发送方不必等接收方——这正是 CPU 调度与 GPU 计算得以重叠的物理前提（第 21 课）。",
                    "en": "The root cause is Python's global interpreter lock: threads can't truly parallelize CPU and GPU work. Split into processes, tokenize/forward/detokenize run physically at once, and ZMQ's per-process message queue decouples them asynchronously so the sender need not wait — the physical prerequisite for overlapping CPU scheduling with GPU compute (Lesson 21).",
                },
            },
        ],
        "open": [
            {
                "zh": "假设你要给 srt/ 加一个支持新硬件后端的注意力内核。结合本课的“分层 + 子包”地图，说说你大概会改动哪些目录、又能基本不碰哪些目录？为什么这种高内聚、低耦合的切分对一个上千人参与的开源项目很重要？",
                "en": "Suppose you add an attention kernel for a new hardware backend to srt/. Using this lesson's layered + sub-package map, which directories would you likely touch and which could you mostly leave alone? Why does this high-cohesion, low-coupling split matter for a thousand-contributor open-source project?",
            },
            {
                "zh": "有人提议把三个进程合并成一个进程里的三个线程，“省掉 ZMQ 的序列化开销”。结合 GIL 与 CPU/GPU 并行的现实，评价这个提议的利弊，以及它会如何影响“零开销重叠调度”。",
                "en": "Someone proposes merging the three processes into three threads in one process to save ZMQ serialization overhead. Considering the GIL and the reality of CPU/GPU parallelism, weigh this proposal's pros and cons and how it would affect zero-overhead overlap scheduling.",
            },
        ],
    },
    "03-life-of-a-request.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在调度器的事件循环里，连续批处理“插队”的能力主要来自哪一步？",
                    "en": "In the scheduler's event loop, where does continuous batching's ability to cut in mainly come from?",
                },
                "opts": [
                    {
                        "zh": "get_next_batch_to_run：它每一圈都重新组批，所以新请求随时能加入、已结束的随时让位，GPU 几乎不空转",
                        "en": "get_next_batch_to_run: it re-forms the batch every turn, so a new request can join and a finished one yield any turn, keeping the GPU busy",
                    },
                    {"zh": "recv_requests：它一次性收完所有请求后才开始算", "en": "recv_requests: it gathers all requests once before computing anything"},
                    {"zh": "process_batch_result：它把整批一起结束再开下一批", "en": "process_batch_result: it ends the whole batch together before starting the next"},
                    {"zh": "on_idle：它在空闲时预先生成 token", "en": "on_idle: it pre-generates tokens while idle"},
                ],
                "answer": 0,
                "why": {
                    "zh": "事件循环每一圈都调用 get_next_batch_to_run 重新组批，这正是“插队/让位”的发生点：新请求下一圈即可加入，结束的请求立刻让出槽位，所以批次始终接近满载（第 5、20 课）。这是 SGLang 高 GPU 利用率的核心机制。",
                    "en": "Each loop turn calls get_next_batch_to_run to re-form the batch — exactly where cut-in/yield happens: a new request joins next turn, a finished one frees its slot at once, so the batch stays near full (Lessons 5, 20). This is the core of SGLang's high GPU utilization.",
                },
            },
            {
                "q": {
                    "zh": "关于预填充（prefill）和解码（decode）两个阶段，下面哪种说法最准确？",
                    "en": "Which statement about the prefill and decode phases is most accurate?",
                },
                "opts": [
                    {
                        "zh": "预填充把整段提示一次性读入、计算密集、通常只做一次；解码每次只算最新一个 token、访存密集、循环很多次。一条请求 = 一次预填充 + 很多次解码",
                        "en": "Prefill reads the whole prompt at once, is compute-bound, usually done once; decode computes only the newest token each turn, is memory-bound, loops many times. One request = one prefill + many decodes",
                    },
                    {"zh": "预填充和解码完全一样，只是名字不同", "en": "Prefill and decode are identical, just named differently"},
                    {"zh": "解码计算密集、只做一次；预填充访存密集、循环很多次", "en": "Decode is compute-bound and done once; prefill is memory-bound and loops many times"},
                    {"zh": "两个阶段都不碰 KV 缓存", "en": "Neither phase touches the KV cache"},
                ],
                "answer": 0,
                "why": {
                    "zh": "预填充把几百上千个提示 token 并行算成一大片、把 KV 缓存填好，是计算密集的、通常只做一次；解码每圈只算最新一个 token，瓶颈在反复搬运越来越长的 KV 缓存，是访存密集的、要循环很多次。分清这两段，是理解后续所有显存/带宽优化的钥匙（第 6、22、29–32 课）。",
                    "en": "Prefill computes hundreds-to-thousands of prompt tokens in parallel as a slab and fills the KV cache — compute-bound, usually once; decode computes only the newest token per turn, bottlenecked by moving an ever-growing KV cache — memory-bound, looping many times. Telling them apart is the key to all later memory/bandwidth optimizations (Lessons 6, 22, 29–32).",
                },
            },
            {
                "q": {
                    "zh": "为什么前端看到的回答是“一个词一个词蹦出来”的流式效果？",
                    "en": "Why does the answer appear word by word as a streaming effect in the frontend?",
                },
                "opts": [
                    {
                        "zh": "因为生成是自回归逐 token 的，DetokenizerManager 做增量反分词，只把这次新增、能安全显示的文字推回，经主进程 SSE 流式发出，无需等整段生成完",
                        "en": "Because generation is autoregressive token-by-token; the DetokenizerManager does incremental detokenization, pushing only the newly added, safe-to-show text back via the main process as SSE, without waiting for the whole generation",
                    },
                    {"zh": "因为模型先把整段答案算完，再人为切成小段慢慢显示", "en": "Because the model finishes the whole answer first, then artificially slices it to display slowly"},
                    {"zh": "因为网络带宽不足，只能一段段传", "en": "Because of insufficient bandwidth, it can only send in pieces"},
                    {"zh": "因为前端 JavaScript 故意加了打字机动画", "en": "Because the frontend JavaScript deliberately adds a typewriter animation"},
                ],
                "answer": 0,
                "why": {
                    "zh": "生成本身就是逐 token 的：每算出一个 token，Detokenizer 就把“真正新增、能安全显示”的那一小段（用 sent_offset 记住进度）增量地翻成文字，经主进程以 SSE 推给用户。所以用户在第一个 token 算出的瞬间就开始看到回答，而不是等整段生成结束——这对长回答的体验至关重要。",
                    "en": "Generation is inherently token-by-token: as each token is produced, the Detokenizer incrementally turns the genuinely new, safe-to-show slice (tracking progress via sent_offset) into text and the main process streams it as SSE. The user sees the answer the instant the first token is computed, not after the whole generation — crucial for long-answer experience.",
                },
            },
        ],
        "open": [
            {
                "zh": "把一条 generate 请求的全程画成一张图，标出三个进程的边界和每一次 ZMQ 传递。然后解释：当 Scheduler 正在为第 N 个 token 做前向时，Detokenizer 和主进程可能分别在做什么？这种跨进程流水线如何同时提升吞吐与首 token 延迟体验？",
                "en": "Draw a generate request's whole path, marking the three process boundaries and each ZMQ hop. Then explain: while the Scheduler forwards token N, what might the Detokenizer and the main process each be doing? How does this cross-process pipeline improve both throughput and first-token latency experience?",
            },
            {
                "zh": "一个客服系统里，请求们的提示长度差异很大、到达时机也不同。结合“每圈重新组批”的事件循环与“预填充 vs 解码”的区别，说说为什么调度器循环必须足够快，以及把太长的提示一次喂进去会带来什么问题、SGLang 用什么手段缓解（提示：分块预填充）。",
                "en": "In a customer-service system, requests vary widely in prompt length and arrival time. Using the re-batch-every-turn loop and the prefill-vs-decode distinction, explain why the scheduler loop must be fast, what problem feeding one very long prompt at once causes, and how SGLang mitigates it (hint: chunked prefill).",
            },
        ],
    },
}


def render(fname, lang):
    """Return the self-test HTML block for ``fname`` in ``lang`` ('' if none)."""
    data = QUIZZES.get(fname)
    if not data or not (data.get("mcq") or data.get("open")):
        return ""
    out = ['<div class="selftest">', f'<h2>{_HEAD[lang]}</h2>']
    for i, item in enumerate(data.get("mcq", []), 1):
        shuffled, ans = _shuffle(item["opts"], item["answer"], f"{fname}:{i}")
        opts = "\n".join(f"    <li>{o[lang]}</li>" for o in shuffled)
        letter = chr(65 + ans)
        out.append(
            f'<div class="quiz">\n'
            f'  <div class="qn">{i}. {item["q"][lang]}</div>\n'
            f'  <ol class="opts">\n{opts}\n  </ol>\n'
            f'  <details class="accordion">\n'
            f'    <summary>{_SEE[lang]} <span class="hint">{_CLICK[lang]}</span></summary>\n'
            f'    <div class="acc-body"><div class="qa"><div class="a">'
            f'<strong>{_ANS[lang]}{letter}</strong>{_SEP[lang]}{item["why"][lang]}'
            f"</div></div></div>\n"
            f"  </details>\n"
            f"</div>"
        )
    opens = data.get("open", [])
    if opens:
        lis = "\n".join(f"    <li>{o[lang]}</li>" for o in opens)
        out.append(
            '<div class="card spark">\n'
            f'  <div class="tag">{_OPEN[lang]}</div>\n'
            f"  <ul>\n{lis}\n  </ul>\n"
            "</div>"
        )
    out.append("</div>")
    return "\n".join(out)


def _validate():
    """Fail fast on authoring mistakes in QUIZZES (clear message names the lesson)."""
    for fname, data in QUIZZES.items():
        for qi, item in enumerate(data.get("mcq", []), 1):
            opts = item["opts"]
            if not (0 <= item["answer"] < len(opts)):
                raise ValueError(
                    f"quizzes[{fname!r}] Q{qi}: answer {item['answer']} out of range 0..{len(opts) - 1}"
                )
            for o in opts:
                if not ({"zh", "en"} <= o.keys()):
                    raise ValueError(f"quizzes[{fname!r}] Q{qi}: an option is missing zh/en")
            if not ({"zh", "en"} <= item["q"].keys() and {"zh", "en"} <= item["why"].keys()):
                raise ValueError(f"quizzes[{fname!r}] Q{qi}: q/why missing zh/en")
        for oi, o in enumerate(data.get("open", []), 1):
            if not ({"zh", "en"} <= o.keys()):
                raise ValueError(f"quizzes[{fname!r}] open{oi}: missing zh/en")


_validate()
