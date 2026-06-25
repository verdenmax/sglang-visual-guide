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
    "04-autoregressive-and-kv-cache.html": {
        "mcq": [
            {
                "q": {
                    "zh": "KV 缓存到底省掉了什么计算？",
                    "en": "What computation does the KV cache actually save?",
                },
                "opts": [
                    {"zh": "历史 token 的 K/V 只读不变，缓存后每生成一个新 token 就不必重算整段历史的 K/V 与注意力", "en": "Historical tokens' K/V are read-only, so caching them avoids recomputing the whole history's K/V and attention for each new token"},
                    {"zh": "省掉了模型权重的加载，让模型只加载一次", "en": "It avoids reloading model weights, loading them only once"},
                    {"zh": "省掉了分词（tokenize）的开销", "en": "It avoids the tokenization cost"},
                    {"zh": "省掉了采样（sampling）这一步", "en": "It removes the sampling step"},
                ],
                "answer": 0,
                "why": {
                    "zh": "注意力里每个 token 的 K/V 在它生成时就固定了。新 token 只需算自己的 Q/K/V，再和缓存里的历史 K/V 做注意力——把每步从“重算整段”O(t²) 压到“只算新词对历史”O(t)。这与权重加载、分词、采样无关。",
                    "en": "Each token's K/V is fixed once generated. A new token computes only its own Q/K/V and attends against the cached history — squeezing each step from re-running the whole sequence O(t²) to O(t). Unrelated to weight loading, tokenization, or sampling.",
                },
            },
            {
                "q": {
                    "zh": "prefill 与 decode 两个阶段，哪个是“访存密集（memory-bound）”的？为什么这很重要？",
                    "en": "Which phase, prefill or decode, is memory-bound, and why does it matter?",
                },
                "opts": [
                    {"zh": "decode：每步只算 1 个新 token 但要读整个 KV 缓存，算力小、带宽吃紧，所以把多条请求拼批一起做才划算", "en": "decode: each step computes just 1 token but reads the whole KV cache — little compute, bandwidth-bound — so batching many requests together pays off"},
                    {"zh": "prefill：因为它一次处理整段 prompt", "en": "prefill: because it processes the whole prompt at once"},
                    {"zh": "两个都不是，瓶颈永远在网络", "en": "Neither; the bottleneck is always the network"},
                    {"zh": "两个都是纯计算密集，和访存无关", "en": "Both are purely compute-bound, unrelated to memory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "decode 单步算力极小，时间几乎都花在把 KV 缓存与权重从显存搬进计算单元，故访存密集；正因如此，一次权重读取服务多条请求的“连续批处理”能显著提升吞吐。prefill 则是计算密集（大矩阵乘）。",
                    "en": "decode does tiny compute per step; time goes to moving the cache and weights from HBM, so it's memory-bound. That's exactly why continuous batching (one weight read serving many requests) lifts throughput. prefill is compute-bound (big matmuls).",
                },
            },
            {
                "q": {
                    "zh": "为什么说 KV 缓存的大小会“限制并发请求数”？",
                    "en": "Why does the size of the KV cache cap the number of concurrent requests?",
                },
                "opts": [
                    {"zh": "显存池容量固定，每条请求按“上下文长度 × 每 token 字节”占用缓存，所以上下文越长、并发越少，二者互相挤占同一块显存", "en": "The HBM pool is fixed; each request consumes 'context length × bytes/token' of cache, so longer context means fewer concurrent requests — they contend for the same HBM"},
                    {"zh": "因为 CPU 核心数有限", "en": "Because the number of CPU cores is limited"},
                    {"zh": "因为每条请求要独占一张 GPU", "en": "Because each request needs its own GPU"},
                    {"zh": "因为网络带宽有限", "en": "Because network bandwidth is limited"},
                ],
                "answer": 0,
                "why": {
                    "zh": "每 token 缓存 ≈ 2×层×KV头×head_dim×字节，随上下文线性增长。固定显存里能放的 token 总数有限，于是“能并发多少条”直接由缓存省不省决定——这正是调度器每步要算“显存够不够再收一条”的原因。",
                    "en": "Per-token cache ≈ 2×layers×KV-heads×head_dim×bytes, growing linearly with context. Fixed HBM holds a fixed token budget, so concurrency is decided by how frugal the cache is — which is why the scheduler checks 'enough HBM to admit one more?' each step.",
                },
            },
        ],
        "open": [
            {
                "zh": "用一个 7B、32 层、8 个 KV 头、head_dim=128、fp16 的模型，估算单条请求在 4096 token 上下文下的 KV 缓存占用。再想想：如果改用 fp8 存 KV，会省多少？这对“能并发多少条”意味着什么？",
                "en": "For a 7B model with 32 layers, 8 KV heads, head_dim=128, fp16, estimate one request's KV-cache footprint at a 4096-token context. Then: if you store KV in fp8, how much do you save, and what does that mean for concurrency?",
            },
            {
                "zh": "自回归是“天生串行”的——单条请求的延迟被逐 token 循环卡死，多算几个 token 也摊薄不了。既然单条快不了，工程上为什么“把更多请求横向拼在一起”几乎是唯一出路？这把你引向了后面哪几课？",
                "en": "Autoregression is inherently serial — one request's latency is pinned by the per-token loop and can't be amortized. Since a single request can't be sped up, why is 'packing more requests side by side' almost the only way out, and which later lessons does that point you to?",
            },
        ],
    },
    "05-continuous-batching.html": {
        "mcq": [
            {
                "q": {
                    "zh": "连续批处理相比静态（padded）批处理，最根本的赢面在哪里？",
                    "en": "What is the most fundamental reason continuous batching beats static (padded) batching?",
                },
                "opts": [
                    {"zh": "它在每个 decode 步重新组批，完成的请求当场离场、等待的立刻补入，批次始终满载，GPU 不为单条请求空转", "en": "It re-forms the batch every decode step — finished requests leave on the spot, waiting ones fill in at once — so the batch stays full and the GPU never idles on any single request"},
                    {"zh": "它把模型权重压缩得更小，从而读得更快", "en": "It compresses the model weights so they read faster"},
                    {"zh": "它跳过了采样步骤，直接取概率最大的 token", "en": "It skips sampling and always takes the argmax token"},
                    {"zh": "它把 decode 变成了计算密集，从而吃满算力", "en": "It turns decode into a compute-bound phase to saturate the ALUs"},
                ],
                "answer": 0,
                "why": {
                    "zh": "静态批处理一次组批、绑死到底：整批等最慢的那条，完成的槽位空转（padding 浪费），新请求要等整批排空。连续批处理每步重组批，让批“永远满”——因为 decode 访存密集，读一次权重就能服务一整批，批越满、固定开销摊得越薄、吞吐越高。它不改权重、不掉精度、不跳采样。",
                    "en": "Static batching forms the batch once and chains it: the batch waits on the slowest, finished slots idle (padding waste), new requests wait for the drain. Continuous batching re-forms the batch each step to keep it full — and since decode is memory-bound, one weight read serves a whole batch, so a fuller batch amortizes the fixed cost more, lifting throughput. It changes no weights, loses no accuracy, skips no sampling.",
                },
            },
            {
                "q": {
                    "zh": "是什么让“某条请求完成后能在批的中途立刻离场、并马上释放显存”成为可能？",
                    "en": "What makes it possible for a finished request to leave the batch mid-flight and free its HBM immediately?",
                },
                "opts": [
                    {"zh": "每条请求的 KV 缓存按行/槽位独立管理，互不纠缠，所以一条结束就能原地回收它的槽位、立刻分给下一个排队请求", "en": "Each request's KV cache is managed independently by row/slot and isn't entangled with others, so when one ends its slots are reclaimed in place and handed to the next queued request at once"},
                    {"zh": "因为所有请求共享同一份 KV 缓存，删一条不影响别人", "en": "Because all requests share one KV cache, so deleting one doesn't affect others"},
                    {"zh": "因为 GPU 会在每步自动垃圾回收整段显存", "en": "Because the GPU auto garbage-collects all HBM every step"},
                    {"zh": "因为静态批处理本来就支持中途退出", "en": "Because static batching already supports leaving mid-batch"},
                ],
                "answer": 0,
                "why": {
                    "zh": "第 4 课讲过：KV 缓存放进一个大显存池、按行/槽位分配，一条请求的 K/V 跟别人互不纠缠。正因如此，请求一结束，它那几行槽位就能立即回收、分给下一个等待者，不必等整批结束或重排——这正是连续批处理能“满载流动”的物理前提。",
                    "en": "Lesson 4 showed the KV cache lives in a big HBM pool, allocated by row/slot, with one request's K/V not entangled with others'. So the moment a request ends, its slots are reclaimed and given to the next waiter — no waiting for the whole batch, no reshuffle. That's the physical prerequisite for the batch to flow at full load.",
                },
            },
            {
                "q": {
                    "zh": "为什么说“连续批处理要求调度器本身必须足够便宜”？",
                    "en": "Why must the scheduler itself be cheap for continuous batching to pay off?",
                },
                "opts": [
                    {"zh": "因为每个 decode 步都要重组批（过滤完成、接纳等待、回收/分配槽位），若每步的 CPU 调度开销过大，GPU 反而会在等组队时空转", "en": "Because every decode step re-forms the batch (filter finished, admit waiting, reclaim/allocate slots); if per-step CPU scheduling is too costly, the GPU ends up idling while waiting for the batch to be formed"},
                    {"zh": "因为调度器要在每步重新加载模型权重", "en": "Because the scheduler reloads model weights each step"},
                    {"zh": "因为调度器负责做矩阵乘法", "en": "Because the scheduler performs the matmuls"},
                    {"zh": "因为便宜的调度器精度更高", "en": "Because a cheaper scheduler is more accurate"},
                ],
                "answer": 0,
                "why": {
                    "zh": "连续批处理把组批逻辑放进了热路径——每步都要做账务。若 CPU 这部分太慢，GPU 就会在等 CPU 组队时空转，吞吐红利被吃掉。这正是 SGLang 用零开销重叠调度器（第 21 课）把组批开销藏进上一步 GPU 计算时间的原因；长 prompt 还要靠分块预填充（第 22 课）避免堵塞。",
                    "en": "Continuous batching puts batch-forming on the hot path — bookkeeping every step. If that CPU work is slow, the GPU idles waiting for the batch, eating the throughput dividend. That's why SGLang's zero-overhead overlap scheduler (Lesson 21) hides the batch-forming cost inside the previous step's GPU compute, and chunked prefill (Lesson 22) keeps long prompts from clogging the queue.",
                },
            },
        ],
        "open": [
            {
                "zh": "假设一个批里有 8 条请求，生成长度分别是 5、5、6、500、5、5、6、5 个 token。用静态批处理 vs 连续批处理分别跑，估算两种方式下“有效 token / 总槽位·步数”的利用率差距，说说浪费到底发生在哪里。",
                "en": "Suppose a batch holds 8 requests with output lengths 5, 5, 6, 500, 5, 5, 6, 5 tokens. Run it with static vs continuous batching and estimate the 'useful tokens / (slots × steps)' utilization gap; explain where exactly the waste occurs.",
            },
            {
                "zh": "连续批处理在每步都问“有没有 prefill 活要先做？没有再做 decode”。如果一直优先 prefill，会怎样影响正在 decode 的请求的延迟？反过来一直优先 decode 又会怎样？这把你引向了第 20 课的什么取舍？",
                "en": "Continuous batching asks each step 'is there prefill work to do first? if not, do decode.' If prefill is always prioritized, how does that affect the latency of already-decoding requests? And if decode is always prioritized? Which tradeoff in Lesson 20 does this lead you to?",
            },
        ],
    },
    "06-paged-attention-and-paged-kv.html": {
        "mcq": [
            {
                "q": {
                    "zh": "为什么“给每条请求按最大长度预留一整块连续显存”会大把浪费 HBM？",
                    "en": "Why does 'reserving one contiguous HBM block per request, sized for the max length' waste so much HBM?",
                },
                "opts": [
                    {"zh": "它同时造成内部碎片（多数请求远到不了最大长度，预留的空槽被锁死没人能用）和外部碎片（请求来去后剩下的零散缝隙拼不成一整块连续显存）", "en": "It causes both internal fragmentation (most requests never reach max length, so reserved empty slots are locked and unusable) and external fragmentation (scattered gaps left as requests come and go can't combine into one contiguous block)"},
                    {"zh": "它会让模型权重占用更多显存", "en": "It makes the model weights occupy more HBM"},
                    {"zh": "它强制每个 token 都重算注意力，浪费算力", "en": "It forces recomputing attention for every token, wasting compute"},
                    {"zh": "它把 decode 变成计算密集，吃满了 ALU", "en": "It turns decode compute-bound and saturates the ALUs"},
                ],
                "answer": 0,
                "why": {
                    "zh": "连续预留按上限定大小，可绝大多数请求实际长度远低于上限，剩下的槽位空着却被锁死（内部碎片）；请求不断来去，释放出的空闲块大小不一、夹在中间，新请求要连续一整块却拼不出来（外部碎片）。两者叠加导致显存没用满就塞不下新请求，直接压低并发上限。它与权重大小、是否重算注意力、decode 的访存/计算属性都无关。",
                    "en": "Contiguous reservation sizes to the cap, but most requests run far shorter, so the leftover slots sit empty yet locked (internal fragmentation); as requests come and go, freed blocks of uneven sizes get wedged in between, and a new request needing one contiguous block can't assemble it (external fragmentation). Together they leave HBM 'full' before it's actually full, capping concurrency. It has nothing to do with weight size, recomputing attention, or decode's memory/compute nature.",
                },
            },
            {
                "q": {
                    "zh": "在 PagedAttention 里，是什么让“一条请求的 KV 可以散落在物理上不相邻的块里”仍然能正确算注意力？",
                    "en": "In PagedAttention, what lets a request's KV live in physically non-adjacent blocks while attention still computes correctly?",
                },
                "opts": [
                    {"zh": "一张页表把“逻辑页号 → 物理块号”映射起来，注意力算子先查页表、再去对应物理块把 K/V gather 回来，所以物理块放哪都行、无需连续", "en": "A page table maps 'logical page → physical block id'; the attention kernel consults the table, then gathers K/V from the corresponding physical blocks, so blocks can sit anywhere and need not be contiguous"},
                    {"zh": "硬件会在每步自动把散落的块搬到一起变连续", "en": "The hardware auto-defragments scattered blocks into a contiguous region every step"},
                    {"zh": "所有请求其实共用同一块物理显存，无所谓位置", "en": "All requests actually share one physical block, so position is irrelevant"},
                    {"zh": "页足够小，小到不可能不连续", "en": "Pages are so small that non-contiguity can't happen"},
                ],
                "answer": 0,
                "why": {
                    "zh": "分页的核心是“固定页 + 页表”这层间接映射，正如操作系统虚拟内存。注意力不按连续地址直取，而是先查页表拿到物理块号、再 gather，所以物理块爱放哪放哪——这正是消灭外部碎片、并允许按需增长的根本。硬件不会自动整理碎片；各请求的页通常是各自独立的（除非第 7 课的前缀共享显式让页表指向同一块）。",
                    "en": "Paging's core is the 'fixed pages + page table' indirection, just like OS virtual memory. Attention doesn't fetch by contiguous address; it consults the table for the physical block id, then gathers — so blocks can go anywhere, which is what kills external fragmentation and enables on-demand growth. Hardware does not auto-defragment; each request's pages are normally independent (unless Lesson 7's prefix sharing explicitly points page tables at the same block).",
                },
            },
            {
                "q": {
                    "zh": "分页为第 7 课的 RadixAttention 前缀共享铺了什么路？",
                    "en": "How does paging pave the way for Lesson 7's RadixAttention prefix sharing?",
                },
                "opts": [
                    {"zh": "一旦 KV 是固定大小、由页表索引的物理块，不同请求的页表就能指向同一批物理块——相同前缀（如同一段 system prompt）的 KV 因此可以只存一份、被多请求共享", "en": "Once KV is fixed-size, page-table-indexed physical blocks, different requests' page tables can point at the same physical blocks — so identical prefixes (e.g. a shared system prompt) need only one KV copy, shared across requests"},
                    {"zh": "分页把 KV 压缩了一半，所以省下的空间能放更多前缀", "en": "Paging halves KV size, so the saved space holds more prefixes"},
                    {"zh": "分页让每条请求都强制使用同一个固定前缀", "en": "Paging forces every request to use the same fixed prefix"},
                    {"zh": "分页把前缀缓存搬到了 CPU 内存里", "en": "Paging moves the prefix cache into CPU memory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "分页的页表间接层带来一个自由：多张页表可以指向同一物理块。若两条请求开头是逐字节相同的前缀，它们前几页的 KV 完全一样，与其各存一份，不如让两张页表都指向那同一批物理页——这就是 RadixAttention（第 7 课）前缀共享的物理基础。它不靠压缩、不强制统一前缀，也与是否搬到 CPU（HiCache 见第 29–32 课）无关。",
                    "en": "The page-table indirection grants one freedom: many page tables can point at the same physical block. If two requests share a byte-identical prefix, their first pages of KV are identical, so instead of two copies you let both page tables point at the same physical pages — the physical basis for RadixAttention (Lesson 7) prefix sharing. It relies on neither compression nor forcing a common prefix, and is unrelated to moving to CPU (HiCache, Lessons 29–32).",
                },
            },
        ],
        "open": [
            {
                "zh": "设 page_size=16、显存上限相当于能放 1000 个 token 的 KV。比较两种放法在“同时能跑多少条请求”上的差别：(a) 每条请求按最大长度 256 连续预留；(b) 分页按需分配，平均每条请求实际只生成 40 个 token。分别估算并发上限，并指出浪费分别发生在哪里。",
                "en": "Let page_size=16 and an HBM budget that holds KV for 1000 tokens. Compare the two layouts on 'how many requests can run at once': (a) each request reserves contiguously for max length 256; (b) paged on-demand allocation where each request actually generates ~40 tokens on average. Estimate the concurrency cap for each and pinpoint where the waste occurs.",
            },
            {
                "zh": "page_size 该选大还是选小？请从“内部碎片（每条请求末尾未填满的那一页）”与“页表长度 / 算子里的间接寻址开销”两端，说明选 1、16、32 各自的取舍；如果你的工作负载是大量很短的请求，你会倾向哪一端，为什么？",
                "en": "Should page_size be large or small? From two ends — 'internal fragmentation (each request's last unfilled page)' and 'page-table length / kernel indirection overhead' — explain the tradeoff of choosing 1, 16, or 32; if your workload is many very short requests, which end would you lean toward, and why?",
            },
        ],
    },
    "07-radixattention-and-prefix-caching.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在一次 match_prefix 命中后，被“复用、零重算”的到底是什么？",
                    "en": "After a match_prefix hit, what exactly gets 'reused with zero recompute'?",
                },
                "opts": [
                    {"zh": "命中的那段前缀对应的 KV 缓存（物理块索引）——注意力算子顺着树节点拿到这些已算好的 K/V，直接当作本请求的一部分，省掉了这段前缀的 prefill 计算", "en": "The KV cache (physical block indices) of the matched prefix — the attention kernel follows the tree nodes to those already-computed K/V and treats them as part of this request, skipping that prefix's prefill compute"},
                    {"zh": "模型权重，省去了重新加载权重的开销", "en": "The model weights, saving the cost of reloading weights"},
                    {"zh": "上一条请求生成的最终文本，直接拷给新请求当输出", "en": "The previous request's final text, copied over as the new request's output"},
                    {"zh": "采样用的随机数种子，保证两条请求结果一致", "en": "The sampling random seed, so both requests produce identical results"},
                ],
                "answer": 0,
                "why": {
                    "zh": "RadixAttention 复用的是相同前缀的 KV 缓存：树的边上挂着 token 串与对应的物理 KV 块，命中后这些 K/V 无需重算，注意力直接 gather 使用，从而省掉这段前缀的 prefill 算力（也省了重复存储）。它复用的不是权重、不是文本输出、更不是随机种子。",
                    "en": "RadixAttention reuses the matched prefix's KV cache: tree edges carry token runs and their physical KV blocks, so on a hit those K/V need no recompute and attention gathers them directly, saving that prefix's prefill compute (and duplicate storage). It is not the weights, not the text output, and not the random seed.",
                },
            },
            {
                "q": {
                    "zh": "为什么 RadixAttention 用一棵基数树来组织前缀缓存，而不是用一个“整串 prompt → KV”的哈希表？",
                    "en": "Why does RadixAttention organize the prefix cache as a radix tree instead of a hash map from 'whole prompt → KV'?",
                },
                "opts": [
                    {"zh": "因为前缀是层级化的：树支持最长前缀匹配、部分复用、共享祖先与廉价分裂边，而整串哈希只认完全相等——开头相同但结尾不同就会整条 miss，无法部分复用", "en": "Because prefixes are hierarchical: a tree supports longest-prefix match, partial reuse, shared ancestors, and cheap edge splits, whereas a whole-string hash only knows exact equality — same start but different end means a full miss with no partial reuse"},
                    {"zh": "因为哈希表在 GPU 上无法实现", "en": "Because hash maps can't be implemented on a GPU"},
                    {"zh": "因为基数树查找的时间复杂度永远是 O(1)，比哈希表快", "en": "Because radix-tree lookup is always O(1), faster than a hash map"},
                    {"zh": "因为树能把 KV 压缩得更小，省显存", "en": "Because a tree compresses KV smaller, saving HBM"},
                ],
                "answer": 0,
                "why": {
                    "zh": "关键在于前缀的层级结构：请求们可能共享 100 个 token，其中一部分又额外共享更多。基数树天然支持沿树下行的最长前缀匹配、逐 token 的部分复用、把公共前缀升格为共享父节点、并在分歧点廉价地分裂一条边。整串哈希只能判断“是否完全相等”，开头相同结尾不同便全部 miss。这与“能否在 GPU 实现”“复杂度”“压缩”都无关。",
                    "en": "The point is the hierarchy of prefixes: requests may share 100 tokens, and some of them more on top. A radix tree naturally supports longest-prefix match down the tree, token-by-token partial reuse, promoting a common prefix to a shared parent, and cheaply splitting an edge at the fork. A whole-string hash only tests exact equality and misses entirely when starts match but ends differ. It has nothing to do with GPU feasibility, complexity, or compression.",
                },
            },
            {
                "q": {
                    "zh": "缓存满了要驱逐，为什么 RadixAttention 的 LRU 驱逐必须尊重引用计数？",
                    "en": "When the cache is full and must evict, why must RadixAttention's LRU eviction respect reference counts?",
                },
                "opts": [
                    {"zh": "因为正在运行的请求会给它正在使用的前缀路径加锁（引用计数 > 0）；若驱逐了它正靠着 decode 的那段前缀 KV，这条请求会当场崩溃，所以只驱逐没有任何在跑请求引用的叶子", "en": "Because a running request locks the prefix path it is using (refcount > 0); evicting the prefix KV it is decoding against would crash that request, so eviction only touches leaves no running request references"},
                    {"zh": "因为引用计数能让缓存命中率自动翻倍", "en": "Because reference counts automatically double the cache hit rate"},
                    {"zh": "因为不数引用计数，LRU 就无法判断哪个叶子最久未访问", "en": "Because without counting references, LRU can't tell which leaf is least-recently-used"},
                    {"zh": "因为引用计数决定了 page_size 的大小", "en": "Because reference counts determine the page_size"},
                ],
                "answer": 0,
                "why": {
                    "zh": "驱逐会释放节点的物理 KV 块。但一条正在 decode 的请求，时刻依赖它前缀路径上的那批 KV；如果这些 KV 被驱逐，注意力就读到无效/被覆盖的数据，请求直接崩。引用计数（lock_ref）正是为此：在跑的请求给路径加锁，引用计数 > 0 的节点绝不被驱逐，LRU 只敢动无人引用的叶子。它与命中率翻倍、LRU 的时间判定、page_size 都无关。",
                    "en": "Eviction frees a node's physical KV blocks. But a request currently decoding depends every step on the KV along its prefix path; if that KV is evicted, attention reads invalid/overwritten data and the request crashes. Reference counting (lock_ref) exists for exactly this: running requests lock the path, nodes with refcount > 0 are never evicted, and LRU only touches unreferenced leaves. It is unrelated to doubling hit rate, LRU's recency test, or page_size.",
                },
            },
        ],
        "open": [
            {
                "zh": "用一个 5 轮的客服对话说明 RadixAttention 如何“跨轮自动复用”：写出第 3 轮请求的 token 序列大致由哪几部分组成，指出其中哪一段会被 match_prefix 命中、引擎真正要新算的是什么，并解释为什么这一切不需要用户给任何提示。",
                "en": "Using a 5-turn customer-support chat, explain how RadixAttention 'auto-reuses across turns': sketch what the turn-3 request's token sequence is composed of, point out which part match_prefix hits and what the engine actually computes anew, and explain why none of this needs any user hint.",
            },
            {
                "zh": "树里已存着前缀 “You are a helpful assistant. Translate the following” 这一条边；现在来了一条新请求，它只共享到 “You are a helpful assistant. ” 就发散成 “Summarize …”。请描述 match_prefix/insert 在这一步会对树做什么（包含“分裂边”），并说明分裂后哪个节点成为共享祖先、各挂哪一支。",
                "en": "The tree already holds an edge for the prefix 'You are a helpful assistant. Translate the following'; now a new request arrives that only shares up to 'You are a helpful assistant. ' before diverging into 'Summarize …'. Describe what match_prefix/insert does to the tree at this step (including the edge split), and state which node becomes the shared ancestor and what branches hang off it.",
            },
        ],
    },
    "08-throughput-vs-latency.html": {
        "mcq": [
            {
                "q": {
                    "zh": "一条请求“从到达到吐出第一个 token”的时间（TTFT）主要被什么支配？",
                    "en": "What mainly dominates a request's TTFT (time from arrival to the first token)?",
                },
                "opts": [
                    {"zh": "prefill 的计算量，以及它在等待队列里的排队时间（前面还有多少活）", "en": "The prefill compute, plus its queueing time in the waiting queue (how much work is ahead of it)"},
                    {"zh": "decode 阶段的批大小——批越大首 token 越快", "en": "The decode-stage batch size — bigger batch means a faster first token"},
                    {"zh": "模型权重的总字节数，与是否排队无关", "en": "The total bytes of model weights, independent of any queueing"},
                    {"zh": "采样温度（temperature）的取值", "en": "The value of the sampling temperature"},
                ],
                "answer": 0,
                "why": {
                    "zh": "TTFT 是“首 token 延迟”：请求必须先被调度进来、完成它 prompt 的 prefill，才能产出第一个 token。所以它由两件事决定——prefill 本身的计算量，以及在等待队列里排了多久（前面活越多、排队越长，TTFT 越大）。decode 批大小支配的是 token 间延迟 ITL，不是 TTFT；权重字节数和温度都不是主因。",
                    "en": "TTFT is the time to first token: a request must be scheduled in and finish its prompt's prefill before emitting token one. So it is set by two things — the prefill compute itself, and how long it queued (more work ahead ⇒ longer queue ⇒ larger TTFT). Decode batch size governs ITL, not TTFT; weight bytes and temperature are not the drivers.",
                },
            },
            {
                "q": {
                    "zh": "为什么把批（同时在跑的请求数）调大，能提升总吞吐，却又会拉高单请求延迟？",
                    "en": "Why does enlarging the batch (concurrent requests) raise total throughput yet also increase per-request latency?",
                },
                "opts": [
                    {"zh": "因为 decode 访存密集，读一次权重对 1 条和 100 条代价几乎一样，大批把这笔固定开销摊薄（吞吐↑）；但同一步要为更多请求一起算、新请求排在更多活后面（每步更久、排队更长 ⇒ 延迟↑）", "en": "Because decode is memory-bound: reading weights once costs nearly the same for 1 or 100 requests, so a big batch amortizes that fixed cost (tput↑); but one step now computes for more requests and newcomers queue behind more work (heavier step, longer queue ⇒ latency↑)"},
                    {"zh": "因为大批会自动降低模型精度，从而既快又省", "en": "Because a big batch automatically lowers model precision, making it both faster and cheaper"},
                    {"zh": "因为大批让每条请求各自独占一张 GPU", "en": "Because a big batch gives each request its own dedicated GPU"},
                    {"zh": "因为吞吐和延迟其实互不相关，调大批两者都会变好", "en": "Because throughput and latency are unrelated, so a bigger batch improves both"},
                ],
                "answer": 0,
                "why": {
                    "zh": "这正是核心张力：decode 是访存密集的，权重读取的固定开销与批里请求数几乎无关，批越大摊得越薄、GPU 越饱和 ⇒ 吞吐↑。但同一个 decode 步要为更多请求一起前向，单步更久（ITL↑），新请求也要排在更多活后面（TTFT↑），于是单请求延迟上升。没有免费午餐，只能按 SLA 选工作点。其余选项都与机制不符。",
                    "en": "This is the core tension: decode is memory-bound, so the fixed cost of reading weights barely depends on how many requests share the batch; a bigger batch amortizes it thinner and saturates the GPU ⇒ tput↑. But one decode step now forwards more requests (longer step ⇒ ITL↑) and newcomers wait behind more work (TTFT↑), so per-request latency rises. No free lunch — pick the point by SLA. The other options misstate the mechanism.",
                },
            },
            {
                "q": {
                    "zh": "在 SGLang 里，调大 mem_fraction_static 与调大 max_running_requests，主要各自换来什么？",
                    "en": "In SGLang, what do raising mem_fraction_static and raising max_running_requests each mainly buy you?",
                },
                "opts": [
                    {"zh": "mem_fraction_static 给 KV 池更多显存 ⇒ 能装下更多 token ⇒ 抬高并发的天花板；max_running_requests 直接提高并发上限 ⇒ 批更大。两者都把工作点推向高吞吐、但也推高延迟，且 KV 占多了留给激活/CUDA Graph 的显存就少", "en": "mem_fraction_static gives the KV pool more HBM ⇒ more tokens fit ⇒ raises the concurrency ceiling; max_running_requests directly lifts the concurrency cap ⇒ bigger batch. Both push the point toward high throughput (and higher latency), and more KV leaves less HBM for activations/CUDA Graphs"},
                    {"zh": "两者都只影响日志详细程度，与性能无关", "en": "Both only affect logging verbosity, unrelated to performance"},
                    {"zh": "mem_fraction_static 控制学习率，max_running_requests 控制 batch 的精度", "en": "mem_fraction_static controls the learning rate, max_running_requests controls batch precision"},
                    {"zh": "两者都会降低吞吐以换取更低延迟", "en": "Both reduce throughput in exchange for lower latency"},
                ],
                "answer": 0,
                "why": {
                    "zh": "mem_fraction_static 划走多少 HBM 给 KV 缓存池，就决定能同时容纳多少 token，也就抬高了并发的天花板（更多请求 ⇒ 批更大 ⇒ 吞吐↑）；但显存零和，给 KV 多了，留给模型激活和 CUDA Graph 就少。max_running_requests 则是并发上限的硬阀门，直接决定批能做多大。两者都把工作点往高吞吐推，同时推高延迟——这正是吞吐-延迟曲线上的选点动作。",
                    "en": "How much HBM mem_fraction_static carves for the KV cache pool sets how many tokens fit at once, raising the concurrency ceiling (more requests ⇒ bigger batch ⇒ tput↑); but HBM is zero-sum, so more for KV leaves less for activations and CUDA Graphs. max_running_requests is the hard cap on concurrency, directly bounding batch size. Both push the operating point toward high throughput while raising latency — exactly the act of picking a point on the curve.",
                },
            },
        ],
        "open": [
            {
                "zh": "用一句话分别定义 TTFT、ITL/TPOT、throughput、goodput，并指出每个指标“主要被什么支配”。然后解释：为什么一个吞吐很高的系统，goodput 仍可能很低？给出一个会发生这种情况的具体场景。",
                "en": "Define TTFT, ITL/TPOT, throughput, and goodput each in one sentence, and state what mainly dominates each. Then explain: why can a system with very high throughput still have low goodput? Give a concrete scenario where this happens.",
            },
            {
                "zh": "你的服务有“TTFT < 300ms”的 SLA，但当前为了冲吞吐把 max_running_requests 调得很大，导致部分请求 TTFT 超标。请描述你会怎样在吞吐-延迟曲线上重新选点（涉及 max_running_requests、mem_fraction_static、chunked_prefill_size 中至少两个），并说明分块预填充（第 22 课）在“长 prompt 引发延迟尖峰”时如何帮你稳住 SLA。",
                "en": "Your service has an SLA of 'TTFT < 300ms', but to chase throughput you set max_running_requests very large, causing some requests to blow past the TTFT bound. Describe how you would re-pick a point on the throughput-latency curve (touching at least two of max_running_requests, mem_fraction_static, chunked_prefill_size), and explain how chunked prefill (Lesson 22) helps hold the SLA when a long prompt triggers a latency spike.",
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
