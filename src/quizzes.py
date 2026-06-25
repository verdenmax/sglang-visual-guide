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
    "09-structured-generation-language.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 SGLang 程序里，<code>gen(name, …)</code> 和 <code>select(name, choices=[…])</code> 的本质区别是什么？",
                    "en": "In an SGLang program, what is the essential difference between <code>gen(name, …)</code> and <code>select(name, choices=[…])</code>?",
                },
                "opts": [
                    {"zh": "gen 是“自由填空”，模型生成任意文本填入命名槽；select 是“受限多选”，解码被限制在给定选项内、只能挑一个——本质是带约束的 gen", "en": "gen is a 'free fill-in' where the model writes any text into a named slot; select is a 'constrained choice' whose decoding is restricted to the given options (pick one) — essentially a constrained gen"},
                    {"zh": "gen 只能用于 system 消息，select 只能用于 user 消息", "en": "gen works only in system messages, select only in user messages"},
                    {"zh": "gen 同步执行并阻塞，select 永远异步且不返回结果", "en": "gen runs synchronously and blocks, select is always async and never returns a result"},
                    {"zh": "两者完全等价，select 只是 gen 的别名", "en": "They are fully equivalent; select is just an alias for gen"},
                ],
                "answer": 0,
                "why": {
                    "zh": "gen 让模型自由生成填入命名槽（用 s[\"name\"] 取出）；select 把解码限制在 choices 内，保证输出一定是合法选项之一。源码里 gen 一旦收到 choices 就直接返回 SglSelect，说明 select 本质上就是“带约束的 gen”——这正是约束解码的入口。其余选项都与机制不符。",
                    "en": "gen lets the model generate freely into a named slot (read via s[\"name\"]); select restricts decoding to the choices, guaranteeing the output is one of the valid options. In the source, a gen that receives choices returns an SglSelect, so select is essentially a 'constrained gen' — the entry point to constrained decoding. The other options misstate the mechanism.",
                },
            },
            {
                "q": {
                    "zh": "为什么用 SGLang 这样的 DSL 写多步 LLM 程序，通常优于手搓裸 HTTP/OpenAI API 调用？",
                    "en": "Why is writing multi-step LLM programs in a DSL like SGLang usually better than hand-rolling raw HTTP/OpenAI API calls?",
                },
                "opts": [
                    {"zh": "因为控制流与结构都写在普通 Python 里（清晰可调试）、约束解码保证输出可靠、且运行时能自动复用共享前缀的 KV——这些在裸 API 里都得自己手工实现", "en": "Because control flow and structure live in plain Python (clear, debuggable), constrained decoding makes outputs reliable, and the runtime auto-reuses KV across shared prefixes — all of which you'd have to hand-build with a raw API"},
                    {"zh": "因为 DSL 会自动把模型权重压缩一半，从而更省显存", "en": "Because the DSL automatically halves the model weights, saving HBM"},
                    {"zh": "因为 DSL 让每条请求独占一张 GPU，从而更快", "en": "Because the DSL gives each request its own dedicated GPU, making it faster"},
                    {"zh": "因为裸 API 无法生成文本，只能用 DSL 调用模型", "en": "Because raw APIs cannot generate text; only a DSL can call the model"},
                ],
                "answer": 0,
                "why": {
                    "zh": "DSL 的三重红利：① 结构与控制流都在 Python，可读可调试；② select/regex/json 等约束解码让输出合法可解析；③ 框架看见结构后，能自动复用共享前缀的 KV（第 7 课 RadixAttention）、fork/join 共享父前缀（第 11 课）。这些在裸 API 里全靠自己拼字符串、解析、管历史、手动缓存。其余选项与事实不符。",
                    "en": "The DSL buys three things: ① structure and control flow in Python (readable, debuggable); ② constrained decoding (select/regex/json) for valid, parseable output; ③ once the framework sees the structure, it auto-reuses shared-prefix KV (Lesson 7, RadixAttention) and shares parent prefixes across fork/join (Lesson 11). With a raw API you do all the string-building, parsing, history management, and caching yourself. The others are false.",
                },
            },
            {
                "q": {
                    "zh": "“把程序写成结构化的 SGLang 函数”是如何让运行时获得自动前缀复用机会的？",
                    "en": "How does 'writing the program as a structured SGLang function' give the runtime its opportunity for automatic prefix reuse?",
                },
                "opts": [
                    {"zh": "因为结构化程序让框架在执行前就看清整体结构——哪些请求/分支共享同一段前缀提示——于是运行时只算一次前缀、复用其 KV；裸 API 那种“一发一收、互相看不见”的调用拿不到这种线索", "en": "Because a structured program lets the framework see the whole structure before execution — which requests/branches share the same prefix prompt — so the runtime computes the prefix once and reuses its KV; the raw-API 'fire one, read one, blind to each other' style can't expose such hints"},
                    {"zh": "因为 SGLang 会把所有请求的输出拼接成一条超长序列再一起算", "en": "Because SGLang concatenates all requests' outputs into one giant sequence and computes them together"},
                    {"zh": "因为结构化程序会自动降低采样温度，从而命中缓存", "en": "Because a structured program automatically lowers the sampling temperature to hit the cache"},
                    {"zh": "因为前缀复用只和磁盘缓存有关，与程序怎么写无关", "en": "Because prefix reuse only concerns disk caching and is unrelated to how the program is written"},
                ],
                "answer": 0,
                "why": {
                    "zh": "gen 不立即执行，而是构造 IR 节点，组成一棵“待执行意图树”。正因是声明式的树，框架在执行前就能看见全局结构：哪些请求共享同一 system 提示或少样本前缀、哪些 fork 分支共享父节点前缀。于是运行时只对共享前缀算一次、复用其 KV（第 7 课）。这正是结构化写法相对裸 API 的独有红利。其余选项都与机制不符。",
                    "en": "gen doesn't execute immediately; it builds an IR node, and the nodes form a declarative 'tree of intent.' Because it's a tree, the framework sees the global structure before execution: which requests share a system prompt or few-shot prefix, which fork branches share the parent prefix. So the runtime computes a shared prefix once and reuses its KV (Lesson 7). That is the structured style's unique edge over a raw API. The others misstate the mechanism.",
                },
            },
        ],
        "open": [
            {
                "zh": "用 SGLang 写一个被 <code>@sgl.function</code> 装饰、首参为 <code>s</code> 的两步程序：第一步用 <code>gen</code> 让模型先写推理，第二步先用 <code>s[\"…\"]</code> 取出推理、再用 <code>select</code> 在固定标签里给出最终分类。说明每个 gen/select 各对应一次什么样的模型调用，以及为什么这两步能共享同一段前缀的 KV。",
                "en": "Write a two-step SGLang program decorated with <code>@sgl.function</code> whose first arg is <code>s</code>: step one uses <code>gen</code> to let the model write its reasoning, step two reads it back with <code>s[\"…\"]</code> and then uses <code>select</code> to emit a final classification from fixed labels. Explain what kind of model call each gen/select corresponds to, and why the two steps can share the same prefix's KV.",
            },
            {
                "zh": "你的团队现在用裸 OpenAI API 手写一个“先抽取要点、再据要点生成摘要”的两段式流程，受困于格式不稳、历史拼接易错、相同 system 提示被反复重算。请论证迁移到 SGLang DSL 能在哪三个方面带来改善（控制流/约束解码/自动前缀复用），并各举一个具体例子。",
                "en": "Your team currently hand-writes a two-stage 'extract key points, then summarize from them' flow with the raw OpenAI API, struggling with unstable formats, error-prone history concatenation, and the same system prompt being recomputed repeatedly. Argue how migrating to the SGLang DSL improves things along three axes (control flow / constrained decoding / automatic prefix reuse), with one concrete example each.",
            },
        ],
    },
    "10-interpreter-and-tracer.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 SGLang 里，<code>解释（interpret）</code>和<code>追踪（trace）</code>这两种模式最本质的区别是什么？",
                    "en": "In SGLang, what is the most essential difference between <code>interpret</code> and <code>trace</code>?",
                },
                "opts": [
                    {"zh": "解释由 StreamExecutor 真正逐步调用后端、生成 token 并改变状态；追踪用假后端符号化走一遍、不调用模型，只为看清结构并抽出静态前缀", "en": "Interpret really drives the backend step by step via StreamExecutor, generates tokens, and mutates state; trace walks symbolically with a dummy backend, never calls the model, and only discovers structure and extracts the static prefix"},
                    {"zh": "解释只能跑在 GPU 上，追踪只能跑在 CPU 上", "en": "Interpret only runs on GPU, trace only runs on CPU"},
                    {"zh": "两者都会调用模型，区别只是追踪更快", "en": "Both call the model; trace is merely faster"},
                    {"zh": "解释用于训练，追踪用于评测", "en": "Interpret is for training, trace is for evaluation"},
                ],
                "answer": 0,
                "why": {
                    "zh": "解释是常规执行路径：StreamExecutor 逐表达式 _execute，gen/select 真的打后端，把结果接到 text_、存进 variables，状态一步步生长。追踪是编译/分析路径：trace_program 用假后端符号化走一遍，完全不碰模型，只发现结构并抽出开头那段恒定的静态前缀。其余选项都与机制不符。",
                    "en": "Interpret is the normal execution path: StreamExecutor _executes per expression, gen/select really hits the backend, appends to text_ and stores into variables, growing state step by step. Trace is the compile/analysis path: trace_program walks symbolically with a dummy backend, never touching the model, only discovering structure and extracting the constant leading static prefix. The others misstate the mechanism.",
                },
            },
            {
                "q": {
                    "zh": "追踪为什么要“不调用模型”地走一遍程序？它最有价值的产物是什么？",
                    "en": "Why does trace walk the program 'without calling the model'? What is its most valuable product?",
                },
                "opts": [
                    {"zh": "为了在不花算力的前提下看清程序结构，并抽出“静态前缀”（开头恒定、直到第一个依赖模型输出处为止的那段提示），供运行时只算一次、复用其 KV", "en": "To see the program's structure without spending compute, and to extract the 'static prefix' (the constant leading prompt up to the first place that depends on a model output), so the runtime can compute it once and reuse its KV"},
                    {"zh": "为了把模型权重压缩，从而省显存", "en": "To compress model weights and save HBM"},
                    {"zh": "为了提前把所有 gen 的答案缓存到磁盘", "en": "To pre-cache every gen's answer to disk"},
                    {"zh": "为了降低采样温度让输出更确定", "en": "To lower sampling temperature for more deterministic output"},
                ],
                "answer": 0,
                "why": {
                    "zh": "追踪的目的是“先看懂形状”：用假后端符号化走一遍，遇到依赖模型结果的节点就停，于是能在零算力下发现结构、抽出开头那段恒定的静态前缀。运行时据此只对共享前缀算一次注意力、复用其 KV（第 7 课 RadixAttention），还能据此编译/优化。这与压权重、缓存答案、调温度都无关。",
                    "en": "Trace aims to 'understand the shape first': a symbolic walk with a dummy backend that stops at any node depending on a model result, so it discovers structure and extracts the constant leading static prefix at zero compute. The runtime then computes attention for that shared prefix once and reuses its KV (Lesson 7, RadixAttention), and can compile/optimize from it. It has nothing to do with compressing weights, caching answers, or temperature.",
                },
            },
            {
                "q": {
                    "zh": "解释模式下，状态对象 <code>s</code>（ProgramState）里到底存了哪些东西？<code>s[\"name\"]</code> 取的是什么？",
                    "en": "In interpret mode, what does the state object <code>s</code> (ProgramState) actually hold, and what does <code>s[\"name\"]</code> read?",
                },
                "opts": [
                    {"zh": "它累积完整文本 text_（即前缀）、role 消息 messages_、以及按名字存放的命名结果 variables；s[\"name\"] 取的就是 variables[\"name\"]——某个 gen/select 的输出", "en": "It accumulates the full text text_ (the prefix), role messages messages_, and named results stored in variables; s[\"name\"] reads variables[\"name\"] — the output of some gen/select"},
                    {"zh": "它只存一个布尔标志，表示程序是否结束", "en": "It only stores a boolean flag for whether the program finished"},
                    {"zh": "它存的是模型权重的指针", "en": "It stores pointers to the model weights"},
                    {"zh": "它把所有内容编码成单个整数 token id", "en": "It encodes everything into a single integer token id"},
                ],
                "answer": 0,
                "why": {
                    "zh": "StreamExecutor 就是 s 背后的“账本”：text_ 是不断生长的完整文本（也正是后续请求的前缀），messages_ 记 role 消息，variables 按名字存每个 gen/select 的结果。所以你在 Python 里写 s[\"reason\"]，取的就是 variables[\"reason\"]。其余选项都与源码字段不符。",
                    "en": "StreamExecutor is the 'ledger' behind s: text_ is the ever-growing full text (also the prefix for later requests), messages_ logs role messages, and variables stores each gen/select result by name. So s[\"reason\"] in Python reads variables[\"reason\"]. The others don't match the source fields.",
                },
            },
        ],
        "open": [
            {
                "zh": "用自己的话描述：当你调用一个含两个 <code>gen</code> 的 <code>@sgl.function</code> 时，StreamExecutor 从被调用到产出最终文本，依次发生了什么？请点明每个 gen 如何打后端、结果如何写回 <code>text_</code> 与 <code>variables</code>，以及 <code>s[\"name\"]</code> 是从哪里取值的。",
                "en": "In your own words, describe what StreamExecutor does — from being called to producing the final text — when you invoke an <code>@sgl.function</code> with two <code>gen</code>s. Point out how each gen hits the backend, how results are written back into <code>text_</code> and <code>variables</code>, and where <code>s[\"name\"]</code> reads from.",
            },
            {
                "zh": "假设你的服务里有一万条请求都以同一段很长的系统提示 + few-shot 模板开头。请论证：为什么先对程序做一次“追踪”抽出静态前缀，会对运行时的吞吐有帮助？它和第 7 课的 RadixAttention 前缀复用是怎么衔接的？追踪到哪里会停下、为什么？",
                "en": "Suppose your service has ten thousand requests all starting with the same long system prompt + few-shot template. Argue why running a 'trace' once to extract the static prefix helps the runtime's throughput. How does it connect to Lesson 7's RadixAttention prefix reuse? Where does tracing stop, and why?",
            },
        ],
    },
    "11-fork-join-and-prefix-sharing.html": {
        "mcq": [
            {
                "q": {
                    "zh": "当你对状态调用 <code>s.fork(n)</code> 时，这 n 个分支之间<strong>共享</strong>的到底是什么？",
                    "en": "When you call <code>s.fork(n)</code> on a state, what exactly do the n branches <strong>share</strong>?",
                },
                "opts": [
                    {"zh": "它们都继承到 fork 那一刻为止<strong>已建好的同一段提示前缀</strong>（system + 少样本 + 问题等），之后各自独立生成", "en": "They all inherit the <strong>same prompt prefix built up to the fork point</strong> (system + few-shot + question, etc.), then generate independently afterward"},
                    {"zh": "它们共享同一个随机种子，因此必然生成完全相同的文本", "en": "They share one random seed, so they would necessarily produce identical text"},
                    {"zh": "它们共享一块独占的 GPU，其他请求不能用", "en": "They share a dedicated GPU that no other request may use"},
                    {"zh": "它们什么都不共享，fork 等价于从零开始发 n 个全新请求", "en": "They share nothing; fork is equivalent to firing n brand-new requests from scratch"},
                ],
                "answer": 0,
                "why": {
                    "zh": "fork 从当前状态克隆出 n 个子状态，每个都带着<strong>到此为止建好的同一段前缀</strong>，然后各走各路独立生成。正因为前缀相同，RadixAttention（第 7 课）才能只算一次、给所有分支复用。分支用不同随机性可得不同结果，并非必然相同；也不涉及独占 GPU；更不是从零开始。",
                    "en": "fork clones n sub-states from the current state, each carrying <strong>the same prefix built so far</strong>, after which they generate independently. Precisely because the prefix is identical, RadixAttention (Lesson 7) computes it once and reuses it across branches. Branches with different randomness yield different results (not necessarily identical); no dedicated GPU is involved; and it is not from scratch.",
                },
            },
            {
                "q": {
                    "zh": "本课反复强调“fork 和 RadixAttention 是一个想法的两半”。这句话最准确的含义是？",
                    "en": "This lesson repeatedly stresses that 'fork and RadixAttention are two halves of one idea.' What does that most precisely mean?",
                },
                "opts": [
                    {"zh": "fork 是“声明端”——在程序里说出“这些分支开头相同”；RadixAttention 是“兑现端”——把那段共享前缀的 KV 只算一次、让所有分支复用。前端声明结构，运行时把结构变现成省下的算力", "en": "fork is the 'declaration side' — it states in the program that 'these branches share an opening'; RadixAttention is the 'redemption side' — it computes that shared prefix's KV once and reuses it across branches. The front-end declares structure, the runtime cashes it in as saved compute"},
                    {"zh": "fork 会把 RadixAttention 关掉，改用磁盘缓存", "en": "fork turns RadixAttention off and switches to disk caching"},
                    {"zh": "两者是互斥的优化，开了 fork 就不能用前缀复用", "en": "They are mutually exclusive optimizations; enabling fork disables prefix reuse"},
                    {"zh": "fork 只影响前端可读性，对运行时算力毫无影响", "en": "fork only affects front-end readability and has no effect on runtime compute"},
                ],
                "answer": 0,
                "why": {
                    "zh": "fork 在 DSL 里明示“这 n 个分支共享同一段前缀”，等于零猜测地把结构告诉运行时；RadixAttention 收到这个结构信号，就把共享前缀的注意力只算一次、所有分支共用同一份 KV。一个负责说出结构，一个负责把结构变现。二者协同而非互斥，也不会关掉对方或退化成磁盘缓存。",
                    "en": "fork explicitly states in the DSL that 'these n branches share one prefix,' telling the runtime the structure with zero guessing; RadixAttention takes that signal and computes the shared prefix's attention once, with all branches sharing one KV. One states the structure, the other cashes it in. They cooperate rather than exclude, and neither disables the other nor degrades to disk caching.",
                },
            },
            {
                "q": {
                    "zh": "从一段<strong>很长的共享前缀</strong> fork 出 8 个分支，相比用 for 循环串行跑 8 次完整提示，代价差别大致是？",
                    "en": "Forking 8 branches off a <strong>long shared prefix</strong>, versus a for loop running 8 full prompts serially — roughly how does the cost differ?",
                },
                "opts": [
                    {"zh": "fork ≈ “1 段前缀 + 8 段短后缀”（前缀只算一次）；串行循环 ≈ “8 × 完整提示”（前缀被重算 8 遍）。前缀越长，fork 省得越多，且无依赖分支还能并发", "en": "fork ≈ '1 prefix + 8 short suffixes' (prefix computed once); the serial loop ≈ '8 × full prompt' (prefix recomputed 8 times). The longer the prefix, the more fork saves, and independent branches can also run concurrently"},
                    {"zh": "两者代价完全相同，fork 只是写法更短", "en": "The costs are identical; fork is merely shorter to write"},
                    {"zh": "fork 更贵，因为要额外存 8 份前缀 KV", "en": "fork is more expensive because it must store 8 copies of the prefix KV"},
                    {"zh": "串行循环更省，因为它从不并发", "en": "The serial loop is cheaper because it never runs concurrently"},
                ],
                "answer": 0,
                "why": {
                    "zh": "串行循环里 8 次请求开头那段长前缀一模一样，却被从头重算 8 遍，白烧 7 份前缀算力；fork 把“共享前缀 + 8 条后缀”一次声明清楚，运行时只算一份前缀 KV、8 个分支共用，各自只补一段短后缀，无依赖时还能并发。fork 并不会存 8 份前缀（恰恰只存一份），代价也绝不相同。",
                    "en": "In the serial loop, the same long prefix opens all 8 requests yet is recomputed 8 times, wasting 7 prefixes of compute; fork declares 'shared prefix + 8 suffixes' once, so the runtime computes one prefix KV shared by all 8 branches, each adding only a short suffix, with concurrency when independent. fork does not store 8 prefixes (it stores exactly one), and the costs are far from identical.",
                },
            },
        ],
        "open": [
            {
                "zh": "用 SGLang 写一段 best-of-n 的伪代码：先往 <code>s</code> 拼好一段较长的系统提示 + 问题，再 <code>forks = s.fork(8)</code> 让 8 个分支各自 <code>gen(\"answer\")</code>，最后 <code>forks.join()</code> 收回 8 个答案。请说明：这 8 个分支共享的是哪一段、各自独立算的是哪一段，以及运行时（RadixAttention，第 7 课）据此把什么只算了一次。",
                "en": "Write best-of-n pseudocode in SGLang: build a longish system prompt + question into <code>s</code>, then <code>forks = s.fork(8)</code> so 8 branches each do <code>gen(\"answer\")</code>, finally <code>forks.join()</code> to gather 8 answers. Explain which segment the 8 branches share, which segment each computes independently, and what the runtime (RadixAttention, Lesson 7) thereby computes only once.",
            },
            {
                "zh": "有人说“fork 只是个写起来方便的语法糖，去掉它用 for 循环效果一样”。请结合“fork 是声明端、RadixAttention 是兑现端”这层关系反驳：为什么把分叉结构<strong>显式声明</strong>出来，对运行时的前缀复用和并发都更有利？<code>join</code> 在其中又承担了什么角色？",
                "en": "Someone claims 'fork is just convenient syntactic sugar; drop it and a for loop does the same.' Rebut this using the 'fork is the declaration side, RadixAttention the redemption side' relationship: why does <strong>explicitly declaring</strong> the branch structure benefit both prefix reuse and concurrency at the runtime? And what role does <code>join</code> play in all this?",
            },
        ],
    },
    "12-backends-and-openai-compat.html": {
        "mcq": [
            {
                "q": {
                    "zh": "同一个 SGLang 前端程序，分别用 <code>RuntimeEndpoint</code>（本地运行时）和 <code>OpenAI</code>（托管）两个后端跑。哪个后端能真正给你 RadixAttention 前缀缓存与 fork 分支共享（第 7、11 课）？为什么？",
                    "en": "Run the same SGLang front-end program on two backends: <code>RuntimeEndpoint</code> (local runtime) and <code>OpenAI</code> (hosted). Which backend actually gives you RadixAttention prefix caching and fork branch sharing (Lessons 7, 11), and why?",
                },
                "opts": [
                    {"zh": "<strong>RuntimeEndpoint</strong>：它把请求打向你自己的本地 SGLang 服务器（白盒），运行时能看见前缀结构、把共享前缀只算一次；托管模型是黑盒，缓存与否由服务商决定，你无从声明也无法依赖", "en": "<strong>RuntimeEndpoint</strong>: it sends requests to your own local SGLang server (a white box), so the runtime sees the prefix structure and computes a shared prefix once; a hosted model is a black box where caching is the provider's call, which you can neither declare nor rely on"},
                    {"zh": "OpenAI 后端，因为闭源模型一定比本地模型更会缓存", "en": "The OpenAI backend, because closed models always cache better than local ones"},
                    {"zh": "两个后端完全一样，前缀缓存与后端无关", "en": "Both backends are identical; prefix caching has nothing to do with the backend"},
                    {"zh": "都不行，前缀缓存只能手动在提示里实现", "en": "Neither; prefix caching can only be done manually in the prompt"},
                ],
                "answer": 0,
                "why": {
                    "zh": "RadixAttention 与 fork 共享是<strong>运行时的本事</strong>，只有当请求真正落到你能掌控的本地 SGLang 运行时（RuntimeEndpoint 打向 /generate）时才生效。托管 API 把模型藏在黑盒后，你无法声明前缀结构、无法保证缓存命中，因此失去这份确定性红利。两个后端在程序里写法相同，但能力天差地别。",
                    "en": "RadixAttention and fork sharing are <strong>runtime capabilities</strong>, effective only when the request actually lands on a local SGLang runtime you control (RuntimeEndpoint posting to /generate). A hosted API hides the model behind a black box, so you cannot declare the prefix structure or guarantee cache hits, losing that deterministic benefit. The two backends look identical in code but differ vastly in capability.",
                },
            },
            {
                "q": {
                    "zh": "“OpenAI 兼容”涉及两个相反方向。下面哪一项正确区分了它们？",
                    "en": "'OpenAI-compatible' involves two opposite directions. Which option correctly distinguishes them?",
                },
                "opts": [
                    {"zh": "方向①“SGLang 程序 → OpenAI 后端”：你的程序当客户端去调 OpenAI 托管模型；方向②“OpenAI 客户端 → SGLang 服务器”：SGLang 服务器本身兼容 OpenAI API，别人的 OpenAI 客户端改个 base_url 就能接入你的私有部署", "en": "Direction ① 'SGLang program → OpenAI backend': your program is the client calling OpenAI's hosted models; direction ② 'OpenAI client → SGLang server': the SGLang server is itself OpenAI-compatible, so someone else's OpenAI client plugs into your private deployment by just changing base_url"},
                    {"zh": "两个方向是一回事，都是指 SGLang 调用 OpenAI", "en": "The two directions are the same thing — both mean SGLang calling OpenAI"},
                    {"zh": "只有 OpenAI 能当服务端，SGLang 永远是客户端", "en": "Only OpenAI can be the server; SGLang is always the client"},
                    {"zh": "“兼容”仅指数据格式相同，与谁调用谁无关", "en": "'Compatible' refers only to identical data formats, independent of who calls whom"},
                ],
                "answer": 0,
                "why": {
                    "zh": "两个方向恰好相反：方向①里 SGLang 是客户端、OpenAI 是服务端（用 OpenAI 后端消费托管模型）；方向②里 SGLang 是服务端、OpenAI 客户端是客户端（SGLang 服务器对外暴露标准 OpenAI 接口，任何 OpenAI SDK/LangChain 改 base_url 即可接入）。一个是消费别人的模型，一个是把自己的服务接入生态。",
                    "en": "The directions are exactly opposite: in ① SGLang is the client and OpenAI the server (using the OpenAI backend to consume hosted models); in ② SGLang is the server and an OpenAI client is the client (the SGLang server exposes a standard OpenAI interface, so any OpenAI SDK/LangChain plugs in by changing base_url). One consumes others' models; the other plugs your own service into the ecosystem.",
                },
            },
            {
                "q": {
                    "zh": "你决定把生产服务从本地 <code>RuntimeEndpoint</code> 换成托管 <code>OpenAI</code> 后端。除了一行 <code>set_default_backend</code>，你主要<strong>交易（牺牲）</strong>了什么？",
                    "en": "You decide to switch your production service from the local <code>RuntimeEndpoint</code> to the hosted <code>OpenAI</code> backend. Beyond a one-line <code>set_default_backend</code>, what do you mainly <strong>trade away</strong>?",
                },
                "opts": [
                    {"zh": "换来可移植性与免自建 GPU，但牺牲了前缀缓存与 fork 共享的确定性红利、严格的约束解码能力，并改为按量计费——因为托管模型是黑盒，你无法掌控其内部优化", "en": "You gain portability and skip self-hosting GPUs, but sacrifice the deterministic prefix-cache and fork-sharing benefits and strict constrained decoding, and switch to per-token billing — because the hosted model is a black box whose internal optimizations you cannot control"},
                    {"zh": "什么都不牺牲，托管后端在所有维度都严格更优", "en": "You sacrifice nothing; the hosted backend is strictly better on every dimension"},
                    {"zh": "只牺牲启动速度，运行时能力完全一致", "en": "You only sacrifice startup speed; runtime capabilities are identical"},
                    {"zh": "牺牲了程序的可读性，必须重写整棵意图树", "en": "You sacrifice program readability and must rewrite the entire intent tree"},
                ],
                "answer": 0,
                "why": {
                    "zh": "本地运行时是白盒，吃满 RadixAttention 前缀缓存（第 7 课）、fork 分支共享（第 11 课）与严格约束解码；托管 API 是黑盒，这些确定性红利无从声明、无法依赖，约束解码受限，且按 token 计费。好处是可移植、免运维 GPU。程序本身一字不改（这正是 BaseBackend 抽象的价值），所以牺牲的不是可读性，而是运行时能力与成本结构。",
                    "en": "The local runtime is a white box delivering full RadixAttention prefix caching (Lesson 7), fork sharing (Lesson 11), and strict constrained decoding; a hosted API is a black box where those deterministic benefits cannot be declared or relied upon, constrained decoding is limited, and you pay per token. The upside is portability and no GPU ops. The program itself is unchanged (the whole point of the BaseBackend abstraction), so what you trade is runtime capability and cost structure, not readability.",
                },
            },
        ],
        "open": [
            {
                "zh": "用一两句话向同事解释“为什么同一个 SGLang 程序换后端只需改一行 <code>set_default_backend</code>”。请点出 <code>BaseBackend</code> 这层抽象接口的作用，以及程序为什么“不知道也不在乎”自己跑在哪个后端上。",
                "en": "In a sentence or two, explain to a colleague why switching backends for the same SGLang program takes only one line, <code>set_default_backend</code>. Point out the role of the <code>BaseBackend</code> abstract interface and why the program 'neither knows nor cares' which backend runs it.",
            },
            {
                "zh": "<code>RuntimeEndpoint</code> 被称为“Part 3（前端）与 Part 4+（运行时）对接的接缝”。请结合它把 <code>gen/fork</code> 翻成 <code>POST /generate</code> 的事实，说明：为什么只有这个后端能兑现 RadixAttention 前缀缓存（第 7 课）与 fork 分支共享（第 11 课），以及这次 HTTP 请求接下来会交给第 13–17 课的哪些组件处理？",
                "en": "<code>RuntimeEndpoint</code> is called 'the seam joining Part 3 (front-end) to Part 4+ (runtime).' Using the fact that it translates <code>gen/fork</code> into <code>POST /generate</code>, explain why only this backend can deliver RadixAttention prefix caching (Lesson 7) and fork branch sharing (Lesson 11), and which components from Lessons 13–17 will handle that HTTP request next.",
            },
        ],
    },
    "13-engine-and-http-server.html": {
        "mcq": [
            {
                "q": {
                    "zh": "关于离线 <code>Engine</code> 和在线 server，下面哪种说法最准确？",
                    "en": "Which statement about the offline <code>Engine</code> and the online server is most accurate?",
                },
                "opts": [
                    {
                        "zh": "在线 server 用 <code>launch_server</code> 起一个 FastAPI 应用，<strong>包住一个 Engine</strong> 并暴露 HTTP 路由；离线 Engine 是纯进程内 Python API，跳过 HTTP 直接调 <code>generate</code>。两者<strong>共享同一套运行时</strong>，server 没有重新实现任何运行时逻辑",
                        "en": "The online server uses <code>launch_server</code> to start a FastAPI app that <strong>wraps an Engine</strong> and exposes HTTP routes; the offline Engine is a pure in-process Python API calling <code>generate</code> without HTTP. Both <strong>share one runtime</strong>; the server reimplements no runtime logic",
                    },
                    {"zh": "在线 server 重新实现了一套独立的调度与前向逻辑，和 Engine 互不相干", "en": "The online server reimplements its own scheduling and forward logic, unrelated to the Engine"},
                    {"zh": "离线 Engine 也走 HTTP，只是端口不对外暴露", "en": "The offline Engine also uses HTTP, just on a non-exposed port"},
                    {"zh": "两者底层运行时不同，所以要分别学两遍", "en": "Their underlying runtimes differ, so you must learn each separately"},
                ],
                "answer": 0,
                "why": {
                    "zh": "server = Engine + 一层 HTTP 外壳。<code>launch_server</code> 内部构造一个 Engine，再用 FastAPI/uvicorn 包起来，暴露原生 <code>POST /generate</code> 与 OpenAI 兼容路由；离线 Engine 直接走链路后半段、跳过 HTTP。底层运行时（TokenizerManager、Scheduler、Detokenizer）完全相同，所以从第 14 课起的内容两者共用，只需学一遍。",
                    "en": "server = Engine + an HTTP shell. <code>launch_server</code> builds an Engine internally, then wraps it with FastAPI/uvicorn exposing native <code>POST /generate</code> and OpenAI-compatible routes; the offline Engine takes the back half of the chain, skipping HTTP. The underlying runtime (TokenizerManager, Scheduler, Detokenizer) is identical, so everything from Lesson 14 on is shared — learn it once.",
                },
            },
            {
                "q": {
                    "zh": "你要在强化学习训练里做 rollout（让训练器频繁生成样本）。为什么通常优先选离线 <code>Engine</code> 而不是起一个 HTTP server？",
                    "en": "You need rollout in RL training (the trainer generates samples frequently). Why usually prefer the offline <code>Engine</code> over standing up an HTTP server?",
                },
                "opts": [
                    {
                        "zh": "因为离线 Engine 是<strong>进程内直接调用</strong>，没有 HTTP 解析、序列化和网络往返的开销，延迟最低；训练器可以在同一个进程里直接 <code>generate</code>，最适合高频 rollout",
                        "en": "Because the offline Engine is an <strong>in-process direct call</strong> with no HTTP parsing, serialization, or network round-trip — lowest latency; the trainer can <code>generate</code> in the same process, ideal for high-frequency rollout",
                    },
                    {"zh": "因为 HTTP server 无法加载大模型", "en": "Because an HTTP server cannot load large models"},
                    {"zh": "因为离线 Engine 生成质量更高、精度不同", "en": "Because the offline Engine produces higher-quality, different-precision outputs"},
                    {"zh": "因为只有 Engine 支持采样参数", "en": "Because only the Engine supports sampling params"},
                ],
                "answer": 0,
                "why": {
                    "zh": "RL rollout 调用极其频繁，每多一层 HTTP+序列化+网络往返都会被放大成可观的开销。离线 Engine 在同一进程内直接返回 Python 结果，省掉这几层，延迟最低；训练器可直接持有 Engine 对象做 rollout（第 51 课）。生成质量与采样能力两者一致——差别只在“要不要那层 HTTP”。",
                    "en": "RL rollout calls are extremely frequent, so every extra HTTP+serialization+network round-trip is magnified into real overhead. The offline Engine returns Python results in the same process, dropping those layers for lowest latency; the trainer can hold the Engine object directly for rollout (Lesson 51). Output quality and sampling are identical — the only difference is whether you want that HTTP layer.",
                },
            },
            {
                "q": {
                    "zh": "构造 <code>sgl.Engine(model_path=...)</code> 时，<code>Engine.__init__</code> 主要做了什么？",
                    "en": "When you construct <code>sgl.Engine(model_path=...)</code>, what does <code>Engine.__init__</code> mainly do?",
                },
                "opts": [
                    {
                        "zh": "解析 <code>ServerArgs</code>，然后调用 <code>_launch_subprocesses</code> 拉起三进程：主进程内的 TokenizerManager，加上 Scheduler 与 DetokenizerManager 两个子进程，并建好 ZMQ 通道（即第 2 课的三进程模型）",
                        "en": "Parse <code>ServerArgs</code>, then call <code>_launch_subprocesses</code> to spin up three processes: TokenizerManager in the main process plus Scheduler and DetokenizerManager subprocesses, and wire up ZMQ (Lesson 2's three-process model)",
                    },
                    {"zh": "仅仅把模型权重加载到 CPU 内存，不创建任何进程", "en": "Merely load model weights into CPU memory, creating no processes"},
                    {"zh": "启动一个 FastAPI 服务器并监听端口", "en": "Start a FastAPI server and listen on a port"},
                    {"zh": "什么都不做，等到第一次 generate 才初始化", "en": "Do nothing, initializing only on the first generate"},
                ],
                "answer": 0,
                "why": {
                    "zh": "构造 Engine 就是给整套运行时“点火”：先把 kwargs 收成 <code>ServerArgs</code>，再 <code>_launch_subprocesses</code> 在主进程建 TokenizerManager、各起一个子进程跑 Scheduler 与 Detokenizer，并连好 ZMQ 通道。注意它<strong>不</strong>启动 FastAPI——HTTP 那层是 server 的事。在线 server 也先构造 Engine，所以同样经过这一步（第 2 课）。",
                    "en": "Constructing the Engine ignites the whole runtime: gather kwargs into <code>ServerArgs</code>, then <code>_launch_subprocesses</code> builds TokenizerManager in the main process, starts one subprocess each for Scheduler and Detokenizer, and wires ZMQ. Note it does <strong>not</strong> start FastAPI — the HTTP layer is the server's job. The online server also builds an Engine first, so it goes through this too (Lesson 2).",
                },
            },
        ],
        "open": [
            {
                "zh": "用“同一个后厨、两种点单方式”的类比，向同事解释离线 Engine 与在线 server 的关系。请明确说出：server 是否重新实现了运行时？从第 14 课起学的 TokenizerManager、Scheduler、Detokenizer 对两者是不是同一套？以及为什么这意味着你“只需学一遍底层”。",
                "en": "Using the 'one kitchen, two ways to order' analogy, explain to a colleague the relationship between the offline Engine and the online server. State clearly: does the server reimplement the runtime? Are the TokenizerManager, Scheduler, and Detokenizer (from Lesson 14 on) the same for both? And why does that mean you 'learn the internals only once'?",
            },
            {
                "zh": "给定两个场景：(a) 给一万条 prompt 批量打分做离线评测；(b) 给一个面向公网、多语言客户端的聊天产品提供服务。分别该选离线 Engine 还是在线 server？请从开销、并发、跨语言/跨机访问、OpenAI 兼容生态几个角度论证你的取舍。",
                "en": "Two scenarios: (a) batch-score ten thousand prompts for offline eval; (b) serve a public-facing, multi-language chat product. For each, pick the offline Engine or the online server, and justify your choice from overhead, concurrency, cross-language/cross-machine access, and the OpenAI-compatible ecosystem.",
            },
        ],
    },
    "14-tokenizer-manager.html": {
        "mcq": [
            {
                "q": {
                    "zh": "TokenizerManager 运行在哪个进程？为什么这样安排？",
                    "en": "Which process does the TokenizerManager run in, and why?",
                },
                "opts": [
                    {
                        "zh": "跑在<strong>主进程</strong>（和 HTTP server / Engine 同进程），<strong>不</strong>在 GPU 子进程里——因为分词/反分词是 CPU 密集的字符串工作，隔在 GPU 进程之外才能让 CPU 与 GPU <strong>重叠</strong>（零开销，第 21 课）",
                        "en": "In the <strong>main process</strong> (with the HTTP server / Engine), <strong>not</strong> the GPU subprocess — because tokenize/detokenize are CPU-bound string work, and keeping them off the GPU process lets CPU and GPU <strong>overlap</strong> (zero-overhead, Lesson 21)",
                    },
                    {"zh": "跑在 GPU 子进程里，和 Scheduler 一起做前向计算", "en": "In the GPU subprocess, doing forward passes alongside the Scheduler"},
                    {"zh": "每条请求各起一个独立进程，用完即销毁", "en": "A fresh process per request, destroyed after use"},
                    {"zh": "它没有自己的进程，只是 Scheduler 里的一个函数", "en": "It has no process of its own — just a function inside the Scheduler"},
                ],
                "answer": 0,
                "why": {
                    "zh": "TokenizerManager 是运行时的前门，坐在主进程做 CPU 侧预处理，绝不碰 GPU。把 GPU 循环单独关进子进程后，CPU 在给下一批分词的同时，GPU 正在为上一批做前向，两者真正并行——这正是第 21 课“零开销调度”的根基，而它与调度器之间那一跳是 ZMQ/IPC 接缝。",
                    "en": "The TokenizerManager is the runtime's front door, doing CPU-side preprocessing in the main process and never touching the GPU. With the GPU loop locked in its own subprocess, the CPU tokenizes the next batch while the GPU forwards the previous one — true overlap, the foundation of Lesson 21's zero-overhead scheduling. The hop to the scheduler is the ZMQ/IPC seam.",
                },
            },
            {
                "q": {
                    "zh": "几百条请求并发涌入，<code>rid</code>（请求号）最关键的作用是什么？",
                    "en": "With hundreds of concurrent requests, what is the most critical role of the <code>rid</code> (request id)?",
                },
                "opts": [
                    {
                        "zh": "作为跨进程的“身份证”：后台输出<strong>乱序交错</strong>地流回来时，TokenizerManager 凭 <code>rid</code> 在 <code>rid_to_state</code> 里找到正在 <code>await</code> 的那个协程，把每段输出<strong>准确投回</strong>原调用方",
                        "en": "A cross-process 'ID card': when outputs flow back <strong>interleaved and out of order</strong>, the TokenizerManager uses the <code>rid</code> in <code>rid_to_state</code> to find the awaiting coroutine and <strong>route each chunk back</strong> to the right caller",
                    },
                    {"zh": "决定请求的采样温度和 top_p", "en": "It sets the request's sampling temperature and top_p"},
                    {"zh": "它是 token id 的别名，分词后才生成", "en": "It is an alias for token ids, generated after tokenization"},
                    {"zh": "用来给请求排优先级，数字越小越先算", "en": "It prioritizes requests — smaller numbers compute first"},
                ],
                "answer": 0,
                "why": {
                    "zh": "请求发出后主进程不阻塞，而是把 <code>rid → 协程状态</code> 记进 <code>rid_to_state</code>。回包乱序交错地流回时，唯有靠唯一的 <code>rid</code> 才能对号入座，把输出交还给正确的等待协程（第 17 课流式回传）。没有 rid，几百条并发请求的回包就会张冠李戴。",
                    "en": "After sending, the main process does not block — it records <code>rid → coroutine state</code> in <code>rid_to_state</code>. As replies stream back interleaved, only the unique <code>rid</code> can match each chunk to the correct awaiting coroutine (streaming, Lesson 17). Without it, hundreds of concurrent replies would go to the wrong callers.",
                },
            },
            {
                "q": {
                    "zh": "在 <code>generate_request</code> 里，TokenizerManager 究竟把<strong>什么</strong>通过 ZMQ 发给调度器子进程？",
                    "en": "Inside <code>generate_request</code>, what exactly does the TokenizerManager send over ZMQ to the scheduler subprocess?",
                },
                "opts": [
                    {
                        "zh": "一个已分词、带唯一 <code>rid</code> 的 <code>TokenizedGenerateReqInput</code>（含 token id 与采样参数，第 16 课）——调度器只认 token，从不直接接触原始文本",
                        "en": "A tokenized <code>TokenizedGenerateReqInput</code> with a unique <code>rid</code> (token ids + sampling params, Lesson 16) — the scheduler only speaks tokens and never sees raw text",
                    },
                    {"zh": "原始 prompt 文本字符串，让调度器自己去分词", "en": "The raw prompt text string, leaving the scheduler to tokenize"},
                    {"zh": "模型权重和 KV 缓存指针", "en": "Model weights and KV-cache pointers"},
                    {"zh": "一个 HTTP 请求对象，交给调度器解析", "en": "An HTTP request object for the scheduler to parse"},
                ],
                "answer": 0,
                "why": {
                    "zh": "TokenizerManager 先 <code>_tokenize_one_request</code> 得到 token id、构造并校验 <code>SamplingParams</code>，再打包成带 <code>rid</code> 的 <code>TokenizedGenerateReqInput</code>（第 16 课的 io_struct 消息），用 <code>_send_one_request</code> 经 ZMQ socket 投给调度器。文本→token 的翻译在主进程就做完了，GPU 子进程因此只需面对 token，专心计算。",
                    "en": "The TokenizerManager first runs <code>_tokenize_one_request</code> to get token ids, builds and verifies <code>SamplingParams</code>, then packs a <code>TokenizedGenerateReqInput</code> with a <code>rid</code> (Lesson 16's io_struct message) and dispatches it via <code>_send_one_request</code> over a ZMQ socket. The text→token translation is done in the main process, so the GPU subprocess only faces tokens and just computes.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“酒店前台接待员”的类比，向同事解释 TokenizerManager 的职责。请讲清楚：它把口语需求翻译成什么（token id + 参数 + 预订号 rid），为什么自己从不“进后厨”（不碰 GPU），以及它如何同时招呼几十位客人还不把回包交错。",
                "en": "Using the 'hotel front-desk receptionist' analogy, explain the TokenizerManager's job to a colleague. Make clear: what it translates spoken needs into (token ids + params + a booking number rid), why it never 'enters the kitchen' (never touches the GPU), and how it juggles dozens of guests without mixing up replies.",
            },
            {
                "zh": "假设有人提议把分词直接搬进 GPU 子进程、省掉 TokenizerManager 与调度器之间的 ZMQ 一跳。请从 CPU/GPU 重叠（第 21 课）、GIL、以及主进程 vs 子进程职责分离的角度，论证这样做会损失什么。",
                "en": "Suppose someone proposes moving tokenization into the GPU subprocess to drop the ZMQ hop between the TokenizerManager and the scheduler. Argue what would be lost, from the angles of CPU/GPU overlap (Lesson 21), the GIL, and the main-process vs subprocess separation of duties.",
            },
        ],
    },
    "15-openai-anthropic-ollama-compat.html": {
        "mcq": [
            {
                "q": {
                    "zh": "兼容层（OpenAI/Anthropic/Ollama）究竟做了什么，才能让现成客户端不改代码就接入 SGLang？",
                    "en": "What does the compat layer (OpenAI/Anthropic/Ollama) actually do so existing clients work against SGLang unchanged?",
                },
                "opts": [
                    {
                        "zh": "它是一组 <strong>serving 类</strong>，只做 <strong>schema 翻译 + 聊天模板套用</strong>：把方言请求拼成原生 <code>GenerateReqInput</code>（第 16 课）交给 TokenizerManager（第 14 课），再把输出映射回该协议的响应/SSE——<strong>不重新实现引擎</strong>",
                        "en": "A set of <strong>serving classes</strong> that only do <strong>schema translation + chat templating</strong>: shape the dialect request into a native <code>GenerateReqInput</code> (Lesson 16) for the TokenizerManager (Lesson 14), then map outputs back to that protocol's response/SSE — <strong>not a reimplemented engine</strong>",
                    },
                    {"zh": "它为每种协议各跑一套独立的推理引擎和 KV 缓存", "en": "It runs a separate inference engine and KV cache per protocol"},
                    {"zh": "它把 SGLang 的请求转发到真正的 OpenAI 服务器去计算", "en": "It forwards SGLang requests to the real OpenAI servers to compute"},
                    {"zh": "它要求客户端改用 SGLang 专属 SDK 才能工作", "en": "It requires clients to switch to a SGLang-only SDK"},
                ],
                "answer": 0,
                "why": {
                    "zh": "运行时本身是<strong>协议无关</strong>的，只认 <code>GenerateReqInput</code>。适配器仅在最外层做两件窄事：把 OpenAI/Anthropic/Ollama 字段映射成原生字段（schema 翻译），以及把 messages 按模型模板拼成 prompt（聊天模板）。因为职责窄，新增协议只需加一个适配器，而最热的推理快路径始终共享、只优化一遍。",
                    "en": "The runtime is <strong>protocol-agnostic</strong> and only knows <code>GenerateReqInput</code>. The adapter does two narrow things at the outermost layer: map OpenAI/Anthropic/Ollama fields to native fields (schema translation) and assemble messages into a prompt per the model's template (chat templating). Because the job is narrow, adding a protocol is just one more adapter, and the hottest fast path stays shared and optimized once.",
                },
            },
            {
                "q": {
                    "zh": "“OpenAI” 在 SGLang 里有两个方向。本课（方向 B）讲的是哪一个？",
                    "en": "\"OpenAI\" has two directions in SGLang. Which one does this lesson (Direction B) cover?",
                },
                "opts": [
                    {
                        "zh": "一个 OpenAI <strong>客户端</strong>（别人的 openai SDK / LangChain）<strong>向内</strong>调用你的 SGLang 服务器——你的部署<strong>伪装成 OpenAI 端点</strong>，对方只改 <code>base_url</code> 就接入",
                        "en": "An OpenAI <strong>client</strong> (someone's openai SDK / LangChain) calls <strong>IN</strong> to your SGLang server — your deployment <strong>masquerades as an OpenAI endpoint</strong>, and they plug in by just changing <code>base_url</code>",
                    },
                    {"zh": "你的 SGLang 前端程序向外调用 OpenAI 当后端（那是第 12 课的方向 A）", "en": "Your SGLang frontend program calls OUT to OpenAI as a backend (that is Direction A, Lesson 12)"},
                    {"zh": "SGLang 把请求和响应都代理给真正的 OpenAI", "en": "SGLang proxies both request and response to the real OpenAI"},
                    {"zh": "两个方向其实是同一回事，不必区分", "en": "The two directions are the same thing and need not be distinguished"},
                ],
                "answer": 0,
                "why": {
                    "zh": "方向 A（第 12 课）：SGLang 程序是<strong>客户端</strong>，把 OpenAI 当后端调用。方向 B（本课）：OpenAI 客户端是调用方，你的 SGLang 服务器是被调用的<strong>服务器</strong>，戴上 OpenAI 的面具让整个生态插进来。一个是“SGLang 打 OpenAI”，一个是“OpenAI 打 SGLang”，方向相反，切勿混淆。",
                    "en": "Direction A (Lesson 12): the SGLang program is the <strong>client</strong>, calling OpenAI as a backend. Direction B (this lesson): the OpenAI client is the caller and your SGLang server is the callee, wearing an OpenAI mask so the whole ecosystem plugs in. One is 'SGLang calling OpenAI,' the other 'OpenAI calling SGLang' — opposite, never conflate.",
                },
            },
            {
                "q": {
                    "zh": "为什么任意一个 OpenAI 客户端都能直接打到 SGLang 服务器上？",
                    "en": "Why does any OpenAI client work against the SGLang server out of the box?",
                },
                "opts": [
                    {
                        "zh": "因为 SGLang 实现了 OpenAI 的 HTTP 协议：<code>/v1/chat/completions</code> 由 <code>OpenAIServingChat</code> 接管，把 OpenAI 形状的请求翻成原生请求、走同一套运行时，再把结果映射回 <code>chat.completion</code> 对象/SSE 块，所以客户端只需把 <code>base_url</code> 指向 SGLang",
                        "en": "Because SGLang implements OpenAI's HTTP protocol: <code>/v1/chat/completions</code> is handled by <code>OpenAIServingChat</code>, which translates the OpenAI-shaped request into a native one, runs the same runtime, and maps results back to <code>chat.completion</code> objects/SSE chunks — so the client only points <code>base_url</code> at SGLang",
                    },
                    {"zh": "因为 SGLang 内部偷偷调用了真正的 OpenAI API", "en": "Because SGLang secretly calls the real OpenAI API under the hood"},
                    {"zh": "因为 OpenAI 客户端会自动检测并切换到 SGLang 原生协议", "en": "Because OpenAI clients auto-detect and switch to SGLang's native protocol"},
                    {"zh": "因为两套协议字段完全相同，无需任何翻译", "en": "Because the two protocols' fields are identical, needing no translation"},
                ],
                "answer": 0,
                "why": {
                    "zh": "OpenAI 客户端只会按 OpenAI 的 HTTP 约定发请求、解析响应。SGLang 的兼容层在 <code>entrypoints/openai/</code> 里实现了这套约定：每条路由绑一个 serving 类负责翻译，请求被拼成原生 <code>GenerateReqInput</code> 走共享快路径，响应再被格式化回 OpenAI 的对象或流式块。客户端感知不到背后是 SGLang，只需改 <code>base_url</code>。",
                    "en": "An OpenAI client only sends requests and parses responses per OpenAI's HTTP contract. SGLang's compat layer under <code>entrypoints/openai/</code> implements that contract: each route binds a serving class that translates, the request becomes a native <code>GenerateReqInput</code> on the shared fast path, and the response is formatted back into OpenAI objects or stream chunks. The client never notices SGLang underneath — just change <code>base_url</code>.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“门口一排多语翻译官”的类比，向同事解释兼容层。请讲清楚：客人各说 OpenAI/Anthropic/Ollama 方言，翻译官把它们都翻成厨房唯一听得懂的“本店语言”（原生 GenerateReqInput），厨房只有一个、做菜路径共享，菜做好再翻回各自方言端上。并说明为什么这是“薄适配器 + 共享内核”而不是给每种协议各写一个引擎。",
                "en": "Using the 'row of multilingual translators at the door' analogy, explain the compat layer to a colleague. Make clear: guests speak the OpenAI/Anthropic/Ollama dialects, the translators convert each into the one house language the kitchen understands (native GenerateReqInput), there is only one kitchen with a shared cooking path, and the dish is translated back into each dialect. Explain why this is 'thin adapters + shared core' rather than a separate engine per protocol.",
            },
            {
                "zh": "请把两个“OpenAI 方向”讲清楚并对比：方向 A（第 12 课）SGLang 前端程序向外调用 OpenAI 当后端，谁是客户端？方向 B（本课）OpenAI 客户端向内调用 SGLang 服务器，谁是服务器？再说明在方向 B 里，一个 <code>/v1/chat/completions</code> 请求如何经 <code>OpenAIServingChat</code> 套聊天模板、拼成 <code>GenerateReqInput</code>、交给 TokenizerManager，最后映射回 SSE。",
                "en": "Distinguish and compare the two 'OpenAI directions': in Direction A (Lesson 12) the SGLang frontend program calls OUT to OpenAI as a backend — who is the client? In Direction B (this lesson) an OpenAI client calls IN to the SGLang server — who is the server? Then describe how, in Direction B, a <code>/v1/chat/completions</code> request flows through <code>OpenAIServingChat</code> to apply the chat template, build a <code>GenerateReqInput</code>, hand it to the TokenizerManager, and finally map back to SSE.",
            },
        ],
    },
    "16-io-structs-and-ipc.html": {
        "mcq": [
            {
                "q": {
                    "zh": "SGLang 的三个进程之间为什么要靠 <code>io_struct.py</code> 的 dataclass + ZMQ 传消息，而不是直接共享对象？",
                    "en": "Why do SGLang's three processes communicate via <code>io_struct.py</code> dataclasses + ZMQ instead of sharing objects directly?",
                },
                "opts": [
                    {
                        "zh": "因为三进程（第 2 课）<strong>各有独立内存空间</strong>，一个进程的对象另一个根本读不到——拆进程是为绕开 GIL、让 CPU/GPU 重叠（第 21 课）的必然代价，于是只能把消息<strong>用 dataclass 定形状、经 ZMQ 序列化</strong>过线，换来便宜、显式、可打印可断点的协作",
                        "en": "Because the three processes (Lesson 2) own <strong>separate memory</strong>, so one process's objects are unreadable to another — splitting processes is the necessary cost of dodging the GIL and overlapping CPU/GPU (Lesson 21), so messages must be <strong>shaped by dataclasses and serialized over ZMQ</strong>, buying cheap, explicit, printable/breakpoint-able cooperation",
                    },
                    {"zh": "因为 dataclass 比共享内存运行得更快，纯粹是性能优化", "en": "Because dataclasses run faster than shared memory, purely a performance optimization"},
                    {"zh": "因为 Python 根本不支持进程间共享内存", "en": "Because Python has no way to share memory between processes at all"},
                    {"zh": "因为只有这样才能调用 GPU 上的 CUDA kernel", "en": "Because only this lets you call CUDA kernels on the GPU"},
                ],
                "answer": 0,
                "why": {
                    "zh": "关键不是“快”，而是“边界”：三进程是独立 OS 进程，内存彼此不可见。拆进程是为了绕开 GIL、让 CPU 与 GPU 真正重叠，代价就是不能再共享变量，只能传消息。<code>io_struct.py</code> 用 dataclass 把消息形状钉死、ZMQ 负责序列化传输，得到一套显式、可调试的跨进程协议。",
                    "en": "The point isn't 'speed' but 'boundaries': the three are separate OS processes with mutually invisible memory. Splitting them dodges the GIL and overlaps CPU/GPU; the cost is no shared variables, only passing messages. <code>io_struct.py</code> pins message shapes with dataclasses and ZMQ handles serialized transport, giving an explicit, debuggable cross-process protocol.",
                },
            },
            {
                "q": {
                    "zh": "当请求跨过 ZMQ 真正送达调度器（第 18 课）时，传过去的是哪个结构、装的是什么？",
                    "en": "When a request actually crosses ZMQ to the scheduler (Lesson 18), which struct travels, and what does it carry?",
                },
                "opts": [
                    {
                        "zh": "是 <code>TokenizedGenerateReqInput</code>，装的是<strong>已分词的 token id</strong>（<code>input_ids</code>）+ 采样参数 + <code>rid</code>——原始文本已在主进程被分词，<strong>过线后下游只认 token，不再碰人类文本</strong>",
                        "en": "It's <code>TokenizedGenerateReqInput</code>, carrying <strong>already-tokenized ids</strong> (<code>input_ids</code>) + sampling params + <code>rid</code> — the raw text was tokenized in the main process, and <strong>past the wire downstream only knows tokens, not human text</strong>",
                    },
                    {"zh": "是 <code>GenerateReqInput</code>，装的是用户的原始文本字符串", "en": "It's <code>GenerateReqInput</code>, carrying the user's raw text string"},
                    {"zh": "是 <code>BatchStrOutput</code>，装的是解码后的文本", "en": "It's <code>BatchStrOutput</code>, carrying decoded text"},
                    {"zh": "直接把 tokenizer 对象本身共享给调度器进程", "en": "The tokenizer object itself is shared directly with the scheduler process"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<code>GenerateReqInput</code> 是用户视角的请求，只活在主进程、<strong>不跨进程</strong>。分词后才打包成 <code>TokenizedGenerateReqInput</code>，里面是 <code>input_ids</code> 而非原始文本，这个才是被 ZMQ 序列化送往调度器的消息。所以跨线的是 token，不是文本——这正是分词留在主进程、把语言学负担挡在最外层的体现（第 14 课）。",
                    "en": "<code>GenerateReqInput</code> is the user-facing request that lives only in the main process and <strong>does not cross processes</strong>. Only after tokenization is it packed into <code>TokenizedGenerateReqInput</code>, which holds <code>input_ids</code> rather than raw text and is the message ZMQ serializes to the scheduler. So what crosses is tokens, not text — reflecting that tokenization stays in the main process, keeping the linguistic burden at the outermost layer (Lesson 14).",
                },
            },
            {
                "q": {
                    "zh": "调度器把多条请求攒成一批一起算，回来的 <code>BatchTokenIDOutput</code> 是交错的批量输出。<code>rid</code> 在这里起什么作用？",
                    "en": "The scheduler batches many requests together, so the returning <code>BatchTokenIDOutput</code> is interleaved and batched. What does <code>rid</code> do here?",
                },
                "opts": [
                    {
                        "zh": "<code>rid</code> 是贯穿每条消息的<strong>请求号</strong>：批量输出里每段 token 都带着自己的 <code>rid</code>，TokenizerManager 据此把交错的输出<strong>拆开、投回正确的等待协程</strong>，没有它几百条并发回包就会张冠李戴",
                        "en": "<code>rid</code> is the <strong>request id</strong> threaded through every message: each token segment in the batched output carries its own <code>rid</code>, so the TokenizerManager <strong>splits and routes</strong> the interleaved output back to the right awaiting coroutine; without it, hundreds of concurrent replies would be misdelivered",
                    },
                    {"zh": "<code>rid</code> 只是日志用的可选字段，去掉也不影响路由", "en": "<code>rid</code> is just an optional logging field; removing it doesn't affect routing"},
                    {"zh": "<code>rid</code> 决定每条请求用哪块 GPU 计算", "en": "<code>rid</code> decides which GPU computes each request"},
                    {"zh": "<code>rid</code> 是采样参数的一部分，控制温度与 top_p", "en": "<code>rid</code> is part of the sampling params, controlling temperature and top_p"},
                ],
                "answer": 0,
                "why": {
                    "zh": "因为输出是批量、交错的（一条消息里混着第 7、3、9 号的 token），必须有一个稳定标识把每段对号入座。<code>rid</code> 从 <code>GenerateReqInput</code> 处生成，一路复制到 Tokenized 结构、再随 <code>Batch*Output</code> 的 <code>rids</code> 列表流回，是异步路由（第 14 课）与流式回传（第 17 课）共同依赖的地基。",
                    "en": "Because outputs are batched and interleaved (one message mixes tokens for requests 7, 3, 9), a stable identifier is needed to place each segment. <code>rid</code> is created at <code>GenerateReqInput</code>, copied into the tokenized struct, and flows back in the <code>rids</code> lists of the <code>Batch*Output</code> — the foundation shared by async routing (Lesson 14) and streaming (Lesson 17).",
                },
            },
        ],
        "open": [
            {
                "zh": "用“部门之间流转的标准化表单”的类比，描述一条请求在三进程间的生命周期：它如何从 <code>GenerateReqInput</code>（受理单）变成 <code>TokenizedGenerateReqInput</code>（工单，跨 ZMQ 送往调度器）、再到 <code>BatchTokenIDOutput</code>（结果清单）、最后 <code>BatchStrOutput</code>（打印好的信）。请特别说明：为什么“进”是单条而“出”是批量，以及每张表单右上角的 <code>rid</code> 起什么作用。",
                "en": "Using the 'standardized forms moving between departments' analogy, describe a request's lifecycle across the three processes: how it goes from <code>GenerateReqInput</code> (intake form) to <code>TokenizedGenerateReqInput</code> (work order, crossing ZMQ to the scheduler), to <code>BatchTokenIDOutput</code> (results sheet), and finally <code>BatchStrOutput</code> (printed letter). Explain in particular why 'in' is singular while 'out' is batched, and what the <code>rid</code> stamped on each form does.",
            },
            {
                "zh": "有人主张“让三个进程共享同一块内存、直接读写对方的请求对象，省掉序列化开销不是更快吗？”请你反驳：从 GIL、CPU/GPU 重叠（第 21 课）、容错隔离、可调试性几个角度，说明 SGLang 为什么宁可付出序列化成本也要用 dataclass + ZMQ 的“一切皆消息”设计。",
                "en": "Someone argues: 'Just let the three processes share one block of memory and read/write each other's request objects directly — wouldn't skipping serialization be faster?' Rebut this: from the angles of the GIL, CPU/GPU overlap (Lesson 21), fault isolation, and debuggability, explain why SGLang prefers the 'everything is a message' design with dataclasses + ZMQ even at the cost of serialization.",
            },
        ],
    },
    "17-detokenizer-and-streaming.html": {
        "mcq": [
            {
                "q": {
                    "zh": "反分词器每跑一步都把<strong>到目前为止的全部 token</strong> 解码成整段文本、整段发给客户端——这种做法的根本问题是什么？正确做法又是什么？",
                    "en": "Each step, the detokenizer decodes <strong>all tokens so far</strong> into the full text and sends the whole thing to the client. What's the fundamental problem, and what's the right approach?",
                },
                "opts": [
                    {
                        "zh": "全量重发让传输量随长度<strong>平方膨胀（O(n²)）</strong>，且客户端反复收到重复前缀；正确做法是<strong>增量</strong>：每条请求记一个 <code>sent_offset</code>，每步只发新增片段 <code>output_str[sent_offset:]</code> 再推进偏移量，每个字符<strong>只过线一次</strong>",
                        "en": "Resending everything makes traffic grow <strong>quadratically (O(n²))</strong> with length, and the client keeps receiving duplicate prefixes; the right approach is <strong>incremental</strong>: each request keeps a <code>sent_offset</code> and each step emits only the new slice <code>output_str[sent_offset:]</code> then advances the offset, so every character <strong>crosses the wire once</strong>",
                    },
                    {"zh": "没有问题，全量重发最简单也最快，无需优化", "en": "No problem at all — full resends are simplest and fastest, no optimization needed"},
                    {"zh": "问题在于解码太慢，应该改用 GPU 来做反分词", "en": "The problem is decoding is too slow; detokenize should be moved to the GPU"},
                    {"zh": "问题在于 token id 不够多，应该等全部生成完再一次性发", "en": "The problem is too few token ids; you should wait until all are generated and send once"},
                ],
                "answer": 0,
                "why": {
                    "zh": "朴素的“每步整段重发”有两个致命缺陷：传输量 O(n²) 平方膨胀，且重复前缀把“算新增”的负担甩给客户端。增量切片用 <code>sent_offset</code> 记“已发到哪”，每步只截 <code>output_str[sent_offset:]</code> 这一小段再推进偏移量，传输量线性、客户端直接拼接，天然适配 SSE 流。",
                    "en": "Naive full resends have two fatal flaws: O(n²) traffic and duplicate prefixes that dump the 'find the new part' burden on the client. Incremental slicing uses <code>sent_offset</code> to track 'how far sent', emitting only <code>output_str[sent_offset:]</code> each step then advancing — linear traffic, the client just concatenates, and it fits SSE naturally.",
                },
            },
            {
                "q": {
                    "zh": "某一步解码后，文本尾部正好是<strong>半个汉字</strong>（不完整的 UTF-8 序列）。反分词器应该怎么做？为什么？",
                    "en": "After a step, the tail of the text is exactly <strong>half a Chinese character</strong> (an incomplete UTF-8 sequence). What should the detokenizer do, and why?",
                },
                "opts": [
                    {
                        "zh": "<strong>先把这半个字符扣住、不发</strong>，<code>sent_offset</code> 暂不推进，等下一步更多 token 到来拼成完整字符再吐——否则客户端屏幕会闪出乱码方块 <code>�</code>，流出半个字符",
                        "en": "<strong>Hold the half-character back, emit nothing</strong>, and don't advance <code>sent_offset</code> yet — wait for more tokens next step to complete the character before emitting; otherwise the client flashes a garbled box <code>�</code>, streaming half a character",
                    },
                    {"zh": "直接把半个字符发出去，客户端会自动修复乱码", "en": "Just send the half-character; the client will auto-repair the garbling"},
                    {"zh": "丢弃这半个字符，继续解码后面的 token", "en": "Discard the half-character and keep decoding later tokens"},
                    {"zh": "立刻终止这条请求的生成，报 UTF-8 错误", "en": "Immediately terminate this request's generation and raise a UTF-8 error"},
                ],
                "answer": 0,
                "why": {
                    "zh": "一个 token ≠ 一个字符：汉字、emoji 常跨多个字节甚至多个 token。若把不完整 UTF-8 直接发出，屏幕会出现 <code>�</code>。所以反分词器必须把还不构成完整字符的字节<strong>先攒着</strong>，等拼成完整字符再发——这正是流式输出看起来顺滑、绝不蹦半个字的原因。",
                    "en": "One token ≠ one character: Chinese characters and emojis often span multiple bytes or even tokens. Sending incomplete UTF-8 as-is shows a <code>�</code>. So the detokenizer must <strong>hold back</strong> bytes that don't yet form a complete character and emit only once whole — exactly why streaming looks smooth and never spits half a character.",
                },
            },
            {
                "q": {
                    "zh": "为什么 SGLang 要把反分词单独放进一个<strong>子进程</strong>，而不是在调度器或主进程里顺手做掉？",
                    "en": "Why does SGLang put detokenize in its <strong>own subprocess</strong> rather than doing it inside the scheduler or main process?",
                },
                "opts": [
                    {
                        "zh": "因为反分词是<strong>纯 CPU 的字符串活</strong>，放进独立进程才能和 GPU 的前向循环<strong>重叠</strong>、互不等待（零开销思想，第 21 课）；这也呼应第 2 课的三进程模型——分词/调度+前向/反分词各占一进程",
                        "en": "Because detokenize is <strong>pure CPU string work</strong>; isolating it in its own process lets it <strong>overlap</strong> the GPU forward loop without either waiting (the zero-overhead idea, Lesson 21) — echoing Lesson 2's three-process model: tokenize / schedule+forward / detokenize each in one process",
                    },
                    {"zh": "因为反分词需要直接访问 GPU 显存，必须独立进程", "en": "Because detokenize needs direct GPU VRAM access, requiring a separate process"},
                    {"zh": "因为子进程比主进程拥有更高的 CPU 优先级", "en": "Because subprocesses get higher CPU priority than the main process"},
                    {"zh": "纯粹是历史遗留，没有实际意义", "en": "Purely a historical accident with no real significance"},
                ],
                "answer": 0,
                "why": {
                    "zh": "反分词把 token id 拼回文本是 CPU 密集的字符串工作，和 GPU 矩阵乘毫无关系。若塞进 GPU 进程，CPU 在拼字符串时 GPU 只能干等。拆成独立子进程后，反分词能和 GPU 前向<strong>真正并行重叠</strong>——这正是第 21 课零开销调度的根基，也和第 2 课的三进程边界一脉相承。",
                    "en": "Stitching token ids back into text is CPU-bound string work, unrelated to GPU matmuls. Inside the GPU process, the GPU would idle while the CPU builds strings. As a separate subprocess, detokenize <strong>truly overlaps</strong> the GPU forward — the foundation of Lesson 21's zero-overhead scheduling and consistent with Lesson 2's three-process boundaries.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“同声传译只说新增的词、并等完整词再出声”的类比，完整描述一条流式请求从 GPU 吐出 token id 到客户端看到“打字机”文字的全过程。请点明：<code>BatchTokenIDOutput</code>→<code>BatchStrOutput</code>（第 16 课）的消息往返、反分词器为什么把结果发回 TokenizerManager（第 14 课）而不是直接给客户端、<code>sent_offset</code> 如何保证“只增不重”，以及不完整 UTF-8 为什么要先攒着。",
                "en": "Using the 'simultaneous interpreter speaks only the new words and waits for a whole word' analogy, describe end-to-end how a streaming request goes from the GPU emitting token ids to the client seeing 'typewriter' text. Call out: the <code>BatchTokenIDOutput</code>→<code>BatchStrOutput</code> (Lesson 16) round-trip, why the detokenizer sends results back to the TokenizerManager (Lesson 14) instead of straight to the client, how <code>sent_offset</code> guarantees 'add-only, never-repeat', and why incomplete UTF-8 must be held back.",
            },
            {
                "zh": "生成什么时候算结束？请列出三种停止条件（EOS token / 停止串 / max_new_tokens，详见第 28 课），分别说明它们对最终输出的影响，并解释反分词器在结束时为什么要调用 <code>trim_matched_stop</code> 把停止串裁掉、再发出最后一段 <code>output_str[sent_offset:]</code>。最后说明 <strong>HTTP 服务路由</strong>是如何用 <code>StreamingResponse</code>（<code>text/event-stream</code>）把 TokenizerManager 逐条 yield 的片段变成 SSE “打字机”的（离线 Engine 则不走 SSE）。",
                "en": "When does generation end? List the three stop conditions (EOS token / stop string / max_new_tokens, see Lesson 28), explain each one's effect on the final output, and explain why on finish the detokenizer calls <code>trim_matched_stop</code> to cut the stop string before emitting the last <code>output_str[sent_offset:]</code>. Finally, explain how the <strong>HTTP server route</strong> turns the slices TokenizerManager yields into an SSE 'typewriter' via <code>StreamingResponse</code> (<code>text/event-stream</code>) — and why the offline Engine path has no SSE.",
            },
        ],
    },
    "18-scheduler-event-loop.html": {
        "mcq": [
            {
                "q": {
                    "zh": "调度器 <code>event_loop_normal</code> 每一个 step（一圈）依次做哪五件事？<strong>顺序</strong>为什么重要？",
                    "en": "What five things does the scheduler's <code>event_loop_normal</code> do each step (one revolution), and why does the <strong>order</strong> matter?",
                },
                "opts": [
                    {
                        "zh": "<code>recv_requests</code>（收）→ <code>process_input_requests</code>（入等待队列）→ <code>get_next_batch_to_run</code>（组这一步的批）→ <code>run_batch</code>（上 GPU 前向+采样）→ <code>process_batch_result</code>（追加 token/清退完成的/释放 KV/发反分词），然后回到开头；顺序重要是因为<strong>必须先收进新请求、再组批，组完批才能前向，前向出结果才能收尾</strong>",
                        "en": "<code>recv_requests</code> (receive) → <code>process_input_requests</code> (into waiting queue) → <code>get_next_batch_to_run</code> (form this step's batch) → <code>run_batch</code> (GPU forward+sample) → <code>process_batch_result</code> (append tokens/evict finished/free KV/send to detok), then loop; the order matters because <strong>you must take in new reqs before forming a batch, form a batch before forwarding, and have results before finishing</strong>",
                    },
                    {"zh": "先 <code>run_batch</code> 再 <code>recv_requests</code>，因为要先算完才收新请求", "en": "First <code>run_batch</code> then <code>recv_requests</code>, since you compute before receiving"},
                    {"zh": "只有 <code>get_next_batch_to_run</code> 一步，其余都在 GPU 内核里完成", "en": "Only <code>get_next_batch_to_run</code>; everything else happens inside the GPU kernel"},
                    {"zh": "顺序无所谓，五步可以任意打乱并行执行", "en": "Order is irrelevant; the five steps can run in any shuffled, parallel order"},
                ],
                "answer": 0,
                "why": {
                    "zh": "事件循环是 <code>while True</code>，每圈五步有严格<strong>数据依赖</strong>：收件箱里没新请求就无从入队，等待队列里没请求就组不出 prefill 批，没组好批就无法前向，没前向结果就无法追加 token、判完成、释放 KV。打乱顺序会破坏依赖。记住这五步的顺序，就拿到了 Part 5 的主索引。",
                    "en": "The loop is <code>while True</code>, and the five steps have strict <strong>data dependencies</strong>: no new reqs means nothing to enqueue, an empty waiting queue means no prefill batch, no batch means no forward, and no forward result means you can't append tokens, detect finish, or free KV. Shuffling breaks the dependency. Memorizing this order is the master index to Part 5.",
                },
            },
            {
                "q": {
                    "zh": "为什么调度器要在<strong>每一个 step</strong> 都重新调用 <code>get_next_batch_to_run</code> 把批重组一遍，而不是一次组好用到底？",
                    "en": "Why does the scheduler re-call <code>get_next_batch_to_run</code> to re-form the batch <strong>every step</strong>, rather than forming it once and reusing it?",
                },
                "opts": [
                    {
                        "zh": "因为这正是<strong>连续批处理</strong>（第 5 课）的物理现场：每步重组才能让<strong>完成的请求当场离场、立刻释放 KV 槽</strong>，让<strong>等待的请求下一步就补入</strong>，批次始终满载，GPU 不为任何单条请求空转",
                        "en": "Because this is where <strong>continuous batching</strong> (Lesson 5) physically happens: re-forming each step lets <strong>finished reqs leave on the spot and free their KV slots</strong> and <strong>waiting reqs fill in next step</strong>, keeping the batch full so the GPU never idles for any single request",
                    },
                    {"zh": "因为 GPU 要求每步换一个全新的 batch 对象，否则会报错", "en": "Because the GPU requires a brand-new batch object each step or it errors"},
                    {"zh": "因为重组批能提高单条请求的解码速度", "en": "Because re-forming speeds up a single request's decoding"},
                    {"zh": "纯粹是为了日志统计方便，没有性能意义", "en": "Purely for logging convenience, with no performance meaning"},
                ],
                "answer": 0,
                "why": {
                    "zh": "若一次组好绑死到底，就退化成静态批处理：整批要等最慢的那条，完成的槽位空转、新请求进不来。每步重组让批成为“被重新计算的结果”而非静态对象——完成的过滤离场、等待的接纳补入，批永远满。这就是连续批处理，也是高吞吐的第一引擎。",
                    "en": "Forming once and locking it degrades to static batching: the whole batch waits for the slowest req, finished slots idle, new reqs can't enter. Re-forming each step makes the batch a 'recomputed result' rather than a static object — finished filtered out, waiting admitted in, always full. That is continuous batching, the first engine of high throughput.",
                },
            },
            {
                "q": {
                    "zh": "在一个 step 里，<strong>哪部分是调度器在 CPU 上的决策、哪部分是真正烧算力的 GPU 计算</strong>？这个区分如何引出 <code>event_loop_overlap</code>（第 21 课）？",
                    "en": "Within one step, <strong>which part is the scheduler's CPU decision and which is the actual GPU compute</strong>? How does this split motivate <code>event_loop_overlap</code> (Lesson 21)?",
                },
                "opts": [
                    {
                        "zh": "<code>get_next_batch_to_run</code> / <code>process_batch_result</code> 是<strong>纯 CPU 的轻量记账与决策</strong>，<code>run_batch</code>（→ModelRunner.forward）才是<strong>GPU 计算</strong>；normal 版让两者<strong>串行</strong>，CPU 组批时 GPU 空等，所以第 21 课用<strong>流水线</strong>把 CPU 部分藏进上一步 GPU 计算里",
                        "en": "<code>get_next_batch_to_run</code> / <code>process_batch_result</code> are <strong>pure-CPU lightweight accounting and decision</strong>, while <code>run_batch</code> (→ModelRunner.forward) is the <strong>GPU compute</strong>; the normal loop runs them <strong>serially</strong> so the GPU idles while the CPU forms the batch — hence Lesson 21 <strong>pipelines</strong> to hide the CPU part behind the previous step's GPU compute",
                    },
                    {"zh": "全部五步都在 GPU 上执行，没有 CPU 参与", "en": "All five steps run on the GPU; no CPU involvement"},
                    {"zh": "<code>run_batch</code> 是 CPU 决策，组批才是 GPU 计算", "en": "<code>run_batch</code> is the CPU decision, while forming the batch is the GPU compute"},
                    {"zh": "调度器本身就在 GPU 内核里跑，无所谓重叠", "en": "The scheduler itself runs inside the GPU kernel, so overlap is moot"},
                ],
                "answer": 0,
                "why": {
                    "zh": "调度器“只决策、不计算”：组批和收尾是 CPU 上的轻量记账，唯独 <code>run_batch</code> 把批交给 TpWorker→ModelRunner 在 GPU 上 forward（第 24 课）。normal 版严格串行，CPU 忙时 GPU 闲、反之亦然。由于循环速度直接给吞吐封顶，第 21 课的 overlap 版用结果队列把上一步收尾推迟、与本步 GPU 计算重叠，从而把 CPU 时间藏起来。",
                    "en": "The scheduler 'decides, never computes': batching and finishing are light CPU accounting, while only <code>run_batch</code> hands the batch to TpWorker→ModelRunner for GPU forward (Lesson 24). The normal loop is strictly serial — CPU busy means GPU idle and vice versa. Since loop speed caps throughput, Lesson 21's overlap version defers the previous step's finishing via a result queue to overlap this step's GPU compute, hiding the CPU time.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“机场塔台雷达扫描”的类比，完整描述调度器 <code>event_loop_normal</code> 一个 step 的全过程。请逐一对应五步：<code>recv_requests</code>（收 TokenizedGenerateReqInput，第 16 课）、<code>process_input_requests</code>（入等待队列、处理 abort/flush）、<code>get_next_batch_to_run</code>（prefill 优先否则 decode，第 19/20 课）、<code>run_batch</code>（→ModelRunner.forward，第 24 课）、<code>process_batch_result</code>（追加 token、检测完成、释放 KV、发 DetokenizerManager，第 17 课），并点明“回环”为什么是心跳的灵魂。",
                "en": "Using the 'air-traffic controller's radar sweep' analogy, describe one full step of the scheduler's <code>event_loop_normal</code>. Map each of the five stages: <code>recv_requests</code> (receive TokenizedGenerateReqInput, Lesson 16), <code>process_input_requests</code> (enqueue into waiting, handle abort/flush), <code>get_next_batch_to_run</code> (prefill first else decode, Lessons 19/20), <code>run_batch</code> (→ModelRunner.forward, Lesson 24), and <code>process_batch_result</code> (append tokens, detect finished, free KV, send to DetokenizerManager, Lesson 17), and explain why the 'loop-back' is the soul of the heartbeat.",
            },
            {
                "zh": "“循环速度直接给吞吐封顶”——请论证这句话，并解释为什么调度器是<strong>单线程、每 TP rank 唯一的决策者</strong>、独占 KV 缓存账本。在此基础上说明 <code>event_loop_overlap</code>（第 21 课）如何用流水线优化、长 prompt 为何需要分块预填充（第 22 课）来避免堵住这条循环。",
                "en": "Argue the claim 'loop speed directly caps throughput,' and explain why the scheduler is <strong>single-threaded, the sole decision-maker per TP rank</strong>, owning the KV-cache ledger. Building on that, explain how <code>event_loop_overlap</code> (Lesson 21) optimizes via pipelining, and why a long prompt needs chunked prefill (Lesson 22) to avoid clogging this loop.",
            },
        ],
    },
    "19-req-and-schedule-batch.html": {
        "mcq": [
            {
                "q": {
                    "zh": "<code>Req</code> 和 <code>ScheduleBatch</code> 在<strong>生命周期</strong>上最本质的区别是什么？",
                    "en": "What is the most essential <strong>lifecycle</strong> difference between <code>Req</code> and <code>ScheduleBatch</code>?",
                },
                "opts": [
                    {
                        "zh": "<code>Req</code> 是<strong>持久</strong>的——一条请求从 waiting 到 finished 一直活着；<code>ScheduleBatch</code> 是<strong>临时</strong>的——每个 step 重建一次、用完即弃，所以同一条 Req 会出现在许多<strong>连续的 decode 批</strong>里",
                        "en": "<code>Req</code> is <strong>persistent</strong>—one request stays alive from waiting to finished; <code>ScheduleBatch</code> is <strong>ephemeral</strong>—rebuilt every step and discarded, so the same Req appears in many <strong>successive decode batches</strong>",
                    },
                    {"zh": "两者都临时，每个 token 都重建一次，互为副本", "en": "Both are temporary, rebuilt per token, mirroring each other"},
                    {"zh": "<code>Req</code> 临时、<code>ScheduleBatch</code> 持久，批一旦建好就用到引擎关停", "en": "<code>Req</code> is temporary and <code>ScheduleBatch</code> persistent, the batch lasting until engine shutdown"},
                    {"zh": "两者都持久，整个会话只各有一个实例", "en": "Both are persistent, with exactly one instance each per session"},
                ],
                "answer": 0,
                "why": {
                    "zh": "这正是把两者拆开的原因：批是事件循环每步<strong>重新计算的结果</strong>（第 18 课），转瞬即逝；而一条请求要跨越几十上百拍 decode 才生成完，必须有个<strong>持久</strong>对象一路携带它的输出 token、KV 索引与状态。Req 持久、批临时，是 Part 5 的主轴。",
                    "en": "This is exactly why they are split: the batch is a <strong>recomputed result</strong> of each loop step (Lesson 18), ephemeral; while a request takes tens to hundreds of decode beats to finish, needing a <strong>persistent</strong> object to carry its output tokens, KV indices, and status throughout. Req persistent, batch ephemeral—the spine of Part 5.",
                },
            },
            {
                "q": {
                    "zh": "<code>ScheduleBatch</code> 的 <strong>EXTEND（prefill）批</strong>和 <strong>DECODE 批</strong>本质差别是什么？",
                    "en": "What is the essential difference between a <code>ScheduleBatch</code>'s <strong>EXTEND (prefill) batch</strong> and a <strong>DECODE batch</strong>?",
                },
                "opts": [
                    {
                        "zh": "EXTEND 由 <code>prepare_for_extend()</code> 构造，<strong>接纳新请求</strong>、一次吃掉每条整段 prompt（变长张量、为新请求分配 KV）；DECODE 由 <code>prepare_for_decode()</code> 构造，<strong>老请求每人各 +1 个 token</strong>（规整的每请求 1 token、各再要一个 KV 槽）",
                        "en": "EXTEND, built by <code>prepare_for_extend()</code>, <strong>admits new requests</strong> and eats each whole prompt at once (ragged tensors, allocate KV for new reqs); DECODE, built by <code>prepare_for_decode()</code>, gives <strong>each running request +1 token</strong> (regular 1-token-per-req, one more KV slot each)",
                    },
                    {"zh": "EXTEND 在 GPU 上跑，DECODE 在 CPU 上跑，互不相干", "en": "EXTEND runs on GPU, DECODE on CPU, unrelated to each other"},
                    {"zh": "两者完全一样，只是名字不同", "en": "They are identical, differing only in name"},
                    {"zh": "DECODE 接纳新请求并处理整段 prompt，EXTEND 只吐一个 token", "en": "DECODE admits new requests and processes whole prompts, EXTEND emits just one token"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<code>forward_mode</code> 决定这一拍“怎么算”。EXTEND 处理 prompt：长度各异 ⇒ 变长张量、为新请求分配 KV，且只算前缀没命中的那截（第 7 课）。DECODE 处理生成：每条只吐一个 token ⇒ 张量规整、各再要一个 KV 槽。一条请求一生只 extend 一次（或被切成几块，第 22 课），却 decode 很多次。",
                    "en": "<code>forward_mode</code> decides 'how' this beat computes. EXTEND handles the prompt: varying lengths ⇒ ragged tensors, allocate KV for new reqs, and only the prefix-missed part (Lesson 7). DECODE handles generation: each emits one token ⇒ regular tensors, one more KV slot each. A request extends once in its life (or split into chunks, Lesson 22) but decodes many times.",
                },
            },
            {
                "q": {
                    "zh": "<code>filter_batch()</code> 在调度里干了什么？为什么它对吞吐至关重要？",
                    "en": "What does <code>filter_batch()</code> do in scheduling, and why is it crucial for throughput?",
                },
                "opts": [
                    {
                        "zh": "它在飞行途中把<strong>已完成（finished）的请求当场从批里剔除</strong>、释放它们的 KV 槽，腾出的容量让等待队列的新请求补入——这就是<strong>连续批处理</strong>（第 5 课）的“腾槽”现场",
                        "en": "It <strong>drops finished requests from the batch on the spot</strong> mid-flight and frees their KV slots, so the freed capacity lets new requests from the waiting queue fill in—the 'slot-freeing' of <strong>continuous batching</strong> (Lesson 5)",
                    },
                    {"zh": "它把整批排序，让最长的请求排在最前面", "en": "It sorts the whole batch so the longest request comes first"},
                    {"zh": "它把批复制一份做备份，从不删除任何请求", "en": "It copies the batch as a backup and never removes any request"},
                    {"zh": "它只在引擎关停时调用一次，用来清空所有请求", "en": "It is called once at engine shutdown to clear all requests"},
                ],
                "answer": 0,
                "why": {
                    "zh": "源码里 <code>filter_batch</code> 用 <code>keep_indices</code> 只保留 <code>not finished()</code> 的请求，重建 <code>reqs</code> 与对应张量。若不在途中剔除完成者，整批就退化成静态批处理——要等最慢那条、完成的槽位空转。当场清退 + 释放 KV + 接纳等待者，批才永远满载，这是高吞吐的根。",
                    "en": "In source, <code>filter_batch</code> keeps only requests that are <code>not finished()</code> via <code>keep_indices</code>, rebuilding <code>reqs</code> and the matching tensors. Without dropping finished ones mid-flight, the batch degrades to static batching—waiting for the slowest, finished slots idle. Evict on the spot + free KV + admit waiters keeps the batch full, the root of high throughput.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“航空公司：乘客（Req）vs 航班舱单（ScheduleBatch）”的类比，完整讲清这两个数据结构的分工。请覆盖：Req 携带哪些字段（输入/输出 token、采样参数、KV 索引、prefix_indices、finished 状态）、它的 waiting→running→finished 生命周期；以及 ScheduleBatch 为何<strong>每步重建</strong>、它的 EXTEND/DECODE 两种 <code>forward_mode</code> 各自如何铺张量与分配 KV。",
                "en": "Using the 'airline: passenger (Req) vs flight manifest (ScheduleBatch)' analogy, explain the division of labor between the two data structures. Cover: which fields Req carries (input/output tokens, sampling params, KV indices, prefix_indices, finished status) and its waiting→running→finished lifecycle; and why ScheduleBatch is <strong>rebuilt each step</strong>, plus how its two <code>forward_mode</code>s (EXTEND/DECODE) lay out tensors and allocate KV.",
            },
            {
                "zh": "顺着这一课往后想：<code>get_next_batch_to_run</code>（第 18/20 课）每步要在“组一个 EXTEND 批接纳新请求”和“组一个 DECODE 批推进老请求”之间抉择。请说明这个抉择如何与 KV 池容量、等待队列长度、<code>filter_batch</code> 腾出的空槽相互作用，并联系分块预填充（第 22 课）为何能避免一个超长 prompt 把这条循环堵死。",
                "en": "Thinking forward from this lesson: <code>get_next_batch_to_run</code> (Lessons 18/20) must choose each step between 'form an EXTEND batch to admit new requests' and 'form a DECODE batch to advance running ones.' Explain how this choice interacts with KV-pool capacity, waiting-queue length, and the empty slots freed by <code>filter_batch</code>, and connect it to why chunked prefill (Lesson 22) prevents one very long prompt from clogging this loop.",
            },
        ],
    },
    "20-schedule-policy.html": {
        "mcq": [
            {
                "q": {
                    "zh": "缓存感知的 <strong>LPM（最长前缀匹配）</strong>排序和 <strong>FCFS</strong> 各自优化的是什么？",
                    "en": "What does cache-aware <strong>LPM (longest-prefix-match)</strong> ordering optimize, versus <strong>FCFS</strong>?",
                },
                "opts": [
                    {
                        "zh": "LPM 优化<strong>缓存命中率</strong>——把 prompt 前缀已在 RadixAttention 缓存里的请求顶到前面、接纳它们几乎免费；FCFS 优化<strong>公平性</strong>——严格按到达顺序，简单可预测",
                        "en": "LPM optimizes <strong>cache hit rate</strong>—bumping requests whose prompt prefix is already in the RadixAttention cache to the front so admitting them is nearly free; FCFS optimizes <strong>fairness</strong>—strictly by arrival, simple and predictable",
                    },
                    {"zh": "两者都优化显存占用，和缓存无关", "en": "Both optimize memory usage, unrelated to the cache"},
                    {"zh": "LPM 优化公平、FCFS 优化命中率，正好相反", "en": "LPM optimizes fairness and FCFS optimizes hit rate, exactly reversed"},
                    {"zh": "两者都只优化单请求延迟，不影响吞吐", "en": "Both only optimize single-request latency, with no effect on throughput"},
                ],
                "answer": 0,
                "why": {
                    "zh": "LPM 让“开头已缓存”的请求插队，相同前缀只算一次，直接抬高 RadixAttention 命中率，这是真实流量提速的来源（第 7 课）；FCFS 不看缓存、只按到达顺序，换来的是公平、简单与可预测的延迟。队列过长时 LPM 还会临时退回 FCFS 以免排序成瓶颈。",
                    "en": "LPM lets cached-opening requests jump ahead so a shared prefix is computed once, directly raising the RadixAttention hit rate—the source of real-traffic speedups (Lesson 7); FCFS ignores the cache and goes by arrival, buying fairness, simplicity and predictable latency. On long queues LPM even falls back to FCFS so sorting never becomes the bottleneck.",
                },
            },
            {
                "q": {
                    "zh": "<code>PrefillAdder</code> 决定“这一拍能塞几个请求”时，靠的是哪<strong>两个预算</strong>？",
                    "en": "When <code>PrefillAdder</code> decides 'how many requests fit this beat', which <strong>two budgets</strong> does it use?",
                },
                "opts": [
                    {
                        "zh": "<strong>token 预算</strong>（别让这一步 prefill 太大，连着分块预填充）+ <strong>显存预算</strong>（KV 池要有足够空闲槽，绝不接纳放不下的请求）；任一耗尽即停",
                        "en": "A <strong>token budget</strong> (keep this prefill step from getting too big, tied to chunked prefill) + a <strong>memory budget</strong> (enough free KV-pool slots, never admit a req that won't fit); stop when either runs out",
                    },
                    {"zh": "CPU 核数预算 + 网络带宽预算", "en": "A CPU-core budget + a network-bandwidth budget"},
                    {"zh": "磁盘空间预算 + 进程数预算", "en": "A disk-space budget + a process-count budget"},
                    {"zh": "只有一个 token 预算，显存从不参与判断", "en": "Only a token budget; memory never enters the decision"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<code>add_one_req</code> 算出 <code>total_tokens = cand_extend_input_len + max_new + page_size</code>，若 <code>≥ rem_total_tokens</code> 就 <code>NO_TOKEN</code>；同时检查 KV 池空闲槽（第 4/30 课）。token 预算限制单拍 prefill 规模（第 22 课分块），显存预算保证放得下——两把尺子任一满即停手。",
                    "en": "<code>add_one_req</code> computes <code>total_tokens = cand_extend_input_len + max_new + page_size</code>; if <code>≥ rem_total_tokens</code> it returns <code>NO_TOKEN</code>, and it also checks free KV-pool slots (Lessons 4/30). The token budget caps the per-beat prefill size (chunking, Lesson 22), the memory budget guarantees it fits—whichever yardstick fills first stops admission.",
                },
            },
            {
                "q": {
                    "zh": "调度器每一拍面对的“<strong>prefill 与 decode 的接纳张力</strong>”指的是什么？",
                    "en": "What is the '<strong>prefill-vs-decode admission tension</strong>' the scheduler faces each beat?",
                },
                "opts": [
                    {
                        "zh": "多接纳新 prefill 能<strong>增吞吐</strong>，却会<strong>抬高在跑请求的延迟</strong>；只推进老 decode 则<strong>保延迟</strong>但少了新吞吐——策略每拍在两者间权衡",
                        "en": "Admitting more new prefills <strong>grows throughput</strong> but <strong>raises latency for running decodes</strong>; just advancing old decodes <strong>keeps latency low</strong> but adds no new throughput—the policy balances the two each beat",
                    },
                    {"zh": "prefill 和 decode 必须在不同 GPU 上跑，调度器只是选卡", "en": "Prefill and decode must run on different GPUs; the scheduler just picks the card"},
                    {"zh": "两者完全独立，从不互相影响", "en": "They are fully independent and never affect each other"},
                    {"zh": "张力指的是 CPU 和磁盘之间的带宽争用", "en": "The tension refers to bandwidth contention between CPU and disk"},
                ],
                "answer": 0,
                "why": {
                    "zh": "prefill 吃整段 prompt、占算力，能把新请求拉进批里增吞吐，但和在跑的 decode 抢资源、抬高它们的延迟；只 decode 则延迟稳但吞吐不长。调度器每拍用策略在这架天平上加砝码（第 18/8 课），这正是组批决策的核心权衡。",
                    "en": "Prefill eats whole prompts and burns compute, pulling new requests into the batch to grow throughput, but it contends with running decodes and raises their latency; decode-only keeps latency steady but adds no throughput. The scheduler weights this scale with policy each beat (Lessons 18/8)—the core trade-off of batch formation.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“聪明领位员（host / bouncer）”的类比讲清调度策略的两个动作：<strong>排序</strong>（<code>SchedulePolicy.calc_priority</code> 怎样按 LPM / FCFS / priority 就地重排等待队列）与<strong>限流</strong>（<code>PrefillAdder</code> 怎样在 token + 显存双预算下逐个 <code>add_one_req</code>、必要时切块）。再说明为什么“缓存感知排序”会直接抬高 RadixAttention 命中率，从而体现调度器与缓存的<strong>协同设计</strong>。",
                "en": "Using the 'smart host / bouncer' analogy, explain the two actions of the schedule policy: <strong>ordering</strong> (how <code>SchedulePolicy.calc_priority</code> reorders the waiting queue in place by LPM / FCFS / priority) and <strong>throttling</strong> (how <code>PrefillAdder</code> calls <code>add_one_req</code> one by one under the token + memory double budget, chunking when needed). Then explain why cache-aware ordering directly raises the RadixAttention hit rate, illustrating the scheduler-cache <strong>co-design</strong>.",
            },
            {
                "zh": "顺着这一课往后想：若你的线上流量里成千上万请求共享同一段超长系统提示，你会选 LPM 还是 FCFS？为什么？再结合 <code>PrefillAdder</code> 的双预算，说明一个超长 prompt 如何被 token 预算与<strong>分块预填充（第 22 课）</strong>挡住、避免它独占一拍把 decode 全堵死，并联系 KV 池容量（第 4/30 课）与吞吐 / 延迟（第 8 课）的权衡。",
                "en": "Thinking forward: if your production traffic has thousands of requests sharing one very long system prompt, would you pick LPM or FCFS, and why? Then, using <code>PrefillAdder</code>'s double budget, explain how a very long prompt is held back by the token budget and <strong>chunked prefill (Lesson 22)</strong>—so it can't monopolize a beat and clog all decodes—connecting it to KV-pool capacity (Lessons 4/30) and the throughput/latency trade-off (Lesson 8).",
            },
        ],
    },
    "21-zero-overhead-overlap-scheduler.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 <code>event_loop_overlap</code> 里，<strong>什么和什么重叠</strong>？请精确说出哪部分在 GPU、哪部分在 CPU。",
                    "en": "In <code>event_loop_overlap</code>, <strong>what overlaps with what</strong>? Name precisely which part runs on the GPU and which on the CPU.",
                },
                "opts": [
                    {
                        "zh": "第 N 步的 <strong>GPU 前向</strong>（<code>run_batch</code>→ModelRunner.forward）与 <strong>CPU 组第 N+1 步的批</strong> + <strong>CPU 收尾第 N−1 步的结果</strong>（<code>process_batch_result</code>）<strong>同拍并行</strong>：发射本步前向后不等它，CPU 立刻去组下一批、并从 <code>result_queue</code> 取出上一步结果做收尾",
                        "en": "Step N's <strong>GPU forward</strong> (<code>run_batch</code>→ModelRunner.forward) runs <strong>in the same beat as</strong> the <strong>CPU forming step N+1's batch</strong> plus the <strong>CPU finishing step N−1's result</strong> (<code>process_batch_result</code>): after launching this step's forward without waiting, the CPU immediately forms the next batch and pops the previous step's result from <code>result_queue</code> to finish it",
                    },
                    {"zh": "两个不同请求的 GPU 前向在同一张卡上并行，CPU 不参与", "en": "Two different requests' GPU forwards run in parallel on one card; the CPU is uninvolved"},
                    {"zh": "GPU 前向与 GPU 采样重叠，全程没有 CPU 工作", "en": "The GPU forward overlaps with GPU sampling, with no CPU work at all"},
                    {"zh": "本步的 CPU 收尾与本步的 GPU 前向重叠（同一步内并行）", "en": "This step's CPU finishing overlaps with this step's own GPU forward (parallel within the same step)"},
                ],
                "answer": 0,
                "why": {
                    "zh": "调度器“只决策、不计算”：组批/采样/收尾是 CPU 轻量活，唯独 <code>run_batch</code> 烧 GPU。overlap 版发射第 N 步前向后<strong>立刻 append 进 <code>result_queue</code> 不阻塞</strong>，趁 GPU 算 N 的当口，CPU 去组 N+1 的批、并 <code>popleft</code> 出 N−1 的结果做收尾。于是每一拍的 CPU 时间都藏进同拍 GPU 计算的影子里。",
                    "en": "The scheduler 'decides, never computes': forming/sampling/finishing are light CPU work, while only <code>run_batch</code> burns GPU. The overlap loop launches step N's forward and <strong>appends to <code>result_queue</code> without blocking</strong>; while the GPU computes N, the CPU forms N+1's batch and <code>popleft</code>s N−1's result to finish it. So each beat's CPU time hides in the same beat's GPU compute shadow.",
                },
            },
            {
                "q": {
                    "zh": "为什么这套重叠能把<strong>调度开销藏到近乎为零</strong>，让人称它“零开销”？",
                    "en": "Why does this overlap hide <strong>scheduling overhead down to near zero</strong>, earning the name 'zero-overhead'?",
                },
                "opts": [
                    {
                        "zh": "因为 CPU 的组批/采样/记账时间，正好落在 GPU 那一拍前向计算的“影子”里被<strong>完全遮住</strong>——GPU 从不为等 CPU 而空转，CPU 开销不再出现在关键路径上，所以对外表现为近乎零的调度开销、GPU 满载",
                        "en": "Because the CPU's forming/sampling/bookkeeping time falls inside the 'shadow' of that beat's GPU forward and is <strong>fully masked</strong>—the GPU never idles waiting on the CPU, so the CPU cost leaves the critical path, presenting as near-zero scheduling overhead with the GPU saturated",
                    },
                    {"zh": "因为重叠让 CPU 的调度代码运行得更快了", "en": "Because overlap makes the CPU scheduling code itself run faster"},
                    {"zh": "因为它彻底删掉了组批和采样这两步", "en": "Because it deletes the batch-forming and sampling steps entirely"},
                    {"zh": "因为前向被搬到了 CPU 上，省掉了 GPU", "en": "Because the forward is moved onto the CPU, eliminating the GPU"},
                ],
                "answer": 0,
                "why": {
                    "zh": "“零开销”不是没成本，而是把成本<strong>藏到不影响吞吐的地方</strong>。串行版里 CPU 干活时 GPU 空等，这段空等直接给吞吐封顶；overlap 把 CPU 调度与 GPU 前向并到同一拍，CPU 时间被 GPU 计算遮住，GPU 利用率拉满。decode 步 GPU 极快、CPU 占比大，收益尤其明显。",
                    "en": "'Zero-overhead' isn't cost-free; it moves the cost <strong>somewhere that doesn't dent throughput</strong>. In the serial version the GPU idles while the CPU works, and that idle caps throughput; overlap puts CPU scheduling and the GPU forward in the same beat, masking CPU time behind GPU compute and saturating utilization. The win is biggest on decode, where the GPU is very fast and the CPU share is large.",
                },
            },
            {
                "q": {
                    "zh": "重叠调度器的<strong>主要代价</strong>是什么？为什么需要一个 <code>result_queue</code>？",
                    "en": "What is the overlap scheduler's <strong>main cost</strong>, and why is a <code>result_queue</code> needed?",
                },
                "opts": [
                    {
                        "zh": "代价是<strong>多一拍延迟</strong>——你处理的永远是<strong>上一步</strong>的结果，总比实时慢一步（第 8 课的吞吐/延迟权衡），外加跨拍状态管理更难；<code>result_queue</code> 正是用来暂存“已发射但还没收尾”的批，把收尾<strong>推迟一拍</strong>，等 GPU 算完再 <code>popleft</code> 处理",
                        "en": "The cost is <strong>one extra beat of latency</strong>—you always process the <strong>previous</strong> step's result, forever one step behind real time (Lesson 8's throughput/latency trade-off), plus harder cross-beat state; the <code>result_queue</code> holds 'launched but not yet finished' batches, <strong>deferring the finish one beat</strong> so it's <code>popleft</code>-ed once the GPU is done",
                    },
                    {"zh": "代价是吞吐下降，<code>result_queue</code> 用来限流", "en": "The cost is lower throughput; the <code>result_queue</code> throttles intake"},
                    {"zh": "没有代价，<code>result_queue</code> 只是日志缓冲", "en": "There is no cost; the <code>result_queue</code> is just a logging buffer"},
                    {"zh": "代价是显存翻倍，<code>result_queue</code> 缓存 KV", "en": "The cost is doubled VRAM; the <code>result_queue</code> caches KV"},
                ],
                "answer": 0,
                "why": {
                    "zh": "源码里 <code>run_batch</code> 后立刻 <code>result_queue.append((batch.copy(), result))</code> 不阻塞，下一拍才 <code>popleft</code> 做 <code>process_batch_result</code>——所以收尾恒落后一拍，这就是 +1 拍延迟。下一批要在当前结果未出时就搭好，采样 token / KV 记账存在跨拍依赖，靠 <code>batch_overlap</code> 的 future/event 串好。某些必须先拿到上一步结果的情形会临时退回不重叠。",
                    "en": "In source, right after <code>run_batch</code> we <code>result_queue.append((batch.copy(), result))</code> without blocking, and only the next beat <code>popleft</code>s it for <code>process_batch_result</code>—so finishing always lags one beat, hence +1 latency. The next batch is built before the current result exists, giving sampled-token / KV-bookkeeping cross-beat dependencies threaded via <code>batch_overlap</code>'s future/event objects. Cases needing the prior result first fall back to no-overlap temporarily.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“接力赛 / 两段式流水线”的类比，完整讲清 <code>event_loop_overlap</code> 一个 step 的错位流水。请逐一覆盖：①<strong>发射</strong>第 N 步前向（<code>run_batch</code> 不等结果、把 <code>(batch.copy(), result)</code> 压进 <code>result_queue</code>）；②趁 GPU 算 N，CPU 组第 N+1 步的批；③<code>pop_and_process</code> 收尾第 N−1 步（<code>process_batch_result</code>：追加 token、判完成、释放 KV、发反分词第 17 课）。并解释“慢一拍”为何既是代价又是“零开销”的来源，对照朴素 <code>event_loop_normal</code>（第 18 课）说明 GPU 为什么从不空转。",
                "en": "Using the 'relay race / two-stage assembly line' analogy, walk through one step of <code>event_loop_overlap</code>'s staggered pipeline. Cover each: (1) <strong>launch</strong> step N's forward (<code>run_batch</code> without waiting, push <code>(batch.copy(), result)</code> onto <code>result_queue</code>); (2) while the GPU computes N, the CPU forms step N+1's batch; (3) <code>pop_and_process</code> finishes step N−1 (<code>process_batch_result</code>: append tokens, detect finished, free KV, send to detok, Lesson 17). Explain why being 'one beat behind' is both the cost and the source of 'zero-overhead,' and, contrasting naive <code>event_loop_normal</code> (Lesson 18), why the GPU never idles.",
            },
            {
                "zh": "论证“重叠调度器与连续批处理是绝配”：连续批处理（第 5 课）让批<strong>永远满载</strong>，重叠调度器让<strong>喂满批的调度不要钱</strong>。请说明两者各解决吞吐的哪一半，为什么 decode 密集场景下重叠收益最大（联系第 8 课吞吐/延迟、第 24 课模型前向），以及它如何与 CUDA Graph（第 27 课）叠加把单步成本进一步压薄。再讨论：在什么样的负载或延迟敏感场景下，+1 拍延迟可能是不可接受的取舍？",
                "en": "Argue that 'the overlap scheduler pairs perfectly with continuous batching': continuous batching (Lesson 5) keeps the batch <strong>always full</strong>, while the overlap scheduler makes <strong>keeping it full free</strong>. Explain which half of the throughput problem each solves, why the overlap win is largest in decode-heavy regimes (tie to Lesson 8 throughput/latency and Lesson 24 model forward), and how it stacks with CUDA Graph (Lesson 27) to further thin the per-step cost. Then discuss: under what workloads or latency-sensitive scenarios might the +1-beat latency be an unacceptable trade-off?",
            },
        ],
    },
    "22-chunked-prefill.html": {
        "mcq": [
            {
                "q": {
                    "zh": "为什么不分块时，一个 32k token 的超长 prompt 会成为大问题？",
                    "en": "Without chunking, why does a 32k-token huge prompt become a big problem?",
                },
                "opts": [
                    {
                        "zh": "因为它若在<strong>一拍里整段 prefill</strong>，这一步的前向计算量暴涨、<strong>独占 GPU 几十毫秒</strong>，批里所有正在 decode 的请求<strong>全部 stall</strong>、token 吐不出来，于是 TTFT/ITL 延迟尖峰（第 8 课）——一头鲸鱼堵死整池",
                        "en": "Because if it is <strong>prefilled whole in one step</strong>, that step's forward compute explodes and <strong>monopolizes the GPU for tens of ms</strong>; every decoding request in the batch <strong>stalls</strong>, no tokens emitted, so TTFT/ITL spike (Lesson 8)—one whale clogs the whole pool",
                    },
                    {"zh": "因为 32k token 的 KV 缓存一定放不进显存，必然 OOM", "en": "Because a 32k-token KV cache can never fit in VRAM and must OOM"},
                    {"zh": "因为长 prompt 会让采样温度失效，输出乱码", "en": "Because a long prompt breaks sampling temperature, producing garbage output"},
                    {"zh": "因为调度器无法对超过 4k token 的请求排序", "en": "Because the scheduler cannot order requests longer than 4k tokens"},
                ],
                "answer": 0,
                "why": {
                    "zh": "这是<strong>时间维度的拥塞</strong>，不是显存问题——KV 槽位也许够，但“一拍算太多”让 GPU 被一个请求独占，批里其他人的 decode 只能干等。用户观感就是流式输出突然顿住（ITL 尖峰）、新请求首字变慢（TTFT 尖峰）。分块正是为摊平这种尖峰而生。",
                    "en": "It's <strong>congestion along the time axis</strong>, not a memory problem—KV slots may be plenty, but 'too much compute in one step' lets one request monopolize the GPU while everyone else's decode waits. Users see streaming freeze (ITL spike) and slower first tokens (TTFT spike). Chunking exists to flatten exactly this spike.",
                },
            },
            {
                "q": {
                    "zh": "分块预填充<strong>具体怎么工作</strong>，才能既推进长请求、又不卡住别人？",
                    "en": "How does chunked prefill <strong>actually work</strong> so it advances the long request without stalling others?",
                },
                "opts": [
                    {
                        "zh": "把大 prefill 切成<strong>固定大小的 token 块</strong>（由 <code>chunked_prefill_size</code> 设定），分摊到<strong>多拍</strong>，并和别人的 decode <strong>混在同一个批</strong>里前向；KV 一块块填，只有<strong>最后一块</strong>填完该请求才转入 decode",
                        "en": "Split the big prefill into <strong>fixed-size token chunks</strong> (set by <code>chunked_prefill_size</code>) across <strong>several steps</strong>, forwarded <strong>in the same batch mixed with</strong> others' decode; KV fills chunk by chunk, and only after the <strong>last chunk</strong> does the request enter decode",
                    },
                    {"zh": "把长请求挪到队列最末尾，等所有 decode 都结束再单独整段 prefill", "en": "Move the long request to the very end of the queue and prefill it whole only after all decodes finish"},
                    {"zh": "把 prompt 压缩到 4k token 以内再一次性 prefill", "en": "Compress the prompt to under 4k tokens then prefill it all at once"},
                    {"zh": "为长请求单独开一张 GPU，和 decode 物理隔离", "en": "Give the long request its own dedicated GPU, physically isolated from decode"},
                ],
                "answer": 0,
                "why": {
                    "zh": "关键有二：① <strong>切块</strong>让每拍 prefill 有界、没有哪一步巨大；② <strong>混合批</strong>让这一个 chunk 和一堆 decode 同框前向，于是每拍 = 一个有界 prefill 块 + 全体各吐一个 token，规模可控、节奏稳定。排到最后会饿死长请求、首字延迟无限长，所以不可取。",
                    "en": "Two keys: (1) <strong>chunking</strong> bounds prefill per step so no step is huge; (2) the <strong>mixed batch</strong> forwards that one chunk together with a crowd of decodes, so each step = one bounded prefill chunk + one token from everyone—controlled and steady. Queuing it last would starve the long request with unbounded first-token latency, so that's no good.",
                },
            },
            {
                "q": {
                    "zh": "<code>chunked_prefill_size</code> 这个旋钮调小或调大，分别意味着什么取舍？",
                    "en": "Tuning the <code>chunked_prefill_size</code> knob smaller vs larger means what trade-off?",
                },
                "opts": [
                    {
                        "zh": "<strong>调小</strong>：每拍 prefill 更轻、decode 更顺滑，但<strong>总步数更多</strong>、长请求自己的首字更慢；<strong>调大</strong>：步数更少、长请求更快收尾，但每拍更重、<strong>尖峰风险回升</strong>（退回第 8 课吞吐/延迟权衡）",
                        "en": "<strong>Smaller</strong>: lighter prefill per step and smoother decode, but <strong>more total steps</strong> and a slower whale first byte; <strong>larger</strong>: fewer steps and a faster whale finish, but heavier steps and <strong>the spike risk returns</strong> (back to Lesson 8's throughput/latency trade-off)",
                    },
                    {"zh": "调小提升吞吐、调大降低吞吐，与延迟无关", "en": "Smaller raises throughput, larger lowers it, with no latency effect"},
                    {"zh": "它只控制日志频率，对性能没有影响", "en": "It only controls logging frequency and has no performance effect"},
                    {"zh": "它决定 KV 缓存的页大小（page_size），越大越省显存", "en": "It sets the KV cache page size, where larger saves more VRAM"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<code>chunked_prefill_size</code> 设定每拍能吃多少 prefill token，正是平滑与步数之间那把旋钮。没有放之四海的最优值——延迟敏感、decode 密集就偏小；吞吐优先、长请求为主就偏大。它控制的是“每一拍 prefill 的有界上限”，而非显存页大小或日志。",
                    "en": "<code>chunked_prefill_size</code> sets how many prefill tokens fit per step—the very knob between smoothness and step count. There's no universal optimum: go smaller for latency-sensitive, decode-heavy loads; larger for throughput-first, long-prompt-heavy ones. It controls the bounded per-step prefill cap, not the VRAM page size or logging.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“吃巨无霸大餐还要继续聊天”的类比，完整讲清分块预填充。请逐一覆盖：①<strong>问题</strong>——一个 32k prompt 若一拍整段 prefill，会独占 GPU、让批里所有 decode <strong>stall</strong>，造成 TTFT/ITL 尖峰（第 8 课）；②<strong>做法</strong>——<code>PrefillAdder</code>（第 20 课）在 <code>add_one_req</code> 里执行每拍 token 预算，装不下就把请求<strong>钳到剩余预算</strong>（<code>trunc_len</code> 按 page 对齐）、记为 <code>new_chunked_req</code> 下一拍续，KV 一块块填（第 4 课），最后一块才转 decode；③<strong>混合批</strong>——这一个 chunk 和别人的 decode 同批前向，事件循环（第 18 课）照常推进。",
                "en": "Using the 'eating a giant meal while still chatting' analogy, fully explain chunked prefill. Cover each: (1) the <strong>problem</strong>—a 32k prompt prefilled whole in one step monopolizes the GPU and <strong>stalls</strong> every decode in the batch, causing TTFT/ITL spikes (Lesson 8); (2) the <strong>fix</strong>—<code>PrefillAdder</code> (Lesson 20) enforces the per-step token budget in <code>add_one_req</code>, clamping a non-fitting request to the remaining budget (<code>trunc_len</code>, page-aligned), recording <code>new_chunked_req</code> for next step, filling KV chunk by chunk (Lesson 4), entering decode only after the last chunk; (3) the <strong>mixed batch</strong>—that one chunk is forwarded alongside others' decode, the event loop (Lesson 18) advancing as usual.",
            },
            {
                "zh": "讨论分块预填充的<strong>取舍</strong>与它在系统里的位置。分块<strong>多花几拍</strong>、长请求自己的 TTFT 被摊薄，却换来<strong>全局延迟平滑</strong>与更高的达标吞吐（goodput-under-SLA）——请说明为什么在长短请求混跑的真实负载里这笔买卖通常值得，<code>chunked_prefill_size</code> 偏小/偏大各自的代价（联系第 8 课），以及为什么把 prefill 与 decode 混在一拍里仍有张力，从而引出 PD 分离（第 45 课）把两者彻底拆到不同实例的动机。",
                "en": "Discuss chunked prefill's <strong>trade-off</strong> and its place in the system. Chunking costs <strong>a few extra steps</strong> and thins the whale's own TTFT, but buys <strong>smooth global latency</strong> and higher goodput-under-SLA—explain why this is usually worth it in real mixed long/short workloads, the costs of <code>chunked_prefill_size</code> being too small vs too large (tie to Lesson 8), and why mixing prefill and decode in one step still has tension, motivating PD disaggregation (Lesson 45) that splits the two onto separate instances entirely.",
            },
        ],
    },
    "23-dp-controller-and-pp-scheduling.html": {
        "mcq": [
            {
                "q": {
                    "zh": "<strong>数据并行 DP</strong> 里，每个副本和 <code>DataParallelController</code> 各自扮演什么角色？",
                    "en": "In <strong>data parallel (DP)</strong>, what role does each replica vs the <code>DataParallelController</code> play?",
                },
                "opts": [
                    {
                        "zh": "每个副本是一台<strong>完整运行时</strong>（自带调度器+TP worker+KV 缓存+<strong>一整份模型</strong>），副本间请求级隔离、互不交互；控制器<strong>不做前向</strong>，只按轮询/负载把进来的请求<strong>扇出</strong>给某个就绪副本",
                        "en": "Each replica is a <strong>full runtime</strong> (its own scheduler+TP worker+KV cache+<strong>a full model copy</strong>), request-isolated and non-interacting; the controller does <strong>no forward</strong>, only <strong>fanning out</strong> incoming requests round-robin/load-aware to a ready replica",
                    },
                    {"zh": "副本之间共享同一份模型权重，控制器负责合并它们的输出", "en": "Replicas share one set of weights, and the controller merges their outputs"},
                    {"zh": "控制器把每个请求切成多段，分给不同副本各算一段再拼回", "en": "The controller splits each request into pieces, one per replica, then stitches them back"},
                    {"zh": "副本只持有部分层，控制器负责在副本间传递激活值", "en": "Each replica holds only some layers, and the controller passes activations between them"},
                ],
                "answer": 0,
                "why": {
                    "zh": "DP 是<strong>复制整个运行时</strong>：每副本一整份模型、各跑各的事件循环、各管各的 KV 账本，请求落到哪个副本就由那个副本独立处理到底。控制器只做一件极轻的事——“分给谁”，从不碰 GPU。切层分激活那是 PP；共享权重合输出并非 DP 的工作方式。",
                    "en": "DP <strong>replicates the whole runtime</strong>: each replica holds a full model, runs its own event loop, manages its own KV ledger, and whatever replica a request lands on handles it end-to-end. The controller does one ultra-light thing—'to whom'—never touching the GPU. Splitting layers/passing activations is PP; sharing weights and merging outputs is not how DP works.",
                },
            },
            {
                "q": {
                    "zh": "<strong>流水线并行 PP</strong> 为什么要让<strong>多个 micro-batch 同时在飞</strong>，而不是一次只跑一个 batch？",
                    "en": "Why does <strong>pipeline parallel (PP)</strong> keep <strong>multiple micro-batches in flight</strong> instead of running one batch at a time?",
                },
                "opts": [
                    {
                        "zh": "因为层被切成段分到多卡，若只跑一个 batch，则某段工作时其它段全<strong>闲等</strong>（流水线<strong>气泡</strong>）；让多个 micro-batch 错峰流动，可使 stage1 跑 B 的同时 stage2 跑 A，<strong>各段都忙</strong>、气泡被摊薄",
                        "en": "Because layers are split into stages across cards; with one batch, while one stage works the others <strong>idle</strong> (the pipeline <strong>bubble</strong>). Staggering multiple micro-batches lets stage1 run B while stage2 runs A, so <strong>all stages stay busy</strong> and the bubble is amortized",
                    },
                    {"zh": "因为多个 micro-batch 能共享同一份 KV 缓存，省显存", "en": "Because multiple micro-batches can share one KV cache to save VRAM"},
                    {"zh": "因为这样可以跳过段间通信，避免传激活值", "en": "Because it skips inter-stage communication and avoids passing activations"},
                    {"zh": "因为单个 batch 无法被采样，必须拆成 micro-batch", "en": "Because a single batch cannot be sampled and must be split into micro-batches"},
                ],
                "answer": 0,
                "why": {
                    "zh": "PP 的瓶颈是<strong>填充/排空气泡</strong>：接力式前向里，单 batch 时总有段在空等。多 micro-batch 错峰让管线进入<strong>稳态满载</strong>（不同 stage 处理不同 micro-batch），micro-batch 越多、满载占比越高、气泡占比越趋近零。这也是 <code>event_loop_pp</code> 必须比 normal 循环复杂的原因——它要同时追踪一串在不同段的 micro-batch。",
                    "en": "PP's bottleneck is the <strong>fill/drain bubble</strong>: in relay-style forward, with one batch some stage always idles. Staggered micro-batches drive the pipe into <strong>steady-state full load</strong> (different stages on different micro-batches); more micro-batches ⇒ higher full-load fraction ⇒ bubble approaching zero. That's why <code>event_loop_pp</code> must be more complex than the normal loop—it tracks a string of micro-batches at different stages.",
                },
            },
            {
                "q": {
                    "zh": "把 <strong>TP、PP、DP</strong> 一句话区分，哪种说法最准确？",
                    "en": "Distinguishing <strong>TP, PP, DP</strong> in one line, which statement is most accurate?",
                },
                "opts": [
                    {
                        "zh": "<strong>TP</strong> 切<strong>一层内的矩阵</strong>、<strong>PP</strong> 切<strong>跨层的段</strong>、<strong>DP</strong> 复制<strong>整个副本</strong>；调度器编排 DP（控制器分发）与 PP（pp 循环），而 TP 藏在<strong>模型前向内部</strong>、对调度器基本透明",
                        "en": "<strong>TP</strong> splits <strong>matrices within a layer</strong>, <strong>PP</strong> splits <strong>stages across layers</strong>, <strong>DP</strong> replicates <strong>whole replicas</strong>; the scheduler orchestrates DP (controller dispatch) and PP (pp loop), while TP hides <strong>inside the model forward</strong>, largely transparent to the scheduler",
                    },
                    {"zh": "三者都是切分模型的层，只是切的段数不同", "en": "All three split the model's layers, differing only in stage count"},
                    {"zh": "三者都由 DataParallelController 统一编排", "en": "All three are orchestrated uniformly by the DataParallelController"},
                    {"zh": "TP 复制整模型、PP 复制整模型、DP 切矩阵", "en": "TP replicates the whole model, PP replicates the whole model, DP splits matrices"},
                ],
                "answer": 0,
                "why": {
                    "zh": "三维正交，靠“复制什么/切分什么”区分：DP=整模型复制（副本级）、PP=层切段（stage 级）、TP=单层矩阵切分。大部署是 TP×PP×DP 叠加。关键是分清<strong>调度</strong>问题（DP/PP 改变控制流，归 Part 5）与<strong>计算</strong>问题（TP 在 forward 内部，第 24/46 课）。",
                    "en": "Three orthogonal dims, told apart by 'what it replicates/splits': DP=whole-model replicate (replica level), PP=layers into stages (stage level), TP=split a single layer's matrices. Big deployments stack TP×PP×DP. The key is separating the <strong>scheduling</strong> problem (DP/PP change control flow, hence Part 5) from the <strong>compute</strong> problem (TP inside forward, Lessons 24/46).",
                },
            },
        ],
        "open": [
            {
                "zh": "用“超市收银通道 vs 传送带工位”的类比，完整讲清 DP 与 PP 的区别。请逐一覆盖：①<strong>DP</strong>——为何复制<strong>整个运行时</strong>（每副本自带调度器+TP+KV+一整份模型）、<code>DataParallelController</code> 如何用 <code>round_robin_scheduler</code> 把请求<strong>轮询扇出</strong>且自己不碰 GPU、为什么它是<strong>吞吐乘法器</strong>但救不了“装不下”的模型；②<strong>PP</strong>——为何把<strong>层切成段</strong>能装下超大模型、<code>event_loop_pp</code> 为何要让<strong>多个 micro-batch 错峰在飞</strong>、<strong>填充/排空气泡</strong>从哪来又怎么靠 micro-batch 数摊平。",
                "en": "Using the 'supermarket checkout lanes vs conveyor stations' analogy, fully explain DP vs PP. Cover each: (1) <strong>DP</strong>—why it replicates the <strong>whole runtime</strong> (each replica with its own scheduler+TP+KV+a full model), how <code>DataParallelController</code> <strong>round-robins/fans out</strong> requests via <code>round_robin_scheduler</code> without touching the GPU, and why it is a <strong>throughput multiplier</strong> yet can't rescue a model that 'doesn't fit'; (2) <strong>PP</strong>—why <strong>splitting layers into stages</strong> fits a huge model, why <code>event_loop_pp</code> keeps <strong>multiple micro-batches staggered in flight</strong>, and where the <strong>fill/drain bubble</strong> comes from and how micro-batch count amortizes it.",
            },
            {
                "zh": "讨论 <strong>TP × PP × DP</strong> 如何在一个大集群里拼起来，以及为什么本课归在“调度”里。请说明：①一个“TP=8、PP=2、DP=4”的部署各维度分别在切/复制什么、总卡数如何算；②为什么调度器编排的是 <strong>DP（控制器分发）与 PP（pp 循环推进 micro-batch）</strong>，而 <strong>TP 对调度器透明</strong>（藏在 <code>forward</code> 里，第 24/46 课）；③据此给出选型口诀——模型太大、流量太大分别该上哪种并行，并指出深入课程（第 46 课 TP/PP/EP/DP、第 47 课 EPLB）。",
                "en": "Discuss how <strong>TP × PP × DP</strong> compose in a large cluster, and why this lesson belongs to 'scheduling'. Explain: (1) for a 'TP=8, PP=2, DP=4' deployment, what each dimension splits/replicates and how the total card count is computed; (2) why the scheduler orchestrates <strong>DP (controller dispatch) and PP (the pp loop advancing micro-batches)</strong> while <strong>TP is transparent to the scheduler</strong> (hidden in <code>forward</code>, Lessons 24/46); (3) from this, state the selection rule—which parallelism to reach for when the model is too big vs traffic too heavy—and point to the deep dives (Lesson 46 TP/PP/EP/DP, Lesson 47 EPLB).",
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
