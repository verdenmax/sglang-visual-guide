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
                        "zh": "Scheduler 子进程握着 GPU：它持有 TpModelWorker/ModelRunner 做前向，张量并行时每个 TP rank 各起一个；两个 Manager 只是一进一出的“翻译官”",
                        "en": "The Scheduler subprocess holds the GPU: it owns TpModelWorker/ModelRunner for the forward (one per TP rank under tensor parallelism); the two Managers are just in/out translators",
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
                    "zh": "调度器“只决策、不计算”：组批和收尾是 CPU 上的轻量记账，唯独 <code>run_batch</code> 把批交给 TpModelWorker→ModelRunner 在 GPU 上 forward（第 24 课）。normal 版严格串行，CPU 忙时 GPU 闲、反之亦然。由于循环速度直接给吞吐封顶，第 21 课的 overlap 版用结果队列把上一步收尾推迟、与本步 GPU 计算重叠，从而把 CPU 时间藏起来。",
                    "en": "The scheduler 'decides, never computes': batching and finishing are light CPU accounting, while only <code>run_batch</code> hands the batch to TpModelWorker→ModelRunner for GPU forward (Lesson 24). The normal loop is strictly serial — CPU busy means GPU idle and vice versa. Since loop speed caps throughput, Lesson 21's overlap version defers the previous step's finishing via a result queue to overlap this step's GPU compute, hiding the CPU time.",
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
                    "zh": "源码里 <code>run_batch</code> 后立刻 <code>result_queue.append((batch.copy(), result))</code> 不阻塞，下一拍才 <code>popleft</code> 做 <code>process_batch_result</code>——所以收尾恒落后一拍，这就是 +1 拍延迟。下一批要在当前结果未出时就搭好，采样 token / KV 记账存在跨拍依赖，靠 <code>overlap_utils.py</code> 的 <code>FutureMap</code> 串好。某些必须先拿到上一步结果的情形会临时退回不重叠。",
                    "en": "In source, right after <code>run_batch</code> we <code>result_queue.append((batch.copy(), result))</code> without blocking, and only the next beat <code>popleft</code>s it for <code>process_batch_result</code>—so finishing always lags one beat, hence +1 latency. The next batch is built before the current result exists, giving sampled-token / KV-bookkeeping cross-beat dependencies threaded via the <code>FutureMap</code> in <code>overlap_utils.py</code>. Cases needing the prior result first fall back to no-overlap temporarily.",
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
    "24-model-runner-and-forward-batch.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 SGLang 里，<code>ModelRunner</code> 和调度器（第 18 课）各自的职责边界是什么？",
                    "en": "In SGLang, what is the responsibility boundary between <code>ModelRunner</code> and the scheduler (Lesson 18)?",
                },
                "opts": [
                    {
                        "zh": "调度器<strong>只决策不计算</strong>（收请求、组批、定策略），每步调 <code>run_batch</code> 把活儿派下去；<strong>ModelRunner 才把一批 token 真正在 GPU 上算成 logits</strong>——它是“决策→计算”的那道边界，<strong>每个 TP rank 一个</strong>，由 TpModelWorker 持有",
                        "en": "The scheduler <strong>only decides, never computes</strong> (receive, batch, set policy) and calls <code>run_batch</code> each step to dispatch work; <strong>ModelRunner is what actually turns a batch of tokens into logits on the GPU</strong> — it is the decide→compute boundary, <strong>one per TP rank</strong>, owned by the TpModelWorker",
                    },
                    {"zh": "调度器既决策又亲自跑前向，ModelRunner 只负责采样出 token", "en": "The scheduler both decides and runs the forward itself; ModelRunner only samples the token"},
                    {"zh": "整个进程共用一个 ModelRunner，所有 TP rank 都调它", "en": "The whole process shares one ModelRunner that all TP ranks call"},
                    {"zh": "ModelRunner 负责接收 HTTP 请求并分词，调度器负责前向计算", "en": "ModelRunner receives HTTP requests and tokenizes; the scheduler does the forward"},
                ],
                "answer": 0,
                "why": {
                    "zh": "第五部分讲的都是调度器“怎么决策”，它<strong>从不碰 GPU</strong>；真正把 token 算成 logits 的是 ModelRunner。调用链是 <code>run_batch</code> → TpModelWorker → 它独占的 ModelRunner，<strong>每个 TP rank 一个</strong>，各算模型一片再靠集合通信拼齐。所以它是“决策变计算”的边界，而不是共享单例，也不管接请求/分词。",
                    "en": "Part 5 is all about how the scheduler decides; it <strong>never touches the GPU</strong>. What turns tokens into logits is ModelRunner. The chain is <code>run_batch</code> → TpModelWorker → the one ModelRunner it owns, <strong>one per TP rank</strong>, each computing a slice and stitching via collectives. So it is the decide→compute boundary, not a shared singleton, and it doesn't ingest requests/tokenize.",
                },
            },
            {
                "q": {
                    "zh": "<code>ForwardBatch</code> 与 <code>ScheduleBatch</code>（第 19 课）是什么关系？",
                    "en": "What is the relationship between <code>ForwardBatch</code> and <code>ScheduleBatch</code> (Lesson 19)?",
                },
                "opts": [
                    {
                        "zh": "ForwardBatch 是 ScheduleBatch 的 <strong>GPU 视图</strong>——同一批请求换成 GPU 看得懂的张量与元数据：<code>input_ids</code>、<code>positions</code>（喂 RoPE）、<code>forward_mode</code>（选前向路径）、注意力元数据 + <code>out_cache_loc</code>（KV 读写槽位）；ScheduleBatch 记“谁要算”，ForwardBatch 记“GPU 这一步怎么算”",
                        "en": "ForwardBatch is the <strong>GPU view</strong> of a ScheduleBatch — the same batch re-expressed as GPU tensors/metadata: <code>input_ids</code>, <code>positions</code> (for RoPE), <code>forward_mode</code> (picks the path), attention metadata + <code>out_cache_loc</code> (KV read/write slots); ScheduleBatch records 'who needs computing', ForwardBatch records 'how the GPU computes this step'",
                    },
                    {"zh": "两者完全相同，只是改了个名字", "en": "They are identical, just renamed"},
                    {"zh": "ForwardBatch 是采样结果，ScheduleBatch 是输入提示", "en": "ForwardBatch is the sampling result; ScheduleBatch is the input prompt"},
                    {"zh": "ScheduleBatch 在 GPU 上、ForwardBatch 在 CPU 上，互为备份", "en": "ScheduleBatch lives on GPU and ForwardBatch on CPU, mirroring each other"},
                ],
                "answer": 0,
                "why": {
                    "zh": "ModelRunner 先把调度器给的 ScheduleBatch（一堆 Req）<strong>翻译</strong>成 ForwardBatch——GPU 前向真正需要的那几样：要喂哪些 token、每个 token 的位置、是预填充还是解码、注意力后端与 KV 槽位。一句话：ScheduleBatch=“谁要算”，ForwardBatch=“GPU 具体怎么算这一步”。它不是采样结果，也不是简单改名或 CPU 备份。",
                    "en": "ModelRunner first <strong>translates</strong> the scheduler's ScheduleBatch (a set of Reqs) into a ForwardBatch — exactly what the GPU forward needs: which tokens, each token's position, prefill vs decode, attention backend and KV slots. In short ScheduleBatch='who needs computing', ForwardBatch='how the GPU computes this step'. It is not a sampling result, a mere rename, or a CPU mirror.",
                },
            },
            {
                "q": {
                    "zh": "为什么 <strong>解码（DECODE）</strong>前向适合走 CUDA Graph 重放（第 27 课），而<strong>预填充（EXTEND）</strong>多走即时路径？",
                    "en": "Why does the <strong>DECODE</strong> forward suit CUDA-graph replay (Lesson 27) while <strong>EXTEND</strong>/prefill mostly runs eager?",
                },
                "opts": [
                    {
                        "zh": "解码每步每条请求只产出 <strong>1 个新 token</strong>，形状<strong>高度规整</strong>（可 padding 到几档固定桶），且每步算得少、内核启动开销占比高，正好让重放<strong>抹平启动开销</strong>；预填充长度千变万化、形状每批都不同、本就计算密集，录图既不划算也录不过来",
                        "en": "Decode emits <strong>just 1 new token</strong> per request per step with <strong>highly regular shapes</strong> (paddable to a few fixed buckets), and with little compute per step the launch overhead dominates, so replay <strong>flattens that overhead</strong>; prefill has wildly varying lengths, different shapes every batch, and is already compute-bound, so recording graphs is neither worthwhile nor feasible",
                    },
                    {"zh": "因为解码不需要读 KV 缓存，所以可以录图", "en": "Because decode doesn't read the KV cache, it can be graphed"},
                    {"zh": "因为预填充必须在 CPU 上跑，无法用 GPU 图", "en": "Because prefill must run on CPU and can't use a GPU graph"},
                    {"zh": "因为解码比预填充计算量更大，更值得优化", "en": "Because decode is more compute-heavy than prefill and more worth optimizing"},
                ],
                "answer": 0,
                "why": {
                    "zh": "CUDA 图录的是<strong>固定形状、固定地址</strong>的内核序列，重放只能套同样的形状。解码恰好“小而规整、重复千万次”，是访存密集、启动开销占大头的阶段，图重放收益最大；预填充“大而多变、各算各的”，是计算密集阶段，形状几乎每批都不同，更适合灵活的即时路径。解码并非不读 KV，预填充也不在 CPU 跑，解码单步算量也更小。",
                    "en": "A CUDA graph records a <strong>fixed-shape, fixed-address</strong> kernel sequence; replay only fits the same shapes. Decode is 'small, regular, repeated millions of times', memory-bound with launch overhead dominating, so replay pays most; prefill is 'big, varied, each its own', compute-bound with shapes differing nearly every batch, better on the eager path. Decode does read KV, prefill doesn't run on CPU, and decode does less compute per step.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“车间机台操作工”的类比，完整讲清一次前向在 ModelRunner 里怎么走完。请逐一覆盖：①调用链——第 18 课调度器 <code>run_batch</code> → TpModelWorker → <strong>每 rank 一个</strong>的 ModelRunner，以及为什么 8 个 rank 必须<strong>步调一致</strong>跑同一批（集合通信会死锁）；②ModelRunner 先把 <code>ScheduleBatch</code> 翻成 <code>ForwardBatch</code>，并说明 <code>positions/forward_mode/</code>注意力元数据/<code>out_cache_loc</code> 各驱动了什么；③前向内部 <strong>嵌入 → N 层解码层（注意力读写 KV 池、命中时复用 RadixAttention 前缀，第 7 课）→ 末端归一 + <code>lm_head</code> → logits</strong>，并指出 <strong>logits→token</strong> 的转折点（<code>sample()</code> 交给第 28 课采样器）发生在哪一步。",
                "en": "Using the 'machine operator on the floor' analogy, fully explain how one forward runs through ModelRunner. Cover each: (1) the call chain — scheduler <code>run_batch</code> (Lesson 18) → TpModelWorker → the <strong>one-per-rank</strong> ModelRunner, and why 8 ranks must run the same batch in <strong>lockstep</strong> (collectives deadlock otherwise); (2) ModelRunner first translates <code>ScheduleBatch</code> into <code>ForwardBatch</code>, and what <code>positions/forward_mode/</code>attention metadata/<code>out_cache_loc</code> each drive; (3) the forward internals <strong>embed → N decoder layers (attention reads/writes the KV pool, reusing a RadixAttention prefix on a hit, Lesson 7) → final norm + <code>lm_head</code> → logits</strong>, and where the <strong>logits→token</strong> turning point (<code>sample()</code> handing off to the Lesson 28 Sampler) happens.",
            },
            {
                "zh": "围绕 <strong>EXTEND vs DECODE</strong> 展开，把“同一个 forward 的两种性格”讲透，并说明它如何决定 Part 6 后续几课的脉络。请说明：①两者在 token 数、形状规整度、瓶颈（预填充<strong>计算密集</strong> vs 解码<strong>访存密集</strong>）上的差异；②为什么解码适合 <strong>CUDA Graph 重放</strong>（第 27 课）而预填充多走即时/分块（第 22 课）；③为什么 ModelRunner 还要<strong>同时</strong>握住模型（第 25/26 课）、KV 池（第 30 课）、注意力后端（第 33 课）乃至草稿模型（第 43 课），并据此说明 Part 6 接下来分别拆解的是哪几样零件。",
                "en": "Centered on <strong>EXTEND vs DECODE</strong>, fully explain the 'two personalities of one forward' and how it shapes the rest of Part 6. Explain: (1) their differences in token count, shape regularity, and bottleneck (prefill <strong>compute-bound</strong> vs decode <strong>memory-bound</strong>); (2) why decode suits <strong>CUDA-graph replay</strong> (Lesson 27) while prefill mostly runs eager/chunked (Lesson 22); (3) why ModelRunner must hold the model (Lessons 25/26), KV pool (Lesson 30), attention backend (Lesson 33), and even a draft model (Lesson 43) <strong>all at once</strong>, and from that, which parts the rest of Part 6 takes apart one by one.",
            },
        ],
    },
    "25-model-loading-and-weights.html": {
        "mcq": [
            {
                "q": {
                    "zh": "<code>DefaultModelLoader</code> 为什么要把权重做成<strong>流式（streaming）</strong>读取，而不是先全读进主机内存再分发？",
                    "en": "Why does <code>DefaultModelLoader</code> read weights via <strong>streaming</strong> instead of reading everything into host RAM first and then distributing?",
                },
                "opts": [
                    {
                        "zh": "权重是一个 <code>(name, tensor)</code> <strong>生成器</strong>，读一个、灌一个、丢一个，<strong>主机内存上界与模型总大小解耦</strong>——哪怕几百 GB 的模型，任意时刻内存里也只驻留当前这一个张量，不会把内存撑爆",
                        "en": "Weights are a <code>(name, tensor)</code> <strong>generator</strong>: read one, load one, drop one, so the <strong>host-memory bound is decoupled from total model size</strong> — even a hundreds-of-GB model keeps only the current tensor in memory at any instant and won't blow up RAM",
                    },
                    {"zh": "因为流式加载比一次性加载在数值上更精确", "en": "Because streaming loads are numerically more accurate than one-shot loads"},
                    {"zh": "因为只有流式才能把权重放到 GPU，整读无法上卡", "en": "Because only streaming can place weights on the GPU; a full read can't reach the card"},
                    {"zh": "因为流式加载会自动跳过不需要的层，省下计算", "en": "Because streaming automatically skips unneeded layers, saving compute"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<code>_get_all_weights</code> 返回的是生成器，<code>model.load_weights</code> 逐张量灌入，所以任意时刻内存里只有当前张量（及其目标分片），上界与模型大小<strong>解耦</strong>。这才是同一台机器既能装小模型又能装几百 GB 大模型的原因。它跟数值精度无关，整读其实也能上卡，更不会自动跳层。",
                    "en": "<code>_get_all_weights</code> returns a generator and <code>model.load_weights</code> pours tensors in one by one, so only the current tensor (and its target slice) lives in memory at any instant, the bound <strong>decoupled</strong> from model size. That's why one machine fits both small and hundreds-of-GB models. It has nothing to do with numerical precision, a full read can still reach the card, and nothing auto-skips layers.",
                },
            },
            {
                "q": {
                    "zh": "把 HuggingFace 权重名映射到 SGLang 内部参数、并<strong>融合</strong> q/k/v、gate/up 的逻辑，写在哪里？",
                    "en": "Where does the logic that maps HuggingFace weight names to SGLang internal params and <strong>fuses</strong> q/k/v, gate/up live?",
                },
                "opts": [
                    {
                        "zh": "写在<strong>每个模型类自己的</strong> <code>load_weights(weights)</code> 里——它内置映射表，知道 <code>q_proj/k_proj/v_proj</code> 该塞进打包权重的哪一段偏移；Loader 只负责<strong>把张量流式读出来</strong>，不关心模型内部叫什么",
                        "en": "In <strong>each model class's own</strong> <code>load_weights(weights)</code> — it carries a mapping table knowing which offset segment of the packed weight each of <code>q_proj/k_proj/v_proj</code> goes into; the Loader only <strong>streams tensors out</strong> and doesn't care about a model's internal names",
                    },
                    {"zh": "写在 <code>DefaultModelLoader</code> 里，对所有模型用同一张硬编码映射表", "en": "In <code>DefaultModelLoader</code>, using one hard-coded mapping table for all models"},
                    {"zh": "写在 safetensors 文件的元数据里，加载时自动套用", "en": "In the safetensors file metadata, applied automatically at load"},
                    {"zh": "不需要映射，HF 名字和 SGLang 参数名永远一一对应", "en": "No mapping is needed; HF names and SGLang param names always match one to one"},
                ],
                "answer": 0,
                "why": {
                    "zh": "Loader 只流式读张量，<strong>不该也不关心</strong>某模型内部叫什么；而“这个名字对应哪个参数、要不要融合/转置”是模型私有知识，封装在模型的 <code>load_weights</code> 里。所以新增模型<strong>不必碰 Loader</strong>，只写好 <code>load_weights</code>（第 26 课）即可。磁盘上 q/k/v 是分开三份，内存里要打包一份，正需要这层映射。",
                    "en": "The Loader only streams tensors and <strong>shouldn't</strong> know a model's internals; which param a name maps to and whether to fuse/transpose is model-private, encapsulated in the model's <code>load_weights</code>. So adding a model <strong>needs no Loader change</strong> — just write its <code>load_weights</code> (Lesson 26). q/k/v are three separate parts on disk but one packed param in memory, which is exactly why this mapping is needed.",
                },
            },
            {
                "q": {
                    "zh": "在 8 卡张量并行（TP）部署里，加载时每张卡的显存里存的是什么权重？",
                    "en": "In an 8-GPU tensor-parallel (TP) deployment, what weights end up in each card's memory at load time?",
                },
                "opts": [
                    {
                        "zh": "<strong>只存本 rank 那一片</strong>：q/k/v、gate/up 走列并行（按输出维切），o_proj、down_proj 走行并行（按输入维切），每卡只接自己 1/8 的权重，加载本身就是分布式的",
                        "en": "<strong>Only this rank's slice</strong>: q/k/v and gate/up go column-parallel (split on output dim), o_proj and down_proj go row-parallel (split on input dim), each card takes only its 1/8 of weights — loading itself is distributed",
                    },
                    {"zh": "每张卡都存一整份完整权重，靠冗余提高可靠性", "en": "Each card stores a full copy of the weights for redundancy/reliability"},
                    {"zh": "只有 0 号卡存权重，其余卡每步向它请求", "en": "Only card 0 stores weights; the others request from it each step"},
                    {"zh": "权重存在 CPU 内存，GPU 每步临时拷贝需要的部分", "en": "Weights live in CPU RAM; the GPU copies the needed part each step"},
                ],
                "answer": 0,
                "why": {
                    "zh": "TP 下每个权重<strong>按 rank 切片</strong>：列并行（q/k/v、gate/up）每卡算一段输出，行并行（o_proj、down_proj）每卡算部分和再 all-reduce。Loader/模型加载时就只把本 rank 的那一片搬上这张卡，于是 8 卡各只有 1/8 权重（第 24/46 课）。并非整份冗余、单卡集中或常驻 CPU。",
                    "en": "Under TP each weight is <strong>sliced per rank</strong>: column-parallel (q/k/v, gate/up) computes a slice of output per card, row-parallel (o_proj, down_proj) computes partial sums then all-reduce. The Loader/model moves only this rank's slice onto the card, so 8 cards each hold 1/8 of the weights (Lessons 24/46). It is not full redundancy, a single hub card, or CPU-resident weights.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“编号纸箱布置新家”的类比，完整讲清 <code>DefaultModelLoader</code> 把一堆磁盘文件变成“每卡各就各位的张量”的全过程。请逐一覆盖：①<strong>两步走</strong>——<code>_initialize_model</code> 先搭空壳（结构有了、权重是占位张量），再 <code>load_weights_and_postprocess</code> 把 <code>_get_all_weights</code> 的<strong>生成器</strong>交给 <code>model.load_weights</code> 逐张量灌入；②为什么必须<strong>流式</strong>（主机内存上界与模型大小解耦）；③<strong>名字映射 + 融合</strong>为什么交给模型自己的 <code>load_weights</code>（第 26 课），以及 q/k/v、gate/up 如何打包；④<strong>TP 切片</strong>（列/行并行，第 46 课）与 <strong>dtype/量化</strong>（FP8/INT4 + scales，第 35 课）各在这步做了什么。",
                "en": "Using the 'numbered flat-pack boxes furnishing a home' analogy, fully explain how <code>DefaultModelLoader</code> turns a pile of disk files into 'all-in-place tensors per card'. Cover each: (1) the <strong>two steps</strong> — <code>_initialize_model</code> builds the shell (structure present, params placeholders), then <code>load_weights_and_postprocess</code> hands <code>_get_all_weights</code>'s <strong>generator</strong> to <code>model.load_weights</code> to pour tensors in; (2) why <strong>streaming</strong> is required (host-memory bound decoupled from model size); (3) why <strong>name mapping + fusion</strong> live in the model's own <code>load_weights</code> (Lesson 26), and how q/k/v, gate/up are packed; (4) what <strong>TP slicing</strong> (column/row-parallel, Lesson 46) and <strong>dtype/quantization</strong> (FP8/INT4 + scales, Lesson 35) each do at this step.",
            },
            {
                "zh": "SGLang 不止一种 Loader：<code>DefaultModelLoader</code> 是常路，另有 <code>DummyModelLoader</code>、<code>ShardedStateLoader</code>、<code>LayeredModelLoader</code>/远程加载等变体。请说明：①各自适合什么场景（测试量显存、按已切分状态快启、按层省内存、远程分发）；②为什么把“加载逻辑统一收口”能让 SGLang 用一套框架托起上百种模型 + 各类量化格式；③如果你要加载一个 <strong>FP8 量化</strong>的 checkpoint，加载这一步相比 bf16 多做了什么（读 scales、按量化布局摆放、前向用量化内核或惰性反量化，第 35 课）。",
                "en": "SGLang has more than one Loader: <code>DefaultModelLoader</code> is the common path, with <code>DummyModelLoader</code>, <code>ShardedStateLoader</code>, <code>LayeredModelLoader</code>/remote variants. Explain: (1) what each suits (testing/memory measurement, fast start from an already-sliced state, layer-by-layer low memory, remote distribution); (2) why 'funneling loading through one point' lets SGLang support hundreds of models + many quant formats under one framework; (3) if you load an <strong>FP8-quantized</strong> checkpoint, what loading does beyond bf16 (read scales, arrange by quant layout, forward uses quantized kernels or lazy dequant, Lesson 35).",
            },
        ],
    },
    "26-writing-a-model.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 SGLang 里“写一个模型”（如 Llama），模型作者真正<strong>新写</strong>的主要是什么？",
                    "en": "When you 'write a model' (e.g. Llama) in SGLang, what does the author mainly write <strong>anew</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>架构 + load_weights 名字映射</strong>：用哪些层、各维度、残差怎么连，再把 HF 权重名映射到内部参数；注意力内核、TP 通信、KV 管理都用 SGLang 现成的<strong>并行层</strong>，作者不实现底层算子",
                        "en": "<strong>Architecture + load_weights name mapping</strong>: which layers, what dims, how residuals connect, plus mapping HF weight names to internal params; attention kernels, TP comms, KV mgmt all reuse SGLang's ready <strong>parallel layers</strong> — the author writes no low-level ops",
                    },
                    {"zh": "从零手写注意力 CUDA 内核和跨卡 all-reduce 通信", "en": "Hand-write the attention CUDA kernel and cross-GPU all-reduce from scratch"},
                    {"zh": "实现自己的 KV 缓存分页与显存分配器", "en": "Implement its own KV-cache paging and GPU memory allocator"},
                    {"zh": "编写张量并行的权重切分与通信调度逻辑", "en": "Write the tensor-parallel weight-slicing and communication scheduling logic"},
                ],
                "answer": 0,
                "why": {
                    "zh": "模型文件就是把 SGLang 现成的并行层（<code>QKVParallelLinear</code>、<code>RowParallelLinear</code>、<code>RadixAttention</code>、<code>RMSNorm</code> 等）<strong>组装</strong>起来：作者只描述架构并写 <code>load_weights</code> 的名字映射（第 25 课）。注意力内核、TP 通信、KV 管理、CUDA 图全是白拿的——这正是“加模型便宜、day-0 支持广”的根因。",
                    "en": "A model file <strong>assembles</strong> SGLang's ready parallel layers (<code>QKVParallelLinear</code>, <code>RowParallelLinear</code>, <code>RadixAttention</code>, <code>RMSNorm</code>, ...): the author only describes the architecture and writes <code>load_weights</code> name mapping (Lesson 25). Attention kernels, TP comms, KV mgmt, CUDA graphs come free — the root reason adding a model is cheap and day-0 support is broad.",
                },
            },
            {
                "q": {
                    "zh": "Llama 的四（五）个类是怎么<strong>层层套娃</strong>的？",
                    "en": "How do Llama's four (five) classes <strong>nest</strong>?",
                },
                "opts": [
                    {
                        "zh": "<code>LlamaAttention</code> 和 <code>LlamaMLP</code> 组成 <code>LlamaDecoderLayer</code>；N 层 + 嵌入 + 末端归一组成 <code>LlamaModel</code>；再加 <code>lm_head</code> 与 <code>forward</code> 就是对运行时暴露的 <code>LlamaForCausalLM</code>",
                        "en": "<code>LlamaAttention</code> and <code>LlamaMLP</code> compose <code>LlamaDecoderLayer</code>; N layers + embed + final norm compose <code>LlamaModel</code>; add <code>lm_head</code> and <code>forward</code> and you get <code>LlamaForCausalLM</code>, the class the runtime sees",
                    },
                    {"zh": "五个类彼此平级，由调度器在运行时按需拼接", "en": "The five classes are siblings, stitched on demand by the scheduler at runtime"},
                    {"zh": "<code>LlamaForCausalLM</code> 最底层，注意力包着整模型", "en": "<code>LlamaForCausalLM</code> is innermost; attention wraps the whole model"},
                    {"zh": "只有一个大类，所有逻辑写在一个 forward 里", "en": "There is only one big class with all logic in a single forward"},
                ],
                "answer": 0,
                "why": {
                    "zh": "套娃关系是 <code>LlamaAttention</code>/<code>LlamaMLP</code> → <code>LlamaDecoderLayer</code> → <code>LlamaModel</code> → <code>LlamaForCausalLM</code>。每个类只负责一件事、内部都用 SGLang 现成层。ModelRunner（第 24 课）握住的是最外层 <code>LlamaForCausalLM</code>，它提供 <code>forward(...)→logits</code> 与 <code>load_weights</code>。",
                    "en": "The nesting is <code>LlamaAttention</code>/<code>LlamaMLP</code> → <code>LlamaDecoderLayer</code> → <code>LlamaModel</code> → <code>LlamaForCausalLM</code>. Each class does one thing, all using SGLang's ready layers. ModelRunner (Lesson 24) holds the outermost <code>LlamaForCausalLM</code>, which offers <code>forward(...)→logits</code> and <code>load_weights</code>.",
                },
            },
            {
                "q": {
                    "zh": "<code>forward_batch</code> 被一路传到每一层注意力，它扮演什么角色？",
                    "en": "<code>forward_batch</code> is threaded into every layer's attention — what role does it play?",
                },
                "opts": [
                    {
                        "zh": "它是模型文件与运行时之间的<strong>那道缝</strong>：告诉注意力该读/写 KV 池的哪些槽位、用哪个注意力后端（第 24/33 课），模型只管算对张量",
                        "en": "It is <strong>the seam</strong> between model file and runtime: it tells attention which KV-pool slots to read/write and which attention backend to use (Lessons 24/33); the model just computes tensors correctly",
                    },
                    {"zh": "它保存模型权重，前向时按需加载到 GPU", "en": "It stores model weights and loads them to the GPU on demand during forward"},
                    {"zh": "它是采样参数容器，决定 temperature/top-p", "en": "It is a sampling-params container deciding temperature/top-p"},
                    {"zh": "它只在预填充用到，解码阶段不传", "en": "It is only used in prefill and not passed during decode"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<code>forward_batch</code>（第 24 课）携带 KV 槽位、注意力元数据等，逐层穿到 <code>self_attn</code>，让注意力知道当前 k/v 写到池子哪、历史从哪读（第 33 课）。这正是“模型只管把张量算对、运行时负责 KV 寻址”的分界。它不存权重、不是采样容器，预填充与解码都要传。",
                    "en": "<code>forward_batch</code> (Lesson 24) carries KV slots and attention metadata, threading layer by layer into <code>self_attn</code> so attention knows where to write current k/v and read history (Lesson 33). That is the boundary 'model computes tensors right, runtime owns KV addressing.' It doesn't store weights, isn't a sampling container, and is passed in both prefill and decode.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“标准乐高积木拼模型”的类比，完整讲清在 SGLang 里“写一个 Llama”到底写了什么、又白拿了什么。请逐一覆盖：①四（五）个类如何<strong>层层套娃</strong>（<code>LlamaAttention</code>→<code>LlamaMLP</code>→<code>LlamaDecoderLayer</code>→<code>LlamaModel</code>→<code>LlamaForCausalLM</code>）；②每个类用到哪些 SGLang 现成层（<code>QKVParallelLinear</code>/<code>RowParallelLinear</code>/<code>RadixAttention</code>/<code>VocabParallelEmbedding</code>/<code>RMSNorm</code>/RoPE）；③一层里“input_norm→注意力(读写 KV)→残差→post_norm→MLP→残差”的结构；④作者真正<strong>新写</strong>的只有架构 + <code>load_weights</code> 映射（第 25 课），而注意力内核、TP 切片/通信（第 46 课）、KV 管理（第 33 课）都是白拿。",
                "en": "Using the 'building with standard LEGO bricks' analogy, fully explain what 'writing a Llama' in SGLang actually writes and what it gets for free. Cover each: (1) how the four (five) classes <strong>nest</strong> (<code>LlamaAttention</code>→<code>LlamaMLP</code>→<code>LlamaDecoderLayer</code>→<code>LlamaModel</code>→<code>LlamaForCausalLM</code>); (2) which ready SGLang layers each uses (<code>QKVParallelLinear</code>/<code>RowParallelLinear</code>/<code>RadixAttention</code>/<code>VocabParallelEmbedding</code>/<code>RMSNorm</code>/RoPE); (3) the in-layer structure 'input_norm→attention(read/write KV)→residual→post_norm→MLP→residual'; (4) that the author truly writes anew only the architecture + <code>load_weights</code> mapping (Lesson 25), while attention kernels, TP slicing/comms (Lesson 46), KV mgmt (Lesson 33) come free.",
            },
            {
                "zh": "解释为什么 SGLang 能对新开源模型做到近乎 <strong>day-0 支持</strong>、模型覆盖面极广。请说明：①“模型文件薄、底层库厚”的分工——<code>LlamaForCausalLM.forward</code> 几乎只是把活儿交给 <code>self.model</code> 再交给 <code>logits_processor</code>/<code>lm_head</code>，没有 CUDA/all-reduce/KV 分页的影子；②<code>forward_batch</code> 作为模型与运行时之间的缝，如何让同一份模型文件适配不同注意力后端与并行配置（第 24/33/46 课）；③MoE（第 34 课）、多模态（第 49 课）等更复杂架构为何只是“同套脚手架上多插几种层”，而非推倒重来。",
                "en": "Explain why SGLang achieves near <strong>day-0 support</strong> for new open models with very broad coverage. Cover: (1) the 'thin model file, thick library' split — <code>LlamaForCausalLM.forward</code> basically just hands work to <code>self.model</code> then <code>logits_processor</code>/<code>lm_head</code>, with no trace of CUDA/all-reduce/KV paging; (2) how <code>forward_batch</code>, as the model/runtime seam, lets one model file adapt to different attention backends and parallel configs (Lessons 24/33/46); (3) why more complex architectures like MoE (Lesson 34) and multimodal (Lesson 49) are just 'plug a few extra layers into the same scaffolding' rather than a rewrite.",
            },
        ],
    },
    "27-cuda-graph-capture-and-replay.html": {
        "mcq": [
            {
                "q": {
                    "zh": "CUDA Graph 在解码前向里主要消灭的是哪种开销？",
                    "en": "What overhead does a CUDA graph mainly eliminate in the decode forward?",
                },
                "opts": [
                    {
                        "zh": "<strong>逐个内核的启动（launch）开销</strong>——一次解码前向要发起几百个小内核，每次发起都有一笔<strong>固定的 CPU 启动费</strong>；解码内核太短，这笔费用反而主导整步、GPU 干等 CPU。图把整条前向录成一张，<strong>一次提交整体重放</strong>，把几百次发起压成一次",
                        "en": "<strong>Per-kernel launch overhead</strong> — one decode forward launches hundreds of tiny kernels, each with a <strong>fixed CPU issue fee</strong>; decode kernels are so short the fee dominates the step and the GPU waits on the CPU. The graph records the whole forward and <strong>replays it as a single submission</strong>, collapsing hundreds of launches into one",
                    },
                    {"zh": "矩阵乘法本身的浮点计算量（FLOPs）", "en": "The floating-point compute (FLOPs) of the matmuls themselves"},
                    {"zh": "KV 缓存占用的显存", "en": "The GPU memory used by the KV cache"},
                    {"zh": "网络上 token 传输的带宽", "en": "Network bandwidth for transferring tokens"},
                ],
                "answer": 0,
                "why": {
                    "zh": "图重放不改变要算多少（FLOPs 不变），也不省显存或网络；它省的是 CPU <strong>逐个发起内核</strong>的固定开销。解码每步每条只算 1 个 token、内核极短，发起费占比最高，所以重放收益最大——一次提交替代几百次发起，GPU 不再停下来等 CPU 递下一个内核。",
                    "en": "Replay doesn't change how much is computed (FLOPs are the same), nor save memory or network; it removes the fixed CPU cost of <strong>launching kernels one by one</strong>. Decode computes just 1 token per request per step with extremely short kernels, so the launch fee dominates and replay pays most — one submission replaces hundreds of launches, and the GPU stops waiting on the CPU for the next kernel.",
                },
            },
            {
                "q": {
                    "zh": "为什么 SGLang 要为<strong>一组固定 batch 尺寸</strong>分别录图，运行时还要把真实 batch <strong>padding</strong> 到最近的桶？",
                    "en": "Why does SGLang record graphs for a <strong>set of fixed batch sizes</strong> and then <strong>pad</strong> the real batch up to the nearest bucket at run time?",
                },
                "opts": [
                    {
                        "zh": "因为图绑定录制时的<strong>静态形状与显存地址</strong>，重放只能套同样的形状；真实 batch 每步可能不同，不可能为每个具体大小都录一张。于是只录几档桶（1/2/4/8/…/max），真实 batch <strong>向上取整到最近桶</strong>（<code>_pad_to_bucket</code>）后重放——多算几行 padding 的代价，远小于省下的几百次内核启动",
                        "en": "Because a graph binds the <strong>static shapes and memory addresses</strong> from capture, replay only fits the same shape; the real batch can differ every step, so you can't record one per exact size. Hence only a few buckets (1/2/4/8/…/max) are recorded, and the real batch is <strong>rounded up to the nearest bucket</strong> (<code>_pad_to_bucket</code>) before replay — the few padded rows cost far less than the hundreds of launches saved",
                    },
                    {"zh": "因为 GPU 一次只能处理 2 的幂次大小的 batch", "en": "Because the GPU can only process power-of-two batch sizes"},
                    {"zh": "因为 padding 能提高数值精度", "en": "Because padding improves numerical precision"},
                    {"zh": "因为每个 batch 尺寸需要单独的 GPU", "en": "Because each batch size needs its own GPU"},
                ],
                "answer": 0,
                "why": {
                    "zh": "CUDA 图是“录死”的：形状和地址在捕获时就固定，重放无法适配任意大小。逐个大小录图既不划算也录不过来，所以分桶 + 向上 padding 是用“少量空算”换“能复用整张图”。这与 GPU 是否支持任意大小、精度、或硬件数量都无关。",
                    "en": "A CUDA graph is frozen: shape and address are fixed at capture, so replay can't adapt to arbitrary sizes. Recording one per size is neither worthwhile nor feasible, so bucketing + rounding up trades a little wasted compute for reusing a whole graph. It has nothing to do with power-of-two limits, precision, or hardware count.",
                },
            },
            {
                "q": {
                    "zh": "为什么<strong>解码</strong>走 CUDA 图，而<strong>预填充</strong>大多不走？CUDA 图又如何与<strong>重叠调度器（第 21 课）</strong>配合？",
                    "en": "Why is <strong>decode</strong> graphed while <strong>prefill</strong> mostly isn't, and how does the CUDA graph pair with the <strong>overlap scheduler (Lesson 21)</strong>?",
                },
                "opts": [
                    {
                        "zh": "解码形状高度规整（batch 稳定、序列每步只长 1），天然能套固定桶；预填充长度多变、一次算几百上千 token、几乎每批形状不同，录图不划算。配合上：<strong>GPU 重放整张图时，CPU 正好腾手去调度下一步</strong>，两者叠加让 GPU 几乎不空转",
                        "en": "Decode shapes are highly regular (steady batch, sequence growing by 1), fitting fixed buckets naturally; prefill lengths vary, computing hundreds-to-thousands of tokens with a different shape almost every batch, so graphing isn't worthwhile. Pairing: <strong>while the GPU replays the whole graph, the CPU is freed to schedule the next step</strong>, so together the GPU barely idles",
                    },
                    {"zh": "解码必须在 CPU 上跑，所以用图；预填充在 GPU 上跑", "en": "Decode must run on CPU so it uses a graph; prefill runs on GPU"},
                    {"zh": "预填充不需要注意力，所以不用图", "en": "Prefill doesn't need attention, so no graph"},
                    {"zh": "重叠调度器让 CPU 和 GPU 同时跑同一个内核", "en": "The overlap scheduler makes CPU and GPU run the same kernel simultaneously"},
                ],
                "answer": 0,
                "why": {
                    "zh": "图要求静态形状：解码规整、可分桶，预填充多变、不划算，这就是“解码走图、预填充走即时”的根因（第 24 课）。重叠调度（第 21 课）让 CPU 在 GPU 重放期间排下一步，两者天作之合——GPU 几乎不留空隙。解码并非在 CPU 跑，预填充也需要注意力，CPU/GPU 也不是跑同一个内核。",
                    "en": "Graphs need static shapes: decode is regular and bucketable, prefill is varied and not worthwhile — the root reason for 'decode graphed, prefill eager' (Lesson 24). The overlap scheduler (Lesson 21) lets the CPU line up the next step while the GPU replays, a perfect match leaving almost no GPU idle. Decode doesn't run on CPU, prefill does need attention, and CPU/GPU don't run the same kernel.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“自动钢琴纸卷 / 录好的宏”的类比，把 CUDA Graph 的捕获与重放从头讲透。请覆盖：①问题——一次解码前向要发起几百个小内核（每层 norm、q/k/v、注意力、o_proj、gate/up、down…），每次发起有<strong>固定 CPU 启动费</strong>，解码内核太短时这笔费用为何主导整步、GPU 为何干等 CPU；②解法——图如何把整条前向的内核序列与依赖<strong>录一次</strong>、再<strong>一次提交整体重放</strong>；③SGLang 的工程做法——启动时 <code>BaseCudaGraphRunner.capture</code> 为一组尺寸（1/2/4/8/…/max）各录一张，运行时用 <code>_pad_to_bucket</code> 向上取整到最近桶、<code>can_run_graph</code> 通过后重放（第 24 课）。",
                "en": "Using the 'player-piano roll / recorded macro' analogy, fully explain CUDA-graph capture and replay from scratch. Cover: (1) the problem — one decode forward launches hundreds of tiny kernels (per layer: norm, q/k/v, attention, o_proj, gate/up, down…), each with a <strong>fixed CPU launch fee</strong>, and why that fee dominates the step when decode kernels are short, leaving the GPU waiting on the CPU; (2) the fix — how a graph <strong>records</strong> the whole forward's kernel sequence and dependencies once and <strong>replays it as a single submission</strong>; (3) SGLang's engineering — at startup <code>BaseCudaGraphRunner.capture</code> records one graph per size (1/2/4/8/…/max), and at run time <code>_pad_to_bucket</code> rounds up to the nearest bucket and replays once <code>can_run_graph</code> passes (Lesson 24).",
            },
            {
                "zh": "围绕“图必须静态”这条硬约束，说明它如何决定 SGLang 的设计，并解释由此带来的权衡。请说明：①静态形状 → <strong>batch 分桶 + padding</strong>；静态地址 → <strong>预分配静态缓冲、每步拷入</strong>；捕获区内<strong>禁止数据相关控制流</strong> → 动态算子留图外或用分段/可断开图（第 33 课）；②为什么这三条合起来正好解释“<strong>解码走图、预填充走即时</strong>”；③回报与代价——解码吞吐大涨、与重叠调度器（第 21 课）天作之合，但录很多档要花<strong>启动时间</strong>与<strong>显存</strong>（每张图握着自己的静态缓冲），桶太密太疏各有什么坏处；并联系投机解码（第 43 课）为何要为更复杂的形状专门考虑录图。",
                "en": "Centered on the hard constraint 'a graph must be static', explain how it shapes SGLang's design and the resulting tradeoffs. Cover: (1) static shapes → <strong>batch bucketing + padding</strong>; static addresses → <strong>pre-allocated static buffers copied into each step</strong>; <strong>no data-dependent control flow</strong> in the captured region → dynamic ops outside or a piecewise/breakable graph (Lesson 33); (2) why these three together explain '<strong>decode graphed, prefill eager</strong>'; (3) payoff vs cost — big decode throughput and a perfect match with the overlap scheduler (Lesson 21), but recording many sizes costs <strong>startup time</strong> and <strong>memory</strong> (each graph holds its own static buffers), and what goes wrong if buckets are too dense or too sparse; and relate to why speculative decoding (Lesson 43) must specially consider capturing more complex shapes.",
            },
        ],
    },
    "28-sampler-and-sampling-params.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在采样管线里，<strong>温度（temperature）</strong>到底改变了什么？",
                    "en": "In the sampling pipeline, what does <strong>temperature</strong> actually change?",
                },
                "opts": [
                    {
                        "zh": "它把 logits <strong>除以 T</strong> 来重塑分布的<strong>陡峭程度</strong>：T&lt;1 让分布更尖、更确定（大热门更稳），T&gt;1 让它更平、更随机（冷门也有机会），<strong>T=0 退化为贪心 argmax</strong>。它<strong>不改变 token 之间的排名</strong>，只改变高低之间的悬殊程度",
                        "en": "It <strong>divides the logits by T</strong> to reshape the distribution's <strong>steepness</strong>: T&lt;1 sharpens it (more deterministic, the favorite is steadier), T&gt;1 flattens it (more random, longshots get a chance), <strong>T=0 collapses to greedy argmax</strong>. It <strong>does not change the ranking</strong> of tokens, only how lopsided the gaps are",
                    },
                    {"zh": "它直接删掉概率最低的若干 token", "en": "It directly deletes the lowest-probability tokens"},
                    {"zh": "它决定一次生成多少个 token", "en": "It decides how many tokens to generate at once"},
                    {"zh": "它给已经出现过的词扣分以抑制复读", "en": "It penalizes already-seen words to suppress repetition"},
                ],
                "answer": 0,
                "why": {
                    "zh": "温度是唯一重塑整条曲线陡峭程度的旋钮：除以 T 改变高低差距而非排名，T=0 等价贪心。删低概率 token 是 top-k/top-p/min-p 的活，扣分是 penalties 的活，生成多少是 max_new_tokens 的活——别混淆。",
                    "en": "Temperature is the only knob that reshapes curve steepness: dividing by T changes the gaps, not the ranking, and T=0 equals greedy. Deleting low-prob tokens is top-k/top-p/min-p's job, penalizing is penalties' job, and how many tokens is max_new_tokens — don't conflate them.",
                },
            },
            {
                "q": {
                    "zh": "<strong>结构化输出</strong>（如强制合法 JSON，第 48 课）是怎么挂进采样、保证输出永远语法合法的？",
                    "en": "How does <strong>structured output</strong> (e.g. forcing valid JSON, Lesson 48) hook into sampling to guarantee grammar-valid output?",
                },
                "opts": [
                    {
                        "zh": "在<strong>采样之前</strong>，约束引擎把所有“此刻语法不允许”的 token 的 logit <strong>置为 −∞</strong>（mask 掉）；softmax 后它们概率正好是 0，<strong>采样根本不可能选到</strong>。于是从机制上保证合法，而不是事后校验、失败再重试",
                        "en": "<strong>Before sampling</strong>, the constraint engine sets the logit of every “grammar-disallowed-right-now” token to <strong>−∞</strong> (masks it); after softmax their probability is exactly 0, so <strong>sampling can never pick them</strong>. Validity is guaranteed by mechanism, not by validating afterward and retrying on failure",
                    },
                    {"zh": "它在采样之后检查 token，若不合法就重新采样", "en": "It checks the token after sampling and resamples if invalid"},
                    {"zh": "它把温度调到 0，强制贪心", "en": "It sets temperature to 0 to force greedy"},
                    {"zh": "它训练一个专门的模型只输出 JSON", "en": "It trains a dedicated model that only outputs JSON"},
                ],
                "answer": 0,
                "why": {
                    "zh": "关键在“采样前对 logits 动手”：把违规 token 置 −∞，softmax 后概率为 0，机制上就选不到。这比“事后校验+重试”高效且确定。它和温度、专用模型无关——logit-bias、min_tokens/EOS 抑制也都挂在同一个采样前的位置。",
                    "en": "The key is acting on logits before sampling: set violating tokens to −∞ and after softmax their prob is 0, so they're mechanically unselectable — more efficient and certain than validate-then-retry. It's unrelated to temperature or a dedicated model; logit-bias and min_tokens/EOS suppression hook at the same pre-sampling spot.",
                },
            },
            {
                "q": {
                    "zh": "关于<strong>同一个 batch 里的多条请求</strong>，下面哪句是对的？",
                    "en": "About <strong>multiple requests in the same batch</strong>, which statement is correct?",
                },
                "opts": [
                    {
                        "zh": "它们可以<strong>各用各的采样参数</strong>：每条请求的 <span class='mono'>SamplingParams</span> 被打包进 <span class='mono'>SamplingBatchInfo</span> 的张量里（如 <span class='mono'>temperatures</span> 是一整个张量），Sampler <strong>逐请求向量化</strong>处理——A 请求贪心、B 请求温度 0.9、C 请求 top_p=0.8 可以在同一步里一起算",
                        "en": "They can each use <strong>their own sampling params</strong>: every request's <span class='mono'>SamplingParams</span> is packed into <span class='mono'>SamplingBatchInfo</span> tensors (e.g. <span class='mono'>temperatures</span> is a whole tensor), and the Sampler processes them <strong>per-request, vectorized</strong> — request A greedy, B at temperature 0.9, C at top_p=0.8 can all be computed together in one step",
                    },
                    {"zh": "整个 batch 必须共用同一套采样参数", "en": "The whole batch must share one set of sampling params"},
                    {"zh": "每条请求都要单独跑一次 Sampler", "en": "Each request must run the Sampler separately one at a time"},
                    {"zh": "batch 里只有第一条请求的参数生效", "en": "Only the first request's params in the batch take effect"},
                ],
                "answer": 0,
                "why": {
                    "zh": "SamplingBatchInfo 把每请求参数打包成张量，所以 div_(temperatures) 之类操作天然逐请求向量化，一个 batch 里参数可各不相同、一次算完。无需逐条单跑，也不是共用或只取第一条——这正是高吞吐批处理的关键。",
                    "en": "SamplingBatchInfo packs per-request params into tensors, so ops like div_(temperatures) are inherently per-request vectorized: params can differ across the batch and are computed in one pass. No per-request looping, no shared-only or first-only behavior — this is key to high-throughput batching.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“按词表开的加权摸彩”这个类比，把采样管线从头讲透。请覆盖：①起点——model.forward（第 24 课）出的 logits 是词表里每个 token 一个分数，为什么它还不是答案；②管线顺序——惩罚（repetition/frequency/presence，反复读机）→ 温度（÷T：&lt;1 尖、&gt;1 平、=0 贪心，且只改悬殊不改排名）→ top-k（限个数）/ top-p（限累计概率）/ min-p（设相对地板）三道闸门 → softmax → 多项式采样，并说明每一步为什么排在这个位置；③贪心 vs 采样的区别（确定可复现 vs 随机多样），以及“截断管质量、温度管多样性”这句话的含义。",
                "en": "Using the 'weighted lottery over the vocabulary' analogy, fully explain the sampling pipeline from scratch. Cover: (1) the start — model.forward (Lesson 24) yields logits, one score per vocab token, and why that isn't the answer yet; (2) the pipeline order — penalties (repetition/frequency/presence, anti-parrot) → temperature (÷T: &lt;1 sharp, &gt;1 flat, =0 greedy, and it only changes lopsidedness not ranking) → top-k (caps count) / top-p (caps cumulative prob) / min-p (relative floor) gates → softmax → multinomial sample, explaining why each step sits where it does; (3) greedy vs sampling (deterministic/reproducible vs random/diverse), and what 'truncation governs quality, temperature governs diversity' means.",
            },
            {
                "zh": "解释为什么 Sampler 是很多功能<strong>挂钩的枢纽</strong>，以及它如何接回整条主线。请说明：①结构化输出（第 48 课）如何在采样前把违规 token 的 logit 置 −∞，从机制上保证语法合法；②同一位置还挂着哪些钩子——logit-bias、min_tokens/EOS 抑制、确定性推理（钉死 RNG 种子）；③为什么“同一 batch 不同请求可用不同参数”在工程上重要（SamplingBatchInfo 把参数打包成张量、向量化）；④把采样接回请求循环（第 18 课）的第 4 步与自回归（第 4 课）：采到的 token 追加回 Req、再喂下一步。",
                "en": "Explain why the Sampler is a <strong>hub where many features hook in</strong>, and how it ties back to the main line. Cover: (1) how structured output (Lesson 48) sets violating tokens' logits to −∞ before sampling, guaranteeing grammar validity by mechanism; (2) what other hooks live at the same spot — logit-bias, min_tokens/EOS suppression, deterministic inference (pinning the RNG seed); (3) why 'different requests in one batch can use different params' matters in practice (SamplingBatchInfo packs params into tensors, vectorized); (4) tie sampling back to step 4 of the request loop (Lesson 18) and autoregression (Lesson 4): the sampled token is appended to the Req and fed into the next step.",
            },
        ],
    },
    "29-radixattention-implementation.html": {
        "mcq": [
            {
                "q": {
                    "zh": "一个 <strong>TreeNode 的 value</strong> 字段到底装的是什么？",
                    "en": "What does a <strong>TreeNode's value</strong> field actually hold?",
                },
                "opts": [
                    {
                        "zh": "装的是这段 token 对应的 <strong>KV 槽位号（indices）</strong>——也就是<strong>指向显存池（第 30 课）的指针</strong>，而<strong>不是 KV 张量本身</strong>。树只当索引，真正的 K/V 张量躺在池里；两条请求共享前缀，就是它们拿到同一串 indices、指向同一批物理槽位",
                        "en": "It holds the <strong>KV slot numbers (indices)</strong> for this run — i.e. <strong>pointers into the memory pool (Lesson 30)</strong>, <strong>not the KV tensors themselves</strong>. The tree is just the index; the real K/V tensors live in the pool. Two requests sharing a prefix get the same run of indices pointing at the same physical slots",
                    },
                    {"zh": "装的是这段 token 的 K/V 张量本体", "en": "It holds the actual K/V tensors for this run"},
                    {"zh": "装的是这段 token 的原始文本字符串", "en": "It holds the raw text string of this run"},
                    {"zh": "装的是子节点的列表", "en": "It holds the list of child nodes"},
                ],
                "answer": 0,
                "why": {
                    "zh": "树是索引、池是存储是本课的灵魂：value 存的是指向池子的 indices，不是张量。正因为存的是指针，复用前缀才能零拷贝——两请求拿到同一串 indices 就指向同一批物理槽位。张量本体在第 30 课的池里；子节点在 children 字段，不在 value。",
                    "en": "Tree-is-index, pool-is-storage is the soul of this lesson: value stores indices into the pool, not tensors. Because it's a pointer, prefix reuse is zero-copy — two requests get the same indices pointing at the same physical slots. The tensors live in the Lesson-30 pool; children live in the children field, not value.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>match_prefix</span> 下行时，<strong>什么情况下会调用 <span class='mono'>_split_node</span> 把一个节点切成两层</strong>？",
                    "en": "While <span class='mono'>match_prefix</span> descends, <strong>when does it call <span class='mono'>_split_node</span> to cut a node into two levels</strong>?",
                },
                "opts": [
                    {
                        "zh": "当传入 token 只匹配了某条边 <span class='mono'>key</span> 的<strong>前一部分</strong>就分叉时（<span class='mono'>prefix_len &lt; len(child.key)</span>）：必须在<strong>分歧点</strong>把节点切开，公共前缀升为可共享的父节点、原来的尾巴降为子节点，这样新旧两条路径才能共享前半段",
                        "en": "When the incoming tokens match only the <strong>front part</strong> of an edge's <span class='mono'>key</span> before diverging (<span class='mono'>prefix_len &lt; len(child.key)</span>): the node must be split at the <strong>divergence point</strong> — the common prefix becomes a shareable parent and the original tail becomes a child, so the old and new paths can share the front part",
                    },
                    {"zh": "每次匹配到一个完整节点时都会分裂一次", "en": "It splits once every time a full node is matched"},
                    {"zh": "当某个节点的 lock_ref 超过阈值时", "en": "When a node's lock_ref exceeds a threshold"},
                    {"zh": "当显存池满了需要驱逐时", "en": "When the pool is full and needs eviction"},
                ],
                "answer": 0,
                "why": {
                    "zh": "分裂只发生在“半条边匹配”这一刻：代码里 prefix_len < len(child.key) 为真就调 _split_node，在分歧点把公共前缀提为父、尾巴降为子。整边命中不分裂，直接收下继续下探。分裂与 lock_ref、驱逐无关——它纯粹是为了让共享前缀按需长出来。",
                    "en": "Splitting happens only at the 'half-edge match' instant: when prefix_len < len(child.key) the code calls _split_node, lifting the common prefix into a parent and trimming the tail into a child. A whole-edge hit doesn't split — it's taken and descent continues. Splitting is unrelated to lock_ref or eviction; it purely grows shared prefixes on demand.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>inc_lock_ref</span> 从命中节点<strong>一路向上给每个祖先加锁</strong>，它到底在防什么？",
                    "en": "<span class='mono'>inc_lock_ref</span> walks <strong>upward locking every ancestor</strong> from the matched node — what is it actually preventing?",
                },
                "opts": [
                    {
                        "zh": "防止<strong>正在被在跑请求使用的 KV 被驱逐回收</strong>：驱逐是从叶子往上回收的（第 32 课），只要这段前缀在飞，它的每个祖先都不能被当成可驱逐叶子清掉，否则请求会读到一片被回收的垃圾槽位。请求结束再 <span class='mono'>dec_lock_ref</span> 逐节点解锁",
                        "en": "It prevents <strong>KV currently in use by a running request from being evicted</strong>: eviction reclaims from leaves upward (Lesson 32), so while this prefix is in flight none of its ancestors may be treated as an evictable leaf, or the request would read reclaimed garbage slots. On finish, <span class='mono'>dec_lock_ref</span> unlocks node by node",
                    },
                    {"zh": "防止两条请求同时匹配到同一个前缀", "en": "It prevents two requests from matching the same prefix at once"},
                    {"zh": "防止节点的 key 被其它请求修改", "en": "It prevents a node's key from being modified by other requests"},
                    {"zh": "防止树的深度超过最大限制", "en": "It prevents the tree depth from exceeding a maximum"},
                ],
                "answer": 0,
                "why": {
                    "zh": "lock_ref 守护“在用的 KV 不被回收”这条铁律。驱逐自叶向上回收，所以要把命中节点到根的整条链都加锁，任一祖先被清都会让在飞请求读到垃圾。共享前缀本就允许多请求同时命中（不是要防这个）；锁与改 key、限深度无关。",
                    "en": "lock_ref guards the rule that in-use KV is never reclaimed. Eviction reclaims leaf-upward, so the whole chain from the matched node to the root is locked; reclaiming any ancestor would let an in-flight request read garbage. Shared prefixes are meant to be hit by many requests at once (not what this prevents); locking is unrelated to editing keys or limiting depth.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“共享文件夹树（路径 trie）”这个类比，把 <span class='mono'>RadixCache</span> 的数据结构与三大操作讲透。请覆盖：①一个 <span class='mono'>TreeNode</span> 的五个字段各装什么——key（边上一段 token）、children（首 token 索引的字典）、value（指向池子的 KV indices，<strong>不是张量</strong>，第 30 课）、lock_ref（在用计数）、last_access_time（LRU，第 32 课）；②<span class='mono'>match_prefix</span> 如何沿 children 下行、逐边匹配，以及“半条边匹配”时 <span class='mono'>_split_node</span> 为什么要在分歧点把公共前缀提为父节点；③为什么说“树是索引、池是存储”，以及共享的本质是复用指针而非复制 KV。",
                "en": "Using the 'shared folder tree (path trie)' analogy, fully explain <span class='mono'>RadixCache</span>'s data structure and three core operations. Cover: (1) what each of a <span class='mono'>TreeNode</span>'s five fields holds — key (a run of tokens on the edge), children (dict keyed by first token), value (KV indices pointing into the pool, <strong>not tensors</strong>, Lesson 30), lock_ref (in-use count), last_access_time (LRU, Lesson 32); (2) how <span class='mono'>match_prefix</span> descends via children matching edge by edge, and why a half-edge match makes <span class='mono'>_split_node</span> lift the common prefix into a parent at the divergence point; (3) why 'tree is index, pool is storage,' and that sharing reuses pointers rather than copying KV.",
            },
            {
                "zh": "把 <span class='mono'>match_prefix</span> → <span class='mono'>insert</span> → <span class='mono'>inc/dec_lock_ref</span> 串成一条请求生命周期讲清楚。请说明：①match 返回的两样东西（命中的 KV indices 用于复用、最深匹配节点用作落脚点）；②insert 如何把分叉后缀挂成新子节点、并与 match 共用同一套分裂逻辑保持树规范；③请求开始用某段前缀时为什么要从命中节点<strong>向上锁到根</strong>（驱逐自叶向上，第 32 课），结束时 dec_lock_ref 如何逐节点解锁、归零后才重新可驱逐；④把这棵树接回上层：概念动机在第 7 课、indices 指向的池在第 30 课、HiCache 分层子类在第 31 课、缓存感知调度在第 20 课。",
                "en": "Trace a request's lifecycle through <span class='mono'>match_prefix</span> → <span class='mono'>insert</span> → <span class='mono'>inc/dec_lock_ref</span>. Cover: (1) the two things match returns (matched KV indices for reuse, deepest matched node as a foothold); (2) how insert attaches the diverging suffix as a new child and shares match's split logic to keep the tree canonical; (3) why starting to use a prefix walks the lock <strong>upward to the root</strong> (eviction is leaf-upward, Lesson 32), and how dec_lock_ref unlocks node by node, with a node becoming evictable again only at zero; (4) tie the tree back up the stack: the concept motivation in Lesson 7, the pool the indices point into in Lesson 30, the HiCache tiering subclass in Lesson 31, and cache-aware scheduling in Lesson 20.",
            },
        ],
    },
    "31-hicache-tiering.html": {
        "mcq": [
            {
                "q": {
                    "zh": "HiCache 把前缀缓存铺在<strong>三层</strong>上，这三层从快到慢、从小到大依次是什么？",
                    "en": "HiCache spreads the prefix cache across <strong>three tiers</strong> — from fastest/smallest to slowest/largest, what are they?",
                },
                "opts": [
                    {
                        "zh": "<strong>GPU HBM（热）→ CPU 主机内存（温）→ 磁盘 / 对象存储（冷）</strong>：HBM 最快最贵最小、前向直接读；CPU 内存大 10–100 倍、当写回暂存层；磁盘几乎无限、放超大共享前缀。被驱逐的 KV 向下沉、命中时向上取",
                        "en": "<strong>GPU HBM (hot) → CPU host memory (warm) → disk / object store (cold)</strong>: HBM is fastest/priciest/smallest and read directly by the forward; CPU RAM is 10–100× bigger as the writeback stash; disk is near-infinite for huge shared prefixes. Evicted KV sinks down, hits fetch up",
                    },
                    {"zh": "L1 缓存 → L2 缓存 → L3 缓存，全在 GPU 芯片内部", "en": "L1 → L2 → L3 cache, all inside the GPU chip"},
                    {"zh": "磁盘（热）→ CPU 内存（温）→ GPU HBM（冷），越往下越快", "en": "Disk (hot) → CPU RAM (warm) → GPU HBM (cold), faster going down"},
                    {"zh": "三个不同 GPU 的 HBM，靠 NVLink 连起来", "en": "Three different GPUs' HBM linked by NVLink"},
                ],
                "answer": 0,
                "why": {
                    "zh": "三层是 GPU HBM（热、最快最小）→ CPU 内存（温、大 10–100 倍）→ 磁盘/对象存储（冷、近乎无限）。方向是被驱逐的 KV 写回下沉、命中时预取上移。它不是片上 L1/L2/L3，也不是多 GPU 的 HBM；把顺序倒过来（磁盘最快）更是错的。",
                    "en": "The tiers are GPU HBM (hot, fastest/smallest) → CPU RAM (warm, 10–100× bigger) → disk/object store (cold, near-infinite). The direction is: evicted KV writes back down, hits prefetch up. It's not on-chip L1/L2/L3, nor multi-GPU HBM; reversing the order (disk fastest) is wrong.",
                },
            },
            {
                "q": {
                    "zh": "为什么 <span class='mono'>HiCacheController</span> 要把<strong>写回和预取放到后台线程 / 拷贝流</strong>上、与 GPU 计算重叠，而不是在主调度循环里同步做？",
                    "en": "Why does <span class='mono'>HiCacheController</span> run <strong>writeback and prefetch on background threads / copy streams</strong> overlapped with GPU compute, instead of doing them synchronously in the main scheduling loop?",
                },
                "opts": [
                    {
                        "zh": "否则调度器会<strong>停下来干等一次 CPU↔GPU 拷贝</strong>，把本该省下的时间又赔进去。放后台与计算重叠（第 21 课精神）后，当前批在 GPU 算前向的同时，控制器并行地把上批 KV 往下搬、把下批要用的 KV 往上预取，等前向真正要用时它<strong>已经</strong>在 HBM 里了",
                        "en": "Otherwise the scheduler would <strong>stall waiting on a CPU↔GPU copy</strong>, giving back the time it was supposed to save. Run in the background overlapped with compute (Lesson 21's spirit), so while the current batch runs its forward the controller shuttles the last batch's KV down and prefetches the next batch's KV up in parallel — by the time the forward needs it, it's <strong>already</strong> in HBM",
                    },
                    {"zh": "后台线程能让拷贝本身变得更快", "en": "Background threads make the copy itself faster"},
                    {"zh": "因为 Python 的 GIL 不允许在主线程里做拷贝", "en": "Because Python's GIL forbids copies on the main thread"},
                    {"zh": "为了把 KV 数据加密后再落盘", "en": "To encrypt the KV before writing to disk"},
                ],
                "answer": 0,
                "why": {
                    "zh": "拷贝要花时间，若让调度器同步等一次 CPU↔GPU 拷贝，就抵消了 HiCache 省下的重算。放后台 + 与计算重叠（第 21 课），I/O 与前向并行，前向要用时 KV 已就位，调度器几乎不为 I/O 阻塞。后台并不会让拷贝更快，也与 GIL、加密无关。",
                    "en": "Copies take time; making the scheduler synchronously wait on a CPU↔GPU copy would cancel the recompute HiCache saved. Background + overlap with compute (Lesson 21) runs I/O in parallel with the forward, so the KV is ready when needed and the scheduler is almost never blocked on I/O. Background doesn't speed the copy itself, and it's unrelated to the GIL or encryption.",
                },
            },
            {
                "q": {
                    "zh": "相比第 29 课朴素的 HBM-only 基数树，HiCache 在一次驱逐 → 再命中时，本质上<strong>用什么换了什么</strong>？",
                    "en": "Compared with Lesson 29's naive HBM-only radix tree, on an evict-then-rehit, what does HiCache fundamentally <strong>trade for what</strong>?",
                },
                "opts": [
                    {
                        "zh": "用一次<strong>便宜的 CPU→GPU 拷贝</strong>，换掉一次<strong>昂贵的前向重算</strong>。朴素缓存驱逐即丢弃，再命中等于没命中、得从头算几千 token；HiCache 把驱逐变成降级（写回下层），再命中只需把备份拷回 GPU",
                        "en": "A cheap <strong>CPU→GPU copy</strong> in place of an expensive <strong>forward recompute</strong>. The naive cache drops on eviction, so a rehit is a miss and recomputes thousands of tokens; HiCache turns eviction into demotion (writeback to a lower tier), so a rehit just copies the backup back to GPU",
                    },
                    {"zh": "用更多 GPU 显存，换更低的 CPU 占用", "en": "More GPU memory in exchange for lower CPU usage"},
                    {"zh": "用更高的精度，换更小的模型体积", "en": "Higher precision in exchange for a smaller model"},
                    {"zh": "用更长的上下文窗口，换更短的输出", "en": "A longer context window in exchange for shorter outputs"},
                ],
                "answer": 0,
                "why": {
                    "zh": "HiCache 的核心交易是“拷贝换计算”：把驱逐从“丢弃→重算”改成“降级→拷回”，再命中只付一次便宜的 CPU→GPU 拷贝，省掉一次昂贵的前向。它不是省显存换 CPU（反而要多花 CPU/磁盘），也与精度、上下文窗口无关。",
                    "en": "HiCache's core trade is copy-for-compute: it changes eviction from drop→recompute into demote→copy-back, so a rehit pays one cheap CPU→GPU copy and skips an expensive forward. It's not saving GPU memory at the cost of CPU (it actually spends extra CPU/disk), and it's unrelated to precision or context window.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“书桌 → 抽屉 → 异地仓库”这个三层类比，把 HiCache 的机制讲透。请覆盖：①三层各对应什么（GPU HBM 热 / CPU 内存 温 / 磁盘 冷），它们在<strong>延迟与容量</strong>上的取舍；②<span class='mono'>HiRadixCache</span> 为什么是第 29 课 <span class='mono'>RadixCache</span> 的<strong>子类</strong>、它在 <span class='mono'>TreeNode</span> 上多挂的 <span class='mono'>host_value</span> 用来记什么；③两个方向的动作——驱逐时 <span class='mono'>write_backup</span> 把 KV<strong>写回下沉</strong>、命中下层时 <span class='mono'>load_back</span> <strong>预取上移</strong>；④为什么这套机制相对朴素 HBM-only 基数树（第 29 课）能提升有效命中率。",
                "en": "Using the 'desk → drawer → off-site warehouse' three-tier analogy, fully explain HiCache. Cover: (1) what each tier maps to (GPU HBM hot / CPU RAM warm / disk cold) and their <strong>latency vs capacity</strong> trade-offs; (2) why <span class='mono'>HiRadixCache</span> is a <strong>subclass</strong> of Lesson 29's <span class='mono'>RadixCache</span> and what the extra <span class='mono'>host_value</span> field on <span class='mono'>TreeNode</span> records; (3) the two directional actions — <span class='mono'>write_backup</span> <strong>writes KV back down</strong> on eviction, <span class='mono'>load_back</span> <strong>prefetches up</strong> on a lower-tier hit; (4) why this raises the effective hit rate over the naive HBM-only tree of Lesson 29.",
            },
            {
                "zh": "说清 HiCache 的<strong>代价、收益与适用场景</strong>，并把它接回相邻几课。请说明：①核心交易是“用一次便宜的 CPU→GPU 拷贝换掉一次昂贵的重算”，以及为什么写回 / 预取要放<strong>后台线程并与计算重叠</strong>（第 21 课精神）、不堵调度；②代价有哪些（额外 CPU 内存 / 磁盘、拷贝带宽、多层一致性复杂度），以及为什么 HiCache 是<strong>可选开关</strong>；③它对哪类负载收益最大（大前缀、高复用、塞不进 HBM——长系统提示、大 RAG、多轮聊天），收益最终体现为吞吐 / 延迟（第 8 课）；④把它接回上下文：索引指向的显存池见第 30 课、驱逐与命中率见第 32 课。",
                "en": "Explain HiCache's <strong>cost, payoff, and where it fits</strong>, tying it back to neighboring lessons. Cover: (1) the core trade — one cheap CPU→GPU copy replacing one expensive recompute — and why writeback/prefetch run on <strong>background threads overlapped with compute</strong> (Lesson 21's spirit) without stalling the scheduler; (2) the costs (extra CPU RAM/disk, copy bandwidth, cross-tier consistency complexity) and why HiCache is an <strong>optional flag</strong>; (3) which workloads benefit most (big prefixes, high reuse, won't fit in HBM — long system prompts, big RAG, many-turn chats), with the payoff showing up as throughput/latency (Lesson 8); (4) tie it back: the memory pool the indices point into is Lesson 30, and eviction & hit rate is Lesson 32.",
            },
        ],
    },
    "32-eviction-and-hit-rate.html": {
        "mcq": [
            {
                "q": {
                    "zh": "显存吃紧、默认用 LRU 策略时，<span class='mono'>evict</span> 会优先清掉<strong>哪一个</strong>节点？",
                    "en": "Under memory pressure with the default LRU strategy, which node does <span class='mono'>evict</span> clear <strong>first</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>最久未被访问、且 lock_ref=0 的那个可驱逐叶子</strong>：LRU 的 <span class='mono'>get_priority</span> 直接返回 <span class='mono'>last_access_time</span>，时间最老的排在堆顶最先被清；而且只有叶子能驱逐、被锁的根本不在候选集里",
                        "en": "<strong>The oldest-untouched evictable leaf with lock_ref=0</strong>: LRU's <span class='mono'>get_priority</span> just returns <span class='mono'>last_access_time</span>, so the oldest sits at the heap top and goes first; only leaves are evictable and locked nodes aren't even candidates",
                    },
                    {"zh": "命中次数最多的那个热门前缀，因为它最占地方", "en": "The most-hit popular prefix, because it takes the most space"},
                    {"zh": "树里最深的那个节点，不管它是不是叶子", "en": "The deepest node in the tree, whether or not it's a leaf"},
                    {"zh": "随便挑一个，驱逐是随机的", "en": "Any node at random — eviction is random"},
                ],
                "answer": 0,
                "why": {
                    "zh": "默认 LRU 的优先级就是 last_access_time，最久未访问者排堆顶先走；同时只有 lock_ref=0 的叶子才在可驱逐集里。清最热门的恰恰相反，深节点若非叶子不能清，驱逐也绝非随机。",
                    "en": "Default LRU's priority is last_access_time, so the oldest-untouched pops first; and only lock_ref=0 leaves are in the evictable set. Clearing the hottest is backwards, a non-leaf deep node can't be cleared, and eviction is not random.",
                },
            },
            {
                "q": {
                    "zh": "为什么<strong>只有叶子</strong>可被驱逐，而一个 <span class='mono'>lock_ref&gt;0</span> 的节点<strong>永远</strong>不可被驱逐？",
                    "en": "Why are <strong>only leaves</strong> evictable, and why is a <span class='mono'>lock_ref&gt;0</span> node <strong>never</strong> evictable?",
                },
                "opts": [
                    {
                        "zh": "<strong>非叶节点下面还挂着更长的路径，那条路径依赖这段前缀的 KV，抽掉它孩子就悬空</strong>；而 lock_ref&gt;0 表示有在跑的请求正用这段前缀——它的前向正在读那批 KV 槽位，回收了就会读到被覆盖的垃圾、结果崩坏。所以回收只能从叶往根、且绕开在用链",
                        "en": "<strong>A non-leaf still has longer paths hanging below that depend on this prefix's KV — pull it and its children dangle</strong>; and lock_ref&gt;0 means a running request is using this prefix — its forward is reading those KV slots, so reclaiming them would read overwritten garbage and corrupt the result. Reclaim must go leaf-to-root and skip the in-use chain",
                    },
                    {"zh": "叶子比内部节点占用更多显存，清它们最划算", "en": "Leaves use more memory than internal nodes, so clearing them pays most"},
                    {"zh": "因为内部节点没有 KV 槽位可释放", "en": "Because internal nodes have no KV slots to free"},
                    {"zh": "lock_ref 只是个统计字段，对驱逐没有实际约束", "en": "lock_ref is just a stats field with no real constraint on eviction"},
                ],
                "answer": 0,
                "why": {
                    "zh": "内部节点有更长路径依赖其前缀 KV，清它会让孩子悬空，所以回收从叶往根。lock_ref>0 的节点正被在飞前向读着，回收其槽位会破坏正确性，是铁律而非统计。内部节点同样持有 KV 槽位，体积也并非叶子更大。",
                    "en": "Internal nodes have longer paths depending on their prefix KV; clearing them dangles children, so reclaim is leaf-to-root. A lock_ref>0 node is being read by an in-flight forward, so freeing its slots breaks correctness — an iron rule, not mere stats. Internal nodes also hold KV slots, and leaves aren't bigger.",
                },
            },
            {
                "q": {
                    "zh": "<strong>命中率</strong>衡量什么，为什么说它最终等价于<strong>吞吐</strong>？",
                    "en": "What does <strong>hit rate</strong> measure, and why is it ultimately equivalent to <strong>throughput</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>它衡量有多少 prompt token 是从缓存直接拿到、不必重算的比例</strong>。命中越多→需要跑前向重算的 token 越少→同样的 GPU 时间能服务更多请求，于是吞吐越高（第 8 课）。高前缀共享（第 7/29 课）+ 缓存感知调度（第 20 课）把命中率抬上去，HiCache（第 31 课）再抬高有效命中率",
                        "en": "<strong>It measures the fraction of prompt tokens served straight from the cache with no recompute</strong>. More hits → fewer tokens to recompute in a forward → the same GPU time serves more requests, so throughput rises (Lesson 8). High prefix sharing (Lessons 7/29) + cache-aware scheduling (Lesson 20) lift it; HiCache (Lesson 31) raises the effective hit rate",
                    },
                    {"zh": "它衡量 GPU 显存占用率，越满吞吐越高", "en": "It measures GPU memory utilization — fuller means higher throughput"},
                    {"zh": "它衡量模型预测下一个 token 的准确率", "en": "It measures the model's next-token prediction accuracy"},
                    {"zh": "它衡量网络请求的成功率，与重算无关", "en": "It measures network request success rate, unrelated to recompute"},
                ],
                "answer": 0,
                "why": {
                    "zh": "命中率 = 从缓存白拿、不必重算的 prompt token 占比。命中越多、重算越少，同样 GPU 时间服务更多请求 → 吞吐越高（第 8 课）。它不是显存占用率、不是预测准确率、也不是网络成功率。",
                    "en": "Hit rate = the fraction of prompt tokens taken free from the cache without recompute. More hits, less recompute, more requests per unit GPU time → higher throughput (Lesson 8). It's not memory utilization, not prediction accuracy, and not network success rate.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“图书管理员清理书架”的类比，把驱逐机制讲透。请覆盖：①驱逐<strong>何时</strong>被触发（节点占着第 30 课池里的 KV 槽位，分配器凑不出新请求要的空间时，调度器才触发 <span class='mono'>evict</span> 回收若干槽位），为什么它是被动按需、而非定时清理；②<strong>清谁</strong>由 <span class='mono'>EvictionStrategy</span> 的 <span class='mono'>get_priority</span> 排序，默认 LRU=<span class='mono'>last_access_time</span>（最久未访问先走），并简述 LFU / FIFO / MRU 的区别；③为什么<strong>只有叶子</strong>可驱逐、回收从叶往根；④铁律：<span class='mono'>lock_ref&gt;0</span> 的节点（正被在跑请求使用，第 29 课）永不被驱逐，<span class='mono'>inc/dec_lock_ref</span> 如何在可驱逐集与受保护集之间搬节点。",
                "en": "Using the 'librarian weeding the shelves' analogy, fully explain eviction. Cover: (1) <strong>when</strong> it triggers (nodes hold KV slots from Lesson 30's pool; only when the allocator can't assemble the slots a new request needs does the scheduler fire <span class='mono'>evict</span> to reclaim some), and why it's reactive/on-demand rather than scheduled cleanup; (2) <strong>who</strong> gets cleared, ordered by <span class='mono'>EvictionStrategy</span>'s <span class='mono'>get_priority</span>, default LRU=<span class='mono'>last_access_time</span> (oldest-untouched first), and briefly LFU / FIFO / MRU; (3) why <strong>only leaves</strong> are evictable and reclaim goes leaf-to-root; (4) the iron rule: a <span class='mono'>lock_ref&gt;0</span> node (in use by a running request, Lesson 29) is never evicted, and how <span class='mono'>inc/dec_lock_ref</span> move nodes between the evictable and protected sets.",
            },
            {
                "zh": "说清前缀缓存的<strong>经济学</strong>，并把它接回相邻几课。请说明：①<strong>命中率</strong>衡量什么（从缓存白拿、不重算的 prompt token 占比），它如何被高前缀共享（第 7/29 课）+ 缓存感知调度（第 20 课，主动重排队列优先放命中请求）抬高，又如何转化为更高吞吐（第 8 课）；②核心<strong>取舍</strong>：驱逐later重算 vs 留在 HBM——留占显存挤掉并发、驱省显存却赔上一次几千 token 的重算，为什么整个内存部分（第 29–32 课）都在把这笔取舍做好；③HiCache（第 31 课）如何通过把被驱逐前缀沉到下层、抬高<strong>有效</strong>命中率。",
                "en": "Explain the <strong>economics</strong> of the prefix cache, tying it back to neighboring lessons. Cover: (1) what <strong>hit rate</strong> measures (fraction of prompt tokens taken from cache without recompute), how it's raised by high prefix sharing (Lessons 7/29) + cache-aware scheduling (Lesson 20, which reorders the queue to favor cache hits), and how it converts into higher throughput (Lesson 8); (2) the core <strong>trade</strong>: evict-and-recompute-later vs keep-in-HBM — keeping costs HBM and squeezes out concurrency, evicting costs a future thousands-token recompute, and why the whole memory part (Lessons 29–32) is about making this trade well; (3) how HiCache (Lesson 31) raises the <strong>effective</strong> hit rate by sinking evicted prefixes to lower tiers.",
            },
        ],
    },
    "30-paged-memory-pools.html": {
        "mcq": [
            {
                "q": {
                    "zh": "从一条请求出发取到它的<strong>物理 K/V</strong>，要走<strong>哪条两跳寻址链路</strong>？",
                    "en": "Starting from a request, what is the <strong>two-hop addressing path</strong> to its <strong>physical K/V</strong>?",
                },
                "opts": [
                    {
                        "zh": "请求 →（<span class='mono'>ReqToTokenPool</span> 按 <span class='mono'>req_pool_idx</span> 取行）→ 一串 <strong>token 槽位号 (indices)</strong> →（<span class='mono'>token_to_kv</span> 池按槽位号）→ <strong>每层的物理 K/V</strong>。账本只记编号、不装张量；仓库才装真正吃显存的 K/V",
                        "en": "request →(<span class='mono'>ReqToTokenPool</span> reads the row by <span class='mono'>req_pool_idx</span>)→ a run of <strong>token slot ids (indices)</strong> →(<span class='mono'>token_to_kv</span> pool by slot)→ <strong>per-layer physical K/V</strong>. The ledger holds only numbers, no tensors; the warehouse holds the K/V that actually eats memory",
                    },
                    {"zh": "请求直接在自己的表里存 K/V 张量，一跳到位", "en": "the request stores K/V tensors directly in its own table, one hop"},
                    {"zh": "请求 → 基数树节点 → 节点里直接存的 K/V 张量", "en": "request → radix tree node → K/V tensors stored directly in the node"},
                    {"zh": "请求 → 分配器 → 分配器内部缓存的 K/V", "en": "request → allocator → K/V cached inside the allocator"},
                ],
                "answer": 0,
                "why": {
                    "zh": "两个池、两跳是本课核心：ReqToTokenPool 是每请求私有账本（按 req_pool_idx 索引），只记 token 槽位号；token_to_kv 池（MHATokenToKVPool）才按槽位号存每层真正的 K/V。请求不直接存张量；树存的也是 indices 不是张量（第 29 课）；分配器只发槽/回收槽，不存 K/V。",
                    "en": "Two pools, two hops is the core: ReqToTokenPool is the per-request private ledger (indexed by req_pool_idx) holding only token slot numbers; the token_to_kv pool (MHATokenToKVPool) stores the real per-layer K/V by slot. The request doesn't store tensors; the tree also stores indices not tensors (Lesson 29); the allocator only hands out/reclaims slots, it stores no K/V.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>ReqToTokenPool</span> 和 <span class='mono'>token_to_kv</span> 池<strong>各自映射什么</strong>？为什么要拆成两张表？",
                    "en": "What does <span class='mono'>ReqToTokenPool</span> map vs. the <span class='mono'>token_to_kv</span> pool, and why split into two tables?",
                },
                "opts": [
                    {
                        "zh": "<span class='mono'>ReqToTokenPool</span> 映射“每条请求 → 它依次拥有的 token 槽位号列表”（每请求记账，按 <span class='mono'>req_pool_idx</span> 索引）；<span class='mono'>token_to_kv</span> 池映射“token 槽位号 → 每层物理 K/V”（共享大仓库）。拆开是为了让槽位能<strong>非连续</strong>（分页，第 6 课）且<strong>被多请求共享</strong>（第 29 课），账本只管“哪些编号是我的”",
                        "en": "<span class='mono'>ReqToTokenPool</span> maps 'each request → the list of token slot numbers it owns' (per-request bookkeeping, indexed by <span class='mono'>req_pool_idx</span>); the <span class='mono'>token_to_kv</span> pool maps 'token slot id → per-layer physical K/V' (one shared warehouse). The split lets slots be <strong>non-contiguous</strong> (paging, Lesson 6) and <strong>shared across requests</strong> (Lesson 29), while the ledger only tracks 'which numbers are mine'",
                    },
                    {"zh": "两张表装的内容完全一样，只是为了冗余备份", "en": "the two tables hold identical content, just for redundant backup"},
                    {"zh": "ReqToTokenPool 装 K/V 张量，token_to_kv 池装编号", "en": "ReqToTokenPool holds K/V tensors and the token_to_kv pool holds the numbers"},
                    {"zh": "拆开纯粹是历史遗留，没有实际作用", "en": "the split is purely a historical accident with no real purpose"},
                ],
                "answer": 0,
                "why": {
                    "zh": "账本（ReqToTokenPool）记编号、按 req_pool_idx 索引；仓库（token_to_kv 池）按槽位号存每层真 K/V。若合成一张让请求直接存 K/V，就既无法表达非连续的分页槽位、也无法让两请求共享同一批物理槽位——拆开正是为这两件事。两表内容不同、角色不同，绝非冗余或遗留。",
                    "en": "The ledger (ReqToTokenPool) holds numbers indexed by req_pool_idx; the warehouse (token_to_kv pool) holds the real per-layer K/V by slot. Merging them so a request stores K/V directly would make non-contiguous paged slots inexpressible and cross-request sharing impossible — the split exists for exactly these. The two tables differ in content and role; not redundancy or legacy.",
                },
            },
            {
                "q": {
                    "zh": "第 29 课基数树节点的 <span class='mono'>value</span> 既然是 <strong>indices 而不是张量</strong>，那“<strong>两请求共享前缀</strong>”在内存池里到底意味着什么？",
                    "en": "Since a Lesson-29 node's <span class='mono'>value</span> is <strong>indices, not tensors</strong>, what does 'two requests sharing a prefix' actually mean in the pools?",
                },
                "opts": [
                    {
                        "zh": "两条请求的账本里写着<strong>同一批 token 槽位号</strong>，于是注意力都去 <span class='mono'>token_to_kv</span> 池读<strong>同一批物理 K/V</strong>——共享部分<strong>零额外显存</strong>。驱逐（第 32 课）就是把这些槽位号还给分配器；正因为树存的是编号而非张量，复用才能零拷贝",
                        "en": "both requests' ledgers carry <strong>the same token slot numbers</strong>, so attention reads <strong>the same physical K/V</strong> in the <span class='mono'>token_to_kv</span> pool — the shared part costs <strong>zero extra memory</strong>. Eviction (Lesson 32) returns those slot numbers to the allocator; precisely because the tree stores numbers not tensors, reuse is zero-copy",
                    },
                    {"zh": "引擎把共享前缀的 K/V 复制一份给每条请求", "en": "the engine copies the shared prefix's K/V once per request"},
                    {"zh": "两请求各自重算一遍共享前缀的 K/V", "en": "each request recomputes the shared prefix's K/V on its own"},
                    {"zh": "共享只发生在 CPU 内存，GPU 上仍是各存各的", "en": "sharing only happens in CPU memory; on GPU each keeps its own copy"},
                ],
                "answer": 0,
                "why": {
                    "zh": "树存的是 indices，所以共享 = 复用指针：两请求账本指向同一批槽位号 → 同一批物理 K/V → 共享部分零额外显存。这正是不复制、不重算的根源。驱逐则把这串槽位号还回分配器（第 32 课）。说复制、重算或仅 CPU 共享都与“索引与存储分离”的设计相悖。",
                    "en": "The tree stores indices, so sharing = reusing pointers: both ledgers point at the same slot numbers → the same physical K/V → zero extra memory for the shared part. That's the root of no-copy, no-recompute. Eviction returns the run of slot numbers to the allocator (Lesson 32). Claims of copying, recomputing, or CPU-only sharing all contradict the index-vs-storage separation.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“衣帽间 + 储物柜大厅”这个类比，把分页内存池的<strong>两个池与两跳寻址</strong>讲透。请覆盖：①<span class='mono'>ReqToTokenPool</span> 为什么像“取衣牌”——它按 <span class='mono'>req_pool_idx</span> 索引、每行记“这条请求依次拥有哪些 token 槽位号”，<strong>只记编号不装张量</strong>；②<span class='mono'>MHATokenToKVPool</span> 为什么像“那排储物柜”——它为<strong>每一层</strong>各开一个大张量、按槽位号存真正的 K/V；③两跳寻址链路：请求 →（账本取行）→ token 槽位号 →（仓库按号取）→ 物理 K/V；④<span class='mono'>TokenToKVPoolAllocator</span> 为什么像“服务员”——以页（第 6 课）发槽/回收槽，总槽位在开机按 <span class='mono'>mem_fraction_static</span>（第 8 课）钉死 = 并发上限（第 4 课）。",
                "en": "Using the 'coat-check + locker room' analogy, fully explain the <strong>two pools and two-hop addressing</strong> of paged memory pools. Cover: (1) why <span class='mono'>ReqToTokenPool</span> is like the 'claim ticket' — indexed by <span class='mono'>req_pool_idx</span>, each row recording 'which token slot numbers this request owns, in order,' <strong>numbers only, no tensors</strong>; (2) why <span class='mono'>MHATokenToKVPool</span> is like 'that bank of lockers' — a big tensor <strong>per layer</strong>, storing the real K/V by slot; (3) the two-hop path: request →(ledger row)→ token slot ids →(warehouse by slot)→ physical K/V; (4) why <span class='mono'>TokenToKVPoolAllocator</span> is like the 'attendant' — handing out/reclaiming slots by page (Lesson 6), with total slots nailed at startup by <span class='mono'>mem_fraction_static</span> (Lesson 8) = the concurrency ceiling (Lesson 4).",
            },
            {
                "zh": "把“<strong>树是索引、池是存储</strong>”接回第 29 课，讲清楚共享与驱逐在内存层的真相。请说明：①为什么 <span class='mono'>TreeNode</span> 的 <span class='mono'>value</span> 装的是 token 槽位号（indices）而非 K/V 张量，“树是索引/去重层、池是存储层”如何分工；②“两请求共享前缀”在池子里意味着两份账本写着<strong>同一批槽位号</strong>、指向<strong>同一批物理 K/V</strong>、共享部分<strong>零额外显存</strong>；③为什么要把账本（ReqToTokenPool）和仓库（token_to_kv 池）<strong>拆成两张表</strong>——为支持非连续的分页槽位（第 6 课）与跨请求共享（第 29 课）；④驱逐（第 32 课）如何把 value 里那串槽位号<strong>还给分配器</strong>，让别的 token 来占。",
                "en": "Tie 'tree is index, pool is storage' back to Lesson 29 and explain the truth of sharing and eviction at the memory layer. Cover: (1) why a <span class='mono'>TreeNode</span>'s <span class='mono'>value</span> holds token slot numbers (indices) not K/V tensors, and how 'tree as index/dedup layer, pool as storage layer' divides labor; (2) why 'two requests sharing a prefix' means both ledgers carry <strong>the same slot numbers</strong>, pointing at <strong>the same physical K/V</strong>, with the shared part costing <strong>zero extra memory</strong>; (3) why the ledger (ReqToTokenPool) and warehouse (token_to_kv pool) are <strong>split into two tables</strong> — to support non-contiguous paged slots (Lesson 6) and cross-request sharing (Lesson 29); (4) how eviction (Lesson 32) <strong>returns</strong> the run of slot numbers in a node's value to the allocator so other tokens can occupy them.",
            },
        ],
    },
    "33-attention-backend-abstraction.html": {
        "mcq": [
            {
                "q": {
                    "zh": "模型 <span class='mono'>forward</span> 里那行 <span class='mono'>self.attn(q,k,v,forward_batch)</span>，调用的 <span class='mono'>RadixAttention</span> 层<strong>本身</strong>到底做了什么？",
                    "en": "What does the <span class='mono'>RadixAttention</span> layer <strong>itself</strong> — the one called by <span class='mono'>self.attn(q,k,v,forward_batch)</span> in a model's <span class='mono'>forward</span> — actually do?",
                },
                "opts": [
                    {
                        "zh": "<strong>它不含 kernel，只持有形状参数（头数 / head_dim / 缩放 / 层号），把真正的矩阵乘 + softmax 委托给一个 <span class='mono'>AttentionBackend</span></strong>。算注意力的脏活由后端的 kernel 干，层本身只是个稳定的调用入口",
                        "en": "<strong>It holds no kernel — only shape params (heads / head_dim / scale / layer id) — and delegates the real matmul + softmax to an <span class='mono'>AttentionBackend</span></strong>. The dirty work of computing attention is the backend's kernel; the layer is just a stable call site",
                    },
                    {"zh": "它内部直接写死了一段 FlashInfer CUDA kernel，亲自完成全部注意力计算", "en": "It hardcodes a FlashInfer CUDA kernel inside and computes all of attention itself"},
                    {"zh": "它负责把 KV 写到磁盘并管理驱逐策略", "en": "It writes KV to disk and manages the eviction policy"},
                    {"zh": "它只是一个占位层，前向时被直接跳过", "en": "It's just a placeholder layer, skipped at forward time"},
                ],
                "answer": 0,
                "why": {
                    "zh": "RadixAttention 是模型前向调用的那一层，但它不实现 kernel：它持有形状参数，把注意力数学委托给可替换的 AttentionBackend。写死 kernel 恰恰是它要避免的；驱逐/落盘是缓存子系统（第 31/32 课）的事；它绝非占位层。",
                    "en": "RadixAttention is the layer the model's forward calls, but it implements no kernel: it carries shape params and delegates the math to a swappable AttentionBackend. Hardcoding a kernel is exactly what it avoids; eviction/disk is the cache subsystem (Lessons 31/32); it is not a placeholder.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>forward_extend</span> 和 <span class='mono'>forward_decode</span> 为什么要做成<strong>两条分开的路径</strong>？",
                    "en": "Why are <span class='mono'>forward_extend</span> and <span class='mono'>forward_decode</span> kept as <strong>two separate paths</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>因为两种场景的形状与访存模式截然不同</strong>：EXTEND/prefill 是一批<strong>新 token</strong>一起做<strong>完整注意力</strong>（受因果掩码）；DECODE 是<strong>1 个 query token</strong> 注意<strong>整段已缓存的 KV</strong>。形状不同、最优 kernel 不同，分开才能各自最优（DECODE 还最适合 CUDA graph，第 27 课）",
                        "en": "<strong>Because the two cases differ sharply in shape and memory access</strong>: EXTEND/prefill is a batch of <strong>new tokens</strong> doing <strong>full attention</strong> (under the causal mask); DECODE is <strong>1 query token</strong> attending the <strong>entire cached KV</strong>. Different shapes, different optimal kernels — splitting lets each be optimal (and DECODE fits CUDA graph best, Lesson 27)",
                    },
                    {"zh": "纯粹是历史遗留，两条路径其实跑的是完全相同的代码", "en": "Purely historical; the two paths actually run identical code"},
                    {"zh": "一条给 NVIDIA 用、一条给 AMD 用", "en": "One is for NVIDIA and the other for AMD"},
                    {"zh": "extend 算注意力、decode 只负责采样下一个 token", "en": "extend computes attention while decode only samples the next token"},
                ],
                "answer": 0,
                "why": {
                    "zh": "EXTEND（prefill）是多个新 token 的完整注意力，DECODE 是单 query 对整段缓存 KV——算术形状与访存模式不同，所以拆成两条路径各用最优 kernel；DECODE 固定形状还便于 CUDA graph。两条路并非同一份代码，也不是按硬件区分，decode 同样要算注意力。",
                    "en": "EXTEND (prefill) is full attention over many new tokens; DECODE is a single query over the whole cached KV — different arithmetic shapes and access patterns, so they split to use the best kernel each; DECODE's fixed shape also suits CUDA graph. The paths aren't identical code, aren't split by hardware, and decode still computes attention.",
                },
            },
            {
                "q": {
                    "zh": "把注意力定义成 <span class='mono'>AttentionBackend</span> 这个抽象基类，最核心的好处是什么？",
                    "en": "What is the core benefit of defining attention as the <span class='mono'>AttentionBackend</span> abstract base class?",
                },
                "opts": [
                    {
                        "zh": "<strong>新 kernel、新硬件只要实现这份契约就能接进来，不必改任何模型文件</strong>。SGLang 支持几十个模型（第 26 课），若 kernel 写死在每个模型里，每出一个新后端都要逐个改——接口把“调用注意力”和“实现注意力”解耦，注意力成了可替换的策略对象",
                        "en": "<strong>A new kernel or new hardware plugs in just by implementing the contract, without touching any model file</strong>. SGLang supports dozens of models (Lesson 26); if the kernel were hardcoded per model, every new backend would mean editing them all — the interface decouples 'calling attention' from 'implementing attention,' making attention a swappable strategy object",
                    },
                    {"zh": "它能让注意力计算自动变快，无需任何 kernel 优化", "en": "It magically makes attention faster without any kernel optimization"},
                    {"zh": "它把所有后端的代码合并成一个文件，便于阅读", "en": "It merges every backend's code into one file for easier reading"},
                    {"zh": "它强制所有硬件都使用完全相同的 kernel", "en": "It forces all hardware to use the exact same kernel"},
                ],
                "answer": 0,
                "why": {
                    "zh": "接口的价值是可插拔：实现 AttentionBackend 的几个方法就能加新后端/新硬件，模型文件一行不动。它不会凭空提速（提速靠后端 kernel），不是把代码合并，更不是强制统一 kernel——恰恰相反，它让不同硬件用不同 kernel 而上层不变。",
                    "en": "The interface's value is pluggability: implement a few AttentionBackend methods to add a backend/hardware, model files untouched. It doesn't magically speed things up (the backend kernel does), doesn't merge code, and doesn't force one kernel — quite the opposite, it lets different hardware use different kernels while the top stays stable.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“<strong>可换头的电钻</strong>”这个类比，把注意力后端抽象讲透。请覆盖：①模型里的注意力层（<span class='mono'>RadixAttention</span> 的 <span class='mono'>nn.Module</span>，第 29 课）为什么像“钻身”——它<strong>不含 kernel</strong>，只持有形状参数，把真正的数学<strong>委托</strong>给 <span class='mono'>AttentionBackend</span>（ABC）；②有哪些“钻头”——FlashInfer（NV 高性能默认）、Triton（可移植兜底）、FlashAttention 3、AMD/NPU 等（第 42 课），由 <span class='mono'>--attention-backend</span> 或按硬件自动选；③模型作者只写 <span class='mono'>self.attn(q,k,v,forward_batch)</span>（第 26 课），为什么“换钻头不用改钻身”。",
                "en": "Using the '<strong>power drill with interchangeable bits</strong>' analogy, fully explain the attention backend abstraction. Cover: (1) why the model's attention layer (the <span class='mono'>RadixAttention</span> <span class='mono'>nn.Module</span>, Lesson 29) is like the 'drill body' — it <strong>holds no kernel</strong>, only shape params, and <strong>delegates</strong> the real math to an <span class='mono'>AttentionBackend</span> (ABC); (2) what the 'bits' are — FlashInfer (NV high-perf default), Triton (portable fallback), FlashAttention 3, AMD/NPU (Lesson 42), chosen by <span class='mono'>--attention-backend</span> or auto by hardware; (3) why the model author only writes <span class='mono'>self.attn(q,k,v,forward_batch)</span> (Lesson 26) and 'swapping the bit needs no change to the body.'",
            },
            {
                "zh": "说清一次注意力调用在后端内部<strong>从头到尾</strong>发生了什么，并接回相邻几课。请说明：①什么叫“<strong>规划元数据</strong>（plan metadata）”——在跑 kernel 之前，后端要先算清每条请求读 KV 池（第 30 课）的<strong>哪些页</strong>、<strong>因果掩码</strong>、<strong>序列长度</strong>，以及 CUDA graph（第 27 课）需要的固定形状缓冲；②后端如何按 <span class='mono'>forward_mode</span> 分派到 <strong>EXTEND/prefill</strong>（新 token 完整注意力）与 <strong>DECODE</strong>（1 个 query 注意整段缓存 KV）两条路；③为什么 DECODE 这条固定形状的路最适合 CUDA graph；④更底层的 kernel 怎么写（第 38/40 课）与多硬件（第 42 课）如何在这条接口线之下独立演化。",
                "en": "Explain what happens inside a backend during one attention call <strong>end to end</strong>, tying it back to neighboring lessons. Cover: (1) what 'plan metadata' means — before running the kernel, the backend computes which <strong>pages</strong> of the KV pool (Lesson 30) each request reads, the <strong>causal mask</strong>, the <strong>sequence lengths</strong>, and the fixed-shape buffers a CUDA graph (Lesson 27) needs; (2) how the backend dispatches by <span class='mono'>forward_mode</span> to <strong>EXTEND/prefill</strong> (new tokens, full attention) vs <strong>DECODE</strong> (1 query over the whole cached KV); (3) why the fixed-shape DECODE path best suits CUDA graph; (4) how lower-level kernels (Lessons 38/40) and multi-hardware support (Lesson 42) evolve independently below this interface line.",
            },
        ],
    },
    "34-moe-layer.html": {
        "mcq": [
            {
                "q": {
                    "zh": "一个 MoE 层里有 64 个专家、<span class='mono'>top_k=2</span>。路由器（router/gate）对<strong>每个 token</strong> 做了什么？",
                    "en": "An MoE layer has 64 experts with <span class='mono'>top_k=2</span>. What does the router/gate do for <strong>each token</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>给 64 个专家各打一个分，只挑出得分最高的 2 个真正参与计算</strong>，其余 62 个对这个 token 一动不动；最后按路由权重把这 2 个专家的输出加权合并。这就是<strong>稀疏激活</strong>",
                        "en": "<strong>It scores all 64 experts and keeps only the top-2 to actually compute</strong>; the other 62 do nothing for this token; the 2 outputs are then combined weighted by the routing scores. That is <strong>sparse activation</strong>",
                    },
                    {"zh": "让 token 依次穿过全部 64 个专家，再取平均", "en": "Sends the token through all 64 experts in turn, then averages"},
                    {"zh": "随机挑 2 个专家，不看 token 内容", "en": "Picks 2 experts at random, ignoring the token's content"},
                    {"zh": "整个 batch 共用同一个被选中的专家", "en": "The whole batch shares one single chosen expert"},
                ],
                "answer": 0,
                "why": {
                    "zh": "路由器是个很小的线性层，对每个 token 给所有专家打分并取 top-k（这里 2/64），只有这 2 个专家计算、再按权重合并——这正是稀疏。它不会让 token 走遍全部专家（那就退化成稠密了），不是随机挑，也不是整个 batch 共用一个：路由是<strong>逐 token</strong> 的。",
                    "en": "The router is a tiny linear layer that scores all experts per token and takes top-k (here 2 of 64); only those 2 compute, then combine by weight — that is sparsity. It does not run the token through all experts (that would be dense), is not random, and is not shared batch-wide: routing is <strong>per-token</strong>.",
                },
            },
            {
                "q": {
                    "zh": "为什么说 MoE “scale 参数而不 scale 每 token 的算力”？",
                    "en": "Why is MoE said to 'scale parameters, not per-token compute'?",
                },
                "opts": [
                    {
                        "zh": "<strong>专家越多，这一层的参数（知识容量）越大；但每个 token 只走固定的 <span class='mono'>top-k</span> 个专家，单 token 的 FLOPs 由 top-k 钉死，几乎不随专家总数增长</strong>。于是模型能做到千亿参数，而推理每 token 算力基本恒定",
                        "en": "<strong>More experts means more parameters (knowledge capacity) in the layer; but each token visits only a fixed <span class='mono'>top-k</span>, so per-token FLOPs are pinned by top-k and barely grow with the expert count</strong>. So the model can reach hundreds of billions of params while per-token inference compute stays roughly constant",
                    },
                    {"zh": "因为 MoE 的专家比稠密 FFN 算得更快，单个专家本身就提速了", "en": "Because an MoE expert is intrinsically faster to compute than a dense FFN"},
                    {"zh": "因为 MoE 不需要任何矩阵乘法", "en": "Because MoE needs no matrix multiplication at all"},
                    {"zh": "因为参数被压缩到几乎为零", "en": "Because the parameters are compressed to nearly zero"},
                ],
                "answer": 0,
                "why": {
                    "zh": "关键在解耦：参数随专家数涨（容量变大），而每 token 只激活 top-k 个专家，算力被 top-k 钉住。不是单个专家更快（专家就是小 FFN），更不是没有矩阵乘或参数为零——恰恰相反，参数很多，只是显存要装下全部专家。",
                    "en": "The point is decoupling: parameters grow with experts (more capacity), while each token activates only top-k, so compute is pinned by top-k. It's not that a single expert is faster (an expert is a small FFN), nor that there's no matmul or zero params — quite the opposite, params are many; memory must hold them all.",
                },
            },
            {
                "q": {
                    "zh": "SGLang 的 <span class='mono'>FusedMoE</span> 层把什么“融合（fuse）”进了 kernel？",
                    "en": "What does SGLang's <span class='mono'>FusedMoE</span> layer 'fuse' into kernels?",
                },
                "opts": [
                    {
                        "zh": "<strong>把路由 + 按专家分组 + 分组/批量 GEMM 融合进少数几个 kernel，避免那段“for 每个专家”的慢 Python 循环</strong>——否则会有几十次 kernel 启动和零碎小算子，GPU 大量空转",
                        "en": "<strong>It fuses routing + grouping-by-expert + grouped/batched GEMM into a few kernels, avoiding a slow 'for each expert' Python loop</strong> — which otherwise means dozens of kernel launches and tiny scattered ops with the GPU mostly idle",
                    },
                    {"zh": "把所有专家的权重压缩成一个专家", "en": "Compresses all experts' weights into a single expert"},
                    {"zh": "把注意力和 MoE 合并成同一层", "en": "Merges attention and MoE into the same layer"},
                    {"zh": "把多张 GPU 融合成一张逻辑 GPU", "en": "Fuses multiple GPUs into one logical GPU"},
                ],
                "answer": 0,
                "why": {
                    "zh": "FusedMoE 的“融合”指把路由、分组、分组 GEMM 压进少数 kernel，消灭逐专家的慢 Python 循环与零碎小算子。它不会把专家压成一个（那就没有混合专家了），不合并注意力，也不负责把多卡变一卡——跨卡是专家并行（第 46 课）的事。",
                    "en": "FusedMoE's 'fuse' means pressing routing, grouping, and grouped GEMM into a few kernels, killing the slow per-expert Python loop and tiny ops. It does not collapse experts into one (that would end the mixture), does not merge attention, and does not turn many GPUs into one — cross-GPU is expert parallelism (Lesson 46).",
                },
            },
        ],
        "open": [
            {
                "zh": "用“<strong>医院分诊台</strong>”这个类比，把 MoE 层讲透。请覆盖：①一个<strong>稠密 FFN</strong> 让每个 token 都穿过同一个大 MLP，参数和单 token 算力<strong>焊死</strong>；②MoE 层 = <strong>N 个小专家（FFN）+ 一个路由器/门控</strong>，路由器对<strong>每个 token</strong> 打分、取 <span class='mono'>top-k</span>（如 64 选 2），只有这 k 个专家计算 → <strong>稀疏</strong>；③为什么这样能“<strong>scale 参数而不 scale 每 token 算力</strong>”，并点名 DeepSeek-V3 / Mixtral / Qwen-MoE。",
                "en": "Using the '<strong>hospital triage desk</strong>' analogy, fully explain the MoE layer. Cover: (1) a <strong>dense FFN</strong> runs every token through the same big MLP, <strong>welding</strong> parameters to per-token compute; (2) an MoE layer = <strong>N small experts (FFNs) + a router/gate</strong>; the router scores <strong>each token</strong> and takes <span class='mono'>top-k</span> (e.g. 2 of 64), so only those k compute → <strong>sparse</strong>; (3) why this 'scales parameters, not per-token compute,' naming DeepSeek-V3 / Mixtral / Qwen-MoE.",
            },
            {
                "zh": "说清一次 MoE 前向<strong>从头到尾</strong>的计算，并接回相邻几课。请说明：①四步流程——<strong>路由 → 按专家分组 → 分组/批量 GEMM（第 38 课）→ 散回并按路由权重加权合并</strong>，以及 <span class='mono'>FusedMoE</span> 为什么要把这些融合进 kernel（避免逐专家的慢 Python 循环）；②扩展到多卡时，<strong>专家并行（EP，第 46 课）</strong>如何把专家摊到不同 GPU、token 经 all-to-all 分发到专家所在卡再合并；③<strong>EPLB（第 47 课）</strong>为什么要均衡“热门专家”，以及 MoE 的三笔代价（路由不均衡、all-to-all 通信、装下全部专家的显存）。",
                "en": "Explain one MoE forward <strong>end to end</strong>, tying it to neighboring lessons. Cover: (1) the four steps — <strong>route → group by expert → grouped/batched GEMM (Lesson 38) → scatter back and combine weighted by routing scores</strong> — and why <span class='mono'>FusedMoE</span> fuses these into kernels (avoiding a slow per-expert Python loop); (2) scaling out, how <strong>expert parallelism (EP, Lesson 46)</strong> spreads experts across GPUs and all-to-all dispatches tokens to their expert's card before combining; (3) why <strong>EPLB (Lesson 47)</strong> balances 'hot experts,' and MoE's three costs (routing imbalance, all-to-all comm, memory to hold all experts).",
            },
        ],
    },
    "35-quantization.html": {
        "mcq": [
            {
                "q": {
                    "zh": "量化把权重从 fp16 压到 INT4，本质上是用什么<strong>换</strong>什么？",
                    "en": "Quantizing weights from fp16 to INT4 fundamentally <strong>trades</strong> what for what?",
                },
                "opts": [
                    {
                        "zh": "<strong>用一点点精度，换显存、访存带宽，以及（在低精度硬件上）算力</strong>——更少的比特 + 一个 scale 近似原权重，于是模型更小、解码要搬的字节更少、跑得更快",
                        "en": "<strong>A little accuracy in exchange for HBM, memory bandwidth, and (on low-precision hardware) FLOPs</strong> — fewer bits + a scale approximate the original weights, so the model is smaller, decode moves fewer bytes, and it runs faster",
                    },
                    {"zh": "用更多显存换更高精度", "en": "More HBM in exchange for higher accuracy"},
                    {"zh": "用算力换模型质量，完全不省显存", "en": "Compute for model quality, saving no HBM at all"},
                    {"zh": "什么都不损失，纯粹免费加速", "en": "Loses nothing — pure free speedup"},
                ],
                "answer": 0,
                "why": {
                    "zh": "量化是一笔取舍：牺牲<strong>一点点精度</strong>，换来显存减半到四分之一、解码访存带宽减半，低精度硬件上还顺带省算力。它不是用更多显存换精度（恰恰相反），也不是不省显存，更不是完全无损——只是损失常常小到要跑评测才看得出。",
                    "en": "Quantization is a trade: give up <strong>a little accuracy</strong> for HBM halved-to-quartered, decode bandwidth halved, and FLOPs too on low-precision hardware. It is not more HBM for accuracy (the opposite), does not skip HBM savings, and is not lossless — the loss is just usually too small to see outside a benchmark.",
                },
            },
            {
                "q": {
                    "zh": "AWQ/GPTQ 这类“<strong>仅权重</strong>”量化和 FP8 这类“<strong>权重+激活</strong>”量化，最关键的差别是什么？",
                    "en": "What's the key difference between <strong>weight-only</strong> quant (AWQ/GPTQ) and <strong>weight + activation</strong> quant (FP8)?",
                },
                "opts": [
                    {
                        "zh": "<strong>仅权重</strong>把权重存成 4 位省显存，但计算前要<strong>反量化回高精度</strong>再做矩阵乘，省存+带宽不省算力；<strong>权重+激活</strong>两边都低精度，直接跑<strong>低精度 GEMM</strong>（FP8），<strong>连算力一起省</strong>，但更挑硬件和精度",
                        "en": "<strong>Weight-only</strong> stores 4-bit weights to save HBM but <strong>dequantizes back to high precision</strong> before the matmul, saving storage+bandwidth not FLOPs; <strong>weight+activation</strong> keeps both low precision and runs a <strong>low-precision GEMM</strong> (FP8), <strong>saving FLOPs too</strong>, but is pickier about hardware and accuracy",
                    },
                    {"zh": "仅权重更省显存，权重+激活完全不省显存", "en": "Weight-only saves HBM; weight+activation saves none"},
                    {"zh": "两者完全等价，只是名字不同", "en": "They are identical, just named differently"},
                    {"zh": "权重+激活只能用于训练，不能推理", "en": "Weight+activation is training-only, never for inference"},
                ],
                "answer": 0,
                "why": {
                    "zh": "仅权重（AWQ/GPTQ）算前要把 4 位权重<strong>反量化</strong>回高精度，所以只省存储和带宽，对访存密集的解码特别划算；权重+激活（FP8）两边都低精度，能直接跑 FP8 GEMM <strong>把算力也省了</strong>，吞吐最高但需要 FP8 硬件、精度更需小心。两者都省显存，并非等价，FP8 也照样用于推理。",
                    "en": "Weight-only (AWQ/GPTQ) must <strong>dequantize</strong> 4-bit weights to high precision before computing, so it saves only storage and bandwidth — great for memory-bound decode; weight+activation (FP8) keeps both low precision and runs FP8 GEMM to <strong>save FLOPs too</strong>, highest throughput but needs FP8 hardware and more care on accuracy. Both save HBM, they're not equivalent, and FP8 is very much used for inference.",
                },
            },
            {
                "q": {
                    "zh": "一组 INT4 权重共享的那个 <span class='mono'>scale</span>（缩放因子）到底是什么，<strong>per-group</strong> 粒度又意味着什么？",
                    "en": "What exactly is the <span class='mono'>scale</span> shared by a group of INT4 weights, and what does a <strong>per-group</strong> granularity mean?",
                },
                "opts": [
                    {
                        "zh": "<strong>scale 是把低位整数还原回真实数值范围的乘数</strong>：真实值 ≈ 整数 × scale；<strong>per-group</strong> 指每一小块权重各算一个 scale，比 per-tensor（整张一个）更细、精度更高，代价是 scale 本身要占点空间、算起来更复杂",
                        "en": "<strong>The scale is the multiplier that restores low-bit ints to a real numeric range</strong>: real ≈ int × scale; <strong>per-group</strong> means each small block of weights gets its own scale — finer than per-tensor (one for the whole weight) and more accurate, at the cost of the scales taking some space and complexity",
                    },
                    {"zh": "scale 是模型的学习率，训练时才用", "en": "The scale is the model's learning rate, used only in training"},
                    {"zh": "scale 就是 INT4 整数本身，没有额外的数", "en": "The scale is just the INT4 integer itself, with no extra number"},
                    {"zh": "per-group 表示整个模型只有一个全局 scale", "en": "Per-group means the whole model has a single global scale"},
                ],
                "answer": 0,
                "why": {
                    "zh": "scale 是反量化的乘数：把存下来的低位整数乘以 scale，才近似还原成原来的浮点值（真实值 ≈ 整数 × scale）。粒度从粗到细是 per-tensor → per-channel → per-group/blockwise，越细越能贴合局部动态范围、精度越高，但要存更多 scale、计算更复杂。它不是学习率，也不是整数本身，per-group 更不是全局一个 scale。",
                    "en": "The scale is the dequant multiplier: multiply the stored low-bit int by the scale to approximately recover the original float (real ≈ int × scale). Granularity goes per-tensor → per-channel → per-group/blockwise; finer fits the local dynamic range better for higher accuracy, but stores more scales and costs more compute. It's not a learning rate, not the integer itself, and per-group is certainly not one global scale.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“<strong>JPEG / MP3 压缩</strong>”这个类比把量化讲透，并算清它<strong>省的是什么</strong>。请覆盖：①权重平时是 fp16（16 位），量化用<strong>更少比特（FP8/FP4/INT4）+ 一个 scale</strong> 近似原值，<strong>真实值 ≈ 整数 × scale</strong>；②为什么<strong>大头省的是内存而非算力</strong>——显存减半到四分之一（给 KV 缓存和并发腾地方，第 4/8 课），以及解码<strong>访存密集</strong>、权重搬得少一半就快一半；③为什么低比特还能用——LLM 权重<strong>冗余、宽容</strong>，真正危险的是<strong>离群值</strong>，而 <strong>AWQ</strong>（按激活幅度保护关键通道）和 <strong>GPTQ</strong>（带校准集逐层最小化重建误差）正是靠聪明地处理它们把精度损失压到几乎看不出。",
                "en": "Using the '<strong>JPEG / MP3 compression</strong>' analogy, fully explain quantization and pin down <strong>what it saves</strong>. Cover: (1) weights are normally fp16 (16-bit); quantization approximates them with <strong>fewer bits (FP8/FP4/INT4) + a scale</strong>, where <strong>real ≈ int × scale</strong>; (2) why the <strong>big win is memory, not compute</strong> — HBM halved-to-quartered (room for KV cache and concurrency, Lessons 4/8), and decode being <strong>memory-bound</strong> so halving the weight bytes halves the move; (3) why low bits still work — LLM weights are <strong>redundant and tolerant</strong>, the real danger is <strong>outliers</strong>, and <strong>AWQ</strong> (protecting salient channels by activation magnitude) and <strong>GPTQ</strong> (a calibrated, per-layer minimization of reconstruction error) keep the accuracy loss nearly invisible by handling them cleverly.",
            },
            {
                "zh": "说清量化在 SGLang 里<strong>怎么“插”进线性层</strong>，并把它放进“一切皆可插拔”的脉络里。请说明：①每种格式提供一个 <span class='mono'>QuantizationConfig</span> → <span class='mono'>LinearMethod</span>（如 <span class='mono'>Fp8LinearMethod</span>），<strong>替换线性层“怎么存权重、怎么做这次矩阵乘”</strong>，于是模型文件（第 26 课）里的 <span class='mono'>RowParallelLinear</span> 一个字不改、由配置决定底下是 fp16 还是 fp8；②权重从<strong>预量化 checkpoint</strong> 读入或在加载时<strong>即时量化</strong>（第 25 课）；③<strong>FP8</strong> 的 E4M3 与<strong>动态/静态</strong>激活定标、<span class='mono'>cutlass</span>/<span class='mono'>Marlin</span> 快路与 H100/B200 原生 FP8；④<strong>KV 缓存量化</strong>是相关但独立的旋钮，并点明量化是继注意力后端（第 33 课）、KV 池之后“可插拔”的第三个例子。",
                "en": "Explain <strong>how quantization 'plugs into' linear layers</strong> in SGLang, and place it in the 'everything pluggable' arc. Cover: (1) each format provides a <span class='mono'>QuantizationConfig</span> → <span class='mono'>LinearMethod</span> (e.g. <span class='mono'>Fp8LinearMethod</span>) that <strong>replaces how a linear layer stores its weight and does its matmul</strong>, so the <span class='mono'>RowParallelLinear</span> in the model file (Lesson 26) is unchanged and config decides fp16 vs fp8 underneath; (2) weights come from a <strong>pre-quantized checkpoint</strong> or are <strong>quantized on the fly</strong> at load (Lesson 25); (3) <strong>FP8</strong>'s E4M3 and <strong>dynamic/static</strong> activation scaling, the <span class='mono'>cutlass</span>/<span class='mono'>Marlin</span> fast paths and native FP8 on H100/B200; (4) <strong>KV-cache quantization</strong> as a related-but-separate knob, and naming quantization the third 'pluggable' example after the attention backend (Lesson 33) and the KV pool.",
            },
        ],
    },
    "36-rope-norm-and-ops.html": {
        "mcq": [
            {
                "q": {
                    "zh": "注意力本身对 token 的顺序<strong>不敏感</strong>，RoPE 是怎么把<strong>位置</strong>注入进去的？",
                    "en": "Attention itself is <strong>insensitive</strong> to token order — how does RoPE inject <strong>position</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>按位置把每个 q/k 向量旋转一个角度（角度 ∝ 位置）</strong>，旋转后两个向量的点积只依赖它们位置<strong>之差</strong>，于是注意力变成<strong>相对位置感知</strong>",
                        "en": "<strong>It rotates each q/k vector by an angle ∝ its position</strong>; after rotation the dot product of two vectors depends only on the <strong>difference</strong> of their positions, so attention becomes <strong>relative-position aware</strong>",
                    },
                    {"zh": "给每个词学习一个绝对位置向量，加到词向量上", "en": "It learns an absolute position vector per word and adds it to the token embedding"},
                    {"zh": "在 softmax 后乘一个和位置成正比的标量", "en": "It multiplies a scalar proportional to position after the softmax"},
                    {"zh": "把 KV 缓存按位置重新排序", "en": "It reorders the KV cache by position"},
                ],
                "answer": 0,
                "why": {
                    "zh": "RoPE 不“加”位置向量，而是按位置<strong>旋转</strong> q/k。点积的数学决定了：两个旋转过的向量做点积，结果只取决于两个旋转角的<strong>差</strong>，也就是相对位置——于是注意力天然感知 token 间的距离，且不引入任何可学习参数。它不是绝对位置嵌入、不是 softmax 后的标量、也和重排 KV 无关。",
                    "en": "RoPE doesn't add a position vector — it <strong>rotates</strong> q/k by position. The dot-product math means two rotated vectors' product depends only on the <strong>difference</strong> of their rotation angles, i.e. relative position — so attention naturally feels inter-token distance, with no learnable parameters. It's not absolute position embedding, not a post-softmax scalar, and unrelated to reordering KV.",
                },
            },
            {
                "q": {
                    "zh": "一个在 4k 上训练的模型，想直接服务 32k 的长文本，RoPE 系的方法（线性缩放 / NTK / YaRN）靠什么做到？",
                    "en": "To serve 32k text with a 4k-trained model, how do RoPE methods (linear scaling / NTK / YaRN) pull it off?",
                },
                "opts": [
                    {
                        "zh": "<strong>拉伸旋转频率</strong>，把超出训练范围的位置角度压回模型熟悉的区间——无需重训，只换一个 RoPE 变体（由 <span class='mono'>get_rope</span> 按配置造出），模型代码不变",
                        "en": "<strong>Stretch the rotation frequencies</strong> so out-of-range position angles fall back into the familiar range — no retraining, just a different RoPE variant (built by <span class='mono'>get_rope</span> from config), model code unchanged",
                    },
                    {"zh": "把多出来的 token 直接丢弃", "en": "Drop the extra tokens beyond 4k"},
                    {"zh": "把模型权重重新量化到更低比特", "en": "Re-quantize the model weights to lower bits"},
                    {"zh": "在 32k 上从头重训一个新模型", "en": "Train a brand-new model from scratch at 32k"},
                ],
                "answer": 0,
                "why": {
                    "zh": "上下文扩展的关键，是<strong>巧妙地拉伸旋转频率</strong>：线性缩放等比压缩位置、NTK 不均匀调整高低频、YaRN 再加温度分段，让 4k 训练的角度范围覆盖 32k+。这些都是 RoPE 变体，通过同一个 <span class='mono'>get_rope(...)</span> 工厂按配置造出，<strong>不必重训、不改模型代码</strong>。丢 token 会丢信息，量化是另一回事，重训则违背了“无需重训”的初衷。",
                    "en": "Context extension hinges on <strong>cleverly stretching the rotation frequencies</strong>: linear scaling compresses positions proportionally, NTK adjusts high/low frequencies unevenly, YaRN adds temperature and segmenting — covering 32k+ with the 4k-trained angle range. All are RoPE variants built by the same <span class='mono'>get_rope(...)</span> factory from config, with <strong>no retraining and no model-code change</strong>. Dropping tokens loses info, quantization is unrelated, and retraining defeats the whole point.",
                },
            },
            {
                "q": {
                    "zh": "相比经典 LayerNorm，Llama 系常用的 RMSNorm 关键差别是什么？",
                    "en": "Versus classic LayerNorm, what's the key difference of the RMSNorm used by the Llama family?",
                },
                "opts": [
                    {
                        "zh": "<strong>RMSNorm 不减均值、不加偏置</strong>，只把向量 ÷ 均方根再乘一个 scale——比 LayerNorm（减均值 ÷ 标准差 + scale + shift）<strong>更便宜</strong>，LLM 上效果一样好，还常和残差加法融合",
                        "en": "<strong>RMSNorm does no mean subtraction and has no bias</strong> — it just divides the vector by its root-mean-square and multiplies a scale; <strong>cheaper</strong> than LayerNorm (subtract mean, ÷ std, + scale + shift), as good for LLMs, and often fused with the residual add",
                    },
                    {"zh": "RMSNorm 精度更高但慢得多", "en": "RMSNorm is more accurate but far slower"},
                    {"zh": "RMSNorm 需要额外的偏置参数", "en": "RMSNorm needs an extra bias parameter"},
                    {"zh": "两者完全一样，只是命名不同", "en": "They're identical, just named differently"},
                ],
                "answer": 0,
                "why": {
                    "zh": "LayerNorm 要减均值、除标准差，并带 scale+shift 两组参数；RMSNorm 砍掉了减均值和偏置，只做 ÷ 均方根再乘 scale——计算更少、参数更少、<strong>更便宜</strong>，而大量实践证明 LLM 上效果不输 LayerNorm，因此成了主流。它不是更慢，也不需要额外偏置，更不是和 LayerNorm 完全等价。",
                    "en": "LayerNorm subtracts the mean, divides by std, and carries scale+shift; RMSNorm drops mean subtraction and bias, doing only ÷ root-mean-square then scale — less compute, fewer params, <strong>cheaper</strong>, and practice shows it matches LayerNorm for LLMs, hence its dominance. It isn't slower, needs no extra bias, and is not equivalent to LayerNorm.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“<strong>指南针 + 夹角</strong>”的类比把 RoPE 讲透，并说清它为什么能<strong>扩展上下文</strong>。请覆盖：①注意力的点积对顺序<strong>不敏感</strong>，所以必须注入位置；②RoPE 不“加”位置向量，而是按位置把 q/k <strong>旋转</strong>一个角度（角度 ∝ 位置），旋转后点积只依赖位置<strong>之差</strong> ⇒ <strong>相对位置感知</strong>，且<strong>无可学习参数</strong>；③线性缩放 / NTK-aware / YaRN 如何<strong>拉伸旋转频率</strong>，让 4k 训练的模型服务 32k+，且都由 <span class='mono'>get_rope(...)</span> 按配置造变体、<strong>模型代码不变</strong>（前向应用见第 33 课注意力之前）。",
                "en": "Using the '<strong>compass + angle gap</strong>' analogy, fully explain RoPE and why it enables <strong>context extension</strong>. Cover: (1) the attention dot product is <strong>order-insensitive</strong>, so position must be injected; (2) RoPE doesn't add a position vector but <strong>rotates</strong> q/k by an angle ∝ position, so the dot product depends only on the <strong>difference</strong> of positions ⇒ <strong>relative-position aware</strong>, with <strong>no learnable parameters</strong>; (3) how linear scaling / NTK-aware / YaRN <strong>stretch the rotation frequencies</strong> to serve 32k+ from a 4k-trained model, all built by <span class='mono'>get_rope(...)</span> from config with <strong>model code unchanged</strong> (applied right before attention, Lesson 33).",
            },
            {
                "zh": "解释<strong>为什么要把小算子融合</strong>，并把 RMSNorm、SiluAndMul、fused add-norm 放进“可插拔的硬件优化层”这条脉络里。请说明：①GPU 上每次 kernel 启动 + 显存读写都有<strong>固定开销</strong>，一串小算子各跑各的会让数据在显存和计算单元间<strong>来回搬好几趟</strong>，而搬运常比计算还慢；②融合把相邻小算子并成<strong>一个 kernel</strong>（如 <span class='mono'>SiluAndMul</span> 合 SiLU(gate)×up、fused add-norm 合残差加法+RMSNorm、qk-norm+rope），省掉中间往返；③这些都是可复用、硬件优化的 <span class='mono'>nn.Module</span>，模型只管调用（第 26 课），层内部按平台挑融合/专属 kernel——继注意力后端（第 33 课）、量化（第 35 课）之后“一切皆可插拔”在算子维的体现，更底层 kernel/融合见第 38 课。",
                "en": "Explain <strong>why small ops get fused</strong>, and place RMSNorm, SiluAndMul, and fused add-norm in the 'pluggable hardware-optimized layers' arc. Cover: (1) on a GPU every kernel launch + HBM read/write carries <strong>fixed overhead</strong>, and a chain of separate small ops makes data <strong>shuttle back and forth</strong> between HBM and compute several times — often slower than the math; (2) fusion merges neighboring small ops into <strong>one kernel</strong> (e.g. <span class='mono'>SiluAndMul</span> for SiLU(gate)×up, fused add-norm for residual-add+RMSNorm, qk-norm+rope), skipping the intermediate round-trips; (3) all are reusable, hardware-optimized <span class='mono'>nn.Module</span>s the model merely calls (Lesson 26), and the layer picks a fused/platform kernel internally — 'everything pluggable' along the ops dimension after the attention backend (Lesson 33) and quantization (Lesson 35); lower-level kernels/fusion are Lesson 38.",
            },
        ],
    },
    "37-logits-and-vocab-parallel.html": {
        "mcq": [
            {
                "q": {
                    "zh": "模型主体算完后，<span class='mono'>lm_head</span> 这个输出头到底<strong>产出什么</strong>？",
                    "en": "After the model body finishes, what does the <span class='mono'>lm_head</span> output head actually <strong>produce</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>把每个位置的 hidden 投影成一条“词表大小”的分数向量 = logits</strong>——长度等于词表 token 数，每个分数表示该 token 有多“该”当下一个词，随后交采样器（第 28 课）选出 token",
                        "en": "<strong>It projects each position's hidden into a 'vocab-sized' score vector = logits</strong> — length equals the number of vocab tokens, each score saying how 'fitting' that token is as the next word, then handed to the Sampler (Lesson 28) to pick a token",
                    },
                    {"zh": "直接产出一个最终的 token 字符串，不再需要采样", "en": "It directly produces a final token string, no sampling needed"},
                    {"zh": "产出注意力权重矩阵，给下一层用", "en": "It produces an attention weight matrix for the next layer"},
                    {"zh": "产出 KV 缓存，供后续解码复用", "en": "It produces the KV cache for later decode reuse"},
                ],
                "answer": 0,
                "why": {
                    "zh": "lm_head 是个 <span class='mono'>[hidden × vocab]</span> 的大矩阵，把 hidden 向量乘上去，得到一条<strong>长度等于词表大小</strong>的 logits——每个词表 token 一个分数。它<strong>不</strong>直接出 token（那是采样器第 28 课的活），不是注意力权重，也不是 KV 缓存。logits 只是“原始打分”，还要经过采样才落成具体 token。",
                    "en": "lm_head is a big <span class='mono'>[hidden × vocab]</span> matrix; multiplying the hidden vector through it yields logits <strong>as long as the vocab</strong> — one score per vocab token. It does <strong>not</strong> emit a token directly (that's the Sampler's job, Lesson 28), nor attention weights, nor the KV cache. Logits are just raw scores that still need sampling to become a concrete token.",
                },
            },
            {
                "q": {
                    "zh": "为什么要把 lm_head 沿<strong>词表维度</strong>切分（词表并行），而不是放一张卡上？",
                    "en": "Why split lm_head along the <strong>vocab dimension</strong> (vocab parallelism) instead of keeping it on one GPU?",
                },
                "opts": [
                    {
                        "zh": "<strong>词表很大（3 万~25 万），lm_head/嵌入是十几亿参数的大矩阵，一张卡装不下也算不动</strong>；按词表维切到各 TP rank，每卡只持有、只算一段，<strong>显存与算力均摊</strong>，且与 q/k/v、MLP 的 TP 切分（第 25/46 课）一脉相承",
                        "en": "<strong>The vocab is huge (30k–250k), so lm_head/embedding are billion-param matrices too big for one GPU to hold or compute</strong>; splitting the vocab dim across TP ranks lets each GPU hold and compute one segment, <strong>spreading HBM and FLOPs</strong>, consistent with the q/k/v and MLP TP splits (Lessons 25/46)",
                    },
                    {"zh": "切分能提高单个 token 的预测精度", "en": "Splitting improves the prediction accuracy of a single token"},
                    {"zh": "因为词表必须按字母顺序排在不同卡上", "en": "Because the vocab must be alphabetically ordered across GPUs"},
                    {"zh": "为了让每张卡都能独立完整地输出 token，不需通信", "en": "So each GPU can independently emit full tokens with no communication"},
                ],
                "answer": 0,
                "why": {
                    "zh": "切词表纯粹是<strong>规模</strong>问题：vocab=15 万、hidden=8192 时，单这张表就十几亿参数，TP 下放不下一张卡、也不该让一卡独扛这步乘法。按词表维切给各 rank，每卡只算一段，显存算力均摊，和其它层的 TP 切分一致。它不提升精度（数学等价），与字母顺序无关，而且恰恰<strong>需要</strong> all-gather 通信才能拼出完整 logits。",
                    "en": "Splitting the vocab is purely a <strong>scale</strong> issue: at vocab=150k, hidden=8192 this one table is over a billion params, too big for one GPU under TP and not something one GPU should bear alone. Sharding the vocab dim across ranks lets each compute one segment, spreading HBM and FLOPs, in line with other layers' TP. It doesn't improve accuracy (it's mathematically equivalent), has nothing to do with alphabetical order, and in fact <strong>requires</strong> an all-gather to assemble the full logits.",
                },
            },
            {
                "q": {
                    "zh": "每个 rank 只算出“本段词表”的部分 logits，完整 logits 是<strong>怎么拼出来</strong>的？",
                    "en": "Each rank computes only its segment's partial logits — how is the <strong>full</strong> logits vector assembled?",
                },
                "opts": [
                    {
                        "zh": "<strong>用一次跨卡 all-gather，把各 rank 的残缺片段按顺序首尾拼接</strong>成长度等于整个词表的完整向量（单卡时跳过）；若只做贪心 argmax，还能就地汇总局部最大、免去物化完整向量",
                        "en": "<strong>A cross-GPU all-gather concatenates each rank's partial segment end-to-end</strong> into a full vector as long as the whole vocab (skipped on a single GPU); for plain greedy argmax it can instead reduce local maxima in place, avoiding materializing the full vector",
                    },
                    {"zh": "每张卡各自把自己那段补全成完整词表，不需通信", "en": "Each GPU pads its own segment into a full vocab, no communication"},
                    {"zh": "用一次 all-reduce 把各段相加成完整 logits", "en": "An all-reduce sums the segments into full logits"},
                    {"zh": "由 CPU 把各卡结果收集后再发回", "en": "The CPU collects all GPU results and sends them back"},
                ],
                "answer": 0,
                "why": {
                    "zh": "各 rank 的片段<strong>互不重叠</strong>（rank0 管词 0–37499，rank1 管 37500–…），所以要把它们<strong>拼接</strong>而非相加——这正是 <strong>all-gather</strong> 干的事：每卡发出本段、收下别人段，按 rank 顺序拼成完整向量；单卡时直接跳过。注意是 all-gather（拼接）不是 all-reduce（求和），更不靠 CPU 中转；贪心解码时还能就地求全局最大、免去拼出完整向量。",
                    "en": "The ranks' segments are <strong>non-overlapping</strong> (rank0 owns tokens 0–37499, rank1 37500–…), so they must be <strong>concatenated</strong>, not summed — exactly what <strong>all-gather</strong> does: each GPU sends its segment and receives the others', stitched in rank order into the full vector; skipped on one GPU. Note it's all-gather (concatenate), not all-reduce (sum), and not via a CPU detour; for greedy decode it can reduce the global max in place and skip building the full vector.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“<strong>超大词典 + N 个管理员</strong>”的类比把输出头与词表并行讲透，并算清<strong>末位切片</strong>省下的算力。请覆盖：①<span class='mono'>lm_head</span> 是 <span class='mono'>[hidden × vocab]</span> 大矩阵，把每位置 hidden 投影成<strong>词表大小的 logits</strong>，再交采样器（第 28 课）出 token；②词表很大（3 万~25 万）⇒ 这张表十几亿参数，TP 下放不下一卡，于是 <span class='mono'>VocabParallelEmbedding</span>/<span class='mono'>ParallelLMHead</span> 按<strong>词表维</strong>切给各 rank，每卡算一段、再 <strong>all-gather</strong> 拼成完整 logits（和 q/k/v、MLP 的切法同源，第 25/46 课）；③<strong>末位切片</strong>——一条 2000 token 的 prompt，预填充虽算了全部位置的 hidden，但<strong>只有最后一位</strong>的 logits 用来预测下一个词，于是把 lm_head 的输入从 2000 行压到 <strong>1 行</strong>，这步算力直接砍掉两千倍（除非要 logprob 才多留几行）。",
                "en": "Using the '<strong>giant dictionary + N librarians</strong>' analogy, fully explain the output head and vocab parallelism, and pin down the FLOPs saved by the <strong>last-token slice</strong>. Cover: (1) <span class='mono'>lm_head</span> is a big <span class='mono'>[hidden × vocab]</span> matrix that projects each position's hidden into <strong>vocab-sized logits</strong>, handed to the Sampler (Lesson 28) for a token; (2) the vocab is huge (30k–250k) ⇒ this table is a billion+ params, too big for one GPU under TP, so <span class='mono'>VocabParallelEmbedding</span>/<span class='mono'>ParallelLMHead</span> split the <strong>vocab dim</strong> across ranks, each computing a segment then <strong>all-gather</strong>-ing into full logits (same idea as the q/k/v and MLP splits, Lessons 25/46); (3) the <strong>last-token slice</strong> — for a 2000-token prompt, prefill computes hidden for all positions but only the <strong>last position's</strong> logits predict the next word, so shrinking lm_head's input from 2000 rows to <strong>1 row</strong> cuts this step's compute by 2000× (unless logprobs are requested, keeping a few more rows).",
            },
            {
                "zh": "说清 <span class='mono'>LogitsProcessor</span> 如何把输出头<strong>串成一条流水线</strong>，并把它放进“整条前向收束”的脉络里。请说明：①它的三步——<strong>末位切片</strong>（按 <span class='mono'>logits_metadata</span> 只留每条请求的末位 hidden）→ <strong>词表并行投影</strong>（<span class='mono'>ParallelLMHead</span> 算本 rank 那段）→ <strong>跨卡 all-gather</strong>（<span class='mono'>do_tensor_parallel_all_gather</span> 控制，单卡跳过）；②采样前“最后一公里”的后处理都作用在这条 logits 向量上——<strong>logprob</strong>（log-softmax 取对数概率）、<strong>结构化输出词表掩码</strong>（第 48 课，把非法 token 的 logit 压成负无穷）、<strong>logit bias</strong>，且都必须在<strong>采样之前</strong>施加才能从根上封死非法 token；③本课如何收束整条前向：第 24 课 ModelRunner.forward → 模型主体（第 26 课）→ LogitsProcessor → logits → Sampler（第 28 课）→ token，并把 Part 8（注意力第 33、MoE 第 34、量化第 35、RoPE/Norm 第 36、本课 logits/词表并行）连成“可插拔算子层”的全貌。",
                "en": "Explain how <span class='mono'>LogitsProcessor</span> <strong>chains the output head into a pipeline</strong>, and place it in the 'closing the whole forward' arc. Cover: (1) its three steps — <strong>last-token slice</strong> (keep only each request's last hidden per <span class='mono'>logits_metadata</span>) → <strong>vocab-parallel projection</strong> (<span class='mono'>ParallelLMHead</span> computes this rank's segment) → <strong>cross-GPU all-gather</strong> (gated by <span class='mono'>do_tensor_parallel_all_gather</span>, skipped on one GPU); (2) the 'last mile' post-processing before sampling that all acts on this logits vector — <strong>logprob</strong> (log-softmax for log-probabilities), the <strong>structured-output vocab mask</strong> (Lesson 48, driving illegal tokens' logits to negative infinity), and <strong>logit bias</strong>, all of which must be applied <strong>before sampling</strong> to seal off illegal tokens at the root; (3) how this lesson closes the whole forward: Lesson 24's ModelRunner.forward → model body (Lesson 26) → LogitsProcessor → logits → Sampler (Lesson 28) → token, tying Part 8 (attention L33, MoE L34, quant L35, RoPE/Norm L36, and this lesson's logits/vocab parallelism) into the full 'pluggable operator layer' picture.",
            },
        ],
    },
    "38-sgl-kernel-overview.html": {
        "mcq": [
            {
                "q": {
                    "zh": "为什么 SGLang 要把热路径核函数放进一个<strong>独立的 sgl-kernel 工程</strong>、用 <strong>AOT</strong> 预编译成 <span class='mono'>.so</span> 随 wheel 发布，而不是启动时现编？",
                    "en": "Why does SGLang put hot-path kernels in a <strong>separate sgl-kernel project</strong> and <strong>AOT</strong>-compile them into a <span class='mono'>.so</span> shipped in the wheel, instead of compiling at startup?",
                },
                "opts": [
                    {
                        "zh": "<strong>CUDA kernel 经 nvcc 编译/链接要几分钟，若每次启动现编延迟无法接受；把稳定通用的 kernel 提前编进 .so，装好即用、启动零编译开销</strong>，这正契合第 4 课解码带宽受限、kernel 质量直接决定吞吐",
                        "en": "<strong>A CUDA kernel takes minutes to compile/link via nvcc; compiling on every startup is unacceptable. Pre-compiling stable, general kernels into the .so means install-and-go with zero startup compile cost</strong>, fitting Lesson 4's bandwidth-bound decode where kernel quality decides throughput",
                    },
                    {"zh": "因为 Python 不能调用 C++ 代码，必须先编成独立可执行文件", "en": "Because Python cannot call C++ code, so it must be built into a standalone executable first"},
                    {"zh": "为了让每个用户在自己机器上重新训练模型权重", "en": "To let each user re-train the model weights on their own machine"},
                    {"zh": "因为 PyTorch 不支持自定义算子，只能走子进程", "en": "Because PyTorch does not support custom ops, so a subprocess is the only option"},
                ],
                "answer": 0,
                "why": {
                    "zh": "AOT 的核心动机是<strong>把编译挪到发布时</strong>：nvcc 编译耗时几分钟，提前编进 wheel 里的 <span class='mono'>.so</span> 就能启动零开销。Python 当然能通过 <span class='mono'>torch.ops</span> 调 C++/CUDA，PyTorch 也明确支持自定义算子,与训练权重、子进程都无关。",
                    "en": "AOT's core motivation is <strong>moving compilation to release time</strong>: nvcc takes minutes, so pre-compiling into the wheel's <span class='mono'>.so</span> gives zero startup overhead. Python certainly can call C++/CUDA via <span class='mono'>torch.ops</span>, and PyTorch explicitly supports custom ops; this has nothing to do with re-training weights or subprocesses.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>sgl-kernel/python/sgl_kernel/</span> 下的薄包装（如 <span class='mono'>attention.py</span> 的 <span class='mono'>merge_state_v2</span>）<strong>主要负责什么</strong>？",
                    "en": "What is the <strong>main job</strong> of the thin wrappers under <span class='mono'>sgl-kernel/python/sgl_kernel/</span> (e.g. <span class='mono'>merge_state_v2</span> in <span class='mono'>attention.py</span>)?",
                },
                "opts": [
                    {
                        "zh": "<strong>做形状/dtype 校验与规整、按需分配输出张量，然后把活儿转交给 <span class='mono'>torch.ops.sgl_kernel.&lt;name&gt;</span></strong>——包装薄如纸，真正的算法在编译好的 csrc kernel 里",
                        "en": "<strong>Validate/normalize shapes and dtypes, allocate output tensors as needed, then forward to <span class='mono'>torch.ops.sgl_kernel.&lt;name&gt;</span></strong> — the wrapper is paper-thin, the real algorithm is in the compiled csrc kernel",
                    },
                    {"zh": "在 Python 里用纯 Python 循环实现完整的注意力计算", "en": "Implement the full attention computation with pure-Python loops"},
                    {"zh": "在运行时用 nvcc 重新编译整个 .so", "en": "Re-compile the entire .so with nvcc at runtime"},
                    {"zh": "负责把模型权重从磁盘加载进显存", "en": "Load the model weights from disk into device memory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "薄包装只做三件小事:dtype 转换/校验、分配输出、<span class='mono'>torch.ops</span> 转交,几乎不含算法逻辑。它<strong>不</strong>在 Python 里算注意力(那在 kernel 里),不在运行时重编 AOT 的 .so,也不负责加载权重。",
                    "en": "A thin wrapper does just three small things: dtype cast/validate, allocate outputs, forward to <span class='mono'>torch.ops</span>, with almost no algorithmic logic. It does <strong>not</strong> compute attention in Python (that's in the kernel), does not re-compile the AOT .so at runtime, and does not load weights.",
                },
            },
            {
                "q": {
                    "zh": "Python 代码最终是<strong>通过什么机制</strong>跳进编译进 <span class='mono'>.so</span> 的 C++/CUDA kernel 的？",
                    "en": "Through <strong>what mechanism</strong> does Python code ultimately jump into the C++/CUDA kernel compiled into the <span class='mono'>.so</span>?",
                },
                "opts": [
                    {
                        "zh": "<strong>注册胶水 <span class='mono'>common_extension.cc</span> 用 <span class='mono'>TORCH_LIBRARY_FRAGMENT</span> 把每个 kernel 注册成 PyTorch 自定义算子，挂在 <span class='mono'>torch.ops.sgl_kernel.*</span> 命名空间下，Python 由此直接调入原生代码</strong>",
                        "en": "<strong>The registration glue <span class='mono'>common_extension.cc</span> uses <span class='mono'>TORCH_LIBRARY_FRAGMENT</span> to register each kernel as a PyTorch custom op under the <span class='mono'>torch.ops.sgl_kernel.*</span> namespace, through which Python calls straight into native code</strong>",
                    },
                    {"zh": "通过 HTTP 请求把张量发到一个本地 GPU 服务进程", "en": "By sending tensors over HTTP to a local GPU service process"},
                    {"zh": "通过把 Python 代码翻译成 CUDA 源码再编译", "en": "By transpiling the Python code into CUDA source and compiling it"},
                    {"zh": "通过 ctypes 直接 dlopen 一个匿名 .so，绕过 PyTorch", "en": "By dlopen-ing an anonymous .so directly via ctypes, bypassing PyTorch"},
                ],
                "answer": 0,
                "why": {
                    "zh": "kernel 经 <span class='mono'>common_extension.cc</span>(及 rocm/musa 变体)用 <span class='mono'>TORCH_LIBRARY_FRAGMENT</span> 注册成 <span class='mono'>torch.ops.sgl_kernel.*</span>,这是 Python 进入原生代码的标准门。它不走 HTTP,不在运行时翻译 Python,也不绕开 PyTorch 裸 dlopen。",
                    "en": "Kernels are registered via <span class='mono'>common_extension.cc</span> (and rocm/musa variants) using <span class='mono'>TORCH_LIBRARY_FRAGMENT</span> as <span class='mono'>torch.ops.sgl_kernel.*</span>, the standard door from Python into native code. It is not HTTP, not runtime transpilation of Python, and not a bare dlopen bypassing PyTorch.",
                },
            },
        ],
        "open": [
            {
                "zh": "用 <span class='mono'>merge_state_v2</span> 为例，描述 sgl-kernel 的<strong>通用调用模式</strong>：薄 Python 包装(dtype 转换 + 输出分配) → <span class='mono'>torch.ops.sgl_kernel.&lt;name&gt;</span> → 编译进 .so 的 csrc kernel。并说明这个算子在做什么(合并两段 split-KV 的部分注意力状态 v/s)，以及为什么把「校验/分配」留在 Python、把「计算」留在 kernel 是合理的分层。",
                "en": "Using <span class='mono'>merge_state_v2</span> as the example, describe sgl-kernel's <strong>general calling pattern</strong>: thin Python wrapper (dtype cast + output allocation) → <span class='mono'>torch.ops.sgl_kernel.&lt;name&gt;</span> → the csrc kernel compiled into the .so. Explain what the op does (merging two split-KV partial attention states v/s), and why keeping 'validate/allocate' in Python and 'compute' in the kernel is a sound layering.",
            },
            {
                "zh": "对比 <strong>AOT</strong> 与 <strong>JIT</strong>(第 39 课前瞻)两条编译路线:各自适合什么样的 kernel、各自的启动/首次开销与改动代价(AOT 改动要重打整个 wheel,JIT 可即时生效)如何?并结合 <span class='mono'>csrc/</span> 的目录划分(attention/gemm/moe/quantization 等)与第 33–36 课的上层,说明为什么 sgl-kernel 是整个推理引擎的性能地基。",
                "en": "Contrast the <strong>AOT</strong> and <strong>JIT</strong> (Lesson 39 preview) compile paths: what kind of kernel each suits, and their startup/first-time costs and change costs (AOT requires re-packing the whole wheel, JIT can take effect instantly). Then, drawing on <span class='mono'>csrc/</span>'s directory split (attention/gemm/moe/quantization, etc.) and the Lesson 33–36 upper layers, explain why sgl-kernel is the performance foundation of the whole inference engine.",
            },
        ],
    },
    "39-jit-kernel.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 <span class='mono'>jit_kernel</span> 这条 <strong>JIT（运行时即时编译）</strong>路线里，<span class='mono'>load_jit(...)</span> 第一次和后续调用的开销有何不同？",
                    "en": "On the <span class='mono'>jit_kernel</span> <strong>JIT (runtime just-in-time)</strong> path, how does the cost of the first <span class='mono'>load_jit(...)</span> call differ from later calls?",
                },
                "opts": [
                    {
                        "zh": "<strong>首次调用付一次性 nvcc 编译成本，随后把编好的 .so 按唯一标记缓存；后续调用命中缓存直接复用、不再重编，与 AOT 同速</strong>",
                        "en": "<strong>The first call pays a one-time nvcc compile cost, then caches the built .so by a unique marker; later calls hit the cache and reuse it with no recompile, as fast as AOT</strong>",
                    },
                    {"zh": "每次调用都会重新编译一遍源码，所以永远比 AOT 慢", "en": "Every call recompiles the source, so it is always slower than AOT"},
                    {"zh": "首次调用不编译，要等到第二次调用才触发 nvcc", "en": "The first call does not compile; nvcc is only triggered on the second call"},
                    {"zh": "它完全不编译，直接解释执行 CUDA 源码", "en": "It never compiles and just interprets the CUDA source directly"},
                ],
                "answer": 0,
                "why": {
                    "zh": "JIT 的精髓是<strong>编译一次、之后复用</strong>：首次现场跑 nvcc 付一次成本，之后命中按唯一标记缓存的 <span class='mono'>.so</span>，和 AOT 一样快。它绝不是每次重编，也不是解释执行。",
                    "en": "The essence of JIT is <strong>compile once, reuse later</strong>: the first call runs nvcc on the spot for a one-time cost, then later calls hit the <span class='mono'>.so</span> cached by a unique marker, as fast as AOT. It does not recompile each time, nor interpret.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>load_jit</span> 中的一个 <strong>wrapper</strong> 是 <span class='mono'>(export_name, kernel_name)</span> 二元组，它的作用是什么？",
                    "en": "A <strong>wrapper</strong> in <span class='mono'>load_jit</span> is an <span class='mono'>(export_name, kernel_name)</span> tuple. What is its role?",
                },
                "opts": [
                    {
                        "zh": "<strong>把一个面向 Python 的可调用名字映射到底层某个 C++/CUDA 核类</strong>，编译后即可从 Python 调用该核",
                        "en": "<strong>It maps a Python-facing callable name to some underlying C++/CUDA kernel class</strong>, so after compilation the kernel can be called from Python",
                    },
                    {"zh": "它指定该核要在哪块 GPU 上运行", "en": "It specifies which GPU the kernel should run on"},
                    {"zh": "它是缓存 .so 用的文件系统路径", "en": "It is the filesystem path used to cache the .so"},
                    {"zh": "它声明该核所需的最小显存大小", "en": "It declares the minimum VRAM the kernel needs"},
                ],
                "answer": 0,
                "why": {
                    "zh": "wrapper 做的是<strong>名字到核类的映射</strong>：<span class='mono'>export_name</span> 是 Python 侧可调用名，<span class='mono'>kernel_name</span> 是 C++/CUDA 核类。它和运行设备、缓存路径、显存声明都无关。",
                    "en": "A wrapper does <strong>name-to-kernel-class mapping</strong>: <span class='mono'>export_name</span> is the Python-side callable name, <span class='mono'>kernel_name</span> is the C++/CUDA kernel class. It has nothing to do with device, cache path, or VRAM.",
                },
            },
            {
                "q": {
                    "zh": "在为新算子选择 <strong>AOT（sgl-kernel，第38课）</strong>还是 <strong>JIT（jit_kernel）</strong>时，下列哪种判断<strong>最合理</strong>？",
                    "en": "When choosing <strong>AOT (sgl-kernel, Lesson 38)</strong> vs <strong>JIT (jit_kernel)</strong> for a new op, which judgment is <strong>most sound</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>稳定、高频、要求开箱即用、且部署环境可能没有编译器 → AOT；实验性、需按 GPU 架构定制 codegen、想避免 wheel 膨胀 → JIT</strong>",
                        "en": "<strong>Stable, hot, must be ready out of the box, and deploy env may lack a compiler → AOT; experimental, needs architecture-specific codegen, wants to avoid wheel bloat → JIT</strong>",
                    },
                    {"zh": "永远选 JIT，因为它没有任何代价", "en": "Always choose JIT because it has no cost"},
                    {"zh": "永远选 AOT，因为 JIT 无法达到与编译核相同的性能", "en": "Always choose AOT because JIT can never match compiled-kernel performance"},
                    {"zh": "随便选，二者在分发与首调延迟上完全等价", "en": "Pick either; the two are fully equivalent in shipping and first-call latency"},
                ],
                "answer": 0,
                "why": {
                    "zh": "选择本质是权衡：AOT 开箱即用、零首调延迟但 wheel 大且需提前枚举架构；JIT 灵活、wheel 瘦但有首调编译延迟且需工具链。命中缓存后 JIT 与 AOT 同速，所以并非“JIT 永远慢”，也不是“二者等价”。",
                    "en": "The choice is a trade-off: AOT is ready out of the box with zero first-call latency but a large wheel and pre-enumerated architectures; JIT is flexible with a slim wheel but has first-call compile latency and needs a toolchain. After a cache hit JIT equals AOT, so it is neither 'always slower' nor 'equivalent'.",
                },
            },
        ],
        "open": [
            {
                "zh": "用一句话解释 <span class='mono'>load_jit</span> 的“<strong>按需编译 + 缓存 .so</strong>”机制，并说明唯一标记（marker）在缓存复用中的作用，以及为什么这种设计能让 JIT 在保留灵活性的同时仍保持高性能。",
                "en": "In one sentence explain <span class='mono'>load_jit</span>'s '<strong>compile on demand + cache the .so</strong>' mechanism, the role of the unique marker in cache reuse, and why this design lets JIT stay high-performance while keeping flexibility.",
            },
            {
                "zh": "对比 <strong>AOT</strong> 与 <strong>JIT</strong> 两条编译路线“殊途同归”的地方与分歧点：两者最终都成为指向已编译核的 <span class='mono'>torch.ops</span> 风格调用，差别只在“何时/如何构建与分发”。结合 <span class='mono'>activation.py</span> 按 dtype 建模块、<span class='mono'>norm.py</span> 的 JIT <span class='mono'>fused_add_rmsnorm</span>（第36课所用），说明各自适合什么样的 kernel。",
                "en": "Contrast where <strong>AOT</strong> and <strong>JIT</strong> 'reach the same destination' yet diverge: both become <span class='mono'>torch.ops</span>-style callables into compiled kernels, differing only in 'when/how they're built and shipped'. Using <span class='mono'>activation.py</span> building per-dtype modules and <span class='mono'>norm.py</span>'s JIT <span class='mono'>fused_add_rmsnorm</span> (used by Lesson 36), explain what kind of kernel each suits.",
            },
        ],
    },
    "40-attention-kernel-dissection.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 <strong>decode（解码）</strong>注意力核函数里，紧接着拿到一行 Q 之后，核函数真正做的<strong>第一步</strong>是什么？",
                    "en": "In a <strong>decode</strong> attention kernel, right after receiving one row of Q, what is the kernel's true <strong>first step</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>用 <span class='mono'>page_table</span> 从分页 KV 缓存里收集（gather）该序列散落在非连续页上的 K、V 块</strong>，之后才做 Q·Kᵀ → softmax → ·V",
                        "en": "<strong>Use the <span class='mono'>page_table</span> to gather this sequence's K/V blocks scattered across non-contiguous pages from the paged KV cache</strong>, and only then do Q·Kᵀ → softmax → ·V",
                    },
                    {"zh": "立刻把整条序列的完整注意力分数矩阵物化到显存", "en": "Immediately materialise the full attention score matrix for the whole sequence in memory"},
                    {"zh": "先对 V 做一次全量归一化，再去读 K", "en": "First fully normalise V, then read K"},
                    {"zh": "把 KV 从分页布局复制成一整块连续显存再开始算", "en": "Copy KV from the paged layout into one contiguous block before computing"},
                ],
                "answer": 0,
                "why": {
                    "zh": "decode 的真正第一步是<strong>收集而非相乘</strong>：KV 被分页缓存切成非连续页，核函数靠 <span class='mono'>page_table</span> 把逻辑块翻译成物理页地址并 gather 回来，之后才进入点积与归一。它不会物化整张分数矩阵，也不会先复制成连续显存。",
                    "en": "The true first step in decode is <strong>gather, not multiply</strong>: the paged cache stores KV in non-contiguous pages, so the kernel uses the <span class='mono'>page_table</span> to translate logical blocks to physical page addresses and gather them before any dot product. It does not materialise the full score matrix nor pre-copy into contiguous memory.",
                },
            },
            {
                "q": {
                    "zh": "<strong>在线 softmax（FlashAttention 风格）</strong>之所以能在<strong>分块 tiling</strong> 流式处理时仍算出正确的归一化，关键机制是什么？",
                    "en": "What is the key mechanism that lets <strong>online softmax (FlashAttention-style)</strong> still produce a correct normalisation while processing in <strong>tiles</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>维护运行最大值（running max）+ 运行分母（累计指数和），每来一个瓦片就按指数比例修正旧累计值，一遍流式扫描即得等价结果</strong>",
                        "en": "<strong>Keep a running max + a running denominator (accumulated exp sum), rescaling the old totals by an exponential factor for each new tile, getting an equivalent result in one streaming pass</strong>",
                    },
                    {"zh": "先把所有瓦片的分数全部存下来，最后统一做一次标准 softmax", "en": "Store all tiles' scores first, then do one standard softmax at the end"},
                    {"zh": "对每个瓦片各自独立做 softmax，再简单求平均", "en": "Do an independent softmax per tile, then just average them"},
                    {"zh": "跳过求最大值的步骤，直接对原始分数取指数", "en": "Skip computing the max and just exponentiate the raw scores"},
                ],
                "answer": 0,
                "why": {
                    "zh": "在线 softmax 的核心是<strong>两个运行累计量</strong>：运行最大值与运行分母。新瓦片到来时用其局部最大值缩放之前的累计结果再加入贡献，因此无需存全部分数、一遍扫完即与全局 softmax 等价。逐瓦片独立 softmax 求平均是错误的，也不能跳过求最大值。",
                    "en": "Online softmax hinges on <strong>two running quantities</strong>: a running max and a running denominator. When a new tile arrives, it rescales the previously accumulated result by the new local max and adds the contribution, so it needs no full score storage and one pass equals a global softmax. Per-tile independent softmax then averaging is wrong, and the max step cannot be skipped.",
                },
            },
            {
                "q": {
                    "zh": "关于 <strong>prefill（预填充）核函数</strong>与 <strong>decode 核函数</strong>的差异，下列哪种描述<strong>最准确</strong>？",
                    "en": "Which description of the difference between the <strong>prefill kernel</strong> and the <strong>decode kernel</strong> is <strong>most accurate</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>prefill 查询位置很多、是又宽又厚的类 GEMM 稠密计算（偏算力）；decode 每步只有一个查询位置、又瘦又长、以按 page_table 在 KV 上 gather 为主（受带宽限制）</strong>",
                        "en": "<strong>Prefill has many query positions and is a wide, thick GEMM-like dense compute (compute-leaning); decode has one query position per step, is skinny and long, dominated by page_table gather over KV (bandwidth-bound)</strong>",
                    },
                    {"zh": "两者完全相同，只是 decode 跑在更快的 GPU 上", "en": "They are identical; decode just runs on a faster GPU"},
                    {"zh": "prefill 受带宽限制，decode 受算力限制", "en": "Prefill is bandwidth-bound and decode is compute-bound"},
                    {"zh": "decode 每步要处理 prompt 的全部 token，所以比 prefill 更稠密", "en": "Decode processes all prompt tokens each step, so it is denser than prefill"},
                ],
                "answer": 0,
                "why": {
                    "zh": "两者形状与瓶颈相反：prefill 是对很多查询位置的类 GEMM 稠密大计算，偏算力；decode 每步仅一个查询位置，主体是 KV gather，受显存带宽限制（第4课）。所以 SGLang 为它们走不同核函数路径，绝非相同，也不是“prefill 受带宽限制”。",
                    "en": "Their shapes and bottlenecks are opposite: prefill is a GEMM-like dense compute over many query positions (compute-leaning); decode has one query position per step and is dominated by KV gather, bandwidth-bound (Lesson 4). Hence SGLang takes different kernel paths; they are not identical, and prefill is not the bandwidth-bound one.",
                },
            },
        ],
        "open": [
            {
                "zh": "用自己的话描述一次 decode 注意力核函数的完整主线：从拿到一行 Q 开始，如何用 <span class='mono'>page_table</span> 从分页 KV 缓存收集非连续的 K、V，如何用<strong>分块 tiling</strong> 把瓦片流进 SRAM，并用<strong>在线 softmax</strong> 一遍算完 Q·Kᵀ → softmax → ·V。说明为什么这套设计本质上是“围绕带宽做减法”。",
                "en": "In your own words, trace the full main line of one decode attention kernel: starting from one row of Q, how it uses the <span class='mono'>page_table</span> to gather non-contiguous K/V from the paged KV cache, how it streams tiles into SRAM via <strong>tiling</strong>, and how <strong>online softmax</strong> computes Q·Kᵀ → softmax → ·V in one pass. Explain why this design is essentially 'subtraction around bandwidth'.",
            },
            {
                "zh": "结合 MLA（DeepSeek 风格）的真实皱褶说明 <span class='mono'>cutlass_mla_decode</span> 的封装做了什么：KV 为何存成压缩潜变量（<span class='mono'>D_latent=512</span>）+ rope（<span class='mono'>D_rope=64</span>），为什么要把头数补齐到 <span class='mono'>MAX_HEADS=128</span> 再派发给编译核函数，以及它和注意力后端（第33课）、CUDA Graph（第41课）的关系。",
                "en": "Using MLA's (DeepSeek-style) real wrinkle, explain what the <span class='mono'>cutlass_mla_decode</span> wrapper does: why KV is stored as a compressed latent (<span class='mono'>D_latent=512</span>) + rope (<span class='mono'>D_rope=64</span>), why heads are padded to <span class='mono'>MAX_HEADS=128</span> before dispatching to the compiled kernel, and how it relates to attention backends (Lesson 33) and CUDA Graph (Lesson 41).",
            },
        ],
    },
    "41-operator-fusion-and-cuda-graph.html": {
        "mcq": [
            {
                "q": {
                    "zh": "把多个小算子<strong>融合（fusion）</strong>成一个内核，最直接节省的两样东西是什么？",
                    "en": "When several small ops are <strong>fused</strong> into one kernel, what two things are most directly saved?",
                },
                "opts": [
                    {
                        "zh": "<strong>中间临时张量的 HBM 往返（写出又读回）+ 多次内核启动开销</strong>",
                        "en": "<strong>The intermediate temporary's HBM round-trips (write-out then read-back) + multiple kernel launch overheads</strong>",
                    },
                    {"zh": "模型参数量与显存总占用", "en": "Model parameter count and total memory footprint"},
                    {"zh": "注意力的序列长度上限", "en": "The maximum sequence length of attention"},
                    {"zh": "量化所需的校准数据集大小", "en": "The size of the calibration dataset needed for quantization"},
                ],
                "answer": 0,
                "why": {
                    "zh": "融合把相邻算子合在一个内核里，silu 这类中间结果留在<strong>寄存器</strong>而非写回 <span class='mono'>HBM</span> 再读出，省掉访存往返；同时多次启动被压成一次，省掉启动开销。它不改变参数量、序列上限或量化数据。",
                    "en": "Fusion merges adjacent ops into one kernel; intermediates like silu stay in <strong>registers</strong> instead of being written to <span class='mono'>HBM</span> and read back, saving round-trips, while several launches collapse into one, saving launch overhead. It changes neither parameter count, sequence limits, nor quantization data.",
                },
            },
            {
                "q": {
                    "zh": "<strong>CUDA Graph</strong>（第27课，捕获/重放）对被捕获的内核序列有什么<strong>硬性要求</strong>？",
                    "en": "What <strong>hard requirement</strong> does a <strong>CUDA graph</strong> (Lesson 27, capture/replay) impose on the captured kernel sequence?",
                },
                "opts": [
                    {
                        "zh": "<strong>形状必须静态、且不能有依赖数据的控制流（host 侧分支）</strong>",
                        "en": "<strong>Shapes must be static and there must be no data-dependent control flow (host-side branches)</strong>",
                    },
                    {"zh": "所有算子都必须用 FP8 量化", "en": "Every op must be FP8-quantized"},
                    {"zh": "必须运行在多卡张量并行下才能捕获", "en": "It can only be captured under multi-GPU tensor parallelism"},
                    {"zh": "每次重放都要重新编译内核", "en": "Each replay must recompile the kernels"},
                ],
                "answer": 0,
                "why": {
                    "zh": "图捕获记录的是固定指针与启动参数，因此<strong>形状必须静态、无数据依赖分支</strong>，否则录下的图对不上。它与量化、并行度无关，且重放正是为了<strong>免去</strong>重新提交/编译。",
                    "en": "Capture records fixed pointers and launch parameters, so <strong>shapes must be static with no data-dependent branches</strong>, else the recorded graph no longer matches. It is unrelated to quantization or parallelism, and replay exists precisely to <strong>avoid</strong> re-submission/recompilation.",
                },
            },
            {
                "q": {
                    "zh": "遇到<strong>动态形状或数据依赖分支</strong>的算子，SGLang 如何既享受图的低开销又不被它们拖累？",
                    "en": "For ops with <strong>dynamic shapes or data-dependent branches</strong>, how does SGLang still enjoy the graph's low overhead without being held back?",
                },
                "opts": [
                    {
                        "zh": "<strong>用分段/可打断图：捕获静态子段，动态部分夹在小图之间以 eager 即时执行</strong>",
                        "en": "<strong>Use piecewise/breakable graphs: capture the static sub-spans and run the dynamic bits eagerly between the small graphs</strong>",
                    },
                    {"zh": "强行把动态算子也塞进同一张图，靠运气重放", "en": "Force the dynamic ops into the same graph and hope the replay works"},
                    {"zh": "彻底放弃 CUDA Graph，整条前向都用 eager", "en": "Abandon CUDA graphs entirely and run the whole forward eagerly"},
                    {"zh": "把动态形状一律 padding 到最大值，从不分段", "en": "Always pad dynamic shapes to the maximum and never split"},
                ],
                "answer": 0,
                "why": {
                    "zh": "分段/可打断图把前向拆成若干静态子段分别捕获，动态部分（动态形状、host 分支）在小图之间以 eager 执行，从而既保住大多数内核的零提交开销，又不破坏已捕获的静态段。把动态算子硬塞进一张图会破坏捕获，整条 eager 又丢掉收益。",
                    "en": "Piecewise/breakable graphs split the forward into static sub-spans captured separately, running the dynamic parts (dynamic shapes, host branches) eagerly between the small graphs, keeping zero-submission cost on most kernels without breaking the captured spans. Forcing dynamic ops into one graph breaks capture, and going fully eager forfeits the benefit.",
                },
            },
        ],
        "open": [
            {
                "zh": "以 <span class='mono'>SiluAndMul</span> 为例，说明<strong>未融合</strong>的 <span class='mono'>forward_native</span>（先 silu 再乘）相比<strong>融合</strong>的 <span class='mono'>forward_cuda</span>（一个 kernel 同时做）多付出了哪些代价：从 <span class='mono'>HBM</span> 往返与<strong>启动开销</strong>两个角度展开，并说明这对带宽受限的解码阶段（第4课）为何重要。",
                "en": "Using <span class='mono'>SiluAndMul</span>, explain what extra cost the <strong>unfused</strong> <span class='mono'>forward_native</span> (silu then multiply) pays versus the <strong>fused</strong> <span class='mono'>forward_cuda</span> (one kernel does both): develop it from both <span class='mono'>HBM</span> round-trips and <strong>launch overhead</strong>, and explain why this matters for the bandwidth-bound decode stage (Lesson 4).",
            },
            {
                "zh": "解释融合与 CUDA Graph 如何<strong>配合</strong>：为什么“少 + 融 + 静态形状”的内核序列正好满足图的捕获前提，从而能<strong>捕获一次、反复重放</strong>把 CPU 提交开销降到近零；再说明被融合/捕获的内核本体来自第38课（AOT）与第39课（JIT）这一事实意味着什么。",
                "en": "Explain how fusion and CUDA graphs <strong>cooperate</strong>: why a 'fewer + fused + static-shape' kernel sequence exactly meets the graph's capture premise so it can be <strong>captured once and replayed</strong> to drive CPU submission cost near zero; then explain what it means that the fused/captured kernels themselves come from Lesson 38 (AOT) and Lesson 39 (JIT).",
            },
        ],
    },
    "42-multi-hardware-backends.html": {
        "mcq": [
            {
                "q": {
                    "zh": "<strong>平台抽象</strong>（基类 <span class='mono'>SRTPlatform</span>）的作用是什么？",
                    "en": "What is the role of the <strong>platform abstraction</strong> (the base class <span class='mono'>SRTPlatform</span>)?",
                },
                "opts": [
                    {
                        "zh": "<strong>让一套引擎跑遍多种芯片：每种芯片子类化它并提供工厂方法、能力标志与生命周期钩子</strong>",
                        "en": "<strong>Let one engine run on many chips: each chip subclasses it and provides factory methods, capability flags, and lifecycle hooks</strong>",
                    },
                    {"zh": "把模型权重压缩成 FP8 以节省显存", "en": "Compress model weights to FP8 to save memory"},
                    {"zh": "决定哪个请求先被调度执行", "en": "Decide which request gets scheduled first"},
                    {"zh": "为每块芯片重写整个调度器与模型代码", "en": "Rewrite the entire scheduler and model code for each chip"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>SRTPlatform</span>（在 <span class='mono'>srt/platforms/interface.py</span>）声明每芯片必须回答的工厂方法（如 <span class='mono'>get_default_attention_backend()</span>）、能力标志与钩子（如 <span class='mono'>apply_server_args_defaults()</span>）；各芯片子类化它（<span class='mono'>CudaSRTPlatform</span> 等）。它既不做量化也不做调度，更不是为每块芯片重写上层。",
                    "en": "<span class='mono'>SRTPlatform</span> (in <span class='mono'>srt/platforms/interface.py</span>) declares the per-chip factory methods (e.g. <span class='mono'>get_default_attention_backend()</span>), capability flags, and hooks (e.g. <span class='mono'>apply_server_args_defaults()</span>) every chip must answer; each chip subclasses it (<span class='mono'>CudaSRTPlatform</span>, etc.). It neither quantizes nor schedules, and certainly does not rewrite the upper layers per chip.",
                },
            },
            {
                "q": {
                    "zh": "在 SGLang 里，下面哪一项是<strong>每芯片专属、必须替换</strong>的，而不是可移植的？",
                    "en": "In SGLang, which of the following is <strong>per-chip and must be swapped</strong>, rather than portable?",
                },
                "opts": [
                    {
                        "zh": "<strong>内核（第38/39课）、注意力后端（第33课）与平台钩子</strong>",
                        "en": "<strong>The kernels (Lessons 38/39), the attention backend (Lesson 33), and the platform hooks</strong>",
                    },
                    {"zh": "调度器（第18课）", "en": "The scheduler (Lesson 18)"},
                    {"zh": "模型结构（第26课）", "en": "The model structure (Lesson 26)"},
                    {"zh": "输入输出（IO）流程", "en": "The input/output (IO) flow"},
                ],
                "answer": 0,
                "why": {
                    "zh": "上层是<strong>硬件无关</strong>的：调度器、模型、IO 流程换芯片时原样复用。真正按芯片替换的只有内核（AOT/JIT）、注意力后端和平台钩子，移植新硬件主要就是补齐这一小撮。",
                    "en": "The upper layers are <strong>hardware-agnostic</strong>: the scheduler, model, and IO flow are reused as-is when chips change. Only the kernels (AOT/JIT), the attention backend, and the platform hooks are truly swapped per chip — porting new hardware is mainly filling in that small handful.",
                },
            },
            {
                "q": {
                    "zh": "<strong>能力标志</strong>（如 <span class='mono'>support_cuda_graph</span>、<span class='mono'>supports_fp8</span>）让上层能做什么？",
                    "en": "What do <strong>capability flags</strong> (e.g. <span class='mono'>support_cuda_graph</span>, <span class='mono'>supports_fp8</span>) let the upper layers do?",
                },
                "opts": [
                    {
                        "zh": "<strong>询问“这块芯片能否做 X”，并据此自适应（如不支持 CUDA Graph 就退回逐算子执行）</strong>",
                        "en": "<strong>Ask 'can this chip do X' and adapt accordingly (e.g. fall back to per-operator execution when CUDA Graph isn't supported)</strong>",
                    },
                    {"zh": "强制所有芯片都必须支持同一套特性", "en": "Force every chip to support the exact same feature set"},
                    {"zh": "把上层代码按芯片拆成多份分别维护", "en": "Split the upper-layer code into per-chip copies to maintain separately"},
                    {"zh": "在运行时重新编译整个模型", "en": "Recompile the entire model at runtime"},
                ],
                "answer": 0,
                "why": {
                    "zh": "能力标志是布尔判断，让上层统一地问“能不能做 X”，得到否定回答就走兜底路径（如退回 eager），而不是为每块芯片写一堆 if-else 或直接崩溃。它不强制统一特性，也不需要拆分或重编上层。",
                    "en": "Capability flags are boolean judgments letting upper layers uniformly ask 'can you do X' and take a fallback path (e.g. revert to eager) on a 'no', instead of writing piles of if-else per chip or crashing outright. They neither force a uniform feature set nor require splitting or recompiling the upper layers.",
                },
            },
        ],
        "open": [
            {
                "zh": "用“旅行充电器 + 转换头”的类比，解释 <span class='mono'>SRTPlatform</span> 这层平台抽象如何让上层（调度器第18课、模型第26课）保持<strong>硬件无关</strong>：说明工厂方法（如 <span class='mono'>get_default_attention_backend()</span>、<span class='mono'>get_graph_runner_cls()</span>）与生命周期钩子（<span class='mono'>apply_server_args_defaults()</span>）各自把“具体芯片的能力”翻译成什么统一接口，以及为什么这能把“移植新硬件”从大事变成小事。",
                "en": "Using the 'travel charger + adapter head' analogy, explain how the <span class='mono'>SRTPlatform</span> platform-abstraction layer keeps the upper layers (scheduler Lesson 18, model Lesson 26) <strong>hardware-agnostic</strong>: spell out what uniform interface the factory methods (e.g. <span class='mono'>get_default_attention_backend()</span>, <span class='mono'>get_graph_runner_cls()</span>) and the lifecycle hook (<span class='mono'>apply_server_args_defaults()</span>) each translate 'a specific chip's capability' into, and why this turns 'porting new hardware' from a big job into a small one.",
            },
            {
                "zh": "结合 <span class='mono'>srt/platforms/</span> 与 <span class='mono'>srt/hardware_backend/</span> 两棵树，说明 SGLang 支持的硬件谱系（NVIDIA、AMD、昇腾 NPU、Intel XPU、摩尔线程 MUSA、苹果 MLX、CPU）是如何组织的；并解释“把每芯片专属的那一小撮（内核第38/39课、注意力后端第33课、平台钩子）隔离好，上层就能复用”为何是“从单卡到跨硬件大型集群”的工程基础（呼应第62课“一切皆可插拔”主题）。",
                "en": "Using the two trees <span class='mono'>srt/platforms/</span> and <span class='mono'>srt/hardware_backend/</span>, describe how SGLang's hardware lineup (NVIDIA, AMD, Ascend NPU, Intel XPU, Moore Threads MUSA, Apple MLX, CPU) is organized; then explain why 'isolating the small handful that is per-chip (kernels Lessons 38/39, attention backend Lesson 33, platform hooks) so the upper layers can be reused' is the engineering basis for 'single GPU to cross-hardware large clusters' (echoing Lesson 62's 'everything is pluggable' theme).",
            },
        ],
    },
    "43-speculative-decoding-overview.html": {
        "mcq": [
            {
                "q": {
                    "zh": "投机解码的核心分工是怎样的？",
                    "en": "What is the core division of labor in speculative decoding?",
                },
                "opts": [
                    {
                        "zh": "<strong>便宜的 draft 模型先提议 k 个 token，昂贵的 target 模型一次前向并行验完这 k 个</strong>",
                        "en": "<strong>A cheap draft model proposes k tokens, and the expensive target model verifies all k in one parallel forward</strong>",
                    },
                    {"zh": "target 模型逐个 token 提议，draft 模型逐个验证", "en": "The target proposes tokens one by one and the draft verifies them one by one"},
                    {"zh": "两个同样大的模型轮流生成 token", "en": "Two equally large models take turns generating tokens"},
                    {"zh": "draft 模型既提议又最终决定输出，target 只做缓存", "en": "The draft both proposes and decides the final output while the target only caches"},
                ],
                "answer": 0,
                "why": {
                    "zh": "投机解码让便宜的 <span class='mono'>draft</span> 自回归提议 k 个候选，再由昂贵的 <span class='mono'>target</span> 在<strong>一次前向</strong>里并行给所有 k 个位置打分验证；这样一次昂贵前向就能定下多个 token。其余选项颠倒了大小模型的职责。",
                    "en": "Speculative decoding has the cheap <span class='mono'>draft</span> autoregressively propose k candidates, then the expensive <span class='mono'>target</span> verify all k positions in parallel in <strong>one forward</strong>; one expensive forward thus pins down many tokens. The other options invert the roles of the small and big models.",
                },
            },
            {
                "q": {
                    "zh": "为什么说投机解码是<strong>无损</strong>的，且最坏情况也不会更慢？",
                    "en": "Why is speculative decoding <strong>lossless</strong>, and never slower even in the worst case?",
                },
                "opts": [
                    {
                        "zh": "<strong>接受/拒绝判据与 target 采样数学等价，输出分布相同；又因总会吐一个 bonus_token，每步至少产出 1 个 token</strong>",
                        "en": "<strong>The accept/reject criterion is mathematically equivalent to target sampling, so the distribution is identical; and since a bonus_token is always emitted, every step yields at least 1 token</strong>",
                    },
                    {"zh": "因为 draft 模型经过特殊微调，永远猜对", "en": "Because the draft model is specially fine-tuned to always guess correctly"},
                    {"zh": "因为它跳过了 target 的前向计算", "en": "Because it skips the target's forward computation"},
                    {"zh": "因为它把输出量化成 FP8 来加速", "en": "Because it quantizes the output to FP8 to speed up"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>accept</span>/拒绝采用与 target 采样等价的判据，故输出分布<strong>可证明无损</strong>；即便 <span class='mono'>accept_rate</span> 很低、所有草稿都被拒，target 仍在分歧点补出一个一定吐的 <span class='mono'>bonus_token</span>，于是退化成普通自回归而非更慢。它既不跳过 target 前向，也与量化无关。",
                    "en": "<span class='mono'>accept</span>/reject uses a criterion equivalent to target sampling, so the distribution is <strong>provably lossless</strong>; even if <span class='mono'>accept_rate</span> is low and every draft is rejected, the target still emits one always-produced <span class='mono'>bonus_token</span> at the divergence point, degrading to plain autoregression rather than being slower. It neither skips the target forward nor relates to quantization.",
                },
            },
            {
                "q": {
                    "zh": "关于 <span class='mono'>accept_rate</span>(α) 与 <span class='mono'>accept_length</span>(τ)，下面哪项正确？",
                    "en": "Regarding <span class='mono'>accept_rate</span>(α) and <span class='mono'>accept_length</span>(τ), which is correct?",
                },
                "opts": [
                    {
                        "zh": "<strong>α = correct_drafts / proposed_drafts，不含 bonus；τ = 每验证步平均吐出 token 数，含 bonus</strong>",
                        "en": "<strong>α = correct_drafts / proposed_drafts, excluding the bonus; τ = avg tokens emitted per verify step, including the bonus</strong>",
                    },
                    {"zh": "α 和 τ 是同一个量的两种写法", "en": "α and τ are two notations for the same quantity"},
                    {"zh": "α 含 bonus 而 τ 不含 bonus", "en": "α includes the bonus while τ excludes it"},
                    {"zh": "两者越小代表加速越大", "en": "Smaller values of both mean more speedup"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>accept_rate</span> 衡量单个草稿 token 被接受的概率，<strong>排除</strong> bonus；<span class='mono'>accept_length</span> 衡量每次验证平均产出几个 token，<strong>包含</strong>那个一定吐的 bonus。二者不是同一量，且都<strong>越大越快</strong>。",
                    "en": "<span class='mono'>accept_rate</span> measures the acceptance probability of a single draft token and <strong>excludes</strong> the bonus; <span class='mono'>accept_length</span> measures how many tokens each verify step yields on average and <strong>includes</strong> the always-emitted bonus. They are not the same quantity, and larger is faster for both.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话描述投机解码的一整轮循环：<span class='mono'>draft</span> 提议 k 个、<span class='mono'>target</span> 一次前向并行验证、<span class='mono'>accept</span> 最长正确前缀并补 <span class='mono'>bonus_token</span>、再向前滚动。并解释为什么这能攻击第4课所讲的<strong>带宽受限</strong>瓶颈：一次昂贵的“全权重搬运”如何被摊薄到多个 token 上。",
                "en": "In your own words, describe one full round of speculative decoding: the <span class='mono'>draft</span> proposes k, the <span class='mono'>target</span> verifies in one parallel forward, we <span class='mono'>accept</span> the longest correct prefix plus a <span class='mono'>bonus_token</span>, then roll forward. Then explain why this attacks the <strong>bandwidth-bound</strong> bottleneck from Lesson 4: how one expensive 'full weight haul' is amortized across many tokens.",
            },
            {
                "zh": "解释 <span class='mono'>accept_rate</span>(α) 与 <span class='mono'>accept_length</span>(τ) 的区别（谁含 bonus、谁不含），以及它们如何共同决定加速比；并说明 SGLang 为何用 <span class='mono'>SpeculativeAlgorithm</span> 枚举（EAGLE、EAGLE3、NGRAM、STANDALONE、DFLASH、FROZEN_KV_MTP、NONE）让“出草稿的方法”可插拔，而 <span class='mono'>BaseSpecWorker</span> 又如何用 <span class='mono'>target_worker</span> 与 <span class='mono'>draft_worker</span> 统一编排提议与验证（EAGLE 详见第44课，验证复用第28课采样器）。",
                "en": "Explain the difference between <span class='mono'>accept_rate</span>(α) and <span class='mono'>accept_length</span>(τ) (which includes the bonus, which excludes it) and how together they determine the speedup; then explain why SGLang uses the <span class='mono'>SpeculativeAlgorithm</span> enum (EAGLE, EAGLE3, NGRAM, STANDALONE, DFLASH, FROZEN_KV_MTP, NONE) to make 'how to draft' pluggable, and how <span class='mono'>BaseSpecWorker</span> orchestrates proposing and verifying via <span class='mono'>target_worker</span> and <span class='mono'>draft_worker</span> (EAGLE is Lesson 44, verification reuses the Lesson 28 sampler).",
            },
        ],
    },
    "44-eagle-and-next-gen.html": {
        "mcq": [
            {
                "q": {
                    "zh": "EAGLE 的“特征级起草”具体指什么？",
                    "en": "What exactly is EAGLE's 'feature-level drafting'?",
                },
                "opts": [
                    {
                        "zh": "<strong>复用 target 模型最后的隐藏状态 + 上一个采样 token，去预测下一个特征，草稿头很小且与 target 天然对齐</strong>",
                        "en": "<strong>It reuses the target model's last hidden state + the previously sampled token to predict the next feature, so the draft head is tiny and naturally aligned with the target</strong>",
                    },
                    {"zh": "训练一个和 target 一样大的独立草稿模型从头生成", "en": "Training a separate draft model as large as the target to generate from scratch"},
                    {"zh": "把 target 的权重量化成 FP8 再起草", "en": "Quantizing the target's weights to FP8 before drafting"},
                    {"zh": "直接复制 target 上一步的输出 token 作为草稿", "en": "Directly copying the target's previous output token as the draft"},
                ],
                "answer": 0,
                "why": {
                    "zh": "EAGLE 不另起一个完整模型，而是在<strong>特征层面</strong>续写：用 target 的<span class='mono'>隐藏状态</span>（第8课）加上一个采样 token 预测下一个特征，草稿头很小且与 target 高度对齐，<span class='mono'>accept_rate</span>(α) 因此更高。",
                    "en": "EAGLE does not raise a full second model; it continues at the <strong>feature level</strong>: using the target's <span class='mono'>hidden state</span> (Lesson 8) plus the previously sampled token to predict the next feature. The draft head is tiny and highly aligned with the target, so <span class='mono'>accept_rate</span>(α) is higher.",
                },
            },
            {
                "q": {
                    "zh": "为什么 EAGLE 提议一棵 token <strong>树</strong>能比朴素的<strong>链</strong>接受更多 token？",
                    "en": "Why does EAGLE proposing a token <strong>tree</strong> accept more tokens than a naive <strong>chain</strong>?",
                },
                "opts": [
                    {
                        "zh": "<strong>链只押一条续写，中途错则后续全废；树每步保留 topk 分支，一次目标前向并行考察多条续写，接受最长合法路径</strong>",
                        "en": "<strong>A chain bets on one continuation and wastes the rest if it errs midway; a tree keeps topk branches per step, examines many continuations in parallel in one target forward, and accepts the longest valid path</strong>",
                    },
                    {"zh": "树会多跑几次 target 前向，所以接受更多", "en": "A tree runs several more target forwards, so it accepts more"},
                    {"zh": "树把草稿模型换成了更大的模型", "en": "A tree swaps the draft model for a larger one"},
                    {"zh": "树跳过了验证步骤直接接受所有节点", "en": "A tree skips verification and accepts all nodes directly"},
                ],
                "answer": 0,
                "why": {
                    "zh": "语言天生分叉，链只赌一条路；EAGLE 在 <span class='mono'>spec_steps</span> 步里每步留 <span class='mono'>topk</span> 分支组成树，<strong>同样的目标前向次数</strong>下并行覆盖更多续写，接受其中最长合法路径，<span class='mono'>accept_length</span> 因而更高。它并没有增加 target 前向次数。",
                    "en": "Language naturally branches and a chain bets on one path; EAGLE keeps <span class='mono'>topk</span> branches per step across <span class='mono'>spec_steps</span> to form a tree, covering more continuations in parallel under the <strong>same number of target forwards</strong> and accepting the longest valid path, so <span class='mono'>accept_length</span> rises. It does not add target forwards.",
                },
            },
            {
                "q": {
                    "zh": "在 <span class='mono'>EagleVerifyInput</span> 里，哪一项让结构成为「树」而非「链」，整棵树又靠什么一次验证？",
                    "en": "In <span class='mono'>EagleVerifyInput</span>, which field makes the structure a 'tree' not a 'chain', and what verifies the whole tree at once?",
                },
                "opts": [
                    {
                        "zh": "<strong>retrieve_next_sibling 的兄弟链接造就了树；custom_mask 树注意力让每个节点只看祖先，一次目标前向验完整棵树</strong>",
                        "en": "<strong>retrieve_next_sibling's sibling links make it a tree; the custom_mask tree attention lets each node see only its ancestors, verifying the whole tree in one target forward</strong>",
                    },
                    {"zh": "draft_token 决定树形，验证靠逐个 token 串行前向", "en": "draft_token defines the shape, and verification is serial token-by-token forwards"},
                    {"zh": "spec_steps 造就了树，验证靠跳过注意力掩码", "en": "spec_steps makes the tree, and verification skips the attention mask"},
                    {"zh": "topk 造就了树，验证由 draft 模型自己完成", "en": "topk makes the tree, and the draft model verifies itself"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>retrieve_next_sibling</span> 的<strong>兄弟链接</strong>正是「树而非链」的关键（配合 <span class='mono'>retrieve_index</span> / <span class='mono'>retrieve_next_token</span> 编码树）；<span class='mono'>custom_mask</span> 实现<strong>树注意力</strong>，让每个节点只看见祖先，于是每条根→节点路径像普通序列被独立打分，<strong>一次</strong>目标前向即可验完整棵树，接受最长合法路径 + <span class='mono'>bonus_token</span>。",
                    "en": "<span class='mono'>retrieve_next_sibling</span>'s <strong>sibling links</strong> are exactly what makes it a tree not a chain (together with <span class='mono'>retrieve_index</span> / <span class='mono'>retrieve_next_token</span> encoding the tree); <span class='mono'>custom_mask</span> implements <strong>tree attention</strong> so each node sees only its ancestors, letting every root→node path be scored independently like a normal sequence, verifying the whole tree in <strong>one</strong> target forward and accepting the longest valid path + <span class='mono'>bonus_token</span>.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释 EAGLE 的两大改进：(1) 为什么<strong>特征级起草</strong>（复用第8课的隐藏状态 + 上一个 token）能让草稿头很小却与 target 对齐、从而抬高 <span class='mono'>accept_rate</span>(α)；(2) 为什么<strong>树状草稿</strong>在不增加目标前向次数的前提下能抬高 <span class='mono'>accept_length</span>(τ，含 bonus)。",
                "en": "In your own words, explain EAGLE's two improvements: (1) why <strong>feature-level drafting</strong> (reusing the Lesson 8 hidden state + previous token) makes the draft head tiny yet aligned with the target, raising <span class='mono'>accept_rate</span>(α); and (2) why <strong>tree drafting</strong> raises <span class='mono'>accept_length</span>(τ, including the bonus) without adding target forwards.",
            },
            {
                "zh": "描述 <span class='mono'>EagleVerifyInput</span> 如何用 <span class='mono'>custom_mask</span> 与 <span class='mono'>retrieve_index</span> / <span class='mono'>retrieve_next_token</span> / <span class='mono'>retrieve_next_sibling</span> 编码一棵 token 树，并解释一次目标前向里<strong>树注意力</strong>如何把每条根→节点路径当作正常序列打分、接受最长合法路径再补 <span class='mono'>bonus_token</span>；再说明第33课的注意力<strong>后端</strong>在其中扮演什么角色，以及为什么命名要用 <span class='mono'>accept</span>/<span class='mono'>bonus_token</span> 而非 <span class='mono'>accepted</span>/<span class='mono'>verified_id</span>。",
                "en": "Describe how <span class='mono'>EagleVerifyInput</span> encodes a token tree using <span class='mono'>custom_mask</span> and <span class='mono'>retrieve_index</span> / <span class='mono'>retrieve_next_token</span> / <span class='mono'>retrieve_next_sibling</span>, and explain how, in one target forward, <strong>tree attention</strong> scores every root→node path like a normal sequence, accepts the longest valid path, then appends a <span class='mono'>bonus_token</span>; then explain the role of the Lesson 33 attention <strong>backend</strong> here, and why the naming uses <span class='mono'>accept</span>/<span class='mono'>bonus_token</span> rather than <span class='mono'>accepted</span>/<span class='mono'>verified_id</span>.",
            },
        ],
    },
    "45-pd-disaggregation.html": {
        "mcq": [
            {
                "q": {
                    "zh": "为什么要把 prefill 和 decode 拆到不同的 GPU 池？",
                    "en": "Why split prefill and decode into separate GPU pools?",
                },
                "opts": [
                    {
                        "zh": "<strong>两阶段资源画像相反：prefill 是算力受限的一次大并行，decode 是带宽受限的逐 token 生成，同卡共存会在 TTFT 与 ITL 间互相干扰</strong>",
                        "en": "<strong>The two phases have opposite resource profiles: prefill is a compute-bound parallel pass, decode is bandwidth-bound token-by-token generation, and co-locating them makes TTFT and ITL interfere</strong>",
                    },
                    {"zh": "因为 decode 比 prefill 需要更大的批所以必须分开", "en": "Because decode needs larger batches than prefill so they must be separated"},
                    {"zh": "为了把模型权重量化成 FP8 再服务", "en": "To quantize the model weights to FP8 before serving"},
                    {"zh": "分开后就不再需要 KV 缓存了", "en": "After splitting, the KV cache is no longer needed"},
                ],
                "answer": 0,
                "why": {
                    "zh": "回忆第4课：<strong>prefill</strong> 把整段 prompt 一次并行算完，打满算力（compute-bound）；<strong>decode</strong> 每步只出 1 个 token 却要重读权重 + KV，卡在带宽（bandwidth-bound）。同卡共存时长 prefill 会顶高 decode 的 <span class='mono'>ITL</span>，这就是 TTFT vs ITL 的张力，chunked prefill（第22课）只能缓解。分池后各自吃饱自己的瓶颈、独立调优。",
                    "en": "Recall Lesson 4: <strong>prefill</strong> computes the whole prompt in one parallel pass and saturates compute (compute-bound); <strong>decode</strong> emits 1 token per step yet re-reads weights + KV, stuck on bandwidth (bandwidth-bound). Co-located, a long prefill raises decode's <span class='mono'>ITL</span>—the TTFT vs ITL tension that chunked prefill (Lesson 22) only mitigates. Split into pools, each saturates its own bottleneck and is tuned independently.",
                },
            },
            {
                "q": {
                    "zh": "PD 分离时，prefill 完成后到底有什么东西被传到 decode GPU？",
                    "en": "In PD disaggregation, what exactly is transferred to the decode GPU after prefill finishes?",
                },
                "opts": [
                    {
                        "zh": "<strong>请求的 KV 缓存（第30课的分页 KV）经 RDMA / NVLink 搬到 decode GPU，再开始流式吐 token</strong>",
                        "en": "<strong>The request's KV cache (Lesson 30's paged KV) is shipped over RDMA / NVLink to the decode GPU, which then streams tokens</strong>",
                    },
                    {"zh": "整套模型权重被复制过去", "en": "The entire set of model weights is copied over"},
                    {"zh": "只把已生成的最后一个 token 传过去", "en": "Only the last generated token is sent over"},
                    {"zh": "把 prefill GPU 的梯度传过去", "en": "The prefill GPU's gradients are transferred"},
                ],
                "answer": 0,
                "why": {
                    "zh": "被搬运的是请求算好的 <strong>KV 缓存</strong>——以页为单位存放（第30课分页 KV）。传输走 <span class='mono'>RDMA</span> / <span class='mono'>NVLink</span> 高速互联，目标是让传输时间远小于省下的重复计算；KV 落地后 decode 池才开始逐 token 流式生成。",
                    "en": "What's shipped is the request's computed <strong>KV cache</strong>—stored page by page (Lesson 30's paged KV). The transfer rides <span class='mono'>RDMA</span> / <span class='mono'>NVLink</span> so transfer time stays far below the recomputation it saves; once the KV lands, the decode pool begins streaming tokens one at a time.",
                },
            },
            {
                "q": {
                    "zh": "关于 <span class='mono'>BaseKVSender</span> 的 <span class='mono'>poll()</span> 契约，下面哪句对？",
                    "en": "About <span class='mono'>BaseKVSender</span>'s <span class='mono'>poll()</span> contract, which is correct?",
                },
                "opts": [
                    {
                        "zh": "<strong>poll 非阻塞，返回 KVPoll 状态（Bootstrapping / WaitingForInput / Transferring / Success / Failed），Success 即 KV 送达、decode 起步</strong>",
                        "en": "<strong>poll is non-blocking and returns a KVPoll status (Bootstrapping / WaitingForInput / Transferring / Success / Failed); Success means KV arrived and decode starts</strong>",
                    },
                    {"zh": "poll 会阻塞直到传输完成才返回", "en": "poll blocks until the transfer completes before returning"},
                    {"zh": "poll 负责真正推送 KV 页，send 只是宣告", "en": "poll actually pushes the KV pages while send only announces"},
                    {"zh": "poll 由 decode 侧独有，prefill 侧没有", "en": "poll exists only on the decode side, not on prefill"},
                ],
                "answer": 0,
                "why": {
                    "zh": "发送方接口很克制：<span class='mono'>init</span> 宣告要搬多少页，<span class='mono'>send</span> 推送 KV 页，<span class='mono'>poll</span> <strong>非阻塞</strong>返回 <span class='mono'>KVPoll</span> 状态，<span class='mono'>get_transfer_metric</span> 给出 <span class='mono'>KVTransferMetric</span>。decode 侧 <span class='mono'>BaseKVReceiver</span> 对称（init + send_metadata + poll）。poll 返回 <strong>Success</strong> 表示 KV 已落地，decode 可开吐 token；后端可换 Mooncake / NIXL / ascend。",
                    "en": "The sender interface is spare: <span class='mono'>init</span> announces how many pages move, <span class='mono'>send</span> pushes the KV pages, <span class='mono'>poll</span> returns a <strong>non-blocking</strong> <span class='mono'>KVPoll</span> status, and <span class='mono'>get_transfer_metric</span> yields a <span class='mono'>KVTransferMetric</span>. The decode-side <span class='mono'>BaseKVReceiver</span> is symmetric (init + send_metadata + poll). poll returning <strong>Success</strong> means the KV has landed and decode can begin; backends are swappable: Mooncake / NIXL / ascend.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释 PD 分离要解决的核心冲突：为什么 <strong>prefill</strong>（算力受限）和 <strong>decode</strong>（带宽受限）放同一张 GPU 上会在 <span class='mono'>TTFT</span> 与 <span class='mono'>ITL</span> 之间互相拖累（联系第4课、第8/22课），以及分成 prefill 池 + decode 池后为什么每个池都能独立调优、各自吃饱自己的瓶颈。",
                "en": "In your own words, explain the core conflict PD disaggregation solves: why putting <strong>prefill</strong> (compute-bound) and <strong>decode</strong> (bandwidth-bound) on the same GPU makes them drag each other between <span class='mono'>TTFT</span> and <span class='mono'>ITL</span> (link to Lessons 4 and 8/22), and why splitting into a prefill pool + decode pool lets each pool be tuned independently and saturate its own bottleneck.",
            },
            {
                "zh": "描述一次成功的 KV 搬运在 SGLang 抽象下的完整流程：从 prefill 侧 <span class='mono'>BaseKVSender</span> 的 <span class='mono'>init → send → poll(KVPoll)</span> 到 decode 侧镜像的 <span class='mono'>BaseKVReceiver</span>（init + send_metadata + poll），说明 <span class='mono'>poll</span> 为什么必须非阻塞、Success 意味着什么，被搬运的对象（第30课分页 KV）走什么互联，以及路由器（第13课）在其中如何为请求配对 prefill / decode worker；并前瞻第46/47课分离 + 大规模 EP。",
                "en": "Describe the full flow of one successful KV transfer under SGLang's abstraction: from the prefill-side <span class='mono'>BaseKVSender</span>'s <span class='mono'>init → send → poll(KVPoll)</span> to the mirrored decode-side <span class='mono'>BaseKVReceiver</span> (init + send_metadata + poll); explain why <span class='mono'>poll</span> must be non-blocking, what Success means, which interconnect the transferred object (Lesson 30's paged KV) rides, and how the router (Lesson 13) pairs a prefill / decode worker per request; then forward-ref Lessons 46/47's disaggregation + large-scale EP.",
            },
        ],
    },
    "46-tp-pp-ep-dp-parallelism.html": {
        "mcq": [
            {
                "q": {
                    "zh": "下面哪一项最准确地概括了四种并行各自“切”的对象？",
                    "en": "Which option most accurately summarizes what each of the four parallelisms splits?",
                },
                "opts": [
                    {
                        "zh": "<strong>TP 切一层之内的权重矩阵；PP 切层成连续阶段；EP 切 MoE 的专家集合；DP 复制模型、切请求</strong>",
                        "en": "<strong>TP splits the weight matrices within a layer; PP splits layers into consecutive stages; EP splits the MoE expert set; DP replicates the model and splits requests</strong>",
                    },
                    {"zh": "四种都在切请求，只是粒度不同", "en": "All four split requests, just at different granularities"},
                    {"zh": "TP 切层、PP 切矩阵、EP 切请求、DP 切专家", "en": "TP splits layers, PP splits matrices, EP splits requests, DP splits experts"},
                    {"zh": "它们都把模型量化成 FP8 后再分发", "en": "They all quantize the model to FP8 before distributing it"},
                ],
                "answer": 0,
                "why": {
                    "zh": "记住两类问题：<strong>TP/PP/EP</strong> 切<strong>模型本身</strong>——TP 切一层内的权重矩阵（第25/37课 q/k/v、MLP 列、词表），PP 切层与层的边界成阶段（第23课），EP 切 MoE 的专家（第34课）；<strong>DP</strong> 不切模型而是<strong>复制模型、切请求</strong>（第23课 dp-controller）。前三种分担一份模型的体积与算力，DP 分担吞吐。",
                    "en": "Two kinds of question: <strong>TP/PP/EP</strong> split the <strong>model itself</strong>—TP the weight matrices within a layer (Lessons 25/37 q/k/v, MLP columns, vocab), PP the layer boundaries into stages (Lesson 23), EP the MoE experts (Lesson 34); <strong>DP</strong> doesn't split the model but <strong>replicates it and splits requests</strong> (Lesson 23 dp-controller). The first three share one model's size and compute; DP shares throughput.",
                },
            },
            {
                "q": {
                    "zh": "关于“每一步通信什么”，下面哪句对应正确？",
                    "en": "About \"what each communicates per step\", which mapping is correct?",
                },
                "opts": [
                    {
                        "zh": "<strong>TP 每层一次 all-reduce/all-gather；PP 仅阶段交界 send/recv 激活；EP 每步两次 all-to-all 路由 token；DP 每步近乎零通信</strong>",
                        "en": "<strong>TP does an all-reduce/all-gather per layer; PP only send/recv activations at boundaries; EP does two all-to-all per step to route tokens; DP communicates near-zero per step</strong>",
                    },
                    {"zh": "TP 近乎零通信，DP 每层都要 all-reduce", "en": "TP communicates near-zero while DP all-reduces every layer"},
                    {"zh": "EP 只在阶段交界 send/recv，PP 每步两次 all-to-all", "en": "EP only send/recv at boundaries while PP does two all-to-all per step"},
                    {"zh": "四种并行每步都做一次完全相同的 all-reduce", "en": "All four do an identical all-reduce every step"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<strong>TP</strong> 把单层切片，每层都要把部分和合并，故每层一次 <span class='mono'>all_reduce</span>（或词表并行的 <span class='mono'>all_gather</span>），通信最频繁、需 NVLink。<strong>PP</strong> 只在阶段交界 <span class='mono'>send</span>/<span class='mono'>recv</span> 激活，链路便宜。<strong>EP</strong> 每步两次 <span class='mono'>all_to_all</span>（送去专家、再送回），随 token×top-k 增长。<strong>DP</strong> 副本独立，推理时每步近乎零通信。",
                    "en": "<strong>TP</strong> shards a single layer and must merge partial sums, so one <span class='mono'>all_reduce</span> per layer (or <span class='mono'>all_gather</span> for vocab parallel)—most frequent, needs NVLink. <strong>PP</strong> only <span class='mono'>send</span>/<span class='mono'>recv</span> activations at boundaries—cheap link. <strong>EP</strong> does two <span class='mono'>all_to_all</span> per step (to experts and back), growing with token×top-k. <strong>DP</strong> replicas are independent, near-zero communication per step at inference.",
                },
            },
            {
                "q": {
                    "zh": "SGLang 如何用代码统一表达 TP/PP/EP/DP，使它们可以组合（如 TP=8 × PP=2 × DP=4）？",
                    "en": "How does SGLang express TP/PP/EP/DP in code so they compose (e.g. TP=8 × PP=2 × DP=4)?",
                },
                "opts": [
                    {
                        "zh": "<strong>四者都是同一个 GroupCoordinator（带 rank/ranks/world_size/rank_in_group，暴露 all_reduce/all_gather/all_to_all_single/next_rank/prev_rank）的不同实例，建立在不同 rank 布局上</strong>",
                        "en": "<strong>All four are different instances of the same GroupCoordinator (carrying rank/ranks/world_size/rank_in_group, exposing all_reduce/all_gather/all_to_all_single/next_rank/prev_rank) over different rank layouts</strong>",
                    },
                    {"zh": "为每种并行各写一套独立的进程组类，互不复用", "en": "A separate, non-reusable process-group class is written for each parallelism"},
                    {"zh": "靠一个全局变量记录当前并行模式，一次只能开一种", "en": "A global variable records the current mode and only one parallelism can run at a time"},
                    {"zh": "把四种并行编译成不同的 CUDA kernel 选择", "en": "The four are compiled into different CUDA kernel choices"},
                ],
                "answer": 0,
                "why": {
                    "zh": "关键统一在 <span class='mono'>parallel_state.py</span> 的 <span class='mono'>GroupCoordinator</span>：它就是“一个进程组”，带 <span class='mono'>rank</span>/<span class='mono'>ranks</span>/<span class='mono'>world_size</span>/<span class='mono'>rank_in_group</span>，暴露 <span class='mono'>all_reduce</span>/<span class='mono'>all_gather</span>/<span class='mono'>all_to_all_single</span>/<span class='mono'>next_rank</span>/<span class='mono'>prev_rank</span>。TP/PP/EP/DP 只是它在不同 rank 布局上的实例，每张卡可同时属于多个组，于是天然可叠乘组合成 TP=8 × PP=2 × DP=4。",
                    "en": "The unification lives in <span class='mono'>parallel_state.py</span>'s <span class='mono'>GroupCoordinator</span>: it is \"one process group\" carrying <span class='mono'>rank</span>/<span class='mono'>ranks</span>/<span class='mono'>world_size</span>/<span class='mono'>rank_in_group</span> and exposing <span class='mono'>all_reduce</span>/<span class='mono'>all_gather</span>/<span class='mono'>all_to_all_single</span>/<span class='mono'>next_rank</span>/<span class='mono'>prev_rank</span>. TP/PP/EP/DP are just its instances over different rank layouts; each card can belong to several groups at once, so they compose into TP=8 × PP=2 × DP=4.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话区分 TP 与 PP：为什么说 TP 是“横向切一层的宽度”（把一层的权重矩阵分片，每层都要 all-reduce 合并、需 NVLink、通常单机内，联系第25/37课），而 PP 是“纵向切层的深度”（把层分成连续阶段、只在交界 send/recv 激活、链路便宜但有流水线气泡、靠微批次填满，联系第23课）；并说明二者为什么可以同时使用。",
                "en": "In your own words, distinguish TP from PP: why TP \"splits the width of a layer\" (sharding a layer's weight matrices, all-reducing every layer, needing NVLink, usually single node; link Lessons 25/37) while PP \"splits the depth of the stack\" (layers into consecutive stages, send/recv activations only at boundaries, cheap link but pipeline bubbles filled by micro-batches; link Lesson 23); and explain why the two can be used together.",
            },
            {
                "zh": "描述 EP 在一个 MoE 步中的完整数据流（第34课）：路由器选 top-k 专家后，第一次 all-to-all 把每个 token 送到拥有其专家的 rank、专家计算、第二次 all-to-all 送回，并说明通信量为何随 token×top-k 增长；再解释 SGLang 的 Attention-DP 如何把“注意力 DP + 专家 EP”组合起来，以及为什么所有这些最终都落在同一个 GroupCoordinator 抽象上；前瞻第47课大规模 EP + EPLB。",
                "en": "Describe EP's full dataflow in one MoE step (Lesson 34): after the router picks top-k experts, a first all-to-all sends each token to the rank owning its expert, experts compute, a second all-to-all sends results back; explain why communication grows with token×top-k. Then explain how SGLang's Attention-DP combines \"attention DP + expert EP\", and why all of this ultimately rests on the same GroupCoordinator abstraction; forward-ref Lesson 47's large-scale EP + EPLB.",
            },
        ],
    },
    "47-large-scale-ep-and-eplb.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在大规模 EP 下，为什么少数“热点专家”会拖垮整步的吞吐？",
                    "en": "Under large-scale EP, why do a few \"hot experts\" drag down the whole step's throughput?",
                },
                "opts": [
                    {
                        "zh": "<strong>EP 每步的 all-to-all 是同步墙，整步耗时由最忙的 rank 决定；热点专家所在 GPU 处理的 token 远多于平均，其它 GPU 只能干等</strong>",
                        "en": "<strong>EP's per-step all-to-all is a sync wall; the step time is set by the busiest rank, so the GPU holding hot experts processes far more tokens than average while the others just wait</strong>",
                    },
                    {"zh": "热点专家会占用更多显存，导致其它专家被换出", "en": "Hot experts use more VRAM, forcing other experts to be swapped out"},
                    {"zh": "路由器在热点专家上算得更慢，因为它的权重更大", "en": "The router computes slower on hot experts because their weights are larger"},
                    {"zh": "热点专家需要 FP8 量化，而冷门专家用 FP16", "en": "Hot experts require FP8 quantization while cold experts use FP16"},
                ],
                "answer": 0,
                "why": {
                    "zh": "EP 的一步 all-to-all 是一道<strong>集合同步</strong>（接第46课）：所有 rank 必须在同一屏障会合，整步墙时间由<strong>最忙的 rank</strong>决定。MoE 路由天然倾斜（接第34课），热点专家被大量 token 选中，若它们集中在某几张卡，这些卡每步处理的 token 远超平均，其余 GPU 空转。裸 EP 因此浪费算力。",
                    "en": "EP's per-step all-to-all is a <strong>collective synchronization</strong> (ties to Lesson 46): all ranks must meet at the same barrier, so the step's wall time is set by the <strong>busiest rank</strong>. MoE routing is inherently skewed (Lesson 34); hot experts are chosen by many tokens, and if they cluster on a few cards, those cards process far more tokens per step than average while the rest spin idle. Raw EP thus wastes compute.",
                },
            },
            {
                "q": {
                    "zh": "EPLB（专家并行负载均衡器）周期性地测量和重排的分别是什么？",
                    "en": "What does EPLB (Expert-Parallel Load Balancer) periodically measure and rebalance?",
                },
                "opts": [
                    {
                        "zh": "<strong>测量每个专家收到的 token 数（每专家负载），再解一个新的专家→GPU 放置方案把负载摊平，必要时复制热门专家</strong>",
                        "en": "<strong>Measures how many tokens each expert received (per-expert load), then solves a new expert→GPU placement that flattens load, replicating hot experts when needed</strong>",
                    },
                    {"zh": "测量每张 GPU 的温度，再降低热点卡的频率", "en": "Measures each GPU's temperature, then throttles the hot card's clock"},
                    {"zh": "测量每个 token 的长度，再重排 KV cache 的分页", "en": "Measures each token's length, then rearranges KV-cache paging"},
                    {"zh": "测量路由器的权重梯度，再微调模型参数", "en": "Measures the router's weight gradients, then fine-tunes model parameters"},
                ],
                "answer": 0,
                "why": {
                    "zh": "EPLB 做两件事：<strong>测量</strong>每个专家在最近一段时间实际收到多少 token（由专家分布记录器累加），与<strong>重排</strong>——基于统计解一个新的<strong>专家→GPU 放置</strong>，让每张卡每步处理的 token 数尽量相等。武器有二：在 GPU 间迁移专家、以及把太热的专家<strong>复制到多张卡</strong>分摊流量。",
                    "en": "EPLB does two things: <strong>measure</strong> how many tokens each expert actually received over a recent window (accumulated by the expert-distribution recorder), and <strong>rebalance</strong>—solving from that statistic a new <strong>expert→GPU placement</strong> so each card processes roughly equal tokens per step. Its two weapons: migrate experts between GPUs, and <strong>replicate</strong> a too-hot expert onto several cards to split its traffic.",
                },
            },
            {
                "q": {
                    "zh": "SGLang 的 <span class='mono'>EPLBManager</span> 用哪两个钩子实现“高频采集 + 低频重排”的循环？",
                    "en": "Which two hooks of SGLang's <span class='mono'>EPLBManager</span> implement the \"high-frequency collect + low-frequency rebalance\" loop?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>on_forward_pass_end()</span> 每次前向后累加每专家负载统计；<span class='mono'>rebalance()</span> 周期性解新放置并更新 <span class='mono'>expert_location</span></strong>",
                        "en": "<strong><span class='mono'>on_forward_pass_end()</span> accumulates per-expert load after each forward; <span class='mono'>rebalance()</span> periodically solves a new placement and updates <span class='mono'>expert_location</span></strong>",
                    },
                    {"zh": "<span class='mono'>on_token_start()</span> 与 <span class='mono'>flush_kv()</span>", "en": "<span class='mono'>on_token_start()</span> and <span class='mono'>flush_kv()</span>"},
                    {"zh": "<span class='mono'>compile()</span> 与 <span class='mono'>capture_graph()</span>", "en": "<span class='mono'>compile()</span> and <span class='mono'>capture_graph()</span>"},
                    {"zh": "<span class='mono'>quantize()</span> 与 <span class='mono'>dequantize()</span>", "en": "<span class='mono'>quantize()</span> and <span class='mono'>dequantize()</span>"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>EPLBManager</span> 挂在 <span class='mono'>ModelRunner</span> 上：<span class='mono'>on_forward_pass_end()</span> 在<strong>每一次前向之后</strong>被调用，把这一步各专家命中的 token 数累加进统计（高频、轻量）；<span class='mono'>rebalance()</span> <strong>周期性</strong>触发，解出新放置（热点可能被复制）并更新 <span class='mono'>expert_location</span>（专家→GPU 映射，低频、较重）。二者解耦，既跟上负载变化又不淹没正常推理。",
                    "en": "<span class='mono'>EPLBManager</span> hangs off the <span class='mono'>ModelRunner</span>: <span class='mono'>on_forward_pass_end()</span> is called <strong>after every forward</strong> to accumulate this step's per-expert token counts (high-frequency, lightweight); <span class='mono'>rebalance()</span> is triggered <strong>periodically</strong> to solve a new placement (hot experts may be replicated) and update <span class='mono'>expert_location</span> (the expert→GPU map; low-frequency, heavier). Decoupling them tracks load changes without drowning out normal inference.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释为什么裸 EP 会浪费算力：MoE 路由的长尾分布（第34课）如何制造热点专家，而 EP 的 all-to-all 同步墙（第46课）如何让整步耗时由最忙的 rank 决定；再说明为什么仅仅“把专家在 GPU 间换位置”不足以解决问题，而必须<strong>复制热门专家</strong>才能把一个不可分割的热点切成可并行的多份。",
                "en": "In your own words, explain why raw EP wastes compute: how MoE routing's long-tail distribution (Lesson 34) creates hot experts, and how EP's all-to-all sync wall (Lesson 46) makes the step time depend on the busiest rank; then explain why merely \"swapping experts between GPUs\" is not enough and why <strong>replicating hot experts</strong> is necessary to cut an indivisible hotspot into parallelizable copies.",
            },
            {
                "zh": "描述 <span class='mono'>EPLBManager</span> 的完整循环：<span class='mono'>on_forward_pass_end()</span> 如何经由专家分布记录器累加每专家负载、<span class='mono'>rebalance()</span> 如何周期性求解新的专家→GPU 放置并写回 <span class='mono'>expert_location</span>、后续步骤如何按这张新地图路由；解释为什么把“高频轻量的采集”与“低频较重的重排”解耦很关键，并前瞻第61/62课“可插拔、为规模而设计”的主题。",
                "en": "Describe the full <span class='mono'>EPLBManager</span> loop: how <span class='mono'>on_forward_pass_end()</span> accumulates per-expert load via the expert-distribution recorder, how <span class='mono'>rebalance()</span> periodically solves a new expert→GPU placement and writes it back to <span class='mono'>expert_location</span>, and how subsequent steps route against this new map; explain why decoupling \"high-frequency lightweight collection\" from \"low-frequency heavier rebalancing\" is essential, and forward-ref Lessons 61/62's \"pluggable, designed-for-scale\" themes.",
            },
        ],
    },
    "48-structured-outputs-and-jump-forward.html": {
        "mcq": [
            {
                "q": {
                    "zh": "受限解码究竟在“哪一步、对什么”动手，才能保证输出永远合法？",
                    "en": "Where exactly, and on what, does constrained decoding act to guarantee always-valid output?",
                },
                "opts": [
                    {
                        "zh": "<strong>在采样之前，对 logits（第37课）盖一层词表掩码，把所有会破坏语法的 token 设为 −∞，于是采样器（第28课）无论怎么选都合法</strong>",
                        "en": "<strong>Before sampling, it overlays a vocab mask on the logits (Lesson 37), setting every grammar-breaking token to −∞, so whatever the sampler (Lesson 28) picks stays valid</strong>",
                    },
                    {"zh": "在采样之后丢弃非法 token 并重试，直到碰巧得到合法结果", "en": "After sampling, discard illegal tokens and retry until a valid result happens to appear"},
                    {"zh": "修改采样算法本身，让 top-p 只在合法 token 上归一化", "en": "Rewrite the sampling algorithm so top-p normalizes only over legal tokens"},
                    {"zh": "对模型权重做微调，让它学会只输出 JSON", "en": "Fine-tune the model weights so it learns to emit only JSON"},
                ],
                "answer": 0,
                "why": {
                    "zh": "SGLang 把约束编译成 FSM，在每个解码步、<strong>采样之前</strong>向 FSM 要“允许集合”，据此构造词表掩码并把非法 token 的 logit 设为 <span class='mono'>−∞</span>（接第37课）。softmax 后这些 token 概率为零，采样器（第28课）无论用贪心还是 top-p 都选不到。因为改的是 logits 而非采样算法，故能与各种采样策略叠加。",
                    "en": "SGLang compiles the constraint into an FSM and, at each decode step <strong>before sampling</strong>, asks the FSM for the allowed set, builds a vocab mask, and sets illegal tokens' logits to <span class='mono'>−∞</span> (Lesson 37). After softmax those probabilities are zero, so the sampler (Lesson 28)—greedy or top-p—cannot pick them. Editing logits rather than the sampler lets it stack with any sampling strategy.",
                },
            },
            {
                "q": {
                    "zh": "“跳跃前进（压缩 FSM）”节省的是什么开销？",
                    "en": "What overhead does \"jump-forward (compressed FSM)\" save?",
                },
                "opts": [
                    {
                        "zh": "<strong>当语法在某段只有唯一续接（如 JSON 强制键名 <span class='mono'>\"name\":</span>），直接拼接整段并让 FSM 一次跳过，省掉这些被钉死 token 的模型前向</strong>",
                        "en": "<strong>When the grammar has a unique continuation over a span (e.g. the forced JSON key <span class='mono'>\"name\":</span>), it splices the whole span and jumps the FSM ahead at once, skipping the model forwards for those nailed-down tokens</strong>",
                    },
                    {"zh": "省掉构造词表掩码的显存分配", "en": "Saves the VRAM allocation for building the vocab mask"},
                    {"zh": "省掉把 logits 从 GPU 拷回 CPU 的传输", "en": "Saves transferring logits from GPU back to CPU"},
                    {"zh": "省掉对采样结果做 JSON 解析的校验", "en": "Saves running JSON parsing to validate the sampled result"},
                ],
                "answer": 0,
                "why": {
                    "zh": "很多时候 FSM 在一段连续状态里“允许集合”都只有一个 token——文法完全确定，没有自由度。逐 token 解码仍会为这些被钉死的 token 逐个跑模型前向，纯属浪费。<span class='mono'>OutlinesJumpForwardMap.jump_forward_symbol(state)</span> 返回该段强制字符串，直接 splice 进输出并让 FSM 一次性前跳：既更快，又因模型不必预测样板而质量更好。",
                    "en": "Often the FSM's allowed set is a single token across a run of states—the grammar is fully deterministic with no freedom. Per-token decode still runs a forward for each nailed-down token, pure waste. <span class='mono'>OutlinesJumpForwardMap.jump_forward_symbol(state)</span> returns that forced string, splices it in, and jumps the FSM ahead at once: faster, and better quality since the model needn't predict boilerplate.",
                },
            },
            {
                "q": {
                    "zh": "为什么投机解码（第43课）需要语法对象提供 <span class='mono'>rollback(k)</span>？",
                    "en": "Why does speculative decoding (Lesson 43) need the grammar object's <span class='mono'>rollback(k)</span>?",
                },
                "opts": [
                    {
                        "zh": "<strong>草稿模型一次提出多个 token，FSM 先假定全部接受而前进；当验证拒绝其中后几个，必须 <span class='mono'>rollback(k)</span> 精确退回，让 FSM 状态与真正被接受的序列一致</strong>",
                        "en": "<strong>The draft model proposes several tokens at once and the FSM advances assuming all are accepted; when verification rejects the trailing ones, it must <span class='mono'>rollback(k)</span> to keep the FSM state consistent with the truly accepted sequence</strong>",
                    },
                    {"zh": "为了在生成结束后撤销整段输出并重新开始", "en": "To undo the whole output after generation and restart"},
                    {"zh": "为了把 FSM 状态保存到磁盘以便断点续传", "en": "To checkpoint the FSM state to disk for resuming later"},
                    {"zh": "为了在温度过高时回退到贪心采样", "en": "To fall back to greedy sampling when temperature is too high"},
                ],
                "answer": 0,
                "why": {
                    "zh": "投机解码里草稿模型一口气给出若干候选，目标模型并行验证，常只接受前几个。FSM 在验证前必须假定全部接受并据此前进；一旦后几个被拒，就要用 <span class='mono'>rollback(k)</span> 把多走的 k 步退回，确保 FSM 状态与真正被接受的序列严格一致，否则后续掩码会基于错误状态构造、约束失效。",
                    "en": "In speculative decoding the draft model emits several candidates that the target verifies in parallel, often accepting only the first few. The FSM must advance assuming all are accepted; once the trailing ones are rejected it uses <span class='mono'>rollback(k)</span> to undo the extra k steps so the FSM state matches the truly accepted sequence—otherwise later masks build on a wrong state and the constraint breaks.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话讲清受限解码为什么能<strong>保证</strong>输出合法而不只是“大概率合法”：把 JSON-Schema/正则/EBNF 编译成 FSM 后，<span class='mono'>fill_vocab_mask</span> 如何在每步采样前把非法 token 的 logit（第37课）设为 −∞，使采样器（第28课）无论策略如何都选不到；再说明这与“生成后校验+重试”相比的根本差别。",
                "en": "In your own words, explain why constrained decoding <strong>guarantees</strong> valid output rather than just \"probably valid\": after compiling JSON-Schema/regex/EBNF into an FSM, how <span class='mono'>fill_vocab_mask</span> sets illegal tokens' logits (Lesson 37) to −∞ before each sampling step so the sampler (Lesson 28) cannot pick them regardless of strategy; then contrast this with \"generate then validate and retry.\"",
            },
            {
                "zh": "描述跳跃前进的完整收益与机制：压缩 FSM 如何识别“确定性跨度”，<span class='mono'>jump_forward_symbol(state)</span> 如何吐出被钉死的字符串并让 FSM 一次前跳从而省去模型前向；解释为什么它<strong>既更快又质量更好</strong>，并结合 <span class='mono'>accept_token</span> / <span class='mono'>is_terminated</span> 说明整台每请求 FSM 如何把推进、掩码、跳跃、终止拼成一个闭环。",
                "en": "Describe jump-forward's full payoff and mechanism: how the compressed FSM identifies \"deterministic spans,\" how <span class='mono'>jump_forward_symbol(state)</span> emits the nailed-down string and jumps the FSM ahead to skip model forwards; explain why it is <strong>both faster and higher quality</strong>, and use <span class='mono'>accept_token</span> / <span class='mono'>is_terminated</span> to show how the per-request FSM closes the loop of advance, mask, jump, and terminate.",
            },
        ],
    },
    "49-multimodal-vlm-serving.html": {
        "mcq": [
            {
                "q": {
                    "zh": "把一个纯文本引擎变成 VLM 引擎，SGLang 只动了哪两处“接缝”？",
                    "en": "To turn a text engine into a VLM engine, which two \"seams\" does SGLang change?",
                },
                "opts": [
                    {
                        "zh": "<strong>输入接缝（处理器把媒体转张量并插占位符 token）与嵌入接缝（<span class='mono'>general_mm_embed_routine</span> 跑编码器并在占位符处缝合媒体嵌入），其余引擎不动</strong>",
                        "en": "<strong>The input seam (processor turns media into tensors and inserts placeholder tokens) and the embed seam (<span class='mono'>general_mm_embed_routine</span> runs encoders and splices media embeddings at placeholders); the rest of the engine is untouched</strong>",
                    },
                    {"zh": "调度器（第18课）与分页 KV 缓存（第30课）", "en": "The scheduler (L18) and the paged KV cache (L30)"},
                    {"zh": "RadixAttention（第29课）与采样器（第28课）", "en": "RadixAttention (L29) and the sampler (L28)"},
                    {"zh": "需要重写整条流水线并新增一台 VLM 专用引擎", "en": "Rewrite the whole pipeline and add a dedicated VLM-only engine"},
                ],
                "answer": 0,
                "why": {
                    "zh": "VLM 支持被压缩成两处接缝：<strong>输入接缝</strong>在分词/预处理阶段（第14课）由每模型的 <span class='mono'>BaseMultimodalProcessor</span> 把原始像素/音频转张量并往 token 流插入占位符 token；<strong>嵌入接缝</strong>在前向入口由 <span class='mono'>general_mm_embed_routine</span> 跑编码器并把媒体嵌入散射到占位符位置。两处之外，调度、分页 KV、注意力、采样全部复用。",
                    "en": "VLM support is compressed into two seams: the <strong>input seam</strong> in tokenize/preprocess (L14), where each model's <span class='mono'>BaseMultimodalProcessor</span> turns raw pixels/audio into tensors and inserts placeholder tokens; and the <strong>embed seam</strong> at the forward entry, where <span class='mono'>general_mm_embed_routine</span> runs encoders and scatters media embeddings to placeholder positions. Beyond those two, scheduling, paged KV, attention, and sampling are all reused.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>general_mm_embed_routine</span> 究竟把媒体嵌入放到 token 序列的哪些位置？",
                    "en": "Where exactly does <span class='mono'>general_mm_embed_routine</span> place media embeddings in the token sequence?",
                },
                "opts": [
                    {
                        "zh": "<strong>放到那些 <span class='mono'>input_ids == 该模态占位符 token</span> 的位置——由 <span class='mono'>data_embedding_funcs</span> 以 <span class='mono'>placeholder_tokens</span> 为键驱动分发</strong>",
                        "en": "<strong>At the positions where <span class='mono'>input_ids == that modality's placeholder token</span> — dispatched by <span class='mono'>data_embedding_funcs</span> keyed on <span class='mono'>placeholder_tokens</span></strong>",
                    },
                    {"zh": "统一追加到序列末尾，作为额外的上下文前缀", "en": "Appended uniformly at the end of the sequence as an extra context prefix"},
                    {"zh": "插到序列最前面，覆盖系统提示词", "en": "Prepended at the very front, overwriting the system prompt"},
                    {"zh": "随机散布到所有文本 token 之间以增强鲁棒性", "en": "Scattered randomly among all text tokens to improve robustness"},
                ],
                "answer": 0,
                "why": {
                    "zh": "处理器在 token 流里为每个媒体项预留了一串占位符 token。前向时 <span class='mono'>general_mm_embed_routine</span> 先正常嵌入文本 id，再跑各模态编码器得到媒体嵌入，最后<strong>精确地</strong>把它们散射到 <span class='mono'>input_ids</span> 等于该模态占位符的槽位上。分发表 <span class='mono'>data_embedding_funcs</span> 以 <span class='mono'>placeholder_tokens</span> 为键，决定哪种占位符用哪个编码器填，保证位置与数量一一对应。",
                    "en": "The processor reserves a run of placeholder tokens in the token stream for each media item. At forward time <span class='mono'>general_mm_embed_routine</span> first embeds text ids normally, then runs each modality's encoder to get media embeddings, and finally <strong>precisely</strong> scatters them onto the slots where <span class='mono'>input_ids</span> equals that modality's placeholder. The table <span class='mono'>data_embedding_funcs</span>, keyed on <span class='mono'>placeholder_tokens</span>, decides which placeholder is filled by which encoder, keeping positions and counts in one-to-one correspondence.",
                },
            },
            {
                "q": {
                    "zh": "为什么缝合完媒体嵌入后，调度器、分页 KV、RadixAttention、采样器一行都不用改？",
                    "en": "Why, after splicing media embeddings, do the scheduler, paged KV, RadixAttention, and sampler not change a single line?",
                },
                "opts": [
                    {
                        "zh": "<strong>因为缝合后序列只是一行普通嵌入向量，语言模型主体分辨不出哪段来自图片，下游所有按位置/按 token 工作的组件无需感知模态</strong>",
                        "en": "<strong>Because after the splice the sequence is just a row of ordinary embedding vectors; the language-model body cannot tell which segment came from a picture, so every downstream component working by position/per-token needs no modality awareness</strong>",
                    },
                    {"zh": "因为媒体嵌入被单独存到另一块显存，不进入主序列", "en": "Because media embeddings are stored in a separate VRAM block and never enter the main sequence"},
                    {"zh": "因为编码器已经把图片转成了文本字符串再分词", "en": "Because the encoder already converts the image into a text string and re-tokenizes it"},
                    {"zh": "因为 VLM 关闭了连续批处理与前缀复用以简化逻辑", "en": "Because VLM disables continuous batching and prefix reuse to simplify the logic"},
                ],
                "answer": 0,
                "why": {
                    "zh": "占位符保证缝合前后序列长度不变，缝合只是把占位嵌入替换成等量的媒体嵌入。替换后这一行是均质的嵌入向量，语言模型不在乎来源。于是调度器（第18课）照样连续批处理、分页 KV（第30课）照样按页存取、RadixAttention（第29课）照样前缀复用、采样器（第28课）照样从 logits 采样——多模态信息在嵌入接缝处一次性融入，后续无需再感知。",
                    "en": "Placeholders keep the sequence length identical before and after; the splice merely replaces placeholder embeddings with an equal number of media embeddings. Afterward the row is a homogeneous set of embedding vectors and the language model does not care about origin. So the scheduler (L18) still does continuous batching, paged KV (L30) still pages, RadixAttention (L29) still reuses prefixes, and the sampler (L28) still samples from logits — multimodal info is fused once at the embed seam and never needs to be seen again.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话讲清 VLM 的“两处接缝”如何把一台文本引擎升级成多模态引擎：<strong>输入接缝</strong>里 <span class='mono'>BaseMultimodalProcessor</span> 如何把原始像素/音频转张量并往 token 流插占位符 token（第14课）；<strong>嵌入接缝</strong>里 <span class='mono'>general_mm_embed_routine</span> 如何先嵌入文本、再跑 ViT/音频编码器、并在 <span class='mono'>input_ids==占位符</span> 处散射缝合；说明 <span class='mono'>data_embedding_funcs</span> 与 <span class='mono'>placeholder_tokens</span> 各自的角色。",
                "en": "In your own words, explain how VLM's \"two seams\" upgrade a text engine into a multimodal one: in the <strong>input seam</strong>, how <span class='mono'>BaseMultimodalProcessor</span> turns raw pixels/audio into tensors and inserts placeholder tokens into the token stream (L14); in the <strong>embed seam</strong>, how <span class='mono'>general_mm_embed_routine</span> embeds text first, then runs the ViT/audio encoder, and scatter-splices where <span class='mono'>input_ids==placeholder</span>; describe the roles of <span class='mono'>data_embedding_funcs</span> and <span class='mono'>placeholder_tokens</span>.",
            },
            {
                "zh": "解释为什么“VLM = 处理媒体 + 在占位符处织入嵌入，而不是一台新引擎”：占位符如何保证序列长度在缝合前后一致，从而让调度器（第18课）、分页 KV（第30课）、RadixAttention（第29课）、采样器（第28课）全部原样复用；再讨论若编码器自带 CUDA Graph 运行器（第27课），它如何嵌入这条路径而不打扰下游。",
                "en": "Explain why \"VLM = process media + weave embeddings in at placeholders, not a new engine\": how placeholders keep the sequence length identical before and after the splice so that the scheduler (L18), paged KV (L30), RadixAttention (L29), and sampler (L28) are all reused as-is; then discuss how, if an encoder carries its own CUDA-graph runner (L27), it fits into this path without disturbing the downstream.",
            },
        ],
    },
    "50-multi-lora-batching.html": {
        "mcq": [
            {
                "q": {
                    "zh": "一个 LoRA 适配器在数学上到底是什么？",
                    "en": "What is a LoRA adapter, mathematically?",
                },
                "opts": [
                    {
                        "zh": "<strong>一个低秩小增量：学两个小矩阵 <span class='mono'>A、B</span>，让有效权重变为 <span class='mono'>W + B·A</span>，其秩 <span class='mono'>r ≪ d</span>，远小于矩阵维度</strong>",
                        "en": "<strong>A small low-rank delta: learn two small matrices <span class='mono'>A, B</span> so the effective weight becomes <span class='mono'>W + B·A</span>, with rank <span class='mono'>r ≪ d</span>, far smaller than the matrix dimension</strong>",
                    },
                    {"zh": "对整个权重矩阵 W 重新做全量微调后的新副本", "en": "A fresh copy of the entire weight matrix W after full fine-tuning"},
                    {"zh": "一份独立的完整基座模型，只是参数更少", "en": "A standalone full base model that merely has fewer parameters"},
                    {"zh": "一段缓存的 KV，用来跳过 prefill", "en": "A cached KV block used to skip prefill"},
                ],
                "answer": 0,
                "why": {
                    "zh": "LoRA 不重训庞大的 <span class='mono'>W</span>，而是学两个小矩阵 <span class='mono'>A、B</span>，让有效权重为 <span class='mono'>W + B·A</span>，秩 <span class='mono'>r ≪ d</span>，所以增量 ΔW=B·A 又小又便宜。一份基座可挂许多这样的适配器，每个是一种微调“技能”，SGLang 记为 <span class='mono'>LoRAAdapter</span>。",
                    "en": "LoRA does not retrain the huge <span class='mono'>W</span>; it learns two small matrices <span class='mono'>A, B</span> so the effective weight is <span class='mono'>W + B·A</span> with rank <span class='mono'>r ≪ d</span>, making the delta ΔW=B·A small and cheap. One base carries many such adapters, each a fine-tuned skill, represented in SGLang as <span class='mono'>LoRAAdapter</span>.",
                },
            },
            {
                "q": {
                    "zh": "同一个批次里不同请求各要各的适配器时，多 LoRA 真正的难点是什么？",
                    "en": "When different requests in one batch each want a different adapter, what is the real hard part of multi-LoRA?",
                },
                "opts": [
                    {
                        "zh": "<strong>批处理：<span class='mono'>prepare_lora_batch(forward_batch)</span> 逐请求收集所需适配器并在内存里摆放，使一次<strong>分段/分组 GEMM</strong> 能在一个内核里应用每个请求各自的 ΔW</strong>",
                        "en": "<strong>Batching: <span class='mono'>prepare_lora_batch(forward_batch)</span> gathers each request's needed adapter and stages them in memory so one <strong>segmented/grouped GEMM</strong> applies each request's own ΔW in a single kernel</strong>",
                    },
                    {"zh": "把每个适配器单独循环跑一趟小 GEMM 才是最优解", "en": "Looping a separate small GEMM per adapter is the optimal solution"},
                    {"zh": "必须为每个适配器各起一份完整基座模型", "en": "You must spin up a full base model per adapter"},
                    {"zh": "需要重启服务才能切换适配器", "en": "You must restart the service to switch adapters"},
                ],
                "answer": 0,
                "why": {
                    "zh": "若按适配器逐个循环，就退化成多次小 GEMM，把连续批处理（第5课）的吞吐拆散。<span class='mono'>prepare_lora_batch</span> 接过当前 ForwardBatch（第24课），逐请求看清每行需要哪个适配器，把权重<strong>分段摆放</strong>，使后续一次分段/分组 GEMM 就在一次内核启动里把每个请求各自的 ΔW 都应用上去。",
                    "en": "Looping per adapter degrades into many small GEMMs and shreds the throughput of continuous batching (L5). <span class='mono'>prepare_lora_batch</span> takes the current ForwardBatch (L24), sees per request which adapter each row needs, and <strong>stages/segments</strong> the weights so a single segmented/grouped GEMM applies each request's own ΔW in one kernel launch.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>max_loras_per_batch</span> 限定的是什么？适配器如何在运行时增删？",
                    "en": "What does <span class='mono'>max_loras_per_batch</span> cap, and how are adapters added/removed at runtime?",
                },
                "opts": [
                    {
                        "zh": "<strong>它限定同一批次内可共存的不同适配器数量（池/显存上限）；适配器经 <span class='mono'>load_lora_adapter</span> / <span class='mono'>unload_lora_adapter</span> 在运行时增删，无需重启</strong>",
                        "en": "<strong>It caps how many distinct adapters can coexist in one batch (pool/memory bound); adapters are added/removed at runtime via <span class='mono'>load_lora_adapter</span> / <span class='mono'>unload_lora_adapter</span>, with no restart</strong>",
                    },
                    {"zh": "它限定基座模型的最大层数", "en": "It caps the maximum number of layers in the base model"},
                    {"zh": "它限定每个请求能生成的最大 token 数", "en": "It caps the maximum tokens each request can generate"},
                    {"zh": "它限定一次只能加载一个适配器，且需重启", "en": "It allows loading only one adapter at a time and requires a restart"},
                ],
                "answer": 0,
                "why": {
                    "zh": "适配器很小，池子能装很多个；真正约束是 <span class='mono'>max_loras_per_batch</span>——同一批次内最多共存多少个不同适配器（受 LoRA 内存池/显存限制）。新技能用 <span class='mono'>load_lora_adapter</span> 进池、旧技能用 <span class='mono'>unload_lora_adapter</span> 出池，全程不需重启服务。",
                    "en": "Adapters are small, so the pool holds many; the real constraint is <span class='mono'>max_loras_per_batch</span> — how many distinct adapters can coexist in one batch (bounded by the LoRA memory pool/VRAM). New skills enter via <span class='mono'>load_lora_adapter</span> and old ones leave via <span class='mono'>unload_lora_adapter</span>, all with no service restart.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话讲清“一份基座 + 适配器池”为什么比起 N 份完整模型更省：LoRA 适配器作为低秩小增量 <span class='mono'>W + B·A</span>（秩 <span class='mono'>r ≪ d</span>）如何让一份基座（第25课）挂许多“技能”；再说明 <span class='mono'>max_loras_per_batch</span> 限定同批可共存的适配器数，以及 <span class='mono'>load_lora_adapter</span> / <span class='mono'>unload_lora_adapter</span> 如何在运行时增删而无需重启。",
                "en": "In your own words, explain why \"one base + an adapter pool\" beats N full models: how a LoRA adapter as a small low-rank delta <span class='mono'>W + B·A</span> (rank <span class='mono'>r ≪ d</span>) lets one base (L25) carry many \"skills\"; then describe how <span class='mono'>max_loras_per_batch</span> caps distinct adapters coexisting in a batch, and how <span class='mono'>load_lora_adapter</span> / <span class='mono'>unload_lora_adapter</span> add/remove them at runtime with no restart.",
            },
            {
                "zh": "解释批处理为何是多 LoRA 的真正难点：当同批请求各要各的适配器（req1→A、req2→B、req3→A…）时，<span class='mono'>prepare_lora_batch(forward_batch)</span> 如何接过 ForwardBatch（第24课）逐请求收集并摆放适配器，使<strong>分段/分组 GEMM</strong> 在一次内核里应用各自 ΔW，从而不拆散连续批处理（第5课）的吞吐；再对比第51课的 RL 也换权重，但换的是整个模型。",
                "en": "Explain why batching is the real hard part of multi-LoRA: when requests in one batch each want a different adapter (req1→A, req2→B, req3→A…), how <span class='mono'>prepare_lora_batch(forward_batch)</span> takes the ForwardBatch (L24), gathers and stages adapters per request so a <strong>segmented/grouped GEMM</strong> applies each ΔW in one kernel without shredding continuous-batching throughput (L5); then contrast with L51's RL, which also swaps weights but swaps the whole model.",
            },
        ],
    },
    "51-rl-rollout-and-weight-sync.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 RL/RLHF 后训练里，每一轮迭代的两半各是什么？为什么对 rollout 引擎有特殊约束？",
                    "en": "In RL/RLHF post-training, what are the two halves of each iteration, and why does that constrain the rollout engine?",
                },
                "opts": [
                    {
                        "zh": "<strong>策略模型先<strong>生成</strong>大量样本（rollout），训练器再<strong>打分算梯度并更新权重</strong>；下一轮 rollout 必须用<strong>刚更新的权重</strong>，所以引擎权重每步都要变</strong>",
                        "en": "<strong>The policy model first <strong>generates</strong> many samples (rollout), then the trainer <strong>scores, computes a gradient and updates weights</strong>; the next rollout must use the <strong>just-updated weights</strong>, so the engine's weights change every step</strong>",
                    },
                    {"zh": "两半分别是预填充与解码，权重始终不变", "en": "The two halves are prefill and decode, and weights never change"},
                    {"zh": "只做生成、从不更新权重，所以权重恒定", "en": "It only generates and never updates weights, so weights stay constant"},
                    {"zh": "训练器和 rollout 完全独立，互不依赖权重", "en": "The trainer and rollout are fully independent and never share weights"},
                ],
                "answer": 0,
                "why": {
                    "zh": "一轮迭代 = rollout（SGLang 高吞吐<strong>生成</strong>）+ 训练器<strong>打分、算梯度、更新策略权重</strong>。关键耦合是：下一轮 rollout 必须用刚训好的权重采样，否则训练跑偏，所以 rollout 引擎的权重<strong>每个训练步都要换一次</strong>。",
                    "en": "One iteration = rollout (high-throughput SGLang <strong>generation</strong>) + trainer <strong>scoring, gradient, policy-weight update</strong>. The crux is that the next rollout must sample with the just-trained weights or training drifts, so the rollout engine's weights <strong>get swapped every training step</strong>.",
                },
            },
            {
                "q": {
                    "zh": "为什么 SGLang 不每个训练步都<strong>重启服务器</strong>来加载新 checkpoint，而是做<strong>原地热更新</strong>？",
                    "en": "Why does SGLang do an <strong>in-place hot update</strong> instead of <strong>restarting the server</strong> each step to load a new checkpoint?",
                },
                "opts": [
                    {
                        "zh": "<strong>重启要重载 checkpoint、重新预热、重捕 CUDA Graph、重建 KV 池，几十秒～分钟级，乘以上万步会压垮计算；<span class='mono'>update_weights_from_tensor</span> 把新张量直接写进运行中的参数，CUDA Graph/KV 池原地不动</strong>",
                        "en": "<strong>A restart must reload the checkpoint, re-warm, re-capture CUDA graphs and rebuild KV pools (tens of seconds to minutes), and times thousands of steps it crushes compute; <span class='mono'>update_weights_from_tensor</span> writes new tensors straight into the running parameters while CUDA graphs/KV pools stay put</strong>",
                    },
                    {"zh": "重启更快，因为能清空所有缓存", "en": "Restarting is faster because it clears all caches"},
                    {"zh": "原地更新会破坏 CUDA Graph，所以只能重启", "en": "In-place update breaks CUDA graphs, so restarting is required"},
                    {"zh": "新权重无法写进显存，只能从磁盘重载", "en": "New weights cannot be written to VRAM, so a disk reload is mandatory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "冷启动的一次性开销（重载权重、预热、重捕 CUDA Graph、重建 KV 池）动辄几十秒以上，RL 上万步根本扛不住。原地更新让进程不重启、CUDA Graph 与 KV 池保留，只把模型参数显存里的数值换成新张量，毫秒级完成，下一轮 rollout 立即可用。",
                    "en": "Cold-start one-time costs (reload weights, warm up, re-capture CUDA graphs, rebuild KV pools) run tens of seconds or more, untenable across thousands of RL steps. In-place update keeps the process and CUDA graphs/KV pools alive, swapping only the values in the model-parameter memory for the new tensors in milliseconds, ready for the next rollout immediately.",
                },
            },
            {
                "q": {
                    "zh": "<span class='mono'>update_weights_from_tensor</span> 与 <span class='mono'>update_weights_from_distributed</span> 的区别，以及 <span class='mono'>load_format=\"flattened_bucket\"</span> 解决什么？",
                    "en": "What's the difference between <span class='mono'>update_weights_from_tensor</span> and <span class='mono'>update_weights_from_distributed</span>, and what does <span class='mono'>load_format=\"flattened_bucket\"</span> solve?",
                },
                "opts": [
                    {
                        "zh": "<strong>from_tensor 用于<strong>同卡共置</strong>（张量已在本地显存，直接原地写）；from_distributed 用于训练器在<strong>别的 GPU</strong>，经第46课进程组<strong>流式</strong>传权重；flattened_bucket 把<strong>成千张量打包成一块 buffer 整体一次搬运</strong>再到岸重建，降低逐张量开销</strong>",
                        "en": "<strong>from_tensor is for <strong>same-GPU co-location</strong> (tensors already in local memory, written in place); from_distributed is for a trainer on <strong>other GPUs</strong>, <strong>streaming</strong> weights over Lesson 46's process group; flattened_bucket <strong>packs thousands of tensors into one buffer moved once</strong> and reconstructs on arrival, cutting per-tensor overhead</strong>",
                    },
                    {"zh": "两者完全相同，flattened_bucket 只是改个日志格式", "en": "They are identical, and flattened_bucket merely changes a log format"},
                    {"zh": "from_distributed 只能传一个张量，flattened_bucket 用于压缩精度", "en": "from_distributed can only move one tensor, and flattened_bucket is for quantization"},
                    {"zh": "from_tensor 必须重启进程，flattened_bucket 用于切分 KV 缓存", "en": "from_tensor must restart the process, and flattened_bucket is for splitting the KV cache"},
                ],
                "answer": 0,
                "why": {
                    "zh": "共置同卡时新张量已在显存，from_tensor 直接原地写；分离部署时 from_distributed 经 GroupCoordinator（第46课）把权重从训练 rank 流式传进推理 rank。大模型有上万命名张量，逐个搬会把发起开销放大几千倍，flattened_bucket 用 <span class='mono'>FlattenedTensorBucket</span> 拼成一块连续 buffer、整体一次传输/拷贝、再到岸重建，作为 <span class='mono'>load_format</span> 叠加在前两者之上。",
                    "en": "Co-located on the same GPUs the new tensors are already in memory, so from_tensor writes in place; disaggregated, from_distributed streams weights from trainer ranks into inference ranks via the GroupCoordinator (L46). A large model has thousands of named tensors, and moving them one by one amplifies launch overhead thousands of times, so flattened_bucket uses <span class='mono'>FlattenedTensorBucket</span> to pack them into one contiguous buffer, transfer/copy once and reconstruct on arrival, layered as a <span class='mono'>load_format</span> on top of the first two.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话讲清 RL 的“rollout↔训练”闭环：策略模型生成样本（rollout，正是 SGLang 擅长的批量生成）、训练器打分算梯度更新权重，为什么下一轮 rollout 必须用<strong>刚更新的权重</strong>；再说明为何每步<strong>重启服务器</strong>（重载 checkpoint、重新预热、重捕 CUDA Graph）不可接受，而 <span class='mono'>update_weights_from_tensor</span> 的原地热更新让 CUDA Graph 与 KV 池原地不动。",
                "en": "In your own words, explain the RL \"rollout↔training\" loop: the policy model generates samples (rollout, exactly SGLang's batched generation), the trainer scores, computes a gradient and updates weights, and why the next rollout must use the <strong>just-updated weights</strong>; then explain why <strong>restarting the server</strong> each step (reload checkpoint, re-warm, re-capture CUDA graphs) is unacceptable while <span class='mono'>update_weights_from_tensor</span>'s in-place hot update keeps CUDA graphs and KV pools in place.",
            },
            {
                "zh": "对比三种权重同步路径并说明 <span class='mono'>FlattenedTensorBucket</span> 的价值：<span class='mono'>update_weights_from_tensor</span>（同卡共置）、<span class='mono'>update_weights_from_distributed</span>（跨卡经第46课进程组流式传）、<span class='mono'>load_format=\"flattened_bucket\"</span>（把上万命名张量打包成一块 buffer 整体一次搬运再重建）。再把它和第50课 LoRA 换权重做对比——同样是换权重，但 RL 换的是整个主干、LoRA 只换小 adapter，并说明为何同一个 SGLang 进程能既当服务器又当 rollout worker。",
                "en": "Contrast the three weight-sync paths and explain the value of <span class='mono'>FlattenedTensorBucket</span>: <span class='mono'>update_weights_from_tensor</span> (same-GPU co-location), <span class='mono'>update_weights_from_distributed</span> (cross-GPU streaming over Lesson 46's process group), and <span class='mono'>load_format=\"flattened_bucket\"</span> (pack thousands of named tensors into one buffer moved once then reconstructed). Then contrast with Lesson 50's LoRA weight swap — both swap weights, but RL swaps the whole backbone while LoRA swaps only a small adapter — and explain why one SGLang process can be both the server and the rollout worker.",
            },
        ],
    },
    "52-diffusion-models.html": {
        "mcq": [
            {
                "q": {
                    "zh": "扩散模型的生成方式和自回归 LLM 有什么本质不同？",
                    "en": "How does a diffusion model fundamentally differ from an autoregressive LLM?",
                },
                "opts": [
                    {
                        "zh": "<strong>自回归 LLM 每次前向<strong>吐一个 token</strong> 再喂回，序列变长；扩散模型从<strong>纯随机噪声</strong>出发，<strong>迭代去噪 N 步</strong>，对同一个潜变量反复精修，无 token 反馈，步数固定</strong>",
                        "en": "<strong>An autoregressive LLM emits <strong>one token per forward</strong> and feeds it back, growing the sequence; a diffusion model starts from <strong>pure random noise</strong> and <strong>iteratively denoises N steps</strong>, refining the same latent with no token feedback and a fixed step count</strong>",
                    },
                    {"zh": "两者都一个 token 一个 token 地生成，只是模型不同", "en": "Both generate token by token, only the model differs"},
                    {"zh": "扩散模型也有 token 级反馈，只是步数更多", "en": "Diffusion also has token-level feedback, just with more steps"},
                    {"zh": "扩散模型一次前向直接出图，从不迭代", "en": "Diffusion produces the image in a single forward and never iterates"},
                ],
                "answer": 0,
                "why": {
                    "zh": "自回归是变长的接龙，每个 token 依赖上一个；扩散是定长精修，从噪声出发，每一步去掉一点噪声，对同一个潜变量重复 N 次（如 20–50 步），没有 token 级反馈。",
                    "en": "Autoregression is variable-length chaining where each token depends on the last; diffusion is fixed-length refinement, starting from noise and removing a little each step, repeated N times (e.g. 20–50) over the same latent with no token feedback.",
                },
            },
            {
                "q": {
                    "zh": "一条扩散管线（<span class='mono'>PipelineConfig</span>）的三个零件分别负责什么？",
                    "en": "What do the three parts of a diffusion pipeline (<span class='mono'>PipelineConfig</span>) each do?",
                },
                "opts": [
                    {
                        "zh": "<strong>文本编码器把<strong>提示词变条件</strong>；<strong>DiT</strong>（扩散 Transformer）是<strong>去噪器</strong>，每个去噪步运行一次；<strong>VAE</strong> 在<strong>潜空间与像素空间</strong>之间映射</strong>",
                        "en": "<strong>Text encoders turn the <strong>prompt into conditioning</strong>; the <strong>DiT</strong> (Diffusion Transformer) is the <strong>denoiser</strong> run once per step; the <strong>VAE</strong> maps between <strong>latent and pixel space</strong></strong>",
                    },
                    {"zh": "三者都是去噪器，只是精度不同", "en": "All three are denoisers, only with different precision"},
                    {"zh": "DiT 负责编码提示词，VAE 负责采样噪声", "en": "The DiT encodes the prompt and the VAE samples noise"},
                    {"zh": "文本编码器负责解码像素，VAE 负责条件", "en": "The text encoder decodes pixels and the VAE handles conditioning"},
                ],
                "answer": 0,
                "why": {
                    "zh": "文本编码器产出 conditioning；DiT 是核心去噪器，每步跑一次；VAE 在压缩潜空间与真实像素之间往返，去噪全程在潜空间、最后才用 VAE 解码成图。无分类器引导（<span class='mono'>should_use_guidance</span>/<span class='mono'>embedded_cfg_scale</span>）把结果推向提示词。",
                    "en": "Text encoders produce conditioning; the DiT is the core denoiser run once per step; the VAE shuttles between compressed latent and real pixels, with denoising happening in latent space and only the final VAE decode producing the image. CFG (<span class='mono'>should_use_guidance</span>/<span class='mono'>embedded_cfg_scale</span>) steers toward the prompt.",
                },
            },
            {
                "q": {
                    "zh": "SGLang 扩散服务复用了哪些既有基础设施，TeaCache 又做了什么？",
                    "en": "Which existing infrastructure does SGLang diffusion reuse, and what does TeaCache do?",
                },
                "opts": [
                    {
                        "zh": "<strong>复用 <span class='mono'>sgl-kernel</span> 算子、调度循环、<strong>CUDA Graph 捕获</strong>（去噪步形状固定）、量化、多硬件后端、OpenAI 兼容 API；<strong>TeaCache</strong> 缓存某步结果，变化小于阈值（累积 L1 距离）时<strong>跳过冗余去噪步</strong></strong>",
                        "en": "<strong>Reuses <span class='mono'>sgl-kernel</span> ops, the scheduler loop, <strong>CUDA-graph capture</strong> (fixed-shape denoise step), quantization, multi-hardware backends and the OpenAI-compatible API; <strong>TeaCache</strong> caches a step's result and <strong>skips redundant denoise steps</strong> when the change is below an accumulated-L1 threshold</strong>",
                    },
                    {"zh": "扩散服务从零另起一套引擎，TeaCache 用来扩大噪声", "en": "Diffusion builds a brand-new engine from scratch, and TeaCache adds noise"},
                    {"zh": "只复用 API，TeaCache 用来重启服务器", "en": "Only the API is reused, and TeaCache restarts the server"},
                    {"zh": "复用全部基础设施，但 TeaCache 会增加去噪步数", "en": "All infra is reused, but TeaCache increases the number of denoise steps"},
                ],
                "answer": 0,
                "why": {
                    "zh": "扩散没有另起炉灶，而是借用同一套服务化地基：sgl-kernel（第38课）、调度循环、CUDA Graph（第27课）、量化（第35课）、多硬件（第42课）、OpenAI API（第15课）乃至 PD 分离。TeaCache 是叠在上面的扩散特有优化：缓存去噪步结果，变化足够小就跳过，提速而几乎不损画质。",
                    "en": "Diffusion doesn't start over; it borrows the same serving foundation: sgl-kernel (L38), the scheduler loop, CUDA graph (L27), quantization (L35), multi-hardware (L42), the OpenAI API (L15), even PD disaggregation. TeaCache is a diffusion-specific optimization layered on top: cache a denoise step's result and skip it when the change is small enough, speeding up with almost no quality loss.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话讲清扩散模型的“从噪声走向图像”：为什么它是<strong>定长 N 步迭代去噪</strong>而不是自回归的 token 接龙；说明文本编码器、DiT 去噪器、VAE 三个零件各自的角色，以及无分类器引导如何把结果推向提示词。",
                "en": "In your own words, explain diffusion's \"from noise toward an image\": why it is a <strong>fixed-length N-step iterative denoise</strong> rather than autoregressive token chaining; describe the roles of the text encoder, the DiT denoiser and the VAE, and how classifier-free guidance steers the result toward the prompt.",
            },
            {
                "zh": "解释“一套引擎，两种范式”：列出 SGLang 扩散服务复用了哪些你前面学过的部件（如第38课 sgl-kernel、第27课 CUDA Graph、第35课量化、第42课多硬件、第15课 OpenAI API），并说明为什么<strong>去噪步形状固定</strong>特别适合 CUDA Graph 捕获，以及 TeaCache 如何通过累积 L1 距离阈值跳过冗余步。",
                "en": "Explain \"one stack, two paradigms\": list which previously learned pieces SGLang diffusion reuses (e.g. L38 sgl-kernel, L27 CUDA graph, L35 quantization, L42 multi-hardware, L15 OpenAI API), explain why the <strong>fixed-shape denoise step</strong> is especially suited to CUDA-graph capture, and how TeaCache skips redundant steps via an accumulated-L1 threshold.",
            },
        ],
    },
    "53-build-and-run.html": {
        "mcq": [
            {
                "q": {
                    "zh": "为什么说 <span class='mono'>ServerArgs</span> 是“把全书旋钮收成 CLI flag”？这对调优部署意味着什么？",
                    "en": "Why is <span class='mono'>ServerArgs</span> described as \"gathering the whole guide's knobs into CLI flags\", and what does that mean for tuning a deployment?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>ServerArgs</span> 是一个 dataclass，<strong>每个字段自动映射成一个 --kebab-case flag</strong>（如 <span class='mono'>tp_size</span>→<span class='mono'>--tp-size</span>）；前面学的每个组件都成了一个旋钮，调优 = 设置对的 flag</strong>",
                        "en": "<strong><span class='mono'>ServerArgs</span> is a dataclass and <strong>each field auto-maps to a --kebab-case flag</strong> (e.g. <span class='mono'>tp_size</span>→<span class='mono'>--tp-size</span>); every component you learned becomes a knob, so tuning = setting the right flags</strong>",
                    },
                    {"zh": "ServerArgs 只存模型路径，其它参数都写在配置文件里", "en": "ServerArgs only stores the model path; everything else lives in a config file"},
                    {"zh": "每个 flag 都要手写解析函数，和组件没有对应关系", "en": "Each flag needs a hand-written parser and has no correspondence to components"},
                    {"zh": "flag 名是随机生成的，无法从字段名推断", "en": "Flag names are randomly generated and cannot be inferred from field names"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>ServerArgs</span> 是 <span class='mono'>@dataclasses.dataclass</span>，字段名按 kebab-case 规则自动变成 CLI flag。于是“一墙 flag”其实就是字段清单，每个 flag 背后都是你学过的机制（并行、注意力后端、量化、KV 显存……），调优部署就是拧对这些旋钮。",
                    "en": "<span class='mono'>ServerArgs</span> is a <span class='mono'>@dataclasses.dataclass</span> whose field names auto-become kebab-case CLI flags. The 'wall of flags' is just the field list, and each flag is a mechanism you already learned (parallelism, attention backend, quantization, KV memory…), so tuning is twisting the right knobs.",
                },
            },
            {
                "q": {
                    "zh": "你想让 KV 缓存池占用更多 GPU 显存，并在 flashinfer / triton / fa 之间切换注意力内核，分别该设哪个 flag？",
                    "en": "To give the KV cache pool more GPU memory and to switch the attention kernel among flashinfer / triton / fa, which flags do you set?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>--mem-fraction-static</span> 控制 KV 池占的 GPU 显存（第30/31课）；<span class='mono'>--attention-backend</span> 选注意力内核（第33课）</strong>",
                        "en": "<strong><span class='mono'>--mem-fraction-static</span> controls the GPU memory the KV pool gets (Lessons 30/31); <span class='mono'>--attention-backend</span> picks the attention kernel (Lesson 33)</strong>",
                    },
                    {"zh": "<span class='mono'>--tp-size</span> 控制显存，<span class='mono'>--quantization</span> 选注意力内核", "en": "<span class='mono'>--tp-size</span> controls memory and <span class='mono'>--quantization</span> picks the attention kernel"},
                    {"zh": "<span class='mono'>--max-running-requests</span> 控制显存，<span class='mono'>--dp-size</span> 选注意力内核", "en": "<span class='mono'>--max-running-requests</span> controls memory and <span class='mono'>--dp-size</span> picks the attention kernel"},
                    {"zh": "<span class='mono'>--chunked-prefill-size</span> 控制显存，<span class='mono'>--speculative-algorithm</span> 选注意力内核", "en": "<span class='mono'>--chunked-prefill-size</span> controls memory and <span class='mono'>--speculative-algorithm</span> picks the attention kernel"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>--mem-fraction-static</span> 决定静态显存中分给 KV 池的比例；<span class='mono'>--attention-backend</span> 在 flashinfer/triton/fa 之间选内核。其它 flag 各管各的：<span class='mono'>--tp-size</span>/<span class='mono'>--dp-size</span> 管并行，<span class='mono'>--quantization</span> 管量化，<span class='mono'>--chunked-prefill-size</span> 切 prefill，<span class='mono'>--max-running-requests</span> 限批大小。",
                    "en": "<span class='mono'>--mem-fraction-static</span> decides the share of static memory given to the KV pool; <span class='mono'>--attention-backend</span> chooses among flashinfer/triton/fa. Other flags do their own jobs: <span class='mono'>--tp-size</span>/<span class='mono'>--dp-size</span> for parallelism, <span class='mono'>--quantization</span> for quantization, <span class='mono'>--chunked-prefill-size</span> to split prefills, <span class='mono'>--max-running-requests</span> to cap the batch.",
                },
            },
            {
                "q": {
                    "zh": "在批量离线评测脚本里，不想起 HTTP 服务、不走网络，直接在内存里调用生成，应选哪种模式？",
                    "en": "For a batch offline-eval script that wants no HTTP server and no network, calling generation directly in memory, which mode should you choose?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>Engine</span>——进程内 / 离线模式，在 Python 脚本里直接实例化，无 HTTP；而 HTTP 服务器是面向在线、多客户端的常驻服务</strong>",
                        "en": "<strong><span class='mono'>Engine</span> — the in-process / offline mode, instantiated directly in a Python script with no HTTP; the HTTP server is the long-lived online service for many clients</strong>",
                    },
                    {"zh": "必须启动 HTTP 服务器，再用 localhost 自己请求自己", "en": "You must launch the HTTP server and then request yourself over localhost"},
                    {"zh": "两种模式都需要网络，Engine 只是端口不同", "en": "Both modes need the network; Engine just uses a different port"},
                    {"zh": "Engine 不支持生成，只能做健康检查", "en": "Engine cannot generate and is only for health checks"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>Engine</span> 是离线 / 进程内门面：直接 import 实例化，在内存里调用生成，省去起服务和发 HTTP 的全部开销，适合脚本化与评测。HTTP 服务器则是在线门面，监听端口、对外暴露 <span class='mono'>/generate</span> 与 <span class='mono'>/v1/...</span>。两者共享同一内核。",
                    "en": "<span class='mono'>Engine</span> is the offline / in-process facade: import and instantiate it, call generation in memory, skipping all the overhead of standing up a service and firing HTTP — ideal for scripting and eval. The HTTP server is the online facade, listening on a port and exposing <span class='mono'>/generate</span> and <span class='mono'>/v1/...</span>. Both share the same core.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话描述一条 <span class='mono'>python -m sglang.launch_server</span> 命令背后发生的事：从命令行入口用 <span class='mono'>prepare_server_args</span> 解析进 <span class='mono'>ServerArgs</span>、再把它交给 <span class='mono'>run_server</span> 启动引擎三件套（第13课），再到暴露 HTTP 端点（第15课）。并说明为什么“每个 dataclass 字段自动映射成 --kebab-case flag”能把“一墙 flag”变成“我早就理解的组件”。",
                "en": "In your own words, describe what happens behind one <span class='mono'>python -m sglang.launch_server</span> command: from the entry point using <span class='mono'>prepare_server_args</span> to parse into <span class='mono'>ServerArgs</span>, to handing it to <span class='mono'>run_server</span> to boot the engine trio (Lesson 13), to exposing the HTTP endpoint (Lesson 15). Explain why \"each dataclass field auto-maps to a --kebab-case flag\" turns a 'wall of flags' into 'components I already understand'.",
            },
            {
                "zh": "对比 <span class='mono'>Engine</span>（离线 / 进程内，无 HTTP）与 HTTP 服务器（在线）两种门面：各自的使用场景、开销与对外接口是什么？再举出至少 4 个你会用到的 flag（如 <span class='mono'>--tp-size</span>、<span class='mono'>--quantization</span>、<span class='mono'>--mem-fraction-static</span>、<span class='mono'>--max-running-requests</span>），分别说明它们控制什么、对应前面哪一课。",
                "en": "Contrast the two facades — <span class='mono'>Engine</span> (offline / in-process, no HTTP) versus the HTTP server (online): what are their use cases, overhead, and external interfaces? Then name at least 4 flags you'd use (e.g. <span class='mono'>--tp-size</span>, <span class='mono'>--quantization</span>, <span class='mono'>--mem-fraction-static</span>, <span class='mono'>--max-running-requests</span>), explaining what each controls and which earlier lesson it maps to.",
            },
        ],
    },
    "54-benchmark-and-profiling.html": {
        "mcq": [
            {
                "q": {
                    "zh": "把 batch 堆大后，<span class='mono'>output_throughput</span> 上去了，但单条请求感觉更卡了。最可能是哪个指标变差，为什么？",
                    "en": "After enlarging the batch, <span class='mono'>output_throughput</span> went up but a single request feels choppier. Which metric most likely got worse, and why?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>tpot</span>（token 间隔 / ITL）变大——吞吐与延迟是取舍（第8课），堆大 batch 提高了整体每秒 token，却拉长了单条请求相邻 token 的间隔</strong>",
                        "en": "<strong><span class='mono'>tpot</span> (inter-token latency / ITL) increased — throughput and latency trade off (Lesson 8); a bigger batch raises overall tokens/s but stretches the gap between a single request's adjacent tokens</strong>",
                    },
                    {"zh": "<span class='mono'>completed</span> 变小，因为请求都失败了", "en": "<span class='mono'>completed</span> dropped because requests all failed"},
                    {"zh": "<span class='mono'>input_throughput</span> 变大导致更卡", "en": "<span class='mono'>input_throughput</span> grew, which causes the choppiness"},
                    {"zh": "和任何指标都无关，纯粹是网络抖动", "en": "Unrelated to any metric; purely network jitter"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>output_throughput</span> 衡量整台服务器每秒吐多少 token，而单条请求的顺滑度由 <span class='mono'>tpot</span>（相邻 token 间隔，即 ITL）决定。堆大 batch 提高总吞吐，却可能延长每条请求的 token 间隔，于是用户感觉变卡——这正是第8课吞吐/延迟取舍在压测数字上的体现。",
                    "en": "<span class='mono'>output_throughput</span> measures the whole server's tokens/s, but a single request's smoothness is set by <span class='mono'>tpot</span> (the gap between adjacent tokens, i.e. ITL). A bigger batch lifts total throughput yet can lengthen each request's inter-token gap, so it feels choppier — exactly Lesson 8's throughput/latency trade-off showing up in benchmark numbers.",
                },
            },
            {
                "q": {
                    "zh": "压测报告里 <span class='mono'>mean_ttft_ms</span> 看着不错，但用户抱怨“有时候巨慢”。你最该盯哪个数字？",
                    "en": "In the report <span class='mono'>mean_ttft_ms</span> looks fine, yet users complain it's \"sometimes painfully slow.\" Which number should you watch most?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>p99_ttft_ms</span>——延迟是分布不是单值；负载下的长尾(p99)才是用户真正感受到的卡顿，只看 mean 会掩盖尾部</strong>",
                        "en": "<strong><span class='mono'>p99_ttft_ms</span> — latency is a distribution, not one value; under load the tail (p99) is what users actually feel, and the mean hides it</strong>",
                    },
                    {"zh": "只看 <span class='mono'>mean_ttft_ms</span> 就够了，分位数没意义", "en": "<span class='mono'>mean_ttft_ms</span> alone is enough; percentiles are meaningless"},
                    {"zh": "<span class='mono'>request_throughput</span>，因为它等于延迟", "en": "<span class='mono'>request_throughput</span>, because it equals latency"},
                    {"zh": "<span class='mono'>completed</span>，越大延迟越低", "en": "<span class='mono'>completed</span>; bigger means lower latency"},
                ],
                "answer": 0,
                "why": {
                    "zh": "同一批请求的 ttft 是一条分布（mean/median/p90/p95/p99）。高并发下排队、CUDA graph 重放抖动、偶发长 prompt 会制造长尾，99% 很快但 1% 极慢正是用户印象最深的体验。所以负载之下要报 <span class='mono'>p99</span>，而不是只报均值。",
                    "en": "The ttft of one batch is a distribution (mean/median/p90/p95/p99). Under concurrency, queuing, CUDA-graph replay jitter, and occasional long prompts create a tail; 99% fast but 1% very slow is the impression users remember. So under load report <span class='mono'>p99</span>, not just the mean.",
                },
            },
            {
                "q": {
                    "zh": "压测数字证实“确实慢”，但你不知道慢在哪一步。下一步该用什么、怎么用？",
                    "en": "The benchmark confirms \"it's slow,\" but you don't know which step. What's the next tool and how do you use it?",
                },
                "opts": [
                    {
                        "zh": "<strong>用 profiler：设 <span class='mono'>SGLANG_TORCH_PROFILER_DIR</span> 或调 <span class='mono'>/start_profile</span>+<span class='mono'>/stop_profile</span> 抓 torch trace，在 <span class='mono'>chrome://tracing</span>/Perfetto 看每个 CUDA kernel，分辨 prefill/decode、gap、CUDA graph(第27课)</strong>",
                        "en": "<strong>Use the profiler: set <span class='mono'>SGLANG_TORCH_PROFILER_DIR</span> or call <span class='mono'>/start_profile</span>+<span class='mono'>/stop_profile</span> to capture a torch trace, then in <span class='mono'>chrome://tracing</span>/Perfetto inspect each CUDA kernel to tell prefill/decode, gaps, CUDA graph (Lesson 27) apart</strong>",
                    },
                    {"zh": "再跑一遍 bench_serving，数字会自己解释原因", "en": "Re-run bench_serving; the numbers will explain the cause by themselves"},
                    {"zh": "直接重启服务器，慢的问题会消失", "en": "Just restart the server and the slowness goes away"},
                    {"zh": "把 mean 改成 median 上报就行", "en": "Report median instead of mean and you're done"},
                ],
                "answer": 0,
                "why": {
                    "zh": "benchmark 是黑盒（量化“有多慢”），profiler 是白盒（定位“为什么慢”）。设 <span class='mono'>SGLANG_TORCH_PROFILER_DIR</span> 或用 <span class='mono'>/start_profile</span>+<span class='mono'>/stop_profile</span> 抓 trace，在 <span class='mono'>chrome://tracing</span> 打开时间线，就能看到每个 CUDA kernel、prefill/decode 区段、gap(launch 开销)与 CUDA graph 重放，把“它慢”翻译成“这个 kernel/这段空隙的问题”。",
                    "en": "The benchmark is black-box (quantify \"how slow\"); the profiler is white-box (locate \"why slow\"). Set <span class='mono'>SGLANG_TORCH_PROFILER_DIR</span> or use <span class='mono'>/start_profile</span>+<span class='mono'>/stop_profile</span> to capture a trace, open the timeline in <span class='mono'>chrome://tracing</span>, and see every CUDA kernel, prefill/decode regions, gaps (launch overhead), and CUDA-graph replay — translating \"it's slow\" into \"this kernel/gap is the problem.\"",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释“压测负责量化、分析负责定位”的两段式循环：<span class='mono'>bench_serving</span> 如何向运行中的服务器发射负载并产出 <span class='mono'>BenchmarkMetrics</span>？再说明 <span class='mono'>ttft</span>（=prefill 延迟）与 <span class='mono'>tpot</span>（=token 间隔/ITL）分别对应用户的什么体验，并联系第4课（prefill 计算受限 vs decode 带宽受限）。",
                "en": "In your own words explain the two-stage loop of \"benchmark quantifies, profile locates\": how does <span class='mono'>bench_serving</span> fire a load at a running server and produce <span class='mono'>BenchmarkMetrics</span>? Then explain what <span class='mono'>ttft</span> (= prefill latency) and <span class='mono'>tpot</span> (= inter-token latency / ITL) each correspond to in the user's experience, tying in Lesson 4 (prefill compute-bound vs decode bandwidth-bound).",
            },
            {
                "zh": "为什么一份诚实的压测报告必须给出分位数（mean/median/p90/p95/p99）而不是只报均值？请结合“长尾在负载下产生的来源”作答。另外，<span class='mono'>bench_serving</span> 与 <span class='mono'>bench_one_batch</span> 各适合什么场景，为什么 kernel 级 A/B 用后者？",
                "en": "Why must an honest benchmark report give percentiles (mean/median/p90/p95/p99) rather than only the mean? Answer in terms of where the tail comes from under load. Also, when is <span class='mono'>bench_serving</span> appropriate versus <span class='mono'>bench_one_batch</span>, and why use the latter for kernel-level A/B?",
            },
        ],
    },
    "55-test-suite-and-ci.html": {
        "mcq": [
            {
                "q": {
                    "zh": "你想验证一个采样函数算得对，希望它在每个 PR 上都快速跑一遍。该写哪种测试，怎么写？",
                    "en": "You want to verify a sampling function computes correctly and have it run fast on every PR. Which kind of test, and how?",
                },
                "opts": [
                    {
                        "zh": "<strong>单元测试：直接 <span class='mono'>import</span> 这个函数，构造输入并 <span class='mono'>assert</span> 输出——不启动服务、毫秒级、CI 每个 PR 都跑</strong>",
                        "en": "<strong>A unit test: directly <span class='mono'>import</span> the function, build inputs and <span class='mono'>assert</span> the output — no server, millisecond-scale, CI runs it on every PR</strong>",
                    },
                    {"zh": "端到端测试：必须先用 <span class='mono'>popen_launch_server</span> 拉起服务器才能测函数", "en": "An e2e test: you must launch a server with <span class='mono'>popen_launch_server</span> first to test a function"},
                    {"zh": "不写测试，直接在生产里观察", "en": "Skip tests and just observe in production"},
                    {"zh": "只能手动在 REPL 里试，无法自动化", "en": "Only manual REPL trials; it can't be automated"},
                ],
                "answer": 0,
                "why": {
                    "zh": "单元测试不需要服务器：import 目标函数/类、构造输入、断言输出即可。它快、稳、便宜，所以 CI 在每个 PR 上都跑。端到端测试才需要 <span class='mono'>popen_launch_server</span> 拉起真实服务器，用于校验整套系统的输出/准确率。",
                    "en": "A unit test needs no server: import the target function/class, build inputs, assert the output. It's fast, stable, and cheap, so CI runs it on every PR. Only e2e tests need <span class='mono'>popen_launch_server</span> to launch a real server, used to verify the whole system's outputs/accuracy.",
                },
            },
            {
                "q": {
                    "zh": "SGLang 的测试为什么继承 <span class='mono'>CustomTestCase</span> 而不是裸的 <span class='mono'>unittest.TestCase</span>？它解决了什么问题？",
                    "en": "Why do SGLang tests inherit <span class='mono'>CustomTestCase</span> instead of bare <span class='mono'>unittest.TestCase</span>? What problem does it solve?",
                },
                "opts": [
                    {
                        "zh": "<strong>它包裹 <span class='mono'>setUpClass</span>，保证即便 setup 抛异常 <span class='mono'>tearDownClass</span> 也一定执行——原生 unittest 会跳过收尾，导致泄漏端口/GPU 进程、连累后面的测试</strong>",
                        "en": "<strong>It wraps <span class='mono'>setUpClass</span> so that <span class='mono'>tearDownClass</span> always runs even if setup raises — plain unittest skips teardown, leaking ports/GPU processes and cascade-failing later tests</strong>",
                    },
                    {"zh": "它让测试跑得更快，因为跳过了 teardown", "en": "It makes tests faster by skipping teardown"},
                    {"zh": "它替换了 pytest，成为新的运行器", "en": "It replaces pytest as the new runner"},
                    {"zh": "它自动给每个测试加上网络 mock", "en": "It auto-mocks the network for every test"},
                ],
                "answer": 0,
                "why": {
                    "zh": "原生 <span class='mono'>unittest</span> 在 <span class='mono'>setUpClass</span> 失败时会跳过 <span class='mono'>tearDownClass</span>。在会启动服务器的套件里，这意味着已占用的端口和拉起的 GPU 进程不会被回收，后续测试连环失败。<span class='mono'>CustomTestCase</span> 包裹 setUpClass、保证清理一定发生，让一个坏掉的 setup 不会毒害整套测试。",
                    "en": "Plain <span class='mono'>unittest</span> skips <span class='mono'>tearDownClass</span> when <span class='mono'>setUpClass</span> fails. In a suite that launches servers, that means occupied ports and spawned GPU processes aren't reclaimed, and later tests cascade-fail. <span class='mono'>CustomTestCase</span> wraps setUpClass and guarantees cleanup happens, so one broken setup doesn't poison the whole suite.",
                },
            },
            {
                "q": {
                    "zh": "在 CI 真正跑测试之前，是什么先把代码风格/lint 卡了一道关？测试又是怎样在 GPU 上跑完一整套的？",
                    "en": "Before CI even runs the tests, what first gates code style/lint? And how does the whole suite finish on GPUs?",
                },
                "opts": [
                    {
                        "zh": "<strong><span class='mono'>pre-commit</span> 先在本地和 CI 卡风格/lint；测试登记在 <span class='mono'>test/registered/</span> 下，被分片(partition)派发到打标签的 GPU 运行机上并行跑完</strong>",
                        "en": "<strong><span class='mono'>pre-commit</span> first gates style/lint locally and in CI; tests are registered under <span class='mono'>test/registered/</span> and partitioned across labelled GPU runners to run in parallel</strong>",
                    },
                    {"zh": "没有任何风格检查；所有测试在一台机器上串行跑", "en": "No style check at all; all tests run serially on one machine"},
                    {"zh": "<span class='mono'>pytest</span> 负责 lint，测试只在 CPU 上跑", "en": "<span class='mono'>pytest</span> does the linting and tests run only on CPU"},
                    {"zh": "CI 随机抽一个测试跑，其余忽略", "en": "CI runs one random test and ignores the rest"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>pre-commit</span> 在 CI 开跑前先把风格/lint 卡一道——风格不过后面都免谈。测试登记在 <span class='mono'>test/registered/</span>，CI 把它们分片派发到多台打标签的 GPU 机并行执行，于是庞大的套件也能在合理时间内跑完。贡献者工作流：在 <span class='mono'>test/registered/unit/…</span> 找/加测试 → 本地 pytest → push → CI 真机重跑。",
                    "en": "<span class='mono'>pre-commit</span> gates style/lint before CI starts — if style fails, nothing else matters. Tests live under <span class='mono'>test/registered/</span>, and CI partitions them across multiple labelled GPU machines to run in parallel, so even a huge suite finishes in reasonable time. Contributor workflow: find/add a test under <span class='mono'>test/registered/unit/…</span> → local pytest → push → CI re-runs on real hardware.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释单元测试与端到端测试的分工：为什么单元测试能在每个 PR 上快跑，而端到端测试要用 <span class='mono'>popen_launch_server</span> 拉起真实的 <span class='mono'>sglang.launch_server</span>（第13课）再校验输出/准确率？两者各能抓到哪类 bug？",
                "en": "In your own words explain the division of labor between unit and e2e tests: why can unit tests run fast on every PR, while e2e tests must launch a real <span class='mono'>sglang.launch_server</span> (Lesson 13) via <span class='mono'>popen_launch_server</span> and then verify outputs/accuracy? What class of bug does each catch?",
            },
            {
                "zh": "详细说明 <span class='mono'>CustomTestCase</span> 为什么要包裹 <span class='mono'>setUpClass</span>：原生 <span class='mono'>unittest</span> 在 setup 失败时的行为是什么，会造成什么后果（端口/GPU 进程泄漏、连累后面测试）？再把测试体系和第56课（运行这些测试的 PR 流程）联系起来。",
                "en": "Explain in detail why <span class='mono'>CustomTestCase</span> wraps <span class='mono'>setUpClass</span>: what does plain <span class='mono'>unittest</span> do when setup fails, and what consequences follow (leaked ports/GPU processes, cascade-failing later tests)? Then connect the test system to Lesson 56 (the PR flow that runs these tests).",
            },
        ],
    },
    "56-code-conventions-and-pr.html": {
        "mcq": [
            {
                "q": {
                    "zh": "你刚 clone 完代码，想给 SGLang 贡献一个功能。正确的起手流程是怎样的？",
                    "en": "You've just cloned the code and want to contribute a feature to SGLang. What's the correct opening flow?",
                },
                "opts": [
                    {
                        "zh": "<strong>先 <span class='mono'>fork</span> 官方仓库（新贡献者没有 push 权限）→ clone 你的 fork → <span class='mono'>git checkout -b</span> 开分支 → 改动 → <span class='mono'>pre-commit</span> + 测试 → push → 对着 <span class='mono'>main</span> 开 PR</strong>",
                        "en": "<strong>First <span class='mono'>fork</span> the official repo (new contributors have no push access) → clone your fork → <span class='mono'>git checkout -b</span> a branch → change → <span class='mono'>pre-commit</span> + test → push → open a PR against <span class='mono'>main</span></strong>",
                    },
                    {"zh": "直接 push 到官方仓库的 <span class='mono'>main</span> 分支", "en": "Push directly to the official repo's <span class='mono'>main</span> branch"},
                    {"zh": "在官方仓库上直接改，不开分支也不 fork", "en": "Edit the official repo directly, no branch and no fork"},
                    {"zh": "先开 PR，再慢慢补代码和测试", "en": "Open the PR first, then slowly fill in code and tests later"},
                ],
                "answer": 0,
                "why": {
                    "zh": "新贡献者没有官方仓库的写权限，所以必须先 fork 一份属于自己的副本，再 clone 这个 fork、开分支改动。改完用 <span class='mono'>pre-commit</span> 过格式/lint、按第55课补测试并本地跑通，最后 push 到你的 fork 并对着 <span class='mono'>main</span> 开 PR。每一环都不能跳。",
                    "en": "New contributors have no write access to the official repo, so you must fork your own copy first, then clone that fork and branch. After changing, run <span class='mono'>pre-commit</span> for format/lint, add a test per Lesson 55 and run it locally, then push to your fork and open a PR against <span class='mono'>main</span>. No link can be skipped.",
                },
            },
            {
                "q": {
                    "zh": "你跑 <span class='mono'>pre-commit run --all-files</span>，第一次就失败了，还改动了你的文件。这说明什么？该怎么办？",
                    "en": "You run <span class='mono'>pre-commit run --all-files</span> and it fails the first time, having also modified your files. What does that mean and what should you do?",
                },
                "opts": [
                    {
                        "zh": "<strong>它在就地自动修复（重排 import、补空行等），不是 bug——把这些改动 add 进来再重跑，反复直到全部 PASS，再开 PR</strong>",
                        "en": "<strong>It auto-fixes in place (reordering imports, adding blank lines, etc.), not a bug — add those changes and re-run, repeating until everything PASSES, then open the PR</strong>",
                    },
                    {"zh": "说明 <span class='mono'>pre-commit</span> 坏了，应该卸载它", "en": "It means <span class='mono'>pre-commit</span> is broken and should be uninstalled"},
                    {"zh": "忽略它，直接 push，CI 会帮你修", "en": "Ignore it and just push; CI will fix it for you"},
                    {"zh": "把被改动的文件 revert 掉再 push", "en": "Revert the modified files and then push"},
                ],
                "answer": 0,
                "why": {
                    "zh": "<span class='mono'>pre-commit</span> 是格式/lint 关卡，第一次失败通常是它在就地修复你的文件（black、isort 等）。正确做法是把这些自动改动 add 进来后再跑一次，反复重跑直到全绿。必须在开 PR 之前过关，因为 CI（第55课）会跑完全相同的检查，本地不过 CI 也不过，PR 会被卡住。",
                    "en": "<span class='mono'>pre-commit</span> is the format/lint gate, and a first failure is usually it fixing your files in place (black, isort, etc.). The right move is to add those auto-changes and run again, repeating until all green. It must pass before you open a PR, because CI (Lesson 55) runs the exact same checks — if it fails locally it fails in CI, and the PR is blocked.",
                },
            },
            {
                "q": {
                    "zh": "你改了 <span class='mono'>python/sglang/srt/</span> 下的一个文件，还想顺手更新文档。按 SGLang 的约定该怎么做？",
                    "en": "You modified a file under <span class='mono'>python/sglang/srt/</span> and also want to update docs. What do SGLang's conventions require?",
                },
                "opts": [
                    {
                        "zh": "<strong>去 <span class='mono'>test/registered/unit/</span> 给你的改动加/补测试；文档只写在 <span class='mono'>docs_new/</span>（改老的 <span class='mono'>docs/</span> 会被 lint 拒绝）</strong>",
                        "en": "<strong>Add/extend a test under <span class='mono'>test/registered/unit/</span> for your change; docs go only in <span class='mono'>docs_new/</span> (editing the old <span class='mono'>docs/</span> is rejected by lint)</strong>",
                    },
                    {"zh": "运行时核心代码不需要测试；文档随便写在 <span class='mono'>docs/</span> 或 <span class='mono'>docs_new/</span>", "en": "Runtime-core code needs no test; docs can go in either <span class='mono'>docs/</span> or <span class='mono'>docs_new/</span>"},
                    {"zh": "另起一套新结构，不必模仿现有模块模式", "en": "Invent a brand-new structure; no need to follow existing module patterns"},
                    {"zh": "测试和文档都交给 reviewer 补", "en": "Leave both tests and docs for the reviewer to add"},
                ],
                "answer": 0,
                "why": {
                    "zh": "改运行时核心（<span class='mono'>python/sglang/srt/</span>）必须在 <span class='mono'>test/registered/unit/</span> 补上覆盖，没有测试护栏不会被接受。文档目录已经冻结，老 <span class='mono'>docs/</span> 的改动会被 lint 直接拒绝，新文档一律写 <span class='mono'>docs_new/</span>。再加上顺着已有可插拔接缝（第33/42课）模仿现成结构，reviewer 才审得快。",
                    "en": "Changing runtime core (<span class='mono'>python/sglang/srt/</span>) requires coverage under <span class='mono'>test/registered/unit/</span>; without a test guardrail it won't be accepted. The docs tree is frozen — edits to the old <span class='mono'>docs/</span> are rejected by lint, so all new docs go in <span class='mono'>docs_new/</span>. Plus, follow existing pluggable seams (Lessons 33/42) so a reviewer reviews fast.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话讲清楚从 fork 到 PR 合并的完整流程，并说明 <span class='mono'>pre-commit</span> 为什么是一道必须在开 PR 之前通过的关卡（联系第55课的 CI）。",
                "en": "In your own words, walk through the full flow from fork to merged PR, and explain why <span class='mono'>pre-commit</span> is a gate that must pass before opening a PR (connect it to CI in Lesson 55).",
            },
            {
                "zh": "什么样的 PR 是「好 PR」？从大小、聚焦、测试、动机四个角度说明，并解释「一切皆可插拔」（新后端第33/42课、新模型第26课、新处理器第49课）为什么让贡献变得容易、也让你的改动更容易被接受。",
                "en": "What makes a PR a 'good PR'? Discuss size, focus, tests, and motivation, and explain why 'everything is pluggable' (new backend Lessons 33/42, new model Lesson 26, new processor Lesson 49) makes contributing easy and your change more likely to be accepted.",
            },
        ],
    },
    "58-radixattention-as-a-first-class-idea.html": {
        "mcq": [
            {
                "q": {
                    "zh": "本课说 RadixAttention 是 SGLang 的「一等公民」而不是「贴上去的优化」。这句话最准确的含义是？",
                    "en": "This lesson says RadixAttention is a 'first-class idea' in SGLang, not a 'bolt-on optimization'. What does that most accurately mean?",
                },
                "opts": [
                    {
                        "zh": "<strong>「KV 缓存就是一棵共享前缀的基数树」被当作核心数据结构，整个引擎围着它设计，而不是事后在旁边贴一个独立的 LRU 字典</strong>",
                        "en": "<strong>'The KV cache is a radix tree of shared prefixes' is treated as the core data structure the whole engine is designed around, not a separate LRU dictionary glued on afterward</strong>",
                    },
                    {"zh": "它是一个运行最快的注意力 CUDA kernel，和缓存结构无关", "en": "It is just the fastest attention CUDA kernel, unrelated to cache structure"},
                    {"zh": "它只是 DSL 层一个可选的语法糖，底层并不依赖它", "en": "It is merely optional syntactic sugar at the DSL layer that the engine does not rely on"},
                    {"zh": "它是一个独立微服务，需要单独部署", "en": "It is a standalone microservice that must be deployed separately"},
                ],
                "answer": 0,
                "why": {
                    "zh": "一等公民意味着这个念头被刻意、反复地写进系统多个层次：缓存本身的形状就是基数树。调度、显存、分层都围着它协同，所以改它会牵动全局——这正是「结构」而非「外挂」的标志。",
                    "en": "First-class means the idea is written, on purpose and repeatedly, into many layers: the cache's very shape is a radix tree. Scheduling, memory, and tiering all coordinate around it, so changing it ripples across the whole system — the hallmark of a structure, not a bolt-on.",
                },
            },
            {
                "q": {
                    "zh": "下列哪一组课程讲的「其实是同一件事」——共享前缀这一个念头在不同层换了张面孔？",
                    "en": "Which group of lessons are 'really about the same thing' — the single idea of shared prefixes wearing a different face at each layer?",
                },
                "opts": [
                    {
                        "zh": "<strong>DSL fork/join（第11课）、前缀缓存概念（第7课）、基数树实现（第29课）、分页 KV（第30课，承第6课）、缓存感知调度（第20课）、HiCache 分层（第31课）</strong>",
                        "en": "<strong>DSL fork/join (Lesson 11), prefix-cache concept (Lesson 7), radix-tree implementation (Lesson 29), paged KV (Lesson 30, on Lesson 6), cache-aware scheduling (Lesson 20), HiCache tiering (Lesson 31)</strong>",
                    },
                    {"zh": "量化、张量并行、投机解码、采样温度", "en": "Quantization, tensor parallelism, speculative decoding, sampling temperature"},
                    {"zh": "日志格式、CI 流水线、PR 规范、文档目录", "en": "Log format, CI pipeline, PR conventions, docs directory"},
                    {"zh": "REST 路由、鉴权、限流、监控面板", "en": "REST routing, auth, rate limiting, monitoring dashboard"},
                ],
                "answer": 0,
                "why": {
                    "zh": "这六课分别在语言层、概念层、实现层、物理层、调度层、分层存储层描述同一个共享前缀世界观：故意制造共享、承诺只算一次、落地成基数树、指向分页、优先命中、按冷热分层。把它们叠在一起才看见贯穿的主轴。",
                    "en": "These six lessons describe the same shared-prefix worldview at the language, concept, implementation, physical, scheduling, and tiering layers: deliberately create sharing, promise to compute once, land as a radix tree, point at pages, favor hits, and tier by temperature. Stack them and the spine appears.",
                },
            },
            {
                "q": {
                    "zh": "把一次请求放到这棵树上，它的「生命周期」按正确顺序是？",
                    "en": "Placed on this tree, what is a request's 'lifecycle' in the correct order?",
                },
                "opts": [
                    {
                        "zh": "<strong>match_prefix 匹配最长已缓存前缀 → 复用该前缀的 KV（inc_lock_ref 钉住）→ 只对新后缀做前向计算 → insert 写回树并在分叉处劈开节点</strong>",
                        "en": "<strong>match_prefix finds the longest cached prefix → reuse that prefix's KV (inc_lock_ref pins it) → forward-compute only the new suffix → insert writes back, splitting the node at divergence</strong>",
                    },
                    {"zh": "重新计算整条序列的 KV → 写满整棵树 → 再删掉重复", "en": "Recompute the whole sequence's KV → fill the entire tree → then delete duplicates"},
                    {"zh": "先 evict 所有叶子 → 再匹配前缀 → 最后才计算", "en": "Evict all leaves first → then match prefix → compute last"},
                    {"zh": "只算前缀、丢弃新后缀，以节省显存", "en": "Compute only the prefix and discard the new suffix to save memory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "精髓是「匹配最长前缀 → 复用 KV → 只算新后缀 → 写回树」。复用的前缀要 inc_lock_ref 钉住防止半路被回收，新后缀算完后 insert 挂回树、在分叉处劈开节点，让未来请求也能共享。整个过程避免了对相同开头的重复计算。",
                    "en": "The essence is 'match the longest prefix → reuse KV → compute only the new suffix → write back to the tree'. The reused prefix is pinned by inc_lock_ref so it isn't evicted mid-flight, and after the new suffix is computed, insert hangs it back and splits the node at divergence so future requests can share it. This avoids recomputing identical openings.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释：为什么把「共享前缀」当作一等公民数据结构（基数树），而不是引擎旁边一个贴上去的 LRU 缓存，会让调度（第20课）、显存分页（第30课）和分层（第31课）都能围着它协同？对比这两种做法的根本差别。",
                "en": "In your own words, explain why treating 'shared prefixes' as a first-class data structure (a radix tree) rather than an LRU cache bolted beside the engine lets scheduling (Lesson 20), paged memory (Lesson 30), and tiering (Lesson 31) all coordinate around it. Contrast the fundamental difference between the two approaches.",
            },
            {
                "zh": "挑出本课提到的至少三处「共享前缀的不同面孔」（如 DSL fork/join 第11课、前缀缓存概念第7课、基数树实现第29课），说明它们骨子里是同一件事；再用「匹配最长前缀 → 复用 KV → 只算新后缀」串起一次请求在 RadixCache 里走完的全过程。",
                "en": "Pick at least three of the 'different faces of shared prefixes' from this lesson (e.g., DSL fork/join Lesson 11, the prefix-cache concept Lesson 7, the radix-tree implementation Lesson 29), explain why they are the same thing underneath, then use 'match the longest prefix → reuse KV → compute only the new suffix' to trace a request's full path through RadixCache.",
            },
        ],
    },
    "59-zero-overhead-scheduling.html": {
        "mcq": [
            {
                "q": {
                    "zh": "本课说零开销调度的「一条主线」是什么？",
                    "en": "What is the 'single thread' running through zero-overhead scheduling in this lesson?",
                },
                "opts": [
                    {
                        "zh": "<strong>CPU 绝不能让 GPU 等：任何在前向之前或之后的 CPU 工作都是潜在「气泡」，要把它藏到 GPU 计算背后</strong>",
                        "en": "<strong>The CPU must never make the GPU wait: any CPU work before or after the forward is a potential 'bubble' to be hidden behind GPU compute</strong>",
                    },
                    {"zh": "让 CPU 完全不做任何工作，所有逻辑都搬到 GPU 上", "en": "Make the CPU do no work at all and move every bit of logic onto the GPU"},
                    {"zh": "通过提高 CPU 主频来追上 GPU 的速度", "en": "Raise the CPU clock speed to catch up with the GPU"},
                    {"zh": "把每一步都串行执行以保证结果正确", "en": "Run every step serially to guarantee correctness"},
                ],
                "answer": 0,
                "why": {
                    "zh": "零开销不是 CPU 不干活，而是 CPU 的活不占额外墙钟时间——它们被重叠进 GPU 前向里。现代 GPU 极快，前向前后那点 CPU 工作就成了 GPU 干等的气泡，整个调度子系统的目标就是消灭这些气泡。",
                    "en": "Zero-overhead doesn't mean the CPU is idle; it means the CPU's work costs no extra wall-clock time because it is overlapped into the GPU forward. A modern GPU is so fast that the CPU work around the forward becomes an idle bubble, and the whole scheduling subsystem aims to kill those bubbles.",
                },
            },
            {
                "q": {
                    "zh": "重叠调度（第21课）打开 enable_overlap 后，GPU 在跑 forward(批 N) 的同时，CPU 在做什么？",
                    "en": "With overlap scheduling (Lesson 21) and enable_overlap on, what is the CPU doing while the GPU runs forward(batch N)?",
                },
                "opts": [
                    {
                        "zh": "<strong>已经在组装批 N+1，并且并行处理批 N-1 的结果，让 GPU 一刻不停</strong>",
                        "en": "<strong>Already assembling batch N+1 and processing batch N-1's results in parallel, so the GPU never idles</strong>",
                    },
                    {"zh": "空等 GPU 返回，什么也不做", "en": "Idly waiting for the GPU to return, doing nothing"},
                    {"zh": "重新计算批 N 的 KV 缓存以校验结果", "en": "Recomputing batch N's KV cache to verify the result"},
                    {"zh": "把 GPU 暂停下来等 CPU 把队列清空", "en": "Pausing the GPU until the CPU drains the queue"},
                ],
                "answer": 0,
                "why": {
                    "zh": "朴素事件循环是串行的：组批→前向→处理结果，GPU 在两段 CPU 时间里都闲着。重叠调度把循环改成流水线：GPU 跑 forward(N) 时 CPU 同时组装 N+1、处理 N-1，CPU 开销被塞进 GPU 前向的影子里，墙钟上几乎看不见。",
                    "en": "A naive event loop is serial: form batch → forward → process results, leaving the GPU idle during both CPU segments. Overlap scheduling turns it into a pipeline: while the GPU runs forward(N), the CPU assembles N+1 and processes N-1, so the CPU overhead is tucked into the GPU forward's shadow and is nearly invisible in wall-clock time.",
                },
            },
            {
                "q": {
                    "zh": "下列「CPU 气泡 → SGLang 如何藏它」的配对，哪一组完全正确？",
                    "en": "Which group of 'CPU bubble → how SGLang hides it' pairings is entirely correct?",
                },
                "opts": [
                    {
                        "zh": "<strong>调度/结果处理→重叠调度(第18/21课)；长 prefill 堵塞→分块预填充(第22课)；内核启动开销→CUDA 图(第27课)；分词解码→进程拆分+IPC(第14/16课)</strong>",
                        "en": "<strong>schedule/result processing→overlap scheduler (L18/21); long-prefill stall→chunked prefill (L22); kernel-launch overhead→CUDA graphs (L27); tokenize/detokenize→process split+IPC (L14/16)</strong>",
                    },
                    {"zh": "调度→量化；长 prefill→张量并行；内核启动→温度采样；分词→负载均衡", "en": "schedule→quantization; long prefill→tensor parallelism; kernel launch→temperature sampling; tokenize→load balancing"},
                    {"zh": "所有气泡都只能靠加 GPU 显存解决", "en": "Every bubble can only be solved by adding GPU memory"},
                    {"zh": "气泡只存在于 prefill，decode 阶段没有气泡", "en": "Bubbles exist only in prefill; decode has no bubbles"},
                ],
                "answer": 0,
                "why": {
                    "zh": "六课六机制其实是同一哲学的不同侧面：事件循环+重叠调度藏掉调度与结果处理，分块预填充防止长 prefill 堵死流水线，CUDA 图消除逐个内核启动的开销，进程拆分+IPC 让分词解码离开前向关键路径。",
                    "en": "Six lessons, six mechanisms, all sides of one philosophy: the event loop + overlap scheduler hide scheduling and result processing, chunked prefill stops a long prefill from clogging the pipeline, CUDA graphs erase per-kernel launch overhead, and the process split + IPC move tokenize/detokenize off the forward's critical path.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释：为什么「GPU 越快，CPU 气泡越致命」？结合一次 decode 前向只要几毫秒、而调度+采样+拼 token 也是毫秒级这个事实，说明为什么消灭气泡从锦上添花升级成了决定吞吐的头等大事。",
                "en": "In your own words, explain why 'the faster the GPU, the deadlier the CPU bubble'. Using the fact that a decode forward may take only a few milliseconds while scheduling + sampling + appending tokens is also milliseconds, argue why killing bubbles graduates from a nice-to-have into the top factor deciding throughput.",
            },
            {
                "zh": "挑出本课提到的至少三处「零开销主线的不同面孔」（如事件循环第18课、重叠调度第21课、分块预填充第22课、CUDA 图第27课、进程拆分第14/16课），说明它们骨子里都是同一句话：找到每个 CPU 气泡，把它藏到 GPU 计算背后或挪到别的进程。",
                "en": "Pick at least three 'different faces of the zero-overhead thread' from this lesson (e.g., the event loop Lesson 18, overlap scheduling Lesson 21, chunked prefill Lesson 22, CUDA graphs Lesson 27, the process split Lesson 14/16), and explain why they are all the same sentence underneath: find every CPU bubble and hide it behind GPU compute or move it to another process.",
            },
        ],
    },
    "60-two-workloads-one-engine.html": {
        "mcq": [
            {
                "q": {
                    "zh": "本课的「一条根本对立」指的是什么？",
                    "en": "What is the 'one root tension' this lesson is about?",
                },
                "opts": [
                    {
                        "zh": "<strong>prefill 是算力受限（compute-bound）的一次大并行，decode 是带宽受限（bandwidth-bound）的逐字生成，两者性质相反却要共用一台引擎</strong>",
                        "en": "<strong>prefill is a compute-bound big parallel pass while decode is bandwidth-bound token-by-token generation; opposite in nature yet sharing one engine</strong>",
                    },
                    {"zh": "prefill 和 decode 其实是同一种负载，只是名字不同", "en": "prefill and decode are really the same workload under different names"},
                    {"zh": "对立只存在于多卡之间，单卡上没有矛盾", "en": "The tension exists only across multiple GPUs, never on a single one"},
                    {"zh": "decode 才是算力受限，prefill 是带宽受限", "en": "decode is the compute-bound one and prefill is bandwidth-bound"},
                ],
                "answer": 0,
                "why": {
                    "zh": "prefill 一次处理整段 prompt，权重读一次被成百上千 token 复用，算术强度高、吃满乘加单元，是 compute-bound；decode 每步只出一个 token 却要重读整套权重+全部 KV，算术强度低、乘加大半空转，是 bandwidth-bound。一台引擎要同时服务两者，这是第4课埋下的根本对立。",
                    "en": "Prefill processes the whole prompt at once—weights read once, reused by hundreds of tokens—so its arithmetic intensity is high and it saturates the math units: compute-bound. Decode emits one token per step yet re-reads the full weights + all KV, low arithmetic intensity, math units mostly idle: bandwidth-bound. One engine must serve both—the root tension seeded in Lesson 4.",
                },
            },
            {
                "q": {
                    "zh": "为什么在同一台 GPU 上 prefill 和 decode 会「互相打架」？",
                    "en": "Why do prefill and decode 'fight each other' on the same GPU?",
                },
                "opts": [
                    {
                        "zh": "<strong>一个超长 prompt 的 prefill 是一大块算力密集的活，串行排队时会堵住所有正在解码的请求，让大家的 ITL 飙高、TTFT 与 ITL 互相拉扯</strong>",
                        "en": "<strong>A very long prompt's prefill is a big compute-heavy chunk; queued serially it blocks every decoding request, spiking everyone's ITL and pitting TTFT against ITL</strong>",
                    },
                    {"zh": "因为它们使用不同的注意力算法，互相不兼容", "en": "Because they use different attention algorithms that are incompatible"},
                    {"zh": "因为 GPU 显存不够大，无法同时放下两份权重", "en": "Because GPU memory is too small to hold two copies of the weights"},
                    {"zh": "其实并不打架，先来先做就能两全其美", "en": "They don't actually fight; first-come-first-served satisfies both"},
                ],
                "answer": 0,
                "why": {
                    "zh": "prefill 想要大块、独占、算力拉满；decode 想要频繁、低延迟、雨露均沾。把两种负载塞进一个串行队列，长 prefill 一上 GPU 就把正在解码的请求全堵住，ITL 瞬间飙高（第8课的吞吐 vs 延迟）。一台 GPU 无法同时满足两种诉求，所以必须在调度层动脑筋。",
                    "en": "Prefill wants big, exclusive, compute-maxed; decode wants frequent, low-latency, fair-shared. Stuff both into one serial queue and a long prefill, once on the GPU, blocks every decoding request and spikes ITL (Lesson 8's throughput vs latency). One GPU can't satisfy both, so the scheduler must get clever.",
                },
            },
            {
                "q": {
                    "zh": "「同卡分时（chunked prefill）」与「分机分离（PD 分离）」这两条路线的区别是什么？",
                    "en": "What distinguishes the 'co-located (chunked prefill)' path from the 'separated (PD disaggregation)' path?",
                },
                "opts": [
                    {
                        "zh": "<strong>chunked prefill 把 prefill 切片插进 decode 缝隙，让一块 GPU 时间分片地服务两者（第22课）；PD 分离给两种负载各一池 GPU 再搬 KV，让各自饱和各自瓶颈、互不干扰（第45课）</strong>",
                        "en": "<strong>chunked prefill slices prefill into decode's gaps so one GPU time-shares both (L22); PD disaggregation gives each workload its own GPU pool and transfers the KV, so each saturates its own bottleneck without interference (L45)</strong>",
                    },
                    {"zh": "两者完全相同，只是叫法不一样", "en": "They are identical, just named differently"},
                    {"zh": "chunked prefill 需要两池 GPU，PD 分离只需要一块 GPU", "en": "chunked prefill needs two GPU pools while PD needs only one GPU"},
                    {"zh": "PD 分离不需要传输 KV，chunked prefill 才需要", "en": "PD disaggregation needs no KV transfer; only chunked prefill does"},
                ],
                "answer": 0,
                "why": {
                    "zh": "两条路在解同一道题：chunked prefill 是共置方案，用聪明调度把 prefill 切片、和 decode 拼进同一 batch，时间分片地共用一块 GPU；PD 分离是分离方案，物理上拆成 prefill 池和 decode 池，靠 BaseKVManager 这层接缝把 KV 从 prefill 侧搬到 decode 侧。一个靠调度，一个靠物理分开。",
                    "en": "Both solve the same problem: chunked prefill is the co-located answer—smart scheduling slices prefill and packs it into the same batch as decode, time-sharing one GPU; PD disaggregation is the separated answer—physically split into prefill and decode pools, with the BaseKVManager seam moving KV from the prefill side to the decode side. One via scheduling, one via physical separation.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释：为什么单纯给 decode 阶段堆更多算力几乎没用，而把 batch 做大才是真正的解药？请结合「decode 每步要重读整套权重+全部 KV」这一事实，说明它为何是 bandwidth-bound 而非 compute-bound。",
                "en": "In your own words, explain why simply piling more compute onto the decode phase barely helps, while growing the batch is the real cure. Using the fact that 'each decode step re-reads the full weights + all KV', argue why decode is bandwidth-bound rather than compute-bound.",
            },
            {
                "zh": "本课说「一个轴看懂全局」。请从第4、8、22、45课中至少挑三处，说明 prefill-vs-decode 这一条根本对立如何解释了这些看似无关的设计——它们要么靠聪明调度（共置），要么靠物理分开（分离）。",
                "en": "The lesson claims 'one axis explains the whole'. Pick at least three of Lessons 4, 8, 22, 45 and show how the prefill-vs-decode root tension explains these seemingly unrelated designs—each reconciling it either by smart scheduling (co-located) or by physical separation (disaggregated).",
            },
        ],
    },
    "61-draft-for-parallel-verify.html": {
        "mcq": [
            {
                "q": {
                    "zh": "本课提炼出的「通用设计思路」是什么？",
                    "en": "What 'general design move' does this lesson distill?",
                },
                "opts": [
                    {
                        "zh": "<strong>花便宜的、可并行的工作，去绕开一个串行瓶颈——把延迟受限的串行步骤变成吞吐受限的并行步骤</strong>",
                        "en": "<strong>Spend cheap, parallelizable work to dodge a sequential bottleneck—turn latency-bound serial steps into throughput-bound parallel ones</strong>",
                    },
                    {"zh": "用更大的模型一次性生成所有 token", "en": "Use a bigger model to generate all tokens at once"},
                    {"zh": "把 decode 改成训练那样的一次性并行前向，从而改变输出", "en": "Rewrite decode as a one-shot parallel forward like training, changing the output"},
                    {"zh": "只在多卡场景才有效，单卡无法应用", "en": "It only works across multiple GPUs, never on a single one"},
                ],
                "answer": 0,
                "why": {
                    "zh": "投机解码背后真正深刻的不是某个新机制，而是一种贯穿 SGLang 的通用思路：decode 天生串行、带宽受限（第4课），就花一份便宜草稿把「k 个串行 forward」换成「一次并行验证」（第43课）。本质是拿闲置算力换被省下的串行时间。",
                    "en": "What's deep about speculative decoding isn't a new mechanism but a general move across SGLang: decode is inherently serial and bandwidth-bound (Lesson 4), so spend a cheap draft to swap 'k serial forwards' for 'one parallel verify' (Lesson 43). The essence is trading spare compute for saved sequential time.",
                },
            },
            {
                "q": {
                    "zh": "草稿模型猜出 k 个 token 后，目标模型如何处理它们？输出会和普通 decode 不同吗？",
                    "en": "After the draft proposes k tokens, how does the target model handle them, and does the output differ from plain decode?",
                },
                "opts": [
                    {
                        "zh": "<strong>目标模型一次并行 forward 验证全部 k 个，accept 最长正确前缀并在第一处分歧收下 bonus token；验证是无损的，输出与逐字 decode 完全一致</strong>",
                        "en": "<strong>The target verifies all k in one parallel forward, accepts the longest correct prefix and takes a bonus token at the first disagreement; verification is lossless, output identical to plain decode</strong>",
                    },
                    {"zh": "目标模型直接信任草稿，不做验证，所以更快但会改变输出", "en": "The target trusts the draft without verifying, so it's faster but changes the output"},
                    {"zh": "目标模型逐个重新生成这 k 个 token，没有任何并行", "en": "The target regenerates the k tokens one-by-one with no parallelism"},
                    {"zh": "只有草稿和目标完全一致时才能用，否则结果出错", "en": "It only works when draft and target fully agree, otherwise results are wrong"},
                ],
                "answer": 0,
                "why": {
                    "zh": "验证比生成便宜且可并行：把 k 个候选拼成一个序列做一次 forward，目标模型同时算出每个位置自己会选什么。一致的最长正确前缀全部 accept，第一处分歧处目标这次顺带算出的正确 token 就是 bonus token。因为最终都以目标模型为准，输出与普通 decode 完全一致——无损。",
                    "en": "Verifying is cheaper than generating and parallelizes: stitch the k candidates into one sequence for a single forward, and the target computes what it would pick at each position. The matching longest correct prefix is all accepted; at the first disagreement the correct token the target computed is the bonus token. Since the target always has final say, output is identical to plain decode—lossless.",
                },
            },
            {
                "q": {
                    "zh": "下列哪些 SGLang 技巧和投机解码是「同一个形状」？",
                    "en": "Which SGLang tricks share the 'same shape' as speculative decoding?",
                },
                "opts": [
                    {
                        "zh": "<strong>CUDA Graph 录制固定序列让 GPU 重放、省掉每步 CPU 停顿（第27课），以及重叠调度让 CPU 工作与 GPU forward 并行（第59课）——都是把串行步骤变并行</strong>",
                        "en": "<strong>CUDA Graph records a fixed sequence so the GPU replays it without per-step CPU stalls (Lesson 27), and the overlap scheduler runs CPU work parallel with the GPU forward (Lesson 59)—both turn serial steps into parallel ones</strong>",
                    },
                    {"zh": "量化和剪枝，因为它们都让模型变小", "en": "Quantization and pruning, because both shrink the model"},
                    {"zh": "只有 EAGLE 的 token 树算同一形状，其余都无关", "en": "Only EAGLE's token tree counts; the rest are unrelated"},
                    {"zh": "没有任何其他技巧是同一形状，投机解码是孤立的", "en": "No other trick shares the shape; speculative decoding is isolated"},
                ],
                "answer": 0,
                "why": {
                    "zh": "认得「把串行变并行」这个形状后就会到处看到它：CUDA Graph 把本该一步步发指令的串行过程压成一次性重放（第27课）；重叠调度让 CPU 与 GPU 并行而非互相等待（第59课）；EAGLE 用 token 树一次验证多条路径（第44课）。它们都是同一条主线的不同切面。",
                    "en": "Once you recognize 'turn serial into parallel', you see it everywhere: CUDA Graph compresses a step-by-step serial dispatch into a one-shot replay (Lesson 27); the overlap scheduler runs CPU and GPU in parallel instead of waiting (Lesson 59); EAGLE verifies many paths at once with a token tree (Lesson 44). All facets of one through-line.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释这笔「交易」：投机解码花了什么、换回了什么？请结合「decode 是串行且带宽受限、每步算力闲置」这一事实（第4课），说明为什么那次并行验证几乎是「免费」的。",
                "en": "In your own words, explain the 'trade': what does speculative decoding spend and what does it get back? Using the fact that decode is serial and bandwidth-bound with idle compute per step (Lesson 4), argue why that parallel verify is nearly 'free'.",
            },
            {
                "zh": "本课说看懂这个形状后，你会自动问「哪里还能草稿一把、再并行验证？」。请从第27、43、44、59课中至少挑三处，说明「把串行步骤变成并行步骤」这条统一原则如何解释这些看似无关的设计，并说明为何投机解码的验证必须保持无损。",
                "en": "The lesson says that once you see the shape you'll ask 'where else could we draft-and-verify?'. Pick at least three of Lessons 27, 43, 44, 59 and show how the unifying principle 'turn serial steps into parallel ones' explains these seemingly unrelated designs, and explain why speculative decoding's verification must stay lossless.",
            },
        ],
    },
    "62-everything-is-pluggable.html": {
        "mcq": [
            {
                "q": {
                    "zh": "本课提炼出的「反复出现的设计范型」是什么？",
                    "en": "What 'recurring design archetype' does this lesson distill?",
                },
                "opts": [
                    {
                        "zh": "<strong>面向接口（抽象基类）编程，把每条会变化的轴推到接口背后，具体实现在部署时选定——核心保持稳定</strong>",
                        "en": "<strong>Program to an interface (an ABC), push every variable axis behind it, and select the concrete implementation at deploy time—keeping the core stable</strong>",
                    },
                    {"zh": "把所有功能都硬编码进调度器以追求最高性能", "en": "Hard-code every feature into the scheduler for maximum performance"},
                    {"zh": "每加一种硬件就重写一遍模型执行循环", "en": "Rewrite the model-execution loop for each new piece of hardware"},
                    {"zh": "在运行时随机选择实现以提升鲁棒性", "en": "Pick implementations randomly at runtime to improve robustness"},
                ],
                "answer": 0,
                "why": {
                    "zh": "这是一节综合课：它点亮贯穿全书的暗线——一个接口 + 多个实现 + 部署时选一个。引擎核心（调度器、模型执行循环、内存池）保持稳定，变化被关在可插拔的边缘。",
                    "en": "This is a synthesis lesson: it lights up the book's through-line—one interface + many implementations + pick one at deploy time. The engine core (scheduler, model-execution loop, memory pools) stays stable while variation is fenced behind pluggable edges.",
                },
            },
            {
                "q": {
                    "zh": "下列哪一项最准确地描述了 SGLang 里「稳定核心」与「可插拔边缘」的关系？",
                    "en": "Which best describes the relation between SGLang's 'stable core' and its 'pluggable edges'?",
                },
                "opts": [
                    {
                        "zh": "<strong>核心（Scheduler、模型执行循环、内存池）只认抽象接口；注意力后端、平台、量化、投机、并行、语法、KV 传输等都是满足接口的可替换实现</strong>",
                        "en": "<strong>The core (Scheduler, model loop, memory pools) knows only abstract interfaces; the attention backend, platform, quantization, speculation, parallelism, grammar, KV transfer are swappable implementations satisfying those interfaces</strong>",
                    },
                    {"zh": "核心每次部署都要重新编译以适配具体实现", "en": "The core must be recompiled per deploy to fit each implementation"},
                    {"zh": "边缘组件必须直接修改内存池才能工作", "en": "Edge components must edit the memory pools directly to work"},
                    {"zh": "只有注意力后端是可插拔的，其余都焊死在核心里", "en": "Only the attention backend is pluggable; everything else is welded into the core"},
                ],
                "answer": 0,
                "why": {
                    "zh": "核心稳定，是其他一切能自由替换的前提。注意力后端（第33课）、平台（第42课）、量化（第35课）、投机（第43课）、并行（第46课）、语法（第48课）、KV 传输（第45课）都是同一条缝的化身：核心只认接口，不认实现。",
                    "en": "The core's stability is the precondition for everything else being swappable. The attention backend (L33), platform (L42), quantization (L35), speculation (L43), parallelism (L46), grammar (L48), and KV transfer (L45) are all incarnations of the same seam: the core knows only the interface, never the implementation.",
                },
            },
            {
                "q": {
                    "zh": "为什么这种「面向接口」的结构让 SGLang 易于贡献？",
                    "en": "Why does this 'program-to-an-interface' structure make SGLang easy to contribute to?",
                },
                "opts": [
                    {
                        "zh": "<strong>因为加入新硬件或新算法通常只需实现一个已存在的接口，再在部署时选上，核心代码一行不改、风险被圈在新实现内</strong>",
                        "en": "<strong>Because adding new hardware or an algorithm usually only requires implementing an existing interface and selecting it at deploy time—no core code changes and risk is contained inside the new impl</strong>",
                    },
                    {"zh": "因为贡献者每次都要重写调度器与内存池", "en": "Because contributors must rewrite the scheduler and memory pools each time"},
                    {"zh": "因为所有实现共享同一份全局可变状态", "en": "Because all implementations share one global mutable state"},
                    {"zh": "因为接口允许跳过验证以加快合并", "en": "Because interfaces let you skip verification to merge faster"},
                ],
                "answer": 0,
                "why": {
                    "zh": "接口是契约，核心是稳定的舞台，社区在边缘并行扩张。贡献通常不是重写引擎，而是沿着已有的缝补上一个新实现——这正是「写一个模型」（第26课）也只是面向稳定层 API 编程的原因。",
                    "en": "The interface is the contract, the core is the stable stage, and the community expands in parallel at the edge. Contributing usually isn't rewriting the engine but filling a new implementation along an existing seam—which is why 'writing a model' (L26) is also just programming against stable layer APIs.",
                },
            },
        ],
        "open": [
            {
                "zh": "用你自己的话解释「面向接口编程，部署时选实现」这个范型：以注意力后端（第33课）为例，说明模型代码为什么只调用 self.attn(...) 而不在意背后是 FlashInfer、Triton 还是 FA，以及这种解耦带来了什么好处。",
                "en": "In your own words, explain the 'program to an interface, choose at deploy time' archetype: using the attention backend (Lesson 33) as an example, explain why model code only calls self.attn(...) without caring whether FlashInfer, Triton, or FA sits behind it, and what this decoupling buys you.",
            },
            {
                "zh": "本课说看懂这个形状后，看任何一课都会先问「缝在哪、接口是什么、实现有几个、部署时怎么选」。请从第35、42、43、45、46、48 课中至少挑三处，说明它们如何共享同一形状，并解释为什么把变化关在可插拔边缘能让引擎核心保持简洁可靠。",
                "en": "The lesson says that once you see the shape, you read any lesson by asking 'where is the seam, what is the interface, how many implementations, how is one chosen at deploy time'. Pick at least three of Lessons 35, 42, 43, 45, 46, 48 and show how they share the same shape, then explain why fencing variation behind pluggable edges keeps the engine core simple and reliable.",
            },
        ],
    },
    "63-built-for-throughput.html": {
        "mcq": [
            {
                "q": {
                    "zh": "本课提炼出贯穿全书的「唯一北极星」是什么？",
                    "en": "What is the single 'north star' this lesson distills as running through the whole guide?",
                },
                "opts": [
                    {
                        "zh": "<strong>让 GPU 一直忙着做有用的 token 工作——最大化每秒真正花在有用计算上的 FLOPs 与显存带宽（即吞吐）</strong>",
                        "en": "<strong>Keep the GPU busy doing useful token work—maximize the FLOPs and memory bandwidth actually spent on useful compute per second (i.e. throughput)</strong>",
                    },
                    {"zh": "在任何代价下把单个请求的延迟压到最低", "en": "Minimize a single request's latency at any cost"},
                    {"zh": "尽可能减少代码行数以方便维护", "en": "Reduce lines of code as much as possible for maintainability"},
                    {"zh": "让每个组件都能独立部署在不同机器上", "en": "Let every component deploy independently on separate machines"},
                ],
                "answer": 0,
                "why": {
                    "zh": "这是收尾的综合课：SGLang 几乎每个设计都在为同一目标服务——让 GPU 一刻不空，且每一秒都在做真正要交付的有用计算。批处理、分页、前缀缓存、重叠调度、CUDA Graph、内核融合、量化并行都是这颗北极星的不同切面。",
                    "en": "This is the closing synthesis lesson: almost every SGLang design serves one goal—never let the GPU idle, and make every second do useful compute that will actually be delivered. Batching, paging, prefix caching, overlap scheduling, CUDA graphs, kernel fusion, quantization, and parallelism are all facets of that north star.",
                },
            },
            {
                "q": {
                    "zh": "为什么说「批处理」（第5课）是吞吐的地基？",
                    "en": "Why is 'batching' (Lesson 5) called the bedrock of throughput?",
                },
                "opts": [
                    {
                        "zh": "<strong>因为一次解码里最贵的动作是把整套模型权重从显存读入计算单元，其成本几乎与服务的请求数无关；攒大批能把这次昂贵的读取摊薄到许多请求头上</strong>",
                        "en": "<strong>Because the most expensive act in a decode step is reading the whole model weights from memory into the compute units, a cost nearly independent of how many requests are served; a big batch amortizes that one read across many requests</strong>",
                    },
                    {"zh": "因为批处理能让每个请求都用上独立的 GPU", "en": "Because batching gives each request its own dedicated GPU"},
                    {"zh": "因为批处理避免了对显存的任何访问", "en": "Because batching avoids any memory access at all"},
                    {"zh": "因为批处理把模型权重永久缓存在 CPU 里", "en": "Because batching permanently caches the weights in CPU memory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "权重读取成本固定，与批内请求数几乎无关。让同一次读取服务 64 个请求而非 1 个，有效吞吐就接近翻 64 倍。这也是为什么分页（第6、30课）、量化与并行（第35、46课）都在为「更大的批」腾空间。",
                    "en": "The weight-read cost is fixed and nearly independent of batch size. Making one read serve 64 requests instead of 1 lifts effective throughput nearly 64×. That is also why paging (Lessons 6, 30) and quantization/parallelism (Lessons 35, 46) all free room for a bigger batch.",
                },
            },
            {
                "q": {
                    "zh": "下列哪一组属于「延迟护栏」——即在最大化系统吞吐的同时，避免牺牲单个用户？",
                    "en": "Which pair are the 'latency safeguards'—features that avoid sacrificing the individual user while the system is maximized?",
                },
                "opts": [
                    {
                        "zh": "<strong>分块预填充（第22课）与投机解码（第43课）</strong>",
                        "en": "<strong>Chunked prefill (Lesson 22) and speculative decoding (Lesson 43)</strong>",
                    },
                    {"zh": "前缀缓存（第7课）与重叠调度（第21课）", "en": "Prefix caching (Lesson 7) and overlap scheduling (Lesson 21)"},
                    {"zh": "CUDA Graph（第27课）与内核融合（第41课）", "en": "CUDA Graphs (Lesson 27) and kernel fusion (Lesson 41)"},
                    {"zh": "分页（第6课）与量化（第35课）", "en": "Paging (Lesson 6) and quantization (Lesson 35)"},
                ],
                "answer": 0,
                "why": {
                    "zh": "分块预填充把长 prompt 切块、与解码交错，避免一个大请求霸占整步；投机解码用小模型猜、大模型验，一步多吐 token 而不增加权重读取。两者的精神都是「在最大化系统的同时不牺牲那个具体的人」，而非牺牲吞吐换延迟。其余各项都是纯粹的吞吐杠杆。",
                    "en": "Chunked prefill slices a long prompt and interleaves it with decode so one big request can't hog a whole step; speculative decoding guesses with a small model and verifies with the big one, emitting several tokens per step without extra weight reads. Both are about maximizing the system without sacrificing the specific human, not trading throughput for latency. The other options are pure throughput levers.",
                },
            },
        ],
        "open": [
            {
                "zh": "用「吞吐眼镜」重读 SGLang：任选三个机制（如批处理第5课、前缀缓存第7/29课、重叠调度第21/59课、CUDA Graph 第27课、内核融合第38/41课），分别说明它消灭的是「空转」「重复劳动」还是「装不下」这三种 GPU 时间浪费中的哪一种，以及它如何最终服务于「每秒有用 token 工作」这颗北极星。",
                "en": "Re-read SGLang through the 'throughput glasses': pick any three mechanisms (e.g. batching L5, prefix caching L7/L29, overlap scheduling L21/L59, CUDA graphs L27, kernel fusion L38/L41) and, for each, say which of the three GPU-time wastes—idling, redundant work, or not-fitting—it eliminates, and how it ultimately serves the north star of 'useful token work per second'.",
            },
            {
                "zh": "本课说调度（SchedulePolicy / CacheAwarePolicy）同时追求「大批」与「高前缀命中」，并称二者都服务吞吐而非互相打架。请解释为什么这两个目标是同一颗北极星的两个分量，并以 LPM（最长前缀匹配）为例说明它如何同时帮助组成大批与集中缓存命中。",
                "en": "The lesson says scheduling (SchedulePolicy / CacheAwarePolicy) pursues both 'big batch' and 'high prefix-hit', calling them both servants of throughput rather than rivals. Explain why these two goals are two components of the same north star, and use LPM (longest-prefix-match) as an example to show how it simultaneously helps form a big batch and concentrate cache hits.",
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
