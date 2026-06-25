"""Part 1 - The Big Picture. Lesson content for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.
"""

LESSON_01 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
SGLang 是一个<strong>面向大语言模型与多模态模型的高性能服务引擎（serving engine）</strong>：
模型已经训练好了，它要解决的是<strong>怎样把模型在线上又快又省地“跑起来对外服务”</strong>——
在成千上万条并发请求下保持<strong>低延迟、高吞吐</strong>地生成 token。它由<strong>两半</strong>组成：
一半是让你像写程序一样编排 LLM 调用的<strong>前端语言（frontend DSL）</strong>，
另一半是真正把请求执行出来的<strong>运行时引擎（runtime engine）</strong>。
本课先从一万米高空看清它的轮廓，后面的课再一层层钻进引擎内部。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 SGLang 想成一间<strong>火爆的餐厅后厨</strong>。如果每来一位客人就单独起一口锅、从头备一遍料，厨房很快就瘫了。
  聪明的后厨做两件事：第一，<strong>把同时到达的订单合并成“一批”一起下锅</strong>（连续批处理，continuous batching）——
  灶台（GPU）一直满负荷转，不为等某一单而空转；第二，<strong>提前备好的高汤、酱底可以反复复用</strong>（前缀缓存，prefix cache）——
  很多订单开头都是同一份“底料”，备一次就够，后面的单直接取用。SGLang 做的正是这两件事：<strong>边批边复用</strong>，
  让同一张昂贵的 GPU 服务尽可能多的请求。
</div>

<h2>它到底解决什么问题</h2>
<p>训练一个大模型很难，但训练完之后，真正面向用户的环节是<strong>推理服务（inference serving）</strong>：
把模型权重加载进显存，接收用户的提示词，一个 token 接一个 token 地生成回答，再把结果返回。
听起来简单，难点全在<strong>规模与效率</strong>上——一张 H100 GPU 很贵，而生成是<strong>逐 token、自回归</strong>的串行过程，
天然“算得慢、等得久”。如果实现得笨拙，GPU 大量时间在<strong>空转</strong>：要么在等 CPU 准备下一批活，
要么在为重复的前缀一遍遍地重算注意力。服务引擎的全部价值，就是<strong>把这些浪费榨干</strong>。</p>

<p>SGLang 的答案可以浓缩成三招：<strong>连续批处理</strong>让 GPU 始终满载、<strong>RadixAttention 前缀缓存</strong>让共享开头只算一次、
<strong>零开销重叠调度器</strong>把 CPU 的调度开销藏进 GPU 的计算时间里。正因如此，它能以<strong>更低的成本</strong>
扛住<strong>更高的并发</strong>，每天在生产环境里生成<strong>万亿级</strong>的 token。它也不挑硬件：从<strong>单张 GPU</strong> 到
<strong>大型集群</strong>，覆盖 NVIDIA、AMD、Google TPU、Ascend NPU 乃至 CPU。</p>

<p>为什么“服务”值得专门做一个引擎？因为推理的成本结构和训练完全不同。训练是<strong>一次性</strong>的离线大计算，
而服务是<strong>永远在线</strong>的：只要产品还在用，请求就源源不断地来，每一毫秒延迟、每一分 GPU 利用率，
都会被乘以巨大的请求量放大成真金白银的成本。于是“把一张卡用到极致”不再是锦上添花，而是<strong>能不能盈利</strong>的分水岭。
SGLang 之所以被 xAI、Cursor、各大云厂商等广泛采用、在超过 <strong>40 万张 GPU</strong> 上运行，靠的正是它在这个“在线、并发、逐 token”
的苛刻场景里把效率做到了行业标杆——它已经成为开源推理引擎事实上的标准之一。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>SGLang = 前端语言（怎么“表达”一段 LLM 程序）+ 运行时引擎（怎么“执行”得又快又省）</strong>。
  前端 <span class="mono">python/sglang/lang/</span> 提供 <span class="inline">gen</span> / <span class="inline">fork</span> /
  <span class="inline">join</span> 等原语，让多步推理、并行分支、结构化输出写起来像普通代码；
  运行时 <span class="mono">python/sglang/srt/</span>（srt = SGLang RunTime）才是真正的引擎，
  把每条请求送过<strong>分词 → 调度 → 模型前向 → 采样 → 反分词</strong>这条流水线。它快，是因为
  <strong>RadixAttention</strong>、<strong>零开销重叠调度器</strong>与<strong>连续批处理</strong>三件法宝。
</div>

<h2>两半各管什么：前端 DSL 对运行时引擎</h2>
<p>初学者最容易混淆的，是“SGLang 这个名字到底指哪一半”。其实它<strong>同时指两半</strong>，而且分工非常清晰：
前端负责<strong>“怎么说”</strong>——把一段可能包含多次模型调用、分支与并行的复杂逻辑，用简洁的原语表达出来；
运行时负责<strong>“怎么做”</strong>——把这些请求在 GPU 上高效地执行出来。前端是给写应用的人用的便利层，
运行时才是这套教程的主角，也是性能的来源。</p>

<div class="cols">
  <div class="col"><h4>前端语言 DSL（lang/）</h4><p>让你像写程序一样编排 LLM：<span class="inline">gen</span> 生成一段、
  <span class="inline">fork</span> 分出并行分支、<span class="inline">join</span> 合并结果，还支持<strong>结构化输出</strong>与多轮控制。
  它把“多步、并行、带约束”的提示逻辑变得可读可复用。问的是<strong>“这段 LLM 程序怎么写”</strong>。</p></div>
  <div class="col"><h4>运行时引擎（srt/）</h4><p>真正的服务引擎：接收请求，做<strong>分词、调度、模型前向、采样、反分词</strong>，
  内置 RadixAttention、连续批处理、分页注意力、张量/流水线/专家并行等。问的是<strong>“怎么把请求执行得又快又省”</strong>。
  本教程从第 2 课起，主要钻进这一半。</p></div>
</div>

<p>记住这条主线：<strong>前端“表达”，后端“执行”</strong>。你完全可以只用运行时（直接发 HTTP 请求或用 <span class="inline">Engine</span> 类），
而不写一行前端 DSL；但两者配合时，前端的并行原语能让运行时<strong>更容易发现可批处理、可缓存的机会</strong>。</p>

<p>举个直觉例子：假设你要让模型<strong>从三个角度</strong>分别点评同一段文字，再汇总成一段结论。用普通代码，你得发三次串行请求、
自己拼接；而用前端 DSL，你可以 <span class="inline">fork</span> 出三条并行分支各写一段 <span class="inline">gen</span>，
最后 <span class="inline">join</span> 起来。关键在于——这三条分支<strong>共享同一段开头</strong>（同样的原文与系统提示），
运行时一眼就能看出“这段前缀只需算一次”，于是 RadixAttention 直接命中缓存。<strong>前端把并行结构显式地“说”出来，
后端就更容易把活儿“批着做、缓存着做”</strong>，这就是两半配合的价值。</p>

<h2>运行时的分层：从一行调用到 GPU 内核</h2>
<p>把运行时引擎竖着剖开，你会看到清晰的<strong>分层</strong>。最上面是你写的一行 <span class="inline">generate</span>，
最下面是贴着硬件跑的<strong>算子内核</strong>；中间每一层都为“又快又省”服务。这张分层图是后续所有课程的总纲——
之后每一课基本上都是在放大其中的某一层。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">顶层</span><span class="name">前端 DSL / API</span></div><div class="ld">你写的 <span class="mono">gen/fork/join</span> 或一行 <span class="mono">llm.generate(...)</span>，也可走 OpenAI 兼容的 HTTP 接口。</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">① 接入</span><span class="name">TokenizerManager</span></div><div class="ld">服务入口：把文本提示<strong>分词</strong>成 token id，校验参数，再把请求交给调度器。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">② 大脑</span><span class="name">Scheduler</span></div><div class="ld">引擎的“大脑”：决定<strong>谁进哪一批</strong>、何时前向，管理 KV 缓存与 RadixAttention，跑零开销重叠调度。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">③ 执行</span><span class="name">ModelRunner</span></div><div class="ld">真正在 GPU 上做<strong>模型前向</strong>：把一批 token 喂进模型，算出下一个 token 的概率分布（logits）。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">④ 采样</span><span class="name">Sampler</span></div><div class="ld">按温度 / top-p / top-k 等策略，从概率分布里<strong>采出下一个 token</strong>；结构化输出也在这里约束。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">出口</span><span class="name">DetokenizerManager</span></div><div class="ld">把生成的 token id <strong>反分词</strong>成人类可读的文本，流式返回给用户。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">内核</span><span class="name">Attention / 算子内核</span></div><div class="ld">垫在最底层的高性能算子：RadixAttention、分页注意力、量化与各家硬件后端（FlashInfer 等）。</div></div>
</div>

<p>这张图也解释了 SGLang 为什么能跨硬件：上面几层是<strong>与硬件无关</strong>的调度与控制逻辑，
真正贴着芯片的只有最底下的算子内核。换一类硬件（AMD、TPU、NPU、CPU），主要是<strong>替换内核后端</strong>，
上层几乎不动——这正是“从单卡到大型集群、覆盖多种硬件”的工程底气（第 42 课展开硬件后端）。</p>

<p>顺带厘清几个常被混淆的角色：<strong>TokenizerManager 与 DetokenizerManager</strong> 是一进一出的“翻译官”，
负责文本与 token id 之间的互转；<strong>Scheduler</strong> 是决策中枢，只<strong>决定</strong>怎么组批、怎么管缓存，但<strong>不亲手算</strong>；
真正在 GPU 上做矩阵乘法的是 <strong>ModelRunner</strong>；而 <strong>Sampler</strong> 负责“从概率里选字”。
把“谁决策、谁执行、谁翻译”分清楚，后面看任何一条链路都不会迷路。</p>

<h2>一次请求的一生：五步流水线</h2>
<p>把上面的分层“横过来看”，就是<strong>一条请求从进到出的流水线</strong>。这里只建立直觉，第 3 课会带你逐步放大每一站，
看一条 <span class="inline">generate</span> 请求究竟经历了什么。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>分词 Tokenize</h4><p class="mono">TokenizerManager</p><p>把用户的文本提示切成 token id 序列，做参数校验，登记为一个待处理请求。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>调度 Schedule</h4><p class="mono">Scheduler</p><p>把多条请求<strong>合并成一批</strong>，复用共享前缀的 KV 缓存，决定本步前向哪些 token。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>前向 Forward</h4><p class="mono">ModelRunner</p><p>在 GPU 上跑一次模型前向，得到每条请求“下一个 token”的概率分布（logits）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>采样 Sample</h4><p class="mono">Sampler</p><p>按采样策略选出下一个 token；未结束的请求把新 token 接回去，回到第 2 步继续。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>反分词 Detokenize</h4><p class="mono">DetokenizerManager</p><p>把 token id 转回文本，<strong>流式</strong>地一段段返回给用户，直到生成结束。</p></div></div>
</div>

<p>注意第 2~4 步会<strong>循环很多次</strong>：自回归生成是“算一个 token、接回去、再算下一个”的反复过程。
连续批处理的精妙之处在于——<strong>每一步都能让新请求“插队”进当前批次</strong>，已结束的请求即时让出位置，
GPU 因此几乎不空转。下面这张极简流程图，是同一件事的“一眼版”：</p>

<p>这里还藏着一个常被忽略的细节：生成分两个阶段。第一步把整段提示一次性喂进去、把它的 KV 缓存填好，叫
<strong>预填充（prefill）</strong>；之后一个个往外蹦 token，每次只算最新的那一个，叫<strong>解码（decode）</strong>。
预填充是“计算密集”的，解码是“访存密集”的，两者特性不同，调度策略也不同——SGLang 甚至支持把它们
<strong>拆到不同机器</strong>上做（prefill-decode 分离）。这些都会在后面的链路课里展开，这里只需先记住：
<strong>一条请求 = 一次预填充 + 很多次解码</strong>。</p>

<div class="flow">
  <div class="node hl"><div class="nt">请求</div><div class="nd">文本提示</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">分词</div><div class="nd">TokenizerManager</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">调度</div><div class="nd">Scheduler · 批处理</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">前向+采样</div><div class="nd">ModelRunner · Sampler</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">反分词</div><div class="nd">DetokenizerManager</div></div>
</div>

<h2>为什么快：三件法宝对应的瓶颈</h2>
<p>“高性能”不是口号，而是针对推理服务里<strong>三个具体瓶颈</strong>各下一剂猛药。下表把每件法宝和它要消灭的浪费一一对上，
每一项都会在后面的专门课程里展开：</p>

<table class="t">
  <tr><th>核心特性</th><th>它解决的瓶颈</th><th>一句话原理</th><th>展开</th></tr>
  <tr><td><strong>RadixAttention</strong></td><td class="mono">重复前缀被反复重算</td><td>用基数树组织共享前缀的 KV 缓存，相同开头只算一次</td><td>第 7 课</td></tr>
  <tr><td><strong>零开销重叠调度器</strong></td><td class="mono">GPU 等 CPU 而空转</td><td>GPU 算当前批时，CPU 已在准备下一批，两者重叠</td><td>第 21 课</td></tr>
  <tr><td><strong>连续批处理</strong></td><td class="mono">批内有空位、GPU 利用率低</td><td>每步都让新请求插队、已结束的让位，批次始终“满”</td><td>第 5 课</td></tr>
  <tr><td><strong>分页注意力 / 量化</strong></td><td class="mono">显存碎片与带宽吃紧</td><td>分页管理 KV 显存、低精度（FP8/INT4）压缩计算与存储</td><td>第 6、35 课</td></tr>
</table>

<p>这四项叠在一起，就是 SGLang “又快又省”的来源：<strong>少算（前缀复用）、不闲（重叠 + 连续批处理）、少占（分页 + 量化）</strong>。
它们不是孤立技巧，而是围绕同一目标——<strong>把每一张昂贵 GPU 的每一刻都用在刀刃上</strong>——彼此咬合的设计。</p>

<p>值得强调的是，这些优化对<strong>使用者基本透明</strong>：你写的还是那一行 <span class="inline">generate</span>，不需要手动管理批次、
缓存或显存。引擎在背后<strong>自动</strong>把可合并的请求合批、把可复用的前缀缓存、把可重叠的工作错峰。这正是“引擎”二字的含义——
<strong>把复杂的性能工程封装起来，让上层只管表达意图</strong>。后面的每一课，本质上都是揭开某一层的盖子，看这份“自动”背后到底做了什么。</p>

<h2>亲手摸一下：最小的 Engine 用法</h2>
<p>抛开 HTTP 服务器，SGLang 还提供一个直接可用的 <span class="inline">Engine</span> 类，几行代码就能离线批量推理。
注意这里展示的是<strong>用法</strong>；这门课真正要做的，是带你钻进 <span class="inline">Engine</span> 背后，看这几行调用在引擎里到底发生了什么。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::Engine</span><span class="ln">离线批量推理</span></div>
  <pre><span class="kw">import</span> sglang <span class="kw">as</span> sgl

llm = sgl.Engine(model_path=<span class="st">"qwen/qwen2.5-0.5b-instruct"</span>)   <span class="cm"># 加载模型、拉起子进程</span>

prompts = [<span class="st">"Hello, my name is"</span>, <span class="st">"The capital of France is"</span>]
params  = {<span class="st">"temperature"</span>: <span class="nb">0.8</span>, <span class="st">"top_p"</span>: <span class="nb">0.95</span>}

outputs = llm.generate(prompts, params)               <span class="cm"># 一次喂一批，引擎自动批处理</span>
<span class="kw">for</span> p, o <span class="kw">in</span> zip(prompts, outputs):
    print(p, o[<span class="st">"text"</span>])

llm.shutdown()</pre>
</div>

<p><span class="inline">Engine</span> 的源码注释把它的内部讲得很清楚：一个引擎由三大组件构成——<strong>TokenizerManager</strong>（分词、把请求发给调度器）、
子进程里的 <strong>Scheduler</strong>（接收请求、组批、前向、把输出 token 发给反分词器）、以及子进程里的 <strong>DetokenizerManager</strong>（反分词、把结果发回）。
进程之间通过 <strong>ZMQ 的 IPC</strong> 通信。这正是上面分层图与流水线图在代码里的对应。线上服务则由
<span class="mono">python/sglang/launch_server.py</span> 启动一个 HTTP 服务器把这套引擎包起来，对外暴露 OpenAI 兼容接口与原生 <span class="mono">/generate</span> 端点。</p>

<p>为什么要拆成<strong>多个子进程</strong>、用 IPC 通信，而不是一个大循环？因为分词、反分词是 CPU 上的字符串处理，模型前向是 GPU 上的重计算，
把它们放进<strong>各自独立的进程</strong>，就能让 CPU 的活儿和 GPU 的活儿真正<strong>并行流水</strong>起来，互不阻塞——这又回到了“零开销重叠”的思想。
你现在不必记住每个组件的细节，只要建立这张<strong>心智地图</strong>：一行 <span class="inline">generate</span> 落下去，背后是一支分工明确的小队
在协同流水。带着这张图，我们下一课就<strong>让它动起来</strong>，跟着一条真实请求走一遍“一次请求的一生”。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>定位</strong>：SGLang 是<strong>推理服务引擎</strong>，解决“训练好的模型怎样又快又省地上线服务”，不做训练、不是数据库。</li>
    <li><strong>两半</strong>：前端 DSL（<span class="mono">lang/</span>，gen/fork/join 表达 LLM 程序）+ 运行时引擎（<span class="mono">srt/</span>，真正执行）。<strong>前端表达，后端执行</strong>。</li>
    <li><strong>流水线</strong>：一次请求经历<strong>分词 → 调度 → 前向 → 采样 → 反分词</strong>五步，中间三步循环到生成结束。</li>
    <li><strong>为何快</strong>：RadixAttention（前缀复用）+ 零开销重叠调度器（不空转）+ 连续批处理（批次常满）+ 分页/量化（省显存）。</li>
    <li><strong>跨硬件</strong>：上层与硬件无关，仅最底层算子内核随硬件更换——从单卡到大型集群，覆盖 NVIDIA/AMD/TPU/NPU/CPU。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
SGLang is a <strong>high-performance serving engine for large language models and multimodal models</strong>:
the model is already trained, and what SGLang solves is <strong>how to run it online fast and cheaply</strong> —
generating tokens at <strong>low latency and high throughput</strong> under thousands of concurrent requests.
It comes in <strong>two halves</strong>: a <strong>frontend language (DSL)</strong> that lets you orchestrate LLM calls
like a program, and a <strong>runtime engine</strong> that actually executes the requests.
This lesson sketches its outline from 10,000 feet; later lessons zoom into the engine layer by layer.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture SGLang as a <strong>busy restaurant kitchen</strong>. If you fired up a fresh pan and prepped every
  ingredient from scratch for each guest, the kitchen would collapse. A smart kitchen does two things: first,
  it <strong>merges orders that arrive together into one batch</strong> on the stove (continuous batching) —
  the burner (GPU) stays fully loaded instead of idling for one order; second, it <strong>reuses stock and base
  sauces prepped ahead of time</strong> (the prefix cache) — many orders start from the same base, so prep it once
  and later orders just take it. SGLang does exactly this: <strong>batch and reuse</strong>, so one expensive GPU
  serves as many requests as possible.
</div>

<h2>What problem does it actually solve</h2>
<p>Training a big model is hard, but once trained, the user-facing phase is <strong>inference serving</strong>:
load the weights into GPU memory, take a user prompt, generate the answer one token at a time, and return it.
That sounds simple, but the difficulty is all about <strong>scale and efficiency</strong> — an H100 GPU is expensive,
and generation is an inherently serial, <strong>autoregressive, token-by-token</strong> process that is "slow and waity"
by nature. Done clumsily, the GPU spends huge stretches <strong>idle</strong>: either waiting for the CPU to prepare the
next batch, or recomputing attention over repeated prefixes again and again. The whole value of a serving engine is to
<strong>squeeze out that waste</strong>.</p>

<p>SGLang's answer boils down to three moves: <strong>continuous batching</strong> keeps the GPU fully loaded,
<strong>RadixAttention prefix caching</strong> computes a shared opening only once, and the
<strong>zero-overhead overlap scheduler</strong> hides CPU scheduling cost behind GPU compute time. As a result it can
sustain <strong>higher concurrency at lower cost</strong>, generating <strong>trillions</strong> of tokens a day in
production. It is also hardware-agnostic: from a <strong>single GPU</strong> to <strong>large clusters</strong>, across
NVIDIA, AMD, Google TPU, Ascend NPU, and even CPUs.</p>

<p>Why does "serving" deserve a dedicated engine? Because inference has a very different cost structure from training.
Training is a <strong>one-off</strong> offline mega-computation, whereas serving is <strong>always on</strong>: as long as the
product is in use, requests keep streaming in, and every millisecond of latency and every percent of GPU utilization gets
multiplied by enormous request volume into real money. So "use one card to the absolute limit" is no longer a nice-to-have
but the dividing line of <strong>whether you can be profitable</strong>. SGLang is widely adopted by xAI, Cursor, and major
cloud providers, running on more than <strong>400,000 GPUs</strong>, precisely because it pushes efficiency to an industry
benchmark in this demanding "online, concurrent, token-by-token" setting — it has become one of the de facto standards
among open-source inference engines.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  In one line: <strong>SGLang = a frontend language (how you "express" an LLM program) + a runtime engine
  (how it "executes" fast and cheaply)</strong>. The frontend <span class="mono">python/sglang/lang/</span> offers
  <span class="inline">gen</span> / <span class="inline">fork</span> / <span class="inline">join</span> primitives so multi-step
  reasoning, parallel branches and structured output read like ordinary code; the runtime
  <span class="mono">python/sglang/srt/</span> (srt = SGLang RunTime) is the real engine, pushing each request through
  <strong>tokenize → schedule → model forward → sample → detokenize</strong>. It is fast thanks to three weapons:
  <strong>RadixAttention</strong>, the <strong>zero-overhead overlap scheduler</strong>, and <strong>continuous batching</strong>.
</div>

<h2>What each half owns: frontend DSL vs runtime engine</h2>
<p>Beginners most often ask "which half does the name SGLang refer to?" The answer is <strong>both</strong>, with a very
clean split: the frontend handles <strong>"how you say it"</strong> — expressing complex logic that may include multiple
model calls, branches and parallelism through compact primitives; the runtime handles <strong>"how it gets done"</strong> —
executing those requests efficiently on the GPU. The frontend is a convenience layer for app authors; the runtime is the
star of this guide and the source of the performance.</p>

<div class="cols">
  <div class="col"><h4>Frontend language DSL (lang/)</h4><p>Orchestrate LLMs like a program: <span class="inline">gen</span>
  generates a span, <span class="inline">fork</span> spawns parallel branches, <span class="inline">join</span> merges
  results, plus <strong>structured output</strong> and multi-turn control. It makes "multi-step, parallel, constrained"
  prompt logic readable and reusable. It asks <strong>"how do I write this LLM program?"</strong></p></div>
  <div class="col"><h4>Runtime engine (srt/)</h4><p>The real serving engine: it receives requests and does
  <strong>tokenize, schedule, model forward, sample, detokenize</strong>, with RadixAttention, continuous batching,
  paged attention, and tensor/pipeline/expert parallelism built in. It asks <strong>"how do I execute requests fast and
  cheaply?"</strong> From Lesson 2 on, this guide mostly digs into this half.</p></div>
</div>

<p>Keep this throughline: <strong>the frontend "expresses", the backend "executes"</strong>. You can use the runtime alone
(hit the HTTP endpoint, or use the <span class="inline">Engine</span> class) without writing a line of frontend DSL; but
when combined, the frontend's parallel primitives make it <strong>easier for the runtime to spot batchable, cacheable
opportunities</strong>.</p>

<p>A quick intuition: suppose you want the model to critique the same passage <strong>from three angles</strong> and then
summarize. In plain code you would fire three serial requests and stitch them yourself; with the frontend DSL you
<span class="inline">fork</span> three parallel branches, each writing a <span class="inline">gen</span>, then
<span class="inline">join</span> them. The key is that all three branches <strong>share the same opening</strong> (the same
source text and system prompt), so the runtime can instantly see "this prefix only needs to be computed once" and
RadixAttention hits the cache. <strong>The frontend says the parallel structure explicitly, and the backend can then
batch and cache the work</strong> — that is the value of the two halves working together.</p>

<h2>The runtime in layers: from one call down to GPU kernels</h2>
<p>Slice the runtime engine vertically and you see clean <strong>layers</strong>. At the top is the single
<span class="inline">generate</span> you wrote; at the bottom are <strong>operator kernels</strong> hugging the hardware;
every layer in between exists to serve "fast and cheap". This layer diagram is the master map for the whole guide —
nearly every later lesson zooms into one of these layers.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Top</span><span class="name">Frontend DSL / API</span></div><div class="ld">Your <span class="mono">gen/fork/join</span> or a single <span class="mono">llm.generate(...)</span>, or an OpenAI-compatible HTTP call.</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">① Ingress</span><span class="name">TokenizerManager</span></div><div class="ld">The service entry: <strong>tokenizes</strong> the prompt into token ids, validates params, hands the request to the scheduler.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">② Brain</span><span class="name">Scheduler</span></div><div class="ld">The engine's "brain": decides <strong>who joins which batch</strong> and when to forward, manages KV cache and RadixAttention, runs the zero-overhead overlap schedule.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">③ Execute</span><span class="name">ModelRunner</span></div><div class="ld">Runs the <strong>model forward</strong> on the GPU: feeds a batch of tokens through the model to produce next-token logits.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">④ Sample</span><span class="name">Sampler</span></div><div class="ld">Picks the <strong>next token</strong> from the distribution by temperature / top-p / top-k; structured output is constrained here too.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Egress</span><span class="name">DetokenizerManager</span></div><div class="ld"><strong>Detokenizes</strong> generated token ids back into human-readable text, streamed to the user.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Kernels</span><span class="name">Attention / operator kernels</span></div><div class="ld">High-performance ops at the bottom: RadixAttention, paged attention, quantization, and per-vendor backends (FlashInfer, etc.).</div></div>
</div>

<p>This diagram also explains why SGLang spans hardware: the upper layers are <strong>hardware-agnostic</strong> scheduling
and control logic; only the bottom operator kernels truly touch the chip. Switching hardware (AMD, TPU, NPU, CPU) mostly
means <strong>swapping the kernel backend</strong> while the upper layers barely change — the engineering basis for "single
GPU to large clusters across many hardware" (kernel backends in Lesson 42).</p>

<p>It helps to disambiguate a few often-confused roles: <strong>TokenizerManager and DetokenizerManager</strong> are the
"translators" on the way in and out, converting between text and token ids; the <strong>Scheduler</strong> is the
decision center that only <strong>decides</strong> how to batch and manage cache but does <strong>not compute itself</strong>;
the one doing the matrix multiplies on the GPU is <strong>ModelRunner</strong>; and the <strong>Sampler</strong> "picks the
word from the probabilities". Once you separate "who decides, who executes, who translates", no later path will confuse you.</p>

<h2>Life of a request: a five-step pipeline</h2>
<p>Turn the layer diagram sideways and you get <strong>a request's pipeline from in to out</strong>. We only build intuition
here; Lesson 3 zooms into each station to see what a <span class="inline">generate</span> request actually goes through.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Tokenize</h4><p class="mono">TokenizerManager</p><p>Cut the user's text prompt into a token-id sequence, validate params, register it as a pending request.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Schedule</h4><p class="mono">Scheduler</p><p>Merge multiple requests into <strong>one batch</strong>, reuse shared-prefix KV cache, decide which tokens to forward this step.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Forward</h4><p class="mono">ModelRunner</p><p>Run one model forward on the GPU to get each request's "next token" probability distribution (logits).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Sample</h4><p class="mono">Sampler</p><p>Pick the next token by the sampling policy; unfinished requests append the new token and loop back to step 2.</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>Detokenize</h4><p class="mono">DetokenizerManager</p><p>Turn token ids back into text and <strong>stream</strong> it back piece by piece until generation ends.</p></div></div>
</div>

<p>Note that steps 2-4 <strong>loop many times</strong>: autoregressive generation is a repeated "compute one token, append
it, compute the next". The elegance of continuous batching is that <strong>each step lets new requests "cut in" to the
current batch</strong> while finished ones immediately yield their slot, so the GPU almost never idles. The minimal flow
chart below is the "at-a-glance" version of the same thing:</p>

<p>There is also an often-missed detail: generation has two phases. The first step feeds the whole prompt in at once and
fills its KV cache — called <strong>prefill</strong>; afterwards tokens pop out one by one, each step computing only the
newest one — called <strong>decode</strong>. Prefill is compute-bound, decode is memory-bound; their characteristics differ,
and so do the scheduling strategies — SGLang can even split them across <strong>different machines</strong> (prefill-decode
disaggregation). All of this is expanded in the later path lessons; here just remember: <strong>one request = one prefill +
many decodes</strong>.</p>

<div class="flow">
  <div class="node hl"><div class="nt">Request</div><div class="nd">text prompt</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Tokenize</div><div class="nd">TokenizerManager</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Schedule</div><div class="nd">Scheduler · batching</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Forward+Sample</div><div class="nd">ModelRunner · Sampler</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Detokenize</div><div class="nd">DetokenizerManager</div></div>
</div>

<h2>Why it is fast: three weapons against three bottlenecks</h2>
<p>"High performance" is not a slogan; it is one strong remedy aimed at each of <strong>three concrete bottlenecks</strong>
in inference serving. The table pairs each weapon with the waste it kills, and each item is expanded in a dedicated later
lesson:</p>

<table class="t">
  <tr><th>Core feature</th><th>Bottleneck it solves</th><th>One-line principle</th><th>Detail</th></tr>
  <tr><td><strong>RadixAttention</strong></td><td class="mono">repeated prefixes recomputed</td><td>organize shared-prefix KV cache in a radix tree; a shared opening is computed once</td><td>Lesson 7</td></tr>
  <tr><td><strong>Zero-overhead overlap scheduler</strong></td><td class="mono">GPU idles waiting on CPU</td><td>while the GPU runs the current batch, the CPU is already preparing the next; the two overlap</td><td>Lesson 21</td></tr>
  <tr><td><strong>Continuous batching</strong></td><td class="mono">empty slots, low GPU utilization</td><td>each step lets new requests cut in and finished ones yield, so the batch stays "full"</td><td>Lesson 5</td></tr>
  <tr><td><strong>Paged attention / quantization</strong></td><td class="mono">memory fragmentation &amp; bandwidth</td><td>page-manage KV memory; low precision (FP8/INT4) compresses compute and storage</td><td>Lessons 6 &amp; 35</td></tr>
</table>

<p>Stacked together, these four are where SGLang's "fast and cheap" comes from: <strong>compute less (prefix reuse), never
idle (overlap + continuous batching), occupy less (paging + quantization)</strong>. They are not isolated tricks but a set
of interlocking designs around one goal — <strong>spend every moment of every expensive GPU on what matters</strong>.</p>

<p>Crucially, these optimizations are <strong>largely transparent to the user</strong>: you still write that one
<span class="inline">generate</span> line and never hand-manage batches, caches, or memory. Behind the scenes the engine
<strong>automatically</strong> batches mergeable requests, caches reusable prefixes, and staggers overlappable work. That is
exactly what "engine" means — <strong>it encapsulates the hard performance engineering so the upper layer only expresses
intent</strong>. Every later lesson is essentially lifting the lid on one layer to see what that "automatic" really does.</p>

<h2>Hands-on: the smallest Engine usage</h2>
<p>Skipping the HTTP server, SGLang also exposes a directly usable <span class="inline">Engine</span> class — a few lines do
offline batch inference. Note this shows <strong>usage</strong>; what this guide really does is take you behind
<span class="inline">Engine</span> to see what these few calls actually trigger inside the engine.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::Engine</span><span class="ln">offline batch inference</span></div>
  <pre><span class="kw">import</span> sglang <span class="kw">as</span> sgl

llm = sgl.Engine(model_path=<span class="st">"qwen/qwen2.5-0.5b-instruct"</span>)   <span class="cm"># load model, spawn subprocesses</span>

prompts = [<span class="st">"Hello, my name is"</span>, <span class="st">"The capital of France is"</span>]
params  = {<span class="st">"temperature"</span>: <span class="nb">0.8</span>, <span class="st">"top_p"</span>: <span class="nb">0.95</span>}

outputs = llm.generate(prompts, params)               <span class="cm"># feed a batch; the engine batches automatically</span>
<span class="kw">for</span> p, o <span class="kw">in</span> zip(prompts, outputs):
    print(p, o[<span class="st">"text"</span>])

llm.shutdown()</pre>
</div>

<p>The <span class="inline">Engine</span> source comment spells out its internals clearly: an engine is built from three
components — <strong>TokenizerManager</strong> (tokenizes and sends requests to the scheduler), a <strong>Scheduler</strong>
in a subprocess (receives requests, forms batches, forwards them, sends output tokens to the detokenizer), and a
<strong>DetokenizerManager</strong> in a subprocess (detokenizes and sends results back). The processes talk over
<strong>ZMQ IPC</strong>. That is exactly the layer diagram and pipeline above, in code. A live deployment is started by
<span class="mono">python/sglang/launch_server.py</span>, which wraps this engine in an HTTP server exposing OpenAI-compatible
APIs and a native <span class="mono">/generate</span> endpoint.</p>

<p>Why split into <strong>multiple subprocesses</strong> talking over IPC, instead of one big loop? Because tokenize and
detokenize are CPU string processing while the model forward is heavy GPU compute; putting them in <strong>separate
processes</strong> lets the CPU work and the GPU work truly <strong>pipeline in parallel</strong> without blocking each other —
back to the "zero-overhead overlap" idea. You need not memorize every component yet; just form this <strong>mental
map</strong>: one <span class="inline">generate</span> call drops in, and behind it a small, clearly divided team pipelines the
work. With that map, the next lesson <strong>brings it to life</strong> by following a real request through its "life of a
request".</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Role</strong>: SGLang is an <strong>inference serving engine</strong> — it solves "how to put a trained model
    online fast and cheaply"; it does not train, and it is not a database.</li>
    <li><strong>Two halves</strong>: frontend DSL (<span class="mono">lang/</span>, gen/fork/join to express LLM programs) +
    runtime engine (<span class="mono">srt/</span>, the real executor). <strong>Frontend expresses, backend executes</strong>.</li>
    <li><strong>Pipeline</strong>: a request goes <strong>tokenize → schedule → forward → sample → detokenize</strong>; the
    middle three loop until generation ends.</li>
    <li><strong>Why fast</strong>: RadixAttention (prefix reuse) + zero-overhead overlap scheduler (no idling) + continuous
    batching (full batches) + paging/quantization (memory savings).</li>
    <li><strong>Cross-hardware</strong>: upper layers are hardware-agnostic; only the bottom operator kernels change per
    hardware — single GPU to large clusters across NVIDIA/AMD/TPU/NPU/CPU.</li>
  </ul>
</div>
""",
}

LESSON_02 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课我们从一万米高空看清了 SGLang 的轮廓：它是一台<strong>推理服务引擎</strong>，由<strong>前端语言</strong>和
<strong>运行时引擎</strong>两半组成。这一课我们落到地面，给整个代码仓库画一张<strong>全景地图</strong>：
代码住在哪两个大房间里、运行时这一半又切成哪些小科室、线上跑起来时这些代码<strong>分裂成几个进程</strong>、
进程之间又靠什么说话。读完这一课，后面每钻进任何一个目录，你都知道自己<strong>站在地图的哪个位置</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 SGLang 想成一栋<strong>运转中的公司办公楼</strong>。一楼<strong>前台（TokenizerManager）</strong>接待来访者，
  把口头需求登记成标准工单；二楼<strong>调度室（Scheduler）</strong>是大脑，决定哪些工单<strong>并成一批</strong>、先做谁、
  怎么排队，但调度长自己<strong>不动手干活</strong>；真正流水线上拧螺丝的，是车间里的<strong>技工（TpWorker / ModelRunner）</strong>，
  他们守着<strong>机房里昂贵的机器（GPU 与算子内核）</strong>；活干完，<strong>打包发货处（DetokenizerManager）</strong>
  把半成品翻译成客户看得懂的成品寄回去。关键是：<strong>前台、调度＋车间、发货处是三个独立的部门（进程）</strong>，
  靠<strong>内部邮件系统（ZMQ）</strong>传递工单，谁也不挡谁的路——这样前台接待和车间开工才能<strong>同时进行</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  整个仓库可以先粗暴地切成<strong>两半加一层底座</strong>：<span class="mono">python/sglang/lang/</span> 是<strong>前端 DSL</strong>
  （怎么“表达”一段 LLM 程序），<span class="mono">python/sglang/srt/</span> 是<strong>运行时引擎</strong>
  （srt = <strong>SGLang RunTime</strong>，怎么“执行”得又快又省）；垫在两者下面的，是
  <span class="mono">sgl-kernel/</span>（提前编译好的 C++/CUDA 算子）和 <span class="mono">python/sglang/jit_kernel/</span>
  （即时编译的轻量内核）。<strong>这门教程的主角是 srt/</strong>——它内部又分成十几个各管一摊的子包，是后面所有课程的地图底图。
</div>

<h2>代码仓库的两半：lang/ 与 srt/</h2>
<p>初学者打开仓库常被几十个目录吓到，其实<strong>第一刀只要切两半</strong>就清楚了：一半教你<strong>怎么把一段 LLM 程序写出来</strong>，
另一半负责<strong>把请求又快又省地跑出来</strong>。前者是给写应用的人用的便利层，后者才是性能的来源，也是我们要深挖的地方。
名字里的 <span class="inline">SGLang</span> 同时指这两半，但二者的代码量、复杂度、和与硬件的距离完全不同。</p>

<div class="cols">
  <div class="col"><h4>前端语言 DSL · <span class="mono">python/sglang/lang/</span></h4><p>提供 <span class="inline">gen</span>（生成一段）、
  <span class="inline">fork</span>（分出并行分支）、<span class="inline">join</span>（合并结果）等原语，外加 <span class="mono">system/user/assistant</span>
  角色与结构化输出。它把“多步、并行、带约束”的提示逻辑写得像普通程序。关键文件：<span class="mono">api.py</span>、<span class="mono">ir.py</span>、
  <span class="mono">interpreter.py</span>。它问的是<strong>“这段 LLM 程序怎么写”</strong>。</p></div>
  <div class="col"><h4>运行时引擎 · <span class="mono">python/sglang/srt/</span></h4><p>真正的服务引擎：接收请求，做<strong>分词 → 调度 →
  模型前向 → 采样 → 反分词</strong>，内置 RadixAttention、连续批处理、分页注意力、张量/流水线/专家并行。它问的是
  <strong>“怎么把请求执行得又快又省”</strong>。从这一课起，我们基本只在这一半里活动；它也是整个仓库里<strong>最厚的一摞代码</strong>。</p></div>
</div>

<p>记住这条分界线：<strong>lang/ 表达，srt/ 执行</strong>。你完全可以只用 srt/（直接发 HTTP 请求或用 <span class="inline">Engine</span> 类）
而一行前端都不写；反过来，前端把并行结构显式“说”出来，又能帮后端更容易发现可批处理、可缓存的机会。两半的配合，
正是第 1 课那个“三个角度点评同一段文字”的例子背后的道理。</p>

<p>为什么一个推理引擎要把“语言”和“运行时”绑在同一个仓库里？因为二者本是<strong>同一件事的两端</strong>：前端越能把
“哪些调用可以并行、哪些前缀彼此共享”表达清楚，后端就越有机会<strong>把这些结构兑现成真实的加速</strong>。很多别的框架只做运行时、
把前端交给用户硬拼，结果调度器拿到的只是一堆“看不出关系”的孤立请求，错失了批处理与缓存的良机。SGLang 把两端放在一起，
正是想让“表达”和“执行”之间的信息<strong>不在中途丢失</strong>。也正因如此，名字里的 <span class="inline">Lang</span> 并不只是“语言”，
更是“让运行时看得懂的那种语言”——这是它区别于一般推理服务器的设计哲学。不过对初学者来说，<strong>记住主角是 srt/ 就够了</strong>：
前端是锦上添花，运行时才是这台引擎真正的发动机舱，也是这门课接下来要逐层拆开的对象。</p>

<h2>运行时的包结构：从入口到内核</h2>
<p>把 <span class="mono">srt/</span> 竖着剖开，会看到一摞<strong>分层</strong>：最上面是接客的入口，往下是决策的大脑，再往下是真正在 GPU 上算的执行层，
最底下是贴着硬件的算子内核。这张分层图是整套教程的<strong>总纲</strong>——后面几乎每一课都是在放大其中的某一层。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">入口</span><span class="name">entrypoints/</span></div><div class="ld">对外的门面：<span class="mono">http_server.py</span>（OpenAI 兼容接口 + 原生 <span class="mono">/generate</span>）、<span class="mono">engine.py</span>（离线 <span class="mono">Engine</span> 类）、<span class="mono">launch_server.py</span>。第 13–17 课。</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">接入</span><span class="name">managers/ · 前后翻译</span></div><div class="ld"><span class="mono">tokenizer_manager.py</span> 分词、<span class="mono">detokenizer_manager.py</span> 反分词，把文本与 token id 互转，是引擎的“进出口”。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">大脑</span><span class="name">managers/ · Scheduler</span></div><div class="ld"><span class="mono">scheduler.py</span> 决定谁进哪一批、何时前向，管 KV 缓存与队列。第 18–23 课专讲调度。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">执行</span><span class="name">model_executor/ · model_loader/ · models/</span></div><div class="ld"><span class="mono">ModelRunner</span> 在 GPU 上做模型前向；<span class="mono">model_loader/</span> 加载权重，<span class="mono">models/</span> 放各模型结构。第 24–28 课。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">采样</span><span class="name">sampling/ · constrained/</span></div><div class="ld">从 logits 里按温度/top-p/top-k 选出下一个 token；<span class="mono">constrained/</span> 负责结构化输出的约束（JSON、正则等）。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">算子</span><span class="name">layers/ · mem_cache/</span></div><div class="ld"><span class="mono">layers/</span> 是注意力等算子层（第 33–37 课），<span class="mono">mem_cache/</span> 管 KV 缓存与 RadixAttention 的内存（第 29–32 课）。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">硬件</span><span class="name">sgl-kernel/ · jit_kernel/ · distributed/</span></div><div class="ld">贴着芯片的高性能内核（AOT 的 <span class="mono">sgl-kernel</span> 与即时编译的 <span class="mono">jit_kernel</span>）与多卡通信。第 38–42 课。</div></div>
</div>

<p>这张图也解释了 SGLang 为什么能跨硬件：上面几层是<strong>与硬件无关</strong>的调度与控制逻辑，真正贴着芯片的只有最底下的算子内核。
换一类硬件（AMD、TPU、NPU、CPU），主要是<strong>替换内核后端</strong>，上层几乎不动。下面这张表把运行时的主要子包、它各自负责什么、
以及会在<strong>哪一课</strong>展开，一一对上——它就是你日后“迷路时”的导航索引：</p>

<p>为什么运行时要切成<strong>这么多个子包</strong>，而不是揉成一大坨？因为每个子包都对应一个<strong>可以独立演进的关注点</strong>：
今天换一个更快的注意力内核，只动 <span class="mono">layers/</span>；明天支持一个新模型，只在 <span class="mono">models/</span> 加一个文件；
后天想换一种 KV 缓存策略，只碰 <span class="mono">mem_cache/</span>。这种<strong>高内聚、低耦合</strong>的切分，让一个上千人参与的开源项目
还能保持可维护——你改你的内核，我加我的模型，调度器那颗大脑基本不受影响。对学习者也一样友好：你<strong>不必一次读懂整台引擎</strong>，
完全可以挑一个子包深扎进去，而这门教程的部分划分，基本就是顺着这些子包的边界来安排的。</p>

<table class="t">
  <tr><th>子包</th><th>它负责什么</th><th>展开</th></tr>
  <tr><td class="mono">entrypoints/</td><td>对外入口：HTTP 服务器、<span class="inline">Engine</span> 类、启动脚本</td><td>第 13–17 课</td></tr>
  <tr><td class="mono">managers/</td><td>分词、反分词与 <strong>Scheduler 调度</strong>（引擎大脑）</td><td>第 18–23 课</td></tr>
  <tr><td class="mono">model_executor/</td><td><span class="inline">ModelRunner</span>：在 GPU 上组织一次模型前向</td><td>第 24–28 课</td></tr>
  <tr><td class="mono">model_loader/ · models/</td><td>加载权重、定义各模型（Llama、Qwen 等）的网络结构</td><td>第 24–28 课</td></tr>
  <tr><td class="mono">mem_cache/</td><td>KV 缓存与 <strong>RadixAttention</strong> 的内存管理、分页</td><td>第 29–32 课</td></tr>
  <tr><td class="mono">layers/</td><td>注意力、归一化、MoE 等算子层与各家后端</td><td>第 33–37 课</td></tr>
  <tr><td class="mono">distributed/</td><td>张量/流水线/专家并行的多卡通信</td><td>第 46 课</td></tr>
  <tr><td class="mono">speculative/ · lora/ · disaggregation/ · multimodal/</td><td>投机解码、LoRA、PD 分离、多模态等进阶专题</td><td>进阶部分</td></tr>
</table>

<h2>三进程模型：谁在哪个进程、用什么通信</h2>
<p>上面那摞分层是<strong>静态的代码结构</strong>；可一旦引擎<strong>跑起来</strong>，它就不是一个大循环，而是<strong>裂成好几个进程</strong>并行干活。
这是 SGLang 架构里最容易被忽略、却最关键的一点：<strong>分词、调度＋前向、反分词分别住在不同的进程里</strong>，靠 ZMQ 的 IPC 传消息。
为什么要这么拆？因为分词/反分词是 CPU 上的字符串处理，模型前向是 GPU 上的重计算，把它们放进各自独立的进程，
CPU 的活和 GPU 的活才能<strong>真正并行流水</strong>，互不阻塞——这正是“零开销重叠”的工程基础（第 21 课展开）。</p>

<div class="flow">
  <div class="node hl"><div class="nt">TokenizerManager</div><div class="nd">主进程 · 分词、收发请求</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Scheduler + TpWorker</div><div class="nd">子进程 · 每 TP rank 一个 · GPU 前向</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">DetokenizerManager</div><div class="nd">子进程 · 反分词</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">回到 TokenizerManager</div><div class="nd">主进程 · 流式返回</div></div>
</div>

<p>注意三者的角色千万别混：<strong>TokenizerManager 在主进程</strong>（HTTP 服务器、<span class="inline">Engine</span> 也都在主进程）；
<strong>Scheduler 在子进程</strong>，而且<strong>张量并行（TP）每一路 rank 各起一个 Scheduler 进程</strong>，每个都持有自己的
<span class="inline">TpWorker</span> 与 <span class="inline">ModelRunner</span>，真正占着一张 GPU 干前向；<strong>DetokenizerManager 又是另一个子进程</strong>。
请记牢：<strong>Scheduler 才是握着 GPU 的那个进程</strong>，TokenizerManager 与 DetokenizerManager 只是一进一出的“翻译官”，三者不可混为一谈。</p>

<p>这套“谁在哪个进程、怎么被拉起来”的逻辑，源码里写得很直白。下面这段就是引擎启动时<strong>分裂进程</strong>的骨架——
它先给三方分配 ZMQ 端口，再依次拉起 Scheduler 子进程、Detokenizer 子进程，最后把 TokenizerManager 留在主进程：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::_launch_subprocesses</span><span class="ln">三进程的分裂点</span></div>
  <pre><span class="kw">def</span> <span class="fn">_launch_subprocesses</span>(cls, server_args, ...):
    <span class="cm"># 1) 给三方分配 IPC 端口（ZMQ 通信）</span>
    port_args = PortArgs.init_new(server_args)

    <span class="cm"># 2) Scheduler 子进程：每个 TP rank 一个，持有 TpWorker/ModelRunner，做 GPU 前向</span>
    scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
        server_args, port_args, run_scheduler_process_func)

    <span class="cm"># 3) DetokenizerManager 子进程：反分词，把结果发回</span>
    detoken_procs, _ = cls._launch_detokenizer_subprocesses(
        server_args=server_args, port_args=port_args, ...)

    <span class="cm"># 4) TokenizerManager 留在主进程：分词、收发请求</span>
    tokenizer_manager, template_manager = init_tokenizer_manager_func(
        server_args, port_args)

    scheduler_init_result.wait_for_ready()   <span class="cm"># 等模型在子进程里加载完</span>
    <span class="kw">return</span> tokenizer_manager, template_manager, port_args, scheduler_init_result, ...</pre>
</div>

<p>源码注释把三大组件讲得很清楚：<strong>TokenizerManager</strong> 分词并把请求发给调度器；子进程里的 <strong>Scheduler</strong> 收请求、组批、前向、
把输出 token 发给反分词器；子进程里的 <strong>DetokenizerManager</strong> 反分词、把结果发回。<strong>HTTP 服务器、Engine、TokenizerManager 都在主进程</strong>，
进程间一律走 <strong>ZMQ 的 IPC</strong>（每个进程一个端口）。线上服务由 <span class="mono">launch_server.py</span> 把这套引擎包进 HTTP 服务器对外暴露；
而所有这些组件的“开关与旋钮”——模型路径、张量并行度、显存占比、最大并发等——都集中在 <span class="mono">srt/server_args.py</span> 的
<span class="inline">ServerArgs</span> 里，是你日后调参时最常翻的一页。</p>

<p>为什么偏要用 ZMQ 的进程间通信，而不是把三者写成同一个进程里的几个线程？根子上还是<strong>Python 与 GPU 的现实约束</strong>：
Python 有全局解释器锁（GIL），多线程很难让 CPU 的分词和 GPU 的前向真正并行；而拆成<strong>独立进程</strong>，各自有独立的解释器，
分词、前向、反分词就能在物理上<strong>同时推进</strong>，谁也不抢谁的锁。ZMQ 则提供了一套轻量、跨进程的消息队列：每个进程绑一个端口，
请求与结果像传送带上的包裹一样在它们之间流动，发送方不必等接收方处理完——这种<strong>异步解耦</strong>正是“零开销重叠调度”的物理基础。
代价是引擎启动时要多分配几个端口、多拉起几个子进程，并用一个<strong>看门狗（watchdog）</strong>盯着：万一某个子进程崩了，主进程能及时发现并退出，
而不是傻等。理解了“为什么拆进程、为什么用消息队列”，你就抓住了 SGLang 架构里最反直觉、也最关键的一块拼图。</p>

<h2>从哪里开始读</h2>
<p>有了这张地图，第一次读源码就不必从头啃。给三条推荐路线：① 想看<strong>请求怎么进来</strong>，从 <span class="mono">entrypoints/http_server.py</span> 的
<span class="mono">/generate</span> 入口顺藤摸瓜（正是第 3 课要走的那条线，深入版在第 13–17 课）；② 想看<strong>引擎的大脑</strong>，直接读
<span class="mono">managers/scheduler.py</span> 的事件循环（第 18–23 课）；③ 想看<strong>token 怎么在 GPU 上算出来</strong>，去
<span class="mono">model_executor/</span> 跟 <span class="inline">ModelRunner.forward</span>（第 24–28 课）。无论从哪条进去，
随时回到这张“两半 + 分层 + 三进程”的地图，你就不会迷路。下一课，我们就沿着第①条线，<strong>跟着一条真实请求走一遍它的一生</strong>。</p>

<p>读源码时还有个实用的小窍门：<strong>顺着进程边界读，而不是顺着文件名读</strong>。很多文件名看着相邻，运行时却分属不同进程、隔着一层 ZMQ，
硬把它们当“一个调用栈”去理解就会卡住。正确的姿势是先问自己一句：“<strong>我现在看的这段代码，跑在主进程还是子进程？</strong>”——
TokenizerManager 的方法都在主进程，Scheduler 与 ModelRunner 的方法都在子进程，两边通过 io_struct 里定义的消息结构通信。
把这条“进程边界”当作读码时的<strong>第一坐标轴</strong>，再叠加上面那张分层图作第二坐标轴，你就能给任何一个函数<strong>精确定位</strong>：
它在哪个进程、属于哪一层、对应未来哪一课。这套定位习惯，会让后面几十课的源码阅读都事半功倍。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>两半</strong>：<span class="mono">python/sglang/lang/</span>（前端 DSL，表达）+ <span class="mono">python/sglang/srt/</span>（运行时引擎，执行；srt = SGLang RunTime），底下垫着 <span class="mono">sgl-kernel/</span> 与 <span class="mono">jit_kernel/</span> 算子。</li>
    <li><strong>srt 分层</strong>：entrypoints（入口）→ managers（翻译 + Scheduler 大脑）→ model_executor/models（执行）→ sampling（采样）→ layers/mem_cache（算子与缓存）→ kernels/distributed（硬件与多卡）。</li>
    <li><strong>三进程</strong>：TokenizerManager（主进程）↔ Scheduler+TpWorker（子进程，每 TP rank 一个，握着 GPU）↔ DetokenizerManager（子进程），靠 <strong>ZMQ 的 IPC</strong> 通信。</li>
    <li><strong>别混淆</strong>：Scheduler 才是持有 ModelRunner、占着 GPU 的进程；两个 Manager 只是一进一出的“翻译官”。</li>
    <li><strong>调参入口</strong>：模型路径、并行度、显存占比等集中在 <span class="mono">srt/server_args.py</span> 的 <span class="inline">ServerArgs</span>。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson we sketched SGLang from 10,000 feet: it is an <strong>inference serving engine</strong> made of two halves,
a <strong>frontend language</strong> and a <strong>runtime engine</strong>. This lesson comes down to ground level and draws a
<strong>map of the whole repository</strong>: which two big rooms the code lives in, how the runtime half splits into smaller
departments, how that code <strong>forks into several processes</strong> at run time, and what those processes use to talk.
After this, whenever you dive into any directory you will know <strong>where you are on the map</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture SGLang as a <strong>working office building</strong>. The <strong>front desk on floor one (TokenizerManager)</strong>
  greets visitors and turns spoken requests into standard work orders; the <strong>dispatch room on floor two (Scheduler)</strong>
  is the brain that decides which orders to <strong>merge into one batch</strong>, who goes first, and how to queue — but the
  dispatcher <strong>does no hands-on work</strong>; the ones actually turning bolts on the line are the
  <strong>technicians (TpWorker / ModelRunner)</strong> who tend the <strong>expensive machines in the machine room (GPUs and
  operator kernels)</strong>; when a job is done, <strong>shipping &amp; packing (DetokenizerManager)</strong> translates the
  half-product into a finished item the customer can read and mails it back. The key: <strong>front desk, dispatch+floor, and
  shipping are three independent departments (processes)</strong> that pass work orders via the <strong>internal mail system
  (ZMQ)</strong>, so none blocks another — that is how reception and the shop floor run <strong>at the same time</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The whole repo can first be split crudely into <strong>two halves plus a foundation</strong>:
  <span class="mono">python/sglang/lang/</span> is the <strong>frontend DSL</strong> (how you "express" an LLM program),
  <span class="mono">python/sglang/srt/</span> is the <strong>runtime engine</strong> (srt = <strong>SGLang RunTime</strong>, how it
  "executes" fast and cheaply); underneath both sit <span class="mono">sgl-kernel/</span> (ahead-of-time compiled C++/CUDA ops)
  and <span class="mono">python/sglang/jit_kernel/</span> (lightweight just-in-time kernels). <strong>The star of this guide is
  srt/</strong> — internally it splits into a dozen-odd sub-packages, the base map for every later lesson.
</div>

<h2>The repo's two halves: lang/ and srt/</h2>
<p>Beginners opening the repo get scared by dozens of directories, but <strong>one first cut into two halves</strong> makes it
clear: one half teaches you <strong>how to write an LLM program</strong>, the other <strong>runs requests fast and cheaply</strong>.
The former is a convenience layer for app authors; the latter is the source of performance and what we dig into. The name
<span class="inline">SGLang</span> refers to both halves, but their code size, complexity, and distance from the hardware differ entirely.</p>

<div class="cols">
  <div class="col"><h4>Frontend DSL · <span class="mono">python/sglang/lang/</span></h4><p>Offers <span class="inline">gen</span> (generate a span),
  <span class="inline">fork</span> (spawn parallel branches), <span class="inline">join</span> (merge results), plus
  <span class="mono">system/user/assistant</span> roles and structured output. It makes "multi-step, parallel, constrained" prompt
  logic read like an ordinary program. Key files: <span class="mono">api.py</span>, <span class="mono">ir.py</span>,
  <span class="mono">interpreter.py</span>. It asks <strong>"how do I write this LLM program?"</strong></p></div>
  <div class="col"><h4>Runtime engine · <span class="mono">python/sglang/srt/</span></h4><p>The real serving engine: it receives requests and does
  <strong>tokenize → schedule → model forward → sample → detokenize</strong>, with RadixAttention, continuous batching, paged
  attention, and tensor/pipeline/expert parallelism built in. It asks <strong>"how do I execute requests fast and cheaply?"</strong>
  From this lesson on we mostly live in this half; it is also <strong>the thickest stack of code</strong> in the repo.</p></div>
</div>

<p>Keep this dividing line: <strong>lang/ expresses, srt/ executes</strong>. You can use srt/ alone (hit the HTTP endpoint or use the
<span class="inline">Engine</span> class) without writing any frontend; conversely, when the frontend says the parallel structure
explicitly, it helps the backend spot batchable, cacheable opportunities — exactly the logic behind Lesson 1's "critique the same
passage from three angles" example.</p>

<p>Why bind a "language" and a "runtime" in one repo? Because they are <strong>two ends of the same thing</strong>: the more clearly the
frontend can express "which calls can run in parallel, which prefixes are shared", the more chances the backend has to
<strong>cash those structures into real speedups</strong>. Many other frameworks do only the runtime and leave the frontend for users to
hand-stitch, so the scheduler receives a pile of isolated requests "with no visible relationships" and misses batching and caching
opportunities. SGLang puts both ends together precisely so the information between "express" and "execute" is <strong>not lost in
transit</strong>. That is also why the <span class="inline">Lang</span> in the name is not just "language" but "the kind of language the
runtime can understand" — a design philosophy that sets it apart from a plain inference server. For a beginner, though,
<strong>remembering that srt/ is the star is enough</strong>: the frontend is icing, while the runtime is this engine's real engine bay,
and the object this guide takes apart layer by layer.</p>

<h2>The runtime's package layout: from entry to kernel</h2>
<p>Slice <span class="mono">srt/</span> vertically and you see a stack of <strong>layers</strong>: at the top the entry that greets
clients, below it the decision-making brain, below that the layer that actually computes on the GPU, and at the bottom the
operator kernels hugging the hardware. This layer diagram is the <strong>master map</strong> of the whole guide — nearly every
later lesson zooms into one of these layers.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Entry</span><span class="name">entrypoints/</span></div><div class="ld">The public facade: <span class="mono">http_server.py</span> (OpenAI-compatible APIs + native <span class="mono">/generate</span>), <span class="mono">engine.py</span> (offline <span class="mono">Engine</span> class), <span class="mono">launch_server.py</span>. Lessons 13–17.</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">Ingress</span><span class="name">managers/ · translators</span></div><div class="ld"><span class="mono">tokenizer_manager.py</span> tokenizes, <span class="mono">detokenizer_manager.py</span> detokenizes — converting text and token ids both ways, the engine's "in/out ports".</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">Brain</span><span class="name">managers/ · Scheduler</span></div><div class="ld"><span class="mono">scheduler.py</span> decides who joins which batch and when to forward, manages KV cache and queues. Lessons 18–23 cover scheduling.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Execute</span><span class="name">model_executor/ · model_loader/ · models/</span></div><div class="ld"><span class="mono">ModelRunner</span> runs the model forward on the GPU; <span class="mono">model_loader/</span> loads weights, <span class="mono">models/</span> holds model definitions. Lessons 24–28.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Sample</span><span class="name">sampling/ · constrained/</span></div><div class="ld">Pick the next token from logits by temperature/top-p/top-k; <span class="mono">constrained/</span> enforces structured output (JSON, regex, etc.).</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Ops</span><span class="name">layers/ · mem_cache/</span></div><div class="ld"><span class="mono">layers/</span> holds attention and other operator layers (Lessons 33–37); <span class="mono">mem_cache/</span> manages KV cache and RadixAttention memory (Lessons 29–32).</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Hardware</span><span class="name">sgl-kernel/ · jit_kernel/ · distributed/</span></div><div class="ld">High-performance kernels hugging the chip (AOT <span class="mono">sgl-kernel</span> and JIT <span class="mono">jit_kernel</span>) plus multi-GPU communication. Lessons 38–42.</div></div>
</div>

<p>This diagram also explains why SGLang spans hardware: the upper layers are <strong>hardware-agnostic</strong> scheduling and
control; only the bottom kernels truly touch the chip. Switching hardware (AMD, TPU, NPU, CPU) mostly means <strong>swapping the
kernel backend</strong> while upper layers barely change. The table below pairs the runtime's main sub-packages, what each owns,
and <strong>which lesson</strong> expands it — your navigation index for when you "get lost" later:</p>

<p>Why split the runtime into <strong>so many sub-packages</strong> rather than one big blob? Because each sub-package maps to a
<strong>concern that can evolve independently</strong>: swap in a faster attention kernel today by touching only
<span class="mono">layers/</span>; support a new model tomorrow by adding one file in <span class="mono">models/</span>; change a KV
cache strategy the day after by touching only <span class="mono">mem_cache/</span>. This <strong>high-cohesion, low-coupling</strong>
split keeps a thousand-contributor open-source project maintainable — you change your kernel, I add my model, and the scheduler brain is
largely unaffected. It is friendly to learners too: you <strong>need not understand the whole engine at once</strong>; you can pick one
sub-package and drill in, and this guide's part structure mostly follows these sub-package boundaries.</p>

<table class="t">
  <tr><th>Sub-package</th><th>What it owns</th><th>Detail</th></tr>
  <tr><td class="mono">entrypoints/</td><td>Public entry: HTTP server, <span class="inline">Engine</span> class, launch scripts</td><td>Lessons 13–17</td></tr>
  <tr><td class="mono">managers/</td><td>Tokenize, detokenize, and <strong>Scheduler</strong> (the engine's brain)</td><td>Lessons 18–23</td></tr>
  <tr><td class="mono">model_executor/</td><td><span class="inline">ModelRunner</span>: organizes one model forward on the GPU</td><td>Lessons 24–28</td></tr>
  <tr><td class="mono">model_loader/ · models/</td><td>Load weights; define model architectures (Llama, Qwen, etc.)</td><td>Lessons 24–28</td></tr>
  <tr><td class="mono">mem_cache/</td><td>KV cache and <strong>RadixAttention</strong> memory management, paging</td><td>Lessons 29–32</td></tr>
  <tr><td class="mono">layers/</td><td>Attention, normalization, MoE operator layers and backends</td><td>Lessons 33–37</td></tr>
  <tr><td class="mono">distributed/</td><td>Tensor/pipeline/expert-parallel multi-GPU communication</td><td>Lesson 46</td></tr>
  <tr><td class="mono">speculative/ · lora/ · disaggregation/ · multimodal/</td><td>Speculative decoding, LoRA, PD disaggregation, multimodal, etc.</td><td>Advanced parts</td></tr>
</table>

<h2>The three-process model: who lives in which process, talking how</h2>
<p>The stack above is the <strong>static code structure</strong>; but once the engine <strong>runs</strong>, it is not one big loop but
<strong>forks into several processes</strong> working in parallel. This is the most overlooked yet most crucial point of SGLang's
architecture: <strong>tokenize, schedule+forward, and detokenize live in different processes</strong>, passing messages over ZMQ
IPC. Why split this way? Because tokenize/detokenize are CPU string processing while the model forward is heavy GPU compute;
putting them in separate processes lets CPU work and GPU work <strong>truly pipeline in parallel</strong> without blocking — the
engineering basis for "zero-overhead overlap" (Lesson 21).</p>

<div class="flow">
  <div class="node hl"><div class="nt">TokenizerManager</div><div class="nd">main process · tokenize, send/recv</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Scheduler + TpWorker</div><div class="nd">subprocess · one per TP rank · GPU forward</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">DetokenizerManager</div><div class="nd">subprocess · detokenize</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">back to TokenizerManager</div><div class="nd">main process · stream out</div></div>
</div>

<p>Never confuse the three roles: <strong>TokenizerManager is in the main process</strong> (so are the HTTP server and the
<span class="inline">Engine</span>); <strong>Scheduler is in a subprocess</strong>, and <strong>tensor parallelism (TP) starts one
Scheduler process per rank</strong>, each holding its own <span class="inline">TpWorker</span> and <span class="inline">ModelRunner</span>
and occupying one GPU to do the forward; <strong>DetokenizerManager is yet another subprocess</strong>. Remember: <strong>the Scheduler
is the process holding the GPU</strong>, while the two Managers are just "translators" on the way in and out — do not lump them together.</p>

<p>This "who lives in which process, how they get launched" logic is spelled out plainly in the source. The snippet below is the
skeleton of <strong>process forking</strong> at engine startup — it first allocates ZMQ ports for all three, then launches the
Scheduler subprocess and the Detokenizer subprocess, and finally leaves the TokenizerManager in the main process:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::_launch_subprocesses</span><span class="ln">the three-process fork point</span></div>
  <pre><span class="kw">def</span> <span class="fn">_launch_subprocesses</span>(cls, server_args, ...):
    <span class="cm"># 1) allocate IPC ports for all three (ZMQ communication)</span>
    port_args = PortArgs.init_new(server_args)

    <span class="cm"># 2) Scheduler subprocess: one per TP rank, holds TpWorker/ModelRunner, does GPU forward</span>
    scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
        server_args, port_args, run_scheduler_process_func)

    <span class="cm"># 3) DetokenizerManager subprocess: detokenize, send results back</span>
    detoken_procs, _ = cls._launch_detokenizer_subprocesses(
        server_args=server_args, port_args=port_args, ...)

    <span class="cm"># 4) TokenizerManager stays in the main process: tokenize, send/recv</span>
    tokenizer_manager, template_manager = init_tokenizer_manager_func(
        server_args, port_args)

    scheduler_init_result.wait_for_ready()   <span class="cm"># wait for the model to load in the subprocess</span>
    <span class="kw">return</span> tokenizer_manager, template_manager, port_args, scheduler_init_result, ...</pre>
</div>

<p>The source comment spells out the three components: <strong>TokenizerManager</strong> tokenizes and sends requests to the
scheduler; the <strong>Scheduler</strong> in a subprocess receives requests, forms batches, forwards them, and sends output tokens to
the detokenizer; the <strong>DetokenizerManager</strong> in a subprocess detokenizes and sends results back. <strong>The HTTP server,
Engine, and TokenizerManager all run in the main process</strong>, and processes talk strictly over <strong>ZMQ IPC</strong> (a port
per process). A live deployment is wrapped in an HTTP server by <span class="mono">launch_server.py</span>; and every component's
"switches and knobs" — model path, tensor-parallel size, memory fraction, max concurrency — live in
<span class="inline">ServerArgs</span> in <span class="mono">srt/server_args.py</span>, the page you will most often consult when tuning.</p>

<p>Why use ZMQ inter-process communication rather than writing all three as threads inside one process? At root it is the
<strong>reality of Python and the GPU</strong>: Python's global interpreter lock (GIL) makes it hard for CPU tokenization and GPU
forward to truly run in parallel as threads; split into <strong>separate processes</strong>, each with its own interpreter, tokenize,
forward and detokenize can advance <strong>physically at the same time</strong> without fighting over one lock. ZMQ then provides a
lightweight, cross-process message queue: each process binds a port, and requests and results flow between them like parcels on a
conveyor, the sender not waiting for the receiver to finish — this <strong>asynchronous decoupling</strong> is the physical basis of
"zero-overhead overlap scheduling". The cost is a few extra ports and subprocesses at startup, plus a <strong>watchdog</strong> keeping
an eye out: if a subprocess crashes, the main process notices and exits promptly instead of hanging forever. Once you understand "why
split processes, why a message queue", you have grasped the most counterintuitive and most crucial piece of SGLang's architecture.</p>

<h2>Where to start reading</h2>
<p>With this map, your first source read need not start from scratch. Three recommended routes: ① to see <strong>how a request comes
in</strong>, follow the <span class="mono">/generate</span> entry in <span class="mono">entrypoints/http_server.py</span> (exactly the
path Lesson 3 walks; the deep version is Lessons 13–17); ② to see <strong>the engine's brain</strong>, read the event loop in
<span class="mono">managers/scheduler.py</span> (Lessons 18–23); ③ to see <strong>how a token is computed on the GPU</strong>, go to
<span class="mono">model_executor/</span> and follow <span class="inline">ModelRunner.forward</span> (Lessons 24–28). Whichever door you
enter, keep returning to this "two halves + layers + three processes" map and you will not get lost. Next lesson we take route ①
and <strong>follow a real request through its whole life</strong>.</p>

<p>One practical trick when reading the source: <strong>read along process boundaries, not along file names</strong>. Many files look
adjacent yet live in different processes at run time, separated by a ZMQ hop; forcing them into "one call stack" gets you stuck. The
right move is to first ask: "<strong>is the code I'm reading running in the main process or a subprocess?</strong>" — TokenizerManager's
methods are all in the main process, Scheduler's and ModelRunner's are all in a subprocess, and the two sides communicate via the
message structs defined in io_struct. Treat this "process boundary" as your <strong>first coordinate axis</strong> while reading code,
overlay the layer diagram above as the second, and you can <strong>pinpoint</strong> any function: which process it is in, which layer
it belongs to, which future lesson it maps to. This locating habit pays off across the dozens of source-reading lessons ahead.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Two halves</strong>: <span class="mono">python/sglang/lang/</span> (frontend DSL, expresses) + <span class="mono">python/sglang/srt/</span> (runtime engine, executes; srt = SGLang RunTime), with <span class="mono">sgl-kernel/</span> and <span class="mono">jit_kernel/</span> kernels underneath.</li>
    <li><strong>srt layers</strong>: entrypoints (entry) → managers (translators + Scheduler brain) → model_executor/models (execute) → sampling → layers/mem_cache (ops &amp; cache) → kernels/distributed (hardware &amp; multi-GPU).</li>
    <li><strong>Three processes</strong>: TokenizerManager (main) ↔ Scheduler+TpWorker (subprocess, one per TP rank, holds the GPU) ↔ DetokenizerManager (subprocess), talking over <strong>ZMQ IPC</strong>.</li>
    <li><strong>Don't confuse</strong>: the Scheduler is the process holding ModelRunner and the GPU; the two Managers are just in/out "translators".</li>
    <li><strong>Tuning entry</strong>: model path, parallelism, memory fraction, etc. live in <span class="inline">ServerArgs</span> in <span class="mono">srt/server_args.py</span>.</li>
  </ul>
</div>
""",
}

LESSON_03 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
地图已经画好，这一课我们<strong>让它动起来</strong>：跟着一条真实的 <span class="inline">generate</span> 请求，
从你按下回车那一刻起，看它怎样<strong>穿过三个进程、被组成一批、在 GPU 上算出一个又一个 token，再流式地变回文字</strong>寄还给你。
这是一次“一万米高空”的端到端追踪——每一站的内部细节都留给后面的专门课程放大，这里只把<strong>整条链路串成一根线</strong>，
让你心里有一张能随时回放的“请求一生”动图。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把一条请求想成<strong>在餐厅点的一桌菜</strong>。你在<strong>柜台（TokenizerManager）</strong>报出需求，服务员把它写成标准菜单；
  单子递进<strong>厨房调度台（Scheduler）</strong>，大厨不会一桌一桌地做，而是把同时进来的单子<strong>并成一批</strong>一起下锅；
  <strong>灶台（GPU / ModelRunner）</strong>每“开一次火”，就让批里每桌都往前推进<strong>一道工序</strong>；上菜不是一次端齐，而是
  <strong>一道一道地分批上（流式输出）</strong>——这正是自回归生成“一次蹦一个字”的样子。整桌菜分两段：开头<strong>备料、起锅底</strong>
  （预填充，一次性把整段提示读进去）最费功夫，之后<strong>一道道出菜</strong>（解码，每次只算最新一个字）则是反复的小步快跑。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一条 <span class="inline">generate</span> 的全程其实就是上一课那张“三进程图”被<strong>真正走了一遍</strong>：
  <strong>HTTP 入口 → TokenizerManager 分词 →（ZMQ）→ Scheduler 组批 → ModelRunner 前向 → Sampler 采样 →
  DetokenizerManager 反分词 →（ZMQ）→ 流式 SSE 返回</strong>。其中最关键的认知是：<strong>中间一大段会循环很多次</strong>——
  一次<strong>预填充</strong>把提示读进去，之后很多次<strong>解码</strong>一个字一个字往外蹦，连续批处理就在每一圈里“插队、让位”，
  让 GPU 几乎不空转。
</div>

<h2>端到端：一条 generate 的全程</h2>
<p>先把整条链路铺开看一眼。下面这张“追踪图”按<strong>进程边界</strong>分成三段：主进程负责<strong>进与出</strong>（分词、最后的流式返回），
Scheduler 子进程是<strong>真正干活</strong>的中段（组批、前向、采样），Detokenizer 子进程负责把 token <strong>翻回文字</strong>。
进程之间的每一次跨越，都是一次 <span class="inline">ZMQ</span> 消息传递：</p>

<div class="trace">
  <div class="tcap">追踪一条 <b>POST /generate</b>：从文本提示到流式文本，跨越 <b>三个进程</b>（→ 表示一次 ZMQ 传递）</div>
  <div class="stations">
    <div class="stn"><h5>① 主进程 · 入口</h5><div class="cellrow"><span class="vc hot">HTTP /generate</span></div><div class="cellrow"><span class="vc">TokenizerManager</span><span class="vc blue">分词</span></div><div class="tlab">文本 → token id，建 TokenizedGenerateReqInput</div></div>
    <div class="op">ZMQ →</div>
    <div class="stn"><h5>② 子进程 · 大脑+灶台</h5><div class="cellrow"><span class="vc">Scheduler</span><span class="vc blue">组批</span></div><div class="cellrow"><span class="vc hot">ModelRunner 前向</span><span class="vc">Sampler</span></div><div class="tlab">recv → 队列 → get_next_batch → run_batch → 下一个 token</div></div>
    <div class="op">ZMQ →</div>
    <div class="stn"><h5>③ 子进程 · 出口</h5><div class="cellrow"><span class="vc">DetokenizerManager</span><span class="vc blue">反分词</span></div><div class="cellrow"><span class="vc hot">SSE 流式</span><span class="vc dim">回主进程</span></div><div class="tlab">token id → 文字增量，经主进程 SSE 推回</div></div>
  </div>
</div>

<p>顺着这张图走一遍：你 <span class="mono">POST /generate</span> 的请求先落到 <span class="mono">entrypoints/http_server.py</span> 的处理函数，
它把负载包成一个 <span class="inline">GenerateReqInput</span>，交给 <span class="inline">TokenizerManager.generate_request</span>。
TokenizerManager 在<strong>主进程</strong>里把文本分词成 token id、归一化参数，打包成 <span class="inline">TokenizedGenerateReqInput</span>
（定义在 <span class="mono">managers/io_struct.py</span>），通过 ZMQ 发给 Scheduler 子进程。Scheduler 收下、排队、组批、在 GPU 上前向并采样，
把新生成的 token 发给 Detokenizer 子进程；后者做<strong>增量反分词</strong>，把“这次新增的几个字”发回主进程，由 HTTP 层以
<strong>SSE（text/event-stream）</strong>一段段推给你，直到遇到结束符或达到长度上限。</p>

<p>这里有个容易被忽略却很要紧的设计：<strong>反分词必须是“增量”的</strong>。模型每吐一个 token，并不一定对应一个完整的汉字或单词——
有些字要好几个 token 拼起来才完整。如果每来一个 token 就把<strong>整段</strong>已生成内容从头反分词一遍，既浪费又可能把半个字符错误地切出来。
所以 DetokenizerManager 会记住“上次已经发到第几个字符”，每次只把<strong>真正新增、且能安全显示</strong>的那一小段文字推出去
（源码里叫 <span class="inline">sent_offset</span>）。这就是你在前端看到答案“一个词一个词蹦出来”的真实来源：
不是模型写完再返回，而是引擎<strong>边算边以增量的方式</strong>把文字流给你。流式（streaming）之所以能极大改善体验，
正是因为用户在<strong>第一个 token 算出来的瞬间</strong>就开始看到回答，而不必干等整段生成结束——这对动辄几百 token 的长回答尤其关键。</p>

<p>把这条全程再压缩成一句口诀，方便记忆：<strong>“主进程分词、子进程算、子进程翻、主进程吐”</strong>。请求进来在主进程被分词、登记，
跨过一道 ZMQ 到 Scheduler 子进程被组批、前向、采样，新 token 再跨一道 ZMQ 到 Detokenizer 子进程被翻成文字，最后绕回主进程以 SSE 流出。
注意这四段之间<strong>不是同步阻塞</strong>的——当 Scheduler 正为第 N 个 token 做前向时，Detokenizer 可能还在翻第 N-1 个 token，
而主进程已经把第 N-2 个 token 推给了用户，三个进程像接力赛一样<strong>同时在跑不同的棒次</strong>。正是这种跨进程的流水线，
让“一次请求的一生”在宏观上看是一条直线，在微观上却是三方<strong>并行协作</strong>的结果。带着这张端到端的动图，
我们就准备好钻进任何一个环节去看它的内部了。</p>

<h2>调度器循环：请求如何被组批执行</h2>
<p>整条链路里，<strong>最值得先看清</strong>的是 Scheduler 这颗大脑——因为它是个<strong>永不停歇的循环</strong>。它不是“来一条算一条”，
而是反复地做四件事：<strong>收请求 → 组下一批 → 在 GPU 上跑这一批 → 处理结果</strong>。这四步就是引擎的心跳，下面把它拆成竖直的步骤：</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>收请求 recv_requests</h4><p class="mono">Scheduler · 经 ZMQ</p><p>从 TokenizerManager 收下新到的 <span class="inline">TokenizedGenerateReqInput</span>，放进<strong>等待队列</strong>。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>组下一批 get_next_batch_to_run</h4><p class="mono">Scheduler · 决策</p><p>决定这一步该跑<strong>预填充</strong>（吃新请求的提示）还是<strong>解码</strong>（给在跑的请求各推进一个 token），并把它们拼成一个 <span class="inline">ScheduleBatch</span>。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>跑这一批 run_batch</h4><p class="mono">TpWorker · ModelRunner</p><p>在 GPU 上做一次模型前向得到 logits，再由 <span class="inline">Sampler</span> 采出每条请求的下一个 token。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>处理结果 process_batch_result</h4><p class="mono">Scheduler → Detok</p><p>把新 token 发给反分词器；没结束的请求<strong>接回新 token</strong>，下一圈继续；结束的请求<strong>让出位置</strong>。</p></div></div>
</div>

<p>这套循环在源码里短得惊人——下面就是“正常模式”的事件循环骨架，把上面四步几乎一一对应（还有个 <span class="inline">event_loop_overlap</span>
重叠版，会在 CPU 准备下一批的同时让 GPU 先算当前批，第 21 课展开）：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::event_loop_normal</span><span class="ln">引擎的心跳</span></div>
  <pre><span class="kw">def</span> <span class="fn">event_loop_normal</span>(self):
    <span class="st">&quot;&quot;&quot;A normal scheduler loop.&quot;&quot;&quot;</span>
    <span class="kw">while</span> True:
        <span class="cm"># ① 收请求（经 ZMQ 从 TokenizerManager 来），进等待队列</span>
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        <span class="cm"># ② 组下一批：这一步跑预填充还是解码？</span>
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        <span class="kw">if</span> batch:
            <span class="cm"># ③ 在 GPU 上前向 + 采样</span>
            result = self.run_batch(batch)
            <span class="cm"># ④ 处理结果：把新 token 发给反分词器</span>
            self.process_batch_result(batch, result)
        <span class="kw">else</span>:
            self.on_idle()

        self.last_batch = batch</pre>
</div>

<p>这里藏着 SGLang 高吞吐的<strong>秘密入口</strong>：第 ② 步的 <span class="inline">get_next_batch_to_run</span> 每一圈都<strong>重新组批</strong>，
所以<strong>连续批处理</strong>就在这里“插队”——新请求随时能挤进当前批，已结束的随时让位（第 5 课）；而<strong>RadixAttention</strong>
让共享前缀的 KV 缓存只算一次（第 7 课）；组批与排队的具体策略是第 20 课，整个调度循环的源码精读是第 18 课。</p>

<p>再点一个关键细节：第 ③ 步的 <span class="inline">run_batch</span> 才是<strong>唯一真正碰 GPU</strong> 的地方，它把整批请求交给
<span class="inline">TpWorker</span> 里的 <span class="inline">ModelRunner.forward</span>，在显卡上做一次模型前向，算出每条请求“下一个 token”的
logits，再交给 <span class="inline">Sampler</span> 按温度 / top-p / top-k 采样。注意整个循环是<strong>单线程</strong>地一圈一圈跑的——
没有满天飞的回调，没有复杂的多线程同步，就是朴素的 <span class="inline">while True</span>。这种<strong>“一个大循环统管一切”</strong>的设计
看似简单，却正是它好懂、好调、好优化的原因：你只要盯住这一个循环，就能回答“此刻引擎在干什么、下一步会干什么”。
而那个重叠版的 <span class="inline">event_loop_overlap</span> 之所以更快，是因为它把第 ③ 步“算这一批”和第 ④ 步“处理上一批结果”
错开成流水线，让 GPU 在算的同时 CPU 已经在准备——但骨架仍是同一个循环，理解了 normal 版，overlap 版就只是它的“加速变体”。</p>

<h2>解码循环：为什么要转很多圈</h2>
<p>很多人对“一条请求”最大的误解，是以为模型“一口气把答案写完”。其实<strong>生成是逐 token、自回归</strong>的：算出一个 token、把它接回输入、
再算下一个，如此反复，直到模型吐出结束符或到达最大长度。所以第 2~4 步会<strong>循环几十上百次</strong>，每一圈只往前挪一个字：</p>

<div class="flow">
  <div class="node hl"><div class="nt">收/组批</div><div class="nd">recv + get_next_batch</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">前向</div><div class="nd">ModelRunner · logits</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">采样</div><div class="nd">Sampler · 下一个 token</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">未结束？</div><div class="nd">接回 token，回到组批</div></div>
</div>

<p>正因为每一圈都要重新组批，连续批处理才有“插队”的机会：<strong>一条新请求不必等当前这批全部跑完</strong>，下一圈就能加入；
一条已结束的请求也会立刻让出它在批里的位置，省下的算力马上分给别人。这就是为什么 SGLang 的 GPU 利用率能压得这么满——
循环的<strong>每一步</strong>都是一次重新洗牌的机会，而不是“一批跑到死”。</p>

<p>顺便点破一个新手常踩的坑：这里的“批（batch）”和训练时的“batch”<strong>不是一回事</strong>。训练时一个 batch 里的样本长度往往被对齐、一起进一起出；
而推理服务里，每条请求<strong>进来的时机不同、要生成的长度也不同</strong>——有的刚发来要做预填充，有的已经吐了几十个 token 正在解码，有的马上就要结束。
连续批处理（continuous batching，也叫 in-flight batching）的精髓，就是<strong>不强求大家步调一致</strong>：每一步都按当前在场的请求<strong>动态拼一个批</strong>，
来了就加、走了就撤。也正因如此，调度器的循环必须<strong>足够快</strong>——它每一圈的开销都会乘以“循环很多次”而被放大，这正是“零开销重叠调度器”
（第 21 课）要把 CPU 组批开销藏进 GPU 计算时间里的原因。把“逐 token 循环”和“每圈重组批”这两件事叠在一起想，你就明白了
SGLang 吞吐与延迟优化的<strong>所有故事，几乎都发生在这个循环里</strong>。</p>

<h2>预填充 vs 解码：两种不同的活</h2>
<p>循环里其实混着<strong>两种性质完全不同</strong>的步骤，分清它们是理解后面所有性能课的前提。<strong>预填充（prefill）</strong>是请求刚进来时，
把<strong>整段提示一次性</strong>读进模型、把它的 KV 缓存填好——一段提示有几百上千个 token，要并行算一大片，是<strong>计算密集（compute-bound）</strong>的，
但通常<strong>只做一次</strong>。<strong>解码（decode）</strong>是之后每一圈只算<strong>最新的那一个 token</strong>，计算量很小，瓶颈在于反复<strong>搬运庞大的 KV 缓存</strong>，
是<strong>访存密集（memory-bound）</strong>的，而且要<strong>循环很多次</strong>。下面这张时间线把两者并排对照：</p>

<div class="timeline">
  <div class="lane"><div class="lane-label">预填充</div><div class="tslot now">读入整段提示</div><div class="tslot span">一次并行算一大片 · 计算密集 · 通常只做一次</div></div>
  <div class="lane"><div class="lane-label">解码</div><div class="tslot">+1 token</div><div class="tslot">+1 token</div><div class="tslot">+1 token</div><div class="tslot">+1 token</div><div class="tslot now">… 循环到结束</div></div>
  <div class="lane"><div class="lane-label">瓶颈</div><div class="tslot span">预填充卡在算力（矩阵乘）</div><div class="tslot span">解码卡在显存带宽（搬 KV）</div></div>
</div>

<p>两者特性不同，调度策略也不同：把太长的提示一次喂进去会撑爆显存，于是有<strong>分块预填充（chunked prefill，第 22 课）</strong>
把长提示切片喂；预填充和解码抢同一张卡又会互相拖累，于是 SGLang 甚至能把两者<strong>拆到不同机器</strong>上做
（<strong>预填充-解码分离</strong>）。但在最朴素的单机情形里，你只要记住这条主线：<strong>一条请求 = 一次预填充 + 很多次解码</strong>，
而连续批处理让无数条这样的请求在同一张卡上<strong>错峰流水</strong>。这就是“一次请求的一生”的全部骨架——后面的每一课，
都是在给这根骨架的某一节添上血肉。</p>

<p>为什么一定要分清这两段？因为它们决定了后面几乎所有性能优化的<strong>用力方向</strong>。预填充是计算密集的，优化它要在
<strong>算力</strong>上做文章：更高效的注意力内核、更好的并行切分；解码是访存密集的，优化它要在<strong>显存</strong>上做文章：
KV 缓存怎么排布、怎么分页、怎么压缩、怎么复用共享前缀（这正是 KV 缓存第 4、29–32 课，分页注意力第 6 课，RadixAttention 第 7 课的主战场）。
一个常见的直觉误区是“以为生成的全部成本都在‘想答案’”，其实对长对话来说，<strong>反复搬运越来越长的 KV 缓存</strong>才是解码阶段真正的开销大头。
所以当你日后看到各种“省显存、提带宽”的花式技巧时，回到这张预填充-解码时间线，你就明白它们到底在<strong>救哪一段的命</strong>。
把这条主线刻进脑子里：<strong>预填充一次、解码千百次；预填充拼算力、解码拼显存</strong>——它会是你理解整个第二部分的钥匙。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>全程</strong>：HTTP <span class="mono">/generate</span> → TokenizerManager 分词 →（ZMQ）→ Scheduler 组批 → ModelRunner 前向 → Sampler 采样 → DetokenizerManager 反分词 →（ZMQ）→ SSE 流式返回。</li>
    <li><strong>进程边界</strong>：分词与返回在<strong>主进程</strong>，组批+前向+采样在 <strong>Scheduler 子进程</strong>，反分词在 <strong>Detokenizer 子进程</strong>；每次跨进程都是一次 ZMQ 传递。</li>
    <li><strong>调度心跳</strong>：<span class="inline">event_loop_normal</span> 反复做 recv_requests → get_next_batch_to_run → run_batch → process_batch_result。</li>
    <li><strong>循环本质</strong>：自回归逐 token，第 2~4 步循环很多次；连续批处理在每一圈“插队/让位”，GPU 几乎不空转。</li>
    <li><strong>两段</strong>：预填充（读整段提示，计算密集，通常一次）vs 解码（每次一个 token，访存密集，循环多次）。一条请求 = 一次预填充 + 很多次解码。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
The map is drawn; this lesson <strong>brings it to life</strong>: we follow a real <span class="inline">generate</span> request from
the moment you hit enter, watching it <strong>cross three processes, get merged into a batch, compute one token after another on the
GPU, then turn back into text</strong> streamed home to you. This is a "10,000-foot" end-to-end trace — each station's internals are
left to later dedicated lessons; here we just <strong>string the whole path into one line</strong>, so you carry a replayable mental
animation of a "request's life".
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of one request as <strong>a table's worth of dishes ordered at a restaurant</strong>. You state your needs at the
  <strong>counter (TokenizerManager)</strong> and the server writes a standard menu ticket; the ticket goes into the
  <strong>kitchen dispatch (Scheduler)</strong>, where the chef does not cook table by table but <strong>merges tickets that arrive
  together into one batch</strong>; each time the <strong>stove (GPU / ModelRunner)</strong> "fires once", every table in the batch
  advances <strong>one step</strong>; dishes are not served all at once but <strong>course by course (streaming output)</strong> —
  exactly how autoregressive generation "pops one word at a time". The whole meal has two phases: the opening
  <strong>prep and base stock</strong> (prefill, reading the entire prompt in at once) takes the most effort, after which
  <strong>serving course by course</strong> (decode, computing only the newest word each time) is a repeated quick small step.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  A <span class="inline">generate</span>'s whole journey is just last lesson's "three-process diagram" <strong>actually walked
  once</strong>: <strong>HTTP entry → TokenizerManager tokenize → (ZMQ) → Scheduler batch → ModelRunner forward → Sampler sample →
  DetokenizerManager detokenize → (ZMQ) → streaming SSE back</strong>. The crucial insight: <strong>the big middle stretch loops many
  times</strong> — one <strong>prefill</strong> reads the prompt in, then many <strong>decodes</strong> pop out word by word, and
  continuous batching "cuts in / yields" on every turn so the GPU almost never idles.
</div>

<h2>End-to-end: a generate's whole journey</h2>
<p>First lay the whole path out at a glance. The trace below is split by <strong>process boundary</strong> into three segments: the
main process handles <strong>in and out</strong> (tokenize, and the final streaming return), the Scheduler subprocess is the
<strong>real-work</strong> middle (batch, forward, sample), and the Detokenizer subprocess turns tokens <strong>back into text</strong>.
Every crossing between processes is one <span class="inline">ZMQ</span> message:</p>

<div class="trace">
  <div class="tcap">Tracing one <b>POST /generate</b>: from text prompt to streamed text, across <b>three processes</b> (→ is one ZMQ hop)</div>
  <div class="stations">
    <div class="stn"><h5>① Main · entry</h5><div class="cellrow"><span class="vc hot">HTTP /generate</span></div><div class="cellrow"><span class="vc">TokenizerManager</span><span class="vc blue">tokenize</span></div><div class="tlab">text → token ids, build TokenizedGenerateReqInput</div></div>
    <div class="op">ZMQ →</div>
    <div class="stn"><h5>② Subprocess · brain+stove</h5><div class="cellrow"><span class="vc">Scheduler</span><span class="vc blue">batch</span></div><div class="cellrow"><span class="vc hot">ModelRunner forward</span><span class="vc">Sampler</span></div><div class="tlab">recv → queue → get_next_batch → run_batch → next token</div></div>
    <div class="op">ZMQ →</div>
    <div class="stn"><h5>③ Subprocess · exit</h5><div class="cellrow"><span class="vc">DetokenizerManager</span><span class="vc blue">detokenize</span></div><div class="cellrow"><span class="vc hot">SSE stream</span><span class="vc dim">via main</span></div><div class="tlab">token ids → text delta, pushed back as SSE via main</div></div>
  </div>
</div>

<p>Walk it through: your <span class="mono">POST /generate</span> first lands in the handler in
<span class="mono">entrypoints/http_server.py</span>, which wraps the payload into a <span class="inline">GenerateReqInput</span> and hands it
to <span class="inline">TokenizerManager.generate_request</span>. In the <strong>main process</strong>, TokenizerManager tokenizes the
text into token ids, normalizes params, packs a <span class="inline">TokenizedGenerateReqInput</span> (defined in
<span class="mono">managers/io_struct.py</span>), and sends it over ZMQ to the Scheduler subprocess. The Scheduler receives, queues,
batches, forwards and samples on the GPU, and sends newly generated tokens to the Detokenizer subprocess; the latter does
<strong>incremental detokenization</strong>, sends "the few new characters this time" back to the main process, and the HTTP layer
pushes them to you piece by piece as <strong>SSE (text/event-stream)</strong>, until a stop token or the length limit.</p>

<p>Here is an easily overlooked but important design: <strong>detokenization must be "incremental"</strong>. Each token the model emits
does not necessarily map to a complete character or word — some characters take several tokens to form. Re-detokenizing the
<strong>entire</strong> generated text from scratch on every new token would be wasteful and could wrongly slice a half character. So the
DetokenizerManager remembers "how many characters were already sent" and each time pushes only the small slice that is
<strong>genuinely new and safe to display</strong> (called <span class="inline">sent_offset</span> in the source). That is the real
source of answers appearing "word by word" in your UI: not the model finishing then returning, but the engine streaming text
<strong>incrementally as it computes</strong>. Streaming improves experience so much precisely because the user starts seeing the answer
<strong>the instant the first token is computed</strong>, instead of waiting for the whole generation to finish — especially crucial for
long answers of hundreds of tokens.</p>

<p>Compress the whole path into one mnemonic: <strong>"main tokenizes, subprocess computes, subprocess translates, main emits"</strong>.
A request enters and is tokenized and registered in the main process, crosses one ZMQ hop to the Scheduler subprocess to be batched,
forwarded and sampled, the new token crosses another ZMQ hop to the Detokenizer subprocess to be turned into text, and finally loops
back to the main process to stream out as SSE. Note these four stages are <strong>not synchronously blocking</strong> — while the
Scheduler forwards token N, the Detokenizer may still be translating token N-1, and the main process has already pushed token N-2 to the
user; the three processes run <strong>different relay legs at the same time</strong>. It is exactly this cross-process pipeline that makes
"a request's life" look like a straight line at the macro level yet, microscopically, the result of three parties <strong>collaborating
in parallel</strong>. Carrying this end-to-end animation, we are ready to dive into any stage and see its internals.</p>

<h2>The scheduler loop: how requests get batched and run</h2>
<p>Across the whole path, the thing <strong>most worth seeing first</strong> is the Scheduler brain — because it is a
<strong>never-stopping loop</strong>. It does not "compute one as each arrives" but repeatedly does four things: <strong>receive
requests → form the next batch → run that batch on the GPU → process results</strong>. These four are the engine's heartbeat;
here they are unrolled as vertical steps:</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>recv_requests</h4><p class="mono">Scheduler · via ZMQ</p><p>Receive newly arrived <span class="inline">TokenizedGenerateReqInput</span> from TokenizerManager and put it in the <strong>waiting queue</strong>.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>get_next_batch_to_run</h4><p class="mono">Scheduler · decide</p><p>Decide whether this step runs <strong>prefill</strong> (eat new requests' prompts) or <strong>decode</strong> (advance running requests by one token each), and assemble a <span class="inline">ScheduleBatch</span>.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>run_batch</h4><p class="mono">TpWorker · ModelRunner</p><p>Run one model forward on the GPU for logits, then <span class="inline">Sampler</span> picks each request's next token.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>process_batch_result</h4><p class="mono">Scheduler → Detok</p><p>Send new tokens to the detokenizer; unfinished requests <strong>append the new token</strong> and loop on; finished ones <strong>yield their slot</strong>.</p></div></div>
</div>

<p>This loop is astonishingly short in the source — below is the "normal mode" event-loop skeleton, mapping almost one-to-one to the
four steps above (there is also an <span class="inline">event_loop_overlap</span> variant that lets the GPU run the current batch while
the CPU prepares the next, expanded in Lesson 21):</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::event_loop_normal</span><span class="ln">the engine's heartbeat</span></div>
  <pre><span class="kw">def</span> <span class="fn">event_loop_normal</span>(self):
    <span class="st">&quot;&quot;&quot;A normal scheduler loop.&quot;&quot;&quot;</span>
    <span class="kw">while</span> True:
        <span class="cm"># 1) receive requests (via ZMQ from TokenizerManager), enqueue</span>
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        <span class="cm"># 2) form the next batch: prefill or decode this step?</span>
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        <span class="kw">if</span> batch:
            <span class="cm"># 3) forward + sample on the GPU</span>
            result = self.run_batch(batch)
            <span class="cm"># 4) process results: send new tokens to the detokenizer</span>
            self.process_batch_result(batch, result)
        <span class="kw">else</span>:
            self.on_idle()

        self.last_batch = batch</pre>
</div>

<p>Here is the <strong>secret entrance</strong> to SGLang's throughput: step 2's <span class="inline">get_next_batch_to_run</span>
<strong>re-forms the batch every turn</strong>, so <strong>continuous batching</strong> "cuts in" right here — a new request can squeeze
into the current batch any turn, a finished one yields any turn (Lesson 5); <strong>RadixAttention</strong> computes a shared prefix's
KV cache only once (Lesson 7); the concrete batching/queuing policy is Lesson 20, and a close source read of the whole loop is Lesson 18.</p>

<p>One more key detail: step 3's <span class="inline">run_batch</span> is the <strong>only place that actually touches the GPU</strong>. It
hands the whole batch to <span class="inline">ModelRunner.forward</span> inside the <span class="inline">TpWorker</span>, runs one model
forward on the card to get each request's "next token" logits, then passes them to the <span class="inline">Sampler</span> to sample by
temperature / top-p / top-k. Note the whole loop runs <strong>single-threaded</strong>, turn by turn — no callbacks flying around, no
complex multi-thread synchronization, just a plain <span class="inline">while True</span>. This <strong>"one big loop governs
everything"</strong> design looks simple but is exactly why it is easy to understand, debug and optimize: watch this one loop and you
can answer "what is the engine doing now, what will it do next". The overlap variant <span class="inline">event_loop_overlap</span> is
faster because it staggers step 3 "run this batch" and step 4 "process the last batch's result" into a pipeline, so the CPU prepares
while the GPU computes — but the skeleton is the same loop; once you get the normal version, overlap is just its "accelerated variant".</p>

<h2>The decode loop: why it goes around so many times</h2>
<p>The biggest misconception about "one request" is that the model "writes the whole answer in one go". In fact <strong>generation is
token-by-token, autoregressive</strong>: compute one token, append it to the input, compute the next, and so on until the model emits a
stop token or hits the max length. So steps 2–4 <strong>loop tens to hundreds of times</strong>, advancing one word per turn:</p>

<div class="flow">
  <div class="node hl"><div class="nt">recv/batch</div><div class="nd">recv + get_next_batch</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">forward</div><div class="nd">ModelRunner · logits</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">sample</div><div class="nd">Sampler · next token</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">unfinished?</div><div class="nd">append token, back to batch</div></div>
</div>

<p>Because the batch is re-formed every turn, continuous batching gets its chance to "cut in": <strong>a new request need not wait for
the current batch to finish</strong> — it can join next turn; a finished request immediately yields its slot in the batch, and the
freed compute goes to others at once. That is why SGLang keeps GPU utilization so high — <strong>every step</strong> of the loop is a
chance to reshuffle, rather than "run one batch to the bitter end".</p>

<p>One trap beginners hit: this "batch" is <strong>not the same</strong> as a training "batch". In training, samples in a batch are often
length-aligned and go in and out together; in inference serving, each request <strong>arrives at a different time and generates a
different length</strong> — one just sent in and needs prefill, one has emitted dozens of tokens and is decoding, one is about to finish.
The essence of continuous batching (also called in-flight batching) is to <strong>not force everyone in lockstep</strong>: each step
<strong>dynamically assembles a batch</strong> from whoever is present, adding arrivals and dropping departures. That is also why the
scheduler loop must be <strong>fast enough</strong> — its per-turn overhead gets multiplied by "loops many times", which is exactly why
the zero-overhead overlap scheduler (Lesson 21) hides CPU batching overhead behind GPU compute. Stack "token-by-token loop" and
"re-batch every turn" together and you see that <strong>almost the entire story of SGLang's throughput and latency lives in this loop</strong>.</p>

<h2>Prefill vs decode: two different jobs</h2>
<p>The loop actually mixes <strong>two completely different kinds</strong> of step, and telling them apart is the prerequisite for every
later performance lesson. <strong>Prefill</strong> happens when a request just arrives: read the <strong>entire prompt in at once</strong>
and fill its KV cache — a prompt has hundreds to thousands of tokens computed in parallel as a big slab, so it is
<strong>compute-bound</strong>, but usually done <strong>only once</strong>. <strong>Decode</strong> is every later turn computing only
<strong>the single newest token</strong>; the compute is tiny, the bottleneck is repeatedly <strong>moving the huge KV cache</strong>, so
it is <strong>memory-bound</strong> and <strong>loops many times</strong>. The timeline below puts them side by side:</p>

<div class="timeline">
  <div class="lane"><div class="lane-label">Prefill</div><div class="tslot now">read whole prompt</div><div class="tslot span">one big parallel slab · compute-bound · usually once</div></div>
  <div class="lane"><div class="lane-label">Decode</div><div class="tslot">+1 token</div><div class="tslot">+1 token</div><div class="tslot">+1 token</div><div class="tslot">+1 token</div><div class="tslot now">… loop to end</div></div>
  <div class="lane"><div class="lane-label">Bottleneck</div><div class="tslot span">prefill is bound by compute (matmul)</div><div class="tslot span">decode is bound by memory bandwidth (KV moves)</div></div>
</div>

<p>Their natures differ, so do their scheduling strategies: feeding too long a prompt in one shot blows up memory, hence
<strong>chunked prefill (Lesson 22)</strong> slices long prompts; prefill and decode fighting over the same card drag each other down,
so SGLang can even split them across <strong>different machines</strong> (<strong>prefill-decode disaggregation</strong>). But in the
plainest single-machine case, just remember this throughline: <strong>one request = one prefill + many decodes</strong>, and continuous
batching lets countless such requests <strong>pipeline out of phase</strong> on the same card. That is the entire skeleton of a
"request's life" — every later lesson adds flesh to one joint of this skeleton.</p>

<p>Why insist on telling the two phases apart? Because they decide the <strong>direction of effort</strong> for almost every later
optimization. Prefill is compute-bound, so optimizing it works on <strong>compute</strong>: more efficient attention kernels, better
parallel partitioning; decode is memory-bound, so optimizing it works on <strong>memory</strong>: how the KV cache is laid out, paged,
compressed, and reused across shared prefixes (the home turf of KV cache in Lessons 4 and 29–32, paged attention in Lesson 6, and
RadixAttention in Lesson 7). A common intuition trap is "thinking all the cost is in thinking up the answer", when in fact for long
conversations <strong>repeatedly moving an ever-growing KV cache</strong> is the real bulk of decode-phase cost. So when you later meet
all kinds of "save memory, lift bandwidth" tricks, return to this prefill–decode timeline and you will see exactly <strong>which phase
they are saving</strong>. Carve this throughline into your mind: <strong>prefill once, decode hundreds of times; prefill races on
compute, decode races on memory</strong> — it is the key to understanding all of Part 2.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Whole path</strong>: HTTP <span class="mono">/generate</span> → TokenizerManager tokenize → (ZMQ) → Scheduler batch → ModelRunner forward → Sampler sample → DetokenizerManager detokenize → (ZMQ) → SSE streaming back.</li>
    <li><strong>Process boundaries</strong>: tokenize and return in the <strong>main process</strong>, batch+forward+sample in the <strong>Scheduler subprocess</strong>, detokenize in the <strong>Detokenizer subprocess</strong>; each crossing is one ZMQ hop.</li>
    <li><strong>Scheduler heartbeat</strong>: <span class="inline">event_loop_normal</span> repeatedly does recv_requests → get_next_batch_to_run → run_batch → process_batch_result.</li>
    <li><strong>Loop essence</strong>: autoregressive token-by-token, steps 2–4 loop many times; continuous batching "cuts in / yields" each turn so the GPU barely idles.</li>
    <li><strong>Two phases</strong>: prefill (read the whole prompt, compute-bound, usually once) vs decode (one token per turn, memory-bound, many loops). One request = one prefill + many decodes.</li>
  </ul>
</div>
""",
}
