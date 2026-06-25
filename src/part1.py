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
上层几乎不动——这正是“从单卡到大型集群、覆盖多种硬件”的工程底气（第 6 课展开硬件后端）。</p>

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
  <tr><td><strong>零开销重叠调度器</strong></td><td class="mono">GPU 等 CPU 而空转</td><td>GPU 算当前批时，CPU 已在准备下一批，两者重叠</td><td>第 8 课</td></tr>
  <tr><td><strong>连续批处理</strong></td><td class="mono">批内有空位、GPU 利用率低</td><td>每步都让新请求插队、已结束的让位，批次始终“满”</td><td>第 9 课</td></tr>
  <tr><td><strong>分页注意力 / 量化</strong></td><td class="mono">显存碎片与带宽吃紧</td><td>分页管理 KV 显存、低精度（FP8/INT4）压缩计算与存储</td><td>第 10 课</td></tr>
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
GPU to large clusters across many hardware" (kernel backends in Lesson 6).</p>

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
  <tr><td><strong>Zero-overhead overlap scheduler</strong></td><td class="mono">GPU idles waiting on CPU</td><td>while the GPU runs the current batch, the CPU is already preparing the next; the two overlap</td><td>Lesson 8</td></tr>
  <tr><td><strong>Continuous batching</strong></td><td class="mono">empty slots, low GPU utilization</td><td>each step lets new requests cut in and finished ones yield, so the batch stays "full"</td><td>Lesson 9</td></tr>
  <tr><td><strong>Paged attention / quantization</strong></td><td class="mono">memory fragmentation &amp; bandwidth</td><td>page-manage KV memory; low precision (FP8/INT4) compresses compute and storage</td><td>Lesson 10</td></tr>
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
