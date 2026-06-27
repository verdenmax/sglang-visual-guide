"""Part 4 - Entrypoints & orchestration. Lessons (L13-L17) for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons open the runtime's outer shell: the Engine / HTTP server entry
(L13), the TokenizerManager front door (L14), the OpenAI/Anthropic/Ollama compat
layer (L15), the io_struct IPC messages between processes (L16), and the
DetokenizerManager + SSE streaming exit (L17).
"""

LESSON_13 = {
    "zh": r"""
<p class="lead">
欢迎来到 Part 4——我们终于从前端 DSL 走进了<strong>运行时（runtime）</strong>的大门。这一课先回答一个最基本的问题：
你写好的程序，到底通过<strong>什么入口</strong>把请求喂给 SGLang 的引擎？答案有两个，而且只有两个：<strong>离线的
<span class="inline">Engine</span></strong>（纯进程内 Python API）和<strong>在线的 HTTP server</strong>（FastAPI 服务）。
看懂这两扇门、以及它们<strong>共享同一套运行时</strong>这件事，是读懂后面每一课的钥匙。
</p>

<p>为什么先讲入口？因为 Part 1–3 我们一直站在“调用者”的视角：写 DSL、写 prompt、选后端。从这一课起视角翻转——
我们要钻进运行时<strong>内部</strong>，看请求进来之后究竟发生了什么。而一切的起点，就是这两个入口：它们是<strong>外部世界与引擎内部的边界</strong>。
搞清楚边界在哪、有几种形态、它们背后是不是同一套机器，你才不会在后面十几课的组件细节里迷失方向。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象一家餐馆有两种点单方式。一种是<strong>堂食柜台</strong>：你直接走到窗口、隔着台面冲后厨喊单，没有电话、没有中间人——
  这就是<strong>离线 Engine</strong>，在同一个进程里直接调用，开销最低。另一种是<strong>电话外卖热线</strong>：你拨号、报地址、
  话务员替你把单子转给后厨——这就是<strong>在线 server</strong>，多了一层网络与协议，但任何人、任何客户端都能远程下单。
  关键是：<strong>后厨只有一个</strong>。无论你走柜台还是打电话，做菜的厨房一模一样——server 不过是给同一个 Engine
  套了个“接电话的前台”。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  SGLang 提供两条驱动运行时的路径，它们都继承自同一个抽象基类 <span class="inline">EngineBase</span>：
  <strong>离线 <span class="mono">Engine</span></strong> 是纯 Python 对象，<span class="mono">llm.generate(...)</span> 直接返回结果，
  适合批量任务、评测、以及 RL rollout（训练器进程内直接调）；<strong>在线 server</strong> 用
  <span class="mono">launch_server()</span> 起一个 FastAPI/uvicorn 应用，把一个 Engine 包起来，对外暴露原生
  <span class="mono">POST /generate</span> 和 OpenAI 兼容路由。<strong>底层运行时完全相同</strong>：server 只是 Engine 的 HTTP 外壳。
</div>

<p>为什么要专门设两扇门，而不是只留一个 HTTP server？因为这两类需求<strong>天差地别</strong>。一类是“把推理<strong>嵌进</strong>我的程序里”——
训练循环、评测脚本、数据流水线，它们本来就是 Python 进程，最不想要的就是再起一个网络服务、自己当客户端去打 HTTP，平白多出延迟和运维。
另一类是“把推理<strong>开放</strong>给外界”——要面对未知数量的并发客户端、跨机器、跨语言，这时 HTTP 才是刚需。SGLang 没有逼你二选一，
而是用 <span class="mono">EngineBase</span> 把<strong>同一套运行时核心</strong>包装成两种入口，各取所需。理解这一点，本课其余内容都是它的展开。</p>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="两种入口对比：左边离线 Engine 在你的 Python 进程内直接调用 engine.generate()，无网络；右边在线 HTTP Server 用 FastAPI 把同一个 Engine 包起来，对外暴露 /generate 与 /v1 路由；两者底层是同一套运行时核心">
    <line x1="380" y1="20" x2="380" y2="196" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="28" y="36" style="font-weight:700;fill:var(--muted)">离线 Engine · 进程内</text>
    <rect x="34" y="52" width="300" height="84" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="184" y="80" text-anchor="middle">你的 Python 程序（评测 / RL）</text>
    <text x="184" y="104" text-anchor="middle" class="mono" style="font-size:13px">engine.generate()</text>
    <text x="184" y="124" text-anchor="middle" style="fill:var(--muted);font-size:12px">直接调用，无 HTTP、无网络</text>
    <path d="M 184 136 L 184 204" style="fill:none;stroke:var(--accent);stroke-width:1.5"/>
    <polygon points="184,210 177,196 191,196" style="fill:var(--accent)"/>
    <text x="410" y="36" style="font-weight:700;fill:var(--muted)">在线 HTTP Server</text>
    <rect x="408" y="50" width="100" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="458" y="70" text-anchor="middle" class="mono" style="font-size:12px">curl</text>
    <rect x="520" y="50" width="100" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="570" y="70" text-anchor="middle" class="mono" style="font-size:12px">Python</text>
    <rect x="632" y="50" width="100" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="682" y="70" text-anchor="middle" style="font-size:12px">其他语言</text>
    <path d="M 458 80 L 520 112" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="520,116 513,102 527,102" style="fill:var(--blue)"/>
    <path d="M 570 80 L 570 112" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="570,116 563,102 577,102" style="fill:var(--blue)"/>
    <path d="M 682 80 L 620 112" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="620,116 613,102 627,102" style="fill:var(--blue)"/>
    <rect x="470" y="116" width="200" height="56" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="570" y="140" text-anchor="middle">FastAPI · launch_server</text>
    <text x="570" y="160" text-anchor="middle" class="mono" style="font-size:11px">POST /generate · /v1/...</text>
    <path d="M 570 172 L 520 204" style="fill:none;stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="520,210 513,196 527,196" style="fill:var(--teal)"/>
    <rect x="120" y="210" width="520" height="66" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="240" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">同一套运行时核心 · 同一个 Engine</text>
    <text x="380" y="262" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--muted)">Tokenizer · Scheduler · Detokenizer</text>
  </svg>
  <div class="figcap"><b>图 1 · 两种入口：Engine vs HTTP Server</b> — 左边离线 <span class="mono">Engine</span> 在你的 Python 进程内直接调用 <span class="mono">engine.generate()</span>，无网络；右边在线 server 用 FastAPI 把同一个 Engine 包起来，对外暴露 <span class="mono">/generate</span> 与 <span class="mono">/v1/...</span>。同一个内核，两层外皮。</div>
</div>

<h2>离线 Engine：一个纯进程内的 Python API</h2>
<p>离线 <span class="inline">Engine</span> 是最“素”的入口：没有 HTTP、没有端口、没有网络。你在自己的 Python 脚本里
<span class="mono">import sglang as sgl</span>，构造一个 <span class="mono">sgl.Engine(model_path=...)</span>，然后调用
<span class="mono">llm.generate(prompts, sampling_params)</span>，结果<strong>当场以 Python 字典/列表返回</strong>。整个过程跑在
<strong>同一个进程</strong>里（准确说，主进程里还有 TokenizerManager，调度器与反分词器是子进程，这点马上展开）。
因为少了序列化、HTTP 解析、网络往返这几层开销，Engine 是<strong>延迟最低、最适合嵌入</strong>的方式。</p>

<p>这里要澄清一个常见误解：“离线”并<strong>不</strong>意味着“单进程单线程”。即便是离线 Engine，内部依然是第 2 课讲过的
<strong>三进程模型</strong>——主进程里跑 TokenizerManager，调度器（Scheduler）和反分词器（DetokenizerManager）各自是独立子进程，
彼此用 ZMQ 通信。所谓“纯进程内 Python API”，指的是<strong>你的调用代码</strong>和 Engine 在同一进程，<span class="mono">generate()</span>
是一次普通的 Python 函数调用、直接拿到返回值，<strong>而不需要经过 HTTP 这一圈</strong>。换句话说，离线/在线的区别只在“最外面那层门”，
不在引擎内部的并行结构。</p>

<p>还有一点值得点明：离线 Engine 的返回是<strong>同步</strong>的——<span class="mono">generate(prompts, ...)</span> 传入一批 prompt，
它内部会把这批请求送进调度器、连续批处理、再把每条的结果收齐，<strong>一次性返回一个列表</strong>。你也可以传单条 prompt 拿单条结果。
正因为接口这么“朴素”，它和 NumPy、PyTorch 那样的库一样好嵌入：没有 server 要启动、没有端口要协调、没有客户端要写，
一个 import、一次构造、一次调用，就拿到了 SGLang 全部的运行时能力。这正是批处理与 RL 场景偏爱它的根本原因。</p>

<div class="cols">
  <div class="col"><h4>离线 Engine</h4><p><span class="mono">sgl.Engine(model_path=...)</span> + <span class="mono">.generate(...)</span>。
  纯进程内调用、<strong>无 HTTP</strong>、无网络往返。适合<strong>批量推理、离线评测、RL rollout</strong>（训练器进程内直接调，开销最低）。
  缺点：只能在<strong>本进程的 Python</strong> 里用，跨机/跨语言无能为力。</p></div>
  <div class="col"><h4>在线 server</h4><p><span class="mono">launch_server(server_args)</span> 起 FastAPI 应用，暴露
  <strong>HTTP 接口</strong>。适合<strong>生产部署、多客户端、跨语言</strong>访问——任何能发 HTTP 的客户端都能接入。
  代价：多一层网络与序列化开销，要管端口、并发与鉴权。</p></div>
</div>

<h2>在线 server：给 Engine 套一层 HTTP</h2>
<p>在线 server 用一条命令就能拉起：<span class="mono">python -m sglang.launch_server --model-path ...</span>。它在内部
<strong>构造一个 Engine</strong>，再用 FastAPI/uvicorn 把它包起来，对外开放两类路由：一是 SGLang 的<strong>原生接口</strong>
<span class="mono">POST /generate</span>；二是一整套 <strong>OpenAI 兼容路由</strong>，如 <span class="mono">/v1/chat/completions</span>、
<span class="mono">/v1/completions</span> 等（第 15 课专门讲兼容层）。于是一个 HTTP 请求进来，会被翻译成对内部 Engine 的调用，
再把结果（可流式）写回响应。下面这张表把三类入口和“谁在用”对齐：</p>

<p>为什么生产环境几乎都用 server 而不是 Engine？因为真实部署面对的是<strong>很多并发客户端、来自不同机器甚至不同语言</strong>的访问。
HTTP 是天然的跨语言、跨网络协议：Python、Go、前端 JavaScript、甚至 curl 都能直接打过来。更重要的是 OpenAI 兼容层让你的私有部署
<strong>无缝接入现成生态</strong>——别人的 OpenAI SDK、LangChain 只要把 <span class="mono">base_url</span> 一改就能用（这正是第 12 课讲的
“OpenAI 客户端 → SGLang 服务器”那个方向）。这些都是离线 Engine 给不了的：它只活在<strong>你自己那一个 Python 进程</strong>里。</p>

<table class="t">
  <tr><th>入口</th><th>形态</th><th>典型使用者</th></tr>
  <tr><td class="mono">sgl.Engine().generate()</td><td>进程内 Python 调用，无 HTTP</td><td>批处理脚本、评测、RL 训练器（第 51 课）</td></tr>
  <tr><td class="mono">POST /generate</td><td>SGLang 原生 HTTP 接口</td><td>前端 RuntimeEndpoint、自定义客户端</td></tr>
  <tr><td class="mono">/v1/chat/completions</td><td>OpenAI 兼容 HTTP 路由</td><td>OpenAI SDK / LangChain 等现成生态（第 15 课）</td></tr>
</table>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/http_server.py ::generate_request</span><span class="ln">HTTP /generate 路由：交给 TokenizerManager，再流式/一次性返回</span></div>
  <pre><span class="kw">@app.post</span>(<span class="st">"/generate"</span>)
<span class="kw">async def</span> generate_request(obj: GenerateReqInput, request: Request):
    <span class="cm"># HTTP 入口：把解析好的请求交给 TokenizerManager，</span>
    <span class="cm"># 然后以 SSE 流式回传，或一次性返回一个 JSON 响应</span>
    ...</pre>
</div>

<p>两个最常见的在线调用例子。其一，直接打 SGLang 原生接口：
<span class="mono">curl localhost:30000/generate -d '{"text": "中国的首都是", "sampling_params": {"max_new_tokens": 16}}'</span>。
其二，走 OpenAI 兼容路由，把请求 POST 到 <span class="mono">/v1/chat/completions</span>（body 里带 <span class="mono">messages</span> 与
<span class="mono">model</span>）——现成的 OpenAI SDK 只要把 <span class="mono">base_url</span> 改成你的服务地址就能直接命中（第 15 课展开）。</p>

<h2>同一套运行时：server 只是外壳</h2>
<p>这是本课<strong>最重要</strong>的一句话：无论从哪扇门进来，<strong>底层运行时是同一套</strong>。HTTP 请求经过 FastAPI 应用后，
最终落到一个 Engine 上；而这个 Engine 内部，就是我们后面要逐课拆解的那几个组件——<strong>TokenizerManager</strong>（第 14 课，
负责分词与请求登记）、<strong>Scheduler</strong>（第 18 课，组批与调度）、<strong>DetokenizerManager</strong>（反分词与流式回传）。
也就是说，从第 14 课起学的所有东西，<strong>离线 Engine 和在线 server 完全共用</strong>，你只需理解一遍。</p>

<div class="flow">
  <div class="node"><div class="nt">HTTP 请求</div><div class="nd">/generate · /v1/chat/...</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">FastAPI 应用</div><div class="nd">launch_server 包的壳</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Engine</div><div class="nd">共享运行时核心</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">三大组件</div><div class="nd">Tokenizer · Scheduler · Detokenizer</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="在线服务的请求扇出：多个并发客户端各发一个 HTTP 请求，汇聚到同一个 HTTP server（FastAPI 路由），server 把请求交给 TokenizerManager，再送往持有 GPU 的 Scheduler；一个 server 把众多并发请求漏斗式汇入共享引擎">
    <text x="28" y="30" style="font-weight:700;fill:var(--muted)">多个并发客户端 → 一个 server → 共享引擎</text>
    <rect x="20" y="52" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="76" text-anchor="middle">客户端 ①</text>
    <rect x="20" y="100" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="124" text-anchor="middle">客户端 ②</text>
    <rect x="20" y="148" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="172" text-anchor="middle">客户端 ③</text>
    <rect x="20" y="196" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="220" text-anchor="middle">客户端 ④</text>
    <path d="M 160 71 L 246 148" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <path d="M 160 119 L 246 150" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <path d="M 160 167 L 246 152" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <path d="M 160 215 L 246 154" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <polygon points="252,150 238,143 238,157" style="fill:var(--line)"/>
    <rect x="252" y="110" width="150" height="80" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="327" y="140" text-anchor="middle">HTTP Server</text>
    <text x="327" y="160" text-anchor="middle" style="fill:var(--muted);font-size:12px">FastAPI 路由</text>
    <text x="327" y="180" text-anchor="middle" class="mono" style="font-size:11px">/generate · /v1</text>
    <path d="M 402 150 L 452 150" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <polygon points="454,150 440,143 440,157" style="fill:var(--line)"/>
    <rect x="454" y="122" width="140" height="56" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="524" y="146" text-anchor="middle">TokenizerManager</text>
    <text x="524" y="166" text-anchor="middle" style="fill:var(--muted);font-size:12px">分词 + 登记</text>
    <path d="M 594 150 L 640 150" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <polygon points="642,150 628,143 628,157" style="fill:var(--line)"/>
    <rect x="642" y="122" width="100" height="56" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="692" y="146" text-anchor="middle">Scheduler</text>
    <text x="692" y="166" text-anchor="middle" style="fill:var(--muted);font-size:12px">GPU 前向</text>
  </svg>
  <div class="figcap"><b>图 2 · 在线服务的请求扇出</b> — 多个并发客户端各发一个 HTTP 请求，汇聚到同一个 <span class="mono">HTTP server</span>（FastAPI 路由）；server 把请求交给 <span class="mono">TokenizerManager</span>，再送往持有 GPU 的 <span class="mono">Scheduler</span>。一个 server 把众多请求漏斗式汇入共享引擎。</div>
</div>

<p>注意离线 Engine 走的是<strong>同一条链路的后半段</strong>：它直接构造 Engine、跳过了最左边的“HTTP 请求 + FastAPI 应用”两格，
其余完全一致。这正是为什么我们说<strong>server = Engine + 一层 HTTP 外壳</strong>——server 没有<strong>重新实现</strong>任何运行时逻辑，
它只负责“接电话、转单、回话”。</p>

<p>把这层关系想透，对学习路线有一个极大的好处：<strong>你不必为“离线”和“在线”各学一遍内部机制</strong>。第 14 课的
TokenizerManager 怎么分词与登记请求、第 18 课的 Scheduler 怎么连续批处理、反分词器怎么增量流式回传——这些<strong>对两扇门是同一套</strong>。
server 端额外要操心的只是 HTTP 层面的事：路由、并发、鉴权、SSE 流式响应的封装。所以读完 Part 4 之后，无论你将来用 Engine 嵌进训练框架，
还是用 server 对外提供服务，<strong>底层那套运行时你都已经懂了</strong>。这也是 SGLang 用 <span class="mono">EngineBase</span> 抽象基类
把两条路径统一起来的工程意图：把“入口形态”和“运行时核心”解耦，核心只写一遍、复用到底。</p>

<h2>Engine.__init__ 启动三进程</h2>
<p>回忆第 2 课的“三进程模型”：SGLang 把<strong>分词、调度+前向、反分词</strong>拆到不同进程，用 ZMQ 做进程间通信。
而<strong>拉起这三个进程的，正是 <span class="inline">Engine.__init__</span></strong>（server 也是先构造 Engine，所以同样经过这一步）。
构造一个 Engine，就等于把整套运行时“点火”：</p>

<p>为什么要拆成三个进程、而不是塞进一个进程的三个线程？根子在 Python 的全局解释器锁（GIL）：分词、调度、反分词若挤在同一进程，
会互相抢锁、串行排队。拆成独立进程后，三者能<strong>真正并行</strong>、用 ZMQ 异步传消息，形成一条流水线——这也是第 2 课反复强调的设计动机。
你只需记住：构造 Engine 这一下，背后是“起进程 + 连通道 + 载模型”的一整套点火流程，而非简单的对象初始化。这套流程一旦跑完，引擎就进入随时待命、可以接单的状态。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>解析 ServerArgs</h4><p>把 <span class="mono">model_path</span> 等关键字参数收成一个 <span class="mono">ServerArgs</span>，决定并行度、显存、后端等一切配置。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>启动子进程</h4><p>调用 <span class="mono">_launch_subprocesses()</span>：在<strong>主进程</strong>内建 TokenizerManager，并<strong>各起一个子进程</strong>跑 Scheduler 与 DetokenizerManager。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>建立 ZMQ 通道</h4><p>为进程间通信建好 socket，把主进程与两个子进程<strong>连成一条流水线</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>就绪等单</h4><p>三进程各就各位，<span class="mono">generate()</span> 一调用，请求就能沿链路跑起来。</p></div></div>
</div>

<p>这一步之所以重要，是因为它揭示了“构造一个 Engine 的代价”：<span class="mono">sgl.Engine(...)</span> 不是一行轻量赋值，
而是<strong>把整套运行时点火</strong>——拉起子进程、加载模型权重、分配 KV 缓存、建立 ZMQ 通道，全在 <span class="mono">__init__</span> 里完成。
所以 Engine 对象应当<strong>构造一次、长期复用</strong>，而不是每来一个请求就 new 一个。在线 server 同样是先构造一个 Engine、长期持有，
再用 FastAPI 把它对外暴露——这又一次印证了“server = 长期持有的 Engine + HTTP 壳”。下面这段就是 Engine 类的真实定义与点火逻辑：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::Engine</span><span class="ln">构造 Engine = 点火整套运行时</span></div>
  <pre><span class="kw">class</span> Engine(EngineScoreMixin, EngineBase):
    <span class="cm"># 推理引擎的入口，由三个组件构成：</span>
    <span class="cm"># 1. TokenizerManager  分词并把请求发给调度器（主进程）</span>
    <span class="cm"># 2. Scheduler  子进程：收请求、组批、前向、把 token 发给反分词器</span>
    <span class="cm"># 3. DetokenizerManager  子进程：反分词并把结果送回 TokenizerManager</span>

    <span class="kw">def</span> __init__(self, **kwargs):
        server_args = self.server_args_class(**kwargs)   <span class="cm"># 解析配置</span>
        <span class="cm"># 启动子进程：主进程内的 TokenizerManager + Scheduler/Detokenizer 子进程</span>
        (tokenizer_manager, template_manager, port_args,
         scheduler_init_result, subprocess_watchdog) = self._launch_subprocesses(
            server_args=server_args, ...)
        self.tokenizer_manager = tokenizer_manager       <span class="cm"># 之后 generate() 走它入口</span></pre>
</div>

<h2>什么时候用哪个</h2>
<p>选择其实很直观：<strong>要不要跨进程/跨网络？</strong>如果你的代码就在<strong>同一个 Python 进程</strong>里、追求最低开销——比如
批量跑评测集、做离线数据生成、或在 RL 训练里让训练器<strong>进程内直接 rollout</strong>（第 51 课），那就用<strong>离线 Engine</strong>。
如果你要把模型<strong>作为服务</strong>对外提供、面对很多并发客户端、或客户端是别的语言/别的机器，那就用<strong>在线 server</strong>，
让它的 HTTP 与 OpenAI 兼容层替你接住整个生态。记住：两者性能核心一致，差别只在<strong>那层 HTTP 要不要</strong>。</p>

<p>再举两个具体例子把这个判断落地。例子一：你要在一万条评测样本上批量跑模型、统计准确率。这是典型的离线批处理——没有外部客户端、
追求吞吐与最低开销，<strong>离线 Engine</strong> 是首选，一段 Python 脚本里 <span class="mono">generate</span> 一把全收。例子二：你要上线一个
面向公网的聊天产品，前端是浏览器、移动端可能是 Swift/Kotlin，还想让合作方用现成的 OpenAI SDK 接入。这是典型的在线服务——多并发、
跨语言、要 OpenAI 兼容，<strong>在线 server</strong> 当仁不让。值得强调的是，从评测脚本切到生产服务，你<strong>不需要重学引擎内部</strong>，
只是把“入口”从一个 Python 对象换成一个 HTTP 端点而已——这正是本课想让你建立的最终直觉。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>两扇门</strong>：离线 <span class="mono">Engine</span>（纯进程内 Python API，无 HTTP）与在线 server（<span class="mono">launch_server</span> 起的 FastAPI 应用），都继承自 <span class="mono">EngineBase</span>。</li>
    <li><strong>离线 Engine</strong>：<span class="mono">sgl.Engine(model_path=...).generate(prompts, sampling_params)</span>，开销最低，适合批处理、评测、RL rollout（训练器进程内直接调，第 51 课）。</li>
    <li><strong>在线 server</strong>：<span class="mono">python -m sglang.launch_server</span> 暴露原生 <span class="mono">POST /generate</span> 与 OpenAI 兼容路由（<span class="mono">/v1/chat/completions</span>，第 15 课），适合生产/多客户端/跨语言。</li>
    <li><strong>同一套运行时</strong>：server 不重写任何逻辑，只是 Engine 的 HTTP 外壳；从第 14 课起的 TokenizerManager（第 14 课）、Scheduler（第 18 课）、Detokenizer 两者共用。</li>
    <li><strong>点火</strong>：<span class="mono">Engine.__init__</span> 解析 ServerArgs 并 <span class="mono">_launch_subprocesses</span> 拉起三进程（第 2 课的三进程模型）。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Welcome to Part 4 — we finally step from the front-end DSL through the door of the <strong>runtime</strong>. This lesson
answers the most basic question: through <strong>what entry point</strong> does your program feed requests into SGLang's
engine? There are exactly two answers: the offline <strong><span class="inline">Engine</span></strong> (a pure in-process
Python API) and the online <strong>HTTP server</strong> (a FastAPI app). Understanding these two doors — and that they
<strong>share one runtime underneath</strong> — is the key to every later lesson.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture a restaurant with two ways to order. One is the <strong>walk-in counter</strong>: you step to the window and call
  your order straight to the kitchen — no phone, no middleman. That's the <strong>offline Engine</strong>: an in-process call
  with the lowest overhead. The other is the <strong>phone-order hotline</strong>: you dial, give your address, and an
  operator relays the ticket to the kitchen. That's the <strong>online server</strong>: a layer of network and protocol, but
  anyone, any client, can order remotely. The point: <strong>there is only one kitchen</strong>. Counter or phone, the
  kitchen is identical — the server is just a "front desk that answers calls" bolted onto the same Engine.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  SGLang offers two paths to drive the runtime, both inheriting from the same abstract base <span class="inline">EngineBase</span>:
  the <strong>offline <span class="mono">Engine</span></strong> is a plain Python object whose <span class="mono">llm.generate(...)</span>
  returns results directly — ideal for batch jobs, evaluation, and RL rollout (the trainer calls it in-process); the
  <strong>online server</strong> uses <span class="mono">launch_server()</span> to start a FastAPI/uvicorn app wrapping an Engine,
  exposing the native <span class="mono">POST /generate</span> plus OpenAI-compatible routes. <strong>The runtime underneath is
  identical</strong>: the server is just an HTTP wrapper around the Engine.
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="Two entry points compared: on the left the offline Engine calls engine.generate() directly inside your Python process with no network; on the right the online HTTP Server wraps the same Engine with FastAPI and exposes /generate and /v1 routes; both share one runtime core underneath">
    <line x1="380" y1="20" x2="380" y2="196" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="28" y="36" style="font-weight:700;fill:var(--muted)">Offline Engine · in-process</text>
    <rect x="34" y="52" width="300" height="84" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="184" y="80" text-anchor="middle">Your Python program (eval / RL)</text>
    <text x="184" y="104" text-anchor="middle" class="mono" style="font-size:13px">engine.generate()</text>
    <text x="184" y="124" text-anchor="middle" style="fill:var(--muted);font-size:12px">direct call, no HTTP, no network</text>
    <path d="M 184 136 L 184 204" style="fill:none;stroke:var(--accent);stroke-width:1.5"/>
    <polygon points="184,210 177,196 191,196" style="fill:var(--accent)"/>
    <text x="410" y="36" style="font-weight:700;fill:var(--muted)">Online HTTP Server</text>
    <rect x="408" y="50" width="100" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="458" y="70" text-anchor="middle" class="mono" style="font-size:12px">curl</text>
    <rect x="520" y="50" width="100" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="570" y="70" text-anchor="middle" class="mono" style="font-size:12px">Python</text>
    <rect x="632" y="50" width="100" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="682" y="70" text-anchor="middle" style="font-size:12px">other langs</text>
    <path d="M 458 80 L 520 112" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="520,116 513,102 527,102" style="fill:var(--blue)"/>
    <path d="M 570 80 L 570 112" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="570,116 563,102 577,102" style="fill:var(--blue)"/>
    <path d="M 682 80 L 620 112" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="620,116 613,102 627,102" style="fill:var(--blue)"/>
    <rect x="470" y="116" width="200" height="56" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="570" y="140" text-anchor="middle">FastAPI · launch_server</text>
    <text x="570" y="160" text-anchor="middle" class="mono" style="font-size:11px">POST /generate · /v1/...</text>
    <path d="M 570 172 L 520 204" style="fill:none;stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="520,210 513,196 527,196" style="fill:var(--teal)"/>
    <rect x="120" y="210" width="520" height="66" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="240" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">One shared runtime core · one Engine</text>
    <text x="380" y="262" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--muted)">Tokenizer · Scheduler · Detokenizer</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Two entry points: Engine vs HTTP Server</b> — on the left the offline <span class="mono">Engine</span> calls <span class="mono">engine.generate()</span> directly inside your Python process, no network; on the right the online server wraps the same Engine with FastAPI, exposing <span class="mono">/generate</span> and <span class="mono">/v1/...</span>. One core, two skins.</div>
</div>

<h2>Offline Engine: a pure in-process Python API</h2>
<p>The offline <span class="inline">Engine</span> is the barest entry: no HTTP, no port, no network. In your own Python script
you <span class="mono">import sglang as sgl</span>, build a <span class="mono">sgl.Engine(model_path=...)</span>, then call
<span class="mono">llm.generate(prompts, sampling_params)</span> — results come back <strong>right away as Python dicts/lists</strong>.
It all runs in <strong>one process</strong> (precisely: the main process also holds the TokenizerManager, with the scheduler and
detokenizer as subprocesses — more on that soon). With no serialization, HTTP parsing, or network round-trip, the Engine is the
<strong>lowest-latency, most embeddable</strong> way in.</p>

<div class="cols">
  <div class="col"><h4>Offline Engine</h4><p><span class="mono">sgl.Engine(model_path=...)</span> + <span class="mono">.generate(...)</span>.
  Pure in-process call, <strong>no HTTP</strong>, no network round-trip. Best for <strong>batch inference, offline eval, RL rollout</strong>
  (the trainer calls it in-process, lowest overhead). Downside: usable only from <strong>Python in this process</strong> — no cross-machine/cross-language.</p></div>
  <div class="col"><h4>Online server</h4><p><span class="mono">launch_server(server_args)</span> starts a FastAPI app exposing an
  <strong>HTTP interface</strong>. Best for <strong>production serving, multi-client, language-agnostic</strong> access — any HTTP
  client connects. Cost: an extra network + serialization layer, plus ports, concurrency, and auth to manage.</p></div>
</div>

<h2>Online server: wrap the Engine in HTTP</h2>
<p>The online server starts with one command: <span class="mono">python -m sglang.launch_server --model-path ...</span>. Internally
it <strong>builds an Engine</strong>, then wraps it with FastAPI/uvicorn, exposing two kinds of routes: SGLang's <strong>native</strong>
<span class="mono">POST /generate</span>, and a full set of <strong>OpenAI-compatible routes</strong> like
<span class="mono">/v1/chat/completions</span> and <span class="mono">/v1/completions</span> (Lesson 15 covers the compat layer). So an
incoming HTTP request is translated into a call on the inner Engine, and the result (optionally streamed) is written back. The table
aligns the three entry points with "who uses each":</p>

<table class="t">
  <tr><th>Entry</th><th>Shape</th><th>Typical user</th></tr>
  <tr><td class="mono">sgl.Engine().generate()</td><td>In-process Python call, no HTTP</td><td>Batch scripts, eval, RL trainer (Lesson 51)</td></tr>
  <tr><td class="mono">POST /generate</td><td>SGLang's native HTTP interface</td><td>Front-end RuntimeEndpoint, custom clients</td></tr>
  <tr><td class="mono">/v1/chat/completions</td><td>OpenAI-compatible HTTP route</td><td>OpenAI SDK / LangChain ecosystem (Lesson 15)</td></tr>
</table>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/http_server.py ::generate_request</span><span class="ln">the HTTP /generate route: hand to TokenizerManager, then stream or return once</span></div>
  <pre><span class="kw">@app.post</span>(<span class="st">"/generate"</span>)
<span class="kw">async def</span> generate_request(obj: GenerateReqInput, request: Request):
    <span class="cm"># the HTTP entry: pass the parsed request to the TokenizerManager,</span>
    <span class="cm"># then stream results back (SSE) or return one JSON response</span>
    ...</pre>
</div>

<p>Two of the most common online calls. First, hit SGLang's native route directly:
<span class="mono">curl localhost:30000/generate -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16}}'</span>.
Second, use the OpenAI-compatible route by POSTing to <span class="mono">/v1/chat/completions</span> (with <span class="mono">messages</span> and
<span class="mono">model</span> in the body) — an off-the-shelf OpenAI SDK just points its <span class="mono">base_url</span> at your server to hit it (Lesson 15).</p>

<h2>One runtime: the server is just a shell</h2>
<p>This is the lesson's <strong>most important</strong> sentence: whichever door you enter, <strong>the runtime underneath is the same</strong>.
After passing the FastAPI app, an HTTP request lands on an Engine; and inside that Engine are exactly the components we'll dissect lesson
by lesson — <strong>TokenizerManager</strong> (Lesson 14, tokenizing and registering requests), <strong>Scheduler</strong> (Lesson 18,
batching and scheduling), <strong>DetokenizerManager</strong> (detokenizing and streaming back). So everything from Lesson 14 onward is
<strong>shared by both offline Engine and online server</strong> — you only learn it once.</p>

<div class="flow">
  <div class="node"><div class="nt">HTTP request</div><div class="nd">/generate · /v1/chat/...</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">FastAPI app</div><div class="nd">shell from launch_server</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Engine</div><div class="nd">shared runtime core</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Three components</div><div class="nd">Tokenizer · Scheduler · Detokenizer</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="Online request fan-out: many concurrent clients each send one HTTP request, funneling into a single HTTP server (FastAPI route); the server hands the request to the TokenizerManager, then on to the GPU-holding Scheduler; one server funnels many concurrent requests into the shared engine">
    <text x="28" y="30" style="font-weight:700;fill:var(--muted)">Many concurrent clients → one server → shared engine</text>
    <rect x="20" y="52" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="76" text-anchor="middle">client ①</text>
    <rect x="20" y="100" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="124" text-anchor="middle">client ②</text>
    <rect x="20" y="148" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="172" text-anchor="middle">client ③</text>
    <rect x="20" y="196" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="220" text-anchor="middle">client ④</text>
    <path d="M 160 71 L 246 148" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <path d="M 160 119 L 246 150" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <path d="M 160 167 L 246 152" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <path d="M 160 215 L 246 154" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <polygon points="252,150 238,143 238,157" style="fill:var(--line)"/>
    <rect x="252" y="110" width="150" height="80" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="327" y="140" text-anchor="middle">HTTP Server</text>
    <text x="327" y="160" text-anchor="middle" style="fill:var(--muted);font-size:12px">FastAPI route</text>
    <text x="327" y="180" text-anchor="middle" class="mono" style="font-size:11px">/generate · /v1</text>
    <path d="M 402 150 L 452 150" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <polygon points="454,150 440,143 440,157" style="fill:var(--line)"/>
    <rect x="454" y="122" width="140" height="56" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="524" y="146" text-anchor="middle">TokenizerManager</text>
    <text x="524" y="166" text-anchor="middle" style="fill:var(--muted);font-size:12px">tokenize + register</text>
    <path d="M 594 150 L 640 150" style="fill:none;stroke:var(--line);stroke-width:1.5"/>
    <polygon points="642,150 628,143 628,157" style="fill:var(--line)"/>
    <rect x="642" y="122" width="100" height="56" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="692" y="146" text-anchor="middle">Scheduler</text>
    <text x="692" y="166" text-anchor="middle" style="fill:var(--muted);font-size:12px">GPU forward</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Online request fan-out</b> — many concurrent clients each send one HTTP request, funneling into a single <span class="mono">HTTP server</span> (FastAPI route); the server hands the request to the <span class="mono">TokenizerManager</span>, then on to the GPU-holding <span class="mono">Scheduler</span>. One server funnels many requests into the shared engine.</div>
</div>

<p>Note the offline Engine walks the <strong>same chain's back half</strong>: it builds the Engine directly, skipping the leftmost two
cells ("HTTP request + FastAPI app"), and the rest is identical. That's exactly why we say <strong>server = Engine + an HTTP shell</strong> —
the server <strong>reimplements no</strong> runtime logic; it only "answers the call, relays the ticket, replies."</p>

<h2>Engine.__init__ launches three processes</h2>
<p>Recall Lesson 2's "three-process model": SGLang splits <strong>tokenizing, scheduling+forward, detokenizing</strong> across processes,
talking over ZMQ. And <strong>what spins those processes up is <span class="inline">Engine.__init__</span></strong> (the server builds an
Engine first, so it goes through this too). Constructing an Engine is igniting the whole runtime:</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Parse ServerArgs</h4><p>Collect kwargs like <span class="mono">model_path</span> into a <span class="mono">ServerArgs</span> deciding parallelism, memory, backend — every config.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Launch subprocesses</h4><p>Call <span class="mono">_launch_subprocesses()</span>: build TokenizerManager in the <strong>main process</strong>, and start <strong>one subprocess each</strong> for Scheduler and DetokenizerManager.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Wire up ZMQ</h4><p>Create the sockets for inter-process communication, stringing the main process and the two subprocesses into <strong>one pipeline</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Ready to serve</h4><p>All three processes in place; the moment <span class="mono">generate()</span> is called, a request runs down the chain.</p></div></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::Engine</span><span class="ln">building an Engine = igniting the whole runtime</span></div>
  <pre><span class="kw">class</span> Engine(EngineScoreMixin, EngineBase):
    <span class="cm"># Entry point to the inference engine, made of three components:</span>
    <span class="cm"># 1. TokenizerManager  tokenizes and sends requests to the scheduler (main proc)</span>
    <span class="cm"># 2. Scheduler  subprocess: receives, batches, forwards, sends tokens to detokenizer</span>
    <span class="cm"># 3. DetokenizerManager  subprocess: detokenizes and sends results back</span>

    <span class="kw">def</span> __init__(self, **kwargs):
        server_args = self.server_args_class(**kwargs)   <span class="cm"># parse config</span>
        <span class="cm"># launch subprocesses: TokenizerManager (main) + Scheduler/Detokenizer (subprocs)</span>
        (tokenizer_manager, template_manager, port_args,
         scheduler_init_result, subprocess_watchdog) = self._launch_subprocesses(
            server_args=server_args, ...)
        self.tokenizer_manager = tokenizer_manager       <span class="cm"># generate() goes through it</span></pre>
</div>

<h2>When to use which</h2>
<p>The choice is intuitive: <strong>do you need to cross processes/network?</strong> If your code is in the <strong>same Python process</strong>
and wants the lowest overhead — running an eval set in batch, generating offline data, or letting an RL trainer <strong>roll out in-process</strong>
(Lesson 51) — use the <strong>offline Engine</strong>. If you want to serve the model <strong>as a service</strong> to many concurrent clients, or
clients in another language/machine, use the <strong>online server</strong> and let its HTTP and OpenAI-compat layer catch the whole ecosystem.
Remember: their performance core is identical; the only difference is <strong>whether you want that HTTP layer</strong>.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Two doors</strong>: offline <span class="mono">Engine</span> (pure in-process Python API, no HTTP) and online server (a FastAPI app from <span class="mono">launch_server</span>), both inheriting <span class="mono">EngineBase</span>.</li>
    <li><strong>Offline Engine</strong>: <span class="mono">sgl.Engine(model_path=...).generate(prompts, sampling_params)</span>, lowest overhead, best for batch, eval, RL rollout (trainer calls in-process, Lesson 51).</li>
    <li><strong>Online server</strong>: <span class="mono">python -m sglang.launch_server</span> exposes native <span class="mono">POST /generate</span> plus OpenAI-compatible routes (<span class="mono">/v1/chat/completions</span>, Lesson 15); best for production/multi-client/cross-language.</li>
    <li><strong>One runtime</strong>: the server rewrites no logic — just an HTTP shell over the Engine; from Lesson 14 on, TokenizerManager (Lesson 14), Scheduler (Lesson 18), Detokenizer are shared by both.</li>
    <li><strong>Ignition</strong>: <span class="mono">Engine.__init__</span> parses ServerArgs and <span class="mono">_launch_subprocesses</span> spins up the three processes (Lesson 2's three-process model).</li>
  </ul>
</div>
""",
}

LESSON_14 = {
    "zh": r"""
<p class="lead">
上一课我们说，无论走离线 <span class="inline">Engine</span> 还是在线 server，请求最终都落到<strong>同一套运行时</strong>上。
那么请求进入运行时之后，<strong>第一个</strong>接住它的组件是谁？答案就是本课的主角——<strong>TokenizerManager</strong>。
它是整个运行时的<strong>前门（front door）</strong>：跑在<strong>主进程</strong>里（和 HTTP server / Engine 同进程），
而<strong>不</strong>在那个吃显存的 GPU 子进程里。每一条请求都要先经过它分词、打包、登记，再被送往后面的调度器。
</p>

<p>为什么要先单独讲它？因为 TokenizerManager 是<strong>外部文本世界</strong>与<strong>内部 token 世界</strong>的翻译官，
也是<strong>异步并发的总枢纽</strong>。你发来的是一串字符串，引擎内部认的却是一串 token id；几百条请求同时涌入，
谁的回包该交还给谁——这些都由它统筹。看懂这扇前门怎么运转，后面第 16 课的 io_struct 消息、第 17 课的反分词与流式回传、
第 18 课的调度器才能串成一条完整的链路。</p>

<p>把这层关系想透很重要：它意味着<strong>整个运行时只有一个“说人话”的地方</strong>。再往后，所有组件——调度器、各种注意力后端、KV 缓存管理器——
都只跟 token id 打交道，谁也不需要懂 UTF-8、不需要懂哪个子词该怎么拼。TokenizerManager 把这层“语言学负担”一次性挡在最外层，
让内部组件保持<strong>纯数值、纯张量</strong>的干净世界。这也是为什么我们要在 Part 4 一开始、紧跟入口之后就讲它：
它是整条链路上<strong>唯一面向人类文本</strong>的关口，越早理解，后面学起来就越省力、越不容易迷路。</p>

<p>还有一个角度值得点明：TokenizerManager 不只是“分词器的薄包装”。<strong>分词器（tokenizer）</strong>本身只是个把字符串变成数字的工具，
而 <strong>TokenizerManager</strong> 是围绕它建起来的<strong>异步管理层</strong>——它持有 tokenizer、持有 ZMQ socket、持有那张 <span class="mono">rid_to_state</span> 表，
还跑着一个后台的接收循环不断从子进程取回包。换句话说，名字里的 “Manager”（管理器）才是重点：它<strong>管理</strong>的是“一条请求从文本进、到结果出”的<strong>完整生命周期与并发状态</strong>，
分词只是它开场要做的第一件事而已。带着这个区分去读源码，你就不会把它和普通的 HuggingFace tokenizer 混为一谈。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 TokenizerManager 想成酒店的<strong>前台接待员</strong>。客人（请求）走到前台，用<strong>口语</strong>说出需求（一串文本）；
  接待员把它<strong>翻译成一张标准工单</strong>——填上房型与要求（采样参数）、贴上一个<strong>预订号</strong>（rid），
  然后把工单<strong>送进后台办公室</strong>（调度器子进程）去真正安排。客人不必走进后厨，只在前台<strong>等叫号</strong>；
  接待员则<strong>同时招呼几十位客人</strong>，谁的结果出来了就准确无误地交还给谁。前台从不亲自做菜（不碰 GPU），
  它只负责<strong>翻译、登记、转单、等回话、交付</strong>——这正是 TokenizerManager 在干的事。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  TokenizerManager 活在<strong>主进程</strong>，是 CPU 侧的异步协调中心。它的核心方法 <span class="mono">generate_request</span>
  是一个 <span class="mono">async</span> 协程：拿到请求后，先把文本<strong>分词</strong>成 token id（纯 CPU 的字符串工作），
  构造好<strong>采样参数</strong>，打包成一个带唯一 <span class="mono">rid</span> 的 <span class="mono">TokenizedGenerateReqInput</span>（第 16 课），
  通过 <strong>ZMQ</strong> 发给调度器子进程；随后<strong>登记一份按 rid 索引的异步状态</strong>，并 <span class="mono">await</span>
  从反分词器（第 17 课）流回来的输出，逐段 <span class="mono">yield</span> 给调用方（HTTP 的 SSE 或 Engine）。
  之所以把分词/反分词与 GPU 循环<strong>隔在不同进程</strong>，是为了让 CPU 与 GPU <strong>重叠</strong>起来（零开销思想，第 21 课）。
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="分词流程：字符串 SGLang 很快 先被切成 SG Lang 很 快 四个子词，再各自映射成整数 token id 12 8801 332 1567">
    <text x="380" y="26" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:13px">文本 → 子词 → token id</text>
    <rect x="265" y="42" width="230" height="44" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="380" y="70" text-anchor="middle" class="mono" style="fill:var(--ink);font-size:15px">SGLang 很快</text>
    <text x="432" y="108" text-anchor="middle" style="fill:var(--muted);font-size:10px">分词 tokenize</text>
    <line x1="380" y1="86" x2="380" y2="122" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="375,116 385,116 380,124" style="fill:var(--muted)"/>
    <rect x="90"  y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="240" y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="390" y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="540" y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="155" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"SG"</text>
    <text x="305" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"Lang"</text>
    <text x="455" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"很"</text>
    <text x="605" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"快"</text>
    <text x="432" y="196" text-anchor="middle" style="fill:var(--muted);font-size:10px">查词表 vocab lookup → id</text>
    <line x1="155" y1="168" x2="155" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="150,200 160,200 155,208" style="fill:var(--muted)"/>
    <line x1="305" y1="168" x2="305" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="300,200 310,200 305,208" style="fill:var(--muted)"/>
    <line x1="455" y1="168" x2="455" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="450,200 460,200 455,208" style="fill:var(--muted)"/>
    <line x1="605" y1="168" x2="605" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="600,200 610,200 605,208" style="fill:var(--muted)"/>
    <rect x="90"  y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="240" y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="390" y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="540" y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="155" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">12</text>
    <text x="305" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">8801</text>
    <text x="455" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">332</text>
    <text x="605" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">1567</text>
    <text x="380" y="278" text-anchor="middle" style="fill:var(--teal);font-size:11px">引擎内部只认右边这串整数，不再认识原始文本</text>
  </svg>
  <div class="figcap"><b>图 1 · 分词：文本 → token id</b> — 字符串 <span class="mono">"SGLang 很快"</span> 先被切成 <span class="mono">["SG","Lang","很","快"]</span> 四个子词，再各自查词表映射成整数 token id <span class="mono">[12, 8801, 332, 1567]</span>（id 仅为示意）。从此往里，引擎只认这串数字。</div>
</div>

<p>举个具体例子感受一下分词的粒度：英文句子 <span class="mono">"Hello, world!"</span> 大约切成 4 个 token（<span class="mono">["Hello", ",", " world", "!"]</span>），
而中文 <span class="mono">"很快"</span> 通常是 2 个 token。同一段文字到底是几个 token，取决于模型自带的词表与 BPE 合并规则——这正是 TokenizerManager 在前门替你算好的第一件事。</p>

<h2>它在哪个进程？为什么是 CPU 的活</h2>
<p>回忆第 2 课的<strong>三进程模型</strong>：分词在主进程，调度+前向在调度器子进程，反分词在反分词器子进程。
TokenizerManager 就坐在<strong>主进程</strong>这一格——和 HTTP server / Engine 同进程，<strong>不</strong>在 GPU 子进程里。
这不是随手的安排，而是刻意的<strong>进程边界</strong>设计。</p>

<p>关键在于：<strong>分词与反分词本质是 CPU 密集的 Python 字符串处理</strong>——查词表、拼 BPE、处理特殊 token，
全是 CPU 在忙，和 GPU 的矩阵乘法毫无关系。如果把这些活塞进 GPU 子进程，CPU 在分词时 GPU 只能干等，
反之亦然，两边互相拖累、串行排队。把 GPU 循环单独关进一个进程、让 TokenizerManager 在主进程做 CPU 侧的预处理，
两者就能<strong>真正并行重叠</strong>：GPU 正在为上一批做前向时，CPU 已经在给下一批分词了。
这正是 SGLang “零开销调度”（第 21 课）的根基，而 TokenizerManager ↔ 调度器之间那一跳，就是 <strong>ZMQ/IPC 的接缝</strong>。</p>

<p>这条进程边界还带来一个常被忽视的好处：<strong>容错与隔离</strong>。GPU 子进程一旦因为显存溢出（OOM）或 CUDA 错误而崩溃，
主进程里的 TokenizerManager 仍然活着，能感知到子进程异常并对外报错、清理挂起的请求状态，而不是整个服务一起静默死掉。
反过来，主进程里某条请求的分词出了问题，也不会污染正在 GPU 上跑的其他请求。<strong>把“易错的、重的 GPU 工作”和“轻的、面向人类文本的协调工作”分到两个进程</strong>，
让系统在工程上更稳健、更好观测——这是三进程模型除了性能之外的另一层深意。</p>

<div class="cols">
  <div class="col"><h4>主进程 · TokenizerManager</h4><p>CPU 侧前门：<strong>分词</strong>、建采样参数、打包 <span class="mono">TokenizedGenerateReqInput</span>、按 <span class="mono">rid</span> 登记异步状态、<span class="mono">await</span> 回包。<strong>不碰 GPU</strong>，纯 Python 字符串与协程调度。</p></div>
  <div class="col"><h4>子进程 · Scheduler + GPU</h4><p>收到已分词的请求后做<strong>连续批处理</strong>与前向（第 18 课），吃显存、跑 kernel。它<strong>只认 token id</strong>，从不直接接触原始文本，专心喂饱 GPU。</p></div>
</div>

<p>这里还藏着一个常被忽略的细节：拆进程不仅是为了重叠算力，也是为了绕开 Python 的<strong>全局解释器锁（GIL）</strong>。
如果分词、调度、反分词都挤在同一进程的不同线程里，它们会因为抢同一把 GIL 而<strong>事实上串行</strong>，多核优势荡然无存。
拆成独立进程后，每个进程有自己的解释器与 GIL，三者才能在多核 CPU 上<strong>真正并行</strong>，再用 ZMQ 把消息异步串成流水线。
所以 TokenizerManager 待在主进程、调度器待在子进程，既是<strong>职责分离</strong>，也是<strong>性能必需</strong>——这一点在第 2 课与第 21 课会反复出现。</p>

<h2>generate_request：一条请求的旅程</h2>
<p>当一条请求到来，<span class="mono">generate_request</span> 这个 <span class="mono">async</span> 协程把它一步步推下去：
先<strong>规范化</strong>参数，再<strong>分词</strong>得到 token id，<strong>构造采样参数</strong>，打包成带 <span class="mono">rid</span> 的
<span class="mono">TokenizedGenerateReqInput</span>，经 <strong>ZMQ</strong> 发往调度器，最后 <span class="mono">await</span> 回流的输出并逐段交还。
为什么要先<strong>规范化</strong>？因为外部进来的请求形态各异：可能是单条文本、可能是一批文本、可能直接给了 token id、还可能带着图片或音频。
TokenizerManager 先把这些情况统一成内部规范的形状，再分支处理“单条”还是“成批”。下面把这条主干拆成五步：</p>

<p>这五步里，<strong>第 1 步和第 5 步</strong>是理解整个设计的关键。第 1 步把“人类文本”收口成“token 序列”，从此往里一切都是数值；
第 5 步则把主进程从“同步等待”里解放出来——<span class="mono">async for</span> 加 <span class="mono">yield</span> 意味着这条协程在等回包时会<strong>主动让出</strong>，
让事件循环去推进别的请求。于是同一个主进程能用<strong>单线程的协程并发</strong>同时照看成百上千条请求，而不需要为每条请求开一个线程。
这正是 <span class="mono">generate_request</span> 写成 <span class="mono">async</span> 的根本原因：它要做的不是“算”，而是“高效地等与转”。</p>

<p>顺带厘清第 2 步里的<strong>采样参数</strong>为什么也在前门构造：温度、top_p、最大新生成 token 数这些是<strong>请求级</strong>的旋钮，
每条请求都可能不同。TokenizerManager 在打包时就把它们收成一个 <span class="mono">SamplingParams</span> 并做 <span class="mono">normalize / verify</span> 校验，
于是<strong>非法参数能在最前面被拦下</strong>（比如越界的 top_p），不必等跑到 GPU 上才报错，省掉一次昂贵的无效前向。
把“参数校验”放在前门，和“分词”放在前门是同一个工程直觉：<strong>越早把脏活、轻活做完，越能让后面的重活保持纯净高效</strong>。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>分词 tokenize</h4><p><span class="mono">_tokenize_one_request</span> 把文本喂给 tokenizer，得到 <span class="mono">input_ids</span>（纯 CPU 工作，刻意留在 GPU 进程之外）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>建采样参数</h4><p>把 <span class="mono">temperature / top_p / max_new_tokens</span> 等收成一个 <span class="mono">SamplingParams</span>，并 <span class="mono">normalize / verify</span> 校验。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>打包 + 贴 rid</h4><p>构造 <span class="mono">TokenizedGenerateReqInput</span>（第 16 课），带上唯一请求号 <span class="mono">rid</span>，作为跨进程的“身份证”。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>ZMQ 发送</h4><p><span class="mono">_send_one_request</span> 把这个对象经 <strong>ZMQ socket</strong> 投递给调度器子进程，主进程不再阻塞等待。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>登记并 await</h4><p>在 <span class="mono">rid_to_state</span> 里登记一份异步状态，<span class="mono">await</span> 反分词器流回的输出，逐段 <span class="mono">yield</span> 给调用方。</p></div></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/io_struct.py ::GenerateReqInput</span><span class="ln">入口解析出的请求结构：文本或 token、采样参数、是否流式</span></div>
  <pre>@dataclasses.dataclass
<span class="kw">class</span> GenerateReqInput:
    <span class="cm"># the parsed request that flows from the entry into the engine</span>
    text = None            <span class="cm"># the prompt (or a batch of prompts)</span>
    input_ids = None       <span class="cm"># OR pre-tokenized token ids</span>
    sampling_params = None <span class="cm"># temperature / top_p / max_new_tokens ...</span>
    stream = False         <span class="cm"># stream tokens back as they are produced</span>
    rid = None             <span class="cm"># request id (assigned per request)</span></pre>
</div>

<p>注意 <span class="mono">text</span> 与 <span class="mono">input_ids</span> 是<strong>二选一</strong>：要么给原始文本、让前门替你分词，要么直接给好 token id 跳过分词这一步。
无论走哪条路，TokenizerManager 都会贴上一个唯一的 <span class="mono">rid</span>、收好 <span class="mono">sampling_params</span>，再决定是否按 <span class="mono">stream</span> 流式回传。</p>

<h2>rid 的往返：异步枢纽如何对号入座</h2>
<p>TokenizerManager 最精妙的地方是它的<strong>异步路由</strong>。几百条请求并发涌入，每条都被分配一个唯一的 <span class="mono">rid</span>；
请求发出后，主进程<strong>不阻塞</strong>，而是把 <span class="mono">rid → 协程状态</span> 记进一张表（<span class="mono">rid_to_state</span>）。
后台的输出<strong>乱序交错</strong>地流回来——某一刻可能先到第 7 号的一个 token，下一刻又到第 3 号的——
TokenizerManager 凭 <span class="mono">rid</span> 把每段输出<strong>准确投递</strong>回正在 <span class="mono">await</span> 的那个协程。
没有 <span class="mono">rid</span>，几百条并发请求的回包就会张冠李戴。它还负责 <strong>abort</strong>（中止某条请求）和多模态的<strong>图像/音频预处理</strong>。</p>

<div class="flow">
  <div class="node"><div class="nt">TokenizerManager</div><div class="nd">分词 + 贴 rid，主进程</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Scheduler</div><div class="nd">组批 + 前向（子进程）</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Detokenizer</div><div class="nd">反分词 token→文本</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">回到 TokenizerManager</div><div class="nd">按 rid 投回原协程</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="TokenizerManager 与 Scheduler 之间通过 ZMQ 通信：主进程把已分词请求经 PUSH socket 发往子进程，子进程把结果经 PULL socket 回传，中间是一条进程边界">
    <text x="380" y="26" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:13px">TokenizerManager ↔ Scheduler（ZMQ / IPC）</text>
    <line x1="380" y1="48" x2="380" y2="270" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="380" y="288" text-anchor="middle" style="fill:var(--muted);font-size:10px">进程边界 process boundary</text>
    <rect x="40" y="70" width="270" height="130" rx="12" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="175" y="104" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:14px">TokenizerManager</text>
    <text x="175" y="126" text-anchor="middle" style="fill:var(--muted);font-size:12px">主进程 · CPU · 分词</text>
    <text x="175" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">await 回包，按 rid 投回协程</text>
    <rect x="450" y="70" width="270" height="130" rx="12" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="585" y="104" text-anchor="middle" style="fill:var(--blue);font-weight:700;font-size:14px">Scheduler</text>
    <text x="585" y="126" text-anchor="middle" style="fill:var(--muted);font-size:12px">子进程 · GPU · 组批前向</text>
    <text x="585" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">只认 token id</text>
    <text x="380" y="96" text-anchor="middle" style="fill:var(--muted);font-size:10px">ZMQ PUSH：已分词请求 (rid)</text>
    <circle cx="310" cy="110" r="6" style="fill:var(--accent);stroke:var(--accent-ink);stroke-width:1"/>
    <circle cx="450" cy="110" r="6" style="fill:var(--blue);stroke:var(--blue);stroke-width:1"/>
    <line x1="316" y1="110" x2="444" y2="110" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="444,105 444,115 450,110" style="fill:var(--muted)"/>
    <text x="380" y="180" text-anchor="middle" style="fill:var(--teal);font-size:10px">ZMQ PULL：输出（经 Detok）</text>
    <circle cx="310" cy="160" r="6" style="fill:var(--teal);stroke:var(--teal);stroke-width:1"/>
    <circle cx="450" cy="160" r="6" style="fill:var(--teal);stroke:var(--teal);stroke-width:1"/>
    <line x1="444" y1="160" x2="316" y2="160" style="stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="316,155 316,165 310,160" style="fill:var(--teal)"/>
    <text x="175" y="190" text-anchor="middle" style="fill:var(--faint);font-size:10px">PUSH / PULL socket</text>
    <text x="585" y="190" text-anchor="middle" style="fill:var(--faint);font-size:10px">PULL / PUSH socket</text>
  </svg>
  <div class="figcap"><b>图 2 · TokenizerManager ↔ Scheduler（IPC）</b> — 主进程里的 <span class="mono">TokenizerManager</span> 把已分词请求经 <strong>ZMQ PUSH</strong> 投过进程边界，交给子进程的 <span class="mono">Scheduler</span>（独占 GPU），结果再经 <strong>PULL</strong> 回传；两端只靠 <span class="mono">rid</span> 对号入座。</div>
</div>

<p>注意这条回路的<strong>闭环</strong>形状：请求从 TokenizerManager 出发，绕过调度器与反分词器，最后<strong>又回到</strong> TokenizerManager。
也就是说，TokenizerManager 既是<strong>入口</strong>也是<strong>出口</strong>——它把请求送出去，也负责把结果收回来交还给调用方。
反分词器（第 17 课）之所以把结果发回 TokenizerManager 而不是直接回给客户端，正是因为只有 TokenizerManager 那张 <span class="mono">rid_to_state</span>
表知道“这个 rid 对应的是哪个正在 <span class="mono">await</span> 的协程、该怎么 <span class="mono">yield</span> 给上层的 HTTP SSE 或 Engine 调用者”。
这把“谁在等结果”的状态<strong>集中在前门一处</strong>，是这套异步架构能简洁运转的关键。</p>

<table class="t">
  <tr><th>TokenizerManager 负责什么</th><th>具体含义</th></tr>
  <tr><td class="mono">tokenize</td><td>把文本分词成 token id（CPU 工作，挡在 GPU 进程之外）</td></tr>
  <tr><td class="mono">sampling params</td><td>构造并校验 <span class="mono">SamplingParams</span>（温度、top_p、max_new_tokens 等）</td></tr>
  <tr><td class="mono">rid</td><td>给每条请求贴唯一请求号，作为跨进程身份与回包路由依据</td></tr>
  <tr><td class="mono">async routing</td><td>维护 <span class="mono">rid_to_state</span>，把乱序回包投回正确的等待协程</td></tr>
  <tr><td class="mono">abort</td><td>处理请求中止，清理对应的挂起状态</td></tr>
  <tr><td class="mono">mm preprocess</td><td>多模态请求的图像/音频预处理</td></tr>
</table>

<p>把这张表连起来看：TokenizerManager 是一个<strong>只做 CPU 侧协调、绝不碰 GPU</strong> 的异步枢纽。它把“文本→token、
贴号、转单、等回、交付”这一整套前门工作全包了，从而让 GPU 子进程能心无旁骛地<strong>只认 token、只管算</strong>。
顺带一提，表里的<strong>多模态预处理</strong>也安排在这里很自然：把一张图片解码、缩放、切成图像 patch，同样是<strong>CPU 密集</strong>的预处理，
和文本分词是同一类活，理应在主进程一并做完，再把处理好的多模态张量随 token 一起打包送往 GPU 子进程。
所以无论输入是纯文本还是图文混合，TokenizerManager 都是那道<strong>统一的、CPU 侧的预处理关口</strong>。
下面这段就是 <span class="mono">generate_request</span> 的真实骨架：分词、登记、ZMQ 发送、再 <span class="mono">await</span> 回包。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/tokenizer_manager.py ::TokenizerManager.generate_request</span><span class="ln">前门：分词 → ZMQ 发送 → await 回包</span></div>
  <pre><span class="kw">async def</span> generate_request(self, obj, request=None):
    self.auto_create_handle_loop()
    obj.normalize_batch_and_arguments()
    self._init_req_state(obj, request)        <span class="cm"># 按 rid 登记异步状态</span>

    <span class="kw">if</span> obj.is_single:
        <span class="cm"># 分词（CPU 工作，留在 GPU 进程之外）</span>
        tokenized_obj = <span class="kw">await</span> self._tokenize_one_request(obj)
        state = self.rid_to_state[obj.rid]    <span class="cm"># 用 rid 找到这条请求的状态</span>
        self._send_one_request(tokenized_obj) <span class="cm"># 经 ZMQ 发给调度器子进程</span>
        <span class="kw">async for</span> response <span class="kw">in</span> self._wait_one_response(obj, request):
            <span class="kw">yield</span> response                    <span class="cm"># await 回流输出，逐段交还调用方</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>前门 / 主进程</strong>：TokenizerManager 是运行时的前门，跑在<strong>主进程</strong>（与 HTTP server / Engine 同进程），<strong>不</strong>在 GPU 子进程里。</li>
    <li><strong>五步主干</strong>：<span class="mono">generate_request</span> 协程——分词 → 建采样参数 → 打包带 <span class="mono">rid</span> 的 <span class="mono">TokenizedGenerateReqInput</span>（第 16 课）→ 经 <strong>ZMQ</strong> 发给调度器 → <span class="mono">await</span> 回包并逐段 <span class="mono">yield</span>。</li>
    <li><strong>rid 的作用</strong>：唯一请求号是跨进程身份证；几百条并发请求乱序回包，靠 <span class="mono">rid_to_state</span> 投回正确的等待协程（第 17 课流式回传）。</li>
    <li><strong>为什么 CPU 上分词</strong>：分词/反分词是 CPU 密集的字符串工作，隔在 GPU 进程之外才能让 CPU 与 GPU <strong>重叠</strong>（零开销，第 21 课）。</li>
    <li><strong>还负责</strong>：请求 <span class="mono">abort</span> 与多模态图像/音频预处理；它是异步并发的总枢纽，自身从不接触 GPU。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson said: whether you enter through the offline <span class="inline">Engine</span> or the online server, requests land on
<strong>one shared runtime</strong>. So once a request is inside, which component catches it <strong>first</strong>? That is today's star —
the <strong>TokenizerManager</strong>. It is the runtime's <strong>front door</strong>: it runs in the <strong>main process</strong>
(alongside the HTTP server / Engine), <strong>not</strong> in the GPU-hungry subprocess. Every request is first tokenized, packed and
registered here before being sent on to the scheduler.
</p>

<p>Why give it its own lesson? Because the TokenizerManager is the translator between the outside <strong>text world</strong> and the inner
<strong>token world</strong>, and the <strong>async concurrency hub</strong>. You send a string; the engine speaks token ids. Hundreds of
requests pour in at once — whose reply goes back to whom is its job to coordinate. Understanding this front door is what lets Lesson 16's
io_struct messages, Lesson 17's detokenize/streaming, and Lesson 18's scheduler chain into one coherent flow.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of the TokenizerManager as a hotel <strong>front-desk receptionist</strong>. A guest (request) walks up and states their need in
  <strong>spoken words</strong> (raw text). The receptionist <strong>translates it into a standardized ticket</strong> — fills in the room
  type and requests (sampling params), staples on a <strong>booking number</strong> (rid), and <strong>sends the ticket to the back office</strong>
  (the scheduler subprocess) to actually arrange things. The guest never enters the kitchen; they just <strong>wait to be called</strong>.
  Meanwhile the receptionist <strong>juggles dozens of guests at once</strong>, handing each reply back to exactly the right person. The desk
  never cooks (never touches the GPU): it only <strong>translates, registers, forwards, awaits, and delivers</strong> — exactly the TokenizerManager's job.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The TokenizerManager lives in the <strong>main process</strong> as the CPU-side async coordinator. Its core method
  <span class="mono">generate_request</span> is an <span class="mono">async</span> coroutine: it <strong>tokenizes</strong> the text into token ids
  (pure CPU string work), builds the <strong>sampling params</strong>, packs everything into a <span class="mono">TokenizedGenerateReqInput</span>
  (Lesson 16) carrying a unique <span class="mono">rid</span>, and sends it over <strong>ZMQ</strong> to the scheduler subprocess. It then
  <strong>registers a per-request async state keyed by rid</strong> and <span class="mono">await</span>s the outputs streaming back from the
  detokenizer (Lesson 17), <span class="mono">yield</span>ing them to the caller (HTTP SSE or the Engine). Isolating tokenize/detokenize from the
  GPU loop in separate processes is what lets CPU and GPU <strong>overlap</strong> (the zero-overhead idea, Lesson 21).
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="Tokenize flow: the string SGLang plus a Chinese phrase is first split into subwords SG Lang and two characters, then each maps to an integer token id 12 8801 332 1567">
    <text x="380" y="26" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:13px">text → subwords → token ids</text>
    <rect x="265" y="42" width="230" height="44" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="380" y="70" text-anchor="middle" class="mono" style="fill:var(--ink);font-size:15px">SGLang 很快</text>
    <text x="432" y="108" text-anchor="middle" style="fill:var(--muted);font-size:10px">tokenize</text>
    <line x1="380" y1="86" x2="380" y2="122" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="375,116 385,116 380,124" style="fill:var(--muted)"/>
    <rect x="90"  y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="240" y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="390" y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="540" y="124" width="130" height="44" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="155" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"SG"</text>
    <text x="305" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"Lang"</text>
    <text x="455" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"很"</text>
    <text x="605" y="152" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">"快"</text>
    <text x="432" y="196" text-anchor="middle" style="fill:var(--muted);font-size:10px">vocab lookup → id</text>
    <line x1="155" y1="168" x2="155" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="150,200 160,200 155,208" style="fill:var(--muted)"/>
    <line x1="305" y1="168" x2="305" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="300,200 310,200 305,208" style="fill:var(--muted)"/>
    <line x1="455" y1="168" x2="455" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="450,200 460,200 455,208" style="fill:var(--muted)"/>
    <line x1="605" y1="168" x2="605" y2="206" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="600,200 610,200 605,208" style="fill:var(--muted)"/>
    <rect x="90"  y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="240" y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="390" y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="540" y="208" width="130" height="44" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="155" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">12</text>
    <text x="305" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">8801</text>
    <text x="455" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">332</text>
    <text x="605" y="236" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:14px">1567</text>
    <text x="380" y="278" text-anchor="middle" style="fill:var(--teal);font-size:11px">inside the engine only these integers exist, not the raw text</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Tokenize: text → token ids</b> — the string <span class="mono">"SGLang 很快"</span> is first split into the four subwords <span class="mono">["SG","Lang","很","快"]</span>, then each is looked up in the vocab and mapped to an integer token id <span class="mono">[12, 8801, 332, 1567]</span> (ids illustrative). From here inward, the engine only speaks these numbers.</div>
</div>

<p>A concrete example for the granularity of tokenization: the English sentence <span class="mono">"Hello, world!"</span> tokenizes into about 4 tokens
(<span class="mono">["Hello", ",", " world", "!"]</span>), while the Chinese <span class="mono">"很快"</span> is usually 2 tokens. How many tokens a piece of text
becomes depends on the model's own vocab and BPE merge rules — exactly the first thing the TokenizerManager computes for you at the front door.</p>

<h2>Which process? Why is it CPU work</h2>
<p>Recall Lesson 2's <strong>three-process model</strong>: tokenize in the main process, schedule+forward in the scheduler subprocess,
detokenize in the detokenizer subprocess. The TokenizerManager sits in the <strong>main process</strong> slot — same process as the HTTP
server / Engine, <strong>not</strong> in the GPU subprocess. This is a deliberate <strong>process-boundary</strong> design, not an accident.</p>

<p>The key: <strong>tokenize and detokenize are CPU-bound Python string work</strong> — table lookups, BPE merges, special-token handling,
all CPU, nothing to do with GPU matmuls. If you crammed them into the GPU subprocess, the GPU would idle while the CPU tokenizes and vice
versa — the two would serialize and drag each other down. By locking the GPU loop in its own process and letting the TokenizerManager do the
CPU-side preprocessing in the main process, the two can <strong>truly overlap</strong>: while the GPU forwards the previous batch, the CPU is
already tokenizing the next. This is the foundation of SGLang's zero-overhead scheduling (Lesson 21), and the TokenizerManager ↔ scheduler
hop is exactly the <strong>ZMQ/IPC seam</strong>.</p>

<div class="cols">
  <div class="col"><h4>Main process · TokenizerManager</h4><p>CPU-side front door: <strong>tokenize</strong>, build sampling params, pack a <span class="mono">TokenizedGenerateReqInput</span>, register async state by <span class="mono">rid</span>, <span class="mono">await</span> replies. <strong>Never touches the GPU</strong> — pure Python strings and coroutine scheduling.</p></div>
  <div class="col"><h4>Subprocess · Scheduler + GPU</h4><p>Given already-tokenized requests, it does <strong>continuous batching</strong> and forward passes (Lesson 18), burning VRAM and kernels. It <strong>only speaks token ids</strong>, never sees raw text, focused on keeping the GPU fed.</p></div>
</div>

<h2>generate_request: a request's journey</h2>
<p>When a request arrives, the <span class="mono">async</span> coroutine <span class="mono">generate_request</span> pushes it down step by step:
<strong>normalize</strong> the args, <strong>tokenize</strong> into token ids, <strong>build sampling params</strong>, pack a
<span class="mono">TokenizedGenerateReqInput</span> with a <span class="mono">rid</span>, send over <strong>ZMQ</strong> to the scheduler, then
<span class="mono">await</span> the streamed outputs and hand them back piece by piece. Here is the trunk in five steps:</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>tokenize</h4><p><span class="mono">_tokenize_one_request</span> feeds text to the tokenizer, yielding <span class="mono">input_ids</span> (pure CPU work, deliberately kept off the GPU process).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>build sampling params</h4><p>Collect <span class="mono">temperature / top_p / max_new_tokens</span> into a <span class="mono">SamplingParams</span>, then <span class="mono">normalize / verify</span> it.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>pack + stamp rid</h4><p>Build a <span class="mono">TokenizedGenerateReqInput</span> (Lesson 16) carrying a unique <span class="mono">rid</span> — the cross-process "ID card".</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>ZMQ send</h4><p><span class="mono">_send_one_request</span> dispatches the object over a <strong>ZMQ socket</strong> to the scheduler subprocess; the main process does not block.</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>register and await</h4><p>Register an async state in <span class="mono">rid_to_state</span>, <span class="mono">await</span> the outputs streaming back from the detokenizer, and <span class="mono">yield</span> them to the caller.</p></div></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/io_struct.py ::GenerateReqInput</span><span class="ln">the parsed request struct: text or tokens, sampling params, stream flag</span></div>
  <pre>@dataclasses.dataclass
<span class="kw">class</span> GenerateReqInput:
    <span class="cm"># the parsed request that flows from the entry into the engine</span>
    text = None            <span class="cm"># the prompt (or a batch of prompts)</span>
    input_ids = None       <span class="cm"># OR pre-tokenized token ids</span>
    sampling_params = None <span class="cm"># temperature / top_p / max_new_tokens ...</span>
    stream = False         <span class="cm"># stream tokens back as they are produced</span>
    rid = None             <span class="cm"># request id (assigned per request)</span></pre>
</div>

<p>Note that <span class="mono">text</span> and <span class="mono">input_ids</span> are <strong>mutually exclusive</strong>: either pass raw text and let the front
door tokenize it, or pass token ids directly to skip tokenization. Either way the TokenizerManager stamps a unique <span class="mono">rid</span>, collects the
<span class="mono">sampling_params</span>, and decides whether to return the output as a <span class="mono">stream</span>.</p>

<h2>The rid round-trip: how the hub routes replies</h2>
<p>The TokenizerManager's cleverest part is its <strong>async routing</strong>. Hundreds of requests arrive concurrently, each assigned a unique
<span class="mono">rid</span>; after sending, the main process <strong>does not block</strong> — it records <span class="mono">rid → coroutine state</span>
in a map (<span class="mono">rid_to_state</span>). Outputs flow back <strong>interleaved and out of order</strong> — a token for #7 now, one for #3
next — and the TokenizerManager uses the <span class="mono">rid</span> to <strong>route each chunk back</strong> to the exact coroutine that is
<span class="mono">await</span>ing it. Without the <span class="mono">rid</span>, hundreds of concurrent replies would go to the wrong callers. It also
handles <strong>abort</strong> and, for multimodal, <strong>image/audio preprocessing</strong>.</p>

<div class="flow">
  <div class="node"><div class="nt">TokenizerManager</div><div class="nd">tokenize + stamp rid, main proc</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Scheduler</div><div class="nd">batch + forward (subproc)</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Detokenizer</div><div class="nd">tokens → text</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">back to TokenizerManager</div><div class="nd">route by rid to the coroutine</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="TokenizerManager talks to the Scheduler over ZMQ: the main process sends the tokenized request through a PUSH socket to the subprocess, which sends results back through a PULL socket, with a process boundary in the middle">
    <text x="380" y="26" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:13px">TokenizerManager ↔ Scheduler (ZMQ / IPC)</text>
    <line x1="380" y1="48" x2="380" y2="270" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="380" y="288" text-anchor="middle" style="fill:var(--muted);font-size:10px">process boundary</text>
    <rect x="40" y="70" width="270" height="130" rx="12" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="175" y="104" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:14px">TokenizerManager</text>
    <text x="175" y="126" text-anchor="middle" style="fill:var(--muted);font-size:12px">main process · CPU · tokenize</text>
    <text x="175" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">await replies, route by rid</text>
    <rect x="450" y="70" width="270" height="130" rx="12" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="585" y="104" text-anchor="middle" style="fill:var(--blue);font-weight:700;font-size:14px">Scheduler</text>
    <text x="585" y="126" text-anchor="middle" style="fill:var(--muted);font-size:12px">subprocess · GPU · batch+forward</text>
    <text x="585" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">speaks only token ids</text>
    <text x="380" y="96" text-anchor="middle" style="fill:var(--muted);font-size:10px">ZMQ PUSH: tokenized request (rid)</text>
    <circle cx="310" cy="110" r="6" style="fill:var(--accent);stroke:var(--accent-ink);stroke-width:1"/>
    <circle cx="450" cy="110" r="6" style="fill:var(--blue);stroke:var(--blue);stroke-width:1"/>
    <line x1="316" y1="110" x2="444" y2="110" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="444,105 444,115 450,110" style="fill:var(--muted)"/>
    <text x="380" y="180" text-anchor="middle" style="fill:var(--teal);font-size:10px">ZMQ PULL: output (via Detok)</text>
    <circle cx="310" cy="160" r="6" style="fill:var(--teal);stroke:var(--teal);stroke-width:1"/>
    <circle cx="450" cy="160" r="6" style="fill:var(--teal);stroke:var(--teal);stroke-width:1"/>
    <line x1="444" y1="160" x2="316" y2="160" style="stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="316,155 316,165 310,160" style="fill:var(--teal)"/>
    <text x="175" y="190" text-anchor="middle" style="fill:var(--faint);font-size:10px">PUSH / PULL socket</text>
    <text x="585" y="190" text-anchor="middle" style="fill:var(--faint);font-size:10px">PULL / PUSH socket</text>
  </svg>
  <div class="figcap"><b>Fig 2 · TokenizerManager ↔ Scheduler (IPC)</b> — the <span class="mono">TokenizerManager</span> in the main process pushes the tokenized request over <strong>ZMQ PUSH</strong> across the process boundary to the <span class="mono">Scheduler</span> subprocess (which alone holds the GPU); results return over <strong>PULL</strong>. Both ends match replies up purely by <span class="mono">rid</span>.</div>
</div>

<table class="t">
  <tr><th>What the TokenizerManager owns</th><th>Meaning</th></tr>
  <tr><td class="mono">tokenize</td><td>turn text into token ids (CPU work, kept off the GPU process)</td></tr>
  <tr><td class="mono">sampling params</td><td>build and verify <span class="mono">SamplingParams</span> (temperature, top_p, max_new_tokens, ...)</td></tr>
  <tr><td class="mono">rid</td><td>stamp each request with a unique id — cross-process identity and reply routing</td></tr>
  <tr><td class="mono">async routing</td><td>maintain <span class="mono">rid_to_state</span>, route out-of-order replies to the right awaiting coroutine</td></tr>
  <tr><td class="mono">abort</td><td>handle request cancellation, clean up the pending state</td></tr>
  <tr><td class="mono">mm preprocess</td><td>image/audio preprocessing for multimodal requests</td></tr>
</table>

<p>Read the table as a whole: the TokenizerManager is an async hub that does <strong>only CPU-side coordination and never touches the GPU</strong>.
It owns the whole front-door job — text→tokens, stamp, forward, await, deliver — so the GPU subprocess can <strong>speak only tokens and just
compute</strong>. Below is the real skeleton of <span class="mono">generate_request</span>: tokenize, register, ZMQ send, then
<span class="mono">await</span> the replies.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/tokenizer_manager.py ::TokenizerManager.generate_request</span><span class="ln">front door: tokenize → ZMQ send → await replies</span></div>
  <pre><span class="kw">async def</span> generate_request(self, obj, request=None):
    self.auto_create_handle_loop()
    obj.normalize_batch_and_arguments()
    self._init_req_state(obj, request)        <span class="cm"># register async state by rid</span>

    <span class="kw">if</span> obj.is_single:
        <span class="cm"># tokenize (CPU work, kept off the GPU process)</span>
        tokenized_obj = <span class="kw">await</span> self._tokenize_one_request(obj)
        state = self.rid_to_state[obj.rid]    <span class="cm"># find this request's state by rid</span>
        self._send_one_request(tokenized_obj) <span class="cm"># send over ZMQ to the scheduler subproc</span>
        <span class="kw">async for</span> response <span class="kw">in</span> self._wait_one_response(obj, request):
            <span class="kw">yield</span> response                    <span class="cm"># await streamed outputs, hand back to caller</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Front door / main process</strong>: the TokenizerManager is the runtime's front door, running in the <strong>main process</strong> (with the HTTP server / Engine), <strong>not</strong> in the GPU subprocess.</li>
    <li><strong>Five-step trunk</strong>: the <span class="mono">generate_request</span> coroutine — tokenize → build sampling params → pack a <span class="mono">TokenizedGenerateReqInput</span> with a <span class="mono">rid</span> (Lesson 16) → send over <strong>ZMQ</strong> to the scheduler → <span class="mono">await</span> replies and <span class="mono">yield</span> them.</li>
    <li><strong>What the rid is for</strong>: the unique request id is the cross-process ID card; with hundreds of concurrent, out-of-order replies, <span class="mono">rid_to_state</span> routes each back to the right awaiting coroutine (streaming, Lesson 17).</li>
    <li><strong>Why tokenize on CPU</strong>: tokenize/detokenize are CPU-bound string work; keeping them off the GPU process lets CPU and GPU <strong>overlap</strong> (zero-overhead, Lesson 21).</li>
    <li><strong>Also owns</strong>: request <span class="mono">abort</span> and multimodal image/audio preprocessing; it is the async concurrency hub and itself never touches the GPU.</li>
  </ul>
</div>
""",
}

LESSON_15 = {
    "zh": r"""
<p class="lead">
上一课我们看清了运行时的<strong>前门</strong>——TokenizerManager。但有个问题它回答不了：外面那么多客户端，
有的说 <strong>OpenAI 方言</strong>、有的说 <strong>Anthropic 方言</strong>、有的说 <strong>Ollama 方言</strong>，
它们怎么可能一字不改就接进同一套运行时？答案是本课的主角——<strong>兼容层（compat layer）</strong>。
它让 SGLang 服务器<strong>同时会说好几种协议</strong>，于是<strong>现成的客户端不用改代码</strong>就能直接用。
</p>

<p>为什么要专门讲兼容层？因为它是 SGLang “即插即用”到现有生态的<strong>关键缝隙</strong>。你不必为 SGLang 重写一个 SDK，
也不必让团队学一套新 API——把 OpenAI 客户端的 <span class="mono">base_url</span> 一改，指向你的 SGLang 部署，<strong>立刻就能跑</strong>。
LangChain、<span class="mono">openai</span> Python 包、各种围绕 OpenAI 接口建起来的工具，统统能无缝接入。这一层薄薄的适配器，
就是私有部署接管整个生态的<strong>那把万能钥匙</strong>。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把兼容层想成<strong>站在门口的一排多语翻译官</strong>。客人各说各的方言：有人讲 OpenAI 话，有人讲 Anthropic 话，
  有人讲 Ollama 话。翻译官把每一种方言<strong>都翻成厨房唯一听得懂的“本店语言”</strong>——也就是原生的
  <span class="mono">GenerateReqInput</span>（第 16 课）。厨房（运行时）压根不知道客人本来说的是哪种话，它只照着这份标准工单做菜；
  菜做好了，翻译官再<strong>把同一盘菜翻译回各位客人的方言</strong>端上去。客人觉得“这家店居然会说我的话”，
  其实厨房只有一个、本店语言只有一种——会的只是<strong>门口那排翻译官</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  兼容层住在 <span class="mono">entrypoints/openai/</span> 等目录里，由一组<strong>serving 类</strong>构成。以 OpenAI 为例：
  <span class="mono">/v1/chat/completions</span> 由 <span class="mono">OpenAIServingChat</span> 接管，
  <span class="mono">/v1/completions</span> 由 <span class="mono">OpenAIServingCompletion</span>，还有 <span class="mono">/v1/embeddings</span> 等。
  每个 serving 类干的事都一样：把一个 <strong>OpenAI 形状</strong>的请求<strong>翻译</strong>过来——套用聊天模板、构造采样参数——
  拼成<strong>原生的 <span class="mono">GenerateReqInput</span></strong>（第 16 课），交给 TokenizerManager（第 14 课）；
  等运行时产出结果，再把它<strong>映射回</strong> OpenAI 的响应对象或 SSE 流式块（<span class="mono">chat.completion.chunk</span>）。
  Anthropic（<span class="mono">/v1/messages</span>）和 Ollama 的适配器对它们各自的 schema 做同一件事。
</div>

<h2>一条请求的协议之旅</h2>
<p>先把整条路径在脑子里走一遍。一个请求的旅程是：<strong>客户端 SDK → 协议适配器 → 原生请求 → 同一套运行时 → 适配器格式化响应</strong>。
注意中间那一段——<strong>“同一套运行时”</strong>——无论入口说的是哪种方言，到了这里都收敛成一条共享的快路径。
适配器只在<strong>最外层</strong>做翻译，绝不重新实现引擎。</p>

<p>这正是兼容层设计的精髓：运行时本身是<strong>协议无关（protocol-agnostic）</strong>的，它只认 <span class="mono">GenerateReqInput</span>。
适配器要做的仅有两件事——<strong>schema 翻译</strong>（把 OpenAI/Anthropic/Ollama 的字段映射成原生字段）和<strong>聊天模板套用</strong>
（把 messages 列表按模型的模板拼成一段 prompt）。因为职责这么窄，<strong>新增一种协议非常便宜</strong>：写一个新的适配器即可，
而最热的那条推理快路径<strong>始终是共享的、只优化一遍</strong>。如果每加一种协议都要重写一遍引擎，那才是工程灾难——
兼容层用“薄适配器 + 共享内核”优雅地避开了它。</p>

<p>再把这件事说透一层：所谓“翻译”其实是<strong>双向</strong>的。进来的方向，适配器把方言请求拍平成原生请求；出去的方向，
它还要把运行时吐出的<strong>原始 token / 文本</strong>重新包装成该协议期望的形状。对 OpenAI 来说，一次性返回的结果要裹成一个
<span class="mono">chat.completion</span> 对象，带上 <span class="mono">choices</span>、<span class="mono">usage</span> 等字段；
流式返回则要切成一连串 <span class="mono">chat.completion.chunk</span> 的 SSE 事件，逐块吐给客户端（第 17 课）。
Anthropic 与 Ollama 的响应结构又各不相同，于是“出口翻译”同样由各自的适配器负责。<strong>进出两个方向都只是格式转换</strong>，
中间那段真正“做菜”的运行时完全没变——这就是为什么我们反复强调：兼容层只是门口的翻译官，不是另一个厨房。</p>

<p>还要澄清一个常见误解：兼容层<strong>不是</strong>把请求“转发”给真正的 OpenAI 去算。很多人第一次听“OpenAI 兼容”会以为
SGLang 在背后偷偷调用了 openai.com——<strong>恰恰相反</strong>。token 是<strong>你自己的 SGLang 部署、用你自己的 GPU、跑你自己的模型</strong>算出来的，
兼容层只是让请求/响应的<strong>外壳</strong>长得像 OpenAI，好让现成工具认得。换句话说，你得到的是“OpenAI 的接口形状”加“你自己的模型与算力”，
既享受了生态的便利，又完全掌控了模型与数据。这一点对私有化部署尤其关键。</p>

<p>这正是“多方言、单内核”在代码层面的样子。换个角度看，这套设计也极大降低了<strong>维护成本</strong>：性能优化、bug 修复、新特性
（比如更快的批处理、新的注意力后端）只要落在共享的运行时上，<strong>所有协议同时受益</strong>，无需逐个适配器重做一遍。
适配器很薄、很稳定，真正在快速演进的内核只有一份。这种“把易变的协议表层和稳定的高性能内核分开”的分层，
是大型系统应对“既要兼容多方、又要持续优化”这对矛盾的经典解法。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">客户端</span><span class="name">OpenAI SDK / LangChain / curl</span></div><div class="ld">说 OpenAI 方言：发 <span class="mono">/v1/chat/completions</span>，body 是 <span class="mono">messages</span> 列表。只把 <span class="mono">base_url</span> 指向 SGLang 即可，<strong>代码不改</strong>。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">适配器</span><span class="name">协议层 OpenAI / Anthropic / Ollama</span></div><div class="ld">对应的 serving 类做<strong>schema 翻译 + 聊天模板</strong>，把方言请求拼成原生形状。新增协议=加一个适配器。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">本店语言</span><span class="name">GenerateReqInput（第 16 课）</span></div><div class="ld">运行时唯一听得懂的原生请求对象：token/文本 + 采样参数 + rid。<strong>协议无关</strong>，三种方言到此收敛成一种。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">前门</span><span class="name">TokenizerManager（第 14 课）</span></div><div class="ld">接过原生请求，分词、贴 rid、经 ZMQ 送往调度器。<strong>共享的快路径</strong>从这里开始，对所有协议一视同仁。</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 770 270" role="img" aria-label="三种协议各走自己的适配器，做完 schema 翻译与聊天模板后收敛成同一个 GenerateReqInput，引擎内核只看到统一形态">
    <text x="24" y="24" style="font-weight:700;fill:var(--muted)">三种协议入口</text>
    <rect x="24" y="36" width="196" height="48" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="122" y="58" text-anchor="middle" style="font-weight:700;fill:var(--blue)">OpenAI</text>
    <text x="122" y="75" text-anchor="middle" class="mono" style="font-size:11px">/v1/chat/completions</text>
    <rect x="24" y="98" width="196" height="48" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="122" y="120" text-anchor="middle" style="font-weight:700;fill:var(--teal)">Anthropic</text>
    <text x="122" y="137" text-anchor="middle" class="mono" style="font-size:11px">/v1/messages</text>
    <rect x="24" y="160" width="196" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="122" y="182" text-anchor="middle" style="font-weight:700;fill:var(--amber)">Ollama</text>
    <text x="122" y="199" text-anchor="middle" class="mono" style="font-size:11px">/api/generate</text>
    <rect x="288" y="60" width="150" height="124" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="363" y="106" text-anchor="middle" style="font-weight:700;fill:var(--muted)">适配器层</text>
    <text x="363" y="128" text-anchor="middle" style="fill:var(--faint);font-size:12px">schema 翻译</text>
    <text x="363" y="146" text-anchor="middle" style="fill:var(--faint);font-size:12px">+ 聊天模板</text>
    <line x1="220" y1="60" x2="288" y2="100" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="220" y1="122" x2="288" y2="122" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="220" y1="184" x2="288" y2="144" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="506" y="84" width="208" height="76" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="610" y="114" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">一个内部请求</text>
    <text x="610" y="134" text-anchor="middle" class="mono" style="font-size:12px">GenerateReqInput</text>
    <line x1="438" y1="122" x2="506" y2="122" style="stroke:var(--accent);stroke-width:1.5"/>
    <text x="610" y="186" text-anchor="middle" style="fill:var(--muted);font-size:12px">引擎内核只看到这份统一形态</text>
  </svg>
  <div class="figcap"><b>图 1 · 三种协议 → 一个内部请求</b> — OpenAI、Anthropic、Ollama 三种入口各走自己的适配器，做完 <span class="mono">schema</span> 翻译与聊天模板后，统统收敛成同一个 <span class="mono">GenerateReqInput</span>；引擎内核只看到这份统一形态，再也分不清请求原本说的是哪种方言。</div>
</div>

<h2>路由 → serving 类 → 原生映射</h2>
<p>具体到代码，每条 HTTP 路由都绑着一个负责翻译的 serving 类。下面这张表把“外界看到的路由”和“内部接手的类”对齐，
你能一眼看出兼容层的<strong>分工地图</strong>：不同方言走不同适配器，但箭头最终都指向同一个 <span class="mono">GenerateReqInput</span>。</p>

<table class="t">
  <tr><th>HTTP 路由</th><th>接手的 serving 类</th><th>翻译成的原生形态</th></tr>
  <tr><td class="mono">/v1/chat/completions</td><td class="mono">OpenAIServingChat</td><td>套聊天模板 → <span class="mono">GenerateReqInput</span></td></tr>
  <tr><td class="mono">/v1/completions</td><td class="mono">OpenAIServingCompletion</td><td>原始 prompt → <span class="mono">GenerateReqInput</span></td></tr>
  <tr><td class="mono">/v1/messages</td><td>Anthropic 适配器</td><td>messages → <span class="mono">GenerateReqInput</span></td></tr>
  <tr><td class="mono">/api/generate</td><td>Ollama 适配器</td><td>Ollama 字段 → <span class="mono">GenerateReqInput</span></td></tr>
</table>

<p>表里有个一致的模式值得点明：<strong>路由形态各异，终点完全相同</strong>。<span class="mono">OpenAIServingChat</span> 要先“套聊天模板”，
因为 chat 接口给的是一串 role 消息，必须按模型自己的模板拼成单段 prompt；而 <span class="mono">/v1/completions</span> 直接给 prompt 文本，
适配器省掉模板这一步。Anthropic 的 <span class="mono">/v1/messages</span> 字段名和 OpenAI 不同（比如 system 单独放、<span class="mono">max_tokens</span> 必填），
Ollama 的 <span class="mono">/api/generate</span> 又是另一套，但<strong>每个适配器最后吐出的都是同一个 <span class="mono">GenerateReqInput</span></strong>。
这就是“多方言、单内核”在代码层面的样子。</p>

<p>这就是“多方言、单内核”在代码层面的样子。值得补一句：表里只列了最有代表性的几条路由，OpenAI 协议本身还包含
<span class="mono">/v1/embeddings</span>（向量化）、<span class="mono">/v1/models</span>（列出模型）等更多端点，它们各自也由对应的 serving 类接手。
但无论端点多少，<strong>套路始终一致</strong>：每个端点对应一个 serving 类，类里完成“入口翻译 → 原生请求 → 共享运行时 → 出口翻译”这条流水线。
理解了 chat 这一条，其余端点你都能<strong>举一反三</strong>。</p>

<h2>原生 /generate 与 /v1/chat/completions 的对照</h2>
<p>为了真正体会“翻译”发生在哪，把<strong>原生接口</strong>和<strong>OpenAI 兼容接口</strong>并排看一眼最直观。它们最终驱动的是同一套运行时，
区别只在<strong>请求/响应的外壳长什么样</strong>——原生接口是 SGLang 自己的形状，OpenAI 接口是为了兼容生态而模仿的形状。</p>

<div class="cols">
  <div class="col"><h4>原生 <span class="mono">POST /generate</span></h4><p>SGLang 自家形状：body 直接给 <span class="mono">text</span> 或 <span class="mono">input_ids</span> 加 <span class="mono">sampling_params</span>，
  响应也是 SGLang 原生 JSON。<strong>不套聊天模板</strong>、字段最贴近内核，是第 13 课讲的原生 HTTP 入口，适合自定义客户端与前端 RuntimeEndpoint。</p></div>
  <div class="col"><h4><span class="mono">/v1/chat/completions</span></h4><p>OpenAI 形状：body 是 <span class="mono">messages</span> 角色列表，由 <span class="mono">OpenAIServingChat</span> <strong>套用聊天模板</strong>后翻成
  <span class="mono">GenerateReqInput</span>；响应被映射成 <span class="mono">chat.completion</span> 对象或 <span class="mono">chat.completion.chunk</span> SSE 流（第 17 课）。任何 OpenAI 客户端开箱即用。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 770 250" role="img" aria-label="OpenAI 的 messages 角色列表先经聊天模板拼成单段 prompt 字符串，再被分词成 token">
    <text x="24" y="24" style="font-weight:700;fill:var(--muted)">之前 · messages 列表</text>
    <rect x="24" y="36" width="190" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="34" y="56" class="mono" style="font-size:11px">{role: system, ...}</text>
    <rect x="24" y="74" width="190" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="34" y="94" class="mono" style="font-size:11px">{role: user, ...}</text>
    <rect x="24" y="112" width="190" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="34" y="132" class="mono" style="font-size:11px">{role: assistant, ...}</text>
    <line x1="214" y1="89" x2="272" y2="89" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="272" y="52" width="206" height="74" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="375" y="80" text-anchor="middle" style="font-weight:700;fill:var(--amber)">聊天模板</text>
    <text x="375" y="102" text-anchor="middle" class="mono" style="font-size:10px">&lt;|im_start|&gt;role…&lt;|im_end|&gt;</text>
    <line x1="478" y1="89" x2="536" y2="89" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="536" y="52" width="210" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="641" y="71" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">单段 prompt 字符串</text>
    <text x="641" y="88" text-anchor="middle" class="mono" style="font-size:10px">"&lt;|im_start|&gt;system…"</text>
    <line x1="641" y1="96" x2="641" y2="134" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="536" y="134" width="210" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="641" y="159" text-anchor="middle" class="mono" style="fill:var(--teal);font-size:12px">tokens [12, 8801, …]</text>
    <text x="375" y="200" text-anchor="middle" style="fill:var(--faint);font-size:12px">之后：单段字符串 → 分词成 token 交给运行时</text>
  </svg>
  <div class="figcap"><b>图 2 · 套用 chat 模板</b> — OpenAI 给的是一串 <span class="mono">{role, content}</span> 消息（system / user / assistant）；适配器按模型的聊天模板（如 <span class="mono">&lt;|im_start|&gt;role…&lt;|im_end|&gt;</span>）把它们拼成<strong>单段 prompt 字符串</strong>，再分词成 token——这正是 chat 接口比原生 <span class="mono">/generate</span> 多出来的那一步。</div>
</div>

<p>举个具体例子。OpenAI 的 <span class="mono">messages</span> 就是一串带角色的对象，例如
<span class="mono">[{"role":"system","content":"你是助手"}, {"role":"user","content":"你好"}]</span>；
套用聊天模板后，会被拼成一段带特殊标记的纯文本，例如
<span class="mono">&lt;|im_start|&gt;system 你是助手&lt;|im_end|&gt;&lt;|im_start|&gt;user 你好&lt;|im_end|&gt;&lt;|im_start|&gt;assistant</span>，
最后这段字符串再被<strong>分词成 token</strong> 交给运行时。原生 <span class="mono">/generate</span> 直接收 prompt 文本，省掉的正是这一步。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/openai/serving_chat.py ::OpenAIServingChat</span><span class="ln">把 OpenAI 聊天请求适配成 SGLang 内部请求</span></div>
  <pre><span class="kw">class</span> OpenAIServingChat(OpenAIServingBase):
    <span class="cm"># adapts OpenAI /v1/chat/completions to SGLang's internal request</span>
    <span class="kw">def</span> _convert_to_internal_request(self, request):
        <span class="cm"># 1) apply the chat template to `messages` -&gt; one prompt string</span>
        <span class="cm"># 2) map OpenAI params (temperature, max_tokens, ...) -&gt; sampling_params</span>
        <span class="cm"># 3) return a GenerateReqInput the engine understands</span>
        ...</pre>
</div>

<p>把这层对照想透，你就会明白一个工程上的取舍：原生接口<strong>更贴近内核、开销更小、字段最灵活</strong>，但只有了解 SGLang 的人会用；
OpenAI 接口<strong>多了一层翻译、要套聊天模板</strong>，却换来了<strong>整个生态的即插即用</strong>。两者并存、共享同一套运行时，
正是 SGLang 的聪明之处——它没有逼你二选一，而是同时提供“给内行的原生快路径”和“给生态的兼容外壳”，
你按场景挑用即可。这与第 13 课“离线 Engine vs 在线 server”是同一种设计哲学：<strong>入口形态可以多样，运行时核心只写一遍</strong>。</p>

<h2>两个“OpenAI 方向”：别搞混 A 与 B</h2>
<p>这里必须把一个<strong>极易混淆</strong>的点钉死，它直接关系到你对第 12 课和本课的理解。“OpenAI” 在 SGLang 里出现过<strong>两次</strong>，
但它们是<strong>两个完全独立、方向相反</strong>的东西：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>两个独立的 OpenAI 方向，方向相反，切勿混为一谈</b></div>
  <div class="cells"><span class="lab">方向 A（第 12 课）</span><span class="cell">SGLang <strong>前端程序</strong></span><span class="sep">→</span><span class="cell hl">向外调用 OpenAI 当后端</span><span class="q">你的程序是客户端</span></div>
  <div class="cells"><span class="lab">方向 B（本课）</span><span class="cell">OpenAI <strong>客户端</strong></span><span class="sep">→</span><span class="cell hl">向内调用 SGLang 服务器</span><span class="q">你的部署当 OpenAI 端点</span></div>
</div>

<p>说清楚这两者。<strong>方向 A</strong>（第 12 课）：你用 SGLang DSL 写的前端程序，把 OpenAI 当成<strong>后端</strong>去调用——
此时<strong>你的程序是客户端</strong>，OpenAI 在远端出 token。<strong>方向 B</strong>（就是本课）：一个 OpenAI <strong>客户端</strong>
（别人的 <span class="mono">openai</span> SDK、LangChain）反过来调用<strong>你的 SGLang 服务器</strong>——此时<strong>你的 SGLang 部署伪装成一个 OpenAI 端点</strong>，
对方只要改一下 <span class="mono">base_url</span> 就接进来了。一个是“SGLang 打 OpenAI”，一个是“OpenAI 打 SGLang”，
方向截然相反。本课讲的<strong>始终是方向 B</strong>：让你的部署戴上 OpenAI 的面具，于是整个生态的工具都能插进来。
把这两个方向分清楚，你就不会在“到底谁是客户端、谁是服务器”上迷路。</p>

<p>把这两个方向分清楚，你就不会在“到底谁是客户端、谁是服务器”上迷路。一个判别小窍门：看<strong>token 在哪边产生</strong>。
方向 A 里 token 在<strong>远端的 OpenAI</strong> 生成，你的 SGLang 程序只是发问、收答；方向 B 里 token 在<strong>你自己的 SGLang 运行时</strong>生成，
外面那个 OpenAI 客户端只是发问、收答。同样一个名字、同样一套 HTTP 形状，<strong>谁在算</strong>这件事却完全相反——抓住这一点，两个方向就再也混不到一起。</p>

<p>顺带说一句它们为什么会“撞名”。OpenAI 的 HTTP 接口因为生态庞大，已经事实上成了大模型服务的<strong>通用方言</strong>：
既有海量客户端按它发请求（所以 SGLang 要在方向 B 扮演这个端点），也有海量服务按它提供能力（所以 SGLang 在方向 A 能把它当后端调）。
正因为这套接口<strong>两头都流行</strong>，“OpenAI” 才会在 SGLang 的入口（本课）和后端（第 12 课）两处同时出现。
理解了这层背景，你看任何“X 兼容”的系统时，都该先问一句：<strong>这次我是在实现这套协议，还是在消费这套协议？</strong></p>

<p>下面这段就是兼容层的真实骨架——<span class="mono">OpenAIServingChat</span> 类，以及它把一个 OpenAI 形状的
<span class="mono">ChatCompletionRequest</span> 翻译成原生 <span class="mono">GenerateReqInput</span>、再交给 <span class="mono">tokenizer_manager.generate_request</span> 的过程：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/openai/serving_chat.py ::OpenAIServingChat</span><span class="ln">OpenAI 形状 → 原生 GenerateReqInput → 前门</span></div>
  <pre><span class="kw">class</span> OpenAIServingChat(OpenAIServingBase):
    <span class="cm"># /v1/chat/completions 的处理器</span>

    <span class="kw">def</span> _convert_to_internal_request(self, request, raw_request=None):
        <span class="cm"># 把 messages 套用聊天模板，拼成单段 prompt</span>
        processed_messages = self._process_messages(request, is_multimodal)
        sampling_params = request.to_sampling_params(...)  <span class="cm"># OpenAI 参数 → 采样参数</span>
        <span class="cm"># 拼成运行时唯一听得懂的“本店语言”：原生 GenerateReqInput（第 16 课）</span>
        adapted_request = GenerateReqInput(
            <span class="kw">**</span>prompt_kwargs, sampling_params=sampling_params,
            stream=request.stream, rid=request.rid, ...)
        <span class="kw">return</span> adapted_request, request

    <span class="kw">async def</span> _generate_chat_stream(self, adapted_request, request, raw_request):
        <span class="cm"># 交给 TokenizerManager（第 14 课）——和原生入口走完全相同的快路径</span>
        <span class="kw">async for</span> content <span class="kw">in</span> self.tokenizer_manager.generate_request(
                adapted_request, raw_request):
            <span class="kw">yield</span> content        <span class="cm"># 再把输出映射回 chat.completion.chunk（第 17 课）</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>兼容层 = 翻译官</strong>：SGLang 服务器同时会说多种<strong>协议</strong>，让现成客户端<strong>不改代码</strong>就能接入；它只做 schema 翻译 + 聊天模板，<strong>不重新实现引擎</strong>。</li>
    <li><strong>请求路径</strong>：客户端 SDK → 协议适配器 → 原生 <span class="mono">GenerateReqInput</span>（第 16 课）→ <strong>同一套运行时</strong>（TokenizerManager，第 14 课）→ 适配器格式化响应（SSE，第 17 课）。</li>
    <li><strong>路由映射</strong>：<span class="mono">/v1/chat/completions</span>→<span class="mono">OpenAIServingChat</span>、<span class="mono">/v1/completions</span>→<span class="mono">OpenAIServingCompletion</span>、<span class="mono">/v1/messages</span>→Anthropic、<span class="mono">/api/generate</span>→Ollama，终点都是 <span class="mono">GenerateReqInput</span>。</li>
    <li><strong>两个 OpenAI 方向</strong>：A（第 12 课）是 SGLang 程序<strong>向外</strong>调 OpenAI 当后端；B（本课）是 OpenAI 客户端<strong>向内</strong>调 SGLang 服务器。方向相反，切勿混淆。</li>
    <li><strong>为什么薄适配器</strong>：运行时协议无关，新增协议很便宜，最热的推理快路径始终共享、只优化一遍。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson we saw the runtime's <strong>front door</strong> — the TokenizerManager. But one question it can't answer: out there
are clients speaking the <strong>OpenAI dialect</strong>, the <strong>Anthropic dialect</strong>, the <strong>Ollama dialect</strong>
— how can they plug into one runtime <strong>without changing a line</strong>? The answer is today's star — the
<strong>compat layer</strong>. It makes the SGLang server <strong>speak several protocols at once</strong>, so <strong>existing
clients work unchanged</strong>.
</p>

<p>Why a whole lesson on the compat layer? Because it is the <strong>seam</strong> that lets SGLang drop into the existing ecosystem.
You don't rewrite an SDK for SGLang, nor make your team learn a new API — just point an OpenAI client's
<span class="mono">base_url</span> at your SGLang deployment and <strong>it runs immediately</strong>. LangChain, the
<span class="mono">openai</span> Python package, any tool built around the OpenAI API — all plug in seamlessly. This thin adapter
layer is the <strong>master key</strong> by which a private deployment takes over the whole ecosystem.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the compat layer as <strong>a row of multilingual translators standing at the door</strong>. Guests each speak their own
  dialect: some OpenAI, some Anthropic, some Ollama. The translators convert <strong>each dialect into the one house language</strong>
  the kitchen understands — the native <span class="mono">GenerateReqInput</span> (Lesson 16). The kitchen (the runtime) has no idea
  which dialect a guest originally spoke; it just cooks from that standard ticket. When the dish is ready, the translators
  <strong>translate the same dish back into each guest's dialect</strong> and serve it. Guests think "this place speaks my language!"
  — but there is only one kitchen and one house language; only <strong>the translators at the door</strong> are multilingual.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The compat layer lives under <span class="mono">entrypoints/openai/</span> and friends, built from a set of <strong>serving classes</strong>.
  Take OpenAI: <span class="mono">/v1/chat/completions</span> is handled by <span class="mono">OpenAIServingChat</span>,
  <span class="mono">/v1/completions</span> by <span class="mono">OpenAIServingCompletion</span>, plus <span class="mono">/v1/embeddings</span> and more.
  Every serving class does the same job: <strong>translate</strong> an <strong>OpenAI-shaped</strong> request — apply the chat template,
  build sampling params — into a <strong>native <span class="mono">GenerateReqInput</span></strong> (Lesson 16) handed to the
  TokenizerManager (Lesson 14); then <strong>map</strong> the runtime's outputs <strong>back</strong> into OpenAI response objects or SSE
  chunks (<span class="mono">chat.completion.chunk</span>). The Anthropic (<span class="mono">/v1/messages</span>) and Ollama adapters do the
  same for their own schemas.
</div>

<h2>A request's protocol journey</h2>
<p>Walk the whole path in your head first. A request's journey is: <strong>client SDK → protocol adapter → native request → the same
runtime → adapter formats the response</strong>. Note the middle stretch — <strong>"the same runtime"</strong> — whichever dialect the
entry spoke, it converges here onto one shared fast path. The adapter only translates at the <strong>outermost layer</strong>; it never
reimplements the engine.</p>

<p>That is the essence of the compat-layer design: the runtime itself is <strong>protocol-agnostic</strong> — it only knows
<span class="mono">GenerateReqInput</span>. The adapter does just two things — <strong>schema translation</strong> (map OpenAI/Anthropic/Ollama
fields onto native fields) and <strong>chat templating</strong> (assemble the messages list into a single prompt per the model's template).
Because its job is so narrow, <strong>adding a protocol is cheap</strong>: write one new adapter. The hottest inference fast path stays
<strong>shared and optimized once</strong>. If every new protocol meant rewriting the engine, that would be an engineering disaster — the
compat layer sidesteps it elegantly with "thin adapters + shared core."</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Client</span><span class="name">OpenAI SDK / LangChain / curl</span></div><div class="ld">Speaks the OpenAI dialect: posts <span class="mono">/v1/chat/completions</span> with a <span class="mono">messages</span> list. Just point <span class="mono">base_url</span> at SGLang — <strong>no code change</strong>.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Adapter</span><span class="name">Protocol layer OpenAI / Anthropic / Ollama</span></div><div class="ld">The serving class does <strong>schema translation + chat templating</strong>, shaping the dialect request into native form. New protocol = one more adapter.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">House language</span><span class="name">GenerateReqInput (Lesson 16)</span></div><div class="ld">The only native request the runtime understands: token/text + sampling params + rid. <strong>Protocol-agnostic</strong>; three dialects converge into one here.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Front door</span><span class="name">TokenizerManager (Lesson 14)</span></div><div class="ld">Takes the native request, tokenizes, stamps a rid, ships it to the scheduler via ZMQ. The <strong>shared fast path</strong> starts here, identical for all protocols.</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 770 270" role="img" aria-label="three protocols each take their own adapter; after schema translation and chat templating they converge into one GenerateReqInput that the engine core sees as a single unified form">
    <text x="24" y="24" style="font-weight:700;fill:var(--muted)">Three protocol entries</text>
    <rect x="24" y="36" width="196" height="48" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="122" y="58" text-anchor="middle" style="font-weight:700;fill:var(--blue)">OpenAI</text>
    <text x="122" y="75" text-anchor="middle" class="mono" style="font-size:11px">/v1/chat/completions</text>
    <rect x="24" y="98" width="196" height="48" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="122" y="120" text-anchor="middle" style="font-weight:700;fill:var(--teal)">Anthropic</text>
    <text x="122" y="137" text-anchor="middle" class="mono" style="font-size:11px">/v1/messages</text>
    <rect x="24" y="160" width="196" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="122" y="182" text-anchor="middle" style="font-weight:700;fill:var(--amber)">Ollama</text>
    <text x="122" y="199" text-anchor="middle" class="mono" style="font-size:11px">/api/generate</text>
    <rect x="288" y="60" width="150" height="124" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="363" y="106" text-anchor="middle" style="font-weight:700;fill:var(--muted)">Adapter layer</text>
    <text x="363" y="128" text-anchor="middle" style="fill:var(--faint);font-size:12px">schema translation</text>
    <text x="363" y="146" text-anchor="middle" style="fill:var(--faint);font-size:12px">+ chat template</text>
    <line x1="220" y1="60" x2="288" y2="100" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="220" y1="122" x2="288" y2="122" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="220" y1="184" x2="288" y2="144" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="506" y="84" width="208" height="76" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="610" y="114" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">One internal request</text>
    <text x="610" y="134" text-anchor="middle" class="mono" style="font-size:12px">GenerateReqInput</text>
    <line x1="438" y1="122" x2="506" y2="122" style="stroke:var(--accent);stroke-width:1.5"/>
    <text x="610" y="186" text-anchor="middle" style="fill:var(--muted);font-size:12px">the engine core sees only this unified form</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Three protocols → one internal request</b> — OpenAI, Anthropic and Ollama each take their own adapter; after <span class="mono">schema</span> translation and chat templating they all converge into one <span class="mono">GenerateReqInput</span>. The engine core sees only this unified form and can no longer tell which dialect the request originally spoke.</div>
</div>

<h2>Route → serving class → native mapping</h2>
<p>In code, each HTTP route is bound to a serving class that does the translating. The table below aligns "the route the world sees"
with "the class that takes it inside" — at a glance you see the compat layer's <strong>division-of-labor map</strong>: different dialects
take different adapters, but every arrow ends at the same <span class="mono">GenerateReqInput</span>.</p>

<table class="t">
  <tr><th>HTTP route</th><th>Serving class</th><th>Native form produced</th></tr>
  <tr><td class="mono">/v1/chat/completions</td><td class="mono">OpenAIServingChat</td><td>apply chat template → <span class="mono">GenerateReqInput</span></td></tr>
  <tr><td class="mono">/v1/completions</td><td class="mono">OpenAIServingCompletion</td><td>raw prompt → <span class="mono">GenerateReqInput</span></td></tr>
  <tr><td class="mono">/v1/messages</td><td>Anthropic adapter</td><td>messages → <span class="mono">GenerateReqInput</span></td></tr>
  <tr><td class="mono">/api/generate</td><td>Ollama adapter</td><td>Ollama fields → <span class="mono">GenerateReqInput</span></td></tr>
</table>

<p>A consistent pattern is worth naming: <strong>routes differ in shape, the destination is identical</strong>.
<span class="mono">OpenAIServingChat</span> must "apply the chat template" first, because the chat endpoint gives a list of role messages that
must be assembled into a single prompt per the model's own template; <span class="mono">/v1/completions</span> gets prompt text directly,
so its adapter skips templating. Anthropic's <span class="mono">/v1/messages</span> names fields differently (system kept separate,
<span class="mono">max_tokens</span> required), and Ollama's <span class="mono">/api/generate</span> is yet another shape — but
<strong>every adapter emits the same <span class="mono">GenerateReqInput</span></strong>. That is "many dialects, one core" at the code level.</p>

<h2>Native /generate vs /v1/chat/completions</h2>
<p>To really feel <em>where</em> the "translation" happens, put the <strong>native interface</strong> and the <strong>OpenAI-compatible
interface</strong> side by side. They ultimately drive the same runtime; the difference is only <strong>what the request/response envelope
looks like</strong> — native is SGLang's own shape, OpenAI is a shape mimicked for ecosystem compatibility.</p>

<div class="cols">
  <div class="col"><h4>Native <span class="mono">POST /generate</span></h4><p>SGLang's own shape: the body gives <span class="mono">text</span> or <span class="mono">input_ids</span> plus <span class="mono">sampling_params</span>,
  and the response is native SGLang JSON. <strong>No chat template</strong>, fields closest to the core — the native HTTP entry from Lesson 13, ideal for custom clients and the front-end RuntimeEndpoint.</p></div>
  <div class="col"><h4><span class="mono">/v1/chat/completions</span></h4><p>OpenAI shape: the body is a <span class="mono">messages</span> role list that <span class="mono">OpenAIServingChat</span> <strong>runs through the chat template</strong> and translates into
  <span class="mono">GenerateReqInput</span>; the response is mapped to a <span class="mono">chat.completion</span> object or a <span class="mono">chat.completion.chunk</span> SSE stream (Lesson 17). Any OpenAI client works out of the box.</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 770 250" role="img" aria-label="OpenAI's messages role list is first assembled by the chat template into a single prompt string, then tokenized into tokens">
    <text x="24" y="24" style="font-weight:700;fill:var(--muted)">Before · messages list</text>
    <rect x="24" y="36" width="190" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="34" y="56" class="mono" style="font-size:11px">{role: system, ...}</text>
    <rect x="24" y="74" width="190" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="34" y="94" class="mono" style="font-size:11px">{role: user, ...}</text>
    <rect x="24" y="112" width="190" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="34" y="132" class="mono" style="font-size:11px">{role: assistant, ...}</text>
    <line x1="214" y1="89" x2="272" y2="89" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="272" y="52" width="206" height="74" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="375" y="80" text-anchor="middle" style="font-weight:700;fill:var(--amber)">chat template</text>
    <text x="375" y="102" text-anchor="middle" class="mono" style="font-size:10px">&lt;|im_start|&gt;role…&lt;|im_end|&gt;</text>
    <line x1="478" y1="89" x2="536" y2="89" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="536" y="52" width="210" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="641" y="71" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">single prompt string</text>
    <text x="641" y="88" text-anchor="middle" class="mono" style="font-size:10px">"&lt;|im_start|&gt;system…"</text>
    <line x1="641" y1="96" x2="641" y2="134" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="536" y="134" width="210" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="641" y="159" text-anchor="middle" class="mono" style="fill:var(--teal);font-size:12px">tokens [12, 8801, …]</text>
    <text x="375" y="200" text-anchor="middle" style="fill:var(--faint);font-size:12px">After: one string → tokenized and handed to the runtime</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Applying the chat template</b> — OpenAI gives a list of <span class="mono">{role, content}</span> messages (system / user / assistant); the adapter assembles them per the model's chat template (e.g. <span class="mono">&lt;|im_start|&gt;role…&lt;|im_end|&gt;</span>) into a <strong>single prompt string</strong>, then tokenizes it — exactly the extra step the chat endpoint has over native <span class="mono">/generate</span>.</div>
</div>

<p>A concrete example. OpenAI's <span class="mono">messages</span> is just a list of role-tagged objects, e.g.
<span class="mono">[{"role":"system","content":"You are helpful"}, {"role":"user","content":"hi"}]</span>;
after applying the chat template it becomes one marked-up plain-text string, e.g.
<span class="mono">&lt;|im_start|&gt;system You are helpful&lt;|im_end|&gt;&lt;|im_start|&gt;user hi&lt;|im_end|&gt;&lt;|im_start|&gt;assistant</span>,
and that string is finally <strong>tokenized into tokens</strong> for the runtime. Native <span class="mono">/generate</span> takes prompt text directly, skipping exactly this step.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/openai/serving_chat.py ::OpenAIServingChat</span><span class="ln">adapts an OpenAI chat request into SGLang's internal request</span></div>
  <pre><span class="kw">class</span> OpenAIServingChat(OpenAIServingBase):
    <span class="cm"># adapts OpenAI /v1/chat/completions to SGLang's internal request</span>
    <span class="kw">def</span> _convert_to_internal_request(self, request):
        <span class="cm"># 1) apply the chat template to `messages` -&gt; one prompt string</span>
        <span class="cm"># 2) map OpenAI params (temperature, max_tokens, ...) -&gt; sampling_params</span>
        <span class="cm"># 3) return a GenerateReqInput the engine understands</span>
        ...</pre>
</div>

<h2>Two "OpenAI directions": don't conflate A and B</h2>
<p>Here we must nail down a <strong>very easy to confuse</strong> point that directly bears on understanding both Lesson 12 and this one.
"OpenAI" appears <strong>twice</strong> in SGLang, but they are <strong>two completely independent, opposite-facing</strong> things:</p>

<div class="cellgroup">
  <div class="cg-cap"><b>Two independent OpenAI directions, opposite-facing — never conflate them</b></div>
  <div class="cells"><span class="lab">Direction A (Lesson 12)</span><span class="cell">SGLang <strong>frontend program</strong></span><span class="sep">→</span><span class="cell hl">calls OUT to OpenAI as backend</span><span class="q">your program is the client</span></div>
  <div class="cells"><span class="lab">Direction B (this lesson)</span><span class="cell">OpenAI <strong>client</strong></span><span class="sep">→</span><span class="cell hl">calls IN to the SGLang server</span><span class="q">your deployment is the OpenAI endpoint</span></div>
</div>

<p>Spell them out. <strong>Direction A</strong> (Lesson 12): a frontend program you wrote in the SGLang DSL calls OpenAI as a
<strong>backend</strong> — here <strong>your program is the client</strong> and OpenAI produces tokens remotely. <strong>Direction B</strong>
(this lesson): an OpenAI <strong>client</strong> (someone's <span class="mono">openai</span> SDK, LangChain) calls <strong>your SGLang server</strong>
in turn — here <strong>your SGLang deployment masquerades as an OpenAI endpoint</strong>, and they plug in by just changing
<span class="mono">base_url</span>. One is "SGLang calling OpenAI," the other "OpenAI calling SGLang" — opposite directions. This lesson is
<strong>always Direction B</strong>: put an OpenAI mask on your deployment so the whole ecosystem of tools plugs in. Keep these two
straight and you'll never get lost over "who is the client and who is the server."</p>

<p>Below is the compat layer's real skeleton — the <span class="mono">OpenAIServingChat</span> class, and how it translates an
OpenAI-shaped <span class="mono">ChatCompletionRequest</span> into a native <span class="mono">GenerateReqInput</span> and hands it to
<span class="mono">tokenizer_manager.generate_request</span>:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/openai/serving_chat.py ::OpenAIServingChat</span><span class="ln">OpenAI shape → native GenerateReqInput → front door</span></div>
  <pre><span class="kw">class</span> OpenAIServingChat(OpenAIServingBase):
    <span class="cm"># Handler for /v1/chat/completions requests</span>

    <span class="kw">def</span> _convert_to_internal_request(self, request, raw_request=None):
        <span class="cm"># Apply the chat template, assembling messages into a single prompt</span>
        processed_messages = self._process_messages(request, is_multimodal)
        sampling_params = request.to_sampling_params(...)  <span class="cm"># OpenAI params → sampling params</span>
        <span class="cm"># Build the only "house language" the runtime knows: native GenerateReqInput (Lesson 16)</span>
        adapted_request = GenerateReqInput(
            <span class="kw">**</span>prompt_kwargs, sampling_params=sampling_params,
            stream=request.stream, rid=request.rid, ...)
        <span class="kw">return</span> adapted_request, request

    <span class="kw">async def</span> _generate_chat_stream(self, adapted_request, request, raw_request):
        <span class="cm"># Hand to the TokenizerManager (Lesson 14) — the SAME fast path as the native entry</span>
        <span class="kw">async for</span> content <span class="kw">in</span> self.tokenizer_manager.generate_request(
                adapted_request, raw_request):
            <span class="kw">yield</span> content        <span class="cm"># then map outputs back to chat.completion.chunk (Lesson 17)</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Compat layer = translators</strong>: the SGLang server speaks several <strong>protocols</strong> so existing clients work <strong>unchanged</strong>; it only does schema translation + chat templating, <strong>not a new engine</strong>.</li>
    <li><strong>Request path</strong>: client SDK → protocol adapter → native <span class="mono">GenerateReqInput</span> (Lesson 16) → <strong>the same runtime</strong> (TokenizerManager, Lesson 14) → adapter formats the response (SSE, Lesson 17).</li>
    <li><strong>Route map</strong>: <span class="mono">/v1/chat/completions</span>→<span class="mono">OpenAIServingChat</span>, <span class="mono">/v1/completions</span>→<span class="mono">OpenAIServingCompletion</span>, <span class="mono">/v1/messages</span>→Anthropic, <span class="mono">/api/generate</span>→Ollama — all ending at <span class="mono">GenerateReqInput</span>.</li>
    <li><strong>Two OpenAI directions</strong>: A (Lesson 12) is an SGLang program calling <strong>OUT</strong> to OpenAI as backend; B (this lesson) is an OpenAI client calling <strong>IN</strong> to the SGLang server. Opposite — don't conflate.</li>
    <li><strong>Why thin adapters</strong>: the runtime is protocol-agnostic, adding a protocol is cheap, and the hottest inference fast path is always shared and optimized once.</li>
  </ul>
</div>
""",
}

LESSON_16 = {
    "zh": r"""
<p class="lead">
第 14 课我们看清了<strong>前门</strong>（TokenizerManager）、第 15 课看清了<strong>多协议翻译官</strong>，但有一件事一直被我们一带而过：
这三个进程<strong>不共享内存</strong>，它们之间到底<strong>传的是什么</strong>？答案就是本课的主角——<strong>io_struct.py</strong> 里那一组消息结构。
进程之间无法直接读对方的变量，只能<strong>互相投递消息</strong>；而 <span class="mono">io_struct.py</span> 正是用一批 Python <strong>dataclass</strong>
把这些消息<strong>定死了形状</strong>，再经 <strong>ZMQ</strong> 序列化后在进程间穿梭。
</p>

<p>为什么要专门为它开一课？因为这组结构其实是整个运行时的<strong>“类型系统”与线缆协议（wire protocol）</strong>。
后面每一个组件——调度器（第 18 课）、worker、反分词器（第 17 课）——彼此说话，<strong>说的都是这些 struct</strong>。
你只要把“一条请求在进程间如何变换形态”这条线索抓住，后面那些看似复杂的组件就会像串珠子一样被这根线穿起来。
反过来，如果不先认清这套消息，你读调度器源码时会被各种 <span class="mono">TokenizedGenerateReqInput</span>、<span class="mono">BatchTokenIDOutput</span>
绕晕——它们不是细节，而是<strong>骨架</strong>。</p>

<p>还有一个更深的角度：把通信<strong>显式化成数据结构</strong>，本身就是一种工程纪律。进程边界一旦用 dataclass 钉死，
“谁该传什么字段、谁负责填、谁负责读”就变成<strong>白纸黑字</strong>，可打印、可断点、可单测。相比起隐式的共享状态，
这种“一切皆消息”的设计虽然多写了几个类，却换来了<strong>可调试性与可演化性</strong>——这正是分布式系统里反复被验证的智慧。</p>

<p>不妨先把这套设计的三个关键词记在心里：<strong>形状、传输、身份</strong>。形状由 dataclass 负责——它规定每条消息有哪些字段、各是什么类型；
传输由 ZMQ 负责——它把对象序列化成字节、经 socket 送到另一进程、再反序列化还原；身份由 <span class="mono">rid</span> 负责——它给每条请求一个唯一编号，
让交错的批量回包能各归其位。这三件事一旦拆清楚，<span class="mono">io_struct.py</span> 这份文件读起来就不再是一堆零散的类，而是一套<strong>层次分明的协议</strong>。
本课接下来的每一节，都是围绕这三个关键词展开的。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把这套消息结构想成<strong>在各部门之间流转的标准化表单与信封</strong>。同一桩业务，每经过一个部门就被<strong>换成一种新的正式表单</strong>：
  前台收的是<strong>受理单</strong>（原始诉求：文本、采样要求、是否流式）；分词后变成一张<strong>工单</strong>（token id + 参数，送进车间）；
  车间产出<strong>结果清单</strong>（一批 token id）；最后誊清成一封<strong>打印好的信</strong>（解码后的文本）寄回。每一张表单的右上角都盖着<strong>同一个案件编号</strong>
  （<span class="mono">rid</span>），于是哪怕几十桩业务的表单在传送带上交错前进，也绝不会张冠李戴、寄错人。部门之间<strong>从不共用一个抽屉</strong>（不共享内存），
  全靠<strong>把表单递过去</strong>（传消息）来协作。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  <span class="mono">io_struct.py</span> 住在 <span class="mono">python/sglang/srt/managers/</span> 下，是一份纯粹的<strong>消息定义文件</strong>：里面几乎没有算法，
  只有一个个 <span class="mono">@dataclass</span>。它们沿着请求的生命周期排开——<span class="mono">GenerateReqInput</span>（前门收到的用户级请求）→
  <span class="mono">TokenizedGenerateReqInput</span>（分词后、<strong>真正跨 ZMQ 送往调度器</strong>的那个）→ <span class="mono">BatchTokenIDOutput</span>
  （调度器/worker 产出的批量 token id）→ <span class="mono">BatchStrOutput</span>（反分词器解码出的文本）。每个 struct 都明确标注了
  <strong>谁生产、谁消费</strong>。它们经 ZMQ 的 IPC socket（每个进程一个端口）序列化传输，靠 <span class="mono">rid</span> 把交错的批量输出<strong>对号入座</strong>回各自的请求。
  此外还有 abort（中止）、flush_cache（清缓存）、health（健康检查）等独立的<strong>控制消息</strong>。
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="进程间的消息类型：TokenizerManager、Scheduler、DetokenizerManager 三个进程之间流动的 dataclass 消息，每条箭头标注它承载的 struct">
    <text x="20" y="32" style="font-weight:700;fill:var(--muted)">三进程 · 各自独立内存 · 只靠投递 dataclass 协作</text>
    <text x="95" y="70" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">GenerateReqInput</text>
    <text x="95" y="86" text-anchor="middle" style="font-size:10px;fill:var(--muted)">客户端 → 前门（不跨进程）</text>
    <line x1="95" y1="92" x2="95" y2="116" style="stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="95,122 90,112 100,112" style="fill:var(--blue)"/>
    <rect x="20" y="122" width="150" height="62" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="95" y="150" text-anchor="middle" style="font-weight:700;font-size:13px">TokenizerManager</text>
    <text x="95" y="170" text-anchor="middle" style="fill:var(--muted);font-size:11px">前门 · 分词</text>
    <rect x="325" y="122" width="150" height="62" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="400" y="150" text-anchor="middle" style="font-weight:700;font-size:13px;fill:var(--accent-ink)">Scheduler</text>
    <text x="400" y="170" text-anchor="middle" style="fill:var(--muted);font-size:11px">调度 · 前向</text>
    <rect x="630" y="122" width="150" height="62" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="705" y="150" text-anchor="middle" style="font-weight:700;font-size:12.5px">DetokenizerManager</text>
    <text x="705" y="170" text-anchor="middle" style="fill:var(--muted);font-size:11px">反分词</text>
    <text x="247" y="118" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--ink)">TokenizedGenerateReqInput</text>
    <line x1="170" y1="140" x2="319" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="325,140 315,135 315,145" style="fill:var(--line)"/>
    <text x="552" y="118" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--ink)">BatchTokenIDOutput</text>
    <line x1="475" y1="140" x2="624" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="630,140 620,135 620,145" style="fill:var(--line)"/>
    <line x1="705" y1="184" x2="705" y2="236" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="705" y1="236" x2="95" y2="236" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="95" y1="236" x2="95" y2="190" style="stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="95,184 90,194 100,194" style="fill:var(--teal)"/>
    <text x="400" y="228" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--teal)">BatchStrOutput → 经前门投回客户端</text>
  </svg>
  <div class="figcap"><b>图 1 · 进程间的消息类型</b> — 三个进程不共享内存，只靠投递 dataclass 协作：<span class="mono">GenerateReqInput</span> 进前门后变成 <span class="mono">TokenizedGenerateReqInput</span> 送往调度器，调度器产出 <span class="mono">BatchTokenIDOutput</span> 交反分词器，最后 <span class="mono">BatchStrOutput</span> 经前门投回客户端。</div>
</div>

<h2>为什么是“传消息”而不是“共享内存”</h2>
<p>回忆第 2 课的<strong>三进程模型</strong>：主进程（TokenizerManager）、调度器子进程、反分词器子进程。它们是<strong>三个独立的操作系统进程</strong>，
各有各的内存空间——主进程里的一个 Python 对象，调度器进程<strong>看不见、也碰不到</strong>。这不是缺陷，而是刻意为之：拆进程能绕开
Python 的 GIL、让 CPU 与 GPU 真正重叠（第 21 课）。但代价是：它们不能再靠“读同一个变量”来协作，<strong>只能靠互相投递消息</strong>。</p>

<p>于是问题变成：消息长什么样、怎么过线？SGLang 的答案朴素而有效——用 <strong>dataclass 定义形状</strong>，用 <strong>ZMQ 做传输</strong>。
dataclass 让每条消息的字段一目了然、可读可打印；ZMQ 提供轻量的进程间 socket，把对象<strong>序列化</strong>成字节流送过去、再<strong>反序列化</strong>还原。
两者搭配，得到的是一套<strong>便宜、显式、好调试</strong>的跨进程通信。你甚至可以把某条消息打印出来，一眼看清它带了哪些 token、什么采样参数。</p>

<p>这里值得多想一层：序列化确实有开销——把对象拍扁成字节、再还原，并不是零成本。但 SGLang 之所以甘愿付这笔钱，是因为换回来的东西更值钱：
其一，三进程能在多核 CPU 上<strong>真正并行</strong>，绕开 GIL 的串行枷锁；其二，GPU 子进程在跑前向时，主进程能同时给下一批分词，<strong>CPU 与 GPU 重叠</strong>（第 21 课）；
其三，万一 GPU 子进程因显存溢出（OOM）崩溃，主进程仍然活着，能感知异常、清理挂起状态，而不是整个服务一起静默死掉。把通信显式化成消息，
正是把这些<strong>并行、重叠、容错</strong>的好处一并拿下的代价——而这笔代价，相对它换回的吞吐与稳健，是非常划算的。</p>

<div class="cols">
  <div class="col"><h4>共享内存？<strong>不</strong></h4><p>三个进程<strong>各有独立内存空间</strong>，一个进程的对象另一个进程读不到。这是拆进程绕开 GIL、让 CPU/GPU 重叠（第 21 课）必然付出的边界。</p></div>
  <div class="col"><h4>传消息？<strong>是</strong></h4><p>协作只能靠<strong>投递消息</strong>：用 <span class="mono">dataclass</span> 定形状、用 <strong>ZMQ</strong> 序列化过线。便宜、显式、可打印断点，是分布式协作的经典选择。</p></div>
</div>

<h2>一条请求的生命周期：消息形态的接力</h2>
<p>真正把这套设计讲活的，是<strong>跟着一条请求走一遍</strong>，看它每过一个进程边界就<strong>换一种 struct</strong>。这是一场严丝合缝的接力：
前一棒的“输出消息”恰好是后一棒的“输入消息”，而贯穿全程的接力棒，就是那个 <span class="mono">rid</span>。</p>

<div class="flow">
  <div class="node"><div class="nt">GenerateReqInput</div><div class="nd">前门收到的用户请求：文本+参数+stream</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">TokenizedGenerateReqInput</div><div class="nd">分词后，<strong>经 ZMQ 送往调度器</strong></div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">BatchTokenIDOutput</div><div class="nd">调度器/worker 产出的批量 token id</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">BatchStrOutput</div><div class="nd">反分词器解码出的文本</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">按 rid 投回客户端</div><div class="nd">交还正在 await 的那条请求</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="一次往返：请求经 ZMQ socket 在三进程间右行，响应再原路返回，构成一个回路；强调这些是纯 Python dataclass 经 IPC 序列化，而非进程间 HTTP">
    <text x="20" y="32" style="font-weight:700;fill:var(--muted)">一次往返 · 经 ZMQ IPC socket 收发</text>
    <text x="380" y="74" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">请求 → pickle 序列化 → ZMQ 送往下一进程</text>
    <line x1="109" y1="116" x2="109" y2="86" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="109" y1="86" x2="651" y2="86" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="651" y1="86" x2="651" y2="116" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="651,122 646,112 656,112" style="fill:var(--blue)"/>
    <rect x="24" y="120" width="170" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="109" y="148" text-anchor="middle" style="font-weight:700;font-size:13px">TokenizerManager</text>
    <text x="109" y="168" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--muted)">主进程</text>
    <rect x="295" y="120" width="170" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="380" y="148" text-anchor="middle" style="font-weight:700;font-size:13px">Scheduler</text>
    <text x="380" y="168" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--muted)">子进程</text>
    <rect x="566" y="120" width="170" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="651" y="148" text-anchor="middle" style="font-weight:700;font-size:12.5px">DetokenizerManager</text>
    <text x="651" y="168" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--muted)">子进程</text>
    <circle cx="244" cy="152" r="11" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="244" y="156" text-anchor="middle" class="mono" style="font-size:9px;fill:var(--amber)">zmq</text>
    <circle cx="515" cy="152" r="11" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="515" y="156" text-anchor="middle" class="mono" style="font-size:9px;fill:var(--amber)">zmq</text>
    <line x1="651" y1="188" x2="651" y2="246" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="651" y1="246" x2="109" y2="246" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="109" y1="246" x2="109" y2="190" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="109,184 104,194 114,194" style="fill:var(--teal)"/>
    <text x="380" y="238" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--teal)">响应 ← ZMQ 收到 → unpickle 还原 ← BatchStrOutput</text>
    <rect x="190" y="276" width="380" height="32" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="297" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">纯 Python dataclass · 经 IPC 序列化 · 不是进程间 HTTP</text>
  </svg>
  <div class="figcap"><b>图 2 · 一次往返</b> — 请求被 pickle 序列化、经 <span class="mono">zmq</span> socket 右行穿过三进程，响应再原路返回，构成一个回路。注意：进程间穿梭的是<strong>纯 Python dataclass</strong>（经 IPC 序列化），<strong>不是</strong>进程间的 HTTP——HTTP 只在最外层面向客户端。</div>
</div>

<p>一个具体的画面：主进程把 <span class="mono">TokenizedGenerateReqInput</span> 对象交给 ZMQ，ZMQ 用 <span class="mono">pickle</span>（部分通道用 <span class="mono">msgpack</span>）把它拍成字节流，经 IPC socket 送到调度器子进程，再在那头反序列化还原成同形状的对象——<strong>两端拿到的是同一个 dataclass，而不是一段 HTTP body</strong>。</p>

<p>把这条链路逐棒说清楚。<strong>第一棒</strong>，TokenizerManager（第 14 课）收到 <span class="mono">GenerateReqInput</span>——这是<strong>用户视角</strong>的请求：
里面是<strong>原始文本</strong>、采样参数、是否流式（<span class="mono">stream</span>），还可能带图片/音频。注意：<strong>这个对象不跨进程</strong>，它只活在主进程里。
<strong>第二棒</strong>，分词把文本变成 token id，打包成 <span class="mono">TokenizedGenerateReqInput</span>——<strong>这个</strong>才是真正被 ZMQ 序列化、
<strong>穿过进程边界送给调度器</strong>的消息。换句话说，<strong>跨过 ZMQ 到调度器的不是原始文本，而是已分词的 token id</strong>。这是本课最该记牢的一句话。</p>

<p>为什么要在“进”的那一刻就完成从文本到 token 的转换？因为这道转换是<strong>纯 CPU 的字符串活</strong>，理应留在主进程、挡在 GPU 进程之外（第 14 课）。
一旦过了 ZMQ 这道线，下游所有组件——调度器、各种注意力后端、KV 缓存——就<strong>只跟 token id 打交道</strong>，谁也不需要懂 UTF-8、不需要懂子词怎么拼。
这层“语言学负担”被一次性收口在最外层，内部世界因此保持<strong>纯数值、纯张量</strong>的干净。也正因如此，<span class="mono">TokenizedGenerateReqInput</span>
里虽然还留着一份 <span class="mono">input_text</span>，但那只是给人看的调试线索，真正驱动模型前向的，<strong>始终是 <span class="mono">input_ids</span></strong>。</p>

<p><strong>第三棒</strong>，调度器（第 18 课）把多条请求组成 <span class="mono">ScheduleBatch</span>（第 19 课）跑前向，产出 <span class="mono">BatchTokenIDOutput</span>——
注意它是<strong>批量</strong>的：一条消息里装着这一批<strong>多条请求</strong>各自的新 token id，靠 <span class="mono">rids</span> 列表标明每段属于谁。
<strong>第四棒</strong>，反分词器（第 17 课）把 token id 解码回文本，封成 <span class="mono">BatchStrOutput</span>（带 <span class="mono">output_strs</span>）。
<strong>最后一棒</strong>，结果流回 TokenizerManager，凭 <span class="mono">rid</span> 投递回那条正在 <span class="mono">await</span> 的协程，逐段交还客户端。</p>

<table class="t">
  <tr><th>消息 struct</th><th>关键字段</th><th>谁生产 → 谁消费</th></tr>
  <tr><td class="mono">GenerateReqInput</td><td>text / input_ids、sampling_params、stream、rid</td><td>客户端/兼容层 → TokenizerManager（<strong>不跨进程</strong>）</td></tr>
  <tr><td class="mono">TokenizedGenerateReqInput</td><td>rid、input_ids、sampling_params、stream</td><td>TokenizerManager →（<strong>ZMQ</strong>）→ Scheduler</td></tr>
  <tr><td class="mono">BatchTokenIDOutput</td><td>rids、decode_ids/output_ids、finished_reasons</td><td>Scheduler/worker →（ZMQ）→ Detokenizer</td></tr>
  <tr><td class="mono">BatchStrOutput</td><td>rids、output_strs、output_ids、finish 信息</td><td>Detokenizer →（ZMQ）→ TokenizerManager → 客户端</td></tr>
</table>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/io_struct.py ::BatchTokenIDOutput</span><span class="ln">Scheduler → Detokenizer：一步的原始 token 输出</span></div>
  <pre><span class="kw">@dataclasses.dataclass</span>
<span class="kw">class</span> BatchTokenIDOutput(BaseBatchReq):
    <span class="cm"># Scheduler -&gt; Detokenizer: one step's raw token outputs to be turned into text</span>
    finished_reasons: list      <span class="cm"># None / stop / length, per request</span>
    decoded_texts: list         <span class="cm"># text decoded so far (for incremental decode)</span>
    decode_ids: list            <span class="cm"># the token ids to detokenize</span>
    read_offsets: list          <span class="cm"># where to resume detokenizing each request</span>
    output_ids: list            <span class="cm"># the newly produced token ids</span>
    skip_special_tokens: list</pre>
</div>

<p>具体感受一下这条边的“批量”性质：某一步里调度器同时跑了 <span class="mono">rid=7</span>、<span class="mono">rid=3</span>、<span class="mono">rid=9</span> 三条请求，它就把这三条的新 token id 一起塞进<strong>同一个</strong> <span class="mono">BatchTokenIDOutput</span>，靠 <span class="mono">BaseBatchReq</span> 里的 <span class="mono">rids</span> 列表标明每段归谁；这条消息再经 ZMQ（<span class="mono">pickle</span>/<span class="mono">msgpack</span> 序列化）送到反分词器。</p>

<p>把这张表竖着读，你会发现一个优雅的规律：<strong>上一行的“消费者”往往就是下一行的“生产者”</strong>，四个 struct 首尾相接，
正好拼成请求从“人类文本”到“token”再回到“人类文本”的<strong>完整闭环</strong>。还要留意<strong>单数与复数</strong>的切换：进的方向是<strong>单条</strong>
（<span class="mono">GenerateReqInput</span>/<span class="mono">TokenizedGenerateReqInput</span>，带 <span class="mono">rid</span>），出的方向是<strong>批量</strong>
（<span class="mono">Batch*Output</span>，带 <span class="mono">rids</span>）——因为调度器是把多条请求<strong>攒成一批</strong>一起算的，输出自然是批量的，这正是连续批处理（第 18 课）的体现。</p>

<h2>rid：贯穿始终的“案件编号”</h2>
<p>这套消息能在三进程间<strong>不乱套</strong>，全靠一个朴素的字段：<span class="mono">rid</span>（request id，请求号）。它在 <span class="mono">GenerateReqInput</span>
处生成，被复制进 <span class="mono">TokenizedGenerateReqInput</span> 一路带去调度器，再随 <span class="mono">BatchTokenIDOutput</span>/<span class="mono">BatchStrOutput</span>
的 <span class="mono">rids</span> 列表流回来。因为调度器把多条请求<strong>攒成一批</strong>一起算，回来的输出是<strong>交错、批量</strong>的——某条消息里同时装着第 7 号、第 3 号、第 9 号的 token。
没有 <span class="mono">rid</span>，TokenizerManager 根本无法把这批交错输出<strong>拆开投回</strong>正确的等待协程。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>rid 如何让交错的批量输出对号入座</b></div>
  <div class="cells"><span class="lab">出</span><span class="cell">rid=7：tok</span><span class="cell">rid=3：tok</span><span class="cell hl">rid=9：tok</span><span class="sep">→</span><span class="cell">一条 BatchTokenIDOutput</span><span class="q">批量、交错</span></div>
  <div class="cells"><span class="lab">分</span><span class="cell hl">按 rid 拆</span><span class="sep">→</span><span class="cell">7→协程A</span><span class="cell">3→协程B</span><span class="cell">9→协程C</span><span class="q">各回各家</span></div>
</div>

<p>所以 <span class="mono">rid</span> 不是可有可无的元数据，而是这套“一切皆消息”体系能<strong>并发而不混乱</strong>的根。它像快递面单上的<strong>运单号</strong>：
包裹在分拣中心被打散、混装、又重新分发，全程能精准追踪、最终送到正确的人手里，靠的就是那串号。理解了 <span class="mono">rid</span> 的往返，
你就理解了 SGLang 异步路由（第 14 课）与流式回传（第 17 课）共同依赖的<strong>地基</strong>。下面这段就是那个真正跨 ZMQ 的核心消息的真实骨架：</p>

<p>最后把视角拉高一层。这组 struct 之所以重要，是因为它定义了运行时各组件之间<strong>唯一合法的对话方式</strong>：调度器（第 18 课）只会收到
<span class="mono">TokenizedGenerateReqInput</span>、只会吐出 <span class="mono">BatchTokenIDOutput</span>；反分词器（第 17 课）只认 <span class="mono">BatchTokenIDOutput</span>、
只产出 <span class="mono">BatchStrOutput</span>。换句话说，每个组件的<strong>输入输出契约</strong>都写在 <span class="mono">io_struct.py</span> 里，清清楚楚。
正因如此，当你后面去读 <span class="mono">ScheduleBatch</span>（第 19 课）、连续批处理（第 18 课）那些更复杂的代码时，只要先问一句“它收什么 struct、发什么 struct”，
就能迅速定位它在整条流水线里的位置。这份消息定义文件，因此是你<strong>读懂整个运行时的一把总钥匙</strong>，也是后续每一课都会反复回看的地图起点。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/io_struct.py ::TokenizedGenerateReqInput</span><span class="ln">真正跨 ZMQ 送往调度器的消息</span></div>
  <pre><span class="kw">@dataclass</span>
<span class="kw">class</span> TokenizedGenerateReqInput(BaseReq):   <span class="cm"># BaseReq 提供 rid 字段（跨进程身份证）</span>
    input_text: str                       <span class="cm"># 原始文本（便于调试/日志）</span>
    input_ids: Optional[array[int]]        <span class="cm"># 分词后的 token id —— 真正喂给模型的</span>
    mm_inputs: object                      <span class="cm"># 多模态输入（图像/音频）</span>
    sampling_params: SamplingParams        <span class="cm"># 温度/top_p/max_new_tokens 等</span>
    return_logprob: bool
    stream: bool                           <span class="cm"># 是否流式回传（第 17 课）</span>
    <span class="cm"># …随后经 ZMQ 序列化，跨进程送达调度器（第 18 课）</span></pre>
</div>

<p>注意这个结构里<strong>没有</strong>原始的 <span class="mono">text</span> 列表那种用户级形态，取而代之的是已经分好的 <span class="mono">input_ids</span>——
这正印证了那句关键话：<strong>过了 ZMQ 这道线，下游只认 token，不再碰人类文本</strong>。<span class="mono">input_text</span> 仍被保留，但只为调试与日志之便，真正驱动模型的是 <span class="mono">input_ids</span>。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>不共享内存，只传消息</strong>：三进程（第 2 课）各有独立内存，靠 <span class="mono">io_struct.py</span> 里的 <span class="mono">@dataclass</span> 消息 + <strong>ZMQ</strong> 序列化协作。</li>
    <li><strong>生命周期 = 消息接力</strong>：<span class="mono">GenerateReqInput</span> → <span class="mono">TokenizedGenerateReqInput</span> → <span class="mono">BatchTokenIDOutput</span> → <span class="mono">BatchStrOutput</span>，每过一个进程换一种 struct。</li>
    <li><strong>跨 ZMQ 的是 token，不是文本</strong>：真正送往调度器（第 18 课）的是已分词的 <span class="mono">TokenizedGenerateReqInput</span>（带 <span class="mono">input_ids</span>），<span class="mono">GenerateReqInput</span> 只活在主进程。</li>
    <li><strong>rid 贯穿始终</strong>：请求号让交错的<strong>批量</strong>输出（<span class="mono">rids</span> 列表）对号入座，投回正确的等待协程（第 14 课异步路由、第 17 课流式）。</li>
    <li><strong>它是运行时的“类型系统”</strong>：调度器、worker、反分词器都说这套 struct；abort/flush_cache/health 是独立的控制消息，同样是显式的跨进程信号。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Lesson 14 showed the <strong>front door</strong> (TokenizerManager) and Lesson 15 the <strong>multi-protocol translators</strong>, but one thing kept
slipping by: these three processes <strong>do not share memory</strong> — so what exactly <strong>travels between them</strong>? That is today's star: the
group of message structs in <strong>io_struct.py</strong>. Processes cannot read each other's variables; they can only <strong>pass messages</strong>. And
<span class="mono">io_struct.py</span> pins down the <strong>shape</strong> of those messages as Python <strong>dataclasses</strong>, serialized over <strong>ZMQ</strong>.
</p>

<p>Why a whole lesson for it? Because this set of structs is effectively the runtime's <strong>"type system" and wire protocol</strong>. Every later
component — the scheduler (Lesson 18), the worker, the detokenizer (Lesson 17) — talks to the others <strong>in these structs</strong>. Grab the single
thread of "how a request changes shape across processes," and those seemingly complex components string together like beads. Skip it, and the scheduler
source will drown you in <span class="mono">TokenizedGenerateReqInput</span> and <span class="mono">BatchTokenIDOutput</span> — they are not details, they are the <strong>skeleton</strong>.</p>

<p>There is a deeper angle too: making communication <strong>explicit as data structures</strong> is itself an engineering discipline. Once the process
boundary is pinned by dataclasses, "who sends which field, who fills it, who reads it" becomes <strong>black and white</strong> — printable, breakpoint-able,
unit-testable. Compared with implicit shared state, this "everything is a message" design costs a few extra classes but buys <strong>debuggability and
evolvability</strong> — a lesson distributed systems keep re-teaching.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of these structs as <strong>standardized forms and envelopes moving between departments</strong>. For one case, each department turns it into a
  <strong>new official form</strong>: the front desk takes an <strong>intake form</strong> (raw ask: text, sampling prefs, stream flag); tokenization turns it into a
  <strong>work order</strong> (token ids + params, sent to the floor); the floor produces a <strong>results sheet</strong> (a batch of token ids); finally it is typed up as a
  <strong>printed letter</strong> (decoded text) and mailed back. Every form is stamped with the <strong>same case number</strong> (<span class="mono">rid</span>) in the corner, so even with
  dozens of cases interleaving on the conveyor belt, nothing is misrouted. Departments <strong>never share a drawer</strong> (no shared memory) — they cooperate purely by
  <strong>handing forms across</strong> (passing messages).
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  <span class="mono">io_struct.py</span> lives under <span class="mono">python/sglang/srt/managers/</span> and is a pure <strong>message-definition file</strong>: almost no
  algorithms, just <span class="mono">@dataclass</span> after <span class="mono">@dataclass</span>. They line up along the request's lifecycle —
  <span class="mono">GenerateReqInput</span> (the user-facing request the front door receives) → <span class="mono">TokenizedGenerateReqInput</span> (post-tokenization, the one that
  <strong>actually crosses ZMQ to the scheduler</strong>) → <span class="mono">BatchTokenIDOutput</span> (batched token ids from scheduler/worker) →
  <span class="mono">BatchStrOutput</span> (text decoded by the detokenizer). Each struct marks <strong>who produces and who consumes</strong> it. They travel over ZMQ IPC
  sockets (a port per process), and the <span class="mono">rid</span> routes interleaved batched outputs back to the right request. There are also standalone
  <strong>control messages</strong>: abort, flush_cache, health.
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="The IPC message types: dataclass messages flowing between the three processes TokenizerManager, Scheduler and DetokenizerManager, each arrow labeled with the struct it carries">
    <text x="20" y="32" style="font-weight:700;fill:var(--muted)">Three processes · separate memory · cooperate only by passing dataclasses</text>
    <text x="95" y="70" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">GenerateReqInput</text>
    <text x="95" y="86" text-anchor="middle" style="font-size:10px;fill:var(--muted)">client → front door (no IPC)</text>
    <line x1="95" y1="92" x2="95" y2="116" style="stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="95,122 90,112 100,112" style="fill:var(--blue)"/>
    <rect x="20" y="122" width="150" height="62" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="95" y="150" text-anchor="middle" style="font-weight:700;font-size:13px">TokenizerManager</text>
    <text x="95" y="170" text-anchor="middle" style="fill:var(--muted);font-size:11px">front door · tokenize</text>
    <rect x="325" y="122" width="150" height="62" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="400" y="150" text-anchor="middle" style="font-weight:700;font-size:13px;fill:var(--accent-ink)">Scheduler</text>
    <text x="400" y="170" text-anchor="middle" style="fill:var(--muted);font-size:11px">schedule · forward</text>
    <rect x="630" y="122" width="150" height="62" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="705" y="150" text-anchor="middle" style="font-weight:700;font-size:12.5px">DetokenizerManager</text>
    <text x="705" y="170" text-anchor="middle" style="fill:var(--muted);font-size:11px">detokenize</text>
    <text x="247" y="118" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--ink)">TokenizedGenerateReqInput</text>
    <line x1="170" y1="140" x2="319" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="325,140 315,135 315,145" style="fill:var(--line)"/>
    <text x="552" y="118" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--ink)">BatchTokenIDOutput</text>
    <line x1="475" y1="140" x2="624" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="630,140 620,135 620,145" style="fill:var(--line)"/>
    <line x1="705" y1="184" x2="705" y2="236" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="705" y1="236" x2="95" y2="236" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="95" y1="236" x2="95" y2="190" style="stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="95,184 90,194 100,194" style="fill:var(--teal)"/>
    <text x="400" y="228" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--teal)">BatchStrOutput → routed back to client via front door</text>
  </svg>
  <div class="figcap"><b>Figure 1 · The IPC message types</b> — the three processes share no memory and cooperate only by passing dataclasses: a <span class="mono">GenerateReqInput</span> becomes a <span class="mono">TokenizedGenerateReqInput</span> sent to the scheduler, which emits a <span class="mono">BatchTokenIDOutput</span> to the detokenizer, whose <span class="mono">BatchStrOutput</span> is routed back to the client via the front door.</div>
</div>

<h2>Why message passing, not shared memory</h2>
<p>Recall the <strong>three-process model</strong> (Lesson 2): the main process (TokenizerManager), the scheduler subprocess, the detokenizer subprocess. They are
<strong>three separate OS processes</strong>, each with its own memory — an object in the main process is <strong>invisible and untouchable</strong> to the scheduler. This is
deliberate: splitting processes sidesteps Python's GIL and lets CPU and GPU truly overlap (Lesson 21). The price: they can no longer cooperate by "reading the same
variable" — only by <strong>handing messages back and forth</strong>.</p>

<p>So the question becomes: what do messages look like, and how do they cross the wire? SGLang's answer is plain and effective — define <strong>shape with
dataclasses</strong>, do <strong>transport with ZMQ</strong>. Dataclasses make every field obvious and printable; ZMQ provides lightweight inter-process sockets that
<strong>serialize</strong> an object to bytes, send it, and <strong>deserialize</strong> it on the other side. Together they yield communication that is <strong>cheap, explicit, and
debuggable</strong> — you can literally print a message and see its tokens and sampling params.</p>

<div class="cols">
  <div class="col"><h4>Shared memory? <strong>No</strong></h4><p>The three processes each own <strong>separate memory</strong>; one process's objects are unreadable to another. That is the boundary cost of splitting processes to dodge the GIL and overlap CPU/GPU (Lesson 21).</p></div>
  <div class="col"><h4>Message passing? <strong>Yes</strong></h4><p>Cooperation is purely by <strong>passing messages</strong>: shape with <span class="mono">dataclass</span>, cross the wire via <strong>ZMQ</strong> serialization. Cheap, explicit, printable and breakpoint-able — the classic distributed choice.</p></div>
</div>

<h2>A request's lifecycle: a relay of message shapes</h2>
<p>What brings this design alive is <strong>following one request</strong> and watching it <strong>swap structs</strong> at each process boundary. It is a tight relay: each
leg's "output message" is exactly the next leg's "input message," and the baton carried throughout is the <span class="mono">rid</span>.</p>

<div class="flow">
  <div class="node"><div class="nt">GenerateReqInput</div><div class="nd">user request at the door: text+params+stream</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">TokenizedGenerateReqInput</div><div class="nd">post-tokenize, <strong>sent via ZMQ to scheduler</strong></div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">BatchTokenIDOutput</div><div class="nd">batched token ids from scheduler/worker</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">BatchStrOutput</div><div class="nd">text decoded by the detokenizer</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">route by rid to client</div><div class="nd">hand back to the awaiting request</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="One round-trip: the request travels right through the three processes over ZMQ sockets and the response returns the same way, forming a loop; these are plain Python dataclasses serialized over IPC, not HTTP between processes">
    <text x="20" y="32" style="font-weight:700;fill:var(--muted)">One round-trip · sent and received over ZMQ IPC sockets</text>
    <text x="380" y="74" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">request → pickle-serialize → ZMQ to next process</text>
    <line x1="109" y1="116" x2="109" y2="86" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="109" y1="86" x2="651" y2="86" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="651" y1="86" x2="651" y2="116" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="651,122 646,112 656,112" style="fill:var(--blue)"/>
    <rect x="24" y="120" width="170" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="109" y="148" text-anchor="middle" style="font-weight:700;font-size:13px">TokenizerManager</text>
    <text x="109" y="168" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--muted)">main process</text>
    <rect x="295" y="120" width="170" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="380" y="148" text-anchor="middle" style="font-weight:700;font-size:13px">Scheduler</text>
    <text x="380" y="168" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--muted)">subprocess</text>
    <rect x="566" y="120" width="170" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="651" y="148" text-anchor="middle" style="font-weight:700;font-size:12.5px">DetokenizerManager</text>
    <text x="651" y="168" text-anchor="middle" class="mono" style="font-size:10.5px;fill:var(--muted)">subprocess</text>
    <circle cx="244" cy="152" r="11" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="244" y="156" text-anchor="middle" class="mono" style="font-size:9px;fill:var(--amber)">zmq</text>
    <circle cx="515" cy="152" r="11" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="515" y="156" text-anchor="middle" class="mono" style="font-size:9px;fill:var(--amber)">zmq</text>
    <line x1="651" y1="188" x2="651" y2="246" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="651" y1="246" x2="109" y2="246" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <line x1="109" y1="246" x2="109" y2="190" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="109,184 104,194 114,194" style="fill:var(--teal)"/>
    <text x="380" y="238" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--teal)">response ← ZMQ receives → unpickle ← BatchStrOutput</text>
    <rect x="190" y="276" width="380" height="32" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="297" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">plain Python dataclasses · serialized over IPC · not HTTP</text>
  </svg>
  <div class="figcap"><b>Figure 2 · One round-trip</b> — the request is pickle-serialized and travels right through the three processes over <span class="mono">zmq</span> sockets, then the response returns the same way, forming a loop. Note: what crosses between processes are <strong>plain Python dataclasses</strong> (serialized over IPC), <strong>not</strong> HTTP between processes — HTTP exists only at the outermost edge facing the client.</div>
</div>

<p>A concrete picture: the main process hands the <span class="mono">TokenizedGenerateReqInput</span> object to ZMQ, which uses <span class="mono">pickle</span> (and <span class="mono">msgpack</span> on some channels) to flatten it into bytes, sends it over an IPC socket to the scheduler subprocess, and rebuilds the same-shaped object on the other side — <strong>both ends hold the same dataclass, not an HTTP body</strong>.</p>

<p>Leg by leg. <strong>Leg one</strong>: the TokenizerManager (Lesson 14) receives a <span class="mono">GenerateReqInput</span> — the <strong>user-facing</strong> request:
<strong>raw text</strong>, sampling params, a <span class="mono">stream</span> flag, maybe images/audio. Note: <strong>this object does not cross processes</strong>; it lives only in the main
process. <strong>Leg two</strong>: tokenization turns text into token ids and packs a <span class="mono">TokenizedGenerateReqInput</span> — <strong>this</strong> is the message actually
serialized by ZMQ and <strong>sent across the boundary to the scheduler</strong>. In other words, <strong>what crosses ZMQ to the scheduler is tokenized ids, not raw text</strong>.
That is the one line to memorize this lesson.</p>

<p><strong>Leg three</strong>: the scheduler (Lesson 18) groups requests into a <span class="mono">ScheduleBatch</span> (Lesson 19), runs the forward pass, and emits a
<span class="mono">BatchTokenIDOutput</span> — note it is <strong>batched</strong>: one message holds the new token ids for <strong>many requests at once</strong>, with a
<span class="mono">rids</span> list marking which segment belongs to whom. <strong>Leg four</strong>: the detokenizer (Lesson 17) decodes ids back to text as a
<span class="mono">BatchStrOutput</span> (with <span class="mono">output_strs</span>). <strong>Final leg</strong>: results flow back to the TokenizerManager, which uses the
<span class="mono">rid</span> to route each piece to the right <span class="mono">await</span>ing coroutine and hands it to the client.</p>

<table class="t">
  <tr><th>Message struct</th><th>Key fields</th><th>Producer → Consumer</th></tr>
  <tr><td class="mono">GenerateReqInput</td><td>text / input_ids, sampling_params, stream, rid</td><td>client/compat → TokenizerManager (<strong>no IPC</strong>)</td></tr>
  <tr><td class="mono">TokenizedGenerateReqInput</td><td>rid, input_ids, sampling_params, stream</td><td>TokenizerManager →(<strong>ZMQ</strong>)→ Scheduler</td></tr>
  <tr><td class="mono">BatchTokenIDOutput</td><td>rids, decode_ids/output_ids, finished_reasons</td><td>Scheduler/worker →(ZMQ)→ Detokenizer</td></tr>
  <tr><td class="mono">BatchStrOutput</td><td>rids, output_strs, output_ids, finish info</td><td>Detokenizer →(ZMQ)→ TokenizerManager → client</td></tr>
</table>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/io_struct.py ::BatchTokenIDOutput</span><span class="ln">Scheduler -&gt; Detokenizer: a step's raw token outputs</span></div>
  <pre><span class="kw">@dataclasses.dataclass</span>
<span class="kw">class</span> BatchTokenIDOutput(BaseBatchReq):
    <span class="cm"># Scheduler -&gt; Detokenizer: one step's raw token outputs to be turned into text</span>
    finished_reasons: list      <span class="cm"># None / stop / length, per request</span>
    decoded_texts: list         <span class="cm"># text decoded so far (for incremental decode)</span>
    decode_ids: list            <span class="cm"># the token ids to detokenize</span>
    read_offsets: list          <span class="cm"># where to resume detokenizing each request</span>
    output_ids: list            <span class="cm"># the newly produced token ids</span>
    skip_special_tokens: list</pre>
</div>

<p>Feel the "batched" nature of this edge concretely: in one step the scheduler runs <span class="mono">rid=7</span>, <span class="mono">rid=3</span>, and <span class="mono">rid=9</span> together, so it packs all three requests' new token ids into a <strong>single</strong> <span class="mono">BatchTokenIDOutput</span>, with the <span class="mono">rids</span> list from <span class="mono">BaseBatchReq</span> marking which segment is whose; that one message is then sent over ZMQ (serialized with <span class="mono">pickle</span>/<span class="mono">msgpack</span>) to the detokenizer.</p>

<p>Read the table top-to-bottom and a neat pattern appears: <strong>each row's consumer is the next row's producer</strong>; the four structs link end-to-end into the
<strong>full loop</strong> from "human text" to "tokens" and back to "human text." Note the switch from <strong>singular to plural</strong>: inbound is <strong>single</strong>
(<span class="mono">GenerateReqInput</span>/<span class="mono">TokenizedGenerateReqInput</span>, carrying <span class="mono">rid</span>), outbound is <strong>batched</strong>
(<span class="mono">Batch*Output</span>, carrying <span class="mono">rids</span>) — because the scheduler <strong>batches many requests</strong> together, so outputs are naturally batched.
That is continuous batching (Lesson 18) showing through.</p>

<h2>rid: the case number threaded throughout</h2>
<p>What keeps these messages <strong>from getting scrambled</strong> across three processes is one humble field: <span class="mono">rid</span> (request id). It is created at the
<span class="mono">GenerateReqInput</span>, copied into <span class="mono">TokenizedGenerateReqInput</span> all the way to the scheduler, and flows back in the
<span class="mono">rids</span> lists of <span class="mono">BatchTokenIDOutput</span>/<span class="mono">BatchStrOutput</span>. Because the scheduler <strong>batches many requests</strong> together, the
returning output is <strong>interleaved and batched</strong> — one message holds tokens for request 7, request 3, request 9. Without <span class="mono">rid</span>, the
TokenizerManager could not <strong>split and route</strong> that interleaved batch back to the right awaiting coroutines.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>How rid routes interleaved batched outputs</b></div>
  <div class="cells"><span class="lab">out</span><span class="cell">rid=7: tok</span><span class="cell">rid=3: tok</span><span class="cell hl">rid=9: tok</span><span class="sep">→</span><span class="cell">one BatchTokenIDOutput</span><span class="q">batched, interleaved</span></div>
  <div class="cells"><span class="lab">split</span><span class="cell hl">by rid</span><span class="sep">→</span><span class="cell">7→coro A</span><span class="cell">3→coro B</span><span class="cell">9→coro C</span><span class="q">each to its own</span></div>
</div>

<p>So <span class="mono">rid</span> is not optional metadata — it is the root that lets this "everything is a message" system run <strong>concurrent yet unscrambled</strong>. It is
like the <strong>tracking number</strong> on a parcel: packages get broken apart, co-mingled, and redistributed at the sorting center, yet are tracked precisely and delivered to
the right person — all by that number. Understand the round-trip of <span class="mono">rid</span> and you understand the <strong>foundation</strong> shared by SGLang's async routing
(Lesson 14) and streaming (Lesson 17). Here is the real skeleton of the core message that actually crosses ZMQ:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/io_struct.py ::TokenizedGenerateReqInput</span><span class="ln">the message that actually crosses ZMQ to the scheduler</span></div>
  <pre><span class="kw">@dataclass</span>
<span class="kw">class</span> TokenizedGenerateReqInput(BaseReq):   <span class="cm"># BaseReq supplies rid (the cross-process id)</span>
    input_text: str                       <span class="cm"># raw text (kept for debug/logging)</span>
    input_ids: Optional[array[int]]        <span class="cm"># tokenized ids — what actually feeds the model</span>
    mm_inputs: object                      <span class="cm"># multimodal inputs (image/audio)</span>
    sampling_params: SamplingParams        <span class="cm"># temperature/top_p/max_new_tokens, ...</span>
    return_logprob: bool
    stream: bool                           <span class="cm"># whether to stream back (Lesson 17)</span>
    <span class="cm"># ...then serialized by ZMQ and sent to the scheduler (Lesson 18)</span></pre>
</div>

<p>Notice this struct has <strong>no</strong> user-level <span class="mono">text</span>-list shape; instead it carries the already-tokenized <span class="mono">input_ids</span> — confirming the key
line: <strong>past the ZMQ wire, downstream only knows tokens, never human text</strong>. <span class="mono">input_text</span> is retained, but only for debugging and logs; what drives the
model is <span class="mono">input_ids</span>.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>No shared memory, only messages</strong>: the three processes (Lesson 2) own separate memory and cooperate via <span class="mono">@dataclass</span> messages in <span class="mono">io_struct.py</span> + <strong>ZMQ</strong> serialization.</li>
    <li><strong>Lifecycle = a relay of messages</strong>: <span class="mono">GenerateReqInput</span> → <span class="mono">TokenizedGenerateReqInput</span> → <span class="mono">BatchTokenIDOutput</span> → <span class="mono">BatchStrOutput</span>, swapping struct at each process.</li>
    <li><strong>What crosses ZMQ is tokens, not text</strong>: the scheduler (Lesson 18) actually receives the tokenized <span class="mono">TokenizedGenerateReqInput</span> (with <span class="mono">input_ids</span>); <span class="mono">GenerateReqInput</span> lives only in the main process.</li>
    <li><strong>rid threads throughout</strong>: the request id routes interleaved <strong>batched</strong> outputs (the <span class="mono">rids</span> list) back to the right awaiting coroutine (async routing Lesson 14, streaming Lesson 17).</li>
    <li><strong>It is the runtime's "type system"</strong>: scheduler, worker, and detokenizer all speak these structs; abort/flush_cache/health are standalone control messages.</li>
  </ul>
</div>
""",
}

LESSON_17 = {
    "zh": r"""
<p class="lead">
第 14 课讲了运行时的<strong>入口</strong>——TokenizerManager 把人类文本变成 token id 送进 GPU。
这一课讲<strong>出口</strong>：GPU 一个个吐出来的 token id，怎样变回人类能读的文字，再像打字机一样
<strong>一段一段</strong>流回客户端。负责这件事的，是一个独立的子进程——<strong>DetokenizerManager（反分词器）</strong>。
它是整条链路上<strong>最后一个会说人话的组件</strong>，也是 Part 4 的收尾。读懂它，你就完整走完了一条请求<strong>从文本进、到文本出</strong>的全程。
</p>

<p>为什么单独开一进程、单独讲一课？因为把 token id 拼回文本，远不是“查个词表”那么简单。
它要解决两个真正棘手的问题：一是<strong>增量</strong>——每一步只该把<strong>新增的那一小段</strong>发出去，
而不是把整段已生成的文本重新解码、重新发一遍；二是<strong>多字节边界</strong>——一个 token 可能只解码出半个汉字、
半个 emoji，必须<strong>攒够一个完整字符</strong>再吐，绝不能把半个字符喂给屏幕。把这两件事彻底想透，你就懂了
为什么流式输出看起来那么自然顺滑，背后却需要一套精巧而严密的<strong>偏移量记账</strong>。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把反分词器想成一位<strong>同声传译</strong>。讲者（GPU）滔滔不绝地往外蹦词，传译每隔几秒开口一次——
  但他<strong>只说“上次开口之后新冒出来的那几个词”</strong>，绝不会把整段演讲从头再复述一遍（那是 <span class="mono">sent_offset</span> 在记“说到哪了”）。
  更妙的是：当一个音节只说了一半时，他会<strong>先憋住</strong>，等这个词完整了再出声——绝不蹦出半个音节
  （这正是“不完整 token 要先攒着”的处理）。于是听众耳朵里听到的，永远是<strong>连贯、完整、只增不重</strong>的句子。
  反分词器干的，正是这位传译的活：<strong>只发新增、攒满才发、按号交付</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  回忆第 2 课的<strong>三进程模型</strong>：分词在主进程、调度+前向在调度器子进程、<strong>反分词在反分词器子进程</strong>。
  调度器每跑完一步，就把这一批请求<strong>新生成的 token id</strong> 打包成 <span class="mono">BatchTokenIDOutput</span>（第 16 课）
  发给反分词器；反分词器把 id 解码成文本，再打包成 <span class="mono">BatchStrOutput</span> 发回 <strong>TokenizerManager</strong>（第 14 课），
  由它按 <span class="mono">rid</span> 投回正在 <span class="mono">await</span> 的协程，最终以 <strong>SSE</strong> 流给 HTTP 客户端。
  之所以让反分词单独占一个进程，是因为它是<strong>纯 CPU 的字符串活</strong>——可以和 GPU 的前向循环<strong>重叠</strong>起来，
  互不等待（零开销思想，第 21 课）。
</div>

<h2>它在哪：运行时的最后一道门</h2>
<p>反分词器是这条流水线的<strong>出口闸门</strong>。它<strong>只认 token id，绝不碰 GPU</strong>：输入是调度器送来的一批批
新 token，输出是一段段可读文本。它和分词器恰好镜像——一个把文本拆成数字（进），一个把数字拼回文本（出）。
注意它把结果发回的是 <strong>TokenizerManager</strong> 而不是客户端：因为只有 TokenizerManager 那张
<span class="mono">rid_to_state</span> 表知道“这个 rid 对应哪个正在等待的协程、该怎么 <span class="mono">yield</span> 给上层的 SSE”。</p>

<p>这里还有一个容易被忽略的对称之美：分词器与反分词器是同一枚硬币的两面，却<strong>各自独立、互不知情</strong>。
分词器在主进程把人类文本压成数字，反分词器在另一个子进程把数字还原成人类文本，两者之间隔着整条 GPU 流水线，
靠的全是 <span class="mono">rid</span> 这枚“身份证”把进与出对上号。也正因为反分词器只在出口侧工作、不参与采样与前向，
它可以做得非常<strong>轻、纯、可独立重启</strong>——万一它崩了，运行时甚至能在不影响 GPU 前向的前提下把它拉起来重连。
把“面向人类文本的脏活”收束在进出两端的两个 CPU 进程里，正是这套架构清爽好维护的关键。</p>

<p>把这条出口链路顺一遍：<strong>调度器</strong>（子进程）前向算完、吐出本步新生成的 token id；
<strong>DetokenizerManager</strong>（子进程）把 id 解码成文本、做增量切片；切好的片段回到主进程的
<strong>TokenizerManager</strong>，按 <span class="mono">rid</span> 投回正在等待的协程；最终以 <strong>SSE</strong>
逐段流给 <strong>HTTP 客户端</strong>，呈现“打字机”式的逐字浮现。四个环节、三道进程边界，全靠 <span class="mono">rid</span> 串成一条闭环。</p>

<h2>增量反分词：sent_offset 的妙处</h2>
<p>这是本课<strong>最核心</strong>、也最容易被忽略的一环。最朴素的做法是：每一步都把<strong>到目前为止的全部 token</strong>
解码成完整文本，整段发给客户端。这看似简单，却有两个致命问题：第一，每步都重发整段，传输量随长度<strong>平方级</strong>膨胀
（O(n²)）；第二，客户端会反复收到<strong>重复</strong>的前缀，得自己去算“哪部分是新的”。</p>

<p>正确做法是<strong>增量</strong>：每条请求维护一个偏移量 <span class="mono">sent_offset</span>，记录“已经发出去到哪个字符了”。
每一步解码出当前的完整文本 <span class="mono">output_str</span> 后，<strong>只截取新增的那一小段</strong>
<span class="mono">output_str[sent_offset:]</span> 发出去，然后把 <span class="mono">sent_offset</span> 推进到
<span class="mono">len(output_str)</span>。下一步再来，又只发新的尾巴。这样每个字符<strong>只过线一次</strong>，传输量是线性的。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>收到新 token id</h4><p>调度器送来本步新生成的 token id（随 <span class="mono">BatchTokenIDOutput</span>，第 16 课），追加到该请求已有的 id 序列。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>解码出完整文本</h4><p>把当前 id 序列解码，得到到目前为止的完整文本 <span class="mono">output_str</span>（含多字节边界处理）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>只切新增片段</h4><p>取 <span class="mono">incremental = output_str[sent_offset:]</span>——只要上次发送之后<strong>新冒出来</strong>的那一小段。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>推进偏移量</h4><p><span class="mono">sent_offset = len(output_str)</span>，记下“已发到这”，保证下一步不会重复发送已发过的字符。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>打包成 SSE 片段</h4><p>新增片段随 <span class="mono">BatchStrOutput</span> 回到 TokenizerManager，再以一个 SSE chunk 流给客户端——这就是“打字机”效果。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="增量反分词：token id 逐个到达，只把新增的那一小段解码成文本增量，用 sent_offset 记住上次解码到哪、从那里续上">
    <text x="24" y="34" style="font-weight:700;fill:var(--accent-ink)">增量反分词：只解码“新增的那一小段”</text>
    <rect x="24" y="52" width="330" height="214" rx="12" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="44" y="80" style="font-weight:700;fill:var(--muted)">第 1 步 · 到达 token id</text>
    <rect x="44" y="94" width="66" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="77" y="116" text-anchor="middle" class="mono" style="font-size:12px">12</text>
    <rect x="120" y="94" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="162" y="116" text-anchor="middle" class="mono" style="font-size:12px">8801</text>
    <text x="218" y="116" style="fill:var(--faint);font-size:13px">解码↓</text>
    <text x="44" y="158" style="fill:var(--muted);font-size:12px">文本增量（只发新增）</text>
    <rect x="44" y="168" width="160" height="36" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="124" y="191" text-anchor="middle" class="mono" style="font-size:13px">“SG”+“Lang”</text>
    <text x="44" y="238" class="mono" style="font-size:12px;fill:var(--accent-ink)">sent_offset: 0 → 6</text>
    <rect x="378" y="52" width="358" height="214" rx="12" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="398" y="80" style="font-weight:700;fill:var(--muted)">第 2 步 · 又到达一个 token</text>
    <rect x="398" y="94" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="434" y="116" text-anchor="middle" class="mono" style="font-size:12px">332</text>
    <text x="484" y="116" style="fill:var(--faint);font-size:13px">从 sent_offset=6 续上↓</text>
    <text x="398" y="158" style="fill:var(--muted);font-size:12px">文本增量（只发新增）</text>
    <rect x="398" y="168" width="96" height="36" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="446" y="191" text-anchor="middle" class="mono" style="font-size:14px">“很”</text>
    <text x="398" y="238" class="mono" style="font-size:12px;fill:var(--accent-ink)">sent_offset: 6 → 7</text>
    <rect x="24" y="280" width="712" height="32" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="301" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">绝不每步重解码整段：只把 output_str[sent_offset:] 这一小段发出去</text>
  </svg>
  <div class="figcap"><b>图 1 · 增量反分词</b> — token id 一个个到达：<span class="mono">[12, 8801]</span> 解码出 “SG”+“Lang”，下一个 <span class="mono">[332]</span> 解码出 “很”。靠 <span class="mono">sent_offset</span> 记住“上次解码到哪”，每步只解码<strong>新增的那一小段</strong>，而不是把整段已生成文本重新解码一遍。微妙之处：有的 token 要和后面的字节<strong>拼起来才构成一个完整字符</strong>，增量反分词会自动等齐再吐。</div>
</div>

<p>举一个落到字符上的具体例子：某一步 <span class="mono">sent_offset = 6</span>，本步把序列解码出 <span class="mono">output_str = "SGLang很"</span>（长度 7），于是只取 <span class="mono">output_str[6:] = "很"</span> 发出去，再把 <span class="mono">sent_offset</span> 推进到 7。对应到客户端，这一步就是一条 SSE 帧：<span class="mono">data: {"text": "很"}</span>。</p>

<p>把这套记账想象成一支<strong>不断右移的游标</strong>：游标左边是“已经交付、不再触碰”的历史，游标右边是“刚长出来、待交付”的新文本。
每一步，反分词器先把游标右边的新文本切下来发走，再把游标推到文本末尾。于是无论生成多长，<strong>已发送的部分永远不会被重新计算或重新传输</strong>，
这就是增量相对全量在工程上的根本胜利。源码里这枚游标就叫 <span class="mono">sent_offset</span>，它被存在每条请求的解码状态 <span class="mono">DecodeStatus</span> 里，
和 <span class="mono">rid</span> 一一对应——成百上千条请求并发流式时，<strong>每条都有自己独立的游标</strong>，谁也不会串到谁的进度。</p>

<div class="cols">
  <div class="col"><h4>❌ 每步全量重解码</h4><p>每步都解码全部 token 并发送整段文本。客户端反复收到重复前缀，传输量随长度<strong>平方膨胀（O(n²)）</strong>，长输出时尤其浪费，还把“算新增”的负担甩给客户端。</p></div>
  <div class="col"><h4>✅ 增量切片（sent_offset）</h4><p>每步只发 <span class="mono">output_str[sent_offset:]</span> 这一小段新文本，并推进偏移量。每个字符<strong>只过线一次</strong>，传输量线性，客户端直接拼接即可，天然适配 SSE 流。</p></div>
</div>

<h2>多字节边界：为什么要先攒着</h2>
<p>增量切片还有个隐藏难点：<strong>一个 token 不等于一个字符</strong>。一个汉字、一个 emoji 往往要好几个字节、
有时跨好几个 token 才能拼完整。如果某一步解码出的尾部是<strong>不完整的 UTF-8 序列</strong>（半个汉字、坏掉的 emoji），
直接发出去，客户端屏幕上就会闪出一个乱码方块 <span class="mono">�</span>。所以反分词器必须<strong>把还不构成完整字符的字节先扣住</strong>，
等下一步更多 token 到来、拼成完整字符后再吐。下面这张表展示 <span class="mono">sent_offset</span> 如何随步推进，
并在出现半个字符时<strong>按兵不动</strong>：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>sent_offset 随步推进：高亮的是这一步<strong>新发出</strong>的片段</b></div>
  <div class="cells"><span class="lab">第 1 步</span><span class="cell">output_str=<span class="mono">"你好"</span></span><span class="sep">→</span><span class="cell hl">发 "你好"</span><span class="q">sent_offset: 0 → 2</span></div>
  <div class="cells"><span class="lab">第 2 步</span><span class="cell">尾部是半个字符</span><span class="sep">→</span><span class="cell">扣住不发</span><span class="q">sent_offset 不动: 2</span></div>
  <div class="cells"><span class="lab">第 3 步</span><span class="cell">output_str=<span class="mono">"你好世界"</span></span><span class="sep">→</span><span class="cell hl">发 "世界"</span><span class="q">sent_offset: 2 → 4</span></div>
  <div class="cells"><span class="lab">第 4 步</span><span class="cell">output_str=<span class="mono">"你好世界！"</span></span><span class="sep">→</span><span class="cell hl">发 "！"</span><span class="q">sent_offset: 4 → 5</span></div>
</div>

<p>这也解释了为什么 <span class="mono">sent_offset</span> 在“扣住半个字符”的那一步<strong>不能推进</strong>：偏移量代表“已经安全交付的边界”，
而半个字符还没构成可显示的内容，自然不能算进“已交付”。等下一步 token 补齐了这个字符，反分词器会把从旧偏移量到新边界之间<strong>完整冒出来的那一段</strong>
一次性发走，再推进游标。换句话说，<strong>偏移量只会停在完整字符的边界上</strong>，永远不会卡在一个字符的中间——这正是“绝不流出半个汉字、半个 emoji”的底层保证。
理解了这一点，你就能体会到：流式输出顺滑的表象之下，是一套对<strong>字符边界</strong>极其小心的字节级记账。</p>
<p>流式输出总要有个尽头。一条请求的生成会在三种情况下停下，反分词器/调度器据此在<strong>停止边界</strong>处把输出裁齐
（采样与停止细节见第 28 课）：</p>

<table class="t">
  <tr><th>停止条件</th><th>含义</th><th>对输出的影响</th></tr>
  <tr><td class="mono">EOS token</td><td>模型生成了序列结束符（end-of-sequence）</td><td>自然收尾，本条请求结束流式</td></tr>
  <tr><td class="mono">stop string</td><td>命中用户指定的停止字符串（如 "\n\n"）</td><td>在匹配处<strong>裁掉</strong>停止串及其后内容</td></tr>
  <tr><td class="mono">max_new_tokens</td><td>达到本条请求允许的最大新生成 token 数</td><td>强制截断，发出最后一段后结束</td></tr>
</table>

<p>为什么要在收尾时专门“裁齐”？因为<strong>停止串本身不应该出现在最终输出里</strong>。设想你设了停止串 <span class="mono">"\n\n"</span>，
模型一旦生成出它，就说明“该停了”，但这两个换行只是<strong>触发信号</strong>，不是用户想要的内容；若原样发给客户端，就会在答案末尾拖出一截多余的空行。
所以反分词器在物化完整文本后，用 <span class="mono">trim_matched_stop</span> 把<strong>匹配到的停止串及其之后</strong>统统切掉，只保留干净的正文。
EOS 则相反——它是一个<strong>特殊 token</strong>，本就不会被解码成可见字符，所以无需额外裁剪，自然消失。这种“按停止原因区别处理”的细节，
正是让最终输出既<strong>恰好停在该停的地方</strong>、又<strong>不带任何控制噪声</strong>的关键。</p>

<p>还要厘清一个常见误解：<strong>停止判定本身不在反分词器做</strong>。是调度器在采样那一步（第 28 课）判断“这条请求该不该停、为什么停”，
把 <span class="mono">finished_reasons</span> 随 <span class="mono">BatchTokenIDOutput</span> 一起告诉反分词器；反分词器只负责<strong>据此把文本裁齐并收尾</strong>。
这又是一次清爽的职责分离：<strong>调度器决定“停不停”，反分词器决定“怎么把停的那一刻的文本交付干净”</strong>，两者各司其职，互不越界。
这又是一次清爽的职责分离：把“做决定”和“干脏活”拆到不同组件，每一块都能保持简单、可测、可独立替换，这种模式在 SGLang 里随处可见。</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="流式 SSE：每生成一个 token 就在对应时刻发出一条 data 帧给客户端，用户看到文字逐段浮现，而不是等整段答案算完再一次性返回">
    <text x="24" y="34" style="font-weight:700;fill:var(--accent-ink)">流式 SSE：每个 token 即时变成一条 data: 帧</text>
    <rect x="24" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="104" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"SG"}</text>
    <rect x="210" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="290" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"Lang"}</text>
    <rect x="396" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="476" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"很"}</text>
    <rect x="582" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="662" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"好"}</text>
    <line x1="104" y1="100" x2="104" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="290" y1="100" x2="290" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="476" y1="100" x2="476" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="662" y1="100" x2="662" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="24" y1="150" x2="742" y2="150" style="stroke:var(--muted);stroke-width:2"/>
    <text x="738" y="154" style="fill:var(--muted);font-size:12px">时间→</text>
    <text x="104" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₁</text>
    <text x="290" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₂</text>
    <text x="476" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₃</text>
    <text x="662" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₄</text>
    <text x="24" y="206" style="fill:var(--muted);font-size:12px">客户端逐步看到的文字：</text>
    <rect x="24" y="216" width="150" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="99" y="241" text-anchor="middle" class="mono" style="font-size:13px">SG</text>
    <rect x="210" y="216" width="150" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="285" y="241" text-anchor="middle" class="mono" style="font-size:13px">SGLang</text>
    <rect x="396" y="216" width="150" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="471" y="241" text-anchor="middle" class="mono" style="font-size:13px">SGLang很</text>
    <rect x="582" y="216" width="160" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="662" y="241" text-anchor="middle" class="mono" style="font-size:13px">SGLang很好</text>
    <text x="24" y="284" style="fill:var(--amber);font-weight:700;font-size:13px">文字逐段浮现，用户无需等整段答案算完 —— 这就是“打字机”体验</text>
  </svg>
  <div class="figcap"><b>图 2 · 流式 SSE 帧</b> — 时间轴上，每生成一个 token 就在对应时刻 <span class="mono">t_N</span> 发出一条 <span class="mono">data:</span> 帧推给客户端；用户因此看到文字<strong>一段段浮现</strong>，而不是干等整段答案算完再一次性返回。</div>
</div>

<p>当生成结束时，反分词器会<strong>物化</strong>这条请求的完整文本、调用 <span class="mono">trim_matched_stop</span> 把停止串裁掉，
再发出最后那一段尾巴（同样只发 <span class="mono">output_str[sent_offset:]</span>），然后清理掉这条请求的解码状态。
而把每一段新增片段送回客户端的是这样一条链路：DetokenizerManager 发回的文本片段先由 TokenizerManager <strong>逐条 yield</strong>，
再由 <strong>HTTP 服务路由</strong>（<span class="mono">http_server.py</span> 的 <span class="mono">generate_request</span>）用 <span class="mono">StreamingResponse</span>、
<span class="mono">media_type="text/event-stream"</span>，把每个 chunk 作为一条 <strong>Server-Sent Events</strong> 推给浏览器（离线 Engine 路径不走 SSE，直接产出文本）——
这就是你看到文字“一个个蹦出来”的打字机体验，也是大模型聊天界面那种逐字浮现的手感的真正来源。下面是反分词器增量逻辑的真实骨架：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/detokenizer_manager.py ::DetokenizerManager</span><span class="ln">增量切片：output_str[sent_offset:] + 推进 offset</span></div>
  <pre><span class="cm"># 每条请求的解码状态里记着一个 sent_offset</span>
<span class="kw">class</span> DecodeStatus:
    decoded_text: str
    sent_offset: int = <span class="st">0</span>   <span class="cm"># 已发送到的字符位置</span>

<span class="cm"># 生成结束：物化完整文本、裁掉停止串、只发尾巴</span>
output_str = self.trim_matched_stop(
    s.get_decoded_text() + new_text,
    recv_obj.finished_reasons[i],
    recv_obj.no_stop_trim[i],
)
incremental_output = output_str[s.sent_offset :]  <span class="cm"># 只切新增片段</span>
s.sent_offset = len(output_str)                   <span class="cm"># 推进偏移量</span>
output_strs.append(incremental_output)</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/detokenizer_manager.py ::DetokenizerManager</span><span class="ln">把 token id 增量解码成文本、流式回传</span></div>
  <pre><span class="kw">class</span> DetokenizerManager:
    <span class="cm"># 把 token id 变回文本，并把增量片段流式发给客户端</span>
    <span class="kw">def</span> event_loop(self):
        <span class="cm"># 从 Scheduler 收 BatchTokenIDOutput，增量反分词，</span>
        <span class="cm"># 再发出 BatchStrOutput</span>
        ...
    <span class="kw">def</span> handle_batch_token_id_out(self, recv_obj):
        ...   <span class="cm"># 只解码各请求 read_offset 之后的新文本</span>
    <span class="kw">def</span> trim_matched_stop(self, ...):
        ...   <span class="cm"># 在匹配到的停止串处把文本裁掉</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>出口 / 独立子进程</strong>：DetokenizerManager 是运行时的出口，跑在<strong>自己的子进程</strong>里（第 2 课三进程模型），只认 token id、绝不碰 GPU。</li>
    <li><strong>消息往返</strong>：调度器发来 <span class="mono">BatchTokenIDOutput</span>（新 token id），反分词器解码后回 <span class="mono">BatchStrOutput</span>（第 16 课）给 TokenizerManager（第 14 课），再按 <span class="mono">rid</span> 投回协程。</li>
    <li><strong>增量是关键</strong>：每条请求记 <span class="mono">sent_offset</span>，每步只发 <span class="mono">output_str[sent_offset:]</span> 再推进偏移量；避免全量重发的 O(n²) 与重复前缀。</li>
    <li><strong>多字节边界</strong>：半个汉字/坏 emoji 等不完整 UTF-8 要<strong>先扣住</strong>，攒成完整字符再吐，绝不流出半个字符。</li>
    <li><strong>停止与 SSE</strong>：EOS / 停止串 / max_new_tokens 三选一收尾（第 28 课）；TokenizerManager <strong>逐条 yield</strong> 片段，再由 <strong>HTTP 服务路由</strong>用 <span class="mono">StreamingResponse</span> 包成 SSE “打字机”（离线 Engine 不走 SSE）。独立成进程是为了和 GPU 重叠（零开销，第 21 课）。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Lesson 14 covered the runtime's <strong>entrance</strong> — the TokenizerManager turns human text into token ids and sends them to the GPU.
This lesson covers the <strong>exit</strong>: how the token ids the GPU emits one at a time become readable text again, then stream back to the
client <strong>piece by piece</strong>, like a typewriter. The component responsible is its own subprocess — the
<strong>DetokenizerManager</strong>. It is the <strong>last component in the chain that speaks human language</strong>, and it closes Part 4.
</p>

<p>Why give it its own process and its own lesson? Because stitching token ids back into text is far more than a table lookup. It solves two
genuinely tricky problems: first, <strong>incrementality</strong> — each step must emit only the <strong>small new slice</strong>, not re-decode
and re-send the whole generated text every time; second, <strong>multi-byte boundaries</strong> — a token may decode to only half a Chinese
character or half an emoji, so it must <strong>wait for a complete character</strong> before emitting, never feeding a half-character to the
screen. Grasp these two and you'll see why streaming looks so smooth while quietly relying on careful <strong>offset bookkeeping</strong>.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of the detokenizer as a <strong>simultaneous interpreter</strong>. The speaker (GPU) keeps spilling out words; every few seconds the
  interpreter opens their mouth — but speaks <strong>only the words that appeared since they last spoke</strong>, never replaying the whole
  speech (that's <span class="mono">sent_offset</span> remembering "how far I've spoken"). Better still: when a syllable is only half out, the
  interpreter <strong>holds back</strong> and waits for the full word before voicing it — never a half-syllable (that's the "hold incomplete
  tokens" handling). So the audience always hears <strong>coherent, complete, never-repeated</strong> sentences. That's exactly the
  detokenizer's job: <strong>emit only the new, wait until whole, deliver by id</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Recall Lesson 2's <strong>three-process model</strong>: tokenize in the main process, schedule+forward in the scheduler subprocess,
  <strong>detokenize in the detokenizer subprocess</strong>. After each step, the scheduler packs the batch's <strong>newly generated token
  ids</strong> into a <span class="mono">BatchTokenIDOutput</span> (Lesson 16) and sends it to the detokenizer; the detokenizer decodes ids into
  text, packs a <span class="mono">BatchStrOutput</span>, and sends it back to the <strong>TokenizerManager</strong> (Lesson 14), which routes it
  by <span class="mono">rid</span> to the <span class="mono">await</span>ing coroutine, finally streaming it to the HTTP client via
  <strong>SSE</strong>. Detokenize gets its own process because it is <strong>pure CPU string work</strong> that can <strong>overlap</strong> with
  the GPU forward loop without either waiting on the other (the zero-overhead idea, Lesson 21).
</div>

<h2>Where it lives: the runtime's last door</h2>
<p>The detokenizer is the pipeline's <strong>exit gate</strong>. It <strong>only speaks token ids and never touches the GPU</strong>: input is
the batches of new tokens from the scheduler, output is slices of readable text. It mirrors the tokenizer exactly — one splits text into numbers
(in), the other stitches numbers back into text (out). Note it sends results back to the <strong>TokenizerManager</strong>, not the client:
only the TokenizerManager's <span class="mono">rid_to_state</span> map knows "which awaiting coroutine this rid belongs to and how to
<span class="mono">yield</span> it to the upstream SSE".</p>

<p>Trace the exit path: the <strong>scheduler</strong> (subproc) finishes its forward and emits this step's new token ids; the
<strong>DetokenizerManager</strong> (subproc) decodes ids into text and does the incremental slice; the slice returns to the main process's
<strong>TokenizerManager</strong>, routed by <span class="mono">rid</span> to the awaiting coroutine; and finally it streams piece by piece via
<strong>SSE</strong> to the <strong>HTTP client</strong>, surfacing as a "typewriter". Four stages, three process boundaries, all stitched into a
loop by the <span class="mono">rid</span>.</p>

<h2>Incremental detokenize: the magic of sent_offset</h2>
<p>This is the lesson's <strong>most central</strong> — and most overlooked — piece. The naive approach: every step, decode <strong>all tokens
so far</strong> into the full text and send the whole thing. It looks simple but has two fatal flaws: first, resending the whole string each step
makes traffic grow <strong>quadratically</strong> (O(n²)) with length; second, the client repeatedly receives <strong>duplicate</strong> prefixes
and must itself figure out "which part is new".</p>

<p>The right approach is <strong>incremental</strong>: each request keeps an offset <span class="mono">sent_offset</span> recording "which
character I've sent up to". After decoding the current full text <span class="mono">output_str</span> each step, emit <strong>only the new
slice</strong> <span class="mono">output_str[sent_offset:]</span>, then advance <span class="mono">sent_offset</span> to
<span class="mono">len(output_str)</span>. Next step, again emit only the new tail. Every character <strong>crosses the wire exactly once</strong>;
traffic is linear.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>receive new token ids</h4><p>The scheduler delivers this step's newly generated token ids (in a <span class="mono">BatchTokenIDOutput</span>, Lesson 16), appended to the request's existing id sequence.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>decode the full text</h4><p>Decode the current id sequence into the full-text-so-far <span class="mono">output_str</span> (with multi-byte boundary handling).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>slice only the new part</h4><p>Take <span class="mono">incremental = output_str[sent_offset:]</span> — only the small slice that <strong>appeared since</strong> the last send.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>advance the offset</h4><p><span class="mono">sent_offset = len(output_str)</span>, recording "sent up to here" so the next step never re-sends already-sent characters.</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>pack as an SSE chunk</h4><p>The new slice rides a <span class="mono">BatchStrOutput</span> back to the TokenizerManager, then streams to the client as one SSE chunk — the "typewriter" effect.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="Incremental detokenize: token ids arrive one by one; only the small new slice is decoded into a text delta, using sent_offset to remember where decoding left off and resume from there">
    <text x="24" y="34" style="font-weight:700;fill:var(--accent-ink)">Incremental detokenize: decode only the new piece</text>
    <rect x="24" y="52" width="330" height="214" rx="12" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="44" y="80" style="font-weight:700;fill:var(--muted)">Step 1 · token ids arrive</text>
    <rect x="44" y="94" width="66" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="77" y="116" text-anchor="middle" class="mono" style="font-size:12px">12</text>
    <rect x="120" y="94" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="162" y="116" text-anchor="middle" class="mono" style="font-size:12px">8801</text>
    <text x="216" y="116" style="fill:var(--faint);font-size:13px">decode↓</text>
    <text x="44" y="158" style="fill:var(--muted);font-size:12px">text delta (emit only the new)</text>
    <rect x="44" y="168" width="160" height="36" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="124" y="191" text-anchor="middle" class="mono" style="font-size:13px">"SG"+"Lang"</text>
    <text x="44" y="238" class="mono" style="font-size:12px;fill:var(--accent-ink)">sent_offset: 0 → 6</text>
    <rect x="378" y="52" width="358" height="214" rx="12" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="398" y="80" style="font-weight:700;fill:var(--muted)">Step 2 · one more token arrives</text>
    <rect x="398" y="94" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="434" y="116" text-anchor="middle" class="mono" style="font-size:12px">332</text>
    <text x="484" y="116" style="fill:var(--faint);font-size:13px">resume from sent_offset=6↓</text>
    <text x="398" y="158" style="fill:var(--muted);font-size:12px">text delta (emit only the new)</text>
    <rect x="398" y="168" width="96" height="36" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="446" y="191" text-anchor="middle" class="mono" style="font-size:14px">"很"</text>
    <text x="398" y="238" class="mono" style="font-size:12px;fill:var(--accent-ink)">sent_offset: 6 → 7</text>
    <rect x="24" y="280" width="712" height="32" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="301" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">never re-decode the whole sequence: emit just output_str[sent_offset:]</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Incremental detokenize</b> — token ids arrive one by one: <span class="mono">[12, 8801]</span> decode into "SG"+"Lang", and the next <span class="mono">[332]</span> decodes into "很". A <span class="mono">sent_offset</span> remembers "where decoding left off", so each step decodes <strong>only the small new piece</strong> instead of re-decoding the whole generated sequence. Subtlety: some tokens only complete a character when <strong>combined</strong> with later bytes — incremental detokenize waits until it is whole before emitting.</div>
</div>

<p>A concrete, character-level example: at some step <span class="mono">sent_offset = 6</span>, the step decodes <span class="mono">output_str = "SGLang很"</span> (length 7), so it takes only <span class="mono">output_str[6:] = "很"</span> to emit and advances <span class="mono">sent_offset</span> to 7. On the client side, that step is exactly one SSE frame: <span class="mono">data: {"text": "很"}</span>.</p>

<p>To make the difference between "resend everything" and "send only the new" obvious, compare them side by side:</p>

<div class="cols">
  <div class="col"><h4>❌ Re-decode all each step</h4><p>Decode all tokens and send the whole text every step. The client keeps receiving duplicate prefixes, traffic grows <strong>quadratically (O(n²))</strong> with length — especially wasteful for long outputs — and the burden of "finding the new part" is dumped on the client.</p></div>
  <div class="col"><h4>✅ Incremental slice (sent_offset)</h4><p>Each step sends only the small new slice <span class="mono">output_str[sent_offset:]</span> and advances the offset. Every character <strong>crosses the wire once</strong>, traffic is linear, the client just concatenates, and it maps naturally onto an SSE stream.</p></div>
</div>

<h2>Multi-byte boundaries: why hold back</h2>
<p>Incremental slicing hides another difficulty: <strong>one token is not one character</strong>. A Chinese character or an emoji often takes
several bytes — sometimes spanning several tokens — to complete. If a step decodes a tail that is an <strong>incomplete UTF-8 sequence</strong>
(half a character, a broken emoji) and sends it as-is, the client flashes a garbled box <span class="mono">�</span>. So the detokenizer must
<strong>hold back bytes that don't yet form a complete character</strong> and wait until more tokens arrive to complete it. The table below shows
how <span class="mono">sent_offset</span> advances per step and <strong>stays put</strong> when only half a character is available:</p>

<div class="cellgroup">
  <div class="cg-cap"><b>sent_offset advancing per step: the highlight is the slice <strong>newly emitted</strong> this step</b></div>
  <div class="cells"><span class="lab">Step 1</span><span class="cell">output_str=<span class="mono">"你好"</span></span><span class="sep">→</span><span class="cell hl">emit "你好"</span><span class="q">sent_offset: 0 → 2</span></div>
  <div class="cells"><span class="lab">Step 2</span><span class="cell">tail is half a char</span><span class="sep">→</span><span class="cell">hold, emit nothing</span><span class="q">sent_offset stays: 2</span></div>
  <div class="cells"><span class="lab">Step 3</span><span class="cell">output_str=<span class="mono">"你好世界"</span></span><span class="sep">→</span><span class="cell hl">emit "世界"</span><span class="q">sent_offset: 2 → 4</span></div>
  <div class="cells"><span class="lab">Step 4</span><span class="cell">output_str=<span class="mono">"你好世界！"</span></span><span class="sep">→</span><span class="cell hl">emit "！"</span><span class="q">sent_offset: 4 → 5</span></div>
</div>

<h2>Stop conditions and SSE: when generation ends</h2>
<p>Streaming always needs an end. A request's generation halts under one of three conditions, and the detokenizer/scheduler trims the output at
the <strong>stop boundary</strong> accordingly (sampling and stop details in Lesson 28):</p>

<table class="t">
  <tr><th>Stop condition</th><th>Meaning</th><th>Effect on output</th></tr>
  <tr><td class="mono">EOS token</td><td>the model generated the end-of-sequence token</td><td>natural finish, this request ends streaming</td></tr>
  <tr><td class="mono">stop string</td><td>a user-specified stop string is matched (e.g. "\n\n")</td><td><strong>trim off</strong> the stop string and everything after it</td></tr>
  <tr><td class="mono">max_new_tokens</td><td>reached this request's max allowed new tokens</td><td>force-truncate, emit the final slice and end</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Streaming SSE: each generated token becomes a data frame sent to the client at its moment in time, so the user sees text appear progressively instead of waiting for the whole answer">
    <text x="24" y="34" style="font-weight:700;fill:var(--accent-ink)">Streaming SSE: each token becomes a data: frame at once</text>
    <rect x="24" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="104" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"SG"}</text>
    <rect x="210" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="290" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"Lang"}</text>
    <rect x="396" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="476" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"很"}</text>
    <rect x="582" y="56" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="662" y="83" text-anchor="middle" class="mono" style="font-size:12px">data: {"text":"好"}</text>
    <line x1="104" y1="100" x2="104" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="290" y1="100" x2="290" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="476" y1="100" x2="476" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="662" y1="100" x2="662" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="24" y1="150" x2="742" y2="150" style="stroke:var(--muted);stroke-width:2"/>
    <text x="738" y="154" style="fill:var(--muted);font-size:12px">time→</text>
    <text x="104" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₁</text>
    <text x="290" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₂</text>
    <text x="476" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₃</text>
    <text x="662" y="170" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">t₄</text>
    <text x="24" y="206" style="fill:var(--muted);font-size:12px">text the client sees, step by step:</text>
    <rect x="24" y="216" width="150" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="99" y="241" text-anchor="middle" class="mono" style="font-size:13px">SG</text>
    <rect x="210" y="216" width="150" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="285" y="241" text-anchor="middle" class="mono" style="font-size:13px">SGLang</text>
    <rect x="396" y="216" width="150" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="471" y="241" text-anchor="middle" class="mono" style="font-size:13px">SGLang很</text>
    <rect x="582" y="216" width="160" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="662" y="241" text-anchor="middle" class="mono" style="font-size:13px">SGLang很好</text>
    <text x="24" y="284" style="fill:var(--amber);font-weight:700;font-size:13px">Text appears progressively; no waiting for the whole answer —— the “typewriter” feel</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Streaming SSE frames</b> — on a time axis, each generated token becomes one <span class="mono">data:</span> frame sent to the client at its moment <span class="mono">t_N</span>; so the user watches the text <strong>appear progressively</strong> instead of waiting for the whole answer to finish before getting it all at once.</div>
</div>

<p>When generation ends, the detokenizer <strong>materializes</strong> the request's full text, calls <span class="mono">trim_matched_stop</span>
to cut off the stop string, emits that last tail (again only <span class="mono">output_str[sent_offset:]</span>), and then deletes the request's
decode state. Each new slice flows back via the TokenizerManager (which <strong>yields</strong> chunks per request), and the <strong>HTTP server route</strong>
(<span class="mono">http_server.py</span>'s <span class="mono">generate_request</span>) wraps them with <span class="mono">StreamingResponse</span> /
<span class="mono">media_type="text/event-stream"</span>, pushing each chunk as one <strong>Server-Sent Event</strong> to the browser (the offline Engine path has no SSE) — that's the typewriter experience of text "popping out one bit at a time". Below is the real
skeleton of the detokenizer's incremental logic:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/detokenizer_manager.py ::DetokenizerManager</span><span class="ln">incremental slice: output_str[sent_offset:] + advance offset</span></div>
  <pre><span class="cm"># each request's decode state carries a sent_offset</span>
<span class="kw">class</span> DecodeStatus:
    decoded_text: str
    sent_offset: int = <span class="st">0</span>   <span class="cm"># character position already sent</span>

<span class="cm"># finished: materialize full text, trim the stop, emit the tail</span>
output_str = self.trim_matched_stop(
    s.get_decoded_text() + new_text,
    recv_obj.finished_reasons[i],
    recv_obj.no_stop_trim[i],
)
incremental_output = output_str[s.sent_offset :]  <span class="cm"># slice only the new part</span>
s.sent_offset = len(output_str)                   <span class="cm"># advance the offset</span>
output_strs.append(incremental_output)</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/detokenizer_manager.py ::DetokenizerManager</span><span class="ln">incrementally decode token ids into text and stream them back</span></div>
  <pre><span class="kw">class</span> DetokenizerManager:
    <span class="cm"># turns token ids back into text and streams deltas to the client</span>
    <span class="kw">def</span> event_loop(self):
        <span class="cm"># receive BatchTokenIDOutput from the Scheduler, detokenize</span>
        <span class="cm"># incrementally, emit BatchStrOutput</span>
        ...
    <span class="kw">def</span> handle_batch_token_id_out(self, recv_obj):
        ...   <span class="cm"># decode only the NEW text since each request's read_offset</span>
    <span class="kw">def</span> trim_matched_stop(self, ...):
        ...   <span class="cm"># cut the text at a matched stop sequence</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Exit / own subprocess</strong>: the DetokenizerManager is the runtime's exit, running in <strong>its own subprocess</strong> (Lesson 2's three-process model); it only speaks token ids and never touches the GPU.</li>
    <li><strong>Message round-trip</strong>: the scheduler sends <span class="mono">BatchTokenIDOutput</span> (new token ids); the detokenizer decodes and returns <span class="mono">BatchStrOutput</span> (Lesson 16) to the TokenizerManager (Lesson 14), routed back by <span class="mono">rid</span>.</li>
    <li><strong>Incremental is key</strong>: each request keeps a <span class="mono">sent_offset</span>; each step emits only <span class="mono">output_str[sent_offset:]</span> then advances the offset — avoiding the O(n²) and duplicate prefixes of full resends.</li>
    <li><strong>Multi-byte boundaries</strong>: incomplete UTF-8 (half a Chinese character / broken emoji) must be <strong>held back</strong> until a full character forms — never stream half a character.</li>
    <li><strong>Stop and SSE</strong>: generation ends on EOS / stop string / max_new_tokens (Lesson 28); the TokenizerManager <strong>yields</strong> slices and the <strong>HTTP server route</strong> wraps them into an SSE "typewriter" via <span class="mono">StreamingResponse</span> (the offline Engine path has no SSE). A separate process so it can overlap the GPU (zero-overhead, Lesson 21).</li>
  </ul>
</div>
""",
}
