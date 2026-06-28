"""Part 12 · Practice & contributing (L53-57).

Lesson content for the practice/contributing part of the SGLang visual guide: build & run,
benchmark & profiling, test suite & CI, code conventions & PR, and a whole-guide glossary.
L57 (glossary) is exempt from the MIN_CJK check (shell/check_html SOFT_EXEMPT) and has no quiz.
Each LESSON_XX is a {"zh": html, "en": html} dict consumed via registry.CONTENT.
"""

LESSON_53 = {"zh": r"""
<p class="lead">前面 52 课我们把 SGLang 的内部零件拆了个遍：分词、调度、注意力后端、KV 缓存、量化、并行、投机解码……现在终于到了把它们<strong>真正跑起来</strong>的时刻。本课只回答一个问题：怎么启动一个 SGLang 服务？答案出奇地简单——<span class="mono">python -m sglang.launch_server --model-path &lt;模型&gt;</span>，剩下的全是旋钮。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象你买了一台高端音响功放：背后有一排接口和旋钮——<strong>音量</strong>、<strong>低音</strong>、<strong>声道数</strong>、<strong>输入源选择</strong>。你不需要懂内部电路，只要知道每个旋钮控制什么，就能把声音调到最佳。<span class="mono">ServerArgs</span> 就是这台功放的<strong>面板</strong>：每一个字段都是一个旋钮，前面 52 课你学的每个内部组件，到这里都变成了面板上一个可以拧的 <span class="mono">--flag</span>。</p>
<p>启动服务 = 插电开机；调优部署 = 拧对旋钮。你之所以现在能看懂这一排旋钮，是因为你已经认识了它们背后接的每一个零件。这正是本课与众不同之处：它不引入任何新概念，只把你已经掌握的一切，收束成一条你能亲手敲下、亲眼看它跑起来的真实命令。</p>
<p>更妙的是，这块面板对“专家”和“新手”同样友好：什么都不调，全用默认值，功放照样能出声——SGLang 会为大多数字段推断合理默认值，让你一条最短命令就能跑起来；而当你想压榨性能时，每一个旋钮又都触手可及，且彼此正交。换句话说，<strong>简单的事保持简单，复杂的事才需要动手</strong>，这正是一块好面板该有的样子。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>一条命令 <span class="mono">python -m sglang.launch_server</span>（新版亦提供等价命令 <span class="mono">sglang serve</span>） 背后发生的事：命令行入口（<span class="mono">__main__</span>）先用 <span class="mono">prepare_server_args</span> 把命令行参数解析进一个庞大的 <span class="mono">ServerArgs</span> dataclass，再把这份配置交给 <span class="mono">run_server</span>，由它据此启动引擎（<strong>TokenizerManager + Scheduler + DetokenizerManager</strong>，第13课），再对外提供一个 HTTP 服务（<span class="mono">/generate</span>、OpenAI 兼容的 <span class="mono">/v1/...</span>，第15课）。</p>
<p>另有一个 <span class="mono">Engine</span> 类——<strong>进程内 / 离线</strong>模式，不开 HTTP，专为脚本和评测而生。两者共享同一套内核，只是“门面”不同。本课最大的洞见是：<strong><span class="mono">ServerArgs</span> 把整本书的知识收成了一排 CLI 旋钮</strong>。</p>
<p>回头看你走过的路：从第5课的批处理、第13课的三件套链路，到第22课的分块预填充、第30课的 KV 显存、第33课的注意力后端、第35课的量化、第43课的投机解码、第46课的并行、第47课的 EPLB——这些曾经各自为政的知识点，如今都汇聚到同一个入口。这正是把它放在课程接近尾声讲的原因：只有先理解了每个零件，你才能真正读懂这一排 flag 的含义，而不是把它们当成一串需要死记硬背的咒语。本课不教新机制，它教你<strong>如何把已学的一切真正落地成一条命令</strong>。</p>
</div>

<h2>一、一条命令背后的故事</h2>
<p>启动一个在线服务，你只需要：<span class="mono">python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct</span>。这条命令的入口（<span class="mono">__main__</span>）做的第一件事，是用 <span class="mono">prepare_server_args</span> 把你在命令行敲下的一堆 <span class="mono">--flag</span> 解析进 <span class="mono">ServerArgs</span> 这个数据类，再把解析好的 <span class="mono">ServerArgs</span> 交给 <span class="mono">run_server</span> 去启动引擎。<span class="mono">ServerArgs</span> 里有几十上百个字段，每一个都对应着前面某一课讲过的某个机制。</p>
<p>不妨把它想成一次“开机自检 + 组装流水线”的过程。<span class="mono">prepare_server_args</span> 不只是把字符串塞进字段，它还会做合法性校验、推断未显式指定的默认值、处理互相冲突的选项（比如某些后端不支持某种量化时会报错或回退）。校验通过后，框架才放心地按这份配置去申请显存、加载权重、构建 KV 缓存池。如果你启动时漏了必填的 <span class="mono">--model-path</span>，或者给了一个不存在的注意力后端名，错误就会在这一步被早早拦下，而不是等到第一个请求进来才崩溃。这种“配置先行、校验在前”的设计，让一次成功的启动几乎就等于一次成功的部署。</p>
<p>解析完成后，框架据此<strong>启动引擎</strong>：拉起 <span class="mono">TokenizerManager</span> 负责把文本转成 token、拉起 <span class="mono">Scheduler</span> 负责批处理与显存管理、拉起 <span class="mono">DetokenizerManager</span> 把生成的 token 流式还原成文本（这正是第13课讲的三件套）。最后，它绑定一个 HTTP 端口，对外暴露 <span class="mono">/generate</span> 原生接口和一整套 OpenAI 兼容的 <span class="mono">/v1/chat/completions</span>、<span class="mono">/v1/completions</span> 接口（第15课）。从此你就有了一个能接受请求、吐出 token 的推理服务器。</p>
<p>值得强调的是：这条命令看似简单，背后却把前面十几课讲过的进程编排、零拷贝通信、连续批处理一次性串了起来。你不必再手写任何胶水代码——框架替你把分词进程、调度进程、反分词进程都拉起来，并用进程间通信把它们连成一条流水线。换句话说，<span class="mono">launch_server</span> 是整本书所有内核机制的<strong>总开关</strong>，按下它，一台完整的推理服务就活了过来。如果你只想在脚本里离线生成、不需要对外服务，那就跳过 HTTP，改用后面要讲的 <span class="mono">Engine</span>。</p>
<p>这条流水线一旦建好，运行时就进入我们前面反复强调的循环：请求进来排队，调度器把它们拼成连续的批次，模型一步步前向、采样出 token，反分词器再把 token 流式吐回客户端。<span class="mono">launch_server</span> 的职责到“把这套机器搭起来并守住端口”为止；真正决定它跑得快不快、稳不稳的，是你启动时给的那组 flag。这也是为什么我们说，<strong>启动命令是“配置”，运行行为是“后果”</strong>——想改后果，就回到那一排旋钮上去改配置，而不是改代码。</p>

<div class="flow"><div class="node">CLI flags</div><div class="arrow">→</div><div class="node">ServerArgs</div><div class="arrow">→</div><div class="node">启动引擎<br>Tokenizer/Scheduler/Detokenizer</div><div class="arrow">→</div><div class="node">HTTP 端点<br>/generate · /v1/...</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="启动路径：命令行旗标解析进 ServerArgs，run_server 据此构建引擎（分词器、调度器、反分词器），再暴露 HTTP / OpenAI 兼容端点">
    <rect x="20" y="44" width="150" height="58" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="95" y="70" text-anchor="middle" style="fill:var(--blue);font-weight:700">CLI 旗标</text>
    <text x="95" y="90" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--muted)">--model-path …</text>
    <rect x="224" y="44" width="150" height="58" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="299" y="70" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">ServerArgs</text>
    <text x="299" y="90" text-anchor="middle" style="font-size:10px;fill:var(--muted)">解析 + 校验</text>
    <rect x="428" y="44" width="150" height="58" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="503" y="70" text-anchor="middle" class="mono" style="fill:var(--teal);font-weight:700;font-size:13px">run_server</text>
    <text x="503" y="90" text-anchor="middle" style="font-size:10px;fill:var(--muted)">构建引擎</text>
    <rect x="632" y="44" width="150" height="58" rx="10" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="707" y="70" text-anchor="middle" style="fill:var(--purple);font-weight:700">HTTP 端点</text>
    <text x="707" y="90" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--muted)">/generate · /v1</text>
    <line x1="170" y1="73" x2="218" y2="73" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="218,68 218,78 224,73" style="fill:var(--muted)"/>
    <line x1="374" y1="73" x2="422" y2="73" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="422,68 422,78 428,73" style="fill:var(--muted)"/>
    <line x1="578" y1="73" x2="626" y2="73" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="626,68 626,78 632,73" style="fill:var(--muted)"/>
    <line x1="503" y1="102" x2="503" y2="150" style="stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="498,150 508,150 503,156" style="fill:var(--teal)"/>
    <rect x="250" y="156" width="420" height="120" rx="12" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="460" y="180" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">引擎 · 进程内</text>
    <rect x="266" y="196" width="120" height="60" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="326" y="222" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">分词器</text>
    <text x="326" y="240" text-anchor="middle" style="fill:var(--muted);font-size:10px">Tokenizer</text>
    <rect x="400" y="196" width="120" height="60" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="460" y="222" text-anchor="middle" style="fill:var(--blue);font-weight:700;font-size:12px">调度器</text>
    <text x="460" y="240" text-anchor="middle" style="fill:var(--muted);font-size:10px">Scheduler</text>
    <rect x="534" y="196" width="120" height="60" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="594" y="222" text-anchor="middle" style="fill:var(--purple);font-weight:700;font-size:12px">反分词器</text>
    <text x="594" y="240" text-anchor="middle" style="fill:var(--muted);font-size:10px">Detokenizer</text>
  </svg>
  <div class="figcap"><b>图 1 · 启动路径</b> — 命令行 <span class="mono">--flag</span> 解析进 <span class="mono">ServerArgs</span>，<span class="mono">run_server</span> 据此构建引擎（分词器 + 调度器 + 反分词器），再暴露 HTTP / OpenAI 兼容端点（<span class="mono">/generate</span>、<span class="mono">/v1/...</span>）。</div>
</div>

<h2>二、关键洞见：每个字段就是一个旋钮</h2>
<p>这是本课真正想让你记住的事。<span class="mono">ServerArgs</span> 是一个 <span class="mono">@dataclasses.dataclass</span>，它的每个字段都会<strong>自动映射</strong>成一个 <span class="mono">--kebab-case</span> 风格的命令行参数：字段 <span class="mono">tp_size</span> → 参数 <span class="mono">--tp-size</span>，字段 <span class="mono">mem_fraction_static</span> → 参数 <span class="mono">--mem-fraction-static</span>。所以你看到的那“一面墙的 flag”，本质上就是这个 dataclass 字段的清单。</p>
<p>规则简单到可以一句话概括：<strong>下划线变连字符，前面加两道杠</strong>。字段类型还顺带决定了这个 flag 怎么用——布尔字段通常变成一个无需取值的开关（如 <span class="mono">--enable-eplb</span>），整数和浮点字段则要跟一个数值，字符串字段跟一个名字（如后端名、量化方式名）。正因为这套映射是机械且一致的，SGLang 的帮助文档（<span class="mono">--help</span>）几乎是自动生成的，你看到的每条 flag 说明，往往就来自字段旁边那行注释。理解了这一层，你查文档、读别人脚本、写自己的部署命令，都会快上一个数量级。</p>
<p>这意味着：<strong>调优一个部署，就是设置对的 flag</strong>，而你早就知道每个 flag 背后是什么。<span class="mono">--tp-size</span> / <span class="mono">--dp-size</span> 是张量并行 / 数据并行（第46课）；<span class="mono">--attention-backend</span> 在 flashinfer / triton / fa 之间切换注意力内核（第33课）；<span class="mono">--quantization</span> 选 fp8 / awq / gptq 量化（第35课）；<span class="mono">--mem-fraction-static</span> 决定 KV 池能拿多少 GPU 显存（第30/31课）；<span class="mono">--chunked-prefill-size</span> 把超长 prefill 切块（第22课）；<span class="mono">--max-running-requests</span> 给批大小设上限（第5课）；<span class="mono">--speculative-algorithm</span> 开投机解码（第43课）；<span class="mono">--enable-eplb</span> 打开专家并行负载均衡（第47课）。认识了这张映射表，“一堵 flag 墙”就变成了“我早就理解的那些组件”。</p>
<p>这种自动映射不是偶然，而是一种刻意的设计：把所有可调项集中在一个 dataclass 里，既方便统一文档化、统一校验，又让命令行、配置文件、Python API 三种入口共享同一份“真相来源”。所以当你在别人的启动脚本里看到一长串 <span class="mono">--flag</span> 时，不要被吓到——你完全可以反过来，从字段名推断它对应哪一课的机制。遇到不认识的 flag，去 <span class="mono">ServerArgs</span> 里搜对应的 snake_case 字段，往往就能在注释里找到它的含义和默认值。掌握了“字段 ↔ flag”的双向翻译，你就拿到了打开任何 SGLang 部署的钥匙。</p>
<p>举个例子体会这种“认得出”的感觉：当你看到一条很长的命令同时带着 <span class="mono">--tp-size 8 --quantization fp8 --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-eplb</span>，你不再需要逐个查文档，而是一眼就能翻译成一句话——“把模型按 8 路张量并行切开、用 fp8 量化省显存、给 KV 池留 85% 的显存、把超长 prefill 切成 4096 一块、并打开专家并行负载均衡”。这正是本课承诺的回报：当每个 flag 都对应你学过的一课，整条命令就从“一串咒语”变成了“一份你看得懂的部署说明书”。</p>

<table class="t"><tr><th>关键 flag</th><th>它控制什么（→ 对应课程）</th></tr>
<tr><td><span class="mono">--tp-size</span> / <span class="mono">--dp-size</span></td><td>张量并行 / 数据并行的规模（第46课）</td></tr>
<tr><td><span class="mono">--attention-backend</span></td><td>注意力内核：flashinfer / triton / fa（第33课）</td></tr>
<tr><td><span class="mono">--quantization</span></td><td>权重量化方式：fp8 / awq / gptq（第35课）</td></tr>
<tr><td><span class="mono">--mem-fraction-static</span></td><td>静态分配（权重 + KV 池）占多少 GPU 显存（第30/31课）</td></tr>
<tr><td><span class="mono">--chunked-prefill-size</span></td><td>把超长 prefill 切块以平滑显存（第22课）</td></tr>
<tr><td><span class="mono">--max-running-requests</span></td><td>同时在跑的请求数上限，即批大小帽（第5课）</td></tr>
<tr><td><span class="mono">--speculative-algorithm</span></td><td>投机解码算法的开关与选择（第43课）</td></tr>
<tr><td><span class="mono">--enable-eplb</span></td><td>专家并行负载均衡 EPLB（第47课）</td></tr>
</table>

<h2>三、两副面孔：Engine 离线 vs HTTP 在线</h2>
<p>同一套内核，SGLang 给了你两种用法。<strong>HTTP 服务器</strong>是默认的“在线”模式：你启动一个常驻进程，它监听端口，客户端通过网络发请求——这是生产部署、对外提供 API 的标准姿势。<strong>Engine</strong> 则是“离线 / 进程内”模式：你在自己的 Python 脚本里直接 <span class="mono">import</span> 并实例化它，不开 HTTP、不走网络，直接在内存里调用生成。它特别适合批量评测、数据生成、做实验——省去了起服务、发 HTTP 请求的所有开销。两者背后是同一个引擎，区别只在“有没有那层 HTTP 门面”。</p>
<p>怎么选？如果你要对外提供服务、支持多客户端并发、或者要接进现有的 OpenAI 生态工具链，就用 HTTP 服务器；如果你只是想在一个脚本里跑几千条 prompt 做离线评测、或在训练流程里嵌入一次性的批量生成，<span class="mono">Engine</span> 更轻、更快、也更好调试。关键是要意识到：两种模式的<strong>调优旋钮完全一样</strong>——无论走哪条路，你设置的都是同一个 <span class="mono">ServerArgs</span>（或它的等价构造参数），所以本课学到的“flag ↔ 组件”映射，对在线和离线两种场景同时生效。这也正是 SGLang 设计的优雅之处：内核只有一个，门面可以换。</p>

<div class="cols"><div class="col"><strong>Engine（离线 / 进程内）</strong><br>在 Python 脚本里直接实例化，无 HTTP、无网络。直接内存调用生成。适合脚本化、批量评测、离线数据生成。启动快、开销小。</div><div class="col"><strong>HTTP server（在线）</strong><br>常驻进程监听端口，客户端经网络发请求。暴露 <span class="mono">/generate</span> 与 OpenAI 兼容 <span class="mono">/v1/...</span>。适合生产部署、对外提供服务、多客户端并发。</div></div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="两种用法对比：Engine 离线进程内直接调用 generate 不走网络，HTTP 服务器在线监听端口接收 REST 请求，底层是同一内核">
    <rect x="30" y="40" width="320" height="150" rx="12" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="190" y="66" text-anchor="middle" style="fill:var(--teal);font-weight:700">Engine · 离线（进程内）</text>
    <rect x="50" y="84" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="64" y="106" class="mono" style="font-size:11px">import sglang as sgl</text>
    <rect x="50" y="126" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="64" y="148" class="mono" style="font-size:11px">engine.generate(prompts)</text>
    <text x="190" y="180" text-anchor="middle" style="fill:var(--muted);font-size:11px">无网络 · 批量 / 离线</text>
    <rect x="430" y="40" width="320" height="150" rx="12" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="590" y="66" text-anchor="middle" style="fill:var(--blue);font-weight:700">HTTP 服务器 · 在线</text>
    <rect x="450" y="84" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--blue);stroke-width:1.5"/>
    <text x="464" y="106" class="mono" style="font-size:11px">launch_server 进程</text>
    <rect x="450" y="126" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--blue);stroke-width:1.5"/>
    <text x="464" y="148" class="mono" style="font-size:11px">POST /generate（REST）</text>
    <text x="590" y="180" text-anchor="middle" style="fill:var(--muted);font-size:11px">多客户端 · 在线服务</text>
    <line x1="190" y1="190" x2="190" y2="232" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="185,232 195,232 190,238" style="fill:var(--muted)"/>
    <line x1="590" y1="190" x2="590" y2="232" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="585,232 595,232 590,238" style="fill:var(--muted)"/>
    <rect x="150" y="238" width="480" height="48" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="390" y="260" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">同一内核</text>
    <text x="390" y="278" text-anchor="middle" style="fill:var(--muted);font-size:10px">引擎 + ServerArgs</text>
  </svg>
  <div class="figcap"><b>图 2 · 离线 Engine vs 在线 HTTP</b> — 左：<span class="mono">import sglang</span>，进程内直接 <span class="mono">engine.generate(...)</span>，不走网络（批量 / 离线）；右：启动服务进程，客户端发 REST 请求（在线、多客户端）。底层是同一套内核。</div>
</div>

<p>把这两副面孔放在一起看，就更能体会“一套内核、两种门面”的价值：你在开发阶段可以用 <span class="mono">Engine</span> 在脚本里快速迭代、验证 prompt 与采样参数，等逻辑稳定了，再用<strong>完全相同的一组 flag</strong> 切换到 HTTP 服务器对外上线，几乎零迁移成本。这种一致性不是巧合，而是因为两者底层都把配置塞进同一个 <span class="mono">ServerArgs</span>。所以当有人问“离线脚本里那个参数，在线服务怎么设”，答案永远是同一个：找到对应的字段名，把下划线换成连字符，加上两道杠即可。</p>

<h2>四、快速上手四步走</h2>
<p>把它跑起来其实就四步。第一步<strong>安装</strong>：装好 SGLang 及其依赖。第二步<strong>启动并等待就绪</strong>：执行 <span class="mono">launch_server</span>，然后耐心等日志打印出模型加载完成、服务就绪的提示——大模型加载需要时间，别急着发请求。第三步<strong>探活</strong>：<span class="mono">curl http://localhost:30000/health</span>，确认服务活着。第四步<strong>发请求</strong>：向 <span class="mono">/generate</span> POST 一段 JSON，拿到生成结果。至于怎么系统地压测吞吐与延迟、找到最优的旋钮组合，我们留到第54课的基准测试再细讲。</p>
<p>这里有几个新手常踩的坑值得提醒。其一，<strong>不要在“就绪”日志出现之前就发请求</strong>：模型权重往往有几十 GB，从磁盘加载、搬上 GPU、再编译 CUDA Graph 都要时间，过早请求只会得到连接被拒。其二，<strong>显存不够时先调 <span class="mono">--mem-fraction-static</span></strong>：它直接决定 KV 池的大小，调小一点能给权重和激活让出空间，调大一点能容纳更多并发请求。其三，<strong>多卡部署先想清楚并行策略</strong>：<span class="mono">--tp-size</span> 把单个模型切到多张卡上（适合大模型放不下时），<span class="mono">--dp-size</span> 则是多份副本各自服务（适合提高吞吐）。把这四步走顺、再把这几个旋钮调对，你就拥有了一个稳定、可对外服务的 SGLang 实例；剩下的，就是按业务负载去精调那一排你早已认识的 flag。</p>
<p>最后给一条心法：<strong>先用默认值把服务跑起来，再逐个旋钮地改</strong>。一上来就堆一长串自己也不确定的 flag，出了问题很难定位是哪一个的锅。正确的做法是先 <span class="mono">--model-path</span> 一条命令起服务、确认健康，再根据观察到的瓶颈针对性地加 flag：显存吃紧就调 <span class="mono">--mem-fraction-static</span> 或开 <span class="mono">--quantization</span>，吞吐不够就加并行或放大 <span class="mono">--max-running-requests</span>，长上下文卡顿就调 <span class="mono">--chunked-prefill-size</span>。每次只动一个变量，你才能清楚地知道每个旋钮带来的真实收益——这也正是下一课基准测试要教你的事。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>安装 Install</h4><p class="mono">pip install / 源码安装</p><p>装好 SGLang 及其依赖（CUDA、注意力后端等），确认 GPU 驱动就绪。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>启动 Launch</h4><p class="mono">python -m sglang.launch_server</p><p>执行启动命令，<strong>耐心等待</strong>日志打印模型加载完成、服务就绪——大模型加载需要时间。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>探活 Health</h4><p class="mono">GET /health</p><p><span class="mono">curl http://localhost:30000/health</span>，确认服务活着、能接收请求。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>发请求 Generate</h4><p class="mono">POST /generate</p><p>向 <span class="mono">/generate</span> POST 一段 JSON，拿到生成结果；系统压测见第54课。</p></div></div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/server_args.py ::ServerArgs</span><span class="ln">ServerArgs 把全书旋钮收成 CLI flag</span></div><pre>@dataclasses.dataclass
class ServerArgs:
    model_path: str                    # --model-path: 要服务的 HF 模型
    tp_size: int = 1                   # --tp-size: 张量并行 rank 数 (第46课)
    dp_size: int = 1                   # --dp-size: 数据并行副本数 (第46课)
    mem_fraction_static: float = None  # --mem-fraction-static: 静态分配(权重+KV池)的 GPU 显存比例 (第30课)
    attention_backend: str = None      # --attention-backend: flashinfer / triton / fa (第33课)
    chunked_prefill_size: int = None   # --chunked-prefill-size: 切分超长 prefill (第22课)
    quantization: str = None           # --quantization: fp8 / awq / gptq / ... (第35课)
    max_running_requests: int = None   # --max-running-requests: 批大小上限 (第5课)
    # ... 还有几十个字段；每个都自动映射成一个 --kebab-case CLI flag</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/launch_server.py ::run_server</span><span class="ln">按 ServerArgs 选择并启动对应形态的服务器</span></div><pre>def run_server(server_args):
    # 按解析好的 ServerArgs 启动对应形态的服务器
    if server_args.grpc_mode:
        serve_grpc(server_args)             # gRPC 入口
    else:
        # 默认路径：HTTP / OpenAI 兼容服务器
        from sglang.srt.entrypoints.http_server import launch_server
        launch_server(server_args)          # 构建引擎 + HTTP 路由</pre></div>

<p>两个具体例子帮你把链路对上号。<strong>在线：</strong><span class="mono">python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --tp-size 2</span> —— 这串 flag 被解析进 <span class="mono">ServerArgs</span>，交给 <span class="mono">run_server</span> 走默认分支构建引擎，最终在 <span class="mono">:30000</span> 上提供 HTTP 服务。<strong>离线：</strong><span class="mono">sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct").generate(prompts)</span> —— 同一份配置走 <span class="mono">Engine</span> 形态，在进程内直接生成，完全跳过网络。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li>启动只需一条命令：<span class="mono">python -m sglang.launch_server --model-path &lt;模型&gt; [flags]</span>。</li>
<li>命令行入口用 <span class="mono">prepare_server_args</span> 把 argv 解析进 <span class="mono">ServerArgs</span> → 交给 <span class="mono">run_server</span> 启动引擎（三件套，第13课）→ 暴露 HTTP（<span class="mono">/generate</span>、<span class="mono">/v1/...</span>，第15课）。</li>
<li><strong>核心洞见</strong>：<span class="mono">ServerArgs</span> 每个 dataclass 字段自动映射成一个 <span class="mono">--kebab-case</span> flag，前 52 课的每个组件都变成一个旋钮。</li>
<li>调优 = 拧对 flag：<span class="mono">--tp-size</span>/<span class="mono">--dp-size</span>、<span class="mono">--attention-backend</span>、<span class="mono">--quantization</span>、<span class="mono">--mem-fraction-static</span>、<span class="mono">--chunked-prefill-size</span>、<span class="mono">--max-running-requests</span> 等。</li>
<li><span class="mono">Engine</span>（离线/进程内，无 HTTP）vs HTTP 服务器（在线）：同一内核，两种门面。</li>
<li>上手四步：安装 → 启动并等就绪 → <span class="mono">/health</span> 探活 → POST <span class="mono">/generate</span>；压测见第54课。</li>
</ul></div>
""", "en": r"""
<p class="lead">Across the last 52 lessons we took SGLang apart piece by piece: tokenizing, scheduling, attention backends, the KV cache, quantization, parallelism, speculative decoding… Now it's finally time to <strong>actually run it</strong>. This lesson answers one question: how do you start an SGLang server? The answer is surprisingly simple — <span class="mono">python -m sglang.launch_server --model-path &lt;model&gt;</span> — and everything else is just knobs.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a high-end audio amplifier: a row of jacks and knobs on the back — <strong>volume</strong>, <strong>bass</strong>, <strong>channel count</strong>, <strong>input source</strong>. You don't need to understand the circuitry; if you know what each knob controls, you can dial the sound in perfectly. <span class="mono">ServerArgs</span> is that amplifier's <strong>front panel</strong>: every field is a knob, and every internal component you learned over 52 lessons becomes one twistable <span class="mono">--flag</span> on that panel.</p>
<p>Starting the server = plugging it in and powering on; tuning the deployment = twisting the right knobs. The reason you can read this wall of knobs today is that you already know every part wired behind it. That's what sets this lesson apart: it introduces no new concept, it just gathers everything you've mastered into one command you can type yourself and watch come alive.</p>
<p>Better still, this panel is friendly to "experts" and "beginners" alike: tweak nothing, take all the defaults, and the amplifier still makes sound — SGLang infers sensible defaults for most fields so one minimal command gets you running; yet when you want to squeeze out performance, every knob is within reach and orthogonal to the others. In other words, <strong>simple things stay simple and only the complex things need hands-on work</strong> — exactly what a good panel should be.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>What one command — <span class="mono">python -m sglang.launch_server</span> (newer builds also offer the equivalent <span class="mono">sglang serve</span>) — does under the hood: its entry point (<span class="mono">__main__</span>) first uses <span class="mono">prepare_server_args</span> to parse the command line into a big <span class="mono">ServerArgs</span> dataclass, then hands that config to <span class="mono">run_server</span>, which boots the engine (<strong>TokenizerManager + Scheduler + DetokenizerManager</strong>, Lesson 13) and serves an HTTP endpoint (<span class="mono">/generate</span>, OpenAI-compatible <span class="mono">/v1/...</span>, Lesson 15).</p>
<p>There's also an <span class="mono">Engine</span> class — an <strong>in-process / offline</strong> mode with no HTTP, built for scripting and eval. Both share the same core; only the "facade" differs. The key insight of this lesson: <strong><span class="mono">ServerArgs</span> gathers the whole guide's knowledge into one row of CLI knobs</strong>.</p>
<p>Look back at the path you've walked: from batching in Lesson 5 and the trio pipeline in Lesson 13, to chunked prefill in Lesson 22, KV memory in Lesson 30, attention backends in Lesson 33, quantization in Lesson 35, speculative decoding in Lesson 43, parallelism in Lesson 46, and EPLB in Lesson 47 — all these once-separate ideas now converge on one entry point. That's exactly why this lesson sits near the end of the course: only after understanding each part can you truly read this row of flags rather than treat them as an incantation to memorize. This lesson teaches no new mechanism; it teaches <strong>how to land everything you've learned into one command</strong>.</p>
</div>

<h2>1. The story behind one command</h2>
<p>To launch an online service you only need: <span class="mono">python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct</span>. The first thing this command's entry point (<span class="mono">__main__</span>) does is call <span class="mono">prepare_server_args</span> to parse the bunch of <span class="mono">--flag</span>s you typed into the <span class="mono">ServerArgs</span> dataclass, then hand the parsed <span class="mono">ServerArgs</span> to <span class="mono">run_server</span> to launch. <span class="mono">ServerArgs</span> holds dozens upon dozens of fields, and each one corresponds to a mechanism from some earlier lesson.</p>
<p>Think of it as a "power-on self-test plus assembly line". <span class="mono">prepare_server_args</span> doesn't just stuff strings into fields; it validates them, infers defaults you didn't specify, and resolves conflicting options (e.g. erroring or falling back when a backend doesn't support a given quantization). Only after validation passes does the framework confidently allocate memory, load weights, and build the KV pool according to that config. If you forget the required <span class="mono">--model-path</span>, or pass a non-existent attention backend name, the error is caught early here rather than crashing when the first request arrives. This "config first, validate up front" design means a successful launch is almost equivalent to a successful deployment.</p>
<p>Once parsed, the framework <strong>boots the engine</strong>: it brings up <span class="mono">TokenizerManager</span> to turn text into tokens, <span class="mono">Scheduler</span> to handle batching and memory management, and <span class="mono">DetokenizerManager</span> to stream generated tokens back into text (exactly the trio from Lesson 13). Finally it binds an HTTP port and exposes the native <span class="mono">/generate</span> endpoint plus a full set of OpenAI-compatible <span class="mono">/v1/chat/completions</span> and <span class="mono">/v1/completions</span> endpoints (Lesson 15). And now you have an inference server that accepts requests and emits tokens.</p>
<p>Worth stressing: the command looks simple, but behind it sits all the process orchestration, zero-copy communication, and continuous batching from the previous dozen lessons, wired up at once. You no longer write any glue code — the framework brings up the tokenizer, scheduler, and detokenizer processes for you and connects them into a pipeline via inter-process communication. In other words, <span class="mono">launch_server</span> is the <strong>master switch</strong> for every kernel mechanism in the whole guide; flip it and a complete inference service comes alive. If you only want offline generation in a script with no external serving, skip HTTP and use the <span class="mono">Engine</span> covered below.</p>
<p>Once this pipeline is built, the runtime enters the loop we've emphasized throughout: requests arrive and queue, the scheduler packs them into continuous batches, the model runs forward step by step and samples tokens, and the detokenizer streams tokens back to the client. <span class="mono">launch_server</span>'s job ends at "stand up this machine and hold the port"; what actually decides whether it runs fast and stable is the set of flags you gave at startup. That's why we say <strong>the launch command is the "config" and runtime behavior is the "consequence"</strong> — to change the consequence, go back to that row of knobs and change the config, not the code.</p>

<div class="flow"><div class="node">CLI flags</div><div class="arrow">→</div><div class="node">ServerArgs</div><div class="arrow">→</div><div class="node">boot engine<br>Tokenizer/Scheduler/Detokenizer</div><div class="arrow">→</div><div class="node">HTTP endpoint<br>/generate · /v1/...</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Boot path: command-line flags parse into ServerArgs, run_server builds the engine (tokenizer, scheduler, detokenizer) and exposes an HTTP / OpenAI-compatible endpoint">
    <rect x="20" y="44" width="150" height="58" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="95" y="70" text-anchor="middle" style="fill:var(--blue);font-weight:700">CLI flags</text>
    <text x="95" y="90" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--muted)">--model-path …</text>
    <rect x="224" y="44" width="150" height="58" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="299" y="70" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">ServerArgs</text>
    <text x="299" y="90" text-anchor="middle" style="font-size:10px;fill:var(--muted)">parse + validate</text>
    <rect x="428" y="44" width="150" height="58" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="503" y="70" text-anchor="middle" class="mono" style="fill:var(--teal);font-weight:700;font-size:13px">run_server</text>
    <text x="503" y="90" text-anchor="middle" style="font-size:10px;fill:var(--muted)">build engine</text>
    <rect x="632" y="44" width="150" height="58" rx="10" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="707" y="70" text-anchor="middle" style="fill:var(--purple);font-weight:700">HTTP endpoint</text>
    <text x="707" y="90" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--muted)">/generate · /v1</text>
    <line x1="170" y1="73" x2="218" y2="73" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="218,68 218,78 224,73" style="fill:var(--muted)"/>
    <line x1="374" y1="73" x2="422" y2="73" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="422,68 422,78 428,73" style="fill:var(--muted)"/>
    <line x1="578" y1="73" x2="626" y2="73" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="626,68 626,78 632,73" style="fill:var(--muted)"/>
    <line x1="503" y1="102" x2="503" y2="150" style="stroke:var(--teal);stroke-width:1.5"/>
    <polygon points="498,150 508,150 503,156" style="fill:var(--teal)"/>
    <rect x="250" y="156" width="420" height="120" rx="12" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="460" y="180" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">engine · in-process</text>
    <rect x="266" y="196" width="120" height="60" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="326" y="222" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:11px">Tokenizer</text>
    <text x="326" y="240" text-anchor="middle" style="fill:var(--muted);font-size:10px">Manager</text>
    <rect x="400" y="196" width="120" height="60" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="460" y="222" text-anchor="middle" style="fill:var(--blue);font-weight:700;font-size:11px">Scheduler</text>
    <text x="460" y="240" text-anchor="middle" style="fill:var(--muted);font-size:10px">batch + fwd</text>
    <rect x="534" y="196" width="120" height="60" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="594" y="222" text-anchor="middle" style="fill:var(--purple);font-weight:700;font-size:11px">Detokenizer</text>
    <text x="594" y="240" text-anchor="middle" style="fill:var(--muted);font-size:10px">Manager</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Boot path</b> — command-line <span class="mono">--flag</span>s parse into <span class="mono">ServerArgs</span>; <span class="mono">run_server</span> builds the engine (tokenizer + scheduler + detokenizer) and then exposes an HTTP / OpenAI-compatible endpoint (<span class="mono">/generate</span>, <span class="mono">/v1/...</span>).</div>
</div>

<h2>2. The key insight: every field is a knob</h2>
<p>This is what the lesson really wants you to remember. <span class="mono">ServerArgs</span> is a <span class="mono">@dataclasses.dataclass</span>, and every field <strong>auto-maps</strong> to a <span class="mono">--kebab-case</span> command-line flag: field <span class="mono">tp_size</span> → flag <span class="mono">--tp-size</span>, field <span class="mono">mem_fraction_static</span> → flag <span class="mono">--mem-fraction-static</span>. So that "wall of flags" you see is essentially the list of this dataclass's fields.</p>
<p>The rule is simple enough to state in one line: <strong>underscores become hyphens, prefixed with two dashes</strong>. The field type even dictates how the flag is used — a boolean field usually becomes a valueless switch (like <span class="mono">--enable-eplb</span>), int and float fields take a number, and string fields take a name (a backend, a quantization scheme). Because the mapping is mechanical and consistent, SGLang's <span class="mono">--help</span> is essentially auto-generated, and each flag's description often comes straight from the comment next to its field. Grasp this layer and reading docs, others' scripts, and writing your own launch command all get an order of magnitude faster.</p>
<p>That means: <strong>tuning a deployment is just setting the right flags</strong>, and you already know what each one does. <span class="mono">--tp-size</span> / <span class="mono">--dp-size</span> are tensor / data parallelism (Lesson 46); <span class="mono">--attention-backend</span> switches the attention kernel among flashinfer / triton / fa (Lesson 33); <span class="mono">--quantization</span> picks fp8 / awq / gptq (Lesson 35); <span class="mono">--mem-fraction-static</span> decides how much GPU memory the KV pool gets (Lessons 30/31); <span class="mono">--chunked-prefill-size</span> splits long prefills (Lesson 22); <span class="mono">--max-running-requests</span> caps the batch size (Lesson 5); <span class="mono">--speculative-algorithm</span> enables speculative decoding (Lesson 43); <span class="mono">--enable-eplb</span> turns on expert-parallel load balancing (Lesson 47). Knowing this map turns "a wall of flags" into "the components I already understand".</p>
<p>This auto-mapping is no accident but a deliberate design: centralizing every tunable in one dataclass makes it easy to document and validate uniformly, and lets three entry points — the command line, config files, and the Python API — share one source of truth. So when you see a long string of <span class="mono">--flag</span>s in someone else's launch script, don't be intimidated — you can work backwards, inferring from the field name which lesson's mechanism it maps to. Hit a flag you don't recognize? Search <span class="mono">ServerArgs</span> for the matching snake_case field and the comment usually tells you its meaning and default. Master the two-way translation of "field ↔ flag" and you hold the key to any SGLang deployment.</p>
<p>For a taste of that "I recognize this" feeling: when you see a long command carrying <span class="mono">--tp-size 8 --quantization fp8 --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-eplb</span>, you no longer look each one up — you read it at a glance as one sentence: "shard the model 8-way tensor-parallel, save memory with fp8 quantization, give the KV pool 85% of memory, chunk long prefills into 4096-token blocks, and turn on expert-parallel load balancing." That's the payoff this lesson promised: when every flag maps to a lesson you've studied, the whole command turns from "a string of incantations" into "a deployment spec you can actually read".</p>

<table class="t"><tr><th>Key flag</th><th>What it controls (→ which lesson)</th></tr>
<tr><td><span class="mono">--tp-size</span> / <span class="mono">--dp-size</span></td><td>Scale of tensor / data parallelism (Lesson 46)</td></tr>
<tr><td><span class="mono">--attention-backend</span></td><td>Attention kernel: flashinfer / triton / fa (Lesson 33)</td></tr>
<tr><td><span class="mono">--quantization</span></td><td>Weight quantization: fp8 / awq / gptq (Lesson 35)</td></tr>
<tr><td><span class="mono">--mem-fraction-static</span></td><td>GPU memory fraction for static allocation (weights + KV pool) (Lessons 30/31)</td></tr>
<tr><td><span class="mono">--chunked-prefill-size</span></td><td>Split long prefills to smooth memory (Lesson 22)</td></tr>
<tr><td><span class="mono">--max-running-requests</span></td><td>Cap on concurrent requests, i.e. batch size (Lesson 5)</td></tr>
<tr><td><span class="mono">--speculative-algorithm</span></td><td>Enable / choose the speculative decoding algorithm (Lesson 43)</td></tr>
<tr><td><span class="mono">--enable-eplb</span></td><td>Expert-parallel load balancing, EPLB (Lesson 47)</td></tr>
</table>

<h2>3. Two faces: offline Engine vs online HTTP</h2>
<p>Same core, two ways to use it. The <strong>HTTP server</strong> is the default "online" mode: you start a long-lived process that listens on a port and clients send requests over the network — the standard posture for production and serving an external API. The <strong>Engine</strong> is the "offline / in-process" mode: you <span class="mono">import</span> and instantiate it directly in your own Python script, with no HTTP and no network, calling generation straight in memory. It's perfect for batch eval, data generation, and experiments — skipping all the overhead of standing up a service and firing HTTP requests. Both sit on the same engine; the only difference is whether that HTTP facade is there.</p>
<p>How to choose? If you need to serve external clients, support many concurrent users, or plug into existing OpenAI-ecosystem tooling, use the HTTP server; if you just want to run a few thousand prompts in a script for offline eval, or embed a one-shot batch generation inside a training pipeline, <span class="mono">Engine</span> is lighter, faster, and easier to debug. The key realization: both modes share the <strong>exact same tuning knobs</strong> — whichever path you take, you set the same <span class="mono">ServerArgs</span> (or its equivalent constructor args), so the "flag ↔ component" map from this lesson applies to online and offline alike. That's the elegance of SGLang's design: one core, swappable facades.</p>

<div class="cols"><div class="col"><strong>Engine (offline / in-process)</strong><br>Instantiate directly in a Python script, no HTTP, no network. Call generation in memory. Great for scripting, batch eval, offline data generation. Fast to start, low overhead.</div><div class="col"><strong>HTTP server (online)</strong><br>Long-lived process listening on a port, clients send requests over the network. Exposes <span class="mono">/generate</span> and OpenAI-compatible <span class="mono">/v1/...</span>. Great for production, external APIs, many concurrent clients.</div></div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Two ways to use SGLang: the offline in-process Engine calls generate directly with no network, the online HTTP server listens on a port for REST requests; the same core sits underneath">
    <rect x="30" y="40" width="320" height="150" rx="12" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="190" y="66" text-anchor="middle" style="fill:var(--teal);font-weight:700">Engine · offline (in-process)</text>
    <rect x="50" y="84" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="64" y="106" class="mono" style="font-size:11px">import sglang as sgl</text>
    <rect x="50" y="126" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="64" y="148" class="mono" style="font-size:11px">engine.generate(prompts)</text>
    <text x="190" y="180" text-anchor="middle" style="fill:var(--muted);font-size:11px">no network · batch / offline</text>
    <rect x="430" y="40" width="320" height="150" rx="12" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="590" y="66" text-anchor="middle" style="fill:var(--blue);font-weight:700">HTTP server · online</text>
    <rect x="450" y="84" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--blue);stroke-width:1.5"/>
    <text x="464" y="106" class="mono" style="font-size:11px">launch_server process</text>
    <rect x="450" y="126" width="280" height="34" rx="7" style="fill:var(--panel-2);stroke:var(--blue);stroke-width:1.5"/>
    <text x="464" y="148" class="mono" style="font-size:11px">POST /generate (REST)</text>
    <text x="590" y="180" text-anchor="middle" style="fill:var(--muted);font-size:11px">many clients · online serving</text>
    <line x1="190" y1="190" x2="190" y2="232" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="185,232 195,232 190,238" style="fill:var(--muted)"/>
    <line x1="590" y1="190" x2="590" y2="232" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="585,232 595,232 590,238" style="fill:var(--muted)"/>
    <rect x="150" y="238" width="480" height="48" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="390" y="260" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">same core</text>
    <text x="390" y="278" text-anchor="middle" style="fill:var(--muted);font-size:10px">engine + ServerArgs</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Offline Engine vs online HTTP</b> — left: <span class="mono">import sglang</span> and call <span class="mono">engine.generate(...)</span> in-process, no network (batch / offline); right: launch a server process and clients send REST requests (online, many clients). The same core sits underneath both.</div>
</div>

<p>Putting the two faces side by side drives home the value of "one core, two facades": during development you can iterate quickly in a script with <span class="mono">Engine</span>, validating prompts and sampling params, and once the logic stabilizes you flip to the HTTP server for production using the <strong>exact same set of flags</strong>, at almost zero migration cost. That consistency isn't a coincidence — both stuff their config into the same <span class="mono">ServerArgs</span> underneath. So when someone asks "how do I set that offline-script parameter on the online service", the answer is always the same: find the matching field name, swap underscores for hyphens, and prefix two dashes.</p>

<h2>4. Quick start in four steps</h2>
<p>Getting it running is really just four steps. Step one, <strong>install</strong>: set up SGLang and its dependencies. Step two, <strong>launch and wait for ready</strong>: run <span class="mono">launch_server</span>, then patiently wait for the log to print that the model has loaded and the service is ready — large models take time to load, so don't rush your first request. Step three, <strong>probe health</strong>: <span class="mono">curl http://localhost:30000/health</span> to confirm it's alive. Step four, <strong>send a request</strong>: POST some JSON to <span class="mono">/generate</span> and get your generation back. As for how to systematically stress-test throughput and latency, we'll save that for the benchmarking in Lesson 54.</p>
<p>A few beginner pitfalls are worth flagging. First, <strong>don't send requests before the "ready" log appears</strong>: model weights are often tens of GB, and loading from disk, moving to GPU, and compiling CUDA graphs all take time — request too early and you just get connection refused. Second, <strong>when memory is tight, reach for <span class="mono">--mem-fraction-static</span> first</strong>: it directly sets the KV pool size, so shrink it to free room for weights and activations, or grow it to hold more concurrent requests. Third, <strong>think through your parallelism on multi-GPU</strong>: <span class="mono">--tp-size</span> shards one model across cards (when it won't fit on one), while <span class="mono">--dp-size</span> runs replicas each serving independently (to raise throughput). Nail these four steps and a few knobs, and you have a stable, externally serviceable SGLang instance; the rest is fine-tuning that row of flags you already recognize to your workload.</p>
<p>One closing principle: <strong>get the service up on defaults first, then change one knob at a time</strong>. Piling on a long string of flags you're unsure about makes failures hard to pin on any single one. The right move is to launch with just <span class="mono">--model-path</span>, confirm health, then add flags targeting the bottleneck you observe: tight memory → tune <span class="mono">--mem-fraction-static</span> or enable <span class="mono">--quantization</span>; low throughput → add parallelism or raise <span class="mono">--max-running-requests</span>; long-context stalls → adjust <span class="mono">--chunked-prefill-size</span>. Change one variable at a time and you'll know each knob's real payoff — which is exactly what the next lesson on benchmarking will teach.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Install</h4><p class="mono">pip install / from source</p><p>Set up SGLang and its dependencies (CUDA, attention backends, etc.) and confirm the GPU driver is ready.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Launch</h4><p class="mono">python -m sglang.launch_server</p><p>Run the launch command and <strong>wait patiently</strong> for the log to say the model has loaded and the service is ready — large models take time.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Health</h4><p class="mono">GET /health</p><p><span class="mono">curl http://localhost:30000/health</span> to confirm it's alive and ready to accept requests.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Generate</h4><p class="mono">POST /generate</p><p>POST some JSON to <span class="mono">/generate</span> and get your result; systematic stress-testing is in Lesson 54.</p></div></div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/server_args.py ::ServerArgs</span><span class="ln">ServerArgs gathers the whole guide's knobs into CLI flags</span></div><pre>@dataclasses.dataclass
class ServerArgs:
    model_path: str                    # --model-path: the HF model to serve
    tp_size: int = 1                   # --tp-size: tensor-parallel ranks (Lesson 46)
    dp_size: int = 1                   # --dp-size: data-parallel replicas (Lesson 46)
    mem_fraction_static: float = None  # --mem-fraction-static: GPU mem fraction for static alloc (weights + KV pool) (Lesson 30)
    attention_backend: str = None      # --attention-backend: flashinfer / triton / fa (Lesson 33)
    chunked_prefill_size: int = None   # --chunked-prefill-size: split long prefills (Lesson 22)
    quantization: str = None           # --quantization: fp8 / awq / gptq / ... (Lesson 35)
    max_running_requests: int = None   # --max-running-requests: batch cap (Lesson 5)
    # ... dozens more fields; each auto-maps to a --kebab-case CLI flag</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/launch_server.py ::run_server</span><span class="ln">pick and launch the right server flavor from ServerArgs</span></div><pre>def run_server(server_args):
    # launch the right server flavor based on the parsed ServerArgs.
    if server_args.grpc_mode:
        serve_grpc(server_args)             # gRPC entrypoint
    else:
        # the default: HTTP / OpenAI-compatible server
        from sglang.srt.entrypoints.http_server import launch_server
        launch_server(server_args)          # builds engine + HTTP routes</pre></div>

<p>Two concrete examples to line up the path. <strong>Online:</strong> <span class="mono">python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --tp-size 2</span> — these flags are parsed into <span class="mono">ServerArgs</span>, handed to <span class="mono">run_server</span>, which takes the default branch to build the engine and serves HTTP on <span class="mono">:30000</span>. <strong>Offline:</strong> <span class="mono">sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct").generate(prompts)</span> — the same config takes the <span class="mono">Engine</span> flavor, generating in-process and skipping the network entirely.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li>Launching takes one command: <span class="mono">python -m sglang.launch_server --model-path &lt;model&gt; [flags]</span>.</li>
<li>The entry point uses <span class="mono">prepare_server_args</span> to parse argv into <span class="mono">ServerArgs</span> → hands it to <span class="mono">run_server</span> to boot the engine (the trio, Lesson 13) → exposes HTTP (<span class="mono">/generate</span>, <span class="mono">/v1/...</span>, Lesson 15).</li>
<li><strong>Core insight</strong>: each <span class="mono">ServerArgs</span> dataclass field auto-maps to a <span class="mono">--kebab-case</span> flag, so every component from the first 52 lessons becomes a knob.</li>
<li>Tuning = setting the right flags: <span class="mono">--tp-size</span>/<span class="mono">--dp-size</span>, <span class="mono">--attention-backend</span>, <span class="mono">--quantization</span>, <span class="mono">--mem-fraction-static</span>, <span class="mono">--chunked-prefill-size</span>, <span class="mono">--max-running-requests</span>, and more.</li>
<li><span class="mono">Engine</span> (offline / in-process, no HTTP) vs HTTP server (online): same core, two facades.</li>
<li>Four steps to start: install → launch and wait for ready → probe <span class="mono">/health</span> → POST <span class="mono">/generate</span>; benchmarking in Lesson 54.</li>
</ul></div>
"""}
LESSON_54 = {"zh": r"""
<p class="lead">第8课我们讲过<strong>吞吐</strong>与<strong>延迟</strong>是一对此消彼长的指标，但那时只是定性地说"它们会冲突"。这一课我们要学会真正<strong>量化</strong>它们：用 <span class="mono">bench_serving</span> 给一台正在运行的服务器打分，读懂 <span class="mono">request_throughput</span>、<span class="mono">ttft</span>、<span class="mono">tpot</span> 这些数字，再用 <strong>profiler</strong> 把"它很慢"变成"就是这个 kernel 慢"。一句话：<strong>压测负责回答"有多慢"，分析负责回答"为什么慢"</strong>。这两件事缺一不可：只压测不分析，你知道系统慢却找不到病根；只分析不压测，你盯着一堆 kernel 却不知道哪个真正影响了用户。把它们配成一对，才构成一个能持续改进性能的闭环。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把服务器想象成一家<strong>餐厅</strong>。<strong>压测（benchmark）</strong>就像派一群顾客按固定节奏进店点餐，然后在门口记录两件事：每小时上了多少道菜（<span class="mono">throughput</span>），以及每位顾客从坐下到<strong>第一口菜上桌</strong>等了多久（这就是 <span class="mono">ttft</span>），后续每道菜之间的间隔有多稳（这就是 <span class="mono">tpot</span>）。你站在门口只看到这些<strong>黑盒数字</strong>，并不知道厨房里发生了什么。</p>
<p>而 <strong>性能分析（profile）</strong>就是直接走进厨房，架起一台摄像机录下整条流水线：哪一步在切菜（prefill）、哪一步在装盘（decode）、哪个厨师在发呆等炉子（gap/bubble）。当门口的数字难看时，只有走进厨房才知道到底是谁拖了后腿。<strong>先在门口量化，再进厨房定位</strong>——这就是本课的全部思路。这个类比还提醒我们一件事：摄像机本身也会占用厨房的人手，录像会让出菜稍微变慢，所以你不会全天候开着它，而是在怀疑某段时间有问题时才打开、录一小段就关掉。profiler 同理——开着它会拖慢服务、产生庞大的 trace，因此只在定位问题时短暂启用，平时让服务器全速干活。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>性能调优永远是一个<strong>两段式循环</strong>：<span class="mono">benchmark → profile → 优化 → 再 benchmark</span>。<strong>benchmark 是黑盒</strong>，它向一台真实运行的服务器发射可配置的请求流（请求速率、输入/输出长度、ShareGPT 这类数据集），最后吐出一份 <span class="mono">BenchmarkMetrics</span> 汇总，告诉你"快还是慢、慢在第几分位"。<strong>profiler 是白盒</strong>，它录下 GPU 上每一个 CUDA kernel 的时间线，告诉你"慢在哪个阶段、哪个 kernel"。</p>
<p>关键认知：在压力之下，延迟不是一个数，而是一条<strong>分布</strong>——mean / median / p90 / p95 / p99。普通用户感受到的卡顿往往来自<strong>长尾（p99）</strong>，而不是平均值。所以一份诚实的压测报告必须给出<strong>分位数</strong>，而不是只报一个漂亮的均值。再换个角度看这两件工具的分工：benchmark 给你的是<strong>结果</strong>（多快、多稳、能扛多大并发），profiler 给你的是<strong>过程</strong>（这段时间里 GPU 到底在忙什么）。结果告诉你"该不该优化、优化目标是否达成"，过程告诉你"具体动哪里"。一个负责对外承诺 SLA，一个负责对内指导改代码，二者合起来才能把性能工作从玄学变成工程。</p>
<p>这个两段式循环不是做一次就完事，而是要反复迭代：每次优化后都重新压测确认数字真的变好了，再决定是否还需要继续分析下一个瓶颈。性能优化没有终点，只有"在当前 SLA 下是否够用"的判断。</p>
</div>

<h2>一、压测：给运行中的服务器打分</h2>
<p>SGLang 自带的压测工具是 <span class="mono">python -m sglang.bench_serving</span>（新版已迁移为 <span class="mono">python -m sglang.benchmark.serving</span>，旧入口仍可用）。它的工作方式很直白：你先把服务器跑起来，然后这个脚本<strong>不修改服务器</strong>，只是按你设定的<strong>请求速率（request rate）</strong>和<strong>输入/输出长度</strong>，从外部发射一批请求，逐个记录每条请求的 <span class="mono">ttft</span> 和 <span class="mono">tpot</span>，跑完后聚合成一份 <span class="mono">BenchmarkMetrics</span>。你可以用 ShareGPT 这样的真实对话数据集，让请求长度分布贴近线上场景。</p>
<p>为什么强调"对运行中的服务器"？因为压测要测的不是某段代码理论上能多快，而是<strong>真实部署在并发压力下的综合表现</strong>。请求速率决定了压力大小：速率越高，调度器排队越久，<span class="mono">ttft</span> 的长尾就越明显；输入长度影响 prefill 的计算量，输出长度影响 decode 的步数。把这些旋钮调到贴近线上流量，压测数字才有参考价值，否则容易得到"实验室里很快、上线就崩"的误导结论。</p>
<p>这里有个容易混淆的对照：<span class="mono">bench_serving</span> 测的是<strong>整台服务器在并发负载下的端到端表现</strong>；而 <span class="mono">bench_one_batch</span> 测的是<strong>单个 batch 在隔离环境下的延迟</strong>，根本不需要起服务器，特别适合做 kernel 级别的 A/B 对比（比如换一个 attention 实现快了多少）。一句话区分：要看"系统级表现"用 <span class="mono">bench_serving</span>，要看"某段计算本身有多快"用 <span class="mono">bench_one_batch</span>。两者各有用武之地：前者贴近用户真实体验，后者排除了排队、网络、调度的干扰，能干净地比较两份内核实现的纯计算耗时。做优化时常常是先用 <span class="mono">bench_serving</span> 发现整体慢，再用 <span class="mono">bench_one_batch</span> 把某个可疑算子单独拎出来精确对比。还要注意数据集的选择会显著影响结论：用全是等长的合成请求，得到的数字往往过于乐观，因为它掩盖了真实流量里长短不一带来的排队与碎片化；而用 ShareGPT 这类真实对话，请求长度参差，更能暴露调度器在混合负载下的真实表现。所以"用什么数据集压测"本身就是一个需要交代清楚的实验条件。</p>

<h2>二、读懂那些数字：吞吐与延迟</h2>
<p>压测报告里的数字分成两大类。<strong>吞吐类</strong>回答"单位时间干了多少活"：<span class="mono">request_throughput</span> 是每秒完成多少条请求；<span class="mono">input_throughput</span> 是每秒吃进多少 prompt token；<span class="mono">output_throughput</span> 是每秒吐出多少 token——这个值约等于<strong>decode 速度</strong>，是衡量生成快慢最直接的指标。吞吐高，意味着同样的 GPU 单位时间能服务更多用户、摊薄每条请求的成本，所以它是运维和成本核算最关心的数字。</p>
<p><strong>延迟类</strong>回答"用户要等多久"：<span class="mono">ttft</span>（Time To First Token）是从发出请求到<strong>第一个 token 出现</strong>的时间，本质上就是 <strong>prefill 延迟</strong>，也是用户盯着屏幕"还没出字"那段焦灼的等待；<span class="mono">tpot</span>（Time Per Output Token）是<strong>相邻两个 token 之间的间隔</strong>，也叫 inter-token latency（ITL），决定了文字往外蹦得<strong>顺不顺、卡不卡</strong>。这两个指标对应着用户体验的两个阶段：<span class="mono">ttft</span> 决定"点了发送之后多久看到反应"，<span class="mono">tpot</span> 决定"开始出字之后读起来流不流畅"。一个聊天应用如果 ttft 很长，用户会觉得"卡住了没反应"；如果 tpot 不稳，用户会觉得"字一顿一顿地蹦"。</p>
<p>回想第8课：吞吐高未必延迟低，把 batch 堆大能拉高 <span class="mono">output_throughput</span>，却可能把单条请求的 <span class="mono">tpot</span> 拖长——这正是压测要帮你权衡的取舍。换句话说，同一份硬件上，你可以选择"高吞吐、稍高延迟"的服务策略（适合离线批量任务），也可以选择"低延迟、稍低吞吐"的策略（适合实时交互）。压测的价值就在于把这条取舍曲线<strong>量化成具体数字</strong>，让你能根据业务的 SLA（比如"p99 的 ttft 必须低于 500ms"）去反推该怎么配置服务器。没有这些数字，所谓"调优"就只是凭感觉拍脑袋。</p>

<p>这两个吞吐与延迟指标之间还有一层换算直觉值得建立：一条请求的端到端耗时大致等于 <span class="mono">ttft</span> 加上"输出 token 数 × <span class="mono">tpot</span>"。也就是说，prefill 阶段贡献了开头那一下等待，decode 阶段则按 token 数线性累加。对短输出请求，<span class="mono">ttft</span> 往往主导体验；对长输出请求（比如写长文、出代码），<span class="mono">tpot</span> 才是大头。理解这个拆分，你就知道：想让"首屏出字"更快该去优化 prefill，想让"长篇生成"整体更快该去优化 decode——而 trace 里 prefill 区段和 decode 区段的相对长度，正好印证你的请求形态落在哪一侧。</p>

<h2>三、为什么必须看分位数</h2>
<p>同一批请求的 <span class="mono">ttft</span> 和 <span class="mono">tpot</span> 从来不是一个固定值，而是一条<strong>分布</strong>。压测会给出 mean、median、p90、p95、p99 多个口径。为什么 p99 这么重要？因为在高并发下，调度排队、CUDA graph 重放(replay)的抖动、偶发的长 prompt 都会制造<strong>长尾</strong>——也许 99% 的请求都很快，但那 1% 的卡顿恰恰是用户印象最深的"它有时候巨慢"。只报均值会掩盖这种尾部痛苦。</p>
<p>举个直观的例子：假设 100 条请求里 99 条的 ttft 是 100ms，1 条因为排在长 prompt 后面变成了 5000ms。算平均值大约是 149ms，看上去很漂亮；但那 1 条 5000ms 的请求背后是一个真实的、正在抓狂的用户。<span class="mono">p99</span> 会忠实地把这个 5000ms 暴露出来，而均值把它稀释掉了。线上系统的口碑往往不是被平均体验决定的，而是被<strong>最差的那批体验</strong>决定的——这就是为什么严肃的 SLA 几乎都用分位数来定义，而不是用均值。</p>
<p>所以记住：<strong>负载之下，报 p99，别只报 mean</strong>。同时也别忘了 median（p50）——当 mean 和 median 差得很远时，本身就说明分布是偏斜的、存在长尾，这是一个值得深究的信号。一份合格的压测报告应该把 mean / median / p90 / p95 / p99 一起列出来，让你既看到典型情况，也看到最坏情况。</p>
<p>分位数还有一个实用价值：它让<strong>容量规划</strong>变得有据可依。当你逐步提高请求速率，p99 的 ttft 会先平稳、再在某个临界点陡然飙升——那个拐点就是这台服务器的<strong>承载上限</strong>。低于拐点，加压几乎不影响尾延迟；越过拐点，队列开始堆积，尾延迟一发不可收拾。把"速率—p99"这条曲线画出来，你就能回答"这台机器在保证 SLA 的前提下最多能扛多少 QPS"，进而决定要不要扩容、扩几台。这远比"看 GPU 利用率拍脑袋"靠谱得多。</p>

<h2>四、性能分析：把"慢"定位到 kernel</h2>
<p>当数字难看时，光知道"慢"没用，你得知道"慢在哪"。SGLang 支持设置环境变量 <span class="mono">SGLANG_TORCH_PROFILER_DIR</span>（或调用 <span class="mono">/start_profile</span> 与 <span class="mono">/stop_profile</span> 两个 HTTP 端点）来抓取一段 <strong>torch profiler trace</strong>，然后用 Chrome 的 <span class="mono">chrome://tracing</span> 或 Perfetto 打开。这条时间线会画出 GPU 上<strong>每一个 CUDA kernel</strong>，于是你能一眼分辨出 <strong>prefill 区段</strong>和 <strong>decode 区段</strong>，看到 kernel 之间的<strong>空隙（gap/bubble，往往是 launch 开销）</strong>，也能看到第27课讲的 <strong>CUDA graph 重放</strong>把许多小 kernel 合并后的样子。至此，"它很慢"就被翻译成了"就是这个 kernel / 这段空隙的问题"。</p>
<p>用环境变量和用 HTTP 端点抓 trace 的区别在于时机：环境变量适合"从头到尾全程录制"，端点则适合"在服务器已经跑起来、想精准框出某一小段时间"——你在压测脚本发射负载前调 <span class="mono">/start_profile</span>，结束后调 <span class="mono">/stop_profile</span>，就只录下这段最有代表性的负载，trace 文件也不会大到打不开。打开 trace 后要重点看三件事：哪些 kernel 占了最多时间（热点）、kernel 之间有没有大段空白（说明 CPU 来不及喂 GPU，是 launch 或调度开销）、decode 区段的小 kernel 是否已经被 CUDA graph 收拢成密实的一片。</p>
<p>把这套思路和前面几课串起来：第4课告诉我们 prefill 是<strong>计算受限（compute-bound）</strong>、decode 是<strong>带宽受限（bandwidth-bound）</strong>，所以在 trace 里这两段长得就不一样——prefill 区段往往是少数几个又大又满的矩阵乘 kernel，decode 区段则是大量又小又碎、每步只处理一个 token 的 kernel；第8课的吞吐/延迟取舍正是这些指标在度量的东西；第27课的 CUDA graph 会让 decode 区段的小 kernel 变得密实整齐，消除掉那些 launch 空隙。把握住"benchmark 量化、profile 定位"这条主线，你就能在"它慢"和"为什么慢"之间自由切换。下一课（第55课）转向测试套件与 CI——确保引擎不仅跑得快，更要跑得对。</p>
<p>最后强调一个常被忽视的纪律：<strong>压测的可重复性</strong>。同一台机器、同一个模型，如果两次压测的请求速率、输入输出长度、数据集、并发数不一致，得到的数字就没有可比性。要做有意义的 A/B（比如"开了 CUDA graph 快多少"），必须<strong>只改一个变量</strong>，其余全部固定，并且每次都先充分预热再正式测量。同样地，profile 也要在<strong>稳态</strong>下采集——别把冷启动、第一次 graph 捕获那几步算进来，否则 trace 里全是一次性的开销，会把你引向错误的瓶颈。养成"固定变量、预热、采稳态、报分位"的习惯，你的性能数字才经得起复现和质疑，优化才不会建立在噪声之上。</p>

<div class="flow"><div class="node">负载生成器<br><span class="mono">bench_serving</span></div><div class="arrow">→</div><div class="node">运行中的<br>SGLang 服务器</div><div class="arrow">→</div><div class="node">逐请求收集<br>TTFT / TPOT</div><div class="arrow">→</div><div class="node">聚合成<br><span class="mono">BenchmarkMetrics</span></div></div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="压测流水线：负载生成器按速率发射 N 个请求到 SGLang 服务器，服务器流式响应，逐请求采集 TTFT 与 TPOT，最后汇总成吞吐与 P50/P95/P99 百分位">
    <text x="20" y="28" style="font-weight:700;fill:var(--accent-ink)">压测流水线：发请求 → 采延迟 → 算百分位</text>
    <rect x="18" y="50" width="165" height="66" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="100" y="78" text-anchor="middle" style="font-weight:700">负载生成器</text>
    <text x="100" y="100" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">N 请求 @ rate</text>
    <text x="192" y="90" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="205" y="50" width="165" height="66" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="287" y="78" text-anchor="middle" style="font-weight:700">SGLang 服务器</text>
    <text x="287" y="100" text-anchor="middle" style="font-size:11px;fill:var(--muted)">流式响应</text>
    <text x="379" y="90" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="392" y="50" width="165" height="66" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="474" y="78" text-anchor="middle" style="font-weight:700">逐请求采集</text>
    <text x="474" y="100" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">TTFT · TPOT</text>
    <text x="566" y="90" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="579" y="50" width="165" height="66" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="661" y="78" text-anchor="middle" style="font-weight:700">汇总指标</text>
    <text x="661" y="100" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">P50·P95·P99</text>
    <text x="18" y="150" style="fill:var(--muted);font-size:12px">延迟分布（右偏长尾）</text>
    <line x1="40" y1="215" x2="740" y2="215" style="stroke:var(--line);stroke-width:1.5"/>
    <polyline points="40,212 130,168 210,176 330,198 470,208 620,212 740,214" style="fill:none;stroke:var(--accent);stroke-width:2"/>
    <line x1="210" y1="176" x2="210" y2="215" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="210" y="232" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">P50</text>
    <line x1="560" y1="210" x2="560" y2="215" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="560" y="232" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--amber)">P95</text>
    <line x1="660" y1="213" x2="660" y2="215" style="stroke:var(--purple);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="664" y="232" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--purple)">P99 尾</text>
  </svg>
  <div class="figcap"><b>图 1 · 压测流水线与百分位</b> — 负载生成器按 <span class="mono">request_rate</span> / 并发把 N 个请求打到服务器，服务器流式返回，逐请求记下 <span class="mono">TTFT</span> 与 <span class="mono">TPOT</span>，再汇总成吞吐与 P50/P95/P99；右偏长尾里 <span class="mono">P99</span> 才是用户真正的痛点。</div>
</div>

<p><strong>一个具体例子</strong>：<span class="mono">python -m sglang.bench_serving --backend sglang --request-rate 16 --num-prompts 1000</span> 会以 16 req/s 的速率把 1000 条 prompt 打到服务器，跑完后打印 <span class="mono">output_throughput</span> 以及 P50/P99 的 <span class="mono">ttft</span>；如果尾延迟难看，再开 profiler 看一段 trace，就能判断到底是 attention 还是 GEMM 占了大头。</p>

<table class="t"><tr><th>指标</th><th>含义</th></tr>
<tr><td><span class="mono">request_throughput</span></td><td>每秒完成的请求数（req/s）</td></tr>
<tr><td><span class="mono">input_throughput</span></td><td>每秒吃进的 prompt token 数（tokens/s）</td></tr>
<tr><td><span class="mono">output_throughput</span></td><td>每秒吐出的 token 数 ≈ decode 速度</td></tr>
<tr><td><span class="mono">ttft</span></td><td>Time To First Token = prefill 延迟，用户等待出字的时间</td></tr>
<tr><td><span class="mono">tpot</span></td><td>Time Per Output Token = token 间隔（ITL），decode 顺滑度</td></tr>
<tr><td><span class="mono">p99</span></td><td>分布的尾部——负载之下用户真正感受到的卡顿</td></tr></table>

<div class="cols"><div class="col"><strong>benchmark（黑盒数字）</strong><br>向真实服务器发射负载，问"有多慢？慢在第几分位？"。产出 <span class="mono">request_throughput</span>、<span class="mono">ttft</span>、<span class="mono">tpot</span> 等汇总数字，负责<strong>量化</strong>。</div><div class="col"><strong>profiler（白盒时间线）</strong><br>录下每个 CUDA kernel 的时间线，问"为什么慢？慢在哪个 kernel / 哪个阶段？"。产出可在 <span class="mono">chrome://tracing</span> 打开的 trace，负责<strong>定位</strong>。</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="对比：基准只给出 tokens/s 与 TTFT 等结果数字，内部不可见是黑盒；profiler 时间线展开 GPU 上的 Attn 与 GEMM 等 kernel，看清 GEMM 最久是瓶颈，是白盒">
    <text x="20" y="28" style="font-weight:700;fill:var(--accent-ink)">基准（黑盒）vs profiler（白盒）</text>
    <rect x="18" y="48" width="360" height="224" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="198" y="74" text-anchor="middle" style="font-weight:700;fill:var(--muted)">基准：只有结果</text>
    <rect x="120" y="92" width="156" height="92" rx="8" style="fill:var(--ink);stroke:var(--line);stroke-width:1.5;opacity:0.85"/>
    <text x="198" y="148" text-anchor="middle" style="font-size:38px;fill:var(--faint)">?</text>
    <text x="34" y="120" style="font-size:12px;fill:var(--muted)">请求流 →</text>
    <text x="198" y="212" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--ink)">tokens/s · TTFT</text>
    <text x="198" y="246" text-anchor="middle" style="font-size:12px;fill:var(--muted)">知结果，不知原因</text>
    <rect x="422" y="48" width="360" height="224" rx="10" style="fill:var(--panel-2);stroke:var(--accent);stroke-width:1.5"/>
    <text x="602" y="74" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">profiler：看见过程</text>
    <line x1="442" y1="172" x2="762" y2="172" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="442" y="122" width="78" height="48" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="481" y="151" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">Attn</text>
    <rect x="524" y="122" width="140" height="48" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="594" y="151" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--amber)">GEMM</text>
    <rect x="668" y="122" width="30" height="48" rx="5" style="fill:var(--faint);stroke:var(--line);stroke-width:1.5"/>
    <rect x="702" y="122" width="60" height="48" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="732" y="151" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">Attn</text>
    <text x="683" y="192" text-anchor="middle" style="font-size:10px;fill:var(--faint)">bubble</text>
    <text x="602" y="232" text-anchor="middle" style="font-size:12px;fill:var(--ink)">GPU kernel 时间线</text>
    <text x="602" y="258" text-anchor="middle" style="font-weight:700;fill:var(--amber)">GEMM 最久 → 瓶颈</text>
  </svg>
  <div class="figcap"><b>图 2 · 黑盒数字 vs 白盒时间线</b> — 基准给出 <span class="mono">tokens/s</span>、<span class="mono">TTFT</span> 等结果，却看不到内部（黑盒）；profiler 把盒子打开成一条 GPU kernel 时间线，<span class="mono">Attn</span> / <span class="mono">GEMM</span> / 空隙各占多少一目了然，于是能定位到「<span class="mono">GEMM</span> 最久」这种瓶颈（白盒）。</div>
</div>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>预热服务器（warm up）</h4><p>先发几条请求，让 CUDA graph 捕获、显存分配、缓存都就绪，避免冷启动污染数据。</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>发射负载</h4><p>设定 <span class="mono">request rate</span> / 输入输出长度 / 数据集（如 ShareGPT），正式打压。</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>读取分位数</h4><p>看 mean / median / p90 / p95 / <span class="mono">p99</span>，重点关注尾部。</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>打开 profiler trace</h4><p>用 <span class="mono">chrome://tracing</span> / Perfetto 查看每个 kernel，定位瓶颈。</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/benchmark/serving.py ::BenchmarkMetrics</span><span class="ln">吞吐 + TTFT/TPOT 的分位数汇总</span></div><pre>@dataclass
class BenchmarkMetrics:
    completed: int                 # 完成的请求数
    request_throughput: float      # 每秒请求数 (req/s)
    input_throughput: float        # 每秒输入(prompt) token 数
    output_throughput: float       # 每秒输出 token 数 (decode 速度)
    mean_ttft_ms: float            # Time To First Token = prefill 延迟
    median_ttft_ms: float
    p99_ttft_ms: float             # 负载下用户真正感受到的尾部
    mean_tpot_ms: float            # Time Per Output Token = token 间隔 (ITL)
    p99_tpot_ms: float
    # ... 还有 p90 / p95 变体, 以及 e2e 端到端延迟</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/benchmark/serving.py ::benchmark</span><span class="ln">负载生成器：按速率打请求，采集并汇总延迟指标</span></div><pre>async def benchmark(backend, api_url, base_url, model_id, tokenizer,
                    input_requests, request_rate, max_concurrency, ...):
    # 负载生成器：按目标 request_rate 发射 input_requests
    # （受 max_concurrency 限流），逐请求记录 TTFT/TPOT，
    # 再归约成吞吐 + 延迟百分位（BenchmarkMetrics）。
    ...</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>两段式循环</strong>：<span class="mono">benchmark</span> 量化"有多慢"，<span class="mono">profile</span> 定位"为什么慢"。</li>
<li><span class="mono">python -m sglang.bench_serving</span> 向<strong>运行中的</strong>服务器发射可配置负载，产出 <span class="mono">BenchmarkMetrics</span> 汇总。</li>
<li><strong>吞吐</strong>：<span class="mono">request_throughput</span> / <span class="mono">input_throughput</span> / <span class="mono">output_throughput</span>（输出吞吐≈decode 速度）。</li>
<li><strong>延迟</strong>：<span class="mono">ttft</span> = prefill 等待，<span class="mono">tpot</span> = token 间隔(ITL) = decode 顺滑度。</li>
<li>延迟是<strong>分布</strong>：报 mean / median / <strong>p99</strong>，负载之下尾部才是用户的真实体验。</li>
<li><span class="mono">bench_one_batch</span> 在<strong>无服务器</strong>下测单 batch 延迟，适合 kernel 级 A/B。</li>
<li>用 <span class="mono">SGLANG_TORCH_PROFILER_DIR</span> 或 <span class="mono">/start_profile</span>+<span class="mono">/stop_profile</span> 抓 trace，在 <span class="mono">chrome://tracing</span> 看 prefill/decode、gap、CUDA graph(第27课)。</li>
<li>串联：第8课吞吐/延迟取舍、第4课 prefill 计算受限 vs decode 带宽受限；下节第55课转向测试与 CI。</li>
</ul></div>
""", "en": r"""
<p class="lead">In Lesson 8 we learned that <strong>throughput</strong> and <strong>latency</strong> are a trade-off pair, but back then we only said qualitatively that "they conflict." This lesson is about truly <strong>quantifying</strong> them: using <span class="mono">bench_serving</span> to score a running server, reading numbers like <span class="mono">request_throughput</span>, <span class="mono">ttft</span>, <span class="mono">tpot</span>, and then using a <strong>profiler</strong> to turn "it's slow" into "it's exactly this kernel that's slow." In one line: <strong>benchmark answers "how slow," profiling answers "why slow."</strong> Neither can be skipped: benchmark without profile and you know the system is slow but can't find the root cause; profile without benchmark and you stare at a pile of kernels without knowing which actually hurt users. Pair them up and you have a closed loop that keeps improving performance.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture the server as a <strong>restaurant</strong>. A <strong>benchmark</strong> is like sending a stream of customers in at a fixed pace to order, then standing at the door recording two things: how many dishes come out per hour (<span class="mono">throughput</span>), and how long each guest waits from sitting down to the <strong>first dish arriving</strong> (that's <span class="mono">ttft</span>), plus how steady the gaps between later dishes are (that's <span class="mono">tpot</span>). At the door you only see these <strong>black-box numbers</strong>; you have no idea what's happening in the kitchen.</p>
<p>A <strong>profile</strong> means walking straight into the kitchen and mounting a camera over the whole line: which step is chopping (prefill), which is plating (decode), which cook is idly waiting on a burner (a gap/bubble). When the numbers at the door look bad, only by going into the kitchen do you find who held things up. <strong>Quantify at the door, then locate in the kitchen</strong> — that's the whole idea of this lesson. This analogy also reminds us of something: the camera itself ties up kitchen hands, recording makes plating a bit slower, so you don't leave it on all day but only switch it on when you suspect a stretch is problematic, record a short clip, and turn it off. The profiler is the same — leaving it on slows the service and produces huge traces, so enable it only briefly while locating a problem and otherwise let the server run at full speed.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Performance tuning is always a <strong>two-stage loop</strong>: <span class="mono">benchmark → profile → optimize → benchmark again</span>. <strong>The benchmark is black-box</strong>: it fires a configurable request load (request rate, input/output lengths, datasets like ShareGPT) at a real running server and finally emits a <span class="mono">BenchmarkMetrics</span> summary telling you "fast or slow, and slow at which percentile." <strong>The profiler is white-box</strong>: it records the timeline of every CUDA kernel on the GPU, telling you "which phase, which kernel is slow."</p>
<p>The key insight: under load, latency is not a single number but a <strong>distribution</strong> — mean / median / p90 / p95 / p99. The lag ordinary users feel usually comes from the <strong>tail (p99)</strong>, not the mean. So an honest benchmark report must give <strong>percentiles</strong>, not just one pretty average. Another way to see the division of labor between these two tools: the benchmark gives you <strong>results</strong> (how fast, how steady, how much concurrency it sustains), the profiler gives you <strong>process</strong> (what the GPU is actually busy with during that time). Results tell you "whether to optimize and whether the goal is met," process tells you "exactly where to act." One commits SLAs externally, the other guides code changes internally; together they turn performance work from black magic into engineering.</p>
<p>This two-stage loop isn't a one-shot but iterates repeatedly: after each optimization, re-benchmark to confirm the numbers really improved, then decide whether to keep analyzing the next bottleneck. Performance optimization has no end, only the judgment of "is it good enough under the current SLA."</p>
</div>

<h2>1. Benchmark: scoring a running server</h2>
<p>SGLang's built-in benchmark tool is <span class="mono">python -m sglang.bench_serving</span> (now moved to <span class="mono">python -m sglang.benchmark.serving</span>; the old entry still works). Its workflow is straightforward: you start the server first, then this script <strong>does not modify the server</strong> — it just fires a batch of requests from outside at your chosen <strong>request rate</strong> and <strong>input/output lengths</strong>, records each request's <span class="mono">ttft</span> and <span class="mono">tpot</span>, and aggregates them into a <span class="mono">BenchmarkMetrics</span> when done. You can use a real conversation dataset like ShareGPT so request-length distributions match production.</p>
<p>Why insist on "a running server"? Because what we want to measure is not how fast some code can theoretically be, but the <strong>overall behavior of a real deployment under concurrent pressure</strong>. The request rate sets the pressure: the higher the rate, the longer the scheduler queues and the more pronounced the <span class="mono">ttft</span> tail; input length affects prefill compute, output length affects the number of decode steps. Only by tuning these knobs close to real traffic do the numbers become meaningful — otherwise you get the misleading "fast in the lab, crashes in production" conclusion.</p>
<p>Here's an easily confused contrast: <span class="mono">bench_serving</span> measures the <strong>end-to-end behavior of a whole server under concurrent load</strong>; whereas <span class="mono">bench_one_batch</span> measures the <strong>latency of a single batch in isolation</strong>, needing no server at all, ideal for kernel-level A/B comparisons (e.g., how much faster a new attention impl is). One-line distinction: use <span class="mono">bench_serving</span> for "system-level behavior," use <span class="mono">bench_one_batch</span> for "how fast a piece of compute itself is." Each has its place: the former hugs real user experience, while the latter removes the interference of queuing, network, and scheduling to cleanly compare the pure compute time of two kernel implementations. When optimizing, you often first use <span class="mono">bench_serving</span> to find the system is slow, then use <span class="mono">bench_one_batch</span> to pull a suspect operator out for a precise comparison. Note too that the choice of dataset significantly affects conclusions: synthetic requests all of equal length tend to give overly optimistic numbers because they hide the queuing and fragmentation that varied lengths cause in real traffic; whereas a real conversation set like ShareGPT, with uneven request lengths, better exposes the scheduler's true behavior under mixed load. So "which dataset you benchmark with" is itself an experimental condition that must be stated clearly.</p>

<h2>2. Reading the numbers: throughput and latency</h2>
<p>The numbers in a benchmark report fall into two groups. <strong>Throughput</strong> answers "how much work per unit time": <span class="mono">request_throughput</span> is requests finished per second; <span class="mono">input_throughput</span> is prompt tokens ingested per second; <span class="mono">output_throughput</span> is tokens emitted per second — this value roughly equals <strong>decode speed</strong> and is the most direct measure of generation speed. High throughput means the same GPU can serve more users per unit time and amortize the cost per request, so it's the number ops and cost accounting care about most.</p>
<p><strong>Latency</strong> answers "how long the user waits": <span class="mono">ttft</span> (Time To First Token) is the time from sending a request to the <strong>first token appearing</strong>, essentially the <strong>prefill latency</strong>, that anxious stretch where the user stares at a screen with "no text yet"; <span class="mono">tpot</span> (Time Per Output Token) is the <strong>gap between two adjacent tokens</strong>, also called inter-token latency (ITL), which determines how <strong>smooth or choppy</strong> the text streams out. These two map to two stages of user experience: <span class="mono">ttft</span> decides "how long after hitting send before I see a reaction," <span class="mono">tpot</span> decides "once text starts, how fluently it reads." If a chat app's ttft is long, users feel "it froze, no response"; if tpot is unstable, users feel "the text stutters out."</p>
<p>Recall Lesson 8: high throughput doesn't mean low latency — piling up the batch can raise <span class="mono">output_throughput</span> yet stretch a single request's <span class="mono">tpot</span>, exactly the trade-off the benchmark helps you weigh. In other words, on the same hardware you can choose a "high throughput, slightly higher latency" strategy (good for offline batch jobs) or a "low latency, slightly lower throughput" strategy (good for real-time interaction). The value of benchmarking is <strong>quantifying this trade-off curve into concrete numbers</strong>, so you can work backward from your business SLA (e.g., "p99 ttft must be under 500ms") to how the server should be configured. Without these numbers, so-called "tuning" is just guessing.</p>

<p>One more conversion intuition worth building between these throughput and latency metrics: a request's end-to-end time is roughly <span class="mono">ttft</span> plus "number of output tokens × <span class="mono">tpot</span>." That is, the prefill phase contributes the initial wait, while the decode phase accumulates linearly with token count. For short-output requests, <span class="mono">ttft</span> often dominates the experience; for long-output requests (writing long text, generating code), <span class="mono">tpot</span> is the bigger share. Grasp this split and you know: to make "first text on screen" faster, optimize prefill; to make "long-form generation" faster overall, optimize decode — and the relative lengths of the prefill and decode regions in the trace confirm exactly which side your request shape falls on.</p>

<h2>3. Why you must look at percentiles</h2>
<p>The <span class="mono">ttft</span> and <span class="mono">tpot</span> of one batch of requests are never a fixed value but a <strong>distribution</strong>. The benchmark reports mean, median, p90, p95, p99. Why does p99 matter so much? Because under high concurrency, scheduling queues, jitter from CUDA-graph replay, and the occasional long prompt all create a <strong>long tail</strong> — maybe 99% of requests are fast, but that 1% of lag is exactly the "sometimes it's painfully slow" impression users remember most. Reporting only the mean hides that tail pain.</p>
<p>A concrete example: suppose of 100 requests, 99 have a ttft of 100ms and 1 — stuck behind a long prompt — becomes 5000ms. The mean is about 149ms, which looks great; but behind that one 5000ms request is a real, exasperated user. <span class="mono">p99</span> faithfully exposes that 5000ms, while the mean dilutes it away. The reputation of a production system is often determined not by the average experience but by the <strong>worst batch of experiences</strong> — which is why serious SLAs are almost always defined with percentiles, not means.</p>
<p>So remember: <strong>under load, report p99, not just the mean</strong>. And don't forget the median (p50) — when mean and median diverge widely, that itself signals a skewed distribution with a long tail, a signal worth digging into. A solid benchmark report should list mean / median / p90 / p95 / p99 together, letting you see both the typical case and the worst case.</p>
<p>Percentiles also have a practical payoff: they make <strong>capacity planning</strong> evidence-based. As you gradually raise the request rate, p99 ttft stays flat and then, past some critical point, suddenly spikes — that knee is the server's <strong>carrying limit</strong>. Below the knee, adding load barely affects tail latency; past it, queues pile up and tail latency runs away. Plot this "rate vs p99" curve and you can answer "how much QPS can this machine sustain while meeting the SLA," and from there decide whether and by how much to scale out. That is far more reliable than eyeballing GPU utilization and guessing.</p>

<h2>4. Profiling: pinning "slow" to a kernel</h2>
<p>When the numbers look bad, knowing "it's slow" is useless — you need to know "slow where." SGLang lets you set the environment variable <span class="mono">SGLANG_TORCH_PROFILER_DIR</span> (or call the <span class="mono">/start_profile</span> and <span class="mono">/stop_profile</span> HTTP endpoints) to capture a <strong>torch profiler trace</strong>, then open it in Chrome's <span class="mono">chrome://tracing</span> or Perfetto. This timeline draws <strong>every CUDA kernel</strong> on the GPU, so you can tell at a glance the <strong>prefill region</strong> from the <strong>decode region</strong>, see the <strong>gaps (bubbles, often launch overhead)</strong> between kernels, and see what Lesson 27's <strong>CUDA-graph replay</strong> looks like after merging many small kernels. At this point "it's slow" is translated into "it's exactly this kernel / this gap that's the problem."</p>
<p>The difference between capturing a trace via the environment variable versus the HTTP endpoints is timing: the env var suits "record from start to finish," while the endpoints suit "the server is already running and you want to frame a small slice precisely" — you call <span class="mono">/start_profile</span> before your benchmark fires the load and <span class="mono">/stop_profile</span> after, so only the most representative slice is recorded and the trace file won't be too big to open. Once open, focus on three things: which kernels take the most time (hot spots), whether there are large blanks between kernels (the CPU can't feed the GPU fast enough — launch or scheduling overhead), and whether the small kernels in the decode region have already been gathered into a dense band by CUDA graph.</p>
<p>Tying this to earlier lessons: Lesson 4 told us prefill is <strong>compute-bound</strong> and decode is <strong>bandwidth-bound</strong>, so the two regions look different in the trace — the prefill region is often a handful of large, full matmul kernels, while the decode region is a swarm of tiny, fragmented kernels each handling just one token per step; Lesson 8's throughput/latency trade-off is precisely what these metrics measure; Lesson 27's CUDA graph makes the small kernels in the decode region dense and tidy, eliminating those launch gaps. Hold onto the main thread of "benchmark quantifies, profile locates" and you can freely switch between "it's slow" and "why it's slow." Next lesson (Lesson 55) turns to the test suite &amp; CI — making sure the engine is not just fast but correct.</p>
<p>Finally, one often-overlooked discipline: <strong>benchmark reproducibility</strong>. On the same machine and model, if two runs differ in request rate, input/output lengths, dataset, or concurrency, their numbers aren't comparable. To do a meaningful A/B (e.g., "how much faster with CUDA graph on"), you must <strong>change only one variable</strong>, fix everything else, and warm up fully before measuring each time. Likewise, profile in <strong>steady state</strong> — don't include cold start or the first graph-capture steps, or the trace will be full of one-off overhead that leads you to the wrong bottleneck. Build the habit of "fix variables, warm up, sample steady state, report percentiles," and your performance numbers will withstand reproduction and scrutiny, so optimization isn't built on noise.</p>

<div class="flow"><div class="node">load generator<br><span class="mono">bench_serving</span></div><div class="arrow">→</div><div class="node">running<br>SGLang server</div><div class="arrow">→</div><div class="node">collect per-request<br>TTFT / TPOT</div><div class="arrow">→</div><div class="node">aggregate into<br><span class="mono">BenchmarkMetrics</span></div></div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="Benchmark pipeline: the load generator fires N requests at a target rate into the SGLang server, the server streams responses, TTFT and TPOT are recorded per request, then aggregated into throughput and P50/P95/P99 percentiles">
    <text x="20" y="28" style="font-weight:700;fill:var(--accent-ink)">Benchmark pipeline: fire → collect → percentiles</text>
    <rect x="18" y="50" width="165" height="66" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="100" y="78" text-anchor="middle" style="font-weight:700">load generator</text>
    <text x="100" y="100" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">N reqs @ rate</text>
    <text x="192" y="90" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="205" y="50" width="165" height="66" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="287" y="78" text-anchor="middle" style="font-weight:700">SGLang server</text>
    <text x="287" y="100" text-anchor="middle" style="font-size:11px;fill:var(--muted)">streaming</text>
    <text x="379" y="90" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="392" y="50" width="165" height="66" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="474" y="78" text-anchor="middle" style="font-weight:700">per request</text>
    <text x="474" y="100" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">TTFT · TPOT</text>
    <text x="566" y="90" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="579" y="50" width="165" height="66" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="661" y="78" text-anchor="middle" style="font-weight:700">aggregate</text>
    <text x="661" y="100" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">P50·P95·P99</text>
    <text x="18" y="150" style="fill:var(--muted);font-size:12px">latency dist (right-skewed tail)</text>
    <line x1="40" y1="215" x2="740" y2="215" style="stroke:var(--line);stroke-width:1.5"/>
    <polyline points="40,212 130,168 210,176 330,198 470,208 620,212 740,214" style="fill:none;stroke:var(--accent);stroke-width:2"/>
    <line x1="210" y1="176" x2="210" y2="215" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="210" y="232" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">P50</text>
    <line x1="560" y1="210" x2="560" y2="215" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="560" y="232" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--amber)">P95</text>
    <line x1="660" y1="213" x2="660" y2="215" style="stroke:var(--purple);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="664" y="232" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--purple)">P99 tail</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Benchmark pipeline &amp; percentiles</b> — the load generator fires N requests at the server by <span class="mono">request_rate</span> / concurrency, the server streams back, each request's <span class="mono">TTFT</span> and <span class="mono">TPOT</span> is recorded, then aggregated into throughput and P50/P95/P99; in the right-skewed tail, <span class="mono">P99</span> is the user's real pain point.</div>
</div>

<p><strong>A concrete example</strong>: <span class="mono">python -m sglang.bench_serving --backend sglang --request-rate 16 --num-prompts 1000</span> drives 1000 prompts at 16 req/s and, when done, prints <span class="mono">output_throughput</span> plus P50/P99 <span class="mono">ttft</span>; if the tail looks bad, capture a profiler trace next and you can tell whether attention or GEMM dominates.</p>

<table class="t"><tr><th>Metric</th><th>Meaning</th></tr>
<tr><td><span class="mono">request_throughput</span></td><td>requests finished per second (req/s)</td></tr>
<tr><td><span class="mono">input_throughput</span></td><td>prompt tokens ingested per second (tokens/s)</td></tr>
<tr><td><span class="mono">output_throughput</span></td><td>tokens emitted per second ≈ decode speed</td></tr>
<tr><td><span class="mono">ttft</span></td><td>Time To First Token = prefill latency, the wait before text appears</td></tr>
<tr><td><span class="mono">tpot</span></td><td>Time Per Output Token = inter-token latency (ITL), decode smoothness</td></tr>
<tr><td><span class="mono">p99</span></td><td>the tail of the distribution — what users actually feel under load</td></tr></table>

<div class="cols"><div class="col"><strong>benchmark (black-box numbers)</strong><br>Fires load at a real server, asking "how slow? slow at which percentile?". Produces summary numbers like <span class="mono">request_throughput</span>, <span class="mono">ttft</span>, <span class="mono">tpot</span> — it <strong>quantifies</strong>.</div><div class="col"><strong>profiler (white-box timeline)</strong><br>Records the timeline of every CUDA kernel, asking "why slow? which kernel / which phase?". Produces a trace you open in <span class="mono">chrome://tracing</span> — it <strong>locates</strong>.</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Contrast: the benchmark only gives result numbers like tokens/s and TTFT with internals hidden (black box); the profiler timeline unfolds GPU kernels like Attn and GEMM, showing GEMM is longest and the bottleneck (white box)">
    <text x="20" y="28" style="font-weight:700;fill:var(--accent-ink)">benchmark (black box) vs profiler (white box)</text>
    <rect x="18" y="48" width="360" height="224" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="198" y="74" text-anchor="middle" style="font-weight:700;fill:var(--muted)">benchmark: results only</text>
    <rect x="120" y="92" width="156" height="92" rx="8" style="fill:var(--ink);stroke:var(--line);stroke-width:1.5;opacity:0.85"/>
    <text x="198" y="148" text-anchor="middle" style="font-size:38px;fill:var(--faint)">?</text>
    <text x="34" y="120" style="font-size:12px;fill:var(--muted)">requests →</text>
    <text x="198" y="212" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--ink)">tokens/s · TTFT</text>
    <text x="198" y="246" text-anchor="middle" style="font-size:12px;fill:var(--muted)">outcome, not why</text>
    <rect x="422" y="48" width="360" height="224" rx="10" style="fill:var(--panel-2);stroke:var(--accent);stroke-width:1.5"/>
    <text x="602" y="74" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">profiler: see the process</text>
    <line x1="442" y1="172" x2="762" y2="172" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="442" y="122" width="78" height="48" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="481" y="151" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">Attn</text>
    <rect x="524" y="122" width="140" height="48" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="594" y="151" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--amber)">GEMM</text>
    <rect x="668" y="122" width="30" height="48" rx="5" style="fill:var(--faint);stroke:var(--line);stroke-width:1.5"/>
    <rect x="702" y="122" width="60" height="48" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="732" y="151" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">Attn</text>
    <text x="683" y="192" text-anchor="middle" style="font-size:10px;fill:var(--faint)">bubble</text>
    <text x="602" y="232" text-anchor="middle" style="font-size:12px;fill:var(--ink)">GPU kernel timeline</text>
    <text x="602" y="258" text-anchor="middle" style="font-weight:700;fill:var(--amber)">GEMM longest → bottleneck</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Black-box numbers vs white-box timeline</b> — the benchmark gives results like <span class="mono">tokens/s</span> and <span class="mono">TTFT</span> but hides the internals (black box); the profiler opens the box into a GPU kernel timeline where the shares of <span class="mono">Attn</span> / <span class="mono">GEMM</span> / gaps are obvious, so you can pin a bottleneck like "<span class="mono">GEMM</span> is longest" (white box).</div>
</div>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>warm up the server</h4><p>Send a few requests first so CUDA-graph capture, memory allocation, and caches are ready, avoiding cold-start contamination.</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>run the load</h4><p>Set <span class="mono">request rate</span> / input-output lengths / dataset (e.g. ShareGPT) and fire for real.</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>read the percentiles</h4><p>Look at mean / median / p90 / p95 / <span class="mono">p99</span>, focusing on the tail.</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>open the profiler trace</h4><p>Use <span class="mono">chrome://tracing</span> / Perfetto to inspect each kernel and locate the bottleneck.</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/benchmark/serving.py ::BenchmarkMetrics</span><span class="ln">throughput + the TTFT/TPOT percentile summary</span></div><pre>@dataclass
class BenchmarkMetrics:
    completed: int                 # number of finished requests
    request_throughput: float      # requests / second
    input_throughput: float        # input (prompt) tokens / second
    output_throughput: float       # output tokens / second  (decode speed)
    mean_ttft_ms: float            # Time To First Token = prefill latency
    median_ttft_ms: float
    p99_ttft_ms: float             # the tail users actually feel under load
    mean_tpot_ms: float            # Time Per Output Token = inter-token latency (ITL)
    p99_tpot_ms: float
    # ... also p90 / p95 variants, and e2e latency</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/benchmark/serving.py ::benchmark</span><span class="ln">load generator: fire requests at a rate, collect &amp; reduce latency metrics</span></div><pre>async def benchmark(backend, api_url, base_url, model_id, tokenizer,
                    input_requests, request_rate, max_concurrency, ...):
    # the load generator: fire input_requests at the target request_rate
    # (capped by max_concurrency), record TTFT/TPOT per request, then
    # reduce to throughput + latency percentiles (BenchmarkMetrics).
    ...</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>Two-stage loop</strong>: <span class="mono">benchmark</span> quantifies "how slow," <span class="mono">profile</span> locates "why slow."</li>
<li><span class="mono">python -m sglang.bench_serving</span> fires a configurable load at a <strong>running</strong> server and produces a <span class="mono">BenchmarkMetrics</span> summary.</li>
<li><strong>Throughput</strong>: <span class="mono">request_throughput</span> / <span class="mono">input_throughput</span> / <span class="mono">output_throughput</span> (output throughput ≈ decode speed).</li>
<li><strong>Latency</strong>: <span class="mono">ttft</span> = prefill wait, <span class="mono">tpot</span> = inter-token latency (ITL) = decode smoothness.</li>
<li>Latency is a <strong>distribution</strong>: report mean / median / <strong>p99</strong> — under load the tail is the real user experience.</li>
<li><span class="mono">bench_one_batch</span> measures single-batch latency with <strong>no server</strong>, ideal for kernel-level A/B.</li>
<li>Use <span class="mono">SGLANG_TORCH_PROFILER_DIR</span> or <span class="mono">/start_profile</span>+<span class="mono">/stop_profile</span> to capture a trace, view prefill/decode, gaps, CUDA graph (Lesson 27) in <span class="mono">chrome://tracing</span>.</li>
<li>Connections: Lesson 8's throughput/latency trade-off, Lesson 4's prefill compute-bound vs decode bandwidth-bound; Lesson 55 turns to tests &amp; CI next.</li>
</ul></div>
"""}
LESSON_55 = {"zh": r"""
<p class="lead">没有测试的代码就像没有刹车的赛车——跑得越快越危险。这一课我们看 SGLang 是怎么被测试的：<strong>单元测试</strong>（不启动服务，直接验证函数与类）与<strong>端到端测试</strong>（真的拉起一个服务器再发请求），以及把它们串起来跑在 GPU 上的 <span class="mono">CI</span> 流水线。一个推理引擎每天都在被无数 PR 修改，如果没有一张严密的测试网兜底，任何一次看似无害的改动都可能悄悄破坏正确性或性能。理解了测试体系，你给 SGLang 加功能时才能心里有底，也才知道自己的改动该补哪一类测试、会在 CI 的哪个环节被检验，更能在失败时迅速判断问题出在风格、单测还是端到端这一层。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一座新装修的房子要交付验收。<strong>单元测试</strong>像是工人单独检查每一个插座：拿电笔点一下，确认有没有电——又快又便宜，不用通整栋楼的电。<strong>端到端测试</strong>则像是真的合上总闸、烧一壶水、开一次空调：把整套系统跑起来，看实际效果对不对。两种检查缺一不可：插座单测能快速定位某根线接错了，而通电跑一遍才知道整屋协同有没有问题。只验插座不通电，可能漏掉线路之间的相互干扰；只通电不验插座，出了问题又很难说清到底是哪一根线的错。</p>
<p>更妙的是验收流程里那条铁律：<strong>无论检查中途出了什么岔子，最后都必须把电闸拉下来、把工具收走</strong>。否则下一户人家来验收时，发现总闸还合着、工具乱放，整个排期就被拖垮，而且他们还会以为是自己这边出了问题，白白浪费大量排查时间。SGLang 的测试基类干的正是这件「保证善后」的事：哪怕开场布置出错，也要先把现场恢复干净，再把错误报出来。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>SGLang 是一个服务化的推理引擎，它的正确性既体现在「单个算子/函数算得对」，也体现在「整台服务器端到端答得对」。所以测试天然分成两层：底层用 Python 自带的 <span class="mono">unittest</span> 框架写断言，用 <span class="mono">pytest</span> 作为运行器；上层在 CI 里把成百上千个测试<strong>分片</strong>派发到打了标签的 GPU 机器上并行跑完。所有测试都继承一个叫 <span class="mono">CustomTestCase</span> 的基类，它用一个不起眼但关键的小技巧保证「就算开场失败，也一定收尾」，从而不会泄漏端口和 GPU 进程、连累后面的测试。把这几件事连起来看，你会发现整个体系都在围绕一个目标设计：让大规模、要真拉服务器的测试既能并行跑得快，又能在出错时干净利落、互不污染。</p>
</div>

<h2>一、两种测试：单元 vs 端到端</h2>
<p><strong>单元测试（unit test）</strong>不需要服务器。你直接 <span class="mono">import</span> 一个函数或类，构造输入，断言输出符合预期。它快、稳、便宜，所以在每一个 <span class="mono">PR</span> 上都会被 CI 跑一遍。比如一个采样函数、一个张量布局的转换、一个调度器的小逻辑，都可以这样被单独验证。因为不牵涉网络、不占显存、不拉起进程，单元测试往往在毫秒级就给出结果，开发时可以反复快速迭代。它的另一个好处是<strong>定位精准</strong>：一旦失败，你几乎能立刻锁定是哪一个函数、哪一个分支算错了，而不必在一大坨日志里大海捞针。正因为又快又准，单元测试构成了整张测试网最密的那一层，把绝大多数低级错误挡在合并之前。</p>
<p><strong>端到端测试（e2e test）</strong>需要真正的服务器。它通过一个像 <span class="mono">popen_launch_server</span> 这样的辅助函数，拉起一个真实的 <span class="mono">sglang.launch_server</span>（见<strong>第13课</strong>），向它发请求，然后检查返回的文本、或者跑一个数据集核对<strong>准确率</strong>。这类测试慢、吃 GPU，但它是唯一能证明「整套系统真的能正确服务」的方式：分词、调度、批处理、注意力内核、采样、流式返回，所有环节必须协同正确，端到端测试才会通过。单元测试保证每个零件没问题，端到端测试保证整台机器装起来还能跑——两者缺一不可，谁也替代不了谁。一个常见的误区是只写单元测试就觉得万事大吉，但很多 bug 恰恰藏在「零件之间的接缝」处：单看每个函数都对，连起来却因为约定不一致而出错，这种问题只有端到端测试才抓得住。</p>
<p>反过来，只依赖端到端测试也不行：它太慢太贵，无法在每个 PR 上覆盖所有细节，而且一旦失败，你只知道「整台服务器答错了」，却要花大力气才能定位到具体哪一行代码。所以健康的做法是金字塔形：大量便宜的单元测试打底，少量昂贵的端到端测试守住关键路径。</p>

<div class="cols"><div class="col">
<p><strong>单元测试 · 不启动服务</strong></p>
<p>直接 <span class="mono">import</span> 目标函数/类 → 构造输入 → <span class="mono">assert</span> 输出。毫秒级，每个 PR 都跑。定位精准：哪根「线」接错了一目了然。</p>
</div><div class="col">
<p><strong>端到端测试 · 启动服务</strong></p>
<p>用 <span class="mono">popen_launch_server</span> 拉起真实服务器 → 发请求 → 校验输出/准确率。慢、吃 GPU，但能证明整屋「通电」无误。</p>
</div></div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="单元测试不起服务器，直接 import 函数并断言输出，毫秒级无 GPU；端到端测试用 popen_launch_server 拉起真实服务器子进程，发 HTTP 请求校验后再拆除">
    <line x1="390" y1="20" x2="390" y2="282" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="24" y="32" style="font-weight:700;fill:var(--blue)">单元测试 · 不起服务器</text>
    <rect x="24" y="48" width="220" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="134" y="72" text-anchor="middle" class="mono" style="font-size:12px">import 函数 / 类</text>
    <line x1="134" y1="86" x2="134" y2="106" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="134,112 128,104 140,104" style="fill:var(--line)"/>
    <rect x="24" y="112" width="220" height="38" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="134" y="136" text-anchor="middle" class="mono" style="font-size:12px">assert 输出</text>
    <rect x="24" y="186" width="260" height="46" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="154" y="206" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">毫秒级 · 无 GPU</text>
    <text x="154" y="223" text-anchor="middle" style="fill:var(--teal);font-size:11px">失败时定位精准</text>
    <text x="414" y="32" style="font-weight:700;fill:var(--purple)">端到端测试 · 起一个服务器</text>
    <rect x="414" y="48" width="320" height="36" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="574" y="71" text-anchor="middle" class="mono" style="font-size:12px">popen_launch_server</text>
    <line x1="574" y1="84" x2="574" y2="100" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="574,106 568,98 580,98" style="fill:var(--line)"/>
    <rect x="414" y="106" width="320" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="574" y="129" text-anchor="middle" style="font-size:12px">真实服务器子进程</text>
    <line x1="574" y1="142" x2="574" y2="158" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="574,164 568,156 580,156" style="fill:var(--line)"/>
    <rect x="414" y="164" width="320" height="36" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="574" y="187" text-anchor="middle" style="font-size:12px">HTTP 请求 / 响应</text>
    <rect x="414" y="220" width="320" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="574" y="247" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">慢 · 吃 GPU · 跑完拆除</text>
  </svg>
  <div class="figcap"><b>图 1 · 单元测试 vs 端到端测试</b> — 单元测试直接 <span class="mono">import</span> 函数并断言输出，毫秒级、不占 GPU；端到端测试用 <span class="mono">popen_launch_server</span> 拉起真实服务器子进程，发 HTTP 请求校验后再把它拆除。</div>
</div>

<p>举两个具体例子：一个采样辅助函数的单元测试只需 <span class="mono">import</span> 它、喂一组 logits、断言挑出的 token 对不对，整个过程在毫秒级完成、完全不碰 GPU；而一个端到端准确率测试会先用 <span class="mono">popen_launch_server</span> 拉起服务器，再在 <span class="mono">GSM8K</span> 数据集上跑一遍，断言得分高于某个阈值（比如 0.8）才算通过。下面这个辅助函数就是端到端测试的「起服务器」入口：</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/test/test_utils.py ::popen_launch_server</span><span class="ln">起一个真实服务器子进程，轮询 /health_generate 直到就绪</span></div><pre>def popen_launch_server(model, base_url, timeout, api_key=None,
                        other_args=None, ...):
    # spawn a REAL server subprocess:
    #   sglang serve --model-path model ...other_args
    # then poll /health_generate until ready (or raise on `timeout`). e2e tests
    # hit this server over HTTP, then tear it down.
    ...
    return process</pre></div>

<h2>二、基类 CustomTestCase：保证善后</h2>
<p>SGLang 的测试不直接继承裸的 <span class="mono">unittest.TestCase</span>，而是继承 <span class="mono">CustomTestCase</span>。它做的事看似微小却很关键：<strong>包裹 <span class="mono">setUpClass</span>，使得即便 <span class="mono">setUpClass</span> 抛异常，<span class="mono">tearDownClass</span> 也一定会被调用</strong>。这里要先理解 <span class="mono">setUpClass</span> 与 <span class="mono">tearDownClass</span> 是一对：前者在一个测试类里所有用例开始前跑一次（往往就是在这里拉起服务器、占用端口、申请显存），后者在所有用例结束后跑一次（负责把这些资源全部释放掉）。</p>
<p>为什么重要？原生 <span class="mono">unittest</span> 的行为是：如果 <span class="mono">setUpClass</span> 失败，它会<strong>跳过</strong> <span class="mono">tearDownClass</span>。在一个会启动服务器的测试套件里，这意味着已经占用的<strong>端口</strong>和拉起的 <span class="mono">GPU</span> 进程不会被回收——于是端口被占、显存被占，后面的测试想再拉一台服务器时，要么端口冲突起不来，要么显存不足直接崩，最终<strong>连环失败</strong>。更糟的是，这种失败往往看起来和真正出问题的那个测试毫无关系，排查起来极其费神。<span class="mono">CustomTestCase</span> 用 <span class="mono">__init_subclass__</span> 在每个子类定义时自动把 <span class="mono">setUpClass</span> 包一层 <span class="mono">try/except</span>，异常时先调用 <span class="mono">tearDownClass</span> 清理、再把异常重新抛出，从而保证清理一定发生。这让「一个坏掉的 setup 不会毒害整套测试」，正是上面类比里那条「无论如何都要拉闸收工」的铁律。</p>
<p>这里有个值得细品的设计选择：为什么用 <span class="mono">__init_subclass__</span> 而不是让每个测试自己记得写 <span class="mono">try/finally</span>？因为靠人自觉是最不可靠的——总会有人忘记。把这层保护放进基类，意味着只要你继承了 <span class="mono">CustomTestCase</span>，就<strong>自动</strong>获得这份健壮性，不需要任何额外代码，也不可能漏写。这是一种「让正确的事情默认发生」的工程智慧：与其在评审里反复提醒，不如把规则固化进基础设施，从根上杜绝这类资源泄漏。注意它在 <span class="mono">except</span> 里调用完 <span class="mono">tearDownClass</span> 后用裸 <span class="mono">raise</span> 把原异常原样抛出，所以测试报告里看到的仍是真正的失败原因，清理动作是「悄悄」补上的，不会掩盖问题。</p>
<p>顺带一提，这种「在套件级别启动一次、结束清理一次」的设计，正是为了摊薄启动成本：拉起一台服务器动辄要几十秒加载权重，如果每个测试方法都各拉一次，时间根本承受不起。所以同一个测试类里的多个 <span class="mono">test_</span> 方法会<strong>共享</strong>一台由 <span class="mono">setUpClass</span> 拉起的服务器，跑完所有方法再统一在 <span class="mono">tearDownClass</span> 里关闭。这也反过来解释了为什么 <span class="mono">tearDownClass</span> 的可靠执行如此关键——它管理着一个被全类共享、又格外昂贵的资源，一旦泄漏，代价远比一个普通对象大得多。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/test/test_utils.py ::CustomTestCase</span><span class="ln">包裹 setUpClass，保证失败时也跑 tearDownClass，不泄漏端口/进程</span></div><pre>class CustomTestCase(unittest.TestCase):
    # base class for SGLang tests
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # wrap setUpClass so tearDownClass is ALWAYS called, even if setUpClass
        # raises. Plain unittest SKIPS tearDownClass on setup failure, which would
        # leak ports / GPU processes and cascade-fail later tests.
        setup = cls.setUpClass
        @classmethod
        def safe_setUpClass(klass):
            try:
                setup.__func__(klass)
            except Exception:
                klass.tearDownClass()   # clean up before re-raising
                raise
        cls.setUpClass = safe_setUpClass</pre></div>

<h2>三、CI 流水线：分片、并行、门禁</h2>
<p>测试登记在 <span class="mono">test/registered/</span> 目录下，CI 会把它们<strong>分片（partition）</strong>派发到多台打了标签的 <span class="mono">GPU</span> 运行机上并行执行，这样庞大的套件也能在合理时间里跑完。试想如果几百个端到端测试都挤在一台机器上一个接一个地跑，光是反复拉起、关闭服务器就要花上几个小时；而分片之后，每台带特定标签（标明 GPU 型号、数量）的机器只认领属于自己的那一片，整体耗时被压到可以接受的范围。在 CI 真正开跑之前，<span class="mono">pre-commit</span> 会先在本地和 CI 上把<strong>代码风格与 lint</strong> 卡一道关——风格不过，后面都免谈，这样能在最便宜的阶段就拦下格式问题，不浪费宝贵的 GPU 机时。</p>
<p>这套「先便宜后昂贵」的分层把关其实是 CI 设计的通用原则：越靠前的关卡越快越省，越靠后的关卡越慢越贵。<span class="mono">pre-commit</span> 几乎瞬间就能在本地跑完，能在你提交之前就揪出缩进、空行、import 顺序之类的小毛病；接着是单元测试这层，秒级到分钟级；最后才是要真拉服务器的端到端测试，可能要占用多卡跑上十几分钟。把关卡按成本从低到高排列，意味着大多数问题都在最省钱的环节被挡下，只有真正通过了前面所有检查的改动，才值得动用昂贵的 GPU 集群去做最终验证。这也是为什么强烈建议你在本地就装好 <span class="mono">pre-commit</span> 钩子：与其推上去等 CI 几分钟后告诉你格式错了，不如在 <span class="mono">git commit</span> 的一瞬间就自动修好。</p>
<p>一个贡献者的典型工作流是：在 <span class="mono">test/registered/unit/…</span> 下<strong>找到或新增</strong>对应的测试，先用 <span class="mono">pytest</span> 在本地跑通，然后推送；CI 会在真实 GPU 上把它重新跑一遍。知道这条链路，你就能为自己的改动<strong>自信地补上覆盖</strong>：改了采样逻辑就加个采样的单元测试，改了某个模型的支持就加个端到端的准确率测试。这把第13课（端到端测试拉起的那台服务器）和<strong>第56课</strong>（真正运行这些测试的 PR 流程）串了起来。项目里还有一个专门写 SGLang 测试的约定/技能可以参考，照着它的模板写，能少踩很多坑、更容易通过评审。</p>
<p>这里「打标签的运行机」是关键一环：不同测试对硬件的要求不同——有的只需一张消费级显卡，有的要多卡做张量并行，有的要特定型号才能复现某个内核的行为。CI 通过标签把每一片测试精确地投递到符合要求的机器上，既不会让小测试白占大机器，也不会让大测试被分到带不动的机器上。理解了这套分片加标签的调度，你就能明白为什么有时你的 PR 要排队等某类 GPU 空闲，也能在写测试时主动声明它真正需要的资源，避免浪费集群算力、拖慢所有人的 CI。</p>

<div class="flow"><div class="node">PR push</div><div class="arrow">→</div><div class="node">pre-commit 风格/lint</div><div class="arrow">→</div><div class="node">CI 分片 · GPU 运行机</div><div class="arrow">→</div><div class="node">合并门禁 merge</div></div>

<div class="fig">
  <svg viewBox="0 0 820 280" role="img" aria-label="PR 推送先过 pre-commit 风格与 lint，再进入 CI 把测试分桶并行跑在 GPU runners 上（单元、端到端、准确率三片），全绿才进入合并">
    <rect x="20" y="112" width="120" height="48" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="80" y="141" text-anchor="middle" style="font-weight:700;font-size:12px">PR 推送</text>
    <line x1="140" y1="136" x2="168" y2="136" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="174,136 166,131 166,141" style="fill:var(--line)"/>
    <rect x="174" y="112" width="146" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="247" y="133" text-anchor="middle" style="font-weight:700;font-size:12px">pre-commit</text>
    <text x="247" y="150" text-anchor="middle" style="font-size:11px">lint / 格式</text>
    <line x1="320" y1="136" x2="348" y2="136" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="354,136 346,131 346,141" style="fill:var(--line)"/>
    <rect x="354" y="40" width="286" height="200" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="497" y="62" text-anchor="middle" style="fill:var(--muted);font-weight:700;font-size:12px">CI 分桶并行 · GPU runners</text>
    <rect x="376" y="78" width="242" height="42" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="497" y="104" text-anchor="middle" style="font-size:12px">单元 · 分片 1</text>
    <rect x="376" y="128" width="242" height="42" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="497" y="154" text-anchor="middle" style="font-size:12px">端到端 · 分片 2</text>
    <rect x="376" y="178" width="242" height="42" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="497" y="204" text-anchor="middle" style="font-size:12px">准确率 · 分片 3</text>
    <line x1="640" y1="140" x2="668" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="674,140 666,135 666,145" style="fill:var(--line)"/>
    <rect x="674" y="116" width="130" height="48" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="739" y="139" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">合并 merge</text>
    <text x="739" y="156" text-anchor="middle" style="fill:var(--teal);font-size:11px">全绿才合并</text>
  </svg>
  <div class="figcap"><b>图 2 · CI 流水线</b> — 推上去的 PR 先过 <span class="mono">pre-commit</span>（lint/格式），再进入 CI：把测试<strong>分桶</strong>派到多台 GPU runner 上并行跑（单元、端到端、准确率三片），全部变绿才进入合并门禁。</div>
</div>

<p>正因为有分桶，CI 才能把成百上千个测试文件拆散、分到多台 runner 上同时跑，让整个套件在并行下几分钟跑完，而不是串行排上几个小时。</p>

<h2>四、把工具对号入座</h2>
<p>记住几个关键辅助件各自的角色，写测试时就不会乱。它们分工明确：基类管「善后」，启动器管「拉起服务」，运行器管「发现并执行」。理清这三者，你就能看懂任意一个 SGLang 测试文件的骨架。</p>

<table class="t"><tr><th>测试辅助件</th><th>角色</th></tr>
<tr><td><span class="mono">CustomTestCase</span></td><td>测试基类：包裹 setUpClass，保证失败也收尾，不泄漏端口/GPU 进程</td></tr>
<tr><td><span class="mono">popen_launch_server</span></td><td>端到端测试用它拉起真实 sglang.launch_server，再发请求校验</td></tr>
<tr><td><span class="mono">pytest</span></td><td>运行器：发现并执行 unittest 写的用例，本地与 CI 共用</td></tr>
</table>

<p>把它落到动作上，一次完整的「写测试到合并」是这样一条直线：先在登记目录里写好测试并继承 <span class="mono">CustomTestCase</span>，本地用 <span class="mono">pytest</span> 反复跑到通过，推送上 PR 触发 <span class="mono">pre-commit</span> 与 CI，最后 CI 在 GPU 上分片重跑、全绿即可进入合并门禁。当你下次打开任意一个 SGLang 测试文件，就能一眼读懂它的骨架：开头继承 <span class="mono">CustomTestCase</span>，<span class="mono">setUpClass</span> 里用 <span class="mono">popen_launch_server</span> 拉起服务器并记下进程句柄，若干个 <span class="mono">test_</span> 方法发请求并断言输出或准确率，<span class="mono">tearDownClass</span> 里把服务器关掉。理解了这套约定，你既能看懂别人的测试，也能照葫芦画瓢为自己的改动写出符合规范、稳定可靠的覆盖，让 CI 这张安全网真正为你的代码兜底。</p>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>写一个测试</h4><p class="mono">test/registered/unit/…</p><p>在对应目录找到或新增测试，继承 CustomTestCase。</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>本地跑通</h4><p class="mono">pytest</p><p>用 pytest 在本地运行，快速迭代到通过。</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>push 到 PR</h4><p class="mono">git push</p><p>推送后触发 CI，pre-commit 先卡风格/lint。</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>CI 变绿</h4><p class="mono">GPU runners</p><p>CI 在真实 GPU 上分片重跑，全绿即可合并。</p></div></div></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>两类测试</strong>：单元测试不启动服务，直接 import 后断言，快、每个 PR 都跑；端到端测试用 <span class="mono">popen_launch_server</span> 拉起真实服务器再校验输出/准确率。</li>
<li>SGLang 用 Python 自带的 <span class="mono">unittest</span> 框架写断言，用 <span class="mono">pytest</span> 作运行器。</li>
<li><strong>所有测试继承 <span class="mono">CustomTestCase</span></strong>，它包裹 <span class="mono">setUpClass</span>，保证即便 setup 失败 <span class="mono">tearDownClass</span> 也一定执行——避免泄漏端口/GPU 进程、连累后面的测试。</li>
<li><strong>CI</strong> 把 <span class="mono">test/registered/</span> 下的测试<strong>分片</strong>派发到打标签的 GPU 机并行跑；<span class="mono">pre-commit</span> 先卡风格/lint。</li>
<li>贡献者工作流：在 <span class="mono">test/registered/unit/…</span> 找/加测试 → 本地 <span class="mono">pytest</span> → push → CI 真机重跑。串起第13课（服务器）与第56课（PR 流程）。</li>
<li>记住测试金字塔：大量便宜的单元测试打底，少量昂贵的端到端测试守关键路径；前者抓零件 bug，后者抓接缝处的协同 bug。</li>
<li>分层把关原则：越靠前的关卡越快越省（pre-commit），越靠后越慢越贵（GPU 端到端）；本地装好 pre-commit 钩子能在提交瞬间就修好格式，省下排队等 CI 的时间。</li>
</ul></div>
""", "en": r"""
<p class="lead">Code without tests is like a race car without brakes—the faster it goes, the more dangerous it gets. This lesson looks at how SGLang is tested: <strong>unit tests</strong> (no server, assert directly on functions and classes) and <strong>end-to-end tests</strong> (actually launch a server and send requests), plus the <span class="mono">CI</span> pipeline that runs them all on GPUs. Once you understand the test system, you can add features to SGLang with confidence.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a freshly renovated house going through final inspection. A <strong>unit test</strong> is like an electrician checking each outlet on its own: touch a test pen to it and confirm there's current—fast and cheap, no need to power the whole building. An <strong>end-to-end test</strong> is like actually flipping the main breaker, boiling a kettle, and running the AC: bring the whole system up and see if it really works. You need both: the outlet unit test pinpoints which wire was miswired, while powering everything up reveals whether the whole house cooperates.</p>
<p>Even better is the iron rule of the inspection process: <strong>no matter what goes wrong mid-check, you must flip the breaker back off and pack up your tools</strong>. Otherwise the next household's inspection finds the main breaker still on and tools scattered, and the whole schedule is derailed. SGLang's test base class does exactly this "guarantee cleanup" job.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>SGLang is a server-style inference engine, and its correctness shows up both in "a single operator/function computes right" and in "the whole server answers right end to end." So testing naturally splits into two layers: the lower layer writes assertions with Python's built-in <span class="mono">unittest</span> framework, using <span class="mono">pytest</span> as the runner; the upper layer, in CI, <strong>partitions</strong> hundreds of tests across labelled GPU machines to run in parallel. Every test inherits a base class called <span class="mono">CustomTestCase</span>, which uses an inconspicuous but crucial trick to guarantee "even if setup fails, teardown still runs," so it never leaks ports or GPU processes and never cascade-fails later tests.</p>
</div>

<h2>1. Two kinds of tests: unit vs end-to-end</h2>
<p>A <strong>unit test</strong> needs no server. You directly <span class="mono">import</span> a function or class, build inputs, and assert the output matches expectations. It's fast, stable, and cheap, so CI runs it on every <span class="mono">PR</span>. A sampling function, a tensor-layout conversion, a small bit of scheduler logic—all can be verified in isolation this way.</p>
<p>An <strong>e2e test</strong> needs a real server. Through a helper like <span class="mono">popen_launch_server</span>, it launches a real <span class="mono">sglang.launch_server</span> (see <strong>Lesson 13</strong>), sends requests, then checks the returned text—or runs a dataset to verify <strong>accuracy</strong>. These tests are slow and GPU-hungry, but they're the only way to prove "the whole system really serves correctly."</p>

<div class="cols"><div class="col">
<p><strong>Unit test · no server</strong></p>
<p>Directly <span class="mono">import</span> the target function/class → build inputs → <span class="mono">assert</span> output. Millisecond-scale, runs on every PR. Pinpoints precisely which "wire" was miswired.</p>
</div><div class="col">
<p><strong>E2E test · launch a server</strong></p>
<p>Use <span class="mono">popen_launch_server</span> to bring up a real server → send requests → verify outputs/accuracy. Slow and GPU-hungry, but proves the whole house is "powered" correctly.</p>
</div></div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="A unit test launches no server: it imports a function and asserts its output, millisecond-scale with no GPU; an e2e test uses popen_launch_server to spawn a real server subprocess, sends HTTP requests to verify, then tears it down">
    <line x1="390" y1="20" x2="390" y2="282" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="24" y="32" style="font-weight:700;fill:var(--blue)">Unit test · no server</text>
    <rect x="24" y="48" width="240" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="144" y="72" text-anchor="middle" class="mono" style="font-size:12px">import function / class</text>
    <line x1="144" y1="86" x2="144" y2="106" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="144,112 138,104 150,104" style="fill:var(--line)"/>
    <rect x="24" y="112" width="240" height="38" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="144" y="136" text-anchor="middle" class="mono" style="font-size:12px">assert output</text>
    <rect x="24" y="186" width="280" height="46" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="164" y="206" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">ms-scale · no GPU</text>
    <text x="164" y="223" text-anchor="middle" style="fill:var(--teal);font-size:11px">precise on failure</text>
    <text x="414" y="32" style="font-weight:700;fill:var(--purple)">E2E test · launches a server</text>
    <rect x="414" y="48" width="320" height="36" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="574" y="71" text-anchor="middle" class="mono" style="font-size:12px">popen_launch_server</text>
    <line x1="574" y1="84" x2="574" y2="100" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="574,106 568,98 580,98" style="fill:var(--line)"/>
    <rect x="414" y="106" width="320" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="574" y="129" text-anchor="middle" style="font-size:12px">real server subprocess</text>
    <line x1="574" y1="142" x2="574" y2="158" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="574,164 568,156 580,156" style="fill:var(--line)"/>
    <rect x="414" y="164" width="320" height="36" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="574" y="187" text-anchor="middle" style="font-size:12px">HTTP request / response</text>
    <rect x="414" y="220" width="320" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="574" y="247" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">slow · GPU · torn down</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Unit test vs e2e test</b> — a unit test just <span class="mono">import</span>s the function and asserts its output—millisecond-scale, no GPU; an e2e test uses <span class="mono">popen_launch_server</span> to spawn a real server subprocess, sends HTTP requests to verify, then tears it down.</div>
</div>

<p>Two concrete examples: a unit test for a sampling helper just <span class="mono">import</span>s it, feeds a set of logits, and asserts the chosen token is right—all in milliseconds, never touching a GPU; an e2e accuracy test instead calls <span class="mono">popen_launch_server</span> to bring up a server, runs the <span class="mono">GSM8K</span> dataset through it, and passes only if the score is above a threshold (say 0.8). The helper below is exactly that "launch a server" entry point for e2e tests:</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/test/test_utils.py ::popen_launch_server</span><span class="ln">spawn a real server subprocess, poll /health_generate until ready</span></div><pre>def popen_launch_server(model, base_url, timeout, api_key=None,
                        other_args=None, ...):
    # spawn a REAL server subprocess:
    #   sglang serve --model-path model ...other_args
    # then poll /health_generate until ready (or raise on `timeout`). e2e tests
    # hit this server over HTTP, then tear it down.
    ...
    return process</pre></div>

<h2>2. The base class CustomTestCase: guarantee cleanup</h2>
<p>SGLang tests don't inherit bare <span class="mono">unittest.TestCase</span>; they inherit <span class="mono">CustomTestCase</span>. What it does looks tiny but matters a lot: it <strong>wraps <span class="mono">setUpClass</span> so that even if <span class="mono">setUpClass</span> raises, <span class="mono">tearDownClass</span> is still called</strong>.</p>
<p>Why does this matter? Plain <span class="mono">unittest</span> behaves like this: if <span class="mono">setUpClass</span> fails, it <strong>skips</strong> <span class="mono">tearDownClass</span>. In a suite that launches servers, that means the <strong>ports</strong> already taken and the <span class="mono">GPU</span> processes already spawned won't be reclaimed—so ports stay occupied, VRAM stays occupied, and later tests cascade-fail. <span class="mono">CustomTestCase</span> guarantees cleanup happens, so "one broken setup doesn't poison the whole suite." This is exactly the "flip the breaker and pack up no matter what" rule from the analogy.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/test/test_utils.py ::CustomTestCase</span><span class="ln">wraps setUpClass so tearDownClass always runs, leaking no ports / processes</span></div><pre>class CustomTestCase(unittest.TestCase):
    # base class for SGLang tests
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # wrap setUpClass so tearDownClass is ALWAYS called, even if setUpClass
        # raises. Plain unittest SKIPS tearDownClass on setup failure, which would
        # leak ports / GPU processes and cascade-fail later tests.
        setup = cls.setUpClass
        @classmethod
        def safe_setUpClass(klass):
            try:
                setup.__func__(klass)
            except Exception:
                klass.tearDownClass()   # clean up before re-raising
                raise
        cls.setUpClass = safe_setUpClass</pre></div>

<h2>3. The CI pipeline: partitioning, parallelism, gating</h2>
<p>Tests are registered under <span class="mono">test/registered/</span>, and CI <strong>partitions</strong> them across multiple labelled <span class="mono">GPU</span> runners to execute in parallel, so even a huge suite finishes in reasonable time. Before CI even starts, <span class="mono">pre-commit</span> first gates <strong>code style and lint</strong> locally and in CI—if style doesn't pass, nothing else matters.</p>
<p>A contributor's typical workflow: <strong>find or add</strong> the matching test under <span class="mono">test/registered/unit/…</span>, run it locally with <span class="mono">pytest</span> until it passes, then push; CI re-runs it on real GPUs. Knowing this chain lets you <strong>confidently add coverage</strong> for your change. This ties Lesson 13 (the server those e2e tests launch) to <strong>Lesson 56</strong> (the PR flow that actually runs these tests). There's also a dedicated project convention/skill for writing SGLang tests to reference.</p>

<div class="flow"><div class="node">PR push</div><div class="arrow">→</div><div class="node">pre-commit style/lint</div><div class="arrow">→</div><div class="node">CI partitioned · GPU runners</div><div class="arrow">→</div><div class="node">merge gate</div></div>

<div class="fig">
  <svg viewBox="0 0 820 280" role="img" aria-label="A pushed PR first clears pre-commit lint and format, then enters CI which partitions tests across GPU runners (unit, e2e, accuracy shards) running in parallel; only all-green reaches merge">
    <rect x="20" y="112" width="120" height="48" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="80" y="141" text-anchor="middle" style="font-weight:700;font-size:12px">PR push</text>
    <line x1="140" y1="136" x2="168" y2="136" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="174,136 166,131 166,141" style="fill:var(--line)"/>
    <rect x="174" y="112" width="146" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="247" y="133" text-anchor="middle" style="font-weight:700;font-size:12px">pre-commit</text>
    <text x="247" y="150" text-anchor="middle" style="font-size:11px">lint / format</text>
    <line x1="320" y1="136" x2="348" y2="136" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="354,136 346,131 346,141" style="fill:var(--line)"/>
    <rect x="354" y="40" width="286" height="200" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="497" y="62" text-anchor="middle" style="fill:var(--muted);font-weight:700;font-size:12px">CI partitioned · GPU runners</text>
    <rect x="376" y="78" width="242" height="42" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="497" y="104" text-anchor="middle" style="font-size:12px">unit · shard 1</text>
    <rect x="376" y="128" width="242" height="42" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="497" y="154" text-anchor="middle" style="font-size:12px">e2e · shard 2</text>
    <rect x="376" y="178" width="242" height="42" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="497" y="204" text-anchor="middle" style="font-size:12px">accuracy · shard 3</text>
    <line x1="640" y1="140" x2="668" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="674,140 666,135 666,145" style="fill:var(--line)"/>
    <rect x="674" y="116" width="130" height="48" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="739" y="139" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">merge</text>
    <text x="739" y="156" text-anchor="middle" style="fill:var(--teal);font-size:11px">all green to merge</text>
  </svg>
  <div class="figcap"><b>Fig 2 · CI pipeline</b> — a pushed PR first clears <span class="mono">pre-commit</span> (lint/format), then enters CI: it <strong>partitions</strong> the tests across many GPU runners to run in parallel (unit, e2e, accuracy shards), and only when all turn green does it reach the merge gate.</div>
</div>

<p>Partitioning is exactly why CI can split hundreds of test files across many runners and run them at once, finishing the whole suite in a few parallel minutes instead of hours of serial launches.</p>

<h2>4. Matching the tools to their roles</h2>
<p>Remember the role of each key helper and you won't get lost when writing tests:</p>

<table class="t"><tr><th>Test helper</th><th>Role</th></tr>
<tr><td><span class="mono">CustomTestCase</span></td><td>Test base class: wraps setUpClass, guarantees cleanup on failure, leaks no ports / GPU processes</td></tr>
<tr><td><span class="mono">popen_launch_server</span></td><td>E2E tests use it to launch a real sglang.launch_server, then send requests and verify</td></tr>
<tr><td><span class="mono">pytest</span></td><td>Runner: discovers and executes unittest-written cases, shared by local and CI</td></tr>
</table>

<p>Put into actions, one complete "write a test to merge" is a straight line:</p>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>write a test</h4><p class="mono">test/registered/unit/…</p><p>Find or add a test in the right directory, inheriting CustomTestCase.</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>run locally</h4><p class="mono">pytest</p><p>Run it locally with pytest, iterating fast until it passes.</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>push to PR</h4><p class="mono">git push</p><p>Pushing triggers CI; pre-commit gates style/lint first.</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>CI turns green</h4><p class="mono">GPU runners</p><p>CI re-runs partitioned on real GPUs; all green means merge-ready.</p></div></div></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>Two kinds of tests</strong>: unit tests launch no server, just import and assert—fast, run on every PR; e2e tests use <span class="mono">popen_launch_server</span> to launch a real server, then verify outputs/accuracy.</li>
<li>SGLang writes assertions with Python's built-in <span class="mono">unittest</span> framework and uses <span class="mono">pytest</span> as the runner.</li>
<li><strong>All tests inherit <span class="mono">CustomTestCase</span></strong>, which wraps <span class="mono">setUpClass</span> so that even if setup fails, <span class="mono">tearDownClass</span> still runs—avoiding leaked ports/GPU processes and cascade-failing later tests.</li>
<li><strong>CI</strong> <strong>partitions</strong> the tests under <span class="mono">test/registered/</span> across labelled GPU machines to run in parallel; <span class="mono">pre-commit</span> gates style/lint first.</li>
<li>Contributor workflow: find/add a test under <span class="mono">test/registered/unit/…</span> → local <span class="mono">pytest</span> → push → CI re-runs on real hardware. Ties Lesson 13 (the server) to Lesson 56 (the PR flow).</li>
<li>Remember the test pyramid: lots of cheap unit tests at the base, a few expensive e2e tests guarding the critical path; unit tests catch part-level bugs, e2e catches the seam/integration bugs.</li>
<li>Layered gating: earlier gates are cheaper and faster (pre-commit), later ones slower and costlier (GPU e2e); installing the pre-commit hook locally fixes formatting the moment you commit, saving the wait in the CI queue.</li>
</ul></div>
"""}
LESSON_56 = {"zh": r"""
<p class="lead">写出能跑的代码只是第一步；想把它合并进 SGLang 主仓库，你还得让代码<strong>符合规范</strong>、<strong>通过检查</strong>、<strong>带上测试</strong>，再用一个清晰的 <span class="mono">Pull Request</span> 把它交给维护者审阅。本课讲清楚从 fork 到 PR 合并的完整流程，以及背后那些"约定俗成"的规矩。把这套流程走顺，你的第一个贡献就能<strong>少走很多弯路</strong>，也能让维护者更愿意花时间看你的代码。流程本身并不复杂，难的是把每一步都做到位、形成习惯。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把贡献代码想象成<strong>给一本开源教科书投稿</strong>。你不能直接在出版社的母版上涂改（新贡献者没有官方仓库的写权限），所以你先<strong>复印一份属于自己的副本</strong>（fork），在副本上单开一张稿纸写你的章节（branch），写完后用<strong>统一的排版工具</strong>把字体、缩进、标点都规整一遍（<span class="mono">pre-commit</span>），附上一道<strong>自测题</strong>证明内容正确（unit test），最后把稿子<strong>寄给编辑部</strong>请求采纳（open a PR）。编辑（CI 与维护者）会用和你一样的排版工具复查一遍——如果你没排版，稿子直接被退回。</p>
<p>这个类比里还有两层值得记住的道理。其一，<strong>排版工具不是来刁难你的</strong>：它把所有投稿都规整成同一种样式，编辑才不会被五花八门的格式分散注意力，可以专心看你的<strong>内容</strong>。其二，<strong>一篇好稿子是小而完整的</strong>：与其一次塞给编辑一本厚厚的合集，不如每次只投一个章节，附上清楚的"我为什么写这一章"，编辑读得快、改得也快。代码贡献完全是同一回事——<strong>格式自动化、改动小而聚焦、动机讲清楚</strong>，你的"稿子"就更容易被采纳。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>这份指南反复强调一句话：<strong>SGLang 里几乎一切都是可插拔的</strong>。正因为如此，贡献门槛其实很低——大多数新功能都能<strong>嵌进一条已有的接缝</strong>：加一个新后端（注意力后端 第33课）、加一个新模型文件（第26课）、加一个新处理器（第49课），都不需要改动引擎核心。换句话说，你不是在重写发动机，而是在一个设计好的插槽里安一个新零件。规范流程的作用，就是保证这个新零件装上去之后，整台机器仍然干净、可测、可维护。</p>
<p>从更高的视角看，<strong>规范流程和可插拔架构其实是一体两面</strong>。架构提供了清晰的接缝，让你知道"新东西该装在哪"；规范流程则提供了清晰的交付路径，让你知道"装好之后怎么安全地交出去"。两者配合，才让一个有成百上千贡献者的项目能够<strong>持续、快速、可靠地演进</strong>而不崩塌。本课正是把前面五十多课学到的"怎么造零件"，收束成"怎么把零件交付到主干"——这是你成为 SGLang 贡献者的<strong>最后一公里</strong>。</p>
</div>

<h2>整条流程：从 fork 到 PR</h2>
<p>新贡献者<strong>不能直接向官方仓库 push</strong>，所以第一步永远是 <strong>fork</strong>——在 GitHub 上点一下，把 <span class="mono">sgl-project/sglang</span> 复制成 <span class="mono">&lt;你的用户名&gt;/sglang</span>。接着把<strong>你自己的 fork</strong> clone 到本地，<span class="mono">git checkout -b my-feature</span> 开一条分支，在分支上做改动。改完之后不要急着 push：先用 <span class="mono">pre-commit</span> 把格式和 lint 过一遍，再按 <strong>第55课</strong> 的方法补一个单元测试并在本地跑通，确认通过后才 <span class="mono">git push origin my-feature</span>，最后在 GitHub 上对着 <span class="mono">main</span> 分支开一个 <strong>Pull Request</strong>。整条链路是：<strong>fork → clone → branch → change → pre-commit → test → push → PR</strong>，每一环都不能跳。</p>
<p>为什么要在自己的分支上工作，而不是直接在 <span class="mono">main</span> 上改？因为<strong>分支让每一项工作彼此隔离</strong>。一个分支只承载一个功能或一处修复，万一这个改动有问题，删掉分支即可，不会污染你 fork 上干净的 <span class="mono">main</span>。这也方便你<strong>同时推进多个独立的贡献</strong>：每个想法一条分支，互不干扰。分支名最好能<strong>说明意图</strong>，例如 <span class="mono">fix-sampling-overflow</span> 或 <span class="mono">add-glm-model</span>，让 reviewer 一看名字就大概知道你在做什么。</p>
<p>开 PR 时还有一个常被忽略的细节：要让 PR 的<strong>目标分支是官方仓库的 <span class="mono">main</span></strong>，源分支是你 fork 上的 <span class="mono">my-feature</span>。GitHub 会自动比较两边的差异，把你的提交集中展示给维护者。从这一刻起，你的代码就进入了<strong>公开评审</strong>：CI 会自动触发，维护者会留下评论，你可能需要根据反馈在同一条分支上继续 commit、再 push，PR 会自动更新。这个"提交—反馈—修订"的循环，正是开源协作的核心。</p>

<div class="fig">
  <svg viewBox="0 0 800 240" role="img" aria-label="贡献流程：fork 复制仓库 → 开分支 → pre-commit → 跑测试 → 推送 → 开 PR，再由 CI 与审阅者把关合并">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">贡献者路径</text>
    <rect x="24" y="64" width="104" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="76" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">Fork</text>
    <text x="76" y="108" text-anchor="middle" style="font-size:11px;fill:var(--muted)">复制仓库</text>
    <rect x="148" y="64" width="104" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="200" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">分支</text>
    <text x="200" y="108" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">git branch</text>
    <rect x="272" y="64" width="104" height="58" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="324" y="90" text-anchor="middle" class="mono" style="font-size:12px;font-weight:700">pre-commit</text>
    <text x="324" y="108" text-anchor="middle" style="font-size:11px;fill:var(--muted)">自动格式化</text>
    <rect x="396" y="64" width="104" height="58" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="448" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">测试</text>
    <text x="448" y="108" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">pytest</text>
    <rect x="520" y="64" width="104" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="572" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">推送</text>
    <text x="572" y="108" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">git push</text>
    <rect x="644" y="64" width="104" height="58" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="696" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">开 PR</text>
    <text x="696" y="108" text-anchor="middle" style="font-size:11px;fill:var(--muted)">合并请求</text>
    <line x1="130" y1="93" x2="146" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="140,89 146,93 140,97" style="fill:var(--faint)"/>
    <line x1="254" y1="93" x2="270" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="264,89 270,93 264,97" style="fill:var(--faint)"/>
    <line x1="378" y1="93" x2="394" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="388,89 394,93 388,97" style="fill:var(--faint)"/>
    <line x1="502" y1="93" x2="518" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="512,89 518,93 512,97" style="fill:var(--faint)"/>
    <line x1="626" y1="93" x2="642" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="636,89 642,93 636,97" style="fill:var(--faint)"/>
    <line x1="696" y1="122" x2="696" y2="156" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="692,150 696,158 700,150" style="fill:var(--faint)"/>
    <rect x="24" y="158" width="724" height="54" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="386" y="181" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--amber)">CI + 审阅者把关</text>
    <text x="386" y="200" text-anchor="middle" style="font-size:12px;fill:var(--muted)">通过后才能合并进 main</text>
  </svg>
  <div class="figcap"><b>图 1 · 贡献流程</b> — fork 复制仓库 → 开分支 → <span class="mono">pre-commit</span> 自动格式化/lint → 跑测试 → push → 开 PR；最后由 CI 与审阅者一起把关，通过后才合并进 <span class="mono">main</span>。</div>
</div>

<h2>pre-commit：风格与 lint 的关卡</h2>
<p><span class="mono">pre-commit</span> 是 SGLang 的<strong>统一格式化与静态检查关卡</strong>。三条命令就能用起来：<span class="mono">pip3 install pre-commit</span> 装工具，<span class="mono">pre-commit install</span> 把它<strong>注册成 git 钩子</strong>（这样你每次 commit 时它会自动跑），<span class="mono">pre-commit run --all-files</span> 对所有文件跑一遍配置好的检查——black、isort 等等——并<strong>自动应用能修的改动</strong>。这里有个关键细节：<strong>第一次跑很可能失败</strong>，因为它会就地改写你的文件（重排 import、补空行、统一引号）。这不是错误，而是它在帮你修。你要做的是<strong>把这些自动改动 add 进来，然后再跑一次</strong>，<strong>反复重跑直到全部 PASS</strong> 为止。必须在开 PR 之前就让它全绿，因为 CI（第55课）会跑<strong>完全相同</strong>的检查，你本地不过，CI 也不会过，PR 会被卡住无法合并。</p>
<p>为什么要把这些琐碎的格式问题交给工具，而不是靠人去盯？因为<strong>统一的代码风格是大型协作项目的基础设施</strong>。当成百上千个贡献者往同一个仓库里提交代码时，如果每个人的缩进、引号、import 顺序都不一样，diff 里就会塞满与逻辑无关的"噪音"，reviewer 很难分辨哪些是真正的改动。<span class="mono">pre-commit</span> 把"风格"这件事彻底<strong>自动化、机械化</strong>，让人类的注意力可以集中在<strong>逻辑和设计</strong>上。这也是为什么它被设成一道硬关卡：风格不过，后面的讨论都无从谈起。</p>
<p>把它<strong>注册成 git 钩子</strong>（<span class="mono">pre-commit install</span>）还有一层好处：从此每次 <span class="mono">git commit</span> 它都会自动在改动的文件上跑一遍，<strong>在问题进入历史之前就拦下来</strong>。你不必记得手动运行，也不会因为忘了格式化而在 push 之后才被 CI 打回。换句话说，本地钩子和远端 CI 跑的是同一套检查，只是<strong>把关卡前移</strong>到了你的机器上，让反馈来得更早、更便宜。</p>
<p>实践中有一个高频踩坑点值得专门提醒：很多新人第一次跑 <span class="mono">pre-commit</span> 看到一堆 "Failed" 就慌了，以为自己代码写错了。其实绝大多数时候，那只是 black 或 isort 在<strong>悄悄帮你把格式改好</strong>，文件已经被修改但还没 add。正确的反应不是去 revert，而是 <span class="mono">git add</span> 这些被工具改过的文件，再 <span class="mono">pre-commit run --all-files</span> 跑第二遍——这一遍通常就全 PASS 了。养成"<strong>跑、add、再跑</strong>"的肌肉记忆，你就再也不会被这道关卡卡住。</p>

<h2>那些重要的约定</h2>
<p>除了格式，SGLang 还有几条<strong>约定俗成</strong>的规矩，违反了同样会被 lint 或 reviewer 挡下来：</p>
<p><strong>① 改了 <span class="mono">python/sglang/srt/</span> 下的文件，就要补测试。</strong>先去 <span class="mono">test/registered/unit/</span> 看有没有对应的测试文件，有就给你的改动<strong>加上覆盖</strong>，没有就新建一个（参考第55课）。运行时核心代码没有测试护栏，是不会被轻易接受的。测试不只是为了证明"现在能跑"，更是为了<strong>把你的预期固化下来</strong>：将来别人改动这块代码时，你的测试会替你站岗，第一时间发现回归。</p>
<p><strong>② 文档只能写在 <span class="mono">docs_new/</span> 下。</strong>老的 <span class="mono">docs/</span> 目录已经冻结，对它的改动会<strong>被 lint 直接拒绝</strong>。写新文档、改说明，一律去 <span class="mono">docs_new/</span>。这类"目录级"的约定看似琐碎，但它保证了整个项目的<strong>结构一致</strong>，新人也能凭约定快速找到该改的地方，而不必每次都去问维护者。</p>
<p><strong>③ 顺着已有的模块模式走，别另起炉灶。</strong>SGLang 到处都是可插拔的接缝：注意力后端（第33课）、平台抽象（第42课）。加东西的时候<strong>模仿这些现成的结构</strong>，而不是发明一套新写法——这样 reviewer 一眼就能看懂，审得也快。一个常见的反例是：为了实现一个本可以挂进现有后端接口的功能，却另写了一套并行的机制，结果代码重复、难以维护，几乎一定会在评审中被要求重构。<strong>复用现有接缝，永远比新造结构更受欢迎。</strong></p>
<p>除了这三条，还有一些<strong>小而重要的习惯</strong>值得养成：commit message 写清楚<strong>这次提交做了什么</strong>，而不是含糊的"update"；改动尽量<strong>只碰与本次目标相关的文件</strong>，不要顺手重排一堆无关代码（那会让 diff 变脏）；提交前再用 <span class="mono">git diff</span> 自检一遍，确认没有把调试用的 <span class="mono">print</span>、临时文件或本地配置一起带上去。这些习惯单看都很琐碎，但它们共同决定了 reviewer 打开你 PR 时的<strong>第一印象</strong>——一个干净、克制、目标明确的 diff，远比内容本身更能赢得信任。</p>

<h2>什么是一个"好 PR"</h2>
<p>一个<strong>好的 PR</strong> 有四个特征：<strong>小</strong>（改动集中，不夹带无关修改）、<strong>聚焦</strong>（只做一件事）、<strong>带测试</strong>（证明它真的工作）、并在描述里<strong>解释"为什么"</strong>（动机比代码本身更重要）。这样的 PR 让审阅变得很快。反过来，一个又大、又没测试、又不解释动机的 PR，会让 reviewer 望而生畏，迟迟得不到合并。记住：把工作拆小、说清动机、配上测试，是让你的贡献<strong>尽快被接受</strong>的最实际办法。</p>
<p>为什么"小"如此重要？因为<strong>评审是有成本的</strong>。一个改了 80 行、只做一件事的 PR，reviewer 几分钟就能在脑子里完整推演一遍；而一个改了 2000 行、横跨十几个文件、还夹带无关重构的 PR，reviewer 要么花一整天硬啃，要么干脆一直拖着不看。如果你手上的功能确实很大，正确的做法是<strong>拆成一串小 PR</strong>，每个都能独立审、独立合并，而不是攒成一个巨无霸。小 PR 不仅审得快，出了问题也更容易定位和回滚。拆分本身也是一种沟通：它告诉 reviewer 你<strong>已经把复杂问题想清楚、分解好了</strong>，每一步都站得住脚，这种"可被增量验证"的特性，正是大型项目最看重的工程素养。</p>
<p>而"解释为什么"则常被新人忽视。代码本身只告诉 reviewer "你做了什么"，但<strong>"为什么这么做"</strong>——你在解决什么问题、为什么选这个方案、考虑过哪些替代——只有你知道。把这些写进 PR 描述，reviewer 才能判断你的改动是否合理，而不必反复追问。一个好的描述往往包含：<strong>动机</strong>（解决什么问题）、<strong>做法</strong>（怎么改的）、<strong>验证</strong>（加了哪些测试、跑了什么）。这三样齐全，你的 PR 就已经赢在了起跑线上。如果你的改动涉及某个 issue，记得在描述里<strong>引用它的编号</strong>，让评审能顺藤摸瓜地理解上下文；如果是性能相关的改动，附上<strong>改动前后的对比数据</strong>，会比任何文字都更有说服力。</p>
<p>把这些串起来看：你在第26课学会写一个模型文件、在第33/42课理解了可插拔的接缝、在第55课学会写测试和理解 CI——本课就是教你<strong>如何把这些成果安全地交付出去</strong>。规范不是束缚，而是让你的贡献<strong>更容易被看懂、被信任、被合并</strong>的通行证。术语见第57课。</p>
<p>最后给第一次贡献者一句实在的建议：<strong>从小处开始</strong>。不必一上来就挑战重构整个引擎，先找一个修个拼写错误、补一段文档、给某个函数加一个单元测试这样的小改动练手，把<strong>整条 fork → pre-commit → 测试 → PR 的流程完整走一遍</strong>。当你第一个小 PR 被合并、看到自己的名字出现在贡献者列表里时，你已经掌握了最重要的东西——不是某个具体的命令，而是<strong>一套能反复复用的、安全的交付习惯</strong>。之后再去做更大的功能，你会发现流程早已了然于胸，真正的精力可以全部投在<strong>设计与实现</strong>本身。</p>

<div class="vflow">
<div class="step"><div class="num">1</div><div class="sc"><h4>Fork 官方仓库</h4><p>你没有 push 权限，先复制一份属于自己的副本。</p></div></div>
<div class="step"><div class="num">2</div><div class="sc"><h4>Clone 你的 fork 并开分支</h4><p class="mono">git checkout -b my-feature</p></div></div>
<div class="step"><div class="num">3</div><div class="sc"><h4>做改动</h4><p>顺着已有的可插拔接缝，别另起炉灶。</p></div></div>
<div class="step"><div class="num">4</div><div class="sc"><h4>pre-commit 直到全绿</h4><p class="mono">pre-commit run --all-files</p></div></div>
<div class="step"><div class="num">5</div><div class="sc"><h4>补测试并本地跑通</h4><p>参考第55课，在 test/registered/unit/ 加覆盖。</p></div></div>
<div class="step"><div class="num">6</div><div class="sc"><h4>Push 到你的 fork</h4><p class="mono">git push origin my-feature</p></div></div>
<div class="step"><div class="num">7</div><div class="sc"><h4>对着 main 开 PR</h4><p>进入公开评审，CI 自动触发。</p></div></div>
</div>

<table class="t">
<tr><th>步骤</th><th>命令 / 关卡</th></tr>
<tr><td>装并注册钩子</td><td><span class="mono">pip3 install pre-commit</span> → <span class="mono">pre-commit install</span></td></tr>
<tr><td>跑全部格式 / lint</td><td><span class="mono">pre-commit run --all-files</span>（失败就重跑直到 PASS）</td></tr>
<tr><td>补测试并本地验证</td><td><span class="mono">pytest test/registered/unit/ -v</span></td></tr>
<tr><td>推送并发起合并</td><td><span class="mono">git push origin my-feature</span> → open PR against <span class="mono">main</span></td></tr>
</table>

<div class="cols">
<div class="col"><strong>✅ 好 PR</strong><br>改动小而聚焦；带上对应的单元测试；描述里说清"为什么这么改"；顺着已有模块模式。reviewer 几分钟就能看懂、很快合并。</div>
<div class="col"><strong>❌ 难审的 PR</strong><br>一次塞进上千行、夹带无关改动；没有任何测试；描述只有一句"fix bug"，不解释动机。reviewer 不敢合，PR 长期搁置。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 330" role="img" aria-label="好 PR（小而有测试）vs 难审 PR：好 PR 审得快，难审 PR 慢或长期停滞">
    <rect x="20" y="20" width="368" height="294" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="412" y="20" width="368" height="294" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="36" y="36" width="336" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="204" y="63" text-anchor="middle" style="font-size:14px;font-weight:700;fill:var(--teal)">✅ 好 PR · 小而有测试</text>
    <rect x="428" y="36" width="336" height="44" rx="8" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="596" y="63" text-anchor="middle" style="font-size:14px;font-weight:700;fill:var(--red)">❌ 难审 PR · 大而杂</text>
    <rect x="44" y="104" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="116" style="font-size:13px">小而聚焦</text>
    <rect x="44" y="136" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="148" style="font-size:13px">一个关注点</text>
    <rect x="44" y="168" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="180" style="font-size:13px">含单元测试</text>
    <rect x="44" y="200" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="212" style="font-size:13px">描述清晰</text>
    <rect x="436" y="104" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="116" style="font-size:13px">改动巨大</text>
    <rect x="436" y="136" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="148" style="font-size:13px">多个关注点</text>
    <rect x="436" y="168" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="180" style="font-size:13px">没有测试</text>
    <rect x="436" y="200" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="212" style="font-size:13px">描述含糊</text>
    <text x="44" y="246" style="font-size:12px;fill:var(--muted)">审阅耗时</text>
    <rect x="44" y="256" width="120" height="26" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="104" y="274" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--teal)">快</text>
    <text x="44" y="304" style="font-size:12px;fill:var(--teal)">→ 几分钟审完</text>
    <text x="436" y="246" style="font-size:12px;fill:var(--muted)">审阅耗时</text>
    <rect x="436" y="256" width="320" height="26" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="596" y="274" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--red)">慢 / 停滞</text>
    <text x="436" y="304" style="font-size:12px;fill:var(--red)">→ 长期搁置</text>
  </svg>
  <div class="figcap"><b>图 2 · 好 PR vs 难审 PR</b> — 好 PR 改动小、只做一件事、带测试、描述清晰，审阅几分钟搞定；难审 PR 改动巨大、夹带多个关注点、没有测试、描述含糊，审阅慢甚至长期停滞。</div>
</div>

<div class="flow">
<div class="node">本地 pre-commit + pytest</div>
<div class="arrow">→</div>
<div class="node">push 到你的 fork</div>
<div class="arrow">→</div>
<div class="node">CI 跑相同的检查（第55课）</div>
<div class="arrow">→</div>
<div class="node">维护者 review</div>
<div class="arrow">→</div>
<div class="node">合并进 main</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">docs_new/docs/developer_guide/contribution_guide.mdx</span><span class="ln">fork → 分支 → pre-commit → 测试 → PR</span></div><pre># 1) fork on GitHub, then clone YOUR fork
git clone https://github.com/&lt;your_user_name&gt;/sglang.git
git checkout -b my-feature

# 2) format + lint before pushing (re-run until clean)
pip3 install pre-commit
pre-commit install
pre-commit run --all-files

# 3) add + run a unit test for your change (Lesson 55)
pytest test/registered/unit/ -v

# 4) push your branch and open a Pull Request against main
git push origin my-feature</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">.pre-commit-config.yaml</span><span class="ln">提交即运行的钩子：格式化、lint、拼写、私钥检测…</span></div><pre># .pre-commit-config.yaml — runs automatically on `git commit`
repos:
  - repo: pre-commit-hooks      # trailing-whitespace, end-of-file-fixer,
    hooks: [check-yaml, check-ast, detect-private-key, ...]
  - repo: isort                 # sort imports
  - repo: ruff                  # Python lint
  - repo: black (black-jupyter) # Python format
  - repo: codespell             # spell-check
  - repo: clang-format          # C++/CUDA format
  - repo: local
    hooks: [check-chinese-characters, sort-ci-permissions, ...]</pre></div>

<p>具体怎么用？<span class="mono">pip install pre-commit &amp;&amp; pre-commit install</span> 一次性把这些钩子<strong>挂上 git</strong>，之后每次 <span class="mono">git commit</span> 它们都会自动跑：<span class="mono">ruff</span> + <span class="mono">black</span> 直接把风格问题改好，于是 reviewer 不必纠结格式，可以<strong>专心看逻辑</strong>。再配合“一个 PR 只做一件事、并带上测试”，你的改动就<strong>又好审又好合</strong>。</p>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li>新贡献者没有官方仓库的写权限，流程从 <strong>fork</strong> 开始：fork → clone 你的 fork → 开分支 → 改动 → <span class="mono">pre-commit</span> → 测试 → push → 开 PR（对着官方 <span class="mono">main</span> 主分支）。</li>
<li><span class="mono">pre-commit</span> 是格式 / lint 关卡：<span class="mono">pip3 install pre-commit</span>、<span class="mono">pre-commit install</span>（注册 git 钩子）、<span class="mono">pre-commit run --all-files</span>；第一次失败是它在就地修复，<strong>反复重跑直到全绿</strong>再开 PR。</li>
<li>CI（第55课）会跑<strong>相同</strong>的检查，本地不过 CI 也不过，PR 会被卡住无法合并。</li>
<li>改 <span class="mono">python/sglang/srt/</span> 就要在 <span class="mono">test/registered/unit/</span> 补测试；文档只写 <span class="mono">docs_new/</span>（改老 <span class="mono">docs/</span> 会被 lint 拒绝）。</li>
<li>顺着已有可插拔接缝贡献——新后端（第33/42课）、新模型（第26课）、新处理器（第49课）——大多数情况下不必改动引擎核心。</li>
<li>好 PR = 小、聚焦、带测试、说清"为什么"；改动越小越聚焦，越容易被快速评审和合并。术语见第57课。</li>
</ul>
</div>
""", "en": r"""
<p class="lead">Writing code that runs is only step one; to get it merged into the SGLang main repo you also have to make your code <strong>follow the conventions</strong>, <strong>pass the checks</strong>, and <strong>come with a test</strong>, then hand it to the maintainers for review via a clean <span class="mono">Pull Request</span>. This lesson walks through the full flow from fork to merged PR, plus the unwritten rules behind it. Getting this flow right lets your first contribution <strong>avoid a lot of detours</strong>, and makes maintainers more willing to spend time on your code. The flow itself isn't complicated; the hard part is doing every step well until it becomes a habit.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Think of contributing code as <strong>submitting a chapter to an open-source textbook</strong>. You can't scribble on the publisher's master copy (new contributors have no write access to the official repo), so you first <strong>photocopy your own copy</strong> (fork), open a fresh sheet for your chapter (branch), and after writing you run a <strong>uniform formatting tool</strong> to fix fonts, indentation and punctuation (<span class="mono">pre-commit</span>), attach a <strong>self-check question</strong> proving the content is correct (unit test), and finally <strong>mail it to the editors</strong> asking them to adopt it (open a PR). The editors (CI and maintainers) re-run the exact same formatting tool to double-check — if you skipped formatting, the submission bounces straight back.</p>
<p>Two more lessons are worth remembering from this analogy. First, <strong>the formatting tool isn't there to harass you</strong>: it normalizes every submission into one style so the editor isn't distracted by wildly different formats and can focus on your <strong>content</strong>. Second, <strong>a good submission is small and complete</strong>: rather than handing the editor a thick anthology all at once, submit one chapter at a time with a clear "why I wrote this chapter", so the editor reads and revises it fast. Contributing code is exactly the same — <strong>automate formatting, keep changes small and focused, and state your motivation clearly</strong>, and your "submission" is far easier to adopt.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>This guide keeps repeating one line: <strong>almost everything in SGLang is pluggable</strong>. Precisely because of that, the bar to contribute is low — most new features <strong>slot into an existing seam</strong>: add a new backend (attention backend, Lesson 33), add a new model file (Lesson 26), add a new processor (Lesson 49), none of which touch the engine core. In other words you're not rewriting the engine, you're fitting a new part into a designed socket. The job of the conventions and flow is to make sure that, once installed, the machine stays clean, testable and maintainable.</p>
<p>From a higher vantage point, <strong>the conventions and the pluggable architecture are two sides of one coin</strong>. The architecture provides clear seams that tell you "where the new thing goes"; the conventions provide a clear delivery path that tells you "how to hand it over safely once it's built". Together they let a project with hundreds or thousands of contributors evolve <strong>continuously, quickly and reliably</strong> without collapsing. This lesson gathers the "how to build a part" you learned across the previous fifty-odd lessons into "how to deliver that part into the trunk" — it's your <strong>last mile</strong> to becoming a SGLang contributor.</p>
</div>

<h2>The whole flow: from fork to PR</h2>
<p>New contributors <strong>cannot push directly to the official repo</strong>, so step one is always <strong>fork</strong> — one click on GitHub copies <span class="mono">sgl-project/sglang</span> into <span class="mono">&lt;your_user_name&gt;/sglang</span>. Then clone <strong>your own fork</strong> locally, run <span class="mono">git checkout -b my-feature</span> to open a branch, and make your change on it. Don't rush to push afterwards: first run <span class="mono">pre-commit</span> over formatting and lint, then add a unit test the way <strong>Lesson 55</strong> describes and run it locally, and only once it's green do <span class="mono">git push origin my-feature</span> and finally open a <strong>Pull Request</strong> against the <span class="mono">main</span> branch on GitHub. The chain is <strong>fork → clone → branch → change → pre-commit → test → push → PR</strong>, and no link can be skipped.</p>
<p>Why work on your own branch instead of editing <span class="mono">main</span> directly? Because <strong>a branch isolates each piece of work</strong>. One branch carries one feature or one fix, so if a change turns out wrong you just delete the branch — it never pollutes the clean <span class="mono">main</span> on your fork. It also lets you <strong>pursue several independent contributions at once</strong>: one idea per branch, none interfering. Branch names should <strong>signal intent</strong>, e.g. <span class="mono">fix-sampling-overflow</span> or <span class="mono">add-glm-model</span>, so a reviewer roughly knows what you're doing from the name alone.</p>
<p>When opening the PR there's an often-missed detail: the PR's <strong>target branch must be the official repo's <span class="mono">main</span></strong>, with the source being <span class="mono">my-feature</span> on your fork. GitHub automatically diffs the two sides and presents your commits to the maintainers. From that moment your code enters <strong>public review</strong>: CI triggers automatically, maintainers leave comments, and you may need to keep committing on the same branch in response to feedback and push again — the PR updates itself. This "submit–feedback–revise" loop is the heart of open-source collaboration.</p>

<div class="fig">
  <svg viewBox="0 0 800 240" role="img" aria-label="Contribution flow: fork repo → branch → pre-commit → test → push → open PR, gated by CI and reviewers before merge">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">Contributor path</text>
    <rect x="24" y="64" width="104" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="76" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">Fork</text>
    <text x="76" y="108" text-anchor="middle" style="font-size:11px;fill:var(--muted)">copy repo</text>
    <rect x="148" y="64" width="104" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="200" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">Branch</text>
    <text x="200" y="108" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">git branch</text>
    <rect x="272" y="64" width="104" height="58" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="324" y="90" text-anchor="middle" class="mono" style="font-size:12px;font-weight:700">pre-commit</text>
    <text x="324" y="108" text-anchor="middle" style="font-size:11px;fill:var(--muted)">auto-format</text>
    <rect x="396" y="64" width="104" height="58" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="448" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">Test</text>
    <text x="448" y="108" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">pytest</text>
    <rect x="520" y="64" width="104" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="572" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">Push</text>
    <text x="572" y="108" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">git push</text>
    <rect x="644" y="64" width="104" height="58" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="696" y="90" text-anchor="middle" style="font-size:13px;font-weight:700">Open PR</text>
    <text x="696" y="108" text-anchor="middle" style="font-size:11px;fill:var(--muted)">pull request</text>
    <line x1="130" y1="93" x2="146" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="140,89 146,93 140,97" style="fill:var(--faint)"/>
    <line x1="254" y1="93" x2="270" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="264,89 270,93 264,97" style="fill:var(--faint)"/>
    <line x1="378" y1="93" x2="394" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="388,89 394,93 388,97" style="fill:var(--faint)"/>
    <line x1="502" y1="93" x2="518" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="512,89 518,93 512,97" style="fill:var(--faint)"/>
    <line x1="626" y1="93" x2="642" y2="93" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="636,89 642,93 636,97" style="fill:var(--faint)"/>
    <line x1="696" y1="122" x2="696" y2="156" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="692,150 696,158 700,150" style="fill:var(--faint)"/>
    <rect x="24" y="158" width="724" height="54" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="386" y="181" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--amber)">CI + reviewers gate</text>
    <text x="386" y="200" text-anchor="middle" style="font-size:12px;fill:var(--muted)">must pass before merge to main</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Contribution flow</b> — fork the repo → branch → <span class="mono">pre-commit</span> auto-formats/lints → run tests → push → open a PR; CI and reviewers gate it together, and it merges into <span class="mono">main</span> only after it passes.</div>
</div>

<h2>pre-commit: the style &amp; lint gate</h2>
<p><span class="mono">pre-commit</span> is SGLang's <strong>unified formatting and static-check gate</strong>. Three commands get you going: <span class="mono">pip3 install pre-commit</span> installs the tool, <span class="mono">pre-commit install</span> <strong>registers it as a git hook</strong> (so it runs automatically on every commit), and <span class="mono">pre-commit run --all-files</span> runs all configured checks — black, isort, etc. — over every file and <strong>applies the fixes it can</strong>. One crucial detail: <strong>the first run will likely fail</strong>, because it rewrites your files in place (reordering imports, adding blank lines, normalizing quotes). That's not an error, it's the tool fixing things for you. What you do is <strong>add those automatic changes, then run it again</strong>, <strong>re-running until everything PASSES</strong>. It must be fully green before you open a PR, because CI (Lesson 55) runs the <strong>exact same</strong> checks — if it doesn't pass locally, it won't pass in CI either, and the PR will be blocked from merging.</p>
<p>Why hand these tedious formatting matters to a tool instead of relying on people to watch for them? Because <strong>a uniform code style is infrastructure for a large collaborative project</strong>. When hundreds or thousands of contributors push to the same repo, if everyone's indentation, quotes, and import order differ, diffs fill with "noise" unrelated to logic, and a reviewer struggles to tell which lines are the real change. <span class="mono">pre-commit</span> fully <strong>automates and mechanizes</strong> "style", freeing human attention for <strong>logic and design</strong>. That's why it's set up as a hard gate: if style doesn't pass, no further discussion can even begin.</p>
<p>Registering it as a <strong>git hook</strong> (<span class="mono">pre-commit install</span>) has a further benefit: from then on every <span class="mono">git commit</span> runs it over the changed files, <strong>catching problems before they enter history</strong>. You needn't remember to run it manually, and you won't get bounced by CI after pushing just because you forgot to format. In other words, the local hook and remote CI run the same set of checks — the gate is simply <strong>moved earlier</strong>, onto your machine, so feedback arrives sooner and cheaper.</p>
<p>A frequent pitfall deserves a specific warning: many newcomers panic at a wall of "Failed" on their first <span class="mono">pre-commit</span> run, thinking they wrote bad code. In fact, most of the time that's just black or isort <strong>quietly fixing your formatting</strong> — the files are already modified but not yet added. The right reaction isn't to revert, but to <span class="mono">git add</span> the tool-modified files and run <span class="mono">pre-commit run --all-files</span> a second time — that pass usually comes out all PASS. Build the muscle memory of "<strong>run, add, run again</strong>" and this gate will never block you again.</p>

<h2>The conventions that matter</h2>
<p>Beyond formatting, SGLang has a few <strong>unwritten rules</strong> that, if broken, will likewise be stopped by lint or a reviewer:</p>
<p><strong>① Touch a file under <span class="mono">python/sglang/srt/</span>, add a test.</strong> First check <span class="mono">test/registered/unit/</span> for an existing test file; if there is one, <strong>add coverage</strong> for your change, and if there isn't, create one (see Lesson 55). Runtime-core code with no test guardrail won't be accepted easily. A test isn't just proof that it "runs now" — it <strong>locks in your expectation</strong>: when someone else later changes this code, your test stands guard and catches the regression immediately.</p>
<p><strong>② Documentation only goes under <span class="mono">docs_new/</span>.</strong> The old <span class="mono">docs/</span> directory is frozen, and edits to it are <strong>rejected outright by lint</strong>. New docs and explanation edits all go to <span class="mono">docs_new/</span>. Such a "directory-level" convention looks trivial, but it guarantees the project's <strong>structural consistency</strong>, so newcomers can find the right place to edit by convention rather than asking a maintainer each time.</p>
<p><strong>③ Follow existing module patterns; don't invent new structure.</strong> SGLang is full of pluggable seams: the attention backend (Lesson 33), the platform abstraction (Lesson 42). When adding something, <strong>mimic these existing structures</strong> rather than inventing a new style — that way a reviewer understands it at a glance and reviews it fast. A common anti-example: to implement something that could have plugged into an existing backend interface, someone writes a parallel mechanism instead, producing duplicated, hard-to-maintain code that will almost certainly be asked to be refactored in review. <strong>Reusing an existing seam is always more welcome than inventing new structure.</strong></p>
<p>Beyond these three, a few <strong>small but important habits</strong> are worth building: write a commit message that clearly states <strong>what this commit does</strong>, not a vague "update"; keep the change <strong>touching only files relevant to this goal</strong>, without casually reshuffling unrelated code (that dirties the diff); and before committing, self-review with <span class="mono">git diff</span> to confirm you didn't drag along a debug <span class="mono">print</span>, a temp file, or local config. Each habit looks trivial alone, but together they determine the reviewer's <strong>first impression</strong> when opening your PR — a clean, restrained, clearly-targeted diff wins trust far more than the content itself.</p>

<h2>What makes a "good PR"</h2>
<p>A <strong>good PR</strong> has four traits: <strong>small</strong> (focused changes, no unrelated edits smuggled in), <strong>focused</strong> (does one thing), <strong>tested</strong> (proves it actually works), and it <strong>explains "why"</strong> in the description (the motivation matters more than the code itself). Such a PR makes review fast. Conversely, a PR that's huge, untested, and unexplained intimidates the reviewer and lingers unmerged. Remember: splitting work small, stating the motivation, and bringing a test is the most practical way to get your contribution <strong>accepted quickly</strong>.</p>
<p>Why does "small" matter so much? Because <strong>review has a cost</strong>. A PR that changes 80 lines and does one thing can be fully reasoned through in a reviewer's head in minutes; a PR that changes 2000 lines across a dozen files with unrelated refactors mixed in either takes a whole day to grind through, or just sits unreviewed. If your feature really is large, the right move is to <strong>split it into a series of small PRs</strong>, each independently reviewable and mergeable, rather than piling up one behemoth. Small PRs are not only reviewed faster — when something breaks, they're easier to locate and roll back. The split itself is a form of communication: it tells the reviewer you've <strong>already thought the complex problem through and decomposed it</strong>, with each step standing on its own — and that "incrementally verifiable" quality is exactly the engineering maturity large projects value most.</p>
<p>And "explain why" is often overlooked by newcomers. The code itself only tells a reviewer "what you did", but <strong>"why you did it this way"</strong> — what problem you're solving, why you chose this approach, which alternatives you considered — only you know. Put these into the PR description so the reviewer can judge whether your change is sound without repeatedly asking. A good description usually contains: the <strong>motivation</strong> (what problem it solves), the <strong>approach</strong> (how you changed it), and the <strong>verification</strong> (which tests you added, what you ran). With all three present, your PR is already winning at the starting line. If your change relates to an issue, remember to <strong>reference its number</strong> in the description so reviewers can trace the context; and if it's a performance change, attaching <strong>before/after comparison numbers</strong> is more persuasive than any prose.</p>
<p>Tie it all together: you learned to write a model file in Lesson 26, understood the pluggable seams in Lessons 33/42, and learned to write tests and understand CI in Lesson 55 — this lesson teaches you <strong>how to deliver those results safely</strong>. The conventions aren't shackles; they're a passport that makes your contribution <strong>easier to understand, trust, and merge</strong>. Glossary in Lesson 57.</p>
<p>One practical piece of advice for first-time contributors: <strong>start small</strong>. Don't open by trying to refactor the whole engine — pick a small change first, like fixing a typo, adding a paragraph of docs, or adding a single unit test to some function, and <strong>walk the entire fork → pre-commit → test → PR flow end to end</strong>. When your first small PR is merged and you see your name in the contributors list, you've already grasped the most important thing — not some specific command, but <strong>a reusable, safe set of delivery habits</strong>. When you later take on bigger features, you'll find the flow is second nature, and your real energy can go entirely into the <strong>design and implementation</strong> themselves.</p>

<div class="vflow">
<div class="step"><div class="num">1</div><div class="sc"><h4>Fork the official repo</h4><p>You have no push access — make your own copy first.</p></div></div>
<div class="step"><div class="num">2</div><div class="sc"><h4>Clone your fork and branch</h4><p class="mono">git checkout -b my-feature</p></div></div>
<div class="step"><div class="num">3</div><div class="sc"><h4>Make the change</h4><p>Along an existing pluggable seam — don't invent new structure.</p></div></div>
<div class="step"><div class="num">4</div><div class="sc"><h4>pre-commit until all green</h4><p class="mono">pre-commit run --all-files</p></div></div>
<div class="step"><div class="num">5</div><div class="sc"><h4>Add a test, run it locally</h4><p>Per Lesson 55, add coverage under test/registered/unit/.</p></div></div>
<div class="step"><div class="num">6</div><div class="sc"><h4>Push to your fork</h4><p class="mono">git push origin my-feature</p></div></div>
<div class="step"><div class="num">7</div><div class="sc"><h4>Open a PR against main</h4><p>Public review begins; CI triggers automatically.</p></div></div>
</div>

<table class="t">
<tr><th>Step</th><th>Command / gate</th></tr>
<tr><td>Install &amp; register hook</td><td><span class="mono">pip3 install pre-commit</span> → <span class="mono">pre-commit install</span></td></tr>
<tr><td>Run all formatting / lint</td><td><span class="mono">pre-commit run --all-files</span> (re-run until PASS)</td></tr>
<tr><td>Add test &amp; verify locally</td><td><span class="mono">pytest test/registered/unit/ -v</span></td></tr>
<tr><td>Push &amp; open merge request</td><td><span class="mono">git push origin my-feature</span> → open PR against <span class="mono">main</span></td></tr>
</table>

<div class="cols">
<div class="col"><strong>✅ Good PR</strong><br>Small and focused; ships with the matching unit test; the description says "why this change"; follows existing module patterns. A reviewer understands it in minutes and merges quickly.</div>
<div class="col"><strong>❌ Hard-to-review PR</strong><br>Thousands of lines at once with unrelated edits mixed in; no tests at all; a one-line "fix bug" description with no motivation. The reviewer is afraid to merge, and the PR sits idle.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 330" role="img" aria-label="A good PR (small, tested) vs a hard-to-review PR: good is reviewed fast, hard is slow or stalled">
    <rect x="20" y="20" width="368" height="294" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="412" y="20" width="368" height="294" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="36" y="36" width="336" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="204" y="63" text-anchor="middle" style="font-size:14px;font-weight:700;fill:var(--teal)">✅ Good PR · small, tested</text>
    <rect x="428" y="36" width="336" height="44" rx="8" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="596" y="63" text-anchor="middle" style="font-size:14px;font-weight:700;fill:var(--red)">❌ Hard PR · big, mixed</text>
    <rect x="44" y="104" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="116" style="font-size:13px">Small &amp; focused</text>
    <rect x="44" y="136" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="148" style="font-size:13px">One concern</text>
    <rect x="44" y="168" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="180" style="font-size:13px">Has unit tests</text>
    <rect x="44" y="200" width="14" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="212" style="font-size:13px">Clear description</text>
    <rect x="436" y="104" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="116" style="font-size:13px">Huge diff</text>
    <rect x="436" y="136" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="148" style="font-size:13px">Many concerns</text>
    <rect x="436" y="168" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="180" style="font-size:13px">No tests</text>
    <rect x="436" y="200" width="14" height="14" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="458" y="212" style="font-size:13px">Vague description</text>
    <text x="44" y="246" style="font-size:12px;fill:var(--muted)">Review time</text>
    <rect x="44" y="256" width="120" height="26" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="104" y="274" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--teal)">fast</text>
    <text x="44" y="304" style="font-size:12px;fill:var(--teal)">→ merged in minutes</text>
    <text x="436" y="246" style="font-size:12px;fill:var(--muted)">Review time</text>
    <rect x="436" y="256" width="320" height="26" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="596" y="274" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--red)">slow / stalled</text>
    <text x="436" y="304" style="font-size:12px;fill:var(--red)">→ sits idle</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Good PR vs hard-to-review PR</b> — a good PR is small, does one thing, ships tests, and has a clear description, so review takes minutes; a hard PR is huge, mixes many concerns, has no tests, and is vaguely described, so review is slow or stalls indefinitely.</div>
</div>

<div class="flow">
<div class="node">Local pre-commit + pytest</div>
<div class="arrow">→</div>
<div class="node">push to your fork</div>
<div class="arrow">→</div>
<div class="node">CI runs the same checks (Lesson 55)</div>
<div class="arrow">→</div>
<div class="node">maintainer review</div>
<div class="arrow">→</div>
<div class="node">merge into main</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">docs_new/docs/developer_guide/contribution_guide.mdx</span><span class="ln">fork → branch → pre-commit → test → PR</span></div><pre># 1) fork on GitHub, then clone YOUR fork
git clone https://github.com/&lt;your_user_name&gt;/sglang.git
git checkout -b my-feature

# 2) format + lint before pushing (re-run until clean)
pip3 install pre-commit
pre-commit install
pre-commit run --all-files

# 3) add + run a unit test for your change (Lesson 55)
pytest test/registered/unit/ -v

# 4) push your branch and open a Pull Request against main
git push origin my-feature</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">.pre-commit-config.yaml</span><span class="ln">hooks that run on every commit: format, lint, spell, secret-scan…</span></div><pre># .pre-commit-config.yaml — runs automatically on `git commit`
repos:
  - repo: pre-commit-hooks      # trailing-whitespace, end-of-file-fixer,
    hooks: [check-yaml, check-ast, detect-private-key, ...]
  - repo: isort                 # sort imports
  - repo: ruff                  # Python lint
  - repo: black (black-jupyter) # Python format
  - repo: codespell             # spell-check
  - repo: clang-format          # C++/CUDA format
  - repo: local
    hooks: [check-chinese-characters, sort-ci-permissions, ...]</pre></div>

<p>How do you use it? <span class="mono">pip install pre-commit &amp;&amp; pre-commit install</span> wires these hooks <strong>into git</strong> once, and from then on every <span class="mono">git commit</span> runs them automatically: <span class="mono">ruff</span> + <span class="mono">black</span> fix style issues for you, so reviewers don't argue about formatting and can <strong>focus on the logic</strong>. Pair that with “one concern per PR, with tests” and your change is <strong>both easy to review and easy to merge</strong>.</p>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li>New contributors have no write access to the official repo, so the flow starts with <strong>fork</strong>: fork → clone your fork → branch → change → <span class="mono">pre-commit</span> → test → push → open PR (against <span class="mono">main</span>).</li>
<li><span class="mono">pre-commit</span> is the style / lint gate: <span class="mono">pip3 install pre-commit</span>, <span class="mono">pre-commit install</span> (registers a git hook), <span class="mono">pre-commit run --all-files</span>; the first failure is it fixing things in place — <strong>re-run until green</strong> before opening a PR.</li>
<li>CI (Lesson 55) runs the <strong>same</strong> checks; if it fails locally it fails in CI, and the PR is blocked.</li>
<li>Touch <span class="mono">python/sglang/srt/</span> and you must add a test under <span class="mono">test/registered/unit/</span>; docs only go in <span class="mono">docs_new/</span> (editing the old <span class="mono">docs/</span> is rejected).</li>
<li>Contribute along existing pluggable seams — a new backend (Lessons 33/42), a new model (Lesson 26), a new processor (Lesson 49) — without touching the engine core.</li>
<li>A good PR = small, focused, tested, and explains "why"; the smaller and more focused the change, the faster it gets reviewed and merged. Glossary in Lesson 57.</li>
</ul>
</div>
"""}
LESSON_57 = {"zh": r"""
<p class="lead">这是全书的<strong>术语速查表</strong>，把前 56 课出现的关键概念按主题归拢成几张表，每条配<strong>一句话定义</strong>与<strong>所属课次</strong>，方便日后随时回查。读到某个词记不清时，翻到这里、再顺着课次回正文细看。</p>

<div class="fig">
  <svg viewBox="0 0 840 430" role="img" aria-label="全书地图：13 个部分沿一次请求的一生展开，分为入口与搭建（1-4）、核心引擎（5-9）、扩展进阶实践（10-13）三大段">
    <text x="16" y="26" style="font-weight:700;fill:var(--accent-ink)">全书地图 · 一次请求的一生</text>

    <rect x="12" y="40" width="816" height="110" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.25;stroke-dasharray:5 5"/>
    <text x="24" y="60" style="font-weight:700;fill:var(--muted);font-size:13px">入口与搭建（1–4）</text>
    <rect x="28" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="119" y="106" text-anchor="middle">1 概览</text>
    <text x="119" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L01–03</text>
    <rect x="226" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="317" y="106" text-anchor="middle">2 基础</text>
    <text x="317" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L04–08</text>
    <rect x="424" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="515" y="106" text-anchor="middle">3 前端 DSL</text>
    <text x="515" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L09–12</text>
    <rect x="622" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="713" y="106" text-anchor="middle">4 入口</text>
    <text x="713" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L13–17</text>

    <line x1="420" y1="151" x2="420" y2="159" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="420,163 415,154 425,154" style="fill:var(--muted)"/>

    <rect x="12" y="160" width="816" height="110" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.25;stroke-dasharray:5 5"/>
    <text x="24" y="180" style="font-weight:700;fill:var(--muted);font-size:13px">核心引擎（5–9）</text>
    <rect x="28" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="102" y="226" text-anchor="middle">5 调度器</text>
    <text x="102" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L18–23</text>
    <rect x="188" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="262" y="226" text-anchor="middle">6 模型执行</text>
    <text x="262" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L24–28</text>
    <rect x="348" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="422" y="226" text-anchor="middle">7 KV 缓存</text>
    <text x="422" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L29–32</text>
    <rect x="508" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="582" y="226" text-anchor="middle">8 注意力</text>
    <text x="582" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L33–37</text>
    <rect x="668" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="742" y="226" text-anchor="middle">9 内核硬件</text>
    <text x="742" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L38–42</text>

    <line x1="420" y1="271" x2="420" y2="279" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="420,283 415,274 425,274" style="fill:var(--muted)"/>

    <rect x="12" y="280" width="816" height="110" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.25;stroke-dasharray:5 5"/>
    <text x="24" y="300" style="font-weight:700;fill:var(--muted);font-size:13px">扩展 · 进阶 · 实践（10–13）</text>
    <rect x="28" y="324" width="182" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="119" y="346" text-anchor="middle">10 性能</text>
    <text x="119" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L43–48</text>
    <rect x="226" y="324" width="182" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="317" y="346" text-anchor="middle">11 进阶</text>
    <text x="317" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L49–52</text>
    <rect x="424" y="324" width="182" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="515" y="346" text-anchor="middle">12 实践</text>
    <text x="515" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L53–57</text>
    <rect x="622" y="324" width="182" height="52" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="713" y="346" text-anchor="middle">13 设计主题</text>
    <text x="713" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L58–63</text>

    <text x="24" y="410" style="fill:var(--faint);font-size:12px">沿箭头自上而下，即一次请求从进入到沉淀为设计主题的路径。</text>
  </svg>
  <div class="figcap"><b>图 · 全书地图</b> — 13 个部分沿“一次请求的一生”展开：1–4 入口与搭建、5–9 核心引擎、10–13 扩展进阶实践。</div>
</div>

<p>这张图就是后文术语的坐标系：遇到生词，回到它所属的部分即可。</p>

<h2>一、总览与基础</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>SGLang</strong></td><td>面向 LLM 的高性能推理与服务引擎：前端 DSL + 运行时（SRT）两层</td><td>第 1 课</td></tr>
  <tr><td><strong>Token</strong></td><td>模型处理的最小单位，文本经分词器切成的整数 id 序列</td><td>第 4 课</td></tr>
  <tr><td><strong>自回归 / Autoregression</strong></td><td>每次前向只生成一个 token，再喂回去生成下一个</td><td>第 4 课</td></tr>
  <tr><td><strong>Prefill / Decode</strong></td><td>prefill 一次并行算完整段提示词（算力受限）；decode 逐 token 生成（带宽受限）</td><td>第 4、8 课</td></tr>
  <tr><td><strong>KV 缓存</strong></td><td>缓存每个 token 的 Key/Value，避免重复计算历史；显存的主要消耗</td><td>第 4 课</td></tr>
  <tr><td><strong>连续批处理 / Continuous batching</strong></td><td>请求随到随入批、完成即出批，让 GPU 始终满载</td><td>第 5 课</td></tr>
  <tr><td><strong>PagedAttention / 分页 KV</strong></td><td>把 KV 按固定大小的页分配，消除碎片、支持共享</td><td>第 6 课</td></tr>
  <tr><td><strong>RadixAttention（概念）</strong></td><td>用基数树共享相同前缀的 KV，命中即复用，省算省显存</td><td>第 7 课</td></tr>
  <tr><td><strong>吞吐 vs 延迟</strong></td><td>系统视角（tokens/s）与用户视角（TTFT/TPOT）的核心取舍</td><td>第 8 课</td></tr>
</table>

<h2>二、前端 DSL</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>SGLang DSL</strong></td><td>用 Python 写复杂生成流程的领域语言：gen/select/fork 等原语</td><td>第 9 课</td></tr>
  <tr><td><strong>解释器 / Tracer</strong></td><td>执行 DSL 程序；tracer 先静态展开出依赖图以便并行</td><td>第 10 课</td></tr>
  <tr><td><strong>fork / join</strong></td><td>把一个生成分叉成多条并行分支，再合并；天然共享前缀</td><td>第 11 课</td></tr>
  <tr><td><strong>后端 / OpenAI 兼容</strong></td><td>DSL 可跑在本地运行时或远端 OpenAI 等后端之上</td><td>第 12 课</td></tr>
</table>

<h2>三、入口与分发</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>Engine / HTTP Server</strong></td><td>Engine 是进程内离线入口；HTTP Server 在其上提供在线接口</td><td>第 13 课</td></tr>
  <tr><td><strong>TokenizerManager</strong></td><td>入口进程：分词、组装请求、通过 IPC 与调度器通信</td><td>第 14 课</td></tr>
  <tr><td><strong>OpenAI 兼容层</strong></td><td>把 OpenAI/Anthropic/Ollama 协议翻译成 SGLang 内部请求</td><td>第 15 课</td></tr>
  <tr><td><strong>IO 结构 / IPC</strong></td><td>进程间传递的请求/响应数据结构，经 ZMQ 等通道</td><td>第 16 课</td></tr>
  <tr><td><strong>DetokenizerManager / 流式</strong></td><td>把生成的 token 增量解码成文本，边生成边流式返回</td><td>第 17 课</td></tr>
</table>

<h2>四、调度器</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>调度事件循环</strong></td><td>调度器主循环：收请求→组批→前向→出结果，永不停歇</td><td>第 18 课</td></tr>
  <tr><td><strong>Req / ScheduleBatch</strong></td><td>Req 是单个请求的运行态；ScheduleBatch 是一次调度组成的批</td><td>第 19 课</td></tr>
  <tr><td><strong>调度策略</strong></td><td>决定先跑谁：优先级、公平、最长前缀匹配等</td><td>第 20 课</td></tr>
  <tr><td><strong>零开销重叠调度</strong></td><td>把 CPU 调度与 GPU 前向重叠，隐藏调度开销</td><td>第 21 课</td></tr>
  <tr><td><strong>分块 prefill / Chunked prefill</strong></td><td>把长 prompt 的 prefill 切块，与 decode 交错，降低 TTFT 抖动</td><td>第 22 课</td></tr>
  <tr><td><strong>DP 控制器 / PP 调度</strong></td><td>数据并行的请求分发与流水线并行的阶段调度</td><td>第 23 课</td></tr>
</table>

<h2>五、模型执行</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>ModelRunner / ForwardBatch</strong></td><td>ModelRunner 驱动一次前向；ForwardBatch 是其输入的统一批结构</td><td>第 24 课</td></tr>
  <tr><td><strong>权重加载</strong></td><td>按分片/量化格式把模型权重载入各 rank 显存</td><td>第 25 课</td></tr>
  <tr><td><strong>写一个模型</strong></td><td>用 SGLang 的层与 forward 约定实现一个新模型</td><td>第 26 课</td></tr>
  <tr><td><strong>CUDA Graph 捕获/重放</strong></td><td>把静态形状的前向录成图，重放时几乎零启动开销</td><td>第 27 课</td></tr>
  <tr><td><strong>Sampler / 采样参数</strong></td><td>从 logits 按 temperature/top-p/top-k 等采样出下一个 token</td><td>第 28 课</td></tr>
</table>

<h2>六、KV 缓存与内存</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>RadixAttention 实现</strong></td><td>用基数树索引前缀 KV，匹配最长公共前缀并复用其缓存</td><td>第 29 课</td></tr>
  <tr><td><strong>分页内存池</strong></td><td>按页管理 KV 显存，token→页的映射表支撑分页与共享</td><td>第 30 课</td></tr>
  <tr><td><strong>HiCache 分层</strong></td><td>把 KV 在 GPU/CPU/磁盘多级缓存间分层存放与换入换出</td><td>第 31 课</td></tr>
  <tr><td><strong>淘汰 / 命中率</strong></td><td>LRU 淘汰可淘汰的叶子、锁定在用节点；命中率直接等于吞吐</td><td>第 32 课</td></tr>
</table>

<h2>七、注意力与算子层</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>Attention 后端抽象</strong></td><td>注意力是可换的策略：FlashInfer/Triton/FA，按硬件选</td><td>第 33 课</td></tr>
  <tr><td><strong>MoE 层</strong></td><td>路由到 top-k 专家做稀疏计算；FusedMoE 融合路由+GEMM</td><td>第 34 课</td></tr>
  <tr><td><strong>量化 / Quantization</strong></td><td>用更少比特（FP8/INT4/AWQ/GPTQ）省显存与带宽</td><td>第 35 课</td></tr>
  <tr><td><strong>RoPE / 归一化 / 算子</strong></td><td>旋转位置编码、RMSNorm 等轻量算子，常被融合</td><td>第 36 课</td></tr>
  <tr><td><strong>Logits / 词表并行</strong></td><td>lm_head 把 hidden 投影成词表大小的分数；按词表维 TP 切分</td><td>第 37 课</td></tr>
</table>

<h2>八、内核与硬件</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>sgl-kernel</strong></td><td>独立 C++/CUDA 工程，AOT 编译成 .so，暴露为 torch.ops.sgl_kernel.*</td><td>第 38 课</td></tr>
  <tr><td><strong>JIT kernel</strong></td><td>运行时按需编译小内核并缓存，灵活但有首次编译开销</td><td>第 39 课</td></tr>
  <tr><td><strong>Attention kernel</strong></td><td>decode 内核按 page_table 收集分页 KV，做分块 + 在线 softmax</td><td>第 40 课</td></tr>
  <tr><td><strong>算子融合 / CUDA Graph</strong></td><td>融合算子省 HBM 往返；静态形状利于图捕获，二者协同</td><td>第 41 课</td></tr>
  <tr><td><strong>多硬件后端</strong></td><td>平台抽象 + 各芯片内核，让一套引擎跑遍 NVIDIA/AMD/NPU/...</td><td>第 42 课</td></tr>
</table>

<h2>九、性能创新专题</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>投机解码 / Speculative</strong></td><td>草稿模型猜 k 个 token，目标模型一次验完，无损加速</td><td>第 43 课</td></tr>
  <tr><td><strong>接受率 / 接受长度</strong></td><td>accept_rate(α，不含 bonus) 与 accept_length(τ，含 bonus)，衡量加速</td><td>第 43 课</td></tr>
  <tr><td><strong>EAGLE</strong></td><td>复用目标隐藏态在特征级草拟，提议 token 树 + 树注意力一次验证</td><td>第 44 课</td></tr>
  <tr><td><strong>PD 分离</strong></td><td>prefill 与 decode 分机，KV 跨机传输，各自吃满瓶颈</td><td>第 45 课</td></tr>
  <tr><td><strong>TP / PP / EP / DP</strong></td><td>张量/流水/专家/数据四种并行，统一在 GroupCoordinator 之下</td><td>第 46 课</td></tr>
  <tr><td><strong>EPLB</strong></td><td>专家并行负载均衡：测量专家负载并周期性重平衡放置</td><td>第 47 课</td></tr>
  <tr><td><strong>结构化输出 / 跳跃前进</strong></td><td>语法 FSM 屏蔽 logits 保证合法；确定段跳跃前进免调模型</td><td>第 48 课</td></tr>
</table>

<h2>十、进阶选读</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>多模态 VLM</strong></td><td>处理器插占位符，编码器出嵌入再缝合进序列，其余引擎不变</td><td>第 49 课</td></tr>
  <tr><td><strong>多 LoRA 批处理</strong></td><td>一个底座 + 适配器池；同批不同请求用不同 LoRA，分组 GEMM 一次算</td><td>第 50 课</td></tr>
  <tr><td><strong>RL Rollout / 权重同步</strong></td><td>SGLang 当 RL 的生成引擎；原地热更权重免重启</td><td>第 51 课</td></tr>
  <tr><td><strong>扩散模型 / Diffusion</strong></td><td>迭代去噪而非自回归；复用 sgl-kernel/调度/CUDA Graph</td><td>第 52 课</td></tr>
</table>

<h2>十一、实战与工具</h2>
<table class="t">
  <tr><th>术语</th><th>一句话定义</th><th>课次</th></tr>
  <tr><td><strong>ServerArgs</strong></td><td>中心配置 dataclass，每个字段自动映射成一个 --kebab CLI flag</td><td>第 53 课</td></tr>
  <tr><td><strong>BenchmarkMetrics</strong></td><td>压测汇总：吞吐 + TTFT/TPOT 的 mean/median/p90/p95/p99</td><td>第 54 课</td></tr>
  <tr><td><strong>TTFT / TPOT</strong></td><td>首 token 时延（prefill）/ 每 token 时延（decode，即 ITL）</td><td>第 54 课</td></tr>
  <tr><td><strong>CustomTestCase</strong></td><td>测试基类：包裹 setUpClass 保证失败也清理，不泄漏端口/进程</td><td>第 55 课</td></tr>
  <tr><td><strong>pre-commit / PR 流程</strong></td><td>fork→分支→pre-commit→测试→PR；改 srt 就补测试，文档进 docs_new</td><td>第 56 课</td></tr>
</table>
""", "en": r"""
<p class="lead">This is the whole guide's <strong>quick-reference glossary</strong>. It gathers the key concepts from the first 56 lessons into a few themed tables, each with a <strong>one-line definition</strong> and its <strong>lesson number</strong> for easy back-reference. When a term slips your mind, flip here, then follow the lesson number back to the main text.</p>

<div class="fig">
  <svg viewBox="0 0 840 430" role="img" aria-label="Map of the whole guide: the 13 parts along the life of a request, grouped into entry and setup (1-4), the core engine (5-9), and scale, advanced and practice (10-13)">
    <text x="16" y="26" style="font-weight:700;fill:var(--accent-ink)">Map · the life of a request</text>

    <rect x="12" y="40" width="816" height="110" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.25;stroke-dasharray:5 5"/>
    <text x="24" y="60" style="font-weight:700;fill:var(--muted);font-size:13px">Entry &amp; setup (1–4)</text>
    <rect x="28" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="119" y="106" text-anchor="middle">1 Overview</text>
    <text x="119" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L01–03</text>
    <rect x="226" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="317" y="106" text-anchor="middle">2 Foundations</text>
    <text x="317" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L04–08</text>
    <rect x="424" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="515" y="106" text-anchor="middle">3 Frontend DSL</text>
    <text x="515" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L09–12</text>
    <rect x="622" y="84" width="182" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="713" y="106" text-anchor="middle">4 Entrypoints</text>
    <text x="713" y="124" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L13–17</text>

    <line x1="420" y1="151" x2="420" y2="159" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="420,163 415,154 425,154" style="fill:var(--muted)"/>

    <rect x="12" y="160" width="816" height="110" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.25;stroke-dasharray:5 5"/>
    <text x="24" y="180" style="font-weight:700;fill:var(--muted);font-size:13px">The core engine (5–9)</text>
    <rect x="28" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="102" y="226" text-anchor="middle">5 Scheduler</text>
    <text x="102" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L18–23</text>
    <rect x="188" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="262" y="226" text-anchor="middle">6 Model exec</text>
    <text x="262" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L24–28</text>
    <rect x="348" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="422" y="226" text-anchor="middle">7 KV cache</text>
    <text x="422" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L29–32</text>
    <rect x="508" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="582" y="226" text-anchor="middle">8 Attention</text>
    <text x="582" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L33–37</text>
    <rect x="668" y="204" width="148" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="742" y="226" text-anchor="middle">9 Kernels &amp; HW</text>
    <text x="742" y="244" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L38–42</text>

    <line x1="420" y1="271" x2="420" y2="279" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="420,283 415,274 425,274" style="fill:var(--muted)"/>

    <rect x="12" y="280" width="816" height="110" rx="10" style="fill:none;stroke:var(--line);stroke-width:1.25;stroke-dasharray:5 5"/>
    <text x="24" y="300" style="font-weight:700;fill:var(--muted);font-size:13px">Scale · advanced · practice (10–13)</text>
    <rect x="28" y="324" width="182" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="119" y="346" text-anchor="middle">10 Performance</text>
    <text x="119" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L43–48</text>
    <rect x="226" y="324" width="182" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="317" y="346" text-anchor="middle">11 Advanced</text>
    <text x="317" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L49–52</text>
    <rect x="424" y="324" width="182" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="515" y="346" text-anchor="middle">12 Practice</text>
    <text x="515" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L53–57</text>
    <rect x="622" y="324" width="182" height="52" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="713" y="346" text-anchor="middle">13 Design themes</text>
    <text x="713" y="364" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">L58–63</text>

    <text x="24" y="410" style="fill:var(--faint);font-size:12px">Follow the arrows top-down: a request enters and settles into design themes.</text>
  </svg>
  <div class="figcap"><b>Map · the whole guide</b> — the 13 parts along the life of a request: 1–4 entry &amp; setup, 5–9 the core engine, 10–13 scale, advanced &amp; practice.</div>
</div>

<p>This map is the coordinate system for the terms below: meet a new word, jump back to its part.</p>

<h2>1. Overview &amp; foundations</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>SGLang</strong></td><td>A high-performance LLM inference &amp; serving engine: a frontend DSL + a runtime (SRT)</td><td>L1</td></tr>
  <tr><td><strong>Token</strong></td><td>The model's smallest unit — the integer-id sequence text is tokenized into</td><td>L4</td></tr>
  <tr><td><strong>Autoregression</strong></td><td>Each forward emits one token, fed back to produce the next</td><td>L4</td></tr>
  <tr><td><strong>Prefill / Decode</strong></td><td>Prefill computes the whole prompt in parallel (compute-bound); decode emits token-by-token (bandwidth-bound)</td><td>L4, L8</td></tr>
  <tr><td><strong>KV cache</strong></td><td>Caches each token's Key/Value to avoid recomputing history; the main VRAM consumer</td><td>L4</td></tr>
  <tr><td><strong>Continuous batching</strong></td><td>Requests join/leave the batch as they arrive/finish, keeping the GPU saturated</td><td>L5</td></tr>
  <tr><td><strong>PagedAttention / paged KV</strong></td><td>Allocate KV in fixed-size pages — no fragmentation, sharing possible</td><td>L6</td></tr>
  <tr><td><strong>RadixAttention (concept)</strong></td><td>Share KV of identical prefixes via a radix tree; reuse on a hit to save compute &amp; VRAM</td><td>L7</td></tr>
  <tr><td><strong>Throughput vs latency</strong></td><td>The core trade-off: system view (tokens/s) vs user view (TTFT/TPOT)</td><td>L8</td></tr>
</table>

<h2>2. Frontend DSL</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>SGLang DSL</strong></td><td>A Python DSL for complex generation flows: gen / select / fork primitives</td><td>L9</td></tr>
  <tr><td><strong>Interpreter / Tracer</strong></td><td>Runs a DSL program; the tracer pre-expands the dependency graph for parallelism</td><td>L10</td></tr>
  <tr><td><strong>fork / join</strong></td><td>Split one generation into parallel branches and merge them; shares prefixes naturally</td><td>L11</td></tr>
  <tr><td><strong>Backends / OpenAI-compat</strong></td><td>The DSL runs on the local runtime or a remote backend like OpenAI</td><td>L12</td></tr>
</table>

<h2>3. Entrypoints &amp; dispatch</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>Engine / HTTP Server</strong></td><td>Engine is the in-process offline entry; the HTTP Server adds an online API on top</td><td>L13</td></tr>
  <tr><td><strong>TokenizerManager</strong></td><td>The entry process: tokenize, assemble requests, talk to the scheduler over IPC</td><td>L14</td></tr>
  <tr><td><strong>OpenAI-compat layer</strong></td><td>Translates the OpenAI / Anthropic / Ollama protocols into SGLang's internal requests</td><td>L15</td></tr>
  <tr><td><strong>IO structs / IPC</strong></td><td>The request/response data structures passed between processes over channels like ZMQ</td><td>L16</td></tr>
  <tr><td><strong>DetokenizerManager / streaming</strong></td><td>Incrementally decodes generated tokens into text and streams it as it goes</td><td>L17</td></tr>
</table>

<h2>4. The scheduler</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>Scheduler event loop</strong></td><td>The main loop: receive → batch → forward → emit, running forever</td><td>L18</td></tr>
  <tr><td><strong>Req / ScheduleBatch</strong></td><td>Req is one request's running state; ScheduleBatch is the batch formed for one step</td><td>L19</td></tr>
  <tr><td><strong>Schedule policy</strong></td><td>Decides who runs first: priority, fairness, longest-prefix match, etc.</td><td>L20</td></tr>
  <tr><td><strong>Zero-overhead overlap scheduler</strong></td><td>Overlaps CPU scheduling with the GPU forward to hide scheduling cost</td><td>L21</td></tr>
  <tr><td><strong>Chunked prefill</strong></td><td>Splits a long prompt's prefill into chunks, interleaved with decode, to smooth TTFT</td><td>L22</td></tr>
  <tr><td><strong>DP controller / PP scheduling</strong></td><td>Request dispatch for data parallelism and stage scheduling for pipeline parallelism</td><td>L23</td></tr>
</table>

<h2>5. Model execution</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>ModelRunner / ForwardBatch</strong></td><td>ModelRunner drives one forward; ForwardBatch is its unified batched input</td><td>L24</td></tr>
  <tr><td><strong>Weight loading</strong></td><td>Loads model weights into each rank's VRAM by shard / quantization format</td><td>L25</td></tr>
  <tr><td><strong>Writing a model</strong></td><td>Implement a new model using SGLang's layers and forward conventions</td><td>L26</td></tr>
  <tr><td><strong>CUDA graph capture/replay</strong></td><td>Record a static-shape forward as a graph; replay with near-zero launch overhead</td><td>L27</td></tr>
  <tr><td><strong>Sampler / sampling params</strong></td><td>Samples the next token from logits via temperature / top-p / top-k etc.</td><td>L28</td></tr>
</table>

<h2>6. KV cache &amp; memory</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>RadixAttention impl</strong></td><td>Indexes prefix KV in a radix tree; matches the longest common prefix and reuses its cache</td><td>L29</td></tr>
  <tr><td><strong>Paged memory pools</strong></td><td>Manage KV VRAM by page; a token→page table enables paging and sharing</td><td>L30</td></tr>
  <tr><td><strong>HiCache tiering</strong></td><td>Tier KV across GPU / CPU / disk caches, swapping in and out</td><td>L31</td></tr>
  <tr><td><strong>Eviction / hit rate</strong></td><td>LRU evicts evictable leaves, locks in-use nodes; hit rate equals throughput</td><td>L32</td></tr>
</table>

<h2>7. Attention &amp; layers</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>Attention backend abstraction</strong></td><td>Attention is a swappable strategy: FlashInfer / Triton / FA, picked by hardware</td><td>L33</td></tr>
  <tr><td><strong>The MoE layer</strong></td><td>Routes to top-k experts for sparse compute; FusedMoE fuses routing + GEMM</td><td>L34</td></tr>
  <tr><td><strong>Quantization</strong></td><td>Use fewer bits (FP8 / INT4 / AWQ / GPTQ) to save VRAM and bandwidth</td><td>L35</td></tr>
  <tr><td><strong>RoPE / norm / ops</strong></td><td>Rotary position embedding, RMSNorm and other lightweight ops, often fused</td><td>L36</td></tr>
  <tr><td><strong>Logits / vocab parallel</strong></td><td>lm_head projects hidden into vocab-sized scores; sharded along the vocab dim under TP</td><td>L37</td></tr>
</table>

<h2>8. Kernels &amp; hardware</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>sgl-kernel</strong></td><td>A standalone C++/CUDA project, AOT-compiled into a .so, exposed as torch.ops.sgl_kernel.*</td><td>L38</td></tr>
  <tr><td><strong>JIT kernel</strong></td><td>Compiles small kernels on demand at runtime and caches them — flexible, with a first-call cost</td><td>L39</td></tr>
  <tr><td><strong>Attention kernel</strong></td><td>The decode kernel gathers paged KV via the page_table, tiling + online softmax</td><td>L40</td></tr>
  <tr><td><strong>Fusion / CUDA graph</strong></td><td>Fused ops save HBM round-trips; static shapes suit graph capture — they cooperate</td><td>L41</td></tr>
  <tr><td><strong>Multi-hardware backends</strong></td><td>A platform abstraction + per-chip kernels run one engine on NVIDIA / AMD / NPU / ...</td><td>L42</td></tr>
</table>

<h2>9. Performance innovations</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>Speculative decoding</strong></td><td>A draft model proposes k tokens; the target verifies them in one pass — lossless speedup</td><td>L43</td></tr>
  <tr><td><strong>accept_rate / accept_length</strong></td><td>α (per-draft, excludes bonus) and τ (per verify step, includes bonus) measure the speedup</td><td>L43</td></tr>
  <tr><td><strong>EAGLE</strong></td><td>Drafts at feature level reusing the target hidden state; a token tree verified by tree attention</td><td>L44</td></tr>
  <tr><td><strong>PD disaggregation</strong></td><td>Split prefill and decode across pools; transfer KV across; each saturates its bottleneck</td><td>L45</td></tr>
  <tr><td><strong>TP / PP / EP / DP</strong></td><td>Tensor / pipeline / expert / data parallelism, unified under one GroupCoordinator</td><td>L46</td></tr>
  <tr><td><strong>EPLB</strong></td><td>Expert-parallel load balancer: measure expert load and periodically rebalance placement</td><td>L47</td></tr>
  <tr><td><strong>Structured outputs / jump-forward</strong></td><td>A grammar FSM masks logits to stay valid; jump-forward over fixed spans skips the model</td><td>L48</td></tr>
</table>

<h2>10. Advanced (optional)</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>Multimodal VLM</strong></td><td>The processor inserts placeholders; the encoder's embeddings are spliced in; the rest is unchanged</td><td>L49</td></tr>
  <tr><td><strong>Multi-LoRA batching</strong></td><td>One base + an adapter pool; different requests in one batch use different LoRAs via grouped GEMM</td><td>L50</td></tr>
  <tr><td><strong>RL rollout / weight sync</strong></td><td>SGLang as the RL generation engine; in-place weight updates without restart</td><td>L51</td></tr>
  <tr><td><strong>Diffusion models</strong></td><td>Iterative denoising rather than autoregression; reuses sgl-kernel / scheduler / CUDA graph</td><td>L52</td></tr>
</table>

<h2>11. Practice &amp; tooling</h2>
<table class="t">
  <tr><th>Term</th><th>One-line definition</th><th>Lesson</th></tr>
  <tr><td><strong>ServerArgs</strong></td><td>The central config dataclass; each field auto-maps to a --kebab CLI flag</td><td>L53</td></tr>
  <tr><td><strong>BenchmarkMetrics</strong></td><td>The benchmark summary: throughput + mean/median/p90/p95/p99 of TTFT/TPOT</td><td>L54</td></tr>
  <tr><td><strong>TTFT / TPOT</strong></td><td>Time To First Token (prefill) / Time Per Output Token (decode, i.e. ITL)</td><td>L54</td></tr>
  <tr><td><strong>CustomTestCase</strong></td><td>Test base class: wraps setUpClass so teardown runs on failure — no leaked ports/processes</td><td>L55</td></tr>
  <tr><td><strong>pre-commit / PR flow</strong></td><td>fork→branch→pre-commit→test→PR; touch srt then add a test, docs go under docs_new</td><td>L56</td></tr>
</table>
"""}
