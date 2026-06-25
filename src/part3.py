"""Part 3 - The frontend language. Lessons (L09-L12) for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons cover the SGLang frontend DSL (the project's namesake): the
structured-generation language (L09), the interpreter & tracer (L10), fork/join
and how it maps onto RadixAttention prefix sharing (L11), and the backend
interface / OpenAI compat that bridges to the runtime (L12).
"""

LESSON_09 = {
    "zh": r"""
<p class="lead">
前两部分讲的都是<strong>运行时（runtime）</strong>——KV 缓存、连续批处理、分页、前缀复用。但这个项目的名字
<strong>SGLang = Structured Generation Language（结构化生成语言）</strong>，落点其实在 <strong>Language</strong> 这个词上：
它首先是一门<strong>把"调大模型"写成程序</strong>的前端语言。本课就回到项目的<strong>同名起点</strong>，看清它最核心的形态——
一个用 <span class="inline">@sgl.function</span> 装饰的普通 Python 函数，函数里用 <strong>gen</strong> 留下"待填的空"、用
<strong>select</strong> 留下"多选题"，把一段多步对话<strong>写成可读、可控、可复用</strong>的程序。理解了这一课，你就握住了整个项目<strong>名字背后的那把钥匙</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  SGLang 程序就像一张<strong>填空表格 / 邮件合并模板（mail-merge）</strong>：你把<strong>固定的文字</strong>提前写死，在该让模型作答的地方
  留一个个<strong>带标签的空格</strong>。普通的空格（<strong>gen</strong>）让模型<strong>自由填写</strong>一段话；而<strong>选择题式</strong>的空格（<strong>select</strong>）
  只给几个<strong>预设选项</strong>，模型只能<strong>在选项里挑一个</strong>。你负责设计这张表的<strong>骨架与空位</strong>，模型负责<strong>把空填满</strong>——
  填完之后，每个空都贴着标签，你用 <span class="inline">s["标签"]</span> 就能<strong>把答案取出来</strong>接着用。整张表的<strong>题面是你定的，答案是模型填的</strong>，分工清清楚楚。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>SGLang 让你用普通 Python 控制流，写出"多次调用大模型"的程序</strong>。函数接收一个<strong>状态对象 s</strong>，
  你用 <span class="inline">s += sgl.user("…")</span> 往里<strong>拼提示词</strong>，用 <span class="inline">gen(...)</span> 插入<strong>生成槽</strong>，
  两次 gen 之间可以写<strong>任意 if / for / 普通 Python</strong>。这带来三重红利：① <strong>结构与控制流</strong>都在 Python 里、清晰可调试；
  ② <strong>约束解码</strong>（select、regex、json）让输出<strong>可靠可解析</strong>；③ 运行时<strong>自动复用共享前缀的 KV</strong>
  （第 7 课 RadixAttention），多步程序天然省算力。说到底，<strong>SGLang 把"调模型"从"发网络请求"升级成了"写程序"</strong>——
  你像写普通函数那样组织提示词、控制流和约束，框架则在背后把效率与可靠性<strong>一并兜住</strong>。
</div>

<h2>一个 SGLang 程序长什么样</h2>
<p>SGLang 程序的最小单元，是一个被 <span class="inline">@sgl.function</span> 装饰的<strong>普通 Python 函数</strong>，它的<strong>第一个参数永远是状态 s</strong>。
你不是"调用一次 API 拿一段文本"，而是<strong>在函数体里一步步搭建这段对话</strong>：用 <strong>role 助手</strong>（<span class="mono">system</span> /
<span class="mono">user</span> / <span class="mono">assistant</span>）拼出消息，用 <strong>gen</strong> 在需要模型作答的位置<strong>挖一个带名字的空</strong>。
函数里出现<strong>几个 gen，就是几次对大模型的调用</strong>——它们串成一个<strong>多步（multi-call）LLM 程序</strong>，gen 之间可以插入任意 Python 逻辑。</p>

<p>这正是 SGLang 与"<strong>一问一答</strong>"式聊天接口的根本区别。在传统接口里，一次请求只能"<strong>问一句、收一段</strong>"，要做多步推理就得在外面写一堆胶水代码，
手动把上一步的回答<strong>剪切、粘贴、再拼进下一次请求</strong>。而在 SGLang 里，这段多步流程<strong>本身就是一个函数</strong>：状态 s 像一条<strong>不断生长的对话录</strong>，
每写一句、每挖一个空、每取一次结果，都<strong>顺着同一个 s 往下走</strong>。于是"问—想—再问—作答"这种<strong>链式对话</strong>，读起来就像一段普通的、自上而下的程序，
而不是一堆零散的网络调用。把多步逻辑收进一个函数，既<strong>易读易测</strong>，也让框架能<strong>看清这几步之间的关系</strong>，为后面的自动优化埋下伏笔。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>拼第一段提示词</h4><p><span class="mono">s += sgl.user("问题：…")</span>——把固定文字和变量<strong>合并进状态 s</strong>，就像写表格的题面。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>第一次 gen：让模型先想</h4><p><span class="mono">s += sgl.assistant(gen("reason", max_tokens=256))</span>——挖一个名叫 <strong>reason</strong> 的空，模型填入推理过程。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>普通 Python 用中间结果</h4><p>用 <span class="mono">s["reason"]</span> 取出上一步答案，<strong>写 if/for 决定下一步问什么</strong>——控制流就是普通 Python。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>第二次 gen：得到最终答案</h4><p><span class="mono">gen("answer", stop="\n")</span>——再挖一个空收尾。两个 gen ⇒ <strong>两次调用</strong>，共享同一段前缀。</p></div></div>
</div>

<h2>核心原语：gen 与 select，以及 role 助手</h2>
<p>SGLang 的表达力集中在<strong>一小撮原语</strong>上。最常用的是 <strong>gen</strong>（自由填空）和 <strong>select</strong>（受限多选）：前者让模型<strong>自由生成</strong>，
后者把解码<strong>限制在给定选项内</strong>——这正是"约束解码"的入口，让分类、路由这类任务的输出<strong>永远合法、可直接用</strong>。配合 role 助手与多模态的
<span class="mono">image</span>，你就能搭出从纯文本到图文混合的各种程序。</p>

<table class="t">
  <tr><th>原语</th><th>作用</th><th>取结果</th></tr>
  <tr><td class="mono">gen(name, max_tokens, stop=…)</td><td>挖一个<strong>自由填空槽</strong>，模型生成任意文本填入</td><td class="mono">s["name"]</td></tr>
  <tr><td class="mono">select(name, choices=[…])</td><td><strong>受限多选</strong>：解码被限制在选项内，只能挑一个</td><td class="mono">s["name"]</td></tr>
  <tr><td class="mono">system / user / assistant</td><td><strong>role 助手</strong>：标注消息角色，拼出多轮对话</td><td>—</td></tr>
  <tr><td class="mono">image(path_or_url)</td><td><strong>多模态</strong>：把图片塞进提示，供视觉模型读取</td><td>—</td></tr>
</table>

<p>这里值得点破一个关系：<strong>select 本质上是 gen 的"受限版"</strong>。在源码里，<span class="mono">gen(...)</span> 一旦收到 <span class="mono">choices=[…]</span>，
就直接返回一个 <span class="mono">SglSelect</span> 而不是 <span class="mono">SglGen</span>——也就是说，<strong>约束（choices / regex / json_schema）</strong>是 gen 自带的能力，
让"自由填空"在需要时<strong>收紧成"按规则填空"</strong>。这一点正是 DSL 相对裸 API 的关键优势之一。</p>

<p>为什么"<strong>受限</strong>"这么重要？设想你要让模型做情感分类，只允许输出"正面/负面/中性"三个词。若用普通 gen 自由生成，模型可能回一句"<strong>我觉得这条评论整体偏正面</strong>"，
你还得写正则去抠、去归一化，稍不留神就崩。而 <span class="mono">select</span> 把<strong>解码这一步本身</strong>就限制在三个选项内——模型在每个位置只能从合法 token 里挑，
<strong>从源头上保证输出一定落在你给的集合里</strong>。同理，<span class="mono">regex</span> 能逼出合法的电话号码或日期，<span class="mono">json_schema</span> 能逼出结构完整的 JSON。
这种"<strong>把格式要求下沉到解码器</strong>"的能力，让 SGLang 程序的输出<strong>可直接喂给下游代码</strong>，省掉了裸 API 时代大量"解析 + 重试 + 容错"的脏活。</p>

<h2>为什么要 DSL，而不是手搓裸 API？</h2>
<p>同样是"调两次模型、把第一次结果喂给第二次"，用裸 HTTP / OpenAI API 手写，你得<strong>自己拼字符串、自己解析、自己管历史、自己保证格式</strong>；
而 SGLang 把这些<strong>沉到语言与运行时里</strong>，让你只写"<strong>想做什么</strong>"，把"<strong>怎么做才高效、才可靠</strong>"交给框架。</p>

<p>打个比方：裸 API 像是用<strong>汇编</strong>写程序——每一步搬运、拼接、清理都得亲力亲为，能跑，但又啰嗦又易错；而 SGLang 像是用<strong>高级语言</strong>写程序——
你声明意图，编译器/运行时替你把寄存器分配、内存复用这些<strong>又脏又关键的活</strong>干好。对 LLM 应用而言，这些"又脏又关键的活"恰恰是<strong>提示词拼接、对话历史管理、
输出格式校验、跨调用的缓存复用</strong>。把它们交给框架，你的业务代码就只剩下<strong>清爽的几行控制流和几个命名空位</strong>，既不容易写错，也更容易在团队里被别人读懂、接手、改写。</p>

<div class="cols">
  <div class="col"><h4>SGLang DSL：你写什么</h4><p>一个 <span class="mono">@sgl.function</span>，里面是<strong>普通 Python 控制流 + gen/select 空位</strong>。
  你声明"<strong>这里要填一段</strong>、那里<strong>从这几个里选</strong>"，用 <span class="mono">s["name"]</span> 取结果。<strong>结构、约束、多步逻辑</strong>都在你眼前。</p></div>
  <div class="col"><h4>裸 API：框架替你自动做什么</h4><p>① 多次调用间<strong>共享前缀的 KV 自动复用</strong>（第 7 课），无需你手动缓存；
  ② <strong>约束解码</strong>保证 select/json 输出合法；③ 调度、连续批处理、分页全在<strong>运行时</strong>透明发生。手搓 API 这些<strong>全得自己来</strong>。</p></div>
</div>

<p>第三重红利最隐蔽也最值钱：<strong>当你把程序"结构化"地写出来，框架就能看见结构、据此自动优化</strong>。多个请求若<strong>共享同一段前缀</strong>
（同一个 system 提示、同一段少样本示例），运行时会<strong>只算一次前缀、复用它的 KV</strong>；一个函数 <span class="mono">fork</span> 出多个分支时
（第 11 课），各分支也<strong>共享父节点的前缀缓存</strong>。换句话说，<strong>"写成结构化程序"本身，就是在给运行时喂优化线索</strong>——
这是裸 API 那种"一发一收、互相看不见"的调用方式永远拿不到的。</p>

<p>这一点值得再咀嚼一下。裸 API 的世界里，每次调用都是<strong>一座孤岛</strong>：服务端收到一串文本、吐出一串文本，<strong>既不知道这串文本和上一次的有何关联，
也不知道接下来还会不会复用</strong>。哪怕你连发一万条<strong>开头完全相同</strong>的请求（比如同一段冗长的 system 提示），服务端也只能<strong>把那段前缀一遍遍重新计算</strong>，
白白烧掉算力。而 SGLang 把整段多步逻辑<strong>表达成一个程序、一棵树</strong>之后，框架就拥有了"<strong>上帝视角</strong>"：它能在执行前看出"<strong>这十几个请求开头那段是一样的</strong>"、
"<strong>这几个分支是从同一个父状态长出来的</strong>"，于是<strong>把公共前缀的 KV 缓存只算一次、让大家共用</strong>。第 7 课的 RadixAttention 正是这套自动前缀复用的引擎，
而本课的"<strong>结构化写法</strong>"，就是把这台引擎<strong>喂饱所需信息</strong>的前端。<strong>你写得越结构化，框架能省的就越多</strong>——这是声明式 DSL 独有的复利。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>命名的空位 = 可取用的结果</b>：每个 gen/select 都贴着标签，填完即可用 <span class="mono">s["name"]</span> 取出，串成多步程序</div>
  <div class="cells"><span class="lab">gen("reason")</span><span class="cell hl">模型填入推理</span><span class="sep">→</span><span class="cell q">s["reason"] 取出</span></div>
  <div class="cells"><span class="lab">select("label")</span><span class="cell">正面</span><span class="cell hl">负面</span><span class="cell">中性</span><span class="sep">→</span><span class="cell q">s["label"] 只会是三者之一</span></div>
  <div class="cells"><span class="lab">gen("answer")</span><span class="cell hl">基于上两步作答</span><span class="sep">→</span><span class="cell q">s["answer"] 最终输出</span></div>
</div>

<h2>源码一瞥：gen 其实是个"延迟节点"</h2>
<p>有意思的是，<span class="mono">gen(...)</span> 被调用时<strong>并不会立刻去请求模型</strong>——它只是<strong>构造并返回一个 IR 节点</strong>
（<span class="mono">SglGen</span>，或带 choices 时的 <span class="mono">SglSelect</span>），描述"<strong>这里要生成什么、带什么约束</strong>"。
真正的执行交给<strong>解释器</strong>（第 10 课）：它顺着这些节点跑、按需调用后端（第 12 课）。下面是 <span class="mono">lang/api.py</span> 里 <span class="mono">gen</span> 的真实片段：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/api.py ::gen</span><span class="ln">gen：构造一个生成 IR 节点</span></div>
  <pre><span class="kw">def</span> gen(
    name: Optional[str] = <span class="kw">None</span>,
    max_tokens: Optional[int] = <span class="kw">None</span>,
    stop: Optional[Union[str, List[str]]] = <span class="kw">None</span>,
    <span class="cm"># …temperature / top_p / regex / json_schema / choices 等众多采样与约束参数…</span>
    choices: Optional[List[str]] = <span class="kw">None</span>,
    regex: Optional[str] = <span class="kw">None</span>,
):
    <span class="cm"># 带 choices 时，gen 自动退化成"受限多选"——即 select</span>
    <span class="kw">if</span> choices:
        <span class="kw">return</span> SglSelect(name, choices, ...)
    <span class="cm"># 否则返回一个自由生成节点，交给解释器按需执行</span>
    <span class="kw">return</span> SglGen(name, max_tokens, ..., stop, ..., regex, ...)</pre>
</div>

<p>读懂这段，你就抓住了 SGLang 前端的灵魂：<strong>你写的程序不是"立即执行的代码"，而是一棵"待执行的意图树"</strong>。
正因为是树、是声明式的，框架才能<strong>在执行前看清全局结构</strong>——哪些前缀共享、哪些分支能并行、哪些输出要约束——
再交给解释器（第 10 课）、fork/join（第 11 课）、后端（第 12 课）逐层兑现。带着"<strong>函数即程序、gen 即空位、结构即优化线索</strong>"
这三句话往下读，后面三课会把这棵意图树<strong>怎么跑起来、怎么并行、怎么落到引擎</strong>讲透。</p>

<p>最后把本课放回 Part 3 的主线：这一部分讲的是 SGLang 的<strong>前端语言</strong>，而本课是它的<strong>地基与门面</strong>——先认清"<strong>一个程序到底长什么样、由哪些原语搭成、为什么这样设计</strong>"。
你已经看到，一段多步对话被<strong>写成一个带命名空位的 Python 函数</strong>之后，既清晰好读，又给了运行时<strong>自动优化的抓手</strong>。接下来第 10 课会揭开<strong>解释器与 tracer</strong>，
讲清这棵意图树<strong>究竟是怎么被一步步执行的</strong>；第 11 课讲 <strong>fork/join</strong>，把"一个父状态分叉出多个分支、再合并"映射到 RadixAttention 的前缀共享上；
第 12 课讲<strong>后端接口与 OpenAI 兼容</strong>，把这门语言<strong>真正接到引擎</strong>上。读完这四课，你就能从"写一个 SGLang 程序"一路看到"它在 GPU 上高效跑起来"的完整链路。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>SGLang 是项目的同名起点</strong>：Structured Generation Language——一门把"调大模型"写成<strong>普通 Python 程序</strong>的前端 DSL。</li>
    <li><strong>程序形态</strong>：<span class="mono">@sgl.function</span> 装饰、首参为状态 <span class="mono">s</span>；用 <span class="mono">s += sgl.user(…)</span> 拼提示，用 <span class="mono">gen(name,…)</span> 挖生成槽，用 <span class="mono">s["name"]</span> 取结果。<strong>多个 gen = 多步 LLM 程序</strong>，其间可写任意 if/for。</li>
    <li><strong>核心原语</strong>：<span class="mono">gen</span>（自由填空）、<span class="mono">select</span>（受限多选，约束解码）、<span class="mono">system/user/assistant</span>（role）、<span class="mono">image</span>（多模态）。源码里带 <span class="mono">choices</span> 的 gen 直接返回 <span class="mono">SglSelect</span>。</li>
    <li><strong>为何用 DSL</strong>：① 控制流与结构都在 Python；② 约束解码保证输出可靠；③ 运行时<strong>自动复用共享前缀 KV</strong>（第 7 课 RadixAttention）——结构化程序天然给框架喂优化线索。</li>
    <li><strong>本质</strong>：gen 不立即执行，而是构造 IR 节点，组成一棵<strong>待执行意图树</strong>，交给解释器（第 10 课）、fork/join（第 11 课）、后端（第 12 课）逐层兑现。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
The previous parts were all about the <strong>runtime</strong> — KV cache, continuous batching, paging, prefix reuse. But the
project's name, <strong>SGLang = Structured Generation Language</strong>, really lands on the word <strong>Language</strong>:
first and foremost it is a <strong>front-end language for writing "calling an LLM" as a program</strong>. This lesson returns to
the project's <strong>namesake origin</strong> and its core shape — an ordinary Python function decorated with
<span class="inline">@sgl.function</span>, where <strong>gen</strong> leaves "blanks to fill" and <strong>select</strong> leaves
"multiple-choice blanks," turning a multi-step dialogue into a <strong>readable, controllable, reusable</strong> program.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  An SGLang program is like a <strong>fill-in-the-blank form / mail-merge template</strong>: you write the <strong>fixed text</strong>
  ahead of time and leave <strong>labeled blanks</strong> wherever the model should answer. A plain blank (<strong>gen</strong>) lets the
  model <strong>write freely</strong>; a <strong>multiple-choice</strong> blank (<strong>select</strong>) offers only a few
  <strong>preset options</strong>, so the model can only <strong>pick one</strong>. You design the form's <strong>skeleton and slots</strong>;
  the model <strong>fills the blanks</strong> — and afterward each blank carries a label, so <span class="inline">s["label"]</span>
  pulls the answer back out for reuse.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>SGLang lets you write "multi-call LLM" programs using ordinary Python control flow</strong>. The function
  takes a <strong>state object s</strong>; you <strong>build the prompt</strong> with <span class="inline">s += sgl.user("…")</span> and
  insert <strong>generation slots</strong> with <span class="inline">gen(...)</span>, with <strong>arbitrary if / for / plain Python</strong>
  between two gens. This buys three things: ① <strong>structure and control flow</strong> live in Python — clear and debuggable;
  ② <strong>constrained decoding</strong> (select, regex, json) makes outputs <strong>reliable and parseable</strong>; ③ the runtime
  <strong>automatically reuses KV across shared prefixes</strong> (Lesson 7, RadixAttention).
</div>

<h2>What an SGLang program looks like</h2>
<p>The smallest unit is an <strong>ordinary Python function</strong> decorated with <span class="inline">@sgl.function</span>, whose
<strong>first argument is always the state s</strong>. You don't "call one API and get a string" — you <strong>build the dialogue step by
step</strong> in the body: use <strong>role helpers</strong> (<span class="mono">system</span> / <span class="mono">user</span> /
<span class="mono">assistant</span>) to assemble messages, and <strong>gen</strong> to <strong>dig a named blank</strong> wherever the model
answers. <strong>However many gens appear, that's how many LLM calls happen</strong> — they chain into a <strong>multi-call</strong> program,
with arbitrary Python logic between them.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Build the first prompt</h4><p><span class="mono">s += sgl.user("Q: …")</span> — merge fixed text and variables <strong>into state s</strong>, like writing the form's question.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>First gen: let it think</h4><p><span class="mono">s += sgl.assistant(gen("reason", max_tokens=256))</span> — dig a blank named <strong>reason</strong> for the model's reasoning.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Use the result in plain Python</h4><p>Read <span class="mono">s["reason"]</span>, then <strong>write if/for to decide the next prompt</strong> — control flow is just Python.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Second gen: final answer</h4><p><span class="mono">gen("answer", stop="\n")</span> — dig one more blank. Two gens ⇒ <strong>two calls</strong>, sharing one prefix.</p></div></div>
</div>

<h2>Core primitives: gen and select, plus role helpers</h2>
<p>SGLang's expressiveness concentrates in <strong>a tiny set of primitives</strong>. The most common are <strong>gen</strong> (free fill-in)
and <strong>select</strong> (constrained choice): the former lets the model <strong>generate freely</strong>, the latter <strong>restricts decoding to
the given options</strong> — the entry point to "constrained decoding," making outputs of tasks like classification or routing
<strong>always valid and directly usable</strong>. With role helpers and the multimodal <span class="mono">image</span>, you can build anything
from pure text to mixed image-text programs.</p>

<table class="t">
  <tr><th>Primitive</th><th>What it does</th><th>Read result</th></tr>
  <tr><td class="mono">gen(name, max_tokens, stop=…)</td><td>Dig a <strong>free fill-in slot</strong>; the model generates any text</td><td class="mono">s["name"]</td></tr>
  <tr><td class="mono">select(name, choices=[…])</td><td><strong>Constrained choice</strong>: decoding restricted to options, pick one</td><td class="mono">s["name"]</td></tr>
  <tr><td class="mono">system / user / assistant</td><td><strong>Role helpers</strong>: tag message roles to build multi-turn chat</td><td>—</td></tr>
  <tr><td class="mono">image(path_or_url)</td><td><strong>Multimodal</strong>: feed an image into the prompt for a vision model</td><td>—</td></tr>
</table>

<p>One relationship is worth spelling out: <strong>select is essentially a "constrained" gen</strong>. In the source, once
<span class="mono">gen(...)</span> receives <span class="mono">choices=[…]</span> it returns an <span class="mono">SglSelect</span> rather than an
<span class="mono">SglGen</span> — meaning <strong>constraints (choices / regex / json_schema)</strong> are built into gen, tightening "free fill-in"
into "fill by rule" when needed. This is one of the DSL's key advantages over a raw API.</p>

<h2>Why a DSL instead of hand-rolling a raw API?</h2>
<p>For the same "call the model twice, feed the first result into the second," hand-writing raw HTTP / OpenAI API means you must
<strong>concatenate strings, parse, manage history, and enforce format yourself</strong>; SGLang <strong>sinks all that into the language and
runtime</strong>, so you write only "<strong>what to do</strong>" and leave "<strong>how to do it efficiently and reliably</strong>" to the framework.</p>

<div class="cols">
  <div class="col"><h4>SGLang DSL: what you write</h4><p>One <span class="mono">@sgl.function</span> with <strong>plain Python control flow + gen/select slots</strong>.
  You declare "<strong>fill here</strong>, choose from these there," and read with <span class="mono">s["name"]</span>. <strong>Structure, constraints, multi-step logic</strong> are all in plain sight.</p></div>
  <div class="col"><h4>Raw API: what the framework auto-does</h4><p>① <strong>KV across shared prefixes is reused automatically</strong> (Lesson 7), no manual caching;
  ② <strong>constrained decoding</strong> guarantees valid select/json output; ③ scheduling, continuous batching, paging all happen transparently in the <strong>runtime</strong>. With a raw API you'd do <strong>all of it yourself</strong>.</p></div>
</div>

<p>The third benefit is the subtlest and most valuable: <strong>when you write the program structurally, the framework can see the structure and
optimize from it</strong>. If multiple requests <strong>share a prefix</strong> (same system prompt, same few-shot examples), the runtime
<strong>computes the prefix once and reuses its KV</strong>; when one function <span class="mono">fork</span>s into branches (Lesson 11), the branches
<strong>share the parent's prefix cache</strong>. In other words, <strong>writing a structured program is itself feeding optimization hints to the
runtime</strong> — something the raw-API style of "fire one, read one, blind to each other" can never obtain.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>A named blank = a readable result</b>: every gen/select carries a label; once filled, read it with <span class="mono">s["name"]</span> to chain a multi-step program</div>
  <div class="cells"><span class="lab">gen("reason")</span><span class="cell hl">model fills reasoning</span><span class="sep">→</span><span class="cell q">read s["reason"]</span></div>
  <div class="cells"><span class="lab">select("label")</span><span class="cell">positive</span><span class="cell hl">negative</span><span class="cell">neutral</span><span class="sep">→</span><span class="cell q">s["label"] is one of the three</span></div>
  <div class="cells"><span class="lab">gen("answer")</span><span class="cell hl">answer from prior steps</span><span class="sep">→</span><span class="cell q">s["answer"] final output</span></div>
</div>

<h2>A peek at the source: gen is a "deferred node"</h2>
<p>Interestingly, calling <span class="mono">gen(...)</span> <strong>does not immediately hit the model</strong> — it just <strong>constructs and returns an IR
node</strong> (<span class="mono">SglGen</span>, or <span class="mono">SglSelect</span> when given choices) describing "<strong>what to generate here, with what
constraints</strong>." Actual execution is left to the <strong>interpreter</strong> (Lesson 10), which walks these nodes and calls the backend (Lesson 12)
as needed. Here is the real snippet of <span class="mono">gen</span> from <span class="mono">lang/api.py</span>:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/api.py ::gen</span><span class="ln">gen: construct a generation IR node</span></div>
  <pre><span class="kw">def</span> gen(
    name: Optional[str] = <span class="kw">None</span>,
    max_tokens: Optional[int] = <span class="kw">None</span>,
    stop: Optional[Union[str, List[str]]] = <span class="kw">None</span>,
    <span class="cm"># …temperature / top_p / regex / json_schema / choices and more sampling+constraint args…</span>
    choices: Optional[List[str]] = <span class="kw">None</span>,
    regex: Optional[str] = <span class="kw">None</span>,
):
    <span class="cm"># With choices, gen degrades to a "constrained choice" — i.e. select</span>
    <span class="kw">if</span> choices:
        <span class="kw">return</span> SglSelect(name, choices, ...)
    <span class="cm"># Otherwise return a free-generation node for the interpreter to run on demand</span>
    <span class="kw">return</span> SglGen(name, max_tokens, ..., stop, ..., regex, ...)</pre>
</div>

<p>Grasp this and you have the soul of the SGLang front end: <strong>the program you write is not "code that runs immediately" but a "tree of intent
to be executed."</strong> Precisely because it is a declarative tree, the framework can <strong>see the whole structure before execution</strong> — which
prefixes are shared, which branches can run in parallel, which outputs need constraints — and hand it to the interpreter (Lesson 10), fork/join
(Lesson 11), and backends (Lesson 12) to realize layer by layer. Carry "<strong>function is a program, gen is a blank, structure is an optimization
hint</strong>" forward, and the next three lessons will show how this intent tree <strong>runs, parallelizes, and lands on the engine</strong>.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>SGLang is the project's namesake origin</strong>: Structured Generation Language — a front-end DSL for writing "calling an LLM" as an <strong>ordinary Python program</strong>.</li>
    <li><strong>Program shape</strong>: decorated with <span class="mono">@sgl.function</span>, first arg is state <span class="mono">s</span>; build prompts with <span class="mono">s += sgl.user(…)</span>, dig generation slots with <span class="mono">gen(name,…)</span>, read with <span class="mono">s["name"]</span>. <strong>Multiple gens = a multi-call LLM program</strong> with arbitrary if/for in between.</li>
    <li><strong>Core primitives</strong>: <span class="mono">gen</span> (free fill-in), <span class="mono">select</span> (constrained choice / decoding), <span class="mono">system/user/assistant</span> (roles), <span class="mono">image</span> (multimodal). In the source a gen with <span class="mono">choices</span> returns <span class="mono">SglSelect</span>.</li>
    <li><strong>Why a DSL</strong>: ① control flow and structure live in Python; ② constrained decoding guarantees reliable output; ③ the runtime <strong>auto-reuses shared-prefix KV</strong> (Lesson 7, RadixAttention) — a structured program naturally feeds optimization hints to the framework.</li>
    <li><strong>Essence</strong>: gen doesn't execute immediately; it builds an IR node, and the nodes form a <strong>tree of intent</strong> realized by the interpreter (Lesson 10), fork/join (Lesson 11), and backends (Lesson 12).</li>
  </ul>
</div>
""",
}

LESSON_10 = {
    "zh": r"""
<p class="lead">
上一课我们看清了 SGLang 程序<strong>长什么样</strong>——一个被 <span class="inline">@sgl.function</span> 装饰、用 <strong>gen / select</strong> 留空的普通 Python 函数，
它本质是一棵<strong>"待执行的意图树"</strong>。这一课要回答的问题是：<strong>这棵树到底怎么"跑起来"？</strong>答案是 SGLang 有<strong>两种把程序变现的模式</strong>——
<strong>解释（interpret）</strong>，由 <span class="inline">StreamExecutor</span> 真刀真枪地<strong>逐步调用后端、生成文本</strong>；以及<strong>追踪（trace）</strong>，
由 <span class="inline">trace_program</span> <strong>不调用模型</strong>地<strong>符号化走一遍</strong>，只为<strong>看清结构、抽出静态前缀</strong>。理解这两种模式，你就握住了 SGLang 前端<strong>从"写出来"到"跑出来"</strong>的那条主轴。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把一个 SGLang 程序想成一张<strong>菜谱</strong>。你有两种用法：第一种是<strong>现在就照着做（解释）</strong>——一步步切菜、下锅、翻炒，每一步都<strong>真的产生了一道菜</strong>，
  做到哪一步、锅里是什么样，全是<strong>实打实的结果</strong>。第二种是<strong>先从头读一遍（追踪）</strong>——你<strong>不真的开火</strong>，只是顺着菜谱把<strong>前面那段固定步骤</strong>看清楚：
  要哪些食材、开头那几步对每次做菜都一样，于是你<strong>提前列好购物清单、把料备齐（mise en place）</strong>。读到"<strong>尝一口再决定加多少盐</strong>"这种<strong>依赖临场结果</strong>的步骤就停下——
  因为那必须真的做了才知道。<strong>解释 = 现在就做；追踪 = 先读懂形状、把能提前备的备好</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>同一个意图树，既能"执行"也能"分析"</strong>。<strong>解释模式</strong>是常规路径——<span class="inline">StreamExecutor</span> 接过函数，
  <strong>逐个表达式</strong>地驱动：遇到固定文字就拼进状态，遇到 <strong>gen / select</strong> 就<strong>调用后端</strong>（第 12 课），把模型吐出的内容<strong>流式</strong>接到文本末尾，并存进命名变量；
  两次 gen 之间的 <strong>if / for 就是普通 Python</strong>，照常运行。<strong>追踪模式</strong>是编译/分析路径——<span class="inline">trace_program</span> 用一个<strong>假后端</strong>把函数<strong>符号化</strong>走一遍，
  <strong>完全不碰模型</strong>，只为搞清<strong>程序的结构</strong>，并<strong>抽出开头那段恒定的前缀</strong>（static prefix），交给运行时<strong>预计算、缓存、共享</strong>（呼应第 7 课 RadixAttention）。
  <strong>一个回答"现在产出什么"，一个回答"这程序长什么形状"</strong>——这就是解释器与 tracer 的分工。
</div>

<h2>解释路径：StreamExecutor 如何驱动一个程序</h2>
<p>当你<strong>调用</strong>一个 <span class="inline">@sgl.function</span> 时，SGLang 会建一个 <strong>StreamExecutor</strong> 和一个 <strong>ProgramState（即你函数里的 s）</strong>。
执行器在<strong>后台线程</strong>里跑一个 worker：你函数体里每写一句 <span class="mono">s += …</span>，本质都是把一个<strong>表达式（expr）</strong>丢进它的队列，worker 再<strong>逐个 _execute</strong>。
固定文字直接<strong>拼到 text_ 末尾</strong>；碰到 <strong>gen</strong> 就调 <span class="mono">backend.generate(...)</span>，把结果<strong>接到 text_、存进 variables[name]</strong>；碰到 <strong>select</strong> 就让后端<strong>在候选里挑一个</strong>。
关键在于：<strong>解释就是"边走边产出"</strong>——文本和变量都是<strong>真实地、一步步长出来</strong>的。</p>

<p>这里有个容易被忽略但很重要的细节：StreamExecutor 把活儿放在<strong>后台线程</strong>里跑。为什么要这样？因为这给了解释器<strong>"并发"的可能</strong>——当两个 gen <strong>彼此没有数据依赖</strong>
（第二个 gen 不需要读第一个 gen 的结果）时，执行器可以<strong>同时把它们都提交给后端</strong>，而不必傻等第一个算完再算第二个。你函数里写的 <span class="mono">s += …</span> 之所以不会立刻阻塞，
正是因为它只是<strong>把表达式投进队列</strong>，真正的等待发生在你<strong>读取结果</strong>（<span class="mono">s["name"]</span>）的那一刻——这时如果结果还没好，才会<strong>阻塞等待对应的事件</strong>。
这套"<strong>提交即返回、读取才等待</strong>"的设计，正是第 11 课 fork/join 能把多分支真正<strong>并行跑起来</strong>的地基。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">程序</span><span class="name">@sgl.function 函数</span></div><div class="ld">你写的意图树：role 消息 + <span class="mono">gen/select</span> 槽 + 普通 Python 控制流。被调用时交给执行器。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">驱动</span><span class="name">StreamExecutor</span></div><div class="ld">后台线程逐个 <span class="mono">_execute</span> 表达式：固定文字直接拼接，<span class="mono">gen/select</span> 则向后端发起调用。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">执行</span><span class="name">Backend（第 12 课）</span></div><div class="ld">真正产出 token 的地方：本地运行时或 OpenAI 兼容接口；<span class="mono">gen</span> 可<strong>流式</strong>返回。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">状态</span><span class="name">ProgramState s</span></div><div class="ld">累积 <span class="mono">text_</span>（完整文本/前缀）、<span class="mono">messages_</span>（role 消息）、<span class="mono">variables</span>（命名结果，<span class="mono">s["name"]</span> 从这里取）。</div></div>
</div>

<h2>追踪路径：不调用模型，只看清形状</h2>
<p>追踪是另一条路。<span class="inline">trace_program</span> / <span class="inline">extract_prefix_by_tracing</span> 会建一个 <strong>TracerProgramState</strong> 和一个<strong>假后端（BaseBackend）</strong>，
然后<strong>符号化</strong>地把你的函数走一遍——<strong>它根本不调用模型</strong>。它要的不是"答案"，而是<strong>程序的骨架</strong>：依次出现了哪些固定文字、哪些 gen、哪些 select、哪里 fork。
最有用的产物是<strong>静态前缀（static prefix）</strong>：从头开始、<strong>直到第一个依赖模型输出的地方为止</strong>，那一整段<strong>恒定不变的开头提示</strong>。源码里就是把开头连续的 <span class="mono">SglConstantText</span> 拼起来，
<strong>一遇到 gen 这种"要真算才知道"的节点就停</strong>。抽出这段前缀，运行时就能<strong>只算一次、反复复用它的 KV</strong>（第 7 课），或据此<strong>编译/优化</strong>整个程序。</p>

<p>为什么"<strong>遇到 gen 就停</strong>"是必须的？因为静态前缀的全部价值，在于它<strong>对每一次调用都一模一样</strong>——只有这样，运行时才敢把它<strong>算一次然后到处复用</strong>。
而 gen 的输出是<strong>模型现场生成</strong>的，每次都可能不同，一旦把它算进前缀，前缀就<strong>不再恒定</strong>、复用也就无从谈起。所以 tracer 像一个<strong>谨慎的侦察兵</strong>：
它只敢把"<strong>无论如何都不会变</strong>"的那段开头标记为可复用，剩下"<strong>得看模型脸色</strong>"的部分一律留给真正的解释执行。这也解释了源码里为什么用 <span class="mono">StopTracing</span> 这类信号：
当符号化走到一个<strong>无法在不调用模型时确定</strong>的位置，追踪就<strong>体面地停下</strong>，把已经看清的结构与前缀<strong>交出来</strong>即可。</p>

<div class="cols">
  <div class="col"><h4>解释（interpret）· 常规路径</h4><p><strong>现在就执行</strong>。由 <span class="mono">StreamExecutor</span> 逐步驱动，<strong>真的调用后端</strong>生成每个 gen，可<strong>流式</strong>输出；<strong>会改变状态</strong>——text_、variables 一步步长出来。无数据依赖时还能<strong>并发跑多个 gen</strong>（为第 11 课 fork/join 铺路）。<strong>目的：产出结果。</strong></p></div>
  <div class="col"><h4>追踪（trace）· 编译/分析路径</h4><p><strong>先看懂形状</strong>。由 <span class="mono">trace_program</span> 用假后端<strong>符号化</strong>走一遍，<strong>完全不碰模型</strong>、不产出真实 token；只<strong>发现结构、抽出静态前缀</strong>，遇到依赖模型结果的节点<strong>就停</strong>。<strong>目的：理解 + 预计算共享前缀，便于缓存与优化。</strong></p></div>
</div>

<h2>两种模式到底差在哪</h2>
<p>把两条路径并排放在一张表里，差别一目了然：核心问的问题不同（"产出什么" vs "长什么样"），<strong>是否调用模型</strong>不同，<strong>有没有副作用</strong>不同，<strong>产物</strong>也不同。
记住这张表，你就不会把"跑程序"和"分析程序"混为一谈。</p>

<p>再换个角度想：解释和追踪其实是<strong>同一棵意图树的两种"读法"</strong>。同一段程序，解释器把它当成"<strong>要立刻照办的指令</strong>"，一条条执行、一步步落实；
而 tracer 把它当成"<strong>有待研究的图纸</strong>"，只看走向、不动真格。正因为 SGLang 程序是<strong>声明式</strong>的（gen 不立即执行，只构造 IR 节点），框架才有底气在<strong>真正开跑之前</strong>，
先用追踪把它<strong>端详一遍</strong>：哪段开头是恒定的、哪些请求会撞上同一段前缀、哪些分支可以并行。这种"<strong>先看后跑</strong>"的能力，是裸 API 那种"<strong>发一条、收一条、彼此看不见</strong>"的写法<strong>永远拿不到</strong>的——
这也正好接上了第 9 课的结论：<strong>把程序写成结构化的意图树，本身就是在给运行时喂优化线索</strong>。</p>

<table class="t">
  <tr><th>维度</th><th>解释 interpret</th><th>追踪 trace</th></tr>
  <tr><td>驱动者</td><td class="mono">StreamExecutor</td><td class="mono">trace_program / TracerProgramState</td></tr>
  <tr><td>调用模型？</td><td><strong>是</strong>，每个 gen/select 都打后端</td><td><strong>否</strong>，用假后端符号化走一遍</td></tr>
  <tr><td>流式输出？</td><td><strong>可以</strong>，gen 边生成边返回</td><td><strong>不</strong>，没有真实 token</td></tr>
  <tr><td>有副作用？</td><td><strong>有</strong>：text_/variables 真实生长</td><td><strong>几乎无</strong>：只记录结构</td></tr>
  <tr><td>产物</td><td>最终文本 + 命名结果 <span class="mono">s["name"]</span></td><td>程序结构 + <strong>静态前缀</strong>（可缓存/复用）</td></tr>
  <tr><td>何时用</td><td>真正运行一个程序</td><td>预计算共享前缀、编译/优化</td></tr>
</table>

<h2>状态对象 s：解释器把结果存在哪</h2>
<p>无论哪种模式，核心都绕着<strong>状态对象 s</strong>（ProgramState）转。解释模式下，<strong>StreamExecutor 就是 s 背后的"账本"</strong>：它一边累积<strong>完整文本 text_</strong>（这正是后续请求的"前缀"），
一边把<strong>role 消息</strong>记进 messages_，还把每个 gen/select 的结果<strong>按名字存进 variables</strong>。于是你在 Python 里写 <span class="mono">s["reason"]</span>，
取的就是 <span class="mono">variables["reason"]</span>。下面这张图把"<strong>一个 gen 走完，状态里多了什么</strong>"画出来：</p>

<p>把状态想成"<strong>一条不断生长的对话录</strong>"会很贴切：你每拼一句提示，<span class="mono">text_</span> 就<strong>长一截</strong>、<span class="mono">messages_</span> 就<strong>多一条</strong>；
每让模型填一个空，结果既<strong>接到 text_ 末尾</strong>（成为后续生成的上下文），又<strong>单独按名字存一份</strong>到 variables（方便你随手取用）。这就是为什么同一段程序里，
<strong>越往后的 gen，看到的前缀越长</strong>——因为前面所有的提示与回答都已经<strong>沉淀在 text_ 里</strong>了。理解了这一点，你就明白第 7 课的前缀复用为何如此自然：
多步程序、多轮对话本就<strong>共享着越来越长的开头</strong>，运行时只要把这段共享前缀的 KV<strong>缓存下来</strong>，后续每一步都能<strong>少算一大截</strong>。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>ProgramState 持有什么</b>：解释器每执行一步，就往 s 的三个篮子里塞东西——文本、消息、命名变量</div>
  <div class="cells"><span class="lab">s += sgl.user("Q…")</span><span class="cell">追加到 text_</span><span class="sep">→</span><span class="cell">messages_ 记一条 user</span></div>
  <div class="cells"><span class="lab">gen("reason")</span><span class="cell hl">后端生成文本</span><span class="sep">→</span><span class="cell">接到 text_ 末尾</span><span class="sep">+</span><span class="cell q">variables["reason"]</span></div>
  <div class="cells"><span class="lab">读 s["reason"]</span><span class="cell q">= variables["reason"]</span><span class="sep">→</span><span class="cell">普通 Python 拿去用</span></div>
</div>

<h2>看一眼源码：StreamExecutor</h2>
<p>解释器的心脏就是 <span class="mono">StreamExecutor</span>。它在<strong>后台线程</strong>里持有 backend、变量表 variables、完整文本 text_ 等字段；<span class="mono">submit</span> 把表达式投进队列，
worker 取出后 <span class="mono">_execute</span>，其中 <span class="mono">_execute_gen</span> 才是<strong>真正调用模型</strong>、把结果<strong>写回 text_ 与 variables</strong> 的地方。下面是它的真实骨架：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/interpreter.py ::StreamExecutor</span><span class="ln">解释器：逐表达式执行、把 gen 结果写回状态</span></div>
  <pre><span class="kw">class</span> StreamExecutor:
    <span class="st">&quot;&quot;&quot;A stream executor that executes SGL expressions in a background thread.&quot;&quot;&quot;</span>

    <span class="kw">def</span> __init__(self, backend, arguments, ...):
        self.backend = backend          <span class="cm"># 选定的后端（第 12 课）</span>
        self.variables = {}             <span class="cm"># 命名结果：s["name"] 从这里取</span>
        self.text_ = <span class="st">""</span>             <span class="cm"># 不断生长的完整文本 = 前缀</span>
        self.messages_ = []             <span class="cm"># OpenAI 格式的 role 消息</span>
        <span class="cm"># …后台 worker 线程消费队列里的表达式…</span>

    <span class="kw">def</span> submit(self, expr):             <span class="cm"># 把一个 gen/select 节点丢进队列</span>
        self._init_var_event(expr)
        self.queue.put(expr)

    <span class="kw">def</span> _execute_gen(self, expr):       <span class="cm"># 真正调用模型的地方</span>
        comp, meta_info = self.backend.generate(self, ...)
        self.text_ += comp              <span class="cm"># 把生成结果接到文本末尾</span>
        self.variables[expr.name] = comp  <span class="cm"># 存进命名变量，供 s["name"] 读取</span></pre>
</div>

<p>读懂这段，你就抓住了解释器的灵魂：<strong>它是一个"逐表达式消费、把结果不断累积进状态"的循环</strong>。固定文字直接拼接，gen/select 打后端、再写回 text_ 与 variables——
状态就是这样<strong>一步步长大</strong>的。而 tracer 走的是同一棵树，却把"打后端"<strong>换成了符号占位</strong>，于是能在<strong>不花一分算力</strong>的前提下看清结构、抽出前缀。
带着"<strong>解释 = 执行并累积状态；追踪 = 符号化看形状、抽前缀</strong>"这条主线往下走：第 11 课会讲解释器如何在<strong>无数据依赖</strong>时把多个分支 <strong>fork/join 并发</strong>跑，
并让它们<strong>共享父前缀的 KV</strong>（第 7 课）；第 12 课则揭开 gen 真正落地的那一层——<strong>后端接口与 OpenAI 兼容</strong>。</p>

<p>一句话收束：<strong>解释器与 tracer，是同一门语言的"执行器"与"读心术"</strong>。前者让你的程序<strong>此刻就产出结果</strong>，后者让框架<strong>提前看懂程序的形状</strong>，
把"<strong>不变的开头</strong>"挑出来缓存复用。两者合在一起，才让 SGLang 既<strong>好写好读</strong>，又<strong>跑得又快又省</strong>——这正是"结构化生成语言"四个字的底气所在。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>一棵树，两种跑法</strong>：同一个 <span class="mono">@sgl.function</span> 意图树，既能被<strong>解释</strong>（执行）也能被<strong>追踪</strong>（分析）。</li>
    <li><strong>解释 = StreamExecutor</strong>：后台线程<strong>逐表达式 _execute</strong>，固定文字拼进 text_，<span class="mono">gen/select</span> <strong>调用后端</strong>（第 12 课），结果<strong>流式</strong>接到 text_ 并存进 <span class="mono">variables[name]</span>；两次 gen 间是普通 Python。</li>
    <li><strong>追踪 = trace_program</strong>：用<strong>假后端符号化</strong>走一遍，<strong>不调用模型</strong>，发现结构并<strong>抽出静态前缀</strong>（开头恒定那段，遇 gen 即停），供运行时<strong>预计算/缓存/共享</strong>。</li>
    <li><strong>为什么要两种</strong>：解释回答"<strong>现在产出什么</strong>"；追踪回答"<strong>这程序长什么形状</strong>"——抽出共享前缀让运行时<strong>只算一次、复用 KV</strong>（第 7 课 RadixAttention），并支撑编译与优化。</li>
    <li><strong>ProgramState s 持有什么</strong>：<span class="mono">text_</span>（完整文本/前缀）、<span class="mono">messages_</span>（role 消息）、<span class="mono">variables</span>（命名结果，<span class="mono">s["name"]</span> 即从这里取）。</li>
    <li><strong>承上启下</strong>：DSL 原语（第 9 课）→ 解释/追踪（本课）→ fork/join 并发（第 11 课）→ 后端落地（第 12 课）。一句话：解释器负责把程序跑出结果，tracer 负责提前读懂程序的形状。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson we saw what an SGLang program <strong>looks like</strong> — an ordinary Python function decorated with <span class="inline">@sgl.function</span>, leaving blanks via
<strong>gen / select</strong>, essentially a <strong>"tree of intent to be executed."</strong> This lesson answers: <strong>how does that tree actually run?</strong> SGLang has
<strong>two ways to realize a program</strong> — <strong>interpret</strong>, where <span class="inline">StreamExecutor</span> really <strong>drives the backend step by step and generates text</strong>;
and <strong>trace</strong>, where <span class="inline">trace_program</span> walks the function <strong>symbolically WITHOUT calling the model</strong>, just to <strong>see the structure and extract the static prefix</strong>.
Grasp these two modes and you hold the main axis of the SGLang front end — <strong>from "written" to "run."</strong>
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of an SGLang program as a <strong>recipe</strong>. You can use it two ways. First, <strong>cook it step by step right now (interpret)</strong> — chop, sear, stir, and each step
  <strong>really produces food</strong>; how far you've gotten and what's in the pan are <strong>concrete results</strong>. Second, <strong>read it through first (trace)</strong> — you
  <strong>don't turn on the stove</strong>; you just follow the <strong>fixed leading steps</strong> to see what's needed, so you can <strong>prepare a shopping list and mise en place</strong> ahead of time.
  You stop at the first step like "<strong>taste it, then decide how much salt</strong>" that <strong>depends on a live result</strong> — that one you can only know by actually cooking.
  <strong>Interpret = do it now; trace = understand the shape first and pre-stage what you can.</strong>
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>the same tree of intent can be "executed" or "analyzed."</strong> <strong>Interpret</strong> is the normal path — <span class="inline">StreamExecutor</span> takes the function and
  drives it <strong>expression by expression</strong>: fixed text is appended to the state; a <strong>gen / select</strong> <strong>calls the backend</strong> (Lesson 12), streaming the model's output onto the text and into a named variable;
  the <strong>if / for between two gens is plain Python</strong> that runs normally. <strong>Trace</strong> is the compile/analysis path — <span class="inline">trace_program</span> walks the function <strong>symbolically with a dummy backend</strong>,
  <strong>never touching the model</strong>, just to learn the program's <strong>structure</strong> and <strong>extract the constant leading prompt</strong> (the static prefix), which the runtime can <strong>precompute, cache, and share</strong> (echoing Lesson 7, RadixAttention).
  <strong>One answers "what to produce now," the other answers "what shape is this program"</strong> — that is the division of labor between interpreter and tracer.
</div>

<h2>The interpret path: how StreamExecutor drives a program</h2>
<p>When you <strong>call</strong> an <span class="inline">@sgl.function</span>, SGLang builds a <strong>StreamExecutor</strong> and a <strong>ProgramState (the s in your function)</strong>.
The executor runs a worker in a <strong>background thread</strong>: every <span class="mono">s += …</span> in your body essentially drops an <strong>expression (expr)</strong> into its queue, and the worker <strong>_executes them one by one</strong>.
Fixed text is <strong>appended to text_</strong>; a <strong>gen</strong> calls <span class="mono">backend.generate(...)</span> and <strong>appends the result to text_ and stores it in variables[name]</strong>; a <strong>select</strong> asks the backend to <strong>pick one of the choices</strong>.
The key idea: <strong>interpreting is "produce as you go"</strong> — text and variables <strong>really grow step by step</strong>.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Program</span><span class="name">@sgl.function</span></div><div class="ld">Your tree of intent: role messages + <span class="mono">gen/select</span> slots + plain Python control flow. On call, handed to the executor.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Driver</span><span class="name">StreamExecutor</span></div><div class="ld">A background thread <span class="mono">_execute</span>s each expression: fixed text is appended, <span class="mono">gen/select</span> calls the backend.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">Execute</span><span class="name">Backend (Lesson 12)</span></div><div class="ld">Where tokens are actually produced: local runtime or OpenAI-compatible API; <span class="mono">gen</span> can return <strong>streaming</strong>.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">State</span><span class="name">ProgramState s</span></div><div class="ld">Accumulates <span class="mono">text_</span> (full text/prefix), <span class="mono">messages_</span> (role messages), <span class="mono">variables</span> (named results, read via <span class="mono">s["name"]</span>).</div></div>
</div>

<h2>The trace path: see the shape without calling the model</h2>
<p>Trace is the other road. <span class="inline">trace_program</span> / <span class="inline">extract_prefix_by_tracing</span> builds a <strong>TracerProgramState</strong> and a <strong>dummy backend (BaseBackend)</strong>,
then walks your function <strong>symbolically</strong> — <strong>it never calls the model</strong>. It wants not "answers" but the program's <strong>skeleton</strong>: which fixed texts, gens, selects, and forks appear in order.
The most useful product is the <strong>static prefix</strong>: from the start <strong>up to the first place that depends on a model output</strong>, that whole stretch of <strong>constant leading prompt</strong>. In the source it concatenates the leading run of <span class="mono">SglConstantText</span> and
<strong>stops at the first gen-like node that "only a real run could know."</strong> Extract that prefix and the runtime can <strong>compute it once and reuse its KV</strong> (Lesson 7), or <strong>compile/optimize</strong> the whole program from it.</p>

<div class="cols">
  <div class="col"><h4>Interpret · the normal path</h4><p><strong>Execute now.</strong> Driven step by step by <span class="mono">StreamExecutor</span>, it <strong>really calls the backend</strong> for each gen and can <strong>stream</strong>; it <strong>mutates state</strong> — text_ and variables grow. With no data dependency it can even <strong>run several gens concurrently</strong> (paving the way for Lesson 11 fork/join). <strong>Goal: produce results.</strong></p></div>
  <div class="col"><h4>Trace · the compile/analysis path</h4><p><strong>Understand the shape first.</strong> Driven by <span class="mono">trace_program</span> with a dummy backend, it walks <strong>symbolically</strong>, <strong>never touching the model</strong> and producing no real tokens; it only <strong>discovers structure and extracts the static prefix</strong>, <strong>stopping</strong> at any node that depends on a model result. <strong>Goal: understand + precompute shared prefixes for caching and optimization.</strong></p></div>
</div>

<h2>Exactly where the two modes differ</h2>
<p>Side by side in one table, the difference is obvious: the core question differs ("what to produce" vs "what shape"), <strong>whether the model is called</strong> differs, <strong>side effects</strong> differ, and the <strong>product</strong> differs.
Remember this table and you'll never conflate "running a program" with "analyzing a program."</p>

<table class="t">
  <tr><th>Dimension</th><th>Interpret</th><th>Trace</th></tr>
  <tr><td>Driver</td><td class="mono">StreamExecutor</td><td class="mono">trace_program / TracerProgramState</td></tr>
  <tr><td>Calls model?</td><td><strong>Yes</strong>, every gen/select hits the backend</td><td><strong>No</strong>, symbolic walk with a dummy backend</td></tr>
  <tr><td>Streams?</td><td><strong>Can</strong>, gen returns as it generates</td><td><strong>No</strong>, there are no real tokens</td></tr>
  <tr><td>Side effects?</td><td><strong>Yes</strong>: text_/variables really grow</td><td><strong>Almost none</strong>: only records structure</td></tr>
  <tr><td>Product</td><td>Final text + named results <span class="mono">s["name"]</span></td><td>Program structure + <strong>static prefix</strong> (cacheable/reusable)</td></tr>
  <tr><td>When used</td><td>Actually running a program</td><td>Precompute shared prefix, compile/optimize</td></tr>
</table>

<h2>The state object s: where the interpreter stores results</h2>
<p>Either way, everything revolves around the <strong>state object s</strong> (ProgramState). In interpret mode, <strong>StreamExecutor is the "ledger" behind s</strong>: it accumulates the <strong>full text text_</strong> (this is precisely the "prefix" for later requests),
records <strong>role messages</strong> into messages_, and stores each gen/select result <strong>by name into variables</strong>. So when you write <span class="mono">s["reason"]</span> in Python,
you're reading <span class="mono">variables["reason"]</span>. The diagram below draws "<strong>what gets added to the state when one gen completes</strong>":</p>

<div class="cellgroup">
  <div class="cg-cap"><b>What ProgramState holds</b>: each step the interpreter runs drops things into s's three baskets — text, messages, named variables</div>
  <div class="cells"><span class="lab">s += sgl.user("Q…")</span><span class="cell">append to text_</span><span class="sep">→</span><span class="cell">messages_ logs a user turn</span></div>
  <div class="cells"><span class="lab">gen("reason")</span><span class="cell hl">backend generates text</span><span class="sep">→</span><span class="cell">append to text_</span><span class="sep">+</span><span class="cell q">variables["reason"]</span></div>
  <div class="cells"><span class="lab">read s["reason"]</span><span class="cell q">= variables["reason"]</span><span class="sep">→</span><span class="cell">used in plain Python</span></div>
</div>

<h2>A peek at the source: StreamExecutor</h2>
<p>The heart of the interpreter is <span class="mono">StreamExecutor</span>. In a <strong>background thread</strong> it holds the backend, the variables table, the full text text_, and more; <span class="mono">submit</span> drops an expression into the queue,
the worker pops it and <span class="mono">_execute</span>s, and <span class="mono">_execute_gen</span> is where the model is <strong>actually called</strong> and the result is <strong>written back into text_ and variables</strong>. Here is its real skeleton:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/interpreter.py ::StreamExecutor</span><span class="ln">interpreter: execute per-expression, write gen results back to state</span></div>
  <pre><span class="kw">class</span> StreamExecutor:
    <span class="st">&quot;&quot;&quot;A stream executor that executes SGL expressions in a background thread.&quot;&quot;&quot;</span>

    <span class="kw">def</span> __init__(self, backend, arguments, ...):
        self.backend = backend          <span class="cm"># the chosen backend (Lesson 12)</span>
        self.variables = {}             <span class="cm"># named results: s["name"] reads from here</span>
        self.text_ = <span class="st">""</span>             <span class="cm"># the ever-growing full text = the prefix</span>
        self.messages_ = []             <span class="cm"># role messages in OpenAI format</span>
        <span class="cm"># …a background worker thread consumes expressions from the queue…</span>

    <span class="kw">def</span> submit(self, expr):             <span class="cm"># drop a gen/select node into the queue</span>
        self._init_var_event(expr)
        self.queue.put(expr)

    <span class="kw">def</span> _execute_gen(self, expr):       <span class="cm"># where the model is actually called</span>
        comp, meta_info = self.backend.generate(self, ...)
        self.text_ += comp              <span class="cm"># append the generated result to the text</span>
        self.variables[expr.name] = comp  <span class="cm"># store into a named variable for s["name"]</span></pre>
</div>

<p>Read this and you grasp the interpreter's soul: <strong>it is a loop that "consumes expressions one by one and keeps accumulating results into state."</strong> Fixed text is appended; gen/select hits the backend and writes back into text_ and variables —
that's how state <strong>grows step by step</strong>. The tracer walks the same tree but <strong>swaps "hit the backend" for a symbolic placeholder</strong>, so it can see the structure and extract the prefix <strong>without spending a single FLOP</strong>.
Carry "<strong>interpret = execute and accumulate state; trace = symbolically read the shape and extract the prefix</strong>" forward: Lesson 11 shows how the interpreter runs branches <strong>fork/join concurrently</strong> when there's no data dependency
and lets them <strong>share the parent prefix's KV</strong> (Lesson 7); Lesson 12 unveils where gen actually lands — the <strong>backend interface and OpenAI compatibility.</strong></p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>One tree, two ways to run</strong>: the same <span class="mono">@sgl.function</span> tree can be <strong>interpreted</strong> (executed) or <strong>traced</strong> (analyzed).</li>
    <li><strong>Interpret = StreamExecutor</strong>: a background thread <strong>_executes per expression</strong>; fixed text is appended to text_, <span class="mono">gen/select</span> <strong>calls the backend</strong> (Lesson 12), results <strong>stream</strong> onto text_ and into <span class="mono">variables[name]</span>; between two gens is plain Python.</li>
    <li><strong>Trace = trace_program</strong>: a <strong>symbolic walk with a dummy backend</strong>, <strong>never calling the model</strong>, discovering structure and <strong>extracting the static prefix</strong> (the constant leading run, stopping at the first gen), for the runtime to <strong>precompute/cache/share</strong>.</li>
    <li><strong>Why two modes</strong>: interpret answers "<strong>what to produce now</strong>"; trace answers "<strong>what shape is this program</strong>" — extracting the shared prefix lets the runtime <strong>compute it once and reuse the KV</strong> (Lesson 7, RadixAttention) and enables compilation/optimization.</li>
    <li><strong>What ProgramState s holds</strong>: <span class="mono">text_</span> (full text/prefix), <span class="mono">messages_</span> (role messages), <span class="mono">variables</span> (named results, read via <span class="mono">s["name"]</span>).</li>
    <li><strong>Where it fits</strong>: DSL primitives (Lesson 9) → interpret/trace (this lesson) → fork/join concurrency (Lesson 11) → backend landing (Lesson 12).</li>
  </ul>
</div>
""",
}

LESSON_11 = {
    "zh": r"""
<p class="lead">
上一课（第 10 课）我们看清了解释器<strong>如何把一棵意图树一步步跑起来</strong>。但真正让 SGLang 与众不同的，是它能让一个程序
<strong>从当前状态分叉出多个并行分支</strong>——这就是 <span class="inline">s.fork(n)</span>。而本课最该记住的一句话是：
<strong>前端的 fork 和运行时的 RadixAttention（第 7 课）其实是同一个想法的两半</strong>。fork 让你在<strong>程序里声明"这里要分叉"</strong>，
RadixAttention 则在<strong>引擎里把这次分叉的共享前缀只算一次、给所有分支复用</strong>。一个负责<strong>说出结构</strong>，一个负责<strong>把结构变现成省下来的算力</strong>。
理解了这层对应，你才算真正读懂了 SGLang 这个名字里"<strong>结构化（Structured）</strong>"三个字的分量：所谓结构化，不只是为了好读，更是为了让运行时<strong>看得见结构、据此省下算力</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  fork/join 就像一本<strong>"你来决定结局"的冒险书（choose-your-own-adventure）</strong>：所有读者都先读<strong>同样的前几章</strong>（共享前缀），
  读到某个岔路口，书会说"<strong>想救公主翻到第 50 页，想屠龙翻到第 80 页</strong>"——于是同一个故事<strong>分叉出好几条不同的结局</strong>。
  关键在于：<strong>共享的前几章只印一次</strong>，不会为每条支线<strong>把开头重抄一遍</strong>；每个读者只需各自读自己那条<strong>短短的后半段</strong>。
  最后你把几条结局<strong>放在一起比一比</strong>，挑出最满意的——这就是 <strong>join</strong>（把各分支的结果收回来）。fork = 在共享开头处分叉，join = 把分头得到的答案聚合回来，一分一合，干净利落。
  你可以顺着这个比喻记住全课：<strong>共享的章节只印一次</strong>对应 RadixAttention 的前缀只算一次；<strong>多条支线各读各的</strong>对应各分支独立生成；<strong>把结局摆一起比较</strong>对应 join 收集变量。整本书的"骨架共享、结局分叉、最后汇总"，就是 fork/join 的全部精神。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<span class="inline">s.fork(n)</span> 从<strong>当前状态</strong>克隆出 <strong>n 个并行续写分支</strong>，每个分支都<strong>继承到目前为止建好的同一段提示前缀</strong>，
  然后各自独立往下生成（比如采样 n 个不同答案，或并行回答 n 个子问题）；<strong>join / gather</strong> 再把每个分支<strong>新产生的命名变量收集回父状态</strong>，供你比较、投票或汇总。
  它之所以强大，<strong>不是因为"省事"，而是因为省算力</strong>：既然 n 个分支共享完全相同的前缀，RadixAttention 就<strong>把这段前缀的 KV 只算一次、让所有分支共用</strong>（第 7 课）。
  于是从一段长前缀 fork 出 8 个分支，代价约等于"<strong>1 段前缀 + 8 段很短的后缀</strong>"，而不是"<strong>8 段完整提示</strong>"。fork 因此不只是个便利函数，<strong>它就是"前缀共享"在编程模型里的样子</strong>：你用一行代码声明意图，框架替你把"少算 7 份前缀"这笔账<strong>悄悄结清</strong>。
</div>

<h2>fork 到底做了什么：从一个状态长出 N 个分支</h2>
<p>把状态 <span class="mono">s</span> 想成一条<strong>不断生长的对话录</strong>（第 9、10 课）。在某个时刻，你已经往 <span class="mono">s</span> 里拼好了一长段提示——比如系统设定、少样本示例、用户问题。
此刻调用 <span class="mono">s.fork(n)</span>，就相当于在<strong>这条对话录的当前末端</strong>，<strong>原地复制出 n 个一模一样的副本</strong>，每个副本都带着<strong>到此为止的全部前缀</strong>。
接下来，这 n 个分支<strong>各走各路</strong>：分支 0 可以用一种采样温度生成、分支 1 用另一种；或者你给每个分支<strong>追加一个不同的子问题</strong>。它们彼此独立，互不干扰。
等所有分支都生成完，你调用 <span class="mono">join</span>（默认 <span class="mono">gather_variable</span> 模式），框架就<strong>把每个分支新写下的命名变量，回填进父状态 s</strong>——
比如父状态里的某个变量会变成一个<strong>列表</strong>，装着 n 个分支各自的答案，让你一眼就能比较、投票、择优。这也解释了 <span class="mono">join</span> 默认叫 <span class="mono">gather_variable</span> 的原因——它做的正是"<strong>把散落在各分支的命名结果，按名字归拢回父状态</strong>"这件事。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>建好共享前缀</h4><p>先往 <span class="mono">s</span> 里拼好 system + few-shot + 问题——这段<strong>所有分支都将共享</strong>，是 fork 之所以省算力的根。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>s.fork(n)：原地分叉</h4><p>从当前状态克隆出 <strong>n 个并行子状态</strong>，每个都继承<strong>同一段前缀</strong>；返回一个 <span class="mono">ProgramStateGroup</span>。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>各分支独立生成</h4><p>每个分支 <span class="mono">forks[i] += gen("answer")</span> 各写各的；无数据依赖的分支可<strong>并发执行</strong>（解释器，第 10 课）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>join：把答案收回父状态</h4><p><span class="mono">forks.join()</span> 把各分支<strong>新产生的变量</strong>聚合回 <span class="mono">s</span>，于是 <span class="mono">s["answer"]</span> 成为 n 个结果的列表，可比较 / 投票 / 汇总。</p></div></div>
</div>

<h2>核心：fork 与 RadixAttention 是一个想法的两半</h2>
<p>这是全课<strong>最该划重点</strong>的地方。第 7 课讲过：RadixAttention 把所有请求的前缀<strong>组织成一棵基数树（radix tree）</strong>，<strong>公共前缀只存一份 KV、被多个请求共享</strong>。
而 fork 做的事，恰好就是<strong>在程序里主动制造出"一段公共前缀 + 多条不同后缀"的形状</strong>。换句话说：<strong>fork 是"声明端"，RadixAttention 是"兑现端"</strong>。
你在 DSL 里写下 <span class="mono">s.fork(8)</span>，等于明明白白告诉运行时："<strong>这 8 个分支的开头是完全一样的</strong>"；运行时收到这个信号，就把那段共享前缀的注意力<strong>只算一次</strong>，
8 个分支<strong>共用同一份前缀 KV</strong>，各自只为自己那段短后缀新增计算。没有 fork 的声明，运行时也能靠 RadixAttention 去<strong>事后发现</strong>前缀相同；但有了 fork，这种共享是<strong>结构上明示、零猜测</strong>的。</p>

<p>不妨把这层关系再说透一点。第 9 课讲过 SGLang 是"<strong>把调模型写成程序</strong>"，第 10 课讲过这程序其实是一棵<strong>意图树</strong>。fork 在这棵树上做的，是<strong>从某个节点长出 n 条平行的枝</strong>——而这恰好<strong>同构</strong>于 RadixAttention 维护的那棵前缀基数树：父节点是共享前缀，子节点是各分支的不同后缀。
所以你在前端写下的<strong>程序结构</strong>，几乎可以<strong>一一对应</strong>到运行时缓存里那棵树的形状。<strong>程序怎么分叉，KV 缓存就怎么共享</strong>——这就是"结构化"能换来效率的根本原因：你越是把"哪里共享、哪里发散"<strong>显式写清楚</strong>，框架能复用的就越多、要重算的就越少。这也是为什么 fork 不该被当成普通的"开几个线程"，它<strong>携带的是语义信息</strong>，告诉运行时"这几条枝同根"。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>共享前缀只算一次，分支各加短后缀</b>：高亮格子是 fork 出的所有分支<strong>共用的同一段前缀 KV</strong>（第 7 课 RadixAttention 复用），白格子才是各分支自己新算的部分</div>
  <div class="cells"><span class="lab">分支 0</span><span class="cell hl">系统</span><span class="cell hl">少样本</span><span class="cell hl">问题</span><span class="sep">→</span><span class="cell">答案 A</span></div>
  <div class="cells"><span class="lab">分支 1</span><span class="cell hl">系统</span><span class="cell hl">少样本</span><span class="cell hl">问题</span><span class="sep">→</span><span class="cell">答案 B</span></div>
  <div class="cells"><span class="lab">分支 2</span><span class="cell hl">系统</span><span class="cell hl">少样本</span><span class="cell hl">问题</span><span class="sep">→</span><span class="cell">答案 C</span></div>
  <div class="cells"><span class="lab">复用</span><span class="cell q">高亮三段 = 整棵基数树里只存一份、三分支共享的前缀 KV</span></div>
</div>

<h2>串行循环 vs fork：代价差在哪</h2>
<p>"best-of-n"这类需求，最朴素的写法是<strong>写个 for 循环跑 n 次</strong>：每次都把<strong>完整的提示</strong>重新发一遍。问题在于，这 n 次请求<strong>开头那段长前缀是一模一样的</strong>，
却被<strong>从头到尾重算了 n 遍</strong>——白白烧掉 (n−1) 份前缀算力，而且各次串行、互相看不见。换成 <span class="mono">s.fork(n)</span>，你把"<strong>共享前缀 + n 条后缀</strong>"的结构<strong>一次性声明清楚</strong>，
运行时只算<strong>一份前缀</strong>、再加 n 段短后缀，且无依赖的分支还能并发。前缀越长、n 越大，省得越多。换个角度看，串行循环和 fork 在<strong>最终结果上完全等价</strong>（都得到 n 个候选），区别<strong>纯粹在于代价</strong>：前者把"共享"这件事<strong>藏在了你看不见的地方</strong>，运行时只能把每次请求当成<strong>互不相干的孤岛</strong>，于是被迫重算前缀；后者把"共享"<strong>摆到明面上</strong>，运行时一眼看穿、只算一次。这正呼应了第 9 课那句话：<strong>写得越结构化，框架能省的就越多</strong>。fork 就是这条原则在"并行"维度上的具体兑现。</p>

<div class="cols">
  <div class="col"><h4>串行 for 循环：N × 完整提示</h4><p>循环里跑 n 次 gen，每次都把<strong>整段长前缀重新计算一遍</strong>。代价 ≈ <strong>n × (前缀 + 后缀)</strong>，前缀被重算 n 次；各次串行，运行时<strong>看不出它们开头相同</strong>。</p></div>
  <div class="col"><h4>s.fork(n)：1 × 前缀 + N × 后缀</h4><p>一次声明分叉，运行时把<strong>共享前缀的 KV 只算一次</strong>（第 7 课），n 个分支各自只补<strong>一段短后缀</strong>。代价 ≈ <strong>前缀 + n × 后缀</strong>；无依赖分支还能并发（第 10 课）。</p></div>
</div>

<h2>典型用法：fork 能干哪些活</h2>
<p>只要你的任务能写成"<strong>共享一段开头，再分头做几件事</strong>"，fork 就用得上。下面四类是最常见的模式——它们的共同点都是<strong>从同一个父前缀分叉</strong>，因此都自动吃到前缀复用的红利，几乎不用你额外操心。</p>

<table class="t">
  <tr><th>模式</th><th>怎么分叉</th><th>join 之后做什么</th></tr>
  <tr><td class="mono">best-of-n 并行采样</td><td>同一提示 fork n 个分支，各用不同随机性<strong>生成 n 个候选</strong></td><td>取回 n 个答案，<strong>打分 / 投票</strong>选最优</td></tr>
  <tr><td class="mono">branch-and-evaluate</td><td>先生成若干<strong>不同选项</strong>，每个选项一个分支</td><td>对每个选项<strong>独立评估</strong>，再聚合比较</td></tr>
  <tr><td class="mono">map over a list</td><td>对列表里<strong>每个元素开一个分支</strong>，共享同一段说明前缀</td><td>收齐<strong>每个元素的处理结果</strong>成列表</td></tr>
  <tr><td class="mono">tree-of-thought</td><td>在每个思考节点<strong>分叉出多条推理路径</strong>，可逐层再 fork</td><td>沿树<strong>探索 / 剪枝</strong>，回收最优路径</td></tr>
</table>

<p>这里再点一句并发：fork 出来的分支<strong>彼此没有数据依赖</strong>时，解释器（第 10 课）可以<strong>并发地推进它们</strong>，而不必傻等一个跑完再跑下一个。
所以 fork 同时带来<strong>两种好处</strong>：算力上靠 RadixAttention<strong>省下重复前缀</strong>，时间上靠并发<strong>把多分支叠起来跑</strong>。需要提醒的是：并发能不能真的发生，取决于分支之间<strong>有没有数据依赖</strong>——若分支 1 要用到分支 0 的结果，它就只能<strong>等</strong>；但像 best-of-n 这种各分支<strong>毫不相干</strong>的场景，就能畅快地并发。下面看源码里 fork 的真身。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/interpreter.py ::ProgramState.fork</span><span class="ln">fork：从当前状态分叉出 N 个子状态</span></div>
  <pre><span class="kw">def</span> fork(
    self,
    size: int = 1,
    position_ids_offset: Optional[List[int]] = <span class="kw">None</span>,
):
    <span class="cm"># 让底层执行器从当前状态克隆出 size 个分支——它们共享已建好的前缀</span>
    stream_executors = self.stream_executor.fork(size, position_ids_offset)
    states = [ProgramState(x) <span class="kw">for</span> x <span class="kw">in</span> stream_executors]
    <span class="cm"># 打包成一个组；稍后 group.join() 把各分支的变量收回父状态</span>
    state_group = ProgramStateGroup(states, self)
    <span class="kw">return</span> state_group</pre>
</div>

<p>短短几行，却把本课的两半串了起来：<span class="mono">stream_executor.fork(size, …)</span> 在<strong>运行时层</strong>真正完成"<strong>从当前前缀克隆出 size 个分支</strong>"——
这正是 RadixAttention 前缀树<strong>长出 size 个共享父节点的子节点</strong>的时刻；返回的 <span class="mono">ProgramStateGroup</span> 则给你 <span class="mono">join</span>，
按 <span class="mono">gather_variable</span> 模式<strong>把各分支新变量回填进父状态</strong>。一句 <span class="mono">s.fork(n)</span>，前端声明与运行时复用就此<strong>合二为一</strong>。
往后第 12 课会讲<strong>后端接口</strong>，把这套 fork/join 真正落到具体引擎上——至此 Part 3 的"语言 → 解释 → 分叉 → 落地"四步就连成了一条完整链路。读到这里，你应当能把全书前后打通：第 7 课的前缀复用是<strong>引擎的本事</strong>，本课的 fork 是<strong>语言的表达</strong>，二者一里一外、一声明一兑现，共同撑起 SGLang"<strong>又好写又高效</strong>"的承诺。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>fork 做什么</strong>：<span class="mono">s.fork(n)</span> 从<strong>当前状态</strong>分叉出 n 个并行分支，每个<strong>继承同一段已建好的前缀</strong>，再各自独立生成；<span class="mono">join</span>（gather）把各分支<strong>新命名变量收回父状态</strong>供比较 / 投票 / 汇总。</li>
    <li><strong>核心对应</strong>：fork 是"<strong>声明端</strong>"，RadixAttention（第 7 课）是"<strong>兑现端</strong>"——同一想法的两半。fork 在程序里说出"<strong>这些分支开头相同</strong>"，运行时就把共享前缀的 KV<strong>只算一次、所有分支复用</strong>。</li>
    <li><strong>代价</strong>：从长前缀 fork 出 8 个分支 ≈ <strong>1 段前缀 + 8 段短后缀</strong>，而非 8 段完整提示；串行 for 循环则会把前缀<strong>重算 n 遍</strong>。前缀越长、n 越大，省得越多。</li>
    <li><strong>典型用法</strong>：best-of-n 并行采样、branch-and-evaluate、map over a list、tree-of-thought——共同点都是<strong>共享一段开头、再分头做事</strong>。</li>
    <li><strong>并发</strong>：无数据依赖的分支可由解释器（第 10 课）<strong>并发推进</strong>。于是 fork 同时省算力（前缀复用）又省时间（并发）。源码见 <span class="mono">interpreter.py ::ProgramState.fork</span>，第 12 课接到具体后端。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson (Lesson 10) showed how the interpreter <strong>runs a tree of intent step by step</strong>. But what really sets SGLang apart is that a program can
<strong>fork multiple parallel branches off the current state</strong> — that is <span class="inline">s.fork(n)</span>. The one line to remember today:
<strong>the front-end's fork and the runtime's RadixAttention (Lesson 7) are two halves of one idea</strong>. fork lets you <strong>declare "branch here" in the program</strong>;
RadixAttention <strong>computes that branch's shared prefix once in the engine and reuses it for every branch</strong>. One <strong>states the structure</strong>, the other <strong>cashes it in as saved compute</strong>.
Grasp this correspondence and you finally understand the weight of the word <strong>Structured</strong> in SGLang's name.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  fork/join is like a <strong>choose-your-own-adventure book</strong>: every reader first reads <strong>the same opening chapters</strong> (the shared prefix); at a fork in the road the
  book says "<strong>to save the princess turn to page 50, to slay the dragon turn to page 80</strong>" — so one story <strong>branches into several different endings</strong>.
  The key: <strong>the shared opening is printed only once</strong>, never <strong>re-copied for each branch</strong>; each reader only reads their own <strong>short second half</strong>.
  Finally you <strong>lay the endings side by side</strong> and pick the best — that is <strong>join</strong> (gathering each branch's result back). fork = branch at the shared opening, join = aggregate the separately-obtained answers.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <span class="inline">s.fork(n)</span> clones <strong>n parallel continuation branches</strong> from the <strong>current state</strong>, each <strong>inheriting the same prompt prefix built so far</strong>,
  then each generates independently (e.g. n different samples, or n different sub-questions); <strong>join / gather</strong> then <strong>collects each branch's newly-captured variables back into the parent state</strong> for you to compare, vote, or aggregate.
  Its power is <strong>not convenience but saved compute</strong>: since the n branches share an identical prefix, RadixAttention <strong>computes that prefix's KV once and shares it across all branches</strong> (Lesson 7).
  So forking 8 branches off a long prefix costs roughly "<strong>1 prefix + 8 very short suffixes</strong>," not "<strong>8 full prompts</strong>." fork is therefore not just a convenience — <strong>it is what "prefix sharing" looks like in the programming model</strong>.
</div>

<h2>What fork does: N branches grown from one state</h2>
<p>Think of state <span class="mono">s</span> as an <strong>ever-growing dialogue record</strong> (Lessons 9, 10). At some moment you have built up a long prompt into <span class="mono">s</span> — system setup, few-shot examples, the user question.
Calling <span class="mono">s.fork(n)</span> now <strong>clones n identical copies at the current tail</strong> of that record, each carrying <strong>the entire prefix so far</strong>.
The n branches then <strong>go their separate ways</strong>: branch 0 may sample at one temperature, branch 1 at another; or you <strong>append a different sub-question</strong> to each. They are independent and do not interfere.
Once all branches finish, you call <span class="mono">join</span> (default <span class="mono">gather_variable</span> mode), and the framework <strong>copies each branch's newly-written named variables back into the parent state s</strong> —
e.g. a variable in the parent becomes a <strong>list</strong> holding all n branches' answers, ready to compare, vote, or pick the best.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Build the shared prefix</h4><p>First assemble system + few-shot + question into <span class="mono">s</span> — this segment is <strong>shared by all branches</strong> and is the root of fork's savings.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>s.fork(n): branch in place</h4><p>Clone <strong>n parallel sub-states</strong> from the current state, each inheriting <strong>the same prefix</strong>; returns a <span class="mono">ProgramStateGroup</span>.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Each branch generates independently</h4><p>Each branch <span class="mono">forks[i] += gen("answer")</span> writes its own; branches with no data dependency can <strong>run concurrently</strong> (interpreter, Lesson 10).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>join: gather answers back</h4><p><span class="mono">forks.join()</span> aggregates each branch's <strong>new variables</strong> back into <span class="mono">s</span>, so <span class="mono">s["answer"]</span> becomes a list of n results to compare / vote / aggregate.</p></div></div>
</div>

<h2>The core: fork and RadixAttention are two halves of one idea</h2>
<p>This is the part to <strong>highlight hardest</strong>. Lesson 7 explained: RadixAttention organizes all requests' prefixes into a <strong>radix tree</strong>, where a <strong>common prefix stores one copy of KV shared by many requests</strong>.
What fork does is exactly <strong>deliberately create a "one common prefix + many different suffixes" shape in the program</strong>. In other words: <strong>fork is the "declaration side," RadixAttention is the "redemption side."</strong>
Writing <span class="mono">s.fork(8)</span> in the DSL tells the runtime plainly: "<strong>these 8 branches have an identical opening.</strong>" The runtime takes that signal and computes the shared prefix's attention <strong>only once</strong>;
the 8 branches <strong>share one prefix KV</strong>, each adding compute only for its own short suffix. Without fork's declaration, RadixAttention could still <strong>discover after the fact</strong> that prefixes match; with fork, the sharing is <strong>structurally explicit, zero-guess</strong>.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>Shared prefix computed once, each branch adds a short suffix</b>: highlighted cells are the <strong>same prefix KV shared by all forked branches</strong> (Lesson 7 RadixAttention reuse); white cells are what each branch computes anew</div>
  <div class="cells"><span class="lab">Branch 0</span><span class="cell hl">system</span><span class="cell hl">few-shot</span><span class="cell hl">question</span><span class="sep">→</span><span class="cell">answer A</span></div>
  <div class="cells"><span class="lab">Branch 1</span><span class="cell hl">system</span><span class="cell hl">few-shot</span><span class="cell hl">question</span><span class="sep">→</span><span class="cell">answer B</span></div>
  <div class="cells"><span class="lab">Branch 2</span><span class="cell hl">system</span><span class="cell hl">few-shot</span><span class="cell hl">question</span><span class="sep">→</span><span class="cell">answer C</span></div>
  <div class="cells"><span class="lab">reuse</span><span class="cell q">the three highlighted runs = one prefix KV stored once in the radix tree, shared by all three branches</span></div>
</div>

<h2>Serial loop vs fork: where the cost differs</h2>
<p>For "best-of-n," the naïve way is a <strong>for loop running n times</strong>: each iteration resends the <strong>full prompt</strong>. The problem: those n requests share <strong>an identical long prefix</strong>,
yet it gets <strong>recomputed n times end to end</strong> — burning (n−1) prefixes of compute, and the iterations run serially, blind to one another. With <span class="mono">s.fork(n)</span>, you <strong>declare the "shared prefix + n suffixes" structure once</strong>,
the runtime computes <strong>one prefix</strong> plus n short suffixes, and independent branches can run concurrently. The longer the prefix and the larger n, the more you save.</p>

<div class="cols">
  <div class="col"><h4>Serial for loop: N × full prompt</h4><p>Run n gens in a loop, each <strong>recomputing the whole long prefix</strong>. Cost ≈ <strong>n × (prefix + suffix)</strong>, prefix recomputed n times; iterations serial, runtime <strong>can't tell their openings match</strong>.</p></div>
  <div class="col"><h4>s.fork(n): 1 × prefix + N × suffix</h4><p>Declare the fork once; the runtime computes the <strong>shared prefix KV only once</strong> (Lesson 7), n branches each add only a <strong>short suffix</strong>. Cost ≈ <strong>prefix + n × suffix</strong>; independent branches also run concurrently (Lesson 10).</p></div>
</div>

<h2>Common uses: what fork is good for</h2>
<p>Whenever a task can be phrased as "<strong>share an opening, then do several things separately</strong>," fork applies. The four patterns below are the most common — all share the trait of <strong>branching off the same parent prefix</strong>, so all automatically enjoy prefix reuse.</p>

<table class="t">
  <tr><th>Pattern</th><th>How it forks</th><th>What join does next</th></tr>
  <tr><td class="mono">best-of-n sampling</td><td>fork n branches off one prompt, each generating <strong>n candidates</strong> with different randomness</td><td>gather n answers, <strong>score / vote</strong> for the best</td></tr>
  <tr><td class="mono">branch-and-evaluate</td><td>generate several <strong>distinct options</strong>, one branch per option</td><td><strong>evaluate each option</strong> independently, then aggregate to compare</td></tr>
  <tr><td class="mono">map over a list</td><td>open <strong>one branch per element</strong>, sharing the same instruction prefix</td><td>collect <strong>each element's result</strong> into a list</td></tr>
  <tr><td class="mono">tree-of-thought</td><td>at each thinking node <strong>fork several reasoning paths</strong>, forking again layer by layer</td><td><strong>explore / prune</strong> down the tree, recover the best path</td></tr>
</table>

<p>One more word on concurrency: when forked branches <strong>have no data dependency</strong>, the interpreter (Lesson 10) can <strong>advance them concurrently</strong> rather than waiting for one to finish before the next.
So fork brings <strong>two benefits at once</strong>: compute-wise, RadixAttention <strong>saves the duplicated prefix</strong>; time-wise, concurrency <strong>overlaps the branches</strong>. Now the real fork in the source.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/interpreter.py ::ProgramState.fork</span><span class="ln">fork: branch N sub-states off the current state</span></div>
  <pre><span class="kw">def</span> fork(
    self,
    size: int = 1,
    position_ids_offset: Optional[List[int]] = <span class="kw">None</span>,
):
    <span class="cm"># Have the executor clone `size` branches from the current state — they share the built prefix</span>
    stream_executors = self.stream_executor.fork(size, position_ids_offset)
    states = [ProgramState(x) <span class="kw">for</span> x <span class="kw">in</span> stream_executors]
    <span class="cm"># Pack into a group; later group.join() copies each branch's variables back to the parent</span>
    state_group = ProgramStateGroup(states, self)
    <span class="kw">return</span> state_group</pre>
</div>

<p>A few lines, but they tie the two halves together: <span class="mono">stream_executor.fork(size, …)</span> does the real "<strong>clone size branches from the current prefix</strong>" at the <strong>runtime layer</strong> —
exactly the moment RadixAttention's prefix tree <strong>grows size children off a shared parent node</strong>; the returned <span class="mono">ProgramStateGroup</span> gives you <span class="mono">join</span>,
which in <span class="mono">gather_variable</span> mode <strong>copies each branch's new variables back into the parent state</strong>. With one <span class="mono">s.fork(n)</span>, front-end declaration and runtime reuse become <strong>one and the same</strong>.
Lesson 12 next covers the <strong>backend interface</strong>, landing this fork/join onto concrete engines — completing Part 3's chain of "language → interpret → fork → land."</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>What fork does</strong>: <span class="mono">s.fork(n)</span> branches n parallel continuations off the <strong>current state</strong>, each <strong>inheriting the same already-built prefix</strong>, then generating independently; <span class="mono">join</span> (gather) copies each branch's <strong>new named variables back into the parent</strong> for compare / vote / aggregate.</li>
    <li><strong>The core correspondence</strong>: fork is the "<strong>declaration side</strong>," RadixAttention (Lesson 7) the "<strong>redemption side</strong>" — two halves of one idea. fork states "<strong>these branches share an opening</strong>"; the runtime computes the shared prefix's KV <strong>once and reuses it for every branch</strong>.</li>
    <li><strong>Cost</strong>: forking 8 branches off a long prefix ≈ <strong>1 prefix + 8 short suffixes</strong>, not 8 full prompts; a serial for loop <strong>recomputes the prefix n times</strong>. The longer the prefix and larger n, the more you save.</li>
    <li><strong>Common uses</strong>: best-of-n sampling, branch-and-evaluate, map over a list, tree-of-thought — all share "<strong>share an opening, then do things separately</strong>."</li>
    <li><strong>Concurrency</strong>: branches with no data dependency can be <strong>advanced concurrently</strong> by the interpreter (Lesson 10). So fork saves compute (prefix reuse) and time (concurrency) at once. Source: <span class="mono">interpreter.py ::ProgramState.fork</span>; Lesson 12 lands it on concrete backends.</li>
  </ul>
</div>
""",
}

LESSON_12 = {
    "zh": r"""
<p class="lead">
从第 9 课到第 11 课，我们一直在<strong>写程序</strong>：用 DSL 描述意图（第 9 课）、让解释器把意图树跑起来（第 10 课）、用 fork/join 分叉并行（第 11 课）。
但有一个问题一直被悄悄绕开——<strong>这些 <span class="inline">gen</span>、<span class="inline">select</span>、<span class="inline">fork</span> 最终到底是谁在执行？</strong>
答案就是本课的主角：<strong>后端（Backend）</strong>。SGLang 的设计精髓在于：<strong>同一个前端程序，可以原封不动地跑在不同的后端上</strong>——
本地的 SGLang 运行时、OpenAI、Anthropic、VertexAI……程序本身<strong>完全不知道</strong>自己跑在哪个后端。这一课收束 Part 3（前端），
并在 <span class="mono">RuntimeEndpoint</span> 这个本地运行时后端上，<strong>无缝接到 Part 4 之后的运行时世界</strong>：你写的 fork/gen，从这里开始变成一次次发往调度器的 HTTP 调用。
理解这一层，你就理解了 SGLang 架构里最优雅的一笔：<strong>前端负责"怎么写"，后端负责"在哪跑"，两者通过一道抽象接口彻底分开</strong>，谁也不必迁就谁。
正因如此，你今天用本地 GPU 调试出来的程序，明天想搬到云端托管模型上做演示，几乎是零成本的——这正是"一次编写、随处运行"在大模型时代的具体体现。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  后端接口就像一只<strong>万能遥控器</strong>：同一排按钮（开机、换台、调音量）能驱动<strong>任意一台电视</strong>——客厅的、卧室的、朋友家的。
  你按"换台"时，根本不用关心面前是哪个牌子。但有个关键差别：<strong>只有你自己那台电视（本地运行时）</strong>才暴露<strong>高级功能</strong>——
  画质增强、录制、定时（对应<strong>前缀缓存、约束解码、分支共享</strong>）；而通用的别人家电视（托管 API）只认得<strong>最基础的几个键</strong>。
  遥控器的"<strong>一套按钮驱动所有电视</strong>"对应 <span class="mono">BaseBackend</span> 抽象接口；"<strong>只有自家电视有高级键</strong>"对应：本地运行时才能给你 RadixAttention 与约束解码，
  托管模型是黑盒，能不能缓存、能不能严格约束，<strong>你说了不算</strong>。记住这只遥控器，你就记住了整课：<strong>接口统一，能力分层</strong>，这正是后端设计的灵魂所在，也是本课最值得带走的一句话。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<span class="inline">BaseBackend</span> 是一层<strong>抽象接口</strong>，规定了"一个后端必须会做哪些事"——<span class="mono">generate</span>（生成）、<span class="mono">select</span>（在候选里选）、流式返回、role 消息处理等等。
  你的 DSL 程序只调用这些<strong>抽象方法</strong>，<strong>从不直接碰具体引擎</strong>。于是换后端只需一行 <span class="mono">set_default_backend(...)</span>：
  传入 <span class="mono">RuntimeEndpoint("http://localhost:30000")</span> 就跑在<strong>本地 SGLang 运行时</strong>（第 13–17 课要搭的那个 <span class="mono">/generate</span> HTTP 服务），
  传入 <span class="mono">OpenAI(...)</span> 就让<strong>同一个程序</strong>改打 OpenAI 的托管模型。但天下没有免费的午餐：<strong>本地运行时</strong>给你 RadixAttention 前缀缓存（第 7 课）、约束解码、快速批处理；
  <strong>托管后端</strong>换来的是<strong>可移植性</strong>，代价是<strong>丢掉前缀缓存与 fork 共享</strong>那些红利——那些模型是黑盒，缓存与否由服务商决定，约束解码也受限。
</div>

<h2>一、BaseBackend：让程序与引擎解耦的那道缝</h2>
<p>回想第 10 课：解释器（StreamExecutor）逐个执行意图树上的表达式，遇到固定文字就直接拼接，遇到 <span class="mono">gen</span> / <span class="mono">select</span> 就<strong>向后端发起一次调用</strong>。
这里的"后端"，正是任何继承自 <span class="mono">BaseBackend</span> 的类。<span class="mono">BaseBackend</span> 把"<strong>一个能执行 SGLang 程序的引擎</strong>"抽象成一组方法：怎么生成 token、怎么在候选项里做 <span class="mono">select</span>、
怎么流式吐字、怎么处理 system/user/assistant 这些 role。<strong>程序面对的永远是这层接口，而不是某个具体引擎</strong>。打个比方，这就像电源插座的国家标准：电器只管长一个标准插头，至于墙后接的是水电、火电还是核电，电器一概不问。这正是软件工程里最经典的"<strong>面向接口编程</strong>"：
上层不依赖下层的实现细节，只依赖一份契约。于是同一棵意图树，今天可以跑在本地 GPU 上、明天可以跑在云端 API 上，<strong>程序一个字都不用改</strong>。
这种解耦带来的好处远不止"换后端方便"：它让前端的设计者可以<strong>专心打磨表达力</strong>（怎么把复杂的多轮、分叉、约束写得优雅），让后端的工程师可以<strong>专心打磨性能</strong>（怎么把 token 算得更快），
两拨人沿着同一条接口各自演进，互不掣肘。每当 SGLang 新增一种后端（比如接入某个新厂商的托管模型），存量的程序<strong>自动获得</strong>这份能力，无需逐个改写；
反过来，每当某个程序写好，它也<strong>自动具备</strong>了在所有现有后端上运行的资格。这就是抽象接口的复利：<strong>N 个程序 × M 个后端，只需各写一次，就能两两组合</strong>。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">程序</span><span class="name">@sgl.function 意图树</span></div><div class="ld">你用 DSL 写的 <span class="mono">gen/select/fork</span>（第 9–11 课）。它<strong>只调用抽象方法</strong>，不知道自己跑在哪个后端上。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">接口</span><span class="name">BaseBackend（抽象层）</span></div><div class="ld">规定"一个后端必须会做什么"：<span class="mono">generate / select</span>、流式、role 处理。这道缝<strong>把程序和引擎解耦</strong>。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">本地</span><span class="name">RuntimeEndpoint（首选）</span></div><div class="ld">打向本地 <span class="mono">/generate</span> HTTP 服务（第 13–17 课）。<strong>独享</strong> RadixAttention 前缀缓存、约束解码、快速批处理。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">托管</span><span class="name">OpenAI / Anthropic / VertexAI …</span></div><div class="ld">让同一程序改打<strong>托管模型</strong>：可移植，但模型是黑盒，<strong>失去</strong>前缀缓存与 fork 共享红利。</div></div>
</div>

<h2>二、RuntimeEndpoint：前端落到运行时的那个接缝</h2>
<p>在所有后端里，<span class="mono">RuntimeEndpoint</span> 最关键——<strong>它就是 Part 3（前端）与 Part 4+（运行时）严丝合缝对接的那个接缝</strong>。
它指向一个本地启动的 SGLang 服务（默认 <span class="mono">http://localhost:30000</span>），把程序里每一次 <span class="mono">gen</span> / <span class="mono">fork</span> 翻译成<strong>一次 POST /generate 的 HTTP 请求</strong>，
发给我们将在第 13–17 课逐层拆解的那台服务器——入口的 TokenizerManager、大脑 Scheduler、执行的 ModelRunner。<strong>正因为请求落到了我们自己的运行时</strong>，
第 7 课的 RadixAttention 前缀缓存、第 11 课 fork 的分支共享、以及约束解码，<strong>才真正生效</strong>：你 fork 出的 8 个分支共享同一段前缀，运行时只算一次——这件事只有在 <span class="mono">RuntimeEndpoint</span> 后端上才兑现得了。
一句 <span class="mono">set_default_backend(RuntimeEndpoint("http://localhost:30000"))</span>，就把前端的全部表达力，焊接到了我们即将深入的那台引擎上。
这也解释了一个常见困惑：为什么很多 SGLang 教程开头都要你先<strong>起一个本地服务器</strong>，再跑程序？因为只有先有了那台监听 <span class="mono">/generate</span> 的运行时，<span class="mono">RuntimeEndpoint</span> 才有地方把请求打过去。
程序和服务器是<strong>两个进程</strong>：程序是客户端，运行时是服务端，中间隔着一条 HTTP 连接。把这条连接想象成一根管子——前端把意图塞进管子这头，运行时在另一头把 token 算出来再顺着管子流回。
后面第 13–17 课，我们就沿着这根管子一路走到底，看清管子另一头那台引擎的五脏六腑。</p>

<div class="flow">
  <div class="node hl"><div class="nt">程序</div><div class="nd">s += gen("answer")</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">RuntimeEndpoint</div><div class="nd">本地运行时后端</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">POST /generate</div><div class="nd">HTTP 请求 · 第 13–17 课</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Scheduler</div><div class="nd">调度 + RadixAttention（第 7 课）</div></div>
</div>

<h2>三、本地运行时 vs 托管 API：你在交易什么</h2>
<p>换后端不是免费的。<span class="mono">RuntimeEndpoint</span> 和 <span class="mono">OpenAI</span> 在程序里长得一模一样，但拿到手的<strong>能力天差地别</strong>。
核心在于：<strong>本地运行时是你能看穿、能掌控的白盒</strong>，所以第 7 课的前缀缓存、第 11 课的 fork 共享、严格的约束解码（保证输出合法 JSON / 正则）统统可用；
而<strong>托管模型是黑盒</strong>——服务商或许内部也做缓存，但你<strong>无从声明、无法依赖</strong>，约束解码通常只能退化成"在提示里求它"。下面把这笔交易摆开。
顺带提醒一个常被忽视的隐性成本：托管 API 是<strong>按 token 计费</strong>的，你 fork 出的每一个分支、每一次重试，都在真金白银地花钱；而本地运行时一旦 GPU 就绪，跑多少分支几乎是<strong>边际成本趋零</strong>的。
所以"用 fork 做 best-of-32"这种在本地稀松平常的玩法，搬到托管后端上可能就<strong>贵得离谱</strong>。选后端从来不是单看"哪个模型聪明"，而要把<strong>性能、可控性、成本、运维负担</strong>放在一起权衡。</p>

<div class="cols">
  <div class="col"><h4>本地运行时（RuntimeEndpoint）</h4><p><strong>白盒、可掌控</strong>：吃满 RadixAttention 前缀缓存（第 7 课）、fork 分支共享（第 11 课）、严格约束解码、快速批处理。代价是你得<strong>自己部署、自己管 GPU</strong>（第 13–17 课教你搭）。<strong>追求性能与可控，就用它。</strong></p></div>
  <div class="col"><h4>托管 API（OpenAI / Anthropic …）</h4><p><strong>黑盒、可移植</strong>：一行换后端就能用上顶级闭源模型，无需自备 GPU。代价是<strong>丢掉前缀缓存与 fork 共享</strong>的确定性红利，约束解码受限，且<strong>按量计费</strong>。<strong>追求省心与可移植，就用它。</strong></p></div>
</div>

<h2>四、两个方向别搞混：谁在调用谁的 OpenAI 接口</h2>
<p>"OpenAI 兼容"这个词最容易让人犯晕，因为它牵涉<strong>两个相反的方向</strong>，务必分清：</p>

<table class="t">
  <tr><th>方向</th><th>谁是客户端 / 谁是服务端</th><th>含义</th></tr>
  <tr><td class="mono">SGLang 程序 → OpenAI 后端</td><td>你的程序是客户端，OpenAI 是服务端</td><td>用 <span class="mono">OpenAI(...)</span> 后端，让<strong>你的 SGLang 程序去调用 OpenAI 的托管模型</strong></td></tr>
  <tr><td class="mono">OpenAI 客户端 → SGLang 服务器</td><td>别人的程序是客户端，你的 SGLang 是服务端</td><td>SGLang 服务器<strong>本身就兼容 OpenAI API</strong>，任何 OpenAI 客户端都能直接指过来</td></tr>
</table>

<p>换句话说：<strong>SGLang 既能当客户端去打 OpenAI（第一个方向），又能当服务端被 OpenAI 客户端打（第二个方向）</strong>。
第二个方向尤其实用——你用第 13–17 课搭好的 SGLang 服务器，对外暴露的就是标准 OpenAI 接口，于是<strong>任何现成的 OpenAI SDK、LangChain、第三方工具，改一下 base_url 就能无缝接入你的私有部署</strong>，
完全不必为 SGLang 重写一行客户端代码。一个"兼容"，两个方向，既是<strong>消费别人的模型</strong>的桥，也是<strong>把自己的服务接入整个生态</strong>的桥。
为什么这件事如此重要？因为 OpenAI 的接口格式早已成为<strong>事实标准</strong>：海量工具、框架、SDK 都围绕它构建。SGLang 服务器主动兼容这套接口，等于<strong>免费搭上了整个生态的便车</strong>——
你不需要说服任何下游工具去适配 SGLang，只要它们能调 OpenAI，就能调你。这是一种聪明的"<strong>站在巨人肩膀上</strong>"：用最小的工程代价，换来最大的互操作性。
所以当有人问"SGLang 和 OpenAI 是什么关系"，正确的回答是：<strong>它们既不是非此即彼，也不是简单替代——SGLang 可以调 OpenAI，也可以假装成 OpenAI，端看你站在桥的哪一头。</strong></p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/backend/runtime_endpoint.py ::RuntimeEndpoint</span><span class="ln">本地运行时后端：每次生成都 POST /generate</span></div>
  <pre><span class="kw">class</span> RuntimeEndpoint(BaseBackend):
    <span class="kw">def</span> __init__(self, base_url, api_key=<span class="kw">None</span>, ...):
        super().__init__()
        self.base_url = base_url
        <span class="cm"># 启动握手：先向本地运行时问清楚加载了哪个模型</span>
        res = http_request(self.base_url + <span class="st">"/get_model_info"</span>, ...)
        self.model_info = res.json()

    <span class="kw">def</span> cache_prefix(self, prefix_str):
        <span class="cm"># 把共享前缀以 max_new_tokens=0 打过去，提前喂进 RadixAttention 缓存（第 7 课）</span>
        http_request(self.base_url + <span class="st">"/generate"</span>,
            json={<span class="st">"text"</span>: prefix_str, <span class="st">"sampling_params"</span>: {<span class="st">"max_new_tokens"</span>: 0}})

    <span class="kw">def</span> generate(self, s, sampling_params):
        <span class="cm"># 程序里的一次 gen，在这里变成一次 POST /generate 的 HTTP 调用</span>
        data = {<span class="st">"text"</span>: s.text_, <span class="st">"sampling_params"</span>: {...}}
        res = http_request(self.base_url + <span class="st">"/generate"</span>, json=data)
        <span class="kw">return</span> res.json()</pre>
</div>

<p>这几行就是<strong>前端通往运行时的那扇门</strong>：构造时先 <span class="mono">/get_model_info</span> 握手确认模型；<span class="mono">cache_prefix</span> 用 <span class="mono">max_new_tokens=0</span> 把共享前缀<strong>预热进缓存</strong>，
正是 fork（第 11 课）省算力的底层机关；而 <span class="mono">generate</span> 则把程序里的每个 <span class="mono">gen</span> 老老实实翻成<strong>一次 POST /generate</strong>。
读懂它，你就读懂了 Part 3 与 Part 4 的接缝——<strong>下一课起，我们就走进这个 <span class="mono">/generate</span> 背后的服务器</strong>，看 TokenizerManager、Scheduler、ModelRunner 如何把这一次 HTTP 请求，变成 GPU 上真正的 token。
至此，前端三部曲——<strong>第 9 课怎么写、第 10 课怎么跑、第 11 课怎么分叉</strong>——就在这道后端接缝上画下句点；而它同时也是下一段旅程的起点：穿过 <span class="mono">RuntimeEndpoint</span> 这扇门，我们正式从"语言"走向"引擎"。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>统一接口</strong>：<span class="mono">BaseBackend</span> 是抽象层，规定 <span class="mono">generate/select</span>、流式、role 处理等契约。DSL 程序（第 9–11 课）只调抽象方法，<strong>不知道也不在乎</strong>自己跑在哪个后端——换后端只需一行 <span class="mono">set_default_backend(...)</span>。</li>
    <li><strong>本地运行时</strong>：<span class="mono">RuntimeEndpoint("http://localhost:30000")</span> 把 <span class="mono">gen/fork</span> 翻成 <strong>POST /generate</strong>，打向第 13–17 课要搭的服务器。<strong>只有它</strong>能给你 RadixAttention 前缀缓存（第 7 课）、fork 分支共享（第 11 课）、约束解码与快速批处理。</li>
    <li><strong>托管后端</strong>：<span class="mono">OpenAI / Anthropic / VertexAI</span> 让同一程序改打闭源模型——可移植、免自建 GPU，但<strong>丢掉前缀缓存与 fork 共享</strong>（模型是黑盒），约束解码受限，且按量计费。</li>
    <li><strong>两个方向别混</strong>：① <strong>SGLang 程序 → OpenAI 后端</strong>（你的程序调 OpenAI）；② <strong>OpenAI 客户端 → SGLang 服务器</strong>（SGLang 服务器本身兼容 OpenAI API，任何 OpenAI 客户端改 base_url 即可接入你的私有部署）。</li>
    <li><strong>承上启下</strong>：<span class="mono">RuntimeEndpoint</span> 是 Part 3（前端）焊接 Part 4+（运行时）的接缝。源码见 <span class="mono">lang/backend/runtime_endpoint.py</span>，下一课起走进 <span class="mono">/generate</span> 背后的服务器。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
From Lesson 9 through Lesson 11 we have been <strong>writing programs</strong>: describing intent in the DSL (Lesson 9), letting the interpreter run the tree of intent (Lesson 10), forking parallel branches with fork/join (Lesson 11).
But one question has been quietly dodged all along — <strong>who actually executes all these <span class="inline">gen</span>, <span class="inline">select</span>, <span class="inline">fork</span> calls?</strong>
The answer is today's star: the <strong>Backend</strong>. The heart of SGLang's design is that <strong>the same front-end program runs unchanged on different backends</strong> —
the local SGLang runtime, OpenAI, Anthropic, VertexAI… and the program itself <strong>has no idea</strong> which backend it runs on. This lesson closes Part 3 (the front-end)
and, through the <span class="mono">RuntimeEndpoint</span> local-runtime backend, <strong>snaps seamlessly onto the runtime world of Part 4+</strong>: your fork/gen become HTTP calls to the scheduler we study next.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  A backend interface is like a <strong>universal remote</strong>: one set of buttons (power, channel, volume) drives <strong>any TV</strong> — the living-room one, the bedroom one, a friend's.
  When you press "channel," you needn't care what brand sits in front of you. But there is a key difference: <strong>only your own TV (the local runtime)</strong> exposes the <strong>advanced features</strong> —
  picture enhancement, recording, scheduling (i.e. <strong>prefix cache, constrained decoding, branch sharing</strong>); a generic stranger's TV (a hosted API) only knows the <strong>basic buttons</strong>.
  "<strong>One button set drives all TVs</strong>" maps to the <span class="mono">BaseBackend</span> abstract interface; "<strong>only your own TV has the advanced keys</strong>" maps to: only the local runtime gives you RadixAttention and constrained decoding,
  while hosted models are black boxes — whether they cache, whether they strictly constrain, is <strong>not up to you</strong>. Remember this remote and you remember the whole lesson: <strong>one interface, layered capabilities</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <span class="inline">BaseBackend</span> is an <strong>abstract interface</strong> that prescribes what a backend must be able to do — <span class="mono">generate</span>, <span class="mono">select</span> (pick among candidates), streaming, role handling, and so on.
  Your DSL program calls only these <strong>abstract methods</strong> and <strong>never touches a concrete engine directly</strong>. So switching backends is one line, <span class="mono">set_default_backend(...)</span>:
  pass <span class="mono">RuntimeEndpoint("http://localhost:30000")</span> to run on the <strong>local SGLang runtime</strong> (the <span class="mono">/generate</span> HTTP server we build in Lessons 13–17),
  pass <span class="mono">OpenAI(...)</span> and the <strong>same program</strong> targets OpenAI's hosted models instead. But there is no free lunch: the <strong>local runtime</strong> gives you RadixAttention prefix caching (Lesson 7), constrained decoding, fast batching;
  a <strong>hosted backend</strong> buys you <strong>portability</strong> at the cost of <strong>losing prefix-cache and fork-sharing</strong> benefits — those models are black boxes, caching is the provider's call, and constrained decoding is limited.
</div>

<h2>1. BaseBackend: the seam that decouples program from engine</h2>
<p>Recall Lesson 10: the interpreter (StreamExecutor) executes the expressions of the intent tree one by one — fixed text is appended, and a <span class="mono">gen</span> / <span class="mono">select</span> <strong>issues a call to the backend</strong>.
That "backend" is any class inheriting from <span class="mono">BaseBackend</span>. <span class="mono">BaseBackend</span> abstracts "<strong>an engine that can execute an SGLang program</strong>" into a set of methods: how to generate tokens, how to <span class="mono">select</span> among candidates,
how to stream, how to handle system/user/assistant roles. <strong>The program always faces this interface, never a specific engine</strong>. This is the classic "<strong>program to an interface</strong>":
the upper layer depends on a contract, not on the lower layer's implementation. So the same intent tree can run on local GPUs today and a cloud API tomorrow, <strong>with not one character changed</strong>.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Program</span><span class="name">@sgl.function intent tree</span></div><div class="ld">Your DSL <span class="mono">gen/select/fork</span> (Lessons 9–11). It <strong>calls only abstract methods</strong> and has no idea which backend it runs on.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Interface</span><span class="name">BaseBackend (abstract)</span></div><div class="ld">Prescribes what a backend must do: <span class="mono">generate / select</span>, streaming, role handling. This seam <strong>decouples program from engine</strong>.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">Local</span><span class="name">RuntimeEndpoint (preferred)</span></div><div class="ld">Targets the local <span class="mono">/generate</span> HTTP server (Lessons 13–17). <strong>Exclusively</strong> gives RadixAttention prefix cache, constrained decoding, fast batching.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Hosted</span><span class="name">OpenAI / Anthropic / VertexAI …</span></div><div class="ld">Points the same program at <strong>hosted models</strong>: portable, but the model is a black box, <strong>losing</strong> prefix-cache and fork-sharing benefits.</div></div>
</div>

<h2>2. RuntimeEndpoint: the seam where the front-end lands on the runtime</h2>
<p>Among all backends, <span class="mono">RuntimeEndpoint</span> matters most — <strong>it is the seam where Part 3 (front-end) snaps precisely onto Part 4+ (runtime)</strong>.
It points at a locally launched SGLang server (default <span class="mono">http://localhost:30000</span>) and translates every <span class="mono">gen</span> / <span class="mono">fork</span> in the program into <strong>a POST /generate HTTP request</strong>,
sent to the server we dissect layer by layer in Lessons 13–17 — the TokenizerManager at the door, the Scheduler as the brain, the ModelRunner doing the forward. <strong>Precisely because the request lands on our own runtime</strong>,
the RadixAttention prefix cache (Lesson 7), fork's branch sharing (Lesson 11), and constrained decoding <strong>actually take effect</strong>: the 8 branches you fork share one prefix and the runtime computes it once — something only the <span class="mono">RuntimeEndpoint</span> backend can deliver.
One line, <span class="mono">set_default_backend(RuntimeEndpoint("http://localhost:30000"))</span>, welds the front-end's full expressiveness onto the very engine we are about to open up.</p>

<div class="flow">
  <div class="node hl"><div class="nt">Program</div><div class="nd">s += gen("answer")</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">RuntimeEndpoint</div><div class="nd">local-runtime backend</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">POST /generate</div><div class="nd">HTTP request · Lessons 13–17</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Scheduler</div><div class="nd">schedule + RadixAttention (Lesson 7)</div></div>
</div>

<h2>3. Local runtime vs hosted API: what are you trading?</h2>
<p>Switching backends is not free. <span class="mono">RuntimeEndpoint</span> and <span class="mono">OpenAI</span> look identical in the program, but the <strong>capabilities you get differ vastly</strong>.
The crux: <strong>the local runtime is a white box you can see through and control</strong>, so the prefix cache (Lesson 7), fork sharing (Lesson 11), and strict constrained decoding (guaranteeing valid JSON / regex output) are all available;
while a <strong>hosted model is a black box</strong> — the provider may cache internally, but you <strong>cannot declare or rely on it</strong>, and constrained decoding usually degrades to "begging for it in the prompt." Here is the trade laid out.</p>

<div class="cols">
  <div class="col"><h4>Local runtime (RuntimeEndpoint)</h4><p><strong>White box, controllable</strong>: full RadixAttention prefix cache (Lesson 7), fork branch sharing (Lesson 11), strict constrained decoding, fast batching. The cost: you must <strong>deploy and manage GPUs yourself</strong> (Lessons 13–17 teach you). <strong>For performance and control, use this.</strong></p></div>
  <div class="col"><h4>Hosted API (OpenAI / Anthropic …)</h4><p><strong>Black box, portable</strong>: one line switches to a top closed model, no GPUs needed. The cost: you <strong>lose the deterministic prefix-cache and fork-sharing</strong> benefits, constrained decoding is limited, and you <strong>pay per token</strong>. <strong>For convenience and portability, use this.</strong></p></div>
</div>

<h2>4. Don't mix the two directions: who calls whose OpenAI API</h2>
<p>"OpenAI-compatible" is the most confusing phrase because it involves <strong>two opposite directions</strong> — keep them straight:</p>

<table class="t">
  <tr><th>Direction</th><th>Who is client / who is server</th><th>Meaning</th></tr>
  <tr><td class="mono">SGLang program → OpenAI backend</td><td>your program is the client, OpenAI is the server</td><td>Use the <span class="mono">OpenAI(...)</span> backend so <strong>your SGLang program calls OpenAI's hosted models</strong></td></tr>
  <tr><td class="mono">OpenAI client → SGLang server</td><td>someone's program is the client, your SGLang is the server</td><td>The SGLang server <strong>is itself OpenAI-API-compatible</strong>; any OpenAI client can point straight at it</td></tr>
</table>

<p>In other words: <strong>SGLang can be a client calling OpenAI (direction one), and a server called by OpenAI clients (direction two)</strong>.
The second direction is especially handy — the SGLang server you stand up in Lessons 13–17 exposes a standard OpenAI interface, so <strong>any existing OpenAI SDK, LangChain, or third-party tool plugs into your private deployment by just changing base_url</strong>,
without rewriting a single line of client code for SGLang. One "compatibility," two directions: a bridge both to <strong>consume others' models</strong> and to <strong>plug your own service into the whole ecosystem</strong>.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/lang/backend/runtime_endpoint.py ::RuntimeEndpoint</span><span class="ln">local-runtime backend: every generate POSTs /generate</span></div>
  <pre><span class="kw">class</span> RuntimeEndpoint(BaseBackend):
    <span class="kw">def</span> __init__(self, base_url, api_key=<span class="kw">None</span>, ...):
        super().__init__()
        self.base_url = base_url
        <span class="cm"># Startup handshake: ask the local runtime which model is loaded</span>
        res = http_request(self.base_url + <span class="st">"/get_model_info"</span>, ...)
        self.model_info = res.json()

    <span class="kw">def</span> cache_prefix(self, prefix_str):
        <span class="cm"># Send the shared prefix with max_new_tokens=0 to pre-warm RadixAttention (Lesson 7)</span>
        http_request(self.base_url + <span class="st">"/generate"</span>,
            json={<span class="st">"text"</span>: prefix_str, <span class="st">"sampling_params"</span>: {<span class="st">"max_new_tokens"</span>: 0}})

    <span class="kw">def</span> generate(self, s, sampling_params):
        <span class="cm"># A gen in the program becomes one POST /generate HTTP call here</span>
        data = {<span class="st">"text"</span>: s.text_, <span class="st">"sampling_params"</span>: {...}}
        res = http_request(self.base_url + <span class="st">"/generate"</span>, json=data)
        <span class="kw">return</span> res.json()</pre>
</div>

<p>These few lines are <strong>the door from front-end to runtime</strong>: the constructor handshakes via <span class="mono">/get_model_info</span> to confirm the model; <span class="mono">cache_prefix</span> uses <span class="mono">max_new_tokens=0</span> to <strong>pre-warm the shared prefix into the cache</strong>,
the very mechanism behind fork's compute savings (Lesson 11); and <span class="mono">generate</span> faithfully translates each <span class="mono">gen</span> in the program into <strong>one POST /generate</strong>.
Understand this and you understand the seam between Part 3 and Part 4 — <strong>from the next lesson on, we step into the server behind that <span class="mono">/generate</span></strong>, watching TokenizerManager, Scheduler, and ModelRunner turn this one HTTP request into real tokens on the GPU.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>One interface</strong>: <span class="mono">BaseBackend</span> is the abstract layer prescribing the contract — <span class="mono">generate/select</span>, streaming, role handling. The DSL program (Lessons 9–11) calls only abstract methods and <strong>neither knows nor cares</strong> which backend runs it — switching is one line, <span class="mono">set_default_backend(...)</span>.</li>
    <li><strong>Local runtime</strong>: <span class="mono">RuntimeEndpoint("http://localhost:30000")</span> translates <span class="mono">gen/fork</span> into <strong>POST /generate</strong> to the server built in Lessons 13–17. <strong>Only it</strong> gives you RadixAttention prefix cache (Lesson 7), fork branch sharing (Lesson 11), constrained decoding, and fast batching.</li>
    <li><strong>Hosted backends</strong>: <span class="mono">OpenAI / Anthropic / VertexAI</span> point the same program at closed models — portable, no self-hosted GPU, but you <strong>lose prefix-cache and fork-sharing</strong> (black-box model), constrained decoding is limited, and you pay per token.</li>
    <li><strong>Two directions, don't mix</strong>: ① <strong>SGLang program → OpenAI backend</strong> (your program calls OpenAI); ② <strong>OpenAI client → SGLang server</strong> (the SGLang server is itself OpenAI-compatible; any OpenAI client plugs into your private deployment by changing base_url).</li>
    <li><strong>Bridge forward</strong>: <span class="mono">RuntimeEndpoint</span> is the seam welding Part 3 (front-end) onto Part 4+ (runtime). Source in <span class="mono">lang/backend/runtime_endpoint.py</span>; from the next lesson we step into the server behind <span class="mono">/generate</span>.</li>
  </ul>
</div>
""",
}
