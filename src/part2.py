"""Part 2 - Foundations. Lesson content (L04-L08) for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons lay the inference foundations every later part assumes: autoregressive
decoding + the KV cache (L04), continuous batching (L05), paged KV (L06),
RadixAttention prefix caching (L07, the centerpiece) and the throughput-vs-latency
tension that drives scheduling (L08).
"""

LESSON_04 = {
    "zh": r"""
<p class="lead">
上一部分我们鸟瞰了 SGLang 的全景。从这一课起，补齐"<strong>为什么需要这些设计</strong>"的前置基础。
第一块基石，是几乎所有 LLM 推理优化都绕不开的东西——<strong>KV 缓存</strong>。要讲清它，得先看懂
大模型是怎么"<strong>一个字一个字</strong>"往外蹦的：理解了自回归生成，你就会明白 KV 缓存不是锦上添花，
而是让生成"<strong>从能跑到跑得起</strong>"的关键。它简单到一句话就能说清，却又深刻到贯穿后面每一课。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  算一道很长的<strong>竖式乘法</strong>时，你会把每一步的<strong>部分积写在草稿纸上</strong>，下一步直接取用，
  而不是从头再乘一遍。KV 缓存就是大模型的这张<strong>草稿纸</strong>：把已经算过的中间结果（每个词的 Key/Value）
  存起来，之后每生成一个新词都<strong>直接复用历史、绝不重算</strong>。撕掉草稿纸（没有缓存），每写一个字都要把
  前面所有字重算一遍——慢到没法用。
</div>

<h2>自回归生成：一次只蹦一个 token</h2>
<p>大语言模型是<strong>自回归（autoregressive）</strong>的：它一次只预测<strong>下一个</strong> token，然后把这个新 token
<strong>接回输入末尾</strong>，再预测下一个，如此反复，直到蹦出一个"结束符"或达到长度上限。换句话说，模型生成一句话不是
"一口气想好"，而是<strong>一个字一个字地循环</strong>，每一步都要回头看它<strong>已经写下的全部内容</strong>。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>喂入当前序列</h4><p>把"prompt + 已生成的 token"整段作为输入。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>前向一遍</h4><p>模型输出<strong>下一个 token 的概率分布</strong>（logits）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>采样一个 token</h4><p>按温度/top-p 等策略选出下一个 token。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>接回去，循环</h4><p>把新 token 拼到序列末尾，回到第 1 步——直到结束。</p></div></div>
</div>

<p>关键在于<strong>注意力（attention）</strong>：每生成一个新 token，它都要和<strong>前面所有 token</strong>"打个招呼"
（算注意力），才知道在上下文里该说什么。于是问题来了——前面那些 token 的信息，每一步都要用，<strong>难道每步都重算吗？</strong></p>

<p>还要点破一个本质：自回归是<strong>天生串行</strong>的——第 N 个 token 必须等第 N−1 个真的被采样出来、接回输入，才能开算。
这和训练时"一整段已知文本可以并行算"完全不同（训练时用掩码一次就能算完所有位置）。<strong>推理的串行性</strong>意味着：你没法靠
"多算几个 token"把单条请求的延迟摊薄，单条请求的速度被<strong>逐 token 的循环</strong>死死卡住。既然单条快不了，工程上唯一的出路就是
"<strong>把更多请求横向拼在一起、共享每一步的固定开销</strong>"——这一脚正好踩在批处理和缓存的门槛上，也是整个 Part 2 想让你建立的直觉。</p>

<h2>没有缓存：O(t²) 的重复劳动</h2>
<p>注意力的核心是：每个 token 都会算出三样东西——<span class="inline">Q</span>（Query，我想找什么）、
<span class="inline">K</span>（Key，我能被找到的标签）、<span class="inline">V</span>（Value，我携带的信息）。
新 token 用自己的 <span class="mono">Q</span> 去和<strong>所有历史 token 的 <span class="mono">K</span></strong> 做点积，
得到"该关注谁"，再加权汇总它们的 <span class="mono">V</span>。</p>

<div class="cols">
  <div class="col"><h4>❌ 朴素做法（无缓存）</h4><p>每生成一个新 token，就把"整段序列"<strong>从头重新跑一遍</strong>：
  重算每个历史 token 的 Q/K/V 和它们之间的全部注意力。第 t 步要处理 t 个 token，代价 <span class="mono">O(t²)</span>，
  而历史 token 的 K/V <strong>明明每次都算出一样的结果</strong>——纯属浪费。</p></div>
  <div class="col"><h4>✅ 有缓存</h4><p>历史 token 的 K/V 在它们<strong>第一次生成时就固定了，永不改变</strong>。把它们
  <strong>存下来</strong>，新 token 只算<strong>自己</strong>的 Q/K/V，再拿自己的 Q 和缓存里的 K/V 做注意力。每步只需
  <span class="mono">O(t)</span>，省掉了所有重复计算。</p></div>
</div>

<p>一句话：<strong>历史 token 的 K/V 是"只读"的</strong>，算一次就能反复用。把它们缓存起来，把每步的计算从"重算整段"
压成"只算新词对历史的注意力"——这就是 KV 缓存。给个体感：续写一段 1000 token 的回答，无缓存要做约百万量级的"整段重算"，
有缓存则降到千量级的"增量"，<strong>差出三个数量级</strong>。所以 KV 缓存不是可选优化，而是让长文本生成<strong>从不可用到可用</strong>的分水岭。</p>

<h2>跟着一步 decode：缓存里到底发生了什么</h2>
<p>抽象的 O 记号不如走一遍具体的。设 prompt 是"<strong>法国的首都是</strong>"，模型要续写"<strong>巴黎</strong>"。
预填充阶段，模型把这 6 个字一次性算完，缓存里就<strong>躺好了 6 行 K/V</strong>，并吐出第 1 个输出 token"巴"。接着进入逐步解码，
每一步都是同样的三个动作——<strong>追加一行、读取全部、预测下一个</strong>：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>decode 逐步看缓存</b>：每步只新增一行 K/V，却要和缓存里<strong>全部</strong>历史做注意力（缓存只增不改）</div>
  <div class="cells"><span class="lab">prefill</span><span class="cell hl">法</span><span class="cell hl">国</span><span class="cell hl">的</span><span class="cell hl">首</span><span class="cell hl">都</span><span class="cell hl">是</span><span class="sep">→</span><span class="cell q">缓存 6 行，输出"巴"</span></div>
  <div class="cells"><span class="lab">decode①</span><span class="cell">…6 行…</span><span class="cell hl">巴</span><span class="sep">→</span><span class="cell q">追加 1 行=7 行 · 用"巴"的 Q 读 7 行 → 输出"黎"</span></div>
  <div class="cells"><span class="lab">decode②</span><span class="cell">…7 行…</span><span class="cell hl">黎</span><span class="sep">→</span><span class="cell q">追加 1 行=8 行 · 用"黎"的 Q 读 8 行 → 输出&lt;结束&gt;</span></div>
</div>

<p>看清楚两个细节。第一，<strong>每一步都只算"一个新 token"</strong>，但它要和<strong>缓存里的全部历史</strong>做注意力——
所以随着生成变长，每步要读的缓存越来越大，单步算力却很小，时间几乎都花在"<strong>把缓存从显存搬进计算单元</strong>"上，
这正是 decode "<strong>访存密集</strong>"的根源，也解释了为什么提高 decode 吞吐的关键不在算力、而在<strong>显存带宽与批大小</strong>。
第二，缓存<strong>只追加、不修改</strong>，历史行写进去就再也不动——正是这个"只读、可共享"的性质，让后面
<strong>多条请求共享同一段前缀的 KV</strong>（第 7 课 RadixAttention）能够成立。换句话说，本课讲的"缓存"不只是一次性能优化，
它的数据结构性质，直接决定了整个引擎后面能玩出多少花样。</p>

<h2>一个常被忽略的细节：GQA 让 KV 头远少于 Q 头</h2>
<p>你可能注意到上面账本里写的是"<strong>8 个 KV 头</strong>"，而这类模型明明有 32 个注意力头。这是
<strong>分组查询注意力（GQA, Grouped-Query Attention）</strong>的功劳：让多个 Q 头<strong>共享同一组 K/V 头</strong>。
对生成质量影响很小，却能把 KV 缓存直接<strong>缩到约 1/4</strong>。为什么在前置课里专门点它？因为 KV 缓存是显存大头，
而 GQA 是<strong>从模型结构上</strong>给缓存"瘦身"的办法——它和 SGLang 在<strong>工程上</strong>的省显存招式（分页、前缀复用、
量化）形成互补：一个改"每个 token 占多大"，一群改"怎么放、放几份、用几位精度"。读后面的内存管理课时，记得这两条线是
并行推进、互相叠加的。</p>

<h2>缓存怎么存：在 SGLang 里它是一个"池子"</h2>
<p>在 SGLang 运行时里，所有 token 的 K/V 不是零散地挂在各个请求上，而是统一放进一个<strong>大显存池</strong>。
以最常见的多头注意力为例，这个池子由 <span class="inline">MHATokenToKVPool</span> 管理——它在启动时按
"<strong>能放多少 token × 多少层 × 多少个 KV 头 × 每头维度</strong>"一次性开辟好 K 和 V 两块大缓冲区，之后每来一个
新 token，就往里<strong>写一行</strong>。把缓存做成"池 + 按行分配"，是后面<strong>分页（第 6 课）</strong>与
<strong>前缀复用（第 7 课）</strong>能玩出花样的物理基础。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::MHATokenToKVPool</span><span class="ln">KV 缓存池的形状</span></div>
  <pre><span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="kw">def</span> __init__(
        self,
        size: int,          <span class="cm"># 池子能容纳的 token 总数（决定并发 × 上下文上限）</span>
        page_size: int,     <span class="cm"># 分页大小（第 6 课）</span>
        dtype: torch.dtype, <span class="cm"># fp16 / bf16 / fp8…（量化越低越省）</span>
        head_num: int,      <span class="cm"># KV 头数（GQA 下远小于 Q 头数）</span>
        head_dim: int,      <span class="cm"># 每个头的维度</span>
        layer_num: int,     <span class="cm"># 模型层数——每层都有独立的 K/V</span>
        device: str,
        ...
    ):</pre>
</div>

<p>从这个构造函数就能读出缓存大小的"账本"：<strong>每个 token</strong> 占用的字节 ≈
<span class="mono">2(K和V) × layer_num × head_num × head_dim × dtype 字节数</span>。注意它<strong>不随模型参数量直接走</strong>，
而是随<strong>上下文长度</strong>线性增长——上下文越长、并发越多，缓存吃的显存越多。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>一个示意账本</b>：某 7B 模型，32 层、8 个 KV 头（GQA）、head_dim=128、fp16(2 字节)，每 token 的 KV 缓存：</div>
  <div class="cells"><span class="lab">每 token</span><span class="cell">2</span><span class="sep">×</span><span class="cell">32 层</span><span class="sep">×</span><span class="cell">8 头</span><span class="sep">×</span><span class="cell">128</span><span class="sep">×</span><span class="cell">2B</span><span class="sep">=</span><span class="cell hl">≈128 KB</span></div>
  <div class="cells"><span class="lab">2048 token</span><span class="cell">128 KB</span><span class="sep">×</span><span class="cell">2048</span><span class="sep">=</span><span class="cell hl">≈256 MB（一条请求！）</span></div>
</div>

<p>几百条并发请求，光是 KV 缓存就能吃掉几十上百 GB 显存。<strong>这就是为什么"省显存"是推理引擎的头等大事</strong>——
后面的分页、前缀复用、量化、KV 缓存量化，全是在和这本"账"较劲。</p>

<p>还有一层因果常被忽略：<strong>缓存的总容量，直接决定了能同时服务多少请求</strong>。显存池就那么大，能放下的 token 总数是固定的；
每条请求都要占走"上下文长度 × 每 token 字节"那么多缓存。于是<strong>上下文越长、并发就越少</strong>，二者在同一块显存上互相挤占。
这意味着 KV 缓存不只是性能问题，更是<strong>容量与并发的硬约束</strong>——SGLang 的调度器（第 18 课起）每一步都要回答"显存还够不够再收一条请求"，
而它能回答得多大胆，取决于缓存被管理得多省。把这条因果链记牢：<strong>缓存省一分，能并发的请求就多一分，整机吞吐就高一分</strong>。
这也是我们要花整整一个"内存管理"部分（第 29–32 课）去深挖它的原因。换个角度说，<strong>一个推理引擎的水平高低，很大程度上就是"同样的显存能塞下多少并发、缓存利用率有多高"</strong>——而这一切的起点，正是看懂本课讲的"缓存是什么、为什么省、省在哪里"。</p>

<h2>prefill 与 decode：一条请求的两个阶段</h2>
<p>有了 KV 缓存的视角，就能看清生成被天然分成<strong>两个特性迥异的阶段</strong>：</p>

<table class="t">
  <tr><th>阶段</th><th>做什么</th><th>计算特性</th><th>缓存动作</th></tr>
  <tr><td><strong>预填充 prefill</strong></td><td>把<strong>整段 prompt 一次性</strong>喂进去，并行算出所有 token 的 K/V，得到第 1 个输出 token</td><td class="mono">计算密集（大矩阵乘，GPU 吃满）</td><td>一次<strong>填满</strong>整段历史</td></tr>
  <tr><td><strong>解码 decode</strong></td><td>之后<strong>逐 token</strong> 生成，每步只算 1 个新 token</td><td class="mono">访存密集（算得少，但每步要读整个 KV 缓存）</td><td>每步<strong>追加一行</strong></td></tr>
</table>

<p>记住这条铁律：<strong>一条请求 = 一次 prefill + 很多次 decode</strong>。两个阶段的"性格"完全不同——prefill 像
"<strong>一口气读完一本书</strong>"（吞吐型、计算密集），decode 像"<strong>逐句往下写</strong>"（延迟型、访存密集）。
正因为 decode 是访存密集（每步的计算量很小，时间几乎都花在把 KV 缓存和权重从显存搬进计算单元），<strong>把多条请求
拼成一批一起做</strong>就特别划算——一次权重读取服务多条请求，GPU 不再为单条请求空等。这正是下一课<strong>连续批处理</strong>的动机；
而两阶段特性的差异，最终还催生了把它们<strong>拆到不同机器</strong>的 PD 分离（第 45 课）。</p>

<p>把这一课放回整个 Part 2 的脉络里看，会更清楚它为什么排第一。自回归 + KV 缓存定义了推理的两个"<strong>基本盘</strong>"：
<strong>串行的逐 token 循环</strong>（决定了单条延迟的下限）和<strong>只增不改、随上下文线性膨胀的缓存</strong>（决定了并发的上限）。
接下来的四课，本质都是在这两个基本盘上做文章——<strong>第 5 课</strong>用连续批处理把"串行单条"变成"并行多条"，榨干 decode 时被浪费的算力；
<strong>第 6 课</strong>用分页把"线性膨胀的缓存"装得更紧、碎片更少；<strong>第 7 课</strong>用 RadixAttention 让<strong>不同请求共享前缀的缓存</strong>，
把"只读"性质变现成真金白银的省算省存；<strong>第 8 课</strong>则把这一切放到"<strong>吞吐 vs 延迟</strong>"的天平上，告诉你这些招式各自在天平的哪一端用力。
带着"缓存=钱、并发=吞吐、串行=延迟下限"这三把尺子往下读，后面每一个设计你都能一眼看出它在省什么、换什么。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>自回归</strong>：模型一次只生成一个 token，再接回输入循环——每步都要回看全部历史。</li>
    <li><strong>KV 缓存</strong>：历史 token 的 K/V 只读不变，存下来即可复用，把每步计算从重算整段 <span class="mono">O(t²)</span> 压到 <span class="mono">O(t)</span>。</li>
    <li><strong>显存大头</strong>：缓存 ≈ 2×层×KV头×head_dim×字节/每 token，随<strong>上下文长度</strong>线性膨胀——省显存是头等大事。</li>
    <li><strong>两阶段</strong>：prefill 计算密集（一次填满），decode 访存密集（逐行追加）；一条请求 = 1 次 prefill + N 次 decode。</li>
    <li><strong>承上启下</strong>：decode 访存密集 ⇒ 批处理划算（第 5 课）；缓存按池/行管理 ⇒ 分页（第 6 课）与前缀复用（第 7 课）成为可能。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Part 1 gave you the bird's-eye view of SGLang. From here we fill in the <strong>"why these designs exist"</strong>
foundations. The first cornerstone underlies nearly every LLM-inference optimization: the <strong>KV cache</strong>.
To explain it we first need to see how a model emits text <strong>one token at a time</strong>: once you understand
autoregressive generation, you'll see the KV cache isn't a nice-to-have — it's what makes generation
<strong>go from "possible" to "practical."</strong>
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  When you do a long piece of <strong>long-hand multiplication</strong>, you <strong>write the partial products on
  scratch paper</strong> and reuse them on the next step, instead of multiplying from scratch again. The KV cache is the
  model's scratch paper: it <strong>stores the intermediate results</strong> (each word's Key/Value) so every new word
  <strong>reuses history and never recomputes it</strong>. Throw the scratch paper away (no cache) and writing each new
  letter means redoing all the previous ones — too slow to be usable.
</div>

<h2>Autoregressive generation: one token at a time</h2>
<p>An LLM is <strong>autoregressive</strong>: it predicts only the <strong>next</strong> token, <strong>appends it to the
input</strong>, predicts the next, and repeats until it emits an end token or hits a length limit. So the model doesn't
"think the whole sentence up at once" — it loops <strong>one token at a time</strong>, and every step looks back at
<strong>everything it has written so far</strong>.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Feed the current sequence</h4><p>Input = "prompt + already-generated tokens".</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>One forward pass</h4><p>The model outputs a <strong>probability distribution for the next token</strong> (logits).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Sample one token</h4><p>Pick the next token by temperature/top-p, etc.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Append, loop</h4><p>Append the new token to the sequence; back to step 1 — until done.</p></div></div>
</div>

<p>The crux is <strong>attention</strong>: each new token must "greet" (attend to) <strong>all previous tokens</strong> to
know what to say in context. Which raises the question — that history is needed every single step, so <strong>do we recompute
it every step?</strong></p>

<p>One more essential point: autoregression is <strong>inherently serial</strong> — token N can't start until token N−1 has actually
been sampled and appended. This is unlike training, where a whole known text is computed in parallel (a causal mask scores all
positions at once). The <strong>serial nature of inference</strong> means you cannot amortize a single request's latency by
"computing a few more tokens"; one request's speed is hard-capped by the <strong>per-token loop</strong>. Since a single request
can't be sped up, the only engineering way out is to "<strong>pack more requests side by side and share each step's fixed cost</strong>"
— which lands squarely on the doorstep of batching and caching, the intuition this whole Part 2 is built to give you.</p>

<h2>Without a cache: O(t²) of repeated work</h2>
<p>Attention has each token produce three things: <span class="inline">Q</span> (Query — what I'm looking for),
<span class="inline">K</span> (Key — the label I can be found by), <span class="inline">V</span> (Value — the information I
carry). A new token uses its <span class="mono">Q</span> against <strong>all historical tokens' <span class="mono">K</span></strong>
to decide "whom to attend to", then sums their <span class="mono">V</span>.</p>

<div class="cols">
  <div class="col"><h4>❌ Naive (no cache)</h4><p>For every new token, <strong>re-run the whole sequence from scratch</strong>:
  recompute every historical token's Q/K/V and all the attention among them. Step t costs <span class="mono">O(t²)</span>,
  and the historical K/V <strong>come out identical every time</strong> — pure waste.</p></div>
  <div class="col"><h4>✅ With a cache</h4><p>Historical tokens' K/V are <strong>fixed the moment they were first generated</strong>.
  Store them; a new token computes only <strong>its own</strong> Q/K/V, then attends its Q against the cached K/V. Each step is
  just <span class="mono">O(t)</span>, dropping all the repeated work.</p></div>
</div>

<p>In one line: <strong>historical K/V are "read-only"</strong>, computed once and reused. Caching them squeezes each step
from "recompute the whole sequence" to "just the new word's attention over history" — that is the KV cache.</p>

<h2>Follow one decode step: what happens inside the cache</h2>
<p>Abstract big-O is less convincing than a concrete trace. Let the prompt be "<strong>The capital of France is</strong>" and the
model continue with "<strong>Paris</strong>". In prefill, the model computes all prompt tokens at once, so the cache holds
<strong>6 rows of K/V</strong> and emits the 1st output token "Par". Then step-by-step decode begins, each step doing the same
three actions — <strong>append a row, read everything, predict the next</strong>:</p>

<div class="cellgroup">
  <div class="cg-cap"><b>decode, cache step by step</b>: each step adds one K/V row but attends to <strong>all</strong> history (the cache only grows, never changes)</div>
  <div class="cells"><span class="lab">prefill</span><span class="cell hl">The</span><span class="cell hl">capital</span><span class="cell hl">of</span><span class="cell hl">France</span><span class="cell hl">is</span><span class="sep">→</span><span class="cell q">cache 5 rows, output "Par"</span></div>
  <div class="cells"><span class="lab">decode①</span><span class="cell">…5 rows…</span><span class="cell hl">Par</span><span class="sep">→</span><span class="cell q">append 1 = 6 rows · "Par" Q reads 6 rows → "is"</span></div>
  <div class="cells"><span class="lab">decode②</span><span class="cell">…6 rows…</span><span class="cell hl">is</span><span class="sep">→</span><span class="cell q">append 1 = 7 rows · "is" Q reads 7 rows → &lt;eos&gt;</span></div>
</div>

<p>Notice two details. First, <strong>each step computes only "one new token"</strong>, yet it attends to <strong>all of the cached
history</strong> — so as generation grows, each step reads an ever-larger cache while doing tiny compute, spending its time
"<strong>moving the cache from HBM into the compute units</strong>". That is the root of decode being <strong>memory-bound</strong>,
and why the key to decode throughput isn't FLOPs but <strong>memory bandwidth and batch size</strong>. Second, the cache is
<strong>append-only, never modified</strong> — once a history row is written it never moves. It is exactly this "read-only,
shareable" property that lets later requests <strong>share the same prefix's KV</strong> (Lesson 7, RadixAttention). In other
words, the "cache" here is not just a one-off optimization; its data-structure properties directly decide how many tricks the
whole engine can pull off downstream.</p>

<h2>An often-missed detail: GQA makes KV heads far fewer than Q heads</h2>
<p>You may have noticed the ledger said "<strong>8 KV heads</strong>" while such models have 32 attention heads. That's
<strong>Grouped-Query Attention (GQA)</strong>: many Q heads <strong>share one group of K/V heads</strong>. It barely affects
quality yet shrinks the KV cache to about <strong>1/4</strong>. Why call it out in a foundations lesson? Because the KV cache is
the HBM dominator, and GQA is the <strong>architectural</strong> way to slim it — complementary to SGLang's <strong>engineering</strong>
HBM tricks (paging, prefix reuse, quantization): one changes "how big each token is", the others change "how to lay it out, how
many copies, how many bits". When you read the memory-management lessons, remember these two lines advance in parallel and stack.</p>

<h2>How it's stored: a "pool" in SGLang</h2>
<p>In the SGLang runtime, all tokens' K/V don't hang off individual requests — they go into one big <strong>HBM pool</strong>.
For ordinary multi-head attention that pool is managed by <span class="inline">MHATokenToKVPool</span>: at startup it carves
out two big buffers for K and V sized by "<strong>how many tokens × layers × KV heads × head_dim</strong>", and every new token
<strong>writes one row</strong>. Making the cache a "pool + per-row allocation" is the physical basis for the tricks in
<strong>paging (Lesson 6)</strong> and <strong>prefix reuse (Lesson 7)</strong>.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::MHATokenToKVPool</span><span class="ln">shape of the KV pool</span></div>
  <pre><span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="kw">def</span> __init__(
        self,
        size: int,          <span class="cm"># total tokens the pool holds (concurrency × context cap)</span>
        page_size: int,     <span class="cm"># paging granularity (Lesson 6)</span>
        dtype: torch.dtype, <span class="cm"># fp16 / bf16 / fp8… (lower = cheaper)</span>
        head_num: int,      <span class="cm"># KV heads (far fewer than Q heads under GQA)</span>
        head_dim: int,      <span class="cm"># per-head dimension</span>
        layer_num: int,     <span class="cm"># layers — each has its own K/V</span>
        device: str,
        ...
    ):</pre>
</div>

<p>The constructor reveals the cache's "ledger": <strong>per token</strong> the bytes ≈
<span class="mono">2(K and V) × layer_num × head_num × head_dim × dtype-bytes</span>. Note it <strong>doesn't scale directly with
parameter count</strong> but with <strong>context length</strong> — longer context and more concurrency mean more cache HBM.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>A back-of-envelope ledger</b>: a 7B model, 32 layers, 8 KV heads (GQA), head_dim=128, fp16 (2 bytes), per token:</div>
  <div class="cells"><span class="lab">per token</span><span class="cell">2</span><span class="sep">×</span><span class="cell">32 L</span><span class="sep">×</span><span class="cell">8 H</span><span class="sep">×</span><span class="cell">128</span><span class="sep">×</span><span class="cell">2B</span><span class="sep">=</span><span class="cell hl">≈128 KB</span></div>
  <div class="cells"><span class="lab">2048 tok</span><span class="cell">128 KB</span><span class="sep">×</span><span class="cell">2048</span><span class="sep">=</span><span class="cell hl">≈256 MB (one request!)</span></div>
</div>

<p>A few hundred concurrent requests and the KV cache alone eats tens to hundreds of GB. <strong>That is why "save HBM" is an
inference engine's number-one job</strong> — paging, prefix reuse, quantization and KV-cache quantization all fight over this ledger.</p>

<p>There's a causal layer that's easy to miss: <strong>the cache's total capacity directly caps how many requests you can serve at
once</strong>. The HBM pool is fixed, so the total number of tokens it can hold is fixed; every request consumes "context length ×
bytes per token" of cache. Thus <strong>longer context means lower concurrency</strong> — the two contend for the same HBM. So the
KV cache isn't just a performance issue but a <strong>hard capacity-vs-concurrency constraint</strong>: the SGLang scheduler (from
Lesson 18) must answer "is there enough HBM to admit one more request?" every step, and how boldly it can answer depends on how
frugally the cache is managed. Keep this chain in mind: <strong>save a bit of cache → admit a few more requests → higher whole-machine
throughput</strong>. That is exactly why we spend a whole "memory management" part (Lessons 29–32) digging into it.</p>

<h2>prefill vs decode: a request's two phases</h2>
<p>With the KV-cache lens, generation naturally splits into <strong>two phases with opposite characters</strong>:</p>

<table class="t">
  <tr><th>Phase</th><th>What it does</th><th>Compute character</th><th>Cache action</th></tr>
  <tr><td><strong>prefill</strong></td><td>Feed the <strong>whole prompt at once</strong>, compute all tokens' K/V in parallel, produce the 1st output token</td><td class="mono">compute-bound (big matmuls, GPU saturated)</td><td><strong>fills</strong> the whole history once</td></tr>
  <tr><td><strong>decode</strong></td><td>Then generate <strong>token by token</strong>, one new token per step</td><td class="mono">memory-bound (little compute, reads the whole KV cache each step)</td><td><strong>appends one row</strong> per step</td></tr>
</table>

<p>Remember the iron rule: <strong>one request = one prefill + many decodes</strong>. The two phases have completely different
personalities — prefill is like "<strong>reading a whole book in one breath</strong>" (throughput, compute-bound), decode like
"<strong>writing it out line by line</strong>" (latency, memory-bound). Precisely because decode is memory-bound (its compute is
tiny; the time goes to moving the KV cache and weights from HBM into the compute units), <strong>batching many requests together</strong>
pays off — one weight read serves many requests so the GPU stops idling on a single one. That is the motivation for the next lesson,
<strong>continuous batching</strong>; and the phase asymmetry ultimately motivates splitting them onto <strong>different machines</strong>
in PD disaggregation (Lesson 45).</p>

<p>Put this lesson back into Part 2's arc and you see why it comes first. Autoregression + the KV cache define inference's two
<strong>fundamentals</strong>: a <strong>serial per-token loop</strong> (which sets the floor on single-request latency) and an
<strong>append-only cache that grows linearly with context</strong> (which sets the ceiling on concurrency). The next four lessons
are all moves on these two fundamentals — <strong>Lesson 5</strong> uses continuous batching to turn "serial single" into
"parallel many", reclaiming the FLOPs wasted during decode; <strong>Lesson 6</strong> uses paging to pack that linearly-growing
cache tighter with less fragmentation; <strong>Lesson 7</strong> uses RadixAttention to let <strong>different requests share a
prefix's cache</strong>, cashing the "read-only" property into real compute and memory savings; <strong>Lesson 8</strong> puts it
all on the <strong>throughput-vs-latency</strong> scale and tells you which end each trick pushes on. Read on with three rulers —
"cache = money, concurrency = throughput, serial = latency floor" — and you'll see at a glance what every later design saves and trades.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Autoregressive</strong>: the model emits one token at a time, appends it, loops — each step looks back at all history.</li>
    <li><strong>KV cache</strong>: historical K/V are read-only, so cache and reuse them; each step drops from re-running the whole sequence <span class="mono">O(t²)</span> to <span class="mono">O(t)</span>.</li>
    <li><strong>HBM dominator</strong>: cache ≈ 2×layers×KV-heads×head_dim×bytes per token, growing with <strong>context length</strong> — saving HBM is job #1.</li>
    <li><strong>Two phases</strong>: prefill is compute-bound (fills once), decode is memory-bound (appends a row); one request = 1 prefill + N decodes.</li>
    <li><strong>Bridges</strong>: memory-bound decode ⇒ batching pays off (Lesson 5); pool/row-managed cache ⇒ paging (Lesson 6) and prefix reuse (Lesson 7) become possible.</li>
  </ul>
</div>
""",
}

LESSON_05 = {"zh": "PLACEHOLDER 连续批处理 占位内容，稍后替换为完整正文，确保模块可被导入与构建流程跑通。" * 4,
             "en": "PLACEHOLDER continuous batching placeholder body, replaced later with the full lesson so the module imports and the build runs." * 3}

LESSON_06 = {"zh": "PLACEHOLDER 分页 KV 占位内容，稍后替换为完整正文，确保模块可被导入与构建流程跑通。" * 4,
             "en": "PLACEHOLDER paged KV placeholder body, replaced later with the full lesson so the module imports and the build runs." * 3}

LESSON_07 = {"zh": "PLACEHOLDER RadixAttention 占位内容，稍后替换为完整正文，确保模块可被导入与构建流程跑通。" * 4,
             "en": "PLACEHOLDER RadixAttention placeholder body, replaced later with the full lesson so the module imports and the build runs." * 3}

LESSON_08 = {"zh": "PLACEHOLDER 吞吐 vs 延迟 占位内容，稍后替换为完整正文，确保模块可被导入与构建流程跑通。" * 4,
             "en": "PLACEHOLDER throughput vs latency placeholder body, replaced later with the full lesson so the module imports and the build runs." * 3}
