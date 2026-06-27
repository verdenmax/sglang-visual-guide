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

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  大模型是<strong>自回归</strong>的：每生成一个 token 都要回看<strong>全部历史</strong>。朴素做法每步重算整段、代价高到不可用；
  <strong>KV 缓存</strong>把历史 token 只读不变的 Key/Value 存下来反复用，把每步从"重算整段"压成"只算新词对历史"。
  代价是<strong>显存</strong>——缓存随上下文线性膨胀，正是后面<strong>分页、前缀复用、量化</strong>要对付的"大头"。
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

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="对比：无缓存时每个解码步都要重算全部历史 token（不断变大的三角形），有 KV 缓存时每步只算新 token 的 K/V 并复用缓存中的历史">
    <line x1="380" y1="28" x2="380" y2="276" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="30" y="44" style="font-weight:700;fill:var(--red)">❌ 无缓存：每步重算全部历史</text>
    <polygon points="96,70 96,210 232,210" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="94" style="fill:var(--muted);font-size:12px">步①</text>
    <rect x="96" y="78" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="134" style="fill:var(--muted);font-size:12px">步②</text>
    <rect x="96" y="118" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="128" y="118" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="174" style="fill:var(--muted);font-size:12px">步③</text>
    <rect x="96" y="158" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="128" y="158" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="160" y="158" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="244" style="fill:var(--red);font-size:12px">重算量 ∝ t：每步 O(t)，累计 O(t²)</text>
    <text x="404" y="44" style="font-weight:700;fill:var(--teal)">✅ 有 KV 缓存：只算新 token，复用历史</text>
    <text x="404" y="94" style="fill:var(--muted);font-size:12px">步①</text>
    <rect x="470" y="78" width="28" height="22" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="404" y="134" style="fill:var(--muted);font-size:12px">步②</text>
    <rect x="470" y="118" width="44" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="518" y="118" width="28" height="22" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="404" y="174" style="fill:var(--muted);font-size:12px">步③</text>
    <rect x="470" y="158" width="88" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="562" y="158" width="28" height="22" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="606" y="112" width="128" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="670" y="138" text-anchor="middle" style="fill:var(--teal);font-size:11px">缓存=只读复用</text>
    <text x="670" y="158" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">新 token=1 行</text>
    <text x="404" y="244" style="fill:var(--teal);font-size:12px">每步只算 1 个新 token：O(1) 计算 + 复用缓存</text>
  </svg>
  <div class="figcap"><b>图 1 · 有缓存 vs 无缓存</b> — 左：无缓存，每个 decode 步都把全部历史重算一遍（不断变大的三角形，累计 O(t²)）；右：有 KV 缓存，每步只算新 token 的 K/V，历史只读复用。</div>
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

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::MHATokenToKVPool</span><span class="ln">KV 的物理存储：逐层的 K/V 缓冲，按 token 槽位索引</span></div>
  <pre><span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="cm"># KV 的物理存储：逐层的 K 和 V 缓冲，按 token 槽位索引</span>
    <span class="kw">def</span> __init__(self, size, dtype, head_num, head_dim, layer_num, ...):
        <span class="cm"># size = 能容纳的最大 token 数；为每一层分配 k_buffer/v_buffer</span>
        ...
    <span class="kw">def</span> set_kv_buffer(self, layer, loc, cache_k, cache_v):
        ...   <span class="cm"># 把这个 token 的 K/V 写到槽位 `loc`</span>
    <span class="kw">def</span> get_kv_buffer(self, layer):
        ...   <span class="cm"># 读回 K/V 供注意力内核使用</span></pre>
</div>

<p>从这个构造函数就能读出缓存大小的"账本"：<strong>每个 token</strong> 占用的字节 ≈
<span class="mono">2(K和V) × layer_num × head_num × head_dim × dtype 字节数</span>。注意它<strong>不随模型参数量直接走</strong>，
而是随<strong>上下文长度</strong>线性增长——上下文越长、并发越多，缓存吃的显存越多。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>一个示意账本</b>：某 7B 模型，32 层、8 个 KV 头（GQA）、head_dim=128、fp16(2 字节)，每 token 的 KV 缓存：</div>
  <div class="cells"><span class="lab">每 token</span><span class="cell">2</span><span class="sep">×</span><span class="cell">32 层</span><span class="sep">×</span><span class="cell">8 头</span><span class="sep">×</span><span class="cell">128</span><span class="sep">×</span><span class="cell">2B</span><span class="sep">=</span><span class="cell hl">≈128 KB</span></div>
  <div class="cells"><span class="lab">2048 token</span><span class="cell">128 KB</span><span class="sep">×</span><span class="cell">2048</span><span class="sep">=</span><span class="cell hl">≈256 MB（一条请求！）</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="折线图：KV 缓存随序列长度线性增长，横轴为 token 数、纵轴为 KV 显存（MB）；示意 Llama-2-7B（无 GQA）每 token 的 KV 约 0.5 MB，2048 token 约 1 GB 每请求">
    <line x1="96" y1="40" x2="96" y2="246" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="96" y1="246" x2="712" y2="246" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="30" y="34" style="fill:var(--muted);font-size:12px">KV 显存（MB）</text>
    <text x="380" y="270" text-anchor="middle" style="fill:var(--muted);font-size:12px">序列长度（token）</text>
    <text x="90" y="250" text-anchor="end" style="fill:var(--faint);font-size:11px">0</text>
    <text x="90" y="150" text-anchor="end" style="fill:var(--faint);font-size:11px">512</text>
    <text x="90" y="64" text-anchor="end" style="fill:var(--faint);font-size:11px">1024</text>
    <text x="96" y="262" text-anchor="middle" style="fill:var(--faint);font-size:11px">0</text>
    <text x="404" y="262" text-anchor="middle" style="fill:var(--faint);font-size:11px">1024</text>
    <text x="668" y="262" text-anchor="middle" style="fill:var(--faint);font-size:11px">2048</text>
    <line x1="96" y1="246" x2="668" y2="60" style="stroke:var(--accent);stroke-width:2.5"/>
    <line x1="668" y1="60" x2="668" y2="246" style="stroke:var(--accent);stroke-width:1;stroke-dasharray:4 4"/>
    <circle cx="668" cy="60" r="5" style="fill:var(--accent);stroke:var(--accent-ink);stroke-width:1.5"/>
    <rect x="356" y="80" width="336" height="54" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="524" y="103" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--accent-ink)">Llama-2-7B（无 GQA）：每 token ≈ 0.5 MB</text>
    <text x="524" y="123" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--accent-ink)">2048 token ≈ 1 GB / 请求（示意）</text>
    <text x="150" y="206" style="fill:var(--muted);font-size:12px">线性增长：上下文翻倍 → 显存翻倍</text>
  </svg>
  <div class="figcap"><b>图 2 · KV 缓存随 token 线性增长</b> — 横轴是序列长度、纵轴是 KV 显存；缓存随上下文成正比上升（数字为示意：Llama-2-7B 无 GQA 每 token 约 0.5 MB、2048 token 约 1 GB/请求；启用 GQA 的模型如前述账本只需约 1/4）。</div>
</div>

<p>给个更具体的数：一条 <strong>2048 token</strong> 的请求按上面账本约占 <strong>256 MB</strong> KV 缓存；<strong>100 条并发</strong>就是约 <strong>25 GB</strong>——一张 80 GB 显卡近三分之一的显存，全压在 KV 上。</p>

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

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  An LLM is <strong>autoregressive</strong>: each new token must look back at <strong>all history</strong>. Recomputing the whole
  sequence every step is prohibitively expensive; the <strong>KV cache</strong> stores each token's read-only Key/Value once and
  reuses it, squeezing each step from "recompute everything" to "just the new word's attention over history". The price is
  <strong>HBM</strong> — the cache grows linearly with context, the very thing <strong>paging, prefix reuse and quantization</strong> later fight.
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

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="Contrast: without a cache every decode step recomputes all past tokens (an ever-growing triangle), while with a KV cache each step computes only the new token's K/V and reuses the cached history">
    <line x1="380" y1="28" x2="380" y2="276" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="30" y="44" style="font-weight:700;fill:var(--red)">❌ No cache: recompute all history each step</text>
    <polygon points="96,70 96,210 232,210" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="94" style="fill:var(--muted);font-size:12px">step①</text>
    <rect x="96" y="78" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="134" style="fill:var(--muted);font-size:12px">step②</text>
    <rect x="96" y="118" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="128" y="118" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="174" style="fill:var(--muted);font-size:12px">step③</text>
    <rect x="96" y="158" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="128" y="158" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="160" y="158" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="30" y="244" style="fill:var(--red);font-size:12px">recompute ∝ t: O(t) per step, O(t²) total</text>
    <text x="404" y="44" style="font-weight:700;fill:var(--teal)">✅ KV cache: compute new token, reuse history</text>
    <text x="404" y="94" style="fill:var(--muted);font-size:12px">step①</text>
    <rect x="470" y="78" width="28" height="22" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="404" y="134" style="fill:var(--muted);font-size:12px">step②</text>
    <rect x="470" y="118" width="44" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="518" y="118" width="28" height="22" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="404" y="174" style="fill:var(--muted);font-size:12px">step③</text>
    <rect x="470" y="158" width="88" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="562" y="158" width="28" height="22" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <rect x="606" y="112" width="128" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="670" y="138" text-anchor="middle" style="fill:var(--teal);font-size:11px">cache=read-only</text>
    <text x="670" y="158" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">new token=1 row</text>
    <text x="404" y="244" style="fill:var(--teal);font-size:12px">only 1 new token per step: O(1) compute + reuse</text>
  </svg>
  <div class="figcap"><b>Fig 1 · With-cache vs no-cache</b> — Left: no cache, every decode step recomputes all history (an ever-growing triangle, O(t²) total); right: with a KV cache, each step computes only the new token's K/V and reuses read-only history.</div>
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

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::MHATokenToKVPool</span><span class="ln">the physical KV store: per-layer K/V buffers indexed by token slot</span></div>
  <pre><span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="cm"># the physical KV store: per-layer K and V buffers, indexed by token slot</span>
    <span class="kw">def</span> __init__(self, size, dtype, head_num, head_dim, layer_num, ...):
        <span class="cm"># size = max tokens that fit; allocate k_buffer/v_buffer for each layer</span>
        ...
    <span class="kw">def</span> set_kv_buffer(self, layer, loc, cache_k, cache_v):
        ...   <span class="cm"># write this token's K/V at slot `loc`</span>
    <span class="kw">def</span> get_kv_buffer(self, layer):
        ...   <span class="cm"># read K/V back for the attention kernel</span></pre>
</div>

<p>The constructor reveals the cache's "ledger": <strong>per token</strong> the bytes ≈
<span class="mono">2(K and V) × layer_num × head_num × head_dim × dtype-bytes</span>. Note it <strong>doesn't scale directly with
parameter count</strong> but with <strong>context length</strong> — longer context and more concurrency mean more cache HBM.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>A back-of-envelope ledger</b>: a 7B model, 32 layers, 8 KV heads (GQA), head_dim=128, fp16 (2 bytes), per token:</div>
  <div class="cells"><span class="lab">per token</span><span class="cell">2</span><span class="sep">×</span><span class="cell">32 L</span><span class="sep">×</span><span class="cell">8 H</span><span class="sep">×</span><span class="cell">128</span><span class="sep">×</span><span class="cell">2B</span><span class="sep">=</span><span class="cell hl">≈128 KB</span></div>
  <div class="cells"><span class="lab">2048 tok</span><span class="cell">128 KB</span><span class="sep">×</span><span class="cell">2048</span><span class="sep">=</span><span class="cell hl">≈256 MB (one request!)</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="Line chart: KV cache grows linearly with sequence length; x-axis is token count, y-axis is KV memory (MB); illustrative Llama-2-7B (no GQA) with about 0.5 MB of KV per token, so 2048 tokens is about 1 GB per request">
    <line x1="96" y1="40" x2="96" y2="246" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="96" y1="246" x2="712" y2="246" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="30" y="34" style="fill:var(--muted);font-size:12px">KV memory (MB)</text>
    <text x="380" y="270" text-anchor="middle" style="fill:var(--muted);font-size:12px">sequence length (tokens)</text>
    <text x="90" y="250" text-anchor="end" style="fill:var(--faint);font-size:11px">0</text>
    <text x="90" y="150" text-anchor="end" style="fill:var(--faint);font-size:11px">512</text>
    <text x="90" y="64" text-anchor="end" style="fill:var(--faint);font-size:11px">1024</text>
    <text x="96" y="262" text-anchor="middle" style="fill:var(--faint);font-size:11px">0</text>
    <text x="404" y="262" text-anchor="middle" style="fill:var(--faint);font-size:11px">1024</text>
    <text x="668" y="262" text-anchor="middle" style="fill:var(--faint);font-size:11px">2048</text>
    <line x1="96" y1="246" x2="668" y2="60" style="stroke:var(--accent);stroke-width:2.5"/>
    <line x1="668" y1="60" x2="668" y2="246" style="stroke:var(--accent);stroke-width:1;stroke-dasharray:4 4"/>
    <circle cx="668" cy="60" r="5" style="fill:var(--accent);stroke:var(--accent-ink);stroke-width:1.5"/>
    <rect x="356" y="80" width="336" height="54" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="524" y="103" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--accent-ink)">Llama-2-7B (no GQA): KV ≈ 0.5 MB / token</text>
    <text x="524" y="123" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--accent-ink)">2048 tokens ≈ 1 GB / request (illustrative)</text>
    <text x="150" y="206" style="fill:var(--muted);font-size:12px">linear growth: double context → double memory</text>
  </svg>
  <div class="figcap"><b>Fig 2 · KV cache grows linearly with tokens</b> — x-axis is sequence length, y-axis is KV memory; the cache rises in proportion to context (numbers illustrative: Llama-2-7B without GQA ≈ 0.5 MB per token, 2048 tokens ≈ 1 GB/request; GQA models like the ledger above need only ~1/4).</div>
</div>

<p>To make it concrete: one <strong>2048-token</strong> request takes about <strong>256 MB</strong> of KV cache by the ledger above; <strong>100 concurrent</strong> requests is roughly <strong>25 GB</strong> — nearly a third of an 80 GB GPU, all spent on KV.</p>

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

LESSON_05 = {
    "zh": r"""
<p class="lead">
上一课我们看清了一条铁律：decode 是<strong>访存密集</strong>的——每一步只算一个新 token，算力几乎闲着，时间全花在
把权重和 KV 缓存从显存搬进计算单元。这就埋下了本课的核心动机：既然单条请求快不了，那就<strong>把更多请求拼在一起</strong>，
让同一次"权重读取"服务尽可能多的请求。怎么拼，决定了 GPU 是满载飞奔还是空转干等——这正是<strong>连续批处理
（continuous batching）</strong>要解决的问题，也是 SGLang 高吞吐的第一块基石。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象两种班车。一种是<strong>包车（charter bus）</strong>：必须<strong>坐满才发车</strong>，路上<strong>一站不停</strong>，
  非要等到终点才让所有人一起下。哪怕你三站就到，也得陪着全程；车上有人提前到了目的地，座位也只能空着干等。
  另一种是<strong>随上随下的循环接驳车（hop-on/hop-off shuttle）</strong>：<strong>每一站都停</strong>，到站的人立刻下、
  腾出座位；在站台等的人立刻补上来。车子<strong>永远是满的</strong>，没有空座、没有空等。静态批处理像前者，
  连续批处理像后者——而后者把同一辆车（GPU）的利用率拉到了极致。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>连续批处理 = 调度器在每个 decode 步都重新组建这一批</strong>。
  完成的请求<strong>当场离开、立刻释放它的 KV 槽位</strong>，排队的请求<strong>下一步就能加入</strong>——
  批次始终保持"<strong>满载</strong>"，GPU 不为任何单条请求空转。它也叫<strong>迭代级批处理（iteration-level batching）</strong>
  或<strong>在途批处理（in-flight batching）</strong>。在 SGLang 里，调度循环每一轮都调用
  <span class="inline">get_next_batch_to_run()</span> 来"为这一步组队"。
</div>

<h2>静态批处理：等最慢的那一个</h2>
<p>最朴素的批处理叫<strong>静态批处理（static / padded batching）</strong>：把同时到达的几条请求<strong>凑成一批</strong>，
一起送进模型，然后<strong>从头跑到尾</strong>。听起来很合理，但它有三个致命的浪费，根子都在"<strong>一批请求被绑死在一起</strong>"。</p>

<p>第一，<strong>整批要等最慢的那一条</strong>。同一批里，有的请求生成 10 个 token 就遇到结束符，有的要生成 500 个。
但静态批处理必须等到<strong>最长的那条</strong>跑完，整批才算结束。早早完成的请求只能<strong>陪跑</strong>，它的位置一直占着、却不产出有效 token。
第二，<strong>填充（padding）浪费</strong>：完成的序列虽然不再有意义，槽位却空着随大流被一遍遍前向，纯属空转算力。
第三，<strong>新请求要等整批排空</strong>：哪怕 GPU 此刻有空闲容量，新到的请求也进不来，必须<strong>等当前这批彻底结束</strong>才能开始下一批。
结果就是 GPU 利用率被按在地上摩擦——批越往后，还在"真干活"的请求越少，算力浪费越严重。</p>

<table class="t">
  <tr><th>维度</th><th>静态批处理（包车）</th><th>连续批处理（接驳车）</th></tr>
  <tr><td><strong>组批时机</strong></td><td>一次组好，<strong>绑死到底</strong></td><td class="mono">每个 decode 步<strong>重新组队</strong></td></tr>
  <tr><td><strong>请求完成</strong></td><td>等最慢的，<strong>整批一起结束</strong></td><td class="mono">谁完成谁<strong>当场离开</strong>，立刻释放 KV 槽</td></tr>
  <tr><td><strong>新请求加入</strong></td><td>必须<strong>等整批排空</strong></td><td class="mono"><strong>下一步</strong>就能补进来</td></tr>
  <tr><td><strong>GPU 利用率</strong></td><td>批后期大量空转（padding 浪费）</td><td class="mono">批次始终满载，<strong>持续饱和</strong></td></tr>
</table>

<p>把这三种浪费叠在一起，后果是惊人的。设想一批 8 条请求，其中 7 条只生成五六个 token 就结束，唯独 1 条要生成 500 个。
静态批处理会拖着这 7 个早早完成的"空槽"陪那 1 条长请求<strong>一路跑完 500 步</strong>——也就是说，绝大多数步里，批中真正在产出有效 token 的只有 1 条，
另外 7 个槽位纯粹在空转。利用率因此可能掉到<strong>个位数百分比</strong>。现实流量里请求长短本就参差不齐，这种"一条拖全批"的情况<strong>是常态而非例外</strong>，
所以静态批处理在生产环境里几乎不可接受。理解了这个痛点，你就明白连续批处理为什么是"刚需"，而不是锦上添花。</p>


<div class="fig">
  <svg viewBox="0 0 760 340" role="img" aria-label="静态批处理与连续批处理的 GPU 利用率时间线对比：静态批处理整批等最慢的请求、留下大片空转，连续批处理完成即离场、新请求立刻补满，GPU 持续满载">
    <line x1="700" y1="34" x2="700" y2="308" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="700" y="324" text-anchor="middle" style="fill:var(--muted);font-size:11px">批结束</text>
    <text x="8" y="26" style="font-weight:700;fill:var(--red)">静态批处理：整批等最慢的 R4 → 大量空转</text>
    <text x="8" y="58" class="mono" style="font-size:11px;fill:var(--muted)">槽1</text>
    <rect x="80" y="42" width="170" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="165" y="57" text-anchor="middle" class="mono" style="font-size:11px">R1 忙</text>
    <rect x="250" y="42" width="450" height="22" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="475" y="57" text-anchor="middle" style="fill:var(--red);font-size:11px">空转</text>
    <text x="8" y="88" class="mono" style="font-size:11px;fill:var(--muted)">槽2</text>
    <rect x="80" y="72" width="350" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="255" y="87" text-anchor="middle" class="mono" style="font-size:11px">R2 忙</text>
    <rect x="430" y="72" width="270" height="22" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="565" y="87" text-anchor="middle" style="fill:var(--red);font-size:11px">空转</text>
    <text x="8" y="118" class="mono" style="font-size:11px;fill:var(--muted)">槽3</text>
    <rect x="80" y="102" width="90" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="125" y="117" text-anchor="middle" class="mono" style="font-size:11px">R3 忙</text>
    <rect x="170" y="102" width="530" height="22" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="435" y="117" text-anchor="middle" style="fill:var(--red);font-size:11px">空转</text>
    <text x="8" y="148" class="mono" style="font-size:11px;fill:var(--muted)">槽4</text>
    <rect x="80" y="132" width="620" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="390" y="147" text-anchor="middle" class="mono" style="font-size:11px">R4 忙</text>
    <text x="8" y="184" style="font-weight:700;fill:var(--teal)">连续批处理：完成即离场，新请求立刻补满 → 持续满载</text>
    <text x="8" y="212" class="mono" style="font-size:11px;fill:var(--muted)">槽1</text>
    <rect x="80" y="196" width="170" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="165" y="211" text-anchor="middle" class="mono" style="font-size:11px">R1</text>
    <rect x="250" y="196" width="230" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="365" y="211" text-anchor="middle" class="mono" style="font-size:11px">R5</text>
    <rect x="480" y="196" width="220" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="590" y="211" text-anchor="middle" class="mono" style="font-size:11px">R7</text>
    <text x="8" y="242" class="mono" style="font-size:11px;fill:var(--muted)">槽2</text>
    <rect x="80" y="226" width="350" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="255" y="241" text-anchor="middle" class="mono" style="font-size:11px">R2</text>
    <rect x="430" y="226" width="270" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="565" y="241" text-anchor="middle" class="mono" style="font-size:11px">R6</text>
    <text x="8" y="272" class="mono" style="font-size:11px;fill:var(--muted)">槽3</text>
    <rect x="80" y="256" width="90" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="125" y="271" text-anchor="middle" class="mono" style="font-size:11px">R3</text>
    <rect x="170" y="256" width="270" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="305" y="271" text-anchor="middle" class="mono" style="font-size:11px">R8</text>
    <rect x="440" y="256" width="260" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="570" y="271" text-anchor="middle" class="mono" style="font-size:11px">R9</text>
    <text x="8" y="302" class="mono" style="font-size:11px;fill:var(--muted)">槽4</text>
    <rect x="80" y="286" width="620" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="390" y="301" text-anchor="middle" class="mono" style="font-size:11px">R4</text>
  </svg>
  <div class="figcap"><b>图 A · 静态 vs 连续批处理时间线</b> — 上：静态批处理里早早完成的请求只能占着槽位空转，整批被最慢的 R4 拖到底；下：连续批处理一旦有请求完成，新请求立刻补进同一槽位，时间线被填满，GPU 持续满载。</div>
</div>

<div class="card"><div class="tag">🔢 一个具体的例子</div>
<p>设想一批 <strong>8 个槽位</strong>：其中 7 条请求各生成约 6 个 token 就结束，剩下 1 条要生成 500 个。静态批处理会让这 7 个早早完成的槽位
<strong>空转约 494 步</strong>——绝大多数步里只有 1 / 8 的槽位在产出有效 token，等效利用率约 <strong>12%</strong>。连续批处理则在每一步把腾空的槽位
立刻补满，把等效占用率拉到接近 <strong>100%</strong>。同一张 GPU，仅凭"组批方式"的改变，吞吐就能提升 <strong>约 4–8×</strong>。</p></div>

<h2>连续批处理：每一步都重新组队</h2>
<p>连续批处理的核心动作只有一句话：<strong>不要一次组好就不动了，而是在每一个 decode 步都把这一批重新拼一遍。</strong>
把生成想成一个永不停歇的循环，调度器在循环的每一轮都做四件事——<strong>清退完成的、放进等待的、组成新一批、前向一步</strong>：</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>清退完成的请求</h4><p>上一步谁吐出了结束符或到了长度上限，就把它<strong>移出批次</strong>，<strong>立刻释放它占的 KV 缓存槽位</strong>。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>接纳等待的请求</h4><p>检查显存够不够、调度策略允许谁先上，把队列里的请求<strong>填进刚腾出的位置</strong>（必要时先做它的 prefill）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>组成这一步的批</h4><p>把当前所有在跑的请求拼成一个 batch——<strong>有 prefill 活就先做 prefill，否则做一批 decode</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>前向一步，循环</h4><p>整批一起前向一个 token，更新各自的 KV 缓存，回到第 1 步——批次始终<strong>填满、流动</strong>。</p></div></div>
</div>

<p>关键洞察在于<strong>"完成的请求能在中途立刻离场"</strong>。为什么它能立刻走、还能立刻释放显存？因为上一课讲过：
每条请求的 KV 缓存是<strong>按行、按槽位独立管理的</strong>，一条请求的 K/V 跟别人的互不纠缠。所以某条请求一旦结束，
它那几行 KV 槽位可以<strong>原地回收</strong>，马上分给下一个排队的请求——不需要等整批结束、不需要重排。
下面这张图把"一个 4 槽位的批"在连续几步里的流动画出来：完成的（✓）立刻被新来的（★）顶替，批永远是满的。</p>

<p>这里有个常被混淆的点值得点破：<strong>连续批处理并不是"把不同请求的 token 拼成更长的序列一起算"</strong>，
每条请求依然各算各的、各自维护自己的 KV 缓存、各自在自己那一行上自回归。它"批"的是<strong>同一个 decode 步里、来自不同请求的那一个 token</strong>——
把 N 条请求<strong>当前要生成的第 N 个 token</strong>横向并排，<strong>共享同一次权重读取</strong>一起前向。所以"批大小"指的是
<strong>同时在跑的请求条数</strong>，而不是序列长度。想清楚这一点，你就能理解为什么"完成的请求离场"只是把某一行抽走、丝毫不影响别的行继续跑。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>一个 4 槽位批次的流动</b>：每个 decode 步，完成的请求（✓）让位，等待的请求（★）当场补入——槽位永不空置</div>
  <div class="cells"><span class="lab">step t</span><span class="cell hl">R1</span><span class="cell hl">R2</span><span class="cell hl">R3</span><span class="cell hl">R4</span><span class="sep">→</span><span class="cell q">R2 吐出结束符 ✓</span></div>
  <div class="cells"><span class="lab">step t+1</span><span class="cell hl">R1</span><span class="cell">★R5</span><span class="cell hl">R3</span><span class="cell hl">R4</span><span class="sep">→</span><span class="cell q">R5 补入 R2 的空槽，立即开跑</span></div>
  <div class="cells"><span class="lab">step t+2</span><span class="cell hl">R1</span><span class="cell hl">R5</span><span class="cell hl">R3</span><span class="cell">★R6</span><span class="sep">→</span><span class="cell q">R4 完成 ✓，R6 立刻顶上</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 290" role="img" aria-label="一个 4 槽位批次在连续步上的网格：列是步、行是槽，请求随到随走——刚完成的槽位空出、新请求中途补入，运行中、新加入、刚完成用不同颜色区分">
    <text x="160" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t</text>
    <text x="320" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t+1</text>
    <text x="480" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t+2</text>
    <text x="640" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t+3</text>
    <text x="8" y="68" class="mono" style="font-size:11px;fill:var(--muted)">槽1</text>
    <text x="8" y="114" class="mono" style="font-size:11px;fill:var(--muted)">槽2</text>
    <text x="8" y="160" class="mono" style="font-size:11px;fill:var(--muted)">槽3</text>
    <text x="8" y="206" class="mono" style="font-size:11px;fill:var(--muted)">槽4</text>
    <rect x="90" y="44" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="160" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R1</text>
    <rect x="90" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="160" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="90" y="136" width="140" height="38" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="160" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--red)">R3 ✓</text>
    <rect x="90" y="182" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="160" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R4</text>
    <rect x="250" y="44" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="320" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R1</text>
    <rect x="250" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="320" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="250" y="136" width="140" height="38" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="320" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">★R5</text>
    <rect x="250" y="182" width="140" height="38" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="320" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--red)">R4 ✓</text>
    <rect x="410" y="44" width="140" height="38" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="480" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--red)">R1 ✓</text>
    <rect x="410" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="480" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="410" y="136" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="480" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R5</text>
    <rect x="410" y="182" width="140" height="38" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="480" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">★R6</text>
    <rect x="570" y="44" width="140" height="38" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="640" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">★R7</text>
    <rect x="570" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="570" y="136" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R5</text>
    <rect x="570" y="182" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R6</text>
    <rect x="90" y="244" width="16" height="12" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="112" y="254" style="font-size:11px;fill:var(--muted)">运行中</text>
    <rect x="280" y="244" width="16" height="12" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="302" y="254" style="font-size:11px;fill:var(--muted)">新加入 ★</text>
    <rect x="500" y="244" width="16" height="12" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="522" y="254" style="font-size:11px;fill:var(--muted)">刚完成 ✓</text>
  </svg>
  <div class="figcap"><b>图 B · 请求随到随走</b> — 列是相邻的调度步、行是 4 个批槽。蓝＝运行中，橙＝本步新加入（★），红＝本步刚完成（✓）。R3、R4、R1 先后完成腾出槽位，R5、R6、R7 中途补入——批次不靠整批排空，而是逐槽流动。</div>
</div>

<h2>为什么它能赢：回到"访存密集"</h2>
<p>连续批处理之所以是<strong>吞吐的核武器</strong>，根子要接回第 4 课。decode 是<strong>访存密集</strong>的：每生成一个 token，
都要把<strong>整个模型的权重</strong>从显存读一遍，但真正的矩阵乘法只服务<strong>一个</strong> token，算力被严重浪费。
而权重读取这件事，<strong>不管批里有 1 条还是 100 条请求，代价几乎一样</strong>——读一次权重，可以同时给一批请求做前向。
于是<strong>把请求"摞"在一起</strong>，就等于把"读权重"这笔固定开销<strong>摊薄到了每条请求头上</strong>：批越大、摊得越薄、单位算力产出的有效 token 越多。</p>

<p>静态批处理也想吃这个红利，但它<strong>守不住"批是满的"</strong>：随着请求陆续完成，批里的有效请求越来越少，摊薄效应迅速衰减，
后半程几乎在为空槽位空转。连续批处理的全部价值，就是<strong>每一步都把批重新填满</strong>——让"权重读一次、服务一大批"这件好事
<strong>从头到尾持续发生</strong>。直观地说：同样一张 GPU，静态批处理可能只有六七成时间在真干活，连续批处理能逼近满载。
这也是为什么连续批处理常被视为现代推理引擎<strong>提升吞吐最重要的单项技术</strong>——它不改模型、不掉精度，纯靠"组批方式"就把利用率拉满。</p>

<h2>SGLang 里的实现：每轮调用 get_next_batch_to_run</h2>
<p>在 SGLang 的运行时里，调度器（第 18 课的事件循环会细讲）<strong>每一轮迭代都调用一次</strong>
<span class="inline">get_next_batch_to_run()</span>，由它来"<strong>为这一步组队</strong>"。这个方法先把上一步已完成的请求过滤出去、
把新到的 prefill 合并进在跑的批，然后做一个关键判断：<strong>有没有 prefill 活要干？有就先返回一个 prefill 批；没有，就组一个 decode 批。</strong>
正是这一句"每步都重新决定批是什么"，把连续批处理落到了代码里。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.get_next_batch_to_run</span><span class="ln">每个调度步组建这一步的批</span></div>
  <pre><span class="kw">def</span> get_next_batch_to_run(self) -&gt; Optional[ScheduleBatch]:
    <span class="cm"># …先把上一步完成的请求过滤掉、把 prefill 合并进 running_batch…</span>

    <span class="cm"># 优先尝试组一个 prefill 批（有新 prompt 要预填充就先做）</span>
    new_batch = self.get_new_batch_prefill()

    <span class="kw">if</span> new_batch <span class="kw">is not</span> <span class="kw">None</span>:
        <span class="cm"># Run prefill first if possible —— 有 prefill 活，先做它</span>
        ret = new_batch
    <span class="kw">else</span>:
        <span class="cm"># 否则做一批 decode：把当前在跑的请求组成 decode 批</span>
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch <span class="kw">if not</span> self.running_batch.is_empty() <span class="kw">else</span> <span class="kw">None</span>
    <span class="kw">return</span> ret</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_batch.py ::ScheduleBatch</span><span class="ln">调度器一次前向跑的批：请求随到随走</span></div>
  <pre><span class="kw">class</span> ScheduleBatch:
    <span class="cm"># 一次前向跑的批：请求随到随走、动态增减</span>
    <span class="kw">def</span> init_new(reqs, ...):
        ...   <span class="cm"># 从等待队列新建一个批</span>
    <span class="kw">def</span> prepare_for_extend(self):
        ...   <span class="cm"># 准备一次 PREFILL（新 prompt，并行计算）</span>
    <span class="kw">def</span> prepare_for_decode(self):
        ...   <span class="cm"># 准备一个 DECODE 步（每个在跑请求生成一个新 token）</span>
    <span class="kw">def</span> filter_batch(self, ...):
        ...   <span class="cm"># 丢掉已完成的请求，其余继续跑</span>
    <span class="kw">def</span> merge_batch(self, other):
        ...   <span class="cm"># 把新加入的请求并进在跑的批</span></pre>
</div>

<p>读懂这段就抓住了连续批处理的灵魂：<strong>批不是一个静态对象，而是每一步被重新计算出来的结果</strong>。
完成的请求在过滤阶段离场、释放 KV 槽，等待的请求在 <span class="inline">get_new_batch_prefill</span> 阶段被接纳——
循环每转一圈，批就被刷新一次。"谁先上、上多少"由<strong>调度策略</strong>（第 20 课）决定，而<strong>prefill 和 decode 谁优先</strong>的取舍，
也正是在这里拍板的。</p>

<h2>代价与权衡：天下没有免费的午餐</h2>
<p>连续批处理把利用率拉满，但代价是<strong>每一步都要做更多的"账务工作"</strong>：过滤完成的、接纳等待的、重组批次、
管理一个个 KV 槽位的回收与分配。这意味着<strong>调度器本身必须足够便宜</strong>——如果每步的 Python 调度开销太大，
GPU 反而会在等 CPU 组队时空转，得不偿失。这就引出了 SGLang 的两个关键后续设计：</p>

<div class="cols">
  <div class="col"><h4>调度必须"零开销"</h4><p>既然每步都要重组批，调度逻辑就成了<strong>热路径</strong>。SGLang 用
  <strong>零开销重叠调度器</strong>（第 21 课）把 CPU 的组批开销<strong>藏进 GPU 上一步的计算时间里</strong>，
  让 GPU 永不为调度空等——这正是连续批处理能真正兑现吞吐红利的前提。</p></div>
  <div class="col"><h4>长 prompt 不能堵住队列</h4><p>如果某条请求的 prompt 极长，它的 prefill 会<strong>霸占一整步</strong>、
  把正在 decode 的请求全卡住，破坏批次的流动。SGLang 用<strong>分块预填充</strong>（第 22 课）把长 prefill
  <strong>切成小块</strong>，穿插进 decode 步里，保持批次顺滑。</p></div>
</div>

<p>把这一课放回 Part 2 的主线：第 4 课告诉我们 decode 访存密集、KV 缓存按槽位独立管理；本课正是<strong>把这两个性质变现</strong>——
因为访存密集，所以拼批划算；因为 KV 按槽独立，所以完成的请求能中途离场、立刻释放显存。连续批处理因此成为引擎吞吐的
<strong>第一引擎</strong>。接下来第 6 课会讲分页让 KV 槽位<strong>装得更紧、碎片更少</strong>，第 7 课讲 RadixAttention 让不同请求<strong>共享前缀缓存</strong>，
而第 8 课会把这些招式统一放到"<strong>吞吐 vs 延迟</strong>"的天平上称一称。带着"批要永远满、调度要够便宜"这两把尺子往下读，
后面每一个调度与内存设计，你都能一眼看出它在为连续批处理"保驾护航"。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>静态批处理</strong>：一次组批、绑死到底——整批等最慢的那条，完成的槽位空转（padding 浪费），新请求要等整批排空。GPU 利用率低。</li>
    <li><strong>连续批处理</strong>：又叫迭代级 / 在途批处理，<strong>每个 decode 步重新组批</strong>。完成的请求当场离场并释放 KV 槽，等待的下一步即加入，批次<strong>始终满载</strong>。</li>
    <li><strong>为什么赢</strong>：decode 访存密集，读一次权重可服务一整批；连续批处理保证"批永远满"，把权重读取的固定开销<strong>持续摊薄</strong>到每条请求。</li>
    <li><strong>在 SGLang 里</strong>：调度循环每轮调用 <span class="mono">get_next_batch_to_run()</span>——有 prefill 活先做 prefill，否则组一个 decode 批。</li>
    <li><strong>代价</strong>：每步账务更重 ⇒ 调度必须便宜（第 21 课零开销重叠调度器），长 prompt 要切块（第 22 课分块预填充）。前置依赖：KV 槽位（第 4 课）。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson nailed one iron law: decode is <strong>memory-bound</strong> — each step computes just one new token while the
compute units idle, with time spent moving weights and the KV cache out of HBM. That seeds this lesson's core motivation:
since a single request can't be sped up, <strong>pack more requests together</strong> so one "weight read" serves as many
requests as possible. <em>How</em> you pack them decides whether the GPU sprints at full load or idles — and that is exactly
what <strong>continuous batching</strong> solves, the first cornerstone of SGLang's high throughput.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture two buses. A <strong>charter bus</strong> <strong>won't leave until it's full</strong> and <strong>makes no stops</strong>
  — everyone must ride all the way to the terminus. Even if your stop is third, you ride the whole route; passengers who
  arrived early just sit in seats that produce nothing. A <strong>hop-on/hop-off shuttle</strong> <strong>stops at every stop</strong>:
  whoever has arrived gets off and frees a seat, and whoever is waiting hops on instantly. The bus stays <strong>always full</strong> —
  no empty seats, no idle waiting. Static batching is the charter bus; continuous batching is the shuttle — and the shuttle
  pushes that one bus (the GPU) to its utilization limit.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>continuous batching = the scheduler re-forms the batch at every decode step</strong>.
  A finished request <strong>leaves on the spot and frees its KV slot immediately</strong>; a waiting request
  <strong>joins on the very next step</strong> — so the batch stays <strong>full</strong> and the GPU never idles on any single
  request. It's also called <strong>iteration-level batching</strong> or <strong>in-flight batching</strong>. In SGLang, the
  scheduler loop calls <span class="inline">get_next_batch_to_run()</span> each round to "form the batch for this step."
</div>

<h2>Static batching: waiting on the slowest one</h2>
<p>The naive approach is <strong>static (padded) batching</strong>: gather a few requests that arrive together into a batch,
push them through the model, and <strong>run them start to finish as a group</strong>. It sounds reasonable, but it has three
fatal wastes — all rooted in "<strong>a batch of requests being chained together</strong>."</p>

<p>First, <strong>the whole batch waits for the slowest</strong>. Within one batch, some requests hit their end token after 10
tokens, others need 500. But static batching only finishes when the <strong>longest</strong> one does. Requests that finished
early just <strong>ride along</strong>, holding their slot while producing nothing. Second, <strong>padding waste</strong>: finished
sequences are meaningless, yet their slots tag along through every forward pass — pure wasted compute. Third, <strong>new requests
wait for the whole batch to drain</strong>: even if the GPU has free capacity right now, an arriving request can't enter until the
current batch <strong>fully ends</strong>. The result: GPU utilization is dragged down — the later the batch, the fewer requests are
still doing real work.</p>

<table class="t">
  <tr><th>Dimension</th><th>Static batching (charter bus)</th><th>Continuous batching (shuttle)</th></tr>
  <tr><td><strong>When batched</strong></td><td>Formed once, <strong>chained to the end</strong></td><td class="mono"><strong>Re-formed every decode step</strong></td></tr>
  <tr><td><strong>On finish</strong></td><td>Wait for slowest, <strong>whole batch ends together</strong></td><td class="mono">Whoever finishes <strong>leaves on the spot</strong>, frees its KV slot</td></tr>
  <tr><td><strong>New request joins</strong></td><td>Must <strong>wait for the batch to drain</strong></td><td class="mono">Can join on the <strong>next step</strong></td></tr>
  <tr><td><strong>GPU utilization</strong></td><td>Idles late in the batch (padding waste)</td><td class="mono">Batch always full, <strong>stays saturated</strong></td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 760 340" role="img" aria-label="GPU-utilization timeline comparing static vs continuous batching: static batching makes the whole batch wait on the slowest request leaving large idle gaps, while continuous batching backfills freed slots so the GPU stays full">
    <line x1="700" y1="34" x2="700" y2="308" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="700" y="324" text-anchor="middle" style="fill:var(--muted);font-size:11px">batch ends</text>
    <text x="8" y="26" style="font-weight:700;fill:var(--red)">Static batching: whole batch waits on slowest R4 → big idle gaps</text>
    <text x="8" y="58" class="mono" style="font-size:11px;fill:var(--muted)">slot1</text>
    <rect x="80" y="42" width="170" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="165" y="57" text-anchor="middle" class="mono" style="font-size:11px">R1 busy</text>
    <rect x="250" y="42" width="450" height="22" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="475" y="57" text-anchor="middle" style="fill:var(--red);font-size:11px">idle</text>
    <text x="8" y="88" class="mono" style="font-size:11px;fill:var(--muted)">slot2</text>
    <rect x="80" y="72" width="350" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="255" y="87" text-anchor="middle" class="mono" style="font-size:11px">R2 busy</text>
    <rect x="430" y="72" width="270" height="22" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="565" y="87" text-anchor="middle" style="fill:var(--red);font-size:11px">idle</text>
    <text x="8" y="118" class="mono" style="font-size:11px;fill:var(--muted)">slot3</text>
    <rect x="80" y="102" width="90" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="125" y="117" text-anchor="middle" class="mono" style="font-size:11px">R3 busy</text>
    <rect x="170" y="102" width="530" height="22" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="435" y="117" text-anchor="middle" style="fill:var(--red);font-size:11px">idle</text>
    <text x="8" y="148" class="mono" style="font-size:11px;fill:var(--muted)">slot4</text>
    <rect x="80" y="132" width="620" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="390" y="147" text-anchor="middle" class="mono" style="font-size:11px">R4 busy</text>
    <text x="8" y="184" style="font-weight:700;fill:var(--teal)">Continuous batching: finished leaves, newcomer backfills → stays full</text>
    <text x="8" y="212" class="mono" style="font-size:11px;fill:var(--muted)">slot1</text>
    <rect x="80" y="196" width="170" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="165" y="211" text-anchor="middle" class="mono" style="font-size:11px">R1</text>
    <rect x="250" y="196" width="230" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="365" y="211" text-anchor="middle" class="mono" style="font-size:11px">R5</text>
    <rect x="480" y="196" width="220" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="590" y="211" text-anchor="middle" class="mono" style="font-size:11px">R7</text>
    <text x="8" y="242" class="mono" style="font-size:11px;fill:var(--muted)">slot2</text>
    <rect x="80" y="226" width="350" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="255" y="241" text-anchor="middle" class="mono" style="font-size:11px">R2</text>
    <rect x="430" y="226" width="270" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="565" y="241" text-anchor="middle" class="mono" style="font-size:11px">R6</text>
    <text x="8" y="272" class="mono" style="font-size:11px;fill:var(--muted)">slot3</text>
    <rect x="80" y="256" width="90" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="125" y="271" text-anchor="middle" class="mono" style="font-size:11px">R3</text>
    <rect x="170" y="256" width="270" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="305" y="271" text-anchor="middle" class="mono" style="font-size:11px">R8</text>
    <rect x="440" y="256" width="260" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="570" y="271" text-anchor="middle" class="mono" style="font-size:11px">R9</text>
    <text x="8" y="302" class="mono" style="font-size:11px;fill:var(--muted)">slot4</text>
    <rect x="80" y="286" width="620" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="390" y="301" text-anchor="middle" class="mono" style="font-size:11px">R4</text>
  </svg>
  <div class="figcap"><b>Fig A · static vs continuous batching timeline</b> — Top: in static batching, requests that finish early just hold their slots idle while the whole batch is dragged to the end by the slowest R4. Bottom: in continuous batching, the moment a request finishes a new one backfills the same slot, the timeline fills up, and the GPU stays saturated.</div>
</div>

<div class="card"><div class="tag">🔢 A concrete example</div>
<p>Picture a batch of <strong>8 slots</strong>: 7 requests each finish after about 6 tokens, while 1 needs 500. Static batching keeps those 7 early-finished slots
<strong>idle for ~494 steps</strong> — in most steps only 1 of 8 slots produces a useful token, an effective utilization of about <strong>12%</strong>. Continuous batching
backfills every freed slot on the spot, lifting effective occupancy toward <strong>100%</strong>. On the same GPU, purely by changing <em>how</em> it batches, throughput
rises by <strong>~4–8×</strong>.</p></div>

<h2>Continuous batching: re-form the batch every step</h2>
<p>The core action is one sentence: <strong>don't form the batch once and freeze it — re-assemble the batch at every decode step.</strong>
Think of generation as a never-ending loop; each round the scheduler does four things — <strong>evict the finished, admit the waiting,
form a new batch, forward one step</strong>:</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Evict finished requests</h4><p>Whoever emitted an end token or hit the length limit last step is <strong>removed from the batch</strong>, <strong>freeing its KV-cache slot immediately</strong>.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Admit waiting requests</h4><p>Check whether HBM allows it and which the policy lets in first, then <strong>fill the freed slots</strong> from the queue (running its prefill first if needed).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Form this step's batch</h4><p>Assemble all currently-running requests into one batch — <strong>do prefill first if there's prefill work, else a decode batch</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Forward one step, loop</h4><p>Forward the whole batch one token, update each KV cache, back to step 1 — the batch stays <strong>full and flowing</strong>.</p></div></div>
</div>

<p>The key insight is that <strong>a finished request can leave mid-flight, instantly</strong>. Why can it leave — and free its
HBM — right away? Because, as last lesson showed, each request's KV cache is <strong>managed independently by row/slot</strong>;
one request's K/V is not entangled with another's. So the moment a request ends, its KV slots can be <strong>reclaimed in place</strong>
and handed to the next queued request — no waiting for the whole batch, no reshuffling. The figure below traces a 4-slot batch over
consecutive steps: the finished (✓) are replaced by newcomers (★), so the batch is always full.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>Flow of a 4-slot batch</b>: each decode step, finished requests (✓) yield and waiting ones (★) fill in on the spot — slots never sit empty</div>
  <div class="cells"><span class="lab">step t</span><span class="cell hl">R1</span><span class="cell hl">R2</span><span class="cell hl">R3</span><span class="cell hl">R4</span><span class="sep">→</span><span class="cell q">R2 emits end token ✓</span></div>
  <div class="cells"><span class="lab">step t+1</span><span class="cell hl">R1</span><span class="cell">★R5</span><span class="cell hl">R3</span><span class="cell hl">R4</span><span class="sep">→</span><span class="cell q">R5 fills R2's slot, starts at once</span></div>
  <div class="cells"><span class="lab">step t+2</span><span class="cell hl">R1</span><span class="cell hl">R5</span><span class="cell hl">R3</span><span class="cell">★R6</span><span class="sep">→</span><span class="cell q">R4 finishes ✓, R6 steps in</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 290" role="img" aria-label="a grid of a 4-slot batch over consecutive steps: columns are steps, rows are slots, requests join and leave mid-stream as freed slots are backfilled, with running, newly-admitted and just-finished shown in different colors">
    <text x="160" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t</text>
    <text x="320" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t+1</text>
    <text x="480" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t+2</text>
    <text x="640" y="30" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:12px">step t+3</text>
    <text x="8" y="68" class="mono" style="font-size:11px;fill:var(--muted)">slot1</text>
    <text x="8" y="114" class="mono" style="font-size:11px;fill:var(--muted)">slot2</text>
    <text x="8" y="160" class="mono" style="font-size:11px;fill:var(--muted)">slot3</text>
    <text x="8" y="206" class="mono" style="font-size:11px;fill:var(--muted)">slot4</text>
    <rect x="90" y="44" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="160" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R1</text>
    <rect x="90" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="160" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="90" y="136" width="140" height="38" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="160" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--red)">R3 ✓</text>
    <rect x="90" y="182" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="160" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R4</text>
    <rect x="250" y="44" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="320" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R1</text>
    <rect x="250" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="320" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="250" y="136" width="140" height="38" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="320" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">★R5</text>
    <rect x="250" y="182" width="140" height="38" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="320" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--red)">R4 ✓</text>
    <rect x="410" y="44" width="140" height="38" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="480" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--red)">R1 ✓</text>
    <rect x="410" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="480" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="410" y="136" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="480" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R5</text>
    <rect x="410" y="182" width="140" height="38" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="480" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">★R6</text>
    <rect x="570" y="44" width="140" height="38" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="640" y="68" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">★R7</text>
    <rect x="570" y="90" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="114" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R2</text>
    <rect x="570" y="136" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="160" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R5</text>
    <rect x="570" y="182" width="140" height="38" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="206" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">R6</text>
    <rect x="90" y="244" width="16" height="12" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="112" y="254" style="font-size:11px;fill:var(--muted)">running</text>
    <rect x="280" y="244" width="16" height="12" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="302" y="254" style="font-size:11px;fill:var(--muted)">newly admitted ★</text>
    <rect x="500" y="244" width="16" height="12" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="522" y="254" style="font-size:11px;fill:var(--muted)">just finished ✓</text>
  </svg>
  <div class="figcap"><b>Fig B · requests join and leave per step</b> — Columns are consecutive schedule steps; rows are the 4 batch slots. Blue = running, amber = admitted this step (★), red = finished this step (✓). R3, R4, R1 finish in turn and free their slots while R5, R6, R7 backfill mid-stream — the batch flows slot-by-slot instead of draining as a whole.</div>
</div>

<h2>Why it wins: back to "memory-bound"</h2>
<p>Continuous batching is a <strong>throughput weapon</strong>, and the root reason traces back to Lesson 4. decode is
<strong>memory-bound</strong>: to produce one token you read the <strong>entire model's weights</strong> from HBM, yet the actual
matmul serves just <strong>one</strong> token — compute is badly underused. And reading the weights <strong>costs about the same
whether the batch holds 1 request or 100</strong> — one weight read can forward a whole batch at once. So <strong>stacking requests
together</strong> amortizes that fixed "read the weights" cost <strong>across every request</strong>: the bigger the batch, the thinner
the amortization, the more useful tokens per unit of compute.</p>

<p>Static batching wants this dividend too, but it <strong>can't keep the batch full</strong>: as requests finish one by one, the
batch's effective size shrinks, the amortization decays fast, and the back half practically idles over empty slots. The entire value
of continuous batching is <strong>refilling the batch every step</strong> — making "read weights once, serve a big batch" happen
<strong>continuously, end to end</strong>. Intuitively: on the same GPU, static batching might do real work only ~60–70% of the time,
while continuous batching approaches full load. That's why it's widely seen as the <strong>single most important throughput technique</strong>
in modern inference engines — it changes no model and loses no accuracy, lifting utilization purely by <em>how</em> it batches.</p>

<h2>In SGLang: calling get_next_batch_to_run every round</h2>
<p>In SGLang's runtime, the scheduler (its event loop is detailed in Lesson 18) calls
<span class="inline">get_next_batch_to_run()</span> <strong>once per iteration</strong> to "<strong>form the batch for this step</strong>."
The method first filters out requests that finished last step and merges newly arrived prefill into the running batch, then makes a key
decision: <strong>is there prefill work? If so, return a prefill batch; otherwise, build a decode batch.</strong> That one "decide the
batch afresh each step" is exactly how continuous batching lands in code.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.get_next_batch_to_run</span><span class="ln">form this step's batch every schedule round</span></div>
  <pre><span class="kw">def</span> get_next_batch_to_run(self) -&gt; Optional[ScheduleBatch]:
    <span class="cm"># …first filter out last step's finished reqs, merge prefill into running_batch…</span>

    <span class="cm"># Try to form a prefill batch first (do new prompts' prefill if any)</span>
    new_batch = self.get_new_batch_prefill()

    <span class="kw">if</span> new_batch <span class="kw">is not</span> <span class="kw">None</span>:
        <span class="cm"># Run prefill first if possible</span>
        ret = new_batch
    <span class="kw">else</span>:
        <span class="cm"># Otherwise build a decode batch from the currently running requests</span>
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch <span class="kw">if not</span> self.running_batch.is_empty() <span class="kw">else</span> <span class="kw">None</span>
    <span class="kw">return</span> ret</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_batch.py ::ScheduleBatch</span><span class="ln">the batch the scheduler runs in one forward; requests join and leave</span></div>
  <pre><span class="kw">class</span> ScheduleBatch:
    <span class="cm"># the batch run in one forward; requests join and leave it over time</span>
    <span class="kw">def</span> init_new(reqs, ...):
        ...   <span class="cm"># build a fresh batch from waiting requests</span>
    <span class="kw">def</span> prepare_for_extend(self):
        ...   <span class="cm"># set up a PREFILL pass (new prompts, computed in parallel)</span>
    <span class="kw">def</span> prepare_for_decode(self):
        ...   <span class="cm"># set up a DECODE step (one new token per running request)</span>
    <span class="kw">def</span> filter_batch(self, ...):
        ...   <span class="cm"># drop finished requests, keep the rest running</span>
    <span class="kw">def</span> merge_batch(self, other):
        ...   <span class="cm"># fold newly-admitted requests into the running batch</span></pre>
</div>

<p>Read this and you grasp the soul of continuous batching: <strong>the batch is not a static object but a result recomputed every step</strong>.
Finished requests leave during the filter phase and free their KV slots; waiting requests are admitted in the
<span class="inline">get_new_batch_prefill</span> phase — each loop refreshes the batch. <strong>Who joins, and how many</strong> is decided
by the <strong>schedule policy</strong> (Lesson 20), and the <strong>prefill-vs-decode priority</strong> tradeoff is settled right here too.</p>

<h2>The cost and tradeoff: no free lunch</h2>
<p>Continuous batching maxes out utilization, but the price is <strong>more "bookkeeping" every step</strong>: filtering the finished,
admitting the waiting, re-forming the batch, reclaiming and allocating KV slots one by one. This means <strong>the scheduler itself must
be cheap enough</strong> — if the per-step Python scheduling overhead is too large, the GPU ends up idling while the CPU forms the batch,
defeating the purpose. That motivates two key SGLang follow-ups:</p>

<div class="cols">
  <div class="col"><h4>Scheduling must be "zero-overhead"</h4><p>Since the batch is re-formed every step, the scheduling logic is on the
  <strong>hot path</strong>. SGLang's <strong>zero-overhead overlap scheduler</strong> (Lesson 21) <strong>hides the CPU batch-forming cost
  inside the previous step's GPU compute</strong>, so the GPU never idles on scheduling — the prerequisite for actually cashing in the
  throughput dividend.</p></div>
  <div class="col"><h4>Long prompts mustn't clog the queue</h4><p>If a request's prompt is extremely long, its prefill can <strong>hog a whole
  step</strong> and stall every decoding request, breaking the batch's flow. SGLang's <strong>chunked prefill</strong> (Lesson 22) <strong>slices
  a long prefill into chunks</strong>, interleaving them with decode steps to keep the batch smooth.</p></div>
</div>

<p>Place this lesson back on Part 2's throughline: Lesson 4 taught that decode is memory-bound and the KV cache is managed independently
per slot; this lesson <strong>cashes both properties in</strong> — because decode is memory-bound, batching pays; because KV is per-slot
independent, a finished request can leave mid-flight and free HBM at once. Continuous batching thus becomes the engine's <strong>first
throughput engine</strong>. Next, Lesson 6 packs KV slots <strong>tighter with less fragmentation</strong> via paging, Lesson 7 lets requests
<strong>share prefix cache</strong> with RadixAttention, and Lesson 8 weighs all of this on the <strong>throughput vs latency</strong> scale.
Read on with two rulers — "keep the batch full" and "keep scheduling cheap" — and you'll see how every later scheduling and memory design
exists to safeguard continuous batching.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Static batching</strong>: formed once, chained to the end — the batch waits on the slowest, finished slots idle (padding waste), new requests wait for the batch to drain. Low GPU utilization.</li>
    <li><strong>Continuous batching</strong>: a.k.a. iteration-level / in-flight batching, <strong>re-forms the batch every decode step</strong>. Finished requests leave on the spot and free KV slots; waiting ones join next step; the batch stays <strong>full</strong>.</li>
    <li><strong>Why it wins</strong>: decode is memory-bound, one weight read serves a whole batch; continuous batching keeps the batch full, <strong>continuously amortizing</strong> the fixed weight-read cost across every request.</li>
    <li><strong>In SGLang</strong>: the scheduler loop calls <span class="mono">get_next_batch_to_run()</span> each round — prefill first if there's prefill work, else build a decode batch.</li>
    <li><strong>Cost</strong>: heavier per-step bookkeeping ⇒ scheduling must be cheap (Lesson 21 zero-overhead overlap scheduler), long prompts must be chunked (Lesson 22 chunked prefill). Prereq: KV slots (Lesson 4).</li>
  </ul>
</div>
""",
}

LESSON_06 = {
    "zh": r"""
<p class="lead">
上一课我们让批次"<strong>永远满载</strong>"，可这背后藏着一个没说破的前提：完成的请求要能<strong>立刻释放显存</strong>、
等待的请求要能<strong>立刻拿到显存</strong>。问题是——KV 缓存到底是怎么在显存里安家的？如果按老办法<strong>给每条请求预留一整块连续显存</strong>，
你会发现显存被大把大把地浪费、并发数被死死卡住。本课讲的<strong>PagedAttention 与分页 KV</strong>，
就是把操作系统几十年前解决内存碎片的"<strong>分页</strong>"老智慧搬到 KV 缓存上，让显存<strong>装得更紧、几乎零碎片</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象一家酒店有两种订房方式。老派做法是：每来一位客人，<strong>不管他住几晚，先把一整层连号的房间整片包给他</strong>——
  按"可能住的最长天数"预留。客人大多只住两三晚，<strong>整层却一直锁着不让别人住</strong>；而且必须是<strong>连号的一整层</strong>，
  东一间西一间的零散空房凑不成一层，只能干耗着。新派做法是：<strong>按晚、按单间出租</strong>，<strong>住几晚开几间</strong>，
  房间<strong>不必相邻</strong>，前台用一本<strong>登记簿</strong>记下"<strong>张三 → 301、507、902 房</strong>"。客人退房，房间当晚就回收给下一位。
  老派就是"<strong>连续预留</strong>"，新派就是"<strong>分页</strong>"——后者把同样的楼栋塞进多得多的客人。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>PagedAttention 把每条请求的 KV 缓存切成固定大小的"页（block）"，用一张"页表"把逻辑位置映射到物理页，按需分配、用完即还。</strong>
  这正是操作系统<strong>虚拟内存 / 分页</strong>的翻版——进程看到的是连续的逻辑地址，物理上却散落在任意页框里，一张页表把两者对上。
  因为页表负责"找页"，<strong>物理页根本不需要连续</strong>，于是碎片几乎消失、显存利用率逼近 100%。它由 vLLM 首创，如今被
  SGLang 在内的几乎所有推理引擎广泛采用。
</div>

<h2>痛点：连续预留为什么烧显存</h2>
<p>回到第 4 课：每条请求的 KV 缓存随着生成<strong>越长越大</strong>，但你<strong>事先并不知道</strong>它最终会生成多少 token。
最朴素的做法是<strong>按"最大长度"给每条请求预留一整块连续显存</strong>——比如上限 2048，就先占满 2048 个 token 的 KV 空间。
这种"<strong>连续预留</strong>"看似省事，却同时踩中两种经典的内存碎片，把宝贵的 HBM 大把浪费掉。</p>

<div class="cols">
  <div class="col"><h4>内部碎片（over-allocation）</h4><p>绝大多数请求<strong>远到不了最大长度</strong>：上限留 2048，实际可能只生成 60 个 token。
  那剩下的<strong>近 2000 个槽位</strong>从头到尾<strong>空着、却被这条请求锁死</strong>，谁也用不了。预留越大、浪费越狠——这叫<strong>内部碎片</strong>。</p></div>
  <div class="col"><h4>外部碎片（external fragmentation）</h4><p>请求来来去去，释放出的空闲块<strong>大小不一、夹在中间</strong>。
  新请求要一整块<strong>连续</strong>显存，可剩下的全是<strong>东一小段西一小段</strong>的缝隙，<strong>加起来够、却拼不成一整块</strong>，只能眼睁睁闲置——这叫<strong>外部碎片</strong>。</p></div>
</div>

<p>两种碎片叠加，后果很直接：<strong>显存明明没用满，却已经塞不下新请求了</strong>。而在第 5 课我们说过，<strong>并发请求数（批大小）</strong>
是吞吐的命门——批越大、权重读取的固定开销摊得越薄。连续预留把显存白白烧在空槽和缝隙上，<strong>等于直接砍掉了你能同时跑的请求数</strong>，
吞吐天花板被死死压低。所以"怎么放 KV"不是细节，而是<strong>决定并发上限的大事</strong>。</p>

<p>用一个具体的数字感受一下浪费有多狠。假设显存只够放 10000 个 token 的 KV，模型把最大长度设成 2048。连续预留下，
每条请求一进来就独占 2048 个槽位，于是<strong>最多只能同时跑 4 条请求</strong>（4×2048≈8192，第 5 条就放不下了）。可现实里这 4 条平均
也许只生成了 100 个 token——真正用到的总共才 400 个槽位，<strong>剩下 9600 个槽位全在空转</strong>，利用率不到 5%。换成分页按需分配，
同样 10000 个槽位、平均每条 100 token，就能<strong>同时塞下近 100 条请求</strong>。并发从 4 跳到近 100，吞吐随之水涨船高——
这就是"怎么放 KV"能左右整机吞吐的真实量级。</p>

<table class="t">
  <tr><th>浪费来源</th><th>连续预留（老派包整层）</th><th>分页 KV（按页出租）</th></tr>
  <tr><td><strong>内部碎片</strong></td><td>按最大长度预留，没用到的槽位全锁死</td><td class="mono">按页<strong>按需</strong>分配，最多浪费<strong>不到一页</strong></td></tr>
  <tr><td><strong>外部碎片</strong></td><td>要一整块连续显存，缝隙拼不起来</td><td class="mono">物理页<strong>无需连续</strong>，任意空页都能用</td></tr>
  <tr><td><strong>能否提前定长</strong></td><td>必须预估上限，估大了浪费、估小了截断</td><td class="mono">边生成<strong>边长页</strong>，不必预知长度</td></tr>
  <tr><td><strong>对并发的影响</strong></td><td>显存早早耗尽，<strong>并发被压低</strong></td><td class="mono">显存装得紧，<strong>并发显著提高</strong></td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="上半区是连续分配：变长请求 A、B、C 之间夹着无法利用的碎片空洞，一条需要 6 格的长请求虽然总空闲够、却没有一段连续空间而放不下；下半区是分页：KV 切成固定大小的页，那条长请求的 6 个页散落填进任意空槽，零浪费">
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">连续分配：变长请求留下无法利用的空洞</text>
    <rect x="24" y="42" width="732" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="30" y="48" width="120" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="69" text-anchor="middle" class="mono" style="font-size:11px">req A</text>
    <rect x="154" y="48" width="64" height="32" rx="4" style="fill:var(--faint);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="186" y="68" text-anchor="middle" style="font-size:10px;fill:var(--muted)">碎片</text>
    <rect x="222" y="48" width="96" height="32" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="270" y="69" text-anchor="middle" class="mono" style="font-size:11px">req B</text>
    <rect x="322" y="48" width="80" height="32" rx="4" style="fill:var(--faint);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="362" y="68" text-anchor="middle" style="font-size:10px;fill:var(--muted)">碎片</text>
    <rect x="406" y="48" width="72" height="32" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="442" y="69" text-anchor="middle" class="mono" style="font-size:11px">req C</text>
    <rect x="482" y="48" width="96" height="32" rx="4" style="fill:var(--faint);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="530" y="68" text-anchor="middle" style="font-size:10px;fill:var(--muted)">碎片</text>
    <rect x="582" y="48" width="168" height="32" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="666" y="68" text-anchor="middle" style="font-size:10px;fill:var(--faint)">空闲</text>
    <text x="24" y="106" style="fill:var(--red);font-size:12px;font-weight:700">长请求需 6 格：总空闲够，却没有一段连续 → 放不下</text>
    <line x1="24" y1="120" x2="756" y2="120" style="stroke:var(--line);stroke-width:1;stroke-dasharray:5 5"/>
    <text x="24" y="148" style="font-weight:700;fill:var(--accent-ink)">分页：KV 切成固定大小的页，填进任意空槽 → 零浪费</text>
    <rect x="24" y="164" width="732" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="30" y="170" width="58" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="90" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="150" y="170" width="58" height="32" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="210" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="270" y="170" width="58" height="32" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="330" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="390" y="170" width="58" height="32" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="450" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="510" y="170" width="58" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="570" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="630" y="170" width="58" height="32" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="690" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="24" y="228" style="fill:var(--teal);font-size:12px;font-weight:700">长请求的 6 个红页散落各处、全部填满——物理页无需相邻</text>
  </svg>
  <div class="figcap"><b>图 1 · 碎片化 vs 分页</b> — 连续分配会留下放不下长请求的碎片空洞（总空闲够、却没有连续段）；分页把 KV 切成固定大小的页，散落填进任意空槽，零浪费。</div>
</div>

<p>用一个具体口径感受这种差别：取 <span class="mono">page_size = 16</span> token/页，一条只生成 60 token 的请求只需 <span class="mono">⌈60/16⌉ = 4</span> 页（64 槽），
末页仅浪费 4 个槽，<strong>内部碎片不到 7%</strong>；而连续预留按 2048 上限算，<strong>同一条请求浪费约 97%</strong>。把碎片率从近 50%（典型连续预留场景）压到 <strong>5% 以下</strong>，
正是分页能把并发翻十几倍的来源。</p>

<h2>分页的核心：固定页 + 页表</h2>
<p>PagedAttention 的思路只有两件东西：<strong>固定大小的页</strong>，和<strong>一张把逻辑映射到物理的页表</strong>。
把一条请求的 KV 序列想成一条<strong>逻辑上连续的 token 流</strong>，但底层把它切成每 <span class="mono">page_size</span> 个 token 一页，
<strong>每页单独从显存池里领</strong>。哪一页放在物理显存的哪个位置完全无所谓——一张<strong>页表</strong>记下"第几页 → 物理块号"，需要时照着它去取。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>切页</h4><p>请求的 token 流按 <strong>page_size</strong>（如 16）切成一页页，<strong>逻辑上连续</strong>，物理上分开存。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>按需领页</h4><p>序列每长满一页，就向分配器<strong>再要一个空闲物理块</strong>——<strong>用多少领多少</strong>，绝不预留到顶。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>页表登记</h4><p>把"<strong>逻辑页号 → 物理块号</strong>"写进这条请求的页表（SGLang 里即 token→槽位的映射）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>用完即还</h4><p>请求一结束，它占的所有页<strong>整批回收进空闲池</strong>，下一秒就能分给别的请求。</p></div></div>
</div>

<p>关键在于第 3 步那张<strong>页表</strong>：注意力算子要回看历史 token 的 K/V 时，<strong>不直接按连续地址去取</strong>，
而是<strong>先查页表、再去对应的物理块里把 K/V "捡（gather）"回来</strong>。正因为有了这层间接映射，<strong>物理块爱放哪放哪、根本不用连续</strong>——
这就是分页能消灭外部碎片的根本原因。下面这张图把"逻辑连续、物理分散、页表牵线"画了出来：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>页表把逻辑页映射到物理块</b>：同一条请求的 KV 在物理上散落各处，页表负责对上号——物理块<strong>无需相邻</strong></div>
  <div class="cells"><span class="lab">逻辑页 0</span><span class="cell hl">tok 0–15</span><span class="sep">→</span><span class="cell q">物理块 #7</span></div>
  <div class="cells"><span class="lab">逻辑页 1</span><span class="cell hl">tok 16–31</span><span class="sep">→</span><span class="cell q">物理块 #2</span></div>
  <div class="cells"><span class="lab">逻辑页 2</span><span class="cell hl">tok 32–47</span><span class="sep">→</span><span class="cell q">物理块 #9</span></div>
  <div class="cells"><span class="lab">逻辑页 3</span><span class="cell">tok 48–…（生成中）</span><span class="sep">→</span><span class="cell q">满页时再领一个空闲块</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 280" role="img" aria-label="页表把逻辑位置映射到分散的物理页：左边是一条请求逻辑上连续的 token 位置 0、1、2、3，中间是页表把逻辑页 0、1 映射到物理块号 #7、#2，右边是散落各处、互不相邻的物理页；箭头显示逻辑页 0 指向较低处的 #7、逻辑页 1 指向较高处的 #2">
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">逻辑 token 位置（连续）</text>
    <rect x="24" y="40" width="150" height="30" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="99" y="60" text-anchor="middle" class="mono" style="font-size:11px">pos 0</text>
    <rect x="24" y="74" width="150" height="30" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="99" y="94" text-anchor="middle" class="mono" style="font-size:11px">pos 1</text>
    <rect x="24" y="108" width="150" height="30" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="99" y="128" text-anchor="middle" class="mono" style="font-size:11px">pos 2</text>
    <rect x="24" y="142" width="150" height="30" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="99" y="162" text-anchor="middle" class="mono" style="font-size:11px">pos 3</text>
    <text x="24" y="196" style="font-size:11px;fill:var(--faint)">page_size = 2：pos 0–1 → 逻辑页 0，pos 2–3 → 逻辑页 1</text>
    <text x="312" y="26" style="font-weight:700;fill:var(--accent-ink)">页表 page table</text>
    <rect x="312" y="40" width="190" height="40" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="407" y="65" text-anchor="middle" class="mono" style="font-size:11px">逻辑页 0 → #7</text>
    <rect x="312" y="92" width="190" height="40" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="407" y="117" text-anchor="middle" class="mono" style="font-size:11px">逻辑页 1 → #2</text>
    <text x="600" y="26" style="font-weight:700;fill:var(--muted)">物理页（分散）</text>
    <rect x="600" y="40" width="150" height="30" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="675" y="60" text-anchor="middle" class="mono" style="font-size:11px">phys #2</text>
    <rect x="600" y="78" width="150" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="675" y="98" text-anchor="middle" style="font-size:10px;fill:var(--faint)">空闲 #4</text>
    <rect x="600" y="116" width="150" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="675" y="136" text-anchor="middle" style="font-size:10px;fill:var(--faint)">空闲 #5</text>
    <rect x="600" y="154" width="150" height="30" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="675" y="174" text-anchor="middle" class="mono" style="font-size:11px">phys #7</text>
    <rect x="600" y="192" width="150" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="675" y="212" text-anchor="middle" style="font-size:10px;fill:var(--faint)">空闲 #9</text>
    <path d="M 174 55 C 240 55, 250 60, 312 60" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <path d="M 174 125 C 240 125, 250 112, 312 112" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <path d="M 502 60 C 555 60, 545 169, 600 169" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="600,169 586,162 586,176" style="fill:var(--blue)"/>
    <path d="M 502 112 C 555 112, 545 55, 600 55" style="fill:none;stroke:var(--amber);stroke-width:1.5"/>
    <polygon points="600,55 586,48 586,62" style="fill:var(--amber)"/>
    <text x="312" y="200" style="font-size:11px;fill:var(--muted)">逻辑连续 · 物理分散 · 页表牵线</text>
  </svg>
  <div class="figcap"><b>图 2 · 页表映射</b> — 逻辑上连续的 token 位置（0、1、2、3）经页表映射到分散的物理页（#7、#2）：逻辑连续、物理分散。</div>
</div>

<p>这里有个常被忽略的取舍：<strong>page_size 不是越小越好</strong>。页越小，内部浪费越少（最多浪费小半页），但页表更长、查表与算子里的间接寻址开销更大；
页越大，寻址更省，但每条请求末尾"<strong>没填满的那一页</strong>"浪费就更多。常见取值是 <span class="mono">1、16、32</span> 这类，
具体由模型与硬件权衡决定——这正是工程里"<strong>碎片 vs 开销</strong>"的经典折中。</p>

<p>还要澄清一个容易混淆的点：<strong>页是按 token 切的，不是按字节切的</strong>。一页装 <span class="mono">page_size</span> 个 token 的 K 和 V，
每个 token 的 KV 大小由层数、注意力头数、头维度决定，是固定的。所以"领一页"就是"领下 page_size 个 token 的 KV 空间"，
回收时也以页为单位整批归还。正因为粒度统一、对齐干净，分配器才能用一句 <span class="mono">free_pages[:num_pages]</span> 这样的<strong>极轻量操作</strong>
完成分配，而不必像通用内存分配器那样维护复杂的空闲链表、做首次适配或最佳适配的搜索——这也是分页在推理热路径上<strong>足够便宜</strong>的原因之一。</p>

<h2>SGLang 里的落地：分页分配器与 req_to_token</h2>
<p>在 SGLang 里，这套机制由一个<strong>分页的 token→KV 分配器</strong>承担：它管理着整池 KV 槽位，但<strong>以"页"为最小分配单位</strong>派发。
配合第 4 课讲过的 KV 池，还有一张 <span class="inline">req_to_token</span> 映射，记录<strong>每条请求的 token → 物理槽位</strong>——
它就是上面那张"页表"在代码里的化身。下面这段就是分页分配器的骨架，<strong>alloc 按页摘、free 按页还</strong>，全程以 <span class="mono">page_size</span> 为粒度：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/allocator/paged.py ::PagedTokenToKVPoolAllocator</span><span class="ln">分页分配器：按页发放 KV 槽位</span></div>
  <pre><span class="kw">class</span> <span class="st">PagedTokenToKVPoolAllocator</span>(BaseTokenToKVPoolAllocator):
    <span class="cm"># 管理 KV cache 索引的分配器；每个请求的输出始终是 page 对齐的</span>
    <span class="kw">def</span> __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size        <span class="cm"># 总槽位数 ÷ 每页大小 = 页数</span>

    <span class="kw">def</span> alloc(self, need_size):
        <span class="cm"># page 对齐分配：返回若干页展开后的物理索引</span>
        num_pages = need_size // self.page_size
        out_pages = self.free_pages[:num_pages]   <span class="cm"># 从空闲页里摘走 num_pages 页</span>
        self.free_pages = self.free_pages[num_pages:]
        out_indices = (out_pages[:, <span class="kw">None</span>] * self.page_size
                       + torch.arange(self.page_size, device=self.device)).reshape(-1)
        <span class="kw">return</span> out_indices                        <span class="cm"># 展开成 token 级槽位索引</span>

    <span class="kw">def</span> free(self, free_index):
        <span class="cm"># 请求结束：把它占用的页号去重后整批还回空闲池</span>
        free_page_indices = torch.unique(free_index // self.page_size)
        self.free_pages = torch.cat((free_page_indices, self.free_pages))</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::ReqToTokenPool</span><span class="ln">请求 → 它的 token 槽位（索引层）</span></div>
  <pre><span class="kw">class</span> <span class="st">ReqToTokenPool</span>:
    <span class="cm"># 把每条请求映射到承载其 KV 的 token 槽位（索引层）</span>
    <span class="kw">def</span> __init__(self, size, max_context_len, ...):
        <span class="cm"># req_to_token[req][pos] -&gt; 该 token 的物理 KV 槽位</span>
        self.req_to_token = ...   <span class="cm"># 形状 [size, max_context_len]</span>
    <span class="kw">def</span> alloc(self, reqs):
        ...   <span class="cm"># 为每条新请求分配一行 req_pool_idx（整行索引）</span>
    <span class="kw">def</span> free(self, req):
        ...   <span class="cm"># 请求结束后把它的 req_pool_idx 行还回池中</span></pre>
</div>

<p>读懂这段就抓住了分页的精髓：<strong>显存不再按"请求"整片预留，而是按"页"零售。</strong>
<span class="inline">alloc</span> 从 <span class="mono">free_pages</span> 里<strong>摘走</strong>几页、把页号乘 <span class="mono">page_size</span>
再加页内偏移<strong>展开成 token 槽位索引</strong>；<span class="inline">free</span> 在请求结束时把那些索引<strong>除以 page_size 去重得到页号、整批还回</strong>。
注意类的注释那句"输出始终 page 对齐"——这保证了页与页之间<strong>干净对齐、好回收</strong>，正是连续批处理（第 5 课）能"完成即释放、随到随补"的物理基础。</p>

<h2>为什么这是第 7 课的跳板</h2>
<p>分页最深远的意义，不止于省显存。一旦 KV 被切成<strong>固定大小、由页表索引的物理块</strong>，一件神奇的事就成为可能：
<strong>不同请求的页表，可以指向同一个物理块</strong>。设想两条请求开头是同一段 system prompt——它们前几页的 KV <strong>逐字节相同</strong>，
那为什么要各存一份？让两张页表<strong>都指向那同一批物理页</strong>不就行了？这正是<strong>第 7 课 RadixAttention 前缀共享</strong>要做的事。</p>

<div class="flow">
  <div class="node hl"><div class="nt">固定页 + 页表</div><div class="nd">本课：按页索引、物理可分散</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">页可被多请求指向</div><div class="nd">页表间接层带来的自由</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">共享前缀 KV</div><div class="nd">第 7 课 RadixAttention</div></div>
</div>

<p>这层"<strong>多张页表指向同一块</strong>"的能力，正是间接寻址送给我们的红利。回想老派的连续预留：每条请求的 KV 是一整片私有显存，
想共享也无从下手——你没法让两块连续区"部分重叠"。而分页把 KV 拆成独立的小块、用页表牵线之后，<strong>共享就退化成"两张表写同一个块号"这么简单</strong>。
第 7 课会进一步用一棵<strong>基数树（radix tree）</strong>把所有请求的前缀组织起来，自动发现"谁和谁开头相同"，再让它们的页表复用同一批物理页——
不仅省显存，<strong>相同前缀的 prefill 计算也只做一次</strong>。可以说，没有本课的分页地基，第 7 课的前缀共享根本无从谈起。</p>

<p>把这一课放回 Part 2 主线：第 4 课建好 KV 池、按槽位独立管理；第 5 课靠连续批处理把批喂满；本课用<strong>分页</strong>把每个槽位<strong>装得更紧、碎片更少</strong>，
解开"连续预留"对并发的枷锁；第 7 课则顺着页表这层间接，把"<strong>能否复用别人的 KV</strong>"变成现实。再往后，KV 缓存的更多内幕、
以及把 KV 在显存/内存/磁盘间分层的 HiCache，会在<strong>第 29–32 课</strong>展开。带着"<strong>固定页、页表索引、按需分配、用完即还</strong>"这十二个字往下读，
你会发现 SGLang 的内存系统几乎都建在这块地基上。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>痛点</strong>：给每条请求按最大长度预留一整块<strong>连续</strong>显存，会同时产生<strong>内部碎片</strong>（多数请求到不了最大长度，空槽锁死）和<strong>外部碎片</strong>（释放的缝隙拼不成连续块），把 HBM 大把烧掉、<strong>压低并发上限</strong>。</li>
    <li><strong>PagedAttention</strong>：把 KV 切成<strong>固定大小的页</strong>，用一张<strong>页表</strong>把<strong>逻辑位置 → 物理块号</strong>映射起来（即操作系统虚拟内存 / 分页）。<strong>按需领页、用完即还</strong>，物理块<strong>无需连续</strong>，碎片几近消失。</li>
    <li><strong>为何能非连续</strong>：注意力算子<strong>先查页表、再 gather</strong> 物理块里的 K/V，所以页放在哪都行——这是消灭外部碎片的根本。</li>
    <li><strong>在 SGLang 里</strong>：<span class="mono">PagedTokenToKVPoolAllocator</span> 以 <span class="mono">page_size</span> 为粒度 <span class="mono">alloc</span>/<span class="mono">free</span>，配合 <span class="mono">req_to_token</span> 记录每条请求的 token→槽位（即页表，承接第 4 课 KV 池）。</li>
    <li><strong>取舍 & 前瞻</strong>：page_size 小则碎片少但查表开销大，大则反之。分页让<strong>不同请求的页表指向同一物理块</strong>成为可能——这正是<strong>第 7 课 RadixAttention</strong> 前缀共享的跳板；KV 内幕与 HiCache 见<strong>第 29–32 课</strong>。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson kept the batch <strong>always full</strong>, but that hides an unstated premise: a finished request must
<strong>free its HBM immediately</strong>, and a waiting one must <strong>get HBM immediately</strong>. The question is —
how does the KV cache actually live in HBM? If you use the old way and <strong>reserve a contiguous block of HBM per request</strong>,
you'll find HBM wasted by the bucketload and concurrency capped hard. This lesson's <strong>PagedAttention and paged KV</strong>
borrow the operating system's decades-old "<strong>paging</strong>" wisdom for fighting fragmentation and apply it to the KV cache,
so HBM packs <strong>tighter, with near-zero fragmentation</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture a hotel with two booking styles. The old way: for each guest, <strong>reserve a whole contiguous floor up front</strong>
  no matter how many nights they stay — sized for the <strong>longest possible</strong> stay. Most guests stay two or three nights, yet
  <strong>the whole floor stays locked away from everyone else</strong>; worse, it must be a <strong>contiguous floor</strong>, so scattered
  empty rooms here and there can never add up to one. The new way: <strong>rent single rooms by the night</strong>, <strong>open as many as
  you need</strong>, rooms <strong>need not be adjacent</strong>, and the front desk keeps a <strong>directory</strong>: "<strong>Zhang San → rooms
  301, 507, 902</strong>." A guest checks out, the room is reclaimed that night. The old way is "<strong>contiguous reservation</strong>"; the new
  one is "<strong>paging</strong>" — and it stuffs far more guests into the same building.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>PagedAttention cuts each request's KV cache into fixed-size "pages (blocks)" and uses a "page table" to map logical
  positions to physical pages, allocating on demand and freeing when done.</strong> This is exactly the OS's <strong>virtual memory / paging</strong>:
  a process sees a contiguous logical address space, but physically it's scattered across arbitrary page frames, with a page table tying the two
  together. Because the page table does the "finding," <strong>physical pages need not be contiguous</strong>, so fragmentation nearly vanishes and
  HBM utilization approaches 100%. Pioneered by vLLM, it's now broadly adopted by nearly every inference engine, SGLang included.
</div>

<h2>The pain: why contiguous reservation burns HBM</h2>
<p>Back to Lesson 4: each request's KV cache <strong>grows as it generates</strong>, but you <strong>don't know in advance</strong> how many tokens
it will produce. The naive approach is to <strong>reserve one contiguous block of HBM per request, sized for the max length</strong> — cap of 2048
means grabbing KV space for 2048 tokens up front. This "<strong>contiguous reservation</strong>" looks convenient but hits two classic fragmentation
problems at once, wasting precious HBM.</p>

<div class="cols">
  <div class="col"><h4>Internal fragmentation (over-allocation)</h4><p>Most requests <strong>never reach the max length</strong>: cap of 2048, but maybe only
  60 tokens are generated. The remaining <strong>~2000 slots</strong> sit <strong>empty yet locked</strong> by that request — usable by no one. The bigger
  the reservation, the worse the waste. That's <strong>internal fragmentation</strong>.</p></div>
  <div class="col"><h4>External fragmentation</h4><p>Requests come and go, leaving free blocks of <strong>uneven sizes wedged in between</strong>. A new
  request wants one <strong>contiguous</strong> block, but what's left is <strong>little gaps scattered around</strong> — <strong>enough in total, yet
  un-assemblable into one block</strong>, so it idles. That's <strong>external fragmentation</strong>.</p></div>
</div>

<p>Stack the two and the consequence is blunt: <strong>HBM isn't even full, yet a new request no longer fits</strong>. And as Lesson 5 said, the number
of <strong>concurrent requests (batch size)</strong> is the throughput lifeline — bigger batches amortize the fixed weight-read cost thinner. Contiguous
reservation burns HBM on empty slots and gaps, which <strong>directly cuts how many requests you can run at once</strong>, pressing the throughput ceiling
down hard. So "how you lay out KV" isn't a detail — it's <strong>what sets the concurrency limit</strong>.</p>

<table class="t">
  <tr><th>Waste source</th><th>Contiguous reservation (whole floor)</th><th>Paged KV (rent by page)</th></tr>
  <tr><td><strong>Internal frag.</strong></td><td>Reserve for max length, unused slots locked</td><td class="mono"><strong>On-demand</strong> per page, waste <strong>&lt; one page</strong></td></tr>
  <tr><td><strong>External frag.</strong></td><td>Wants one contiguous block, gaps don't combine</td><td class="mono">Physical pages <strong>need not be contiguous</strong>, any free page works</td></tr>
  <tr><td><strong>Fixed length up front?</strong></td><td>Must estimate a cap; too big wastes, too small truncates</td><td class="mono">Grow <strong>page by page</strong> while generating, no need to know length</td></tr>
  <tr><td><strong>Effect on concurrency</strong></td><td>HBM runs out early, <strong>concurrency capped</strong></td><td class="mono">HBM packs tight, <strong>concurrency rises notably</strong></td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="Top half is contiguous allocation: variable-length requests A, B, C leave unusable fragmentation holes between them; a long request needing 6 cells has enough total free space but no contiguous run, so it doesn't fit. Bottom half is paging: KV is cut into fixed-size pages, and the long request's 6 pages fill any free slot, scattered, with zero waste">
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">Contiguous allocation: variable-length requests leave unusable holes</text>
    <rect x="24" y="42" width="732" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="30" y="48" width="120" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="90" y="69" text-anchor="middle" class="mono" style="font-size:11px">req A</text>
    <rect x="154" y="48" width="64" height="32" rx="4" style="fill:var(--faint);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="186" y="68" text-anchor="middle" style="font-size:10px;fill:var(--muted)">waste</text>
    <rect x="222" y="48" width="96" height="32" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="270" y="69" text-anchor="middle" class="mono" style="font-size:11px">req B</text>
    <rect x="322" y="48" width="80" height="32" rx="4" style="fill:var(--faint);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="362" y="68" text-anchor="middle" style="font-size:10px;fill:var(--muted)">waste</text>
    <rect x="406" y="48" width="72" height="32" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="442" y="69" text-anchor="middle" class="mono" style="font-size:11px">req C</text>
    <rect x="482" y="48" width="96" height="32" rx="4" style="fill:var(--faint);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="530" y="68" text-anchor="middle" style="font-size:10px;fill:var(--muted)">waste</text>
    <rect x="582" y="48" width="168" height="32" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="666" y="68" text-anchor="middle" style="font-size:10px;fill:var(--faint)">free</text>
    <text x="24" y="106" style="fill:var(--red);font-size:12px;font-weight:700">Long request needs 6 cells: total free is enough, but no contiguous run → won't fit</text>
    <line x1="24" y1="120" x2="756" y2="120" style="stroke:var(--line);stroke-width:1;stroke-dasharray:5 5"/>
    <text x="24" y="148" style="font-weight:700;fill:var(--accent-ink)">Paging: KV cut into fixed-size pages, fill any free slot → zero waste</text>
    <rect x="24" y="164" width="732" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="30" y="170" width="58" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="90" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="150" y="170" width="58" height="32" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="210" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="270" y="170" width="58" height="32" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="330" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="390" y="170" width="58" height="32" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="450" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="510" y="170" width="58" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="570" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="630" y="170" width="58" height="32" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="690" y="170" width="58" height="32" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="24" y="228" style="fill:var(--teal);font-size:12px;font-weight:700">The long request's 6 red pages scatter everywhere and all fit — physical pages need not be adjacent</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Fragmentation vs paging</b> — contiguous allocation leaves holes a long request can't fit (enough total free space, but no contiguous run); paging cuts KV into fixed-size pages that fill any free slot, scattered, with zero waste.</div>
</div>

<p>A concrete sense of the gap: take <span class="mono">page_size = 16</span> tokens/page. A request that generates only 60 tokens needs just
<span class="mono">⌈60/16⌉ = 4</span> pages (64 slots), wasting only 4 slots on the last page — <strong>under 7% internal fragmentation</strong>; contiguous
reservation against a 2048 cap <strong>wastes ~97% for the same request</strong>. Cutting the fragmentation rate from nearly 50% (a typical contiguous
case) down to <strong>under 5%</strong> is exactly where paging's order-of-magnitude concurrency gain comes from.</p>

<h2>The core of paging: fixed pages + a page table</h2>
<p>PagedAttention needs just two things: <strong>fixed-size pages</strong> and <strong>a page table mapping logical to physical</strong>. Think of a request's
KV as a <strong>logically contiguous token stream</strong>, but underneath it's sliced into pages of <span class="mono">page_size</span> tokens each, with
<strong>every page drawn separately from the pool</strong>. Which page sits where in physical HBM is irrelevant — a <strong>page table</strong> records "page N →
physical block id," and you follow it when you need to fetch.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Slice into pages</h4><p>The token stream is cut by <strong>page_size</strong> (e.g. 16) into pages — <strong>logically contiguous</strong>, physically separate.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Allocate on demand</h4><p>Each time the sequence fills a page, ask the allocator for <strong>one more free block</strong> — <strong>take only what you use</strong>, never reserve to the cap.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Record in the page table</h4><p>Write "<strong>logical page → physical block</strong>" into this request's page table (in SGLang, the token→slot mapping).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Free when done</h4><p>When a request ends, all its pages are <strong>reclaimed in one batch</strong> into the free pool, ready for the next request instantly.</p></div></div>
</div>

<p>The crux is the <strong>page table</strong> in step 3: when the attention kernel looks back at historical K/V, it <strong>doesn't fetch by contiguous address</strong>
but <strong>first consults the page table, then "gathers" K/V from the corresponding physical blocks</strong>. Thanks to this indirection, <strong>physical blocks can
go anywhere — they need not be contiguous</strong>, which is exactly why paging kills external fragmentation. The figure below shows "logically contiguous,
physically scattered, page-table-linked":</p>

<div class="cellgroup">
  <div class="cg-cap"><b>The page table maps logical pages to physical blocks</b>: one request's KV is physically scattered, the table reconciles it — blocks <strong>need not be adjacent</strong></div>
  <div class="cells"><span class="lab">logical page 0</span><span class="cell hl">tok 0–15</span><span class="sep">→</span><span class="cell q">phys block #7</span></div>
  <div class="cells"><span class="lab">logical page 1</span><span class="cell hl">tok 16–31</span><span class="sep">→</span><span class="cell q">phys block #2</span></div>
  <div class="cells"><span class="lab">logical page 2</span><span class="cell hl">tok 32–47</span><span class="sep">→</span><span class="cell q">phys block #9</span></div>
  <div class="cells"><span class="lab">logical page 3</span><span class="cell">tok 48–… (generating)</span><span class="sep">→</span><span class="cell q">grab a free block when the page fills</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 280" role="img" aria-label="The page table maps logical positions to scattered physical pages: on the left a request's logically contiguous token positions 0, 1, 2, 3; in the middle a page table mapping logical pages 0 and 1 to physical block ids #7 and #2; on the right physical pages scattered around, non-adjacent; arrows show logical page 0 pointing down to #7 and logical page 1 pointing up to #2">
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">Logical token positions (contiguous)</text>
    <rect x="24" y="40" width="150" height="30" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="99" y="60" text-anchor="middle" class="mono" style="font-size:11px">pos 0</text>
    <rect x="24" y="74" width="150" height="30" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="99" y="94" text-anchor="middle" class="mono" style="font-size:11px">pos 1</text>
    <rect x="24" y="108" width="150" height="30" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="99" y="128" text-anchor="middle" class="mono" style="font-size:11px">pos 2</text>
    <rect x="24" y="142" width="150" height="30" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="99" y="162" text-anchor="middle" class="mono" style="font-size:11px">pos 3</text>
    <text x="24" y="196" style="font-size:11px;fill:var(--faint)">page_size = 2: pos 0–1 → logical page 0, pos 2–3 → logical page 1</text>
    <text x="312" y="26" style="font-weight:700;fill:var(--accent-ink)">page table</text>
    <rect x="312" y="40" width="190" height="40" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="407" y="65" text-anchor="middle" class="mono" style="font-size:11px">logical page 0 → #7</text>
    <rect x="312" y="92" width="190" height="40" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="407" y="117" text-anchor="middle" class="mono" style="font-size:11px">logical page 1 → #2</text>
    <text x="600" y="26" style="font-weight:700;fill:var(--muted)">Physical pages (scattered)</text>
    <rect x="600" y="40" width="150" height="30" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="675" y="60" text-anchor="middle" class="mono" style="font-size:11px">phys #2</text>
    <rect x="600" y="78" width="150" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="675" y="98" text-anchor="middle" style="font-size:10px;fill:var(--faint)">free #4</text>
    <rect x="600" y="116" width="150" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="675" y="136" text-anchor="middle" style="font-size:10px;fill:var(--faint)">free #5</text>
    <rect x="600" y="154" width="150" height="30" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="675" y="174" text-anchor="middle" class="mono" style="font-size:11px">phys #7</text>
    <rect x="600" y="192" width="150" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 3"/>
    <text x="675" y="212" text-anchor="middle" style="font-size:10px;fill:var(--faint)">free #9</text>
    <path d="M 174 55 C 240 55, 250 60, 312 60" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <path d="M 174 125 C 240 125, 250 112, 312 112" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <path d="M 502 60 C 555 60, 545 169, 600 169" style="fill:none;stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="600,169 586,162 586,176" style="fill:var(--blue)"/>
    <path d="M 502 112 C 555 112, 545 55, 600 55" style="fill:none;stroke:var(--amber);stroke-width:1.5"/>
    <polygon points="600,55 586,48 586,62" style="fill:var(--amber)"/>
    <text x="24" y="218" style="font-size:11px;fill:var(--muted)">logically contiguous · physically scattered · linked by the table</text>
  </svg>
  <div class="figcap"><b>Fig 2 · The page table</b> — logically contiguous token positions (0, 1, 2, 3) map through the page table to scattered physical pages (#7, #2): logically contiguous, physically scattered.</div>
</div>

<p>One often-missed tradeoff: <strong>smaller page_size is not always better</strong>. Smaller pages waste less internally (at most a fraction of a page), but the
page table grows longer and table lookups / kernel indirection cost more; bigger pages address more cheaply but waste more on each request's <strong>last unfilled page</strong>.
Common values are <span class="mono">1, 16, 32</span> and the like, decided by model/hardware tradeoffs — the classic "<strong>fragmentation vs overhead</strong>" balance.</p>

<h2>In SGLang: the paged allocator and req_to_token</h2>
<p>In SGLang this mechanism is carried by a <strong>paged token→KV allocator</strong>: it manages a whole pool of KV slots but hands them out <strong>with the "page" as the
minimum unit</strong>. Paired with the KV pool from Lesson 4, a <span class="inline">req_to_token</span> map records <strong>each request's token → physical slot</strong> — the
in-code incarnation of that "page table." Below is the allocator's skeleton: <strong>alloc takes pages, free returns pages</strong>, all at <span class="mono">page_size</span> granularity:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/allocator/paged.py ::PagedTokenToKVPoolAllocator</span><span class="ln">paged allocator: hands out KV slots by page</span></div>
  <pre><span class="kw">class</span> <span class="st">PagedTokenToKVPoolAllocator</span>(BaseTokenToKVPoolAllocator):
    <span class="cm"># An allocator managing the indices to kv cache data; output is page-aligned</span>
    <span class="kw">def</span> __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size        <span class="cm"># total slots / page_size = #pages</span>

    <span class="kw">def</span> alloc(self, need_size):
        <span class="cm"># page-aligned allocation: return the physical indices of some pages</span>
        num_pages = need_size // self.page_size
        out_pages = self.free_pages[:num_pages]   <span class="cm"># take num_pages off the free list</span>
        self.free_pages = self.free_pages[num_pages:]
        out_indices = (out_pages[:, <span class="kw">None</span>] * self.page_size
                       + torch.arange(self.page_size, device=self.device)).reshape(-1)
        <span class="kw">return</span> out_indices                        <span class="cm"># expand to token-level slot indices</span>

    <span class="kw">def</span> free(self, free_index):
        <span class="cm"># on finish: dedup the page ids it held and return them in one batch</span>
        free_page_indices = torch.unique(free_index // self.page_size)
        self.free_pages = torch.cat((free_page_indices, self.free_pages))</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::ReqToTokenPool</span><span class="ln">maps each request to its token slots (the index layer)</span></div>
  <pre><span class="kw">class</span> <span class="st">ReqToTokenPool</span>:
    <span class="cm"># maps each request -&gt; the token slots that hold its KV (the index layer)</span>
    <span class="kw">def</span> __init__(self, size, max_context_len, ...):
        <span class="cm"># req_to_token[req][pos] -&gt; the physical KV slot for that token</span>
        self.req_to_token = ...   <span class="cm"># shape [size, max_context_len]</span>
    <span class="kw">def</span> alloc(self, reqs):
        ...   <span class="cm"># assign one req_pool_idx row per new request</span>
    <span class="kw">def</span> free(self, req):
        ...   <span class="cm"># release that request's req_pool_idx row back to the pool</span></pre>
</div>

<p>Read this and you grasp the essence of paging: <strong>HBM is no longer reserved wholesale per "request" but retailed per "page."</strong>
<span class="inline">alloc</span> <strong>takes</strong> some pages off <span class="mono">free_pages</span>, then multiplies page ids by <span class="mono">page_size</span> and adds the
in-page offset to <strong>expand into token slot indices</strong>; <span class="inline">free</span> divides those indices by page_size, dedups to page ids, and
<strong>returns them in one batch</strong> when the request ends. Note the class comment "output is page-aligned" — it guarantees pages stay <strong>cleanly aligned and easy to
reclaim</strong>, the physical basis for continuous batching's (Lesson 5) "free on finish, fill on arrival."</p>

<h2>Why this is the springboard to Lesson 7</h2>
<p>Paging's deepest significance isn't just saving HBM. Once KV is cut into <strong>fixed-size, page-table-indexed physical blocks</strong>, something magical becomes possible:
<strong>different requests' page tables can point at the same physical block</strong>. Imagine two requests starting with the same system prompt — their first few pages of KV are
<strong>byte-for-byte identical</strong>, so why store two copies? Just let both page tables <strong>point at that same set of physical pages</strong>. That's exactly what
<strong>Lesson 7's RadixAttention prefix sharing</strong> does.</p>

<div class="flow">
  <div class="node hl"><div class="nt">Fixed pages + page table</div><div class="nd">This lesson: indexed by page, physically scattered</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">A page can be pointed at by many requests</div><div class="nd">freedom from the page-table indirection</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Share prefix KV</div><div class="nd">Lesson 7 RadixAttention</div></div>
</div>

<p>Place this lesson back on Part 2's throughline: Lesson 4 built the KV pool with per-slot management; Lesson 5 kept the batch full via continuous batching; this lesson uses
<strong>paging</strong> to pack each slot <strong>tighter with less fragmentation</strong>, unshackling concurrency from "contiguous reservation"; Lesson 7 then follows the page-table
indirection to make "<strong>can I reuse someone else's KV</strong>" real. Further on, more KV-cache internals and HiCache — tiering KV across HBM/host/disk — unfold in
<strong>Lessons 29–32</strong>. Read on holding twelve words — "<strong>fixed pages, page-table indexed, allocate on demand, free when done</strong>" — and you'll see SGLang's
memory system is built almost entirely on this foundation.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>The pain</strong>: reserving one <strong>contiguous</strong> HBM block per request sized for the max length causes both <strong>internal fragmentation</strong> (most requests never reach max, empty slots locked) and <strong>external fragmentation</strong> (freed gaps can't combine into a contiguous block), burning HBM and <strong>capping concurrency</strong>.</li>
    <li><strong>PagedAttention</strong>: slice KV into <strong>fixed-size pages</strong> and use a <strong>page table</strong> to map <strong>logical position → physical block id</strong> (OS virtual memory / paging). <strong>Allocate on demand, free when done</strong>; physical blocks <strong>need not be contiguous</strong>, so fragmentation nearly vanishes.</li>
    <li><strong>Why non-contiguous works</strong>: the attention kernel <strong>consults the page table, then gathers</strong> K/V from physical blocks, so a page can sit anywhere — the root of killing external fragmentation.</li>
    <li><strong>In SGLang</strong>: <span class="mono">PagedTokenToKVPoolAllocator</span> does <span class="mono">alloc</span>/<span class="mono">free</span> at <span class="mono">page_size</span> granularity, with <span class="mono">req_to_token</span> recording each request's token→slot (the page table, building on Lesson 4's KV pool).</li>
    <li><strong>Tradeoff &amp; forward look</strong>: small page_size means less fragmentation but more lookup overhead, large means the reverse. Paging makes <strong>different requests' page tables point at the same physical block</strong> possible — the springboard for <strong>Lesson 7's RadixAttention</strong> prefix sharing; KV internals and HiCache in <strong>Lessons 29–32</strong>.</li>
  </ul>
</div>
""",
}

LESSON_07 = {
    "zh": r"""
<p class="lead">
上一课的分页给了我们一个意味深长的能力：<strong>不同请求的页表，可以指向同一批物理块</strong>。这一课就把这个能力<strong>用到极致</strong>。
真实流量里有一个被反复忽视的事实——<strong>请求们的开头常常一模一样</strong>：同一段 system prompt、同一组 few-shot 示例、同一套 agent 脚手架，
还有多轮对话里<strong>每一轮都要把整段历史重发一遍</strong>。如果每条请求、每一轮都把这段相同的开头<strong>从头算一遍 KV</strong>，那是天大的浪费。
<strong>RadixAttention</strong> 的答案是：把所有请求的 KV 组织进一棵<strong>基数树</strong>，让相同前缀<strong>只算一次、自动跨请求跨轮复用</strong>。
这正是 SGLang 在真实负载上"快得不讲道理"的核心秘密。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象一份团队<strong>共享的、不断被续写的"目录树"</strong>——有点像编辑器里的<strong>自动补全</strong>。许多文档的开头都一样：
  "<strong>第一章 公司简介……</strong>"。第一个人把这段开头写进了树里、存成几页纸；第二个人要写的文档<strong>开头一字不差</strong>，
  他<strong>根本不必重抄</strong>——顺着树往下走，发现"第一章 公司简介"这条路径已经写好了，<strong>直接复用那几页</strong>，
  只在自己内容<strong>真正不同的地方</strong>另起一个分叉，续上属于自己的新页。谁的开头和别人相同，谁就<strong>白捡</strong>已经写好的部分；
  只有<strong>发散之处</strong>才需要动笔。这棵<strong>共享、协作续写的目录树</strong>，就是 RadixAttention 的精神。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>RadixAttention 把所有请求的 KV 缓存全部存进一棵基数树（压缩前缀树），以 token id 序列为键；
  新请求沿树匹配最长前缀，命中部分的 KV 直接复用、绝不重算，发散的后缀作为新分支插入。</strong>
  缓存键就是 <strong>token 序列本身</strong>，所以共享是<strong>全自动</strong>的——无需用户给任何提示，跨请求、跨对话轮都能命中。
  树的叶子按 <strong>LRU 驱逐</strong>，并用<strong>引用计数</strong>保护：正在被运行中请求使用的 KV<strong>绝不会被驱逐</strong>。
  树节点指向的，正是第 6 课那种<strong>分页物理块</strong>——共享前缀，本质就是<strong>多个节点路径共用同一批物理页</strong>。
</div>

<h2>真实负载都在共享前缀</h2>
<p>先把"浪费"看清楚。生产环境里的请求<strong>极少是孤立的</strong>，它们的开头高度雷同，大致有四类共享来源：
<strong>系统提示</strong>（每条请求都顶着同一段几百 token 的 system prompt）、<strong>few-shot 示例</strong>（同一批演示样例反复前置）、
<strong>多轮对话</strong>（第 N 轮要把前 N−1 轮的全部历史重新发一遍）、<strong>agent 脚手架</strong>（同一套工具说明、格式约束、角色设定打头）。
这些开头动辄数百上千 token，<strong>如果每条请求、每一轮都把它的 KV 从头算一遍</strong>，prefill 的算力就被<strong>大块大块地烧在重复内容上</strong>。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>两条请求共享同一段长前缀</b>：开头的 system prompt 逐 token 相同（高亮），只有结尾发散——共享段的 KV <strong>只算一次</strong>、被两条请求复用</div>
  <div class="cells"><span class="lab">请求 A</span><span class="cell hl">You</span><span class="cell hl">are</span><span class="cell hl">a</span><span class="cell hl">helpful</span><span class="cell hl">assistant</span><span class="sep">|</span><span class="cell q">总结这篇财报</span></div>
  <div class="cells"><span class="lab">请求 B</span><span class="cell hl">You</span><span class="cell hl">are</span><span class="cell hl">a</span><span class="cell hl">helpful</span><span class="cell hl">assistant</span><span class="sep">|</span><span class="cell q">翻成法语</span></div>
  <div class="cells"><span class="lab">复用</span><span class="cell hl">↑ 共享前缀 KV：只 prefill 一次，两条请求都指向同一批物理页 ↑</span><span class="sep">|</span><span class="cell q">发散后缀各算各的</span></div>
</div>

<p>这里要接回第 6 课：因为分页让 KV 变成<strong>固定大小、由页表索引的物理块</strong>，"两条请求复用同一段前缀"才退化成<strong>两条树路径指向同一批物理页</strong>这么轻巧的事。
省下的不止显存——<strong>相同前缀的 prefill 计算也彻底省掉了</strong>。命中率越高，省得越狠，这正是 RadixAttention 价值的来源。
换个角度想：一段 500 token 的系统提示，若有 1000 条请求都顶着它，朴素做法要把这 500 token 的 KV <strong>算上 1000 遍</strong>；
而 RadixAttention 只在<strong>第一条请求</strong>到来时算一次，后面 999 条全是<strong>查树命中、直接复用</strong>。省下的算力不是百分之几，而是<strong>成百上千倍</strong>地压缩了重复 prefill——
这就是为什么命中率高的真实流量里，RadixAttention 几乎是"免费提速"。</p>

<div class="fig">
  <svg viewBox="0 0 760 250" role="img" aria-label="命中缓存省下的计算：请求 1 首次计算 500 token 前缀，请求 2 复用缓存前缀、只计算 20 个新 token">
    <text x="24" y="32" style="font-weight:700;fill:var(--muted)">命中缓存省下的计算</text>
    <text x="24" y="78" style="fill:var(--ink);font-size:13px">请求 1（首次）</text>
    <rect x="170" y="60" width="300" height="36" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="320" y="83" text-anchor="middle" class="mono" style="font-size:12px">前缀 500 token · 首次 prefill 计算</text>
    <text x="24" y="158" style="fill:var(--ink);font-size:13px">请求 2（命中）</text>
    <rect x="170" y="140" width="300" height="36" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="320" y="163" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">复用缓存前缀 500 token · 0 计算</text>
    <rect x="478" y="140" width="60" height="36" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="508" y="163" text-anchor="middle" class="mono" style="font-size:11px">+20</text>
    <text x="548" y="163" style="fill:var(--amber);font-size:12px">新 token 才需计算</text>
    <rect x="170" y="200" width="368" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="354" y="221" text-anchor="middle" style="fill:var(--teal);font-weight:700">命中缓存：省下 500 token 的 prefill，只算 20 token</text>
  </svg>
  <div class="figcap"><b>图 · 命中缓存省下的计算</b> — 请求 2 复用请求 1 的前缀 KV（虚线＝复用、零重算），只为它新增的 20 个 token 付出计算。例：一条 520 token 的请求只算 20，约 <b>96%</b> 的 prefill 被省掉。</div>
</div>

<h2>RadixAttention：把 KV 存进基数树</h2>
<p>核心数据结构是一棵<strong>基数树（radix tree）</strong>，也叫<strong>压缩前缀树</strong>：键是 <strong>token id 序列</strong>，
每条<strong>边</strong>上挂着一段连续的 token 串（一个"token run"）以及对应的<strong>KV 物理块</strong>。一条新请求来了，它要做的事只有一个动词——<strong>沿树往下匹配</strong>：</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>match_prefix：沿树下行匹配</h4><p>拿请求的 token 序列当键，从根节点出发逐边比对，找出<strong>最长的已缓存前缀</strong>。命中多深，就有多少 KV 可以白拿。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>复用命中的前缀 KV</h4><p>匹配到的那段前缀，其 KV <strong>直接复用、零重算</strong>——注意力算子顺着节点拿到物理块索引，就像它本就属于这条请求。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>插入发散后缀为新分支</h4><p>从发散点往后是这条请求<strong>独有</strong>的内容，把它<strong>作为一条新分支插入</strong>树中，并为它新算、新分配 KV 块。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>必要时在发散点分裂边</h4><p>若匹配恰好停在某条边的<strong>中间</strong>，就在该处<strong>把这条边一分为二</strong>（split）：前半段成为共享父节点，后半段与新后缀各成一支。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="基数树：根节点连到共享的系统提示前缀节点，再分叉成两个不同用户问题的子节点；共享前缀用强调色表示只存一份">
    <text x="24" y="32" style="font-weight:700;fill:var(--muted)">基数树：共享前缀只存一份</text>
    <rect x="330" y="48" width="100" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="380" y="70" text-anchor="middle" class="mono" style="font-size:12px">root</text>
    <line x1="380" y1="82" x2="380" y2="116" style="stroke:var(--accent);stroke-width:2"/>
    <rect x="250" y="116" width="260" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="380" y="138" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">前缀："你是一个助手"</text>
    <text x="380" y="153" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">已缓存 · 跨请求复用</text>
    <text x="524" y="138" style="fill:var(--accent);font-size:12px">← 共享前缀只存一份</text>
    <line x1="330" y1="160" x2="200" y2="216" style="stroke:var(--teal);stroke-width:2"/>
    <line x1="430" y1="160" x2="560" y2="216" style="stroke:var(--blue);stroke-width:2"/>
    <rect x="80" y="216" width="240" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="200" y="238" text-anchor="middle" style="fill:var(--ink)">问题 A：总结这篇财报</text>
    <text x="200" y="253" text-anchor="middle" style="fill:var(--teal);font-size:11px">发散后缀 · 各算各的</text>
    <rect x="440" y="216" width="240" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="560" y="238" text-anchor="middle" style="fill:var(--ink)">问题 B：翻成法语</text>
    <text x="560" y="253" text-anchor="middle" style="fill:var(--blue);font-size:11px">发散后缀 · 各算各的</text>
  </svg>
  <div class="figcap"><b>图 · 基数树（共享前缀）</b> — 两条请求共用同一段 system prompt 前缀（强调色，只存一份、只算一次），在真正不同的用户问题处分叉成两支。例：共享 12 个 token，命中后第二条请求只需为它独有的后缀新建一支分支。</div>
</div>

<p>"<strong>分裂边</strong>"是基数树的精髓所在。设树里已存着 <span class="mono">"You are a helpful assistant. Translate"</span> 这一长串，
新请求只共享到 <span class="mono">"You are a helpful assistant. "</span> 就分道扬镳。此时不需要推倒重来，只要在分歧点把原来那条长边<strong>剪成两段</strong>：
公共部分升格为<strong>共享祖先</strong>，原内容的剩余部分与新请求的后缀<strong>各挂一支</strong>。一次廉价的指针操作，就让两条请求<strong>精确共享到逐 token 的边界</strong>。</p>

<h2>跨请求、跨轮自动复用</h2>
<p>因为缓存键就是 <strong>token 序列本身</strong>，共享是<strong>全自动</strong>的：不仅不同请求之间能命中，<strong>同一段多轮对话的相邻轮之间</strong>更是命中大户。
多轮对话的机制是——第 N 轮请求会把<strong>前面所有轮的历史原封不动重发一遍</strong>，再接上这一轮的新问题。于是<strong>前 N−1 轮的全部 KV 都是已缓存的前缀</strong>，
match_prefix 一路命中到历史末尾，<strong>引擎只需为最新一轮真正付出算力</strong>。对话越长，复用的前缀越长，省下的越多。</p>

<div class="flow">
  <div class="node hl"><div class="nt">第 1 轮</div><div class="nd">系统提示 + Q1 完成 prefill，整段进树</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">第 2 轮</div><div class="nd">重发"系统提示+Q1+A1"，前缀全命中，只算 Q2</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">第 3 轮</div><div class="nd">前缀更长、命中更多，只为最新一轮付算力</div></div>
</div>

<p>关键是这一切<strong>不需要用户给任何提示</strong>：你不必告诉引擎"这两条请求开头相同"或"这是同一段对话的第三轮"。
引擎拿到 token 序列，<strong>查一次树就知道能复用多少</strong>。这种"无需配合、自动生效"的特性，让 RadixAttention 在<strong>任意真实流量</strong>上都能吃到红利，
也是它被第 1 课、第 3 课反复点名为 SGLang 招牌特性的原因。配合<strong>缓存感知调度</strong>（第 20 课），引擎甚至会主动把"开头相同"的请求<strong>排到一起</strong>跑，进一步抬高命中率。
值得强调的是，RadixAttention 与前几课的设计是<strong>层层叠加</strong>而非互相替代的：连续批处理负责"把批喂满"，分页负责"把每个槽位装紧"，
而本课负责"<strong>让相同的开头只算一次</strong>"。三者一起作用时，同一张 GPU 在真实对话流量上的有效吞吐，往往是朴素实现的<strong>数倍乃至十几倍</strong>。</p>

<h2>为什么是树而非哈希表，以及如何驱逐</h2>
<p>你也许会问：要查"开头是否相同"，用一个<strong>哈希表</strong>把"整串 prompt → KV"映射起来不也行吗？答案是<strong>远远不够</strong>。
前缀是<strong>层级化</strong>的：请求 A、B、C 可能共享 100 个 token，其中 A、B 又额外多共享 50 个。哈希表只认<strong>整串相等</strong>，
做不到<strong>最长前缀匹配</strong>、做不到<strong>部分复用</strong>、更没有"共享祖先"和"廉价分裂"的概念。基数树则天生为前缀而生：</p>

<table class="t">
  <tr><th>能力</th><th>哈希表（整串映射）</th><th>基数树（压缩前缀树）</th></tr>
  <tr><td><strong>匹配方式</strong></td><td>只命中<strong>完全相同</strong>的整串</td><td class="mono">沿树下行做<strong>最长前缀匹配</strong></td></tr>
  <tr><td><strong>部分复用</strong></td><td>开头相同但结尾不同 ⇒ <strong>全部 miss</strong></td><td class="mono">命中多深复用多深，<strong>逐 token 共享</strong></td></tr>
  <tr><td><strong>层级共享</strong></td><td>无法表达"共享祖先"</td><td class="mono">公共前缀自然成为<strong>共享父节点</strong></td></tr>
  <tr><td><strong>发散处理</strong></td><td>只能新增一条独立条目</td><td class="mono">在分歧点<strong>廉价分裂一条边</strong></td></tr>
</table>

<p>缓存总会满，于是需要<strong>驱逐</strong>。RadixAttention 用 <strong>LRU 策略驱逐叶子</strong>：最久没被访问的叶子节点优先被踢，腾出物理块。
但有一条<strong>铁律</strong>——<strong>引用计数</strong>：每条正在运行的请求会给它正在使用的那条路径<strong>加锁（lock_ref）</strong>，
凡是<strong>引用计数大于 0</strong> 的节点，<strong>绝不会被驱逐</strong>。道理很直白：一条请求正靠着某段前缀 KV 做 decode，
你若把它的 KV 驱逐了，这条请求就<strong>当场崩掉</strong>。所以驱逐只敢动那些<strong>没有任何在跑请求引用的叶子</strong>，安全与高命中率由此兼得。
为什么只驱逐<strong>叶子</strong>而不是中间节点？因为内部节点是<strong>共享祖先</strong>，可能正被许多分支依赖；先从最外层、最久没人用的叶子下手，
既不破坏别人的共享前缀，又能稳稳回收空间。驱逐之后那段前缀只是"忘掉了"，下次再有人用到，重新算一遍、重新进树即可——<strong>正确性永远不受影响，受影响的只是命中率</strong>。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.match_prefix</span><span class="ln">沿基数树匹配最长已缓存前缀</span></div>
  <pre><span class="kw">def</span> match_prefix(self, params: MatchPrefixParams) -&gt; MatchResult:
    <span class="cm"># Find the longest cached prefix of `key` in the radix tree.</span>
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)

    <span class="kw">if</span> self.disable <span class="kw">or</span> len(key) == 0:
        <span class="kw">return</span> self._empty_match_result

    key = key.page_aligned(self.page_size)        <span class="cm"># 截到 page_size 的整数倍</span>

    <span class="cm"># 沿树下行，拿回命中前缀对应的 KV 块索引与终止节点</span>
    value, last_node = self._match_prefix_helper(self.root_node, key)
    <span class="kw">if</span> value:
        value = torch.cat(value)                  <span class="cm"># 命中的各段 KV 索引拼成一条</span>
    <span class="kw">return</span> MatchResult(
        device_indices=value,                     <span class="cm"># 可复用的前缀 KV 索引（可能为空）</span>
        last_device_node=last_node, last_host_node=last_node,
        best_match_node=last_node,
    )</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::TreeNode</span><span class="ln">基数树的节点：children/key/value + 引用计数与访问时间</span></div>
  <pre><span class="kw">class</span> TreeNode:
    <span class="kw">def</span> __init__(self):
        self.children = defaultdict(TreeNode)  <span class="cm"># 基数树的边</span>
        self.parent = <span class="kw">None</span>
        self.key = <span class="kw">None</span>        <span class="cm"># 这条入边上的 token id 区间</span>
        self.value = <span class="kw">None</span>      <span class="cm"># 该区间对应的 KV 缓存槽</span>
        self.lock_ref = 0      <span class="cm"># &gt; 0 表示在用 -&gt; 不会被驱逐</span>
        self.last_access_time = ...   <span class="cm"># 用于 LRU 驱逐</span></pre>
</div>

<p>读懂这段就抓住了 RadixAttention 的灵魂：<strong>一次 match_prefix，就把"这条请求能白拿多少 KV"算清楚了</strong>。
返回的 <span class="inline">device_indices</span> 是<strong>可复用前缀的物理块索引</strong>——长度可能为 0（全新请求），也可能一路命中到历史末尾（多轮对话第 N 轮）。
注释里那句"匹配若停在某段内部会<strong>分裂一次节点</strong>"，正是上文说的廉价 split。把它和第 6 课的分页、第 5 课的连续批处理拼起来，
你就看清了 SGLang 内存系统的三层底座：<strong>按槽独立（第 4 课）→ 分页紧凑（第 6 课）→ 前缀共享（本课）</strong>。</p>

<p>把这一课放回主线：第 4 课建好 KV 池、第 5 课把批喂满、第 6 课用分页装得更紧——本课则让不同请求、不同对话轮<strong>共享同一批前缀 KV</strong>，
把"重复计算"和"重复存储"<strong>一并消灭</strong>。再往后，<strong>缓存感知调度</strong>（第 20 课）会主动撮合"开头相同"的请求以抬高命中率，
而把前缀缓存在<strong>显存/内存/磁盘间分层</strong>的 <strong>HiCache</strong> 会在<strong>第 29–32 课</strong>展开。带着"<strong>token 序列即缓存键、最长前缀即复用</strong>"这把尺子往下读，
你会发现 SGLang 在真实流量上的高吞吐，几乎都源自这棵不起眼的基数树。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>痛点</strong>：真实负载大量<strong>共享前缀</strong>——系统提示、few-shot 示例、多轮对话（每轮重发全史）、agent 脚手架。逐请求/逐轮重算这段开头的 KV 是巨大浪费。</li>
    <li><strong>RadixAttention</strong>：把所有 KV 存进<strong>基数树（压缩前缀树）</strong>，键为 token id 序列；<span class="mono">match_prefix</span> 命中的前缀 KV <strong>直接复用、零重算</strong>，发散后缀<strong>插入新分支</strong>，命中停在边中间则<strong>分裂该边</strong>。</li>
    <li><strong>全自动</strong>：缓存键就是 token 序列，<strong>跨请求、跨对话轮</strong>自动命中，无需用户提示；树节点指向第 6 课的<strong>分页物理块</strong>，共享即多路径共用同一批物理页。</li>
    <li><strong>驱逐</strong>：叶子按 <strong>LRU</strong> 驱逐，但用<strong>引用计数</strong>保护——正被运行中请求使用的 KV <strong>绝不驱逐</strong>，否则该请求会崩。</li>
    <li><strong>为何用树非哈希表</strong>：前缀是<strong>层级化</strong>的，树支持<strong>最长前缀匹配、部分复用、共享祖先、廉价分裂</strong>；整串哈希只认完全相等，开头相同结尾不同即全 miss。前置：分页（第 6 课）、KV 缓存（第 4 课）；延伸：缓存感知调度（第 20 课）、HiCache（第 29–32 课）。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson's paging handed us a loaded capability: <strong>different requests' page tables can point at the same physical blocks</strong>.
This lesson pushes that to the limit. Real traffic hides one constantly-overlooked fact — <strong>requests usually start out identical</strong>:
the same system prompt, the same few-shot examples, the same agent scaffold, and in multi-turn chat <strong>every turn resends the entire history</strong>.
Recomputing that shared opening's KV for every request and every turn is enormous waste. <strong>RadixAttention</strong>'s answer: store every
request's KV in a <strong>radix tree</strong>, so a shared prefix is <strong>computed once and reused automatically, across requests and across turns</strong>.
This is the core secret behind SGLang being "unreasonably fast" on real workloads.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture a team's <strong>shared, continuously-extended "filing tree"</strong> — a bit like editor <strong>autocomplete</strong>. Many documents
  open the same way: "<strong>Chapter 1, Company Overview…</strong>". The first author writes that opening into the tree as a few stored pages;
  the next author whose document <strong>starts word-for-word the same</strong> needn't recopy a thing — walking down the tree, they find the path
  "Chapter 1, Company Overview" already written and <strong>reuse those pages</strong>, branching off only where their text <strong>actually diverges</strong>
  to add their own new pages. Whoever starts like someone else <strong>gets that part for free</strong>; only the <strong>divergence</strong> needs fresh ink.
  That shared, collaboratively-extended filing tree is the spirit of RadixAttention.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>RadixAttention stores every request's KV in a radix tree (compressed prefix trie) keyed by the token-id sequence;
  a new request matches the longest prefix down the tree, reuses the matched KV with zero recompute, and inserts its divergent suffix as a new branch.</strong>
  The cache key is the <strong>token sequence itself</strong>, so sharing is <strong>fully automatic</strong> — no user hints, hitting across requests and across chat turns.
  Leaves are evicted by <strong>LRU</strong> but protected by <strong>reference counting</strong>: KV in use by a running request is <strong>never evicted</strong>.
  Tree nodes point at Lesson 6's <strong>paged physical blocks</strong> — sharing a prefix just means <strong>multiple paths share the same physical pages</strong>.
</div>

<h2>Real workloads share prefixes</h2>
<p>First, see the waste clearly. Production requests are <strong>rarely isolated</strong>; their openings are highly identical, from roughly four sources:
<strong>system prompts</strong> (every request carries the same few-hundred-token preamble), <strong>few-shot examples</strong> (the same demos prepended over and over),
<strong>multi-turn chat</strong> (turn N resends all of turns 1..N−1 verbatim), and <strong>agent scaffolds</strong> (the same tool specs, format rules, persona up front).
These openings run hundreds-to-thousands of tokens; <strong>recomputing their KV from scratch per request and per turn</strong> burns prefill compute <strong>in huge blocks on duplicate content</strong>.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>Two requests share one long prefix</b>: the leading system prompt is token-identical (highlighted); only the tail diverges — the shared KV is <strong>computed once</strong> and reused by both</div>
  <div class="cells"><span class="lab">Req A</span><span class="cell hl">You</span><span class="cell hl">are</span><span class="cell hl">a</span><span class="cell hl">helpful</span><span class="cell hl">assistant</span><span class="sep">|</span><span class="cell q">summarize this report</span></div>
  <div class="cells"><span class="lab">Req B</span><span class="cell hl">You</span><span class="cell hl">are</span><span class="cell hl">a</span><span class="cell hl">helpful</span><span class="cell hl">assistant</span><span class="sep">|</span><span class="cell q">translate to French</span></div>
  <div class="cells"><span class="lab">reuse</span><span class="cell hl">↑ shared prefix KV: prefill once, both requests point at the same physical pages ↑</span><span class="sep">|</span><span class="cell q">divergent suffixes computed separately</span></div>
</div>

<p>Tie this back to Lesson 6: because paging turned KV into <strong>fixed-size, page-table-indexed physical blocks</strong>, "two requests reuse a prefix" collapses to
<strong>two tree paths pointing at the same physical pages</strong> — trivially cheap. And it saves more than memory: <strong>the prefill compute for that shared prefix is gone too</strong>.
The higher the hit rate, the bigger the win — that's where RadixAttention's value comes from.</p>

<div class="fig">
  <svg viewBox="0 0 760 250" role="img" aria-label="Compute saved by a cache hit: request 1 first computes a 500-token prefix, request 2 reuses the cached prefix and computes only 20 new tokens">
    <text x="24" y="32" style="font-weight:700;fill:var(--muted)">Compute saved by a cache hit</text>
    <text x="24" y="78" style="fill:var(--ink);font-size:13px">Request 1 (first time)</text>
    <rect x="170" y="60" width="300" height="36" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="320" y="83" text-anchor="middle" class="mono" style="font-size:12px">prefix 500 tok · first prefill compute</text>
    <text x="24" y="158" style="fill:var(--ink);font-size:13px">Request 2 (hit)</text>
    <rect x="170" y="140" width="300" height="36" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="320" y="163" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">reuse cached prefix 500 tok · 0 compute</text>
    <rect x="478" y="140" width="60" height="36" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="508" y="163" text-anchor="middle" class="mono" style="font-size:11px">+20</text>
    <text x="548" y="163" style="fill:var(--amber);font-size:12px">new tokens to compute</text>
    <rect x="170" y="200" width="368" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="354" y="221" text-anchor="middle" style="fill:var(--teal);font-weight:700">Cache hit: save 500 tokens of prefill, compute only 20</text>
  </svg>
  <div class="figcap"><b>Fig · Compute saved by a cache hit</b> — request 2 reuses request 1's prefix KV (dashed = reused, zero recompute) and pays compute only for its 20 new tokens. e.g. a 520-token request computes only 20 — about <b>96%</b> of prefill is skipped.</div>
</div>

<h2>RadixAttention: store KV in a radix tree</h2>
<p>The core structure is a <strong>radix tree</strong> (a.k.a. <strong>compressed prefix trie</strong>): the key is the <strong>token-id sequence</strong>,
and each <strong>edge</strong> carries a run of consecutive tokens plus the corresponding <strong>KV blocks</strong>. A new request does just one verb — <strong>match down the tree</strong>:</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>match_prefix: walk down the tree</h4><p>Use the request's token sequence as the key; from the root, compare edge by edge to find the <strong>longest cached prefix</strong>. The deeper the hit, the more KV is free.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Reuse the matched prefix KV</h4><p>The matched prefix's KV is <strong>reused with zero recompute</strong> — the attention kernel follows the nodes to the physical block indices, as if it always belonged to this request.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Insert the divergent suffix as a new branch</h4><p>Everything past the divergence point is <strong>unique</strong> to this request; <strong>insert it as a new branch</strong> and compute/allocate fresh KV blocks for it.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Split an edge at the divergence if needed</h4><p>If the match ends in the <strong>middle</strong> of an edge, <strong>split that edge in two</strong>: the front becomes a shared parent, the back and the new suffix each form a branch.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="Radix tree: a root connects to a shared system-prompt prefix node, which forks into two children for two different user questions; the shared prefix uses the accent color to show it is stored once">
    <text x="24" y="32" style="font-weight:700;fill:var(--muted)">Radix tree: a shared prefix is stored once</text>
    <rect x="330" y="48" width="100" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="380" y="70" text-anchor="middle" class="mono" style="font-size:12px">root</text>
    <line x1="380" y1="82" x2="380" y2="116" style="stroke:var(--accent);stroke-width:2"/>
    <rect x="250" y="116" width="260" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="380" y="138" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">prefix: "You are a helper"</text>
    <text x="380" y="153" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">cached · reused across requests</text>
    <text x="524" y="138" style="fill:var(--accent);font-size:12px">← shared prefix stored once</text>
    <line x1="330" y1="160" x2="200" y2="216" style="stroke:var(--teal);stroke-width:2"/>
    <line x1="430" y1="160" x2="560" y2="216" style="stroke:var(--blue);stroke-width:2"/>
    <rect x="80" y="216" width="240" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="200" y="238" text-anchor="middle" style="fill:var(--ink)">Question A: summarize report</text>
    <text x="200" y="253" text-anchor="middle" style="fill:var(--teal);font-size:11px">divergent suffix · computed apart</text>
    <rect x="440" y="216" width="240" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="560" y="238" text-anchor="middle" style="fill:var(--ink)">Question B: translate to French</text>
    <text x="560" y="253" text-anchor="middle" style="fill:var(--blue);font-size:11px">divergent suffix · computed apart</text>
  </svg>
  <div class="figcap"><b>Fig · Radix tree (shared prefix)</b> — two requests share one system-prompt prefix (accent = stored once, computed once) and fork only where the user questions truly differ. e.g. 12 shared tokens; after a hit the second request only builds a branch for its unique suffix.</div>
</div>

<p><strong>Splitting an edge</strong> is the essence of a radix tree. Suppose the tree already holds <span class="mono">"You are a helpful assistant. Translate"</span>,
and a new request only shares up to <span class="mono">"You are a helpful assistant. "</span> before diverging. No need to rebuild — just <strong>cut that long edge in two</strong> at the split:
the common part becomes a <strong>shared ancestor</strong>; the leftover and the new suffix <strong>each hang off it</strong>. One cheap pointer operation lets two requests <strong>share down to the exact token boundary</strong>.</p>

<h2>Automatic reuse across requests and turns</h2>
<p>Because the cache key is the <strong>token sequence itself</strong>, sharing is <strong>fully automatic</strong>: hits happen not only across different requests but —
even more so — <strong>between adjacent turns of one chat</strong>. Multi-turn works like this: turn N resends <strong>all prior turns verbatim</strong>, then appends the new question.
So <strong>all KV of turns 1..N−1 is an already-cached prefix</strong>; match_prefix hits all the way to the end of history, and <strong>the engine only pays compute for the latest turn</strong>.
The longer the chat, the longer the reused prefix, the bigger the savings.</p>

<div class="flow">
  <div class="node hl"><div class="nt">Turn 1</div><div class="nd">system prompt + Q1 prefilled, whole thing enters the tree</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Turn 2</div><div class="nd">resends "prompt+Q1+A1", prefix all hits, only Q2 computed</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Turn 3</div><div class="nd">longer prefix, more hits, pay only for the newest turn</div></div>
</div>

<p>And all of this needs <strong>no hints from the user</strong>: you never tell the engine "these two requests start the same" or "this is turn three of one chat".
It takes the token sequence and <strong>one tree lookup tells it how much can be reused</strong>. This "works automatically, no cooperation required" property lets RadixAttention
profit on <strong>any real traffic</strong> — the reason Lessons 1 and 3 keep naming it a signature SGLang feature. Paired with <strong>cache-aware scheduling</strong> (Lesson 20),
the engine even actively groups same-opening requests together to push hit rates higher.</p>

<h2>Why a tree, not a hash map — and how eviction works</h2>
<p>You might ask: to check "do the openings match", couldn't a <strong>hash map</strong> from "whole prompt → KV" do it? <strong>Far from enough</strong>.
Prefixes are <strong>hierarchical</strong>: requests A, B, C may share 100 tokens, of which A and B share 50 more. A hash map only knows <strong>whole-string equality</strong> —
no <strong>longest-prefix match</strong>, no <strong>partial reuse</strong>, and no notion of "shared ancestor" or "cheap split". A radix tree is built for prefixes:</p>

<table class="t">
  <tr><th>Capability</th><th>Hash map (whole-string)</th><th>Radix tree (compressed trie)</th></tr>
  <tr><td><strong>Matching</strong></td><td>only hits an <strong>exact</strong> whole string</td><td class="mono">walks down for <strong>longest-prefix match</strong></td></tr>
  <tr><td><strong>Partial reuse</strong></td><td>same start, different end ⇒ <strong>full miss</strong></td><td class="mono">reuse as deep as it hits, <strong>token by token</strong></td></tr>
  <tr><td><strong>Hierarchical sharing</strong></td><td>can't express a "shared ancestor"</td><td class="mono">common prefix is a <strong>shared parent</strong> naturally</td></tr>
  <tr><td><strong>Divergence</strong></td><td>only adds a separate entry</td><td class="mono"><strong>cheaply splits an edge</strong> at the fork</td></tr>
</table>

<p>The cache fills up, so it needs <strong>eviction</strong>. RadixAttention evicts <strong>leaves by LRU</strong>: the least-recently-accessed leaf goes first, freeing physical blocks.
But one <strong>iron rule</strong> applies — <strong>reference counting</strong>: each running request <strong>locks (lock_ref)</strong> the path it is using, and any node with
<strong>refcount &gt; 0</strong> is <strong>never evicted</strong>. The reason is plain: a request is decoding against some prefix's KV right now; evict that KV and the request
<strong>crashes on the spot</strong>. So eviction only touches <strong>leaves no running request references</strong> — safety and high hit rate at once.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.match_prefix</span><span class="ln">match the longest cached prefix down the radix tree</span></div>
  <pre><span class="kw">def</span> match_prefix(self, params: MatchPrefixParams) -&gt; MatchResult:
    <span class="cm"># Find the longest cached prefix of `key` in the radix tree.</span>
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)

    <span class="kw">if</span> self.disable <span class="kw">or</span> len(key) == 0:
        <span class="kw">return</span> self._empty_match_result

    key = key.page_aligned(self.page_size)        <span class="cm"># truncate to a multiple of page_size</span>

    <span class="cm"># walk down the tree; get matched-prefix KV indices + terminal node</span>
    value, last_node = self._match_prefix_helper(self.root_node, key)
    <span class="kw">if</span> value:
        value = torch.cat(value)                  <span class="cm"># concat matched KV index segments</span>
    <span class="kw">return</span> MatchResult(
        device_indices=value,                     <span class="cm"># reusable prefix KV indices (may be empty)</span>
        last_device_node=last_node, last_host_node=last_node,
        best_match_node=last_node,
    )</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::TreeNode</span><span class="ln">a radix-tree node: children/key/value + lock-ref and access time</span></div>
  <pre><span class="kw">class</span> TreeNode:
    <span class="kw">def</span> __init__(self):
        self.children = defaultdict(TreeNode)  <span class="cm"># the radix-tree edges</span>
        self.parent = <span class="kw">None</span>
        self.key = <span class="kw">None</span>        <span class="cm"># the token-id span on the edge into this node</span>
        self.value = <span class="kw">None</span>      <span class="cm"># the KV cache slots for that span</span>
        self.lock_ref = 0      <span class="cm"># &gt; 0 means in use -&gt; protected from eviction</span>
        self.last_access_time = ...   <span class="cm"># for LRU eviction</span></pre>
</div>

<p>Read this and you've got RadixAttention's soul: <strong>one match_prefix computes exactly how much KV this request gets for free</strong>.
The returned <span class="inline">device_indices</span> are the <strong>physical block indices of the reusable prefix</strong> — possibly length 0 (a brand-new request),
possibly hitting all the way to the end of history (turn N of a chat). The docstring's note that a match ending inside a segment <strong>splits a node once</strong> is exactly the cheap split above.
Stack this with Lesson 6's paging and Lesson 5's continuous batching and you see SGLang's three-layer memory foundation: <strong>per-slot (L4) → paged (L6) → prefix-shared (this lesson)</strong>.</p>

<p>Back to the main thread: Lesson 4 built the KV pool, Lesson 5 kept the batch full, Lesson 6 packed it tight with paging — this lesson lets different requests and turns
<strong>share the same prefix KV</strong>, killing <strong>both duplicate compute and duplicate storage</strong>. Later, <strong>cache-aware scheduling</strong> (Lesson 20) actively matches up
same-opening requests to raise hit rates, and <strong>HiCache</strong>, which tiers the prefix cache <strong>across HBM/host/disk</strong>, comes in <strong>Lessons 29–32</strong>.
Read on with this yardstick — <strong>the token sequence is the cache key; the longest prefix is reuse</strong> — and you'll find SGLang's real-traffic throughput nearly all springs from this humble radix tree.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Pain</strong>: real workloads heavily <strong>share prefixes</strong> — system prompts, few-shot examples, multi-turn chat (each turn resends all history), agent scaffolds. Recomputing that opening's KV per request/turn is huge waste.</li>
    <li><strong>RadixAttention</strong>: store all KV in a <strong>radix tree (compressed prefix trie)</strong> keyed by token ids; <span class="mono">match_prefix</span> reuses the matched prefix KV with <strong>zero recompute</strong>, <strong>inserts</strong> the divergent suffix as a new branch, and <strong>splits an edge</strong> when the match ends mid-edge.</li>
    <li><strong>Automatic</strong>: the cache key is the token sequence, so hits happen <strong>across requests and across chat turns</strong> with no user hints; tree nodes point at Lesson 6's <strong>paged physical blocks</strong>, sharing = many paths reusing the same pages.</li>
    <li><strong>Eviction</strong>: leaves go by <strong>LRU</strong> but are guarded by <strong>reference counting</strong> — KV in use by a running request is <strong>never evicted</strong>, or that request would crash.</li>
    <li><strong>Why a tree not a hash map</strong>: prefixes are <strong>hierarchical</strong>; a tree gives <strong>longest-prefix match, partial reuse, shared ancestors, cheap splits</strong>; a whole-string hash only knows exact equality and misses entirely when starts match but ends differ. Prereqs: paging (L6), KV cache (L4); extensions: cache-aware scheduling (L20), HiCache (L29–32).</li>
  </ul>
</div>
""",
}

LESSON_08 = {
    "zh": r"""
<p class="lead">
从第 4 课到第 7 课，我们一路在堆"<strong>更快、更省</strong>"的招式：连续批处理把 GPU 喂满、分页让显存装得紧、
RadixAttention 共享前缀。但这些优化最终都要回答一个无法回避的问题：你到底想要<strong>每秒服务最多的请求</strong>，
还是<strong>让每个请求都回得最快</strong>？这两件事<strong>不能同时拉满</strong>。本课是 Part 2 的收尾，也是整个运行时的"<strong>定盘星</strong>"——
理解了<strong>吞吐 vs 延迟</strong>这把天平，你才看得懂后面调度器（第 18 课）存在的全部意义。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象一家餐厅在纠结两种经营哲学。第一种<strong>"让每桌都快"</strong>：客人一坐下立刻上菜、吃完就走，<strong>单桌等待极短</strong>——
  但厨房一次只伺候一两桌，<strong>每小时喂饱的总人数</strong>很有限。第二种<strong>"喂饱最多人"</strong>：攒够一大批客人一起开席、
  厨房一锅出菜摊薄成本，<strong>每小时总产能</strong>拉满——可代价是任何<strong>单个</strong>客人都得排更久的队、等更大批的菜一起好。
  <strong>"上菜快"就是低延迟，"喂人多"就是高吞吐</strong>，而那个"一批坐多少人"的旋钮，<strong>往哪拧都得牺牲另一头</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>吞吐和延迟是同一条曲线的两端，批大小是滑块。</strong>
  批越大，权重读取被摊得越薄、GPU 越饱和，<strong>总吞吐越高</strong>；但每条请求要排在更多活后面、每一步更重，
  <strong>单请求延迟越高</strong>。批越小则反过来：延迟低、吞吐低、GPU 空转。<strong>没有免费的午餐</strong>——你只能根据自己的
  <strong>SLA（服务等级目标）</strong>在曲线上<strong>选一个工作点</strong>。SGLang 的调度器、内存池、分块预填充，
  本质上都是<strong>帮你把这个工作点选得更好</strong>的工具。
</div>

<h2>四个指标：先把"快"和"多"说清楚</h2>
<p>谈权衡之前，得先有<strong>能量化的尺子</strong>。推理服务里有四个核心指标，<strong>各自被不同的瓶颈支配</strong>，
分清它们是理解一切调优的前提：</p>

<table class="t">
  <tr><th>指标</th><th>含义</th><th>主要被什么支配</th></tr>
  <tr><td><strong>TTFT</strong><br><span class="mono">首 token 延迟</span></td><td>从请求到达，到吐出<strong>第一个</strong> token 的时间</td><td class="mono">prefill 计算量 + <strong>排队等待</strong>（队列里前面有多少活）</td></tr>
  <tr><td><strong>ITL / TPOT</strong><br><span class="mono">token 间延迟</span></td><td>生成阶段<strong>相邻两个</strong> token 的间隔（每个输出 token 的时间）</td><td class="mono"><strong>decode 批大小</strong> + 显存带宽（批越大，单步越久）</td></tr>
  <tr><td><strong>Throughput</strong><br><span class="mono">吞吐</span></td><td><strong>所有</strong>请求合计、每秒产出的 token 数</td><td class="mono">GPU 是否饱和、批是否够大够满</td></tr>
  <tr><td><strong>Goodput</strong><br><span class="mono">有效吞吐</span></td><td>其中<strong>满足 SLA</strong> 的那部分吞吐才算数</td><td class="mono">吞吐与延迟的<strong>共同结果</strong></td></tr>
</table>

<p>这张表里藏着最关键的一组对照：<strong>TTFT 主要由 prefill 和排队决定，ITL 主要由 decode 批大小决定</strong>。
为什么要分这么细？因为它们的"敌人"不同——TTFT 怕的是<strong>队列太长</strong>或某条 prompt 极长把大家堵住；
ITL 怕的是<strong>批太大</strong>导致每一步算得太久。一个把 TTFT 压到极低的系统（小批、立刻服务），ITL 和吞吐未必好；
反过来，一个吞吐爆表的系统（大批、攒一攒再发），TTFT 可能很难看。<strong>Goodput</strong> 之所以重要，是因为它逼你同时盯住两端：
吞吐再高，如果一半请求都违反了延迟 SLA，那部分吞吐<strong>对用户毫无价值</strong>。真正要最大化的，是<strong>满足 SLA 前提下的吞吐</strong>。</p>

<p>再把这四个指标放进"用户视角"里体会一遍：用户按下回车后，盯着光标等的那段空白，就是 <strong>TTFT</strong>——它最影响"<strong>这服务卡不卡</strong>"的第一印象；
等文字开始往外蹦，<strong>一个字一个字冒出来的速度</strong>就是 <strong>ITL</strong>，它决定"<strong>读起来顺不顺</strong>"；而运维和老板真正关心的是
<strong>throughput</strong>——同一张昂贵的 GPU，每秒能服务多少 token、摊到每个请求的成本是多少。<strong>goodput</strong> 则是把前三者捏合起来的"<strong>结账指标</strong>"：
只有那些<strong>既产出了 token、又没违反延迟承诺</strong>的请求，才算"真正交付"。一个常见的认知误区是只盯着 throughput 这一个数字去优化，
结果把批拉得过大、TTFT/ITL 双双爆表，<strong>跑分很好看、用户却在抱怨卡顿</strong>——这正是 goodput 要替你纠偏的地方。</p>

<p>把 TTFT 和 ITL/TPOT 放回<strong>一条请求的时间轴</strong>上，它俩的分工就一目了然：TTFT 是开头那一段的"等待"，TPOT 是后面每个字之间的"节奏"。</p>

<div class="fig">
  <svg viewBox="0 0 760 280" role="img" aria-label="一条请求的时间轴：先是一个宽的 PREFILL 块，之后是许多细的 DECODE 刻度；TTFT 是从请求到达到吐出第一个 token 的时间（预填充结束），TPOT 是相邻两个解码 token 之间的间隔">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">TTFT vs TPOT（一条请求的时间轴）</text>
    <line x1="72" y1="60" x2="228" y2="60" style="stroke:var(--accent);stroke-width:1.5"/>
    <line x1="72" y1="54" x2="72" y2="66" style="stroke:var(--accent);stroke-width:1.5"/>
    <line x1="228" y1="54" x2="228" y2="66" style="stroke:var(--accent);stroke-width:1.5"/>
    <text x="150" y="48" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px;font-weight:700">TTFT = 首 token 延迟</text>
    <rect x="72" y="84" width="156" height="80" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="150" y="120" text-anchor="middle" style="font-size:13px;font-weight:700">PREFILL</text>
    <text x="150" y="140" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">并行算整段提示</text>
    <text x="468" y="96" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">DECODE · 逐 token（循环很多次）</text>
    <rect x="240" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="270" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="300" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="330" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="360" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="390" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="420" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="450" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="480" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="510" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="540" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="570" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="600" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="630" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="660" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="690" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <line x1="72" y1="76" x2="72" y2="200" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="228" y1="76" x2="228" y2="200" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="252" y1="182" x2="270" y2="182" style="stroke:var(--red);stroke-width:1.5"/>
    <line x1="252" y1="176" x2="252" y2="188" style="stroke:var(--red);stroke-width:1.5"/>
    <line x1="270" y1="176" x2="270" y2="188" style="stroke:var(--red);stroke-width:1.5"/>
    <text x="261" y="204" text-anchor="middle" style="fill:var(--red);font-size:11px;font-weight:700">TPOT</text>
    <text x="430" y="204" text-anchor="middle" style="fill:var(--red);font-size:11px">= 相邻两个解码 token 的间隔</text>
    <line x1="40" y1="226" x2="720" y2="226" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M720 226 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="668" y="248" style="fill:var(--faint);font-size:11px">时间 →</text>
    <text x="72" y="248" text-anchor="middle" style="fill:var(--faint);font-size:11px">请求到达</text>
    <text x="228" y="266" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">首 token（预填充结束）</text>
  </svg>
  <div class="figcap"><b>图 1 · TTFT vs TPOT（一条请求的时间轴）</b> — 一条请求 = <strong>一个宽 PREFILL 块</strong> + 后面<strong>许多细 DECODE 刻度</strong>。<strong>TTFT</strong> 是从请求到达到吐出<strong>第一个</strong> token（预填充结束）的时间；<strong>TPOT</strong> 是<strong>相邻两个</strong>解码 token 之间的间隔。前者决定"等多久才开口"，后者决定"开口后吐字多快"。</div>
</div>

<p>举个具体的数字感受一下：一个配置良好的 7B 服务，单条请求的 <strong>TTFT≈80&nbsp;ms</strong>（按下回车到第一个字冒出来），<strong>TPOT≈15&nbsp;ms</strong>（之后每个字约 15 毫秒、约合 65&nbsp;token/s 的阅读速度）；而整机在大批并发下的<strong>总吞吐可达数千 token/s</strong>。注意这两组数字会随批大小一起漂移：把并发拉大，总吞吐升到几千，但单条的 TTFT/TPOT 也会跟着变大——这正是下一节那条曲线要刻画的取舍。</p>

<p>这些数字怎么量出来？SGLang 自带的压测脚本会把一次跑测的结果汇总成一个数据类——<strong>吞吐</strong>三件套加上 <strong>TTFT/TPOT 的均值与分位数</strong>，p99 尤其重要，因为它代表<strong>负载下最慢那批用户</strong>的真实体验：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/benchmark/serving.py ::BenchmarkMetrics</span><span class="ln">压测汇总：吞吐 + TTFT/TPOT 的分位数</span></div>
  <pre><span class="kw">@dataclass</span>
<span class="kw">class</span> <span class="fn">BenchmarkMetrics</span>:
    request_throughput: <span class="kw">float</span>   <span class="cm"># requests / second</span>
    input_throughput: <span class="kw">float</span>     <span class="cm"># input tokens / second</span>
    output_throughput: <span class="kw">float</span>    <span class="cm"># output tokens / second (decode speed)</span>
    mean_ttft_ms: <span class="kw">float</span>         <span class="cm"># Time To First Token = prefill latency</span>
    p99_ttft_ms: <span class="kw">float</span>          <span class="cm"># the tail users feel under load</span>
    mean_tpot_ms: <span class="kw">float</span>         <span class="cm"># Time Per Output Token = inter-token latency</span>
    p99_tpot_ms: <span class="kw">float</span>
    <span class="cm"># ... also median / p90 / p95 variants</span></pre>
</div>

<h2>核心张力：批大小这一个旋钮，两头不可兼得</h2>
<p>所有的纠结都收敛到<strong>一个旋钮——批大小（同时在跑的请求数）</strong>上。把它从小往大拧，工作点就沿着曲线滑动，
<strong>吞吐和延迟此消彼长</strong>，请看这条"操作点"的滑动：</p>

<div class="flow">
  <div class="node"><div class="nt">批很小</div><div class="nd">延迟最低（每步轻、马上服务）<br>但吞吐低、<strong>GPU 大量空转</strong></div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">SLA 工作点</div><div class="nd">在<strong>满足延迟 SLA</strong> 的前提下<br>把批尽量调大 ⇒ <strong>goodput 最大</strong></div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">批很大</div><div class="nd">吞吐最高（权重读取摊到最薄、GPU 饱和）<br>但<strong>每请求延迟飙升</strong>、可能违反 SLA</div></div>
</div>

<p>为什么大批<strong>提升吞吐</strong>？回到第 4、5 课：decode 访存密集，读一次权重的开销<strong>不管批里 1 条还是 100 条几乎一样</strong>，
批越大就把这笔固定开销摊得越薄，单位算力产出的有效 token 越多，GPU 越接近满载。为什么大批<strong>又伤延迟</strong>？因为同一个
decode 步要为<strong>更多</strong>请求一起算，<strong>单步耗时变长</strong>（ITL 上升）；而且新来的请求要排在<strong>更多活</strong>后面才轮到，
<strong>TTFT 也被拉长</strong>。这就是天平的两端——<strong>同一个动作，喂饱了吞吐，就拖慢了单请求</strong>。所以不存在"最优批大小"这种放之四海皆准的答案，
<strong>只存在"对你的 SLA 而言最优的工作点"</strong>。</p>

<p>这条曲线还有一个容易被忽略的细节：它<strong>不是一条直线，而是会"拐弯"的</strong>。在批很小的左半段，每往里加一条请求，吞吐几乎<strong>线性上涨</strong>、
而延迟只是温和增加——这是"性价比最高"的一段，加并发近乎白赚吞吐。但一旦 GPU 接近<strong>饱和</strong>，曲线就进入<strong>收益递减区</strong>：
再加请求，吞吐几乎不再涨（GPU 已经满了），延迟却开始<strong>陡峭飙升</strong>（每条都要排更久）。聪明的工作点，恰恰就落在这个<strong>"拐点"附近</strong>——
在它之前你在浪费 GPU，在它之后你在白白牺牲延迟。SGLang 的调度器和这些旋钮，本质上就是在帮你<strong>稳稳地停在拐点上、并随负载漂移而实时微调</strong>。
理解了这一点，你就明白为什么"把并发开到最大"几乎总是错的——那只会把你推到曲线最右端，吞吐没多换来多少，延迟却已崩了。</p>

<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="吞吐与延迟随批大小/并发变化的权衡曲线：吞吐先近似线性上涨、GPU 饱和后趋于平台；单请求延迟一路上升、过拐点后陡升；最佳工作点落在吞吐已高而延迟仍可接受的甜点区附近">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">吞吐 vs 延迟的权衡（横轴 = 批大小 / 并发）</text>
    <text x="100" y="52" style="fill:var(--accent-ink);font-size:12px;font-weight:700">吞吐 tokens/s</text>
    <text x="660" y="52" text-anchor="end" style="fill:var(--red);font-size:12px;font-weight:700">延迟 ms</text>
    <rect x="396" y="56" width="70" height="192" rx="4" style="fill:var(--accent-soft)"/>
    <line x1="431" y1="56" x2="431" y2="248" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="431" y="72" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px;font-weight:700">甜点区 / knee</text>
    <line x1="92" y1="56" x2="92" y2="248" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="672" y1="56" x2="672" y2="248" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="92" y1="248" x2="700" y2="248" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M700 248 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <path d="M92 236 C 180 200, 250 150, 320 122 C 400 100, 520 94, 672 90" style="fill:none;stroke:var(--accent);stroke-width:2.5"/>
    <path d="M92 242 C 220 238, 360 224, 431 205 C 520 180, 600 120, 672 72" style="fill:none;stroke:var(--red);stroke-width:2.5"/>
    <circle cx="431" cy="98" r="4.5" style="fill:var(--accent)"/>
    <text x="168" y="150" style="fill:var(--accent);font-size:11px">近似线性上涨</text>
    <text x="556" y="112" style="fill:var(--accent);font-size:11px">趋于平台</text>
    <text x="560" y="158" style="fill:var(--red);font-size:11px">延迟陡升</text>
    <text x="206" y="236" style="fill:var(--red);font-size:11px">延迟温和</text>
    <text x="431" y="278" text-anchor="middle" style="fill:var(--muted);font-size:11px">吞吐已高、延迟仍可接受</text>
    <text x="384" y="304" text-anchor="middle" style="fill:var(--faint);font-size:11px">批大小 / 并发 →</text>
  </svg>
  <div class="figcap"><b>图 2 · 吞吐–延迟权衡曲线</b> — 横轴是批大小/并发：<strong>吞吐</strong>（紫线）先近似线性上涨、GPU 饱和后趋于<strong>平台</strong>；<strong>单请求延迟</strong>（红线）一路上升、过拐点后<strong>陡升</strong>。最佳工作点落在<strong>甜点区（knee）</strong>——吞吐已经很高、延迟却还在 SLA 之内；拐点左侧在浪费 GPU，右侧在白白牺牲延迟。</div>
</div>

<h2>负载不同，平衡手段也不同</h2>
<p>真实流量并不只有一种形状。<strong>prefill-heavy</strong>（长 prompt、短输出，如文档问答）和 <strong>decode-heavy</strong>（短 prompt、长输出，如长文生成）
对这把天平的拉扯方向完全不同，需要不同的手段去稳住工作点：</p>

<div class="cols">
  <div class="col"><h4>分块预填充（第 22 课）</h4><p>一条<strong>超长 prompt</strong> 的 prefill 会霸占一整步，把所有人的 decode 卡住——
  表现为<strong>突然的延迟尖峰</strong>（ITL 抖动）。分块预填充把长 prefill <strong>切成小块</strong>、穿插进 decode 步里，
  让吞吐和延迟<strong>都不至于因一条长请求崩盘</strong>。</p></div>
  <div class="col"><h4>PD 分离（第 45 课）</h4><p>prefill 和 decode 的瓶颈本就不同（一个算力密集、一个访存密集）。
  <strong>PD 分离</strong>把两个阶段<strong>拆到不同的 GPU 池</strong>上，各自独立调批、独立调优——
  prefill 那边追吞吐，decode 那边保延迟，<strong>互不拖累</strong>。</p></div>
</div>

<p>为什么"负载形状"会改变天平的玩法？因为 <strong>prefill 和 decode 占用的资源根本不是同一种</strong>。
prefill 一次要并行处理 prompt 里成百上千个 token，是<strong>算力密集</strong>的——它能轻易把 GPU 的浮点单元喂满，但耗时长、会<strong>独占整步</strong>；
decode 每步只算一个 token，是<strong>访存密集</strong>的，靠大批来摊薄权重读取。把这两种性格迥异的活混在同一个调度循环里，就会互相伤害：
一条长 prompt 的 prefill 一插进来，正在顺畅 decode 的几十条请求<strong>集体卡顿一下</strong>，用户那头就是<strong>文字突然停顿</strong>的延迟尖峰。
这正是分块预填充与 PD 分离要解决的问题——前者把"插队的大活"切碎、穿插着做，后者干脆把两种活<strong>请到不同的厨房</strong>。
所以同样一句"平衡吞吐与延迟"，在 prefill-heavy 和 decode-heavy 两种负载下，你拧的旋钮、用的手段并不一样。</p>

<h2>SGLang 的旋钮：你怎么在曲线上选点</h2>
<p>这套理论在 SGLang 里落成几个具体的 <span class="inline">server_args</span> 旋钮，外加调度器的<strong>排队策略</strong>。
它们合起来决定了你的工作点落在曲线哪儿：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>SGLang 调延迟/吞吐的主要旋钮</b>：每个旋钮把工作点往"吞吐"或"延迟"一头推</div>
  <div class="cells"><span class="lab mono">max_running_requests</span><span class="cell hl">并发上限</span><span class="sep">→</span><span class="cell q">调大 ⇒ 批更大、吞吐↑、延迟↑；调小则反之（直接限批）</span></div>
  <div class="cells"><span class="lab mono">mem_fraction_static</span><span class="cell hl">静态显存（权重+KV池）占比</span><span class="sep">→</span><span class="cell q">占得多 ⇒ KV 池能装更多 token ⇒ 容得下更高并发 ⇒ 吞吐↑（但留给激活/图的显存变少）</span></div>
  <div class="cells"><span class="lab mono">chunked_prefill_size</span><span class="cell hl">prefill 分块大小</span><span class="sep">→</span><span class="cell q">块越小 ⇒ 长 prompt 越不堵 decode ⇒ 延迟尖峰↓（吞吐略降）</span></div>
  <div class="cells"><span class="lab">调度策略（第 20 课）</span><span class="cell">排队顺序</span><span class="sep">→</span><span class="cell q">决定等待队列里谁先上，平衡<strong>公平性 / 延迟 / 缓存命中</strong></span></div>
</div>

<p>其中 <span class="inline">mem_fraction_static</span> 和并发的关系值得点透：它划走多少 HBM 给 <strong>KV 缓存池</strong>，
就直接决定了<strong>能同时容纳多少 token</strong>（回顾第 4 课），也就决定了<strong>并发上限的天花板</strong>——
KV 池越大、能塞下的请求越多、批越能做大、吞吐越高；但显存是<strong>零和</strong>的，给 KV 多了，留给模型激活和 CUDA Graph 的就少了。
而<strong>谁先从等待队列里被放进来</strong>，则由调度策略拍板。下面这段就是 SGLang 给等待队列<strong>排优先级</strong>的入口——
每个调度步都会调用它，为"这一步先服务谁"定序：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy.calc_priority</span><span class="ln">为等待队列排定本步优先级</span></div>
  <pre><span class="kw">class</span> SchedulePolicy:
    <span class="cm"># …根据策略为 waiting_queue 中的请求排定顺序…</span>

    <span class="kw">def</span> calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = <span class="kw">None</span>
    ) -&gt; <span class="kw">None</span>:
        policy = self._determine_active_policy(waiting_queue)

        <span class="kw">if</span> self.policy == CacheAgnosticPolicy.FCFS:
            <span class="kw">if</span> self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            <span class="kw">return</span></pre>
</div>

<p>看懂这段，你就摸到了天平的"<strong>手柄</strong>"：<span class="inline">calc_priority</span> 不生产 token，它只决定<strong>等待队列里谁排在前面</strong>——
而排队顺序直接影响每条请求的 <strong>TTFT</strong>、影响缓存命中、影响整体 goodput。把它和上面的旋钮放在一起，你就拥有了
<strong>在吞吐-延迟曲线上选点</strong>的全套工具：旋钮决定批能多大、并发多高，策略决定批里<strong>先装谁</strong>。
这正是 Part 5 整个调度器存在的理由——<strong>它就是为了在这条曲线上替你做最优选择而生的</strong>。</p>

<p>最后把第 8 课放回整张地图收个尾。Part 2 这几课其实讲的是同一个故事的不同侧面：第 4 课告诉你 decode 为什么访存密集、KV 为什么按槽位管理；
第 5 课用连续批处理把"批永远满"变现成吞吐；第 6、7 课用分页与前缀共享让显存装得更紧、算得更省。<strong>本课则把前面所有招式统一放上同一杆秤</strong>——
它们最终都是在<strong>吞吐-延迟曲线</strong>上替你争取一个更好的工作点。带着这杆秤往后读，你会发现 Part 5 的调度器、Part 7 的内存设计、
乃至分块预填充与 PD 分离，<strong>没有一个是为"炫技"而存在</strong>：它们要么把曲线整体往右上方推（同样延迟下吞吐更高），要么帮你在曲线上<strong>更稳、更准地选点</strong>。
理解了"<strong>没有免费午餐、只有按 SLA 选点</strong>"这一句，你就拿到了读懂整个 SGLang 运行时的钥匙。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>四个指标</strong>：TTFT（首 token，受 prefill+排队 支配）、ITL/TPOT（token 间隔，受 decode 批大小 支配）、Throughput（全部请求每秒 token 数）、Goodput（满足 SLA 的那部分吞吐）。</li>
    <li><strong>核心张力</strong>：批越大 ⇒ 摊薄权重读取、GPU 饱和 ⇒ 吞吐↑，但每请求排更久、单步更重 ⇒ 延迟↑。批越小则反之。<strong>没有免费午餐，只能按 SLA 选工作点。</strong></li>
    <li><strong>负载相关</strong>：prefill-heavy 与 decode-heavy 拉扯方向不同；分块预填充（第 22 课）防长 prompt 引发延迟尖峰，PD 分离（第 45 课）让两阶段各自独立调优。</li>
    <li><strong>SGLang 旋钮</strong>：<span class="mono">max_running_requests</span>（并发上限）、<span class="mono">mem_fraction_static</span>（KV 池显存 ⇒ 容得下多少并发）、<span class="mono">chunked_prefill_size</span>；调度策略（第 20 课）排队列顺序。</li>
    <li><strong>承上启下</strong>：本课收尾 Part 2，并点明<strong>调度器（第 18 课）存在的全部意义</strong>就是在这条曲线上替你导航。前置：连续批处理（第 5 课）、KV 并发（第 4 课）。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
From Lesson 4 to 7 we stacked tricks for "<strong>faster and cheaper</strong>": continuous batching keeps the GPU fed,
paging packs HBM tightly, RadixAttention shares prefixes. But every one of these must answer one unavoidable question:
do you want to <strong>serve the most requests per second</strong>, or <strong>return each request the fastest</strong>?
You <strong>cannot max both</strong>. This lesson closes Part 2 and is the runtime's <strong>compass</strong> — only once you
grasp the <strong>throughput vs latency</strong> balance does the scheduler (Lesson 18) make full sense.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture a restaurant torn between two philosophies. First, <strong>"make every table fast"</strong>: seat a guest, serve
  instantly, they leave — <strong>per-table wait is tiny</strong>, but the kitchen only handles a table or two at a time, so
  <strong>total people fed per hour</strong> stays low. Second, <strong>"feed the most people"</strong>: gather a big group,
  start a seating together, cook one big batch to amortize cost — <strong>hourly capacity</strong> is maxed, but any
  <strong>single</strong> diner queues longer and waits for a bigger batch to finish. <strong>"Serve fast" is low latency,
  "feed many" is high throughput</strong>, and the "how many per seating" knob <strong>sacrifices one end whichever way you turn it</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>throughput and latency are two ends of one curve, and batch size is the slider.</strong>
  Bigger batches amortize the weight read and saturate the GPU ⇒ <strong>higher total throughput</strong>; but each request
  waits behind more work and each step is heavier ⇒ <strong>higher per-request latency</strong>. Smaller batches flip it:
  low latency, low throughput, idle GPU. <strong>No free lunch</strong> — you can only <strong>pick a point on the curve</strong>
  for your <strong>SLA</strong>. SGLang's scheduler, memory pool, and chunked prefill are all tools to <strong>help you pick that point well</strong>.
</div>

<h2>Four metrics: pin down "fast" vs "many" first</h2>
<p>Before trading off, you need <strong>measurable rulers</strong>. Serving has four core metrics, each <strong>dominated by a
different bottleneck</strong>; telling them apart is the prerequisite for understanding any tuning:</p>

<table class="t">
  <tr><th>Metric</th><th>Meaning</th><th>Dominated by</th></tr>
  <tr><td><strong>TTFT</strong><br><span class="mono">time to first token</span></td><td>From arrival to the <strong>first</strong> token emitted</td><td class="mono">prefill compute + <strong>queueing</strong> (work ahead in queue)</td></tr>
  <tr><td><strong>ITL / TPOT</strong><br><span class="mono">inter-token latency</span></td><td>Gap between <strong>adjacent</strong> output tokens (time per output token)</td><td class="mono"><strong>decode batch size</strong> + memory bandwidth (bigger ⇒ slower step)</td></tr>
  <tr><td><strong>Throughput</strong><br><span class="mono">tokens/sec</span></td><td>Tokens/sec produced across <strong>all</strong> requests</td><td class="mono">GPU saturation; is the batch big and full</td></tr>
  <tr><td><strong>Goodput</strong><br><span class="mono">SLA-meeting tput</span></td><td>Only the throughput that <strong>meets the SLA</strong> counts</td><td class="mono"><strong>joint result</strong> of throughput and latency</td></tr>
</table>

<p>The table hides the key contrast: <strong>TTFT is set by prefill and queueing; ITL is set by decode batch size.</strong>
Why split so finely? Their enemies differ — TTFT fears a <strong>long queue</strong> or one giant prompt blocking everyone;
ITL fears a <strong>batch so big</strong> that each step drags. A system that crushes TTFT (small batch, serve now) may have
poor ITL and throughput; one with sky-high throughput (gather, then send) may have ugly TTFT. <strong>Goodput</strong> matters
because it forces you to watch both ends: however high throughput is, if half the requests violate the latency SLA, that
throughput is <strong>worthless to users</strong>. What you truly maximize is <strong>throughput under the SLA</strong>.</p>

<p>Put TTFT and ITL/TPOT back on <strong>one request's timeline</strong> and their split is obvious: TTFT is the opening "wait", TPOT is the "rhythm" of every character after it.</p>

<div class="fig">
  <svg viewBox="0 0 760 280" role="img" aria-label="One request's timeline: a wide PREFILL block then many thin DECODE ticks; TTFT is the time from request arrival to the first token emitted (end of prefill), TPOT is the gap between consecutive decode tokens">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">TTFT vs TPOT (one request's timeline)</text>
    <line x1="72" y1="60" x2="228" y2="60" style="stroke:var(--accent);stroke-width:1.5"/>
    <line x1="72" y1="54" x2="72" y2="66" style="stroke:var(--accent);stroke-width:1.5"/>
    <line x1="228" y1="54" x2="228" y2="66" style="stroke:var(--accent);stroke-width:1.5"/>
    <text x="150" y="48" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px;font-weight:700">TTFT = time to first token</text>
    <rect x="72" y="84" width="156" height="80" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="150" y="120" text-anchor="middle" style="font-size:13px;font-weight:700">PREFILL</text>
    <text x="150" y="140" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">whole prompt in parallel</text>
    <text x="468" y="96" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">DECODE · per token (loops many times)</text>
    <rect x="240" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="270" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="300" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="330" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="360" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="390" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="420" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="450" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="480" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="510" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="540" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="570" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="600" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="630" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="660" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="690" y="104" width="12" height="60" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <line x1="72" y1="76" x2="72" y2="200" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="228" y1="76" x2="228" y2="200" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="252" y1="182" x2="270" y2="182" style="stroke:var(--red);stroke-width:1.5"/>
    <line x1="252" y1="176" x2="252" y2="188" style="stroke:var(--red);stroke-width:1.5"/>
    <line x1="270" y1="176" x2="270" y2="188" style="stroke:var(--red);stroke-width:1.5"/>
    <text x="261" y="204" text-anchor="middle" style="fill:var(--red);font-size:11px;font-weight:700">TPOT</text>
    <text x="430" y="204" text-anchor="middle" style="fill:var(--red);font-size:11px">= gap between consecutive decode tokens</text>
    <line x1="40" y1="226" x2="720" y2="226" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M720 226 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="668" y="248" style="fill:var(--faint);font-size:11px">time →</text>
    <text x="72" y="248" text-anchor="middle" style="fill:var(--faint);font-size:11px">request arrives</text>
    <text x="228" y="266" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">first token (prefill ends)</text>
  </svg>
  <div class="figcap"><b>Fig 1 · TTFT vs TPOT (one request's timeline)</b> — one request = <strong>one wide PREFILL block</strong> + many <strong>thin DECODE ticks</strong>. <strong>TTFT</strong> is the time from arrival to the <strong>first</strong> token emitted (end of prefill); <strong>TPOT</strong> is the gap between <strong>consecutive</strong> decode tokens. The first decides "how long until it speaks", the second "how fast it types after".</div>
</div>

<p>Some concrete numbers: on a well-tuned 7B service one request sees <strong>TTFT≈80&nbsp;ms</strong> (enter to first token) and <strong>TPOT≈15&nbsp;ms</strong> (~15 ms per token after, ~65&nbsp;tokens/s reading speed), while the whole box under heavy concurrency reaches <strong>several thousand tokens/s total throughput</strong>. Both per-request numbers drift with batch size: crank concurrency and total throughput climbs into the thousands, but each request's TTFT/TPOT grows too — exactly the trade-off the next section's curve captures.</p>

<p>How are these measured? SGLang's bundled benchmark script rolls one run up into a dataclass — the three <strong>throughput</strong> figures plus the <strong>mean and percentiles of TTFT/TPOT</strong>; p99 matters most, since it is the real experience of the <strong>slowest users under load</strong>:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/benchmark/serving.py ::BenchmarkMetrics</span><span class="ln">the benchmark summary: throughput + TTFT/TPOT percentiles</span></div>
  <pre><span class="kw">@dataclass</span>
<span class="kw">class</span> <span class="fn">BenchmarkMetrics</span>:
    request_throughput: <span class="kw">float</span>   <span class="cm"># requests / second</span>
    input_throughput: <span class="kw">float</span>     <span class="cm"># input tokens / second</span>
    output_throughput: <span class="kw">float</span>    <span class="cm"># output tokens / second (decode speed)</span>
    mean_ttft_ms: <span class="kw">float</span>         <span class="cm"># Time To First Token = prefill latency</span>
    p99_ttft_ms: <span class="kw">float</span>          <span class="cm"># the tail users feel under load</span>
    mean_tpot_ms: <span class="kw">float</span>         <span class="cm"># Time Per Output Token = inter-token latency</span>
    p99_tpot_ms: <span class="kw">float</span>
    <span class="cm"># ... also median / p90 / p95 variants</span></pre>
</div>

<h2>The core tension: one knob, batch size, can't win both ends</h2>
<p>All the agonizing converges onto <strong>one knob — batch size (requests running concurrently)</strong>. Turn it from small
to large and the operating point slides along the curve, <strong>throughput and latency trading off</strong>:</p>

<div class="flow">
  <div class="node"><div class="nt">Tiny batch</div><div class="nd">Lowest latency (light step, served now)<br>but low throughput, <strong>GPU idles a lot</strong></div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">SLA point</div><div class="nd">Grow the batch <strong>as large as the latency SLA allows</strong> ⇒ <strong>max goodput</strong></div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Huge batch</div><div class="nd">Max throughput (weight read amortized, GPU saturated)<br>but <strong>per-request latency spikes</strong>, may break SLA</div></div>
</div>

<p>Why do big batches <strong>raise throughput</strong>? Back to Lessons 4–5: decode is memory-bound, and the cost of reading
weights once is <strong>nearly the same for 1 request or 100</strong>; a bigger batch amortizes that fixed cost thinner, yielding
more useful tokens per unit of compute and pushing the GPU toward full load. Why do big batches <strong>hurt latency</strong>?
Because one decode step now computes for <strong>more</strong> requests, so the <strong>step takes longer</strong> (ITL rises),
and a new request waits behind <strong>more work</strong> before its turn (<strong>TTFT lengthens</strong>). That's the balance —
<strong>the same move that feeds throughput slows each request</strong>. So there is no universal "optimal batch size",
<strong>only the optimal operating point for your SLA</strong>.</p>

<div class="fig">
  <svg viewBox="0 0 760 320" role="img" aria-label="Throughput and latency vs batch size/concurrency trade-off curve: throughput rises near-linearly then plateaus once the GPU saturates; per-request latency keeps climbing and spikes after the knee; the best operating point sits near the sweet spot where throughput is already high but latency is still acceptable">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">Throughput vs latency (x = batch size / concurrency)</text>
    <text x="100" y="52" style="fill:var(--accent-ink);font-size:12px;font-weight:700">throughput tokens/s</text>
    <text x="660" y="52" text-anchor="end" style="fill:var(--red);font-size:12px;font-weight:700">latency ms</text>
    <rect x="396" y="56" width="70" height="192" rx="4" style="fill:var(--accent-soft)"/>
    <line x1="431" y1="56" x2="431" y2="248" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="431" y="72" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px;font-weight:700">sweet spot / knee</text>
    <line x1="92" y1="56" x2="92" y2="248" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="672" y1="56" x2="672" y2="248" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="92" y1="248" x2="700" y2="248" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M700 248 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <path d="M92 236 C 180 200, 250 150, 320 122 C 400 100, 520 94, 672 90" style="fill:none;stroke:var(--accent);stroke-width:2.5"/>
    <path d="M92 242 C 220 238, 360 224, 431 205 C 520 180, 600 120, 672 72" style="fill:none;stroke:var(--red);stroke-width:2.5"/>
    <circle cx="431" cy="98" r="4.5" style="fill:var(--accent)"/>
    <text x="168" y="150" style="fill:var(--accent);font-size:11px">near-linear rise</text>
    <text x="560" y="112" style="fill:var(--accent);font-size:11px">plateaus</text>
    <text x="566" y="158" style="fill:var(--red);font-size:11px">latency spikes</text>
    <text x="204" y="236" style="fill:var(--red);font-size:11px">gentle</text>
    <text x="431" y="278" text-anchor="middle" style="fill:var(--muted);font-size:11px">high tput, latency still OK</text>
    <text x="384" y="304" text-anchor="middle" style="fill:var(--faint);font-size:11px">batch size / concurrency →</text>
  </svg>
  <div class="figcap"><b>Fig 2 · The throughput-latency trade-off curve</b> — x is batch size/concurrency: <strong>throughput</strong> (purple) rises near-linearly then <strong>plateaus</strong> once the GPU saturates; <strong>per-request latency</strong> (red) keeps climbing and <strong>spikes</strong> past the knee. The best operating point sits at the <strong>sweet spot (knee)</strong> — throughput already high, latency still within the SLA; left of the knee you waste GPU, right of it you waste latency.</div>
</div>

<h2>Different loads, different balancing tools</h2>
<p>Real traffic isn't one shape. <strong>Prefill-heavy</strong> (long prompt, short output, e.g. doc QA) and <strong>decode-heavy</strong>
(short prompt, long output, e.g. long-form generation) pull the balance in different directions and need different tools:</p>

<div class="cols">
  <div class="col"><h4>Chunked prefill (Lesson 22)</h4><p>One <strong>giant prompt</strong>'s prefill can hog a whole step and stall
  everyone's decode — showing up as a <strong>sudden latency spike</strong> (ITL jitter). Chunked prefill <strong>slices</strong>
  the long prefill and interleaves it with decode steps, so neither throughput nor latency <strong>collapses over one long request</strong>.</p></div>
  <div class="col"><h4>PD disaggregation (Lesson 45)</h4><p>Prefill and decode have different bottlenecks (compute- vs memory-bound).
  <strong>PD disaggregation</strong> splits the two stages onto <strong>separate GPU pools</strong>, each batched and tuned
  independently — prefill chases throughput, decode protects latency, <strong>without dragging each other</strong>.</p></div>
</div>

<h2>SGLang's knobs: how you pick a point on the curve</h2>
<p>This theory lands as a few concrete <span class="inline">server_args</span> knobs in SGLang, plus the scheduler's
<strong>queue policy</strong>. Together they decide where your operating point sits on the curve:</p>

<div class="cellgroup">
  <div class="cg-cap"><b>SGLang's main latency/throughput knobs</b>: each pushes the operating point toward "throughput" or "latency"</div>
  <div class="cells"><span class="lab mono">max_running_requests</span><span class="cell hl">concurrency cap</span><span class="sep">→</span><span class="cell q">Larger ⇒ bigger batch, tput↑, latency↑; smaller flips it (caps the batch directly)</span></div>
  <div class="cells"><span class="lab mono">mem_fraction_static</span><span class="cell hl">static HBM share (weights + KV pool)</span><span class="sep">→</span><span class="cell q">More ⇒ KV pool holds more tokens ⇒ higher concurrency fits ⇒ tput↑ (less HBM left for activations/graphs)</span></div>
  <div class="cells"><span class="lab mono">chunked_prefill_size</span><span class="cell hl">prefill chunk size</span><span class="sep">→</span><span class="cell q">Smaller ⇒ long prompts stall decode less ⇒ latency spikes↓ (tput slightly lower)</span></div>
  <div class="cells"><span class="lab">schedule policy (Lesson 20)</span><span class="cell">queue order</span><span class="sep">→</span><span class="cell q">Decides who in the waiting queue goes first, balancing <strong>fairness / latency / cache hit</strong></span></div>
</div>

<p>The link between <span class="inline">mem_fraction_static</span> and concurrency is worth spelling out: how much HBM it carves
for the <strong>KV cache pool</strong> directly sets <strong>how many tokens fit at once</strong> (recall Lesson 4), hence the
<strong>ceiling on concurrency</strong> — a larger KV pool fits more requests, allows bigger batches, raises throughput; but HBM
is <strong>zero-sum</strong>, so more for KV means less for activations and CUDA Graphs. And <strong>who gets admitted from the
waiting queue</strong> is decided by the schedule policy. Below is SGLang's entry point that <strong>prioritizes</strong> the
waiting queue — called every schedule step to order "who to serve first this step":</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy.calc_priority</span><span class="ln">orders the waiting queue this step</span></div>
  <pre><span class="kw">class</span> SchedulePolicy:
    <span class="cm"># …order the requests in waiting_queue per the active policy…</span>

    <span class="kw">def</span> calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = <span class="kw">None</span>
    ) -&gt; <span class="kw">None</span>:
        policy = self._determine_active_policy(waiting_queue)

        <span class="kw">if</span> self.policy == CacheAgnosticPolicy.FCFS:
            <span class="kw">if</span> self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            <span class="kw">return</span></pre>
</div>

<p>Read this and you've grabbed the balance's <strong>handle</strong>: <span class="inline">calc_priority</span> produces no tokens —
it only decides <strong>who sits at the front of the waiting queue</strong>, which directly shapes each request's <strong>TTFT</strong>,
cache hits, and overall goodput. Put it alongside the knobs above and you hold the full toolkit for <strong>picking a point on the
throughput-latency curve</strong>: the knobs set how big the batch and concurrency can be, the policy sets <strong>who gets loaded
first</strong>. This is exactly why Part 5's scheduler exists — <strong>it is built to make that optimal choice on the curve for you</strong>.</p>

<p>To close, put Lesson 8 back on the whole map. The Part 2 lessons are facets of one story: Lesson 4 told you why decode is
memory-bound and why KV is managed by slots; Lessons 5–7 are three ways to lift the throughput end (batch more, pack tighter,
reuse prefixes); and this lesson hands you the <strong>scale</strong> they all sit on. Read on with it and you'll see Part 5's
scheduler, <strong>Part 7's memory design</strong>, and even chunked prefill and PD disaggregation are all just trying to win you a
better operating point on the <strong>throughput-latency curve</strong>. That single ruler — "what does this design buy on the
curve, and what does it cost?" — is the lens for everything that follows.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Four metrics</strong>: TTFT (first token, set by prefill+queueing), ITL/TPOT (inter-token gap, set by decode batch size), Throughput (tokens/sec across all requests), Goodput (the throughput that meets the SLA).</li>
    <li><strong>Core tension</strong>: bigger batch ⇒ amortized weight read, saturated GPU ⇒ tput↑, but each request waits longer and each step is heavier ⇒ latency↑. Smaller flips it. <strong>No free lunch — pick the operating point by SLA.</strong></li>
    <li><strong>Load-dependent</strong>: prefill-heavy vs decode-heavy pull differently; chunked prefill (Lesson 22) stops long prompts from causing latency spikes, PD disaggregation (Lesson 45) tunes each stage independently.</li>
    <li><strong>SGLang knobs</strong>: <span class="mono">max_running_requests</span> (concurrency cap), <span class="mono">mem_fraction_static</span> (KV-pool HBM ⇒ how much concurrency fits), <span class="mono">chunked_prefill_size</span>; schedule policy (Lesson 20) orders the queue.</li>
    <li><strong>Closing Part 2</strong>: this lesson ends Part 2 and shows that <strong>the scheduler (Lesson 18) exists</strong> to navigate this very curve for you. Prereqs: continuous batching (Lesson 5), KV concurrency (Lesson 4).</li>
  </ul>
</div>
"""}
