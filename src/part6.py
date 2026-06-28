"""Part 6 - Model execution. Lessons (L24-L28) for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons cover what run_batch triggers on the GPU: the ModelRunner +
ForwardBatch (L24), model loading & weights (L25), how a model file is written
(L26, Llama as the example), CUDA graph capture/replay (L27), and the Sampler
turning logits into the next token (L28).
"""

LESSON_24 = {
    "zh": r"""
<p class="lead">
第五部分我们一直在讲调度器怎么"<strong>决策</strong>"——收请求、组批、定策略。但它<strong>只决策、不亲手算</strong>。
真正在 GPU 上把一批 token 跑成 logits 的，是这一课的主角 <span class="inline">ModelRunner</span>。它是
<strong>"决策"变成"计算"的那道门</strong>：第五部分到此为止是控制流，从这一课起我们走进真正的数学。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 ModelRunner 想成车间里的<strong>机台操作工</strong>：调度器（工头）递来一张<strong>工单</strong>（ForwardBatch：要算哪些 token、
  是预填充还是解码、KV 放在哪），操作工照单<strong>开动机器</strong>（<span class="mono">model.forward</span>），读出<strong>仪表读数</strong>（logits），
  再把读数交给<strong>质检员</strong>（Sampler）挑出这一步的产物（下一个 token）。工头从不碰机器，碰机器的只有操作工。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  每个 TP rank 有<strong>一个 ModelRunner</strong>（由 TpModelWorker 持有），它握着<strong>加载好的模型</strong>（第 25/26 课）和
  <strong>KV 缓存池</strong>（第 30 课）。一次 <span class="mono">forward(forward_batch)</span> 把一批请求跑出 logits，再由
  <span class="mono">sample()</span> 交给采样器（第 28 课）出 token。而 <span class="inline">ForwardBatch</span> 就是
  <strong>ScheduleBatch（第 19 课）的 GPU 视图</strong>——同一批请求，换成 GPU 看得懂的张量与元数据。
</div>

<h2>run_batch 之后：谁把这一批递到了 ModelRunner 手里</h2>
<p>先把调用链补全。第 18 课的调度器主循环每一步都会调 <span class="mono">run_batch</span>，但 run_batch 本身其实很<strong>薄</strong>：它把刚组好的 <span class="mono">ScheduleBatch</span> 交给 <strong>TpModelWorker</strong>，TpModelWorker 再原样转手喂给它<strong>独占的那一个 ModelRunner</strong>。这条链路之所以要分这么多层，是因为每一层只盯着一件事——调度器只关心"这一步该算哪一批"，TpModelWorker 只关心"我这个 rank 的边界与跨卡通信"，ModelRunner 只关心"把张量在这张卡上真正算出来"。层与层之间职责清晰、互不越界，正是这套代码能被上千名贡献者协作维护的原因。</p>
<p>这条链还藏着一个容易忽视却致命的约束：在 8 路 TP 的部署里，8 个 TpModelWorker 各自的 ModelRunner 必须<strong>严格同步地跑同一批</strong>。它们各算模型的一片（权重按行/列切开，第 25/46 课），层中途要靠 all-reduce / all-gather 把注意力输出和残差拼齐——只要有一个 rank 早走或晚走半步，集合通信就会<strong>整组卡死</strong>。这就是为什么进入前向之前，ScheduleBatch 的内容必须在所有 rank 上<strong>逐字节一致</strong>：同样的 token、同样的形状、同样的 forward_mode。理解了这一点，你就明白第 18 课强调的"调度决策必须确定性、可复现"不是洁癖，而是分布式前向能不死锁的硬前提。</p>

<h2>从 ScheduleBatch 到 ForwardBatch：换一副 GPU 的眼镜</h2>
<p>先厘清一个常被混淆的层级：<strong>TpModelWorker</strong> 是张量并行里"一个 rank"的封装，它内部<strong>持有一个 ModelRunner</strong>；
所以在 8 路 TP 的部署里，有 8 个 TpModelWorker、8 个 ModelRunner，各自<strong>只算模型的一片</strong>（权重按列/行切开，第 25/46 课），
再靠集合通信把结果拼起来。调度器（第 18 课）每步调 <span class="mono">run_batch</span>，本质就是把活儿派给这些 ModelRunner。
记住这条分工：<strong>调度器只有一个大脑、负责决策；ModelRunner 有很多双手、负责把决策算出来</strong>。</p>
<p>调度器交出来的是 <span class="mono">ScheduleBatch</span>（一堆 Req 的集合，第 19 课）。ModelRunner 要先把它翻译成
<span class="mono">ForwardBatch</span>——GPU 前向真正需要的那几样东西：<strong>input_ids</strong>（这一步要喂的 token）、
<strong>positions</strong>（每个 token 的位置，喂给 RoPE）、<strong>forward_mode</strong>（是 EXTEND/预填充还是 DECODE/解码）、
<strong>注意力元数据</strong>（用哪个 attention 后端、KV 该读写到池子的哪些槽位，第 33 课）、以及指向 <strong>KV 池</strong>的指针。
一句话：ScheduleBatch 记的是"<strong>谁要算</strong>"，ForwardBatch 记的是"<strong>GPU 具体怎么算这一步</strong>"。
这层翻译看似琐碎，却是性能关键：positions、注意力元数据、KV 槽位若组织得当，下游的注意力内核（第 33 课）和 CUDA 图（第 27 课）才能跑得又快又稳。</p>

<div class="flow">
  <div class="node"><div class="nt">ScheduleBatch</div><div class="nd">一批 Req（第 19 课）</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">ForwardBatch</div><div class="nd">GPU 视图：ids/positions/mode/KV</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">model.forward</div><div class="nd">嵌入→层→norm→lm_head</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">logits</div><div class="nd">每条请求的下一词分布</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Sampler</div><div class="nd">挑出下一个 token（第 28 课）</div></div>
</div>

<table class="t">
  <tr><th>ForwardBatch 字段</th><th>是什么</th><th>给谁用</th></tr>
  <tr><td class="mono">input_ids</td><td>这一步要前向的 token id</td><td>嵌入层</td></tr>
  <tr><td class="mono">positions</td><td>每个 token 的绝对位置</td><td>RoPE / 位置编码（第 36 课）</td></tr>
  <tr><td class="mono">forward_mode</td><td>EXTEND（预填充）/ DECODE（解码）</td><td>决定走哪条前向路径</td></tr>
  <tr><td class="mono">attn metadata</td><td>注意力后端 + KV 读写槽位</td><td>attention 内核（第 33 课）</td></tr>
  <tr><td class="mono">out_cache_loc</td><td>新 token 的 K/V 写到池子哪里</td><td>KV 缓存池（第 30 课）</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="一次前向：ModelRunner 拿到一个 ForwardBatch（input_ids、positions、seq_lens、out_cache_loc），从左到右跑过模型的层堆叠，产出下一个 token 的 logits">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">一次前向：输入 → 模型 → logits</text>
    <rect x="24" y="58" width="210" height="150" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="129" y="84" text-anchor="middle" style="font-weight:700;fill:var(--blue)">ForwardBatch</text>
    <rect x="40" y="98" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="113" class="mono" style="font-size:11px">input_ids</text>
    <rect x="40" y="124" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="139" class="mono" style="font-size:11px">positions</text>
    <rect x="40" y="150" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="165" class="mono" style="font-size:11px">seq_lens</text>
    <rect x="40" y="176" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="191" class="mono" style="font-size:11px">out_cache_loc</text>
    <line x1="234" y1="133" x2="300" y2="133" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="300,128 312,133 300,138" style="fill:var(--line)"/>
    <rect x="312" y="58" width="230" height="150" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="427" y="84" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">model（层堆叠）</text>
    <rect x="330" y="98" width="194" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="427" y="115" text-anchor="middle" style="font-size:12px">嵌入 embed</text>
    <rect x="330" y="128" width="194" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="427" y="145" text-anchor="middle" style="font-size:12px">N 层解码层</text>
    <rect x="330" y="158" width="194" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="427" y="175" text-anchor="middle" style="font-size:12px">norm + lm_head</text>
    <line x1="552" y1="133" x2="612" y2="133" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="612,128 624,133 612,138" style="fill:var(--line)"/>
    <rect x="632" y="92" width="144" height="82" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="704" y="128" text-anchor="middle" style="font-weight:700;fill:var(--teal)">logits</text>
    <text x="704" y="150" text-anchor="middle" style="font-size:11px;fill:var(--muted)">下一个 token 的分布</text>
  </svg>
  <div class="figcap"><b>图 1 · 一次前向</b> — ModelRunner 拿到一个 ForwardBatch（input_ids / positions / seq_lens / out_cache_loc），从左到右跑过模型的层堆叠（嵌入 → N 层解码层 → norm + lm_head），最后产出下一个 token 的 logits。</div>
</div>

<p>这套字段并不是凭空设计的，它直接对应源码里 <span class="mono">ForwardBatch</span> 这个数据类——一次前向所需的全部输入，被打包进同一个结构体里：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/forward_batch_info.py ::ForwardBatch</span><span class="ln">一次前向所需的全部输入打成一个结构体</span></div>
  <pre><span class="kw">class</span> ForwardBatch:
    <span class="cm"># 一次前向的全部输入，打包在一起。</span>
    forward_mode: ForwardMode      <span class="cm"># EXTEND（预填充）或 DECODE（解码）</span>
    batch_size: int
    input_ids: torch.Tensor        <span class="cm"># 这一步要跑的 token</span>
    req_pool_indices: torch.Tensor <span class="cm"># 哪些请求（req_to_token 里的行）</span>
    seq_lens: torch.Tensor         <span class="cm"># 每条请求的上下文长度</span>
    out_cache_loc: torch.Tensor    <span class="cm"># 新 KV 写到哪里</span>
    ...</pre>
</div>

<p>把这张表再往下读一层会更有味道。<strong>positions</strong> 绝不是摆设：它直接喂进 RoPE，决定每个 token 的旋转相位，所以预填充和解码一旦把绝对位置算错，注意力就会"看错"远近关系，输出立刻乱套——这也是分块预填充（第 22 课）必须小心维护 positions 连续性的原因。<strong>forward_mode</strong> 像一个总闸，它一拨下去就决定了后面整条路径：走变长的预填充内核还是规整的解码内核、能不能套 CUDA 图、注意力后端该用哪种 mask。<strong>注意力元数据</strong>则负责把"逻辑上的第 k 个 token"翻译成"物理上 KV 池里的哪一页、哪一个槽位"，它和 <span class="mono">out_cache_loc</span> 配合，明确告诉内核：新算出来的 K/V 写到哪、历史 K/V 又从哪读。可以说，ForwardBatch 这几个字段合起来就是一张"<strong>GPU 寻址表</strong>"，前向里的每一次读写都照着它走；它们组织得好不好，直接决定下游注意力内核（第 33 课）和 CUDA 图（第 27 课）能不能既快又稳。</p>

<h2>ModelRunner.forward：一次前向怎么走</h2>
<p>核心分两条路：能用 <strong>CUDA Graph</strong>（第 27 课）就<strong>重放</strong>一张录好的图，否则<strong>即时（eager）</strong>地把这一批真算一遍。
判断的关键是 <span class="mono">can_run_graph</span>——通常<strong>解码步</strong>（形状规整、batch 小）才走图重放，<strong>预填充</strong>因为长度多变多走即时路径
（或分片预填充的图）。下面是 <span class="mono">_forward_raw</span> 的真实骨架：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/model_runner.py ::ModelRunner.forward</span><span class="ln">前向的分派</span></div>
  <pre><span class="kw">def</span> _forward_raw(self, forward_batch, pp_proxy_tensors, ...):
    <span class="cm"># 能否重放解码 CUDA 图：模式匹配 + 有图 + 形状能套</span>
    can_run_graph = (forward_batch.forward_mode.is_cuda_graph()
                     <span class="kw">and</span> self.decode_cuda_graph_runner
                     <span class="kw">and</span> self.decode_cuda_graph_runner.can_run_graph(forward_batch))
    <span class="kw">if</span> can_run_graph:                          <span class="cm"># 解码：重放录好的图（第 27 课）</span>
        ret = self.decode_cuda_graph_runner.execute(forward_batch, ...)
        <span class="kw">return</span> ModelRunnerOutput(logits_output=ret, can_run_graph=True)
    self._prepare_eager_forward_batch(forward_batch)  <span class="cm"># 即时路径：DP/padding 归一</span>
    <span class="kw">if</span> forward_batch.forward_mode.is_extend(...):   <span class="cm"># 预填充：真算这一批</span>
        ...                                       <span class="cm"># 否则解码 eager 兜底</span></pre>
</div>

<p>不管走哪条路，最终都进到 <span class="mono">model.forward(input_ids, positions, forward_batch)</span>：嵌入 → 一层层
<strong>解码层</strong>（注意力 + MLP，第 26 课）→ 末尾归一化 → <span class="mono">lm_head</span> 投影出 <strong>logits</strong>。
ModelRunner 再调 <span class="mono">sample()</span> 把 logits 交给采样器（第 28 课）。</p>

<p>再细看这段分派为什么这么写。第一道判断 <span class="mono">can_run_graph</span> 是三个条件的<strong>与</strong>：模式得是能套图的（解码类）、这张图<strong>确实已经录过</strong>、并且当前 batch 的形状能套进某个录好的桶。三者缺一，就走不了重放——这正是为什么图重放被放在最前面短路返回：它是"快路径"，能走则走、走不了再退而求其次。退下来后先调 <span class="mono">_prepare_eager_forward_batch</span>，它做的是即时路径的<strong>归一化收尾</strong>：补齐数据并行（DP）下各 rank 的 padding、对齐张量形状，让后面的内核能正常跑。归一化之后再按 <span class="mono">is_extend</span> 分叉——预填充就老老实实把这一批真算一遍，否则落到解码的即时兜底。把这段读懂，你就能把"为什么我的解码没走 CUDA 图"这类问题精确定位到三个开关上的某一个，而不是泛泛地猜。</p>

<p>值得多看一眼<strong>注意力层在这里干了什么</strong>：它从 ForwardBatch 拿到 KV 槽位，把当前 token 的 K/V <strong>写进 KV 池</strong>，
再把当前 Q 和<strong>历史 K/V</strong>（可能是 RadixAttention 复用来的前缀，第 7 课）做注意力。换句话说，第 4–7 课讲的缓存与复用，
正是在 ModelRunner 跑 forward 的这一刻<strong>真正发生</strong>的——前面几课建立的所有直觉，到这里第一次"通上电"。也正因为前向要频繁读写 KV 池、
要调用具体硬件的注意力内核，ModelRunner 必须<strong>同时</strong>握住模型、KV 池和注意力后端三样东西，缺一不可。</p>

<p>沿着前向再走细一点。嵌入层把 input_ids 查成词向量后，数据进入一摞结构完全相同的<strong>解码层</strong>。每一层里，先做一次归一化，再进注意力子层：Q/K/V 由线性层投影出来，K/V 顺手按 <span class="mono">out_cache_loc</span> 写进 KV 池，注意力内核再让当前 Q 去看自己以及所有历史 K/V。如果这段前缀此前已被别的请求算过、并被 RadixAttention（第 7 课）挂在基数树上，那么这部分历史 K/V 根本不必重算，<strong>直接复用</strong>——第 4–7 课讲的缓存复用，到这一层才落到具体内核上。注意力输出经残差相加，再进 MLP 子层（又一次归一化 + 两个线性层 + 激活），又一次残差。如此 N 层叠完，做末端归一化，<span class="mono">lm_head</span> 把最后的隐藏向量投影到整个词表维度，得到 <strong>logits</strong>。这里有个常被忽略的优化：解码时其实只需要<strong>每条请求最后一个位置</strong>的 logits（下一个词只看最后一步），所以前向通常先做一次"取最后 token"的切片再投影，省下大量无用的 lm_head 计算。logits 一到手，ModelRunner 立刻调 <span class="mono">sample()</span>，下一课的采样器才正式登场——<strong>整条流水线里 logits→token 的那个转折点，就发生在这里</strong>。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>嵌入 Embed</h4><p>input_ids → 词向量；positions 备好给 RoPE。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>N 层解码层</h4><p>每层：注意力（读写 KV 池）+ MLP + 归一化（第 26 课）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>末端归一 + lm_head</h4><p>最后一层 hidden → 词表维度的 <strong>logits</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>采样</h4><p><span class="mono">sample()</span> → Sampler 出下一个 token（第 28 课）。</p></div></div>
</div>

<h2>EXTEND 与 DECODE：同一个 forward，两种性格</h2>
<p>还记得第 4 课的两阶段吗？它们在 ModelRunner 这里落成两种前向：</p>

<div class="cols">
  <div class="col"><h4>EXTEND / 预填充前向</h4><p>一次处理<strong>整段提示</strong>的多个 token（变长），<strong>计算密集</strong>、并行度高；
  形状多变，多走<strong>即时</strong>路径或分片/分块（第 22 课）。它把这批 token 的 K/V <strong>写满</strong>缓存。</p></div>
  <div class="col"><h4>DECODE / 解码前向</h4><p>每条请求<strong>只算 1 个新 token</strong>，形状规整、batch 固定，正好适合
  <strong>CUDA Graph 重放</strong>（第 27 课）——把上百个小内核的启动开销一次性抹掉。它给缓存<strong>追加一行</strong>。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="ForwardBatch 把若干预填充与解码请求并进一个批：一条 200 token 的预填充加上三条各 1 token 的解码，拼成长度 203 的 input_ids，由一次前向消费">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">ForwardBatch：prefill 与 decode 并进一个批</text>
    <rect x="24" y="52" width="220" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="34" y="70" style="font-size:12px;fill:var(--amber);font-weight:700">req1 · EXTEND / 预填充</text>
    <text x="34" y="86" class="mono" style="font-size:11px;fill:var(--amber)">200 tokens</text>
    <rect x="24" y="100" width="220" height="32" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="34" y="120" style="font-size:12px;fill:var(--purple)">req2 · DECODE · 1 token</text>
    <rect x="24" y="140" width="220" height="32" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="34" y="160" style="font-size:12px;fill:var(--purple)">req3 · DECODE · 1 token</text>
    <rect x="24" y="180" width="220" height="32" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="34" y="200" style="font-size:12px;fill:var(--purple)">req4 · DECODE · 1 token</text>
    <line x1="254" y1="132" x2="326" y2="132" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="326,127 338,132 326,137" style="fill:var(--line)"/>
    <rect x="346" y="70" width="300" height="120" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="496" y="96" text-anchor="middle" style="font-weight:700;fill:var(--blue)">一个 ForwardBatch · input_ids</text>
    <rect x="360" y="118" width="170" height="30" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <text x="445" y="138" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--amber)">200（req1 预填充）</text>
    <rect x="540" y="118" width="28" height="30" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <rect x="572" y="118" width="28" height="30" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <rect x="604" y="118" width="28" height="30" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <text x="496" y="174" text-anchor="middle" class="mono" style="font-size:11px">长度 = 200 + 1 + 1 + 1 = 203</text>
    <line x1="496" y1="190" x2="496" y2="222" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="491,222 496,234 501,222" style="fill:var(--line)"/>
    <rect x="396" y="236" width="200" height="50" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="496" y="266" text-anchor="middle" style="font-weight:700;fill:var(--teal)">一次 forward 吃下整批</text>
  </svg>
  <div class="figcap"><b>图 2 · 一个批装下两种请求</b> — 一条 EXTEND/预填充（200 token）和三条 DECODE/解码（各 1 token）被打进同一个 ForwardBatch，<span class="mono">input_ids</span> 长度 = 200+1+1+1 = 203，由一次前向一口气算完。</div>
</div>

<p>举个具体例子：把 <strong>3 条解码请求</strong>（每条只算 1 个新 token）和 <strong>1 条 200 token 的预填充</strong>请求拼进<strong>同一个 ForwardBatch</strong>，<span class="mono">input_ids</span> 的长度就是 <span class="mono">200 + 1 + 1 + 1 = 203</span>。预填充那段的 <span class="mono">forward_mode</span> 是 <span class="mono">EXTEND</span>（一次喂进 200 个 token），三条解码则是 <span class="mono">DECODE</span>（每条只追加 1 个 token）；它们共用同一次前向、同一份注意力元数据，由 GPU 一口气算完，这正是连续批处理（第 5 课）把预填充和解码混批跑的样子。</p>

<p>为什么偏偏是<strong>解码</strong>适合 CUDA 图、而预填充不适合？关键在"形状是否稳定"。CUDA 图录制的是一串固定的内核调用和固定的显存地址，录好之后重放时只能套用<strong>同样的张量形状、同样的指针</strong>。解码每步每条请求恰好只产出 1 个新 token，batch 大小在一段时间内不变、序列每步只长 1，形状高度规整，于是可以把不同的实际长度<strong>padding 到几档固定的桶</strong>，每档录一张图反复重放——而解码恰恰是启动开销占比最高的阶段（每步算得少、内核调得多），抹平这部分开销收益最大。预填充则正相反：提示长度千变万化、一次要算几百上千个 token，形状几乎每批都不同，硬要为每种形状录图既不划算也录不过来，所以它更愿意走即时（eager）路径，或采用分块/分片的折中（第 22、27 课）。一句话：<strong>解码是"小而规整、重复千万次"，正好踩在图重放的甜点区</strong>；预填充是"大而多变、各算各的"，让灵活的即时路径来扛更合适。</p>

<p>这两种性格背后其实是两种<strong>瓶颈</strong>。预填充一次要处理几百上千个 token，算术密度高，GPU 的计算单元几乎被喂满，是典型的<strong>计算密集（compute-bound）</strong>——此时启动开销相对算量微不足道，套不套图差别不大，真正的优化方向是把这堆矩阵乘做大做满。解码每步每条请求只算 1 个 token，矩阵又瘦又小，GPU 大量时间其实卡在<strong>搬数据</strong>上（读权重、读 KV），是典型的<strong>访存密集（memory-bound）</strong>；这时候每个内核都很短，启动与调度的固定开销反而成了大头，于是用 CUDA 图把上百次内核启动一次性抹平，收益立竿见影。看懂"预填充缺算力、解码缺带宽"，你就懂了为什么同一个 forward 要分两种路径，也懂了第 27 课为何把图重放主要用在解码上。这条"算力 vs 带宽"的分界线，会在后面讲注意力内核、量化、投机解码时反复出现，是贯穿整个推理优化的一把通用尺子。</p>

<p>这也解释了 ModelRunner 为什么还顺手管着：<strong>CUDA 图运行器</strong>（第 27 课）、<strong>注意力后端</strong>（第 33 课）、
<strong>KV 池</strong>（第 30 课），以及投机解码时的<strong>草稿模型</strong>（第 43 课）——它是这些 GPU 侧部件的总装车间。
往后第 25 课讲它手里的模型是<strong>怎么加载</strong>的，第 26 课讲这模型是<strong>怎么写</strong>的。</p>

<p>不妨把 ModelRunner 看作一台 GPU 侧的<strong>总装机</strong>：它把分散的零件拼成一条能跑的流水线。模型给它"算什么"的结构，KV 池给它"历史在哪"的记忆，注意力后端给它"在这块硬件上怎么算注意力"的手艺，CUDA 图运行器给它"解码时怎么跑得更省"的加速档，草稿模型（第 43 课）则在投机解码时给它"先猜几个、再一次性验证"的并行档。这些部件各有各的课，但它们都<strong>挂在 ModelRunner 上、由同一次 forward 串起来</strong>。也正因为零件都集中在这里，调试一次"前向算错/算慢"的问题时，你几乎总是从 ModelRunner 入手：先看 forward_mode 走对了没、再看注意力元数据和 KV 槽位对不对、最后看是不是错误地套用了某张形状不匹配的 CUDA 图。把这台总装机的内部接线图记在脑子里，往后四课就是在逐一拆解它的每一个零件，每拆一个，你对这次 forward 的理解就更立体一分，最终把这条流水线看成一个整体。</p>

<p>把这一课放回整张图：第 1–8 课告诉你"为什么要这样设计"，第 13–17 课讲请求"怎么进来"，第 18–23 课讲调度器"怎么决策"，
而<strong>这一课是决策落地成算力的那一下</strong>。再往后，Part 6 的其余四课就沿着 ModelRunner 手里的几样东西展开：模型权重怎么<strong>加载</strong>上卡（第 25 课）、
一个模型文件<strong>怎么写</strong>（第 26 课，以 Llama 为例）、解码为什么能用 <strong>CUDA 图</strong>把启动开销抹平（第 27 课）、
以及 logits 怎么变成 token 的<strong>采样器</strong>（第 28 课）。带着"<strong>ModelRunner = 决策→计算的边界、每 rank 一个、握着模型+KV池+注意力后端</strong>"
这把钥匙，后面四课你会读得很顺。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>ModelRunner</strong>：每个 TP rank 一个，是<strong>"决策→计算"的边界</strong>；持有模型 + KV 池，<span class="mono">forward</span>→logits，<span class="mono">sample</span>→token。</li>
    <li><strong>ForwardBatch</strong>：ScheduleBatch（第 19 课）的 <strong>GPU 视图</strong>——ids/positions/forward_mode/注意力元数据/KV 指针。</li>
    <li><strong>一次前向</strong>：嵌入 → N 层解码层 → 归一 + lm_head → logits → Sampler。</li>
    <li><strong>两种前向</strong>：EXTEND（预填充，计算密集，多即时）vs DECODE（解码，形状规整，走 CUDA 图重放，第 27 课）。</li>
    <li><strong>总装车间</strong>：ModelRunner 还管 CUDA 图、注意力后端、KV 池、草稿模型——Part 6 的其余几课逐一展开。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Part 5 was all about how the scheduler <strong>decides</strong> — receive, batch, set policy. But it
<strong>only decides; it never computes</strong>. The thing that actually turns a batch of tokens into logits on the
GPU is this lesson's star, <span class="inline">ModelRunner</span>. It is <strong>the door where "decide" becomes
"compute"</strong>: Part 5 was control flow; from here we step into the real math.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of ModelRunner as the <strong>machine operator</strong> on the floor: the scheduler (foreman) hands over a
  <strong>work order</strong> (ForwardBatch: which tokens, prefill or decode, where the KV lives); the operator
  <strong>runs the machine</strong> (<span class="mono">model.forward</span>), reads the <strong>gauge</strong> (logits), and
  passes it to the <strong>inspector</strong> (Sampler) who picks this step's output (the next token). The foreman never
  touches the machine — only the operator does.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  Each TP rank has <strong>one ModelRunner</strong> (owned by the TpModelWorker); it holds the <strong>loaded model</strong>
  (Lessons 25/26) and the <strong>KV pool</strong> (Lesson 30). One <span class="mono">forward(forward_batch)</span> runs a
  batch into logits, and <span class="mono">sample()</span> hands them to the Sampler (Lesson 28) for the token. A
  <span class="inline">ForwardBatch</span> is <strong>the GPU view of a ScheduleBatch (Lesson 19)</strong> — the same
  batch, re-expressed as tensors and metadata the GPU understands.
</div>

<h2>After run_batch: who handed this batch to ModelRunner</h2>
<p>Let's complete the call chain. The scheduler's main loop (Lesson 18) calls <span class="mono">run_batch</span> every step,
but run_batch itself is <strong>thin</strong>: it passes the freshly assembled <span class="mono">ScheduleBatch</span> to the
<strong>TpModelWorker</strong>, which hands it straight to <strong>the one ModelRunner it owns</strong>. The chain is layered this
deep because each layer watches exactly one thing — the scheduler cares about "which batch to run this step", the TpModelWorker
cares about "my rank's boundary and cross-card communication", and the ModelRunner cares about "actually computing the
tensors on this card". Clean, non-overlapping responsibilities are precisely what lets thousands of contributors maintain
this code together.</p>
<p>The chain also hides a subtle but fatal constraint: in an 8-way TP deployment, the 8 TpModelWorkers' ModelRunners must run
the <strong>same batch in strict lockstep</strong>. Each computes a slice of the model (weights split row/column-wise,
Lessons 25/46), and mid-layer they stitch attention outputs and residuals back together via all-reduce / all-gather — if
any rank steps even half a beat early or late, the collective <strong>deadlocks the whole group</strong>. That is why,
before the forward, a ScheduleBatch's contents must be <strong>byte-for-byte identical</strong> across all ranks: same
tokens, same shapes, same forward_mode. Once you see this, Lesson 18's insistence that scheduling be deterministic and
reproducible is no fussiness — it is the hard prerequisite for a distributed forward that doesn't hang.</p>

<h2>From ScheduleBatch to ForwardBatch: a GPU pair of glasses</h2>
<p>The scheduler hands over a <span class="mono">ScheduleBatch</span> (a set of Reqs, Lesson 19). ModelRunner first
translates it into a <span class="mono">ForwardBatch</span> — exactly what a GPU forward needs: <strong>input_ids</strong>
(the tokens to feed this step), <strong>positions</strong> (each token's position, for RoPE), <strong>forward_mode</strong>
(EXTEND/prefill vs DECODE), <strong>attention metadata</strong> (which backend, which KV slots to read/write, Lesson 33),
and pointers into the <strong>KV pool</strong>. In one line: ScheduleBatch records "<strong>who needs computing</strong>";
ForwardBatch records "<strong>how the GPU computes this step</strong>".</p>

<div class="flow">
  <div class="node"><div class="nt">ScheduleBatch</div><div class="nd">a batch of Reqs (L19)</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">ForwardBatch</div><div class="nd">GPU view: ids/positions/mode/KV</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">model.forward</div><div class="nd">embed→layers→norm→lm_head</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">logits</div><div class="nd">next-token dist per request</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Sampler</div><div class="nd">pick the next token (L28)</div></div>
</div>

<table class="t">
  <tr><th>ForwardBatch field</th><th>what it is</th><th>used by</th></tr>
  <tr><td class="mono">input_ids</td><td>token ids to forward this step</td><td>the embedding</td></tr>
  <tr><td class="mono">positions</td><td>each token's absolute position</td><td>RoPE / pos encoding (L36)</td></tr>
  <tr><td class="mono">forward_mode</td><td>EXTEND (prefill) / DECODE</td><td>picks the forward path</td></tr>
  <tr><td class="mono">attn metadata</td><td>attention backend + KV slots</td><td>attention kernel (L33)</td></tr>
  <tr><td class="mono">out_cache_loc</td><td>where new K/V are written</td><td>KV cache pool (L30)</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="one forward: ModelRunner takes a ForwardBatch (input_ids, positions, seq_lens, out_cache_loc), runs left-to-right through the model's layer stack, and produces next-token logits">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">one forward: inputs → model → logits</text>
    <rect x="24" y="58" width="210" height="150" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="129" y="84" text-anchor="middle" style="font-weight:700;fill:var(--blue)">ForwardBatch</text>
    <rect x="40" y="98" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="113" class="mono" style="font-size:11px">input_ids</text>
    <rect x="40" y="124" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="139" class="mono" style="font-size:11px">positions</text>
    <rect x="40" y="150" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="165" class="mono" style="font-size:11px">seq_lens</text>
    <rect x="40" y="176" width="178" height="22" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="48" y="191" class="mono" style="font-size:11px">out_cache_loc</text>
    <line x1="234" y1="133" x2="300" y2="133" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="300,128 312,133 300,138" style="fill:var(--line)"/>
    <rect x="312" y="58" width="230" height="150" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="427" y="84" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">model (layer stack)</text>
    <rect x="330" y="98" width="194" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="427" y="115" text-anchor="middle" style="font-size:12px">embed</text>
    <rect x="330" y="128" width="194" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="427" y="145" text-anchor="middle" style="font-size:12px">N decoder layers</text>
    <rect x="330" y="158" width="194" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1"/>
    <text x="427" y="175" text-anchor="middle" style="font-size:12px">norm + lm_head</text>
    <line x1="552" y1="133" x2="612" y2="133" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="612,128 624,133 612,138" style="fill:var(--line)"/>
    <rect x="632" y="92" width="144" height="82" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="704" y="128" text-anchor="middle" style="font-weight:700;fill:var(--teal)">logits</text>
    <text x="704" y="150" text-anchor="middle" style="font-size:11px;fill:var(--muted)">next-token dist</text>
  </svg>
  <div class="figcap"><b>Fig 1 · one forward</b> — ModelRunner takes a ForwardBatch (input_ids / positions / seq_lens / out_cache_loc), runs left-to-right through the model's layer stack (embed → N decoder layers → norm + lm_head), and finally produces next-token logits.</div>
</div>

<p>These fields aren't invented out of thin air; they map directly onto the <span class="mono">ForwardBatch</span> dataclass in the source — every input of a single forward pass, packed into one struct:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/forward_batch_info.py ::ForwardBatch</span><span class="ln">one struct holding every input of a single forward pass</span></div>
  <pre><span class="kw">class</span> ForwardBatch:
    <span class="cm"># every input of ONE forward pass, packed together.</span>
    forward_mode: ForwardMode      <span class="cm"># EXTEND (prefill) or DECODE</span>
    batch_size: int
    input_ids: torch.Tensor        <span class="cm"># tokens to run this step</span>
    req_pool_indices: torch.Tensor <span class="cm"># which requests (rows in req_to_token)</span>
    seq_lens: torch.Tensor         <span class="cm"># per-request context length</span>
    out_cache_loc: torch.Tensor    <span class="cm"># where to write the new KV</span>
    ...</pre>
</div>

<p>Read that table one level deeper and it gets tastier. <strong>positions</strong> is no decoration: it feeds straight into
RoPE and sets each token's rotation phase, so the moment prefill or decode gets an absolute position wrong, attention
"misjudges" near-vs-far relationships and the output goes haywire — that's exactly why chunked prefill (Lesson 22) must
carefully preserve positional continuity. <strong>forward_mode</strong> is the master switch; one flip decides the whole
path: variable-length prefill kernel vs regular decode kernel, whether a CUDA graph applies, which mask the attention
backend uses. The <strong>attention metadata</strong> translates "the logical k-th token" into "which physical page and slot
in the KV pool", and together with <span class="mono">out_cache_loc</span> it tells the kernel exactly where to write the
new K/V and where to read the historical K/V. In short, these ForwardBatch fields together form a <strong>GPU addressing
table</strong> that every read and write in the forward follows; how well they are organized directly decides whether the
downstream attention kernel (Lesson 33) and CUDA graph (Lesson 27) run fast and stable.</p>

<h2>ModelRunner.forward: how one forward goes</h2>
<p>It forks two ways: if a <strong>CUDA graph</strong> (Lesson 27) applies, <strong>replay</strong> a recorded graph;
otherwise run the batch <strong>eagerly</strong>. The key is <span class="mono">can_run_graph</span> — usually only
<strong>decode steps</strong> (regular shapes, small batch) take graph replay, while <strong>prefill</strong> (variable
lengths) mostly runs eager (or a piecewise prefill graph). Here is the real skeleton of <span class="mono">_forward_raw</span>:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/model_runner.py ::ModelRunner.forward</span><span class="ln">the forward dispatch</span></div>
  <pre><span class="kw">def</span> _forward_raw(self, forward_batch, pp_proxy_tensors, ...):
    <span class="cm"># Can we replay the decode CUDA graph: mode matches + graph exists + shape fits</span>
    can_run_graph = (forward_batch.forward_mode.is_cuda_graph()
                     <span class="kw">and</span> self.decode_cuda_graph_runner
                     <span class="kw">and</span> self.decode_cuda_graph_runner.can_run_graph(forward_batch))
    <span class="kw">if</span> can_run_graph:                          <span class="cm"># decode: replay the recorded graph (L27)</span>
        ret = self.decode_cuda_graph_runner.execute(forward_batch, ...)
        <span class="kw">return</span> ModelRunnerOutput(logits_output=ret, can_run_graph=True)
    self._prepare_eager_forward_batch(forward_batch)  <span class="cm"># eager path: DP/padding normalize</span>
    <span class="kw">if</span> forward_batch.forward_mode.is_extend(...):   <span class="cm"># prefill: run the batch</span>
        ...                                       <span class="cm"># else eager decode fallback</span></pre>
</div>

<p>Either path ends in <span class="mono">model.forward(input_ids, positions, forward_batch)</span>: embed → a stack of
<strong>decoder layers</strong> (attention + MLP, Lesson 26) → final norm → the <span class="mono">lm_head</span> projecting
out <strong>logits</strong>. ModelRunner then calls <span class="mono">sample()</span> to hand logits to the Sampler (Lesson 28).</p>

<p>Look closer at why the dispatch is written this way. The first test, <span class="mono">can_run_graph</span>, is an
<strong>and</strong> of three conditions: the mode must be graph-able (decode-like), the graph must <strong>actually have
been recorded</strong>, and the current batch shape must fit one of the recorded buckets. Miss any one and replay is off —
which is why graph replay sits up front and short-circuits: it is the "fast path", taken when possible, otherwise we fall
back. After falling back, <span class="mono">_prepare_eager_forward_batch</span> does the eager path's <strong>normalization
cleanup</strong>: padding for data-parallel (DP) across ranks, aligning tensor shapes so the following kernels run cleanly.
Then it forks on <span class="mono">is_extend</span> — prefill honestly runs the whole batch, otherwise it lands in the
eager decode fallback. Read this and you can pinpoint a "why didn't my decode use the CUDA graph" question to one of those
three switches, instead of guessing in the dark.</p>

<p>Worth a closer look at <strong>what the attention layer does here</strong>: it takes the KV slots from ForwardBatch,
<strong>writes</strong> the current token's K/V <strong>into the KV pool</strong>, then attends the current Q against the
<strong>historical K/V</strong> (possibly a prefix reused via RadixAttention, Lesson 7). In other words, the caching and
reuse from Lessons 4–7 <strong>actually happen</strong> at this very moment when ModelRunner runs the forward — every
intuition built in those earlier lessons gets "powered on" here for the first time. And precisely because the forward
constantly reads/writes the KV pool and calls hardware-specific attention kernels, ModelRunner must hold model, KV pool,
and attention backend <strong>all at once</strong> — none optional.</p>

<p>Walk the forward one notch finer. After the embedding looks input_ids up into token vectors, the data enters a stack of
identically structured <strong>decoder layers</strong>. In each layer comes a norm, then the attention sub-layer: Q/K/V are
projected by linear layers, K/V are written into the KV pool at <span class="mono">out_cache_loc</span>, and the attention
kernel lets the current Q look at itself plus all historical K/V. If that prefix was already computed by another request
and hung on the radix tree by RadixAttention (Lesson 7), those historical K/V need no recompute — they are
<strong>reused directly</strong>; the cache reuse of Lessons 4–7 only lands on a concrete kernel right here. The attention
output is residual-added, then through the MLP sub-layer (another norm + two linears + activation), another residual.
Stack N such layers, do a final norm, and <span class="mono">lm_head</span> projects the last hidden vector to the full
vocab size to get <strong>logits</strong>. One commonly missed optimization: decode only needs the logits at <strong>each
request's last position</strong> (the next word only looks at the last step), so the forward usually slices "take the last
token" before projecting, saving a lot of useless lm_head compute. The moment logits exist, ModelRunner immediately calls
<span class="mono">sample()</span>, where the next lesson's Sampler enters — <strong>the logits→token turning point of the
whole pipeline happens right here</strong>.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Embed</h4><p>input_ids → token vectors; positions ready for RoPE.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>N decoder layers</h4><p>each: attention (read/write KV pool) + MLP + norms (Lesson 26).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Final norm + lm_head</h4><p>last hidden → vocab-sized <strong>logits</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Sample</h4><p><span class="mono">sample()</span> → Sampler emits the next token (Lesson 28).</p></div></div>
</div>

<h2>EXTEND vs DECODE: one forward, two personalities</h2>
<p>Remember the two phases from Lesson 4? They become two forwards here in ModelRunner:</p>

<div class="cols">
  <div class="col"><h4>EXTEND / prefill forward</h4><p>Processes many tokens of the <strong>whole prompt</strong> at once (variable
  length), <strong>compute-bound</strong>, highly parallel; shapes vary, so it mostly runs <strong>eager</strong> or
  piecewise/chunked (Lesson 22). It <strong>fills</strong> this batch's K/V into the cache.</p></div>
  <div class="col"><h4>DECODE / decode forward</h4><p>Each request computes <strong>just 1 new token</strong> — regular shapes,
  fixed batch — perfect for <strong>CUDA graph replay</strong> (Lesson 27), erasing the launch overhead of hundreds of tiny
  kernels. It <strong>appends one row</strong> to the cache.</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="ForwardBatch unifies several prefill and decode requests into one batch: one 200-token prefill plus three 1-token decodes are packed into a length-203 input_ids consumed by a single forward">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">ForwardBatch: prefill + decode in one batch</text>
    <rect x="24" y="52" width="220" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="34" y="70" style="font-size:12px;fill:var(--amber);font-weight:700">req1 · EXTEND / prefill</text>
    <text x="34" y="86" class="mono" style="font-size:11px;fill:var(--amber)">200 tokens</text>
    <rect x="24" y="100" width="220" height="32" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="34" y="120" style="font-size:12px;fill:var(--purple)">req2 · DECODE · 1 token</text>
    <rect x="24" y="140" width="220" height="32" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="34" y="160" style="font-size:12px;fill:var(--purple)">req3 · DECODE · 1 token</text>
    <rect x="24" y="180" width="220" height="32" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="34" y="200" style="font-size:12px;fill:var(--purple)">req4 · DECODE · 1 token</text>
    <line x1="254" y1="132" x2="326" y2="132" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="326,127 338,132 326,137" style="fill:var(--line)"/>
    <rect x="346" y="70" width="300" height="120" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="496" y="96" text-anchor="middle" style="font-weight:700;fill:var(--blue)">one ForwardBatch · input_ids</text>
    <rect x="360" y="118" width="170" height="30" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <text x="445" y="138" text-anchor="middle" class="mono" style="font-size:10px;fill:var(--amber)">200 (req1 prefill)</text>
    <rect x="540" y="118" width="28" height="30" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <rect x="572" y="118" width="28" height="30" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <rect x="604" y="118" width="28" height="30" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <text x="496" y="174" text-anchor="middle" class="mono" style="font-size:11px">length = 200 + 1 + 1 + 1 = 203</text>
    <line x1="496" y1="190" x2="496" y2="222" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="491,222 496,234 501,222" style="fill:var(--line)"/>
    <rect x="396" y="236" width="200" height="50" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="496" y="266" text-anchor="middle" style="font-weight:700;fill:var(--teal)">one forward consumes it all</text>
  </svg>
  <div class="figcap"><b>Fig 2 · one batch holds both kinds</b> — one EXTEND/prefill (200 tokens) and three DECODE/decode (1 token each) are packed into a single ForwardBatch; <span class="mono">input_ids</span> length = 200+1+1+1 = 203, consumed by one forward pass.</div>
</div>

<p>A concrete example: pack <strong>3 decode requests</strong> (each computing just 1 new token) and <strong>1 prefill of 200 tokens</strong> into <strong>one ForwardBatch</strong>, and <span class="mono">input_ids</span> has length <span class="mono">200 + 1 + 1 + 1 = 203</span>. The prefill segment's <span class="mono">forward_mode</span> is <span class="mono">EXTEND</span> (200 tokens fed at once), while the three decodes are <span class="mono">DECODE</span> (one appended token each); they share a single forward pass and one set of attention metadata, computed by the GPU in one shot — exactly how continuous batching (Lesson 5) runs prefill and decode mixed together.</p>

<p>Why is it <strong>decode</strong> that suits CUDA graphs while prefill doesn't? It comes down to "is the shape stable".
A CUDA graph records a fixed sequence of kernel calls at fixed memory addresses; once recorded, replay can only apply the
<strong>same tensor shapes and same pointers</strong>. Decode produces exactly 1 new token per request per step, the batch
size stays fixed for a while, and each sequence grows by just 1 — highly regular shapes — so varying real lengths can be
<strong>padded into a few fixed buckets</strong>, each recorded once and replayed many times. And decode is precisely the
phase where launch overhead dominates (little compute per step, lots of kernel dispatch), so erasing it pays the most.
Prefill is the opposite: prompt lengths vary wildly and hundreds-to-thousands of tokens are computed at once, so shapes
differ almost every batch; recording a graph per shape is neither worthwhile nor feasible, so prefill prefers the eager
path or a chunked/piecewise compromise (Lessons 22, 27). In one line: <strong>decode is "small, regular, repeated
millions of times" — right in the sweet spot of graph replay</strong>; prefill is "big, varied, each its own", better
carried by the flexible eager path.</p>

<p>These two personalities really reflect two <strong>bottlenecks</strong>. Prefill processes hundreds-to-thousands of
tokens at once with high arithmetic intensity, nearly saturating the GPU's compute units — classically
<strong>compute-bound</strong>; here launch overhead is negligible next to the math, graph or not makes little difference,
and the real optimization is making those matmuls big and full. Decode computes just 1 token per request per step, with
thin, small matrices, so the GPU spends much of its time <strong>moving data</strong> (reading weights, reading KV) —
classically <strong>memory-bound</strong>; now each kernel is tiny, and the fixed cost of launch/dispatch becomes the bulk,
so flattening hundreds of kernel launches with a CUDA graph pays off immediately. Once you see "prefill lacks compute,
decode lacks bandwidth", you understand why one forward splits into two paths, and why Lesson 27 mostly applies graph
replay to decode. This "compute vs bandwidth" dividing line reappears again and again when we get to attention kernels,
quantization, and speculative decoding — a universal yardstick running through the whole of inference optimization.</p>

<p>This also explains why ModelRunner also owns: the <strong>CUDA-graph runner</strong> (Lesson 27), the <strong>attention
backend</strong> (Lesson 33), the <strong>KV pool</strong> (Lesson 30), and a <strong>draft model</strong> for speculative
decoding (Lesson 43) — it is the assembly shop for these GPU-side parts. Next, Lesson 25 shows how its model gets
<strong>loaded</strong>, and Lesson 26 shows how that model is <strong>written</strong>.</p>

<p>Think of ModelRunner as a GPU-side <strong>assembly machine</strong>: it bolts scattered parts into one running pipeline.
The model gives it the "what to compute" structure, the KV pool gives it the "where history lives" memory, the attention
backend gives it the "how to do attention on this hardware" craft, the CUDA-graph runner gives it the "how to run decode
cheaper" overdrive, and the draft model (Lesson 43) gives it the "guess a few, verify in one shot" parallel gear for
speculative decoding. Each part has its own lesson, but they all <strong>hang on ModelRunner and are strung together by
the same forward</strong>. Because the parts converge here, debugging a "forward computed wrong / ran slow" issue almost
always starts at ModelRunner: first check forward_mode went the right way, then check the attention metadata and KV slots,
finally check whether a shape-mismatched CUDA graph was wrongly applied. Keep this assembly machine's internal wiring in
your head, and the next four lessons are just taking apart its parts one by one — each one you remove makes your picture of
this forward more three-dimensional, until you see the whole pipeline as one.</p>

<p>Place this lesson back in the whole map: Lessons 1–8 tell you "why it's designed this way", Lessons 13–17 cover how
requests "get in", Lessons 18–23 cover how the scheduler "decides" — and <strong>this lesson is the instant where a
decision lands as compute</strong>. From here, the rest of Part 6 unfolds along the few things in ModelRunner's hands: how
model weights are <strong>loaded</strong> onto the card (Lesson 25), how a model file is <strong>written</strong> (Lesson 26,
with Llama as the example), why decode can use a <strong>CUDA graph</strong> to flatten launch overhead (Lesson 27), and the
<strong>Sampler</strong> turning logits into a token (Lesson 28). Carry the key — "<strong>ModelRunner = the decide→compute
boundary, one per rank, holding model + KV pool + attention backend</strong>" — and the next four lessons read smoothly.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>ModelRunner</strong>: one per TP rank, the <strong>decide→compute boundary</strong>; holds model + KV pool, <span class="mono">forward</span>→logits, <span class="mono">sample</span>→token.</li>
    <li><strong>ForwardBatch</strong>: the <strong>GPU view</strong> of a ScheduleBatch (L19) — ids/positions/forward_mode/attn metadata/KV pointers.</li>
    <li><strong>One forward</strong>: embed → N decoder layers → norm + lm_head → logits → Sampler.</li>
    <li><strong>Two forwards</strong>: EXTEND (prefill, compute-bound, mostly eager) vs DECODE (regular shapes, CUDA-graph replay, L27).</li>
    <li><strong>Assembly shop</strong>: ModelRunner also owns the CUDA graph, attention backend, KV pool, draft model — the rest of Part 6 unpacks them.</li>
  </ul>
</div>
""",
}

LESSON_25 = {
    "zh": r"""
<p class="lead">
上一课的 <span class="inline">ModelRunner</span> 张口就要"<strong>加载好的模型</strong>"——可这模型到底是<strong>怎么从硬盘搬上 GPU</strong> 的？
一个 70B 的权重有上百 GB、散在几十个文件里，名字还是 HuggingFace 的命名，和 SGLang 内部的模块对不上。这一课就讲那个把"<strong>一堆磁盘文件</strong>"
变成"<strong>每张卡上各就各位的张量</strong>"的幕后角色：<span class="inline">DefaultModelLoader</span>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把加载模型想成<strong>用一堆编号纸箱布置新家</strong>：先读<strong>装箱清单</strong>（HuggingFace 的权重名，如 <span class="mono">model.layers.0.self_attn.q_proj.weight</span>），
  把每个零件认领到<strong>对的房间</strong>（SGLang 里对应的模块参数），需要时把<strong>层板裁到合适尺寸</strong>（按张量并行切成本 rank 的那一片），
  再按<strong>统一的漆面</strong>装好（转成 bf16/fp16，或装成量化格式）。关键是：<strong>不要一次把所有箱子全倒进客厅</strong>——而是<strong>一箱一箱地流式搬运</strong>，
  否则主机内存当场爆掉。Loader 干的就是这套"照单认领、按需裁切、流式装配"的活。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  <span class="mono">DefaultModelLoader.load_model</span> 是绝大多数模型的<strong>共同入口</strong>：它先按 <span class="mono">model_config</span> 搭出一个<strong>空壳模型</strong>
  （结构有了、权重还是占位），再打开磁盘上的 <strong>safetensors 分片</strong>，把张量<strong>一个一个地流式喂进去</strong>。喂的过程由<strong>每个模型自己的</strong>
  <span class="mono">load_weights(weights)</span> 接手：它知道 HF 的名字该<strong>映射</strong>到哪个内部参数、哪些权重要<strong>融合</strong>、本 rank 该拿<strong>哪一片</strong>。
  正因为加载逻辑被这样统一收口，SGLang 才能用一套框架托起<strong>上百种模型 + 各类量化格式</strong>。
</div>

<h2>load_model：先搭空壳，再流式灌权重</h2>
<p>整条加载路径其实只有两步，但每一步都暗藏讲究。<strong>第一步</strong>，在目标设备（GPU）和目标 dtype 的上下文里，用 <span class="mono">_initialize_model</span>
按配置<strong>实例化模型结构</strong>——这时各层的线性层、注意力、嵌入都建好了，但参数还是<strong>未初始化的占位张量</strong>（已在 GPU 上按全尺寸开好，只是尚未填入真值）。<strong>第二步</strong>，
调 <span class="mono">load_weights_and_postprocess</span>，它从 <span class="mono">_get_all_weights</span> 拿到一个<strong>权重迭代器</strong>，再交给
<span class="mono">model.load_weights(weights)</span> 把真值<strong>逐张量灌进去</strong>。注意这里的精妙：权重不是"先全读进内存再分发"，而是一个 <span class="mono">(name, tensor)</span>
的<strong>生成器</strong>——读一个、灌一个、丢一个，<strong>主机内存占用被牢牢卡在一个很小的上界</strong>，哪怕模型有几百 GB 也不会把内存撑爆。这也意味着加载是<strong>可被流水线化</strong>的：当一个张量正在被映射、切片、搬卡时，下一个张量可以同时从磁盘预读，磁盘 I/O 与 CPU/GPU 处理彼此重叠，整体加载时间被进一步压短。</p>

<p>为什么非要<strong>流式（streaming）</strong>不可？设想一台 8 卡机要加载 70B：如果先把全部权重读进主机内存再切分下发，单是这份临时副本就可能逼近甚至超过物理内存，
更别提 safetensors 的零拷贝优势会被白白浪费。流式加载把"读—映射—切片—搬卡"做成一条<strong>边读边丢的流水线</strong>：任意时刻内存里只驻留<strong>当前这一个张量</strong>
（及其正在处理的目标分片），上界与模型总大小<strong>解耦</strong>。这就是为什么同一台机器，既能加载几 GB 的小模型，也能加载几百 GB 的大模型——内存压力几乎是<strong>常数</strong>。
理解了这一点，你也就懂了为什么 SGLang 默认用 safetensors、为什么"加载慢"往往是磁盘/网络带宽而非内存瓶颈。</p>

<p>再补一个常被忽略的细节：流式之所以能成立，<strong>safetensors 这种格式功不可没</strong>。它把每个张量的名字、形状、dtype、字节偏移都写在文件头的索引里，于是 Loader 可以<strong>不解析整份文件</strong>就精确定位"我现在要的这个张量在第几个分片、从哪个字节开始"，再用内存映射（mmap）<strong>零拷贝</strong>地把它取出来。对比早年的 <span class="mono">.bin</span>（pickle）格式——必须把整个对象图反序列化进内存才能取用——safetensors 的"<strong>按需、零拷贝、可流式</strong>"正是大模型时代的刚需。这也是为什么本课的索引步骤要先打开 <span class="mono">*.safetensors.index.json</span>：它就是那张"哪个权重在哪个分片"的总目录，Loader 照着它一片片地按需读取。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>定位 checkpoint</h4><p>按 <span class="mono">model_path</span> 找到本地或下载好的目录（HF / ModelScope）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>safetensors 分片</h4><p>按索引打开多个 <span class="mono">*.safetensors</span>，零拷贝、可流式逐张量读取。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>名字映射</h4><p>HF 权重名 →（模型自己的 <span class="mono">load_weights</span>）→ SGLang 内部参数，常顺手<strong>融合</strong> q/k/v、gate/up。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>按 TP 切片</h4><p>每个权重按列/行并行切开，本 rank <strong>只接自己那一片</strong>（第 24/46 课）。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>dtype / 量化</h4><p>转成 bf16/fp16，或装成 FP8/INT4/AWQ/GPTQ 并带上 scales（第 35 课）。</p></div></div>
  <div class="step"><div class="num">6</div><div class="sc"><h4>落到 GPU</h4><p>写进空壳模型对应参数的显存槽位，加载完即 <span class="mono">model.eval()</span>。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 290" role="img" aria-label="加载流程：磁盘上的 safetensors 分片经 Loader 流式读取，转 dtype 或量化并带上 scales，最后把每个 rank 的切片散落到各自 GPU 显存">
    <text x="20" y="28" style="font-weight:700;fill:var(--muted)">加载流程：磁盘分片 → dtype / 量化 → 各 rank 显存</text>
    <text x="20" y="56" style="fill:var(--faint);font-size:12px">磁盘 safetensors 分片</text>
    <rect x="20" y="66" width="196" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="118" y="89" text-anchor="middle" class="mono" style="font-size:11px">model-00001-of-00002</text>
    <rect x="30" y="110" width="196" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="128" y="133" text-anchor="middle" class="mono" style="font-size:11px">model-00002-of-00002</text>
    <line x1="232" y1="112" x2="270" y2="112" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="270,107 280,112 270,117" style="fill:var(--faint)"/>
    <rect x="284" y="80" width="150" height="66" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="359" y="107" text-anchor="middle" style="font-weight:700;fill:var(--blue);font-size:13px">Loader 流式读取</text>
    <text x="359" y="127" text-anchor="middle" class="mono" style="font-size:11px">(name, tensor)…</text>
    <line x1="440" y1="112" x2="478" y2="112" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="478,107 488,112 478,117" style="fill:var(--faint)"/>
    <rect x="492" y="80" width="150" height="66" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="567" y="107" text-anchor="middle" style="font-weight:700;fill:var(--amber);font-size:13px">转 dtype / 量化</text>
    <text x="567" y="127" text-anchor="middle" class="mono" style="font-size:11px">bf16 · FP8 + scales</text>
    <line x1="644" y1="112" x2="664" y2="42" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="644" y1="112" x2="664" y2="100" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="644" y1="112" x2="664" y2="158" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="644" y1="112" x2="664" y2="216" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <rect x="664" y="20" width="120" height="44" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="724" y="40" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--teal)">GPU0 显存</text>
    <text x="724" y="56" text-anchor="middle" class="mono" style="font-size:10px">rank0 切片</text>
    <rect x="664" y="78" width="120" height="44" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="724" y="98" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--blue)">GPU1 显存</text>
    <text x="724" y="114" text-anchor="middle" class="mono" style="font-size:10px">rank1 切片</text>
    <rect x="664" y="136" width="120" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="724" y="156" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--amber)">GPU2 显存</text>
    <text x="724" y="172" text-anchor="middle" class="mono" style="font-size:10px">rank2 切片</text>
    <rect x="664" y="194" width="120" height="44" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="724" y="214" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--purple)">GPU3 显存</text>
    <text x="724" y="230" text-anchor="middle" class="mono" style="font-size:10px">rank3 切片</text>
  </svg>
  <div class="figcap"><b>图 1 · 加载流程</b> — 磁盘上的 safetensors 分片经 Loader <strong>流式</strong>读出，逐个张量转 dtype 或按量化布局摆好并带上 scales，最后只把<strong>每个 rank 的切片</strong>散落进各自 GPU 显存。</div>
</div>

<h2>名字映射与融合：HF 的名字怎么对上 SGLang 的参数</h2>
<p>磁盘上的权重用的是 HuggingFace 的命名习惯，比如注意力的查询投影叫 <span class="mono">model.layers.0.self_attn.q_proj.weight</span>，
键、值各自又是 <span class="mono">k_proj</span>、<span class="mono">v_proj</span>。但 SGLang 为了<strong>性能</strong>，往往把 q/k/v <strong>三个投影融合成一个打包权重</strong>
（一次大矩阵乘代替三次小矩阵乘），把 MLP 的 <span class="mono">gate_proj</span> 和 <span class="mono">up_proj</span> 也<strong>融成一个</strong>。这就带来一个"对不上"的问题：
磁盘里是<strong>分开的三份</strong>，内存里要的是<strong>合并的一份</strong>。解决它的，正是每个模型类自己实现的 <span class="mono">load_weights(weights)</span>——
它内置一张"<strong>映射表</strong>"，知道遇到 <span class="mono">q_proj/k_proj/v_proj</span> 时分别该塞进打包权重的<strong>哪一段偏移</strong>，遇到 <span class="mono">gate/up</span> 时又该怎么拼。</p>

<p>把"映射"这件事交给<strong>模型自己</strong>而不是 Loader，是一处关键的职责划分。Loader 只负责<strong>把张量流式读出来</strong>（它不关心也不该关心某个模型内部叫什么），
而"这个名字对应我的哪个参数、要不要融合、要不要转置"全是<strong>模型私有的知识</strong>，自然封装在模型的 <span class="mono">load_weights</span> 里。于是新增一个模型时，
你<strong>不必碰 Loader</strong>，只要在模型文件里写好它的 <span class="mono">load_weights</span>（第 26 课）即可——这就是为什么 SGLang 能相对轻松地支持上百种模型架构。
映射与融合也不是可有可无的细节：融合直接决定了前向时矩阵乘的形状，融合得对，后续的张量并行切片和量化才能顺理成章地接上。</p>

<p>这里值得多想一层：为什么"<strong>融合</strong>"对性能这么关键？GPU 跑一次矩阵乘有固定的<strong>启动与调度开销</strong>，把 q、k、v 三个小投影合成一个大投影，意味着一次大矩阵乘代替三次小矩阵乘——既摊薄了启动开销，又让 GPU 的计算单元被喂得更满，访存也更连续。gate/up 融成一个 <span class="mono">gate_up_proj</span> 同理。代价是加载时要做"<strong>反向拼装</strong>"：磁盘上是三份独立张量，必须按正确的偏移、正确的顺序拼进同一个打包参数，任何一段对错位，前向算出来的就是乱码。所以 <span class="mono">load_weights</span> 里那张映射表看似琐碎，实则是<strong>正确性与性能的双重保险</strong>——它把"磁盘的存储布局"和"内存的计算布局"这两套不同的世界精确地缝合在一起。</p>

<div class="cols">
  <div class="col"><h4>磁盘上的 HF 权重名</h4><p>分开存：<span class="mono">...self_attn.q_proj.weight</span>、<span class="mono">k_proj.weight</span>、<span class="mono">v_proj.weight</span>；
  MLP 是 <span class="mono">gate_proj</span> 与 <span class="mono">up_proj</span> 两份。命名跟着 HuggingFace 走，<strong>逐个投影各一份</strong>。</p></div>
  <div class="col"><h4>SGLang 内部参数</h4><p>融合存：q/k/v 打包成<strong>一个</strong> <span class="mono">qkv_proj</span>，gate/up 打包成<strong>一个</strong> <span class="mono">gate_up_proj</span>。
  <span class="mono">load_weights</span> 按偏移把分开的 HF 权重<strong>填进打包参数的各段</strong>，一次大矩阵乘替代多次小乘。</p></div>
</div>

<h2>为 TP 切片，按 dtype/量化落地</h2>
<p>映射之后还有两道工序。<strong>其一是张量并行（TP）切片</strong>：在多卡部署里，每个权重并不会整份地存在每张卡上，而是<strong>按 rank 切成片</strong>。
注意力的 q/k/v、MLP 的 gate/up 走<strong>列并行</strong>（按输出维切，每卡算一段输出），注意力的 <span class="mono">o_proj</span>、MLP 的 <span class="mono">down_proj</span> 走<strong>行并行</strong>
（按输入维切，每卡算部分和再 all-reduce）。Loader/模型在加载时就<strong>只把本 rank 的那一片搬上这张卡</strong>，所以 8 卡各自的显存里都<strong>只有 1/8 的权重</strong>
（第 24/46 课）。这也解释了为什么加载本身就是分布式的：每张卡读的是同一批文件，但<strong>各取所需、各留一片</strong>。</p>

<p><strong>其二是 dtype 与量化</strong>。最常见的是把权重<strong>转成 bf16 或 fp16</strong> 再上卡；但如果 checkpoint 本身是<strong>量化格式</strong>——FP8、INT4、AWQ、GPTQ（第 35 课）——
加载时就要连同它的<strong>缩放因子（scales）/零点</strong>一起读进来，按量化布局摆好。之后前向要么用<strong>专门的量化内核</strong>直接在低比特上算，要么在用到时<strong>惰性反量化</strong>回高精度。
量化的意义在于<strong>用更小的显存装下更大的模型</strong>，代价是要正确处理 scales 与打包格式——而这些细节同样被收进各量化方法的加载逻辑里，Loader 只需把字节流交给它们。
正因为"切片"和"量化"都在加载这一步被妥善安排，上层的 ModelRunner（第 24 课）才能拿到一个<strong>开箱即用、各就各位</strong>的模型。</p>

<p>把列并行和行并行放在一起再看一眼，会更清楚加载为什么要"按方向"切。<strong>列并行</strong>切的是权重的<strong>输出维</strong>：q/k/v、gate/up 这些"<strong>向外扩张</strong>"的投影，把输出通道平均分给各 rank，每张卡只算自己负责的那几列输出，彼此<strong>互不依赖</strong>，算完直接拼接即可。<strong>行并行</strong>切的是<strong>输入维</strong>：<span class="mono">o_proj</span>、<span class="mono">down_proj</span> 这些"<strong>向内收拢</strong>"的投影，每张卡只拿到输入的一部分、算出一个<strong>部分和</strong>，必须再用 all-reduce 把各卡的部分和相加才得到正确结果。这一"列切—算—行切—规约"的配对不是随意的：它让每层中间正好只需要一次集合通信，把跨卡开销压到最低。而加载阶段的职责，就是<strong>提前把每个权重按正确的方向切好、只把本 rank 的那一片落到本卡</strong>，前向时才不必再临时搬运。第 46 课会把这套并行切分讲透，这里你只需记住：<strong>权重怎么切，是在加载时就定下的</strong>。</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="权重矩阵 A 沿第二维（列）切成 A_1 A_2 A_3 A_4 四片，TP 的 GPU0 到 GPU3 各只持有一列块，rank i 计算 Y_i 等于 X 乘 A_i">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">权重 A 按列切到各 TP rank：A = [A_1 | A_2 | A_3 | A_4]</text>
    <rect x="20" y="58" width="56" height="110" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="48" y="118" text-anchor="middle" class="mono">X</text>
    <text x="92" y="120" text-anchor="middle" style="fill:var(--faint);font-size:18px">×</text>
    <rect x="112" y="58" width="84" height="110" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="154" y="118" text-anchor="middle" class="mono">A_1</text>
    <rect x="200" y="58" width="84" height="110" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="242" y="118" text-anchor="middle" class="mono">A_2</text>
    <rect x="288" y="58" width="84" height="110" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="330" y="118" text-anchor="middle" class="mono">A_3</text>
    <rect x="376" y="58" width="84" height="110" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="418" y="118" text-anchor="middle" class="mono">A_4</text>
    <text x="240" y="192" text-anchor="middle" style="fill:var(--faint);font-size:12px">一个权重矩阵沿第二维（列）被切成 4 块</text>
    <line x1="154" y1="168" x2="105" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="242" y1="168" x2="295" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="330" y1="168" x2="485" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="418" y1="168" x2="675" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <rect x="20" y="214" width="170" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="105" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--blue)">GPU 0 · rank 0</text>
    <text x="105" y="262" text-anchor="middle" class="mono" style="font-size:11px">存 A_1 → Y_0 = X·A_1</text>
    <rect x="210" y="214" width="170" height="64" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="295" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--teal)">GPU 1 · rank 1</text>
    <text x="295" y="262" text-anchor="middle" class="mono" style="font-size:11px">存 A_2 → Y_1 = X·A_2</text>
    <rect x="400" y="214" width="170" height="64" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="485" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--amber)">GPU 2 · rank 2</text>
    <text x="485" y="262" text-anchor="middle" class="mono" style="font-size:11px">存 A_3 → Y_2 = X·A_3</text>
    <rect x="590" y="214" width="170" height="64" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="675" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--purple)">GPU 3 · rank 3</text>
    <text x="675" y="262" text-anchor="middle" class="mono" style="font-size:11px">存 A_4 → Y_3 = X·A_4</text>
  </svg>
  <div class="figcap"><b>图 2 · 权重按 TP rank 列切</b> — 一个权重矩阵 A 沿第二维被切成 <span class="mono">A = [A_1 | A_2 | A_3 | A_4]</span>，每个 TP rank（GPU 0…3）只持有自己那一列块，rank i 只算自己那段输出 <span class="mono">Y_i = X·A_i</span>。</div>
</div>

<p>这种"按列切"的逻辑，正是列并行线性层 <span class="mono">ColumnParallelLinear</span> 在代码里干的事：权重 A 沿<strong>第二维（列）</strong>切成 <span class="mono">A = [A_1 | … | A_p]</span>，rank i 只存 <span class="mono">A_i</span>、只算自己那段输出 <span class="mono">Y_i = X·A_i</span>；若设 <span class="mono">gather_output=True</span>，再把各片 all-gather 拼回完整的 Y。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/linear.py ::ColumnParallelLinear</span><span class="ln">权重按列切到各 TP rank：A = [A_1 | … | A_p]</span></div>
  <pre><span class="kw">class</span> ColumnParallelLinear(LinearBase):
    <span class="cm"># Y = X·A, with A split COLUMN-wise across TP ranks:</span>
    <span class="cm">#   A = [A_1 | A_2 | ... | A_p]</span>
    <span class="cm"># rank i stores only A_i and computes its slice Y_i = X·A_i;</span>
    <span class="cm"># gather_output=True all-gathers the pieces back into full Y.</span>
    <span class="kw">def</span> __init__(self, input_size, output_size, *,
                 gather_output=False, tp_rank=None, tp_size=None, ...):
        <span class="cm"># this rank keeps output_size / tp_size columns of the weight</span>
        ...</pre>
</div>

<p><strong>举个具体例子。</strong>某 MLP 的上投影 <span class="mono">up_proj</span> 输出维 <span class="mono">output_size = 11008</span>，在 <span class="mono">--tp-size 4</span> 下走列并行，于是每个 rank 只持有 <span class="mono">11008 / 4 = 2752</span> 列；这 2752 列正是从磁盘上的 <span class="mono">model-00001-of-00002.safetensors</span> 等分片里，按本 rank 的偏移流式读出后落到该卡显存的——这也正好把上面"图 1 加载流程"和"图 2 列切"两张图串了起来。</p>

<table class="t">
  <tr><th>加载要操心的事</th><th>为什么必须在这步做</th></tr>
  <tr><td>流式读取（边读边丢）</td><td>主机内存上界<strong>与模型大小解耦</strong>，几百 GB 也不爆内存</td></tr>
  <tr><td class="mono">load_weights</td><td>把 HF 名字映射到内部参数，并<strong>融合</strong> q/k/v、gate/up</td></tr>
  <tr><td>按 TP 切片</td><td>每卡只留<strong>自己那一片</strong>权重，显存才装得下（第 46 课）</td></tr>
  <tr><td>dtype / 量化</td><td>转 bf16/fp16 或装 FP8/INT4 + scales，决定前向用<strong>哪种内核</strong>（第 35 课）</td></tr>
  <tr><td><span class="mono">model.eval()</span></td><td>权重就位后切到推理模式，交给 ModelRunner（第 24 课）</td></tr>
</table>

<h2>不止一种 Loader：DefaultModelLoader 只是常路</h2>
<p>SGLang 里其实有<strong>一族</strong> Loader，对应不同的加载场景，但<span class="mono">DefaultModelLoader</span> 是<strong>最常走的那条路</strong>。
了解其余几种，能帮你看懂"加载"在不同部署下的变体：</p>

<p>这些变体共享同一套<strong>骨架</strong>，只在"权重从哪来、怎么摆"上各有侧重：<span class="mono">DummyModelLoader</span> 跳过真权重，专为压测与结构验证服务，启动极快；<span class="mono">ShardedStateLoader</span> 把"已经按某并行度切好的状态"直接喂回来，省掉每次重启时的重复切片，对反复拉起同一配置的线上服务很友好；按层加载与远程加载则在<strong>峰值内存</strong>和<strong>分发速度</strong>上各做取舍。但无论哪种，最终都殊途同归地走到同一个动作：<strong>把张量灌进模型参数</strong>。这就是"<strong>多方言、单内核</strong>"思路在加载侧的翻版：加载入口可以千变万化，模型那端只认 <span class="mono">load_weights</span> 一个接口。</p>

<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">常路</span><span class="name">DefaultModelLoader</span></div><div class="ld">从磁盘读 safetensors/bin 分片，<strong>流式</strong>喂给 <span class="mono">model.load_weights</span>。绝大多数模型、绝大多数部署都走它。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">测试</span><span class="name">DummyModelLoader</span></div><div class="ld">不读真权重，直接<strong>填随机/占位</strong>。用来量显存、测吞吐、跑结构正确性，不关心数值。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">快启</span><span class="name">ShardedStateLoader</span></div><div class="ld">从<strong>已按 TP 切好</strong>的分片状态直接载入，省去每次重新切片，适合反复重启同一并行度的服务。</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">省内存</span><span class="name">LayeredModelLoader / 远程</span></div><div class="ld">按层逐步加载，或从远程实例/对象存储拉取，进一步压低峰值内存或加速分发。</div></div>
</div>

<p>下面是 <span class="mono">DefaultModelLoader.load_model</span> 的真实骨架。读它你会清楚看到那"两步走"：<strong>先 <span class="mono">_initialize_model</span> 搭壳，
再把流式权重交给 <span class="mono">model.load_weights</span></strong>。注意 <span class="mono">_get_all_weights</span> 返回的是<strong>生成器</strong>，这正是"流式"的实现根基。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_loader/loader.py ::DefaultModelLoader</span><span class="ln">搭壳 + 流式灌权重</span></div>
  <pre><span class="kw">def</span> load_model(self, *, model_config, device_config):
    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)
    <span class="kw">with</span> set_default_torch_dtype(model_config.dtype):
        <span class="kw">with</span> target_device:
            model = _initialize_model(model_config, self.load_config, quant_config)  <span class="cm"># 空壳</span>
        self.load_weights_and_postprocess(
            model, self._get_all_weights(model_config, model), target_device)        <span class="cm"># 流式权重</span>
    <span class="kw">return</span> model.eval()

<span class="kw">def</span> load_weights_and_postprocess(model, weights, target_device):
    ...
    model.load_weights(weights)   <span class="cm"># 逐个 (name, tensor) 灌进模型参数</span></pre>
</div>

<p>把这一课接回主线：第 24 课的 ModelRunner 张口要的"加载好的模型"，就是 <span class="mono">DefaultModelLoader</span> 这样搭壳、映射、切片、量化、流式灌好之后交出来的成品。
往后第 26 课会带你<strong>真正写一个模型文件</strong>，重点正是它的 <span class="mono">load_weights</span> 怎么写（名字映射与融合都在那儿落地）；第 35 课讲<strong>量化</strong>格式如何加载与计算；
第 46 课讲<strong>张量并行</strong>怎么把权重按列/行切到多卡。带着"<strong>Loader = 把磁盘文件流式装配成每卡各就各位的张量</strong>"这把钥匙，这几课你会读得很顺。</p>

<p>最后把这一课的因果链收一收：模型之所以能"开箱即用"，是因为 Loader 在加载的那一刻就把<strong>四件麻烦事</strong>一并办妥了——名字对齐（映射）、形状打包（融合）、按卡切片（TP）、精度落地（dtype/量化），而且全程<strong>流式</strong>不爆内存。这四件事但凡漏一件，前向都会出问题：映射错了对不上参数，融合错了矩阵乘形状不对，切片错了跨卡通信对不齐，量化处理错了数值直接崩。正因为它们被集中、规范地安排在加载这一步，ModelRunner 才敢在前向里<strong>什么都不问、拿来就算</strong>。这就是"<strong>把复杂性收敛到边界</strong>"的工程美学：加载这道关把脏活累活全扛了，上层于是干净，整条推理流水线也因此更稳更快。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>两步走</strong>：<span class="mono">_initialize_model</span> 先搭<strong>空壳</strong>，再 <span class="mono">model.load_weights</span> 把权重<strong>流式</strong>灌进去。</li>
    <li><strong>流式加载</strong>：权重是 <span class="mono">(name, tensor)</span> 生成器，读一个灌一个，<strong>主机内存上界与模型大小解耦</strong>。</li>
    <li><strong>名字映射 + 融合</strong>：HF 名 →（模型自己的 <span class="mono">load_weights</span>）→ 内部参数，常把 q/k/v、gate/up <strong>融成打包权重</strong>。</li>
    <li><strong>TP 切片</strong>：列并行（q/k/v、gate/up）/行并行（o_proj、down_proj），每卡<strong>只留自己那一片</strong>（第 46 课）。</li>
    <li><strong>dtype/量化</strong>：转 bf16/fp16，或装 FP8/INT4/AWQ/GPTQ + scales（第 35 课）；<span class="mono">DefaultModelLoader</span> 是常路，另有 dummy/sharded/layered 等变体。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Last lesson's <span class="inline">ModelRunner</span> casually asked for "<strong>the loaded model</strong>" — but how does that model
actually <strong>move from disk onto the GPU</strong>? A 70B's weights are hundreds of GB, scattered across dozens of files, still
named the HuggingFace way, not matching SGLang's internal modules. This lesson is about the backstage role that turns "<strong>a
pile of disk files</strong>" into "<strong>tensors in their right place on each card</strong>": <span class="inline">DefaultModelLoader</span>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of loading a model as <strong>furnishing a new home from numbered flat-pack boxes</strong>: first read the <strong>parts
  list</strong> (the HuggingFace weight names, e.g. <span class="mono">model.layers.0.self_attn.q_proj.weight</span>), match each part
  to the <strong>right room</strong> (the corresponding SGLang module param), <strong>cut each shelf to fit</strong> when needed (slice
  it into this rank's share under tensor parallelism), and assemble in the <strong>right finish</strong> (cast to bf16/fp16, or load a
  quantized format). The key: <strong>don't dump all the boxes into the living room at once</strong> — carry them in <strong>one box at
  a time, streamed</strong>, or host RAM blows up on the spot. That "claim-by-list, cut-to-fit, stream-assemble" is exactly the Loader's job.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  <span class="mono">DefaultModelLoader.load_model</span> is the <strong>common entry</strong> for almost every model: it first builds an
  <strong>empty shell</strong> from <span class="mono">model_config</span> (structure present, weights still placeholders), then opens the
  <strong>safetensors shards</strong> on disk and <strong>streams the tensors in one by one</strong>. The feeding is handled by each
  model's <strong>own</strong> <span class="mono">load_weights(weights)</span>: it knows which internal param an HF name <strong>maps</strong>
  to, which weights to <strong>fuse</strong>, and which <strong>slice</strong> this rank takes. Because loading is funneled through this one
  point, SGLang can support <strong>hundreds of models + many quant formats</strong> under one framework.
</div>

<h2>load_model: build the shell first, then stream the weights in</h2>
<p>The whole load path is just two steps, but each hides nuance. <strong>Step one</strong>: within the target device (GPU) and target
dtype context, <span class="mono">_initialize_model</span> <strong>instantiates the structure</strong> — all the linear layers, attention,
and embeddings are built, but the params are still <strong>uninitialized placeholder tensors</strong> (already allocated at full size on the GPU, just not yet filled with real values).
<strong>Step two</strong>: call <span class="mono">load_weights_and_postprocess</span>, which gets a <strong>weight iterator</strong> from
<span class="mono">_get_all_weights</span> and hands it to <span class="mono">model.load_weights(weights)</span> to <strong>pour the real values
in, tensor by tensor</strong>. Note the subtlety: weights are not "read fully into memory then distributed" — they are a
<span class="mono">(name, tensor)</span> <strong>generator</strong>: read one, load one, drop one, so <strong>host memory stays pinned to a
small upper bound</strong> even for a hundreds-of-GB model.</p>

<p>Why insist on <strong>streaming</strong>? Picture an 8-GPU box loading a 70B: if you read all weights into host RAM before slicing and
dispatching, that temporary copy alone could approach or exceed physical memory, and safetensors' zero-copy advantage is wasted.
Streaming makes "read—map—slice—move-to-card" a <strong>read-and-drop pipeline</strong>: at any instant only the <strong>current tensor</strong>
(and the target slice being processed) lives in memory, so the bound is <strong>decoupled</strong> from the model's total size. That's why
the same machine can load a few-GB small model and a hundreds-of-GB large one — memory pressure is nearly <strong>constant</strong>. Once you
see this, you understand why SGLang defaults to safetensors, and why "slow loading" is usually disk/network bandwidth, not a memory bottleneck.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Locate the checkpoint</h4><p>Find the local or downloaded dir from <span class="mono">model_path</span> (HF / ModelScope).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>safetensors shards</h4><p>Open multiple <span class="mono">*.safetensors</span> by index; zero-copy, readable tensor-by-tensor.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Name mapping</h4><p>HF weight name →(the model's own <span class="mono">load_weights</span>)→ SGLang internal param, often <strong>fusing</strong> q/k/v, gate/up.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Slice for TP</h4><p>Each weight split column/row-wise; this rank <strong>takes only its slice</strong> (Lessons 24/46).</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>dtype / quant</h4><p>Cast to bf16/fp16, or load FP8/INT4/AWQ/GPTQ with scales (Lesson 35).</p></div></div>
  <div class="step"><div class="num">6</div><div class="sc"><h4>Land on GPU</h4><p>Write into the shell model's param slots; once loaded, <span class="mono">model.eval()</span>.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 290" role="img" aria-label="Load flow: safetensors shards on disk are streamed out by the Loader, cast to dtype or quantized with scales, then each rank's slice is scattered into its own GPU VRAM">
    <text x="20" y="28" style="font-weight:700;fill:var(--muted)">Load flow: disk shards → dtype / quant → each rank's VRAM</text>
    <text x="20" y="56" style="fill:var(--faint);font-size:12px">disk safetensors shards</text>
    <rect x="20" y="66" width="196" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="118" y="89" text-anchor="middle" class="mono" style="font-size:11px">model-00001-of-00002</text>
    <rect x="30" y="110" width="196" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="128" y="133" text-anchor="middle" class="mono" style="font-size:11px">model-00002-of-00002</text>
    <line x1="232" y1="112" x2="270" y2="112" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="270,107 280,112 270,117" style="fill:var(--faint)"/>
    <rect x="284" y="80" width="150" height="66" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="359" y="107" text-anchor="middle" style="font-weight:700;fill:var(--blue);font-size:13px">Loader streams</text>
    <text x="359" y="127" text-anchor="middle" class="mono" style="font-size:11px">(name, tensor)…</text>
    <line x1="440" y1="112" x2="478" y2="112" style="stroke:var(--faint);stroke-width:1.5"/>
    <polygon points="478,107 488,112 478,117" style="fill:var(--faint)"/>
    <rect x="492" y="80" width="150" height="66" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="567" y="107" text-anchor="middle" style="font-weight:700;fill:var(--amber);font-size:13px">cast dtype / quant</text>
    <text x="567" y="127" text-anchor="middle" class="mono" style="font-size:11px">bf16 · FP8 + scales</text>
    <line x1="644" y1="112" x2="664" y2="42" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="644" y1="112" x2="664" y2="100" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="644" y1="112" x2="664" y2="158" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="644" y1="112" x2="664" y2="216" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <rect x="664" y="20" width="120" height="44" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="724" y="40" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--teal)">GPU0 VRAM</text>
    <text x="724" y="56" text-anchor="middle" class="mono" style="font-size:10px">rank0 slice</text>
    <rect x="664" y="78" width="120" height="44" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="724" y="98" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--blue)">GPU1 VRAM</text>
    <text x="724" y="114" text-anchor="middle" class="mono" style="font-size:10px">rank1 slice</text>
    <rect x="664" y="136" width="120" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="724" y="156" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--amber)">GPU2 VRAM</text>
    <text x="724" y="172" text-anchor="middle" class="mono" style="font-size:10px">rank2 slice</text>
    <rect x="664" y="194" width="120" height="44" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="724" y="214" text-anchor="middle" style="font-weight:700;font-size:11px;fill:var(--purple)">GPU3 VRAM</text>
    <text x="724" y="230" text-anchor="middle" class="mono" style="font-size:10px">rank3 slice</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Load flow</b> — safetensors shards on disk are <strong>streamed</strong> out by the Loader; each tensor is cast to dtype or arranged by the quant layout with scales, then only <strong>each rank's slice</strong> is scattered into its own GPU VRAM.</div>
</div>

<h2>Name mapping &amp; fusion: matching HF names to SGLang params</h2>
<p>The on-disk weights use HuggingFace naming, e.g. attention's query projection is <span class="mono">model.layers.0.self_attn.q_proj.weight</span>,
with keys and values as <span class="mono">k_proj</span> and <span class="mono">v_proj</span>. But for <strong>performance</strong>, SGLang often
<strong>fuses q/k/v into one packed weight</strong> (one big matmul instead of three small ones) and fuses MLP's <span class="mono">gate_proj</span>
and <span class="mono">up_proj</span> into <strong>one</strong>. That creates a "mismatch": disk has <strong>three separate parts</strong>, memory wants
<strong>one merged part</strong>. Resolving it is each model class's own <span class="mono">load_weights(weights)</span> — it carries a built-in
"<strong>mapping table</strong>" that knows which <strong>offset segment</strong> of the packed weight each of
<span class="mono">q_proj/k_proj/v_proj</span> goes into, and how to stitch <span class="mono">gate/up</span>.</p>

<p>Handing "mapping" to the <strong>model itself</strong> rather than the Loader is a key division of duties. The Loader only <strong>streams
tensors out</strong> (it doesn't, and shouldn't, care what a model calls things internally), while "which of my params this name maps to,
whether to fuse, whether to transpose" is <strong>model-private knowledge</strong>, naturally encapsulated in the model's
<span class="mono">load_weights</span>. So when adding a new model you <strong>don't touch the Loader</strong> — you just write its
<span class="mono">load_weights</span> in the model file (Lesson 26). That's why SGLang can support hundreds of architectures relatively easily.
Mapping and fusion aren't optional details either: fusion directly sets the matmul shapes in the forward, and only when fused correctly can
the downstream TP slicing and quantization line up cleanly.</p>

<div class="cols">
  <div class="col"><h4>HF weight names on disk</h4><p>Stored separately: <span class="mono">...self_attn.q_proj.weight</span>, <span class="mono">k_proj.weight</span>,
  <span class="mono">v_proj.weight</span>; MLP is two parts, <span class="mono">gate_proj</span> and <span class="mono">up_proj</span>. Naming follows
  HuggingFace, <strong>one part per projection</strong>.</p></div>
  <div class="col"><h4>SGLang internal params</h4><p>Stored fused: q/k/v packed into <strong>one</strong> <span class="mono">qkv_proj</span>, gate/up into
  <strong>one</strong> <span class="mono">gate_up_proj</span>. <span class="mono">load_weights</span> writes the separate HF weights <strong>into each
  segment of the packed param</strong> by offset — one big matmul replaces several small ones.</p></div>
</div>

<h2>Slice for TP, land by dtype/quant</h2>
<p>After mapping come two more steps. <strong>First, tensor-parallel (TP) slicing</strong>: in a multi-GPU deployment, each weight is not stored
whole on every card but <strong>sliced per rank</strong>. Attention's q/k/v and MLP's gate/up go <strong>column-parallel</strong> (split on the
output dim, each card computes a slice of the output), while attention's <span class="mono">o_proj</span> and MLP's <span class="mono">down_proj</span>
go <strong>row-parallel</strong> (split on the input dim, each card computes a partial sum, then all-reduce). The Loader/model moves <strong>only
this rank's slice</strong> onto this card during loading, so each of 8 cards holds <strong>only 1/8 of the weights</strong> (Lessons 24/46). That's
why loading is itself distributed: every card reads the same files but <strong>takes what it needs and keeps one slice</strong>.</p>

<p><strong>Second, dtype and quantization</strong>. Most commonly, weights are <strong>cast to bf16 or fp16</strong> before going on-card; but if the
checkpoint is itself a <strong>quantized format</strong> — FP8, INT4, AWQ, GPTQ (Lesson 35) — loading must read in its <strong>scales / zero-points</strong>
too and arrange them by the quant layout. The forward then either uses <strong>dedicated quantized kernels</strong> directly in low bits, or
<strong>lazily dequantizes</strong> back to high precision when used. Quantization's point is to <strong>fit a bigger model in less memory</strong>, at the
cost of correctly handling scales and packing formats — details likewise tucked into each quant method's loading logic, with the Loader just handing it
the byte stream. Because both "slicing" and "quantization" are arranged here at load time, the upper-level ModelRunner (Lesson 24) receives an
<strong>out-of-the-box, all-in-place</strong> model.</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="weight matrix A is split along its second dimension (columns) into A_1 A_2 A_3 A_4, and TP GPU0 through GPU3 each hold only one column block, with rank i computing Y_i equals X times A_i">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">weight A split column-wise across TP ranks: A = [A_1 | A_2 | A_3 | A_4]</text>
    <rect x="20" y="58" width="56" height="110" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="48" y="118" text-anchor="middle" class="mono">X</text>
    <text x="92" y="120" text-anchor="middle" style="fill:var(--faint);font-size:18px">×</text>
    <rect x="112" y="58" width="84" height="110" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="154" y="118" text-anchor="middle" class="mono">A_1</text>
    <rect x="200" y="58" width="84" height="110" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="242" y="118" text-anchor="middle" class="mono">A_2</text>
    <rect x="288" y="58" width="84" height="110" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="330" y="118" text-anchor="middle" class="mono">A_3</text>
    <rect x="376" y="58" width="84" height="110" rx="4" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="418" y="118" text-anchor="middle" class="mono">A_4</text>
    <text x="240" y="192" text-anchor="middle" style="fill:var(--faint);font-size:12px">one weight matrix split column-wise (2nd dim) into 4 blocks</text>
    <line x1="154" y1="168" x2="105" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="242" y1="168" x2="295" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="330" y1="168" x2="485" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="418" y1="168" x2="675" y2="212" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <rect x="20" y="214" width="170" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="105" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--blue)">GPU 0 · rank 0</text>
    <text x="105" y="262" text-anchor="middle" class="mono" style="font-size:11px">holds A_1 → Y_0 = X·A_1</text>
    <rect x="210" y="214" width="170" height="64" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="295" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--teal)">GPU 1 · rank 1</text>
    <text x="295" y="262" text-anchor="middle" class="mono" style="font-size:11px">holds A_2 → Y_1 = X·A_2</text>
    <rect x="400" y="214" width="170" height="64" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="485" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--amber)">GPU 2 · rank 2</text>
    <text x="485" y="262" text-anchor="middle" class="mono" style="font-size:11px">holds A_3 → Y_2 = X·A_3</text>
    <rect x="590" y="214" width="170" height="64" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="675" y="240" text-anchor="middle" style="font-weight:700;font-size:12px;fill:var(--purple)">GPU 3 · rank 3</text>
    <text x="675" y="262" text-anchor="middle" class="mono" style="font-size:11px">holds A_4 → Y_3 = X·A_4</text>
  </svg>
  <div class="figcap"><b>Fig 2 · weights sharded across TP ranks</b> — one weight matrix A is split along its second dimension into <span class="mono">A = [A_1 | A_2 | A_3 | A_4]</span>; each TP rank (GPU 0…3) holds only its own column block and computes just its slice of the output <span class="mono">Y_i = X·A_i</span>.</div>
</div>

<p>This "split column-wise" logic is exactly what the column-parallel linear layer <span class="mono">ColumnParallelLinear</span> does in code: weight A is split along its <strong>second dimension (columns)</strong> into <span class="mono">A = [A_1 | … | A_p]</span>, rank i stores only <span class="mono">A_i</span> and computes just its slice of the output <span class="mono">Y_i = X·A_i</span>; with <span class="mono">gather_output=True</span>, the pieces are all-gathered back into the full Y.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/linear.py ::ColumnParallelLinear</span><span class="ln">weight split column-wise across TP ranks: A = [A_1 | … | A_p]</span></div>
  <pre><span class="kw">class</span> ColumnParallelLinear(LinearBase):
    <span class="cm"># Y = X·A, with A split COLUMN-wise across TP ranks:</span>
    <span class="cm">#   A = [A_1 | A_2 | ... | A_p]</span>
    <span class="cm"># rank i stores only A_i and computes its slice Y_i = X·A_i;</span>
    <span class="cm"># gather_output=True all-gathers the pieces back into full Y.</span>
    <span class="kw">def</span> __init__(self, input_size, output_size, *,
                 gather_output=False, tp_rank=None, tp_size=None, ...):
        <span class="cm"># this rank keeps output_size / tp_size columns of the weight</span>
        ...</pre>
</div>

<p><strong>A concrete example.</strong> An MLP's up-projection <span class="mono">up_proj</span> has output dim <span class="mono">output_size = 11008</span>; under <span class="mono">--tp-size 4</span> it goes column-parallel, so each rank holds only <span class="mono">11008 / 4 = 2752</span> columns. Those 2752 columns are exactly what gets streamed out of the on-disk shards like <span class="mono">model-00001-of-00002.safetensors</span> by this rank's offset and landed into its GPU VRAM — tying "Fig 1 load flow" and "Fig 2 column split" together.</p>

<table class="t">
  <tr><th>What loading must handle</th><th>Why it must happen at this step</th></tr>
  <tr><td>Streaming (read-and-drop)</td><td>Host-memory bound is <strong>decoupled from model size</strong> — hundreds of GB won't blow up RAM</td></tr>
  <tr><td class="mono">load_weights</td><td>Maps HF names to internal params and <strong>fuses</strong> q/k/v, gate/up</td></tr>
  <tr><td>Slice for TP</td><td>Each card keeps <strong>only its slice</strong> so the weights fit in memory (Lesson 46)</td></tr>
  <tr><td>dtype / quant</td><td>Cast bf16/fp16 or load FP8/INT4 + scales, deciding <strong>which kernel</strong> the forward uses (Lesson 35)</td></tr>
  <tr><td><span class="mono">model.eval()</span></td><td>Switch to inference mode once weights are in place; hand to ModelRunner (Lesson 24)</td></tr>
</table>

<h2>More than one Loader: DefaultModelLoader is just the common path</h2>
<p>SGLang actually has a <strong>family</strong> of Loaders for different scenarios, but <span class="mono">DefaultModelLoader</span> is <strong>the most
common path</strong>. Knowing the others helps you see how "loading" varies across deployments:</p>

<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">common</span><span class="name">DefaultModelLoader</span></div><div class="ld">Reads safetensors/bin shards from disk and <strong>streams</strong> them to <span class="mono">model.load_weights</span>. Most models, most deployments use it.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">testing</span><span class="name">DummyModelLoader</span></div><div class="ld">Reads no real weights; just fills <strong>random/placeholder</strong> values. For measuring memory, throughput, and structural correctness, not numerics.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">fast start</span><span class="name">ShardedStateLoader</span></div><div class="ld">Loads directly from an <strong>already-TP-sliced</strong> sharded state, skipping re-slicing each time — ideal for repeatedly restarting the same parallelism.</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">low memory</span><span class="name">LayeredModelLoader / remote</span></div><div class="ld">Loads layer by layer, or pulls from a remote instance/object store, to further cut peak memory or speed up distribution.</div></div>
</div>

<p>Here is the real skeleton of <span class="mono">DefaultModelLoader.load_model</span>. Read it and the "two steps" are clear: <strong>first
<span class="mono">_initialize_model</span> builds the shell, then the streamed weights go to <span class="mono">model.load_weights</span></strong>. Note
that <span class="mono">_get_all_weights</span> returns a <strong>generator</strong> — that is the very basis of "streaming".</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_loader/loader.py ::DefaultModelLoader</span><span class="ln">build shell + stream weights</span></div>
  <pre><span class="kw">def</span> load_model(self, *, model_config, device_config):
    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)
    <span class="kw">with</span> set_default_torch_dtype(model_config.dtype):
        <span class="kw">with</span> target_device:
            model = _initialize_model(model_config, self.load_config, quant_config)  <span class="cm"># shell</span>
        self.load_weights_and_postprocess(
            model, self._get_all_weights(model_config, model), target_device)        <span class="cm"># streamed weights</span>
    <span class="kw">return</span> model.eval()

<span class="kw">def</span> load_weights_and_postprocess(model, weights, target_device):
    ...
    model.load_weights(weights)   <span class="cm"># pour each (name, tensor) into the model's params</span></pre>
</div>

<p>Tie this back to the main thread: the "loaded model" that Lesson 24's ModelRunner demanded is exactly what
<span class="mono">DefaultModelLoader</span> hands over after building the shell, mapping, slicing, quantizing, and streaming the weights in. Next,
Lesson 26 walks you through <strong>actually writing a model file</strong>, focused on writing its <span class="mono">load_weights</span> (where name
mapping and fusion land); Lesson 35 covers how <strong>quantized</strong> formats load and compute; Lesson 46 covers how <strong>tensor parallelism</strong>
slices weights column/row-wise across cards. With the key "<strong>Loader = stream disk files into all-in-place tensors per card</strong>", these
lessons will read smoothly.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Two steps</strong>: <span class="mono">_initialize_model</span> builds the <strong>shell</strong>, then <span class="mono">model.load_weights</span> <strong>streams</strong> the weights in.</li>
    <li><strong>Streaming</strong>: weights are a <span class="mono">(name, tensor)</span> generator — read one, load one — so the <strong>host-memory bound is decoupled from model size</strong>.</li>
    <li><strong>Name mapping + fusion</strong>: HF name →(the model's own <span class="mono">load_weights</span>)→ internal param, often <strong>fusing</strong> q/k/v and gate/up into packed weights.</li>
    <li><strong>TP slicing</strong>: column-parallel (q/k/v, gate/up) / row-parallel (o_proj, down_proj); each card <strong>keeps only its slice</strong> (Lesson 46).</li>
    <li><strong>dtype/quant</strong>: cast bf16/fp16, or load FP8/INT4/AWQ/GPTQ + scales (Lesson 35); <span class="mono">DefaultModelLoader</span> is the common path, with dummy/sharded/layered variants.</li>
  </ul>
</div>
""",
}

LESSON_26 = {"zh": r"""
<p class="lead">
上一课我们看了权重<strong>怎么进来</strong>（load_weights，第 25 课）。这一课回答一个更让人好奇的问题：在 SGLang 里
<strong>"写一个模型"到底要写什么</strong>？答案出乎意料地轻——一个模型文件，本质上就是把 SGLang 早就备好的
<strong>并行层</strong>像积木一样<strong>搭起来</strong>。你不写注意力内核、不写跨卡通信、不写 KV 管理，只描述<strong>架构</strong>。
这正是 SGLang "对新模型几乎是 day-0 支持、模型覆盖面极广"背后那个朴素的工程真相。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把写模型想成<strong>用标准乐高积木拼模型</strong>：SGLang 已经把每一种砖块都<strong>注塑成型</strong>了——并行线性层、注意力、归一化、旋转位置编码，
  全是现成、已调好、能跨卡的零件。你要做的只有两件事：①照图纸把砖块<strong>卡成正确的形状</strong>（架构：用哪些层、维度多大、残差怎么连）；
  ②给每个零件<strong>贴标签</strong>，说清"磁盘上哪个盒子里的零件该装到这儿"（load_weights 的名字映射）。你<strong>从不需要自己注塑一块砖</strong>——
  注意力怎么在 GPU 上算、8 张卡怎么把结果拼齐、KV 该写到池子哪一页，都封装在砖块内部，与你无关。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一个 SGLang 模型文件就是一组互相嵌套的 PyTorch <span class="mono">nn.Module</span>。以 Llama 为例，四个类<strong>层层套娃</strong>：
  <span class="mono">LlamaAttention</span>（注意力）→ <span class="mono">LlamaMLP</span>（前馈）→ 二者组成 <span class="mono">LlamaDecoderLayer</span>（一层解码层）→
  N 层 + 嵌入 + 末端归一组成 <span class="mono">LlamaModel</span> → 再加 <span class="mono">lm_head</span> 与 <span class="mono">forward</span> 就是
  <span class="mono">LlamaForCausalLM</span>（对外的整模型）。每个类内部用的全是 SGLang 的<strong>可复用层</strong>。所以"加新模型"在绝大多数情况下
  <strong>就是一次组装</strong>——写一份这样的文件、注册一下，就能上线服务。
</div>

<h2>四个类，层层套娃：Llama 是怎么搭出来的</h2>
<p>先把这四层拆开看，你会发现每一层都<strong>只负责一件事</strong>，而且每一件事用的零件都来自 SGLang，而非你手写。最里层是
<span class="mono">LlamaAttention</span>：它先用一个<strong>列并行线性层</strong> <span class="mono">QKVParallelLinear</span> 把隐藏向量一次投影出 q/k/v
（三者被打包在一个权重里，正是第 25 课 load_weights 要做融合映射的原因），对 q/k 施加 <strong>RoPE</strong> 旋转位置编码，再把 q 和历史
k/v 交给 <span class="mono">RadixAttention</span>（注意力层）去算——这一步会<strong>读写 KV 池</strong>，具体用哪个注意力后端、KV 落在池子哪些槽位，
全由传进来的 <span class="mono">forward_batch</span> 决定（第 33 课）；最后 o_proj 用<strong>行并行线性层</strong> <span class="mono">RowParallelLinear</span>
把多头输出投影回隐藏维。注意：q/k/v、gate/up 走<strong>列并行</strong>（按输出维切片），o_proj、down_proj 走<strong>行并行</strong>（按输入维切片再 all-reduce），
这套切法第 46 课会细讲，但模型作者<strong>只是挑用哪种层，切片与通信都在层内部自动发生</strong>。</p>
<p>外面一层是 <span class="mono">LlamaMLP</span>：gate/up 投影 + <strong>SiLU</strong> 激活（<span class="mono">SiluAndMul</span>）+ down 投影，没有任何花活。再外一层
<span class="mono">LlamaDecoderLayer</span> 把注意力和 MLP 缝合成一层标准 Transformer 块，关键在<strong>残差结构</strong>：input_norm → 注意力 → 残差相加 →
post_norm → MLP → 残差相加。<span class="mono">LlamaModel</span> 则是骨架：<span class="mono">embed_tokens</span>（词表并行嵌入，<span class="mono">VocabParallelEmbedding</span>）
→ N 层解码层 → 末端 <span class="mono">RMSNorm</span>。最外层 <span class="mono">LlamaForCausalLM</span> 才是对运行时暴露的"整模型"：它持有 <span class="mono">model</span>、
一个 <span class="mono">lm_head</span>（<span class="mono">ParallelLMHead</span>，权重可与嵌入共享 tie_word_embeddings），一个 <span class="mono">forward(input_ids, positions,
forward_batch)</span> 返回 logits，以及 <span class="mono">load_weights</span>（第 25 课）。ModelRunner（第 24 课）握住的就是这个最外层类。</p>

<p>这种"套娃"不是为了好看，而是 PyTorch <span class="mono">nn.Module</span> 组合范式的自然结果：父模块在 <span class="mono">__init__</span> 里把子模块当成普通属性挂上去，<span class="mono">forward</span> 里再按顺序调用它们，于是"注册参数、搬到 GPU、保存 / 加载"这些事 PyTorch 会沿着这棵模块树<strong>自动递归</strong>完成。这也解释了第 25 课的 <span class="mono">load_weights</span> 为什么能按 <span class="mono">model.layers.0.self_attn.qkv_proj</span> 这样的<strong>点分路径</strong>精确定位每个权重——路径就是套娃的层级，名字天然对应模块树上的位置。理解了这一点，你看任何一个 SGLang 模型文件都能一眼分辨"谁套着谁"，进而知道某个权重该塞进哪个子模块。</p>

<p>再看 <span class="mono">LlamaAttention</span> 里一个体现"组装"威力的细节：它要处理<strong>分组查询注意力</strong>（GQA）——查询头多、键值头少。代码里据 <span class="mono">tp_size</span> 把总头数、总 KV 头数分摊到每张卡，当 KV 头数比卡数还少时就<strong>复制</strong>而非切分。这些边界逻辑看着琐碎，但它们只决定"每张卡分到几个头、维度多大"，算出来的 <span class="mono">num_heads</span>、<span class="mono">head_dim</span> 再原样交给 <span class="mono">QKVParallelLinear</span> 和 <span class="mono">RadixAttention</span>。也就是说，连 GQA 这种看似要动注意力实现的特性，在模型文件里也只是<strong>几行维度算术</strong>，真正的注意力计算仍然原封不动地复用库里的那一个算子。</p>

<p>同样的"组装"思路也贯穿 <span class="mono">LlamaMLP</span>：gate 与 up 两个投影被合并成一个 <span class="mono">MergedColumnParallelLinear</span>（磁盘上是分开的两份权重，加载时由 load_weights 拼成一份，第 25 课），<span class="mono">SiluAndMul</span> 则把"对 gate 做 SiLU、再与 up 逐元素相乘"这两步融合进一个算子。你会发现一个反复出现的模式：<strong>凡是能融合的相邻操作，库都提前打包好了</strong>——QKV 合一、gate/up 合一、norm 与残差合一。模型作者只要知道"该用哪个打包好的层"，就自动享受到这些算子级优化，而不必理解它们为什么更快。这种"默认就快"的设计，让一份朴素的模型文件也能跑出接近手工优化的性能。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">入口</span><span class="name">LlamaForCausalLM</span></div><div class="ld">对运行时暴露的整模型：持有 <span class="mono">model</span> + <span class="mono">lm_head</span>，提供 <span class="mono">forward(...)→logits</span> 与 <span class="mono">load_weights</span>（第 25 课）。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">骨架</span><span class="name">LlamaModel</span></div><div class="ld"><span class="mono">embed_tokens</span>（词表并行嵌入）→ N× 解码层 → 末端 <span class="mono">RMSNorm</span>。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">× N</span><span class="name">LlamaDecoderLayer</span></div><div class="ld">一层标准块：input_norm → 注意力 → 残差 → post_norm → MLP → 残差。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">子层</span><span class="name">LlamaAttention</span></div><div class="ld">QKV 列并行投影 + RoPE + <span class="mono">RadixAttention</span>（读写 KV 池，第 33 课）+ o_proj 行并行。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">子层</span><span class="name">LlamaMLP</span></div><div class="ld">gate/up 列并行 + <span class="mono">SiLU</span> + down 行并行。</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">出口</span><span class="name">lm_head</span></div><div class="ld">把末端 hidden 投影到词表维，得到 <strong>logits</strong>（可与嵌入共享权重）。</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 340" role="img" aria-label="解码器模型的层栈：embed_tokens 经过 N 层 LlamaDecoderLayer（每层含 self_attn 与 MLP，配 RMSNorm 与残差）再到末端 norm、lm_head，最后得到 logits">
    <rect x="255" y="20" width="250" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="380" y="40" text-anchor="middle" class="mono" style="font-size:12px">embed_tokens</text>
    <line x1="380" y1="50" x2="380" y2="68" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,72 375,64 385,64" style="fill:var(--muted)"/>
    <rect x="170" y="72" width="420" height="142" rx="8" style="fill:var(--panel-2);stroke:var(--accent);stroke-width:1.5"/>
    <text x="186" y="92" style="font-weight:700;fill:var(--accent-ink);font-size:13px">N × LlamaDecoderLayer</text>
    <rect x="520" y="78" width="60" height="22" rx="11" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="550" y="93" text-anchor="middle" class="mono" style="font-size:11px">× 32</text>
    <rect x="190" y="104" width="180" height="48" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="280" y="124" text-anchor="middle" style="font-size:12px">RMSNorm + self_attn</text>
    <text x="280" y="142" text-anchor="middle" style="fill:var(--muted);font-size:11px">注意力子层</text>
    <rect x="394" y="104" width="176" height="48" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="482" y="124" text-anchor="middle" style="font-size:12px">RMSNorm + MLP</text>
    <text x="482" y="142" text-anchor="middle" style="fill:var(--muted);font-size:11px">前馈子层</text>
    <text x="380" y="182" text-anchor="middle" style="fill:var(--purple);font-weight:700;font-size:12px">+ 残差 (residual) × 2，逐层向下传递</text>
    <text x="380" y="202" text-anchor="middle" style="fill:var(--faint);font-size:11px">Llama-2-7B：32 层 · hidden 4096</text>
    <line x1="380" y1="214" x2="380" y2="232" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,236 375,228 385,228" style="fill:var(--muted)"/>
    <rect x="265" y="236" width="230" height="28" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="380" y="255" text-anchor="middle" class="mono" style="font-size:12px">norm (RMSNorm)</text>
    <line x1="380" y1="264" x2="380" y2="280" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,284 375,276 385,276" style="fill:var(--muted)"/>
    <rect x="265" y="284" width="230" height="28" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="303" text-anchor="middle" class="mono" style="font-size:12px">lm_head</text>
    <line x1="380" y1="312" x2="380" y2="326" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,330 375,322 385,322" style="fill:var(--muted)"/>
    <text x="430" y="334" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">→ logits</text>
  </svg>
  <div class="figcap"><b>图 1 · 解码器模型的层栈</b> — 一个 SGLang 模型就是把 <span class="mono">embed_tokens</span> → N×<span class="mono">LlamaDecoderLayer</span>（每层 = self_attn + MLP，配 RMSNorm 与残差）→ 末端 <span class="mono">norm</span> → <span class="mono">lm_head</span> 竖着叠起来。Llama-2-7B 就是 32 层、hidden 4096 的这样一摞。</div>
</div>

<h2>一层里发生了什么：跟着残差走一遍</h2>
<p>值得强调的是，<strong>层的选择本身就编码了并行策略</strong>。当作者把 q/k/v 投影写成 <span class="mono">QKVParallelLinear</span>、把 o_proj 写成 <span class="mono">RowParallelLinear</span> 时，他没有写任何一行通信代码，却已经决定了：这层在张量并行下，权重按输出维切给各卡、各卡先各算一段、再在 o_proj 之后用一次 all-reduce 把结果拼齐（第 46 课）。换句话说，"选哪种并行层"就是在"选这层怎么跨卡分布"。同理，<span class="mono">VocabParallelEmbedding</span> 把词表维切开、<span class="mono">ParallelLMHead</span> 把输出投影切开。作者要操心的只是"维度对不对、切法配不配套"，真正的切片与集合通信全在层内部默默发生——这正是模型文件能保持简短的关键。</p>
<p>把镜头推进到<strong>单独一层</strong>解码层，你能看清"组装"具体组装出了什么。数据带着上一层的残差进来，先做一次
<span class="mono">input_layernorm</span>（RMSNorm，第 36 课）；归一化后的向量进注意力子层——投影出 q/k/v、给 q/k 转上 RoPE、当前 k/v
按 <span class="mono">forward_batch</span> 指定的槽位<strong>写进 KV 池</strong>、再让 q 去看自己与所有历史 k/v（若前缀已被 RadixAttention 缓存就直接复用，第 7 课）；
注意力输出经 o_proj 回到隐藏维，与残差<strong>相加</strong>。接着 <span class="mono">post_attention_layernorm</span> 再归一化一次，进 MLP，又一次残差相加。
就这么六步，一层结束，把 <span class="mono">(hidden_states, residual)</span> 交给下一层。N 层叠完做末端归一，<span class="mono">lm_head</span> 投影出 logits。
注意 <span class="mono">forward_batch</span> 是怎么<strong>一路穿到</strong>每一层注意力的——它就是模型文件与运行时之间的<strong>那道缝</strong>：模型只管"把张量算对"，
"该读写 KV 池哪些槽位"由运行时通过 forward_batch 告诉它。</p>

<p>这里还藏着一个常被忽略的细节：<span class="mono">residual</span> 是<strong>跨层传递</strong>的。注意看代码里第一层进来时 residual 为 None、于是把输入本身当残差；之后每层都把"归一化前的值"作为残差留到相加时用。SGLang 的 <span class="mono">RMSNorm</span> 还把"归一化 + 残差相加"<strong>融合</strong>成一次调用——<span class="mono">input_layernorm(hidden, residual)</span> 同时返回新 hidden 和更新后的 residual，省去一次显存往返。这种把常见组合"焊死"成单个高效算子的做法，正是底层库替你做掉的优化：你照着调用就行，完全不必关心它内部怎么把两步合一。模型作者眼里只有"先 norm 再 attn 再残差"这条清晰的逻辑，看不到也不需要看到融合内核的存在。</p>

<p>顺带说清一个容易混的点：解码阶段每一步其实只前向<strong>一个</strong>新 token，但它要看<strong>全部历史</strong>的 K/V；预填充阶段则一次前向<strong>整段</strong>提示的多个 token。同一份 <span class="mono">forward</span> 代码两种情形都能跑，靠的就是 <span class="mono">forward_batch</span> 里的 <span class="mono">forward_mode</span> 和注意力元数据替模型把差异吸收掉——模型文件里你看不到 EXTEND / DECODE 的分叉，那都在注意力层与 ModelRunner（第 24 课）里处理了。这进一步印证了本课主线：模型作者描述的是"<strong>静态的架构</strong>"，而"<strong>动态的怎么跑</strong>"由运行时与库层接管。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>input_norm</h4><p>对输入做 RMSNorm（第 36 课），保留 residual 备用。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>注意力（读写 KV）</h4><p>QKV 投影 + RoPE；按 <span class="mono">forward_batch</span> 写/读 KV 池，q 看历史 k/v（第 33 课）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>残差相加</h4><p>注意力输出经 o_proj 回隐藏维，与 residual 相加。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>post_norm</h4><p>再做一次 RMSNorm，准备进 MLP。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>MLP</h4><p>gate/up + <span class="mono">SiLU</span> + down，提取非线性特征。</p></div></div>
  <div class="step"><div class="num">6</div><div class="sc"><h4>残差相加 → 下一层</h4><p>与 residual 相加，交出 <span class="mono">(hidden, residual)</span>。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="LlamaAttention.forward 把注意力委托给 self.attn：hidden_states 经 qkv_proj、拆分 q/k/v、rotary_emb，进入高亮的 self.attn（RadixAttention，SGLang 缝合点），再经 o_proj 得到 output">
    <text x="20" y="32" style="font-weight:700;fill:var(--muted);font-size:13px">LlamaAttention.forward(positions, hidden_states, forward_batch)</text>
    <rect x="20" y="68" width="96" height="48" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="68" y="96" text-anchor="middle" class="mono" style="font-size:11px">hidden</text>
    <line x1="116" y1="92" x2="128" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="132,92 124,87 124,97" style="fill:var(--muted)"/>
    <rect x="130" y="68" width="96" height="48" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="178" y="96" text-anchor="middle" class="mono" style="font-size:11px">qkv_proj</text>
    <line x1="226" y1="92" x2="238" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="242,92 234,87 234,97" style="fill:var(--muted)"/>
    <rect x="240" y="68" width="96" height="48" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="288" y="90" text-anchor="middle" class="mono" style="font-size:11px">split</text>
    <text x="288" y="106" text-anchor="middle" class="mono" style="font-size:11px">q,k,v</text>
    <line x1="336" y1="92" x2="348" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="352,92 344,87 344,97" style="fill:var(--muted)"/>
    <rect x="350" y="68" width="96" height="48" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="398" y="96" text-anchor="middle" class="mono" style="font-size:11px">rotary_emb</text>
    <line x1="446" y1="92" x2="458" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="462,92 454,87 454,97" style="fill:var(--muted)"/>
    <rect x="460" y="58" width="100" height="68" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2.5"/>
    <text x="510" y="86" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:12px">self.attn</text>
    <text x="510" y="104" text-anchor="middle" class="mono" style="font-size:10px">RadixAttention</text>
    <line x1="560" y1="92" x2="572" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="576,92 568,87 568,97" style="fill:var(--muted)"/>
    <rect x="574" y="68" width="96" height="48" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="622" y="96" text-anchor="middle" class="mono" style="font-size:11px">o_proj</text>
    <line x1="670" y1="92" x2="682" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="686,92 678,87 678,97" style="fill:var(--muted)"/>
    <rect x="684" y="68" width="96" height="48" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="732" y="96" text-anchor="middle" class="mono" style="font-size:11px">output</text>
    <line x1="510" y1="126" x2="510" y2="158" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <polygon points="510,162 505,154 515,154" style="fill:var(--accent)"/>
    <rect x="360" y="162" width="300" height="64" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="510" y="186" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:12px">SGLang 缝合点</text>
    <text x="510" y="206" text-anchor="middle" style="fill:var(--ink);font-size:11px">KV 缓存 + 分页 + 注意力后端在此接入</text>
    <text x="510" y="221" text-anchor="middle" style="fill:var(--muted);font-size:11px">由 forward_batch 驱动（第 33 课）</text>
  </svg>
  <div class="figcap"><b>图 2 · forward() 把注意力委托给 self.attn</b> — <span class="mono">LlamaAttention.forward</span> 自己只做投影、拆分、RoPE 这些"对齐张量"的活，真正的注意力计算交给高亮的 <span class="mono">self.attn</span>（<span class="mono">RadixAttention</span>）——这正是 KV 缓存与注意力后端插进来的<strong>缝</strong>，由 <span class="mono">forward_batch</span> 驱动。</div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/models/llama.py ::LlamaAttention</span><span class="ln">注意力层把算子委托给 self.attn（RadixAttention）</span></div>
  <pre><span class="kw">class</span> <span class="st">LlamaAttention</span>(nn.Module):
    <span class="kw">def</span> __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)   <span class="cm"># 融合的 q,k,v 投影</span>
        self.o_proj   = RowParallelLinear(...)
        self.rotary_emb = get_rope(...)
        self.attn = RadixAttention(...)          <span class="cm"># &lt;- SGLang 的那道缝</span>
    <span class="kw">def</span> forward(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)  <span class="cm"># KV 缓存 + 后端</span>
        output, _ = self.o_proj(attn_output)
        <span class="kw">return</span> output</pre>
</div>

<p>把这段代码和上图对起来看，你会发现整个 <span class="mono">LlamaAttention</span> 里<strong>没有一行注意力数学</strong>：它只负责投影、拆分、转 RoPE，然后把 q/k/v 原样递给 <span class="mono">self.attn</span>。最能说明问题的是这个具体例子——<strong>只要把 <span class="mono">self.attn = RadixAttention(...)</span> 这一行写进新模型</strong>，它就自动继承了前缀缓存（第 7 课）、分页 KV 与所有注意力后端（第 33 课），你不必碰任何一行缓存或内核代码。再配一组真实数字感受一下规模：<span class="mono">Llama-2-7B</span> 就是把上面这层注意力 + MLP 打包成 <strong>32 个</strong> <span class="mono">LlamaDecoderLayer</span>、hidden 维 <strong>4096</strong> 竖着叠起来——同一份 <span class="mono">LlamaAttention</span> 代码被实例化 32 次，每次只是 <span class="mono">layer_id</span> 与切片不同，算子始终是库里那一个。</p>

<h2>谁写、谁给：一张分工表把成本说清</h2>
<p>现在回到那个核心命题：为什么"加一个模型"在 SGLang 里这么便宜？看下面这张表——四（五）个类各自的职责，<strong>无一例外</strong>
都是在"挑选并连接"SGLang 现成的层，没有一行是在实现底层算子。把它和右边"你写 vs SGLang 给"的对照一起读，你就明白：模型作者真正
<strong>新写</strong>的内容只有两类——<strong>架构</strong>（用哪些层、维度多大、残差怎么连）和 <strong>load_weights 的名字映射</strong>（哪个 HF 权重对应内部哪个参数、要不要融合）。
其余的注意力内核、TP 通信、KV 管理、CUDA 图，全是<strong>白拿</strong>的。这就是"广泛模型支持"成本低的根因。</p>

<table class="t">
  <tr><th>类</th><th>职责（都在"组装现成层"）</th><th>用到的 SGLang 层</th></tr>
  <tr><td class="mono">LlamaAttention</td><td>q/k/v 投影 + RoPE + 注意力 + 输出投影</td><td class="mono">QKVParallelLinear / RadixAttention / RowParallelLinear</td></tr>
  <tr><td class="mono">LlamaMLP</td><td>gate/up + 激活 + down</td><td class="mono">MergedColumnParallelLinear / SiluAndMul</td></tr>
  <tr><td class="mono">LlamaDecoderLayer</td><td>缝合注意力 + MLP，定义残差结构</td><td class="mono">RMSNorm（×2）</td></tr>
  <tr><td class="mono">LlamaModel</td><td>嵌入 → N 层 → 末端归一</td><td class="mono">VocabParallelEmbedding / RMSNorm</td></tr>
  <tr><td class="mono">LlamaForCausalLM</td><td>整模型：forward→logits + load_weights</td><td class="mono">ParallelLMHead / LogitsProcessor</td></tr>
</table>

<p>把这张表再读深一层会更有体会：凡是带 "Parallel" 字样的层，背后都站着一整套张量并行的切片与通信实现；<span class="mono">RadixAttention</span> 背后是前缀缓存复用（第 7 课）+ 分页 KV + 多种硬件注意力后端（第 33 课）；<span class="mono">RMSNorm</span>、RoPE 背后是手写的融合内核（第 36 课）。这些每一项单独拎出来都是一篇大文章，但对模型作者而言它们只是<strong>一行构造、一次调用</strong>。这就是"复杂度下沉、接口上浮"的威力：把难的东西沉到可复用层里，露在模型文件表面的，只剩下"这个模型长什么样"。</p>

<div class="cols">
  <div class="col"><h4>你写的（模型文件）</h4><p><strong>架构</strong>：用哪些层、各维度、残差结构、N 层怎么叠；<strong>load_weights</strong>：HF 权重名 → 内部参数名的映射与 q/k/v、gate/up 融合（第 25 课）。仅此而已——一份文件、几百行。</p></div>
  <div class="col"><h4>SGLang 提供的（白拿）</h4><p>并行线性层（列/行并行，自动切片 + 通信，第 46 课）、<strong>注意力后端</strong>与 KV 管理（第 33 课）、RoPE/RMSNorm（第 36 课）、CUDA 图（第 27 课）、采样（第 28 课）。你<strong>一行都不用碰</strong>。</p></div>
</div>

<p>反过来想，这套分工也<strong>约束</strong>了模型作者：你能表达的架构，受限于 SGLang 已提供的层。如果一个新模型用了某种全新的注意力变体或激活，而库里还没有对应层，那就需要<strong>先给库添一块新积木</strong>（一个新的可复用层），再在模型文件里组装它。但现实是，主流开源模型的构件高度同质——都是某种线性投影、某种注意力、某种归一化的排列组合，所以绝大多数新模型落地时，库里的积木<strong>早已齐备</strong>，作者真的只需组装。这正是"day-0 支持"在工程上能成立的前提，也是为什么 SGLang 的模型目录能轻松扩张到上百个文件而不失控。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/models/llama.py ::LlamaForCausalLM</span><span class="ln">组装即模型</span></div>
  <pre><span class="kw">class</span> <span class="st">LlamaDecoderLayer</span>(nn.Module):
    <span class="kw">def</span> forward(self, positions, hidden_states, forward_batch, residual):
        <span class="cm"># 一层 = 归一化 → 注意力 → 残差 → 归一化 → MLP → 残差</span>
        <span class="kw">if</span> residual <span class="kw">is</span> None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        <span class="kw">else</span>:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)   <span class="cm"># forward_batch 穿到注意力</span>
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        <span class="kw">return</span> hidden_states, residual

<span class="kw">class</span> <span class="st">LlamaForCausalLM</span>(nn.Module):
    <span class="kw">def</span> forward(self, input_ids, positions, forward_batch, ...):
        hidden_states = self.model(input_ids, positions, forward_batch)   <span class="cm"># 嵌入 → N 层 → 末端归一</span>
        <span class="kw">return</span> self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)   <span class="cm"># → logits</span></pre>
</div>

<p>读这段代码最该记住的是它的<strong>朴素</strong>：<span class="mono">LlamaForCausalLM.forward</span> 几乎什么都没做——把活儿交给
<span class="mono">self.model</span> 跑出 hidden，再交给 <span class="mono">logits_processor</span> 配合 <span class="mono">lm_head</span> 出 logits；
<span class="mono">LlamaDecoderLayer.forward</span> 也只是把归一化、注意力、MLP、残差按顺序串起来。没有 CUDA、没有 all-reduce、没有 KV 分页的影子，
那些复杂度全被关进了它调用的那些层里。这种"模型文件薄、底层库厚"的分工，正是为什么一个新开源模型发布当天，社区往往只要<strong>照葫芦画瓢写一份这样的文件</strong>
就能让 SGLang 跑起来；多模态（第 49 课）、MoE（第 34 课）这些更复杂的架构，也只是在<strong>同一套脚手架</strong>上多插几种层而已。</p>

<p>最后值得点一句"<strong>注册</strong>"这步。写完模型文件还要把类登记到模型注册表里（按架构名，比如 config 里 <span class="mono">architectures</span> 字段写着 <span class="mono">LlamaForCausalLM</span>），运行时据此从权重目录的 <span class="mono">config.json</span> 找到该用哪个类。这一步通常只是一行映射，却是"配置驱动、自动派发"的关键：用户启动服务时只给一个模型路径，TokenizerManager 与 ModelRunner（第 24 课）顺着 config 的架构名就能找到对应文件、实例化、加载权重、开跑。于是从"社区提交一个模型文件"到"线上能服务这个模型"，中间<strong>不需要改任何调度、批处理或内核代码</strong>——这条干净的边界，才是广泛模型支持真正的工程红利。也正因如此，当你下次想给 SGLang 适配一个新模型时，思路应该是"<strong>先去 models 目录找一个最像的现有文件照着改</strong>"，而不是从零设计——绝大多数时候，你要做的只是调整层的种类与维度、改对 load_weights 的名字映射、再注册一下而已。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>模型文件 = 组装</strong>：SGLang 模型就是一组嵌套的 <span class="mono">nn.Module</span>，内部全用现成的<strong>并行层</strong>搭成，作者基本不写底层算子。</li>
    <li><strong>Llama 四（五）类套娃</strong>：<span class="mono">LlamaAttention</span> → <span class="mono">LlamaMLP</span> → <span class="mono">LlamaDecoderLayer</span> → <span class="mono">LlamaModel</span> → <span class="mono">LlamaForCausalLM</span>。</li>
    <li><strong>你只写两样</strong>：架构（哪些层、维度、残差）+ <span class="mono">load_weights</span> 名字映射（第 25 课）；注意力、TP 通信、KV 管理由 SGLang 白给。</li>
    <li><strong>forward_batch 是那道缝</strong>：它一路穿到每层注意力，告诉模型该读写 KV 池哪些槽位（第 24/33 课）——模型与运行时的边界。</li>
    <li><strong>所以加模型便宜</strong>：新开源模型常能 day-0 支持，写一份文件、注册即服务；MoE/多模态只是同套脚手架上多插几种层（第 34/49 课）。</li>
  </ul>
</div>
""",
             "en": r"""
<p class="lead">
Last lesson covered how weights <strong>get in</strong> (load_weights, Lesson 25). This one answers a more intriguing question: in SGLang,
what does it actually take to <strong>"write a model"</strong>? The answer is surprisingly light — a model file is essentially just
<strong>snapping together</strong> the <strong>parallel layers</strong> SGLang already ships, like LEGO. You write no attention kernel, no cross-GPU
communication, no KV management — only the <strong>architecture</strong>. That is the plain engineering truth behind SGLang's
"day-0 support for new models, very broad coverage."
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of writing a model as <strong>building with standard LEGO bricks</strong>: SGLang has already <strong>molded every brick</strong> —
  parallel linear layers, attention, norm, rotary embedding — all ready-made, tuned, and cross-GPU capable. You do just two things:
  ① snap the bricks <strong>into the right shape</strong> (the architecture: which layers, what dims, how residuals connect); and
  ② <strong>label</strong> each part — "which box on disk supplies the part that goes here" (load_weights name mapping). You <strong>never mold a brick yourself</strong>:
  how attention runs on the GPU, how 8 cards stitch results, which KV-pool page to write — all sealed inside the brick, none of your concern.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  A SGLang model file is a set of nested PyTorch <span class="mono">nn.Module</span>s. Take Llama — four classes <strong>nest like dolls</strong>:
  <span class="mono">LlamaAttention</span> → <span class="mono">LlamaMLP</span> → both compose <span class="mono">LlamaDecoderLayer</span> (one decoder layer) →
  N layers + embed + final norm compose <span class="mono">LlamaModel</span> → add <span class="mono">lm_head</span> and <span class="mono">forward</span> and you get
  <span class="mono">LlamaForCausalLM</span> (the model the runtime sees). Each class internally uses SGLang's <strong>reusable layers</strong>. So "adding a model"
  is, in the vast majority of cases, <strong>one act of assembly</strong> — write such a file, register it, and it's served.
</div>

<h2>Four classes nesting: how Llama is built</h2>
<p>Pull these four layers apart and you'll find each <strong>does just one thing</strong>, and every piece it uses comes from SGLang, not from your hand.
Innermost is <span class="mono">LlamaAttention</span>: it projects q/k/v in one shot via a <strong>column-parallel</strong> <span class="mono">QKVParallelLinear</span>
(the three are packed in one weight — exactly why load_weights does fusion mapping, Lesson 25), applies <strong>RoPE</strong> to q/k, then hands q and the
historical k/v to <span class="mono">RadixAttention</span> (the attention layer) — this step <strong>reads/writes the KV pool</strong>, with the attention backend
and which pool slots all dictated by the incoming <span class="mono">forward_batch</span> (Lesson 33); finally o_proj projects multi-head output back to
hidden dim via the <strong>row-parallel</strong> <span class="mono">RowParallelLinear</span>. Note: q/k/v, gate/up go <strong>column-parallel</strong> (split on output dim),
o_proj/down_proj go <strong>row-parallel</strong> (split on input dim then all-reduce); Lesson 46 details this slicing, but the model author <strong>only picks which
layer to use — slicing and communication happen inside the layer automatically</strong>.</p>
<p>One layer out is <span class="mono">LlamaMLP</span>: gate/up projection + <strong>SiLU</strong> (<span class="mono">SiluAndMul</span>) + down projection, nothing fancy.
Outer still, <span class="mono">LlamaDecoderLayer</span> stitches attention and MLP into one standard Transformer block; the key is the <strong>residual structure</strong>:
input_norm → attention → residual add → post_norm → MLP → residual add. <span class="mono">LlamaModel</span> is the skeleton: <span class="mono">embed_tokens</span>
(vocab-parallel embedding, <span class="mono">VocabParallelEmbedding</span>) → N decoder layers → final <span class="mono">RMSNorm</span>. The outermost
<span class="mono">LlamaForCausalLM</span> is the "whole model" exposed to the runtime: it holds <span class="mono">model</span>, an <span class="mono">lm_head</span>
(<span class="mono">ParallelLMHead</span>, weight optionally shared with embed via tie_word_embeddings), a <span class="mono">forward(input_ids, positions,
forward_batch)</span> returning logits, and <span class="mono">load_weights</span> (Lesson 25). ModelRunner (Lesson 24) holds exactly this outermost class.</p>

<p>This "nesting" isn't for looks; it's the natural result of PyTorch's <span class="mono">nn.Module</span> composition: a parent module hangs child modules as ordinary attributes in <span class="mono">__init__</span> and calls them in order in <span class="mono">forward</span>, so "register params, move to GPU, save / load" are done <strong>recursively and automatically</strong> by PyTorch along this module tree. That's also why Lesson 25's <span class="mono">load_weights</span> can pinpoint each weight by a <strong>dotted path</strong> like <span class="mono">model.layers.0.self_attn.qkv_proj</span> — the path is the nesting hierarchy, and the name maps naturally to a position in the module tree. Grasp this and you can tell at a glance "who nests whom" in any SGLang model file, and thus which submodule a given weight belongs in.</p>

<p>Look at one detail in <span class="mono">LlamaAttention</span> that shows the power of "assembly": it handles <strong>grouped-query attention</strong> (GQA) — many query heads, few key-value heads. The code divides total heads and total KV heads across cards by <span class="mono">tp_size</span>, and when KV heads are fewer than cards it <strong>replicates</strong> rather than splits. This boundary logic looks fiddly, but it only decides "how many heads each card gets and what dim," and the resulting <span class="mono">num_heads</span>, <span class="mono">head_dim</span> are handed as-is to <span class="mono">QKVParallelLinear</span> and <span class="mono">RadixAttention</span>. So even a feature like GQA that seems to touch the attention implementation is, in the model file, just <strong>a few lines of dimension arithmetic</strong> — the actual attention compute still reuses that one library op untouched.</p>

<p>The same "assembly" idea runs through <span class="mono">LlamaMLP</span>: the gate and up projections are merged into one <span class="mono">MergedColumnParallelLinear</span> (two separate weights on disk, stitched into one by load_weights at load time, Lesson 25), and <span class="mono">SiluAndMul</span> fuses "SiLU on gate, then element-wise multiply with up" into one op. You'll notice a recurring pattern: <strong>whenever adjacent operations can be fused, the library has pre-packaged them</strong> — QKV into one, gate/up into one, norm with residual into one. The author only needs to know "which packaged layer to use" to automatically enjoy these op-level optimizations, without understanding why they're faster. This "fast by default" design lets even a plain model file run at close to hand-optimized performance.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">entry</span><span class="name">LlamaForCausalLM</span></div><div class="ld">The model the runtime sees: holds <span class="mono">model</span> + <span class="mono">lm_head</span>, offers <span class="mono">forward(...)→logits</span> and <span class="mono">load_weights</span> (Lesson 25).</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">skeleton</span><span class="name">LlamaModel</span></div><div class="ld"><span class="mono">embed_tokens</span> (vocab-parallel) → N× decoder layers → final <span class="mono">RMSNorm</span>.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">× N</span><span class="name">LlamaDecoderLayer</span></div><div class="ld">One standard block: input_norm → attention → residual → post_norm → MLP → residual.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">sublayer</span><span class="name">LlamaAttention</span></div><div class="ld">QKV column-parallel proj + RoPE + <span class="mono">RadixAttention</span> (reads/writes KV pool, Lesson 33) + o_proj row-parallel.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">sublayer</span><span class="name">LlamaMLP</span></div><div class="ld">gate/up column-parallel + <span class="mono">SiLU</span> + down row-parallel.</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">exit</span><span class="name">lm_head</span></div><div class="ld">Projects final hidden to vocab dim to get <strong>logits</strong> (weight may be shared with embed).</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 340" role="img" aria-label="A decoder model's layer stack: embed_tokens through N LlamaDecoderLayers (each with self_attn and MLP, plus RMSNorm and residual), then a final norm, lm_head, and finally logits">
    <rect x="255" y="20" width="250" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="380" y="40" text-anchor="middle" class="mono" style="font-size:12px">embed_tokens</text>
    <line x1="380" y1="50" x2="380" y2="68" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,72 375,64 385,64" style="fill:var(--muted)"/>
    <rect x="170" y="72" width="420" height="142" rx="8" style="fill:var(--panel-2);stroke:var(--accent);stroke-width:1.5"/>
    <text x="186" y="92" style="font-weight:700;fill:var(--accent-ink);font-size:13px">N × LlamaDecoderLayer</text>
    <rect x="520" y="78" width="60" height="22" rx="11" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="550" y="93" text-anchor="middle" class="mono" style="font-size:11px">× 32</text>
    <rect x="190" y="104" width="180" height="48" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="280" y="124" text-anchor="middle" style="font-size:12px">RMSNorm + self_attn</text>
    <text x="280" y="142" text-anchor="middle" style="fill:var(--muted);font-size:11px">attention sublayer</text>
    <rect x="394" y="104" width="176" height="48" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="482" y="124" text-anchor="middle" style="font-size:12px">RMSNorm + MLP</text>
    <text x="482" y="142" text-anchor="middle" style="fill:var(--muted);font-size:11px">feed-forward sublayer</text>
    <text x="380" y="182" text-anchor="middle" style="fill:var(--purple);font-weight:700;font-size:12px">+ residual × 2, passed down layer by layer</text>
    <text x="380" y="202" text-anchor="middle" style="fill:var(--faint);font-size:11px">Llama-2-7B: 32 layers · hidden 4096</text>
    <line x1="380" y1="214" x2="380" y2="232" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,236 375,228 385,228" style="fill:var(--muted)"/>
    <rect x="265" y="236" width="230" height="28" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="380" y="255" text-anchor="middle" class="mono" style="font-size:12px">norm (RMSNorm)</text>
    <line x1="380" y1="264" x2="380" y2="280" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,284 375,276 385,276" style="fill:var(--muted)"/>
    <rect x="265" y="284" width="230" height="28" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="380" y="303" text-anchor="middle" class="mono" style="font-size:12px">lm_head</text>
    <line x1="380" y1="312" x2="380" y2="326" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="380,330 375,322 385,322" style="fill:var(--muted)"/>
    <text x="430" y="334" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">→ logits</text>
  </svg>
  <div class="figcap"><b>Fig 1 · A decoder model's layer stack</b> — a SGLang model just stacks <span class="mono">embed_tokens</span> → N×<span class="mono">LlamaDecoderLayer</span> (each = self_attn + MLP, with RMSNorm and residual) → final <span class="mono">norm</span> → <span class="mono">lm_head</span> vertically. Llama-2-7B is exactly such a stack of 32 layers with hidden 4096.</div>
</div>

<h2>What happens in one layer: follow the residual</h2>
<p>Worth stressing: <strong>the choice of layer itself encodes the parallel strategy</strong>. When the author writes the q/k/v projection as <span class="mono">QKVParallelLinear</span> and o_proj as <span class="mono">RowParallelLinear</span>, they write no line of communication code, yet have already decided: under tensor parallelism this layer's weight is split on the output dim across cards, each card computes a slice, then one all-reduce after o_proj stitches results back (Lesson 46). In other words, "which parallel layer you pick" is "how this layer distributes across GPUs." Likewise <span class="mono">VocabParallelEmbedding</span> splits the vocab dim and <span class="mono">ParallelLMHead</span> splits the output projection. The author only worries about "are the dims right, do the slicing schemes match" — actual slicing and collective comms happen silently inside the layer, which is exactly why a model file stays short.</p>
<p>Zoom into a <strong>single</strong> decoder layer and you see what "assembly" actually assembles. Data arrives carrying the previous layer's residual, first goes
through <span class="mono">input_layernorm</span> (RMSNorm, Lesson 36); the normalized vector enters the attention sublayer — project q/k/v, spin RoPE onto q/k,
<strong>write</strong> the current k/v into the KV pool at slots specified by <span class="mono">forward_batch</span>, then let q attend over itself and all historical k/v
(if the prefix was cached by RadixAttention, reuse it directly, Lesson 7); attention output goes through o_proj back to hidden dim and is <strong>added to the residual</strong>.
Then <span class="mono">post_attention_layernorm</span> normalizes again, enters the MLP, another residual add. Six steps, layer done, handing
<span class="mono">(hidden_states, residual)</span> to the next layer. After N layers, final norm, <span class="mono">lm_head</span> projects out logits.
Notice how <span class="mono">forward_batch</span> <strong>threads all the way</strong> into each layer's attention — it is <strong>the seam</strong> between the model file and the runtime:
the model only "computes tensors right," while "which KV-pool slots to read/write" is told to it by the runtime via forward_batch.</p>

<p>One more easily-missed detail: <span class="mono">residual</span> is <strong>passed across layers</strong>. Note in the code that the first layer arrives with residual None, so it treats the input itself as the residual; afterward every layer keeps "the value before norm" as the residual to add later. SGLang's <span class="mono">RMSNorm</span> even <strong>fuses</strong> "normalize + residual-add" into one call — <span class="mono">input_layernorm(hidden, residual)</span> returns both the new hidden and the updated residual, saving a memory round-trip. "Welding" a common combo into a single efficient op is exactly the optimization the underlying library does for you: you just call it, never minding how it merges two steps internally. The author sees only the clean logic "norm then attn then residual," and neither sees nor needs to see the fused kernel.</p>

<p>While we're here, clear up an easily-confused point: in decode each step actually forwards just <strong>one</strong> new token, yet it must attend over <strong>all historical</strong> K/V; in prefill it forwards the <strong>whole</strong> prompt's many tokens at once. The same <span class="mono">forward</span> code runs both cases because <span class="mono">forward_mode</span> and attention metadata inside <span class="mono">forward_batch</span> absorb the difference for the model — you see no EXTEND / DECODE branching in the model file; that's all handled in the attention layer and ModelRunner (Lesson 24). This further confirms the lesson's through-line: the author describes the <strong>static architecture</strong>, while <strong>how it dynamically runs</strong> is taken over by the runtime and library layers.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>input_norm</h4><p>RMSNorm the input (Lesson 36), keep residual aside.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Attention (reads/writes KV)</h4><p>QKV proj + RoPE; write/read KV pool per <span class="mono">forward_batch</span>, q attends history (Lesson 33).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Residual add</h4><p>Attention output via o_proj back to hidden dim, add to residual.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>post_norm</h4><p>RMSNorm again, ready for MLP.</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>MLP</h4><p>gate/up + <span class="mono">SiLU</span> + down, extract nonlinear features.</p></div></div>
  <div class="step"><div class="num">6</div><div class="sc"><h4>Residual add → next layer</h4><p>Add to residual, hand off <span class="mono">(hidden, residual)</span>.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="LlamaAttention.forward delegates attention to self.attn: hidden_states through qkv_proj, split q/k/v, rotary_emb, into the highlighted self.attn (RadixAttention, the SGLang seam), then o_proj to output">
    <text x="20" y="32" style="font-weight:700;fill:var(--muted);font-size:13px">LlamaAttention.forward(positions, hidden_states, forward_batch)</text>
    <rect x="20" y="68" width="96" height="48" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="68" y="96" text-anchor="middle" class="mono" style="font-size:11px">hidden</text>
    <line x1="116" y1="92" x2="128" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="132,92 124,87 124,97" style="fill:var(--muted)"/>
    <rect x="130" y="68" width="96" height="48" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="178" y="96" text-anchor="middle" class="mono" style="font-size:11px">qkv_proj</text>
    <line x1="226" y1="92" x2="238" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="242,92 234,87 234,97" style="fill:var(--muted)"/>
    <rect x="240" y="68" width="96" height="48" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="288" y="90" text-anchor="middle" class="mono" style="font-size:11px">split</text>
    <text x="288" y="106" text-anchor="middle" class="mono" style="font-size:11px">q,k,v</text>
    <line x1="336" y1="92" x2="348" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="352,92 344,87 344,97" style="fill:var(--muted)"/>
    <rect x="350" y="68" width="96" height="48" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="398" y="96" text-anchor="middle" class="mono" style="font-size:11px">rotary_emb</text>
    <line x1="446" y1="92" x2="458" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="462,92 454,87 454,97" style="fill:var(--muted)"/>
    <rect x="460" y="58" width="100" height="68" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2.5"/>
    <text x="510" y="86" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:12px">self.attn</text>
    <text x="510" y="104" text-anchor="middle" class="mono" style="font-size:10px">RadixAttention</text>
    <line x1="560" y1="92" x2="572" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="576,92 568,87 568,97" style="fill:var(--muted)"/>
    <rect x="574" y="68" width="96" height="48" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="622" y="96" text-anchor="middle" class="mono" style="font-size:11px">o_proj</text>
    <line x1="670" y1="92" x2="682" y2="92" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="686,92 678,87 678,97" style="fill:var(--muted)"/>
    <rect x="684" y="68" width="96" height="48" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="732" y="96" text-anchor="middle" class="mono" style="font-size:11px">output</text>
    <line x1="510" y1="126" x2="510" y2="158" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <polygon points="510,162 505,154 515,154" style="fill:var(--accent)"/>
    <rect x="360" y="162" width="300" height="64" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="510" y="186" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:12px">the SGLang seam</text>
    <text x="510" y="206" text-anchor="middle" style="fill:var(--ink);font-size:11px">KV cache + paging + attention backend plug in here</text>
    <text x="510" y="221" text-anchor="middle" style="fill:var(--muted);font-size:11px">driven by forward_batch (Lesson 33)</text>
  </svg>
  <div class="figcap"><b>Fig 2 · forward() delegates attention to self.attn</b> — <span class="mono">LlamaAttention.forward</span> itself only does the "line up the tensors" work — projection, split, RoPE — and hands the real attention compute to the highlighted <span class="mono">self.attn</span> (<span class="mono">RadixAttention</span>). That is the <strong>seam</strong> where the KV cache and attention backend plug in, driven by <span class="mono">forward_batch</span>.</div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/models/llama.py ::LlamaAttention</span><span class="ln">the attention layer delegates the op to self.attn (RadixAttention)</span></div>
  <pre><span class="kw">class</span> <span class="st">LlamaAttention</span>(nn.Module):
    <span class="kw">def</span> __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)   <span class="cm"># fused q,k,v projection</span>
        self.o_proj   = RowParallelLinear(...)
        self.rotary_emb = get_rope(...)
        self.attn = RadixAttention(...)          <span class="cm"># &lt;- the SGLang seam</span>
    <span class="kw">def</span> forward(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)  <span class="cm"># KV cache + backend</span>
        output, _ = self.o_proj(attn_output)
        <span class="kw">return</span> output</pre>
</div>

<p>Line this code up with the figure above and you'll see that the whole <span class="mono">LlamaAttention</span> contains <strong>not one line of attention math</strong>: it only projects, splits, and spins RoPE, then hands q/k/v as-is to <span class="mono">self.attn</span>. The most telling concrete example: <strong>just writing the line <span class="mono">self.attn = RadixAttention(...)</span> into a new model</strong> is all it takes to inherit prefix caching (Lesson 7), paged KV, and every attention backend (Lesson 33) — you never touch a line of cache or kernel code. For a feel of the scale, some real numbers: <span class="mono">Llama-2-7B</span> is exactly this attention + MLP packed into <strong>32</strong> <span class="mono">LlamaDecoderLayer</span>s with hidden dim <strong>4096</strong>, stacked vertically — the same <span class="mono">LlamaAttention</span> code is instantiated 32 times, differing only by <span class="mono">layer_id</span> and slicing, while the op is always that one from the library.</p>

<h2>Who writes, who provides: one table makes the cost clear</h2>
<p>Back to the core claim: why is "adding a model" so cheap in SGLang? Read the table below — the responsibility of each of the four (five) classes is,
<strong>without exception</strong>, "select and connect" SGLang's ready-made layers; not one line implements a low-level op. Read it together with the right-hand
"what you write vs what SGLang provides" and it clicks: what a model author truly writes anew is only two kinds of thing — the <strong>architecture</strong>
(which layers, what dims, how residuals connect) and the <strong>load_weights name mapping</strong> (which HF weight maps to which internal param, whether to fuse).
Everything else — attention kernels, TP communication, KV management, CUDA graphs — comes <strong>for free</strong>. That is the root cause of cheap "broad model support."</p>

<table class="t">
  <tr><th>Class</th><th>Responsibility (all "assembling ready layers")</th><th>SGLang layers used</th></tr>
  <tr><td class="mono">LlamaAttention</td><td>q/k/v proj + RoPE + attention + output proj</td><td class="mono">QKVParallelLinear / RadixAttention / RowParallelLinear</td></tr>
  <tr><td class="mono">LlamaMLP</td><td>gate/up + activation + down</td><td class="mono">MergedColumnParallelLinear / SiluAndMul</td></tr>
  <tr><td class="mono">LlamaDecoderLayer</td><td>stitch attention + MLP, define residual structure</td><td class="mono">RMSNorm (×2)</td></tr>
  <tr><td class="mono">LlamaModel</td><td>embed → N layers → final norm</td><td class="mono">VocabParallelEmbedding / RMSNorm</td></tr>
  <tr><td class="mono">LlamaForCausalLM</td><td>whole model: forward→logits + load_weights</td><td class="mono">ParallelLMHead / LogitsProcessor</td></tr>
</table>

<p>Read this table one level deeper and it lands harder: every layer with "Parallel" in its name stands on a whole tensor-parallel slicing-and-communication implementation; behind <span class="mono">RadixAttention</span> sit prefix-cache reuse (Lesson 7) + paged KV + multiple hardware attention backends (Lesson 33); behind <span class="mono">RMSNorm</span> and RoPE sit hand-written fused kernels (Lesson 36). Each of these alone is a long story, yet to the model author they are just <strong>one line of construction, one call</strong>. That's the power of "complexity sinks, interface floats": push the hard things down into reusable layers, and all that's left on the surface of a model file is "what this model looks like."</p>

<div class="cols">
  <div class="col"><h4>What you write (the model file)</h4><p><strong>Architecture</strong>: which layers, the dims, residual structure, how N layers stack; <strong>load_weights</strong>: HF weight name → internal param mapping and q/k/v, gate/up fusion (Lesson 25). That's all — one file, a few hundred lines.</p></div>
  <div class="col"><h4>What SGLang provides (for free)</h4><p>Parallel linear layers (column/row-parallel, auto slicing + comms, Lesson 46), the <strong>attention backend</strong> and KV management (Lesson 33), RoPE/RMSNorm (Lesson 36), CUDA graphs (Lesson 27), sampling (Lesson 28). You <strong>touch none of it</strong>.</p></div>
</div>

<p>Flip it around and this division also <strong>constrains</strong> the author: the architectures you can express are bounded by the layers SGLang already provides. If a new model uses some brand-new attention variant or activation that the library lacks, you must <strong>first add a new brick to the library</strong> (a new reusable layer), then assemble it in the model file. But in reality, mainstream open models are highly homogeneous in their building blocks — all permutations of some linear projection, some attention, some normalization — so for the vast majority of new models the library's bricks are <strong>already complete</strong>, and the author truly just assembles. That is the engineering premise that makes "day-0 support" possible, and why SGLang's model directory can grow comfortably to hundreds of files without losing control.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/models/llama.py ::LlamaForCausalLM</span><span class="ln">assembly is the model</span></div>
  <pre><span class="kw">class</span> <span class="st">LlamaDecoderLayer</span>(nn.Module):
    <span class="kw">def</span> forward(self, positions, hidden_states, forward_batch, residual):
        <span class="cm"># one layer = norm -> attention -> residual -> norm -> MLP -> residual</span>
        <span class="kw">if</span> residual <span class="kw">is</span> None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        <span class="kw">else</span>:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)   <span class="cm"># forward_batch threads to attention</span>
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        <span class="kw">return</span> hidden_states, residual

<span class="kw">class</span> <span class="st">LlamaForCausalLM</span>(nn.Module):
    <span class="kw">def</span> forward(self, input_ids, positions, forward_batch, ...):
        hidden_states = self.model(input_ids, positions, forward_batch)   <span class="cm"># embed -> N layers -> final norm</span>
        <span class="kw">return</span> self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)   <span class="cm"># -> logits</span></pre>
</div>

<p>The thing to remember about this code is its <strong>plainness</strong>: <span class="mono">LlamaForCausalLM.forward</span> does almost nothing — hands work to
<span class="mono">self.model</span> for hidden, then to <span class="mono">logits_processor</span> with <span class="mono">lm_head</span> for logits;
<span class="mono">LlamaDecoderLayer.forward</span> just strings norm, attention, MLP, residual in order. No CUDA, no all-reduce, no trace of KV paging —
all that complexity is sealed inside the layers it calls. This "thin model file, thick underlying library" split is exactly why, the day a new open model drops,
the community often only needs to <strong>write such a file by the same template</strong> to get SGLang running it; more complex architectures like multimodal (Lesson 49)
and MoE (Lesson 34) just plug a few extra layers into the <strong>same scaffolding</strong>.</p>

<p>One last word on the <strong>registration</strong> step. After writing the model file you register the class in the model registry (by architecture name — e.g. config's <span class="mono">architectures</span> field says <span class="mono">LlamaForCausalLM</span>), and the runtime uses that to find which class to use from the weight directory's <span class="mono">config.json</span>. This step is usually one line of mapping, yet it's the key to "config-driven, automatic dispatch": the user starts the server with just a model path, and TokenizerManager and ModelRunner (Lesson 24) follow config's architecture name to locate the file, instantiate it, load weights, and run. So from "the community submits a model file" to "this model is served in production," <strong>no scheduling, batching, or kernel code needs to change</strong> — that clean boundary is the real engineering dividend of broad model support. That's also why, the next time you want to bring up a new model in SGLang, the mindset should be "<strong>go to the models directory, find the closest existing file, and adapt it</strong>" rather than designing from scratch — in the vast majority of cases, all you do is adjust the kinds and dims of layers, get the load_weights name mapping right, and register it.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Model file = assembly</strong>: a SGLang model is nested <span class="mono">nn.Module</span>s built entirely from ready-made <strong>parallel layers</strong>; the author barely writes low-level ops.</li>
    <li><strong>Llama's four (five) nesting classes</strong>: <span class="mono">LlamaAttention</span> → <span class="mono">LlamaMLP</span> → <span class="mono">LlamaDecoderLayer</span> → <span class="mono">LlamaModel</span> → <span class="mono">LlamaForCausalLM</span>.</li>
    <li><strong>You write only two things</strong>: architecture (which layers, dims, residuals) + <span class="mono">load_weights</span> name mapping (Lesson 25); attention, TP comms, KV mgmt come free.</li>
    <li><strong>forward_batch is the seam</strong>: it threads into every layer's attention, telling the model which KV-pool slots to read/write (Lessons 24/33) — the model/runtime boundary.</li>
    <li><strong>So adding a model is cheap</strong>: new open models often get day-0 support — write one file, register, served; MoE/multimodal just plug extra layers into the same scaffolding (Lessons 34/49).</li>
  </ul>
</div>
"""}

LESSON_27 = {"zh": r"""
<p class="lead">
上一课的 ModelRunner 在解码时有一条"快路径"：能用 CUDA Graph 就<strong>重放</strong>一张录好的图，否则才即时（eager）地一层层算。这一课就把那条快路径拆开看：
一次解码前向其实要在 GPU 上<strong>发起几百个小内核</strong>，而每发起一个，CPU 都要付一笔固定的"启动费"。当每个 token 本身算得很少时，<strong>这笔启动费会反过来主导整步耗时</strong>——
GPU 算完一个内核就干等着，等 CPU 把下一个内核递上去。CUDA Graph 要解决的，正是这个"<strong>per-step 内核启动开销</strong>"。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把一次前向想成弹一首固定的曲子。<strong>没有图</strong>时，就像钢琴家<strong>现场一个键一个键地按</strong>（每按一下=启动一个内核）——手再快，每个音符之间也有"抬手再落下"的固定延迟。
  CUDA Graph 像一卷<strong>自动钢琴的打孔纸卷（player-piano roll）/ 录好的宏</strong>：你<strong>只把整首曲子录一次</strong>，之后让纸卷<strong>一口气自动播放</strong>，中间不再需要人逐个按键。
  但纸卷是<strong>定死的</strong>（静态形状）：孔位、长度都固定，所以你得<strong>按常见曲长各留一卷</strong>（batch-size 分桶），短曲子就<strong>补空拍 padding</strong> 到最接近的那一卷长度去播放。一句话：<strong>录一次、放多次，灵活性换来了速度</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  CUDA Graph 把<strong>整条前向的内核序列及其依赖</strong>录成<strong>一张图</strong>，之后<strong>一次提交、整体重放</strong>，把成百上千次"CPU 逐个发起内核"压成"一次提交"，从此 CPU 不再是解码步里的瓶颈。
  SGLang 在<strong>启动时</strong>就为<strong>一组 batch 尺寸</strong>（如 1、2、4、8…直到 max）各录一张图（<span class="mono">BaseCudaGraphRunner.capture</span>）；运行时把真实 batch
  <strong>向上 padding 到最近的已录桶</strong>，再重放那张图（<span class="mono">can_run_graph</span> + 执行，第 24 课）。因为解码形状规整、预填充长度多变，所以<strong>解码走图、预填充多走即时</strong>。
</div>

<h2>问题：一次解码前向，要发起几百个小内核</h2>
<p>先把"开销"具体化。一层 Transformer 解码层并不是"一个大内核"，而是一连串小算子：输入归一化、Q/K/V 三个投影、把 K/V 写进缓存、注意力、输出投影 o_proj、又一次归一化、MLP 的 gate/up、激活、down……
几十层叠下来，一次前向就是<strong>几百个内核排队上 GPU</strong>。每发起一个内核，CPU 都要做一套固定动作：准备参数、配置启动、塞进 CUDA 流——这套动作的耗时<strong>和内核算多大无关</strong>，是一笔<strong>固定的启动费</strong>。</p>
<p>这笔费用在<strong>预填充</strong>里不显眼：一次要算几百上千个 token，每个内核本身就很重，启动费相对算量小到可以忽略。可到了<strong>解码</strong>（第 4 课），每条请求每步只算 <strong>1 个新 token</strong>，矩阵又瘦又小，<strong>每个内核都极短</strong>——
这时候"发起内核"的固定开销反而成了大头。最坏的情形是：GPU 把一个小内核几微秒就算完了，却要<strong>停下来等 CPU</strong> 把下一个内核排好递过来。算力明明闲着，整步时间却被一串"等 CPU"拖长。这就是 CUDA Graph 要消灭的敌人。</p>
<p>换个角度量化一下这件事的严重性。假设一个模型有 32 层，每层粗算十来个小算子，一次前向就是<strong>三四百个内核</strong>；若每个内核的发起费是几微秒，单这一项累计起来就有上百微秒甚至毫秒级。而解码时单个内核的<strong>真实计算</strong>可能也只有几微秒——于是你会看到一条荒诞却真实的曲线：<strong>GPU 实际算的时间，可能还不如它等 CPU 发起内核的时间长</strong>。
这也解释了一个常见的困惑：为什么有人把 batch 调小、把模型换小，吞吐不升反降？因为模型越小、batch 越小，<strong>每个内核越短</strong>，启动开销占比就越高，瓶颈彻底从"算力"挪到了"发起内核"这件纯 CPU 的杂事上。理解了这一点，你就明白 CUDA Graph 不是锦上添花的小优化，而是把解码从"被 CPU 拖着走"里解放出来的<strong>关键一招</strong>。</p>

<div class="cols">
  <div class="col"><h4>❌ 无图（eager）：CPU 逐个发起</h4><p>CPU 发起内核①→GPU 算①→CPU 发起内核②→GPU 算②……几百次往返。每次发起都有<strong>固定启动费</strong>；解码内核太短，GPU 常<strong>干等 CPU</strong>。启动开销<strong>主导</strong>整步，GPU 利用率低。</p></div>
  <div class="col"><h4>✅ 有图（replay）：一次提交</h4><p>启动时把整条前向<strong>录成一张图</strong>；运行时 CPU <strong>只提交一次</strong>，GPU 顺着图把几百个内核<strong>连续跑完</strong>，中间不再回头等 CPU。启动开销几乎<strong>归零</strong>，GPU 被<strong>喂满</strong>。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="每步启动开销有图与无图对比：无图时每个解码步是 GPU 内核之间夹着许多 CPU 发起内核的小间隙，启动开销主导；有图时一次提交重放整条序列，间隙消失，GPU 背靠背连续跑">
    <text x="16" y="32" style="font-weight:700;fill:var(--red)">无图（eager）：每步夹满 CPU 发起间隙</text>
    <text x="16" y="64" style="fill:var(--muted);font-size:12px">GPU 时间线（一个解码步）</text>
    <rect x="56" y="72" width="700" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="64" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="98" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="128" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="162" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="196" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="230" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="264" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="298" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="332" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="366" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="400" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="434" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="476" y="97" style="fill:var(--faint);font-size:12px">… 几百个内核 × 间隙，启动开销主导整步</text>
    <rect x="64" y="124" width="14" height="14" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="86" y="135" style="fill:var(--blue);font-size:11px">内核（算）</text>
    <rect x="180" y="124" width="14" height="14" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="202" y="135" style="fill:var(--amber);font-size:11px">等 CPU 发起（GPU 干等）</text>
    <text x="16" y="190" style="font-weight:700;fill:var(--teal)">有图（replay）：一次提交，间隙消失</text>
    <text x="16" y="222" style="fill:var(--muted);font-size:12px">GPU 时间线（一个解码步）</text>
    <rect x="56" y="230" width="700" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="64" y="236" width="40" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="84" y="255" text-anchor="middle" class="mono" style="font-size:10px">提交</text>
    <rect x="108" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="184" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="260" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="336" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="412" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="488" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="580" y="254" style="fill:var(--teal);font-size:12px">→ 内核背靠背连续跑，无间隙</text>
  </svg>
  <div class="figcap"><b>图 1 · 每步启动开销：有图 vs 无图</b> — <strong>无图</strong>时每个解码步在 GPU 内核之间夹着许多“<strong>等 CPU 发起</strong>”的小间隙，小 batch 下启动开销主导整步；<strong>有图</strong>时只需<strong>一次提交</strong>就重放整条录好的序列，间隙消失、GPU 背靠背跑满。</div>
</div>

<h2>解法：录一次，重放整条前向</h2>
<p>CUDA Graph 的核心动作只有两步：<strong>捕获（capture）</strong>与<strong>重放（replay）</strong>。捕获时，让模型<strong>空跑一遍</strong>前向，CUDA 在底层把这一遍<strong>所有内核的调用顺序和彼此依赖</strong>记成一张有向图，但<strong>并不真正执行计算</strong>；
之后重放时，只需把这张图作为<strong>单个提交</strong>交给 GPU，硬件就按图把整串内核<strong>一次跑完</strong>，CPU 完全不必再逐个发起。一次提交替代几百次发起——这就是开销被抹平的地方。打个比方：捕获像是把一整套流水线动作<strong>录成一段视频</strong>，重放则是<strong>按下播放键</strong>，机器自己顺着录好的动作连贯做完，不必每一步都等人喊口令。</p>
<p>但图是<strong>录死的</strong>：它绑定了录制时的<strong>张量形状和显存地址</strong>。真实请求的 batch 大小每步都可能不同，不可能为每一个具体大小都录一张。SGLang 的做法是<strong>分桶 + padding</strong>：启动时只为<strong>几档固定 batch 尺寸</strong>（1、2、4、8…直到 max）各录一张图（<span class="mono">capture</span> 遍历这些尺寸逐个录）；
运行时拿到真实 batch，先<strong>向上取整到最近的那个桶</strong>（<span class="mono">_pad_to_bucket</span>），把张量 padding 到该尺寸，再重放对应的图。比如真实 batch=5，就 padding 到桶 8 的图去跑——多算 3 行空请求的代价，远小于省下的几百次内核启动。</p>
<p>这里有个容易混淆的点值得点破：<strong>捕获时并不是真的在算这一批用户请求</strong>。捕获阶段喂进去的是一批<strong>假数据（dummy）</strong>，目的只是让前向的<strong>每一个内核都被触发一次</strong>，好让 CUDA 把它们的调用顺序和依赖关系记下来；记的是"<strong>动作的剧本</strong>"，不是"<strong>具体的数值</strong>"。所以重放时换上真实数据，跑的还是同一套剧本、同一串内核、同一组显存地址，只是缓冲里的内容变了。
正因如此，运行器必须预先备好一组<strong>固定地址的输入/输出缓冲</strong>：每一步把真实的 input_ids、positions 等<strong>拷进</strong>这些固定缓冲，再触发重放，图就会从这些固定地址读数、往固定地址写结果。这也是为什么代码里 <span class="mono">buffers</span> 是和图绑在一起、由运行器统一持有的——它们是图能够反复复用的物理前提。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>启动时·捕获</h4><p>让模型空跑前向，CUDA 把整条内核序列与依赖<strong>录成一张图</strong>（不真算）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>为一组尺寸各录一张</h4><p>遍历 batch 尺寸桶 1/2/4/8/…/max，每档录一张图，连同<strong>静态缓冲</strong>一起留着。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>运行时·padding</h4><p>真实 batch 向上取整到<strong>最近的已录桶</strong>（<span class="mono">_pad_to_bucket</span>），张量补到该尺寸。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>重放</h4><p><span class="mono">can_run_graph</span> 通过 → <strong>一次提交</strong>重放那张图（第 24 课），取出真实部分的 logits。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 270" role="img" aria-label="捕获一次重放多次：左侧一次性捕获把固定 batch 形状的内核序列录成一张图对象，右侧在多个解码步里用新数据重放同一张录好的图">
    <text x="16" y="34" style="font-weight:700;fill:var(--amber)">捕获 · 一次性</text>
    <rect x="16" y="46" width="236" height="180" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="32" y="70" style="fill:var(--muted);font-size:12px">固定 batch 形状 · 空跑一遍前向</text>
    <rect x="32" y="84" width="44" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="54" y="104" text-anchor="middle" class="mono" style="font-size:11px">k1</text>
    <text x="82" y="104" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="92" y="84" width="44" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="114" y="104" text-anchor="middle" class="mono" style="font-size:11px">k2</text>
    <text x="142" y="104" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="152" y="84" width="44" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="174" y="104" text-anchor="middle" class="mono" style="font-size:11px">k3</text>
    <text x="202" y="104" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="212" y="84" width="28" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="226" y="104" text-anchor="middle" class="mono" style="font-size:11px">…</text>
    <text x="32" y="144" style="font-size:12px">记录全部内核的调用顺序 + 依赖</text>
    <rect x="32" y="156" width="204" height="54" rx="8" style="fill:var(--panel-2);stroke:var(--amber);stroke-width:1.5"/>
    <text x="134" y="180" text-anchor="middle" style="font-weight:700">几百个小内核 → 录成一张图</text>
    <text x="134" y="198" text-anchor="middle" style="fill:var(--muted);font-size:11px">不真正计算，只记“剧本”</text>
    <line x1="252" y1="130" x2="316" y2="130" style="stroke:var(--amber);stroke-width:1.5"/>
    <polygon points="324,130 310,123 310,137" style="fill:var(--amber)"/>
    <text x="284" y="120" text-anchor="middle" style="fill:var(--amber);font-size:12px">record</text>
    <rect x="326" y="92" width="132" height="76" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="392" y="124" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">图对象</text>
    <text x="392" y="146" text-anchor="middle" class="mono" style="font-size:11px">graph + 静态缓冲</text>
    <line x1="458" y1="130" x2="522" y2="130" style="stroke:var(--accent);stroke-width:1.5"/>
    <polygon points="530,130 516,123 516,137" style="fill:var(--accent)"/>
    <text x="492" y="120" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">replay × 多次</text>
    <text x="540" y="34" style="font-weight:700;fill:var(--teal)">重放 · 多次（每个解码步）</text>
    <rect x="540" y="46" width="244" height="34" rx="7" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="548" y="68" style="font-size:12px">解码步 1 · 一次提交重放（新数据 · 同形状）</text>
    <rect x="540" y="88" width="244" height="34" rx="7" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="548" y="110" style="font-size:12px">解码步 2 · 一次提交重放（新数据 · 同形状）</text>
    <rect x="540" y="130" width="244" height="34" rx="7" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="548" y="152" style="font-size:12px">解码步 3 · 一次提交重放（新数据 · 同形状）</text>
    <text x="540" y="190" style="fill:var(--muted);font-size:12px">… 同一张图被重放成百上千次，几乎零启动开销</text>
  </svg>
  <div class="figcap"><b>图 2 · 捕获一次 → 重放多次</b> — 左边<strong>只捕获一次</strong>：用固定 batch 形状空跑一遍前向，把几百个内核的调用顺序与依赖录成一张<strong>图对象</strong>（不真算）；右边在每个解码步都<strong>廉价地重放</strong>这张录好的图——形状不变、只换数据。</div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py ::BaseCudaGraphRunner</span><span class="ln">捕获与分桶 padding</span></div>
  <pre><span class="kw">class</span> BaseCudaGraphRunner(BaseRunner):
    <span class="cm"># 启动时把整条前向录成图；运行时按桶 padding 后重放</span>
    buffers: ForwardInputBuffers       <span class="cm"># 预分配的静态输入缓冲（地址固定）</span>
    backend: BaseCudaGraphBackend      <span class="cm"># 负责 capture / replay 机制</span>

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> _pad_to_bucket(raw_size, buckets):
        <span class="cm"># 把真实 batch 向上取整到最近的已录制桶</span>
        <span class="kw">assert</span> raw_size &lt;= buckets[-1]     <span class="cm"># 超过最大桶 → can_run_graph 应已拒绝</span>
        index = bisect.bisect_left(buckets, raw_size)
        <span class="kw">return</span> buckets[index]

    <span class="kw">@abstractmethod</span>
    <span class="kw">def</span> capture(self):                  <span class="cm"># 一次性：遍历所有 batch 尺寸逐个录图</span>
        ...                            <span class="cm"># 每个尺寸调 capture_one_shape 录一张图</span></pre>
</div>

<h2>约束：图为什么必须"静态"</h2>
<p>能享受重放的红利，是有代价的：图要求<strong>静态的形状与地址</strong>。这条硬约束直接决定了 SGLang 的三个工程选择。第一，<strong>形状必须固定</strong>，所以要<strong>分桶 + padding</strong>，不能让 batch 自由变化。
第二，<strong>地址必须固定</strong>，所以运行器预先开好一组<strong>静态输入/输出缓冲</strong>（上面代码里的 <span class="mono">buffers</span>），每步把真实数据<strong>拷进</strong>这些固定缓冲再重放，而不是每次新分配。
第三，<strong>捕获区域内不能有依赖数据的控制流</strong>——图录的是一条固定路径，不能"这一步走 if、下一步走 else"。所以有些算子只能<strong>留在图外</strong>，或者用"分段/可断开（piecewise）"的图把动态算子<strong>切出去</strong>、其余部分照录（第 33 课讲注意力后端时会细说）。</p>
<p>把这三条连起来，就彻底解释了上一课那句"<strong>解码走图、预填充多走即时</strong>"：解码每步形状高度规整（batch 在一段时间内不变、序列每步只长 1），天然能套进固定的桶；预填充长度千变万化、一次要算几百上千个 token，几乎每批形状都不同，硬要为每种形状录图既不划算也录不过来，于是它更愿意走灵活的即时路径。</p>
<p>第三条约束尤其值得多想一层。所谓"<strong>数据相关的控制流</strong>"，指的是<strong>要根据张量里的具体数值才能决定走哪条分支</strong>的逻辑——比如"如果这一步命中了某个缓存就走 A、否则走 B"。这种分支没法录进图，因为图是一条<strong>录死的直路</strong>，它不会在重放时临场判断。
解决办法有两种：要么把这类动态算子<strong>挪到图外</strong>，让图只覆盖那段"无论数据如何、动作都一样"的稳定区间；要么用<strong>分段 / 可断开（piecewise）</strong>的图，把整条前向<strong>在动态算子处切成几段</strong>，每段各录一张稳定的子图，中间留出口给动态逻辑插手。后者让更多算子也能享受重放，但实现更复杂——第 33 课讲注意力后端时会看到，不同后端对"哪些能录进图"给出的答案并不一样，这正是 SGLang 把图运行器和注意力后端分层设计的原因之一。</p>

<table class="t">
  <tr><th>约束</th><th>为什么</th><th>SGLang 的应对</th></tr>
  <tr><td>形状必须静态</td><td>图绑定录制时的张量形状</td><td>batch 分桶 + 向上 padding</td></tr>
  <tr><td class="mono">地址必须静态</td><td>图绑定录制时的显存指针</td><td>预分配静态缓冲，每步拷入</td></tr>
  <tr><td>无数据相关控制流</td><td>图是一条固定路径，不能 if/else</td><td>动态算子留图外 / 分段图（第 33 课）</td></tr>
  <tr><td>桶有上限 max</td><td>不能为任意大小录图</td><td>超过 max 的 batch 拒绝走图，转即时</td></tr>
</table>

<div class="cellgroup">
  <div class="cg-cap"><b>batch-size 分桶 + padding</b>：启动时只录这几档；真实 batch=5 时<strong>向上取整到桶 8</strong> 的图去重放（高亮档）</div>
  <div class="cells"><span class="lab">已录桶</span><span class="cell">1</span><span class="sep">·</span><span class="cell">2</span><span class="sep">·</span><span class="cell">4</span><span class="sep">·</span><span class="cell hl">8</span><span class="sep">·</span><span class="cell">16</span><span class="sep">·</span><span class="cell">…</span><span class="sep">·</span><span class="cell">max</span></div>
  <div class="cells"><span class="lab">真实=5</span><span class="cell">5 行真实</span><span class="sep">+</span><span class="cell">3 行 padding</span><span class="sep">→</span><span class="cell hl">套桶 8 的图</span><span class="sep">→</span><span class="cell q">重放后只取前 5 行 logits</span></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py ::BaseCudaGraphRunner._pad_to_bucket</span><span class="ln">把真实 batch 大小向上取整到最近的已捕获分桶</span></div>
  <pre><span class="kw">@staticmethod</span>
<span class="kw">def</span> _pad_to_bucket(raw_size, buckets):
    <span class="cm"># graphs are captured only for a FIXED set of batch sizes</span>
    <span class="cm"># (buckets). Round a real size UP to the nearest bucket so a</span>
    <span class="cm"># recorded graph can be replayed; can_run_graph must already</span>
    <span class="cm"># have rejected raw_size &gt; max(buckets).</span>
    index = bisect.bisect_left(buckets, raw_size)
    <span class="kw">return</span> buckets[index]   <span class="cm"># smallest bucket &gt;= raw_size</span></pre>
</div>

<p><strong>具体例子：</strong>设已捕获的桶为 <span class="mono">[1, 2, 4, 8, 16, …]</span>。来一个真实 batch=5：<span class="mono">bisect_left</span> 落在 <strong>8</strong> 上，于是 <strong>padding 到桶 8</strong>——多塞 <strong>3 行空请求</strong>去重放桶 8 的那张图，跑完只取前 5 行的 logits。注意这套机制只对<strong>解码</strong>奏效：解码每步形状固定（每条请求只算 <strong>1 个新 token</strong>），能稳稳落进某个桶；而<strong>预填充</strong>长度千变万化、几乎每批形状都不同，所以多走即时、不走图。</p>

<h2>回报：解码吞吐大涨，且与重叠调度天作之合</h2>
<p>消灭了每步几百次内核启动，<strong>解码吞吐通常有可观增益</strong>——尤其在小模型、短内核、大并发的场景，启动开销占比最高，收益也最明显。更妙的是它和<strong>重叠调度器（第 21 课）</strong>是<strong>天作之合</strong>：
图重放期间，GPU 只管闷头跑这一整张图、几乎不回头找 CPU；与此同时 CPU 正好腾出手去<strong>调度下一步</strong>（采样、组下一个 batch）。两者一叠加，<strong>GPU 在重放、CPU 在排下一步</strong>，GPU 几乎不留空隙——这正是高吞吐推理引擎的核心节奏。</p>
<p>当然也有成本，这是个真实的权衡：录很多档尺寸要花<strong>启动时间</strong>（每张图都要空跑一遍前向去录），也要吃<strong>显存</strong>（每张图都握着自己那份静态缓冲）。所以桶不是越多越好——太密则启动慢、占显存；太疏则 padding 浪费大。
这条"<strong>启动成本/显存 ↔ 运行时省下的启动开销</strong>"的权衡，会在后面反复出现：第 33 课的分段图让动态算子也能部分享受重放，第 43 课的投机解码则要为"草稿+验证"这种更复杂的形状专门考虑怎么录图。把这一课记牢：<strong>CUDA Graph = 把"逐个发起内核"换成"一次重放整条前向"，用静态形状换掉启动开销</strong>。</p>
<p>最后把这一课放回整条 GPU 侧流水线。第 24 课的 ModelRunner 是"决策→计算"的边界，它在解码时优先走的那条快路径，骨子里就是本课的图重放；第 25/26 课告诉你它手里的模型怎么加载、怎么写；而本课补上了"<strong>这台机器为什么跑得这么快</strong>"的答案——不是靠某个更聪明的算法，而是靠<strong>把 CPU 从"逐个发起内核"的苦役里彻底解放出来</strong>。
顺着这条线再往下，第 33 课会讲注意力后端如何决定"哪些能录进图"，第 43 课的投机解码会把图重放推到更复杂的形状上。带着"<strong>图 = 录一次、重放整条、用静态换启动开销</strong>"这把钥匙，你会发现后面很多看似零散的性能优化，其实都在围绕同一个主题打转：<strong>别让 GPU 闲着，也别让 CPU 拖后腿</strong>。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <strong>① 敌人是启动开销：</strong>一次解码前向要发起几百个小内核，每次发起有固定 CPU 启动费；解码内核太短，这笔费用反而主导整步、GPU 干等 CPU。
  <strong>② 解法是录一次重放整条：</strong>CUDA Graph 把整条前向的内核序列录成一张图，运行时一次提交整体重放，把几百次发起压成一次，CPU 不再逐个递内核。
  <strong>③ 代价是静态：</strong>图绑定形状与地址，故需 batch 分桶 + padding（<span class="mono">_pad_to_bucket</span>）、预分配静态缓冲、捕获区内禁止数据相关控制流——这也是<strong>解码走图、预填充走即时</strong>的根因。
  <strong>④ SGLang 怎么做：</strong>启动时 <span class="mono">BaseCudaGraphRunner.capture</span> 为一组尺寸（1/2/4/8/…/max）各录一张；运行时 padding 到最近桶后 <span class="mono">can_run_graph</span> + 重放（第 24 课）。
  <strong>⑤ 回报与搭档：</strong>解码吞吐大涨，并与重叠调度器（第 21 课）天作之合——GPU 重放、CPU 排下一步，GPU 几乎不空转；代价是启动时间与显存（分段图见第 33 课，投机解码见第 43 课）。
</div>
""", "en": r"""
<p class="lead">
Last lesson's ModelRunner had a decode "fast path": if a CUDA graph is available it <strong>replays</strong> a pre-recorded graph, otherwise it computes eagerly layer by layer. This lesson opens up that fast path.
One decode forward actually has to <strong>launch hundreds of tiny GPU kernels</strong>, and every launch costs the CPU a fixed "issue fee". When each token computes very little, <strong>that launch fee ends up dominating the step</strong> —
the GPU finishes one kernel and then just waits for the CPU to hand it the next. CUDA graphs exist to kill exactly this <strong>per-step kernel-launch overhead</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of a forward as playing one fixed song. <strong>Without a graph</strong>, it is like a pianist <strong>pressing each key live</strong> (each press = launching a kernel) — no matter how fast the hands, there's a fixed "lift and drop" delay between notes.
  A CUDA graph is like a <strong>player-piano roll / a recorded macro</strong>: you <strong>record the whole song once</strong>, then let the roll <strong>play it in one go</strong>, with no human pressing keys one by one.
  But the roll is <strong>fixed</strong> (static shapes): the holes and length are set, so you <strong>keep one roll per common song length</strong> (batch-size bucket) and <strong>pad shorter songs</strong> up to the nearest roll length to play them.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  A CUDA graph records the <strong>entire forward's kernel sequence and its dependencies</strong> into <strong>one graph</strong>, then <strong>replays it as a single submission</strong>, collapsing hundreds of "CPU launches a kernel" into "one submission".
  SGLang <strong>captures</strong> graphs <strong>at startup</strong> for a <strong>set of batch sizes</strong> (e.g. 1, 2, 4, 8, …, max) via <span class="mono">BaseCudaGraphRunner.capture</span>; at run time it <strong>pads the real batch up to the nearest captured bucket</strong>
  and replays that graph (<span class="mono">can_run_graph</span> + execute, Lesson 24). Because decode shapes are regular and prefill lengths vary, <strong>decode is graphed, prefill mostly runs eager</strong>.
</div>

<h2>The problem: one decode forward launches hundreds of tiny kernels</h2>
<p>Let's make "overhead" concrete. A Transformer decoder layer is not "one big kernel" but a chain of small ops: input norm, the Q/K/V projections, writing K/V into the cache, attention, the o_proj output projection, another norm, the MLP gate/up, activation, down…
Stack dozens of layers and one forward becomes <strong>hundreds of kernels queued onto the GPU</strong>. Each launch makes the CPU do a fixed routine: prepare args, configure the launch, enqueue into the CUDA stream — and that cost is <strong>independent of how big the kernel is</strong>; it is a <strong>fixed launch fee</strong>.</p>
<p>This fee is invisible in <strong>prefill</strong>: it computes hundreds-to-thousands of tokens at once, each kernel is heavy, so the launch fee is negligible against the compute. But in <strong>decode</strong> (Lesson 4), each request computes just <strong>1 new token</strong> per step, the matrices are thin and small, so <strong>every kernel is extremely short</strong> —
and now the fixed cost of "launching a kernel" becomes the dominant term. Worst case: the GPU finishes a tiny kernel in microseconds, then has to <strong>stop and wait for the CPU</strong> to enqueue the next one. The compute units sit idle while the step is stretched by a string of "wait for CPU". That is the enemy CUDA graphs eliminate.</p>

<div class="cols">
  <div class="col"><h4>❌ No graph (eager): launch one by one</h4><p>CPU launches kernel ①→GPU runs ①→CPU launches ②→GPU runs ②… hundreds of round-trips. Each launch has a <strong>fixed fee</strong>; decode kernels are so short the GPU often <strong>waits on the CPU</strong>. Launch overhead <strong>dominates</strong>, GPU utilization is low.</p></div>
  <div class="col"><h4>✅ Graph (replay): one submission</h4><p>At startup the whole forward is <strong>recorded into one graph</strong>; at run time the CPU <strong>submits just once</strong> and the GPU runs the hundreds of kernels <strong>back-to-back</strong> along the graph, never turning back to wait. Launch overhead is nearly <strong>zero</strong>, the GPU is <strong>saturated</strong>.</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Per-step launch overhead with vs without a graph: without a graph each decode step is many small CPU kernel-launch gaps between GPU kernels and launch overhead dominates; with a graph one submission replays the whole recorded sequence so gaps vanish and the GPU runs back-to-back">
    <text x="16" y="32" style="font-weight:700;fill:var(--red)">No graph (eager): each step is full of CPU launch gaps</text>
    <text x="16" y="64" style="fill:var(--muted);font-size:12px">GPU timeline (one decode step)</text>
    <rect x="56" y="72" width="700" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="64" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="98" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="128" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="162" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="196" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="230" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="264" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="298" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="332" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="366" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="400" y="78" width="34" height="28" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="434" y="78" width="30" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="476" y="97" style="fill:var(--faint);font-size:12px">… many kernels × gaps, overhead dominates</text>
    <rect x="64" y="124" width="14" height="14" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="86" y="135" style="fill:var(--blue);font-size:11px">kernel (compute)</text>
    <rect x="200" y="124" width="14" height="14" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="222" y="135" style="fill:var(--amber);font-size:11px">wait for CPU launch (GPU idles)</text>
    <text x="16" y="190" style="font-weight:700;fill:var(--teal)">With graph (replay): one submission, gaps gone</text>
    <text x="16" y="222" style="fill:var(--muted);font-size:12px">GPU timeline (one decode step)</text>
    <rect x="56" y="230" width="700" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="64" y="236" width="40" height="28" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="84" y="255" text-anchor="middle" class="mono" style="font-size:10px">submit</text>
    <rect x="108" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="184" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="260" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="336" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="412" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="488" y="236" width="76" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="580" y="254" style="fill:var(--teal);font-size:12px">→ kernels run back-to-back, no gaps</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Per-step launch overhead: with vs without a graph</b> — <strong>without a graph</strong> every decode step sandwiches many small “<strong>wait for CPU launch</strong>” gaps between GPU kernels, and at small batch the launch overhead dominates the step; <strong>with a graph</strong> a single <strong>submission</strong> replays the whole recorded sequence, the gaps vanish and the GPU runs back-to-back.</div>
</div>

<h2>The fix: record once, replay the whole forward</h2>
<p>A CUDA graph has just two moves: <strong>capture</strong> and <strong>replay</strong>. During capture, the model <strong>runs the forward once</strong> while CUDA records, underneath, <strong>the call order and dependencies of all kernels</strong> into a directed graph — <strong>without actually computing</strong>;
then at replay, you hand that graph to the GPU as a <strong>single submission</strong> and the hardware runs the whole kernel string <strong>in one shot</strong>, with the CPU never launching anything one by one. One submission replaces hundreds of launches — that is where the overhead vanishes.</p>
<p>But a graph is <strong>frozen</strong>: it binds the <strong>tensor shapes and memory addresses</strong> from capture time. Real requests have a batch size that may change every step, so you can't record one graph per exact size. SGLang's answer is <strong>bucketing + padding</strong>: at startup it records a graph for only <strong>a few fixed batch sizes</strong> (1, 2, 4, 8, …, max) — <span class="mono">capture</span> iterates over these sizes recording each;
at run time it takes the real batch, <strong>rounds it up to the nearest bucket</strong> (<span class="mono">_pad_to_bucket</span>), pads the tensors to that size, and replays the matching graph. E.g. a real batch of 5 pads up to the bucket-8 graph — the cost of 3 extra padded rows is far smaller than the hundreds of kernel launches it saves.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Startup · capture</h4><p>Run the forward once while CUDA <strong>records the whole kernel sequence and dependencies into a graph</strong> (no real compute).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>One graph per size</h4><p>Iterate batch buckets 1/2/4/8/…/max, record a graph for each, kept alongside its <strong>static buffers</strong>.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Run time · pad</h4><p>Round the real batch up to the <strong>nearest captured bucket</strong> (<span class="mono">_pad_to_bucket</span>); pad tensors to that size.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Replay</h4><p><span class="mono">can_run_graph</span> passes → <strong>one submission</strong> replays that graph (Lesson 24); slice out the real rows' logits.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 270" role="img" aria-label="Capture once, replay many: on the left a one-time capture records the kernel sequence for a fixed batch shape into a graph object; on the right many decode steps cheaply replay that recorded graph with new data and the same shape">
    <text x="16" y="34" style="font-weight:700;fill:var(--amber)">Capture · once</text>
    <rect x="16" y="46" width="236" height="180" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="32" y="70" style="fill:var(--muted);font-size:12px">fixed batch shape · run forward once</text>
    <rect x="32" y="84" width="44" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="54" y="104" text-anchor="middle" class="mono" style="font-size:11px">k1</text>
    <text x="82" y="104" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="92" y="84" width="44" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="114" y="104" text-anchor="middle" class="mono" style="font-size:11px">k2</text>
    <text x="142" y="104" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="152" y="84" width="44" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="174" y="104" text-anchor="middle" class="mono" style="font-size:11px">k3</text>
    <text x="202" y="104" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="212" y="84" width="28" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="226" y="104" text-anchor="middle" class="mono" style="font-size:11px">…</text>
    <text x="32" y="144" style="font-size:12px">record every kernel's call order + deps</text>
    <rect x="32" y="156" width="204" height="54" rx="8" style="fill:var(--panel-2);stroke:var(--amber);stroke-width:1.5"/>
    <text x="134" y="180" text-anchor="middle" style="font-weight:700">hundreds of kernels → one graph</text>
    <text x="134" y="198" text-anchor="middle" style="fill:var(--muted);font-size:11px">no real compute, only the “script”</text>
    <line x1="252" y1="130" x2="316" y2="130" style="stroke:var(--amber);stroke-width:1.5"/>
    <polygon points="324,130 310,123 310,137" style="fill:var(--amber)"/>
    <text x="284" y="120" text-anchor="middle" style="fill:var(--amber);font-size:12px">record</text>
    <rect x="326" y="92" width="132" height="76" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="392" y="124" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">Graph object</text>
    <text x="392" y="146" text-anchor="middle" class="mono" style="font-size:11px">graph + static buffers</text>
    <line x1="458" y1="130" x2="522" y2="130" style="stroke:var(--accent);stroke-width:1.5"/>
    <polygon points="530,130 516,123 516,137" style="fill:var(--accent)"/>
    <text x="492" y="120" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">replay × many</text>
    <text x="540" y="34" style="font-weight:700;font-size:13px;fill:var(--teal)">Replay · many (every decode step)</text>
    <rect x="540" y="46" width="244" height="34" rx="7" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="556" y="68" style="font-size:12px">decode step 1 · replay</text>
    <rect x="540" y="88" width="244" height="34" rx="7" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="556" y="110" style="font-size:12px">decode step 2 · replay</text>
    <rect x="540" y="130" width="244" height="34" rx="7" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="556" y="152" style="font-size:12px">decode step 3 · replay</text>
    <text x="540" y="190" style="fill:var(--muted);font-size:12px">… same graph replayed many times</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Capture once → replay many</b> — on the left you <strong>capture only once</strong>: run one forward at a fixed batch shape to record hundreds of kernels' call order and deps into a <strong>graph object</strong> (no real compute); on the right every decode step <strong>cheaply replays</strong> that recorded graph — same shape, just new data.</div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py ::BaseCudaGraphRunner</span><span class="ln">capture &amp; bucket padding</span></div>
  <pre><span class="kw">class</span> BaseCudaGraphRunner(BaseRunner):
    <span class="cm"># record the whole forward at startup; pad to a bucket and replay at run time</span>
    buffers: ForwardInputBuffers       <span class="cm"># pre-allocated static input buffers (fixed addrs)</span>
    backend: BaseCudaGraphBackend      <span class="cm"># owns the capture / replay mechanics</span>

    <span class="kw">@staticmethod</span>
    <span class="kw">def</span> _pad_to_bucket(raw_size, buckets):
        <span class="cm"># round the real batch up to the nearest captured bucket</span>
        <span class="kw">assert</span> raw_size &lt;= buckets[-1]     <span class="cm"># over max bucket -> can_run_graph should reject</span>
        index = bisect.bisect_left(buckets, raw_size)
        <span class="kw">return</span> buckets[index]

    <span class="kw">@abstractmethod</span>
    <span class="kw">def</span> capture(self):                  <span class="cm"># one-time: iterate all batch sizes, record a graph each</span>
        ...                            <span class="cm"># each size calls capture_one_shape to record one graph</span></pre>
</div>

<h2>Constraints: why the graph must be "static"</h2>
<p>The replay payoff has a price: the graph demands <strong>static shapes and addresses</strong>. This hard constraint directly drives three of SGLang's engineering choices. First, <strong>shapes must be fixed</strong>, hence <strong>bucketing + padding</strong> — the batch can't vary freely.
Second, <strong>addresses must be fixed</strong>, so the runner pre-allocates a set of <strong>static input/output buffers</strong> (the <span class="mono">buffers</span> in the code above) and each step <strong>copies</strong> the real data into those fixed buffers before replay, instead of allocating anew.
Third, <strong>no data-dependent control flow inside the captured region</strong> — a graph records one fixed path, it can't "take the if this step, the else next step". So some ops must <strong>stay outside</strong> the graph, or a "piecewise / breakable" graph <strong>splits out</strong> the dynamic ops and records the rest (covered in Lesson 33 on attention backends).</p>
<p>Chain these three together and last lesson's line "<strong>decode is graphed, prefill runs eager</strong>" is fully explained: decode shapes are highly regular each step (batch steady for a while, sequence growing by 1), so they fit fixed buckets naturally; prefill lengths vary wildly, computing hundreds-to-thousands of tokens at once with a different shape almost every batch, so recording a graph per shape is neither worthwhile nor feasible — it prefers the flexible eager path.</p>

<table class="t">
  <tr><th>Constraint</th><th>Why</th><th>SGLang's response</th></tr>
  <tr><td>Shapes must be static</td><td>Graph binds capture-time tensor shapes</td><td>Bucket the batch + pad up</td></tr>
  <tr><td class="mono">Addresses must be static</td><td>Graph binds capture-time memory pointers</td><td>Pre-allocate static buffers, copy in each step</td></tr>
  <tr><td>No data-dependent control flow</td><td>A graph is one fixed path, no if/else</td><td>Dynamic ops outside / piecewise graph (Lesson 33)</td></tr>
  <tr><td>Buckets have a max</td><td>Can't record a graph for any size</td><td>Batch over max is rejected, falls back to eager</td></tr>
</table>

<div class="cellgroup">
  <div class="cg-cap"><b>batch-size buckets + padding</b>: only these sizes are recorded; a real batch of 5 <strong>rounds up to the bucket-8</strong> graph for replay (highlighted)</div>
  <div class="cells"><span class="lab">buckets</span><span class="cell">1</span><span class="sep">·</span><span class="cell">2</span><span class="sep">·</span><span class="cell">4</span><span class="sep">·</span><span class="cell hl">8</span><span class="sep">·</span><span class="cell">16</span><span class="sep">·</span><span class="cell">…</span><span class="sep">·</span><span class="cell">max</span></div>
  <div class="cells"><span class="lab">real=5</span><span class="cell">5 real rows</span><span class="sep">+</span><span class="cell">3 padding rows</span><span class="sep">→</span><span class="cell hl">use bucket-8 graph</span><span class="sep">→</span><span class="cell q">after replay keep first 5 rows' logits</span></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py ::BaseCudaGraphRunner._pad_to_bucket</span><span class="ln">round a real batch size up to the nearest captured bucket</span></div>
  <pre><span class="kw">@staticmethod</span>
<span class="kw">def</span> _pad_to_bucket(raw_size, buckets):
    <span class="cm"># graphs are captured only for a FIXED set of batch sizes</span>
    <span class="cm"># (buckets). Round a real size UP to the nearest bucket so a</span>
    <span class="cm"># recorded graph can be replayed; can_run_graph must already</span>
    <span class="cm"># have rejected raw_size &gt; max(buckets).</span>
    index = bisect.bisect_left(buckets, raw_size)
    <span class="kw">return</span> buckets[index]   <span class="cm"># smallest bucket &gt;= raw_size</span></pre>
</div>

<p><strong>Concrete example:</strong> say the captured buckets are <span class="mono">[1, 2, 4, 8, 16, …]</span>. A real batch of 5 arrives: <span class="mono">bisect_left</span> lands on <strong>8</strong>, so it <strong>pads up to bucket 8</strong> — adding <strong>3 empty rows</strong> to replay the bucket-8 graph, then keeps only the first 5 rows' logits. This only helps <strong>decode</strong>: each decode step has a fixed shape (every request computes just <strong>1 new token</strong>), so it drops cleanly into a bucket; <strong>prefill</strong> lengths vary wildly with a different shape almost every batch, so it runs eager rather than graphed.</p>

<h2>The payoff: big decode throughput, and a perfect match with the overlap scheduler</h2>
<p>Having killed the hundreds of launches per step, <strong>decode throughput usually gains a lot</strong> — most of all on small models, short kernels, large concurrency, where launch overhead is the highest share. Even better, it is a <strong>perfect match for the overlap scheduler (Lesson 21)</strong>:
while a graph replays, the GPU just powers through the whole graph and barely turns back to the CPU; meanwhile the CPU is freed to <strong>schedule the next step</strong> (sampling, building the next batch). Stack the two and <strong>the GPU replays while the CPU lines up the next step</strong>, leaving the GPU almost no idle gap — the core rhythm of a high-throughput inference engine.</p>
<p>There is of course a cost, a real tradeoff: recording many sizes takes <strong>startup time</strong> (each graph requires running the forward once to record) and eats <strong>memory</strong> (each graph holds its own static buffers). So more buckets isn't strictly better — too dense and startup is slow and memory-hungry; too sparse and padding waste grows.
This "<strong>startup cost / memory ↔ runtime launch overhead saved</strong>" tradeoff recurs later: Lesson 33's piecewise graphs let dynamic ops partly enjoy replay, and Lesson 43's speculative decoding must specially consider how to capture the more complex "draft + verify" shapes. Remember this lesson: <strong>a CUDA graph swaps "launch kernels one by one" for "replay the whole forward in one shot", trading static shapes for gone launch overhead</strong>.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <strong>① The enemy is launch overhead:</strong> one decode forward launches hundreds of tiny kernels, each launch a fixed CPU fee; decode kernels are so short the fee dominates the step and the GPU waits on the CPU.
  <strong>② The fix is record-once-replay-whole:</strong> a CUDA graph records the entire forward's kernel sequence into one graph and replays it as a single submission, collapsing hundreds of launches into one.
  <strong>③ The price is static:</strong> the graph binds shape and address, so you need batch bucketing + padding (<span class="mono">_pad_to_bucket</span>), pre-allocated static buffers, and no data-dependent control flow in the captured region — the very reason <strong>decode is graphed, prefill runs eager</strong>.
  <strong>④ How SGLang does it:</strong> at startup <span class="mono">BaseCudaGraphRunner.capture</span> records one graph per size (1/2/4/8/…/max); at run time it pads to the nearest bucket then <span class="mono">can_run_graph</span> + replay (Lesson 24).
  <strong>⑤ Payoff &amp; partner:</strong> big decode throughput, and a perfect match with the overlap scheduler (Lesson 21) — GPU replays, CPU lines up the next step, GPU barely idles; the cost is startup time and memory (piecewise graphs in Lesson 33, speculative in Lesson 43).
</div>
"""}

LESSON_28 = {"zh": r"""
<p class="lead">
第 24 课的 ModelRunner 把一批 token 跑成了 <strong>logits</strong>——词表里每个 token 一个分数。可分数不是答案，
<strong>下一个 token 才是答案</strong>。从一排几万个分数里挑出"这一步到底吐哪个字"，正是这一课主角 <span class="inline">Sampler</span> 干的活。
它是 Part 6 的收尾，也是请求循环（第 18 课）那一圈里的<strong>第 4 步</strong>：<strong>logits → 下一个 token</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把采样想成一场<strong>按词表开的加权摸彩</strong>：每个 token 是一张彩票，logits 决定它中奖的<strong>权重</strong>。
  <strong>温度</strong>负责调"赔率有多悬殊"——温度很低（冷），几乎每次都摸到大热门；温度很高（热），冷门也可能爆出。
  <strong>top-k / top-p</strong> 则在开摸之前先把"绝无可能"的烂票<strong>扔出箱子</strong>，这样你永远不会摸到一张胡言乱语。
  而<strong>贪心（greedy）</strong>干脆不摸了，直接把权重最大的那张拿走。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  <span class="inline">Sampler</span> 是一个 <span class="mono">nn.Module</span>，输入是 <span class="mono">model.forward</span> 出的 logits，
  输出是<strong>每条请求的下一个 token id</strong>。它读的另一份输入叫 <span class="mono">SamplingBatchInfo</span>：把整批请求的采样参数
  <strong>打包成张量</strong>，于是<strong>同一个 batch 里的不同请求可以用各自不同的参数</strong>（温度、top_p…），一次向量化算完。
  而每条请求的那套旋钮，源头是用户随请求发来的 <span class="mono">SamplingParams</span>（由 TokenizerManager 在入站时构建，第 14 课）。
</div>

<h2>从一排分数到一个字：采样流水线</h2>
<p>logits 不能直接当答案，因为它只是<strong>未归一化的偏好分</strong>。Sampler 要把它<strong>一步步</strong>塑形，最后才掷一次"骰子"。
完整管线（对每条请求并行做）是这样的：先做<strong>惩罚</strong>（repetition / frequency / presence penalty——压制已经出现过的词，避免复读机），
再做<strong>温度</strong>缩放（拿 logits 去除以温度：小于 1 让分布更尖、更确定，大于 1 让它更平、更敢冒险，等于 0 就退化成贪心 argmax），
接着 <strong>softmax</strong> 把分数<strong>归一成概率</strong>，
然后是三道<strong>截断闸门</strong>——<strong>top-k</strong>（只留分数最高的 k 个）、<strong>top-p / nucleus</strong>（按概率从大到小累加，留下刚好凑够 p 的最小集合）、
<strong>min-p</strong>（凡是概率低于"最大概率 × 某个比例"的，一律丢掉）——最后用<strong>多项式采样</strong>掷一次得到结果。
如果整条请求是贪心的，前面这些花活全跳过，直接 <span class="mono">argmax</span> 取最大。</p>

<p>顺序为什么是这个顺序，值得多想一层。<strong>惩罚必须在最前</strong>，因为它要修改的是"原始偏好"——你得先把"这个词刚说过、扣分"算进去，后面的塑形才作数。
<strong>温度紧随其后</strong>，它是唯一一个<strong>重塑整条曲线陡峭程度</strong>的旋钮：温度不改变谁高谁低的<strong>排名</strong>，只改变高低之间的<strong>悬殊程度</strong>，所以必须在截断之前先把曲线调到位。
<strong>top-k / top-p / min-p 三道闸门负责"划定候选圈"</strong>：它们都不改概率的相对大小，只决定"哪些 token 还有资格被摸到"，把没希望的尾巴一刀切掉，既挡住胡言乱语，又给真正合理的几个词留出空间。
最后多项式采样掷骰——其实 <strong>softmax 就夹在温度和闸门中间</strong>：正是它把 logits 变成概率，top-p 的累加、min-p 的比例才有的算，所以完整次序是<strong>惩罚 → 温度 → softmax → 截断 → 多项式采样</strong>——先塑形、再划圈、最后掷骰，每一步只干一件事，这正是这套代码清晰好维护的原因。</p>

<p>这里要澄清一个新手最容易踩的坑：<strong>logits 不是概率</strong>。它只是一排有大有小、甚至有正有负的<strong>实数分数</strong>，彼此之间没有"加起来等于一"的约束，所以你不能直接拿它当概率去抽签。
真正把分数变成概率的是<strong>softmax</strong>：它先对每个分数取指数（让差距按指数放大、并保证全为正），再除以总和归一，得到一排<strong>和为一</strong>的概率。
正因为有这个指数，<strong>温度的威力才那么大</strong>——你在 softmax <strong>之前</strong>把 logits 除以一个小于一的温度，相当于把所有分差<strong>等比例放大</strong>，指数一吃，大的更大、小的几乎归零，分布瞬间变尖；
反过来除以一个大于一的温度，分差被压扁，softmax 出来就接近"人人有份"的均匀分布。理解了"温度作用在 softmax 之前、靠指数放大"，你才真正懂了为什么同一个 0.1 的温度差，在不同区间感受完全不同。</p>

<p>还要强调采样是<strong>逐请求并行</strong>而非逐请求串行的。整批请求的 logits 是一个二维张量（行是请求、列是词表），上面这套惩罚、除温度、softmax、截断、采样，全部以<strong>张量算子</strong>一次性铺开在整批上跑，
而不是写个 for 循环一条条算。这就是为什么把每条请求的参数<strong>打包成张量</strong>如此关键：温度是一列张量、top_p 是一列张量，算子按行广播，天然支持"<strong>这一批里张三贪心、李四温度 0.9、王五还要 top_p=0.8</strong>"同时算完。
高吞吐推理的底层哲学贯穿始终——<strong>能向量化的绝不循环，能一次算完的绝不分多次</strong>，采样这一环也不例外。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>logits</h4><p>model.forward 出的原始分数：词表里<strong>每个 token 一个分</strong>（第 24 课）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>惩罚 penalties</h4><p>repetition / frequency / presence——压低已出现过的词，<strong>反复读机</strong>。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>温度 temperature</h4><p>logits ÷ T：&lt;1 更尖更稳，&gt;1 更平更野，<strong>=0 ⇒ 贪心</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>softmax</h4><p>把 logits 指数<strong>归一成概率</strong>（top-p 的累加、min-p 的比例都要在概率上算）。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>top-k / top-p / min-p → 采样</h4><p>三道闸门在概率上<strong>划定候选圈</strong>，再<strong>多项式掷骰</strong>出下一个 token（贪心则 argmax）。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 280" role="img" aria-label="采样漏斗：从全词表 logits 出发，经温度、top-k、top-p 逐级收窄候选集，最后采样出一个 token">
    <rect x="16" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="80" y="64" text-anchor="middle" style="font-weight:700;fill:var(--blue)">logits（全词表）</text>
    <line x1="22" y1="212" x2="138" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="26" y="182" width="11" height="30" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="40" y="157" width="11" height="55" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="54" y="192" width="11" height="20" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="68" y="142" width="11" height="70" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="82" y="172" width="11" height="40" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="96" y="187" width="11" height="25" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="110" y="152" width="11" height="60" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="124" y="177" width="11" height="35" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <text x="80" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">每个 token 一个分</text>
    <text x="158" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="172" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="236" y="64" text-anchor="middle" style="font-weight:700;fill:var(--amber)">温度 temperature</text>
    <line x1="178" y1="212" x2="294" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="182" y="192" width="11" height="20" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="196" y="182" width="11" height="30" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="210" y="197" width="11" height="15" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="224" y="117" width="11" height="95" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="238" y="187" width="11" height="25" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="252" y="194" width="11" height="18" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="266" y="172" width="11" height="40" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="280" y="190" width="11" height="22" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <text x="236" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">÷T 重塑陡峭度</text>
    <text x="314" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="328" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="392" y="64" text-anchor="middle" style="font-weight:700;fill:var(--teal)">top-k（k=4）</text>
    <line x1="334" y1="212" x2="450" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="346" y="117" width="16" height="95" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="372" y="172" width="16" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="398" y="182" width="16" height="30" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="424" y="187" width="16" height="25" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="392" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">留最高 k 个</text>
    <text x="470" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="484" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="548" y="64" text-anchor="middle" style="font-weight:700;fill:var(--purple)">top-p（核）</text>
    <line x1="490" y1="212" x2="606" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="518" y="117" width="18" height="95" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <rect x="552" y="172" width="18" height="40" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="548" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">累计≥p 的核</text>
    <text x="626" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="640" y="44" width="128" height="190" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="704" y="64" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">token</text>
    <line x1="646" y1="212" x2="762" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <text x="704" y="110" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:12px">京</text>
    <rect x="693" y="117" width="22" height="95" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="704" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">采样一个</text>
  </svg>
  <div class="figcap"><b>图 1 · 采样漏斗：logits → 温度 → top-k → top-p → token</b> — 候选集在每一级被收窄：全词表 logits 先被温度重塑陡峭度（不改排名、只改悬殊），再被 top-k 砍到最高 k 个、被 top-p 收成核（nucleus），最后只掷出一个 token。</div>
</div>

<h2>四个旋钮各自在干什么</h2>
<p>用户能拧的旋钮就那么几个，但每个的脾气很不一样，搞混了输出就会"太呆"或"太疯"。一句话记法：<strong>温度调悬殊、top-k 限个数、top-p 限累计、min-p 设地板、惩罚反复读</strong>。
温度是<strong>连续</strong>地控制随机性的总阀门；top-k 是个<strong>硬上限</strong>，无论分布多平都只留 k 个；top-p 则是<strong>自适应</strong>的——分布很尖时它可能只留 1～2 个，分布很平时它会多留几个，正好把"该确定时确定、该发散时发散"自动化了；
min-p 用"相对最大值的比例"当门槛，比固定的 top-k 更鲁棒。它们常常<strong>叠加使用</strong>（先 top-k 再 top-p 再 min-p），层层收窄候选集。</p>

<table class="t">
  <tr><th>参数</th><th>它做什么</th></tr>
  <tr><td class="mono">temperature</td><td>logits 除以它：&lt;1 让分布更尖更确定，&gt;1 更平更随机，<strong>=0 退化为贪心 argmax</strong></td></tr>
  <tr><td class="mono">top_k</td><td>只保留分数<strong>最高的 k 个</strong> token，其余直接丢弃（硬性个数上限）</td></tr>
  <tr><td class="mono">top_p</td><td>按概率从大到小累加，留下<strong>刚好凑够 p</strong> 的最小集合（nucleus，自适应大小）</td></tr>
  <tr><td class="mono">min_p</td><td>丢掉概率低于"<strong>最大概率 × min_p</strong>"的所有 token（相对地板）</td></tr>
  <tr><td class="mono">penalties</td><td>repetition / frequency / presence——<strong>压低已出现的词</strong>，抑制复读</td></tr>
</table>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/sampling/sampling_params.py ::SamplingParams</span><span class="ln">控制“如何”从 logits 抽取下一个 token 的旋钮</span></div>
  <pre><span class="kw">class</span> SamplingParams:
    <span class="cm"># knobs shaping HOW the next token is drawn from logits.</span>
    max_new_tokens: <span class="kw">int</span> = 128
    temperature: <span class="kw">float</span> = 1.0      <span class="cm"># &lt;1 sharpen, &gt;1 flatten</span>
    top_k: <span class="kw">int</span> = TOP_K_ALL        <span class="cm"># 1&lt;&lt;30 = keep whole vocab</span>
    top_p: <span class="kw">float</span> = 1.0            <span class="cm"># nucleus: smallest set, cumprob &gt;= p</span>
    min_p: <span class="kw">float</span> = 0.0
    frequency_penalty: <span class="kw">float</span> = 0.0
    presence_penalty: <span class="kw">float</span> = 0.0
    repetition_penalty: <span class="kw">float</span> = 1.0
    ...</pre>
</div>

<p>几个具体例子帮你把旋钮对上行为：<span class="mono">temperature=0</span> ⇒ <strong>贪心 / argmax</strong>，每步直接取概率最大的那个 token，完全确定、可复现；
<span class="mono">top_p=0.9, top_k=50</span> ⇒ 先把候选限制在<strong>最高的 50 个</strong>里，再从中取累计概率刚好凑够 0.9 的那一小撮核（nucleus）来采样；
默认的 <span class="mono">temperature=1.0, top_p=1.0</span>（且 <span class="mono">top_k = 1&lt;&lt;30</span> 等于整张词表）⇒ <strong>不做任何重塑或截断</strong>，就是对原始分布的"原汁原味"采样。</p>

<p>把这张表读活：温度=0 是一个特殊而常用的设置——它让输出<strong>完全确定</strong>，同样的输入永远得到同样的输出，做评测、做单测、要可复现时几乎都用它。
而做创意写作、对话、需要多样性时，则会把温度抬到 0.7～1.0，再配 top_p≈0.9 之类的截断，让它"<strong>在合理范围内自由发挥</strong>"。
理解了每个旋钮，你就能把"模型输出太死板"或"模型胡说八道"这类抱怨，精确翻译成"该调哪个参数"，而不是瞎试。</p>

<p>再点破 top-k 与 top-p 的<strong>本质区别</strong>，这是面试与调参里最常被问倒的一点。top-k 是<strong>"按名次"</strong>截断——不管分布长什么样，永远只留前 k 名，所以分布很尖时它可能多留了一堆其实没必要的尾巴，分布很平时又可能误杀了不少势均力敌的好词。
top-p 则是<strong>"按累计概率"</strong>截断——它会根据当下分布的形状<strong>动态伸缩候选集</strong>：当模型很笃定（某个词概率 0.95）时，它可能只留 1 个；当模型很犹豫（一堆词各 0.1）时，它会多留十几个。
正因为这种自适应，top-p 在实践中往往比固定的 top-k 更省心，也更不容易在"该确定时发散、该发散时收死"上翻车。而 min-p 又补上一条更稳的地板：它不看名次也不看累计，只问"你的概率有没有达到最大概率的某个零头"，达不到就一律踢掉，对极端分布尤其鲁棒。三者各管一个维度，叠起来用才周全。</p>

<h2>贪心 vs 采样：要确定，还是要多样</h2>
<p>整条管线的最后一掷，只有两种性格。<strong>贪心</strong>每步都拿概率最大的那个 token，<strong>确定、可复现，但容易呆板、爱重复</strong>；
<strong>采样</strong>则按概率<strong>掷骰子</strong>，<strong>有随机性、更多样、更像人</strong>，但也可能偶尔抽到次优的词。
代码里这是一个清爽的二分支：<span class="mono">is_all_greedy</span> 为真就 <span class="mono">argmax</span> 收工，否则走"温度 → softmax → top-k/top-p/min-p → 多项式采样"那条完整路径。</p>

<div class="cols">
  <div class="col"><h4>贪心 Greedy（argmax）</h4><p>每步取概率<strong>最大</strong>的 token；<strong>确定、可复现</strong>，评测/单测/要稳的场景首选。
  代价：输出<strong>呆板、易复读</strong>，多样性差。等价于 temperature=0，连骰子都不掷。</p></div>
  <div class="col"><h4>采样 Sampling（multinomial）</h4><p>按概率<strong>随机抽</strong>一个 token；<strong>多样、有创造力、更自然</strong>。
  靠 temperature/top-k/top-p/min-p <strong>控制随机的程度与范围</strong>。代价：<strong>不可复现</strong>（除非固定 RNG）、偶有次优。</p></div>
</div>

<p>实践中怎么在两者之间选？<strong>要稳就贪心，要活就采样</strong>：跑评测、做单测、要逐字复现的，统一用贪心（或温度=0）；做对话、写作、要多样性的，用采样并把温度配到 0.7～1.0。
调参时顺手的次序是"<strong>先定温度、再定截断</strong>"：温度决定整体的冒险程度，截断再在此基础上划定安全边界，两步各管一段、互不打架，调起来心里有数。</p>

<p>这里有个常见误区：以为"采样=不可控"。其实采样的随机性是被那几个旋钮<strong>牢牢框住</strong>的——top-k/top-p 先把没希望的词全部踢出候选，
所以掷骰子时<strong>箱子里根本没有烂票</strong>，再随机也随机不到胡言乱语上去。温度则决定"在这些合理候选之间，要多大胆地冒险"。
换句话说，<strong>截断管质量，温度管多样性</strong>，两者配合，才能既不呆板又不离谱。</p>

<h2>采样之前的"钩子"：约束、偏置与确定性</h2>
<p>采样不是孤立的一掷，它是很多功能<strong>挂钩</strong>的地方，而且大多挂在"softmax 之前对 logits 动手"这一步。最重要的是<strong>结构化输出</strong>（第 48 课）：
当你要求模型<strong>必须</strong>输出合法 JSON、或匹配某个正则/语法时，约束引擎会在采样前把所有"此刻语法不允许"的 token 的 logit <strong>置为 −∞</strong>（mask 掉），
于是 softmax 后它们的概率正好是 0，<strong>采样根本不可能选到它们</strong>——这就从机制上保证了输出永远语法合法，而不是事后去校验、失败再重试。</p>

<p>同一个位置还挂着别的钩子：<strong>logit-bias</strong> 让用户手动给某些 token 加减分（鼓励或禁止特定词）；<strong>min_tokens / EOS 抑制</strong>在生成长度还没达标时，
把结束符 EOS 的 logit 压到 −∞，强迫模型<strong>先别停</strong>；<strong>确定性推理</strong>则把随机数发生器（RNG）的种子<strong>钉死</strong>（用 positions 派生每个位置的种子），
让"采样"这件本质随机的事也能逐 token 复现。这些功能各有各的课，但它们都在<strong>同一个 logits 上、采样这一掷之前</strong>动手，这正是"为什么 Sampler 是个枢纽"的原因。</p>

<p>为什么这些钩子非得挤在采样<strong>之前</strong>、而不是采样之后再补救？因为采样是一道<strong>不可逆的"坍缩"</strong>：一旦掷了骰子，一整排概率就塌成了一个确定的 token id，信息全丢了。
事后想纠正——比如发现选了个违反语法的词——就只能<strong>整步重来</strong>，既慢又可能反复失败。而在采样前对 logits 动手，等于在"还是一排概率"的阶段就把规则<strong>焊进分布本身</strong>：违规的词概率直接归零，越界的结束符永远抽不到，被偏置的词按你的意愿增减。
这是一种"<strong>用约束塑造可能性、而非事后否决结果</strong>"的思路，既高效又确定，也是为什么结构化输出、min_tokens、logit-bias 这些看似不相干的功能，最终都汇聚到采样器这一个收口上。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>top-p 截断</b>（设 p=0.9）：概率从大到小排好，<strong>累加</strong>到刚好 ≥ 0.9 就停，<strong>保留这段前缀（高亮），其余全丢</strong></div>
  <div class="cells"><span class="lab">排序概率</span><span class="cell hl">0.50</span><span class="sep">+</span><span class="cell hl">0.30</span><span class="sep">+</span><span class="cell hl">0.12</span><span class="sep">|</span><span class="cell">0.05</span><span class="sep">+</span><span class="cell">0.03</span></div>
  <div class="cells"><span class="lab">累计和</span><span class="cell">0.50</span><span class="sep">→</span><span class="cell">0.80</span><span class="sep">→</span><span class="cell hl">0.92 ✓ 停</span><span class="sep">→</span><span class="cell q">后面的 0.05/0.03 被切掉，不参与采样</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="在排序后的概率分布上做 top-p 核截断：概率从大到小排好，累计到刚好达到 p=0.9 的最小前缀被保留为核，其后的长尾被切掉">
    <rect x="68" y="70" width="370" height="170" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="253" y="62" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:13px">核 nucleus：保留前 5 个</text>
    <text x="565" y="62" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:13px">尾部切掉 cut tail</text>
    <line x1="60" y1="240" x2="690" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="60" y1="78" x2="690" y2="78" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:6 4"/>
    <text x="694" y="82" style="fill:var(--amber);font-size:12px;font-weight:700">p = 0.9</text>
    <line x1="442" y1="56" x2="442" y2="248" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <rect x="80" y="104" width="50" height="136" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="155" y="152" width="50" height="88" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="230" y="176" width="50" height="64" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="305" y="196" width="50" height="44" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="380" y="212" width="50" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="455" y="220" width="50" height="20" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:3 3"/>
    <rect x="530" y="228" width="50" height="12" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:3 3"/>
    <rect x="605" y="232" width="50" height="8" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:3 3"/>
    <text x="105" y="98" text-anchor="middle" style="fill:var(--teal);font-size:11px">.34</text>
    <text x="180" y="146" text-anchor="middle" style="fill:var(--teal);font-size:11px">.22</text>
    <text x="255" y="170" text-anchor="middle" style="fill:var(--teal);font-size:11px">.16</text>
    <text x="330" y="190" text-anchor="middle" style="fill:var(--teal);font-size:11px">.11</text>
    <text x="405" y="206" text-anchor="middle" style="fill:var(--teal);font-size:11px">.07</text>
    <text x="480" y="214" text-anchor="middle" style="fill:var(--faint);font-size:11px">.05</text>
    <text x="555" y="222" text-anchor="middle" style="fill:var(--faint);font-size:11px">.03</text>
    <text x="630" y="226" text-anchor="middle" style="fill:var(--faint);font-size:11px">.02</text>
    <polyline points="105,179 180,139 255,110 330,91 405,78 480,69 555,64 630,60" style="fill:none;stroke:var(--amber);stroke-width:2"/>
    <circle cx="105" cy="179" r="3" style="fill:var(--amber)"/>
    <circle cx="180" cy="139" r="3" style="fill:var(--amber)"/>
    <circle cx="255" cy="110" r="3" style="fill:var(--amber)"/>
    <circle cx="330" cy="91" r="3" style="fill:var(--amber)"/>
    <circle cx="405" cy="78" r="4" style="fill:var(--amber);stroke:var(--accent-ink);stroke-width:1.5"/>
    <circle cx="480" cy="69" r="3" style="fill:var(--amber)"/>
    <circle cx="555" cy="64" r="3" style="fill:var(--amber)"/>
    <circle cx="630" cy="60" r="3" style="fill:var(--amber)"/>
    <text x="150" y="50" style="fill:var(--amber);font-size:11px">累计概率 cumprob ↗</text>
    <text x="105" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t1</text>
    <text x="180" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t2</text>
    <text x="255" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t3</text>
    <text x="330" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t4</text>
    <text x="405" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t5</text>
    <text x="480" y="258" text-anchor="middle" class="mono" style="fill:var(--faint);font-size:11px">t6</text>
    <text x="555" y="258" text-anchor="middle" class="mono" style="fill:var(--faint);font-size:11px">t7</text>
    <text x="630" y="258" text-anchor="middle" class="mono" style="fill:var(--faint);font-size:11px">t8</text>
    <text x="253" y="284" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">累计 .34→.56→.72→.83→.90 ✓ 停</text>
    <text x="565" y="284" text-anchor="middle" style="fill:var(--muted);font-size:12px">.05 / .03 / .02 ✕ 丢弃</text>
  </svg>
  <div class="figcap"><b>图 2 · top-p 核截断（在排序后的概率分布上）</b> — 概率从大到小排好（柱），沿橙色曲线累加；累计刚好达到 p=0.9 的最小前缀（前 5 个）被保留为核（高亮区），其后的长尾 t6–t8 被切掉、永不参与采样。</div>
</div>

<h2>看一眼真正的代码</h2>
<p>下面是 <span class="mono">Sampler.forward</span> 的真实骨架（class 约第 68 行，forward 约第 93 行）。看那个清爽的<strong>贪心 vs 采样</strong>二分支，
以及标准路径上"<strong>温度 → softmax → 截断+采样</strong>"的次序——和上面讲的管线一一对得上：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/sampler.py ::Sampler</span><span class="ln">logits → 下一个 token</span></div>
  <pre><span class="kw">def</span> forward(self, logits_output, sampling_info, ...):
    logits = logits_output.next_token_logits
    logits = self._preprocess_logits(logits, sampling_info)   <span class="cm"># 自定义处理 + 清理 NaN</span>
    <span class="kw">if</span> sampling_info.is_all_greedy:                           <span class="cm"># 整批贪心：直接取最大</span>
        batch_next_token_ids = torch.argmax(logits, -1)
    <span class="kw">else</span>:
        logits.div_(sampling_info.temperatures)               <span class="cm"># 1) 温度缩放（按请求向量化）</span>
        logits[:] = torch.softmax(logits, dim=-1)             <span class="cm"># 2) 转成概率</span>
        probs = logits
        batch_next_token_ids = self._sample_from_probs(       <span class="cm"># 3) top-k/top-p/min-p + 多项式采样</span>
            probs, sampling_info, positions, simple_sampling_case)
    <span class="kw">return</span> batch_next_token_ids</pre>
</div>

<p>注意几个细节：<span class="mono">sampling_info.temperatures</span> 是<strong>一整个张量</strong>——每条请求一个温度，所以 <span class="mono">div_</span> 是逐请求向量化的，
这正是"<strong>一个 batch 里不同请求用不同参数</strong>"在代码里的样子。<span class="mono">is_all_greedy</span> 是个整批级的快捷判断：只有当这一批<strong>全都</strong>是贪心时才走 argmax 快路径，
否则统一走采样路径（贪心请求在那条路上用各自的温度/参数也能得到等价结果）。<span class="mono">_sample_from_probs</span> 内部才真正调用 top-k/top-p/min-p 的算子和多项式采样。</p>

<p>把这一课接回整条主线：这是请求循环（第 18 课）一圈里的<strong>第 4 步</strong>。model.forward（第 24 课）出 logits，Sampler 把它变成下一个 token，
这个 token 会被<strong>追加回 Req</strong>，下一步又喂进模型——这正是<strong>自回归</strong>（第 4 课）"吐一个、接上去、再吐一个"的循环。
Part 6 到此收尾：第 24 课讲 ModelRunner 这道"决策→计算"的门，第 25/26 课讲模型怎么加载、怎么写，第 27 课讲解码为何能用 CUDA 图，
而本课补上最后一环——<strong>算出来的 logits，究竟怎么变成你屏幕上的那个字</strong>。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>Sampler</strong>：一个 <span class="mono">nn.Module</span>，把 <strong>logits → 下一个 token id</strong>；读 <span class="mono">SamplingBatchInfo</span>（把每请求参数打包成张量）。</li>
    <li><strong>管线</strong>：惩罚 → 温度（÷T；&lt;1 尖、&gt;1 平、=0 贪心）→ softmax → top-k / top-p / min-p 截断 → 多项式采样。</li>
    <li><strong>旋钮</strong>：<span class="mono">SamplingParams</span> 逐请求设温度/top_p/top_k/min_p/惩罚；<strong>同一 batch 不同请求可用不同参数</strong>（向量化）。</li>
    <li><strong>贪心 vs 采样</strong>：argmax 确定可复现 vs 多项式随机多样；截断管质量、温度管多样性。</li>
    <li><strong>钩子</strong>：结构化输出（第 48 课）在采样前把违规 token 置 −∞；logit-bias、min_tokens/EOS 抑制、确定性 RNG 都挂在这里。它是请求循环（第 18 课）的第 4 步、自回归（第 4 课）的一环。</li>
  </ul>
</div>
""", "en": r"""
<p class="lead">
The ModelRunner of Lesson 24 turned a batch of tokens into <strong>logits</strong> — one score per vocabulary token. But a score isn't an answer;
<strong>the next token is the answer</strong>. Picking "which token do we emit this step" out of a row of tens of thousands of scores is exactly what this lesson's star, the <span class="inline">Sampler</span>, does.
It closes Part 6, and it is <strong>step 4</strong> of the request loop (Lesson 18): <strong>logits → the next token</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of sampling as a <strong>weighted lottery over the vocabulary</strong>: every token is a ticket and the logits set its <strong>odds</strong>.
  <strong>Temperature</strong> reshapes how lopsided the odds are — cold (low T) almost always draws the favorite; hot (high T) lets wild cards win.
  <strong>top-k / top-p</strong> first <strong>throw the no-hope tickets out of the box</strong>, so you can never draw garbage.
  And <strong>greedy</strong> skips the draw entirely and just takes the heaviest ticket.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  The <span class="inline">Sampler</span> is an <span class="mono">nn.Module</span>; its input is the logits from <span class="mono">model.forward</span>,
  its output is <strong>the next token id for each request</strong>. Its other input is <span class="mono">SamplingBatchInfo</span>: the per-request params
  <strong>packed into tensors</strong>, so <strong>different requests in ONE batch can use different params</strong> (temperature, top_p…) in one vectorized pass.
  Each request's knob set comes from the user's <span class="mono">SamplingParams</span> (built by the TokenizerManager on intake, Lesson 14).
</div>

<h2>From a row of scores to one token: the sampling pipeline</h2>
<p>logits can't be the answer directly — they're just <strong>unnormalized preferences</strong>. The Sampler reshapes them <strong>step by step</strong>, then rolls a single die.
The full pipeline (done in parallel per request): first <strong>penalties</strong> (repetition / frequency / presence — discourage words already produced, kill the parrot),
then <strong>temperature</strong> scaling (divide the logits by T: &lt;1 sharpens and steadies, &gt;1 flattens and dares, =0 collapses to greedy argmax), then <strong>softmax</strong> turns the scores into <strong>probabilities</strong>,
then three <strong>truncation gates</strong> — <strong>top-k</strong> (keep the k highest), <strong>top-p / nucleus</strong> (sort by prob descending and keep the smallest set summing to p),
<strong>min-p</strong> (drop every token below "max prob × a fraction") — finally a <strong>multinomial sample</strong> rolls the die.
If the whole request is greedy, all that is skipped and it's just <span class="mono">argmax</span>.</p>

<p>The order is worth a second thought. <strong>Penalties come first</strong> because they edit the "raw preference" — you must fold in "this word was just said, dock points" before any reshaping counts.
<strong>Temperature is next</strong>; it's the only knob that <strong>reshapes the steepness of the whole curve</strong>: it never changes the <strong>ranking</strong> of who's higher, only how <strong>lopsided</strong> the gaps are, so it must run before truncation.
<strong>top-k / top-p / min-p draw the candidate circle</strong>: none of them changes relative probabilities, they only decide "which tokens are still eligible", chopping off the hopeless tail — blocking garbage while leaving room for the genuinely reasonable few.
A multinomial roll comes last — <strong>softmax actually sits between temperature and the gates</strong>: it's what turns logits into probabilities, which top-p's cumsum and min-p's ratio then operate on, so the full order is <strong>penalties → temperature → softmax → truncation → multinomial sample</strong> — shape, then circle, then roll, each step doing one thing, which is why this code stays clean.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>logits</h4><p>raw scores from model.forward: <strong>one score per token</strong> in the vocabulary (Lesson 24).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>penalties</h4><p>repetition / frequency / presence — push down words already produced, <strong>anti-parrot</strong>.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>temperature</h4><p>logits ÷ T: &lt;1 sharper and steadier, &gt;1 flatter and wilder, <strong>=0 ⇒ greedy</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>softmax</h4><p>exponentiate logits into <strong>probabilities</strong> (top-p's cumsum and min-p's ratio both need probs).</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>top-k / top-p / min-p → sample</h4><p>three gates <strong>draw the candidate circle</strong> on the probs, then <strong>roll a multinomial die</strong> for the next token (or argmax if greedy).</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 280" role="img" aria-label="Sampling funnel: starting from full-vocab logits, temperature, top-k and top-p narrow the candidate set stage by stage, until one token is sampled">
    <rect x="16" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="80" y="64" text-anchor="middle" style="font-weight:700;fill:var(--blue)">logits (full vocab)</text>
    <line x1="22" y1="212" x2="138" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="26" y="182" width="11" height="30" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="40" y="157" width="11" height="55" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="54" y="192" width="11" height="20" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="68" y="142" width="11" height="70" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="82" y="172" width="11" height="40" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="96" y="187" width="11" height="25" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="110" y="152" width="11" height="60" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <rect x="124" y="177" width="11" height="35" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <text x="80" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">one score per token</text>
    <text x="158" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="172" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="236" y="64" text-anchor="middle" style="font-weight:700;fill:var(--amber)">temperature</text>
    <line x1="178" y1="212" x2="294" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="182" y="192" width="11" height="20" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="196" y="182" width="11" height="30" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="210" y="197" width="11" height="15" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="224" y="117" width="11" height="95" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="238" y="187" width="11" height="25" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="252" y="194" width="11" height="18" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="266" y="172" width="11" height="40" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <rect x="280" y="190" width="11" height="22" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <text x="236" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">÷T reshapes steepness</text>
    <text x="314" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="328" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="392" y="64" text-anchor="middle" style="font-weight:700;fill:var(--teal)">top-k (k=4)</text>
    <line x1="334" y1="212" x2="450" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="346" y="117" width="16" height="95" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="372" y="172" width="16" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="398" y="182" width="16" height="30" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="424" y="187" width="16" height="25" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="392" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">keep k highest</text>
    <text x="470" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="484" y="44" width="128" height="190" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="548" y="64" text-anchor="middle" style="font-weight:700;fill:var(--purple)">top-p (nucleus)</text>
    <line x1="490" y1="212" x2="606" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="518" y="117" width="18" height="95" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <rect x="552" y="172" width="18" height="40" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="548" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">smallest set, cumprob≥p</text>
    <text x="626" y="138" text-anchor="middle" style="fill:var(--muted);font-size:18px">→</text>
    <rect x="640" y="44" width="128" height="190" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="704" y="64" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">token</text>
    <line x1="646" y1="212" x2="762" y2="212" style="stroke:var(--faint);stroke-width:1"/>
    <text x="704" y="110" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:12px">tok</text>
    <rect x="693" y="117" width="22" height="95" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="704" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">sample one</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Sampling funnel: logits → temperature → top-k → top-p → token</b> — the candidate set narrows at each stage: full-vocab logits are first reshaped by temperature (ranking unchanged, only lopsidedness), then chopped to the k highest by top-k, shrunk to the nucleus by top-p, and finally one token is rolled.</div>
</div>

<h2>What each of the four knobs does</h2>
<p>The user only gets a handful of knobs, but each has a different temperament — mix them up and the output goes "too dull" or "too wild". One-line memory: <strong>temperature sets lopsidedness, top-k caps the count, top-p caps the cumulative, min-p sets a floor, penalties fight repetition</strong>.
Temperature is a <strong>continuous</strong> master valve for randomness; top-k is a <strong>hard cap</strong> that keeps exactly k no matter how flat the distribution; top-p is <strong>adaptive</strong> — it may keep 1–2 tokens when the distribution is sharp and several when it's flat, automating "be decisive when you should, diverge when you should";
min-p uses "a fraction of the max" as a threshold, more robust than a fixed top-k. They're often <strong>stacked</strong> (top-k then top-p then min-p), narrowing the candidate set in layers.</p>

<table class="t">
  <tr><th>Param</th><th>What it does</th></tr>
  <tr><td class="mono">temperature</td><td>divide logits by it: &lt;1 sharpens (more deterministic), &gt;1 flattens (more random), <strong>=0 collapses to greedy argmax</strong></td></tr>
  <tr><td class="mono">top_k</td><td>keep only the <strong>k highest-scoring</strong> tokens, discard the rest (a hard count cap)</td></tr>
  <tr><td class="mono">top_p</td><td>sort by prob descending, keep the smallest set <strong>summing to p</strong> (nucleus, adaptive size)</td></tr>
  <tr><td class="mono">min_p</td><td>drop every token below "<strong>max prob × min_p</strong>" (a relative floor)</td></tr>
  <tr><td class="mono">penalties</td><td>repetition / frequency / presence — <strong>push down words already seen</strong>, suppress repeats</td></tr>
</table>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/sampling/sampling_params.py ::SamplingParams</span><span class="ln">the knobs controlling HOW the next token is drawn from logits</span></div>
  <pre><span class="kw">class</span> SamplingParams:
    <span class="cm"># knobs shaping HOW the next token is drawn from logits.</span>
    max_new_tokens: <span class="kw">int</span> = 128
    temperature: <span class="kw">float</span> = 1.0      <span class="cm"># &lt;1 sharpen, &gt;1 flatten</span>
    top_k: <span class="kw">int</span> = TOP_K_ALL        <span class="cm"># 1&lt;&lt;30 = keep whole vocab</span>
    top_p: <span class="kw">float</span> = 1.0            <span class="cm"># nucleus: smallest set, cumprob &gt;= p</span>
    min_p: <span class="kw">float</span> = 0.0
    frequency_penalty: <span class="kw">float</span> = 0.0
    presence_penalty: <span class="kw">float</span> = 0.0
    repetition_penalty: <span class="kw">float</span> = 1.0
    ...</pre>
</div>

<p>A few concrete examples to map knobs to behavior: <span class="mono">temperature=0</span> ⇒ <strong>greedy / argmax</strong> — take the highest-prob token each step, fully deterministic and reproducible;
<span class="mono">top_p=0.9, top_k=50</span> ⇒ first restrict candidates to the <strong>50 highest</strong>, then sample from the nucleus whose cumulative prob just reaches 0.9 within them;
the default <span class="mono">temperature=1.0, top_p=1.0</span> (with <span class="mono">top_k = 1&lt;&lt;30</span>, the whole vocab) ⇒ <strong>no reshaping or truncation at all</strong> — plain, unmodified sampling of the raw distribution.</p>

<p>Read the table live: temperature=0 is a special, common setting — it makes output <strong>fully deterministic</strong>, the same input always yields the same output, which is what you want for evals, unit tests, reproducibility.
For creative writing, chat, or anything needing diversity, you raise temperature to 0.7–1.0 and add a truncation like top_p≈0.9, letting it "<strong>roam freely within a sane range</strong>".
Once you understand each knob, you can translate "output too rigid" or "model rambling" into "which param to turn" instead of guessing.</p>

<h2>Greedy vs sampling: determinism, or diversity</h2>
<p>The final roll has only two temperaments. <strong>Greedy</strong> takes the highest-prob token every step — <strong>deterministic, reproducible, but dull and repetition-prone</strong>;
<strong>sampling</strong> <strong>rolls a die</strong> by probability — <strong>random, diverse, more human</strong>, but occasionally draws a suboptimal word.
In code it's a crisp two-way branch: if <span class="mono">is_all_greedy</span>, do <span class="mono">argmax</span> and finish; otherwise take the full "temperature → softmax → top-k/top-p/min-p → multinomial" path.</p>

<div class="cols">
  <div class="col"><h4>Greedy (argmax)</h4><p>take the <strong>highest</strong>-prob token each step; <strong>deterministic, reproducible</strong>, the default for evals/tests/stable scenarios.
  Cost: output is <strong>dull, prone to repeats</strong>, low diversity. Equivalent to temperature=0 — no die rolled at all.</p></div>
  <div class="col"><h4>Sampling (multinomial)</h4><p>draw a token <strong>at random</strong> by probability; <strong>diverse, creative, more natural</strong>.
  temperature/top-k/top-p/min-p <strong>control how much and how wide</strong> the randomness is. Cost: <strong>not reproducible</strong> (unless RNG is pinned), occasionally suboptimal.</p></div>
</div>

<p>A common misconception: "sampling = uncontrollable". In fact the randomness is <strong>tightly fenced</strong> by those knobs — top-k/top-p first kick every hopeless word out of the candidate set,
so when you roll the die <strong>there are no garbage tickets in the box</strong>; however random, you can't draw nonsense. Temperature then decides "how boldly to gamble among these sane candidates".
In other words, <strong>truncation governs quality, temperature governs diversity</strong>; together they keep output neither dull nor unhinged.</p>

<h2>The hooks before sampling: constraints, bias, determinism</h2>
<p>Sampling isn't an isolated roll; it's where many features <strong>hook in</strong>, and most of them act "on the logits, before softmax". The most important is <strong>structured output</strong> (Lesson 48):
when you require the model to emit <strong>valid</strong> JSON, or match a regex/grammar, the constraint engine sets the logit of every "grammar-disallowed-right-now" token to <strong>−∞</strong> (masks it) before sampling,
so after softmax their probability is exactly 0 and <strong>sampling can never pick them</strong> — guaranteeing grammar-valid output by mechanism, not by validating afterward and retrying on failure.</p>

<p>The same spot carries other hooks: <strong>logit-bias</strong> lets the user manually add/subtract on specific tokens (encourage or ban words); <strong>min_tokens / EOS suppression</strong> pushes the EOS logit to −∞
while the generation length hasn't been met, forcing the model <strong>not to stop yet</strong>; <strong>deterministic inference</strong> <strong>pins the RNG</strong> seed (derived per position from positions),
making the inherently random act of "sampling" reproducible token by token. These features each have their own lesson, but they all act on <strong>the same logits, before this one roll</strong> — which is why the Sampler is a hub.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>top-p truncation</b> (p=0.9): probs sorted descending, <strong>accumulate</strong> until just ≥ 0.9, then <strong>keep that prefix (highlighted), drop the rest</strong></div>
  <div class="cells"><span class="lab">sorted probs</span><span class="cell hl">0.50</span><span class="sep">+</span><span class="cell hl">0.30</span><span class="sep">+</span><span class="cell hl">0.12</span><span class="sep">|</span><span class="cell">0.05</span><span class="sep">+</span><span class="cell">0.03</span></div>
  <div class="cells"><span class="lab">running sum</span><span class="cell">0.50</span><span class="sep">→</span><span class="cell">0.80</span><span class="sep">→</span><span class="cell hl">0.92 ✓ stop</span><span class="sep">→</span><span class="cell q">the trailing 0.05/0.03 are cut, never sampled</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="top-p nucleus cut on a sorted probability distribution: probabilities sorted descending, accumulated until just reaching p=0.9; the smallest prefix forming the nucleus is kept, the long tail beyond it is cut">
    <rect x="68" y="70" width="370" height="170" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="253" y="62" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:13px">nucleus: keep first 5</text>
    <text x="565" y="62" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:13px">cut tail</text>
    <line x1="60" y1="240" x2="690" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="60" y1="78" x2="690" y2="78" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:6 4"/>
    <text x="694" y="82" style="fill:var(--amber);font-size:12px;font-weight:700">p = 0.9</text>
    <line x1="442" y1="56" x2="442" y2="248" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <rect x="80" y="104" width="50" height="136" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="155" y="152" width="50" height="88" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="230" y="176" width="50" height="64" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="305" y="196" width="50" height="44" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="380" y="212" width="50" height="28" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="455" y="220" width="50" height="20" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:3 3"/>
    <rect x="530" y="228" width="50" height="12" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:3 3"/>
    <rect x="605" y="232" width="50" height="8" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:3 3"/>
    <text x="105" y="98" text-anchor="middle" style="fill:var(--teal);font-size:11px">.34</text>
    <text x="180" y="146" text-anchor="middle" style="fill:var(--teal);font-size:11px">.22</text>
    <text x="255" y="170" text-anchor="middle" style="fill:var(--teal);font-size:11px">.16</text>
    <text x="330" y="190" text-anchor="middle" style="fill:var(--teal);font-size:11px">.11</text>
    <text x="405" y="206" text-anchor="middle" style="fill:var(--teal);font-size:11px">.07</text>
    <text x="480" y="214" text-anchor="middle" style="fill:var(--faint);font-size:11px">.05</text>
    <text x="555" y="222" text-anchor="middle" style="fill:var(--faint);font-size:11px">.03</text>
    <text x="630" y="226" text-anchor="middle" style="fill:var(--faint);font-size:11px">.02</text>
    <polyline points="105,179 180,139 255,110 330,91 405,78 480,69 555,64 630,60" style="fill:none;stroke:var(--amber);stroke-width:2"/>
    <circle cx="105" cy="179" r="3" style="fill:var(--amber)"/>
    <circle cx="180" cy="139" r="3" style="fill:var(--amber)"/>
    <circle cx="255" cy="110" r="3" style="fill:var(--amber)"/>
    <circle cx="330" cy="91" r="3" style="fill:var(--amber)"/>
    <circle cx="405" cy="78" r="4" style="fill:var(--amber);stroke:var(--accent-ink);stroke-width:1.5"/>
    <circle cx="480" cy="69" r="3" style="fill:var(--amber)"/>
    <circle cx="555" cy="64" r="3" style="fill:var(--amber)"/>
    <circle cx="630" cy="60" r="3" style="fill:var(--amber)"/>
    <text x="150" y="50" style="fill:var(--amber);font-size:11px">cumulative prob ↗</text>
    <text x="105" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t1</text>
    <text x="180" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t2</text>
    <text x="255" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t3</text>
    <text x="330" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t4</text>
    <text x="405" y="258" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:11px">t5</text>
    <text x="480" y="258" text-anchor="middle" class="mono" style="fill:var(--faint);font-size:11px">t6</text>
    <text x="555" y="258" text-anchor="middle" class="mono" style="fill:var(--faint);font-size:11px">t7</text>
    <text x="630" y="258" text-anchor="middle" class="mono" style="fill:var(--faint);font-size:11px">t8</text>
    <text x="253" y="284" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">cum .34→.56→.72→.83→.90 ✓ stop</text>
    <text x="565" y="284" text-anchor="middle" style="fill:var(--muted);font-size:12px">.05 / .03 / .02 ✕ dropped</text>
  </svg>
  <div class="figcap"><b>Fig 2 · top-p nucleus cut on a sorted prob distribution</b> — probabilities are sorted descending (bars) and accumulated along the amber curve; the smallest prefix whose cumulative prob just reaches p=0.9 (the first 5) is kept as the nucleus (highlighted region), while the long tail t6–t8 is cut and never sampled.</div>
</div>

<h2>A look at the real code</h2>
<p>Here is the real skeleton of <span class="mono">Sampler.forward</span> (class ~line 68, forward ~line 93). Note the crisp <strong>greedy vs sampling</strong> branch,
and on the standard path the "<strong>temperature → softmax → truncate+sample</strong>" order — matching the pipeline above one-to-one:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/sampler.py ::Sampler</span><span class="ln">logits → the next token</span></div>
  <pre><span class="kw">def</span> forward(self, logits_output, sampling_info, ...):
    logits = logits_output.next_token_logits
    logits = self._preprocess_logits(logits, sampling_info)   <span class="cm"># custom processors + NaN cleanup</span>
    <span class="kw">if</span> sampling_info.is_all_greedy:                           <span class="cm"># whole batch greedy: take the max</span>
        batch_next_token_ids = torch.argmax(logits, -1)
    <span class="kw">else</span>:
        logits.div_(sampling_info.temperatures)               <span class="cm"># 1) temperature scale (per-request, vectorized)</span>
        logits[:] = torch.softmax(logits, dim=-1)             <span class="cm"># 2) turn into probabilities</span>
        probs = logits
        batch_next_token_ids = self._sample_from_probs(       <span class="cm"># 3) top-k/top-p/min-p + multinomial sample</span>
            probs, sampling_info, positions, simple_sampling_case)
    <span class="kw">return</span> batch_next_token_ids</pre>
</div>

<p>Note the details: <span class="mono">sampling_info.temperatures</span> is <strong>a whole tensor</strong> — one temperature per request, so <span class="mono">div_</span> is per-request vectorized,
exactly how "<strong>different requests in one batch use different params</strong>" looks in code. <span class="mono">is_all_greedy</span> is a batch-level shortcut: only when the batch is <strong>entirely</strong> greedy does it take the argmax fast path,
otherwise everything goes the sampling route (greedy requests still get equivalent results there via their own temperature/params). <span class="mono">_sample_from_probs</span> is where top-k/top-p/min-p kernels and the multinomial sample actually run.</p>

<p>Tie it back to the main line: this is <strong>step 4</strong> of the request loop (Lesson 18). model.forward (Lesson 24) produces logits, the Sampler turns them into the next token,
which is <strong>appended back to the Req</strong> and fed into the model next step — exactly the <strong>autoregression</strong> (Lesson 4) loop of "emit one, append, emit again".
Part 6 closes here: Lesson 24 was the ModelRunner "decide → compute" door, Lessons 25/26 how the model loads and is written, Lesson 27 why decode can use CUDA graphs,
and this lesson adds the last link — <strong>how the computed logits actually become the token on your screen</strong>.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Sampler</strong>: an <span class="mono">nn.Module</span> turning <strong>logits → next token id</strong>; reads <span class="mono">SamplingBatchInfo</span> (per-request params packed into tensors).</li>
    <li><strong>Pipeline</strong>: penalties → temperature (÷T; &lt;1 sharp, &gt;1 flat, =0 greedy) → softmax → top-k / top-p / min-p truncation → multinomial sample.</li>
    <li><strong>Knobs</strong>: <span class="mono">SamplingParams</span> sets temperature/top_p/top_k/min_p/penalties per request; <strong>different requests in one batch can use different params</strong> (vectorized).</li>
    <li><strong>Greedy vs sampling</strong>: argmax is deterministic/reproducible vs multinomial is random/diverse; truncation governs quality, temperature governs diversity.</li>
    <li><strong>Hooks</strong>: structured output (Lesson 48) sets disallowed-token logits to −∞ before sampling; logit-bias, min_tokens/EOS suppression, deterministic RNG all hook here. It is step 4 of the request loop (Lesson 18), one turn of autoregression (Lesson 4).</li>
  </ul>
</div>
"""}
