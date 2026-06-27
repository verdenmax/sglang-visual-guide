"""Part 8 - Attention & layers. Lessons (L33-L37) for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons cover the operator layer the model is built from: the pluggable
attention backend (L33), the MoE layer (L34), quantization (L35), RoPE/norm and
other ops (L36), and logits processing + vocab-parallel output (L37).
"""

LESSON_33 = {"zh": r"""
<p class="lead">
第 26 课你会亲手写一个模型，里面有一行 <span class="mono">self.attn(q, k, v, forward_batch)</span>——注意力就发生在这里。
但你写模型时<strong>从不写注意力的 CUDA kernel</strong>，也从不关心它到底跑在 NVIDIA 还是 AMD 上。这一课要讲清楚的，就是这行调用背后的<strong>可替换抽象</strong>：
模型里的注意力层（<span class="inline">RadixAttention</span> 这个 <span class="mono">nn.Module</span>，概念见第 7 课、实现见第 29 课）<strong>本身不含 kernel</strong>，
它把真正的注意力数学<strong>委托</strong>给一个 <span class="mono">AttentionBackend</span>——一个抽象基类（ABC）。换哪个后端、跑哪个 kernel，是<strong>部署时的选择</strong>，不是模型作者的事。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把注意力层想成一把<strong>可换头的电钻</strong>：要干的活（"钻一个孔"＝做注意力）是<strong>固定</strong>的，但你会根据材料<strong>卡上不同的钻头/批头</strong>——
  钻金属用一种、钻木头用另一种。NVIDIA 显卡卡上 <strong>FlashInfer</strong> 钻头，想要到处都能跑就换上 <strong>Triton</strong> 钻头，AMD 机器卡上对应的 AMD 钻头。
  钻头变了，<strong>钻身（模型本体）一行都不用改</strong>。模型只管握着钻身按下扳机 <span class="mono">self.attn(...)</span>，至于此刻是哪个钻头在转，由<strong>用谁来部署</strong>决定。
更妙的是，换钻头不需要把整把钻拆开重装：你不动模型、不动那行调用，只在启动时拨一下 <span class="mono">--attention-backend</span> 这个开关，转动的钻头就换了——这正是"可插拔"四个字最直观的样子。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  记住一句话：<strong>注意力是一个"策略对象"，不是被焊死的代码</strong>。SGLang 把"调用注意力"和"实现注意力"彻底分开——
  前者是模型里那个稳定不变的 <span class="mono">RadixAttention</span> 层，后者是一族可插拔的 <span class="mono">AttentionBackend</span> 实现。
  这正是整个引擎"<strong>一切皆可插拔</strong>"主题的又一个化身：新 kernel、新硬件，只要去实现 <span class="mono">AttentionBackend</span> 的那几个方法就能接进来，
  <strong>不必碰任何一个模型文件</strong>。后端由 <span class="mono">--attention-backend</span> 显式指定，或按硬件/模型<strong>自动</strong>挑选。
</div>

<h2>模型里的层 vs 真正算注意力的后端</h2>
<p>先把两个常被混为一谈的东西分开。<strong>第一个</strong>是模型里的<strong>注意力层</strong>：它是一个 <span class="mono">RadixAttention</span> 的 <span class="mono">nn.Module</span>，
就是第 26 课里你写模型时 <span class="mono">forward</span> 中调用的那个 <span class="mono">self.attn</span>。这个名字你在第 7 课（前缀共享的概念）和第 29 课（基数树的实现）都见过——
此处它指的是"模型前向里被调用的那一层"。关键在于：<strong>这一层里没有任何 kernel</strong>。它持有的是<strong>形状参数</strong>（多少个 query/KV 头、head_dim、缩放系数、层号），
真正的矩阵乘 + softmax + 加权求和这套数学，它一概<strong>委托出去</strong>。</p>
<p><strong>第二个</strong>就是被委托的对象：<span class="mono">AttentionBackend</span>，一个抽象基类。它定义了一份<strong>契约</strong>——
<span class="mono">init_forward_metadata(fb)</span>（每次前向先规划元数据）、<span class="mono">forward_extend(...)</span>（prefill 路径）、<span class="mono">forward_decode(...)</span>（解码路径）。
基类本身大多是 <span class="mono">raise NotImplementedError()</span>，把"具体怎么算"留给子类填。于是模型作者面对的是<strong>稳定的层接口</strong>，
而性能工程师面对的是<strong>可替换的后端实现</strong>，两者在 <span class="mono">AttentionBackend</span> 这条接口线上<strong>解耦</strong>。模型只喊一句 <span class="mono">self.attn(q,k,v,forward_batch)</span>，
背后是谁在转，它毫不知情、也无需知情。</p>
<p>为什么一定要把这两件事掰开？因为它们的<strong>变化频率天差地别</strong>。模型结构相对稳定：一个 Llama、一个 Qwen，写好了就很少动；而注意力的<strong>实现</strong>却日新月异——
今天 FlashInfer 出了更快的版本，明天 FlashAttention 又压低了一档延迟，后天还要支持一种全新的加速卡。如果把易变的实现焊死在稳定的模型里，那每一次 kernel 升级都要去<strong>惊动几十个模型文件</strong>，
风险大、改动散、回归难测。SGLang 的做法是把这条<strong>变化的断层线</strong>正好画在 <span class="mono">AttentionBackend</span> 上：断层之上（模型 + 层）稳定，断层之下（后端 + kernel）自由翻新，互不牵连。
这也是为什么你读模型代码时几乎看不到 CUDA——注意力的全部重活，早就被这层抽象<strong>挡在了模型视野之外</strong>。</p>
<p>还要点破一个容易误会的地方：<span class="mono">RadixAttention</span> 这个名字在第 7 课、第 29 课、和本课出现，指的是<strong>同一个类</strong>，但语境侧重不同。第 7 课讲它<strong>为什么</strong>能让前缀共享，
第 29 课钻进它<strong>怎么</strong>用基数树管理 KV 索引，而本课关心的是它在模型前向里<strong>作为一层被调用</strong>时——把注意力数学<strong>转手</strong>给后端的那一刻。三课拼起来，你才看到这个类的全貌：
它既是缓存索引的持有者，又是注意力调用的入口，<strong>唯独不是注意力 kernel 的实现者</strong>。把"持有索引"和"实现计算"这两件事在同一个名字下分清楚，是读懂这部分代码的关键，也是后面几课不再迷路的地基。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">模型</span><span class="name">某个 Transformer 模型（第 26 课）</span></div><div class="ld">前向里写着 <span class="mono">self.attn(q, k, v, forward_batch)</span>——只调用，不实现。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">层</span><span class="name">RadixAttention（nn.Module，第 29 课）</span></div><div class="ld">持有头数 / head_dim / 缩放 / 层号等<strong>形状参数</strong>；<strong>不含 kernel</strong>，把数学<strong>委托</strong>给后端。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">接口</span><span class="name">AttentionBackend（抽象基类 ABC）</span></div><div class="ld">定义契约：<span class="mono">init_forward_metadata</span> / <span class="mono">forward_extend</span> / <span class="mono">forward_decode</span>。基类只 <span class="mono">raise NotImplementedError</span>。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">实现</span><span class="name">FlashInfer / Triton / FlashAttention 3 / AMD·NPU…</span></div><div class="ld">各自带着真正的 CUDA / Triton <strong>kernel</strong>。由 <span class="mono">--attention-backend</span> 或按硬件自动选定。</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="模型里的 RadixAttention 调用 AttentionBackend 抽象基类，其下扇出到 FlashInfer、Triton、FlashAttention 3、Torch 原生等多个具体后端">
    <rect x="300" y="18" width="200" height="48" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="400" y="38" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">模型 · RadixAttention</text>
    <text x="400" y="56" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:12px">self.attn(q,k,v,fb)</text>
    <line x1="400" y1="66" x2="400" y2="98" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="400,104 393,92 407,92" style="fill:var(--line)"/>
    <rect x="288" y="104" width="224" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--accent);stroke-width:2"/>
    <text x="400" y="125" text-anchor="middle" class="mono" style="font-weight:700">AttentionBackend</text>
    <text x="400" y="144" text-anchor="middle" style="fill:var(--muted);font-size:12px">抽象基类（ABC）· 一份契约</text>
    <line x1="400" y1="156" x2="104" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="400" y1="156" x2="298" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="400" y1="156" x2="492" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="400" y1="156" x2="686" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="14" y="216" width="180" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="104" y="242" text-anchor="middle" style="fill:var(--blue);font-weight:700">FlashInfer</text>
    <text x="104" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">NVIDIA 默认</text>
    <rect x="208" y="216" width="180" height="58" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="298" y="242" text-anchor="middle" style="fill:var(--teal);font-weight:700">Triton</text>
    <text x="298" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">可移植兜底</text>
    <rect x="402" y="216" width="180" height="58" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="492" y="242" text-anchor="middle" style="fill:var(--amber);font-weight:700">FlashAttn 3</text>
    <text x="492" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">FA3 · 高性能</text>
    <rect x="596" y="216" width="180" height="58" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="686" y="242" text-anchor="middle" style="fill:var(--purple);font-weight:700">Torch 原生</text>
    <text x="686" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">纯 PyTorch</text>
  </svg>
  <div class="figcap"><b>图 1 · 模型 → 抽象基类 → 具体后端</b> — 模型里的 <span class="mono">RadixAttention</span> 只调用 <span class="mono">AttentionBackend</span> 这一个抽象基类，其下扇出到多个具体实现（FlashInfer / Triton / FA3 / Torch 原生）；换哪个后端，模型一行都不用改。</div>
</div>

<h2>有哪些后端，它们各自强在哪</h2>
<p>后端是一族而非一个。<strong>FlashInfer</strong> 是高性能 CUDA 实现，在很多 NVIDIA 显卡上是<strong>默认</strong>选择——它把分页 KV 布局、各种掩码、CUDA graph 集成都打磨得很深。
<strong>Triton</strong> 后端用 Triton 语言写成，<strong>可移植性</strong>是它的卖点：在 FlashInfer 还没支持、或硬件/数据类型不匹配时，它是稳妥的<strong>兜底</strong>。
<strong>FlashAttention 3</strong> 是又一族高性能 kernel，在合适的硬件与场景上能再压一截延迟。除此之外还有<strong>硬件专属</strong>的后端——比如面向 AMD、NPU 的实现（多硬件话题见第 42 课）。
选哪个，要么你用 <span class="mono">--attention-backend</span> 一锤定音，要么引擎按你的<strong>硬件型号、模型结构、数据类型</strong>替你自动判定。</p>
<p>把这张表记在心里：它们提供的是<strong>同一份契约的不同实现</strong>——同样接收 q/k/v 和 <span class="mono">forward_batch</span>，同样返回注意力输出，只是内部 kernel 不同、适配的硬件不同。
正因为契约一致，<strong>切后端不需要改模型</strong>；正因为实现可换，<strong>新 kernel、新硬件能独立演进</strong>。这就是"接口"二字的全部价值。</p>
<p>那么"自动挑选"到底依据什么？大体三件事：一是<strong>硬件</strong>——你跑在什么型号的 GPU/加速卡上，决定了哪些 kernel 可用、哪个最快；二是<strong>模型结构</strong>——
有的模型用 MHA、有的用 GQA/MLA，head_dim 和头数不同，并非每个后端都覆盖每种形状；三是<strong>数据类型</strong>——fp16、bf16、fp8 各有各的 kernel 支持矩阵。
引擎把这三者一拼，挑出一个既<strong>能跑</strong>又<strong>尽量快</strong>的默认后端；而你随时可以用 <span class="mono">--attention-backend</span> 把这个自动决定<strong>覆盖掉</strong>，强制指定某个后端——
比如在排查问题时切到可移植的 Triton 当对照基准，确认是不是某个高性能 kernel 的边界情况。这种"<strong>默认自动、允许手动</strong>"的设计，正是可插拔抽象给运维留下的灵活度。</p>

<table class="t">
  <tr><th>后端</th><th>强项</th><th>什么时候用</th></tr>
  <tr><td class="mono">FlashInfer</td><td>高性能 CUDA，分页 KV / 掩码 / CUDA graph 打磨深</td><td>许多 NVIDIA 显卡上的<strong>默认</strong></td></tr>
  <tr><td class="mono">Triton</td><td>用 Triton 写，<strong>可移植</strong>、兼容面广</td><td>通用<strong>兜底</strong>；FlashInfer 不支持的硬件/类型</td></tr>
  <tr><td class="mono">FlashAttention 3</td><td>又一族高性能 kernel</td><td>合适硬件上进一步压低延迟</td></tr>
  <tr><td class="mono">AMD / NPU…</td><td>硬件专属实现</td><td>非 NVIDIA 硬件（第 42 课）</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 280" role="img" aria-label="部署时按硬件与模型选后端：Hopper 加通用模型选 FlashInfer 或 FA3，AMD 或其他硬件选 Triton，兜底选 Torch 原生，由 --attention-backend 指定或自动">
    <text x="24" y="30" style="fill:var(--muted);font-size:13px">部署时由 <tspan class="mono" style="fill:var(--ink)">--attention-backend</tspan> 选定（或按硬件自动）</text>
    <rect x="24" y="54" width="250" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="149" y="86" text-anchor="middle" style="fill:var(--ink)">Hopper / 通用模型</text>
    <rect x="24" y="128" width="250" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="149" y="160" text-anchor="middle" style="fill:var(--ink)">AMD / 其他硬件</text>
    <rect x="24" y="202" width="250" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="149" y="234" text-anchor="middle" style="fill:var(--ink)">兜底（任意硬件）</text>
    <line x1="274" y1="80" x2="500" y2="80" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="506,80 494,74 494,86" style="fill:var(--line)"/>
    <line x1="274" y1="154" x2="500" y2="154" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="506,154 494,148 494,160" style="fill:var(--line)"/>
    <line x1="274" y1="228" x2="500" y2="228" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="506,228 494,222 494,234" style="fill:var(--line)"/>
    <rect x="506" y="54" width="270" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="641" y="86" text-anchor="middle" style="fill:var(--blue);font-weight:700">FlashInfer / FA3</text>
    <rect x="506" y="128" width="270" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="641" y="160" text-anchor="middle" style="fill:var(--teal);font-weight:700">Triton</text>
    <rect x="506" y="202" width="270" height="52" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="641" y="234" text-anchor="middle" style="fill:var(--purple);font-weight:700">Torch 原生</text>
  </svg>
  <div class="figcap"><b>图 2 · 部署时按硬件/模型选后端</b> — 启动时用 <span class="mono">--attention-backend</span> 指定（或自动）：Hopper＋通用模型 → FlashInfer/FA3；AMD/其他 → Triton；兜底 → Torch 原生。换的是后端，模型代码一行不动。</div>
</div>

<h2>一次前向：先规划元数据，再分派 EXTEND / DECODE 两条路</h2>
<p>后端每次前向具体做两件事。<strong>第一件是"规划元数据"（plan metadata）</strong>：在真正调 kernel 之前，后端要先算清楚这一批请求的<strong>读写地图</strong>——
每条请求要从 KV 池（第 30 课）的<strong>哪些页</strong>读取已缓存的 K/V、<strong>因果掩码</strong>长什么样、各条<strong>序列长度</strong>是多少、CUDA graph（第 27 课）需要的固定形状缓冲怎么填。
这一步不算注意力，但它把"接下来 kernel 要怎么读内存"准备好，所以叫"规划"。<strong>第二件</strong>才是<strong>跑对应的 kernel</strong>。</p>
<p>而"对应的 kernel"有<strong>两条截然不同的路径</strong>，这是本课最该记住的一点。<strong>EXTEND / prefill</strong>：一批<strong>新 token</strong>同时进来（比如刚收到的 prompt），
它们之间要做<strong>完整的注意力</strong>——每个新 token 既看自己也看前面的新 token（受因果掩码约束）。<strong>DECODE</strong>：自回归生成阶段，<strong>只有 1 个 query token</strong>，
它要注意<strong>整段已缓存的 KV</strong>（历史所有 token）。两条路的算术形状、访存模式、最优 kernel 都不一样，所以后端为它们准备<strong>分开的实现</strong>——
<span class="mono">forward_extend</span> 和 <span class="mono">forward_decode</span>。统一入口 <span class="mono">forward</span> 看 <span class="mono">forward_batch.forward_mode</span> 判断走哪条，再分派下去。
顺带一提，正是 DECODE 这条"1 个 query、固定形状"的路径，最适合用 CUDA graph（第 27 课）把启动开销吃掉。</p>
<p>再把"规划元数据"讲得更实一点，你就明白它为什么不可省。注意力 kernel 要读 KV，而 KV 在池里是<strong>分页存放</strong>的（第 30 课）——同一条请求的历史 token，其 K/V 可能散落在<strong>互不相邻的若干页</strong>里。
kernel 自己并不知道"这条请求该读哪些页"，必须有人先把这张<strong>页表</strong>算好、连同每条请求的序列长度、因果掩码的形状一起，整理成 kernel 能直接吃的张量。这就是"规划"在做的事：
它把"逻辑上的 token 序列"翻译成"物理上的内存地址清单"。在用 CUDA graph 时这一步尤其讲究——graph 要求形状固定、地址稳定，所以元数据里那些<strong>会变的部分</strong>（每次 batch 不同的页号、长度）
要小心地填进预留好的固定缓冲里，而不能每次新建张量。理解了这层，你就明白为什么后端代码里 <span class="mono">init_forward_metadata</span> 还要再分成 out-graph 与 in-graph 两半——
一半干"不能被录进图的活"（动态形状、host 端计算），一半干"能被录进图、之后自动重放的活"。规划做对了，kernel 才能闷头快跑。</p>

<div class="cols">
  <div class="col"><h4>EXTEND / prefill（forward_extend）</h4><p>一批<strong>新 token</strong>一起进来（如刚到的 prompt）。它们做<strong>完整注意力</strong>：每个新 token 看自己 + 前面的新 token，受<strong>因果掩码</strong>约束。算术量大、并行度高，是"读一大片、算一大片"。</p></div>
  <div class="col"><h4>DECODE（forward_decode）</h4><p>自回归生成，<strong>只有 1 个 query token</strong>。它注意<strong>整段已缓存的 KV</strong>（历史全部 token，来自第 30 课的池）。形状固定、批内规整，最适合 <strong>CUDA graph</strong>（第 27 课）固化启动开销。</p></div>
</div>

<p>把一次注意力调用从头到尾串起来，就是下面这条流水：模型喊一声，后端先规划、再按模式挑路径、跑 kernel、把输出交回模型继续往下算。</p>

<div class="flow">
  <div class="node"><div class="nt">self.attn(...)</div><div class="nd">模型层发起调用<br>（第 26 课）</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">规划元数据</div><div class="nd">读哪些 KV 页、掩码、<br>序列长度（第 30 课）</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">挑路径</div><div class="nd">看 forward_mode：<br>EXTEND 还是 DECODE</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">跑 kernel</div><div class="nd">FlashInfer / Triton…<br>真正的矩阵乘 + softmax</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">输出</div><div class="nd">注意力结果<br>交回模型继续前向</div></div>
</div>

<h2>为什么要做成一个接口</h2>
<p>把这一切收束成一个问题：为什么不直接在模型里写死注意力，而要绕一层抽象基类？答案就是这一课的灵魂——<strong>让新 kernel 和新硬件能插进来，而不必改任何模型文件</strong>。
SGLang 支持几十个模型（第 26 课），如果注意力 kernel 写死在每个模型里，那么每出一个更快的 kernel、每支持一种新加速卡，你都得去<strong>逐个模型文件</strong>改一遍——这是维护灾难。
反过来，把注意力定义成 <span class="mono">AttentionBackend</span> 这份契约后，加一个新后端＝<strong>新写一个子类、实现那几个方法</strong>，模型文件<strong>一行不动</strong>就能享受到它。</p>
<p>这正与引擎别处反复出现的设计哲学同构：注意力是一个<strong>策略对象</strong>，和 KV 缓存策略、调度策略、量化策略（第 35 课）一样，都被抽象成"可替换的零件"。
你写模型时只依赖<strong>稳定的层接口</strong>，把"跑得多快、跑在什么硬件上"这些<strong>易变的、和部署强相关的</strong>决定，留给后端去承担。
更底层的 kernel 本身怎么写（第 38/40 课）、多硬件适配的细节（第 42 课），都在这条接口线之下独立演化，<strong>对模型完全透明</strong>。这就是抽象的回报：上层稳定，下层自由。</p>
<p>最后用一句话把整课收束起来：<strong>注意力是"做什么"固定、"怎么做"可换的一道工序</strong>。"做什么"——给定 q/k/v 和这一批的读写地图，算出注意力输出——由 <span class="mono">AttentionBackend</span> 的契约钉死，
谁来实现都必须满足；"怎么做"——用哪个 kernel、读分页 KV 的哪条访存路径、要不要套 CUDA graph、跑在 NVIDIA 还是 AMD——则完全交给具体后端，按硬件和场景各显神通。
模型作者站在"做什么"这一侧，享受永不变的 <span class="mono">self.attn(...)</span>；性能与硬件工程师站在"怎么做"那一侧，自由迭代而不惊动任何模型。这条清晰的分工线，正是 SGLang 能<strong>同时</strong>支持几十个模型、又<strong>同时</strong>拥抱层出不穷的新 kernel 和新加速卡的根本原因。</p>

<p>下面是这份契约的真身——<span class="mono">AttentionBackend</span> 抽象基类。注意 <span class="mono">forward</span> 如何按 <span class="mono">forward_mode</span> 把活分派到 <span class="mono">forward_decode</span> / <span class="mono">forward_extend</span> 两条路，而它们在基类里都是 <span class="mono">NotImplementedError</span>，等子类来填：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/base_attn_backend.py ::AttentionBackend</span><span class="ln">委托契约：规划元数据 + 两条前向路径</span></div>
  <pre><span class="kw">class</span> <span class="st">AttentionBackend</span>(ABC):
    <span class="cm"># 所有注意力后端的基类。模型只调用它，不关心内部跑哪个 kernel</span>

    <span class="kw">def</span> init_forward_metadata(self, forward_batch):
        <span class="cm"># 每次前向先“规划元数据”：每条请求读 KV 池的哪些页、掩码、序列长度…</span>
        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)

    <span class="kw">def</span> forward(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        <span class="cm"># 统一入口：按 forward_mode 分派到 decode / extend 两条不同路径</span>
        <span class="kw">if</span> forward_batch.forward_mode.is_decode():
            <span class="kw">return</span> self.forward_decode(q, k, v, layer, forward_batch)
        <span class="kw">else</span>:
            <span class="kw">return</span> self.forward_extend(q, k, v, layer, forward_batch)

    <span class="kw">def</span> forward_decode(self, q, k, v, layer, forward_batch, ...):
        <span class="cm"># DECODE 路径：1 个 query token 注意整段已缓存的 KV</span>
        <span class="kw">raise</span> NotImplementedError()   <span class="cm"># 由具体后端（FlashInfer / Triton…）实现</span>

    <span class="kw">def</span> forward_extend(self, q, k, v, layer, forward_batch, ...):
        <span class="cm"># EXTEND / prefill 路径：一批新 token 做完整注意力</span>
        <span class="kw">raise</span> NotImplementedError()</pre>
</div>

<p>抽象基类只立下契约，真正干活的是<strong>具体后端</strong>。下面是 <span class="mono">FlashInferAttnBackend</span>——它<strong>继承</strong> <span class="mono">AttentionBackend</span>，把那三个方法<strong>填上 FlashInfer 的真实实现</strong>：<span class="mono">init_forward_metadata</span> 建好 FlashInfer 需要的 wrapper/索引，<span class="mono">forward_extend</span> 走 prefill，<span class="mono">forward_decode</span> 走解码。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/flashinfer_backend.py ::FlashInferAttnBackend</span><span class="ln">具体后端：用 FlashInfer 实现抽象基类的契约</span></div>
  <pre><span class="kw">class</span> <span class="st">FlashInferAttnBackend</span>(AttentionBackend):
    <span class="cm"># 面向 NVIDIA GPU 的具体后端，基于 FlashInfer</span>
    <span class="cm"># 库（分页 KV + 快速 prefill/decode kernel）。</span>
    <span class="kw">def</span> init_forward_metadata(self, forward_batch):
        <span class="cm"># 构建 FlashInfer 每步所需的 wrapper / 索引。</span>
        ...
    <span class="kw">def</span> forward_extend(self, q, k, v, layer, forward_batch):
        ...   <span class="cm"># prefill：对整段新 token 做注意力</span>
    <span class="kw">def</span> forward_decode(self, q, k, v, layer, forward_batch):
        ...   <span class="cm"># decode：1 个新 query token 对已缓存 K/V</span></pre>
</div>

<p>具体到命令行：<span class="mono">--attention-backend flashinfer</span> 就挑中了上面这个实现，换成 <span class="mono">--attention-backend triton</span> 或 <span class="mono">fa3</span> 则换上另一族 kernel——<strong>模型代码一个字都不用动</strong>。因为模型前向里永远只有那一句 <span class="mono">self.attn(...)</span>，被换掉的后端只是悄悄在它底下转动。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <strong>① 模型里的注意力层（RadixAttention，第 29 课）不含 kernel</strong>——它持有形状参数，把真正的注意力数学<strong>委托</strong>给一个 <span class="mono">AttentionBackend</span>（ABC）。
  <strong>② 后端是一族可换实现</strong>：FlashInfer（NV 高性能默认）、Triton（可移植兜底）、FlashAttention 3、以及 AMD/NPU 等硬件专属（第 42 课）；由 <span class="mono">--attention-backend</span> 或按硬件自动选。
  <strong>③ 后端每次前向先"规划元数据"</strong>（读哪些 KV 页、掩码、序列长度，第 30 课），再分派到 <strong>EXTEND/prefill</strong>（新 token 做完整注意力）或 <strong>DECODE</strong>（1 个 query 注意整段缓存 KV）两条独立路径，并集成分页 KV + CUDA graph（第 27 课）。
  <strong>④ 做成接口，是为了让新 kernel/新硬件插进来而不必碰任何模型文件</strong>（第 26 课）——注意力是个<strong>策略对象</strong>。模型只喊 <span class="mono">self.attn(q,k,v,forward_batch)</span>，跑哪个后端是部署选择。kernel 细节见第 38/40 课。
</div>
""",
             "en": r"""
<p class="lead">
In Lesson 26 you'll write a model by hand, and inside it sits one line — <span class="mono">self.attn(q, k, v, forward_batch)</span> — where attention happens.
Yet you <strong>never write the attention CUDA kernel</strong>, and you never care whether it runs on NVIDIA or AMD. This lesson is about the <strong>swappable abstraction</strong> behind that call:
the model's attention layer (the <span class="inline">RadixAttention</span> <span class="mono">nn.Module</span>, concept in Lesson 7, implementation in Lesson 29) <strong>contains no kernel</strong>;
it <strong>delegates</strong> the actual attention math to an <span class="mono">AttentionBackend</span> — an abstract base class (ABC). Which backend, which kernel, is a <strong>deployment choice</strong>, not the model author's job.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the attention layer as a <strong>power drill with interchangeable bits</strong>: the job ("make a hole" = compute attention) is <strong>fixed</strong>, but you <strong>snap in a different bit</strong> for the material —
  one for metal, another for wood. On NVIDIA you snap in the <strong>FlashInfer</strong> bit; for run-anywhere portability you swap in the <strong>Triton</strong> bit; on an AMD box you fit the matching AMD bit.
  The bit changes, but <strong>the drill body (the model itself) needs zero edits</strong>. The model just holds the body and pulls the trigger <span class="mono">self.attn(...)</span>; which bit is spinning is decided by <strong>who deploys it</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Remember one line: <strong>attention is a "strategy object," not welded-in code</strong>. SGLang cleanly separates "calling attention" from "implementing attention" —
  the former is the stable <span class="mono">RadixAttention</span> layer inside the model, the latter is a family of pluggable <span class="mono">AttentionBackend</span> implementations.
  This is another incarnation of the engine's "<strong>everything pluggable</strong>" theme: a new kernel or new hardware plugs in just by implementing the few <span class="mono">AttentionBackend</span> methods,
  <strong>without touching a single model file</strong>. The backend is chosen explicitly by <span class="mono">--attention-backend</span>, or <strong>auto</strong>-selected by hardware/model.
</div>

<h2>The layer in the model vs. the backend that does the math</h2>
<p>First separate two things people conflate. <strong>One</strong> is the model's <strong>attention layer</strong>: a <span class="mono">RadixAttention</span> <span class="mono">nn.Module</span>,
the very <span class="mono">self.attn</span> you call inside a model's <span class="mono">forward</span> in Lesson 26. You've met this name in Lesson 7 (the prefix-sharing concept) and Lesson 29 (the radix-tree implementation) —
here it means "the layer the model calls in its forward." The key point: <strong>this layer holds no kernel</strong>. It carries <strong>shape parameters</strong> (how many query/KV heads, head_dim, the scale, the layer id);
the actual matmul + softmax + weighted sum it <strong>delegates away</strong>.</p>
<p><strong>Two</strong> is the thing it delegates to: <span class="mono">AttentionBackend</span>, an abstract base class. It defines a <strong>contract</strong> —
<span class="mono">init_forward_metadata(fb)</span> (plan metadata each forward), <span class="mono">forward_extend(...)</span> (the prefill path), <span class="mono">forward_decode(...)</span> (the decode path).
The base class is mostly <span class="mono">raise NotImplementedError()</span>, leaving "how to actually compute" to subclasses. So the model author faces a <strong>stable layer interface</strong>,
while the performance engineer faces a <strong>swappable backend implementation</strong>, the two <strong>decoupled</strong> along the <span class="mono">AttentionBackend</span> line. The model just shouts <span class="mono">self.attn(q,k,v,forward_batch)</span>,
blissfully unaware of who is spinning behind it — and it needn't know.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Model</span><span class="name">some Transformer model (Lesson 26)</span></div><div class="ld">its forward says <span class="mono">self.attn(q, k, v, forward_batch)</span> — calls, never implements.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">Layer</span><span class="name">RadixAttention (nn.Module, Lesson 29)</span></div><div class="ld">holds <strong>shape params</strong> (heads / head_dim / scale / layer id); <strong>no kernel</strong>, <strong>delegates</strong> the math to a backend.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Interface</span><span class="name">AttentionBackend (abstract base class, ABC)</span></div><div class="ld">defines the contract: <span class="mono">init_forward_metadata</span> / <span class="mono">forward_extend</span> / <span class="mono">forward_decode</span>. Base is just <span class="mono">raise NotImplementedError</span>.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Impls</span><span class="name">FlashInfer / Triton / FlashAttention 3 / AMD·NPU…</span></div><div class="ld">each carries the real CUDA / Triton <strong>kernel</strong>. Chosen by <span class="mono">--attention-backend</span> or auto by hardware.</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="The model's RadixAttention calls the AttentionBackend abstract base class, which fans out to concrete backends: FlashInfer, Triton, FlashAttention 3, Torch-native">
    <rect x="300" y="18" width="200" height="48" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="400" y="38" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">Model · RadixAttention</text>
    <text x="400" y="56" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:12px">self.attn(q,k,v,fb)</text>
    <line x1="400" y1="66" x2="400" y2="98" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="400,104 393,92 407,92" style="fill:var(--line)"/>
    <rect x="288" y="104" width="224" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--accent);stroke-width:2"/>
    <text x="400" y="125" text-anchor="middle" class="mono" style="font-weight:700">AttentionBackend</text>
    <text x="400" y="144" text-anchor="middle" style="fill:var(--muted);font-size:12px">abstract base class (ABC) · one contract</text>
    <line x1="400" y1="156" x2="104" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="400" y1="156" x2="298" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="400" y1="156" x2="492" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="400" y1="156" x2="686" y2="214" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="14" y="216" width="180" height="58" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="104" y="242" text-anchor="middle" style="fill:var(--blue);font-weight:700">FlashInfer</text>
    <text x="104" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">NVIDIA default</text>
    <rect x="208" y="216" width="180" height="58" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="298" y="242" text-anchor="middle" style="fill:var(--teal);font-weight:700">Triton</text>
    <text x="298" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">portable fallback</text>
    <rect x="402" y="216" width="180" height="58" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="492" y="242" text-anchor="middle" style="fill:var(--amber);font-weight:700">FlashAttn 3</text>
    <text x="492" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">FA3 · high-perf</text>
    <rect x="596" y="216" width="180" height="58" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="686" y="242" text-anchor="middle" style="fill:var(--purple);font-weight:700">Torch-native</text>
    <text x="686" y="261" text-anchor="middle" style="fill:var(--muted);font-size:11px">pure PyTorch</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Model → ABC → concrete backends</b> — the model's <span class="mono">RadixAttention</span> only calls one abstract base class, <span class="mono">AttentionBackend</span>, which fans out to several concrete implementations (FlashInfer / Triton / FA3 / Torch-native); swapping a backend changes not a line of the model.</div>
</div>

<h2>Which backends exist, and what each is good at</h2>
<p>Backends are a family, not a single thing. <strong>FlashInfer</strong> is a high-performance CUDA implementation, the <strong>default</strong> on many NVIDIA GPUs — it polishes paged-KV layout, masks, and CUDA-graph integration deeply.
The <strong>Triton</strong> backend is written in Triton; <strong>portability</strong> is its selling point: when FlashInfer doesn't yet support a case, or the hardware/dtype doesn't match, it's the safe <strong>fallback</strong>.
<strong>FlashAttention 3</strong> is another family of high-performance kernels that can shave off more latency on the right hardware and shapes. Beyond these there are <strong>hardware-specific</strong> backends — e.g. for AMD and NPU (multi-hardware is Lesson 42).
To pick one, either you nail it down with <span class="mono">--attention-backend</span>, or the engine auto-decides from your <strong>GPU model, model architecture, and dtype</strong>.</p>
<p>Keep this table in mind: they are <strong>different implementations of the same contract</strong> — all take q/k/v and <span class="mono">forward_batch</span>, all return the attention output, only the inner kernel and target hardware differ.
Because the contract is identical, <strong>swapping backends needs no model change</strong>; because the implementation is swappable, <strong>new kernels and new hardware can evolve independently</strong>. That is the whole value of an "interface."</p>

<table class="t">
  <tr><th>Backend</th><th>Strength</th><th>When used</th></tr>
  <tr><td class="mono">FlashInfer</td><td>high-perf CUDA; deep paged-KV / mask / CUDA-graph polish</td><td><strong>default</strong> on many NVIDIA GPUs</td></tr>
  <tr><td class="mono">Triton</td><td>written in Triton, <strong>portable</strong>, broad compatibility</td><td>general <strong>fallback</strong>; hardware/dtypes FlashInfer can't cover</td></tr>
  <tr><td class="mono">FlashAttention 3</td><td>another family of high-perf kernels</td><td>shave more latency on suitable hardware</td></tr>
  <tr><td class="mono">AMD / NPU…</td><td>hardware-specific implementations</td><td>non-NVIDIA hardware (Lesson 42)</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 280" role="img" aria-label="At deploy the backend is chosen by hardware and model: Hopper plus a general model picks FlashInfer or FA3, AMD or other hardware picks Triton, fallback picks Torch-native, selected by --attention-backend or auto">
    <text x="24" y="30" style="fill:var(--muted);font-size:13px">at deploy, <tspan class="mono" style="fill:var(--ink)">--attention-backend</tspan> selects (or auto by hardware)</text>
    <rect x="24" y="54" width="250" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="149" y="86" text-anchor="middle" style="fill:var(--ink)">Hopper / general model</text>
    <rect x="24" y="128" width="250" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="149" y="160" text-anchor="middle" style="fill:var(--ink)">AMD / other hardware</text>
    <rect x="24" y="202" width="250" height="52" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="149" y="234" text-anchor="middle" style="fill:var(--ink)">fallback (any hardware)</text>
    <line x1="274" y1="80" x2="500" y2="80" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="506,80 494,74 494,86" style="fill:var(--line)"/>
    <line x1="274" y1="154" x2="500" y2="154" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="506,154 494,148 494,160" style="fill:var(--line)"/>
    <line x1="274" y1="228" x2="500" y2="228" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="506,228 494,222 494,234" style="fill:var(--line)"/>
    <rect x="506" y="54" width="270" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="641" y="86" text-anchor="middle" style="fill:var(--blue);font-weight:700">FlashInfer / FA3</text>
    <rect x="506" y="128" width="270" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="641" y="160" text-anchor="middle" style="fill:var(--teal);font-weight:700">Triton</text>
    <rect x="506" y="202" width="270" height="52" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="641" y="234" text-anchor="middle" style="fill:var(--purple);font-weight:700">Torch-native</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Backend chosen at deploy by hardware/model</b> — at startup <span class="mono">--attention-backend</span> selects (or auto): Hopper + general model → FlashInfer/FA3; AMD/other → Triton; fallback → Torch-native. What changes is the backend, never a line of model code.</div>
</div>

<h2>One forward: plan metadata, then dispatch EXTEND / DECODE</h2>
<p>Each forward, a backend does two things. <strong>First, "plan metadata."</strong> Before calling any kernel, the backend works out this batch's <strong>read/write map</strong> —
which <strong>pages</strong> of the KV pool (Lesson 30) each request reads its cached K/V from, what the <strong>causal mask</strong> looks like, each <strong>sequence length</strong>, and how to fill the fixed-shape buffers a CUDA graph (Lesson 27) needs.
This step computes no attention, but it prepares "how the kernel will read memory next," hence "planning." <strong>Second</strong> comes <strong>running the right kernel</strong>.</p>
<p>And the "right kernel" has <strong>two distinct paths</strong> — the single most important takeaway here. <strong>EXTEND / prefill</strong>: a batch of <strong>new tokens</strong> arrives together (e.g. a freshly received prompt),
and they do <strong>full attention</strong> — each new token attends to itself and the new tokens before it (under the causal mask). <strong>DECODE</strong>: the autoregressive generation step, with <strong>just 1 query token</strong>,
which attends to the <strong>entire cached KV</strong> (all historical tokens). The two paths differ in arithmetic shape, memory-access pattern, and optimal kernel, so the backend keeps <strong>separate implementations</strong> —
<span class="mono">forward_extend</span> and <span class="mono">forward_decode</span>. The unified entry <span class="mono">forward</span> reads <span class="mono">forward_batch.forward_mode</span> to decide which path, then dispatches.
Incidentally, it's exactly the DECODE path — "1 query, fixed shape" — that's the perfect fit for a CUDA graph (Lesson 27) to swallow launch overhead.</p>

<div class="cols">
  <div class="col"><h4>EXTEND / prefill (forward_extend)</h4><p>A batch of <strong>new tokens</strong> arrives together (e.g. a fresh prompt). They do <strong>full attention</strong>: each new token sees itself + the new tokens before it, under the <strong>causal mask</strong>. Heavy arithmetic, high parallelism — "read a big chunk, compute a big chunk."</p></div>
  <div class="col"><h4>DECODE (forward_decode)</h4><p>Autoregressive generation, <strong>just 1 query token</strong>. It attends to the <strong>entire cached KV</strong> (all historical tokens, from Lesson 30's pool). Fixed shape, regular across the batch — the perfect fit for a <strong>CUDA graph</strong> (Lesson 27) to fix launch overhead.</p></div>
</div>

<p>Stringing one attention call end to end gives this pipeline: the model shouts, the backend plans, picks a path by mode, runs the kernel, and hands the output back to the model to continue.</p>

<div class="flow">
  <div class="node"><div class="nt">self.attn(...)</div><div class="nd">model layer calls<br>(Lesson 26)</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">plan metadata</div><div class="nd">which KV pages, mask,<br>seq lengths (Lesson 30)</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">pick path</div><div class="nd">read forward_mode:<br>EXTEND or DECODE</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">run kernel</div><div class="nd">FlashInfer / Triton…<br>the real matmul + softmax</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">output</div><div class="nd">attention result<br>back to the model</div></div>
</div>

<h2>Why make it an interface</h2>
<p>Collapse it all into one question: why not hardcode attention in the model — why route through an ABC? The answer is this lesson's soul: <strong>so new kernels and new hardware can plug in without touching any model file</strong>.
SGLang supports dozens of models (Lesson 26). If the attention kernel were hardcoded into each model, then every faster kernel and every new accelerator would force you to edit <strong>every model file</strong> — a maintenance disaster.
Conversely, once attention is defined as the <span class="mono">AttentionBackend</span> contract, adding a new backend = <strong>writing one subclass and implementing those few methods</strong>, and model files <strong>change not a line</strong> yet benefit from it.</p>
<p>This mirrors a design philosophy that recurs all over the engine: attention is a <strong>strategy object</strong>, just like the KV-cache strategy, the scheduling strategy, and the quantization strategy (Lesson 35) — all abstracted into "swappable parts."
You write your model depending only on a <strong>stable layer interface</strong>, leaving the <strong>volatile, deployment-specific</strong> decisions of "how fast, on what hardware" to the backend.
How the lower-level kernels are written (Lessons 38/40) and the details of multi-hardware support (Lesson 42) evolve independently below this interface line, <strong>fully transparent to the model</strong>. That's the payoff of abstraction: stable on top, free below.</p>

<p>Below is the contract itself — the <span class="mono">AttentionBackend</span> ABC. Note how <span class="mono">forward</span> dispatches by <span class="mono">forward_mode</span> to <span class="mono">forward_decode</span> / <span class="mono">forward_extend</span>, both of which are <span class="mono">NotImplementedError</span> in the base, waiting for a subclass:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/base_attn_backend.py ::AttentionBackend</span><span class="ln">the delegation contract: plan metadata + two forward paths</span></div>
  <pre><span class="kw">class</span> <span class="st">AttentionBackend</span>(ABC):
    <span class="cm"># Base class of all attention backends. The model only calls it; it</span>
    <span class="cm"># doesn't care which kernel runs inside.</span>

    <span class="kw">def</span> init_forward_metadata(self, forward_batch):
        <span class="cm"># Each forward, first “plan metadata”: which KV pages each request</span>
        <span class="cm"># reads, the mask, sequence lengths…</span>
        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)

    <span class="kw">def</span> forward(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        <span class="cm"># Unified entry: dispatch by forward_mode to decode / extend paths</span>
        <span class="kw">if</span> forward_batch.forward_mode.is_decode():
            <span class="kw">return</span> self.forward_decode(q, k, v, layer, forward_batch)
        <span class="kw">else</span>:
            <span class="kw">return</span> self.forward_extend(q, k, v, layer, forward_batch)

    <span class="kw">def</span> forward_decode(self, q, k, v, layer, forward_batch, ...):
        <span class="cm"># DECODE path: 1 query token attends the whole cached KV</span>
        <span class="kw">raise</span> NotImplementedError()   <span class="cm"># filled by a concrete backend</span>

    <span class="kw">def</span> forward_extend(self, q, k, v, layer, forward_batch, ...):
        <span class="cm"># EXTEND / prefill path: a batch of new tokens, full attention</span>
        <span class="kw">raise</span> NotImplementedError()</pre>
</div>

<p>The ABC only lays down the contract; the real work is done by a <strong>concrete backend</strong>. Below is <span class="mono">FlashInferAttnBackend</span> — it <strong>subclasses</strong> <span class="mono">AttentionBackend</span> and <strong>fills those three methods with FlashInfer's real implementation</strong>: <span class="mono">init_forward_metadata</span> builds the wrappers/indices FlashInfer needs, <span class="mono">forward_extend</span> takes the prefill path, <span class="mono">forward_decode</span> takes the decode path.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/flashinfer_backend.py ::FlashInferAttnBackend</span><span class="ln">a concrete backend: implements the ABC contract via FlashInfer</span></div>
  <pre><span class="kw">class</span> <span class="st">FlashInferAttnBackend</span>(AttentionBackend):
    <span class="cm"># a CONCRETE backend for NVIDIA GPUs, built on the FlashInfer</span>
    <span class="cm"># library (paged KV + fast prefill/decode kernels).</span>
    <span class="kw">def</span> init_forward_metadata(self, forward_batch):
        <span class="cm"># build the per-step wrappers/indices FlashInfer needs.</span>
        ...
    <span class="kw">def</span> forward_extend(self, q, k, v, layer, forward_batch):
        ...   <span class="cm"># prefill: attend over the whole new chunk</span>
    <span class="kw">def</span> forward_decode(self, q, k, v, layer, forward_batch):
        ...   <span class="cm"># decode: one new query token vs cached K/V</span></pre>
</div>

<p>Concretely on the command line: <span class="mono">--attention-backend flashinfer</span> picks the implementation above, while <span class="mono">--attention-backend triton</span> or <span class="mono">fa3</span> swaps in another family of kernels — <strong>without touching a single word of model code</strong>. Because the model forward only ever has that one line <span class="mono">self.attn(...)</span>, the swapped backend simply spins underneath it.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <strong>① The model's attention layer (RadixAttention, Lesson 29) holds no kernel</strong> — it carries shape params and <strong>delegates</strong> the real attention math to an <span class="mono">AttentionBackend</span> (ABC).
  <strong>② Backends are a swappable family</strong>: FlashInfer (NV high-perf default), Triton (portable fallback), FlashAttention 3, plus hardware-specific ones like AMD/NPU (Lesson 42); chosen by <span class="mono">--attention-backend</span> or auto by hardware.
  <strong>③ Each forward the backend first "plans metadata"</strong> (which KV pages, mask, seq lengths, Lesson 30), then dispatches to <strong>EXTEND/prefill</strong> (new tokens, full attention) or <strong>DECODE</strong> (1 query over the whole cached KV) — separate paths — integrating paged KV + CUDA graph (Lesson 27).
  <strong>④ Making it an interface lets new kernels/hardware plug in without touching any model file</strong> (Lesson 26) — attention is a <strong>strategy object</strong>. The model just calls <span class="mono">self.attn(q,k,v,forward_batch)</span>; which backend runs is a deployment choice. Kernel details in Lessons 38/40.
</div>
"""}

LESSON_34 = {"zh": r"""
<p class="lead">
第 26 课你会亲手写一个模型，里面那行 <span class="mono">self.mlp(x)</span> 在很多新模型里早已不是一个普通的多层感知机，而是一个 <strong>MoE 层</strong>（Mixture of Experts，混合专家）。
DeepSeek-V3、Mixtral、Qwen-MoE 都用它。它的核心反直觉之处在于：<strong>模型参数可以涨到几十上百倍，而每个 token 实际花掉的算力几乎不变</strong>。
这一课要讲清楚的，就是这件"看似免费的午餐"是怎么做到的——靠的是把一个大 FFN 拆成<strong>很多个小专家</strong>，再加一个<strong>路由器</strong>，让每个 token 只走其中<strong>极少数几个</strong>专家。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 MoE 层想成一家大医院的<strong>分诊台</strong>。医院里坐着<strong>几十位专科医生</strong>（专家），但每位病人（token）来了，分诊护士（路由器）<strong>不会</strong>让他挨个看遍所有科室——
  那样既慢又浪费。护士只看一眼症状，<strong>挑出最相关的两位专科</strong>（top-2），比如"心内科 + 内分泌科"，病人只去这两间诊室。
  于是整家医院<strong>专科越开越多、总知识越来越厚</strong>，但<strong>每个病人的就诊时间几乎不变</strong>——他始终只看两位医生。这正是 MoE 的灵魂：用"<strong>稀疏</strong>"换"<strong>更大的总容量</strong>"，
  而不是让每个人都为这份庞大买单。最后再把两位医生的意见<strong>按相关度加权</strong>汇总成一份诊断，就是这个 token 的输出。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  记住一句话：<strong>稠密层让每个 token 走同一个大网络，MoE 层让每个 token 只走被选中的几个小网络</strong>。前者参数和算力<strong>绑死</strong>——想更聪明就得更慢；
  后者把两者<strong>解耦</strong>——专家数（参数、知识容量）可以独立往上堆，而每个 token 的 FLOPs 由 <span class="mono">top-k</span> 钉住、基本恒定。
  这是当代大模型"既要大、又要快"的关键招式。SGLang 用一个叫 <span class="mono">FusedMoE</span> 的层把它实现得又对又快：路由、分组、按专家批量矩阵乘<strong>融合进 kernel</strong>，
  而不是用一段慢吞吞的 Python 循环去逐个专家算。专家还能被摊到不同 GPU 上（专家并行，第 46 课），热门专家的负载由 EPLB 来均衡（第 47 课）。
</div>

<h2>从稠密 FFN 到 MoE：多了一群专家和一个路由器</h2>
<p>先看它替代了什么。一个普通的 Transformer 块里，注意力之后跟着一个<strong>稠密 FFN</strong>：就是一个大号的 MLP（两层线性 + 激活），<strong>每一个 token 都要完整地穿过它一次</strong>。
这意味着——想让这个 FFN 更有"知识"，唯一的办法是把它<strong>加宽加大</strong>，可一旦加大，<strong>每个 token 的计算量就同步上涨</strong>。参数量和单 token 算力<strong>焊死在一起</strong>，这就是稠密层的天花板。</p>
<p>MoE 层换了一种结构。它把那个大 FFN 拆成 <strong>N 个专家</strong>（每个专家就是一个<strong>更小的 FFN</strong>，比如 64 个、256 个），再额外配一个<strong>路由器 / 门控（router / gate）</strong>——
一个很小的线性层。每来一个 token，路由器先给<strong>每个专家打一个分</strong>，然后<strong>只挑出得分最高的 top-k 个</strong>（比如 64 选 2）。
<strong>只有这 k 个专家真正参与计算</strong>，其余几十个专家<strong>对这个 token 一动不动</strong>。这就叫<strong>稀疏激活</strong>：一层里有海量参数，但每个 token 只点亮其中一小撮。</p>
<p>关键的账要算清楚：假设有 64 个专家、每个 token 走 2 个，那么这一层的<strong>参数</strong>大约是单个 FFN 的 64 倍（知识容量大涨），可每个 token 的<strong>计算量</strong>只相当于"过 2 个小 FFN"——
和一个中等稠密 FFN 差不多，<strong>几乎不随专家总数增长</strong>。于是你能把模型做到几百亿、上千亿参数，而推理时每个 token 的实际 FLOPs 仍被 <span class="mono">top-k</span> 牢牢摁住。
这就是为什么人们说 MoE "<strong>scale 参数而不 scale 每 token 算力</strong>"。当然，"参数多"意味着<strong>显存得装下全部专家</strong>，哪怕大多数专家对某个 token 没出力——这笔账留到本课末尾的权衡里再算。</p>
<p>再换个角度体会这件事的妙处。稠密 FFN 像让<strong>每个学生都修同一门通识大课</strong>：课越厚、内容越多，每个人花的时间就越长，没有例外。MoE 则像一所<strong>开了几百门选修课的大学</strong>：
课程总量（知识容量）可以无限往上加，但每个学生<strong>每学期只选两门</strong>，他的"学习负担"始终恒定。学校越办越大、越办越全，单个学生的时间表却不变——这正是"<strong>容量与算力解耦</strong>"最朴素的样子。
更妙的是，不同学生选的两门课<strong>各不相同</strong>：有人选"代码 + 数学"，有人选"诗歌 + 历史"，于是同一所大学能同时服务千差万别的需求，而不必逼每个人都把所有课上一遍。
落到模型上，这意味着不同类型的 token（不同语言、不同主题、不同结构）会<strong>自发地被路由到擅长它的那几个专家</strong>，专家之间因此分化出隐性的"专长"——这正是"混合专家"四个字的字面含义。</p>

<div class="cols">
  <div class="col"><h4>稠密 FFN（dense）</h4><p><strong>每个 token 都穿过同一个大 MLP</strong>。想更有知识只能把它加宽——<strong>参数和单 token 算力一起涨</strong>，焊死在一起。结构简单、负载天然均衡，但"大"和"快"不可兼得。</p></div>
  <div class="col"><h4>MoE 层（sparse）</h4><p><strong>N 个小专家 + 1 个路由器</strong>，每个 token 只走 <span class="mono">top-k</span> 个（如 64 选 2）。<strong>参数随专家数猛涨，单 token 算力被 top-k 钉住</strong>。又大又快，代价是路由、通信与显存（见下文）。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="路由器给一个 token 对全部 8 个专家打分，只选 top-k=2 个专家计算，其余专家跳过，最后按门控权重把两个专家的输出加权合并">
    <text x="24" y="20" style="font-weight:700;fill:var(--muted)">一个 token → 路由器打分 → 只走 top-k 个专家</text>
    <rect x="24" y="120" width="96" height="44" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="72" y="147" text-anchor="middle" class="mono">token x</text>
    <line x1="120" y1="142" x2="150" y2="142" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="150" y="104" width="140" height="76" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="220" y="134" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">路由器 / 门控</text>
    <text x="220" y="156" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">给 8 个专家打分</text>
    <line x1="290" y1="142" x2="340" y2="29" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="63" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="97" style="stroke:var(--blue);stroke-width:1.5"/>
    <line x1="290" y1="142" x2="340" y2="131" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="165" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="199" style="stroke:var(--blue);stroke-width:1.5"/>
    <line x1="290" y1="142" x2="340" y2="233" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="267" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="340" y="16" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="33" class="mono" style="fill:var(--faint);font-size:12px">E0</text>
    <text x="462" y="33" text-anchor="end" style="fill:var(--faint);font-size:11px">跳过</text>
    <rect x="340" y="50" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="67" class="mono" style="fill:var(--faint);font-size:12px">E1</text>
    <text x="462" y="67" text-anchor="end" style="fill:var(--faint);font-size:11px">跳过</text>
    <rect x="340" y="84" width="130" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="350" y="101" class="mono" style="fill:var(--blue);font-size:12px">E2</text>
    <text x="462" y="101" text-anchor="end" style="fill:var(--blue);font-size:11px">选中 ·0.6</text>
    <rect x="340" y="118" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="135" class="mono" style="fill:var(--faint);font-size:12px">E3</text>
    <text x="462" y="135" text-anchor="end" style="fill:var(--faint);font-size:11px">跳过</text>
    <rect x="340" y="152" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="169" class="mono" style="fill:var(--faint);font-size:12px">E4</text>
    <text x="462" y="169" text-anchor="end" style="fill:var(--faint);font-size:11px">跳过</text>
    <rect x="340" y="186" width="130" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="350" y="203" class="mono" style="fill:var(--blue);font-size:12px">E5</text>
    <text x="462" y="203" text-anchor="end" style="fill:var(--blue);font-size:11px">选中 ·0.4</text>
    <rect x="340" y="220" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="237" class="mono" style="fill:var(--faint);font-size:12px">E6</text>
    <text x="462" y="237" text-anchor="end" style="fill:var(--faint);font-size:11px">跳过</text>
    <rect x="340" y="254" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="271" class="mono" style="fill:var(--faint);font-size:12px">E7</text>
    <text x="462" y="271" text-anchor="end" style="fill:var(--faint);font-size:11px">跳过</text>
    <line x1="470" y1="97" x2="520" y2="132" style="stroke:var(--blue);stroke-width:1.5"/>
    <line x1="470" y1="199" x2="520" y2="160" style="stroke:var(--blue);stroke-width:1.5"/>
    <rect x="520" y="110" width="150" height="74" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="595" y="140" text-anchor="middle" style="fill:var(--teal);font-weight:700">加权合并</text>
    <text x="595" y="162" text-anchor="middle" class="mono" style="font-size:11px">0.6·E2 + 0.4·E5</text>
    <line x1="670" y1="147" x2="700" y2="147" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="706" y="151" style="fill:var(--muted);font-size:12px">输出</text>
  </svg>
  <div class="figcap"><b>图 1 · 路由器把每个 token 发给它的 top-k 专家</b> — 路由器给全部 8 个专家打分，<strong>只选 top-k=2</strong>（这里 E2、E5）真正计算，其余专家对该 token <strong>一动不动</strong>；最后按门控权重 <span class="mono">0.6·E2 + 0.4·E5</span> 合并成输出。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="稠密 FFN 的参数容量与每 token 算力焊死在一起，二者一样大；稀疏 MoE 的参数容量很大但每 token 算力被 top-k 钉得很小">
    <line x1="390" y1="40" x2="390" y2="232" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">稠密 FFN：全部参数都算</text>
    <text x="24" y="92" style="fill:var(--muted);font-size:12px">参数容量</text>
    <rect x="150" y="76" width="110" height="26" rx="5" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="24" y="142" style="fill:var(--muted);font-size:12px">每 token 算力</text>
    <rect x="150" y="126" width="110" height="26" rx="5" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="150" y="196" style="fill:var(--accent-ink);font-weight:700;font-size:12px">参数 = 算力，焊死</text>
    <text x="410" y="26" style="font-weight:700;fill:var(--accent-ink)">稀疏 MoE：只算 k 个专家</text>
    <text x="410" y="92" style="fill:var(--muted);font-size:12px">参数容量</text>
    <rect x="545" y="76" width="195" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="555" y="93" style="fill:var(--blue);font-size:11px">N 个专家（很大）</text>
    <text x="410" y="142" style="fill:var(--muted);font-size:12px">每 token 算力</text>
    <rect x="545" y="126" width="60" height="26" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="611" y="143" style="fill:var(--teal);font-size:11px">只算 k 个</text>
    <text x="410" y="196" style="fill:var(--teal);font-weight:700;font-size:12px">容量大涨，算力被 top-k 钉住</text>
  </svg>
  <div class="figcap"><b>图 2 · 稠密（全部参数）vs 稀疏 MoE（只算 k 个专家）</b> — 稠密 FFN 的<strong>参数容量与每 token 算力焊死</strong>、一起涨；稀疏 MoE 有一个<strong>巨大的专家池</strong>（容量大涨），但每个 token 只激活 <span class="mono">N 选 k</span> 个 → <strong>算力被 top-k 钉得很小</strong>。</div>
</div>

<h2>一次 MoE 前向：路由 → 分组 → 分组 GEMM → 加权合并</h2>
<p>把一个 token 进入 MoE 层后的旅程拆成四步，你就懂了这层在算什么。<strong>第一步，路由（route）</strong>：路由器对每个 token 算出对所有专家的得分，取 <span class="mono">top-k</span>，得到"这个 token 该去哪几个专家"以及对应的<strong>路由权重</strong>（一组归一化的分数）。
<strong>第二步，分组（group）</strong>：一个 batch 里有成千上万个 token，各自被分派到不同专家。直接逐 token 调用专家会极其低效，所以要<strong>按目标专家把 token 重新归拢到一起</strong>——
所有要去 1 号专家的 token 排成一摞，要去 2 号的排成另一摞，以此类推。</p>
<p>为什么"分组"这一步不可省？因为 GPU 最怕<strong>零碎</strong>。如果不分组，引擎就得"这个 token 去 7 号、下个去 12 号、再下个去 3 号"地<strong>一个一个发</strong>，每次都只喂给某个专家一丁点数据——
GPU 的算力像一条宽阔的高速公路，你却让车一辆一辆地过收费站，<strong>大部分车道空着</strong>。分组就是先把所有去同一个专家的 token<strong>攒成一大批</strong>，让每个专家一次吃下一整摞，把高速公路<strong>填满</strong>。
这也解释了 MoE 为什么对 batch 大小敏感：batch 越大，每个专家分到的 token 越多、那一摞越厚，grouped GEMM 的效率就越高；反之 batch 太小、专家太多时，每摞都很薄，硬件利用率反而上不去。</p>
<p><strong>第三步，分组 / 批量 GEMM（grouped / batched GEMM）</strong>：现在每个专家面对的是<strong>一整摞属于它的 token</strong>，于是可以做一次<strong>高效的批量矩阵乘</strong>，
把"几十个专家、每个一小批"合成一个规整的、对 GPU 友好的大算子，而不是几十次零碎的小矩阵乘。这是 MoE 能跑得快的<strong>算力核心</strong>（grouped GEMM 的 kernel 细节见第 38 课）。
<strong>第四步，散回并加权合并（scatter & combine）</strong>：把每个专家算出的结果<strong>送回它原来的 token 位置</strong>，再按第一步那组<strong>路由权重加权求和</strong>——
一个走了"专家 7 和专家 12"的 token，最终输出 = 0.7×专家7 + 0.3×专家12（权重来自路由器）。这一步之后，MoE 层的输出形状和稠密 FFN 完全一样，下游<strong>毫无感知</strong>。
正因为输入输出形状都和稠密 FFN 对齐，MoE 才能<strong>无缝替换</strong>模型里原来那个 <span class="mono">self.mlp</span>——注意力层、归一化、残差连接统统不用改一行，这也是第 26 课你写模型时，把稠密块换成 MoE 块如此轻松的原因。</p>
<p>这里有个工程上的大坑：上面四步如果用<strong>一段 Python 循环"for 每个专家"</strong>去做，会慢得离谱——几十次 kernel 启动、几十次小算子、来回搬数据，GPU 大量时间在<strong>空转等待</strong>。
SGLang 的 <span class="mono">FusedMoE</span> 层正是为此而生：它把<strong>路由 + 分组 + 分组 GEMM 融合进少数几个 kernel</strong>，<strong>没有逐专家的慢 Python 循环</strong>。
"融合（fuse）"在这里的意思就是——本来要分好几趟、还夹着 Python 调度的活，被压进一两个 GPU kernel 一气呵成，启动开销和中间搬运统统省掉。换句话说，<strong>慢的不是矩阵乘本身，而是"碎"</strong>：
几十个专家被拆成几十次独立调用时，真正算数的时间可能只占一小半，剩下的全耗在<strong>调度、等待、搬运</strong>上。<span class="mono">FusedMoE</span> 把这些碎活缝合成一整块连续的 GPU 工作，让硬件一口气跑完，这才是它"又对又快"里"快"的来源。下面就是这个层的真身：</p>

<div class="flow">
  <div class="node"><div class="nt">token x</div><div class="nd">一个 token 进入<br>MoE 层（第 26 课）</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">路由器 / 门控</div><div class="nd">给每个专家打分<br>取 top-k（如 64 选 2）</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">分组 + 分组 GEMM</div><div class="nd">同专家 token 归拢<br>批量矩阵乘（第 38 课）</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">加权合并</div><div class="nd">按路由权重<br>0.7·E7 + 0.3·E12</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">输出</div><div class="nd">形状同稠密 FFN<br>下游无感知</div></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/moe/fused_moe_triton/layer.py ::FusedMoE</span><span class="ln">N 个专家 + 路由 top-k，融合进 kernel</span></div>
  <pre><span class="kw">class</span> <span class="st">FusedMoE</span>(torch.nn.Module):
    <span class="cm"># MoE 层：同时持有 gate_up_proj(w13) 与 down_proj(w2) 两组专家权重</span>
    <span class="kw">def</span> __init__(
        self,
        num_experts: int,      <span class="cm"># 这一层里专家的总数（如 64 / 256）</span>
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,   <span class="cm"># 每个 token 选几个专家（如 2）</span>
        ...
    ):
        self.layer_id = layer_id
        self.top_k = top_k             <span class="cm"># 稀疏：只有 top_k 个专家真正参与</span>
        self.num_experts = num_experts <span class="cm"># 参数随它涨，单 token 算力随 top_k 定</span>
        self.moe_ep_size = get_parallel().moe_ep_size  <span class="cm"># 专家并行规模（第 46 课）</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/moe/topk.py ::TopKConfig</span><span class="ln">路由配置：每 token 选几个专家、如何打分/归一</span></div>
  <pre><span class="kw">@dataclass</span>
<span class="kw">class</span> <span class="st">TopKConfig</span>:
    <span class="cm"># 路由器怎么为每个 token 挑专家。</span>
    top_k: int                       <span class="cm"># 每个 token 选几个专家（如 2）</span>
    use_grouped_topk: bool = False   <span class="cm"># 分组受限路由（DeepSeek）</span>
    renormalize: bool = True         <span class="cm"># 对 top-k 门控权重重新归一化</span>
    scoring_func: str = <span class="st">"softmax"</span>    <span class="cm"># 对专家得分用 softmax / sigmoid</span>
    num_expert_group: Optional[int] = None
    topk_group: Optional[int] = None
    ...</pre>
</div>

<p>举两个具体例子把数字坐实。<strong>Mixtral</strong>：每个 token 在 8 个专家里选 2 个（<span class="mono">top_k=2, N=8</span>）——参数大约是单个 FFN 的 8 倍，但每个 token 只付"<strong>2 个专家</strong>"的算力。<strong>DeepSeek-V3</strong>：专家上百，于是先把专家分成若干<strong>组</strong>，用<strong>分组 top-k</strong>（<span class="mono">use_grouped_topk=True</span>，配合 <span class="mono">num_expert_group / topk_group</span>）——先在组级筛掉大部分组，再在选中的组里挑专家，既省路由开销又利于专家并行下的通信。两种都只是 <span class="mono">TopKConfig</span> 里几个字段的不同取值。</p>

<h2>几个绕不开的名词</h2>
<p>MoE 的文档里高频出现一小撮术语，提前钉死含义，后面读 DeepSeek-V3、Mixtral 的代码就不会卡壳。它们其实就是上面四步流程里的几个关键角色，把它们对号入座即可。要特别分清两个最容易混的：
<strong>专家（expert）</strong>是真正"干活、存知识"的小 FFN，一层有 N 个；<strong>路由器（router）</strong>则只是个<strong>极小的线性层</strong>，它本身不存知识、不做重活，只负责"<strong>派活</strong>"——给每个 token 决定该找哪几个专家。
专家是<strong>多</strong>而<strong>大</strong>的（参数主要堆在这里），路由器是<strong>单</strong>而<strong>小</strong>的（参数可以忽略不计）。记住这组对比，下面这张表就一目了然了。</p>

<table class="t">
  <tr><th>名词</th><th>是什么</th><th>作用</th></tr>
  <tr><td class="mono">expert（专家）</td><td>一个<strong>较小的 FFN</strong>，一层里有 N 个</td><td>承载知识；只有被选中的才计算</td></tr>
  <tr><td class="mono">router / gate（路由器）</td><td>一个<strong>很小的线性层</strong></td><td>给每个 token 对各专家<strong>打分</strong>，决定去向与权重</td></tr>
  <tr><td class="mono">top-k</td><td>每个 token <strong>选中的专家数</strong>（如 2）</td><td>把<strong>单 token 算力钉死</strong>，实现稀疏</td></tr>
  <tr><td class="mono">grouped GEMM</td><td>按专家<strong>分组的批量矩阵乘</strong></td><td>把零碎小算子合成 GPU 友好的大算子（第 38 课）</td></tr>
  <tr><td class="mono">EP（专家并行）</td><td>把专家<strong>摊到多张 GPU</strong></td><td>容纳放不下的海量专家（第 46 课）</td></tr>
</table>

<h2>扩展到多卡：专家并行与负载均衡</h2>
<p>专家一多，单张 GPU 就<strong>装不下全部专家</strong>了——这正是 MoE 的代价之一。解法叫<strong>专家并行（Expert Parallelism，EP，第 46 课）</strong>：把 N 个专家<strong>分散到不同 GPU 上</strong>，
每张卡只持有一部分专家。但 token 是按路由结果走的，一个 token 想去的专家可能<strong>不在它当前所在的卡上</strong>，于是需要一次<strong>全员对全员的分发（all-to-all dispatch）</strong>——
把每个 token 送到它的专家所在的那张卡，算完再<strong>收集回来（combine）</strong>做加权合并。专家并行让模型总容量突破单卡显存的限制，代价是多了这趟<strong>跨卡通信</strong>。这趟通信不是小数目：每一层 MoE 都要做一来一回两次 all-to-all（先把 token 分发出去、算完再收回来），层数一多、卡数一多，<strong>通信就可能成为新的瓶颈</strong>，
所以专家并行的部署里，网络带宽和通信与计算的重叠（让通信藏在计算背后）就格外关键——这也是后面专门用一整课讲专家并行的原因。</p>
<p>还有一个绕不开的麻烦：<strong>路由不均衡</strong>。路由器是学出来的，难免出现某几个"<strong>热门专家</strong>"被远超平均的 token 选中，而另一些专家门可罗雀。
在专家并行下，热门专家所在的那张 GPU 就成了<strong>瓶颈</strong>——别的卡都算完了在等它。<strong>EPLB（Expert Parallelism Load Balancer，第 47 课）</strong>专治此症：
它通过<strong>复制热门专家、重排专家到 GPU 的摆放</strong>，把负载摊匀，让各卡忙闲均衡。下面这组对比，直观展示几个 token 各自被路由到不同专家——注意高亮的就是每个 token 选中的 top-2：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>四个 token，各自被路由到不同的 top-2 专家</b>：高亮格是路由器为该 token 选中、并真正参与计算的专家；其余几十个专家对它<strong>一动不动</strong></div>
  <div class="cells"><span class="lab">token「天」</span><span class="cell hl">E7 ·0.7</span><span class="cell hl">E12 ·0.3</span><span class="cell">E3</span><span class="cell">E40</span><span class="sep">→</span><span class="cell q">输出 = 0.7·专家7 + 0.3·专家12（按路由权重加权）</span></div>
  <div class="cells"><span class="lab">token「气」</span><span class="cell">E7</span><span class="cell hl">E3 ·0.6</span><span class="cell hl">E51 ·0.4</span><span class="cell">E12</span><span class="sep">→</span><span class="cell q">同一层、不同 token，选中的专家可以完全不同</span></div>
  <div class="cells"><span class="lab">token「真」</span><span class="cell hl">E7 ·0.55</span><span class="cell">E3</span><span class="cell hl">E40 ·0.45</span><span class="cell">E9</span><span class="sep">→</span><span class="cell q">E7 又被选中——它是个<strong>热门专家</strong>，负载偏重</span></div>
  <div class="cells"><span class="lab">token「好」</span><span class="cell hl">E7 ·0.5</span><span class="cell hl">E12 ·0.5</span><span class="cell">E51</span><span class="cell">E3</span><span class="sep">→</span><span class="cell q">E7 第三次中选 → 热门专家所在卡成瓶颈，靠 EPLB 摊匀（第 47 课）</span></div>
</div>

<p>把账本合起来看 MoE 的权衡：<strong>收益</strong>是用<strong>不大的计算</strong>换来<strong>巨大的有效模型</strong>（更多专家＝更多知识容量）；<strong>代价</strong>有三笔——
其一，<strong>路由不均衡</strong>会让热门专家拖慢全局（EPLB 来治）；其二，专家并行下的 <strong>all-to-all 通信</strong>是实打实的开销；其三，<strong>显存要装下全部专家</strong>，哪怕大多数对某个 token 没出力。
理解了这三笔代价，你才算真正读懂 MoE：它不是免费的午餐，而是把"算力账"换成了"<strong>通信账 + 显存账 + 均衡账</strong>"——而 SGLang 的 <span class="mono">FusedMoE</span>、专家并行（第 46 课）、EPLB（第 47 课）正是为压低这三笔账单而生。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <strong>① MoE 层 = N 个小专家（FFN）+ 一个路由器</strong>：路由器给每个 <strong>token</strong> 对各专家打分，只选 <span class="mono">top-k</span>（如 64 选 2），仅这 k 个专家计算 → <strong>稀疏</strong>。于是<strong>参数随专家数猛涨，单 token 算力被 top-k 钉死</strong>。DeepSeek-V3 / Mixtral / Qwen-MoE 都用它。
  <strong>② 计算四步</strong>：路由 → 按专家分组 → <strong>分组 / 批量 GEMM</strong> → 散回并按路由权重加权合并。SGLang 的 <span class="mono">FusedMoE</span> 把路由 + 分组 GEMM <strong>融合进 kernel</strong>，没有逐专家的慢 Python 循环（kernel 见第 38 课）。
  <strong>③ 扩展到多卡</strong>：<strong>专家并行（EP，第 46 课）</strong>把专家摊到不同 GPU，token 经 all-to-all 分发到专家所在卡再合并；<strong>EPLB（第 47 课）</strong>均衡"热门专家"的负载。
  <strong>④ 权衡</strong>：用不大的计算换巨大的有效模型，代价是<strong>路由不均衡 + all-to-all 通信 + 装下全部专家的显存</strong>。写模型时怎么摆这一层见第 26 课。
</div>
""",
             "en": r"""
<p class="lead">
In Lesson 26 you'll write a model by hand, and in many newer models that line <span class="mono">self.mlp(x)</span> is no longer a plain multilayer perceptron but a <strong>MoE layer</strong> (Mixture of Experts).
DeepSeek-V3, Mixtral, and Qwen-MoE all use it. Its counter-intuitive heart: <strong>the parameter count can grow tens or hundreds of times while the compute each token actually spends stays nearly flat</strong>.
This lesson explains how that "seemingly free lunch" works — by splitting one big FFN into <strong>many small experts</strong>, adding a <strong>router</strong>, and letting each token visit only a <strong>tiny few</strong> of them.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the MoE layer as a big hospital's <strong>triage desk</strong>. The hospital staffs <strong>dozens of specialists</strong> (experts), but when a patient (token) arrives, the triage nurse (router) <strong>does not</strong> send them through every department —
  that would be slow and wasteful. The nurse takes one look at the symptoms and <strong>picks the two most relevant specialties</strong> (top-2), say "cardiology + endocrinology," and the patient visits only those two rooms.
  So the hospital can <strong>keep opening more specialties, accumulating ever more expertise</strong>, while <strong>each patient's visit time stays nearly constant</strong> — they always see just two doctors. That is the soul of MoE: trade "<strong>sparsity</strong>" for "<strong>far larger total capacity</strong>,"
  instead of making everyone pay for that vastness. Finally the two doctors' opinions are <strong>combined weighted by relevance</strong> into one diagnosis — the token's output.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Remember one line: <strong>a dense layer runs every token through the same big network; an MoE layer runs each token through only a chosen few small networks</strong>. The former <strong>welds</strong> parameters to compute — smarter means slower;
  the latter <strong>decouples</strong> them — the expert count (parameters, knowledge capacity) can climb on its own, while per-token FLOPs are pinned by <span class="mono">top-k</span> and stay roughly fixed.
  This is the modern move for "big yet fast." SGLang implements it correctly and quickly with a layer called <span class="mono">FusedMoE</span>: routing, grouping, and per-expert batched matmul are <strong>fused into kernels</strong>,
  not done by a sluggish Python loop over experts. Experts can also be spread across GPUs (expert parallelism, Lesson 46), and hot-expert load is balanced by EPLB (Lesson 47).
</div>

<h2>From a dense FFN to MoE: a crowd of experts plus a router</h2>
<p>First, what it replaces. In a normal Transformer block, attention is followed by a <strong>dense FFN</strong>: a large MLP (two linears + activation) that <strong>every token passes through in full</strong>.
That means the only way to make this FFN more "knowledgeable" is to <strong>widen and enlarge it</strong> — but once enlarged, <strong>every token's compute rises in lockstep</strong>. Parameter count and per-token compute are <strong>welded together</strong>; that is the dense layer's ceiling.</p>
<p>An MoE layer swaps the structure. It splits that big FFN into <strong>N experts</strong> (each a <strong>smaller FFN</strong> — say 64 or 256 of them) and adds a <strong>router / gate</strong> —
a tiny linear layer. For each incoming token, the router first <strong>scores every expert</strong>, then <strong>keeps only the top-k</strong> highest (e.g. 2 of 64).
<strong>Only those k experts actually compute</strong>; the other dozens <strong>do nothing for this token</strong>. That is <strong>sparse activation</strong>: a layer holds enormous parameters, but each token lights up only a small handful.</p>
<p>Do the arithmetic: with 64 experts and 2 per token, this layer's <strong>parameters</strong> are roughly 64x a single FFN (huge knowledge capacity), yet each token's <strong>compute</strong> is just "pass through 2 small FFNs" —
about a medium dense FFN, and <strong>nearly independent of the total expert count</strong>. So you can scale a model to tens or hundreds of billions of parameters while per-token inference FLOPs stay firmly pinned by <span class="mono">top-k</span>.
That is why people say MoE "<strong>scales parameters, not per-token compute</strong>." Of course, "more parameters" means <strong>memory must hold every expert</strong>, even those idle for a given token — a cost we settle in the trade-offs at the end.</p>

<div class="cols">
  <div class="col"><h4>Dense FFN</h4><p><strong>Every token passes through the same big MLP</strong>. More knowledge means widening it — <strong>parameters and per-token compute rise together</strong>, welded. Simple and naturally balanced, but "big" and "fast" can't coexist.</p></div>
  <div class="col"><h4>MoE layer (sparse)</h4><p><strong>N small experts + 1 router</strong>; each token visits only <span class="mono">top-k</span> (e.g. 2 of 64). <strong>Parameters soar with expert count; per-token compute pinned by top-k</strong>. Big and fast — at the cost of routing, communication, and memory (below).</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="The router scores one token against all 8 experts, keeps only top-k=2 experts to compute, skips the rest, then combines the two expert outputs weighted by the gate scores">
    <text x="24" y="20" style="font-weight:700;fill:var(--muted)">one token → router scores → only top-k experts run</text>
    <rect x="24" y="120" width="96" height="44" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="72" y="147" text-anchor="middle" class="mono">token x</text>
    <line x1="120" y1="142" x2="150" y2="142" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="150" y="104" width="140" height="76" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="220" y="134" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">router / gate</text>
    <text x="220" y="156" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">scores 8 experts</text>
    <line x1="290" y1="142" x2="340" y2="29" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="63" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="97" style="stroke:var(--blue);stroke-width:1.5"/>
    <line x1="290" y1="142" x2="340" y2="131" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="165" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="199" style="stroke:var(--blue);stroke-width:1.5"/>
    <line x1="290" y1="142" x2="340" y2="233" style="stroke:var(--faint);stroke-width:1"/>
    <line x1="290" y1="142" x2="340" y2="267" style="stroke:var(--faint);stroke-width:1"/>
    <rect x="340" y="16" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="33" class="mono" style="fill:var(--faint);font-size:12px">E0</text>
    <text x="462" y="33" text-anchor="end" style="fill:var(--faint);font-size:11px">skip</text>
    <rect x="340" y="50" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="67" class="mono" style="fill:var(--faint);font-size:12px">E1</text>
    <text x="462" y="67" text-anchor="end" style="fill:var(--faint);font-size:11px">skip</text>
    <rect x="340" y="84" width="130" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="350" y="101" class="mono" style="fill:var(--blue);font-size:12px">E2</text>
    <text x="462" y="101" text-anchor="end" style="fill:var(--blue);font-size:11px">pick ·0.6</text>
    <rect x="340" y="118" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="135" class="mono" style="fill:var(--faint);font-size:12px">E3</text>
    <text x="462" y="135" text-anchor="end" style="fill:var(--faint);font-size:11px">skip</text>
    <rect x="340" y="152" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="169" class="mono" style="fill:var(--faint);font-size:12px">E4</text>
    <text x="462" y="169" text-anchor="end" style="fill:var(--faint);font-size:11px">skip</text>
    <rect x="340" y="186" width="130" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="350" y="203" class="mono" style="fill:var(--blue);font-size:12px">E5</text>
    <text x="462" y="203" text-anchor="end" style="fill:var(--blue);font-size:11px">pick ·0.4</text>
    <rect x="340" y="220" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="237" class="mono" style="fill:var(--faint);font-size:12px">E6</text>
    <text x="462" y="237" text-anchor="end" style="fill:var(--faint);font-size:11px">skip</text>
    <rect x="340" y="254" width="130" height="26" rx="5" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5"/>
    <text x="350" y="271" class="mono" style="fill:var(--faint);font-size:12px">E7</text>
    <text x="462" y="271" text-anchor="end" style="fill:var(--faint);font-size:11px">skip</text>
    <line x1="470" y1="97" x2="520" y2="132" style="stroke:var(--blue);stroke-width:1.5"/>
    <line x1="470" y1="199" x2="520" y2="160" style="stroke:var(--blue);stroke-width:1.5"/>
    <rect x="520" y="110" width="150" height="74" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="595" y="140" text-anchor="middle" style="fill:var(--teal);font-weight:700">combine</text>
    <text x="595" y="162" text-anchor="middle" class="mono" style="font-size:11px">0.6·E2 + 0.4·E5</text>
    <line x1="670" y1="147" x2="700" y2="147" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="706" y="151" style="fill:var(--muted);font-size:12px">output</text>
  </svg>
  <div class="figcap"><b>Fig 1 · The router sends each token to its top-k experts</b> — the router scores all 8 experts but <strong>keeps only top-k=2</strong> (here E2, E5) to actually compute; the rest <strong>do nothing</strong> for this token. The output is combined by gate weights <span class="mono">0.6·E2 + 0.4·E5</span>.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="A dense FFN welds parameter capacity to per-token compute so both bars are equal; a sparse MoE has a huge parameter capacity but its per-token compute bar is pinned small by top-k">
    <line x1="390" y1="40" x2="390" y2="232" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">Dense FFN: all params active</text>
    <text x="24" y="92" style="fill:var(--muted);font-size:12px">capacity</text>
    <rect x="150" y="76" width="110" height="26" rx="5" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="24" y="142" style="fill:var(--muted);font-size:12px">per-tok compute</text>
    <rect x="150" y="126" width="110" height="26" rx="5" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="150" y="196" style="fill:var(--accent-ink);font-weight:700;font-size:12px">params = compute, welded</text>
    <text x="410" y="26" style="font-weight:700;fill:var(--accent-ink)">Sparse MoE: only k experts run</text>
    <text x="410" y="92" style="fill:var(--muted);font-size:12px">capacity</text>
    <rect x="545" y="76" width="195" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="555" y="93" style="fill:var(--blue);font-size:11px">N experts (huge)</text>
    <text x="410" y="142" style="fill:var(--muted);font-size:12px">per-tok compute</text>
    <rect x="545" y="126" width="60" height="26" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="611" y="143" style="fill:var(--teal);font-size:11px">only k</text>
    <text x="410" y="196" style="fill:var(--teal);font-weight:700;font-size:12px">capacity soars, compute pinned by top-k</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Dense (all params) vs sparse MoE (only k experts)</b> — a dense FFN <strong>welds capacity to per-token compute</strong>, so both rise together; a sparse MoE owns a <strong>huge expert pool</strong> (capacity soars) yet each token activates only <span class="mono">k of N</span> → <strong>compute stays small, pinned by top-k</strong>.</div>
</div>

<h2>One MoE forward: route → group → grouped GEMM → weighted combine</h2>
<p>Break a token's journey through an MoE layer into four steps and you'll see what it computes. <strong>Step 1, route</strong>: the router scores each token against all experts, takes <span class="mono">top-k</span>, yielding "which experts this token goes to" plus the matching <strong>routing weights</strong> (a set of normalized scores).
<strong>Step 2, group</strong>: a batch holds thousands of tokens, each dispatched to different experts. Calling experts token-by-token is wildly inefficient, so tokens are <strong>regrouped by target expert</strong> —
all tokens bound for expert 1 in one pile, those for expert 2 in another, and so on.</p>
<p><strong>Step 3, grouped / batched GEMM</strong>: each expert now faces <strong>a whole pile of its own tokens</strong>, so it can do one <strong>efficient batched matmul</strong>,
fusing "dozens of experts, each a small batch" into a regular, GPU-friendly big operator rather than dozens of tiny scattered matmuls. This is the <strong>compute core</strong> that lets MoE run fast (grouped-GEMM kernel details in Lesson 38).
<strong>Step 4, scatter & combine</strong>: send each expert's result <strong>back to its original token position</strong>, then <strong>weighted-sum by the routing weights</strong> from step 1 —
a token that went to "expert 7 and expert 12" gets output = 0.7×expert7 + 0.3×expert12 (weights from the router). After this, the MoE layer's output shape is identical to a dense FFN's; downstream is <strong>none the wiser</strong>.</p>
<p>Here's a big engineering trap: doing those four steps with a <strong>Python loop "for each expert"</strong> is absurdly slow — dozens of kernel launches, dozens of tiny operators, data shuffled back and forth, the GPU mostly <strong>idling and waiting</strong>.
SGLang's <span class="mono">FusedMoE</span> exists precisely for this: it <strong>fuses routing + grouping + grouped GEMM into a few kernels</strong>, with <strong>no slow per-expert Python loop</strong>.
"Fuse" here means what used to take several passes, with Python scheduling in between, is pressed into one or two GPU kernels in one shot — launch overhead and intermediate movement gone. Here is the layer itself:</p>

<div class="flow">
  <div class="node"><div class="nt">token x</div><div class="nd">a token enters the<br>MoE layer (Lesson 26)</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">router / gate</div><div class="nd">score every expert<br>take top-k (e.g. 2 of 64)</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">group + grouped GEMM</div><div class="nd">pile same-expert tokens<br>batched matmul (Lesson 38)</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">weighted combine</div><div class="nd">by routing weights<br>0.7·E7 + 0.3·E12</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">output</div><div class="nd">same shape as dense FFN<br>downstream unaware</div></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/moe/fused_moe_triton/layer.py ::FusedMoE</span><span class="ln">N experts + top-k routing, fused into kernels</span></div>
  <pre><span class="kw">class</span> <span class="st">FusedMoE</span>(torch.nn.Module):
    <span class="cm"># MoE layer: holds both gate_up_proj(w13) and down_proj(w2) expert weights</span>
    <span class="kw">def</span> __init__(
        self,
        num_experts: int,      <span class="cm"># total experts in this layer (e.g. 64 / 256)</span>
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,   <span class="cm"># experts picked per token (e.g. 2)</span>
        ...
    ):
        self.layer_id = layer_id
        self.top_k = top_k             <span class="cm"># sparse: only top_k experts actually run</span>
        self.num_experts = num_experts <span class="cm"># params grow with this; compute set by top_k</span>
        self.moe_ep_size = get_parallel().moe_ep_size  <span class="cm"># expert-parallel size (Lesson 46)</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/moe/topk.py ::TopKConfig</span><span class="ln">routing config: how many experts per token, scoring &amp; renorm</span></div>
  <pre><span class="kw">@dataclass</span>
<span class="kw">class</span> <span class="st">TopKConfig</span>:
    <span class="cm"># how the router picks experts for each token.</span>
    top_k: int                       <span class="cm"># experts per token (e.g. 2)</span>
    use_grouped_topk: bool = False   <span class="cm"># group-limited routing (DeepSeek)</span>
    renormalize: bool = True         <span class="cm"># renormalize the top-k gate weights</span>
    scoring_func: str = <span class="st">"softmax"</span>    <span class="cm"># softmax / sigmoid over expert scores</span>
    num_expert_group: Optional[int] = None
    topk_group: Optional[int] = None
    ...</pre>
</div>

<p>Two concrete examples to pin the numbers. <strong>Mixtral</strong>: each token picks 2 of 8 experts (<span class="mono">top_k=2, N=8</span>) — roughly 8x a single FFN's parameters, but each token pays only "<strong>2 experts'</strong>" compute. <strong>DeepSeek-V3</strong>: with hundreds of experts it first splits them into <strong>groups</strong> and uses <strong>grouped top-k</strong> (<span class="mono">use_grouped_topk=True</span>, with <span class="mono">num_expert_group / topk_group</span>) — prune most groups at the group level, then pick experts within the chosen groups, saving routing cost and easing all-to-all communication under expert parallelism. Both are just different field values in <span class="mono">TopKConfig</span>.</p>

<h2>A few unavoidable terms</h2>
<p>MoE docs lean on a small set of terms. Pin their meanings now and the DeepSeek-V3 / Mixtral code won't trip you up later. They are just the key roles in the four-step flow above — match each to its slot.</p>

<table class="t">
  <tr><th>Term</th><th>What it is</th><th>Role</th></tr>
  <tr><td class="mono">expert</td><td>a <strong>smaller FFN</strong>; N of them per layer</td><td>holds knowledge; computes only if selected</td></tr>
  <tr><td class="mono">router / gate</td><td>a <strong>tiny linear layer</strong></td><td><strong>scores</strong> each token over experts; sets routing & weights</td></tr>
  <tr><td class="mono">top-k</td><td><strong>experts picked</strong> per token (e.g. 2)</td><td><strong>pins per-token compute</strong>; makes it sparse</td></tr>
  <tr><td class="mono">grouped GEMM</td><td><strong>batched matmul grouped</strong> by expert</td><td>fuses tiny ops into a GPU-friendly big one (Lesson 38)</td></tr>
  <tr><td class="mono">EP (expert parallel)</td><td>spread experts <strong>across GPUs</strong></td><td>hold a mass of experts too big for one card (Lesson 46)</td></tr>
</table>

<h2>Scaling out: expert parallelism and load balancing</h2>
<p>Once there are many experts, a single GPU <strong>can't hold them all</strong> — one of MoE's costs. The fix is <strong>Expert Parallelism (EP, Lesson 46)</strong>: spread the N experts <strong>across different GPUs</strong>,
each card holding only some. But tokens follow routing, and a token's chosen expert may <strong>not live on its current card</strong>, so an <strong>all-to-all dispatch</strong> is needed —
send each token to the card holding its expert, compute, then <strong>collect back (combine)</strong> for the weighted sum. Expert parallelism lets total capacity exceed one card's memory, at the cost of this <strong>cross-GPU communication</strong>.</p>
<p>There's another unavoidable headache: <strong>routing imbalance</strong>. The router is learned, so inevitably a few "<strong>hot experts</strong>" get chosen by far more tokens than average, while others sit nearly empty.
Under expert parallelism, the GPU holding a hot expert becomes a <strong>bottleneck</strong> — the other cards finish and wait on it. <strong>EPLB (Expert Parallelism Load Balancer, Lesson 47)</strong> cures exactly this:
by <strong>replicating hot experts and rearranging the expert-to-GPU placement</strong>, it spreads the load so cards stay evenly busy. The comparison below shows several tokens each routed to different experts — the highlights are each token's chosen top-2:</p>

<div class="cellgroup">
  <div class="cg-cap"><b>Four tokens, each routed to a different top-2 of experts</b>: highlighted cells are the experts the router chose for that token and that actually compute; the other dozens <strong>do nothing</strong> for it</div>
  <div class="cells"><span class="lab">token "sky"</span><span class="cell hl">E7 ·0.7</span><span class="cell hl">E12 ·0.3</span><span class="cell">E3</span><span class="cell">E40</span><span class="sep">→</span><span class="cell q">output = 0.7·expert7 + 0.3·expert12 (weighted by routing scores)</span></div>
  <div class="cells"><span class="lab">token "air"</span><span class="cell">E7</span><span class="cell hl">E3 ·0.6</span><span class="cell hl">E51 ·0.4</span><span class="cell">E12</span><span class="sep">→</span><span class="cell q">same layer, different token — the chosen experts can be entirely different</span></div>
  <div class="cells"><span class="lab">token "true"</span><span class="cell hl">E7 ·0.55</span><span class="cell">E3</span><span class="cell hl">E40 ·0.45</span><span class="cell">E9</span><span class="sep">→</span><span class="cell q">E7 chosen again — it's a <strong>hot expert</strong>, carrying heavier load</span></div>
  <div class="cells"><span class="lab">token "good"</span><span class="cell hl">E7 ·0.5</span><span class="cell hl">E12 ·0.5</span><span class="cell">E51</span><span class="cell">E3</span><span class="sep">→</span><span class="cell q">E7 picked a third time → its card bottlenecks; EPLB evens it out (Lesson 47)</span></div>
</div>

<p>Tally MoE's trade-off as a ledger: the <strong>gain</strong> is a <strong>huge effective model</strong> (more experts = more knowledge capacity) for <strong>modest compute</strong>; the <strong>costs</strong> are three —
first, <strong>routing imbalance</strong> lets hot experts drag the whole step (EPLB cures it); second, the <strong>all-to-all communication</strong> under expert parallelism is real overhead; third, <strong>memory must hold every expert</strong>, even those idle for a token.
Understand these three and you truly understand MoE: it is no free lunch but a swap of the "compute bill" for a "<strong>communication + memory + balance bill</strong>" — and SGLang's <span class="mono">FusedMoE</span>, expert parallelism (Lesson 46), and EPLB (Lesson 47) exist to shrink those three bills.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <strong>① An MoE layer = N small experts (FFNs) + one router</strong>: the router scores every <strong>token</strong> over experts and keeps only <span class="mono">top-k</span> (e.g. 2 of 64); only those k compute → <strong>sparse</strong>. So <strong>parameters soar with expert count while per-token compute is pinned by top-k</strong>. DeepSeek-V3 / Mixtral / Qwen-MoE all use it.
  <strong>② Four compute steps</strong>: route → group by expert → <strong>grouped / batched GEMM</strong> → scatter back and combine weighted by routing scores. SGLang's <span class="mono">FusedMoE</span> <strong>fuses routing + grouped GEMM into kernels</strong>, with no slow per-expert Python loop (kernels in Lesson 38).
  <strong>③ Scaling out</strong>: <strong>expert parallelism (EP, Lesson 46)</strong> spreads experts across GPUs; tokens all-to-all dispatch to the card holding their expert, then combine. <strong>EPLB (Lesson 47)</strong> balances "hot expert" load.
  <strong>④ Trade-off</strong>: a huge effective model for modest compute, at the cost of <strong>routing imbalance + all-to-all comm + memory to hold all experts</strong>. How to place this layer when writing a model: Lesson 26.
</div>
"""}

LESSON_35 = {
    "zh": r"""
<p class="lead">
模型又大又慢，很大程度上是因为它太"重"——几百亿个参数，每个都用 16 位浮点存着。<strong>量化（quantization）</strong>
是给这些数字"<strong>瘦身</strong>"的招式：用更少的比特（8 位、4 位）加上一个<strong>缩放因子</strong>来近似原来的权重，
既省显存、又省带宽，于是模型更小、跑得更快——代价只是<strong>一点点</strong>精度。本课讲清楚它为什么有效、有哪些格式、以及 SGLang 怎么把它<strong>插</strong>进线性层。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  量化就像给模型权重做 <strong>JPEG / MP3 压缩</strong>：原图每个像素用很多位精确存，压缩后每个值只留<strong>更少的比特</strong>，
  再配一个<strong>统一的缩放（scale）</strong>把范围还原回来。你损失了一丢丢看不太出来的细节，换来的是<strong>文件小得多、传输快得多</strong>。
  "<strong>分组缩放</strong>"就更像把图片切成小块分别压缩——每个小块保留自己的动态范围，失真更小。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  权重平时是 fp16/bf16（16 位）。量化把它们存成 <strong>8 位（FP8）甚至 4 位（INT4/FP4）</strong> + 一个 scale。
  <strong>更小的权重 = 更少显存</strong>（给 KV 缓存和并发腾地方，第 4/8 课）<strong>+ 更少的访存带宽</strong>（解码是访存密集的，第 4 课）<strong>⇒ 更快</strong>。
  在 SGLang 里，每种量化格式提供一个 <span class="mono">LinearMethod</span>，替换掉线性层"怎么存权重、怎么做矩阵乘"——和注意力后端（第 33 课）一样，量化也是<strong>可插拔的策略</strong>。
</div>

<h2>为什么量化能加速：显存与带宽</h2>
<p>很多人以为量化主要省的是算力，其实在 LLM 推理里，它省的<strong>大头是内存</strong>。第一，权重占的<strong>显存</strong>直接减半甚至减到四分之一——
70B 模型 fp16 要 140GB，换成 INT4 只要约 35GB，于是单卡能装下更大的模型，或者腾出显存给 KV 缓存、容纳更高并发（第 4/8 课）。
第二，也是更关键的：解码阶段是<strong>访存密集</strong>的（第 4 课），每生成一个 token 都要把<strong>整个模型权重</strong>从显存搬进计算单元一遍——
权重小一半，要搬的字节就少一半，<strong>带宽瓶颈直接缓解，解码就更快</strong>。所以量化对"延迟敏感、单请求解码"的场景收益尤其明显。
第三，在支持低精度计算的硬件上（如 H100/B200 的 FP8），还能直接用<strong>低精度矩阵乘内核</strong>把算力也一并省下。</p>

<p>把内存账算具体些：一个 70 亿参数（7B）的模型，fp16 下每个权重 2 字节，光权重就要约 14GB；换成 INT4，每个权重只占半个字节（外加每组一个很小的 scale），权重整体塌缩到约 3.5GB，<strong>连四分之一都不到</strong>。省下的这十来个 GB 不是凭空消失，而是直接变成更多的 <strong>KV 缓存槽位</strong>（第 8 课）：KV 池能多放几千个 token、多塞几十条并发请求，于是同一张卡的<strong>吞吐</strong>和<strong>可服务并发</strong>都水涨船高（第 4/8 课）。这也是为什么"量化省的是内存"在工程上往往比"省算力"更值钱——在 LLM 服务里，显存常常才是真正卡住并发的那道墙，而带宽才是拖慢解码的那根绳。</p>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="同一份权重在 fp16、fp8、int4 下的显存占用对比，位宽越低占用越小">
    <text x="20" y="30" style="font-weight:700;fill:var(--ink)">同一份权重 · 位宽越低越省显存</text>
    <text x="20" y="76" style="fill:var(--muted);font-size:13px">fp16 · 2B</text>
    <rect x="130" y="58" width="540" height="28" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="400" y="77" text-anchor="middle" style="font-size:12px">×1 基准</text>
    <text x="20" y="136" style="fill:var(--muted);font-size:13px">fp8 · 1B</text>
    <rect x="130" y="118" width="270" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="265" y="137" text-anchor="middle" style="font-size:12px">×½</text>
    <text x="20" y="196" style="fill:var(--muted);font-size:13px">int4 · 0.5B</text>
    <rect x="130" y="178" width="135" height="28" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="197" y="197" text-anchor="middle" style="font-size:12px">×¼</text>
    <text x="20" y="234" style="fill:var(--faint);font-size:12px">更小 = 更省显存 + 带宽</text>
  </svg>
  <div class="figcap"><b>图 1 · 位宽与显存：fp16 vs fp8 vs int4</b> — 同一份权重，fp16 每参数 2 字节为基准，fp8 减半（1 字节），int4 仅四分之一（0.5 字节）；更小 = 更省显存与访存带宽。</div>
</div>

<div class="card detail">
  <div class="tag">🧮 具体例子</div>
  <strong>例：一个 7B 模型。</strong>fp16 下权重约 <strong>14 GB</strong>（每参数 2 字节）；换成 <strong>fp8</strong> 约 <strong>7 GB</strong>（每参数 1 字节）；再压到 <strong>int4</strong> 只约 <strong>3.5 GB</strong>（每参数 0.5 字节）。启动时加 <span class="mono">--quantization fp8</span>，激活走<strong>动态 per-tensor</strong> 定标（或权重配 <span class="mono">[128,128]</span> 分块 scale），省下的十来 GB 直接变成更多 KV 缓存槽位与并发。
</div>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>原始 fp16 权重</h4><p>每个数 16 位，精确但又大又慢搬。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>定标 + 取整</h4><p>为一组数算一个 <span class="mono">scale</span>，把它们映射到 8/4 位整数或低精度浮点。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>紧凑存储</h4><p>显存里只存<strong>低位权重 + scale</strong>，体积小一半到四分之一。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>反量化 / 低精度矩阵乘</h4><p>要么先还原回高精度再算，要么直接跑量化 kernel（最快）。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 270" role="img" aria-label="量化流程：高精度权重→量化→低位存储→反量化或低位 kernel→矩阵乘">
    <text x="12" y="34" style="font-weight:700;fill:var(--ink)">量化 → 低位存储 → 用时反量化</text>
    <rect x="12" y="100" width="140" height="72" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="82" y="130" text-anchor="middle" style="font-size:12px">fp16 权重</text>
    <text x="82" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">高精度</text>
    <polygon points="154,131 166,136 154,141" style="fill:var(--muted)"/>
    <rect x="168" y="100" width="140" height="72" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="238" y="130" text-anchor="middle" style="font-size:12px">量化</text>
    <text x="238" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">scale + 转低位</text>
    <polygon points="310,131 322,136 310,141" style="fill:var(--muted)"/>
    <rect x="324" y="100" width="140" height="72" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="394" y="130" text-anchor="middle" style="font-size:12px">低位权重</text>
    <text x="394" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">+ scale 存储</text>
    <polygon points="466,131 478,136 466,141" style="fill:var(--muted)"/>
    <rect x="480" y="100" width="140" height="72" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="550" y="130" text-anchor="middle" style="font-size:12px">反量化</text>
    <text x="550" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">或低位 kernel</text>
    <polygon points="622,131 634,136 622,141" style="fill:var(--muted)"/>
    <rect x="636" y="100" width="140" height="72" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="706" y="130" text-anchor="middle" style="font-size:12px">矩阵乘</text>
    <text x="706" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">matmul</text>
    <text x="12" y="214" style="fill:var(--faint);font-size:12px">显存只存低位权重 + scale；计算时还原到可用范围</text>
  </svg>
  <div class="figcap"><b>图 2 · 量化 → 低位存储 → 用时反量化</b> — 高精度权重经量化（定 scale、转 fp8/int4）压成紧凑低位权重（连同 scale）；计算时再反量化（或直接跑低位 kernel）回到可用范围去做矩阵乘。</div>
</div>

<h2>有哪些格式：从 FP8 到 AWQ/GPTQ</h2>
<p>量化不是一种方法，而是一<strong>族</strong>。按比特数和"量化谁"可以铺开成下面这张表：</p>

<table class="t">
  <tr><th>格式</th><th>位数</th><th>量化对象</th><th>备注</th></tr>
  <tr><td><strong>FP8</strong>（E4M3）</td><td class="mono">8</td><td>权重 + 激活</td><td>H100/B200 硬件加速，可跑 FP8 GEMM，精度损失很小</td></tr>
  <tr><td><strong>FP4 / INT4</strong></td><td class="mono">4</td><td>权重（常）</td><td>压得最狠，省显存最多，需更细的 scale 保精度</td></tr>
  <tr><td><strong>AWQ</strong></td><td class="mono">4</td><td>仅权重</td><td>按激活重要性保护关键通道的免校准 PTQ</td></tr>
  <tr><td><strong>GPTQ</strong></td><td class="mono">4</td><td>仅权重</td><td>逐层最小化误差的校准式 PTQ</td></tr>
</table>

<p>一条关键区分是<strong>"只量化权重"还是"权重+激活都量化"</strong>。<strong>仅权重（weight-only，如 AWQ/GPTQ）</strong>：权重存成 4 位省显存，
但计算时<strong>先反量化回高精度</strong>再做矩阵乘——省了存和带宽，没省算力，对解码（访存密集）特别划算。<strong>权重+激活（如 FP8）</strong>：两边都压到低精度，
直接跑<strong>低精度矩阵乘内核</strong>（如 FP8 GEMM），<strong>连算力一起省</strong>，是吞吐场景的最爱，但对硬件和精度更挑剔。</p>

<div class="cols">
  <div class="col"><h4>仅权重量化（weight-only）</h4><p>权重 4 位、激活仍高精度；算前<strong>反量化</strong>。主要省<strong>显存 + 带宽</strong>。AWQ/GPTQ 属此类，对延迟敏感、单请求解码收益大。</p></div>
  <div class="col"><h4>权重+激活量化</h4><p>两边都低精度，跑<strong>低精度 GEMM</strong>（FP8）。<strong>显存 + 带宽 + 算力</strong>全省，吞吐最高；需要硬件支持，精度更要小心。</p></div>
</div>

<h2>为什么低比特还能用：冗余、离群值与 AWQ/GPTQ</h2>
<p>把权重从 16 位砍到 4 位，凭什么模型还几乎不掉点？根本原因是 LLM 的权重<strong>高度冗余、对噪声很宽容</strong>：几百亿参数里，绝大多数数值都挤在一个不大的范围内，单个权重抖动一点点，被后面层层加权求和一平均，对最终 logits 的影响微乎其微——这和给照片做有损压缩却几乎看不出来是一个道理。真正危险的是<strong>离群值（outlier）</strong>：极少数幅度特别大的权重或激活，如果用一个粗糙的 scale 去套整组，它们要么被<strong>截断（clip）</strong>而丢掉信息，要么逼着 scale 被迫拉大，从而让其余正常值的精度<strong>全被陪葬</strong>。所以现代量化方法的功夫，几乎全花在<strong>怎么伺候好这一小撮离群值</strong>上——这也是不同方法拉开质量差距的地方。</p>
<p><strong>AWQ</strong>（Activation-aware Weight Quantization）的洞见是：不是所有权重通道都一样重要，<strong>那些会乘上大激活的通道（salient channel）</strong>对输出影响最大。AWQ 用一小批校准数据统计各通道的激活幅度，给重要通道<strong>放大保护</strong>（等价于偷偷给它更细的有效精度）再量化，于是免去逐权重的繁重优化，却能保住关键信息，量化得又快又准。<strong>GPTQ</strong> 走的是另一条路：它带一个<strong>校准集</strong>，<strong>逐层</strong>地解一个最小二乘问题，一边量化一边补偿——每量化掉一列权重，就把引入的误差<strong>反向修正</strong>到尚未量化的列上去，让整层的<strong>重建误差</strong>（量化前后这一层输出之差）最小。两者都属于<strong>训练后量化（PTQ）</strong>：不重训模型，只用极少校准数据，几分钟到几十分钟，就能把一个 fp16 检查点压成一个高质量的 4 位版本，精度损失常常小到跑评测才看得出来。</p>
<p>那精度到底掉多少？经验上有个大致的台阶：<strong>FP8</strong> 几乎无损，常常和 bf16 在评测分数上看不出差别，所以在 H100/B200 上越来越像"默认就该开"的选项；<strong>INT4（AWQ/GPTQ）</strong>会掉一点点，通常在零点几个百分点的量级，对大多数应用完全可以接受，换来的却是三四倍的显存节省；只有压到更激进的 <strong>3 位甚至 2 位</strong>，质量才会明显劣化，需要更精巧的方法去抢救。一个常被忽略的细节是：模型<strong>越大越耐压</strong>——70B 量化到 4 位的相对损失，往往比 7B 量化到 4 位还小，因为大模型的冗余度更高、容错空间更大。这也解释了为什么生产里"大模型 + 激进量化"常常比"小模型 + 全精度"更划算。</p>

<h2>FP8 细看：E4M3、E5M2 与动态/静态定标</h2>
<p>FP8 不是一种格式而是两种，区别在那 8 个比特怎么<strong>分配给指数和尾数</strong>。<strong>E4M3</strong>（4 位指数、3 位尾数）尾数多、精度高，但能表示的范围偏窄，适合数值范围相对可控的<strong>权重和前向激活</strong>，是推理里最常用的一档。<strong>E5M2</strong>（5 位指数、2 位尾数）范围更大但精度更糙，更适合数值跨度极大的<strong>梯度</strong>，主要在训练里露脸。推理量化默认就盯着 <strong>E4M3</strong>：在 8 位里塞进一个"能用的浮点"，比同样 8 位的 INT8 多了<strong>自带动态范围</strong>这个好处，这也是 FP8 的精度损失常常比 INT8 还小的原因。</p>
<p>权重的 scale 在量化时就定死了，<strong>激活</strong>的 scale 却有两种玩法。<strong>动态定标（dynamic）</strong>在运行时按当前这一批激活的实际幅度临时算出 scale，精度最稳，但每一步都要多做一次求最大值的统计；<strong>静态定标（static）</strong>则用校准阶段预先算好的固定 scale，省掉运行时统计、最快，但碰上分布漂移就会差一点。算得快不快还要看<strong>kernel</strong>：SGLang 在有 FP8 张量核的硬件上会走 <span class="mono">cutlass</span> 的 FP8 GEMM 快路（代码里那句 <span class="mono">cutlass_fp8_supported()</span> 就是在探测这条路通不通），4 位权重则常借道 <span class="mono">Marlin</span> 这类专门的低位 GEMM kernel。<strong>H100 / B200</strong> 之所以是 FP8 的主场，是因为它们的 Tensor Core <strong>原生支持 FP8 矩阵乘</strong>——不是软件模拟，而是硬件一拍算一整片，于是低精度带来的算力优势才真正兑现，而不只是省了搬运。</p>

<h2>scale 是灵魂：粒度决定精度</h2>
<p>量化最核心的不是"砍位数"，而是那个<strong>缩放因子 scale</strong>——它决定了低位数能覆盖多大的数值范围。一组权重共用一个 scale，
范围跨度越大、低位表示越粗，<strong>scale 的粒度</strong>就成了精度与开销的取舍：<strong>per-tensor</strong>（整张权重一个 scale，最省、最糙）、
<strong>per-channel</strong>（每个输出通道一个 scale，常用，精度好不少）、<strong>per-group / blockwise</strong>（每一小块一个 scale，最细、精度最高，但 scale 本身也占点空间、算起来更复杂）。
好的量化方案（AWQ/GPTQ/FP8 配分组 scale）正是靠<strong>更聪明地选 scale</strong>，把精度损失压到几乎看不出来。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>一个权重块 + 它的 scale</b>：低位整数乘以共享 scale 才还原成近似的原值（per-group 量化）</div>
  <div class="cells"><span class="lab">int4 块</span><span class="cell">7</span><span class="cell">-3</span><span class="cell">5</span><span class="cell">0</span><span class="sep">×</span><span class="cell hl">scale=0.08</span><span class="sep">≈</span><span class="cell q">0.56 / -0.24 / 0.40 / 0.00</span></div>
</div>

<p>实践里粒度怎么选，是一道"精度 ÷ 开销"的算术题。<strong>per-tensor</strong> 最省、最快，但一旦权重里混进离群值，整张共用的 scale 就会被它撑大，把其余正常值的有效精度统统拉低，所以多用于对精度不敏感、追求极致速度的场合。<strong>per-channel</strong>（每个输出通道一个 scale）是个甜点：开销几乎可忽略，却能隔离掉"某几个通道天生幅度大"这种最常见的离群模式，因此成了 FP8 权重的常见默认。<strong>per-group / blockwise</strong>（如每 128 个元素一个 scale）最细，4 位量化几乎都得靠它才扛得住——位数已经被砍到只剩十几个台阶，必须用很贴合局部的 scale 才不至于把信息抹平。代价是这些 scale 自己也要存（通常用 fp16），还要在 GEMM 里随权重一起被读进来参与反量化，所以"更细"从来不是免费午餐，而是拿一点额外显存和算力去买精度。</p>

<h2>SGLang 怎么接：量化也是可插拔策略</h2>
<p>SGLang 不为每种格式重写模型，而是用一层抽象：每种格式有一个 <span class="mono">QuantizationConfig</span>，它提供一个 <span class="mono">LinearMethod</span>
（如 <span class="mono">Fp8LinearMethod</span>），<strong>替换掉线性层"怎么创建/存权重"和"怎么做这次矩阵乘"</strong>。于是模型文件（第 26 课）照常写
<span class="mono">RowParallelLinear</span> 之类的层，至于这层底下是 fp16 还是 fp8、走不走量化 kernel，全由配置注入——这正是继注意力后端（第 33 课）、KV 池之后，
SGLang"<strong>一切皆可插拔</strong>"的第三个例子。权重则由模型加载器（第 25 课）从<strong>预量化的 checkpoint</strong> 读入，或在加载时<strong>顺手量化</strong>。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/quantization/fp8.py ::Fp8LinearMethod</span><span class="ln">线性层的 FP8 方法</span></div>
  <pre><span class="kw">class</span> Fp8LinearMethod(LinearMethodBase):
    <span class="cm"># FP8 线性层方法，支持多种量化方案：</span>
    <span class="cm"># - 逐通道权重 + 逐 token 激活量化</span>
    <span class="cm"># - 逐张量权重 + 逐张量激活量化</span>
    <span class="cm"># - 分块(blockwise)权重 + 分块激活量化</span>
    <span class="cm"># 支持的 checkpoint：FP8，或 FP16/BF16（加载时即时量化到 FP8）</span>
    <span class="kw">def</span> __init__(self, quant_config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()  <span class="cm"># 有 FP8 硬件就走快路</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/quantization/fp8.py ::Fp8Config</span><span class="ln">FP8 配置：激活量化方案、按块缩放、忽略层</span></div>
  <pre><span class="kw">class</span> Fp8Config(QuantizationConfig):
    <span class="cm"># 告诉 SGLang 该怎么用 FP8 跑这个模型。</span>
    <span class="kw">def</span> __init__(self, is_checkpoint_fp8_serialized=False,
                 activation_scheme="dynamic",  <span class="cm"># "dynamic" 或 "static"</span>
                 ignored_layers=None,
                 weight_block_size=None):       <span class="cm"># 如 [128,128] 分块 scale</span>
        ...
    <span class="cm"># -&gt; 给每个 Linear 层一个 Fp8LinearMethod 来量化它。</span></pre>
</div>

<p>还有一个<strong>相关但独立</strong>的旋钮：<strong>KV 缓存量化</strong>。前面说的都是给<strong>权重</strong>瘦身，但解码时不断堆积的 <strong>KV 缓存</strong>（第 8 课）本身也是显存大户，尤其在长上下文、高并发下，它往往比权重还吃显存。把 KV 也从 fp16 压到 FP8 甚至更低，能让 KV 池一下子多装将近一倍的 token，<strong>直接拔高可服务的上下文长度与并发数</strong>。它和权重量化是<strong>两套独立的配置</strong>：你可以只量化权重、只量化 KV、或者两个一起开——因为它们压的是不同的东西、走的也是不同的代码路径（一个落在线性层的 <span class="mono">LinearMethod</span> 上，一个落在 KV 池的存取上）。把这两个旋钮分开来看，能帮你在"省显存"和"保精度"之间更精细地调。</p>

<p>退一步看，量化是 SGLang"<strong>一切皆可插拔</strong>"哲学的<strong>第三个</strong>样板：第一个是注意力后端（第 33 课）——同一个注意力层，底下换 FlashAttention 还是 FlashInfer，由配置说了算；第二个是 KV 池——同一套读写接口，底下可以是连续、分页或分层缓存；现在是量化——同一个 <span class="mono">RowParallelLinear</span>，底下是 fp16 还是 fp8、走不走量化 kernel，同样由一个 <span class="mono">QuantizationConfig</span> 注入，模型文件（第 26 课）一个字都不用改。这正是<strong>第 8 部分</strong>的主线：把 Transformer 的每一块——注意力、MoE（第 34 课）、量化、各类算子（第 36 课）——都打磨成<strong>可替换的策略</strong>，于是新格式、新硬件、新算法都能直接"插"进来，而不必每次都重写模型。理解了这层抽象，你看 SGLang 加一种新量化方案，就只是"再写一个 <span class="mono">LinearMethod</span>"那么自然，而整套服务调度、KV 管理、注意力计算都原封不动地复用。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>量化</strong>：用更少比特（FP8/FP4/INT4）+ 一个 scale 近似权重，<strong>省显存、省带宽 ⇒ 更快</strong>，代价是一点点精度。</li>
    <li><strong>省的是内存</strong>：显存减半到四分之一 + 解码访存带宽减半（第 4/8 课）；低精度硬件还能省算力。</li>
    <li><strong>两类</strong>：仅权重（AWQ/GPTQ，算前反量化，省存+带宽）vs 权重+激活（FP8 GEMM，连算力一起省）。</li>
    <li><strong>scale 粒度</strong>：per-tensor / per-channel / per-group，越细精度越高、开销越大。</li>
    <li><strong>可插拔</strong>：<span class="mono">QuantizationConfig</span> 提供 <span class="mono">LinearMethod</span> 替换线性层的存权重/矩阵乘；由加载器（第 25 课）读入。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
Models are big and slow largely because they're "heavy" — tens of billions of parameters, each stored in 16-bit float.
<strong>Quantization</strong> is how you put those numbers on a <strong>diet</strong>: approximate the original weights with
<strong>fewer bits</strong> (8, 4) plus a <strong>scale factor</strong>, saving HBM and bandwidth so the model is smaller and faster —
at the cost of <strong>a little</strong> accuracy. This lesson covers why it works, which formats exist, and how SGLang <strong>plugs</strong> it into linear layers.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Quantization is like <strong>JPEG / MP3 compression</strong> for weights: instead of storing each value at full precision,
  keep <strong>fewer bits</strong> plus a shared <strong>scale</strong> to restore the range. You lose a sliver of barely-noticeable
  detail in exchange for a file that's <strong>far smaller and moves far faster</strong>. "<strong>Group scaling</strong>" is like
  compressing the image in small blocks — each block keeps its own dynamic range, so distortion stays low.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  Weights are normally fp16/bf16 (16-bit). Quantization stores them in <strong>8-bit (FP8) or even 4-bit (INT4/FP4)</strong> + a scale.
  <strong>Smaller weights = less HBM</strong> (room for KV cache and concurrency, Lessons 4/8) <strong>+ less memory bandwidth</strong>
  (decode is memory-bound, Lesson 4) <strong>⇒ faster</strong>. In SGLang each format provides a <span class="mono">LinearMethod</span> that
  replaces how a linear layer stores weights and does its matmul — like the attention backend (Lesson 33), quantization is a <strong>pluggable strategy</strong>.
</div>

<h2>Why quantization speeds things up: HBM and bandwidth</h2>
<p>People assume quantization mainly saves compute, but in LLM inference the <strong>big win is memory</strong>. First, the weights'
<strong>HBM</strong> footprint halves or quarters — a 70B model is 140GB in fp16 but ~35GB in INT4, so a GPU fits a bigger model, or
frees HBM for the KV cache and higher concurrency (Lessons 4/8). Second, and more importantly: decode is <strong>memory-bound</strong>
(Lesson 4) — every generated token drags the <strong>whole weight set</strong> from HBM into the compute units once; halve the bytes and
you halve the move, so the <strong>bandwidth bottleneck eases and decode gets faster</strong>. So quantization shines for latency-sensitive,
single-request decode. Third, on hardware with low-precision compute (FP8 on H100/B200), a <strong>low-precision matmul kernel</strong> saves FLOPs too.</p>

<p>Make the memory math concrete: a 7-billion-parameter (7B) model at fp16 stores each weight in 2 bytes, so the weights alone need ~14GB; in INT4 each weight is half a byte (plus a tiny per-group scale), collapsing the weights to ~3.5GB — <strong>under a quarter</strong>. Those reclaimed gigabytes don't vanish; they turn directly into more <strong>KV-cache slots</strong> (Lesson 8): the KV pool holds thousands more tokens and dozens more concurrent requests, so the same GPU's <strong>throughput</strong> and <strong>serviceable concurrency</strong> both climb (Lessons 4/8). That's why "quantization saves memory" is usually worth more in practice than "saves compute" — in LLM serving, HBM is often the wall that actually caps concurrency, and bandwidth is the rope that drags decode down.</p>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="same weights compared in fp16, fp8 and int4: lower bit-width uses less memory">
    <text x="20" y="30" style="font-weight:700;fill:var(--ink)">Same weights · lower bits, less VRAM</text>
    <text x="20" y="76" style="fill:var(--muted);font-size:13px">fp16 · 2B</text>
    <rect x="130" y="58" width="540" height="28" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="400" y="77" text-anchor="middle" style="font-size:12px">×1 base</text>
    <text x="20" y="136" style="fill:var(--muted);font-size:13px">fp8 · 1B</text>
    <rect x="130" y="118" width="270" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="265" y="137" text-anchor="middle" style="font-size:12px">×½</text>
    <text x="20" y="196" style="fill:var(--muted);font-size:13px">int4 · 0.5B</text>
    <rect x="130" y="178" width="135" height="28" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="197" y="197" text-anchor="middle" style="font-size:12px">×¼</text>
    <text x="20" y="234" style="fill:var(--faint);font-size:12px">smaller = less VRAM + bandwidth</text>
  </svg>
  <div class="figcap"><b>Fig 1 · bit-width vs memory: fp16 vs fp8 vs int4</b> — same weights: fp16 is 2 bytes/param (baseline), fp8 halves it (1 byte), int4 is a quarter (0.5 byte); smaller = less VRAM and memory bandwidth.</div>
</div>

<div class="card detail">
  <div class="tag">🧮 Concrete example</div>
  <strong>Example: a 7B model.</strong> In fp16 the weights are ~<strong>14 GB</strong> (2 bytes/param); in <strong>fp8</strong> ~<strong>7 GB</strong> (1 byte/param); in <strong>int4</strong> only ~<strong>3.5 GB</strong> (0.5 byte/param). Launch with <span class="mono">--quantization fp8</span> and activations use <strong>dynamic per-tensor</strong> scaling (or weights with <span class="mono">[128,128]</span> block scales) — the dozen-odd GB reclaimed turns straight into more KV-cache slots and concurrency.
</div>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Original fp16 weight</h4><p>16 bits each — precise but big and slow to move.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Scale + round</h4><p>Compute one <span class="mono">scale</span> for a group, map them to 8/4-bit ints or low-precision floats.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Compact storage</h4><p>HBM holds only <strong>low-bit weights + scale</strong>, half to a quarter the size.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Dequant / low-precision matmul</h4><p>Either restore to high precision first, or run a quantized kernel (fastest).</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 270" role="img" aria-label="quantization flow: high-precision weights, quantize, store low-bit, dequantize or low-bit kernel, matmul">
    <text x="12" y="34" style="font-weight:700;fill:var(--ink)">quantize → store low-bit → dequant</text>
    <rect x="12" y="100" width="140" height="72" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="82" y="130" text-anchor="middle" style="font-size:12px">fp16 weights</text>
    <text x="82" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">high precision</text>
    <polygon points="154,131 166,136 154,141" style="fill:var(--muted)"/>
    <rect x="168" y="100" width="140" height="72" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="238" y="130" text-anchor="middle" style="font-size:12px">quantize</text>
    <text x="238" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">scale + cast</text>
    <polygon points="310,131 322,136 310,141" style="fill:var(--muted)"/>
    <rect x="324" y="100" width="140" height="72" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="394" y="130" text-anchor="middle" style="font-size:12px">low-bit weights</text>
    <text x="394" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">+ scales stored</text>
    <polygon points="466,131 478,136 466,141" style="fill:var(--muted)"/>
    <rect x="480" y="100" width="140" height="72" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="550" y="130" text-anchor="middle" style="font-size:12px">dequantize</text>
    <text x="550" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">or low-bit kernel</text>
    <polygon points="622,131 634,136 622,141" style="fill:var(--muted)"/>
    <rect x="636" y="100" width="140" height="72" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="706" y="130" text-anchor="middle" style="font-size:12px">matmul</text>
    <text x="706" y="150" text-anchor="middle" style="fill:var(--muted);font-size:11px">GEMM</text>
    <text x="12" y="214" style="fill:var(--faint);font-size:12px">HBM stores only low-bit weights + scales; restore on use</text>
  </svg>
  <div class="figcap"><b>Fig 2 · quantize → store low-bit → dequantize on use</b> — high-precision weights are quantized (pick a scale, cast to fp8/int4) into compact low-bit weights (with their scales); at compute they're dequantized (or fed to a low-bit kernel) back to a usable range for the matmul.</div>
</div>

<h2>The formats: from FP8 to AWQ/GPTQ</h2>
<p>Quantization isn't one method but a <strong>family</strong>. By bit-count and "what gets quantized" it spreads into this table:</p>

<table class="t">
  <tr><th>Format</th><th>Bits</th><th>Quantizes</th><th>Notes</th></tr>
  <tr><td><strong>FP8</strong> (E4M3)</td><td class="mono">8</td><td>weights + activations</td><td>HW-accelerated on H100/B200, can run FP8 GEMM, tiny accuracy loss</td></tr>
  <tr><td><strong>FP4 / INT4</strong></td><td class="mono">4</td><td>weights (usually)</td><td>most compression, most HBM saved; needs finer scales for accuracy</td></tr>
  <tr><td><strong>AWQ</strong></td><td class="mono">4</td><td>weight-only</td><td>calibration-light PTQ protecting salient channels by activation importance</td></tr>
  <tr><td><strong>GPTQ</strong></td><td class="mono">4</td><td>weight-only</td><td>calibrated PTQ minimizing per-layer error</td></tr>
</table>

<p>A key distinction is <strong>"weight-only" vs "weight + activation"</strong>. <strong>Weight-only (AWQ/GPTQ)</strong>: weights stored in 4 bits
to save HBM, but compute <strong>dequantizes back to high precision</strong> first — saving storage and bandwidth, not FLOPs, which is great for
memory-bound decode. <strong>Weight + activation (FP8)</strong>: both sides low precision, running a <strong>low-precision matmul kernel</strong>
(FP8 GEMM) — <strong>saving FLOPs too</strong>, the favorite for throughput, but pickier about hardware and accuracy.</p>

<div class="cols">
  <div class="col"><h4>Weight-only</h4><p>4-bit weights, high-precision activations; <strong>dequant before compute</strong>. Saves <strong>HBM + bandwidth</strong>. AWQ/GPTQ are here — big win for latency-sensitive single-request decode.</p></div>
  <div class="col"><h4>Weight + activation</h4><p>Both low precision, run a <strong>low-precision GEMM</strong> (FP8). Saves <strong>HBM + bandwidth + FLOPs</strong>, highest throughput; needs hardware support, accuracy needs more care.</p></div>
</div>

<h2>Why low bits still work: redundancy, outliers, AWQ/GPTQ</h2>
<p>Cut weights from 16 bits to 4 and the model barely loses quality — how? The root reason is that LLM weights are <strong>highly redundant and tolerant of noise</strong>: among tens of billions of parameters, the vast majority sit in a modest range, and one weight wobbling slightly gets averaged away through layer after layer of weighted sums, with negligible effect on the final logits — the same reason a lossy-compressed photo still looks fine. The real danger is <strong>outliers</strong>: the rare weights or activations with unusually large magnitude; if a coarse scale tries to cover the whole group, they get <strong>clipped</strong> and lose information, or they force the scale large and <strong>drag down the precision of every normal value with them</strong>. So nearly all the effort in modern quantization goes into <strong>handling that handful of outliers</strong> — which is where methods pull apart in quality.</p>
<p><strong>AWQ</strong> (Activation-aware Weight Quantization)'s insight: not all weight channels matter equally — the <strong>salient channels</strong> that get multiplied by large activations dominate the output. AWQ uses a small calibration batch to measure each channel's activation magnitude, <strong>scales up and protects</strong> the important channels (effectively giving them finer precision) before quantizing, so it skips heavy per-weight optimization yet keeps the critical information, quantizing fast and accurately. <strong>GPTQ</strong> takes another route: with a <strong>calibration set</strong> it solves a least-squares problem <strong>layer by layer</strong>, quantizing and compensating as it goes — each time it quantizes a column of weights it <strong>back-corrects</strong> the introduced error onto the not-yet-quantized columns, minimizing the layer's <strong>reconstruction error</strong> (the output difference before vs after). Both are <strong>post-training quantization (PTQ)</strong>: no retraining, just a little calibration data, turning an fp16 checkpoint into a high-quality 4-bit version in minutes, with accuracy loss often so small you only see it on a benchmark.</p>
<p>So how much accuracy actually drops? Empirically there's a rough staircase: <strong>FP8</strong> is nearly lossless, often indistinguishable from bf16 on benchmark scores, so on H100/B200 it increasingly feels like a "just turn it on" default; <strong>INT4 (AWQ/GPTQ)</strong> gives up a little, usually on the order of a few tenths of a percent — perfectly acceptable for most applications in exchange for a 3–4× HBM saving; only at more aggressive <strong>3-bit or even 2-bit</strong> does quality degrade noticeably and need fancier methods to rescue. One often-missed detail: <strong>bigger models compress better</strong> — a 70B quantized to 4 bits typically loses relatively less than a 7B at 4 bits, because the larger model is more redundant and has more error budget. That's why in production "big model + aggressive quant" often beats "small model + full precision."</p>

<h2>FP8 up close: E4M3, E5M2, and dynamic vs static scaling</h2>
<p>FP8 isn't one format but two, differing in how the 8 bits are <strong>split between exponent and mantissa</strong>. <strong>E4M3</strong> (4 exponent, 3 mantissa bits) has more mantissa, so higher precision but a narrower representable range — good for the relatively bounded <strong>weights and forward activations</strong>, and the most common choice for inference. <strong>E5M2</strong> (5 exponent, 2 mantissa) reaches a wider range at coarser precision, better for wide-ranging <strong>gradients</strong>, mostly seen in training. Inference quantization targets <strong>E4M3</strong> by default: packing a usable float into 8 bits brings <strong>built-in dynamic range</strong> that same-width INT8 lacks, which is why FP8 often loses even less accuracy than INT8.</p>
<p>A weight's scale is fixed at quantization time, but an <strong>activation's</strong> scale has two styles. <strong>Dynamic scaling</strong> computes the scale at runtime from the current batch's actual magnitude — most robust, but adds a max-reduction every step; <strong>static scaling</strong> uses a fixed scale precomputed during calibration — no runtime stats, fastest, but drifts a bit if the distribution shifts. Speed also hinges on the <strong>kernel</strong>: on hardware with FP8 tensor cores SGLang takes the <span class="mono">cutlass</span> FP8 GEMM fast path (the <span class="mono">cutlass_fp8_supported()</span> line is probing exactly whether that path is open), while 4-bit weights often go through a dedicated low-bit GEMM kernel like <span class="mono">Marlin</span>. <strong>H100 / B200</strong> are FP8's home turf because their Tensor Cores <strong>natively support FP8 matmul</strong> — not software emulation but hardware computing a whole tile per clock, which is how the low-precision compute advantage actually materializes instead of merely saving data movement.</p>

<h2>The scale is the soul: granularity sets accuracy</h2>
<p>The heart of quantization isn't "fewer bits" but that <strong>scale factor</strong> — it sets how big a numeric range the low bits cover.
A group of weights shares one scale, so the wider the range and coarser the low-bit representation, the more the <strong>scale granularity</strong>
becomes the accuracy-vs-overhead trade: <strong>per-tensor</strong> (one scale for the whole weight, cheapest, coarsest), <strong>per-channel</strong>
(one per output channel, common, noticeably better), <strong>per-group / blockwise</strong> (one per small block, finest, best accuracy, but the
scales cost a bit of space and complexity). Good schemes (AWQ/GPTQ/FP8 with group scales) win by <strong>choosing scales more cleverly</strong>, keeping the accuracy loss nearly invisible.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>A weight block + its scale</b>: low-bit ints times the shared scale recover the approximate originals (per-group quant)</div>
  <div class="cells"><span class="lab">int4 block</span><span class="cell">7</span><span class="cell">-3</span><span class="cell">5</span><span class="cell">0</span><span class="sep">×</span><span class="cell hl">scale=0.08</span><span class="sep">≈</span><span class="cell q">0.56 / -0.24 / 0.40 / 0.00</span></div>
</div>

<p>Choosing granularity in practice is an "accuracy ÷ overhead" calculation. <strong>per-tensor</strong> is cheapest and fastest, but one outlier in the weights stretches the whole shared scale and drags down the effective precision of every other value, so it suits accuracy-insensitive, speed-at-all-costs cases. <strong>per-channel</strong> (one scale per output channel) is the sweet spot: near-zero overhead, yet it isolates the most common outlier pattern — "a few channels are just naturally large" — which is why it's the common default for FP8 weights. <strong>per-group / blockwise</strong> (e.g. one scale per 128 elements) is finest, and 4-bit quantization almost always needs it to survive — with only a dozen-odd levels left, the scale must hug the local range or it smears the information away. The cost: those scales must be stored too (usually fp16) and read in alongside the weights during the GEMM to dequantize, so "finer" is never a free lunch — it spends a bit of extra HBM and compute to buy accuracy.</p>

<h2>How SGLang plugs it in: quantization as a pluggable strategy</h2>
<p>SGLang doesn't rewrite models per format; it uses one abstraction: each format has a <span class="mono">QuantizationConfig</span> that
provides a <span class="mono">LinearMethod</span> (e.g. <span class="mono">Fp8LinearMethod</span>) which <strong>replaces how a linear layer
creates/stores its weight and does its matmul</strong>. So the model file (Lesson 26) still writes layers like
<span class="mono">RowParallelLinear</span>; whether that layer is fp16 or fp8 underneath, and whether it runs a quantized kernel, is injected
by config — the third example (after the attention backend, Lesson 33, and the KV pool) of SGLang's <strong>"everything pluggable"</strong>.
Weights are read by the model loader (Lesson 25) from a <strong>pre-quantized checkpoint</strong>, or <strong>quantized on the fly</strong> at load.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/quantization/fp8.py ::Fp8LinearMethod</span><span class="ln">the FP8 method for a linear layer</span></div>
  <pre><span class="kw">class</span> Fp8LinearMethod(LinearMethodBase):
    <span class="cm"># FP8 linear method, supporting several quant schemes:</span>
    <span class="cm"># - per-channel weight + per-token activation quant</span>
    <span class="cm"># - per-tensor weight + per-tensor activation quant</span>
    <span class="cm"># - blockwise weight + blockwise activation quant</span>
    <span class="cm"># checkpoints: FP8, or FP16/BF16 (quantized to FP8 at load time)</span>
    <span class="kw">def</span> __init__(self, quant_config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()  <span class="cm"># fast path if FP8 HW exists</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/quantization/fp8.py ::Fp8Config</span><span class="ln">FP8 config: activation scheme, block scaling, ignored layers</span></div>
  <pre><span class="kw">class</span> Fp8Config(QuantizationConfig):
    <span class="cm"># tells SGLang HOW to run a model in FP8.</span>
    <span class="kw">def</span> __init__(self, is_checkpoint_fp8_serialized=False,
                 activation_scheme="dynamic",  <span class="cm"># "dynamic" or "static"</span>
                 ignored_layers=None,
                 weight_block_size=None):       <span class="cm"># e.g. [128,128] block scales</span>
        ...
    <span class="cm"># -&gt; hands each Linear layer an Fp8LinearMethod to quantize it.</span></pre>
</div>

<p>There's a <strong>related but separate</strong> knob: <strong>KV-cache quantization</strong>. Everything above slimmed the <strong>weights</strong>, but the <strong>KV cache</strong> that piles up during decode (Lesson 8) is itself an HBM hog — under long context and high concurrency it often eats more memory than the weights. Compressing KV from fp16 to FP8 or lower lets the KV pool hold nearly twice the tokens, <strong>directly raising serviceable context length and concurrency</strong>. It's a <strong>separate configuration</strong> from weight quantization: you can quantize only weights, only KV, or both — because they compress different things on different code paths (one in the linear layer's <span class="mono">LinearMethod</span>, the other in the KV pool's load/store). Keeping the two knobs distinct lets you tune the "save HBM" vs "keep accuracy" trade more finely.</p>

<p>Step back and quantization is the <strong>third</strong> exemplar of SGLang's <strong>"everything pluggable"</strong> philosophy: the first was the attention backend (Lesson 33) — one attention layer, FlashAttention or FlashInfer underneath chosen by config; the second was the KV pool — one read/write interface over contiguous, paged, or tiered caches; now quantization — one <span class="mono">RowParallelLinear</span>, fp16 or fp8 underneath and quant-kernel or not, injected by a single <span class="mono">QuantizationConfig</span>, with the model file (Lesson 26) unchanged. That's the through-line of <strong>Part 8</strong>: make every Transformer block — attention, MoE (Lesson 34), quantization, the various ops (Lesson 36) — a <strong>swappable strategy</strong>, so new formats, hardware, and algorithms can "plug in" without rewriting the model each time. Once you see this abstraction, SGLang adding a new quantization scheme is just "write one more <span class="mono">LinearMethod</span>," while the whole serving scheduler, KV management, and attention compute are reused untouched.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Quantization</strong>: approximate weights with fewer bits (FP8/FP4/INT4) + a scale — <strong>saves HBM & bandwidth ⇒ faster</strong>, at a little accuracy.</li>
    <li><strong>The win is memory</strong>: HBM halved-to-quartered + decode bandwidth halved (Lessons 4/8); low-precision hardware saves FLOPs too.</li>
    <li><strong>Two kinds</strong>: weight-only (AWQ/GPTQ, dequant before compute, saves storage+bandwidth) vs weight+activation (FP8 GEMM, saves FLOPs too).</li>
    <li><strong>Scale granularity</strong>: per-tensor / per-channel / per-group — finer = more accurate, more overhead.</li>
    <li><strong>Pluggable</strong>: <span class="mono">QuantizationConfig</span> provides a <span class="mono">LinearMethod</span> replacing the linear layer's weight storage/matmul; read by the loader (Lesson 25).</li>
  </ul>
</div>
""",
}

LESSON_36 = {"zh": r"""
<p class="lead">
第 26 课你会发现，写一个 Transformer 模型其实就是把一堆<strong>现成的层拼起来</strong>：注意力是一层、MLP 是一层、归一化是一层、给位置编码的 RoPE 也是一层。前面几课讲的注意力（第 33 课）、MoE（第 34 课）、量化（第 35 课）是大块头，
这一课要补上那些<strong>更小但同样不可或缺</strong>的算子——RoPE、RMSNorm、以及一批<strong>融合算子</strong>。它们都被打磨成可复用的 <span class="mono">nn.Module</span>，模型只管<strong>调用</strong>，底下跑哪个平台的哪个 kernel、要不要融合，由这些层自己挑。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 <strong>RoPE</strong> 想成给每个词的<strong>指南针</strong>拨一个角度：词在句子里越靠后，指针就被多转一点。两个词之间感受到的"距离"，就是两根指针的<strong>夹角</strong>——
  谁都没有一个绝对的"门牌号"，但只要看夹角，就知道彼此<strong>相隔多远</strong>。再把 <strong>RMSNorm</strong> 想成一个<strong>轻量的音量平衡器</strong>：它不挪动音乐的"基线"（不减均值），只是把每段声音的<strong>整体响度</strong>拉到统一水平，让后面的层听得清楚、训练得稳。
  指南针负责"我在哪、我们隔多远"，音量平衡器负责"大家响度一致"——两个小零件，撑起了整座 Transformer 的稳定运转。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  记住一句话：<strong>这些小算子，是模型用来"搭积木"的标准件</strong>。它们和注意力后端（第 33 课）、量化（第 35 课）同属一个主题——<strong>可插拔的、硬件优化过的层</strong>。
  模型文件（第 26 课）里只写 <span class="mono">self.rotary_emb(positions, q, k)</span>、<span class="mono">self.input_layernorm(x)</span> 这样的<strong>调用</strong>，至于底下是 CUDA 融合 kernel、Triton 实现，还是 AMD/NPU 的专属版本，
  全由这些层在运行时按<strong>平台</strong>自动挑选。于是同一份模型代码，换块卡就能跑；新出一个更快的融合 kernel，也只要在层内部接上，模型一个字都不用改。
</div>

<h2>RoPE：用"旋转"给注意力注入位置</h2>
<p>先说一个反直觉的事实：<strong>注意力本身没有顺序概念</strong>。注意力算的是 query 和 key 的点积，而点积对"谁先谁后"是<strong>不敏感</strong>的——把句子里的词打乱，纯注意力算出来的关系几乎一样。可语言显然<strong>高度依赖顺序</strong>，"狗咬人"和"人咬狗"天差地别。
所以必须有人把<strong>位置信息</strong>喂进去。<strong>RoPE（Rotary Position Embedding，旋转位置编码）</strong>给出的答案非常优雅：不往向量里<strong>加</strong>一个位置向量，而是按位置把每个 q/k 向量<strong>旋转</strong>一个角度——位置越靠后，转得越多。</p>
<p>妙处在于点积的数学。把 query 旋转它所在位置的角度、把 key 旋转它所在位置的角度，再做点积，结果只取决于两个角度的<strong>差</strong>，也就是两个 token 的<strong>相对位置</strong>。于是注意力一下子变得<strong>相对位置感知</strong>：模型关心的不再是"这个词在第几个绝对位置"，而是"这两个词<strong>相隔多远</strong>"。
这正符合语言的本质——一个代词指向几个词之前的名词，靠的是<strong>距离</strong>而非绝对坐标。RoPE 不引入任何要学习的额外参数，纯靠固定频率的旋转就把相对位置"焊"进了注意力，既省参数又泛化得好，因此成了 Llama、Qwen 等主流模型的标配。</p>
<p>落到代码里，RoPE 是一个 <span class="mono">RotaryEmbedding</span> 层，构造时用 <span class="mono">get_rope(...)</span> 按配置造出对应变体，前向时<strong>紧贴在注意力之前</strong>对 q/k 施加旋转（注意力本身见第 33 课）。它预先算好每个位置的 <span class="mono">cos/sin</span> 缓存，前向时按 token 的 <span class="mono">positions</span> 把对应的 cos/sin 取出来，旋转向量的前 <span class="mono">rotary_dim</span> 维，剩下的维度原样透传。整个过程没有矩阵乘，只是一组逐元素的乘加，因此非常便宜，常常还会被<strong>融合</strong>进相邻算子里一并完成。</p>
<p>这里还藏着一个容易忽略的设计：旋转<strong>不是把整个头维度都转</strong>，而只转前 <span class="mono">rotary_dim</span> 维，后面的维度原封不动透传。为什么要留一截不转？因为 RoPE 的本意是把<strong>位置</strong>编码进一部分子空间，让模型既能感知相对位置、又保留一部分<strong>与位置无关</strong>的特征通道，二者各司其职。另一个细节是 <span class="mono">is_neox_style</span>——它决定向量里哪些维度<strong>配成一对</strong>一起旋转（相邻配对，还是前后半配对），不同模型家族的约定不同，但都收敛在同一个 <span class="mono">RotaryEmbedding</span> 接口下。对模型作者来说，这些都被封装在层里：你写模型时只调用 <span class="mono">self.rotary_emb(positions, q, k)</span>，至于转多少维、怎么配对、cos/sin 怎么缓存，全不用操心。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>拿到 q / k</h4><p>注意力前，先有投影出来的 query、key 向量——但它们此刻<strong>不含任何位置信息</strong>。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>按位置施加 RoPE</h4><p>用每个 token 的 <span class="mono">positions</span> 取出对应 <span class="mono">cos/sin</span>，把 q/k 各<strong>旋转</strong>一个 ∝ 位置的角度。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>点积变成相对感知</h4><p>旋转后的 q·k 只依赖两者位置之<strong>差</strong>——注意力自动获得<strong>相对位置</strong>感。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>送入注意力</h4><p>带位置的 q/k 交给注意力后端（第 33 课）算 softmax 加权——顺序信息已经在里面了。</p></div></div>
</div>

<p>再把"旋转"这件事画得更直观一些：同一个向量，在位置 0 不转、位置 1 转一点、位置 2 转得更多——位置每往后一步，指针就多偏一个固定角度。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>同一个向量，被不同位置旋转不同角度</b>：位置越靠后，转得越多（角度 ∝ 位置）</div>
  <div class="cells"><span class="lab">向量 v</span><span class="cell">pos 0 → 0°</span><span class="sep">→</span><span class="cell">pos 1 → 30°</span><span class="sep">→</span><span class="cell hl">pos 2 → 60°</span><span class="sep">⇒</span><span class="cell q">q·k 只看夹角差 = 相对位置</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="RoPE 按位置旋转 q/k：同一向量在不同位置被旋转不同角度，点积只依赖两角之差即相对位置">
    <text x="24" y="34" style="font-weight:700;fill:var(--muted)">旋转平面 · 单位圆</text>
    <line x1="84" y1="170" x2="318" y2="170" style="stroke:var(--faint);stroke-width:1.5"/>
    <line x1="195" y1="58" x2="195" y2="282" style="stroke:var(--faint);stroke-width:1.5"/>
    <circle cx="195" cy="170" r="108" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="195" y1="170" x2="303" y2="170" style="stroke:var(--blue);stroke-width:2.5"/>
    <circle cx="303" cy="170" r="4" style="fill:var(--blue)"/>
    <text x="282" y="190" class="mono" style="fill:var(--blue);font-weight:700;font-size:12px">q@p0</text>
    <line x1="195" y1="170" x2="288" y2="116" style="stroke:var(--teal);stroke-width:2.5"/>
    <circle cx="288" cy="116" r="4" style="fill:var(--teal)"/>
    <text x="294" y="110" class="mono" style="fill:var(--teal);font-weight:700;font-size:12px">q@p1</text>
    <line x1="195" y1="170" x2="249" y2="77" style="stroke:var(--amber);stroke-width:2.5"/>
    <circle cx="249" cy="77" r="4" style="fill:var(--amber)"/>
    <text x="220" y="64" class="mono" style="fill:var(--amber);font-weight:700;font-size:12px">q@p2</text>
    <path d="M237,170 A42,42 0 0 0 216,134" style="fill:none;stroke:var(--amber);stroke-width:1.5"/>
    <text x="240" y="150" style="fill:var(--amber);font-size:12px">θ∝pos</text>
    <rect x="430" y="92" width="320" height="120" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="452" y="122" style="font-weight:700;fill:var(--accent-ink)">点积只看夹角差</text>
    <text x="452" y="154" class="mono" style="font-size:13px">q·k ∝ cos(θq − θk)</text>
    <text x="452" y="182" class="mono" style="font-size:13px">θq − θk ∝ p − p′</text>
    <text x="452" y="206" style="fill:var(--teal);font-weight:700">⇒ 相对位置</text>
  </svg>
  <div class="figcap"><b>图 1 · RoPE 按位置旋转 q/k</b> — 同一向量在 p0/p1/p2 被旋转 0°/30°/60°（角度 ∝ 位置）；q 与 k 点积只依赖两角之差，于是注意力只感受<strong>相对位置</strong>。</div>
</div>

<h2>RoPE 如何撑长上下文</h2>
<p>RoPE 还顺手解决了一个工程上的大难题：<strong>上下文长度扩展</strong>。一个模型训练时只见过比如 4k token 的序列，旋转角度的"频率表"也是按 4k 调好的。直接拿去服务 32k 的长文本，靠后位置的旋转角度会跑到训练时<strong>从没见过的范围</strong>，模型一脸懵，效果崩坏。
解决办法不是重训，而是<strong>巧妙地拉伸旋转频率</strong>，让 4k 训出来的模型也能在 32k 甚至更长上从容工作。</p>
<p>常见的几招都是 RoPE 的变体：<strong>线性缩放（linear scaling）</strong>最朴素——把所有位置等比例压缩，相当于把"尺子"整体拉长，让 32k 的位置落回模型熟悉的角度范围，简单但高频信息会被牺牲一些；<strong>NTK-aware</strong> 更聪明，它<strong>不均匀</strong>地调整不同频率，对高频（管局部细节）改动小、对低频（管长程关系）拉伸大，从而在扩展长度的同时尽量保住短距离的分辨力；
<strong>YaRN</strong> 则在 NTK 的基础上再做温度与分段处理，是当下长上下文模型很常用的方案。这些变体在 SGLang 里都通过同一个 <span class="mono">get_rope(...)</span> 工厂按配置造出来——你在模型配置里写明 <span class="mono">rope_scaling</span> 用哪种、放大多少倍，工厂就返回对应的 <span class="mono">RotaryEmbedding</span> 子类，<strong>模型代码完全不变</strong>。这又是"可插拔"的一个缩影：换一种长度扩展策略，只是换一个 RoPE 变体而已。</p>
<p>为什么"拉伸频率"这件事行得通，值得再多想一层。RoPE 的旋转角度其实是一组<strong>不同频率</strong>的正弦波叠出来的：高频通道转得快，负责区分<strong>相邻几个 token</strong> 的细微先后；低频通道转得慢，负责刻画<strong>跨越很远</strong>的长程关系。当序列拉长到训练没见过的 32k，最先"出界"的是那些低频通道——它们要表达的距离远远超出了训练时的角度范围。于是聪明的做法不是把所有频率一刀切地压缩（那会连高频的局部分辨力也牺牲掉，正是<strong>线性缩放</strong>的短板），而是像 <strong>NTK-aware</strong> 那样<strong>对低频多拉伸、对高频少动</strong>，把"撑长度"的代价尽量摊到对长程关系不那么敏感的通道上。<strong>YaRN</strong> 更进一步，对不同频段分段处理、并引入一个温度项校正注意力的分布，使得扩展后的模型在长文本上既不丢局部细节、又能稳住长程依赖。理解了"频率有高低、职责有分工"，你就明白这些方法不是玄学调参，而是顺着 RoPE 的数学结构，<strong>有针对性地</strong>动那些该动的频率。</p>

<table class="t">
  <tr><th>算子</th><th>做什么</th><th>为什么需要它</th></tr>
  <tr><td class="mono">RoPE</td><td>按位置<strong>旋转</strong> q/k，注入相对位置</td><td>注意力本身无序；让点积感知 token 间<strong>距离</strong>，还能扩展上下文（NTK/YaRN/线性）</td></tr>
  <tr><td class="mono">RMSNorm</td><td>÷ 均方根再乘缩放，<strong>稳定数值</strong></td><td>比 LayerNorm 更便宜（无减均值/无偏置），LLM 上效果一样好；常和残差加法<strong>融合</strong></td></tr>
  <tr><td class="mono">SiluAndMul</td><td>把门控 MLP 的 <strong>SiLU(gate) × up</strong> 合成一个 kernel</td><td>少一次中间张量的读写往返，省显存带宽、提速</td></tr>
  <tr><td class="mono">fused add-norm</td><td>残差加法 + 归一化<strong>合并</strong>成一步</td><td>避免额外的内存往返；qk-norm+rope 也可融合，小算子越融越快</td></tr>
</table>

<h2>RMSNorm：更轻的归一化</h2>
<p>每个 Transformer 块里都夹着<strong>归一化</strong>层，作用是把激活值的尺度<strong>压稳</strong>，让深层网络训练得动、推理得稳。经典做法是 <strong>LayerNorm</strong>：先<strong>减去均值</strong>、再<strong>除以标准差</strong>把数据标准化，最后用可学习的 <strong>scale 和 shift（偏置）</strong>调回合适的分布。它有效，但每一步都要算均值、算方差，还多带一个偏置参数。</p>
<p>Llama 一系用的是更精简的 <strong>RMSNorm（Root Mean Square Norm）</strong>。它<strong>不减均值</strong>、<strong>不加偏置</strong>，只做一件事：把向量除以它自己的<strong>均方根</strong>（元素平方和的平均再开根号），然后乘一个可学习的 scale。少了减均值这一步，也省了偏置参数——<strong>更便宜、更快</strong>，而大量实践证明在大语言模型上<strong>效果与 LayerNorm 不相上下</strong>。这就是为什么主流 LLM 几乎清一色转向了 RMSNorm。它还经常和前面的<strong>残差加法融合</strong>成一个 kernel（fused add-norm），把"加残差"和"归一化"两步并作一次内存读写完成，进一步省带宽。</p>
<p>有人会问：减均值这一步真的可有可无吗？直觉上 LayerNorm 的"减均值"是在做<strong>居中（centering）</strong>，把激活的偏移量去掉。但研究和实践都发现，对 Transformer 这种结构，真正起稳定作用的是<strong>缩放（rescaling）</strong>那一步——也就是把向量的整体尺度拉到统一量级；而居中带来的收益很有限，却要白白多算一遍均值。RMSNorm 正是抓住了这一点，<strong>只保留缩放、砍掉居中</strong>，用更少的算术拿到几乎一样的稳定性。再加上去掉偏置参数，整层的参数量和访存都更省。把它放进"算子是可插拔标准件"的主题里看也很自然：归一化这件事"做什么"（把尺度压稳）是固定的，"怎么做"（LayerNorm 还是 RMSNorm、要不要和残差融合、跑哪个平台的 kernel）则是可替换的实现细节——模型只写一句 <span class="mono">self.input_layernorm(x)</span>，底下换哪种归一化，对模型代码完全透明。</p>

<div class="cols">
  <div class="col"><h4>LayerNorm（经典）</h4><p><strong>减均值</strong> → <strong>÷ 标准差</strong> → 乘 scale + 加 <strong>shift（偏置）</strong>。把数据完整标准化，步骤多、要算均值方差、还多一个偏置参数。</p></div>
  <div class="col"><h4>RMSNorm（Llama 系）</h4><p><strong>不减均值、不加偏置</strong>：只 <strong>÷ 均方根</strong> 再乘 scale。更少计算、更少参数，<strong>更便宜</strong>，LLM 上效果一样好，常和残差加法<strong>融合</strong>。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="RMSNorm 先除以均方根再乘可学习权重：输入尺度参差，归一后整体拉平，再按逐通道权重缩放得到输出">
    <text x="40" y="34" style="font-weight:700;fill:var(--muted)">前：尺度参差</text>
    <line x1="40" y1="240" x2="224" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="46" y="100" width="26" height="140" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="82" y="185" width="26" height="55" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="118" y="75" width="26" height="165" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="154" y="155" width="26" height="85" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="190" y="125" width="26" height="115" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <line x1="40" y1="130" x2="224" y2="130" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="40" y="124" style="fill:var(--amber);font-size:12px">RMS</text>
    <text x="252" y="86" class="mono" style="font-size:12px">RMS=√(mean x²)</text>
    <rect x="252" y="104" width="120" height="32" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="312" y="125" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:13px">÷ RMS</text>
    <rect x="252" y="158" width="120" height="32" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="312" y="179" text-anchor="middle" class="mono" style="fill:var(--purple);font-size:13px">× 权重 w</text>
    <line x1="392" y1="170" x2="486" y2="170" style="stroke:var(--muted);stroke-width:2"/>
    <polygon points="500,170 486,163 486,177" style="fill:var(--muted)"/>
    <text x="520" y="34" style="font-weight:700;fill:var(--teal)">后：归一+缩放</text>
    <line x1="516" y1="240" x2="700" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="522" y="130" width="26" height="110" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="558" y="145" width="26" height="95" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="594" y="122" width="26" height="118" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="630" y="148" width="26" height="92" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="666" y="112" width="26" height="128" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
  </svg>
  <div class="figcap"><b>图 2 · RMSNorm：÷RMS 再乘权重</b> — 输入各通道尺度参差；先除以均方根把整体尺度<strong>拉平</strong>，再乘逐通道<strong>可学习权重</strong> w 得到输出（不减均值、不加偏置）。</div>
</div>

<p>举两个具体例子：<strong>RoPE</strong> 让注意力只认<strong>相对位置</strong>——<span class="mono">q@5 · k@3</span> 的结果只取决于 <span class="mono">5−3=2</span>，与它们落在 5/3 还是 105/103 无关，这正是模型能<strong>外推到更长序列</strong>的根基；<strong>RMSNorm</strong> 比 LayerNorm 便宜，是因为它<strong>跳过了减均值（居中）和偏置</strong>，只保留"除以均方根再缩放"这一步。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/layernorm.py ::RMSNorm</span><span class="ln">按均方根归一（不减均值），再乘可学习权重</span></div>
  <pre><span class="kw">class</span> RMSNorm(nn.Module):
    <span class="cm"># 按均方根归一（不减均值），再乘缩放</span>
    <span class="kw">def</span> forward_native(self, x, residual=<span class="kw">None</span>):
        <span class="kw">if</span> residual <span class="kw">is not None</span>:
            x = x + residual                       <span class="cm"># 融合：先加残差，再归一</span>
        x = x.to(torch.float32)
        variance = x.pow(<span class="st">2</span>).mean(dim=-<span class="st">1</span>, keepdim=<span class="kw">True</span>)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        <span class="kw">return</span> (x * self.weight).to(orig_dtype)    <span class="cm"># 逐通道可学习缩放</span></pre>
</div>

<h2>融合算子：把小步骤并成一个 kernel</h2>
<p>最后讲一个贯穿全课的主题——<strong>算子融合（fusion）</strong>。GPU 上有个朴素的事实：每启动一个 kernel、每把张量从显存读进来再写回去，都有<strong>固定开销</strong>。一连串小算子如果各跑各的，数据就要在显存和计算单元之间<strong>来回搬好几趟</strong>，而搬运往往比计算还慢。融合就是把几个挨着的小算子<strong>合并成一个 kernel</strong>，让数据进来一次、把活全干完再出去，省掉中间那些内存往返。</p>
<p>典型的例子：门控 MLP（第 34 课里 MoE 的专家、以及普通 MLP 都用门控结构）需要算 <span class="mono">SiLU(gate) × up</span>——<strong>SiluAndMul</strong> 就把激活函数和逐元素相乘<strong>融进一个 kernel</strong>，免去一个巨大中间张量的读写；前面说的 <strong>fused add-norm</strong> 把残差加法和 RMSNorm 并作一步；还有 <strong>qk-norm + RoPE</strong> 也能融合。这些层都是<strong>可复用、硬件优化过</strong>的标准件：模型只管调用，层内部会根据平台挑<strong>融合版还是平台专属版</strong>的 kernel——这正是继注意力后端（第 33 课）、量化（第 35 课）之后，"<strong>一切皆可插拔</strong>"主题在<strong>算子</strong>这一维上的体现。更底层的 kernel 怎么写、怎么融合，是第 38 课的内容；这里你只需记住：模型搭积木，积木自己懂硬件。</p>
<p>把这一课收束起来：注意力和 MLP 那些大矩阵乘是 Transformer 的"主肌肉"，但真正让它跑得<strong>又对又快</strong>的，是这些不起眼的小算子。RoPE 让注意力懂得<strong>位置与距离</strong>，并顺手把上下文撑长；RMSNorm 用最省的方式把数值<strong>稳住</strong>；融合算子则把零碎的小步骤<strong>攒成一趟</strong>，省下宝贵的访存带宽。而贯穿始终的，是同一套设计哲学——它们都是<strong>可复用、按平台自适应</strong>的 <span class="mono">nn.Module</span>。模型作者（第 26 课）站在"调用"这一侧，写下 <span class="mono">self.rotary_emb(...)</span>、<span class="mono">self.input_layernorm(...)</span> 这些稳定的接口；性能与硬件工程师站在"实现"那一侧，自由地把新的融合 kernel、新的平台后端接进去，而模型代码<strong>一个字都不用改</strong>。理解了这层分工，你再看 SGLang 支持一种新模型，往往就是"把这些标准件按新结构重新拼一遍"——这正是第 8 部分反复强调的主题：<strong>把 Transformer 的每一块都打磨成可替换的策略</strong>。</p>

<p>下面是 RoPE 真身的核心几行——<span class="mono">forward_native</span> 如何按位置取出 cos/sin，并把旋转施加到 q 和 k 上：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/rotary_embedding/base.py ::RotaryEmbedding</span><span class="ln">前向：按位置把旋转施加到 q/k</span></div>
  <pre><span class="kw">def</span> forward_native(self, positions, query, key, ...):
    <span class="cm"># 按每个 token 的 position 取出预算好的 cos / sin</span>
    cos_sin = self.cos_sin_cache.index_select(<span class="st">0</span>, positions)
    cos, sin = cos_sin.chunk(<span class="st">2</span>, dim=-<span class="st">1</span>)
    query_rot = query[..., : self.rotary_dim]   <span class="cm"># 只旋转前 rotary_dim 维</span>
    query_rot = self._apply_rotary_emb_wrapped(query_rot, cos, sin, self.is_neox_style)
    key_rot = key[..., : self.rotary_dim]
    key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
    <span class="kw">return</span> query, key                          <span class="cm"># q/k 已带上位置，交给注意力</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <strong>① RoPE 用旋转注入位置</strong>：注意力本身无序，RoPE 按位置把 q/k <strong>旋转</strong>一个角度，旋转后点积只依赖位置<strong>之差</strong>——注意力变成<strong>相对位置感知</strong>，且不引入可学习参数。
  <strong>② RoPE 撑长上下文</strong>：线性缩放 / NTK-aware / YaRN <strong>拉伸旋转频率</strong>，让 4k 训练的模型服务 32k+；都由 <span class="mono">get_rope(...)</span> 按配置造出变体，模型代码不变。
  <strong>③ RMSNorm vs LayerNorm</strong>：RMSNorm <strong>只 ÷ 均方根再乘 scale</strong>，<strong>不减均值、不加偏置</strong>——比 LayerNorm 便宜，LLM 上效果一样好，常和残差加法融合。
  <strong>④ 融合小算子</strong>：<span class="mono">SiluAndMul</span>、fused add-norm、qk-norm+rope 把相邻小算子并成一个 kernel，<strong>省内存往返</strong>。这些都是可复用、硬件优化的层（第 26 课搭、第 38 课讲 kernel/融合）。
</div>
""",
             "en": r"""
<p class="lead">
In Lesson 26 you'll see that writing a Transformer is really just <strong>snapping together ready-made layers</strong>: attention is a layer, the MLP is a layer, normalization is a layer, and RoPE — which supplies position — is a layer too. The previous lessons covered the heavyweights: attention (Lesson 33), MoE (Lesson 34), quantization (Lesson 35).
This lesson fills in the <strong>smaller but equally essential</strong> ops — RoPE, RMSNorm, and a family of <strong>fused ops</strong>. Each ships as a reusable <span class="mono">nn.Module</span>; the model just <strong>calls</strong> it, and the layer itself picks which platform's kernel to run and whether to fuse.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of <strong>RoPE</strong> as turning each word's <strong>compass needle</strong> by how far into the sentence it sits: the later the word, the more the needle is rotated. The "distance" two words feel is just the <strong>angle between their needles</strong> —
  nobody has an absolute "street address", yet from the angle alone you know <strong>how far apart</strong> they are. And think of <strong>RMSNorm</strong> as a <strong>lightweight volume-leveler</strong>: it doesn't move the music's baseline (no mean subtraction), it just pulls each segment's <strong>overall loudness</strong> to a uniform level so later layers hear clearly and training stays stable.
  The compass handles "where I am, how far apart we are"; the volume-leveler keeps "everyone at the same loudness" — two tiny parts that keep the whole Transformer running smoothly.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Remember one line: <strong>these small ops are the standard parts the model builds with</strong>. They belong to the same theme as the attention backend (Lesson 33) and quantization (Lesson 35) — <strong>pluggable, hardware-optimized layers</strong>.
  The model file (Lesson 26) only writes <strong>calls</strong> like <span class="mono">self.rotary_emb(positions, q, k)</span> or <span class="mono">self.input_layernorm(x)</span>; whether the real work is a fused CUDA kernel, a Triton impl, or an AMD/NPU-specific variant is chosen by the layer at runtime by <strong>platform</strong>.
  So the same model code runs on a different card unchanged, and a faster fused kernel can be wired in without touching a single model line.
</div>

<h2>RoPE: injecting position by "rotation"</h2>
<p>Start with a counter-intuitive fact: <strong>attention itself has no sense of order</strong>. Attention computes dot products of queries and keys, and a dot product is <strong>insensitive</strong> to who comes first — shuffle the words and pure attention sees almost the same relations. Yet language is <strong>deeply order-dependent</strong>: "dog bites man" and "man bites dog" are worlds apart.
So something must feed in <strong>position</strong>. <strong>RoPE (Rotary Position Embedding)</strong> answers elegantly: instead of <strong>adding</strong> a position vector, it <strong>rotates</strong> each q/k vector by an angle — the later the position, the more it turns.</p>
<p>The magic is in the dot-product math. Rotate the query by the angle of its position and the key by the angle of its position, then take the dot product, and the result depends only on the <strong>difference</strong> of the two angles — the <strong>relative position</strong> of the two tokens. So attention suddenly becomes <strong>relative-position aware</strong>: the model no longer cares about a token's absolute index, but about <strong>how far apart</strong> two tokens are.
That matches the nature of language — a pronoun referring back to a noun a few words earlier does so by <strong>distance</strong>, not absolute coordinates. RoPE adds no learnable parameters; pure fixed-frequency rotation bakes relative position into attention, saving parameters and generalizing well, which is why it's standard in Llama, Qwen, and friends.</p>
<p>In code, RoPE is a <span class="mono">RotaryEmbedding</span> layer; <span class="mono">get_rope(...)</span> builds the right variant from config, and the forward applies the rotation to q/k <strong>right before attention</strong> (attention itself is Lesson 33). It precomputes a <span class="mono">cos/sin</span> cache per position, picks out the entries for each token's <span class="mono">positions</span> at forward time, rotates the first <span class="mono">rotary_dim</span> dimensions of the vector, and passes the rest through. No matmul — just elementwise multiply-adds — so it's cheap, and often <strong>fused</strong> into a neighboring op.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Have q / k</h4><p>Before attention we have projected query/key vectors — but right now they carry <strong>no position info</strong>.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Apply RoPE by position</h4><p>Use each token's <span class="mono">positions</span> to pick <span class="mono">cos/sin</span>, and <strong>rotate</strong> q/k each by an angle ∝ position.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Dot product turns relative</h4><p>Rotated q·k depends only on the <strong>difference</strong> of positions — attention gains <strong>relative position</strong> for free.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Into attention</h4><p>Position-bearing q/k go to the attention backend (Lesson 33) for softmax weighting — order is now inside.</p></div></div>
</div>

<p>Picture the rotation even more concretely: the same vector is unrotated at position 0, turned a bit at position 1, and turned more at position 2 — every step later adds a fixed angle.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>One vector, rotated by different positions</b>: the later the position, the larger the rotation (angle ∝ position)</div>
  <div class="cells"><span class="lab">vector v</span><span class="cell">pos 0 → 0°</span><span class="sep">→</span><span class="cell">pos 1 → 30°</span><span class="sep">→</span><span class="cell hl">pos 2 → 60°</span><span class="sep">⇒</span><span class="cell q">q·k sees only the angle gap = relative position</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="RoPE rotates q/k by position: the same vector is turned by different angles at different positions, and the dot product depends only on the angle difference, i.e. relative position">
    <text x="24" y="34" style="font-weight:700;fill:var(--muted)">rotation plane · unit circle</text>
    <line x1="84" y1="170" x2="318" y2="170" style="stroke:var(--faint);stroke-width:1.5"/>
    <line x1="195" y1="58" x2="195" y2="282" style="stroke:var(--faint);stroke-width:1.5"/>
    <circle cx="195" cy="170" r="108" style="fill:none;stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="195" y1="170" x2="303" y2="170" style="stroke:var(--blue);stroke-width:2.5"/>
    <circle cx="303" cy="170" r="4" style="fill:var(--blue)"/>
    <text x="282" y="190" class="mono" style="fill:var(--blue);font-weight:700;font-size:12px">q@p0</text>
    <line x1="195" y1="170" x2="288" y2="116" style="stroke:var(--teal);stroke-width:2.5"/>
    <circle cx="288" cy="116" r="4" style="fill:var(--teal)"/>
    <text x="294" y="110" class="mono" style="fill:var(--teal);font-weight:700;font-size:12px">q@p1</text>
    <line x1="195" y1="170" x2="249" y2="77" style="stroke:var(--amber);stroke-width:2.5"/>
    <circle cx="249" cy="77" r="4" style="fill:var(--amber)"/>
    <text x="220" y="64" class="mono" style="fill:var(--amber);font-weight:700;font-size:12px">q@p2</text>
    <path d="M237,170 A42,42 0 0 0 216,134" style="fill:none;stroke:var(--amber);stroke-width:1.5"/>
    <text x="240" y="150" style="fill:var(--amber);font-size:12px">θ∝pos</text>
    <rect x="430" y="92" width="320" height="120" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="452" y="122" style="font-weight:700;fill:var(--accent-ink)">dot sees angle gap</text>
    <text x="452" y="154" class="mono" style="font-size:13px">q·k ∝ cos(θq − θk)</text>
    <text x="452" y="182" class="mono" style="font-size:13px">θq − θk ∝ p − p′</text>
    <text x="452" y="206" style="fill:var(--teal);font-weight:700">⇒ relative pos</text>
  </svg>
  <div class="figcap"><b>Fig 1 · RoPE rotates q/k by position</b> — the same vector is turned 0°/30°/60° at p0/p1/p2 (angle ∝ position); the q·k dot product depends only on the angle difference, so attention feels only <strong>relative position</strong>.</div>
</div>

<h2>How RoPE enables context-length extension</h2>
<p>RoPE also solves a big engineering headache: <strong>context-length extension</strong>. A model trained on, say, 4k-token sequences has its rotation "frequency table" tuned for 4k. Serve 32k of long text directly and the rotation angles at late positions run into a range the model has <strong>never seen</strong>, and quality collapses.
The fix isn't retraining but <strong>cleverly stretching the rotation frequencies</strong>, so a 4k-trained model works comfortably at 32k and beyond.</p>
<p>The common tricks are all RoPE variants: <strong>linear scaling</strong> is the simplest — compress all positions proportionally, like lengthening the ruler, so 32k positions fall back into familiar angles; simple, but it sacrifices some high-frequency detail. <strong>NTK-aware</strong> is smarter, adjusting different frequencies <strong>unevenly</strong> — small changes to high frequencies (local detail) and larger stretching of low frequencies (long-range relations) — extending length while preserving short-distance resolution.
<strong>YaRN</strong> adds temperature and segmenting on top of NTK and is a very common choice for today's long-context models. In SGLang all of these are produced by the same <span class="mono">get_rope(...)</span> factory from config — you declare in the model config which <span class="mono">rope_scaling</span> and what factor, and the factory returns the matching <span class="mono">RotaryEmbedding</span> subclass, with <strong>model code unchanged</strong>. Another miniature of "pluggability": swapping a length-extension strategy is just swapping a RoPE variant.</p>

<table class="t">
  <tr><th>Op</th><th>What it does</th><th>Why you need it</th></tr>
  <tr><td class="mono">RoPE</td><td><strong>Rotates</strong> q/k by position, injecting relative position</td><td>Attention is orderless; makes the dot product feel token <strong>distance</strong>, and extends context (NTK/YaRN/linear)</td></tr>
  <tr><td class="mono">RMSNorm</td><td>÷ root-mean-square then scale, <strong>stabilizing numbers</strong></td><td>Cheaper than LayerNorm (no mean subtraction / no bias), as good for LLMs; often <strong>fused</strong> with the residual add</td></tr>
  <tr><td class="mono">SiluAndMul</td><td>Fuses a gated MLP's <strong>SiLU(gate) × up</strong> into one kernel</td><td>One fewer round-trip for the intermediate tensor — saves bandwidth, runs faster</td></tr>
  <tr><td class="mono">fused add-norm</td><td><strong>Merges</strong> the residual add + normalization into one step</td><td>Avoids an extra memory round-trip; qk-norm+rope can fuse too — fusing small ops keeps getting faster</td></tr>
</table>

<h2>RMSNorm: a lighter normalization</h2>
<p>Every Transformer block sandwiches in a <strong>normalization</strong> layer whose job is to keep the <strong>scale</strong> of activations stable, so deep nets train and infer reliably. The classic recipe is <strong>LayerNorm</strong>: <strong>subtract the mean</strong>, <strong>divide by the standard deviation</strong> to standardize, then apply a learnable <strong>scale and shift (bias)</strong>. Effective, but every step computes mean and variance, plus it carries a bias parameter.</p>
<p>The Llama family uses the leaner <strong>RMSNorm (Root Mean Square Norm)</strong>. It does <strong>no mean subtraction</strong> and has <strong>no bias</strong>, doing just one thing: divide the vector by its own <strong>root-mean-square</strong> (the square root of the mean of squared elements), then multiply by a learnable scale. Dropping mean subtraction and the bias makes it <strong>cheaper and faster</strong>, and abundant practice shows it works <strong>just as well as LayerNorm</strong> for LLMs — which is why mainstream LLMs went almost uniformly to RMSNorm. It's also often <strong>fused with the preceding residual add</strong> (fused add-norm), doing the "add residual" and "normalize" steps in one memory round-trip to save more bandwidth.</p>

<div class="cols">
  <div class="col"><h4>LayerNorm (classic)</h4><p><strong>Subtract mean</strong> → <strong>÷ std</strong> → multiply by scale + add <strong>shift (bias)</strong>. Fully standardizes the data: more steps, computes mean/variance, plus one extra bias parameter.</p></div>
  <div class="col"><h4>RMSNorm (Llama family)</h4><p><strong>No mean subtraction, no bias</strong>: just <strong>÷ root-mean-square</strong> then multiply by scale. Less compute, fewer params, <strong>cheaper</strong>, as good for LLMs, often <strong>fused</strong> with the residual add.</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="RMSNorm divides by root-mean-square then multiplies by a learnable weight: inputs have uneven scale, normalization flattens the overall scale, then a per-channel weight rescales to the output">
    <text x="40" y="34" style="font-weight:700;fill:var(--muted)">Before: uneven scale</text>
    <line x1="40" y1="240" x2="224" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="46" y="100" width="26" height="140" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="82" y="185" width="26" height="55" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="118" y="75" width="26" height="165" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="154" y="155" width="26" height="85" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="190" y="125" width="26" height="115" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <line x1="40" y1="130" x2="224" y2="130" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="40" y="124" style="fill:var(--amber);font-size:12px">RMS</text>
    <text x="252" y="86" class="mono" style="font-size:12px">RMS=√(mean x²)</text>
    <rect x="252" y="104" width="120" height="32" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="312" y="125" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:13px">÷ RMS</text>
    <rect x="252" y="158" width="120" height="32" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="312" y="179" text-anchor="middle" class="mono" style="fill:var(--purple);font-size:13px">× weight w</text>
    <line x1="392" y1="170" x2="486" y2="170" style="stroke:var(--muted);stroke-width:2"/>
    <polygon points="500,170 486,163 486,177" style="fill:var(--muted)"/>
    <text x="520" y="34" style="font-weight:700;fill:var(--teal)">After: norm+scale</text>
    <line x1="516" y1="240" x2="700" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="522" y="130" width="26" height="110" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="558" y="145" width="26" height="95" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="594" y="122" width="26" height="118" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="630" y="148" width="26" height="92" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="666" y="112" width="26" height="128" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
  </svg>
  <div class="figcap"><b>Fig 2 · RMSNorm: ÷RMS then scale</b> — input channels have uneven scale; dividing by the root-mean-square <strong>flattens</strong> the overall scale, then a per-channel <strong>learnable weight</strong> w rescales to the output (no mean subtraction, no bias).</div>
</div>

<p>Two concrete examples: <strong>RoPE</strong> makes attention see only <strong>relative position</strong> — <span class="mono">q@5 · k@3</span> depends only on <span class="mono">5−3=2</span>, the same whether they sit at 5/3 or 105/103, which is exactly why the model can <strong>extrapolate to longer sequences</strong>; <strong>RMSNorm</strong> is cheaper than LayerNorm because it <strong>skips mean-centering and the bias</strong>, keeping only "divide by root-mean-square, then scale".</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/layernorm.py ::RMSNorm</span><span class="ln">normalize by root-mean-square (no mean subtraction), then scale</span></div>
  <pre><span class="kw">class</span> RMSNorm(nn.Module):
    <span class="cm"># normalize by Root-Mean-Square (NO mean subtraction), then scale.</span>
    <span class="kw">def</span> forward_native(self, x, residual=<span class="kw">None</span>):
        <span class="kw">if</span> residual <span class="kw">is not None</span>:
            x = x + residual                       <span class="cm"># fused add, then norm</span>
        x = x.to(torch.float32)
        variance = x.pow(<span class="st">2</span>).mean(dim=-<span class="st">1</span>, keepdim=<span class="kw">True</span>)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        <span class="kw">return</span> (x * self.weight).to(orig_dtype)    <span class="cm"># learned per-channel scale</span></pre>
</div>

<h2>Fused ops: merging small steps into one kernel</h2>
<p>Finally, a theme running through the whole lesson — <strong>operator fusion</strong>. On a GPU there's a plain fact: every kernel launch, every read of a tensor from HBM and write back, carries a <strong>fixed overhead</strong>. A chain of small ops each running on its own makes data <strong>shuttle back and forth</strong> between HBM and compute several times, and the shuttling is often slower than the math. Fusion <strong>merges several neighboring small ops into one kernel</strong>, so data comes in once, gets all the work done, and leaves — skipping the intermediate round-trips.</p>
<p>Typical examples: a gated MLP (the experts in Lesson 34's MoE, and ordinary MLPs, both use gating) needs <span class="mono">SiLU(gate) × up</span> — <strong>SiluAndMul</strong> fuses the activation and the elementwise multiply <strong>into one kernel</strong>, skipping the read/write of a huge intermediate tensor; the <strong>fused add-norm</strong> above merges the residual add and RMSNorm; and <strong>qk-norm + RoPE</strong> can fuse too. These layers are all <strong>reusable, hardware-optimized</strong> standard parts: the model just calls them, and the layer internally picks a <strong>fused or platform-specific</strong> kernel — this is "<strong>everything is pluggable</strong>" expressed along the <strong>ops</strong> dimension, after the attention backend (Lesson 33) and quantization (Lesson 35). How the lower-level kernels are written and fused is Lesson 38; here, just remember: the model builds with blocks, and the blocks know the hardware.</p>

<p>Below are the core lines of RoPE itself — how <span class="mono">forward_native</span> picks cos/sin by position and applies the rotation to q and k:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/rotary_embedding/base.py ::RotaryEmbedding</span><span class="ln">forward: applying the rotation to q/k by position</span></div>
  <pre><span class="kw">def</span> forward_native(self, positions, query, key, ...):
    <span class="cm"># pick the precomputed cos / sin for each token's position</span>
    cos_sin = self.cos_sin_cache.index_select(<span class="st">0</span>, positions)
    cos, sin = cos_sin.chunk(<span class="st">2</span>, dim=-<span class="st">1</span>)
    query_rot = query[..., : self.rotary_dim]   <span class="cm"># rotate only the first rotary_dim dims</span>
    query_rot = self._apply_rotary_emb_wrapped(query_rot, cos, sin, self.is_neox_style)
    key_rot = key[..., : self.rotary_dim]
    key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
    <span class="kw">return</span> query, key                          <span class="cm"># q/k now carry position, on to attention</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <strong>① RoPE injects position via rotation</strong>: attention is orderless, so RoPE <strong>rotates</strong> q/k by an angle ∝ position; after rotation the dot product depends only on the <strong>difference</strong> of positions — attention becomes <strong>relative-position aware</strong>, with no learnable parameters.
  <strong>② RoPE enables context extension</strong>: linear scaling / NTK-aware / YaRN <strong>stretch the rotation frequencies</strong> so a 4k-trained model serves 32k+; all built by <span class="mono">get_rope(...)</span> from config, model code unchanged.
  <strong>③ RMSNorm vs LayerNorm</strong>: RMSNorm <strong>only ÷ root-mean-square then scale</strong>, with <strong>no mean subtraction, no bias</strong> — cheaper than LayerNorm, as good for LLMs, often fused with the residual add.
  <strong>④ Fuse small ops</strong>: <span class="mono">SiluAndMul</span>, fused add-norm, qk-norm+rope merge neighboring small ops into one kernel to <strong>save memory round-trips</strong>. All are reusable, hardware-optimized layers (built in Lesson 26; kernels/fusion in Lesson 38).
</div>
"""}

LESSON_37 = {
    "zh": r"""
<p class="lead">
模型一路算到最后，手里是每个位置的一串<strong>隐藏向量</strong>。可用户要的是"<strong>下一个词</strong>"，不是一串向量。
把隐藏向量翻译成"<strong>每个词的得分</strong>"的，是模型的<strong>输出头 lm_head</strong>；而因为词表动辄十几万，这个头大到一张卡装不下——
于是有了<strong>词表并行（vocab parallel）</strong>。这一课讲清楚 logits 怎么来、词表头为什么要切开、以及 <span class="mono">LogitsProcessor</span> 怎么把它收束好交给采样器。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把输出头想成一本<strong>超大词典</strong>，要给查询词打"和每个词条的匹配分"。词典太厚，于是<strong>拆给 N 个管理员</strong>，
  每人只负责其中<strong>一段词条</strong>、只打自己那段的分；最后大家把各自的分数表<strong>汇总（all-gather）</strong>成完整排名。
  还有个省力诀窍：你只关心<strong>刚写下的最后一个字</strong>该接什么，所以只问"<strong>最后一位</strong>"的分，不必把整句重新打分。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  末端归一（第 36 课）后，每个位置一个隐藏向量。<strong>lm_head</strong>（一个 [hidden × vocab] 的大矩阵）把它投影成
  <strong>每个词表 token 一个分数 = logits</strong>，再由采样器（第 28 课）选出 token。词表很大（3 万~25 万），
  lm_head 与词嵌入矩阵都很重，所以在张量并行（第 46 课）下按<strong>词表维度切分到各 rank</strong>，每卡只算自己那一片，
  再 <strong>all-gather</strong> 拼成完整 logits。<span class="mono">LogitsProcessor</span> 负责把这套流程（含<strong>只取末位</strong>的优化）串起来。
</div>

<h2>从 hidden 到 logits：输出头 lm_head</h2>
<p>模型的主体（第 26 课）一层层算下来，给每个输入位置产出一个 <strong>hidden 向量</strong>（维度就是模型的隐藏维，比如 4096）。
但这只是"中间表示"，不是答案。要预测下一个词，得知道<strong>词表里每个候选 token 有多"合适"</strong>——这就是 <span class="mono">lm_head</span> 干的事：
它是一个形状为 <span class="mono">[hidden_dim × vocab_size]</span> 的大矩阵，把 hidden 向量乘上去，得到一个<strong>长度等于词表大小</strong>的分数向量，
这就是 <strong>logits</strong>。logits 越大的 token，模型越认为它该是下一个词。logits 接着交给采样器（第 28 课）做温度、top-k/p、采样，最终落成一个具体 token。
很多模型还会让 lm_head 与输入词嵌入<strong>共享权重</strong>（tied embedding），因为"把词变向量"和"把向量变回词分数"本就是一对互逆的事。</p>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="最后一个隐藏向量乘以 lm_head 权重矩阵，得到长度等于词表的 logits，每个词一个分数，再 argmax 或采样选出下一个 token">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">hidden × lm_head → logits</text>
    <rect x="40" y="70" width="54" height="120" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="67" y="60" text-anchor="middle" style="font-size:12px;fill:var(--muted)">hidden</text>
    <text x="67" y="135" text-anchor="middle" class="mono" style="font-size:12px">4096</text>
    <text x="112" y="136" text-anchor="middle" style="font-size:20px;fill:var(--muted)">×</text>
    <rect x="134" y="58" width="190" height="144" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="229" y="120" text-anchor="middle" class="mono" style="font-size:13px;fill:var(--accent-ink)">lm_head</text>
    <text x="229" y="142" text-anchor="middle" style="font-size:12px;fill:var(--accent-ink)">[hidden × vocab]</text>
    <text x="229" y="164" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--accent-ink)">4096 × 32000</text>
    <text x="346" y="136" text-anchor="middle" style="font-size:20px;fill:var(--muted)">→</text>
    <rect x="368" y="70" width="54" height="120" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="395" y="60" text-anchor="middle" style="font-size:12px;fill:var(--muted)">logits</text>
    <text x="395" y="135" text-anchor="middle" class="mono" style="font-size:12px">32000</text>
    <text x="444" y="136" text-anchor="middle" style="font-size:20px;fill:var(--muted)">→</text>
    <rect x="470" y="100" width="286" height="60" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="613" y="126" text-anchor="middle" style="font-size:13px">argmax / 采样</text>
    <text x="613" y="146" text-anchor="middle" style="font-size:12px;fill:var(--muted)">选出下一个 token</text>
  </svg>
  <div class="figcap"><b>图 1 · hidden × lm_head → logits</b> — 最后一位的隐藏向量（长 4096）乘以 lm_head 权重 [hidden × vocab]，得到长度等于词表（32000）的 logits，每个词一个分数；再由 argmax 或采样选出下一个 token。</div>
</div>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>主体前向</h4><p>嵌入 → N 层 → 末端归一（第 36 课），得到每位置的 hidden。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>取末位</h4><p>解码时<strong>只留每条请求的最后一个位置</strong>——只有它的 logits 用来预测下一个词。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>lm_head 投影</h4><p>hidden × lm_head → 词表维度的 <strong>logits</strong>（按 rank 各算一片）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>跨卡汇总 + 采样</h4><p><strong>all-gather</strong> 拼出完整 logits → 采样器（第 28 课）出 token。</p></div></div>
</div>

<div class="flow">
  <div class="node"><div class="nt">hidden·末位</div><div class="nd">每条请求最后一位<br>的隐藏向量</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">lm_head·按词表切片</div><div class="nd">本 rank 只算<br>自己那段词表的分</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">all-gather</div><div class="nd">跨卡拼接各段<br>残缺分数向量</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">完整 logits</div><div class="nd">长度 = 词表大小<br>每个 token 一个分</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Sampler·第 28 课</div><div class="nd">温度 / top-k/p<br>选出下一个 token</div></div>
</div>

<h2>只取末位：一行 logits 胜过两千行</h2>
<p>"只取最后一位"听上去像个小优化，实则省下的算力相当可观。设想一条 2000 token 的 prompt：预填充阶段，模型确实要为<strong>全部 2000 个位置</strong>都算出 hidden——因为每一层的注意力都需要前面所有位置的表示。但当我们要预测"<strong>第 2001 个 token</strong>"时，真正有用的只有<strong>最后一个位置</strong>的 hidden：它已经"看过"了整段 prompt，它的 logits 才是下一个词的分布。前面 1999 个位置的 logits，在普通生成里<strong>一个都用不上</strong>。</p>
<p>于是 <span class="mono">LogitsProcessor</span> 在调用 lm_head 之前，先把 hidden 从 <span class="mono">[2000, hidden]</span> 裁成 <span class="mono">[1, hidden]</span>，<strong>只让最后一行过那张 [hidden × vocab] 的大矩阵</strong>。lm_head 是整个前向里数一数二贵的矩阵乘（词表几万到几十万列），把它的输入从 2000 行压到 1 行，等于把这步的算力和访存直接<strong>砍掉了两千倍</strong>。批处理时也一样：每条请求只贡献它自己的末位，一个 batch 有多少条请求，就只算多少行 logits，而不是把整批 prompt 的全部位置都算一遍。</p>
<p>唯一的例外是<strong>用户要 logprob</strong>：要返回每个 prompt token 的对数概率（评测、打分、推测解码的草稿校验都会用到），就得保留那些位置的 logits。<span class="mono">logits_metadata</span> 正是用来记录"这一批里哪些位置需要完整 logits、哪些只要末位"，让 LogitsProcessor 按需裁剪——默认省到只剩末位，需要时才多留几行。这也解释了为什么同一份代码，普通生成飞快、而开了 <span class="mono">logprobs</span> 的请求会稍慢一点：后者让那张大矩阵乘多吃了好些行输入。</p>

<p>解码阶段这个优化更是天然成立：每步只往前走一个 token，每条请求的 hidden 本来就只有<strong>一行</strong>（最新那个位置），lm_head 自然只算一行。所以"末位切片"在预填充时是<strong>主动裁剪</strong>、在解码时是<strong>本就如此</strong>——两种模式（第 4 课）在这里被统一成同一句话：<strong>每条请求只取一行 hidden 去算 logits</strong>。也正因为输出头只处理这薄薄的几行，它在整个前向里的耗时占比通常很小；真正的大头还是中间那几十层注意力与 MLP。把这件事想清楚，你就能解释一个常见困惑：模型参数里词表头占了一大块<strong>显存</strong>，但它在每步解码里花的<strong>时间</strong>却很少——因为算的行数实在太少了。</p>

<h2>词表太大：为什么要按 TP 切分</h2>
<p>这里藏着一个容易被忽视的规模问题：<strong>词表非常大</strong>。常见模型的词表在 3 万到 25 万 token 之间，而 lm_head 是
<span class="mono">[hidden × vocab]</span>、词嵌入是 <span class="mono">[vocab × hidden]</span>——当 vocab=15 万、hidden=8192 时，<strong>单这一张表就有十几亿参数</strong>，
占整个模型相当可观的一块。在张量并行（第 46 课，把每层矩阵切到多卡）下，<strong>没法把整张词表头塞进一张卡</strong>，也不该让一张卡独扛这么大的乘法和显存。</p>

<p>解决办法叫<strong>词表并行（vocab parallelism）</strong>：<span class="mono">VocabParallelEmbedding</span> 与 <span class="mono">ParallelLMHead</span>
把<strong>词表维度</strong>切成若干段，分给各个 TP rank——比如 4 卡、15 万词表，每卡负责约 3.75 万个 token。前向时，<strong>每卡只用自己那段 lm_head 算出"本段词的 logits"</strong>，
得到的是一个<strong>残缺的</strong>分数向量（只有自己负责那段是真值）。要得到完整的 logits，就需要把各卡的片段拼起来——这一步是跨卡的 <strong>all-gather</strong>（或在 argmax/采样里就地汇总）。
这样，谁也不用独存整张词表头，显存与算力都被均摊到了多卡上。</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="词表按 TP rank 切分：rank0 拥有 token 0 到 16000，rank1 拥有 16000 到 32000，各算自己那段的 logits，再 all-gather 拼成完整的 32000 长 logits">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">词表按 rank 切 → all-gather</text>
    <rect x="24" y="120" width="54" height="80" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="51" y="110" text-anchor="middle" style="font-size:12px;fill:var(--muted)">hidden</text>
    <text x="51" y="165" text-anchor="middle" class="mono" style="font-size:11px">4096</text>
    <line x1="78" y1="160" x2="132" y2="95" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="78" y1="160" x2="132" y2="225" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="134" y="60" width="220" height="70" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="244" y="86" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--teal)">rank0 · lm_head</text>
    <text x="244" y="108" text-anchor="middle" style="font-size:12px">vocab 0–16000</text>
    <rect x="134" y="190" width="220" height="70" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="244" y="216" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">rank1 · lm_head</text>
    <text x="244" y="238" text-anchor="middle" style="font-size:12px">vocab 16000–32000</text>
    <line x1="354" y1="95" x2="408" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="354" y1="225" x2="408" y2="160" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="410" y="115" width="140" height="70" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="480" y="148" text-anchor="middle" class="mono" style="font-size:13px;fill:var(--accent-ink)">all-gather</text>
    <text x="480" y="168" text-anchor="middle" style="font-size:11px;fill:var(--accent-ink)">拼接各段</text>
    <text x="566" y="155" text-anchor="middle" style="font-size:20px;fill:var(--muted)">→</text>
    <rect x="588" y="115" width="168" height="70" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="672" y="148" text-anchor="middle" style="font-size:13px;fill:var(--purple)">完整 logits</text>
    <text x="672" y="168" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--purple)">长度 32000</text>
  </svg>
  <div class="figcap"><b>图 2 · 词表按 TP rank 切分</b> — lm_head 的词表维切给各 rank：rank0 管 token 0–16000、rank1 管 16000–32000，各自只算本段 logits；再用 all-gather 把两段首尾拼成完整的 32000 长 logits。</div>
</div>

<p>把这套机制摊开看：每个 rank 手里的 lm_head 只有 <span class="mono">[hidden × vocab/N]</span> 那一窄条，算出来的自然是一个<strong>只有自己那段是真值、其余全空</strong>的部分分数向量。<strong>all-gather</strong> 做的事，就是让 N 张卡各自把"本段那截"发出去、也收下别人那截，最后<strong>按 rank 顺序首尾拼接</strong>成一条完整的、长度等于整个词表的 logits。值得一提的是，如果只是要<strong>贪心解码</strong>（argmax 取分数最大那个 token），其实可以更省：每张卡先在<strong>本段内求局部最大</strong>，只把"局部最大值 + 它对应的全局 token id"这一点点信息做 all-reduce，就能选出全局最大，<strong>根本不必把十几万维的完整向量物化出来</strong>。SGLang 会在不需要完整 logits（不算 logprob、不做复杂采样）时尽量走这种更省带宽的就地汇总路径。</p>

<p>词表并行并不是一个孤立技巧，它和第 25/46 课里 q/k/v、MLP 的切法是<strong>同一套思路</strong>：把一个大矩阵沿某个维度切成 N 片、每卡算一片、再用一次集合通信把结果对齐。注意力把<strong>注意力头</strong>切开、MLP 把<strong>中间维</strong>切开，而这里把<strong>词表维</strong>切开；区别只在"切哪一维、最后用 all-reduce 还是 all-gather 收口"。理解了词表并行，也就把 TP 在模型首尾两端（输入嵌入、输出头）和中间各层的切分图景<strong>补全</strong>了——整张网络从头到尾都是按同一条规则切开、再拼回来的。</p>

<p>实现里还有两个常被问到的细节。其一是<strong>词表对齐</strong>：词表大小未必能被 rank 数整除（比如 15 万词表切 8 卡除不尽），所以 <span class="mono">VocabParallelEmbedding</span> 会先把词表<strong>向上补齐</strong>到能整除的大小，多出来的是"假 token"，它们的 logit 会被屏蔽掉、永远不会被选中，只为让每张卡分到等长的一段、对齐通信。其二是<strong>嵌入与输出头共享</strong>：当模型用 tied embedding 时，输入端的 <span class="mono">VocabParallelEmbedding</span> 和输出端的 <span class="mono">ParallelLMHead</span> 其实是<strong>同一份按词表切好的权重</strong>，输入时用它把 token id 查成向量、输出时用它把向量打成分数，省下一整张大表的显存。无论共享与否，切分的维度和通信的方式都一致。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>15 万词表切到 4 个 rank</b>：每个 rank 只算自己那段的 logits，再 all-gather 拼成完整向量</div>
  <div class="cells"><span class="lab">rank0</span><span class="cell hl">token 0–37499</span><span class="sep">·</span><span class="lab">rank1</span><span class="cell hl">37500–74999</span><span class="sep">·</span><span class="lab">rank2</span><span class="cell hl">75000–112499</span><span class="sep">·</span><span class="lab">rank3</span><span class="cell hl">112500–149999</span></div>
</div>

<div class="cols">
  <div class="col"><h4>整张词表头放一张卡</h4><p>lm_head 与嵌入十几亿参数全压在一卡上：<strong>显存吃紧、这步算力也成瓶颈</strong>，和 TP 切分其它层的思路也不一致。</p></div>
  <div class="col"><h4>词表并行</h4><p>每 rank 只持有、只算<strong>一段词表</strong>；末了 <strong>all-gather</strong> 拼完整 logits。<strong>显存与算力均摊</strong>，与 TP 一脉相承。</p></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/vocab_parallel_embedding.py ::ParallelLMHead</span><span class="ln">输出投影：把 hidden 映射成每个词的 logit，词表按 TP 切</span></div>
  <pre><span class="kw">class</span> ParallelLMHead(VocabParallelEmbedding):
    <span class="cm"># 输出投影：hidden_state -&gt; 每个词表 token 一个 logit。</span>
    <span class="cm"># 词表维按 TP rank 切分（每个 rank 只持有一段行），</span>
    <span class="cm"># 词表大小会向上补齐到能被 TP 数整除。</span>
    <span class="kw">def</span> __init__(self, num_embeddings, embedding_dim, *, bias=False, ...):
        <span class="cm"># num_embeddings = 词表大小，embedding_dim = 隐藏维</span>
        ...
    <span class="cm"># 每个 rank 只产出自己那段词表的 logits；all-gather</span>
    <span class="cm"># （在 LogitsProcessor 里）把各段拼成完整词表向量。</span></pre>
</div>

<p>举个具体的数：Llama 词表 <span class="mono">32000</span>、hidden <span class="mono">4096</span>，于是 lm_head 是一张 <span class="mono">4096 × 32000</span> 的矩阵。开 <span class="mono">--tp-size 2</span> 时，每个 rank 只持有 <strong>16000</strong> 行词表、只算这 16000 个 token 的 logits；随后 all-gather 把两段拼成长度 <span class="mono">32000</span> 的完整 logits 向量。</p>

<h2>LogitsProcessor 的活：末位切片 + 跨卡汇总</h2>
<p>把上面这套流程编排起来的，是 <span class="mono">LogitsProcessor</span>。它在前向的最后接过 hidden states，做三件事：
①<strong>末位切片</strong>——解码时，每条请求其实只需要<strong>最后一个位置</strong>的 logits 来预测下一个词；而预填充时虽然算了整段 prompt 的 hidden，
但同样<strong>只有最后一位的 logits 有用</strong>（除非要算 logprob）。所以先把 hidden 裁到"每条请求的末位"，能大幅减少 lm_head 这步的计算量。
②<strong>词表并行投影</strong>——用 <span class="mono">ParallelLMHead</span> 算出本 rank 负责那段词表的 logits。③<strong>跨卡 all-gather</strong>——
把各 rank 的片段汇总成完整 logits（由 <span class="mono">do_tensor_parallel_all_gather</span> 控制，单卡时跳过）。最后产出 <span class="mono">next_token_logits</span> 交给采样器（第 28 课）。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/logits_processor.py ::LogitsProcessor</span><span class="ln">输出头的编排</span></div>
  <pre><span class="kw">class</span> LogitsProcessor(nn.Module):
    <span class="kw">def</span> forward(self, input_ids, hidden_states,
                lm_head: VocabParallelEmbedding, logits_metadata):
        <span class="cm"># 1) 解码/预填充都只取每条请求的“最后一位” hidden</span>
        pruned = self._get_pruned_states(hidden_states, logits_metadata)
        <span class="cm"># 2) 词表并行投影：本 rank 只算自己那段词表的 logits</span>
        logits = torch.matmul(pruned, lm_head.weight.T)  <span class="cm"># [tokens, vocab/这段]</span>
        <span class="kw">if</span> self.do_tensor_parallel_all_gather:       <span class="cm"># 3) 跨卡拼成完整 logits</span>
            logits = tensor_model_parallel_all_gather(logits)
        <span class="kw">return</span> LogitsProcessorOutput(next_token_logits=logits)</pre>
</div>

<h2>采样之前的最后一公里：logprob、结构化掩码与 logit bias</h2>
<p>logits 拼齐之后、真正采样之前，还有几道"挂"在输出头附近的后处理，全都作用在这条词表维向量上。其一是 <strong>logprob</strong>：把 logits 过一次 log-softmax，就得到每个 token 的对数概率；用户要 top-k logprob、要给 prompt 里每个 token 打分、或推测解码要校验草稿 token，都从这里取数。其二是<strong>结构化输出的词表掩码</strong>（第 48 课）：当请求要求输出必须是合法 JSON 或匹配某个语法时，约束引擎会算出"当前这一步<strong>哪些 token 允许、哪些禁止</strong>"，把禁止的那些 token 的 logit 直接压到负无穷，于是采样器无论如何都选不到它们。</p>
<p>其三是 <strong>logit bias</strong>：用户可以手动给某些 token 加正负偏置，抬高或压低它们被选中的概率——同样是在这条 logits 向量的对应位置上加一个数。这几样东西的共同点是：它们都<strong>不改变 lm_head 和词表并行的主干</strong>，只是在"完整 logits → 采样器（第 28 课）"这段缝隙里，按每条请求的需求<strong>对分数做微调或屏蔽</strong>。把它们和末位切片、跨卡汇总放在一起，就构成了 <span class="mono">LogitsProcessor</span> 到 <span class="mono">Sampler</span> 之间这"最后一公里"的全部工作：先裁、再投影、再拼齐、再按需打掩码加偏置，最后才交给采样器落子。</p>

<p>这里有个容易踩的顺序问题值得点明：<strong>掩码和偏置必须在 logits 上、采样之前施加</strong>，而不能在采样之后补救。因为采样器是从一个概率分布里抽样，一旦某个非法 token 进入了候选、被 top-k/p 选中并抽出来，就已经晚了；唯有事先把它的 logit 压成负无穷，softmax 之后它的概率才<strong>恰好为零</strong>，从根上断绝被选中的可能。结构化输出之所以能<strong>保证</strong>每一步都合法，正是靠这种"在分数层面提前封死"的机制，而不是生成完再校验、出错再重试。这也是为什么 logits 这条向量是兵家必争之地：它是"模型的原始判断"与"用户的硬约束"<strong>唯一能干净汇合</strong>的地方。</p>

<p>这一课也把<strong>整条前向路径收束</strong>了：第 24 课 ModelRunner.forward → 模型主体（第 26 课，含注意力第 33 课、MoE 第 34 课、RoPE/Norm 第 36 课）
→ <strong>LogitsProcessor → logits → Sampler（第 28 课）→ 下一个 token</strong>。词表并行也是继 q/k/v、MLP 之后又一个 TP 切分的例子（第 25/46 课），
它和前面的量化（第 35 课）、注意力后端（第 33 课）一起，构成了 Part 8"算子层"的全貌：模型只管<strong>调用</strong>这些可复用、可切分、可换内核的层，真正的工程复杂度都被收进了层内部。</p>

<p>站远一点看，这一课其实给 Part 8 画上了句号。第 33 课讲注意力后端怎么把"算注意力"这件事抽象成可换的 kernel，第 34 课讲 MoE 怎么用门控只激活少数专家，第 35 课讲量化怎么用更少比特换显存与带宽，第 36 课讲 RoPE 注入位置、RMSNorm 做归一、以及小算子怎么融合——而本课的输出头与词表并行，正是这条流水线的<strong>最后一站</strong>：把算了半天的 hidden 翻译回人能用的 token。这些层无一例外都遵循同一条设计哲学：<strong>模型文件只负责"按顺序调用"，真正的并行、量化、换内核、集合通信都被收进层的内部</strong>。正因如此，SGLang 才能用一份很薄的模型代码，去适配千变万化的硬件、并行度和精度配置——而你现在已经能从 hidden 一路讲到 token，把这条路上的每一站都说清楚了——嵌入怎么进、注意力怎么算、归一怎么稳、输出头怎么切、采样怎么落子，整条流水线再没有黑箱。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>lm_head</strong>：把每位置的 hidden 投影成<strong>词表大小的分数 = logits</strong>，交采样器（第 28 课）出 token。</li>
    <li><strong>词表很大</strong>（3 万~25 万）⇒ lm_head/嵌入是大矩阵，TP 下放不下一张卡。</li>
    <li><strong>词表并行</strong>：<span class="mono">VocabParallelEmbedding</span>/<span class="mono">ParallelLMHead</span> 按词表维切到各 rank，每卡算一段，再 <strong>all-gather</strong> 拼完整。</li>
    <li><strong>末位切片</strong>：只需每条请求最后一位的 logits，先裁 hidden 大幅省算力。</li>
    <li><strong>收束前向</strong>：ModelRunner.forward → 模型 → LogitsProcessor → logits → Sampler → token；又一个 TP 切分例子。</li>
  </ul>
</div>
""",
    "en": r"""
<p class="lead">
After the model finishes computing, it holds a stream of <strong>hidden vectors</strong> per position. But the user wants
"<strong>the next word</strong>", not a vector. Translating a hidden vector into "<strong>a score for every word</strong>" is the
model's <strong>output head, lm_head</strong>; and because vocabularies run into the hundreds of thousands, that head is too big
for one GPU — hence <strong>vocab parallelism</strong>. This lesson covers how logits arise, why the vocab head is split, and how
<span class="mono">LogitsProcessor</span> wraps it up for the Sampler.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the output head as a <strong>giant dictionary</strong> that must score a query against every entry. The dictionary is
  too thick, so it's <strong>split among N librarians</strong>, each scoring only <strong>their slice of entries</strong>; at the end
  they <strong>pool their score sheets (all-gather)</strong> into one full ranking. One more shortcut: you only care what follows the
  <strong>last word you just wrote</strong>, so you ask only for the <strong>last position's</strong> scores, not a re-score of the whole sentence.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  After the final norm (Lesson 36), each position has a hidden vector. The <strong>lm_head</strong> (a [hidden × vocab] matrix)
  projects it to <strong>a score per vocab token = logits</strong>, which the Sampler (Lesson 28) turns into a token. The vocab is huge
  (30k–250k), so lm_head and the embedding are heavy — under tensor parallelism (Lesson 46) they're <strong>split along the vocab
  dimension across ranks</strong>, each GPU computing only its slice, then an <strong>all-gather</strong> assembles the full logits.
  <span class="mono">LogitsProcessor</span> orchestrates this (including the <strong>last-token-only</strong> optimization).
</div>

<h2>From hidden to logits: the lm_head output</h2>
<p>The model body (Lesson 26) computes layer by layer, producing a <strong>hidden vector</strong> per input position (its dim is the
model's hidden size, e.g. 4096). But that's an intermediate representation, not an answer. To predict the next word you need to know
<strong>how "fitting" each candidate token in the vocab is</strong> — that is what <span class="mono">lm_head</span> does: a matrix of
shape <span class="mono">[hidden_dim × vocab_size]</span> that, multiplied with the hidden vector, yields a score vector <strong>as long
as the vocabulary</strong> = the <strong>logits</strong>. The larger a token's logit, the more the model thinks it should come next.
Logits then go to the Sampler (Lesson 28) for temperature, top-k/p and sampling, ending in a concrete token. Many models <strong>tie</strong>
the lm_head with the input embedding, since "word→vector" and "vector→word scores" are naturally inverse operations.</p>

<div class="fig">
  <svg viewBox="0 0 780 250" role="img" aria-label="The last hidden vector times the lm_head weight matrix gives a logits vector as long as the vocab, one score per token, then argmax or sample picks the next token">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">hidden × lm_head → logits</text>
    <rect x="40" y="70" width="54" height="120" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="67" y="60" text-anchor="middle" style="font-size:12px;fill:var(--muted)">hidden</text>
    <text x="67" y="135" text-anchor="middle" class="mono" style="font-size:12px">4096</text>
    <text x="112" y="136" text-anchor="middle" style="font-size:20px;fill:var(--muted)">×</text>
    <rect x="134" y="58" width="190" height="144" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="229" y="120" text-anchor="middle" class="mono" style="font-size:13px;fill:var(--accent-ink)">lm_head</text>
    <text x="229" y="142" text-anchor="middle" style="font-size:12px;fill:var(--accent-ink)">[hidden × vocab]</text>
    <text x="229" y="164" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--accent-ink)">4096 × 32000</text>
    <text x="346" y="136" text-anchor="middle" style="font-size:20px;fill:var(--muted)">→</text>
    <rect x="368" y="70" width="54" height="120" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="395" y="60" text-anchor="middle" style="font-size:12px;fill:var(--muted)">logits</text>
    <text x="395" y="135" text-anchor="middle" class="mono" style="font-size:12px">32000</text>
    <text x="444" y="136" text-anchor="middle" style="font-size:20px;fill:var(--muted)">→</text>
    <rect x="470" y="100" width="286" height="60" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="613" y="126" text-anchor="middle" style="font-size:13px">argmax / sample</text>
    <text x="613" y="146" text-anchor="middle" style="font-size:12px;fill:var(--muted)">pick next token</text>
  </svg>
  <div class="figcap"><b>Fig 1 · hidden × lm_head → logits</b> — the last position's hidden vector (length 4096) times the lm_head weight [hidden × vocab] gives a logits vector as long as the vocab (32000), one score per token; argmax or sampling then picks the next token.</div>
</div>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Body forward</h4><p>embed → N layers → final norm (Lesson 36) → a hidden per position.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Take the last token</h4><p>In decode, keep <strong>only each request's last position</strong> — only its logits predict the next word.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>lm_head projection</h4><p>hidden × lm_head → vocab-sized <strong>logits</strong> (each rank computes a slice).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Gather + sample</h4><p><strong>all-gather</strong> assembles full logits → Sampler (Lesson 28) emits the token.</p></div></div>
</div>

<div class="flow">
  <div class="node"><div class="nt">hidden·last pos</div><div class="nd">the last position's<br>hidden per request</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">lm_head·vocab slice</div><div class="nd">this rank scores only<br>its segment of vocab</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">all-gather</div><div class="nd">stitch the partial<br>segments across GPUs</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">full logits</div><div class="nd">length = vocab size,<br>one score per token</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Sampler·Lesson 28</div><div class="nd">temperature / top-k/p<br>picks the next token</div></div>
</div>

<h2>Take only the last position: one row of logits beats two thousand</h2>
<p>"Keep only the last position" sounds like a minor tweak, but the compute it saves is substantial. Picture a 2000-token prompt: during prefill the model genuinely has to compute hidden for <strong>all 2000 positions</strong> — every layer's attention needs the representations of all preceding positions. But when we want to predict the <strong>2001st token</strong>, the only useful hidden is the <strong>last position's</strong>: it has already "seen" the whole prompt, and its logits are the next word's distribution. The logits of the other 1999 positions are, in ordinary generation, <strong>completely unused</strong>.</p>
<p>So before calling lm_head, <span class="mono">LogitsProcessor</span> first prunes hidden from <span class="mono">[2000, hidden]</span> to <span class="mono">[1, hidden]</span>, <strong>letting only the last row pass through that [hidden × vocab] matrix</strong>. lm_head is one of the most expensive matmuls in the whole forward (tens to hundreds of thousands of vocab columns); shrinking its input from 2000 rows to 1 cuts this step's FLOPs and HBM traffic by <strong>2000×</strong>. Batching works the same way: each request contributes only its own last position, so a batch computes only as many logit rows as it has requests, not every position of every prompt.</p>
<p>The one exception is when <strong>the user asks for logprobs</strong>: returning the log-probability of each prompt token (used by eval, scoring, and speculative-decoding draft verification) requires keeping those positions' logits. <span class="mono">logits_metadata</span> is exactly what records "which positions in this batch need full logits and which need only the last" so LogitsProcessor prunes on demand — down to just the last by default, keeping extra rows only when needed. That's why the same code is fast for ordinary generation but a touch slower with <span class="mono">logprobs</span> on: the latter feeds that big matmul several more rows.</p>
<p>In decode this optimization is automatic: each step advances one token, so each request's hidden is already just <strong>one row</strong> (the newest position) and lm_head naturally scores one row. So "last-token slice" is an <strong>active prune</strong> in prefill and <strong>already-the-case</strong> in decode — the two modes (Lesson 4) collapse to one sentence here: <strong>each request takes one row of hidden to compute logits</strong>. Precisely because the output head touches only these few rows, its share of forward time is usually small; the real cost is the dozens of attention and MLP layers in the middle. Grasp this and you can explain a common puzzle: the vocab head is a big chunk of <strong>HBM</strong>, yet spends little <strong>time</strong> per decode step — because it computes so few rows.</p>

<h2>The vocab is huge: why split it across TP</h2>
<p>There's an easily-missed scale problem here: the <strong>vocab is very large</strong>. Common models have 30k–250k tokens, and lm_head
is <span class="mono">[hidden × vocab]</span>, the embedding <span class="mono">[vocab × hidden]</span> — at vocab=150k, hidden=8192,
<strong>this one table alone is over a billion parameters</strong>, a sizable chunk of the model. Under tensor parallelism (Lesson 46, which
splits each layer's matrices across GPUs), you <strong>can't put the whole vocab head on one GPU</strong>, nor should one GPU bear that
much matmul and HBM.</p>

<p>The fix is <strong>vocab parallelism</strong>: <span class="mono">VocabParallelEmbedding</span> and <span class="mono">ParallelLMHead</span>
split the <strong>vocab dimension</strong> into segments, one per TP rank — e.g. 4 GPUs, 150k vocab, ~37.5k tokens each. On the forward,
<strong>each GPU uses only its slice of lm_head to compute "logits for its segment of words"</strong>, producing a <strong>partial</strong>
score vector (only its own segment is real). To get the full logits you stitch the segments together — that step is a cross-GPU
<strong>all-gather</strong> (or gathered in-place inside argmax/sampling). So no one holds the whole vocab head; HBM and FLOPs are spread across GPUs.</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="The vocab is sharded across TP ranks: rank0 owns tokens 0 to 16000, rank1 owns 16000 to 32000, each computes logits for its slice, then all-gather concatenates them into the full 32000-length logits">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">vocab sharded by rank → all-gather</text>
    <rect x="24" y="120" width="54" height="80" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="51" y="110" text-anchor="middle" style="font-size:12px;fill:var(--muted)">hidden</text>
    <text x="51" y="165" text-anchor="middle" class="mono" style="font-size:11px">4096</text>
    <line x1="78" y1="160" x2="132" y2="95" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="78" y1="160" x2="132" y2="225" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="134" y="60" width="220" height="70" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="244" y="86" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--teal)">rank0 · lm_head</text>
    <text x="244" y="108" text-anchor="middle" style="font-size:12px">vocab 0–16000</text>
    <rect x="134" y="190" width="220" height="70" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="244" y="216" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--amber)">rank1 · lm_head</text>
    <text x="244" y="238" text-anchor="middle" style="font-size:12px">vocab 16000–32000</text>
    <line x1="354" y1="95" x2="408" y2="140" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="354" y1="225" x2="408" y2="160" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="410" y="115" width="140" height="70" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="480" y="148" text-anchor="middle" class="mono" style="font-size:13px;fill:var(--accent-ink)">all-gather</text>
    <text x="480" y="168" text-anchor="middle" style="font-size:11px;fill:var(--accent-ink)">concat slices</text>
    <text x="566" y="155" text-anchor="middle" style="font-size:20px;fill:var(--muted)">→</text>
    <rect x="588" y="115" width="168" height="70" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="672" y="148" text-anchor="middle" style="font-size:13px;fill:var(--purple)">full logits</text>
    <text x="672" y="168" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--purple)">length 32000</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Vocab sharded across TP ranks</b> — lm_head's vocab dimension is split across ranks: rank0 owns tokens 0–16000, rank1 owns 16000–32000, each computes only its slice's logits; an all-gather then concatenates the two into the full 32000-length logits.</div>
</div>

<p>Unpacking the mechanism: each rank's lm_head is just that narrow strip <span class="mono">[hidden × vocab/N]</span>, so what it computes is a partial score vector that is <strong>real only for its own segment and empty elsewhere</strong>. What <strong>all-gather</strong> does is have all N GPUs each send out "their stretch" and receive everyone else's, finally <strong>concatenating them end-to-end in rank order</strong> into one full logits vector as long as the entire vocab. Worth noting: for plain <strong>greedy decoding</strong> (argmax of the highest score), you can do even better — each GPU first takes a <strong>local max within its segment</strong> and all-reduces only "the local max value + its global token id", which suffices to pick the global max <strong>without ever materializing the full hundred-thousand-dim vector</strong>. SGLang takes this cheaper in-place reduction path whenever full logits aren't needed (no logprobs, no complex sampling).</p>

<p>Vocab parallelism isn't an isolated trick — it's the <strong>same idea</strong> as the q/k/v and MLP splits in Lessons 25/46: cut a big matrix along some dimension into N slices, each GPU computes one, then one collective realigns the results. Attention splits the <strong>attention heads</strong>, MLP splits the <strong>intermediate dim</strong>, and here we split the <strong>vocab dim</strong>; the only difference is "which dim, and whether all-reduce or all-gather closes it out". Grasp vocab parallelism and you've <strong>completed</strong> the picture of TP's sharding at the model's two ends (input embedding, output head) and its middle layers — the whole network, end to end, is cut and reassembled by one rule.</p>

<p>Two implementation details people often ask about. First, <strong>vocab alignment</strong>: the vocab size needn't divide evenly by the rank count (e.g. 150k across 8 GPUs doesn't), so <span class="mono">VocabParallelEmbedding</span> first <strong>pads the vocab up</strong> to a divisible size; the extras are "fake tokens" whose logits get masked and can never be selected, existing only to give each GPU an equal-length segment and aligned communication. Second, <strong>shared embedding and output head</strong>: when a model uses tied embeddings, the input-side <span class="mono">VocabParallelEmbedding</span> and the output-side <span class="mono">ParallelLMHead</span> are actually the <strong>same vocab-sharded weight</strong> — used to look token ids into vectors on input and to score vectors into logits on output, saving a whole big table's worth of HBM. Shared or not, the split dimension and the communication are identical.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>150k vocab split across 4 ranks</b>: each rank computes only its segment's logits, then all-gather assembles the full vector</div>
  <div class="cells"><span class="lab">rank0</span><span class="cell hl">token 0–37499</span><span class="sep">·</span><span class="lab">rank1</span><span class="cell hl">37500–74999</span><span class="sep">·</span><span class="lab">rank2</span><span class="cell hl">75000–112499</span><span class="sep">·</span><span class="lab">rank3</span><span class="cell hl">112500–149999</span></div>
</div>

<div class="cols">
  <div class="col"><h4>Whole vocab head on one GPU</h4><p>lm_head + embedding's billion+ params all on one GPU: <strong>HBM pressure, and this step's compute becomes a bottleneck</strong>, inconsistent with TP-splitting the other layers.</p></div>
  <div class="col"><h4>Vocab parallel</h4><p>Each rank holds and computes <strong>one vocab segment</strong>; then <strong>all-gather</strong> assembles full logits. <strong>HBM and FLOPs spread</strong>, in line with TP.</p></div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/vocab_parallel_embedding.py ::ParallelLMHead</span><span class="ln">output projection: hidden -&gt; per-token logit, vocab sharded by TP</span></div>
  <pre><span class="kw">class</span> ParallelLMHead(VocabParallelEmbedding):
    <span class="cm"># output projection: hidden_state -&gt; one logit per vocab token.</span>
    <span class="cm"># the VOCAB dimension is sharded across TP ranks (each rank owns a</span>
    <span class="cm"># slice of rows); vocab size is padded to divide evenly by TP size.</span>
    <span class="kw">def</span> __init__(self, num_embeddings, embedding_dim, *, bias=False, ...):
        <span class="cm"># num_embeddings = vocab size, embedding_dim = hidden size</span>
        ...
    <span class="cm"># each rank produces logits for ITS vocab slice; an all-gather</span>
    <span class="cm"># (in LogitsProcessor) stitches them into the full-vocab vector.</span></pre>
</div>

<p>A concrete instance: Llama's vocab is <span class="mono">32000</span> and hidden <span class="mono">4096</span>, so lm_head is a <span class="mono">4096 × 32000</span> matrix. Under <span class="mono">--tp-size 2</span> each rank holds only <strong>16000</strong> vocab rows and computes logits for just those 16000 tokens; an all-gather then forms the full <span class="mono">32000</span>-length logits vector.</p>

<h2>LogitsProcessor's job: last-token slice + cross-GPU gather</h2>
<p>The thing that orchestrates all this is <span class="mono">LogitsProcessor</span>. At the end of the forward it takes hidden states and
does three things: ① <strong>last-token slice</strong> — in decode each request only needs the <strong>last position's</strong> logits to predict
the next word; in prefill, although hidden was computed for the whole prompt, <strong>only the last position's logits matter</strong> (unless
computing logprobs). So pruning hidden to "each request's last position" greatly cuts the lm_head work. ② <strong>vocab-parallel projection</strong>
— use <span class="mono">ParallelLMHead</span> to compute the logits for this rank's vocab segment. ③ <strong>cross-GPU all-gather</strong> —
assemble the segments into full logits (gated by <span class="mono">do_tensor_parallel_all_gather</span>, skipped on a single GPU). The result,
<span class="mono">next_token_logits</span>, goes to the Sampler (Lesson 28).</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/logits_processor.py ::LogitsProcessor</span><span class="ln">orchestrating the output head</span></div>
  <pre><span class="kw">class</span> LogitsProcessor(nn.Module):
    <span class="kw">def</span> forward(self, input_ids, hidden_states,
                lm_head: VocabParallelEmbedding, logits_metadata):
        <span class="cm"># 1) Both decode and prefill keep only each request's LAST hidden</span>
        pruned = self._get_pruned_states(hidden_states, logits_metadata)
        <span class="cm"># 2) Vocab-parallel projection: this rank computes its vocab segment</span>
        logits = torch.matmul(pruned, lm_head.weight.T)  <span class="cm"># [tokens, vocab/segment]</span>
        <span class="kw">if</span> self.do_tensor_parallel_all_gather:       <span class="cm"># 3) stitch into full logits</span>
            logits = tensor_model_parallel_all_gather(logits)
        <span class="kw">return</span> LogitsProcessorOutput(next_token_logits=logits)</pre>
</div>

<h2>The last mile before sampling: logprobs, structured masks, and logit bias</h2>
<p>After logits are stitched and before sampling actually happens, a few post-processing steps "hang" near the output head, all acting on this vocab-dimension vector. First is <strong>logprob</strong>: a log-softmax over logits gives each token's log-probability; whether the user wants top-k logprobs, to score each prompt token, or speculative decoding needs to verify draft tokens, the numbers come from here. Second is the <strong>structured-output vocab mask</strong> (Lesson 48): when a request demands valid JSON or a grammar match, the constraint engine computes "which tokens are <strong>allowed vs forbidden</strong> at this step" and drives the forbidden tokens' logits to negative infinity, so the sampler can never pick them.</p>
<p>Third is <strong>logit bias</strong>: the user can manually add a positive or negative bias to certain tokens, raising or lowering their odds of being chosen — again, just adding a number at the corresponding position of this logits vector. What these share is that they <strong>don't touch the lm_head and vocab-parallel backbone</strong>; they merely <strong>tweak or mask scores</strong> per request in the gap between "full logits → the Sampler (Lesson 28)". Put together with the last-token slice and the cross-GPU gather, they form the entire "last mile" between <span class="mono">LogitsProcessor</span> and <span class="mono">Sampler</span>: prune, project, stitch, then mask-and-bias on demand, and only then hand off to the sampler to make its move.</p>
<p>One ordering pitfall worth naming: <strong>masks and biases must be applied on the logits, before sampling</strong>, never patched up afterward. Because the sampler draws from a probability distribution, once an illegal token has entered the candidate set, been selected by top-k/p and drawn, it's already too late; only by driving its logit to negative infinity beforehand does its post-softmax probability become <strong>exactly zero</strong>, cutting off any chance of selection at the root. Structured output can <strong>guarantee</strong> legality at every step precisely because of this "seal it off at the score level in advance" mechanism, not by generating then validating then retrying on error. That's why this logits vector is so contested: it's the <strong>only place where</strong> "the model's raw judgment" and "the user's hard constraints" can cleanly meet.</p>

<p>This lesson also <strong>closes the whole forward path</strong>: Lesson 24's ModelRunner.forward → the model body (Lesson 26, with
attention L33, MoE L34, RoPE/Norm L36) → <strong>LogitsProcessor → logits → Sampler (Lesson 28) → the next token</strong>. Vocab parallelism
is another TP-sharding example after q/k/v and MLP (Lessons 25/46), and together with quantization (Lesson 35) and the attention backend
(Lesson 33) it completes Part 8's "operator layer": the model just <strong>calls</strong> these reusable, shardable, kernel-swappable layers,
and the real engineering complexity is tucked inside them.</p>

<p>Step back and this lesson really puts a period on Part 8. Lesson 33 showed how the attention backend abstracts "compute attention" into a swappable kernel, Lesson 34 how MoE gates to activate only a few experts, Lesson 35 how quantization trades fewer bits for HBM and bandwidth, Lesson 36 how RoPE injects position, RMSNorm normalizes, and small ops fuse — and this lesson's output head and vocab parallelism are the pipeline's <strong>final stop</strong>: translating all that hard-won hidden back into a token a human can use. Every one of these layers follows the same design philosophy: <strong>the model file only "calls in order", while the real parallelism, quantization, kernel swaps, and collectives are tucked inside the layers</strong>. That's exactly why SGLang can fit a very thin model file to wildly varying hardware, parallelism degrees, and precision configs — and you can now narrate the whole way from hidden to token, explaining every stop along the road: how embeddings go in, how attention computes, how norm steadies, how the output head splits, how sampling lands the move, with no black box left in the pipeline.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>lm_head</strong>: projects each position's hidden into <strong>vocab-sized scores = logits</strong>, fed to the Sampler (Lesson 28).</li>
    <li><strong>Vocab is huge</strong> (30k–250k) ⇒ lm_head/embedding are big matrices, too big for one GPU under TP.</li>
    <li><strong>Vocab parallel</strong>: <span class="mono">VocabParallelEmbedding</span>/<span class="mono">ParallelLMHead</span> split vocab across ranks, each computes a segment, then <strong>all-gather</strong> assembles the full logits.</li>
    <li><strong>Last-token slice</strong>: only each request's last position's logits are needed, so prune hidden first to save compute.</li>
    <li><strong>Closes the forward</strong>: ModelRunner.forward → model → LogitsProcessor → logits → Sampler → token; another TP-sharding example.</li>
  </ul>
</div>
""",
}
