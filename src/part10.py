"""Part 10 · Performance innovations (L43-48).

Lesson content for the headline performance-innovation part of the SGLang visual guide:
speculative decoding, EAGLE, PD disaggregation, TP/PP/EP/DP, EPLB, and structured outputs.
Each LESSON_XX is a {"zh": html, "en": html} dict consumed via registry.CONTENT.
"""

LESSON_43 = {"zh": r"""
<p class="lead">普通自回归一次目标前向只吐<strong>一个</strong> token，而解码阶段是<strong>带宽受限</strong>的（第4课）：每一步都要把全部权重和 KV 从 HBM 重新读一遍，算力被活活饿着。投机解码（Speculative decoding）打破"一次前向一个 token"的天花板——让一个便宜的<span class="mono">draft</span>草稿模型先猜 k 个，再让昂贵的<span class="mono">target</span>目标模型<strong>一次前向</strong>把这 k 个全部验完，从而在一步里吐出多个 token，且输出分布与原始目标采样<strong>完全一致</strong>（无损）。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把目标模型想成一位<strong>资深主编</strong>：他逐字逐句亲自写稿又快又准，但每写一个字都要把整本风格手册翻一遍，慢得让人心疼。于是请来一位<strong>实习生（草稿模型）</strong>，让他凭直觉飞快地先写出一整句草稿（k 个词）。主编不再一个字一个字写，而是把这一整句草稿<strong>一眼扫过去做"批改"</strong>：从头读，只要和他心里想的一致就打勾，直到遇到第一个不对的字为止——前面打勾的部分全部采纳，再由主编<strong>顺手补上一个他自己确定的字</strong>（bonus_token）。实习生猜得越准，主编一次批改就能定下越多的字；可万一实习生跑题，主编也只是从分歧点重写，<strong>绝不会让质量打折</strong>。这就是"便宜的人狂猜、昂贵的人一次验完"的分工。</p>
<p>这个类比里有三处和真实算法严丝合缝的细节值得点出。其一，主编"一眼扫过去批改"对应目标模型把 k 个位置塞进<strong>同一次前向</strong>并行打分——他读完整句草稿花的力气，和只读一个字差不多，这正是加速的来源。其二，"遇到第一个不对的字就停、前面全采纳、后面全作废"对应只保留<strong>最长正确前缀</strong>：因为后面的字都是顺着那个错字往下写的，地基歪了，上面盖的全得拆。其三，"主编顺手补一个自己确定的字"对应那个雷打不动的 <span class="mono">bonus_token</span>：哪怕实习生整句都跑偏、一个字没被采纳，主编也总归落下了一个字，所以这套流程<strong>永远不会比主编自己一个字一个字写更慢</strong>，只会更快或持平。</p></div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>解码慢的根因不是算力不够，而是<strong>每步只产出一个 token</strong>却要把整套权重搬一遍，GPU 在等内存。投机解码的全部威力都来自一个朴素的事实：<strong>验证 k 个 token 几乎和验证 1 个 token 一样便宜</strong>，因为目标模型可以把 k 个位置<strong>并行打分</strong>放进同一次前向里。于是只要草稿够准，我们就用"一次前向 + 多个被接受的 token"摊薄了那次昂贵的权重搬运。SGLang 把这件事做成<strong>可插拔</strong>的：用一个 <span class="mono">SpeculativeAlgorithm</span> 枚举来选"怎么出草稿"，用一个 <span class="mono">BaseSpecWorker</span> 来统一编排"谁提议、谁验证"。算得快不快，最终只看两个数：接受率和接受长度。</p>
<p>还要破除一个常见误解：投机解码<strong>不是用小模型的质量去近似大模型</strong>，最终每一个吐出的 token 都经过目标模型的把关，质量与纯目标采样无异。草稿模型在这里扮演的只是一个"<strong>大胆的猜测器</strong>"，它猜错没有任何质量代价，只有"这次猜白费了、得重新搬一次权重"的速度代价。因此选草稿模型的标准和选生成模型完全不同：我们不在乎它单独用时输出好不好，只在乎它与目标模型"<strong>想法有多合拍</strong>"——合拍即高 α。这也是为什么 EAGLE 这类方法要直接复用目标模型的隐藏态来训练草稿头（第44课）：让草稿"读到"目标的内部状态，才能猜得格外像目标。</p></div>

<h2>一、为什么自回归会被带宽卡住</h2>
<p>回忆第4课：在解码阶段，每生成一个 token，GPU 都要把模型的<strong>全部权重</strong>外加不断增长的 <span class="mono">KV cache</span> 从 HBM 读进片上，做完一次矩阵乘，再吐出<strong>恰好一个</strong> token。这一步的算术强度极低——搬进来的字节远多于要做的浮点运算，于是瓶颈是<strong>显存带宽</strong>而非算力，GPU 的张量核心大部分时间在空转等数据。换句话说，<strong>每个 token 都要付一次"全权重搬运"的固定成本</strong>，而这次搬运里真正干的活（一个 token 的前向）少得可怜。如果能在<strong>同一次搬运</strong>里多产出几个 token，单位 token 的成本就被摊薄了，这正是投机解码瞄准的命门。</p>
<p>这里有个容易被忽视却至关重要的不对称性：<strong>前向的成本几乎不随一次喂入多少个 token 而线性增长</strong>。预填充（prefill）阶段之所以高效，正是因为它把一整段 prompt 的所有位置塞进同一次前向并行算完；而解码之所以低效，是因为它被迫一次只算一个新位置。投机解码本质上是在<strong>把"解码"重新变回"小批 prefill"</strong>：用草稿先猜出未来的 k 个位置，让目标模型像处理 prompt 那样把这 k 个位置一口气并行算掉。只要这 k 个位置里有相当一部分能被接受，我们就用<strong>一次前向的代价换来了多个 token 的产出</strong>，带宽这道墙就被绕了过去。值得强调的是，加速的上限并不取决于草稿模型有多快（它本来就便宜），而取决于<strong>草稿猜得有多准</strong>——准，才能让昂贵的目标前向不白跑。</p>

<h2>二、草稿提议 + 目标验证：一次前向多个 token</h2>
<p>投机解码引入两个模型：一个便宜的 <span class="mono">draft</span>（草稿/小模型）和一个昂贵的 <span class="mono">target</span>（目标/大模型）。流程是：草稿模型先<strong>自回归地提议 k 个 token</strong>（它很小，连跑 k 步也便宜）；接着目标模型把"原始上下文 + 这 k 个草稿"拼在一起，<strong>一次前向</strong>就对所有 k 个位置并行打分。然后从头开始<strong>逐个比对</strong>：草稿 token 落在目标分布里就<span class="mono">accept</span>接受，直到遇到第一个被拒的位置；我们保留<strong>最长的正确前缀</strong>，并且无论如何都由目标模型在分歧点（或全部接受后的下一位）补出<strong>一个一定会吐的</strong> <span class="mono">bonus_token</span>。当草稿质量高时，一次目标前向就能产出"若干被接受的草稿 + 1 个 bonus"，速度倍增；而由于接受/拒绝采用了与目标采样数学等价的判据，<strong>最终输出分布与直接用目标模型采样逐位生成完全相同——这是可证明的无损</strong>。</p>
<p>这里"逐个比对"的判据值得展开。对每个草稿位置，目标模型给出它自己的概率分布，把草稿提议 token 在该分布下的概率与草稿模型当初给它的概率相比：若目标更"认可"这个 token 就接受，否则以一定概率拒绝，并从一个经过精心构造的<strong>残差分布</strong>里重采一个 token 顶替——正是这个残差校正保证了整体分布与纯目标采样毫厘不差。一旦某个位置被拒，<strong>它之后的所有草稿 token 都必须丢弃</strong>（因为它们建立在一个错误的前缀之上），这也解释了为什么我们只能保留"最长正确前缀"。无论这一步接受了 0 个还是 k 个草稿，目标模型都会在末尾免费追加一个由它自己采出的 <span class="mono">bonus_token</span>，所以哪怕草稿全军覆没，这一步也至少前进 1 个 token，绝不比普通自回归慢。</p>

<div class="flow"><div class="node">draft 提议 k 个 token</div><div class="arrow">→</div><div class="node">target 一次前向并行验完 k 个</div><div class="arrow">→</div><div class="node">接受最长正确前缀 + 补 bonus_token</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="草稿提议 k 个，目标一次前向并行验证，接受最长前缀再补一个 bonus">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">草稿提议 k 个 token</text>
    <rect x="24" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="66" y="64" text-anchor="middle" class="mono" style="font-size:12px">d1</text>
    <rect x="120" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="162" y="64" text-anchor="middle" class="mono" style="font-size:12px">d2</text>
    <rect x="216" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="258" y="64" text-anchor="middle" class="mono" style="font-size:12px">d3</text>
    <rect x="312" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="354" y="64" text-anchor="middle" class="mono" style="font-size:12px">d4</text>
    <line x1="210" y1="80" x2="210" y2="108" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="104" style="fill:var(--muted);font-size:12px">target：一次前向</text>
    <rect x="24" y="114" width="540" height="40" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="294" y="139" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">ONE forward 并行验完 k 个</text>
    <line x1="210" y1="154" x2="210" y2="182" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="178" style="fill:var(--muted);font-size:12px">接受最长前缀 + bonus</text>
    <rect x="24" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="212" text-anchor="middle" class="mono" style="font-size:12px">d1 ✓</text>
    <rect x="120" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="162" y="212" text-anchor="middle" class="mono" style="font-size:12px">d2 ✓</text>
    <rect x="216" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="258" y="212" text-anchor="middle" class="mono" style="font-size:12px">d3 ✓</text>
    <rect x="312" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="354" y="212" text-anchor="middle" class="mono" style="font-size:12px">bonus</text>
    <rect x="430" y="190" width="84" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="472" y="212" text-anchor="middle" class="mono" style="font-size:12px">d4 ✗</text>
    <rect x="24" y="258" width="16" height="12" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="48" y="269" style="fill:var(--muted);font-size:12px">接受</text>
    <rect x="120" y="258" width="16" height="12" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="144" y="269" style="fill:var(--muted);font-size:12px">拒绝</text>
  </svg>
  <div class="figcap"><b>图 1 · 草稿提议 → 目标一次验证</b> — 便宜的草稿模型先猜 k 个 token，昂贵的 target 在<strong>同一次前向</strong>里并行打分；从头比对，匹配的前缀（绿）被接受，第一个不匹配处（红）拒绝其后全部，再由 target 补一个 <span class="mono">bonus</span>。图中 4 个里中了 3 个，加 bonus 共吐 4 个 token。</div>
</div>

<h2>三、两个核心指标：accept_rate(α) 与 accept_length(τ)</h2>
<p>衡量投机解码好坏，论文里用两个术语，务必分清楚。<strong>accept_rate</strong>（接受率 α）= <span class="mono">correct_drafts / proposed_drafts</span>，即<strong>每个草稿 token</strong>被接受的概率，它<strong>不包含</strong>那个白送的 bonus_token——它衡量草稿模型"猜得准不准"。<strong>accept_length</strong>（接受长度 τ）= <strong>每个验证步平均吐出的 token 数</strong>，它<strong>包含</strong>那个一定会吐的 bonus_token——它衡量"一次目标前向到底能产出几个 token"。两者都越大越好：α 越高说明草稿越靠谱，τ 越大说明每次昂贵前向摊到的 token 越多、加速越猛。直觉上 τ 随 α 单调上升，但因为有 bonus，<strong>即使 α 很低，τ 也至少为 1</strong>（最坏情况退化成普通自回归，绝不更慢）。</p>
<p>把这两个指标和成本放在一起，就能估算加速比。设草稿一次提议 k 个，则每个验证步的<strong>收益</strong>是 τ 个 token，而<strong>代价</strong>主要是一次目标前向加上草稿模型跑 k 步的小开销。粗略地说，理想加速比约等于 τ 除以"一次目标前向 + 草稿开销"折算出的等效前向数；当草稿很小、几乎免费时，加速比就近似正比于 τ。这也揭示了一个实践权衡：<strong>k 不是越大越好</strong>。k 太小，τ 的上限被压低，省不了多少；k 太大，靠后的草稿位置因为前缀更长、越来越难猜中，α 会下滑，被拒后白白浪费草稿算力，还增大了一次验证的张量规模。因此真实系统会围绕模型对、任务难度去调 k，让 α 和 τ 的乘积效应落在最甜的点上。一个常被引用的经验是：<strong>α 决定了 τ 的天花板，而 bonus 决定了 τ 的地板</strong>。在实践中，人们会把这两个指标连同端到端吞吐一起打点观测：α 偏低往往提示草稿模型与目标不够合拍、需要换草稿方法或重训草稿头；τ 上不去则提示 k 设得太保守或接受太苛刻。把它们当成<strong>投机解码的体温计</strong>，调参时就有了明确的抓手。</p>

<div class="cols"><div class="col"><strong>基线（普通自回归）</strong><br/>每次目标前向 = <span class="mono">1</span> 个 token<br/>付一次全权重搬运只换一个 token<br/>带宽受限、算力空转</div><div class="col"><strong>投机解码</strong><br/>每次目标前向 = <span class="mono">m</span> 个 token（被接受前缀 + bonus）<br/>同一次搬运摊薄到多个 token<br/>m 由 accept_length(τ) 决定</div></div>

<div class="fig">
  <svg viewBox="0 0 800 260" role="img" aria-label="基线每次前向只吐 1 个 token，投机每次前向吐 m 个 token，两条时间线对比">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">基线：1 token / 前向</text>
    <rect x="24" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="70" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="124" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="170" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="224" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="270" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="324" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="370" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="424" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="470" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="524" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="570" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <text x="628" y="65" style="fill:var(--amber);font-weight:700;font-size:12px">6 前向</text>
    <text x="24" y="128" style="font-weight:700;fill:var(--muted)">投机：m token / 前向 (m&gt;1)</text>
    <rect x="24" y="140" width="92" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="70" y="165" text-anchor="middle" class="mono" style="font-size:11px">3 tok</text>
    <rect x="124" y="140" width="92" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="170" y="165" text-anchor="middle" class="mono" style="font-size:11px">3 tok</text>
    <text x="228" y="165" style="fill:var(--teal);font-weight:700;font-size:12px">2 前向 · 同样 6 token</text>
    <line x1="24" y1="210" x2="616" y2="210" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="232" style="fill:var(--faint);font-size:12px">时间 →（更少昂贵前向 = 更快）</text>
  </svg>
  <div class="figcap"><b>图 2 · 基线 vs 投机时间线</b> — 基线每次昂贵的目标前向只产出 <strong>1</strong> 个 token，6 个 token 要 6 次前向；投机每次前向平均吐 <span class="mono">m</span>(&gt;1) 个被接受 token，同样 6 个 token 只需 2 次前向 → <strong>更少昂贵前向、更短总时间</strong>。</div>
</div>

<h2>四、SGLang 把"出草稿的方法"做成可插拔</h2>
<p>不同的"怎么提议草稿"对应不同算法，SGLang 用一个 <span class="mono">SpeculativeAlgorithm</span> 枚举来选择，成员包括 <span class="mono">EAGLE</span>、<span class="mono">EAGLE3</span>、<span class="mono">NGRAM</span>、<span class="mono">STANDALONE</span>、<span class="mono">DFLASH</span>、<span class="mono">FROZEN_KV_MTP</span>、<span class="mono">NONE</span>。它们共享同一套"提议→验证→接受"骨架，只在"草稿从哪来"上不同：有的训练一个轻量草稿头复用目标的隐藏态（EAGLE 系列，详见<strong>第44课</strong>），有的直接用 n-gram 命中历史文本，有的挂一个独立小模型。编排这一切的是 <span class="mono">BaseSpecWorker</span>：它持有一个 <span class="mono">target_worker</span>（大模型，负责一次验完 k 个）和一个 <span class="mono">draft_worker</span>（小模型，负责提议 k 个），并暴露 <span class="mono">clear_cache_pool</span> 在多次运行之间重置草稿/目标的 KV 池。验证这一步本身复用了第28课的<strong>采样器</strong>——接受/拒绝判据就是在采样层面比对目标分布。</p>
<p>把"出草稿"和"验证编排"解耦带来巨大的工程红利：无论草稿来自训练好的 EAGLE 头、廉价的 n-gram 命中，还是一个独立小模型，<span class="mono">BaseSpecWorker</span> 看到的都是同一组接口——拿到 k 个候选、组织一次目标前向、按统一判据接受、追加 bonus、清理缓存池进入下一轮。于是研究者要发明新的草稿方法，只需往 <span class="mono">SpeculativeAlgorithm</span> 枚举里加一个成员并实现对应的 <span class="mono">draft_worker</span>，<strong>验证侧的无损性证明、批处理与缓存管理统统不必重写</strong>。这也呼应了 SGLang 反复出现的设计哲学：把"会变的部分"（怎么猜）和"不该变的部分"（怎么无损地验）清晰隔开。<span class="mono">NONE</span> 成员则是这条谱系的退化端点——选它就等于关掉投机解码、回到第4课那个老老实实一次一个 token 的世界，方便在草稿收益不划算时一键回退。</p>

<table class="t"><tr><th>SpeculativeAlgorithm 取值</th><th>出草稿的思路</th></tr>
<tr><td><span class="mono">EAGLE / EAGLE3</span></td><td>训练轻量草稿头，复用目标隐藏态自回归提议（第44课）</td></tr>
<tr><td><span class="mono">NGRAM</span></td><td>用 n-gram 从已生成/提示文本里直接命中候选，零额外模型</td></tr>
<tr><td><span class="mono">STANDALONE</span></td><td>挂一个独立的小草稿模型来提议</td></tr>
<tr><td><span class="mono">DFLASH / FROZEN_KV_MTP</span></td><td>多 token 预测类方案，复用/冻结 KV 来高效出多个草稿</td></tr>
<tr><td><span class="mono">NONE</span></td><td>关闭投机解码，退化为普通自回归</td></tr></table>

<p>把一整轮看成一个循环：草稿提议、目标一次验证、接受最长正确前缀并补 bonus、然后带着新接受的 token 向前滚动，进入下一轮。循环里真正昂贵的只有那<strong>一次</strong>目标前向，而它一口气定下了 m 个 token。</p>
<p>最后强调一点工程上的取舍：投机解码省的是<strong>时间</strong>，花的是<strong>计算与显存</strong>。每一步都要额外跑一遍草稿模型，目标前向也要为 k 个草稿位置多算一些（被拒的那部分纯属浪费），还要为草稿/目标各自维护 KV 池，这正是 <span class="mono">clear_cache_pool</span> 存在的原因。所以在<strong>请求稀疏、GPU 本就闲着</strong>（低延迟、强带宽受限）的场景里，投机解码收益最大——反正算力空着，拿去赌草稿很划算；而在<strong>高并发、GPU 已被大批次喂饱</strong>（算力受限）的场景里，多出来的草稿计算可能挤占本就紧张的算力，收益变薄甚至为负。因此它不是无脑全开的开关，而是要结合负载、模型对和 α/τ 实测来决定——这也正是 SGLang 把它做成<strong>可插拔、可一键退回 <span class="mono">NONE</span></strong> 的现实理由。</p>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>propose 提议</h4><p class="mono">draft_worker</p><p>草稿模型自回归地提议 k 个草稿 token。</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>verify 验证</h4><p class="mono">target_worker</p><p>目标模型一次前向并行给 k 个位置打分。</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>accept 接受</h4><p class="mono">correct_drafts + bonus_token</p><p>保留最长正确前缀，并一定补出一个 bonus_token。</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>roll forward 滚动</h4><p class="mono">next round</p><p>带着新接受的 token 进入下一轮提议。</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker</span><span class="ln">草稿 worker 提议 + 目标 worker 一次验完</span></div><pre>class BaseSpecWorker(ABC):
    # owns two model workers: a cheap draft + the expensive target

    @property
    @abstractmethod
    def target_worker(self):     # the big model — verifies k drafts in ONE forward
        ...
    @property
    @abstractmethod
    def draft_worker(self):      # the small model — proposes k draft tokens
        ...
    @abstractmethod
    def clear_cache_pool(self):  # reset the draft/target KV pools between runs
        ...</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/spec_info.py ::SpeculativeAlgorithm</span><span class="ln">枚举：选哪种投机算法（或不投机）</span></div><pre>class SpeculativeAlgorithm(Enum):
    # which speculative method to run (or NONE = plain decoding).
    EAGLE = auto()
    EAGLE3 = auto()
    NGRAM = auto()
    STANDALONE = auto()
    NONE = auto()
    @classmethod
    def from_string(cls, name):
        # "eagle" -&gt; EAGLE ; None -&gt; NONE
        ...</pre></div>

<p><span class="mono">--speculative-algorithm EAGLE</span> 即开启投机解码；选 <span class="mono">NONE</span>（默认）就关掉、退回普通自回归。再看一个具体例子：草稿一次提议 4 个 token、其中 3 个被接受，则这<strong>一次</strong> target 前向就吐出 3 + 1 个 bonus = <strong>4 个 token</strong>；接受率（α）越高，每次昂贵前向省下的步数越多，端到端加速就越大。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li>普通自回归一次前向只吐 1 个 token，且解码<strong>带宽受限</strong>（第4课）：每步重读全部权重 + KV，GPU 在等内存。</li>
<li>投机解码用便宜 <span class="mono">draft</span> 提议 k 个、昂贵 <span class="mono">target</span> <strong>一次前向并行验完</strong>，<span class="mono">accept</span> 最长正确前缀再加一个一定吐的 <span class="mono">bonus_token</span>，输出分布<strong>可证明无损</strong>。</li>
<li><strong>accept_rate</strong>(α) = <span class="mono">correct_drafts / proposed_drafts</span>，<strong>不含</strong> bonus；<strong>accept_length</strong>(τ) = 每验证步平均吐出 token 数，<strong>含</strong> bonus；两者越大越快。</li>
<li>SGLang 用 <span class="mono">SpeculativeAlgorithm</span> 枚举（EAGLE、EAGLE3、NGRAM、STANDALONE、DFLASH、FROZEN_KV_MTP、NONE）选草稿方法；<span class="mono">BaseSpecWorker</span> 持有 <span class="mono">target_worker</span> 与 <span class="mono">draft_worker</span>。</li>
<li>命名约定：用 <span class="mono">accept</span> / <span class="mono">bonus_token</span> / <span class="mono">correct_drafts</span>；EAGLE 详见第44课，验证复用第28课采样器，攻击的瓶颈来自第4课。</li>
</ul></div>
""", "en": r"""
<p class="lead">Plain autoregression emits exactly <strong>one</strong> token per target forward, and decode is <strong>bandwidth-bound</strong> (Lesson 4): every step re-reads all weights and the KV cache from HBM, so the compute units sit starved. Speculative decoding breaks the "one token per forward" ceiling — a cheap <span class="mono">draft</span> model first guesses k tokens, then the expensive <span class="mono">target</span> model <strong>verifies all k in ONE forward</strong>, so a single step can emit several tokens while the output distribution stays <strong>provably identical</strong> to plain target sampling (lossless).</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Think of the target model as a <strong>senior editor</strong>: writing every word personally is fast and accurate, but each word forces a full re-read of the style manual — painfully slow. So we hire an <strong>intern (the draft model)</strong> who blasts out a whole draft sentence (k words) from intuition. Instead of writing word by word, the editor <strong>skims the whole draft to "mark it up"</strong>: reading from the start, ticking each word that matches what they had in mind, until the first wrong word. Everything ticked is kept, and the editor <strong>appends one word they are sure of</strong> (the bonus_token). The better the intern guesses, the more words get fixed in one pass; and if the intern goes off-topic, the editor simply rewrites from the divergence point — <strong>never compromising quality</strong>. That is the "cheap guesser, expensive single-pass verifier" division of labor.</p>
<p>Three details of this analogy map exactly onto the real algorithm. First, the editor "skimming the whole draft to mark it up" corresponds to the target packing k positions into <strong>one forward</strong> to score in parallel — reading the whole draft sentence costs about as much as reading a single word, and that is the source of the speedup. Second, "stop at the first wrong word, keep everything before it, void everything after" corresponds to keeping only the <strong>longest correct prefix</strong>: every later word was written following that wrong word, so once the foundation is crooked, everything built on top must be torn down. Third, the editor "appending one word they are sure of" corresponds to the unbreakable <span class="mono">bonus_token</span>: even if the intern's whole sentence goes off-topic and not a single word is kept, the editor still lays down one word, so the process is <strong>never slower than the editor writing word by word</strong> — only faster or tied.</p></div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Slow decode is not a compute shortage — it is producing <strong>one token per step</strong> while hauling the entire weight set, so the GPU waits on memory. The whole power of speculative decoding rests on one plain fact: <strong>verifying k tokens is almost as cheap as verifying 1</strong>, because the target can <strong>score k positions in parallel</strong> inside the same forward. So as long as the draft is accurate, we amortize that expensive weight haul across "one forward + many accepted tokens." SGLang makes this <strong>pluggable</strong>: a <span class="mono">SpeculativeAlgorithm</span> enum picks "how to draft," and a <span class="mono">BaseSpecWorker</span> orchestrates "who proposes, who verifies." How fast it runs comes down to two numbers: acceptance rate and acceptance length.</p>
<p>One common misconception must be dispelled: speculative decoding <strong>does not approximate the big model with the small model's quality</strong> — every emitted token is vetted by the target model, so quality is indistinguishable from pure target sampling. The draft model plays only the role of a "<strong>bold guesser</strong>": a wrong guess carries no quality cost, only the speed cost of "this guess was wasted, we must haul the weights again." The criteria for picking a draft model are therefore entirely different from picking a generation model: we do not care how good its standalone output is, only how "<strong>aligned its thinking is</strong>" with the target — alignment means high α. That is why methods like EAGLE train the draft head by directly reusing the target's hidden states (Lesson 44): letting the draft "read" the target's internal state is what makes it guess so much like the target.</p></div>

<h2>1. Why autoregression is bandwidth-bound</h2>
<p>Recall Lesson 4: during decode, generating one token forces the GPU to read the model's <strong>entire weights</strong> plus the ever-growing <span class="mono">KV cache</span> from HBM into the chip, do one matmul, and emit <strong>exactly one</strong> token. The arithmetic intensity is tiny — far more bytes moved than flops done — so the bottleneck is <strong>memory bandwidth</strong>, not compute, and the tensor cores idle waiting for data. In other words, <strong>every token pays a fixed "full weight haul" cost</strong>, yet that haul does pitifully little work (one token's forward). If we could produce several tokens within the <strong>same haul</strong>, the per-token cost drops — exactly the weak spot speculative decoding attacks.</p>
<p>There is an easily overlooked but crucial asymmetry here: <strong>a forward's cost barely grows with how many tokens you feed it at once</strong>. Prefill is efficient precisely because it stuffs all positions of a whole prompt into one parallel forward; decode is inefficient because it is forced to compute one new position at a time. Speculative decoding essentially <strong>turns "decode" back into "mini-prefill"</strong>: the draft guesses the next k future positions, and the target computes those k positions in parallel just like it would a prompt. As long as a good fraction of those k positions get accepted, we have <strong>traded the cost of one forward for the output of many tokens</strong>, sidestepping the bandwidth wall. Worth stressing: the speedup ceiling does not hinge on how fast the draft model is (it is cheap by construction), but on <strong>how accurately the draft guesses</strong> — accuracy is what keeps the expensive target forward from being wasted.</p>

<h2>2. Draft proposes + target verifies: many tokens per forward</h2>
<p>Speculative decoding uses two models: a cheap <span class="mono">draft</span> (small) and an expensive <span class="mono">target</span> (big). The procedure: the draft model <strong>autoregressively proposes k tokens</strong> (it is small, so even k steps are cheap); the target then concatenates "original context + those k drafts" and <strong>scores all k positions in parallel in ONE forward</strong>. Next we <strong>compare from the start</strong>: a draft token that lands in the target distribution is <span class="mono">accept</span>ed, until the first rejected position; we keep the <strong>longest correct prefix</strong>, and in all cases the target emits one <strong>always-produced</strong> <span class="mono">bonus_token</span> at the divergence point (or right after a full accept). When draft quality is high, one target forward yields "several accepted drafts + 1 bonus," a big speedup; and because accept/reject uses a criterion mathematically equivalent to target sampling, <strong>the final output distribution is exactly the same as sampling token-by-token from the target — provably lossless</strong>.</p>
<p>The "compare from the start" criterion deserves unpacking. At each draft position the target produces its own probability distribution; we compare the draft token's probability under the target distribution against the probability the draft model originally assigned it: if the target "endorses" the token more, accept it; otherwise reject it with some probability and resample a replacement from a carefully constructed <strong>residual distribution</strong> — it is exactly this residual correction that guarantees the overall distribution matches pure target sampling to the letter. Once a position is rejected, <strong>all draft tokens after it must be discarded</strong> (they were built on a wrong prefix), which is why we can only keep the "longest correct prefix." Regardless of whether this step accepted 0 or k drafts, the target appends one free <span class="mono">bonus_token</span> sampled by itself at the end, so even if every draft is wiped out, the step still advances at least 1 token — never slower than plain autoregression.</p>

<div class="flow"><div class="node">draft proposes k tokens</div><div class="arrow">→</div><div class="node">target verifies all k in ONE forward</div><div class="arrow">→</div><div class="node">accept longest correct prefix + bonus_token</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="draft proposes k tokens, target verifies in one forward, accept longest prefix plus a bonus">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">draft proposes k tokens</text>
    <rect x="24" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="66" y="64" text-anchor="middle" class="mono" style="font-size:12px">d1</text>
    <rect x="120" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="162" y="64" text-anchor="middle" class="mono" style="font-size:12px">d2</text>
    <rect x="216" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="258" y="64" text-anchor="middle" class="mono" style="font-size:12px">d3</text>
    <rect x="312" y="42" width="84" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="354" y="64" text-anchor="middle" class="mono" style="font-size:12px">d4</text>
    <line x1="210" y1="80" x2="210" y2="108" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="104" style="fill:var(--muted);font-size:12px">target: ONE forward</text>
    <rect x="24" y="114" width="540" height="40" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="294" y="139" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">ONE forward verifies all k</text>
    <line x1="210" y1="154" x2="210" y2="182" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="178" style="fill:var(--muted);font-size:12px">accept prefix + bonus</text>
    <rect x="24" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="66" y="212" text-anchor="middle" class="mono" style="font-size:12px">d1 ✓</text>
    <rect x="120" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="162" y="212" text-anchor="middle" class="mono" style="font-size:12px">d2 ✓</text>
    <rect x="216" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="258" y="212" text-anchor="middle" class="mono" style="font-size:12px">d3 ✓</text>
    <rect x="312" y="190" width="84" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="354" y="212" text-anchor="middle" class="mono" style="font-size:12px">bonus</text>
    <rect x="430" y="190" width="84" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="472" y="212" text-anchor="middle" class="mono" style="font-size:12px">d4 ✗</text>
    <rect x="24" y="258" width="16" height="12" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="48" y="269" style="fill:var(--muted);font-size:12px">accept</text>
    <rect x="120" y="258" width="16" height="12" rx="3" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="144" y="269" style="fill:var(--muted);font-size:12px">reject</text>
  </svg>
  <div class="figcap"><b>Fig 1 · draft proposes → target verifies once</b> — the cheap draft guesses k tokens; the expensive target scores them all in <strong>ONE forward</strong>. Comparing from the start, the matching prefix (green) is accepted, the first mismatch (red) rejects everything after it, and the target appends one <span class="mono">bonus</span>. Here 3 of 4 match, so 3 + bonus = 4 tokens come out.</div>
</div>

<h2>3. Two core metrics: accept_rate(α) and accept_length(τ)</h2>
<p>To judge speculative decoding, the papers use two terms — keep them straight. <strong>accept_rate</strong> (α) = <span class="mono">correct_drafts / proposed_drafts</span>, the probability that <strong>a single draft token</strong> is accepted; it <strong>excludes</strong> the free bonus_token — it measures how accurately the draft guesses. <strong>accept_length</strong> (τ) = <strong>the average number of tokens emitted per verify step</strong>; it <strong>includes</strong> the always-emitted bonus_token — it measures how many tokens one target forward actually produces. Both are better when larger: higher α means a more reliable draft, larger τ means more tokens amortizing each expensive forward, i.e. more speedup. Intuitively τ rises monotonically with α, but thanks to the bonus, <strong>even if α is very low, τ is at least 1</strong> (the worst case degrades to plain autoregression — never slower).</p>
<p>Put these two metrics together with cost and you can estimate the speedup. If the draft proposes k per round, the <strong>gain</strong> per verify step is τ tokens, while the <strong>cost</strong> is mainly one target forward plus the small overhead of running the draft for k steps. Roughly, the ideal speedup is about τ divided by the effective number of forwards that "one target forward + draft overhead" amounts to; when the draft is tiny and nearly free, the speedup is approximately proportional to τ. This reveals a practical trade-off: <strong>bigger k is not always better</strong>. Too small a k caps τ low and saves little; too large a k makes the later draft positions — sitting on longer prefixes — harder to guess, so α drops, rejected drafts waste compute, and the verify forward grows larger. Real systems therefore tune k around the model pair and task difficulty to land the α·τ product at the sweet spot. A frequently cited rule of thumb: <strong>α sets the ceiling of τ, while the bonus sets the floor of τ</strong>. In practice people instrument both metrics alongside end-to-end throughput: a low α usually signals that the draft is poorly aligned with the target and the drafting method should be swapped or the draft head retrained; a τ that will not climb signals that k is set too conservatively or acceptance is too strict. Treat them as the <strong>thermometer of speculative decoding</strong> and tuning gains a concrete handle.</p>

<div class="cols"><div class="col"><strong>Baseline (plain autoregression)</strong><br/>per target forward = <span class="mono">1</span> token<br/>one full weight haul buys one token<br/>bandwidth-bound, compute idle</div><div class="col"><strong>Speculative decoding</strong><br/>per target forward = <span class="mono">m</span> tokens (accepted prefix + bonus)<br/>same haul amortized over many tokens<br/>m is governed by accept_length(τ)</div></div>

<div class="fig">
  <svg viewBox="0 0 800 260" role="img" aria-label="baseline emits 1 token per forward versus speculative emits m tokens per forward, two timelines">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">baseline: 1 tok / forward</text>
    <rect x="24" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="70" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="124" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="170" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="224" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="270" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="324" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="370" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="424" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="470" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <rect x="524" y="40" width="92" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="570" y="65" text-anchor="middle" class="mono" style="font-size:11px">1 tok</text>
    <text x="628" y="65" style="fill:var(--amber);font-weight:700;font-size:12px">6 fwds</text>
    <text x="24" y="128" style="font-weight:700;fill:var(--muted)">spec: m tok / forward (m&gt;1)</text>
    <rect x="24" y="140" width="92" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="70" y="165" text-anchor="middle" class="mono" style="font-size:11px">3 tok</text>
    <rect x="124" y="140" width="92" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="170" y="165" text-anchor="middle" class="mono" style="font-size:11px">3 tok</text>
    <text x="228" y="165" style="fill:var(--teal);font-weight:700;font-size:12px">2 fwds · same 6 tok</text>
    <line x1="24" y1="210" x2="616" y2="210" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="232" style="fill:var(--faint);font-size:12px">time →  (fewer costly forwards = faster)</text>
  </svg>
  <div class="figcap"><b>Fig 2 · baseline vs speculative timeline</b> — each expensive target forward in the baseline yields just <strong>1</strong> token, so 6 tokens need 6 forwards; speculative emits <span class="mono">m</span>(&gt;1) accepted tokens per forward, so the same 6 tokens take only 2 forwards → <strong>fewer costly forwards, shorter total time</strong>.</div>
</div>

<h2>4. SGLang makes "how to draft" pluggable</h2>
<p>Different ways to "propose drafts" map to different algorithms, which SGLang selects via a <span class="mono">SpeculativeAlgorithm</span> enum whose members include <span class="mono">EAGLE</span>, <span class="mono">EAGLE3</span>, <span class="mono">NGRAM</span>, <span class="mono">STANDALONE</span>, <span class="mono">DFLASH</span>, <span class="mono">FROZEN_KV_MTP</span>, <span class="mono">NONE</span>. They share one "propose→verify→accept" skeleton and differ only in "where the draft comes from": some train a lightweight draft head reusing the target's hidden states (the EAGLE family, see <strong>Lesson 44</strong>), some hit history text with n-grams, some attach an independent small model. Orchestrating all this is <span class="mono">BaseSpecWorker</span>: it holds a <span class="mono">target_worker</span> (the big model, verifies k in one forward) and a <span class="mono">draft_worker</span> (the small model, proposes k), and exposes <span class="mono">clear_cache_pool</span> to reset the draft/target KV pools between runs. Verification itself reuses the <strong>sampler</strong> from Lesson 28 — the accept/reject criterion compares against the target distribution at the sampling layer.</p>
<p>Decoupling "drafting" from "verification orchestration" pays a huge engineering dividend: whether the draft comes from a trained EAGLE head, cheap n-gram hits, or an independent small model, <span class="mono">BaseSpecWorker</span> sees the same set of interfaces — obtain k candidates, organize one target forward, accept under a uniform criterion, append the bonus, clear the cache pool, and move to the next round. So a researcher inventing a new drafting method only needs to add a member to the <span class="mono">SpeculativeAlgorithm</span> enum and implement the corresponding <span class="mono">draft_worker</span>; <strong>the losslessness proof on the verify side, plus batching and cache management, never need rewriting</strong>. This echoes a recurring SGLang design philosophy: cleanly separate "what changes" (how to guess) from "what must not change" (how to verify losslessly). The <span class="mono">NONE</span> member is the degenerate endpoint of this family — selecting it turns speculative decoding off and returns to the Lesson 4 world of honestly emitting one token at a time, a one-switch fallback for when drafting is not worth its cost.</p>

<table class="t"><tr><th>SpeculativeAlgorithm value</th><th>Drafting idea</th></tr>
<tr><td><span class="mono">EAGLE / EAGLE3</span></td><td>train a lightweight draft head, reuse target hidden states to propose autoregressively (Lesson 44)</td></tr>
<tr><td><span class="mono">NGRAM</span></td><td>hit candidates directly from generated/prompt text via n-grams, zero extra model</td></tr>
<tr><td><span class="mono">STANDALONE</span></td><td>attach an independent small draft model to propose</td></tr>
<tr><td><span class="mono">DFLASH / FROZEN_KV_MTP</span></td><td>multi-token-prediction schemes, reuse/freeze KV to emit many drafts efficiently</td></tr>
<tr><td><span class="mono">NONE</span></td><td>disable speculative decoding, degrade to plain autoregression</td></tr></table>

<p>View a full round as a loop: the draft proposes, the target verifies in one pass, we accept the longest correct prefix plus the bonus, then roll forward with the newly accepted tokens into the next round. The only truly expensive part of the loop is that <strong>one</strong> target forward — and it pins down m tokens at once.</p>
<p>One final engineering trade-off to stress: speculative decoding saves <strong>time</strong> by spending <strong>compute and memory</strong>. Each step runs the draft model extra, the target forward computes a bit more for the k draft positions (the rejected ones being pure waste), and KV pools must be maintained for both draft and target — which is exactly why <span class="mono">clear_cache_pool</span> exists. So in scenarios with <strong>sparse requests where the GPU is already idle</strong> (low latency, strongly bandwidth-bound), speculative decoding pays off the most — the compute is sitting free anyway, so betting it on drafts is a bargain; whereas in <strong>high-concurrency scenarios where the GPU is already saturated by large batches</strong> (compute-bound), the extra draft compute can crowd out already-scarce compute, thinning the benefit or even making it negative. It is therefore not a switch to blindly leave on, but one to decide based on load, the model pair, and measured α/τ — which is precisely the practical reason SGLang makes it <strong>pluggable, with a one-switch fallback to <span class="mono">NONE</span></strong>.</p>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>propose</h4><p class="mono">draft_worker</p><p>the draft autoregressively proposes k draft tokens.</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>verify</h4><p class="mono">target_worker</p><p>the target scores all k positions in one forward.</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>accept</h4><p class="mono">correct_drafts + bonus_token</p><p>keep the longest correct prefix and always emit a bonus_token.</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>roll forward</h4><p class="mono">next round</p><p>carry the newly accepted tokens into the next propose.</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker</span><span class="ln">the draft worker proposes, the target worker verifies in one pass</span></div><pre>class BaseSpecWorker(ABC):
    # owns two model workers: a cheap draft + the expensive target

    @property
    @abstractmethod
    def target_worker(self):     # the big model — verifies k drafts in ONE forward
        ...
    @property
    @abstractmethod
    def draft_worker(self):      # the small model — proposes k draft tokens
        ...
    @abstractmethod
    def clear_cache_pool(self):  # reset the draft/target KV pools between runs
        ...</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/spec_info.py ::SpeculativeAlgorithm</span><span class="ln">enum: which speculative algorithm (or none)</span></div><pre>class SpeculativeAlgorithm(Enum):
    # which speculative method to run (or NONE = plain decoding).
    EAGLE = auto()
    EAGLE3 = auto()
    NGRAM = auto()
    STANDALONE = auto()
    NONE = auto()
    @classmethod
    def from_string(cls, name):
        # "eagle" -&gt; EAGLE ; None -&gt; NONE
        ...</pre></div>

<p><span class="mono">--speculative-algorithm EAGLE</span> turns it on; <span class="mono">NONE</span> (the default) turns it off and falls back to plain autoregression. Concretely: if the draft proposes 4 tokens and 3 match, that <strong>single</strong> target forward emits 3 + 1 bonus = <strong>4 tokens</strong>; the higher the acceptance rate (α), the more steps each expensive forward skips, and the bigger the end-to-end speedup.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li>Plain autoregression emits 1 token per forward and decode is <strong>bandwidth-bound</strong> (Lesson 4): every step re-reads all weights + KV, so the GPU waits on memory.</li>
<li>Speculative decoding has a cheap <span class="mono">draft</span> propose k and an expensive <span class="mono">target</span> <strong>verify all k in ONE forward</strong>, <span class="mono">accept</span> the longest correct prefix plus one always-emitted <span class="mono">bonus_token</span>; the output distribution is <strong>provably lossless</strong>.</li>
<li><strong>accept_rate</strong>(α) = <span class="mono">correct_drafts / proposed_drafts</span>, <strong>excludes</strong> the bonus; <strong>accept_length</strong>(τ) = avg tokens emitted per verify step, <strong>includes</strong> the bonus; larger is faster for both.</li>
<li>SGLang selects the drafting method via the <span class="mono">SpeculativeAlgorithm</span> enum (EAGLE, EAGLE3, NGRAM, STANDALONE, DFLASH, FROZEN_KV_MTP, NONE); <span class="mono">BaseSpecWorker</span> owns a <span class="mono">target_worker</span> and a <span class="mono">draft_worker</span>.</li>
<li>Naming: use <span class="mono">accept</span> / <span class="mono">bonus_token</span> / <span class="mono">correct_drafts</span>; EAGLE is Lesson 44, verification reuses the Lesson 28 sampler, and the bottleneck it attacks comes from Lesson 4.</li>
</ul></div>
"""}
LESSON_44 = {"zh": r"""
<p class="lead">第43课我们学会了「草稿 + 验证」：一个小模型先猜 <span class="mono">k</span> 个 token，目标模型一次前向把它们全部验证，接受率 <span class="mono">accept_rate</span>（α）和接受长度 <span class="mono">accept_length</span>（τ）决定了加速比，最后还要补一个 <span class="mono">bonus_token</span>。但朴素草稿有两个硬伤：它只能猜一条「链」，而且草稿模型本身就是一整个第二模型。<strong>EAGLE</strong> 把这两件事都做得更聪明——它在<strong>特征层面</strong>起草，并且提出的是一棵<strong>树</strong>而不是一条链。这一课我们就来拆解 EAGLE 以及它身后的下一代家族。记住一条主线：所有改进都在回答同一个问题——如何让一次本就昂贵的目标前向，确认尽可能多的 token。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象你在玩「猜成语接龙」。<strong>朴素草稿（链）</strong>像是闭着眼睛一口气往下背五个字：只要第二个字背错了，后面三个字全部作废，前功尽弃。<strong>EAGLE 的树状草稿</strong>则像是在每一步都准备好<span class="mono">topk</span>个最可能的分支——「这一步可能是甲、可能是乙、可能是丙」——然后把所有这些可能性编织成一棵分叉的树。老师（目标模型）只看一眼整棵树，就能沿着最长的那条正确路径一路打勾。即使某一条分支错了，旁边的<strong>兄弟分支</strong>还在，不至于满盘皆输。树覆盖的「后续可能」比单链多得多，所以一次验证能接受的 token 也更多。</p>
<p>另外，朴素草稿要养一个完整的「第二个学生」来猜词，成本不低。EAGLE 聪明在：它不另起炉灶，而是直接借用老师脑中刚刚算好的<strong>隐藏状态</strong>（第8课）当线索，只配一个很小的「草稿头」来续写下一个特征。线索来自老师本人，所以草稿天然和老师对齐，猜得又快又准。你可以把这想象成：与其请一个陌生的家教来另猜一遍答案，不如直接看老师草稿纸上写到一半的思路，顺着往下补一句——既省力，又不容易跑偏。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>把投机解码看成一场赌注：每一次目标模型前向都很贵，我们希望「一次前向换回尽量多的已确认 token」。第43课用链把期望收益从 1 提到 τ；EAGLE 在<strong>不增加目标前向次数</strong>的前提下，用两招继续抬高 τ：<strong>特征级起草</strong>让草稿更准（每个分支更可能被接受），<strong>树状草稿 + 树注意力</strong>让一次验证同时考察很多条候选路径（接受到更长的那条）。两者叠加，<span class="mono">accept_length</span> 显著上升，而每个 token 的目标算力摊销下降。这就是为什么从 <span class="mono">EAGLE</span> 到 <span class="mono">EAGLE3</span>，再到 <span class="mono">DFLASH</span>、<span class="mono">FROZEN_KV_MTP</span>、<span class="mono">STANDALONE</span>、<span class="mono">NGRAM</span> 这一整列 <span class="mono">SpeculativeAlgorithm</span>，都是围绕「让草稿更准、让一次验证覆盖更多」这条主线在演化。值得强调的是，这一切都建立在第43课那条无损保证之上：树注意力只是改变了「怎么提议、怎么并行验证」，最终的接受/拒绝判据依旧与目标采样数学等价，所以无论树有多复杂，输出分布都和普通自回归一模一样，绝不会因为追求速度而牺牲质量。</p>
</div>

<div class="fig">
  <svg viewBox="0 0 780 320" role="img" aria-label="EAGLE 候选 token 树：根是上一个真实 token，向 topk 个子分支展开，每个子再生孙，目标一次前向验证整棵树，接受 root→A→a1→x 这条最长合法路径">
    <text x="78" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">根</text>
    <text x="235" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">topk 子</text>
    <text x="399" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">孙</text>
    <text x="600" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">更深 ✓</text>
    <line x1="126" y1="160" x2="200" y2="63" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="270" y1="63" x2="360" y2="37" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="438" y1="37" x2="540" y2="37" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="126" y1="160" x2="200" y2="160" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="126" y1="160" x2="200" y2="257" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="270" y1="63" x2="360" y2="95" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="270" y1="160" x2="360" y2="160" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="270" y1="257" x2="360" y2="257" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="30" y="138" width="96" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="78" y="156" text-anchor="middle" style="font-weight:700;fill:var(--ink);font-size:12px">root</text>
    <text x="78" y="173" text-anchor="middle" style="fill:var(--muted);font-size:10px">上一个 token</text>
    <rect x="200" y="46" width="70" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="235" y="68" text-anchor="middle" class="mono" style="font-size:12px">A ✓</text>
    <rect x="200" y="143" width="70" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="235" y="165" text-anchor="middle" class="mono" style="font-size:12px">B</text>
    <rect x="200" y="240" width="70" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="235" y="262" text-anchor="middle" class="mono" style="font-size:12px">C</text>
    <rect x="360" y="20" width="78" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="399" y="42" text-anchor="middle" class="mono" style="font-size:12px">a1 ✓</text>
    <rect x="360" y="78" width="78" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="399" y="100" text-anchor="middle" class="mono" style="font-size:12px">a2</text>
    <rect x="360" y="143" width="78" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="399" y="165" text-anchor="middle" class="mono" style="font-size:12px">b1</text>
    <rect x="360" y="240" width="78" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="399" y="262" text-anchor="middle" class="mono" style="font-size:12px">c1</text>
    <rect x="540" y="20" width="120" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="600" y="42" text-anchor="middle" class="mono" style="font-size:12px">x ✓ 接受</text>
    <text x="40" y="306" style="fill:var(--teal);font-size:12px">绿色 = 接受的最长路径 root→A→a1→x</text>
  </svg>
  <div class="figcap"><b>图 1 · EAGLE 候选 token 树</b> — 根是上一个真实 token，向 <span class="mono">topk</span> 个子分支展开，每个子再生孙，共 <span class="mono">spec_steps</span> 层；目标<strong>一次前向</strong>验证整棵树，接受 <span class="mono">root→A→a1→x</span> 这条最长合法路径，旁支被掩码隔开互不污染。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="链式草稿对比树状草稿：左边一条直链 root 到 t4，t2 错则后面全废，一次只查一条；右边一棵树同时押多条分支，更可能命中更长前缀，一次接受更多 token">
    <line x1="390" y1="30" x2="390" y2="270" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="24" y="42" style="font-weight:700;fill:var(--muted)">链式草稿</text>
    <line x1="82" y1="121" x2="92" y2="121" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="150" y1="121" x2="160" y2="121" style="stroke:var(--amber);stroke-width:3"/>
    <line x1="218" y1="121" x2="228" y2="121" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="286" y1="121" x2="296" y2="121" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 4"/>
    <rect x="24" y="104" width="58" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="53" y="126" text-anchor="middle" class="mono" style="font-size:12px">root</text>
    <rect x="92" y="104" width="58" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="121" y="126" text-anchor="middle" class="mono" style="font-size:12px">t1 ✓</text>
    <rect x="160" y="104" width="58" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="189" y="126" text-anchor="middle" class="mono" style="font-size:12px">t2 ✗</text>
    <rect x="228" y="104" width="58" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="257" y="126" text-anchor="middle" class="mono" style="font-size:12px">t3</text>
    <rect x="296" y="104" width="58" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="325" y="126" text-anchor="middle" class="mono" style="font-size:12px">t4</text>
    <text x="24" y="182" style="fill:var(--amber);font-size:12px">t2 错 → t3/t4 全废</text>
    <text x="24" y="206" style="fill:var(--muted);font-size:12px">一次前向只查 1 条续写</text>
    <text x="410" y="42" style="font-weight:700;fill:var(--accent-ink)">树状草稿</text>
    <line x1="468" y1="121" x2="520" y2="71" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="468" y1="121" x2="520" y2="121" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="468" y1="121" x2="520" y2="171" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="576" y1="71" x2="636" y2="57" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="576" y1="71" x2="636" y2="109" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="410" y="104" width="58" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="439" y="126" text-anchor="middle" class="mono" style="font-size:12px">root</text>
    <rect x="520" y="54" width="56" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="548" y="76" text-anchor="middle" class="mono" style="font-size:12px">A ✓</text>
    <rect x="520" y="104" width="56" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="548" y="126" text-anchor="middle" class="mono" style="font-size:12px">B</text>
    <rect x="520" y="154" width="56" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="548" y="176" text-anchor="middle" class="mono" style="font-size:12px">C</text>
    <rect x="636" y="40" width="56" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="664" y="62" text-anchor="middle" class="mono" style="font-size:12px">a1 ✓</text>
    <rect x="636" y="92" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="664" y="114" text-anchor="middle" class="mono" style="font-size:12px">a2</text>
    <text x="410" y="214" style="fill:var(--teal);font-size:12px">多分支 → 命中更长前缀</text>
    <text x="410" y="238" style="fill:var(--muted);font-size:12px">一次前向接受更多 token</text>
  </svg>
  <div class="figcap"><b>图 2 · 链式草稿 vs 树状草稿</b> — 左边一条直链：<span class="mono">t2</span> 一错，<span class="mono">t3/t4</span> 全废，一次前向只考察 1 条续写；右边一棵树同时押 <span class="mono">topk</span> 条分支，更长前缀更可能命中，于是同样一次目标前向能接受更多 token。</div>
</div>

<h2>一、从「链」到「树」：为什么 EAGLE 不再浪费</h2>
<p>朴素草稿模型一次提议一条<strong>链</strong>：token #1 → #2 → … → #k。验证时，只要中途某个 token 和目标分布不符，从那里往后全部丢弃。问题是，语言天生有分叉：在很多位置上，「下一个词」本来就有好几个都说得通。链只押注其中一个，赌错就血本无归。EAGLE 的核心洞察是——既然一次目标前向已经付了钱，何不让它同时验证<strong>多条</strong>候选？于是 EAGLE 在 <span class="mono">spec_steps</span> 个草稿步里，每一步都保留 <span class="mono">topk</span> 个候选分支，把它们组织成一棵共有 <span class="mono">draft_token_num</span> 个节点的<strong>token 树</strong>。根到任意节点的每一条路径，都是一段合法的候选续写。</p>
<p>这里要特别理解链的「连锁失效」是多么浪费。假设草稿一次提议 5 个 token，单个 token 被接受的概率是 0.8，那么整条链全中的概率只有 0.8 的五次方，约等于三成出头；一旦第二个 token 被否，第三到第五个不论多么合理都只能丢弃。换句话说，链的期望接受长度被「最早一个错误」死死卡住。树则把赌注分散开：当第二步同时押了好几个分支，哪怕主分支错了，旁边的兄弟分支仍有机会延续下去，于是「最早一个错误」不再是全局的死刑，而只是砍掉一棵子树。覆盖面变大，期望接受长度自然水涨船高。这就是从链到树最根本的收益来源——用一次本就要付费的目标前向，换回对更多「合理后续」的并行考察。</p>
<p>需要厘清的一点是：树并不是「无脑加宽」。<span class="mono">topk</span> 太大、<span class="mono">spec_steps</span> 太深，<span class="mono">draft_token_num</span> 会迅速膨胀，目标前向要同时处理的节点变多，单次前向本身也会变慢；而且越靠近叶子的分支命中率越低，收益递减。所以实际部署里，树的形状是一个需要按模型和负载调优的超参数：既要让树足够宽以覆盖语言的分叉，又不能宽到让验证本身成为新的瓶颈。理解了这个张力，你才能解释为什么不同任务、不同模型会选不同的 <span class="mono">topk</span> 与 <span class="mono">spec_steps</span> 组合，而不是一味把树堆大。</p>

<h2>二、特征级起草：借老师的隐藏状态</h2>
<p>为什么链能换成树，是因为我们换了「起草」的方式。先看 EAGLE 在起草端的第二个关键改进。</p>
<p>EAGLE 不再训练一个独立的完整草稿模型，而是<strong>在特征层面</strong>续写。具体来说，它复用目标模型最后一层的<strong>隐藏状态</strong>（第8课讲的那个高维向量），再结合上一步已经采样出来的 token，去预测「下一个特征」。因为输入线索直接来自目标模型本身，草稿头只需要很小的一点参数，就能和目标模型保持高度对齐——对齐度高，意味着草稿被接受的概率（α）天然更高。这就是 EAGLE 名字里「特征级自回归」的含义：在特征空间里做下一步预测，而不是从头跑一个语言模型。</p>
<p>为什么「在特征空间预测」比「在 token 空间另起一个模型」更划算？因为目标模型的隐藏状态已经凝聚了它对上下文的全部理解，是一份信息量极高的「中间答案」。独立草稿模型必须从离散的 token 重新把这份理解学一遍，既费参数又容易和目标模型产生系统性偏差；而 EAGLE 直接站在目标模型的肩膀上，从隐藏状态出发只补一小步外推，自然又快又准。这种「借力」也解释了为什么 EAGLE 的草稿头训练成本低、推理开销小：它要做的不是从零理解语言，而是预测目标模型在下一步「大概会想成什么样的特征」。线索越贴近目标，分支命中率就越高，整棵树里被接受的路径也就越长。</p>

<h2>三、树注意力：一次前向验证整棵树</h2>
<p>有了树，怎么用一次目标前向把它整棵验证完？关键是<strong>树注意力（tree attention）</strong>。先回忆普通的因果注意力：第 t 个 token 只能看见它前面的所有 token，形成一条直线上的可见关系。树注意力把这条直线推广到树：每个节点能看见的，正好是从根到它这一路的所有祖先，而所有不在这条路径上的旁支节点全部被掩码挡住。</p>
<p>EAGLE 构造一个 <span class="mono">custom_mask</span>，让每个节点只能「看见」它的<strong>祖先</strong>，看不到旁支。这样，每一条「根 → 节点」的路径在注意力里都表现得就像一段正常的连续序列，被独立打分，互不串味。树的结构由三组索引编码：<span class="mono">retrieve_index</span> 把扁平位置映射回树节点，<span class="mono">retrieve_next_token</span> 记录某条分支往下走的孩子，<span class="mono">retrieve_next_sibling</span> 记录<strong>兄弟</strong>链接——正是这组兄弟链接，让结构成为一棵<strong>树</strong>而非一条链。验证时沿树游走，接受最长的合法路径，再补上 <span class="mono">bonus_token</span>；注意 <span class="mono">accept_length</span> 是<strong>含 bonus</strong>计数的。真正在 GPU 上跑这张树掩码核函数的，是第33课讲的注意力<strong>后端</strong>。</p>
<p>为什么必须是「只看祖先」这样一张特制掩码？因为如果让兄弟节点彼此可见，那两条本该独立的候选续写就会互相污染：B 分支的 token 会错误地进入 A 分支的注意力上下文，打出来的分数也就不再代表「沿 A 这条路往下走」的真实概率。只有严格遮住旁支、只放行祖先，每条根到节点的路径才能被当成一段干净的普通序列来评分，验证结果才和「逐条链分别验证」数学等价——但代价却只有一次目标前向。把这些扁平排布的节点重新组织回树形、并在接受阶段沿着孩子与兄弟指针正确游走，靠的就是 <span class="mono">retrieve_index</span>、<span class="mono">retrieve_next_token</span> 和 <span class="mono">retrieve_next_sibling</span> 这三组索引的配合。可以说，<span class="mono">custom_mask</span> 负责「在一次前向里把树压平了算」，而这三组检索索引负责「算完之后把结果还原回树、找出最长合法路径」。</p>

<h2>四、下一代家族：EAGLE3 与更远</h2>
<p>把起草端（特征级）和验证端（树注意力）这两块拼起来，EAGLE 的完整收益就清楚了：草稿更对齐让每个分支更可能命中，树状结构让一次验证覆盖更多分支，二者相乘把接受长度推得更高。明白了这套地基，再看更新的变体就只是「在同一框架里换零件」。还有一点值得补充：EAGLE 的草稿头是要训练的，它学习的目标正是「在给定隐藏状态和上一个 token 时，目标模型下一步的特征长什么样」，因此草稿头和具体的目标模型是绑定的；换了目标模型，通常也要重新训练或适配对应的草稿头。这也解释了 <span class="mono">NGRAM</span> 这类免训练变体的存在意义——当输出高度重复时，直接用历史 n-gram 提议就能「零成本」起草，省去训练环节。</p>
<p>EAGLE3 以及更广的家族沿着同一条线继续推进：更多草稿层、多 token 预测（MTP）、n-gram 草稿等等。第43课里的 <span class="mono">SpeculativeAlgorithm</span> 枚举——<span class="mono">EAGLE</span>、<span class="mono">EAGLE3</span>、<span class="mono">DFLASH</span>、<span class="mono">FROZEN_KV_MTP</span>、<span class="mono">STANDALONE</span>、<span class="mono">NGRAM</span>——就是这条演化路线的不同取舍。无论哪一种，最终都要靠注意力后端把树掩码真正算出来，而衡量好坏的尺子始终是同一把：在相同目标前向次数下，能不能接受到更长的路径。命名上记牢：用 <span class="mono">accept</span> / <span class="mono">bonus_token</span>，绝不写成 <span class="mono">accepted</span> / <span class="mono">verified_id</span>；<span class="mono">accept_length</span> 永远含 bonus。</p>
<p>这些变体可以理解为在「草稿要多准」和「草稿要多便宜」之间的不同权衡。<span class="mono">EAGLE3</span> 用更深的草稿结构和更强的特征融合换取更高的接受率；<span class="mono">FROZEN_KV_MTP</span> 思路是冻结部分 KV、用多 token 预测一次多吐几个候选；<span class="mono">NGRAM</span> 干脆不训练草稿头，直接用历史 n-gram 统计来「免费」提议续写，特别适合高度重复的输出；<span class="mono">STANDALONE</span> 则允许挂一个独立的小草稿模型。它们共享同一套验证骨架：构树、用 <span class="mono">custom_mask</span> 做树注意力、沿检索索引接受最长合法路径、补 <span class="mono">bonus_token</span>。也正因为验证侧统一，SGLang 才能把「怎么出草稿」做成可插拔的枚举，而把「怎么验证」沉淀成稳定的公共逻辑。理解了 EAGLE 这一课，你就掌握了整个家族的共同地基：让草稿更对齐、让一次验证覆盖更多，最终在不增加目标算力的前提下把 <span class="mono">accept_length</span> 抬上去。</p>

<div class="layers">
<div class="layer"><strong>根 root</strong>：上一个已确认 token（树高从这里算起）</div>
<div class="layer"><strong>第1层（topk 个孩子）</strong>：分支A ✓ ｜ 分支B ｜ 分支C —— 每个都是一个候选 next token</div>
<div class="layer"><strong>第2层（孙子）</strong>：A→a1 ✓ ｜ A→a2 ｜ B→b1 ｜ C→c1 —— 兄弟之间互不可见</div>
<div class="layer"><strong>第3层</strong>：A→a1→x ✓ —— <span class="mono">绿色 ✓ 路径 root→A→a1→x</span> 被接受为最长合法路径，末端再补 bonus_token</div>
</div>

<div class="cols">
<div class="col"><strong>链式草稿（朴素）</strong><br>root → t1 → t2 → t3 → t4<br>只押一条路；t2 若错，t3/t4 全废。<br>一次目标前向只考察<strong>1 条</strong>续写，期望接受长度受限于单条命中率。</div>
<div class="col"><strong>树状草稿（EAGLE）</strong><br>root 同时分叉出 topk 个分支，逐层成树。<br>一次目标前向<strong>并行</strong>考察很多条续写，接受其中最长合法的一条。<br>覆盖更多「合理后续」→ 同样的前向次数换来更高的 <span class="mono">accept_length</span>。</div>
</div>

<table class="t">
<tr><th>EagleVerifyInput 字段</th><th>它编码了什么</th></tr>
<tr><td><span class="mono">draft_token</span></td><td>整棵树所有节点的候选 token，扁平排列</td></tr>
<tr><td><span class="mono">custom_mask</span></td><td>树注意力掩码：每个节点只能看见自己的祖先</td></tr>
<tr><td><span class="mono">retrieve_index</span></td><td>扁平位置 → 对应的树节点</td></tr>
<tr><td><span class="mono">retrieve_next_token</span></td><td>分支往下走的孩子链接（下一个 token）</td></tr>
<tr><td><span class="mono">retrieve_next_sibling</span></td><td>兄弟链接 —— 让结构成为「树」而非「链」</td></tr>
<tr><td><span class="mono">spec_steps</span></td><td>草稿深度（树高）</td></tr>
<tr><td><span class="mono">topk</span></td><td>每一步保留的分支数</td></tr>
<tr><td><span class="mono">draft_token_num</span></td><td>一次目标前向验证的树节点总数</td></tr>
</table>

<div class="flow">
<div class="node">复用目标模型隐藏状态 + 上一个 token</div>
<div class="arrow">→</div>
<div class="node">草稿头逐步构建 topk 树（spec_steps 层）</div>
<div class="arrow">→</div>
<div class="node">树注意力一次前向验证整棵树</div>
<div class="arrow">→</div>
<div class="node">接受最长合法路径 + bonus_token</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/eagle_info.py ::EagleVerifyInput</span><span class="ln">树状草稿 + 树注意力：一次前向验证整棵候选树</span></div><pre>class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor          # all tree nodes' candidate tokens, flattened
    custom_mask: torch.Tensor          # tree-attention mask: each node sees only its ancestors
    retrieve_index: torch.Tensor       # flat position -&gt; tree node
    retrieve_next_token: torch.Tensor  # tree child links (next token down a branch)
    retrieve_next_sibling: torch.Tensor  # tree sibling links -&gt; this is what makes it a TREE, not a chain
    spec_steps: int                    # draft depth (tree height)
    topk: int                          # branches kept per step
    draft_token_num: int               # total tree nodes verified in one target forward</pre></div>

<p>举个具体例子：<span class="mono">--speculative-eagle-topk 8 --speculative-num-steps 5 --speculative-num-draft-tokens 64</span> 会构建一棵<strong>深度 5、每层 top-8</strong> 的树，共 <span class="mono">64</span> 个候选 token，由目标模型<strong>一次前向</strong>全部验证。注意 EAGLE 在<strong>特征空间</strong>起草，所以它的草稿头很便宜——只续写一个特征向量，而不是从头跑一个完整语言模型。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/eagle_worker_v2.py ::EAGLEWorkerV2</span><span class="ln">EAGLE 工作器：草稿头展开 token 树，target 一次验证</span></div><pre>class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(self, server_args, ..., target_worker):
        self.topk = server_args.speculative_eagle_topk          # branches/level
        self.speculative_num_steps = server_args.speculative_num_steps   # tree depth
        self.speculative_num_draft_tokens = \
            server_args.speculative_num_draft_tokens             # nodes to verify

    # 起草委托给 self.draft_worker（一个 EagleDraftWorker）：
    #   self.draft_worker.draft(batch) 用 EAGLE 头展开候选 token 树

    def verify(self, batch):
        ...   # target scores the whole tree in ONE forward, accept a path</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li>EAGLE 在<strong>特征级</strong>起草：复用目标模型的<span class="mono">隐藏状态</span>（第8课）+ 上一个采样 token 预测下一个特征，草稿头很小且与目标天然对齐，<span class="mono">accept_rate</span>（α）更高。它站在目标模型的肩膀上做一小步外推，而非从零理解语言。</li>
<li>EAGLE 提议一棵<strong>token 树</strong>而非链：在 <span class="mono">spec_steps</span> 个草稿步里每步保留 <span class="mono">topk</span> 分支，共 <span class="mono">draft_token_num</span> 个节点。链会因「最早一个错误」连锁失效，树则把赌注分散到多条分支，覆盖更多合理后续。</li>
<li><strong>树注意力</strong>用 <span class="mono">custom_mask</span> 让每个节点只看祖先（遮住旁支以免污染）；<span class="mono">retrieve_index</span> / <span class="mono">retrieve_next_token</span> / <span class="mono">retrieve_next_sibling</span> 编码树，其中<strong>兄弟链接</strong>正是「树而非链」的关键。</li>
<li>一次目标前向验证整棵树，接受最长合法路径 + <span class="mono">bonus_token</span>；树覆盖更多续写 → 同样前向次数下 <span class="mono">accept_length</span> 更高（含 bonus）。判据仍与目标采样等价，因此<strong>无损</strong>。</li>
<li><span class="mono">EAGLE3</span> 及家族（<span class="mono">DFLASH</span>、<span class="mono">FROZEN_KV_MTP</span>、<span class="mono">STANDALONE</span>、<span class="mono">NGRAM</span>）继续推进，是「草稿多准」与「草稿多便宜」之间的不同权衡；它们共享同一套验证骨架。真正跑树掩码核的是第33课的注意力<strong>后端</strong>。命名用 <span class="mono">accept</span>/<span class="mono">bonus_token</span>，不写 <span class="mono">accepted</span>/<span class="mono">verified_id</span>。</li>
</ul></div>
""", "en": r"""
<p class="lead">In Lesson 43 we learned "draft + verify": a small model guesses <span class="mono">k</span> tokens, the target model verifies them all in one forward, and the accept rate <span class="mono">accept_rate</span> (α) and accept length <span class="mono">accept_length</span> (τ) set the speedup, with a <span class="mono">bonus_token</span> tacked on at the end. But naive drafting has two weaknesses: it can only guess one "chain", and the draft model is itself a whole second model. <strong>EAGLE</strong> does both smarter—it drafts at the <strong>feature level</strong>, and it proposes a <strong>tree</strong> instead of a chain. This lesson unpacks EAGLE and the next-gen family behind it.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Imagine a word-chain guessing game. <strong>Naive drafting (a chain)</strong> is like reciting five characters in one breath with your eyes shut: if the second one is wrong, the last three are all wasted. <strong>EAGLE's tree drafting</strong> instead prepares the <span class="mono">topk</span> most likely branches at every step—"this step could be A, or B, or C"—and weaves all those possibilities into a branching tree. The teacher (target model) glances at the whole tree once and ticks off the longest correct path. Even if one branch is wrong, its <strong>sibling</strong> branches survive, so nothing collapses entirely. A tree covers far more "continuations" than a single chain, so one verification accepts more tokens.</p>
<p>Also, naive drafting must raise a full "second student" to guess words, which is costly. EAGLE is clever: it does not start from scratch but borrows the <strong>hidden state</strong> the teacher just computed (Lesson 8) as the clue, attaching only a tiny "draft head" to write the next feature. The clue comes from the teacher itself, so the draft is naturally aligned with the teacher—fast and accurate.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Think of speculative decoding as a bet: every target forward is expensive, and we want "as many confirmed tokens as possible per forward". Lesson 43 raised the expected payoff from 1 to τ using a chain; EAGLE pushes τ higher <strong>without adding target forwards</strong> via two moves: <strong>feature-level drafting</strong> makes the draft more accurate (each branch is more likely accepted), and <strong>tree drafting + tree attention</strong> lets one verification examine many candidate paths at once (accepting the longest). Together, <span class="mono">accept_length</span> rises noticeably while per-token target compute amortizes down. That is why the whole list of <span class="mono">SpeculativeAlgorithm</span>—from <span class="mono">EAGLE</span> to <span class="mono">EAGLE3</span>, then <span class="mono">DFLASH</span>, <span class="mono">FROZEN_KV_MTP</span>, <span class="mono">STANDALONE</span>, <span class="mono">NGRAM</span>—evolves along the same line: make the draft more accurate, and cover more per verification.</p>
</div>

<div class="fig">
  <svg viewBox="0 0 780 320" role="img" aria-label="EAGLE candidate-token tree: the root is the last real token, branching into topk children, each into grandchildren; the target verifies the whole tree in one forward and accepts the longest valid path root to A to a1 to x">
    <text x="78" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">root</text>
    <text x="235" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">children</text>
    <text x="399" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">grandkids</text>
    <text x="600" y="16" text-anchor="middle" style="fill:var(--muted);font-size:12px">deeper ✓</text>
    <line x1="126" y1="160" x2="200" y2="63" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="270" y1="63" x2="360" y2="37" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="438" y1="37" x2="540" y2="37" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="126" y1="160" x2="200" y2="160" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="126" y1="160" x2="200" y2="257" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="270" y1="63" x2="360" y2="95" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="270" y1="160" x2="360" y2="160" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="270" y1="257" x2="360" y2="257" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="30" y="138" width="96" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="78" y="156" text-anchor="middle" style="font-weight:700;fill:var(--ink);font-size:12px">root</text>
    <text x="78" y="173" text-anchor="middle" style="fill:var(--muted);font-size:10px">last token</text>
    <rect x="200" y="46" width="70" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="235" y="68" text-anchor="middle" class="mono" style="font-size:12px">A ✓</text>
    <rect x="200" y="143" width="70" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="235" y="165" text-anchor="middle" class="mono" style="font-size:12px">B</text>
    <rect x="200" y="240" width="70" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="235" y="262" text-anchor="middle" class="mono" style="font-size:12px">C</text>
    <rect x="360" y="20" width="78" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="399" y="42" text-anchor="middle" class="mono" style="font-size:12px">a1 ✓</text>
    <rect x="360" y="78" width="78" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="399" y="100" text-anchor="middle" class="mono" style="font-size:12px">a2</text>
    <rect x="360" y="143" width="78" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="399" y="165" text-anchor="middle" class="mono" style="font-size:12px">b1</text>
    <rect x="360" y="240" width="78" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="399" y="262" text-anchor="middle" class="mono" style="font-size:12px">c1</text>
    <rect x="540" y="20" width="120" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="600" y="42" text-anchor="middle" class="mono" style="font-size:12px">x ✓ accept</text>
    <text x="40" y="306" style="fill:var(--teal);font-size:12px">teal = accepted longest path root→A→a1→x</text>
  </svg>
  <div class="figcap"><b>Fig 1 · EAGLE candidate-token tree</b> — the root is the last real token, branching into <span class="mono">topk</span> children, each into grandchildren for <span class="mono">spec_steps</span> levels; the target verifies the whole tree in <strong>one forward</strong> and accepts the longest valid path <span class="mono">root→A→a1→x</span>, with side branches masked off so they cannot contaminate each other.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Chain draft vs tree draft: on the left a single chain root to t4, where a wrong t2 wastes the rest and one forward checks just one path; on the right a tree bets on many branches at once, more likely to match a longer prefix and accept more tokens per step">
    <line x1="390" y1="30" x2="390" y2="270" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="24" y="42" style="font-weight:700;fill:var(--muted)">Chain draft</text>
    <line x1="82" y1="121" x2="92" y2="121" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="150" y1="121" x2="160" y2="121" style="stroke:var(--amber);stroke-width:3"/>
    <line x1="218" y1="121" x2="228" y2="121" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="286" y1="121" x2="296" y2="121" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 4"/>
    <rect x="24" y="104" width="58" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="53" y="126" text-anchor="middle" class="mono" style="font-size:12px">root</text>
    <rect x="92" y="104" width="58" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="121" y="126" text-anchor="middle" class="mono" style="font-size:12px">t1 ✓</text>
    <rect x="160" y="104" width="58" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="189" y="126" text-anchor="middle" class="mono" style="font-size:12px">t2 ✗</text>
    <rect x="228" y="104" width="58" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="257" y="126" text-anchor="middle" class="mono" style="font-size:12px">t3</text>
    <rect x="296" y="104" width="58" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="325" y="126" text-anchor="middle" class="mono" style="font-size:12px">t4</text>
    <text x="24" y="182" style="fill:var(--amber);font-size:12px">t2 wrong → t3/t4 wasted</text>
    <text x="24" y="206" style="fill:var(--muted);font-size:12px">1 path per forward</text>
    <text x="410" y="42" style="font-weight:700;fill:var(--accent-ink)">Tree draft</text>
    <line x1="468" y1="121" x2="520" y2="71" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="468" y1="121" x2="520" y2="121" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="468" y1="121" x2="520" y2="171" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="576" y1="71" x2="636" y2="57" style="stroke:var(--teal);stroke-width:3"/>
    <line x1="576" y1="71" x2="636" y2="109" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="410" y="104" width="58" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="439" y="126" text-anchor="middle" class="mono" style="font-size:12px">root</text>
    <rect x="520" y="54" width="56" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="548" y="76" text-anchor="middle" class="mono" style="font-size:12px">A ✓</text>
    <rect x="520" y="104" width="56" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="548" y="126" text-anchor="middle" class="mono" style="font-size:12px">B</text>
    <rect x="520" y="154" width="56" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="548" y="176" text-anchor="middle" class="mono" style="font-size:12px">C</text>
    <rect x="636" y="40" width="56" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="664" y="62" text-anchor="middle" class="mono" style="font-size:12px">a1 ✓</text>
    <rect x="636" y="92" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="664" y="114" text-anchor="middle" class="mono" style="font-size:12px">a2</text>
    <text x="410" y="214" style="fill:var(--teal);font-size:12px">more branches → longer prefix hit</text>
    <text x="410" y="238" style="fill:var(--muted);font-size:12px">more tokens accepted / forward</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Chain draft vs tree draft</b> — left: a single chain where one wrong <span class="mono">t2</span> wastes <span class="mono">t3/t4</span>, so one forward examines just 1 continuation; right: a tree bets on <span class="mono">topk</span> branches at once, so a longer prefix is more likely to match and the same target forward accepts more tokens.</div>
</div>

<h2>1. From "chain" to "tree": why EAGLE stops wasting</h2>
<p>A naive draft model proposes one <strong>chain</strong> per round: token #1 → #2 → … → #k. During verification, as soon as some token disagrees with the target distribution, everything after it is discarded. The problem: language naturally branches—at many positions several "next words" are all plausible. A chain bets on just one and loses everything when wrong. EAGLE's key insight is—since a target forward is already paid for, why not verify <strong>many</strong> candidates at once? So across <span class="mono">spec_steps</span> draft steps, EAGLE keeps <span class="mono">topk</span> candidate branches at each step, organizing them into a <strong>token tree</strong> of <span class="mono">draft_token_num</span> nodes. Every root→node path is a valid candidate continuation.</p>

<h2>2. Feature-level drafting: borrowing the teacher's hidden state</h2>
<p>EAGLE no longer trains a separate full draft model; it continues <strong>at the feature level</strong>. Concretely, it reuses the target model's last-layer <strong>hidden state</strong> (that high-dimensional vector from Lesson 8), combined with the previously sampled token, to predict the "next feature". Because the input clue comes directly from the target model itself, the draft head needs only a tiny amount of parameters to stay highly aligned with the target—high alignment means the draft's acceptance probability (α) is naturally higher. That is what "feature-level autoregression" in EAGLE's name means: predicting the next step in feature space, not running a language model from scratch.</p>

<h2>3. Tree attention: verify the whole tree in one forward</h2>
<p>With a tree, how do we verify it all in a single target forward? The key is <strong>tree attention</strong>. EAGLE builds a <span class="mono">custom_mask</span> so each node can only "see" its <strong>ancestors</strong>, not side branches. This way every "root → node" path behaves like a normal contiguous sequence in attention, scored independently without cross-contamination. The tree structure is encoded by three index groups: <span class="mono">retrieve_index</span> maps flat positions back to tree nodes, <span class="mono">retrieve_next_token</span> records the child going down a branch, and <span class="mono">retrieve_next_sibling</span> records <strong>sibling</strong> links—exactly these sibling links make the structure a <strong>tree</strong> rather than a chain. Verification walks the tree, accepts the longest valid path, then appends the <span class="mono">bonus_token</span>; note <span class="mono">accept_length</span> is counted <strong>including the bonus</strong>. The one actually running this tree-mask kernel on the GPU is the attention <strong>backend</strong> from Lesson 33.</p>

<h2>4. The next-gen family: EAGLE3 and beyond</h2>
<p>EAGLE3 and the wider family push along the same line: more draft depth, multi-token prediction (MTP), n-gram drafts, and more. The <span class="mono">SpeculativeAlgorithm</span> enum from Lesson 43—<span class="mono">EAGLE</span>, <span class="mono">EAGLE3</span>, <span class="mono">DFLASH</span>, <span class="mono">FROZEN_KV_MTP</span>, <span class="mono">STANDALONE</span>, <span class="mono">NGRAM</span>—is just different trade-offs along this path. Whichever one, it ultimately relies on the attention backend to actually compute the tree mask, and the yardstick is always the same: under the same number of target forwards, can we accept a longer path? On naming, remember: use <span class="mono">accept</span> / <span class="mono">bonus_token</span>, never <span class="mono">accepted</span> / <span class="mono">verified_id</span>; <span class="mono">accept_length</span> always includes the bonus.</p>

<div class="layers">
<div class="layer"><strong>root</strong>: the last confirmed token (tree height starts here)</div>
<div class="layer"><strong>Layer 1 (topk children)</strong>: branch A ✓ ｜ branch B ｜ branch C — each a candidate next token</div>
<div class="layer"><strong>Layer 2 (grandchildren)</strong>: A→a1 ✓ ｜ A→a2 ｜ B→b1 ｜ C→c1 — siblings cannot see each other</div>
<div class="layer"><strong>Layer 3</strong>: A→a1→x ✓ — <span class="mono">green ✓ path root→A→a1→x</span> is accepted as the longest valid path, then a bonus_token is appended</div>
</div>

<div class="cols">
<div class="col"><strong>Chain draft (naive)</strong><br>root → t1 → t2 → t3 → t4<br>Bets on one path only; if t2 is wrong, t3/t4 are wasted.<br>One target forward examines just <strong>1</strong> continuation, expected accept length capped by single-path hit rate.</div>
<div class="col"><strong>Tree draft (EAGLE)</strong><br>root forks into topk branches, growing layer by layer into a tree.<br>One target forward examines many continuations <strong>in parallel</strong>, accepting the longest valid one.<br>Covers more "plausible continuations" → higher <span class="mono">accept_length</span> for the same number of forwards.</div>
</div>

<table class="t">
<tr><th>EagleVerifyInput field</th><th>what it encodes</th></tr>
<tr><td><span class="mono">draft_token</span></td><td>all tree nodes' candidate tokens, flattened</td></tr>
<tr><td><span class="mono">custom_mask</span></td><td>tree-attention mask: each node sees only its ancestors</td></tr>
<tr><td><span class="mono">retrieve_index</span></td><td>flat position → tree node</td></tr>
<tr><td><span class="mono">retrieve_next_token</span></td><td>child link going down a branch (next token)</td></tr>
<tr><td><span class="mono">retrieve_next_sibling</span></td><td>sibling links — what makes it a "tree", not a chain</td></tr>
<tr><td><span class="mono">spec_steps</span></td><td>draft depth (tree height)</td></tr>
<tr><td><span class="mono">topk</span></td><td>branches kept per step</td></tr>
<tr><td><span class="mono">draft_token_num</span></td><td>total tree nodes verified in one target forward</td></tr>
</table>

<div class="flow">
<div class="node">reuse target hidden state + previous token</div>
<div class="arrow">→</div>
<div class="node">draft head builds topk tree (spec_steps levels)</div>
<div class="arrow">→</div>
<div class="node">tree attention verifies the whole tree in one forward</div>
<div class="arrow">→</div>
<div class="node">accept longest valid path + bonus_token</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/eagle_info.py ::EagleVerifyInput</span><span class="ln">a token tree + tree attention: verify the whole candidate tree in one forward</span></div><pre>class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor          # all tree nodes' candidate tokens, flattened
    custom_mask: torch.Tensor          # tree-attention mask: each node sees only its ancestors
    retrieve_index: torch.Tensor       # flat position -&gt; tree node
    retrieve_next_token: torch.Tensor  # tree child links (next token down a branch)
    retrieve_next_sibling: torch.Tensor  # tree sibling links -&gt; this is what makes it a TREE, not a chain
    spec_steps: int                    # draft depth (tree height)
    topk: int                          # branches kept per step
    draft_token_num: int               # total tree nodes verified in one target forward</pre></div>

<p>A concrete example: <span class="mono">--speculative-eagle-topk 8 --speculative-num-steps 5 --speculative-num-draft-tokens 64</span> builds a <strong>depth-5, top-8</strong> tree of <span class="mono">64</span> candidate tokens, all verified by the target in <strong>one forward</strong>. Note EAGLE drafts in <strong>feature space</strong>, so its draft head is cheap—it just continues one feature vector instead of running a full language model from scratch.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/eagle_worker_v2.py ::EAGLEWorkerV2</span><span class="ln">EAGLE worker: draft head expands a token tree, target verifies once</span></div><pre>class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(self, server_args, ..., target_worker):
        self.topk = server_args.speculative_eagle_topk          # branches/level
        self.speculative_num_steps = server_args.speculative_num_steps   # tree depth
        self.speculative_num_draft_tokens = \
            server_args.speculative_num_draft_tokens             # nodes to verify

    # drafting is delegated to self.draft_worker (an EagleDraftWorker):
    #   self.draft_worker.draft(batch)  -&gt; EAGLE head expands a TREE

    def verify(self, batch):
        ...   # target scores the whole tree in ONE forward, accept a path</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li>EAGLE drafts at the <strong>feature level</strong>: it reuses the target model's <span class="mono">hidden state</span> (Lesson 8) + the previously sampled token to predict the next feature, so the draft head is tiny and naturally aligned with the target, giving higher <span class="mono">accept_rate</span> (α).</li>
<li>EAGLE proposes a <strong>token tree</strong>, not a chain: across <span class="mono">spec_steps</span> draft steps it keeps <span class="mono">topk</span> branches per step, totaling <span class="mono">draft_token_num</span> nodes.</li>
<li><strong>Tree attention</strong> uses <span class="mono">custom_mask</span> so each node sees only its ancestors; <span class="mono">retrieve_index</span> / <span class="mono">retrieve_next_token</span> / <span class="mono">retrieve_next_sibling</span> encode the tree, where the <strong>sibling links</strong> are exactly what makes it a tree rather than a chain.</li>
<li>One target forward verifies the whole tree, accepting the longest valid path + <span class="mono">bonus_token</span>; a tree covers more continuations → higher <span class="mono">accept_length</span> (including the bonus) for the same number of forwards.</li>
<li><span class="mono">EAGLE3</span> and the family (<span class="mono">DFLASH</span>, <span class="mono">FROZEN_KV_MTP</span>, <span class="mono">STANDALONE</span>, <span class="mono">NGRAM</span>) push further; the one actually running the tree-mask kernel is the attention <strong>backend</strong> from Lesson 33. Use <span class="mono">accept</span>/<span class="mono">bonus_token</span>, not <span class="mono">accepted</span>/<span class="mono">verified_id</span>.</li>
</ul></div>
"""}
LESSON_45 = {"zh": r"""
<p class="lead">一个请求其实有两副"面孔"：<strong>prefill（预填充）</strong>把整段提示词一次性吃进去，<strong>decode（解码）</strong>一个 token 一个 token 地往外吐。它们对硬件的胃口完全相反——一个吃<span class="mono">算力</span>，一个吃<span class="mono">带宽</span>。把它们绑在同一张 GPU 上，就像让短跑选手和马拉松选手共用一条跑道，互相拖后腿。本课讲 SGLang 的 <strong>PD 分离（Prefill–decode disaggregation）</strong>：把这两个阶段拆到不同的 GPU 池里，各自吃饱自己的瓶颈。</p>
<p>这件事为什么值得单独一课？因为它是当下大规模在线推理<strong>能不能既快又省</strong>的关键开关之一。把两副面孔强行塞在一张卡上，你永远要在 TTFT 和 ITL 之间二选一地妥协；而一旦分池，两个指标就能<strong>各自被优化到位</strong>。读完本课，你应能说清三件事：为什么要分、分了之后被搬运的到底是什么、以及 SGLang 用什么接口把这次搬运抽象成可插拔的组件。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一家<strong>餐厅</strong>。<span class="mono">备料区</span>（prefill）要把一整筐食材在短时间内集中处理完——这是一阵<strong>爆发式的重体力活</strong>，需要大量人手同时挥刀切菜（算力密集）。<span class="mono">出餐区</span>（decode）则是一道一道地把菜端给客人，每道之间要不停跑回厨房取调料、看菜谱（反复读取权重与 KV），人手大多在<strong>等待和搬运</strong>，刀工反而闲着（带宽密集）。</p>
<p>如果让<strong>同一批厨师</strong>既备料又出餐，那么每当来了一桌大单（长提示词），所有厨师都被拉去切菜，正在等菜的客人就只能干等——这就是长 prefill 把大家的 decode 延迟<strong>顶停</strong>的画面。PD 分离的做法是：<strong>专门一组人只备料，另一组人只出餐</strong>，备好的半成品（KV 缓存）用<span class="mono">传送带</span>快速送过去。两组人各司其职，谁都不被对方拖慢。</p>
<p>这个类比还能再延伸一层：备料区和出餐区<strong>人数可以分开配</strong>。中午大量外卖订单涌入（长 prompt 高峰），就多派人去备料；晚上堂食客人慢慢吃、要的菜多（长输出），就多派人去出餐。两个区的人手比例随客流灵活调整，互不牵制——这正对应 prefill 池与 decode 池可以按流量画像独立伸缩的运维自由。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>回忆<strong>第4课</strong>：prefill 是把整个 prompt 在<strong>一次大并行</strong>里算完，GPU 的矩阵乘单元被打满，属于<strong>compute-bound（算力受限）</strong>；decode 每步只生成一个 token，却要把<strong>全部权重 + KV</strong> 重新读一遍，算力单元大半空转，属于<strong>bandwidth-bound（带宽受限）</strong>。两者放同一张卡上必然抢资源：一段长 prefill 一插进来，正在解码的请求 ITL（token 间延迟）就被拉高，这正是 <strong>TTFT vs ITL</strong> 的张力（<strong>第8课 / 第22课</strong>的 chunked prefill 能缓解但消不掉）。</p>
<p>PD 分离从<strong>物理上</strong>把两个阶段拆开：一个 <strong>prefill 池</strong>只做预填充，一个 <strong>decode 池</strong>只做解码；prefill 完成后把请求的 <strong>KV 缓存</strong>（<strong>第30课</strong>的分页 KV）经 RDMA / NVLink 这类高速互联<strong>搬运</strong>到 decode GPU，由后者流式吐 token。每个池可以<strong>独立调整规模与参数</strong>，各自榨干自己的瓶颈。前瞻<strong>第46 / 47课</strong>：分离 + 大规模 EP，正是 DeepSeek 级别服务的搭法。</p>
<p>换个角度看，PD 分离其实是把"<strong>专业化分工</strong>"这一工程通则用到了推理硬件上。就像现代工厂不会让一个工人既当车床又当质检，而是按工序拆成专门的工位、各自配最合适的设备；推理服务也不该让一张 GPU 既扛算力密集的 prefill 又扛带宽密集的 decode。分工的代价是工位之间多了一道<strong>传递</strong>（KV 搬运），收益是每个工位都能被打磨到极致利用率。能不能从分工中获益，取决于<strong>传递成本是否足够低</strong>——这把我们引向了它对高速互联的硬性依赖。</p>
</div>

<h2>一、为什么要分？两副面孔的资源冲突</h2>
<p>一个请求的生命周期里，<strong>prefill</strong> 与 <strong>decode</strong> 对 GPU 的需求是<strong>镜像相反</strong>的。Prefill 把长度为 N 的提示词当成一个大批次同时送进网络，所有 token 并行计算，矩阵乘法把 GPU 的 <span class="mono">Tensor Core</span> 喂得满满当当——它<strong>缺的是算力</strong>，带宽绰绰有余。Decode 反过来：每一步只有 1 个新 token，计算量极小，但每生成一个 token 都要把<strong>整套模型权重</strong>和<strong>已积累的 KV 缓存</strong>从显存里重新读出来，瓶颈完全卡在<strong>显存带宽</strong>上，算力单元大量闲置。</p>
<p>这种"一忙一闲"的错位不是偶尔出现，而是<strong>贯穿请求始终</strong>的结构性特征：只要模型是自回归生成的，prefill 与 decode 的资源画像就天然相反。也正因为它是结构性的，单纯靠调度技巧（优先级、切片、抢占）只能<strong>缓和</strong>而无法<strong>消除</strong>，真正的解法必须落到"把两类工作放到不同硬件上"这一层。这就是 PD 分离要做的事。</p>
<p>当二者共享一张 GPU，冲突就不可避免。调度器要么优先插入 prefill（拉低正在解码请求的 ITL），要么优先 decode（拖慢新请求的 TTFT）。<span class="mono">chunked prefill</span>（第22课）把长 prefill 切成小块穿插进 decode 之间，能把抖动压小，但只要两类工作还在<strong>同一批算力上排队</strong>，干扰就无法根除。PD 分离的洞见很直接：<strong>既然两副面孔吃的不是同一样东西，就别让它们抢同一张桌子</strong>。</p>
<p>把这种不匹配量化一下会更有体感。Prefill 的<strong>算术强度</strong>（每读一字节权重做多少次乘加）很高，因为一整批 token 共享同一份权重读取；decode 的算术强度则低得多，几乎每出一个 token 都要重新搬一遍权重，于是真正决定 decode 速度的不是峰值算力而是<strong>显存带宽</strong>。这意味着：给 decode 多堆算力几乎没用，给 prefill 多堆带宽也帮不上忙。当二者绑在一张卡上，无论你怎么调，总有一半硬件资源处于<strong>结构性闲置</strong>——要么算力闲，要么带宽闲。</p>
<p>更棘手的是<strong>延迟耦合</strong>。在线服务里 TTFT（首 token 延迟）由 prefill 决定，ITL（token 间延迟）由 decode 决定，二者往往是不同请求、不同用户最在意的指标。同卡共存时，一条长 prompt 的 prefill 会霸占算力若干毫秒，期间所有正在解码的请求都被<strong>挂起</strong>，ITL 出现尖刺。chunked prefill 把长 prefill 切片穿插，能把尖刺摊成小毛刺，却也因为反复打断 decode 带来额外调度开销；本质上它是在<strong>同一资源池内做时间复用</strong>，而 PD 分离直接做<strong>空间隔离</strong>，从根上消除这种抢占。</p>

<div class="flow"><div class="node">请求到达</div><div class="arrow">→</div><div class="node">prefill 池<br><span class="mono">算力密集</span></div><div class="arrow">→</div><div class="node">KV 传输<br><span class="mono">RDMA/NVLink</span></div><div class="arrow">→</div><div class="node">decode 池<br><span class="mono">带宽密集</span></div><div class="arrow">→</div><div class="node">流式吐 token</div></div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="prefill 池算好整段 KV，经 RDMA/NVLink 传到 decode 池，decode 读这份 KV 逐 token 流式输出">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">一次请求穿过两个池</text>
    <rect x="20" y="96" width="74" height="66" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="57" y="126" text-anchor="middle" style="font-size:12px">请求</text>
    <text x="57" y="146" text-anchor="middle" style="fill:var(--faint);font-size:11px">prompt</text>
    <line x1="98" y1="129" x2="126" y2="129" style="stroke:var(--line);stroke-width:2"/>
    <polygon points="128,129 118,124 118,134" style="fill:var(--line)"/>
    <rect x="132" y="90" width="150" height="78" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="207" y="116" text-anchor="middle" style="fill:var(--blue);font-weight:700">Prefill 池</text>
    <text x="207" y="136" text-anchor="middle" style="font-size:12px">一次算完整段</text>
    <text x="207" y="155" text-anchor="middle" class="mono" style="font-size:11px">→ KV 缓存</text>
    <line x1="286" y1="129" x2="314" y2="129" style="stroke:var(--amber);stroke-width:2"/>
    <polygon points="316,129 306,124 306,134" style="fill:var(--amber)"/>
    <rect x="320" y="96" width="140" height="66" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="390" y="124" text-anchor="middle" style="fill:var(--amber);font-weight:700">KV 传输</text>
    <text x="390" y="144" text-anchor="middle" class="mono" style="font-size:11px">RDMA / NVLink</text>
    <line x1="464" y1="129" x2="492" y2="129" style="stroke:var(--amber);stroke-width:2"/>
    <polygon points="494,129 484,124 484,134" style="fill:var(--amber)"/>
    <rect x="498" y="90" width="150" height="78" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="573" y="116" text-anchor="middle" style="fill:var(--teal);font-weight:700">Decode 池</text>
    <text x="573" y="136" text-anchor="middle" style="font-size:12px">逐 token 生成</text>
    <text x="573" y="155" text-anchor="middle" class="mono" style="font-size:11px">读 KV</text>
    <line x1="652" y1="129" x2="680" y2="129" style="stroke:var(--line);stroke-width:2"/>
    <polygon points="682,129 672,124 672,134" style="fill:var(--line)"/>
    <rect x="686" y="96" width="96" height="66" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="734" y="124" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">流式输出</text>
    <text x="734" y="144" text-anchor="middle" class="mono" style="font-size:11px">tokens</text>
  </svg>
  <div class="figcap"><b>图 1 · 两池流水线</b> — 请求先进 prefill 池算出整段 KV，经 RDMA/NVLink 传到 decode 池，decode 读着这份 KV 逐 token 流式输出。</div>
</div>

<h2>二、分离之后：两个独立的池</h2>
<p>PD 分离把集群切成两半。<strong>Prefill 池</strong>追求<strong>吞吐</strong>：它处理的是一阵阵爆发式的大批并行计算，适合配置高算力、可以容忍稍大的批；它的产出不是 token，而是<strong>一份算好的 KV 缓存</strong>。<strong>Decode 池</strong>追求<strong>低延迟、稳定的 token 流</strong>：它需要的是高显存带宽与小而稳的批，把每个请求的 ITL 压到最低。两个池的规模、批策略、并行方式都能<strong>各自独立调优</strong>——这是绑在一起时做不到的。</p>
<p>"独立"还体现在<strong>故障与扩缩容</strong>上。decode 池某个节点挂了，路由器可以把后续请求改派给别的 decode worker，而 prefill 池完全不受影响；反过来想临时加大 prefill 吞吐，直接往 prefill 池里加卡即可，不必动 decode。这种<strong>解耦的弹性</strong>，让在线服务在流量突变时能更平滑地应对，也让灰度升级、容量规划这些运维动作可以分池进行、互不打断。</p>
<p>关键的衔接动作是 <strong>KV 缓存的搬运</strong>。Prefill 算完后，请求对应的那些<strong>分页 KV</strong>（第30课，KV 以页为单位存放）要从 prefill GPU 的显存<strong>送到</strong> decode GPU 的显存。这一步走的是 <span class="mono">RDMA</span> 或 <span class="mono">NVLink</span> 这类高速互联，目标是让传输时间远小于它省下的重复计算。一个<strong>路由器 / 负载均衡器</strong>（第13课）负责<strong>为每个请求配对</strong>一个 prefill worker 和一个 decode worker。</p>
<p>独立调优带来一个很实用的杠杆：<strong>池子配比</strong>。如果线上流量是"长 prompt、短输出"（比如检索增强问答），prefill 压力大，就多配 prefill 节点；如果是"短 prompt、长输出"（比如长篇创作、推理链），decode 压力大，就多配 decode 节点。这个 P∶D 比例可以随流量画像<strong>弹性调整</strong>，甚至按时段伸缩——这是单卡共存模式根本给不了的运维自由度。两个池还能各自选最划算的硬件：prefill 池上算力强的卡，decode 池上带宽/显存大的卡。</p>
<p>当然，分离不是免费的午餐。它引入了一次<strong>跨节点的 KV 传输</strong>，多了网络这一环；如果互联不够快，搬运时间会反过来侵蚀掉省下的收益，甚至抬高 TTFT。所以 PD 分离的成立前提是<strong>有足够快的互联</strong>（RDMA/NVLink）以及<strong>把传输与计算尽量重叠</strong>。理解这条边界，才知道它适合大规模、互联完善的集群，而不是随便两张卡都该拆。</p>

<div class="cols"><div class="col"><strong>Prefill 节点</strong><br><span class="mono">compute-bound</span><br>一次大并行算完整段 prompt<br>打满 Tensor Core / 算力<br>产出：算好的 <strong>KV 缓存</strong></div><div class="col"><strong>Decode 节点</strong><br><span class="mono">bandwidth-bound</span><br>每步只生成 1 个 token<br>反复重读权重 + KV，算力闲置<br>产出：<strong>流式 token</strong></div></div>

<div class="fig">
  <svg viewBox="0 0 800 290" role="img" aria-label="左 prefill 节点算力受限：整段 prompt 并行、算力打满；右 decode 节点带宽受限：每步 1 token 读大 KV 缓存、算力闲置">
    <line x1="400" y1="44" x2="400" y2="272" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="30" y="32" style="font-weight:700;fill:var(--blue)">Prefill 节点</text>
    <rect x="30" y="44" width="140" height="26" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="100" y="62" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">compute-bound</text>
    <text x="30" y="92" style="font-size:12px;fill:var(--muted)">整段 prompt 并行 → 算力打满</text>
    <rect x="40" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="82" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="124" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="166" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="208" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="40" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="82" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="124" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="166" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="208" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="40" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="82" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="124" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="166" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="208" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <text x="252" y="158" style="fill:var(--blue);font-size:12px">算力全忙</text>
    <text x="30" y="258" style="font-size:12px">瓶颈：GPU 算力 / FLOPs</text>
    <text x="430" y="32" style="font-weight:700;fill:var(--teal)">Decode 节点</text>
    <rect x="430" y="44" width="160" height="26" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="510" y="62" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--teal)">bandwidth-bound</text>
    <text x="430" y="92" style="font-size:12px;fill:var(--muted)">每步 1 token → 算力闲置</text>
    <rect x="430" y="112" width="30" height="22" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.2"/>
    <rect x="472" y="112" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="514" y="112" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="430" y="142" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="472" y="142" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="514" y="142" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="430" y="172" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="472" y="172" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="514" y="172" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <line x1="598" y1="150" x2="550" y2="150" style="stroke:var(--amber);stroke-width:2.5"/>
    <polygon points="548,150 558,145 558,155" style="fill:var(--amber)"/>
    <text x="556" y="138" style="fill:var(--amber);font-size:11px">带宽满载</text>
    <rect x="600" y="108" width="120" height="92" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="660" y="148" text-anchor="middle" style="fill:var(--teal);font-weight:700">大 KV 缓存</text>
    <text x="660" y="170" text-anchor="middle" style="font-size:11px">每步全读</text>
    <text x="430" y="258" style="font-size:12px">瓶颈：显存带宽</text>
  </svg>
  <div class="figcap"><b>图 2 · 算力受限 vs 带宽受限</b> — prefill 把整段 prompt 一次并行算完、算力全忙；decode 每步只出 1 token，却要满带宽重读一大块 KV，算力大半闲置。分池后各按自己的瓶颈选硬件、定批量。</div>
</div>

<p><strong>落到部署上：</strong>启动时加 <span class="mono">--disaggregation-mode prefill</span> 与 <span class="mono">--disaggregation-mode decode</span>，就把同一份模型拆成 prefill worker 与 decode worker 两组进程。设想一个 <strong>2000 token</strong> 的长提示词：prefill 池一次把它整段算成 KV，这份 KV 只<strong>搬运一次</strong>到 decode 池；之后 decode 连续吐出几百个 token，<strong>全程不再重算 prompt</strong>。若线上是"长 prompt、短答案"，就多配 prefill 卡；若是"短 prompt、长答案"，就多配 decode 卡——P∶D 比例随流量自由伸缩。</p>

<h2>三、SGLang 怎么抽象这次搬运</h2>
<p>不同集群的互联硬件五花八门，SGLang 不把传输逻辑写死，而是藏在一个<strong>可插拔的连接器</strong>后面。Prefill 侧持有一个 <span class="mono">BaseKVSender</span>，decode 侧持有一个镜像的 <span class="mono">BaseKVReceiver</span>。发送方的接口很克制：<span class="mono">init</span> 先<strong>宣告</strong>这次要搬多少页 KV；<span class="mono">send</span> 把这个请求的 KV 页<strong>推</strong>给 decode worker；<span class="mono">poll</span> 返回一个<strong>非阻塞</strong>的 <span class="mono">KVPoll</span> 状态（Bootstrapping / WaitingForInput / Transferring / Success / Failed），让调度器不必卡死等待；<span class="mono">get_transfer_metric</span> 则吐出 <span class="mono">KVTransferMetric</span>（字节数、延迟）供观测。</p>
<p>这层抽象的好处是<strong>后端可替换</strong>：具体实现有 <span class="mono">Mooncake</span>、<span class="mono">NIXL</span>、<span class="mono">ascend</span> 等几种连接器，各自对接不同的互联栈，但上层调度看到的都是同一套 <span class="mono">init / send / poll</span> 契约。Decode 侧的 <span class="mono">BaseKVReceiver</span> 与之对称：<span class="mono">init</span> + <span class="mono">send_metadata</span> + <span class="mono">poll</span>——decode 先用 <span class="mono">send_metadata</span> 把自己这边的 KV 槽位索引<strong>通告</strong>给 prefill，prefill 据此把 KV 直接写进 decode 显存（常用单边 RDMA）；当 poll 返回 <span class="mono">Success</span>，就意味着 KV 已落地，decode 可以开吐 token。</p>
<p>为什么 <span class="mono">poll</span> 一定要<strong>非阻塞</strong>？因为调度器是单线程驱动一大批请求的状态机，它绝不能为了等某一次 KV 传输而把整批都卡住。非阻塞的 poll 让调度器每轮只<strong>瞄一眼</strong>当前状态：还在 <span class="mono">Bootstrapping</span>（建链握手）就先去伺候别的请求，变成 <span class="mono">Transferring</span> 就继续等，直到 <span class="mono">Success</span> 才把该请求推进到解码、或 <span class="mono">Failed</span> 时走重试/降级。这种"状态机 + 轮询"的设计，正是高并发服务里把慢 I/O 和快调度解耦的经典手法。</p>
<p>把发送方接口设计得如此<strong>克制</strong>（只有 init/send/poll/get_transfer_metric 四个动作）也是有意为之：接口越窄，新后端越好接。想接入一种新互联，只要实现这几个方法，让 poll 正确地把底层进度翻译成 <span class="mono">KVPoll</span> 五态即可，上层的路由、调度、配对逻辑一行都不用改。<span class="mono">get_transfer_metric</span> 返回的 <span class="mono">KVTransferMetric</span>（字节数、延迟）则喂给监控，让你能定位"是不是某条互联在拖后腿"。</p>

<table class="t"><tr><th>方法</th><th>所在侧 / 作用</th></tr><tr><td><span class="mono">init(num_kv_indices, aux_index)</span></td><td>prefill 侧：宣告本次要搬多少页 KV</td></tr><tr><td><span class="mono">send(kv_indices)</span></td><td>prefill 侧：把该请求的 KV 页推给 decode worker</td></tr><tr><td><span class="mono">poll() → KVPoll</span></td><td>双侧：非阻塞返回传输状态</td></tr><tr><td><span class="mono">get_transfer_metric()</span></td><td>prefill 侧：返回 KVTransferMetric（字节、延迟）</td></tr><tr><td><span class="mono">BaseKVReceiver.send_metadata()</span></td><td>decode 侧：把本侧 KV 槽位索引通告给 prefill（与 send 配对）</td></tr></table>

<p>把一次成功的搬运拆成时间线，就是下面这条竖向流程。注意 <span class="mono">poll</span> 在中间反复被调用：它<strong>不阻塞</strong>，每次只报告当前状态，直到 <span class="mono">Success</span>。</p>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>init</h4><p>宣告要搬 num_kv_indices 页 KV</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>send</h4><p>把请求的 KV 页推向 decode 侧</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>poll → KVPoll</h4><p>Transferring（仍在搬运，<strong>非阻塞</strong>）</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>poll → Success</h4><p>KV 已送达，decode 开始吐 token</p></div></div></div>

<h2>四、把它放回大图里</h2>
<p>PD 分离不是孤立技巧，它把前面几课串了起来：第4课告诉我们两阶段的资源画像，第8 / 22课的 TTFT/ITL 张力是它要解决的痛点，第30课的分页 KV 是被搬运的对象，第13课的路由器负责配对 worker。这一串引用不是凑数，而是说明 PD 分离本质上是建立在前面所有机制之上的<strong>系统级集成</strong>：少了任何一环都搭不起来。再往前看，<strong>第46 / 47课</strong>会把<strong>分离</strong>与<strong>大规模专家并行（EP）</strong>叠在一起——prefill 池、decode 池各自再做 EP 切分，正是 DeepSeek 级别在线服务的真实搭法。</p>
<p>理解了"两副面孔、分池而治、KV 搬运"，就握住了现代大规模推理服务的主骨架，也就理解了后续高阶课程的地基。</p>
<p>不妨把整条链路再走一遍当作复盘：请求先到<strong>路由器</strong>，被指派一个 prefill worker；prefill 把整段 prompt 一次算完，得到一份分页 KV，并通过 <span class="mono">BaseKVSender</span> 的 <span class="mono">init → send</span> 把这些页推向被配对的 decode worker；decode 侧 <span class="mono">BaseKVReceiver</span> 通过 <span class="mono">send_metadata</span> 通告 KV 落点，双方各自 <span class="mono">poll</span> 直到状态翻到 <span class="mono">Success</span>；此刻 KV 已在 decode GPU 的显存里就位，decode 接手，逐 token 流式返回给用户。整个过程里，prefill 池始终在打满算力地"备料"，decode 池始终在带宽满载地"出餐"，没有谁为对方空转。可以说，分离把"硬件利用率"这件事从单卡内部的无奈妥协，变成了整个集群层面可设计、可度量、可优化、可弹性伸缩的清晰分工。</p>
<p>最后强调一个容易被忽略的点：PD 分离改变的是<strong>物理部署</strong>，而不是模型本身的数学。同一个模型、同一份权重、同样的 KV，只是把"算 KV"和"用 KV"这两件事放到了不同的机器上，中间加了一次显式搬运。正因为它只动部署不动语义，才能作为一块<strong>可组合的积木</strong>，和量化、张量并行、专家并行（EP）等技术叠加使用——这也是为什么下一阶段（第46/47课）能把它和大规模 EP 拼在一起，搭出 DeepSeek 级别的在线推理系统。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::BaseKVSender</span><span class="ln">prefill 侧把 KV 传给 decode 侧：init → send → poll(KVPoll)</span></div><pre>class BaseKVSender(ABC):                 # lives on the PREFILL side
    @abstractmethod
    def init(self, num_kv_indices, aux_index=None):
        ...                              # announce how many KV pages will move
    @abstractmethod
    def send(self, kv_indices):
        ...                              # push this request's KV pages to the decode worker
    @abstractmethod
    def poll(self):                      # non-blocking status -&gt; KVPoll
        ...                              # Bootstrapping / WaitingForInput / Transferring / Success / Failed
    @abstractmethod
    def get_transfer_metric(self):       # -&gt; KVTransferMetric (bytes, latency)
        ...
# BaseKVReceiver mirrors this on the DECODE side (init + send_metadata + poll)</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::BaseKVReceiver</span><span class="ln">decode 侧：接收 prefill 传来的 KV，再开始解码</span></div><pre>class BaseKVReceiver(ABC):
    # the DECODE side of PD disaggregation: pull the KV that the
    # prefill node produced, into this node's KV pool.
    @abstractmethod
    def init(self, ...): ...                        # set up the receive session
    @abstractmethod
    def send_metadata(self, kv_indices, ...): ...   # announce where the KV lands
    @abstractmethod
    def poll(self) -&gt; KVPoll:
        ...   # WaitingForInput -&gt; Transferring -&gt; Success
    # once KV has arrived, the decode node generates tokens from it.</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>两副面孔，资源相反</strong>：prefill 是 compute-bound 的一次大并行；decode 是 bandwidth-bound 的逐 token 生成。同卡共存必然在 TTFT 与 ITL 之间打架。</li>
<li><strong>PD 分离 = 分池而治</strong>：prefill 池只做预填充，decode 池只做解码，各自吃饱自己的瓶颈、独立调优。</li>
<li><strong>被搬运的是 KV 缓存</strong>：prefill 算好的分页 KV（第30课）经 RDMA / NVLink 送到 decode GPU，再开始流式吐 token。</li>
<li><strong>可插拔连接器</strong>：<span class="mono">BaseKVSender</span>（init / send / poll → KVPoll / get_transfer_metric）与镜像的 <span class="mono">BaseKVReceiver</span>；后端有 Mooncake、NIXL、ascend。</li>
<li><strong>poll 非阻塞</strong>：状态在 Bootstrapping / WaitingForInput / Transferring / Success / Failed 间流转，Success 即 KV 落地、decode 起步。</li>
<li><strong>承上启下</strong>：路由器（第13课）配对 worker；前瞻第46 / 47课，分离 + 大规模 EP 是 DeepSeek 级服务的搭法。</li>
</ul></div>
""", "en": r"""
<p class="lead">A request really wears two faces: <strong>prefill</strong> swallows the entire prompt in one shot, while <strong>decode</strong> emits one token at a time. Their appetites for hardware are exact opposites—one eats <span class="mono">compute</span>, the other eats <span class="mono">bandwidth</span>. Bolting them onto the same GPU is like making a sprinter and a marathoner share one lane; they drag each other down. This lesson covers SGLang's <strong>PD disaggregation (Prefill–decode disaggregation)</strong>: split the two phases into separate GPU pools so each can saturate its own bottleneck.</p>
<p>Why does this deserve its own lesson? Because it is one of the key switches that decides whether large-scale online inference can be <strong>both fast and cheap</strong>. Cram the two faces onto one card and you forever compromise between TTFT and ITL; split them into pools and the two metrics can each be <strong>optimized properly</strong>. By the end you should be able to state three things clearly: why to split, what exactly gets transferred after the split, and what interface SGLang uses to abstract that transfer into a pluggable component.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a <strong>restaurant</strong>. The <span class="mono">prep station</span> (prefill) has to process a whole basket of ingredients in a short burst—a <strong>burst of heavy lifting</strong> that needs many hands chopping at once (compute-intensive). The <span class="mono">serving station</span> (decode) hands out dishes one by one, and between each it keeps running back to the kitchen for seasoning and to read the recipe (re-reading weights and KV); the hands are mostly <strong>waiting and carrying</strong> while the knife skills sit idle (bandwidth-intensive).</p>
<p>If the <strong>same cooks</strong> do both prep and serving, then every time a big order arrives (a long prompt) all the cooks get yanked off to chop, and the customers waiting for their next dish just sit there—this is exactly how a long prefill <strong>stalls</strong> everyone's decode latency. PD disaggregation says: <strong>one crew only preps, another crew only serves</strong>, and the half-finished work (the KV cache) is rushed across on a <span class="mono">conveyor belt</span>. Each crew does its own job and neither slows the other.</p>
<p>The analogy stretches one layer further: the prep and serving stations <strong>can be staffed separately</strong>. When a lunchtime flood of delivery orders pours in (a peak of long prompts), send more people to prep; in the evening when dine-in guests linger and order many courses (long outputs), send more people to serve. The headcount ratio of the two stations flexes with traffic, neither constraining the other—exactly mirroring the operational freedom of scaling the prefill pool and decode pool independently by traffic profile.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Recall <strong>Lesson 4</strong>: prefill computes the whole prompt in <strong>one big parallel pass</strong>, saturating the GPU's matrix-multiply units—it is <strong>compute-bound</strong>; decode produces just one token per step yet must re-read <strong>all the weights + KV</strong> every step, leaving the math units mostly idle—it is <strong>bandwidth-bound</strong>. On the same card they inevitably fight: slip in a long prefill and the ITL (inter-token latency) of decoding requests jumps. This is the <strong>TTFT vs ITL</strong> tension (<strong>Lessons 8 / 22</strong>'s chunked prefill mitigates but doesn't eliminate it).</p>
<p>PD disaggregation splits the two phases <strong>physically</strong>: a <strong>prefill pool</strong> does only prefills, a <strong>decode pool</strong> does only decodes; once prefill finishes, the request's <strong>KV cache</strong> (<strong>Lesson 30</strong>'s paged KV) is <strong>transferred</strong> over a fast interconnect like RDMA / NVLink to the decode GPU, which then streams tokens. Each pool can be <strong>sized and tuned independently</strong> and each saturates its own bottleneck. Forward-ref <strong>Lessons 46 / 47</strong>: disaggregation + large-scale EP is how DeepSeek-scale serving is built.</p>
<p>Seen from another angle, PD disaggregation simply applies the engineering principle of <strong>specialization of labor</strong> to inference hardware. Just as a modern factory won't have one worker run both the lathe and quality control, but splits the line into specialized stations each fitted with the right equipment, an inference service shouldn't make one GPU carry both compute-bound prefill and bandwidth-bound decode. The cost of the division is an extra <strong>hand-off</strong> between stations (the KV transfer); the reward is that every station can be polished to peak utilization. Whether you profit from the division depends on whether the <strong>hand-off cost is low enough</strong>—which leads us to its hard dependency on fast interconnects.</p>
</div>

<h2>1. Why split? Two faces, conflicting resources</h2>
<p>Across a request's lifetime, <strong>prefill</strong> and <strong>decode</strong> place <strong>mirror-opposite</strong> demands on the GPU. Prefill feeds an N-token prompt into the network as one big batch, all tokens computed in parallel, the matrix multiplies keeping the <span class="mono">Tensor Cores</span> fully fed—it is <strong>short on compute</strong>, with bandwidth to spare. Decode is the reverse: each step has just 1 new token and tiny compute, but every emitted token requires re-reading the <strong>entire set of weights</strong> and the <strong>accumulated KV cache</strong> from memory, so the bottleneck is squarely <strong>memory bandwidth</strong> while the math units sit idle.</p>
<p>This "one busy, one idle" mismatch isn't occasional but a <strong>structural feature that runs through the whole request</strong>: as long as the model generates autoregressively, prefill and decode have inherently opposite resource profiles. And precisely because it is structural, pure scheduling tricks (priorities, slicing, preemption) can only <strong>ease</strong> it, never <strong>eliminate</strong> it; the real fix must drop down to the level of "put the two kinds of work on different hardware." That is exactly what PD disaggregation does.</p>
<p>When the two share one GPU, conflict is unavoidable. The scheduler must either favor prefill (raising the ITL of decoding requests) or favor decode (slowing new requests' TTFT). <span class="mono">Chunked prefill</span> (Lesson 22) slices a long prefill into small pieces interleaved between decode steps, shrinking the jitter—but as long as both kinds of work <strong>queue on the same compute</strong>, the interference can't be fully removed. The insight of PD disaggregation is blunt: <strong>since the two faces don't eat the same thing, don't make them fight over one table</strong>.</p>
<p>Quantifying the mismatch makes it tangible. Prefill's <strong>arithmetic intensity</strong> (multiply-adds per byte of weights read) is high, because a whole batch of tokens shares one weight read; decode's arithmetic intensity is far lower, re-hauling the weights for almost every token, so what really decides decode speed is not peak compute but <strong>memory bandwidth</strong>. This means: piling more compute onto decode barely helps, and piling more bandwidth onto prefill doesn't help either. Bolt them onto one card and, however you tune it, half the hardware sits in <strong>structural idleness</strong>—either compute idle or bandwidth idle.</p>
<p>Worse is the <strong>latency coupling</strong>. In online serving TTFT (time to first token) is set by prefill and ITL (inter-token latency) by decode, and these are often the metrics that different requests and different users care about most. Co-located, the prefill of one long prompt monopolizes compute for several milliseconds, during which all decoding requests are <strong>suspended</strong> and ITL spikes. Chunked prefill slices the long prefill and interleaves it, flattening the spike into small ripples, but at the price of scheduling overhead from repeatedly interrupting decode; in essence it does <strong>time-multiplexing within one resource pool</strong>, whereas PD disaggregation does <strong>spatial isolation</strong> directly, removing the preemption at the root.</p>

<div class="flow"><div class="node">request arrives</div><div class="arrow">→</div><div class="node">prefill pool<br><span class="mono">compute-bound</span></div><div class="arrow">→</div><div class="node">KV transfer<br><span class="mono">RDMA/NVLink</span></div><div class="arrow">→</div><div class="node">decode pool<br><span class="mono">bandwidth-bound</span></div><div class="arrow">→</div><div class="node">stream tokens</div></div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="prefill pool computes the whole KV, transferred over RDMA/NVLink to the decode pool, which reads it and streams tokens one by one">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">one request crosses two pools</text>
    <rect x="20" y="96" width="74" height="66" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="57" y="126" text-anchor="middle" style="font-size:12px">request</text>
    <text x="57" y="146" text-anchor="middle" style="fill:var(--faint);font-size:11px">prompt</text>
    <line x1="98" y1="129" x2="126" y2="129" style="stroke:var(--line);stroke-width:2"/>
    <polygon points="128,129 118,124 118,134" style="fill:var(--line)"/>
    <rect x="132" y="90" width="150" height="78" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="207" y="116" text-anchor="middle" style="fill:var(--blue);font-weight:700">Prefill pool</text>
    <text x="207" y="136" text-anchor="middle" style="font-size:12px">computes all KV</text>
    <text x="207" y="155" text-anchor="middle" class="mono" style="font-size:11px">→ KV cache</text>
    <line x1="286" y1="129" x2="314" y2="129" style="stroke:var(--amber);stroke-width:2"/>
    <polygon points="316,129 306,124 306,134" style="fill:var(--amber)"/>
    <rect x="320" y="96" width="140" height="66" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="390" y="124" text-anchor="middle" style="fill:var(--amber);font-weight:700">KV transfer</text>
    <text x="390" y="144" text-anchor="middle" class="mono" style="font-size:11px">RDMA / NVLink</text>
    <line x1="464" y1="129" x2="492" y2="129" style="stroke:var(--amber);stroke-width:2"/>
    <polygon points="494,129 484,124 484,134" style="fill:var(--amber)"/>
    <rect x="498" y="90" width="150" height="78" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="573" y="116" text-anchor="middle" style="fill:var(--teal);font-weight:700">Decode pool</text>
    <text x="573" y="136" text-anchor="middle" style="font-size:12px">1 token/step</text>
    <text x="573" y="155" text-anchor="middle" class="mono" style="font-size:11px">reads KV</text>
    <line x1="652" y1="129" x2="680" y2="129" style="stroke:var(--line);stroke-width:2"/>
    <polygon points="682,129 672,124 672,134" style="fill:var(--line)"/>
    <rect x="686" y="96" width="96" height="66" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="734" y="124" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">Stream</text>
    <text x="734" y="144" text-anchor="middle" class="mono" style="font-size:11px">tokens</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Two-pool pipeline</b> — a request enters the prefill pool, which computes the whole prompt's KV; that KV is transferred over RDMA/NVLink to the decode pool, which reads it and streams tokens one by one.</div>
</div>

<h2>2. After the split: two independent pools</h2>
<p>PD disaggregation cuts the cluster in two. The <strong>prefill pool</strong> chases <strong>throughput</strong>: it handles bursts of big parallel computation, suits high-compute configs and can tolerate slightly larger batches; its output isn't tokens but <strong>a computed KV cache</strong>. The <strong>decode pool</strong> chases <strong>low, steady token latency</strong>: it wants high memory bandwidth and small, stable batches to drive each request's ITL down. The two pools' sizes, batching strategies, and parallelism can be <strong>tuned independently</strong>—impossible when they're bolted together.</p>
<p>"Independent" also shows up in <strong>failure handling and scaling</strong>. If a node in the decode pool dies, the router can redirect subsequent requests to other decode workers while the prefill pool is wholly unaffected; conversely, to temporarily boost prefill throughput you just add cards to the prefill pool without touching decode. This <strong>decoupled elasticity</strong> lets online serving ride out traffic swings more smoothly, and lets ops actions like canary upgrades and capacity planning proceed pool by pool without interrupting each other.</p>
<p>The crucial hand-off is the <strong>KV-cache transfer</strong>. After prefill finishes, the request's <strong>paged KV</strong> (Lesson 30, KV stored page by page) must be <strong>shipped</strong> from the prefill GPU's memory to the decode GPU's memory. This step rides a high-speed interconnect like <span class="mono">RDMA</span> or <span class="mono">NVLink</span>, the goal being a transfer time far smaller than the recomputation it saves. A <strong>router / load-balancer</strong> (Lesson 13) is responsible for <strong>pairing</strong> a prefill worker with a decode worker per request.</p>
<p>Independent tuning brings a very practical lever: the <strong>pool ratio</strong>. If production traffic is "long prompts, short outputs" (e.g. retrieval-augmented QA), prefill is the pressure point, so provision more prefill nodes; if it is "short prompts, long outputs" (e.g. long-form writing, reasoning chains), decode is the pressure point, so provision more decode nodes. This P∶D ratio can be <strong>elastically adjusted</strong> with the traffic profile, even scaled by time of day—an operational freedom the single-card co-location mode simply cannot offer. The two pools can also each pick the most cost-effective hardware: high-compute cards for the prefill pool, high-bandwidth / large-memory cards for the decode pool.</p>
<p>Of course, disaggregation is no free lunch. It introduces a <strong>cross-node KV transfer</strong>, adding the network as one more link; if the interconnect isn't fast enough, transfer time eats back into the savings and can even raise TTFT. So PD disaggregation is premised on having a <strong>fast enough interconnect</strong> (RDMA/NVLink) and on <strong>overlapping transfer with compute</strong> as much as possible. Understanding this boundary tells you it suits large, well-interconnected clusters—not any random pair of cards that should be split.</p>

<div class="cols"><div class="col"><strong>Prefill node</strong><br><span class="mono">compute-bound</span><br>computes the whole prompt in one parallel pass<br>saturates Tensor Cores / compute<br>output: a computed <strong>KV cache</strong></div><div class="col"><strong>Decode node</strong><br><span class="mono">bandwidth-bound</span><br>generates only 1 token per step<br>re-reads weights + KV, compute idle<br>output: <strong>streamed tokens</strong></div></div>

<div class="fig">
  <svg viewBox="0 0 800 290" role="img" aria-label="left prefill node is compute-bound: whole prompt in parallel, compute maxed; right decode node is bandwidth-bound: 1 token per step reading a large KV cache, compute idle">
    <line x1="400" y1="44" x2="400" y2="272" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="30" y="32" style="font-weight:700;fill:var(--blue)">Prefill node</text>
    <rect x="30" y="44" width="140" height="26" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="100" y="62" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">compute-bound</text>
    <text x="30" y="92" style="font-size:12px;fill:var(--muted)">all tokens parallel → busy</text>
    <rect x="40" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="82" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="124" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="166" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="208" y="112" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="40" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="82" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="124" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="166" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="208" y="142" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="40" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="82" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="124" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="166" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="208" y="172" width="30" height="22" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <text x="252" y="158" style="fill:var(--blue);font-size:12px">all busy</text>
    <text x="30" y="258" style="font-size:12px">bottleneck: GPU FLOPs</text>
    <text x="430" y="32" style="font-weight:700;fill:var(--teal)">Decode node</text>
    <rect x="430" y="44" width="160" height="26" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="510" y="62" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--teal)">bandwidth-bound</text>
    <text x="430" y="92" style="font-size:12px;fill:var(--muted)">1 token/step → mostly idle</text>
    <rect x="430" y="112" width="30" height="22" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.2"/>
    <rect x="472" y="112" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="514" y="112" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="430" y="142" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="472" y="142" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="514" y="142" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="430" y="172" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="472" y="172" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <rect x="514" y="172" width="30" height="22" rx="3" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.2"/>
    <line x1="598" y1="150" x2="550" y2="150" style="stroke:var(--amber);stroke-width:2.5"/>
    <polygon points="548,150 558,145 558,155" style="fill:var(--amber)"/>
    <text x="556" y="138" style="fill:var(--amber);font-size:11px">bandwidth</text>
    <rect x="600" y="108" width="120" height="92" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="660" y="148" text-anchor="middle" style="fill:var(--teal);font-weight:700">large KV</text>
    <text x="660" y="170" text-anchor="middle" style="font-size:11px">read each step</text>
    <text x="430" y="258" style="font-size:12px">bottleneck: memory BW</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Compute-bound vs bandwidth-bound</b> — prefill computes the whole prompt in one parallel pass with compute fully busy; decode emits just 1 token per step yet re-reads a large KV cache at full bandwidth, leaving compute mostly idle. Split into pools, each picks its own hardware and batch size.</div>
</div>

<p><strong>In deployment:</strong> launch with <span class="mono">--disaggregation-mode prefill</span> and <span class="mono">--disaggregation-mode decode</span> to split one model into prefill workers and decode workers. Take a <strong>2000-token</strong> prompt: the prefill pool computes its KV in one pass, and that KV is <strong>transferred once</strong> to the decode pool; decode then streams hundreds of tokens <strong>without recomputing the prompt</strong>. If traffic is "long prompt, short answer," provision more prefill cards; if "short prompt, long answer," provision more decode cards — the P∶D ratio flexes with the traffic.</p>

<h2>3. How SGLang abstracts the transfer</h2>
<p>Interconnect hardware varies wildly across clusters, so SGLang doesn't hard-code the transfer logic—it hides it behind a <strong>pluggable connector</strong>. The prefill side holds a <span class="mono">BaseKVSender</span>; the decode side holds a mirrored <span class="mono">BaseKVReceiver</span>. The sender's interface is deliberately spare: <span class="mono">init</span> first <strong>announces</strong> how many KV pages will move; <span class="mono">send</span> <strong>pushes</strong> this request's KV pages to the decode worker; <span class="mono">poll</span> returns a <strong>non-blocking</strong> <span class="mono">KVPoll</span> status (Bootstrapping / WaitingForInput / Transferring / Success / Failed) so the scheduler never has to block; <span class="mono">get_transfer_metric</span> yields a <span class="mono">KVTransferMetric</span> (bytes, latency) for observability.</p>
<p>The payoff of this abstraction is <strong>swappable backends</strong>: concrete connectors include <span class="mono">Mooncake</span>, <span class="mono">NIXL</span>, and <span class="mono">ascend</span>, each binding a different interconnect stack, yet the scheduler above sees the same <span class="mono">init / send / poll</span> contract. The decode-side <span class="mono">BaseKVReceiver</span> is symmetric: <span class="mono">init</span> + <span class="mono">send_metadata</span> + <span class="mono">poll</span> — the decode side first uses <span class="mono">send_metadata</span> to <strong>advertise</strong> its own KV slot indices to the prefill side, which then writes the KV straight into decode memory (typically one-sided RDMA); and when poll returns <span class="mono">Success</span> the KV has landed and decode can start emitting tokens.</p>
<p>Why must <span class="mono">poll</span> be <strong>non-blocking</strong>? Because the scheduler is a single thread driving the state machine of a large batch of requests, and it must never stall the whole batch waiting on one KV transfer. A non-blocking poll lets the scheduler just <strong>glance</strong> at the current status each round: still <span class="mono">Bootstrapping</span> (handshake) means go serve other requests first, <span class="mono">Transferring</span> means keep waiting, only <span class="mono">Success</span> advances that request to decode, and <span class="mono">Failed</span> triggers retry / fallback. This "state machine + polling" design is the classic way high-concurrency services decouple slow I/O from fast scheduling.</p>
<p>Designing the sender interface so <strong>spare</strong> (only four actions: init/send/poll/get_transfer_metric) is also deliberate: the narrower the interface, the easier new backends are to plug in. To adopt a new interconnect you only implement these methods and make poll correctly translate the underlying progress into the five <span class="mono">KVPoll</span> states; not a line of the upper routing, scheduling, or pairing logic needs to change. The <span class="mono">KVTransferMetric</span> (bytes, latency) returned by <span class="mono">get_transfer_metric</span> feeds monitoring, letting you pinpoint "is some interconnect dragging us down."</p>

<table class="t"><tr><th>Method</th><th>Side / role</th></tr><tr><td><span class="mono">init(num_kv_indices, aux_index)</span></td><td>prefill side: announce how many KV pages will move</td></tr><tr><td><span class="mono">send(kv_indices)</span></td><td>prefill side: push this request's KV pages to the decode worker</td></tr><tr><td><span class="mono">poll() → KVPoll</span></td><td>both sides: non-blocking transfer status</td></tr><tr><td><span class="mono">get_transfer_metric()</span></td><td>prefill side: returns KVTransferMetric (bytes, latency)</td></tr><tr><td><span class="mono">BaseKVReceiver.send_metadata()</span></td><td>decode side: advertise this side's KV slot indices to prefill (paired with send)</td></tr></table>

<p>Unrolling one successful transfer into a timeline gives the vertical sequence below. Note how <span class="mono">poll</span> is called repeatedly in the middle: it <strong>doesn't block</strong>, just reports the current status each time, until <span class="mono">Success</span>.</p>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>init</h4><p>announce num_kv_indices KV pages to move</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>send</h4><p>push the request's KV pages to the decode side</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>poll → KVPoll</h4><p>Transferring (still moving, <strong>non-blocking</strong>)</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>poll → Success</h4><p>KV received, decode starts emitting tokens</p></div></div></div>

<h2>4. Putting it back in the big picture</h2>
<p>PD disaggregation isn't an isolated trick; it strings several past lessons together: Lesson 4 gave us the two-phase resource profiles, Lessons 8 / 22's TTFT/ITL tension is the pain it solves, Lesson 30's paged KV is the object being shipped, and Lesson 13's router pairs the workers. This string of references isn't padding; it shows that PD disaggregation is essentially a <strong>system-level integration</strong> built on top of every mechanism before it—drop any one link and it won't stand up. Looking ahead, <strong>Lessons 46 / 47</strong> stack <strong>disaggregation</strong> with <strong>large-scale expert parallelism (EP)</strong>—the prefill and decode pools each get their own EP sharding, which is the real recipe for DeepSeek-scale online serving. Grasp "two faces, pool-and-conquer, KV transfer," and you hold the backbone of modern large-scale inference serving, and grasp the foundation the later advanced lessons build upon.</p>
<p>Let's walk the whole pipeline once more as a recap: a request first hits the <strong>router</strong> and is assigned a prefill worker; prefill computes the entire prompt in one pass, produces a paged KV, and via <span class="mono">BaseKVSender</span>'s <span class="mono">init → send</span> pushes those pages to the paired decode worker; the decode-side <span class="mono">BaseKVReceiver</span> advertises its KV slots via <span class="mono">send_metadata</span>, both sides <span class="mono">poll</span> until the status flips to <span class="mono">Success</span>; at that moment the KV is in place in the decode GPU's memory, decode takes over and streams tokens back to the user one at a time. Throughout, the prefill pool is always "prepping" at full compute and the decode pool always "serving" at full bandwidth, with neither idling for the other. You could say disaggregation turns "hardware utilization" from a reluctant compromise inside a single card into a clear division of labor at the whole-cluster level that is designable, measurable, optimizable, and elastically scalable.</p>
<p>One last easily-missed point: PD disaggregation changes the <strong>physical deployment</strong>, not the model's own math. The same model, the same weights, the same KV—it merely puts "computing the KV" and "using the KV" on different machines, with one explicit transfer in between. Precisely because it touches deployment without touching semantics, it can act as a <strong>composable building block</strong>, stacked with quantization, tensor parallelism, expert parallelism (EP), and more—which is exactly why the next stage (Lessons 46/47) can combine it with large-scale EP to build a DeepSeek-scale online inference system.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::BaseKVSender</span><span class="ln">the prefill side ships KV to the decode side: init → send → poll(KVPoll)</span></div><pre>class BaseKVSender(ABC):                 # lives on the PREFILL side
    @abstractmethod
    def init(self, num_kv_indices, aux_index=None):
        ...                              # announce how many KV pages will move
    @abstractmethod
    def send(self, kv_indices):
        ...                              # push this request's KV pages to the decode worker
    @abstractmethod
    def poll(self):                      # non-blocking status -&gt; KVPoll
        ...                              # Bootstrapping / WaitingForInput / Transferring / Success / Failed
    @abstractmethod
    def get_transfer_metric(self):       # -&gt; KVTransferMetric (bytes, latency)
        ...
# BaseKVReceiver mirrors this on the DECODE side (init + send_metadata + poll)</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::BaseKVReceiver</span><span class="ln">decode side: receive the KV from prefill, then start decoding</span></div><pre>class BaseKVReceiver(ABC):
    # the DECODE side of PD disaggregation: pull the KV that the
    # prefill node produced, into this node's KV pool.
    @abstractmethod
    def init(self, ...): ...                        # set up the receive session
    @abstractmethod
    def send_metadata(self, kv_indices, ...): ...   # announce where the KV lands
    @abstractmethod
    def poll(self) -&gt; KVPoll:
        ...   # WaitingForInput -&gt; Transferring -&gt; Success
    # once KV has arrived, the decode node generates tokens from it.</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>Two faces, opposite resources</strong>: prefill is a compute-bound parallel pass; decode is bandwidth-bound token-by-token generation. Co-locating them inevitably pits TTFT against ITL.</li>
<li><strong>PD disaggregation = pool and conquer</strong>: the prefill pool only prefills, the decode pool only decodes, each saturating its own bottleneck and tuned independently.</li>
<li><strong>What gets transferred is the KV cache</strong>: prefill's computed paged KV (Lesson 30) is shipped over RDMA / NVLink to the decode GPU, which then streams tokens.</li>
<li><strong>Pluggable connector</strong>: <span class="mono">BaseKVSender</span> (init / send / poll → KVPoll / get_transfer_metric) and the mirrored <span class="mono">BaseKVReceiver</span>; backends include Mooncake, NIXL, ascend.</li>
<li><strong>poll is non-blocking</strong>: status moves through Bootstrapping / WaitingForInput / Transferring / Success / Failed, and Success means KV has landed and decode begins.</li>
<li><strong>Connecting the dots</strong>: the router (Lesson 13) pairs workers; forward-ref Lessons 46 / 47, disaggregation + large-scale EP is the recipe for DeepSeek-scale serving.</li>
</ul></div>
"""}
LESSON_46 = {"zh": r"""
<p class="lead">一块 GPU 放不下、也算不动一个超大模型时，我们就要把"模型"或"请求"摊到许多张 GPU 上。摊开的方式只有四种主轴：<strong>TP（张量并行）</strong>、<strong>PP（流水线并行）</strong>、<strong>EP（专家并行）</strong>、<strong>DP（数据并行）</strong>。这一课把这四种并行讲清楚：各自切的是什么、每一步要通信什么、对网络链路有什么要求；最后揭示 SGLang 的一个关键统一——它们都只是同一个 <span class="mono">GroupCoordinator</span> 抽象在不同 rank 布局上的实例，并且可以彼此<strong>组合</strong>。理解了这四根主轴与它们的组合，你就能看懂任何一份 SGLang 部署命令里 <span class="mono">tp_size</span>、<span class="mono">pp_size</span>、<span class="mono">ep_size</span>、<span class="mono">dp_size</span> 这几个数字究竟在切什么、又各自付出了什么代价。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把推理想象成一家要处理海量订单的大工厂。扩张产能有四种思路：</p>
<p><strong>TP</strong> 像在<strong>同一条流水线内</strong>把"组装一台机器"这件事拆给并排的四个工人同时拧不同的螺丝——每个人只做一部分，最后必须把零件<strong>合并</strong>成完整成品（这一步就是 all-reduce）。合并很频繁，所以四个人要肩并肩坐（NVLink 这种高速近邻链路）。</p>
<p><strong>PP</strong> 像把整条产线拆成<strong>前后衔接的几个车间</strong>：车间一做底盘，做完<strong>传给</strong>车间二装外壳，再传给车间三质检。只有车间交界处才搬运半成品，链路便宜也行；但开工时前几个车间在等料、收尾时后几个车间空着，这就是"流水线气泡"。</p>
<p><strong>EP</strong> 像有很多<strong>专科师傅</strong>（专家），每张订单只需要其中两位师傅。系统先把每张订单<strong>派送</strong>到拥有对应师傅的车间（all-to-all），做完再<strong>送回</strong>（第二次 all-to-all）。师傅越多、每单要找的师傅越多，搬运量越大。</p>
<p><strong>DP</strong> 像直接<strong>复制</strong>一整座工厂，再把订单平均分给每座工厂各自独立处理——几乎不用互相搬运，但每座工厂都得备齐全套设备（整份模型权重）。</p>
<p>现实里这四招往往<strong>同时用上</strong>：先把一座超大工厂复制成好几座（DP），每座工厂内部把产线拆成前后衔接的车间（PP），每个车间里再让并排的工人分担同一道工序（TP），而众多专科师傅则按工种集中到各自的房间（EP）。于是<strong>同一名工人同时隶属好几种编队</strong>——他的"并排组""车间组""复制组"各管各的搬运。能这样自由叠用，正是本课最后要点出的"可组合"。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>四种并行回答的是两个不同的问题。<strong>TP / PP / EP</strong> 在切<strong>模型本身</strong>：TP 切一层之内的权重矩阵（第25、37课的列/行切分与词表切分），PP 切层与层之间的边界（第23课），EP 切 MoE 的专家集合（第34课）。<strong>DP</strong> 不切模型而是切<strong>数据（请求）</strong>：复制模型、分摊请求（第23课的 dp-controller）。一句话：前三种"分担一份模型的体积与算力"，DP"分担吞吐"。SGLang 还有一个变体叫 <strong>Attention-DP</strong>：对注意力做 DP 切分、对专家做 EP 切分，二者并存。真正优雅的是：这四种并行在代码里没有四套实现，而是同一个 <span class="mono">GroupCoordinator</span> 在不同的 rank 分组上跑不同的集合通信，于是它们天然可以叠乘组合，例如 TP=8 × PP=2 × DP=4。</p>
<p>那么什么时候用哪一招？可以顺着"瓶颈是什么"来判断：模型大到<strong>一张卡装不下</strong>，先上 TP 把每一层切薄；单机的卡数还不够，再叠 PP 把层分到更多机器上。模型是<strong>专家众多的 MoE</strong>、总参数把显存撑爆，就上 EP 把专家摊开。模型本来就<strong>装得下</strong>、只是想要<strong>更高吞吐</strong>，那就上 DP 多复制几份、把请求分摊出去。绝大多数真实部署都不是单选题，而是按这套优先级把几招叠起来用：TP 吃满单机内的高速链路，PP 跨机扩展模型规模，EP 摊开专家，DP 复制换吞吐。</p>
</div>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="四种并行切分：TP 切一层内的矩阵、PP 切层堆叠成阶段、EP 把专家摊到各卡、DP 复制整模型分请求">
    <rect x="20" y="20" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="36" y="46" style="font-weight:700;fill:var(--blue)">TP 张量并行</text>
    <text x="36" y="66" style="fill:var(--muted);font-size:12px">每卡持一层的 ¼ 切片</text>
    <rect x="40" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="62" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼层</text>
    <rect x="110" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="132" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼层</text>
    <rect x="180" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="202" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼层</text>
    <rect x="250" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="272" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼层</text>
    <rect x="410" y="20" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="426" y="46" style="font-weight:700;fill:var(--teal)">PP 流水线并行</text>
    <text x="426" y="66" style="fill:var(--muted);font-size:12px">每卡持连续几层（一个阶段）</text>
    <rect x="430" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="452" y="119" text-anchor="middle" class="mono" style="font-size:11px">段1</text>
    <text x="487" y="119" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="500" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="522" y="119" text-anchor="middle" class="mono" style="font-size:11px">段2</text>
    <text x="557" y="119" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="570" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="592" y="119" text-anchor="middle" class="mono" style="font-size:11px">段3</text>
    <text x="627" y="119" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="640" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="662" y="119" text-anchor="middle" class="mono" style="font-size:11px">段4</text>
    <rect x="20" y="170" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="36" y="196" style="font-weight:700;fill:var(--amber)">EP 专家并行</text>
    <text x="36" y="216" style="fill:var(--muted);font-size:12px">每卡持一部分专家</text>
    <rect x="40" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="62" y="269" text-anchor="middle" class="mono" style="font-size:11px">专家</text>
    <rect x="110" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="132" y="269" text-anchor="middle" class="mono" style="font-size:11px">专家</text>
    <rect x="180" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="202" y="269" text-anchor="middle" class="mono" style="font-size:11px">专家</text>
    <rect x="250" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="272" y="269" text-anchor="middle" class="mono" style="font-size:11px">专家</text>
    <rect x="410" y="170" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="426" y="196" style="font-weight:700;fill:var(--purple)">DP 数据并行</text>
    <text x="426" y="216" style="fill:var(--muted);font-size:12px">每卡持整份模型副本</text>
    <rect x="430" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="452" y="269" text-anchor="middle" class="mono" style="font-size:11px">整模</text>
    <rect x="500" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="522" y="269" text-anchor="middle" class="mono" style="font-size:11px">整模</text>
    <rect x="570" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="592" y="269" text-anchor="middle" class="mono" style="font-size:11px">整模</text>
    <rect x="640" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="662" y="269" text-anchor="middle" class="mono" style="font-size:11px">整模</text>
  </svg>
  <div class="figcap"><b>图 1 · 四种切分</b> — 同样 4 张卡：TP 让每卡持一层的 ¼ 切片（切层内矩阵），PP 让每卡持连续几层即一个阶段（切层堆叠），EP 让每卡持一部分专家，DP 让每卡持整份模型副本、只分请求。</div>
</div>

<h2>TP 张量并行：切开"一层之内"的矩阵</h2>
<p><strong>TP（tensor parallel）</strong>把每一层的<strong>权重矩阵</strong>沿某个维度切成若干片，分给同组的各个 rank。最典型的就是第25课讲的注意力 q/k/v 投影按列切、第37课讲的 MLP 上投影按列切、词表（vocab）按行切。每个 rank 只持有矩阵的一片，于是只算出<strong>部分结果</strong>；要得到完整的层输出，就必须把各 rank 的部分和<strong>合并</strong>——列切+行切的组合用一次 <span class="mono">all_reduce</span> 求和，词表并行的 logits 则用 <span class="mono">all_gather</span> 拼接。</p>
<p>TP 的特点是：它把<strong>单层的算力和显存</strong>都摊薄了，一张卡装不下的大矩阵被切成几片分别存放、分别计算。代价是<strong>每一层</strong>都要通信一次（前向一次、反向再一次），通信极其频繁。因此 TP 组内的 GPU 必须用<strong>高速近邻链路</strong>（NVLink / NVSwitch），通常局限在<strong>单机之内</strong>；跨机做 TP 会被慢链路拖垮。</p>
<div class="layers"><div class="layer">Rank0：每一层都只持有该层权重的 1/4 切片</div><div class="layer">Rank1：每一层都只持有该层权重的 1/4 切片</div><div class="layer">Rank2：每一层都只持有该层权重的 1/4 切片</div><div class="layer">Rank3：每一层都只持有该层权重的 1/4 切片</div></div>
<p>注意上图：TP 下<strong>每个 rank 都"纵向"贯穿了所有层</strong>，只是每层都只拿到一小条。这正是 TP 与 PP 的根本区别——TP 横向切一层的宽度，PP 纵向切层的深度。</p>
<p>举个具体的数：一个 70B 模型的单层权重动辄上百 MB，配成 TP=8 时每张卡只存其中 1/8、也只算 1/8，但<strong>每过一层</strong>就要做一次覆盖这 8 张卡的 <span class="mono">all_reduce</span> 把部分和加起来。正因为通信如此密集，TP 的并行度通常<strong>不超过单机的卡数</strong>（例如 8）；再往上扩就该换成 PP 或 DP，否则频繁的跨机 all_reduce 会让通信时间盖过省下来的计算时间。一句话记住 TP 的取舍：它同时摊薄了"单层算力"和"单层显存"两样东西，代价是"每层都要通信、通信最贵、必须近邻高速链路"。</p>
<p>把这些落到一条具体命令上：<span class="mono">--tp-size 8</span> 会把<strong>每一层</strong>的矩阵切到 8 张卡上，每过一层就做<strong>一次</strong> <span class="mono">all_reduce</span> 把部分和加起来——正因为它<strong>每层都通信</strong>，TP 组必须用机内的高速链路（NVLink）；相比之下 <span class="mono">--dp-size</span> 复制整份模型、每步<strong>几乎不通信</strong>，慢链路也无所谓。这正是"TP 吃满机内 NVLink、DP 几乎不通信"这条经验法则的由来。</p>

<h2>PP 流水线并行：把"层"切成前后接力的阶段</h2>
<p><strong>PP（pipeline parallel）</strong>把模型的<strong>层</strong>按顺序分成几个连续的<strong>阶段（stage）</strong>，放在不同 GPU 上。第 1～k 层在 rank0，第 k+1～2k 层在 rank1，依此类推。前向时，激活值从一个阶段<strong>流向</strong>下一个阶段（向 <span class="mono">next_rank</span> 发送、从 <span class="mono">prev_rank</span> 接收，第23课）；反向时梯度沿相反方向回传。</p>
<p>PP 只在<strong>阶段交界处</strong>通信，通信量远小于 TP，普通链路（甚至跨机）都能胜任。但它有个固有问题：若只送一个批次进去，前面的阶段算完就得干等后面阶段，利用率很低。解决办法是把一个大批次拆成许多<strong>微批次（micro-batch）</strong>像流水线一样连续灌入，让所有阶段同时有活干；即便如此，开头和结尾仍有阶段空转，这段浪费叫<strong>流水线气泡（bubble）</strong>。PP 摊薄的是<strong>层数带来的显存</strong>（每张卡只放一部分层），而不是单层的算力。</p>
<p>气泡占比大致正比于<strong>阶段数 ÷ 微批次数</strong>：阶段越多、微批次越少，开头收尾的空转占比就越高；把微批次开多，就能把这段浪费摊薄到可以接受。PP 的最大好处是<strong>只在交界处搬一份激活</strong>，搬运量只与隐藏维度（以及 batch、序列长度）成正比、与层数无关，因此特别适合"模型层数太多、单机装不下、机器之间又只有较慢链路"的场景——先用 PP 把模型<strong>纵向</strong>铺到多台机器上，再在每台机器内部用 TP 做<strong>横向</strong>细切，两者刚好互补。</p>

<h2>EP 专家并行：把 MoE 的"专家"摊到不同 rank</h2>
<p><strong>EP（expert parallel）</strong>专为 MoE 而生（第34课）。MoE 层里有很多专家（FFN），但每个 token 的路由器只挑选其中 top-k 个去计算。EP 把不同的<strong>专家</strong>放在不同 rank 上：每一步，先用一次 <span class="mono">all_to_all</span> 把<strong>每个 token</strong> 送到"它选中的专家所在"的那个 rank，专家算完后再用第二次 <span class="mono">all_to_all</span> 把结果送回原来的 rank。</p>
<p>为什么 EP 用 <span class="mono">all_to_all</span> 而不是 TP 那样的 <span class="mono">all_reduce</span>？因为每个 token 要去的目的地<strong>各不相同</strong>——token A 选了专家3、token B 选了专家7，这是一种"点对点重排"而非"全体求和"。<span class="mono">all_to_all</span> 正是为这种"各发各的、各收各的"而设计的集合通信：每个 rank 同时向所有其它 rank 发送一份不同的数据、也从所有 rank 各收一份，恰好对应"把 token 按所选专家分发到对应 rank"这件事。</p>
<p>EP 摊薄的是<strong>专家的显存</strong>——成百上千个专家分散存放，单卡放不下的总参数被分摊。它的通信量随 <strong>token 数 × top-k</strong> 增长：批次越大、每个 token 选的专家越多，all-to-all 搬运的数据越多。SGLang 的 Attention-DP 常与 EP 搭配：注意力部分按 DP 复制分流、专家部分按 EP 分片，二者在同一部署中并存。这又是一个"不同 GroupCoordinator 各司其职"的实例。大规模 EP 以及专家负载均衡（EPLB）会在<strong>第47课</strong>展开。</p>
<p>EP 与 TP 还有一个关键差别在于<strong>通信量是否随负载浮动</strong>：TP 每层 all_reduce 的大小是固定的（只取决于隐藏维度），而 EP 的 all_to_all 大小随<strong>实际 token 数 × top-k</strong> 变化，batch 越大、每个 token 选的专家越多，搬运的数据越多。更棘手的是<strong>专家负载常常不均</strong>——某些热门专家被大量 token 选中、某些专家几乎闲置，持有热门专家的那个 rank 就会成为整步的瓶颈。如何把专家在 rank 间重新摆放、让负载尽量均衡，正是第47课 EPLB 要解决的问题。</p>

<h2>DP 数据并行 + 统一抽象 GroupCoordinator</h2>
<p><strong>DP（data parallel）</strong>最简单：<strong>复制</strong>整份模型到每个副本，把<strong>请求</strong>分给不同副本各自独立处理（第23课的 dp-controller 负责分发）。每一步几乎<strong>零通信</strong>（仅训练时才需在反向后同步梯度，推理服务里副本间近乎独立），代价是每个副本都要装下<strong>整份模型</strong>，省不了显存只换吞吐。</p>
<p>SGLang 的 <strong>Attention-DP</strong> 值得单独点一句：在大规模 MoE 部署里，注意力部分参数小、却最怕 KV 缓存被重复占用与冗余计算，于是按 DP 分流让每个副本各管一批请求的注意力；而专家部分参数巨大、必须摊开显存，于是按 EP 分片。两种切法在<strong>同一份模型</strong>里各管一段——注意力走 DP、FFN/专家走 EP，互不冲突。这本质上又是"不同的 GroupCoordinator 在不同 rank 布局上各司其职"，只不过这次把 DP 和 EP 拼在了一个模型的不同子模块上。</p>
<table class="t"><tr><th>并行轴</th><th>切什么</th><th>每步通信什么</th><th>链路需求</th></tr>
<tr><td><span class="mono">TP</span></td><td>每层的权重矩阵（一层之内）</td><td>每层一次 all-reduce / all-gather</td><td>很高，需 NVLink，通常单机内</td></tr>
<tr><td><span class="mono">PP</span></td><td>模型的层（分成连续阶段）</td><td>仅阶段交界 send / recv 激活</td><td>低，普通链路即可，可跨机</td></tr>
<tr><td><span class="mono">EP</span></td><td>MoE 的专家集合</td><td>两次 all-to-all（按 token×top-k）</td><td>中高，随 token 量增长</td></tr>
<tr><td><span class="mono">DP</span></td><td>请求（模型整份复制）</td><td>每步近乎零（仅训练同步梯度）</td><td>很低</td></tr></table>
<p>真正的关键统一在这里：SGLang 不为四种并行写四套通信代码，而是用<strong>同一个抽象</strong> <span class="mono">GroupCoordinator</span> 表达全部。一个 <span class="mono">GroupCoordinator</span> 就是"一个进程组"，带着 <span class="mono">rank</span>（本进程全局编号）、<span class="mono">ranks</span>（属于本组的全局 rank 列表）、<span class="mono">world_size</span>（本组大小）、<span class="mono">rank_in_group</span>（组内位置），并暴露 <span class="mono">all_reduce</span> / <span class="mono">all_gather</span> / <span class="mono">all_to_all_single</span> / <span class="mono">next_rank</span> / <span class="mono">prev_rank</span> 这些方法。</p>
<div class="cols"><div class="col"><strong>切"权重"（TP / PP / EP）</strong><br/>把一份模型的矩阵、层、专家分散到多卡，每卡只持有一部分参数；省显存、省单卡算力，但要靠集合通信把结果拼回来。</div><div class="col"><strong>切"数据"（DP）</strong><br/>把整份模型复制到每个副本，按请求分流；每副本独立、几乎不通信，省不了显存只提吞吐。</div></div>
<p>于是 <span class="mono">TP</span>、<span class="mono">PP</span>、<span class="mono">EP</span>、<span class="mono">DP</span> 只是<strong>同一个类的不同实例</strong>，分别建立在不同的 rank 布局上：TP 组把单机 8 卡编为一组、PP 组把跨机的对应 rank 串成阶段、EP 组覆盖所有专家 rank、DP 组把若干副本编为一组。它们因此天然<strong>可组合</strong>——一个部署完全可以是 TP=8 × PP=2 × DP=4，每张卡同时属于多个 GroupCoordinator，在不同的组里走不同的集合通信。</p>
<p>把四招合起来看一个具体例子：假设手上有 64 张 GPU，配成 <strong>TP=8 × PP=2 × DP=4</strong>。每 8 张卡（一台机器内）组成一个 TP 组，靠 NVLink 做每层一次的高频 all_reduce；相邻两个 TP 组首尾相接组成 2 个 PP 阶段，跨机只在阶段交界 send/recv 激活——这样 16 张卡才拼出<strong>一份完整模型副本</strong>；再把这样的副本复制 4 份组成 DP，请求由第23课的 dp-controller 分发给这 4 份副本。于是<strong>同一张卡同时身处三个 GroupCoordinator</strong>：它的 TP 组、它的 PP 组、它的 DP 组，每个组各跑各的集合通信，互不干扰。8×2×4=64，恰好用满全部 GPU。</p>
<p>挑选组合时的取舍始终绕着四件事：<strong>通信量</strong>（TP 最高、DP 最低）、<strong>链路要求</strong>（TP 要 NVLink，PP/DP 普通链路即可）、<strong>流水线气泡</strong>（只有 PP 有，靠微批次缓解）、<strong>显存</strong>（TP/PP/EP 都省显存，DP 不省只换吞吐）。一般经验法则就是：先用 TP 吃满单机内的高速链路，再用 PP 跨机扩展模型规模，MoE 用 EP 把专家摊开，最后用 DP 复制整套换吞吐——四个 GroupCoordinator 各司其职，又彼此叠乘成一套完整的部署拓扑。</p>
<div class="flow"><div class="node">TP 部分层输出</div><div class="arrow">→</div><div class="node">all_reduce 求和</div><div class="arrow">→</div><div class="node">完整层输出</div></div>
<div class="flow"><div class="node">EP token 批</div><div class="arrow">→</div><div class="node">all_to_all 路由到专家</div><div class="arrow">→</div><div class="node">汇总结果</div></div>
<div class="flow"><div class="node">PP 阶段 i 激活</div><div class="arrow">→</div><div class="node">send / recv 给下一阶段</div><div class="arrow">→</div><div class="node">阶段 i+1 继续</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="每种并行各自的通信：TP 用 all-reduce、EP 用 all-to-all、PP 用 send-recv、DP 几乎不通信">
    <text x="24" y="32" style="font-weight:700;fill:var(--muted)">每种并行各自通信什么</text>
    <rect x="24" y="50" width="120" height="40" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="84" y="75" text-anchor="middle" style="font-weight:700;fill:var(--blue)">TP</text>
    <text x="156" y="76" style="fill:var(--muted)">→</text>
    <rect x="180" y="50" width="190" height="40" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="275" y="75" text-anchor="middle" class="mono" style="font-size:12px">all-reduce 求和</text>
    <text x="392" y="75" style="fill:var(--faint);font-size:12px">每层一次 · 最贵 · 需 NVLink</text>
    <rect x="24" y="110" width="120" height="40" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="84" y="135" text-anchor="middle" style="font-weight:700;fill:var(--amber)">EP</text>
    <text x="156" y="136" style="fill:var(--muted)">→</text>
    <rect x="180" y="110" width="190" height="40" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="275" y="135" text-anchor="middle" class="mono" style="font-size:12px">all-to-all 路由</text>
    <text x="392" y="135" style="fill:var(--faint);font-size:12px">随 token × top-k 增长</text>
    <rect x="24" y="170" width="120" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="84" y="195" text-anchor="middle" style="font-weight:700;fill:var(--teal)">PP</text>
    <text x="156" y="196" style="fill:var(--muted)">→</text>
    <rect x="180" y="170" width="190" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="275" y="195" text-anchor="middle" class="mono" style="font-size:12px">send · recv 激活</text>
    <text x="392" y="195" style="fill:var(--faint);font-size:12px">仅在阶段交界 · 便宜</text>
    <rect x="24" y="230" width="120" height="40" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="84" y="255" text-anchor="middle" style="font-weight:700;fill:var(--purple)">DP</text>
    <text x="156" y="256" style="fill:var(--muted)">→</text>
    <rect x="180" y="230" width="190" height="40" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="275" y="255" text-anchor="middle" class="mono" style="font-size:12px">几乎不通信</text>
    <text x="392" y="255" style="fill:var(--faint);font-size:12px">每步 ≈ 0 · 链路无所谓</text>
  </svg>
  <div class="figcap"><b>图 2 · 各自的通信</b> — 每根主轴对应一种集合通信：TP 每层做 all-reduce 求部分和（最贵、需 NVLink），EP 用两次 all-to-all 把 token 路由到专家再送回，PP 只在阶段交界 send/recv 激活，DP 每步近乎零通信。</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/distributed/parallel_state.py ::GroupCoordinator</span><span class="ln">一个统一抽象：TP/PP/EP/DP 都是它在不同 rank 布局上的实例</span></div><pre>class GroupCoordinator:
    # ONE process group; TP / PP / EP / DP are each an instance over a rank layout
    rank: int             # this process's global rank
    ranks: list           # the global ranks that belong to THIS group
    world_size: int       # number of ranks in this group
    rank_in_group: int    # position within the group

    def all_reduce(self, input_):              # TP: sum each rank's partial layer output
        ...
    def all_gather(self, input_, dim=-1):      # gather shards (e.g. vocab-parallel logits)
        ...
    def all_to_all_single(self, output, input):  # EP: route tokens to their expert owners
        ...
    @property
    def next_rank(self):   # PP: the downstream pipeline stage (send activations here)
        ...
    @property
    def prev_rank(self):   # PP: the upstream pipeline stage (recv activations from here)
        ...</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/distributed/parallel_state.py ::GroupCoordinator.all_reduce</span><span class="ln">把张量在本组（如 TP 组）内求和，人人拿到同一结果</span></div><pre>def all_reduce(self, input_: torch.Tensor) -&gt; torch.Tensor:
    # sum `input_` across every rank in THIS group (e.g. the TP group),
    # so each rank ends with the same reduced tensor.
    if self.world_size == 1:        # single GPU: nothing to reduce
        return input_
    ...   # dispatch to the group's all-reduce (custom op / NCCL)
    return input_</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>TP</strong> 切<strong>一层之内的权重矩阵</strong>（q/k/v、MLP 列、词表，第25/37课），每层用 <span class="mono">all_reduce</span>/<span class="mono">all_gather</span> 合并；通信最频繁，需 NVLink、通常单机内。</li>
<li><strong>PP</strong> 切<strong>层</strong>成连续阶段（第23课），只在阶段交界 <span class="mono">send</span>/<span class="mono">recv</span> 激活；链路便宜，但有流水线气泡，靠微批次填满。</li>
<li><strong>EP</strong> 切 MoE 的<strong>专家</strong>（第34课），每步两次 <span class="mono">all_to_all</span> 把 token 路由到专家再送回；通信随 token×top-k 增长，第47课接续大规模 EP + EPLB。</li>
<li><strong>DP</strong> <strong>复制模型、切请求</strong>（第23课 dp-controller），每步近乎零通信但每副本要装整份模型；Attention-DP 是"注意力 DP + 专家 EP"的组合变体。</li>
<li><strong>统一与组合</strong>：四者都是同一个 <span class="mono">GroupCoordinator</span>（带 <span class="mono">rank</span>/<span class="mono">ranks</span>/<span class="mono">world_size</span>/<span class="mono">rank_in_group</span>）的不同实例，可叠乘组合成 TP=8 × PP=2 × DP=4。</li>
<li><strong>怎么选</strong>：装不下→先 TP（单机内）再 PP（跨机）；MoE 专家撑爆显存→EP；装得下只想要更高吞吐→DP。真实部署常按这套优先级把几招叠起来用。</li>
<li><strong>取舍四要素</strong>：通信量、链路要求、流水线气泡、显存占用。TP 通信最贵但省单层算力与显存，DP 通信最省但每副本占整份显存，PP 链路便宜却有气泡，EP 通信随 token×top-k 浮动且要防负载不均。</li>
</ul></div>
""", "en": r"""
<p class="lead">When a model is too big to fit—or too heavy to compute—on a single GPU, we spread the "model" or the "requests" across many GPUs. There are exactly four axes to spread along: <strong>TP (tensor parallel)</strong>, <strong>PP (pipeline parallel)</strong>, <strong>EP (expert parallel)</strong>, and <strong>DP (data parallel)</strong>. This lesson makes all four clear: what each one splits, what it must communicate every step, and what link it demands. Finally it reveals SGLang's key unification—all four are just instances of the same <span class="mono">GroupCoordinator</span> abstraction over different rank layouts, and they <strong>compose</strong>. Once you grasp these four axes and how they combine, you can read any SGLang deployment command and tell exactly what its <span class="mono">tp_size</span>, <span class="mono">pp_size</span>, <span class="mono">ep_size</span>, and <span class="mono">dp_size</span> numbers are splitting and what each one costs.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture inference as a factory handling a flood of orders. There are four ways to scale capacity:</p>
<p><strong>TP</strong> is like having four workers <strong>on one assembly line</strong> each tighten different screws on "building one machine" at the same time—each does a slice, and the parts must be <strong>merged</strong> into a finished product (that merge is all-reduce). Merging is frequent, so the four must sit shoulder to shoulder (a fast neighbor link like NVLink).</p>
<p><strong>PP</strong> is like splitting the whole line into <strong>consecutive workshops</strong>: workshop one builds the chassis, then <strong>hands off</strong> to workshop two for the shell, then to workshop three for QC. Only half-products cross workshop boundaries, so a cheap link is fine; but at startup the later shops wait, and at the end the earlier shops idle—that idle waste is the "pipeline bubble".</p>
<p><strong>EP</strong> is like having many <strong>specialist masters</strong> (experts); each order needs only two of them. The system first <strong>dispatches</strong> each order to the workshop holding its chosen masters (all-to-all), and after they finish <strong>sends it back</strong> (a second all-to-all). The more masters, and the more each order needs, the more shuttling.</p>
<p><strong>DP</strong> is like simply <strong>replicating</strong> the whole factory and splitting orders evenly so each factory works independently—almost no shuttling, but every factory must own the full set of equipment (the whole model's weights).</p>
<p>In practice all four are usually used <strong>at once</strong>: replicate one huge factory into several (DP), inside each factory split the line into consecutive workshops (PP), inside each workshop let neighboring workers share one operation (TP), and gather the many specialist masters into rooms by trade (EP). A single worker thus <strong>belongs to several teams at the same time</strong>—his "neighbor team", "workshop team", and "replica team" each handle their own shuttling. This freedom to stack them is exactly the "composability" the lesson ends on.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>The four parallelisms answer two different questions. <strong>TP / PP / EP</strong> split the <strong>model itself</strong>: TP splits the weight matrices within a layer (the column/row and vocab splits of lessons 25 & 37), PP splits the boundaries between layers (lesson 23), EP splits the set of MoE experts (lesson 34). <strong>DP</strong> doesn't split the model but the <strong>data (requests)</strong>: replicate the model, share out requests (the dp-controller of lesson 23). In a sentence: the first three "share one model's size and compute", while DP "shares throughput". SGLang also has a twist called <strong>Attention-DP</strong>: DP-shard the attention and EP-shard the experts, both at once. The elegant part: these four are not four separate implementations—they are the same <span class="mono">GroupCoordinator</span> running different collectives over different rank groups, so they naturally compose, e.g. TP=8 × PP=2 × DP=4.</p>
<p>So when do you reach for which? Follow the bottleneck: if the model is <strong>too big to fit on one card</strong>, first apply TP to thin each layer; if a single node doesn't have enough cards, add PP to spread layers across more machines. If it is a <strong>many-expert MoE</strong> whose total parameters blow past memory, apply EP to scatter the experts. If the model already <strong>fits</strong> and you just want <strong>more throughput</strong>, apply DP to replicate it and share out requests. Almost every real deployment is not single-choice but stacks several by this priority: TP saturates the fast intra-node link, PP scales model size across nodes, EP spreads experts, DP replicates for throughput.</p>
</div>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="Four ways to split a model: TP splits a layer's matrices, PP splits the layer stack into stages, EP spreads experts across cards, DP replicates the model and splits requests">
    <rect x="20" y="20" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="36" y="46" style="font-weight:700;fill:var(--blue)">TP tensor</text>
    <text x="36" y="66" style="fill:var(--muted);font-size:12px">each holds a ¼ slice of a layer</text>
    <rect x="40" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="62" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼ lyr</text>
    <rect x="110" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="132" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼ lyr</text>
    <rect x="180" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="202" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼ lyr</text>
    <rect x="250" y="92" width="44" height="44" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="272" y="119" text-anchor="middle" class="mono" style="font-size:11px">¼ lyr</text>
    <rect x="410" y="20" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="426" y="46" style="font-weight:700;fill:var(--teal)">PP pipeline</text>
    <text x="426" y="66" style="fill:var(--muted);font-size:12px">each holds consecutive layers (a stage)</text>
    <rect x="430" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="452" y="119" text-anchor="middle" class="mono" style="font-size:11px">stg1</text>
    <text x="487" y="119" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="500" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="522" y="119" text-anchor="middle" class="mono" style="font-size:11px">stg2</text>
    <text x="557" y="119" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="570" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="592" y="119" text-anchor="middle" class="mono" style="font-size:11px">stg3</text>
    <text x="627" y="119" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="640" y="92" width="44" height="44" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="662" y="119" text-anchor="middle" class="mono" style="font-size:11px">stg4</text>
    <rect x="20" y="170" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="36" y="196" style="font-weight:700;fill:var(--amber)">EP expert</text>
    <text x="36" y="216" style="fill:var(--muted);font-size:12px">each holds some experts</text>
    <rect x="40" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="62" y="269" text-anchor="middle" class="mono" style="font-size:11px">exp</text>
    <rect x="110" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="132" y="269" text-anchor="middle" class="mono" style="font-size:11px">exp</text>
    <rect x="180" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="202" y="269" text-anchor="middle" class="mono" style="font-size:11px">exp</text>
    <rect x="250" y="242" width="44" height="44" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="272" y="269" text-anchor="middle" class="mono" style="font-size:11px">exp</text>
    <rect x="410" y="170" width="370" height="130" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="426" y="196" style="font-weight:700;fill:var(--purple)">DP data</text>
    <text x="426" y="216" style="fill:var(--muted);font-size:12px">each holds a full model replica</text>
    <rect x="430" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="452" y="269" text-anchor="middle" class="mono" style="font-size:11px">full</text>
    <rect x="500" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="522" y="269" text-anchor="middle" class="mono" style="font-size:11px">full</text>
    <rect x="570" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="592" y="269" text-anchor="middle" class="mono" style="font-size:11px">full</text>
    <rect x="640" y="242" width="44" height="44" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="662" y="269" text-anchor="middle" class="mono" style="font-size:11px">full</text>
  </svg>
  <div class="figcap"><b>Fig 1 · four ways to split</b> — same 4 cards: TP gives each a ¼ slice of a layer (split within a layer), PP gives each consecutive layers i.e. a stage (split the stack), EP gives each some experts, DP gives each a full model replica and only splits requests.</div>
</div>

<h2>TP tensor parallel: split the matrices "within a layer"</h2>
<p><strong>TP (tensor parallel)</strong> shards each layer's <strong>weight matrices</strong> along some dimension across the ranks of a group. The classic cases: the q/k/v projections column-split (lesson 25), the MLP up-projection column-split (lesson 37), and the vocab row-split. Each rank holds only one slice of the matrix, so it computes only a <strong>partial result</strong>; to get the full layer output the partial sums must be <strong>merged</strong>—a column+row combo uses one <span class="mono">all_reduce</span> sum, while vocab-parallel logits use <span class="mono">all_gather</span> to concatenate.</p>
<p>TP's signature is that it thins out both the <strong>compute and memory of a single layer</strong>: a giant matrix that won't fit on one card is split into slices, stored and computed separately. The price is that <strong>every layer</strong> communicates (once forward, once backward)—extremely frequent. So GPUs in a TP group must use a <strong>fast neighbor link</strong> (NVLink / NVSwitch) and usually stay <strong>within one node</strong>; doing TP across machines gets choked by slow links.</p>
<div class="layers"><div class="layer">Rank0: holds a 1/4 slice of every layer's weights</div><div class="layer">Rank1: holds a 1/4 slice of every layer's weights</div><div class="layer">Rank2: holds a 1/4 slice of every layer's weights</div><div class="layer">Rank3: holds a 1/4 slice of every layer's weights</div></div>
<p>Note the diagram: under TP <strong>every rank runs "vertically" through all layers</strong>, just holding a thin strip of each. This is the fundamental difference from PP—TP splits the width of a layer, PP splits the depth of the stack.</p>
<p>Concrete numbers help: a 70B model's per-layer weights are easily hundreds of MB; at TP=8 each card stores only 1/8 and computes only 1/8, but <strong>every single layer</strong> must run an <span class="mono">all_reduce</span> across those 8 cards to sum the partial results. Because the communication is so dense, TP's degree usually <strong>does not exceed the number of cards in one node</strong> (e.g. 8); to scale further you switch to PP or DP, otherwise frequent cross-node all_reduce would outweigh the compute it saves. Remember TP's trade-off in one line: it thins both "per-layer compute" and "per-layer memory", at the cost of "communicating every layer—the most expensive—over a fast neighbor link".</p>
<p>Pin this to a real command: <span class="mono">--tp-size 8</span> shards <strong>every layer</strong>'s matrices over 8 GPUs and does <strong>one</strong> <span class="mono">all_reduce</span> per layer to sum the partial results—because it <strong>communicates every layer</strong>, the TP group needs a fast intra-node link (NVLink); by contrast <span class="mono">--dp-size</span> replicates the whole model and <strong>barely communicates</strong> per step, so a slow link is fine. That is exactly where the rule "TP saturates intra-node NVLink, DP barely communicates" comes from.</p>

<h2>PP pipeline parallel: split "layers" into a relay of stages</h2>
<p><strong>PP (pipeline parallel)</strong> splits the model's <strong>layers</strong> in order into consecutive <strong>stages</strong>, placed on different GPUs. Layers 1…k on rank0, k+1…2k on rank1, and so on. In the forward pass activations <strong>flow</strong> from one stage to the next (send to <span class="mono">next_rank</span>, receive from <span class="mono">prev_rank</span>, lesson 23); in the backward pass gradients flow the opposite way.</p>
<p>PP communicates only at <strong>stage boundaries</strong>, far less than TP, so an ordinary link (even cross-node) suffices. But it has an inherent problem: if you feed only one batch, the earlier stages finish and idle while waiting for the later ones—low utilization. The fix is to split a large batch into many <strong>micro-batches</strong> streamed in continuously like a pipeline, keeping all stages busy; even so, the start and end leave stages idle—this waste is the <strong>pipeline bubble</strong>. PP thins out the <strong>memory from layer count</strong> (each card holds only some layers), not the compute of a single layer.</p>
<p>The bubble's share is roughly proportional to <strong>number of stages ÷ number of micro-batches</strong>: more stages or fewer micro-batches mean a larger idle fraction at the start and end; cranking up the micro-batch count thins that waste to an acceptable level. PP's biggest advantage is that it <strong>moves one copy of activations only at boundaries</strong>—the volume scales with the hidden dimension (plus batch and sequence length), not with layer count—so it fits the case of "too many layers to fit in one node, and only slow links between machines": use PP to lay the model out <strong>vertically</strong> across machines, then TP for the <strong>horizontal</strong> fine split inside each machine. The two are exactly complementary.</p>

<h2>EP expert parallel: spread MoE "experts" across ranks</h2>
<p><strong>EP (expert parallel)</strong> is made for MoE (lesson 34). An MoE layer has many experts (FFNs), but each token's router picks only top-k of them to compute. EP places different <strong>experts</strong> on different ranks: each step, a first <span class="mono">all_to_all</span> sends <strong>every token</strong> to the rank that owns "the expert it chose", and after the experts compute, a second <span class="mono">all_to_all</span> brings the results back to the original rank.</p>
<p>Why does EP use <span class="mono">all_to_all</span> rather than TP's <span class="mono">all_reduce</span>? Because each token's destination is <strong>different</strong>—token A picked expert 3, token B picked expert 7—it is a "point-to-point reshuffle", not an "everyone-sum". <span class="mono">all_to_all</span> is the collective built precisely for "each sends its own, each receives its own": every rank simultaneously sends a different chunk to all other ranks and receives a chunk from each, which is exactly "dispatch each token to the rank owning its chosen expert".</p>
<p>EP thins out the <strong>experts' memory</strong>—hundreds or thousands of experts stored across ranks, sharing total parameters that won't fit on one card. Its communication grows with <strong>tokens × top-k</strong>: the larger the batch and the more experts each token picks, the more the all-to-all shuttles. SGLang's Attention-DP often pairs with EP: the attention part DP-replicated, the expert part EP-sharded, both in one deployment—yet another instance of "different GroupCoordinators each doing their job". Large-scale EP and expert load balancing (EPLB) are covered in <strong>lesson 47</strong>.</p>
<p>Another key difference between EP and TP is <strong>whether the communication volume floats with load</strong>: TP's per-layer all_reduce has a fixed size (it depends only on the hidden dimension), whereas EP's all_to_all size varies with the <strong>actual tokens × top-k</strong>—the larger the batch and the more experts each token picks, the more data moves. Trickier still, <strong>expert load is often uneven</strong>—some popular experts are chosen by many tokens while others sit nearly idle, so the rank holding a hot expert becomes the bottleneck for the whole step. Reshuffling experts across ranks to balance load is exactly what lesson 47's EPLB tackles.</p>

<h2>DP data parallel + the unified GroupCoordinator</h2>
<p><strong>DP (data parallel)</strong> is the simplest: <strong>replicate</strong> the whole model to each replica and split the <strong>requests</strong> across replicas to be handled independently (the dp-controller of lesson 23 dispatches them). Each step is nearly <strong>zero communication</strong> (only training needs a gradient sync after backward; in serving the replicas are nearly independent), at the price that every replica must hold the <strong>whole model</strong>—it buys throughput, not memory savings.</p>
<p>SGLang's <strong>Attention-DP</strong> deserves its own note: in large MoE deployments the attention part has few parameters but most fears redundant KV-cache use and duplicated compute, so it is DP-sharded—each replica handles the attention of its own batch of requests; the expert part is huge and must spread its memory, so it is EP-sharded. The two splits each govern a portion of <strong>one model</strong>—attention goes DP, FFN/experts go EP—without conflict. This is again "different GroupCoordinators each doing their job over different rank layouts", only this time DP and EP are combined on different submodules of one model.</p>
<table class="t"><tr><th>Axis</th><th>Splits what</th><th>Communicates per step</th><th>Link need</th></tr>
<tr><td><span class="mono">TP</span></td><td>Each layer's weight matrices (within a layer)</td><td>One all-reduce / all-gather per layer</td><td>Very high, NVLink, usually single node</td></tr>
<tr><td><span class="mono">PP</span></td><td>The model's layers (consecutive stages)</td><td>Only send / recv activations at boundaries</td><td>Low, ordinary link, can cross nodes</td></tr>
<tr><td><span class="mono">EP</span></td><td>The set of MoE experts</td><td>Two all-to-all (by token×top-k)</td><td>Mid-high, grows with token count</td></tr>
<tr><td><span class="mono">DP</span></td><td>Requests (full model replicated)</td><td>Near zero per step (only training grad sync)</td><td>Very low</td></tr></table>
<p>Here is the real unification: SGLang does not write four separate communication codepaths—it expresses all four with <strong>one abstraction</strong>, <span class="mono">GroupCoordinator</span>. A <span class="mono">GroupCoordinator</span> is "one process group" carrying <span class="mono">rank</span> (this process's global id), <span class="mono">ranks</span> (the global ranks belonging to this group), <span class="mono">world_size</span> (the group's size), <span class="mono">rank_in_group</span> (position within the group), and exposing <span class="mono">all_reduce</span> / <span class="mono">all_gather</span> / <span class="mono">all_to_all_single</span> / <span class="mono">next_rank</span> / <span class="mono">prev_rank</span>.</p>
<div class="cols"><div class="col"><strong>Split the "weights" (TP / PP / EP)</strong><br/>Scatter one model's matrices, layers, or experts across cards, each holding only part of the parameters; saves memory and per-card compute, but collectives must stitch results back.</div><div class="col"><strong>Split the "data" (DP)</strong><br/>Replicate the whole model to each replica and split by request; each replica independent with almost no communication—buys throughput, not memory.</div></div>
<p>So <span class="mono">TP</span>, <span class="mono">PP</span>, <span class="mono">EP</span>, <span class="mono">DP</span> are just <strong>different instances of the same class</strong>, each built over a different rank layout: a TP group bundles 8 cards in a node, a PP group strings corresponding ranks across nodes into stages, an EP group spans all expert ranks, a DP group bundles several replicas. They are therefore naturally <strong>composable</strong>—a deployment can be TP=8 × PP=2 × DP=4, where each card belongs to several GroupCoordinators at once, running different collectives in different groups.</p>
<p>Put all four together in a concrete example: say you have 64 GPUs configured as <strong>TP=8 × PP=2 × DP=4</strong>. Every 8 cards (within one machine) form a TP group doing the per-layer all_reduce over NVLink; two adjacent TP groups chained head-to-tail form 2 PP stages, sending/receiving activations across machines only at the boundary—so 16 cards together make <strong>one full model replica</strong>; then replicate that 4 times to form DP, with the lesson-23 dp-controller dispatching requests to the 4 replicas. Thus <strong>one card sits inside three GroupCoordinators at once</strong>: its TP group, its PP group, and its DP group, each running its own collectives without interfering. 8×2×4=64, exactly filling all the GPUs.</p>
<p>Choosing a combination always revolves around four things: <strong>communication volume</strong> (TP highest, DP lowest), <strong>link requirement</strong> (TP needs NVLink, PP/DP an ordinary link), <strong>pipeline bubble</strong> (only PP has it, eased by micro-batches), and <strong>memory</strong> (TP/PP/EP all save memory, DP does not—it only buys throughput). The rule of thumb: first use TP to saturate the fast intra-node link, then PP to scale model size across nodes, EP to spread MoE experts, and finally DP to replicate the whole thing for throughput—four GroupCoordinators each doing their job, stacking into one complete deployment topology.</p>
<div class="flow"><div class="node">TP partial layer output</div><div class="arrow">→</div><div class="node">all_reduce sum</div><div class="arrow">→</div><div class="node">full layer output</div></div>
<div class="flow"><div class="node">EP token batch</div><div class="arrow">→</div><div class="node">all_to_all route to experts</div><div class="arrow">→</div><div class="node">gathered result</div></div>
<div class="flow"><div class="node">PP stage i activations</div><div class="arrow">→</div><div class="node">send / recv to next stage</div><div class="arrow">→</div><div class="node">stage i+1 continues</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="What each parallelism communicates: TP uses all-reduce, EP uses all-to-all, PP uses send-recv, DP barely communicates">
    <text x="24" y="32" style="font-weight:700;fill:var(--muted)">what each parallelism communicates</text>
    <rect x="24" y="50" width="120" height="40" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="84" y="75" text-anchor="middle" style="font-weight:700;fill:var(--blue)">TP</text>
    <text x="156" y="76" style="fill:var(--muted)">→</text>
    <rect x="180" y="50" width="190" height="40" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="275" y="75" text-anchor="middle" class="mono" style="font-size:12px">all-reduce (sum)</text>
    <text x="392" y="75" style="fill:var(--faint);font-size:12px">every layer · priciest · NVLink</text>
    <rect x="24" y="110" width="120" height="40" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="84" y="135" text-anchor="middle" style="font-weight:700;fill:var(--amber)">EP</text>
    <text x="156" y="136" style="fill:var(--muted)">→</text>
    <rect x="180" y="110" width="190" height="40" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="275" y="135" text-anchor="middle" class="mono" style="font-size:12px">all-to-all (route)</text>
    <text x="392" y="135" style="fill:var(--faint);font-size:12px">grows with token × top-k</text>
    <rect x="24" y="170" width="120" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="84" y="195" text-anchor="middle" style="font-weight:700;fill:var(--teal)">PP</text>
    <text x="156" y="196" style="fill:var(--muted)">→</text>
    <rect x="180" y="170" width="190" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="275" y="195" text-anchor="middle" class="mono" style="font-size:12px">send · recv (acts)</text>
    <text x="392" y="195" style="fill:var(--faint);font-size:12px">only at stage edges · cheap</text>
    <rect x="24" y="230" width="120" height="40" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="84" y="255" text-anchor="middle" style="font-weight:700;fill:var(--purple)">DP</text>
    <text x="156" y="256" style="fill:var(--muted)">→</text>
    <rect x="180" y="230" width="190" height="40" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="275" y="255" text-anchor="middle" class="mono" style="font-size:12px">almost none</text>
    <text x="392" y="255" style="fill:var(--faint);font-size:12px">≈ 0 per step · any link</text>
  </svg>
  <div class="figcap"><b>Fig 2 · what each communicates</b> — each axis maps to one collective: TP does an all-reduce every layer to sum partials (priciest, needs NVLink), EP does two all-to-all to route tokens to experts and back, PP only send/recv activations at stage edges, DP is near-zero communication per step.</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/distributed/parallel_state.py ::GroupCoordinator</span><span class="ln">one unified abstraction: TP/PP/EP/DP are all instances over different rank layouts</span></div><pre>class GroupCoordinator:
    # ONE process group; TP / PP / EP / DP are each an instance over a rank layout
    rank: int             # this process's global rank
    ranks: list           # the global ranks that belong to THIS group
    world_size: int       # number of ranks in this group
    rank_in_group: int    # position within the group

    def all_reduce(self, input_):              # TP: sum each rank's partial layer output
        ...
    def all_gather(self, input_, dim=-1):      # gather shards (e.g. vocab-parallel logits)
        ...
    def all_to_all_single(self, output, input):  # EP: route tokens to their expert owners
        ...
    @property
    def next_rank(self):   # PP: the downstream pipeline stage (send activations here)
        ...
    @property
    def prev_rank(self):   # PP: the upstream pipeline stage (recv activations from here)
        ...</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/distributed/parallel_state.py ::GroupCoordinator.all_reduce</span><span class="ln">sum a tensor across this group (e.g. TP), everyone gets the same result</span></div><pre>def all_reduce(self, input_: torch.Tensor) -&gt; torch.Tensor:
    # sum `input_` across every rank in THIS group (e.g. the TP group),
    # so each rank ends with the same reduced tensor.
    if self.world_size == 1:        # single GPU: nothing to reduce
        return input_
    ...   # dispatch to the group's all-reduce (custom op / NCCL)
    return input_</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>TP</strong> splits the <strong>weight matrices within a layer</strong> (q/k/v, MLP columns, vocab; lessons 25/37), merging each layer with <span class="mono">all_reduce</span>/<span class="mono">all_gather</span>; the most frequent communication, needs NVLink, usually single node.</li>
<li><strong>PP</strong> splits <strong>layers</strong> into consecutive stages (lesson 23), only <span class="mono">send</span>/<span class="mono">recv</span> activations at boundaries; cheap link, but has pipeline bubbles filled by micro-batches.</li>
<li><strong>EP</strong> splits MoE <strong>experts</strong> (lesson 34), two <span class="mono">all_to_all</span> per step to route tokens to experts and back; communication grows with token×top-k, continued by large-scale EP + EPLB in lesson 47.</li>
<li><strong>DP</strong> <strong>replicates the model and splits requests</strong> (dp-controller, lesson 23), near-zero per-step communication but every replica holds the whole model; Attention-DP is the "attention DP + expert EP" combined twist.</li>
<li><strong>Unification & composition</strong>: all four are different instances of the same <span class="mono">GroupCoordinator</span> (carrying <span class="mono">rank</span>/<span class="mono">ranks</span>/<span class="mono">world_size</span>/<span class="mono">rank_in_group</span>), composable into TP=8 × PP=2 × DP=4.</li>
<li><strong>How to choose</strong>: won't fit → TP first (intra-node) then PP (cross-node); MoE experts blow past memory → EP; fits but you just want more throughput → DP. Real deployments stack several by this priority.</li>
<li><strong>Four trade-off factors</strong>: communication volume, link requirement, pipeline bubble, memory footprint. TP communicates the most but saves per-layer compute and memory; DP communicates the least but each replica holds the whole model; PP has a cheap link but a bubble; EP's communication floats with token×top-k and must guard against load imbalance.</li>
</ul></div>
"""}
LESSON_47 = {"zh": r"""
<p class="lead">在第34课我们认识了 MoE：路由器为每个 token 挑选 top-k 个专家；在第46课我们看到专家并行（EP）把成百上千个专家分散到许多张 GPU 上。但真实流量并不是均匀的——有的专家天天爆满，有的几乎闲置。本课要讲的，就是当 EP 遇到"忙闲不均"时会发生什么，以及 SGLang 用 <span class="mono">EPLBManager</span>（专家并行负载均衡器）如何把这块"塞车"的瓶颈拉平。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一座大型连锁餐厅有 256 个窗口（专家），分布在 8 个厨房（GPU）里，每个厨房放 32 个窗口。顾客（token）进门后，门口的引导员（路由器）按菜品把每位顾客指派给最合适的 top-k 个窗口。问题是：网红爆款"麻辣烫窗口"门口排起长龙，而"白粥窗口"几乎没人。偏偏餐厅有个铁规矩——<strong>所有厨房必须同时上齐菜，最慢的那个厨房没出完，全场都得干等</strong>。于是只要某个厨房里挤着一个爆款窗口，其余 7 个厨房的厨师全都袖手旁观，整体吞吐被这一个热点活活拖垮。</p>
<p>聪明的店长会怎么做？他每隔一段时间统计各窗口的实际客流，然后<strong>重新安排窗口到厨房的摆放</strong>：把冷门窗口挪走腾地方，把爆款窗口<strong>复制成好几份</strong>分到不同厨房，让排队的人分流到各处。这样每个厨房每一轮接待的顾客数量就拉平了，最慢厨房不再拖后腿。这位"店长"就是 EPLB。</p>
<p>这个类比里有三个细节值得记住。其一，店长<strong>不能只靠开业那天的预测</strong>排座位——今天火的菜明天可能就凉了，所以他必须持续盯客流、动态调整。其二，<strong>光是互换窗口位置救不了真正的爆款</strong>：如果"麻辣烫"一家就占了全店四成客流，无论它坐哪个厨房，那个厨房都会爆，唯一的解法是开分店（复制）。其三，<strong>调整本身有成本</strong>——搬窗口、布置新厨房都要停工片刻，所以店长不会每分钟都重排，而是攒够一段时间的客流数据、确认格局真的变了才动手。这三点，恰好对应 EPLB 的"周期性测量、复制热点、低频重排"。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>EP 解决的是"专家太多放不下一张卡"的容量问题，但它本身不解决"负载倾斜"的效率问题。EP 的一步 all-to-all 是一道同步墙：<strong>必须等最忙的那张卡算完，这一步才能结束</strong>。只要负载倾斜，少数热点专家所在的 GPU 就成为木桶最短的那块板，其它 GPU 白白空转。EPLB 的核心思想是：负载不是固定的，但可以被<strong>测量</strong>和<strong>重新放置</strong>。它周期性地量出每个专家收到多少 token，再解一个新的"专家→GPU"摆放方案把负载摊平——必要时复制热门专家。对 DeepSeek 这种几百专家规模的 MoE 服务，这一步不是锦上添花，而是能不能跑得起来的关键。</p>
<p>把这三课串起来看会更清楚：第34课讲的是 MoE 与 FusedMoE 路由——"一个 token 该交给哪些专家"；第46课讲的是 EP 与 all-to-all——"专家放在不同 GPU 上、token 该怎么跨卡传送"；本课要补上的，是"当传送的负载不均时，怎么把它重新摊平"。三者层层递进：有了 MoE 才有专家，有了 EP 才需要跨卡，有了倾斜才需要均衡。EPLB 站在最上层，把前两课搭好的舞台真正用满。理解了这条线，你就明白为什么大规模 MoE 推理是"路由 + 并行 + 均衡"三件事缺一不可的系统工程。</p>
</div>

<h2>为什么 EP 的负载天然倾斜？</h2>
<p>MoE 的路由器是学出来的，不是均匀分配的。训练让某些专家专精于高频模式（比如常见的语法结构、热门话题），这些专家在推理时自然会被更多 token 选中，形成"热点专家"；而另一些专家对应罕见模式，长期处于半闲置状态。这种<strong>长尾分布</strong>是 MoE 的固有特性，而非偶然。当我们用 EP 把专家平摊到各 GPU 时，热点专家恰好可能集中落在某几张卡上，于是这几张卡每一步要处理的 token 数远超平均值。</p>
<p>不妨用一组具体数字感受这种倾斜。假设一个 MoE 层有 256 个专家、top-k 取 8，一个批次里有 4096 个 token，那么这一层总共要分发 4096×8 = 32768 次专家调用。如果负载完全均匀，每个专家应当收到 128 次；可现实里排名最靠前的几个专家往往各收到上千次，而尾部的几十个专家加起来都不到几百次。把这 256 个专家按 EP 平摊到 8 张卡，每卡 32 个专家——只要排名前几的热点恰好落在同一张卡，这张卡这一步就要扛起好几倍于均值的计算量。倾斜不是"略微不均"，而常常是数量级的差距。</p>
<p>关键在于 EP 的执行模型：每一步前向都包含一次 all-to-all 通信——把各 token 发往它们各自专家所在的 GPU，算完再收回来。这是一道<strong>集合同步</strong>：所有 rank 必须在同一个屏障处会合。于是整步的耗时由<strong>最忙的那个 rank</strong>决定。如果 8 张卡里有 7 张处理 1000 个 token、1 张处理 4000 个 token，那么这一步的墙钟时间就按 4000 个算，前 7 张卡有 75% 的时间在等待。裸 EP 因此浪费了大量算力。</p>
<p>还有一个常被忽视的细节：负载倾斜不是静止的，它会<strong>随时间和输入分布漂移</strong>。同一个模型，面对代码生成、对话聊天、长文档摘要这三类请求，被激活的热点专家可能完全不同；甚至在同一段对话里，话题一变，热点也跟着迁移。这意味着任何"一次性算好、永久固定"的专家摆放都会很快过时。倾斜既是固有的，又是流动的——这正是为什么解决方案必须是<strong>周期性测量、动态调整</strong>，而不能是部署时拍一次脑袋定下来的静态配置。</p>

<h2>EPLB 到底测量和重排了什么？</h2>
<p>EPLB（Expert-Parallel Load Balancer，专家并行负载均衡器）做两件事：<strong>测量</strong>与<strong>重排</strong>。测量指的是统计每个专家在最近一段时间里实际收到了多少 token——这份"每专家负载"统计由专家分布记录器（expert-distribution recorder）在每次前向后累加。重排指的是基于这份统计，解一个新的<strong>专家→GPU 放置方案</strong>，目标是让每张 GPU 每步处理的 token 数尽量相等。</p>
<p>重排有两件武器。第一是<strong>迁移</strong>：把专家在 GPU 之间搬家，让冷热专家搭配摆放，避免热点扎堆。第二是<strong>复制热门专家</strong>：当某个专家实在太热、单卡装不下它的流量时，EPLB 会把它<strong>复制到多张 GPU 上</strong>，让指向它的 token 分摊到这几份副本中。复制是 EPLB 区别于简单"换位置"的精髓——它把一个不可分割的热点切成了可并行的多份。重排完成后，新的放置写回到 <span class="mono">expert_location</span>（专家→GPU 映射），后续的步骤就按这张新地图来路由。</p>
<p>为什么"换位置"不够、非得"复制"？设想最极端的情形：某一个专家独自就吃掉了全系统四成的 token。无论你把它放到哪张卡，那张卡都会爆满，其它卡再闲也帮不上忙——因为一个专家在物理上不可分割。这时唯一的出路就是把它做成多份副本，分散到多张卡上，让原本指向它的洪流被几条河道分流。可以把迁移理解为"重新洗牌座位"，把复制理解为"给热门窗口加开分店"：前者解决分布不均，后者解决单点过热。两者配合，才能真正把每张卡每步的 token 数压到接近平均。</p>
<p>这里也有取舍。复制热门专家要占用额外显存（多存一份专家权重），重排本身也要在 GPU 间搬运参数、有不小的开销。所以 EPLB 不会每一步都重排，而是<strong>攒一段时间的统计、确认负载格局确实变了，才动手</strong>。它在"跟得上负载漂移"与"别让重排开销盖过收益"之间取平衡，这也解释了为什么采集要高频而重排要低频。换句话说，复制的份数也不是越多越好：副本太少压不住热点，副本太多又白白吃掉本可留给 KV 缓存或更多并发的显存。EPLB 求解放置时，正是在"让每张卡负载尽量均衡"和"别为此付出过多显存与搬运代价"这两个目标之间找一个折中点。</p>

<h2>EPLBManager 的运转循环</h2>
<p>SGLang 用 <span class="mono">EPLBManager</span> 把上面的思想落地成一个清晰的循环。它挂在 <span class="mono">ModelRunner</span> 上，借助专家分布记录器持续采样。核心是两个钩子：<span class="mono">on_forward_pass_end()</span> 在<strong>每一次前向之后</strong>被调用，把"这一步每个专家命中了多少 token"累加进统计；<span class="mono">rebalance()</span> 则<strong>周期性</strong>地被触发，解出新的专家放置方案（热点可能被复制），然后更新 <span class="mono">expert_location</span>。注意分工：采集是高频、轻量的（每步都做），重排是低频、较重的（隔一段时间做一次），二者解耦，既能跟上负载变化，又不会让重排开销淹没正常推理。</p>
<p>为什么采集要做成"每步一次、极轻量"？因为只有把统计的颗粒度压到每一步，EPLB 才能看清负载的真实形状，而不是被某个瞬间的抖动误导。<span class="mono">on_forward_pass_end()</span> 做的事很克制：它不在前向的关键路径里做任何重活，只是把这一步各专家命中的计数累加到记录器里——一次简单的加法。正因为它足够便宜，才敢每步都做。这份持续累积的统计，就是 <span class="mono">rebalance()</span> 求解新放置时的唯一依据。</p>
<p>这条循环是有状态的、自适应的：负载在变，统计在累积，放置在周期性刷新。它把"静态的专家摆放"变成了"随流量演化的动态摆放"。对大规模 MoE 服务而言，这正是把裸 EP 的理论吞吐变成实际吞吐的关键一环。</p>
<p>把整个机制再压成一句话：EPLB 让专家的物理位置"跟着流量走"。它不改动模型一个参数、不改变路由器的选择逻辑，只是不断回答一个朴素的问题——"既然这些 token 注定要去这些专家，那把专家摆在哪些 GPU 上、各放几份，才能让每张卡每步的活儿一样多？"答案随流量变化而变化，于是放置也随之刷新。这种"测量—求解—放置—再测量"的闭环，本质上是一个轻量级的在线优化器，嵌在推理主循环的缝隙里默默工作。</p>
<p>从工程视角看，这套设计的优雅之处在于<strong>关注点分离</strong>。模型的前向逻辑完全不需要知道 EPLB 的存在——它只管按当前的 <span class="mono">expert_location</span> 路由 token；而 <span class="mono">EPLBManager</span> 像一个旁挂的观察者，安静地在每步结束时记一笔账，隔一阵子悄悄把地图换新。前向与均衡两条线解耦，使得均衡策略可以独立演进、甚至替换成不同的求解算法，而不必改动模型代码。这种"可插拔、为规模而设计"的思路，正是 SGLang 整体架构的底色，我们会在第61、62课进一步展开。简言之，模型只负责算，均衡只负责摆，二者各司其职、互不打扰。</p>
<p>最后值得强调的是规模效应：专家越多、GPU 越多，负载倾斜带来的浪费就越严重，EPLB 的价值也就越大。在几十张卡、几百个专家的 DeepSeek 级部署里，没有负载均衡的裸 EP 几乎无法把硬件喂满；而有了周期性重平衡，整个集群的算力利用率才能真正逼近理论上限。EPLB 不是可有可无的优化，而是大规模 MoE 推理能否经济地跑起来的前提。</p>

<div class="flow"><div class="node">路由 token 到 top-k 专家</div><div class="arrow">→</div><div class="node">测量每专家负载</div><div class="arrow">→</div><div class="node">重排放置（复制热点）</div><div class="arrow">→</div><div class="node">更平坦的 GPU 负载</div></div>

<div class="cols"><div class="col"><strong>倾斜（裸 EP）</strong><br>8 张卡，1 个热点专家挤在 GPU0。这一步必须等 GPU0 处理完 4000 个 token，其余 7 张卡各 1000 个，处理完只能干等。<span class="mono">最忙者决定整步耗时</span>，75% 算力在空转。</div><div class="col"><strong>平衡（EPLB 之后）</strong><br>热点专家被复制到 3 张卡上分流，冷热专家重新搭配摆放。每张 GPU 每步约 1500 个 token，墙时间由 4000 降到 1500。同步墙不再被单个热点卡住。</div></div>

<table class="t"><tr><th>EPLBManager 钩子</th><th>它做什么</th></tr><tr><td><span class="mono">on_forward_pass_end()</span></td><td>每次前向后调用：把这一步各专家命中的 token 数累加进负载统计（经由专家分布记录器）。高频、轻量。</td></tr><tr><td><span class="mono">rebalance()</span></td><td>周期性调用：解一个新的专家→GPU 放置方案把负载摊平，热点专家可能被复制。低频、较重。</td></tr><tr><td><span class="mono">expert_location</span></td><td>专家→GPU 映射表：被 rebalance 更新，后续步骤按这张新地图路由。</td></tr></table>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>采集统计</h4><p><span class="mono">on_forward_pass_end()</span> 每步把各专家命中的 token 数累加进负载统计（经由专家分布记录器）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>求解放置</h4><p><span class="mono">rebalance()</span> 周期性解一个新的专家→GPU 方案把负载摊平，热点专家可能被复制成多份。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>更新地图</h4><p>把新放置写回 <span class="mono">expert_location</span>（专家→GPU 映射），后续步骤按这张新地图路由。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>重复</h4><p>负载随流量持续演化，统计继续累积，放置周期性刷新——回到第 1 步。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="四张 GPU 各持有专家，GPU2 上是一个热专家收到远超平均的 token（高高的红柱），其余三张近乎空转；同步墙意味着整步耗时由最忙的 GPU2 决定">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">倾斜：一个热专家拖慢整步</text>
    <line x1="30" y1="250" x2="752" y2="250" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="30" y1="80" x2="700" y2="80" style="stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="430" y="72" style="fill:var(--red);font-size:12px;font-weight:700">整步 = 最忙的 GPU</text>
    <rect x="70" y="204" width="60" height="46" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="100" y="232" text-anchor="middle" class="mono" style="font-size:11px">1000</text>
    <text x="100" y="196" text-anchor="middle" style="fill:var(--faint);font-size:11px">空转</text>
    <rect x="50" y="256" width="100" height="28" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="274" text-anchor="middle" style="font-size:12px">GPU0</text>
    <rect x="250" y="204" width="60" height="46" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="280" y="232" text-anchor="middle" class="mono" style="font-size:11px">1000</text>
    <text x="280" y="196" text-anchor="middle" style="fill:var(--faint);font-size:11px">空转</text>
    <rect x="230" y="256" width="100" height="28" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="280" y="274" text-anchor="middle" style="font-size:12px">GPU1</text>
    <rect x="430" y="80" width="60" height="170" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="460" y="104" text-anchor="middle" class="mono" style="font-size:11px">4000</text>
    <text x="460" y="124" text-anchor="middle" style="fill:var(--red);font-size:11px">热专家</text>
    <rect x="410" y="256" width="100" height="28" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="460" y="274" text-anchor="middle" style="font-size:12px">GPU2 · 热点</text>
    <rect x="610" y="204" width="60" height="46" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="232" text-anchor="middle" class="mono" style="font-size:11px">1000</text>
    <text x="640" y="196" text-anchor="middle" style="fill:var(--faint);font-size:11px">空转</text>
    <rect x="590" y="256" width="100" height="28" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="640" y="274" text-anchor="middle" style="font-size:12px">GPU3</text>
  </svg>
  <div class="figcap"><b>图 1 · 专家负载倾斜</b> — 四张 GPU 各持专家，GPU2 上的热专家收到 4000 个 token，其余各 1000 个。同步墙让整步耗时取决于最忙的 GPU2，其余三张干等空转。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="重平衡前后对比：之前一根红色高柱（热专家）远高于其余三根矮柱；EPLB 复制并迁移后，右侧四根柱高度大致相等，新的最大值更低，墙时间下降">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">重平衡：抹平各 GPU 的 token 计数</text>
    <text x="60" y="56" style="fill:var(--muted);font-size:12px">之前 · 倾斜</text>
    <text x="560" y="56" style="fill:var(--teal);font-size:12px">之后 · 平坦</text>
    <line x1="40" y1="240" x2="330" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="40" y1="90" x2="330" y2="90" style="stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="44" y="84" style="fill:var(--red);font-size:11px">旧 max</text>
    <rect x="60" y="200" width="40" height="40" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="130" y="90" width="40" height="150" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="150" y="112" text-anchor="middle" style="fill:var(--red);font-size:10px">热</text>
    <rect x="200" y="200" width="40" height="40" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="270" y="200" width="40" height="40" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="388" y="120" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px;font-weight:700">EPLB</text>
    <text x="388" y="178" text-anchor="middle" style="fill:var(--accent);font-size:30px">→</text>
    <text x="388" y="212" text-anchor="middle" style="fill:var(--muted);font-size:11px">复制+迁移</text>
    <line x1="450" y1="240" x2="760" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="450" y1="168" x2="760" y2="168" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="470" y="160" style="fill:var(--teal);font-size:11px;font-weight:700">新 max ↓</text>
    <rect x="480" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="550" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="620" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="690" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
  </svg>
  <div class="figcap"><b>图 2 · 重平衡抹平负载</b> — 之前一根红色高柱（热逻辑专家）远高于其余；EPLB 复制并迁移后，右侧四柱高度大致相等，新的最大值更低，墙时间随之下降。</div>
</div>

<p>用具体数字感受这一切：若平均每个专家每步收到 128 个 token，一个爆款<strong>逻辑专家</strong>可能收到 640–1280 个，即均值的 <strong>5–10×</strong>。EPLB 把这个热逻辑专家<strong>复制到多个物理槽位</strong>上——于是 <span class="mono">num_physical_experts &gt; num_logical_experts</span>，原本指向它的洪流被几个物理副本分摊。例如 256 个逻辑专家配 288 个物理槽位，意味着有 32 个最热的逻辑专家各多出一份副本，专属于它们的那股流量被切成两半。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/eplb/expert_location.py ::ExpertLocationMetadata</span><span class="ln">放置图：每层哪些物理槽位装哪些逻辑专家</span></div><pre>@dataclass
class ExpertLocationMetadata:
    # the placement map EPLB rewrites to balance load: which PHYSICAL
    # expert slots hold which LOGICAL experts, per layer.
    physical_to_logical_map: torch.Tensor      # (layers, num_physical)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical, X)
    @property
    def num_physical_experts(self):
        return self.physical_to_logical_map.shape[1]
    @property
    def num_logical_experts(self):
        return self.logical_to_all_physical_map.shape[1]</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/eplb/eplb_manager.py ::EPLBManager</span><span class="ln">采集专家负载 + 周期性重平衡放置</span></div><pre>class EPLBManager:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        # tracks per-expert token counts via the expert-distribution recorder
        ...
    def on_forward_pass_end(self):
        # called after each forward: ticks the per-step counter that
        # periodically triggers rebalance (the recorder accumulates the counts)
        ...
    def rebalance(self):
        # periodically: solve a new expert -&gt; GPU placement that flattens load
        # (hot experts may be replicated), then update expert_location
        ...</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li>MoE 路由天然倾斜：少数热点专家被大量 token 选中，长尾分布是固有特性（接第34课 FusedMoE 路由）。</li>
<li>EP 的 all-to-all 是同步墙：整步耗时由<strong>最忙的 rank</strong>决定，热点专家所在 GPU 拖垮全场（接第46课 EP/all-to-all）。</li>
<li><strong>EPLB</strong> 周期性<strong>测量</strong>每专家负载，再<strong>重排</strong>专家→GPU 放置把负载摊平，必要时<strong>复制热门专家</strong>分流。</li>
<li>SGLang 的 <span class="mono">EPLBManager</span>：<span class="mono">on_forward_pass_end()</span> 每步累加统计，<span class="mono">rebalance()</span> 周期性解新放置并更新 <span class="mono">expert_location</span>。</li>
<li>这是 DeepSeek 级 MoE 服务跑得起来的关键；其"可插拔、为规模而设计"的思路在第61/62课会继续展开。</li>
</ul></div>
""", "en": r"""
<p class="lead">In Lesson 34 we met MoE: the router picks top-k experts for each token; in Lesson 46 we saw expert parallelism (EP) spread hundreds of experts across many GPUs. But real traffic is not uniform—some experts are slammed every day, others sit nearly idle. This lesson is about what happens when EP meets that imbalance, and how SGLang's <span class="mono">EPLBManager</span> (Expert-Parallel Load Balancer) flattens that traffic-jam bottleneck.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a huge restaurant chain with 256 counters (experts) spread across 8 kitchens (GPUs), 32 counters per kitchen. Customers (tokens) walk in and a host (the router) assigns each to the most suitable top-k counters by dish. The problem: the viral "spicy hotpot" counter has a line out the door, while the "plain congee" counter sees almost no one. And the restaurant has an iron rule—<strong>all kitchens must serve their dishes at the same time; until the slowest kitchen is done, the whole floor waits</strong>. So if one kitchen happens to hold a viral counter, the cooks in the other 7 kitchens stand idle, and overall throughput is dragged down by that single hotspot.</p>
<p>What would a clever manager do? Every so often he tallies each counter's actual foot traffic, then <strong>rearranges which counters sit in which kitchen</strong>: move idle counters aside to make room, and <strong>replicate the viral counter into several copies</strong> spread across kitchens so the line splits up. Now every kitchen serves about the same number of customers per round, and the slowest kitchen no longer holds everyone back. That "manager" is EPLB.</p>
<p>Three details in this analogy are worth remembering. First, the manager <strong>cannot rely on opening-day forecasts alone</strong>—today's hot dish may go cold tomorrow, so he must keep watching traffic and adjust dynamically. Second, <strong>merely swapping counter positions cannot save a true blockbuster</strong>: if "spicy hotpot" alone draws 40% of all traffic, whichever kitchen it sits in will overflow, and the only fix is to open branches (replicate). Third, <strong>adjusting has a cost</strong>—moving counters and setting up a new kitchen means a brief pause, so the manager doesn't rebalance every minute but waits until enough traffic data confirms the landscape truly shifted. These three points map exactly onto EPLB's "periodic measurement, replicate hotspots, low-frequency rebalancing."</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>EP solves the capacity problem—too many experts to fit on one card—but it does not by itself solve the efficiency problem of load skew. EP's per-step all-to-all is a synchronization wall: <strong>the step cannot finish until the busiest card finishes</strong>. Whenever load is skewed, the GPUs holding the few hot experts become the shortest stave of the barrel, and the other GPUs spin idle. EPLB's core idea is that load is not fixed—it can be <strong>measured</strong> and <strong>re-placed</strong>. It periodically measures how many tokens each expert received, then solves a new expert→GPU placement that flattens the load—replicating hot experts when needed. For DeepSeek-scale MoE serving with hundreds of experts, this is not a nice-to-have; it is what makes serving feasible at all.</p>
<p>Stringing the three lessons together makes it clearer: Lesson 34 is about MoE and FusedMoE routing—"which experts should a token go to"; Lesson 46 is about EP and all-to-all—"experts live on different GPUs, so how do tokens travel across cards"; this lesson adds the missing piece—"when the transported load is uneven, how do we flatten it back out." The three build on each other: MoE gives us experts, EP makes cross-card transport necessary, and skew makes balancing necessary. EPLB sits at the top, finally making full use of the stage the first two lessons set up. Grasp this line and you see why large-scale MoE inference is a systems effort where routing + parallelism + balancing are all indispensable.</p>
</div>

<h2>Why is EP load inherently skewed?</h2>
<p>The MoE router is learned, not a uniform splitter. Training makes some experts specialize in high-frequency patterns (common grammatical structures, popular topics), so at inference those experts are naturally chosen by more tokens, becoming "hot experts"; other experts handle rare patterns and stay half-idle for long stretches. This <strong>long-tail distribution</strong> is an intrinsic property of MoE, not an accident. When EP spreads experts evenly across GPUs, the hot experts may happen to land on just a few cards, so those cards must process far more tokens than average each step.</p>
<p>Some concrete numbers convey the skew. Suppose a MoE layer has 256 experts with top-k = 8, and a batch holds 4096 tokens; that layer dispatches 4096×8 = 32768 expert invocations in all. If load were perfectly uniform, each expert would receive 128; but in reality the top few experts often receive thousands each, while dozens of tail experts together add up to barely a few hundred. Spread those 256 experts across 8 cards by EP, 32 per card—if the top hotspots happen to land on the same card, that card must shoulder several times the average compute this step. Skew is not "slightly uneven"; it is often an order-of-magnitude gap.</p>
<p>The key is EP's execution model: every forward step contains an all-to-all—sending each token to the GPU holding its expert, then collecting results back. This is a <strong>collective synchronization</strong>: all ranks must meet at the same barrier. So the step's wall-clock time is set by the <strong>busiest rank</strong>. If 7 of 8 cards each handle 1000 tokens while 1 handles 4000, the step costs as much as 4000, and the first 7 cards spend 75% of the time waiting. Raw EP thus wastes a great deal of compute.</p>
<p>One often-overlooked detail: load skew is not static—it <strong>drifts over time and with the input distribution</strong>. The same model, facing code generation, chit-chat, and long-document summarization, may activate completely different hot experts; even within one conversation, when the topic shifts, the hotspots migrate with it. This means any "compute once, fix forever" expert placement quickly goes stale. Skew is both intrinsic and fluid—which is precisely why the solution must be <strong>periodic measurement and dynamic adjustment</strong>, not a static config decided once at deploy time.</p>

<h2>What exactly does EPLB measure and rebalance?</h2>
<p>EPLB (Expert-Parallel Load Balancer) does two things: <strong>measure</strong> and <strong>rebalance</strong>. Measuring means counting how many tokens each expert actually received over a recent window—this per-expert load statistic is accumulated by an expert-distribution recorder after every forward. Rebalancing means solving, from that statistic, a new <strong>expert→GPU placement</strong> whose goal is to make each GPU process roughly the same number of tokens per step.</p>
<p>Rebalancing has two weapons. First is <strong>migration</strong>: moving experts between GPUs so hot and cold experts are paired up and hotspots stop clustering. Second is <strong>replicating hot experts</strong>: when an expert is so hot that one card cannot carry its traffic, EPLB <strong>replicates it onto several GPUs</strong>, so tokens pointing to it split across those copies. Replication is what sets EPLB apart from a simple "swap positions"—it cuts an indivisible hotspot into parallelizable copies. Once rebalancing is done, the new placement is written back to <span class="mono">expert_location</span> (the expert→GPU map), and subsequent steps route against this new map.</p>
<p>Why isn't "swapping positions" enough—why must we replicate? Picture the extreme case: a single expert alone eats 40% of the whole system's tokens. No matter which card you place it on, that card overflows, and idle cards cannot help—because one expert is physically indivisible. The only way out is to make multiple copies of it, spread across several cards, so the flood that pointed to it is split into several channels. Think of migration as "reshuffling the seating" and replication as "opening branch stores for the popular counter": the former fixes uneven distribution, the latter fixes single-point overheating. Only together can they truly push each card's per-step token count close to the average.</p>
<p>There are trade-offs here. Replicating a hot expert costs extra VRAM (storing another copy of its weights), and rebalancing itself moves parameters between GPUs at non-trivial cost. So EPLB does not rebalance every step; instead it <strong>accumulates statistics over a window and acts only once it confirms the load landscape has truly shifted</strong>. It balances "keeping up with load drift" against "not letting rebalancing overhead outweigh the gain"—which is exactly why collection is high-frequency while rebalancing is low-frequency. In other words, more replicas is not always better: too few cannot tame the hotspot, while too many needlessly eat VRAM that could have gone to KV cache or more concurrency. When EPLB solves a placement, it is precisely seeking a compromise between "balance each card's load as much as possible" and "don't pay too much VRAM and movement cost for it."</p>

<h2>The EPLBManager loop</h2>
<p>SGLang lands the ideas above as a clean loop in <span class="mono">EPLBManager</span>. It hangs off the <span class="mono">ModelRunner</span> and samples continuously via the expert-distribution recorder. Two hooks are central: <span class="mono">on_forward_pass_end()</span> is called <strong>after every forward</strong> to accumulate "how many tokens hit each expert this step" into the statistic; <span class="mono">rebalance()</span> is triggered <strong>periodically</strong> to solve a new expert placement (hot experts may be replicated) and then update <span class="mono">expert_location</span>. Note the division of labor: collection is high-frequency and lightweight (every step), rebalancing is low-frequency and heavier (every so often); decoupling them lets the system track load changes without letting rebalancing overhead drown out normal inference.</p>
<p>Why make collection "once per step, ultra-light"? Because only by pushing the statistic's granularity down to every step can EPLB see the true shape of the load instead of being misled by a single instant's jitter. <span class="mono">on_forward_pass_end()</span> is deliberately restrained: it does no heavy work on the forward critical path, just adds this step's per-expert counts into the recorder—a simple addition. Precisely because it is cheap enough, it can afford to run every step. This continuously accumulated statistic is the sole basis on which <span class="mono">rebalance()</span> solves a new placement.</p>
<p>This loop is stateful and adaptive: load shifts, statistics accumulate, placement refreshes periodically. It turns a "static expert layout" into a "dynamic layout that evolves with traffic." For large-scale MoE serving, this is exactly the piece that converts raw EP's theoretical throughput into real throughput.</p>
<p>To compress the whole mechanism into one sentence: EPLB lets experts' physical locations "follow the traffic." It changes not a single model parameter and not the router's selection logic; it merely keeps answering a plain question—"given that these tokens are bound for these experts, on which GPUs and in how many copies should we place the experts so that every card does equal work each step?" The answer changes as traffic changes, so the placement refreshes accordingly. This "measure–solve–place–measure-again" closed loop is essentially a lightweight online optimizer, working quietly in the gaps of the inference main loop.</p>
<p>From an engineering standpoint, the elegance of this design is its <strong>separation of concerns</strong>. The model's forward logic need not know EPLB exists at all—it merely routes tokens according to the current <span class="mono">expert_location</span>; meanwhile <span class="mono">EPLBManager</span> acts as a side-attached observer, quietly jotting a note at the end of each step and, every so often, swapping in a fresh map. Decoupling the forward path from balancing lets the balancing strategy evolve independently—even be replaced with a different solver—without touching model code. This "pluggable, designed-for-scale" mindset is the backbone of SGLang's overall architecture, which we expand on in Lessons 61 and 62. In short, the model only computes and the balancer only places—each minds its own job without disturbing the other.</p>
<p>Finally, it is worth stressing the scale effect: the more experts and the more GPUs, the more severe the waste from load skew, and the greater EPLB's value. In a DeepSeek-scale deployment with dozens of cards and hundreds of experts, raw EP without load balancing can barely keep the hardware fed; only with periodic rebalancing can the whole cluster's compute utilization truly approach its theoretical ceiling. EPLB is not an optional optimization but a precondition for running large-scale MoE inference economically.</p>

<div class="flow"><div class="node">route tokens to top-k experts</div><div class="arrow">→</div><div class="node">measure per-expert load</div><div class="arrow">→</div><div class="node">rebalance placement (replicate hot)</div><div class="arrow">→</div><div class="node">flatter GPU load</div></div>

<div class="cols"><div class="col"><strong>Skewed (raw EP)</strong><br>8 cards, 1 hot expert crammed onto GPU0. This step must wait for GPU0 to process 4000 tokens; the other 7 cards handle 1000 each and then just wait. <span class="mono">The busiest sets the step time</span>; 75% of compute spins idle.</div><div class="col"><strong>Balanced (after EPLB)</strong><br>The hot expert is replicated onto 3 cards to split traffic, and hot/cold experts are re-paired. Each GPU now handles ~1500 tokens per step; wall time drops from 4000 to 1500. The sync wall is no longer stalled by a single hot card.</div></div>

<table class="t"><tr><th>EPLBManager hook</th><th>What it does</th></tr><tr><td><span class="mono">on_forward_pass_end()</span></td><td>Called after each forward: accumulate the token count each expert hit this step into the load statistic (via the expert-distribution recorder). High-frequency, lightweight.</td></tr><tr><td><span class="mono">rebalance()</span></td><td>Called periodically: solve a new expert→GPU placement that flattens load; hot experts may be replicated. Low-frequency, heavier.</td></tr><tr><td><span class="mono">expert_location</span></td><td>The expert→GPU map: updated by rebalance, and subsequent steps route against this new map.</td></tr></table>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Collect stats</h4><p><span class="mono">on_forward_pass_end()</span> accumulates each expert's token count into the load statistic every step (via the expert-distribution recorder).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Solve placement</h4><p><span class="mono">rebalance()</span> periodically solves a new expert→GPU plan that flattens load; hot experts may be replicated into several copies.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Update the map</h4><p>Write the new placement back to <span class="mono">expert_location</span> (the expert→GPU map); subsequent steps route against this new map.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Repeat</h4><p>Load keeps evolving with traffic, stats keep accumulating, placement refreshes periodically — back to step 1.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Four GPUs each hold experts; GPU2 holds a hot expert receiving far more tokens (tall red bar) while the others sit near-idle; the sync wall means the step time is set by the busiest GPU2">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">skew: one hot expert stalls step</text>
    <line x1="30" y1="250" x2="752" y2="250" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="30" y1="80" x2="700" y2="80" style="stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="430" y="72" style="fill:var(--red);font-size:12px;font-weight:700">step = busiest GPU</text>
    <rect x="70" y="204" width="60" height="46" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="100" y="232" text-anchor="middle" class="mono" style="font-size:11px">1000</text>
    <text x="100" y="196" text-anchor="middle" style="fill:var(--faint);font-size:11px">idle</text>
    <rect x="50" y="256" width="100" height="28" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="274" text-anchor="middle" style="font-size:12px">GPU0</text>
    <rect x="250" y="204" width="60" height="46" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="280" y="232" text-anchor="middle" class="mono" style="font-size:11px">1000</text>
    <text x="280" y="196" text-anchor="middle" style="fill:var(--faint);font-size:11px">idle</text>
    <rect x="230" y="256" width="100" height="28" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="280" y="274" text-anchor="middle" style="font-size:12px">GPU1</text>
    <rect x="430" y="80" width="60" height="170" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="460" y="104" text-anchor="middle" class="mono" style="font-size:11px">4000</text>
    <text x="460" y="124" text-anchor="middle" style="fill:var(--red);font-size:11px">hot exp</text>
    <rect x="410" y="256" width="100" height="28" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="460" y="274" text-anchor="middle" style="font-size:12px">GPU2 · hot</text>
    <rect x="610" y="204" width="60" height="46" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="640" y="232" text-anchor="middle" class="mono" style="font-size:11px">1000</text>
    <text x="640" y="196" text-anchor="middle" style="fill:var(--faint);font-size:11px">idle</text>
    <rect x="590" y="256" width="100" height="28" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="640" y="274" text-anchor="middle" style="font-size:12px">GPU3</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Skewed expert load</b> — four GPUs each hold experts; the hot expert on GPU2 gets 4000 tokens while the others get 1000 each. The sync wall ties step time to the busiest GPU2, leaving the other three stalling idle.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Before/after rebalance: before, one tall red bar (hot expert) towers over three short bars; after EPLB replicates and migrates, the four right-hand bars are roughly equal height and the new max is lower, so wall time drops">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">rebalance: flatten per-GPU token counts</text>
    <text x="60" y="56" style="fill:var(--muted);font-size:12px">before · skew</text>
    <text x="560" y="56" style="fill:var(--teal);font-size:12px">after · flat</text>
    <line x1="40" y1="240" x2="330" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="40" y1="90" x2="330" y2="90" style="stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="44" y="84" style="fill:var(--red);font-size:11px">old max</text>
    <rect x="60" y="200" width="40" height="40" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="130" y="90" width="40" height="150" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="150" y="112" text-anchor="middle" style="fill:var(--red);font-size:10px">hot</text>
    <rect x="200" y="200" width="40" height="40" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="270" y="200" width="40" height="40" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="388" y="120" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px;font-weight:700">EPLB</text>
    <text x="388" y="178" text-anchor="middle" style="fill:var(--accent);font-size:30px">→</text>
    <text x="388" y="212" text-anchor="middle" style="fill:var(--muted);font-size:11px">copy+move</text>
    <line x1="450" y1="240" x2="760" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="450" y1="168" x2="760" y2="168" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="470" y="160" style="fill:var(--teal);font-size:11px;font-weight:700">new max ↓</text>
    <rect x="480" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="550" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="620" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="690" y="168" width="40" height="72" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
  </svg>
  <div class="figcap"><b>Fig 2 · Rebalance flattens load</b> — before, one tall red bar (hot logical expert) towers over the rest; after EPLB replicates and migrates, the four right-hand bars are roughly equal and the new max is lower, so wall time drops.</div>
</div>

<p>Put concrete numbers on it: if each expert averages 128 tokens per step, a blockbuster <strong>logical expert</strong> might get 640–1280, i.e. <strong>5–10×</strong> the average. EPLB <strong>replicates that hot logical expert onto multiple physical slots</strong>—so <span class="mono">num_physical_experts &gt; num_logical_experts</span>, and the flood that pointed at it is split across the physical copies. For example, 256 logical experts mapped onto 288 physical slots means the 32 hottest logical experts each get an extra copy, halving the traffic any single copy must carry.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/eplb/expert_location.py ::ExpertLocationMetadata</span><span class="ln">placement map: which physical slots hold which logical experts, per layer</span></div><pre>@dataclass
class ExpertLocationMetadata:
    # the placement map EPLB rewrites to balance load: which PHYSICAL
    # expert slots hold which LOGICAL experts, per layer.
    physical_to_logical_map: torch.Tensor      # (layers, num_physical)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical, X)
    @property
    def num_physical_experts(self):
        return self.physical_to_logical_map.shape[1]
    @property
    def num_logical_experts(self):
        return self.logical_to_all_physical_map.shape[1]</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/eplb/eplb_manager.py ::EPLBManager</span><span class="ln">gather expert load + periodically rebalance placement</span></div><pre>class EPLBManager:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        # tracks per-expert token counts via the expert-distribution recorder
        ...
    def on_forward_pass_end(self):
        # called after each forward: ticks the per-step counter that
        # periodically triggers rebalance (the recorder accumulates the counts)
        ...
    def rebalance(self):
        # periodically: solve a new expert -&gt; GPU placement that flattens load
        # (hot experts may be replicated), then update expert_location
        ...</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li>MoE routing is inherently skewed: a few hot experts are chosen by many tokens; the long-tail distribution is intrinsic (ties to Lesson 34 FusedMoE routing).</li>
<li>EP's all-to-all is a sync wall: the step time is set by the <strong>busiest rank</strong>, and the GPU holding hot experts drags down the whole floor (ties to Lesson 46 EP/all-to-all).</li>
<li><strong>EPLB</strong> periodically <strong>measures</strong> per-expert load, then <strong>rebalances</strong> the expert→GPU placement to flatten load, <strong>replicating hot experts</strong> to split traffic when needed.</li>
<li>SGLang's <span class="mono">EPLBManager</span>: <span class="mono">on_forward_pass_end()</span> accumulates stats each step; <span class="mono">rebalance()</span> periodically solves a new placement and updates <span class="mono">expert_location</span>.</li>
<li>This is what makes DeepSeek-scale MoE serving feasible; its "pluggable, designed-for-scale" themes continue in Lessons 61/62.</li>
</ul></div>
"""}
LESSON_48 = {"zh": r"""
<p class="lead">很多真实应用并不满足于模型"大致说对"，而是要求输出<strong>严格合法</strong>：必须是能被解析的 JSON、必须匹配某个正则、必须符合一套语法（grammar）。SGLang 用<strong>受限解码（constrained decoding）</strong>从根上保证这件事——它把约束编译成一台有限状态机（FSM），在每一步采样<strong>之前</strong>就把不合法的 token 从 logits（<span class="inline">第37课</span>）上抹成 <span class="mono">−∞</span>，于是不管采样器（<span class="inline">第28课</span>）怎么掷骰子，输出永远是良构的。本课把这条链路讲透，并介绍它最漂亮的优化：<strong>跳跃前进（jump-forward）</strong>。这也是第十部分的收官一课。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div><p>想象你在机场填一张报关单。表格上每个空格都印好了边框与提示：姓名栏只能写字母，日期栏只能写数字，国籍栏只能从下拉列表里挑一项。哪怕你手再抖、笔再随意，也不可能把出生日期写进姓名栏——因为表格的"模具"在你落笔<strong>之前</strong>就限定了此处合法的选项。受限解码就是给模型套上这样一张"会发光的模具"：每生成一个 token 之前，它先算出"此刻哪些 token 是合法的"，把其余全部划掉变暗，模型只能在亮着的格子里挑。而跳跃前进则像表格里那些"系统已自动填好"的固定字段——既然某一段只有唯一可能的答案，何必再让你一笔一画地照抄一遍？直接替你填好，光标往后一跳即可。</p></div>

<div class="card macro"><div class="tag">🌍 宏观理解</div><p>把这件事放回整条推理流水线来看：模型前向算出 logits（<span class="inline">第37课</span>），采样器据此挑 token（<span class="inline">第28课</span>）。受限解码不另起炉灶，而是<strong>恰好嵌在两者之间</strong>——在 logits 已出、采样未发生的那一瞬，往 logits 上盖一层"词表掩码（vocab mask）"，只放行能让输出继续合法的 token。正因为它改的是 logits 而不是改采样算法，所以它和贪心、温度、top-p、束搜索乃至投机解码（<span class="inline">第43课</span>）都能无缝叠加。掌握这一课，你就理解了 SGLang 如何在"自由生成"与"严格结构"之间取得既正确又高效的平衡。</p></div>

<h2>为什么需要受限解码</h2>
<p>大语言模型本质是一台概率机器：它对下一个 token 给出一整张概率分布，再由采样器抽取。这套机制擅长流畅表达，却<strong>天生不保证结构</strong>。如果你的下游是一个函数调用、一段需要 <span class="mono">json.loads</span> 的配置、或一个要喂进数据库的字段，模型偶尔漏个引号、多个逗号、把数字写成中文，整条管线就会崩。传统做法是"生成后校验、失败就重试"，但重试既慢又烧钱，而且并不能<strong>保证</strong>最终一定合法。设想一个三层嵌套的 JSON，模型在最后一个字段漏了个引号，整次生成全部作废、从头再来；若运气不好连续几次都在不同位置出错，延迟与成本会成倍膨胀，对在线服务尤其致命。</p>
<p>受限解码换了一个思路：与其事后补救，不如<strong>从生成的每一步就杜绝非法</strong>。它要求用户先给出一份"形状说明"——可以是 JSON-Schema、一段正则表达式、或一份 EBNF 文法。SGLang 通过可插拔的语法后端（xgrammar / outlines / llguidance）把这份说明<strong>编译</strong>成一台有限状态机：FSM 的每个状态都精确知道"在当前位置，接下来哪些 token 是被允许的"。于是在每个解码步，SGLang 先问 FSM 要一张"允许集合"，据此构造词表掩码，把不在允许集合里的所有 token 的 logit 直接设为 <span class="mono">−∞</span>。被设成负无穷的 token 经过 softmax 后概率为零，采样器无论用什么策略都不可能选中它们。换言之，<strong>合法性不再是采样器的运气，而是掩码的铁律</strong>。</p>
<p>这里有必要把"职责边界"讲清楚。语法后端并不决定<strong>该选哪一个</strong>合法 token——那仍然是采样器的活，由温度、top-p 等参数掌控；语法后端只决定<strong>哪些 token 有资格被纳入候选</strong>。正是这种关注点分离，让受限解码能够如此干净地与其它机制组合：它本质上是对候选集合的一道纯粹的"前置过滤"，丝毫不触碰挑选这一创造性动作。于是模型在文法允许自由的地方依旧自由表达，只在文法要求严格的地方被圈起来。也正因如此，同一个采样器、同一套温度与 top-p 配置，套上不同 schema 就能产出不同形状却同样合法的结果，无需为每种格式各写一套生成逻辑。</p>

<div class="flow"><div class="node">logits（第37课）</div><div class="arrow">→</div><div class="node">词表掩码：FSM 只放行合法 token，其余设 −∞</div><div class="arrow">→</div><div class="node">采样器（第28课）只能采到合法 token</div><div class="arrow">→</div><div class="node">accept_token：FSM 前进一步</div></div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="受限解码的一步：FSM 当前状态算出词表掩码，把非法 token 的 logits 设为负无穷，采样只会落在合法 token 上">
    <rect x="20" y="18" width="340" height="32" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="190" y="39" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">FSM S0：刚写下 { → 只允许 "</text>

    <text x="178" y="72" text-anchor="middle" class="mono" style="font-size:12px">"</text>
    <text x="242" y="72" text-anchor="middle" class="mono" style="font-size:12px">}</text>
    <text x="306" y="72" text-anchor="middle" class="mono" style="font-size:12px">n</text>
    <text x="370" y="72" text-anchor="middle" class="mono" style="font-size:12px">5</text>
    <text x="434" y="72" text-anchor="middle" class="mono" style="font-size:12px">,</text>

    <text x="20" y="100" style="fill:var(--muted);font-size:12px">logits</text>
    <rect x="150" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="178" y="101" text-anchor="middle" class="mono" style="font-size:11px">2.1</text>
    <rect x="214" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="242" y="101" text-anchor="middle" class="mono" style="font-size:11px">1.8</text>
    <rect x="278" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="101" text-anchor="middle" class="mono" style="font-size:11px">3.0</text>
    <rect x="342" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="370" y="101" text-anchor="middle" class="mono" style="font-size:11px">0.4</text>
    <rect x="406" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="434" y="101" text-anchor="middle" class="mono" style="font-size:11px">1.2</text>

    <text x="20" y="150" style="fill:var(--muted);font-size:12px">FSM 掩码</text>
    <rect x="150" y="132" width="56" height="30" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="178" y="151" text-anchor="middle" style="fill:var(--teal);font-size:12px">✓</text>
    <rect x="214" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="242" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>
    <rect x="278" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>
    <rect x="342" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="370" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>
    <rect x="406" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="434" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>

    <text x="20" y="200" style="fill:var(--muted);font-size:12px">−∞ 后</text>
    <rect x="150" y="182" width="56" height="30" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="178" y="201" text-anchor="middle" class="mono" style="font-size:11px">2.1</text>
    <rect x="214" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="242" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>
    <rect x="278" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>
    <rect x="342" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="370" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>
    <rect x="406" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="434" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>

    <rect x="150" y="232" width="312" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="306" y="254" text-anchor="middle" style="fill:var(--blue);font-size:12px">采样只会选到 " → FSM 前进到 S1</text>
  </svg>
  <div class="figcap"><b>图 1 · 词表掩码挡在采样前</b> — FSM 当前状态算出哪些 token 合法，把非法 token 的 logits 设为 <span class="mono">−∞</span>；采样器只能落在亮着的合法 token 上，随后 FSM 前进一步。</div>
</div>

<h2>每请求一台 FSM：BaseGrammarObject</h2>
<p>结构约束是<strong>逐请求</strong>的：不同请求可能带着不同的 schema，而且每个请求的解码进度各不相同。因此 SGLang 为每个请求维护一个属于它自己的语法对象，抽象基类就是 <span class="mono">BaseGrammarObject</span>。可以把它想成"这一条请求当前走到了 FSM 的哪个状态"的随身记录本，每生成一个 token 就翻一页。它对外暴露的接口非常精简，却覆盖了受限解码的全部需求。之所以采用"每请求一个对象"的设计，是因为受限解码的状态<strong>无法在请求之间共享</strong>：哪怕两条请求用的是同一份 schema，它们也可能停在 FSM 的不同状态上，需要各自独立的进度、各自独立的回退历史。把状态封装进对象，调度器就能在一个批次里同时驱动几十上百条处于不同进度的请求，互不干扰。</p>
<p>第一组是<strong>推进</strong>：当采样器最终选定一个 token，调度器调用 <span class="mono">accept_token(token)</span>，让 FSM 根据这个 token 转移到下一个状态——下一步的"允许集合"也随之更新。第二组是<strong>掩码构造</strong>：<span class="mono">allocate_vocab_mask(...)</span> 先按词表大小、批大小、设备分配一块掩码缓冲区，<span class="mono">fill_vocab_mask(vocab_mask, idx)</span> 再把"此刻哪些 token 合法"标记进去（idx 指明这是批里的第几条请求）。这块掩码随后被叠加到 logits 上。第三组是<strong>回退</strong>：<span class="mono">rollback(k)</span> 让 FSM 倒退 k 步，这在两种场景下至关重要——一是需要回溯（backtracking）的语法，二是投机解码（<span class="inline">第43课</span>）里草稿模型一次提出多个候选 token，但其中部分被验证拒绝，那么 FSM 必须把"已经按这些被拒 token 前进过"的状态精确退回去。最后是<strong>终止判断</strong>：<span class="mono">is_terminated()</span> 检查 FSM 是否已抵达接受状态（accept state），即结构是否已经完整闭合，可以收尾。这四组接口看似简单，却恰好对应了一台 FSM 在解码过程中会经历的全部生命周期事件：前进、查询当前可行集、必要时倒带、以及判断是否抵达终点。掌握了它们，你就掌握了 SGLang 受限解码对外的全部契约。</p>

<div class="cellgroup"><div class="cell">S0 <span class="mono">{</span></div><div class="cell sc">S1 <span class="mono">"name"</span> 强制</div><div class="cell sc">S2 <span class="mono">:</span> 强制</div><div class="cell">S3 值：自由采样</div><div class="cell"><span class="mono">}</span> 接受</div></div>

<table class="t"><tr><th>BaseGrammarObject 方法</th><th>角色</th></tr><tr><td><span class="mono">accept_token(token)</span></td><td>用采样到的 token 推进 FSM 到下一状态</td></tr><tr><td><span class="mono">allocate_vocab_mask(...)</span></td><td>按词表/批/设备分配一块掩码缓冲区</td></tr><tr><td><span class="mono">fill_vocab_mask(mask, idx)</span></td><td>标记当前哪些 token 能让输出保持合法</td></tr><tr><td><span class="mono">rollback(k)</span></td><td>回退 k 步（回溯 / 投机解码被拒）</td></tr><tr><td><span class="mono">is_terminated()</span></td><td>FSM 是否已到达接受状态</td></tr></table>

<h2>跳跃前进：跳过被语法"钉死"的 token</h2>
<p>到这里，受限解码已经能保证<strong>正确</strong>，但还不够<strong>快</strong>。观察一个常见现象：很多时候 FSM 在某个状态下的"允许集合"里<strong>只有一个</strong> token，而且下一个状态又只有一个，再下一个还是只有一个……换句话说，文法在这一段是<strong>完全确定</strong>的，没有任何自由度。最典型的就是 JSON-Schema 里的固定键名：一旦写下 <span class="mono">{</span>，schema 强制接下来必须是 <span class="mono">"name":</span> 这串字符，模型在这里根本没有选择权。可是按部就班的做法，仍然要为这串被"钉死"的 token <strong>逐个调用模型前向</strong>去"预测"它们——这纯属浪费算力，因为答案早已注定。这种被钉死的片段在结构化输出里其实占了相当大的比重：花括号、引号、冒号、逗号、固定的字段名、枚举值的前缀，全都是文法强加的样板，它们的数量往往比真正自由的内容字段还要多。</p>
<p>SGLang 的优化叫<strong>跳跃前进（jump-forward），底层是压缩 FSM（compressed FSM）</strong>。它在编译阶段就识别出这些"确定性跨度"：凡是只有唯一续接的连续 token 段，都被预先压缩、记录下来。运行时一旦 FSM 走到这样的状态，SGLang 不再请模型前向，而是<strong>直接把这整段强制字符串拼接进输出</strong>，并让 FSM 一次性向前跳过对应的若干步。具体落地在 <span class="mono">OutlinesJumpForwardMap.jump_forward_symbol(state)</span>：传入当前状态，它返回该状态起被文法钉死的那段字符串，供调度器直接 splice 进去。这一跳带来双重收益：其一是<strong>更快</strong>——省掉了一连串无意义的前向，结构化生成显著提速；其二是<strong>质量更好</strong>——模型不必再分心去"预测"那些样板字符（boilerplate），可以把注意力集中在真正需要它判断的自由字段上。这一优化为第十部分画上句号。值得一提的是，确定性跨度越长、越频繁，跳跃前进的收益就越大——对那些字段众多、键名固定的复杂 schema，省下的前向次数相当可观，端到端时延可观地下降。</p>

<div class="cols"><div class="col"><strong>逐 token 解码</strong><br>对被钉死的 <span class="mono">" n a m e " :</span> 每个字符都老老实实跑一次模型前向去"预测"，明明只有唯一答案，却步步耗算力。</div><div class="col"><strong>跳跃前进</strong><br>检测到唯一续接，<span class="mono">jump_forward_symbol</span> 直接吐出整段 <span class="mono">"name":</span>，FSM 一次跳过这些步，<strong>不调用模型</strong>，更快也更稳。</div></div>

<div class="fig">
  <svg viewBox="0 0 800 230" role="img" aria-label="jump-forward：被文法钉死的强制串一次性吐出，FSM 跳过对应步数，省下逐字符的模型前向">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">jump-forward：强制串一次吐出</text>

    <text x="20" y="78" style="fill:var(--amber);font-size:12px">逐 token</text>
    <rect x="120" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="146" y="77" text-anchor="middle" class="mono" style="font-size:12px">"</text>
    <rect x="180" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="206" y="77" text-anchor="middle" class="mono" style="font-size:12px">n</text>
    <rect x="240" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="266" y="77" text-anchor="middle" class="mono" style="font-size:12px">a</text>
    <rect x="300" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="326" y="77" text-anchor="middle" class="mono" style="font-size:12px">m</text>
    <rect x="360" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="386" y="77" text-anchor="middle" class="mono" style="font-size:12px">e</text>
    <rect x="420" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="446" y="77" text-anchor="middle" class="mono" style="font-size:12px">"</text>
    <rect x="480" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="506" y="77" text-anchor="middle" class="mono" style="font-size:12px">:</text>
    <text x="120" y="110" style="fill:var(--amber);font-size:12px">7 次模型前向 · 浪费</text>

    <text x="20" y="170" style="fill:var(--teal);font-size:12px">jump-forward</text>
    <rect x="120" y="148" width="412" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="326" y="169" text-anchor="middle" class="mono" style="font-size:12px">"name": 一次跳过</text>
    <rect x="548" y="148" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="608" y="169" text-anchor="middle" style="fill:var(--blue);font-size:12px">值：采样</text>
    <text x="120" y="202" style="fill:var(--teal);font-size:12px">1 步跳过 → 省下 7 次前向</text>
  </svg>
  <div class="figcap"><b>图 2 · 跳过被钉死的强制串</b> — 当 FSM 在一段里只有唯一续接，<span class="mono">jump_forward_symbol</span> 把整段 <span class="mono">"name":</span> 一次拼进输出、跳过对应步数，省掉逐字符的模型前向；之后才回到正常采样去填那个自由的值。</div>
</div>

<div class="card"><div class="tag">🧪 具体例子</div><p>给定 schema <span class="mono">{"name": string}</span>：一旦 FSM 走过 <span class="mono">{</span>，接下来 <span class="mono">"name":</span> 这约 8 个字符<strong>完全确定</strong>、没有任何分支。逐 token 解码要为这 8 个字符各跑一次模型前向；而 <span class="mono">jump_forward_symbol</span> 一次性把 <span class="mono">"name":</span> 整段拼进输出、FSM 跳过这 8 步，<strong>0 次前向</strong>。真正需要采样的只有后面那个自由的<strong>值</strong>（比如 <span class="mono">"Ada"</span>）。</p></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/constrained/base_grammar_backend.py ::BaseGrammarObject</span><span class="ln">每请求一个语法 FSM：屏蔽 logits 保证合法 + 可回滚</span></div><pre>class BaseGrammarObject:                 # one per request: the compiled grammar FSM
    def accept_token(self, token):
        ...                              # advance the FSM by the sampled token
    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        ...                              # make a [vocab] mask buffer
    def fill_vocab_mask(self, vocab_mask, idx):
        ...                              # mark which tokens keep the output grammar-valid
    def rollback(self, k):
        ...                              # back up k steps (backtracking / spec-decode reject)
    def is_terminated(self):
        ...                              # has the grammar reached an accept state?
# jump-forward lives in constrained/outlines_jump_forward.py ::OutlinesJumpForwardMap</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/constrained/outlines_jump_forward.py ::OutlinesJumpForwardMap</span><span class="ln">按 FSM 状态预存"只有一条路"时的强制串</span></div><pre>class OutlinesJumpForwardMap:
    def __init__(self, regex_string):
        # per FSM state, precompute the forced string for any stretch
        # where the path is deterministic (a single way forward).
        self.state_to_jump_forward = init_state_to_jump_forward(regex_string)
    def jump_forward_symbol(self, state):
        # walk while each state has ONE outgoing edge, concatenating
        # the forced characters.
        jump_forward_str = ""
        while state in self.state_to_jump_forward:
            e = self.state_to_jump_forward[state]
            jump_forward_str += e.symbol
            ...
        return jump_forward_str, next_state</pre></div>

<h2>回滚、终止与投机解码的配合</h2>
<p>最后看三个易被忽略却很关键的细节。其一，<strong>回滚不是可有可无的</strong>。在投机解码里，草稿模型一口气提出若干 token，目标模型并行验证，常常只接受前几个、拒绝后几个。语法 FSM 在验证前必须假定这些 token 都被接受、据此前进；一旦验证拒绝，就得用 <span class="mono">rollback(k)</span> 把多走的 k 步精确退回，保证 FSM 状态与真正被接受的序列严格一致——否则后续掩码就会基于错误状态构造，约束随之失效。其二，<strong>终止判断决定何时收尾</strong>：<span class="mono">is_terminated()</span> 返回真，意味着结构已经完整闭合（例如 JSON 的最外层大括号已配平），可以安全停止生成，避免画蛇添足。其三，<strong>掩码构造要高效</strong>：<span class="mono">allocate_vocab_mask</span> 与 <span class="mono">fill_vocab_mask</span> 被设计成可在 GPU 上按批处理的形态，让受限解码的额外开销尽量被前向计算掩盖，从而几乎不拖慢吞吐。这一点尤其重要：受限解码若实现得笨拙，每步都在 CPU 上逐个 token 判断合法性、再同步回 GPU，开销会轻易吃掉它省下的算力。SGLang 的做法是把"判断合法"尽量提前到编译期完成，运行期只剩一次轻量的掩码填充与一次张量相加，于是它能在保证严格结构的同时几乎不付出吞吐代价。把推进、掩码、回滚、终止这四件事拼在一起，就是 SGLang 受限解码完整而稳固的骨架。</p>
<p>还有一条设计层面的经验值得点出：SGLang 刻意把上述一切都收敛到同一个抽象基类之后，让具体的语法引擎成为<strong>可插拔</strong>的部件。无论底层编译器是 xgrammar、outlines 还是 llguidance，调度器只与 <span class="mono">BaseGrammarObject</span> 这套接口打交道——推进、掩码、回滚、终止。这意味着将来出现更快、更具表达力的语法引擎，可以直接替换进来而不必改动解码主循环，甚至不同请求类型可以并存使用不同引擎。让掩码高效的那套工程纪律，同样让整个子系统具备面向未来的弹性：这正是 SGLang 架构里反复出现的"为规模而设计、内部可替换"的理念。把它和前面几课的投机解码、PD 分离、并行与 EPLB 放在一起看，你会发现它们共享同一种工程审美——用清晰的接口隔离复杂度，再在接口背后追求极致性能。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>受限解码保证结构合法</strong>：把 JSON-Schema / 正则 / EBNF 编译成 FSM，在采样前用词表掩码把非法 token 的 logits 设为 −∞，从源头杜绝非法。</li>
<li><strong>恰嵌在 logits 与采样器之间</strong>：改的是 logits（<span class="inline">第37课</span>）而非采样算法（<span class="inline">第28课</span>），因此与各种采样策略可叠加。</li>
<li><strong>BaseGrammarObject 是每请求的 FSM</strong>：<span class="mono">accept_token</span> 推进、<span class="mono">fill_vocab_mask</span> 建掩码、<span class="mono">rollback</span> 回退、<span class="mono">is_terminated</span> 判终止。</li>
<li><strong>跳跃前进省掉被钉死 token 的前向</strong>：压缩 FSM 用 <span class="mono">jump_forward_symbol</span> 直接吐出强制字符串，既更快又让模型更专注，质量更好，跨度越长收益越大。</li>
<li><strong>回滚支撑投机解码</strong>：被拒草稿 token 需要 <span class="mono">rollback(k)</span> 精确还原 FSM 状态，保证后续掩码不被污染（<span class="inline">第43课</span>）。</li>
</ul></div>

<div class="card"><div class="tag">🏁 第十部分小结</div><p>第十部分围绕"如何把 SGLang 推向大规模、高性能"展开：投机解码用草稿+验证把多 token 一次落地，EAGLE 进一步用特征级草稿提升接受率；PD 分离把预填充与解码拆到不同实例，各自吃满算力与显存带宽；张量/流水线/数据/专家四种并行让超大模型横跨众多 GPU；EPLB 在 MoE 场景里动态均衡专家负载、消除热点；而本课的结构化输出与跳跃前进，则在保证输出严格合法的同时省去无谓前向。这些技术看似各管一摊，实则共同回答同一个问题——<strong>在不牺牲正确性的前提下，把每一分算力都用在刀刃上</strong>，从而把 SGLang 稳稳推到大规模、高性能的生产水位。至此，第十部分的拼图全部归位：从"算得更快"到"算得更省"，再到"算得既对又稳"，SGLang 用一组彼此咬合的机制，回应了把大模型搬上真实生产的全部核心挑战。</p></div>
""", "en": r"""
<p class="lead">Many real applications are not satisfied with a model being "roughly right"—they demand output that is <strong>strictly valid</strong>: it must be parseable JSON, must match a regex, or must follow a grammar. SGLang guarantees this at the root with <strong>constrained decoding</strong>: it compiles the constraint into a finite-state machine (FSM) and, <strong>before</strong> sampling at every step, sets the logits (<span class="inline">Lesson 37</span>) of every illegal token to <span class="mono">−∞</span>. So no matter how the sampler (<span class="inline">Lesson 28</span>) rolls the dice, the output is always well-formed. This lesson walks the whole pipeline and presents its most elegant optimization: <strong>jump-forward</strong>. It also closes Part 10.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div><p>Picture filling out a customs form at the airport. Every box is pre-printed with borders and hints: the name box accepts only letters, the date box only digits, the nationality box only an option from a dropdown. No matter how shaky your hand, you simply cannot write a birth date into the name box—because the form's "mold" fixes the legal options <strong>before</strong> your pen touches paper. Constrained decoding fits the model with exactly such a "glowing mold": before producing each token it computes "which tokens are legal right now," dims out the rest, and the model may only pick from the lit boxes. Jump-forward is like those "auto-filled" fixed fields on the form—if a span has only one possible answer, why make you copy it character by character? Just fill it in and jump the cursor ahead.</p></div>

<div class="card macro"><div class="tag">🌍 The big picture</div><p>Place this back in the inference pipeline: the forward pass produces logits (<span class="inline">Lesson 37</span>), and the sampler picks a token from them (<span class="inline">Lesson 28</span>). Constrained decoding does not reinvent anything—it sits <strong>exactly between the two</strong>. In the instant after logits exist but before sampling happens, it overlays a "vocab mask" on the logits, allowing only tokens that keep the output legal. Because it edits logits rather than the sampling algorithm, it stacks seamlessly with greedy, temperature, top-p, beam search, and even speculative decoding (<span class="inline">Lesson 43</span>). Master this lesson and you understand how SGLang balances "free generation" against "strict structure"—correctly and efficiently.</p></div>

<h2>Why constrained decoding is needed</h2>
<p>A large language model is fundamentally a probability machine: it outputs a full distribution over the next token, and the sampler draws from it. This excels at fluent expression but <strong>guarantees nothing structurally</strong>. If your downstream is a function call, a config that must pass <span class="mono">json.loads</span>, or a field bound for a database, an occasional missing quote, an extra comma, or a number written as a word will break the entire pipeline. The traditional fix is "generate, validate, retry on failure," but retrying is slow and costly—and it still cannot <strong>guarantee</strong> the final result is valid. Imagine a three-level nested JSON where the model drops a quote in the very last field: the whole generation is voided and restarts from scratch; if you are unlucky enough to fail repeatedly at different spots, latency and cost balloon, which is especially fatal for online serving.</p>
<p>Constrained decoding takes a different stance: rather than patching up afterward, <strong>forbid illegality at every step of generation</strong>. It asks the user for a "shape spec"—a JSON-Schema, a regular expression, or an EBNF grammar. Through a pluggable grammar backend (xgrammar / outlines / llguidance), SGLang <strong>compiles</strong> that spec into a finite-state machine: each FSM state knows precisely "at this position, which tokens are allowed." At each decode step SGLang asks the FSM for the "allowed set," builds a vocab mask from it, and sets the logit of every token outside the allowed set to <span class="mono">−∞</span>. After softmax those tokens have probability zero, so the sampler—whatever strategy it uses—can never pick them. In other words, <strong>validity is no longer the sampler's luck but the mask's iron law</strong>.</p>
<p>It is worth stressing where the boundary of responsibility lies. The grammar backend does not decide <strong>which</strong> legal token to emit—that remains the sampler's job, governed by temperature, top-p, and the rest. The backend only decides <strong>which tokens are even allowed to be considered</strong>. This separation of concerns is precisely why constrained decoding composes so cleanly: it is a pure pre-filter on the candidate set, leaving the creative act of selection untouched. The model still expresses itself freely wherever the grammar permits freedom, and is merely fenced in wherever the grammar demands rigidity.</p>

<div class="flow"><div class="node">logits (Lesson 37)</div><div class="arrow">→</div><div class="node">vocab mask: FSM allows only valid tokens, rest set to −∞</div><div class="arrow">→</div><div class="node">sampler (Lesson 28) can only pick a valid token</div><div class="arrow">→</div><div class="node">accept_token: FSM advances one step</div></div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="one constrained-decode step: the FSM state computes a vocab mask, sets illegal token logits to negative infinity, so sampling only lands on a valid token">
    <rect x="20" y="18" width="340" height="32" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="190" y="39" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">FSM S0: wrote { → only " allowed</text>

    <text x="178" y="72" text-anchor="middle" class="mono" style="font-size:12px">"</text>
    <text x="242" y="72" text-anchor="middle" class="mono" style="font-size:12px">}</text>
    <text x="306" y="72" text-anchor="middle" class="mono" style="font-size:12px">n</text>
    <text x="370" y="72" text-anchor="middle" class="mono" style="font-size:12px">5</text>
    <text x="434" y="72" text-anchor="middle" class="mono" style="font-size:12px">,</text>

    <text x="20" y="100" style="fill:var(--muted);font-size:12px">logits</text>
    <rect x="150" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="178" y="101" text-anchor="middle" class="mono" style="font-size:11px">2.1</text>
    <rect x="214" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="242" y="101" text-anchor="middle" class="mono" style="font-size:11px">1.8</text>
    <rect x="278" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="101" text-anchor="middle" class="mono" style="font-size:11px">3.0</text>
    <rect x="342" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="370" y="101" text-anchor="middle" class="mono" style="font-size:11px">0.4</text>
    <rect x="406" y="82" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="434" y="101" text-anchor="middle" class="mono" style="font-size:11px">1.2</text>

    <text x="20" y="150" style="fill:var(--muted);font-size:12px">FSM mask</text>
    <rect x="150" y="132" width="56" height="30" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="178" y="151" text-anchor="middle" style="fill:var(--teal);font-size:12px">✓</text>
    <rect x="214" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="242" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>
    <rect x="278" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>
    <rect x="342" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="370" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>
    <rect x="406" y="132" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="434" y="151" text-anchor="middle" style="fill:var(--faint);font-size:12px">✗</text>

    <text x="20" y="200" style="fill:var(--muted);font-size:12px">after −∞</text>
    <rect x="150" y="182" width="56" height="30" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="178" y="201" text-anchor="middle" class="mono" style="font-size:11px">2.1</text>
    <rect x="214" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="242" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>
    <rect x="278" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>
    <rect x="342" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="370" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>
    <rect x="406" y="182" width="56" height="30" rx="5" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="434" y="201" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">−∞</text>

    <rect x="150" y="232" width="312" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="306" y="254" text-anchor="middle" style="fill:var(--blue);font-size:12px">sampler can only pick " → FSM to S1</text>
  </svg>
  <div class="figcap"><b>Fig 1 · the vocab mask gates the sampler</b> — the current FSM state computes which tokens are legal and sets illegal tokens' logits to <span class="mono">−∞</span>; the sampler can only land on a lit, valid token, after which the FSM advances one step.</div>
</div>

<h2>One FSM per request: BaseGrammarObject</h2>
<p>Structural constraints are <strong>per-request</strong>: different requests may carry different schemas, and each request's decode progress differs. SGLang therefore keeps a grammar object that belongs to each request, with the abstract base class <span class="mono">BaseGrammarObject</span>. Think of it as a personal notebook recording "where this request currently stands in the FSM." Its interface is tiny yet covers everything constrained decoding needs. The "one object per request" design exists because constrained-decoding state <strong>cannot be shared across requests</strong>: even two requests using the same schema may sit at different FSM states, each needing its own progress and its own rollback history. Encapsulating state in an object lets the scheduler drive dozens or hundreds of requests at different progress points within one batch without interference.</p>
<p>The first group is <strong>advancing</strong>: once the sampler finally chooses a token, the scheduler calls <span class="mono">accept_token(token)</span>, transitioning the FSM to the next state—and the next step's "allowed set" updates accordingly. The second group is <strong>mask construction</strong>: <span class="mono">allocate_vocab_mask(...)</span> first allocates a mask buffer sized by vocab, batch, and device, then <span class="mono">fill_vocab_mask(vocab_mask, idx)</span> marks "which tokens are legal now" (idx says which request in the batch). That mask is then added onto the logits. The third group is <strong>rollback</strong>: <span class="mono">rollback(k)</span> backs the FSM up k steps, which is crucial in two scenarios—grammars that need backtracking, and speculative decoding (<span class="inline">Lesson 43</span>), where the draft model proposes several candidate tokens at once but some are rejected by verification, so the FSM must precisely undo the states it advanced through for those rejected tokens. Finally, <strong>termination</strong>: <span class="mono">is_terminated()</span> checks whether the FSM has reached an accept state—whether the structure is fully closed and generation can wrap up. These four groups look simple but map exactly onto every lifecycle event an FSM goes through during decoding: advance, query the current feasible set, rewind when needed, and check whether the end has been reached. Master them and you have grasped the entire outward contract of SGLang's constrained decoding.</p>

<div class="cellgroup"><div class="cell">S0 <span class="mono">{</span></div><div class="cell sc">S1 <span class="mono">"name"</span> forced</div><div class="cell sc">S2 <span class="mono">:</span> forced</div><div class="cell">S3 value: free sampling</div><div class="cell"><span class="mono">}</span> accept</div></div>

<table class="t"><tr><th>BaseGrammarObject method</th><th>Role</th></tr><tr><td><span class="mono">accept_token(token)</span></td><td>advance the FSM by the sampled token to the next state</td></tr><tr><td><span class="mono">allocate_vocab_mask(...)</span></td><td>allocate a mask buffer by vocab/batch/device</td></tr><tr><td><span class="mono">fill_vocab_mask(mask, idx)</span></td><td>mark which tokens keep the output legal now</td></tr><tr><td><span class="mono">rollback(k)</span></td><td>back up k steps (backtracking / spec-decode reject)</td></tr><tr><td><span class="mono">is_terminated()</span></td><td>has the FSM reached an accept state?</td></tr></table>

<h2>Jump-forward: skip tokens the grammar nails down</h2>
<p>So far constrained decoding guarantees <strong>correctness</strong> but is not yet <strong>fast</strong>. Observe a common phenomenon: often the FSM's "allowed set" in some state contains <strong>exactly one</strong> token, and the next state again only one, and the one after that still only one... In other words, the grammar is <strong>fully deterministic</strong> over this span, with no freedom at all. The classic case is a fixed key name in a JSON-Schema: once <span class="mono">{</span> is written, the schema forces the string <span class="mono">"name":</span> next, and the model has no choice here. Yet the step-by-step approach still <strong>calls the model forward for each</strong> of these nailed-down tokens to "predict" them—pure wasted compute, since the answer is predetermined. Such nailed-down fragments actually make up a large share of structured output: braces, quotes, colons, commas, fixed field names, and enum prefixes are all boilerplate the grammar imposes, and they often outnumber the genuinely free content fields.</p>
<p>SGLang's optimization is called <strong>jump-forward, backed by a compressed FSM</strong>. At compile time it identifies these "deterministic spans": any run of tokens with a unique continuation is pre-compressed and recorded. At runtime, once the FSM reaches such a state, SGLang no longer runs the model forward but <strong>splices the whole forced string directly into the output</strong> and jumps the FSM ahead by the corresponding steps at once. Concretely this lives in <span class="mono">OutlinesJumpForwardMap.jump_forward_symbol(state)</span>: given the current state, it returns the string the grammar nails down from there for the scheduler to splice in. This jump brings a double payoff: it is <strong>faster</strong>—skipping a run of meaningless forwards visibly speeds up structured generation; and it yields <strong>better quality</strong>—the model no longer gets distracted predicting boilerplate and can focus on the free fields that truly need its judgment. This optimization closes Part 10. Notably, the longer and more frequent the deterministic spans, the greater jump-forward's payoff—for complex schemas with many fields and fixed key names, the saved forwards are substantial and end-to-end latency drops noticeably.</p>

<div class="cols"><div class="col"><strong>Plain per-token decode</strong><br>For the nailed-down <span class="mono">" n a m e " :</span> it dutifully runs a model forward to "predict" each character—burning compute step by step even though there is only one answer.</div><div class="col"><strong>Jump-forward</strong><br>Detecting the unique continuation, <span class="mono">jump_forward_symbol</span> emits the whole <span class="mono">"name":</span> directly; the FSM jumps over these steps <strong>without calling the model</strong>—faster and steadier.</div></div>

<div class="fig">
  <svg viewBox="0 0 800 230" role="img" aria-label="jump-forward: a grammar-forced span is emitted at once, the FSM jumps the matching steps and skips the per-character model forwards">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">jump-forward: emit the forced span once</text>

    <text x="20" y="78" style="fill:var(--amber);font-size:12px">per-token</text>
    <rect x="120" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="146" y="77" text-anchor="middle" class="mono" style="font-size:12px">"</text>
    <rect x="180" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="206" y="77" text-anchor="middle" class="mono" style="font-size:12px">n</text>
    <rect x="240" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="266" y="77" text-anchor="middle" class="mono" style="font-size:12px">a</text>
    <rect x="300" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="326" y="77" text-anchor="middle" class="mono" style="font-size:12px">m</text>
    <rect x="360" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="386" y="77" text-anchor="middle" class="mono" style="font-size:12px">e</text>
    <rect x="420" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="446" y="77" text-anchor="middle" class="mono" style="font-size:12px">"</text>
    <rect x="480" y="56" width="52" height="32" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="506" y="77" text-anchor="middle" class="mono" style="font-size:12px">:</text>
    <text x="120" y="110" style="fill:var(--amber);font-size:12px">7 model forwards · wasted</text>

    <text x="20" y="170" style="fill:var(--teal);font-size:12px">jump-forward</text>
    <rect x="120" y="148" width="412" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="326" y="169" text-anchor="middle" class="mono" style="font-size:12px">"name": jumped at once</text>
    <rect x="548" y="148" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="608" y="169" text-anchor="middle" style="fill:var(--blue);font-size:12px">value: sample</text>
    <text x="120" y="202" style="fill:var(--teal);font-size:12px">1 step → saves 7 forwards</text>
  </svg>
  <div class="figcap"><b>Fig 2 · skip the nailed-down forced span</b> — when the FSM has only one continuation over a span, <span class="mono">jump_forward_symbol</span> splices the whole <span class="mono">"name":</span> in at once and jumps the matching steps, skipping the per-character model forwards; only then does it return to normal sampling for the free value.</div>
</div>

<div class="card"><div class="tag">🧪 Concrete example</div><p>Given the schema <span class="mono">{"name": string}</span>: once the FSM passes <span class="mono">{</span>, the next ~8 characters <span class="mono">"name":</span> are <strong>fully determined</strong>, with no branching. Per-token decode would run a model forward for each of those 8 characters; <span class="mono">jump_forward_symbol</span> instead splices the whole <span class="mono">"name":</span> in at once and the FSM jumps those 8 steps with <strong>zero forwards</strong>. Only the free <strong>value</strong> that follows (e.g. <span class="mono">"Ada"</span>) actually needs sampling.</p></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/constrained/base_grammar_backend.py ::BaseGrammarObject</span><span class="ln">one grammar FSM per request: mask logits to stay valid + rollback</span></div><pre>class BaseGrammarObject:                 # one per request: the compiled grammar FSM
    def accept_token(self, token):
        ...                              # advance the FSM by the sampled token
    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        ...                              # make a [vocab] mask buffer
    def fill_vocab_mask(self, vocab_mask, idx):
        ...                              # mark which tokens keep the output grammar-valid
    def rollback(self, k):
        ...                              # back up k steps (backtracking / spec-decode reject)
    def is_terminated(self):
        ...                              # has the grammar reached an accept state?
# jump-forward lives in constrained/outlines_jump_forward.py ::OutlinesJumpForwardMap</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/constrained/outlines_jump_forward.py ::OutlinesJumpForwardMap</span><span class="ln">per FSM state, precompute the forced string when only one path exists</span></div><pre>class OutlinesJumpForwardMap:
    def __init__(self, regex_string):
        # per FSM state, precompute the forced string for any stretch
        # where the path is deterministic (a single way forward).
        self.state_to_jump_forward = init_state_to_jump_forward(regex_string)
    def jump_forward_symbol(self, state):
        # walk while each state has ONE outgoing edge, concatenating
        # the forced characters.
        jump_forward_str = ""
        while state in self.state_to_jump_forward:
            e = self.state_to_jump_forward[state]
            jump_forward_str += e.symbol
            ...
        return jump_forward_str, next_state</pre></div>

<h2>Rollback, termination, and the speculative-decoding interplay</h2>
<p>Finally, three easily overlooked yet crucial details. First, <strong>rollback is not optional</strong>. In speculative decoding the draft model proposes several tokens at once and the target model verifies them in parallel, often accepting only the first few and rejecting the rest. The grammar FSM must assume all are accepted and advance accordingly before verification; once some are rejected it must use <span class="mono">rollback(k)</span> to precisely undo the extra k steps, keeping the FSM state strictly consistent with the truly accepted sequence—otherwise subsequent masks would be built from a wrong state and the constraint would fail. Second, <strong>termination decides when to stop</strong>: when <span class="mono">is_terminated()</span> returns true the structure is fully closed (e.g. the outermost JSON braces are balanced) and generation can safely stop without adding noise. Third, <strong>mask construction must be efficient</strong>: <span class="mono">allocate_vocab_mask</span> and <span class="mono">fill_vocab_mask</span> are designed to be batched on the GPU so that constrained decoding's overhead is largely hidden behind the forward compute, barely slowing throughput. This matters especially: if constrained decoding were implemented clumsily—judging each token's legality on the CPU every step and syncing back to the GPU—the overhead would easily eat the compute it saves. SGLang instead pushes "deciding legality" into compile time as much as possible, leaving the runtime with just one lightweight mask fill and one tensor add, so it guarantees strict structure at almost no throughput cost. Stitch advancing, masking, rollback, and termination together and you get the complete, sturdy skeleton of SGLang's constrained decoding.</p>
<p>One more design lesson is worth drawing out: SGLang deliberately abstracts all of this behind a single base class so that the concrete grammar engine is <strong>pluggable</strong>. Whether the underlying compiler is xgrammar, outlines, or llguidance, the scheduler talks only to the <span class="mono">BaseGrammarObject</span> interface—advance, mask, rollback, terminate. This means a faster or more expressive grammar engine can be dropped in without touching the decode loop, and different engines can even coexist for different request types. The same discipline that makes the masking efficient also makes the whole subsystem future-proof, which is exactly the kind of "designed-for-scale, swappable internals" philosophy that recurs throughout SGLang's architecture.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>Constrained decoding guarantees valid structure</strong>: compile JSON-Schema / regex / EBNF into an FSM and, before sampling, use a vocab mask to set illegal tokens' logits to −∞.</li>
<li><strong>It sits exactly between logits and the sampler</strong>: it edits logits (<span class="inline">Lesson 37</span>), not the sampling algorithm (<span class="inline">Lesson 28</span>), so it stacks with any sampling strategy.</li>
<li><strong>BaseGrammarObject is the per-request FSM</strong>: <span class="mono">accept_token</span> advances, <span class="mono">fill_vocab_mask</span> builds the mask, <span class="mono">rollback</span> backs up, <span class="mono">is_terminated</span> checks the accept state.</li>
<li><strong>Jump-forward skips forced tokens' forwards</strong>: the compressed FSM uses <span class="mono">jump_forward_symbol</span> to emit the forced string directly—faster, and the model stays focused for better quality.</li>
<li><strong>Rollback underpins speculative decoding</strong>: rejected draft tokens need <span class="mono">rollback(k)</span> to precisely restore the FSM state (<span class="inline">Lesson 43</span>).</li>
</ul></div>

<div class="card"><div class="tag">🏁 Part 10 wrap-up</div><p>Part 10 centered on "how to push SGLang toward large scale and high performance": speculative decoding lands multiple tokens at once via draft+verify, and EAGLE further raises acceptance with feature-level drafting; PD disaggregation splits prefill and decode onto separate instances so each saturates compute and memory bandwidth; tensor/pipeline/data/expert parallelism lets giant models span many GPUs; EPLB dynamically balances expert load in MoE to erase hotspots; and this lesson's structured outputs with jump-forward guarantees strictly valid output while skipping pointless forwards. These techniques look like separate concerns but jointly answer one question—<strong>spend every bit of compute where it matters without sacrificing correctness</strong>—and thereby push SGLang steadily to a large-scale, high-performance production level. With that, Part 10's puzzle is fully assembled: from "compute faster" to "compute cheaper" to "compute both correctly and stably," SGLang answers, with a set of interlocking mechanisms, every core challenge of bringing large models into real production.</p></div>
"""}
