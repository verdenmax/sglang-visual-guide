"""Part 13 · Design themes / synthesis (L58-63).

The final, synthesis part of the SGLang visual guide: six reflective "theme" lessons that
zoom out from individual components to the recurring design principles — RadixAttention as a
first-class idea, zero-overhead scheduling, two workloads/one engine, draft-for-parallel-verify,
everything pluggable, and built for throughput. L63 closes the whole guide.
Each LESSON_XX is a {"zh": html, "en": html} dict consumed via registry.CONTENT.
"""

LESSON_58 = {"zh": r"""
<p class="lead">如果你把前面三十多课像拼图一样摊开，会发现有一块图案反复出现——<strong>共享前缀</strong>。它不是某一处的小聪明，而是 SGLang 从语言层一路贯穿到显存层的<strong>一等公民（first-class idea）</strong>：把所有请求看成一棵<span class="mono">共享前缀的基数树</span>，匹配最长的已缓存前缀、复用它的 KV、只为新出现的后缀做计算。看懂这一个念头，等于一次性串起本指南三分之一的内容。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一座<strong>家族图书馆</strong>。每本新书并不是从第一页重新印刷，而是先去书架上找：是否已经有一本书，前面几百页和我完全一样？如果有，就直接把那叠相同的纸张<strong>借过来共用</strong>，只单独印刷从分叉处开始的新章节。书架本身被组织成一棵<strong>树</strong>：树根是所有书共同的开头，越往下走分叉越多，每个分叉点保存着「从这里开始大家就不一样了」的位置。</p>
<p>更妙的是，这套「共享开头」的规矩不只是图书管理员一个人的私房技巧。写书的人（DSL）会<strong>故意</strong>让几个版本共用同一段开头；图书馆的扩建蓝图（缓存概念）一开始就写明要这么做；书架的物理格子（分页显存）按这套规矩摆放；连排队借书的顺序（调度）都优先照顾「开头能对上库存」的人；甚至当架子放不下时，旧书会被<strong>分层</strong>挪到地下室、再挪到外仓——但树形结构始终不变。这就是「一等公民」的意思：同一个念头被反复、刻意地写进系统的每一层，而不是临时想起来才补一笔。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>很多系统会把「前缀缓存」当成事后补丁：先把推理引擎写完，再在旁边贴一个 LRU 字典缓一缓。SGLang 走的是相反的路——它把<strong>「KV 缓存本身就是一棵共享前缀的基数树」</strong>当作核心数据结构，整个引擎围着它转。<span class="mono">RadixAttention</span> 不是一个孤立的算子，而是一种<strong>组织全局的视角</strong>：DSL 的 <span class="mono">fork/join</span>（<strong>第11课</strong>）在语言层制造共享前缀；前缀缓存的概念（<strong>第7课</strong>）在早期就被立起来；基数树的实现（<strong>第29课</strong>）让缓存真正变成 <span class="mono">RadixCache</span> 与 <span class="mono">TreeNode</span>；分页 KV（<strong>第30课</strong>，承接 <strong>第6课</strong>）是这些节点指向的物理页；缓存感知调度（<strong>第20课</strong>）让命中率高的请求优先跑、好让缓存真正赚回成本；HiCache（<strong>第31课</strong>）则把同一棵前缀树分层铺到 GPU/CPU/磁盘。把它们连起来看，是一张网，而不是一串孤点；少了任何一根线，这整张网都会松垮、不再成形。</p>
</div>

<h2>一、为什么「共享前缀」配得上「一等公民」这个称号</h2>
<p>判断一个想法是不是系统的一等公民，标准很简单：它有没有被<strong>刻意、反复</strong>地写进多个互不相邻的层次，并且改动它会牵动全局。共享前缀完全符合。它最早只是一个朴素的观察：在真实负载里，大量请求的开头是<strong>重复</strong>的——同一段系统提示词、同一份少样本示例、同一棵对话历史、同一次 <span class="mono">fork</span> 出来的多个分支。既然开头一样，它们在注意力计算中产生的 KV 也<strong>逐字节相同</strong>。重复计算这些 KV，是在白白烧显存和算力。</p>
<p>SGLang 没有把这件事降级成「可有可无的优化」，而是把它升格成<strong>数据结构的形状</strong>。一旦你接受「缓存是一棵树」，许多设计就不再是巧合：<span class="mono">match_prefix</span> 自然就是「沿树走到最深的已知节点」；<span class="mono">insert</span> 自然就是「在前缀分叉处把节点劈开」；<span class="mono">evict</span> 自然就是「从叶子往回收，因为叶子才是没人共享的尾巴」；而 <span class="mono">inc_lock_ref</span> 自然就是「正在用的节点要钉住，别被回收」。树这个形状一旦确立，算法是被它「逼」出来的，而不是临时拼凑、东拼西凑硬搭出来的。</p>
<p>反过来想也成立：如果当初没把它当一等公民，会发生什么？你大概会先写一个朴素的逐请求计算引擎，然后在某天发现「系统提示词被重复算了上千遍」，于是匆忙加一个哈希表缓存整段提示的 KV。但很快你就会撞墙——用户的提示只有<strong>前缀</strong>相同、后面各不一样，整段哈希根本命中不了；于是你被迫去做<strong>最长公共前缀</strong>匹配，而高效地对海量序列做前缀匹配，最自然的数据结构恰恰就是<strong>基数树</strong>。也就是说，只要你认真对待「前缀复用」这件事，几乎必然会被推向同一个终点。SGLang 的聪明之处，是<strong>从一开始</strong>就承认这个终点、直接把引擎建在它上面，省去了所有「先做错、再补救」的弯路。这就是把正确的念头提前升格为结构的价值。</p>

<p>这也回答了一个常被忽略的问题：为什么 SGLang 敢在如此多的地方依赖前缀共享，而不担心它「偶尔不命中就白忙一场」？因为当它是结构而非外挂时，不命中只是退化成「整段都是新后缀」的普通情形——树照样能容纳、插入、回收，没有任何特殊路径需要兜底。一个被升格为数据结构的念头，天然具备<strong>优雅降级</strong>的能力：命中多就省得多，命中少也不会更糟，绝不会因为「缓存没准备好」而崩。这种「最坏情况也不变坏」的鲁棒性，正是把它做成一等公民才换来的、最容易被忽视却最珍贵的红利。</p>

<h2>二、同一个念头，在每一层换了张面孔</h2>
<p>真正让人拍案的是：<strong>共享前缀</strong>在不同层次会<strong>换装登场</strong>，但骨子里是同一件事。在 DSL 层（<strong>第11课</strong>），你写 <span class="mono">fork</span> 让一个提示分裂成几条推理分支，<span class="mono">join</span> 再把它们汇合——你是在<strong>语言层显式制造</strong>一个共享前缀，期待底层别重复算。在概念层（<strong>第7课</strong>），「前缀缓存」作为一个承诺被提出：相同开头只算一次。在实现层（<strong>第29课</strong>），这个承诺落地成 <span class="mono">RadixCache</span>，每个 <span class="mono">TreeNode</span> 的边上挂着一段 token-id 跨度（<span class="mono">key</span>）和对应的 KV 索引（<span class="mono">value</span>）。在物理层（<strong>第30课</strong>，建立在 <strong>第6课</strong> 的分页之上），节点的 <span class="mono">value</span> 指向的是一页页 paged-KV。在调度层（<strong>第20课</strong>），调度器会偏向那些<strong>前缀命中率高</strong>的请求，因为让缓存被复用，它才算没白存。在分层存储层（<strong>第31课</strong>），HiCache 把这棵树按冷热分层，热的留在 GPU、温的下放 CPU、冷的写到磁盘。</p>
<p>六张面孔，一个灵魂。它们之所以能严丝合缝地咬合，正是因为大家都默认了同一个共享前缀的世界观。你若只学某一课，会觉得它是个局部技巧；你若把第7、11、20、29、30、31课叠在一起看，才会看见那条贯穿始终的主轴。</p>
<p>不妨顺着一条具体的链路把它走一遍。一个多轮对话应用，开发者在 DSL 里用 <span class="mono">fork</span>（第11课）把同一段历史分裂成三条候选回复；这三条在概念上（第7课）共享同一个开头，所以引擎只应算一次。落到实现（第29课），这段历史变成 <span class="mono">RadixCache</span> 里一个被三条请求共享的 <span class="mono">TreeNode</span>，它的 <span class="mono">value</span> 指向若干物理页（第30课，承第6课）。调度器（第20课）看到这三条请求前缀高度重合，便把它们排在一起、几乎同时跑，让那段共享 KV 在显存里「热」着被反复命中。等对话变长、显存吃紧，HiCache（第31课）再把这段已经不那么热的历史整段下放到 CPU 甚至磁盘，需要时再整段调回。你看，从开发者敲下 <span class="mono">fork</span> 的那一刻，到这段 KV 在三级存储间迁徙，<strong>自始至终是同一棵树上的同一个节点</strong>在被不同层次以不同方式照料。这就是「一个念头贯穿全栈」最具体的样子。</p>

<div class="fig">
  <svg viewBox="0 0 780 320" role="img" aria-label="放射网：中心的 RadixAttention 前缀共享，向外连到 DSL 的 fork、KV 缓存的基数树、调度器的 LPM 缓存感知排序、HiCache 的内存分层">
    <line x1="390" y1="160" x2="270" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="390" y1="160" x2="510" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="390" y1="160" x2="270" y2="224" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="390" y1="160" x2="510" y2="224" style="stroke:var(--line);stroke-width:1.5"/>
    <ellipse cx="390" cy="160" rx="118" ry="42" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="390" y="153" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">RadixAttention</text>
    <text x="390" y="175" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">前缀共享 · 一个念头</text>
    <rect x="30" y="40" width="240" height="56" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="44" y="64" style="fill:var(--blue);font-weight:700">DSL（第11课）</text>
    <text x="44" y="84" style="font-size:12px">fork 共用提示前缀</text>
    <rect x="510" y="40" width="240" height="56" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="524" y="64" style="fill:var(--teal);font-weight:700">KV 缓存（第29课）</text>
    <text x="524" y="84" style="font-size:12px">基数树存前缀</text>
    <rect x="30" y="224" width="240" height="56" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="44" y="248" style="fill:var(--amber);font-weight:700">调度器（第20课）</text>
    <text x="44" y="268" style="font-size:12px">LPM 缓存感知排序</text>
    <rect x="510" y="224" width="240" height="56" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="524" y="248" style="fill:var(--purple);font-weight:700">内存分层（第31课）</text>
    <text x="524" y="268" style="font-size:12px">HiCache 留驻热前缀</text>
  </svg>
  <div class="figcap"><b>图 1 · 放射网：同一个念头反复出现</b> — 中心的 <span class="mono">RadixAttention</span>（前缀共享）向四方辐射：DSL 的 <span class="mono">fork</span> 制造共享前缀、KV 缓存把它存成基数树、调度器用 LPM 缓存感知排序优先放行命中、HiCache 把热前缀留驻内存。一个念头，处处复用。</div>
</div>

<h2>三、从「贴上去的优化」到「长在骨架里的结构」</h2>
<p>差别不只是审美。把前缀共享当外挂，意味着缓存和引擎是两套各自为政的逻辑，缓存对调度一无所知，调度也不会为缓存让路，命中纯靠运气。把它当一等公民，意味着调度器知道树的形状、能预估命中、能主动把相似请求排在一起；意味着显存管理器知道哪些页被多少请求共享、不能随便回收；意味着分层存储能沿着树的边界整段搬运而不是零散拷贝。一个念头被升格为结构后，全系统才能围绕它协同。</p>
<p>这也解释了为什么 <span class="mono">RadixAttention</span> 这个名字里同时含着「Radix（基数树）」和「Attention（注意力）」：注意力计算消费 KV，而 KV 被组织成一棵基数树。算子和数据结构在这里是<strong>同一枚硬币的两面</strong>，不是先后关系。</p>
<p>还有一个容易被低估的好处：当缓存是全系统共享的一棵树，<strong>跨请求</strong>的复用就成了默认行为，而不是额外功能。两个毫不相干的用户，只要碰巧用了同一段系统提示词，就会自动落在同一个共享前缀节点上，第二个人省下的计算是「免费」的。把这个能力换成外挂式缓存，你得额外设计一套跨请求的查找与去重逻辑；而在树形结构里，它只是 <span class="mono">match_prefix</span> 自然走到了同一个节点而已。换句话说，一等公民的设计让「最值钱的那部分复用」变成了零成本的副产品——这正是它在真实高并发负载下能省下大量算力的根本原因。</p>

<h2>四、一次请求，是怎么在这棵树上走完一生的</h2>
<p>把视角拉回单个请求：它带着一串 token 进来，先做 <span class="mono">match_prefix</span>，沿树往下走，找到与库存重合的<strong>最长前缀</strong>；这段前缀的 KV 被原样<strong>复用</strong>（并 <span class="mono">inc_lock_ref</span> 钉住，防止半路被回收）；引擎只对<strong>新出现的后缀</strong>真正做前向计算；算完后通过 <span class="mono">insert</span> 把这段新后缀挂回树上，在分叉处把节点劈开，让未来的请求又能共享到它。这条「匹配最长前缀 → 复用 KV → 只算新后缀 → 写回树」的生命周期，就是 <span class="mono">RadixAttention</span> 的全部精髓，下面四张图把它从不同侧面拆开看。</p>
<p>这里有两个细节值得停下来体会。其一，<span class="mono">inc_lock_ref</span> 的「钉住」之所以必要，是因为复用和回收在同一棵树上并存：当请求 A 正靠着某段共享前缀往下算时，回收逻辑（<span class="mono">evict</span>）绝不能把这段前缀的页抢走，否则 A 算到一半就「踩空」了。引用计数让「谁在用」变得可见，回收只敢从无人引用的叶子下手。其二，<span class="mono">insert</span> 的「劈开节点」是树能持续长出共享的关键：当新后缀与某个已有节点只共享前半段时，系统会在分叉处把老节点一分为二——公共的上半段继续被双方共享，各自不同的下半段成为两个孩子。正是这个「按需劈开」的动作，让基数树始终保持「能共享的尽量共享、该分叉的精确分叉」的紧凑形态，而不会退化成一堆互不相干的整串。理解了钉住与劈开，你就理解了这棵树<strong>为什么既安全又高效</strong>。</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="匹配复用只算新后缀：请求 tokens 分成共享前缀与新后缀，match_prefix 命中前缀复用其 KV 零计算，模型只前向新后缀">
    <text x="40" y="36" style="font-weight:700;fill:var(--muted)">请求 tokens ＝ [ 共享前缀 ｜ 新后缀 ]</text>
    <rect x="40" y="62" width="420" height="48" rx="8" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="250" y="84" text-anchor="middle" style="fill:var(--muted);font-weight:700">共享前缀</text>
    <text x="250" y="102" text-anchor="middle" style="fill:var(--faint);font-size:12px">已缓存 KV · 灰</text>
    <rect x="468" y="62" width="272" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:2"/>
    <text x="604" y="84" text-anchor="middle" style="fill:var(--amber);font-weight:700">新后缀</text>
    <text x="604" y="102" text-anchor="middle" style="fill:var(--amber);font-size:12px">唯一要算的部分</text>
    <line x1="40" y1="124" x2="460" y2="124" style="stroke:var(--teal);stroke-width:1.5"/>
    <text x="40" y="152" style="fill:var(--teal);font-weight:700">match_prefix 命中</text>
    <text x="40" y="172" style="font-size:12px">复用其 KV · 0 计算</text>
    <line x1="468" y1="124" x2="740" y2="124" style="stroke:var(--amber);stroke-width:1.5"/>
    <text x="468" y="152" style="fill:var(--amber);font-weight:700">只对新后缀前向计算</text>
    <text x="468" y="172" style="font-size:12px">算量 ＝ 后缀长度</text>
    <rect x="40" y="208" width="360" height="56" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="56" y="232" style="fill:var(--teal);font-weight:700">基数树：此前缀已驻留</text>
    <text x="56" y="252" style="font-size:12px">沿树走到最深已知节点，直接复用</text>
    <rect x="468" y="208" width="272" height="56" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="484" y="232" style="fill:var(--accent-ink);font-weight:700">省下的就是前缀那段</text>
    <text x="484" y="252" style="font-size:12px">而非整条序列重算</text>
  </svg>
  <div class="figcap"><b>图 2 · 匹配 → 复用 → 只算新后缀</b> — 请求的 token 切成「共享前缀（灰、已缓存）｜新后缀（高亮）」；<span class="mono">match_prefix</span> 在树里找到前缀、原样复用它的 KV（0 计算），模型只对<strong>新后缀</strong>做前向。算量等于后缀长度，而不是整条序列。</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.insert</span><span class="ln">把一段 token+KV 插入树，让后续请求共享该前缀</span></div><pre>def insert(self, params: InsertParams) -&gt; InsertResult:
    # 把一条序列的 token + KV 写进基数树，让后续请求能共享这段前缀；
    # key 先做分页对齐（page_size 的整数倍）。
    key = params.key.page_aligned(self.page_size)
    value = params.value[:len(key)]
    prefix_len, last_node = self._insert_helper(
        self.root_node, key, value, ...)   # 在分叉处劈开节点
    return InsertResult(prefix_len=prefix_len)   # 有多少前缀本就存在</pre></div>

<div class="card detail"><div class="tag">🧪 具体例子</div>
<p>设想一段被所有请求共用的 <strong>1000 token 系统提示词</strong>。第一条请求来时，它被 <span class="mono">insert</span> 进树一次，对应的 KV 落在若干分页上。第二条请求带着同样的开头进来，<span class="mono">match_prefix</span> 一口气命中全部 1000 个 token——<strong>0 次重算</strong>——引擎只对它自己那句问题做前向。第三、第四条也照此办理：那段共享前缀越被复用，省下的算力越多。</p>
<p>而这<strong>同一个基数树念头</strong>同时撑起了四处：DSL 的 <span class="mono">fork</span> 让分支共享开头、缓存把前缀存成树、LPM 缓存感知调度把命中高的请求排在一起、HiCache 再按冷热把这棵树分层铺到 GPU/CPU/磁盘。一个 <span class="mono">insert</span>，喂养了全栈。</p>
</div>

<table class="t">
<tr><th>共享前缀出现的地方</th><th>它在这里扮演的角色</th></tr>
<tr><td>DSL 的 fork/join（<strong>第11课</strong>）</td><td>语言层<strong>故意</strong>让多个分支共用同一段开头，是「共享前缀」的源头</td></tr>
<tr><td>前缀缓存概念（<strong>第7课</strong>）</td><td>早期立下的承诺：相同开头只算一次，为后续实现定调</td></tr>
<tr><td>基数树实现（<strong>第29课</strong>）</td><td>承诺落地：<span class="mono">RadixCache</span> 由 <span class="mono">TreeNode</span> 组成，缓存就是这棵树</td></tr>
<tr><td>分页 KV（<strong>第30课</strong>，承 <strong>第6课</strong>）</td><td>树节点的 <span class="mono">value</span> 指向的物理页，是前缀真正存放的地方</td></tr>
<tr><td>缓存感知调度（<strong>第20课</strong>）</td><td>优先放行前缀命中率高的请求，让缓存的投入真正被赚回</td></tr>
<tr><td>HiCache 分层（<strong>第31课</strong>）</td><td>把同一棵前缀树按冷热分层铺到 GPU/CPU/磁盘</td></tr>
</table>

<div class="layers">
<div class="layer">根 root：所有请求共同的起点（空前缀）</div>
<div class="layer">共享前缀节点：<span class="mono">"你是一个有用的助手……"</span> 的 token 跨度 + KV 索引，被许多请求共用</div>
<div class="layer">分叉点：从这里 token 开始不同，节点被劈开成多个孩子</div>
<div class="layer">分支 A 叶子：用户问题甲的独有后缀</div>
<div class="layer">分支 B 叶子：用户问题乙的独有后缀（与 A 共享上面整段前缀）</div>
</div>

<div class="cols">
<div class="col"><strong>贴上去的优化（bolt-on）</strong><br/>缓存是引擎旁的一个 LRU 字典；调度对它一无所知；命中靠运气；分层只能整块零散拷贝；改缓存不牵动全局。</div>
<div class="col"><strong>一等公民数据结构（first-class）</strong><br/>缓存就是一棵基数树；调度能预估命中并主动聚类；显存按共享关系钉住/回收；分层沿树边界整段搬运；一个念头统一全系统。</div>
</div>

<div class="flow">
<div class="node">请求带 token 进来</div>
<div class="arrow">→</div>
<div class="node">match_prefix 匹配最长已缓存前缀</div>
<div class="arrow">→</div>
<div class="node">复用该前缀的 KV（inc_lock_ref 钉住）</div>
<div class="arrow">→</div>
<div class="node">只对新后缀做前向计算</div>
<div class="arrow">→</div>
<div class="node">insert 写回树，分叉处劈开节点</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache</span><span class="ln">KV 缓存就是一棵共享前缀的基数树</span></div><pre>class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)  # edge -&gt; child: the radix tree itself
        self.key = None                        # the token-id span on the edge into this node
        self.value = None                      # the KV cache indices for that span

class RadixCache(BasePrefixCache):
    # the prefix cache IS a radix tree of shared-prefix nodes
    def match_prefix(self, params):   # walk the tree; return the longest already-cached prefix
        ...
    def insert(self, params):         # add a sequence's KV, splitting nodes where prefixes diverge
        ...
    def evict(self, params):          # LRU-evict evictable leaves to reclaim pages
        ...
    def inc_lock_ref(self, node):     # pin in-use nodes so eviction can't take them
        ...</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>一等公民，而非外挂：</strong>SGLang 把「KV 缓存是一棵<span class="mono">共享前缀的基数树</span>」当作核心数据结构，整个引擎围着它转，而不是事后贴一个 LRU 缓存。</li>
<li><strong>同一念头的六张面孔：</strong>DSL fork/join（<strong>第11课</strong>）、前缀缓存概念（<strong>第7课</strong>）、基数树实现（<strong>第29课</strong>）、分页 KV（<strong>第30课</strong>，承 <strong>第6课</strong>）、缓存感知调度（<strong>第20课</strong>）、HiCache 分层（<strong>第31课</strong>）讲的都是同一件事。</li>
<li><strong>生命周期：</strong>匹配最长前缀 → 复用 KV → 只算新后缀 → 写回树，是 <span class="mono">RadixAttention</span> 的全部精髓。</li>
<li><strong>结构逼出算法：</strong>一旦缓存是树，<span class="mono">match_prefix/insert/evict/inc_lock_ref</span> 就是树形状的自然推论。</li>
<li><strong>看懂它即串起三分之一指南：</strong>识别这一个组织性念头，前面许多看似独立的课会突然连成一张网。</li>
</ul></div>
""", "en": r"""
<p class="lead">If you spread out the first thirty-odd lessons like puzzle pieces, one pattern keeps reappearing—<strong>shared prefixes</strong>. It is not a local trick tucked in one corner; it is a <strong>first-class idea</strong> that runs from SGLang's language layer all the way down to its memory layer: treat every request as a node in one <span class="mono">radix tree of shared prefixes</span>, match the longest already-cached prefix, reuse its KV, and compute only the newly-appeared suffix. Grasp this single idea and you stitch together a third of the whole guide at once.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a <strong>family library</strong>. A new book is not reprinted from page one; it first checks the shelves: is there already a book whose first few hundred pages are identical to mine? If so, it simply <strong>borrows and shares</strong> that identical stack of paper and only prints the new chapters that begin at the point of divergence. The shelf itself is organized as a <strong>tree</strong>: the root is the opening common to all books, branching grows as you descend, and each fork records the position where "from here on we differ."</p>
<p>Even better, this "share the opening" rule is not one librarian's private habit. The authors (the DSL) <strong>deliberately</strong> make several versions share the same opening; the library's expansion blueprint (the cache concept) states this intent from the start; the physical shelf slots (paged memory) are laid out by this rule; even the borrowing queue (scheduling) favors readers whose opening matches the stock; and when shelves overflow, old books are <strong>tiered</strong> down to the basement, then to off-site storage—yet the tree shape never changes. That is what "first-class" means: the same idea written, on purpose and repeatedly, into every layer of the system, rather than patched in as an afterthought.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Many systems treat the "prefix cache" as an afterthought patch: finish the inference engine, then bolt an LRU dictionary onto the side. SGLang takes the opposite path—it makes <strong>"the KV cache itself is a radix tree of shared prefixes"</strong> the core data structure, and the entire engine revolves around it. <span class="mono">RadixAttention</span> is not an isolated operator but a <strong>way of organizing everything</strong>: the DSL's <span class="mono">fork/join</span> (<strong>Lesson 11</strong>) manufactures shared prefixes at the language layer; the prefix-cache concept (<strong>Lesson 7</strong>) is raised early on; the radix-tree implementation (<strong>Lesson 29</strong>) makes the cache literally a <span class="mono">RadixCache</span> of <span class="mono">TreeNode</span>s; paged KV (<strong>Lesson 30</strong>, building on <strong>Lesson 6</strong>) is the physical pages those nodes point at; cache-aware scheduling (<strong>Lesson 20</strong>) lets high-hit requests run first so the cache pays off; and HiCache (<strong>Lesson 31</strong>) tiers that same prefix tree across GPU/CPU/disk. Connected, they form a web, not a string of isolated dots; remove any one thread and the web sags.</p>
</div>

<h2>1. Why "shared prefixes" earns the title "first-class"</h2>
<p>The test for whether an idea is a system's first-class citizen is simple: is it written, <strong>on purpose and repeatedly</strong>, into multiple non-adjacent layers, and does changing it ripple across the whole system? Shared prefixes pass cleanly. It started as a plain observation: in real workloads, many requests begin <strong>identically</strong>—the same system prompt, the same few-shot examples, the same conversation history, the same several branches <span class="mono">fork</span>ed from one prompt. Since the openings match, the KV they produce in attention is <strong>byte-for-byte identical</strong>. Recomputing that KV burns memory and compute for nothing.</p>
<p>SGLang did not demote this to an "optional optimization"; it promoted it to <strong>the shape of a data structure</strong>. Once you accept "the cache is a tree," many designs stop being coincidences: <span class="mono">match_prefix</span> is naturally "walk to the deepest known node," <span class="mono">insert</span> is naturally "split a node where prefixes diverge," <span class="mono">evict</span> is naturally "reclaim from the leaves, because leaves are the tails no one shares," and <span class="mono">inc_lock_ref</span> is naturally "pin nodes in use so eviction can't take them." Once the tree shape is fixed, the algorithms are <strong>forced out</strong> by it rather than cobbled together.</p>
<p>The reverse thought confirms it: what if it had not been treated as first-class from the start? You would likely first write a naive per-request engine, then one day discover "the system prompt was recomputed a thousand times," and hastily add a hash-table cache of the whole prompt's KV. But you would soon hit a wall—users' prompts share only the <strong>prefix</strong> while the rest differs, so whole-string hashing never hits; you would be forced into <strong>longest-common-prefix</strong> matching, and the most natural data structure for efficiently prefix-matching a flood of sequences is precisely a <strong>radix tree</strong>. In other words, take "prefix reuse" seriously and you are almost inevitably pushed to the same destination. SGLang's cleverness is to acknowledge that destination <strong>from the outset</strong> and build the engine directly on it, skipping all the "do it wrong, then patch" detours. That is the value of promoting the right idea to a structure early.</p>

<p>This also answers an often-overlooked question: why does SGLang dare to depend on prefix sharing in so many places without worrying that "an occasional miss wastes all the effort"? Because when it is a structure rather than a bolt-on, a miss merely degrades to the ordinary case of "the whole thing is a new suffix"—the tree can still hold, insert, and reclaim it, with no special fallback path needed. An idea promoted to a data structure naturally has <strong>graceful degradation</strong>: more hits save more, fewer hits are no worse, and it never crashes because "the cache wasn't ready." This "the worst case never gets worse" robustness is precisely the most overlooked yet most precious dividend of making it first-class.</p>

<h2>2. One idea, a different face at every layer</h2>
<p>What is truly striking is that <strong>shared prefixes</strong> appear <strong>in costume</strong> at different layers while staying the same thing underneath. At the DSL layer (<strong>Lesson 11</strong>), you write <span class="mono">fork</span> to split one prompt into branches and <span class="mono">join</span> to merge them—you are <strong>explicitly manufacturing</strong> a shared prefix at the language layer, expecting the engine not to recompute it. At the conceptual layer (<strong>Lesson 7</strong>), the "prefix cache" is offered as a promise: identical openings are computed once. At the implementation layer (<strong>Lesson 29</strong>), that promise becomes <span class="mono">RadixCache</span>, where each <span class="mono">TreeNode</span> carries a token-id span on its edge (<span class="mono">key</span>) and the matching KV indices (<span class="mono">value</span>). At the physical layer (<strong>Lesson 30</strong>, built on the paging of <strong>Lesson 6</strong>), a node's <span class="mono">value</span> points at pages of paged-KV. At the scheduling layer (<strong>Lesson 20</strong>), the scheduler favors requests with <strong>high prefix hit rate</strong>, because reuse is what makes the stored KV worthwhile. At the tiered-storage layer (<strong>Lesson 31</strong>), HiCache tiers that tree by temperature—hot on GPU, warm on CPU, cold on disk.</p>
<p>Six faces, one soul. They mesh so tightly precisely because they all assume the same shared-prefix worldview. Study any single lesson and it looks like a local trick; stack Lessons 7, 11, 20, 29, 30, and 31 together and you finally see the spine running through them all.</p>
<p>It helps to walk one concrete chain end to end. In a multi-turn chat app, the developer uses <span class="mono">fork</span> in the DSL (Lesson 11) to split the same history into three candidate replies; conceptually (Lesson 7) the three share one opening, so the engine should compute it once. In implementation (Lesson 29), that history becomes one <span class="mono">TreeNode</span> in <span class="mono">RadixCache</span> shared by three requests, whose <span class="mono">value</span> points at several physical pages (Lesson 30, on Lesson 6). The scheduler (Lesson 20), seeing the three requests' prefixes overlap heavily, places them together and runs them almost simultaneously, keeping that shared KV "hot" in memory for repeated hits. As the conversation grows and memory tightens, HiCache (Lesson 31) tiers that now-cooler history down to CPU or even disk, recalling it as a whole segment when needed. From the moment the developer types <span class="mono">fork</span> to the migration of that KV across three storage tiers, it is <strong>the same node on the same tree</strong> being tended by different layers in different ways. That is the most concrete picture of "one idea running through the full stack."</p>

<div class="fig">
  <svg viewBox="0 0 780 320" role="img" aria-label="Radial web: the central RadixAttention prefix sharing radiates to the DSL fork, the KV cache radix tree, the scheduler's cache-aware LPM ordering, and HiCache memory tiers">
    <line x1="390" y1="160" x2="270" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="390" y1="160" x2="510" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="390" y1="160" x2="270" y2="224" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="390" y1="160" x2="510" y2="224" style="stroke:var(--line);stroke-width:1.5"/>
    <ellipse cx="390" cy="160" rx="118" ry="42" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="390" y="153" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">RadixAttention</text>
    <text x="390" y="175" text-anchor="middle" style="fill:var(--accent-ink);font-size:12px">prefix sharing · one idea</text>
    <rect x="30" y="40" width="240" height="56" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="44" y="64" style="fill:var(--blue);font-weight:700">DSL (L11)</text>
    <text x="44" y="84" style="font-size:12px">fork shares a prefix</text>
    <rect x="510" y="40" width="240" height="56" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="524" y="64" style="fill:var(--teal);font-weight:700">KV cache (L29)</text>
    <text x="524" y="84" style="font-size:12px">radix tree of prefixes</text>
    <rect x="30" y="224" width="240" height="56" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="44" y="248" style="fill:var(--amber);font-weight:700">Scheduler (L20)</text>
    <text x="44" y="268" style="font-size:12px">cache-aware LPM order</text>
    <rect x="510" y="224" width="240" height="56" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="524" y="248" style="fill:var(--purple);font-weight:700">Memory tiers (L31)</text>
    <text x="524" y="268" style="font-size:12px">HiCache keeps hot ones</text>
  </svg>
  <div class="figcap"><b>Fig 1 · The radial web: one idea recurring everywhere</b> — the central <span class="mono">RadixAttention</span> (prefix sharing) radiates outward: the DSL's <span class="mono">fork</span> manufactures shared prefixes, the KV cache stores them as a radix tree, the scheduler orders by cache-aware LPM so hits run first, and HiCache keeps hot prefixes in memory. One idea, reused everywhere.</div>
</div>

<h2>3. From "an optimization glued on" to "a structure grown into the skeleton"</h2>
<p>The difference is not merely aesthetic. Treating prefix sharing as a bolt-on means the cache and the engine are two separate logics: the cache knows nothing about scheduling, scheduling never yields for the cache, and hits are pure luck. Treating it as first-class means the scheduler knows the tree's shape, can estimate hits, and can actively cluster similar requests; it means the memory manager knows how many requests share which pages and must not reclaim them carelessly; it means tiered storage can move whole segments along tree boundaries instead of scattered copies. Only after an idea is promoted to a structure can the whole system coordinate around it.</p>
<p>This also explains why the name <span class="mono">RadixAttention</span> holds both "Radix (radix tree)" and "Attention": attention consumes KV, and KV is organized as a radix tree. The operator and the data structure here are <strong>two sides of one coin</strong>, not a before-and-after.</p>
<p>There is another easily-underestimated benefit: when the cache is one tree shared across the whole system, <strong>cross-request</strong> reuse becomes the default behavior rather than an extra feature. Two completely unrelated users, as long as they happen to use the same system prompt, automatically land on the same shared-prefix node, and the compute the second one saves is "free." Replace this with a bolt-on cache and you must design a separate cross-request lookup-and-dedup logic; in the tree structure it is just <span class="mono">match_prefix</span> naturally walking to the same node. In other words, the first-class design turns "the most valuable reuse" into a zero-cost by-product—which is exactly why it saves enormous compute under real high-concurrency workloads.</p>

<h2>4. How a single request lives out its life on this tree</h2>
<p>Zoom back to one request: it arrives with a token sequence, first runs <span class="mono">match_prefix</span>, walks down the tree, and finds the <strong>longest prefix</strong> overlapping with the stock; that prefix's KV is <strong>reused</strong> as-is (and <span class="mono">inc_lock_ref</span> pins it so it won't be evicted mid-flight); the engine truly forward-computes only the <strong>newly-appeared suffix</strong>; afterward <span class="mono">insert</span> hangs that new suffix back onto the tree, splitting a node at the divergence so future requests can share it too. This lifecycle—"match the longest prefix → reuse KV → compute only the new suffix → write back to the tree"—is the entire essence of <span class="mono">RadixAttention</span>, and the four diagrams below pull it apart from different angles.</p>
<p>Two details deserve a pause. First, the "pinning" of <span class="mono">inc_lock_ref</span> is necessary because reuse and reclamation coexist on the same tree: while request A is computing down some shared prefix, the reclamation logic (<span class="mono">evict</span>) must never snatch that prefix's pages away, or A would "step into the void" mid-computation. Reference counting makes "who is using it" visible, so eviction only dares touch leaves no one references. Second, the "node splitting" of <span class="mono">insert</span> is the key to the tree continually growing shared structure: when a new suffix shares only the first half of an existing node, the system splits the old node in two at the divergence—the common upper half stays shared by both, the differing lower halves become two children. This "split on demand" keeps the radix tree in a compact "share what can be shared, fork exactly where it must" shape, rather than degenerating into a heap of unrelated full strings. Understand pinning and splitting and you understand <strong>why this tree is both safe and efficient</strong>.</p>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Match reuse compute-only-suffix: a request's tokens split into shared prefix and new suffix; match_prefix reuses the prefix KV for zero compute and the model forwards only the new suffix">
    <text x="40" y="36" style="font-weight:700;fill:var(--muted)">request tokens = [ shared prefix | new suffix ]</text>
    <rect x="40" y="62" width="420" height="48" rx="8" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="250" y="84" text-anchor="middle" style="fill:var(--muted);font-weight:700">shared prefix</text>
    <text x="250" y="102" text-anchor="middle" style="fill:var(--faint);font-size:12px">cached KV · greyed</text>
    <rect x="468" y="62" width="272" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:2"/>
    <text x="604" y="84" text-anchor="middle" style="fill:var(--amber);font-weight:700">new suffix</text>
    <text x="604" y="102" text-anchor="middle" style="fill:var(--amber);font-size:12px">the only computed part</text>
    <line x1="40" y1="124" x2="460" y2="124" style="stroke:var(--teal);stroke-width:1.5"/>
    <text x="40" y="152" style="fill:var(--teal);font-weight:700">match_prefix hits</text>
    <text x="40" y="172" style="font-size:12px">reuse its KV · 0 compute</text>
    <line x1="468" y1="124" x2="740" y2="124" style="stroke:var(--amber);stroke-width:1.5"/>
    <text x="468" y="152" style="fill:var(--amber);font-weight:700">forward only the suffix</text>
    <text x="468" y="172" style="font-size:12px">cost = suffix length</text>
    <rect x="40" y="208" width="360" height="56" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="56" y="232" style="fill:var(--teal);font-weight:700">radix tree: prefix resident</text>
    <text x="56" y="252" style="font-size:12px">walk to the deepest known node, reuse</text>
    <rect x="468" y="208" width="272" height="56" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="484" y="232" style="fill:var(--accent-ink);font-weight:700">saved = the prefix span</text>
    <text x="484" y="252" style="font-size:12px">not the whole sequence</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Match → reuse → compute only the new suffix</b> — a request's tokens split into "shared prefix (greyed, cached) | new suffix (highlighted)"; <span class="mono">match_prefix</span> finds the prefix in the tree and reuses its KV as-is (0 compute), so the model forwards only the <strong>new suffix</strong>. Cost equals the suffix length, not the whole sequence.</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.insert</span><span class="ln">insert a span of tokens+KV so later requests share this prefix</span></div><pre>def insert(self, params: InsertParams) -&gt; InsertResult:
    # add a sequence's tokens + KV into the radix tree so later
    # requests can SHARE this prefix. The key is page-aligned first.
    key = params.key.page_aligned(self.page_size)
    value = params.value[:len(key)]
    prefix_len, last_node = self._insert_helper(
        self.root_node, key, value, ...)   # split nodes at divergence
    return InsertResult(prefix_len=prefix_len)   # how much already existed</pre></div>

<div class="card detail"><div class="tag">🧪 Concrete example</div>
<p>Picture a <strong>1000-token system prompt</strong> shared by every request. The first request <span class="mono">insert</span>s it into the tree once, and its KV lands on a handful of pages. The next request arrives with the same opening, and <span class="mono">match_prefix</span> hits all 1000 tokens at once—<strong>0 recompute</strong>—so the engine forwards only its own question. The third and fourth do the same: the more that shared prefix is reused, the more compute is saved.</p>
<p>And this <strong>same radix idea</strong> powers four places at once: the DSL's <span class="mono">fork</span> makes branches share an opening, the cache stores prefixes as a tree, cache-aware LPM scheduling clusters high-hit requests, and HiCache tiers that tree across GPU/CPU/disk by temperature. One <span class="mono">insert</span> feeds the whole stack.</p>
</div>

<table class="t">
<tr><th>Where shared prefixes appear</th><th>The role it plays there</th></tr>
<tr><td>DSL fork/join (<strong>Lesson 11</strong>)</td><td>The language layer <strong>deliberately</strong> makes branches share one opening—the source of the "shared prefix"</td></tr>
<tr><td>Prefix-cache concept (<strong>Lesson 7</strong>)</td><td>An early promise: identical openings are computed once, setting the tone for the implementation</td></tr>
<tr><td>Radix-tree implementation (<strong>Lesson 29</strong>)</td><td>The promise lands: <span class="mono">RadixCache</span> made of <span class="mono">TreeNode</span>s—the cache IS this tree</td></tr>
<tr><td>Paged KV (<strong>Lesson 30</strong>, on <strong>Lesson 6</strong>)</td><td>The physical pages a node's <span class="mono">value</span> points at—where prefixes actually live</td></tr>
<tr><td>Cache-aware scheduling (<strong>Lesson 20</strong>)</td><td>Lets high prefix-hit requests through first so the stored KV truly pays off</td></tr>
<tr><td>HiCache tiering (<strong>Lesson 31</strong>)</td><td>Tiers that same prefix tree by temperature across GPU/CPU/disk</td></tr>
</table>

<div class="layers">
<div class="layer">root: the common starting point of all requests (empty prefix)</div>
<div class="layer">shared-prefix node: the token span of <span class="mono">"You are a helpful assistant…"</span> + KV indices, shared by many requests</div>
<div class="layer">divergence point: tokens differ from here, so the node is split into multiple children</div>
<div class="layer">branch A leaf: the unique suffix of user question one</div>
<div class="layer">branch B leaf: the unique suffix of user question two (sharing the whole prefix above with A)</div>
</div>

<div class="cols">
<div class="col"><strong>Bolt-on optimization</strong><br/>The cache is an LRU dictionary beside the engine; scheduling knows nothing of it; hits are luck; tiering can only copy scattered blocks; changing the cache ripples nowhere.</div>
<div class="col"><strong>First-class data structure</strong><br/>The cache IS a radix tree; scheduling estimates hits and actively clusters; memory pins/reclaims by sharing relationships; tiering moves whole segments along tree boundaries; one idea unifies the whole system.</div>
</div>

<div class="flow">
<div class="node">request arrives with tokens</div>
<div class="arrow">→</div>
<div class="node">match_prefix finds the longest cached prefix</div>
<div class="arrow">→</div>
<div class="node">reuse that prefix's KV (inc_lock_ref pins it)</div>
<div class="arrow">→</div>
<div class="node">forward-compute only the new suffix</div>
<div class="arrow">→</div>
<div class="node">insert writes back to the tree, splitting at divergence</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache</span><span class="ln">the KV cache IS a radix tree of shared prefixes</span></div><pre>class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)  # edge -&gt; child: the radix tree itself
        self.key = None                        # the token-id span on the edge into this node
        self.value = None                      # the KV cache indices for that span

class RadixCache(BasePrefixCache):
    # the prefix cache IS a radix tree of shared-prefix nodes
    def match_prefix(self, params):   # walk the tree; return the longest already-cached prefix
        ...
    def insert(self, params):         # add a sequence's KV, splitting nodes where prefixes diverge
        ...
    def evict(self, params):          # LRU-evict evictable leaves to reclaim pages
        ...
    def inc_lock_ref(self, node):     # pin in-use nodes so eviction can't take them
        ...</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>First-class, not bolt-on:</strong> SGLang treats "the KV cache is a <span class="mono">radix tree of shared prefixes</span>" as the core data structure the whole engine revolves around, not an LRU cache glued on afterward.</li>
<li><strong>Six faces of one idea:</strong> DSL fork/join (<strong>Lesson 11</strong>), the prefix-cache concept (<strong>Lesson 7</strong>), the radix-tree implementation (<strong>Lesson 29</strong>), paged KV (<strong>Lesson 30</strong>, on <strong>Lesson 6</strong>), cache-aware scheduling (<strong>Lesson 20</strong>), and HiCache tiering (<strong>Lesson 31</strong>) are all the same thing.</li>
<li><strong>The lifecycle:</strong> match the longest prefix → reuse KV → compute only the new suffix → write back to the tree is the entire essence of <span class="mono">RadixAttention</span>.</li>
<li><strong>Structure forces the algorithms:</strong> once the cache is a tree, <span class="mono">match_prefix/insert/evict/inc_lock_ref</span> are natural consequences of the tree shape.</li>
<li><strong>Grasp it and a third of the guide connects:</strong> recognizing this one organizing idea suddenly weaves many seemingly independent lessons into a web.</li>
</ul></div>
"""}
LESSON_59 = {"zh": r"""
<p class="lead">如果把前面许多课串成一条线，你会发现它们其实都在回答同一个问题：<strong>如何让 GPU 永远不等 CPU</strong>。这就是 SGLang 调度子系统的设计北极星——<span class="mono">零开销调度（zero-overhead scheduling）</span>。现代 GPU 的前向计算快到离谱，任何在前向<strong>之前</strong>或<strong>之后</strong>由 CPU 做的工作（调度、分配、分词、解码、拼下一批），都会变成 GPU 干等的"气泡（bubble）"。本课不教新机制，而是把第18、21、22、27、14、16 课像珠子一样串起来，让你看清那条贯穿始终的主线。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一条高级餐厅的<strong>开放式厨房</strong>。主厨（GPU）刀工快得吓人，一道菜十秒钟出锅。可是如果每次出菜后都要主厨自己去洗菜、切配、摆盘、写下一单（这些都是 CPU 的活），那主厨大部分时间都在<strong>空等</strong>，灶台凉着。聪明的做法是：在主厨炒<strong>第 N 道菜</strong>的同时，旁边的小工已经把<strong>第 N+1 道菜</strong>的料备好了，同时把<strong>第 N-1 道菜</strong>端走摆盘。这样主厨的灶台<strong>一刻不停</strong>。SGLang 的 <span class="mono">overlap scheduler</span> 干的就是这件事：把所有 CPU 的"备菜、摆盘"藏到 GPU"炒菜"的时间背后。</p>
<p>这个类比还能再推一步。如果某位客人点了一道需要慢炖两小时的硬菜（超长 prefill），聪明的厨房不会让主厨守着这一锅、把别人的菜全晾着，而是把慢炖<strong>分几次照看</strong>、其间照常出别的快手菜——这正是分块预填充。如果摆盘、写单这些杂事太多，干脆<strong>另雇一个传菜员</strong>在别的工位上专门干，主厨连看都不用看——这正是进程拆分。你会发现，一家高效厨房的全部智慧，都浓缩成一句话：<strong>千万别让主厨闲下来</strong>。</p></div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>"零开销"不是说 CPU 不干活，而是说 CPU 的活<strong>不占用额外的墙钟时间</strong>——它们被<strong>重叠（overlap）</strong>进了 GPU 的前向计算里。一个推理引擎的每一步，逻辑上是：CPU 调度选批 → GPU 前向 → CPU 处理结果（采样、拼 token、检测停止）。如果三者串行，GPU 在前后两段 CPU 时间里都<strong>闲着</strong>。SGLang 的整套调度设计，从<strong>事件循环（第18课）</strong>、<strong>重叠调度（第21课）</strong>、<strong>分块预填充（第22课）</strong>、<strong>CUDA 图（第27课）</strong>，到<strong>进程拆分 + IPC（第14/16课）</strong>，本质上都是同一句话的不同侧面：<span class="mono">找到每一个 CPU 气泡，把它藏到 GPU 的计算背后</span>。</p>
<p>之所以值得单独用一课来讲这个"哲学"，是因为它能<strong>反过来指导你读懂整个系统</strong>。很多人学 SGLang 时把这些机制当成一张<strong>互不相关的特性清单</strong>逐条死记，结果调参时只能照抄别人的命令行。但只要你抓住"别让 GPU 等"这一条主线，就能<strong>自上而下</strong>地理解每个机制存在的理由：它不是为了炫技，而是在堵某一个具体的气泡。本课的目标，就是把这条贯穿前面许多课的隐线<strong>显式地拎出来</strong>，让你从"记住很多招"升级到"看懂一套心法"。</p></div>

<h2>一、气泡从哪来：GPU 太快，CPU 太"碍事"</h2>
<p>先把问题量化。一次 decode 前向，在大模型上可能只需要几毫秒甚至更短。可是围绕它的 CPU 工作并不少：要从等待队列里<strong>挑出能跑的请求</strong>、给它们<strong>分配 KV cache 的页</strong>、把张量<strong>搬上设备</strong>；前向结束后，又要<strong>采样</strong>下一个 token、把 token <strong>追加</strong>到各个序列、<strong>判断是否触发停止条件</strong>、把完成的结果<strong>发回</strong>给分词进程。这些 CPU 操作每一项都是微秒到毫秒级的，单看不起眼，但只要它们<strong>串在 GPU 前后</strong>，GPU 就得乖乖等着——这段空等，就是气泡。</p>
<p>气泡之所以可怕，是因为它<strong>不产生任何价值却消耗墙钟时间</strong>。GPU 是整台机器里最贵的资源，买它就是为了让它满载算矩阵乘法。可一旦它在每一步前后都要停下来等 CPU 把下一批准备好，它的<strong>实际利用率</strong>就被腰斩。你以为买了一块能跑满的卡，结果一半时间它在打盹。更糟的是，这种浪费<strong>随步数累积</strong>：生成一个一千 token 的回答要跑一千步，每步漏掉的那点时间乘以一千，就是肉眼可见的延迟和吞吐损失。换个角度看，吞吐量几乎正比于 GPU 的有效占用率，每堵住一个气泡，就等于<strong>白捡</strong>了一截算力，而且这截算力是<strong>已经付过钱</strong>的——它本来就在那张卡里，只是之前被白白浪费掉了。</p>
<p>关键洞察是：<strong>GPU 越快，气泡的相对占比越大</strong>。十年前 GPU 慢，CPU 那点开销淹没在计算里看不出来；今天 GPU 一步只要几毫秒，CPU 的几毫秒就成了<strong>对半开</strong>的浪费。换句话说，硬件越先进，软件不把 CPU 开销藏好，浪费就越严重——这是一个<strong>反直觉</strong>的结论：升级显卡如果不配套优化调度，多花的钱可能有一半打了水漂。于是"消灭气泡"从一个锦上添花的优化，升级成了<strong>决定吞吐的头等大事</strong>。这正是 SGLang 把调度器命名为"零开销"的原因——它的目标函数就是让 GPU 占用率逼近 100%。</p>

<h2>二、主线：事件循环与重叠调度</h2>
<p>一切的心脏是<strong>调度器的事件循环（第18课）</strong>。它是一个永不停歇的 <span class="mono">while True</span>：收请求、组批、发前向、收结果、再组下一批。整个推理引擎就靠这个循环一圈一圈地把系统"泵"起来，每转一圈就吐出一批 token。朴素的事件循环是<strong>串行</strong>的——组完批才发前向，前向回来才处理结果，处理完才组下一批。这就是气泡的温床：在"发前向"和"收结果"这两个边界上，GPU 和 CPU 必然有一方在等另一方。</p>
<p><strong>重叠调度（第21课）</strong>把这个串行循环改造成<strong>流水线</strong>：当 GPU 正在跑 <span class="mono">forward(批 N)</span> 时，CPU<strong>不闲着</strong>——它已经在<strong>组装批 N+1</strong>（挑请求、分配页、搬张量），并且<strong>并行处理批 N-1 的结果</strong>（采样、拼 token、检测停止）。注意这里有一个微妙但重要的点：CPU 处理的是<strong>上一步</strong>的结果，而不是刚发出去那一步的——因为刚发出的前向还没算完。正是这种"<strong>错一拍</strong>"的安排，让 CPU 永远有活干、GPU 永远不用等。</p>
<p>这一切由 <span class="mono">enable_overlap</span> 开关控制（即 <span class="mono">--disable-overlap-schedule</span> 的反面）。一旦打开，CPU 的调度与结果处理就被"塞进"了 GPU 前向的影子里，墙钟时间上几乎<strong>不可见</strong>。代价是引擎要同时持有<strong>多步的中间状态</strong>（N-1、N、N+1），实现复杂度更高，也更难调试——但这正是"零开销"四个字真正落地、值得付出的地方。如果你把 overlap 关掉，引擎会退回到那个朴素串行循环，正确性不变，但 GPU 利用率和吞吐都会明显下滑。这也解释了为什么默认就是打开的：对绝大多数线上负载来说，多花一点内存换回近乎翻倍的有效算力，是再划算不过的交易。</p>

<p><strong>一个具体的数字感</strong>：当批很小（比如只有几条 decode 请求）时，GPU 的一次前向可能只要 2~3 毫秒，而 CPU 这一拍要做的挑批、分配、采样加起来也可能逼近 2 毫秒——串行执行时这 2 毫秒几乎让每步耗时<strong>翻倍</strong>。打开重叠后，这段 CPU 工作被整体藏进 GPU 前向的影子里，墙钟时间几乎只由 GPU 决定，吞吐因此回到"<strong>只受 GPU 限制</strong>"的理想值——这正是"零开销"调度名副其实之处。批越大，GPU 前向越长，CPU 那点开销就越容易被完全吃掉，但小批场景恰恰是重叠收益最明显的地方。</p>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="串行时间线 GPU 在 CPU 调度与处理期间出现气泡，重叠时间线 CPU 与 GPU 前向并行运行气泡消失">
    <text x="20" y="28" style="font-weight:700;fill:var(--muted)">串行：GPU 有气泡</text>
    <text x="20" y="60" style="fill:var(--faint);font-size:12px">GPU</text>
    <rect x="64" y="40" width="110" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="119" y="61" text-anchor="middle" style="font-size:12px">前向 N</text>
    <rect x="174" y="40" width="92" height="32" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="220" y="61" text-anchor="middle" style="fill:var(--red);font-size:12px">空等</text>
    <rect x="266" y="40" width="110" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="321" y="61" text-anchor="middle" style="font-size:12px">前向 N+1</text>
    <rect x="376" y="40" width="92" height="32" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="422" y="61" text-anchor="middle" style="fill:var(--red);font-size:12px">空等</text>
    <rect x="468" y="40" width="110" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="523" y="61" text-anchor="middle" style="font-size:12px">前向 N+2</text>
    <text x="20" y="104" style="fill:var(--faint);font-size:12px">CPU</text>
    <rect x="174" y="86" width="92" height="28" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="220" y="105" text-anchor="middle" class="mono" style="font-size:11px">调度+处理</text>
    <rect x="376" y="86" width="92" height="28" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="422" y="105" text-anchor="middle" class="mono" style="font-size:11px">调度+处理</text>
    <line x1="20" y1="150" x2="780" y2="150" style="stroke:var(--line);stroke-width:1;stroke-dasharray:5 5"/>
    <text x="20" y="184" style="font-weight:700;fill:var(--accent-ink)">重叠：气泡消失</text>
    <text x="20" y="216" style="fill:var(--faint);font-size:12px">GPU</text>
    <rect x="64" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="124" y="217" text-anchor="middle" style="font-size:12px">前向 N</text>
    <rect x="184" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="244" y="217" text-anchor="middle" style="font-size:12px">前向 N+1</text>
    <rect x="304" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="364" y="217" text-anchor="middle" style="font-size:12px">前向 N+2</text>
    <rect x="424" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="484" y="217" text-anchor="middle" style="font-size:12px">前向 N+3</text>
    <text x="20" y="260" style="fill:var(--faint);font-size:12px">CPU</text>
    <rect x="64" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="124" y="261" text-anchor="middle" class="mono" style="font-size:11px">调度/处理</text>
    <rect x="184" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="244" y="261" text-anchor="middle" class="mono" style="font-size:11px">调度/处理</text>
    <rect x="304" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="364" y="261" text-anchor="middle" class="mono" style="font-size:11px">调度/处理</text>
    <rect x="424" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="484" y="261" text-anchor="middle" class="mono" style="font-size:11px">调度/处理</text>
    <text x="560" y="231" style="fill:var(--teal);font-weight:700;font-size:13px">CPU‖GPU 并行</text>
  </svg>
  <div class="figcap"><b>图 59-1 · CPU 气泡藏到前向之后</b> — 串行时 GPU 在 CPU 调度与处理期间空等（气泡）；重叠时 CPU 与 GPU 前向并行，GPU 背靠背连跑，气泡消失。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="三级流水线 构建 前向 处理 三条泳道沿时间步错开一拍 稳定态同一拍三段并行">
    <text x="20" y="28" style="font-weight:700;fill:var(--muted)">重叠流水：三段错一拍</text>
    <rect x="498" y="64" width="134" height="158" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="115" y="52" text-anchor="middle" style="fill:var(--faint);font-size:12px">t1</text>
    <text x="265" y="52" text-anchor="middle" style="fill:var(--faint);font-size:12px">t2</text>
    <text x="415" y="52" text-anchor="middle" style="fill:var(--faint);font-size:12px">t3</text>
    <text x="565" y="52" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">t4</text>
    <text x="20" y="92" style="fill:var(--muted);font-size:12px">构建</text>
    <text x="20" y="152" style="fill:var(--muted);font-size:12px">前向</text>
    <text x="20" y="212" style="fill:var(--muted);font-size:12px">处理</text>
    <rect x="55" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="115" y="93" text-anchor="middle" class="mono" style="font-size:11px">构建 B1</text>
    <rect x="205" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="265" y="93" text-anchor="middle" class="mono" style="font-size:11px">构建 B2</text>
    <rect x="355" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="415" y="93" text-anchor="middle" class="mono" style="font-size:11px">构建 B3</text>
    <rect x="505" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="565" y="93" text-anchor="middle" class="mono" style="font-size:11px">构建 B4</text>
    <rect x="205" y="132" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="265" y="153" text-anchor="middle" class="mono" style="font-size:11px">前向 B1</text>
    <rect x="355" y="132" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="415" y="153" text-anchor="middle" class="mono" style="font-size:11px">前向 B2</text>
    <rect x="505" y="132" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="565" y="153" text-anchor="middle" class="mono" style="font-size:11px">前向 B3</text>
    <rect x="355" y="192" width="120" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="415" y="213" text-anchor="middle" class="mono" style="font-size:11px">处理 B1</text>
    <rect x="505" y="192" width="120" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="565" y="213" text-anchor="middle" class="mono" style="font-size:11px">处理 B2</text>
    <text x="20" y="258" style="fill:var(--accent-ink);font-weight:700;font-size:13px">同一拍：构建 N+1 ‖ 前向 N ‖ 处理 N-1</text>
  </svg>
  <div class="figcap"><b>图 59-2 · 重叠流水线</b> — 构建、前向、处理三条泳道各错开一拍；进入稳定态后（t4），同一拍里 CPU 构建批 N+1、GPU 跑前向 N、CPU 处理批 N-1，三段始终并行。</div>
</div>

<h2>三、其他珠子：分块预填充、CUDA 图、进程拆分</h2>
<p>同一条主线还串着另外三颗珠子。<strong>分块预填充（第22课）</strong>解决的是另一种气泡：一个超长 prompt 的 prefill 如果一口气跑完，会占满整个前向，让所有正在 decode 的请求<strong>全体卡住</strong>等它，造成延迟尖刺。SGLang 把长 prefill <strong>切成小块</strong>，与 decode <strong>交替</strong>进行——每一步既塞进一点新 prompt 的 prefill，又照顾正在生成的老请求，避免单个大活堵死整条流水线。本质还是那句话："别让某一段把 GPU 的节奏打乱"，把一个会引发长气泡的大任务摊薄成许多不打扰别人的小任务。</p>
<p><strong>CUDA 图（第27课）</strong>对付的是<strong>内核启动开销</strong>这种气泡：每发射一个 CUDA kernel，CPU 都要付出固定的启动成本，decode 一步要发几十上百个 kernel，加起来就是可观的 CPU 时间，而且这段时间 GPU 往往在等下一个 kernel 被提交。CUDA 图把整串 kernel<strong>录制成一张图</strong>，之后一次<strong>重放</strong>就能在 GPU 上跑完整条序列，CPU 几乎不再逐个参与发射——这是"别让 CPU 成为 GPU 瓶颈"的又一种形态，专门针对 decode 这种<strong>kernel 多、单个小</strong>、最容易被启动开销拖累的场景。</p>
<p>最后是<strong>进程拆分 + IPC（第14/16课）</strong>。SGLang 把 <span class="mono">TokenizerManager</span> 和 <span class="mono">DetokenizerManager</span> 放在<strong>独立于</strong>调度器 + GPU 的<strong>另外的进程</strong>里，靠 IPC（进程间通信）传消息。这样分词、解码这些纯 CPU 活，根本<strong>不在</strong>前向的关键路径上——它们在别的进程里和 GPU 计算天然并行，连"重叠"都不用刻意安排，操作系统的进程调度就帮你把它们错开了。这也顺带绕开了 Python 的全局解释器锁（GIL）：分词进程的 CPU 占用不会拖慢调度进程。可以说，进程拆分是把气泡<strong>物理隔离</strong>到了关键路径之外。</p>

<h2>四、把所有珠子串成一句话</h2>
<p>现在退后一步看全局。第18、21、22、27、14、16 课，表面上是六个不同的机制，骨子里是<strong>同一个设计哲学</strong>的六种实现：<span class="mono">枚举每一个可能让 GPU 干等的 CPU 气泡，然后把它藏到 GPU 计算背后，或者干脆挪到别的进程</span>。事件循环给出了"泵"的骨架，重叠调度把调度和结果处理藏进影子，分块预填充把长任务摊薄，CUDA 图消掉启动开销，进程拆分把分词解码搬走——每一颗珠子瞄准的都是一种特定形状的气泡。</p>
<p>理解了这条主线，你再回头看任何一个调度相关的开关（<span class="mono">--disable-overlap-schedule</span>、<span class="mono">--chunked-prefill-size</span>、<span class="mono">--disable-cuda-graph</span>），都会立刻明白它在和哪一种气泡作战，也能预判关掉它会让哪种气泡重新冒出来。这就是融会贯通的力量：你记住的不再是六个孤立的参数，而是<strong>一条能自己推导出新优化的原则</strong>。下次遇到一个新的 CPU 开销，你会本能地问："它能不能藏到 GPU 计算背后？能不能挪到别的进程？"——这，就是零开销调度作为整个调度子系统设计北极星的真正含义。</p>

<h2>五、一个具体的回合：把原则走一遍</h2>
<p>抽象讲了这么多，我们把它落到一次真实的生成回合上，看看零开销原则如何在每一拍里发力。假设此刻引擎正在为几十条并发请求做 decode，同时还有一条携带超长 prompt 的新请求刚刚到达。如果没有任何优化，这一拍会是这样：CPU 先停下来给新请求做完整的 prefill（很慢），所有老请求干等；prefill 算完才轮到 decode；decode 的前向发出去后，CPU 又得逐个发射上百个 kernel；前向回来后再串行地采样、拼 token、检测停止。GPU 在这一拍里至少有<strong>三段</strong>明显的空闲。</p>
<p>现在打开 SGLang 的全套机制。<strong>分块预填充</strong>先把那条长 prompt 切成小块，本拍只塞进一块，和几十条老请求的 decode <strong>拼成同一个批</strong>一起算，没人被饿死。<strong>CUDA 图</strong>让这个批的上百个 kernel 一次重放完成，CPU 不再逐个发射。<strong>重叠调度</strong>则保证：当这一拍的前向在 GPU 上跑时，CPU 已经在<strong>挑下一拍的请求、分配页</strong>，同时把<strong>上一拍</strong>采样出来的 token <strong>拼回</strong>各序列、检测哪些请求该停了。而那些该停的请求要回传给用户，这一步走的是<strong>进程拆分</strong>的通道——<span class="mono">DetokenizerManager</span> 在另一个进程里把 token id 还原成文字，完全不占用当前前向的关键路径。</p>
<p>一拍下来，原本三段空闲被填得满满当当：长 prefill 被摊薄、kernel 启动被消除、调度与结果处理被藏进影子、分词解码被物理隔离到别的进程。GPU 几乎从头忙到尾。把这一拍重复一千次，就是一次完整的生成；把零开销原则贯彻到每一拍，就是 SGLang 高吞吐的底层秘密。<strong>你会发现，所谓"零开销"，并不是某一个聪明的算法，而是一种近乎偏执的习惯：见到任何一段 GPU 可能干等的时间，就想方设法把别的有用工作填进去。</strong></p>

<div class="flow"><div class="node">CPU 选批/分配</div><div class="arrow">→</div><div class="node">GPU 前向(N)</div><div class="arrow">→</div><div class="node">CPU 采样/拼token</div><div class="arrow">→</div><div class="node">GPU 闲等(气泡)</div><div class="arrow">→</div><div class="node">CPU 选批(N+1)</div></div>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>GPU 正在跑 forward(批 N)</h4><p class="mono">run_batch(batch_N)</p><p>这一步是整个流水线里最贵、最耗时的，GPU 满负荷运转。</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>同一时刻 · CPU 组装批 N+1</h4><p class="mono">get_next_batch_to_run()</p><p>挑请求、分配 KV 页、搬张量，全都在 GPU 忙的影子里悄悄做完。</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>同一时刻 · CPU 处理批 N-1 的结果</h4><p class="mono">process_batch_result(prev)</p><p>采样、拼 token、检测停止、回发结果，与 GPU 计算并行。</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>→ GPU 永不空等</h4><p class="mono">enable_overlap=True</p><p>灶台一刻不停，CPU 开销在墙钟上彻底隐身。</p></div></div></div>

<table class="t"><tr><th>CPU 气泡</th><th>SGLang 如何把它藏起来</th><th>对应课</th></tr>
<tr><td>调度选批 / 结果处理</td><td>事件循环 + 重叠调度，CPU 活塞进 GPU 前向影子里</td><td>第18 / 21 课</td></tr>
<tr><td>超长 prefill 堵塞流水线</td><td>分块预填充，把长 prefill 切块与 decode 交替</td><td>第22 课</td></tr>
<tr><td>每步内核启动开销</td><td>CUDA 图录制一次、重放多次，CPU 不再逐个发射</td><td>第27 课</td></tr>
<tr><td>分词 / 解码占用关键路径</td><td>进程拆分 + IPC，Tokenizer/Detokenizer 跑在别的进程</td><td>第14 / 16 课</td></tr></table>

<div class="cols"><div class="col"><strong>overlap 关闭</strong><br>组批 → 前向 → 处理结果 串行执行，GPU 在两段 CPU 时间里都<span class="mono">空闲</span>，占用率掉到一半甚至更低，气泡清晰可见。</div><div class="col"><strong>overlap 打开</strong><br>GPU 跑 forward(N) 的同时，CPU 已备好 N+1、处理完 N-1，GPU<span class="mono">始终忙碌</span>，墙钟上看不到 CPU 开销——这就是零开销。</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler</span><span class="ln">GPU 跑第 N 步时，CPU 已在准备第 N+1 步</span></div><pre>class Scheduler:
    def __init__(self, server_args, ...):
        # CPU bubbles are the enemy: hide them behind the GPU forward
        self.enable_overlap = not server_args.disable_overlap_schedule
        ...
    def event_loop_overlap(self):
        # while the GPU runs forward(batch N), the CPU already builds batch N+1,
        # and the PREVIOUS step's result is processed in parallel -&gt; GPU never idles
        while True:
            batch = self.get_next_batch_to_run()    # CPU: schedule + allocate
            result = self.run_batch(batch)          # launch GPU forward (async)
            self.process_batch_result(prev_result)  # overlap: handle the prior step
            prev_result = result</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.event_loop_overlap</span><span class="ln">重叠循环：处理上一批结果的同时，GPU 跑当前批</span></div><pre>def event_loop_overlap(self):
    # overlap CPU work with the GPU forward: finish the PREVIOUS batch's
    # results while the current batch is still running on the GPU.
    self.result_queue = deque()
    while True:
        recv = self.request_receiver.recv_requests()
        self.process_input_requests(recv)
        batch = self.get_next_batch_to_run()        # build batch N (CPU)
        result = self.run_batch(batch)              # launch forward N (GPU)
        self.result_queue.append((batch, result))
        # one step out of phase: process batch N-1 now
        self.process_batch_result(*self.result_queue.popleft())</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>一条主线</strong>：CPU 绝不能让 GPU 等。任何前向<strong>前后</strong>的 CPU 工作都是潜在气泡。</li>
<li><strong>重叠是核心手段</strong>：GPU 跑 forward(N) 时，CPU 已组装 N+1、并行处理 N-1（<span class="mono">enable_overlap</span>，第18/21课）。</li>
<li><strong>多种气泡多种藏法</strong>：分块预填充防长 prefill 堵塞（第22课）、CUDA 图消内核启动开销（第27课）、进程拆分让分词解码离开关键路径（第14/16课）。</li>
<li><strong>GPU 越快，气泡越致命</strong>：消灭气泡从锦上添花升级为决定吞吐的头等大事，硬件越先进越要把 CPU 开销藏好。</li>
<li><strong>融会贯通</strong>：六课六机制，同一个哲学——找到每个 CPU 气泡，把它藏到 GPU 计算背后。记住这一条原则，胜过死记六个孤立的参数，因为它能让你<strong>自己推导</strong>出下一个优化。</li>
</ul></div>
""", "en": r"""
<p class="lead">If you thread together many of the earlier lessons, you find they all answer one question: <strong>how do we make sure the GPU never waits for the CPU?</strong> This is the design north star of SGLang's scheduling subsystem—<span class="mono">zero-overhead scheduling</span>. A modern GPU's forward pass is absurdly fast, so any work the CPU does <strong>before</strong> or <strong>after</strong> the forward (scheduling, allocating, tokenizing, detokenizing, building the next batch) becomes a "bubble" where the GPU sits idle. This lesson teaches no new mechanism; instead it strings lessons 14, 16, 18, 21, 22, and 27 together like beads so you can see the single thread running through them all.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture the <strong>open kitchen</strong> of a high-end restaurant. The head chef (the GPU) has terrifying knife skills—a dish leaves the pan in ten seconds. But if after every dish the chef has to wash, chop, plate, and write the next ticket themselves (all CPU work), the chef spends most of the time <strong>idle</strong>, the stove going cold. The smart move: while the chef cooks <strong>dish N</strong>, a runner has already prepped <strong>dish N+1</strong> and is plating and carrying away <strong>dish N-1</strong>. The chef's stove <strong>never stops</strong>. SGLang's <span class="mono">overlap scheduler</span> does exactly this: it hides all the CPU "prep and plating" behind the GPU's "cooking" time.</p>
<p>The analogy stretches further. If a guest orders a dish that needs two hours of slow braising (a very long prefill), a clever kitchen won't let the chef babysit that one pot while everyone else's food goes cold; it <strong>tends the braise in installments</strong>, cooking quick dishes in between—that is chunked prefill. If plating and ticket-writing pile up, just <strong>hire a separate runner</strong> at another station to do nothing else, so the chef never even glances at it—that is the process split. You'll find the whole wisdom of an efficient kitchen distills into one line: <strong>never let the chef sit idle</strong>.</p></div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>"Zero-overhead" doesn't mean the CPU does no work—it means the CPU's work costs <strong>no extra wall-clock time</strong>, because it is <strong>overlapped</strong> into the GPU's forward pass. Each step of an inference engine is logically: CPU schedule/pick batch → GPU forward → CPU process results (sample, append token, check stop). Run serially, the GPU sits <strong>idle</strong> during both CPU segments. SGLang's whole scheduling design—from the <strong>event loop (L18)</strong>, the <strong>overlap scheduler (L21)</strong>, <strong>chunked prefill (L22)</strong>, <strong>CUDA graphs (L27)</strong>, to the <strong>process split + IPC (L14/16)</strong>—is the same sentence from different angles: <span class="mono">find every CPU bubble and hide it behind GPU compute</span>.</p>
<p>This "philosophy" deserves its own lesson because it can <strong>turn around and help you read the whole system</strong>. Many people learning SGLang treat these mechanisms as an <strong>unrelated feature checklist</strong> to memorize one by one, and end up only able to copy someone else's command line when tuning. But once you seize the single thread "don't let the GPU wait," you can understand <strong>top-down</strong> why each mechanism exists: not to show off, but to plug one specific bubble. The goal of this lesson is to <strong>pull that hidden thread out into the open</strong>, upgrading you from "memorizing many moves" to "grasping one underlying art."</p></div>

<h2>1. Where bubbles come from: the GPU is fast, the CPU is "in the way"</h2>
<p>Quantify the problem first. A single decode forward on a large model may take only a few milliseconds, or less. Yet the CPU work around it is not small: it must <strong>pick runnable requests</strong> from the waiting queue, <strong>allocate KV-cache pages</strong> for them, and <strong>move tensors</strong> onto the device; after the forward it must <strong>sample</strong> the next token, <strong>append</strong> it to each sequence, <strong>check stop conditions</strong>, and <strong>send</strong> finished results back to the tokenizer process. Each of these is microseconds to milliseconds—trivial alone, but the moment they are <strong>strung before and after</strong> the GPU, the GPU must wait. That idle gap is the bubble.</p>
<p>A bubble is dangerous because it <strong>produces no value yet consumes wall-clock time</strong>. The GPU is the most expensive resource in the whole machine; you bought it to keep it saturated with matrix multiplies. The instant it must stop before and after every step to wait for the CPU to prepare the next batch, its <strong>real utilization</strong> is halved. You thought you bought a card that runs flat-out, but half the time it dozes. Worse, this waste <strong>accumulates across steps</strong>: generating a thousand-token answer takes a thousand steps, and the bit of time lost each step times a thousand is visible latency and throughput loss. Seen another way, throughput is roughly proportional to the GPU's effective occupancy, and every bubble you plug is a slice of compute you get <strong>for free</strong>—compute you have <strong>already paid for</strong>, that was sitting in the card all along and merely wasted before.</p>
<p>The key insight: <strong>the faster the GPU, the larger the bubble's relative share</strong>. A decade ago GPUs were slow and that CPU overhead drowned in the compute; today a GPU step takes a few milliseconds, so a few CPU milliseconds become a <strong>fifty-fifty</strong> waste. In other words, the more advanced the hardware, the worse the waste if the software fails to hide CPU overhead—a <strong>counter-intuitive</strong> conclusion: upgrading the GPU without matching scheduler optimization may pour half the extra money down the drain. So "killing bubbles" graduates from a nice-to-have into the <strong>number-one factor deciding throughput</strong>. That is exactly why SGLang names its scheduler "zero-overhead"—its objective is to push GPU occupancy toward 100%.</p>

<h2>2. The main thread: the event loop and overlap scheduling</h2>
<p>At the heart of everything is the <strong>scheduler's event loop (L18)</strong>. It is a never-resting <span class="mono">while True</span>: take requests, form a batch, launch the forward, collect results, form the next batch. The whole inference engine "pumps" the system around this loop, emitting one batch of tokens per turn. A naive event loop is <strong>serial</strong>—form the batch, then launch, then wait for results, then form the next. That is the breeding ground for bubbles: at the two boundaries "launch forward" and "collect results", either the GPU or the CPU must be waiting on the other.</p>
<p>The <strong>overlap scheduler (L21)</strong> turns that serial loop into a <strong>pipeline</strong>: while the GPU runs <span class="mono">forward(batch N)</span>, the CPU <strong>is not idle</strong>—it is already <strong>assembling batch N+1</strong> (pick requests, allocate pages, move tensors) and <strong>processing batch N-1's results in parallel</strong> (sample, append token, check stop). Note a subtle but important point: the CPU processes the results of the <strong>previous</strong> step, not the one just launched—because that forward hasn't finished yet. It is precisely this <strong>off-by-one-beat</strong> arrangement that keeps the CPU always busy and the GPU never waiting.</p>
<p>All of this is controlled by the <span class="mono">enable_overlap</span> switch (the opposite of <span class="mono">--disable-overlap-schedule</span>). Once enabled, the CPU's scheduling and result handling are "tucked into" the shadow of the GPU forward, becoming nearly <strong>invisible</strong> in wall-clock time. The cost is that the engine must hold <strong>several steps' intermediate state</strong> at once (N-1, N, N+1), raising implementation complexity and making debugging harder—but this is where the words "zero-overhead" actually land, and it is worth it. Turn overlap off and the engine falls back to that naive serial loop: correctness unchanged, but GPU utilization and throughput drop noticeably. This also explains why it is on by default: for the vast majority of production loads, trading a little extra memory for nearly double the effective compute is as good a bargain as it gets.</p>

<p><strong>A concrete sense of scale</strong>: when the batch is small (say only a few decode requests), one GPU forward may take just 2~3 ms, while the CPU's pick/allocate/sample work for this beat can approach 2 ms—run serially, that 2 ms nearly <strong>doubles</strong> the per-step time. Turn overlap on and that CPU work is tucked entirely into the GPU forward's shadow, so wall-clock time is set almost solely by the GPU and throughput returns to the ideal "<strong>GPU-bound only</strong>" value—exactly what makes "zero-overhead" scheduling live up to its name. The larger the batch, the longer the GPU forward and the more easily that CPU cost is fully absorbed, but small-batch is precisely where overlap pays off most.</p>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="serial timeline the GPU idles during CPU schedule and processing, overlap timeline the CPU runs concurrently with the GPU forward and the bubbles vanish">
    <text x="20" y="28" style="font-weight:700;fill:var(--muted)">Serial: GPU has bubbles</text>
    <text x="20" y="60" style="fill:var(--faint);font-size:12px">GPU</text>
    <rect x="64" y="40" width="110" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="119" y="61" text-anchor="middle" style="font-size:12px">forward N</text>
    <rect x="174" y="40" width="92" height="32" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="220" y="61" text-anchor="middle" style="fill:var(--red);font-size:12px">idle</text>
    <rect x="266" y="40" width="110" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="321" y="61" text-anchor="middle" style="font-size:12px">forward N+1</text>
    <rect x="376" y="40" width="92" height="32" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="422" y="61" text-anchor="middle" style="fill:var(--red);font-size:12px">idle</text>
    <rect x="468" y="40" width="110" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="523" y="61" text-anchor="middle" style="font-size:12px">forward N+2</text>
    <text x="20" y="104" style="fill:var(--faint);font-size:12px">CPU</text>
    <rect x="174" y="86" width="92" height="28" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="220" y="105" text-anchor="middle" class="mono" style="font-size:11px">sched+proc</text>
    <rect x="376" y="86" width="92" height="28" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="422" y="105" text-anchor="middle" class="mono" style="font-size:11px">sched+proc</text>
    <line x1="20" y1="150" x2="780" y2="150" style="stroke:var(--line);stroke-width:1;stroke-dasharray:5 5"/>
    <text x="20" y="184" style="font-weight:700;fill:var(--accent-ink)">Overlap: bubbles gone</text>
    <text x="20" y="216" style="fill:var(--faint);font-size:12px">GPU</text>
    <rect x="64" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="124" y="217" text-anchor="middle" style="font-size:12px">forward N</text>
    <rect x="184" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="244" y="217" text-anchor="middle" style="font-size:12px">forward N+1</text>
    <rect x="304" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="364" y="217" text-anchor="middle" style="font-size:12px">forward N+2</text>
    <rect x="424" y="196" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="484" y="217" text-anchor="middle" style="font-size:12px">forward N+3</text>
    <text x="20" y="260" style="fill:var(--faint);font-size:12px">CPU</text>
    <rect x="64" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="124" y="261" text-anchor="middle" class="mono" style="font-size:11px">sched/proc</text>
    <rect x="184" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="244" y="261" text-anchor="middle" class="mono" style="font-size:11px">sched/proc</text>
    <rect x="304" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="364" y="261" text-anchor="middle" class="mono" style="font-size:11px">sched/proc</text>
    <rect x="424" y="242" width="120" height="28" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="484" y="261" text-anchor="middle" class="mono" style="font-size:11px">sched/proc</text>
    <text x="560" y="231" style="fill:var(--teal);font-weight:700;font-size:13px">CPU ‖ GPU busy</text>
  </svg>
  <div class="figcap"><b>Fig 59-1 · CPU bubbles hidden behind the forward</b> — serial: the GPU idles during CPU schedule and processing (the bubble); overlap: the CPU runs concurrently with the GPU forward, the GPU runs back-to-back, the bubbles vanish.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="a three-stage pipeline build forward process with three lanes staggered one step apart, in steady state one beat runs all three stages in parallel">
    <text x="20" y="28" style="font-weight:700;fill:var(--muted)">Overlap pipeline: 3 stages, 1 step apart</text>
    <rect x="498" y="64" width="134" height="158" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="115" y="52" text-anchor="middle" style="fill:var(--faint);font-size:12px">t1</text>
    <text x="265" y="52" text-anchor="middle" style="fill:var(--faint);font-size:12px">t2</text>
    <text x="415" y="52" text-anchor="middle" style="fill:var(--faint);font-size:12px">t3</text>
    <text x="565" y="52" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">t4</text>
    <text x="20" y="92" style="fill:var(--muted);font-size:12px">build</text>
    <text x="20" y="152" style="fill:var(--muted);font-size:12px">forward</text>
    <text x="20" y="212" style="fill:var(--muted);font-size:12px">process</text>
    <rect x="55" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="115" y="93" text-anchor="middle" class="mono" style="font-size:11px">build B1</text>
    <rect x="205" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="265" y="93" text-anchor="middle" class="mono" style="font-size:11px">build B2</text>
    <rect x="355" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="415" y="93" text-anchor="middle" class="mono" style="font-size:11px">build B3</text>
    <rect x="505" y="72" width="120" height="32" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="565" y="93" text-anchor="middle" class="mono" style="font-size:11px">build B4</text>
    <rect x="205" y="132" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="265" y="153" text-anchor="middle" class="mono" style="font-size:11px">fwd B1</text>
    <rect x="355" y="132" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="415" y="153" text-anchor="middle" class="mono" style="font-size:11px">fwd B2</text>
    <rect x="505" y="132" width="120" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="565" y="153" text-anchor="middle" class="mono" style="font-size:11px">fwd B3</text>
    <rect x="355" y="192" width="120" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="415" y="213" text-anchor="middle" class="mono" style="font-size:11px">proc B1</text>
    <rect x="505" y="192" width="120" height="32" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="565" y="213" text-anchor="middle" class="mono" style="font-size:11px">proc B2</text>
    <text x="20" y="258" style="fill:var(--accent-ink);font-weight:700;font-size:13px">one beat: build N+1 ‖ forward N ‖ process N-1</text>
  </svg>
  <div class="figcap"><b>Fig 59-2 · the overlap pipeline</b> — build, forward, and process lanes are each staggered one step apart; once in steady state (t4), a single beat has the CPU build batch N+1, the GPU run forward N, and the CPU process batch N-1—all three always in parallel.</div>
</div>

<h2>3. The other beads: chunked prefill, CUDA graphs, process split</h2>
<p>The same thread strings three more beads. <strong>Chunked prefill (L22)</strong> tackles a different bubble: if a very long prompt's prefill runs all at once, it fills the entire forward and every request currently decoding gets <strong>stalled</strong> waiting for it, causing a latency spike. SGLang <strong>slices</strong> the long prefill into chunks and <strong>interleaves</strong> them with decode—each step packs in a little new-prompt prefill while still serving the old generating requests—so a single big job can't clog the whole pipeline. Still the same idea: "don't let one segment break the GPU's rhythm," thinning a long-bubble-causing big task into many small ones that disturb no one.</p>
<p><strong>CUDA graphs (L27)</strong> fight the <strong>kernel-launch overhead</strong> bubble: every CUDA kernel launch costs the CPU a fixed amount, a decode step launches dozens to hundreds of kernels, which adds up to real CPU time, and during it the GPU often waits for the next kernel to be submitted. CUDA graphs <strong>record the whole chain of kernels into one graph</strong> and later <strong>replay</strong> it to run the entire sequence on the GPU, so the CPU barely participates in launching one by one—another form of "don't let the CPU bottleneck the GPU," aimed squarely at decode, the <strong>many-small-kernel</strong> scenario most easily dragged down by launch overhead.</p>
<p>Finally, the <strong>process split + IPC (L14/16)</strong>. SGLang puts <span class="mono">TokenizerManager</span> and <span class="mono">DetokenizerManager</span> in <strong>separate processes</strong> from the scheduler + GPU, passing messages over IPC (inter-process communication). So pure-CPU work like tokenize and detokenize is <strong>not</strong> on the forward's critical path at all—it runs in other processes, naturally parallel to GPU compute, no deliberate "overlap" even required, with the OS process scheduler staggering them for you. It also sidesteps Python's global interpreter lock (GIL): the tokenizer process's CPU usage can't slow the scheduler process. The process split is, in effect, <strong>physically isolating</strong> the bubble off the critical path.</p>

<h2>4. Threading every bead into one sentence</h2>
<p>Now step back and see the whole. Lessons 18, 21, 22, 27, 14, 16 look like six different mechanisms, but at the bone they are six implementations of <strong>one design philosophy</strong>: <span class="mono">enumerate every CPU bubble that could make the GPU wait, then hide it behind GPU compute, or move it to another process entirely</span>. The event loop gives the "pump" skeleton, overlap scheduling hides scheduling and result processing in the shadow, chunked prefill thins long tasks, CUDA graphs erase launch overhead, the process split moves tokenize/detokenize away—each bead targets one specific shape of bubble.</p>
<p>Once you grasp this thread, every scheduling-related switch (<span class="mono">--disable-overlap-schedule</span>, <span class="mono">--chunked-prefill-size</span>, <span class="mono">--disable-cuda-graph</span>) immediately reveals which kind of bubble it is fighting, and you can predict which bubble re-emerges when you turn it off. That is the power of synthesis: what you remember is no longer six isolated parameters but <strong>one principle that lets you derive new optimizations yourself</strong>. Next time you meet a new CPU cost, you'll instinctively ask: "Can it hide behind GPU compute? Can it move to another process?"—and that is the true meaning of zero-overhead scheduling as the design north star of the entire scheduling subsystem.</p>

<h2>5. One concrete round: walking the principle through</h2>
<p>After all the abstraction, let's ground it in a real generation round and watch the zero-overhead principle act on every beat. Suppose the engine is decoding dozens of concurrent requests when a new request carrying a very long prompt arrives. With no optimization, this beat goes: the CPU stops to run the new request's full prefill (slow) while every old request waits idle; only after prefill does decode get a turn; once the decode forward is launched the CPU must fire off hundreds of kernels one by one; after the forward returns it serially samples, appends tokens, and checks stops. The GPU has at least <strong>three</strong> obvious idle stretches in this single beat.</p>
<p>Now turn on SGLang's full machinery. <strong>Chunked prefill</strong> first slices that long prompt into chunks and packs only one chunk into this beat, <strong>batched together</strong> with the dozens of old requests' decode, so no one starves. <strong>CUDA graphs</strong> let that batch's hundreds of kernels replay in one shot, the CPU no longer launching them individually. <strong>Overlap scheduling</strong> ensures that while this beat's forward runs on the GPU, the CPU is already <strong>picking the next beat's requests and allocating pages</strong> while <strong>appending the previous beat's sampled tokens</strong> back into their sequences and checking which requests should stop. Those finishing requests are sent back to the user through the <strong>process-split</strong> channel—<span class="mono">DetokenizerManager</span> turns token ids back into text in another process, never touching the current forward's critical path.</p>
<p>After one beat, the three idle stretches are packed full: the long prefill is thinned, kernel launches are erased, scheduling and result processing are hidden in the shadow, and tokenize/detokenize are physically isolated into other processes. The GPU is busy almost end to end. Repeat this beat a thousand times and you have a full generation; carry the zero-overhead principle into every beat and you have SGLang's underlying secret to high throughput. <strong>You'll find that "zero-overhead" is not one clever algorithm but a near-obsessive habit: whenever you see any stretch where the GPU might wait, find some other useful work to stuff into it.</strong></p>

<div class="flow"><div class="node">CPU pick/allocate</div><div class="arrow">→</div><div class="node">GPU forward(N)</div><div class="arrow">→</div><div class="node">CPU sample/append</div><div class="arrow">→</div><div class="node">GPU idle (bubble)</div><div class="arrow">→</div><div class="node">CPU pick(N+1)</div></div>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>GPU is running forward(batch N)</h4><p class="mono">run_batch(batch_N)</p><p>This is the most expensive, longest step in the pipeline; the GPU runs at full tilt.</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>same instant · CPU assembles batch N+1</h4><p class="mono">get_next_batch_to_run()</p><p>Pick requests, allocate KV pages, move tensors—all done quietly in the GPU's busy shadow.</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>same instant · CPU processes batch N-1's results</h4><p class="mono">process_batch_result(prev)</p><p>Sample, append token, check stop, send results back—parallel to GPU compute.</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>→ GPU never idles</h4><p class="mono">enable_overlap=True</p><p>The stove never stops; CPU overhead is fully invisible in wall-clock time.</p></div></div></div>

<table class="t"><tr><th>CPU bubble</th><th>how SGLang hides it</th><th>lesson</th></tr>
<tr><td>schedule / result processing</td><td>event loop + overlap scheduler, CPU work tucked into the GPU forward's shadow</td><td>L18 / L21</td></tr>
<tr><td>a long prefill clogs the pipeline</td><td>chunked prefill, slice the long prefill and interleave with decode</td><td>L22</td></tr>
<tr><td>per-step kernel-launch overhead</td><td>CUDA graphs record once, replay many times, CPU stops launching one by one</td><td>L27</td></tr>
<tr><td>tokenize / detokenize on the critical path</td><td>process split + IPC, Tokenizer/Detokenizer run in other processes</td><td>L14 / L16</td></tr></table>

<div class="cols"><div class="col"><strong>overlap OFF</strong><br>pick → forward → process results run serially; the GPU is <span class="mono">idle</span> during both CPU segments, occupancy drops to half or worse, and the bubble is plainly visible.</div><div class="col"><strong>overlap ON</strong><br>while the GPU runs forward(N), the CPU has already prepped N+1 and finished N-1; the GPU is <span class="mono">always busy</span> and the CPU overhead is invisible in wall-clock time—this is zero-overhead.</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler</span><span class="ln">while the GPU runs step N, the CPU is already preparing step N+1</span></div><pre>class Scheduler:
    def __init__(self, server_args, ...):
        # CPU bubbles are the enemy: hide them behind the GPU forward
        self.enable_overlap = not server_args.disable_overlap_schedule
        ...
    def event_loop_overlap(self):
        # while the GPU runs forward(batch N), the CPU already builds batch N+1,
        # and the PREVIOUS step's result is processed in parallel -&gt; GPU never idles
        while True:
            batch = self.get_next_batch_to_run()    # CPU: schedule + allocate
            result = self.run_batch(batch)          # launch GPU forward (async)
            self.process_batch_result(prev_result)  # overlap: handle the prior step
            prev_result = result</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.event_loop_overlap</span><span class="ln">overlap loop: process the previous batch while the GPU runs the current one</span></div><pre>def event_loop_overlap(self):
    # overlap CPU work with the GPU forward: finish the PREVIOUS batch's
    # results while the current batch is still running on the GPU.
    self.result_queue = deque()
    while True:
        recv = self.request_receiver.recv_requests()
        self.process_input_requests(recv)
        batch = self.get_next_batch_to_run()        # build batch N (CPU)
        result = self.run_batch(batch)              # launch forward N (GPU)
        self.result_queue.append((batch, result))
        # one step out of phase: process batch N-1 now
        self.process_batch_result(*self.result_queue.popleft())</pre></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>One thread</strong>: the CPU must never make the GPU wait. Any CPU work <strong>before or after</strong> the forward is a potential bubble.</li>
<li><strong>Overlap is the core tool</strong>: while the GPU runs forward(N), the CPU already assembles N+1 and processes N-1 in parallel (<span class="mono">enable_overlap</span>, L18/L21).</li>
<li><strong>Many bubbles, many hiding tricks</strong>: chunked prefill stops long-prefill stalls (L22), CUDA graphs erase kernel-launch overhead (L27), the process split moves tokenize/detokenize off the critical path (L14/L16).</li>
<li><strong>The faster the GPU, the deadlier the bubble</strong>: killing bubbles graduates from nice-to-have to the top factor deciding throughput; the more advanced the hardware, the more you must hide CPU overhead.</li>
<li><strong>Synthesis</strong>: six lessons, six mechanisms, one philosophy—find every CPU bubble and hide it behind GPU compute. Remembering this one principle beats memorizing six isolated parameters, because it lets you <strong>derive</strong> the next optimization yourself.</li>
</ul></div>
"""}
LESSON_60 = {"zh": r"""
<p class="lead">前面五十多课，我们拆过调度器、拆过 KV 缓存、拆过注意力后端、拆过分离式部署。这一课不引入任何新机制，而是退后一步，找出那条贯穿全书的<strong>主线</strong>：在一台 GPU 上，<span class="mono">prefill</span>（预填充）和 <span class="mono">decode</span>（解码）根本就是<strong>两种不同的负载</strong>。看懂这一个对立，你会发现 SGLang 许多看似无关的设计，其实都在回答同一个问题。这一课的目标不是教你新东西，而是把前面学过的零件<strong>串成一条线</strong>，让你在脑子里建立一张"为什么这么设计"的地图。</p>
<p>方法很简单：我们只盯住一条轴——prefill 和 decode 的对立——然后看 SGLang 是怎么沿着这条轴做取舍的。你会惊讶地发现，调度、批处理、KV 缓存、分离部署这些原本各讲各的话题，串起来竟是同一个故事。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一家餐厅的厨房，要同时干两件性质完全不同的活。</p>
<p>第一件是<strong>备料</strong>：开店前把一大筐菜一次性洗好切好。这活儿<strong>吃刀工和人手</strong>——人越多、案板越大，切得越快，所有灶台、所有厨师全开足马力，这是<span class="mono">compute-bound</span>（算力受限）。</p>
<p>第二件是<strong>上菜</strong>：客人点一道、做一道，一道接一道地端出去。这活儿不缺刀工，缺的是<strong>来回跑腿</strong>——每上一道都要跑一趟冷库取食材、跑一趟取调料，厨师的手大半时间在等，真正在切的时间很少，这是 <span class="mono">bandwidth-bound</span>（带宽受限）。</p>
<p>同一个厨房（一块 GPU）必须把两件活都干好。要命的是它们会<strong>互相抢厨房</strong>：一旦埋头切那一大筐菜，正等着上菜的客人就全都卡住了。SGLang 的许多设计，本质上就是在调解"备料"和"上菜"这两种负载的矛盾。</p>
<p>顺着这个类比再想一层：你不会因为"上菜要跑腿"就去给厨师换一把更快的刀——刀再快也省不了跑冷库的那几趟。同理，decode 慢不是因为算力不够，而是因为来回搬数据慢，换更猛的算力卡未必管用。聪明的餐厅老板会怎么办？要么<strong>把备料切成小段，插在上菜的空当里做</strong>（这就是 chunked prefill）；要么<strong>干脆开两个厨房</strong>，一个专门备料、一个专门上菜，备好的料用传送带送过去（这就是 PD 分离）。两种思路，正对应 SGLang 的两条路线。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>这是一条<strong>根本对立（root tension）</strong>，不是某个模块的细节：</p>
<ul>
<li><strong>Prefill 是一次大并行</strong>：把整段 prompt 一口气喂进模型，几百上千个 token 同时过每一层，GPU 的乘加单元被<strong>喂得饱饱的</strong>——算力是瓶颈。</li>
<li><strong>Decode 是逐字挤牙膏</strong>：一次只生成一个 token，但每生成一个就要把<strong>整套权重 + 全部 KV 缓存重新读一遍</strong>，乘加单元大半时间在空转——显存带宽是瓶颈。</li>
<li><strong>一个引擎要同时服务两者</strong>，而它们在同一块 GPU 上<strong>互相打架</strong>。</li>
</ul>
<p>SGLang 有两条调解路线：要么<strong>同卡分时</strong>（第22课的 chunked prefill，把 prefill 切片，插进 decode 的缝隙里），要么<strong>分机分离</strong>（第45课的 PD 分离，给两种负载各自一池 GPU，再把 KV 搬过去）。两条路看似毫不相干，背后却是同一道题。</p>
</div>

<h2>一、为什么 prefill 与 decode 是两种负载</h2>
<p>这正是<strong>第4课</strong>埋下的种子。当时我们第一次区分了 <span class="mono">compute-bound</span> 与 <span class="mono">bandwidth-bound</span>：一个算子到底卡在"算得不够快"还是"数据搬得不够快"，取决于它的<strong>算术强度</strong>（每读一字节数据能做多少次乘加）。</p>
<p>Prefill 的算术强度<strong>很高</strong>。一次处理整段 prompt，意味着同一份权重被成百上千个 token 复用——权重读一次，做几百次乘加。GPU 的张量核心被喂满，这是它最擅长、最划算的工况。</p>
<p>给个直观的量级感受：几千个 token 的 prompt 基本一口气流过网络的每一层，里面的矩阵乘又大又厚，GPU 跑在接近峰值算力的状态。这正是硬件厂商标在宣传页上那个漂亮 FLOPs 数字所对应的工况——而 prefill 是推理里少数几个真能逼近这个数字的阶段。</p>
<p>Decode 的算术强度<strong>很低</strong>。每步只生成一个 token，却要把<strong>整套模型权重</strong>外加<strong>这条序列至今全部的 KV 缓存</strong>从显存里重新读一遍，读完只做了一个 token 的乘加。绝大部分时间花在<strong>搬数据</strong>上，乘加单元闲着。所以单纯堆算力对 decode 几乎没用，真正的解药是把 batch 做大（让一次权重读取服务更多序列）——但这又牵出新的张力。</p>
<p>把这两条放在一起看，你会得到一个很反直觉的结论：<strong>同一块 GPU，跑 prefill 时几乎被榨干，跑 decode 时却大半空转。</strong>不是 GPU 偷懒，而是 decode 这种负载天生喂不饱它——瓶颈在显存到计算单元之间那条数据通道，不在算力本身。理解了这一点，你才能明白后面所有调度技巧的出发点：它们要么想办法让 decode 蹭上 prefill 的算力，要么干脆把两种负载彻底隔开，各自喂饱各自的瓶颈。</p>
<p>更进一步，这条对立还解释了"为什么 LLM 推理跟训练手感如此不同"。训练几乎全是 compute-bound 的大矩阵乘，堆算力立竿见影；而在线推理里，真正决定用户体验的 decode 阶段是 bandwidth-bound 的，单纯换更强算力的卡未必能让吐字更快。SGLang 之所以在调度、批处理、KV 管理上下这么多功夫，根子都在这条 prefill-vs-decode 的轴上。</p>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="算术强度屋顶线：decode 低强度落在带宽受限的斜坡上，prefill 高强度落在算力受限的平台上">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">算术强度：屋顶线</text>
    <line x1="70" y1="58" x2="70" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="70" y1="240" x2="745" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M745 240 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="408" y="270" text-anchor="middle" style="fill:var(--faint);font-size:12px">算术强度 FLOPs/byte →</text>
    <text x="78" y="52" style="fill:var(--faint);font-size:11px">达到算力 ↑</text>
    <line x1="70" y1="240" x2="380" y2="90" style="stroke:var(--purple);stroke-width:2"/>
    <line x1="380" y1="90" x2="730" y2="90" style="stroke:var(--purple);stroke-width:2"/>
    <text x="196" y="148" transform="rotate(-26 196 148)" style="fill:var(--purple);font-size:11px">带宽屋顶</text>
    <text x="556" y="82" text-anchor="middle" style="fill:var(--purple);font-size:11px">算力屋顶 · 峰值</text>
    <line x1="380" y1="90" x2="380" y2="240" style="stroke:var(--faint);stroke-width:1;stroke-dasharray:4 4"/>
    <text x="380" y="256" text-anchor="middle" style="fill:var(--faint);font-size:10px">脊点</text>
    <text x="208" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">带宽受限区</text>
    <text x="558" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">算力受限区</text>
    <circle cx="170" cy="192" r="7" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:2"/>
    <text x="186" y="186" style="fill:var(--blue);font-size:12px;font-weight:700">decode</text>
    <text x="186" y="202" style="fill:var(--blue);font-size:11px">低强度 · 带宽受限</text>
    <circle cx="556" cy="90" r="7" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:2"/>
    <text x="556" y="118" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">prefill</text>
    <text x="556" y="134" text-anchor="middle" style="fill:var(--amber);font-size:11px">高强度 · 算力受限</text>
  </svg>
  <div class="figcap"><b>图 1 · 算术强度屋顶线</b> — 横轴是算术强度（每读一字节做多少乘加）。<span class="mono">decode</span> 强度低，落在左边的<strong>带宽斜坡</strong>上，算力大半空着；<span class="mono">prefill</span> 强度高，落在右边的<strong>算力平台</strong>上，张量核心被喂满。脊点左边带宽受限、右边算力受限。</div>
</div>

<p>举个具体数字：prefill 一次大矩阵乘里，权重每读一字节能摊上<strong>成千上万次</strong>乘加（compute-bound）；而 decode 每生成一个 token，每读一字节往往只做<strong>个位数</strong>次乘加（bandwidth-bound）。同一块卡，两种负载的算术强度能差出几个数量级——这正是它们落在屋顶线两端的原因。</p>

<h2>二、这个对立如何制造吞吐与延迟的拉扯</h2>
<p><strong>第8课</strong>讲的<strong>吞吐 vs 延迟</strong>之争，根子就在这里。两个关键指标：<span class="mono">TTFT</span>（首字延迟，主要由 prefill 决定）和 <span class="mono">ITL</span>（字间延迟，由 decode 决定）。</p>
<p>把两种负载塞进一个串行队列会发生什么？一个超长 prompt 的 prefill 是一大块算力密集的活，它一旦上 GPU，就会把<strong>所有正在解码的请求全部堵住</strong>——大家的 ITL 瞬间飙高，输出卡顿。反过来，若一味优先 decode 的流畅，新请求的 prefill 又迟迟排不上，TTFT 变长。</p>
<p>于是矛盾凸显：<strong>prefill 想要大块、独占、算力拉满；decode 想要频繁、低延迟、雨露均沾。</strong>同一块 GPU 没法同时满足。这就是为什么 SGLang 不能简单地"先来先做"，而必须在调度层动脑筋。</p>
<p>把这层张力说得再具体一点：假设此刻 GPU 上有 30 条序列在逐字解码，每一步几毫秒，用户能看到字一个个稳稳冒出来。这时来了一个 8000 token 的长 prompt，它的 prefill 需要把这 8000 个 token 一次性过完整个网络——这一大块算力密集的活若整块占住 GPU，那 30 条解码序列就得集体卡顿几十甚至上百毫秒，用户端立刻感到"卡了一下"。反过来，如果为了不卡顿而把这个 prefill 无限往后推，新用户的首字（TTFT）又迟迟出不来。<strong>无论先满足谁，另一方都受损</strong>——这就是一台引擎服务两种负载的代价。</p>
<p>正因为这是结构性的矛盾、而非某次调参没调好，所以它需要结构性的答案。下面两节就是 SGLang 给出的两条结构性出路：一条不动硬件、靠调度（chunked prefill），一条直接动硬件、靠分机（PD 分离）。</p>

<h2>三、同卡分时：chunked prefill</h2>
<p><strong>第22课</strong>的 <span class="mono">chunked prefill</span>（分块预填充）是<strong>共置（co-located）</strong>路线的答案：既然一整块 prefill 会堵住 decode，那就<strong>别让它一整块上</strong>。把长 prompt 切成若干小片，每个 step 只处理一片 prefill，并把它和正在跑的 decode <strong>拼进同一个 batch 一起算</strong>。</p>
<p>这样 GPU 在每一步里都<strong>同时</strong>干着两种活：一小片 prefill 借机会喂饱算力，一批 decode 顺道蹭上这次权重读取。谁都不至于饿死——prefill 不再独占整个 GPU 把 decode 闷住，decode 也不必干等一个巨长 prefill 跑完。这是用<strong>聪明的调度</strong>，在一块 GPU 上把两种负载<strong>时间分片</strong>地揉到一起。</p>
<p>它精妙的地方在于<strong>顺势而为</strong>：decode 本来就是 bandwidth-bound、算力大半空着，那块空着的算力正好拿来跑一小片 compute-bound 的 prefill，几乎不额外占用带宽。换句话说，chunked prefill 让两种瓶颈互补的负载<strong>互相填空</strong>——decode 的空闲算力填给 prefill，prefill 顺手把这一步的权重读取分摊掉。切片的大小（chunk size）成了一个调节旋钮：切得太大，单片 prefill 又开始堵 decode；切得太小，调度和启动的固定开销又摊不下来。这正是第22课反复权衡的那个甜点区。</p>

<h2>四、分机分离：PD 分离</h2>
<p><strong>第45课</strong>的 <span class="mono">PD 分离</span>（Prefill-Decode Disaggregation）是另一条路：既然两种负载性质对立、互相干扰，那干脆<strong>物理上分开</strong>。给 prefill 一池 GPU，给 decode 另一池 GPU，prompt 先在 prefill 池跑完、算出 KV 缓存，再把<strong>KV 跨网络搬到</strong> decode 池继续逐字生成。</p>
<p>好处是<strong>各自饱和各自的瓶颈、互不打扰</strong>：prefill 池可以把算力榨干，decode 池可以专心把 batch 做大、把带宽用足，两边能独立扩缩容、独立调优。代价是多了一道<strong>KV 传输</strong>——这正是下面那段代码所守护的接缝。</p>
<p>为什么这种"看起来更费事"的方案反而值得？因为共置方案里两种负载永远在抢同一块 GPU，调度器再聪明，prefill 切片和 decode 之间总有一点互相挤压；而分离之后，prefill 池里全是清一色的大并行、可以用最适合 compute-bound 的批策略，decode 池里全是清一色的逐字生成、可以把 batch 堆到带宽极限——<strong>每一池都不再被另一种负载拖累</strong>。在超大规模、对 TTFT 和 ITL 都有严格 SLA 的线上服务里，这种物理隔离往往比"在一块卡上精打细算"更省心、也更稳。</p>
<p>当然，分离不是免费的午餐。KV 缓存可能高达每条序列几百 MB，要在 prefill 池算完后跨网络搬到 decode 池，这条传输链路的带宽与延迟就成了新的关键路径——搬得慢，decode 侧就得干等。所以 PD 分离把一个"GPU 内部的调度问题"换成了一个"跨机的数据搬运问题"，而这个搬运问题的接口，正由 <span class="mono">BaseKVManager</span> 来定义。</p>

<div class="fig">
  <svg viewBox="0 0 820 320" role="img" aria-label="同机时分对比分离独立池：左边一块 GPU 池时分 prefill 与 decode 互相争抢，右边 prefill 池与 decode 池分开、各自饱和，中间靠 KV 传输相连">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">同机时分 vs 分离独立池</text>
    <rect x="24" y="56" width="372" height="236" rx="12" style="fill:var(--panel-2);stroke:var(--line)"/>
    <text x="210" y="84" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">同机时分 · 争抢</text>
    <rect x="60" y="104" width="300" height="118" rx="10" style="fill:var(--faint);stroke:var(--line)" fill-opacity="0.06"/>
    <text x="210" y="126" text-anchor="middle" style="fill:var(--muted);font-size:11px">一块 GPU 池</text>
    <rect x="80" y="142" width="92" height="58" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="126" y="176" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">prefill</text>
    <rect x="196" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="220" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="244" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="268" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="292" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="316" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="210" y="252" text-anchor="middle" style="fill:var(--muted);font-size:11px">两种负载互相争抢</text>
    <rect x="424" y="56" width="372" height="236" rx="12" style="fill:var(--panel-2);stroke:var(--line)"/>
    <text x="610" y="84" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">分离独立池 · 隔离</text>
    <rect x="448" y="120" width="124" height="96" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="510" y="158" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">prefill 池</text>
    <text x="510" y="178" text-anchor="middle" style="fill:var(--amber);font-size:11px">算力拉满</text>
    <rect x="648" y="120" width="124" height="96" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="710" y="158" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">decode 池</text>
    <text x="710" y="178" text-anchor="middle" style="fill:var(--blue);font-size:11px">带宽用足</text>
    <line x1="572" y1="168" x2="640" y2="168" style="stroke:var(--teal);stroke-width:2"/>
    <path d="M648 168 l-9 -4 v8 z" style="fill:var(--teal)"/>
    <text x="610" y="150" text-anchor="middle" style="fill:var(--teal);font-size:11px">KV 传输</text>
    <text x="610" y="252" text-anchor="middle" style="fill:var(--muted);font-size:11px">各自饱和 · 互不打扰</text>
  </svg>
  <div class="figcap"><b>图 2 · 同机时分 vs 分离独立池</b> — 左边一块 GPU 池<strong>时分</strong>跑 prefill 与 decode，两种负载互相争抢、彼此打断；右边把 <span class="mono">prefill 池</span>与 <span class="mono">decode 池</span><strong>分开</strong>，各自按自己的瓶颈定容量、组批，中间靠一次 <strong>KV 传输</strong>相连，互不打扰。</div>
</div>

<p>正因为分离，你可以按真实流量<strong>各自扩缩容</strong>：prompt 很长、prefill 吃紧就多加 prefill 卡；输出很长、decode 吃紧就多加 decode 卡，两边互不牵制。这种独立伸缩，是共置方案给不了的。</p>

<div class="cols"><div class="col">
<p><strong>Prefill：一次大并行（算力受限）</strong></p>
<p>整段 prompt 一口气进来，几百上千 token 同时过每一层。权重读一次、复用千百次，GPU 乘加单元<strong>被喂满</strong>。</p>
<p><span class="mono">compute-bound</span>：瓶颈是算得多快。一块又宽又厚的活。</p>
</div><div class="col">
<p><strong>Decode：逐字挤牙膏（带宽受限）</strong></p>
<p>一次只出一个 token，却要把整套权重 + 全部 KV 重读一遍，乘加单元大半空转。</p>
<p><span class="mono">bandwidth-bound</span>：瓶颈是搬得多快。一根又细又长的活。</p>
</div></div>

<table class="t">
<tr><th>设计回应</th><th>它在解的矛盾</th><th>出处</th></tr>
<tr><td>根本对比：compute-bound vs bandwidth-bound</td><td>为什么两种负载天生不同</td><td>第4课</td></tr>
<tr><td>吞吐 vs 延迟（TTFT vs ITL）</td><td>长 prefill 堵住大家的 decode</td><td>第8课</td></tr>
<tr><td>同卡分时：chunked prefill</td><td>切片 prefill，插进 decode 缝隙，谁都不饿死</td><td>第22课</td></tr>
<tr><td>分机分离：PD 分离</td><td>各给一池 GPU + 搬 KV，各自饱和不互扰</td><td>第45课</td></tr>
</table>

<div class="flow"><div class="node">请求 request</div><div class="arrow">→</div><div class="node">prefill 负载（算力受限）</div><div class="arrow">→</div><div class="node">KV 缓存</div><div class="arrow">→</div><div class="node">decode 负载（带宽受限）</div><div class="arrow">→</div><div class="node">逐字流式输出</div></div>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>共置 co-located</h4><p>一块 GPU <strong>时间分片</strong>同时跑两种负载——chunked prefill 把 prefill 切片，插进 decode 的缝隙里。</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>分离 disaggregated</h4><p>prefill 池与 decode 池<strong>各自独立</strong>，中间靠 <strong>KV 传输</strong>把状态从 prefill 侧搬到 decode 侧。</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>同一根本对立</h4><p>两种调解：要么聪明调度，要么物理分开。</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::BaseKVManager</span><span class="ln">让 prefill 与 decode 分居两池的接缝</span></div><pre>class BaseKVManager(ABC):
    # base class for managing KV-transfer state — the seam that lets prefill
    # and decode live on SEPARATE pools
    @abstractmethod
    def __init__(self, args, disaggregation_mode, ...):
        # disaggregation_mode says which side this is: PREFILL or DECODE
        ...
    @abstractmethod
    def register_to_bootstrap(self):
        # register so a prefill worker can be PAIRED with a decode worker,
        # then KV is transferred from the prefill side to the decode side
        ...</pre></div>

<p>看这段代码：<span class="mono">BaseKVManager</span> 就是 PD 分离那条路的"接缝"。构造时传入的 <span class="mono">disaggregation_mode</span> 决定了当前进程是 prefill 侧还是 decode 侧——同一套接口，两种角色。<span class="mono">register_to_bootstrap()</span> 让一个 prefill worker 能和一个 decode worker <strong>配对</strong>，配对成功后，KV 缓存就从 prefill 侧传到 decode 侧。整个分离式架构的复杂度，都被这层抽象包住了。</p>
<p>留意这里的设计哲学：它是一个 <span class="mono">ABC</span>（抽象基类），所有方法都标了 <span class="mono">@abstractmethod</span>。这意味着 SGLang 并不绑定某一种具体的传输实现——底层可以是 RDMA、可以是 NVLink、也可以是别的后端，只要它实现了这套接口。<strong>"prefill 和 decode 分居两池"这个抽象，和"具体用什么管子搬 KV"这个实现，被干净地解耦开了。</strong>这也呼应了全书反复出现的另一条主线：把"是什么"和"怎么做"分层，让上层的调度决策不必关心下层的硬件细节。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::KVPoll</span><span class="ln">KV 传输的状态机：轮询直到 Success</span></div><pre>class KVPoll:
    # the state of a prefill -&gt; decode KV transfer; poll() returns one
    # of these until the transfer is done (or failed).
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4</pre></div>

<p>这段 <span class="mono">KVPoll</span> 就是上面那道 KV 传输接缝的"仪表盘"：每一条跨池搬运的 KV，都在这五个整数状态里走——从 <span class="mono">Bootstrapping</span>（建连配对）到 <span class="mono">WaitingForInput</span>、<span class="mono">Transferring</span>（搬运中），一路轮询直到 <span class="mono">Success</span>，中途出错则落到 <span class="mono">Failed</span>。decode 侧正是靠不停 <span class="mono">poll()</span> 这个状态，判断 prefill 算好的 KV 到没到、能不能开始逐字生成。一个朴素的状态机，把"跨机搬数据"这件麻烦事收拢成了一个可观测、可重试的循环。</p>

<h2>五、一个轴，看懂许多决定</h2>
<p>现在把镜头拉到最远。第4课的 compute-bound vs bandwidth-bound、第8课的吞吐 vs 延迟、第22课的 chunked prefill、第45课的 PD 分离——这四课乍看分属四个话题，其实是同一根轴上的四个点。<strong>根本对立</strong>（prefill 与 decode 性质相反）产生<strong>张力</strong>（它们抢同一块 GPU），张力逼出<strong>两种调解</strong>（共置分时、或分机分离）。</p>
<p>掌握了这条轴，很多原本要死记硬背的设计决策就变成了可以推导的结论。为什么要有 chunked prefill？因为不想让长 prefill 堵 decode。为什么要做 PD 分离？因为想让两种瓶颈各自饱和。为什么 decode 阶段拼命想把 batch 做大？因为它 bandwidth-bound，多一条序列几乎不额外读权重。<strong>看似零散的优化，背后是同一句话：prefill 和 decode 是两种负载，一套引擎要么聪明地让它们共处，要么干脆把它们分开。</strong></p>
<p>所以下次你再遇到 SGLang 里某个陌生的调度参数或部署模式，别急着死记，先问自己一句：<strong>它是在帮 prefill 和 decode 更好地共处，还是在帮它们更彻底地分开？</strong>答案几乎总能落在这条轴的某一端。这就是"一个轴看懂全局"的真正含义——你记住的不再是几十条孤立的规则，而是一条能生成这些规则的根本原理。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>一条根本对立</strong>：prefill 是 <span class="mono">compute-bound</span> 的大并行一次过，decode 是 <span class="mono">bandwidth-bound</span> 的逐字挤牙膏（第4课）。</li>
<li><strong>对立制造张力</strong>：长 prefill 会一口气堵住大家的 decode，于是有了吞吐 vs 延迟、TTFT vs ITL 的拉扯（第8课）。</li>
<li><strong>同卡分时</strong>：chunked prefill 把 prefill 切片、插进 decode 缝隙，一块 GPU 时间分片地服务两种负载（第22课）。</li>
<li><strong>分机分离</strong>：PD 分离给两种负载各一池 GPU，再把 KV 搬过去，各自饱和各自瓶颈、互不干扰（第45课）。</li>
<li><strong>一个轴看懂全局</strong>：prefill-vs-decode 这一条轴，能解释 SGLang 许多看似无关的设计——要么靠聪明的调度共处，要么靠物理上的分开隔离。</li>
</ul></div>
""", "en": r"""
<p class="lead">Over fifty-some lessons we've taken apart the scheduler, the KV cache, the attention backends, the disaggregated deployment. This lesson adds no new mechanism. It steps back to find the one <strong>through-line</strong> running across the whole book: on a single GPU, <span class="mono">prefill</span> and <span class="mono">decode</span> are fundamentally <strong>two different workloads</strong>. Grasp this one contrast and you'll see that many of SGLang's seemingly unrelated decisions are answering the very same question. The goal here isn't to teach you something new but to <strong>thread the parts you've already learned into one line</strong>, building a mental map of "why it's designed this way".</p>
<p>The method is simple: we fix our eyes on a single axis—the prefill-vs-decode contrast—and watch how SGLang makes its trade-offs along it. You'll be surprised to find that scheduling, batching, KV caching, and disaggregated deployment, topics that once stood apart, turn out to be one story.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a restaurant kitchen doing two jobs of completely different nature at once.</p>
<p>The first is <strong>prep</strong>: before opening, wash and chop a whole crate of vegetables in one go. This job is <strong>about knife-hands</strong>—the more cooks and the bigger the boards, the faster it goes; every stove and every cook runs flat out. That's <span class="mono">compute-bound</span>.</p>
<p>The second is <strong>serving</strong>: a guest orders one dish, you make one, plate after plate. This job isn't short on knife skills, it's short on <strong>legwork</strong>—every dish means a trip to the cold store for ingredients and a trip for seasoning, so the cook's hands wait most of the time and chop very little. That's <span class="mono">bandwidth-bound</span>.</p>
<p>One kitchen (one GPU) must do both well. The catch: they <strong>fight over the kitchen</strong>. The moment you bury yourself chopping that big crate, every guest waiting to be served is stuck. Many of SGLang's designs are, at heart, reconciling these two workloads—"prep" and "serving".</p>
<p>Follow the analogy one layer deeper: you wouldn't hand the cook a faster knife just because "serving means legwork"—no knife saves those trips to the cold store. Likewise, decode is slow not because compute is short but because shuttling data is slow, and a beefier compute card may not help. What would a smart owner do? Either <strong>chop the prep into small bits and slot them into the gaps between serving</strong> (that's chunked prefill); or <strong>just open two kitchens</strong>, one only prepping, one only serving, with a conveyor moving the prepped goods across (that's PD disaggregation). Two ideas, matching SGLang's two paths.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>This is a <strong>root tension</strong>, not a detail of any one module:</p>
<ul>
<li><strong>Prefill is one big parallel pass</strong>: the whole prompt goes in at once, hundreds or thousands of tokens flow through every layer together, and the GPU's multiply-add units are <strong>kept fully fed</strong>—compute is the bottleneck.</li>
<li><strong>Decode squeezes out one token at a time</strong>: each step generates a single token, yet must <strong>re-read the entire set of weights + all the KV cache</strong>, leaving the math units mostly idle—memory bandwidth is the bottleneck.</li>
<li><strong>One engine must serve both</strong>, and on the same GPU they <strong>fight each other</strong>.</li>
</ul>
<p>SGLang has two ways to reconcile this: either <strong>time-share one GPU</strong> (Lesson 22's chunked prefill, slicing prefill and interleaving it into decode's gaps), or <strong>separate the machines</strong> (Lesson 45's PD disaggregation, giving each workload its own pool of GPUs and transferring the KV across). The two paths look unrelated, but they answer the same question.</p>
</div>

<h2>1. Why prefill and decode are two workloads</h2>
<p>This is exactly the seed planted in <strong>Lesson 4</strong>. There we first distinguished <span class="mono">compute-bound</span> from <span class="mono">bandwidth-bound</span>: whether an operator is stuck "not computing fast enough" or "not moving data fast enough" depends on its <strong>arithmetic intensity</strong> (how many multiply-adds you do per byte read).</p>
<p>Prefill has <strong>high</strong> arithmetic intensity. Processing the whole prompt at once means the same weights are reused by hundreds or thousands of tokens—read the weights once, do hundreds of multiply-adds. The tensor cores are kept full; it's the GPU's best, most economical regime.</p>
<p>To picture the scale: a prompt of a few thousand tokens flows through every layer of the network in essentially one shot, the GEMMs are big and fat, and the GPU runs near its peak FLOPs. This is the regime hardware vendors quote their headline numbers for—and prefill is one of the few inference phases that actually reaches it.</p>
<p>Decode has <strong>low</strong> arithmetic intensity. Each step generates one token, yet must re-read the <strong>entire model's weights</strong> plus <strong>all of this sequence's KV cache so far</strong> from memory, only to do one token's worth of multiply-adds. Most of the time is spent <strong>moving data</strong> while the math units sit idle. So piling on raw compute barely helps decode; the real cure is to grow the batch (so one weight read serves more sequences)—which raises a new tension.</p>
<p>Put the two side by side and you reach a counterintuitive conclusion: <strong>the same GPU is nearly wrung dry during prefill, yet mostly idle during decode.</strong> The GPU isn't slacking—decode is a workload that inherently can't keep it fed; the bottleneck is the data path from memory to the compute units, not the compute itself. Grasp this and you understand the starting point of every scheduling trick that follows: either find a way for decode to piggyback on prefill's compute, or separate the two workloads entirely so each saturates its own bottleneck.</p>
<p>Further, this contrast explains "why LLM inference feels so different from training." Training is almost entirely compute-bound big matmuls, where adding compute pays off immediately; in online serving, the decode phase that actually shapes the user experience is bandwidth-bound, and a stronger compute card alone may not make tokens come out faster. The reason SGLang invests so much in scheduling, batching, and KV management all traces back to this prefill-vs-decode axis.</p>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Arithmetic-intensity roofline: decode at low intensity sits on the bandwidth-bound ramp, prefill at high intensity sits on the compute-bound plateau">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">Arithmetic intensity: roofline</text>
    <line x1="70" y1="58" x2="70" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="70" y1="240" x2="745" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M745 240 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="408" y="270" text-anchor="middle" style="fill:var(--faint);font-size:12px">arithmetic intensity FLOPs/byte →</text>
    <text x="78" y="52" style="fill:var(--faint);font-size:11px">achieved FLOP/s ↑</text>
    <line x1="70" y1="240" x2="380" y2="90" style="stroke:var(--purple);stroke-width:2"/>
    <line x1="380" y1="90" x2="730" y2="90" style="stroke:var(--purple);stroke-width:2"/>
    <text x="196" y="148" transform="rotate(-26 196 148)" style="fill:var(--purple);font-size:11px">bw roof</text>
    <text x="556" y="82" text-anchor="middle" style="fill:var(--purple);font-size:11px">compute roof · peak</text>
    <line x1="380" y1="90" x2="380" y2="240" style="stroke:var(--faint);stroke-width:1;stroke-dasharray:4 4"/>
    <text x="380" y="256" text-anchor="middle" style="fill:var(--faint);font-size:10px">ridge</text>
    <text x="208" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">bandwidth-bound</text>
    <text x="558" y="228" text-anchor="middle" style="fill:var(--muted);font-size:11px">compute-bound</text>
    <circle cx="170" cy="192" r="7" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:2"/>
    <text x="186" y="186" style="fill:var(--blue);font-size:12px;font-weight:700">decode</text>
    <text x="186" y="202" style="fill:var(--blue);font-size:11px">low AI · bw-bound</text>
    <circle cx="556" cy="90" r="7" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:2"/>
    <text x="556" y="118" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">prefill</text>
    <text x="556" y="134" text-anchor="middle" style="fill:var(--amber);font-size:11px">high AI · compute-bound</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Arithmetic-intensity roofline</b> — the x-axis is arithmetic intensity (multiply-adds per byte read). <span class="mono">decode</span> is low and sits on the left <strong>bandwidth ramp</strong>, most of its compute idle; <span class="mono">prefill</span> is high and sits on the right <strong>compute plateau</strong>, tensor cores kept full. Left of the ridge is bandwidth-bound, right of it compute-bound.</div>
</div>

<p>Concretely: in one big prefill GEMM, every byte of weights read is amortized over <strong>thousands</strong> of multiply-adds (compute-bound); in decode, each generated token does only a <strong>handful</strong> of multiply-adds per byte read (bandwidth-bound). On the same card the two workloads' arithmetic intensities differ by orders of magnitude—which is exactly why they land at opposite ends of the roofline.</p>

<h2>2. How this contrast creates the throughput-vs-latency pull</h2>
<p><strong>Lesson 8</strong>'s <strong>throughput vs latency</strong> fight is rooted right here. Two key metrics: <span class="mono">TTFT</span> (time to first token, driven mainly by prefill) and <span class="mono">ITL</span> (inter-token latency, driven by decode).</p>
<p>What happens if you stuff both workloads into one serial queue? A very long prompt's prefill is a big compute-heavy chunk; once it hits the GPU it <strong>blocks every request that's currently decoding</strong>—everyone's ITL spikes and output stutters. Conversely, if you always favor smooth decode, new requests' prefill never gets scheduled and TTFT climbs.</p>
<p>So the conflict surfaces: <strong>prefill wants big, exclusive, compute-maxed; decode wants frequent, low-latency, fair-shared.</strong> One GPU can't satisfy both at once. That's why SGLang can't simply go "first come first served" and must get clever at the scheduling layer.</p>
<p>To make the tension concrete: suppose 30 sequences are decoding token by token on the GPU right now, a few milliseconds per step, so the user sees characters stream out steadily. Now an 8000-token prompt arrives; its prefill must push all 8000 tokens through the whole network in one pass—and if that compute-heavy block occupies the GPU as one chunk, those 30 decoding sequences stall for tens or even hundreds of milliseconds, and the user feels a "hitch". Conversely, if you defer that prefill indefinitely to avoid the hitch, the new user's first token (TTFT) never arrives. <strong>Whichever side you favor, the other suffers</strong>—that's the price of one engine serving two workloads.</p>
<p>Precisely because this is a structural conflict—not a one-off mistuning—it needs a structural answer. The next two sections give SGLang's two structural ways out: one leaves the hardware alone and relies on scheduling (chunked prefill); the other changes the hardware layout directly and relies on separate machines (PD disaggregation).</p>

<h2>3. Co-located time-sharing: chunked prefill</h2>
<p><strong>Lesson 22</strong>'s <span class="mono">chunked prefill</span> is the <strong>co-located</strong> answer: since a whole block of prefill blocks decode, <strong>don't let it go in as one block</strong>. Slice the long prompt into chunks, process only one chunk of prefill per step, and <strong>pack it into the same batch</strong> as the decodes already running.</p>
<p>Now in every step the GPU does <strong>both</strong> jobs: a small prefill chunk grabs the chance to fill the compute, and a batch of decodes piggybacks on that same weight read. Neither starves—prefill no longer monopolizes the whole GPU and smothers decode, and decode no longer idles waiting for one giant prefill to finish. It's <strong>smart scheduling</strong> that <strong>time-slices</strong> two workloads onto one GPU.</p>
<p>The elegance is that it <strong>goes with the grain</strong>: decode is bandwidth-bound with most of its compute sitting idle, and that idle compute is exactly what a small compute-bound prefill chunk can use, with almost no extra bandwidth cost. In other words, chunked prefill lets two workloads with complementary bottlenecks <strong>fill each other's gaps</strong>—decode's spare compute goes to prefill, and prefill amortizes that step's weight read. The chunk size becomes a tuning knob: too big and a single chunk starts blocking decode again; too small and the fixed scheduling/launch overhead can't be amortized. That's the sweet spot Lesson 22 keeps weighing.</p>

<h2>4. Separated machines: PD disaggregation</h2>
<p><strong>Lesson 45</strong>'s <span class="mono">PD disaggregation</span> (Prefill-Decode Disaggregation) is the other path: since the two workloads are opposite in nature and interfere, just <strong>separate them physically</strong>. Give prefill one pool of GPUs and decode another; the prompt runs through the prefill pool first to compute its KV cache, then the <strong>KV is transferred across the network</strong> to the decode pool to keep generating token by token.</p>
<p>The upside is that <strong>each pool saturates its own bottleneck without interference</strong>: the prefill pool can wring out compute, the decode pool can focus on growing the batch and using bandwidth fully, and each can scale and tune independently. The cost is an extra <strong>KV transfer</strong>—which is exactly the seam the code below guards.</p>
<p>Why is this "seemingly more troublesome" approach worth it? Because in the co-located approach the two workloads forever contend for the same GPU; however clever the scheduler, there's always some squeeze between prefill chunks and decode. Once separated, the prefill pool is all uniform big parallel passes and can use the batching strategy best suited to compute-bound work, while the decode pool is all uniform token-by-token generation and can pile the batch up to the bandwidth limit—<strong>each pool is no longer dragged down by the other workload</strong>. In very large-scale online serving with strict SLAs on both TTFT and ITL, this physical isolation is often more reliable and less fiddly than penny-pinching on one card.</p>
<p>Of course, separation is no free lunch. The KV cache can be hundreds of MB per sequence, and after the prefill pool computes it, it must be moved across the network to the decode pool, so that transfer link's bandwidth and latency become a new critical path—move it slowly and the decode side just waits. So PD disaggregation swaps an "in-GPU scheduling problem" for a "cross-machine data-movement problem", and the interface for that movement is exactly what <span class="mono">BaseKVManager</span> defines.</p>

<div class="fig">
  <svg viewBox="0 0 820 320" role="img" aria-label="Co-located time-share vs disaggregated pools: on the left one GPU pool time-shares prefill and decode that contend, on the right a prefill pool and decode pool are separate and each saturates, joined by a KV transfer">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">Co-located vs disaggregated</text>
    <rect x="24" y="56" width="372" height="236" rx="12" style="fill:var(--panel-2);stroke:var(--line)"/>
    <text x="210" y="84" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">co-located · contend</text>
    <rect x="60" y="104" width="300" height="118" rx="10" style="fill:var(--faint);stroke:var(--line)" fill-opacity="0.06"/>
    <text x="210" y="126" text-anchor="middle" style="fill:var(--muted);font-size:11px">one GPU pool</text>
    <rect x="80" y="142" width="92" height="58" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="126" y="176" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">prefill</text>
    <rect x="196" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="220" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="244" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="268" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="292" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="316" y="142" width="16" height="58" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="210" y="252" text-anchor="middle" style="fill:var(--muted);font-size:11px">two workloads contend</text>
    <rect x="424" y="56" width="372" height="236" rx="12" style="fill:var(--panel-2);stroke:var(--line)"/>
    <text x="610" y="84" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">disaggregated · isolated</text>
    <rect x="448" y="120" width="124" height="96" rx="10" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="510" y="158" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">prefill pool</text>
    <text x="510" y="178" text-anchor="middle" style="fill:var(--amber);font-size:11px">compute-max</text>
    <rect x="648" y="120" width="124" height="96" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="710" y="158" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">decode pool</text>
    <text x="710" y="178" text-anchor="middle" style="fill:var(--blue);font-size:11px">bw-full</text>
    <line x1="572" y1="168" x2="640" y2="168" style="stroke:var(--teal);stroke-width:2"/>
    <path d="M648 168 l-9 -4 v8 z" style="fill:var(--teal)"/>
    <text x="610" y="150" text-anchor="middle" style="fill:var(--teal);font-size:11px">KV transfer</text>
    <text x="610" y="252" text-anchor="middle" style="fill:var(--muted);font-size:11px">each saturates · isolated</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Co-located vs disaggregated</b> — on the left one GPU pool <strong>time-shares</strong> prefill and decode, so the two workloads contend and interrupt each other; on the right the <span class="mono">prefill pool</span> and <span class="mono">decode pool</span> are <strong>separate</strong>, each sized and batched for its own bottleneck and joined by a single <strong>KV transfer</strong>, so neither disturbs the other.</div>
</div>

<p>Because they're separated, you can <strong>scale each to real traffic</strong>: add prefill GPUs when prompts are long and prefill is the squeeze; add decode GPUs when outputs are long and decode is the squeeze—neither side constrains the other. That independent scaling is something the co-located approach simply can't give you.</p>

<div class="cols"><div class="col">
<p><strong>Prefill: one big parallel pass (compute-bound)</strong></p>
<p>The whole prompt arrives at once; hundreds-to-thousands of tokens flow through every layer together. Weights are read once and reused hundreds of times, keeping the GPU's multiply-add units <strong>fully fed</strong>.</p>
<p><span class="mono">compute-bound</span>: the bottleneck is how fast you compute. A wide, thick chunk of work.</p>
</div><div class="col">
<p><strong>Decode: one token at a time (bandwidth-bound)</strong></p>
<p>Only one token per step, yet the full weights + all KV are re-read, leaving the math units mostly idle.</p>
<p><span class="mono">bandwidth-bound</span>: the bottleneck is how fast you move data. A skinny, long strip of work.</p>
</div></div>

<table class="t">
<tr><th>Design response</th><th>The tension it resolves</th><th>Source</th></tr>
<tr><td>Root contrast: compute-bound vs bandwidth-bound</td><td>why the two workloads are inherently different</td><td>Lesson 4</td></tr>
<tr><td>Throughput vs latency (TTFT vs ITL)</td><td>a long prefill blocks everyone's decode</td><td>Lesson 8</td></tr>
<tr><td>Co-located time-share: chunked prefill</td><td>slice prefill into decode's gaps, nobody starves</td><td>Lesson 22</td></tr>
<tr><td>Separated machines: PD disaggregation</td><td>a pool each + transfer KV, each saturates without interfering</td><td>Lesson 45</td></tr>
</table>

<div class="flow"><div class="node">request</div><div class="arrow">→</div><div class="node">prefill workload (compute-bound)</div><div class="arrow">→</div><div class="node">KV cache</div><div class="arrow">→</div><div class="node">decode workload (bandwidth-bound)</div><div class="arrow">→</div><div class="node">streamed output</div></div>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>co-located</h4><p>One GPU <strong>time-shares</strong> both workloads—chunked prefill slices prefill and inserts it into decode's gaps.</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>disaggregated</h4><p>The prefill pool and decode pool are <strong>independent</strong>, joined by a <strong>KV transfer</strong> that moves state from the prefill side to the decode side.</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>same root tension</h4><p>Two reconciliations: smart scheduling, or physical separation.</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::BaseKVManager</span><span class="ln">the seam that lets prefill and decode live on separate pools</span></div><pre>class BaseKVManager(ABC):
    # base class for managing KV-transfer state — the seam that lets prefill
    # and decode live on SEPARATE pools
    @abstractmethod
    def __init__(self, args, disaggregation_mode, ...):
        # disaggregation_mode says which side this is: PREFILL or DECODE
        ...
    @abstractmethod
    def register_to_bootstrap(self):
        # register so a prefill worker can be PAIRED with a decode worker,
        # then KV is transferred from the prefill side to the decode side
        ...</pre></div>

<p>Look at the code: <span class="mono">BaseKVManager</span> is the "seam" of the PD-disaggregation path. The <span class="mono">disaggregation_mode</span> passed at construction decides whether this process is the prefill side or the decode side—one interface, two roles. <span class="mono">register_to_bootstrap()</span> lets a prefill worker get <strong>paired</strong> with a decode worker; once paired, the KV cache is transferred from the prefill side to the decode side. The whole complexity of the disaggregated architecture is wrapped up behind this abstraction.</p>
<p>Notice the design philosophy here: it's an <span class="mono">ABC</span> (abstract base class) with every method marked <span class="mono">@abstractmethod</span>. That means SGLang isn't bound to one concrete transport implementation—the backend can be RDMA, NVLink, or something else, as long as it implements this interface. <strong>The abstraction "prefill and decode live in separate pools" is cleanly decoupled from the implementation "which pipe actually moves the KV".</strong> This echoes another through-line of the whole book: layer the "what" apart from the "how", so the upper scheduling decisions need not care about the lower hardware details.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/disaggregation/base/conn.py ::KVPoll</span><span class="ln">the KV-transfer state machine: poll until Success</span></div><pre>class KVPoll:
    # the state of a prefill -&gt; decode KV transfer; poll() returns one
    # of these until the transfer is done (or failed).
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4</pre></div>

<p>This <span class="mono">KVPoll</span> is the "dashboard" of that KV-transfer seam: every cross-pool KV move walks through these five integer states—from <span class="mono">Bootstrapping</span> (connect and pair) through <span class="mono">WaitingForInput</span> and <span class="mono">Transferring</span> (in flight), polled until <span class="mono">Success</span>, dropping to <span class="mono">Failed</span> if anything breaks. The decode side keeps calling <span class="mono">poll()</span> on this state to tell whether the KV computed by prefill has arrived and token-by-token generation can begin. A plain little state machine that wraps the messy business of "moving data across machines" into one observable, retryable loop.</p>

<h2>5. One axis explains many decisions</h2>
<p>Now pull the camera all the way back. Lesson 4's compute-bound vs bandwidth-bound, Lesson 8's throughput vs latency, Lesson 22's chunked prefill, Lesson 45's PD disaggregation—these four lessons look like four separate topics, but they are four points on the same axis. The <strong>root tension</strong> (prefill and decode are opposite in nature) produces a <strong>pull</strong> (they contend for the same GPU), and the pull forces <strong>two reconciliations</strong> (co-located time-sharing, or separated machines).</p>
<p>Master this axis and many design decisions that once needed rote memorization become derivable conclusions. Why have chunked prefill? Because you don't want a long prefill to block decode. Why do PD disaggregation? Because you want each bottleneck to saturate on its own. Why does the decode phase strive so hard to grow the batch? Because it's bandwidth-bound, so one more sequence reads almost no extra weights. <strong>Seemingly scattered optimizations all rest on the same sentence: prefill and decode are two workloads, and one engine must either let them coexist cleverly or separate them outright.</strong></p>
<p>So next time you meet an unfamiliar scheduling knob or deployment mode in SGLang, don't rush to memorize it—first ask yourself: <strong>is it helping prefill and decode coexist better, or helping them separate more thoroughly?</strong> The answer almost always lands on one end of this axis. That's what "one axis explains the whole" really means—what you remember is no longer dozens of isolated rules, but the one root principle that generates them.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>One root tension</strong>: prefill is a <span class="mono">compute-bound</span> big parallel pass; decode is a <span class="mono">bandwidth-bound</span> one-token-at-a-time squeeze (Lesson 4).</li>
<li><strong>The tension creates a pull</strong>: a long prefill blocks everyone's decode, giving rise to throughput vs latency, TTFT vs ITL (Lesson 8).</li>
<li><strong>Co-located time-sharing</strong>: chunked prefill slices prefill into decode's gaps, so one GPU time-slices to serve both workloads (Lesson 22).</li>
<li><strong>Separated machines</strong>: PD disaggregation gives each workload its own GPU pool and transfers the KV across, so each saturates its own bottleneck without interference (Lesson 45).</li>
<li><strong>One axis explains the whole</strong>: the prefill-vs-decode axis explains many seemingly unrelated SGLang decisions—reconciled either by smart scheduling or by physical separation.</li>
</ul></div>
"""}
LESSON_61 = {"zh": r"""
<p class="lead">投机解码（speculative decoding）背后真正深刻的，并不是某一个新机制，而是一种贯穿了整套 SGLang 设计的<strong>通用思路</strong>：用便宜的、可并行的工作，去绕开一个串行瓶颈。本课我们不再讲新零件，而是<strong>退后一步</strong>，把第4、27、43、44、59 课串成一张网，看清这条反复出现的主线。当你能在不同的优化里认出同一个形状，你就从"记住一堆技巧"升级成了"掌握一种思维方式"——这正是一节"设计主题"课想给你的东西。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象你在批改一叠学生的连环计算题：每一步都依赖上一步的结果，所以你只能<strong>一题接一题</strong>地往下算，没法跳着做——这就是串行瓶颈。现在换个办法：让一个反应很快但偶尔出错的助教，<strong>先把后面 k 步的答案都猜出来</strong>（草稿很便宜）。然后你作为权威，<strong>一眼扫过去同时核对这 k 步</strong>——核对是可以并行的，因为答案都已经摆在面前了。凡是助教猜对的前缀，你直接采纳（<span class="mono">accept</span>）；第一处猜错的地方，你顺手写下正确答案当作<span class="mono">bonus token</span>。结果：你花一次"并行核对"的力气，就推进了好几步，而最终答案和你自己一题题硬算<strong>完全一样</strong>。这正是"用草稿换并行验证"。</p>
<p>注意这里的不对称：让助教从头算出答案很慢（要一步步推导），但你拿着现成答案去<strong>核对</strong>却很快——一眼就能看出对不对。生成是串行的、昂贵的，验证是并行的、便宜的。投机解码做的，就是把这份不对称变成现金：让便宜的一方去猜，让昂贵的一方只做它最擅长、又恰好闲着的并行核对。哪怕助教十次里错三次，你也只是丢掉那三次的后半段重来，从没冒过"批错卷子"的风险——因为<strong>最终拍板的永远是你</strong>。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>decode（逐字生成）天生是<strong>串行</strong>的：第 N+1 个 token 必须等第 N 个算完才能开始（第4课）。更糟的是，每一步只动一个 token，GPU 的算力几乎闲着，却把显存带宽吃满——它是<strong>带宽受限（bandwidth-bound）</strong>、<strong>延迟受限（latency-bound）</strong>的。投机解码花一份便宜的草稿，把"一个一个生成 k 个 token（k 个串行、带宽受限的步骤）"，变成"在<strong>一次并行 forward</strong> 里同时验证 k 个 token"（第43课）——而那次 forward 用的是本来就闲着的算力。EAGLE 把并行验证推得更远：用一棵 token <strong>树</strong>，一次验证许多条候选路径（第44课）。这跟 SGLang 别处的赢法是<strong>同一个形状</strong>：CUDA Graph 把固定序列录下来重放，省掉每步的 CPU 停顿（第27课）；重叠调度让 CPU 工作和 GPU forward 并行跑（第59课）。统一的原则只有一句：<strong>把延迟受限的串行步骤，变成吞吐受限的并行步骤——拿闲置算力去换被省下的串行时间。</strong></p>
<p>这里要分清两个常被混淆的概念：<strong>延迟（latency）</strong>是"一件事从头到尾要等多久"，<strong>吞吐（throughput）</strong>是"单位时间能完成多少件事"。串行步骤之所以痛，是因为它们卡在延迟上——每一步都要等上一步，时间被一格一格地串起来。把它们改造成并行步骤，本质上是把"按延迟收费"的活，搬到"按吞吐收费"的赛道上：GPU 本来就擅长一次做一大批，于是同样的总工作量，被摊平到一次并行操作里完成。投机解码、CUDA Graph、重叠调度，全都在做这件同样的事——只是分别作用在"生成 token""发射内核""CPU 与 GPU 协作"这三个不同的串行环节上。</p>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="串行 decode 需要 k 次目标前向，而草稿提议 k 个 token 后只需一次并行验证">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">串行 decode：k 次目标前向（慢）</text>
    <rect x="20" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="66" y="60" text-anchor="middle" style="font-size:12px">目标前向</text>
    <text x="66" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok1</text>
    <text x="126" y="67" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="140" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="186" y="60" text-anchor="middle" style="font-size:12px">目标前向</text>
    <text x="186" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok2</text>
    <text x="246" y="67" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="260" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="60" text-anchor="middle" style="font-size:12px">目标前向</text>
    <text x="306" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok3</text>
    <text x="366" y="67" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="380" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="426" y="60" text-anchor="middle" style="font-size:12px">目标前向</text>
    <text x="426" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok4</text>
    <text x="488" y="58" style="fill:var(--muted);font-size:12px">… 共 k 次串行</text>
    <text x="488" y="78" style="fill:var(--amber);font-size:12px;font-weight:700">每步都等上一步</text>
    <line x1="20" y1="108" x2="760" y2="108" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="20" y="138" style="font-weight:700;fill:var(--accent-ink)">草稿 → 一次并行验证（快）</text>
    <rect x="20" y="154" width="112" height="46" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="76" y="174" text-anchor="middle" style="font-size:12px">草稿模型</text>
    <text x="76" y="190" text-anchor="middle" style="fill:var(--muted);font-size:11px">便宜</text>
    <text x="146" y="180" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="160" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="175" y="180" text-anchor="middle" class="mono" style="font-size:11px">t1</text>
    <rect x="194" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="209" y="180" text-anchor="middle" class="mono" style="font-size:11px">t2</text>
    <rect x="228" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="243" y="180" text-anchor="middle" class="mono" style="font-size:11px">t3</text>
    <rect x="262" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="277" y="180" text-anchor="middle" class="mono" style="font-size:11px">t4</text>
    <text x="226" y="214" text-anchor="middle" style="fill:var(--muted);font-size:11px">k 个候选 token</text>
    <text x="306" y="180" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="320" y="154" width="244" height="46" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="442" y="174" text-anchor="middle" style="font-weight:700;fill:var(--teal)">目标：一次并行验证 k 个</text>
    <text x="442" y="190" text-anchor="middle" style="fill:var(--muted);font-size:11px">一次 forward，算力本就闲着</text>
    <text x="576" y="180" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="590" y="154" width="170" height="46" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="675" y="174" text-anchor="middle" style="font-size:12px;fill:var(--accent-ink)">一次前向</text>
    <text x="675" y="190" text-anchor="middle" style="font-size:12px;fill:var(--accent-ink)">接受 ≤ k 个 token</text>
  </svg>
  <div class="figcap"><b>图 1 · k 次串行 vs 一次并行验证</b> — 上：基线逐 token decode，k 个 token = k 次串行目标前向，每步都得等上一步。下：便宜草稿先提议 k 个候选，目标只做<strong>一次</strong>并行 forward 同时验证全部 k 个 → 一次昂贵前向最多吐出 k 个 token。对比 k 次前向 vs 1 次。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 280" role="img" aria-label="权衡：花一点便宜的并行草稿算力，去换省下的昂贵串行目标步">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">权衡：便宜的并行草稿算力 ↔ 省下的串行步数</text>
    <line x1="140" y1="96" x2="640" y2="96" style="stroke:var(--line);stroke-width:3"/>
    <line x1="390" y1="96" x2="390" y2="150" style="stroke:var(--line);stroke-width:2"/>
    <polygon points="390,150 370,192 410,192" style="fill:var(--faint);stroke:var(--line);stroke-width:1.5"/>
    <line x1="235" y1="96" x2="235" y2="106" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="545" y1="96" x2="545" y2="106" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="120" y="106" width="230" height="62" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="235" y="132" text-anchor="middle" style="font-weight:700;fill:var(--amber)">花：便宜的并行草稿算力</text>
    <text x="235" y="152" text-anchor="middle" style="fill:var(--muted);font-size:11px">草稿 + 一次验证一批候选</text>
    <rect x="430" y="106" width="230" height="62" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="545" y="132" text-anchor="middle" style="font-weight:700;fill:var(--teal)">省：昂贵的串行目标步</text>
    <text x="545" y="152" text-anchor="middle" style="fill:var(--muted);font-size:11px">原本 k 次 forward</text>
    <text x="390" y="228" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">命中率高 → 净赚；命中率低 → 接近打平</text>
    <text x="390" y="250" text-anchor="middle" style="fill:var(--muted);font-size:12px">用一点闲置算力，换被省下的串行延迟</text>
  </svg>
  <div class="figcap"><b>图 2 · 这笔交易的天平</b> — 你<strong>花</strong>掉一点额外算力（草稿模型 + 并行验证几个候选 token，便宜在它只是一次批量 forward），去<strong>省</strong>掉昂贵的串行目标步。命中率高时净赚，命中率低时接近打平（break-even）——本质是拿闲置算力换串行延迟。</div>
</div>

<h2>串行的墙：decode 为什么慢</h2>
<p>训练时，一整句话的所有 token 可以一次性喂进 GPU，并行算完——这是 GPU 最擅长的、<strong>compute-bound</strong> 的活。但 decode 不行：你还没生成第 N 个 token，就无从谈起第 N+1 个，因为后者要把前者当输入。于是生成 k 个 token 就是 k 次 forward，一次只前进一格。第4课讲过，这样的单步 forward 里，GPU 的浮点算力大量闲置，真正的瓶颈是把庞大的权重和 KV 缓存从显存搬进计算单元的<strong>带宽</strong>。换句话说，decode 把一台擅长"大批量并行计算"的机器，逼成了"一次只算一点点、还在等内存"的状态。这堵<strong>串行的墙</strong>，就是后面所有技巧想翻越的对象。</p>
<p>为什么会这样？关键在<strong>算术强度</strong>——也就是每从显存读一个字节的权重，能摊到多少次乘加运算。prefill 一次处理成百上千个 token，同一份权重被反复复用，算术强度高，算力被吃满；decode 每步只算一个 token，却同样要把整套权重、整份 KV 缓存重新读一遍，算术强度低到可怜，乘加单元大半时间在空转、只是干等数据从显存流过来。于是单纯往 decode 上堆更多算力几乎没用——瓶颈根本不在算力，而在带宽和"必须一步等一步"的串行性。理解了这一点，你就明白：要让 decode 变快，要么把更多 token 塞进同一步（增大 batch、复用那次权重搬运），要么想办法<strong>一次推进多个 token</strong>，把串行步数本身压下去。投机解码走的正是后一条路。</p>

<h2>草稿换验证：把串行变并行</h2>
<p>关键洞察是：<strong>验证比生成便宜得多，而且可以并行</strong>。如果我手里已经有了一串"候选的 k 个 token"，我可以把它们一次性拼成一个序列，做<strong>一次</strong> forward，让昂贵的目标模型在同一次前向里，同时算出"在每个位置上，目标模型自己会选什么"。凡是候选和目标选择一致的<strong>最长正确前缀</strong>，全部 <span class="mono">accept</span>；在第一处分歧，目标模型这一次 forward 顺带算出的那个正确 token 就是 <span class="mono">bonus token</span>，免费收下。</p>
<p>这里有个常被忽略的妙处：那次验证的 forward，本来就是 decode 每步都要做的、带宽受限的前向——权重和 KV 反正都要从显存搬一遍。既然算力本来闲着，那么"顺手再多算 k 个位置"几乎不增加成本，却把原本 k 次串行的搬运压缩成一次。也就是说，并行验证几乎是<strong>免费</strong>搭车的：付出的只是那个小草稿模型的一点点计算，换回的是被省下的、最昂贵的串行带宽时间。</p>
<p>那"候选的 k 个 token"从哪来？来自一个<strong>便宜的草稿模型</strong>（draft）——它小、快，逐个吐 token 也不心疼（第43课）。于是整笔交易变成：<strong>花一点便宜的草稿计算，换回一次本来就闲着的并行验证</strong>，而这次验证一口气推进了多个 token。草稿猜得越准，单次目标 forward 吐出的 token 越多，串行步数被压缩得越狠；草稿偶尔猜错也不要紧，因为最终一切以目标模型为准，错误的候选在验证时被丢弃，<strong>正确性由目标模型兜底</strong>。这就是为什么投机解码能在不牺牲质量的前提下提速：它从不赌博，只是把"先猜后验"这件事做得很便宜。</p>
<p>值得强调的是这笔交易的"杠杆比"：草稿模型通常只有目标模型的几十分之一大小，跑一次草稿的开销几乎可以忽略；而它换回的，是把若干次最昂贵的目标 forward 压成一次。命中率越高、草稿与目标越"合拍"，这个杠杆越划算。这也解释了一个看似反直觉的现象：投机解码在<strong>低并发、延迟敏感</strong>的场景收益最大——因为那时 GPU 的算力闲得最厉害，正好有大把空算力可以拿来做并行验证；而当 batch 已经很大、算力本就接近打满时，闲置算力不多，草稿换验证的便宜午餐也就没那么香了。看懂"拿闲置算力换串行时间"这句话，你就能预判它在什么负载下值得开。</p>

<h2>EAGLE 与 token 树：把并行验证推到极致</h2>
<p>既然验证可以并行，为什么只验证一条路径？EAGLE 让草稿模型产出一棵 token <strong>树</strong>：在每个分叉点保留几个高概率候选，于是一次 forward 里<strong>同时验证许多条候选路径</strong>（第44课）。用注意力掩码把这棵树编排进一次前向，目标模型仍然只跑一次，却覆盖了指数级的候选组合，最终沿着被接受的路径走到最长正确前缀。这把"并行验证"这件事的杠杆撑到了最大：闲置的算力越多，越值得一次多验证几条路。</p>
<p>换个角度看，token 树就是把"草稿换验证"这笔交易做得<strong>更划算</strong>：同样一次目标 forward，线性候选只能验证一条 k 长的猜测，而树状候选能在分叉处押多个赌注，命中率更高，平均每次验证 accept 下来的 token 也更多。代价是验证时要处理更多候选位置——但这恰恰用的是 decode 阶段闲置的算力，所以仍然是拿"本来就闲着的计算"去换"被省下的串行步骤"。EAGLE 之所以是这条主线的高光，正是因为它把"哪里有闲置算力，就在哪里多做并行验证"这个原则用到了极致。</p>
<p>这里也藏着一个工程上的平衡：树越大，一次验证能覆盖的候选越多，但编排树、算注意力掩码、处理更多位置的开销也越大。所以 EAGLE 并不是把树无限撑大，而是在"多押几注的收益"和"验证一次的成本"之间找一个甜点。这又一次印证了本课的主线——所有这些优化都不是免费的魔法，而是一笔笔需要权衡的<strong>交易</strong>：你总是在花一种便宜的资源，去换另一种昂贵资源的节省，关键是看清楚两边各值多少。掌握了这套"权衡的眼光"，你读任何一个新优化时，都会本能地去找它在拿什么换什么。</p>

<h2>同一个形状：一张跨课的网</h2>
<p>一旦你认得"把串行步骤变成并行步骤"这个形状，就会在 SGLang 里到处看到它。CUDA Graph（第27课）把一段固定的 kernel 序列录制下来，让 GPU 直接重放，省掉每一步内核启动时 CPU 的串行停顿——同样是"把本该一步步发指令的串行过程，压成一次性的并行重放"。重叠调度（第59课）让 CPU 的调度、采样工作和 GPU 的 forward<strong>并行</strong>跑，而不是一个等一个。它们和投机解码不是孤立的技巧，而是同一条主线的不同切面：<strong>哪里有串行的墙，就想办法用闲置资源做并行的事去翻越它</strong>。</p>
<p>把这几课并排看，结构惊人地一致：每一处都先识别出一个"必须一步接一步"的串行瓶颈（逐 token 生成、逐内核启动、CPU 与 GPU 互等），再找到一份"本来就闲着或很便宜"的资源（闲置算力、可重放的固定序列、可并行的另一颗芯片），最后用这份资源把串行过程<strong>批量地、并行地</strong>一次做完。投机解码花的是草稿算力，CUDA Graph 花的是一次性的录制开销，重叠调度花的是多一份调度逻辑——付出的都很便宜，换回的都是被省下的、最贵的串行延迟。</p>
<p>看懂这张网，你下次遇到瓶颈时就会自动问一句："这里能不能也草稿一把、再并行验证？"——这正是本课作为"设计主题"课的价值：它不教你新零件，而是给你一副能在不同场景里复用的<strong>眼镜</strong>。而且别忘了，投机解码的验证是<strong>无损（lossless）</strong>的：输出和老老实实逐字 decode 完全一致，它只省时间，不改结果。这一点至关重要——任何"用并行换串行"的技巧，只有在保证结果不变的前提下才值得用，否则就成了拿正确性换速度的劣质交易。投机解码、CUDA Graph、重叠调度的共同底线，都是<strong>结果严格不变，只优化怎么算</strong>。</p>

<div class="flow"><div class="node">串行 decode：token N+1 等 token N</div><div class="arrow">→</div><div class="node">k 个 token = k 次串行 forward（带宽受限，算力闲置）</div><div class="arrow">→</div><div class="node">便宜草稿先猜 k 个候选</div><div class="arrow">→</div><div class="node">目标模型一次并行 forward 验证全部 k 个</div></div>

<div class="cols"><div class="col"><strong>花什么：便宜的草稿计算</strong><br>跑一个小的草稿模型，逐个吐出 k 个候选 token。计算量小、用的是本来闲着的算力，是这笔交易里付出的成本。</div><div class="col"><strong>省什么：串行、带宽受限的步骤</strong><br>原本要 k 次各前进一格的串行 forward，被压成一次并行验证；省下的是延迟受限的串行时间，而结果保持无损。</div></div>

<table class="t"><tr><th>串行的墙</th><th>用什么并行技巧翻越</th><th>课</th></tr><tr><td>逐 token 生成，一步等一步</td><td>投机解码：草稿换一次并行验证</td><td>第43课</td></tr><tr><td>只验证一条路径太浪费</td><td>EAGLE token 树：一次验证多条候选路径</td><td>第44课</td></tr><tr><td>每步内核启动的 CPU 停顿</td><td>CUDA Graph：录制固定序列，GPU 重放</td><td>第27课</td></tr><tr><td>CPU 调度与 GPU forward 互相等待</td><td>重叠调度：CPU 工作与 GPU forward 并行</td><td>第59课</td></tr></table>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>propose 提议</h4><p>草稿模型提出 k 个候选 token —— 便宜、可逐个生成。</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>verify 验证</h4><p>目标模型一次并行 forward，同时验证全部 k 个位置。</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>accept 接受</h4><p>accept 最长正确前缀（与目标选择一致的部分）。</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>bonus 补偿</h4><p>在第一处分歧收下 bonus token —— 输出无损，与逐字 decode 完全一致。</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker</span><span class="ln">便宜草稿 + 昂贵目标：串行 decode 变一次并行验证</span></div><pre>class BaseSpecWorker(ABC):
    # the trade made concrete: a cheap draft + the expensive target
    @property
    @abstractmethod
    def draft_worker(self):     # cheap model: proposes k tokens (the speculation)
        ...
    @property
    @abstractmethod
    def target_worker(self):    # expensive model: VERIFIES all k in ONE parallel forward
        ...
    # net: many tokens per target forward when the draft is good — sequential
    # decode steps become one parallel verify
</pre></div>

<p>具体感受一下规模：在<strong>一次</strong>目标 forward 里验证 64 个候选 token，开销和一次普通 decode 步几乎一样（权重、KV 反正都要从显存搬一遍），却可能一口气接受好几个 token。而 <span class="mono">custom_mask</span>（树注意力掩码）保证树上每个分支只沿着<strong>自己那条路径</strong>做注意力、互不串味——于是同一次 forward 里，许多条候选路径被同时、却互不干扰地验证。这正是"草稿换并行验证"在 EAGLE 树形态下的落地：把一棵草稿树压进一次前向。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/eagle_info.py ::EagleVerifyInput</span><span class="ln">一次并行验证整棵草稿树所需的全部输入</span></div><pre>class EagleVerifyInput(SpecInput):
    # everything the target needs to verify a whole draft TREE in ONE
    # forward: the candidate tokens + a tree attention mask + the
    # retrieve indices that map tree nodes back to accepted paths.
    draft_token: torch.Tensor       # the proposed tree of tokens
    custom_mask: torch.Tensor       # tree attention mask
    positions: torch.Tensor
    retrieve_index: torch.Tensor    # tree -&gt; sequence bookkeeping
    spec_steps: int
    topk: int
    draft_token_num: int            # nodes verified this forward
</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li>decode 天生<strong>串行</strong>且<strong>带宽受限</strong>：token N+1 要等 token N，每步算力闲置、瓶颈在显存带宽（第4课）。</li>
<li>核心交易：花一份<strong>便宜的草稿</strong>，把"k 个串行 forward"换成"<strong>一次并行验证</strong> k 个 token"（第43课）。</li>
<li>EAGLE 用 token <strong>树</strong>把并行验证推到极致，一次验证多条候选路径（第44课）。</li>
<li>同一个形状：CUDA Graph 重放固定序列省 CPU 停顿（第27课）、重叠调度让 CPU 与 GPU 并行（第59课）——都是把<strong>串行步骤变并行</strong>。</li>
<li>验证是<strong>无损</strong>的：<span class="mono">accept</span> 最长正确前缀 + <span class="mono">bonus token</span>，输出与逐字 decode 完全一致。</li>
<li>统一原则：把延迟受限的串行步骤变成吞吐受限的并行步骤，<strong>拿闲置算力换被省下的串行时间</strong>。</li>
</ul>
</div>
""", "en": r"""
<p class="lead">What's truly deep about speculative decoding isn't one new mechanism — it's a <strong>general design move</strong> threaded through all of SGLang: spend cheap, parallelizable work to dodge a sequential bottleneck. This lesson adds no new part; instead we <strong>zoom out</strong> and weave lessons 4, 27, 43, 44, and 59 into a single web, so you can see the recurring through-line.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Imagine grading a stack of multi-step chained calculations: each step depends on the previous result, so you can only work <strong>one step after another</strong> — that's the sequential bottleneck. Now try a different trick: let a fast-but-occasionally-wrong assistant <strong>guess the next k steps' answers ahead of time</strong> (the draft is cheap). Then, as the authority, you <strong>scan and check all k at once</strong> — checking is parallel because every answer is already in front of you. Every correct prefix the assistant guessed, you <span class="mono">accept</span> directly; at the first mistake, you jot down the right answer yourself as the <span class="mono">bonus token</span>. The result: one "parallel check" advances you several steps, and the final answer is <strong>identical</strong> to grinding through each step yourself. That's "draft for parallel verify."</p>
<p>Notice the asymmetry: having the assistant derive each answer from scratch is slow (step-by-step reasoning), but checking a ready-made answer is fast — you can see at a glance whether it's right. Generating is serial and expensive; verifying is parallel and cheap. Speculative decoding turns that asymmetry into cash: let the cheap party guess, and let the expensive party do only what it's best at and happens to be idle for — parallel checking. Even if the assistant is wrong three times in ten, you just redo the tail of those three; you never risk "mis-grading the paper," because <strong>the final call is always yours</strong>.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Decode (token-by-token generation) is inherently <strong>sequential</strong>: token N+1 can't start until token N is done (Lesson 4). Worse, each step touches just one token, so the GPU's compute sits nearly idle while memory bandwidth is saturated — it's <strong>bandwidth-bound</strong> and <strong>latency-bound</strong>. Speculative decoding spends a cheap draft to turn "generate k tokens one-by-one (k sequential, bandwidth-bound steps)" into "VERIFY k tokens in <strong>one parallel forward</strong>" (Lesson 43) — a forward that uses the compute which was idle anyway. EAGLE pushes parallel-verify further with a token <strong>tree</strong>, verifying many candidate paths at once (Lesson 44). And it's the <strong>same shape</strong> as other SGLang wins: CUDA Graph records a fixed sequence so the GPU replays it without per-step CPU stalls (Lesson 27); the overlap scheduler runs CPU work in parallel with the GPU forward (Lesson 59). The unifying principle is one sentence: <strong>turn latency-bound serial steps into throughput-bound parallel ones — trade spare compute for saved sequential time.</strong></p>
<p>Two often-confused notions are worth separating here: <strong>latency</strong> is "how long one thing takes end to end," while <strong>throughput</strong> is "how many things finish per unit time." Serial steps hurt because they're stuck on latency — each waits on the previous, time strung together one slot at a time. Reshaping them into parallel steps essentially moves work that was "billed by latency" onto a track "billed by throughput": the GPU is built to do a big batch at once, so the same total work gets flattened into a single parallel operation. Speculative decoding, CUDA Graph, and the overlap scheduler all do this same thing — they just act on three different serial links: "generating tokens," "launching kernels," and "CPU-GPU cooperation."</p>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="serial decode needs k target forwards, while drafting k tokens needs only one parallel verify">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">Serial decode: k target forwards (slow)</text>
    <rect x="20" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="66" y="60" text-anchor="middle" style="font-size:12px">Target fwd</text>
    <text x="66" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok1</text>
    <text x="126" y="67" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="140" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="186" y="60" text-anchor="middle" style="font-size:12px">Target fwd</text>
    <text x="186" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok2</text>
    <text x="246" y="67" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="260" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="306" y="60" text-anchor="middle" style="font-size:12px">Target fwd</text>
    <text x="306" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok3</text>
    <text x="366" y="67" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="380" y="42" width="92" height="40" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="426" y="60" text-anchor="middle" style="font-size:12px">Target fwd</text>
    <text x="426" y="76" text-anchor="middle" class="mono" style="font-size:11px">→ tok4</text>
    <text x="488" y="58" style="fill:var(--muted);font-size:12px">… k serial forwards</text>
    <text x="488" y="78" style="fill:var(--amber);font-size:12px;font-weight:700">each waits on last</text>
    <line x1="20" y1="108" x2="760" y2="108" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="20" y="138" style="font-weight:700;fill:var(--accent-ink)">Draft → ONE parallel verify (fast)</text>
    <rect x="20" y="154" width="112" height="46" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="76" y="174" text-anchor="middle" style="font-size:12px">Draft model</text>
    <text x="76" y="190" text-anchor="middle" style="fill:var(--muted);font-size:11px">cheap</text>
    <text x="146" y="180" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="160" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="175" y="180" text-anchor="middle" class="mono" style="font-size:11px">t1</text>
    <rect x="194" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="209" y="180" text-anchor="middle" class="mono" style="font-size:11px">t2</text>
    <rect x="228" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="243" y="180" text-anchor="middle" class="mono" style="font-size:11px">t3</text>
    <rect x="262" y="160" width="30" height="32" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="277" y="180" text-anchor="middle" class="mono" style="font-size:11px">t4</text>
    <text x="226" y="214" text-anchor="middle" style="fill:var(--muted);font-size:11px">k candidate tokens</text>
    <text x="306" y="180" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="320" y="154" width="244" height="46" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="442" y="174" text-anchor="middle" style="font-weight:700;fill:var(--teal)">Target: verify all k at once</text>
    <text x="442" y="190" text-anchor="middle" style="fill:var(--muted);font-size:11px">one forward, compute idle anyway</text>
    <text x="576" y="180" text-anchor="middle" style="fill:var(--faint)">→</text>
    <rect x="590" y="154" width="170" height="46" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="675" y="174" text-anchor="middle" style="font-size:12px;fill:var(--accent-ink)">ONE forward</text>
    <text x="675" y="190" text-anchor="middle" style="font-size:12px;fill:var(--accent-ink)">accept ≤ k tokens</text>
  </svg>
  <div class="figcap"><b>Fig 1 · k serial forwards vs ONE parallel verify</b> — top: baseline token-by-token decode, k tokens = k serial target forwards, each waiting on the last. bottom: a cheap draft proposes k candidates, the target does <strong>one</strong> parallel forward verifying all k → up to k tokens from a single expensive forward. Contrast k forwards vs 1.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 280" role="img" aria-label="the trade: spend a little cheap parallel draft compute to save expensive serial target steps">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">The trade: cheap parallel draft compute ↔ saved serial steps</text>
    <line x1="140" y1="96" x2="640" y2="96" style="stroke:var(--line);stroke-width:3"/>
    <line x1="390" y1="96" x2="390" y2="150" style="stroke:var(--line);stroke-width:2"/>
    <polygon points="390,150 370,192 410,192" style="fill:var(--faint);stroke:var(--line);stroke-width:1.5"/>
    <line x1="235" y1="96" x2="235" y2="106" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="545" y1="96" x2="545" y2="106" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="120" y="106" width="230" height="62" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="235" y="132" text-anchor="middle" style="font-weight:700;fill:var(--amber)">Spend: cheap draft compute</text>
    <text x="235" y="152" text-anchor="middle" style="fill:var(--muted);font-size:11px">draft + verify a batch at once</text>
    <rect x="430" y="106" width="230" height="62" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="545" y="132" text-anchor="middle" style="font-weight:700;fill:var(--teal)">Save: serial target steps</text>
    <text x="545" y="152" text-anchor="middle" style="fill:var(--muted);font-size:11px">was k forwards</text>
    <text x="390" y="228" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">High acceptance → net win; low → near break-even</text>
    <text x="390" y="250" text-anchor="middle" style="fill:var(--muted);font-size:12px">spend spare compute, buy saved serial latency</text>
  </svg>
  <div class="figcap"><b>Fig 2 · the balance of this trade</b> — you <strong>spend</strong> a little extra compute (the draft model + verifying a few candidate tokens in parallel — cheap because it's one batched forward) to <strong>save</strong> expensive serial target steps. Net win when acceptance is high, near break-even when it's low — it's trading spare compute for saved serial latency.</div>
</div>

<h2>The serial wall: why decode is slow</h2>
<p>In training, all tokens of a sentence go into the GPU at once and compute in parallel — the GPU's favorite, <strong>compute-bound</strong> kind of work. Decode can't do that: you can't even speak of token N+1 before token N exists, because the latter is the former's input. So generating k tokens means k forwards, advancing one slot at a time. As Lesson 4 showed, in such a single-step forward the GPU's FLOPs sit mostly idle; the real bottleneck is moving huge weights and the KV cache from memory into the compute units — <strong>bandwidth</strong>. In other words, decode forces a machine built for "big parallel batches" into "compute a tiny bit, then wait on memory." This <strong>serial wall</strong> is exactly what every later trick tries to climb.</p>
<p>Why does this happen? It comes down to <strong>arithmetic intensity</strong> — how many multiply-adds you get per byte of weights read from memory. Prefill processes hundreds or thousands of tokens at once, reusing the same weights over and over: high intensity, math units saturated. Decode computes one token per step yet still re-reads the entire weight set and full KV cache, so intensity is pitifully low and the math units idle most of the time, just waiting for data to stream in from memory. That's why piling more compute onto decode barely helps — the bottleneck isn't compute at all, it's bandwidth and the "each step waits on the last" serial nature. Grasp this and you see the cure: either pack more tokens into the same step (bigger batch, amortizing that weight read) or find a way to <strong>advance multiple tokens at once</strong>, compressing the serial step count itself. Speculative decoding takes the latter road.</p>

<h2>Draft for verify: turning serial into parallel</h2>
<p>The key insight: <strong>verifying is far cheaper than generating, and it parallelizes</strong>. If I already hold a string of "k candidate tokens," I can stitch them into one sequence and do <strong>one</strong> forward, letting the expensive target model compute, in that same pass, "what the target itself would pick at each position." Every <strong>longest correct prefix</strong> where candidate and target agree gets <span class="mono">accept</span>ed; at the first disagreement, the correct token the target computed in that very forward is the <span class="mono">bonus token</span>, taken for free.</p>
<p>Here's an often-missed beauty: that verification forward is exactly the bandwidth-bound pass decode has to do every step anyway — the weights and KV must be hauled from memory regardless. Since the compute was idle, "checking k more positions on the side" adds almost no cost, yet collapses what were k serial hauls into one. In other words, the parallel verify rides along almost <strong>for free</strong>: the only price is a bit of compute from the small draft model, and the payoff is the saved, most-expensive serial bandwidth time.</p>
<p>Where do the "k candidate tokens" come from? From a <strong>cheap draft model</strong> — small and fast, so spitting out tokens one-by-one doesn't hurt (Lesson 43). The whole trade becomes: <strong>spend a little cheap draft compute, get back one parallel verify that was idle anyway</strong>, and that verify advances multiple tokens at once. The better the draft guesses, the more tokens per target forward, and the harder the serial step count is compressed. A wrong guess does no harm either, because everything is settled by the target model — bad candidates are discarded at verify time, and <strong>correctness is backstopped by the target</strong>. That's why speculative decoding speeds things up without sacrificing quality: it never gambles, it just makes "guess then verify" very cheap.</p>
<p>Worth stressing is the "leverage ratio" of this trade: the draft model is typically tens of times smaller than the target, so one draft pass costs almost nothing, yet it can collapse several of the most expensive target forwards into one. The higher the hit rate and the more "in tune" the draft is with the target, the better the leverage. This explains a seemingly counterintuitive fact: speculative decoding pays off most in <strong>low-concurrency, latency-sensitive</strong> settings — that's when the GPU's compute is idlest, with plenty of spare FLOPs to spend on parallel verify; whereas when the batch is already large and compute is near saturation, there's little idle compute left, and the cheap lunch of draft-for-verify isn't as tasty. Understand "trade spare compute for serial time" and you can predict which workloads make it worth enabling.</p>

<h2>EAGLE and the token tree: pushing parallel-verify to the limit</h2>
<p>If verification parallelizes, why verify only one path? EAGLE has the draft produce a token <strong>tree</strong>: at each branch point it keeps a few high-probability candidates, so one forward <strong>verifies many candidate paths at once</strong> (Lesson 44). An attention mask arranges the tree into a single pass; the target still runs only once yet covers exponentially many candidate combinations, walking the accepted path to the longest correct prefix. This maxes out the leverage of "parallel verify": the more idle compute there is, the more it pays to verify several paths in one shot.</p>
<p>Put another way, the token tree makes the "draft for verify" trade <strong>even more profitable</strong>: for the same single target forward, a linear candidate can only check one k-long guess, while a tree places multiple bets at the branch points, lands more hits, and accepts more tokens per verify on average. The cost is handling more candidate positions at verify time — but that's exactly the compute decode leaves idle, so it's still spending "compute that was idle anyway" to buy "saved serial steps." EAGLE is the high point of this through-line precisely because it pushes "wherever there's idle compute, do more parallel verification there" to its limit.</p>
<p>There's also an engineering balance hidden here: a bigger tree covers more candidates per verify, but arranging the tree, computing the attention mask, and handling more positions all cost more too. So EAGLE doesn't grow the tree without bound — it finds a sweet spot between "the payoff of placing more bets" and "the cost of one verify." This again confirms the lesson's through-line: none of these optimizations is free magic; each is a <strong>trade</strong> to be weighed — you're always spending one cheap resource to buy the saving of a more expensive one, and the key is seeing clearly what each side is worth. With this "eye for trade-offs," whenever you read a new optimization you'll instinctively look for what it's spending to buy what.</p>

<h2>The same shape: a cross-lesson web</h2>
<p>Once you recognize the shape "turn serial steps into parallel ones," you see it everywhere in SGLang. CUDA Graph (Lesson 27) records a fixed kernel sequence so the GPU just replays it, removing the CPU's serial stall at every kernel launch — again "compress a step-by-step serial dispatch into a one-shot parallel replay." The overlap scheduler (Lesson 59) runs CPU scheduling and sampling <strong>in parallel</strong> with the GPU forward instead of one waiting on the other. These aren't isolated tricks but different facets of one through-line: <strong>wherever there's a serial wall, use idle resources to do parallel work and climb it</strong>.</p>
<p>Lay these lessons side by side and the structure is strikingly uniform: each first identifies a "must go step-by-step" serial bottleneck (token-by-token generation, per-kernel launch, CPU and GPU waiting on each other), then finds a resource that is "idle anyway or very cheap" (idle compute, a replayable fixed sequence, another chip that can run in parallel), and finally uses that resource to do the serial process <strong>in one parallel batch</strong>. Speculative decoding spends draft compute, CUDA Graph spends a one-time recording cost, the overlap scheduler spends one extra copy of scheduling logic — each price is cheap, and each payoff is the saved, most-expensive serial latency.</p>
<p>See this web, and next time you hit a bottleneck you'll automatically ask: "could we draft here too, then verify in parallel?" — that's the value of this lesson as a "design theme" lesson: it teaches no new part, it hands you a pair of <strong>glasses</strong> you can reuse across scenarios. And remember, speculative decoding's verification is <strong>lossless</strong>: the output is identical to plain token-by-token decode; it saves time without changing the result. This matters enormously — any "parallel for serial" trick is only worth using if it guarantees the result is unchanged, otherwise it's a shoddy trade of correctness for speed. The shared bottom line of speculative decoding, CUDA Graph, and the overlap scheduler is the same: <strong>the result is strictly unchanged; only how it's computed is optimized</strong>.</p>

<div class="flow"><div class="node">Serial decode: token N+1 waits on token N</div><div class="arrow">→</div><div class="node">k tokens = k serial forwards (bandwidth-bound, compute idle)</div><div class="arrow">→</div><div class="node">Cheap draft guesses k candidates first</div><div class="arrow">→</div><div class="node">Target model verifies all k in one parallel forward</div></div>

<div class="cols"><div class="col"><strong>Spend: cheap draft compute</strong><br>Run a small draft model to emit k candidate tokens one-by-one. Little compute, using FLOPs that were idle anyway — the cost paid in this trade.</div><div class="col"><strong>Save: serial, bandwidth-bound steps</strong><br>k serial forwards that each advance one slot collapse into one parallel verify; what's saved is latency-bound serial time, and the result stays lossless.</div></div>

<table class="t"><tr><th>The serial wall</th><th>Parallel trick that climbs it</th><th>Lesson</th></tr><tr><td>Token-by-token generation, step waits on step</td><td>Speculative decoding: draft for one parallel verify</td><td>Lesson 43</td></tr><tr><td>Verifying only one path is wasteful</td><td>EAGLE token tree: verify many candidate paths at once</td><td>Lesson 44</td></tr><tr><td>CPU stall at every per-step kernel launch</td><td>CUDA Graph: record a fixed sequence, GPU replays</td><td>Lesson 27</td></tr><tr><td>CPU scheduling and GPU forward wait on each other</td><td>Overlap scheduler: CPU work parallel with GPU forward</td><td>Lesson 59</td></tr></table>

<div class="vflow"><div class="step"><div class="num">1</div><div class="sc"><h4>propose</h4><p>Draft model proposes k candidate tokens — cheap, generated one-by-one.</p></div></div><div class="step"><div class="num">2</div><div class="sc"><h4>verify</h4><p>Target model runs one parallel forward, verifying all k positions at once.</p></div></div><div class="step"><div class="num">3</div><div class="sc"><h4>accept</h4><p>accept the longest correct prefix (the part matching the target's choice).</p></div></div><div class="step"><div class="num">4</div><div class="sc"><h4>bonus</h4><p>Take the bonus token at the first disagreement — output is lossless, identical to plain decode.</p></div></div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/base_spec_worker.py ::BaseSpecWorker</span><span class="ln">a cheap draft + an expensive target: serial decode becomes one parallel verify</span></div><pre>class BaseSpecWorker(ABC):
    # the trade made concrete: a cheap draft + the expensive target
    @property
    @abstractmethod
    def draft_worker(self):     # cheap model: proposes k tokens (the speculation)
        ...
    @property
    @abstractmethod
    def target_worker(self):    # expensive model: VERIFIES all k in ONE parallel forward
        ...
    # net: many tokens per target forward when the draft is good — sequential
    # decode steps become one parallel verify
</pre></div>

<p>Feel the scale concretely: verifying 64 candidate tokens in <strong>one</strong> target forward costs about the same as a single normal decode step (the weights and KV must be hauled from memory either way), yet can accept several tokens at once. The <span class="mono">custom_mask</span> (tree attention mask) makes each branch of the tree attend only along <strong>its own path</strong>, never bleeding across branches — so in that one forward, many candidate paths are verified at the same time without interfering. This is "draft for parallel verify" in EAGLE's tree form: a whole draft tree squeezed into a single forward.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/speculative/eagle_info.py ::EagleVerifyInput</span><span class="ln">everything needed to verify the whole draft tree in one forward</span></div><pre>class EagleVerifyInput(SpecInput):
    # everything the target needs to verify a whole draft TREE in ONE
    # forward: the candidate tokens + a tree attention mask + the
    # retrieve indices that map tree nodes back to accepted paths.
    draft_token: torch.Tensor       # the proposed tree of tokens
    custom_mask: torch.Tensor       # tree attention mask
    positions: torch.Tensor
    retrieve_index: torch.Tensor    # tree -&gt; sequence bookkeeping
    spec_steps: int
    topk: int
    draft_token_num: int            # nodes verified this forward
</pre></div>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li>Decode is inherently <strong>serial</strong> and <strong>bandwidth-bound</strong>: token N+1 waits on token N, each step leaves compute idle, the bottleneck is memory bandwidth (Lesson 4).</li>
<li>The core trade: spend a <strong>cheap draft</strong> to swap "k serial forwards" for "<strong>one parallel verify</strong> of k tokens" (Lesson 43).</li>
<li>EAGLE pushes parallel-verify to the limit with a token <strong>tree</strong>, verifying many candidate paths at once (Lesson 44).</li>
<li>Same shape: CUDA Graph replays a fixed sequence to cut CPU stalls (Lesson 27); the overlap scheduler runs CPU parallel with GPU (Lesson 59) — all turning <strong>serial steps into parallel ones</strong>.</li>
<li>Verification is <strong>lossless</strong>: <span class="mono">accept</span> the longest correct prefix + a <span class="mono">bonus token</span>, output identical to plain decode.</li>
<li>Unifying principle: turn latency-bound serial steps into throughput-bound parallel ones — <strong>trade spare compute for saved sequential time</strong>.</li>
</ul>
</div>
"""}
LESSON_62 = {"zh": r"""
<p class="lead">如果你把前面几十课串起来看，会发现 SGLang 反复在做同一件事：把<strong>引擎核心</strong>（调度器、模型执行循环、内存池）保持稳定，而把<strong>每一条会变化的轴</strong>都推到一个<span class="mono">可插拔接口</span>背后——你面向一个抽象基类（ABC）编程，具体实现在<strong>部署时选定</strong>。这一课不讲新机制，而是把这条贯穿全书的暗线明明白白地挑明：<strong>一切皆可插拔</strong>。它是一节<strong>综合课</strong>，目标是把你已经学过的零散知识，收拢成一条可以反复套用的主线。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想想家里墙上的<strong>插座</strong>。墙后的电路（核心）是固定的，可墙上那个孔是一套<strong>标准接口</strong>。台灯、充电器、电饭煲——任何电器只要长着符合规范的插头，就能接进去工作，而你<strong>从不需要拆开墙、改动电路</strong>。SGLang 的引擎核心就是墙后电路，注意力后端、硬件平台、量化方法、投机算法……都是「电器」：它们各自实现同一套插头规范，<span class="mono">部署时</span>把谁插上，引擎就用谁。新硬件、新算法想加入，做法不是改电路，而是<strong>造一个符合规范的插头</strong>。正因为插座是标准的，电器厂商才能各自独立创新，而不必征得电网公司同意；同理，SGLang 的贡献者也能各自独立扩展，而不必触碰核心。</p></div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>这是一个<strong>综合课</strong>：不是介绍某个零件，而是把零件之间的<strong>共同形状</strong>提炼出来。前面每一课其实都是同一个模式的不同切面——<span class="mono">面向接口编程，部署时选实现</span>。一旦你看懂了这个形状，整个引擎就从「一大堆特性」坍缩成「一个反复出现的范型」，而你也立刻知道：<strong>为别处贡献代码</strong>，通常意味着「实现一个已存在的接口」，而非「重写核心」。这种「先认形状、再看细节」的视角，能让你在面对庞大代码库时不至于迷失：你总能先定位「这是哪一条缝」，再决定要不要深入它的具体实现。</p></div>

<h2>稳定的核心，可变的边缘</h2>
<p>把 SGLang 想象成一个同心圆。最里面是<strong>不轻易改动的核心</strong>：<span class="mono">Scheduler</span> 决定每一步处理哪些请求、<span class="mono">ModelRunner</span> 驱动模型前向的执行循环、<span class="mono">内存池</span>（KV 缓存与 token 槽位）管理显存。这三样是引擎的「重心」，它们的稳定，正是其他一切能自由替换的前提。如果核心本身也跟着场景频繁变动，那么任何一处替换都可能牵动全局，整台引擎就会失去可维护性。核心越稳定，边缘就越自由——这是一种看似矛盾、实则互相成全的关系。</p>
<p>而<strong>每一条会随场景变化的轴</strong>——你用哪种注意力内核、跑在什么芯片上、用什么量化、要不要投机解码、怎么切并行、输出要不要受语法约束、KV 怎么跨机传输——都不被硬编码进核心，而是被定义成一个<span class="mono">抽象基类</span>，由若干<strong>并列的具体实现</strong>去满足。核心只认接口，不认实现。这样，变化被关在边缘，核心得以保持简洁与可靠。换句话说，核心面对的永远是一组<strong>稳定的方法签名</strong>，至于这些方法在底层究竟调用了哪种内核、哪块芯片、哪套算法，对核心而言是<strong>透明的</strong>。核心不需要、也不应该知道这些细节。</p>
<p>这种「稳定核心 + 可变边缘」的分层，并非为了好看，而是有深刻的工程理由：它把<strong>正确性</strong>与<strong>多样性</strong>分开管理。核心负责保证调度、执行、显存这些「无论如何都不能错」的逻辑；边缘负责吸纳硬件、算法、格式上的种种差异。两者通过接口这道<strong>窄腰</strong>相连，差异不会渗进核心，核心的稳定也不会限制边缘的繁荣。这正是大型系统能同时做到「可靠」与「灵活」的经典手法——把不变的东西收紧，把善变的东西放开，让两者只在一条清晰的边界上交谈。</p>
<p>值得强调的是，「部署时选定」这四个字本身也很关键。实现的选择既不发生在编译期（那样就太死），也不发生在每一步推理的运行期（那样就太碎、太慢），而恰好落在<strong>部署这一刻</strong>：你启动服务时，根据手上的硬件、模型、负载，把每条缝各选一个实现，之后整个服务周期就稳定地用这套组合。这让选择既足够灵活（换一组参数就换一套实现），又足够稳定（一旦选定，热路径上不再有分支开销）。这正是「接口」与「部署时绑定」搭配使用的精髓。</p>

<h2>同一个模式，遍布整台引擎</h2>
<p>这个「面向接口编程、部署时选实现」的模式，不是某一处的巧合，而是<strong>反复出现的设计主题</strong>。我们在前面的课里逐一见过它的化身，现在把它们摆在一起，那条暗线就清晰可见了：</p>
<p>注意力后端（<strong>第33课</strong>）是最典型的一缝：模型代码只调用 <span class="mono">self.attn(...)</span>，至于背后是 FlashInfer、Triton 还是 FlashAttention，由后端实现去满足同一组方法；模型作者完全不必知道内核细节。硬件平台抽象（<strong>第42课</strong>）让<strong>一套引擎、多种芯片</strong>，平台差异被收进 Platform 接口，换芯片不是换引擎。量化方法（<strong>第35课</strong>）可按模型逐个替换，FP8、INT4 等只是同一接口的不同实现。投机算法家族 EAGLE/NGRAM/…（<strong>第43课</strong>）藏在一个枚举背后，调度路径只认「投机」这件事而不在意具体算法。四条并行轴 TP/PP/DP/EP（<strong>第46课</strong>）被同一个 <span class="mono">GroupCoordinator</span> 统一表达，通信细节被收进一个协调器。语法后端 xgrammar/outlines/llguidance（<strong>第48课</strong>）为结构化输出提供可换的实现，约束引擎与解码循环解耦。KV 传输连接器 Mooncake/NIXL（<strong>第45课</strong>）支撑 PD 分离，跨机搬运 KV 的方式可换而调度不变。甚至「写一个模型」（<strong>第26课</strong>）本身，也是面向稳定的层 API 编程，而非改动引擎——你拼装的是层，而层背后是哪种注意力、哪种量化，又是另一层可插拔。</p>
<p>把它们排在一起，你会看到惊人的一致：每一处都是「一个接口 + 多个实现 + 部署时选一个」。这就是综合课要点亮的那张<strong>跨课之网</strong>。它们看似各管各的领域——硬件、算法、格式、通信——但<strong>骨架是同一副</strong>。一旦你认得这副骨架，先前那些孤立的知识点就会彼此勾连，形成一张可以互相印证的网，而不再是一堆需要死记的特性。更妙的是，当你日后遇到一个全新的领域（比如一种尚未出现的加速器或解码策略），你几乎可以预判 SGLang 会怎么接纳它：定义一个新接口，提供若干实现，部署时选一个。范型本身就是预测力。</p>
<p>这也解释了为什么本课不引入任何新机制，却被放在全书接近收尾的位置。你必须先一课课见过这些缝的具体样子——见过注意力后端怎么调、见过平台怎么抽象、见过并行怎么协调——才能在此刻把它们抽象成同一个形状。综合课的价值不在于新增知识，而在于<strong>重组已有知识</strong>：把散落各处的点收成一条线，让你从「知道很多事」升级为「看懂一件事」。</p>

<h2>为什么这让 SGLang 可扩展</h2>
<p>这种结构带来的最大红利是<strong>可扩展性</strong>。当一块新硬件、一种新量化、一套新注意力内核出现时，你不需要理解、更不需要改动调度器与内存池——你只要<strong>实现那个已经存在的接口</strong>，再在部署时把它选上。核心代码一行不动，风险被牢牢圈在新实现内部。这意味着评审者只需检查「这个新实现是否正确满足接口」，而不必担心它会破坏调度或显存管理这些核心逻辑，评审的范围因此被大幅收窄。</p>
<p>这也重塑了「贡献」的含义：贡献通常不是重写引擎，而是<strong>沿着已有的缝</strong>补上一个新实现。接口是契约，核心是稳定的舞台，社区在边缘并行扩张。多个团队可以同时分别为不同的芯片、不同的量化、不同的投机算法贡献实现，彼此之间几乎不冲突，因为它们都只在自己的实现内部活动，碰不到共享的核心。这正是开源项目能高速演进、却不至于失控的关键：边界清晰，责任明确，改动局部化。</p>
<p>理解了这一点，你再回头看任何一课，都会自然地问：「这一处的缝在哪？接口是什么？实现有几个？部署时怎么选？」——这正是读懂 SGLang 的钥匙。它把一部看似庞杂的引擎，化简为<strong>一个反复使用的范型</strong>；学会这个范型，你不仅读懂了已有的代码，也获得了为它添砖加瓦的最短路径。从今往后，每当你打开一个陌生的模块，先别急着读实现，先找它的抽象基类——那就是这道缝的契约，也是你理解一切的起点。</p>

<h2>一道缝的契约长什么样</h2>
<p>把目光落到最典型的那道缝——注意力后端——你就能看清「接口即契约」到底意味着什么。本课要看的代码文件是 <span class="mono">base_attn_backend.py</span> 里的 <span class="mono">AttentionBackend</span>：它是一个抽象基类，只规定了几个<strong>必须被实现的方法</strong>，却不写任何具体的内核逻辑。<span class="mono">init_forward_metadata</span> 负责为这一批请求做好规划，<span class="mono">forward_extend</span> 负责 prefill 阶段的注意力，<span class="mono">forward_decode</span> 负责 decode 阶段的注意力。模型代码只认这三件事，从不直接触碰任何原始内核。</p>
<p>这份契约的妙处在于它的<strong>克制</strong>：它只声明「你必须能做什么」，而对「你具体怎么做」一字不提。于是 FlashInfer 可以用它高度优化的内核去实现这三个方法，Triton 可以用它的模板去实现，FlashAttention 也可以用自己的方式实现——三者对模型而言<strong>完全等价</strong>，因为它们满足的是同一份契约。部署时你通过一个开关选定其一，模型代码丝毫不变。这就是「程序面向接口，而非面向实现」最朴素也最有力的体现。你甚至可以在不同后端之间来回切换做对比测试，而无需改动哪怕一行模型代码——这种自由，正是接口赋予的。</p>
<p>反过来想，如果没有这道缝会怎样？模型代码里就得到处写「如果是 FlashInfer 就这样、如果是 Triton 就那样」的分支，每加一个后端，整片模型代码都要被翻一遍——这正是<strong>硬编码</strong>的噩梦。接口把这种「按实现分叉」的复杂度<strong>一次性收拢</strong>到一个抽象基类里，让核心和模型代码永远只面对一张干净的方法表。SGLang 之所以能在不同硬件、不同算法上保持同一套模型代码，靠的就是这种克制而稳定的契约。</p>

<p class="lead">所以「一切皆可插拔」不是一句口号，而是 SGLang 反复采用的<strong>工程纪律</strong>：凡是会变的，都先抽象成接口；凡是核心要依赖的，都只依赖接口而非实现。读完这一课，希望你以后翻开任何一个模块，第一反应都是去找它的抽象基类——找到了它，你就找到了这道缝的契约，也就找到了理解与扩展它的起点。</p>

<table class="t"><tr><th>可插拔的缝</th><th>接口 / 抽象</th><th>例子（实现）</th><th>课次</th></tr>
<tr><td>注意力后端</td><td><span class="mono">AttentionBackend</span></td><td>FlashInfer / Triton / FA</td><td>第33课</td></tr>
<tr><td>硬件平台</td><td><span class="mono">Platform</span></td><td>多种芯片，一套引擎</td><td>第42课</td></tr>
<tr><td>量化方法</td><td><span class="mono">QuantizeMethod</span></td><td>按模型逐个替换</td><td>第35课</td></tr>
<tr><td>投机算法</td><td>算法枚举</td><td>EAGLE / NGRAM / …</td><td>第43课</td></tr>
<tr><td>并行轴</td><td><span class="mono">GroupCoordinator</span></td><td>TP / PP / DP / EP</td><td>第46课</td></tr>
<tr><td>语法后端</td><td>Grammar backend</td><td>xgrammar / outlines / llguidance</td><td>第48课</td></tr>
<tr><td>KV 传输</td><td>KV connector</td><td>Mooncake / NIXL</td><td>第45课</td></tr>
<tr><td>写模型</td><td>稳定层 API</td><td>面向层 API 实现模型</td><td>第26课</td></tr></table>

<div class="layers"><div class="layer">🔌 注意力后端（第33课）</div><div class="layer">🔌 硬件平台（第42课）</div><div class="layer"><strong>稳定核心：Scheduler · 模型执行循环 · 内存池</strong></div><div class="layer">🔌 量化 / 投机 / 并行（第35/43/46课）</div><div class="layer">🔌 语法 / KV 传输 / 写模型（第48/45/26课）</div></div>

<div class="cols"><div class="col"><strong>硬编码（一个实现焊死在核心里）</strong><br>新硬件 = 改调度器；新算法 = 改执行循环；任何变化都要动核心，风险扩散、难以协作。</div><div class="col"><strong>面向接口编程（部署时替换）</strong><br>核心只认抽象基类；新实现满足接口即可接入；部署时选定具体实现，核心一行不改。</div></div>

<div class="flow"><div class="node">请求</div><div class="arrow">→</div><div class="node">稳定核心</div><div class="arrow">→</div><div class="node">缝处选中的实现</div><div class="arrow">→</div><div class="node">结果</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/base_attn_backend.py ::AttentionBackend</span><span class="ln">面向接口编程的范型：实现它就能接入，不碰引擎核心</span></div><pre>class AttentionBackend(ABC):
    # the archetypal pluggable seam: model code calls THIS, never a raw kernel
    def init_forward_metadata(self, forward_batch):
        ...   # plan whatever this backend needs for the batch (has a default)
    def forward_extend(self, q, k, v, layer, forward_batch):
        raise NotImplementedError   # prefill attention — each backend overrides
    def forward_decode(self, q, k, v, layer, forward_batch):
        raise NotImplementedError   # decode attention — each backend overrides
    # FlashInfer / Triton / FA each implement these; chosen at deploy time,
    # the model code never changes</pre></div>

<div class="fig">
  <svg viewBox="0 0 800 340" role="img" aria-label="稳定内核向外放射 8 条可插拔接缝：核心固定，边缘可换">
    <line x1="400" y1="172" x2="124" y2="37" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="400" y2="37" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="676" y2="37" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="80" y2="172" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="720" y2="172" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="124" y2="307" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="400" y2="307" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="676" y2="307" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <rect x="58" y="16" width="132" height="42" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="124" y="42" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">注意力后端</text>
    <rect x="334" y="16" width="132" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="400" y="42" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">量化</text>
    <rect x="610" y="16" width="132" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="676" y="42" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">MoE</text>
    <rect x="14" y="151" width="132" height="42" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="80" y="177" text-anchor="middle" style="fill:var(--purple);font-size:12px;font-weight:700">模型</text>
    <rect x="654" y="151" width="132" height="42" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="720" y="177" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">采样器</text>
    <rect x="58" y="286" width="132" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="124" y="312" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">KV 缓存</text>
    <rect x="334" y="286" width="132" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="400" y="312" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">平台·硬件</text>
    <rect x="610" y="286" width="132" height="42" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="676" y="312" text-anchor="middle" style="fill:var(--purple);font-size:12px;font-weight:700">语法后端</text>
    <rect x="300" y="140" width="200" height="64" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="400" y="166" text-anchor="middle" style="fill:var(--accent-ink);font-size:14px;font-weight:700">稳定内核</text>
    <text x="400" y="187" text-anchor="middle" style="fill:var(--muted);font-size:11px">调度器·内存·模型循环</text>
  </svg>
  <div class="figcap"><b>图 1 · 稳定内核 + 可插拔边缘</b> — 中央引擎核心（调度器·内存·模型循环）固定不变，向外放射 8 条接缝，每条都是一个接口、各有可替换实现；核心不动，边缘随部署替换。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="面向接口编程：编译期上层代码依赖抽象接口，部署期再按硬件选一个具体实现注入">
    <rect x="210" y="16" width="360" height="48" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="390" y="36" text-anchor="middle" style="fill:var(--ink);font-size:13px;font-weight:700">上层代码</text>
    <text x="390" y="54" text-anchor="middle" style="fill:var(--muted);font-size:11px">只调用接口，从不认实现</text>
    <line x1="390" y1="64" x2="390" y2="94" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="390,100 383,86 397,86" style="fill:var(--muted)"/>
    <text x="404" y="84" style="fill:var(--blue);font-size:11px">编译期 · 面向接口</text>
    <rect x="210" y="100" width="360" height="54" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="390" y="122" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:13px;font-weight:700">接口 SRTPlatform（抽象基类）</text>
    <text x="390" y="142" text-anchor="middle" style="fill:var(--muted);font-size:11px">稳定的方法签名</text>
    <line x1="390" y1="154" x2="390" y2="184" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="390,190 383,176 397,176" style="fill:var(--muted)"/>
    <text x="404" y="174" style="fill:var(--purple);font-size:11px">部署期 · 选实现并注入</text>
    <line x1="390" y1="190" x2="125" y2="198" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="390" y1="190" x2="655" y2="198" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <rect x="20" y="198" width="210" height="54" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="125" y="220" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">NVIDIA</text>
    <text x="125" y="239" text-anchor="middle" class="mono" style="fill:var(--teal);font-size:11px">→ FlashInfer</text>
    <rect x="285" y="198" width="210" height="54" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="390" y="220" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">AMD</text>
    <text x="390" y="239" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:11px">→ Triton</text>
    <rect x="550" y="198" width="210" height="54" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="655" y="220" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">新芯片</text>
    <text x="655" y="239" text-anchor="middle" style="fill:var(--amber);font-size:11px">= 新增一个子类</text>
  </svg>
  <div class="figcap"><b>图 2 · 面向接口编程 → 部署时选实现</b> — 上层代码在编译期只依赖抽象接口 <span class="mono">SRTPlatform</span>；启动部署时再按开关或检测到的硬件选一个具体实现并注入：NVIDIA 选 FlashInfer、AMD 选 Triton，上层代码一行不改。</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/platforms/interface.py ::SRTPlatform</span><span class="ln">每种硬件一个稳定接缝：基类定接口，子类覆写</span></div><pre>class SRTPlatform(DeviceMixin):
    # one stable seam per hardware platform. Upper layers program to
    # THIS interface; each chip subclasses and overrides the methods.
    supported_quantization: list[str] = []
    def apply_server_args_defaults(self, server_args): ...
    def get_default_attention_backend(self) -&gt; str:
        raise NotImplementedError      # each platform answers
    def get_graph_runner_cls(self) -&gt; type:
        raise NotImplementedError</pre></div>

<p>一个具体的画面：同一套引擎，在 <strong>NVIDIA</strong> 上 <span class="mono">get_default_attention_backend()</span> 返回 FlashInfer，在 <strong>AMD</strong> 上返回 Triton——上层调度与模型循环<strong>从不改动</strong>，因为它们只调用 <span class="mono">SRTPlatform</span> 这个接口，至于背后是哪块芯片、哪种内核，对它们透明。想新增一种加速器，做法不是在调度器里到处加 <span class="mono">if</span> 分支，而是<strong>新增一个 SRTPlatform 子类</strong>、覆写这几个方法，再在部署时选上它即可：改动被收在一个文件里，核心代码一行不动。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>一个模式，处处复现</strong>：面向接口（抽象基类）编程，具体实现在<span class="mono">部署时</span>选定；这是贯穿全书的同一副骨架。</li>
<li><strong>稳定核心</strong>：Scheduler、模型执行循环、内存池保持不变，是一切可替换的前提；核心越稳定，边缘就越自由。</li>
<li><strong>可变边缘</strong>：注意力后端（第33课）、平台（第42课）、量化（第35课）、投机（第43课）、并行（第46课）、语法（第48课）、KV 传输（第45课）、写模型（第26课）都是同一缝的化身。</li>
<li><strong>接口即契约</strong>：抽象基类只规定「必须能做什么」，不规定「怎么做」；多个实现因此对核心完全等价，可自由替换与对比。</li>
<li><strong>可扩展性的来源</strong>：贡献 = 实现一个已存在的接口，而非重写核心；改动局部化，评审范围收窄，多团队可并行推进而互不冲突。</li>
<li><strong>读懂全书的钥匙</strong>：看任何一课都先问「缝在哪、接口是什么、实现有几个、部署时怎么选」，打开陌生模块先找它的抽象基类，找到契约就找到了一切的起点。</li>
</ul></div>
""", "en": r"""
<p class="lead">String the past few dozen lessons together and you notice SGLang doing the same thing over and over: it keeps the <strong>engine core</strong> stable (the scheduler, the model-execution loop, the memory pools) and pushes <strong>every axis that can vary</strong> behind a <span class="mono">pluggable interface</span>—you program against an abstract base class (ABC), and a concrete implementation is <strong>selected at deploy time</strong>. This lesson introduces no new mechanism; it makes the book's hidden through-line explicit: <strong>everything is pluggable</strong>.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Think of a wall <strong>socket</strong>. The wiring behind the wall (the core) is fixed, but the hole on the wall is a <strong>standard interface</strong>. A lamp, a charger, a rice cooker—any appliance with a conforming plug can be connected and just works, and you <strong>never have to open the wall or rewire it</strong>. SGLang's engine core is the wiring; the attention backend, the hardware platform, quantization, speculative algorithms… are the "appliances": each implements the same plug spec, and <span class="mono">at deploy time</span> whichever you plug in is the one the engine uses. New hardware or a new algorithm joins not by rewiring, but by <strong>building a conforming plug</strong>.</p></div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>This is a <strong>synthesis lesson</strong>: instead of introducing a part, it distills the <strong>shared shape</strong> across parts. Every earlier lesson is a different facet of one pattern—<span class="mono">program to an interface, choose an implementation at deploy time</span>. Once you see the shape, the engine collapses from "a pile of features" into "one recurring archetype," and you immediately know that <strong>contributing elsewhere</strong> usually means "implement an existing interface," not "rewrite the core."</p></div>

<h2>A stable core, a variable edge</h2>
<p>Picture SGLang as concentric circles. At the center sits the <strong>core that rarely changes</strong>: the <span class="mono">Scheduler</span> decides which requests run each step, the <span class="mono">ModelRunner</span> drives the model's forward execution loop, and the <span class="mono">memory pools</span> (KV cache and token slots) manage device memory. These three are the engine's center of gravity, and their stability is precisely what lets everything else be swapped freely. If the core itself churned with every scenario, any single swap could ripple across the whole engine and maintainability would collapse.</p>
<p>Meanwhile <strong>every axis that varies with the scenario</strong>—which attention kernel you use, what chip you run on, what quantization, whether to speculate, how to split parallelism, whether output is grammar-constrained, how KV moves across machines—is not hard-coded into the core, but defined as an <span class="mono">abstract base class</span> satisfied by several <strong>parallel concrete implementations</strong>. The core knows only the interface, never the implementation. Variation is fenced into the edge, and the core stays simple and reliable. In other words, the core always faces a set of <strong>stable method signatures</strong>; which kernel, chip, or algorithm those methods ultimately invoke is <strong>transparent</strong> to it.</p>
<p>This "stable core + variable edge" layering isn't cosmetic; it has a deep engineering rationale: it manages <strong>correctness</strong> and <strong>diversity</strong> separately. The core guarantees the scheduling, execution, and memory logic that "must never be wrong"; the edge absorbs every difference in hardware, algorithm, and format. The two meet at a <strong>narrow waist</strong> of interfaces—differences never seep into the core, and the core's stability never caps the edge's flourishing.</p>
<p>It is worth stressing that "at deploy time" matters too. The choice of implementation happens neither at compile time (too rigid) nor on every inference step at runtime (too fragmented and slow), but precisely at <strong>the moment of deployment</strong>: when you launch the service, you pick one implementation per seam based on the hardware, model, and load at hand, and the whole service lifetime then uses that fixed combination. This makes the choice flexible enough (change a set of flags, change the implementations) yet stable enough (once chosen, no branch overhead remains on the hot path). That is the essence of pairing "interfaces" with "deploy-time binding."</p>

<h2>One pattern, all over the engine</h2>
<p>This "program to an interface, choose an implementation at deploy time" pattern is not a one-off coincidence but a <strong>recurring design theme</strong>. We met its incarnations across earlier lessons; line them up and the hidden through-line becomes clear:</p>
<p>The attention backend (<strong>Lesson 33</strong>) is the archetypal seam: model code only calls <span class="mono">self.attn(...)</span>, and whether FlashInfer, Triton, or FlashAttention sits behind it, the backend satisfies the same set of methods—the model author need not know any kernel detail. The hardware platform abstraction (<strong>Lesson 42</strong>) gives you <strong>one engine, many chips</strong>, with chip differences tucked into the Platform interface, so swapping chips is not swapping engines. Quantization methods (<strong>Lesson 35</strong>) are swappable per model; FP8, INT4 and the rest are just different implementations of one interface. The speculative-algorithm family EAGLE/NGRAM/… (<strong>Lesson 43</strong>) hides behind one enum, so the scheduling path only knows "speculation," not the specific algorithm. The four parallelism axes TP/PP/DP/EP (<strong>Lesson 46</strong>) are expressed uniformly by one <span class="mono">GroupCoordinator</span>, with communication detail folded into a single coordinator. The grammar backends xgrammar/outlines/llguidance (<strong>Lesson 48</strong>) provide swappable implementations for structured output, decoupling the constraint engine from the decode loop. The KV-transfer connectors Mooncake/NIXL (<strong>Lesson 45</strong>) power PD disaggregation, with the way KV moves across machines swappable while scheduling stays put. And even "writing a model" (<strong>Lesson 26</strong>) is itself programming against stable layer APIs rather than editing the engine—you assemble layers, and which attention or quantization those layers use is yet another pluggable layer.</p>
<p>Line them up and the consistency is striking: every one is "one interface + many implementations + pick one at deploy time." This is the <strong>cross-lesson web</strong> the synthesis lesson exists to light up. They seem to mind their own domains—hardware, algorithm, format, communication—but the <strong>skeleton is the same</strong>. Once you recognize that skeleton, formerly isolated facts hook into one another, forming a web that cross-validates itself rather than a pile of features to memorize. Better still, when you later meet a brand-new domain (say an accelerator or decoding strategy that does not yet exist), you can almost predict how SGLang will absorb it: define a new interface, supply a few implementations, pick one at deploy time. The archetype itself is predictive power.</p>
<p>This also explains why this lesson introduces no new mechanism yet sits near the book's end. You must first have seen each seam concretely, lesson by lesson—seen how the attention backend is called, how the platform is abstracted, how parallelism is coordinated—before you can abstract them here into one shape. A synthesis lesson's value is not in adding knowledge but in <strong>reorganizing existing knowledge</strong>: gathering scattered dots into one line, upgrading you from "knowing many things" to "understanding one thing."</p>

<h2>Why this makes SGLang extensible</h2>
<p>The biggest dividend of this structure is <strong>extensibility</strong>. When a new chip, a new quantization, or a new attention kernel appears, you need not understand—let alone modify—the scheduler and the memory pools; you simply <strong>implement the interface that already exists</strong> and select it at deploy time. Not a line of core code changes, and the risk is firmly contained inside the new implementation. Reviewers then only need to check "does this new impl correctly satisfy the interface," not worry that it breaks scheduling or memory management.</p>
<p>This also reshapes what "contributing" means: it usually is not rewriting the engine but <strong>filling a new implementation along an existing seam</strong>. The interface is the contract, the core is the stable stage, and the community expands in parallel at the edge. Many teams can contribute implementations for different chips, quantizations, and speculative algorithms at once with almost no conflict, because each operates only inside its own implementation and never touches the shared core. That is exactly what lets an open-source project evolve fast without spinning out of control.</p>
<p>Once you grasp this, you will read any lesson by asking: "Where is the seam here? What is the interface? How many implementations? How is one chosen at deploy time?"—which is exactly the key to reading SGLang. It reduces a seemingly sprawling engine to <strong>one repeatedly reused archetype</strong>; learn that archetype and you not only understand the existing code but also gain the shortest path to adding to it.</p>

<h2>What a seam's contract looks like</h2>
<p>Zoom into the archetypal seam—the attention backend—and you see exactly what "the interface is the contract" means. The code file this lesson examines is <span class="mono">AttentionBackend</span> in <span class="mono">base_attn_backend.py</span>: an abstract base class that prescribes only a few <strong>methods that must be implemented</strong>, while writing no concrete kernel logic at all. <span class="mono">init_forward_metadata</span> plans whatever the batch needs, <span class="mono">forward_extend</span> handles prefill attention, and <span class="mono">forward_decode</span> handles decode attention. Model code knows only these three things and never touches a raw kernel directly.</p>
<p>The beauty of this contract is its <strong>restraint</strong>: it declares only "what you must be able to do," and says nothing about "how you do it." So FlashInfer can implement the three methods with its highly optimized kernels, Triton with its templates, FlashAttention in its own way—and to the model the three are <strong>completely equivalent</strong>, because they satisfy the same contract. At deploy time you pick one via a switch, and the model code does not change in the slightest. This is the plainest and most powerful expression of "program to an interface, not an implementation." You can even flip back and forth between backends for comparison testing without changing a single line of model code—that freedom is exactly what the interface grants.</p>
<p>Imagine the opposite: without this seam, model code would be littered with "if FlashInfer do this, if Triton do that" branches, and every new backend would force a sweep through the whole model code—the very nightmare of <strong>hard-coding</strong>. The interface <strong>collapses</strong> that "fork by implementation" complexity once, into one abstract base class, so the core and model code forever face a single clean table of methods. The reason SGLang keeps one model codebase across different hardware and algorithms is exactly this restrained, stable contract.</p>

<p class="lead">So "everything is pluggable" is not a slogan but an <strong>engineering discipline</strong> SGLang applies again and again: whatever varies is abstracted into an interface first; whatever the core depends on depends only on the interface, never the implementation. After this lesson, may your first instinct on opening any module be to hunt for its abstract base class—find it and you have found the contract of the seam, and with it the starting point for understanding and extending it.</p>

<table class="t"><tr><th>Pluggable seam</th><th>Interface / abstraction</th><th>Example (implementations)</th><th>Lesson</th></tr>
<tr><td>Attention backend</td><td><span class="mono">AttentionBackend</span></td><td>FlashInfer / Triton / FA</td><td>Lesson 33</td></tr>
<tr><td>Hardware platform</td><td><span class="mono">Platform</span></td><td>Many chips, one engine</td><td>Lesson 42</td></tr>
<tr><td>Quantization method</td><td><span class="mono">QuantizeMethod</span></td><td>Swappable per model</td><td>Lesson 35</td></tr>
<tr><td>Speculative algorithm</td><td>Algorithm enum</td><td>EAGLE / NGRAM / …</td><td>Lesson 43</td></tr>
<tr><td>Parallelism axes</td><td><span class="mono">GroupCoordinator</span></td><td>TP / PP / DP / EP</td><td>Lesson 46</td></tr>
<tr><td>Grammar backend</td><td>Grammar backend</td><td>xgrammar / outlines / llguidance</td><td>Lesson 48</td></tr>
<tr><td>KV transfer</td><td>KV connector</td><td>Mooncake / NIXL</td><td>Lesson 45</td></tr>
<tr><td>Write a model</td><td>Stable layer APIs</td><td>Implement a model on layer APIs</td><td>Lesson 26</td></tr></table>

<div class="layers"><div class="layer">🔌 Attention backend (L33)</div><div class="layer">🔌 Hardware platform (L42)</div><div class="layer"><strong>Stable core: Scheduler · model loop · memory pools</strong></div><div class="layer">🔌 Quant / speculation / parallelism (L35/43/46)</div><div class="layer">🔌 Grammar / KV transfer / write a model (L48/45/26)</div></div>

<div class="cols"><div class="col"><strong>Hard-coded (one impl welded into the core)</strong><br>New hardware = edit the scheduler; new algorithm = edit the loop; any change touches the core, risk spreads, collaboration is hard.</div><div class="col"><strong>Program to an interface (swap at deploy)</strong><br>The core knows only the ABC; a new impl plugs in by satisfying the interface; the concrete impl is selected at deploy time, core unchanged.</div></div>

<div class="flow"><div class="node">request</div><div class="arrow">→</div><div class="node">stable core</div><div class="arrow">→</div><div class="node">selected impl at the seam</div><div class="arrow">→</div><div class="node">result</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/base_attn_backend.py ::AttentionBackend</span><span class="ln">the program-to-an-interface archetype: implement it to plug in, never touch the core</span></div><pre>class AttentionBackend(ABC):
    # the archetypal pluggable seam: model code calls THIS, never a raw kernel
    def init_forward_metadata(self, forward_batch):
        ...   # plan whatever this backend needs for the batch (has a default)
    def forward_extend(self, q, k, v, layer, forward_batch):
        raise NotImplementedError   # prefill attention — each backend overrides
    def forward_decode(self, q, k, v, layer, forward_batch):
        raise NotImplementedError   # decode attention — each backend overrides
    # FlashInfer / Triton / FA each implement these; chosen at deploy time,
    # the model code never changes</pre></div>

<div class="fig">
  <svg viewBox="0 0 800 340" role="img" aria-label="A stable core radiating 8 pluggable seams: the core stays fixed, the edges swap">
    <line x1="400" y1="172" x2="124" y2="37" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="400" y2="37" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="676" y2="37" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="80" y2="172" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="720" y2="172" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="124" y2="307" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="400" y2="307" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="400" y1="172" x2="676" y2="307" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <rect x="58" y="16" width="132" height="42" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="124" y="42" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">Attention</text>
    <rect x="334" y="16" width="132" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="400" y="42" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">Quant</text>
    <rect x="610" y="16" width="132" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="676" y="42" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">MoE</text>
    <rect x="14" y="151" width="132" height="42" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="80" y="177" text-anchor="middle" style="fill:var(--purple);font-size:12px;font-weight:700">Model</text>
    <rect x="654" y="151" width="132" height="42" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="720" y="177" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">Sampler</text>
    <rect x="58" y="286" width="132" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="124" y="312" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">KV cache</text>
    <rect x="334" y="286" width="132" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="400" y="312" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">Platform</text>
    <rect x="610" y="286" width="132" height="42" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="676" y="312" text-anchor="middle" style="fill:var(--purple);font-size:12px;font-weight:700">Grammar</text>
    <rect x="300" y="140" width="200" height="64" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="400" y="166" text-anchor="middle" style="fill:var(--accent-ink);font-size:14px;font-weight:700">Stable core</text>
    <text x="400" y="187" text-anchor="middle" style="fill:var(--muted);font-size:11px">scheduler · memory · loop</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Stable core + pluggable edges</b> — the central engine core (scheduler · memory · model loop) stays fixed and radiates 8 seams; each is an interface with swappable implementations. The core stays put; the edges swap at deploy time.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Program to the interface: upper code depends on an abstract interface at compile time, a concrete impl is picked by hardware and injected at deploy time">
    <rect x="210" y="16" width="360" height="48" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="390" y="36" text-anchor="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Upper code</text>
    <text x="390" y="54" text-anchor="middle" style="fill:var(--muted);font-size:11px">calls the interface, never the impl</text>
    <line x1="390" y1="64" x2="390" y2="94" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="390,100 383,86 397,86" style="fill:var(--muted)"/>
    <text x="404" y="84" style="fill:var(--blue);font-size:11px">compile-time · to interface</text>
    <rect x="210" y="100" width="360" height="54" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="390" y="122" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:13px;font-weight:700">interface SRTPlatform (ABC)</text>
    <text x="390" y="142" text-anchor="middle" style="fill:var(--muted);font-size:11px">stable method signatures</text>
    <line x1="390" y1="154" x2="390" y2="184" style="stroke:var(--muted);stroke-width:1.5"/>
    <polygon points="390,190 383,176 397,176" style="fill:var(--muted)"/>
    <text x="404" y="174" style="fill:var(--purple);font-size:11px">deploy-time · pick &amp; inject</text>
    <line x1="390" y1="190" x2="125" y2="198" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="390" y1="190" x2="655" y2="198" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <rect x="20" y="198" width="210" height="54" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="125" y="220" text-anchor="middle" style="fill:var(--teal);font-size:12px;font-weight:700">NVIDIA</text>
    <text x="125" y="239" text-anchor="middle" class="mono" style="fill:var(--teal);font-size:11px">→ FlashInfer</text>
    <rect x="285" y="198" width="210" height="54" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="390" y="220" text-anchor="middle" style="fill:var(--blue);font-size:12px;font-weight:700">AMD</text>
    <text x="390" y="239" text-anchor="middle" class="mono" style="fill:var(--blue);font-size:11px">→ Triton</text>
    <rect x="550" y="198" width="210" height="54" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="655" y="220" text-anchor="middle" style="fill:var(--amber);font-size:12px;font-weight:700">new chip</text>
    <text x="655" y="239" text-anchor="middle" style="fill:var(--amber);font-size:11px">= one new subclass</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Program to the interface → pick the impl at deploy</b> — upper code depends only on the abstract interface <span class="mono">SRTPlatform</span> at compile time; at startup a concrete impl is chosen by a flag or detected hardware and injected: FlashInfer on NVIDIA, Triton on AMD, with the upper code unchanged.</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/platforms/interface.py ::SRTPlatform</span><span class="ln">one stable seam per hardware: base defines the interface, subclasses override</span></div><pre>class SRTPlatform(DeviceMixin):
    # one stable seam per hardware platform. Upper layers program to
    # THIS interface; each chip subclasses and overrides the methods.
    supported_quantization: list[str] = []
    def apply_server_args_defaults(self, server_args): ...
    def get_default_attention_backend(self) -&gt; str:
        raise NotImplementedError      # each platform answers
    def get_graph_runner_cls(self) -&gt; type:
        raise NotImplementedError</pre></div>

<p>A concrete picture: the same engine has <span class="mono">get_default_attention_backend()</span> return FlashInfer on <strong>NVIDIA</strong> and Triton on <strong>AMD</strong>—the upper scheduler and model loop <strong>never change</strong>, because they only call the <span class="mono">SRTPlatform</span> interface; which chip or kernel sits behind it is transparent to them. To add a new accelerator you do not sprinkle <span class="mono">if</span> branches through the scheduler; you <strong>add one SRTPlatform subclass</strong>, override these few methods, and select it at deploy time: the change is contained in a single file and not a line of core code moves.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>One pattern, everywhere</strong>: program to an interface (an ABC); the concrete implementation is selected <span class="mono">at deploy time</span>—the same skeleton runs through the whole book.</li>
<li><strong>Stable core</strong>: the Scheduler, the model-execution loop, and the memory pools stay fixed—the precondition for everything else being swappable; the steadier the core, the freer the edge.</li>
<li><strong>Variable edge</strong>: attention backend (L33), platform (L42), quantization (L35), speculation (L43), parallelism (L46), grammar (L48), KV transfer (L45), writing a model (L26) are all incarnations of the same seam.</li>
<li><strong>The interface is the contract</strong>: an ABC prescribes only "what you must be able to do," not "how"; multiple implementations are therefore equivalent to the core.</li>
<li><strong>Source of extensibility</strong>: contributing = implementing an existing interface, not rewriting the core; changes stay local, review narrows, and teams work in parallel.</li>
<li><strong>The key to reading the whole book</strong>: for any lesson, first ask "where is the seam, what is the interface, how many implementations, how is one chosen at deploy time," and on opening any module, hunt for its abstract base class first.</li>
</ul></div>
"""}
LESSON_63 = {"zh": r"""
<p class="lead">这是全书的最后一课，也是把所有零件拧成一台机器的一课。如果你只想从这本指南里带走一句话，那就是这句：<strong>SGLang 的几乎每一个设计选择，都在为同一颗北极星服务——让 GPU 一直忙着做"有用的 token 工作"</strong>。换句话说，最大化每秒钟真正花在有用计算上的 FLOPs 与显存带宽。把前面 62 课连起来读，你会发现它们不是一堆互不相干的技巧，而是一台精心设计的<span class="mono">吞吐机器</span>的不同切面。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一座 24 小时不打烊的<strong>大型中央厨房</strong>。厨房里最贵的东西是那台巨大的烤炉（GPU 与它要反复读取的模型权重），开一次炉、热一次就要付出巨大的能量成本。一个聪明的厨房长会怎么做？他不会为一份订单单独开炉——他会<strong>攒一大盘菜一起烤</strong>（批处理）；他会把冷藏库切成整齐的<strong>标准格子</strong>，不浪费一寸空间（分页）；同一种酱汁<strong>熬好一次给多桌共用</strong>（前缀缓存）；他让备菜工和烤炉<strong>永远无缝衔接</strong>，炉子一刻不空（重叠调度）；他把"开炉门—放盘—关门"的固定动作<strong>编成一套肌肉记忆</strong>（CUDA Graph）；他磨快每一把刀，让切配本身快到极限（内核与融合）。所有这些都指向同一件事：<span class="mono">那台贵烤炉，一秒都不许闲着，而且每一秒都在烤真正要交付的菜</span>。这本指南讲的，就是这家厨房的全部经营哲学。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>从宏观看，一个推理系统的所有成本，最终都压在"GPU 时间"这一项上。GPU 时间被浪费，无非三种方式：<strong>空转</strong>（CPU 没喂上数据，炉子空着）、<strong>重复劳动</strong>（同样的前缀算了一遍又一遍）、<strong>装不下</strong>（显存碎片或模型太大，本可服务的请求被挡在门外）。SGLang 的整套架构，可以逐条映射成"消灭这三种浪费"的针对性手术：重叠调度与 CUDA Graph 消灭空转，前缀缓存与 RadixAttention 消灭重复，分页、量化与并行消灭装不下。当你建立起这个视角，本书任何一课你都能重新读成一句话——<span class="mono">"这一招，是在为吞吐做的某个深思熟虑的取舍"</span>。这就是这门课、也是这整本书，想交到你手上的那副眼镜。戴上它，你看 SGLang 就不再是一份功能清单，而是一台目标明确、各部件分工协作的精密机器。</p>
</div>

<h2>一颗北极星：每秒有用的 token 工作</h2>
<p>先把"吞吐"这个词说精确。它不是"每秒处理多少请求"那么含糊，而是<strong>每秒钟 GPU 花在有用计算上的比例×有效算力</strong>。一次解码步骤里，GPU 真正昂贵的动作是把整套模型权重从显存读进计算单元——这一步的成本几乎与"这一步服务了多少请求"无关。于是一个朴素却深刻的结论浮现：<span class="mono">如果你能让同一次权重读取同时服务 64 个请求而不是 1 个，你的有效吞吐就接近翻了 64 倍</span>，而几乎没有额外代价。这就是<strong>批处理（第5课）</strong>之所以是一切的地基的原因——它把最贵的那次"读权重"摊薄到了一大批请求头上。指南里后面所有的机制，本质上都在回答同一个问题的不同侧面：<em>如何让这个大批次尽可能大、尽可能满、尽可能少做无用功</em>。把这句话刻在脑子里，你会发现它像一根线，把后面十几课串成了一串：每一课要么在帮批次变大，要么在帮批次里少做白工，要么在帮 GPU 在两步之间不空转。换个说法，整本书都在围绕"那一大盘菜"打转——怎么攒得更满、怎么不重复备料、怎么让烤炉一刻不停。</p>
<p>理解了这一点，"显存"就不再只是一个容量限制，而是直接的吞吐杠杆。批次能开多大，取决于显存里能同时塞下多少请求的 KV 缓存。<strong>分页（第6课、第30课）</strong>把 KV 切成固定大小的页，消灭了碎片，于是同样的显存能装下更多请求，批次就能更大。<strong>量化与并行（第35课、第46课）</strong>则从另一头出手：把模型本身压小、或摊到多张卡上，给批次腾出更多空间。它们看起来在解决不同的问题，但收口都在同一处——<span class="mono">让那个能被一次权重读取摊薄的批次，更大</span>。</p>
<p>值得多停留一秒，去体会这个视角的统一之美。在没有这副眼镜之前，你可能把分页当成"内存管理的技巧"，把量化当成"压缩模型的技巧"，把张量并行当成"分布式系统的技巧"——它们被归在不同的章节、用着不同的术语、由不同的人维护。可一旦你认定北极星是"让被摊薄的批次更大"，这三者立刻显形为同一个动作的三种姿势：都是在为那台贵烤炉准备<strong>更满的一盘菜</strong>。这正是综合课的价值——它不教你任何新机制，而是把你已经学过的所有机制，重新排进同一根因果链里。<span class="mono">当解释一件事的理由收敛到同一句话，你就知道自己真正理解了这套系统的设计意图</span>，而不只是记住了它的零件清单。</p>

<h2>消灭重复，消灭空转</h2>
<p>把批次撑大只是第一招。第二招是<strong>别让 GPU 做白工</strong>。很多请求共享相同的前缀——同一段系统提示、同一份少样本示例、同一轮对话历史。如果每个请求都把这段共享前缀重新算一遍，那是巨大的浪费。<strong>前缀缓存 / RadixAttention（第7课、第29课）</strong>用一棵基数树记住已经算过的 KV，让共享前缀的请求直接复用，跳过冗余计算。这一招省下来的，正是宝贵的 GPU 时间，等价于凭空把吞吐又抬高了一截。</p>
<p>第三招针对"空转"。哪怕批次很大、计算也不重复，如果 CPU 在准备下一批数据时 GPU 只能干等，炉子就空了。<strong>重叠调度器（第21课、第59课）</strong>的全部使命就是消灭这种 CPU 气泡：当 GPU 正在算第 N 步时，CPU 已经在为第 N+1 步排队、采样、组批，两者重叠流水，让 GPU 几乎永不停步。与之配合的是 <strong>CUDA Graph（第27课）</strong>，它把每一步固定的内核启动序列录制成一张图，一次回放，砍掉了逐次启动的 CPU 开销——在小批次、解码密集的场景下，这部分省下的开销相当可观。最后，<strong>内核与融合（第38课、第41课）</strong>把多个算子合并、贴着显存带宽的极限去写，让"真正算的那一段"本身也快到尽头。每一层都在回答同一个问题：<span class="mono">GPU 的每一秒，能不能更满地用在有用的工作上？</span></p>
<p>把这三招放在一起看，会发现它们其实在抢同一秒钟的不同部分。一步解码的时间里，既有"读权重做计算"的有效部分，也有"等数据、启内核、算冗余前缀"的浪费部分。批处理放大了有效部分的产出，前缀缓存切掉了冗余前缀那一块，重叠调度与 CUDA Graph 压扁了等待与启动那两块，内核融合则把有效部分本身再拧紧一圈。于是同一秒钟里，"有用工作"占的比例越来越高，被浪费的边角越来越少——<span class="mono">这就是吞吐机器一寸一寸抠出来的</span>。没有哪一招是银弹，但它们叠在一起，就把那台贵烤炉的利用率推到了逼近物理极限的地步。这也解释了为什么去掉任何一环，整机的吞吐都会明显塌一块：它们不是冗余的保险，而是同一目标上互补的合力。</p>

<h2>调度：把以上全部拧成一个决策</h2>
<p>所有这些杠杆，最终要在一个地方汇合并被同时权衡——<strong>调度</strong>。"下一步该让谁跑"这个看似简单的问题，其实同时背负着两个吞吐目标：一是<strong>把批次塞到最大</strong>（摊薄权重读取），二是<strong>优先挑前缀命中率高的请求</strong>（复用 KV、跳过重复计算）。SGLang 用 <span class="mono">SchedulePolicy</span> 与 <span class="mono">CacheAwarePolicy</span> 把这两个目标编码进了等待队列的优先级计算里——其中 LPM（最长前缀匹配）会把共享前缀的请求聚到一起，让它们既能组成大批、又能集中命中缓存。这不是两个互相打架的目标，而是同一颗北极星的两个分量：<span class="mono">大批 + 高前缀命中，二者都通向吞吐</span>。</p>
<p>但这里有一个必须诚实面对的微妙之处：一味追求系统吞吐，可能会伤害<strong>单个用户</strong>的体验。一个超长 prompt 如果一次性塞进预填充，会把解码批次顶到一边，让其他用户的 token 卡顿。于是指南里那些看似"为延迟"的机制登场了：<strong>分块预填充（第22课）</strong>把长 prompt 切成小块、与解码交错，避免一个大请求霸占整步；<strong>投机解码（第43课）</strong>用小模型猜、大模型验，在不增加权重读取次数的前提下一步吐出多个 token。它们的精神不是"牺牲吞吐换延迟"，而是<span class="mono">在最大化系统的同时，不牺牲那个具体的人</span>——这正是一台成熟的吞吐机器与一台粗暴的吞吐机器的区别。</p>
<p>把这层张力想透，你就握住了阅读 SGLang 的最后一把钥匙。纯粹的吞吐最大化，会把系统逼成一台只看总账、不看个体的机器：它愿意让某个倒霉用户等上几秒，只要整体每秒 token 数更漂亮。但真实服务里，没有人愿意当那个被牺牲的个体。于是成熟的设计在北极星之外，又加了一条约束线——<span class="mono">在不显著拖慢任何单个用户的前提下，把系统推到最大</span>。分块预填充与投机解码正是这条约束线的化身：前者保证长请求不会一口气堵死解码流，后者让单用户的 token 更快吐出而几乎不增加系统负担。读到这里，你应该已经能感到，整本指南其实在反复演示同一种平衡术：既贪婪地榨取硬件，又克制地守护体验，二者在调度器里被同时计入同一笔账。</p>

<table class="t"><tr><th>设计选择</th><th>它买来的吞吐</th><th>课次</th></tr>
<tr><td>批处理</td><td>把一次昂贵的权重读取摊薄到一大批请求上</td><td>第5课</td></tr>
<tr><td>分页（Paged KV）</td><td>消灭 KV 碎片，同样显存装下更多请求 → 批次更大</td><td>第6、30课</td></tr>
<tr><td>前缀缓存 / RadixAttention</td><td>复用共享前缀的 KV，跳过冗余计算</td><td>第7、29课</td></tr>
<tr><td>重叠调度器</td><td>消灭 CPU 气泡，GPU 永不空转</td><td>第21、59课</td></tr>
<tr><td>CUDA Graph</td><td>砍掉逐步内核启动开销</td><td>第27课</td></tr>
<tr><td>内核与融合</td><td>贴着显存带宽极限，让真正的计算快到尽头</td><td>第38、41课</td></tr>
<tr><td>量化与并行</td><td>把更大的模型塞进硬件，给批次腾空间</td><td>第35、46课</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="吞吐杠杆地图：每个设计带来的吞吐——连续批处理让 GPU 不空转、前缀缓存跳过重算、分块预填充不卡解码、CUDA 图与融合杀启动开销、重叠调度隐藏 CPU、投机解码每次前向多吐、量化装更多更快">
    <text x="20" y="30" style="font-weight:700;fill:var(--accent-ink)">吞吐杠杆地图：每个设计 → 它买来的吞吐</text>
    <rect x="20" y="48" width="230" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="135" y="68" text-anchor="middle" style="fill:var(--ink);font-weight:600">连续批处理</text>
    <text x="260" y="68" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="68" style="fill:var(--muted)">GPU 不空转</text>
    <rect x="20" y="86" width="230" height="30" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="135" y="106" text-anchor="middle" style="fill:var(--ink);font-weight:600">前缀缓存</text>
    <text x="260" y="106" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="106" style="fill:var(--muted)">跳过重算</text>
    <rect x="20" y="124" width="230" height="30" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="135" y="144" text-anchor="middle" style="fill:var(--ink);font-weight:600">分块预填充</text>
    <text x="260" y="144" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="144" style="fill:var(--muted)">不卡解码</text>
    <rect x="20" y="162" width="230" height="30" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="135" y="182" text-anchor="middle" style="fill:var(--ink);font-weight:600">CUDA 图 + 融合</text>
    <text x="260" y="182" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="182" style="fill:var(--muted)">杀启动开销</text>
    <rect x="20" y="200" width="230" height="30" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="135" y="220" text-anchor="middle" style="fill:var(--ink);font-weight:600">重叠调度</text>
    <text x="260" y="220" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="220" style="fill:var(--muted)">隐藏 CPU</text>
    <rect x="20" y="238" width="230" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="135" y="258" text-anchor="middle" style="fill:var(--ink);font-weight:600">投机解码</text>
    <text x="260" y="258" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="258" style="fill:var(--muted)">每次前向多吐</text>
    <rect x="20" y="276" width="230" height="30" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="135" y="296" text-anchor="middle" style="fill:var(--ink);font-weight:600">量化</text>
    <text x="260" y="296" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="296" style="fill:var(--muted)">装更多 / 更快</text>
  </svg>
  <div class="figcap"><b>图 1 · 吞吐杠杆地图</b> — 把全书的大杠杆排成一张表：每一行是"杠杆 → 它买来的吞吐"。连续批处理填满 GPU、前缀缓存跳过重算、分块预填充不卡解码、CUDA 图与融合杀掉启动开销、重叠调度隐藏 CPU、投机解码每次前向多吐 token、量化装得更多也更快。</div>
</div>

<div class="flow"><div class="node">请求到达<br><span class="mono">攒批</span></div><div class="arrow">→</div><div class="node">调度排队<br><span class="mono">大批+前缀命中</span></div><div class="arrow">→</div><div class="node">前缀缓存<br><span class="mono">跳过重复</span></div><div class="arrow">→</div><div class="node">预填充<br><span class="mono">分块不霸占</span></div><div class="arrow">→</div><div class="node">解码循环<br><span class="mono">重叠+Graph</span></div><div class="arrow">→</div><div class="node">内核执行<br><span class="mono">贴带宽极限</span></div><div class="arrow">→</div><div class="node">返回 token<br><span class="mono">炉子不空</span></div></div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="完整请求路径，每段标注它的吞吐杠杆：分词、调度（连续批处理）、前缀（基数缓存）、前向（CUDA 图与融合）、采样、反分词、流式（重叠）">
    <text x="20" y="28" style="font-weight:700;fill:var(--accent-ink)">完整请求路径：每段标注它的吞吐杠杆</text>
    <rect x="12" y="70" width="98" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="61" y="97" text-anchor="middle" style="fill:var(--ink)">分词</text>
    <text x="61" y="138" text-anchor="middle" style="fill:var(--muted);font-size:11px">文本→id</text>
    <text x="116" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="124" y="70" width="98" height="44" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="173" y="97" text-anchor="middle" style="fill:var(--ink)">调度</text>
    <text x="173" y="135" text-anchor="middle" style="fill:var(--blue);font-size:11px">连续</text>
    <text x="173" y="149" text-anchor="middle" style="fill:var(--blue);font-size:11px">批处理</text>
    <text x="228" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="236" y="70" width="98" height="44" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="285" y="97" text-anchor="middle" style="fill:var(--ink)">前缀</text>
    <text x="285" y="135" text-anchor="middle" style="fill:var(--teal);font-size:11px">基数</text>
    <text x="285" y="149" text-anchor="middle" style="fill:var(--teal);font-size:11px">缓存</text>
    <text x="340" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="348" y="70" width="98" height="44" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="397" y="97" text-anchor="middle" style="fill:var(--ink)">前向</text>
    <text x="397" y="135" text-anchor="middle" style="fill:var(--purple);font-size:11px">CUDA 图</text>
    <text x="397" y="149" text-anchor="middle" style="fill:var(--purple);font-size:11px">+ 融合</text>
    <text x="452" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="460" y="70" width="98" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="509" y="97" text-anchor="middle" style="fill:var(--ink)">采样</text>
    <text x="509" y="138" text-anchor="middle" style="fill:var(--muted);font-size:11px">选下一 token</text>
    <text x="564" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="572" y="70" width="98" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="621" y="97" text-anchor="middle" style="fill:var(--ink)">反分词</text>
    <text x="621" y="138" text-anchor="middle" style="fill:var(--muted);font-size:11px">id→文本</text>
    <text x="676" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="684" y="70" width="98" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="733" y="97" text-anchor="middle" style="fill:var(--ink)">流式</text>
    <text x="733" y="142" text-anchor="middle" style="fill:var(--amber);font-size:11px">重叠</text>
    <text x="20" y="220" style="fill:var(--muted);font-size:12px">「一次请求的一生」——这次重讲成一个吞吐故事：每一段都挂着它对应的那颗杠杆。</text>
  </svg>
  <div class="figcap"><b>图 2 · 请求路径上的吞吐杠杆</b> — 把"一次请求的一生"重讲成吞吐故事：端到端路径（分词 → 调度 → 前缀 → 前向 → 采样 → 反分词 → 流式）上，给每一段挂一个小标签，标出该处生效的吞吐杠杆——调度处是连续批处理、前缀处是基数缓存、前向处是 CUDA 图与融合、流式处是重叠调度。</div>
</div>

<div class="cols"><div class="col"><strong>吞吐杠杆（把系统推到最大）</strong><ul><li>批处理：摊薄权重读取</li><li>分页：装下更多请求</li><li>前缀缓存：跳过重复计算</li><li>重叠调度：消灭空转</li><li>CUDA Graph：砍启动开销</li><li>内核与融合：榨干带宽</li></ul></div><div class="col"><strong>延迟护栏（不牺牲那个人）</strong><ul><li>分块预填充（第22课）：长 prompt 切块交错，不霸占整步</li><li>投机解码（第43课）：一步多吐 token，不增加权重读取</li><li>本质：在最大化系统的同时守住单用户体验</li></ul></div></div>

<div class="layers"><div class="layer">最外层：调度与批处理——决定"一次喂多少、喂谁"</div><div class="layer">中间层：缓存与分页——决定"装得下多少、能复用多少"</div><div class="layer">内核层：融合与 CUDA Graph——决定"每一步算得多快、启动多省"</div><div class="layer">最内核心：那台不许闲着的 GPU——每一秒都在做有用的 token 工作</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy</span><span class="ln">谁先跑，挑的是"最大化有用 GPU 工作"：大批 + 高前缀命中</span></div><pre>class CacheAwarePolicy(Enum):
    LPM = "lpm"          # longest-prefix-match: group requests that share a prefix
    # ... other cache-aware policies

class SchedulePolicy:
    # who runs next is chosen to MAXIMIZE useful GPU work:
    # pack the biggest batch (amortize the weight read) AND prefer high prefix-hit
    # (reuse KV, skip redundant compute) — both serve throughput
    def calc_priority(self, waiting_queue):
        ...
</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.get_next_batch_to_run</span><span class="ln">连续批处理的核心杠杆：每步决定跑哪个批</span></div><pre>def get_next_batch_to_run(self) -&gt; Optional[ScheduleBatch]:
    # the continuous-batching lever, run EVERY step: merge any finished
    # prefill into the running batch, then decide what to run next —
    # admit a new prefill batch (more work) or keep decoding.
    merge_finished_prefill_into(self.running_batch)
    new_batch = self.get_new_batch_prefill()    # admit within token budget
    if new_batch is not None:
        return new_batch                         # run prefill this step
    return self.running_batch                    # else keep decoding
</pre></div>

<div class="card"><div class="tag">🔎 具体例子</div>
<p>因为 <span class="mono">get_next_batch_to_run</span> 每一步都会跑一次，所以当批次里某个请求一完成、它腾出的槽位会<strong>立刻</strong>被一个等待中的请求填上（这就是连续批处理）——根本不用先把整批排空、再开新批。正是这<strong>同一颗杠杆</strong>，让那台贵烤炉（GPU）始终满载：每一步都在"合并已完成的预填充 → 尽量塞进新预填充 → 否则继续解码"之间做出最划算的决定。</p>
</div>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li><strong>一颗北极星</strong>：SGLang 几乎每个设计都在让 GPU 一直忙着做有用的 token 工作——最大化每秒有用的 FLOPs 与带宽。</li>
<li><strong>批处理是地基（第5课）</strong>：把最贵的"读权重"摊薄到一大批请求上，是吞吐的起点。</li>
<li><strong>显存即吞吐</strong>：分页（第6、30课）、量化与并行（第35、46课）都在为"更大的批次"腾空间。</li>
<li><strong>消灭重复（第7、29课）</strong>：前缀缓存 / RadixAttention 复用共享前缀，跳过冗余计算。</li>
<li><strong>消灭空转（第21、27、59课）</strong>：重叠调度去掉 CPU 气泡，CUDA Graph 去掉启动开销。</li>
<li><strong>榨干带宽（第38、41课）</strong>：内核与融合让真正的计算贴着硬件极限跑。</li>
<li><strong>调度收口</strong>：<span class="mono">SchedulePolicy</span> 与 <span class="mono">CacheAwarePolicy</span> 同时追求"大批"与"高前缀命中"，二者都服务吞吐。</li>
<li><strong>延迟护栏</strong>：分块预填充（第22课）与投机解码（第43课）保证在最大化系统时不牺牲单个用户。</li>
</ul>
</div>

<div class="card"><div class="tag">🏁 全书收束</div>
<p>用一段话回望全书：从<strong>第3课"一次请求的一生"</strong>那张地图出发，我们一路走过批处理、分页、前缀缓存、调度、内核、量化、并行、投机解码……直到此刻。现在你已经能把 SGLang 的每一个部件，都重新读成一个<span class="mono">为吞吐（或在不牺牲单个用户的前提下）做出的深思熟虑的选择</span>。这副"吞吐眼镜"，正是这整本指南从第一课起就在为你打磨的东西——戴上它，再去翻任何一段源码，你看到的将不再是孤立的技巧，而是同一颗北极星下的协同。感谢你同行这 63 课，愿你带着这副眼镜，去读懂、也去改进下一个伟大的推理系统。山高水长，后会有期，江湖再见。🚀</p>
</div>
""", "en": r"""
<p class="lead">This is the last lesson of the whole guide, and the one that tightens every part into a single machine. If you take away just one sentence from this book, let it be this: <strong>almost every design choice in SGLang serves one north star — keep the GPU busy doing USEFUL token work</strong>. In other words, maximize the FLOPs and memory bandwidth actually spent on useful computation every second. Read the previous 62 lessons together and you'll see they aren't a grab-bag of tricks but different facets of one carefully engineered <span class="mono">throughput machine</span>.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a giant <strong>central kitchen</strong> that never closes. The most expensive thing in it is a massive oven (the GPU and the model weights it must read over and over); firing and heating it once costs enormous energy. What does a smart head chef do? He never fires the oven for a single order — he <strong>bakes a whole tray at once</strong> (batching); he cuts the cold store into neat <strong>standard cells</strong> so not an inch is wasted (paging); he <strong>cooks one batch of sauce and shares it across tables</strong> (prefix caching); he keeps the prep crew and the oven in <strong>seamless lockstep</strong> so the oven is never idle (overlap scheduling); he turns "open door — slide tray — close door" into <strong>muscle memory</strong> (CUDA graphs); he keeps every knife razor-sharp so the cutting itself is as fast as possible (kernels and fusion). All of it points to one thing: <span class="mono">that expensive oven must never sit idle for a second — and every second it must be baking food that will actually be served</span>. This guide is the entire operating philosophy of that kitchen.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>At the macro level, every cost of an inference system ultimately collapses onto one line item: "GPU time." GPU time is wasted in exactly three ways: <strong>idling</strong> (the CPU didn't feed it, the oven sits empty), <strong>redundant work</strong> (the same prefix computed again and again), and <strong>not fitting</strong> (memory fragmentation or a too-big model keeps servable requests out the door). SGLang's whole architecture maps cleanly onto "eliminate those three wastes": overlap scheduling and CUDA graphs kill idling, prefix caching and RadixAttention kill redundancy, paging plus quantization and parallelism kill not-fitting. Once you hold this lens, any lesson in this book can be re-read in one sentence — <span class="mono">"this move is a deliberate trade made in the service of throughput."</span> That lens is what this lesson, and this whole book, hands you. Put it on and SGLang stops being a feature list and becomes a precise machine with a clear goal and components dividing the labor in concert.</p>
</div>

<h2>One north star: useful token work per second</h2>
<p>First, let's make "throughput" precise. It is not the vague "how many requests per second," but <strong>the fraction of each second the GPU spends on useful computation × its effective compute</strong>. In a single decode step, the GPU's truly expensive act is reading the entire set of model weights from memory into the compute units — and the cost of that read is almost independent of how many requests the step served. From this a simple yet profound conclusion emerges: <span class="mono">if you can make one weight read serve 64 requests instead of 1, your effective throughput is nearly 64× higher</span>, at almost no added cost. That is why <strong>batching (Lesson 5)</strong> is the bedrock of everything — it amortizes that most expensive weight read across a whole batch of requests. Every later mechanism in this guide is really answering a different side of the same question: <em>how do we make that batch as large, as full, and as free of wasted work as possible?</em> Carve this sentence into your mind and you'll find it acts like a thread, stringing the next dozen-plus lessons into one chain: each lesson is either helping the batch grow, or helping the batch waste less effort, or helping the GPU not idle between steps. Put differently, the whole book circles around "that one big tray" — how to pile it fuller, how to avoid re-prepping ingredients, and how to keep the oven running without a pause.</p>
<p>Once you see this, "memory" stops being merely a capacity limit and becomes a direct throughput lever. How big a batch you can run depends on how many requests' KV caches fit in memory at once. <strong>Paging (Lessons 6 and 30)</strong> slices KV into fixed-size pages, eliminating fragmentation, so the same memory holds more requests and the batch grows. <strong>Quantization and parallelism (Lessons 35 and 46)</strong> attack from the other end: shrink the model itself, or spread it across cards, to free up room for the batch. They look like solutions to different problems, but they converge on one place — <span class="mono">make the batch that a single weight read amortizes even larger</span>.</p>
<p>It's worth pausing a second to feel the unifying beauty of this lens. Before you put these glasses on, you might have filed paging under "a memory-management trick," quantization under "a model-compression trick," and tensor parallelism under "a distributed-systems trick" — sorted into different chapters, dressed in different jargon, maintained by different people. But once you fix the north star as "make the amortized batch larger," all three instantly reveal themselves as three postures of the same act: each prepares a <strong>fuller tray</strong> for that expensive oven. This is the value of a synthesis lesson — it teaches you no new mechanism, but re-arranges every mechanism you've already learned onto one chain of cause and effect. <span class="mono">When the reason for each thing converges to a single sentence, you know you truly understand the system's design intent</span>, not just its parts list.</p>

<h2>Kill the redundancy, kill the idle</h2>
<p>Enlarging the batch is only the first move. The second is <strong>don't let the GPU do busywork</strong>. Many requests share an identical prefix — the same system prompt, the same few-shot examples, the same conversation history. Recomputing that shared prefix for every request is a huge waste. <strong>Prefix caching / RadixAttention (Lessons 7 and 29)</strong> uses a radix tree to remember already-computed KV, letting requests with a shared prefix reuse it and skip the redundant compute. What it saves is precisely the precious GPU time — equivalent to lifting throughput another notch out of thin air.</p>
<p>The third move targets idling. Even with a big batch and no redundant compute, if the GPU waits while the CPU prepares the next batch, the oven goes empty. The <strong>overlap scheduler (Lessons 21 and 59)</strong> exists solely to eliminate that CPU bubble: while the GPU computes step N, the CPU is already queuing, sampling, and forming the batch for step N+1, the two overlapping in a pipeline so the GPU almost never stops. Paired with it is <strong>CUDA Graphs (Lesson 27)</strong>, which records the fixed kernel-launch sequence of each step into one graph and replays it in a single shot, cutting the per-step launch overhead — and in small-batch, decode-heavy regimes that saving is substantial. Finally, <strong>kernels and fusion (Lessons 38 and 41)</strong> merge operators and write them right up against the memory-bandwidth limit, so even "the part that actually computes" runs as fast as it can. Every layer answers the same question: <span class="mono">can each second of the GPU be filled more fully with useful work?</span></p>
<p>Put these three moves side by side and you'll see they are really fighting over different slices of the same second. Within one decode step there's a useful part ("read weights, do compute") and a wasted part ("wait for data, launch kernels, recompute redundant prefixes"). Batching enlarges the output of the useful part, prefix caching cuts away the redundant-prefix slice, overlap scheduling and CUDA graphs flatten the wait and launch slices, and kernel fusion tightens the useful part itself one more turn. So within the same second, the share taken by "useful work" keeps rising and the wasted edges keep shrinking — <span class="mono">that is how a throughput machine is squeezed out, inch by inch</span>. No single move is a silver bullet, but stacked together they push that expensive oven's utilization toward the physical limit. It also explains why removing any one link visibly collapses a chunk of whole-system throughput: they aren't redundant insurance but complementary forces aimed at the same goal.</p>

<h2>Scheduling: tightening all of it into one decision</h2>
<p>All these levers ultimately meet and get weighed in one place — <strong>scheduling</strong>. The deceptively simple question "who runs next?" actually carries two throughput goals at once: one, <strong>pack the batch as large as possible</strong> (amortize the weight read); two, <strong>prefer requests with high prefix-hit</strong> (reuse KV, skip redundant compute). SGLang encodes both into the priority computation of the waiting queue via <span class="mono">SchedulePolicy</span> and <span class="mono">CacheAwarePolicy</span> — where LPM (longest-prefix-match) groups requests that share a prefix so they both form a big batch and concentrate cache hits. These are not two goals fighting each other but two components of the same north star: <span class="mono">big batch + high prefix-hit, both lead to throughput</span>.</p>
<p>But here is a subtlety we must face honestly: chasing system throughput blindly can hurt the <strong>individual user</strong>. A very long prompt, if stuffed into prefill all at once, shoves the decode batch aside and stalls everyone else's tokens. So the mechanisms in this guide that look like they're "for latency" enter the stage: <strong>chunked prefill (Lesson 22)</strong> slices a long prompt into chunks interleaved with decode, so one big request can't hog a whole step; <strong>speculative decoding (Lesson 43)</strong> uses a small model to guess and the big model to verify, emitting several tokens per step without adding weight reads. Their spirit is not "sacrifice throughput for latency" but <span class="mono">maximize the system without sacrificing that specific human being</span> — and that is exactly the difference between a mature throughput machine and a brutish one.</p>
<p>Think this tension all the way through and you hold the last key to reading SGLang. Pure throughput maximization would push the system into a machine that watches only the grand total, never the individual: it would gladly make one unlucky user wait several seconds so long as the aggregate tokens-per-second looks prettier. But in real serving no one wants to be the sacrificed individual. So mature design adds, alongside the north star, a constraint line — <span class="mono">push the system to its max without significantly slowing any single user</span>. Chunked prefill and speculative decoding are the embodiment of that constraint: the former keeps a long request from choking the decode stream in one gulp, the latter makes a single user's tokens arrive faster with almost no added system load. By now you should feel that the whole guide has been demonstrating one balancing act over and over: greedily squeeze the hardware, yet restrain yourself to protect the experience — both booked into the same ledger inside the scheduler.</p>

<table class="t"><tr><th>Design choice</th><th>The throughput it buys</th><th>Lesson</th></tr>
<tr><td>Batching</td><td>amortize one expensive weight read across a whole batch</td><td>Lesson 5</td></tr>
<tr><td>Paging (Paged KV)</td><td>kill KV fragmentation, fit more requests in the same memory → bigger batch</td><td>Lessons 6, 30</td></tr>
<tr><td>Prefix caching / RadixAttention</td><td>reuse shared-prefix KV, skip redundant compute</td><td>Lessons 7, 29</td></tr>
<tr><td>Overlap scheduler</td><td>eliminate CPU bubbles, the GPU never idles</td><td>Lessons 21, 59</td></tr>
<tr><td>CUDA Graphs</td><td>cut per-step kernel launch overhead</td><td>Lesson 27</td></tr>
<tr><td>Kernels and fusion</td><td>ride the memory-bandwidth limit, make the real compute as fast as possible</td><td>Lessons 38, 41</td></tr>
<tr><td>Quantization and parallelism</td><td>fit bigger models onto the hardware, freeing room for the batch</td><td>Lessons 35, 46</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="The throughput-lever map: each design buys throughput — continuous batching keeps the GPU full, prefix cache skips recompute, chunked prefill doesn't stall decode, CUDA graph and fusion kill launch overhead, overlap scheduler hides CPU, speculative decode emits more tokens per forward, quantization fits more and runs faster">
    <text x="20" y="30" style="font-weight:700;fill:var(--accent-ink)">Throughput-lever map: each design → the throughput it buys</text>
    <rect x="20" y="48" width="230" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="135" y="68" text-anchor="middle" style="fill:var(--ink);font-weight:600">Continuous batching</text>
    <text x="260" y="68" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="68" style="fill:var(--muted)">keep GPU full</text>
    <rect x="20" y="86" width="230" height="30" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="135" y="106" text-anchor="middle" style="fill:var(--ink);font-weight:600">Prefix cache</text>
    <text x="260" y="106" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="106" style="fill:var(--muted)">skip recompute</text>
    <rect x="20" y="124" width="230" height="30" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="135" y="144" text-anchor="middle" style="fill:var(--ink);font-weight:600">Chunked prefill</text>
    <text x="260" y="144" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="144" style="fill:var(--muted)">don't stall decode</text>
    <rect x="20" y="162" width="230" height="30" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="135" y="182" text-anchor="middle" style="fill:var(--ink);font-weight:600">CUDA graph + fusion</text>
    <text x="260" y="182" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="182" style="fill:var(--muted)">kill launch overhead</text>
    <rect x="20" y="200" width="230" height="30" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="135" y="220" text-anchor="middle" style="fill:var(--ink);font-weight:600">Overlap scheduler</text>
    <text x="260" y="220" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="220" style="fill:var(--muted)">hide CPU</text>
    <rect x="20" y="238" width="230" height="30" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="135" y="258" text-anchor="middle" style="fill:var(--ink);font-weight:600">Speculative decode</text>
    <text x="260" y="258" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="258" style="fill:var(--muted)">more tokens/forward</text>
    <rect x="20" y="276" width="230" height="30" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="135" y="296" text-anchor="middle" style="fill:var(--ink);font-weight:600">Quantization</text>
    <text x="260" y="296" style="fill:var(--faint);font-weight:700">→</text>
    <text x="288" y="296" style="fill:var(--muted)">fit more / faster</text>
  </svg>
  <div class="figcap"><b>Fig 1 · The throughput-lever map</b> — the guide's big levers in one table: each row is "lever → the throughput it buys." Continuous batching keeps the GPU full, prefix cache skips recompute, chunked prefill doesn't stall decode, CUDA graph and fusion kill launch overhead, overlap scheduler hides the CPU, speculative decode emits more tokens per forward, and quantization fits more while running faster.</div>
</div>

<div class="flow"><div class="node">Request arrives<br><span class="mono">form batch</span></div><div class="arrow">→</div><div class="node">Schedule queue<br><span class="mono">big batch+prefix-hit</span></div><div class="arrow">→</div><div class="node">Prefix cache<br><span class="mono">skip repeats</span></div><div class="arrow">→</div><div class="node">Prefill<br><span class="mono">chunked, no hog</span></div><div class="arrow">→</div><div class="node">Decode loop<br><span class="mono">overlap+graph</span></div><div class="arrow">→</div><div class="node">Kernel exec<br><span class="mono">ride bandwidth</span></div><div class="arrow">→</div><div class="node">Return token<br><span class="mono">oven never idle</span></div></div>

<div class="fig">
  <svg viewBox="0 0 800 250" role="img" aria-label="The full request path annotated with the lever at each stage: tokenize, schedule (continuous batching), prefix (radix cache), forward (CUDA graph and fusion), sample, detokenize, stream (overlap)">
    <text x="20" y="28" style="font-weight:700;fill:var(--accent-ink)">The full request path: the lever at each stage</text>
    <rect x="12" y="70" width="98" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="61" y="97" text-anchor="middle" style="fill:var(--ink)">tokenize</text>
    <text x="61" y="138" text-anchor="middle" style="fill:var(--muted);font-size:11px">text→ids</text>
    <text x="116" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="124" y="70" width="98" height="44" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="173" y="97" text-anchor="middle" style="fill:var(--ink)">schedule</text>
    <text x="173" y="135" text-anchor="middle" style="fill:var(--blue);font-size:11px">continuous</text>
    <text x="173" y="149" text-anchor="middle" style="fill:var(--blue);font-size:11px">batching</text>
    <text x="228" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="236" y="70" width="98" height="44" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="285" y="97" text-anchor="middle" style="fill:var(--ink)">prefix</text>
    <text x="285" y="135" text-anchor="middle" style="fill:var(--teal);font-size:11px">radix</text>
    <text x="285" y="149" text-anchor="middle" style="fill:var(--teal);font-size:11px">cache</text>
    <text x="340" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="348" y="70" width="98" height="44" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="397" y="97" text-anchor="middle" style="fill:var(--ink)">forward</text>
    <text x="397" y="135" text-anchor="middle" style="fill:var(--purple);font-size:11px">CUDA graph</text>
    <text x="397" y="149" text-anchor="middle" style="fill:var(--purple);font-size:11px">+ fusion</text>
    <text x="452" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="460" y="70" width="98" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="509" y="97" text-anchor="middle" style="fill:var(--ink)">sample</text>
    <text x="509" y="138" text-anchor="middle" style="fill:var(--muted);font-size:11px">pick next token</text>
    <text x="564" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="572" y="70" width="98" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="621" y="97" text-anchor="middle" style="fill:var(--ink)">detokenize</text>
    <text x="621" y="138" text-anchor="middle" style="fill:var(--muted);font-size:11px">ids→text</text>
    <text x="676" y="97" text-anchor="middle" style="fill:var(--faint);font-weight:700">→</text>
    <rect x="684" y="70" width="98" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="733" y="97" text-anchor="middle" style="fill:var(--ink)">stream</text>
    <text x="733" y="142" text-anchor="middle" style="fill:var(--amber);font-size:11px">overlap</text>
    <text x="20" y="220" style="fill:var(--muted);font-size:12px">"The life of a request" — retold here as a throughput story: each stage carries the lever that applies to it.</text>
  </svg>
  <div class="figcap"><b>Fig 2 · The throughput lever at each stage</b> — the "life of a request" retold as a throughput story: along the end-to-end path (tokenize → schedule → prefix → forward → sample → detokenize → stream), each stage carries a small tag naming the lever that applies there — continuous batching at schedule, radix cache at prefix, CUDA graph and fusion at forward, overlap scheduling at stream.</div>
</div>

<div class="cols"><div class="col"><strong>Throughput levers (push the system to its max)</strong><ul><li>Batching: amortize the weight read</li><li>Paging: fit more requests</li><li>Prefix caching: skip redundant compute</li><li>Overlap scheduling: eliminate idling</li><li>CUDA Graphs: cut launch overhead</li><li>Kernels and fusion: drain the bandwidth</li></ul></div><div class="col"><strong>Latency safeguards (don't sacrifice the person)</strong><ul><li>Chunked prefill (Lesson 22): slice long prompts, interleave, never hog a whole step</li><li>Speculative decoding (Lesson 43): emit several tokens per step without extra weight reads</li><li>Essence: maximize the system while protecting the individual user's experience</li></ul></div></div>

<div class="layers"><div class="layer">Outermost: scheduling and batching — decide "how much to feed at once, and whom"</div><div class="layer">Middle: caching and paging — decide "how much fits, and how much can be reused"</div><div class="layer">Kernel layer: fusion and CUDA Graphs — decide "how fast each step computes, how cheap the launch"</div><div class="layer">Innermost core: the GPU that must never idle — every second doing useful token work</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy</span><span class="ln">who runs next is chosen to maximize useful GPU work: big batch + high prefix-hit</span></div><pre>class CacheAwarePolicy(Enum):
    LPM = "lpm"          # longest-prefix-match: group requests that share a prefix
    # ... other cache-aware policies

class SchedulePolicy:
    # who runs next is chosen to MAXIMIZE useful GPU work:
    # pack the biggest batch (amortize the weight read) AND prefer high prefix-hit
    # (reuse KV, skip redundant compute) — both serve throughput
    def calc_priority(self, waiting_queue):
        ...
</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.get_next_batch_to_run</span><span class="ln">the continuous-batching lever: each step pick which batch to run</span></div><pre>def get_next_batch_to_run(self) -&gt; Optional[ScheduleBatch]:
    # the continuous-batching lever, run EVERY step: merge any finished
    # prefill into the running batch, then decide what to run next —
    # admit a new prefill batch (more work) or keep decoding.
    merge_finished_prefill_into(self.running_batch)
    new_batch = self.get_new_batch_prefill()    # admit within token budget
    if new_batch is not None:
        return new_batch                         # run prefill this step
    return self.running_batch                    # else keep decoding
</pre></div>

<div class="card"><div class="tag">🔎 A concrete example</div>
<p>Because <span class="mono">get_next_batch_to_run</span> runs once <strong>every step</strong>, the instant a request in the batch finishes, the slot it frees is <strong>immediately</strong> refilled by a waiting request (that is continuous batching) — there's no need to drain the whole batch first and only then start a new one. This <strong>single lever</strong> is why the expensive oven (the GPU) stays saturated: every step it makes the most profitable choice among "merge finished prefill → admit a new prefill if it fits → otherwise keep decoding."</p>
</div>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li><strong>One north star</strong>: almost every SGLang design keeps the GPU busy doing useful token work — maximizing useful FLOPs and bandwidth per second.</li>
<li><strong>Batching is the bedrock (Lesson 5)</strong>: amortizing the most expensive weight read across a whole batch is where throughput begins.</li>
<li><strong>Memory is throughput</strong>: paging (Lessons 6, 30), quantization and parallelism (Lessons 35, 46) all free room for a bigger batch.</li>
<li><strong>Kill redundancy (Lessons 7, 29)</strong>: prefix caching / RadixAttention reuses shared prefixes, skipping redundant compute.</li>
<li><strong>Kill idling (Lessons 21, 27, 59)</strong>: overlap scheduling removes CPU bubbles, CUDA Graphs remove launch overhead.</li>
<li><strong>Drain the bandwidth (Lessons 38, 41)</strong>: kernels and fusion make the real compute ride the hardware limit.</li>
<li><strong>Scheduling closes the loop</strong>: <span class="mono">SchedulePolicy</span> and <span class="mono">CacheAwarePolicy</span> pursue "big batch" and "high prefix-hit" at once — both serve throughput.</li>
<li><strong>Latency safeguards</strong>: chunked prefill (Lesson 22) and speculative decoding (Lesson 43) ensure the system is maximized without sacrificing the individual user.</li>
</ul>
</div>

<div class="card"><div class="tag">🏁 The end</div>
<p>One paragraph to look back over the whole book: starting from the map in <strong>Lesson 3, "the life of a request,"</strong> we walked through batching, paging, prefix caching, scheduling, kernels, quantization, parallelism, speculative decoding… all the way to this moment. You can now re-read every part of SGLang as a <span class="mono">deliberate choice made in the service of throughput (or in service of not sacrificing the individual user)</span>. This pair of "throughput glasses" is exactly what the whole guide has been polishing for you since Lesson 1 — put them on, open any piece of source again, and you'll no longer see isolated tricks but coordination under one north star. Thank you for walking these 63 lessons with me; may you wear these glasses to understand, and to improve, the next great inference system. The road is long and the rivers run far — until we meet again. 🚀</p>
</div>
"""}
