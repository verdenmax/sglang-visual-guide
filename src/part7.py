"""Part 7 - KV cache & memory. Lessons (L29-L32) for the SGLang visual guide.

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons open the memory subsystem: the RadixAttention implementation
(L29, the radix tree data structures), the paged memory pools + allocator (L30),
HiCache GPU/CPU/disk tiering (L31), and eviction policy + hit-rate economics (L32).
"""

LESSON_29 = {"zh": r"""
<p class="lead">
第 7 课讲过<strong>为什么</strong>要共享前缀——同一段系统提示被上千条请求重复，重算它的 KV 是纯浪费。
这一课不再重复"为什么"，而是钻进 <span class="inline">RadixCache</span> 的<strong>代码内部</strong>：
那棵树到底由什么数据结构搭成、一次 <span class="mono">match_prefix</span> 怎样沿树下行、
节点在哪一刻被<strong>分裂</strong>、又靠什么<strong>锁</strong>住正在用的前缀不被回收。把这些看懂，
你就能读懂 <span class="mono">radix_cache.py</span> 这个全引擎最精巧的文件之一。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把基数树想成一棵<strong>共享文件夹树（路径 trie）</strong>：每个文件夹（节点）的"名字"是<strong>一连串路径片段</strong>（一段 token），
  文件夹里不放文件本体，只放一张<strong>便签</strong>，写着"这段内容的 KV 存在仓库的哪些货架号"。你顺着路径<strong>逐段匹配</strong>往下走；
  当两条路径<strong>前半段相同、后半段分叉</strong>时，就把那个文件夹<strong>从分叉点剪成两层</strong>——公共前缀提到父文件夹，各自的尾巴变成两个子文件夹。
  只要还有人正<strong>打开着</strong>某个文件夹（lock_ref&gt;0），它就<strong>不许被删</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  记住一句话：<strong>树是索引，池是仓库</strong>。基数树本身<strong>不存 KV 张量</strong>，它存的是"token 序列 → KV 在池里的槽位号（indices）"这张映射；
  真正的 K/V 张量躺在<strong>显存池</strong>里（第 30 课）。两条请求共享前缀，本质是它们在树上<strong>走过同一批祖先节点</strong>，
  于是拿到<strong>同一串 indices</strong>、指向<strong>同一批物理槽位</strong>——零拷贝、零重算。这棵树还要配合驱逐与锁（第 32 课）、
  HiCache 分层（第 31 课）和缓存感知调度（第 20 课）一起工作。
</div>

<h2>TreeNode：一个节点到底装了什么</h2>
<p>整棵树由 <span class="mono">TreeNode</span> 串成。理解这个类的五个字段，几乎就理解了一切。
<strong>key</strong> 是"进入本节点这条边上的一段 token id"——注意它是<strong>一段</strong>而非一个，把连续无分叉的 token 压在同一条边上，树才不会退化成一条长链。
<strong>children</strong> 是个字典，<strong>用每个子节点 key 的第一个 token 作索引</strong>，这样下行时一步就能定位该走哪个孩子。
<strong>value</strong> 是这段 token 对应的 <strong>KV 槽位号（indices）</strong>——再强调一次，它是<strong>指向池子的指针</strong>（第 30 课），<strong>不是 KV 张量本身</strong>。
<strong>lock_ref</strong> 记录"当前有几条在跑的请求正用着这段前缀"，是保护它不被驱逐（第 32 课）的护身符。
<strong>last_access_time</strong> 给 LRU 用，决定显存吃紧时谁先被赶走。</p>
<p>为什么 <strong>key 要是一段而不是单个 token</strong>？这是基数树（radix tree，又叫压缩前缀树）相对普通前缀树的关键优化。普通 trie 每个节点只放一个字符，
一段没有分叉的长前缀会拉成一条又细又长的链，遍历慢、指针多、内存碎。基数树把"<strong>中间没有任何分叉的连续 token</strong>"压进同一条边、同一个节点，
只有在<strong>真正发生分叉的地方</strong>才设节点。于是一条几千 token 的系统提示，只要没有别的请求在中途岔开，就只占<strong>一个节点</strong>，
匹配它也只是一次 <span class="mono">key.match</span> 的逐位比较。这也解释了 <span class="mono">_split_node</span> 的存在意义：当新请求<strong>恰好在这条长边的中间岔开</strong>，
我们才被迫把这个"压缩节点"在岔口切开——平时能不切就不切，保持树尽量扁、尽量省。</p>
<p>还要留意 <span class="mono">children</span> 用"<strong>子节点 key 的第一个 token</strong>"作字典键这个设计。下行到某节点时，我们手里剩着一串待匹配的 token，
只需取<strong>它的第一个 id</strong>，去 <span class="mono">children</span> 里做一次 O(1) 的哈希查找，就知道"该不该往下走、走哪个孩子"。
正因为任意两个孩子的 key 首 token 必然不同（否则它们就该被合并或分裂），用首 token 作键既无歧义又极快——这是整棵树查询性能的根基。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>一个 TreeNode 的五个关键字段</b>：边上的 token（key）+ 指向孩子的字典 + 指向池子的 indices + 锁计数 + 访问时间</div>
  <div class="cells"><span class="lab">key</span><span class="cell">tok</span><span class="cell">国</span><span class="cell">的</span><span class="sep">→</span><span class="cell q">进入本节点这条边上的一段 token id（一段，非一个）</span></div>
  <div class="cells"><span class="lab">children</span><span class="cell hl">首token→子</span><span class="sep">→</span><span class="cell q">字典，用每个孩子 key 的首 token 作键，一步定位下行方向</span></div>
  <div class="cells"><span class="lab">value</span><span class="cell hl">#1024</span><span class="cell hl">#1025</span><span class="sep">→</span><span class="cell q">这段 token 的 KV 槽位号（指针），张量在池里（第 30 课）</span></div>
  <div class="cells"><span class="lab">lock_ref</span><span class="cell">2</span><span class="sep">→</span><span class="cell q">几条在跑的请求正用这段前缀；&gt;0 则不许驱逐（第 32 课）</span></div>
  <div class="cells"><span class="lab">last_access</span><span class="cell">t₇</span><span class="sep">→</span><span class="cell q">最近访问时间，给 LRU 用（第 32 课）</span></div>
</div>

<table class="t">
  <tr><th>TreeNode 字段</th><th>角色</th><th>为什么需要它</th></tr>
  <tr><td class="mono">key</td><td>边上的一段 token id</td><td>把无分叉的连续 token 压成一条边，避免退化成长链</td></tr>
  <tr><td class="mono">children</td><td>首 token → 子节点的字典</td><td>下行时 O(1) 定位该走哪个孩子</td></tr>
  <tr><td class="mono">value</td><td>KV 槽位号（indices）</td><td>指向池子的指针，复用即复用同一批物理槽位（第 30 课）</td></tr>
  <tr><td class="mono">lock_ref</td><td>在用引用计数</td><td>&gt;0 时锁住，防止在飞的 KV 被驱逐（第 32 课）</td></tr>
  <tr><td class="mono">last_access_time</td><td>LRU 时间戳</td><td>显存吃紧时决定谁先被赶走（第 32 课）</td></tr>
</table>

<h2>match_prefix：沿树下行，在分叉处分裂</h2>
<p>来了一条新请求，它的 token 序列要先去树上<strong>问一句"我有多长一段已经被算过了"</strong>。这就是 <span class="mono">match_prefix</span>。
它从根节点出发，<strong>用当前 token 的第一个 id 去 children 字典里找孩子</strong>；找到后，拿这段 token 和<strong>孩子边上的 key 逐位匹配</strong>。
有两种结果：其一，<strong>整条边都匹配上</strong>——把这个孩子的 value（一串 indices）收进结果，<strong>消费掉</strong>已匹配的 token，再用剩下 token 的首 id 继续往下找；
其二，<strong>只匹配了边的前一半就分叉了</strong>——这时不能整边复用，必须调 <span class="mono">_split_node</span> 在<strong>分歧点把这个节点切成两层</strong>。</p>
<p>分裂做的事很优雅：新建一个父节点，<strong>key 取公共前缀那一截</strong>、value 取对应那段 indices；原节点降为它的孩子，<strong>key 与 value 都只留分歧之后的尾巴</strong>。
于是"公共部分"被提取成一个可被多方共享的祖先，"各自不同的尾巴"成了它的分支。这一步是基数树的灵魂——<strong>共享是在匹配的瞬间、按需地长出来的</strong>，
而不是预先规划好的。<span class="mono">match_prefix</span> 最终返回两样东西：<strong>命中的那串 KV indices</strong>（直接复用，免重算）和<strong>匹配到的最深节点</strong>（后续 insert 与加锁的落脚点）。</p>
<p>举个具体例子把分裂讲实。假设树上已有一个节点，边上的 key 是 token 串 <span class="mono">[你, 是, 一, 个, 助, 手]</span>，它的 value 是这 6 个 token 在池里的 6 个槽位号。
现在来了条请求，前缀是 <span class="mono">[你, 是, 一, 个, 专, 家]</span>——前 4 个 token 一致，第 5 个开始岔开。<span class="mono">key.match</span> 算出 <span class="mono">prefix_len = 4</span>，
小于原边长度 6，于是触发 <span class="mono">_split_node</span>：新建父节点，key 取 <span class="mono">[你, 是, 一, 个]</span>、value 取前 4 个槽位号；原节点降为孩子，key 只剩 <span class="mono">[助, 手]</span>、value 只剩后 2 个槽位号。
这样老请求（…助手）和新请求（…专家）就<strong>共享了"你是一个"这 4 个 token 的 KV</strong>，各自的尾巴成了父节点下的两个分支。注意分裂<strong>不搬动也不重算任何 KV 张量</strong>——
它只是把 key 和 value 这两个轻量数组<strong>切片重挂</strong>，是一次纯指针操作，极快。</p>
<p>再补一个边界细节：当 <span class="mono">page_size &gt; 1</span> 时，匹配与分裂都以<strong>页为粒度</strong>对齐，传入的 token 长度会先被截到页的整数倍再比较。
这是为了和分页内存池（第 30 课）对齐——KV 在池里是<strong>按页分配</strong>的，所以树上的复用边界也必须落在页边界上，否则半页复用会造成槽位错位。
你在代码里看到的 <span class="mono">key.page_aligned(self.page_size)</span> 和 <span class="mono">child_key(self.page_size)</span> 就是在做这件事。这也呼应了那条主线：<strong>树的索引必须时刻和池的物理布局对齐</strong>，两者步调一致，复用才安全。理解了这层对齐约束，你就明白为什么基数树的代码里到处都在和 <span class="mono">page_size</span> 打交道，而不是简单地按单个 token 处理。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>下行匹配 match</h4><p>用当前 token 首 id 在 <span class="mono">children</span> 找孩子，拿边上的 <span class="mono">key</span> 逐位比。整边命中就收下它的 indices、消费 token、继续下探。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>半路分叉就分裂 split</h4><p>只匹配到边的前一半 → <span class="mono">_split_node</span> 在分歧点切成两层：公共前缀升为父、原尾巴降为子，父节点从此可被多方共享。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>挂上分叉后缀 insert</h4><p>匹配点之后剩下的"新尾巴" → 作为一个新子节点 <span class="mono">insert</span> 进去，它的 value 是这段新 token 写入池子拿到的 indices。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>加锁保护 lock</h4><p>从命中的最深节点<span class="mono">向上</span>逐节点 <span class="mono">lock_ref += 1</span>，让驱逐（第 32 课）碰不到这段正在飞的前缀；请求结束再逐节点解锁。</p></div></div>
</div>

<p>真实代码里，下行循环与分裂就长这样——注意 <span class="mono">prefix_len &lt; len(child.key)</span> 这个判断，正是"只匹配了半条边、必须分裂"的那一刻：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/radix_cache.py ::RadixCache._match_prefix_helper</span><span class="ln">下行匹配与按需分裂</span></div>
  <pre><span class="kw">def</span> _match_prefix_helper(self, node, key):
    child_key = key.child_key(self.page_size)   <span class="cm"># 用当前 token 首 id 作索引</span>
    value = []
    <span class="kw">while</span> len(key) &gt; 0 <span class="kw">and</span> child_key <span class="kw">in</span> node.children.keys():
        child = node.children[child_key]
        prefix_len = child.key.match(key, page_size=self.page_size)
        <span class="kw">if</span> prefix_len &lt; len(child.key):       <span class="cm"># 只匹配了半条边 → 必须分裂</span>
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)        <span class="cm"># 收下公共前缀的 KV indices</span>
            node = new_node
            <span class="kw">break</span>
        <span class="kw">else</span>:                                  <span class="cm"># 整条边命中 → 收下、消费 token、继续下探</span>
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            child_key = key.child_key(self.page_size)
    <span class="kw">return</span> value, node                        <span class="cm"># 命中的 indices + 最深节点</span></pre>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="match_prefix 从 root 沿匹配的边逐段下行，高亮最长命中前缀的路径与终点节点，未命中的尾巴作为新分支挂出">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">查询 query：[你 是 一 个 助 手 X]</text>
    <rect x="20" y="40" width="470" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="40" y="60" class="mono" style="font-size:12px;fill:var(--accent-ink)">你 是 一 个 助 手</text>
    <text x="360" y="60" class="mono" style="font-size:12px;fill:var(--faint)">X（未命中）</text>
    <rect x="20" y="150" width="74" height="44" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="57" y="177" text-anchor="middle" style="font-size:13px">root</text>
    <line x1="94" y1="172" x2="168" y2="172" style="stroke:var(--accent);stroke-width:3"/>
    <rect x="168" y="150" width="120" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="228" y="170" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--accent-ink)">[你 是]</text>
    <text x="228" y="186" text-anchor="middle" style="font-size:10px;fill:var(--muted)">共享前缀</text>
    <line x1="288" y1="166" x2="360" y2="112" style="stroke:var(--accent);stroke-width:3"/>
    <line x1="288" y1="180" x2="360" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="360" y="90" width="160" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2.5"/>
    <text x="440" y="110" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--accent-ink)">[一 个 助 手]</text>
    <text x="440" y="126" text-anchor="middle" style="font-size:10px;fill:var(--accent-ink);font-weight:700">最长命中 · 终点</text>
    <rect x="360" y="210" width="120" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="420" y="237" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--teal)">[专 家]</text>
    <line x1="520" y1="112" x2="592" y2="112" style="stroke:var(--blue);stroke-width:2;stroke-dasharray:5 4"/>
    <rect x="592" y="90" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="672" y="110" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">[X …]</text>
    <text x="672" y="126" text-anchor="middle" style="font-size:10px;fill:var(--blue)">未命中尾巴 → 新分支</text>
  </svg>
  <div class="figcap"><b>图 1 · match_prefix 沿 radix 树走到最长命中</b> — 从 <span class="mono">root</span> 出发，用当前 token 的首 id 逐段匹配下行：命中的边连成高亮路径（<span class="mono">[你 是] → [一 个 助 手]</span>），终点节点就是返回的最深匹配；剩下未命中的尾巴 <span class="mono">[X …]</span> 之后会作为一条新分支挂上去。</div>
</div>

<p>上面的 <span class="mono">_match_prefix_helper</span> 是<strong>内部</strong>下行循环；真正被外界调用的<strong>公有入口</strong>是 <span class="mono">match_prefix</span>。它接收一个 <span class="mono">MatchPrefixParams</span>、返回一个 <span class="mono">MatchResult</span>：先把 key 按页对齐，禁用或空 key 直接返回空结果，否则委托 helper 下行（途中可能分裂一个节点），最后刷新访问时间戳供驱逐使用：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.match_prefix</span><span class="ln">公有入口：找最长缓存前缀，必要时分裂节点</span></div>
  <pre><span class="kw">def</span> match_prefix(self, params: MatchPrefixParams) -&gt; MatchResult:
    <span class="cm"># 在 radix 树里找 params.key 的最长缓存前缀。</span>
    <span class="cm"># 若命中点落在某段存储的中间，就把该节点分裂一次以</span>
    <span class="cm"># 暴露边界；同时刷新访问时间戳（之后给驱逐用）。</span>
    key = params.key
    <span class="kw">if</span> self.disable <span class="kw">or</span> len(key) == 0:
        <span class="kw">return</span> self._empty_match_result
    key = key.page_aligned(self.page_size)   <span class="cm"># 先判禁用/空，再按页对齐</span>
    ...
    <span class="kw">return</span> MatchResult(device_indices=..., last_device_node=node)</pre>
</div>

<div class="card detail">
  <div class="tag">🧮 具体例子</div>
  <strong>例 1：共享系统前导。</strong> 两条请求都以同一段 100 token 的系统提示开头。第一条把它算完、<span class="mono">insert</span> 进树；第二条来时 <span class="mono">match_prefix</span> 直接命中这 100 个 token，返回它们在池里的 <strong>100 个 KV 槽位号</strong>——这 100 步前向被整个跳过，只需计算各自新的后缀。
  <strong>例 2：page_size &gt; 1 的对齐。</strong> 设 <span class="mono">page_size = 16</span>，一条 key 长 100。<span class="mono">key.page_aligned(16)</span> 先把长度截到 16 的整数倍即 <strong>96</strong>，再去匹配；尾部 4 个不足一页的 token 不参与前缀复用，留给后续随新内容一起计算。
</div>

<h2>insert 与 lock_ref：挂上新尾巴、锁住在用前缀</h2>
<p>匹配只解决了"已有多少"。剩下那段<strong>从未见过的后缀</strong>怎么办？交给 <span class="mono">insert</span>：它沿着和 match 一样的路径走到匹配点，
把<strong>分叉出来的新 token 作为一个新子节点</strong>挂在最深匹配节点下，新节点的 value 就是这段 token 把 K/V 写进池子后拿到的那串 indices。
若插入途中也遇到"半条边"，同样会先 <span class="mono">_split_node</span> 再挂——所以 match 和 insert 共用同一套分裂逻辑，保证树<strong>永远是规范的</strong>。</p>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="insert 在两个序列分叉处分裂节点：原来一个节点存 [A B C D]，新序列共享 [A B] 后分叉，分裂后 [A B] 成为父节点，[C D] 与新的 [X Y] 成为两个子节点">
    <line x1="400" y1="30" x2="400" y2="300" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">之前 before</text>
    <rect x="40" y="60" width="74" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="77" y="85" text-anchor="middle" style="font-size:13px">root</text>
    <line x1="77" y1="100" x2="160" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="160" y="150" width="180" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:2"/>
    <text x="250" y="170" text-anchor="middle" class="mono" style="font-size:13px">[A B C D]</text>
    <text x="250" y="186" text-anchor="middle" style="font-size:10px;fill:var(--muted)">一个节点存整段</text>
    <rect x="120" y="232" width="260" height="56" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="254" text-anchor="middle" style="font-size:11px;fill:var(--amber)">新序列 [A B X Y]</text>
    <text x="250" y="272" text-anchor="middle" style="font-size:11px;fill:var(--amber)">共享 [A B]，在第 3 个 token 分叉</text>
    <text x="420" y="26" style="font-weight:700;fill:var(--accent-ink)">之后 after</text>
    <rect x="440" y="50" width="74" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="477" y="75" text-anchor="middle" style="font-size:13px">root</text>
    <line x1="477" y1="90" x2="600" y2="120" style="stroke:var(--accent);stroke-width:3"/>
    <rect x="540" y="120" width="120" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2.5"/>
    <text x="600" y="140" text-anchor="middle" class="mono" style="font-size:13px;fill:var(--accent-ink)">[A B]</text>
    <text x="600" y="156" text-anchor="middle" style="font-size:10px;fill:var(--accent-ink);font-weight:700">父 · 公共前缀只存一份</text>
    <line x1="585" y1="164" x2="520" y2="220" style="stroke:var(--teal);stroke-width:2"/>
    <line x1="615" y1="164" x2="700" y2="220" style="stroke:var(--blue);stroke-width:2;stroke-dasharray:5 4"/>
    <rect x="450" y="220" width="120" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="510" y="240" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--teal)">[C D]</text>
    <text x="510" y="256" text-anchor="middle" style="font-size:10px;fill:var(--teal)">原节点的尾巴</text>
    <rect x="640" y="220" width="120" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="700" y="240" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">[X Y]</text>
    <text x="700" y="256" text-anchor="middle" style="font-size:10px;fill:var(--blue)">新序列的尾巴</text>
  </svg>
  <div class="figcap"><b>图 2 · insert 在两个序列分叉处分裂节点</b> — 之前一个节点把 <span class="mono">[A B C D]</span> 整段存在一起；新序列 <span class="mono">[A B X Y]</span> 共享 <span class="mono">[A B]</span> 后分叉。<span class="mono">_split_node</span> 把它切成父 <span class="mono">[A B]</span>（公共前缀，只存一份）+ 子 <span class="mono">[C D]</span>（原尾巴）+ 新的兄弟子节点 <span class="mono">[X Y]</span>，于是公共前缀被两条序列共享。</div>
</div>

<p>最后是<strong>锁</strong>。一条请求开始使用某段命中前缀时，引擎从最深匹配节点<strong>一路向上 walk 到根</strong>，途中每个节点 <span class="mono">lock_ref += 1</span>（这就是 <span class="mono">inc_lock_ref</span>）。
为什么要向上锁整条链？因为驱逐是从叶子往上回收的（第 32 课），只要这段前缀在飞，<strong>它的每一个祖先都不能被当成"可驱逐叶子"清掉</strong>，否则正在算的请求会读到一片被回收的垃圾槽位。
请求结束时调 <span class="mono">dec_lock_ref</span> 反向把计数减回去；当某节点 <span class="mono">lock_ref</span> 归零，它才重新变成"可驱逐"的候选。一句话：<strong>lock_ref 守护的是"正在被使用的 KV 不被回收"这条铁律</strong>。</p>
<p>这里有个容易混淆的点值得点破：<strong>lock_ref 和"被多少请求共享"不是一回事</strong>。一个节点可能被树结构上挂着很多历史请求留下的分支，但只有<strong>此刻真正在跑、真正在读这段 KV</strong> 的请求才会给它加锁。
换句话说，lock_ref 计的是"<strong>正在飞的引用</strong>"，不是"<strong>历史上用过的次数</strong>"。这区分至关重要：驱逐策略要回收的恰恰是那些<strong>留在树上、但当下没人在用</strong>的前缀——它们 lock_ref 为 0、可被安全清掉以腾显存；
而 lock_ref 为正的节点，哪怕 last_access_time 很老、看着像该淘汰，也<strong>绝对不能动</strong>。inc/dec 这一对就是在"尽量长期缓存以提升命中率"和"绝不回收在用数据以保证正确性"之间划出的那条硬边界。</p>

<h2>共享如何真正发生：树是索引，池是存储</h2>
<p>把三件事接起来你就通透了。两条请求若前缀相同，它们在 <span class="mono">match_prefix</span> 里会<strong>走过同一批祖先节点</strong>、
拿到<strong>同一串 value（indices）</strong>，于是它们的注意力都去读池子里<strong>同一批物理槽位</strong>——这就是"共享"的全部真相：不是复制 KV，而是<strong>复用指针</strong>。
等到各自生成出不同的下一个 token，路径才在某个节点<strong>分叉</strong>成两片不同的叶子，各自的新 KV 写进池子的新槽位、挂成各自的新子节点。
树负责"<strong>谁和谁共享、共享到第几个 token</strong>"，池负责"<strong>KV 到底存在哪</strong>"——索引与存储彻底分离，正是这套设计能既省显存又跑得快的根本。</p>
<p>把视角拉远一点，你会看到这套数据结构如何向上托起整个引擎。缓存感知调度（第 20 课）之所以能"<strong>优先放那些前缀已在树上的请求进批</strong>"，靠的就是 <span class="mono">match_prefix</span> 能在调度前快速算出每条候选请求<strong>能复用多长</strong>；
HiCache（第 31 课）则是 <span class="mono">RadixCache</span> 的<strong>分层子类</strong>——当 GPU 显存里的节点被驱逐，它不直接丢，而是把 value 指向的 KV 先<strong>下沉到 CPU 内存甚至磁盘</strong>，需要时再拉回，靠的正是 TreeNode 里那几个 host 相关字段。
可以说，这一课讲的 <span class="mono">TreeNode</span> 五字段、<span class="mono">match_prefix</span> 的下行与分裂、<span class="mono">insert</span> 的挂接、<span class="mono">lock_ref</span> 的加解锁，是后面好几课的<strong>共同地基</strong>。把这棵树读透，
你再看驱逐（第 32 课）、分层（第 31 课）、调度（第 20 课）时就不会迷路——它们都只是在这棵"token 序列 → KV 槽位"的索引树上，添加各自的策略而已。</p>
<p>最后留一个常被问到的细节作收尾：<strong>这棵树是怎么"长大"的</strong>？答案是请求<strong>用完即种</strong>。一条请求在解码过程中或彻底结束时，引擎会把它<strong>实际算出来的那段 token 及其 KV indices</strong> 通过 <span class="mono">insert</span> 写回树里
（对应代码里的 <span class="mono">cache_unfinished_req</span> 与 <span class="mono">cache_finished_req</span>）。于是<strong>每一条请求都在为后来者铺路</strong>：它算过的前缀沉淀成树上的节点，下一条撞上同样前缀的请求一 <span class="mono">match_prefix</span> 就能白捡。
这就形成一个良性循环——越热门的前缀（系统提示、few-shot 例子、公共文档）被越多请求种进树、又被越多请求命中复用，命中率随流量自然爬升。而当显存装不下时，<strong>驱逐</strong>（第 32 课）再按 LRU 把最冷、且 lock_ref 为 0 的叶子摘掉，腾出槽位。
种树、查树、锁树、剪树，这四个动作循环往复，就是 <span class="mono">RadixCache</span> 在一台 GPU 上日夜运转的全部故事。</p>

<div class="flow">
  <div class="node hl"><div class="nt">root</div><div class="nd">空前缀</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">共享祖先</div><div class="nd">"你是一个助手…"<br>两请求都走它 · 同一串 indices</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">叶 A</div><div class="nd">请求 A 的独有后缀<br>→ 池里新槽位</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">叶 B</div><div class="nd">请求 B 的独有后缀<br>→ 池里另一批槽位</div></div>
</div>

<div class="card key">
  <div class="tag">🔑 本课要点</div>
  <strong>① TreeNode 的 value 是 KV 槽位号（indices），不是 KV 张量</strong>——树是索引，池才是存储（第 30 课）。
  <strong>② match_prefix 沿 children 字典下行、逐边匹配；半条边匹配时 _split_node 在分歧点切两层</strong>，公共前缀升为可共享的父。
  <strong>③ insert 把分叉后缀挂成新子节点</strong>，与 match 共用分裂逻辑保持树规范。
  <strong>④ inc/dec_lock_ref 从命中节点向上锁整条链</strong>，守护"在飞的 KV 不被驱逐"（第 32 课）。
  <strong>⑤ 共享 = 两条路径走过同一祖先、拿同一串 indices、指同一批物理槽位</strong>，分叉后各长各的叶子。概念见第 7 课，HiCache 分层见第 31 课，缓存感知调度见第 20 课。
</div>
""",
             "en": r"""
<p class="lead">
Lesson 7 covered <strong>why</strong> we share prefixes — one system prompt repeated by thousands of requests makes recomputing its KV pure waste.
This lesson does not repeat the "why"; it drills into the <strong>code</strong> of <span class="inline">RadixCache</span>:
exactly what data structures build that tree, how one <span class="mono">match_prefix</span> walks down it,
the moment a node gets <strong>split</strong>, and what <strong>locks</strong> an in-use prefix so it can't be reclaimed. Grasp this and you can read
<span class="mono">radix_cache.py</span>, one of the most elegant files in the whole engine.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the radix tree as a <strong>shared folder tree (a trie of file paths)</strong>: each folder (node) is named by a <strong>run of path segments</strong> (a run of tokens),
  and the folder holds no file itself, just a <strong>sticky note</strong> saying "the KV for this run lives on these shelf numbers in the warehouse." You walk down <strong>matching segment by segment</strong>;
  when two paths <strong>agree on a front part then diverge</strong>, you <strong>split that folder into two levels</strong> — the common prefix is lifted into a parent, each tail becomes a child folder.
  As long as someone has a folder <strong>open</strong> (lock_ref&gt;0), it <strong>may not be deleted</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  One sentence: <strong>the tree is the index, the pool is the warehouse</strong>. The radix tree itself <strong>stores no KV tensors</strong>; it stores the mapping "token sequence → KV slot numbers (indices) in the pool."
  The real K/V tensors sit in the <strong>memory pool</strong> (Lesson 30). Two requests sharing a prefix means they <strong>walk the same ancestor nodes</strong> on the tree,
  get the <strong>same run of indices</strong>, pointing at the <strong>same physical slots</strong> — zero copy, zero recompute. This tree also works with eviction and locks (Lesson 32),
  HiCache tiering (Lesson 31), and cache-aware scheduling (Lesson 20).
</div>

<h2>TreeNode: what a node actually holds</h2>
<p>The whole tree is woven from <span class="mono">TreeNode</span>. Understand its five fields and you understand almost everything.
<strong>key</strong> is "a run of token ids on the edge into this node" — note it's a <strong>run</strong>, not one token; packing consecutive un-branched tokens onto one edge keeps the tree from degenerating into a long chain.
<strong>children</strong> is a dict <strong>keyed by the first token of each child's key</strong>, so descending picks the right child in one step.
<strong>value</strong> is the <strong>KV slot numbers (indices)</strong> for this run — again, a <strong>pointer into the pool</strong> (Lesson 30), <strong>not the KV tensors themselves</strong>.
<strong>lock_ref</strong> counts "how many running requests are currently using this prefix," the talisman that protects it from eviction (Lesson 32).
<strong>last_access_time</strong> feeds LRU, deciding who gets evicted first under memory pressure.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>The five key fields of a TreeNode</b>: tokens on the edge (key) + dict of children + indices into the pool + lock count + access time</div>
  <div class="cells"><span class="lab">key</span><span class="cell">tok</span><span class="cell">of</span><span class="cell">the</span><span class="sep">→</span><span class="cell q">a run of token ids on the edge into this node (a run, not one)</span></div>
  <div class="cells"><span class="lab">children</span><span class="cell hl">firstTok→kid</span><span class="sep">→</span><span class="cell q">dict keyed by each child's first token; one-step descent</span></div>
  <div class="cells"><span class="lab">value</span><span class="cell hl">#1024</span><span class="cell hl">#1025</span><span class="sep">→</span><span class="cell q">KV slot numbers (pointers); tensors live in the pool (Lesson 30)</span></div>
  <div class="cells"><span class="lab">lock_ref</span><span class="cell">2</span><span class="sep">→</span><span class="cell q">running requests using this prefix; &gt;0 means no eviction (Lesson 32)</span></div>
  <div class="cells"><span class="lab">last_access</span><span class="cell">t₇</span><span class="sep">→</span><span class="cell q">most-recent access time, for LRU (Lesson 32)</span></div>
</div>

<table class="t">
  <tr><th>TreeNode field</th><th>Role</th><th>Why it's needed</th></tr>
  <tr><td class="mono">key</td><td>a run of token ids on the edge</td><td>packs un-branched tokens onto one edge, avoids a long chain</td></tr>
  <tr><td class="mono">children</td><td>first-token → child dict</td><td>O(1) pick of which child to descend into</td></tr>
  <tr><td class="mono">value</td><td>KV slot numbers (indices)</td><td>pointer into the pool; reuse = the same physical slots (Lesson 30)</td></tr>
  <tr><td class="mono">lock_ref</td><td>in-use reference count</td><td>&gt;0 locks it, so in-flight KV can't be evicted (Lesson 32)</td></tr>
  <tr><td class="mono">last_access_time</td><td>LRU timestamp</td><td>decides who is evicted first under memory pressure (Lesson 32)</td></tr>
</table>

<h2>match_prefix: walk down, split at the divergence</h2>
<p>A new request arrives and its token sequence first <strong>asks the tree "how long a run of me has already been computed?"</strong> That's <span class="mono">match_prefix</span>.
It starts at the root, <strong>uses the current token's first id to find a child in the children dict</strong>, then matches this run of tokens against that <strong>child's edge key, token by token</strong>.
Two outcomes: one, <strong>the whole edge matches</strong> — take that child's value (a run of indices) into the result, <strong>consume</strong> the matched tokens, and continue with the next token's first id;
two, <strong>only the front half of the edge matches before diverging</strong> — now we can't reuse the whole edge, so we call <span class="mono">_split_node</span> to <strong>cut that node into two levels at the divergence point</strong>.</p>
<p>The split is elegant: build a new parent whose <strong>key is the common prefix slice</strong> and whose value is that slice of indices; the original node becomes its child with <strong>key and value trimmed to just the diverging tail</strong>.
So the "common part" is extracted into an ancestor that many can share, and the "differing tails" become its branches. This step is the soul of the radix tree — <strong>sharing grows on demand at the instant of matching</strong>,
not planned ahead. <span class="mono">match_prefix</span> ultimately returns two things: the <strong>matched run of KV indices</strong> (reuse, no recompute) and the <strong>deepest matched node</strong> (the foothold for later insert and locking).</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Walk-down match</h4><p>Use the current token's first id to find a child in <span class="mono">children</span>, compare against its edge <span class="mono">key</span> token by token. Whole-edge hit → take its indices, consume tokens, descend further.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Diverge mid-edge → split</h4><p>Only the front half matches → <span class="mono">_split_node</span> cuts into two levels at the divergence: common prefix becomes the parent, original tail becomes the child, and the parent is now shareable.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Attach the diverging suffix → insert</h4><p>The leftover "new tail" after the match point → <span class="mono">insert</span> it as a new child node whose value is the indices obtained from writing this run's K/V into the pool.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Lock to protect</h4><p>From the deepest matched node walk <span class="mono">upward</span>, doing <span class="mono">lock_ref += 1</span> on each, so eviction (Lesson 32) can't touch this in-flight prefix; unlock node by node when the request finishes.</p></div></div>
</div>

<p>In the real code, the walk-down loop and the split look like this — note the <span class="mono">prefix_len &lt; len(child.key)</span> test, which is exactly the moment "only half the edge matched, must split":</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/radix_cache.py ::RadixCache._match_prefix_helper</span><span class="ln">walk-down match and on-demand split</span></div>
  <pre><span class="kw">def</span> _match_prefix_helper(self, node, key):
    child_key = key.child_key(self.page_size)   <span class="cm"># index by current token's first id</span>
    value = []
    <span class="kw">while</span> len(key) &gt; 0 <span class="kw">and</span> child_key <span class="kw">in</span> node.children.keys():
        child = node.children[child_key]
        prefix_len = child.key.match(key, page_size=self.page_size)
        <span class="kw">if</span> prefix_len &lt; len(child.key):       <span class="cm"># only half the edge matched -&gt; must split</span>
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)        <span class="cm"># take the common prefix's KV indices</span>
            node = new_node
            <span class="kw">break</span>
        <span class="kw">else</span>:                                  <span class="cm"># whole edge hit -&gt; take it, consume, descend</span>
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            child_key = key.child_key(self.page_size)
    <span class="kw">return</span> value, node                        <span class="cm"># matched indices + deepest node</span></pre>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="match_prefix descends from root along matching edges, highlighting the path of the longest matched prefix and the terminal node, while the unmatched tail is attached as a new branch">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">query: [you are an assistant X]</text>
    <rect x="20" y="40" width="470" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="40" y="60" class="mono" style="font-size:12px;fill:var(--accent-ink)">you are an assistant</text>
    <text x="360" y="60" class="mono" style="font-size:12px;fill:var(--faint)">X (no hit)</text>
    <rect x="20" y="150" width="74" height="44" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="57" y="177" text-anchor="middle" style="font-size:13px">root</text>
    <line x1="94" y1="172" x2="168" y2="172" style="stroke:var(--accent);stroke-width:3"/>
    <rect x="168" y="150" width="120" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="228" y="170" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--accent-ink)">[you are]</text>
    <text x="228" y="186" text-anchor="middle" style="font-size:10px;fill:var(--muted)">shared prefix</text>
    <line x1="288" y1="166" x2="360" y2="112" style="stroke:var(--accent);stroke-width:3"/>
    <line x1="288" y1="180" x2="360" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="360" y="90" width="160" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2.5"/>
    <text x="440" y="110" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--accent-ink)">[an assistant]</text>
    <text x="440" y="126" text-anchor="middle" style="font-size:10px;fill:var(--accent-ink);font-weight:700">longest hit · terminal</text>
    <rect x="360" y="210" width="120" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="420" y="237" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--teal)">[an expert]</text>
    <line x1="520" y1="112" x2="592" y2="112" style="stroke:var(--blue);stroke-width:2;stroke-dasharray:5 4"/>
    <rect x="592" y="90" width="160" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="672" y="110" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">[X …]</text>
    <text x="672" y="126" text-anchor="middle" style="font-size:10px;fill:var(--blue)">unmatched tail → new branch</text>
  </svg>
  <div class="figcap"><b>Fig 1 · match_prefix walks the radix tree to the longest hit</b> — starting at <span class="mono">root</span>, it descends edge by edge using the current token's first id: the matched edges form the highlighted path (<span class="mono">[you are] → [an assistant]</span>), and the terminal node is the deepest match returned; the leftover unmatched tail <span class="mono">[X …]</span> is later attached as a new branch.</div>
</div>

<p>The <span class="mono">_match_prefix_helper</span> above is the <strong>internal</strong> walk-down loop; the <strong>public entry</strong> callers actually use is <span class="mono">match_prefix</span>. It takes a <span class="mono">MatchPrefixParams</span> and returns a <span class="mono">MatchResult</span>: it first page-aligns the key, returns an empty result if disabled or the key is empty, otherwise delegates the descent to the helper (which may split a node along the way), and finally refreshes the access timestamps used by eviction:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.match_prefix</span><span class="ln">public entry: find the longest cached prefix, split a node if needed</span></div>
  <pre><span class="kw">def</span> match_prefix(self, params: MatchPrefixParams) -&gt; MatchResult:
    <span class="cm"># find the LONGEST cached prefix of params.key in the radix tree.</span>
    <span class="cm"># if the match ends INSIDE a stored segment, that node is split</span>
    <span class="cm"># once to expose the boundary; access timestamps are refreshed</span>
    <span class="cm"># (used later by eviction).</span>
    key = params.key
    <span class="kw">if</span> self.disable <span class="kw">or</span> len(key) == 0:
        <span class="kw">return</span> self._empty_match_result
    key = key.page_aligned(self.page_size)   <span class="cm"># guard first, then page-align</span>
    ...
    <span class="kw">return</span> MatchResult(device_indices=..., last_device_node=node)</pre>
</div>

<div class="card detail">
  <div class="tag">🧮 Concrete examples</div>
  <strong>Example 1: shared system preamble.</strong> Two requests both begin with the same 100-token system prompt. The first computes it and <span class="mono">insert</span>s it into the tree; when the second arrives, <span class="mono">match_prefix</span> hits all 100 tokens and returns their <strong>100 KV slot numbers</strong> in the pool — those 100 forward steps are skipped entirely, only each request's new suffix is computed.
  <strong>Example 2: alignment with page_size &gt; 1.</strong> Let <span class="mono">page_size = 16</span> and a key of length 100. <span class="mono">key.page_aligned(16)</span> first truncates the length to a multiple of 16, i.e. <strong>96</strong>, before matching; the trailing 4 tokens (less than a full page) don't take part in prefix reuse and are computed later with the new content.
</div>

<h2>insert and lock_ref: attach the new tail, lock the in-use prefix</h2>
<p>Matching only answers "how much already exists." What about the <strong>never-seen suffix</strong> left over? That's <span class="mono">insert</span>: it walks the same path to the match point and
<strong>attaches the diverging new tokens as a new child node</strong> under the deepest matched node, the new node's value being the run of indices from writing this run's K/V into the pool.
If insert also hits a "half edge" along the way, it likewise <span class="mono">_split_node</span> first then attaches — so match and insert share the same split logic, keeping the tree <strong>always canonical</strong>.</p>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="insert splits a node where two sequences diverge: one node held [A B C D]; a new sequence shares [A B] then diverges, so after the split [A B] becomes the parent and [C D] plus the new [X Y] become two children">
    <line x1="400" y1="30" x2="400" y2="300" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">before</text>
    <rect x="40" y="60" width="74" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="77" y="85" text-anchor="middle" style="font-size:13px">root</text>
    <line x1="77" y1="100" x2="160" y2="150" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="160" y="150" width="180" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:2"/>
    <text x="250" y="170" text-anchor="middle" class="mono" style="font-size:13px">[A B C D]</text>
    <text x="250" y="186" text-anchor="middle" style="font-size:10px;fill:var(--muted)">one node holds the whole run</text>
    <rect x="120" y="232" width="260" height="56" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="254" text-anchor="middle" style="font-size:11px;fill:var(--amber)">new sequence [A B X Y]</text>
    <text x="250" y="272" text-anchor="middle" style="font-size:11px;fill:var(--amber)">shares [A B], diverges at the 3rd token</text>
    <text x="420" y="26" style="font-weight:700;fill:var(--accent-ink)">after</text>
    <rect x="440" y="50" width="74" height="40" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="477" y="75" text-anchor="middle" style="font-size:13px">root</text>
    <line x1="477" y1="90" x2="600" y2="120" style="stroke:var(--accent);stroke-width:3"/>
    <rect x="540" y="120" width="120" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2.5"/>
    <text x="600" y="140" text-anchor="middle" class="mono" style="font-size:13px;fill:var(--accent-ink)">[A B]</text>
    <text x="600" y="156" text-anchor="middle" style="font-size:10px;fill:var(--accent-ink);font-weight:700">parent · stored once</text>
    <line x1="585" y1="164" x2="520" y2="220" style="stroke:var(--teal);stroke-width:2"/>
    <line x1="615" y1="164" x2="700" y2="220" style="stroke:var(--blue);stroke-width:2;stroke-dasharray:5 4"/>
    <rect x="450" y="220" width="120" height="44" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="510" y="240" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--teal)">[C D]</text>
    <text x="510" y="256" text-anchor="middle" style="font-size:10px;fill:var(--teal)">original node's tail</text>
    <rect x="640" y="220" width="120" height="44" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="700" y="240" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--blue)">[X Y]</text>
    <text x="700" y="256" text-anchor="middle" style="font-size:10px;fill:var(--blue)">new sequence's tail</text>
  </svg>
  <div class="figcap"><b>Fig 2 · insert splits a node where two sequences diverge</b> — before, one node holds the whole run <span class="mono">[A B C D]</span>; a new sequence <span class="mono">[A B X Y]</span> shares <span class="mono">[A B]</span> then diverges. <span class="mono">_split_node</span> cuts it into parent <span class="mono">[A B]</span> (the shared prefix, stored once) + child <span class="mono">[C D]</span> (the original tail) + a new sibling child <span class="mono">[X Y]</span>, so the shared prefix is held by both sequences.</div>
</div>

<p>Finally, the <strong>lock</strong>. When a request starts using a matched prefix, the engine walks <strong>upward from the deepest matched node all the way to the root</strong>, doing <span class="mono">lock_ref += 1</span> on each node (that's <span class="mono">inc_lock_ref</span>).
Why lock the whole chain upward? Because eviction reclaims from leaves upward (Lesson 32); as long as this prefix is in flight, <strong>none of its ancestors may be treated as an "evictable leaf"</strong>, or a running request would read a swath of reclaimed garbage slots.
On finish, <span class="mono">dec_lock_ref</span> decrements the counts back; once a node's <span class="mono">lock_ref</span> hits zero it becomes an eviction candidate again. In one line: <strong>lock_ref guards the iron rule that KV in use is never reclaimed</strong>.</p>

<h2>How sharing really happens: tree is index, pool is storage</h2>
<p>Tie three things together and it clicks. If two requests share a prefix, in <span class="mono">match_prefix</span> they <strong>walk the same ancestor nodes</strong>,
get the <strong>same run of values (indices)</strong>, so both attention reads hit the <strong>same physical slots</strong> in the pool — that's the whole truth of "sharing": not copying KV, but <strong>reusing pointers</strong>.
Once each generates a different next token, the paths <strong>diverge</strong> at some node into two distinct leaves, each new KV written to fresh slots in the pool and hung as its own new child node.
The tree owns "<strong>who shares with whom, up to which token</strong>"; the pool owns "<strong>where the KV actually lives</strong>" — index and storage fully separated, which is exactly why this design is both memory-thrifty and fast.</p>

<div class="flow">
  <div class="node hl"><div class="nt">root</div><div class="nd">empty prefix</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">shared ancestor</div><div class="nd">"You are an assistant…"<br>both walk it · same indices</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">leaf A</div><div class="nd">request A's own suffix<br>→ fresh slots in pool</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">leaf B</div><div class="nd">request B's own suffix<br>→ other slots in pool</div></div>
</div>

<div class="card key">
  <div class="tag">🔑 Key takeaways</div>
  <strong>① A TreeNode's value is KV slot numbers (indices), not KV tensors</strong> — the tree is the index, the pool is the storage (Lesson 30).
  <strong>② match_prefix descends via the children dict, matching edge by edge; on a half-edge match, _split_node cuts two levels at the divergence</strong>, lifting the common prefix into a shareable parent.
  <strong>③ insert attaches the diverging suffix as a new child</strong>, sharing the split logic with match to keep the tree canonical.
  <strong>④ inc/dec_lock_ref lock the whole chain upward from the matched node</strong>, guarding in-flight KV from eviction (Lesson 32).
  <strong>⑤ Sharing = two paths walk the same ancestor, take the same indices, point at the same physical slots</strong>, then grow separate leaves after diverging. Concept in Lesson 7, HiCache tiering in Lesson 31, cache-aware scheduling in Lesson 20.
</div>
"""}

LESSON_30 = {"zh": r"""
<p class="lead">
上一课我们读懂了基数树：它把"token 序列 → KV 槽位号"这张索引维护得井井有条，可是有一个问题一直被它<strong>故意回避</strong>——
那些真正占显存的 <span class="inline">K/V 张量</span>到底躺在哪？树里存的 <span class="mono">value</span> 只是一串<strong>槽位号（indices）</strong>，
而不是张量本身。这一课就揭开槽位号背后的真身：<strong>两个内存池</strong>，以及一条把请求映射到物理 K/V 的<strong>两级寻址</strong>链路。
读懂它，你就明白显存到底被谁、以什么粒度、在什么时刻分出去又收回来。
本课先建立"两个池 + 一个分配器"的整体地图，再逐一拆解它们各自的形状与职责，最后把这条链路严丝合缝地接回上一课的基数树，让"树存的到底是什么"这个悬念彻底落地。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把这套机制想成<strong>衣帽间 + 储物柜大厅</strong>：你进门把大衣交给服务员，他给你一张<strong>取衣牌</strong>，牌上印着"你的东西放在 12、37、58 号柜"——
  这张牌就是 <span class="mono">ReqToTokenPool</span>，它只记<strong>柜子编号</strong>，不装大衣。真正的大衣（K/V 张量）锁在<strong>那一排储物柜</strong>里，
  这排柜子就是 <span class="mono">token_to_kv</span> 池。发空柜、回收柜的<strong>服务员</strong>就是 <span class="mono">TokenToKVPoolAllocator</span>。
  妙处在于：两个人要存<strong>同一件共享的外套</strong>时，服务员可以把<strong>同一个柜号</strong>同时写进两张取衣牌——这正是前缀共享（第 29 课）。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话定调：<strong>树是索引，池是存储</strong>。第 29 课的树负责"谁和谁共享、共享到第几个 token"，而真正的 K/V 张量住在<strong>两个池</strong>里。
  寻址要走<strong>两跳</strong>：请求 →（<span class="mono">ReqToTokenPool</span>）→ 一串 token 槽位号 →（<span class="mono">token_to_kv</span> 池）→ 物理 K/V。
  池子在开机时按 <span class="mono">mem_fraction_static</span>（第 8 课）一次性预留，总槽位数 = 能装下多少 token = <strong>并发上限</strong>（第 4 课）。
  分页分配（第 6 课）、命中复用（第 29 课）、驱逐回收（第 32 课）全都围着这两个池转。
</div>

<h2>两个池，两跳寻址：K/V 究竟住在哪</h2>
<p>很多人第一次读内存子系统会困惑：明明只是"缓存 KV"，为什么要搞<strong>两个</strong>池？答案藏在"<strong>每条请求的私有账本</strong>"和"<strong>全局共享的大仓库</strong>"这对矛盾里。
<span class="mono">ReqToTokenPool</span> 是<strong>私有账本</strong>：它是一张固定大小的二维表，按 <span class="mono">req_pool_idx</span> 索引行，每一行记着"<strong>这条请求依次拥有哪些 token 槽位号</strong>"。
它<strong>不装任何 K/V 张量</strong>，只装下标——纯粹是<strong>每请求的记账</strong>。而 <span class="mono">MHATokenToKVPool</span> 才是<strong>大仓库</strong>：它为<strong>每一层</strong>各开一个巨大的张量，
按 token 槽位号索引，里面装的是<strong>实打实的 K 和 V</strong>。</p>
<p>于是寻址必然是<strong>两跳</strong>。第一跳：拿请求的 <span class="mono">req_pool_idx</span> 去 <span class="mono">ReqToTokenPool</span> 那一行，读出"我拥有的一串 token 槽位号"。
第二跳：拿这串槽位号去 <span class="mono">token_to_kv</span> 池，按下标取出（或写入）每一层对应的 K/V。<strong>请求 → 槽位号 → 物理 K/V</strong>，中间隔着两层<strong>间接</strong>。
这种"先查账本拿编号、再凭编号取货"的设计不是多此一举——正因为编号这层被独立出来，槽位才能<strong>非连续</strong>（分页，第 6 课）、还能<strong>被多条请求共享</strong>（基数树，第 29 课），
而每条请求的账本只管"哪些编号是我的"，丝毫不关心它们在仓库里是否连续、是否被别人也指着。</p>
<p>再把 <span class="mono">ReqToTokenPool</span> 这张表的形状说细一点，你会更踏实。它的底层就是一个 <span class="mono">torch.zeros((请求数+1, 最大上下文长度), dtype=int32)</span> 的二维张量：
<strong>行</strong>是请求（用 <span class="mono">req_pool_idx</span> 选行），<strong>列</strong>是这条请求第几个 token，<strong>格子里填的就是那个 token 在仓库里的槽位号</strong>。
为什么要预留 <strong>第 0 行作 padding</strong>？因为 CUDA Graph 的定长批次会把多余的 <span class="mono">req_pool_idx</span> 默认填 0，让这些"假请求"的读写都落到第 0 行这片<strong>无害的废格子</strong>里，
不污染真正的请求。<span class="mono">free_slots</span> 则是一份"还有哪些行号空着"的清单，<span class="mono">alloc</span> 从中切出行号发给新请求、<span class="mono">free</span> 把行号还回去——
注意这只是<strong>账本行的分配</strong>，和仓库里 token 槽位的分配是<strong>两套独立的簿记</strong>，别把它们搞混。这也再次印证了"账本归账本、仓库归仓库"的分层。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">入口</span><span class="name">一条在跑的请求</span></div><div class="ld">每条请求持有一个 <span class="mono">req_pool_idx</span>——它在私有账本里的<strong>行号</strong>。</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">① 账本</span><span class="name">ReqToTokenPool</span></div><div class="ld">按 <span class="mono">req_pool_idx</span> 取出这一行：<strong>我依次拥有的 token 槽位号列表</strong>。只记编号，不装张量。</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">② 编号</span><span class="name">token 槽位号 (indices)</span></div><div class="ld">一串指向大仓库的<strong>下标</strong>。可非连续（分页，第 6 课），可被多请求共享（第 29 课）。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">③ 仓库</span><span class="name">MHATokenToKVPool</span></div><div class="ld">为<strong>每一层</strong>各开一个大张量，按槽位号去 <span class="mono">k_buffer</span> / <span class="mono">v_buffer</span> 取或写。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">真身</span><span class="name">物理 K / V 张量</span></div><div class="ld">真正吃显存的就是这里。开机按 <span class="mono">mem_fraction_static</span>（第 8 课）一次性预留。</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="左边 ReqToTokenPool 是索引账本，按请求和位置查出 token 槽位号；右边 TokenToKVPool 按槽位号存每层的物理 K/V；一个箭头从左表的某个槽位号指向右边数据缓冲里的同一个槽位，体现两级间接寻址">
    <text x="20" y="26" style="font-weight:700;fill:var(--accent-ink)">ReqToTokenPool · 索引账本</text>
    <text x="20" y="44" style="fill:var(--muted);font-size:12px">行=请求 · 列=位置 · 格子=token 槽位号（只记编号）</text>
    <text x="142" y="78" text-anchor="middle" style="fill:var(--faint);font-size:12px">位0</text>
    <text x="212" y="78" text-anchor="middle" style="fill:var(--faint);font-size:12px">位1</text>
    <text x="282" y="78" text-anchor="middle" style="fill:var(--faint);font-size:12px">位2</text>
    <rect x="20" y="88" width="84" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="62" y="110" text-anchor="middle" style="font-size:12px">请求A</text>
    <rect x="110" y="88" width="64" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="142" y="110" text-anchor="middle" class="mono" style="font-size:12px">#12</text>
    <rect x="180" y="88" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="212" y="110" text-anchor="middle" class="mono" style="font-size:12px">#40</text>
    <rect x="250" y="88" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="282" y="110" text-anchor="middle" class="mono" style="font-size:12px">#41</text>
    <rect x="20" y="130" width="84" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="62" y="152" text-anchor="middle" style="font-size:12px">请求B</text>
    <rect x="110" y="130" width="64" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="142" y="152" text-anchor="middle" class="mono" style="font-size:12px">#12</text>
    <rect x="180" y="130" width="64" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="212" y="152" text-anchor="middle" class="mono" style="font-size:12px">#88</text>
    <rect x="250" y="130" width="64" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="282" y="152" text-anchor="middle" class="mono" style="font-size:12px">#90</text>
    <text x="430" y="26" style="font-weight:700;fill:var(--accent-ink)">TokenToKVPool · K/V 数据</text>
    <text x="430" y="44" style="fill:var(--muted);font-size:12px">每层一对 K/V 缓冲 · 按槽位号索引</text>
    <text x="430" y="108" style="fill:var(--muted);font-size:12px">第0层</text>
    <rect x="470" y="88" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="498" y="110" text-anchor="middle" class="mono" style="font-size:11px">#10</text>
    <rect x="530" y="88" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="558" y="110" text-anchor="middle" class="mono" style="font-size:11px">#11</text>
    <rect x="590" y="88" width="56" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="618" y="110" text-anchor="middle" class="mono" style="font-size:11px">#12</text>
    <rect x="650" y="88" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="678" y="110" text-anchor="middle" class="mono" style="font-size:11px">#13</text>
    <text x="430" y="160" style="fill:var(--muted);font-size:12px">第1层</text>
    <rect x="470" y="140" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="498" y="162" text-anchor="middle" class="mono" style="font-size:11px">#10</text>
    <rect x="530" y="140" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="558" y="162" text-anchor="middle" class="mono" style="font-size:11px">#11</text>
    <rect x="590" y="140" width="56" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="618" y="162" text-anchor="middle" class="mono" style="font-size:11px">#12</text>
    <rect x="650" y="140" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="678" y="162" text-anchor="middle" class="mono" style="font-size:11px">#13</text>
    <line x1="316" y1="105" x2="586" y2="105" style="stroke:var(--accent);stroke-width:2;stroke-dasharray:5 4"/>
    <polygon points="578,100 590,105 578,110" style="fill:var(--accent)"/>
    <text x="450" y="98" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">槽位号 #12 → 指向数据（间接寻址）</text>
    <text x="430" y="206" style="fill:var(--faint);font-size:12px">同一个槽位号 #12 在每层各取一格 K/V，一次分配跨层通用</text>
  </svg>
  <div class="figcap"><b>图 3 · 两个池：ReqToToken（索引）+ TokenToKV（数据）</b> — 左表按请求行与位置查出 token 槽位号，右边按这些槽位号在每层的 K/V 缓冲里取数据；箭头是那条"先查编号、再凭编号取货"的两级间接链路。</div>
</div>

<h2>分配器：开机预留、入批发槽、驱逐回收</h2>
<p>谁来决定哪些槽位是空的、该发哪一个？是 <span class="mono">TokenToKVPoolAllocator</span>（基类 <span class="mono">BaseTokenToKVPoolAllocator</span>）。它就是衣帽间那位服务员，手里攥着一份<strong>空闲页清单</strong>。
当一批请求被准入（admit）要做前向时，调度器向分配器<strong>申领</strong>所需的 token 槽位——分配器以<strong>页为单位</strong>（固定大小，第 6 课）从空闲清单里切出来发给它们；
请求结束或被驱逐时，再把这些槽位<strong>归还</strong>清单。<span class="mono">available_size</span> 返回的就是"还剩多少页 × 页大小"，也就是<strong>此刻还能再接多少 token</strong>。</p>
<p>关键在于<strong>总量是开机就钉死的</strong>。引擎启动时，按 <span class="mono">mem_fraction_static</span>（第 8 课）算出能拿来做 KV 缓存的那块显存，再除以"每个 token 每层的 K/V 字节数"，
得到<strong>总槽位数 = 池子能装下多少 token</strong>。这个数字直接就是<strong>并发上限</strong>（第 4 课）：所有在跑请求的 token 加起来不能超过它。一旦逼近上限，要么排队、要么靠驱逐（第 32 课）腾位。它把抽象的"显存够不够"翻译成了一个具体、可数的整数，调度器每一步都在拿它做减法心算。
所以分配器虽小，却是<strong>显存预算的总闸门</strong>——发得出槽就能进批，发不出就只能等。它和池子的分工很清爽：<strong>池子是那排柜子本身，分配器是管钥匙的人</strong>。</p>
<p>分配器内部还有两个值得一提的细节。其一是<strong>空闲页与待释放页分离</strong>：<span class="mono">free_pages</span> 是当下可直接发出的空闲页，<span class="mono">release_pages</span> 是刚被归还、还没并回主清单的页；
<span class="mono">available_size</span> 把两者加起来 ×页大小，才是真实可用量。这样分两段是为了把"归还"做成<strong>批量、惰性</strong>的，避免每释放一个槽就立刻做一次昂贵的排序合并。
其二是 <span class="mono">need_sort</span> 这个开关：某些注意力后端要求<strong>同一请求的槽位尽量连续</strong>，于是回收时把空闲页排序、让后续分配尽量挑出连号的页；不需要时就跳过排序、图个快。
你不必记住这些字段名，但要记住<strong>分配器不是一个简单的栈</strong>——它在"分配要快"和"槽位要尽量规整"之间做了不少工程权衡，这些权衡最终都服务于上层注意力内核能否高效读取（第 6 课的分页正是同一动机）。</p>

<div class="flow">
  <div class="node hl"><div class="nt">入批 alloc</div><div class="nd">分配器以页为单位<br>发出空闲 token 槽位</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">写入 write</div><div class="nd">前向把每层 K/V<br>写到这些槽位</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">命中复用</div><div class="nd">第 29 课的树指向<br>同一批槽位 · 零拷贝</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">驱逐 free</div><div class="nd">第 32 课把槽位<br>还回分配器清单</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 340" role="img" aria-label="一排固定大小的 KV 槽位，从 t0 到 t3 四个时刻：请求 X 先占两页、请求 Y 再占两页，占用上升；随后 X 完成把两页归还空闲清单，占用下降；空闲与占用分别标色">
    <text x="20" y="26" style="font-weight:700;fill:var(--accent-ink)">按页分配、用完即还（占用随时间变化）</text>
    <rect x="20" y="40" width="16" height="14" rx="3" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="44" y="52" style="fill:var(--muted);font-size:12px">空闲</text>
    <rect x="100" y="40" width="16" height="14" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="124" y="52" style="fill:var(--muted);font-size:12px">请求X 占用</text>
    <rect x="210" y="40" width="16" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="234" y="52" style="fill:var(--muted);font-size:12px">请求Y 占用</text>
    <text x="24" y="100" style="font-weight:700;fill:var(--muted)">t0</text>
    <rect x="70" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="140" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="210" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="280" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="350" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="100" style="fill:var(--faint);font-size:12px">全部空闲 · 占用 0/6</text>
    <text x="24" y="150" style="font-weight:700;fill:var(--muted)">t1</text>
    <rect x="70" y="128" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="140" y="128" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="210" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="280" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="350" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="150" style="fill:var(--blue);font-size:12px">X 申领 2 页 · 占用 2/6 ↑</text>
    <text x="24" y="200" style="font-weight:700;fill:var(--muted)">t2</text>
    <rect x="70" y="178" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="140" y="178" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="210" y="178" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="280" y="178" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="350" y="178" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="178" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="200" style="fill:var(--teal);font-size:12px">Y 再领 2 页 · 占用 4/6 ↑</text>
    <text x="24" y="250" style="font-weight:700;fill:var(--muted)">t3</text>
    <rect x="70" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3"/>
    <rect x="140" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3"/>
    <rect x="210" y="228" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="280" y="228" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="350" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="250" style="fill:var(--amber);font-size:12px">X 完成，2 页归还清单 · 占用 2/6 ↓</text>
    <text x="70" y="296" style="fill:var(--faint);font-size:12px">虚线框 = 刚归还、回到 free list 的页；分配以页为固定粒度，回收只改清单、不动张量。</text>
  </svg>
  <div class="figcap"><b>图 4 · 按页分配、用完即还（占用随时间变化）</b> — 同一排固定大小的 KV 页随时间被请求陆续申领（占用上升），请求完成后其页归还空闲清单（占用下降）；空闲与占用分别标色，总页数开机即钉死。</div>
</div>

<h2>为什么拆成两张表，而不是合成一张</h2>
<p>把私有账本和共享仓库<strong>分开</strong>，是这套设计最值得玩味的取舍。设想若硬要合成一张"请求直接存自己的 K/V"的表，会立刻撞上两堵墙。
第一堵：<strong>分页</strong>。一条请求的 token 槽位在仓库里往往是<strong>东一块西一块</strong>的（分页分配，第 6 课），不连续。账本这层正好用来<strong>把散落的编号串成一条有序列表</strong>，
让上层以为"我的 KV 是连成一串的"，底层却自由地把它们摆在任意空页里。第二堵：<strong>共享</strong>。两条请求若前缀相同，它们应当指向<strong>同一批物理槽位</strong>（第 29 课），
若 K/V 直接长在请求的私有表里，就<strong>没法共享</strong>——你总不能让两个人的私有储物柜是同一个。</p>
<p>拆开后两难尽解：<strong>账本（ReqToTokenPool）记"哪些编号是我的"，仓库（token_to_kv 池）是一个所有请求共用的大 arena</strong>。
编号可以非连续、可以被多份账本同时引用，而仓库只需老老实实按编号存取，不必关心谁在用、用了几份。这正是"<strong>索引与存储分离</strong>"在内存层的又一次体现——
和第 29 课"树是索引、池是存储"是同一条哲学，只是这里把"池"再细分成了<strong>记账的池</strong>和<strong>存货的池</strong>。下面这张表把两者一字排开。</p>
<p>这种分层还带来一个常被忽略却极其关键的好处：<strong>共享与回收变成纯粹的"改编号"操作，不碰一字节张量</strong>。前缀命中时，引擎只是把命中的那串槽位号<strong>抄进</strong>新请求的账本行；
驱逐时，只是把槽位号从某节点的 <span class="mono">value</span> 里<strong>摘下来还给分配器</strong>。无论共享还是回收，<strong>仓库里的 K/V 张量一动不动</strong>——既不复制，也不搬家。
设想若 K/V 直接长在请求的私有表里，共享就得复制张量、回收就得搬运或清零，每一步都是昂贵的显存带宽开销。正因为把"<strong>谁拥有</strong>"（账本里的编号）和"<strong>东西本体</strong>"（仓库里的张量）彻底解耦，
这些高频操作才退化成<strong>轻量的整数游戏</strong>。一句话：<strong>编号可以随意复制、转让、回收，张量始终安坐原地</strong>——这是整套内存子系统又快又省的根本秘密，也是它能从容支撑成千上万并发请求的底气所在。</p>

<table class="t">
  <tr><th>池 / 组件</th><th>映射 / 装什么</th><th>按什么索引</th></tr>
  <tr><td class="mono">ReqToTokenPool</td><td>每条请求 → 它依次拥有的 token 槽位号列表（只记编号）</td><td class="mono">req_pool_idx</td></tr>
  <tr><td class="mono">MHATokenToKVPool</td><td>每层的物理 K / V 张量（真正吃显存）</td><td class="mono">token 槽位号</td></tr>
  <tr><td class="mono">TokenToKVPoolAllocator</td><td>空闲页清单：发槽 / 回收槽</td><td>页（第 6 课）</td></tr>
</table>

<h2>接回第 29 课：树存的是 indices，不是张量</h2>
<p>现在可以把上一课那句话彻底落地了。基数树 <span class="mono">TreeNode</span> 的 <span class="mono">value</span> 字段，装的<strong>就是这里说的 token 槽位号（indices）</strong>——
它是<strong>指向 token_to_kv 池的下标</strong>，<strong>不是 K/V 张量本身</strong>。所以树是纯粹的<strong>索引与去重</strong>层，池才是<strong>存储</strong>层。
两条请求共享前缀，本质是它们在树上拿到<strong>同一串 indices</strong>，于是注意力都去池里读<strong>同一批物理槽位</strong>——共享的部分<strong>零额外显存</strong>。
当某段前缀被驱逐（第 32 课），做的事就是把它 <span class="mono">value</span> 里那串槽位号<strong>还给分配器</strong>，让别的 token 来占。这就是为什么第 29 课反复强调"<strong>节点装的是指针、不是张量</strong>"——那句话的全部分量，要到这一课见了真正的池子才掂得出来。</p>
<p>把链路完整地连起来：一条请求来了，<span class="mono">match_prefix</span> 在树上命中一段前缀、拿回一串 indices（复用），未命中的尾巴则向分配器<strong>申领新槽位</strong>、
把算出的 K/V <span class="mono">write</span> 进池子、再把这串新 indices 一并记进自己的 <span class="mono">ReqToTokenPool</span> 行、并 <span class="mono">insert</span> 回树供后来者复用。
读、写、共享、回收，全部以"槽位号"为通货在<strong>账本、仓库、树</strong>三者间流转。看懂这点，你就握住了 SGLang 显存管理的<strong>总枢纽</strong>。下面这格把"两请求指向共享槽位"画给你看。</p>
<p>顺便厘清一个常见误解：仓库为什么要<strong>按层各开一个张量</strong>，而不是把所有层堆成一个超大张量？因为注意力是<strong>逐层计算</strong>的——第 L 层只读写第 L 层的 K/V，把每层独立成一个 <span class="mono">k_buffer[L]</span> / <span class="mono">v_buffer[L]</span>，
内核就能直接用槽位号在该层张量里寻址，既清晰又利于各层并行与流水。但<strong>槽位号在所有层之间是共用的</strong>：同一个 token 槽位 #12，在第 0 层、第 1 层……第 N 层都用<strong>同一个下标</strong>去各自的 buffer 取数。
也就是说，账本里那串编号是"<strong>跨层通用的门牌号</strong>"，一次分配、处处可用。这正是为什么释放一个 token 槽位就等于<strong>一次性释放它在所有层占的那块 K/V</strong>——编号回收了，每层对应的格子就同时空了出来，干净利落。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>两级寻址 + 前缀共享</b>：两条请求的账本里写着同一个槽位号（高亮），于是指向池里同一批物理 K/V</div>
  <div class="cells"><span class="lab">请求 A 账本</span><span class="cell hl">#12</span><span class="cell">#40</span><span class="cell">#41</span><span class="sep">→</span><span class="cell q">A 拥有的 token 槽位号（按 req_pool_idx 取这一行）</span></div>
  <div class="cells"><span class="lab">请求 B 账本</span><span class="cell hl">#12</span><span class="cell">#88</span><span class="sep">→</span><span class="cell q">B 也指向 <span class="mono">#12</span>——共享前缀，同一个编号写进两张账本</span></div>
  <div class="cells"><span class="lab">token_to_kv 池</span><span class="cell hl">槽#12 的 K/V</span><span class="sep">→</span><span class="cell q">物理 K/V 只此一份，A 与 B 共读 · 零额外显存（第 29 课）</span></div>
  <div class="cells"><span class="lab">分配器</span><span class="cell">空闲页…</span><span class="sep">→</span><span class="cell q">发出 <span class="mono">#40/#41/#88</span> 等新槽；驱逐时 <span class="mono">#12</span> 归还（第 32 课）</span></div>
</div>

<p>真实代码里，账本就是一张 <span class="mono">int32</span> 二维表，仓库就是每层一个大张量——朴素得几乎不像"高科技"，但正是这份朴素让两级寻址既快又省：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/memory_pool.py ::ReqToTokenPool</span><span class="ln">私有账本 + 每层 K/V 仓库</span></div>
  <pre><span class="kw">class</span> ReqToTokenPool:
    <span class="cm"># 每条请求 → 它依次拥有的 token 槽位号；只记编号，不装张量</span>
    <span class="kw">def</span> __init__(self, size, max_context_len, device, ...):
        self.size = size
        <span class="cm"># [请求数+1, 最大上下文长度] 的 int32 表，按 req_pool_idx 索引行</span>
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device)
        self.free_slots = list(range(<span class="st">1</span>, self._alloc_size))

    <span class="kw">def</span> write(self, indices, values):
        self.req_to_token[indices] = values   <span class="cm"># 把这条请求的一串槽位号写进它那一行</span>

<span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="kw">def</span> _create_buffers(self):
        <span class="cm"># 每层一个大张量 [槽位数, 头数, 头维]——真正的 K / V 就躺在这里</span>
        self.k_buffer = [torch.zeros((self.size + self.page_size,
                         self.head_num, self.head_dim), ...)
                         <span class="kw">for</span> _ <span class="kw">in</span> range(self.layer_num)]
        self.v_buffer = [torch.zeros((self.size + self.page_size,
                         self.head_num, self.v_head_dim), ...)
                         <span class="kw">for</span> _ <span class="kw">in</span> range(self.layer_num)]</pre>
</div>

<p>再单独把<strong>仓库本身</strong>的真身亮出来——它就是 <span class="mono">MHATokenToKVPool</span>：每一层各持有<strong>一个 K 张量、一个 V 张量</strong>，写入时按槽位号 <span class="mono">loc</span> 把这个 token 的 K/V 落到对应那一格：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::MHATokenToKVPool</span><span class="ln">真正的 KV 数据仓库：每层一对 K/V 缓冲</span></div>
  <pre><span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="cm"># 真正的 KV 数据仓库：每层各一个 K、一个 V 张量，</span>
    <span class="cm"># 形状均为 [size, num_kv_heads, head_dim]。ReqToTokenPool 把</span>
    <span class="cm"># 一条请求的各个位置 -&gt; 指向这些缓冲的槽位号。</span>
    <span class="kw">def</span> __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, ...):
        self.k_buffer = [...]   <span class="cm"># 按层排列的 K 张量列表</span>
        self.v_buffer = [...]   <span class="cm"># 按层排列的 V 张量列表</span>
    <span class="kw">def</span> set_kv_buffer(self, layer, loc, cache_k, cache_v):
        <span class="cm"># 把这个 token 的 K/V 写到该层的 `loc` 槽位</span>
        self.k_buffer[layer.layer_id][loc] = cache_k
        self.v_buffer[layer.layer_id][loc] = cache_v</pre>
</div>

<p>举个具体的数：一个 <strong>32 层</strong>的模型，这个池子就持有 <strong>32 个 K 缓冲 + 32 个 V 缓冲</strong>（每层一对）。每个 KV"槽位"存的是<strong>某一个 token、在某一层</strong>的 K 和 V；而 <span class="mono">ReqToTokenPool</span> 里某请求的那一行，给出的正是它各个位置对应的<strong>槽位号列表</strong>——比如 <span class="mono">[#12, #40, #41]</span> 就表示这条请求的第 0/1/2 个 token 分别落在槽位 12、40、41，每个槽位号一次性覆盖全部 32 层。</p>

<div class="card key">
  <div class="tag">🔑 本课要点</div>
  <strong>① 两个池，两跳寻址</strong>：请求 →（ReqToTokenPool）→ token 槽位号 →（token_to_kv 池）→ 物理 K/V。账本只记编号，仓库才装张量。
  <strong>② ReqToTokenPool 是每请求的私有账本</strong>，按 <span class="mono">req_pool_idx</span> 索引；<strong>MHATokenToKVPool 是每层的共享大仓库</strong>，按槽位号索引。
  <strong>③ TokenToKVPoolAllocator 发槽 / 回收槽</strong>，按页（第 6 课）；总槽位在开机按 <span class="mono">mem_fraction_static</span>（第 8 课）钉死 = 并发上限（第 4 课）。
  <strong>④ 第 29 课树里的 value 就是这些 indices，不是张量</strong>——树是索引、池是存储；两请求共享前缀 = 同一串槽位号 = 同一批物理 K/V，零额外显存。
  <strong>⑤ 拆成两张表是为了支持非连续（分页）与跨请求共享（基数树）</strong>；驱逐（第 32 课）就是把槽位号还给分配器。
</div>
""",
             "en": r"""
<p class="lead">
Last lesson we read the radix tree: it keeps the "token sequence → KV slot number" index neat — yet it kept <strong>dodging</strong> one question:
where do the <span class="inline">K/V tensors</span> that actually eat memory live? A node's <span class="mono">value</span> is just a run of <strong>slot numbers (indices)</strong>,
not the tensors themselves. This lesson lifts the lid on what those slot numbers point at: <strong>two memory pools</strong>, and a <strong>two-hop indirection</strong>
mapping a request to its physical K/V. Grasp it and you'll see exactly who hands out GPU memory, at what granularity, and when it's given out and taken back.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture a <strong>coat-check + locker room</strong>: you hand your coat to the attendant and get a <strong>claim ticket</strong> printed with "your stuff is in lockers 12, 37, 58" —
  that ticket is <span class="mono">ReqToTokenPool</span>; it lists <strong>locker numbers only</strong>, never the coats. The actual coats (K/V tensors) sit in <strong>that bank of lockers</strong>,
  which is the <span class="mono">token_to_kv</span> pool. The <strong>attendant</strong> handing out and reclaiming lockers is <span class="mono">TokenToKVPoolAllocator</span>.
  The trick: when two people store <strong>the same shared coat</strong>, the attendant can write <strong>the same locker number</strong> onto both tickets — that's prefix sharing (Lesson 29).
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  One line: <strong>the tree is the index, the pools are the storage</strong>. Lesson 29's tree owns "who shares with whom, up to which token"; the real K/V tensors live in <strong>two pools</strong>.
  Addressing takes <strong>two hops</strong>: request →(<span class="mono">ReqToTokenPool</span>)→ a run of token slot ids →(<span class="mono">token_to_kv</span> pool)→ physical K/V.
  The pools are pre-allocated once at startup, sized by <span class="mono">mem_fraction_static</span> (Lesson 8); total slots = how many tokens fit = the <strong>concurrency ceiling</strong> (Lesson 4).
  Paging (Lesson 6), reuse on hit (Lesson 29), and eviction (Lesson 32) all revolve around these two pools.
</div>

<h2>Two pools, two hops: where K/V actually lives</h2>
<p>First-timers in the memory subsystem ask: it's "just KV caching," why <strong>two</strong> pools? The answer hides in the tension between a "<strong>per-request private ledger</strong>" and a "<strong>globally shared warehouse</strong>."
<span class="mono">ReqToTokenPool</span> is the <strong>private ledger</strong>: a fixed-size 2-D table indexed by <span class="mono">req_pool_idx</span>, each row recording "<strong>which token slot numbers this request owns, in order</strong>."
It holds <strong>no K/V tensors</strong>, only indices — pure <strong>per-request bookkeeping</strong>. <span class="mono">MHATokenToKVPool</span> is the <strong>warehouse</strong>: it opens a giant tensor <strong>per layer</strong>,
indexed by token slot, holding the <strong>actual K and V</strong>.</p>
<p>So addressing is necessarily <strong>two hops</strong>. Hop one: take the request's <span class="mono">req_pool_idx</span> into the <span class="mono">ReqToTokenPool</span> row and read "the run of token slot numbers I own."
Hop two: take that run into the <span class="mono">token_to_kv</span> pool to fetch (or write) the per-layer K/V by index. <strong>Request → slot ids → physical K/V</strong>, with two layers of <strong>indirection</strong> in between.
This "look up the ledger for numbers, then fetch goods by number" is not redundant — precisely because the numbering layer is split out, slots can be <strong>non-contiguous</strong> (paging, Lesson 6) and <strong>shared across requests</strong> (the radix tree, Lesson 29),
while each request's ledger only tracks "which numbers are mine," indifferent to whether they're contiguous in the warehouse or also pointed at by others.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Entry</span><span class="name">a running request</span></div><div class="ld">Each request holds a <span class="mono">req_pool_idx</span> — its <strong>row number</strong> in the private ledger.</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">① Ledger</span><span class="name">ReqToTokenPool</span></div><div class="ld">Read the row by <span class="mono">req_pool_idx</span>: <strong>the list of token slot numbers I own, in order</strong>. Numbers only, no tensors.</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">② Numbers</span><span class="name">token slot ids (indices)</span></div><div class="ld">A run of <strong>indices</strong> into the warehouse. May be non-contiguous (paging, Lesson 6) and shared by many requests (Lesson 29).</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">③ Warehouse</span><span class="name">MHATokenToKVPool</span></div><div class="ld">Opens a big tensor <strong>per layer</strong>; fetch or write into <span class="mono">k_buffer</span> / <span class="mono">v_buffer</span> by slot.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Real thing</span><span class="name">physical K / V tensors</span></div><div class="ld">What truly eats GPU memory. Pre-allocated once by <span class="mono">mem_fraction_static</span> (Lesson 8).</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Left, ReqToTokenPool is the index ledger that maps a request row and position to a token slot number; right, TokenToKVPool stores the per-layer physical K/V indexed by those slot numbers; an arrow runs from a slot number in the left table into the same slot in the right data buffer, showing two-hop indirection">
    <text x="20" y="26" style="font-weight:700;fill:var(--accent-ink)">ReqToTokenPool · index ledger</text>
    <text x="20" y="44" style="fill:var(--muted);font-size:12px">rows=requests · cols=positions · cell=token slot id (numbers only)</text>
    <text x="142" y="78" text-anchor="middle" style="fill:var(--faint);font-size:12px">pos0</text>
    <text x="212" y="78" text-anchor="middle" style="fill:var(--faint);font-size:12px">pos1</text>
    <text x="282" y="78" text-anchor="middle" style="fill:var(--faint);font-size:12px">pos2</text>
    <rect x="20" y="88" width="84" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="62" y="110" text-anchor="middle" style="font-size:12px">req A</text>
    <rect x="110" y="88" width="64" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="142" y="110" text-anchor="middle" class="mono" style="font-size:12px">#12</text>
    <rect x="180" y="88" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="212" y="110" text-anchor="middle" class="mono" style="font-size:12px">#40</text>
    <rect x="250" y="88" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="282" y="110" text-anchor="middle" class="mono" style="font-size:12px">#41</text>
    <rect x="20" y="130" width="84" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="62" y="152" text-anchor="middle" style="font-size:12px">req B</text>
    <rect x="110" y="130" width="64" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="142" y="152" text-anchor="middle" class="mono" style="font-size:12px">#12</text>
    <rect x="180" y="130" width="64" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="212" y="152" text-anchor="middle" class="mono" style="font-size:12px">#88</text>
    <rect x="250" y="130" width="64" height="34" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="282" y="152" text-anchor="middle" class="mono" style="font-size:12px">#90</text>
    <text x="430" y="26" style="font-weight:700;fill:var(--accent-ink)">TokenToKVPool · K/V data</text>
    <text x="430" y="44" style="fill:var(--muted);font-size:12px">a K/V buffer pair per layer · indexed by slot id</text>
    <text x="430" y="108" style="fill:var(--muted);font-size:12px">layer 0</text>
    <rect x="470" y="88" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="498" y="110" text-anchor="middle" class="mono" style="font-size:11px">#10</text>
    <rect x="530" y="88" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="558" y="110" text-anchor="middle" class="mono" style="font-size:11px">#11</text>
    <rect x="590" y="88" width="56" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="618" y="110" text-anchor="middle" class="mono" style="font-size:11px">#12</text>
    <rect x="650" y="88" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="678" y="110" text-anchor="middle" class="mono" style="font-size:11px">#13</text>
    <text x="430" y="160" style="fill:var(--muted);font-size:12px">layer 1</text>
    <rect x="470" y="140" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="498" y="162" text-anchor="middle" class="mono" style="font-size:11px">#10</text>
    <rect x="530" y="140" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="558" y="162" text-anchor="middle" class="mono" style="font-size:11px">#11</text>
    <rect x="590" y="140" width="56" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:2"/>
    <text x="618" y="162" text-anchor="middle" class="mono" style="font-size:11px">#12</text>
    <rect x="650" y="140" width="56" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="678" y="162" text-anchor="middle" class="mono" style="font-size:11px">#13</text>
    <line x1="316" y1="105" x2="586" y2="105" style="stroke:var(--accent);stroke-width:2;stroke-dasharray:5 4"/>
    <polygon points="578,100 590,105 578,110" style="fill:var(--accent)"/>
    <text x="450" y="92" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">slot id #12 → points at data (indirection)</text>
    <text x="430" y="206" style="fill:var(--faint);font-size:12px">slot #12 picks one K/V cell per layer — reused across layers</text>
  </svg>
  <div class="figcap"><b>Fig 3 · Two pools: ReqToToken (index) + TokenToKV (data)</b> — the left table looks up a token slot number by request row and position; the right side fetches data from each layer's K/V buffer by those slot numbers; the arrow is the "look up the number, then fetch by number" two-hop indirection.</div>
</div>

<h2>The allocator: pre-reserve, hand out on admit, reclaim on evict</h2>
<p>Who decides which slots are free and which one to hand out? <span class="mono">TokenToKVPoolAllocator</span> (base <span class="mono">BaseTokenToKVPoolAllocator</span>). It's the coat-check attendant, holding a <strong>free-page list</strong>.
When a batch of requests is admitted to do a forward, the scheduler <strong>requests</strong> the token slots it needs — the allocator carves them out of the free list <strong>by page</strong> (fixed size, Lesson 6) and hands them over;
on finish or eviction the slots are <strong>returned</strong> to the list. <span class="mono">available_size</span> returns "free pages × page size," i.e. <strong>how many more tokens can still be admitted</strong> right now.</p>
<p>The crux: <strong>the total is nailed down at startup</strong>. At launch, the engine computes the GPU memory available for KV cache from <span class="mono">mem_fraction_static</span> (Lesson 8), divides by "K/V bytes per token per layer,"
and gets <strong>total slots = how many tokens the pool fits</strong>. That number is the <strong>concurrency ceiling</strong> (Lesson 4): the tokens of all running requests can't exceed it. Near the ceiling you either queue or evict (Lesson 32) to free room.
So the small allocator is the <strong>master valve of the memory budget</strong> — if it can hand out slots you join the batch, if not you wait. Its division of labor with the pool is clean: <strong>the pool is the bank of lockers, the allocator is who holds the keys</strong>.</p>

<div class="flow">
  <div class="node hl"><div class="nt">admit alloc</div><div class="nd">allocator hands out<br>free token slots by page</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">write</div><div class="nd">forward writes each layer's<br>K/V into those slots</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">reuse on hit</div><div class="nd">Lesson 29's tree points at<br>the same slots · zero copy</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">evict free</div><div class="nd">Lesson 32 returns the slots<br>to the allocator's list</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 340" role="img" aria-label="A row of fixed-size KV slots across four timesteps t0 to t3: request X claims two pages first, request Y claims two more, occupancy rises; then X finishes and returns its two pages to the free list, occupancy drops; free and used are color coded">
    <text x="20" y="26" style="font-weight:700;fill:var(--accent-ink)">Paged alloc on growth, free on finish (occupancy over time)</text>
    <rect x="20" y="40" width="16" height="14" rx="3" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="44" y="52" style="fill:var(--muted);font-size:12px">free</text>
    <rect x="100" y="40" width="16" height="14" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="124" y="52" style="fill:var(--muted);font-size:12px">used by X</text>
    <rect x="210" y="40" width="16" height="14" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="234" y="52" style="fill:var(--muted);font-size:12px">used by Y</text>
    <text x="24" y="100" style="font-weight:700;fill:var(--muted)">t0</text>
    <rect x="70" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="140" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="210" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="280" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="350" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="78" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="100" style="fill:var(--faint);font-size:12px">all free · used 0/6</text>
    <text x="24" y="150" style="font-weight:700;fill:var(--muted)">t1</text>
    <rect x="70" y="128" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="140" y="128" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="210" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="280" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="350" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="128" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="150" style="fill:var(--blue);font-size:12px">X claims 2 pages · used 2/6 ↑</text>
    <text x="24" y="200" style="font-weight:700;fill:var(--muted)">t2</text>
    <rect x="70" y="178" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="140" y="178" width="64" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="210" y="178" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="280" y="178" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="350" y="178" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="178" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="200" style="fill:var(--teal);font-size:12px">Y claims 2 more · used 4/6 ↑</text>
    <text x="24" y="250" style="font-weight:700;fill:var(--muted)">t3</text>
    <rect x="70" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3"/>
    <rect x="140" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:4 3"/>
    <rect x="210" y="228" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="280" y="228" width="64" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="350" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="420" y="228" width="64" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="250" style="fill:var(--amber);font-size:12px">X finishes, 2 pages returned · used 2/6 ↓</text>
    <text x="70" y="296" style="fill:var(--faint);font-size:12px">dashed = pages just returned to the free list; allocation is per fixed-size page, freeing only edits the list, never the tensors.</text>
  </svg>
  <div class="figcap"><b>Fig 4 · Paged alloc on growth, free on finish (occupancy over time)</b> — the same row of fixed-size KV pages is claimed by requests as they decode (occupancy rises), then a finished request's pages return to the free list (occupancy drops); free vs used are color coded, total pages nailed at startup.</div>
</div>

<h2>Why split into two tables, not merge into one</h2>
<p>Splitting the private ledger from the shared warehouse is the design's most instructive trade-off. Imagine forcing a single "request stores its own K/V" table — you hit two walls at once.
Wall one: <strong>paging</strong>. A request's token slots are usually scattered <strong>here and there</strong> in the warehouse (paged allocation, Lesson 6), non-contiguous. The ledger layer is exactly what <strong>threads the scattered numbers into an ordered list</strong>,
letting the upper layer believe "my KV is one contiguous run" while the lower layer freely places them in any free pages. Wall two: <strong>sharing</strong>. Two requests with the same prefix should point at the <strong>same physical slots</strong> (Lesson 29);
if K/V grew directly inside a request's private table, sharing would be <strong>impossible</strong> — you can't make two people's private lockers be the same one.</p>
<p>Split, both dilemmas dissolve: <strong>the ledger (ReqToTokenPool) records "which numbers are mine," the warehouse (token_to_kv pool) is one big arena shared by all requests</strong>.
Numbers may be non-contiguous and referenced by many ledgers at once, while the warehouse just stores/fetches by number, indifferent to who uses it or how many times. This is "<strong>index vs. storage separation</strong>" appearing again at the memory layer —
the same philosophy as Lesson 29's "tree is index, pool is storage," only here the "pool" is further split into a <strong>bookkeeping pool</strong> and a <strong>storage pool</strong>. The table below lays both out side by side.</p>

<table class="t">
  <tr><th>Pool / component</th><th>Maps / holds what</th><th>Indexed by</th></tr>
  <tr><td class="mono">ReqToTokenPool</td><td>each request → the list of token slot numbers it owns (numbers only)</td><td class="mono">req_pool_idx</td></tr>
  <tr><td class="mono">MHATokenToKVPool</td><td>per-layer physical K / V tensors (what really eats memory)</td><td class="mono">token slot id</td></tr>
  <tr><td class="mono">TokenToKVPoolAllocator</td><td>free-page list: hand out / reclaim slots</td><td>page (Lesson 6)</td></tr>
</table>

<h2>Back to Lesson 29: the tree stores indices, not tensors</h2>
<p>Now we can fully ground last lesson's sentence. The radix <span class="mono">TreeNode</span>'s <span class="mono">value</span> field holds <strong>exactly the token slot numbers (indices) discussed here</strong> —
they are <strong>indices into the token_to_kv pool</strong>, <strong>not the K/V tensors themselves</strong>. So the tree is a pure <strong>index/dedup</strong> layer; the pools are the <strong>storage</strong> layer.
Two requests sharing a prefix get <strong>the same run of indices</strong>, so attention reads the <strong>same physical slots</strong> in the pool — the shared part costs <strong>zero extra memory</strong>.
When a prefix is evicted (Lesson 32), the act is simply <strong>returning the slot numbers in its <span class="mono">value</span> to the allocator</strong>, freeing them for other tokens.</p>
<p>Connect the whole chain: a request arrives, <span class="mono">match_prefix</span> hits a prefix on the tree and returns a run of indices (reuse); the unmatched tail <strong>requests fresh slots from the allocator</strong>,
<span class="mono">write</span>s the computed K/V into the pool, records that new run of indices into its own <span class="mono">ReqToTokenPool</span> row, and <span class="mono">insert</span>s it back into the tree for later reuse.
Read, write, share, reclaim — all flow as "slot numbers," the common currency among <strong>ledger, warehouse, and tree</strong>. See this and you hold the <strong>central hub</strong> of SGLang memory management. The cells below draw "two requests pointing at a shared slot."</p>

<div class="cellgroup">
  <div class="cg-cap"><b>Two-hop addressing + prefix sharing</b>: both requests' ledgers write the same slot number (highlighted), pointing at the same physical K/V in the pool</div>
  <div class="cells"><span class="lab">Request A ledger</span><span class="cell hl">#12</span><span class="cell">#40</span><span class="cell">#41</span><span class="sep">→</span><span class="cell q">slot numbers A owns (read this row by req_pool_idx)</span></div>
  <div class="cells"><span class="lab">Request B ledger</span><span class="cell hl">#12</span><span class="cell">#88</span><span class="sep">→</span><span class="cell q">B also points at <span class="mono">#12</span> — shared prefix, same number in both ledgers</span></div>
  <div class="cells"><span class="lab">token_to_kv pool</span><span class="cell hl">K/V of slot#12</span><span class="sep">→</span><span class="cell q">one physical K/V; A and B both read it · zero extra memory (Lesson 29)</span></div>
  <div class="cells"><span class="lab">allocator</span><span class="cell">free pages…</span><span class="sep">→</span><span class="cell q">hands out new slots like <span class="mono">#40/#41/#88</span>; on evict <span class="mono">#12</span> is returned (Lesson 32)</span></div>
</div>

<p>In real code, the ledger is just an <span class="mono">int32</span> 2-D table and the warehouse is one big tensor per layer — almost too plain to look "high-tech," yet that plainness is what makes two-hop addressing both fast and thrifty:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/memory_pool.py ::ReqToTokenPool</span><span class="ln">private ledger + per-layer K/V warehouse</span></div>
  <pre><span class="kw">class</span> ReqToTokenPool:
    <span class="cm"># each request -&gt; the token slot numbers it owns; numbers only, no tensors</span>
    <span class="kw">def</span> __init__(self, size, max_context_len, device, ...):
        self.size = size
        <span class="cm"># [num_reqs+1, max_context_len] int32 table, rows indexed by req_pool_idx</span>
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device)
        self.free_slots = list(range(<span class="st">1</span>, self._alloc_size))

    <span class="kw">def</span> write(self, indices, values):
        self.req_to_token[indices] = values   <span class="cm"># write this request's run of slot numbers into its row</span>

<span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="kw">def</span> _create_buffers(self):
        <span class="cm"># one big tensor per layer [num_slots, heads, head_dim] -- the real K / V lives here</span>
        self.k_buffer = [torch.zeros((self.size + self.page_size,
                         self.head_num, self.head_dim), ...)
                         <span class="kw">for</span> _ <span class="kw">in</span> range(self.layer_num)]
        self.v_buffer = [torch.zeros((self.size + self.page_size,
                         self.head_num, self.v_head_dim), ...)
                         <span class="kw">for</span> _ <span class="kw">in</span> range(self.layer_num)]</pre>
</div>

<p>Now show the <strong>warehouse itself</strong> in the flesh — that's <span class="mono">MHATokenToKVPool</span>: each layer holds <strong>one K tensor and one V tensor</strong>, and a write drops this token's K/V into the cell at slot <span class="mono">loc</span>:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/memory_pool.py ::MHATokenToKVPool</span><span class="ln">the real KV data warehouse: a K/V buffer pair per layer</span></div>
  <pre><span class="kw">class</span> MHATokenToKVPool(KVCache):
    <span class="cm"># the actual KV DATA store: ONE K and ONE V tensor PER layer,</span>
    <span class="cm"># each shaped [size, num_kv_heads, head_dim]. ReqToTokenPool maps</span>
    <span class="cm"># a request's positions -&gt; slot indices INTO these buffers.</span>
    <span class="kw">def</span> __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, ...):
        self.k_buffer = [...]   <span class="cm"># list over layers of K tensors</span>
        self.v_buffer = [...]   <span class="cm"># list over layers of V tensors</span>
    <span class="kw">def</span> set_kv_buffer(self, layer, loc, cache_k, cache_v):
        <span class="cm"># write this token's K/V at slot `loc` for this layer</span>
        self.k_buffer[layer.layer_id][loc] = cache_k
        self.v_buffer[layer.layer_id][loc] = cache_v</pre>
</div>

<p>Concretely: for a <strong>32-layer</strong> model the pool holds <strong>32 K buffers + 32 V buffers</strong> (one pair per layer). Each KV "slot" stores the K and V of <strong>one token at one layer</strong>; a request's row in <span class="mono">ReqToTokenPool</span> gives the <strong>slot-number list</strong> for its positions — e.g. <span class="mono">[#12, #40, #41]</span> means this request's tokens 0/1/2 land in slots 12, 40, 41, and each slot number covers all 32 layers at once.</p>

<div class="card key">
  <div class="tag">🔑 Key takeaways</div>
  <strong>① Two pools, two-hop addressing</strong>: request →(ReqToTokenPool)→ token slot ids →(token_to_kv pool)→ physical K/V. The ledger holds numbers, the warehouse holds tensors.
  <strong>② ReqToTokenPool is the per-request private ledger</strong>, indexed by <span class="mono">req_pool_idx</span>; <strong>MHATokenToKVPool is the per-layer shared warehouse</strong>, indexed by slot number.
  <strong>③ TokenToKVPoolAllocator hands out / reclaims slots</strong> by page (Lesson 6); total slots are nailed at startup by <span class="mono">mem_fraction_static</span> (Lesson 8) = the concurrency ceiling (Lesson 4).
  <strong>④ Lesson 29's node value IS these indices, not tensors</strong> — tree is index, pools are storage; two requests sharing a prefix = the same slot numbers = the same physical K/V, zero extra memory.
  <strong>⑤ Splitting into two tables enables non-contiguity (paging) and cross-request sharing (radix tree)</strong>; eviction (Lesson 32) just returns the slot numbers to the allocator.
</div>
"""}

LESSON_31 = {"zh": r"""
<p class="lead">
第 29 课那棵基数树很聪明，但它有个天花板：它只活在 <strong>GPU 显存（HBM）</strong>里。HBM 又小又贵，再热门的前缀也总有装不下的一天——一旦被驱逐（第 32 课），那段 KV 就被<strong>彻底丢掉</strong>，下次再撞上只能<strong>从头重算</strong>。
可是 CPU 内存比 HBM 大 10–100 倍，磁盘和对象存储更是几乎无限。HiCache 要做的，就是把前缀缓存<strong>从 HBM 一层往下延伸到 CPU 内存、再到磁盘</strong>，让被赶出显存的 KV<strong>沉下去而不是消失</strong>，需要时再<strong>捞回来</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把存储想成<strong>书桌 → 抽屉 → 异地仓库</strong>三层。你<strong>天天用</strong>的几样东西摊在<strong>书桌</strong>上（GPU HBM），伸手就拿；<strong>最近用过</strong>但暂时放下的搁进<strong>抽屉</strong>（CPU 内存），拉开就有；<strong>很少翻</strong>的陈年档案送去<strong>异地仓库</strong>（磁盘）。
  关键是有个<strong>助理</strong>（控制器）在后台默默搬运：你还没开口，他就把你<strong>马上要用</strong>的文件从抽屉、仓库<strong>取上书桌</strong>；你刚用完的，他悄悄<strong>归档下去</strong>。于是你坐在书桌前<strong>永远不必干等</strong>——而要是没有这套三层加助理，桌子一满就只能把东西<strong>扔掉</strong>，下次要用就得<strong>重新做一份</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>HiCache 让前缀缓存长出层级</strong>。它用 <span class="inline">HiRadixCache</span>（第 29 课 <span class="mono">RadixCache</span> 的<strong>子类</strong>）加一个 <span class="mono">HiCacheController</span>，把缓存铺在<strong>三层</strong>上——<strong>GPU HBM（热）</strong>→<strong>CPU 内存（温）</strong>→<strong>磁盘 / 对象存储（冷）</strong>。
  驱逐时不再丢弃，而是把 KV<strong>写回（下沉）</strong>到下层；后来的请求若命中住在下层的前缀，就在前向真正需要之前<strong>预取（上移）</strong>回 HBM。搬运全在<strong>后台线程 / 拷贝流</strong>上做，与 GPU 计算循环<strong>重叠</strong>（第 21 课的精神），调度器不必停下来等一次 CPU↔GPU 拷贝。
  这是一个<strong>可选</strong>特性，靠一个开关打开。
</div>

<h2>问题：HBM 太小，丢掉就得重算</h2>
<p>先把痛点钉死。第 29 课的基数树把"token 序列 → KV 槽位号"索引得很漂亮，可它指向的 KV 张量全躺在 GPU 的<strong>显存池</strong>里（第 30 课），而显存池的总槽位在启动时就被 <span class="mono">mem_fraction_static</span>（第 8 课）<strong>钉死</strong>了。
热门前缀越来越多——长系统提示、动辄上万 token 的 RAG 上下文、几十轮的多轮对话——它们<strong>同时</strong>想驻留，可 HBM 就那么大。于是驱逐（第 32 课）不得不按 LRU 把最冷、且 <span class="mono">lock_ref</span> 为 0 的叶子摘掉腾地方。</p>
<p>朴素基数树的驱逐是<strong>一刀两断</strong>：节点一摘，它 value 指向的那批槽位<strong>立刻还给分配器</strong>，KV 数据<strong>就此蒸发</strong>。问题是——被赶走的前缀往往<strong>过一会儿又来了</strong>（同一个系统提示被下一波请求复用）。这时缓存里<strong>什么都没有</strong>，只能把这几千个 token 的 K/V<strong>从头算一遍</strong>，白白烧一次昂贵的前向算力。
而 CPU 内存就在隔壁，容量是 HBM 的<strong>10–100 倍</strong>，一次 CPU→GPU 拷贝的代价远比<strong>重算几千 token</strong> 便宜得多。把被驱逐的 KV<strong>暂存到 CPU（甚至落盘）</strong>、需要时<strong>拷回来</strong>，显然比扔掉再重算划算——这就是 HiCache 的全部出发点。</p>
<p>换个角度再算一笔账你会更服气。一段长系统提示假设有 8000 个 token，重算它的 K/V 需要把这 8000 个 token 在<strong>每一层</strong>都跑一遍前向，那是实打实的矩阵乘、是 GPU 最贵的算力，还会<strong>顶在首 token 时延（TTFT）的关键路径上</strong>，让用户等得心焦。
反过来，如果这段 KV 早已写回在 CPU 内存里，命中时只需沿 PCIe 把它<strong>搬回显存</strong>——这是一次纯数据搬运，带宽虽不及片内、却<strong>不占用计算单元</strong>，更能和别的批次的前向<strong>重叠</strong>掉。一边是"昂贵且堵在关键路径上的重算"，一边是"便宜且可被隐藏的拷贝"，孰优孰劣一目了然。
HiCache 赌的就是：<strong>对那些被反复复用的大前缀，搬运永远比重算划算</strong>。前缀越大、复用越频繁，这笔账就赢得越多。</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">热</span><span class="name">GPU HBM</span></div><div class="ld">最快、最贵、最小。前向<strong>直接读</strong>这一层；总槽位由 <span class="mono">mem_fraction_static</span> 钉死（第 8 课）。装不下就向下驱逐。延迟≈纳秒级，容量≈几十 GB。</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">温</span><span class="name">CPU 主机内存</span></div><div class="ld">大 10–100 倍、便宜得多。被 HBM 驱逐的 KV <strong>写回到这里</strong>暂存；命中时<strong>拷回</strong> GPU。要先过一次 CPU→GPU 拷贝才能用。延迟≈微秒级，容量≈几百 GB。</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">冷</span><span class="name">磁盘 / 对象存储</span></div><div class="ld">几乎无限、最慢。连 CPU 内存都装不下的超大共享前缀落到这一层（可跨进程、跨机共享）。延迟≈毫秒级，容量≈近乎无限。可选开启。</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="三级存储：GPU HBM 在顶最小最快，CPU 内存居中更大更慢，磁盘/SSD 在底近乎无限最慢；越往下容量越大、延迟越高">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">三级缓存：越往下越大、越慢</text>
    <line x1="70" y1="70" x2="70" y2="250" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="70,262 63,248 77,248" style="fill:var(--line)"/>
    <text x="70" y="60" text-anchor="middle" style="fill:var(--faint);font-size:12px">小·快</text>
    <text x="70" y="282" text-anchor="middle" style="fill:var(--faint);font-size:12px">大·慢</text>
    <rect x="150" y="64" width="200" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="86" text-anchor="middle" style="font-weight:700;fill:var(--amber)">GPU HBM</text>
    <text x="250" y="106" text-anchor="middle" style="font-size:12px">热 · 最快最小</text>
    <rect x="150" y="134" width="320" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="310" y="156" text-anchor="middle" style="font-weight:700;fill:var(--teal)">CPU 内存</text>
    <text x="310" y="176" text-anchor="middle" style="font-size:12px">温 · 大 10×</text>
    <rect x="150" y="204" width="460" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="380" y="226" text-anchor="middle" style="font-weight:700;fill:var(--blue)">磁盘 / SSD</text>
    <text x="380" y="246" text-anchor="middle" style="font-size:12px">冷 · 近乎无限</text>
    <text x="624" y="94" style="fill:var(--muted);font-size:12px">延迟 ≈ ns</text>
    <text x="624" y="164" style="fill:var(--muted);font-size:12px">延迟 ≈ µs</text>
    <text x="624" y="234" style="fill:var(--muted);font-size:12px">延迟 ≈ ms</text>
  </svg>
  <div class="figcap"><b>图 1 · 三级存储</b> — GPU HBM（热）最小最快、前向直接读；CPU 内存（温）大约 10×、被驱逐的 KV 先沉到这；磁盘/SSD（冷）近乎无限。越往下<strong>容量越大、延迟越高</strong>——这正是用拷贝换重算的底气。</div>
</div>

<h2>HiCache 怎么扩展基数树：写回与预取</h2>
<p>HiCache 的两个主角是 <span class="mono">HiRadixCache</span> 和 <span class="mono">HiCacheController</span>。<span class="mono">HiRadixCache</span> 直接<strong>继承</strong>第 29 课的 <span class="mono">RadixCache</span>——同一棵树、同一套 <span class="mono">match_prefix</span> / <span class="mono">insert</span> / <span class="mono">lock_ref</span> 逻辑全部照用，只是在 <span class="mono">TreeNode</span> 上多挂了一个 <span class="mono">host_value</span> 字段，
记录"这段 KV 在<strong>下层（host）</strong>的副本存在哪些槽位"。于是树上每个节点都能回答一个新问题：<strong>我现在到底住在哪一层？</strong>——只在 GPU（<span class="mono">value</span> 有效）、只在 host（被驱逐了，<span class="mono">host_value</span> 有效）、还是<strong>两层都有</strong>。</p>
<p>两个核心动作把三层缝起来。其一是<strong>写回（writeback，向下）</strong>：当一个节点要被逐出 HBM，<span class="mono">write_backup</span> 不直接丢，而是先通过控制器把它的 <span class="mono">value</span>（GPU 槽位）<strong>拷到 host</strong>，把拿到的 host 槽位记进 <span class="mono">host_value</span>，<strong>之后</strong>才放心释放 GPU 槽位。
这段前缀<strong>沉到了温层</strong>，树结构还在、只是数据换了住处。其二是<strong>预取 / 回载（prefetch / load_back，向上）</strong>：当一条新请求 <span class="mono">match_prefix</span> 命中一个<strong>已被驱逐、但 host 上有备份</strong>的节点，<span class="mono">load_back</span> 就把 <span class="mono">host_value</span> 指向的 KV<strong>拷回 GPU</strong>，重新填好 <span class="mono">value</span>，让前向能像往常一样直接读。控制器一路上还会<strong>给祖先加锁</strong>（<span class="mono">inc_lock_ref</span>，第 32 课），免得回载到一半被别的驱逐抢走槽位。</p>
<p>这里有个容易被忽略的细节：写回必须保持<strong>"从根到当前节点是一段连续前缀"</strong>这条不变量——也就是说，要把一个节点写回 host，它的<strong>父节点必须已经在 host 上有备份</strong>，不能跳着写、留下空洞。原因和第 29 课一脉相承：基数树的复用是<strong>沿路径逐段命中</strong>的，如果中间某段在 host 上缺失，下层就算存着后半段也<strong>接不上</strong>，等于白存。所以 <span class="mono">write_backup</span> 里那句"父节点还没备份就先跳过"，正是在维护这条<strong>前缀连续性</strong>。
同理，回载也要<strong>从被命中的最深节点一路往上</strong>，把链条上所有"只在 host、不在 GPU"的祖先节点<strong>一起拷回</strong>，凑成一段完整的、GPU 上可直接读的前缀——这也是 <span class="mono">load_back</span> 里要先沿 <span class="mono">node.parent</span> 向上收集 <span class="mono">nodes_to_load</span> 的缘故。</p>
<p>还要分清两个常被混为一谈的概念：<strong>一段 KV 究竟住在哪一层</strong>，和<strong>这个树节点存不存在</strong>，是两回事。节点被"驱逐"出 HBM 后，它在树上<strong>并没有被删除</strong>——<span class="mono">value</span> 清空了，但 <span class="mono">host_value</span> 还在，节点照样挂在树上当索引。于是同一段前缀可能处于三种状态：<strong>只在 GPU</strong>（刚算出、还没写回）、<strong>GPU 与 host 都有</strong>（写回了但还没被赶出 HBM，是最理想的"双保险"状态）、<strong>只在 host</strong>（已逐出 HBM、等着被预取回来）。控制器要为每个节点<strong>精确记账</strong>它现在是哪种状态，这正是多层缓存比单层复杂的地方，也是后台 I/O 必须小心翼翼之处。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>HBM 满，触发驱逐</h4><p>显存吃紧，第 32 课的 LRU 选中一个最冷、<span class="mono">lock_ref=0</span> 的叶子准备逐出——但这次<strong>不直接丢</strong>。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>写回下沉 writeback ↓</h4><p><span class="mono">write_backup</span> 把节点的 <span class="mono">value</span>（GPU 槽位）拷到 host，记进 <span class="mono">host_value</span>，再释放 GPU 槽位。前缀<strong>沉到 CPU 温层</strong>，树节点仍在。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>后来请求命中下层</h4><p>新请求 <span class="mono">match_prefix</span> 走到这个节点，发现它 <span class="mono">value</span> 已空、但 <span class="mono">host_value</span> 有备份——命中了一段<strong>住在温层</strong>的前缀。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>预取上移 prefetch ↑</h4><p><span class="mono">load_back</span> 把 <span class="mono">host_value</span> 的 KV 拷回 GPU、重填 <span class="mono">value</span>，并 <span class="mono">inc_lock_ref</span> 锁住祖先。前向直接读，<strong>省掉一次重算</strong>。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="命中已下沉到 CPU 的节点时 load_back 把它向上拷回 GPU 换入；GPU 有压力时 write_backup 把冷的 LRU 节点向下写回 CPU 下沉">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">命中深层即换入 · 有压力则下沉</text>
    <text x="52" y="50" style="fill:var(--muted);font-size:12px">GPU HBM（热 · 显存紧张）</text>
    <rect x="40" y="56" width="720" height="78" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="80" y="74" width="180" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="170" y="100" text-anchor="middle" style="font-size:12px">冷 LRU 节点</text>
    <rect x="520" y="74" width="200" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="620" y="100" text-anchor="middle" style="font-size:12px">换入后可直接读</text>
    <text x="52" y="194" style="fill:var(--muted);font-size:12px">CPU host 内存（温 · 备份区）</text>
    <rect x="40" y="200" width="720" height="78" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="80" y="218" width="180" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="170" y="244" text-anchor="middle" style="font-size:12px">下沉的 KV</text>
    <rect x="520" y="218" width="200" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="620" y="244" text-anchor="middle" style="font-size:12px">命中 · host 有备份</text>
    <line x1="170" y1="118" x2="170" y2="214" style="stroke:var(--amber);stroke-width:2"/>
    <polygon points="170,216 164,204 176,204" style="fill:var(--amber)"/>
    <text x="185" y="172" style="fill:var(--amber);font-size:12px">write_backup ↓ 下沉</text>
    <line x1="620" y1="214" x2="620" y2="120" style="stroke:var(--teal);stroke-width:2"/>
    <polygon points="620,118 614,130 626,130" style="fill:var(--teal)"/>
    <text x="605" y="172" text-anchor="end" style="fill:var(--teal);font-size:12px">load_back ↑ 换入</text>
  </svg>
  <div class="figcap"><b>图 2 · 换入与下沉</b> — 前缀命中一个住在 CPU 温层的节点时，<span class="mono">load_back</span> 把它<strong>向上拷回 GPU（换入）</strong>；另一边 GPU 显存吃紧时，最冷的 LRU 节点被 <span class="mono">write_backup</span> <strong>向下写回 CPU（下沉）</strong>。两条箭头：上=换入、下=下沉。</div>
</div>

<h2>对比与后台 I/O：拷贝换重算，且不堵调度</h2>
<p>把"扔掉再重算"和"写回再预取"摆在一起，差别就一目了然。朴素 HBM-only 缓存在驱逐那一刻就<strong>认赔</strong>：数据没了，复用的唯一指望是它<strong>还没被赶走</strong>；一旦赶走，下次命中等于没命中，得<strong>重算</strong>。
HiCache 则把"赶走"变成"<strong>降级</strong>"——数据搬到便宜的下层继续待命，下次命中只需一次<strong>便宜的 CPU→GPU 拷贝</strong>，而不是一次<strong>昂贵的前向重算</strong>。这就是 HiCache 的核心交易：<strong>用拷贝带宽换计算</strong>。</p>
<p>但拷贝本身也要花时间，如果让调度器<strong>停下来干等</strong>一次 CPU↔GPU 拷贝，那就得不偿失了。所以 <span class="mono">HiCacheController</span> 把写回和预取都丢到<strong>后台线程 / 独立拷贝流</strong>上做，和 GPU 的计算循环<strong>重叠</strong>起来（正是第 21 课"重叠调度"的精神）：当前这一批在 GPU 上算前向时，控制器<strong>同时</strong>在后台把上一批驱逐的 KV 往下搬、把下一批要用的 KV 往上预取。
等前向真正需要那段 KV 时，它<strong>已经</strong>躺在 HBM 里了。控制器内部维护着每个节点的<strong>层级状态</strong>和一组<strong>进行中的预取 / 写回操作</strong>，靠队列和事件在后台线程与主调度循环之间协调，保证调度器<strong>几乎不为 I/O 阻塞</strong>。</p>

<div class="cols">
  <div class="col"><h4>朴素 HBM-only（第 29 课）</h4><p>驱逐 = <strong>丢弃</strong>。节点一摘，槽位立刻还给分配器，KV <strong>蒸发</strong>。同一前缀过会儿再来 → 缓存空 → 几千 token <strong>从头重算</strong>，烧一次昂贵前向。复用全靠"还没被赶走"这点运气。</p></div>
  <div class="col"><h4>HiCache 分层</h4><p>驱逐 = <strong>降级</strong>。<span class="mono">write_backup</span> 把 KV 写回 CPU/磁盘，<span class="mono">load_back</span> 命中时拷回。同一前缀再来 → host 命中 → 一次<strong>便宜 CPU→GPU 拷贝</strong>替掉重算。后台 I/O 与计算重叠，<strong>不堵调度</strong>。</p></div>
</div>

<table class="t">
  <tr><th>层级</th><th>相对延迟</th><th>容量</th><th>角色</th></tr>
  <tr><td class="mono">GPU HBM</td><td>最低（纳秒级）</td><td>最小（几十 GB）</td><td>热层：前向直接读；满了向下驱逐</td></tr>
  <tr><td class="mono">CPU 主机内存</td><td>中（微秒级）</td><td>大 10–100 倍（几百 GB）</td><td>温层：写回暂存 + 命中后预取回 GPU</td></tr>
  <tr><td class="mono">磁盘 / 对象存储</td><td>最高（毫秒级）</td><td>近乎无限</td><td>冷层：超大共享前缀；可跨进程/机共享（可选）</td></tr>
</table>

<h2>代价、收益与何时开启</h2>
<p>收益很实在：对那些<strong>装不进 HBM 的大/长共享前缀</strong>，HiCache 把<strong>有效命中率</strong>抬得高得多——长系统提示、大段 RAG 上下文、几十轮聊天历史，这些前缀太大、没法全留在显存，朴素缓存留不住，HiCache 却能让它们在温层/冷层候命，命中时一拷即用。
最终体现为<strong>吞吐与延迟</strong>的改善（第 8 课）：省下的每一次重算，都是省下的前向算力和首 token 时延。</p>
<p>再把收益落到几个典型场景上看得更清楚。<strong>长系统提示</strong>：一份几千 token 的"人设 + 规则 + few-shot 示例"被成千上万条请求共用，它本该是命中率的金矿，可一旦它在 HBM 里待不住被赶走，金矿就塌了；HiCache 让它沉到 CPU 候命，金矿就一直在。<strong>大段 RAG 上下文</strong>：检索拼进来的文档动辄上万 token，多条问同一批文档的请求若能复用这段前缀，省下的是最重的那部分前向；但这些上下文太占显存，正是 HBM 最先想丢的，HiCache 恰好把它接住。
<strong>多轮对话</strong>：聊到几十轮后，前面的历史越积越长，用户每发一句新话，前面那一长串历史 KV 都该被复用而非重算——可活跃会话一多，HBM 根本放不下所有人的历史，HiCache 让不活跃会话的历史<strong>暂退到温层</strong>，等用户回来再<strong>预取</strong>上来，体感上就是"接着聊毫不卡顿"。这三类负载的共性就是那句话：<strong>前缀大、复用高、却塞不进 HBM</strong>。</p>
<p>代价也要诚实摆出来：你要付出<strong>额外的 CPU 内存 / 磁盘</strong>来当下层仓库，付出<strong>拷贝带宽</strong>来上下搬运，还要承担<strong>多层一致性</strong>的复杂度——同一段 KV 可能同时存在于 GPU 和 host，控制器得清楚地记着每个节点"现在住在哪层"、哪些写回 / 预取还在飞，别让数据读错或重复释放。
正因为有这些成本，HiCache 是<strong>可选</strong>的：用一个开关（<span class="mono">hicache</span> 相关的 <span class="mono">server_args</span>）打开。请求前缀小、或共享度本就不高时，朴素基数树足矣；只有当你的负载是"<strong>大前缀、高复用、但塞不进 HBM</strong>"这一类，HiCache 才真正物有所值。下一课（第 32 课）会接着讲驱逐与命中率——HiCache 正是把那里的"驱逐 = 丢弃"改写成了"驱逐 = 下沉"。</p>
<p>用一句话收束这条主线：第 29 课的基数树解决了"<strong>同一层内如何共享</strong>"，而本课的 HiCache 解决了"<strong>跨层如何不丢</strong>"。前者把重复前缀的 KV 复用到极致，后者把放不下的前缀<strong>稳稳接住、延寿保存、再适时唤回</strong>。两课叠在一起，缓存才既<strong>省得彻底</strong>又<strong>留得住</strong>——这正是大前缀、高并发场景下吞吐还能再上一个台阶、而显存却没有变大的底层原因。</p>

<p>下面这段就是 <span class="mono">HiRadixCache</span> 的真身：它<strong>继承自第 29 课的 <span class="mono">RadixCache</span></strong>，<span class="mono">write_backup</span> 正是"向下写回"那一步——把 GPU 槽位拷到 host、记进 <span class="mono">host_value</span>：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/hiradix_cache.py ::HiRadixCache</span><span class="ln">写回下沉：GPU → host</span></div>
  <pre><span class="kw">class</span> HiRadixCache(RadixCache):          <span class="cm"># 第 29 课 RadixCache 的分层子类</span>

    <span class="kw">def</span> write_backup(self, node, write_back=False) -&gt; int:
        <span class="cm"># 把要被逐出的节点 KV 从 GPU 写回（下沉）到 host 内存</span>
        host_indices = self.cache_controller.write(
            device_indices=node.value,        <span class="cm"># GPU 池里的槽位号（第 30 课）</span>
            node_id=node.id,
        )
        <span class="kw">if</span> host_indices <span class="kw">is</span> None:             <span class="cm"># host 也满了 → 先腾出 host 再写</span>
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value, node_id=node.id,
            )
        <span class="kw">if</span> host_indices <span class="kw">is</span> <span class="kw">not</span> None:
            node.host_value = host_indices.clone()  <span class="cm"># 记下它现在也住在 host 这一层</span>
            <span class="kw">if</span> <span class="kw">not</span> write_back:
                self.inc_lock_ref(node)        <span class="cm"># 写回期间锁住，别被抢走</span>
        <span class="kw">return</span> len(host_indices)</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/hiradix_cache.py ::HiRadixCache.load_back</span><span class="ln">命中已下沉节点：把 host KV 拷回 GPU（换入）</span></div>
  <pre>    <span class="kw">def</span> load_back(self, node, mem_quota=None):
        <span class="cm"># 前缀命中落在「已驱逐（host 上）」的节点：从命中点</span>
        <span class="cm"># 向上收集「被逐但有 host 备份」的祖先，锁住它们，再把</span>
        <span class="cm"># 它们的 host KV 拷回 GPU（太小或超配额则跳过）。</span>
        nodes_to_load = []
        <span class="kw">while</span> node.evicted:
            nodes_to_load.insert(0, node)      <span class="cm"># 这段在 host 上，需要换入</span>
            node = node.parent
        self.inc_lock_ref(node)                <span class="cm"># 锁住祖先，别被驱逐抢走</span>
        host_indices = torch.cat([n.host_value <span class="kw">for</span> n <span class="kw">in</span> nodes_to_load])
        device_indices = self.cache_controller.load(host_indices, ...)  <span class="cm"># host -&gt; GPU</span>
        <span class="kw">return</span> device_indices</pre>
</div>

<p>两个具体数字让账更直观。其一，<strong>host/CPU 这层通常是 GPU KV 容量的约 10×</strong>，于是能同时缓存的前缀数量也大致多出一个量级——更多热门前缀得以留存，而不是一满就被丢掉。其二，一次<strong>深层前缀命中</strong>只触发<strong>一次 host→GPU 拷贝</strong>（沿 <span class="mono">node.parent</span> 把链上“只在 host”的祖先一并捞回），就替代了对<strong>成千上万个 token</strong> 的重新前向计算——便宜的搬运换掉了昂贵的重算。</p>

<div class="card key">
  <div class="tag">🔑 本课要点</div>
  <strong>① HBM 又小又贵，朴素基数树（第 29 课）一驱逐就丢、下次得重算</strong>；而 CPU 内存大 10–100 倍、磁盘近乎无限。
  <strong>② HiCache = <span class="mono">HiRadixCache</span>（RadixCache 子类）+ <span class="mono">HiCacheController</span></strong>，把前缀缓存铺在三层：GPU HBM 热 → CPU 内存 温 → 磁盘 冷。
  <strong>③ 两个方向：驱逐时 <span class="mono">write_backup</span> 把 KV 写回下沉、命中时 <span class="mono">load_back</span> 预取上移</strong>，靠 <span class="mono">TreeNode.host_value</span> 记住每段住在哪层。
  <strong>④ 写回 / 预取在后台线程 / 拷贝流上做、与计算重叠（第 21 课精神）</strong>，调度器不为一次 CPU↔GPU 拷贝阻塞。
  <strong>⑤ 核心交易：一次便宜的 CPU→GPU 拷贝换掉一次昂贵的重算</strong>。最适合"大前缀、高复用、塞不进 HBM"的负载（长系统提示、大 RAG、多轮聊天）；代价是额外内存/磁盘、拷贝带宽与多层一致性，故为可选开关。索引指向的池见第 30 课，驱逐见第 32 课，吞吐收益见第 8 课。
</div>
""",
             "en": r"""
<p class="lead">
Lesson 29's radix tree is clever, but it has a ceiling: it lives only in <strong>GPU memory (HBM)</strong>. HBM is small and precious, so even the hottest prefixes eventually won't all fit — and once one is evicted (Lesson 32), that KV is <strong>dropped entirely</strong>, so the next hit must <strong>recompute it from scratch</strong>.
Yet CPU RAM is 10–100× bigger than HBM, and disk / object storage is effectively unlimited. HiCache's job is to <strong>extend the prefix cache one tier down into CPU memory, then disk</strong>, so KV pushed out of HBM <strong>sinks instead of vanishing</strong>, and is <strong>fetched back up</strong> when needed.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of storage as three tiers: <strong>desk → drawer → off-site warehouse</strong>. The few things you use <strong>constantly</strong> stay on your <strong>desk</strong> (GPU HBM), grabbed instantly; things you used <strong>recently</strong> but set aside go in the <strong>drawer</strong> (CPU RAM), one pull away; rarely-touched <strong>archives</strong> ship to the <strong>warehouse</strong> (disk).
  The trick is an <strong>assistant</strong> (the controller) quietly shuttling in the background: before you even ask, he fetches the file you're <strong>about to need</strong> up from the drawer or warehouse <strong>onto your desk</strong>; the thing you just finished he files <strong>back down</strong>. So at your desk you <strong>never wait</strong> — whereas without the three tiers and the assistant, a full desk means you must <strong>throw things away</strong> and <strong>remake them</strong> next time.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  One sentence: <strong>HiCache gives the prefix cache tiers</strong>. It uses <span class="inline">HiRadixCache</span> (a <strong>subclass</strong> of Lesson 29's <span class="mono">RadixCache</span>) plus a <span class="mono">HiCacheController</span> to spread the cache across <strong>three tiers</strong> — <strong>GPU HBM (hot)</strong> → <strong>CPU host memory (warm)</strong> → <strong>disk / object store (cold)</strong>.
  On eviction it no longer drops; it <strong>writes the KV back (down)</strong> into a lower tier; when a later request matches a prefix living in a lower tier, it's <strong>prefetched (up)</strong> into HBM before the forward needs it. All the shuttling happens on <strong>background threads / copy streams</strong>, <strong>overlapped</strong> with the GPU compute loop (Lesson 21's spirit), so the scheduler never stalls on a CPU↔GPU copy.
  It's an <strong>optional</strong> feature, turned on by a flag.
</div>

<h2>The problem: HBM is tiny, and dropping means recompute</h2>
<p>Pin the pain first. Lesson 29's tree indexes "token sequence → KV slot numbers" beautifully, but the KV tensors it points at all live in the GPU's <strong>memory pool</strong> (Lesson 30), whose total slots are <strong>nailed at startup</strong> by <span class="mono">mem_fraction_static</span> (Lesson 8).
Hot prefixes pile up — long system prompts, RAG contexts of tens of thousands of tokens, many-turn chats — all wanting to reside <strong>at once</strong>, but HBM is only so big. So eviction (Lesson 32) must drop the coldest leaf with <span class="mono">lock_ref</span>=0 by LRU to make room.</p>
<p>The naive tree's eviction is a <strong>clean kill</strong>: pull the node and its slots <strong>go straight back to the allocator</strong>, the KV data <strong>gone</strong>. The catch — the evicted prefix often <strong>comes right back</strong> (the same system prompt reused by the next wave). Now the cache holds <strong>nothing</strong>, so those thousands of tokens' K/V must be <strong>recomputed from scratch</strong>, burning an expensive forward for nothing.
Meanwhile CPU RAM sits right next door, <strong>10–100×</strong> the capacity of HBM, and one CPU→GPU copy is far cheaper than <strong>recomputing thousands of tokens</strong>. Stashing evicted KV <strong>into CPU (or onto disk)</strong> and <strong>copying it back</strong> when needed clearly beats drop-then-recompute — that's the whole motivation for HiCache.</p>

<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">Hot</span><span class="name">GPU HBM</span></div><div class="ld">Fastest, priciest, smallest. The forward <strong>reads directly</strong> here; total slots nailed by <span class="mono">mem_fraction_static</span> (Lesson 8). Overflow evicts downward. Latency ≈ ns, capacity ≈ tens of GB.</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">Warm</span><span class="name">CPU host memory</span></div><div class="ld">10–100× bigger, far cheaper. KV evicted from HBM is <strong>written back here</strong> to wait; on a hit it's <strong>copied back</strong> to GPU. Needs one CPU→GPU copy before use. Latency ≈ µs, capacity ≈ hundreds of GB.</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">Cold</span><span class="name">Disk / object store</span></div><div class="ld">Effectively unlimited, slowest. Huge shared prefixes that won't even fit in CPU RAM land here (can be shared across processes/hosts). Latency ≈ ms, capacity ≈ near-infinite. Optional.</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Three tiers: GPU HBM smallest and fastest on top, CPU RAM bigger and slower in the middle, disk/SSD near-infinite and slowest at the bottom; capacity grows downward, latency grows downward">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">Three tiers: bigger &amp; slower downward</text>
    <line x1="70" y1="70" x2="70" y2="250" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="70,262 63,248 77,248" style="fill:var(--line)"/>
    <text x="70" y="60" text-anchor="middle" style="fill:var(--faint);font-size:12px">small·fast</text>
    <text x="70" y="282" text-anchor="middle" style="fill:var(--faint);font-size:12px">big·slow</text>
    <rect x="150" y="64" width="200" height="52" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="86" text-anchor="middle" style="font-weight:700;fill:var(--amber)">GPU HBM</text>
    <text x="250" y="106" text-anchor="middle" style="font-size:12px">hot · fastest</text>
    <rect x="150" y="134" width="320" height="52" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="310" y="156" text-anchor="middle" style="font-weight:700;fill:var(--teal)">CPU RAM</text>
    <text x="310" y="176" text-anchor="middle" style="font-size:12px">warm · 10× bigger</text>
    <rect x="150" y="204" width="460" height="52" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="380" y="226" text-anchor="middle" style="font-weight:700;fill:var(--blue)">Disk / SSD</text>
    <text x="380" y="246" text-anchor="middle" style="font-size:12px">cold · ~infinite</text>
    <text x="624" y="94" style="fill:var(--muted);font-size:12px">lat ≈ ns</text>
    <text x="624" y="164" style="fill:var(--muted);font-size:12px">lat ≈ µs</text>
    <text x="624" y="234" style="fill:var(--muted);font-size:12px">lat ≈ ms</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Three tiers</b> — GPU HBM (hot) is smallest and fastest, read directly by the forward; CPU RAM (warm) is ~10× bigger and catches evicted KV; disk/SSD (cold) is near-infinite. Going down, <strong>capacity grows and latency grows</strong> — that's why trading a copy for a recompute pays off.</div>
</div>

<h2>How HiCache extends the tree: writeback and prefetch</h2>
<p>HiCache's two protagonists are <span class="mono">HiRadixCache</span> and <span class="mono">HiCacheController</span>. <span class="mono">HiRadixCache</span> directly <strong>inherits</strong> Lesson 29's <span class="mono">RadixCache</span> — same tree, same <span class="mono">match_prefix</span> / <span class="mono">insert</span> / <span class="mono">lock_ref</span> logic, reused wholesale — it just hangs one extra field, <span class="mono">host_value</span>, on each <span class="mono">TreeNode</span>,
recording "which host slots hold this run's KV copy in the <strong>lower tier</strong>." So every node can now answer a new question: <strong>which tier do I currently live in?</strong> — GPU only (<span class="mono">value</span> valid), host only (evicted, <span class="mono">host_value</span> valid), or <strong>both</strong>.</p>
<p>Two core actions stitch the tiers together. First, <strong>writeback (down)</strong>: when a node is about to leave HBM, <span class="mono">write_backup</span> doesn't drop it — it first copies its <span class="mono">value</span> (GPU slots) <strong>to host</strong> via the controller, records the host slots in <span class="mono">host_value</span>, and <strong>only then</strong> frees the GPU slots.
The prefix has <strong>sunk into the warm tier</strong>; the tree structure stays, only the data moved house. Second, <strong>prefetch / load_back (up)</strong>: when a new request's <span class="mono">match_prefix</span> hits a node that's <strong>evicted but backed up on host</strong>, <span class="mono">load_back</span> copies the KV that <span class="mono">host_value</span> points at <strong>back to GPU</strong>, refilling <span class="mono">value</span> so the forward reads it as usual. Along the way the controller <strong>locks the ancestors</strong> (<span class="mono">inc_lock_ref</span>, Lesson 32) so the load-in-progress isn't snatched by another eviction.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>HBM full, eviction triggers</h4><p>Memory is tight; Lesson 32's LRU picks the coldest leaf with <span class="mono">lock_ref=0</span> to evict — but this time it <strong>doesn't just drop</strong>.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Writeback down ↓</h4><p><span class="mono">write_backup</span> copies the node's <span class="mono">value</span> (GPU slots) to host, records it in <span class="mono">host_value</span>, then frees the GPU slots. The prefix <strong>sinks to the CPU warm tier</strong>; the tree node remains.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Later request hits a lower tier</h4><p>A new request's <span class="mono">match_prefix</span> reaches this node and finds <span class="mono">value</span> empty but <span class="mono">host_value</span> backed up — a hit on a prefix <strong>living in the warm tier</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Prefetch up ↑</h4><p><span class="mono">load_back</span> copies the <span class="mono">host_value</span> KV back to GPU, refills <span class="mono">value</span>, and <span class="mono">inc_lock_ref</span> locks the ancestors. The forward reads it directly, <strong>skipping a recompute</strong>.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="On a hit to a node already sunk to CPU, load_back copies it up to GPU as a swap-in; under GPU memory pressure write_backup writes a cold LRU node down to CPU as an evict-down">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">Deeper-tier hit → swap in · pressure → evict down</text>
    <text x="52" y="50" style="fill:var(--muted);font-size:12px">GPU HBM (hot · under pressure)</text>
    <rect x="40" y="56" width="720" height="78" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="80" y="74" width="180" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="170" y="100" text-anchor="middle" style="font-size:12px">cold LRU node</text>
    <rect x="520" y="74" width="200" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="620" y="100" text-anchor="middle" style="font-size:12px">swapped-in, usable</text>
    <text x="52" y="194" style="fill:var(--muted);font-size:12px">CPU host RAM (warm · backup)</text>
    <rect x="40" y="200" width="720" height="78" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="80" y="218" width="180" height="42" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="170" y="244" text-anchor="middle" style="font-size:12px">sunk KV copy</text>
    <rect x="520" y="218" width="200" height="42" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="620" y="244" text-anchor="middle" style="font-size:12px">hit · host backup</text>
    <line x1="170" y1="118" x2="170" y2="214" style="stroke:var(--amber);stroke-width:2"/>
    <polygon points="170,216 164,204 176,204" style="fill:var(--amber)"/>
    <text x="185" y="172" style="fill:var(--amber);font-size:12px">write_backup ↓ evict</text>
    <line x1="620" y1="214" x2="620" y2="120" style="stroke:var(--teal);stroke-width:2"/>
    <polygon points="620,118 614,130 626,130" style="fill:var(--teal)"/>
    <text x="605" y="172" text-anchor="end" style="fill:var(--teal);font-size:12px">load_back ↑ swap-in</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Swap-in and evict-down</b> — when a prefix hits a node living in the CPU warm tier, <span class="mono">load_back</span> copies it <strong>up to GPU (swap-in)</strong>; meanwhile, under GPU memory pressure the coldest LRU node is written <strong>down to CPU (evict-down)</strong> by <span class="mono">write_backup</span>. Two arrows: up = swap-in, down = evict-down.</div>
</div>

<h2>Comparison and background I/O: trade a copy for a recompute, without stalling the scheduler</h2>
<p>Put "drop then recompute" next to "writeback then prefetch" and the difference is plain. A naive HBM-only cache <strong>takes the loss</strong> at eviction: the data is gone, and reuse hinges only on it <strong>not having been evicted yet</strong>; once evicted, the next hit is a miss and means a <strong>recompute</strong>.
HiCache turns "evict" into "<strong>demote</strong>" — the data moves to a cheap lower tier and waits, so the next hit costs just one <strong>cheap CPU→GPU copy</strong> rather than one <strong>expensive forward recompute</strong>. That's HiCache's core trade: <strong>copy bandwidth in exchange for compute</strong>.</p>
<p>But the copy itself takes time, and making the scheduler <strong>sit and wait</strong> on a CPU↔GPU copy would defeat the purpose. So <span class="mono">HiCacheController</span> runs both writeback and prefetch on <strong>background threads / a separate copy stream</strong>, <strong>overlapped</strong> with the GPU compute loop (exactly Lesson 21's "overlap schedule" spirit): while the current batch runs its forward on the GPU, the controller is <strong>simultaneously</strong> shuttling the last batch's evicted KV down and prefetching the next batch's KV up.
By the time the forward truly needs that KV, it's <strong>already</strong> sitting in HBM. Inside, the controller tracks each node's <strong>tier state</strong> and a set of <strong>in-flight prefetch / writeback operations</strong>, coordinating between background threads and the main scheduling loop via queues and events, so the scheduler is <strong>almost never blocked on I/O</strong>.</p>

<div class="cols">
  <div class="col"><h4>Naive HBM-only (Lesson 29)</h4><p>Evict = <strong>drop</strong>. Pull the node and slots return to the allocator instantly; the KV <strong>vanishes</strong>. Same prefix returns → empty cache → thousands of tokens <strong>recomputed from scratch</strong>, burning an expensive forward. Reuse relies on the luck of "not evicted yet."</p></div>
  <div class="col"><h4>HiCache tiering</h4><p>Evict = <strong>demote</strong>. <span class="mono">write_backup</span> writes KV back to CPU/disk; <span class="mono">load_back</span> copies it back on a hit. Same prefix returns → host hit → one <strong>cheap CPU→GPU copy</strong> replaces the recompute. Background I/O overlaps with compute, so the <strong>scheduler isn't stalled</strong>.</p></div>
</div>

<table class="t">
  <tr><th>Tier</th><th>Relative latency</th><th>Capacity</th><th>Role</th></tr>
  <tr><td class="mono">GPU HBM</td><td>lowest (ns)</td><td>smallest (tens of GB)</td><td>Hot: forward reads directly; overflow evicts down</td></tr>
  <tr><td class="mono">CPU host memory</td><td>mid (µs)</td><td>10–100× bigger (hundreds of GB)</td><td>Warm: writeback stash + prefetch back to GPU on hit</td></tr>
  <tr><td class="mono">Disk / object store</td><td>highest (ms)</td><td>near-infinite</td><td>Cold: huge shared prefixes; shareable across processes/hosts (optional)</td></tr>
</table>

<h2>Cost, payoff, and when to enable it</h2>
<p>The payoff is real: for <strong>big/long shared prefixes that don't fit in HBM</strong>, HiCache pushes the <strong>effective hit rate</strong> far higher — long system prompts, big RAG contexts, tens of turns of chat history; these prefixes are too big to keep entirely in HBM, so a naive cache can't hold them, but HiCache lets them wait in the warm/cold tier, ready to copy in on a hit.
This shows up as better <strong>throughput and latency</strong> (Lesson 8): every recompute saved is forward compute and time-to-first-token saved.</p>
<p>Be honest about the cost too: you pay <strong>extra CPU RAM / disk</strong> as the lower-tier warehouse, <strong>copy bandwidth</strong> to shuttle up and down, and the complexity of <strong>cross-tier consistency</strong> — the same KV may live in both GPU and host at once, so the controller must clearly track which tier each node lives in and which writebacks / prefetches are in flight, lest data be read wrong or freed twice.
Because of these costs, HiCache is <strong>optional</strong>: turned on by a flag (the <span class="mono">hicache</span>-related <span class="mono">server_args</span>). When request prefixes are small or sharing is low, the naive radix tree suffices; only when your workload is the "<strong>big prefixes, high reuse, but won't fit in HBM</strong>" kind does HiCache truly earn its keep. The next lesson (Lesson 32) continues with eviction and hit rate — HiCache is precisely what rewrites that "evict = drop" into "evict = sink down."</p>

<p>Here is <span class="mono">HiRadixCache</span> in the flesh: it <strong>inherits from Lesson 29's <span class="mono">RadixCache</span></strong>, and <span class="mono">write_backup</span> is exactly the "write back down" step — copying GPU slots to host and recording them in <span class="mono">host_value</span>:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/hiradix_cache.py ::HiRadixCache</span><span class="ln">writeback down: GPU → host</span></div>
  <pre><span class="kw">class</span> HiRadixCache(RadixCache):          <span class="cm"># tiering subclass of Lesson 29's RadixCache</span>

    <span class="kw">def</span> write_backup(self, node, write_back=False) -&gt; int:
        <span class="cm"># copy an about-to-be-evicted node's KV from GPU back (down) to host memory</span>
        host_indices = self.cache_controller.write(
            device_indices=node.value,        <span class="cm"># slot numbers in the GPU pool (Lesson 30)</span>
            node_id=node.id,
        )
        <span class="kw">if</span> host_indices <span class="kw">is</span> None:             <span class="cm"># host is full too -&gt; free host, then write</span>
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value, node_id=node.id,
            )
        <span class="kw">if</span> host_indices <span class="kw">is</span> <span class="kw">not</span> None:
            node.host_value = host_indices.clone()  <span class="cm"># note it now also lives in the host tier</span>
            <span class="kw">if</span> <span class="kw">not</span> write_back:
                self.inc_lock_ref(node)        <span class="cm"># lock during writeback so it isn't snatched</span>
        <span class="kw">return</span> len(host_indices)</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/hiradix_cache.py ::HiRadixCache.load_back</span><span class="ln">hit an evicted node: copy its host KV back to GPU (swap-in)</span></div>
  <pre>    <span class="kw">def</span> load_back(self, node, mem_quota=None):
        <span class="cm"># a prefix hit landed on EVICTED (host-resident) nodes. Walk up</span>
        <span class="cm"># collecting evicted-but-backuped ancestors, lock them, then copy</span>
        <span class="cm"># their host KV back onto the GPU (skip if too small / over quota).</span>
        nodes_to_load = []
        <span class="kw">while</span> node.evicted:
            nodes_to_load.insert(0, node)      <span class="cm"># host-side, needs swap-in</span>
            node = node.parent
        self.inc_lock_ref(node)                <span class="cm"># protect ancestors</span>
        host_indices = torch.cat([n.host_value <span class="kw">for</span> n <span class="kw">in</span> nodes_to_load])
        device_indices = self.cache_controller.load(host_indices, ...)  <span class="cm"># host -&gt; GPU</span>
        <span class="kw">return</span> device_indices</pre>
</div>

<p>Two concrete numbers make the trade vivid. First, the <strong>host/CPU tier is often ~10× the GPU KV capacity</strong>, so it can keep roughly an order of magnitude more prefixes cached — far more hot prefixes stay resident instead of being dropped the moment HBM fills. Second, a <strong>deep prefix hit</strong> triggers just <strong>one host→GPU copy</strong> (walking <span class="mono">node.parent</span> to grab every "host-only" ancestor on the chain at once) instead of recomputing <strong>thousands of tokens</strong> — a cheap data move replaces an expensive forward.</p>

<div class="card key">
  <div class="tag">🔑 Key takeaways</div>
  <strong>① HBM is small and precious; the naive radix tree (Lesson 29) drops on eviction and must recompute next time</strong>; CPU RAM is 10–100× bigger and disk near-infinite.
  <strong>② HiCache = <span class="mono">HiRadixCache</span> (a RadixCache subclass) + <span class="mono">HiCacheController</span></strong>, spreading the prefix cache across three tiers: GPU HBM hot → CPU RAM warm → disk cold.
  <strong>③ Two directions: on eviction <span class="mono">write_backup</span> writes KV back down; on a hit <span class="mono">load_back</span> prefetches it up</strong>, with <span class="mono">TreeNode.host_value</span> remembering which tier each run lives in.
  <strong>④ Writeback / prefetch run on background threads / copy streams, overlapped with compute (Lesson 21's spirit)</strong>, so the scheduler isn't blocked on a CPU↔GPU copy.
  <strong>⑤ Core trade: one cheap CPU→GPU copy in place of one expensive recompute</strong>. It helps most for "big prefixes, high reuse, won't fit in HBM" (long system prompts, big RAG, many-turn chats); the cost is extra RAM/disk, copy bandwidth, and cross-tier consistency, so it's an optional flag. The pool the indices point to is Lesson 30, eviction is Lesson 32, throughput payoff is Lesson 8.
</div>
"""}

LESSON_32 = {"zh": r"""
<p class="lead">
第 29 课那棵基数树看着很美：请求用完即种、来人撞上前缀就白捡。但它有个被刻意回避的难题——<strong>树不能无限长大</strong>。
它的每个节点都攥着第 30 课显存池里的一批 KV 槽位，而显存池是有限的。当槽位快用光、分配器再也凑不出新请求要的空间时，
谁该被<strong>清掉腾位</strong>？清的时候又绝不能误伤<strong>正在被使用</strong>的前缀。这一课讲的就是前缀缓存的<strong>经济学</strong>：
驱逐（evict）选谁、靠什么铁律保命、以及衡量这一切是否划算的终极指标——<strong>命中率</strong>。读懂这一课，你就读懂了第 30 课那只显存池为什么会"满"，也读懂了缓存机制最终如何兑换成第 8 课的吞吐。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把缓存想成一位<strong>清理书架的图书管理员</strong>。书架满了，她得腾地方：优先撤掉那些<strong>很久没人翻过</strong>的书（这就是 LRU——最久未访问优先）。
  但有一条铁规矩——<strong>读者正摊在桌上读的那本，绝不能撤</strong>（对应 <span class="mono">lock_ref&gt;0</span>），哪怕它落了灰、看着像该退架。
  而越多读者反复借阅<strong>同几本热门书</strong>（共享前缀），她需要重新订购（重算）的次数就越少。撤书省地方、留书省订购——她每天就在这条取舍线上拿捏。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  记住一句话：<strong>驱逐是被逼出来的，命中率是它换来的回报</strong>。驱逐不是没事就清，而是<strong>分配器告急</strong>那一刻调度器才触发；
  它只动<strong>可驱逐的叶子</strong>、按策略（默认 LRU）排序、跳过被锁住的节点，把腾出的 KV 槽位还给池（第 30 课）。
  命中率则衡量"<strong>有多少 prompt token 是从缓存白拿、没重算的</strong>"。高前缀共享（第 7/29 课）+ 缓存感知调度（第 20 课）→ 高命中率 → 少重算 → 高吞吐（第 8 课）；
  HiCache（第 31 课）则把被驱逐的前缀沉到下层，进一步抬高<strong>有效</strong>命中率。一句话：<strong>驱逐管腾地方、命中率管收回报，两者都围着"显存有限"这个根本约束打转</strong>。
</div>

<h2>为什么要驱逐：树长在有限的池子上</h2>
<p>第 29 课反复强调"树是索引、池是仓库"。索引可以画得很大，但它指向的<strong>仓库是有限的</strong>——每个 <span class="mono">TreeNode</span> 的 value 是一批真实占着显存的 KV 槽位。
请求不断把算过的前缀<strong>种进树</strong>，节点越积越多，池里的空闲槽位越来越少。直到某一刻，分配器面对一条新请求，<strong>再也凑不出它需要的连续槽位</strong>——这就是触发驱逐的信号。
注意这个触发是<strong>按需、被动</strong>的：不是定时清理，而是"<strong>不腾就跑不动</strong>"时才动手。调度器会算出"还差多少 token 的空间"，把这个数交给 <span class="mono">evict</span>，让它至少回收这么多槽位。</p>
<p>这也解释了缓存为什么是一种<strong>赌注</strong>而非纯收益。把前缀留在树上，它占着 HBM、挤掉了本可用于<strong>更高并发</strong>的空间；把它驱逐掉，未来若再撞上同样前缀就得<strong>从头重算几千 token</strong>。
缓存的全部艺术，就是<strong>赌哪些前缀还会再被用到</strong>、值得继续占着显存，哪些已经凉了、该让位给新请求。驱逐策略，就是这场赌局的下注规则。</p>
<p>顺带厘清一个常见误解：驱逐<strong>不会丢失任何还在生成中的请求的数据</strong>。它回收的永远是<strong>历史请求留下、当下没人在用</strong>的"沉淀前缀"。
一条请求只要还在跑，它的整条前缀链都被锁着（<span class="mono">lock_ref&gt;0</span>），驱逐<strong>根本看不见它们</strong>。所以驱逐发生时，你损失的不是正确性，而仅仅是"<strong>未来可能的复用机会</strong>"——
那段被清掉的 KV 若以后真的又被某请求命中，就得重算；若再没人用到，那这次驱逐就是纯赚。正因为收益和损失都落在<strong>不确定的未来</strong>，缓存才被称为一场赌局，而 LRU 这类策略不过是<strong>用过去的访问规律去猜未来</strong>：最久没人碰的，多半将来也不会有人碰。</p>

<h2>驱逐什么：EvictionStrategy 给可驱逐叶子排序</h2>
<p>具体清谁？由一个 <span class="mono">EvictionStrategy</span> 说了算。它做的事只有一件：给所有<strong>可驱逐的叶子节点</strong>算一个 <span class="mono">get_priority</span>（优先级），
然后把优先级最低的先清掉。默认策略是 <strong>LRU</strong>（<span class="mono">LRUStrategy</span>）——优先级直接取 <span class="mono">last_access_time</span>，<strong>最久没被访问的排最前、最先被赶走</strong>。
除 LRU 外还有别的玩法：<strong>LFU</strong>（最不常用优先，按命中次数）、<strong>FIFO</strong>（按创建时间，最早进的先走）、<strong>MRU</strong>（最近用过的反而先走，某些扫描型负载下有用）。</p>
<p>为什么默认偏偏选 LRU 而不是看起来更"聪明"的 LFU？因为 LRU 的假设最贴合真实的服务流量：<strong>时间局部性</strong>——刚被用过的前缀，短期内极可能再被用到（同一个系统提示正被一波请求连续命中），而很久没人碰的，多半已经凉透。
LFU 虽然记得"谁历史上最热门"，却容易被<strong>陈旧的热门</strong>拖累：某段前缀昨天被命中过几千次、今天已无人问津，LFU 仍因它的高命中计数死死护着它，白占显存。LRU 只看"最近一次"，对流量的冷热切换<strong>反应更快、更不容易被历史包袱拖住</strong>，实现也只需一个时间戳，开销极低。
这就是为什么绝大多数缓存系统、包括这里的前缀缓存，都把 LRU 作为稳妥的默认，只在特定负载画像下才考虑切换到别的策略。</p>
<p>这里有个关键约束：<strong>只有叶子可被驱逐</strong>。为什么不能直接清一个深处的公共前缀？因为只要还有<strong>更长的路径挂在它下面</strong>，那条长路径就<strong>依赖</strong>这段前缀的 KV——
你不能把一段前缀抽掉，却让它的孩子悬空。所以回收永远<strong>从叶子往根</strong>进行：清掉一个叶子，它的父节点若就此变成新叶子、且没被锁，才轮到它成为下一个候选。
清掉一个节点做两件事：把它的 KV 槽位 <span class="mono">free</span> 回分配器（第 30 课），并把它从树上摘除。</p>

<table class="t">
  <tr><th>驱逐策略</th><th>get_priority 取什么</th><th>偏向清掉谁 / 何时用</th></tr>
  <tr><td class="mono">LRUStrategy（默认）</td><td>last_access_time</td><td>最久未访问的先走；通用、最稳，贴合"冷前缀该让位"的直觉</td></tr>
  <tr><td class="mono">LFUStrategy</td><td>(hit_count, last_access)</td><td>命中次数最少的先走；偏向长期保住热门前缀</td></tr>
  <tr><td class="mono">FIFOStrategy</td><td>creation_time</td><td>最早种进树的先走；不看冷热，只看资历</td></tr>
  <tr><td class="mono">MRUStrategy</td><td>-last_access_time</td><td>最近用过的反而先走；少数顺序扫描场景下避免污染</td></tr>
</table>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>空闲槽位告急</h4><p>分配器凑不出新请求要的连续槽位 → 调度器算出还差多少 token，触发 <span class="mono">evict</span> 至少回收这么多。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>按优先级挑叶子</h4><p>把所有<strong>可驱逐叶子</strong>按 <span class="mono">get_priority</span> 排进一个堆（默认 LRU = 时间最老的在堆顶），优先级最低者先出。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>跳过被锁的</h4><p><span class="mono">lock_ref&gt;0</span> 的节点压根不在<strong>可驱逐叶子</strong>集合里——在用的前缀天然被排除，绝不会被选中。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>释放槽位、上浮候选</h4><p>把叶子的 value <span class="mono">free</span> 回池、从树摘除；它父节点若变成新叶且未锁，再压回堆，继续回收直到够数。</p></div></div>
</div>

<h2>铁律：lock_ref&gt;0 的节点永不被驱逐</h2>
<p>整套驱逐里有一条<strong>不可逾越的铁律</strong>：一个 <span class="mono">lock_ref &gt; 0</span> 的节点——也就是当下正被某条在跑的请求使用的前缀（第 29 课）——<strong>永远不可驱逐</strong>。
道理很硬：那段 KV 正被<strong>一次在飞的前向</strong>读着，你若把它的槽位回收了，正在算的请求就会读到一片<strong>被覆盖的垃圾</strong>，结果直接崩坏。
所以"在用"和"可驱逐"是<strong>互斥</strong>的两种状态。第 29 课讲过的 <span class="mono">inc_lock_ref</span> / <span class="mono">dec_lock_ref</span>，正是在这两个集合之间搬动节点：
请求开始用某前缀，就从命中节点向上逐个 <span class="mono">lock_ref += 1</span>，把整条链从<strong>可驱逐集</strong>挪进<strong>受保护集</strong>；请求结束再减回去，归零的节点才重新成为驱逐候选。</p>
<p>把这条铁律和 LRU 合起来看就通透了：<strong>压力之下，被锁的（在用的）前缀一定活下来，而冷清、没人锁的叶子被回收</strong>。
一个节点哪怕 <span class="mono">last_access_time</span> 很老、看着最该淘汰，只要 <span class="mono">lock_ref&gt;0</span> 就<strong>动它不得</strong>；反过来，一个刚被某历史请求种下、但此刻无人在用的叶子，lock_ref 为 0，就是最理想的回收对象。
这正是第 29 课那个区分的延续：lock_ref 计的是"<strong>正在飞的引用</strong>"，不是"<strong>历史用过几次</strong>"——驱逐要回收的，恰恰是<strong>留在树上、当下却没人用</strong>的那些。</p>
<p>举个具体场景把铁律讲实：设想一段很长的共享系统提示，正被<strong>二十条并发请求</strong>同时解码。它那条前缀链上挂着 <span class="mono">lock_ref=20</span>，于是无论引擎显存多紧张，这条链都<strong>动不得</strong>——这正是我们要的，因为清掉它会同时弄崩二十次在飞的前向。
随着这些请求一条条结束，每条都调一次 <span class="mono">dec_lock_ref</span>，把计数 20→19→…→0 地往下减；只有当<strong>最后一条</strong>也结束、计数归零，这条链才重新回到可驱逐集，而且还得它本身是叶子才行。
这就解释了为什么<strong>热门共享前缀天然扛得住压力</strong>：一段前缀越忙、锁它的请求越多，它被钉得越牢。缓存因此<strong>恰好保护了最值得保护的东西</strong>——这甚至不是刻意设计的优化，而是"保证正确性"顺带产生的副作用。</p>

<div class="cellgroup">
  <div class="cg-cap"><b>显存吃紧时一棵树如何收缩</b>：高亮的是 <span class="mono">lock_ref&gt;0</span>、正在被使用、<strong>必定存活</strong>的节点；冷清未锁的叶子被摘掉腾槽位</div>
  <div class="cells"><span class="lab">受保护链</span><span class="cell hl">root</span><span class="cell hl">你是一个</span><span class="cell hl">助手·在跑</span><span class="sep">‖</span><span class="cell q">lock_ref&gt;0：在飞请求正读它 → 驱逐碰不到</span></div>
  <div class="cells"><span class="lab">冷叶 A</span><span class="cell">旧前缀·t₂</span><span class="sep">✂</span><span class="cell q">lock_ref=0 且最久未访问 → LRU 堆顶，先被清</span></div>
  <div class="cells"><span class="lab">冷叶 B</span><span class="cell">旧前缀·t₅</span><span class="sep">✂</span><span class="cell q">lock_ref=0、次老 → 若还不够，接着清；槽位 free 回池</span></div>
  <div class="cells"><span class="lab">清完之后</span><span class="cell hl">root</span><span class="cell hl">你是一个</span><span class="cell hl">助手·在跑</span><span class="sep">→</span><span class="cell q">只剩在用链；空出的槽位还给分配器，容下新请求</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="LRU 驱逐：可回收的冷叶被释放，被锁定的在用节点幸存">
    <line x1="350" y1="66" x2="220" y2="108" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="350" y1="66" x2="520" y2="108" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="220" y1="146" x2="220" y2="190" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="520" y1="146" x2="430" y2="190" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="520" y1="146" x2="620" y2="190" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="305" y="32" width="90" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="350" y="54" text-anchor="middle" class="mono">root</text>
    <rect x="165" y="108" width="110" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="220" y="126" text-anchor="middle">你是一个</text>
    <text x="220" y="140" text-anchor="middle" class="mono" style="font-size:10px">lock&gt;0</text>
    <rect x="465" y="108" width="110" height="38" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="126" text-anchor="middle">共享前缀</text>
    <text x="520" y="140" text-anchor="middle" class="mono" style="font-size:10px">父·可上浮</text>
    <rect x="165" y="190" width="110" height="44" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="220" y="210" text-anchor="middle">助手·在跑</text>
    <text x="220" y="226" text-anchor="middle" class="mono" style="font-size:10px">lock=20 幸存</text>
    <rect x="375" y="190" width="110" height="44" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="430" y="210" text-anchor="middle" style="fill:var(--red);font-weight:700">冷叶 t=2</text>
    <text x="430" y="226" text-anchor="middle" class="mono" style="font-size:10px">LRU→释放</text>
    <rect x="565" y="190" width="110" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="620" y="210" text-anchor="middle">冷叶 t=5</text>
    <text x="620" y="226" text-anchor="middle" class="mono" style="font-size:10px">lock=0 次老</text>
    <text x="24" y="40" style="fill:var(--muted);font-size:12px">显存告急 → 自底向上回收</text>
    <text x="24" y="272" style="fill:var(--red);font-size:12px">红：最久未访问的冷叶，先被 free</text>
    <text x="430" y="272" style="fill:var(--teal);font-size:12px">青：lock&gt;0 在用链，驱逐看不见</text>
  </svg>
  <div class="figcap"><b>图 32·1 · LRU 驱逐冷叶，锁定的在用节点幸存</b> — 显存告急时，最久未访问、<span class="mono">lock_ref=0</span> 的冷叶（红）被选中并 <span class="mono">free</span> 回池；而 <span class="mono">lock_ref&gt;0</span> 的在用前缀链（青）即便更老也绝不被驱逐。释放冷叶后其父节点变成新叶、上浮为下一候选。</div>
</div>

<h2>命中率：这一切的回报，以及那笔取舍</h2>
<p>绕了一圈，驱逐到底图什么？图的是<strong>命中率</strong>——衡量"<strong>有多少 prompt token 是从缓存直接拿到、不必重算的</strong>"那个比例。
命中率是把前面所有努力串起来的<strong>收益指标</strong>：高前缀共享（第 7/29 课）让更多 token 能复用；缓存感知调度（第 20 课）则更进一步，它<strong>主动重排队列</strong>、优先把"前缀已在树上"的请求放进批，让命中真正发生。
命中率越高 → 重算越少 → 吞吐越高（第 8 课）。而 HiCache（第 31 课）的价值，正是把被驱逐的前缀沉到 CPU/磁盘，使<strong>有效命中率</strong>不因一次驱逐就归零——再撞上时拷回来即可，省掉一次重算。</p>
<p>但命中率不是免费的。它背后是一笔<strong>核心取舍</strong>：<strong>驱逐了将来重算，还是留在 HBM 里</strong>。
<strong>留</strong>的代价是占显存——前缀霸着 HBM，留给并发的空间就少，批量做不大；<strong>驱</strong>的代价是未来某次命中变成 miss，得吃一次几千 token 的前向重算。
整个内存这一部分（第 29–32 课）讲的所有机制——基数树共享、分页池、HiCache 分层、LRU 驱逐——本质都是<strong>为了把这笔取舍做得更好</strong>：尽量多留有用的、尽量准地驱冷的，让显存这点稀缺资源换来最高的命中率与吞吐。</p>
<p>还有一个微妙之处把它接回调度：命中率不只是缓存自己的事，它还取决于<strong>请求抵达缓存的先后顺序</strong>。设想两条请求共享一段长前缀，但调度器在它们中间插进了十几条无关请求——等第二条到来时，那段共享前缀可能<strong>早已被驱逐</strong>，本该稳拿的命中就变成了 miss。
这正是缓存感知调度（第 20 课）要<strong>重排队列</strong>、把共享前缀的请求<strong>尽量靠拢</strong>的原因：它主动制造出那些"缓存独自会因驱逐而错失"的命中。
所以驱逐策略和调度策略其实是同一项优化的两半：驱逐这边决定<strong>留什么</strong>，调度那边决定<strong>何时问</strong>，二者合力，才能把有效命中率、进而把吞吐（第 8 课）顶到硬件允许的上限。</p>

<div class="cols">
  <div class="col"><h4>驱逐（evict & 将来重算）</h4><p>立刻把冷前缀的 KV 槽位还给池，<strong>腾出 HBM</strong>给更高并发；代价是若它将来再被命中，要<strong>从头重算几千 token</strong>（一次昂贵前向）。赌"它不会再来"。</p></div>
  <div class="col"><h4>保留（keep in HBM）</h4><p>把前缀一直留在显存里，再撞上就<strong>零重算</strong>、直接复用；代价是它<strong>长期占着 HBM</strong>，挤掉了本可用于更大批量、更高并发的空间。赌"它还会再来"。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="命中率越高，需重算的 prefill token 越少，吞吐越高">
    <line x1="90" y1="50" x2="90" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="90" y1="240" x2="720" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="120" y="204" width="64" height="36" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="270" y="148" width="64" height="92" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="420" y="90" width="64" height="150" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="570" y="40" width="64" height="200" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <line x1="152" y1="204" x2="302" y2="148" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="302" y1="148" x2="452" y2="90" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="452" y1="90" x2="602" y2="40" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="152" y="258" text-anchor="middle" class="mono" style="font-size:11px">0%</text>
    <text x="302" y="258" text-anchor="middle" class="mono" style="font-size:11px">30%</text>
    <text x="452" y="258" text-anchor="middle" class="mono" style="font-size:11px">60%</text>
    <text x="602" y="258" text-anchor="middle" class="mono" style="font-size:11px">90%</text>
    <text x="24" y="44" style="fill:var(--muted);font-size:12px">吞吐 tokens/s</text>
    <text x="405" y="282" text-anchor="middle" style="fill:var(--muted);font-size:12px">前缀缓存命中率</text>
    <text x="112" y="84" style="fill:var(--teal);font-size:12px">命中率越高</text>
    <text x="112" y="102" style="fill:var(--teal);font-size:12px">→ 需重算的 prefill token 越少</text>
  </svg>
  <div class="figcap"><b>图 32·2 · 命中率越高，吞吐越高</b> — 横轴是前缀缓存命中率（0%→90%），纵轴是吞吐（tokens/s）。命中率上升 → 需要重算的 prefill token 减少 → 吞吐随之爬升；命中率越高，要预填的 token 越少，这正是驱逐与缓存机制最终兑换出的回报。</div>
</div>

<p>真正决定"清谁"的那行代码出奇地短——驱逐策略的全部分歧，就浓缩在 <span class="mono">get_priority</span> 返回什么：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/evict_policy.py ::LRUStrategy</span><span class="ln">优先级 = 最近访问时间</span></div>
  <pre><span class="kw">class</span> <span class="st">EvictionStrategy</span>(ABC):
    <span class="cm"># 抽象基类：所有策略只需回答“谁的优先级最低、该先被清”</span>
    <span class="kw">@abstractmethod</span>
    <span class="kw">def</span> get_priority(self, node: TreeNode) -&gt; Union[float, Tuple]:
        <span class="kw">pass</span>

<span class="kw">class</span> <span class="st">LRUStrategy</span>(EvictionStrategy):
    <span class="kw">def</span> get_priority(self, node: TreeNode) -&gt; float:
        <span class="kw">return</span> node.last_access_time   <span class="cm"># 最久未访问 → 值最小 → 堆顶 → 先驱逐</span>

<span class="kw">class</span> <span class="st">LFUStrategy</span>(EvictionStrategy):
    <span class="kw">def</span> get_priority(self, node: TreeNode) -&gt; Tuple[int, float]:
        <span class="kw">return</span> (node.hit_count, node.last_access_time)  <span class="cm"># 命中最少者先走</span></pre>
</div>

<p>上面那行 <span class="mono">get_priority</span> 只回答"<strong>谁该先走</strong>"；真正驱动整轮回收的，是 <span class="mono">RadixCache.evict</span> 里那个<strong>堆循环</strong>——它把可驱逐叶子按优先级堆好，逐个弹出最低者、<span class="mono">free</span> 掉 KV 槽位、把叶子摘除，再把刚变成叶子的父节点压回堆，直到回收够 <span class="mono">num_tokens</span> 个槽位：</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.evict</span><span class="ln">按优先级堆淘汰可回收叶子，释放 KV 槽</span></div>
  <pre><span class="kw">def</span> evict(self, params: <span class="st">EvictParams</span>) -&gt; <span class="st">EvictResult</span>:
    <span class="cm"># 丢掉优先级最低的叶子，回收 `num_tokens` 个 KV 槽位。</span>
    <span class="cm"># 只有可驱逐叶子是候选；被锁住（在用）的节点</span>
    <span class="cm"># 不在集合里，所以在飞请求绝不会丢失自己的缓存。</span>
    heap = [(self.eviction_strategy.get_priority(n), n)
            <span class="kw">for</span> n <span class="kw">in</span> self.evictable_leaves]
    heapq.heapify(heap)
    num_evicted = 0
    <span class="kw">while</span> num_evicted &lt; params.num_tokens <span class="kw">and</span> heap:
        _, x = heapq.heappop(heap)
        self.token_to_kv_pool_allocator.free(x.value)   <span class="cm"># 释放 KV 槽位</span>
        num_evicted += len(x.value)
        self._delete_leaf(x)
        <span class="kw">if</span> <span class="kw">not</span> x.parent.children <span class="kw">and</span> x.parent.lock_ref == 0:
            heapq.heappush(heap, (get_priority(x.parent), x.parent))   <span class="cm"># 父变新叶 → 上浮</span>
    <span class="kw">return</span> <span class="st">EvictResult</span>(num_tokens_evicted=num_evicted)</pre>
</div>

<p>用具体数字感受这套机制：一段 <strong>2000 token 的共享系统提示</strong>在 90% 命中率下，意味着绝大多数请求<strong>直接跳过约 2000 个 prefill token</strong>、几乎零重算就进入解码；而正在跑的请求把这条前缀链锁着（<span class="mono">lock_ref&gt;0</span>），所以哪怕显存告急、<span class="mono">evict</span> 正忙，它也<strong>绝不会在生成中途被清掉</strong>——堆里弹出的只会是没人锁、最久没碰的冷叶。</p>

<div class="card key">
  <div class="tag">🔑 本课要点</div>
  <strong>① 驱逐是被逼的、按需的</strong>：节点占着第 30 课池里的 KV 槽位，分配器凑不出新空间时，调度器才触发 <span class="mono">evict</span> 回收若干槽位。
  <strong>② 清谁由 EvictionStrategy 排序</strong>：默认 LRU（<span class="mono">get_priority = last_access_time</span>，最久未访问先走），另有 LFU / FIFO / MRU；<strong>只有叶子可驱逐</strong>，回收从叶往根。
  <strong>③ 铁律：lock_ref&gt;0 永不驱逐</strong>——在飞的前向正读那段 KV，回收即崩坏；inc/dec_lock_ref 在可驱逐集与受保护集之间搬节点（第 29 课）。
  <strong>④ 命中率是回报指标</strong>：从缓存白拿的 token 占比，由高共享（第 7/29 课）+ 缓存感知调度（第 20 课）抬高 → 少重算 → 高吞吐（第 8 课）；HiCache（第 31 课）抬高有效命中率。
  <strong>⑤ 核心取舍：驱逐后重算 vs 留在 HBM</strong>——留占显存挤掉并发，驱省显存赔上重算；整个内存部分都在把这笔取舍做好。
</div>
""",
             "en": r"""
<p class="lead">
Lesson 29's radix tree looked elegant: requests plant prefixes when done, newcomers that hit them get a free ride. But it sidestepped one hard problem — <strong>the tree can't grow forever</strong>.
Every node holds a batch of KV slots from Lesson 30's pool, and the pool is finite. When free slots run low and the allocator can no longer satisfy a new request, <strong>who gets reclaimed</strong> to make room?
And the reclaim must never touch a prefix that is <strong>currently in use</strong>. This lesson is the <strong>economics</strong> of the prefix cache:
who <strong>evict</strong> picks, what iron rule keeps a node alive, and the ultimate metric that says whether any of it pays off — the <strong>hit rate</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of the cache as a <strong>librarian weeding the shelves</strong>. The shelves are full, so she must make room: she first removes the books <strong>nobody has opened in ages</strong> (that's LRU — least-recently-used first).
  But one iron rule — <strong>never the book a patron is currently reading at a desk</strong> (that's <span class="mono">lock_ref&gt;0</span>), however dusty and overdue it looks.
  And the more readers keep reusing the <strong>same popular books</strong> (shared prefixes), the fewer re-orders (recomputes) she needs. Weeding frees shelf space; keeping saves re-orders — every day she balances on that line.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Keep one line in mind: <strong>eviction is forced on you; hit rate is the payoff it buys</strong>. Eviction isn't idle cleanup — it fires only the moment the <strong>allocator is starved</strong>;
  it touches only <strong>evictable leaves</strong>, orders them by a strategy (LRU by default), skips locked nodes, and returns the freed KV slots to the pool (Lesson 30).
  Hit rate measures "<strong>what fraction of prompt tokens were served from the cache without recompute</strong>." High prefix sharing (Lessons 7/29) + cache-aware scheduling (Lesson 20) → high hit rate → less recompute → higher throughput (Lesson 8);
  HiCache (Lesson 31) sinks evicted prefixes to lower tiers, raising the <strong>effective</strong> hit rate further.
</div>

<h2>Why evict: the tree grows on a finite pool</h2>
<p>Lesson 29 hammered "tree is the index, pool is the warehouse." The index can be drawn huge, but it points into a <strong>finite warehouse</strong> — each <span class="mono">TreeNode</span>'s value is a batch of real, HBM-occupying KV slots.
Requests keep <strong>planting</strong> computed prefixes into the tree, nodes pile up, free slots dwindle. Until one moment the allocator faces a new request and <strong>can no longer assemble the slots it needs</strong> — that is the eviction signal.
Note the trigger is <strong>on-demand and reactive</strong>: not scheduled cleanup, but action taken only when "<strong>you can't run unless you free something</strong>." The scheduler computes "how many tokens short we are" and hands that number to <span class="mono">evict</span>, asking it to reclaim at least that much.</p>
<p>This also explains why caching is a <strong>bet</strong>, not pure gain. Keep a prefix in the tree and it occupies HBM, squeezing out space that could serve <strong>more concurrency</strong>; evict it and a future hit on the same prefix means <strong>recomputing thousands of tokens from scratch</strong>.
The whole art of caching is <strong>betting which prefixes will be reused</strong> and deserve to keep their HBM, versus which have gone cold and should yield to new requests. The eviction strategy is the betting rule of this gamble.</p>

<h2>What to evict: EvictionStrategy orders the evictable leaves</h2>
<p>Who exactly gets cleared? An <span class="mono">EvictionStrategy</span> decides. It does just one thing: compute a <span class="mono">get_priority</span> for every <strong>evictable leaf node</strong>,
then clear the lowest-priority ones first. The default is <strong>LRU</strong> (<span class="mono">LRUStrategy</span>) — priority is simply <span class="mono">last_access_time</span>, so the <strong>oldest-untouched goes first</strong>.
Beyond LRU there are other flavors: <strong>LFU</strong> (least-frequently-used, by hit count), <strong>FIFO</strong> (by creation time, first-in-first-out), and <strong>MRU</strong> (most-recently-used goes first, useful for some scan workloads).</p>
<p>Why default to LRU rather than the seemingly "smarter" LFU? Because LRU's assumption best matches real serving traffic: <strong>temporal locality</strong> — a just-used prefix is very likely to be used again soon (one system prompt hit by a burst of requests), while something untouched for ages is probably stone cold.
LFU remembers "who was historically hottest," but is easily dragged down by <strong>stale popularity</strong>: a prefix hit thousands of times yesterday but idle today still gets fiercely guarded by its high count, wasting HBM. LRU looks only at "the last touch," so it <strong>reacts faster to shifting hot/cold patterns</strong> and isn't held hostage by history — and it needs only a timestamp, almost free to maintain.
That's why most caches, including this prefix cache, pick LRU as the safe default and switch strategies only under specific workload profiles.</p>
<p>A key constraint: <strong>only leaves are evictable</strong>. Why can't you just clear a deep common prefix? Because as long as a <strong>longer path still hangs below it</strong>, that long path <strong>depends</strong> on this prefix's KV —
you can't pull a prefix out and leave its children dangling. So reclaim always proceeds <strong>from leaves toward the root</strong>: clear a leaf, and only if its parent becomes a new leaf and isn't locked does the parent become the next candidate.
Clearing a node does two things: <span class="mono">free</span> its KV slots back to the allocator (Lesson 30), and remove it from the tree.</p>
<p>It's worth clearing up a common worry: eviction <strong>never loses data for any request still generating</strong>. What it reclaims is always the "settled prefixes" <strong>left by past requests and used by no one now</strong>.
As long as a request is running, its entire prefix chain is locked (<span class="mono">lock_ref&gt;0</span>) and eviction <strong>simply can't see it</strong>. So when eviction happens you lose not correctness but merely "<strong>a possible future reuse</strong>" —
if that cleared KV is later hit by some request, it must be recomputed; if no one ever needs it again, the eviction was pure profit. Because both gain and loss land in an <strong>uncertain future</strong>, caching is called a gamble, and strategies like LRU just <strong>guess the future from the past</strong>: whatever went longest untouched probably stays untouched.</p>

<table class="t">
  <tr><th>Strategy</th><th>get_priority returns</th><th>Favors evicting / when</th></tr>
  <tr><td class="mono">LRUStrategy (default)</td><td>last_access_time</td><td>Oldest-untouched first; general, most robust, matches "cold prefix yields"</td></tr>
  <tr><td class="mono">LFUStrategy</td><td>(hit_count, last_access)</td><td>Least-hit first; keeps hot prefixes around longer</td></tr>
  <tr><td class="mono">FIFOStrategy</td><td>creation_time</td><td>Earliest-planted first; by seniority, not temperature</td></tr>
  <tr><td class="mono">MRUStrategy</td><td>-last_access_time</td><td>Most-recent first; avoids pollution in some sequential scans</td></tr>
</table>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Free slots run low</h4><p>Allocator can't assemble the slots a new request needs → scheduler computes how many tokens short, fires <span class="mono">evict</span> to reclaim at least that many.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Pick leaves by priority</h4><p>Put all <strong>evictable leaves</strong> into a heap keyed by <span class="mono">get_priority</span> (default LRU = oldest time on top); lowest priority pops first.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Skip the locked</h4><p>A <span class="mono">lock_ref&gt;0</span> node isn't even in the <strong>evictable-leaf</strong> set — in-use prefixes are excluded by construction and can never be picked.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Free slots, float candidates up</h4><p><span class="mono">free</span> the leaf's value back to the pool and remove it; if its parent becomes a new unlocked leaf, push it back into the heap, repeat until enough.</p></div></div>
</div>

<h2>The iron rule: a lock_ref&gt;0 node is never evicted</h2>
<p>One <strong>inviolable rule</strong> governs all eviction: a node with <span class="mono">lock_ref &gt; 0</span> — a prefix currently used by a running request (Lesson 29) — is <strong>never evictable</strong>.
The reason is hard: that KV is being read by an <strong>in-flight forward</strong>; reclaim its slots and the running request reads <strong>overwritten garbage</strong>, corrupting the result.
So "in use" and "evictable" are <strong>mutually exclusive</strong> states. Lesson 29's <span class="mono">inc_lock_ref</span> / <span class="mono">dec_lock_ref</span> are exactly what move nodes between the two sets:
when a request starts using a prefix, it walks up from the hit node doing <span class="mono">lock_ref += 1</span>, moving the whole chain from the <strong>evictable set</strong> into the <strong>protected set</strong>; when it ends it decrements back, and a node that hits zero becomes an eviction candidate again.</p>
<p>Put this rule together with LRU and it clicks: <strong>under pressure, locked (in-use) prefixes survive, while cold unlocked leaves get reclaimed</strong>.
A node may have a very old <span class="mono">last_access_time</span> and look most disposable, yet if <span class="mono">lock_ref&gt;0</span> you <strong>cannot touch it</strong>; conversely a leaf just planted by some past request but used by nobody now has lock_ref 0 and is the ideal reclaim target.
This continues Lesson 29's distinction: lock_ref counts "<strong>in-flight references</strong>," not "<strong>how many times used historically</strong>" — what eviction reclaims is exactly what's <strong>left in the tree yet used by no one now</strong>.</p>
<p>One more practical angle makes the rule concrete: imagine a long shared system prompt being decoded by twenty concurrent requests. Its prefix chain carries <span class="mono">lock_ref=20</span>, so no matter how memory-starved the engine gets, that chain <strong>cannot be touched</strong> — which is exactly what we want, since evicting it would corrupt twenty in-flight forwards at once. As those requests finish one by one, each calls <span class="mono">dec_lock_ref</span>, dropping the count 20→19→…→0; only when the very last one ends does the chain rejoin the evictable set, and even then only if it's a leaf. This is why <strong>hot shared prefixes naturally survive pressure</strong>: the busier a prefix is, the higher its lock count, the more firmly it's pinned — the cache protects exactly what's most worth protecting, for free, as a side effect of correctness.</p>

<div class="cellgroup">
  <div class="cg-cap"><b>How a tree shrinks under memory pressure</b>: highlighted nodes have <span class="mono">lock_ref&gt;0</span>, are in use, and <strong>must survive</strong>; cold unlocked leaves are dropped to free slots</div>
  <div class="cells"><span class="lab">Protected chain</span><span class="cell hl">root</span><span class="cell hl">"you are a"</span><span class="cell hl">assistant·live</span><span class="sep">‖</span><span class="cell q">lock_ref&gt;0: an in-flight request reads it → eviction can't touch</span></div>
  <div class="cells"><span class="lab">Cold leaf A</span><span class="cell">old prefix·t₂</span><span class="sep">✂</span><span class="cell q">lock_ref=0 and oldest-untouched → top of LRU heap, cleared first</span></div>
  <div class="cells"><span class="lab">Cold leaf B</span><span class="cell">old prefix·t₅</span><span class="sep">✂</span><span class="cell q">lock_ref=0, next-oldest → cleared next if still short; slots freed</span></div>
  <div class="cells"><span class="lab">After clearing</span><span class="cell hl">root</span><span class="cell hl">"you are a"</span><span class="cell hl">assistant·live</span><span class="sep">→</span><span class="cell q">only the in-use chain remains; freed slots return to allocator for new work</span></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="LRU evicts evictable cold leaves; locked in-use nodes survive">
    <line x1="350" y1="66" x2="220" y2="108" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="350" y1="66" x2="520" y2="108" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="220" y1="146" x2="220" y2="190" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="520" y1="146" x2="430" y2="190" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <line x1="520" y1="146" x2="620" y2="190" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="305" y="32" width="90" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="350" y="54" text-anchor="middle" class="mono">root</text>
    <rect x="165" y="108" width="110" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="220" y="126" text-anchor="middle">you are a</text>
    <text x="220" y="140" text-anchor="middle" class="mono" style="font-size:10px">lock&gt;0</text>
    <rect x="465" y="108" width="110" height="38" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="520" y="126" text-anchor="middle">shared pfx</text>
    <text x="520" y="140" text-anchor="middle" class="mono" style="font-size:10px">parent·floats up</text>
    <rect x="165" y="190" width="110" height="44" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="220" y="210" text-anchor="middle">asst·live</text>
    <text x="220" y="226" text-anchor="middle" class="mono" style="font-size:10px">lock=20 survives</text>
    <rect x="375" y="190" width="110" height="44" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="430" y="210" text-anchor="middle" style="fill:var(--red);font-weight:700">cold leaf t=2</text>
    <text x="430" y="226" text-anchor="middle" class="mono" style="font-size:10px">LRU→freed</text>
    <rect x="565" y="190" width="110" height="44" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="620" y="210" text-anchor="middle">cold leaf t=5</text>
    <text x="620" y="226" text-anchor="middle" class="mono" style="font-size:10px">lock=0 older</text>
    <text x="24" y="40" style="fill:var(--muted);font-size:12px">HBM starved → reclaim bottom-up</text>
    <text x="24" y="272" style="fill:var(--red);font-size:12px">red: oldest cold leaf → freed first</text>
    <text x="430" y="272" style="fill:var(--teal);font-size:12px">teal: lock&gt;0 chain survives evict</text>
  </svg>
  <div class="figcap"><b>Fig 32·1 · LRU evicts cold leaves; locked in-use nodes survive</b> — under memory pressure the oldest-untouched, <span class="mono">lock_ref=0</span> cold leaf (red) is selected and <span class="mono">free</span>d back to the pool; the in-use prefix chain with <span class="mono">lock_ref&gt;0</span> (teal) is never evicted even if older. After a leaf is freed its parent becomes a new leaf and floats up as the next candidate.</div>
</div>

<h2>Hit rate: the payoff, and that one trade</h2>
<p>After all this — what is eviction for? For the <strong>hit rate</strong> — the fraction measuring "<strong>how many prompt tokens were served straight from the cache, with no recompute</strong>."
Hit rate is the <strong>payoff metric</strong> that ties every earlier effort together: high prefix sharing (Lessons 7/29) lets more tokens be reused; cache-aware scheduling (Lesson 20) goes further, <strong>reordering the queue</strong> to favor requests whose prefix is already in the tree, so hits actually happen.
Higher hit rate → less recompute → higher throughput (Lesson 8). And HiCache's (Lesson 31) value is to sink evicted prefixes to CPU/disk so the <strong>effective</strong> hit rate doesn't drop to zero on one eviction — a re-hit just copies it back, skipping a recompute.</p>
<p>But hit rate isn't free. Behind it sits a <strong>core trade</strong>: <strong>evict-and-recompute-later vs keep-in-HBM</strong>.
<strong>Keeping</strong> costs HBM — the prefix hogs memory, leaving less room for concurrency, so batches stay small; <strong>evicting</strong> costs a future hit turning into a miss, paying a thousands-token forward recompute.
Every mechanism in this whole memory part (Lessons 29–32) — radix-tree sharing, paged pools, HiCache tiering, LRU eviction — fundamentally exists <strong>to make this trade well</strong>: keep as much useful as possible, evict the cold as accurately as possible, turning scarce HBM into the highest hit rate and throughput.</p>
<p>A subtle point ties it all back to scheduling: hit rate isn't only the cache's job — it depends on the <strong>order requests arrive at the cache</strong>. If two requests share a long prefix but the scheduler interleaves a dozen unrelated requests between them, the shared prefix may be evicted before the second one arrives, turning a guaranteed hit into a miss. That's precisely why cache-aware scheduling (Lesson 20) <strong>reorders the queue</strong> to keep prefix-sharing requests close together — it actively manufactures hits that the cache alone would have lost to eviction. Eviction policy and scheduling policy are thus two halves of one optimization: the eviction side decides <strong>what to keep</strong>, the scheduling side decides <strong>when to ask</strong>, and only together do they push the effective hit rate — and therefore throughput (Lesson 8) — as high as the hardware allows.</p>

<div class="cols">
  <div class="col"><h4>Evict (& recompute later)</h4><p>Immediately return a cold prefix's KV slots to the pool, <strong>freeing HBM</strong> for more concurrency; the cost is that if it's hit again later you must <strong>recompute thousands of tokens</strong> (one expensive forward). Bet "it won't come back."</p></div>
  <div class="col"><h4>Keep (in HBM)</h4><p>Hold the prefix in memory; a re-hit is <strong>zero recompute</strong>, pure reuse; the cost is it <strong>occupies HBM long-term</strong>, squeezing out space for bigger batches and higher concurrency. Bet "it'll come back."</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="higher hit rate means fewer prefill tokens to recompute and higher throughput">
    <line x1="90" y1="50" x2="90" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="90" y1="240" x2="720" y2="240" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="120" y="204" width="64" height="36" rx="4" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <rect x="270" y="148" width="64" height="92" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="420" y="90" width="64" height="150" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="570" y="40" width="64" height="200" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <line x1="152" y1="204" x2="302" y2="148" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="302" y1="148" x2="452" y2="90" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <line x1="452" y1="90" x2="602" y2="40" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="152" y="258" text-anchor="middle" class="mono" style="font-size:11px">0%</text>
    <text x="302" y="258" text-anchor="middle" class="mono" style="font-size:11px">30%</text>
    <text x="452" y="258" text-anchor="middle" class="mono" style="font-size:11px">60%</text>
    <text x="602" y="258" text-anchor="middle" class="mono" style="font-size:11px">90%</text>
    <text x="24" y="44" style="fill:var(--muted);font-size:12px">throughput tok/s</text>
    <text x="405" y="282" text-anchor="middle" style="fill:var(--muted);font-size:12px">prefix cache hit rate</text>
    <text x="112" y="84" style="fill:var(--teal);font-size:12px">higher hit rate</text>
    <text x="112" y="102" style="fill:var(--teal);font-size:12px">→ fewer prefill tokens to recompute</text>
  </svg>
  <div class="figcap"><b>Fig 32·2 · higher hit rate, higher throughput</b> — x is the prefix cache hit rate (0%→90%), y is throughput (tokens/s). As hit rate rises, the recomputed prefill tokens fall, so throughput climbs; a higher hit rate means fewer tokens to prefill — exactly the payoff that eviction and caching ultimately buy.</div>
</div>

<p>The line that actually decides "who to clear" is surprisingly short — a strategy's entire difference is condensed into what <span class="mono">get_priority</span> returns:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">mem_cache/evict_policy.py ::LRUStrategy</span><span class="ln">priority = last access time</span></div>
  <pre><span class="kw">class</span> <span class="st">EvictionStrategy</span>(ABC):
    <span class="cm"># Abstract base: every strategy need only answer “whose priority is lowest, clear first”</span>
    <span class="kw">@abstractmethod</span>
    <span class="kw">def</span> get_priority(self, node: TreeNode) -&gt; Union[float, Tuple]:
        <span class="kw">pass</span>

<span class="kw">class</span> <span class="st">LRUStrategy</span>(EvictionStrategy):
    <span class="kw">def</span> get_priority(self, node: TreeNode) -&gt; float:
        <span class="kw">return</span> node.last_access_time   <span class="cm"># oldest-untouched → smallest → heap top → evicted first</span>

<span class="kw">class</span> <span class="st">LFUStrategy</span>(EvictionStrategy):
    <span class="kw">def</span> get_priority(self, node: TreeNode) -&gt; Tuple[int, float]:
        <span class="kw">return</span> (node.hit_count, node.last_access_time)  <span class="cm"># least-hit goes first</span></pre>
</div>

<p>That <span class="mono">get_priority</span> line only answers "<strong>who goes first</strong>"; what drives the whole reclaim round is the <strong>heap loop</strong> inside <span class="mono">RadixCache.evict</span> — it heaps the evictable leaves by priority, pops the lowest one by one, <span class="mono">free</span>s its KV slots, deletes the leaf, and pushes a parent that just became a leaf back onto the heap, until it has reclaimed enough <span class="mono">num_tokens</span> slots:</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/mem_cache/radix_cache.py ::RadixCache.evict</span><span class="ln">pop lowest-priority evictable leaves off a heap, free their KV slots</span></div>
  <pre><span class="kw">def</span> evict(self, params: <span class="st">EvictParams</span>) -&gt; <span class="st">EvictResult</span>:
    <span class="cm"># free `num_tokens` of KV by dropping the LOWEST-priority leaves.</span>
    <span class="cm"># only EVICTABLE leaves are candidates; locked (in-use) nodes are</span>
    <span class="cm"># excluded, so live requests never lose their cache.</span>
    heap = [(self.eviction_strategy.get_priority(n), n)
            <span class="kw">for</span> n <span class="kw">in</span> self.evictable_leaves]
    heapq.heapify(heap)
    num_evicted = 0
    <span class="kw">while</span> num_evicted &lt; params.num_tokens <span class="kw">and</span> heap:
        _, x = heapq.heappop(heap)
        self.token_to_kv_pool_allocator.free(x.value)   <span class="cm"># release KV slots</span>
        num_evicted += len(x.value)
        self._delete_leaf(x)
        <span class="kw">if</span> <span class="kw">not</span> x.parent.children <span class="kw">and</span> x.parent.lock_ref == 0:
            heapq.heappush(heap, (get_priority(x.parent), x.parent))   <span class="cm"># parent is a new leaf → float up</span>
    <span class="kw">return</span> <span class="st">EvictResult</span>(num_tokens_evicted=num_evicted)</pre>
</div>

<p>Feel the mechanism with concrete numbers: a <strong>2000-token shared system prompt</strong> at a 90% hit rate means most requests <strong>skip about 2000 prefill tokens</strong> and enter decode with almost zero recompute; meanwhile a running request keeps that prefix chain locked (<span class="mono">lock_ref&gt;0</span>), so however starved HBM gets and however busy <span class="mono">evict</span> is, it is <strong>never cleared mid-flight</strong> — the heap only ever pops the unlocked, longest-untouched cold leaves.</p>

<div class="card key">
  <div class="tag">🔑 Key takeaways</div>
  <strong>① Eviction is forced and on-demand</strong>: nodes hold KV slots from Lesson 30's pool; only when the allocator can't assemble new space does the scheduler fire <span class="mono">evict</span> to reclaim some.
  <strong>② What to clear is ordered by EvictionStrategy</strong>: default LRU (<span class="mono">get_priority = last_access_time</span>, oldest first), plus LFU / FIFO / MRU; <strong>only leaves are evictable</strong>, reclaim goes leaf-to-root.
  <strong>③ Iron rule: lock_ref&gt;0 is never evicted</strong> — an in-flight forward is reading that KV, reclaiming it corrupts; inc/dec_lock_ref move nodes between the evictable and protected sets (Lesson 29).
  <strong>④ Hit rate is the payoff metric</strong>: fraction of tokens served from cache, raised by high sharing (Lessons 7/29) + cache-aware scheduling (Lesson 20) → less recompute → higher throughput (Lesson 8); HiCache (Lesson 31) raises effective hit rate.
  <strong>⑤ Core trade: evict-then-recompute vs keep-in-HBM</strong> — keeping costs HBM and concurrency, evicting costs a future recompute; the whole memory part is about making this trade well.
</div>
"""}
