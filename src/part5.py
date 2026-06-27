"""Part 5 - The scheduler (the engine's heart). Lessons (L18-L23).

Each lesson is a dict ``{"zh": html, "en": html}`` consumed by registry.CONTENT.
Only inline-styled, shell.CSS-defined classes are used so the structural checker
(check_html.py) stays at 0 errors / 0 warnings.

These lessons cover the Scheduler subprocess: the event loop (L18), the Req /
ScheduleBatch data (L19), the schedule policy (L20), the zero-overhead overlap
scheduler (L21, the signature optimization), chunked prefill (L22), and DP/PP
scheduling across processes (L23).
"""

LESSON_18 = {"zh": r"""
<p class="lead">
欢迎来到 Part 5——整台引擎的<strong>心脏</strong>。前面四个 Part 我们看完了请求<strong>怎么进来</strong>（HTTP、分词、IPC）、
又<strong>怎么出去</strong>（反分词、流式）。现在镜头对准中间那个真正<strong>做决策</strong>的角色：<strong>调度器（Scheduler）</strong>。
它跑在自己<strong>独立的子进程</strong>里，核心是一段叫 <span class="inline">event_loop_normal</span> 的<strong>无限循环</strong>——
这就是引擎的<strong>心跳</strong>。每跳动一次（我们叫一个 <strong>step</strong>），它都把"收请求 → 入队 → 组批 → 前向 → 处理结果"五件事
飞快地做一遍，然后立刻进入下一跳。这一课，我们就趴在这颗心脏上，听清它每一次跳动的节奏。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把调度器想成<strong>机场塔台的雷达扫描（radar sweep）</strong>。雷达每转一圈，管制员都重复同一套动作：先<strong>接收新到达的航班</strong>、
  把它们纳入<strong>盘旋等待区（holding pattern）</strong>；再<strong>决定这一圈让哪些飞机起飞/降落</strong>（跑道容量有限，不能全放）；
  接着<strong>放行</strong>这些飞机、看着它们执行；最后<strong>处理刚发生的结果</strong>——谁落地了就清出跑道、谁飞走了就销号。然后雷达<strong>转下一圈</strong>，
  周而复始。塔台<strong>只有一个</strong>、<strong>不亲自开飞机</strong>，但<strong>每一圈的调度全由它拍板</strong>。调度器的事件循环，就是这道永不停歇的雷达扫描。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>调度器 = 一个跑在独立子进程里、每秒钟跳动成千上万次的无限循环</strong>。每一次跳动（step），
  它依次做五件事——<span class="inline">recv_requests</span>（收）、<span class="inline">process_input_requests</span>（入队）、
  <span class="inline">get_next_batch_to_run</span>（组批）、<span class="inline">run_batch</span>（前向）、
  <span class="inline">process_batch_result</span>（收尾）。它是引擎里<strong>唯一的决策者</strong>：独占 KV 缓存的<strong>账本</strong>，
  决定每一步<strong>这一批由谁组成</strong>。它<strong>只决策、不计算</strong>——真正的 GPU 矩阵乘交给 TpWorker→ModelRunner（第 24 课）。
  循环越短、转得越快，吞吐就越高，这也正是后面<strong>重叠调度器</strong>（第 21 课）要优化的根。
</div>

<h2>心跳的五个动作：一个 step 里发生了什么</h2>
<p>事件循环本质是 <span class="inline">while True</span>。抛开退出与空闲分支，每一轮（一个 step）从头到尾就是<strong>五步流水</strong>。
记住这个顺序，你就拿到了理解整个 Part 5 的<strong>主索引</strong>——后面每一课都是在把其中某一步<strong>放大讲细</strong>：</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>recv_requests()——收</h4><p>从 ZMQ 收件箱里<strong>取空</strong>所有新到的 <span class="mono">TokenizedGenerateReqInput</span>（第 16 课，由 TokenizerManager 发来）。没有就拿到空列表，绝不阻塞。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>process_input_requests()——入队</h4><p>把新请求挂进<strong>等待队列（waiting queue）</strong>；顺手处理控制消息（abort 中止 / flush 清缓存 / 更新权重等）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>get_next_batch_to_run()——组批</h4><p>组建<strong>这一步</strong>的批：预算够就从等待队列里组一个 <strong>prefill 批</strong>，否则把在跑的请求组成 <strong>decode 批</strong>（第 19/20 课）。连续批处理（第 5 课）就在这里<strong>每步重演</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>run_batch()——前向</h4><p>把这一批交给 TpWorker → <span class="mono">ModelRunner.forward</span> 上 GPU 跑一遍（第 24 课），拿到 logits 后<strong>采样</strong>出新 token。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>process_batch_result()——收尾</h4><p>把新 token 追加到各请求；检测<strong>完成</strong>的（命中停止条件）；<strong>释放它们的 KV 槽位</strong>；把输出发给 DetokenizerManager（第 17 课）。然后<strong>回到第 1 步</strong>。</p></div></div>
</div>

<p>这五步<strong>极短</strong>，却每秒钟跑成千上万遍。关键在于它是一条<strong>闭环</strong>：第 5 步收尾完，立刻又回到第 1 步收新请求——
没有"等这批彻底跑完再接客"的概念。正因为每一圈都重新走第 3 步 <span class="inline">get_next_batch_to_run</span>，
<strong>批才会被一遍遍重新组建</strong>，完成的请求当场离场、等待的请求即刻补入——这正是第 5 课<strong>连续批处理</strong>在物理层面发生的地方。</p>

<p>还要厘清两个常被忽略的细节。其一，<strong>第 1 步 <span class="inline">recv_requests</span> 绝不阻塞</strong>：它只是把收件箱里<strong>此刻已到</strong>的请求一次性取空，
收件箱空就拿到一个空列表、循环照常往下走。这一点至关重要——如果"收请求"会卡住等新请求，那已经在跑的几十条请求就会<strong>跟着一起被冻住</strong>，
GPU 立刻饿死。所以事件循环采用的是<strong>轮询（poll）而非等待（block）</strong>的姿态：每圈瞥一眼收件箱，有就收、没有就算了，绝不为新请求耽误老请求。
其二，当 <span class="inline">get_next_batch_to_run</span> 返回空（既没有 prefill 也没有 decode 要做，即服务器此刻空闲）时，
循环不会傻转空圈，而是走 <span class="inline">on_idle</span> 分支做一次<strong>自检与状态回收</strong>——比如校验 KV 账本有没有泄漏、把空出来的缓存归位，
为下一波流量做好准备。这就是代码里那个 <span class="inline">else</span> 分支的意义。</p>

<p>把镜头再拉远一点接回第 2 课的<strong>三进程模型</strong>：TokenizerManager 在一个进程把文字变 token，Scheduler（连同 TpWorker）在中间这个进程做调度与前向，
DetokenizerManager 在第三个进程把 token 拼回文字。本课讲的事件循环，就是<strong>中间这个进程的全部生命</strong>——它启动后就一头扎进 <span class="inline">while True</span>，
直到 <span class="inline">gracefully_exit</span> 被置位才优雅退出。换句话说，<strong>"调度器进程"约等于"这段事件循环"</strong>：进程在，循环就在转；循环停，进程也就该收摊了。
理解这一点，你就明白为什么说它是引擎的"心脏"——心脏一停，整台机器就死了。</p>

<h2>把循环画成一张图：请求在心脏里的流动</h2>
<p>如果把一个 step 里"数据怎么流"画出来，就是下面这条流水线：请求从收件箱进来，经调度组成批，上 GPU 前向，
结果分两路——新 token 发去反分词、完成的请求清退释放显存——然后<strong>循环回到开头</strong>。注意那条<strong>回环箭头</strong>：
它才是"心跳"的灵魂，意味着这条流水线<strong>永不停机</strong>。</p>

<div class="flow">
  <div class="node"><div class="nt">收件箱</div><div class="nd">ZMQ 收新请求<br>recv_requests</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">调度组批</div><div class="nd">get_next_batch<br>_to_run（决策）</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">GPU 前向</div><div class="nd">run_batch<br>forward+采样</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">结果收尾</div><div class="nd">追加 token<br>清退/发反分词</div></div>
  <div class="arrow">↺</div>
  <div class="node hl"><div class="nt">下一跳</div><div class="nd">回到收件箱<br>step + 1</div></div>
</div>

<p>这张图里只有<strong>"调度组批"</strong>那一格是<strong>纯 CPU 的决策</strong>，<strong>"GPU 前向"</strong>那一格才是真正烧算力的计算。
这个区分至关重要：CPU 在组批时，GPU 是<strong>闲着</strong>的；GPU 在前向时，CPU 也<strong>闲着</strong>。
朴素的 <span class="inline">event_loop_normal</span> 让两者<strong>严格串行</strong>——这就是为什么会有 <span class="inline">event_loop_overlap</span>（第 21 课）：
它是<strong>同一条循环</strong>，但用<strong>流水线</strong>把 CPU 的组批/收尾<strong>藏进上一步 GPU 计算的时间里</strong>，让 GPU 永不空等。</p>

<p>顺带说明流水线里那条<strong>分叉</strong>：第 5 步收尾时，结果其实分两路走。一路是<strong>给客户端的产出</strong>——把这一步新采样出的 token
打包成 <span class="mono">BatchTokenIDOutput</span> 发给 DetokenizerManager（第 17 课），由它拼回文字、流式吐给用户；另一路是<strong>给引擎自己的回收</strong>——
对那些命中停止条件（EOS / 停止串 / 达到 max_new_tokens）而<strong>完成</strong>的请求，把它们移出在跑批、<strong>释放各自的 KV 槽位</strong>归还内存池。
正因为有这条"完成即释放"的回收路，第 3 步下一圈才有空间接纳等待队列里的新请求——<strong>收尾与组批，一退一进，咬合成连续批处理的齿轮</strong>。</p>

<h2>每一步去了哪一课：循环就是 Part 5 的目录</h2>
<p>这五步几乎一一对应接下来的每一课。把这张表当作<strong>导航图</strong>：你随时可以从"心跳的第几步"跳到对应的细节课，
也能反过来在读细节课时，记得它在<strong>整个 step</strong> 里处在什么位置。</p>

<table class="t">
  <tr><th>阶段</th><th>它做什么</th><th>细节在哪一课</th></tr>
  <tr><td><strong>recv_requests</strong></td><td>从 ZMQ 取空收件箱，收 TokenizedGenerateReqInput</td><td class="mono">第 16 课 io_struct / IPC</td></tr>
  <tr><td><strong>process_input_requests</strong></td><td>新请求入<strong>等待队列</strong>，处理 abort/flush 等控制消息</td><td class="mono">第 19 课 Req / ScheduleBatch</td></tr>
  <tr><td><strong>get_next_batch_to_run</strong></td><td>组这一步的批：优先 prefill，否则 decode（连续批处理）</td><td class="mono">第 20 课 调度策略 · 第 22 课分块预填充</td></tr>
  <tr><td><strong>run_batch</strong></td><td>交 TpWorker→ModelRunner.forward 上 GPU，采样出 token</td><td class="mono">第 24 课 模型前向</td></tr>
  <tr><td><strong>process_batch_result</strong></td><td>追加 token、检测完成、释放 KV、发去反分词</td><td class="mono">第 17 课 反分词与流式</td></tr>
</table>

<h2>谁决策、谁计算：单线程的唯一决策者</h2>
<p>整台引擎里有很多角色，但<strong>每个 TP rank 只有一个调度器，且它是单线程的</strong>。这不是限制，而是<strong>设计</strong>：
让<strong>唯一一个</strong>角色掌管 KV 缓存账本、独自决定每一步的批，就<strong>不会有两个人抢同一块显存</strong>、也不需要锁。
分清"<strong>谁决策</strong>"和"<strong>谁计算</strong>"，是读懂调度器代码的钥匙：</p>

<div class="cols">
  <div class="col"><h4>调度器：决策但不计算</h4><p>单进程、单线程，是<strong>唯一决策者</strong>。它<strong>独占 KV 缓存的账本</strong>，
  决定每一步<strong>这批由谁组成</strong>、谁该完成离场、谁能补入。它做的全是<strong>CPU 上的轻量记账与判断</strong>，<strong>不碰一次矩阵乘</strong>——
  正因为轻，它才能每秒跳动上万次。</p></div>
  <div class="col"><h4>TpWorker / ModelRunner：计算但不决策</h4><p>拿到调度器组好的批，老老实实在 <strong>GPU</strong> 上跑
  <span class="mono">forward</span>（第 24 课），算出 logits 再采样。它<strong>不决定批是什么</strong>、也不碰调度策略，只负责<strong>把算力榨干</strong>。
  决策与计算解耦，二者才能在重叠调度器里<strong>并行流水</strong>。</p></div>
</div>

<p>为什么"<strong>独占账本</strong>"这件事如此重要？因为显存是引擎里最稀缺的资源，KV 缓存的每一个槽位都要<strong>精确记账</strong>：哪些槽位正被在跑的请求占用、
还剩多少可分配、某条请求完成后该回收哪几行。如果有<strong>两个</strong>角色都能改这本账，就必然出现"两条请求被分到同一块显存"的灾难，
为了避免就得上锁，而锁又会拖慢这条每秒上万次的热循环。SGLang 的选择干脆利落：<strong>让唯一的、单线程的调度器独占这本账</strong>——
它在第 3 步组批时按账本判断"显存够不够再收一条"，在第 5 步收尾时把完成请求的槽位<strong>当场划掉、还回池子</strong>。
没有竞争、没有锁，决策因此既快又安全。这也解释了为什么<strong>每个 TP rank 各有一个独立调度器</strong>：每张卡的显存是各自的账，自然该由各自的决策者来管。</p>

<p>再把第 3 步多说一句，因为它是整个循环里<strong>最有戏</strong>的一步。它不是简单地"把所有在跑的请求拼起来"，而是要做一个关键取舍：
<strong>这一步到底做 prefill 还是 decode？</strong>预算（显存、token 数）允许时，它优先从等待队列里捞出新请求组一个 <strong>prefill 批</strong>，
先把新 prompt 的 KV 算出来；否则就把当前在跑的请求组成一个 <strong>decode 批</strong>，各推进一个 token。"谁先上、上多少、prefill 还是 decode"
这套取舍逻辑叫<strong>调度策略</strong>（第 20 课），而当某条 prompt 长到会<strong>霸占一整步</strong>、把别的请求卡住时，第 22 课的<strong>分块预填充</strong>
会把它切成小块塞进 decode 步的缝隙里——这些都是在第 3 步这一格里发生的事，本课先记住它的<strong>位置</strong>，细节留给后面。</p>

<h2>真实代码：event_loop_normal 的心跳</h2>
<p>下面是调度器真正的循环骨架。注意它有多<strong>朴素</strong>——就是一个 <span class="inline">while True</span>，
里面把我们讲的五步<strong>原样排开</strong>：收、入队、组批、（有批就）前向+收尾、否则空闲自检。读懂这十几行，
你就读懂了整台引擎的心跳。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.event_loop_normal</span><span class="ln">调度器的心跳：每个 step 跑一圈</span></div>
  <pre><span class="kw">def</span> event_loop_normal(self):
    <span class="cm"># A normal scheduler loop.</span>
    <span class="kw">while</span> <span class="kw">True</span>:
        <span class="kw">if</span> self.gracefully_exit:
            <span class="kw">break</span>

        <span class="cm"># ① 收：取空 ZMQ 收件箱里的新请求</span>
        recv_reqs = self.request_receiver.recv_requests()
        <span class="cm"># ② 入队：新请求进等待队列，处理控制消息</span>
        self.process_input_requests(recv_reqs)

        <span class="cm"># ③ 组批：决定这一步跑什么（prefill 还是 decode）</span>
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        <span class="kw">if</span> batch:
            <span class="cm"># ④ 前向：上 GPU 跑这一批，拿 logits 采样</span>
            result = self.run_batch(batch)
            <span class="cm"># ⑤ 收尾：追加 token、清退完成的、发去反分词</span>
            self.process_batch_result(batch, result)
        <span class="kw">else</span>:
            <span class="cm"># 没活干就自检、回收状态</span>
            self.on_idle()

        self.last_batch = batch</pre>
</div>

<p>把这段和开头的雷达类比对齐：<span class="inline">recv_requests</span> 是"接收新到达"，
<span class="inline">process_input_requests</span> 是"纳入盘旋等待区"，<span class="inline">get_next_batch_to_run</span> 是"决定这一圈放哪些飞机"，
<span class="inline">run_batch</span> 是"放行执行"，<span class="inline">process_batch_result</span> 是"处理刚落地/飞走的"。雷达转下一圈，循环回到 <span class="inline">while True</span> 顶端。
这套对应关系不是牵强的比喻，而是<strong>结构上的同构</strong>：塔台之所以高效，正因为它把"接收—决策—放行—善后"压缩成一套可以无限重复的固定动作；
调度器之所以高吞吐，也正因为它把请求生命周期里的一切，都收敛进这五步、每秒重复上万遍。
<span class="inline">event_loop_overlap</span> 是同一段逻辑的<strong>流水线版</strong>：用一个结果队列把上一步的 <span class="inline">process_batch_result</span>
推迟到下一步与 GPU 计算重叠，从而把 CPU 时间<strong>藏起来</strong>（第 21 课细讲）。</p>

<p>最后强调一个直觉：<strong>这个循环的速度，直接给吞吐封顶。</strong>循环每转一圈才前进一个 decode 步、各请求才各吐一个 token。
如果第 3、5 步的 Python 记账太慢，GPU 就会在等 CPU 组批时<strong>空转</strong>，再强的卡也喂不饱。所以 SGLang 把这条循环
<strong>当作最热的热路径</strong>来打磨——这正是第 21 课重叠调度器存在的全部理由，也是你往后读 Part 5 每一课时应当始终带着的那把尺子：
<strong>任何设计，最终都要服务于"让这颗心脏跳得又快又满"。</strong></p>

<p>把这条主线收个尾：第 18 课（本课）给出<strong>骨架</strong>——五步心跳的顺序与边界；第 19 课打开第 2、3 步，讲清楚一条请求在调度器内部到底长什么样
（<span class="mono">Req</span> 与 <span class="mono">ScheduleBatch</span> 这两个核心数据结构）；第 20 课打开第 3 步的<strong>取舍大脑</strong>，讲调度策略怎么排序、怎么决定谁先上；
第 21 课把整条循环<strong>流水线化</strong>，是吞吐优化的标志性一招；第 22 课处理"长 prompt 堵路"的特例；第 24 课则深入第 4 步，看 GPU 上的前向到底怎么算。
换句话说，<strong>后面六课都是在本课这张五步图上做注解</strong>。只要你脑子里随时能默画出"收 → 入队 → 组批 → 前向 → 收尾 → 回环"这条闭环，
无论读到哪一课，都能立刻定位它在<strong>整个 step</strong> 里的位置，不会迷路。这，就是把调度器当作"心脏"来理解的全部价值。</p>

<h2>把心跳画成一只转轮：四步周而复始</h2>
<p>文字读到这里，不妨把这条闭环<strong>画成一只转轮</strong>：四个动作首尾相接，转轮永远向同一个方向旋转，每转满一圈就是一个 step。
注意中心那枚 <span class="inline">while True</span>——它才是让轮子<strong>永不停转</strong>的发动机。</p>

<div class="fig">
  <svg viewBox="0 0 760 340" role="img" aria-label="调度事件循环画成一只转轮：① 收请求 recv_requests → ② 组批 get_next_batch → ③ 前向 run_batch（GPU）→ ④ 出结果 process_result，再回到 ①，周而复始">
    <circle cx="400" cy="180" r="120" style="fill:none;stroke:var(--faint);stroke-width:1.5;stroke-dasharray:5 5"/>
    <path d="M 496 56 C 590 56, 612 96, 610 152" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 610 160 l -5 -10 l 10 0 z" style="fill:var(--muted)"/>
    <path d="M 610 204 C 610 270, 580 310, 502 310" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 494 310 l 10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <path d="M 298 310 C 200 310, 158 270, 160 204" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 160 196 l -5 10 l 10 0 z" style="fill:var(--muted)"/>
    <path d="M 160 154 C 160 96, 190 56, 303 56" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 311 56 l -10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <circle cx="400" cy="180" r="48" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="400" y="176" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--muted)">while True</text>
    <text x="400" y="195" text-anchor="middle" style="font-size:11px;fill:var(--faint)">step + 1</text>
    <rect x="305" y="34" width="190" height="46" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="400" y="53" text-anchor="middle" style="font-size:13px">① 收请求</text>
    <text x="400" y="71" text-anchor="middle" class="mono" style="font-size:11px">recv_requests</text>
    <rect x="520" y="156" width="180" height="48" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="610" y="176" text-anchor="middle" style="font-size:13px">② 组批</text>
    <text x="610" y="194" text-anchor="middle" class="mono" style="font-size:11px">get_next_batch</text>
    <rect x="300" y="288" width="200" height="46" rx="9" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="400" y="307" text-anchor="middle" style="font-size:13px">③ 前向（GPU）</text>
    <text x="400" y="325" text-anchor="middle" class="mono" style="font-size:11px">run_batch</text>
    <rect x="70" y="156" width="180" height="48" rx="9" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="160" y="176" text-anchor="middle" style="font-size:13px">④ 出结果</text>
    <text x="160" y="194" text-anchor="middle" class="mono" style="font-size:11px">process_result</text>
  </svg>
  <div class="figcap"><b>图 1 · 调度事件循环（环形）</b> — 四个动作首尾相接成一只转轮：① <span class="mono">recv_requests</span> 收请求 → ② <span class="mono">get_next_batch</span> 组批 → ③ <span class="mono">run_batch</span> 上 GPU 前向 → ④ <span class="mono">process_result</span> 出结果，再回到 ①；中心的 <span class="mono">while True</span> 让它每个 step 转一圈、永不停转。</div>
</div>

<h2>批是流动的：等待队列 → 运行批 → 完成离场</h2>
<p>转轮只画了<strong>动作</strong>，没画<strong>请求</strong>。再看一张：等待的请求排在左边队列里，被准入的进入<strong>中间的运行批</strong>（每个 step 前向一次），
命中停止条件的从右边离场并释放 KV——而运行批的<strong>成员每一圈都在变</strong>，这正是连续批处理的现场。</p>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="请求的流动：左边等待队列中的请求被准入中间的运行批，运行批每个 step 前向一次且成员不断变化，命中停止条件的请求从右边离场并释放 KV">
    <text x="100" y="40" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:13px">等待队列</text>
    <rect x="40" y="56" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="76" text-anchor="middle" class="mono" style="font-size:11px">req6</text>
    <rect x="40" y="92" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="112" text-anchor="middle" class="mono" style="font-size:11px">req7</text>
    <rect x="40" y="128" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="148" text-anchor="middle" class="mono" style="font-size:11px">req8</text>
    <rect x="40" y="164" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="184" text-anchor="middle" class="mono" style="font-size:11px">req9</text>
    <line x1="164" y1="120" x2="296" y2="120" style="stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 304 120 l -10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <text x="232" y="110" text-anchor="middle" style="font-size:11px;fill:var(--muted)">准入 admit</text>
    <rect x="300" y="70" width="170" height="150" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="385" y="60" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:13px">运行批 running</text>
    <rect x="315" y="84" width="140" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="385" y="105" text-anchor="middle" class="mono" style="font-size:11px">req0</text>
    <rect x="315" y="122" width="140" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="385" y="143" text-anchor="middle" class="mono" style="font-size:11px">req1</text>
    <rect x="315" y="160" width="140" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="385" y="181" text-anchor="middle" class="mono" style="font-size:11px">req5</text>
    <text x="385" y="212" text-anchor="middle" style="font-size:11px;fill:var(--muted)">成员每个 step 都在变</text>
    <path d="M 330 70 C 330 28, 440 28, 440 70" style="fill:none;stroke:var(--amber);stroke-width:1.5"/>
    <path d="M 440 78 l -5 -10 l 10 0 z" style="fill:var(--amber)"/>
    <text x="385" y="22" text-anchor="middle" style="font-size:11px;fill:var(--amber)">每个 step 前向一次</text>
    <line x1="472" y1="140" x2="596" y2="140" style="stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 604 140 l -10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <text x="536" y="130" text-anchor="middle" style="font-size:11px;fill:var(--muted)">命中停止 → 离场</text>
    <rect x="600" y="84" width="130" height="120" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="72" text-anchor="middle" style="font-weight:700;fill:var(--teal);font-size:13px">完成 finished</text>
    <rect x="614" y="98" width="102" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="118" text-anchor="middle" class="mono" style="font-size:11px">req3 ✓</text>
    <rect x="614" y="134" width="102" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="154" text-anchor="middle" class="mono" style="font-size:11px">req4 ✓</text>
    <text x="665" y="190" text-anchor="middle" style="font-size:11px;fill:var(--muted)">释放 KV 槽位</text>
  </svg>
  <div class="figcap"><b>图 2 · 队列 → 运行批 → 完成</b> — 等待请求排在左侧队列，被<strong>准入</strong>后进入中间的运行批（每个 step 前向一次、成员不断变化），命中停止条件的请求从右侧<strong>离场</strong>并释放各自的 KV 槽位；批的组成因此每一圈都在变，这就是连续批处理的物理现场。</div>
</div>

<h2>run_batch：把一批交给 worker 做一次前向</h2>
<p>第 4 步 <span class="inline">run_batch</span> 本身其实<strong>极薄</strong>——它把整批交给 TpWorker 做<strong>一次</strong>前向，拿回 logits 与采样好的 token，
顺手把 <span class="mono">forward_ct</span> 加一。这个计数器正是<strong>引擎到此跳了多少次心跳</strong>的累加表。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.run_batch</span><span class="ln">把一个批交给模型 worker 做一次前向，拿回下一个 token</span></div>
  <pre><span class="kw">def</span> run_batch(self, batch):
    self.forward_ct += 1
    <span class="cm"># hand the batch to the model worker for ONE forward (prefill or decode);</span>
    <span class="cm"># get back a GenerationBatchResult (logits + sampled next-token ids)</span>
    result = self.model_worker.forward_batch_generation(batch)
    <span class="kw">return</span> result   <span class="cm"># a GenerationBatchResult</span></pre>
</div>

<p><strong>一个具体例子（数 step）：</strong>假设服务器刚启动、<span class="mono">forward_ct = 0</span>。第 1 个 step 组出一个含 3 条请求的 <strong>prefill 批</strong>，
<span class="inline">run_batch</span> 跑完后 <span class="mono">forward_ct = 1</span>；接下来若干 step 都在做 <strong>decode 批</strong>（每条各推进 1 个 token），
跑到第 100 个 step 时 <span class="mono">forward_ct = 100</span>——也就是这颗心脏到此刻已经跳了整整 100 下。</p>

<p><strong>再看一个例子（批大小在变）：</strong>第 1 个 step 运行批 = {req0, req1, req5}，<strong>batch size = 3</strong>；
若 req1 在这一步命中 EOS 而完成离场、同时 req6 从等待队列被准入，那么<strong>下一个 step</strong> 的运行批就变成 {req0, req5, req6}——
人数仍是 3，但<strong>成员换了</strong>。这正是图 2 想说的：批的<strong>大小可能不变，但组成每圈都在重排</strong>。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>调度器是引擎的心脏</strong>：跑在<strong>独立子进程</strong>里，核心是无限循环 <span class="mono">event_loop_normal</span>，每秒跳动成千上万次，每次叫一个 <strong>step</strong>。</li>
    <li><strong>一个 step 的五步</strong>（记住顺序）：recv_requests 收 → process_input_requests 入队 → get_next_batch_to_run 组批 → run_batch 前向 → process_batch_result 收尾 → 回到开头。</li>
    <li><strong>批每步重组</strong>：第 3 步每圈都重来，完成的请求当场离场释放 KV、等待的即刻补入——这就是第 5 课连续批处理的物理现场。</li>
    <li><strong>决策 vs 计算</strong>：调度器单线程、唯一决策者，独占 KV 账本、只做 CPU 轻量记账；TpWorker/ModelRunner 才上 GPU 做 forward（第 24 课）。</li>
    <li><strong>循环速度给吞吐封顶</strong>：CPU 组批与 GPU 计算在 normal 版里串行 ⇒ 第 21 课 <span class="mono">event_loop_overlap</span> 用流水线把 CPU 时间藏进 GPU 计算后。前接第 16/17 课，后接第 19/20/22/24 课。</li>
  </ul>
</div>
""",
             "en": r"""
<p class="lead">
Welcome to Part 5 — the engine's <strong>heart</strong>. The previous four parts showed how a request <strong>comes in</strong>
(HTTP, tokenize, IPC) and <strong>goes out</strong> (detokenize, streaming). Now the camera turns to the one role in the middle
that actually <strong>makes decisions</strong>: the <strong>Scheduler</strong>. It runs in its <strong>own subprocess</strong>, and at
its core is an <strong>infinite loop</strong> called <span class="inline">event_loop_normal</span> — the engine's <strong>heartbeat</strong>.
Each beat (we call it one <strong>step</strong>) runs five things — "receive → enqueue → form batch → forward → process result" —
as fast as possible, then immediately starts the next beat. This lesson sits right on that heart and listens to every beat.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the scheduler as an <strong>air-traffic controller's radar sweep</strong>. Every revolution, the controller repeats the
  same routine: first <strong>take in new arrivals</strong> and put them in the <strong>holding pattern</strong>; then <strong>decide
  which planes land/take off this sweep</strong> (the runway is finite — you can't clear them all); then <strong>clear</strong> those
  planes and watch them execute; finally <strong>handle what just happened</strong> — a plane that landed frees the runway, one that
  departed is closed out. Then the radar <strong>sweeps again</strong>, forever. There is <strong>one</strong> tower, it <strong>flies
  no plane itself</strong>, yet <strong>every sweep's scheduling is its call</strong>. That is the scheduler's event loop.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  In one line: <strong>the scheduler is an infinite loop, in its own subprocess, beating thousands of times per second</strong>.
  Each beat (step) does five things in order — <span class="inline">recv_requests</span> (receive),
  <span class="inline">process_input_requests</span> (enqueue), <span class="inline">get_next_batch_to_run</span> (form batch),
  <span class="inline">run_batch</span> (forward), <span class="inline">process_batch_result</span> (finish). It is the engine's
  <strong>single decision-maker</strong>: it <strong>owns the KV-cache accounting</strong> and decides <strong>who is in this step's
  batch</strong>. It <strong>only decides, never computes</strong> — the real GPU matmuls go to TpWorker→ModelRunner (Lesson 24).
  The shorter and faster the loop, the higher the throughput — exactly what the <strong>overlap scheduler</strong> (Lesson 21) later optimizes.
</div>

<h2>The five actions of a heartbeat: what happens in one step</h2>
<p>The event loop is essentially <span class="inline">while True</span>. Setting aside the exit and idle branches, each round (one
step) is a <strong>five-stage pipeline</strong>. Memorize the order and you hold the <strong>master index</strong> to all of Part 5 —
every later lesson just <strong>zooms into one of these stages</strong>:</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>recv_requests() — receive</h4><p><strong>Drain</strong> the ZMQ inbox of all new <span class="mono">TokenizedGenerateReqInput</span> (Lesson 16, sent by the TokenizerManager). If none, you get an empty list — it never blocks.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>process_input_requests() — enqueue</h4><p>Add new requests to the <strong>waiting queue</strong>; also handle control messages (abort / flush cache / update weights, etc.).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>get_next_batch_to_run() — form batch</h4><p>Build <strong>this step's</strong> batch: budget permitting, form a <strong>prefill batch</strong> from waiting reqs, else a <strong>decode batch</strong> of running reqs (Lessons 19/20). Continuous batching (Lesson 5) <strong>re-happens here every step</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>run_batch() — forward</h4><p>Hand the batch to TpWorker → <span class="mono">ModelRunner.forward</span> on the GPU (Lesson 24), get logits, then <strong>sample</strong> the new tokens.</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>process_batch_result() — finish</h4><p>Append new tokens; detect <strong>finished</strong> reqs (stop conditions); <strong>free their KV slots</strong>; send outputs to the DetokenizerManager (Lesson 17). Then <strong>go back to step 1</strong>.</p></div></div>
</div>

<p>These five steps are <strong>very short</strong>, yet run thousands of times per second. The key is that it's a <strong>closed loop</strong>:
right after step 5 finishes, it returns to step 1 to receive new requests — there is no notion of "finish this batch entirely before
serving anyone new." Because every revolution re-runs step 3 <span class="inline">get_next_batch_to_run</span>, <strong>the batch is
re-formed over and over</strong>, finished reqs leave on the spot and waiting ones fill in instantly — this is exactly where Lesson 5's
<strong>continuous batching</strong> physically happens.</p>

<h2>The loop as a diagram: how a request flows through the heart</h2>
<p>If you draw "how data flows" inside one step, you get this pipeline: requests enter from the inbox, get scheduled into a batch,
go forward on the GPU, and the result splits two ways — new tokens go to detokenize, finished reqs are evicted and free their VRAM —
then it <strong>loops back to the top</strong>. Note the <strong>loop-back arrow</strong>: it is the soul of the "heartbeat," meaning this
pipeline <strong>never stops</strong>.</p>

<div class="flow">
  <div class="node"><div class="nt">Inbox</div><div class="nd">ZMQ recv<br>recv_requests</div></div>
  <div class="arrow">→</div>
  <div class="node hl"><div class="nt">Schedule</div><div class="nd">get_next_batch<br>_to_run (decide)</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">GPU forward</div><div class="nd">run_batch<br>forward+sample</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Finish</div><div class="nd">append tokens<br>evict/detok</div></div>
  <div class="arrow">↺</div>
  <div class="node hl"><div class="nt">Next beat</div><div class="nd">back to inbox<br>step + 1</div></div>
</div>

<p>In this diagram only the <strong>"Schedule"</strong> cell is <strong>pure-CPU decision</strong>, and only the <strong>"GPU forward"</strong>
cell actually burns compute. This split is crucial: while the CPU forms the batch, the GPU is <strong>idle</strong>; while the GPU does
forward, the CPU is <strong>idle</strong>. The plain <span class="inline">event_loop_normal</span> runs them <strong>strictly serially</strong> —
which is why <span class="inline">event_loop_overlap</span> (Lesson 21) exists: it is the <strong>same loop</strong>, but pipelined so the
CPU's batching/finishing is <strong>hidden behind the previous step's GPU compute</strong>, keeping the GPU from ever idling.</p>

<h2>Where each step goes: the loop IS Part 5's table of contents</h2>
<p>These five steps map almost one-to-one onto the lessons ahead. Treat this table as a <strong>navigation map</strong>: jump from "which
step of the heartbeat" to its detail lesson, and conversely, while reading a detail lesson remember where it sits in the <strong>whole step</strong>.</p>

<table class="t">
  <tr><th>Stage</th><th>What it does</th><th>Detailed in</th></tr>
  <tr><td><strong>recv_requests</strong></td><td>Drain the ZMQ inbox, receive TokenizedGenerateReqInput</td><td class="mono">Lesson 16 io_struct / IPC</td></tr>
  <tr><td><strong>process_input_requests</strong></td><td>New reqs into the <strong>waiting queue</strong>; handle abort/flush control msgs</td><td class="mono">Lesson 19 Req / ScheduleBatch</td></tr>
  <tr><td><strong>get_next_batch_to_run</strong></td><td>Form this step's batch: prefill first, else decode (continuous batching)</td><td class="mono">Lesson 20 policy · Lesson 22 chunked prefill</td></tr>
  <tr><td><strong>run_batch</strong></td><td>Hand to TpWorker→ModelRunner.forward on GPU, sample tokens</td><td class="mono">Lesson 24 model forward</td></tr>
  <tr><td><strong>process_batch_result</strong></td><td>Append tokens, detect finished, free KV, send to detokenize</td><td class="mono">Lesson 17 detokenize & streaming</td></tr>
</table>

<h2>Who decides, who computes: the single-threaded sole decision-maker</h2>
<p>There are many roles in the engine, but <strong>each TP rank has exactly one scheduler, and it is single-threaded</strong>. That is not
a limitation but a <strong>design</strong>: letting <strong>one</strong> role own the KV-cache ledger and decide every step's batch means
<strong>no two parties fight over the same VRAM</strong> and no locks are needed. Telling "<strong>who decides</strong>" from
"<strong>who computes</strong>" is the key to reading scheduler code:</p>

<div class="cols">
  <div class="col"><h4>Scheduler: decides, doesn't compute</h4><p>One process, single-threaded, the <strong>sole decision-maker</strong>. It
  <strong>owns the KV-cache ledger</strong>, decides <strong>who is in this step's batch</strong>, who finishes and leaves, who fills in.
  It does only <strong>lightweight CPU accounting and judgment</strong>, <strong>touching not a single matmul</strong> — precisely because
  it is light, it can beat tens of thousands of times per second.</p></div>
  <div class="col"><h4>TpWorker / ModelRunner: computes, doesn't decide</h4><p>It takes the batch the scheduler formed and dutifully runs
  <span class="mono">forward</span> on the <strong>GPU</strong> (Lesson 24), producing logits, then samples. It <strong>does not decide what
  the batch is</strong> nor touch scheduling policy — it just <strong>squeezes the compute</strong>. Decoupling decision from compute is
  what lets the two <strong>pipeline in parallel</strong> in the overlap scheduler.</p></div>
</div>

<h2>Real code: the event_loop_normal heartbeat</h2>
<p>Here is the actual loop skeleton. Notice how <strong>plain</strong> it is — just a <span class="inline">while True</span> with our five
steps <strong>laid out verbatim</strong>: receive, enqueue, form batch, (if any) forward + finish, else idle self-check. Read these dozen
lines and you've read the heartbeat of the entire engine.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.event_loop_normal</span><span class="ln">the scheduler heartbeat: one revolution per step</span></div>
  <pre><span class="kw">def</span> event_loop_normal(self):
    <span class="cm"># A normal scheduler loop.</span>
    <span class="kw">while</span> <span class="kw">True</span>:
        <span class="kw">if</span> self.gracefully_exit:
            <span class="kw">break</span>

        <span class="cm"># (1) receive: drain the ZMQ inbox of new requests</span>
        recv_reqs = self.request_receiver.recv_requests()
        <span class="cm"># (2) enqueue: new reqs into waiting queue, handle control msgs</span>
        self.process_input_requests(recv_reqs)

        <span class="cm"># (3) form batch: decide what this step runs (prefill or decode)</span>
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        <span class="kw">if</span> batch:
            <span class="cm"># (4) forward: run this batch on GPU, sample from logits</span>
            result = self.run_batch(batch)
            <span class="cm"># (5) finish: append tokens, evict finished, send to detok</span>
            self.process_batch_result(batch, result)
        <span class="kw">else</span>:
            <span class="cm"># nothing to do: self-check and re-init states</span>
            self.on_idle()

        self.last_batch = batch</pre>
</div>

<p>Line it up with the opening radar analogy: <span class="inline">recv_requests</span> is "take in arrivals,"
<span class="inline">process_input_requests</span> is "put them in the holding pattern," <span class="inline">get_next_batch_to_run</span>
is "decide which planes this sweep," <span class="inline">run_batch</span> is "clear them to execute," and
<span class="inline">process_batch_result</span> is "handle what just landed/departed." The radar sweeps again — back to the top of
<span class="inline">while True</span>. <span class="inline">event_loop_overlap</span> is the <strong>pipelined version</strong> of the same
logic: a result queue defers the previous step's <span class="inline">process_batch_result</span> to overlap with GPU compute, thereby
<strong>hiding</strong> the CPU time (detailed in Lesson 21).</p>

<p>One final intuition: <strong>the loop's speed directly caps throughput.</strong> Each revolution advances exactly one decode step;
every request emits exactly one token. If the Python accounting in steps 3 and 5 is too slow, the GPU <strong>idles</strong> waiting on the
CPU to form the batch, and no card, however strong, stays fed. So SGLang polishes this loop as <strong>the hottest of hot paths</strong> —
the entire reason Lesson 21's overlap scheduler exists, and the yardstick to carry through all of Part 5:
<strong>every design ultimately serves one goal — keep this heart beating fast and full.</strong></p>

<h2>Draw the heartbeat as a wheel: four steps, spinning forever</h2>
<p>Having read the prose, picture the closed loop as a <strong>wheel</strong>: four actions joined head-to-tail, always turning the same
direction, and one full revolution is one step. Note the <span class="inline">while True</span> at the hub — it is the engine that keeps
the wheel <strong>spinning forever</strong>.</p>

<div class="fig">
  <svg viewBox="0 0 760 340" role="img" aria-label="the scheduler event loop drawn as a wheel: ① recv_requests → ② get_next_batch → ③ run_batch (GPU) → ④ process_result, then back to ①, spinning forever">
    <circle cx="400" cy="180" r="120" style="fill:none;stroke:var(--faint);stroke-width:1.5;stroke-dasharray:5 5"/>
    <path d="M 496 56 C 590 56, 612 96, 610 152" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 610 160 l -5 -10 l 10 0 z" style="fill:var(--muted)"/>
    <path d="M 610 204 C 610 270, 580 310, 502 310" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 494 310 l 10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <path d="M 298 310 C 200 310, 158 270, 160 204" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 160 196 l -5 10 l 10 0 z" style="fill:var(--muted)"/>
    <path d="M 160 154 C 160 96, 190 56, 303 56" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 311 56 l -10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <circle cx="400" cy="180" r="48" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="400" y="176" text-anchor="middle" class="mono" style="font-size:12px;fill:var(--muted)">while True</text>
    <text x="400" y="195" text-anchor="middle" style="font-size:11px;fill:var(--faint)">step + 1</text>
    <rect x="305" y="34" width="190" height="46" rx="9" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="400" y="53" text-anchor="middle" style="font-size:13px">① receive</text>
    <text x="400" y="71" text-anchor="middle" class="mono" style="font-size:11px">recv_requests</text>
    <rect x="520" y="156" width="180" height="48" rx="9" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="610" y="176" text-anchor="middle" style="font-size:13px">② form batch</text>
    <text x="610" y="194" text-anchor="middle" class="mono" style="font-size:11px">get_next_batch</text>
    <rect x="300" y="288" width="200" height="46" rx="9" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="400" y="307" text-anchor="middle" style="font-size:13px">③ forward (GPU)</text>
    <text x="400" y="325" text-anchor="middle" class="mono" style="font-size:11px">run_batch</text>
    <rect x="70" y="156" width="180" height="48" rx="9" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="160" y="176" text-anchor="middle" style="font-size:13px">④ emit result</text>
    <text x="160" y="194" text-anchor="middle" class="mono" style="font-size:11px">process_result</text>
  </svg>
  <div class="figcap"><b>Fig 1 · The scheduler event loop (a wheel)</b> — four actions joined head-to-tail into a wheel: ① <span class="mono">recv_requests</span> receive → ② <span class="mono">get_next_batch</span> form batch → ③ <span class="mono">run_batch</span> forward on the GPU → ④ <span class="mono">process_result</span> emit result, then back to ①; the <span class="mono">while True</span> at the hub makes it turn once per step, spinning forever.</div>
</div>

<h2>The batch is fluid: waiting queue → running batch → finished</h2>
<p>The wheel drew the <strong>actions</strong>, not the <strong>requests</strong>. Here is the other view: waiting requests sit in a queue on the
left, admitted ones enter the <strong>running batch</strong> in the center (forwarded once each step), and requests that hit a stop condition
exit on the right and free their KV — while the running batch's <strong>members change every revolution</strong>, which is continuous batching in the flesh.</p>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="request flow: requests in the left waiting queue are admitted into the center running batch, which is forwarded once per step with changing membership, and requests that hit a stop condition exit on the right and free their KV">
    <text x="100" y="40" text-anchor="middle" style="font-weight:700;fill:var(--muted);font-size:13px">waiting queue</text>
    <rect x="40" y="56" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="76" text-anchor="middle" class="mono" style="font-size:11px">req6</text>
    <rect x="40" y="92" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="112" text-anchor="middle" class="mono" style="font-size:11px">req7</text>
    <rect x="40" y="128" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="148" text-anchor="middle" class="mono" style="font-size:11px">req8</text>
    <rect x="40" y="164" width="120" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="100" y="184" text-anchor="middle" class="mono" style="font-size:11px">req9</text>
    <line x1="164" y1="120" x2="296" y2="120" style="stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 304 120 l -10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <text x="232" y="110" text-anchor="middle" style="font-size:11px;fill:var(--muted)">admit</text>
    <rect x="300" y="70" width="170" height="150" rx="10" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="385" y="60" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink);font-size:13px">running batch</text>
    <rect x="315" y="84" width="140" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="385" y="105" text-anchor="middle" class="mono" style="font-size:11px">req0</text>
    <rect x="315" y="122" width="140" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="385" y="143" text-anchor="middle" class="mono" style="font-size:11px">req1</text>
    <rect x="315" y="160" width="140" height="32" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="385" y="181" text-anchor="middle" class="mono" style="font-size:11px">req5</text>
    <text x="385" y="212" text-anchor="middle" style="font-size:11px;fill:var(--muted)">members change each step</text>
    <path d="M 330 70 C 330 28, 440 28, 440 70" style="fill:none;stroke:var(--amber);stroke-width:1.5"/>
    <path d="M 440 78 l -5 -10 l 10 0 z" style="fill:var(--amber)"/>
    <text x="385" y="22" text-anchor="middle" style="font-size:11px;fill:var(--amber)">one forward per step</text>
    <line x1="472" y1="140" x2="596" y2="140" style="stroke:var(--muted);stroke-width:1.5"/>
    <path d="M 604 140 l -10 -5 l 0 10 z" style="fill:var(--muted)"/>
    <text x="536" y="130" text-anchor="middle" style="font-size:11px;fill:var(--muted)">stop hit → exit</text>
    <rect x="600" y="84" width="130" height="120" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="72" text-anchor="middle" style="font-weight:700;fill:var(--teal);font-size:13px">finished</text>
    <rect x="614" y="98" width="102" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="118" text-anchor="middle" class="mono" style="font-size:11px">req3 ✓</text>
    <rect x="614" y="134" width="102" height="30" rx="6" style="fill:var(--panel-2);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="154" text-anchor="middle" class="mono" style="font-size:11px">req4 ✓</text>
    <text x="665" y="190" text-anchor="middle" style="font-size:11px;fill:var(--muted)">free KV slots</text>
  </svg>
  <div class="figcap"><b>Fig 2 · queue → running batch → finished</b> — waiting requests sit in the left queue; once <strong>admitted</strong> they enter the center running batch (forwarded once per step, membership ever-changing); requests that hit a stop condition <strong>exit</strong> on the right and free their KV slots; the batch composition therefore changes every revolution — continuous batching, physically.</div>
</div>

<h2>run_batch: hand one batch to the worker for a single forward</h2>
<p>Step 4, <span class="inline">run_batch</span>, is itself <strong>very thin</strong> — it hands the whole batch to the TpWorker for
<strong>one</strong> forward, gets back logits and the sampled tokens, and bumps <span class="mono">forward_ct</span> by one. That counter is
the running tally of <strong>how many heartbeats the engine has taken so far</strong>.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.run_batch</span><span class="ln">hand a batch to the model worker for one forward, get the next tokens</span></div>
  <pre><span class="kw">def</span> run_batch(self, batch):
    self.forward_ct += 1
    <span class="cm"># hand the batch to the model worker for ONE forward (prefill or decode);</span>
    <span class="cm"># get back a GenerationBatchResult (logits + sampled next-token ids)</span>
    result = self.model_worker.forward_batch_generation(batch)
    <span class="kw">return</span> result   <span class="cm"># a GenerationBatchResult</span></pre>
</div>

<p><strong>A concrete example (counting steps):</strong> say the server just started with <span class="mono">forward_ct = 0</span>. Step 1 forms a
<strong>prefill batch</strong> of 3 requests, and after <span class="inline">run_batch</span> returns, <span class="mono">forward_ct = 1</span>; the
next several steps run <strong>decode batches</strong> (each advancing every request by 1 token), and by step 100 <span class="mono">forward_ct = 100</span>
— the heart has beaten exactly 100 times by now.</p>

<p><strong>Another example (batch size shifts):</strong> step 1's running batch = {req0, req1, req5}, <strong>batch size = 3</strong>; if req1 hits EOS
this step and leaves while req6 is admitted from the waiting queue, then the <strong>next step's</strong> running batch becomes {req0, req5, req6}
— still 3 members, but the <strong>membership changed</strong>. That is exactly Fig 2's point: the batch <strong>size may hold steady, yet its
composition is reshuffled every revolution</strong>.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>The scheduler is the engine's heart</strong>: it runs in its <strong>own subprocess</strong>; its core is the infinite loop <span class="mono">event_loop_normal</span>, beating thousands of times per second, each beat a <strong>step</strong>.</li>
    <li><strong>The five stages of a step</strong> (memorize the order): recv_requests → process_input_requests → get_next_batch_to_run → run_batch → process_batch_result → back to the top.</li>
    <li><strong>The batch is re-formed every step</strong>: stage 3 reruns each revolution — finished reqs leave on the spot and free KV, waiting ones fill in — this is Lesson 5's continuous batching in the flesh.</li>
    <li><strong>Decide vs compute</strong>: the scheduler is single-threaded, the sole decision-maker, owns the KV ledger, does only light CPU accounting; only TpWorker/ModelRunner runs forward on the GPU (Lesson 24).</li>
    <li><strong>Loop speed caps throughput</strong>: CPU batching and GPU compute are serial in the normal version ⇒ Lesson 21's <span class="mono">event_loop_overlap</span> pipelines to hide CPU time behind GPU compute. Builds on Lessons 16/17; leads to Lessons 19/20/22/24.</li>
  </ul>
</div>
"""}

LESSON_19 = {"zh": r"""
<p class="lead">
上一课我们看清了调度器的<strong>心跳</strong>——那段 <span class="inline">while True</span> 的事件循环（第 18 课）。可它每一跳到底在
<strong>操作什么东西</strong>？答案是两个贯穿整个 Part 5 的核心数据结构：<strong>Req</strong> 与 <strong>ScheduleBatch</strong>。
<span class="inline">Req</span> 是<strong>一条请求自己的状态机</strong>，从生到死它都<strong>活着</strong>；<span class="inline">ScheduleBatch</strong>
是<strong>某一个 step 里被打包一起前向的那一撮 Req</strong>，它<strong>转瞬即逝</strong>——每跳一次就重建一次。看懂这一对"持久 vs 临时"
的分工，你才真正握住了调度器手里那两样工具。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把引擎想成一家<strong>航空公司</strong>。每位<strong>乘客（Req）</strong>都有一份<strong>持久的订单与状态</strong>：从哪上、要去哪、托运了什么、
  现在值机了没、登机了没、到没到。这份记录<strong>从订票一直保留到落地</strong>，跨越很多个航班环节都不丢。而每一次起飞的
  <strong>航班舱单（ScheduleBatch）</strong>则是<strong>临时拼出来</strong>的：它要么是一队<strong>正在登机的新乘客</strong>（extend/prefill 批，
  把刚到的人按行李多少安排座位），要么是一舱<strong>正在巡航的老乘客</strong>（decode 批，所有人一起往前推进一步）。乘客是<strong>持久的</strong>，
  舱单是<strong>每班现拼、落地即弃的</strong>；谁到站了就<strong>下飞机</strong>（filter_batch 把完成的请求清出舱单）。同一位乘客，会出现在
  许多张<strong>连续的巡航舱单</strong>上——这正是 Req 与 ScheduleBatch 必须分开的原因。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>Req 记录"一条请求是谁、走到哪了"，ScheduleBatch 记录"这一步要把哪些 Req 怎样一起算"</strong>。
  <span class="inline">Req</span> 装着输入 token、已生成的输出 token、采样参数、它占用的 KV 缓存索引（第 4/30 课）、前缀匹配信息
  （RadixAttention 第 7 课）、以及完成/中止状态；它在 <strong>waiting → running → finished</strong> 三态间迁移。
  <span class="inline">ScheduleBatch</span> 则带着一个<strong>模式（mode）</strong>：要么 <strong>EXTEND（预填充）</strong>——接纳<strong>新请求</strong>、
  一次吃掉它们整段 prompt；要么 <strong>DECODE</strong>——让<strong>正在跑的请求</strong>每人各吐<strong>一个</strong>新 token。批还负责 KV 池的
  <strong>分配（接纳时占）与释放（完成时还）</strong>，并用 <span class="inline">filter_batch()</span> 在飞行途中<strong>剔除已完成的请求</strong>
  （连续批处理的"腾槽"现场，第 5 课）。一句话收束：<strong>Req 管"是谁、走到哪"，ScheduleBatch 管"这一步算谁、怎么算"</strong>，二者一持久一临时，正是调度器一切决策的支点。
</div>

<h2>一、Req：一条请求的状态机，从生到死都活着</h2>
<p>
<span class="inline">Req</span> 是 SGLang 里<strong>最贴身</strong>的对象——它就是"<strong>一条请求的全部身家</strong>"。源码里这个类的注释只有一句：
<em>"The input and output status of a request."</em> 它至少握着这几样东西：<strong>原始输入 token（origin_input_ids）</strong>、
<strong>逐步追加的输出 token（output_ids）</strong>、<strong>采样参数（sampling_params）</strong>、它在
<strong>KV 缓存里占据的位置（req_pool_idx、kv_committed_len/kv_allocated_len 等）</strong>、以及命中前缀树时的
<strong>prefix_indices（RadixAttention，第 7 课）</strong>。最关键的是，它带着自己的<strong>生命周期</strong>：
被接纳前在 <strong>waiting</strong> 队列里排队，被选中预填充后进入 <strong>running</strong>，吐完最后一个 token（命中 EOS、达到 max_new_tokens 或被 abort）
就变成 <strong>finished</strong>，随后它占的 KV 槽位被释放。下面这张图是它的一生。
</p>

<p>
为什么这些状态非得"长在请求身上"，而不是记在别处？因为大模型推理是<strong>自回归</strong>的（第 4 课）：下一个 token 依赖前面所有 token，KV 缓存里逐格累积的正是这条请求<strong>自己的历史</strong>，
换一条请求就完全对不上。所以 <code>output_ids</code> 必须随它一路增长、<code>req_pool_idx</code> 必须牢牢钉住它在 KV 池里的那块地、<code>sampling_params</code> 必须从头到尾跟着它——这些都是<strong>请求私有、不可共享</strong>的状态。
唯一能被多条请求<strong>共享</strong>的，是命中前缀树的那段公共前缀（<code>prefix_indices</code>，RadixAttention，第 7 课），但那也只是"指向同一段只读 KV 的引用"，各自后续生成依旧泾渭分明。
正是这种"私有状态贯穿一生"的本质，逼着 Req 成为一个持久对象——它装的不是某一拍的临时数据，而是一条请求<strong>从入队到完成</strong>的全部因果。
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>waiting（等待）</h4><p>请求被分词、经 IPC 送进调度器（第 16 课），先入<strong>等待队列</strong>。此刻它还没占 KV，只是排队，等调度策略（第 20 课）来挑。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>running · 第一拍 extend（预填充）</h4><p>被选中后进入 running，在一个 <strong>EXTEND 批</strong>里一次性吃掉整段 prompt：<code>prepare_for_extend()</code> 为它分配 KV 槽、铺好变长 token 张量。前缀命中部分（第 7 课）直接复用，不再重算。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>running · 之后许多拍 decode（解码）</h4><p>同一个 Req 接着出现在<strong>一连串 DECODE 批</strong>里，每拍 <code>prepare_for_decode()</code> 给它要一个新 KV 槽、只算<strong>一个</strong>新 token，<code>output_ids</code> 追加一位。它在这里<strong>停留最久</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>finished / freed（完成与释放）</h4><p>命中 EOS、到达 max_new_tokens 或被 abort，状态转 <strong>finished</strong>；<code>filter_batch()</code> 把它当场清出当前批，KV 槽归还内存池（第 30 课），结果发往反分词器（第 17 课）。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="一个 Req 的状态流转：等待队列 → 运行中（先 prefill 吃整段 prompt，再逐步 decode 每拍 +1）→ 完成；其间这条 Req 始终携带 origin_input_ids 与不断增长的 output_ids">
    <text x="24" y="34" style="font-weight:700;fill:var(--muted)">一个 Req 的一生（状态机）</text>
    <rect x="28" y="64" width="150" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="103" y="92" text-anchor="middle" style="fill:var(--blue);font-weight:700">等待队列</text>
    <text x="103" y="112" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">waiting</text>
    <line x1="178" y1="96" x2="244" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M244 96 l-11 -5 v10 z" style="fill:var(--line)"/>
    <rect x="250" y="52" width="266" height="92" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="383" y="46" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">运行中 running</text>
    <rect x="266" y="82" width="106" height="46" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="319" y="102" text-anchor="middle" style="fill:var(--amber);font-weight:700;font-size:12px">prefill</text>
    <text x="319" y="119" text-anchor="middle" style="font-size:10px;fill:var(--amber)">整段 prompt</text>
    <rect x="392" y="82" width="108" height="46" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="446" y="102" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">decode ×N</text>
    <text x="446" y="119" text-anchor="middle" style="font-size:10px;fill:var(--teal)">逐步 +1</text>
    <path d="M384 80 q22 -18 44 0" style="fill:none;stroke:var(--teal);stroke-width:1.5"/>
    <path d="M428 80 l-10 -1 l4 9 z" style="fill:var(--teal)"/>
    <line x1="516" y1="96" x2="582" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M582 96 l-11 -5 v10 z" style="fill:var(--line)"/>
    <rect x="588" y="64" width="150" height="64" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="663" y="90" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">完成 finished</text>
    <text x="663" y="110" text-anchor="middle" style="font-size:10px;fill:var(--accent-ink)">stop / length</text>
    <text x="28" y="186" style="fill:var(--muted);font-size:12px">这条 Req 一路随身携带的内容条：</text>
    <text x="28" y="218" class="mono" style="font-size:11px;fill:var(--muted)">origin_input_ids</text>
    <rect x="150" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="180" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="210" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="240" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="300" y="218" class="mono" style="font-size:11px;fill:var(--teal)">output_ids</text>
    <rect x="378" y="202" width="28" height="24" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="408" y="202" width="28" height="24" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="438" y="202" width="28" height="24" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="468" y="202" width="28" height="24" rx="4" style="fill:none;stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3"/>
    <line x1="502" y1="214" x2="540" y2="214" style="stroke:var(--teal);stroke-width:1.5"/>
    <path d="M540 214 l-10 -5 v10 z" style="fill:var(--teal)"/>
    <text x="548" y="218" class="mono" style="font-size:11px;fill:var(--teal)">+1 每拍</text>
  </svg>
  <div class="figcap"><b>图 19A · 一个 Req 的状态流转</b> — waiting 排队 → running 先 prefill 吃整段 prompt、再逐步 decode 每拍 +1 → finished（命中 stop 或达到 length）；其间这条 Req 始终携带 origin_input_ids 与不断增长的 output_ids。</div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_batch.py ::Req</span><span class="ln">调度器里一个请求的完整运行态</span></div>
<pre><span class="kw">class</span> Req:
    <span class="cm"># one request's full running state inside the scheduler</span>
    <span class="kw">def</span> __init__(self, rid, origin_input_ids, sampling_params, ...):
        self.rid = rid
        self.origin_input_ids = origin_input_ids  <span class="cm"># the prompt token ids</span>
        self.output_ids = []                       <span class="cm"># tokens generated so far</span>
        self.sampling_params = sampling_params
        self.finished_reason = <span class="kw">None</span>                <span class="cm"># set when the request stops</span>
        <span class="cm"># ... prefix-cache node, KV indices, etc.</span></pre>
</div>

<p>
举个具体例子：当生成命中 EOS，<code>finished_reason</code> 取 <code>FINISH_MATCHED_TOKEN</code>（即 <strong>stop</strong>）；当达到 <code>max_new_tokens=256</code> 上限，则取 <code>FINISH_LENGTH</code>（即 <strong>length</strong>）。再如某一拍 <code>ScheduleBatch</code> 里 <code>len(reqs)=8</code>，就是这 8 条请求被打成一批、一起前向。
</p>

<p>
这条生命线里藏着一个常被忽略的事实：<strong>一条请求只 extend 一次，却 decode 很多很多次</strong>。一段几百 token 的 prompt，预填充时被一拍吃完；
而生成阶段每拍只产一个 token，要生成几百 token 就意味着这条请求要在<strong>几百个连续的 decode 批</strong>里反复露面。正因如此，decode 才是引擎的主战场，
也正因如此，Req 必须是一个能<strong>跨越成百上千拍</strong>而不丢失任何状态的持久对象——它得稳稳记着自己已经吐了哪些 token、KV 写到了第几格、采样到温度多少。
把这一点记牢，你就理解了为什么"持久"二字对 Req 如此关键：它是一条贯穿请求一生的<strong>主线</strong>，而批不过是它途经的一个个<strong>驿站</strong>。
</p>

<h2>二、ScheduleBatch：这一步一起前向的那一撮，带着一个模式</h2>
<p>
如果说 Req 是"一个人"，<span class="inline">ScheduleBatch</span> 就是"<strong>这一趟航班的舱单</strong>"——源码注释：
<em>"Store all information of a batch on the scheduler."</em> 它的<strong>第一字段就是 <code>reqs: List[Req]</code></strong>，
即这一步要一起算的那几条请求；其余字段几乎都是为前向<strong>临时铺好</strong>的张量（input_ids、seq_lens、out_cache_loc 等）。
它最重要的属性是 <strong>forward_mode</strong>，只有两种主形态，决定了这一拍"怎么算"：
</p>

<p>
这里要破除一个常见误解：<strong>批不是"队列"，而是"快照"</strong>。等待队列是长期存在、不断有人进出的；而 ScheduleBatch 是调度器在某一拍<strong>临时从运行集合里挑出一撮、拍下的一张合影</strong>，
照完这张相、送 GPU 算完、收尾完成，这张相片就作废了。下一拍要再算，就再拍一张新的。也正因为它是快照，批里那些 GPU 张量（input_ids、seq_lens、out_cache_loc 等）才敢<strong>每步推倒重来</strong>：
反正没人指望它们活过这一拍。把"队列=长期居所、批=瞬时合影"这组对照刻进脑子，后面第 20 课的调度策略、第 21 课的重叠调度器、第 22 课的分块预填充就都有了统一的坐标系。
顺便说一句，正因为批是瞬时合影，调度器才敢在第 21 课用<strong>流水线</strong>把"上一拍批的收尾"与"这一拍批的前向"叠在一起跑——反正两张合影本就互不相干，谁也不依赖谁的张量存活。
</p>

<div class="cols">
  <div class="col">
    <h4>EXTEND / prefill 批</h4>
    <p>由 <code>prepare_for_extend()</code> 构造，<code>forward_mode = EXTEND</code>。它<strong>接纳新请求</strong>，把每条请求<strong>整段 prompt</strong>的
    token 一次性前向。因为各请求 prompt 长度不同，它要铺<strong>变长（ragged）token 张量</strong>，并为每条新请求<strong>分配 KV 槽</strong>。
    源码里 <code>input_ids</code> 取的是 <code>get_fill_ids()[len(prefix_indices):]</code>——<strong>只算前缀没命中的那截</strong>（第 7 课）。
    长 prompt 太大装不下时，会切成 chunked prefill（第 22 课）。</p>
  </div>
  <div class="col">
    <h4>DECODE 批</h4>
    <p>由 <code>prepare_for_decode()</code> 构造，<code>forward_mode = DECODE</code>。它装的是<strong>正在跑的老请求</strong>，每条<strong>只吐一个</strong>新 token，
    因此张量是规整的<strong>每请求 1 token</strong>，只需为每条请求<strong>再要一个 KV 槽</strong>放新 token。它<strong>反复出现</strong>——一条请求一生中绝大多数拍
    都在 decode 批里度过。这正是吞吐的主战场：批越满，GPU 越划算（连续批处理，第 5 课）。</p>
  </div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="ScheduleBatch 把多条不同长度的 Req 打成一批，交给模型一次前向；这一批共享 input_ids、positions、out_cache_loc 等张量">
    <text x="24" y="34" style="font-weight:700;fill:var(--muted)">ScheduleBatch 把多条不同长度的 Req 打成一批，一次前向</text>
    <rect x="24" y="48" width="420" height="150" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="36" y="70" class="mono" style="font-size:12px;fill:var(--accent-ink)">ScheduleBatch · reqs: List[Req]</text>
    <text x="40" y="98" class="mono" style="font-size:11px;fill:var(--blue)">Req A</text>
    <rect x="104" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="136" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="168" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="200" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="40" y="134" class="mono" style="font-size:11px;fill:var(--amber)">Req B</text>
    <rect x="104" y="120" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="136" y="120" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="40" y="170" class="mono" style="font-size:11px;fill:var(--teal)">Req C</text>
    <rect x="104" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="136" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="168" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="200" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="232" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <line x1="444" y1="123" x2="518" y2="123" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M518 123 l-11 -5 v10 z" style="fill:var(--line)"/>
    <rect x="524" y="92" width="208" height="62" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="628" y="118" text-anchor="middle" style="font-weight:700">模型一次前向</text>
    <text x="628" y="138" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">single forward</text>
    <text x="24" y="240" style="fill:var(--muted);font-size:12px">这一批共享的张量 (shared tensors)：</text>
    <rect x="24" y="252" width="150" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="99" y="274" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">input_ids</text>
    <rect x="190" y="252" width="150" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="265" y="274" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--amber)">positions</text>
    <rect x="356" y="252" width="180" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="446" y="274" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--teal)">out_cache_loc</text>
  </svg>
  <div class="figcap"><b>图 19B · ScheduleBatch 打包多个 Req</b> — 几条长度不一的 Req（A/B/C）被装进同一个 ScheduleBatch，由模型一次前向算完；批里那几样张量 input_ids、positions、out_cache_loc 是这一拍全员共享、用完即弃的。</div>
</div>

<p>
为什么非得把"接纳新人"和"推进老人"分成两种批，而不是混在一起？根本原因在于<strong>两种计算的形状天差地别</strong>。预填充要处理的是<strong>整段变长的 prompt</strong>，
每条请求几十到几千 token 不等，张量是参差不齐的 ragged 布局，还要顺手为新来者在 KV 池里<strong>开辟新空间</strong>；而解码处理的是<strong>清一色每人一个</strong>的新 token，
张量整整齐齐，只需在每条请求已有的 KV 末尾<strong>再追加一格</strong>。两者的访存模式、kernel 选择、显存压力都不同，硬塞进同一种批只会两头不讨好。
所以 SGLang 让 <code>forward_mode</code> 当这个开关：组批时（第 18/20 课）先问一句"这一拍是接新人还是推老人"，再调对应的 <code>prepare_*</code> 把张量铺成该有的样子。
顺带一提，当一段 prompt 长到一个 extend 批都装不下时，它会被切成几段、分多拍喂进去——这就是第 22 课要讲的<strong>分块预填充</strong>，本质仍是 EXTEND，只是把"一口吃完"改成"分几口吃"。
</p>

<h2>三、关键字段一览：各自装了什么、扮演什么角色</h2>
<p>把两个结构的核心字段并排看，"持久 vs 临时"的分工一目了然——Req 的字段<strong>跟着请求走</strong>，ScheduleBatch 的字段<strong>跟着这一步走</strong>。</p>

<table class="t">
  <tr><th>结构</th><th>字段</th><th>扮演的角色</th></tr>
  <tr><td class="mono">Req</td><td class="mono">origin_input_ids / output_ids</td><td>输入 token 与逐步追加的输出 token——请求的"内容"，持久</td></tr>
  <tr><td class="mono">Req</td><td class="mono">sampling_params</td><td>温度、top-p、max_new_tokens 等采样策略，决定怎么取下一个 token</td></tr>
  <tr><td class="mono">Req</td><td class="mono">req_pool_idx / kv_*_len</td><td>它在 KV 缓存里占的位置与长度（第 4/30 课），随生成增长</td></tr>
  <tr><td class="mono">Req</td><td class="mono">prefix_indices</td><td>前缀树命中的复用部分（RadixAttention 第 7 课），extend 时跳过不算</td></tr>
  <tr><td class="mono">Req</td><td class="mono">finished() 状态</td><td>EOS / max_new_tokens / abort——触发 filter_batch 清退与 KV 释放</td></tr>
  <tr><td class="mono">ScheduleBatch</td><td class="mono">reqs: List[Req]</td><td>这一步一起前向的请求列表——批的"骨架"，每步重建</td></tr>
  <tr><td class="mono">ScheduleBatch</td><td class="mono">forward_mode</td><td>EXTEND 还是 DECODE——决定这一拍"怎么算"</td></tr>
  <tr><td class="mono">ScheduleBatch</td><td class="mono">input_ids / seq_lens / out_cache_loc</td><td>为前向临时铺好的张量与 KV 输出位置，交给 ModelRunner（第 24 课）</td></tr>
</table>

<p>
对着这张表再品一遍那条分界线：<strong>凡是"内容会随生成不断长大、必须一路记住"的，都在 Req 里</strong>——输出 token 一个个追加、KV 长度一格格增长、采样状态贯穿始终；
<strong>凡是"只为这一拍前向临时拼出来、算完就扔"的，都在 ScheduleBatch 里</strong>——input_ids、seq_lens、out_cache_loc 这些张量每步都重新计算，从不指望跨步存活。
有意思的是，ScheduleBatch 里那个 <code>reqs: List[Req]</code> 字段是连接两者的<strong>纽带</strong>：批不复制请求的内容，它只<strong>持有一串指向 Req 的引用</strong>，
真正的 token、KV 索引始终长在 Req 身上。所以"批被销毁"从不意味着"请求丢失"——下一拍重建批时，<code>get_next_batch_to_run</code>（第 18/20 课）只是把仍然存活的那几条 Req
重新收拢成一份新名单而已。理解了这层"引用而非拷贝"的关系，你就明白了为什么每步重建批的代价其实很轻：重建的只是一层薄薄的调度外壳，沉重的状态从头到尾都安放在 Req 里。
</p>

<h2>四、一个 decode 批长什么样：几条请求，这一拍各 +1</h2>
<p>把镜头拉近到<strong>某一拍 decode</strong>：批里有好几条 Req，它们各自的序列已经长到不同位置，而<strong>这一步每条恰好再添一个新 token</strong>。
下图每一行是一条请求，高亮格子就是<strong>本拍新生成、刚写进 KV 的那一个 token</strong>。</p>

<div class="cellgroup">
  <div class="cg-cap">一个 DECODE 批（本拍：每条请求各 +1 个新 token）</div>
  <div class="cells">
    <span class="lab">Req&nbsp;A</span><span class="cell">天</span><span class="cell">气</span><span class="cell">很</span><span class="cell hl">好</span><span class="sep"></span><span class="q">+1</span>
  </div>
  <div class="cells">
    <span class="lab">Req&nbsp;B</span><span class="cell">def</span><span class="cell">&nbsp;add</span><span class="cell hl">(a</span><span class="sep"></span><span class="q">+1</span>
  </div>
  <div class="cells">
    <span class="lab">Req&nbsp;C</span><span class="cell">The</span><span class="cell">&nbsp;sky</span><span class="cell">&nbsp;is</span><span class="cell">&nbsp;very</span><span class="cell hl">&nbsp;blue</span><span class="sep"></span><span class="q">+1</span>
  </div>
</div>
<p>注意三条请求<strong>已生成长度各不相同</strong>，却能同处一个 decode 批——因为 decode 张量是规整的"每请求 1 token"，长度差异由各自的
<code>seq_lens</code> 记录。下一拍，假如 Req B 吐出了 EOS，<code>filter_batch()</code> 就把它<strong>当场剔除</strong>、释放它的 KV 槽，
留下的 A、C 继续，空出来的容量还能接纳等待队列里的新请求（第 5 课、第 20 课）。</p>

<p>
把这幅画面与"持久 vs 临时"接起来，整条链路就活了：图里这一拍的三行，是<strong>这一个 step</strong> 临时拼出的 ScheduleBatch；而 A、B、C 三条 Req 早在前面某拍就被 extend 进来、
此后已经走过了若干拍 decode，它们各自的"内容条"会一直加长到完成为止。下一拍的 ScheduleBatch 又是<strong>另一份新名单</strong>——也许 B 走了、也许等待队列里的 D 补了进来——但 A、C 这两条 Req
还是原来那两个对象，带着它们一路攒下的 token 与 KV 继续往前。<strong>批在变，请求还在</strong>：这正是连续批处理（第 5 课）能让 GPU 始终满载的微观机理，也是调度器每一拍都要重算一次批的根本理由。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_batch.py ::ScheduleBatch</span><span class="ln">两个数据结构的骨架</span></div>
<pre><span class="kw">class</span> Req(ReqDllmMixin):
    <span class="cm"># The input and output status of a request.（一条请求的输入/输出状态）</span>
    <span class="kw">def</span> __init__(self, rid, origin_input_ids, sampling_params, ...):
        self.rid = rid                      <span class="cm"># 请求 id</span>
        self.origin_input_ids = origin_input_ids   <span class="cm"># 输入 token</span>
        self.output_ids = array(<span class="st">"q"</span>)         <span class="cm"># 逐步追加的输出 token</span>
        self.sampling_params = sampling_params     <span class="cm"># 采样参数</span>
        self.req_pool_idx = <span class="kw">None</span>            <span class="cm"># 占用的 KV 槽位（第 4/30 课）</span>

<span class="kw">class</span> ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    <span class="cm"># Store all information of a batch on the scheduler.（一个批的全部信息）</span>
    reqs: List[Req]                         <span class="cm"># 这一步一起前向的请求</span>

    <span class="kw">def</span> prepare_for_extend(self):           <span class="cm"># 预填充：接纳新请求、吃整段 prompt</span>
        self.forward_mode = ForwardMode.EXTEND
        <span class="cm"># 只取前缀没命中的那截（RadixAttention 第 7 课）</span>
        input_ids = [r.get_fill_ids()[len(r.prefix_indices):] <span class="kw">for</span> r <span class="kw">in</span> self.reqs]

    <span class="kw">def</span> prepare_for_decode(self):           <span class="cm"># 解码：老请求每人各 +1 个 token</span>
        self.forward_mode = ForwardMode.DECODE
        <span class="cm"># 为每条请求再要一个 KV 槽放新 token，张量规整为每请求 1 token</span>

    <span class="kw">def</span> filter_batch(self, ...):            <span class="cm"># 飞行途中剔除已完成的请求（第 5 课）</span>
        keep_indices = [i <span class="kw">for</span> i <span class="kw">in</span> range(len(self.reqs))
                        <span class="kw">if</span> <span class="kw">not</span> self.reqs[i].finished()]
        self.reqs = [self.reqs[i] <span class="kw">for</span> i <span class="kw">in</span> keep_indices]</pre>
</div>

<p>
这段骨架把全课浓缩成几行可读的代码：<code>Req.__init__</code> 里那几个 <code>self.</code> 字段，就是"随请求一生增长"的私有状态；<code>ScheduleBatch</code> 开头一句 <code>reqs: List[Req]</code>，
道破了批<strong>只持有引用、不拷贝内容</strong>的本质。三个方法恰好对应三件大事：<code>prepare_for_extend</code> 把模式置为 EXTEND、并用 <code>get_fill_ids()[len(prefix_indices):]</code> <strong>只取前缀没命中的那截</strong>去前向；
<code>prepare_for_decode</code> 把模式置为 DECODE、为每条请求各要一个新 KV 槽；<code>filter_batch</code> 则用一句列表推导 <strong>留下 <code>not finished()</code> 的请求、当场踢走已完成者</strong>。
真实源码当然还有大量分支（投机解码、DLLM、PP 分块、DP 等），但抓住这四处骨架，你就握住了调度器手里这两样工具的<strong>主干</strong>，足以读懂后面整个 Part 5 的所有变体。
</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>Req = 一条请求的状态机，持久</strong>：装着输入/输出 token、采样参数、占用的 KV 索引（第 4/30 课）、前缀匹配信息（第 7 课）与完成状态；在 <span class="mono">waiting → running → finished</span> 间迁移，从生到死一直活着。</li>
    <li><strong>ScheduleBatch = 这一步一起前向的那一撮 Req，临时</strong>：第一字段就是 <span class="mono">reqs: List[Req]</span>，其余多是为前向铺好的张量；它<strong>每个 step 重建一次</strong>（第 18 课），用完即弃。</li>
    <li><strong>批有两种模式</strong>：<span class="mono">EXTEND/prefill</span>（<code>prepare_for_extend()</code>，接纳新请求、吃整段 prompt、铺变长张量）对 <span class="mono">DECODE</span>（<code>prepare_for_decode()</code>，老请求每人各 +1 token、张量规整）。</li>
    <li><strong>批管 KV 的占与还</strong>：接纳时分配、完成时释放；<span class="mono">filter_batch()</span> 在途中剔除已完成的请求，腾出槽位给等待者——这就是连续批处理（第 5 课）的现场。</li>
    <li><strong>为何分开</strong>：同一条 Req 会出现在许多<strong>连续的 decode 批</strong>里——Req 持久、批临时。带着这对结构去看调度策略（第 20 课）、分块预填充（第 22 课）与模型前向（第 24 课），全 Part 5 一线贯通。</li>
  </ul>
</div>
""", "en": r"""
<p class="lead">
Last lesson we saw the scheduler's <strong>heartbeat</strong>—that <span class="inline">while True</span> event loop (Lesson 18). But what does each
beat actually <strong>operate on</strong>? The answer is two data structures that run through all of Part 5: <strong>Req</strong> and
<strong>ScheduleBatch</strong>. <span class="inline">Req</span> is <strong>one request's own state machine</strong>—it stays <strong>alive</strong> from
birth to death; <span class="inline">ScheduleBatch</strong> is <strong>the cluster of Reqs forwarded together in ONE step</strong>, and it is
<strong>ephemeral</strong>—rebuilt every beat. Grasp this "persistent vs temporary" split and you hold the two tools in the scheduler's hands.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the engine as an <strong>airline</strong>. Each <strong>passenger (Req)</strong> carries a <strong>persistent booking and state</strong>:
  where they board, where they go, what they checked, whether they've checked in, boarded, arrived. That record <strong>persists from booking to
  landing</strong>, surviving many flight legs. But each departing <strong>flight manifest (ScheduleBatch)</strong> is <strong>assembled per
  departure</strong>: it is either a <strong>boarding group of new passengers</strong> (extend/prefill batch—seating new arrivals by how much
  luggage they carry) or a <strong>cruising cabin of existing passengers</strong> (decode batch—everyone advancing one step together). Passengers are
  <strong>persistent</strong>, manifests are <strong>assembled per flight and discarded on landing</strong>; whoever arrives <strong>deplanes</strong>
  (filter_batch removes finished requests from the manifest). The SAME passenger appears on many <strong>successive cruising manifests</strong>—exactly
  why Req and ScheduleBatch must be split.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  In one line: <strong>Req records "who a request is and how far it got"; ScheduleBatch records "which Reqs to compute together this step, and how."</strong>
  <span class="inline">Req</span> holds input tokens, output tokens generated so far, sampling params, the KV-cache indices it owns (Lessons 4/30),
  prefix-match info (RadixAttention, Lesson 7), and finished/abort status; it moves through <strong>waiting → running → finished</strong>.
  <span class="inline">ScheduleBatch</span> carries a <strong>mode</strong>: either <strong>EXTEND (prefill)</strong>—admit <strong>new requests</strong>
  and consume their whole prompt at once—or <strong>DECODE</strong>—let <strong>running requests</strong> each emit <strong>one</strong> new token. The batch
  also handles KV-pool <strong>alloc (on admit) and free (on finish)</strong>, and <span class="inline">filter_batch()</span> <strong>drops finished
  requests</strong> mid-flight (the "slot freeing" of continuous batching, Lesson 5).
</div>

<h2>1. Req: one request's state machine, alive from birth to death</h2>
<p>
<span class="inline">Req</span> is SGLang's most <strong>intimate</strong> object—it is "<strong>a request's entire net worth</strong>." The class's
docstring is one line: <em>"The input and output status of a request."</em> It holds at least: <strong>original input tokens (origin_input_ids)</strong>,
<strong>incrementally appended output tokens (output_ids)</strong>, <strong>sampling params (sampling_params)</strong>, the <strong>position it occupies in
the KV cache (req_pool_idx, kv_committed_len/kv_allocated_len, ...)</strong>, and, on a prefix-tree hit, its <strong>prefix_indices (RadixAttention,
Lesson 7)</strong>. Crucially it carries its own <strong>lifecycle</strong>: before admission it queues in <strong>waiting</strong>; once chosen for
prefill it enters <strong>running</strong>; after emitting its last token (EOS hit, max_new_tokens reached, or abort) it becomes <strong>finished</strong>,
and its KV slots are freed. Here is its whole life.
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>waiting</h4><p>The request is tokenized and sent into the scheduler over IPC (Lesson 16), landing first in the <strong>waiting queue</strong>. It owns no KV yet—just queuing, waiting for the schedule policy (Lesson 20) to pick it.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>running · first beat: extend (prefill)</h4><p>Once selected it enters running and, in an <strong>EXTEND batch</strong>, consumes its whole prompt at once: <code>prepare_for_extend()</code> allocates its KV slots and lays out variable-length token tensors. Prefix-hit parts (Lesson 7) are reused, not recomputed.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>running · then many beats: decode</h4><p>The same Req then appears in <strong>a long run of DECODE batches</strong>; each beat <code>prepare_for_decode()</code> requests one new KV slot and computes just <strong>one</strong> new token, appending one to <code>output_ids</code>. It <strong>dwells here longest</strong>.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>finished / freed</h4><p>On EOS, max_new_tokens, or abort it turns <strong>finished</strong>; <code>filter_batch()</code> drops it from the current batch on the spot, KV slots return to the pool (Lesson 30), and results flow to the detokenizer (Lesson 17).</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="A Req's lifecycle: waiting queue → running (prefill eats the whole prompt, then decode adds +1 per beat) → finished; throughout, the Req carries origin_input_ids and a growing output_ids">
    <text x="24" y="34" style="font-weight:700;fill:var(--muted)">A Req's whole life (state machine)</text>
    <rect x="28" y="64" width="150" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="103" y="92" text-anchor="middle" style="fill:var(--blue);font-weight:700">waiting queue</text>
    <text x="103" y="112" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">waiting</text>
    <line x1="178" y1="96" x2="244" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M244 96 l-11 -5 v10 z" style="fill:var(--line)"/>
    <rect x="250" y="52" width="266" height="92" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="383" y="46" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">running</text>
    <rect x="266" y="82" width="106" height="46" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="319" y="102" text-anchor="middle" style="fill:var(--amber);font-weight:700;font-size:12px">prefill</text>
    <text x="319" y="119" text-anchor="middle" style="font-size:10px;fill:var(--amber)">whole prompt</text>
    <rect x="392" y="82" width="108" height="46" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="446" y="102" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">decode ×N</text>
    <text x="446" y="119" text-anchor="middle" style="font-size:10px;fill:var(--teal)">step +1</text>
    <path d="M384 80 q22 -18 44 0" style="fill:none;stroke:var(--teal);stroke-width:1.5"/>
    <path d="M428 80 l-10 -1 l4 9 z" style="fill:var(--teal)"/>
    <line x1="516" y1="96" x2="582" y2="96" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M582 96 l-11 -5 v10 z" style="fill:var(--line)"/>
    <rect x="588" y="64" width="150" height="64" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="663" y="90" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700">finished</text>
    <text x="663" y="110" text-anchor="middle" style="font-size:10px;fill:var(--accent-ink)">stop / length</text>
    <text x="28" y="186" style="fill:var(--muted);font-size:12px">This Req carries its content strip all the way:</text>
    <text x="28" y="218" class="mono" style="font-size:11px;fill:var(--muted)">origin_input_ids</text>
    <rect x="150" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="180" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="210" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <rect x="240" y="202" width="28" height="24" rx="4" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="300" y="218" class="mono" style="font-size:11px;fill:var(--teal)">output_ids</text>
    <rect x="378" y="202" width="28" height="24" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="408" y="202" width="28" height="24" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="438" y="202" width="28" height="24" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="468" y="202" width="28" height="24" rx="4" style="fill:none;stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3"/>
    <line x1="502" y1="214" x2="540" y2="214" style="stroke:var(--teal);stroke-width:1.5"/>
    <path d="M540 214 l-10 -5 v10 z" style="fill:var(--teal)"/>
    <text x="548" y="218" class="mono" style="font-size:11px;fill:var(--teal)">+1 / beat</text>
  </svg>
  <div class="figcap"><b>Fig 19A · A Req's lifecycle</b> — waiting in queue → running, first prefill eats the whole prompt, then decode adds +1 each beat → finished (hits stop, or reaches length); throughout, the Req keeps carrying origin_input_ids and a growing output_ids.</div>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_batch.py ::Req</span><span class="ln">one request's full running state inside the scheduler</span></div>
<pre><span class="kw">class</span> Req:
    <span class="cm"># one request's full running state inside the scheduler</span>
    <span class="kw">def</span> __init__(self, rid, origin_input_ids, sampling_params, ...):
        self.rid = rid
        self.origin_input_ids = origin_input_ids  <span class="cm"># the prompt token ids</span>
        self.output_ids = []                       <span class="cm"># tokens generated so far</span>
        self.sampling_params = sampling_params
        self.finished_reason = <span class="kw">None</span>                <span class="cm"># set when the request stops</span>
        <span class="cm"># ... prefix-cache node, KV indices, etc.</span></pre>
</div>

<p>
As a concrete example: when generation hits EOS, <code>finished_reason</code> becomes <code>FINISH_MATCHED_TOKEN</code> (i.e. <strong>stop</strong>); when it reaches <code>max_new_tokens=256</code>, it becomes <code>FINISH_LENGTH</code> (i.e. <strong>length</strong>). Likewise, if one beat's <code>ScheduleBatch</code> has <code>len(reqs)=8</code>, then those 8 requests are packed into one batch and forwarded together.
</p>

<h2>2. ScheduleBatch: the cluster forwarded together this step, carrying a mode</h2>
<p>
If Req is "a person," <span class="inline">ScheduleBatch</span> is "<strong>this flight's manifest</strong>"—docstring: <em>"Store all information of a
batch on the scheduler."</em> Its <strong>first field is literally <code>reqs: List[Req]</code></strong>, the requests to compute this step; almost all
other fields are tensors <strong>laid out temporarily</strong> for the forward pass (input_ids, seq_lens, out_cache_loc, ...). Its most important property is
<strong>forward_mode</strong>, with two main forms that decide "how" this beat computes:
</p>

<div class="cols">
  <div class="col">
    <h4>EXTEND / prefill batch</h4>
    <p>Built by <code>prepare_for_extend()</code>, <code>forward_mode = EXTEND</code>. It <strong>admits new requests</strong> and forwards each request's
    <strong>whole prompt</strong> at once. Because prompts differ in length, it lays out <strong>variable-length (ragged) token tensors</strong> and
    <strong>allocates KV slots</strong> for each new request. In source, <code>input_ids</code> takes <code>get_fill_ids()[len(prefix_indices):]</code>—
    <strong>only the part the prefix missed</strong> (Lesson 7). When a long prompt is too big, it is sliced into chunked prefill (Lesson 22).</p>
  </div>
  <div class="col">
    <h4>DECODE batch</h4>
    <p>Built by <code>prepare_for_decode()</code>, <code>forward_mode = DECODE</code>. It holds <strong>already-running requests</strong>, each emitting
    <strong>just one</strong> new token, so tensors are the regular <strong>1 token per request</strong>, needing only <strong>one more KV slot</strong> per
    request for the new token. It <strong>recurs constantly</strong>—a request spends the vast majority of its beats in decode batches. This is throughput's
    main stage: the fuller the batch, the better the GPU pays off (continuous batching, Lesson 5).</p>
  </div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="ScheduleBatch packs several Reqs of different lengths into one batch handed to the model for a single forward; the batch shares tensors input_ids, positions, out_cache_loc">
    <text x="24" y="34" style="font-weight:700;fill:var(--muted)">ScheduleBatch packs several different-length Reqs into one forward</text>
    <rect x="24" y="48" width="420" height="150" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="36" y="70" class="mono" style="font-size:12px;fill:var(--accent-ink)">ScheduleBatch · reqs: List[Req]</text>
    <text x="40" y="98" class="mono" style="font-size:11px;fill:var(--blue)">Req A</text>
    <rect x="104" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="136" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="168" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="200" y="84" width="28" height="22" rx="4" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="40" y="134" class="mono" style="font-size:11px;fill:var(--amber)">Req B</text>
    <rect x="104" y="120" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="136" y="120" width="28" height="22" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="40" y="170" class="mono" style="font-size:11px;fill:var(--teal)">Req C</text>
    <rect x="104" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="136" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="168" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="200" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <rect x="232" y="156" width="28" height="22" rx="4" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <line x1="444" y1="123" x2="518" y2="123" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M518 123 l-11 -5 v10 z" style="fill:var(--line)"/>
    <rect x="524" y="92" width="208" height="62" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="628" y="118" text-anchor="middle" style="font-weight:700">model forward</text>
    <text x="628" y="138" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--muted)">single forward</text>
    <text x="24" y="240" style="fill:var(--muted);font-size:12px">Tensors shared by this batch:</text>
    <rect x="24" y="252" width="150" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="99" y="274" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--blue)">input_ids</text>
    <rect x="190" y="252" width="150" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="265" y="274" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--amber)">positions</text>
    <rect x="356" y="252" width="180" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="446" y="274" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--teal)">out_cache_loc</text>
  </svg>
  <div class="figcap"><b>Fig 19B · ScheduleBatch packs several Reqs</b> — a few different-length Reqs (A/B/C) are placed into one ScheduleBatch and computed by the model in a single forward; the batch's tensors input_ids, positions, out_cache_loc are shared by everyone this beat and discarded after use.</div>
</div>

<h2>3. Key fields at a glance: what each holds, what role it plays</h2>
<p>Lay the two structures' core fields side by side and the "persistent vs temporary" split is obvious—Req's fields <strong>travel with the request</strong>, ScheduleBatch's fields <strong>travel with this step</strong>.</p>

<table class="t">
  <tr><th>Structure</th><th>Field</th><th>Role</th></tr>
  <tr><td class="mono">Req</td><td class="mono">origin_input_ids / output_ids</td><td>Input tokens and incrementally appended output tokens—the request's "content," persistent</td></tr>
  <tr><td class="mono">Req</td><td class="mono">sampling_params</td><td>Temperature, top-p, max_new_tokens, ...—decides how to pick the next token</td></tr>
  <tr><td class="mono">Req</td><td class="mono">req_pool_idx / kv_*_len</td><td>Its position and length in the KV cache (Lessons 4/30), growing as it generates</td></tr>
  <tr><td class="mono">Req</td><td class="mono">prefix_indices</td><td>Prefix-tree reuse (RadixAttention, Lesson 7), skipped during extend</td></tr>
  <tr><td class="mono">Req</td><td class="mono">finished() status</td><td>EOS / max_new_tokens / abort—triggers filter_batch eviction and KV free</td></tr>
  <tr><td class="mono">ScheduleBatch</td><td class="mono">reqs: List[Req]</td><td>The requests forwarded together this step—the batch's "skeleton," rebuilt each step</td></tr>
  <tr><td class="mono">ScheduleBatch</td><td class="mono">forward_mode</td><td>EXTEND or DECODE—decides "how" this beat computes</td></tr>
  <tr><td class="mono">ScheduleBatch</td><td class="mono">input_ids / seq_lens / out_cache_loc</td><td>Tensors laid out for the forward pass and KV output locations, handed to ModelRunner (Lesson 24)</td></tr>
</table>

<h2>4. What a decode batch looks like: a few requests, each +1 this beat</h2>
<p>Zoom into <strong>one decode beat</strong>: the batch has several Reqs, their sequences grown to different lengths, and <strong>each adds exactly one new
token this step</strong>. Each row below is one request; the highlighted cell is <strong>the single token generated this beat and just written to KV</strong>.</p>

<div class="cellgroup">
  <div class="cg-cap">A DECODE batch (this beat: each request +1 new token)</div>
  <div class="cells">
    <span class="lab">Req&nbsp;A</span><span class="cell">The</span><span class="cell">&nbsp;weather</span><span class="cell">&nbsp;is</span><span class="cell hl">&nbsp;nice</span><span class="sep"></span><span class="q">+1</span>
  </div>
  <div class="cells">
    <span class="lab">Req&nbsp;B</span><span class="cell">def</span><span class="cell">&nbsp;add</span><span class="cell hl">(a</span><span class="sep"></span><span class="q">+1</span>
  </div>
  <div class="cells">
    <span class="lab">Req&nbsp;C</span><span class="cell">The</span><span class="cell">&nbsp;sky</span><span class="cell">&nbsp;is</span><span class="cell">&nbsp;very</span><span class="cell hl">&nbsp;blue</span><span class="sep"></span><span class="q">+1</span>
  </div>
</div>
<p>Note the three requests have <strong>different generated lengths</strong> yet sit in one decode batch—because decode tensors are the regular "1 token per
request," with length differences recorded by each <code>seq_lens</code>. Next beat, if Req B emits EOS, <code>filter_batch()</code> <strong>drops it on the
spot</strong> and frees its KV slots, leaving A and C running; the freed capacity can then admit new requests from the waiting queue (Lessons 5 and 20).</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_batch.py ::ScheduleBatch</span><span class="ln">skeletons of the two data structures</span></div>
<pre><span class="kw">class</span> Req(ReqDllmMixin):
    <span class="cm"># The input and output status of a request.</span>
    <span class="kw">def</span> __init__(self, rid, origin_input_ids, sampling_params, ...):
        self.rid = rid                      <span class="cm"># request id</span>
        self.origin_input_ids = origin_input_ids   <span class="cm"># input tokens</span>
        self.output_ids = array(<span class="st">"q"</span>)         <span class="cm"># incrementally appended output</span>
        self.sampling_params = sampling_params     <span class="cm"># sampling params</span>
        self.req_pool_idx = <span class="kw">None</span>            <span class="cm"># KV slot it owns (Lessons 4/30)</span>

<span class="kw">class</span> ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    <span class="cm"># Store all information of a batch on the scheduler.</span>
    reqs: List[Req]                         <span class="cm"># requests forwarded together this step</span>

    <span class="kw">def</span> prepare_for_extend(self):           <span class="cm"># prefill: admit new reqs, eat whole prompt</span>
        self.forward_mode = ForwardMode.EXTEND
        <span class="cm"># only the part the prefix missed (RadixAttention, Lesson 7)</span>
        input_ids = [r.get_fill_ids()[len(r.prefix_indices):] <span class="kw">for</span> r <span class="kw">in</span> self.reqs]

    <span class="kw">def</span> prepare_for_decode(self):           <span class="cm"># decode: running reqs, +1 token each</span>
        self.forward_mode = ForwardMode.DECODE
        <span class="cm"># one more KV slot per req; tensors regular, 1 token per req</span>

    <span class="kw">def</span> filter_batch(self, ...):            <span class="cm"># drop finished reqs mid-flight (Lesson 5)</span>
        keep_indices = [i <span class="kw">for</span> i <span class="kw">in</span> range(len(self.reqs))
                        <span class="kw">if</span> <span class="kw">not</span> self.reqs[i].finished()]
        self.reqs = [self.reqs[i] <span class="kw">for</span> i <span class="kw">in</span> keep_indices]</pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Req = one request's state machine, persistent</strong>: holds input/output tokens, sampling params, owned KV indices (Lessons 4/30), prefix-match info (Lesson 7), and finished status; moves through <span class="mono">waiting → running → finished</span>, alive from birth to death.</li>
    <li><strong>ScheduleBatch = the cluster of Reqs forwarded together this step, temporary</strong>: its first field is <span class="mono">reqs: List[Req]</span>, the rest mostly tensors laid out for the forward pass; it is <strong>rebuilt every step</strong> (Lesson 18) and discarded after use.</li>
    <li><strong>The batch has two modes</strong>: <span class="mono">EXTEND/prefill</span> (<code>prepare_for_extend()</code>, admit new reqs, eat whole prompt, lay out ragged tensors) vs <span class="mono">DECODE</span> (<code>prepare_for_decode()</code>, running reqs +1 token each, regular tensors).</li>
    <li><strong>The batch manages KV alloc and free</strong>: alloc on admit, free on finish; <span class="mono">filter_batch()</span> drops finished reqs mid-flight, freeing slots for waiters—this is continuous batching (Lesson 5) in the flesh.</li>
    <li><strong>Why split them</strong>: the same Req appears in many <strong>successive decode batches</strong>—Req persists, the batch is ephemeral. Carry this pair into the schedule policy (Lesson 20), chunked prefill (Lesson 22), and model forward (Lesson 24), and all of Part 5 lines up.</li>
  </ul>
</div>
"""}

LESSON_20 = {"zh": r"""
<p class="lead">
上一课我们看清了两个数据结构：持久的 <span class="inline">Req</span> 与临时的 <span class="inline">ScheduleBatch</span>。
这一课回答两个最要命的问题：每一拍（step）里，等待队列那么多请求，<strong>该让谁先跑</strong>？又<strong>能塞下几个</strong>？
答案藏在两个角色里——<span class="inline">SchedulePolicy</span> 负责<strong>排序</strong>（决定先后），
<span class="inline">PrefillAdder</span> 负责<strong>限流</strong>（决定多少）。前者让“开头已经在缓存里”的请求插队，几乎白捡一份吞吐；
后者拿着 token 预算和显存预算两把尺子，往这一拍的 prefill 批里一个一个塞，直到塞不下为止。
调度器和缓存（第 7 课）就是这样<strong>协同设计</strong>的——这正是 SGLang 在真实流量下又快又省的秘密。
把这一课记牢一句话：<strong>先排序、再限流，排序为了喂饱缓存，限流为了不撑爆显存</strong>。这两步合起来，
就是事件循环（第 18 课）里 <span class="inline">get_next_batch_to_run</span> 每一拍真正在做的“拍板”动作，也是整台引擎吞吐与延迟平衡的关键所在。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把调度策略想成一家热门餐厅的<strong>聪明领位员（smart host / bouncer）</strong>。门口排着长队，他不傻乎乎地只按到店先后放人。
  他先<strong>给队伍重新排序</strong>：哪几桌客人“桌子已经半摆好了”（食材备齐 = 缓存命中，对应 <strong>LPM 最长前缀匹配</strong>），就让他们插到前面——
  因为招待他们几乎不费厨房功夫；要是大家都差不多，就按<strong>到店先后（FCFS）</strong>公平放行；遇到订了 VIP 套餐的，就按<strong>显式优先级（priority）</strong>提前。
  排好序之后，他还要<strong>数着两个容量</strong>放人：餐厅<strong>座位</strong>够不够（显存 KV 预算）、后厨<strong>这一轮能炒几个菜</strong>（token 预算）——
  两者哪个先满，这一轮就停止放人。下一拍再来一遍。<strong>排序 + 限流</strong>，就是调度策略的全部。
  这位领位员有个铁律：<strong>绝不放进一个坐不下的客人</strong>——宁可让他多等一拍，也不能让满屋子在吃饭的客人因为超额而被挤掉、上菜变慢。
  他也懂取舍：多放新客进门（多接 prefill）能提高翻台率（吞吐），却会让已经在吃的人上菜更慢（在跑 decode 的延迟变高）；
  只顾招待在座的（只 decode）则上菜快、但门口的队伍迟迟不动。每一拍，他都在这架天平上重新放砝码。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>策略决定顺序，预算决定数量</strong>。<span class="inline">get_next_batch_to_run</span>（第 18 课）每拍先调
  <span class="inline">SchedulePolicy.calc_priority</span> 把<strong>等待队列就地排序</strong>，再交给 <span class="inline">PrefillAdder</span>
  在<strong>双预算</strong>下逐个 <span class="inline">add_one_req</span>，能塞则塞（必要时<strong>切块 chunk</strong>），塞满即止。
  排序的核心模式是<strong>缓存感知的 LPM</strong>：优先让 prompt 前缀已落在 RadixAttention 缓存里的请求先跑——
  接纳它们几乎是“免费”的，因为那段注意力早算过了（第 7 课）。这一步直接抬高缓存命中率，而命中率正是 SGLang 真实速度的来源。
  与此同时，调度器每拍都在<strong>“多接新 prefill”与“只推老 decode”之间权衡</strong>：前者增吞吐却抬高在跑请求的延迟，后者保延迟（第 8 课）。
  策略，就是这场天平每一拍的砝码。再把镜头拉远一点：这一课的两个角色，恰好对应一个朴素却深刻的工程直觉——
  <strong>系统的快，一半来自“把对的活排在前面”，一半来自“别一口吃成胖子”</strong>。排序解决前者，预算解决后者。
  前面我们讲过的连续批处理（第 5 课）让批永远满载、RadixAttention（第 7 课）让共享前缀只算一次，
  而调度策略正是把这两件利器<strong>真正用起来</strong>的那只手：没有缓存感知排序，再好的缓存也喂不饱；没有双预算限流，再大的显存也会被一个超长请求撑爆。
</div>

<h2>谁先跑：三种排序策略各自优化什么</h2>
<p>
<span class="inline">SchedulePolicy</span> 把“先后”抽象成 <span class="mono">calc_priority</span> 一次调用：它就地<strong>重排等待队列</strong>。
模式由工作负载决定——很多请求共享前缀时，<strong>缓存感知</strong>最划算；要公平简单，<strong>FCFS</strong>；要尊重业务优先级，<strong>priority</strong>。
注意一个工程细节：当队列特别长（&gt;128）时，昂贵的前缀匹配会临时退回 FCFS，避免排序本身成为瓶颈。
为什么排序如此关键？因为在连续批处理（第 5 课）里，每一拍能接纳的新请求名额有限，谁排在队首谁就先享受这有限的名额；
而把“前缀已缓存”的请求顶到队首，等于让缓存命中的红利<strong>尽早兑现</strong>——这是一笔几乎不花算力就能拿到的吞吐。
反过来，如果排序时无视缓存，热点前缀的请求散落在队尾，命中红利就被白白拖延甚至漏掉。所以排序的本质，
不是“对谁更公平”，而是“怎样让这一拍跑得最划算”。这也解释了为什么 SGLang 把缓存感知作为默认追求，
只有在缓存被禁用或队列过长时才退回到更朴素的 FCFS。
</p>

<table class="t">
  <tr><th>策略</th><th>怎么排</th><th>优化什么</th><th>代价 / 适用</th></tr>
  <tr><td class="mono">LPM（缓存感知）</td><td>最长前缀匹配在前：前缀已在缓存里的优先</td><td><strong>缓存命中率</strong>→近乎免费的吞吐</td><td>需算前缀；队列过长时退回 FCFS</td></tr>
  <tr><td class="mono">FCFS</td><td>先到先服务，按到达顺序</td><td><strong>公平、简单、可预测</strong></td><td>忽略缓存，热点前缀红利拿不到</td></tr>
  <tr><td class="mono">priority</td><td>按每请求显式优先级排序</td><td><strong>尊重业务轻重缓急</strong>（VIP 先行）</td><td>需上层赋值；可能饿死低优先级</td></tr>
</table>

<h2>能塞几个：PrefillAdder 在双预算下组批</h2>
<p>
排好序后，<span class="inline">PrefillAdder</span> 拿着两把尺子往这一拍的 prefill 批里塞请求：
<strong>token 预算</strong>（<span class="mono">rem_total_tokens / rem_input_tokens</span>，别让这一步 prefill 太大，连着第 22 课的分块预填充），
和<strong>显存预算</strong>（KV 池要有足够空闲槽，第 4/30 课——绝不接纳一个放不下的请求）。
这里要厘清两者的“计量单位”：token 预算数的是<strong>这一拍要算多少个 token</strong>，它直接决定前向计算的规模与耗时；
显存预算数的是<strong>这一拍要占多少个 KV 槽</strong>，它决定能不能把这些 token 的中间状态存下来。一个管“算得过来”，一个管“放得下”，
缺一不可。而且这两个预算不是接纳完才检查，而是<strong>每接一个就当场扣减、当场判断</strong>，所以批是被“贪心地”一点点填满的：
能塞就塞，直到下一个再也塞不进——这既保证了批尽量满载（高吞吐），又保证了永不超额（不崩）。
<span class="mono">add_one_req</span> 逐个尝试，能整段塞就整段，塞不下就<strong>切块（chunk）</strong>，任一预算耗尽就返回 <span class="mono">NO_TOKEN / OTHER</span> 停手。
这两把尺子缺一不可：只看 token 预算，可能算力够却没有 KV 槽存放它的缓存，结果接进来也跑不动；只看显存预算，
可能槽位够却一拍吃进一个超长 prompt，把这一步 prefill 撑得过大、拖慢所有人。<span class="mono">_update_prefill_budget</span> 每接一个就同步扣减两个预算，
保证“接纳”和“实际占用”始终对齐，绝不乐观超额。值得强调的是切块的妙处：当一个超长 prompt 装不进剩余 token 预算时，
它不会被整条拒绝，而是只塞进能装下的前半段、剩下的下一拍续上（第 22 课），这样既不堵死循环，也不让长请求饿死。
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>取队首请求</h4><p>从已排序的等待队列拿下一条，算出它要吃多少 token：<span class="mono">cand_extend_input_len + max_new + page_size</span>。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>查 token 预算</h4><p>若 <span class="mono">total_tokens ≥ rem_total_tokens</span> 直接 <span class="mono">NO_TOKEN</span>；这把尺子限制单拍 prefill 的总量。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>查显存 / 槽位</h4><p>KV 池剩余不足（含 SWA 等约束）同样拒绝——绝不超额接纳一个放不下的请求。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>整段或切块塞入</h4><p>预算够则整段加入 <span class="mono">can_run_list</span>；只够一半就 <strong>chunk</strong>，下一拍续上（第 22 课）。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>更新预算，回到 1</h4><p><span class="mono">_update_prefill_budget</span> 扣减两个预算，循环直到塞满或队列空——这一拍的批就定了。</p></div></div>
</div>

<h2>缓存感知 vs 公平：LPM 与 FCFS 的对照</h2>
<p>
为什么缓存感知排序这么重要？因为它<strong>直接抬高 RadixAttention 命中率</strong>——让“开头一样”的请求扎堆先跑，那段共享前缀只算一次。
这就是调度器与缓存<strong>协同设计</strong>的意义：排序的目标不是抽象的公平，而是把缓存这台“省钱机器”喂到最饱。
可以用一个数字直觉感受它的威力：假设线上一半请求共享同一段 2000 token 的系统提示，命中后这 2000 token 的注意力一次都不用重算，
单条请求的 prefill 计算量可能直接砍掉大半。把这些请求排到前面、扎堆命中，整机吞吐的提升是成倍的——这就是为什么 SGLang 在“多请求共享长前缀”的真实场景里表现尤为亮眼。
但缓存感知也有代价：要算前缀、可能让后到的请求等久一点。于是 FCFS 作为对照保留下来，简单、公平、可预测。
怎么选？看负载画像：如果你的流量里成千上万请求都以同一段系统提示、同一套 few-shot 模板、或同一段对话历史开头，
那 LPM 的命中红利会非常可观，是首选；如果请求之间几乎不共享前缀（比如各不相同的一次性查询），LPM 多花的前缀匹配就成了纯开销，
此时朴素的 FCFS 反而更省、更稳；若有明确的业务分级（付费用户优先、实时接口优先），priority 让你把这份优先级直接写进调度。换句话说，<strong>策略不是越聪明越好，而是要和工作负载匹配</strong>——这正是 SGLang 把策略做成可切换、
并按队列长度自动降级的原因。理解了这一点，你就能根据自己的业务，给调度器选对那把“尺子”，让排序与限流都服务于同一个目标：又快又稳地把每一拍的算力用在刀刃上。</p>

<div class="cols">
  <div class="col"><h4>LPM（缓存感知）</h4><p>把<strong>前缀已命中</strong>的请求顶到队首。优点：命中率高、招待近乎免费、真实流量提速巨大（共享系统提示 / few-shot / 对话历史）。代价：要算前缀；队列过长临时退回 FCFS。<strong>优化命中率</strong>。</p></div>
  <div class="col"><h4>FCFS（先到先服务）</h4><p>严格按到达顺序，谁先来谁先跑。优点：<strong>公平、简单、延迟可预测</strong>，不会让后到的无限等待。代价：忽略缓存，热点前缀的红利白白漏掉。<strong>优化公平性</strong>。</p></div>
</div>

<h2>插队现场：被重排的等待队列</h2>
<p>
下面这张图把抽象的“排序”落到实处：原队列按到达顺序是 A、B、C、D；其中 B 和 D 的开头命中了缓存（半摆好的桌子）。
LPM 把它们<strong>顶到前面</strong>——这一拍先跑 B、D，几乎白捡两份吞吐，A、C 顺延到后面。高亮格就是命中前缀、得以插队的请求。
你可能会问：A、C 被一直往后推，会不会饿死？不会——一方面命中是动态的，A、C 跑过一轮后它们的前缀也会进缓存、下一拍可能反过来命中；
另一方面队列过长时策略会自动退回 FCFS，给“等久了”的请求兜底。所以 LPM 不是无脑插队，而是<strong>在公平的底线之上，优先兑现缓存红利</strong>。
这张图也直观解释了为什么“调度器 + 缓存协同设计”不是一句口号：缓存负责把共享前缀存下来，调度器负责把命中它的请求挑出来先跑，
二者一存一取，缺了任何一半，命中率都上不去。
</p>

<div class="cellgroup">
  <div class="cg-cap">等待队列：LPM 让命中缓存的请求（高亮）插队到前面</div>
  <div class="cells">
    <span class="lab">原始</span><span class="cell">A 到达</span><span class="cell">B 到达</span><span class="cell">C 到达</span><span class="cell">D 到达</span><span class="sep"></span><span class="q">FCFS 顺序</span>
  </div>
  <div class="cells">
    <span class="lab">命中</span><span class="cell">A 未命中</span><span class="cell hl">B 命中前缀</span><span class="cell">C 未命中</span><span class="cell hl">D 命中前缀</span><span class="sep"></span><span class="q">查缓存</span>
  </div>
  <div class="cells">
    <span class="lab">重排</span><span class="cell hl">B 先跑</span><span class="cell hl">D 先跑</span><span class="cell">A 顺延</span><span class="cell">C 顺延</span><span class="sep"></span><span class="q">LPM 顺序</span>
  </div>
</div>

<h2>把排序画出来：LPM 重排与三策略对比</h2>
<p>
前面的文字反复说“缓存感知排序”，下面两张图把它彻底画清楚：图 1 演示 LPM 如何把<strong>共享前缀</strong>的请求顶到队首，
图 2 把 FCFS、LPM、priority 三种策略并排放在一起，一眼看清各自在优化什么、各自的代价与适用场景。
</p>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="LPM 按前缀命中把共享前缀 P 的请求 B、D 从队列中间重排到队首，借助基数缓存的命中提示让共享前缀只算一次">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">排序前 · 按到达顺序（FCFS）</text>
    <rect x="20" y="44" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="63" y="66" text-anchor="middle" class="mono" style="font-size:13px">A</text>
    <text x="63" y="82" text-anchor="middle" style="fill:var(--faint);font-size:11px">前缀 X · 未命中</text>
    <rect x="118" y="44" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="161" y="66" text-anchor="middle" class="mono" style="font-size:13px">B</text>
    <text x="161" y="82" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">前缀 P · 命中 ✓</text>
    <rect x="216" y="44" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="259" y="66" text-anchor="middle" class="mono" style="font-size:13px">C</text>
    <text x="259" y="82" text-anchor="middle" style="fill:var(--faint);font-size:11px">前缀 Y · 未命中</text>
    <rect x="314" y="44" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="357" y="66" text-anchor="middle" class="mono" style="font-size:13px">D</text>
    <text x="357" y="82" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">前缀 P · 命中 ✓</text>
    <rect x="448" y="40" width="292" height="54" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="594" y="62" text-anchor="middle" style="fill:var(--blue);font-weight:700;font-size:12px">RadixAttention 基数缓存</text>
    <text x="594" y="81" text-anchor="middle" style="fill:var(--blue);font-size:11px">前缀 P 已在缓存里 → 提示先跑 B、D</text>
    <line x1="200" y1="104" x2="200" y2="150" style="stroke:var(--accent);stroke-width:2"/>
    <path d="M200 152 l-6 -11 l12 0 z" style="fill:var(--accent);stroke:var(--accent)"/>
    <text x="218" y="135" style="fill:var(--muted);font-size:12px">LPM 重排：共享前缀的请求扎堆顶到队首</text>
    <text x="20" y="186" style="font-weight:700;fill:var(--accent-ink)">排序后 · 最长前缀匹配在前（LPM）</text>
    <rect x="20" y="200" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="63" y="222" text-anchor="middle" class="mono" style="font-size:13px">B</text>
    <text x="63" y="238" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">前缀 P · 先跑</text>
    <rect x="118" y="200" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="161" y="222" text-anchor="middle" class="mono" style="font-size:13px">D</text>
    <text x="161" y="238" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">前缀 P · 先跑</text>
    <rect x="216" y="200" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="259" y="222" text-anchor="middle" class="mono" style="font-size:13px">A</text>
    <text x="259" y="238" text-anchor="middle" style="fill:var(--faint);font-size:11px">顺延</text>
    <rect x="314" y="200" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="357" y="222" text-anchor="middle" class="mono" style="font-size:13px">C</text>
    <text x="357" y="238" text-anchor="middle" style="fill:var(--faint);font-size:11px">顺延</text>
    <rect x="430" y="204" width="310" height="40" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="585" y="228" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">B、D 共享前缀只算一次 → 命中率↑、吞吐↑</text>
  </svg>
  <div class="figcap"><b>图 1 · LPM 按前缀命中重排队列</b> — 原队列按到达顺序是 A、B、C、D；B、D 的前缀 P 已在基数缓存里，LPM 据缓存命中提示把它们顶到队首扎堆先跑，共享前缀只算一次，A、C 顺延，几乎白捡两份吞吐。</div>
</div>

<p>
举个具体的数字例子：线上 1000 条请求里，有 600 条都以同一段 2000 token 的系统提示开头。
若按 FCFS 乱序进来，这 600 条会被未命中的请求隔开、零散地撞缓存，热前缀还可能被别的请求挤出缓存而被迫重算；
而 LPM 把这 600 条<strong>扎堆排到一起</strong>先跑，那段 2000 token 的注意力<strong>只算一次</strong>，剩下 599 条直接复用——
单这一项就能把这批请求的 prefill 计算量砍掉一大半，命中率从“看运气”变成“稳稳吃满”。
</p>

<div class="fig">
  <svg viewBox="0 0 760 250" role="img" aria-label="三种调度策略对比：FCFS 公平简单、LPM 缓存感知最大化前缀复用、priority 尊重业务优先级">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">三种排序策略：各自优化什么</text>
    <rect x="20" y="44" width="226" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="133" y="69" text-anchor="middle" class="mono" style="fill:var(--teal);font-size:14px">FCFS</text>
    <rect x="266" y="44" width="226" height="40" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="379" y="69" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">LPM · 缓存感知</text>
    <rect x="512" y="44" width="226" height="40" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="625" y="69" text-anchor="middle" class="mono" style="fill:var(--amber);font-size:14px">priority</text>
    <rect x="20" y="92" width="226" height="50" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="133" y="112" text-anchor="middle" style="fill:var(--muted);font-size:11px">怎么排</text>
    <text x="133" y="130" text-anchor="middle" style="font-size:12px">先到先服务，按到达</text>
    <rect x="266" y="92" width="226" height="50" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="379" y="112" text-anchor="middle" style="fill:var(--muted);font-size:11px">怎么排</text>
    <text x="379" y="130" text-anchor="middle" style="font-size:12px">最长前缀匹配在前</text>
    <rect x="512" y="92" width="226" height="50" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="625" y="112" text-anchor="middle" style="fill:var(--muted);font-size:11px">怎么排</text>
    <text x="625" y="130" text-anchor="middle" style="font-size:12px">按显式优先级</text>
    <rect x="20" y="150" width="226" height="48" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="133" y="170" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">公平 · 简单 · 可预测</text>
    <text x="133" y="188" text-anchor="middle" style="fill:var(--muted);font-size:11px">忽略缓存红利</text>
    <rect x="266" y="150" width="226" height="48" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="379" y="170" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">最大化前缀复用 · 命中率</text>
    <text x="379" y="188" text-anchor="middle" style="fill:var(--muted);font-size:11px">需算前缀；长队列退回 FCFS</text>
    <rect x="512" y="150" width="226" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="625" y="170" text-anchor="middle" style="fill:var(--amber);font-weight:700;font-size:12px">尊重业务轻重缓急</text>
    <text x="625" y="188" text-anchor="middle" style="fill:var(--muted);font-size:11px">可能饿死低优先级</text>
    <text x="20" y="226" style="fill:var(--faint);font-size:12px">适用：FCFS 多用于前缀少的一次性查询；LPM 多用于共享系统提示 / few-shot；priority 用于付费或实时接口优先。</text>
  </svg>
  <div class="figcap"><b>图 2 · 不同策略对比</b> — FCFS 求公平、简单可预测但拿不到缓存红利；LPM 缓存感知、最大化前缀复用以抬高命中率（代价是算前缀、长队列退回 FCFS）；priority 尊重业务优先级但可能饿死低优先级请求。按工作负载选尺子。</div>
</div>

<p>
再看一个更小的例子体会“扎堆”的意义：队列里 <span class="mono">[聊天A, 文档X, 聊天B, 文档Y, 聊天C]</span> 交错排列，
其中三条“聊天”共享同一段对话系统提示。LPM 会把它们重排成 <span class="mono">[聊天A, 聊天B, 聊天C, 文档X, 文档Y]</span>——
三条聊天连着跑，系统提示这段前缀在缓存里始终<strong>热着</strong>、不会被文档请求挤掉，命中一路保持；
若不重排，每跑一条文档就可能把聊天的前缀挤出缓存，下一条聊天又得重算，命中率被白白拉低。这就是“扎堆”二字最朴素的威力。
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::CacheAwarePolicy</span><span class="ln">缓存感知的调度策略：用基数缓存给队列排序</span></div>
<pre><span class="kw">class</span> <span class="fn">CacheAwarePolicy</span>(Enum):
    <span class="cm"># cache-aware policies: use the radix cache to decide who runs next</span>
    LPM = <span class="st">"lpm"</span>            <span class="cm"># longest-prefix-match: prefer requests sharing a cached prefix</span>
    DFS_WEIGHT = <span class="st">"dfs-weight"</span>
    <span class="cm"># (vs CacheAgnosticPolicy: FCFS, etc. when the cache can't help)</span></pre>
</div>

<h2>读源码：calc_priority 与预算判断</h2>
<p>
策略与预算的真身都在 <span class="mono">schedule_policy.py</span>。<span class="mono">calc_priority</span> 先定活跃策略，再按分支排序；
<span class="mono">PrefillAdder.add_one_req</span> 则用一连串预算判断决定接纳还是拒绝。下面截取最能说明问题的几行。
读这段源码时，请抓住三个关键点：一是 <span class="mono">_determine_active_policy</span> 会在队列过长时把活跃策略悄悄换成 FCFS；
二是 FCFS 分支直接按到达顺序（或叠加 priority）返回，几乎零成本；三是缓存感知分支先做 <span class="mono">_compute_prefix_matches</span> 算出每条请求的命中长度，
再据此做最长前缀排序。预算这边，<span class="mono">total_tokens</span> 把“输入 + 预留输出 + 页对齐”一并算进去，
任何一个超过剩余预算就立刻 <span class="mono">NO_TOKEN</span> 退出——这正是“绝不接纳放不下的请求”在代码里的落点。
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy.calc_priority</span><span class="ln">排序 + 限流</span></div>
<pre><span class="kw">def</span> calc_priority(self, waiting_queue, running_batch=<span class="kw">None</span>):
    policy = self._determine_active_policy(waiting_queue)   <span class="cm"># 队列过长(&gt;128)临时退回 FCFS</span>

    <span class="kw">if</span> self.policy == CacheAgnosticPolicy.FCFS:          <span class="cm"># 公平：按到达顺序</span>
        <span class="kw">if</span> self.enable_priority_scheduling:
            SchedulePolicy._sort_by_priority_and_fcfs(waiting_queue, self.priority_sign)
        <span class="kw">return</span>

    <span class="kw">if</span> <span class="kw">isinstance</span>(policy, CacheAwarePolicy):           <span class="cm"># 缓存感知：抬高命中率</span>
        temporary_deprioritized = self._compute_prefix_matches(waiting_queue, policy)
        <span class="kw">if</span> policy == CacheAwarePolicy.LPM:               <span class="cm"># 最长前缀匹配在前</span>
            SchedulePolicy._sort_by_longest_prefix(waiting_queue, temporary_deprioritized)

<span class="cm"># —— PrefillAdder.add_one_req：双预算判断 ——</span>
total_tokens = cand_extend_input_len + max_new + self.page_size
<span class="kw">if</span> total_tokens &gt;= self.rem_total_tokens:            <span class="cm"># token/显存预算不足 → 拒绝</span>
    <span class="kw">return</span> AddReqResult.NO_TOKEN                       <span class="cm"># 绝不接纳放不下的请求(第 4/30 课)</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>策略排序 + 预算限流是两件事</strong>：<span class="mono">SchedulePolicy.calc_priority</span> 决定<strong>谁先跑</strong>（就地重排等待队列），<span class="mono">PrefillAdder</span> 决定<strong>塞几个</strong>（双预算下逐个 add_one_req）。</li>
    <li><strong>三种模式各有所长</strong>：<span class="mono">LPM</span>（缓存感知）优化<strong>命中率</strong>、让前缀已缓存的请求插队（第 7 课）；<span class="mono">FCFS</span> 优化<strong>公平</strong>、简单可预测；<span class="mono">priority</span> 尊重业务优先级。负载多共享前缀时选 LPM。</li>
    <li><strong>PrefillAdder 的两把尺子</strong>：<strong>token 预算</strong>（单拍 prefill 不过大，连第 22 课分块预填充）+ <strong>显存预算</strong>（KV 池有空闲槽，第 4/30 课）；任一耗尽即停，必要时<strong>切块</strong>。</li>
    <li><strong>prefill 与 decode 的接纳张力</strong>：每拍要在“多接新 prefill 增吞吐、抬延迟”与“只推老 decode 保延迟”之间权衡（第 18/8 课），策略就是这场天平的砝码。</li>
    <li><strong>调度器与缓存协同设计</strong>：缓存感知排序直接抬高 RadixAttention 命中率，而命中率正是 SGLang 真实流量提速的根——排序的目标不是抽象公平，而是把缓存喂饱。一存一取、协同配合，缺任何一半命中率都上不去。</li>
    <li><strong>记住一句话收尾</strong>：先排序、再限流；排序为了喂饱缓存、兑现命中红利，限流为了既满载又不撑爆显存。这一拍由谁组成、有多大，就此拍板，然后交给前向（第 24 课）去算。</li>
  </ul>
</div>
""",
             "en": r"""
<p class="lead">
Last lesson we nailed two data structures: the persistent <span class="inline">Req</span> and the ephemeral <span class="inline">ScheduleBatch</span>.
This lesson answers the two most critical questions of every step: with so many requests waiting, <strong>who runs next</strong>, and <strong>how many fit</strong>?
The answers live in two roles—<span class="inline">SchedulePolicy</span> does the <strong>ordering</strong> (who first),
<span class="inline">PrefillAdder</span> does the <strong>throttling</strong> (how many). The former lets requests whose prefix is already in cache jump the line, harvesting nearly-free throughput;
the latter holds two yardsticks—a token budget and a memory budget—and stuffs requests into this step's prefill batch one by one until nothing more fits.
The scheduler and the cache (Lesson 7) are <strong>co-designed</strong> this way—the secret behind SGLang being fast and cheap under real traffic.
</p>

<div class="card analogy">
  <div class="tag">🔌 Real-world analogy</div>
  Think of the schedule policy as a <strong>smart venue host / bouncer</strong> at a hot restaurant. A long line waits outside, and he does not naively admit people purely by arrival.
  First he <strong>reorders the line</strong>: which parties are "already half-prepped" (ingredients ready = cache hit, i.e. <strong>LPM, longest-prefix-match</strong>) get bumped to the front—
  serving them costs the kitchen almost nothing; if everyone's similar, he admits by <strong>arrival (FCFS)</strong>, fair and simple; and VIP reservations go early by <strong>explicit priority</strong>.
  Once ordered, he admits people <strong>counting two capacities</strong>: are there enough <strong>seats</strong> (the memory KV budget), and how many dishes can the <strong>kitchen</strong> cook this round (the token budget)—
  whichever fills first stops the round. Next beat, repeat. <strong>Order + throttle</strong> is the whole of the schedule policy.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>policy decides order, budget decides count</strong>. <span class="inline">get_next_batch_to_run</span> (Lesson 18) each beat first calls
  <span class="inline">SchedulePolicy.calc_priority</span> to <strong>reorder the waiting queue in place</strong>, then hands off to <span class="inline">PrefillAdder</span>
  to <span class="inline">add_one_req</span> one by one under a <strong>double budget</strong>, admitting where it fits (<strong>chunking</strong> if needed) and stopping when full.
  The core ordering mode is <strong>cache-aware LPM</strong>: prefer requests whose prompt prefix already sits in the RadixAttention cache—
  admitting them is nearly "free" because that attention was already computed (Lesson 7). This step directly raises the cache hit rate, and hit rate is where SGLang's real speed comes from.
  Meanwhile the scheduler each beat <strong>balances "admit new prefills" vs "just decode"</strong>: the former grows throughput but raises latency for running decodes, the latter keeps latency low (Lesson 8).
  The policy is the weight on this scale, beat after beat.
</div>

<h2>Who runs first: what each of the three policies optimizes</h2>
<p>
<span class="inline">SchedulePolicy</span> abstracts "order" into one call to <span class="mono">calc_priority</span>: it <strong>reorders the waiting queue in place</strong>.
The mode is chosen by workload—when many requests share prefixes, <strong>cache-aware</strong> pays off most; for fairness and simplicity, <strong>FCFS</strong>; to honor business priority, <strong>priority</strong>.
Note one engineering detail: when the queue is very long (&gt;128), the expensive prefix matching temporarily falls back to FCFS, so sorting itself never becomes the bottleneck.
</p>

<table class="t">
  <tr><th>Policy</th><th>How it orders</th><th>What it optimizes</th><th>Cost / fit</th></tr>
  <tr><td class="mono">LPM (cache-aware)</td><td>Longest-prefix-match first: cached-prefix reqs jump ahead</td><td><strong>Cache hit rate</strong> → near-free throughput</td><td>Must compute prefixes; falls back to FCFS on long queues</td></tr>
  <tr><td class="mono">FCFS</td><td>First-come-first-served, by arrival</td><td><strong>Fair, simple, predictable</strong></td><td>Ignores cache; misses hot-prefix dividends</td></tr>
  <tr><td class="mono">priority</td><td>Sort by each request's explicit priority</td><td><strong>Honors business urgency</strong> (VIP first)</td><td>Needs upper-layer values; can starve low priority</td></tr>
</table>

<h2>How many fit: PrefillAdder builds a batch under a double budget</h2>
<p>
Once ordered, <span class="inline">PrefillAdder</span> holds two yardsticks and stuffs requests into this beat's prefill batch:
a <strong>token budget</strong> (<span class="mono">rem_total_tokens / rem_input_tokens</span>, keep this prefill step from getting too big—tied to chunked prefill, Lesson 22),
and a <strong>memory budget</strong> (enough free KV-pool slots, Lessons 4/30—never admit a request you can't fit).
<span class="mono">add_one_req</span> tries each in turn: add it whole if it fits, otherwise <strong>chunk</strong> it; once either budget is exhausted it returns <span class="mono">NO_TOKEN / OTHER</span> and stops.
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Take the head request</h4><p>Pull the next req off the sorted waiting queue and compute its token need: <span class="mono">cand_extend_input_len + max_new + page_size</span>.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Check the token budget</h4><p>If <span class="mono">total_tokens ≥ rem_total_tokens</span>, return <span class="mono">NO_TOKEN</span>; this yardstick caps the total prefill per beat.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Check memory / slots</h4><p>Insufficient KV-pool room (incl. SWA constraints) also rejects—never over-admit a request that can't fit.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Add whole or chunk</h4><p>If the budget allows, add the whole req to <span class="mono">can_run_list</span>; if only half fits, <strong>chunk</strong> it and continue next beat (Lesson 22).</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>Update budget, back to 1</h4><p><span class="mono">_update_prefill_budget</span> deducts both budgets; loop until full or the queue empties—this beat's batch is then fixed.</p></div></div>
</div>

<h2>Cache-aware vs fair: LPM against FCFS</h2>
<p>
Why does cache-aware ordering matter so much? Because it <strong>directly raises the RadixAttention hit rate</strong>—by clustering "same opening" requests to run first, the shared prefix is computed once.
That is the meaning of <strong>co-designing</strong> the scheduler with the cache: the goal of ordering is not abstract fairness but feeding the cache—the money-saving machine—as full as possible.
But cache-awareness has a cost: computing prefixes, and possibly making later arrivals wait a bit longer. So FCFS is kept as the counterpoint—simple, fair, predictable.
</p>

<div class="cols">
  <div class="col"><h4>LPM (cache-aware)</h4><p>Bumps <strong>prefix-hit</strong> requests to the front. Pros: high hit rate, near-free admission, huge real-traffic speedups (shared system prompts / few-shot / chat history). Cost: must compute prefixes; falls back to FCFS on long queues. <strong>Optimizes hit rate</strong>.</p></div>
  <div class="col"><h4>FCFS (first-come-first-served)</h4><p>Strictly by arrival, first in first run. Pros: <strong>fair, simple, predictable latency</strong>, no infinite wait for late arrivals. Cost: ignores the cache, leaking the hot-prefix dividend. <strong>Optimizes fairness</strong>.</p></div>
</div>

<h2>The line-jump in action: a reordered waiting queue</h2>
<p>
This diagram grounds the abstract "ordering": the original queue by arrival is A, B, C, D; among them B and D have cache-hitting openings (half-set tables).
LPM <strong>bumps them to the front</strong>—this beat runs B and D first, harvesting two nearly-free units of throughput, and A, C slide back. The highlighted cells are the prefix-hit requests that jump ahead.
</p>

<div class="cellgroup">
  <div class="cg-cap">Waiting queue: LPM bumps cache-hitting requests (highlighted) to the front</div>
  <div class="cells">
    <span class="lab">Raw</span><span class="cell">A arrives</span><span class="cell">B arrives</span><span class="cell">C arrives</span><span class="cell">D arrives</span><span class="sep"></span><span class="q">FCFS order</span>
  </div>
  <div class="cells">
    <span class="lab">Hit?</span><span class="cell">A miss</span><span class="cell hl">B prefix hit</span><span class="cell">C miss</span><span class="cell hl">D prefix hit</span><span class="sep"></span><span class="q">probe cache</span>
  </div>
  <div class="cells">
    <span class="lab">Reordered</span><span class="cell hl">B first</span><span class="cell hl">D first</span><span class="cell">A after</span><span class="cell">C after</span><span class="sep"></span><span class="q">LPM order</span>
  </div>
</div>

<h2>Draw the ordering: LPM reorder and three policies compared</h2>
<p>
The text kept saying "cache-aware ordering"; the two figures below make it concrete: Fig 1 shows how LPM bumps requests that <strong>share a prefix</strong> to the front,
and Fig 2 lines up FCFS, LPM, and priority side by side so you can see at a glance what each optimizes, its cost, and where it fits.
</p>

<div class="fig">
  <svg viewBox="0 0 760 300" role="img" aria-label="LPM reorders the queue by prefix hit, bumping requests B and D that share prefix P from the middle to the front using the radix cache hit hint so the shared prefix is computed once">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">Before · by arrival (FCFS)</text>
    <rect x="20" y="44" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="63" y="66" text-anchor="middle" class="mono" style="font-size:13px">A</text>
    <text x="63" y="82" text-anchor="middle" style="fill:var(--faint);font-size:11px">prefix X · miss</text>
    <rect x="118" y="44" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="161" y="66" text-anchor="middle" class="mono" style="font-size:13px">B</text>
    <text x="161" y="82" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">prefix P · hit ✓</text>
    <rect x="216" y="44" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="259" y="66" text-anchor="middle" class="mono" style="font-size:13px">C</text>
    <text x="259" y="82" text-anchor="middle" style="fill:var(--faint);font-size:11px">prefix Y · miss</text>
    <rect x="314" y="44" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="357" y="66" text-anchor="middle" class="mono" style="font-size:13px">D</text>
    <text x="357" y="82" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">prefix P · hit ✓</text>
    <rect x="448" y="40" width="292" height="54" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="594" y="62" text-anchor="middle" style="fill:var(--blue);font-weight:700;font-size:12px">RadixAttention radix cache</text>
    <text x="594" y="81" text-anchor="middle" style="fill:var(--blue);font-size:11px">prefix P already cached → hint: run B, D first</text>
    <line x1="200" y1="104" x2="200" y2="150" style="stroke:var(--accent);stroke-width:2"/>
    <path d="M200 152 l-6 -11 l12 0 z" style="fill:var(--accent);stroke:var(--accent)"/>
    <text x="218" y="135" style="fill:var(--muted);font-size:12px">LPM reorder: cluster shared-prefix reqs to the front</text>
    <text x="20" y="186" style="font-weight:700;fill:var(--accent-ink)">After · longest-prefix-match first (LPM)</text>
    <rect x="20" y="200" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="63" y="222" text-anchor="middle" class="mono" style="font-size:13px">B</text>
    <text x="63" y="238" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">prefix P · first</text>
    <rect x="118" y="200" width="86" height="46" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="161" y="222" text-anchor="middle" class="mono" style="font-size:13px">D</text>
    <text x="161" y="238" text-anchor="middle" style="fill:var(--accent-ink);font-size:11px">prefix P · first</text>
    <rect x="216" y="200" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="259" y="222" text-anchor="middle" class="mono" style="font-size:13px">A</text>
    <text x="259" y="238" text-anchor="middle" style="fill:var(--faint);font-size:11px">slides back</text>
    <rect x="314" y="200" width="86" height="46" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="357" y="222" text-anchor="middle" class="mono" style="font-size:13px">C</text>
    <text x="357" y="238" text-anchor="middle" style="fill:var(--faint);font-size:11px">slides back</text>
    <rect x="430" y="204" width="310" height="40" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="585" y="228" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">B, D share prefix, computed once → hit rate↑, tput↑</text>
  </svg>
  <div class="figcap"><b>Fig 1 · LPM reorders the queue by prefix hit</b> — the original queue by arrival is A, B, C, D; B and D's prefix P already sits in the radix cache, so LPM follows the cache hit hint and bumps them to the front to run together, the shared prefix is computed once, A and C slide back, harvesting two nearly-free units of throughput.</div>
</div>

<p>
A concrete number example: of 1000 live requests, 600 begin with the same 2000-token system prompt.
Admitted out of order under FCFS, those 600 are separated by misses and hit the cache piecemeal—the hot prefix may even be evicted by other requests and recomputed;
under LPM they are <strong>clustered together</strong> and run first, so that 2000-token attention is <strong>computed once</strong> and the other 599 reuse it directly—
this one effect alone cuts a large share of this batch's prefill compute, turning the hit rate from "luck of the draw" into "steadily maxed out".
</p>

<div class="fig">
  <svg viewBox="0 0 760 250" role="img" aria-label="Three scheduling policies compared: FCFS is fair and simple, LPM is cache-aware and maximizes prefix reuse, priority honors business urgency">
    <text x="20" y="30" style="font-weight:700;fill:var(--muted)">Three ordering policies: what each optimizes</text>
    <rect x="20" y="44" width="226" height="40" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="133" y="69" text-anchor="middle" class="mono" style="fill:var(--teal);font-size:14px">FCFS</text>
    <rect x="266" y="44" width="226" height="40" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="379" y="69" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:14px">LPM · cache-aware</text>
    <rect x="512" y="44" width="226" height="40" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="625" y="69" text-anchor="middle" class="mono" style="fill:var(--amber);font-size:14px">priority</text>
    <rect x="20" y="92" width="226" height="50" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="133" y="112" text-anchor="middle" style="fill:var(--muted);font-size:11px">how it orders</text>
    <text x="133" y="130" text-anchor="middle" style="font-size:12px">first-come, by arrival</text>
    <rect x="266" y="92" width="226" height="50" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="379" y="112" text-anchor="middle" style="fill:var(--muted);font-size:11px">how it orders</text>
    <text x="379" y="130" text-anchor="middle" style="font-size:12px">longest-prefix-match first</text>
    <rect x="512" y="92" width="226" height="50" rx="8" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="625" y="112" text-anchor="middle" style="fill:var(--muted);font-size:11px">how it orders</text>
    <text x="625" y="130" text-anchor="middle" style="font-size:12px">by explicit priority</text>
    <rect x="20" y="150" width="226" height="48" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="133" y="170" text-anchor="middle" style="fill:var(--teal);font-weight:700;font-size:12px">fair · simple · predictable</text>
    <text x="133" y="188" text-anchor="middle" style="fill:var(--muted);font-size:11px">ignores cache dividend</text>
    <rect x="266" y="150" width="226" height="48" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="379" y="170" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">max prefix reuse · hit rate</text>
    <text x="379" y="188" text-anchor="middle" style="fill:var(--muted);font-size:11px">must compute prefixes; long queue → FCFS</text>
    <rect x="512" y="150" width="226" height="48" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="625" y="170" text-anchor="middle" style="fill:var(--amber);font-weight:700;font-size:12px">honors business urgency</text>
    <text x="625" y="188" text-anchor="middle" style="fill:var(--muted);font-size:11px">can starve low priority</text>
    <text x="20" y="226" style="fill:var(--faint);font-size:12px">Fit: FCFS for prefix-poor one-off queries; LPM for shared system prompts / few-shot; priority for paid or real-time endpoints.</text>
  </svg>
  <div class="figcap"><b>Fig 2 · policies compared</b> — FCFS seeks fairness, simple and predictable but misses the cache dividend; LPM is cache-aware and maximizes prefix reuse to raise the hit rate (at the cost of computing prefixes and falling back to FCFS on long queues); priority honors business urgency but can starve low-priority requests. Pick the yardstick by workload.</div>
</div>

<p>
A smaller example to feel why "clustering" matters: the queue <span class="mono">[chatA, docX, chatB, docY, chatC]</span> is interleaved,
and the three "chat" requests share one conversation system prompt. LPM reorders it to <span class="mono">[chatA, chatB, chatC, docX, docY]</span>—
the three chats run back-to-back so that system-prompt prefix stays <strong>hot</strong> in the cache and is never evicted by the doc requests, keeping the hit alive throughout;
without reordering, every doc request run in between can evict the chat prefix, forcing the next chat to recompute it and quietly dragging the hit rate down. That is the plainest power of "clustering".
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::CacheAwarePolicy</span><span class="ln">cache-aware scheduling policies: order the queue using the radix cache</span></div>
<pre><span class="kw">class</span> <span class="fn">CacheAwarePolicy</span>(Enum):
    <span class="cm"># cache-aware policies: use the radix cache to decide who runs next</span>
    LPM = <span class="st">"lpm"</span>            <span class="cm"># longest-prefix-match: prefer requests sharing a cached prefix</span>
    DFS_WEIGHT = <span class="st">"dfs-weight"</span>
    <span class="cm"># (vs CacheAgnosticPolicy: FCFS, etc. when the cache can't help)</span></pre>
</div>

<h2>Read the source: calc_priority and the budget check</h2>
<p>
Both the policy and the budget live in <span class="mono">schedule_policy.py</span>. <span class="mono">calc_priority</span> first determines the active policy, then sorts by branch;
<span class="mono">PrefillAdder.add_one_req</span> decides admit-or-reject through a chain of budget checks. Below are the most telling lines.
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::SchedulePolicy.calc_priority</span><span class="ln">order + throttle</span></div>
<pre><span class="kw">def</span> calc_priority(self, waiting_queue, running_batch=<span class="kw">None</span>):
    policy = self._determine_active_policy(waiting_queue)   <span class="cm"># long queue(&gt;128) falls back to FCFS</span>

    <span class="kw">if</span> self.policy == CacheAgnosticPolicy.FCFS:          <span class="cm"># fair: by arrival order</span>
        <span class="kw">if</span> self.enable_priority_scheduling:
            SchedulePolicy._sort_by_priority_and_fcfs(waiting_queue, self.priority_sign)
        <span class="kw">return</span>

    <span class="kw">if</span> <span class="kw">isinstance</span>(policy, CacheAwarePolicy):           <span class="cm"># cache-aware: raise hit rate</span>
        temporary_deprioritized = self._compute_prefix_matches(waiting_queue, policy)
        <span class="kw">if</span> policy == CacheAwarePolicy.LPM:               <span class="cm"># longest-prefix-match first</span>
            SchedulePolicy._sort_by_longest_prefix(waiting_queue, temporary_deprioritized)

<span class="cm"># —— PrefillAdder.add_one_req: the double-budget check ——</span>
total_tokens = cand_extend_input_len + max_new + self.page_size
<span class="kw">if</span> total_tokens &gt;= self.rem_total_tokens:            <span class="cm"># token/memory budget short -&gt; reject</span>
    <span class="kw">return</span> AddReqResult.NO_TOKEN                       <span class="cm"># never admit one that won't fit (L4/30)</span></pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>Policy ordering and budget throttling are two things</strong>: <span class="mono">SchedulePolicy.calc_priority</span> decides <strong>who runs first</strong> (reorders the waiting queue in place), <span class="mono">PrefillAdder</span> decides <strong>how many fit</strong> (add_one_req under a double budget).</li>
    <li><strong>Three modes, each with a strength</strong>: <span class="mono">LPM</span> (cache-aware) optimizes <strong>hit rate</strong> by bumping prefix-cached reqs ahead (Lesson 7); <span class="mono">FCFS</span> optimizes <strong>fairness</strong>, simple and predictable; <span class="mono">priority</span> honors business priority. Pick LPM when the workload shares prefixes.</li>
    <li><strong>PrefillAdder's two yardsticks</strong>: a <strong>token budget</strong> (keep a beat's prefill from getting too big, tied to chunked prefill in Lesson 22) + a <strong>memory budget</strong> (free KV-pool slots, Lessons 4/30); stop when either is exhausted, <strong>chunk</strong> if needed.</li>
    <li><strong>The prefill-vs-decode admission tension</strong>: each beat balances "admit new prefills to grow throughput but raise latency" vs "just decode to keep latency low" (Lessons 18/8); the policy is the weight on this scale.</li>
    <li><strong>Scheduler and cache are co-designed</strong>: cache-aware ordering directly raises the RadixAttention hit rate, and that hit rate is the root of SGLang's real-traffic speedup—ordering's goal is not abstract fairness but feeding the cache full.</li>
  </ul>
</div>
"""}

LESSON_21 = {"zh": r"""
<p class="lead">
这是 Part 5 的<strong>招牌课</strong>，也是第 1 课、第 3 课都点过名的那项优化：<strong>零开销重叠调度器（zero-overhead overlap scheduler）</strong>。
回忆第 18 课，调度器的每一拍（step）都掺着两种活：一种是 <strong>CPU 上的调度</strong>——组批、采样、记账、给下一步备料；
另一种是 <strong>GPU 上的前向（forward）</strong>。在朴素的 <span class="inline">event_loop_normal</span> 里，这两件事是<strong>严格串行</strong>的：
GPU 算完 → CPU 才动手组下一批 → GPU 再算。问题是，<strong>CPU 组批的那段时间，GPU 整个在干等</strong>。
对解码（decode）这种单拍极快的步骤，CPU 的调度开销可能占到一拍的不小比例——于是相当一部分昂贵的 GPU 算力被白白晾着。
这一课讲的，就是 SGLang 怎么用一招<strong>流水线（pipeline）</strong>把这段空等彻底抹掉：让 GPU <strong>永不停转</strong>。
把这一课记牢一句话：<strong>当 GPU 在算这一步时，CPU 已经在为下一步铺路、并顺手收尾上一步</strong>——三件事咬合成一条永不停歇的传送带，
这正是第 1 课提到的"零开销批调度器"、也是 SGLang 在 v0.4 里拿来当招牌讲的那项硬优化。理解了它，你就理解了为什么 SGLang 在高并发 decode 下能把 GPU 榨到几乎满载。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把一拍调度想成一场<strong>接力赛（relay race）/ 两段式流水线</strong>。朴素循环像只有一名选手在跑：他冲完这一棒，得先<strong>停下来</strong>走回起跑线、系好鞋带、调整呼吸，
  然后才冲下一棒——这中间赛道（GPU）空着没人跑。重叠调度器换成<strong>两人接力</strong>：当前选手（GPU）正在全速冲刺这一棒时，
  下一名选手（CPU）<strong>已经站在交接区把鞋带系好、做好起跑预备</strong>，交接棒时<strong>没有一丝停顿</strong>。
  换个厨房的比方也一样：当前这道菜还在灶上（GPU）翻炒，备餐厨师（CPU）就已经把<strong>下一单</strong>的食材切配装盘、摆好待命——
  <strong>灶台从不空烧</strong>。代价只有一个：你永远比"实时"<strong>慢半拍</strong>，因为收尾的总是上一棒的结果。但只要交接处不停顿，整支队伍的总成绩就由跑得最快的那条腿决定，而不再被换人的间隙拖累。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>把这一步的 CPU 调度，藏进上一步的 GPU 计算里</strong>。<span class="inline">event_loop_overlap</span> 不再"算完→收尾→再组下一批"地串行，
  而是<strong>先发射</strong>本步（第 N 步）的 GPU 前向，<strong>不等它算完</strong>，立刻让 CPU 去组第 N+1 步的批、并<strong>顺手收尾上一步（第 N−1 步）</strong>的结果——
  上一步的结果存在一个 <span class="mono">result_queue</span>（结果队列 / future）里，<strong>推迟一拍</strong>才处理。这样每一拍的 CPU 调度开销，
  都正好落在 GPU 那一拍的计算"影子"里被<strong>完全遮住</strong>，对外表现为<strong>近乎零的调度开销</strong>、GPU 利用率拉满。
  代价是<strong>多一拍延迟</strong>（你总是慢一步）外加更精细的状态管理——下一批要在当前结果还没出来时就先搭好，SGLang 用 <span class="mono">overlap_utils.py</span> 里的 FutureMap 那套
  future/event 机制替你扛住了依赖。这正是 v0.4 主打的"零开销批调度器"；它与连续批处理（第 5 课）是绝配：<strong>批永远满载，而把它喂满的调度还不要钱</strong>。
</div>

<h2>问题：串行循环里，GPU 在 CPU 组批时干等</h2>
<p>
先把账算清。一拍里 GPU 真正烧算力的只有 <span class="mono">run_batch</span>（→ ModelRunner.forward，第 24 课）；
其余的 <span class="mono">get_next_batch_to_run</span>（组批）、<span class="mono">sample</span>（采样）、<span class="mono">process_batch_result</span>（追加 token、判完成、释放 KV、记账、给下一步备输入）
全是<strong>CPU 上的轻量活</strong>。朴素的 <span class="inline">event_loop_normal</span> 把它们<strong>排成一条直线</strong>：CPU 组好批 → GPU 前向 → CPU 收尾 → CPU 再组下一批 → GPU 再前向……
问题出在<strong>衔接处</strong>：每当 CPU 在组批、采样、记账时，<strong>GPU 没有任务可做，只能空转等待</strong>；反过来 GPU 前向时，CPU 也闲着。
两个昂贵的资源<strong>轮流忙、轮流闲</strong>，谁也没吃饱。这种"你忙我闲、我忙你闲"的错峰，就是串行循环最大的浪费源头。对 prefill 这种 GPU 重活，CPU 那点开销占比还小；可一旦进入 decode——
每拍只给每条请求生成一个 token，GPU 前向本身就极快——CPU 那段固定的调度开销立刻变成一拍里<strong>相当显眼的一块</strong>。
循环转得越快、GPU 越闲不下来，这块固定开销就越扎眼：它直接给整机吞吐<strong>封了顶</strong>。
再把这笔账具体化：假设一拍 decode 的 GPU 前向只要 8 毫秒，而 CPU 组批、采样、记账加起来要 4 毫秒，那么串行下一拍就是 12 毫秒，
其中整整三分之一的时间 GPU 在睡觉——意味着你买的那张昂贵显卡，有三分之一的钱花在了等 CPU 上。请求并发越高、批越大，CPU 那段记账反而越重，
这个比例还可能更难看。这正是问题的要害：<strong>不是 GPU 不够快，而是它被一段本可以并行的 CPU 工作活活拖住、轮流空转</strong>。
只要能让这段 CPU 工作"躲"到 GPU 计算的背后去，理论上就能把一拍从 12 毫秒压回 8 毫秒——吞吐凭空多出一半，而且一行模型代码都不用改。
</p>

<div class="cols">
  <div class="col"><h4>朴素串行：event_loop_normal（第 18 课）</h4><p>一拍内 <strong>CPU 组批 → GPU 前向 → CPU 收尾</strong> 顺序排队，互不重叠。GPU 在 CPU 组批/采样/记账时<strong>整段空等</strong>。decode 步 GPU 极快，CPU 开销占比大，<strong>算力被晾着</strong>，循环速度给吞吐封顶。</p></div>
  <div class="col"><h4>重叠流水线：event_loop_overlap（本课）</h4><p>先<strong>发射</strong>本步 GPU 前向（不等它算完），CPU 立刻去组下一批并<strong>收尾上一步</strong>的结果。CPU 调度被<strong>藏进 GPU 计算的影子里</strong>，GPU <strong>不再有空隙</strong>——调度开销近乎归零。代价：慢一拍。</p></div>
</div>

<h2>修复：把下一步的调度，塞进这一步的 GPU 影子里</h2>
<p>
重叠循环的核心动作只有三步，但顺序极讲究。第一，<strong>发射</strong>本步（N）的前向：调用 <span class="mono">run_batch</span> 把批交给 GPU，
但<strong>立即返回、不阻塞等结果</strong>，把这个"已发射、待收尾"的批连同它的结果句柄一起 <span class="mono">append</span> 进 <span class="mono">result_queue</span>。
第二，趁 GPU 正埋头算 N，<strong>CPU 回头去组第 N+1 步的批</strong>（下一拍的 <span class="mono">get_next_batch_to_run</span> 其实是这一拍就开始铺路的）。
第三，<strong>收尾上一步（N−1）</strong>：从 <span class="mono">result_queue</span> 里 <span class="mono">popleft</span> 出上一拍那个批和它<em>现在已经算好</em>的结果，做 <span class="mono">process_batch_result</span>——
追加 token、判完成、释放 KV。注意这个错位：<strong>当我们处理 N−1 的结果时，GPU 正忙着算 N</strong>。于是 CPU 的每一段活，都精确地落在 GPU 计算的"影子"里。
这就是"慢一拍"的由来，也是"零开销"的由来——你处理的永远是<strong>上一步</strong>的结果，但每一步的 CPU 时间都<strong>不再让 GPU 等</strong>。
这里有个容易被忽略的精妙之处：发射前向用的是<strong>异步</strong>语义——<span class="mono">run_batch</span> 把核函数排进 GPU 队列后<strong>不等它跑完就返回</strong>，
所以 CPU 才能立刻腾出手去组下一批。真正的"等待"被推迟到下一拍 <span class="mono">pop_and_process</span> 去队列里取结果时才隐式发生，而那时 GPU 早已算完，几乎不用真等。
换句话说，<strong>CPU 和 GPU 像两条独立的传送带各自满速运转</strong>，只在每拍交接处对一次账：本拍发射的批进队列、上拍的批出队列收尾。
正因为这种"发射—入队—下一拍再收尾"的错位，整条循环才不存在任何一方干等另一方的缝隙。
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>发射本步前向（N），不等它</h4><p><span class="mono">batch = get_next_batch_to_run()</span> 组好第 N 批，<span class="mono">run_batch(batch)</span> 把它<strong>丢上 GPU 就走</strong>，把 <span class="mono">(batch.copy(), result)</span> 压进 <span class="mono">result_queue</span>。GPU 开始算，CPU 不阻塞。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>趁 GPU 忙，CPU 备下一批（N+1）</h4><p>GPU 还在算 N 的当口，CPU 已经在<strong>收新请求、组下一拍的批、采样备料</strong>。这段 CPU 时间<strong>不占用 GPU</strong>——它躲在 N 的计算影子里。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>收尾上一步（N−1）的结果</h4><p>从 <span class="mono">result_queue</span> <span class="mono">popleft</span> 出第 N−1 批——它<strong>现在已算完</strong>。<span class="mono">process_batch_result</span> 追加 token、判完成、<strong>释放 KV</strong>、发反分词（第 17 课）。处理的永远是<strong>上一步</strong>。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>更新 last_batch，回到 1</h4><p><span class="mono">last_batch = batch</span>，循环回到第 1 步发射 N+1。每一拍都是"发射这步 / 收尾上步"的错位流水，GPU 接力棒<strong>无停顿交接</strong>。</p></div></div>
</div>

<h2>GPU 永不空转：把每一拍画在时间轴上</h2>
<p>
下面这张图把"慢一拍"摊开看。把时间横着切成四拍（T1…T4），上面一行是 GPU、下面一行是 CPU。在<strong>朴素串行</strong>里，
两行是<strong>错峰</strong>的——GPU 亮时 CPU 暗、CPU 亮时 GPU 暗，总有一方在空等。而在<strong>重叠</strong>里，<strong>GPU 那一行从头亮到尾、没有一个空格</strong>（高亮格就是"GPU 正在算"），
CPU 那一行则在 GPU 的<strong>同一拍里同时工作</strong>：T2 这一拍，GPU 在算第 2 批，CPU<strong>同时</strong>在组第 3 批、并收尾第 1 批的结果。
你能直观看到两点：其一，GPU<strong>从不空转</strong>，这正是吞吐拉满的根；其二，结果总是<strong>晚一拍</strong>落地——第 1 批的结果要等到 T2 才处理完，这就是多出来的那一拍延迟（第 8 课的吞吐/延迟权衡在这里现形）。
还要点明一个常见误解：这"晚一拍"<strong>并不会让单条请求变慢一倍</strong>，它只是在整条生成的最前面<strong>一次性</strong>多压了一拍的固定延迟——
后续每个 token 仍然是一拍一个、稳定吐出。对一条要生成几百个 token 的长回答来说，开头多等一拍几乎察觉不到；
而它换来的是<strong>每一拍</strong>都把 GPU 喂满，吞吐的收益是持续累积的。所以这笔交易在绝大多数服务场景里都极其划算：用一次性的、微小的延迟，换走全程的算力浪费。
只有在那种"只生成极少 token、且对首包时延锱铢必较"的场景里，才需要重新掂量这一拍是否值得。
</p>

<div class="cellgroup">
  <div class="cg-cap">朴素串行 vs 重叠流水线：高亮格 = GPU 正在算（看 GPU 那一行有没有空格）</div>
  <div class="cells">
    <span class="lab">串行·GPU</span><span class="cell">T1 算批1</span><span class="cell">T2 空等</span><span class="cell">T3 算批2</span><span class="cell">T4 空等</span><span class="sep"></span><span class="q">有空格→算力浪费</span>
  </div>
  <div class="cells">
    <span class="lab">串行·CPU</span><span class="cell">T1 空等</span><span class="cell">T2 组批2</span><span class="cell">T3 空等</span><span class="cell">T4 组批3</span><span class="sep"></span><span class="q">两行错峰、轮流闲</span>
  </div>
  <div class="cells">
    <span class="lab">重叠·GPU</span><span class="cell hl">T1 算批1</span><span class="cell hl">T2 算批2</span><span class="cell hl">T3 算批3</span><span class="cell hl">T4 算批4</span><span class="sep"></span><span class="q">无空格→满载</span>
  </div>
  <div class="cells">
    <span class="lab">重叠·CPU</span><span class="cell">T1 组批2</span><span class="cell">T2 组批3·收尾1</span><span class="cell">T3 组批4·收尾2</span><span class="cell">T4 组批5·收尾3</span><span class="sep"></span><span class="q">藏进 GPU 影子里</span>
  </div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 330" role="img" aria-label="串行 vs 重叠时间轴：串行里 GPU 在 CPU 调度/处理时空等、两行错峰；重叠里 GPU 一行从头满到尾没有空格，CPU 的组批与收尾躲进 GPU 计算影子">
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">串行 event_loop_normal — GPU 有空等</text>
    <text x="24" y="80" style="fill:var(--muted);font-size:12px">GPU</text>
    <text x="24" y="126" style="fill:var(--muted);font-size:12px">CPU</text>
    <rect x="64" y="58" width="124" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="126" y="80" text-anchor="middle" style="fill:var(--red);font-size:12px">空等</text>
    <rect x="188" y="58" width="142" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="259" y="80" text-anchor="middle" class="mono" style="font-size:12px">前向 批N</text>
    <rect x="330" y="58" width="124" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="392" y="80" text-anchor="middle" style="fill:var(--red);font-size:12px">空等</text>
    <rect x="454" y="58" width="124" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="516" y="80" text-anchor="middle" style="fill:var(--red);font-size:12px">空等</text>
    <rect x="578" y="58" width="142" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="649" y="80" text-anchor="middle" class="mono" style="font-size:12px">前向 批N+1</text>
    <rect x="64" y="104" width="124" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="126" y="126" text-anchor="middle" class="mono" style="font-size:12px">调度 批N</text>
    <rect x="188" y="104" width="142" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="259" y="126" text-anchor="middle" style="fill:var(--faint);font-size:12px">闲</text>
    <rect x="330" y="104" width="124" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="392" y="126" text-anchor="middle" class="mono" style="font-size:12px">处理 批N</text>
    <rect x="454" y="104" width="124" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="516" y="126" text-anchor="middle" class="mono" style="font-size:12px">调度 批N+1</text>
    <rect x="578" y="104" width="142" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="649" y="126" text-anchor="middle" style="fill:var(--faint);font-size:12px">闲</text>
    <text x="24" y="160" style="fill:var(--red);font-size:12px">↑ GPU 那行有红色空格 = 算力被晾着</text>
    <text x="24" y="196" style="font-weight:700;fill:var(--accent-ink)">重叠 event_loop_overlap — GPU 不停转，CPU 躲进影子</text>
    <rect x="64" y="212" width="212" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="170" y="234" text-anchor="middle" class="mono" style="font-size:12px">前向 批N</text>
    <rect x="276" y="212" width="212" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="382" y="234" text-anchor="middle" class="mono" style="font-size:12px">前向 批N+1</text>
    <rect x="488" y="212" width="212" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="594" y="234" text-anchor="middle" class="mono" style="font-size:12px">前向 批N+2</text>
    <rect x="64" y="258" width="212" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="170" y="280" text-anchor="middle" class="mono" style="font-size:11px">组 批N+1 · 收尾 批N−1</text>
    <rect x="276" y="258" width="212" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="382" y="280" text-anchor="middle" class="mono" style="font-size:11px">组 批N+2 · 收尾 批N</text>
    <rect x="488" y="258" width="212" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="594" y="280" text-anchor="middle" class="mono" style="font-size:11px">组 批N+3 · 收尾 批N+1</text>
    <text x="24" y="314" style="fill:var(--teal);font-size:12px">↑ GPU 那行从头满到尾、没有一个空格 = 零开销</text>
  </svg>
  <div class="figcap"><b>图 1 · 串行 vs 重叠（时间轴）</b> — 串行里 GPU 与 CPU <strong>错峰</strong>、GPU 在 CPU 组批/处理时<strong>空等</strong>；重叠里 GPU 一行<strong>从头满到尾、没有空格</strong>，CPU 的组批与收尾全躲进 GPU 计算的影子里。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 280" role="img" aria-label="重叠流水线：同一时间格里 GPU 算第 N 批、CPU 同时组第 N+1 批、还收尾第 N−1 批，三条道整整错开一拍">
    <text x="24" y="26" style="font-weight:700;fill:var(--accent-ink)">重叠流水线：同一时刻，三条道各错一拍</text>
    <text x="250" y="50" text-anchor="middle" style="fill:var(--muted);font-size:12px">T1</text>
    <text x="430" y="50" text-anchor="middle" style="fill:var(--muted);font-size:12px">T2</text>
    <text x="610" y="50" text-anchor="middle" style="fill:var(--muted);font-size:12px">T3</text>
    <line x1="340" y1="58" x2="340" y2="224" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="520" y1="58" x2="520" y2="224" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <rect x="156" y="60" width="188" height="168" rx="8" style="fill:none;stroke:var(--accent);stroke-width:1.5;stroke-dasharray:6 4"/>
    <text x="24" y="90" style="fill:var(--muted);font-size:12px">GPU · 前向</text>
    <text x="24" y="146" style="fill:var(--muted);font-size:12px">CPU · 组下一批</text>
    <text x="24" y="202" style="fill:var(--muted);font-size:12px">CPU · 收尾上一批</text>
    <rect x="160" y="64" width="180" height="40" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="250" y="89" text-anchor="middle" class="mono" style="font-size:12px">前向 批N</text>
    <rect x="340" y="64" width="180" height="40" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="430" y="89" text-anchor="middle" class="mono" style="font-size:12px">前向 批N+1</text>
    <rect x="520" y="64" width="180" height="40" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="610" y="89" text-anchor="middle" class="mono" style="font-size:12px">前向 批N+2</text>
    <rect x="160" y="120" width="180" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="250" y="145" text-anchor="middle" class="mono" style="font-size:12px">组 批N+1</text>
    <rect x="340" y="120" width="180" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="430" y="145" text-anchor="middle" class="mono" style="font-size:12px">组 批N+2</text>
    <rect x="520" y="120" width="180" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="610" y="145" text-anchor="middle" class="mono" style="font-size:12px">组 批N+3</text>
    <rect x="160" y="176" width="180" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="201" text-anchor="middle" class="mono" style="font-size:12px">收尾 批N−1</text>
    <rect x="340" y="176" width="180" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="430" y="201" text-anchor="middle" class="mono" style="font-size:12px">收尾 批N</text>
    <rect x="520" y="176" width="180" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="610" y="201" text-anchor="middle" class="mono" style="font-size:12px">收尾 批N+1</text>
    <text x="24" y="252" style="fill:var(--muted);font-size:12px">同一格（T1）里：GPU 算 N、CPU 组 N+1、CPU 收尾 N−1 —— 整整错一拍（one step out of phase）</text>
  </svg>
  <div class="figcap"><b>图 2 · 重叠流水线（错一拍）</b> — 同一个时间格里，GPU 在算第 N 批、CPU <strong>同时</strong>在组第 N+1 批、还顺手收尾第 N−1 批；三条道整整<strong>错开一拍</strong>（one step out of phase），于是没有任何一方干等。</div>
</div>

<p>
举个具体数字：decode 一拍里 GPU 前向约 <strong>8 毫秒</strong>、CPU 调度约 <strong>4 毫秒</strong>。串行下一拍 = 12 毫秒，GPU 利用率只有 <span class="mono">8/12 ≈ 67%</span>；
重叠把那 4 毫秒的 CPU 活塞进 GPU 那 8 毫秒的影子里，一拍压回 <strong>8 毫秒</strong>、GPU 利用率拉到 <span class="mono">≈ 100%</span>——同一张卡吞吐凭空多出约 <strong>50%</strong>，代价只是开头一次性多压一拍（错一拍）。
</p>

<h2>账要算清：零开销不是没代价</h2>
<p>
重叠不是免费午餐，它把"开销"从 GPU 空转<strong>换成了一拍延迟 + 复杂度</strong>。好处一栏很硬：<strong>调度开销近乎归零</strong>、<strong>GPU 利用率拉满</strong>，
decode 密集场景吞吐显著抬升。好处之所以这么实在，是因为它榨的是<strong>本来就闲着的那段 CPU 时间</strong>，等于不花额外成本白捡吞吐。代价一栏也要看清：<strong>+1 拍延迟</strong>——你永远比实时慢一步，对单条请求的首 token / 末 token 时延略有影响（第 8 课）；
还有<strong>状态管理更难</strong>——下一批要在当前结果<em>还没算出来</em>时就先搭好，意味着采样 token、KV 记账等存在<strong>跨拍依赖</strong>，
SGLang 用 <span class="mono">overlap_utils.py</span> 里的 <span class="mono">FutureMap</span> 这套 future/event 对象替你把这些依赖串好、保证正确（而 <span class="mono">result_queue</span> 队列就在 <span class="mono">scheduler.py</span> 里）。还有些场景（如某些需要立刻拿到上一步结果才能组下一批的情形，
源码里的 <span class="mono">is_disable_overlap_for_batch</span>）会临时退回不重叠、先把上一批收尾。理解这张账，你就明白为什么这叫"零开销"——
不是没有成本，而是<strong>把成本藏到了不影响吞吐的地方</strong>。
最后再强调一句工程直觉：重叠之所以能成立，前提是"调度"和"计算"<strong>分属两种不同的硬件资源</strong>（CPU 与 GPU），它们能真正物理并行。
如果两件事抢的是同一块资源，重叠就只是把先后顺序换了个写法、并不会更快。SGLang 这套设计精准地踩在了这个前提上：
GPU 埋头做矩阵乘的几毫秒，对 CPU 来说是大把可用的空闲时间，正好拿来组下一批、采样、记账。
把这块本就闲着的 CPU 时间利用起来，几乎是"无中生有"地榨出了吞吐——这也是为什么它被称作"零开销"而不仅仅是"低开销"。
</p>

<table class="t">
  <tr><th>维度</th><th>朴素 event_loop_normal</th><th>重叠 event_loop_overlap</th></tr>
  <tr><td>CPU 调度开销</td><td>暴露在关键路径，GPU 空等</td><td class="mono">≈ 0（藏进 GPU 影子）</td></tr>
  <tr><td>GPU 利用率</td><td>decode 步明显有空隙</td><td><strong>满载、无空格</strong></td></tr>
  <tr><td>吞吐</td><td>被循环速度封顶</td><td><strong>显著抬升</strong>（v0.4 招牌）</td></tr>
  <tr><td>延迟</td><td>实时收尾</td><td><strong>+1 拍</strong>（总慢一步）</td></tr>
  <tr><td>实现复杂度</td><td>简单、直来直去</td><td>需 <span class="mono">result_queue</span> + 跨拍依赖管理</td></tr>
</table>

<h2>读源码：event_loop_overlap 的错位流水</h2>
<p>
真身就在 <span class="mono">scheduler.py</span> 的 <span class="mono">event_loop_overlap</span>。抓三个关键点：一是 <span class="mono">result_queue</span> 这个双端队列，
存放"已发射但还没收尾"的批；二是 <span class="mono">run_batch</span> 之后<strong>立刻 append、不阻塞</strong>，把收尾推迟到下一拍；
三是 <span class="mono">pop_and_process</span> 从队首取出<strong>上一拍</strong>的批做 <span class="mono">process_batch_result</span>——这一句执行时，GPU 正忙着算本拍。错位，就是这么实现的。
读这段时还可以留意一处细节：批入队前调用了 <span class="mono">batch.copy()</span>。为什么要拷贝？因为本拍的批对象接下来可能被 <span class="mono">get_next_batch_to_run</span> 复用或就地改写，
而队列里那份必须保留"发射时的快照"，等下一拍收尾时才不会读到被篡改的状态——这正是跨拍依赖管理的一个缩影。源码里还有 <span class="mono">is_disable_overlap_for_batch</span> 的分支，
在少数必须立刻拿到上一步结果才能继续的情形下，提前把上一批收尾、临时退回不重叠，保证正确性优先于性能。
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.event_loop_overlap</span><span class="ln">错位流水 / 慢一拍</span></div>
<pre><span class="kw">def</span> event_loop_overlap(self):
    <span class="cm"># 重叠 CPU 调度与 GPU 计算的事件循环</span>
    self.result_queue = deque()                  <span class="cm"># 存"已发射、待收尾"的批</span>

    <span class="kw">def</span> pop_and_process():
        <span class="cm"># 收尾上一步(N-1)的结果——推迟一拍处理</span>
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    <span class="kw">while</span> <span class="kw">True</span>:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()     <span class="cm"># CPU：组本步(N)的批</span>
        self.cur_batch = batch

        <span class="kw">if</span> batch:                                <span class="cm"># 发射本步前向，不等它算完</span>
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))

        <span class="kw">if</span> self.last_batch:                      <span class="cm"># GPU 算 N 时，CPU 回头收尾 N-1</span>
            pop_and_process()

        self.last_batch = batch                  <span class="cm"># 回到开头：下一拍发射 N+1</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.process_batch_result</span><span class="ln">处理（上一批的）前向结果：追加 token、判完成、释放 KV、送去解码</span></div>
<pre><span class="kw">def</span> process_batch_result(self, batch, result):
    <span class="cm"># 在重叠循环里，这一步处理的是"上一批"——此刻当前批正在 GPU 上跑：</span>
    <span class="cm">#  - 把采样出的下一个 token 追加到每条请求</span>
    <span class="cm">#  - 标记已完成的请求（停止符 / 长度）并释放它们的 KV</span>
    <span class="cm">#  - 把输出送给 DetokenizerManager</span>
    ...</pre>
</div>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>问题在串行</strong>：朴素 <span class="mono">event_loop_normal</span>（第 18 课）里 CPU 组批/采样/记账与 GPU 前向<strong>顺序排队</strong>，CPU 干活时 GPU 空等。decode 步 GPU 极快，CPU 开销占比大，循环速度给吞吐<strong>封顶</strong>。</li>
    <li><strong>修复靠重叠</strong>：<span class="mono">event_loop_overlap</span> 先<strong>发射</strong>第 N 步前向（不等），CPU 立刻组第 N+1 批、并<strong>收尾第 N−1 步</strong>的结果（存在 <span class="mono">result_queue</span> / future 里，推迟一拍）。CPU 调度被<strong>藏进 GPU 计算的影子</strong>。</li>
    <li><strong>什么叠什么</strong>：第 N 步的 <strong>GPU 前向</strong> ∥ 第 N+1 步的 <strong>CPU 组批</strong> + 第 N−1 步的 <strong>CPU 收尾</strong>。GPU <strong>从不空转</strong>，调度开销近乎归零。</li>
    <li><strong>代价是慢一拍</strong>：你永远比实时<strong>晚一步</strong>处理结果（第 8 课的吞吐/延迟权衡），外加跨拍状态管理更难——下一批在当前结果未出时就搭好，SGLang 用 <span class="mono">overlap_utils.py</span> 的 <span class="mono">FutureMap</span> 扛依赖。</li>
    <li><strong>与连续批处理是绝配</strong>：连续批处理（第 5 课）让批<strong>永远满载</strong>，重叠调度器让<strong>喂满批的调度不要钱</strong>。这是 v0.4 的招牌优化，也是第 1/3 课点名的那项"零开销"。配合 CUDA Graph（第 27 课）把单步再压薄。</li>
  </ul>
</div>
""",
             "en": r"""
<p class="lead">
This is Part 5's <strong>signature</strong> lesson—the optimization name-dropped back in Lessons 1 and 3: the <strong>zero-overhead overlap scheduler</strong>.
Recall Lesson 18: every scheduler step mixes two kinds of work—<strong>CPU scheduling</strong> (build the batch, sample, bookkeep, prepare the next step's inputs)
and the <strong>GPU forward</strong>. In the naive <span class="inline">event_loop_normal</span>, these are <strong>strictly serial</strong>:
GPU finishes → CPU then forms the next batch → GPU runs again. The catch: <strong>while the CPU forms the batch, the GPU sits idle</strong>.
For decode—where each step is very fast—the fixed CPU overhead becomes a sizable fraction of a step, so a chunk of expensive GPU time is wasted.
This lesson shows how SGLang <strong>pipelines</strong> the two to erase that idle gap entirely: keep the GPU <strong>never idle</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Real-world analogy</div>
  Think of one step as a <strong>relay race / two-stage assembly line</strong>. The naive loop is a single runner: after sprinting this leg he must <strong>stop</strong>, walk back to the line, tie his shoes,
  catch his breath—and only then sprint the next leg, leaving the track (GPU) empty in between. The overlap scheduler uses <strong>two runners</strong>: while the current runner (GPU) sprints all-out,
  the next runner (CPU) is <strong>already at the line, laced up and set</strong>, so the baton passes with <strong>zero pause</strong>.
  Or a kitchen: while the current dish sizzles on the stove (GPU), the prep cook (CPU) is already chopping and plating the <strong>next order</strong>—
  <strong>the stove never burns empty</strong>. The only cost: you are forever <strong>one beat behind</strong>, because the work you finish is always the previous leg's result.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>hide this step's CPU scheduling inside the previous step's GPU compute</strong>. Instead of "compute → finish → form next" in series,
  <span class="inline">event_loop_overlap</span> <strong>launches</strong> step N's GPU forward, <strong>does not wait for it</strong>, and immediately has the CPU form step N+1's batch and <strong>finish the previous (N−1) step's</strong> result—
  the previous result lives in a <span class="mono">result_queue</span> (a future) and is processed <strong>one beat late</strong>. Each step's CPU overhead now falls neatly within the "shadow" of that step's GPU compute and is <strong>fully hidden</strong>,
  presenting as <strong>near-zero scheduling overhead</strong> with the GPU saturated.
  The cost is <strong>one extra beat of latency</strong> (you're always one step behind) plus finer state management—the next batch is built before the current result is known, so sampled tokens and KV bookkeeping have cross-beat dependencies,
  which SGLang handles for you via the <span class="mono">FutureMap</span> in <span class="mono">overlap_utils.py</span> (the <span class="mono">result_queue</span> deque lives in <span class="mono">scheduler.py</span> itself). This is v0.4's "zero-overhead batch scheduler"; it pairs perfectly with continuous batching (Lesson 5): <strong>the batch is always full, and the scheduling that keeps it full costs nothing</strong>.
</div>

<h2>The problem: in the serial loop, the GPU idles while the CPU forms the batch</h2>
<p>
Do the accounting first. The only thing that truly burns GPU in a step is <span class="mono">run_batch</span> (→ ModelRunner.forward, Lesson 24);
the rest—<span class="mono">get_next_batch_to_run</span> (form the batch), <span class="mono">sample</span>, <span class="mono">process_batch_result</span> (append tokens, detect finished, free KV, bookkeep, prep next inputs)—
is all <strong>lightweight CPU work</strong>. The naive <span class="inline">event_loop_normal</span> lines them up <strong>in a straight line</strong>: CPU forms batch → GPU forward → CPU finishes → CPU forms next → GPU forward…
The trouble is at the seams: whenever the CPU is forming/sampling/bookkeeping, <strong>the GPU has nothing to do and just idles</strong>; conversely, the CPU idles during the forward.
Two expensive resources take turns being busy and idle—neither is fed. For prefill (GPU-heavy) the CPU share is small; but once you hit decode—
one token per request per step, an extremely fast forward—that fixed scheduling overhead becomes a <strong>conspicuous slice</strong> of the step.
The faster the loop spins and the less the GPU rests, the more this fixed overhead stings: it <strong>caps</strong> whole-engine throughput.
</p>

<div class="cols">
  <div class="col"><h4>Naive serial: event_loop_normal (Lesson 18)</h4><p>Within a step, <strong>CPU forms batch → GPU forward → CPU finishes</strong> queue up with no overlap. The GPU <strong>idles for the whole stretch</strong> while the CPU forms/samples/bookkeeps. Decode steps make the GPU very fast, so CPU overhead dominates and <strong>compute is wasted</strong>; loop speed caps throughput.</p></div>
  <div class="col"><h4>Overlap pipeline: event_loop_overlap (this lesson)</h4><p><strong>Launch</strong> this step's GPU forward (don't wait), and the CPU immediately forms the next batch and <strong>finishes the previous step</strong>. CPU scheduling is <strong>hidden in the GPU compute's shadow</strong>, so the GPU has <strong>no gaps</strong>—scheduling overhead near zero. Cost: one beat behind.</p></div>
</div>

<h2>The fix: tuck the next step's scheduling into this step's GPU shadow</h2>
<p>
The overlap loop has just three core moves, but the order is everything. First, <strong>launch</strong> step N's forward: call <span class="mono">run_batch</span> to hand the batch to the GPU,
but <strong>return immediately—don't block on the result</strong>—and <span class="mono">append</span> this "launched, pending" batch with its result handle onto <span class="mono">result_queue</span>.
Second, while the GPU is busy on N, <strong>the CPU goes off to form step N+1's batch</strong> (next step's <span class="mono">get_next_batch_to_run</span> is effectively prepped here).
Third, <strong>finish the previous (N−1) step</strong>: <span class="mono">popleft</span> from <span class="mono">result_queue</span> that prior batch and its <em>now-ready</em> result and run <span class="mono">process_batch_result</span>—
append tokens, detect finished, free KV. Note the stagger: <strong>when we process N−1's result, the GPU is busy computing N</strong>. So every slice of CPU work lands inside the GPU compute's shadow.
That's where "one beat behind" comes from, and where "zero-overhead" comes from—you always process the <strong>previous</strong> step's result, yet no step's CPU time <strong>ever makes the GPU wait</strong>.
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Launch step N's forward, don't wait</h4><p><span class="mono">batch = get_next_batch_to_run()</span> forms batch N; <span class="mono">run_batch(batch)</span> <strong>fires it onto the GPU and moves on</strong>, pushing <span class="mono">(batch.copy(), result)</span> onto <span class="mono">result_queue</span>. GPU starts; CPU doesn't block.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>While GPU is busy, CPU preps N+1</h4><p>With the GPU still on N, the CPU is already <strong>receiving requests, forming the next batch, prepping samples</strong>. This CPU time <strong>doesn't occupy the GPU</strong>—it hides in N's compute shadow.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Finish the previous step (N−1)</h4><p><span class="mono">popleft</span> batch N−1 from <span class="mono">result_queue</span>—it's <strong>now done</strong>. <span class="mono">process_batch_result</span> appends tokens, detects finished, <strong>frees KV</strong>, sends to detok (Lesson 17). You always process the <strong>previous</strong> step.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Update last_batch, back to 1</h4><p><span class="mono">last_batch = batch</span>; loop back to launch N+1. Every beat is a staggered "launch this / finish previous" pipeline; the GPU baton passes with <strong>no pause</strong>.</p></div></div>
</div>

<h2>The GPU never idles: draw every beat on a timeline</h2>
<p>
This figure spreads "one beat behind" out in the open. Slice time into four beats (T1…T4); the top row is the GPU, the bottom row the CPU. In the <strong>naive serial</strong> case,
the two rows are <strong>out of phase</strong>—GPU lit while CPU dark, CPU lit while GPU dark, always one side idling. In the <strong>overlap</strong> case, <strong>the GPU row is lit end to end with no empty cell</strong> (highlighted cell = "GPU computing"),
while the CPU row works <strong>in the same beat as the GPU</strong>: at T2, the GPU computes batch 2 while the CPU <strong>simultaneously</strong> forms batch 3 and finishes batch 1's result.
You can see two things at a glance: one, the GPU <strong>never idles</strong>—the root of saturated throughput; two, results always land <strong>one beat late</strong>—batch 1's result isn't finished until T2, which is exactly the extra beat of latency (Lesson 8's throughput/latency trade-off in the flesh).
</p>

<div class="cellgroup">
  <div class="cg-cap">Naive serial vs overlap pipeline: highlighted cell = GPU computing (watch whether the GPU row has gaps)</div>
  <div class="cells">
    <span class="lab">serial·GPU</span><span class="cell">T1 run b1</span><span class="cell">T2 idle</span><span class="cell">T3 run b2</span><span class="cell">T4 idle</span><span class="sep"></span><span class="q">gaps → wasted compute</span>
  </div>
  <div class="cells">
    <span class="lab">serial·CPU</span><span class="cell">T1 idle</span><span class="cell">T2 form b2</span><span class="cell">T3 idle</span><span class="cell">T4 form b3</span><span class="sep"></span><span class="q">rows out of phase</span>
  </div>
  <div class="cells">
    <span class="lab">overlap·GPU</span><span class="cell hl">T1 run b1</span><span class="cell hl">T2 run b2</span><span class="cell hl">T3 run b3</span><span class="cell hl">T4 run b4</span><span class="sep"></span><span class="q">no gaps → saturated</span>
  </div>
  <div class="cells">
    <span class="lab">overlap·CPU</span><span class="cell">T1 form b2</span><span class="cell">T2 form b3·finish b1</span><span class="cell">T3 form b4·finish b2</span><span class="cell">T4 form b5·finish b3</span><span class="sep"></span><span class="q">hidden in GPU shadow</span>
  </div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 330" role="img" aria-label="Serial vs overlap timeline: in serial the GPU idles while the CPU schedules/processes and the two rows are out of phase; in overlap the GPU row is full end to end with no gap, while the CPU's batch-forming and finishing hide in the GPU compute shadow">
    <text x="24" y="26" style="font-weight:700;fill:var(--muted)">Serial event_loop_normal — the GPU idles</text>
    <text x="24" y="80" style="fill:var(--muted);font-size:12px">GPU</text>
    <text x="24" y="126" style="fill:var(--muted);font-size:12px">CPU</text>
    <rect x="64" y="58" width="124" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="126" y="80" text-anchor="middle" style="fill:var(--red);font-size:12px">idle</text>
    <rect x="188" y="58" width="142" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="259" y="80" text-anchor="middle" class="mono" style="font-size:12px">forward N</text>
    <rect x="330" y="58" width="124" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="392" y="80" text-anchor="middle" style="fill:var(--red);font-size:12px">idle</text>
    <rect x="454" y="58" width="124" height="34" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="516" y="80" text-anchor="middle" style="fill:var(--red);font-size:12px">idle</text>
    <rect x="578" y="58" width="142" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="649" y="80" text-anchor="middle" class="mono" style="font-size:12px">forward N+1</text>
    <rect x="64" y="104" width="124" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="126" y="126" text-anchor="middle" class="mono" style="font-size:12px">schedule N</text>
    <rect x="188" y="104" width="142" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="259" y="126" text-anchor="middle" style="fill:var(--faint);font-size:12px">idle</text>
    <rect x="330" y="104" width="124" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="392" y="126" text-anchor="middle" class="mono" style="font-size:12px">process N</text>
    <rect x="454" y="104" width="124" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="516" y="126" text-anchor="middle" class="mono" style="font-size:12px">schedule N+1</text>
    <rect x="578" y="104" width="142" height="34" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 4"/>
    <text x="649" y="126" text-anchor="middle" style="fill:var(--faint);font-size:12px">idle</text>
    <text x="24" y="160" style="fill:var(--red);font-size:12px">↑ red gaps in the GPU row = wasted compute</text>
    <text x="24" y="196" style="font-weight:700;fill:var(--accent-ink)">Overlap event_loop_overlap — GPU never idles; CPU hides in the shadow</text>
    <rect x="64" y="212" width="212" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="170" y="234" text-anchor="middle" class="mono" style="font-size:12px">forward N</text>
    <rect x="276" y="212" width="212" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="382" y="234" text-anchor="middle" class="mono" style="font-size:12px">forward N+1</text>
    <rect x="488" y="212" width="212" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="594" y="234" text-anchor="middle" class="mono" style="font-size:12px">forward N+2</text>
    <rect x="64" y="258" width="212" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="170" y="280" text-anchor="middle" class="mono" style="font-size:11px">build N+1 · finish N−1</text>
    <rect x="276" y="258" width="212" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="382" y="280" text-anchor="middle" class="mono" style="font-size:11px">build N+2 · finish N</text>
    <rect x="488" y="258" width="212" height="34" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="594" y="280" text-anchor="middle" class="mono" style="font-size:11px">build N+3 · finish N+1</text>
    <text x="24" y="314" style="fill:var(--teal);font-size:12px">↑ the GPU row is full end to end, not one gap = zero overhead</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Serial vs overlap (timeline)</b> — in serial the GPU and CPU are <strong>out of phase</strong> and the GPU <strong>idles</strong> while the CPU forms/processes; in overlap the GPU row is <strong>full end to end with no gap</strong>, and the CPU's batch-forming and finishing hide inside the GPU compute's shadow.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 280" role="img" aria-label="Overlap pipeline: within one time slot the GPU computes batch N while the CPU simultaneously builds batch N+1 and finishes batch N−1; the three lanes are staggered by exactly one step">
    <text x="24" y="26" style="font-weight:700;fill:var(--accent-ink)">Overlap pipeline: one time slot, three lanes each one step out of phase</text>
    <text x="250" y="50" text-anchor="middle" style="fill:var(--muted);font-size:12px">T1</text>
    <text x="430" y="50" text-anchor="middle" style="fill:var(--muted);font-size:12px">T2</text>
    <text x="610" y="50" text-anchor="middle" style="fill:var(--muted);font-size:12px">T3</text>
    <line x1="340" y1="58" x2="340" y2="224" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <line x1="520" y1="58" x2="520" y2="224" style="stroke:var(--line);stroke-width:1.2;stroke-dasharray:4 4"/>
    <rect x="156" y="60" width="188" height="168" rx="8" style="fill:none;stroke:var(--accent);stroke-width:1.5;stroke-dasharray:6 4"/>
    <text x="24" y="90" style="fill:var(--muted);font-size:12px">GPU · forward</text>
    <text x="24" y="146" style="fill:var(--muted);font-size:12px">CPU · build next</text>
    <text x="24" y="202" style="fill:var(--muted);font-size:12px">CPU · finish prev</text>
    <rect x="160" y="64" width="180" height="40" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="250" y="89" text-anchor="middle" class="mono" style="font-size:12px">forward N</text>
    <rect x="340" y="64" width="180" height="40" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="430" y="89" text-anchor="middle" class="mono" style="font-size:12px">forward N+1</text>
    <rect x="520" y="64" width="180" height="40" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="610" y="89" text-anchor="middle" class="mono" style="font-size:12px">forward N+2</text>
    <rect x="160" y="120" width="180" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="250" y="145" text-anchor="middle" class="mono" style="font-size:12px">build N+1</text>
    <rect x="340" y="120" width="180" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="430" y="145" text-anchor="middle" class="mono" style="font-size:12px">build N+2</text>
    <rect x="520" y="120" width="180" height="40" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="610" y="145" text-anchor="middle" class="mono" style="font-size:12px">build N+3</text>
    <rect x="160" y="176" width="180" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="201" text-anchor="middle" class="mono" style="font-size:12px">finish N−1</text>
    <rect x="340" y="176" width="180" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="430" y="201" text-anchor="middle" class="mono" style="font-size:12px">finish N</text>
    <rect x="520" y="176" width="180" height="40" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="610" y="201" text-anchor="middle" class="mono" style="font-size:12px">finish N+1</text>
    <text x="24" y="252" style="fill:var(--muted);font-size:12px">Same slot (T1): GPU runs N, CPU builds N+1, CPU finishes N−1 — exactly one step out of phase</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Overlap pipeline (one step out of phase)</b> — within one time slot the GPU computes batch N while the CPU <strong>simultaneously</strong> builds batch N+1 and finishes batch N−1; the three lanes are staggered by exactly <strong>one step</strong> (one step out of phase), so no side ever waits.</div>
</div>

<p>
A concrete number: in one decode beat the GPU forward takes about <strong>8 ms</strong> and CPU scheduling about <strong>4 ms</strong>. Serial makes the beat 12 ms with the GPU only <span class="mono">8/12 ≈ 67%</span> utilized;
overlap tucks those 4 ms of CPU work into the GPU's 8 ms shadow, shrinking the beat back to <strong>8 ms</strong> and pushing GPU utilization to <span class="mono">≈ 100%</span>—the same card yields roughly <strong>50%</strong> more throughput, at the cost of one one-time extra beat (one step out of phase).
</p>

<h2>Account honestly: zero-overhead is not zero-cost</h2>
<p>
Overlap is no free lunch; it <strong>trades GPU idling for one beat of latency plus complexity</strong>. The upside column is hard: <strong>near-zero scheduling overhead</strong>, <strong>saturated GPU utilization</strong>,
and clearly higher throughput in decode-heavy regimes. The cost column matters too: <strong>+1 beat of latency</strong>—you're forever one step behind, nudging single-request token timings (Lesson 8);
and <strong>harder state management</strong>—the next batch is built before the current result <em>exists</em>, so sampled tokens and KV bookkeeping carry <strong>cross-beat dependencies</strong>,
which SGLang threads through the <span class="mono">FutureMap</span> in <span class="mono">overlap_utils.py</span> to keep correct. Some cases (where you must have the previous result before forming the next,
the source's <span class="mono">is_disable_overlap_for_batch</span>) temporarily fall back to no-overlap and finish the prior batch first. Understand this ledger and you see why it's "zero-overhead":
not cost-free, but with the cost <strong>moved somewhere that doesn't dent throughput</strong>.
</p>

<table class="t">
  <tr><th>Dimension</th><th>Naive event_loop_normal</th><th>Overlap event_loop_overlap</th></tr>
  <tr><td>CPU scheduling overhead</td><td>On the critical path; GPU idles</td><td class="mono">≈ 0 (hidden in GPU shadow)</td></tr>
  <tr><td>GPU utilization</td><td>Visible gaps on decode steps</td><td><strong>Saturated, no gaps</strong></td></tr>
  <tr><td>Throughput</td><td>Capped by loop speed</td><td><strong>Markedly higher</strong> (v0.4 headline)</td></tr>
  <tr><td>Latency</td><td>Finished in real time</td><td><strong>+1 beat</strong> (always one behind)</td></tr>
  <tr><td>Implementation complexity</td><td>Simple, straight-line</td><td>Needs <span class="mono">result_queue</span> + cross-beat deps</td></tr>
</table>

<h2>Read the source: event_loop_overlap's staggered pipeline</h2>
<p>
The real thing lives in <span class="mono">scheduler.py</span>'s <span class="mono">event_loop_overlap</span>. Grab three points: one, <span class="mono">result_queue</span>, a deque
holding "launched but not yet finished" batches; two, right after <span class="mono">run_batch</span> we <strong>append immediately, never blocking</strong>, deferring the finish to the next beat;
three, <span class="mono">pop_and_process</span> takes the <strong>previous</strong> beat's batch off the front and runs <span class="mono">process_batch_result</span>—and while that line executes, the GPU is busy on this beat. That's how the stagger is built.
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.event_loop_overlap</span><span class="ln">staggered / one beat behind</span></div>
<pre><span class="kw">def</span> event_loop_overlap(self):
    <span class="cm"># A loop that overlaps CPU scheduling with GPU compute</span>
    self.result_queue = deque()                  <span class="cm"># holds "launched, pending-finish" batches</span>

    <span class="kw">def</span> pop_and_process():
        <span class="cm"># Finish the previous step (N-1) result -- deferred one beat</span>
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    <span class="kw">while</span> <span class="kw">True</span>:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()     <span class="cm"># CPU: form this step's (N) batch</span>
        self.cur_batch = batch

        <span class="kw">if</span> batch:                                <span class="cm"># launch this step's forward, don't wait</span>
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))

        <span class="kw">if</span> self.last_batch:                      <span class="cm"># while GPU runs N, CPU finishes N-1</span>
            pop_and_process()

        self.last_batch = batch                  <span class="cm"># back to top: next beat launches N+1</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.process_batch_result</span><span class="ln">process a batch's forward result: append tokens, check finished, free KV, send to detokenizer</span></div>
<pre><span class="kw">def</span> process_batch_result(self, batch, result):
    <span class="cm"># in the OVERLAP loop this runs for the PREVIOUS batch while the</span>
    <span class="cm"># current batch is already running on the GPU:</span>
    <span class="cm">#  - append the sampled next tokens to each request</span>
    <span class="cm">#  - mark finished requests (stop / length) and free their KV</span>
    <span class="cm">#  - send outputs to the DetokenizerManager</span>
    ...</pre>
</div>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>The problem is serialization</strong>: in naive <span class="mono">event_loop_normal</span> (Lesson 18) CPU forming/sampling/bookkeeping and the GPU forward <strong>queue up in order</strong>, so the GPU idles while the CPU works. Decode makes the GPU very fast, the CPU share dominates, and loop speed <strong>caps</strong> throughput.</li>
    <li><strong>The fix is overlap</strong>: <span class="mono">event_loop_overlap</span> <strong>launches</strong> step N's forward (no wait), and the CPU immediately forms batch N+1 and <strong>finishes step N−1's</strong> result (kept in <span class="mono">result_queue</span> / a future, one beat late). CPU scheduling hides <strong>in the GPU compute's shadow</strong>.</li>
    <li><strong>What overlaps what</strong>: step N's <strong>GPU forward</strong> ∥ step N+1's <strong>CPU batch-forming</strong> + step N−1's <strong>CPU finishing</strong>. The GPU <strong>never idles</strong>; scheduling overhead is near zero.</li>
    <li><strong>The cost is one beat</strong>: you always process results <strong>one step late</strong> (Lesson 8's throughput/latency trade-off), plus harder cross-beat state—the next batch is built before the current result exists, so SGLang carries dependencies via the <span class="mono">FutureMap</span> in <span class="mono">overlap_utils.py</span>.</li>
    <li><strong>It pairs with continuous batching</strong>: continuous batching (Lesson 5) keeps the batch <strong>always full</strong>; the overlap scheduler makes <strong>keeping it full free</strong>. This is v0.4's headline win, the "zero-overhead" cited in Lessons 1/3. Combine with CUDA Graph (Lesson 27) to shave the per-step cost further.</li>
  </ul>
</div>
"""}

LESSON_22 = {"zh": r"""
<p class="lead">
上一课我们看清了调度器“先排序、再限流”的双步动作。这一课盯住限流里最棘手的一种情况：当队首是一个<strong>超长 prompt</strong>（比如 32k token 的长文档）时，会发生什么？
如果让它在<strong>一拍（step）里一口气全部 prefill</strong>，这一步就会变成一头“鲸鱼”——独占整张 GPU 几十毫秒，
排在它后面所有正在 <span class="inline">decode</span> 的请求<strong>全部卡住</strong>，每个人的 token 都吐不出来，延迟瞬间炸裂（TTFT/ITL 飙升，第 8 课）。
<strong>分块预填充（chunked prefill）</strong>就是把这头鲸鱼<strong>切成固定大小的小块</strong>，分摊到好几拍里，
每一拍只啃一小口、并和别人的 decode <strong>混在同一个批</strong>里跑。没有哪一拍是巨大的，所有人的 token 都还在稳定流动。
注意一个常见的误解：分块要解决的<strong>不是“放不下”，而是“算不过来”</strong>——显存里也许有的是 KV 槽位，
真正被一个超长 prefill 卡住的是<strong>时间</strong>，是这一拍前向计算被一个请求独吞，害得别人寸步难行。
理解了这一点，你就会发现分块和上一课的限流其实环环相扣：限流管的是“一拍别接太多请求”，分块管的是“接进来的那个超长请求也别一口吃完”。
记住这一课一句话：<strong>别让一个长请求把一拍撑爆——把它切块，混着 decode 慢慢喂</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把这一拍的 GPU 想成一张<strong>正在热聊的饭桌</strong>。桌上每个人轮流说话（每个请求的 decode 在轮流吐 token），气氛流畅。
  这时上来一道<strong>巨无霸大餐</strong>（32k token 的长 prompt）。笨办法是：让一个人<strong>把整盘菜一口塞进嘴里</strong>——
  他得埋头猛嚼几十秒，这期间<strong>谁也插不上话</strong>，整桌的对话被他一个人噎停了。
  聪明办法是<strong>把大餐切成小口、一口一口吃，同时还能继续聊天</strong>：每一轮只咽下<strong>一小口（一个 chunk）</strong>，
  嚼一下、说句话，再来下一口。这样大餐照样吃得完，桌上的对话（别人的 decode）<strong>一刻也没断</strong>。
  关键的纪律是：<strong>每一口的大小是固定上限的</strong>，绝不允许谁一口把嘴塞满、堵住所有人的发言。
  代价也很实在：这顿大餐要吃好几轮才咽完（多花几拍、它自己的首字更慢一点），但换来的是<strong>满桌人的体验都顺滑</strong>——没人被噎停。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>把“大块 prefill”拆成“多拍小块”，再和 decode 混批，换全局延迟的平滑</strong>。
  谁来切？正是上一课的 <span class="inline">PrefillAdder</span>（第 20 课）。它握着一个 <strong>token 预算</strong>
  （由服务器参数 <span class="mono">chunked_prefill_size</span> 设定每拍能吃多少 prefill token），
  在 <span class="mono">add_one_req</span> 里逐个尝试塞请求：能整段塞下就整段塞，<strong>塞不下就只切下能装的前半段</strong>，剩下的留到下一拍续上。
  于是一个 32k 的 prompt 会被切成若干 chunk，分布在连续好几拍里——它的 KV 缓存（第 4 课）<strong>一块一块地填</strong>，
  只有当<strong>最后一块</strong>填完，这个请求才正式转入 decode、开始吐第一个 token。而这每一拍，调度器跑的都是<strong>混合批（mixed batch）</strong>：
  里面既有“正在 prefill 某一块”的长请求，也有“正在 decode”的其他请求，事件循环（第 18 课）每拍照常推进。
  再把镜头拉远：分块预填充和上一课的限流是同一枚硬币的两面——限流保证“别一拍吃太多”，分块保证“吃不下的也别整条噎着，切开慢慢来”。
  它把一个会制造延迟尖峰的极端情况，驯化成一串平稳的小步，这正是 SGLang 在<strong>长短请求混跑</strong>的真实流量里依然顺滑的关键。
顺带一提，分块也为后面更激进的架构埋下伏笔：既然 prefill 和 decode 的负载特性如此不同，能不能干脆把它们拆到<strong>不同的实例</strong>上各自优化？
那就是第 45 课要讲的 PD 分离。而在单实例内部，分块预填充是把这两种负载<strong>和平共处</strong>的最朴素也最实用的办法。
</div>

<h2>问题：一头鲸鱼堵死整个泳池</h2>
<p>
先看不分块会发生什么。连续批处理（第 5 课）的美好前提是“每一拍都很快、批里大家轮流前进”。
可一旦队首挤进一个 32k token 的 prefill，而你让它<strong>一拍算完</strong>，这一拍的前向计算量就暴涨几十倍——
GPU 被它独占几十毫秒，这期间批里<strong>其他所有 decode 请求只能干等</strong>：它们本该每一拍吐一个 token，现在却要陪着这头鲸鱼一起卡住。
对用户的直接观感就是：正在流式输出的回答<strong>突然顿住</strong>（ITL 尖峰），新请求的首字延迟也被拖长（TTFT 尖峰，第 8 课）。
一个超长请求，毁掉了一整批人的体验。这不是显存放不下的问题——KV 槽位也许够；这是<strong>“一拍算太多”</strong>的问题，
是时间维度上的拥塞。这里值得多想一层：连续批处理之所以高效，靠的是“每一拍都短、批里大家齐步走”，
一旦某一拍因为一个巨型 prefill 而变得奇长，整条流水线的节奏就被打乱——后面所有拍都要为这一拍的失衡买单。
更糟的是，线上流量往往<strong>长短混杂</strong>：99% 是几百 token 的短请求，偶尔混进一个几万 token 的长文档，
而恰恰是这种偶发的长请求，会周期性地制造延迟毛刺，让 P99 延迟变得难看。分块预填充要解决的，正是这种“一条大鱼把整池水搅停”的尖峰。
</p>

<div class="cols">
  <div class="col"><h4>❌ 一拍整段 prefill（尖峰）</h4><p>32k token 在<strong>单拍</strong>算完：这一步巨大无比，独占 GPU 几十毫秒。批里其他请求的 decode <strong>全部 stall</strong>，token 吐不出来。TTFT/ITL <strong>瞬间飙升</strong>（第 8 课），用户看到输出顿住。一头鲸鱼堵死整池。</p></div>
  <div class="col"><h4>✅ 分块 prefill（平滑）</h4><p>32k 被切成<strong>固定大小的 chunk</strong>，分摊到好几拍。每拍只算一个 chunk + <strong>混入大家的 decode</strong>，没有哪一拍巨大。所有人的 token <strong>持续流动</strong>，延迟平稳。多花几拍，换来<strong>全局顺滑</strong>与更好的达标吞吐。</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="长 prefill 切块并与 decode 交错：不分块时一个巨大的 32k prefill 独占一拍、其余请求的 decode 全部 stall；分块后长 prefill 被切成固定大小的小块，块与块之间插入正在进行的 decode，时间线重新变平滑">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">长 prefill 切块、与 decode 交错</text>
    <text x="20" y="86" style="fill:var(--red);font-size:12px;font-weight:700">❌ 不分块</text>
    <rect x="110" y="60" width="280" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="87" text-anchor="middle" style="font-size:12px;font-weight:700">32k prefill · 独占一拍</text>
    <rect x="400" y="60" width="350" height="44" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 3"/>
    <text x="575" y="87" text-anchor="middle" style="fill:var(--red);font-size:12px;font-weight:700">其余请求 decode 全部 stall</text>
    <text x="20" y="192" style="fill:var(--teal);font-size:12px;font-weight:700">✅ 分块</text>
    <rect x="110" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="142" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#1</text>
    <rect x="182" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="196" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="210" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="228" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="260" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#2</text>
    <rect x="300" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="314" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="328" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="346" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="378" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#3</text>
    <rect x="418" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="432" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="446" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="464" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="496" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#N</text>
    <rect x="536" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="550" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="564" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <text x="600" y="193" style="fill:var(--teal);font-size:11px;font-weight:700">→ 末块转 decode</text>
    <line x1="40" y1="252" x2="760" y2="252" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M760 252 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="705" y="274" style="fill:var(--faint);font-size:11px">时间 →</text>
    <text x="120" y="240" class="mono" style="fill:var(--muted);font-size:10px">每个 chunk 之间，正在进行的 decode（蓝条）照常推进</text>
  </svg>
  <div class="figcap"><b>图 1 · 长 prefill 切块、与 decode 交错</b> — 不分块时一个 32k 的 prefill 独占一拍，其余请求的 decode <strong>全部 stall</strong>（红色虚线区）；分块后它被切成固定大小的小块，块与块之间插入正在进行的 decode（蓝条），时间线重新变得平滑。</div>
</div>

<h2>怎么切：把 32k 拆成多拍小块</h2>
<p>
分块的机制朴素而有效：<span class="inline">PrefillAdder</span> 每拍都带着一个固定的 chunk token 预算（<span class="mono">rem_chunk_tokens</span>，源自 <span class="mono">chunked_prefill_size</span>）。
为什么是“每拍都带着”？因为预算是<strong>每接一个请求就当场扣减</strong>的：这一拍可能先塞了几个短请求的整段 prefill，
剩下的预算才轮到队首那个长请求，于是它能切多大的一块，取决于<strong>这一拍前面已经被吃掉了多少</strong>。
当一个请求的待算 token 数 <span class="mono">input_tokens</span> <strong>小于等于</strong>剩余 chunk 预算时，整段一次塞完，走普通（非分块）路径；
一旦 <strong>超过</strong>，就进入分块分支：只截取 <span class="mono">trunc_len</span> 这么多 token（按 <span class="mono">page_size</span> 向下对齐，保证页整齐），
把这一段标记为本拍要算的范围（<span class="mono">set_extend_range</span>），其余留到后续。这个请求被记为 <span class="mono">new_chunked_req</span>，
下一拍它会带着已经填好的前缀回来，从断点处<strong>接着切下一块</strong>。如此往复，prompt 的 KV 缓存一块一块累积，
直到最后一块填完，它才脱离 prefill、转入 decode。这里有个容易忽略的细节：每一块都要<strong>按 page_size 向下对齐</strong>，
因为 KV 缓存是按页（page）管理的（第 4/6 课），切在半页中间会破坏分页结构，所以宁可少切一点、也要切在页边界上。
另一个细节是“前缀”的累积：第 2 拍开始，这个请求的 <span class="mono">prefix_indices</span> 已经包含了前面几块算好的 KV，
于是它每拍真正要算的只是“新的一块”，而不是从头重算——这和 RadixAttention 的前缀复用（第 7 课）是同一种“算过的不再算”的思想。
下面这张图把一个 32k 的 prompt 在连续几拍里的切块过程画了出来。
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>第 1 拍：chunk #1</h4><p>取队首的 32k prompt，<span class="mono">input_tokens &gt; rem_chunk_tokens</span> → 只截 <span class="mono">trunc_len</span>（如 8k，按 page_size 对齐）。填好前 8k 的 KV，混着别人的 decode 一起跑。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>第 2 拍：chunk #2</h4><p>同一请求作为 <span class="mono">new_chunked_req</span> 回来，从第 8k 处接着切下一个 8k 块。又是一个有界小步，decode 照常流动。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>第 3…N 拍：继续</h4><p>一块接一块把 KV 填满（第 4 课）。每一拍都不大，没有哪一步独占 GPU——尖峰被彻底摊平。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>最后一块：转 decode</h4><p>当剩余 token 一拍能装下，走普通路径整段塞完。此刻 prompt 全部 prefill 完毕，请求正式开始吐第一个 token。</p></div></div>
</div>

<h2>混合批：一拍里 prefill 与 decode 同框</h2>
<p>
分块之所以能保住别人的延迟，关键在<strong>混合批</strong>：被切出来的那一个 prefill chunk，并不是单独占一拍，而是和一堆正在 decode 的请求<strong>挤在同一个批</strong>里前向。
于是每一拍的工作量 = 一个有界的 prefill 块 + 全体在跑请求各自的一个 decode token，规模可控、节奏稳定。
这也回答了“为什么不直接让长请求排到最后再跑”——那样它会饿死、首字延迟无限长；而分块让它<strong>边切边混</strong>，
既稳步推进自己的 prefill，又不抢光别人的吞吐。这里还藏着一个调度上的巧思：混合批让 GPU 在同一次前向里<strong>既做计算密集的 prefill、又做访存密集的 decode</strong>，
两种负载的特性正好互补——prefill 吃算力、decode 吃显存带宽，混在一起反而能把 GPU 的算力和带宽都用得更满，而不是让某一种资源闲着。
这也是为什么 SGLang 默认就开启分块预填充：它几乎是“免费的午餐”，既削平了延迟尖峰，又顺手提高了硬件利用率。
下面这张图画的就是某一拍混合批的构成：高亮格是那个长请求的<strong>一个 prefill chunk</strong>，
其余普通格是各个请求的 <strong>decode</strong>。chunk 的大小（高亮格的“宽度”）由 <span class="mono">chunked_prefill_size</span> 决定，正是平滑与步数之间那把可调的旋钮。
</p>

<div class="cellgroup">
  <div class="cg-cap">某一拍的混合批：1 个 prefill chunk（高亮）+ 多个 decode 同批前向</div>
  <div class="cells">
    <span class="lab">成员</span><span class="cell hl">长请求 · chunk#k</span><span class="cell">R2 decode</span><span class="cell">R3 decode</span><span class="cell">R4 decode</span><span class="cell">R5 decode</span><span class="sep"></span><span class="q">混合批</span>
  </div>
  <div class="cells">
    <span class="lab">本拍干啥</span><span class="cell hl">填 8k KV</span><span class="cell">吐 1 token</span><span class="cell">吐 1 token</span><span class="cell">吐 1 token</span><span class="cell">吐 1 token</span><span class="sep"></span><span class="q">有界 + 流动</span>
  </div>
  <div class="cells">
    <span class="lab">下一拍</span><span class="cell hl">chunk#k+1</span><span class="cell">继续 decode</span><span class="cell">继续 decode</span><span class="cell">继续 decode</span><span class="cell">继续 decode</span><span class="sep"></span><span class="q">节奏稳定</span>
  </div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 280" role="img" aria-label="切块前后其他请求的逐 token 延迟对比：切块前出现一个尖锐的延迟尖峰，因为一个大 prefill 把这一拍撑爆；切块后每拍 prefill 有界，曲线平稳、尖峰被削平">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">TTFT/ITL 抖动：切块前后</text>
    <text x="20" y="58" style="fill:var(--red);font-size:12px;font-weight:700">❌ 切块前</text>
    <line x1="90" y1="130" x2="730" y2="130" style="stroke:var(--line);stroke-width:1.2"/>
    <polyline points="90,118 150,120 210,116 270,119 320,114 350,56 372,58 400,116 470,119 540,115 610,118 680,116 730,117" style="fill:none;stroke:var(--red);stroke-width:2"/>
    <text x="361" y="44" text-anchor="middle" style="fill:var(--red);font-size:11px;font-weight:700">大 prefill → 延迟尖峰</text>
    <text x="96" y="122" class="mono" style="fill:var(--muted);font-size:10px">ITL ↑</text>
    <text x="20" y="190" style="fill:var(--teal);font-size:12px;font-weight:700">✅ 切块后</text>
    <line x1="90" y1="250" x2="730" y2="250" style="stroke:var(--line);stroke-width:1.2"/>
    <polyline points="90,232 150,230 210,233 270,229 330,231 390,230 450,232 510,229 570,231 630,230 690,232 730,231" style="fill:none;stroke:var(--teal);stroke-width:2"/>
    <text x="400" y="212" text-anchor="middle" style="fill:var(--teal);font-size:11px;font-weight:700">分块把尖峰削平 · 平稳</text>
    <text x="96" y="242" class="mono" style="fill:var(--muted);font-size:10px">ITL ↑</text>
  </svg>
  <div class="figcap"><b>图 2 · TTFT/ITL 抖动：切块前后</b> — 切块前其他请求的逐 token 延迟出现<strong>尖峰</strong>（一个大 prefill 把这一拍撑爆，第 8 课的 ITL/TTFT 飙升）；切块后每拍 prefill 有界，尖峰被<strong>削平</strong>，延迟曲线保持平稳。</div>
</div>

<h2>取舍：chunk 大小是那把旋钮</h2>
<p>
分块不是免费的：它把一个 prefill 拆成多拍，<strong>总步数变多</strong>，那头鲸鱼自己的 TTFT 也被摊到好几拍、<strong>首字略慢</strong>。
但换来的是<strong>所有人</strong>的延迟都平滑、达标吞吐（goodput-under-SLA）更高——在长短请求混跑的真实负载里，这笔买卖几乎总是值的。
而 <span class="mono">chunked_prefill_size</span> 就是调这笔买卖的旋钮：<strong>切得越小</strong>，每拍 prefill 越轻、decode 越顺滑，但步数更多、长请求首字更慢、固定开销摊得更多；
<strong>切得越大</strong>，步数更少、长请求更快收尾，但每拍更重、尖峰风险回升（退回第 8 课的吞吐/延迟权衡）。
没有放之四海的最优值——它取决于你的延迟 SLA 和长短请求的比例。一个实用的心法是：<strong>先按你能接受的最坏单拍延迟反推 chunk 大小</strong>——
你希望每一拍的 decode 顿挫不超过多少毫秒，就把 prefill 块控制在对应的 token 量级；再根据长请求占比微调。
还要记得，chunk 太小也并非全是好处：步数变多意味着<strong>固定开销</strong>（每拍的组批、采样、内核启动等，第 21 课）被反复摊销，
极端情况下调度本身的开销会反过来吃掉一部分收益。所以这把旋钮的最佳点，往往落在“足够小到不制造尖峰、又足够大到不浪费步数”的中间地带。下面这张表把旋钮的两端效果摆在一起。
</p>

<table class="t">
  <tr><th>chunk 大小</th><th>每拍 prefill</th><th>decode 平滑度</th><th>总步数 / 长请求 TTFT</th><th>适用</th></tr>
  <tr><td class="mono">偏小</td><td>轻、有界</td><td><strong>很顺滑</strong>，尖峰小</td><td>步数多、长请求首字更慢</td><td>延迟敏感、decode 密集</td></tr>
  <tr><td class="mono">偏大</td><td>重、接近整段</td><td>较差，<strong>尖峰回升</strong></td><td>步数少、长请求更快收尾</td><td>吞吐优先、长请求为主</td></tr>
  <tr><td class="mono">关闭分块</td><td>整段一口吃</td><td><strong>最差</strong>，一拍 stall 所有人</td><td>步数最少、但延迟炸裂</td><td>几乎不推荐（短 prompt 才无所谓）</td></tr>
</table>

<h2>读源码：add_one_req 里的切块判断</h2>
<p>
分块的真身在 <span class="mono">schedule_policy.py</span> 的 <span class="inline">PrefillAdder.add_one_req</span>。
它先算出本请求要吃多少 token，做完 token/显存预算检查后，用一个 <span class="mono">if / else</span> 决定：能整段塞就整段，
否则进入 <strong>chunked prefill</strong> 分支，把 <span class="mono">trunc_len</span> 按 <span class="mono">page_size</span> 向下取整，
只提交这一块、并把请求记为 <span class="mono">new_chunked_req</span> 等下一拍续上。读这段源码时，请抓住三个关键点：
一是 <span class="mono">rem_chunk_tokens</span> 为 <span class="kw">None</span> 表示没开分块，此时整条 prompt 一拍算完；
二是分块分支里那行 <span class="mono">// page_size * page_size</span> 就是“向下对齐到页边界”的惯用写法，保证不切在半页中间；
三是 <span class="mono">new_chunked_req</span> 这个字段是“续上”的关键——调度器靠它在下一拍认出“这是上一拍没切完的那个请求”，从断点继续。下面截取最能说明问题的几行。
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::PrefillAdder.add_one_req</span><span class="ln">切块判断</span></div>
<pre><span class="cm"># 整段能装下(input_tokens ≤ 剩余 chunk 预算) → 一次塞完，不切</span>
<span class="kw">elif</span> self.rem_chunk_tokens <span class="kw">is</span> <span class="kw">None</span> <span class="kw">or</span> input_tokens &lt;= self.rem_chunk_tokens:
    req.set_extend_range(<span class="cm"># 提交整条 prompt 这一拍算完</span>
        <span class="kw">len</span>(req.prefix_indices), <span class="kw">len</span>(req.full_untruncated_fill_ids))
    self.can_run_list.append(req)
<span class="kw">else</span>:                                            <span class="cm"># 装不下 → 分块预填充</span>
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size  <span class="cm"># 按页对齐截断</span>
    <span class="kw">if</span> trunc_len &lt;= 0:
        <span class="kw">return</span> AddReqResult.OTHER
    req.set_extend_range(<span class="cm"># 只算前 trunc_len 个 token，其余下一拍续</span>
        <span class="kw">len</span>(req.prefix_indices), <span class="kw">len</span>(req.prefix_indices) + trunc_len)
    self.can_run_list.append(req)
    self.new_chunked_req = req                    <span class="cm"># 记住它，下一拍接着切下一块</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.get_new_batch_prefill</span><span class="ln">在 token 预算内组 prefill 批：长 prompt 分块、不挤占 decode</span></div>
<pre><span class="kw">def</span> get_new_batch_prefill(self):
    <span class="cm"># build the next PREFILL batch within a token budget. A long prompt is</span>
    <span class="cm"># admitted in CHUNKS (chunked_prefill_size) so it doesn't monopolize one</span>
    <span class="cm"># step and stall ongoing decode; the rest of its prefill resumes later.</span>
    ...
    <span class="kw">return</span> new_batch   <span class="cm"># Optional[ScheduleBatch]</span></pre>
</div>

<p>
一个具体例子：以 <span class="mono">--chunked-prefill-size 4096</span> 为例（实际默认值随 GPU 显存自适应，常见 2048 / 4096 / 8192），一个 32768 token 的长 prompt 会被切成 <strong>8 块</strong>（32768 ÷ 4096 = 8 拍），每拍只算 4096 个 prefill token，再混入其余请求的 decode；若把旋钮拧到 <span class="mono">--chunked-prefill-size 2048</span>，同一个 prompt 就切成 <strong>16 块</strong>——更顺滑，但多花一倍步数、长请求首字更慢。
</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>要解决的问题</strong>：一个超长 prompt（如 32k）若在<strong>一拍里整段 prefill</strong>，会独占 GPU、让批里所有 decode 请求 <strong>stall</strong>，造成 TTFT/ITL 延迟尖峰（第 8 课）。一头鲸鱼堵死整池。</li>
    <li><strong>分块的做法</strong>：把大 prefill 切成<strong>固定大小的 token 块</strong>（由 <span class="mono">chunked_prefill_size</span> 设定），分摊到<strong>多拍</strong>，并和别人的 decode <strong>混在同一个批</strong>里跑——没有哪一拍巨大，所有人的 token 持续流动。</li>
    <li><strong>谁来切</strong>：<span class="inline">PrefillAdder</span>（第 20 课）在 <span class="mono">add_one_req</span> 里执行每拍 token 预算，装不下就把请求<strong>钳到剩余预算</strong>（<span class="mono">trunc_len</span> 按 page 对齐），记为 <span class="mono">new_chunked_req</span> 下一拍续；KV 一块块填（第 4 课），最后一块填完才转 decode。</li>
    <li><strong>取舍与旋钮</strong>：分块<strong>多花几拍</strong>、长请求自己的 TTFT 被摊薄，但换来<strong>全局延迟平滑</strong>与更高的达标吞吐。<span class="mono">chunked_prefill_size</span> 越小越顺滑但步数多，越大步数少但尖峰回升（第 8 课的权衡）。</li>
    <li><strong>一句话收尾</strong>：别让一个长请求把一拍撑爆——把它切块、混着 decode 慢慢喂；用多花的几拍，换满桌人的顺滑。这也是后续 PD 分离（第 45 课）把 prefill 与 decode 彻底拆到不同实例的动机起点。</li>
  </ul>
</div>
""",
             "en": r"""
<p class="lead">
Last lesson we saw the scheduler's two-step move: order first, then throttle. This lesson zooms in on the trickiest throttling case: what happens when the head of the queue is a <strong>huge prompt</strong> (say a 32k-token document)?
If you let it <strong>prefill entirely in ONE step</strong>, that step becomes a "whale"—it monopolizes the whole GPU for tens of milliseconds, and every other request that is <span class="inline">decoding</span> behind it <strong>stalls completely</strong>: nobody's tokens come out, and latency spikes (TTFT/ITL blow up, Lesson 8).
<strong>Chunked prefill</strong> cuts that whale into <strong>fixed-size token chunks</strong>, spread over several steps, each step nibbling one small bite while <strong>mixed in the same batch</strong> with everyone else's decode. No step is enormous; everyone's tokens keep flowing.
Remember one line: <strong>never let one long request blow up a step—chunk it and feed it slowly, mixed with decode</strong>.
</p>

<div class="card analogy">
  <div class="tag">🔌 Real-world analogy</div>
  Picture this step's GPU as a <strong>lively dinner table</strong>. Everyone takes turns talking (each request's decode emits a token in turn), conversation flowing.
  Then a <strong>giant meal</strong> arrives (a 32k-token prompt). The dumb move: have one person <strong>stuff the whole plate in their mouth at once</strong>—now they chew head-down for tens of seconds, during which <strong>nobody can get a word in</strong>; one person choked the whole table's chat.
  The smart move is to <strong>eat the giant meal in small bites while still holding the conversation</strong>: each round you swallow just <strong>one bite (one chunk)</strong>, chew, say a line, then take the next bite. The meal still gets finished, and the table's chat (everyone else's decode) <strong>never stops</strong>.
  The key discipline: <strong>each bite has a fixed size cap</strong>—nobody is ever allowed to cram their mouth full and block everyone's turn.
  The cost is real too: the big meal takes several rounds to finish (a few extra steps, and its own first byte is slower), but in exchange <strong>the whole table stays smooth</strong>—nobody gets choked off.
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  In one line: <strong>split a big prefill into many small per-step chunks, mix them with decode, and trade a few extra steps for globally smooth latency</strong>.
  Who does the cutting? The very <span class="inline">PrefillAdder</span> from last lesson (Lesson 20). It holds a <strong>token budget</strong> (the server arg <span class="mono">chunked_prefill_size</span> sets how many prefill tokens fit per step) and, inside <span class="mono">add_one_req</span>, tries each request: if the whole thing fits, commit it; <strong>if it doesn't, slice off only the part that fits</strong> and leave the rest for the next step.
  So a 32k prompt gets cut into several chunks across consecutive steps—its KV cache (Lesson 4) <strong>fills chunk by chunk</strong>, and only after the <strong>last chunk</strong> does that request switch into decode and emit its first token. And every such step runs a <strong>mixed batch</strong>: some requests prefilling a chunk, others decoding, with the event loop (Lesson 18) advancing as usual.
  Pull back: chunked prefill and last lesson's throttling are two sides of one coin—throttling guarantees "don't eat too much in one step," chunking guarantees "what doesn't fit isn't choked down whole either; slice it and take it slow."
  It tames an extreme that would otherwise spike latency into a stream of steady small steps—the key to SGLang staying smooth under <strong>mixed long/short traffic</strong>.
</div>

<h2>The problem: one whale clogs the whole pool</h2>
<p>
First, what happens without chunking. Continuous batching (Lesson 5) rests on a happy premise: "every step is fast, and the batch advances together."
But the moment a 32k-token prefill jumps to the head and you make it <strong>finish in one step</strong>, that step's forward compute explodes tens of times over—
the GPU is monopolized for tens of milliseconds, during which <strong>every other decode request in the batch just waits</strong>: they should emit one token per step, but now they stall alongside the whale.
To the user this reads as a streaming answer that <strong>suddenly freezes</strong> (ITL spike), plus longer first-token latency for new requests (TTFT spike, Lesson 8).
One overlong request ruined an entire batch's experience. This is not a "won't fit in memory" problem—KV slots may be plenty; it is a <strong>"too much compute in one step"</strong> problem,
congestion along the time axis. Chunked prefill exists precisely to flatten this "one big fish stops the whole pool" spike.
</p>

<div class="cols">
  <div class="col"><h4>❌ One-step full prefill (spike)</h4><p>32k tokens finish in a <strong>single step</strong>: that step is enormous, monopolizing the GPU for tens of ms. Every other request's decode in the batch <strong>stalls</strong>, no tokens emitted. TTFT/ITL <strong>spike instantly</strong> (Lesson 8); users see output freeze. One whale clogs the pool.</p></div>
  <div class="col"><h4>✅ Chunked prefill (smooth)</h4><p>The 32k is cut into <strong>fixed-size chunks</strong> over several steps. Each step computes one chunk + is <strong>mixed with everyone's decode</strong>; no step is huge. Everyone's tokens <strong>keep flowing</strong>, latency steady. A few extra steps buy <strong>global smoothness</strong> and better goodput-under-SLA.</p></div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Chunk a long prefill and interleave with decode: without chunking one huge 32k prefill hogs a step and every other request's decode stalls; with chunking the long prefill is split into fixed-size chunks, with ongoing decode slipped between chunks, and the timeline turns smooth again">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">Chunk a long prefill, interleave with decode</text>
    <text x="20" y="86" style="fill:var(--red);font-size:12px;font-weight:700">❌ no chunking</text>
    <rect x="110" y="60" width="280" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="250" y="87" text-anchor="middle" style="font-size:12px;font-weight:700">32k prefill · hogs a step</text>
    <rect x="400" y="60" width="350" height="44" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 3"/>
    <text x="575" y="87" text-anchor="middle" style="fill:var(--red);font-size:12px;font-weight:700">every other decode stalls</text>
    <text x="20" y="192" style="fill:var(--teal);font-size:12px;font-weight:700">✅ chunked</text>
    <rect x="110" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="142" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#1</text>
    <rect x="182" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="196" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="210" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="228" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="260" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#2</text>
    <rect x="300" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="314" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="328" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="346" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="378" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#3</text>
    <rect x="418" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="432" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="446" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="464" y="166" width="64" height="44" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="496" y="193" text-anchor="middle" class="mono" style="font-size:11px;font-weight:700">chunk#N</text>
    <rect x="536" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="550" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <rect x="564" y="174" width="10" height="28" rx="2" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.2"/>
    <text x="600" y="193" style="fill:var(--teal);font-size:11px;font-weight:700">→ last chunk enters decode</text>
    <line x1="40" y1="252" x2="760" y2="252" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M760 252 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="700" y="274" style="fill:var(--faint);font-size:11px">time →</text>
    <text x="120" y="240" class="mono" style="fill:var(--muted);font-size:10px">between chunks, ongoing decode (blue bars) keeps advancing</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Chunk a long prefill, interleave with decode</b> — without chunking one 32k prefill hogs a step and every other request's decode <strong>stalls</strong> (red dashed region); with chunking it is split into fixed-size chunks, with ongoing decode (blue bars) slipped between them, and the timeline turns smooth again.</div>
</div>

<h2>How to cut: split 32k into multi-step chunks</h2>
<p>
The mechanism is plain and effective: each step, <span class="inline">PrefillAdder</span> carries a fixed chunk-token budget (<span class="mono">rem_chunk_tokens</span>, derived from <span class="mono">chunked_prefill_size</span>).
When a request's pending <span class="mono">input_tokens</span> is <strong>less than or equal to</strong> the remaining chunk budget, the whole thing is committed at once via the non-chunked path;
once it <strong>exceeds</strong> it, the chunked branch kicks in: take only <span class="mono">trunc_len</span> tokens (floored to <span class="mono">page_size</span> for page alignment),
mark that slice as this step's extend range (<span class="mono">set_extend_range</span>), and leave the rest. The request is recorded as <span class="mono">new_chunked_req</span>,
and next step it returns with its filled prefix and <strong>slices the next chunk from the cut point</strong>. Round after round, the prompt's KV cache (Lesson 4) accumulates chunk by chunk,
until the last chunk is filled and it finally leaves prefill and enters decode. The diagram below traces a 32k prompt being chunked across consecutive steps.
</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Step 1: chunk #1</h4><p>Take the 32k prompt at the head; <span class="mono">input_tokens &gt; rem_chunk_tokens</span> → slice only <span class="mono">trunc_len</span> (e.g. 8k, page-aligned). Fill the first 8k of KV, running mixed with others' decode.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Step 2: chunk #2</h4><p>The same request returns as <span class="mono">new_chunked_req</span>, slicing the next 8k from offset 8k. Another bounded small step; decode keeps flowing.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Steps 3…N: continue</h4><p>Chunk after chunk fills the KV (Lesson 4). No step is large, none monopolizes the GPU—the spike is fully flattened.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Last chunk: enter decode</h4><p>When the remainder fits in one step, the non-chunked path commits it whole. The prompt is now fully prefilled, and the request emits its first token.</p></div></div>
</div>

<h2>Mixed batch: prefill and decode in the same step</h2>
<p>
The reason chunking protects others' latency is the <strong>mixed batch</strong>: the prefill chunk that was sliced off does not get a step to itself; it shares <strong>one batch</strong> with a crowd of decoding requests in the same forward pass.
So each step's work = one bounded prefill chunk + one decode token from every running request—controlled in size, steady in rhythm.
This also answers "why not just queue the long request last"—that would starve it, with unbounded first-token latency; chunking lets it <strong>slice-and-mix</strong>,
making steady prefill progress without stealing everyone's throughput. The diagram below shows the makeup of one mixed-batch step: the highlighted cell is the long request's <strong>one prefill chunk</strong>,
the rest are each request's <strong>decode</strong>. The chunk's size (the highlighted cell's "width") is set by <span class="mono">chunked_prefill_size</span>—the very knob between smoothness and step count.
</p>

<div class="cellgroup">
  <div class="cg-cap">One step's mixed batch: 1 prefill chunk (highlighted) + several decodes in the same forward</div>
  <div class="cells">
    <span class="lab">members</span><span class="cell hl">long req · chunk#k</span><span class="cell">R2 decode</span><span class="cell">R3 decode</span><span class="cell">R4 decode</span><span class="cell">R5 decode</span><span class="sep"></span><span class="q">mixed batch</span>
  </div>
  <div class="cells">
    <span class="lab">this step</span><span class="cell hl">fill 8k KV</span><span class="cell">emit 1 tok</span><span class="cell">emit 1 tok</span><span class="cell">emit 1 tok</span><span class="cell">emit 1 tok</span><span class="sep"></span><span class="q">bounded + flowing</span>
  </div>
  <div class="cells">
    <span class="lab">next step</span><span class="cell hl">chunk#k+1</span><span class="cell">keep decoding</span><span class="cell">keep decoding</span><span class="cell">keep decoding</span><span class="cell">keep decoding</span><span class="sep"></span><span class="q">steady rhythm</span>
  </div>
</div>

<div class="fig">
  <svg viewBox="0 0 760 280" role="img" aria-label="Other requests' inter-token latency before vs after chunking: before chunking a sharp latency spike appears because one big prefill blows up the step; after chunking each step's prefill is bounded, the curve stays smooth and the spike is capped">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">TTFT/ITL jitter: before vs after chunking</text>
    <text x="20" y="58" style="fill:var(--red);font-size:12px;font-weight:700">❌ before chunking</text>
    <line x1="90" y1="130" x2="730" y2="130" style="stroke:var(--line);stroke-width:1.2"/>
    <polyline points="90,118 150,120 210,116 270,119 320,114 350,56 372,58 400,116 470,119 540,115 610,118 680,116 730,117" style="fill:none;stroke:var(--red);stroke-width:2"/>
    <text x="361" y="44" text-anchor="middle" style="fill:var(--red);font-size:11px;font-weight:700">big prefill → latency spike</text>
    <text x="96" y="122" class="mono" style="fill:var(--muted);font-size:10px">ITL ↑</text>
    <text x="20" y="190" style="fill:var(--teal);font-size:12px;font-weight:700">✅ after chunking</text>
    <line x1="90" y1="250" x2="730" y2="250" style="stroke:var(--line);stroke-width:1.2"/>
    <polyline points="90,232 150,230 210,233 270,229 330,231 390,230 450,232 510,229 570,231 630,230 690,232 730,231" style="fill:none;stroke:var(--teal);stroke-width:2"/>
    <text x="400" y="212" text-anchor="middle" style="fill:var(--teal);font-size:11px;font-weight:700">chunking caps the spike · smooth</text>
    <text x="96" y="242" class="mono" style="fill:var(--muted);font-size:10px">ITL ↑</text>
  </svg>
  <div class="figcap"><b>Fig 2 · TTFT/ITL jitter: before vs after chunking</b> — before chunking other requests' inter-token latency <strong>spikes</strong> (one big prefill blows up the step, the ITL/TTFT spike of Lesson 8); after chunking each step's prefill is bounded, the spike is <strong>capped</strong>, and the curve stays smooth.</div>
</div>

<h2>The trade-off: chunk size is the knob</h2>
<p>
Chunking isn't free: it splits a prefill across steps, so <strong>total step count rises</strong>, and the whale's own TTFT is spread over several steps—its <strong>first byte is a bit slower</strong>.
What you buy is smooth latency for <strong>everyone</strong> and higher goodput-under-SLA—in real mixed long/short workloads this is almost always worth it.
And <span class="mono">chunked_prefill_size</span> is the knob: <strong>smaller</strong> means lighter prefill per step and smoother decode, but more steps, a slower whale first-byte, and more fixed overhead amortized;
<strong>larger</strong> means fewer steps and a faster whale finish, but heavier steps and the spike risk creeping back (back to Lesson 8's throughput/latency trade-off).
There's no universal optimum—it depends on your latency SLA and your long/short mix. The table lays the two ends side by side.
</p>

<table class="t">
  <tr><th>chunk size</th><th>prefill per step</th><th>decode smoothness</th><th>total steps / whale TTFT</th><th>fits</th></tr>
  <tr><td class="mono">smaller</td><td>light, bounded</td><td><strong>very smooth</strong>, tiny spikes</td><td>more steps, slower whale first byte</td><td>latency-sensitive, decode-heavy</td></tr>
  <tr><td class="mono">larger</td><td>heavy, near-whole</td><td>worse, <strong>spike returns</strong></td><td>fewer steps, faster whale finish</td><td>throughput-first, long-prompt-heavy</td></tr>
  <tr><td class="mono">chunking off</td><td>whole in one bite</td><td><strong>worst</strong>, one step stalls all</td><td>fewest steps, but latency blows up</td><td>rarely advised (fine only for short prompts)</td></tr>
</table>

<h2>Read the source: the chunk decision in add_one_req</h2>
<p>
Chunking lives in <span class="mono">schedule_policy.py</span>'s <span class="inline">PrefillAdder.add_one_req</span>.
It first computes how many tokens the request needs, runs the token/memory budget checks, then an <span class="mono">if / else</span> decides: commit the whole thing if it fits,
otherwise take the <strong>chunked prefill</strong> branch—floor <span class="mono">trunc_len</span> to <span class="mono">page_size</span>,
commit only that slice, and record the request as <span class="mono">new_chunked_req</span> to continue next step. The most telling lines follow.
</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/schedule_policy.py ::PrefillAdder.add_one_req</span><span class="ln">chunk decision</span></div>
<pre><span class="cm"># whole fits (input_tokens ≤ remaining chunk budget) → commit at once, no chunking</span>
<span class="kw">elif</span> self.rem_chunk_tokens <span class="kw">is</span> <span class="kw">None</span> <span class="kw">or</span> input_tokens &lt;= self.rem_chunk_tokens:
    req.set_extend_range(<span class="cm"># commit the whole prompt this step</span>
        <span class="kw">len</span>(req.prefix_indices), <span class="kw">len</span>(req.full_untruncated_fill_ids))
    self.can_run_list.append(req)
<span class="kw">else</span>:                                            <span class="cm"># doesn't fit → chunked prefill</span>
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size  <span class="cm"># page-aligned cut</span>
    <span class="kw">if</span> trunc_len &lt;= 0:
        <span class="kw">return</span> AddReqResult.OTHER
    req.set_extend_range(<span class="cm"># compute only the first trunc_len tokens; rest next step</span>
        <span class="kw">len</span>(req.prefix_indices), <span class="kw">len</span>(req.prefix_indices) + trunc_len)
    self.can_run_list.append(req)
    self.new_chunked_req = req                    <span class="cm"># remember it; slice the next chunk next step</span></pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/scheduler.py ::Scheduler.get_new_batch_prefill</span><span class="ln">build a prefill batch within a token budget: chunk long prompts, don't starve decode</span></div>
<pre><span class="kw">def</span> get_new_batch_prefill(self):
    <span class="cm"># build the next PREFILL batch within a token budget. A long prompt is</span>
    <span class="cm"># admitted in CHUNKS (chunked_prefill_size) so it doesn't monopolize one</span>
    <span class="cm"># step and stall ongoing decode; the rest of its prefill resumes later.</span>
    ...
    <span class="kw">return</span> new_batch   <span class="cm"># Optional[ScheduleBatch]</span></pre>
</div>

<p>
A concrete example: take <span class="mono">--chunked-prefill-size 4096</span> (the real default is GPU-memory-adaptive — commonly 2048 / 4096 / 8192), a 32768-token prompt is cut into <strong>8 chunks</strong> (32768 ÷ 4096 = 8 steps), each step computing only 4096 prefill tokens mixed with other requests' decode; turn the knob to <span class="mono">--chunked-prefill-size 2048</span> and the same prompt becomes <strong>16 chunks</strong>—smoother, but twice the steps and a slower whale first byte.
</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>The problem</strong>: a huge prompt (e.g. 32k) prefilled <strong>whole in one step</strong> monopolizes the GPU and <strong>stalls every decode</strong> in the batch, causing TTFT/ITL latency spikes (Lesson 8). One whale clogs the pool.</li>
    <li><strong>The fix</strong>: split the big prefill into <strong>fixed-size token chunks</strong> (set by <span class="mono">chunked_prefill_size</span>) across <strong>several steps</strong>, <strong>mixed in the same batch</strong> with others' decode—no step is huge, everyone's tokens keep flowing.</li>
    <li><strong>Who cuts</strong>: <span class="inline">PrefillAdder</span> (Lesson 20) enforces the per-step token budget in <span class="mono">add_one_req</span>, clamping a non-fitting request to the remaining budget (<span class="mono">trunc_len</span>, page-aligned) and recording <span class="mono">new_chunked_req</span> for next step; KV fills chunk by chunk (Lesson 4), and only after the last chunk does it enter decode.</li>
    <li><strong>Trade-off &amp; knob</strong>: chunking costs <strong>more steps</strong> and a thinner whale TTFT, buying <strong>smooth global latency</strong> and higher goodput-under-SLA. <span class="mono">chunked_prefill_size</span> smaller = smoother but more steps; larger = fewer steps but the spike returns (Lesson 8's trade-off).</li>
    <li><strong>One-line close</strong>: never let one long request blow up a step—chunk it and feed it slowly, mixed with decode; spend a few extra steps to keep the whole table smooth. This is also the seed of the motivation for PD disaggregation (Lesson 45), which splits prefill and decode onto separate instances entirely.</li>
  </ul>
</div>
"""}

LESSON_23 = {"zh": r"""
<p class="lead">
前面五课，我们一直盯着<strong>一个</strong>调度器进程在<strong>一张</strong>卡上的心跳：收请求、组批、前向、收尾。但真实部署里，一台机器常有 8 张卡，一个集群有上百张卡——
模型可能<strong>大到一张卡装不下</strong>，流量也可能<strong>大到一个副本喂不饱</strong>。这一课收尾 Part 5，我们把镜头拉到<strong>多进程、多卡</strong>的尺度，看两种改变<strong>控制流</strong>的并行：
<strong>数据并行 DP</strong>（把整个运行时复制 N 份、把请求分发出去）和<strong>流水线并行 PP</strong>（把模型的层切成几段、让多个 micro-batch 在段间同时流动）。
注意：它们改变的不是矩阵怎么算，而是<strong>谁来分发、有几个 micro-batch 在飞</strong>——这正是调度器要操心的事，也是为什么这两件事会和事件循环、调度策略一起放进 Part 5 来讲。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 <strong>DP</strong> 想成超市里<strong>多条一模一样的收银通道</strong>：每条通道都<strong>设备齐全</strong>（自带收银员、扫码枪、钱箱），门口一个<strong>迎宾员</strong>把每位顾客指向某条通道；
  通道之间互不干扰，开得越多、单位时间结的账越多。把 <strong>PP</strong> 想成<strong>一条带工位的长传送带</strong>：每个工位只做一道工序（贴标→装箱→封口），
  多件商品<strong>同时</strong>骑在带上、各处在不同工位，于是没有哪个工位闲着。DP 是"复制整条通道扩吞吐"，PP 是"把一件活拆成接力让大家都忙起来"——一个管<strong>分发</strong>，一个管<strong>接力</strong>。
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  一句话：<strong>DP 复制"整个模型"、把请求扇出到多个独立副本；PP 切分"模型的层"、让多个 micro-batch 在段间流水。</strong>
  DP 的指挥者是 <span class="inline">DataParallelController</span>——它本身<strong>不做前向</strong>，只负责把进来的请求<strong>轮询或按负载</strong>分给 N 个<strong>完整运行时</strong>（每个副本都有自己的调度器 + TP worker + KV 缓存 + 一整份模型权重）。
  PP 的执行者是 <span class="inline">SchedulerPPMixin.event_loop_pp</span>——它把第 18 课那条单段心跳改造成<strong>流水线循环</strong>，让 stage1 处理 micro-batch B 的同时 stage2 在处理 micro-batch A，
  从而把流水线<strong>填充/排空的"气泡"</strong>摊薄。再叠加<strong>张量并行 TP</strong>（切每一层的矩阵，第 46 课深入），大部署就是 <strong>TP × PP × DP</strong> 三维拼起来的。
</div>

<h2>数据并行 DP：复制整个运行时，把请求扇出</h2>
<p>DP 的思路最朴素：<strong>一个副本喂不饱流量，那就开 N 个副本</strong>。但关键在于——每个副本都是一台<strong>完整的引擎</strong>，
不是共享某块权重，而是<strong>各自持有一整份模型副本</strong>、各自跑一条第 18 课讲的事件循环、各自管自己的 KV 账本。
副本之间<strong>请求级隔离</strong>：顾客 A 在副本 1、顾客 B 在副本 2，两者从不交互。唯一需要的协调，就是门口那个<strong>分发器</strong>——
<span class="inline">DataParallelController</span>。它在最前面接住所有进来的请求，按策略（轮询 <span class="mono">round_robin_scheduler</span>、
或按在跑请求数/token 数的负载感知）挑一个副本，把请求<strong>原样转发</strong>过去，自己绝不碰 GPU。</p>

<div class="flow">
  <div class="node hl"><div class="nt">DataParallelController</div><div class="nd">轮询 / 负载感知<br>round_robin_scheduler</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">副本 1（完整运行时）</div><div class="nd">调度器+TP+KV<br>一整份模型</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">副本 2（完整运行时）</div><div class="nd">调度器+TP+KV<br>一整份模型</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">副本 N（完整运行时）</div><div class="nd">调度器+TP+KV<br>一整份模型</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="数据并行 DP：请求流 r0/r1/r2/r3 进入 DataParallelController，被轮询 0→1→2→0 扇出到三个完整副本，副本 0/1/2 各自持有不同请求">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">请求流（轮询 0→1→2→0）</text>
    <rect x="20" y="78" width="52" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="46" y="96" text-anchor="middle" class="mono" style="font-size:11px">r0</text>
    <rect x="20" y="110" width="52" height="26" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="46" y="128" text-anchor="middle" class="mono" style="font-size:11px">r1</text>
    <rect x="20" y="142" width="52" height="26" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="46" y="160" text-anchor="middle" class="mono" style="font-size:11px">r2</text>
    <rect x="20" y="174" width="52" height="26" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="46" y="192" text-anchor="middle" class="mono" style="font-size:11px">r3</text>
    <line x1="72" y1="135" x2="208" y2="135" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="214,135 204,130 204,140" style="fill:var(--line)"/>
    <rect x="214" y="101" width="168" height="68" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="298" y="128" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">DataParallelController</text>
    <text x="298" y="150" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--accent-ink)">round_robin · 不碰 GPU</text>
    <line x1="382" y1="120" x2="552" y2="52" style="stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="558,50 547,49 550,60" style="fill:var(--blue)"/>
    <line x1="382" y1="135" x2="552" y2="148" style="stroke:var(--amber);stroke-width:1.5"/>
    <polygon points="558,149 547,143 548,154" style="fill:var(--amber)"/>
    <line x1="382" y1="150" x2="552" y2="244" style="stroke:var(--purple);stroke-width:1.5"/>
    <polygon points="558,246 547,238 550,249" style="fill:var(--purple)"/>
    <rect x="558" y="20" width="200" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--blue);stroke-width:1.5"/>
    <text x="568" y="42" style="font-weight:700">副本 0（完整运行时）</text>
    <text x="568" y="60" class="mono" style="font-size:10.5px;fill:var(--muted)">调度器+TP+KV · 一整份模型</text>
    <rect x="568" y="66" width="58" height="14" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <text x="597" y="77" text-anchor="middle" class="mono" style="font-size:9.5px">r0 · r3</text>
    <rect x="558" y="116" width="200" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--amber);stroke-width:1.5"/>
    <text x="568" y="138" style="font-weight:700">副本 1（完整运行时）</text>
    <text x="568" y="156" class="mono" style="font-size:10.5px;fill:var(--muted)">调度器+TP+KV · 一整份模型</text>
    <rect x="568" y="162" width="40" height="14" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <text x="588" y="173" text-anchor="middle" class="mono" style="font-size:9.5px">r1</text>
    <rect x="558" y="212" width="200" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--purple);stroke-width:1.5"/>
    <text x="568" y="234" style="font-weight:700">副本 2（完整运行时）</text>
    <text x="568" y="252" class="mono" style="font-size:10.5px;fill:var(--muted)">调度器+TP+KV · 一整份模型</text>
    <rect x="568" y="258" width="40" height="14" rx="3" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <text x="588" y="269" text-anchor="middle" class="mono" style="font-size:9.5px">r2</text>
  </svg>
  <div class="figcap"><b>图 23A · DP：请求按副本分流</b> — 请求流进入只做分发的 <span class="mono">DataParallelController</span>，按 <span class="mono">round_robin</span> 顺序 <span class="mono">0→1→2→0</span> 扇出到 N 个<strong>完整副本</strong>（每个都自带调度器+TP+KV+一整份模型）；副本 0/1/2 各自持有不同请求（r3 回绕落回副本 0），控制器全程不碰 GPU。</div>
</div>

<p>举两个具体数字落地。<strong><span class="mono">--dp-size 4</span></strong> 表示开 4 个完整副本（副本 0/1/2/3），控制器把请求按 <span class="mono">0→1→2→3→0</span> 的顺序轮流投递、周而复始；
而 <strong><span class="mono">--pp-size 2</span></strong> 则把模型的层切成 2 段、分到 2 张卡上接力。前者扩吞吐、后者扩容量。</p>

<p>这张图要读出三层意思。其一，<strong>控制器在所有副本之前</strong>，是请求进入系统的第一道岔路口，它只做"分给谁"这一个决定。
其二，每个副本都是<strong>自包含</strong>的——它内部就是你前五课学到的全部：一条 <span class="inline">event_loop_normal</span>、一套 prefill/decode 取舍、一本独立 KV 账本。
其三，DP 是<strong>吞吐的乘法器</strong>：N 个副本理论上把整机吞吐放大 N 倍，因为请求被均摊、彼此不抢资源。
要提醒的是，这里讲的是<strong>请求级 DP</strong>；在 MoE 模型里还有更精细的<strong>注意力 DP / 专家 DP</strong>，控制流与通信更复杂，留到第 46/47 课展开，本课不涉及。</p>

<p>再把"完整运行时"这四个字掰开说清楚，因为这是 DP 最容易被误解的地方。很多人初看 DP，会以为是"几张卡分着算同一个请求"——那是 TP，不是 DP。
DP 的副本之间<strong>没有任何张量级的协同</strong>：副本一的 GPU 在算顾客甲的注意力时，副本二的 GPU 完全不知道、也不需要知道顾客甲的存在。
每个副本从权重加载、KV 池初始化、到事件循环启动，都是<strong>独立完成</strong>的，就像把同一台引擎<strong>原封不动地复印</strong>了 N 台。
正因如此，DP 的扩展几乎没有协同开销——加一个副本，就是多开一台完整引擎、在控制器的轮询表里多登记一个 worker，仅此而已。
这种"近乎线性"的扩展性，正是大规模在线服务最看重 DP 的原因：只要单副本能装下模型、流量还在涨，<strong>加副本就能近线性加吞吐</strong>。</p>

<p>顺带说清"负载感知"分发和"轮询"分发的差别，这在生产里很关键。纯轮询假设每条请求<strong>大小相近</strong>，于是雨露均沾、一人一条最公平；
但真实流量里，请求长短悬殊——有人只问一句话，有人贴进来一篇长文档。若还死板地轮询，可能把一串长请求全堆到同一个副本，让它<strong>排长队</strong>，
而隔壁副本早已闲得发慌。负载感知策略（按各副本<strong>在跑请求数</strong>或<strong>未完成 token 数</strong>挑最闲的那个）就是为此而生：它让分发<strong>跟着真实负载走</strong>，
把新请求送到此刻最空的副本，从而把整机的排队延迟压平。代价只是控制器要多记一本<strong>各副本负载的小账</strong>——相比 GPU 前向，这点开销可以忽略不计。</p>

<h2>流水线并行 PP：切层成段，让多个 micro-batch 在飞</h2>
<p>当模型<strong>大到一张卡装不下</strong>时，DP 复制就行不通了——你连一份都放不下，谈何复制 N 份。这时换 PP：把模型的<strong>层</strong>纵向切成几<strong>段（stage）</strong>，
第 1～10 层在 GPU0、第 11～20 层在 GPU1、第 21～30 层在 GPU2……每张卡只持有<strong>一段</strong>层。一个 batch 的前向就变成<strong>接力</strong>：
数据先过 stage1、把激活值传给 stage2、再传 stage3……问题是，若只有一个 batch 在跑，stage2 工作时 stage1/stage3 都<strong>闲着等</strong>，
GPU 利用率惨不忍睹——这就是<strong>流水线气泡（pipeline bubble）</strong>。解法是把一个大 batch 拆成多个 <strong>micro-batch</strong>，让它们<strong>错峰</strong>同时在管线里跑。</p>

<div class="vflow">
  <div class="step"><div class="num">t1</div><div class="sc"><h4>填充期</h4><p>micro-batch A 进 <strong>stage1</strong>；stage2、stage3 还空着——这一拍是气泡的一部分，只有一段在忙。</p></div></div>
  <div class="step"><div class="num">t2</div><div class="sc"><h4>渐满</h4><p>A 推进到 <strong>stage2</strong>，同时 micro-batch B 进 <strong>stage1</strong>；两段在忙，管线开始填满。</p></div></div>
  <div class="step"><div class="num">t3</div><div class="sc"><h4>满载（稳态）</h4><p>A 在 <strong>stage3</strong>、B 在 <strong>stage2</strong>、C 进 <strong>stage1</strong>——<strong>三段全忙</strong>，气泡被摊掉，这是 PP 想长期停留的状态。</p></div></div>
  <div class="step"><div class="num">t4</div><div class="sc"><h4>排空期</h4><p>新 micro-batch 发完后，A 已产出、B/C 陆续走完末段；尾部又出现气泡。<strong>micro-batch 越多，满载占比越高</strong>，气泡越小。</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="流水线并行 PP：模型层分成 S0/S1/S2 三段放在三张卡，micro-batch mb0/mb1/mb2 沿对角线错峰流动，两头是填充与排空气泡、中间是三段全忙的稳态">
    <text x="24" y="40" style="font-weight:700;fill:var(--muted)">阶段（每段一张卡，各持一段层）</text>
    <text x="170" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t0</text>
    <text x="274" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t1</text>
    <text x="378" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t2</text>
    <text x="482" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t3</text>
    <text x="586" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t4</text>
    <rect x="24" y="78" width="86" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="67" y="100" text-anchor="middle" style="font-weight:700">S0 · GPU0</text>
    <text x="67" y="118" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">层 1–10</text>
    <rect x="24" y="146" width="86" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="67" y="168" text-anchor="middle" style="font-weight:700">S1 · GPU1</text>
    <text x="67" y="186" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">层 11–20</text>
    <rect x="24" y="214" width="86" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="67" y="236" text-anchor="middle" style="font-weight:700">S2 · GPU2</text>
    <text x="67" y="254" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">层 21–30</text>
    <rect x="120" y="78" width="100" height="52" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="170" y="109" text-anchor="middle" class="mono" style="font-size:12px">mb0</text>
    <rect x="224" y="78" width="100" height="52" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="274" y="109" text-anchor="middle" class="mono" style="font-size:12px">mb1</text>
    <rect x="328" y="78" width="100" height="52" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="378" y="109" text-anchor="middle" class="mono" style="font-size:12px">mb2</text>
    <rect x="432" y="78" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="482" y="109" text-anchor="middle" style="font-size:11px;fill:var(--faint)">气泡</text>
    <rect x="536" y="78" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="586" y="109" text-anchor="middle" style="font-size:11px;fill:var(--faint)">气泡</text>
    <rect x="120" y="146" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="170" y="177" text-anchor="middle" style="font-size:11px;fill:var(--faint)">气泡</text>
    <rect x="224" y="146" width="100" height="52" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="274" y="177" text-anchor="middle" class="mono" style="font-size:12px">mb0</text>
    <rect x="328" y="146" width="100" height="52" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="378" y="177" text-anchor="middle" class="mono" style="font-size:12px">mb1</text>
    <rect x="432" y="146" width="100" height="52" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="482" y="177" text-anchor="middle" class="mono" style="font-size:12px">mb2</text>
    <rect x="536" y="146" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="586" y="177" text-anchor="middle" style="font-size:11px;fill:var(--faint)">气泡</text>
    <rect x="120" y="214" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="170" y="245" text-anchor="middle" style="font-size:11px;fill:var(--faint)">气泡</text>
    <rect x="224" y="214" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="274" y="245" text-anchor="middle" style="font-size:11px;fill:var(--faint)">气泡</text>
    <rect x="328" y="214" width="100" height="52" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="378" y="245" text-anchor="middle" class="mono" style="font-size:12px">mb0</text>
    <rect x="432" y="214" width="100" height="52" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="482" y="245" text-anchor="middle" class="mono" style="font-size:12px">mb1</text>
    <rect x="536" y="214" width="100" height="52" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="586" y="245" text-anchor="middle" class="mono" style="font-size:12px">mb2</text>
    <line x1="170" y1="130" x2="368" y2="216" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="372,218 361,214 365,224" style="fill:var(--blue)"/>
    <text x="662" y="100" style="font-size:11px;fill:var(--muted)">填充</text>
    <text x="662" y="172" style="font-size:11px;fill:var(--muted)">稳态</text>
    <text x="662" y="240" style="font-size:11px;fill:var(--muted)">排空</text>
  </svg>
  <div class="figcap"><b>图 23B · PP：层分成阶段、微批流水</b> — 模型层切成 <span class="mono">S0/S1/S2</span> 三段放在三张卡，micro-batch <span class="mono">mb0/mb1/mb2</span> 沿对角线错峰流动；<span class="mono">t0–t1</span> 是只有部分段在忙的<strong>填充</strong>气泡、<span class="mono">t2</span> 是三段全忙的<strong>稳态</strong>、<span class="mono">t3–t4</span> 是尾部<strong>排空</strong>气泡。micro-batch 越多，稳态占比越高、气泡越薄。</div>
</div>

<p>看懂这张错峰图，就抓住了 PP 的灵魂：<strong>同一时刻，不同 stage 在处理不同的 micro-batch。</strong>
正因为有<strong>多个 micro-batch 同时在飞</strong>，每段 GPU 才不会空等上一段——这也是为什么 <span class="inline">event_loop_pp</span> 必须比 normal 循环复杂：
它要<strong>同时追踪</strong>好几个在不同 stage 的 micro-batch，管理段间激活值的发送/接收、协调"什么时候发下一个 micro-batch"。
朴素的单段循环每拍只盯一个批；PP 循环每拍要盯一串。<strong>填充期</strong>和<strong>排空期</strong>那两头不可避免有气泡，但只要稳态够长（micro-batch 数远大于 stage 数），气泡占比就趋近于零。</p>

<p>这里值得停下来算一笔直觉账，体会"micro-batch 数为什么要远大于 stage 数"。假设有 4 个 stage、只发 1 个 micro-batch：它要依次走完 4 段才出结果，
这期间任意时刻都只有<strong>一段</strong>在忙，其余三段全闲——利用率约 1/4，气泡高达 3/4，等于白白浪费了四分之三的算力。
若改发 16 个 micro-batch：头几拍仍在填充、尾几拍仍在排空，但中间会有<strong>很长一段</strong>四段全忙的稳态，平摊下来利用率能逼近满载。
直觉就是：<strong>填充与排空的气泡是"固定成本"</strong>（约正比于 stage 数），而稳态满载的收益<strong>随 micro-batch 数增长</strong>——micro-batch 越多，固定气泡被摊得越薄。
这也解释了 PP 的一个隐藏权衡：micro-batch 切得太碎，单个 batch 太小会让每次前向的 GPU 利用率下降；切得太粗，又填不满流水线。真实系统要在两者间找平衡。</p>

<h2>三种并行怎么拼：TP × PP × DP</h2>
<p>DP、PP、还有第 46 课要深讲的 <strong>TP（张量并行）</strong>，是三个<strong>正交</strong>的维度，可以叠在一起用。
区分它们的最快办法，是问一句：<strong>"它复制什么、切分什么？"</strong></p>

<table class="t">
  <tr><th>维度</th><th>对模型做什么</th><th>切/复制的粒度</th><th>谁来编排</th></tr>
  <tr><td><strong>DP 数据并行</strong></td><td><strong>整份模型复制</strong>多份，请求扇出到各副本</td><td class="mono">整个运行时（副本级）</td><td class="mono">DataParallelController</td></tr>
  <tr><td><strong>PP 流水线并行</strong></td><td>把<strong>层切成段</strong>分到不同卡，micro-batch 接力</td><td class="mono">层（stage 级）</td><td class="mono">event_loop_pp（调度器内）</td></tr>
  <tr><td><strong>TP 张量并行</strong></td><td>把<strong>每一层的矩阵</strong>横/纵切，多卡协同算一层</td><td class="mono">单层的权重矩阵</td><td class="mono">模型前向内部（第 24/46 课）</td></tr>
</table>

<p>把三者拼起来想：<strong>TP 在一层之内</strong>（多卡合算同一层的矩阵乘）、<strong>PP 跨层之间</strong>（不同段在不同卡）、<strong>DP 跨整个副本</strong>（整机复制扇出）。
一个典型的大集群可能是 "TP=8、PP=2、DP=4"——即每个副本用 16 张卡（8 路 TP × 2 段 PP）装下一份大模型，再复制 4 份副本扛流量，总共 64 张卡。
而<strong>调度器在其中编排的是 DP（靠控制器分发）和 PP（靠 pp 循环推进 micro-batch）</strong>这两个改变控制流的维度；
TP 则<strong>藏在模型前向内部</strong>，对调度器基本透明——调度器把批交给 ModelRunner，TP 的多卡协同在 <span class="inline">forward</span> 里自动发生（第 24 课）。这就是为什么本课归在 Part 5：DP 与 PP 是<strong>调度</strong>问题，TP 是<strong>计算</strong>问题。</p>

<h2>什么时候用 DP、什么时候用 PP</h2>
<p>两者解决的是<strong>不同的瓶颈</strong>，别混为一谈：</p>

<div class="cols">
  <div class="col"><h4>DP：为吞吐而生（模型能装下）</h4><p>前提是<strong>一份模型已经能放进现有卡组</strong>。流量大、单副本排队严重时，复制 N 份、用控制器把请求摊开，
  吞吐近似线性放大。代价是<strong>显存成倍占用</strong>（每副本一整份权重）。它<strong>不能</strong>帮你装下一个本来就放不下的模型——复制只会让"放不下"变成"放不下 N 次"。</p></div>
  <div class="col"><h4>PP：为容量而生（模型太大）</h4><p>前提是<strong>模型大到一张卡（甚至一组 TP 卡）装不下</strong>。把层切成段、摊到多卡，每卡只扛一段权重，于是"放不下"变"放得下"。
  代价是<strong>填充/排空的气泡</strong>和<strong>段间通信</strong>，要靠多 micro-batch 摊平。它<strong>不直接</strong>提升单请求延迟，主要是<strong>让超大模型跑得起来</strong>。</p></div>
</div>

<p>实践中两者常<strong>同时</strong>出现：先用 PP（和 TP）把一份大模型<strong>装进</strong>一组卡，再用 DP 把这组"装好的副本"<strong>复制</strong>多份扛流量。
所以读到这里，你应当能把这句话翻译成具体动作了——<strong>"装得下"靠切分（PP/TP），"喂得饱"靠复制（DP）</strong>。</p>

<p>最后再把这三维和你前面学过的整套调度知识接上，收一个干净的尾。Part 5 一路讲下来，从第 18 课的<strong>单段心跳</strong>，到第 19 课的请求与批数据结构、第 20 课的调度策略、
第 21 课的重叠流水线、第 22 课的分块预填充——全都是<strong>一个调度器进程、一张卡</strong>之内的故事。本课做的，是把这套"单机心跳"<strong>沿两条新轴展开</strong>：
沿 DP 轴，是把整台心脏<strong>复印多份</strong>、在前面加一个不跳动只分发的控制器；沿 PP 轴，是把一次心跳的前向<strong>拆成接力</strong>、让多个 micro-batch 错峰填满管线。
两条轴都<strong>不改变单次前向的数学</strong>，只改变<strong>控制流</strong>——谁来分发、有几个 micro-batch 在飞——这正是它们归在"调度"这一 Part 的根本理由。
带着这把尺子往后读：第 24 课会钻进单次 <span class="inline">forward</span> 看 TP 怎么在一层内切矩阵，第 46 课把 TP/PP/EP/DP 四维一次性讲透，第 47 课讲 MoE 的专家负载均衡 EPLB。
至此，调度器这颗"心脏"从单卡跳动，到多卡多副本的协同编排，你已经有了完整的骨架图。</p>

<h2>真实代码：控制器怎么轮询分发</h2>
<p>下面是 <span class="inline">DataParallelController</span> 里最核心的轮询分发逻辑。注意它有多<strong>轻</strong>——
就是维护一个 <span class="mono">round_robin_counter</span>，找到下一个<strong>就绪</strong>的 worker，把请求 <span class="mono">sock_send</span> 过去，指针前移取模回绕。它<strong>从不碰 GPU</strong>，只做"分给谁"这一个决定。</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/data_parallel_controller.py ::DataParallelController</span><span class="ln">DP 控制器：轮询把请求分给下一个就绪副本</span></div>
  <pre><span class="kw">def</span> round_robin_scheduler(self, req: Req):
    <span class="cm"># 若指定了 DP rank 就直接路由，否则轮询</span>
    <span class="kw">if</span> self.maybe_external_dp_rank_routing(req):
        <span class="kw">return</span>

    <span class="kw">while</span> <span class="kw">True</span>:
        <span class="kw">if</span> self.status[self.round_robin_counter]:
            <span class="cm"># 这个 worker 就绪：把请求发过去</span>
            sock_send(self.workers[self.round_robin_counter], req)
            self.round_robin_counter = (self.round_robin_counter + <span class="mono">1</span>) % len(
                self.workers
            )
            <span class="kw">break</span>
        <span class="cm"># 否则跳过它，看下一个</span>
        self.round_robin_counter = (self.round_robin_counter + <span class="mono">1</span>) % len(
            self.workers
        )</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/data_parallel_controller.py ::DataParallelController.round_robin_scheduler</span><span class="ln">轮询把请求分发到下一个 DP 工作副本</span></div>
  <pre><span class="kw">def</span> round_robin_scheduler(self, req):
    <span class="cm"># send this request to the next DP worker in rotation, then advance.</span>
    sock_send(self.workers[self.round_robin_counter], req)
    self.round_robin_counter = (self.round_robin_counter + <span class="mono">1</span>) % len(self.workers)</pre>
</div>

<p>把这段和扇出图对齐：<span class="inline">self.workers</span> 就是那 N 个副本的 socket，<span class="inline">round_robin_counter</span> 是迎宾员手里"该轮到第几条通道"的计数器，
<span class="inline">self.status</span> 标记哪些副本就绪（没就绪就跳过）。<span class="inline">sock_send</span> 把请求<strong>原样投递</strong>给选中的副本——之后这条请求的命运就完全交给那个副本内部的调度器了，
控制器立刻回头处理下一条。整段没有一行矩阵乘，这正印证了"<strong>控制器只决策分发、不参与计算</strong>"。负载感知策略（按在跑请求数/token 数挑最闲的副本）也只是把这里的"按指针轮"换成"按账本挑"，本质同样轻量。</p>

<div class="card key">
  <div class="tag">📌 本课要点</div>
  <ul>
    <li><strong>DP（数据并行）</strong>：复制<strong>整个运行时</strong>成 N 个独立副本（各有调度器+TP+KV+一整份模型），<span class="mono">DataParallelController</span> 用轮询/负载感知把请求<strong>扇出</strong>；是<strong>吞吐乘法器</strong>，请求级隔离、副本间不交互。</li>
    <li><strong>PP（流水线并行）</strong>：把模型的<strong>层切成段</strong>分到多卡，<span class="mono">event_loop_pp</span> 让<strong>多个 micro-batch 同时在段间流水</strong>（stage1 跑 B 时 stage2 跑 A），靠 micro-batch 数摊平填充/排空的<strong>气泡</strong>；为<strong>装下超大模型</strong>而生。</li>
    <li><strong>TP × PP × DP 三维正交</strong>：TP 切<strong>一层内的矩阵</strong>、PP 切<strong>跨层的段</strong>、DP 复制<strong>整个副本</strong>。调度器编排 DP（控制器）与 PP（pp 循环）；TP 藏在模型前向里、对调度器透明。</li>
    <li><strong>选型口诀</strong>：模型太大用 PP/TP（切分，"装得下"）；流量太大用 DP（复制，"喂得饱"）；大部署常三者叠加。深入见第 46 课（TP/PP/EP/DP）、第 47 课（EPLB）、第 24 课（前向）、第 18 课（事件循环）、第 2 课（三进程模型）。</li>
  </ul>
</div>
""",
             "en": r"""
<p class="lead">
For five lessons we watched <strong>one</strong> scheduler process beat on <strong>one</strong> card: receive, batch, forward, finish. But real deployments have 8 cards per box and hundreds per cluster—
a model may be <strong>too big for one card</strong>, and traffic may be <strong>too heavy for one replica</strong>. Closing Part 5, we pull the camera back to the <strong>multi-process, multi-card</strong> scale and look at two parallelisms that change the <strong>control flow</strong>:
<strong>data parallel (DP)</strong> (replicate the whole runtime N times and fan requests out) and <strong>pipeline parallel (PP)</strong> (split the model's layers into stages and keep multiple micro-batches flowing across them).
Note: what they change is not how the matrices compute, but <strong>who dispatches and how many micro-batches are in flight</strong>—exactly the scheduler's concern.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of <strong>DP</strong> as <strong>multiple identical checkout lanes</strong> in a store: each lane is <strong>fully equipped</strong> (its own cashier, scanner, register), and a <strong>greeter</strong> at the door routes each customer to a lane;
  lanes don't interfere, and the more you open, the more customers you ring up per minute. Think of <strong>PP</strong> as <strong>one long conveyor with stations</strong>: each station does just one step (label → box → seal),
  and many items ride at once, each at a different station, so no station sits idle. DP is "replicate whole lanes for throughput", PP is "split one job into a relay so everyone stays busy"—one handles <strong>dispatch</strong>, one handles <strong>relay</strong>.
</div>

<div class="card macro">
  <div class="tag">🌍 The big picture</div>
  In one line: <strong>DP replicates "the whole model" and fans requests out to many independent replicas; PP splits "the model's layers" and pipelines multiple micro-batches across stages.</strong>
  DP's conductor is the <span class="inline">DataParallelController</span>—it does <strong>no forward</strong> itself, only dispatching incoming requests <strong>round-robin or load-aware</strong> to N <strong>full runtimes</strong> (each replica has its own scheduler + TP workers + KV cache + a full copy of the model weights).
  PP's executor is <span class="inline">SchedulerPPMixin.event_loop_pp</span>—it reshapes Lesson 18's single-stage heartbeat into a <strong>pipelined loop</strong>, so stage1 processes micro-batch B while stage2 processes micro-batch A,
  amortizing the fill/drain <strong>"bubble"</strong>. Layer on <strong>tensor parallel (TP)</strong> (split each layer's matrices, deep dive Lesson 46) and a big deployment is <strong>TP × PP × DP</strong> stitched in three dimensions.
</div>

<h2>Data parallel (DP): replicate the whole runtime, fan requests out</h2>
<p>DP's idea is the plainest: <strong>if one replica can't keep up with traffic, run N replicas</strong>. The key is that each replica is a <strong>complete engine</strong>—
not sharing some weights, but <strong>each holding a full model copy</strong>, each running a Lesson-18 event loop, each managing its own KV ledger.
Replicas are <strong>request-isolated</strong>: customer A on replica 1, customer B on replica 2, never interacting. The only coordination needed is that <strong>dispatcher</strong> at the door—
the <span class="inline">DataParallelController</span>. It catches all incoming requests up front, picks a replica by policy (round-robin <span class="mono">round_robin_scheduler</span>,
or load-aware by in-flight requests/tokens), <strong>forwards the request as-is</strong>, and never touches the GPU.</p>

<div class="flow">
  <div class="node hl"><div class="nt">DataParallelController</div><div class="nd">round-robin / load-aware<br>round_robin_scheduler</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Replica 1 (full runtime)</div><div class="nd">scheduler+TP+KV<br>a full model</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Replica 2 (full runtime)</div><div class="nd">scheduler+TP+KV<br>a full model</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="nt">Replica N (full runtime)</div><div class="nd">scheduler+TP+KV<br>a full model</div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Data parallel DP: request stream r0/r1/r2/r3 enters the DataParallelController and is fanned out round-robin 0 to 1 to 2 to 0 across three full replicas, with replicas 0/1/2 each holding different requests">
    <text x="20" y="26" style="font-weight:700;fill:var(--muted)">request stream (round-robin 0→1→2→0)</text>
    <rect x="20" y="78" width="52" height="26" rx="5" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="46" y="96" text-anchor="middle" class="mono" style="font-size:11px">r0</text>
    <rect x="20" y="110" width="52" height="26" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="46" y="128" text-anchor="middle" class="mono" style="font-size:11px">r1</text>
    <rect x="20" y="142" width="52" height="26" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="46" y="160" text-anchor="middle" class="mono" style="font-size:11px">r2</text>
    <rect x="20" y="174" width="52" height="26" rx="5" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="46" y="192" text-anchor="middle" class="mono" style="font-size:11px">r3</text>
    <line x1="72" y1="135" x2="208" y2="135" style="stroke:var(--line);stroke-width:1.5"/>
    <polygon points="214,135 204,130 204,140" style="fill:var(--line)"/>
    <rect x="214" y="101" width="168" height="68" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="298" y="128" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">DataParallelController</text>
    <text x="298" y="150" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--accent-ink)">round_robin · no GPU</text>
    <line x1="382" y1="120" x2="552" y2="52" style="stroke:var(--blue);stroke-width:1.5"/>
    <polygon points="558,50 547,49 550,60" style="fill:var(--blue)"/>
    <line x1="382" y1="135" x2="552" y2="148" style="stroke:var(--amber);stroke-width:1.5"/>
    <polygon points="558,149 547,143 548,154" style="fill:var(--amber)"/>
    <line x1="382" y1="150" x2="552" y2="244" style="stroke:var(--purple);stroke-width:1.5"/>
    <polygon points="558,246 547,238 550,249" style="fill:var(--purple)"/>
    <rect x="558" y="20" width="200" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--blue);stroke-width:1.5"/>
    <text x="568" y="42" style="font-weight:700">Replica 0 (full runtime)</text>
    <text x="568" y="60" class="mono" style="font-size:10.5px;fill:var(--muted)">scheduler+TP+KV · a full model</text>
    <rect x="568" y="66" width="58" height="14" rx="3" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1"/>
    <text x="597" y="77" text-anchor="middle" class="mono" style="font-size:9.5px">r0 · r3</text>
    <rect x="558" y="116" width="200" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--amber);stroke-width:1.5"/>
    <text x="568" y="138" style="font-weight:700">Replica 1 (full runtime)</text>
    <text x="568" y="156" class="mono" style="font-size:10.5px;fill:var(--muted)">scheduler+TP+KV · a full model</text>
    <rect x="568" y="162" width="40" height="14" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1"/>
    <text x="588" y="173" text-anchor="middle" class="mono" style="font-size:9.5px">r1</text>
    <rect x="558" y="212" width="200" height="64" rx="8" style="fill:var(--panel-2);stroke:var(--purple);stroke-width:1.5"/>
    <text x="568" y="234" style="font-weight:700">Replica 2 (full runtime)</text>
    <text x="568" y="252" class="mono" style="font-size:10.5px;fill:var(--muted)">scheduler+TP+KV · a full model</text>
    <rect x="568" y="258" width="40" height="14" rx="3" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1"/>
    <text x="588" y="269" text-anchor="middle" class="mono" style="font-size:9.5px">r2</text>
  </svg>
  <div class="figcap"><b>Fig 23A · DP: requests split across replicas</b> — the request stream enters the dispatch-only <span class="mono">DataParallelController</span>, fanned out in <span class="mono">round_robin</span> order <span class="mono">0→1→2→0</span> to N <strong>full replicas</strong> (each with its own scheduler+TP+KV+a full model); replicas 0/1/2 each hold different requests (r3 wraps back to replica 0), and the controller never touches the GPU.</div>
</div>

<p>Two concrete numbers to ground this. <strong><span class="mono">--dp-size 4</span></strong> spins up 4 full replicas (replica 0/1/2/3), and the controller hands requests out in <span class="mono">0→1→2→3→0</span> order, round and round;
while <strong><span class="mono">--pp-size 2</span></strong> cuts the model's layers into 2 stages relayed across 2 cards. The former scales throughput, the latter scales capacity.</p>

<p>Read three things off this figure. First, <strong>the controller sits before all replicas</strong>, the first fork a request hits, making only the "to whom" decision.
Second, each replica is <strong>self-contained</strong>—inside it is everything from the last five lessons: an <span class="inline">event_loop_normal</span>, a prefill/decode trade-off, an independent KV ledger.
Third, DP is a <strong>throughput multiplier</strong>: N replicas can scale whole-box throughput N-fold, since requests are spread and don't contend.
A caveat: this is <strong>request-level DP</strong>; in MoE models there is finer-grained <strong>attention DP / expert DP</strong>, with more nuanced control flow and communication, deferred to Lessons 46/47 and not covered here.</p>

<h2>Pipeline parallel (PP): split layers into stages, keep micro-batches in flight</h2>
<p>When a model is <strong>too big for one card</strong>, DP replication is hopeless—you can't even fit one copy, let alone N. Now use PP: cut the model's <strong>layers</strong> vertically into a few <strong>stages</strong>,
layers 1–10 on GPU0, 11–20 on GPU1, 21–30 on GPU2…, so each card holds <strong>one stage</strong>. A batch's forward becomes a <strong>relay</strong>:
data passes stage1, hands activations to stage2, then stage3… The catch: with only one batch in flight, while stage2 works, stage1/stage3 sit <strong>idle waiting</strong>,
and GPU utilization is dismal—this is the <strong>pipeline bubble</strong>. The fix is to split a big batch into multiple <strong>micro-batches</strong> and run them <strong>staggered</strong>, in flight together.</p>

<div class="vflow">
  <div class="step"><div class="num">t1</div><div class="sc"><h4>Fill</h4><p>micro-batch A enters <strong>stage1</strong>; stage2, stage3 still empty—this tick is part of the bubble, only one stage busy.</p></div></div>
  <div class="step"><div class="num">t2</div><div class="sc"><h4>Ramping</h4><p>A advances to <strong>stage2</strong> while micro-batch B enters <strong>stage1</strong>; two stages busy, the pipe begins to fill.</p></div></div>
  <div class="step"><div class="num">t3</div><div class="sc"><h4>Full (steady state)</h4><p>A in <strong>stage3</strong>, B in <strong>stage2</strong>, C enters <strong>stage1</strong>—<strong>all three busy</strong>, bubble amortized; this is where PP wants to live.</p></div></div>
  <div class="step"><div class="num">t4</div><div class="sc"><h4>Drain</h4><p>After the last micro-batch is launched, A has emitted and B/C walk off the tail stages; a tail bubble reappears. <strong>More micro-batches ⇒ higher full-load fraction</strong>, smaller bubble.</p></div></div>
</div>

<div class="fig">
  <svg viewBox="0 0 780 300" role="img" aria-label="Pipeline parallel PP: model layers split into stages S0/S1/S2 on three cards, micro-batches mb0/mb1/mb2 flow diagonally and staggered, with fill and drain bubbles at the ends and an all-stages-busy steady state in the middle">
    <text x="24" y="40" style="font-weight:700;fill:var(--muted)">stages (one card each, each holds one stage of layers)</text>
    <text x="170" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t0</text>
    <text x="274" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t1</text>
    <text x="378" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t2</text>
    <text x="482" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t3</text>
    <text x="586" y="64" text-anchor="middle" style="fill:var(--muted);font-size:12px">t4</text>
    <rect x="24" y="78" width="86" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="67" y="100" text-anchor="middle" style="font-weight:700">S0 · GPU0</text>
    <text x="67" y="118" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">layers 1–10</text>
    <rect x="24" y="146" width="86" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="67" y="168" text-anchor="middle" style="font-weight:700">S1 · GPU1</text>
    <text x="67" y="186" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">layers 11–20</text>
    <rect x="24" y="214" width="86" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="67" y="236" text-anchor="middle" style="font-weight:700">S2 · GPU2</text>
    <text x="67" y="254" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">layers 21–30</text>
    <rect x="120" y="78" width="100" height="52" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="170" y="109" text-anchor="middle" class="mono" style="font-size:12px">mb0</text>
    <rect x="224" y="78" width="100" height="52" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="274" y="109" text-anchor="middle" class="mono" style="font-size:12px">mb1</text>
    <rect x="328" y="78" width="100" height="52" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="378" y="109" text-anchor="middle" class="mono" style="font-size:12px">mb2</text>
    <rect x="432" y="78" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="482" y="109" text-anchor="middle" style="font-size:11px;fill:var(--faint)">bubble</text>
    <rect x="536" y="78" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="586" y="109" text-anchor="middle" style="font-size:11px;fill:var(--faint)">bubble</text>
    <rect x="120" y="146" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="170" y="177" text-anchor="middle" style="font-size:11px;fill:var(--faint)">bubble</text>
    <rect x="224" y="146" width="100" height="52" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="274" y="177" text-anchor="middle" class="mono" style="font-size:12px">mb0</text>
    <rect x="328" y="146" width="100" height="52" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="378" y="177" text-anchor="middle" class="mono" style="font-size:12px">mb1</text>
    <rect x="432" y="146" width="100" height="52" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="482" y="177" text-anchor="middle" class="mono" style="font-size:12px">mb2</text>
    <rect x="536" y="146" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="586" y="177" text-anchor="middle" style="font-size:11px;fill:var(--faint)">bubble</text>
    <rect x="120" y="214" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="170" y="245" text-anchor="middle" style="font-size:11px;fill:var(--faint)">bubble</text>
    <rect x="224" y="214" width="100" height="52" rx="6" style="fill:var(--panel-2);stroke:var(--faint);stroke-width:1.5;stroke-dasharray:4 4"/>
    <text x="274" y="245" text-anchor="middle" style="font-size:11px;fill:var(--faint)">bubble</text>
    <rect x="328" y="214" width="100" height="52" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="378" y="245" text-anchor="middle" class="mono" style="font-size:12px">mb0</text>
    <rect x="432" y="214" width="100" height="52" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="482" y="245" text-anchor="middle" class="mono" style="font-size:12px">mb1</text>
    <rect x="536" y="214" width="100" height="52" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="586" y="245" text-anchor="middle" class="mono" style="font-size:12px">mb2</text>
    <line x1="170" y1="130" x2="368" y2="216" style="stroke:var(--blue);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="372,218 361,214 365,224" style="fill:var(--blue)"/>
    <text x="662" y="100" style="font-size:11px;fill:var(--muted)">fill</text>
    <text x="662" y="172" style="font-size:11px;fill:var(--muted)">steady</text>
    <text x="662" y="240" style="font-size:11px;fill:var(--muted)">drain</text>
  </svg>
  <div class="figcap"><b>Fig 23B · PP: layers as stages, micro-batches pipelined</b> — model layers split into <span class="mono">S0/S1/S2</span> on three cards, micro-batches <span class="mono">mb0/mb1/mb2</span> flowing diagonally and staggered; <span class="mono">t0–t1</span> are the <strong>fill</strong> bubble with only some stages busy, <span class="mono">t2</span> is the all-stages-busy <strong>steady state</strong>, and <span class="mono">t3–t4</span> are the tail <strong>drain</strong> bubble. More micro-batches ⇒ higher steady-state fraction, thinner bubble.</div>
</div>

<p>Grasp this staggered figure and you've got PP's soul: <strong>at any instant, different stages process different micro-batches.</strong>
Precisely because <strong>multiple micro-batches are in flight</strong>, each stage's GPU never idles waiting on the previous—this is why <span class="inline">event_loop_pp</span> must be more complex than the normal loop:
it <strong>tracks several</strong> micro-batches at different stages, manages inter-stage activation send/recv, and coordinates "when to launch the next micro-batch".
A plain single-stage loop watches one batch per tick; the PP loop watches a string of them. The <strong>fill</strong> and <strong>drain</strong> ends carry unavoidable bubbles, but as long as steady state is long enough (micro-batches ≫ stages), the bubble fraction approaches zero.</p>

<h2>How the three compose: TP × PP × DP</h2>
<p>DP, PP, and the <strong>TP (tensor parallel)</strong> that Lesson 46 details are three <strong>orthogonal</strong> dimensions you can stack.
The fastest way to tell them apart is to ask: <strong>"what does it replicate, what does it split?"</strong></p>

<table class="t">
  <tr><th>Dimension</th><th>What it does to the model</th><th>Granularity of split/replicate</th><th>Who orchestrates</th></tr>
  <tr><td><strong>DP data parallel</strong></td><td><strong>Replicate the whole model</strong>, fan requests to replicas</td><td class="mono">whole runtime (replica level)</td><td class="mono">DataParallelController</td></tr>
  <tr><td><strong>PP pipeline parallel</strong></td><td>Split <strong>layers into stages</strong> across cards, micro-batch relay</td><td class="mono">layers (stage level)</td><td class="mono">event_loop_pp (in scheduler)</td></tr>
  <tr><td><strong>TP tensor parallel</strong></td><td>Split <strong>each layer's matrices</strong>, many cards compute one layer</td><td class="mono">a single layer's weight matrix</td><td class="mono">inside model forward (Lessons 24/46)</td></tr>
</table>

<p>Picture them stacked: <strong>TP within a layer</strong> (many cards co-compute one layer's matmul), <strong>PP across layers</strong> (different stages on different cards), <strong>DP across whole replicas</strong> (replicate the box and fan out).
A typical large cluster might be "TP=8, PP=2, DP=4"—each replica uses 16 cards (8-way TP × 2 PP stages) to hold one copy of a big model, then 4 replicas to carry traffic, 64 cards in all.
And <strong>what the scheduler orchestrates here is DP (via the controller's dispatch) and PP (via the pp loop advancing micro-batches)</strong>—the two dimensions that change control flow;
TP <strong>hides inside the model forward</strong>, largely transparent to the scheduler—the scheduler hands the batch to ModelRunner and TP's multi-card cooperation happens automatically inside <span class="inline">forward</span> (Lesson 24). That's why this lesson belongs in Part 5: DP and PP are <strong>scheduling</strong> problems, TP is a <strong>compute</strong> problem.</p>

<h2>When to use DP vs PP</h2>
<p>They solve <strong>different bottlenecks</strong>—don't conflate them:</p>

<div class="cols">
  <div class="col"><h4>DP: for throughput (model fits)</h4><p>The premise is <strong>one copy already fits the existing cards</strong>. When traffic is high and a single replica queues badly, replicate N copies and let the controller spread requests,
  scaling throughput near-linearly. The cost is <strong>VRAM multiplied</strong> (a full weight set per replica). It <strong>cannot</strong> help you fit a model that doesn't fit—replication just makes "doesn't fit" into "doesn't fit N times".</p></div>
  <div class="col"><h4>PP: for capacity (model too big)</h4><p>The premise is <strong>the model is too big for one card</strong> (even one TP group). Split layers into stages across cards so each holds only one stage of weights, turning "doesn't fit" into "fits".
  The cost is <strong>fill/drain bubbles</strong> and <strong>inter-stage communication</strong>, amortized by many micro-batches. It does <strong>not</strong> directly improve single-request latency—mainly it <strong>makes a huge model runnable</strong>.</p></div>
</div>

<p>In practice the two often appear <strong>together</strong>: first use PP (and TP) to <strong>fit</strong> one big model into a card group, then use DP to <strong>replicate</strong> that ready-made replica to carry traffic.
So by now you should be able to translate this into concrete actions—<strong>"make it fit" via splitting (PP/TP), "keep it fed" via replication (DP)</strong>.</p>

<h2>Real code: how the controller round-robins dispatch</h2>
<p>Below is the core round-robin dispatch in <span class="inline">DataParallelController</span>. Note how <strong>light</strong> it is—
it keeps a <span class="mono">round_robin_counter</span>, finds the next <strong>ready</strong> worker, <span class="mono">sock_send</span>s the request there, and advances the pointer modulo the worker count. It <strong>never touches the GPU</strong>, making only the "to whom" decision.</p>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/data_parallel_controller.py ::DataParallelController</span><span class="ln">DP controller: round-robin to the next ready replica</span></div>
  <pre><span class="kw">def</span> round_robin_scheduler(self, req: Req):
    <span class="cm"># Direct-route if a DP rank is pinned, else round-robin</span>
    <span class="kw">if</span> self.maybe_external_dp_rank_routing(req):
        <span class="kw">return</span>

    <span class="kw">while</span> <span class="kw">True</span>:
        <span class="kw">if</span> self.status[self.round_robin_counter]:
            <span class="cm"># This worker is ready: send the request</span>
            sock_send(self.workers[self.round_robin_counter], req)
            self.round_robin_counter = (self.round_robin_counter + <span class="mono">1</span>) % len(
                self.workers
            )
            <span class="kw">break</span>
        <span class="cm"># Otherwise skip it, look at the next</span>
        self.round_robin_counter = (self.round_robin_counter + <span class="mono">1</span>) % len(
            self.workers
        )</pre>
</div>

<div class="codefile">
  <div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/data_parallel_controller.py ::DataParallelController.round_robin_scheduler</span><span class="ln">round-robin: hand each request to the next DP worker</span></div>
  <pre><span class="kw">def</span> round_robin_scheduler(self, req):
    <span class="cm"># send this request to the next DP worker in rotation, then advance.</span>
    sock_send(self.workers[self.round_robin_counter], req)
    self.round_robin_counter = (self.round_robin_counter + <span class="mono">1</span>) % len(self.workers)</pre>
</div>

<p>Align this with the fan-out figure: <span class="inline">self.workers</span> are the N replicas' sockets, <span class="inline">round_robin_counter</span> is the greeter's "which lane is next" counter,
<span class="inline">self.status</span> marks which replicas are ready (skip the ones that aren't). <span class="inline">sock_send</span> <strong>delivers the request as-is</strong> to the chosen replica—after that the request's fate is entirely up to that replica's internal scheduler,
and the controller immediately turns to the next. Not one line of matmul, confirming "<strong>the controller only decides dispatch, never computes</strong>". Load-aware policy (pick the least-busy replica by in-flight requests/tokens) just swaps "advance the pointer" for "consult the ledger", equally lightweight.</p>

<div class="card key">
  <div class="tag">📌 Key points</div>
  <ul>
    <li><strong>DP (data parallel)</strong>: replicate the <strong>whole runtime</strong> into N independent replicas (each with scheduler+TP+KV+a full model), with <span class="mono">DataParallelController</span> <strong>fanning out</strong> requests round-robin/load-aware; a <strong>throughput multiplier</strong>, request-isolated, replicas don't interact.</li>
    <li><strong>PP (pipeline parallel)</strong>: split the model's <strong>layers into stages</strong> across cards; <span class="mono">event_loop_pp</span> keeps <strong>multiple micro-batches flowing across stages</strong> (stage1 on B while stage2 on A), amortizing the fill/drain <strong>bubble</strong> via micro-batch count; born to <strong>fit a huge model</strong>.</li>
    <li><strong>TP × PP × DP are three orthogonal dims</strong>: TP splits <strong>matrices within a layer</strong>, PP splits <strong>stages across layers</strong>, DP replicates <strong>whole replicas</strong>. The scheduler orchestrates DP (controller) and PP (pp loop); TP hides in the model forward, transparent to the scheduler.</li>
    <li><strong>Selection rule</strong>: model too big ⇒ PP/TP (split, "make it fit"); traffic too heavy ⇒ DP (replicate, "keep it fed"); big deployments stack all three. Deep dives: Lesson 46 (TP/PP/EP/DP), Lesson 47 (EPLB), Lesson 24 (forward), Lesson 18 (event loop), Lesson 2 (3-process model).</li>
  </ul>
</div>
"""}
