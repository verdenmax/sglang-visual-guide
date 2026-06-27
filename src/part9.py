"""Part 9 · Kernels & hardware (L38-42).

Lesson content for the deep kernel + multi-hardware part of the SGLang visual guide.
Each LESSON_XX is a {"zh": html, "en": html} dict consumed via registry.CONTENT.
"""

LESSON_38 = {"zh": r"""
<p class="lead">前面整整一个 Part-8，我们都在 Python 层讲"算法":注意力后端怎么排版 KV(第33课)、MoE 怎么路由(第34课)、量化怎么压缩权重(第35课)、归一化与激活怎么算(第36课)。但这些 Python 代码最终都要落到一段真正在 GPU 上飞速运行的机器码上。<strong>sgl-kernel</strong> 就是承载这段机器码的地方——它是一个<strong>独立的 C++/CUDA 工程</strong>,有自己的 <span class="mono">CMakeLists.txt</span>,把热路径上的核函数<strong>提前编译(AOT)</strong>成一个 <span class="mono">.so</span> 动态库,随 wheel 包一起发布。本课带你俯瞰这个工程的全貌:它怎么组织、怎么把核函数注册成 PyTorch 自定义算子、Python 又是怎么通过 <span class="mono">torch.ops.sgl_kernel.*</span> 把它们调起来的。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>把 SGLang 主工程想象成一台组装好的整车,Python 代码是仪表盘和方向盘——你转动方向盘,车就转弯。而 <strong>sgl-kernel</strong> 是发动机厂:它在另一条生产线上,用完全不同的工艺(C++/CUDA、CMake)把发动机<strong>预先造好</strong>,装箱(打进 <span class="mono">.so</span>)后送到整车厂。你坐进驾驶室时,发动机早就在引擎盖下待命了,你只需踩油门(调用 <span class="mono">torch.ops.sgl_kernel.merge_state_v2</span>),根本不用关心活塞和缸体是怎么铸造的。仪表盘上每个按钮(Python 薄包装)背后,都接着一根线缆(torch 算子注册表)通向发动机里某个具体零件(<span class="mono">csrc/</span> 里的某个 kernel)。</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>为什么要单独搞一个工程来放核函数?因为 GPU kernel 的编译与 Python 的运行节奏完全不同。Python 是解释执行、随改随跑;CUDA kernel 却要经过 <span class="mono">nvcc</span> 编译、链接,耗时几分钟。如果每次启动服务都现编译,延迟无法接受。所以 SGLang 选择把最常用、最稳定的核函数<strong>提前编译(Ahead-Of-Time, AOT)</strong>,塞进 wheel 里的 <span class="mono">.so</span>,装好即用。与之相对的是<strong>即时编译(JIT)</strong>——某些实验性或形状高度可变的 kernel 会在运行时再编(这是第39课的前瞻主题)。回想第4课讲过:解码阶段是<strong>带宽受限</strong>的,每生成一个 token 都要把权重和 KV 从显存搬一遍,核函数写得好不好,直接决定显存带宽用满了几成,也就直接决定吞吐。所以 sgl-kernel 不是锦上添花,而是整个推理引擎的性能地基。</div>

<h2>一、sgl-kernel 是一个独立工程,不是普通 Python 模块</h2>
<p>在仓库根目录下,<span class="mono">sgl-kernel/</span> 与 <span class="mono">python/sglang/</span> 是平级的两个世界。<span class="mono">python/sglang/</span> 是纯 Python 的推理框架;而 <span class="mono">sgl-kernel/</span> 自带一套 <span class="mono">CMakeLists.txt</span> 与 <span class="mono">pyproject.toml</span>,它的核心是 <span class="mono">csrc/</span> 目录——里面是成百上千行的 C++/CUDA 源码,实现了注意力、GEMM、MoE、量化等热路径上的核函数。这些源码经由 CMake 驱动 <span class="mono">nvcc</span> 与 C++ 编译器,被链接进<strong>一个</strong>共享库 <span class="mono">.so</span>。这个 <span class="mono">.so</span> 不会在用户机器上现编,而是在打 wheel 的时候就编好、打进包里。换句话说,你 <span class="mono">pip install sgl-kernel</span> 拿到的,是一份已经针对特定 CUDA 架构编译完成的二进制产物。这就是 AOT 的含义:编译发生在<strong>之前</strong>(发布时),而不是运行时。</p>
<p>为什么要把它独立出来,而不是塞进主框架?第一,语言与工具链完全不同:推理框架是 Python,核函数是 C++/CUDA,两者的构建系统、依赖、发布节奏都不一样,强行混在一起只会互相拖累。第二,编译产物可以单独发版:同一份 Python 框架可以搭配不同 CUDA 架构编出来的 <span class="mono">.so</span>,也方便针对 NVIDIA、AMD、摩尔线程等不同平台分别构建。第三,职责清晰:框架负责"调度与编排",kernel 工程负责"把单个算子算到极致",两边各自演进、互不打扰。理解了这一点,你就明白为什么改一行 Python 立刻生效,而改一行 CUDA 却要重新编译、重新打包整个 wheel——它们根本就是两条生产线。这种分离还带来一个工程上的好处:版本管理更清晰。<span class="mono">sgl-kernel</span> 作为独立的发布单元,有自己的版本号,主框架以依赖的方式锁定它的版本区间。这样一来,某个 kernel 修了 bug 或加了优化,只需发一个新的 sgl-kernel 小版本,主框架升级依赖即可,不必把整个推理框架重新发版;反过来,只想改 Python 侧的调度逻辑时,也完全不必碰那份编译好的 <span class="mono">.so</span>。这种"框架与 kernel 解耦发布"的设计,是大型推理系统能持续快速迭代的关键之一。</p>

<h2>二、csrc 里有什么:按功能分门别类</h2>
<p><span class="mono">csrc/</span> 不是一锅乱炖,而是按算子类别切成了清晰的子目录。<span class="mono">attention/</span> 放注意力相关的核函数(比如下面要剖析的 merge_state);<span class="mono">gemm/</span> 放各种矩阵乘,包括量化 GEMM;<span class="mono">moe/</span> 放专家混合的路由与分发 kernel;<span class="mono">elementwise/</span> 放归一化、激活这类逐元素操作;<span class="mono">allreduce/</span> 放自定义的多卡通信归约;<span class="mono">quantization/</span> 放权重与激活的量化/反量化;<span class="mono">grammar/</span> 放约束解码的语法处理;<span class="mono">speculative/</span> 放投机解码相关 kernel;<span class="mono">memory/</span> 与 <span class="mono">kvcacheio/</span> 处理显存与 KV 缓存的搬运;<span class="mono">cpu/</span> 则放 CPU 后端的实现。把所有这些粘合到 PyTorch 上的,是注册胶水文件 <span class="mono">common_extension.cc</span>(还有 <span class="mono">rocm</span>、<span class="mono">musa</span> 等针对 AMD、摩尔线程平台的变体),它用 <span class="mono">TORCH_LIBRARY_FRAGMENT</span> 把每个 C++ 函数声明成 PyTorch 自定义算子（用 FRAGMENT 变体，多个文件就能往同一个 <span class="mono">sgl_kernel</span> 命名空间里注册）。</p>
<p>这种按功能分目录的好处,是让你能<strong>顺着上层需求一眼找到对应 kernel</strong>。第33课讲注意力后端排版 KV,它要的合并、计算 kernel 就在 <span class="mono">attention/</span>;第34课讲 MoE 路由,对应的分发与聚合在 <span class="mono">moe/</span>;第35课的量化压缩落在 <span class="mono">quantization/</span> 与 <span class="mono">gemm/</span>;第36课的归一化与激活落在 <span class="mono">elementwise/</span>。多卡场景里,张量并行需要的跨卡归约在 <span class="mono">allreduce/</span> 有定制实现,比通用通信库更贴合推理的形状与时机。再加上投机解码、约束解码、KV 缓存搬运这些专门场景各有归处,整个 <span class="mono">csrc/</span> 就像一座按工种分区的工厂车间:你想优化哪一类算子,直接进对应车间即可,不必在一大锅源码里大海捞针。</p>
<p>还要留意一个细节:每个子目录里往往不止一个实现,而是同一类算子针对不同精度、不同硬件、不同形状的<strong>多个变体</strong>。例如 <span class="mono">gemm/</span> 里会同时有 FP16、BF16、FP8 乃至各种量化格式的矩阵乘;<span class="mono">attention/</span> 里 prefill 与 decode 两条路径(回顾第33课)所需的 kernel 也各不相同。注册胶水 <span class="mono">common_extension.cc</span> 的职责,就是把这些变体逐一登记到 <span class="mono">torch.ops.sgl_kernel</span> 命名空间,并为 ROCm(AMD)、MUSA(摩尔线程)等不同后端准备对应的变体文件,让同一个上层调用在不同硬件上落到正确的实现(第42课会专门讲多硬件后端)。正因为有这层清晰的目录与注册约定,sgl-kernel 才能在保持上层接口稳定的同时,容纳数量庞大、且仍在快速增长的 kernel 实现。</p>

<h2>三、从 Python 到 kernel:torch.ops 这条线缆</h2>
<p>核函数编译进 <span class="mono">.so</span> 之后,Python 怎么调它?答案是 PyTorch 的<strong>自定义算子机制</strong>。注册胶水把每个 kernel 挂到 <span class="mono">torch.ops.sgl_kernel</span> 这个命名空间下,于是在 Python 里写 <span class="mono">torch.ops.sgl_kernel.merge_state_v2.default(...)</span> 就能直接跳进编译好的 C++/CUDA 代码。但你通常不会裸调 <span class="mono">torch.ops</span>——在 <span class="mono">sgl-kernel/python/sgl_kernel/</span> 下还有一层<strong>薄薄的 Python 包装</strong>:<span class="mono">attention.py</span>、<span class="mono">gemm.py</span>、<span class="mono">moe.py</span> 等等。这些包装函数做三件小事:校验输入张量的形状与 dtype、按需分配输出张量、然后把活儿转交给 <span class="mono">torch.ops</span>。它们很薄,几乎不含算法逻辑,真正的计算全在 kernel 里。Part-8 那些层——注意力后端(第33课)、MoE(第34课)、量化(第35课)、归一化与激活(第36课)——正是通过这层薄包装,把性能敏感的步骤甩给 sgl-kernel 来跑得飞快。</p>
<p>为什么要多套一层薄包装,而不是让上层直接写 <span class="mono">torch.ops</span>?因为裸算子很"生":它不会替你检查 dtype 对不对、形状能不能对上,也不会替你分配输出缓冲。把这些琐碎但易错的前置工作收进薄包装,上层调用就变得干净又安全——传进张量、拿回结果,中间的校验与分配由包装兜住。更重要的是,这层包装给了一个<strong>稳定的 Python 接口</strong>:哪天底层 kernel 换了实现、改了参数顺序,只要包装函数签名不变,上层代码就一行都不用动。于是 Part-8 的各个后端只认这层薄薄的 Python 函数,而不必关心背后是 AOT 还是 JIT、是 NVIDIA 还是 AMD 的 kernel——这正是分层解耦的价值。</p>
<p>顺带说一句命名:<span class="mono">torch.ops.sgl_kernel</span> 里的 <span class="mono">sgl_kernel</span> 就是这个自定义算子库的"命名空间",它在注册时由 <span class="mono">TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)</span> 这一行确定;末尾的 <span class="mono">.default</span> 则是 PyTorch 为算子生成的默认重载入口。所以当你在源码里看到 <span class="mono">torch.ops.sgl_kernel.merge_state_v2.default(...)</span> 这一长串,它的含义其实很直白:在 <span class="mono">sgl_kernel</span> 命名空间下,找到名为 <span class="mono">merge_state_v2</span> 的自定义算子,调用它的默认实现。这套机制还顺带带来一个好处——这些算子能被 <span class="mono">torch.compile</span> 与 CUDA Graph(第27、41课)识别和捕获,从而参与图级别的优化与重放,而不是变成一个 PyTorch 看不透的黑盒外部调用。这就是为什么 SGLang 宁可走 PyTorch 自定义算子这条"正规路",也不用 <span class="mono">ctypes</span> 去裸调 <span class="mono">.so</span>。</p>

<h2>四、AOT 与 JIT:两条编译路线的取舍</h2>
<p>sgl-kernel 的主路是 <strong>AOT</strong>:稳定、通用、形状常见的 kernel 提前编好放进 wheel,启动零编译开销。但并非所有 kernel 都适合 AOT——有些 kernel 的最优实现强依赖具体形状或硬件特性,提前把所有组合都编出来会让 <span class="mono">.so</span> 体积爆炸、编译时间失控。这类 kernel 更适合 <strong>JIT</strong>:在运行时根据真实形状现编现用,第一次慢一点,之后缓存复用。第39课会专门前瞻 JIT 这条路;第40课则会带你一行行拆解一个真实的注意力 kernel,把本课"俯瞰"的视角下沉到"显微镜"级别。理解了 AOT 与 JIT 的分工,你才能明白为什么有的算子改一改立刻生效、有的却要重新打包整个 wheel。</p>
<p>把这条取舍记牢,对你日后调优很有用。当你发现某个算子是 AOT 编出来的,就要意识到:想改它,得回到 sgl-kernel 工程、改 C++/CUDA、重新编 <span class="mono">.so</span>、重打 wheel,迭代周期以"分钟到小时"计;而 JIT 的算子可以在 Python 侧快速试错,改完下次运行就生效。两条路并非对立,而是<strong>互补</strong>:把那些天天都跑、形状固定的主力 kernel 走 AOT 拿到零启动开销,把那些形状多变、还在打磨的实验 kernel 走 JIT 换取迭代速度。回到全局,正因为解码阶段是带宽受限(第4课)的,主力 kernel 的每一点带宽利用率都直接换算成吞吐,所以它们值得用 AOT 精心预编——这就是 sgl-kernel 作为性能地基的意义所在。</p>
<p>最后给一个判断小窍门:当你拿到一份 SGLang 安装,想知道某个算子走的是哪条路,可以看它从哪里来。如果调用最终落在 <span class="mono">torch.ops.sgl_kernel.*</span> 且对应实现已在随包的 <span class="mono">.so</span> 里,那就是 AOT;如果在第一次调用时才触发一段编译、生成临时模块再加载,那就是 JIT。两者在调用写法上可以几乎一样,差别只在"何时编译"。把这个差别放进脑子里,你日后无论是排查启动慢、还是定位某次改动为何没生效,都能很快锁定问题出在哪一层——是该重打 wheel,还是清一下 JIT 缓存。归根结底,sgl-kernel 这一课要你记住的不是某个具体函数,而是一整套"性能从哪里来"的心智模型:上层 Python 负责把事情安排清楚,底层原生 kernel 负责把每一步算到极致,而 <span class="mono">torch.ops</span> 这根线缆把两者牢牢接在一起。</p>

<div class="layers"><div class="layer">Python 推理层:attention/moe/quant 后端(第33–36课)</div><div class="layer">sgl_kernel 薄包装:attention.py / gemm.py / moe.py(校验+分配输出)</div><div class="layer">torch.ops.sgl_kernel.* 自定义算子命名空间(注册表)</div><div class="layer">csrc/ 里的 C++/CUDA kernel(编译进 .so,AOT)</div></div>

<table class="t"><tr><th>csrc 子目录</th><th>承载的核函数</th></tr>
<tr><td><span class="mono">attention/</span></td><td>注意力计算、split-KV 部分状态合并(merge_state)</td></tr>
<tr><td><span class="mono">gemm/</span></td><td>矩阵乘,含量化 GEMM</td></tr>
<tr><td><span class="mono">moe/</span></td><td>专家混合的路由与分发</td></tr>
<tr><td><span class="mono">elementwise/</span></td><td>逐元素:归一化、激活</td></tr>
<tr><td><span class="mono">allreduce/</span></td><td>自定义多卡通信归约</td></tr>
<tr><td><span class="mono">quantization/</span></td><td>量化与反量化</td></tr>
<tr><td><span class="mono">grammar/</span></td><td>约束解码的语法处理</td></tr>
<tr><td><span class="mono">speculative/</span></td><td>投机解码相关 kernel</td></tr>
<tr><td><span class="mono">memory/</span></td><td>显存管理与搬运</td></tr>
<tr><td><span class="mono">kvcacheio/</span></td><td>KV 缓存读写</td></tr>
<tr><td><span class="mono">cpu/</span></td><td>CPU 后端实现</td></tr>
<tr><td><span class="mono">common_extension.cc</span></td><td>注册胶水(+rocm/musa 变体)</td></tr></table>

<div class="cols"><div class="col"><b>AOT(本课主路)</b><br>发布打 wheel 时用 nvcc 提前编好;<span class="mono">.so</span> 随包发布;启动零编译开销;适合稳定、通用、形状常见的 kernel;改动需重新打包整个 wheel。</div><div class="col"><b>JIT(第39课前瞻)</b><br>运行时按真实形状现编现用;首次有编译延迟,之后缓存复用;适合形状高度可变或实验性的 kernel;改动可即时生效,不必重打 wheel。</div></div>

<div class="flow"><div class="node">CMake 驱动 nvcc 编译 csrc/</div><div class="arrow">→</div><div class="node">链接成 .so 打进 wheel</div><div class="arrow">→</div><div class="node">注册到 torch.ops.sgl_kernel.*</div><div class="arrow">→</div><div class="node">Python 薄包装调用 → kernel 执行</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/attention.py ::merge_state_v2</span><span class="ln">薄包装:dtype 转换 + 输出分配 → torch.ops kernel</span></div><pre>def merge_state_v2(v_a, s_a, v_b, s_b, v_merged=None, s_merged=None):
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # avoid allocating new tensors if outputs are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    # dispatch into the compiled C++/CUDA kernel registered as a torch op
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged</pre></div>
<p>这段代码是整个 sgl-kernel 调用模式的<strong>缩影</strong>。它只做三件事:第一,把状态张量 <span class="mono">s_a</span>、<span class="mono">s_b</span> 统一转成 <span class="mono">float32</span>(dtype 校验/规整);第二,如果调用方没传 <span class="mono">v_merged</span>、<span class="mono">s_merged</span>,就用 <span class="mono">torch.empty_like</span> 分配输出张量,否则复用已有缓冲、避免重复分配;第三,把所有张量交给 <span class="mono">torch.ops.sgl_kernel.merge_state_v2.default</span>,跳进编译好的 C++/CUDA kernel 真正算。包装薄如纸,算法在 kernel 里。它的语义是:把两段<strong>部分注意力状态</strong>(value 张量 v 和对应的 softmax 归一化标量 s)合并成一段,常用于把 split-KV(把一长串 KV 切成几段分别算)的部分结果重新拼回完整注意力输出。<strong>薄 Python 包装 → torch.ops.sgl_kernel.&lt;name&gt; → csrc 编译 kernel</strong>,这条线你会在 sgl-kernel 里看到无数遍。</p>
<p>再多看一眼这段代码的两个小细节,它们体现了 kernel 包装的工程考量。一是<strong>把 <span class="mono">s</span> 统一转成 <span class="mono">float32</span></strong>:部分注意力状态里的归一化标量对数值精度敏感,先升到 fp32 能避免合并时累积误差,这类"为正确性做的 dtype 规整"正适合放在 Python 薄包装里、而不是塞进每个 kernel。二是<strong>输出张量可由调用方传入</strong>:当上层已经准备好缓冲区时,包装就不再 <span class="mono">empty_like</span> 新分配,直接复用——在解码这种每步都要反复调用的热路径上,省下的每一次显存分配都是实打实的开销节约。把这两点放在一起看,你会更体会到"薄包装负责正确性与资源、kernel 负责极致计算"这条分工的精妙。</p>

<div class="fig">
  <svg viewBox="0 0 800 230" role="img" aria-label="一次 kernel 调用穿过三层：Python 薄包装校验形状并分配输出，转交 torch.ops.sgl_kernel 分发，落到 csrc 里编译好的 CUDA 核做真正的 GPU 计算，再把输出张量返回">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">一次调用 · 穿过三层</text>
    <rect x="24" y="52" width="212" height="84" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="130" y="84" text-anchor="middle" style="font-weight:700;fill:var(--blue)">Python 薄包装</text>
    <text x="130" y="106" text-anchor="middle" style="font-size:12px">校验形状</text>
    <text x="130" y="124" text-anchor="middle" style="font-size:12px">分配输出</text>
    <text x="248" y="100" text-anchor="middle" style="fill:var(--muted);font-size:20px">→</text>
    <rect x="272" y="52" width="232" height="84" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="388" y="84" text-anchor="middle" style="font-weight:700;fill:var(--amber)">torch.ops 分发</text>
    <text x="388" y="110" text-anchor="middle" class="mono" style="font-size:11px">sgl_kernel.*</text>
    <text x="516" y="100" text-anchor="middle" style="fill:var(--muted);font-size:20px">→</text>
    <rect x="540" y="52" width="236" height="84" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="658" y="84" text-anchor="middle" style="font-weight:700;fill:var(--teal)">csrc CUDA 核</text>
    <text x="658" y="106" text-anchor="middle" style="font-size:12px">真正的</text>
    <text x="658" y="124" text-anchor="middle" style="font-size:12px">GPU 计算</text>
    <line x1="658" y1="178" x2="146" y2="178" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="138,178 150,172 150,184" style="fill:var(--teal)"/>
    <text x="402" y="170" text-anchor="middle" style="fill:var(--teal);font-size:12px">返回输出张量</text>
  </svg>
  <div class="figcap"><b>图 1 · 一次调用穿过三层</b> — Python 薄包装校验形状、分配输出，转交 <span class="mono">torch.ops.sgl_kernel.*</span> 分发，落到 csrc 里编译好的 CUDA 核做真正的 GPU 计算，再把输出张量返回。</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 290" role="img" aria-label="AOT 与 JIT 两条到达 kernel 的路径：AOT 在 wheel 构建时提前编译、把 .so 打进 wheel、装好即用，成本在安装时；JIT 在首次调用时编译、缓存产物、之后复用，成本在首次调用">
    <line x1="400" y1="24" x2="400" y2="266" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="60" y="40" style="font-weight:700;fill:var(--blue)">AOT · 预编进 wheel</text>
    <rect x="60" y="56" width="280" height="46" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="200" y="84" text-anchor="middle" style="font-size:13px">wheel 构建时提前编译</text>
    <text x="200" y="118" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="60" y="128" width="280" height="46" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="200" y="156" text-anchor="middle" class="mono" style="font-size:12px">.so 打进 wheel</text>
    <text x="200" y="190" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="60" y="200" width="280" height="46" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="200" y="228" text-anchor="middle" style="font-weight:700;fill:var(--teal)">安装即用 · 零延迟</text>
    <text x="200" y="262" text-anchor="middle" style="fill:var(--faint);font-size:12px">成本在安装时</text>
    <text x="460" y="40" style="font-weight:700;fill:var(--amber)">JIT · 运行时编译</text>
    <rect x="460" y="56" width="280" height="46" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="600" y="84" text-anchor="middle" style="font-size:13px">首次调用时编译</text>
    <text x="600" y="118" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="460" y="128" width="280" height="46" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="600" y="156" text-anchor="middle" style="font-size:13px">缓存编译产物</text>
    <text x="600" y="190" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="460" y="200" width="280" height="46" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="600" y="228" text-anchor="middle" style="font-weight:700;fill:var(--purple)">之后缓存复用</text>
    <text x="600" y="262" text-anchor="middle" style="fill:var(--faint);font-size:12px">成本在首次调用</text>
  </svg>
  <div class="figcap"><b>图 2 · AOT vs JIT</b> — AOT 把重核函数在 wheel 构建时提前编好、以 <span class="mono">.so</span> 随包发布，装好即用；JIT 让较轻的核函数在首次调用时现编、缓存后复用。成本一个落在安装时，一个落在首次调用。</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/elementwise.py ::silu_and_mul</span><span class="ln">薄 Python 包装：分配输出 → 调 torch.ops 的 CUDA 核</span></div><pre>def silu_and_mul(input, out=None):
    # thin Python wrapper around the compiled CUDA kernel.
    if out is None:
        # SwiGLU: input is [..., 2h], output is [..., h]
        out = torch.empty(input.shape[:-1] + (input.shape[-1] // 2,),
                          device=input.device, dtype=input.dtype)
    torch.ops.sgl_kernel.silu_and_mul.default(out, input)  # -&gt; csrc CUDA
    return out</pre></div>
<p>举个具体例子:SwiGLU 前馈网络里,门控分支与数值分支被拼在同一个张量的最后一维。设输入是 <span class="mono">[tokens, 2h]</span>,<span class="mono">silu_and_mul</span> 在<strong>一个融合 kernel</strong> 里完成"前一半过 SiLU、再与后一半逐元素相乘",直接吐出 <span class="mono">[tokens, h]</span>——全程不落任何中间临时张量,省下一次显存往返。这正是上面那段薄包装的活法:Python 层只按 <span class="mono">input.shape[-1] // 2</span> 把输出张量分配好,真正的逐元素计算全压进 csrc 里那个 CUDA 核。末尾的 <span class="mono">.default</span> 不是随手写的后缀,而是注册在 <span class="mono">torch.ops.sgl_kernel</span> 命名空间下、那个真正用 C++ 实现的算子<strong>默认重载</strong>入口——和第一段 merge_state 里看到的 <span class="mono">.default</span> 是同一套机制。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>独立工程</strong>:<span class="mono">sgl-kernel/</span> 有自己的 CMake,与 <span class="mono">python/sglang/</span> 平级,核心是 <span class="mono">csrc/</span> 里的 C++/CUDA 源码。</li>
<li><strong>AOT 编译</strong>:热路径 kernel 提前编进一个 <span class="mono">.so</span>,随 wheel 发布,启动零编译开销;对照 JIT(第39课)运行时现编。</li>
<li><strong>torch 自定义算子</strong>:kernel 经 <span class="mono">common_extension.cc</span> 注册成 <span class="mono">torch.ops.sgl_kernel.*</span>,Python 由此跳进原生代码。</li>
<li><strong>薄包装三件事</strong>:<span class="mono">sgl_kernel/python/</span> 下的 attention/gemm/moe 包装只做形状/dtype 校验、分配输出、转交 torch.ops。</li>
<li><strong>谁在用</strong>:Part-8 的注意力(第33课)、MoE(第34课)、量化(第35课)、norm/激活(第36课)都靠它提速;解码带宽受限(第4课)使 kernel 质量直接决定吞吐。</li>
<li><strong>承上启下</strong>:第39课讲 JIT,第40课逐行拆解真实注意力 kernel。</li>
</ul></div>
""", "en": r"""
<p class="lead">For an entire Part-8 we talked "algorithms" at the Python level: how the attention backend lays out the KV (Lesson 33), how MoE routes (Lesson 34), how quantization compresses weights (Lesson 35), how normalization and activation are computed (Lesson 36). But all of that Python eventually has to land on real machine code that runs blazingly fast on the GPU. <strong>sgl-kernel</strong> is where that machine code lives — it is a <strong>separate C++/CUDA project</strong> with its own <span class="mono">CMakeLists.txt</span>, which compiles the hot-path kernels <strong>ahead of time (AOT)</strong> into a single <span class="mono">.so</span> shared library shipped inside the wheel. This lesson gives you the bird's-eye view: how the project is organized, how the kernels are registered as PyTorch custom ops, and how Python calls them through <span class="mono">torch.ops.sgl_kernel.*</span>.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>Think of the main SGLang project as a fully assembled car, with the Python code being the dashboard and steering wheel — you turn the wheel and the car turns. <strong>sgl-kernel</strong> is the engine factory: on a separate production line, with a completely different craft (C++/CUDA, CMake), it <strong>builds the engine in advance</strong>, crates it up (into a <span class="mono">.so</span>) and ships it to the assembly plant. By the time you slide into the driver's seat the engine is already waiting under the hood; you just press the pedal (call <span class="mono">torch.ops.sgl_kernel.merge_state_v2</span>) without caring how the pistons and cylinders were cast. Every button on the dashboard (a thin Python wrapper) is wired (the torch op registry) to a specific part inside the engine (a kernel in <span class="mono">csrc/</span>).</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>Why a separate project just for kernels? Because GPU kernel compilation marches to a completely different beat than Python execution. Python is interpreted and runs as you edit; a CUDA kernel must go through <span class="mono">nvcc</span> compilation and linking, taking minutes. If every service start had to compile on the fly, the latency would be unbearable. So SGLang compiles the most common, most stable kernels <strong>ahead of time (AOT)</strong> and packs them into the <span class="mono">.so</span> in the wheel — install and go. The contrast is <strong>just-in-time (JIT)</strong> compilation, where some experimental or highly shape-variable kernels are compiled at runtime (the forward-looking topic of Lesson 39). Recall Lesson 4: the decode phase is <strong>bandwidth-bound</strong> — every generated token re-reads weights and KV from device memory, so how well a kernel is written directly decides how much of the memory bandwidth gets saturated, and thus the throughput. sgl-kernel is not icing on the cake; it is the performance foundation of the whole inference engine.</div>
<h2>1. sgl-kernel is a standalone project, not an ordinary Python module</h2>
<p>At the repo root, <span class="mono">sgl-kernel/</span> and <span class="mono">python/sglang/</span> are two sibling worlds. <span class="mono">python/sglang/</span> is the pure-Python inference framework; <span class="mono">sgl-kernel/</span> carries its own <span class="mono">CMakeLists.txt</span> and <span class="mono">pyproject.toml</span>, and its heart is the <span class="mono">csrc/</span> directory — hundreds of thousands of lines of C++/CUDA that implement the hot-path kernels for attention, GEMM, MoE, quantization and more. Driven by CMake, <span class="mono">nvcc</span> and the C++ compiler link this source into <strong>one</strong> shared library, the <span class="mono">.so</span>. That <span class="mono">.so</span> is not compiled on the user's machine; it is built and packed at wheel-build time. In other words, when you <span class="mono">pip install sgl-kernel</span> you receive a binary artifact already compiled for a specific CUDA architecture. That is what AOT means: compilation happens <strong>beforehand</strong> (at release time), not at runtime.</p>
<p>Why pull it out instead of folding it into the main framework? First, the languages and toolchains differ entirely: the framework is Python, the kernels are C++/CUDA, and their build systems, dependencies, and release cadences all differ — forcing them together only drags each down. Second, the build artifact can be released on its own: the same Python framework can pair with <span class="mono">.so</span> files compiled for different CUDA architectures, and it is easy to build separately for NVIDIA, AMD, Moore Threads, and other platforms. Third, responsibilities are clear: the framework handles "scheduling and orchestration," the kernel project handles "computing a single op to the limit," and the two evolve independently. This separation also brings cleaner version management: <span class="mono">sgl-kernel</span> is a standalone release unit with its own version number, which the main framework pins as a dependency — a kernel fix or optimization just ships a new sgl-kernel patch release, no need to re-release the whole framework, and a pure-Python scheduling change need not touch the compiled <span class="mono">.so</span> at all.</p>

<h2>2. What lives in csrc: organized by function</h2>
<p><span class="mono">csrc/</span> is not a single soup but is cut into clear subdirectories by op category. <span class="mono">attention/</span> holds attention kernels (such as the merge_state dissected below); <span class="mono">gemm/</span> holds matrix multiplies, including quantized GEMM; <span class="mono">moe/</span> holds the routing and dispatch kernels for mixture-of-experts; <span class="mono">elementwise/</span> holds per-element ops like normalization and activation; <span class="mono">allreduce/</span> holds custom multi-GPU communication reductions; <span class="mono">quantization/</span> holds weight/activation quantize and dequantize; <span class="mono">grammar/</span> holds grammar handling for constrained decoding; <span class="mono">speculative/</span> holds speculative-decoding kernels; <span class="mono">memory/</span> and <span class="mono">kvcacheio/</span> handle device-memory and KV-cache movement; and <span class="mono">cpu/</span> holds the CPU-backend implementations. What glues all of these onto PyTorch is the registration file <span class="mono">common_extension.cc</span> (plus <span class="mono">rocm</span> and <span class="mono">musa</span> variants for AMD and Moore Threads platforms), which uses <span class="mono">TORCH_LIBRARY_FRAGMENT</span> to declare each C++ function as a PyTorch custom op (the FRAGMENT variant lets several files register into the same <span class="mono">sgl_kernel</span> namespace).</p>
<p>The benefit of this by-function layout is that you can <strong>find the matching kernel straight from an upper-level need</strong>. Lesson 33's attention backend laying out KV wants its merge and compute kernels right there in <span class="mono">attention/</span>; Lesson 34's MoE routing maps to dispatch/combine in <span class="mono">moe/</span>; Lesson 35's quantization lands in <span class="mono">quantization/</span> and <span class="mono">gemm/</span>; Lesson 36's norm and activation land in <span class="mono">elementwise/</span>. Note one more detail: each subdirectory often holds not one implementation but <strong>several variants</strong> of the same op class for different precisions, hardware, and shapes — <span class="mono">gemm/</span> carries FP16, BF16, FP8 and quantized matmuls at once, and <span class="mono">attention/</span>'s prefill and decode paths need different kernels. The registration glue registers each variant into <span class="mono">torch.ops.sgl_kernel</span> and prepares ROCm/MUSA variant files so the same upper call lands on the right implementation per hardware (Lesson 42). It is this clean directory-and-registration convention that lets sgl-kernel keep a stable interface while housing a large, still-growing set of kernels.</p>

<h2>3. From Python to kernel: the torch.ops wire</h2>
<p>Once kernels are compiled into the <span class="mono">.so</span>, how does Python call them? Through PyTorch's <strong>custom-op mechanism</strong>. The registration glue hangs each kernel under the <span class="mono">torch.ops.sgl_kernel</span> namespace, so writing <span class="mono">torch.ops.sgl_kernel.merge_state_v2.default(...)</span> in Python jumps straight into the compiled C++/CUDA code. But you rarely call <span class="mono">torch.ops</span> bare — under <span class="mono">sgl-kernel/python/sgl_kernel/</span> there is a layer of <strong>thin Python wrappers</strong>: <span class="mono">attention.py</span>, <span class="mono">gemm.py</span>, <span class="mono">moe.py</span> and so on. These wrapper functions do three small things: validate the shapes and dtypes of input tensors, allocate output tensors as needed, then hand the work off to <span class="mono">torch.ops</span>. They are thin, containing almost no algorithmic logic — the real computation is all in the kernel. The Part-8 backends — attention backend (Lesson 33), MoE (Lesson 34), quantization (Lesson 35), norm/activation (Lesson 36) — use exactly this thin layer to offload performance-sensitive steps to sgl-kernel for speed.</p>
<p>A note on naming: the <span class="mono">sgl_kernel</span> in <span class="mono">torch.ops.sgl_kernel</span> is this custom-op library's namespace, fixed at registration by the line <span class="mono">TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)</span>; the trailing <span class="mono">.default</span> is the default overload entry PyTorch generates for the op. So when you see the long <span class="mono">torch.ops.sgl_kernel.merge_state_v2.default(...)</span> in the source, its meaning is plain: under the <span class="mono">sgl_kernel</span> namespace, find the custom op named <span class="mono">merge_state_v2</span> and call its default implementation. This mechanism also brings a bonus — these ops can be recognized and captured by <span class="mono">torch.compile</span> and CUDA Graph (Lessons 27, 41), taking part in graph-level optimization and replay rather than becoming an opaque external call PyTorch cannot see through. That is why SGLang prefers the "proper road" of PyTorch custom ops over a bare <span class="mono">ctypes</span> dlopen of the <span class="mono">.so</span>.</p>

<h2>4. AOT vs JIT: the trade-off between two compile paths</h2>
<p>sgl-kernel's main road is <strong>AOT</strong>: stable, general, common-shape kernels are compiled ahead and packed into the wheel, with zero compile overhead at startup. But not every kernel suits AOT — some kernels' optimal implementation depends heavily on a specific shape or hardware feature, and compiling all combinations ahead would blow up the <span class="mono">.so</span> size and the build time. Such kernels suit <strong>JIT</strong> better: compiled at runtime against the real shape, slower the first time, then cached and reused. Lesson 39 looks ahead at the JIT path specifically; Lesson 40 walks you line by line through a real attention kernel, dropping this lesson's "bird's-eye" view down to a "microscope" level. Once you grasp the division of labor between AOT and JIT, you understand why some ops take effect the moment you tweak them while others require re-packing the whole wheel.</p>
<p>A handy rule of thumb: given an SGLang install, to know which path an op takes, look at where it comes from. If the call lands on <span class="mono">torch.ops.sgl_kernel.*</span> and the implementation already lives in the shipped <span class="mono">.so</span>, it is AOT; if the first call triggers a compile that generates a temporary module and loads it, it is JIT. The two can look almost identical at the call site — the only difference is "when compilation happens." Keep that difference in mind and, whether you are chasing a slow startup or figuring out why a change did not take effect, you can quickly pin which layer the problem is in — whether to re-pack the wheel or to clear the JIT cache.</p>

<div class="layers"><div class="layer">Python inference layer: attention/moe/quant backends (Lessons 33–36)</div><div class="layer">sgl_kernel thin wrappers: attention.py / gemm.py / moe.py (validate + allocate output)</div><div class="layer">torch.ops.sgl_kernel.* custom-op namespace (registry)</div><div class="layer">C++/CUDA kernels in csrc/ (compiled into the .so, AOT)</div></div>

<table class="t"><tr><th>csrc subdir</th><th>kernels it holds</th></tr>
<tr><td><span class="mono">attention/</span></td><td>attention compute, split-KV partial-state merge (merge_state)</td></tr>
<tr><td><span class="mono">gemm/</span></td><td>matrix multiply, including quantized GEMM</td></tr>
<tr><td><span class="mono">moe/</span></td><td>mixture-of-experts routing and dispatch</td></tr>
<tr><td><span class="mono">elementwise/</span></td><td>per-element: normalization, activation</td></tr>
<tr><td><span class="mono">allreduce/</span></td><td>custom multi-GPU communication reduction</td></tr>
<tr><td><span class="mono">quantization/</span></td><td>quantize and dequantize</td></tr>
<tr><td><span class="mono">grammar/</span></td><td>grammar handling for constrained decoding</td></tr>
<tr><td><span class="mono">speculative/</span></td><td>speculative-decoding kernels</td></tr>
<tr><td><span class="mono">memory/</span></td><td>device-memory management and movement</td></tr>
<tr><td><span class="mono">kvcacheio/</span></td><td>KV-cache read/write</td></tr>
<tr><td><span class="mono">cpu/</span></td><td>CPU-backend implementations</td></tr>
<tr><td><span class="mono">common_extension.cc</span></td><td>registration glue (+rocm/musa variants)</td></tr></table>

<div class="cols"><div class="col"><b>AOT (this lesson's main road)</b><br>Compiled ahead with nvcc at wheel-build time; the <span class="mono">.so</span> ships with the package; zero compile overhead at startup; suits stable, general, common-shape kernels; a change requires re-packing the whole wheel.</div><div class="col"><b>JIT (Lesson 39 preview)</b><br>Compiled at runtime against the real shape; a compile delay the first time, then cached and reused; suits highly shape-variable or experimental kernels; a change can take effect instantly without re-packing the wheel.</div></div>

<div class="flow"><div class="node">CMake drives nvcc to compile csrc/</div><div class="arrow">→</div><div class="node">link into .so, pack into wheel</div><div class="arrow">→</div><div class="node">register as torch.ops.sgl_kernel.*</div><div class="arrow">→</div><div class="node">thin Python wrapper calls → kernel runs</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/attention.py ::merge_state_v2</span><span class="ln">thin wrapper: dtype cast + output allocation → torch.ops kernel</span></div><pre>def merge_state_v2(v_a, s_a, v_b, s_b, v_merged=None, s_merged=None):
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # avoid allocating new tensors if outputs are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    # dispatch into the compiled C++/CUDA kernel registered as a torch op
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged</pre></div>
<p>This snippet is a <strong>microcosm</strong> of the whole sgl-kernel calling pattern. It does just three things: first, cast the state tensors <span class="mono">s_a</span> and <span class="mono">s_b</span> to <span class="mono">float32</span> (dtype validation/normalization); second, if the caller did not pass <span class="mono">v_merged</span> and <span class="mono">s_merged</span>, allocate the outputs with <span class="mono">torch.empty_like</span>, otherwise reuse the provided buffers to avoid re-allocation; third, hand all tensors to <span class="mono">torch.ops.sgl_kernel.merge_state_v2.default</span>, jumping into the compiled C++/CUDA kernel that does the real work. The wrapper is paper-thin; the algorithm is in the kernel. Its semantics: merge two <strong>partial attention states</strong> (a value tensor v and its corresponding softmax-normalization scalar s) into one, commonly used to stitch split-KV partial results (a long KV run cut into segments computed separately) back into a complete attention output. <strong>Thin Python wrapper → torch.ops.sgl_kernel.&lt;name&gt; → compiled csrc kernel</strong> — a wire you will see countless times across sgl-kernel.</p>
<p>Two small details in this code reflect the engineering thinking of a kernel wrapper. One is <strong>casting <span class="mono">s</span> to <span class="mono">float32</span></strong>: the normalization scalar in a partial attention state is sensitive to numeric precision, and promoting to fp32 first avoids accumulated error during the merge — this kind of "dtype normalization for correctness" belongs in the thin Python wrapper, not stuffed into every kernel. The other is that <strong>output tensors can be passed in by the caller</strong>: when the upper layer already has buffers ready, the wrapper skips a fresh <span class="mono">empty_like</span> allocation and reuses them — on the decode hot path that calls this repeatedly every step, every saved allocation is a real cost reduction. Seen together, they sharpen the elegant division of labor: the thin wrapper owns correctness and resources, the kernel owns the all-out computation.</p>

<div class="fig">
  <svg viewBox="0 0 800 230" role="img" aria-label="One kernel call crosses three layers: a thin Python wrapper validates shapes and allocates the output, hands off to the torch.ops.sgl_kernel dispatcher, which lands on the compiled CUDA kernel in csrc that does the real GPU work, then returns the output tensor">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">one call · three layers</text>
    <rect x="24" y="52" width="212" height="84" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="130" y="84" text-anchor="middle" style="font-weight:700;fill:var(--blue)">Python wrapper</text>
    <text x="130" y="106" text-anchor="middle" style="font-size:12px">check shapes</text>
    <text x="130" y="124" text-anchor="middle" style="font-size:12px">alloc output</text>
    <text x="248" y="100" text-anchor="middle" style="fill:var(--muted);font-size:20px">→</text>
    <rect x="272" y="52" width="232" height="84" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="388" y="84" text-anchor="middle" style="font-weight:700;fill:var(--amber)">torch.ops dispatch</text>
    <text x="388" y="110" text-anchor="middle" class="mono" style="font-size:11px">sgl_kernel.*</text>
    <text x="516" y="100" text-anchor="middle" style="fill:var(--muted);font-size:20px">→</text>
    <rect x="540" y="52" width="236" height="84" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="658" y="84" text-anchor="middle" style="font-weight:700;fill:var(--teal)">csrc CUDA kernel</text>
    <text x="658" y="106" text-anchor="middle" style="font-size:12px">real</text>
    <text x="658" y="124" text-anchor="middle" style="font-size:12px">GPU work</text>
    <line x1="658" y1="178" x2="146" y2="178" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:5 4"/>
    <polygon points="138,178 150,172 150,184" style="fill:var(--teal)"/>
    <text x="402" y="170" text-anchor="middle" style="fill:var(--teal);font-size:12px">returns output tensor</text>
  </svg>
  <div class="figcap"><b>Fig 1 · One call crosses three layers</b> — the thin Python wrapper checks shapes and allocates the output, hands off to the <span class="mono">torch.ops.sgl_kernel.*</span> dispatcher, lands on the compiled CUDA kernel in csrc that does the real GPU work, then returns the output tensor.</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 290" role="img" aria-label="Two paths to reach a kernel: AOT compiles ahead at wheel-build time, packs the .so into the wheel, and is ready instantly, with cost paid at install time; JIT compiles on first call, caches the module, and reuses it after, with cost paid at first call">
    <line x1="400" y1="24" x2="400" y2="266" style="stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 5"/>
    <text x="60" y="40" style="font-weight:700;fill:var(--blue)">AOT · .so in the wheel</text>
    <rect x="60" y="56" width="280" height="46" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="200" y="84" text-anchor="middle" style="font-size:13px">compile ahead at build</text>
    <text x="200" y="118" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="60" y="128" width="280" height="46" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="200" y="156" text-anchor="middle" class="mono" style="font-size:12px">.so packed in wheel</text>
    <text x="200" y="190" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="60" y="200" width="280" height="46" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="200" y="228" text-anchor="middle" style="font-weight:700;fill:var(--teal)">ready instantly</text>
    <text x="200" y="262" text-anchor="middle" style="fill:var(--faint);font-size:12px">cost paid at install</text>
    <text x="460" y="40" style="font-weight:700;fill:var(--amber)">JIT · compiled at runtime</text>
    <rect x="460" y="56" width="280" height="46" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="600" y="84" text-anchor="middle" style="font-size:13px">compile on first call</text>
    <text x="600" y="118" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="460" y="128" width="280" height="46" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="600" y="156" text-anchor="middle" style="font-size:13px">cache the module</text>
    <text x="600" y="190" text-anchor="middle" style="fill:var(--muted);font-size:18px">↓</text>
    <rect x="460" y="200" width="280" height="46" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="600" y="228" text-anchor="middle" style="font-weight:700;fill:var(--purple)">reuse after</text>
    <text x="600" y="262" text-anchor="middle" style="fill:var(--faint);font-size:12px">cost paid at first call</text>
  </svg>
  <div class="figcap"><b>Fig 2 · AOT vs JIT</b> — AOT compiles the heavy kernels ahead at wheel-build time and ships them as a <span class="mono">.so</span> in the wheel, ready instantly; JIT compiles the lighter kernels on first use and caches them for reuse. The cost lands at install time versus at first call.</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/elementwise.py ::silu_and_mul</span><span class="ln">thin Python wrapper: alloc output → call the torch.ops CUDA kernel</span></div><pre>def silu_and_mul(input, out=None):
    # thin Python wrapper around the compiled CUDA kernel.
    if out is None:
        # SwiGLU: input is [..., 2h], output is [..., h]
        out = torch.empty(input.shape[:-1] + (input.shape[-1] // 2,),
                          device=input.device, dtype=input.dtype)
    torch.ops.sgl_kernel.silu_and_mul.default(out, input)  # -&gt; csrc CUDA
    return out</pre></div>
<p>A concrete example: in a SwiGLU feed-forward network the gate branch and value branch are concatenated along the last dimension of a single tensor. Given an input of <span class="mono">[tokens, 2h]</span>, <span class="mono">silu_and_mul</span> does "run SiLU on the first half, then multiply elementwise by the second half" inside <strong>one fused kernel</strong> and emits <span class="mono">[tokens, h]</span> directly — no intermediate temp tensor anywhere, saving a round trip to device memory. That is exactly how the thin wrapper above lives: the Python layer only allocates the output by <span class="mono">input.shape[-1] // 2</span>, while the real elementwise compute is pushed entirely into that CUDA kernel in csrc. The trailing <span class="mono">.default</span> is not a casual suffix but the <strong>default overload</strong> entry of the registered C++ op under the <span class="mono">torch.ops.sgl_kernel</span> namespace — the same mechanism as the <span class="mono">.default</span> you saw in the merge_state snippet above.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>Standalone project</strong>: <span class="mono">sgl-kernel/</span> has its own CMake, sits as a sibling of <span class="mono">python/sglang/</span>, and is centered on the C++/CUDA source in <span class="mono">csrc/</span>.</li>
<li><strong>AOT compilation</strong>: hot-path kernels are compiled ahead into one <span class="mono">.so</span> shipped with the wheel, zero compile overhead at startup; contrast with JIT (Lesson 39) compiled at runtime.</li>
<li><strong>torch custom ops</strong>: kernels are registered via <span class="mono">common_extension.cc</span> as <span class="mono">torch.ops.sgl_kernel.*</span>, the door through which Python jumps into native code.</li>
<li><strong>Thin wrapper, three jobs</strong>: the attention/gemm/moe wrappers under <span class="mono">sgl_kernel/python/</span> only validate shapes/dtypes, allocate outputs, and forward to torch.ops.</li>
<li><strong>Who uses it</strong>: Part-8's attention (Lesson 33), MoE (Lesson 34), quantization (Lesson 35), norm/activation (Lesson 36) all rely on it for speed; decode being bandwidth-bound (Lesson 4) makes kernel quality directly decide throughput.</li>
<li><strong>What's next</strong>: Lesson 39 covers JIT, Lesson 40 dissects a real attention kernel line by line.</li>
</ul></div>
"""}
LESSON_39 = {"zh": r"""
<p class="lead">第38课我们看到 <span class="mono">sgl-kernel</span> 走的是 AOT（提前编译）路线：核在打包 wheel 时就被 <span class="mono">nvcc</span> 编进二进制。本课的主角 <span class="mono">python/sglang/jit_kernel/</span> 走的是另一条路——<strong>运行时按需即时编译</strong>（JIT，Just-In-Time）。它不在 wheel 里预先塞进每一个核，而是等到真正需要时，才在本机现场把 C++/CUDA 源码编译成 <span class="mono">.so</span>，然后把结果缓存起来反复复用。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div> AOT 像是工厂出货前就把所有口味的蛋糕全部烤好装箱：你买到就能吃，但箱子又大又重，而且工厂得提前猜你要哪些口味。JIT 则像家里的烤箱：第一次想吃某个口味时，你现场按配方烤一炉，要等一会儿；但烤好之后放进冰箱，之后再想吃就直接取出来，和现成的一样快。烤箱（编译工具链）必须在你家里，配方（源码）也得带着，但你不用背着一整箱蛋糕到处走。</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div> 无论 AOT 还是 JIT，最终都变成一组可以从 Python 直接调用、指向已编译 GPU 核的 <span class="mono">torch.ops</span> 风格调用。两者的差别不在“结果是什么”，而在“何时、如何被构建与分发”。AOT 把编译成本前置到发布阶段、换取开箱即用；JIT 把编译成本推迟到首次调用、换取灵活性与更瘦的 wheel。SGLang 同时拥有这两条路径，让稳定高频的核走 AOT、实验性或需按架构定制的核走 JIT。</div>

<h2>一、什么是 JIT 内核路径</h2>
<p><span class="mono">jit_kernel</span> 模块的核心思想是：把小巧的 C++/CUDA 核当作“源码 + 配方”随包携带，而不是预先编译进二进制。当某个算子第一次被用到时，模块才现场调用编译器把它构建成一个 PyTorch 扩展模块。这样做的直接好处是 wheel 不会因为塞进各种架构、各种数据类型的核而急剧膨胀。比如 <span class="mono">jit_kernel/activation.py</span> 会在首次使用时，按当前的数据类型现场构建一个对应的 JIT 模块；<span class="mono">jit_kernel/norm.py</span> 则提供了一个 JIT 版的 <span class="mono">fused_add_rmsnorm</span>，正是第36课讲 RMSNorm 快速路径时所依赖的那个融合核。</p>
<p>这里值得展开说说“wheel 膨胀”的来由。一个核往往不是只有一个版本：它可能要为 fp16、bf16、fp8 等多种数据类型各编一份，还可能为不同的 GPU 计算能力（如 sm80、sm90）各编一份，再叠加上各种形状或开关的组合，编译产物会以乘法的方式急剧增长。如果全走 AOT，这些组合都得提前枚举并塞进 <span class="mono">.so</span>，wheel 体积会迅速变得难以接受，而其中绝大多数组合在某次具体部署里根本用不到。JIT 的思路恰好相反：发布物里只放精简的源码，真正要用哪个组合，就在现场为那一个组合编译并缓存——把空间成本换成了一次性的时间成本，且只为实际跑到的代码路径付费。</p>
<p>这两个模块很能说明 JIT 的“按需”特性。<span class="mono">activation.py</span> 并不是一上来就把所有数据类型的激活核都编出来，而是等到某个具体 dtype 第一次被用到，才为它构建对应的 JIT 模块——用到 fp16 就编 fp16 的，用到 bf16 就编 bf16 的，没用到的就根本不花编译时间。这种“谁触发、才编谁”的策略，把编译成本精确地花在真正会跑的代码路径上。<span class="mono">norm.py</span> 的 <span class="mono">fused_add_rmsnorm</span> 则是一个典型的融合核：它把“残差相加”与“RMSNorm 归一化”两步合进一个 GPU 核里，少一次显存往返。第36课的 RMSNorm 快速路径正是调用了它——这说明 JIT 路径并不是边角料，而是实实在在地服务于主干推理流程中的关键算子。</p>
<p>为什么不全部走 AOT？因为现实中很多核并不“稳定且通用”。有的算子还在实验阶段、接口经常变；有的核需要根据 GPU 架构生成不同的代码（JIT 路径会查询当前架构，例如通过 <span class="mono">get_jit_cuda_arch</span> 拿到计算能力），再据此选择指令与模板；还有的核要用运行期才知道的形状或开关来参数化。把这些都提前枚举进 wheel 既不现实也太臃肿，交给 JIT 在现场按需生成才是更合适的策略。</p>
<p>换个角度看，AOT 与 JIT 并不是互相取代的对手，而是分工协作的两条流水线。SGLang 把那些被高频调用、形态固定、对启动延迟敏感的核交给 <span class="mono">sgl-kernel</span> 提前编进二进制；把那些还在演化、依赖具体硬件特性、或只在特定数据类型/形状下才会触发的核留给 <span class="mono">jit_kernel</span> 现场构建。这种“按核的生命周期与稳定程度分流”的设计，让发布物保持精简，又不牺牲对前沿硬件与新算子的快速适配能力。理解 JIT 路径，本质上就是理解“把编译这件事从发布阶段挪到运行阶段”所带来的全部好处与代价。</p>

<h2>二、load_jit 的工作机制：编译一次，之后复用</h2>
<p>整个 JIT 路径的统一入口是 <span class="mono">load_jit(...)</span>。你给它一组 C++/CUDA 源码文件，再给它一组“包装器”（wrapper）：每个包装器是一个 <span class="mono">(export_name, kernel_name)</span> 二元组，把一个面向 Python 的可调用名字，映射到底层某个 C++/CUDA 核类。<span class="mono">load_jit</span> 据此编译出一个 torch 扩展模块，并且把构建好的 <span class="mono">.so</span> 按一个唯一标记（marker）缓存起来。除了源码与包装器，它还接受一些可选参数：额外的编译标志、额外的头文件搜索路径、外部依赖（例如 <span class="mono">cutlass</span>）、构建目录等，让现场编译能对接到所需的第三方库与定制选项。</p>
<p>关键在于这个缓存：<strong>第一次调用要付一次性的编译成本</strong>（真正跑一遍 <span class="mono">nvcc</span>，可能要几秒到几十秒），但只要标记命中，<strong>后续调用就直接复用缓存里的 <span class="mono">.so</span></strong>，完全不再重新编译。也就是说，第一次慢，之后就和 AOT 一样快。这个“编译一次、之后永远复用”的设计，是 JIT 在保留灵活性的同时还能保持高性能的根本原因。唯一标记保证了不同的核、不同的参数组合各自对应各自的缓存条目，互不串味——当你改了源码、换了架构或切换了关键编译开关，标记随之改变，于是会触发一次新的编译并生成一份新的缓存，而不会错误地复用旧产物。</p>
<p>从调用者的视角看，这一切几乎是透明的：你只管调用由包装器导出的那个 Python 名字，第一次会“卡”一下（因为在编译），之后就顺畅得感觉不到 JIT 的存在。<span class="mono">load_jit</span> 最终返回的是已加载的扩展 Module，里面挂着按 <span class="mono">export_name</span> 暴露出来的可调用核；上层代码拿到它，就像调用任何普通的已编译算子一样使用。</p>
<p>这里还有一个容易被忽略的细节：缓存通常落在磁盘上的某个构建目录，因此“编译一次、之后复用”的好处不只局限于单次进程内，跨进程、甚至跨多次启动也能受益。只要源码、架构与关键编译选项没变（也就是唯一标记不变），新进程第一次用到这个核时，会直接发现磁盘上已经有现成的 <span class="mono">.so</span>，于是跳过编译直接加载。这意味着在同一台机器上反复启动服务时，真正付出 <span class="mono">nvcc</span> 编译成本的往往只有最初的那一次；之后无论重启多少回，都是秒级加载。理解这一层，你就明白为什么 JIT 在生产里其实没那么“可怕”——首调延迟是一次性的，而缓存是持久的。</p>

<div class="fig">
  <svg viewBox="0 0 800 270" role="img" aria-label="JIT 调用时间线：第一次调用付一次编译成本并把 .so 写入缓存，之后每次调用都命中缓存直接运行">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">首次编译并缓存 .so，之后每次调用直接复用</text>
    <rect x="470" y="17" width="12" height="12" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="488" y="27" style="fill:var(--muted);font-size:12px">编译</text>
    <rect x="556" y="17" width="12" height="12" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="574" y="27" style="fill:var(--muted);font-size:12px">运行</text>
    <rect x="300" y="44" width="150" height="40" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="375" y="62" text-anchor="middle" style="font-size:12px;font-weight:700">缓存目录</text>
    <text x="375" y="78" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:11px">.so</text>
    <line x1="40" y1="210" x2="770" y2="210" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M770 210 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="700" y="232" style="fill:var(--faint);font-size:11px">时间 →</text>
    <rect x="70" y="92" width="90" height="78" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="70" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="115" y="136" text-anchor="middle" style="font-size:12px;font-weight:700">编译 nvcc</text>
    <text x="115" y="194" text-anchor="middle" style="font-size:11px">运行</text>
    <text x="115" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">第 1 次</text>
    <path d="M160 116 L300 64" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <text x="232" y="98" text-anchor="middle" style="fill:var(--muted);font-size:11px">写入</text>
    <rect x="300" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="345" y="194" text-anchor="middle" style="font-size:11px">运行</text>
    <text x="345" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">第 2 次</text>
    <rect x="460" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="505" y="194" text-anchor="middle" style="font-size:11px">运行</text>
    <text x="505" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">第 3 次</text>
    <rect x="620" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="194" text-anchor="middle" style="font-size:11px">运行</text>
    <text x="665" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">第 4 次</text>
    <path d="M345 84 L345 170" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <path d="M420 84 L505 170" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <path d="M440 84 L665 170" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <text x="560" y="120" text-anchor="middle" style="fill:var(--muted);font-size:11px">命中缓存 · 复用 .so</text>
  </svg>
  <div class="figcap"><b>图 1 · 首次编译、之后复用</b> — 第一次调用某个 JIT 算子要付一次性的 <span class="mono">nvcc</span> 编译成本，并把 <span class="mono">.so</span> 写入缓存目录；之后每次调用都命中缓存、只跑运行部分，和 AOT 同速。</div>
</div>

<h2>三、为什么要 JIT：灵活性与代价</h2>
<p>JIT 的核心卖点是灵活性。第一，实验性算子可以随改随用，不必每次都重新发版打 wheel。第二，可以做架构特定的代码生成：查询到具体 GPU 架构后，再决定用哪套模板、开哪些编译标志，从而把每块卡的性能榨到更尽。第三，核可以用运行期的形状或标志来参数化，适配千变万化的输入。第四，wheel 不会臃肿——源码很小，编译产物按需生成并缓存在本地，发布物保持精简。</p>
<p>这四点里，“架构特定的代码生成”尤其能体现 JIT 的价值。同一个算法在不同代的 GPU 上，最优实现往往不一样：新架构可能提供了新的张量核指令、更大的共享内存、或不同的内存层级，针对它们手写或生成专门的代码，才能把硬件吃透。AOT 想覆盖这些差异，就得为每种架构各编一份并全部塞进 wheel；而 JIT 只需在运行时查到“我现在跑在哪种架构上”，再为它现编一份最合身的核。对一个要长期跟进前沿硬件的推理引擎来说，这种“随硬件而变”的能力极其宝贵——它意味着新卡一上市，不必苦等下一个发版周期，就能在上面跑起来并逐步调优。</p>
<p>当然，灵活不是没有代价。最直接的代价是首次调用的编译延迟：第一次用到某个核时，必须停下来等 <span class="mono">nvcc</span> 把它编出来。其次，运行环境里必须真的装着一套可用的编译工具链（编译器、CUDA 头文件等），否则现场编译会直接失败——而 AOT 的二进制则没有这个前提。所以是否选 JIT，本质上是在“灵活与瘦身”和“开箱即用、零首调延迟”之间做权衡。</p>
<p>把这份代价放到 SGLang 的真实场景里看会更清楚。服务通常是长时间运行的：一个推理进程可能要服务几小时甚至几天，那么把首次调用那一两秒的编译成本摊到漫长的服务周期里，几乎可以忽略不计；而它换来的是“同一份代码能在不同架构的卡上各自编出最优核”的能力。反过来，如果是对冷启动极度敏感、或者部署在没有编译器的精简镜像里的场景，JIT 的首调延迟与工具链依赖就会变成真正的痛点——这时候 AOT 才是更稳妥的选择。所以这道选择题没有放之四海皆准的答案，只有结合“核的稳定程度、调用频率、部署环境、对启动延迟的容忍度”之后的最优解。</p>

<h2>四、AOT 与 JIT 殊途同归</h2>
<p>把第38课和本课放在一起看会更清楚：AOT（<span class="mono">sgl-kernel</span>）和 JIT（<span class="mono">jit_kernel</span>）最终都落到“从 Python 调用一个已编译 GPU 核”。它们的终点相同，分歧只在路上——AOT 在打 wheel 时就把核编好、随二进制分发，安装即用；JIT 把源码随包带着，等首次调用时才在本机编译并缓存。理解了这一点，你就能在设计新算子时做出正确取舍：稳定、高频、要求零依赖即用的核交给 AOT；实验性强、需按架构定制、或不想撑大 wheel 的核交给 JIT。两者并存，正是 SGLang 内核体系兼顾性能与灵活的工程智慧。</p>
<p>更进一步说，这两条路并非彼此孤立。一个核常常先以 JIT 形式诞生：在 <span class="mono">jit_kernel</span> 里快速迭代、按架构试验、跑通正确性与性能之后，如果它逐渐稳定下来、被高频复用，就可以“毕业”到 <span class="mono">sgl-kernel</span> 走 AOT，固化进 wheel 享受零首调延迟。反过来，前沿硬件刚发布、新指令还在打磨时，JIT 又能让 SGLang 第一时间在上面跑起来，而不必等下一次发版。正因为终点统一（都是指向已编译核的 <span class="mono">torch.ops</span> 风格调用），上层代码几乎无需关心某个算子究竟来自 AOT 还是 JIT——这层抽象，正是把“何时编译、如何分发”的复杂度封装起来、让引擎既快又灵活的关键。</p>
<p>总结一句：JIT 不是 AOT 的替代品，而是它的互补。第38课的 <span class="mono">sgl-kernel</span> 负责把最稳定、最热的核固化进二进制，换取开箱即用与零首调延迟；本课的 <span class="mono">jit_kernel</span> 负责在运行时按需现场编译那些实验性、需按架构定制、或形状多变的核，换取灵活与瘦身。两条路最终都汇入同一套 <span class="mono">torch.ops</span> 风格调用，共同撑起 SGLang 的内核性能地基，也共同体现了大型推理系统在性能与灵活之间求取平衡的工程智慧。</p>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="AOT 与 JIT 两种策略对比：AOT 把核预编译进 wheel，首次零成本但体积大；JIT 首次现编译并缓存，安装精简、加新核灵活">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">AOT 预发布 vs JIT 按需编译</text>
    <rect x="40" y="50" width="340" height="216" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="60" y="80" style="font-size:13px;font-weight:700;fill:var(--blue)">AOT · sgl-kernel</text>
    <text x="60" y="100" style="fill:var(--muted);font-size:11px">提前编译进 wheel</text>
    <line x1="56" y1="112" x2="364" y2="112" style="stroke:var(--line);stroke-width:1"/>
    <text x="60" y="142" style="fill:var(--muted);font-size:12px">首次调用</text>
    <text x="364" y="142" text-anchor="end" style="font-size:12px;font-weight:700">零成本</text>
    <text x="60" y="182" style="fill:var(--muted);font-size:12px">安装 / wheel</text>
    <text x="364" y="182" text-anchor="end" style="font-size:12px;font-weight:700">偏大</text>
    <text x="60" y="222" style="fill:var(--muted);font-size:12px">加新核</text>
    <text x="364" y="222" text-anchor="end" style="font-size:12px;font-weight:700">需重新发版</text>
    <rect x="420" y="50" width="340" height="216" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="440" y="80" style="font-size:13px;font-weight:700;fill:var(--teal)">JIT · jit_kernel</text>
    <text x="440" y="100" style="fill:var(--muted);font-size:11px">首次现编译并缓存</text>
    <line x1="436" y1="112" x2="744" y2="112" style="stroke:var(--line);stroke-width:1"/>
    <text x="440" y="142" style="fill:var(--muted);font-size:12px">首次调用</text>
    <text x="744" y="142" text-anchor="end" style="font-size:12px;font-weight:700">一次性小成本</text>
    <text x="440" y="182" style="fill:var(--muted);font-size:12px">安装 / wheel</text>
    <text x="744" y="182" text-anchor="end" style="font-size:12px;font-weight:700">精简</text>
    <text x="440" y="222" style="fill:var(--muted);font-size:12px">加新核</text>
    <text x="744" y="222" text-anchor="end" style="font-size:12px;font-weight:700">随改随用</text>
    <text x="400" y="286" text-anchor="middle" style="fill:var(--faint);font-size:11px">同一算子可两条路径并存，殊途同归</text>
  </svg>
  <div class="figcap"><b>图 2 · AOT 预发布 vs JIT 按需编译</b> — AOT（<span class="mono">sgl-kernel</span>）把重核预编进 wheel：首次零成本，但构建慢、wheel 大；JIT（<span class="mono">jit_kernel</span>）首次现场编译并缓存：安装精简、一次性小成本、加新核灵活。</div>
</div>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>首次调用 <span class="mono">load_jit</span></h4><p>目标 <span class="mono">.so</span> 尚不存在，缓存里查不到对应标记。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>现场编译</h4><p>调用 <span class="mono">nvcc</span> 编译 C++/CUDA 源码，付一次性成本（几秒到几十秒）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>写入缓存</h4><p>把编译好的 <span class="mono">.so</span> 按唯一标记（marker）存进缓存。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>后续复用</h4><p>之后的调用命中缓存，直接复用 <span class="mono">.so</span>，不再重新编译，与 AOT 同速。</p></div></div>
</div>

<div class="cols">
  <div class="col"><strong>AOT（sgl-kernel，第38课）</strong>：打 wheel 时就用 <span class="mono">nvcc</span> 把核全部编进二进制。安装即可用，首次调用零编译延迟；但 wheel 体积大、各种架构组合需要提前枚举。</div>
  <div class="col"><strong>JIT（jit_kernel，本课）</strong>：运行时按需编译，首次调用付一次 <span class="mono">nvcc</span> 成本，之后命中缓存与 AOT 同速；wheel 不臃肿、可按 GPU 架构与运行期形状灵活生成，但要求本机有编译工具链。</div>
</div>

<table class="t">
  <tr><th>判断标准</th><th>选择</th></tr>
  <tr><td>核稳定、被高频复用、要求开箱即用</td><td>AOT（sgl-kernel）</td></tr>
  <tr><td>实验性算子，接口频繁迭代</td><td>JIT（jit_kernel）</td></tr>
  <tr><td>需按 GPU 架构定制代码生成</td><td>JIT</td></tr>
  <tr><td>部署环境没有编译器工具链</td><td>AOT</td></tr>
  <tr><td>想避免 wheel 体积膨胀</td><td>JIT</td></tr>
</table>

<div class="flow">
  <div class="node">C++/CUDA 源码 + wrappers</div>
  <div class="arrow">→</div>
  <div class="node"><span class="mono">nvcc</span> 编译</div>
  <div class="arrow">→</div>
  <div class="node">缓存 <span class="mono">.so</span>（按唯一标记）</div>
  <div class="arrow">→</div>
  <div class="node"><span class="mono">torch.ops</span> 风格调用</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/jit_kernel/utils.py ::load_jit</span><span class="ln">按需编译 + 缓存 .so：编译一次，之后复用</span></div><pre>def load_jit(
    *args,                         # unique marker identifying this kernel (distinct per kernel)
    cpp_files=None, cuda_files=None,
    cpp_wrappers=None, cuda_wrappers=None,   # each is a (export_name, kernel_name) tuple
    extra_cuda_cflags=None, extra_include_paths=None,
    extra_dependencies=None,       # e.g. "cutlass"
    build_directory=None,
    header_only=True,
):
    # Build a JIT module from C++/CUDA source ON DEMAND, then cache the
    # compiled .so so later calls reuse it (compile once, reuse forever).
    # A wrapper (export_name, kernel_name) maps the Python-facing name to
    # the C++/CUDA kernel class. Returns the loaded extension Module.
    ...
</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/jit_kernel/activation.py ::silu_and_mul</span><span class="ln">JIT 激活入口：转发到运行时编译并缓存的核</span></div><pre>def silu_and_mul(input, out=None, expert_ids=None, expert_step=1):
    # JIT activation op: run_activation triggers a compile-on-first-use
    # (then cached .so) and dispatches to that kernel.
    return run_activation("silu", input, out, expert_ids, expert_step)
</pre></div>

<p>举个具体例子：同一个 <span class="mono">silu_and_mul</span> 在两条路径里都存在——第38课的 <span class="mono">sgl-kernel</span> 里它是 AOT 版本，随 wheel 出厂即用；本课 <span class="mono">jit_kernel/activation.py</span> 里它是 JIT 版本，第一次调用时才现场编译、之后命中缓存。它的兄弟算子 <span class="mono">gelu_and_mul</span>、<span class="mono">gelu_tanh_and_mul</span> 也都走同一个 <span class="mono">run_activation</span> 入口，只是把激活名从 <span class="mono">"silu"</span> 换成 <span class="mono">"gelu"</span> / <span class="mono">"gelu_tanh"</span>——可见“一个入口、按需编译、多算子复用”正是 JIT 路径的统一套路。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><span class="mono">jit_kernel/</span> 在运行时按需编译小型 C++/CUDA 核，而不是像第38课的 <span class="mono">sgl-kernel</span> 那样提前编进 wheel。</li>
<li>核心入口 <span class="mono">load_jit(...)</span>：传入源码 + 包装器（每个是 <span class="mono">(export_name, kernel_name)</span> 二元组），编出 torch 扩展并按唯一标记缓存 <span class="mono">.so</span>。</li>
<li><strong>编译一次、之后复用</strong>：首次调用付一次 <span class="mono">nvcc</span> 成本，命中缓存后与 AOT 同速。</li>
<li>选 JIT 的理由：灵活（实验性算子、架构特定 codegen、运行期参数化）、wheel 不臃肿；代价是首调延迟 + 需要编译工具链。</li>
<li><span class="mono">activation.py</span> 首次使用时按 dtype 建 JIT 模块；<span class="mono">norm.py</span> 提供 JIT 版 <span class="mono">fused_add_rmsnorm</span>（第36课 RMSNorm 快速路径所用）。</li>
<li>AOT 与 JIT 殊途同归：都成为指向已编译核的 <span class="mono">torch.ops</span> 风格调用，差别只在“何时、如何构建与分发”。</li>
</ul></div>
""", "en": r"""
<p class="lead">In Lesson 38 we saw that <span class="mono">sgl-kernel</span> follows the AOT (ahead-of-time) route: kernels are compiled by <span class="mono">nvcc</span> into the binary when the wheel is built. This lesson's star, <span class="mono">python/sglang/jit_kernel/</span>, takes the other road—<strong>just-in-time compilation at runtime, on demand</strong> (JIT). Instead of stuffing every kernel into the wheel up front, it compiles the C++/CUDA source into a <span class="mono">.so</span> on the local machine only when actually needed, then caches the result for reuse.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div> AOT is like a factory baking every flavor of cake before shipping and packing them all in a box: you can eat the moment you buy, but the box is big and heavy, and the factory has to guess in advance which flavors you'll want. JIT is like the oven at home: the first time you want a flavor you bake a batch from the recipe, which takes a while; but once baked, you put it in the fridge and any later craving is served instantly, as fast as ready-made. The oven (compiler toolchain) must be in your home and you must carry the recipe (source), but you don't haul a whole box of cakes around.</div>

<div class="card macro"><div class="tag">🌍 The big picture</div> Whether AOT or JIT, both end up as a set of <span class="mono">torch.ops</span>-style callables that invoke compiled GPU kernels directly from Python. The difference isn't "what the result is" but "when and how it gets built and shipped." AOT front-loads the compile cost to release time in exchange for being ready out of the box; JIT defers the compile cost to first call in exchange for flexibility and a slimmer wheel. SGLang keeps both paths so stable, hot kernels go AOT while experimental or architecture-specific ones go JIT.</div>

<h2>1. What the JIT kernel path is</h2>
<p>The core idea of the <span class="mono">jit_kernel</span> module is to ship small C++/CUDA kernels as "source + recipe" rather than pre-compiled into the binary. The first time an op is used, the module invokes the compiler on the spot to build it into a PyTorch extension module. The immediate payoff is that the wheel doesn't bloat from packing kernels for every architecture and dtype. For example, <span class="mono">jit_kernel/activation.py</span> builds a corresponding JIT module per dtype on first use; <span class="mono">jit_kernel/norm.py</span> provides a JIT <span class="mono">fused_add_rmsnorm</span>—exactly the fused kernel the RMSNorm fast path from Lesson 36 relies on.</p>
<p>It is worth unpacking where "wheel bloat" comes from. A kernel often has more than one version: it may need a separate build for each dtype (fp16, bf16, fp8, etc.), a separate build for each GPU compute capability (e.g. sm80, sm90), and on top of that the combinations of shapes and flags, so the compiled artifacts grow multiplicatively. If everything went AOT, all these combinations would have to be enumerated up front and packed into the <span class="mono">.so</span>, making the wheel quickly unacceptable in size—while the vast majority of those combinations are never used in any given deployment. JIT takes the opposite approach: the release contains only the slim source, and whichever combination is actually needed gets compiled and cached on the spot for that one combination—trading a space cost for a one-time time cost, and paying only for the code paths that actually run.</p>
<p>These two modules illustrate JIT's "on demand" nature well. <span class="mono">activation.py</span> does not compile activation kernels for every dtype up front; instead, only when a concrete dtype is first used does it build the corresponding JIT module—use fp16 and it compiles the fp16 one, use bf16 and it compiles the bf16 one, and unused ones cost no compile time at all. This "compile only what is triggered" strategy spends the compile cost precisely on the code paths that actually run. <span class="mono">norm.py</span>'s <span class="mono">fused_add_rmsnorm</span> is a typical fused kernel: it merges the "residual add" and "RMSNorm normalization" steps into one GPU kernel, saving a round trip to memory. Lesson 36's RMSNorm fast path calls exactly this—showing the JIT path is not a side note but genuinely serves a key op in the main inference pipeline.</p>
<p>Why not make everything AOT? Because in practice many kernels are not "stable and general." Some ops are still experimental with changing interfaces; some kernels need different code per GPU architecture (the JIT path queries the current architecture, e.g. via <span class="mono">get_jit_cuda_arch</span> to get the compute capability) and picks instructions and templates accordingly; some kernels must be parameterized by shapes or flags known only at runtime. Enumerating all of these into the wheel up front is both unrealistic and bloated, so letting JIT generate them on demand is the better strategy.</p>
<p>Seen from another angle, AOT and JIT are not rivals replacing each other but two cooperating pipelines. SGLang hands the hot, fixed-shape, startup-latency-sensitive kernels to <span class="mono">sgl-kernel</span> to be compiled into the binary ahead of time; it leaves the still-evolving, hardware-feature-dependent, or only-for-specific-dtype/shape kernels to <span class="mono">jit_kernel</span> to build on the spot. This "route by a kernel's lifecycle and stability" design keeps the release lean without sacrificing fast adaptation to cutting-edge hardware and new ops. Understanding the JIT path is essentially understanding all the benefits and costs of "moving compilation from release time to runtime."</p>

<h2>2. How load_jit works: compile once, reuse later</h2>
<p>The unified entry to the whole JIT path is <span class="mono">load_jit(...)</span>. You give it a set of C++/CUDA source files plus a set of "wrappers": each wrapper is an <span class="mono">(export_name, kernel_name)</span> tuple mapping a Python-facing callable name to some underlying C++/CUDA kernel class. From these, <span class="mono">load_jit</span> compiles a torch extension module and caches the built <span class="mono">.so</span> keyed by a unique marker. Besides source and wrappers, it accepts optional parameters: extra compile flags, extra include search paths, external dependencies (e.g. <span class="mono">cutlass</span>), a build directory, and so on, so the on-the-spot compilation can hook into the needed third-party libraries and custom options.</p>
<p>The cache is the key. <strong>The first call pays a one-time compile cost</strong> (an actual <span class="mono">nvcc</span> run, possibly seconds to tens of seconds), but as long as the marker hits, <strong>later calls reuse the cached <span class="mono">.so</span> directly</strong> with no recompilation. In other words: slow the first time, then as fast as AOT. This "compile once, reuse forever" design is exactly why JIT can keep its flexibility while still being high-performance. The unique marker ensures different kernels and different parameter combinations each map to their own cache entry without cross-contamination—when you change the source, switch architecture, or flip a key compile flag, the marker changes accordingly, triggering a fresh compilation and a new cache entry instead of wrongly reusing the old artifact.</p>
<p>From the caller's viewpoint this is almost transparent: you just call the Python name exported by a wrapper, the first call "stalls" briefly (because it's compiling), and afterwards it's so smooth you don't feel JIT is there at all. What <span class="mono">load_jit</span> ultimately returns is the loaded extension Module, carrying the callable kernels exposed under their <span class="mono">export_name</span>; upper-layer code takes it and uses it just like any ordinary compiled op.</p>
<p>There is also an easily-overlooked detail: the cache usually lives in a build directory on disk, so the "compile once, reuse later" benefit is not limited to a single process—it carries across processes and even across multiple restarts. As long as the source, architecture, and key compile options are unchanged (i.e. the unique marker is unchanged), the first time a new process uses the kernel it simply finds a ready-made <span class="mono">.so</span> already on disk and loads it, skipping compilation. This means that when you restart the service repeatedly on the same machine, the actual <span class="mono">nvcc</span> compile cost is usually paid only the very first time; no matter how many times you restart afterwards, it's a sub-second load. Once you understand this, you see why JIT is not so "scary" in production—the first-call latency is one-time, while the cache is persistent.</p>

<div class="fig">
  <svg viewBox="0 0 800 270" role="img" aria-label="JIT call timeline: the first call pays a one-time compile cost and writes the .so into the cache, every later call hits the cache and just runs">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">Compile + cache .so on first call, reuse afterwards</text>
    <rect x="470" y="17" width="12" height="12" rx="3" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="488" y="27" style="fill:var(--muted);font-size:12px">compile</text>
    <rect x="556" y="17" width="12" height="12" rx="3" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="574" y="27" style="fill:var(--muted);font-size:12px">run</text>
    <rect x="300" y="44" width="150" height="40" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="375" y="62" text-anchor="middle" style="font-size:12px;font-weight:700">cache dir</text>
    <text x="375" y="78" text-anchor="middle" class="mono" style="fill:var(--accent-ink);font-size:11px">.so</text>
    <line x1="40" y1="210" x2="770" y2="210" style="stroke:var(--line);stroke-width:1.5"/>
    <path d="M770 210 l-9 -4 v8 z" style="fill:var(--faint)"/>
    <text x="700" y="232" style="fill:var(--faint);font-size:11px">time →</text>
    <rect x="70" y="92" width="90" height="78" rx="4" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <rect x="70" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="115" y="136" text-anchor="middle" style="font-size:12px;font-weight:700">compile nvcc</text>
    <text x="115" y="194" text-anchor="middle" style="font-size:11px">run</text>
    <text x="115" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">1st call</text>
    <path d="M160 116 L300 64" style="stroke:var(--amber);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <text x="232" y="98" text-anchor="middle" style="fill:var(--muted);font-size:11px">write</text>
    <rect x="300" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="345" y="194" text-anchor="middle" style="font-size:11px">run</text>
    <text x="345" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">2nd call</text>
    <rect x="460" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="505" y="194" text-anchor="middle" style="font-size:11px">run</text>
    <text x="505" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">3rd call</text>
    <rect x="620" y="170" width="90" height="40" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="665" y="194" text-anchor="middle" style="font-size:11px">run</text>
    <text x="665" y="230" text-anchor="middle" style="fill:var(--faint);font-size:11px">4th call</text>
    <path d="M345 84 L345 170" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <path d="M420 84 L505 170" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <path d="M440 84 L665 170" style="stroke:var(--teal);stroke-width:1.5;stroke-dasharray:4 3;fill:none"/>
    <text x="560" y="120" text-anchor="middle" style="fill:var(--muted);font-size:11px">cache hit · reuse .so</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Compile first, reuse later</b> — the first call to a JIT op pays a one-time <span class="mono">nvcc</span> compile cost and writes the <span class="mono">.so</span> into the cache dir; every later call hits the cache and runs only, as fast as AOT.</div>
</div>

<h2>3. Why JIT: flexibility and its cost</h2>
<p>JIT's core selling point is flexibility. First, experimental ops can be changed and used freely without re-releasing a wheel each time. Second, it enables architecture-specific code generation: after querying the concrete GPU architecture, it decides which templates and compile flags to use, squeezing more performance out of each card. Third, kernels can be parameterized by runtime shapes or flags, adapting to ever-changing inputs. Fourth, the wheel stays slim—source is tiny, the compiled artifact is generated on demand and cached locally, keeping the release lean.</p>
<p>Of these four, "architecture-specific code generation" especially shows JIT's value. The optimal implementation of the same algorithm often differs across GPU generations: a new architecture may offer new tensor-core instructions, larger shared memory, or a different memory hierarchy, and only by hand-writing or generating code specialized for them can you fully exploit the hardware. For AOT to cover these differences, it must compile a separate build per architecture and pack them all into the wheel; JIT only needs to query "which architecture am I running on now" at runtime and compile a best-fitting kernel for it. For an inference engine that must keep pace with cutting-edge hardware, this "adapt to the hardware" ability is invaluable—it means when a new card hits the market, you can run on it and gradually tune without waiting for the next release cycle.</p>
<p>Flexibility isn't free, of course. The most direct cost is first-call compile latency: the first time a kernel is used, you must stop and wait for <span class="mono">nvcc</span> to build it. Next, the runtime environment must actually have a working compiler toolchain (compiler, CUDA headers, etc.), or on-the-spot compilation simply fails—whereas an AOT binary has no such prerequisite. So choosing JIT is essentially a trade-off between "flexible and slim" and "ready out of the box with zero first-call latency."</p>
<p>This cost is clearer in SGLang's real setting. A service is usually long-running: an inference process may serve for hours or even days, so amortizing the one-or-two-second first-call compile cost over a long serving window is practically negligible; in return you get the ability for "the same code to compile an optimal kernel on each different architecture." Conversely, in scenarios extremely sensitive to cold start, or deployed in a slim image without a compiler, JIT's first-call latency and toolchain dependency become real pain points—there, AOT is the safer choice. So this question has no one-size-fits-all answer, only the best solution after weighing "kernel stability, call frequency, deployment environment, and tolerance for startup latency."</p>

<h2>4. AOT and JIT reach the same destination</h2>
<p>Putting Lesson 38 and this lesson side by side makes it clearer: AOT (<span class="mono">sgl-kernel</span>) and JIT (<span class="mono">jit_kernel</span>) both end up "calling a compiled GPU kernel from Python." Their destination is the same; they only differ on the way there—AOT compiles kernels at wheel-build time and ships them with the binary, ready on install; JIT ships the source and compiles plus caches on the local machine at first call. Once you grasp this, you can make the right call when designing new ops: hand stable, hot, zero-dependency kernels to AOT; hand experimental, architecture-specific, or wheel-slimming kernels to JIT. Having both is precisely the engineering wisdom that lets SGLang's kernel system balance performance and flexibility.</p>
<p>Going further, the two paths are not isolated. A kernel often is born as JIT first: it iterates quickly inside <span class="mono">jit_kernel</span>, is tried per architecture, and once correctness and performance are proven—if it stabilizes and is heavily reused—it can "graduate" to <span class="mono">sgl-kernel</span> on the AOT path, baked into the wheel to enjoy zero first-call latency. Conversely, when new hardware just launches and new instructions are still being polished, JIT lets SGLang run on it immediately without waiting for the next release. Precisely because the destination is unified (both are <span class="mono">torch.ops</span>-style callables into compiled kernels), upper-layer code barely needs to care whether an op comes from AOT or JIT—this abstraction, which encapsulates the complexity of "when to compile and how to ship," is the key to an engine that is both fast and flexible.</p>
<p>In one sentence: JIT is not a replacement for AOT but its complement. Lesson 38's <span class="mono">sgl-kernel</span> bakes the most stable, hottest kernels into the binary in exchange for being ready out of the box with zero first-call latency; this lesson's <span class="mono">jit_kernel</span> compiles the experimental, architecture-specific, or shape-varying kernels on demand at runtime in exchange for flexibility and slimness. Both paths ultimately converge into the same <span class="mono">torch.ops</span>-style calls, together holding up SGLang's kernel performance foundation, and together embodying the engineering wisdom of balancing performance and flexibility in a large inference system.</p>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="AOT vs JIT strategies: AOT prebuilds kernels into the wheel, zero first-call cost but big size; JIT compiles on first use and caches, slim install and easy to add a kernel">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">AOT shipped vs JIT compiled on demand</text>
    <rect x="40" y="50" width="340" height="216" rx="10" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="60" y="80" style="font-size:13px;font-weight:700;fill:var(--blue)">AOT · sgl-kernel</text>
    <text x="60" y="100" style="fill:var(--muted);font-size:11px">prebuilt into the wheel</text>
    <line x1="56" y1="112" x2="364" y2="112" style="stroke:var(--line);stroke-width:1"/>
    <text x="60" y="142" style="fill:var(--muted);font-size:12px">first call</text>
    <text x="364" y="142" text-anchor="end" style="font-size:12px;font-weight:700">zero cost</text>
    <text x="60" y="182" style="fill:var(--muted);font-size:12px">install / wheel</text>
    <text x="364" y="182" text-anchor="end" style="font-size:12px;font-weight:700">larger</text>
    <text x="60" y="222" style="fill:var(--muted);font-size:12px">add a kernel</text>
    <text x="364" y="222" text-anchor="end" style="font-size:12px;font-weight:700">re-release</text>
    <rect x="420" y="50" width="340" height="216" rx="10" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="440" y="80" style="font-size:13px;font-weight:700;fill:var(--teal)">JIT · jit_kernel</text>
    <text x="440" y="100" style="fill:var(--muted);font-size:11px">compiled on first use</text>
    <line x1="436" y1="112" x2="744" y2="112" style="stroke:var(--line);stroke-width:1"/>
    <text x="440" y="142" style="fill:var(--muted);font-size:12px">first call</text>
    <text x="744" y="142" text-anchor="end" style="font-size:12px;font-weight:700">one-time small</text>
    <text x="440" y="182" style="fill:var(--muted);font-size:12px">install / wheel</text>
    <text x="744" y="182" text-anchor="end" style="font-size:12px;font-weight:700">slim</text>
    <text x="440" y="222" style="fill:var(--muted);font-size:12px">add a kernel</text>
    <text x="744" y="222" text-anchor="end" style="font-size:12px;font-weight:700">edit &amp; run</text>
    <text x="400" y="286" text-anchor="middle" style="fill:var(--faint);font-size:11px">one op can live on both paths — same destination</text>
  </svg>
  <div class="figcap"><b>Fig 2 · AOT shipped vs JIT compiled on demand</b> — AOT (<span class="mono">sgl-kernel</span>) prebuilds heavy kernels into the wheel: zero first-call cost, but slow build and a big wheel; JIT (<span class="mono">jit_kernel</span>) compiles on first use and caches: slim install, a tiny one-time cost, and easy to add a kernel.</div>
</div>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>First call to <span class="mono">load_jit</span></h4><p>The target <span class="mono">.so</span> does not exist yet; the marker is not found in the cache.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Compile on the spot</h4><p>Invoke <span class="mono">nvcc</span> to compile the C++/CUDA source, paying a one-time cost (seconds to tens of seconds).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Write to cache</h4><p>Store the compiled <span class="mono">.so</span> into the cache keyed by a unique marker.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Reuse later</h4><p>Subsequent calls hit the cache and reuse the <span class="mono">.so</span> directly, no recompilation, as fast as AOT.</p></div></div>
</div>

<div class="cols">
  <div class="col"><strong>AOT (sgl-kernel, Lesson 38)</strong>: at wheel-build time <span class="mono">nvcc</span> compiles all kernels into the binary. Ready on install, zero first-call compile latency; but the wheel is large and architecture combinations must be enumerated up front.</div>
  <div class="col"><strong>JIT (jit_kernel, this lesson)</strong>: compiles on demand at runtime, the first call pays a one-time <span class="mono">nvcc</span> cost, then cache hits are as fast as AOT; the wheel stays slim and can be generated flexibly per GPU architecture and runtime shape, but it requires a compiler toolchain on the machine.</div>
</div>

<table class="t">
  <tr><th>Criterion</th><th>Choice</th></tr>
  <tr><td>Stable kernel, heavily reused, must be ready out of the box</td><td>AOT (sgl-kernel)</td></tr>
  <tr><td>Experimental op, interface iterates frequently</td><td>JIT (jit_kernel)</td></tr>
  <tr><td>Needs architecture-specific code generation</td><td>JIT</td></tr>
  <tr><td>Deployment environment lacks a compiler toolchain</td><td>AOT</td></tr>
  <tr><td>Want to avoid wheel size bloat</td><td>JIT</td></tr>
</table>

<div class="flow">
  <div class="node">C++/CUDA source + wrappers</div>
  <div class="arrow">→</div>
  <div class="node"><span class="mono">nvcc</span> compile</div>
  <div class="arrow">→</div>
  <div class="node">cached <span class="mono">.so</span> (by unique marker)</div>
  <div class="arrow">→</div>
  <div class="node"><span class="mono">torch.ops</span>-style call</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/jit_kernel/utils.py ::load_jit</span><span class="ln">compile on demand + cache the .so: build once, reuse</span></div><pre>def load_jit(
    *args,                         # unique marker identifying this kernel (distinct per kernel)
    cpp_files=None, cuda_files=None,
    cpp_wrappers=None, cuda_wrappers=None,   # each is a (export_name, kernel_name) tuple
    extra_cuda_cflags=None, extra_include_paths=None,
    extra_dependencies=None,       # e.g. "cutlass"
    build_directory=None,
    header_only=True,
):
    # Build a JIT module from C++/CUDA source ON DEMAND, then cache the
    # compiled .so so later calls reuse it (compile once, reuse forever).
    # A wrapper (export_name, kernel_name) maps the Python-facing name to
    # the C++/CUDA kernel class. Returns the loaded extension Module.
    ...
</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/jit_kernel/activation.py ::silu_and_mul</span><span class="ln">JIT activation entry: forwards to a runtime-compiled, cached kernel</span></div><pre>def silu_and_mul(input, out=None, expert_ids=None, expert_step=1):
    # JIT activation op: run_activation triggers a compile-on-first-use
    # (then cached .so) and dispatches to that kernel.
    return run_activation("silu", input, out, expert_ids, expert_step)
</pre></div>

<p>A concrete example: the same <span class="mono">silu_and_mul</span> exists on both paths—in Lesson 38's <span class="mono">sgl-kernel</span> it is the AOT version, shipped ready with the wheel; here in <span class="mono">jit_kernel/activation.py</span> it is the JIT version, compiled on the spot on the first call and cache-hit thereafter. Its sibling ops <span class="mono">gelu_and_mul</span> and <span class="mono">gelu_tanh_and_mul</span> go through the same <span class="mono">run_activation</span> entry too, only swapping the activation name from <span class="mono">"silu"</span> to <span class="mono">"gelu"</span> / <span class="mono">"gelu_tanh"</span>—so "one entry, compile on demand, reused across ops" is exactly the JIT path's uniform pattern.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><span class="mono">jit_kernel/</span> compiles small C++/CUDA kernels on demand at runtime, instead of shipping them ahead of time in the wheel like Lesson 38's <span class="mono">sgl-kernel</span>.</li>
<li>Core entry <span class="mono">load_jit(...)</span>: pass source + wrappers (each an <span class="mono">(export_name, kernel_name)</span> tuple), it builds a torch extension and caches the <span class="mono">.so</span> by a unique marker.</li>
<li><strong>Compile once, reuse later</strong>: the first call pays a one-time <span class="mono">nvcc</span> cost; after a cache hit it's as fast as AOT.</li>
<li>Reasons to pick JIT: flexibility (experimental ops, architecture-specific codegen, runtime parameterization) and no wheel bloat; the cost is first-call latency + needing a compiler toolchain.</li>
<li><span class="mono">activation.py</span> builds a JIT module per dtype on first use; <span class="mono">norm.py</span> provides a JIT <span class="mono">fused_add_rmsnorm</span> (used by Lesson 36's RMSNorm fast path).</li>
<li>AOT and JIT converge: both become <span class="mono">torch.ops</span>-style callables into compiled kernels; the difference is only "when and how they're built and shipped."</li>
</ul></div>
"""}
LESSON_40 = {"zh": r"""
<p class="lead">第40课，我们把镜头拉到最近，正面解剖一个真实的 <strong>attention kernel</strong>（注意力核函数）。前面几课我们讲过分页 KV 缓存（第30课）、解码阶段的带宽瓶颈（第4课）、以及注意力后端的封装方式（第33课），这一课要把这些拼图拼到一起：当一个序列在 <span class="mono">decode</span>（解码，每步只生成一个新 token）阶段需要做注意力时，GPU 上那段被成千上万次反复调用的核函数，里面到底发生了什么。我们不谈玄学，只谈这段代码每一步搬了哪些字节、为什么这么搬。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把 decode 注意力核函数想象成一个<strong>仓库拣货员</strong>。仓库里的货架（KV 缓存）不是一整排连续摆放的，而是被切成一个个<strong>货格</strong>（page，分页），散落在仓库各处。拣货员手里有一张<strong>取货单</strong>（<span class="mono">page_table</span>，块表），上面写着「这个客户的货分别在 7 号、3 号、19 号格子」。他不会把整个仓库搬空，而是<strong>只按单取那几格</strong>，边取边算账。</p>
<p>更妙的是他算账的方式：他不等所有货都搬出来再统一称重，而是<strong>每取一格就更新一次累计值</strong>——记住目前的最大值和累计权重，新的一格来了就按比例修正之前的结果。这样他全程只需要一个小本子（寄存器），永远不用把整张大表摊在地上。这正是<strong>在线 softmax</strong> 的精髓。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>注意力的数学很短：<span class="mono">softmax(Q·Kᵀ / √d)·V</span>。但 decode 阶段的难点不在算术，而在<strong>数据从哪来、怎么搬</strong>。每生成一个 token，Q 只有一行（一个查询位置），却要和这条序列<strong>历史上所有的 K、V</strong> 做点积。这些 K、V 早就被写进了分页 KV 缓存，物理上散落在非连续的页里。所以核函数的真正主线是：<strong>用 page_table 把分散的 KV 页收集（gather）回来 → 分块（tiling）流式地算 Q·Kᵀ → 在线 softmax → 再乘 V</strong>。因为 decode 是带宽瓶颈型负载，核函数的全部价值就在于「高效地把 KV 字节搬过来」。</p>
</div>

<h2>一、decode 注意力核函数的主线：收集、点积、归一、加权</h2>
<p>先把整条主线说清楚。当调度器决定为某个序列生成下一个 token，它会发起一次 decode 前向。注意力这一步收到的输入是：一行 <strong>Q</strong>（当前查询位置，形状很「瘦」，只有一个 token），加上整条序列<strong>已经缓存好的 K 和 V</strong>。这些 K、V 不在一块连续显存里，而是被分页 KV 缓存切成固定大小的页，分散存放，每个序列拥有哪几页由 <span class="mono">page_table</span> 记录。</p>
<p>于是核函数干的第一件事不是算乘法，而是<strong>查表寻址</strong>：根据 <span class="mono">page_table[b]</span> 找到序列 b 的所有 KV 页编号，再从这些非连续的页里把 K、V 块逐块读进来。读进来之后才进入数学部分：当前这一行 Q 和读进来的这一块 K 做点积，得到一小段注意力分数；这段分数立刻喂给在线 softmax 更新累计值；最后用归一化后的权重去加权对应的 V 块，累加进输出。整个过程是<strong>流式</strong>的——读一块、算一块、丢一块，从不把完整的分数矩阵留在显存里。</p>

<p>这里有一个直觉上的反转值得记住：很多人以为注意力的瓶颈是那一堆乘加，但在 decode 里，乘加其实很少（Q 只有一行），真正昂贵的是<strong>把历史 KV 一字节一字节读进来</strong>。所以核函数的设计目标不是「少算」，而是「少搬、搬得连贯、搬一遍就算完」。带着这个目标去看下面每一个设计选择，你会发现它们指向同一个方向：把宝贵的显存带宽花在刀刃上。</p>

<div class="flow"><div class="node">page_table 收集分页 KV</div><div class="arrow">→</div><div class="node">Q·Kᵀ 分块点积</div><div class="arrow">→</div><div class="node">在线 softmax 归一</div><div class="arrow">→</div><div class="node">·V 加权累加 → 输出</div></div>

<p>注意第一个节点：在很多教科书的注意力图里，第一步是「Q 乘 K」，但在真实的 decode 核函数里，<strong>真正的第一步是「按取货单收集 KV」</strong>。这一步决定了核函数的访存模式，也决定了它能不能跑满显存带宽。因为 KV 缓存通常远大于片上的 SRAM，所以怎么把页搬进来、搬进来的粒度多大，几乎就等于性能本身。</p>

<p>再强调一遍这条主线为什么重要：在 decode 阶段，模型权重的搬运和注意力对 KV 的搬运，共同决定了每一步生成耗时，而其中 KV 的访问会随着序列变长而不断增大。也就是说，序列越长、上下文越多，注意力核函数搬运的 KV 字节就越多，它在整步耗时里的占比也越高。这就是为什么我们必须把这段核函数看得这么仔细：它不是一段可有可无的小函数，而是长上下文推理里实打实的热点。理解它每一步搬了什么，才能理解后端为它做的种种优化到底优化在哪里。</p>

<h2>二、为什么 KV 要分页，以及核函数如何「按页拣货」</h2>
<p>第30课讲过，分页 KV 缓存把每条序列的历史切成固定大小的页，像操作系统的虚拟内存一样按页分配。好处是：不同序列长度差异巨大，按页分配可以避免为长序列预留巨大连续显存、也避免短序列浪费，显存利用率高、还能在序列间共享前缀。但代价是：<strong>同一条序列的 KV 在物理上不再连续</strong>，逻辑上第 0、1、2、3 个 KV 块，可能分别落在 7、3、19、5 号物理页。</p>
<p>核函数要还原逻辑顺序，就得靠 <span class="mono">page_table</span>（也叫 block table）。它是一张「逻辑块号 → 物理页号」的映射表。核函数遍历某个序列需要的逻辑位置，通过 page_table 翻译成物理页地址，再去那一页读 K、V。下面这组格子就是 page_table 收集分页 KV 的样子——逻辑上相邻的页，物理上是跳着取的：</p>

<div class="cellgroup"><div class="cell">页 #7</div><div class="cell">页 #3</div><div class="cell">页 #19</div><div class="cell">页 #5</div><div class="cell">页 #12</div><div class="cell">页 #0</div></div>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="分页 KV 收集：连续的逻辑序列经 page_table 的 kv_indices 指向 KV 池里物理分散的页，核函数据此把正确的页收集起来再做注意力">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">逻辑连续 → kv_indices → 物理分散</text>
    <text x="24" y="56" style="fill:var(--muted);font-size:11px">逻辑序列（连续）</text>
    <rect x="24" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="60" y="86" text-anchor="middle" class="mono" style="font-size:11px">L0</text>
    <rect x="110" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="146" y="86" text-anchor="middle" class="mono" style="font-size:11px">L1</text>
    <rect x="196" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="232" y="86" text-anchor="middle" class="mono" style="font-size:11px">L2</text>
    <rect x="282" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="318" y="86" text-anchor="middle" class="mono" style="font-size:11px">L3</text>
    <line x1="60" y1="98" x2="60" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="146" y1="98" x2="146" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="232" y1="98" x2="232" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="318" y1="98" x2="318" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="138" style="fill:var(--muted);font-size:11px">page_table（kv_indices）</text>
    <rect x="24" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="60" y="170" text-anchor="middle" class="mono" style="font-size:11px">7</text>
    <rect x="110" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="146" y="170" text-anchor="middle" class="mono" style="font-size:11px">3</text>
    <rect x="196" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="232" y="170" text-anchor="middle" class="mono" style="font-size:11px">19</text>
    <rect x="282" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="318" y="170" text-anchor="middle" class="mono" style="font-size:11px">5</text>
    <rect x="424" y="44" width="356" height="242" rx="12" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="440" y="64" style="fill:var(--faint);font-size:11px">KV 池（物理分散的页）</text>
    <rect x="446" y="80" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="488" y="103" text-anchor="middle" class="mono" style="font-size:11px">页 #3</text>
    <rect x="600" y="74" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="642" y="97" text-anchor="middle" class="mono" style="font-size:11px">页 #19</text>
    <rect x="470" y="152" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="512" y="175" text-anchor="middle" class="mono" style="font-size:11px">页 #7</text>
    <rect x="636" y="150" width="84" height="38" rx="6" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="678" y="173" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">页 #0</text>
    <rect x="540" y="224" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="582" y="247" text-anchor="middle" class="mono" style="font-size:11px">页 #5</text>
    <rect x="690" y="224" width="84" height="38" rx="6" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="732" y="247" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">页 #12</text>
    <line x1="60" y1="182" x2="512" y2="171" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="146" y1="182" x2="488" y2="99" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="232" y1="182" x2="642" y2="93" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="318" y1="182" x2="582" y2="243" style="stroke:var(--teal);stroke-width:1.5"/>
    <text x="24" y="306" style="fill:var(--faint);font-size:11px">核函数据 kv_indices 收集这些页 → 当作一条连续序列来算</text>
  </svg>
  <div class="figcap"><b>图 1 · 分页 KV 收集</b> — 逻辑上连续的序列（<span class="mono">L0…L3</span>）经 <span class="mono">page_table</span>（即 <span class="mono">kv_indices</span>）翻译成物理上分散的 KV 页（#7、#3、#19、#5）；核函数先按 <span class="mono">kv_indices</span> 把这些散落的页收集回来，才能把它们当作一条序列来注意。</div>
</div>

<p>这种「跳着取」的访存正是 decode 核函数最大的工程挑战。如果实现得粗糙，跳页会让访存变得零散、带宽利用率低；实现得好，核函数会让一个线程块负责一个序列、或者把多个 KV 页的读取合并成对齐的、连贯的内存事务，尽量把每一次显存读取都用满。这也是为什么同一份注意力数学，不同后端（第33课）跑出来的速度天差地别——差距几乎全在 KV 的访存效率上。</p>

<p>这里还藏着一个常见误解需要澄清：分页并不会让注意力「算得更少」，它改变的只是 KV 在显存里的<strong>组织方式</strong>。点积、softmax、加权这些数学一步都没省，省的是<strong>显存的分配与浪费</strong>，以及让相同前缀能被多条序列共享、不必重复存一遍。换句话说，分页是显存管理层面的胜利，而核函数要做的是「在这种非连续布局上，依然把字节高效地搬进来」。理解了这一点，你就不会把 page_table 误当成某种加速算法——它是寻址机制，是核函数和分页缓存之间的那张地图。</p>

<h2>三、分块（tiling）与在线 softmax：一遍流式算完</h2>
<p>第二个关键设计是<strong>分块</strong>。GPU 的寄存器和共享内存（SRAM）非常小，装不下完整的 K、V，更装不下一条长序列的全部注意力分数矩阵。所以核函数把序列的 KV 切成一个个<strong>瓦片（tile）</strong>，一次只把一个瓦片流进 SRAM 来算，算完就让下一个瓦片覆盖它。核函数<strong>从不一次性把整张分数矩阵物化（materialise）出来</strong>，因为那张矩阵既占显存、又要反复读写、纯属浪费带宽。</p>
<p>但分块带来一个数学难题：softmax 需要先知道所有分数里的最大值、再算所有指数的总和，才能归一化。如果分块流式处理，怎么可能「先看完所有分数」？答案是 FlashAttention 风格的<strong>在线 softmax</strong>：核函数维护两个累计量——一个<strong>运行最大值</strong>（running max）和一个<strong>运行分母</strong>（running denominator，即累计的指数和）。每来一个新瓦片，就用新的局部最大值去修正旧的累计值（按指数比例缩放之前累加的结果），再把这块的贡献加进去。这样<strong>只需要一遍流式扫描</strong>，不用存下所有分数，就能算出和「全局 softmax」完全等价的结果，省显存、更省带宽。</p>
<p>这也解释了 prefill（预填充）和 decode 两种核函数为什么长得不一样。下面把两者并排对比：</p>

<div class="cols"><div class="col"><strong>prefill 核函数</strong><br>处理 prompt 的所有 token，查询位置很多，是一个又宽又厚的、类似 GEMM（大矩阵乘）的稠密计算。计算密集，能把张量核心喂饱，瓶颈更偏算力。</div><div class="col"><strong>decode 核函数</strong><br>每步只有一个查询位置，形状又「瘦」又长，主体是按 page_table 在 KV 缓存上反复 gather。计算量很小、访存量很大，瓶颈是显存带宽（第4课）。</div></div>

<p>正因为两者的形状和瓶颈完全不同，SGLang 会为它们走不同的核函数路径：prefill 像一次大 GEMM，decode 像一次大规模的稀疏拣货。把它们硬塞进同一个实现，往往两头都跑不快。</p>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="分块注意力内循环：一个 Q 瓦片流式遍历各 KV 瓦片，每块算 S 等于 Q 乘 K 转置，用在线 softmax 更新运行最大值与分母，再把乘 V 的结果累加进累加器，完整分数矩阵从不落 HBM">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">一个 Q 瓦片 → 流式遍历 KV 瓦片</text>
    <rect x="24" y="86" width="92" height="120" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="70" y="140" text-anchor="middle" class="mono" style="font-size:12px">Q 瓦片</text>
    <text x="70" y="160" text-anchor="middle" style="fill:var(--muted);font-size:11px">一行查询</text>
    <line x1="116" y1="146" x2="148" y2="146" style="stroke:var(--line);stroke-width:2"/>
    <rect x="150" y="52" width="486" height="208" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:6 5"/>
    <text x="166" y="74" style="fill:var(--accent-ink);font-weight:700;font-size:12px">循环：遍历 KV 瓦片</text>
    <rect x="166" y="90" width="86" height="46" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="209" y="111" text-anchor="middle" class="mono" style="font-size:11px">KV 瓦片</text>
    <text x="209" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">K、V 块</text>
    <line x1="252" y1="113" x2="276" y2="113" style="stroke:var(--line);stroke-width:2"/>
    <rect x="278" y="90" width="92" height="46" rx="6" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="324" y="111" text-anchor="middle" class="mono" style="font-size:11px">S=Q·Kᵀ</text>
    <text x="324" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">本块分数</text>
    <line x1="370" y1="113" x2="394" y2="113" style="stroke:var(--line);stroke-width:2"/>
    <rect x="396" y="90" width="104" height="46" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="448" y="111" text-anchor="middle" style="font-size:11px">在线 softmax</text>
    <text x="448" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">更新 m、ℓ</text>
    <line x1="500" y1="113" x2="524" y2="113" style="stroke:var(--line);stroke-width:2"/>
    <rect x="526" y="90" width="92" height="46" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="572" y="111" text-anchor="middle" class="mono" style="font-size:11px">·V 累加</text>
    <text x="572" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">缩放累加器</text>
    <rect x="166" y="172" width="452" height="66" rx="8" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="182" y="192" style="fill:var(--muted);font-size:11px">运行状态（小本子，留在寄存器）</text>
    <rect x="182" y="202" width="120" height="26" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="242" y="219" text-anchor="middle" class="mono" style="font-size:11px">m 运行最大值</text>
    <rect x="312" y="202" width="118" height="26" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="371" y="219" text-anchor="middle" class="mono" style="font-size:11px">ℓ 运行分母</text>
    <rect x="440" y="202" width="160" height="26" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="520" y="219" text-anchor="middle" class="mono" style="font-size:11px">O 累加器</text>
    <line x1="636" y1="146" x2="668" y2="146" style="stroke:var(--line);stroke-width:2"/>
    <rect x="670" y="120" width="110" height="52" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="725" y="142" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">输出 O</text>
    <text x="725" y="160" text-anchor="middle" style="fill:var(--muted);font-size:10px">本瓦片结果</text>
    <text x="24" y="288" style="fill:var(--faint);font-size:11px">完整分数矩阵从不落 HBM —— 读一块、算一块、丢一块</text>
  </svg>
  <div class="figcap"><b>图 2 · 分块注意力内循环</b> — 一个 Q 瓦片流式遍历各 KV 瓦片：每块算 <span class="mono">S=Q·Kᵀ</span>，用在线 softmax 更新运行最大值 <span class="mono">m</span> 与分母 <span class="mono">ℓ</span>（按比例缩放累加器），再把 <span class="mono">·V</span> 累加进 <span class="mono">O</span>；完整分数矩阵从不落 HBM。</div>
</div>

<p>这套「分块 + 在线 softmax + 分页收集」并不限于 decode。prefill/extend 阶段也走同一套分块 flash-attention，只是查询位置更多、形状更宽。下面这段 <span class="mono">extend_attention_fwd</span> 是 SGLang 里 extend 阶段的 Triton 核函数入口：它用 <span class="mono">kv_indptr</span>/<span class="mono">kv_indices</span> 把已缓存的分页 K、V 收集回来，再按瓦片做 Q·Kᵀ → 在线 softmax → ·V，最后把结果写进 <span class="mono">o_extend</span>：</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/triton_ops/extend_attention.py ::extend_attention_fwd</span><span class="ln">Triton 分块 flash-attention（extend/prefill 阶段）</span></div><pre>def extend_attention_fwd(q_extend, k_extend, v_extend, o_extend,
                         k_buffer, v_buffer,
                         qo_indptr, kv_indptr, kv_indices, ...):
    # Triton flash-attention for the EXTEND (prefill) phase.
    # gather cached K/V by kv_indices (paged), then per Q tile:
    #   tiled Q·Kᵀ -&gt; online softmax (running max/sum) -&gt; ·V,
    # streaming over KV tiles so full scores never hit HBM.
    # writes the attention output into o_extend.
    ...
</pre></div>

<p>举个具体的数字感受分块的价值：一条 4000 token 的序列，若把完整分数矩阵物化出来是 4000×4000≈1600 万个分数；而分块只需同时持有一个瓦片，比如 128×128≈1.6 万个——内存从 <span class="mono">O(seq²)</span> 直接降到 <span class="mono">O(tile)</span>，整整小了三个数量级，这正是一遍流式扫描能放进 SRAM 的原因。</p>

<p>再看 <span class="mono">kv_indices</span> 的威力：同样这条 4000 token 的上下文，按每页 16 个 token 切，就是 250 个分页，物理上散落在 KV 池各处；核函数读 <span class="mono">kv_indices</span> 这 250 个页号，就能把它们当作一条连续序列来注意，既不必预留 4000 token 的连续显存，也不漏掉任何一个历史 token。</p>

<p>还有一个值得记住的细节：在线 softmax 不仅让一遍扫描成为可能，它还和分页、分片自然衔接。当一条很长的序列被切成很多 KV 分片，甚至这些分片被分配给不同的线程块并行处理时，每个分片只能算出自己那段的<strong>局部最大值和局部分母</strong>。要得到全局正确的结果，就需要把这些局部 softmax 结果<strong>按在线 softmax 的规则合并</strong>——这正是后面会提到的 <span class="mono">merge_attn_states.cu</span> 干的事。所以在线 softmax 不是一个孤立的小技巧，而是贯穿「分块、分片、并行合并」的同一套数学骨架。理解了它，你就能把核函数内部的循环和跨线程块的归并看成同一件事的不同尺度。</p>

<h2>四、一个真实的 wrinkle：MLA 的压缩潜变量布局</h2>
<p>把上面所有概念落到一段真实代码上，我们看 DeepSeek 风格的 <strong>MLA</strong>（多头潜在注意力）的 decode 核函数。MLA 有个现实中的小皱褶：它的 KV 不是普通地存成「每个头一份 K、一份 V」，而是存成一个<strong>压缩潜变量</strong>（latent，<span class="mono">D_latent=512</span>）外加一小段 <strong>rope（旋转位置编码）部分</strong>（<span class="mono">D_rope=64</span>），合起来每个位置 576 维。核函数通过 page_table 从 <span class="mono">kv_c_and_k_pe_cache</span> 里读出这份压缩 KV，再把头数<strong>补齐</strong>到核函数的瓦片宽度 <span class="mono">MAX_HEADS=128</span>，然后把活儿派发给编译好的 CUDA 核函数。下面这段 Python 就是这层「准备数据 + 派发」的封装，真正的乘加发生在 <span class="mono">torch.ops.sgl_kernel.cutlass_mla_decode</span> 背后的编译核函数里：</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/attention.py ::cutlass_mla_decode</span><span class="ln">按 page_table 收集分页 KV → 分块 Q·Kᵀ → 在线 softmax → ·V</span></div><pre>def cutlass_mla_decode(
    q_nope, q_pe,                 # query: no-pos-emb part (nope, dim 512) + rope part
    kv_c_and_k_pe_cache,         # paged KV: latent (c) + key-rope; middle dim is PAGE_SIZE
    seq_lens, page_table,         # page_table[b] -&gt; the KV page ids for sequence b
    workspace, sm_scale, num_kv_splits=1,
):
    B_q, H, D_q_nope = q_nope.shape
    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape
    D_latent, D_rope = 512, 64        # MLA: 512 latent + 64 rope = 576
    MAX_HEADS = 128
    if H &lt; MAX_HEADS:               # pad heads up to the kernel's tile width
        q_nope = pad_to(q_nope, MAX_HEADS)
        q_pe   = pad_to(q_pe,   MAX_HEADS)
    out = q_nope.new_empty((B_q, MAX_HEADS, D_latent))
    # the compiled CUDA kernel: gather KV pages via page_table, then
    # tiled Q.K^T -&gt; online softmax -&gt; .V, writing into out
    torch.ops.sgl_kernel.cutlass_mla_decode.default(
        out, q_nope, q_pe, kv_c_and_k_pe_cache,
        seq_lens, page_table, workspace, sm_scale, num_kv_splits)
    return out[:, :H].contiguous()    # drop the head padding
</pre></div>

<p>注意这段 Python 本身<strong>几乎不做数学</strong>：它读形状、定下 MLA 的潜变量与 rope 维度、把头数补齐到 128、开好输出张量，然后把所有指针交给编译核函数。真正的「gather → 分块点积 → 在线 softmax → 加权」全在 <span class="mono">torch.ops.sgl_kernel.cutlass_mla_decode</span> 背后的 C++/CUDA 里。这些 CUDA 源码就放在 <span class="mono">sgl-kernel/csrc/attention/</span> 目录下，例如 <span class="mono">cutlass_mla_kernel.cu</span>、<span class="mono">merge_attn_states.cu</span>（后者负责把多个 KV 分片各自算出的局部 softmax 结果按在线 softmax 规则合并起来）。本课不贴 CUDA，只指路；想深入可以顺着这个路径去读。</p>

<p>为什么 MLA 要这么折腾？因为它把每个头原本各自一份的 K、V，压缩成了一份共享的低维潜变量，再加一小段承载位置信息的 rope。这样存下来的 KV 体积大幅变小，decode 阶段要搬运的字节随之减少——对一个带宽瓶颈的负载来说，这是直接的性能收益。代价是核函数读到的不再是「现成的每头 K、V」，而是压缩布局，需要在核函数内部按 MLA 的方式展开和计算。这正是「真实核函数的皱褶」：教科书公式干干净净，工程实现却要迁就具体的存储布局、对齐要求和瓦片宽度。把头数补齐到 <span class="mono">MAX_HEADS=128</span> 也是同理——编译核函数是按固定瓦片宽度优化的，不足就补齐、算完再把多余的头切掉（<span class="mono">out[:, :H]</span>），用一点点冗余换取核函数走在它最擅长的形状上。这层封装本身被注意力后端（第33课）调用，而当核函数已经足够精炼，下一步要省的就是反复启动它们的开销，那正是 CUDA Graph（第41课）的主场。</p>

<p>最后把几个核心设计选择和它们的理由列成一张表，方便回顾：</p>

<table class="t"><tr><th>设计选择</th><th>为什么这么做</th></tr>
<tr><td>分块 tiling（按瓦片流式处理）</td><td>SRAM/寄存器装不下完整 KV 与分数矩阵；分块让数据刚好放进片上、可流式覆盖</td></tr>
<tr><td>在线 softmax（运行最大值 + 运行分母）</td><td>一遍扫完即得正确归一化，无需存全部分数，省显存、省带宽</td></tr>
<tr><td>分页 KV 布局 + page_table 收集</td><td>序列 KV 非连续存放，显存利用率高、可共享前缀；核函数靠块表还原逻辑顺序</td></tr>
<tr><td>MLA 压缩潜变量（512 + 64）</td><td>KV 体积更小、搬运更省带宽；核函数读压缩布局并补齐头数到 128 的瓦片宽度</td></tr>
</table>

<p>把这些放在一起看：decode 注意力核函数的设计哲学，从头到尾都是<strong>围绕带宽做减法</strong>——少存、少搬、一遍算完。理解了这条主线，你就能看懂注意力后端（第33课）为什么长那样，也为后面要讲的 CUDA Graph（第41课）打好基础——当核函数本身已经足够精炼，下一步要优化的就是「反复启动这些核函数」的开销了。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>decode 核函数的真正第一步是收集，不是相乘</strong>：先用 <span class="mono">page_table</span> 从分页 KV 缓存（第30课）里 gather 非连续的 K、V 页，再做 Q·Kᵀ → softmax → ·V。</li>
<li><strong>分块 tiling</strong>：把 KV 切成瓦片流进 SRAM，绝不物化完整分数矩阵，让数据刚好放进寄存器/共享内存。</li>
<li><strong>在线 softmax</strong>（FlashAttention 风格）：维护运行最大值 + 运行分母，一遍流式扫描得到等价归一化，省显存省带宽。</li>
<li><strong>prefill 与 decode 核函数不同</strong>：prefill 是又宽又厚的类 GEMM 稠密计算；decode 又瘦又长、以 KV gather 为主，受带宽限制（第4课）。</li>
<li><strong>MLA 的现实皱褶</strong>：KV 存成压缩潜变量（512）+ rope（64），核函数读 <span class="mono">kv_c_and_k_pe_cache</span>、补齐头数到 128 再派发；封装在注意力后端（第33课）里，CUDA 源码位于 <span class="mono">sgl-kernel/csrc/attention/</span>。前向引用 CUDA Graph（第41课）。</li>
</ul></div>
""", "en": r"""
<p class="lead">In Lesson 40 we zoom all the way in and dissect a real <strong>attention kernel</strong>. Earlier lessons covered the paged KV cache (Lesson 30), why decode is bandwidth-bound (Lesson 4), and how attention backends wrap kernels (Lesson 33). Now we put the pieces together: when a sequence runs an attention step during <span class="mono">decode</span> (generating one new token per step), what actually happens inside that kernel that gets called tens of thousands of times on the GPU? No hand-waving — just which bytes this code moves at each step, and why.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a decode attention kernel as a <strong>warehouse picker</strong>. The shelves (the KV cache) are not one long contiguous row; they are cut into <strong>bins</strong> (pages) scattered around the warehouse. The picker holds a <strong>pick list</strong> (<span class="mono">page_table</span>, the block table) that says "this customer's goods are in bins 7, 3, 19." He does not empty the whole warehouse — he <strong>fetches only those bins</strong> and tallies as he goes.</p>
<p>Even better is how he tallies: instead of weighing everything at the end, he <strong>updates a running total after each bin</strong> — remembering the current maximum and the accumulated weight, rescaling the earlier result whenever a bigger value shows up. He only ever needs a tiny notepad (registers) and never spreads the whole ledger on the floor. That is exactly <strong>online softmax</strong>.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>The math of attention is short: <span class="mono">softmax(Q·Kᵀ / √d)·V</span>. But in decode the hard part is not arithmetic — it is <strong>where the data lives and how to move it</strong>. For each generated token, Q is just one row (one query position), yet it must dot against <strong>every K and V in this sequence's history</strong>. Those K/V were written long ago into the paged KV cache, physically scattered across non-contiguous pages. So the kernel's real main line is: <strong>gather the scattered KV pages via page_table → stream Q·Kᵀ in tiles → online softmax → multiply by V</strong>. Because decode is bandwidth-bound, the kernel's whole job is to move KV bytes efficiently.</p>
</div>

<h2>1. The main line: gather, dot, normalise, weight</h2>
<p>Let's state the main line clearly. When the scheduler decides to generate the next token for a sequence, it launches a decode forward. The attention step receives one row of <strong>Q</strong> (the current query position, a very "skinny" shape with a single token) plus the entire sequence's <strong>already-cached K and V</strong>. Those K/V do not sit in one contiguous block; the paged KV cache slices them into fixed-size pages stored apart, and which pages a sequence owns is recorded in the <span class="mono">page_table</span>.</p>
<p>So the kernel's first job is not multiplication but <strong>address lookup</strong>: use <span class="mono">page_table[b]</span> to find all KV page ids for sequence b, then read the K/V blocks from those non-contiguous pages. Only then does the math begin: the current Q row dots against the block of K just read, producing a slice of attention scores; that slice immediately feeds online softmax to update the running totals; finally the normalised weights scale the matching V block and accumulate into the output. The whole thing is <strong>streaming</strong> — read a block, compute it, drop it — never keeping the full score matrix in memory.</p>

<div class="flow"><div class="node">gather paged KV via page_table</div><div class="arrow">→</div><div class="node">Q·Kᵀ tiled dot</div><div class="arrow">→</div><div class="node">online softmax normalise</div><div class="arrow">→</div><div class="node">·V weighted accumulate → output</div></div>

<p>Notice the first node: in many textbook attention diagrams step one is "Q times K," but in a real decode kernel <strong>the true first step is "gather the KV per the pick list."</strong> This step sets the kernel's memory-access pattern and decides whether it can saturate memory bandwidth. Since the KV cache is usually far larger than on-chip SRAM, how pages are brought in — and at what granularity — is essentially performance itself.</p>

<h2>2. Why KV is paged, and how the kernel "picks by page"</h2>
<p>As Lesson 30 explained, the paged KV cache cuts each sequence's history into fixed-size pages, allocated by page like an OS virtual-memory system. The upside: sequence lengths vary wildly, so per-page allocation avoids reserving huge contiguous memory for long sequences and avoids waste for short ones — high memory utilisation, plus prefix sharing across sequences. The cost: <strong>a single sequence's KV is no longer physically contiguous</strong>; logical KV blocks 0, 1, 2, 3 might land on physical pages 7, 3, 19, 5.</p>
<p>To restore logical order the kernel relies on the <span class="mono">page_table</span> (also called the block table): a "logical block id → physical page id" map. The kernel walks the logical positions a sequence needs, translates them through the page_table into physical page addresses, and reads K/V there. The cells below show paged KV gathered by a page_table — logically adjacent pages fetched out of physical order:</p>

<div class="cellgroup"><div class="cell">page #7</div><div class="cell">page #3</div><div class="cell">page #19</div><div class="cell">page #5</div><div class="cell">page #12</div><div class="cell">page #0</div></div>

<div class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-label="Paged-KV gather: a contiguous logical sequence maps through the page_table kv_indices to physically scattered pages in the KV pool, and the kernel gathers the right pages before attending">
    <text x="24" y="28" style="font-weight:700;fill:var(--muted)">logical contiguous → kv_indices → physically scattered</text>
    <text x="24" y="56" style="fill:var(--muted);font-size:11px">logical sequence (contiguous)</text>
    <rect x="24" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="60" y="86" text-anchor="middle" class="mono" style="font-size:11px">L0</text>
    <rect x="110" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="146" y="86" text-anchor="middle" class="mono" style="font-size:11px">L1</text>
    <rect x="196" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="232" y="86" text-anchor="middle" class="mono" style="font-size:11px">L2</text>
    <rect x="282" y="64" width="72" height="34" rx="6" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="318" y="86" text-anchor="middle" class="mono" style="font-size:11px">L3</text>
    <line x1="60" y1="98" x2="60" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="146" y1="98" x2="146" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="232" y1="98" x2="232" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="318" y1="98" x2="318" y2="146" style="stroke:var(--line);stroke-width:1.5"/>
    <text x="24" y="138" style="fill:var(--muted);font-size:11px">page_table (kv_indices)</text>
    <rect x="24" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="60" y="170" text-anchor="middle" class="mono" style="font-size:11px">7</text>
    <rect x="110" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="146" y="170" text-anchor="middle" class="mono" style="font-size:11px">3</text>
    <rect x="196" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="232" y="170" text-anchor="middle" class="mono" style="font-size:11px">19</text>
    <rect x="282" y="148" width="72" height="34" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="318" y="170" text-anchor="middle" class="mono" style="font-size:11px">5</text>
    <rect x="424" y="44" width="356" height="242" rx="12" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="440" y="64" style="fill:var(--faint);font-size:11px">KV pool (scattered pages)</text>
    <rect x="446" y="80" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="488" y="103" text-anchor="middle" class="mono" style="font-size:11px">page #3</text>
    <rect x="600" y="74" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="642" y="97" text-anchor="middle" class="mono" style="font-size:11px">page #19</text>
    <rect x="470" y="152" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="512" y="175" text-anchor="middle" class="mono" style="font-size:11px">page #7</text>
    <rect x="636" y="150" width="84" height="38" rx="6" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="678" y="173" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">page #0</text>
    <rect x="540" y="224" width="84" height="38" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="582" y="247" text-anchor="middle" class="mono" style="font-size:11px">page #5</text>
    <rect x="690" y="224" width="84" height="38" rx="6" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="732" y="247" text-anchor="middle" class="mono" style="font-size:11px;fill:var(--faint)">page #12</text>
    <line x1="60" y1="182" x2="512" y2="171" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="146" y1="182" x2="488" y2="99" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="232" y1="182" x2="642" y2="93" style="stroke:var(--teal);stroke-width:1.5"/>
    <line x1="318" y1="182" x2="582" y2="243" style="stroke:var(--teal);stroke-width:1.5"/>
    <text x="24" y="306" style="fill:var(--faint);font-size:11px">kernel gathers these pages by kv_indices → attends as one sequence</text>
  </svg>
  <div class="figcap"><b>Fig 1 · Paged-KV gather</b> — a logically contiguous sequence (<span class="mono">L0…L3</span>) is translated through the <span class="mono">page_table</span> (the <span class="mono">kv_indices</span>) into physically scattered KV pages (#7, #3, #19, #5); the kernel first gathers those scattered pages by <span class="mono">kv_indices</span> before it can attend to them as one sequence.</div>
</div>

<p>This "jump-around" access is the biggest engineering challenge of a decode kernel. Done crudely, page jumps make access scattered and bandwidth poor; done well, the kernel lets one thread block own one sequence, or coalesces several page reads into aligned, contiguous memory transactions, using every memory read fully. That is why the very same attention math runs at wildly different speeds across backends (Lesson 33) — almost all the difference is KV access efficiency.</p>

<h2>3. Tiling and online softmax: done in one streaming pass</h2>
<p>The second key design is <strong>tiling</strong>. A GPU's registers and shared memory (SRAM) are tiny — they cannot hold the full K/V, let alone the whole attention score matrix of a long sequence. So the kernel cuts the KV into <strong>tiles</strong> and streams one tile into SRAM at a time, overwriting it with the next. The kernel <strong>never materialises the full score matrix</strong> at once, because that matrix would occupy memory and be read and written repeatedly — pure wasted bandwidth.</p>
<p>But tiling creates a math problem: softmax needs the maximum of all scores and the sum of all exponentials before it can normalise. If we process in tiles, how can we "see all scores first"? The answer is FlashAttention-style <strong>online softmax</strong>: the kernel keeps two running quantities — a <strong>running max</strong> and a <strong>running denominator</strong> (the accumulated sum of exponentials). For each new tile, it corrects the old totals using the new local max (rescaling the previously accumulated result by an exponential factor) and adds this tile's contribution. Thus <strong>one streaming pass</strong> suffices, with no need to store all scores, yet the result is exactly equivalent to a global softmax — saving memory and bandwidth.</p>
<p>This also explains why prefill and decode kernels look different. Here they are side by side:</p>

<div class="cols"><div class="col"><strong>prefill kernel</strong><br>Processes all prompt tokens, with many query positions — a wide, thick, GEMM-like dense computation. Compute-heavy, it can keep tensor cores fed; the bottleneck leans toward arithmetic.</div><div class="col"><strong>decode kernel</strong><br>One query position per step, a skinny and long shape, dominated by gathering over the KV cache via page_table. Tiny compute, large access; the bottleneck is memory bandwidth (Lesson 4).</div></div>

<p>Because their shapes and bottlenecks differ entirely, SGLang takes different kernel paths for them: prefill like one big GEMM, decode like a large sparse pick. Forcing both into one implementation usually leaves both running slowly.</p>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Tiled attention inner loop: one Q tile streams over the KV tiles, each tile computes S equals Q times K transpose, online softmax updates the running max and sum, then times V is accumulated into the accumulator; the full score matrix never hits HBM">
    <text x="24" y="30" style="font-weight:700;fill:var(--muted)">one Q tile → stream over KV tiles</text>
    <rect x="24" y="86" width="92" height="120" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="70" y="140" text-anchor="middle" class="mono" style="font-size:12px">Q tile</text>
    <text x="70" y="160" text-anchor="middle" style="fill:var(--muted);font-size:11px">one query row</text>
    <line x1="116" y1="146" x2="148" y2="146" style="stroke:var(--line);stroke-width:2"/>
    <rect x="150" y="52" width="486" height="208" rx="10" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5;stroke-dasharray:6 5"/>
    <text x="166" y="74" style="fill:var(--accent-ink);font-weight:700;font-size:12px">loop: over KV tiles</text>
    <rect x="166" y="90" width="86" height="46" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="209" y="111" text-anchor="middle" class="mono" style="font-size:11px">KV tile</text>
    <text x="209" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">K, V block</text>
    <line x1="252" y1="113" x2="276" y2="113" style="stroke:var(--line);stroke-width:2"/>
    <rect x="278" y="90" width="92" height="46" rx="6" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="324" y="111" text-anchor="middle" class="mono" style="font-size:11px">S=Q·Kᵀ</text>
    <text x="324" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">tile scores</text>
    <line x1="370" y1="113" x2="394" y2="113" style="stroke:var(--line);stroke-width:2"/>
    <rect x="396" y="90" width="104" height="46" rx="6" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="448" y="111" text-anchor="middle" style="font-size:11px">online softmax</text>
    <text x="448" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">update m, ℓ</text>
    <line x1="500" y1="113" x2="524" y2="113" style="stroke:var(--line);stroke-width:2"/>
    <rect x="526" y="90" width="92" height="46" rx="6" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="572" y="111" text-anchor="middle" class="mono" style="font-size:11px">·V accum</text>
    <text x="572" y="127" text-anchor="middle" style="fill:var(--muted);font-size:10px">rescale O</text>
    <rect x="166" y="172" width="452" height="66" rx="8" style="fill:var(--panel);stroke:var(--line);stroke-width:1.5"/>
    <text x="182" y="192" style="fill:var(--muted);font-size:11px">running state (kept in registers)</text>
    <rect x="182" y="202" width="120" height="26" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="242" y="219" text-anchor="middle" class="mono" style="font-size:11px">m running max</text>
    <rect x="312" y="202" width="118" height="26" rx="5" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="371" y="219" text-anchor="middle" class="mono" style="font-size:11px">ℓ running sum</text>
    <rect x="440" y="202" width="160" height="26" rx="5" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="520" y="219" text-anchor="middle" class="mono" style="font-size:11px">O accumulator</text>
    <line x1="636" y1="146" x2="668" y2="146" style="stroke:var(--line);stroke-width:2"/>
    <rect x="670" y="120" width="110" height="52" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="725" y="142" text-anchor="middle" style="fill:var(--accent-ink);font-weight:700;font-size:12px">output O</text>
    <text x="725" y="160" text-anchor="middle" style="fill:var(--muted);font-size:10px">this tile</text>
    <text x="24" y="288" style="fill:var(--faint);font-size:11px">full score matrix never hits HBM — read, compute, drop each tile</text>
  </svg>
  <div class="figcap"><b>Fig 2 · Tiled attention inner loop</b> — one Q tile streams over the KV tiles: each tile computes <span class="mono">S=Q·Kᵀ</span>, online softmax updates the running max <span class="mono">m</span> and sum <span class="mono">ℓ</span> (rescaling the accumulator), then <span class="mono">·V</span> is accumulated into <span class="mono">O</span>; the full score matrix never hits HBM.</div>
</div>

<p>This trio — tiling, online softmax, paged gather — is not unique to decode. The prefill/extend phase runs the same tiled flash-attention, just with more query positions and a wider shape. The <span class="mono">extend_attention_fwd</span> below is SGLang's Triton kernel entry for the extend phase: it gathers the already-cached paged K/V via <span class="mono">kv_indptr</span>/<span class="mono">kv_indices</span>, then does tiled Q·Kᵀ → online softmax → ·V per tile, writing the result into <span class="mono">o_extend</span>:</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/attention/triton_ops/extend_attention.py ::extend_attention_fwd</span><span class="ln">Triton tiled flash-attention (extend/prefill phase)</span></div><pre>def extend_attention_fwd(q_extend, k_extend, v_extend, o_extend,
                         k_buffer, v_buffer,
                         qo_indptr, kv_indptr, kv_indices, ...):
    # Triton flash-attention for the EXTEND (prefill) phase.
    # gather cached K/V by kv_indices (paged), then per Q tile:
    #   tiled Q·Kᵀ -&gt; online softmax (running max/sum) -&gt; ·V,
    # streaming over KV tiles so full scores never hit HBM.
    # writes the attention output into o_extend.
    ...
</pre></div>

<p>Concrete numbers show why tiling matters: for a 4000-token sequence, materialising the full score matrix is 4000×4000 ≈ 16M scores; tiling only ever holds one tile, say 128×128 ≈ 16K — memory drops from <span class="mono">O(seq²)</span> to <span class="mono">O(tile)</span>, three orders of magnitude smaller, which is exactly why one streaming pass fits in SRAM.</p>

<p>And the power of <span class="mono">kv_indices</span>: that same 4000-token context, cut into 16-token pages, is 250 pages scattered across the KV pool; the kernel reads those 250 page ids from <span class="mono">kv_indices</span> and attends to them as one contiguous sequence — no need to reserve 4000 tokens of contiguous memory, and not a single history token is missed.</p>

<h2>4. A real wrinkle: MLA's compressed latent layout</h2>
<p>To ground all of this in real code, look at the DeepSeek-style <strong>MLA</strong> (multi-head latent attention) decode kernel. MLA has a real-world wrinkle: its KV is not stored plainly as "one K, one V per head," but as a <strong>compressed latent</strong> (<span class="mono">D_latent=512</span>) plus a small <strong>rope part</strong> (<span class="mono">D_rope=64</span>), 576 dims per position together. The kernel reads this compressed KV from <span class="mono">kv_c_and_k_pe_cache</span> via the page_table, <strong>pads</strong> the head count up to the kernel's tile width <span class="mono">MAX_HEADS=128</span>, then dispatches to the compiled CUDA kernel. The Python below is that "prepare data + dispatch" wrapper; the real multiply-accumulate happens inside the compiled kernel behind <span class="mono">torch.ops.sgl_kernel.cutlass_mla_decode</span>:</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/attention.py ::cutlass_mla_decode</span><span class="ln">gather paged KV via page_table → tiled Q·Kᵀ → online softmax → ·V</span></div><pre>def cutlass_mla_decode(
    q_nope, q_pe,                 # query: no-pos-emb part (nope, dim 512) + rope part
    kv_c_and_k_pe_cache,         # paged KV: latent (c) + key-rope; middle dim is PAGE_SIZE
    seq_lens, page_table,         # page_table[b] -&gt; the KV page ids for sequence b
    workspace, sm_scale, num_kv_splits=1,
):
    B_q, H, D_q_nope = q_nope.shape
    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape
    D_latent, D_rope = 512, 64        # MLA: 512 latent + 64 rope = 576
    MAX_HEADS = 128
    if H &lt; MAX_HEADS:               # pad heads up to the kernel's tile width
        q_nope = pad_to(q_nope, MAX_HEADS)
        q_pe   = pad_to(q_pe,   MAX_HEADS)
    out = q_nope.new_empty((B_q, MAX_HEADS, D_latent))
    # the compiled CUDA kernel: gather KV pages via page_table, then
    # tiled Q.K^T -&gt; online softmax -&gt; .V, writing into out
    torch.ops.sgl_kernel.cutlass_mla_decode.default(
        out, q_nope, q_pe, kv_c_and_k_pe_cache,
        seq_lens, page_table, workspace, sm_scale, num_kv_splits)
    return out[:, :H].contiguous()    # drop the head padding
</pre></div>

<p>Notice this Python does <strong>almost no math</strong>: it reads shapes, fixes MLA's latent and rope dims, pads heads to 128, allocates the output tensor, then hands all the pointers to the compiled kernel. The real "gather → tiled dot → online softmax → weight" all lives in the C++/CUDA behind <span class="mono">torch.ops.sgl_kernel.cutlass_mla_decode</span>. Those CUDA sources sit under <span class="mono">sgl-kernel/csrc/attention/</span> — files like <span class="mono">cutlass_mla_kernel.cu</span> and <span class="mono">merge_attn_states.cu</span> (the latter merges per-KV-split local softmax results under online-softmax rules). We won't paste CUDA here, just point the way; follow the path to dig deeper.</p>

<p>Finally, a table of the core design choices and their reasons, for review:</p>

<table class="t"><tr><th>Design choice</th><th>Why</th></tr>
<tr><td>Tiling (stream by tile)</td><td>SRAM/registers can't hold the full KV or score matrix; tiling fits data on-chip and lets it stream-overwrite</td></tr>
<tr><td>Online softmax (running max + running denom)</td><td>One pass yields correct normalisation without storing all scores — saves memory and bandwidth</td></tr>
<tr><td>Paged KV layout + page_table gather</td><td>Sequence KV stored non-contiguously for high utilisation and prefix sharing; the block table restores logical order</td></tr>
<tr><td>MLA compressed latent (512 + 64)</td><td>Smaller KV, cheaper to move; the kernel reads the compressed layout and pads heads to the 128 tile width</td></tr>
</table>

<p>Seen together: a decode attention kernel's whole design philosophy is <strong>subtraction around bandwidth</strong> — store less, move less, finish in one pass. Grasp this main line and you'll understand why attention backends (Lesson 33) look the way they do, and you'll be ready for CUDA Graph (Lesson 41) next — once the kernel itself is lean enough, the next thing to optimise is the cost of launching these kernels over and over.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>A decode kernel's true first step is gather, not multiply</strong>: use <span class="mono">page_table</span> to gather non-contiguous K/V pages from the paged KV cache (Lesson 30), then do Q·Kᵀ → softmax → ·V.</li>
<li><strong>Tiling</strong>: stream KV tiles into SRAM and never materialise the full score matrix, fitting data into registers/shared memory.</li>
<li><strong>Online softmax</strong> (FlashAttention-style): keep a running max + running denominator for an equivalent normalisation in one streaming pass, saving memory and bandwidth.</li>
<li><strong>Prefill and decode kernels differ</strong>: prefill is a wide, thick GEMM-like dense pass; decode is skinny and long, gather-heavy, bandwidth-bound (Lesson 4).</li>
<li><strong>MLA's real wrinkle</strong>: KV stored as a compressed latent (512) + rope (64); the kernel reads <span class="mono">kv_c_and_k_pe_cache</span>, pads heads to 128, then dispatches; wrapped by attention backends (Lesson 33), with CUDA sources under <span class="mono">sgl-kernel/csrc/attention/</span>. Forward-ref CUDA Graph (Lesson 41).</li>
</ul></div>
"""}
LESSON_41 = {"zh": r"""
<p class="lead">这一课我们把目光从"算法层"彻底沉到"内核层"，讲两种最贴近硬件的提速手段——<strong>算子融合（operator fusion）</strong>与 <strong>CUDA Graph（捕获/重放）</strong>——以及它们如何<strong>互相配合</strong>，共同把解码阶段的吞吐推到极限。融合解决的是"一次计算里访存太多、启动太多"的问题；CUDA Graph 解决的是"CPU 反复提交内核、提交本身成了瓶颈"的问题。两者叠加，才有了今天 SGLang 解码路径那种近乎零开销的内核流水线。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div><p>把一次内核启动想象成"去仓库取一趟货"。<strong>未融合</strong>就像：先去仓库取原料（读 HBM），加工成半成品后再送回仓库（写 HBM），下一道工序又把半成品取出来（再读 HBM）继续做。光是来回搬运就耗掉大量时间，而 GPU 的"仓库"（显存）带宽是有限的。<strong>融合</strong>则像：把相邻几道工序合并在同一个车间一次做完，半成品直接放在手边的工作台（寄存器）上，不再往返仓库。</p><p>而 <span class="mono">CUDA Graph</span> 像是把"今天要跑的整条流水线动作"提前录像一遍：每次开工不用工头（CPU）逐个喊口令，直接<strong>放录像（replay）</strong>，机器照着做即可。但录像有个前提——动作必须每次<strong>一模一样</strong>（形状固定、没有临场看数据才决定的分支），否则录像就对不上了。</p></div>

<div class="card macro"><div class="tag">🌍 宏观理解</div><p>回顾<span class="mono">第4课</span>：解码阶段是<strong>带宽受限（bandwidth-bound）</strong>的，每个 token 的实际算力利用率很低，瓶颈在搬数据和发指令。于是任何能<strong>减少 HBM 往返</strong>、<strong>减少内核启动次数</strong>的手段都直接转化为吞吐。算子融合减少前者与后者，CUDA Graph（<span class="mono">第27课</span>）几乎消灭后者中"CPU 提交"的那部分。这些被融合、被捕获的内核，最终来自<span class="mono">第38课（AOT 预编译）</span>与<span class="mono">第39课（JIT 即时编译）</span>——融合与图只是更好地"编排"它们。</p></div>

<h2>一、算子融合：把多个小内核合成一个</h2>
<p>融合的核心动机有两条。第一是<strong>访存（HBM round-trip）</strong>：未融合时，每个算子都要把输入从显存读进来、把输出写回显存，下一个算子又要重新读一遍。这些中间结果（临时张量）对最终答案没有价值，却白白占用了宝贵的显存带宽。第二是<strong>启动开销（launch overhead）</strong>：每发起一个内核，CPU 都要做一次提交，GPU 都要做一次调度；在解码这种单步计算量很小的场景下，启动开销可能比真正的计算还要久。</p>
<p>以 SGLang 里最典型的 <span class="mono">SiluAndMul</span> 为例（门控激活，gate×up）。未融合的参考实现 <span class="mono">forward_native</span> 是"<strong>先 silu 再相乘</strong>"两步：先对前半段做 <span class="mono">F.silu(gate)</span> 得到一个临时张量并写回 HBM，再把它读出来与后半段 <span class="mono">up</span> 逐元素相乘。这意味着<strong>两趟内核、一次额外的临时张量写入与读取</strong>。融合内核 <span class="mono">forward_cuda</span> 则在<strong>一个 kernel</strong> 里同时完成 silu 与乘法：数据读进寄存器后，silu 的结果直接留在<strong>寄存器</strong>里参与乘法，从不落回显存，最后只写一次最终输出。</p>
<p>SGLang 中类似的融合还有很多：<span class="mono">fused_add_rmsnorm</span>（把残差相加与 RMSNorm 合成一个内核，见<span class="mono">第36课</span>）、把 q/k 的归一化与 RoPE 旋转位置编码融合（fused qk-norm+rope）等。它们的共同点都是：<strong>把"写出去又读回来"的中间张量消灭在寄存器/共享内存里</strong>，并把多次启动压成一次。</p>
<p>为什么访存往返如此致命？因为现代 GPU 的算力增长速度远远快于显存带宽的增长速度，所谓"内存墙"。当一个算子真正要做的乘加运算很少，它的执行时间几乎完全由"从显存读多少、往显存写多少"决定。未融合的两趟实现，等于把一份本来可以留在芯片内部的数据，先搬下楼再搬上楼：silu 的临时张量先被写进 HBM，紧接着又被原封不动地读回来，这一来一回的带宽全是浪费。融合之所以快，本质就是把这份"中间数据"锁在离计算单元最近的寄存器或共享内存里，让它在被消费之前从不离开芯片，从而把宝贵的显存带宽只留给真正必须读写的输入和最终输出。</p>
<p>再看启动开销这一侧。每一次内核启动，CPU 都要走一遍"准备参数、提交到 GPU 队列"的流程，GPU 也要为这次启动做调度与同步。单看一次也许只有几微秒，但解码阶段一个模型前向里可能要发起成百上千个内核，而每步又只解码一个 token——把这些零碎的启动开销累加起来，往往能和真正的计算时间相提并论，甚至更高。融合把"多次启动"压成"一次启动"，正是从根上削减了这部分固定成本。这也解释了为什么融合在解码阶段的收益，常常比在预填充阶段更明显：预填充计算量大、启动开销被摊薄，而解码恰恰是启动开销最敏感的地方。</p>

<div class="flow"><div class="node">读 gate/up (HBM)</div><div class="arrow">→</div><div class="node">silu kernel</div><div class="arrow">→</div><div class="node">写临时张量 (HBM)</div><div class="arrow">→</div><div class="node">读临时 + up (HBM)</div><div class="arrow">→</div><div class="node">mul kernel</div><div class="arrow">→</div><div class="node">写输出 (HBM)</div></div>
<div class="flow"><div class="node">读 x (HBM)</div><div class="arrow">→</div><div class="node">融合 kernel：silu 与 mul 同在寄存器</div><div class="arrow">→</div><div class="node">写输出 (HBM)</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="未融合需多次 HBM 往返，融合在单个 kernel 内于寄存器完成">
    <text x="40" y="34" style="font-weight:700;fill:var(--red)">未融合：op → HBM → op → HBM</text>
    <rect x="40" y="52" width="110" height="46" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="95" y="80" text-anchor="middle">相加</text>
    <text x="161" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="172" y="52" width="110" height="46" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="227" y="80" text-anchor="middle" class="mono" style="font-size:12px">写 HBM</text>
    <text x="293" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="304" y="52" width="110" height="46" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="359" y="80" text-anchor="middle" class="mono" style="font-size:12px">读 HBM</text>
    <text x="425" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="436" y="52" width="110" height="46" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="491" y="80" text-anchor="middle">RMSNorm</text>
    <text x="557" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="568" y="52" width="110" height="46" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="623" y="80" text-anchor="middle" class="mono" style="font-size:12px">写 HBM</text>
    <text x="40" y="128" style="fill:var(--red);font-size:12px">中间张量反复进出显存，访存往返主导耗时</text>
    <text x="40" y="176" style="font-weight:700;fill:var(--teal)">融合：单个 kernel（读一次 · 写一次）</text>
    <rect x="40" y="194" width="150" height="56" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="115" y="227" text-anchor="middle">读一次</text>
    <text x="202" y="227" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="215" y="194" width="290" height="56" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="360" y="220" text-anchor="middle">单 kernel：相加 + RMSNorm</text>
    <text x="360" y="240" text-anchor="middle" style="fill:var(--muted);font-size:12px">全程留在寄存器</text>
    <text x="517" y="227" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="530" y="194" width="150" height="56" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="605" y="227" text-anchor="middle">写一次</text>
    <text x="40" y="282" style="fill:var(--teal);font-size:12px">省去中间 2 次写 + 1 次读 HBM</text>
  </svg>
  <div class="figcap"><b>图 1 · 未融合 vs 融合</b> — 未融合每个算子都把中间张量写回再读出 HBM，访存往返主导耗时；融合在单个 kernel 内于寄存器完成相加与 RMSNorm，只读一次、写一次。</div>
</div>

<h2>二、CUDA Graph：捕获一次，反复重放</h2>
<p>回顾<span class="mono">第27课</span>：CUDA Graph 把一串<strong>固定的内核启动序列</strong>记录下来（capture），之后每一步解码只要<strong>重放（replay）</strong>这张图即可，几乎没有 CPU 端的提交开销。这对解码极其关键，因为解码每步真正算的东西很少，CPU 逐个提交内核的开销会被放大成主要瓶颈。可以这样打个比方：没有图的时候，CPU 像一个必须亲口逐道下令的指挥，每道命令都要先在脑子里整理、再开口传达；有了图，整套命令被提前写成一张乐谱，演奏时只要照谱直奏，指挥几乎不必再开口。解码动辄要重复成千上万步，省下的这点"开口"成本累加起来非常可观，对最终的吞吐曲线影响相当明显，这也是工程上必须把这条路径打磨到位的原因。</p>
<p>但 CUDA Graph 有两条硬约束：<strong>形状必须静态</strong>、<strong>不能有依赖数据的控制流</strong>。捕获时记录的是一组具体的指针与启动参数；如果下一次张量形状变了，或者程序要"看了数据才决定走哪条分支"，录下来的图就不再适用。因此：<strong>形状固定、行为确定的融合内核天然是"图友好（graph-friendly）"的</strong>；而那些<strong>形状会变、或在主机端有分支</strong>的算子，必须留在捕获区<strong>之外</strong>。</p>
<p>SGLang 的做法是引入<strong>分段 / 可打断的图（piecewise / breakable graph，<span class="mono">第27课</span>）</strong>：把整条前向拆成若干"静态子段"分别捕获成小图，中间那些动态的部分（动态形状、host 侧分支）则在小图之间<strong>以 eager 方式即时执行</strong>。这样既享受了大部分内核的零启动开销，又不被少数动态算子拖累。</p>
<p>这里要厘清一个常见误解：CUDA Graph 并不会让单个内核本身算得更快，它优化的是"<strong>怎么把内核交给 GPU</strong>"这件事。一张被捕获的图，记录的是一连串内核的启动顺序、各自的网格/线程块配置以及它们读写的具体显存地址。重放时，GPU 驱动不再需要 CPU 一条一条地解析与提交，而是把这整张"已经排好的指令表"一次性丢给硬件执行。正因为记录的是<strong>具体地址</strong>，所以输入张量必须落在与捕获时相同的缓冲区里、形状也必须一致——这也是为什么 SGLang 会预先分配固定大小的输入输出缓冲区，并按若干预设的 batch 尺寸分别捕获多张图，运行时再挑形状最接近的那张来重放。</p>
<p>那么哪些东西会"打断"图？典型是<strong>依赖数据的控制流</strong>：比如根据某个张量里实际算出来的值，去决定要不要走某条分支、要循环几次、要处理多长的序列。这类逻辑发生在 host（CPU）侧，且每次的走向都可能不同，无法被录进一张固定的图里。变长输入带来的<strong>动态形状</strong>同理——形状一变，捕获时记录的地址与启动参数就全部失配。分段图的智慧就在于：不强求把整条前向塞进一张图，而是承认"有些段天生是动态的"，把它们留给 eager，只把真正静态、可复用的子段固化成图。</p>

<table class="t"><tr><th>被融合的算子</th><th>为什么要融</th></tr>
<tr><td><span class="mono">SiluAndMul</span>（gate×up 激活）</td><td>免去 silu 临时张量的 HBM 写/读，两趟变一趟，一次启动</td></tr>
<tr><td><span class="mono">fused_add_rmsnorm</span>（残差相加 + RMSNorm，第36课）</td><td>相加结果留在寄存器直接归一化，省一次往返与一次启动</td></tr>
<tr><td>fused qk-norm + RoPE</td><td>归一化与旋转位置编码同内核完成，减少访存与提交</td></tr>
<tr><td>融合后形状固定的内核</td><td>静态形状 → 图友好 → 可被 CUDA Graph 捕获重放</td></tr></table>

<div class="fig">
  <svg viewBox="0 0 800 240" role="img" aria-label="融合到固定形状到捕获到重放的流水线">
    <text x="34" y="36" style="font-weight:700;fill:var(--muted)">融合 → 固定形状 → 捕获 → 重放</text>
    <rect x="34" y="60" width="160" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="114" y="88" text-anchor="middle" style="font-weight:700">融合算子</text>
    <text x="114" y="108" text-anchor="middle" style="fill:var(--muted);font-size:12px">减少 kernel 数</text>
    <text x="205" y="96" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="216" y="60" width="160" height="64" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="296" y="88" text-anchor="middle" style="font-weight:700">固定形状</text>
    <text x="296" y="108" text-anchor="middle" style="fill:var(--muted);font-size:12px">按 batch 分桶</text>
    <text x="387" y="96" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="398" y="60" width="160" height="64" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="478" y="88" text-anchor="middle" style="font-weight:700">捕获</text>
    <text x="478" y="108" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">CUDA Graph 录一次</text>
    <text x="569" y="96" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="580" y="60" width="160" height="64" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="660" y="88" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">重放</text>
    <text x="660" y="108" text-anchor="middle" style="fill:var(--muted);font-size:12px">每步解码廉价</text>
    <text x="34" y="170" style="fill:var(--accent-ink);font-size:12px">kernel 更少 → 捕获更易 → CPU 提交开销 ≈ 0</text>
  </svg>
  <div class="figcap"><b>图 2 · 融合喂给 CUDA Graph</b> — 先融合压低 kernel 数，再在固定/分桶形状上整步捕获成一张图，之后每个解码步只需廉价重放，把 CPU 启动开销几乎归零。</div>
</div>

<h2>三、二者如何配合：少 + 融 + 静态 → 捕获 → 重放</h2>
<p>把两件事串起来看，逻辑非常顺：先用<strong>融合</strong>把内核数量压下来、把中间访存消灭掉，让前向里大部分内核既<strong>少</strong>又<strong>形状固定</strong>；这样的内核序列恰好满足 CUDA Graph 的静态前提，于是可以<strong>捕获一次</strong>，之后每步解码<strong>反复重放</strong>，把"CPU 逐个提交内核"的开销几乎归零。对于带宽受限、启动开销敏感的解码阶段（<span class="mono">第4课</span>），这正是吞吐能上去的关键一环。</p>
<p>对于那些实在无法静态化的算子（典型如某些动态形状或数据依赖分支），就交给<strong>分段图</strong>：把它们隔在捕获区之外，用 eager 跑，其余静态子段照常捕获重放。需要强调的是，这些内核本体并不是凭空出现的——它们来自<span class="mono">第38课</span>的 AOT 预编译算子和<span class="mono">第39课</span>的 JIT 即时编译；融合与图只是在更高层把这些内核<strong>编排</strong>得更省、更快。</p>
<p>可以用一个量化的直觉来感受这套配合的威力。假设解码每步要发起一千个内核，每次启动的 CPU 提交开销约两微秒，那么光是提交就要两毫秒；而如果这一千个内核里有九成能被融合压缩并捕获进一张图，重放时这部分几乎不再有逐内核的提交成本，省下的时间就直接变成了更高的 token 生成速率。换句话说，融合负责"把内核变少、变规整"，CUDA Graph 负责"把这批规整的内核以极低成本喷给硬件"，二者一前一后，缺一不可：只融合不入图，仍要逐个提交；只入图不融合，捕获的内核又多又杂，收益有限。</p>
<p>还要记住一个层次关系：融合与图都属于"<strong>编排层</strong>"的优化，它们不生产内核，只决定内核怎么被组织和调用。真正的内核实现住在更底层——要么是<span class="mono">第38课</span>里随包预编译好的 AOT 算子，要么是<span class="mono">第39课</span>里按需即时编译的 JIT 算子。理解这条分工，能帮你在排查性能问题时迅速定位：是某个内核本身慢（该去看 AOT/JIT 实现与 kernel 调优），还是内核没被融合、没被入图导致启动开销爆炸（该去看融合路径与图捕获范围）。</p>

<div class="cols"><div class="col"><strong>融合友好（静态形状，可入图）</strong><ul><li><span class="mono">SiluAndMul</span> 等逐元素融合：形状由隐藏维决定，固定</li><li><span class="mono">fused_add_rmsnorm</span>：残差 + 归一，形状确定</li><li>fused qk-norm + RoPE：固定头维与序列布局</li><li>这些可被一并捕获进同一张图，重放零提交开销</li></ul></div><div class="col"><strong>打断图（依赖数据 / 动态形状）</strong><ul><li>变长输入、动态 batch/seq 形状的算子</li><li>"看数据再决定分支"的 host 侧控制流</li><li>必须留在捕获区之外，或用分段图夹在小图之间</li><li>以 eager 即时执行，避免破坏已捕获的静态子段</li></ul></div></div>

<h2>四、把它们落到一段真实代码上</h2>
<p>下面这段是 SGLang 中 <span class="mono">SiluAndMul</span> 的浓缩版。请重点对比两个前向：<span class="mono">forward_native</span> 是"<strong>两趟 + 临时张量</strong>"的参考实现，<span class="mono">forward_cuda</span> 是"<strong>一个 kernel 同时做 silu 与乘</strong>"的融合实现。后者写出的 <span class="mono">out</span> 只在最后落一次显存，silu 的中间结果全程留在寄存器，因此访存更少、启动更少，且形状固定——正好是 CUDA Graph 喜欢的样子。</p>
<p>注意 <span class="mono">forward_native</span> 里那行 <span class="mono">F.silu(x[..., :d]) * x[..., d:]</span>：它把张量从隐藏维的中点切成前后两半，前半是门控（gate）、后半是上投影（up）。表达式先算 silu，PyTorch 会为这个中间结果分配一块显存并写入，随后乘法再把它读回来——这就是那一次多余的临时张量往返。而 <span class="mono">forward_cuda</span> 先用 <span class="mono">torch.empty</span> 备好输出缓冲，再调用 <span class="mono">silu_and_mul(x, out)</span> 这个融合算子：在 CUDA 上它由 <span class="mono">sglang.jit_kernel.activation</span> 提供的 JIT 融合内核完成（ROCm/XPU 等路径则改用 AOT 的 <span class="mono">torch.ops.sgl_kernel.silu_and_mul</span>），无论走哪条路，都在 GPU 上以单趟方式同时完成切分、silu 与逐元素乘。输出缓冲尺寸只依赖于隐藏维的一半 <span class="mono">d</span>，是个静态值，因此这个内核在解码时形状固定、可以被安心地捕获进图反复重放。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>融合 Fuse</h4><p class="mono">SiluAndMul</p><p>把多个小算子合成一个 kernel，消除临时张量与多次启动。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>定形 Fix shape</h4><p class="mono">static shape</p><p>融合后的内核形状静态、行为确定，满足图的前提。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>捕获 Capture</h4><p class="mono">CUDA Graph</p><p>把这串静态内核录成一张图（第27课）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>重放 Replay</h4><p class="mono">replay</p><p>每步解码直接重放，CPU 提交开销近乎为零。</p></div></div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/activation.py ::SiluAndMul</span><span class="ln">未融合：两趟 + 临时张量；融合：一个 kernel 同时做 silu 与乘</span></div><pre>class SiluAndMul(MultiPlatformOp):
    # gate-and-up activation: split x in half, then silu(gate) * up

    def forward_native(self, x):        # reference: TWO ops + a temporary in HBM
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x):          # FUSED: one kernel does silu AND mul
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)            # -&gt; fused JIT silu+mul kernel (jit_kernel.activation)
        return out</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/elementwise.py ::fused_add_rmsnorm</span><span class="ln">一个 kernel 同时做 残差相加 + RMSNorm（省 HBM 往返）</span></div><pre>def fused_add_rmsnorm(input, residual, weight, eps=1e-6):
    # ONE kernel does residual-add + RMSNorm in place — no temp tensor,
    # no extra HBM round-trip:
    #   Step 1:  residual += input
    #   Step 2:  input = (residual / RMS(residual)) * weight
    ...   # dispatches to the fused FlashInfer/CUDA kernel</pre></div>

<p>具体到一个 Transformer block：每层有<strong>两处 add+norm 接缝</strong>（注意力前、MLP 前）。把每处的“残差相加 + RMSNorm”融成一个 <span class="mono">fused_add_rmsnorm</span> 内核，就能各省去一次对隐藏状态的完整读 + 写；内核更少也意味着要捕获进 CUDA Graph 的启动更少——融合与图的收益在这里叠加。</p>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>融合</strong>把多个小算子合成一个内核，消除中间临时张量的 <span class="mono">HBM</span> 往返，并把多次<strong>启动开销</strong>压成一次。</li>
<li><span class="mono">SiluAndMul</span> 的 <span class="mono">forward_native</span> 是"先 silu 再乘"的两趟参考实现；<span class="mono">forward_cuda</span> 在一个 kernel 里同时做，数据留在寄存器。</li>
<li>SGLang 常见融合：<span class="mono">SiluAndMul</span>、<span class="mono">fused_add_rmsnorm</span>（第36课）、fused qk-norm+RoPE。</li>
<li><strong>CUDA Graph</strong>（第27课）捕获固定内核序列再重放，几乎零提交开销，但要求<strong>静态形状、无数据依赖分支</strong>。</li>
<li>融合后形状固定的内核<strong>图友好</strong>；动态/数据依赖的算子须留在捕获区外，或用<strong>分段/可打断图</strong>夹在小图之间以 eager 执行。</li>
<li><strong>少 + 融 + 静态 → 捕获一次 → 反复重放</strong>，是带宽受限解码阶段（第4课）吞吐的关键；内核本体来自第38课（AOT）/ 第39课（JIT）。</li>
</ul></div>
""", "en": r"""
<p class="lead">In this lesson we drop all the way down from the "algorithm layer" to the "kernel layer" and study two hardware-level speedups — <strong>operator fusion</strong> and <strong>CUDA graphs (capture/replay)</strong> — and, crucially, how they <strong>cooperate</strong> to push decode throughput to the limit. Fusion attacks "too many memory trips and too many launches per computation"; CUDA graphs attack "the CPU keeps submitting kernels, and submission itself becomes the bottleneck." Stacked together, they give SGLang's decode path its near-zero-overhead kernel pipeline.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div><p>Think of one kernel launch as "a round trip to the warehouse." <strong>Unfused</strong> is like: go fetch raw material (read HBM), process it into a half-product and ship it back (write HBM), then the next step fetches that half-product again (read HBM) to keep working. The hauling alone burns huge time, and the GPU's "warehouse" (memory) bandwidth is finite. <strong>Fusion</strong> is like merging several adjacent steps into one workshop done in a single pass, keeping the half-product right on the bench (registers) instead of round-tripping to the warehouse.</p><p>And a <span class="mono">CUDA graph</span> is like recording today's whole assembly-line choreography ahead of time: each run, the foreman (CPU) need not bark every command — just <strong>play the recording (replay)</strong> and the machines follow. The catch: the moves must be <strong>identical</strong> every time (fixed shapes, no branch that is decided by looking at data), or the recording no longer matches.</p></div>

<div class="card macro"><div class="tag">🌍 The big picture</div><p>Recall <span class="mono">Lesson 4</span>: decoding is <strong>bandwidth-bound</strong>; per-token compute utilization is low and the bottleneck is moving data and issuing instructions. So anything that <strong>cuts HBM round-trips</strong> and <strong>cuts kernel launch count</strong> converts directly into throughput. Fusion reduces both; CUDA graphs (<span class="mono">Lesson 27</span>) nearly eliminate the "CPU submission" part of the latter. The fused and captured kernels ultimately come from <span class="mono">Lesson 38 (AOT)</span> and <span class="mono">Lesson 39 (JIT)</span> — fusion and graphs merely <strong>orchestrate</strong> them better.</p></div>

<h2>1. Operator fusion: merge many small kernels into one</h2>
<p>Fusion has two core motivations. First, <strong>memory traffic (HBM round-trips)</strong>: unfused, every op reads its input from memory and writes its output back, and the next op reads it again. These intermediates (temporaries) add nothing to the final answer yet waste precious bandwidth. Second, <strong>launch overhead</strong>: every kernel launch costs a CPU submission and a GPU dispatch; in decode, where each step computes very little, launch overhead can exceed the real compute.</p>
<p>Take SGLang's archetypal <span class="mono">SiluAndMul</span> (gate-and-up activation, gate×up). The unfused reference <span class="mono">forward_native</span> does it in <strong>two steps — silu THEN multiply</strong>: run <span class="mono">F.silu(gate)</span> on the first half into a temporary written to HBM, then read it back and multiply elementwise with the second half <span class="mono">up</span>. That is <strong>two kernel passes plus one extra temporary written and read</strong>. The fused <span class="mono">forward_cuda</span> does silu AND multiply in <strong>one kernel</strong>: once data is in registers, the silu result stays in <strong>registers</strong> for the multiply, never spilling to memory, writing only the final output once.</p>
<p>SGLang has many similar fusions: <span class="mono">fused_add_rmsnorm</span> (residual-add fused with RMSNorm into one kernel, see <span class="mono">Lesson 36</span>), and fusing q/k normalization with RoPE (fused qk-norm+rope). Their common trait: <strong>kill the "write-out-then-read-back" intermediate inside registers/shared memory</strong>, and collapse several launches into one.</p>

<div class="flow"><div class="node">read gate/up (HBM)</div><div class="arrow">→</div><div class="node">silu kernel</div><div class="arrow">→</div><div class="node">write temp (HBM)</div><div class="arrow">→</div><div class="node">read temp + up (HBM)</div><div class="arrow">→</div><div class="node">mul kernel</div><div class="arrow">→</div><div class="node">write output (HBM)</div></div>
<div class="flow"><div class="node">read x (HBM)</div><div class="arrow">→</div><div class="node">fused kernel: silu &amp; mul in registers</div><div class="arrow">→</div><div class="node">write output (HBM)</div></div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="unfused needs many HBM round-trips; fused does it in one kernel in registers">
    <text x="40" y="34" style="font-weight:700;fill:var(--red)">unfused: op → HBM → op → HBM</text>
    <rect x="40" y="52" width="110" height="46" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="95" y="80" text-anchor="middle">add</text>
    <text x="161" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="172" y="52" width="110" height="46" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="227" y="80" text-anchor="middle" class="mono" style="font-size:12px">write HBM</text>
    <text x="293" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="304" y="52" width="110" height="46" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="359" y="80" text-anchor="middle" class="mono" style="font-size:12px">read HBM</text>
    <text x="425" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="436" y="52" width="110" height="46" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="491" y="80" text-anchor="middle">rmsnorm</text>
    <text x="557" y="80" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="568" y="52" width="110" height="46" rx="6" style="fill:var(--red-soft);stroke:var(--red);stroke-width:1.5"/>
    <text x="623" y="80" text-anchor="middle" class="mono" style="font-size:12px">write HBM</text>
    <text x="40" y="128" style="fill:var(--red);font-size:12px">intermediates shuttle in/out of HBM — memory trips dominate</text>
    <text x="40" y="176" style="font-weight:700;fill:var(--teal)">fused: single kernel (read once · write once)</text>
    <rect x="40" y="194" width="150" height="56" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="115" y="227" text-anchor="middle">read once</text>
    <text x="202" y="227" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="215" y="194" width="290" height="56" rx="6" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="360" y="220" text-anchor="middle">one kernel: add + RMSNorm</text>
    <text x="360" y="240" text-anchor="middle" style="fill:var(--muted);font-size:12px">kept in registers</text>
    <text x="517" y="227" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="530" y="194" width="150" height="56" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="605" y="227" text-anchor="middle">write once</text>
    <text x="40" y="282" style="fill:var(--teal);font-size:12px">saves 2 writes + 1 read of HBM</text>
  </svg>
  <div class="figcap"><b>Fig 1 · unfused vs fused</b> — unfused writes each intermediate back and reads it from HBM, so memory round-trips dominate; fused does add and RMSNorm in registers inside one kernel, reading once and writing once.</div>
</div>

<h2>2. CUDA graphs: capture once, replay many times</h2>
<p>Recall <span class="mono">Lesson 27</span>: a CUDA graph records a <strong>fixed sequence of kernel launches</strong> (capture), and then each decode step just <strong>replays</strong> that graph with almost no CPU-side submission overhead. This matters enormously for decode, because each step computes so little that per-kernel CPU submission balloons into the dominant cost.</p>
<p>But CUDA graphs impose two hard constraints: <strong>shapes must be static</strong> and <strong>no data-dependent control branching</strong>. Capture records concrete pointers and launch parameters; if next time a tensor's shape changes, or the program "looks at data to decide which branch to take," the recorded graph no longer applies. Hence: <strong>fixed-shape, deterministic fused kernels are inherently graph-friendly</strong>; ops whose <strong>shapes vary, or that branch on the host side</strong>, must stay <strong>outside</strong> the captured region.</p>
<p>SGLang's answer is <strong>piecewise / breakable graphs (<span class="mono">Lesson 27</span>)</strong>: split the forward into several "static sub-spans," capture each as a small graph, and run the dynamic bits in between (dynamic shapes, host-side branches) <strong>eagerly</strong>. This way you get near-zero launch overhead on most kernels without being held back by the few dynamic ops.</p>

<table class="t"><tr><th>Fused op</th><th>Why fuse it</th></tr>
<tr><td><span class="mono">SiluAndMul</span> (gate×up activation)</td><td>Drops the silu temporary's HBM write/read; two passes become one, one launch</td></tr>
<tr><td><span class="mono">fused_add_rmsnorm</span> (residual-add + RMSNorm, Lesson 36)</td><td>Sum stays in registers and is normalized in place; saves a round-trip and a launch</td></tr>
<tr><td>fused qk-norm + RoPE</td><td>Normalization and rotary embedding in one kernel; less traffic, fewer submissions</td></tr>
<tr><td>Fixed-shape kernels after fusion</td><td>Static shape → graph-friendly → capturable/replayable by a CUDA graph</td></tr></table>

<div class="fig">
  <svg viewBox="0 0 800 240" role="img" aria-label="pipeline: fuse to static shape to capture to replay">
    <text x="34" y="36" style="font-weight:700;fill:var(--muted)">fuse → static shape → capture → replay</text>
    <rect x="34" y="60" width="160" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="114" y="88" text-anchor="middle" style="font-weight:700">fuse ops</text>
    <text x="114" y="108" text-anchor="middle" style="fill:var(--muted);font-size:12px">cut kernel count</text>
    <text x="205" y="96" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="216" y="60" width="160" height="64" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="296" y="88" text-anchor="middle" style="font-weight:700">static shape</text>
    <text x="296" y="108" text-anchor="middle" style="fill:var(--muted);font-size:12px">bucket by batch</text>
    <text x="387" y="96" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="398" y="60" width="160" height="64" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="478" y="88" text-anchor="middle" style="font-weight:700">capture</text>
    <text x="478" y="108" text-anchor="middle" class="mono" style="fill:var(--muted);font-size:12px">CUDA Graph once</text>
    <text x="569" y="96" text-anchor="middle" style="fill:var(--muted)">→</text>
    <rect x="580" y="60" width="160" height="64" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="660" y="88" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">replay</text>
    <text x="660" y="108" text-anchor="middle" style="fill:var(--muted);font-size:12px">cheap each step</text>
    <text x="34" y="170" style="fill:var(--accent-ink);font-size:12px">fewer kernels → easier capture → CPU submit cost ≈ 0</text>
  </svg>
  <div class="figcap"><b>Fig 2 · fusion feeds the CUDA graph</b> — fuse first to cut kernel count, capture the whole step once at fixed/bucketed shapes, then replay cheaply on every decode step, driving CPU launch overhead to nearly zero.</div>
</div>

<h2>3. How they cooperate: fewer + fused + static → capture → replay</h2>
<p>Chaining the two ideas, the logic is clean: first use <strong>fusion</strong> to shrink the kernel count and erase intermediate traffic, so most kernels in the forward are both <strong>few</strong> and <strong>fixed-shape</strong>; such a kernel sequence exactly satisfies the CUDA graph's static premise, so you can <strong>capture once</strong> and then <strong>replay</strong> it every decode step, driving the "CPU submits each kernel" cost to nearly zero. For the bandwidth-bound, launch-sensitive decode stage (<span class="mono">Lesson 4</span>), this is a key reason throughput can climb.</p>
<p>For ops that truly cannot be made static (e.g. some dynamic shapes or data-dependent branches), hand them to <strong>piecewise graphs</strong>: keep them outside the captured region, run them eagerly, and capture/replay the remaining static sub-spans as usual. Note these kernels are not conjured from nowhere — they come from <span class="mono">Lesson 38</span>'s AOT-precompiled kernels and <span class="mono">Lesson 39</span>'s JIT kernels; fusion and graphs merely <strong>orchestrate</strong> them more cheaply and quickly at a higher level.</p>

<div class="cols"><div class="col"><strong>Fusion-friendly (static shape, capturable)</strong><ul><li><span class="mono">SiluAndMul</span> and other elementwise fusions: shape set by hidden dim, fixed</li><li><span class="mono">fused_add_rmsnorm</span>: residual + norm, deterministic shape</li><li>fused qk-norm + RoPE: fixed head dim and sequence layout</li><li>These can be captured together into one graph, replayed with zero submission cost</li></ul></div><div class="col"><strong>Graph-breaking (data-dependent / dynamic shape)</strong><ul><li>Ops with variable-length input or dynamic batch/seq shapes</li><li>Host-side control branching that "branches after looking at data"</li><li>Must stay outside the captured region, or sit between small graphs in a piecewise graph</li><li>Run eagerly so the captured static sub-spans are not broken</li></ul></div></div>

<h2>4. Grounding it in real code</h2>
<p>Below is a condensed version of SGLang's <span class="mono">SiluAndMul</span>. Focus on the two forwards: <span class="mono">forward_native</span> is the "<strong>two passes + a temporary</strong>" reference, while <span class="mono">forward_cuda</span> is the "<strong>one kernel does silu and multiply</strong>" fused version. The latter writes <span class="mono">out</span> to memory only once at the end, keeping silu's intermediate in registers throughout — less traffic, fewer launches, and a fixed shape, exactly what a CUDA graph likes.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Fuse</h4><p class="mono">SiluAndMul</p><p>Merge several small ops into one kernel; kill temporaries and extra launches.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Fix shape</h4><p class="mono">static shape</p><p>The fused kernel is static-shaped and deterministic, meeting the graph premise.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>Capture</h4><p class="mono">CUDA Graph</p><p>Record this static kernel run into a graph (Lesson 27).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Replay</h4><p class="mono">replay</p><p>Each decode step replays it; CPU submission cost near zero.</p></div></div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/layers/activation.py ::SiluAndMul</span><span class="ln">unfused: two passes + a temporary; fused: one kernel does silu and multiply</span></div><pre>class SiluAndMul(MultiPlatformOp):
    # gate-and-up activation: split x in half, then silu(gate) * up

    def forward_native(self, x):        # reference: TWO ops + a temporary in HBM
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x):          # FUSED: one kernel does silu AND mul
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)            # -&gt; fused JIT silu+mul kernel (jit_kernel.activation)
        return out</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">sgl-kernel/python/sgl_kernel/elementwise.py ::fused_add_rmsnorm</span><span class="ln">one kernel does residual-add + RMSNorm (saves HBM round-trips)</span></div><pre>def fused_add_rmsnorm(input, residual, weight, eps=1e-6):
    # ONE kernel does residual-add + RMSNorm in place — no temp tensor,
    # no extra HBM round-trip:
    #   Step 1:  residual += input
    #   Step 2:  input = (residual / RMS(residual)) * weight
    ...   # dispatches to the fused FlashInfer/CUDA kernel</pre></div>

<p>Concretely, every Transformer block has <strong>two add+norm seams</strong> (pre-attention and pre-MLP). Fusing each “residual-add + RMSNorm” into one <span class="mono">fused_add_rmsnorm</span> kernel saves a full read + write of the hidden state per seam; fewer kernels also means fewer launches to capture in the CUDA graph — the wins of fusion and graphs compound here.</p>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li><strong>Fusion</strong> merges several small ops into one kernel, eliminating intermediate temporaries' <span class="mono">HBM</span> round-trips and collapsing many <strong>launches</strong> into one.</li>
<li><span class="mono">SiluAndMul</span>'s <span class="mono">forward_native</span> is the "silu then multiply" two-pass reference; <span class="mono">forward_cuda</span> does both in one kernel with data kept in registers.</li>
<li>Common SGLang fusions: <span class="mono">SiluAndMul</span>, <span class="mono">fused_add_rmsnorm</span> (Lesson 36), fused qk-norm+RoPE.</li>
<li><strong>CUDA graphs</strong> (Lesson 27) capture a fixed kernel sequence and replay it with near-zero submission cost, but require <strong>static shapes and no data-dependent branches</strong>.</li>
<li>Fixed-shape fused kernels are <strong>graph-friendly</strong>; dynamic/data-dependent ops must stay outside the captured region, or sit between small graphs in a <strong>piecewise/breakable graph</strong>, run eagerly.</li>
<li><strong>Fewer + fused + static → capture once → replay</strong> is key to bandwidth-bound decode throughput (Lesson 4); the kernels themselves come from Lesson 38 (AOT) / Lesson 39 (JIT).</li>
</ul></div>
"""}
LESSON_42 = {"zh": r"""
<p class="lead">同一套 SGLang 引擎，既能跑在 NVIDIA 显卡上，也能跑在 AMD 显卡、华为昇腾 NPU、Intel XPU、摩尔线程 MUSA、苹果 MLX，甚至纯 CPU 上。这背后靠的是一层<strong>平台抽象（platform abstraction）</strong>：它让上层引擎对"脚下踩的是哪块芯片"几乎一无所知，从而做到"一份代码，多种硬件"。如果没有这层抽象，每支持一种新芯片，就得把调度、模型、各层代码全部改一遍，工程量会随硬件种类爆炸式增长；有了它，新硬件接入只需"补齐一小块"。本课是第九部分（内核与硬件）的收官课，我们把前面讲过的注意力内核、CUDA Graph、AOT/JIT 内核统统串起来，看看它们是如何被"按芯片替换"的，又是如何在统一的接口下协同工作的。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一台支持多国插座的<strong>旅行充电器</strong>：无论你飞到哪个国家，插头形状千差万别——英标、欧标、美标、国标——但你的手机充电线永远是同一根。中间那个<span class="mono">转换头</span>负责把"墙上的插座"翻译成"手机能用的电"。你出门只需多带几个转换头，而不必为每个国家买一台新手机。</p>
<p>SGLang 的平台抽象就是这个转换头。上层的调度器、模型、各种层就像你的手机充电线，永远不变；底层的芯片就像各国插座，五花八门。<span class="mono">SRTPlatform</span> 这层抽象负责把"具体芯片的能力"翻译成"上层统一的接口"。换芯片，只换转换头，不换手机。更妙的是，这个类比还能解释"能力标志"：有的国家电压是 220V、有的是 110V，聪明的转换头会先"问一句"当前插座支持什么，再决定怎么供电——这正对应上层先问"<span class="mono">supports_fp8</span> 吗""<span class="mono">support_cuda_graph</span> 吗"，再决定走哪条执行路径。</p></div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>从单张显卡到跨越多种硬件的大型集群，SGLang 想做到"一处引擎、处处可跑"。要实现这一点，关键是把"<strong>与硬件无关</strong>"的部分和"<strong>每芯片专属</strong>"的部分彻底分开。调度器（第18课）、模型（第26课）、绝大多数层都不知道、也不需要知道脚下是哪块芯片；真正需要"按芯片替换"的只有内核（第38课 AOT / 第39课 JIT）、注意力后端（第33课）和少数平台钩子。把"专属"的那一小撮东西圈得越小，能"通用"复用的代码就越多，移植新硬件的成本也就越低。这正是本指南反复强调的"<strong>一切皆可插拔</strong>"主题（设计主题前瞻见第62课）：注意力是插拔的、内核是插拔的、并行策略是插拔的，而本课告诉你，连"脚下的芯片"本身也是插拔的。</p></div>

<h2>一、平台抽象：一个基类，每种芯片一个子类</h2>
<p>平台抽象的核心是一个基类 <span class="mono">SRTPlatform</span>，定义在 <span class="mono">srt/platforms/interface.py</span> 里。它本身不直接干活，而是声明了一组"每种芯片都得回答"的问题——这些问题以三种形式出现：<strong>工厂方法</strong>、<strong>能力标志</strong>和<strong>生命周期钩子</strong>。你可以把这个基类理解成一张"考卷"：每块新芯片想接入 SGLang，就得把这张考卷答完；答卷的方式，就是子类化这个基类并覆盖其中的方法。基类里很多方法干脆直接抛出 <span class="mono">NotImplementedError</span>，等于在说"这道题没有标准答案，必须由具体芯片自己填"。</p>
<p><strong>工厂方法</strong>负责"造出"针对当前芯片的组件。比如 <span class="mono">get_default_attention_backend()</span> 返回这块芯片该用哪一族注意力内核（第33课讲过 FlashAttention、FlashInfer、Triton 等后端，不同芯片支持的后端并不相同）；<span class="mono">get_graph_runner_cls()</span> 返回 CUDA Graph 的运行器类（第27课讲过把整个前向过程录制成图来消除逐次启动的开销）。不同芯片在这两个方法里返回不同的实现，而上层只管调用、不管返回的究竟是哪一个具体类——这正是"工厂"二字的含义：上层拿到的是一个统一约定的产品，至于产品在哪条流水线上造出来的，它并不关心。</p>
<p><strong>能力标志</strong>是一组布尔判断，让上层可以问"这块芯片能不能做某件事"。例如 <span class="mono">supports_fp8</span>（能不能跑 FP8 量化）、<span class="mono">support_cuda_graph</span>（能不能用 CUDA Graph）、<span class="mono">support_piecewise_cuda_graph</span>（能不能用分段式 CUDA Graph）。上层据此自适应：如果某块芯片不支持 CUDA Graph，就优雅地退回到逐算子执行，而不是直接崩溃。能力标志的妙处在于，它把"芯片差异"收敛成了一句句简单的提问，上层不必为每块芯片散落一堆 <span class="mono">if 芯片型号 == …</span> 的硬编码分支，而是统一地问"你支持 X 吗"，让代码既干净又好扩展。</p>
<p><strong>生命周期钩子</strong>则是在引擎启动时让芯片"插一脚"调整默认配置。<span class="mono">apply_server_args_defaults()</span> 就是这样一个钩子：每种芯片可以在这里设置自己合适的默认参数，比如某些芯片默认关闭某个尚不稳定的特性，或者把某个并行度调到对自己最友好的值。还有一个 <span class="mono">supported_quantization</span> 列表，声明这块芯片支持哪些量化格式。这些钩子让"按芯片定制默认行为"有了一个规整的落脚点，而不是把特判逻辑塞得到处都是。把这三件套合在一起看，你会发现它们对应着一块芯片接入引擎时要回答的三类问题：<strong>"该用什么组件？"</strong>（工厂方法）、<strong>"你能做什么？"</strong>（能力标志）、<strong>"启动时要不要调点默认值？"</strong>（生命周期钩子）。任何一块新芯片，只要把这三类问题答清楚，就算正式"上户口"了。</p>

<h2>二、两棵树：platforms 与 hardware_backend</h2>
<p>代码里有两棵彼此呼应的目录树，理解它们的分工是读懂这套设计的关键。第一棵是 <span class="mono">srt/platforms/</span>，里面是各芯片对 <span class="mono">SRTPlatform</span> 的子类化：<span class="mono">CudaSRTPlatform</span>（NVIDIA，是树内默认实现），还有 ROCm（AMD）、CPU 等，分别落在 <span class="mono">srt/platforms/{cuda,rocm,cpu}.py</span>。每个子类覆盖基类的工厂方法和能力标志，回答"我这块芯片到底怎么做"。换句话说，<span class="mono">platforms/</span> 这棵树是"决策层"：它决定每块芯片该选哪一族注意力后端、用哪个图运行器、支持哪些量化、启动时改哪些默认值。</p>
<p>第二棵是 <span class="mono">srt/hardware_backend/</span>，按设备组织了一组后端目录：<span class="mono">cpu/ gpu/ npu/ xpu/ musa/ mlx/</span>。它们覆盖了 NVIDIA 与 AMD 的 GPU、华为昇腾（Ascend）NPU、Intel 的 XPU、摩尔线程的 MUSA、苹果的 MLX，以及纯 CPU。如果说 <span class="mono">platforms/</span> 是"决策层"，那么 <span class="mono">hardware_backend/</span> 就是"执行层"：真正与某块设备打交道的底层实现住在这里。需要特别说明的是，谷歌的 TPU 并不在这棵树里，而是通过一个独立的 sglang-jax 项目来支持——这也提醒我们，"多硬件"并不意味着所有硬件都必须挤进同一套代码，有时分立的项目反而更合适。两棵树一上一下、一决策一执行，共同把"换芯片"这件事关进了一个清晰的笼子里。</p>
<p>把这两棵树连起来看，一次请求在引擎里的"芯片之旅"就清楚了：引擎启动时，先根据当前环境选出对应的 <span class="mono">SRTPlatform</span> 子类（比如检测到 NVIDIA 就用 <span class="mono">CudaSRTPlatform</span>），由它的 <span class="mono">apply_server_args_defaults()</span> 钩子把默认参数调到对这块芯片最合适的状态；接着上层向它索要注意力后端和图运行器，工厂方法据此交回对应的实现；运行中，每当上层拿不准"这块芯片能不能用某个加速路径"，就查一查能力标志。整个过程里，调度器和模型自始至终只面对统一接口，从不直接 import 任何具体芯片的模块——这正是"上层与硬件无关"在代码层面的具体体现。</p>

<h2>三、关键洞察：上层与硬件无关</h2>
<p>本课最重要的一句话是：<strong>上层是硬件无关（hardware-agnostic）的</strong>。调度器（第18课）决定谁先跑、谁后跑、怎么拼批；模型（第26课）定义网络结构、各层怎么连；绝大多数层只关心张量进、张量出。它们统统不知道脚下是哪块芯片，也不需要知道。真正"按芯片替换"的东西被压缩到了很小的范围内：内核（第38课 AOT 预编译 / 第39课 JIT 即时编译）、注意力后端（第33课），以及前面讲的那些平台钩子。这个范围越小，移植的代价就越低，复用的价值就越高。</p>
<p>这种分层之所以重要，是因为它把"移植到新硬件"这件听起来很吓人的大事，变成了"只实现一个新的 <span class="mono">SRTPlatform</span> 子类，再配一套新内核"的小事。上层成千上万行的调度与模型代码可以原封不动地复用，连测试都不必重写。能力标志在这里起到了润滑剂的作用：上层不必为每块芯片写一长串 if-else 分支，而是统一地问"<span class="mono">support_cuda_graph</span> 吗？"，得到否定回答就走兜底路径。可以说，"上层与硬件无关"不是一句口号，而是被 <span class="mono">SRTPlatform</span> 这层抽象、连同两棵目录树一起，实打实地用工程手段兑现了的承诺。也正因如此，同一个团队既能在一张消费级显卡上做实验，又能把同一套引擎推上跨越多种芯片的大型生产集群，而中间几乎不用改上层逻辑。</p>
<p>反过来想，如果没有这层抽象，会发生什么？每加一种芯片，调度器里就要多一处分支，模型里又要多一处特判，时间一长，代码会被各种"<span class="mono">if 是某某芯片</span>"的补丁撑得臃肿不堪，谁也不敢动。平台抽象的价值，恰恰在于它把这些本会四处蔓延的差异，全部收拢到 <span class="mono">platforms/</span> 与 <span class="mono">hardware_backend/</span> 两棵树里：差异被"圈养"在固定的位置，上层因而保持干净。这就是工程上常说的"<strong>关注点分离</strong>"——让会变化的部分和保持稳定的部分各归各位，是大型系统能够长期演进、不断接纳新硬件的根本前提。</p>

<div class="layers">
<div class="layer">🧠 调度器（第18课）· 模型（第26课）· 各种层 —— 与硬件无关，不知道脚下是哪块芯片</div>
<div class="layer">🔌 SRTPlatform 抽象层 —— 工厂方法 + 能力标志 + 生命周期钩子（按芯片翻译）</div>
<div class="layer">⚙️ 每芯片内核：注意力后端（第33课）· AOT 内核（第38课）· JIT 内核（第39课）—— 按芯片替换</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="三条横带：上层硬件无关、中间 SRTPlatform 分界线、底层每芯片内核">
    <text x="24" y="28" style="font-weight:700;fill:var(--blue)">上层 · 硬件无关</text>
    <rect x="20" y="36" width="760" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="40" y="50" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="153" y="73" text-anchor="middle" style="font-size:13px">调度器</text>
    <rect x="287" y="50" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="400" y="73" text-anchor="middle" style="font-size:13px">模型</text>
    <rect x="534" y="50" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="647" y="73" text-anchor="middle" style="font-size:13px">注意力抽象</text>
    <text x="24" y="120" style="font-weight:700;fill:var(--amber)">分界线 · SRTPlatform</text>
    <rect x="20" y="128" width="760" height="64" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5;stroke-dasharray:6 4"/>
    <rect x="40" y="142" width="350" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="215" y="165" text-anchor="middle" class="mono" style="font-size:12px">能力标志 supports_fp8…</text>
    <rect x="410" y="142" width="330" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="575" y="165" text-anchor="middle" class="mono" style="font-size:12px">设备操作 device ops</text>
    <text x="24" y="212" style="font-weight:700;fill:var(--purple)">底层 · 每芯片内核</text>
    <rect x="20" y="220" width="760" height="64" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <rect x="40" y="234" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="153" y="257" text-anchor="middle" class="mono" style="font-size:13px">CUDA</text>
    <rect x="287" y="234" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="400" y="257" text-anchor="middle" class="mono" style="font-size:13px">HIP</text>
    <rect x="534" y="234" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="647" y="257" text-anchor="middle" class="mono" style="font-size:13px">CANN</text>
  </svg>
  <div class="figcap"><b>图 A · 三条横带</b> — 上层硬件无关（调度器 / 模型 / 注意力抽象）；中间是 SRTPlatform 分界线（能力标志 + 设备操作），硬件细节只住在这里；底层是每芯片内核（CUDA / HIP / CANN…）。</div>
</div>

<table class="t">
<tr><th>硬件</th><th>对应的后端 / 内核</th></tr>
<tr><td>NVIDIA GPU</td><td>CudaSRTPlatform · hardware_backend/gpu · CUDA 注意力内核</td></tr>
<tr><td>AMD GPU</td><td>ROCm 平台 · hardware_backend/gpu · ROCm/HIP 内核</td></tr>
<tr><td>华为昇腾 NPU</td><td>hardware_backend/npu · Ascend 内核</td></tr>
<tr><td>Intel XPU</td><td>hardware_backend/xpu · XPU 内核</td></tr>
<tr><td>摩尔线程 MUSA</td><td>hardware_backend/musa · MUSA 内核</td></tr>
<tr><td>苹果 MLX</td><td>hardware_backend/mlx · MLX 内核</td></tr>
<tr><td>纯 CPU</td><td>CPU 平台 · hardware_backend/cpu · CPU 内核</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 330" role="img" aria-label="一个 SGLang 引擎扇出到多种芯片，每种都是一个 SRTPlatform 子类，用能力标志回答上层">
    <rect x="300" y="20" width="200" height="48" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="400" y="49" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">SGLang 引擎</text>
    <line x1="400" y1="68" x2="400" y2="98" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="150" y1="98" x2="670" y2="98" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="150" y1="98" x2="150" y2="118" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="410" y1="98" x2="410" y2="118" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="670" y1="98" x2="670" y2="118" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="150" y1="196" x2="150" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="410" y1="196" x2="410" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="670" y1="196" x2="670" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="40" y="118" width="220" height="78" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="150" y="144" text-anchor="middle" style="font-weight:700">NVIDIA · CUDA</text>
    <text x="150" y="164" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform 子类</text>
    <text x="150" y="184" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✓ · graph ✓</text>
    <rect x="300" y="118" width="220" height="78" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="410" y="144" text-anchor="middle" style="font-weight:700">AMD · ROCm</text>
    <text x="410" y="164" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform 子类</text>
    <text x="410" y="184" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✓ · graph ✓</text>
    <rect x="560" y="118" width="220" height="78" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="670" y="144" text-anchor="middle" style="font-weight:700">Ascend · NPU</text>
    <text x="670" y="164" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform 子类</text>
    <text x="670" y="184" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✓ · graph ✗</text>
    <rect x="40" y="232" width="220" height="78" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="150" y="258" text-anchor="middle" style="font-weight:700">Intel · XPU</text>
    <text x="150" y="278" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform 子类</text>
    <text x="150" y="298" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✗ · graph ✗</text>
    <rect x="300" y="232" width="220" height="78" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="410" y="258" text-anchor="middle" style="font-weight:700">Moore · MUSA</text>
    <text x="410" y="278" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform 子类</text>
    <text x="410" y="298" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✗ · graph ✓</text>
    <rect x="560" y="232" width="220" height="78" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="670" y="258" text-anchor="middle" style="font-weight:700">Apple · MLX</text>
    <text x="670" y="278" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform 子类</text>
    <text x="670" y="298" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✗ · graph ✗</text>
  </svg>
  <div class="figcap"><b>图 B · 一个引擎 → 多种芯片</b> — 同一个 SGLang 引擎扇出到 NVIDIA(CUDA)、AMD(ROCm)、昇腾 NPU、Intel XPU、MUSA、MLX；每种都是一个 SRTPlatform 子类，用 supports_fp8 / support_cuda_graph 等能力标志回答上层（有的支持、有的不支持）。</div>
</div>

<div class="cols">
<div class="col"><strong>✅ 可移植（一份代码到处跑）</strong><br/>调度器（第18课）、模型（第26课）、绝大多数层、输入输出（IO）流程。它们与硬件无关，换芯片时原样复用。</div>
<div class="col"><strong>🔧 每芯片专属（必须替换）</strong><br/>各类内核（第38课 / 第39课）、注意力后端（第33课）、平台钩子（工厂方法、能力标志、<span class="mono">apply_server_args_defaults</span>）。移植新硬件主要就是补齐这部分。</div>
</div>

<div class="flow">
<div class="node">上层算子请求（如一次注意力计算）</div>
<div class="arrow">→</div>
<div class="node">SRTPlatform 分派 + 能力检查（support_cuda_graph？supports_fp8？）</div>
<div class="arrow">→</div>
<div class="node">对应设备的内核执行（CUDA / ROCm / NPU / …）</div>
</div>

<h2>四、看一眼基类长什么样</h2>
<p>下面是 <span class="mono">SRTPlatform</span> 的一个忠实的精简版。注意基类里很多方法直接 <span class="mono">raise NotImplementedError</span>——这是在说"这件事每块芯片必须自己回答"；而 <span class="mono">CudaSRTPlatform</span> 作为树内默认实现，给出了 NVIDIA 的答案：支持 FP8、支持 CUDA Graph、也支持分段式 CUDA Graph。树外的新芯片只要照葫芦画瓢，子类化并覆盖这些方法即可，无需改动上层一行代码。你会发现，这段代码里没有任何"业务逻辑"，全是"约定"：它约定了每块芯片要回答哪些问题、以什么签名回答。正是这种"只定约定、不定实现"的克制，让一套引擎得以跨越如此多样的硬件而不至于失控。值得一提的是 <span class="mono">DeviceMixin</span> 与 <span class="mono">CudaDeviceMixin</span> 这类混入（mixin）：它们把"和设备打交道的通用能力"以可组合的方式拼进平台类，让"NVIDIA 平台"既能复用通用设备逻辑、又能叠加自己的特性——这本身又是一处"可插拔"的小设计。当你日后真要为一块新芯片提交 PR 时，起点几乎总是：新建一个文件，子类化 <span class="mono">SRTPlatform</span>，逐个把 <span class="mono">NotImplementedError</span> 替换成你这块芯片的真实答案。</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/platforms/interface.py ::SRTPlatform</span><span class="ln">每种芯片子类化：工厂方法 + 能力标志 + 生命周期钩子</span></div><pre>class SRTPlatform(DeviceMixin):
    # base class for a hardware platform; out-of-tree chips subclass + override
    supported_quantization = []

    def apply_server_args_defaults(self, server_args):   # per-chip default flags
        pass
    def get_default_attention_backend(self) -&gt; str:     # which attention kernel family (Lesson 33)
        raise NotImplementedError
    def get_graph_runner_cls(self) -&gt; type:             # the CUDA-graph runner (Lesson 27)
        raise NotImplementedError

class CudaSRTPlatform(CudaDeviceMixin, SRTPlatform):     # default in-tree NVIDIA platform
    def supports_fp8(self): return True
    def support_cuda_graph(self): return True
    def support_piecewise_cuda_graph(self): return True</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/platforms/cuda.py ::CudaSRTPlatform</span><span class="ln">具体平台：NVIDIA/CUDA 用能力标志回答上层</span></div><pre>class CudaSRTPlatform(CudaDeviceMixin, SRTPlatform):
    # the concrete NVIDIA/CUDA platform. Upper layers ask capability
    # questions; this subclass answers them, so engine code stays
    # hardware-agnostic.
    def supports_fp8(self) -&gt; bool:
        return True
    def support_cuda_graph(self) -&gt; bool:
        return True
    def support_piecewise_cuda_graph(self) -&gt; bool:
        return True</pre></div>

<div class="card"><div class="tag">🧩 具体例子</div><ul>
<li><strong>移植到新加速器</strong>：只需新增一个 <span class="mono">SRTPlatform</span> 子类并配上它的内核，<strong>完全不动</strong>调度器（第18课）与模型（第26课）。</li>
<li><strong>某块芯片不支持 CUDA Graph</strong>：它的 <span class="mono">support_cuda_graph()</span> 直接返回 <span class="mono">False</span>，引擎就跳过图捕获、退回逐算子执行，照样跑通。</li>
</ul></div>

<div class="card key"><div class="tag">📌 本课要点</div><ul>
<li><strong>平台抽象</strong>让一套引擎跑遍多种芯片：基类 <span class="mono">SRTPlatform</span>（在 <span class="mono">srt/platforms/interface.py</span>）声明每芯片的工厂方法、能力标志和生命周期钩子。</li>
<li><strong>工厂方法</strong>如 <span class="mono">get_default_attention_backend()</span>（第33课注意力内核）、<span class="mono">get_graph_runner_cls()</span>（第27课 CUDA Graph 运行器）；<strong>钩子</strong>如 <span class="mono">apply_server_args_defaults()</span>，还有 <span class="mono">supported_quantization</span> 列表声明支持的量化格式。</li>
<li>每种芯片<strong>子类化</strong>基类：<span class="mono">CudaSRTPlatform</span>（NVIDIA）、ROCm（AMD）等；另一棵 <span class="mono">srt/hardware_backend/</span> 树按 <span class="mono">cpu/gpu/npu/xpu/musa/mlx</span> 组织各设备后端，两棵树一"决策"一"执行"。</li>
<li><strong>上层与硬件无关</strong>：调度器（第18课）、模型（第26课）、多数层都不知道芯片；只有内核（第38/39课）、注意力后端（第33课）和平台钩子被按芯片替换，移植新硬件主要就是补齐这一小撮。</li>
<li><strong>能力标志</strong>（<span class="mono">supports_fp8</span>、<span class="mono">support_cuda_graph</span>、<span class="mono">support_piecewise_cuda_graph</span>）让上层询问"这块芯片能做 X 吗"并自适应（如不支持 CUDA Graph 就兜底退回逐算子执行），而不必散落一堆按型号特判的分支。</li>
<li>谷歌 TPU 不在这棵树里，而是经由独立的 sglang-jax 项目支持——多硬件不等于所有硬件都挤进同一套代码。</li>
<li>这是"从单卡到大型集群、跨越多种硬件"的工程基础，也是本指南"<strong>一切皆可插拔</strong>"主题的体现（设计主题前瞻第62课）。</li>
</ul></div>

<div class="card"><div class="tag">🏁 第九部分收官</div><p>到这里，第九部分"内核与硬件"就讲完了。我们从最贴近金属的 AOT 与 JIT 内核出发（第38、39课），看过量化与通信内核，又把 CUDA Graph、注意力后端这些性能利器串联起来，最后用本课的平台抽象把它们收拢成一句话：<strong>把"每芯片专属"的那一小撮东西隔离好，上层就能与硬件无关地复用</strong>。这正是 SGLang 能从单张显卡一路扩展到跨越多种硬件的大型集群的根本原因。带着这套"分层与插拔"的视角，你会在后面的部分里一次次重逢同样的设计哲学。</p></div>
""", "en": r"""
<p class="lead">The same SGLang engine can run on NVIDIA GPUs, but also on AMD GPUs, Huawei Ascend NPUs, Intel XPUs, Moore Threads MUSA, Apple MLX, and even plain CPUs. Behind this lies a layer of <strong>platform abstraction</strong>: it lets the upper engine stay almost entirely ignorant of "which chip is under its feet," achieving "one codebase, many kinds of hardware." This is the closing lesson of Part 9 (Kernels &amp; Hardware); we tie together the attention kernels, CUDA Graphs, and AOT/JIT kernels from earlier lessons and see how they get "swapped per chip."</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a <strong>travel charger</strong> that supports sockets from many countries. No matter which country you fly to, the plug shapes differ wildly — UK, EU, US, China — but your phone cable is always the same one. The <span class="mono">adapter head</span> in the middle translates "the socket on the wall" into "power your phone can use."</p>
<p>SGLang's platform abstraction is exactly that adapter head. The upper scheduler, model, and various modules are like your phone cable, never changing; the chips below are like the assorted sockets. The <span class="mono">SRTPlatform</span> abstraction translates "this specific chip's capabilities" into "a uniform interface for the upper tiers." Swap the chip, swap only the adapter head — not the phone.</p></div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>From a single GPU to large clusters spanning many kinds of hardware, SGLang aims for "one engine, runs everywhere." The key to this is cleanly separating the part that is "<strong>hardware-agnostic</strong>" from the part that is "<strong>per-chip</strong>." The scheduler (Lesson 18), the model (Lesson 26), and most modules don't know — and don't need to know — which chip is underneath; the only things that truly need "swapping per chip" are the kernels (Lesson 38 AOT / Lesson 39 JIT), the attention backend (Lesson 33), and a few platform hooks. This is precisely the guide's recurring "<strong>everything is pluggable</strong>" theme (design-theme forward-ref Lesson 62).</p></div>

<h2>1. The platform abstraction: one base class, one subclass per chip</h2>
<p>The core of the platform abstraction is a base class <span class="mono">SRTPlatform</span>, defined in <span class="mono">srt/platforms/interface.py</span>. It does no work directly; instead it declares a set of questions "every chip must answer" — and these questions take three forms: <strong>factory methods</strong>, <strong>capability flags</strong>, and <strong>lifecycle hooks</strong>.</p>
<p><strong>Factory methods</strong> are responsible for "building" the components for the current chip. For example, <span class="mono">get_default_attention_backend()</span> returns which family of attention kernels this chip should use (Lesson 33 covered backends like FlashAttention, FlashInfer, Triton); <span class="mono">get_graph_runner_cls()</span> returns the CUDA-graph runner class (Lesson 27 covered recording the whole forward pass into a graph to eliminate launch overhead). Different chips return different implementations in these two methods, while the upper tiers just call them, indifferent to which one comes back.</p>
<p><strong>Capability flags</strong> are a set of boolean judgments that let upper tiers ask "can this chip do a certain thing." For instance <span class="mono">supports_fp8</span> (can it run FP8 quantization), <span class="mono">support_cuda_graph</span> (can it use CUDA Graph), <span class="mono">support_piecewise_cuda_graph</span> (can it use piecewise CUDA Graph). Upper tiers adapt accordingly: if some chip doesn't support CUDA Graph, they gracefully fall back to per-operator execution instead of crashing outright.</p>
<p><strong>Lifecycle hooks</strong> let a chip "step in" at engine startup to adjust defaults. <span class="mono">apply_server_args_defaults()</span> is such a hook: each chip can set its own suitable default flags here (e.g. some chips disable a feature by default). There is also a <span class="mono">supported_quantization</span> list declaring which quantization formats this chip supports.</p>

<h2>2. Two trees: platforms and hardware_backend</h2>
<p>The code has two mirroring directory trees. The first is <span class="mono">srt/platforms/</span>, containing each chip's subclassing of <span class="mono">SRTPlatform</span>: <span class="mono">CudaSRTPlatform</span> (NVIDIA, the in-tree default), plus ROCm (AMD), CPU, etc., living in <span class="mono">srt/platforms/{cuda,rocm,cpu}.py</span>. Each subclass overrides the base class's factory methods and capability flags, answering "how my chip does it."</p>
<p>The second is <span class="mono">srt/hardware_backend/</span>, which organizes a set of backend directories by device: <span class="mono">cpu/ gpu/ npu/ xpu/ musa/ mlx/</span>. They cover NVIDIA and AMD GPUs, Huawei Ascend NPUs, Intel XPUs, Moore Threads MUSA, Apple MLX, and plain CPU. Note that Google TPU is not in this tree, but is served via a separate sglang-jax project.</p>

<h2>3. Key insight: the upper tiers are hardware-agnostic</h2>
<p>The most important sentence of this lesson is: <strong>the upper tiers are hardware-agnostic</strong>. The scheduler (Lesson 18) decides who runs first and who runs later; the model (Lesson 26) defines the network structure; most modules only care about tensors in, tensors out. None of them know which chip is underneath, nor do they need to. The things that truly get "swapped per chip" are compressed into a very small scope: the kernels (Lesson 38 AOT precompiled / Lesson 39 JIT just-in-time), the attention backend (Lesson 33), and the platform hooks.</p>
<p>This layering matters because it turns "porting to new hardware," a huge undertaking, into "just implement a new <span class="mono">SRTPlatform</span> subclass plus a set of new kernels," a small one. The thousands of lines of upper scheduling and model code can be reused untouched. Capability flags act as the lubricant here: upper tiers needn't write an if-else branch for every chip, but instead uniformly ask "<span class="mono">support_cuda_graph</span>?" and take the fallback path on a "no."</p>

<div class="layers">
<div class="layer">🧠 Scheduler (Lesson 18) · Model (Lesson 26) · various modules — hardware-agnostic, unaware of which chip is underneath</div>
<div class="layer">🔌 SRTPlatform abstraction layer — factory methods + capability flags + lifecycle hooks (translating per chip)</div>
<div class="layer">⚙️ Per-chip kernels: attention backend (Lesson 33) · AOT kernels (Lesson 38) · JIT kernels (Lesson 39) — swapped per chip</div>
</div>

<div class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-label="Three horizontal bands: hardware-agnostic upper tiers, the SRTPlatform seam, per-chip kernels">
    <text x="24" y="28" style="font-weight:700;fill:var(--blue)">Upper · hardware-agnostic</text>
    <rect x="20" y="36" width="760" height="64" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <rect x="40" y="50" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="153" y="73" text-anchor="middle" style="font-size:13px">Scheduler</text>
    <rect x="287" y="50" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="400" y="73" text-anchor="middle" style="font-size:13px">Model</text>
    <rect x="534" y="50" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="647" y="73" text-anchor="middle" style="font-size:13px">Attention abstraction</text>
    <text x="24" y="120" style="font-weight:700;fill:var(--amber)">Seam · SRTPlatform</text>
    <rect x="20" y="128" width="760" height="64" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5;stroke-dasharray:6 4"/>
    <rect x="40" y="142" width="350" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="215" y="165" text-anchor="middle" class="mono" style="font-size:12px">capability flags supports_fp8…</text>
    <rect x="410" y="142" width="330" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="575" y="165" text-anchor="middle" class="mono" style="font-size:12px">device ops</text>
    <text x="24" y="212" style="font-weight:700;fill:var(--purple)">Lower · per-chip kernels</text>
    <rect x="20" y="220" width="760" height="64" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <rect x="40" y="234" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="153" y="257" text-anchor="middle" class="mono" style="font-size:13px">CUDA</text>
    <rect x="287" y="234" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="400" y="257" text-anchor="middle" class="mono" style="font-size:13px">HIP</text>
    <rect x="534" y="234" width="226" height="36" rx="6" style="fill:var(--panel-2);stroke:var(--line);stroke-width:1.5"/>
    <text x="647" y="257" text-anchor="middle" class="mono" style="font-size:13px">CANN</text>
  </svg>
  <div class="figcap"><b>Fig A · Three bands</b> — the upper tiers are hardware-agnostic (scheduler / model / attention abstraction); the middle is the SRTPlatform seam (capability flags + device ops), where hardware specifics live; the bottom is per-chip kernels (CUDA / HIP / CANN…).</div>
</div>

<table class="t">
<tr><th>Hardware</th><th>Its backend / kernels</th></tr>
<tr><td>NVIDIA GPU</td><td>CudaSRTPlatform · hardware_backend/gpu · CUDA attention kernels</td></tr>
<tr><td>AMD GPU</td><td>ROCm platform · hardware_backend/gpu · ROCm/HIP kernels</td></tr>
<tr><td>Huawei Ascend NPU</td><td>hardware_backend/npu · Ascend kernels</td></tr>
<tr><td>Intel XPU</td><td>hardware_backend/xpu · XPU kernels</td></tr>
<tr><td>Moore Threads MUSA</td><td>hardware_backend/musa · MUSA kernels</td></tr>
<tr><td>Apple MLX</td><td>hardware_backend/mlx · MLX kernels</td></tr>
<tr><td>Plain CPU</td><td>CPU platform · hardware_backend/cpu · CPU kernels</td></tr>
</table>

<div class="fig">
  <svg viewBox="0 0 800 330" role="img" aria-label="One SGLang engine fans out to many chips, each an SRTPlatform subclass answering capability flags">
    <rect x="300" y="20" width="200" height="48" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="400" y="49" text-anchor="middle" style="font-weight:700;fill:var(--accent-ink)">SGLang engine</text>
    <line x1="400" y1="68" x2="400" y2="98" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="150" y1="98" x2="670" y2="98" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="150" y1="98" x2="150" y2="118" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="410" y1="98" x2="410" y2="118" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="670" y1="98" x2="670" y2="118" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="150" y1="196" x2="150" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="410" y1="196" x2="410" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="670" y1="196" x2="670" y2="232" style="stroke:var(--line);stroke-width:1.5"/>
    <rect x="40" y="118" width="220" height="78" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="150" y="144" text-anchor="middle" style="font-weight:700">NVIDIA · CUDA</text>
    <text x="150" y="164" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform subclass</text>
    <text x="150" y="184" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✓ · graph ✓</text>
    <rect x="300" y="118" width="220" height="78" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="410" y="144" text-anchor="middle" style="font-weight:700">AMD · ROCm</text>
    <text x="410" y="164" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform subclass</text>
    <text x="410" y="184" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✓ · graph ✓</text>
    <rect x="560" y="118" width="220" height="78" rx="8" style="fill:var(--amber-soft);stroke:var(--amber);stroke-width:1.5"/>
    <text x="670" y="144" text-anchor="middle" style="font-weight:700">Ascend · NPU</text>
    <text x="670" y="164" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform subclass</text>
    <text x="670" y="184" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✓ · graph ✗</text>
    <rect x="40" y="232" width="220" height="78" rx="8" style="fill:var(--purple-soft);stroke:var(--purple);stroke-width:1.5"/>
    <text x="150" y="258" text-anchor="middle" style="font-weight:700">Intel · XPU</text>
    <text x="150" y="278" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform subclass</text>
    <text x="150" y="298" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✗ · graph ✗</text>
    <rect x="300" y="232" width="220" height="78" rx="8" style="fill:var(--blue-soft);stroke:var(--blue);stroke-width:1.5"/>
    <text x="410" y="258" text-anchor="middle" style="font-weight:700">Moore · MUSA</text>
    <text x="410" y="278" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform subclass</text>
    <text x="410" y="298" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✗ · graph ✓</text>
    <rect x="560" y="232" width="220" height="78" rx="8" style="fill:var(--teal-soft);stroke:var(--teal);stroke-width:1.5"/>
    <text x="670" y="258" text-anchor="middle" style="font-weight:700">Apple · MLX</text>
    <text x="670" y="278" text-anchor="middle" style="font-size:11px;fill:var(--muted)">SRTPlatform subclass</text>
    <text x="670" y="298" text-anchor="middle" class="mono" style="font-size:11px">fp8 ✗ · graph ✗</text>
  </svg>
  <div class="figcap"><b>Fig B · One engine → many chips</b> — the same SGLang engine fans out to NVIDIA (CUDA), AMD (ROCm), Ascend NPU, Intel XPU, MUSA, MLX; each is an SRTPlatform subclass answering flags like supports_fp8 / support_cuda_graph (some yes, some no).</div>
</div>

<div class="cols">
<div class="col"><strong>✅ Portable (one codebase runs everywhere)</strong><br/>The scheduler (Lesson 18), the model (Lesson 26), most modules, and the input/output (IO) pipeline. They are hardware-agnostic and reused as-is when chips change.</div>
<div class="col"><strong>🔧 Per-chip (must be swapped)</strong><br/>The various kernels (Lesson 38 / Lesson 39), the attention backend (Lesson 33), and the platform hooks (factory methods, capability flags, <span class="mono">apply_server_args_defaults</span>). Porting new hardware is mainly filling in this part.</div>
</div>

<div class="flow">
<div class="node">Upper-layer op request (e.g. one attention computation)</div>
<div class="arrow">→</div>
<div class="node">SRTPlatform dispatch + capability check (support_cuda_graph? supports_fp8?)</div>
<div class="arrow">→</div>
<div class="node">The matching device kernel executes (CUDA / ROCm / NPU / …)</div>
</div>

<h2>4. A look at what the base class looks like</h2>
<p>Below is a faithful condensed version of <span class="mono">SRTPlatform</span>. Notice that many methods in the base class directly <span class="mono">raise NotImplementedError</span> — this says "every chip must answer this itself"; while <span class="mono">CudaSRTPlatform</span>, as the in-tree default, gives NVIDIA's answers. A new out-of-tree chip just follows the pattern, subclassing and overriding these methods.</p>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/platforms/interface.py ::SRTPlatform</span><span class="ln">each chip subclasses: factory methods + capability flags + lifecycle hooks</span></div><pre>class SRTPlatform(DeviceMixin):
    # base class for a hardware platform; out-of-tree chips subclass + override
    supported_quantization = []

    def apply_server_args_defaults(self, server_args):   # per-chip default flags
        pass
    def get_default_attention_backend(self) -&gt; str:     # which attention kernel family (Lesson 33)
        raise NotImplementedError
    def get_graph_runner_cls(self) -&gt; type:             # the CUDA-graph runner (Lesson 27)
        raise NotImplementedError

class CudaSRTPlatform(CudaDeviceMixin, SRTPlatform):     # default in-tree NVIDIA platform
    def supports_fp8(self): return True
    def support_cuda_graph(self): return True
    def support_piecewise_cuda_graph(self): return True</pre></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/platforms/cuda.py ::CudaSRTPlatform</span><span class="ln">concrete platform: NVIDIA/CUDA answers the capability flags</span></div><pre>class CudaSRTPlatform(CudaDeviceMixin, SRTPlatform):
    # the concrete NVIDIA/CUDA platform. Upper layers ask capability
    # questions; this subclass answers them, so engine code stays
    # hardware-agnostic.
    def supports_fp8(self) -&gt; bool:
        return True
    def support_cuda_graph(self) -&gt; bool:
        return True
    def support_piecewise_cuda_graph(self) -&gt; bool:
        return True</pre></div>

<div class="card"><div class="tag">🧩 Concrete examples</div><ul>
<li><strong>Porting to a new accelerator</strong>: just add one <span class="mono">SRTPlatform</span> subclass plus its kernels — <strong>without touching</strong> the scheduler (Lesson 18) or the model (Lesson 26).</li>
<li><strong>A chip without CUDA-graph support</strong>: its <span class="mono">support_cuda_graph()</span> simply returns <span class="mono">False</span>, so the engine skips graph capture and falls back to per-operator execution, still running fine.</li>
</ul></div>

<div class="card key"><div class="tag">📌 Key points</div><ul>
<li>The <strong>platform abstraction</strong> lets one engine run on many chips: the base class <span class="mono">SRTPlatform</span> (in <span class="mono">srt/platforms/interface.py</span>) declares per-chip factory methods, capability flags, and lifecycle hooks.</li>
<li><strong>Factory methods</strong> like <span class="mono">get_default_attention_backend()</span> (Lesson 33 attention kernels), <span class="mono">get_graph_runner_cls()</span> (Lesson 27 CUDA-graph runner); <strong>hooks</strong> like <span class="mono">apply_server_args_defaults()</span>.</li>
<li>Each chip <strong>subclasses</strong> the base: <span class="mono">CudaSRTPlatform</span> (NVIDIA), ROCm (AMD), etc.; another tree <span class="mono">srt/hardware_backend/</span> organizes per-device backends by <span class="mono">cpu/gpu/npu/xpu/musa/mlx</span>.</li>
<li>The <strong>upper tiers are hardware-agnostic</strong>: the scheduler (Lesson 18), model (Lesson 26), and most modules don't know the chip; only the kernels (Lessons 38/39), attention backend (Lesson 33), and platform hooks are swapped per chip.</li>
<li><strong>Capability flags</strong> (<span class="mono">supports_fp8</span>, <span class="mono">support_cuda_graph</span>, <span class="mono">support_piecewise_cuda_graph</span>) let upper tiers ask "can this chip do X" and adapt (e.g. fall back when CUDA graph isn't available).</li>
<li>Google TPU is not in this tree but is served via a separate sglang-jax project — many-hardware doesn't mean cramming every chip into one codebase.</li>
<li>This is the engineering basis for "single GPU to large clusters across many kinds of hardware," and an embodiment of the guide's "<strong>everything is pluggable</strong>" theme (design-theme forward-ref Lesson 62).</li>
</ul></div>

<div class="card"><div class="tag">🏁 Part 9 wrap-up</div><p>And that wraps up Part 9, "Kernels &amp; Hardware." We started from the AOT and JIT kernels closest to the metal (Lessons 38, 39), looked at quantization and communication kernels, then strung together performance levers like CUDA Graph and the attention backend, and finally used this lesson's platform abstraction to gather them into one sentence: <strong>isolate the small handful of things that are per-chip well, and the upper tiers can be reused in a hardware-agnostic way</strong>. This is exactly why SGLang can scale from a single GPU all the way to large clusters spanning many kinds of hardware.</p></div>
"""}
