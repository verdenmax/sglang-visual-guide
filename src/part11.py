"""Part 11 · Advanced / optional (L49-52).

Lesson content for the advanced-topics part of the SGLang visual guide: multimodal VLM
serving, multi-LoRA batching, RL rollout & weight sync, and diffusion models.
Each LESSON_XX is a {"zh": html, "en": html} dict consumed via registry.CONTENT.
"""

LESSON_49 = {"zh": r"""
<p class="lead">本课讲 SGLang 如何把一个纯文本推理引擎，变成能同时处理<strong>图像、音频、视频 + 文本</strong>的多模态视觉语言模型（VLM）引擎。核心洞见只有一句话：<strong>只需要改两处接缝（seam），其余引擎完全不动</strong>。第一处是<span class="mono">输入接缝</span>——把原始像素 / 音频转成模型张量，并在 token 流里插入<strong>占位符（placeholder）token</strong>；第二处是<span class="mono">嵌入接缝</span>——在前向计算时，跑各模态的编码器（encoder）产生媒体嵌入，再把它们<strong>缝合（scatter）</strong>到 token 嵌入流里占位符所在的位置。缝合之后，序列就只是一行普通的嵌入向量，调度器（第18课）、分页 KV（第30课）、RadixAttention（第29课）、采样器（第28课）全部照常运行。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把文本引擎想象成一条<strong>填字游戏的横排格子</strong>。平时每个格子里填一个字（文本 token）。现在你想在句子中间嵌入一张图片，怎么办？你<strong>先用铅笔在格子里画上记号</strong>（占位符 token），告诉大家“这里以后要放图片，先占住 5 个格子”。等真正动笔时，你拿出图片，用<strong>专门的相机（ViT 编码器）</strong>把它压成几枚“图片印章”，然后<strong>精确地盖到那些画了记号的格子上</strong>，把铅笔记号替换掉。最后整排格子看起来就是一串均匀的内容——后面读句子的人根本不需要知道哪格原来是文字、哪格原来是图片，他照常一格一格往下读。这就是 VLM：<span class="mono">占位 + 在占位处织入嵌入</span>，而不是另造一台机器。</p>
<p>这个类比里有两个细节值得多看一眼。其一，<strong>画记号的人和盖印章的人是分工的</strong>：画记号发生在“排版阶段”，要先知道这张图大概占几格；盖印章发生在“付印阶段”，此刻才真正把图片压成印章按上去。其二，<strong>记号的数量必须和印章的数量严丝合缝</strong>——你留了 5 格，相机就得恰好产出 5 枚印章，多一枚少一枚都对不上。正因为排版时就把格数定死了，后面无论是装订（拼批）、分页，还是逐格朗读（解码），都能把这段“图片区间”当成普通格子一视同仁，完全不必为图片单开一条流程。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>很多人以为支持多模态需要“重写引擎”，其实不然。语言模型的本体只认<strong>一行嵌入向量</strong>——它不在乎那一行是从文本 id 查表得来的，还是从一张图片编码而来的。所以 SGLang 的策略极其克制：把“多模态”这件事压缩成两个动作。<strong>动作一</strong>在分词 / 预处理阶段（第14课）完成，由每个模型自带的 <span class="mono">BaseMultimodalProcessor</span> 负责把媒体转张量、并往 prompt token 流里插占位符。<strong>动作二</strong>在前向入口完成，由 <span class="mono">general_mm_embed_routine</span> 负责跑编码器、并把媒体嵌入散射到占位符位置。两件事一做完，剩下的整条流水线——调度、分页 KV、注意力、采样——<strong>一行都不用改</strong>。这就是 VLM 支持“便宜”的根本原因。</p>
<p>从更高的视角看，这是一种典型的<strong>“窄腰”架构</strong>：把所有模态的多样性压到两处接缝这条窄腰上，腰以下的引擎只面对统一的“嵌入序列”这一种抽象。图像、音频、视频，无论输入端多么千差万别，到了腰部都被翻译成同一种货币——嵌入；于是引擎核心可以只优化这一种货币的吞吐与延迟，而不必为每种模态各写一套调度与缓存。理解了这条窄腰，你就能预判很多设计问题的答案：新模态怎么加？写个处理器和编码器挂到接缝上。为什么图文混排不需要特殊批处理？因为缝合后它们都是嵌入。为什么 KV 缓存能跨请求复用同一张图的前缀？因为那张图早已变成确定的嵌入序列，与文本前缀别无二致。</p>
</div>

<h2>一、两处接缝：输入接缝与嵌入接缝</h2>
<p>把一个文本引擎升级成 VLM 引擎，SGLang 只在两个明确的位置开口子，这两个口子就是<strong>接缝</strong>。第一处叫<strong>输入接缝（INPUT seam）</strong>，它发生在请求刚进来、还在做分词 / 预处理的阶段（第14课）。每个支持多模态的模型都带一个 <span class="mono">BaseMultimodalProcessor</span> 子类，它做两件事：一是把<strong>原始像素 / 音频波形</strong>按该模型的要求归一化、切块、变换成模型能吃的<strong>张量</strong>；二是往 prompt 的<strong>token 流里插入占位符 token</strong>——也就是在媒体应该出现的位置，塞进一串特定 id（例如一连串 <span class="mono">&lt;image&gt;</span> 占位 id），用来“占住”这张图片将要占据的槽位（slot）。注意：此刻这些占位符还只是普通的整数 id，里面并没有任何图像信息，它们只是<strong>记号</strong>。</p>
<p>第二处叫<strong>嵌入接缝（EMBED seam）</strong>，它发生在前向计算（forward）的最前端，由 <span class="mono">general_mm_embed_routine</span> 统一处理。它先把 token id 照常查嵌入表得到文本嵌入；然后对每个模态，调用该模态的<strong>编码器</strong>（图像走 ViT，音频走音频编码器——有些编码器还自带 CUDA Graph 运行器，见第27课）算出媒体嵌入；最后把这些媒体嵌入<strong>散射（scatter）</strong>回 token 嵌入流——精确地落在那些 <span class="mono">input_ids == 该模态占位符 token</span> 的位置上。整个分发由 <span class="mono">data_embedding_funcs</span> 这张表驱动，它以 <span class="mono">placeholder_tokens</span>（占位符 token）为键，决定“哪种占位符该用哪个编码器去填”。</p>
<p>为什么非要分成两处、而不是在一个地方一口气做完？因为这两件事天然属于<strong>两个不同的阶段</strong>，各自需要的上下文也不同。输入接缝跑在 CPU 侧的请求预处理里，它要看到的是<strong>原始字节</strong>（一张 JPEG、一段 WAV），还要决定这张图按该模型的切块规则会占多少个槽位——这些都和具体张量形状、token 排布强相关，必须在拼批（batch）之前定下来。嵌入接缝则跑在 GPU 侧的前向里，它要看到的是<strong>整批已经对齐好的 token id 张量</strong>，才能在一次散射里把所有请求、所有模态的媒体嵌入一并写进去。把职责这样切开，既让处理器可以逐模型定制（不同 VLM 的切块、归一化规则差别极大），又让嵌入缝合保持成一个通用、与具体模型无关的算子。换句话说，<strong>“怎么把媒体变成槽位”交给模型，“怎么把嵌入塞进槽位”交给引擎</strong>，边界清晰。</p>

<div class="flow"><div class="node">原始图像 / 音频</div><div class="arrow">→</div><div class="node">处理器：转张量 + 插占位符 token</div><div class="arrow">→</div><div class="node">ViT / 音频编码器</div><div class="arrow">→</div><div class="node">在占位符处缝合嵌入</div><div class="arrow">→</div><div class="node">照常解码（调度 / KV / 注意力）</div></div>

<h2>二、缝合发生在哪里：一行嵌入里的“图片格子”</h2>
<p>理解 VLM 最直观的画面，是盯住前向计算入口那一行<strong>token 嵌入</strong>。假设 prompt 是“这是 [图片] 一只猫”，经过输入接缝后，token 流变成：<span class="mono">这 / 是 / IMG / IMG / IMG / 一 / 只 / 猫</span>，其中三个 <span class="mono">IMG</span> 就是占位符 token。<span class="mono">general_mm_embed_routine</span> 先把每个 id 查表得到嵌入，于是得到一整行嵌入向量；接着 ViT 把那张猫图编码成三枚媒体嵌入，<strong>替换</strong>掉那三个 <span class="mono">IMG</span> 槽位原本的占位嵌入。替换完成后，这一行里再也没有“占位符”的概念了——它就是 8 个并排的嵌入向量，前两个来自文字、中间三个来自图片、后三个又来自文字，混在一起完全均质。</p>

<div class="cellgroup"><div class="cell">这</div><div class="cell">是</div><div class="cell sc">🖼️ img-emb</div><div class="cell sc">🖼️ img-emb</div><div class="cell sc">🖼️ img-emb</div><div class="cell">一</div><div class="cell">只</div><div class="cell">猫</div></div>

<p>上面这排格子里，带底色的三格原本是 <span class="mono">IMG</span> 占位符，现在被换成了图片嵌入（<span class="mono">img-emb</span>）。关键是：<strong>格子的数量和位置完全没变</strong>——占位符占了几格，图片嵌入就填几格。这保证了序列长度在缝合前后一致，于是下游所有按位置工作的组件（位置编码、注意力掩码、KV 槽位）都不需要任何特殊处理。</p>
<p>这里还藏着一个常被忽略的细节：<strong>占位符的数量必须事先算准</strong>。处理器在输入接缝插占位符时，并不是随便塞几个，而是按该模型 ViT 的输出长度精确计算——一张图经过切块、池化后会产生多少个视觉 token，就插多少个占位符。如果数量对不上，缝合时散射的源（媒体嵌入）和目标（占位符槽位）就会形状不匹配，直接报错。所以输入接缝和嵌入接缝之间有一个<strong>隐含契约</strong>：处理器承诺“我为这张图留了 N 个槽位”，编码器承诺“我恰好产出 N 个嵌入”，两者由同一个占位符 token 串成一条线。正因为这个契约在预处理阶段就钉死了，下游拼批、分页、注意力才能把这段“图片区间”当成普通 token 一视同仁地处理。多图、多模态混排时也一样：每个媒体项各有自己的一段占位符，互不干扰，散射时各回各的槽位。</p>

<h2>三、为什么其余引擎一行都不用改</h2>
<p>这是本课最重要的一句话：<strong>VLM 支持 = “处理媒体 + 在占位符处织入嵌入”，而不是一台新引擎</strong>。一旦缝合完成，序列就退化成一行普通嵌入，语言模型主体根本分辨不出哪段来自图片。于是引擎的其余部分原封不动地复用：<strong>调度器</strong>（第18课）照样按 token 数排队、做连续批处理；<strong>分页 KV 缓存</strong>（第30课）照样按页分配、按 token 写入 KV；<strong>RadixAttention</strong>（第29课）照样按前缀树复用缓存、做前缀匹配；<strong>采样器</strong>（第28课）照样从 logits 里采样下一个 token。多模态信息在“嵌入接缝”处一次性融入序列后，后续没有任何一处需要知道它的存在。</p>
<p>这种“克制”带来的工程红利非常实在。第一，<strong>复杂度被隔离</strong>：所有和模态相关的脏活——不同图片分辨率、不同音频采样率、不同模型的切块规则——全部关在处理器和编码器这两个盒子里，引擎核心保持单一职责。第二，<strong>性能优化自动继承</strong>：你为文本引擎做的连续批处理、前缀缓存复用、分页显存管理，VLM 请求一行不改地全盘享受；甚至一段“文本前缀 + 同一张图”被多个请求共享时，RadixAttention 仍能在缝合后的嵌入序列上命中前缀缓存。第三，<strong>新模型接入成本低</strong>：要支持一个新的 VLM，通常只需写一个新的 <span class="mono">BaseMultimodalProcessor</span> 子类、挂上它的编码器、在 <span class="mono">data_embedding_funcs</span> 里注册“占位符 → 编码器”的映射，引擎其余部分完全不用碰。这正是把变化点收敛到两处接缝的价值：<strong>新增的是适配器，不是引擎</strong>。</p>

<div class="cols"><div class="col"><strong>VLM 专属（新增的两处接缝）</strong><ul><li>处理器：像素 / 音频 → 张量 + 插占位符</li><li>编码器：ViT / 音频编码器算媒体嵌入</li><li>缝合：在占位符位置散射嵌入</li></ul></div><div class="col"><strong>完全复用（一行不改）</strong><ul><li>调度器：连续批处理（第18课）</li><li>分页 KV 缓存：分页存取（第30课）</li><li>RadixAttention：前缀复用（第29课）</li><li>采样器：从 logits 采样（第28课）</li></ul></div></div>

<h2>四、把两处接缝对到具体职责</h2>
<p>下面这张表把两处接缝、它们所在的阶段、以及各自的职责一一对上。可以看到，<strong>输入接缝</strong>属于分词 / 预处理阶段（第14课），交付的是“张量 + 带占位符的 token 流”；<strong>嵌入接缝</strong>属于前向计算阶段，交付的是“缝合好媒体嵌入的一行向量”。两者一前一后，把媒体从“原始字节”一路接力到“可被语言模型直接消费的嵌入”，中间没有惊动引擎的任何其他部分。</p>
<p>不妨把整条路径再完整走一遍，体会两处接缝如何接力。用户发来一条带图请求，分词器（第14课）先把文本部分切成 token，与此同时输入接缝里的处理器把图片解码、归一化、按模型规则切块成张量，并在文本 token 流的对应位置插入一串占位符 token；此刻请求带着“张量 + 含占位符的 token 序列”进入调度器（第18课）排队、拼批。轮到它前向时，嵌入接缝接手：先把整批 token id 查表得到文本嵌入，再让 ViT/音频编码器把张量编码成媒体嵌入，最后一次散射把媒体嵌入写到占位符槽位上。从这一步往后，序列就是纯粹的嵌入，分页 KV（第30课）按 token 分配缓存、RadixAttention（第29课）做前缀匹配与复用、采样器（第28课）从 logits 采样下一个 token——它们谁都不知道、也不需要知道这里曾经有过一张图。每解码出一个新 token，循环继续，直到生成结束。<strong>整条链路里，“多模态”只在最前面两处接缝短暂现身，之后彻底消融进通用的嵌入流</strong>，这就是 SGLang VLM 设计的全部精髓：用最小的改动面，换来对整套高性能基础设施的完整复用。</p>

<table class="t"><tr><th>接缝</th><th>所在阶段</th><th>职责</th></tr><tr><td>输入处理器 <span class="mono">BaseMultimodalProcessor</span></td><td>分词 / 预处理（第14课）</td><td>原始像素 / 音频 → 模型张量；往 token 流插入占位符 token</td></tr><tr><td>嵌入入口 <span class="mono">general_mm_embed_routine</span></td><td>前向计算入口</td><td>跑编码器算媒体嵌入；在 <span class="mono">input_ids==占位符</span> 处散射缝合</td></tr></table>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/mm_utils.py ::general_mm_embed_routine</span><span class="ln">在占位符处把媒体嵌入拼接进 token 序列</span></div><pre>def general_mm_embed_routine(
    input_ids,                  # prompt tokens, with PLACEHOLDER ids where media goes
    forward_batch,
    language_model,
    data_embedding_funcs,       # {Modality: fn} &mdash; run that modality's encoder to embed it
    placeholder_tokens=None,    # which token id marks each modality's slots
    **kwargs,
):
    # 1) embed the text token ids normally
    # 2) for each modality, run its encoder to get media embeddings
    # 3) SCATTER those embeddings into the token-embedding stream at the
    #    positions where input_ids == that modality's placeholder token
    # 4) hand the merged embeddings to the language model; scheduler / paged KV /
    #    attention downstream are unchanged
    ...</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li><strong>两处接缝</strong>把文本引擎变成 VLM 引擎，其余完全不动：<span class="mono">输入接缝</span> + <span class="mono">嵌入接缝</span>。</li>
<li><strong>输入接缝</strong>由每模型的 <span class="mono">BaseMultimodalProcessor</span> 负责：原始像素 / 音频 → 张量，并往 token 流插入<strong>占位符 token</strong>。</li>
<li><strong>嵌入接缝</strong>由 <span class="mono">general_mm_embed_routine</span> 负责：先嵌入文本，再跑各模态<strong>编码器</strong>（ViT / 音频编码器，可带 CUDA Graph，第27课），把媒体嵌入<strong>散射</strong>到占位符位置。</li>
<li>分发由 <span class="mono">data_embedding_funcs</span> 驱动，以 <span class="mono">placeholder_tokens</span> 为键，决定哪种占位符用哪个编码器填。</li>
<li>缝合后序列只是一行嵌入，所以<strong>调度器（第18课）/ 分页 KV（第30课）/ RadixAttention（第29课）/ 采样器（第28课）全部复用</strong>。</li>
<li>核心洞见：VLM = <span class="mono">处理媒体 + 在占位符处织入嵌入</span>，不是一台新引擎。</li>
<li>这是一种<strong>“窄腰”架构</strong>：所有模态在两处接缝被翻译成统一的嵌入序列，腰以下的引擎只面对这一种抽象，因而新模态只需新增适配器即可接入。</li>
</ul>
</div>
""", "en": r"""
<p class="lead">This lesson shows how SGLang turns a text-only inference engine into a multimodal vision-language model (VLM) engine that can handle <strong>image, audio, video + text</strong> together. The key insight is a single sentence: <strong>you only change two seams; the rest of the engine is untouched</strong>. The first is the <span class="mono">INPUT seam</span> — turn raw pixels / audio into model tensors, and insert <strong>PLACEHOLDER tokens</strong> into the token stream. The second is the <span class="mono">EMBED seam</span> — at forward time, run each modality's encoder to produce media embeddings, then <strong>scatter</strong> them into the token-embedding stream exactly at the placeholder positions. After that splice, the sequence is just a row of ordinary embeddings, so the scheduler (L18), paged KV (L30), RadixAttention (L29), and sampler (L28) all run as usual.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture the text engine as a single <strong>row of crossword squares</strong>. Normally each square holds one character (a text token). Now you want to embed a picture in the middle of the sentence. What do you do? You <strong>first pencil in a mark</strong> (a placeholder token) to announce: "a picture goes here later — reserve 5 squares." When it's actually time to write, you take the photo, use a <strong>dedicated camera (the ViT encoder)</strong> to compress it into a few "picture stamps", and then <strong>stamp them exactly onto the squares you penciled</strong>, replacing the pencil marks. In the end the whole row looks like one even strip of content — whoever reads the sentence afterward never needs to know which square was originally text and which was a picture; they just read square by square. That is VLM: <span class="mono">reserve slots + weave embeddings in at those slots</span>, not building a new machine.</p>
<p>Two details in this analogy deserve a second look. First, <strong>the one who pencils the marks and the one who stamps are different roles</strong>: penciling happens at the "layout stage," which must first know roughly how many squares this picture occupies; stamping happens at the "printing stage," where the picture is finally compressed into stamps and pressed down. Second, <strong>the number of marks must match the number of stamps exactly</strong> — you reserved 5 squares, so the camera must produce exactly 5 stamps; one too many or too few will not line up. Precisely because the square count is fixed at layout time, everything afterward — binding (batching), paging, or reading square by square (decoding) — can treat this "image span" as ordinary squares, with no separate pipeline for the picture.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Many assume that supporting multimodality requires "rewriting the engine." It does not. The language model's body only ever sees <strong>one row of embedding vectors</strong> — it does not care whether a row came from a text-id lookup or from encoding a picture. So SGLang's strategy is extremely restrained: compress "multimodality" into two actions. <strong>Action one</strong> happens in the tokenize / preprocess stage (L14), where each model's own <span class="mono">BaseMultimodalProcessor</span> turns media into tensors and inserts placeholders into the prompt token stream. <strong>Action two</strong> happens at the forward entry, where <span class="mono">general_mm_embed_routine</span> runs the encoders and scatters media embeddings into the placeholder positions. Once those two are done, the rest of the pipeline — scheduling, paged KV, attention, sampling — <strong>does not change a single line</strong>. That is the fundamental reason VLM support is "cheap."</p>
<p>From a higher vantage, this is a textbook <strong>"narrow waist" architecture</strong>: all the diversity of modalities is squeezed onto the narrow waist of the two seams, and everything below the waist faces only one abstraction — a unified "embedding sequence." Image, audio, video, however wildly different at the input end, are all translated into the same currency at the waist — embeddings; so the engine core can optimize throughput and latency for just this one currency, without writing a separate scheduler and cache for each modality. Once you grasp this narrow waist, you can predict the answers to many design questions: how do you add a new modality? Write a processor and an encoder and hook them onto the seams. Why does interleaved image-text need no special batching? Because after the splice they are all embeddings. Why can the KV cache reuse the prefix of the same image across requests? Because that image has long since become a deterministic embedding sequence, no different from a text prefix.</p>
</div>

<h2>1. The two seams: the input seam and the embed seam</h2>
<p>To upgrade a text engine into a VLM engine, SGLang only opens incisions at two precise locations; those incisions are the <strong>seams</strong>. The first is the <strong>INPUT seam</strong>, which happens when a request has just arrived and is still being tokenized / preprocessed (L14). Every multimodal-capable model carries a <span class="mono">BaseMultimodalProcessor</span> subclass that does two things: first, it normalizes, tiles, and transforms the <strong>raw pixels / audio waveform</strong> into the <strong>tensors</strong> the model expects; second, it <strong>inserts PLACEHOLDER tokens into the prompt's token stream</strong> — that is, at the position where the media should appear, it stuffs in a run of specific ids (e.g. a run of <span class="mono">&lt;image&gt;</span> placeholder ids) to "reserve" the slots this picture will occupy. Note: at this moment those placeholders are still just plain integer ids carrying no image information — they are merely <strong>marks</strong>.</p>
<p>The second is the <strong>EMBED seam</strong>, which happens at the very front of the forward pass and is handled uniformly by <span class="mono">general_mm_embed_routine</span>. It first looks up the token ids in the embedding table as usual to get text embeddings; then, for each modality, it calls that modality's <strong>encoder</strong> (image goes through a ViT, audio through an audio encoder — some encoders even carry their own CUDA-graph runner, see L27) to compute media embeddings; finally it <strong>scatters</strong> those media embeddings back into the token-embedding stream — landing exactly on the positions where <span class="mono">input_ids == that modality's placeholder token</span>. The whole dispatch is driven by the table <span class="mono">data_embedding_funcs</span>, keyed by <span class="mono">placeholder_tokens</span>, which decides "which placeholder gets filled by which encoder."</p>
<p>Why split this into two places instead of doing it all at once? Because these two jobs naturally belong to <strong>two different stages</strong> with different context needs. The input seam runs in CPU-side request preprocessing; it must see the <strong>raw bytes</strong> (a JPEG, a WAV clip) and decide how many slots this image will occupy under that model's tiling rules — all strongly tied to concrete tensor shapes and token layout, and all must be fixed before batching. The embed seam runs in the GPU-side forward; it must see the <strong>whole batch of aligned token-id tensors</strong> so it can, in one scatter, write the media embeddings of all requests and all modalities at once. Splitting responsibilities this way lets the processor be customized per model (different VLMs differ wildly in tiling and normalization rules) while keeping the embed splice a generic, model-agnostic operator. In other words, <strong>"how to turn media into slots" is the model's job, "how to put embeddings into slots" is the engine's job</strong> — a clean boundary.</p>

<div class="flow"><div class="node">raw image / audio</div><div class="arrow">→</div><div class="node">processor: to tensors + insert placeholder tokens</div><div class="arrow">→</div><div class="node">ViT / audio encoder</div><div class="arrow">→</div><div class="node">splice embeddings at placeholders</div><div class="arrow">→</div><div class="node">normal decode (sched / KV / attention)</div></div>

<h2>2. Where the splice happens: the "picture squares" inside one row of embeddings</h2>
<p>The most intuitive picture of VLM is to stare at that one row of <strong>token embeddings</strong> at the forward entry. Suppose the prompt is "this is [picture] a cat." After the input seam, the token stream becomes: <span class="mono">this / is / IMG / IMG / IMG / a / cat</span>, where the three <span class="mono">IMG</span> are placeholder tokens. <span class="mono">general_mm_embed_routine</span> first looks up each id to get an embedding, giving a full row of embedding vectors; then the ViT encodes that cat photo into three media embeddings that <strong>replace</strong> the placeholder embeddings originally sitting in those three <span class="mono">IMG</span> slots. After the replacement, this row no longer has any notion of "placeholder" — it is just a strip of embedding vectors side by side, some from text, some from the picture, mixed together completely homogeneously.</p>

<div class="cellgroup"><div class="cell">this</div><div class="cell">is</div><div class="cell sc">🖼️ img-emb</div><div class="cell sc">🖼️ img-emb</div><div class="cell sc">🖼️ img-emb</div><div class="cell">a</div><div class="cell">cat</div></div>

<p>In the row of squares above, the three shaded cells were originally <span class="mono">IMG</span> placeholders and are now replaced by image embeddings (<span class="mono">img-emb</span>). The key point: <strong>the count and positions of the squares do not change at all</strong> — however many slots the placeholders held, that many image embeddings fill them. This guarantees the sequence length is identical before and after the splice, so every downstream component that works by position (positional encoding, attention mask, KV slots) needs no special handling.</p>
<p>There is an often-overlooked detail here: <strong>the number of placeholders must be computed exactly in advance</strong>. When the processor inserts placeholders at the input seam, it does not stuff in some arbitrary count — it computes it precisely from that model's ViT output length: however many visual tokens an image produces after tiling and pooling, that many placeholders go in. If the counts disagree, the scatter's source (media embeddings) and target (placeholder slots) will mismatch in shape and error out immediately. So there is an <strong>implicit contract</strong> between the input seam and the embed seam: the processor promises "I reserved N slots for this image," the encoder promises "I produce exactly N embeddings," and the same placeholder token strings the two together. Precisely because this contract is nailed down in preprocessing, downstream batching, paging, and attention can treat this "image span" as ordinary tokens with no special-casing. The same holds for multiple images and mixed modalities: each media item has its own run of placeholders, mutually disjoint, and the scatter returns each to its own slots.</p>

<h2>3. Why the rest of the engine does not change a single line</h2>
<p>This is the most important sentence of the lesson: <strong>VLM support = "process media + weave embeddings in at placeholders," not a new engine</strong>. Once the splice is done, the sequence degenerates into a row of ordinary embeddings, and the language-model body simply cannot tell which segment came from a picture. So the rest of the engine is reused untouched: the <strong>scheduler</strong> (L18) still queues by token count and does continuous batching; the <strong>paged KV cache</strong> (L30) still allocates by page and writes KV per token; <strong>RadixAttention</strong> (L29) still reuses cache by prefix tree and does prefix matching; the <strong>sampler</strong> (L28) still samples the next token from logits. Once multimodal information is fused into the sequence once at the embed seam, no later stage ever needs to know it exists.</p>
<p>The engineering dividends of this restraint are very concrete. First, <strong>complexity is isolated</strong>: all modality-specific dirty work — varied image resolutions, varied audio sample rates, each model's tiling rules — is confined to the two boxes of processor and encoder, while the engine core keeps a single responsibility. Second, <strong>performance optimizations are inherited for free</strong>: the continuous batching, prefix-cache reuse, and paged VRAM management you built for the text engine are enjoyed wholesale by VLM requests without a line of change; even when a "text prefix + same image" is shared by multiple requests, RadixAttention can still hit the prefix cache on the spliced embedding sequence. Third, <strong>onboarding a new model is cheap</strong>: to support a new VLM you usually only write a new <span class="mono">BaseMultimodalProcessor</span> subclass, hook up its encoder, and register the "placeholder → encoder" mapping in <span class="mono">data_embedding_funcs</span> — the rest of the engine is never touched. That is exactly the value of converging the change points into two seams: <strong>what you add is an adapter, not an engine</strong>.</p>

<div class="cols"><div class="col"><strong>VLM-specific (the two new seams)</strong><ul><li>processor: pixels / audio → tensors + insert placeholders</li><li>encoder: ViT / audio encoder computes media embeddings</li><li>splice: scatter embeddings at placeholder positions</li></ul></div><div class="col"><strong>Fully reused (not a line changed)</strong><ul><li>scheduler: continuous batching (L18)</li><li>paged KV cache: paged storage (L30)</li><li>RadixAttention: prefix reuse (L29)</li><li>sampler: sample from logits (L28)</li></ul></div></div>

<h2>4. Mapping the two seams to concrete responsibilities</h2>
<p>The table below maps the two seams, the stages they live in, and their respective duties one by one. You can see that the <strong>input seam</strong> belongs to the tokenize / preprocess stage (L14) and delivers "tensors + a token stream with placeholders"; the <strong>embed seam</strong> belongs to the forward stage and delivers "a row of vectors with media embeddings spliced in." Together, front and back, they relay media from "raw bytes" all the way to "embeddings the language model can directly consume," without disturbing any other part of the engine in between.</p>
<p>Let us walk the whole path once more to feel how the two seams hand off. A user sends a request with an image; the tokenizer (L14) cuts the text part into tokens, while the processor at the input seam decodes, normalizes, and tiles the image into tensors per the model's rules, and inserts a run of placeholder tokens at the corresponding positions in the text token stream. At this point the request, carrying "tensors + a token sequence with placeholders," enters the scheduler (L18) to queue and batch. When its turn to run forward comes, the embed seam takes over: it looks up the whole batch of token ids to get text embeddings, lets the ViT/audio encoder encode the tensors into media embeddings, and in one scatter writes those media embeddings onto the placeholder slots. From this step on, the sequence is pure embeddings: paged KV (L30) allocates cache per token, RadixAttention (L29) does prefix matching and reuse, and the sampler (L28) samples the next token from logits — none of them knows, or needs to know, that an image was ever here. Each newly decoded token continues the loop until generation ends. <strong>Across the whole pipeline, "multimodality" appears only briefly at the two front seams and afterward dissolves completely into the generic embedding stream</strong> — that is the entire essence of SGLang's VLM design: the smallest change surface in exchange for full reuse of the whole high-performance infrastructure.</p>

<table class="t"><tr><th>Seam</th><th>Stage</th><th>Responsibility</th></tr><tr><td>input processor <span class="mono">BaseMultimodalProcessor</span></td><td>tokenize / preprocess (L14)</td><td>raw pixels / audio → model tensors; insert placeholder tokens into the token stream</td></tr><tr><td>embed entry <span class="mono">general_mm_embed_routine</span></td><td>forward entry</td><td>run encoders to compute media embeddings; scatter-splice where <span class="mono">input_ids==placeholder</span></td></tr></table>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/managers/mm_utils.py ::general_mm_embed_routine</span><span class="ln">splice media embeddings into the token stream at placeholder slots</span></div><pre>def general_mm_embed_routine(
    input_ids,                  # prompt tokens, with PLACEHOLDER ids where media goes
    forward_batch,
    language_model,
    data_embedding_funcs,       # {Modality: fn} &mdash; run that modality's encoder to embed it
    placeholder_tokens=None,    # which token id marks each modality's slots
    **kwargs,
):
    # 1) embed the text token ids normally
    # 2) for each modality, run its encoder to get media embeddings
    # 3) SCATTER those embeddings into the token-embedding stream at the
    #    positions where input_ids == that modality's placeholder token
    # 4) hand the merged embeddings to the language model; scheduler / paged KV /
    #    attention downstream are unchanged
    ...</pre></div>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li><strong>Two seams</strong> turn a text engine into a VLM engine, with everything else untouched: the <span class="mono">input seam</span> + the <span class="mono">embed seam</span>.</li>
<li>The <strong>input seam</strong> is handled by each model's <span class="mono">BaseMultimodalProcessor</span>: raw pixels / audio → tensors, and insert <strong>placeholder tokens</strong> into the token stream.</li>
<li>The <strong>embed seam</strong> is handled by <span class="mono">general_mm_embed_routine</span>: embed text first, then run each modality's <strong>encoder</strong> (ViT / audio encoder, may carry CUDA Graph, L27) and <strong>scatter</strong> media embeddings into the placeholder positions.</li>
<li>Dispatch is driven by <span class="mono">data_embedding_funcs</span>, keyed by <span class="mono">placeholder_tokens</span>, deciding which placeholder is filled by which encoder.</li>
<li>After the splice the sequence is just a row of embeddings, so the <strong>scheduler (L18) / paged KV (L30) / RadixAttention (L29) / sampler (L28) are all reused</strong>.</li>
<li>Core insight: VLM = <span class="mono">process media + weave embeddings in at placeholders</span>, not a new engine.</li>
<li>This is a <strong>"narrow waist" architecture</strong>: all modalities are translated into a unified embedding sequence at the two seams, and the engine below the waist faces only this one abstraction, so a new modality only needs a new adapter to plug in.</li>
</ul>
</div>
"""}
LESSON_50 = {"zh": r"""
<p class="lead">一个底座模型，几十种"技能"。<strong>多 LoRA 批处理（Multi-LoRA batching）</strong>让你只在显存里保留<span class="mono">一份</span>基座权重，外加一个小小的<strong>适配器池（adapter pool）</strong>，就能同时服务大量针对不同任务微调出来的模型。难点不在"加载适配器"，而在"同一个批次里不同请求各要各的适配器"时，如何用<span class="mono">一次</span>内核启动把每个请求自己的增量权重 ΔW 都算对。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>把基座模型想成一台<strong>电动工具的主机</strong>：电机、把手、电池都在主机里，又重又贵。而 LoRA 适配器就像一个个<strong>可换的钻头/批头</strong>——每个钻头很小、很便宜，却让同一台主机分别变成"打孔机""螺丝刀""抛光机"。</p>
<p>过去的笨办法是：要三种功能就买三台完整的主机（三份完整模型），仓库（显存）立刻被占满。聪明的办法是：<strong>一台主机 + 一盒批头</strong>。来活儿了，看这一单需要哪个批头，<span class="mono">装上</span>就干。更妙的是流水线上同时来了好几单：第一单要打孔、第二单要拧螺丝、第三单又要打孔——你不是把主机拆三遍，而是<strong>一次</strong>把这几个批头都摆好，让机械臂在一趟动作里把每个工件按它该用的批头加工完。这"一趟动作处理一批不同批头的活儿"，正是多 LoRA 批处理要解决的事。</p>
<p>这个类比还能再延伸一点：批头盒的容量是有限的，你不可能把全世界的批头都同时插在机械臂上等着用。所以现场只保留"这一轮最可能用到"的那几个批头，用完不需要的就收回盒子、把新需要的拿出来——这正对应运行时的"装载/卸载适配器"，以及"同一轮最多摆几个批头"的上限。整套办法的精髓始终是同一句话：<strong>贵的主机只留一台，便宜的批头按需轮换</strong>。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>LoRA 的核心数学很朴素：不去重新训练庞大的权重矩阵 <span class="mono">W</span>，而是学两个<strong>小矩阵</strong> <span class="mono">A</span> 和 <span class="mono">B</span>，让有效权重变成 <span class="mono">W + B·A</span>。这里的秩 <span class="mono">r</span> 远小于矩阵的维度（<span class="mono">r ≪ d</span>），所以 <span class="mono">B·A</span> 这个增量 ΔW 又小又便宜。一个基座可以挂<strong>许多</strong>这样的适配器，每个代表一种被微调出来的"技能"。SGLang 把单个适配器表示为 <span class="mono">LoRAAdapter</span>。</p>
<p>于是多 LoRA 的真正主张是：<strong>一份基座 + 一个适配器池 + 批次感知的 ΔW 应用</strong>。一次部署就能廉价地服务几十个针对具体任务的适配器，而不必为每个微调各起一份完整模型。</p>
<p>为什么 <span class="mono">B·A</span> 会这么便宜？设权重维度是 <span class="mono">d×d</span>，全量微调要更新约 <span class="mono">d²</span> 个参数；而 LoRA 只学 <span class="mono">A</span>（约 <span class="mono">r×d</span>）和 <span class="mono">B</span>（约 <span class="mono">d×r</span>），合计约 <span class="mono">2·r·d</span> 个参数。当秩 <span class="mono">r</span> 只有几十、而维度 <span class="mono">d</span> 是几千时，参数量会差出两三个数量级。这意味着<strong>训练成本</strong>、<strong>存储成本</strong>、<strong>显存占用</strong>都随之骤降——一个适配器小到可以几十个一起塞进一块卡的余量里，这正是"适配器池"在工程上成立的前提。</p>
</div>

<h2>为什么"一份基座 + 适配器池"</h2>
<p>假设你有 N 个微调版本：一个会写 SQL、一个会医疗问答、一个会法律摘要。最直白的部署是起 N 份完整模型——但每份都要完整地占据显存，而它们之间<strong>99% 的权重是一模一样的</strong>（都源自同一个基座）。这是巨大的浪费。</p>
<p>SGLang 的做法是：显存里只放<span class="mono">一份</span>基座模型（参见 <strong>第25课</strong> 讲的基座权重加载），另外维护一个 <strong>LoRA 内存池</strong>，池子里放着若干个小适配器。每来一个请求，就根据它指定的适配器名，把对应的 <span class="mono">B·A</span> 增量临时叠加到基座输出上。适配器很小，所以池子能装下很多个；真正的约束是 <span class="mono">max_loras_per_batch</span>——它限定<strong>同一个批次</strong>里最多能共存多少个不同的适配器（受池子/显存大小限制）。</p>
<p>更棒的是，适配器可以在<strong>运行时</strong>动态增删：通过 <span class="mono">load_lora_adapter</span> 往池子里加一个新技能，通过 <span class="mono">unload_lora_adapter</span> 把不用的踢出去腾地方——全程<strong>不需要重启</strong>服务。</p>
<p>这种"热插拔"在生产里非常实用。比如你上线了一个新的客服话术适配器，只需把它装进池子，新请求立刻就能指定使用，老请求毫不受影响；当某个适配器很久没人用、又快撑爆池子时，把它卸载即可，腾出的空间留给更热门的技能。整个过程没有停机窗口，也不需要重新加载几十 GB 的基座权重——因为基座从头到尾就是那一份，真正进出的只是几兆到几十兆的小增量。</p>

<h2>批处理才是真正的难点</h2>
<p>如果一个批次里所有请求都用同一个适配器，那事情很简单：把 ΔW 算一遍，加到整批上即可。但现实是——同一批里的请求往往<strong>各要各的适配器</strong>：req1 要 adapterA、req2 要 adapterB、req3 又要 adapterA、req4 要 adapterC……如果按适配器一个一个循环处理，就退化成了多次小 GEMM，把连续批处理（<strong>第5课</strong>）辛辛苦苦攒起来的吞吐又拆散了。</p>
<p>SGLang 的关键函数是 <span class="mono">LoRAManager.prepare_lora_batch(forward_batch)</span>。它接过当前的 <span class="mono">ForwardBatch</span>（<strong>第24课</strong>），<strong>逐请求</strong>看清每一行需要哪个适配器，把这些适配器的权重在内存里<strong>摆放/分段（stage &amp; segment）</strong>好，使得后续可以用一次<strong>分段 / 分组 GEMM（segmented / grouped GEMM）</strong>，在<span class="mono">一次</span>内核启动里就把每个请求<strong>各自的</strong> ΔW 都应用上去，而不是为每个适配器单独跑一趟。</p>
<p>这就是"批次感知"的含义：不是无视适配器差异硬算，也不是为差异付出循环的代价，而是把差异<strong>编码进一次分组矩阵乘法的布局里</strong>，让 GPU 一趟搞定。</p>
<p>不妨把"逐适配器循环"和"分组 GEMM"放在一起比较。逐适配器循环时，假设这批里有 5 个不同适配器，就要发起 5 次小内核，每次都伴随一次内核启动开销、一次显存读写，而每个小内核处理的行数又很少，GPU 的算力利用率很低——这恰恰是连续批处理最想避免的"碎片化"。分组 GEMM 则把这 5 段拼成<strong>一次</strong>调用：启动开销摊到整批、显存只走一遍、GPU 的众多计算单元被同时喂饱。请求数越多、适配器越杂，这种"一趟做完"相对于"循环 N 趟"的优势就越明显。</p>
<p>具体一点说，分组 GEMM（grouped GEMM）的思路是：把一个大批次按"用哪个适配器"切成若干<strong>段（segment）</strong>，每一段对应一个适配器、包含该批里所有用到它的请求行。这些段被并排打包成一次内核调用的输入，内核内部对每一段套用它自己的 <span class="mono">B、A</span> 矩阵。于是从外部看，仍然只有<strong>一次</strong>内核启动、一次显存往返；从内部看，每个请求都精确地拿到了它该有的 ΔW。这正是 <span class="mono">prepare_lora_batch</span> 要在前向之前默默做好的"摆放"工作——计算每段的偏移、长度、对应的适配器索引，把零散的"谁要谁"整理成 GPU 喜欢的规整布局。</p>
<p>这里也能看出 <span class="mono">max_loras_per_batch</span> 为什么重要：段数越多，意味着这一批里活跃的不同适配器越多，需要常驻在池子里、参与这次分组 GEMM 的权重也越多。把它设得过大，会让 LoRA 内存池吃紧；设得过小，则可能逼迫调度器把"想用第 K+1 个适配器"的请求推迟到下一批。它本质上是在<strong>适配器多样性</strong>与<strong>显存预算</strong>之间画的一条线。</p>

<h2>运行时的生命周期</h2>
<p>把整条链路串起来：服务启动时，<span class="mono">LoRAManager</span> 拿到基座模型、<span class="mono">max_loras_per_batch</span> 上限和一个 LoRA 后端（默认 <span class="mono">triton</span>），建立起内存池。运营过程中，新技能通过 <span class="mono">load_lora_adapter</span> 进池、旧技能通过 <span class="mono">unload_lora_adapter</span> 出池。每一步前向，调度器先调用 <span class="mono">prepare_lora_batch</span> 为这一批摆好各请求的适配器，再让基座按常规跑前向、在 LoRA 层用分组 GEMM 叠加各自 ΔW。整套机制让单一部署能像换批头一样，灵活地服务几十种任务专用适配器。</p>
<p>需要区分的是：多 LoRA 换的是<strong>小小的增量</strong>，基座始终不动；而后面 <strong>第51课</strong> 讲的 RL 训练里也会"换权重"，但换的是<span class="mono">整个</span>模型的权重，量级与目的都完全不同。</p>
<p>把视角拉回到一次真实请求：客户端发来一段提示词，并在参数里写明"我要用 <span class="mono">sql-expert</span> 这个适配器"。请求进入调度器排队、与其他请求一起被组成连续批（第5课）。前向开始前，<span class="mono">prepare_lora_batch</span> 扫一遍这批里每个请求的适配器名，发现里面混着 <span class="mono">sql-expert</span>、<span class="mono">med-qa</span>、还有几条不带适配器的纯基座请求——它把不带适配器的当作"零增量"统一处理，把带适配器的按名分段。前向跑到注入了 LoRA 的那些线性层时，先算基座的 <span class="mono">W·x</span>，再用分组 GEMM 把每段各自的 <span class="mono">B·A·x</span> 加上去，得到 <span class="mono">(W + B·A)·x</span>。从采样、KV 缓存到输出，后面的一切都和普通文本推理没有区别。</p>
<p>值得强调的是，这条路径里"用哪个适配器"完全由请求自己携带的参数决定，而不是由部署时写死的全局开关决定。这意味着同一个端点可以同时面向不同租户、不同任务：A 团队的请求带上他们训练的法律适配器，B 团队的请求带上客服适配器，二者在同一批里被并行服务，却各自得到正确的结果。这种"按请求选技能"的能力，正是多 LoRA 在多租户场景下极具性价比的根源。</p>
<p>正因为适配器只是"叠加在基座之上的一层薄增量"，多 LoRA 与 SGLang 其余的机制天然兼容：连续批处理照常攒批、分页 KV 照常按页存取、前缀复用照常命中。LoRA 只在少数被注入的权重层上多做一次分组加法，其余引擎对"这条请求到底用了哪个适配器"几乎无感。这种"低侵入"正是它能在一份部署里廉价托管几十个微调版本的根本原因。</p>

<h2>后端、批与正确性</h2>
<p><span class="mono">LoRAManager</span> 在构造时会接收一个 <span class="mono">lora_backend</span> 参数（默认 <span class="mono">triton</span>），它决定那次分组 GEMM 用什么内核实现来落地。不同后端在性能上各有取舍，但对外暴露的语义是一致的：给定一批请求各自的适配器，输出必须等价于"逐请求单独套用它的 ΔW"。换句话说，分组只是<strong>性能优化的布局技巧</strong>，绝不能改变每个请求的数值结果——这是 <span class="mono">prepare_lora_batch</span> 必须保证的正确性底线。</p>
<p>为了做到这一点，摆放阶段要把三件事对齐：每个请求<strong>属于哪一段</strong>（即用哪个适配器）、每一段在打包缓冲里的<strong>起止偏移</strong>、以及每个适配器权重在 LoRA 内存池里的<strong>实际地址</strong>。这三者一旦对齐，内核就能在遍历每一行时，按它所属段去索引正确的 <span class="mono">A、B</span> 矩阵，算出专属的增量。任何一处错位，都会让某个请求悄悄用上别人的适配器——这类 bug 难查，因为模型不会报错，只会"答得有点不对味"。</p>
<p>还有一个常被忽略的细节：同一批里完全可以混入"不带任何适配器"的纯基座请求。实现上通常把它们当成一个特殊的"零号适配器"或直接跳过增量那一步，从而让带与不带适配器的请求在同一次前向里和平共处。这种灵活性意味着你不必把流量按适配器硬性分流，而可以让调度器自由地把任意组合凑成一批，把 GPU 利用率推到最高。</p>
<p>从容量规划的角度看，<span class="mono">max_loras_per_batch</span> 与批大小、适配器秩 <span class="mono">r</span> 共同决定了 LoRA 内存池的峰值占用。秩越大、同批适配器越多，池子就越吃显存，留给 KV 缓存（第30课）的空间就越少。因此在实际部署里，这几个旋钮往往要联合调优：既要让常用适配器都能进得了同一批以摊薄开销，又不能挤占了 KV 缓存导致可并发的请求数下降。</p>
<p>把这一课收束成一句话：多 LoRA 不是"把模型变多"，而是"把差异变薄"。基座是那台贵重而唯一的主机，适配器是一盒按需轮换的廉价批头，而 <span class="mono">prepare_lora_batch</span> 与分组 GEMM 则是让"同一批里各用各的批头"能在一趟动作里完成的那套机械臂逻辑。理解了这三层——薄增量、适配器池、批次感知的应用——你就抓住了 SGLang 多 LoRA 的全部要义。</p>

<div class="flow"><div class="node">基座输出<br/>W·x</div><div class="arrow">→</div><div class="node">本请求适配器<br/>ΔW = B·A</div><div class="arrow">→</div><div class="node">叠加<br/>(W + B·A)·x</div><div class="arrow">→</div><div class="node">该请求的<br/>适配后输出</div></div>

<div class="cellgroup"><div class="cell">行1 → 适配器 A</div><div class="cell">行2 → 适配器 B</div><div class="cell">行3 → 适配器 A</div><div class="cell">行4 → 适配器 C</div></div>

<table class="t"><tr><th>LoRAManager 方法</th><th>作用</th></tr><tr><td><span class="mono">load_lora_adapter</span></td><td>运行时往内存池里加入一个适配器</td></tr><tr><td><span class="mono">unload_lora_adapter</span></td><td>从池中移除一个适配器以释放空间</td></tr><tr><td><span class="mono">prepare_lora_batch</span></td><td>为当前批次逐请求收集所需适配器并摆放，供一次分组 GEMM 使用</td></tr><tr><td><span class="mono">max_loras_per_batch</span></td><td>限定同一批次内可共存的不同适配器数量（池/显存上限）</td></tr></table>

<div class="cols"><div class="col"><strong>笨办法：N 份完整模型</strong><br/>每个微调各起一份完整权重，显存被几乎相同的权重重复占满，N 越大越浪费。</div><div class="col"><strong>SGLang：1 份基座 + 适配器池</strong><br/>显存只放一份基座，外加一池小适配器；按请求叠加 ΔW，一份部署服务几十种技能。</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/lora/lora_manager.py ::LoRAManager</span><span class="ln">一个底座 + 适配器池：同批不同请求各用各的 LoRA</span></div><pre>class LoRAManager:
    def __init__(self, base_model, ..., max_loras_per_batch, lora_backend="triton", ...):
        # one base model + a memory pool holding up to max_loras_per_batch adapters
        ...
    def load_lora_adapter(self, lora_ref):     # add an adapter to the pool at runtime
        ...
    def unload_lora_adapter(self, lora_ref):   # drop one to free pool space
        ...
    def prepare_lora_batch(self, forward_batch):
        # for THIS batch: gather which adapter each request needs and stage them so a
        # segmented/grouped GEMM applies each request's own delta-W in one kernel
        ...</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li>LoRA 适配器是一个<strong>低秩小增量</strong>：学两个小矩阵 <span class="mono">A、B</span>，让有效权重为 <span class="mono">W + B·A</span>，秩 <span class="mono">r ≪ d</span>；一份基座可挂许多个，每个是一种微调"技能"，SGLang 记为 <span class="mono">LoRAAdapter</span>。</li>
<li>服务 N 个微调不必起 N 份完整模型：SGLang 保留<strong>一份基座 + 一个适配器池</strong>，按请求应用对应适配器。</li>
<li>真正的难点是<strong>批处理</strong>：同批请求可能各要各的适配器，<span class="mono">prepare_lora_batch(forward_batch)</span> 逐请求收集并摆放，使<strong>分段/分组 GEMM</strong> 能在一次内核里应用每个请求各自的 ΔW。</li>
<li><span class="mono">max_loras_per_batch</span> 限定同批可共存的不同适配器数（池/显存约束）；适配器可经 <span class="mono">load_lora_adapter</span> / <span class="mono">unload_lora_adapter</span> 在运行时增删，<strong>无需重启</strong>。</li>
<li>关联：<strong>第24课</strong> ForwardBatch、<strong>第25课</strong> 基座权重、<strong>第5课</strong> 连续批处理；前瞻 <strong>第51课</strong>（RL 也换权重，但换的是<span class="mono">整个</span>模型）。</li>
</ul>
</div>
""", "en": r"""
<p class="lead">One base model, dozens of "skills." <strong>Multi-LoRA batching</strong> lets you keep just <span class="mono">one</span> copy of the base weights in memory, plus a tiny <strong>adapter pool</strong>, yet serve many models each fine-tuned for a different task. The hard part is not "loading an adapter" — it is, when different requests in the <span class="mono">same</span> batch each want a different adapter, computing every request's own weight delta ΔW correctly in <span class="mono">one</span> kernel launch.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Think of the base model as the <strong>body of a power tool</strong>: the motor, handle, and battery live in the body — heavy and expensive. A LoRA adapter is like a <strong>swappable bit</strong> — each bit is small and cheap, yet it turns the same body into a "drill," a "screwdriver," or a "polisher."</p>
<p>The dumb old way: need three functions, buy three whole tool bodies (three full models) — the warehouse (memory) fills up instantly. The smart way: <strong>one body + a box of bits</strong>. A job comes in, see which bit it needs, <span class="mono">snap it on</span>, and go. Even better, several jobs arrive at once on the line: job 1 wants drilling, job 2 wants a screw, job 3 wants drilling again — you don't disassemble the body three times; you <strong>stage</strong> all those bits at once and let the arm finish every piece with its proper bit in a single pass. That "one pass handling a batch of different bits" is exactly what multi-LoRA batching solves.</p>
<p>The analogy stretches a little further: the bit box has finite capacity — you cannot keep every bit in the world plugged into the arm at once. So you keep on hand only the few bits "most likely to be used this round," return the unneeded ones to the box, and fetch the newly needed — which maps exactly to the runtime "load/unload adapter," and to the cap on "how many bits at most per round." The essence of the whole scheme is always the same sentence: <strong>keep just one expensive body, rotate the cheap bits on demand</strong>.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>LoRA's core math is plain: instead of retraining the huge weight matrix <span class="mono">W</span>, you learn two <strong>small matrices</strong> <span class="mono">A</span> and <span class="mono">B</span> so the effective weight becomes <span class="mono">W + B·A</span>. The rank <span class="mono">r</span> is far smaller than the matrix dimension (<span class="mono">r ≪ d</span>), so the delta ΔW = <span class="mono">B·A</span> is small and cheap. One base can carry <strong>many</strong> such adapters, each a different fine-tuned "skill." SGLang represents a single one as a <span class="mono">LoRAAdapter</span>.</p>
<p>So the real claim of multi-LoRA is: <strong>one base + an adapter pool + batch-aware ΔW application</strong>. A single deployment can cheaply serve dozens of task-specific adapters, without spinning up a full model per fine-tune.</p>
<p>Why is <span class="mono">B·A</span> so cheap? Let the weight dimension be <span class="mono">d×d</span>; full fine-tuning updates about <span class="mono">d²</span> parameters, whereas LoRA learns only <span class="mono">A</span> (about <span class="mono">r×d</span>) and <span class="mono">B</span> (about <span class="mono">d×r</span>), roughly <span class="mono">2·r·d</span> parameters in total. When the rank <span class="mono">r</span> is only a few dozen while the dimension <span class="mono">d</span> is a few thousand, the parameter count differs by two or three orders of magnitude. That means <strong>training cost</strong>, <strong>storage cost</strong>, and <strong>memory footprint</strong> all plunge — an adapter is small enough that dozens fit into the spare room of a single card, which is precisely what makes the "adapter pool" practical in engineering terms.</p>
</div>

<h2>Why "one base + an adapter pool"</h2>
<p>Suppose you have N fine-tunes: one writes SQL, one does medical Q&amp;A, one summarizes legal text. The most literal deployment is N full models — but each occupies full memory while <strong>99% of their weights are identical</strong> (all derived from the same base). That is enormous waste.</p>
<p>SGLang's approach: keep just <span class="mono">one</span> base model in memory (see <strong>Lesson 25</strong> on loading base weights), plus a <strong>LoRA memory pool</strong> holding several small adapters. For each request, take the adapter name it specifies and temporarily add the corresponding <span class="mono">B·A</span> delta onto the base output. Adapters are small, so the pool can hold many; the real constraint is <span class="mono">max_loras_per_batch</span> — it caps how many distinct adapters can coexist in <strong>one batch</strong> (bounded by pool/memory size).</p>
<p>Better yet, adapters can be added or removed at <strong>runtime</strong>: <span class="mono">load_lora_adapter</span> adds a new skill to the pool, <span class="mono">unload_lora_adapter</span> drops an unused one to free space — all with <strong>no restart</strong>.</p>
<p>This "hot-swap" is very practical in production. Say you ship a new customer-service-tone adapter: just load it into the pool and new requests can immediately specify it while old requests are untouched; when some adapter has gone unused for a while and is about to overflow the pool, just unload it and free the space for hotter skills. The whole process has no downtime window and needs no reloading of tens of GB of base weights — because the base is that one copy from start to finish, and what actually comes and goes is only the small delta of a few to tens of megabytes.</p>

<h2>Batching is the real hard part</h2>
<p>If every request in a batch used the same adapter, it would be easy: compute ΔW once and add it to the whole batch. But in reality requests in one batch often <strong>each want a different adapter</strong>: req1→adapterA, req2→adapterB, req3→adapterA, req4→adapterC… If you loop over adapters one by one, you degrade into many small GEMMs and shred the throughput that continuous batching (<strong>Lesson 5</strong>) worked so hard to accumulate.</p>
<p>SGLang's key function is <span class="mono">LoRAManager.prepare_lora_batch(forward_batch)</span>. It takes the current <span class="mono">ForwardBatch</span> (<strong>Lesson 24</strong>), looks <strong>per request</strong> at which adapter each row needs, and <strong>stages and segments</strong> those adapter weights in memory so that a single <strong>segmented / grouped GEMM</strong> can apply <strong>each request's own</strong> ΔW in <span class="mono">one</span> kernel launch, rather than running a separate pass per adapter.</p>
<p>That is what "batch-aware" means: not ignoring adapter differences and computing wrong, nor paying a loop's cost for them, but <strong>encoding the differences into the layout of one grouped matrix multiply</strong> so the GPU finishes in a single pass.</p>
<p>It helps to put "per-adapter loop" and "grouped GEMM" side by side. With a per-adapter loop, suppose this batch has 5 distinct adapters: you launch 5 small kernels, each carrying a kernel-launch overhead and a memory round-trip, while each small kernel processes very few rows, so the GPU's compute utilization is low — exactly the "fragmentation" continuous batching most wants to avoid. A grouped GEMM instead stitches these 5 segments into <strong>one</strong> call: the launch overhead is amortized over the whole batch, memory is traversed once, and the GPU's many compute units are fed at once. The more requests and the more varied the adapters, the more this "done in one pass" wins over "looping N times."</p>
<p>Concretely, the idea of a grouped GEMM is: slice the large batch by "which adapter" into several <strong>segments</strong>, each segment corresponding to one adapter and containing all request rows in the batch that use it. These segments are packed side by side as the input of a single kernel call, and inside the kernel each segment applies its own <span class="mono">B, A</span> matrices. So from the outside there is still just <strong>one</strong> kernel launch and one memory round-trip; from the inside, every request gets exactly the ΔW it should. This is precisely the "staging" work <span class="mono">prepare_lora_batch</span> quietly does before the forward — computing each segment's offset, length, and adapter index, turning the scattered "who wants which" into the tidy layout GPUs love.</p>
<p>This also shows why <span class="mono">max_loras_per_batch</span> matters: more segments means more distinct adapters active in this batch, which means more weights must reside in the pool and participate in this grouped GEMM. Set it too large and the LoRA memory pool gets tight; set it too small and the scheduler may be forced to defer a request wanting the "(K+1)-th" adapter to the next batch. It is essentially a line drawn between <strong>adapter diversity</strong> and <strong>memory budget</strong>.</p>

<h2>The runtime lifecycle</h2>
<p>Stringing the whole path together: at startup <span class="mono">LoRAManager</span> receives the base model, the <span class="mono">max_loras_per_batch</span> cap, and a LoRA backend (default <span class="mono">triton</span>), and builds the memory pool. During operation, new skills enter the pool via <span class="mono">load_lora_adapter</span> and old ones leave via <span class="mono">unload_lora_adapter</span>. On each forward step the scheduler first calls <span class="mono">prepare_lora_batch</span> to stage every request's adapter for this batch, then lets the base run its normal forward and add each ΔW in the LoRA layers with a grouped GEMM. This whole mechanism lets a single deployment flexibly serve dozens of task-specific adapters, like swapping bits.</p>
<p>One distinction worth drawing: multi-LoRA swaps a <strong>tiny delta</strong> while the base never moves; whereas the RL training in <strong>Lesson 51</strong> also "swaps weights," but it swaps the <span class="mono">whole</span> model's weights — a different scale and purpose entirely.</p>
<p>Pull the view back to one real request: a client sends a prompt and states in the params "I want the <span class="mono">sql-expert</span> adapter." The request enters the scheduler's queue and is grouped with others into a continuous batch (Lesson 5). Before the forward begins, <span class="mono">prepare_lora_batch</span> scans every request's adapter name in this batch and finds a mix of <span class="mono">sql-expert</span>, <span class="mono">med-qa</span>, and a few pure-base requests carrying no adapter — it treats the adapter-less ones as a "zero delta" handled uniformly and segments the rest by name. When the forward reaches the LoRA-injected linear layers, it first computes the base <span class="mono">W·x</span>, then uses a grouped GEMM to add each segment's own <span class="mono">B·A·x</span>, yielding <span class="mono">(W + B·A)·x</span>. From sampling to KV cache to output, everything afterward is no different from ordinary text inference.</p>
<p>Worth stressing: along this path "which adapter to use" is decided entirely by the parameters the request itself carries, not by a global switch hard-coded at deploy time. This means one endpoint can simultaneously serve different tenants and tasks: team A's requests carry their trained legal adapter, team B's carry a customer-service adapter, and the two are served in parallel within the same batch yet each gets the correct result. This "pick the skill per request" capability is exactly why multi-LoRA is so cost-effective in multi-tenant scenarios.</p>
<p>Precisely because an adapter is just "a thin delta layered on top of the base," multi-LoRA is naturally compatible with the rest of SGLang: continuous batching still accumulates batches, paged KV still pages, prefix reuse still hits. LoRA only does one extra grouped addition on the few injected weight layers, and the rest of the engine is almost oblivious to "which adapter this request actually used." This low-intrusiveness is the root reason it can cheaply host dozens of fine-tunes in a single deployment.</p>

<h2>Backend, batch, and correctness</h2>
<p>When constructed, <span class="mono">LoRAManager</span> receives a <span class="mono">lora_backend</span> argument (default <span class="mono">triton</span>) that decides which kernel implementation realizes that grouped GEMM. Different backends make different performance trade-offs, but the semantics they expose are identical: given a batch's per-request adapters, the output must be equivalent to "applying each request's ΔW individually." In other words, grouping is merely a <strong>performance layout trick</strong> and must never change any request's numeric result — that is the correctness floor <span class="mono">prepare_lora_batch</span> has to guarantee.</p>
<p>To achieve this, the staging phase must align three things: which <strong>segment</strong> each request belongs to (i.e., which adapter), each segment's <strong>start/end offset</strong> in the packed buffer, and each adapter weight's <strong>actual address</strong> in the LoRA memory pool. Once these align, the kernel can, as it walks each row, index the correct <span class="mono">A, B</span> matrices by the row's segment and compute its dedicated delta. Any misalignment lets some request silently use another's adapter — a hard bug to find, because the model does not error, it just "answers a little off."</p>
<p>One often-overlooked detail: a batch may well mix in pure-base requests carrying no adapter. Implementations usually treat them as a special "zero adapter" or simply skip the delta step, letting requests with and without adapters coexist peacefully in one forward. This flexibility means you need not hard-route traffic by adapter; the scheduler can freely assemble any combination into a batch and push GPU utilization to the max.</p>
<p>From a capacity-planning view, <span class="mono">max_loras_per_batch</span>, the batch size, and the adapter rank <span class="mono">r</span> together determine the LoRA memory pool's peak footprint. A larger rank and more adapters per batch make the pool eat more VRAM, leaving less for the KV cache (Lesson 30). So in real deployments these knobs are tuned jointly: let common adapters all fit into one batch to amortize overhead, yet not crowd out the KV cache and reduce achievable concurrency.</p>
<p>To boil this lesson down to one sentence: multi-LoRA is not "making more models," it is "making the difference thin." The base is that one precious, singular tool body; the adapters are a box of cheap bits rotated on demand; and <span class="mono">prepare_lora_batch</span> plus the grouped GEMM are the robotic-arm logic that lets "each row in one batch use its own bit" finish in a single pass. Grasp these three layers — thin delta, adapter pool, batch-aware application — and you have the whole essence of SGLang's multi-LoRA.</p>

<div class="flow"><div class="node">Base output<br/>W·x</div><div class="arrow">→</div><div class="node">This request's adapter<br/>ΔW = B·A</div><div class="arrow">→</div><div class="node">Add<br/>(W + B·A)·x</div><div class="arrow">→</div><div class="node">This request's<br/>adapted output</div></div>

<div class="cellgroup"><div class="cell">row 1 → adapter A</div><div class="cell">row 2 → adapter B</div><div class="cell">row 3 → adapter A</div><div class="cell">row 4 → adapter C</div></div>

<table class="t"><tr><th>LoRAManager method</th><th>Role</th></tr><tr><td><span class="mono">load_lora_adapter</span></td><td>Add an adapter to the memory pool at runtime</td></tr><tr><td><span class="mono">unload_lora_adapter</span></td><td>Remove an adapter from the pool to free space</td></tr><tr><td><span class="mono">prepare_lora_batch</span></td><td>For the current batch, gather each request's needed adapter and stage them for one grouped GEMM</td></tr><tr><td><span class="mono">max_loras_per_batch</span></td><td>Cap on distinct adapters that can coexist in one batch (pool/memory bound)</td></tr></table>

<div class="cols"><div class="col"><strong>Naive: N full models</strong><br/>Each fine-tune runs a full set of weights; memory fills with nearly identical weights repeated — the larger N, the more waste.</div><div class="col"><strong>SGLang: 1 base + adapter pool</strong><br/>Memory holds one base plus a pool of small adapters; add ΔW per request, one deployment serving dozens of skills.</div></div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/lora/lora_manager.py ::LoRAManager</span><span class="ln">one base + an adapter pool: different requests in one batch use different LoRAs</span></div><pre>class LoRAManager:
    def __init__(self, base_model, ..., max_loras_per_batch, lora_backend="triton", ...):
        # one base model + a memory pool holding up to max_loras_per_batch adapters
        ...
    def load_lora_adapter(self, lora_ref):     # add an adapter to the pool at runtime
        ...
    def unload_lora_adapter(self, lora_ref):   # drop one to free pool space
        ...
    def prepare_lora_batch(self, forward_batch):
        # for THIS batch: gather which adapter each request needs and stage them so a
        # segmented/grouped GEMM applies each request's own delta-W in one kernel
        ...</pre></div>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li>A LoRA adapter is a <strong>small low-rank delta</strong>: learn two small matrices <span class="mono">A, B</span> so the effective weight is <span class="mono">W + B·A</span> with rank <span class="mono">r ≪ d</span>; one base carries many, each a fine-tuned "skill," represented in SGLang as <span class="mono">LoRAAdapter</span>.</li>
<li>Serving N fine-tunes need not mean N full models: SGLang keeps <strong>one base + an adapter pool</strong> and applies the right adapter per request.</li>
<li>The real hard part is <strong>batching</strong>: requests in one batch may each want a different adapter; <span class="mono">prepare_lora_batch(forward_batch)</span> gathers and stages them per request so a <strong>segmented/grouped GEMM</strong> applies each request's own ΔW in one kernel.</li>
<li><span class="mono">max_loras_per_batch</span> caps distinct adapters coexisting in a batch (pool/memory bound); adapters are added/removed at runtime via <span class="mono">load_lora_adapter</span> / <span class="mono">unload_lora_adapter</span>, with <strong>no restart</strong>.</li>
<li>Ties: <strong>Lesson 24</strong> ForwardBatch, <strong>Lesson 25</strong> base weights, <strong>Lesson 5</strong> continuous batching; forward-ref <strong>Lesson 51</strong> (RL also swaps weights, but the <span class="mono">whole</span> model).</li>
</ul>
</div>
"""}
LESSON_51 = {"zh": r"""
<p class="lead">在强化学习（RL / RLHF）后训练里，SGLang 不只是一个推理服务器，它还要充当 <strong>rollout 引擎</strong>：每一步训练都要用<strong>刚刚更新过的权重</strong>重新生成样本。本课讲清楚这个"生成→打分→更新权重→再生成"的闭环，以及为什么必须用<span class="mono">update_weights_from_tensor</span> 做<strong>原地权重热更新</strong>，而不能每步重启服务器。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象一支正在备战的<strong>辩论队</strong>：选手先上台<strong>即兴发言</strong>（这就是 rollout，批量"生成"很多回答），教练在台下<strong>打分并记笔记</strong>（reward + 梯度），然后把改进意见<strong>当场塞进选手脑子里</strong>，选手不用回家睡一觉重新背稿，立刻带着新本事再上台。</p>
<p>如果每次教练给完意见，选手都得<strong>卸妆、回宿舍、重新化妆、重新登台</strong>（重启服务器、重载 checkpoint、重新预热、重新捕获 CUDA Graph），那一晚上光折腾就过去了，根本练不了几轮。SGLang 的原地权重更新，就是"<strong>当场塞进脑子</strong>"——选手（CUDA Graph、KV 缓存池）原地不动，只把脑子里的"参数"换掉。</p>
<p>再补一个细节：教练给的笔记如果是厚厚一沓、一张一张递给选手，光递纸就要半天；不如把所有笔记<strong>装订成一本册子一次性递过去</strong>，选手翻到对应页就能看——这正对应后面要讲的 <span class="mono">FlattenedTensorBucket</span>，把成千上万个小张量打包成一整块、一次搬完。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>RL 后训练（PPO、GRPO 等）每一轮迭代分两半：<strong>策略模型生成大量样本</strong>（rollout，正是 SGLang 擅长的高吞吐批量生成），然后<strong>训练器给样本打分</strong>（reward）并算出梯度去<strong>更新策略权重</strong>。关键约束是：<strong>下一轮 rollout 必须用刚更新的权重跑</strong>，所以 rollout 引擎的权重每个训练步都在变。</p>
<p>SGLang 把"服务器"和"RL rollout worker"做成<strong>同一个进程</strong>，只多挂一个权重同步钩子：本地同卡用 <span class="mono">update_weights_from_tensor</span>，跨卡训练器用 <span class="mono">update_weights_from_distributed</span> 通过进程组（<strong>第46课</strong> GroupCoordinator）把权重<strong>流式</strong>传进来。这就是 SGLang 成为众多 RL 框架首选 rollout 后端的原因。</p>
<p>记住这条主线：<strong>生成靠 SGLang、训练靠训练器、衔接靠权重同步</strong>。本课要回答三个问题——闭环为什么必须同步权重、为什么不能靠重启来同步、以及怎样把同步做得足够快（原地写入 + 打包搬运）。把这三点想透，你就理解了 RL 训练里推理引擎扮演的真实角色。</p>
</div>

<h2>一、RL 的闭环：rollout 与训练交替</h2>
<p>把一轮 RL 迭代拆开看，它其实是<strong>推理</strong>和<strong>训练</strong>两个阶段在不停交替。第一阶段，策略模型对一批 prompt <strong>批量生成</strong>很多条回答——这正是 SGLang 的主场，连续批处理、RadixAttention、CUDA Graph 全部派上用场，把吞吐打满。这一步生成出的就是 RL 里要"评价"的轨迹（trajectories）。在 PPO 这类算法里，这些轨迹不仅要算 reward，还要记录每一步的对数概率，供后面计算重要性采样比和优势函数使用；而在 GRPO 里，则会对同一个 prompt 采样一组回答、用组内相对得分来估计优势。无论哪种，<strong>rollout 阶段产出的样本数量都很大</strong>，所以这一步的吞吐直接决定了训练数据的产出速度，这也是为什么大家愿意把 SGLang 这样的高性能推理引擎请进训练循环。</p>
<p>第二阶段，<strong>训练器</strong>登场：用 reward 模型或规则给每条样本<strong>打分</strong>，按 PPO / GRPO 的目标函数算出<strong>梯度</strong>，对策略网络做一次<strong>参数更新</strong>。更新完，新的权重就是下一轮 rollout 要用的权重。注意这里的<strong>耦合</strong>：训练器算出的新权重，必须及时"喂回"给正在跑生成的 SGLang 引擎，否则下一轮还在用旧策略采样，训练就跑偏了。这种"采样用的策略必须紧跟被优化的策略"的要求，正是 on-policy 类算法的核心约束，也是为什么权重同步在 RL 训练里如此关键。如果同步滞后哪怕一两步，采样分布与目标分布之间就会出现偏差，轻则拖慢收敛，重则让训练彻底发散，因此"及时、正确"这两条缺一不可。</p>
<p>于是每个训练步，rollout 引擎的权重<strong>都要换一次</strong>。换权重这件事如果做得慢，整个 RL 训练的墙钟时间就会被它拖死。设想一个典型规模：一轮 rollout 几秒钟就能批量生成完，训练器的前向反向也只要几秒，但如果"换权重"要花几十秒，那么权重同步就成了整条流水线上最大的瓶颈，GPU 大半时间都在空等。这就引出了本课的核心：<strong>怎么又快又对地把新权重塞进运行中的引擎</strong>，让换权重的耗时小到可以忽略。</p>

<div class="vflow">
<div class="step"><div class="num">1</div><div class="sc"><h4>Rollout（生成）</h4><p>策略模型对一批 prompt 批量采样出大量回答 —— SGLang 高吞吐生成</p></div></div>
<div class="step"><div class="num">2</div><div class="sc"><h4>Reward + 梯度（训练器）</h4><p>给样本打分，按 PPO/GRPO 目标算梯度</p></div></div>
<div class="step"><div class="num">3</div><div class="sc"><h4>update_weights（原地热更新）</h4><p>把刚训好的权重写进运行中的模型参数</p></div></div>
<div class="step"><div class="num">4</div><div class="sc"><h4>下一轮 Rollout</h4><p>用刚更新的权重重新生成 → 回到 ①，闭环不断滚动</p></div></div>
</div>

<h2>二、为什么不能每步重启服务器</h2>
<p>最朴素的想法是：训练器存一个新的 checkpoint，SGLang 服务器<strong>重启</strong>一下、把新 checkpoint 重新加载进来不就行了？问题在于一次"冷启动"代价极大：要<strong>重新读权重文件</strong>（第25课的权重加载流程）、<strong>重新预热</strong>、<strong>重新捕获 CUDA Graph</strong>、重建 KV 缓存池。这些一次性开销动辄几十秒甚至上分钟，而 RL 训练动辄成千上万步——每步都重启，光启动时间就会<strong>压倒</strong>真正的计算时间，完全不可接受。这里尤其要强调 CUDA Graph 的重新捕获：它需要把整个前向过程录制一遍，本身就要跑若干次预热前向，代价不菲；而 KV 缓存池则要按显存大小重新规划、重新分配，这些都是只有"启动时"才做一次的重活，决不该被塞进训练内循环里反复执行。</p>
<p>SGLang 的答案是<strong>原地权重更新（in-place weight update）</strong>：进程不重启，CUDA Graph 不重新捕获，KV 池原地保留，只把<strong>模型参数这块显存</strong>里的数值换成新训练好的张量。<span class="mono">update_weights_from_tensor(named_tensors, load_format=…)</span> 直接把新张量写进运行中模型的对应参数上，毫秒级完成，下一轮 rollout 立刻可用。这里的关键洞察是：模型的"骨架"——计算图的结构、算子的排布、KV 池的内存布局——在训练全程<strong>从不改变</strong>，真正变的只有参数张量里的<strong>数值</strong>。既然结构不变，就完全没必要把昂贵的结构性初始化（图捕获、内存规划）重做一遍，只需把数值就地刷新即可。这是一种典型的"只改该改的、别动不该动的"工程取舍。</p>
<p>这正是 SGLang 把<strong>同一个进程</strong>同时当"serving 服务器"和"RL rollout worker"用的底气：两者代码几乎一模一样，差别只是 rollout 模式多挂了一个<strong>权重同步钩子</strong>。同样的"换权重"思路在第50课也见过——LoRA 也是在换权重，只不过它换的是<strong>很小的 adapter</strong>，而 RL 这里换的是<strong>整套主干权重</strong>。换句话说，serving 与 rollout 之间没有架构鸿沟，一个面向终端用户提供推理，一个面向训练器循环喂样本，复用的是同一套连续批处理、同一套 CUDA Graph、同一套 KV 管理，这种复用让 RL 框架几乎零成本地接入 SGLang 的全部推理优化。</p>

<div class="cols">
<div class="col"><strong>❌ 每步重启服务器</strong><br>重载 checkpoint → 重新预热 → 重新捕获 CUDA Graph → 重建 KV 池。一次性开销几十秒～分钟级，乘以上万训练步 = 启动时间压垮计算时间，<span class="mono">不可接受</span>。</div>
<div class="col"><strong>✅ 原地 update_weights</strong><br>进程不重启，CUDA Graph / KV 池原地不动，只把新张量写进模型参数显存。毫秒级完成，下一轮 rollout 立即用上新权重。</div>
</div>

<h2>三、三种权重同步入口：本地、跨卡、打包</h2>
<p>权重从训练器到 rollout 引擎，路径取决于它们的物理位置。如果训练器和推理在<strong>同一组 GPU</strong>上（共置，co-locate），新张量已经在显存里，直接用 <span class="mono">update_weights_from_tensor</span> 把 <span class="mono">named_tensors</span> 原地写进参数即可。如果训练器在<strong>另一批 GPU</strong>上（分离部署），就要用 <span class="mono">update_weights_from_distributed</span>，借助第46课的 <span class="mono">GroupCoordinator</span> 进程组，把权重从训练 rank <strong>流式</strong>传到推理 rank。所谓"流式"，是指不必等整个模型搬完才开始用，而是边传边收、按通信原语（broadcast 等）一层层把参数推过去，既省内存峰值又能与计算重叠。</p>
<p>但不管哪条路，都有一个共同的性能陷阱：一个大模型有<strong>成千上万个</strong>命名张量，如果一个一个地搬，每个张量都要单独发起一次拷贝/传输，<strong>启动与传输开销</strong>会被乘上几千倍，慢得离谱。这就是 <span class="mono">load_format="flattened_bucket"</span> 要解决的问题。可以这样体会：哪怕单个张量只要 1 微秒的发起开销，乘以上万个就是几十毫秒，而这还只是"发起"，真正的数据还没动；在 RL 里这笔账每个训练步都要重算一遍，日积月累就成了不可忽视的浪费。</p>
<p>所以选型很清晰：<strong>同卡共置</strong>用 from_tensor；<strong>跨卡分离</strong>用 from_distributed；<strong>张量太多想省开销</strong>就叠加 flattened_bucket 打包路径。三者并不互斥，flattened_bucket 是作为 <span class="mono">load_format</span> 选项叠加在前两者之上的。实际工程里，训练框架会根据自己的部署拓扑选择路径：把训练与推理塞进同一组卡能省去跨机通信，但要小心两者争抢显存；把训练器单独放一批卡上则更灵活、扩展性更好，代价是每步都要走进程组把权重流过去。无论哪种，目标都一致——让权重同步既正确又不拖慢训练节奏。</p>

<table class="t">
<tr><th>入口 / 方式</th><th>适用场景</th><th>怎么搬</th></tr>
<tr><td><span class="mono">update_weights_from_tensor</span></td><td>训练器与推理<strong>同卡共置</strong>，新张量已在本地显存</td><td>直接把 named_tensors 原地写进运行中模型的参数</td></tr>
<tr><td><span class="mono">update_weights_from_distributed</span></td><td>训练器在<strong>另一批 GPU</strong>，分离部署</td><td>经第46课进程组把权重从训练 rank <strong>流式</strong>传进推理 rank</td></tr>
<tr><td><span class="mono">load_format="flattened_bucket"</span></td><td>张量数量庞大，想把<strong>逐张量开销</strong>降到最低</td><td>多张量打包成一块连续 buffer，<strong>整体一次</strong>传输/拷贝</td></tr>
</table>

<h2>四、FlattenedTensorBucket：把千张量压成一次拷贝</h2>
<p>逐个搬张量为什么慢？因为每次拷贝/传输都有<strong>固定的发起开销</strong>（kernel launch、通信握手等），张量越多、这个开销被放大得越厉害，哪怕每个张量本身很小。一个数十亿参数的大模型，命名张量动辄成千上万个，逐个搬就意味着成千上万次内核启动或通信握手，固定开销累加起来甚至远超真正传数据的时间。<span class="mono">FlattenedTensorBucket</span> 的思路是把<strong>很多命名张量先拼（flatten）成一整块连续 buffer</strong>，<strong>只传输/拷贝一次</strong>，到了引擎侧再按记录的形状/偏移把<strong>各个张量重建</strong>出来，逐张量的开销因此被摊薄到几乎为零。重建时之所以能严丝合缝地还原，是因为打包时已经把每个张量的名字、形状、数据类型和在大 buffer 里的偏移都记录在案，到岸侧照着这份"清单"切片即可。值得一提的是，重建出来的张量与原始 buffer 往往<strong>共享同一块底层显存</strong>（视图切片而非再拷贝），所以"重建"这一步几乎不花额外时间，真正的成本只在那一次整体搬运上。</p>
<p>这种"打包搬运 + 到岸重建"是分布式权重同步里非常实用的优化，尤其在 RL 这种<strong>每步都要全量同步权重</strong>的场景，省下的开销直接转化为更高的训练吞吐。打个比方：逐张量搬运像是把一仓库货物一件一件分别开车送，每趟车都要起步、上路、停车；而打包搬运则是把货物先码进一个集装箱，一趟整柜运过去，到了再拆箱归位——固定成本只付一次。把它和前面的拼图连起来：第13/14课是引擎入口，第25课是权重加载，第46课是进程组通信，第50课是 LoRA 的小幅换权重，本课则是 RL 全量换权重——它们共享同一套底层机制。理解了这条主线，就能看清 SGLang 不只是"快的推理库"，更是一个能被训练循环无缝复用的执行引擎：同一份代码、同一套优化，既服务在线用户，也服务训练器，差别只在那个轻巧的权重同步钩子。这种"一鱼两吃"的设计，正是它在 RL 生态里被广泛采用的根本原因。后续<strong>第59-63课</strong>会继续展开这些设计主题。</p>

<div class="flow">
<div class="node">成千上万个命名张量</div>
<div class="arrow">→</div>
<div class="node">FlattenedTensorBucket（拼成一块 buffer）</div>
<div class="arrow">→</div>
<div class="node">整体一次拷贝/传输</div>
<div class="arrow">→</div>
<div class="node">引擎侧重建 → 写进活的参数</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::update_weights_from_tensor</span><span class="ln">把新训练好的权重原地写进运行中的模型，免重启</span></div><pre>def update_weights_from_tensor(
    self,
    named_tensors,              # [(name, tensor)] 训练器刚训好的权重
    load_format=None,           # "flattened_bucket" 把许多张量打包进一块 buffer
    **kwargs,
):
    if load_format == "flattened_bucket":
        # FlattenedTensorBucket: 把许多命名张量拼成一块 buffer，
        # 整体只传输/拷贝一次，再在引擎侧重建各个张量（开销更低）
        ...
    # 把张量直接写进运行中模型的参数 —— 不重启，
    # CUDA Graph 与 KV 池原地保留
    ...
# update_weights_from_distributed(...) 改为经进程组流式传权重</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li>RL 后训练每轮迭代 = <strong>rollout 生成</strong>（SGLang 高吞吐采样）+ <strong>训练器打分算梯度更新权重</strong>，且下一轮 rollout 必须用<strong>刚更新的权重</strong>。</li>
<li>每步<strong>重启服务器</strong>（重载 checkpoint、重新预热、重捕 CUDA Graph）会让启动时间压垮计算，<strong>不可接受</strong>。</li>
<li><span class="mono">update_weights_from_tensor</span> 做<strong>原地热更新</strong>：CUDA Graph 与 KV 池不动，只把新张量写进模型参数。</li>
<li>训练器在别的 GPU 上时用 <span class="mono">update_weights_from_distributed</span>，经第46课<strong>进程组</strong>流式传权重。</li>
<li><span class="mono">load_format="flattened_bucket"</span> 用 <span class="mono">FlattenedTensorBucket</span> 把千张量<strong>打包成一块 buffer、整体一次搬运</strong>，再到岸重建，大幅降低逐张量开销。</li>
<li>同一个 SGLang 进程<strong>既是 serving 服务器、又是 RL rollout worker</strong>，只差这个权重同步钩子，这也是它成为热门 rollout 后端的原因。</li>
<li>一句话串起来：<strong>生成靠 SGLang、训练靠训练器、衔接靠权重同步</strong>；同步要又快又对，靠的就是原地写入加打包搬运这两手。</li>
</ul>
</div>
""", "en": r"""
<p class="lead">In reinforcement-learning (RL / RLHF) post-training, SGLang is not just an inference server — it doubles as the <strong>rollout engine</strong>: every training step has to regenerate samples with the <strong>weights that were just updated</strong>. This lesson lays out the "generate → score → update weights → generate again" loop, and why we must do an <strong>in-place weight hot-update</strong> via <span class="mono">update_weights_from_tensor</span> instead of restarting the server each step.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture a <strong>debate team</strong> in training camp: a contestant goes on stage and <strong>improvises an answer</strong> (that's the rollout — batched "generation" of many responses), the coach <strong>scores it and takes notes</strong> (reward + gradient), then <strong>pours the feedback straight into the contestant's head</strong> on the spot — no going home to sleep and re-memorize a script; they walk right back on stage with the new skill.</p>
<p>If every time the coach gave feedback the contestant had to <strong>remove makeup, go back to the dorm, re-do makeup, and re-enter</strong> (restart the server, reload the checkpoint, re-warm, re-capture CUDA graphs), the whole night would be gone to logistics and barely a round would be practiced. SGLang's in-place weight update is exactly "<strong>pouring it straight into the head</strong>" — the contestant (CUDA graphs, KV pools) stays put; only the "parameters" in the head get swapped.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>Each iteration of RL post-training (PPO, GRPO, etc.) has two halves: the <strong>policy model generates many samples</strong> (the rollout — exactly the fast batched generation SGLang excels at), then the <strong>trainer scores the samples</strong> (reward) and computes a gradient that <strong>updates the policy weights</strong>. The crux: the <strong>next rollout must run with the just-updated weights</strong>, so the rollout engine's weights change every training step.</p>
<p>SGLang makes the "server" and the "RL rollout worker" the <strong>same process</strong>, differing only by a weight-sync hook: same-GPU co-location uses <span class="mono">update_weights_from_tensor</span>, while a trainer on other GPUs uses <span class="mono">update_weights_from_distributed</span> to <strong>stream</strong> weights in over a process group (<strong>Lesson 46</strong>'s GroupCoordinator). This is why SGLang is a popular rollout backend for RL frameworks.</p>
</div>

<h2>1. The RL loop: rollout and training alternating</h2>
<p>Unpack one RL iteration and it is really <strong>inference</strong> and <strong>training</strong> phases endlessly taking turns. In the first phase, the policy model <strong>batch-generates</strong> many answers for a batch of prompts — SGLang's home turf, with continuous batching, RadixAttention and CUDA graphs all kicking in to max out throughput. What this step produces are the trajectories that RL will "evaluate".</p>
<p>In the second phase, the <strong>trainer</strong> steps in: it <strong>scores</strong> each sample with a reward model or rules, computes the <strong>gradient</strong> per the PPO / GRPO objective, and performs one <strong>parameter update</strong> on the policy network. Once updated, the new weights are the ones the next rollout must use. Note the <strong>coupling</strong>: the new weights the trainer computes must be promptly "fed back" into the SGLang engine that is running generation, otherwise the next rollout still samples from the old policy and training drifts.</p>
<p>So at every training step, the rollout engine's weights <strong>get swapped once</strong>. If swapping weights is slow, the wall-clock time of the entire RL run is dragged down by it. That leads straight to the heart of this lesson: <strong>how to push new weights into a running engine quickly and correctly</strong>.</p>

<div class="vflow">
<div class="step"><div class="num">1</div><div class="sc"><h4>Rollout (generate)</h4><p>the policy model batch-samples many answers for a batch of prompts — high-throughput SGLang generation</p></div></div>
<div class="step"><div class="num">2</div><div class="sc"><h4>Reward + gradient (trainer)</h4><p>score the samples, compute the gradient per the PPO/GRPO objective</p></div></div>
<div class="step"><div class="num">3</div><div class="sc"><h4>update_weights (in-place hot update)</h4><p>write the freshly-trained weights into the running model's parameters</p></div></div>
<div class="step"><div class="num">4</div><div class="sc"><h4>Next rollout</h4><p>regenerate with the just-updated weights → back to ①, the loop keeps rolling</p></div></div>
</div>

<h2>2. Why we can't restart the server every step</h2>
<p>The naïve idea: have the trainer save a new checkpoint and just <strong>restart</strong> the SGLang server to reload it. The problem is a "cold start" is hugely expensive: it must <strong>re-read the weight files</strong> (Lesson 25's weight-loading flow), <strong>re-warm</strong>, <strong>re-capture CUDA graphs</strong>, and rebuild the KV pools. These one-time costs are tens of seconds to minutes, while RL training runs thousands to tens of thousands of steps — restarting each step means startup time would <strong>dominate</strong> actual compute, which is unacceptable.</p>
<p>SGLang's answer is the <strong>in-place weight update</strong>: the process never restarts, CUDA graphs aren't re-captured, KV pools stay in place, and only the values in the <strong>model-parameter memory</strong> get swapped for the freshly-trained tensors. <span class="mono">update_weights_from_tensor(named_tensors, load_format=…)</span> writes the new tensors straight onto the running model's matching parameters in milliseconds, ready for the next rollout immediately.</p>
<p>This is exactly what lets SGLang use <strong>one process</strong> as both the "serving server" and the "RL rollout worker": the code is nearly identical, the only difference being that rollout mode bolts on a <strong>weight-sync hook</strong>. We saw the same "swap the weights" idea in Lesson 50 — LoRA also swaps weights, except it swaps a <strong>tiny adapter</strong>, whereas RL here swaps the <strong>full backbone weights</strong>.</p>

<div class="cols">
<div class="col"><strong>❌ Restart the server every step</strong><br>Reload checkpoint → re-warm → re-capture CUDA graphs → rebuild KV pools. One-time cost of tens of seconds to minutes, times tens of thousands of steps = startup time crushes compute, <span class="mono">unacceptable</span>.</div>
<div class="col"><strong>✅ In-place update_weights</strong><br>No restart, CUDA graphs / KV pools stay put, only the new tensors get written into the model-parameter memory. Done in milliseconds; the next rollout uses the new weights immediately.</div>
</div>

<h2>3. Three weight-sync entry points: local, cross-GPU, packed</h2>
<p>The path from trainer to rollout engine depends on their physical location. If trainer and inference live on the <strong>same set of GPUs</strong> (co-located), the new tensors are already in memory, so just use <span class="mono">update_weights_from_tensor</span> to write the <span class="mono">named_tensors</span> in place onto the parameters. If the trainer lives on <strong>other GPUs</strong> (disaggregated), use <span class="mono">update_weights_from_distributed</span>, which leverages Lesson 46's <span class="mono">GroupCoordinator</span> process group to <strong>stream</strong> the weights from the trainer ranks into the inference ranks.</p>
<p>But either way there is a shared performance trap: a large model has <strong>thousands</strong> of named tensors, and if you move them one at a time, each tensor triggers a separate copy/transfer, so the <strong>launch and transfer overhead</strong> gets multiplied thousands of times — absurdly slow. That is exactly what <span class="mono">load_format="flattened_bucket"</span> is there to solve.</p>
<p>So selection is clear: <strong>same-GPU co-location</strong> uses from_tensor; <strong>cross-GPU disaggregation</strong> uses from_distributed; <strong>too many tensors, want to cut overhead</strong> adds the flattened_bucket packing path. They are not mutually exclusive — flattened_bucket is a <span class="mono">load_format</span> option layered on top of the first two.</p>

<table class="t">
<tr><th>Entry / mode</th><th>When to use</th><th>How it moves</th></tr>
<tr><td><span class="mono">update_weights_from_tensor</span></td><td>Trainer and inference <strong>co-located on the same GPUs</strong>, new tensors already in local memory</td><td>Write named_tensors in place onto the running model's parameters</td></tr>
<tr><td><span class="mono">update_weights_from_distributed</span></td><td>Trainer on <strong>other GPUs</strong>, disaggregated deployment</td><td><strong>Stream</strong> weights from trainer ranks into inference ranks via Lesson 46's process group</td></tr>
<tr><td><span class="mono">load_format="flattened_bucket"</span></td><td>Huge number of tensors, want to minimize <strong>per-tensor overhead</strong></td><td>Pack many tensors into one contiguous buffer, transfer/copy <strong>once as a whole</strong></td></tr>
</table>

<h2>4. FlattenedTensorBucket: collapsing thousands of tensors into one copy</h2>
<p>Why is moving tensors one by one slow? Because every copy/transfer has a <strong>fixed launch cost</strong> (kernel launch, comm handshake, etc.), and the more tensors there are the more that cost is amplified, even if each tensor is tiny. <span class="mono">FlattenedTensorBucket</span>'s idea is to first <strong>flatten many named tensors into one contiguous buffer</strong>, <strong>transfer/copy it only once</strong>, then <strong>reconstruct the individual tensors</strong> on the engine side from the recorded shapes/offsets — so the per-tensor overhead is amortized down to almost nothing.</p>
<p>This "pack-and-ship + reconstruct-on-arrival" is a very practical optimization in distributed weight sync, especially in RL where <strong>full weights are synced every step</strong>; the overhead saved turns directly into higher training throughput. Tie the pieces together: Lessons 13/14 are the engine entrypoints, Lesson 25 is weight loading, Lesson 46 is process-group communication, Lesson 50 is LoRA's small weight swap, and this lesson is RL's full weight swap — they share the same underlying machinery. Lessons <strong>59-63</strong> will keep expanding these design themes.</p>

<div class="flow">
<div class="node">Thousands of named tensors</div>
<div class="arrow">→</div>
<div class="node">FlattenedTensorBucket (packed into one buffer)</div>
<div class="arrow">→</div>
<div class="node">One copy/transfer as a whole</div>
<div class="arrow">→</div>
<div class="node">Reconstruct on engine side → write into live params</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/srt/entrypoints/engine.py ::update_weights_from_tensor</span><span class="ln">write freshly-trained weights into the running model in place, no restart</span></div><pre>def update_weights_from_tensor(
    self,
    named_tensors,              # [(name, tensor)] freshly-trained weights from the trainer
    load_format=None,           # "flattened_bucket" packs many tensors into one buffer
    **kwargs,
):
    if load_format == "flattened_bucket":
        # FlattenedTensorBucket: flatten many named tensors into ONE buffer,
        # transfer/copy once, then reconstruct on the engine side (less overhead)
        ...
    # write the tensors straight into the running model's parameters — no restart,
    # CUDA graphs and KV pools stay in place
    ...
# update_weights_from_distributed(...) streams weights over a process group instead</pre></div>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li>Each RL post-training iteration = <strong>rollout generation</strong> (high-throughput SGLang sampling) + <strong>trainer scoring, gradient, weight update</strong>, and the next rollout must run with the <strong>just-updated weights</strong>.</li>
<li><strong>Restarting the server</strong> every step (reload checkpoint, re-warm, re-capture CUDA graphs) lets startup time crush compute — <strong>unacceptable</strong>.</li>
<li><span class="mono">update_weights_from_tensor</span> does an <strong>in-place hot update</strong>: CUDA graphs and KV pools stay put, only the new tensors are written into the model parameters.</li>
<li>When the trainer lives on other GPUs, <span class="mono">update_weights_from_distributed</span> <strong>streams</strong> the weights over Lesson 46's <strong>process group</strong>.</li>
<li><span class="mono">load_format="flattened_bucket"</span> uses <span class="mono">FlattenedTensorBucket</span> to <strong>pack thousands of tensors into one buffer and move them once</strong>, then reconstruct on arrival, slashing per-tensor overhead.</li>
<li>One SGLang process is <strong>both the serving server and the RL rollout worker</strong>, differing only by this weight-sync hook — which is why it's a popular rollout backend.</li>
<li>In one line: <strong>generation by SGLang, training by the trainer, glued by weight sync</strong>; making that sync fast and correct rests on in-place writes plus packed transfer.</li>
</ul>
</div>
"""}
LESSON_52 = {"zh": r"""
<p class="lead">这是第十一部分的最后一课，也是整份指南的一个漂亮收尾。我们要把目光从"大语言模型"挪开，看向另一类生成任务：<strong>扩散模型</strong>（diffusion models），也就是图像与视频生成。它住在 <span class="mono">sglang-diffusion</span> 子项目里，路径是 <span class="mono">python/sglang/multimodal_gen/</span>（注意是在 <span class="mono">python/sglang/</span> 之下，而不是 <span class="mono">python/sglang/srt/</span>）。扩散模型的计算范式和大语言模型完全不同，但它惊人地<strong>复用了你这一路学过的全部服务化肌肉</strong>。本课的落点就是一句话：<strong>一套引擎，两种范式</strong>。</p>

<div class="card analogy"><div class="tag">🔌 生活类比</div>
<p>想象两位画家。第一位是<strong>口述者</strong>：他一个字一个字地讲故事，说完一个词，听到自己刚说的词，再决定下一个词——这就是自回归大语言模型，每次前向只吐一个 token，再把它喂回去（回顾第4课）。</p>
<p>第二位是<strong>显影师</strong>：他面前是一张布满雪花噪点的相纸，什么也看不清。他不是一笔一笔去画，而是把整张纸反复放进显影液里，每泡一次就<strong>去掉一点点噪声</strong>，让画面稍微清晰一些。泡上二三十次之后，一张完整的图像就从纯噪声里"浮现"出来。这位显影师从不回头听自己说了什么，他做的是对<strong>同一张潜空间画布</strong>反复精修——这正是扩散模型的工作方式：<strong>迭代去噪</strong>（iterative denoising）。</p>
<p>两位画家用的画笔、颜料、画室其实是同一套——只是作画的<strong>节奏与方式</strong>不同。把这套共用的画室换成 SGLang 引擎，把两种作画方式换成自回归与扩散，你就抓住了本课的全部精髓。</p>
</div>

<div class="card macro"><div class="tag">🌍 宏观理解</div>
<p>大语言模型是<strong>变长</strong>的：你不知道它要生成多少个 token，每个 token 都依赖前一个，是一条被反馈链拉长的序列。扩散模型是<strong>定长</strong>的：步数 N（例如 20–50 步）是事先固定的，每一步都对同一个潜变量做一次去噪，没有 token 级别的反馈，本质上是一个<strong>固定长度的精修循环</strong>。范式不同，但服务化要解决的问题——高效算子、调度循环、显存管理、多硬件、对外 API——竟然是同一批。所以 SGLang 没有为扩散另起炉灶，而是<strong>把同一套基础设施借过来用</strong>。</p>
<p>记住这条主线，本课后面的所有细节都会落到一个框架里：先看清扩散"算什么"（迭代去噪），再看清它由"哪三个零件"组成（文本编码器 + DiT + VAE），最后看清 SGLang 是"怎么用旧引擎服务新范式"的。三步走完，你就会明白为什么我们说扩散是整份指南"一切皆可插拔"理念的最佳收尾案例。</p>
</div>

<h2>一、扩散到底在做什么：从噪声走向图像</h2>
<p>自回归生成是"接龙"：模型看一段已经生成的文字，预测下一个最可能的 token，吐出来，再把它接到序列末尾，喂回模型，循环往复，直到遇到结束符。每一次前向只产出一个 token，序列长度随生成过程不断增长。</p>
<p>扩散生成是另一回事。它从一张<strong>纯随机噪声</strong>的潜变量出发，然后把模型运行 N 次。每一次运行——我们叫它一个<strong>去噪步</strong>（denoise step）——都从当前的含噪潜变量里预测并移除一部分噪声，让它朝"干净的图像"挪近一小步。N 步走完，潜变量就从一团雪花变成了一张有内容的图。这里没有 token 一个个往外蹦的过程，整条轨迹是对<strong>同一个潜变量张量</strong>的反复refine，步数是固定的、可预测的。这一点对服务化非常友好：去噪步的计算形状每一步都一样，是一个<strong>静态的、可被反复捕获的算子图</strong>。</p>
<p>为什么是"去噪"而不是"作画"？训练时，扩散模型见过的是这样一个过程：把一张干净图片<strong>逐步加噪</strong>，直到它彻底变成随机噪声；模型学习的，是这个过程的<strong>逆向</strong>——给定一张含噪图和"现在处于第几步"，预测应该减掉多少噪声。于是推理时，我们就反过来用：从纯噪声起步，让模型一步步把噪声"擦掉"，最终还原出一张此前并不存在、却符合提示词描述的全新图像。每一步只走一小段，是因为一次性从纯噪声跳到清晰图太难、误差太大；切成几十小步，模型每步只需做一点点修正，整体质量就稳得多。</p>
<p>正因为步数 N 是事先定好的，扩散推理的<strong>计算图非常规整</strong>：同一个 DiT、同样的输入形状、同样的算子序列，被原封不动地重复 N 次。相比大语言模型那种"序列越生成越长、KV 缓存不断膨胀"的动态形态，扩散的去噪循环是一个<strong>近乎完美的静态循环</strong>。后面你会看到，这种规整性正是 SGLang 各种加速手段（尤其是 CUDA Graph）能直接派上用场的根本原因。</p>

<h2>二、一条扩散管线的三个零件</h2>
<p>一条扩散生成管线由三部分组成，它们被一个 <span class="mono">PipelineConfig</span> 统一描述：</p>
<p>第一是<strong>文本编码器</strong>（text encoder，可能不止一个）：它把你的提示词（prompt）变成<strong>条件向量</strong>（conditioning），告诉后面的去噪器"我要的是一只戴帽子的猫"。第二是 <strong>DiT</strong>（Diffusion Transformer，扩散 Transformer）：它就是那个<strong>去噪器</strong>，每一个去噪步被运行一次，是整条管线里最重、最核心、被反复调用的计算体。第三是 <strong>VAE</strong>（变分自编码器）：它负责在<strong>压缩的潜空间</strong>和真实<strong>像素空间</strong>之间来回映射——去噪全程都发生在小小的潜空间里（省显存、省算力），最后一步才用 VAE 把潜变量解码成真正的图像或视频帧。</p>
<p>此外还有一个关键开关：<strong>无分类器引导</strong>（classifier-free guidance，CFG）。通过 <span class="mono">should_use_guidance</span> 与 <span class="mono">embedded_cfg_scale</span>，模型会把结果更用力地"推"向提示词描述的方向，scale 越大越贴合提示、但过大会损失多样性与真实感。这就是为什么同样的提示词、不同的引导强度，出图风格会明显不同。</p>
<p>把这三个零件串起来看：提示词先经文本编码器变成一组条件向量，这组条件在<strong>每一个去噪步</strong>都被喂给 DiT，指引它往"对的方向"去噪；DiT 在潜空间里反复精修 N 步；最后 VAE 才登场，把潜变量一次性解码成像素。值得强调的是，最重的计算几乎全在 DiT 的那 N 次前向上——文本编码只在开头跑一次，VAE 只在结尾跑一次，<strong>真正决定吞吐与延迟的是去噪循环</strong>。这也解释了为什么后面所有的优化（CUDA Graph、量化、TeaCache 跳步）都瞄准去噪步：那才是热点。</p>
<p><span class="mono">PipelineConfig</span> 把这套结构<strong>声明式地</strong>固化下来：<span class="mono">task_type</span> 说明这是文生图还是图生视频，<span class="mono">dit_config</span>/<span class="mono">vae_config</span>/<span class="mono">text_encoder_configs</span> 分别描述三个零件，<span class="mono">dit_precision</span> 指定去噪器的计算精度。换一个扩散模型，往往只是换一份 <span class="mono">PipelineConfig</span>——上层服务化代码几乎不用动。这种"配置即管线"的设计，正是它能轻松接入 SGLang 既有引擎的前提。</p>

<h2>三、真正的惊喜：它复用了你学过的一切</h2>
<p>到这里你可能以为要学一套全新的系统，恰恰相反。SGLang 的扩散服务<strong>把你整本指南学到的服务化能力原样借了过来</strong>：底层用同一套优化过的 <span class="mono">sgl-kernel</span> 算子（第38课）；用一个高效的<strong>调度循环</strong>来组织请求；对那个形状固定的去噪步做 <strong>CUDA Graph 捕获</strong>（第27课），把每一步的内核启动开销压到极低；用<strong>量化</strong>（第35课）压缩 DiT 与编码器的显存；靠<strong>多硬件后端</strong>（第42课：NVIDIA / AMD / 昇腾 NPU / Apple / 摩尔线程）让同一份代码跑在不同芯片上；对外暴露一个 <strong>OpenAI 兼容的 API</strong>（第15课）；甚至支持 PD 风格的<strong>分离式</strong>部署。</p>
<p>在这套通用地基之上，再叠加扩散特有的优化。最有代表性的是 <strong>TeaCache</strong>：它会<strong>缓存某一去噪步的结果</strong>，当相邻步之间的变化足够小时（在累积的 L1 距离上设一个阈值），就<strong>跳过</strong>那些冗余的去噪步，直接复用缓存。因为扩散是固定 N 步的精修，越到后期每步带来的改变越小，跳过其中一部分几乎不损画质却能显著提速。这类技巧是"长在"通用引擎之上的，而不是另起一摊。</p>
<p>这里值得再咀嚼一下 CUDA Graph 为什么对扩散特别合拍。大语言模型解码时，序列长度每步都在变，KV 缓存不断增长，捕获静态图要费不少心思去处理动态形状；而扩散的去噪步<strong>天生就是定形的</strong>——同一个 DiT、同一批潜变量、同一串算子，重复 N 次纹丝不动。把这一步捕获成 CUDA Graph 之后，每一步都只是"重放"一张录好的内核启动序列，把成百上千次零碎的 kernel launch 开销压到几乎为零。对一个要跑二三十步的去噪循环来说，这种省法会被<strong>放大 N 倍</strong>，收益非常可观。</p>
<p>量化（第35课）在这里同样直接受用。DiT 往往是参数量很大的 Transformer，文本编码器也不小，把它们量化到更低精度，能显著压低显存占用、让更大的扩散模型在同一张卡上跑起来。多硬件后端（第42课）则意味着这条扩散管线不挑芯片：NVIDIA、AMD、昇腾 NPU、Apple、摩尔线程，同一份模型代码都能落地。再加上 OpenAI 兼容的 API（第15课），调用方甚至感觉不到背后是文本模型还是图像模型——接口是统一的，接入成本极低。这些能力没有一项是为扩散"专门重写"的，全是把现成的服务化肌肉<strong>借过来即用</strong>。</p>

<h2>四、把两种范式并排看</h2>
<p>同一套 SGLang 引擎，既服务"一个字一个字往外接"的自回归大语言模型，也服务"对一张潜空间画布反复显影"的扩散模型。算子、调度、显存、CUDA Graph、多硬件、API——这些都是共享的；不同的只是上层的生成范式。这正是 SGLang 设计哲学的体现：<strong>一切皆可插拔</strong>，新能力以最小代价复用既有地基。</p>
<p>把两者并排，你会发现它们的"难点"恰好错位、互补：自回归难在<strong>动态形状与不断增长的 KV 缓存</strong>，于是连续批处理、前缀缓存这些手段大显身手；扩散难在<strong>同一张图要被反复跑很多步</strong>，于是 CUDA Graph 捕获、TeaCache 跳步这类"静态循环优化"价值最大。但支撑它们的底层——高性能算子、统一的调度与显存管理、多硬件适配、对外 API——是<strong>同一套</strong>。这就是为什么 SGLang 能在不重写引擎的前提下，把一个本属于"图像生成"世界的任务，干净利落地接到原本服务大语言模型的流水线上。对使用者而言，这意味着你之前为部署 LLM 学到的那套运维直觉——怎么开服务、怎么调并发、怎么换硬件、怎么压显存——几乎可以原样迁移到扩散场景，学习成本极低。</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>纯随机噪声</h4><p class="mono">latent</p><p>从一团毫无内容的高斯噪声潜变量出发，此刻画面里什么也看不出来。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>去噪步 ×N</h4><p class="mono">DiT</p><p>反复运行同一个 DiT 去噪器（如 20–50 步），每一步去掉一点点噪声，让潜变量朝干净图像挪近一小步。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>VAE 解码</h4><p class="mono">VAE</p><p>把精修完成的潜变量从压缩潜空间解码回真实像素空间。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>最终图像 / 视频</h4><p class="mono">pixels</p><p>一张有内容的图像或一段视频帧从纯噪声里“浮现”出来。</p></div></div>
</div>

<div class="cols">
  <div class="col"><strong>自回归 LLM</strong><br/>变长序列；每次前向只吐一个 token；把 token 喂回去，下一步依赖上一步；有 token 级反馈链。</div>
  <div class="col"><strong>扩散模型</strong><br/>定长 N 步；每步对同一个潜变量去噪；无 token 反馈；是对一张潜空间画布的固定长度精修。</div>
</div>

<table class="t">
  <tr><th>复用的 SGLang 部件</th><th>对扩散的好处</th></tr>
  <tr><td>sgl-kernel 优化算子（第38课）</td><td>DiT / 编码器 / VAE 直接吃到高性能内核</td></tr>
  <tr><td>高效调度循环</td><td>批量编排去噪请求，吞吐更高</td></tr>
  <tr><td>CUDA Graph 捕获（第27课）</td><td>去噪步形状固定，捕获后每步启动开销极低</td></tr>
  <tr><td>量化（第35课）</td><td>压缩 DiT 与编码器显存，跑得起更大模型</td></tr>
  <tr><td>多硬件后端（第42课）</td><td>NVIDIA / AMD / 昇腾 / Apple / 摩尔线程通吃</td></tr>
  <tr><td>OpenAI 兼容 API（第15课）</td><td>对外接口统一，接入成本极低</td></tr>
</table>

<div class="flow">
  <div class="node">提示词</div>
  <div class="arrow">→</div>
  <div class="node">文本编码器</div>
  <div class="arrow">→</div>
  <div class="node">DiT 去噪循环（TeaCache 可跳过步）</div>
  <div class="arrow">→</div>
  <div class="node">VAE 解码</div>
  <div class="arrow">→</div>
  <div class="node">像素</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/multimodal_gen/configs/pipeline_configs/base.py ::PipelineConfig</span><span class="ln">扩散管线 = 文本编码器 + DiT 去噪器 + VAE</span></div><pre>class PipelineConfig:
    # base configuration for a diffusion generation pipeline
    task_type: ModelTaskType            # e.g. text-&gt;image, image-&gt;video
    model_path: str
    dit_config: DiTConfig               # the Diffusion Transformer = the DENOISER (run once per step)
    dit_precision: str = "bf16"
    vae_config: VAEConfig               # VAE: latent &lt;-&gt; pixel space
    text_encoder_configs: tuple         # text conditioning encoders
    should_use_guidance: bool = True    # classifier-free guidance toward the prompt
    embedded_cfg_scale: float = 6.0</pre></div>

<div class="card key"><div class="tag">📌 本课要点</div>
<ul>
<li><strong>不同范式</strong>：自回归 LLM 每次前向吐一个 token 再喂回；扩散模型从纯噪声出发、迭代去噪 N 步，对同一个潜变量反复精修，无 token 反馈。</li>
<li><strong>三个零件</strong>：文本编码器（把提示词变条件）+ DiT 去噪器（每步运行一次）+ VAE（潜空间 &lt;-&gt; 像素），由 <span class="mono">PipelineConfig</span> 统一描述。</li>
<li><strong>无分类器引导</strong>：<span class="mono">should_use_guidance</span> / <span class="mono">embedded_cfg_scale</span> 把结果推向提示词。</li>
<li><strong>一套引擎，两种范式</strong>：复用 sgl-kernel（第38课）、调度循环、CUDA Graph（第27课）、量化（第35课）、多硬件（第42课）、OpenAI API（第15课）乃至 PD 分离。</li>
<li><strong>扩散特有优化</strong>：TeaCache 缓存去噪步结果，变化小于阈值时跳过冗余步，提速而几乎不损画质，是叠在通用引擎之上的加速层。</li>
</ul>
</div>

<div class="card"><div class="tag">🏁 第十一部分小结</div><p>回望整个进阶部分：<strong>多模态</strong>让引擎读懂图文音视频，<strong>多 LoRA</strong>让一份基座同时服务无数定制权重，<strong>RL 权重同步</strong>把训练侧的新权重热更新进推理引擎，而本课的<strong>扩散模型</strong>把图像视频生成接到了同一条流水线上。四种看似毫不相干的进阶能力，落到底层却<strong>复用着同一套 SGLang 引擎</strong>——同样的 sgl-kernel、同样的调度与显存管理、同样的 CUDA Graph、同样的多硬件后端与 OpenAI 兼容 API。这正是 SGLang"一切皆可插拔"哲学最有力的证明：把通用地基做扎实，新能力就只是在上面薄薄地叠一层。至此，第十一部分的进阶之旅画上句号，也为整份指南留下一句注脚——好的系统设计，让看似遥远的能力都能优雅地共享同一条道路。</p></div>
""", "en": r"""
<p class="lead">This is the last lesson of Part 11, and a nice way to close the whole guide. We turn our eyes away from large language models toward a different generation task: <strong>diffusion models</strong> — image and video generation. It lives in the <span class="mono">sglang-diffusion</span> sub-project at <span class="mono">python/sglang/multimodal_gen/</span> (note: under <span class="mono">python/sglang/</span>, NOT <span class="mono">python/sglang/srt/</span>). Diffusion uses a compute pattern completely different from LLMs, yet it astonishingly <strong>reuses every bit of serving muscle you have built up along the way</strong>. The punchline of this lesson is one phrase: <strong>one stack, two paradigms</strong>.</p>

<div class="card analogy"><div class="tag">🔌 Analogy</div>
<p>Picture two painters. The first is a <strong>narrator</strong>: he tells a story one word at a time, hears the word he just said, then decides the next — that's the autoregressive LLM, emitting one token per forward and feeding it back (recall Lesson 4).</p>
<p>The second is a <strong>darkroom developer</strong>: in front of him is photo paper covered in snowy noise, showing nothing. He doesn't draw stroke by stroke; he repeatedly dips the whole sheet into developer fluid, and each dip <strong>removes a little noise</strong>, making the picture slightly clearer. After twenty or thirty dips, a full image "emerges" out of pure noise. This developer never looks back at what he said; he keeps refining the <strong>same latent canvas</strong> over and over — exactly how diffusion works: <strong>iterative denoising</strong>.</p>
<p>The two painters actually use the same brushes, paints and studio — only the <strong>rhythm and manner</strong> of painting differ. Replace that shared studio with the SGLang engine, and the two painting manners with autoregression and diffusion, and you've grasped the whole point of this lesson.</p>
</div>

<div class="card macro"><div class="tag">🌍 The big picture</div>
<p>An LLM is <strong>variable-length</strong>: you don't know how many tokens it will produce, each token depends on the previous one, a sequence stretched out by a feedback chain. A diffusion model is <strong>fixed-length</strong>: the step count N (say 20–50) is decided in advance, every step denoises the same latent, there is no token-level feedback — it is essentially a <strong>fixed-length refinement loop</strong>. The paradigms differ, but the serving problems to solve — fast kernels, a scheduler loop, memory management, multi-hardware, an external API — turn out to be the same set. So SGLang doesn't build a separate stack for diffusion; it <strong>borrows the same infrastructure</strong>.</p>
<p>Keep this through-line in mind and every later detail falls into one frame: first see clearly what diffusion <em>computes</em> (iterative denoising), then which <em>three parts</em> it's made of (text encoder + DiT + VAE), and finally <em>how</em> SGLang serves a new paradigm with an old engine. Walk those three steps and you'll see why we call diffusion the best closing case for the whole guide's "everything is pluggable" idea.</p>
</div>

<h2>1. What diffusion actually does: from noise toward an image</h2>
<p>Autoregressive generation is "word chaining": the model looks at the text generated so far, predicts the next most likely token, emits it, appends it to the sequence, feeds it back, and loops until an end token. Each forward produces exactly one token, and the sequence grows as generation proceeds.</p>
<p>Diffusion generation is something else. It starts from a <strong>pure random noise</strong> latent, then runs the model N times. Each run — we call it a <strong>denoise step</strong> — predicts and removes a portion of the noise from the current noisy latent, nudging it one small step toward "a clean image". After N steps the latent goes from snow to a meaningful picture. There is no token popping out one by one; the whole trajectory is repeated refinement of <strong>the same latent tensor</strong>, with a fixed, predictable number of steps. This is very serving-friendly: the denoise step has the same compute shape every step — a <strong>static op graph that can be captured and replayed</strong>.</p>
<p>Why "denoise" and not "paint"? During training, the diffusion model saw this process: take a clean image and <strong>progressively add noise</strong> until it becomes pure random noise; what the model learns is the <strong>reverse</strong> — given a noisy image and "which step we're at", predict how much noise to subtract. So at inference we run it backwards: start from pure noise and let the model "erase" the noise step by step, finally reconstructing a brand-new image that never existed before yet matches the prompt. Each step moves only a little because jumping from pure noise to a sharp image in one shot is too hard and too error-prone; sliced into dozens of small steps, the model only needs a tiny correction each step, and overall quality is far more stable.</p>
<p>Because the step count N is fixed in advance, diffusion inference has a <strong>very regular compute graph</strong>: the same DiT, the same input shapes, the same op sequence, repeated N times unchanged. Compared to an LLM's dynamic shape — "the sequence grows as it generates and the KV cache keeps swelling" — diffusion's denoise loop is a <strong>near-perfect static loop</strong>. As you'll see, this regularity is exactly why SGLang's acceleration tricks (especially CUDA graph) plug straight in.</p>

<h2>2. The three parts of a diffusion pipeline</h2>
<p>A diffusion generation pipeline has three parts, all described by a single <span class="mono">PipelineConfig</span>:</p>
<p>First, the <strong>text encoder(s)</strong> (there may be more than one): they turn your prompt into a <strong>conditioning</strong> vector that tells the denoiser "I want a cat wearing a hat". Second, the <strong>DiT</strong> (Diffusion Transformer): this is the <strong>denoiser</strong>, run once per denoise step — the heaviest, most central, repeatedly-invoked compute body of the pipeline. Third, the <strong>VAE</strong> (variational autoencoder): it maps back and forth between the compressed <strong>latent space</strong> and real <strong>pixel space</strong> — all the denoising happens in the small latent space (saving memory and compute), and only the final step uses the VAE to decode the latent into an actual image or video frame.</p>
<p>There is also a key switch: <strong>classifier-free guidance</strong> (CFG). Through <span class="mono">should_use_guidance</span> and <span class="mono">embedded_cfg_scale</span>, the model pushes the result harder toward what the prompt describes — a larger scale follows the prompt more closely, but too large costs diversity and realism. That's why the same prompt at different guidance strengths yields visibly different styles.</p>
<p>Stringing the three parts together: the prompt first becomes a set of conditioning vectors via the text encoder, and that conditioning is fed to the DiT at <strong>every denoise step</strong>, guiding it to denoise in the "right direction"; the DiT refines in latent space for N steps; finally the VAE steps in and decodes the latent into pixels in one shot. Worth stressing: nearly all the heavy compute is in the DiT's N forwards — text encoding runs once at the start, the VAE once at the end, so <strong>what really determines throughput and latency is the denoise loop</strong>. This is why all the later optimizations (CUDA graph, quantization, TeaCache step-skipping) target the denoise step: that's the hot spot.</p>
<p><span class="mono">PipelineConfig</span> freezes this structure <strong>declaratively</strong>: <span class="mono">task_type</span> says whether this is text-to-image or image-to-video, <span class="mono">dit_config</span>/<span class="mono">vae_config</span>/<span class="mono">text_encoder_configs</span> describe the three parts, and <span class="mono">dit_precision</span> sets the denoiser's compute precision. Swapping in a different diffusion model is often just swapping a <span class="mono">PipelineConfig</span> — the upper serving code barely changes. This "config as pipeline" design is precisely what lets it plug into SGLang's existing engine so easily.</p>

<h2>3. The real surprise: it reuses everything you learned</h2>
<p>You might expect to learn a brand-new system here; quite the opposite. SGLang's diffusion serving <strong>borrows, as-is, the serving capabilities you studied throughout the guide</strong>: the same optimized <span class="mono">sgl-kernel</span> ops underneath (Lesson 38); an efficient <strong>scheduler loop</strong> to organize requests; <strong>CUDA-graph capture</strong> (Lesson 27) over that fixed-shape denoise step to crush per-step launch overhead; <strong>quantization</strong> (Lesson 35) to compress the DiT and encoders in memory; <strong>multi-hardware backends</strong> (Lesson 42: NVIDIA / AMD / Ascend NPU / Apple / Moore Threads) so one codebase runs on different chips; an <strong>OpenAI-compatible API</strong> (Lesson 15) exposed externally; and even PD-style <strong>disaggregation</strong>.</p>
<p>On top of this general foundation, diffusion-specific optimizations layer on. The most representative is <strong>TeaCache</strong>: it <strong>caches the result of a denoise step</strong>, and when the change between adjacent steps is small enough (a threshold on accumulated L1 distance) it <strong>skips</strong> those redundant denoise steps and reuses the cache. Because diffusion is a fixed N-step refinement, later steps change less and less, so skipping some barely hurts quality while speeding things up noticeably. Such tricks "grow on top of" the general engine rather than forming a separate stack.</p>
<p>It's worth chewing on why CUDA graph fits diffusion so well. When an LLM decodes, the sequence length changes every step and the KV cache keeps growing, so capturing a static graph takes real effort to handle dynamic shapes; diffusion's denoise step, by contrast, is <strong>fixed-shape by nature</strong> — the same DiT, the same latents, the same op chain, repeated N times unchanged. Once that step is captured as a CUDA graph, each step merely "replays" a recorded launch sequence, crushing hundreds or thousands of tiny kernel-launch overheads to almost nothing. For a denoise loop that runs twenty or thirty steps, this saving is <strong>amplified N times</strong> — a substantial gain.</p>
<p>Quantization (Lesson 35) pays off directly here too. The DiT is often a large-parameter Transformer, and the text encoders aren't small either; quantizing them to lower precision substantially cuts memory and lets bigger diffusion models run on the same card. Multi-hardware backends (Lesson 42) mean this pipeline isn't picky about chips: NVIDIA, AMD, Ascend NPU, Apple, Moore Threads — the same model code lands on all of them. Add the OpenAI-compatible API (Lesson 15) and callers can't even tell whether a text or image model is behind it — the interface is unified, integration is near-zero cost. None of these capabilities was "rewritten specially" for diffusion; they are all existing serving muscle <strong>borrowed and used as-is</strong>.</p>

<h2>4. The two paradigms side by side</h2>
<p>The same SGLang engine serves both the autoregressive LLM that "chains one token at a time" and the diffusion model that "develops the same latent canvas repeatedly". Kernels, scheduling, memory, CUDA graph, multi-hardware, API — all shared; only the upper generation paradigm differs. This embodies SGLang's design philosophy: <strong>everything is pluggable</strong>, and new capabilities reuse the existing foundation at minimal cost.</p>
<p>Put side by side, their "hard parts" are nicely offset and complementary: autoregression is hard because of <strong>dynamic shapes and an ever-growing KV cache</strong>, so continuous batching and prefix caching shine; diffusion is hard because <strong>the same image is run many steps over</strong>, so "static-loop optimizations" like CUDA-graph capture and TeaCache step-skipping matter most. But the foundation under both — high-performance kernels, unified scheduling and memory management, multi-hardware support, an external API — is <strong>the same one</strong>. That's why SGLang can cleanly bolt a task from the "image generation" world onto a pipeline originally serving LLMs, without rewriting the engine. For users it means the operational intuition you built deploying LLMs — how to start a server, tune concurrency, switch hardware, squeeze memory — transfers almost unchanged to diffusion, at very low learning cost.</p>

<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>Pure random noise</h4><p class="mono">latent</p><p>Start from a contentless Gaussian-noise latent; nothing is visible in the picture yet.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>Denoise step ×N</h4><p class="mono">DiT</p><p>Run the same DiT denoiser repeatedly (e.g. 20–50 steps); each step removes a little noise, nudging the latent toward a clean image.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>VAE decode</h4><p class="mono">VAE</p><p>Decode the refined latent from compressed latent space back into real pixel space.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>Final image / video</h4><p class="mono">pixels</p><p>A meaningful image or a video frame "emerges" out of pure noise.</p></div></div>
</div>

<div class="cols">
  <div class="col"><strong>Autoregressive LLM</strong><br/>Variable-length sequence; each forward emits one token; the token is fed back, next step depends on the last; a token-level feedback chain.</div>
  <div class="col"><strong>Diffusion model</strong><br/>Fixed N steps; each step denoises the same latent; no token feedback; a fixed-length refinement of one latent canvas.</div>
</div>

<table class="t">
  <tr><th>Reused SGLang piece</th><th>Benefit for diffusion</th></tr>
  <tr><td>sgl-kernel optimized ops (Lesson 38)</td><td>DiT / encoders / VAE get high-performance kernels directly</td></tr>
  <tr><td>Efficient scheduler loop</td><td>Batches denoise requests for higher throughput</td></tr>
  <tr><td>CUDA-graph capture (Lesson 27)</td><td>Denoise step is fixed-shape; once captured, per-step launch cost is tiny</td></tr>
  <tr><td>Quantization (Lesson 35)</td><td>Compresses DiT and encoder memory, fits larger models</td></tr>
  <tr><td>Multi-hardware backends (Lesson 42)</td><td>NVIDIA / AMD / Ascend / Apple / Moore Threads all covered</td></tr>
  <tr><td>OpenAI-compatible API (Lesson 15)</td><td>Unified external interface, near-zero integration cost</td></tr>
</table>

<div class="flow">
  <div class="node">Prompt</div>
  <div class="arrow">→</div>
  <div class="node">Text encoder</div>
  <div class="arrow">→</div>
  <div class="node">DiT denoise loop (TeaCache may skip steps)</div>
  <div class="arrow">→</div>
  <div class="node">VAE decode</div>
  <div class="arrow">→</div>
  <div class="node">Pixels</div>
</div>

<div class="codefile"><div class="cf-head"><span class="dot"></span><span class="path">python/sglang/multimodal_gen/configs/pipeline_configs/base.py ::PipelineConfig</span><span class="ln">a diffusion pipeline = text encoders + the DiT denoiser + a VAE</span></div><pre>class PipelineConfig:
    # base configuration for a diffusion generation pipeline
    task_type: ModelTaskType            # e.g. text-&gt;image, image-&gt;video
    model_path: str
    dit_config: DiTConfig               # the Diffusion Transformer = the DENOISER (run once per step)
    dit_precision: str = "bf16"
    vae_config: VAEConfig               # VAE: latent &lt;-&gt; pixel space
    text_encoder_configs: tuple         # text conditioning encoders
    should_use_guidance: bool = True    # classifier-free guidance toward the prompt
    embedded_cfg_scale: float = 6.0</pre></div>

<div class="card key"><div class="tag">📌 Key points</div>
<ul>
<li><strong>Different paradigm</strong>: an autoregressive LLM emits one token per forward and feeds it back; diffusion starts from pure noise and iteratively denoises N steps, refining the same latent, with no token feedback.</li>
<li><strong>Three parts</strong>: text encoder(s) (prompt → conditioning) + the DiT denoiser (run once per step) + the VAE (latent &lt;-&gt; pixels), all described by <span class="mono">PipelineConfig</span>.</li>
<li><strong>Classifier-free guidance</strong>: <span class="mono">should_use_guidance</span> / <span class="mono">embedded_cfg_scale</span> steer the result toward the prompt.</li>
<li><strong>One stack, two paradigms</strong>: reuses sgl-kernel (Lesson 38), the scheduler loop, CUDA graph (Lesson 27), quantization (Lesson 35), multi-hardware (Lesson 42), the OpenAI API (Lesson 15), and even PD disaggregation.</li>
<li><strong>Diffusion-specific trick</strong>: TeaCache caches a step's result and skips redundant steps when the change is below threshold, speeding up with almost no quality loss — an acceleration layer on top of the general engine.</li>
</ul>
</div>

<div class="card"><div class="tag">🏁 Part 11 wrap-up</div><p>Looking back over the whole advanced part: <strong>multimodality</strong> lets the engine understand text, image, audio and video; <strong>multi-LoRA</strong> lets one base model serve countless customized weights at once; <strong>RL weight sync</strong> hot-updates fresh training-side weights into the inference engine; and this lesson's <strong>diffusion models</strong> connect image and video generation onto the same pipeline. Four seemingly unrelated advanced capabilities, yet underneath they all <strong>reuse the same SGLang engine</strong> — the same sgl-kernel, the same scheduling and memory management, the same CUDA graph, the same multi-hardware backends and OpenAI-compatible API. This is the strongest proof of SGLang's "everything is pluggable" philosophy: make the general foundation solid, and a new capability is just a thin layer on top. With that, Part 11's advanced journey comes to a close, leaving a footnote for the whole guide — good system design lets seemingly distant capabilities elegantly share the same road.</p></div>
"""}
