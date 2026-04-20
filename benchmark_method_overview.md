# Latent Communication：Benchmark 与三类实验方法总览

## 1. 我们采用的 Benchmark

这台机器上的 Latent 通信实验并不是只围绕一个数据集展开，而是覆盖了三类 benchmark：

- 早期构造任务与 toy task
- 经典 QA / 阅读理解 benchmark
- 更接近多智能体协作的 side-channel / planning benchmark

下面按 benchmark 类型总结。

### 1.1 早期构造任务与 toy task

这些任务主要用于快速验证 latent 通道能否替代显式文本上下文，或者用于做通信能力的可控 probe。

| Benchmark / 任务 | 位置 | 典型指标 | 作用 |
|---|---|---|---|
| Code Understanding | [`latent_comm/results_code`](/data1/liyuejia/latent_comm/results_code/results.json) | MC accuracy | 测试代码语义是否能经 latent 传递 |
| Multi-Hop | [`latent_comm/results_multihop_v2`](/data1/liyuejia/latent_comm/results_multihop_v2/results.json) | accuracy | 测试跨段事实组合能力 |
| Planning | [`latent_comm/results_planning_v2`](/data1/liyuejia/latent_comm/results_planning_v2/results.json) | accuracy | 测试规划信息是否能经 latent 传递 |
| Long Context | [`latent_comm/results_long_ctx_v2`](/data1/liyuejia/latent_comm/results_long_ctx_v2/results.json) | EM / F1 或 accuracy | 测试长文本关键信息压缩 |
| Toy training-free prefix probe | [`toy_performance.md`](/data1/liyuejia/toy_performance.md) | keyword hit | 测试“裸 hidden 是否可直接读” |

这类 benchmark 的特点是：

- 上手快
- 可控性强
- 适合验证通信机制是否“有信号”
- 但不适合作为最终主结论

### 1.2 经典 QA / 阅读理解 benchmark

这类 benchmark 是中期主实验的核心。

| Benchmark | 位置 | 任务形式 | 典型指标 | 备注 |
|---|---|---|---|---|
| SQuAD v2 MC 改写 | [`results_squad_mc`](/data1/liyuejia/latent_comm/results_squad_mc/results.json) | 4-way MC | accuracy | 用文章内 hard negative 构造 |
| SQuAD v2 direct QA | [`results_squad_v4`](/data1/liyuejia/latent_comm/results_squad_v4/results.json) | 开放式 QA | EM / F1 | article-level split，最接近原任务 |
| HotpotQA | [`results_hotpotqa_mc`](/data1/liyuejia/latent_comm/results_hotpotqa_mc/results.json) | 4-way MC | accuracy | 多跳阅读理解 |
| TriviaQA | [`results_triviaqa_mc`](/data1/liyuejia/latent_comm/results_triviaqa_mc/results.json) | 4-way MC | accuracy | 开放域证据阅读 |
| BoolQ | [`results_boolq`](/data1/liyuejia/latent_comm/results_boolq/results.json) | Yes/No QA | accuracy | 最干净的原生二分类 benchmark |

这类 benchmark 的特点是：

- 比 toy task 更接近真实任务
- 能比较清晰地区分“latent 有效”与“只是 task 太容易”
- 但有一部分实验为了方便训练做了 MC 化改写，因此需要谨慎解释

### 1.3 Side-channel / 协作式 benchmark

这类 benchmark 主要用于验证“sender 文本消息 + latent side-channel”的组合是否能改善协作。

| Benchmark | 位置 | 任务形式 | 典型指标 | 备注 |
|---|---|---|---|---|
| HotpotQA side-channel | [`paper_table.md`](/data1/liyuejia/latent_side_channel/results/dev500_layer12_paper_eval/paper_table.md) | QA with text+latent side-channel | F1 / subEM / F1_extract | 主结果线 |
| GPQA | [`checkpoints_gpqa_hidden_hook_layer12_e10/train_summary.json`](/data1/liyuejia/latent_side_channel/checkpoints_gpqa_hidden_hook_layer12_e10/train_summary.json) | MC science QA | accuracy | hidden-hook layer12 |
| 2WikiMultihopQA | [`checkpoints_2wiki_hidden_hook_layer12_singlegpu/train_summary.json`](/data1/liyuejia/latent_side_channel/checkpoints_2wiki_hidden_hook_layer12_singlegpu/train_summary.json) | open QA | F1 | 同样是 hidden-hook |
| ALFWorld / planning | [`train_summary.json`](/data1/liyuejia/latent_side_channel/checkpoints_planning_scale40_dev20_probe/train_summary.json), [`rollout_dev.json`](/data1/liyuejia/latent_side_channel/results/alfworld_scale40_dev8_rollout_20260410/rollout_dev.json) | planner-executor 多轮协作 | loss / matched action rate / success rate | 研究 latent 的跨轮累积 |
| BoolQ side-channel pilot | [`final_results.json`](/data1/liyuejia/latent_side_channel/checkpoints_boolq_hidden_hook_layer12_instruction_pilot_500_gpu1_retry/final_results.json) | Yes/No QA + sender note | accuracy / qualitative analysis | 目前还是 pilot 性质 |

### 1.4 其它 probe benchmark

| Benchmark | 位置 | 任务形式 | 指标 | 备注 |
|---|---|---|---|---|
| HiddenBench | [`results_hiddenbench/results.json`](/data1/liyuejia/latent_comm/results_hiddenbench/results.json) | hidden profile reasoning | accuracy | 主要拿来比较 TextMAS / LatentMAS 风格方案 |

## 2. 方法一：Frozen Prefix Latent Communication

### 2.1 方法定义

这是项目最早、也是覆盖 benchmark 最广的一类方法，代码主入口在：

- [latent_comm/model.py](/data1/liyuejia/latent_comm/model.py)

核心流程：

1. Agent A 用冻结的 `Qwen3-8B` 读取 `context`
2. 取最后一层 hidden states
3. 经 `LatentEncoder` 压成 `z`
4. 再经 `LatentDecoder` 或 `LatentDecoderCrossAttn` 解成 B 的 soft prefix
5. Agent B 只看 `question`，结合 prefix 生成最终答案

这类方法的关键特征是：

- 大模型冻结
- 只训练通信模块
- latent 接口是 **prefix token**

### 2.2 跑过的 benchmark

- 早期构造任务：Code / Multi-Hop / Planning / Long Context
- SQuAD MC
- SQuAD direct QA
- HotpotQA
- TriviaQA
- BoolQ
- HiddenBench 的部分相关对照

### 2.3 实验效果

#### 2.3.1 在早期构造任务上

代表性结果可见：

- [`cross_task_summary.json`](/data1/liyuejia/latent_comm/cross_task_summary.json)
- [`cross_task_summary_v2.json`](/data1/liyuejia/latent_comm/cross_task_summary_v2.json)

典型现象：

- Planning / Multi-Hop / Code 等构造任务上，latent 往往远强于弱文本基线
- 例如：
  - Planning v2：`18.0 -> 89.67`
  - Multi-Hop v2：`19.0 -> 92.0/92.5`
  - Code v2：`1.67 -> 49.17`

但这些任务的 baseline 在不同版本之间波动很大，因此它们更适合作为机制验证，而不是最终主结论。

#### 2.3.2 在经典 QA benchmark 上

代表结果：

- SQuAD MC：[`results_squad_mc/results.json`](/data1/liyuejia/latent_comm/results_squad_mc/results.json)
- SQuAD MC fair：[`results_squad_mc_fair/results.json`](/data1/liyuejia/latent_comm/results_squad_mc_fair/results.json)
- SQuAD v4：[`results_squad_v4/results.json`](/data1/liyuejia/latent_comm/results_squad_v4/results.json)
- HotpotQA：[`results_hotpotqa_mc/results.json`](/data1/liyuejia/latent_comm/results_hotpotqa_mc/results.json)
- TriviaQA：[`results_triviaqa_mc/results.json`](/data1/liyuejia/latent_comm/results_triviaqa_mc/results.json)
- BoolQ：[`results_boolq/results.json`](/data1/liyuejia/latent_comm/results_boolq/results.json)

代表性结果如下：

| Benchmark | 文本基线 | latent | 备注 |
|---|---:|---:|---|
| SQuAD MC | 23.67 | 94.17 | MC 改写后 latent 极强 |
| SQuAD MC fair | trained_text 99.00 | 92.00 | 文本训练后仍更强 |
| SQuAD v4 direct QA | F1 18.72 | F1 6.99 | 回到原生 QA 后 latent 明显不够 |
| HotpotQA MC | 21.50 | 43.67 | 有提升，但不是压倒性 |
| TriviaQA MC | 19.17 | 27.17 | 有提升，但幅度有限 |
| BoolQ | full text 86.00 | compressed latent 72.17 | clean benchmark 上有效但不如全文 |

#### 2.3.3 在 BoolQ 上的意义

BoolQ 结果尤其重要，因为它不是人为构造的 MC 任务：

- `chat_full_text = 86.0`
- `chat_no_context = 68.33`
- `latent_compressed = 72.17`
- `latent_full = 73.5`
- `random_z = 13.0`

这说明：

- latent 确实能传递 passage 信息
- 但它还不能替代完整文本

### 2.4 对方法一的判断

方法一的主要贡献是：

- 在很多 benchmark 上证明了 latent channel 不是完全没用
- 在 BoolQ 这种干净任务上给出了最可信的主结果之一

方法一的主要局限是：

- 在原生 QA（例如 SQuAD v4）上不够强
- prefix latent 的注入方式可能太浅
- B 很可能只是在读一个压缩 prefix，而不是与 sender 真正形成 richer side-channel

## 3. 方法二：LoRA End-to-End Latent Communication

### 3.1 方法定义

相关代码：

- [lora_latent_comm/train.py](/data1/liyuejia/lora_latent_comm/train.py)
- [lora_latent_comm/model.py](/data1/liyuejia/lora_latent_comm/model.py)

与方法一相比，这一类方法的变化是：

- 发送端加 `LoRA_A`
- 接收端加 `LoRA_B`
- 仍然保留 latent bottleneck
- 梯度穿过 sender、通信模块和 receiver 三部分

也就是说，它试图把 latent 通信从“只训练接口”推进到“端到端适配”。

### 3.2 跑过的 benchmark

这台机器上这一类方法主要只系统跑过 **BoolQ**。

对应结果：

- [results_boolq_frozen/results.json](/data1/liyuejia/lora_latent_comm/results_boolq_frozen/results.json)
- 以及各类日志：
  - [nohup_boolq_frozen.log](/data1/liyuejia/lora_latent_comm/nohup_boolq_frozen.log)
  - [nohup_boolq_lora_a_only.log](/data1/liyuejia/lora_latent_comm/nohup_boolq_lora_a_only.log)
  - [nohup_boolq_lora_b_only.log](/data1/liyuejia/lora_latent_comm/nohup_boolq_lora_b_only.log)
  - [nohup_boolq_lora_ab.log](/data1/liyuejia/lora_latent_comm/nohup_boolq_lora_ab.log)
  - [nohup_boolq_lora_r8.log](/data1/liyuejia/lora_latent_comm/nohup_boolq_lora_r8.log)

### 3.3 实验效果

代表结果如下：

| 变体 | 最佳准确率 |
|---|---:|
| 小规模 smoke run | 60.00 |
| 冻结 LoRA，只训通信模块 | 74.83 |
| 只训 `LoRA_A + comm` | 73.83 |
| 只训 `LoRA_B + comm` | 73.33 |
| 同时训 `LoRA_A + LoRA_B + comm` | 73.17 |
| `r=8` 低秩 LoRA | 68.17 |

### 3.4 对方法二的判断

方法二最重要的发现不是“端到端显著提升”，恰恰相反：

- 端到端 LoRA 并没有明显超过“冻结大模型，只训练通信模块”的设定
- 当前本机上的最好结果反而是 `freeze_all_lora = 74.83`

因此方法二的结论更偏负面：

- 端到端调 sender / receiver 并不一定能自动改善 latent 通信
- 至少在 BoolQ 上，通信接口设计比 LoRA 端到端更关键

## 4. 方法三：Latent Side-Channel Adapter

### 4.1 方法定义

这是后期更系统的一条主线，对应代码：

- [latent_side_channel/model.py](/data1/liyuejia/latent_side_channel/model.py)
- [latent_side_channel/train_hotpotqa.py](/data1/liyuejia/latent_side_channel/train_hotpotqa.py)
- [latent_side_channel/train_gpqa_cross.py](/data1/liyuejia/latent_side_channel/train_gpqa_cross.py)
- [latent_side_channel/train_alfworld.py](/data1/liyuejia/latent_side_channel/train_alfworld.py)

它和方法一的根本区别是：

1. Agent A 不只是把 context 隐向量压缩后发给 B
2. A 先显式生成一条 **文本消息 / reasoning plan / evidence note**
3. 再从这段 message span 中提取 hidden states
4. 用 `LatentAdapter` 压成 latent token
5. 接收端同时收到：
   - 可见文本消息 `M`
   - latent side-channel `Z`
6. `Z` 可以通过三种方式注入 B：
   - `prefix`
   - `cross_attn`
   - `hidden_cross_attn`

在实际结果里，最有代表性的是：

- `hidden_cross_attn`
- `fusion_layer = 12`

### 4.2 跑过的 benchmark

这类方法实际覆盖了：

- HotpotQA side-channel
- GPQA
- 2WikiMultihopQA
- ALFWorld / multi-round planning
- BoolQ side-channel pilot

### 4.3 实验效果

#### 4.3.1 HotpotQA

主结果文件：

- [paper_table.md](/data1/liyuejia/latent_side_channel/results/dev500_layer12_paper_eval/paper_table.md)

结果：

| 方法 | F1 | subEM | F1_extract | GoldTokRecall |
|---|---:|---:|---:|---:|
| No Communication | 13.58 | 55.40 | 44.55 | 60.60 |
| Text Only | 13.39 | 55.40 | 51.63 | 61.40 |
| Latent Only | 17.61 | 68.00 | 18.95 | 71.77 |
| Text + Latent (Layer 12) | 18.90 | 79.40 | 20.87 | 79.18 |

这是本机上最强、也最可信的 side-channel 主结果线：

- `text + latent` 确实超过了 `text_only`
- 说明 hidden-hook layer12 这条方法线是有效的

#### 4.3.2 GPQA

结果文件：

- [train_summary.json](/data1/liyuejia/latent_side_channel/checkpoints_gpqa_hidden_hook_layer12_e10/train_summary.json)

核心结果：

- `best_dev_text_latent_acc = 32.5`

说明 hidden-hook layer12 至少可以迁移到更难的 MC 科学问答任务。

#### 4.3.3 2WikiMultihopQA

结果文件：

- [train_summary.json](/data1/liyuejia/latent_side_channel/checkpoints_2wiki_hidden_hook_layer12_singlegpu/train_summary.json)

核心结果：

- `best_dev_f1 = 1.72`

这说明 open QA 场景上这条 side-channel 方法目前还没跑通。

#### 4.3.4 ALFWorld / 多轮规划

结果文件：

- [train_summary.json](/data1/liyuejia/latent_side_channel/checkpoints_planning_scale40_dev20_probe/train_summary.json)
- [rollout_dev.json](/data1/liyuejia/latent_side_channel/results/alfworld_scale40_dev8_rollout_20260410/rollout_dev.json)

关键现象：

- 训练阶段，latent 版的 dev loss 明显低于 text-only
  - `best_dev_loss = 1.72`
  - `text_only_baseline_loss = 3.41`
- rollout 阶段：
  - `multi_text` 的 matched action rate = `25.0`
  - `multi_text_latent` = `37.5`
  - 但 success rate 仍然是 `0`

也就是说：

- latent 跨轮累积对 action matching 有帮助
- 但还没有真正转化成任务成功率

#### 4.3.5 BoolQ side-channel pilot

BoolQ 在方法三里又分成两种 supervision：

##### A. Label-only supervision

结果文件：

- [final_results.json](/data1/liyuejia/latent_side_channel/checkpoints_boolq_hidden_hook_layer12_pilot_500/final_results.json)

pilot 结果：

- `no_comm = 12`
- `text_only = 19`
- `text_latent = 71`

但这一版后来被证明是 **标签塌缩**：

- 模型几乎全输出 `YesYesYes...`
- 71% 主要是由于验证集里 `Yes` 比例高

因此这不是一个可信的正结果。

##### B. Instruction-style supervision

结果文件：

- [final_results.json](/data1/liyuejia/latent_side_channel/checkpoints_boolq_hidden_hook_layer12_instruction_pilot_500_gpu1_retry/final_results.json)
- [compare_text_vs_latent_only.json](/data1/liyuejia/latent_side_channel/checkpoints_boolq_hidden_hook_layer12_instruction_pilot_500_gpu1_retry/compare_text_vs_latent_only.json)

直接结果：

| 模式 | accuracy |
|---|---:|
| `no_comm` | 36.0 |
| `text_only` | 68.0 |
| `latent_only` | 0.0 |
| `text_latent` | 0.0 |

但这一版的行为很不一样：

- 它不再塌成全 `Yes`
- `latent_only` 和 `text_latent` 都会输出 evidence-like 片段
- 说明 latent 里确实带有可恢复的语义信息

例如：

- `The Spanish alphabet has 27 letters`
- `Tampa Bay Lightning -- Stanley Cup Finals`
- `Smooth muscle -- Smooth muscle is composed of`

这版失败的根源不是“latent 里没信息”，而是：

- 模型学会了输出证据
- 却没有稳定学会把证据收束成最终 `Answer: Yes/No`

### 4.4 对方法三的判断

方法三是目前这台机器上**最值得继续推进**的一类方法，因为：

- 它在 HotpotQA 上给出了最强主结果
- 在 GPQA 上也表现出了一定迁移性
- 在 ALFWorld 上已经显示出跨轮 latent 的潜力
- 在 BoolQ 上虽然 final decision 还没训好，但已经能看到 latent-only 产生 evidence 的现象

方法三当前最大的挑战是：

- 如何把 latent 中已经存在的证据语义，稳定映射成最终任务答案
- 尤其是在 BoolQ 这类 binary decision task 上，需要更好的 supervision 设计

## 5. 最后的整体结论

如果把三类方法放在一起比较，可以得到一个比较清楚的总结。

### 方法一：Frozen Prefix Latent Communication

- 覆盖 benchmark 最广
- 在 BoolQ 等任务上给出了最可信的基础结论
- 证明 latent 通道确实能传信息
- 但在原生 QA 上能力仍然不足

### 方法二：LoRA End-to-End Latent Communication

- 在 BoolQ 上验证过
- 没有明显超过冻结模型 + 通信模块训练
- 说明端到端 LoRA 并不是核心突破点

### 方法三：Latent Side-Channel Adapter

- 是目前最强、也最有研究价值的一条主线
- 在 HotpotQA 上有最可信的正结果
- 在 GPQA / ALFWorld / BoolQ 上都显示出进一步拓展空间
- 尤其是 BoolQ 的 instruction-style pilot 表明：
  - latent 已经能传证据语义
  - 但 decision head 还没有训好

因此，如果后续只保留一条主线继续往下做，最合理的选择是：

- 继续沿着 **方法三：Latent Side-Channel Adapter**
- 优先围绕：
  - HotpotQA 主结果线
  - BoolQ 的 instruction-style decision supervision
  - ALFWorld 的多轮 latent accumulation

## 6. 推荐的未来实验方向

这一节不是“已经跑过的实验”，而是基于当前项目进展、以及 latent communication 的研究目标，总结出的 **下一阶段最值得推进的 benchmark 方向**。

这些方向的共同目标是：

- 让通信成为 **necessary**，而不是可有可无
- 让 latent 通信的优势不只是“能压缩信息”，而是能传递文本不容易表达的细粒度或软信息
- 让实验从静态 QA 走向真正的多智能体协作

### 6.1 方向 A：部分可观测协作任务

这是目前最成熟、也最容易和已有工作对齐的一类方向。

核心设定是：

- 每个 agent 只能看到任务的一部分信息
- 任一方单独看到的信息都不足以完成任务
- 只有通过通信把两边的信息拼起来，系统才能成功

这类任务非常适合用来回答：

> latent communication 是否真的提升了多智能体的信息整合能力？

#### 6.1.1 Referential Game / Signaling Game

这是 emergent communication 文献里最经典的一类任务。

标准设定：

- Agent A 看到一个 `target` 和一组候选
- Agent B 只看到候选
- A 必须发出信号，让 B 选中正确目标

对 Latent 通信特别相关的两个升级方向是：

- `Colors in Context (Monroe et al.)`
  - 特点：颜色之间差异非常细微，语言描述天然有歧义
  - 研究价值：latent 可能比离散 token 更擅长表达“微妙但连续”的差别

- `Signalling with Large Language Models (Lazaridou 系列)`
  - 特点：用 LLM 直接扮演 signaler / listener
  - 研究价值：可以直接和当前项目的 sender / receiver 结构对接

适合作为推荐方向的原因：

- 标准化程度高
- 结果容易解释
- 很适合直接比较 `text-only`、`latent-only`、`text+latent`

#### 6.1.2 Collaborative Reasoning with Split Information

这是与当前项目最契合的一条推荐方向。

核心思想：

- 问题本身需要两份不同知识才能回答
- Agent A 只拿到知识 A
- Agent B 只拿到知识 B
- 两人单独都无法答题，必须通信

这类任务的关键不是 benchmark 名字本身，而是：

- **必须人为设计 split**
- 保证任一方独立观察时信息都不充分

推荐的现成 benchmark 包括：

- `MuSiQue`
  - 特点：天然是多跳 QA
  - 推荐做法：把支持证据分成两部分给两个 agent

- `2WikiMultihopQA`
  - 特点：当前项目已经在 side-channel 里碰过，复用成本低
  - 推荐做法：重新设计 evidence split，使通信成为必要条件

- `FanOutQA`
  - 特点：需要多次子问题与证据聚合
  - 推荐做法：让不同 agent 各负责不同子问题或证据簇

这一方向最推荐的原因是：

- 和当前 `HotpotQA side-channel` 主线最接近
- 可以自然扩展现有 sender-note + latent 的 pipeline
- 一旦 split 设计得当，结论会比现在的“单边看全文再压缩”更有说服力

### 6.2 方向 B：需要传递不确定性或软信号的任务

这是最能突出 `latent > token` 优势的一类任务，但通常 benchmark 没那么标准，往往需要自己做改造。

核心问题是：

- 有些观察不是一个清晰的离散事实
- 而是一个带噪声、带不确定性、或带分布性质的 soft signal
- 文本很难把这种软信息完整表达出来
- latent 则可能天然更适合保留这些连续信息

#### 6.2.1 Probabilistic Inference Collaboration

推荐设定：

- Agent A 是 observer
  - 看到带噪声的证据
- Agent B 是 reasoner
  - 基于 A 传来的信息完成推理

对比方式：

- text communication：A 只能把观察离散化成一句话
- latent communication：A 可以直接传带有软结构的表示

可选 benchmark 来源包括：

- BIG-bench 里的 probabilistic reasoning 子集
- 或者自己构造带噪观察任务

这类任务的价值在于：

- 能更直接测试 latent 是否能传 `uncertainty`
- 比“只传一句 evidence note”更容易体现连续表征的独特性

#### 6.2.2 MNIST-with-noise / 模糊观察分类协作

这是一个更工程化但很适合做机制验证的方向。

基本设定：

- Agent A 看到带噪声、模糊或部分遮挡的输入
- Agent B 负责最终分类决策
- A 的任务是把观察结果传给 B

研究价值：

- token communication 往往只能传离散描述
- latent communication 可能更容易保留软分类边界或模糊视觉线索

虽然它不像 QA benchmark 那样“自然语言友好”，但很适合作为：

- 机制性实验
- latent 是否能保留 soft posterior 的直接检验

#### 6.2.3 Ambiguity Resolution

推荐设定：

- Agent A 看到一个有歧义的输入，以及消歧线索
- Agent B 只看到原始歧义文本，需要完成下游任务
- A 的任务是传达“应该如何 resolve 这个歧义”

例如：

- 一句有歧义的话 + 一张可以消歧的图片
- 或带歧义的问题 + 外部上下文

这一方向的价值在于：

- resolution 往往不是硬选择，而是 soft preference
- latent communication 更可能保留这种“偏向某种解释”的细微信号

### 6.3 方向 C：Dense Continuous Collaboration

这是研究价值最高的一条方向，也和 `SRMT / POGEMA` 的延伸最契合。

与 QA 类 benchmark 不同，这里通信不再是一次性的，而是：

- agent 需要在每一步持续交换信息
- latent communication 可以成为真正的“连续 side-channel”

这类任务最适合回答：

> latent communication 是否在持续交互和实时决策中比文本更自然、更高效？

#### 6.3.1 Hanabi

Hanabi 是多智能体 RL 最经典的协作 benchmark 之一。

它的特点是：

- 每个 agent 只能看到别人的牌，不能看到自己的牌
- 必须通过有限沟通建立共享信念
- 规则本身就天然围绕“隐式信息传递”设计

为什么非常适合 latent communication：

- 传统 communication 在 Hanabi 中受规则限制
- latent 通道能不能成为“规则外的高带宽协调接口”，是一个很有研究价值的问题

这条线的官方 benchmark 可以参考：

- `The Hanabi Challenge (Bard et al.)`

#### 6.3.2 Overcooked

Overcooked 是双人或多人实时协作任务中的代表。

核心特点：

- 两个 agent 要持续协调角色、路径和时机
- 决策是实时的、连续的
- 错一步可能导致整个流程紊乱

它的价值在于：

- 很适合研究“latent 是否能辅助即时协调”
- 比静态 QA 更贴近真实协作

#### 6.3.3 PettingZoo Cooperative Environments

例如：

- `knights-archers-zombies`
- `cooperative pong`

它们的优点是：

- 开源实现成熟
- 上手成本低
- 很容易快速做通信接口替换

但缺点也很明显：

- 任务复杂度通常不够高
- 不一定能体现 latent communication 的真正独特价值

因此更适合作为：

- 快速原型验证
- 通信接口工程测试

而不是最终主 benchmark。

#### 6.3.4 POGEMA / SRMT 风格网格世界

这是与当前项目研究兴趣最贴近的一块。

推荐原因：

- 你已经熟悉这一类环境
- 可以直接基于现有 SRMT/POGEMA 风格设定扩展
- 最容易做出“partial observability + continuous coordination + communication necessity”三者结合的 benchmark

具体推荐做法包括：

- 保持局部观测，限制单个 agent 的视野
- 让任务必须依赖多 agent 协作完成
- 在每个时间步引入可学习的 text / latent communication channel
- 直接比较：
  - no communication
  - text-only
  - latent-only
  - text+latent

这是最适合做“SRMT 延伸版 latent communication 主实验”的方向。

## 7. 推荐优先级

如果从“成熟度、实现成本、研究价值”三者综合排序，推荐优先级大致如下：

### 第一优先级

- `Collaborative Reasoning with Split Information`
  - 尤其是 `MuSiQue / 2WikiMultihopQA / FanOutQA` 的人工 split 版本
  - 原因：与当前 QA 主线最连续、最容易复用现有代码

### 第二优先级

- `POGEMA / SRMT 风格网格世界`
  - 原因：最能形成你自己的主场 benchmark，也最适合研究连续 latent 协作

### 第三优先级

- `Hanabi`
  - 原因：研究价值高，但工程改造成本会更大

### 第四优先级

- `Probabilistic Inference Collaboration`
  - 原因：最能体现 latent 优势，但需要较多自定义数据和评测设计

### 第五优先级

- `Colors in Context / Referential Game / PettingZoo cooperative`
  - 原因：适合快速原型与验证，但不一定最能支撑最终论文主线

