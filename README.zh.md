# 作为生产决策成本下限的校准编码器分类器

[English](README.md) · **中文**

**`size-decision-general`** 技术报告——一个小体积、非生成式、可校准的决策模型。权重地址：
<https://huggingface.co/sizeai/size-decision-general> ·
<https://www.modelscope.cn/models/eastliu/size-decision-general>

---

## 摘要

生产环境中的 NLP 开销大多花在分类上——内容审核、工单分派、意图路由、相关性过滤——而注意力都给了生成式模型。生成式解码器准确但慢，且在实践中有过自信倾向：它们不提供校准概率，而 9–27B 参数级别的大型"决策"分类器通常也不提供。本报告描述一个 307M 参数的编码器模型：它以单次前向回答选项列表式问题，并返回**经过校准**的选项概率分布（在三个公开分类基准上 ECE 低于 0.06）。它的定位不是与生成式解码器在开放式推理上竞争，而是作为其下方那层**廉价、确定、高吞吐**的能力层——在这里，"校准置信度"让系统可以按概率而非仅按 argmax 做门控。

## 1. 动机

- **分类是跑量层。** 内容审核、路由、分派、过滤都是高频、单项价值极低的任务。
- **解码器单项成本高约 20 倍。** 2B 级指令模型每秒几十条，微调编码器每秒数百条。一天百万条规模下，这个差距是董事会级别的成本决策，而不是微优化。
- **置信度普遍缺位。** 生成式解码器输出文本而非概率；大型决策分类器极少公布校准指标。想做"仅在高置信时执行"的系统，只能对一个从未校准的原始分数硬套阈值。

一个小编码器加上校准头，正对着这个缝隙。

## 2. 模型

| | |
|---|---|
| 编码器 | `jhu-clsp/mmBERT-base`，307M 参数（22 层，hidden 768） |
| 决策头 | 2 层决策头 + 打分头 + 动作头 + 题型嵌入 + 校准张量 |
| 推理 | **非自回归**；单次前向返回选项列表上的完整分布 |
| 选项数 | 实测最多 **151**（`head_max_len` = 1024） |
| 题型 | `choice`（互斥）、`score`（有序评级→期望等级）、`noul`（可答/不可答） |

`choice` 返回 argmax 加完整概率向量。`score` 返回有序等级列表上分布的期望值（单个分级数值）。`noul` 返回"该问题可被回答"的概率。因为答案只需一次前向，置信度是免费得到的。

## 3. 方法

- **目标**：针对严格适当评分规则（RLCD）的强化学习 + 软目标交叉熵，使模型因"分布"获益，而非仅因 argmax。
- **校准**：**按选项数分桶**拟合温度，而非全局单一温度，因为最优温度随选项数变化。
- **数据**：五个单标签分类数据集，加一组合约条款上的 `score`/`noul` 任务；非 `choice` 头使用按整份合约切分的留出集。

## 4. 结果

### 4.1 `choice` 基准

每行都是对公开测试集的完整运行，不是抽样。

| 基准 | 选项数 | n | 准确率 | ECE | Brier | p50 延迟 |
|---|---|---|---|---|---|---|
| Banking77（意图） | 77 | 3080 | **0.9114** | 0.0293 | 0.1416 | 13.3 ms |
| AG News（主题） | 4 | 7600 | **0.9208** | 0.0226 | 0.1299 | 3.8 ms |
| DAIR（情绪） | 6 | 2000 | **0.8555** | 0.0338 | 0.2197 | 3.9 ms |
| CLINC150（意图 + OOS） | 151 | 5500 | 0.8387 | 0.0532 | 0.2517 | 23.1 ms |
| GoEmotions（情绪） | 28 | 5427 | 0.5646 | 0.0177 | 0.5954 | 8.5 ms |

选项数从 4 到 151，全局 ECE 始终低于 0.06。准确率对选项数**并非单调**：28 选项格（GoEmotions）最弱，原因是标签高度近义，而非 28 选项本身难。CLINC150 的 out-of-scope 子集为 0.8667（30 条样本，95% Wilson 区间约 0.70–0.95），样本太小，不宜作为拒识率引用。

### 4.2 `score` 与 `noul`

在各自被训练的任务上测量，并使用**模型从未见过的整份合约**作为无泄漏留出集。

| 类型 | 任务 | 等级数 | 训练拟合 | 留出 |
|---|---|---|---|---|
| `score` | 条款公平性 | 4 | r 0.959 | r **0.940**（英）/ **0.942**（中） |
| `score` | 文档成熟度 | 4 | r 0.999 | r **0.999** |
| `score` | 紧迫程度 | 4 | r 0.941 | r **1.000** |
| `noul` | 高风险/不对等条款 | 2 | AUC 0.999 | AUC **0.977**（英）/ **0.979**（中） |
| `noul` | 需法务审核 | 2 | AUC 0.998 | AUC **0.984** |

条款公平性样本外仅掉约 2 个点（0.96 → 0.94），二分类任务约 1.5 个点，说明这些头是**泛化**而非记忆。文档成熟度与紧迫度的合成留出样本较小（n = 49–56），仅有方向性意义；条款公平性与两个 `noul` 任务各有 n = 500 条留出记录。

### 4.3 口径说明（比较前必读）

- **这些是微调结果，不是零样本。** 各 `choice` 基准的 train split 都在本模型训练混集中，**不可与零样本榜单直接比较**。
- **评测使用的是中文指令**，与训练语言一致；未验证模型对指令语言不敏感。
- **校准为后验拟合**，在评测分布上完成，故 ECE 并非来自独立的留出校准数据。

## 5. 成本与延迟

| 方案 | 单项延迟（代表值） |
|---|---|
| 本模型（`choice`） | **3.8–23.1 ms** p50，单次前向 |
| 4–12B 生成式解码器 | 约 0.5–2 s（数量级） |

这是**架构性**对比，而非同台实测：非生成式编码器一次前向即得到完整的校准分布，成本远低于逐 token 解码，且可跑在 CPU 上。对高频分类，这才是相关的比较维度。上表 p50 引用自模型自身评测，本报告不新增吞吐基准。

## 6. 局限

- **许可**：tokenizer 派生自 Gemma 2，故本模型是 Gemma *Model Derivative*。**商业使用与再分发被允许**，但受 Gemma 条款约束；不存在非商用限制，但对 Gemma 系资产有政策限制的组织应在采用前评估。
- **`choice` 数据为英文**，广泛多语言能力未验证。
- **`score`/`noul` 头是任务特定的**；把它们指向不同的分级或二值问题属于跨任务迁移，预期较弱。
- **超过 151 的选项列表未测试**，且准确率对选项数非单调。
- **无生成能力**：这是固定选项列表上的判别器。

## 7. 复现

权重、tokenizer、运行时配置与一个自包含加载器都在模型仓库中。加载不需要任何项目代码，只需包内的 `decision_model.py`：

```python
from decision_model import DecisionModel

model = DecisionModel.load(hub="hf")        # 或 hub="modelscope"
answer = model.predict(
    "订单 #4471 被重复扣款了。",
    {"pick": {"type": "choice", "instructions": "选择问题类别",
              "criteria": {"billing": "账单", "shipping": "物流"}}},
)
print(answer["answers"]["pick"]["choice"], answer["answers"]["pick"]["probabilities"])
```

## 8. 引用

```bibtex
@techreport{sizeai2026calibrated,
  title       = {A calibrated encoder classifier as the cost floor for production decisions},
  author      = {{sizeai}},
  year        = {2026},
  institution = {sizeai},
  url         = {https://github.com/beidald/size-decision-general},
  note        = {size-decision-general 模型的技术报告。}
}

@misc{sizeai2026sizedecisiongeneral,
  title        = {size-decision-general: a calibrated decision model},
  author       = {{sizeai}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sizeai/size-decision-general}},
  note         = {基于 Laya (Apache-2.0) 与 mmBERT-base (MIT)；tokenizer 派生自
                  Gemma 2，受 Gemma Terms of Use 约束。}
}
```

## 参考

- Laya — <https://github.com/NandhaKishorM/laya>
- mmBERT — <https://huggingface.co/jhu-clsp/mmBERT-base>
- Gemma Terms of Use — <https://ai.google.dev/gemma/terms>
