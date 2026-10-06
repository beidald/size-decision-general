# Milestones

A dated record of what shipped and when. Each entry is immutable; new entries are
appended.

---

## 2026-10-06 — v1, first public release

### The moment

`size-decision-general` is public on three platforms at once. This is the first
model released under this effort: a small, calibrated, non-generative decision
model.

| Platform | URL |
|---|---|
| Hugging Face | <https://huggingface.co/sizeai/size-decision-general> |
| ModelScope | <https://modelscope.cn/models/eastliu/size-decision-general> |
| GitHub (this report) | <https://github.com/beidald/size-decision-general> |

### What shipped

- A 307M-parameter, non-autoregressive decision model: an `mmBERT-base` encoder
  with decision heads, answering three question types (`choice`, `score`,
  `noul`) in a single forward pass and returning a full distribution over the
  option list.
- Up to 151 options per question.
- Calibrated output (global ECE under 0.06 across option counts 4–151).
- Weights, tokenizer, runtime config, a self-contained `decision_model.py`
  loader, and complete licensing files on both model hubs; the technical report
  (bilingual) here on GitHub.

### Headline results

`choice`, full runs on the published test splits:

| Benchmark | Options | n | Accuracy | ECE | Brier | p50 |
|---|---|---|---|---|---|---|
| Banking77 | 77 | 3080 | 0.9114 | 0.0293 | 0.1416 | 13.3 ms |
| AG News | 4 | 7600 | 0.9208 | 0.0226 | 0.1299 | 3.8 ms |
| DAIR | 6 | 2000 | 0.8555 | 0.0338 | 0.2197 | 3.9 ms |
| CLINC150 | 151 | 5500 | 0.8387 | 0.0532 | 0.2517 | 23.1 ms |
| GoEmotions | 28 | 5427 | 0.5646 | 0.0177 | 0.5954 | 8.5 ms |

`score` / `noul`, training fit vs a leakage-free holdout of whole contracts:

| Type | Task | Fit | Holdout |
|---|---|---|---|
| `score` | clause fairness | r 0.959 | r 0.940 (en) / 0.942 (zh) |
| `score` | document maturity | r 0.999 | r 0.999 |
| `score` | urgency level | r 0.941 | r 1.000 |
| `noul` | risky clause | AUC 0.999 | AUC 0.977 (en) / 0.979 (zh) |
| `noul` | needs legal review | AUC 0.998 | AUC 0.984 |

### How it got here — the honest path

The result people remember is the model; the part worth recording is what it took
to report it truthfully.

- An early build raised `score`/`noul` from ~3% to 35% of the training mix and
  the probe still showed them near-random. The first conclusion — "these heads do
  not work" — turned out to be **a bug in the probe, not the model**: the probe
  asked a CLINC banking-intent question while the heads were trained on
  contract-clause judgments, and a second bug read a nonexistent probability key
  and returned a constant 0.5 AUC. Probing the tasks the heads were actually
  trained for showed them working, and a leakage-free holdout showed them
  **generalising** (clause fairness r 0.94 on unseen contracts).
- The holdout also exposed a **replay leak**: the first attempt left the withheld
  contracts in the replay pool, which would have sampled them back into training.
  The holdout is now physically removed from the pool and guarded by a filter.
- Every headline number is labelled for what it is: these are **fine-tuned**
  results (the benchmark train splits are in the training mix) under **Chinese
  instructions**, not zero-shot. That caveat is in the report, the model cards and
  here, on purpose.

The principle recorded with v1: measure the thing you claim, and when the
measurement disagrees with the conclusion, fix the measurement before the story.

### Licensing

- The tokenizer derives from Gemma 2, so the model is a Gemma *Model Derivative*.
- **Commercial use and redistribution are permitted** subject to the Gemma Terms
  of Use; the terms are reproduced verbatim in `GEMMA_TERMS_OF_USE.md` in both
  model repositories, with the required notice and the modified-file list.
- The upstream framework is used under Apache-2.0 and the encoder under MIT;
  attributions are in `NOTICE`.

### Repository contents

Each model repository (HF, ModelScope) contains: `model.safetensors` (643.8 MB),
`encoder/config.json`, three `tokenizer/` files, `rl_agent_config.json`,
`checkpoint_meta.json`, `decision_model.py`, `requirements.txt`, `README.md`,
`LICENSE`, `NOTICE`, `GEMMA_TERMS_OF_USE.md`, plus (`FILES.md`, and
`configuration.json` on ModelScope, `.gitattributes` on Hugging Face).

This repository contains: `README.md`, `README.zh.md` (the report),
`CITATION.cff`, `LICENSE`, `MILESTONES.md` (this file).

### Reproduce

```python
from decision_model import DecisionModel

model = DecisionModel.load(hub="hf")          # or hub="modelscope"
answer = model.predict(
    "I was charged twice for order #4471.",
    {"pick": {"type": "choice", "instructions": "Choose the issue category",
              "criteria": {"billing": "billing", "shipping": "shipping"}}},
)
print(answer["answers"]["pick"]["choice"])
```

### Known limitations at v1

- Fine-tuned, not zero-shot; the benchmark train splits are in the training mix.
- `choice` training data is English; broad multilingual ability is unverified.
- `score`/`noul` are task-specific; cross-task transfer is weak.
- Calibration is post-hoc (fitted on the evaluation distributions).
- Option lists beyond 151 are untested.

### Next

- An independent calibration set so ECE is not measured on the fitting data.
- ONNX/CPU throughput numbers to make the cost argument concrete.
- A held-out benchmark (train split excluded) for a genuine out-of-domain number.
- Broader language coverage.

---

## 2026-10-06 — v1, 首次公开发布（中文）

### 这一刻

`size-decision-general` 已在三个平台同时公开。这是本项目发布的**第一个模型**：一个小体积、可校准、非生成式的决策模型。

| 平台 | 地址 |
|---|---|
| Hugging Face | <https://huggingface.co/sizeai/size-decision-general> |
| ModelScope | <https://modelscope.cn/models/eastliu/size-decision-general> |
| GitHub（本报告） | <https://github.com/beidald/size-decision-general> |

### 发布内容

- 307M 参数、非自回归决策模型：`mmBERT-base` 编码器 + 决策头，单次前向回答 `choice`/`score`/`noul` 三类问题，返回选项列表上的完整分布。
- 单个问题最多 151 个选项。
- 输出经过校准（选项数 4–151 范围内全局 ECE 低于 0.06）。
- 两个模型平台均含权重、tokenizer、运行时配置、自包含的 `decision_model.py` 加载器与完整许可文件；双语技术报告在本 GitHub 仓库。

### 主要结果

`choice`（公开测试集完整运行）：Banking77 0.9114 / AG News 0.9208 / DAIR 0.8555 / CLINC150 0.8387 / GoEmotions 0.5646。

`score`/`noul`（训练拟合 vs 整份合约留出）：条款公平性拟合 r 0.959、留出 r 0.940（英）/0.942（中）；高风险条款 AUC 0.999 → 留出 0.977/0.979；需法务审核 AUC 0.998 → 留出 0.984。

### 过程（诚实记录）

人们记住的是模型；值得记录的是**如实报告它所付出的代价**。

- 早期版本把 `score`/`noul` 占比从 ~3% 提到 35%，探针仍显示接近随机。最初的结论"这两个头不可用"被证明是**探针的 bug，而非模型**：探针问的是 CLINC 银行意图，而这两个头训练在合约条款判断上；另一处 bug 读了不存在的概率键，恒返回 0.5。改用训练任务本身探测后，它们工作正常；无泄漏留出进一步显示它们**能泛化**（未见合约上条款公平性 r 0.94）。
- 留出集还暴露了一处 **replay 泄漏**：首次实现把留出合约留在了 replay 池里，会被重新采样进训练。现在留出集已从池中物理移出，并加了过滤保护。
- 所有主数字都标注了真实口径：**微调后**、**中文指令**，**非零样本**。这一点在报告、模型卡和此处都明确写出，是有意为之。

随 v1 记录的原则：**测量你所声称的东西；当测量与结论不符时，先修测量，再改故事。**

### 许可

- tokenizer 派生自 Gemma 2，故为 Gemma *Model Derivative*。
- **允许商用与再分发**，受 Gemma 条款约束；条款逐字收录在两个模型仓库的 `GEMMA_TERMS_OF_USE.md`，并附必需 notice 与修改文件清单。
- 上游框架 Apache-2.0、编码器 MIT，归属见 `NOTICE`。

### 已知局限

微调而非零样本；`choice` 数据为英文；`score`/`noul` 任务特定；校准为后验拟合；超过 151 选项未测试。

### 下一步

独立校准集 / ONNX-CPU 吞吐 / 留出榜单的域外数字 / 更广语言覆盖。
