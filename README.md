# A calibrated encoder classifier as the cost floor for production decisions

**English** · [中文](README.zh.md)

Technical report for **`size-decision-general`**, a small, non-generative,
calibrated decision model. Model weights:
<https://huggingface.co/sizeai/size-decision-general> ·
<https://www.modelscope.cn/models/sizeai/size-decision-general>

---

## Abstract

Most production NLP spend goes to classification — content moderation, ticket
triage, intent routing, relevance gating — while the attention goes to
generative models. Generative decoders are accurate but slow and, in practice,
overconfident: they report no calibrated probability, and the large
"decider"-style classifiers (9–27B parameters) generally do not either. This
report describes a 307M-parameter encoder model that answers option-list
questions in a single forward pass and returns a **calibrated** distribution over
the options (ECE under 0.06 on three public classification benchmarks). It is
positioned not against generative decoders on open-ended reasoning, but as the
cheap, deterministic, high-throughput layer underneath them, where calibrated
confidence is what lets a system gate on probability rather than only on the
argmax.

## 1. Motivation

- **Classification is the volume layer.** Content moderation, routing, triage
  and filtering are high-frequency, low-marginal-value-per-item tasks.
- **Decoders cost ~20× more per item.** A 2B instruction model runs at tens of
  samples/s; a fine-tuned encoder runs at hundreds. At a million items a day that
  gap is a board-level cost decision, not a micro-optimisation.
- **Confidence is usually missing.** Generative decoders emit text, not a
  probability; large decider classifiers rarely publish a calibration figure.
  Systems that want to "act only when confident" are left to threshold a raw
  score that was never calibrated.

A small encoder with a calibrated head addresses exactly that seam.

## 2. Model

| | |
|---|---|
| Encoder | `jhu-clsp/mmBERT-base`, 307M parameters (22 layers, hidden 768) |
| Heads | 2-layer decision head + scoring heads + action head + question-type embedding + calibration tensor |
| Inference | **non-autoregressive**; one forward pass returns the full distribution over the option list |
| Options | tested up to **151** (`head_max_len` = 1024) |
| Question types | `choice` (mutually exclusive), `score` (ordinal rating → expected level), `noul` (answerable / not) |

`choice` returns an argmax plus a full probability vector over the options.
`score` returns the expected value of a distribution over an **ordered** level
list (a single graded number). `noul` returns a probability that the question is
answerable. Because the answer is a single forward pass, confidence is free.

## 3. Method

- **Objective**: reinforcement learning against strictly proper scoring rules
  (RLCD) combined with soft-target cross-entropy, so the head is rewarded for a
  distribution rather than only for the argmax.
- **Calibration**: temperature scaling fitted **per option-count bucket**
  rather than globally, because the optimal temperature changes with the number
  of options.
- **Data**: five single-label classification datasets plus a set of
  contract-clause `score`/`noul` tasks, with a held-out split of whole contracts
  for the non-`choice` heads.

## 4. Results

### 4.1 `choice` benchmarks

Every row is a full run on the published test split, not a sample.

| Benchmark | Options | n | Accuracy | ECE | Brier | p50 latency |
|---|---|---|---|---|---|---|
| Banking77 (intent) | 77 | 3080 | **0.9114** | 0.0293 | 0.1416 | 13.3 ms |
| AG News (topic) | 4 | 7600 | **0.9208** | 0.0226 | 0.1299 | 3.8 ms |
| DAIR (emotion) | 6 | 2000 | **0.8555** | 0.0338 | 0.2197 | 3.9 ms |
| CLINC150 (intent + OOS) | 151 | 5500 | 0.8387 | 0.0532 | 0.2517 | 23.1 ms |
| GoEmotions (emotion) | 28 | 5427 | 0.5646 | 0.0177 | 0.5954 | 8.5 ms |

Global ECE stays under 0.06 while the option count spans 4 to 151. Accuracy is
**not** monotonic in option count: the 28-option cell (GoEmotions) is the weakest
because its labels are near-synonymous, not because 28 options is hard. CLINC150's
out-of-scope slice scores 0.8667 on 30 samples (95% Wilson interval ≈ 0.70–0.95),
too small to quote as a rejection rate.

### 4.2 `score` and `noul`

Measured on the task each head was trained for, with a leakage-free holdout of
whole contracts the model never saw.

| Type | Task | Levels | Train fit | Holdout |
|---|---|---|---|---|
| `score` | clause fairness | 4 | r 0.959 | r **0.940** (en) / **0.942** (zh) |
| `score` | document maturity | 4 | r 0.999 | r **0.999** |
| `score` | urgency level | 4 | r 0.941 | r **1.000** |
| `noul` | risky / one-sided clause | 2 | AUC 0.999 | AUC **0.977** (en) / **0.979** (zh) |
| `noul` | needs legal review | 2 | AUC 0.998 | AUC **0.984** |

Clause fairness loses ~2 points out of sample (0.96 → 0.94) and the binary tasks
~1.5 points, so the heads generalise rather than memorise. The synthetic holdouts
for document maturity and urgency are small (n = 49–56) and are directional;
clause fairness and the `noul` tasks have n = 500 held-out records each.

### 4.3 Measurement caveats (read before comparing)

- **These are fine-tuned results, not zero-shot.** The train split of each
  `choice` benchmark is part of this model's training mix. Numbers here are not
  comparable to zero-shot leaderboards.
- **The `choice` instructions used in evaluation are Chinese**, matching the
  training language; the model is not verified as instruction-language-agnostic.
- **Calibration is post-hoc**, fitted on the evaluation distributions, so ECE is
  not from independent held-out calibration data.

## 5. Cost and latency

| Regime | Representative latency (per item) |
|---|---|
| This model (`choice`) | **3.8–23.1 ms** p50, single forward pass |
| 4–12B generative decoder | ~0.5–2 s (order of magnitude) |

The point is architectural, not a head-to-head run: a non-generative encoder
returns a full calibrated distribution in one pass at a fraction of the cost of
decoding tokens, and can run on CPU. For high-volume classification this is the
relevant comparison; the existing p50 latencies above are cited from the model's
own evaluation and no new throughput benchmark is reported here.

## 6. Limitations

- **License**: the tokenizer derives from Gemma 2, so the model is a Gemma *Model
  Derivative*. Commercial use and redistribution are permitted subject to the
  Gemma Terms of Use; there is no non-commercial restriction, but organisations
  with a policy against Gemma-derived artefacts should review before adoption.
- **English `choice` data**, so broad multilingual capability is unverified.
- **`score`/`noul` heads are task-specific**; pointing them at a different graded
  or binary question is cross-task transfer and is expected to be weak.
- **Option lists beyond 151 are untested**, and accuracy is non-monotonic in
  option count.
- **No generative ability**: this is a discriminator over a fixed option list.

## 7. Reproducibility

The weights, tokenizer, runtime config and a self-contained loader are in the
model repositories. Loading does not require any project code beyond the
packaged `decision_model.py`:

```python
from decision_model import DecisionModel

model = DecisionModel.load(hub="hf")        # or hub="modelscope"
answer = model.predict(
    "I was charged twice for order #4471.",
    {"pick": {"type": "choice", "instructions": "Choose the issue category",
              "criteria": {"billing": "billing", "shipping": "shipping"}}},
)
print(answer["answers"]["pick"]["choice"], answer["answers"]["pick"]["probabilities"])
```

## 8. Citation

```bibtex
@techreport{sizeai2026calibrated,
  title       = {A calibrated encoder classifier as the cost floor for production decisions},
  author      = {{sizeai}},
  year        = {2026},
  institution = {sizeai},
  url         = {https://github.com/beidald/size-decision-general},
  note        = {Technical report for the size-decision-general model.}
}

@misc{sizeai2026sizedecisiongeneral,
  title        = {size-decision-general: a calibrated decision model},
  author       = {{sizeai}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sizeai/size-decision-general}},
  note         = {Based on Laya (Apache-2.0) and mmBERT-base (MIT); tokenizer
                  derived from Gemma 2, subject to the Gemma Terms of Use.}
}
```

## References

- Laya — <https://github.com/NandhaKishorM/laya>
- mmBERT — <https://huggingface.co/jhu-clsp/mmBERT-base>
- Gemma Terms of Use — <https://ai.google.dev/gemma/terms>
