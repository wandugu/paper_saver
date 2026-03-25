# SAVER: Selective Visual Evidence Routing for Multimodal NER/RE

> The default model in this repository is **SAVER-SIS** (main model), with an optional **SAVER-RL (RES)** extension.

## 1. Method Overview

SAVER follows three core principles:
- Use vision only when the current entity (MNER) or marked entity pair (MRE) is likely to be visually groundable.
- When vision is activated, acquire only a small and complementary multi-image evidence set.
- Use a unified scoring head to combine text and optional visual evidence.

The full pipeline has four stages:
1. **Encoding**: text encoder produces token/span representations; vision encoder first produces global image vectors.
2. **CGG (Conformal Groundability Gate)**: unit-level routing decision for whether visual evidence is needed.
3. **Evidence Constructor (SIS or RES) + Set Transformer**: after activation, select up to K images and aggregate region evidence.
4. **Energy-Inspired Joint Scoring**: unified scoring for MNER (span/type) or MRE (relation).

---

## 2. Task Settings

### MNER
- Input: text + image set.
- Output: entity spans and entity types.
- SAVER performs CGG and evidence selection at candidate-span granularity.

### MRE
- Input: text with a marked head/tail entity pair + image set.
- Output: relation label of the marked pair.
- SAVER performs pair-level routing and evidence selection (pair gate is derived from entity gates by default).

---

## 3. Key Components

### 3.1 CGG (Conformal Groundability Gate)
- Computes groundability score `g(u)` using global image vectors and text-image similarity statistics.
- Uses threshold-based hard routing at inference: `γ(u)=1[g(u)≥τ]`.
- Chooses `τ` on a calibration split via a Clopper-Pearson upper-bound constraint under target risk `α`.

### 3.2 SIS (Submodular Image Selector, default)
- When gate is active, selects at most K images from N candidates.
- Objective balances relevance and coverage/diversity.
- Uses greedy approximation (`1-1/e`) for efficient selection.

### 3.3 RES (Reinforced Evidence Selector, optional)
- Formulates evidence acquisition as sequential decision making with a STOP action.
- Uses cost-aware rewards with CGG-based action masking.
- Reported as an extension in ablations; SIS remains the default model in main tables.

### 3.4 Energy-Inspired Joint Scoring
- Uses standard cross-entropy training (energy notation is used for unified formulation).
- Combines task score, text-vision consistency term, and gate sparsity term.

---

## 4. Experiments

### 4.1 Datasets
- **MRE**: MNRE, MRE-MI
- **MNER**: Twitter-2015, Twitter-2017, MNER-MI, MNER-MI-Plus

Core dataset scales used in the main text:

| Dataset | Task | Train / Dev / Test | Avg. images |
|---|---|---|---|
| MNRE (v2) | RE | 12,247 / 1,624 / 1,614 | 1.00 |
| MRE-MI | RE | 13,504 / 4,500 / 4,500 | 2.80 |
| Twitter-2017 | MNER | 3,373 / 723 / 723 | 1.00 |
| MNER-MI-Plus | MNER | 10,229 / 1,583 / 1,583 | 2.15 |

### 4.2 Metrics
- MRE: micro-F1 / Precision / Recall
- MNER: strict entity-level F1 (boundary + type)
- Selectivity: Risk-Activation-Coverage curves, AURC, ActCov@0.10
- Efficiency: FLOPs/sample, end-to-end P90 latency

---

## 5. Main Results (SAVER)

### 5.1 MRE-MI Main Benchmark

| Method | P↑ | R↑ | F1↑ | AURC↓ | ActCov@0.10↑ | FLOPs (G/sample)↓ | P90 (ms)↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ModernBERT-only | 82.37 | 79.84 | 81.09 | 0.147 | 0.68 | 13 | 17 |
| DeBERTa-v3-only | 81.53 | 79.48 | 80.49 | 0.153 | 0.66 | 18 | 27 |
| HVPNeT | 73.87 | 76.82 | 75.32 | 0.168 | 0.63 | 66 | 99 |
| RSRNeT | 84.78 | 83.06 | 83.89 | 0.129 | 0.74 | 60 | 90 |
| All-Images Attn. | 83.47 | 82.18 | 82.82 | 0.142 | 0.72 | 62 | 93 |
| Top-K by relevance | 85.31 | 83.62 | 84.45 | 0.119 | 0.77 | 51 | 77 |
| GLRA | 85.23 | 83.81 | 84.51 | 0.117 | 0.78 | 56 | 84 |
| Retrieval-Aug. | 84.27 | 82.86 | 83.56 | 0.124 | 0.75 | 55 | 82 |
| **SAVER (full)** | **85.93** | **84.57** | **85.24** | **0.104** | **0.82** | **36** | **54** |
| SAVER w/o CGG | 84.46 | 83.18 | 83.81 | 0.124 | 0.76 | 51 | 77 |
| SAVER w/o SIS | 84.13 | 82.74 | 83.43 | 0.136 | 0.74 | 62 | 93 |
| SAVER w/o J.Score | 85.28 | 84.12 | 84.70 | 0.111 | 0.80 | 37 | 56 |
| SAVER (CGG+SIS) | 85.14 | 84.33 | 84.73 | 0.107 | 0.81 | 35 | 53 |

### 5.2 Additional Benchmarks (F1)

| Dataset | Strong baseline | SAVER | Gain |
|---|---:|---:|---:|
| MNRE | 83.9 (RSRNeT) | **84.7** | +0.8 |
| Twitter-2015 | 76.5 (RSRNeT) | **77.0** | +0.5 |
| Twitter-2017 | 87.9 (RSRNeT) | **88.0** | +0.1 |
| MNER-MI | 76.9 (GLRA-adapt) | **77.3** | +0.4 |
| MNER-MI-Plus | 83.1 (GLRA-adapt) | **83.7** | +0.6 |

### 5.3 CGG Calibration (α=0.10, 1-δ=0.95)

| Dataset | Act. Coverage | Emp. Error | CP Upper |
|---|---:|---:|---:|
| MRE-MI | 0.82 | 0.087 | 0.097 |
| MNRE | 0.82 | 0.083 | 0.098 |
| Twitter-2017 | 0.80 | 0.077 | 0.099 |
| MNER-MI-Plus | 0.81 | 0.084 | 0.099 |

---

## 6. Environment Setup

```bash
pip install -r requirements.txt
```

---

## 7. Data Preparation

### Twitter2015 / Twitter2017
Place data under `data/NER_data`, or update data paths in `run.py`.

### MNRE
Use links in the project documentation or prepare data with the expected directory structure.

> Note: SAVER supports both single-image and multi-image inputs. In multi-image settings, SIS/RES builds compact evidence subsets only when gate activation occurs.

---

## 8. Training and Testing

### NER
```bash
bash run_twitter15.sh
bash run_twitter17.sh
```

### RE
```bash
bash run_re_task.sh
```

### Test-only Example
```bash
python -u run.py \
  --dataset_name="MRE" \
  --bert_name="bert-base-uncased" \
  --seed=1234 \
  --only_test \
  --max_seq=80 \
  --use_prompt \
  --prompt_len=4 \
  --sample_ratio=1.0 \
  --load_path='your_re_ckpt_path'
```

---

## 9. Notes

- This README now presents **SAVER** as the primary method and main result set.
- If updated SAVER architecture/risk-coverage figures are available, replace placeholders in this README accordingly.

## 10. Acknowledgement

- Twitter15/Twitter17 data processing follows UMT.
- MNRE data sourcing follows MEGA.
- Early implementation inspirations include HVPNeT.
