# MANEC — Lymph-Node Metastasis & NEC Component Prediction

Path-feature binary classification pipeline for **MANEC cases**, running **two
independent prediction tasks sequentially**, with **multi-model comparison** on
each task:

| Task key | Task | Label column (default) | Cohort restriction | Main model |
|---|---|---|---|---|
| `ln_metastasis` | Lymph-node metastasis (yes/no) | `LN metastasis` | all MANEC cases | ElasticNet LR |
| `nec_component` | NEC component present (yes/no) | `NEC component` | metastasis-positive cases only | ElasticNet LR |

All paths, label files and cohort filters are configurable — see
[Configuration](#configuration).

Global random seed: **2026** — results are fully reproducible.

## ⚠️ Data & Privacy Notice

- This repository contains **no patient data**, **no label files**, and **no
  model weights**.
- The label Excel files (`SL-*.xlsx`, `FY-*.xlsx`) and the feature `.npz` files
  **must not** be committed to GitHub.
- Users are responsible for ensuring that any data they process is handled in
  accordance with applicable ethics and privacy regulations.

## Installation

System dependency (only needed if you want CJK fonts in plots):

```bash
# Ubuntu / Debian
sudo apt install fonts-noto-cjk