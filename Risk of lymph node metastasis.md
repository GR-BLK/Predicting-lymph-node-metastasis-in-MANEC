# MANEC — Lymph-Node Metastasis Count & NEC Metastasis Count Risk Grouping

Path-feature binary classification pipeline for **MANEC cases**, running **two
independent risk-grouping tasks sequentially**, with **multi-model comparison**
on each task:

| Task key | Task | Label column | Binarization | Cohort restriction | K folds | Main model |
|---|---|---|---|---|---|---|
| `ln_metastasis_count` | LN metastasis **count** risk group | `LN metastasis count` | `>= 5` → high risk | all MANEC cases | 10 | **LightGBM** |
| `nec_metastasis_count` | NEC LN metastasis **count** risk group | `NEC LN metastasis count` | `>= 3` → high risk | `LN metastasis == 1` only | 5 | **RandomForest** |

Both tasks use the **`youden`** threshold rule (maximize Sens + Spec − 1) for
consistent operating points across models.

Global random seed: **2026** — results are fully reproducible.

## ⚠️ Data & Privacy Notice

- This repository contains **no patient data**, **no label files**, and **no
  model weights**.
- The label Excel files (`SL-*.xlsx`, `FY-*.xlsx`) contain **metastasis counts**
  and **must not** be committed to GitHub.
- Users are responsible for ensuring that any data they process is handled in
  accordance with applicable ethics and privacy regulations.

## Installation

System dependency (only needed if you want CJK fonts in plots):

```bash
# Ubuntu / Debian
sudo apt install fonts-noto-cjk