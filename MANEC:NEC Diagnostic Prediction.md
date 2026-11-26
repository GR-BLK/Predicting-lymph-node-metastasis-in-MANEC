# MANEC vs NEC — Path Features Binary Classification

Binary classification comparison pipeline (**MANEC vs NEC**) on pre-extracted
pathological patch features, using **mean-pooled embeddings** as the main
input. Provides:

- **Main model**: mean-pooled patch features + **L1 (LASSO) Logistic Regression**
  with inner nested 5-fold CV for `C` selection.
- **Additional tabular models**: LightGBM, XGBoost, ElasticNet LR, RandomForest
  (all on mean-pooled features; inner nested CV for hyperparameters).
- **Patch-level features MLP baseline**: **CLAM attention pooling** + MLP, with
  inner 80/20 split within the training fold for early stopping.
- **Internal validation**: stratified 10-fold CV, per-model internal-OOF
  threshold optimization (Youden), Platt probability recalibration, BCa
  bootstrap 95% CIs.
- **External test**: full internal-cohort-trained model applied to an
  independent cohort (`FY`); external labels are used **only for evaluation**.

Global random seed: **2026** — results are fully reproducible.

## ⚠️ Data & Privacy Notice

- This repository contains **no patient data** and **no model weights**.
- Users must provide their own `.npz` feature files.
- Users are responsible for ensuring that any data they process is handled
  in accordance with applicable ethics and privacy regulations.
- Do **not** commit `data/`, `results/`, `.npz`, `.svs`, `.tif`, `.geojson`,
  or any `.pkl` model files to GitHub.

## Installation

System dependency (only needed if you want CJK fonts in plots):

```bash
# Ubuntu / Debian
sudo apt install fonts-noto-cjk