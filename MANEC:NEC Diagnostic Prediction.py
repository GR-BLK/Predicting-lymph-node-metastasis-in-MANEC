#!/usr/bin/env python3
"""
Binary classification (MANEC vs NEC) multi-model comparison pipeline using Path features.
Global random seed: 2026; results are fully reproducible.

Path configuration:
  Data directory is specified by env var WSI_DATA_DIR (default: <script_dir>/data);
  it should contain SL/ and FY/ subdirectories.
  Output directory is specified by env var WSI_OUTPUT_DIR
  (default: <script_dir>/results/MANEC_vs_NEC).
"""

import os
import glob
import pickle
import hashlib
import warnings
from collections import Counter
from copy import deepcopy
from datetime import datetime
import multiprocessing

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_CJK_FONT_FILES = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                   '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']
for _fp in _CJK_FONT_FILES:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams['font.family'] = 'Noto Sans CJK JP' if any(os.path.exists(f) for f in _CJK_FONT_FILES) else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
from tqdm import tqdm

warnings.filterwarnings('ignore')

# GPU and CPU hardware detection (fully utilize all hardware resources)
N_CPUS = multiprocessing.cpu_count()
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

os.environ['OMP_NUM_THREADS'] = str(N_CPUS)
os.environ['MKL_NUM_THREADS'] = str(N_CPUS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CPUS)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_CPUS)
torch.set_num_threads(N_CPUS)

if N_GPUS > 0:
    print(f"\n  [Hardware] Detected {N_GPUS} GPU(s): {[torch.cuda.get_device_name(i) for i in range(N_GPUS)]}")
    print(f"  [Hardware] Detected {N_CPUS} CPU threads; using all")
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    SKLEARN_N_JOBS = N_CPUS
    USE_MULTI_GPU = (N_GPUS > 1)
else:
    print("\n  [Hardware] No GPU detected; running on CPU")
    print(f"  [Hardware] Detected {N_CPUS} CPU threads; using all")
    DEVICE = torch.device('cpu')
    torch.backends.cudnn.benchmark = False
    SKLEARN_N_JOBS = N_CPUS
    USE_MULTI_GPU = (N_GPUS > 1)

# Global configuration
SEED = 2026
N_FOLDS = 10
N_CLASSES = 2
CLASSES = ['NEC', 'MANEC']            # Binary labels (index 0=NEC, 1=MANEC)
PATCH_CATEGORIES = ('MANEC', 'NEC')   # Only load patch feature dirs for these two classes

PATCH_FEAT_DIM = 1536
CLAM_HIDDEN_DIM = 128
CLAM_DROPOUT = 0.2

MLP_HIDDEN_DIM = 128
BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.3

LASSO_CV_MAX_ITER = 5000

# Inner nested CV folds (used for per-model hyperparameter and threshold selection)
STAT_INNER_CV_FOLDS = 5
# LR/LASSO: inner 5-fold CV within the training fold to search the optimal C
LR_C_GRID = [0.01, 0.1, 1.0, 2.0, 3.0]

# C-AUC diagnostic (only to check whether the current grid is at the boundary;
# not used for model selection or threshold)
ENABLE_C_AUC_DIAGNOSTIC = True
C_AUC_DIAGNOSTIC_GRID = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0,
                         5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
C_AUC_DIAGNOSTIC_REPEATS = 3

# NN: inner split ratio within the training fold (80% train / 20% inner val,
# for early stopping only)
NN_INNER_VAL_FRAC = 0.2

N_BOOTSTRAP = 10000

# Main-model probability recalibration (Platt scaling / logistic recalibration):
#   - Fit two parameters (slope, intercept) on main-model internal OOF predictions only;
#     external test labels do not participate.
#   - Calibration is a monotone transform and does not change AUC or sample ranking.
#   - Per the analysis requirement, the classification threshold keeps its original
#     numeric value after recalibration (no remapping applied).
ENABLE_MAIN_RECALIBRATION = True
RECALIBRATION_CV_FOLDS = 5
RECALIBRATION_MIN_SAMPLES = 20

ENABLE_THRESHOLD_OPTIMIZATION = True
GLOBAL_THRESHOLD = 0.5
MODEL_THRESHOLDS = {}

LR_LEAK_FIX_VERSION = 8

# Additional models: hyperparameter grids (optimized via inner nested CV,
# same logic as LASSO)
LIGHTGBM_GRID = [
    {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31},
    {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15},
    {'n_estimators': 400, 'learning_rate': 0.03, 'num_leaves': 63},
]
XGBOOST_GRID = [
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
    {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 4},
    {'n_estimators': 400, 'learning_rate': 0.03, 'max_depth': 6},
]
ELASTICNET_GRID = [
    {'C': 0.1, 'l1_ratio': 0.5},
    {'C': 1.0, 'l1_ratio': 0.2},
    {'C': 1.0, 'l1_ratio': 0.5},
    {'C': 1.0, 'l1_ratio': 0.8},
    {'C': 3.0, 'l1_ratio': 0.5},
]
RANDOMFOREST_GRID = [
    {'n_estimators': 300, 'max_depth': None},
    {'n_estimators': 500, 'max_depth': 10},
    {'n_estimators': 500, 'max_depth': 20},
]

# Directory configuration (overridable via env vars for local/cluster/CI reuse;
# defaults relative to the script directory)
#   WSI_DATA_DIR   : root dir of feature data (contains SL/ and FY/ subdirs)
#   WSI_OUTPUT_DIR : output directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('WSI_DATA_DIR', os.path.join(_SCRIPT_DIR, 'data'))
SL_DIR = os.path.join(DATA_DIR, 'SL')
FY_DIR = os.path.join(DATA_DIR, 'FY')
OUTPUT_DIR = os.environ.get(
    'WSI_OUTPUT_DIR', os.path.join(_SCRIPT_DIR, 'results', 'MANEC_vs_NEC'))

for subdir in ['folds', 'models', 'predictions', 'figures', 'baselines']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(SEED)

LABEL_MAP_BINARY = {'NEC': 0, 'MANEC': 1}


def to_binary_labels(categories):
    return np.array([LABEL_MAP_BINARY[c] for c in categories], dtype=np.int64)


# Cache validation: analysis signature / data and fold fingerprints
def get_analysis_signature():
    """Current "analysis signature" used for cache validation.

    Any configuration that affects model selection or threshold must be included in
    the signature; otherwise stale caches would be silently reused after changes.
    """
    return (f"seed={SEED}|classes={CLASSES}|cats={PATCH_CATEGORIES}|folds={N_FOLDS}|"
            f"lrc={LR_C_GRID}|en={ELASTICNET_GRID}|lgb={LIGHTGBM_GRID}|"
            f"xgb={XGBOOST_GRID}|rf={RANDOMFOREST_GRID}|boot={N_BOOTSTRAP}")


def _sha1_of_lines(lines):
    payload = '|'.join(sorted(str(x) for x in lines))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def _data_fingerprint(case_ids, categories=None):
    """Fingerprint of case IDs (optionally with class labels), used to detect data
    additions/removals or label changes."""
    if categories is None:
        return _sha1_of_lines(list(case_ids))
    return _sha1_of_lines([f"{c}:{k}" for c, k in zip(case_ids, categories)])


def _folds_fingerprint(folds):
    """Fold split fingerprint: sorted concatenation of validation cases per fold."""
    return _sha1_of_lines([f"{fi['fold']}:" + ','.join(sorted(fi['val_cases'])) for fi in folds])


def _patch_dir_fingerprint(patch_dir):
    """Patch directory fingerprint computed from filenames only (without reading
    npz contents), used for external cohort cache validation."""
    npz_files = []
    for cat in PATCH_CATEGORIES:
        npz_files.extend(sorted(glob.glob(os.path.join(patch_dir, cat, '*_features.npz'))))
    return _sha1_of_lines([p.replace(patch_dir, '') for p in npz_files])


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


# Bootstrap BCa confidence interval
def auc_bca_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    obs_auc = safe_auc(y_true, y_prob)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)
    boot_aucs = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sample_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        try:
            boot_aucs.append(roc_auc_score(y_true[sample_idx], y_prob[sample_idx]))
        except:
            continue
    boot_aucs = np.array(boot_aucs)
    # Clip to the open interval (0,1) to avoid z0=-inf and a degenerate CI
    # when mean=0 (perfect separation).
    _p = np.clip(np.mean(boot_aucs < obs_auc), 1.0 / (len(boot_aucs) + 1), 1 - 1.0 / (len(boot_aucs) + 1))
    z0 = norm.ppf(_p)
    n = len(y_true)
    jack_aucs = []
    all_idx = np.arange(n)
    for i in range(n):
        jack_idx = np.delete(all_idx, i)
        jack_aucs.append(safe_auc(y_true[jack_idx], y_prob[jack_idx]))
    jack_aucs = np.array(jack_aucs)
    jack_mean = np.mean(jack_aucs)
    num = np.sum((jack_mean - jack_aucs) ** 3)
    den = np.sum((jack_mean - jack_aucs) ** 2)
    a = num / (6 * (den ** 1.5)) if den != 0 else 0.0
    z_alpha2 = norm.ppf(alpha / 2)
    z_1_alpha2 = norm.ppf(1 - alpha / 2)
    p1 = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
    p2 = norm.cdf(z0 + (z0 + z_1_alpha2) / (1 - a * (z0 + z_1_alpha2)))
    ci_lower = np.quantile(boot_aucs, p1)
    ci_upper = np.quantile(boot_aucs, p2)
    return obs_auc, ci_lower, ci_upper


def bca_ci_general(y_true, y_prob, metric_fn, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    obs = metric_fn(y_true, y_prob)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)
    boot_vals = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sample_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        try:
            boot_vals.append(metric_fn(y_true[sample_idx], y_prob[sample_idx]))
        except:
            continue
    boot_vals = np.array(boot_vals)
    _p = np.clip(np.mean(boot_vals < obs), 1.0 / (len(boot_vals) + 1), 1 - 1.0 / (len(boot_vals) + 1))
    z0 = norm.ppf(_p)
    n = len(y_true)
    jack_vals = []
    all_idx = np.arange(n)
    for i in range(n):
        jack_idx = np.delete(all_idx, i)
        try:
            jack_vals.append(metric_fn(y_true[jack_idx], y_prob[jack_idx]))
        except:
            jack_vals.append(np.nan)
    jack_vals = np.array(jack_vals)
    jack_vals = jack_vals[~np.isnan(jack_vals)]
    if len(jack_vals) < 3:
        ci_lower = np.quantile(boot_vals, alpha / 2)
        ci_upper = np.quantile(boot_vals, 1 - alpha / 2)
        return obs, ci_lower, ci_upper
    jack_mean = np.mean(jack_vals)
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = np.sum((jack_mean - jack_vals) ** 2)
    a = num / (6 * (den ** 1.5)) if den != 0 else 0.0
    z_alpha2 = norm.ppf(alpha / 2)
    z_1_alpha2 = norm.ppf(1 - alpha / 2)
    p1 = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
    p2 = norm.cdf(z0 + (z0 + z_1_alpha2) / (1 - a * (z0 + z_1_alpha2)))
    ci_lower = np.quantile(boot_vals, max(0, min(1, p1)))
    ci_upper = np.quantile(boot_vals, max(0, min(1, p2)))
    return obs, ci_lower, ci_upper


def compute_brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return np.mean((y_prob - y_true) ** 2)


def _logit(prob, eps=1e-6):
    """Probability -> logit, clipped to (eps, 1-eps) to avoid infinities."""
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_calibration_slope_intercept(y_true, y_prob):
    """Calibration slope/intercept: logistic regression of outcome on logit(p) as the
    only covariate.

    Ideal values are slope=1, intercept=0. Returns (slope, intercept); returns
    (nan, nan) if fitting fails.
    """
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    lp = _logit(y_prob).reshape(-1, 1)
    model = None
    for kwargs in ({'penalty': None}, {'C': 1e6}):
        try:
            model = LogisticRegression(solver='lbfgs', max_iter=1000, **kwargs).fit(lp, y)
            break
        except Exception:
            model = None
    if model is None:
        return float('nan'), float('nan')
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _bca_ci_from_samples(obs, boot_vals, jack_vals, alpha=0.05):
    """Construct BCa 95% CI from precomputed bootstrap and jackknife samples."""
    boot_vals = np.asarray(boot_vals, dtype=float)
    jack_vals = np.asarray(jack_vals, dtype=float)
    if len(boot_vals) < 10:
        return (float(obs), float('nan'), float('nan'))
    _p = np.clip(np.mean(boot_vals < obs), 1.0 / (len(boot_vals) + 1), 1 - 1.0 / (len(boot_vals) + 1))
    z0 = norm.ppf(_p)
    if len(jack_vals) >= 3:
        jm = jack_vals.mean()
        den = np.sum((jm - jack_vals) ** 2)
        a = np.sum((jm - jack_vals) ** 3) / (6 * den ** 1.5) if den > 0 else 0.0
    else:
        a = 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    p1 = norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    p2 = norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    return (float(obs),
            float(np.quantile(boot_vals, np.clip(p1, 0, 1))),
            float(np.quantile(boot_vals, np.clip(p2, 0, 1))))


def calibration_metric_cis(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """BCa 95% CIs for calibration slope and intercept (stratified resampling by class,
    consistent with other metrics).

    Both slope and intercept are fitted in one resampling pass to avoid running
    10000 iterations twice.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    obs_s, obs_i = fit_calibration_slope_intercept(y, p)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) < 2 or len(idx_neg) < 2 or not np.isfinite(obs_s):
        nan_ci = (float(obs_s), float('nan'), float('nan'))
        return {'cal_slope': float(obs_s), 'cal_slope_ci': nan_ci,
                'cal_intercept': float(obs_i), 'cal_intercept_ci': nan_ci}

    rng = np.random.RandomState(seed)
    boot_s, boot_i = [], []
    for _ in range(n_bootstrap):
        s = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                            rng.choice(idx_neg, len(idx_neg), replace=True)])
        sl, it = fit_calibration_slope_intercept(y[s], p[s])
        if np.isfinite(sl):
            boot_s.append(sl); boot_i.append(it)

    jack_s, jack_i = [], []
    for i in range(len(y)):
        j = np.delete(np.arange(len(y)), i)
        sl, it = fit_calibration_slope_intercept(y[j], p[j])
        if np.isfinite(sl):
            jack_s.append(sl); jack_i.append(it)

    return {'cal_slope': float(obs_s),
            'cal_slope_ci': _bca_ci_from_samples(obs_s, boot_s, jack_s, alpha),
            'cal_intercept': float(obs_i),
            'cal_intercept_ci': _bca_ci_from_samples(obs_i, boot_i, jack_i, alpha)}


# Main-model probability recalibration (Platt scaling)
def fit_platt_scaling(y_true, y_prob):
    """Fit a Platt calibrator on the given predictions:
    p_cal = sigmoid(intercept + slope * logit(p)).

    Only uses the provided predictions (main-model internal OOF); external test
    labels do not participate.
    """
    y = np.asarray(y_true, dtype=int)
    x = _logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    model.fit(x, y)
    return {'method': 'platt',
            'slope': float(model.coef_[0, 0]),
            'intercept': float(model.intercept_[0])}


def apply_platt_scaling(y_prob, calib):
    """Apply Platt calibration (strictly monotone transform, preserves sample
    ordering and AUC)."""
    if calib is None:
        return np.asarray(y_prob, dtype=float)
    a = float(calib['intercept'])
    b = float(calib['slope'])
    return 1.0 / (1.0 + np.exp(-(a + b * _logit(y_prob))))


def cross_fitted_platt_probabilities(y_true, y_prob,
                                     n_splits=RECALIBRATION_CV_FOLDS, seed=SEED):
    """Out-of-fold (cross-fitted) Platt calibrated probabilities: each fold is
    calibrated with the remaining folds, for honest evaluation of calibration gains."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if len(p) < 2 * n_splits or min(np.bincount(y, minlength=2)) < n_splits:
        return p.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob_cal = p.copy()
    for tr, va in skf.split(p.reshape(-1, 1), y):
        calib = fit_platt_scaling(y[tr], p[tr])
        prob_cal[va] = apply_platt_scaling(p[va], calib)
    return prob_cal


def calibration_summary(y_true, y_prob):
    """Calibration summary (point estimates): Brier / slope / intercept / mean
    predicted probability / observed positive rate."""
    slope, intercept = fit_calibration_slope_intercept(y_true, y_prob)
    return {
        'brier': float(compute_brier_score(y_true, y_prob)),
        'slope': float(slope),
        'intercept': float(intercept),
        'mean_pred': float(np.mean(y_prob)),
        'obs_rate': float(np.mean(np.asarray(y_true, dtype=float))),
    }


def _recalibration_signature(calib):
    if calib is None:
        return 'none'
    return f"{calib.get('method', 'platt')}|{float(calib['slope']):.8f}|{float(calib['intercept']):.8f}"


def prob_to_class(prob, threshold=None):
    """Probability -> class mapping: prob >= threshold -> 1 (MANEC), otherwise 0 (NEC).

    threshold supports scalar or sample-aligned array (fold threshold semantics).
    """
    if threshold is None:
        threshold = GLOBAL_THRESHOLD
    prob = np.asarray(prob)
    thr = np.asarray(threshold)
    return (prob >= thr).astype(int)


def round_to_4sf(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float('nan')
    if value == 0:
        return 0.0
    return float(f"{value:.4g}")


def classification_metric_cis(y_true, y_prob, threshold=None, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    if threshold is None:
        threshold = GLOBAL_THRESHOLD

    def _make(kind):
        def metric_fn(yt, yp):
            tn, fp, fn, tp = confusion_matrix(yt, prob_to_class(yp, threshold), labels=[0, 1]).ravel()
            if kind == 'sensitivity':
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if kind == 'specificity':
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            if kind == 'ppv':
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if kind == 'npv':
                return tn / (tn + fn) if (tn + fn) > 0 else 0.0
            if kind == 'accuracy':
                return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            raise ValueError(kind)
        return metric_fn

    return {
        kind + '_ci': bca_ci_general(y_true, y_prob, _make(kind), n_bootstrap, alpha, seed)
        for kind in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']
    }


def compute_oof_metrics_with_ci(y_true, y_prob, threshold=None, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Compute OOF metrics with BCa bootstrap CIs. Classification metrics use the
    given threshold."""
    if threshold is None:
        threshold = GLOBAL_THRESHOLD
    results = {}
    results['brier'] = compute_brier_score(y_true, y_prob)
    results['brier_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: compute_brier_score(yt, yp), n_bootstrap, alpha, seed)
    results['auc_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: safe_auc(yt, yp), n_bootstrap, alpha, seed)
    results['f1_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: f1_score(yt, prob_to_class(yp, threshold), zero_division=0),
        n_bootstrap, alpha, seed)
    # Calibration slope/intercept (logit regression; ideal slope=1 / intercept=0)
    results.update(calibration_metric_cis(y_true, y_prob, n_bootstrap, alpha, seed))
    results.update(classification_metric_cis(y_true, y_prob, threshold=threshold,
                                             n_bootstrap=n_bootstrap, alpha=alpha, seed=seed))
    return results


# Data loading
def load_patch_features(patch_dir, categories=PATCH_CATEGORIES):
    """Load raw patch features for the two classes MANEC / NEC only (excluding
    other classes)."""
    case_ids = []
    patch_list = []
    cats = []
    npz_files = []
    for cat in categories:
        npz_files.extend(sorted(glob.glob(os.path.join(patch_dir, cat, '*_features.npz'))))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found: {patch_dir}/{{MANEC,NEC}}/*_features.npz")
    for npz_path in npz_files:
        basename = os.path.basename(npz_path)
        case_id = basename.rsplit('_features', 1)[0]
        category = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        patch_features = data['features'].astype(np.float32)
        if patch_features.ndim != 2 or patch_features.shape[1] != PATCH_FEAT_DIM:
            raise ValueError(
                f"Unexpected patch feature dimension: {npz_path} shape={patch_features.shape}, "
                f"expected (n_patches, {PATCH_FEAT_DIM})")
        if patch_features.shape[0] == 0:
            raise ValueError(f"Empty patch features (0 patches): {npz_path}")
        case_ids.append(case_id)
        patch_list.append(patch_features)
        cats.append(category)
    # Each case must have exactly one npz; otherwise a case may land in different
    # folds (inner CV leakage) and be double-counted in external tests.
    dups = sorted(c for c, n in Counter(case_ids).items() if n > 1)
    if dups:
        raise ValueError(
            f"Duplicate patch feature case IDs ({len(dups)} cases): {dups[:10]}. "
            f"Ensure exactly one '*_features.npz' per case, or merge by case before running.")
    cats = np.array(cats)
    n_patches = [p.shape[0] for p in patch_list]
    print(f"  Loaded raw patch features: {len(case_ids)} cases, {PATCH_FEAT_DIM}-dim / patches")
    print(f"  Patches per case: min={min(n_patches)}, max={max(n_patches)}, median={np.median(n_patches):.0f}")
    print(f"  Class distribution: {dict(zip(*np.unique(cats, return_counts=True)))}")
    return case_ids, patch_list, cats


def compute_mean_pooled_features(patch_list):
    features = np.array([np.mean(p, axis=0) for p in patch_list], dtype=np.float32)
    return features


def generate_stratified_folds(case_ids, categories, n_folds=N_FOLDS, seed=SEED, save_path=None):
    case_ids = np.array(case_ids)
    categories = np.array(categories)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(case_ids, categories)):
        train_cases = case_ids[train_idx].tolist()
        val_cases = case_ids[val_idx].tolist()
        train_cats = categories[train_idx]
        val_cats = categories[val_idx]
        folds.append({
            'fold': fold_idx,
            'train_cases': train_cases,
            'val_cases': val_cases,
            'train_dist': {c: int(np.sum(train_cats == c)) for c in CLASSES},
            'val_dist': {c: int(np.sum(val_cats == c)) for c in CLASSES},
        })
        print(f"  Fold {fold_idx}: train={len(train_cases)} val={len(val_cases)}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(folds, f)
        print(f"  10-fold split saved: {save_path}")
    return folds


def load_or_create_folds(case_ids, categories, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    folds_path = os.path.join(output_dir, 'folds', 'stratified_folds.pkl')
    if os.path.exists(folds_path):
        with open(folds_path, 'rb') as f:
            folds = pickle.load(f)
        cached_val = sorted(c for fi in folds for c in fi['val_cases'])
        if len(folds) == N_FOLDS and cached_val == sorted(case_ids):
            print(f"  Loaded existing fold split: {folds_path} ({len(folds)} folds / {len(cached_val)} cases)")
            return folds
        print(f"  Cached fold split does not match current data "
              f"(cache {len(folds)} folds / {len(cached_val)} cases, "
              f"current {N_FOLDS} folds / {len(case_ids)} cases); regenerating ...")
    return generate_stratified_folds(case_ids, categories, save_path=folds_path)


# CLAM attention pooling
class CLAMAttentionPooling(nn.Module):
    def __init__(self, input_dim=PATCH_FEAT_DIM, hidden_dim=CLAM_HIDDEN_DIM, dropout=CLAM_DROPOUT):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, patches, mask=None):
        if isinstance(patches, list):
            return self._forward_list(patches)
        else:
            return self._forward_padded(patches, mask)

    def _forward_list(self, patch_list):
        pooled, attn_weights = [], []
        for p in patch_list:
            p = p.to(next(self.attention[0].parameters()).device)
            A = self.attention(p)
            A = torch.softmax(A, dim=0)
            pooled.append((A * p).sum(dim=0))
            attn_weights.append(A.squeeze(-1))
        return torch.stack(pooled), attn_weights

    def _forward_padded(self, patches_padded, mask):
        B, N, D = patches_padded.shape
        patches_flat = patches_padded.view(B * N, D)
        A = self.attention(patches_flat).view(B, N)
        A = A.masked_fill(~mask, float('-inf'))
        A = torch.softmax(A, dim=1)
        pooled = (A.unsqueeze(-1) * patches_padded).sum(dim=1)
        return pooled, A


# Mean-pooled feature standardizer (for linear models)
class BranchA_StatPreprocessor:
    def __init__(self, seed=SEED):
        self.seed = seed
        self.scaler = None
        self.output_dim = None

    def fit(self, X_train_raw, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_raw)
        self.output_dim = X_train_raw.shape[1]
        print(f"    Mean-pooled feature standardization complete; dim={self.output_dim}")
        return self

    def transform(self, X_raw):
        return self.scaler.transform(X_raw).astype(np.float32)

    def save(self, save_dir, fold_idx):
        with open(os.path.join(save_dir, f'stat_scaler_fold_{fold_idx}.pkl'), 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'output_dim': self.output_dim}, f)

    def load(self, load_dir, fold_idx):
        with open(os.path.join(load_dir, f'stat_scaler_fold_{fold_idx}.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.output_dim = data['output_dim']
        return self

    def save_final(self, save_dir, prefix='final'):
        with open(os.path.join(save_dir, f'{prefix}_stat_scaler.pkl'), 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'output_dim': self.output_dim}, f)


# Model definition: patch-level features MLP only (CLAM attention pooling + MLP)
class Baseline4_ImageBranchAOnly(nn.Module):
    """Patch-level features MLP only: CLAM attention pooling + MLP classification
    head (no clinical/fusion)."""
    def __init__(self, mlp_hidden=MLP_HIDDEN_DIM, n_classes=N_CLASSES, seed=SEED, **kwargs):
        super().__init__()
        set_seed(seed)
        self.clam_pooling = CLAMAttentionPooling()
        self.net = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, mlp_hidden), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, patches, masks=None, mixup_lam=None, mixup_perm=None, **kwargs):
        if isinstance(patches, list):
            pooled, attention = self.clam_pooling(patches)
        else:
            pooled, attention = self.clam_pooling(patches, masks)
        if mixup_lam is not None and mixup_perm is not None and self.training:
            pooled = mixup_lam * pooled + (1 - mixup_lam) * pooled[mixup_perm]
        return self.net(pooled), attention, None, None, None, torch.tensor(0.0, device=pooled.device)


class CaseDataset(Dataset):
    def __init__(self, case_ids, patch_list, labels):
        self.case_ids = case_ids
        self.patch_list = [torch.FloatTensor(p) for p in patch_list]
        self.labels = [torch.LongTensor([l]) for l in labels]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        return {
            'case_id': self.case_ids[idx],
            'patches': self.patch_list[idx],
            'label': self.labels[idx].item()
        }


def balanced_collate_fn(batch):
    patches = [b['patches'] for b in batch]
    labels = torch.LongTensor([b['label'] for b in batch])
    case_ids = [b['case_id'] for b in batch]
    return {'case_id': case_ids, 'patches': patches, 'label': labels}


def padded_collate_fn(batch):
    labels = torch.LongTensor([b['label'] for b in batch])
    case_ids = [b['case_id'] for b in batch]
    patch_list = [b['patches'] for b in batch]
    lengths = torch.LongTensor([p.shape[0] for p in patch_list])
    max_len = lengths.max().item()
    patches_padded = torch.zeros(len(batch), max_len, patch_list[0].shape[1])
    for i, p in enumerate(patch_list):
        patches_padded[i, :p.shape[0]] = p
    masks = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        'case_id': case_ids,
        'patches_padded': patches_padded, 'masks': masks,
        'patches_raw': patch_list,
        'label': labels
    }


class MultiGPUModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, patches, masks, **kwargs):
        return self.base_model(patches=patches, masks=masks)


COLLATE_FN = padded_collate_fn


def wrap_model_for_multi_gpu(model):
    if USE_MULTI_GPU:
        model = MultiGPUModelWrapper(model)
        model = nn.DataParallel(model)
    return model


def build_criterion(device=None):
    return torch.nn.BCEWithLogitsLoss(pos_weight=None)


def compute_metrics(y_true, y_pred_prob, y_pred_class):
    metrics = {}
    prob_manec = y_pred_prob[:, 1] if y_pred_prob.ndim > 1 else y_pred_prob
    metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['auc'] = safe_auc(y_true, prob_manec)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class, labels=[0, 1]).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['f1'] = f1_score(y_true, y_pred_class, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
    return metrics


def _is_padded_batch(batch):
    return 'patches_padded' in batch


def _get_model_inputs(batch, device):
    labels = batch['label'].to(device)
    if _is_padded_batch(batch):
        return labels, batch['patches_padded'].to(device), batch['masks'].to(device)
    else:
        patches = [p.to(device) for p in batch['patches']]
        return labels, patches, None


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        labels, patches_or_padded, masks = _get_model_inputs(batch, device)
        optimizer.zero_grad()
        logits, attentions, img_code, cli_code, gate, gate_reg = model(
            patches=patches_or_padded, masks=masks
        )
        loss_cls = criterion(logits, labels.float().unsqueeze(1))
        loss_reg = gate_reg if isinstance(gate_reg, torch.Tensor) else torch.tensor(0.0, device=device)
        if loss_reg.dim() > 0:
            loss_reg = loss_reg.mean()
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(logits.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, all_preds, all_labels


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_gates = [], [], []
    for batch in dataloader:
        labels, patches_or_padded, masks = _get_model_inputs(batch, device)
        logits, _, img_code, cli_code, gate, gate_reg = model(
            patches=patches_or_padded, masks=masks
        )
        loss = criterion(logits, labels.float().unsqueeze(1))
        total_loss += loss.item()
        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if gate is not None:
            if isinstance(gate, tuple):
                g = torch.cat(gate, dim=1).cpu().numpy()
            else:
                g = gate.cpu().numpy()
            all_gates.append(np.atleast_1d(g))
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logits_flat = all_preds.reshape(-1)
    prob_manec = torch.sigmoid(torch.from_numpy(logits_flat)).numpy()
    pred_prob = np.column_stack([1 - prob_manec, prob_manec])
    # Fallback threshold; outer layer recomputes per-fold threshold
    pred_class = prob_to_class(prob_manec, GLOBAL_THRESHOLD)
    gate_vals = np.concatenate(all_gates).flatten() if all_gates else np.array([])
    metrics = compute_metrics(all_labels, pred_prob, pred_class)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['mean_gate'] = float(np.mean(gate_vals)) if len(gate_vals) > 0 else 0.0
    return metrics, pred_prob, pred_class, gate_vals


def train_model(model, train_loader, val_loader, device,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                weight_decay=WEIGHT_DECAY, fold_idx=None):
    """Train the NN; returns (model, best_epoch, best_auc, inner_train_loader, inner_val_loader).

    Early stopping uses the inner validation set; the classification threshold is
    optimized on the model's outer OOF, not on inner_val_loader.
    """
    criterion = build_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    desc = f"    Fold {fold_idx} training" if fold_idx is not None else "    Training"
    for epoch in tqdm(range(max_epochs), desc=desc, unit="epoch", leave=False):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_auc = val_metrics['auc']
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_auc, train_loader, val_loader


# Threshold optimization (invoked on inner OOF / inner validation set)
def search_optimal_threshold(y_true, y_prob_manec):
    thresholds = np.arange(0.01, 0.99, 0.001)
    best_thr, best_youden, best_sens, best_spec = 0.5, -1.0, 0.0, 0.0
    for thr in thresholds:
        y_pred = (y_prob_manec >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1.0
        if youden > best_youden:
            best_youden = youden
            best_thr = thr
            best_sens, best_spec = sens, spec
    return {'threshold': best_thr, 'sensitivity': best_sens, 'specificity': best_spec, 'youden': best_youden}


def optimize_model_threshold(model_key, oof_y_true, oof_y_prob):
    """Optimize a global threshold on the model's own internal OOF (Youden
    maximization); all classification metrics for this model use this threshold."""
    res = search_optimal_threshold(np.asarray(oof_y_true), np.asarray(oof_y_prob))
    MODEL_THRESHOLDS[model_key] = res
    return res


def get_model_threshold(model_key):
    cfg = MODEL_THRESHOLDS.get(model_key)
    return cfg['threshold'] if cfg is not None else GLOBAL_THRESHOLD


def split_inner_train_val(train_cases, y_train_bin, frac=NN_INNER_VAL_FRAC, seed=SEED):
    try:
        inner_train, inner_val = train_test_split(
            train_cases, test_size=frac, random_state=seed, stratify=y_train_bin
        )
    except ValueError:
        inner_train, inner_val = train_test_split(
            train_cases, test_size=frac, random_state=seed
        )
    return list(inner_train), list(inner_val)


# Inner C selection: LASSO LR uses inner 5-fold CV within the training fold to
# search the optimal C.
def select_lr_best_C(X_train, y_train, cv=5, C_grid=None, seed=SEED):
    """LR selection strategy: inner 5-fold CV within the training fold to search
    for the optimal regularization strength C (by mean AUC).

    C selection uses only data within the training fold; the outer validation fold
    is used solely for final evaluation.
    """
    if C_grid is None:
        C_grid = LR_C_GRID
    n_samples = X_train.shape[0]
    if n_samples < 2 * cv or len(np.unique(y_train)) < 2:
        return 1.0
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    best_c, best_auc = C_grid[0], -1.0
    n_unconverged = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        for C in C_grid:
            inner_aucs = []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                model = LogisticRegression(
                    penalty='l1', C=C, solver='saga',
                    max_iter=LASSO_CV_MAX_ITER, random_state=seed, class_weight=None
                )
                model.fit(X_train[tr_idx], y_train[tr_idx])
                inner_aucs.append(safe_auc(y_train[va_idx], model.predict_proba(X_train[va_idx])[:, 1]))
            mean_auc = float(np.mean(inner_aucs))
            if mean_auc > best_auc:
                best_auc, best_c = mean_auc, C
        n_unconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    if n_unconverged > 0:
        print(f"    [Warning] {n_unconverged}/{len(C_grid) * cv} saga fits did not converge "
              f"in inner CV (max_iter={LASSO_CV_MAX_ITER})")
    return best_c


def diagnose_c_auc_curve(X_raw, y, output_dir=OUTPUT_DIR):
    """C-AUC diagnostic curve: repeated stratified CV on full SL with an extended
    C grid to check whether LR_C_GRID is at the boundary.

    Diagnostic visualization only; does not participate in model selection or
    threshold computation; does not affect main results.
    """
    if not ENABLE_C_AUC_DIAGNOSTIC:
        print("  [C-AUC diagnostic] Disabled")
        return None
    y = np.asarray(y)
    n_splits = STAT_INNER_CV_FOLDS
    if len(y) < 2 * n_splits or min(np.bincount(y)) < n_splits:
        print("  [C-AUC diagnostic] Insufficient samples for repeated stratified CV; skipping")
        return None
    print("\n[C-AUC diagnostic] Repeated inner CV on extended grid "
          "(diagnostic only; does not affect model selection) ...")
    X = StandardScaler().fit_transform(np.asarray(X_raw, dtype=np.float64))
    rows = []
    for C in C_AUC_DIAGNOSTIC_GRID:
        aucs = []
        for r in range(C_AUC_DIAGNOSTIC_REPEATS):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + r)
            for tr, va in skf.split(X, y):
                m = LogisticRegression(penalty='l1', C=C, solver='saga',
                                       max_iter=LASSO_CV_MAX_ITER,
                                       random_state=SEED + r, class_weight=None)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m.fit(X[tr], y[tr])
                aucs.append(safe_auc(y[va], m.predict_proba(X[va])[:, 1]))
        rows.append({'C': C, 'mean_auc': float(np.mean(aucs)),
                     'std_auc': float(np.std(aucs))})
        print(f"    C={C:<6g} mean AUC={rows[-1]['mean_auc']:.4f} (±{rows[-1]['std_auc']:.4f})")

    df = pd.DataFrame(rows)
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    df.to_csv(os.path.join(pred_dir, 'c_auc_diagnostic.csv'), index=False)

    best = df.loc[df['mean_auc'].idxmax()]
    grid_upper = max(LR_C_GRID)
    print(f"  Diagnostic best C={best['C']:g} (AUC={best['mean_auc']:.4f}); "
          f"current grid upper bound={grid_upper:g} "
          f"({'at boundary, consider widening grid' if best['C'] >= grid_upper else 'not at boundary'})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['C'], df['mean_auc'], marker='o', color='#2c7bb6', lw=2)
    ax.axvspan(min(LR_C_GRID), grid_upper, color='#fdae61', alpha=0.25,
               label='Current search grid')
    ax.set_xscale('log')
    ax.set_xlabel('Regularization strength C (log scale)', fontsize=12)
    ax.set_ylabel(f'{C_AUC_DIAGNOSTIC_REPEATS}x{n_splits}-fold mean AUC', fontsize=12)
    ax.set_title('C-AUC Diagnostic Curve (not used for model selection)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figures', 'c_auc_diagnostic.tiff'),
                dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}, facecolor='white')
    plt.close()
    print("  C-AUC diagnostic saved: predictions/c_auc_diagnostic.csv, figures/c_auc_diagnostic.tiff")
    return df


# Additional models: hyperparameter factories (for inner nested CV optimization)
def _make_elasticnet(param, seed):
    return LogisticRegression(
        penalty='elasticnet', solver='saga',
        C=param['C'], l1_ratio=param['l1_ratio'],
        max_iter=LASSO_CV_MAX_ITER, random_state=seed, class_weight=None
    )


def _make_randomforest(param, seed):
    return RandomForestClassifier(
        n_estimators=param['n_estimators'], max_depth=param['max_depth'],
        random_state=seed, n_jobs=SKLEARN_N_JOBS
    )


def _make_lightgbm(param, seed):
    return LGBMClassifier(
        n_estimators=param['n_estimators'], learning_rate=param['learning_rate'],
        num_leaves=param['num_leaves'], random_state=seed,
        n_jobs=SKLEARN_N_JOBS, verbose=-1
    )


def _make_xgboost(param, seed):
    return XGBClassifier(
        n_estimators=param['n_estimators'], learning_rate=param['learning_rate'],
        max_depth=param['max_depth'], random_state=seed,
        n_jobs=SKLEARN_N_JOBS, verbosity=0, eval_metric='logloss'
    )


# Model registry: key -> (display name, factory, hyperparameter grid)
NEW_TABULAR_MODELS = {
    'model_lightgbm':     ('LightGBM',                       _make_lightgbm,     LIGHTGBM_GRID),
    'model_xgboost':      ('XGBoost',                        _make_xgboost,      XGBOOST_GRID),
    'model_elasticnet':   ('ElasticNet Logistic Regression',  _make_elasticnet,   ELASTICNET_GRID),
    'model_randomforest': ('RandomForest',                   _make_randomforest, RANDOMFOREST_GRID),
}


def inner_cv_for_model(model_factory, param_grid, X_train_raw, y_train,
                       n_inner=STAT_INNER_CV_FOLDS, seed=SEED):
    """Inner nested CV: select best hyperparameters + generate inner OOF probabilities
    (per-fold standardization, zero leakage).

    Uses the same "inner 5-fold CV within the training fold" strategy as
    select_lr_best_C, applied to additional models beyond LASSO.
    Returns: best_param, inner_oof_prob
    """
    X_train_raw = np.asarray(X_train_raw, dtype=np.float64)
    y_train = np.asarray(y_train)
    n = len(y_train)
    rng = np.random.RandomState(seed)

    n_unique = len(np.unique(y_train))
    if n < 2 * n_inner or n_unique < 2:
        sc = StandardScaler().fit(X_train_raw)
        m = model_factory(param_grid[0], seed)
        m.fit(sc.transform(X_train_raw), y_train)
        return param_grid[0], m.predict_proba(sc.transform(X_train_raw))[:, 1]

    best_param = param_grid[0]
    best_auc = -1.0
    best_oof = np.zeros(n, dtype=np.float64)

    for param in param_grid:
        try:
            skf = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                  random_state=int(rng.randint(0, 10**6)))
            oof = np.zeros(n, dtype=np.float64)
            for tr_local, va_local in skf.split(X_train_raw, y_train):
                sc_local = StandardScaler().fit(X_train_raw[tr_local])
                m = model_factory(param, int(rng.randint(0, 10**6)))
                m.fit(sc_local.transform(X_train_raw[tr_local]), y_train[tr_local])
                oof[va_local] = m.predict_proba(
                    sc_local.transform(X_train_raw[va_local]))[:, 1]
            auc = safe_auc(y_train, oof)
        except Exception:
            auc = -1.0
            oof = np.zeros(n, dtype=np.float64)
        if auc > best_auc:
            best_auc = auc
            best_param = param
            best_oof = oof

    return best_param, best_oof


# External test (main model: image-only linear LASSO LR; threshold = main-model
# internal-OOF optimized; external labels do not participate)
def process_external_test(threshold_config=None, output_dir=OUTPUT_DIR, main_threshold=None,
                          recalibration=None):
    print(f"\n{'='*60}")
    print("  External test inference (main model: mean-pooled patch features + linear LASSO LR)")
    print(f"{'='*60}")
    # Main-model classification threshold: from internal OOF optimization
    if main_threshold is None:
        main_threshold = get_model_threshold('main')

    sl_patch_dir = os.path.join(SL_DIR, 'SL-patch_features')
    case_ids_sl, patch_list_sl, categories_sl = load_patch_features(sl_patch_dir)
    y_sl_bin = to_binary_labels(categories_sl)

    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    stat_features_sl = compute_mean_pooled_features(patch_list_sl)
    final_stat_pre = BranchA_StatPreprocessor(seed=SEED)
    final_stat_pre.fit(stat_features_sl)
    final_stat_pre.save_final(models_dir, 'final')
    X_sl_stat = final_stat_pre.transform(stat_features_sl)

    # Train image-only linear LR (inner 5-fold CV on full SL to search optimal C)
    print("\n  Training image-only linear LR (L1 LogisticRegression) ...")
    best_C = select_lr_best_C(X_sl_stat, y_sl_bin)
    print(f"    Inner CV best C = {best_C:.4g}")

    lr = LogisticRegression(penalty='l1', C=best_C, solver='saga',
                            max_iter=LASSO_CV_MAX_ITER, random_state=SEED, class_weight=None)
    lr.fit(X_sl_stat, y_sl_bin)
    with open(os.path.join(models_dir, 'final_linear_lr.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(models_dir, 'final_stat_scaler_for_lr.pkl'), 'wb') as f:
        pickle.dump(final_stat_pre.scaler, f)
    print(f"  Final LR trained; number of non-zero features: {np.sum(lr.coef_ != 0)}")

    fy_patch_dir = os.path.join(FY_DIR, 'FY-patch_features')
    fy_case_ids, fy_patch_list, fy_categories = load_patch_features(fy_patch_dir)
    fy_y_bin = to_binary_labels(fy_categories)

    fy_stat_features = compute_mean_pooled_features(fy_patch_list)
    fy_stat_reduced = final_stat_pre.transform(fy_stat_features)

    print("\n  External test inference ...")
    all_prob_raw = lr.predict_proba(fy_stat_reduced)
    ext_prob_raw = all_prob_raw[:, 1]
    # Main-model probability recalibration: Platt calibrator fitted on internal OOF
    # (external labels do not participate; monotone transform preserves ranking)
    ext_prob = apply_platt_scaling(ext_prob_raw, recalibration)
    all_prob = np.column_stack([1.0 - ext_prob, ext_prob])
    all_pred = prob_to_class(ext_prob, main_threshold)
    all_true = fy_y_bin

    metrics = compute_metrics(all_true, all_prob, all_pred)

    auc_mean, auc_lower, auc_upper = auc_bca_ci(all_true, ext_prob)

    ext_brier_ci = bca_ci_general(all_true, ext_prob,
        lambda yt, yp: compute_brier_score(yt, yp), N_BOOTSTRAP, 0.05, SEED)
    ext_cat_ci = classification_metric_cis(all_true, ext_prob, threshold=main_threshold,
                                           n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED)
    ext_f1_ci = bca_ci_general(all_true, ext_prob,
        lambda yt, yp: f1_score(yt, prob_to_class(yp, main_threshold), zero_division=0),
        N_BOOTSTRAP, 0.05, SEED)
    # Calibration slope/intercept (BCa method consistent with internal validation)
    ext_cal = calibration_metric_cis(all_true, ext_prob, N_BOOTSTRAP, 0.05, SEED)
    ext_summary_raw = calibration_summary(all_true, ext_prob_raw)
    ext_summary_cal = calibration_summary(all_true, ext_prob)

    print(f"\n  External test results "
          f"(main-model OOF-optimized threshold={main_threshold:.4f}; "
          f"threshold not remapped after recalibration):")
    print(f"    AUC: {auc_mean:.4f} (95% CI: {auc_lower:.4f} - {auc_upper:.4f})")
    print(f"    Accuracy: {metrics['accuracy']:.4f} | Sensitivity: {metrics['sensitivity']:.4f} "
          f"| Specificity: {metrics['specificity']:.4f}")
    print(f"    Brier: {ext_brier_ci[0]:.4g} "
          f"(95% CI: {ext_brier_ci[1]:.4g}-{ext_brier_ci[2]:.4g})")
    print(f"    Cal. Slope: {ext_cal['cal_slope']:.4g} "
          f"(95% CI: {ext_cal['cal_slope_ci'][1]:.4g}-{ext_cal['cal_slope_ci'][2]:.4g})")
    print(f"    Cal. Intercept: {ext_cal['cal_intercept']:.4g} "
          f"(95% CI: {ext_cal['cal_intercept_ci'][1]:.4g}-{ext_cal['cal_intercept_ci'][2]:.4g})")
    print(f"    Before/after recalibration: Brier {ext_summary_raw['brier']:.4f} -> "
          f"{ext_summary_cal['brier']:.4f} | Slope {ext_summary_raw['slope']:.4f} -> "
          f"{ext_summary_cal['slope']:.4f} | Intercept {ext_summary_raw['intercept']:.4f} -> "
          f"{ext_summary_cal['intercept']:.4f}")

    pred_dir = os.path.join(output_dir, 'predictions')
    pd.DataFrame({
        'case_id': fy_case_ids,
        'true_label': [CLASSES[l] for l in all_true],
        'pred_label': [CLASSES[p] for p in all_pred],
        # prob_NEC / prob_MANEC are the final (post-recalibration) probabilities;
        # *_raw are the model's raw outputs
        'prob_NEC': 1.0 - ext_prob,
        'prob_MANEC': ext_prob,
        'prob_NEC_raw': 1.0 - ext_prob_raw,
        'prob_MANEC_raw': ext_prob_raw,
        'threshold': main_threshold,
    }).to_csv(os.path.join(pred_dir, 'main_model_external_predictions.csv'), index=False)

    full_state = {
        'ext_metrics': metrics,
        'ext_auc_ci': (auc_mean, auc_lower, auc_upper),
        'ext_brier_ci': ext_brier_ci,
        'ext_cat_ci': ext_cat_ci,
        'ext_f1_ci': ext_f1_ci,
        'ext_cal': ext_cal,
        'ext_summary_raw': ext_summary_raw,
        'ext_summary_calibrated': ext_summary_cal,
        'recalibration': recalibration,
        'recalibration_sig': _recalibration_signature(recalibration),
        'final_stat_pre': final_stat_pre,
        'stat_dim': final_stat_pre.output_dim,
        'fy_case_ids': fy_case_ids,
        'fy_stat_reduced': fy_stat_reduced,
        'fy_y_bin': fy_y_bin,
        'main_preds': {'prob': all_prob, 'prob_raw': all_prob_raw,
                       'class': all_pred, 'true': all_true},
        'threshold': main_threshold,
        'signature': get_analysis_signature(),
        'sl_fp': _data_fingerprint(case_ids_sl, categories_sl),
        'fy_fp': _patch_dir_fingerprint(fy_patch_dir),
        'leak_fix_version': LR_LEAK_FIX_VERSION,
    }
    with open(os.path.join(pred_dir, 'external_test_state.pkl'), 'wb') as f:
        pickle.dump(full_state, f)
    return full_state


# Baseline models: 10-fold cross-validation
#   - Patch-level features MLP only (CLAM attention pooling; inner 80/20 split of
#     the training fold for early stopping)
#   - Additional tabular models (mean-pooled features; inner nested CV for
#     hyperparameters + inner OOF for fold-level thresholds)
def run_baselines_cv(folds, patch_list_all, categories_all, case_ids, output_dir=OUTPUT_DIR):
    print(f"\n{'='*60}")
    print("  Baseline models: 10-fold cross-validation")
    print(f"{'='*60}")

    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    all_results = {}
    oof_preds = {}
    oof_ci_results = {}

    # --- Patch-level features MLP only (CLAM attention pooling) ---
    print("\n  --- Patch-level features MLP only ---")
    fold_results = []
    oof_y_true = []
    oof_y_prob = []
    oof_case_ids = []
    fold_probs = []
    fold_trues = []

    for fold_info in tqdm(folds, desc="    Patch-level features MLP only", unit="fold"):
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        y_train_bin = to_binary_labels(categories_all[train_indices])
        y_val_bin = to_binary_labels(categories_all[val_indices])

        val_patches = [patch_list_all[i] for i in val_indices]

        # Inner 80/20 split within the training fold
        inner_train_cases, inner_val_cases = split_inner_train_val(train_cases, y_train_bin)
        inner_train_indices = [case_to_idx[c] for c in inner_train_cases if c in case_to_idx]
        inner_val_indices = [case_to_idx[c] for c in inner_val_cases if c in case_to_idx]
        y_inner_train_bin = to_binary_labels(categories_all[inner_train_indices])
        y_inner_val_bin = to_binary_labels(categories_all[inner_val_indices])

        inner_train_patches = [patch_list_all[i] for i in inner_train_indices]
        inner_val_patches = [patch_list_all[i] for i in inner_val_indices]

        inner_train_dataset = CaseDataset(inner_train_cases, inner_train_patches, y_inner_train_bin)
        inner_val_dataset = CaseDataset(inner_val_cases, inner_val_patches, y_inner_val_bin)
        inner_train_loader = DataLoader(inner_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                        collate_fn=COLLATE_FN,
                                        generator=torch.Generator().manual_seed(SEED))
        inner_val_loader = DataLoader(inner_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=COLLATE_FN)
        val_dataset = CaseDataset(val_cases, val_patches, y_val_bin)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=COLLATE_FN)

        model = Baseline4_ImageBranchAOnly(seed=SEED).to(DEVICE)
        if USE_MULTI_GPU:
            model = wrap_model_for_multi_gpu(model)
        model, best_epoch, best_auc, inner_train_loader, inner_val_loader = train_model(
            model, inner_train_loader, inner_val_loader, DEVICE, fold_idx=fold_idx
        )

        criterion_eval = build_criterion(DEVICE)
        # Outer validation fold: accumulate OOF probabilities only; the model
        # optimizes a global threshold on its internal OOF
        val_metrics, val_prob, _, _ = evaluate(model, val_loader, criterion_eval, DEVICE)

        fold_results.append({'fold': fold_idx, 'val_auc': val_metrics['auc']})
        oof_y_true.extend(y_val_bin)
        oof_y_prob.extend(val_prob[:, 1])
        oof_case_ids.extend(val_cases)
        fold_probs.append(np.asarray(val_prob[:, 1]))
        fold_trues.append(np.asarray(y_val_bin))

    oof_yt = np.array(oof_y_true)
    oof_yp = np.array(oof_y_prob)
    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    print(f"    Mean Val AUC: {mean_auc:.4f}")

    # Optimize a threshold independently on this model's internal OOF (Youden
    # maximization) for its subsequent classification
    key = 'baseline_stat_only'
    display_name = 'Patch-level features MLP only'
    thr_res = optimize_model_threshold(key, oof_yt, oof_yp)
    model_thr = get_model_threshold(key)
    print(f"    [Threshold] {display_name}: internal-OOF optimized threshold={model_thr:.4f} "
          f"(Youden={thr_res['youden']:.4f}, Sens={thr_res['sensitivity']:.4f}, "
          f"Spec={thr_res['specificity']:.4f})")

    # Recompute per-fold classification metrics using the model's own threshold
    # (overriding fallback-threshold values computed at training time)
    for i, r in enumerate(fold_results):
        yt_i, pr_i = fold_trues[i], fold_probs[i]
        pred_i = prob_to_class(pr_i, model_thr)
        tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
        r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        r['val_f1'] = f1_score(yt_i, pred_i, zero_division=0)
        r['threshold'] = model_thr

    all_results[key] = {'display_name': display_name, 'fold_results': fold_results,
                        'mean_auc': mean_auc, 'threshold': model_thr}
    oof_preds[key] = {
        'case_ids': oof_case_ids,
        'y_true': oof_yt,
        'y_prob': oof_yp,
        'display_name': display_name,
        'threshold': model_thr,
    }
    oof_ci_results[key] = compute_oof_metrics_with_ci(oof_yt, oof_yp, threshold=model_thr)

    # --- Additional models (mean-pooled patch features + inner nested CV) ---
    print("\n  === Additional models (mean-pooled patch features + inner nested CV) ===")

    # Precompute per-fold mean-pooled features (identical to the LASSO main model)
    # for reuse by all additional models
    fold_tab_data = []
    for fold_info in folds:
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        fold_tab_data.append({
            'fold': fold_idx,
            'val_cases': val_cases,
            'y_train_bin': to_binary_labels(categories_all[train_indices]),
            'y_val_bin': to_binary_labels(categories_all[val_indices]),
            'X_train_mean': compute_mean_pooled_features([patch_list_all[i] for i in train_indices]),
            'X_val_mean': compute_mean_pooled_features([patch_list_all[i] for i in val_indices]),
        })

    for key, (display_name, factory, grid) in NEW_TABULAR_MODELS.items():
        print(f"\n  --- {display_name} ---")
        fold_results = []
        oof_y_true = []
        oof_y_prob = []
        oof_case_ids = []
        fold_probs = []
        fold_trues = []

        for fd in tqdm(fold_tab_data, desc=f"    {display_name}", unit="fold"):
            fold_idx = fd['fold']
            val_cases = fd['val_cases']
            y_train_bin = fd['y_train_bin']
            y_val_bin = fd['y_val_bin']
            X_train_mean = fd['X_train_mean']
            X_val_mean = fd['X_val_mean']

            # Inner 5-fold CV within the training fold to grid-search the best
            # hyperparameters (same logic as LASSO)
            best_param, _ = inner_cv_for_model(
                factory, grid, X_train_mean, y_train_bin, STAT_INNER_CV_FOLDS, seed=SEED + fold_idx
            )

            # Fit scaler + model on the training fold
            scaler = StandardScaler().fit(X_train_mean)
            model = factory(best_param, SEED)
            model.fit(scaler.transform(X_train_mean), y_train_bin)
            y_val_prob = model.predict_proba(scaler.transform(X_val_mean))[:, 1]
            val_auc = safe_auc(y_val_bin, y_val_prob)

            fold_results.append({'fold': fold_idx, 'val_auc': val_auc, 'best_param': best_param})
            oof_y_true.extend(y_val_bin)
            oof_y_prob.extend(y_val_prob)
            oof_case_ids.extend(val_cases)
            fold_probs.append(np.asarray(y_val_prob))
            fold_trues.append(np.asarray(y_val_bin))

        oof_yt = np.array(oof_y_true)
        oof_yp = np.array(oof_y_prob)
        mean_auc = float(np.mean([r['val_auc'] for r in fold_results]))
        print(f"    Mean Val AUC: {mean_auc:.4f}")

        # Optimize a threshold independently on this model's internal OOF (Youden
        # maximization) for its subsequent classification
        thr_res = optimize_model_threshold(key, oof_yt, oof_yp)
        model_thr = get_model_threshold(key)
        print(f"    [Threshold] {display_name}: internal-OOF optimized threshold={model_thr:.4f} "
              f"(Youden={thr_res['youden']:.4f}, Sens={thr_res['sensitivity']:.4f}, "
              f"Spec={thr_res['specificity']:.4f})")

        # Recompute per-fold classification metrics using the model's own threshold
        for i, r in enumerate(fold_results):
            yt_i, pr_i = fold_trues[i], fold_probs[i]
            pred_i = prob_to_class(pr_i, model_thr)
            tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
            r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            r['val_f1'] = f1_score(yt_i, pred_i, zero_division=0)
            r['threshold'] = model_thr

        all_results[key] = {'display_name': display_name, 'fold_results': fold_results,
                            'mean_auc': mean_auc, 'threshold': model_thr}
        oof_preds[key] = {
            'case_ids': oof_case_ids,
            'y_true': oof_yt,
            'y_prob': oof_yp,
            'display_name': display_name,
            'threshold': model_thr,
        }
        oof_ci_results[key] = compute_oof_metrics_with_ci(oof_yt, oof_yp, threshold=model_thr)

    with open(os.path.join(output_dir, 'baselines', 'baselines_cv.pkl'), 'wb') as f:
        pickle.dump({'results': all_results, 'oof_preds': oof_preds,
                     'oof_ci_results': oof_ci_results,
                     'signature': get_analysis_signature(),
                     'folds_fingerprint': _folds_fingerprint(folds),
                     'leak_fix_version': LR_LEAK_FIX_VERSION}, f)
    return all_results, oof_preds, oof_ci_results


# Comprehensive Excel summary
def _generate_comprehensive_summary(main_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=None,
                                     main_oof_metrics_ci=None,
                                     output_dir=OUTPUT_DIR):
    import openpyxl

    def _fmt_ci(ci_tuple):
        if ci_tuple is None or np.isnan(ci_tuple[1]):
            return ''
        return f'{round_to_4sf(ci_tuple[1])}-{round_to_4sf(ci_tuple[2])}'

    def _build_oof_row(name, oof_dict, oof_metrics_ci=None):
        """Internal OOF summary row (each model uses the global threshold optimized
        on its own internal OOF)."""
        thr = oof_dict.get('threshold')
        if thr is None:
            thr = GLOBAL_THRESHOLD
        y_pred = prob_to_class(oof_dict['y_prob'], thr)
        thr_for_f1 = thr
        oof_m = compute_metrics(
            oof_dict['y_true'],
            np.column_stack([1 - oof_dict['y_prob'], oof_dict['y_prob']]),
            y_pred
        )
        row = {
            'Model': name, 'Source': 'Internal OOF summary',
            'AUC': round_to_4sf(safe_auc(oof_dict['y_true'], oof_dict['y_prob'])),
            'AUC_95CI': '',
            'Sensitivity': round_to_4sf(oof_m['sensitivity']),
            'Specificity': round_to_4sf(oof_m['specificity']),
            'F1': round_to_4sf(oof_m['f1']),
            'Accuracy': round_to_4sf(oof_m['accuracy']),
            'PPV': round_to_4sf(oof_m['ppv']), 'NPV': round_to_4sf(oof_m['npv']),
        }
        if oof_metrics_ci is None:
            oof_metrics_ci = compute_oof_metrics_with_ci(oof_dict['y_true'], oof_dict['y_prob'], threshold=thr_for_f1)
        row['AUC_95CI'] = _fmt_ci(oof_metrics_ci.get('auc_ci'))
        row['Brier'] = round_to_4sf(oof_metrics_ci.get('brier', float('nan')))
        row['Brier_95CI'] = _fmt_ci(oof_metrics_ci.get('brier_ci'))
        row['Cal_Slope'] = round_to_4sf(oof_metrics_ci.get('cal_slope', float('nan')))
        row['Cal_Slope_95CI'] = _fmt_ci(oof_metrics_ci.get('cal_slope_ci'))
        row['Cal_Intercept'] = round_to_4sf(oof_metrics_ci.get('cal_intercept', float('nan')))
        row['Cal_Intercept_95CI'] = _fmt_ci(oof_metrics_ci.get('cal_intercept_ci'))
        row['Sensitivity_95CI'] = _fmt_ci(oof_metrics_ci.get('sensitivity_ci'))
        row['Specificity_95CI'] = _fmt_ci(oof_metrics_ci.get('specificity_ci'))
        row['Accuracy_95CI'] = _fmt_ci(oof_metrics_ci.get('accuracy_ci'))
        row['F1_95CI'] = _fmt_ci(oof_metrics_ci.get('f1_ci'))
        row['PPV_95CI'] = _fmt_ci(oof_metrics_ci.get('ppv_ci'))
        row['NPV_95CI'] = _fmt_ci(oof_metrics_ci.get('npv_ci'))
        return row

    rows_internal = []
    rows_external = []

    # 1) Main model: per-fold mean
    cv_csv_path = os.path.join(output_dir, 'predictions', '10fold_cv_summary.csv')
    if os.path.exists(cv_csv_path):
        cv_table = pd.read_csv(cv_csv_path)
        numeric = cv_table[cv_table['Fold'] != 'Mean±Std']
        aucs = numeric['Val_AUC'].astype(float)
        sens = numeric['Sensitivity'].astype(float)
        specs = numeric['Specificity'].astype(float)
        f1s = numeric['F1'].astype(float)
        accs = numeric['Accuracy'].astype(float)
        rows_internal.append({
            'Model': 'Image-only linear LR (main model, per-fold mean)',
            'Source': 'Internal 10-fold CV',
            'AUC': round_to_4sf(aucs.mean()),
            'Sensitivity': round_to_4sf(sens.mean()),
            'Specificity': round_to_4sf(specs.mean()),
            'F1': round_to_4sf(f1s.mean()),
            'Accuracy': round_to_4sf(accs.mean()),
        })

    # 1b) Main-model OOF summary (internal-OOF global threshold)
    if main_oof_metrics_ci is None:
        main_oof_metrics_ci = compute_oof_metrics_with_ci(main_oof['y_true'], main_oof['y_prob'])
    _main_row = _build_oof_row('Image-only linear LR (main model, OOF summary)',
                               main_oof, main_oof_metrics_ci)
    _cross_prob = (main_oof.get('recalibration') or {}).get('oof_prob_cross_fitted')
    if _cross_prob is not None:
        _cross_prob = np.asarray(_cross_prob, dtype=float)
        _yt = np.asarray(main_oof['y_true'], dtype=int)
        _b_ci = bca_ci_general(_yt, _cross_prob,
                               lambda yt, yp: compute_brier_score(yt, yp),
                               N_BOOTSTRAP, 0.05, SEED)
        _cal_cross = calibration_metric_cis(_yt, _cross_prob, N_BOOTSTRAP, 0.05, SEED)
        _main_row.update({
            'Source': 'Internal OOF summary (calibration metrics = cross-fitted)',
            'Brier': round_to_4sf(_b_ci[0]),
            'Brier_95CI': _fmt_ci(_b_ci),
            'Cal_Slope': round_to_4sf(_cal_cross['cal_slope']),
            'Cal_Slope_95CI': _fmt_ci(_cal_cross['cal_slope_ci']),
            'Cal_Intercept': round_to_4sf(_cal_cross['cal_intercept']),
            'Cal_Intercept_95CI': _fmt_ci(_cal_cross['cal_intercept_ci']),
        })
    rows_internal.append(_main_row)

    # 2) Baseline per-fold mean
    baseline_order = [
        ('baseline_stat_only', 'Patch-level features MLP only (baseline)'),
        ('model_lightgbm', 'LightGBM (baseline)'),
        ('model_xgboost', 'XGBoost (baseline)'),
        ('model_elasticnet', 'ElasticNet Logistic Regression (baseline)'),
        ('model_randomforest', 'RandomForest (baseline)'),
    ]
    for key, name in baseline_order:
        if key not in baseline_cv_results:
            continue
        fr = baseline_cv_results[key]['fold_results']
        fold_aucs = [r['val_auc'] for r in fr]
        fold_sens = [r.get('sensitivity', float('nan')) for r in fr]
        fold_specs = [r.get('specificity', float('nan')) for r in fr]
        fold_f1s = [r.get('val_f1', float('nan')) for r in fr]
        rows_internal.append({
            'Model': name, 'Source': 'Internal 10-fold CV',
            'AUC': round_to_4sf(np.nanmean(fold_aucs)),
            'Sensitivity': round_to_4sf(np.nanmean(fold_sens)) if not all(np.isnan(fold_sens)) else float('nan'),
            'Specificity': round_to_4sf(np.nanmean(fold_specs)) if not all(np.isnan(fold_specs)) else float('nan'),
            'F1': round_to_4sf(np.nanmean(fold_f1s)) if not all(np.isnan(fold_f1s)) else float('nan'),
        })

    # 3) Baseline OOF summary
    for key, name in baseline_order:
        if key in baseline_oof_preds:
            ci = baseline_oof_ci_results.get(key) if baseline_oof_ci_results else None
            rows_internal.append(_build_oof_row(name, baseline_oof_preds[key], ci))

    # External
    def _build_ext_row(name, ext_metrics, auc_ci, brier_ci=None, cat_ci=None, f1_ci=None, cal=None):
        row = {
            'Model': name, 'Source': 'External test',
            'AUC': round_to_4sf(auc_ci[0]),
            'Sensitivity': round_to_4sf(ext_metrics['sensitivity']),
            'Specificity': round_to_4sf(ext_metrics['specificity']),
            'F1': round_to_4sf(ext_metrics['f1']),
            'Accuracy': round_to_4sf(ext_metrics['accuracy']),
            'PPV': round_to_4sf(ext_metrics.get('ppv', float('nan'))),
            'NPV': round_to_4sf(ext_metrics.get('npv', float('nan'))),
            'AUC_95CI': _fmt_ci(auc_ci),
        }
        if brier_ci is not None:
            row['Brier'] = round_to_4sf(brier_ci[0])
            row['Brier_95CI'] = _fmt_ci(brier_ci)
        if cal is not None:
            row['Cal_Slope'] = round_to_4sf(cal.get('cal_slope', float('nan')))
            row['Cal_Slope_95CI'] = _fmt_ci(cal.get('cal_slope_ci'))
            row['Cal_Intercept'] = round_to_4sf(cal.get('cal_intercept', float('nan')))
            row['Cal_Intercept_95CI'] = _fmt_ci(cal.get('cal_intercept_ci'))
        if f1_ci is not None:
            row['F1_95CI'] = _fmt_ci(f1_ci)
        if cat_ci is not None:
            row['Sensitivity_95CI'] = _fmt_ci(cat_ci.get('sensitivity_ci'))
            row['Specificity_95CI'] = _fmt_ci(cat_ci.get('specificity_ci'))
            row['PPV_95CI'] = _fmt_ci(cat_ci.get('ppv_ci'))
            row['NPV_95CI'] = _fmt_ci(cat_ci.get('npv_ci'))
            row['Accuracy_95CI'] = _fmt_ci(cat_ci.get('accuracy_ci'))
        return row

    ext_metrics = ext_state['ext_metrics']
    ext_auc_ci = ext_state['ext_auc_ci']
    ext_brier_ci = ext_state.get('ext_brier_ci')
    ext_cat_ci = ext_state.get('ext_cat_ci')
    ext_f1_ci = ext_state.get('ext_f1_ci')
    ext_cal = ext_state.get('ext_cal')
    rows_external.append(_build_ext_row(
        'Image-only linear LR (main model)', ext_metrics, ext_auc_ci,
        ext_brier_ci, ext_cat_ci, ext_f1_ci, ext_cal))

    pred_dir = os.path.join(output_dir, 'predictions')
    xlsx_path = os.path.join(pred_dir, 'all_models_comprehensive_summary.xlsx')
    cols_int = ['Model', 'Source', 'AUC', 'AUC_95CI',
                'Sensitivity', 'Sensitivity_95CI',
                'Specificity', 'Specificity_95CI',
                'F1', 'F1_95CI',
                'Accuracy', 'Accuracy_95CI',
                'PPV', 'PPV_95CI', 'NPV', 'NPV_95CI',
                'Brier', 'Brier_95CI',
                'Cal_Slope', 'Cal_Slope_95CI',
                'Cal_Intercept', 'Cal_Intercept_95CI']
    cols_ext = list(cols_int)

    def _recalib_row(dataset, summary, threshold, note):
        if not summary:
            return None
        return {
            'Dataset': dataset,
            'Threshold': round_to_4sf(threshold) if threshold is not None else '',
            'Brier': round_to_4sf(summary.get('brier')),
            'Cal_Slope': round_to_4sf(summary.get('slope')),
            'Cal_Intercept': round_to_4sf(summary.get('intercept')),
            'Mean predicted probability': round_to_4sf(summary.get('mean_pred')),
            'Observed positive rate': round_to_4sf(summary.get('obs_rate')),
            'Note': note,
        }

    recalib_rows = []
    _mr = main_oof.get('recalibration')
    if _mr is not None:
        recalib_rows.append(_recalib_row(
            'Internal OOF (raw probabilities)', _mr.get('oof_summary_raw'),
            _mr.get('threshold_raw'), 'Main model raw output'))
        recalib_rows.append(_recalib_row(
            'Internal OOF (recalibrated, in-sample fit)',
            _mr.get('oof_summary_in_sample'), main_oof.get('threshold'),
            'Platt calibrator fitted and evaluated on the same OOF (slightly optimistic)'))
        recalib_rows.append(_recalib_row(
            'Internal OOF (recalibrated, cross-fitted)',
            _mr.get('oof_summary_cross_fitted'), main_oof.get('threshold'),
            f'{RECALIBRATION_CV_FOLDS}-fold cross-fitting; honest calibration gain estimate'))
    if ext_state.get('ext_summary_raw'):
        _raw_thr = (_mr or {}).get('threshold_raw', ext_state.get('threshold'))
        recalib_rows.append(_recalib_row(
            'External test (raw probabilities)', ext_state.get('ext_summary_raw'),
            _raw_thr,
            'Main model raw output; threshold from pre-recalibration convention '
            '(external labels used for evaluation only)'))
        recalib_rows.append(_recalib_row(
            'External test (recalibrated)', ext_state.get('ext_summary_calibrated'),
            ext_state.get('threshold'),
            'Calibrator fitted on internal OOF transferred; external labels not used for fitting; '
            'threshold remapped by Platt (operating point unchanged)'))
    recalib_rows = [r for r in recalib_rows if r is not None]

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        if rows_internal:
            df_int = pd.DataFrame(rows_internal)
            df_int[[c for c in cols_int if c in df_int.columns]].to_excel(
                writer, sheet_name='Internal validation', index=False)
        if rows_external:
            df_ext = pd.DataFrame(rows_external)
            df_ext[[c for c in cols_ext if c in df_ext.columns]].to_excel(
                writer, sheet_name='External test', index=False)
        if recalib_rows:
            pd.DataFrame(recalib_rows).to_excel(
                writer, sheet_name='Recalibration comparison', index=False)

    wb = openpyxl.load_workbook(xlsx_path)
    for ws in wb.worksheets:
        for col_letter in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                           'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF']:
            ws.column_dimensions[col_letter].width = 20
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=4, max_col=ws.max_column):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000'
    wb.save(xlsx_path)


def calculate_dca(y_true, y_prob, thresholds=np.arange(0.01, 1.00, 0.01)):
    n = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return thresholds, np.array(net_benefits)


def plot_external_figures(ext_state, output_dir=OUTPUT_DIR):
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')

    main_name = 'Image-only linear LR (main model)'
    main_prob = ext_state['main_preds']['prob'][:, 1]
    y_true = ext_state['main_preds']['true']
    main_auc, main_ci_l, main_ci_u = ext_state['ext_auc_ci']

    model_data = {main_name: main_prob}

    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, main_prob)
    ax.plot(fpr, tpr, color='#d7191c', linestyle='-', lw=2,
            label=f'{main_name} (AUC={main_auc:.3f}, 95%CI {main_ci_l:.3f}-{main_ci_u:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve on External Test Set (Image-only Linear LR)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_roc_comparison.tiff'),
                dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}, facecolor='white')
    plt.close()
    print("  External ROC figure saved (TIFF 600dpi)")

    n_models = len(model_data)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    vmax = len(y_true)
    for idx, name in enumerate(model_data.keys()):
        ax = axes[idx]
        prob = model_data[name]
        y_pred = prob_to_class(prob, ext_state['threshold'])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASSES, fontsize=10); ax.set_yticklabels(CLASSES, fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10); ax.set_ylabel('True Label', fontsize=10)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=14)
    for idx in range(n_models, n_rows * n_cols):
        axes[idx].set_visible(False)
    fig.suptitle(f'Confusion Matrix on External Test Set '
                 f'(threshold={ext_state["threshold"]:.3f})', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_confusion_matrix_comparison.tiff'),
                dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}, facecolor='white')
    plt.close()
    print("  External confusion matrix figure saved")

    fig, ax = plt.subplots(figsize=(10, 8))
    thresholds = np.arange(0.01, 1.00, 0.01)
    prevalence = y_true.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    treat_none = np.zeros_like(thresholds)
    ax.plot(thresholds, treat_all, 'k--', lw=1.8, label='Treat all')
    ax.plot(thresholds, treat_none, 'k:', lw=1.8, label='Treat none')
    _, nb = calculate_dca(y_true, main_prob, thresholds)
    ax.plot(thresholds, nb, color='#d7191c', linestyle='-', lw=2, label=main_name)
    ax.set_xlabel('Threshold Probability', fontsize=13)
    ax.set_ylabel('Net Benefit', fontsize=13)
    ax.set_title('Decision Curve Analysis on External Test Set (Image-only Linear LR)',
                 fontsize=14, fontweight='bold')
    # DCA y-axis convention aligned with other tasks in the same project
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.3, 1.0)
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='gray', fancybox=False)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_dca_curves.tiff'),
                dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}, facecolor='white')
    plt.close()
    print("  External DCA decision curve saved")


def plot_internal_oof_roc(main_oof, main_oof_metrics_ci, baseline_oof_preds,
                          baseline_oof_ci_results=None, output_dir=OUTPUT_DIR):
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')

    def _model_threshold(oof_dict):
        thr = oof_dict.get('threshold')
        return thr if thr is not None else GLOBAL_THRESHOLD

    def _operating_point(yt, yp, thr):
        y_pred = prob_to_class(yp, thr)
        tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sens, spec

    oof_data = []
    main_thr = _model_threshold(main_oof)
    main_sens, main_spec = _operating_point(np.asarray(main_oof['y_true']),
                                            np.asarray(main_oof['y_prob']), main_thr)
    oof_data.append((
        'Image-only linear LR (main model)',
        np.asarray(main_oof['y_true']),
        np.asarray(main_oof['y_prob']),
        main_oof_metrics_ci.get('auc_ci'),
        main_thr, main_sens, main_spec,
    ))

    baseline_order = [
        ('baseline_stat_only', 'Patch-level features MLP only (baseline)'),
        ('model_lightgbm', 'LightGBM (baseline)'),
        ('model_xgboost', 'XGBoost (baseline)'),
        ('model_elasticnet', 'ElasticNet Logistic Regression (baseline)'),
        ('model_randomforest', 'RandomForest (baseline)'),
    ]
    for key, name in baseline_order:
        if key not in baseline_oof_preds:
            continue
        oof = baseline_oof_preds[key]
        ci = None
        if baseline_oof_ci_results and key in baseline_oof_ci_results:
            ci = baseline_oof_ci_results[key].get('auc_ci')
        yt = np.asarray(oof['y_true'])
        yp = np.asarray(oof['y_prob'])
        thr = _model_threshold(oof)
        sens, spec = _operating_point(yt, yp, thr)
        oof_data.append((name, yt, yp, ci, thr, sens, spec))

    colors = ['#d7191c', '#2c7bb6', '#1a9641', '#f9913a', '#6a3d9a',
              '#d01c8b', '#fdae61', '#39568c', '#dfc27d', '#000000']

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, yt, yp, auc_ci, thr, sens, spec) in enumerate(oof_data):
        fpr, tpr, _ = roc_curve(yt, yp)
        color = colors[i % len(colors)]
        if auc_ci is not None:
            label = (f'{name} (AUC={auc_ci[0]:.3f}, 95%CI {auc_ci[1]:.3f}-{auc_ci[2]:.3f})\n'
                     f'Thr={thr:.3f}, Sens={sens:.3f}, Spec={spec:.3f}')
        else:
            label = (f'{name} (AUC={safe_auc(yt, yp):.3f})\n'
                     f'Thr={thr:.3f}, Sens={sens:.3f}, Spec={spec:.3f}')
        ax.plot(fpr, tpr, color=color, linestyle='-', lw=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve on Internal OOF Summary '
                 '(per-model internal-OOF optimized thresholds)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'internal_roc_comparison.tiff'),
                dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}, facecolor='white')
    plt.close()
    print("  Internal OOF summary ROC figure saved (TIFF 600dpi)")


# Main pipeline
def main():
    global GLOBAL_THRESHOLD
    t_start = datetime.now()
    print("=" * 70)
    print("  Path features binary classification (MANEC vs NEC) multi-model comparison pipeline")
    print(f"  Global random seed: {SEED} | Started at: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU parallel mode: {'fold-level parallel' if N_GPUS > 1 else 'single GPU/CPU'} "
          f"| Available GPUs: {N_GPUS}")
    print("=" * 70)

    # Step 1: load data
    step_start = datetime.now()
    print("\n[Step 1] Loading training patch features (MANEC / NEC only) ...")
    sl_patch_dir = os.path.join(SL_DIR, 'SL-patch_features')
    case_ids, patch_list_all, categories_all = load_patch_features(sl_patch_dir)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 2: stratified 10-fold split
    step_start = datetime.now()
    print("\n[Step 2] Generating stratified 10-fold split ...")
    folds = load_or_create_folds(case_ids, categories_all)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Diagnostic: C-AUC curve (only checks whether LR_C_GRID is at boundary;
    # not used for selection)
    if ENABLE_C_AUC_DIAGNOSTIC:
        diag_start = datetime.now()
        diagnose_c_auc_curve(compute_mean_pooled_features(patch_list_all),
                             to_binary_labels(categories_all))
        print(f"  Elapsed: {datetime.now() - diag_start}")

    # Steps 3-8: main model: mean-pooled patch features + linear LASSO LR
    # (standardize on whole training fold + inner 5-fold CV to select C +
    # internal-OOF global threshold)
    step_start = datetime.now()
    print("\n[Step 3-8] Main model (mean-pooled patch features + linear LASSO LR) "
          "10-fold CV ...")
    main_fold_results = []
    main_oof_y_true = []
    main_oof_y_prob = []
    main_oof_case_ids = []
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    for fold_info in tqdm(folds,
                          desc="  Main model (mean-pooled patch features + linear LASSO LR)",
                          unit="fold"):
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        y_train_bin = to_binary_labels(categories_all[train_indices])
        y_val_bin = to_binary_labels(categories_all[val_indices])

        train_patches = [patch_list_all[i] for i in train_indices]
        val_patches = [patch_list_all[i] for i in val_indices]
        X_train_mean = compute_mean_pooled_features(train_patches)
        X_val_mean = compute_mean_pooled_features(val_patches)

        # Standardize on the whole training fold -> inner 5-fold CV to search optimal C
        scaler = StandardScaler()
        X_train_stat = scaler.fit_transform(X_train_mean)
        X_val_stat = scaler.transform(X_val_mean)
        best_C = select_lr_best_C(X_train_stat, y_train_bin)

        lr_main = LogisticRegression(penalty='l1', C=best_C, solver='saga',
                                     max_iter=LASSO_CV_MAX_ITER, random_state=SEED, class_weight=None)
        lr_main.fit(X_train_stat, y_train_bin)
        y_val_prob = lr_main.predict_proba(X_val_stat)
        y_val_class = prob_to_class(y_val_prob[:, 1], GLOBAL_THRESHOLD)
        val_auc = safe_auc(y_val_bin, y_val_prob[:, 1])

        pred_dir = os.path.join(OUTPUT_DIR, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({
            'case_id': val_cases,
            'true_label': [CLASSES[l] for l in y_val_bin],
            'pred_label': [CLASSES[p] for p in y_val_class],
            'prob_NEC': 1.0 - y_val_prob[:, 1],
            'prob_MANEC': y_val_prob[:, 1],
        }).to_csv(os.path.join(pred_dir, f'val_prediction_fold_{fold_idx}.csv'), index=False)

        main_fold_results.append({'fold': fold_idx, 'val_auc': val_auc, 'best_C': best_C})
        main_oof_y_true.extend(y_val_bin)
        main_oof_y_prob.extend(y_val_prob[:, 1])
        main_oof_case_ids.extend(val_cases)

    main_oof = {
        'case_ids': main_oof_case_ids,
        'y_true': np.array(main_oof_y_true),
        'y_prob': np.array(main_oof_y_prob),
        'display_name': 'Image-only linear LR (main model)'
    }
    # Threshold optimization: internal-OOF Youden maximization on the main model,
    # yielding its own threshold (also used as global fallback)
    threshold_config = None
    if ENABLE_THRESHOLD_OPTIMIZATION:
        print("\n[Post-processing] Main-model internal-OOF threshold optimization "
              "(Youden maximization) ...")
        thr_result = optimize_model_threshold('main', main_oof['y_true'], main_oof['y_prob'])
        GLOBAL_THRESHOLD = float(thr_result['threshold'])
        main_oof['threshold'] = GLOBAL_THRESHOLD
        threshold_config = {
            'optimal_threshold': GLOBAL_THRESHOLD,
            'youden_index': thr_result['youden'],
            'sensitivity': thr_result['sensitivity'],
            'specificity': thr_result['specificity'],
        }
        print(f"    Youden index maximization: main-model internal-OOF optimized "
              f"threshold={GLOBAL_THRESHOLD:.4f}, sensitivity={thr_result['sensitivity']:.4f}, "
              f"specificity={thr_result['specificity']:.4f}, "
              f"Youden={thr_result['youden']:.4f}")
        with open(os.path.join(OUTPUT_DIR, 'models', 'threshold_config.pkl'), 'wb') as f:
            pickle.dump(threshold_config, f)
    else:
        main_oof['threshold'] = GLOBAL_THRESHOLD
        print(f"  [Post-processing] Threshold optimization disabled; main model uses "
              f"global default threshold {GLOBAL_THRESHOLD:.4f}")

    # Main-model probability recalibration (Platt scaling): fitted on internal OOF only;
    # external labels do not participate. Threshold is remapped exactly via the
    # monotone Platt transform, keeping the per-case operating point unchanged.
    recalib_info = None
    main_oof_raw_prob = np.array(main_oof['y_prob'], dtype=float)
    if (ENABLE_MAIN_RECALIBRATION
            and len(main_oof_raw_prob) >= RECALIBRATION_MIN_SAMPLES
            and len(np.unique(main_oof['y_true'])) == 2):
        print("\n[Post-processing] Main-model probability recalibration "
              "(Platt scaling, fitted on internal OOF) ...")
        recalib_info = fit_platt_scaling(main_oof['y_true'], main_oof_raw_prob)
        oof_prob_cal = apply_platt_scaling(main_oof_raw_prob, recalib_info)
        oof_prob_cross = cross_fitted_platt_probabilities(
            main_oof['y_true'], main_oof_raw_prob)
        recalib_info.update({
            'fit_data': 'internal OOF',
            'n_fit': int(len(main_oof_raw_prob)),
            'threshold_raw': float(main_oof['threshold']),
            'threshold_calibrated': float(apply_platt_scaling(
                np.array([main_oof['threshold']], dtype=float), recalib_info)[0]),
            'threshold_note': 'threshold remapped exactly by Platt (operating point unchanged)',
            'oof_summary_raw': calibration_summary(main_oof['y_true'], main_oof_raw_prob),
            'oof_summary_in_sample': calibration_summary(main_oof['y_true'], oof_prob_cal),
            'oof_summary_cross_fitted': calibration_summary(main_oof['y_true'], oof_prob_cross),
            'oof_prob_cross_fitted': oof_prob_cross,
        })
        _rs = recalib_info['oof_summary_raw']
        _rci = recalib_info['oof_summary_in_sample']
        _rcf = recalib_info['oof_summary_cross_fitted']
        print(f"    Calibrator: p_cal = sigmoid({recalib_info['intercept']:+.4f} "
              f"{recalib_info['slope']:+.4f} * logit(p))")
        print(f"    Internal OOF: Brier {_rs['brier']:.4f} -> {_rci['brier']:.4f} "
              f"(honest cross-fitted estimate {_rcf['brier']:.4f}) | "
              f"Slope {_rs['slope']:.4f} -> {_rcf['slope']:.4f} (cross-fitted) | "
              f"Intercept {_rs['intercept']:+.4f} -> {_rcf['intercept']:+.4f} (cross-fitted)")
        _thr_raw = float(main_oof['threshold'])
        GLOBAL_THRESHOLD = float(apply_platt_scaling(
            np.array([_thr_raw], dtype=float), recalib_info)[0])
        main_oof['threshold'] = GLOBAL_THRESHOLD
        print(f"    Exact threshold remap: {_thr_raw:.4f} -> {GLOBAL_THRESHOLD:.4f} "
              f"(operating point unchanged)")
        main_oof['y_prob_raw'] = main_oof_raw_prob
        main_oof['y_prob'] = oof_prob_cal
        main_oof['recalibration'] = recalib_info
        if threshold_config is not None:
            threshold_config['optimal_threshold'] = GLOBAL_THRESHOLD
            threshold_config['threshold_raw'] = _thr_raw
            threshold_config['recalibration'] = recalib_info
            with open(os.path.join(OUTPUT_DIR, 'models', 'threshold_config.pkl'), 'wb') as f:
                pickle.dump(threshold_config, f)
        with open(os.path.join(OUTPUT_DIR, 'models', 'main_recalibration.pkl'), 'wb') as f:
            pickle.dump(recalib_info, f)
    else:
        print("  [Post-processing] Disabled / insufficient samples; "
              "skipping main-model probability recalibration")

    # Compute OOF metric CIs once after the threshold is finalized (avoid computing
    # at 0.5 then discarding)
    main_oof_ci = compute_oof_metrics_with_ci(main_oof['y_true'], main_oof['y_prob'],
                                              threshold=main_oof['threshold'])

    print("\n  Main-model 10-fold CV summary:")
    auc_mean, ci_l, ci_u = main_oof_ci['auc_ci']
    print(f"    OOF AUC: {auc_mean:.4f} (95% BCa CI: {ci_l:.4f} - {ci_u:.4f})")
    print(f"    OOF Brier: {main_oof_ci['brier']:.4g} "
          f"(95% BCa CI: {main_oof_ci['brier_ci'][1]:.4g}-{main_oof_ci['brier_ci'][2]:.4g})")
    print(f"    Cal. Slope: {main_oof_ci['cal_slope']:.4g} "
          f"(95% BCa CI: {main_oof_ci['cal_slope_ci'][1]:.4g}-{main_oof_ci['cal_slope_ci'][2]:.4g})")
    print(f"    Cal. Intercept: {main_oof_ci['cal_intercept']:.4g} "
          f"(95% BCa CI: {main_oof_ci['cal_intercept_ci'][1]:.4g}-{main_oof_ci['cal_intercept_ci'][2]:.4g})")
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Export 10-fold summary table (main model = mean-pooled patch features + linear
    # LASSO LR; recompute classification metrics at the global threshold from
    # per-fold OOF probabilities)
    fold_idx_to_res = {r['fold']: r for r in main_fold_results}
    cv_rows = []
    for fold_idx in range(N_FOLDS):
        fold_csv_path = os.path.join(OUTPUT_DIR, 'predictions', f'val_prediction_fold_{fold_idx}.csv')
        df = pd.read_csv(fold_csv_path)
        yt = (df['true_label'] == 'MANEC').astype(int).values
        prob_raw = df['prob_MANEC'].values
        prob = apply_platt_scaling(prob_raw, recalib_info)
        pred = prob_to_class(prob, GLOBAL_THRESHOLD)
        # Per-fold CSV pred_label must reflect the final OOF threshold (files were
        # written with fallback 0.5); probability columns: prob_* are
        # post-recalibration (final), prob_*_raw are model raw outputs
        df['prob_NEC_raw'] = 1.0 - prob_raw
        df['prob_MANEC_raw'] = prob_raw
        df['prob_NEC'] = 1.0 - prob
        df['prob_MANEC'] = prob
        df['pred_label'] = [CLASSES[p] for p in pred]
        df.to_csv(fold_csv_path, index=False)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        fr = fold_idx_to_res.get(fold_idx, {})
        cv_rows.append({
            'Fold': fold_idx,
            'Best_C': fr.get('best_C', ''),
            'Val_AUC': safe_auc(yt, prob_raw),
            'Accuracy': accuracy_score(yt, pred),
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'F1': f1_score(yt, pred, zero_division=0),
        })
    cv_summary_df = pd.DataFrame(cv_rows)
    cv_summary_df.loc['Mean±Std'] = [
        'Mean±Std', '',
        f"{cv_summary_df['Val_AUC'].mean():.4f}±{cv_summary_df['Val_AUC'].std():.4f}",
        f"{cv_summary_df['Accuracy'].mean():.4f}±{cv_summary_df['Accuracy'].std():.4f}",
        f"{cv_summary_df['Sensitivity'].mean():.4f}±{cv_summary_df['Sensitivity'].std():.4f}",
        f"{cv_summary_df['Specificity'].mean():.4f}±{cv_summary_df['Specificity'].std():.4f}",
        f"{cv_summary_df['F1'].mean():.4f}±{cv_summary_df['F1'].std():.4f}",
    ]
    cv_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions', '10fold_cv_summary.csv'), index=False)
    print(f"    10-fold summary table saved: predictions/10fold_cv_summary.csv "
          f"(global threshold={GLOBAL_THRESHOLD:.4f})")

    # Step 9: external test (threshold from main-model internal OOF; external
    # labels do not participate)
    step_start = datetime.now()
    print("\n[Step 9] External test inference (main model) ...")
    ext_state_path = os.path.join(OUTPUT_DIR, 'predictions', 'external_test_state.pkl')
    ext_state = None
    if os.path.exists(ext_state_path):
        try:
            with open(ext_state_path, 'rb') as f:
                ext_state = pickle.load(f)
            _fy_dir = os.path.join(FY_DIR, 'FY-patch_features')
            _cur_rec = _recalibration_signature(recalib_info)
            _ok = (ext_state.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION
                   and 'main_preds' in ext_state
                   and ext_state.get('signature') == get_analysis_signature()
                   and ext_state.get('sl_fp') == _data_fingerprint(case_ids, categories_all)
                   and ext_state.get('fy_fp') == _patch_dir_fingerprint(_fy_dir)
                   and ext_state.get('recalibration_sig', 'none') == _cur_rec
                   and abs(float(ext_state.get('threshold', -1)) - float(GLOBAL_THRESHOLD)) < 1e-9)
            if _ok:
                print("  Found cached external test results "
                      "(signature/data/threshold/calibrator consistent); loading ...")
            else:
                print(f"  External test cache does not match current "
                      f"signature/data/threshold/calibrator "
                      f"(cache thr={ext_state.get('threshold')}, "
                      f"current thr={GLOBAL_THRESHOLD:.4f}, "
                      f"cache cal={ext_state.get('recalibration_sig', 'none')}, "
                      f"current cal={_cur_rec}); recomputing ...")
                ext_state = None
        except Exception as _e:
            print(f"  Failed to read external test cache ({_e}); recomputing ...")
            ext_state = None
    if ext_state is None:
        ext_state = process_external_test(threshold_config=threshold_config,
                                          main_threshold=GLOBAL_THRESHOLD,
                                          recalibration=recalib_info)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 10: baseline models + visualization
    step_start = datetime.now()
    print("\n[Step 10] Baseline models (MLP + additional models) "
          "training and comparison visualization ...")

    baseline_cv_path = os.path.join(OUTPUT_DIR, 'baselines', 'baselines_cv.pkl')
    baseline_cv_results = baseline_oof_preds = baseline_oof_ci_results = None
    if os.path.exists(baseline_cv_path):
        try:
            with open(baseline_cv_path, 'rb') as f:
                data = pickle.load(f)
            if (data.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION
                    and data.get('signature') == get_analysis_signature()
                    and data.get('folds_fingerprint') == _folds_fingerprint(folds)):
                baseline_cv_results = data['results']
                baseline_oof_preds = data['oof_preds']
                baseline_oof_ci_results = data.get('oof_ci_results', {})
                print("    Loaded baseline-model cache (signature/fold split consistent); "
                      "reusing OOF classification metric CIs")
            else:
                print("    Baseline-model cache does not match current signature/fold split; "
                      "recomputing ...")
        except Exception as _e:
            print(f"    Failed to read baseline-model cache ({_e}); recomputing ...")
            baseline_cv_results = None
    if baseline_cv_results is None:
        baseline_cv_results, baseline_oof_preds, baseline_oof_ci_results = run_baselines_cv(
            folds, patch_list_all, categories_all, case_ids
        )

    plot_external_figures(ext_state)
    plot_internal_oof_roc(main_oof, main_oof_ci, baseline_oof_preds,
                          baseline_oof_ci_results=baseline_oof_ci_results)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Comprehensive Excel summary
    print("\n[Summary] Generating comprehensive model comparison Excel ...")
    _generate_comprehensive_summary(main_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=baseline_oof_ci_results,
                                     main_oof_metrics_ci=main_oof_ci)
    print("  Excel saved: predictions/all_models_comprehensive_summary.xlsx")

    t_end = datetime.now()
    print(f"\n{'='*70}")
    print(f"  Pipeline completed! Total elapsed: {t_end - t_start}")
    print(f"  Results directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
