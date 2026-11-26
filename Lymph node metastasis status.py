#!/usr/bin/env python3
"""
Path features (MANEC cases) binary-classification multi-model comparison pipeline
for two independent prediction tasks.
Global random seed: 2026; results are fully reproducible.

Path configuration (all paths can be overridden by environment variables):
  WSI_DATA_DIR        : root dir of feature data (contains SL/ and FY/ subdirs)
  WSI_LABEL_DIR       : dir containing the label xlsx files
  WSI_SL_LABEL_XLSX   : training/internal-validation label file
  WSI_FY_LABEL_XLSX   : external-test label file
  WSI_RESULT_ROOT     : root dir for per-task result subdirectories
"""

import os
import re
import sys
import glob
import hashlib
import pickle
import warnings
from copy import deepcopy
from datetime import datetime
import multiprocessing

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
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
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

sys.modules.setdefault('manec_pipeline', sys.modules[__name__])

# Convergence diagnostics: global warnings are suppressed; here we re-enable
# ConvergenceWarning and count occurrences (no spam) to check whether
# LASSO_CV_MAX_ITER is large enough. Counts are printed once per task.
CONVERGENCE_HITS = []


def _convergence_warning_hook(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, ConvergenceWarning) and len(CONVERGENCE_HITS) < 1000:
        CONVERGENCE_HITS.append(f'{os.path.basename(filename)}:{lineno} {message}')


warnings.showwarning = _convergence_warning_hook
warnings.filterwarnings('always', category=ConvergenceWarning)

# GPU and CPU hardware detection (fully utilize all hardware resources)
N_CPUS = multiprocessing.cpu_count()
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

os.environ['OMP_NUM_THREADS'] = str(N_CPUS)
os.environ['MKL_NUM_THREADS'] = str(N_CPUS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CPUS)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_CPUS)
torch.set_num_threads(N_CPUS)

if N_GPUS > 0:
    print(f"\n  [Hardware] Detected {N_GPUS} GPU(s): "
          f"{[torch.cuda.get_device_name(i) for i in range(N_GPUS)]}")
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
PATCH_CATEGORY = 'MANEC'              # Only load MANEC-category patch features

# Current-task class names / label column (assigned by set_active_task;
# index 0=negative, 1=positive)
CLASSES = ['Negative', 'Positive']
CURRENT_TASK = None
CURRENT_LABEL_COL = None

# Label files (MANEC cases): SL = training/internal validation, FY = external test
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.environ.get('WSI_LABEL_DIR', os.path.join(_SCRIPT_DIR, 'labels'))
SL_LABEL_XLSX = os.environ.get(
    'WSI_SL_LABEL_XLSX',
    os.path.join(LABEL_DIR, 'SL-MANEC_LN_metastasis_NEC.xlsx'))
FY_LABEL_XLSX = os.environ.get(
    'WSI_FY_LABEL_XLSX',
    os.path.join(LABEL_DIR, 'FY-MANEC_LN_metastasis_NEC.xlsx'))
LABEL_ID_COL = 'ID'

RESULT_ROOT = os.environ.get(
    'WSI_RESULT_ROOT', os.path.join(_SCRIPT_DIR, 'results'))

# Two prediction tasks: executed sequentially with independent output dirs.
#   external_preproc: standardization for external test
#     'train_cohort' : external features standardized with SL training-cohort
#                      mean/std (same scaler as training)
#   internal_scaling: feature scaling mode
#     'mean_var'  : subtract mean and divide by std (default)
#     'mean_only' : subtract mean only
#   include_filter: analysis-subpopulation restriction (None = all cases)
#     {'col': ..., 'value': ...} -> keep only rows with col == value
#                                   (applied to both training and external test)
#   internal_preproc: within-task internal CV standardization
#     'per_fold'    : fit StandardScaler on each outer training fold (default, no leakage)
#     'per_cohort'  : fit on the full SL cohort once, used for all folds
#                     (diagnostic; validation-fold info enters the statistics)
#   n_folds: outer CV fold count for the task (minority class must be >= n_folds)
#   main_model: model used as main model for the external test
#     'lasso_lr'      : mean-pooled + LASSO(L1) logistic regression (task-1 default)
#     'elasticnet_lr' : mean-pooled + ElasticNet logistic regression
#                       (task-2; chosen by internal OOF AUC)
#   threshold_rule: OOF threshold rule for the task (only affects the
#                   classification operating point, not AUC)
#     'equal_error' : minimize |Sens-Spec|
#     'youden'      : maximize Sens+Spec-1
#     'composite'   : maximize (Youden + Accuracy + (1-|Sens-Spec|))/3,
#                     ties broken by Youden then Accuracy
#
# IMPORTANT: 'label_col' and 'include_filter.col' must match the column names
# in the input Excel label files. The defaults below are English; if your Excel
# files use Chinese column names (e.g. '淋巴结是否转移', '是否有NEC成分'), update
# these strings, or rename the columns in your Excel, or set env-var overrides.
TASKS = [
    {
        'key': 'ln_metastasis',
        'task_name': 'Lymph node metastasis',
        'label_col': 'LN metastasis',
        'output_dir': os.path.join(RESULT_ROOT, 'LN_metastasis'),
        'classes': ['No metastasis', 'Metastasis'],
        # External test standardizes with the SL training-cohort mean/std
        # (consistent with internal training; external labels do not participate)
        'external_preproc': 'train_cohort',
        'internal_scaling': 'mean_only',
        'include_filter': None,
        'n_folds': 10,
        'threshold_rule': 'composite',
        # Internal OOF reporting and external test both use ElasticNet LR
        'main_model': 'elasticnet_lr',
    },
    {
        # Main analysis restricted to "lymph-node metastasis positive" cases:
        # asks whether an NEC component is present among the metastasized node.
        # NEC-positive rate is zero in metastasis-negative cases (0/20 and 0/11);
        # including them would confound with the easier "metastasis yes/no" task.
        'key': 'nec_component',
        'task_name': 'NEC component present (metastasis-positive cases only)',
        'label_col': 'NEC component',
        'output_dir': os.path.join(RESULT_ROOT, 'NEC_component'),
        'classes': ['No NEC component', 'NEC component present'],
        # External test standardizes with the SL training-cohort mean/std
        'external_preproc': 'train_cohort',
        'include_filter': {'col': 'LN metastasis', 'value': 1},
        'n_folds': 5,
        'threshold_rule': 'composite',
        # Best internal-OOF AUC among candidates is ElasticNet LR; use it as
        # external-test main model
        'main_model': 'elasticnet_lr',
    },
]

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
# LASSO-LR C selection: repeated inner CV + argmax of mean AUC
#   - repeated inner CV: 5-fold CV with different seeds, LR_INNER_CV_REPEATS times,
#     to reduce split-induced noise
#   - rule: take the C with the largest mean AUC over repeats (no 1-SE tolerance)
LR_INNER_CV_REPEATS = 5
# LR/LASSO: inner 5-fold CV within the training fold to search the optimal C grid
# Empirical C-AUC curves (5-fold CV): metastasis task peaks near C~10 and then
# declines; NEC-component task plateaus near C~50. The old grid
# [0.01, 0.1, 1, 2, 3] has neither end in the effective region (C=0.01 yielded
# a degenerate model that was never selected), so the grid is widened empirically.
LR_C_GRID = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0,
             100.0, 150.0, 200.0, 300.0, 500.0]

# NN: inner split ratio within the training fold (80% train / 20% inner val,
# for early stopping only)
NN_INNER_VAL_FRAC = 0.2

N_BOOTSTRAP = 10000

ENABLE_THRESHOLD_OPTIMIZATION = True
# Threshold rules: 'youden' (maximize Sens+Spec-1) | 'equal_error' (minimize |Sens-Spec|)
#                  | 'composite' (equal-weight Youden / Accuracy / balance)
THRESHOLD_RULE = 'equal_error'
GLOBAL_THRESHOLD = 0.5
MODEL_THRESHOLDS = {}

LR_LEAK_FIX_VERSION = 19

# Additional models: hyperparameter grids (inner nested CV optimization,
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

# Directory configuration
DATA_DIR = os.environ.get('WSI_DATA_DIR', os.path.join(_SCRIPT_DIR, 'data'))
SL_DIR = os.path.join(DATA_DIR, 'SL')
FY_DIR = os.path.join(DATA_DIR, 'FY')

# Current-task output directory (assigned by set_active_task)
OUTPUT_DIR = TASKS[0]['output_dir']

OUTPUT_SUBDIRS = ['folds', 'models', 'predictions', 'figures', 'baselines']


def set_active_task(task):
    """Switch the active task: reset output dir, class names and threshold
    state; create output subdirectories."""
    global OUTPUT_DIR, CLASSES
    global CURRENT_TASK, CURRENT_LABEL_COL
    global GLOBAL_THRESHOLD, MODEL_THRESHOLDS
    OUTPUT_DIR = task['output_dir']
    CLASSES = list(task['classes'])
    CURRENT_TASK = task
    CURRENT_LABEL_COL = task['label_col']
    GLOBAL_THRESHOLD = 0.5
    MODEL_THRESHOLDS = {}
    for subdir in OUTPUT_SUBDIRS:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
    return OUTPUT_DIR


def get_ext_preproc_mode():
    """External-test standardization mode ('train_cohort': use SL training-cohort
    mean/std)."""
    if CURRENT_TASK is None:
        return 'train_cohort'
    return CURRENT_TASK.get('external_preproc', 'train_cohort')


def get_include_filter():
    """Analysis-subpopulation filter for the current task (None = all cases)."""
    if CURRENT_TASK is None:
        return None
    return CURRENT_TASK.get('include_filter')


def get_internal_preproc():
    """Internal CV standardization mode ('per_fold' | 'per_cohort')."""
    if CURRENT_TASK is None:
        return 'per_fold'
    return CURRENT_TASK.get('internal_preproc', 'per_fold')


def get_internal_scaling():
    """Feature scaling mode ('mean_var': subtract mean, divide std;
    'mean_only': subtract mean only)."""
    if CURRENT_TASK is None:
        return 'mean_var'
    return CURRENT_TASK.get('internal_scaling', 'mean_var')


def new_feature_scaler():
    """Build a feature scaler for the current task (mean_only skips variance
    scaling)."""
    return StandardScaler(with_std=(get_internal_scaling() != 'mean_only'))


def get_n_folds():
    """Outer CV fold count for the current task."""
    if CURRENT_TASK is None:
        return N_FOLDS
    return int(CURRENT_TASK.get('n_folds', N_FOLDS))


def get_threshold_rule():
    """OOF threshold rule for the current task ('equal_error' | 'youden' |
    'composite')."""
    if CURRENT_TASK is None:
        return THRESHOLD_RULE
    return CURRENT_TASK.get('threshold_rule', THRESHOLD_RULE)


def get_main_model():
    """External-test main model for the current task ('lasso_lr' | 'elasticnet_lr')."""
    if CURRENT_TASK is None:
        return 'lasso_lr'
    return CURRENT_TASK.get('main_model', 'lasso_lr')


def get_main_model_name():
    """Display name of the main model."""
    if get_main_model() == 'elasticnet_lr':
        return 'ElasticNet Logistic Regression'
    return 'LASSO(L1) Logistic Regression'


def get_analysis_signature():
    """Current-task analysis signature used for cache validation
    (population filter / folds / label column / main model / hyperparameter
    grids / seed).

    Any configuration that affects model selection or threshold must be
    included; otherwise stale caches would be silently reused after changes.
    """
    if CURRENT_TASK is None:
        return ''
    f = get_include_filter()
    fdesc = f"{f['col']}=={f['value']}" if f else 'ALL'
    return (f"{CURRENT_TASK['key']}|{CURRENT_TASK['label_col']}|{fdesc}|"
            f"folds={get_n_folds()}|thr={get_threshold_rule()}|"
            f"iprep={get_internal_preproc()}|scale={get_internal_scaling()}|"
            f"main={get_main_model()}|seed={SEED}|boot={N_BOOTSTRAP}|"
            f"lr={list(LR_C_GRID)}|en={ELASTICNET_GRID}|"
            f"lgb={LIGHTGBM_GRID}|xgb={XGBOOST_GRID}|rf={RANDOMFOREST_GRID}")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(SEED)


def load_label_table(xlsx_path, label_col, include_filter=None):
    """Read the label xlsx and return (case_ids, binary-label array).

    y=1 means positive (LN metastasis present / NEC component present); y=0
    means negative. include_filter={'col': ..., 'value': ...} restricts the
    analysis population to rows where col == value (used e.g. to restrict to
    metastasis-positive cases).
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Label file not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    if LABEL_ID_COL not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"{os.path.basename(xlsx_path)} is missing column '{LABEL_ID_COL}' "
            f"or '{label_col}'; actual columns: {list(df.columns)}")
    if include_filter is not None:
        fcol, fval = include_filter['col'], include_filter['value']
        if fcol not in df.columns:
            raise KeyError(
                f"{os.path.basename(xlsx_path)} is missing population-filter "
                f"column '{fcol}'; actual columns: {list(df.columns)}")
        n_before = len(df)
        df = df[df[fcol].astype(int) == int(fval)].reset_index(drop=True)
        print(f"  Population filter: '{fcol}'=={fval} -> retained {len(df)}/{n_before} cases")
        if len(df) == 0:
            raise ValueError(f"No remaining cases after population filter: {fcol}=={fval}")
    ids = df[LABEL_ID_COL].astype(str).str.strip().tolist()
    if len(set(ids)) != len(ids):
        dup = sorted({c for c in ids if ids.count(c) > 1})
        raise ValueError(f"Duplicate IDs in label file {os.path.basename(xlsx_path)}: {dup}")
    y = df[label_col].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Label column '{label_col}' must be 0/1; "
                         f"actual: {sorted(np.unique(y))}")
    if len(np.unique(y)) < 2:
        raise ValueError(f"After population filter, label column '{label_col}' "
                         f"has a single class only; cannot train: {np.unique(y)}")
    print(f"  Labels read: {os.path.basename(xlsx_path)} | col='{label_col}' | n={len(ids)} | "
          f"positive(1)={int(y.sum())} negative(0)={int((y == 0).sum())}")
    return ids, y


_TRAILING_INDEX_RE = re.compile(r'^(.*?)-\d+$')


def normalize_patch_case_id(case_id):
    """Strip trailing numeric index suffix from a patch filename
    (e.g. FY-A1-005 -> FY-A1).

    Used only when the stripped ID actually exists in the label file; see
    load_patch_features_aligned.
    """
    m = _TRAILING_INDEX_RE.match(case_id)
    return m.group(1) if m else case_id


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
    # z0 must be clipped: if every bootstrap AUC is >= (or <) the observed value
    # (perfect separation / saturation), mean(...) hits 0 or 1, norm.ppf returns
    # +/-inf, and the CI degenerates to NaN.
    p_less = np.mean(boot_aucs < obs_auc)
    p_less = np.clip(p_less, 1.0 / (len(boot_aucs) + 1), 1.0 - 1.0 / (len(boot_aucs) + 1))
    z0 = norm.ppf(p_less)
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
    # z0 must be clipped; see auc_bca_ci.
    p_less = np.mean(boot_vals < obs)
    p_less = np.clip(p_less, 1.0 / (len(boot_vals) + 1), 1.0 - 1.0 / (len(boot_vals) + 1))
    z0 = norm.ppf(p_less)
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
    """Calibration slope/intercept: logistic regression of outcome on logit(p) as
    the only covariate.

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
    """BCa 95% CIs for calibration slope and intercept (class-stratified
    resampling, consistent with other metrics).

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
            boot_s.append(sl)
            boot_i.append(it)

    jack_s, jack_i = [], []
    for i in range(len(y)):
        j = np.delete(np.arange(len(y)), i)
        sl, it = fit_calibration_slope_intercept(y[j], p[j])
        if np.isfinite(sl):
            jack_s.append(sl)
            jack_i.append(it)

    return {'cal_slope': float(obs_s),
            'cal_slope_ci': _bca_ci_from_samples(obs_s, boot_s, jack_s, alpha),
            'cal_intercept': float(obs_i),
            'cal_intercept_ci': _bca_ci_from_samples(obs_i, boot_i, jack_i, alpha)}


def prob_to_class(prob, threshold=None):
    """Probability -> class mapping: prob >= threshold -> 1 (positive),
    otherwise 0 (negative).

    threshold is a scalar; if None, uses GLOBAL_THRESHOLD (i.e., the unique
    global threshold optimized on this model's own internal OOF).
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
    """Compute OOF metrics with BCa bootstrap CIs. Classification metrics use
    the given threshold."""
    if threshold is None:
        threshold = GLOBAL_THRESHOLD
    results = {}
    results['brier'] = compute_brier_score(y_true, y_prob)
    results['brier_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: compute_brier_score(yt, yp), n_bootstrap, alpha, seed)
    results['auc_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: safe_auc(yt, yp), n_bootstrap, alpha, seed)
    results['f1_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: f1_score(yt, prob_to_class(yp, threshold)), n_bootstrap, alpha, seed)
    # Calibration slope/intercept (logit regression; ideal slope=1 / intercept=0)
    results.update(calibration_metric_cis(y_true, y_prob, n_bootstrap, alpha, seed))
    results.update(classification_metric_cis(y_true, y_prob, threshold=threshold,
                                             n_bootstrap=n_bootstrap, alpha=alpha, seed=seed))
    return results


# Data loading (MANEC category only; strictly aligned to label-file case IDs)
def load_patch_features_aligned(patch_dir, label_xlsx, label_col, category=PATCH_CATEGORY,
                                include_filter=None):
    """Load raw patch features for the given category (default MANEC) and align
    them case-by-case to the label file.

    Alignment: the label-file ID order is authoritative. If a patch filename
    carries a trailing numeric index suffix (e.g. FY-A1-005 -> FY-A1), the
    stripped ID is used only when it matches an existing label-file ID.
    include_filter restricts the analysis population (e.g., metastasis-positive
    cases only).

    Returns (case_ids, patch_list, y): case_ids and y follow the (filtered)
    label-file order exactly.
    """
    label_ids, y = load_label_table(label_xlsx, label_col, include_filter=include_filter)
    label_id_set = set(label_ids)

    # First match by raw ID; for unmatched ones, try stripping the trailing suffix
    id_to_path = {}
    unmatched = []
    for npz_path in sorted(glob.glob(os.path.join(patch_dir, category, '*_features.npz'))):
        raw_id = os.path.basename(npz_path).rsplit('_features', 1)[0]
        cid = raw_id if raw_id in label_id_set else normalize_patch_case_id(raw_id)
        if cid not in label_id_set:
            unmatched.append(raw_id)
            continue
        if cid in id_to_path:
            raise ValueError(
                f"Patch-feature ID conflict: '{os.path.basename(id_to_path[cid])}' and "
                f"'{os.path.basename(npz_path)}' both map to label ID '{cid}'")
        id_to_path[cid] = npz_path

    if len(id_to_path) == 0:
        raise FileNotFoundError(f"No .npz files found: {patch_dir}/{category}/*_features.npz")
    missing = [cid for cid in label_ids if cid not in id_to_path]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cases in the label file are missing from patch features: {missing}")
    if unmatched:
        if include_filter is not None:
            print(f"  Population filter excluded {len(unmatched)} cases "
                  f"(not in {include_filter['col']}=={include_filter['value']})")
        else:
            print(f"  [Info] {len(unmatched)} patch files did not match a label ID and "
                  f"were ignored: {unmatched}")

    case_ids, patch_list = [], []
    for cid in label_ids:
        data = np.load(id_to_path[cid])
        patch_list.append(data['features'].astype(np.float32))
        case_ids.append(cid)

    n_patches = [p.shape[0] for p in patch_list]
    print(f"  Loaded raw patch features: {len(case_ids)} cases ({category}), "
          f"{PATCH_FEAT_DIM}-dim / patches")
    print(f"  Patches per case: min={min(n_patches)}, max={max(n_patches)}, "
          f"median={np.median(n_patches):.0f}")
    print(f"  Label distribution: {CLASSES[0]}(0)={int((y == 0).sum())}, "
          f"{CLASSES[1]}(1)={int(y.sum())}")
    return case_ids, patch_list, y


def compute_mean_pooled_features(patch_list):
    features = np.array([np.mean(p, axis=0) for p in patch_list], dtype=np.float32)
    return features


def min_class_count(labels):
    """Return the minority-class sample count (for stratified-CV feasibility)."""
    labels = np.asarray(labels)
    return int(min(np.sum(labels == c) for c in np.unique(labels)))


def _folds_fingerprint(case_ids, labels, n_folds, seed=SEED):
    """Fold-cache fingerprint: case order + label values + fold count + seed.

    Validating only (fold count + case set) cannot detect revised labels and
    would silently reuse folds that are mismatched with the current labels.
    """
    h = hashlib.sha1()
    h.update('\n'.join(map(str, case_ids)).encode('utf-8'))
    h.update(b'|labels|')
    h.update('\n'.join(str(int(v)) for v in np.asarray(labels).ravel()).encode('utf-8'))
    h.update(f'|n_folds={n_folds}|seed={seed}'.encode('utf-8'))
    return h.hexdigest()


def generate_stratified_folds(case_ids, labels, n_folds=N_FOLDS, seed=SEED, save_path=None,
                              fingerprint=None):
    """Stratified K-fold split by the current-task binary labels (cached per task)."""
    case_ids = np.array(case_ids)
    labels = np.array(labels)
    n_min = min_class_count(labels)
    if n_min < n_folds:
        dist = {int(c): int(np.sum(labels == c)) for c in np.unique(labels)}
        raise ValueError(
            f"Stratified {n_folds}-fold infeasible: minority class has only {n_min} "
            f"cases (< n_splits={n_folds}). Global warning suppression is active in "
            f"this script, so sklearn's warning would be hidden and stratification "
            f"would silently degrade. Current label distribution={dist}; reduce "
            f"n_folds for this task or switch label column.")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(case_ids, labels)):
        train_cases = case_ids[train_idx].tolist()
        val_cases = case_ids[val_idx].tolist()
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        folds.append({
            'fold': fold_idx,
            'train_cases': train_cases,
            'val_cases': val_cases,
            'train_dist': {'neg': int(np.sum(train_labels == 0)),
                           'pos': int(np.sum(train_labels == 1))},
            'val_dist': {'neg': int(np.sum(val_labels == 0)),
                         'pos': int(np.sum(val_labels == 1))},
        })
        print(f"  Fold {fold_idx}: train={len(train_cases)} val={len(val_cases)} "
              f"(val pos={int(np.sum(val_labels == 1))}, neg={int(np.sum(val_labels == 0))})")
    if save_path:
        payload = {'folds': folds, 'fingerprint': fingerprint} if fingerprint else folds
        with open(save_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"  {n_folds}-fold split saved: {save_path}")
    return folds


def load_or_create_folds(case_ids, labels, output_dir=None, n_folds=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if n_folds is None:
        n_folds = get_n_folds()
    folds_path = os.path.join(output_dir, 'folds', 'stratified_folds.pkl')
    fingerprint = _folds_fingerprint(case_ids, labels, n_folds)
    if os.path.exists(folds_path):
        with open(folds_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            folds = data.get('folds', [])
            cached_fp = data.get('fingerprint')
        else:
            folds = data
            cached_fp = None
        cached_val = sorted(c for fi in folds for c in fi['val_cases'])
        if len(folds) == n_folds and cached_val == sorted(case_ids) and cached_fp == fingerprint:
            print(f"  Loaded existing fold split: {folds_path} ({len(folds)} folds)")
            return folds
        if cached_fp is None:
            print("  Cached fold split lacks label fingerprint (old format); regenerating ...")
        else:
            print(f"  Cached fold split does not match current configuration "
                  f"(cache {len(folds)} folds / {len(cached_val)} cases, "
                  f"current {n_folds} folds / {len(case_ids)} cases); regenerating ...")
    return generate_stratified_folds(case_ids, labels, n_folds=n_folds,
                                     save_path=folds_path, fingerprint=fingerprint)


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


# Mean-pooled feature standardizer (for linear / tree models)
class BranchA_StatPreprocessor:
    def __init__(self, seed=SEED):
        self.seed = seed
        self.scaler = None
        self.output_dim = None

    def fit(self, X_train_raw, y=None):
        self.scaler = new_feature_scaler()
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
    prob_pos = y_pred_prob[:, 1] if y_pred_prob.ndim > 1 else y_pred_prob
    metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['auc'] = safe_auc(y_true, prob_pos)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class, labels=[0, 1]).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['f1'] = f1_score(y_true, y_pred_class)
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
    prob_pos = torch.sigmoid(torch.from_numpy(logits_flat)).numpy()
    pred_prob = np.column_stack([1 - prob_pos, prob_pos])
    # Fallback threshold; outer layer recomputes using this model's internal
    # global OOF threshold
    pred_class = prob_to_class(prob_pos, GLOBAL_THRESHOLD)
    gate_vals = np.concatenate(all_gates).flatten() if all_gates else np.array([])
    metrics = compute_metrics(all_labels, pred_prob, pred_class)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['mean_gate'] = float(np.mean(gate_vals)) if len(gate_vals) > 0 else 0.0
    return metrics, pred_prob, pred_class, gate_vals


def train_model(model, train_loader, val_loader, device,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                weight_decay=WEIGHT_DECAY, fold_idx=None):
    """Train the NN; returns (model, best_epoch, best_auc).

    val_loader is used for early stopping (best-epoch selection); the
    classification threshold is optimized by the caller on the model's outer OOF.
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
    return model, best_epoch, best_auc


# Threshold optimization (invoked per model on its own internal OOF; yields a
# single global threshold for that model)
def search_optimal_threshold(y_true, y_prob_pos, rule=None):
    """Pick the classification threshold on internal OOF by the given rule.

    rule=None uses the current task's threshold_rule.
    'equal_error': minimize |Sens - Spec|; on ties, take the median of the
                   tied thresholds.
    'youden'     : maximize Sens + Spec - 1.
    'composite'  : maximize (Youden + Accuracy + (1-|Sens-Spec|))/3 (equal weight).
    """
    rule = rule or get_threshold_rule()
    thresholds = np.arange(0.01, 0.99, 0.001)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob_pos >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / len(y_true)
        rows.append((float(thr), float(sens), float(spec),
                     float(sens + spec - 1.0), float(abs(sens - spec)), float(acc)))
    if rule == 'equal_error':
        min_diff = min(r[4] for r in rows)
        ties = [r for r in rows if abs(r[4] - min_diff) < 1e-12]
        chosen = ties[len(ties) // 2]
        return {'threshold': chosen[0], 'sensitivity': chosen[1],
                'specificity': chosen[2], 'youden': chosen[3], 'accuracy': chosen[5],
                'rule': 'equal_error', 'abs_sens_spec_diff': chosen[4]}
    if rule == 'composite':
        scores = [r[3] + r[5] + (1.0 - r[4]) for r in rows]
        max_score = max(scores)
        ties = [r for r, s in zip(rows, scores) if abs(s - max_score) < 1e-12]
        max_j = max(r[3] for r in ties)
        ties = [r for r in ties if abs(r[3] - max_j) < 1e-12]
        max_acc = max(r[5] for r in ties)
        ties = [r for r in ties if abs(r[5] - max_acc) < 1e-12]
        chosen = ties[len(ties) // 2]
        return {'threshold': chosen[0], 'sensitivity': chosen[1],
                'specificity': chosen[2], 'youden': chosen[3], 'accuracy': chosen[5],
                'rule': 'composite', 'abs_sens_spec_diff': chosen[4],
                'composite_score': max_score / 3.0}
    best = max(rows, key=lambda r: r[3])
    return {'threshold': best[0], 'sensitivity': best[1],
            'specificity': best[2], 'youden': best[3], 'accuracy': best[5],
            'rule': 'youden', 'abs_sens_spec_diff': best[4]}


def metrics_at_threshold(y_true, y_prob_pos, threshold):
    """OOF classification metrics at a given threshold (used when the threshold
    is set by another model)."""
    y_true = np.asarray(y_true)
    y_pred = prob_to_class(y_prob_pos, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {'threshold': float(threshold), 'sensitivity': float(sens),
            'specificity': float(spec), 'youden': float(sens + spec - 1.0),
            'accuracy': float((tp + tn) / len(y_true)),
            'abs_sens_spec_diff': float(abs(sens - spec))}


def optimize_model_threshold(model_key, oof_y_true, oof_y_prob):
    """Optimize a threshold on the given OOF predictions by the current task's
    threshold_rule.

    The main model and every baseline model call this on their own internal OOF;
    external labels do not participate.
    """
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


# Inner C selection: LASSO LR uses repeated inner K-fold CV within the training
# fold, argmax of mean AUC
def select_lr_best_C(X_train, y_train, cv=5, C_grid=None, seed=SEED, n_repeats=None):
    """LR selection strategy: repeated K-fold inner CV within the training fold,
    pick the C with the largest mean AUC.

    - LR_INNER_CV_REPEATS repeated K-fold splits with different random seeds;
      each repeat's fold-mean AUC is one observation; compute mean and SE to
      reduce split-induced noise.
    - Selection rule: argmax of the repeated-CV mean AUC (no 1-SE tolerance).
    - C selection uses only data within the training fold; the outer validation
      fold is used solely for final evaluation.
    """
    if C_grid is None:
        C_grid = LR_C_GRID
    if n_repeats is None:
        n_repeats = LR_INNER_CV_REPEATS
    n_samples = X_train.shape[0]
    if n_samples < 2 * cv or len(np.unique(y_train)) < 2 or min_class_count(y_train) < cv:
        return 1.0

    # Pre-generate n_repeats stratified splits (within one split set, all C's are
    # fully comparable)
    split_sets = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed + r)
        split_sets.append(list(skf.split(X_train, y_train)))

    def _score(C):
        """Return (C, per-repeat fold-mean-AUC list). Each C's repeats are run
        serially."""
        repeat_aucs = []
        for splits in split_sets:
            fold_aucs = []
            for tr_idx, va_idx in splits:
                model = LogisticRegression(
                    penalty='l1', C=C, solver='saga',
                    max_iter=LASSO_CV_MAX_ITER, random_state=seed, class_weight=None
                )
                model.fit(X_train[tr_idx], y_train[tr_idx])
                fold_aucs.append(safe_auc(y_train[va_idx],
                                          model.predict_proba(X_train[va_idx])[:, 1]))
            repeat_aucs.append(float(np.mean(fold_aucs)))
        return C, repeat_aucs

    # Parallelize across C's (threads) without changing per-C fit order/results
    results = Parallel(n_jobs=min(len(C_grid), SKLEARN_N_JOBS), backend='threading')(
        delayed(_score)(C) for C in C_grid)

    stats = []
    for C, repeat_aucs in results:
        arr = np.asarray(repeat_aucs, dtype=float)
        stats.append((C, float(arr.mean()),
                      float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0))

    best_c, best_mean, best_se = max(stats, key=lambda t: t[1])
    print(f"    [Inner CV {n_repeats}×{cv}-fold] argmax mean AUC: C={best_c:g} "
          f"(AUC={best_mean:.4f}±{best_se:.4f})")
    return best_c


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
    'model_lightgbm':     ('LightGBM',                        _make_lightgbm,     LIGHTGBM_GRID),
    'model_xgboost':      ('XGBoost',                         _make_xgboost,      XGBOOST_GRID),
    'model_elasticnet':   ('ElasticNet Logistic Regression',  _make_elasticnet,   ELASTICNET_GRID),
    'model_randomforest': ('RandomForest',                    _make_randomforest, RANDOMFOREST_GRID),
}


def inner_cv_for_model(model_factory, param_grid, X_train_raw, y_train,
                       n_inner=STAT_INNER_CV_FOLDS, seed=SEED, scale=True):
    """Inner nested CV: select best hyperparameters + generate inner OOF
    probabilities (per-fold standardization, zero leakage).

    Uses the same "inner 5-fold CV within the training fold" strategy as
    select_lr_best_C, applied to additional models beyond LASSO.
    scale=False means inputs are already standardized per task convention;
    inner CV does not refit a scaler.
    Returns: best_param, inner_oof_prob
    """
    X_train_raw = np.asarray(X_train_raw, dtype=np.float64)
    y_train = np.asarray(y_train)
    n = len(y_train)
    rng = np.random.RandomState(seed)

    n_unique = len(np.unique(y_train))
    if n < 2 * n_inner or n_unique < 2 or min_class_count(y_train) < n_inner:
        if scale:
            sc = new_feature_scaler().fit(X_train_raw)
            X_fit, X_prob = sc.transform(X_train_raw), sc.transform(X_train_raw)
        else:
            X_fit = X_prob = X_train_raw
        m = model_factory(param_grid[0], seed)
        m.fit(X_fit, y_train)
        return param_grid[0], m.predict_proba(X_prob)[:, 1]

    best_param = param_grid[0]
    best_auc = -1.0
    best_oof = np.zeros(n, dtype=np.float64)

    for param in param_grid:
        try:
            skf = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                  random_state=int(rng.randint(0, 10**6)))
            oof = np.zeros(n, dtype=np.float64)
            for tr_local, va_local in skf.split(X_train_raw, y_train):
                if scale:
                    sc_local = new_feature_scaler().fit(X_train_raw[tr_local])
                    X_tr = sc_local.transform(X_train_raw[tr_local])
                    X_va = sc_local.transform(X_train_raw[va_local])
                else:
                    X_tr, X_va = X_train_raw[tr_local], X_train_raw[va_local]
                m = model_factory(param, int(rng.randint(0, 10**6)))
                m.fit(X_tr, y_train[tr_local])
                oof[va_local] = m.predict_proba(X_va)[:, 1]
            auc = safe_auc(y_train, oof)
        except Exception:
            auc = -1.0
            oof = np.zeros(n, dtype=np.float64)
        if auc > best_auc:
            best_auc = auc
            best_param = param
            best_oof = oof

    return best_param, best_oof


# External test (main model: task-level configurable 'lasso_lr' | 'elasticnet_lr';
# threshold = main-model internal-OOF optimized; external labels do not participate)
def _resolve_main_oof_threshold(model_key, factory, grid, output_dir=None):
    """Get the specified candidate model's internal-OOF threshold (per
    THRESHOLD_RULE): prefer the baselines cache; otherwise compute on the fly."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    cache_path = os.path.join(output_dir, 'baselines', 'baselines_cv.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            # Must validate cache version and analysis signature; otherwise a
            # stale-rule cache would be read.
            if (data.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION
                    and data.get('analysis_sig') == get_analysis_signature()):
                oof = data.get('oof_preds', {}).get(model_key)
                if oof is not None and 'threshold' in oof:
                    return float(oof['threshold'])
        except Exception:
            pass
    print(f"  No cached internal-OOF threshold for {model_key}; computing ...")
    ids, patches, y = load_patch_features_aligned(
        os.path.join(SL_DIR, 'SL-patch_features'), SL_LABEL_XLSX,
        CURRENT_LABEL_COL, include_filter=get_include_filter())
    X = compute_mean_pooled_features(patches).astype(np.float64)
    y = np.asarray(y)
    folds = load_or_create_folds(ids, y, output_dir=output_dir, n_folds=get_n_folds())
    idx = {c: i for i, c in enumerate(ids)}
    oof = np.zeros(len(y))
    # Exactly same convention as run_baselines_cv: per_fold standardization,
    # outer random seed fixed to SEED. per_cohort is not supported: fitting a
    # single scaler over the full cohort leaks validation-fold info into the
    # statistics and would create a second threshold path different from the
    # baseline cache.
    for fd in folds:
        tr = [idx[c] for c in fd['train_cases']]
        va = [idx[c] for c in fd['val_cases']]
        param, _ = inner_cv_for_model(factory, grid, X[tr], y[tr],
                                      STAT_INNER_CV_FOLDS, seed=SEED + fd['fold'])
        sc = new_feature_scaler().fit(X[tr])
        m = factory(param, SEED)
        m.fit(sc.transform(X[tr]), y[tr])
        oof[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
    return float(search_optimal_threshold(y, oof)['threshold'])


def process_external_test(threshold_config=None, output_dir=None, main_threshold=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    main_model = get_main_model()
    main_model_name = get_main_model_name()
    print(f"\n{'='*60}")
    print(f"  External test inference (main model: {main_model_name}) | "
          f"task: {CURRENT_TASK['task_name']}")
    print(f"{'='*60}")
    # Main-model classification threshold: from internal-OOF optimization
    if main_threshold is None:
        if main_model == 'elasticnet_lr':
            main_threshold = _resolve_main_oof_threshold(
                'model_elasticnet', _make_elasticnet, ELASTICNET_GRID, output_dir)
        else:
            main_threshold = get_model_threshold('main')
    print(f"  Main-model threshold (internal OOF, {get_threshold_rule()}) = "
          f"{main_threshold:.4f}")
    if main_model != 'lasso_lr':
        # Non-LASSO main model: keep Step 8 report fields and record main-model name
        _cfg = dict(threshold_config) if threshold_config else {}
        _cfg.update({'optimal_threshold': float(main_threshold),
                     'model': main_model_name, 'rule': get_threshold_rule()})
        with open(os.path.join(output_dir, 'models', 'threshold_config.pkl'), 'wb') as f:
            pickle.dump(_cfg, f)

    sl_patch_dir = os.path.join(SL_DIR, 'SL-patch_features')
    _, patch_list_sl, y_sl_bin = load_patch_features_aligned(
        sl_patch_dir, SL_LABEL_XLSX, CURRENT_LABEL_COL, include_filter=get_include_filter())

    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    stat_features_sl = compute_mean_pooled_features(patch_list_sl)
    final_stat_pre = BranchA_StatPreprocessor(seed=SEED)
    final_stat_pre.fit(stat_features_sl)
    final_stat_pre.save_final(models_dir, 'final')
    X_sl_stat = final_stat_pre.transform(stat_features_sl)

    if main_model == 'elasticnet_lr':
        # External main model = ElasticNet LR: inner 5-fold CV on full SL to
        # select hyperparameters -> full fit
        print("\n  Training ElasticNet LR main model on full SL data ...")
        # X_sl_stat is already standardized per task convention; inner CV uses
        # scale=False to avoid a second standardization that would mismatch the
        # final clf.fit feature scale.
        best_param, inner_oof = inner_cv_for_model(
            _make_elasticnet, ELASTICNET_GRID, X_sl_stat, y_sl_bin,
            STAT_INNER_CV_FOLDS, seed=SEED, scale=False)
        print(f"    Inner-CV best hyperparameters: {best_param} "
              f"(inner OOF AUC={safe_auc(y_sl_bin, inner_oof):.4f})")
        clf = _make_elasticnet(best_param, SEED)
        clf.fit(X_sl_stat, y_sl_bin)
        with open(os.path.join(models_dir, 'final_elasticnet_lr.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        with open(os.path.join(models_dir, 'final_elasticnet_params.pkl'), 'wb') as f:
            pickle.dump({'best_param': best_param, 'threshold': main_threshold}, f)
        with open(os.path.join(models_dir, 'final_stat_scaler_for_elasticnet.pkl'), 'wb') as f:
            pickle.dump(final_stat_pre.scaler, f)
        print(f"  Final ElasticNet trained; number of non-zero features: "
              f"{np.sum(clf.coef_ != 0)}")
    else:
        # Train LASSO(L1) LR (inner 5-fold CV on full SL to search optimal C)
        print("\n  Training LASSO(L1) LR ...")
        best_C = select_lr_best_C(X_sl_stat, y_sl_bin)
        print(f"    Inner-CV best C = {best_C:.4g}")

        clf = LogisticRegression(penalty='l1', C=best_C, solver='saga',
                                 max_iter=LASSO_CV_MAX_ITER, random_state=SEED, class_weight=None)
        clf.fit(X_sl_stat, y_sl_bin)
        with open(os.path.join(models_dir, 'final_linear_lr.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        with open(os.path.join(models_dir, 'final_stat_scaler_for_lr.pkl'), 'wb') as f:
            pickle.dump(final_stat_pre.scaler, f)
        print(f"  Final LR trained; number of non-zero features: {np.sum(clf.coef_ != 0)}")

    fy_patch_dir = os.path.join(FY_DIR, 'FY-patch_features')
    fy_case_ids, fy_patch_list, fy_y_bin = load_patch_features_aligned(
        fy_patch_dir, FY_LABEL_XLSX, CURRENT_LABEL_COL, include_filter=get_include_filter())

    fy_stat_features = compute_mean_pooled_features(fy_patch_list)
    ext_preproc = get_ext_preproc_mode()
    # FY uses the SL training-cohort mean/std scaler (same scaler as training);
    # external labels do not participate.
    fy_scaler = final_stat_pre.scaler
    fy_stat_reduced = final_stat_pre.transform(fy_stat_features)
    print(f"  External feature preprocessing: SL training-cohort mean/std scaler "
          f"(n={len(fy_case_ids)}; external labels not used)")
    with open(os.path.join(models_dir, f'final_stat_scaler_applied_to_fy_{ext_preproc}.pkl'), 'wb') as f:
        pickle.dump(fy_scaler, f)

    print("\n  External test inference ...")
    all_prob = clf.predict_proba(fy_stat_reduced)
    all_pred = prob_to_class(all_prob[:, 1], main_threshold)
    all_true = fy_y_bin

    metrics = compute_metrics(all_true, all_prob, all_pred)

    auc_mean, auc_lower, auc_upper = auc_bca_ci(all_true, all_prob[:, 1])

    ext_prob = all_prob[:, 1]
    ext_brier_ci = bca_ci_general(all_true, ext_prob,
        lambda yt, yp: compute_brier_score(yt, yp), N_BOOTSTRAP, 0.05, SEED)
    ext_cat_ci = classification_metric_cis(all_true, ext_prob, threshold=main_threshold,
                                           n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED)
    ext_f1_ci = bca_ci_general(all_true, ext_prob,
        lambda yt, yp: f1_score(yt, prob_to_class(yp, main_threshold)), N_BOOTSTRAP, 0.05, SEED)
    # Calibration slope/intercept (BCa method consistent with internal validation)
    ext_cal = calibration_metric_cis(all_true, ext_prob, N_BOOTSTRAP, 0.05, SEED)

    print(f"\n  External test results (main-model OOF-optimized threshold="
          f"{main_threshold:.4f}):")
    print(f"    AUC: {auc_mean:.4f} (95% CI: {auc_lower:.4f} - {auc_upper:.4f})")
    print(f"    Accuracy: {metrics['accuracy']:.4f} | "
          f"Sensitivity: {metrics['sensitivity']:.4f} | "
          f"Specificity: {metrics['specificity']:.4f}")
    print(f"    Cal. Slope: {ext_cal['cal_slope']:.4g} "
          f"(95% CI: {ext_cal['cal_slope_ci'][1]:.4g}-{ext_cal['cal_slope_ci'][2]:.4g})")
    print(f"    Cal. Intercept: {ext_cal['cal_intercept']:.4g} "
          f"(95% CI: {ext_cal['cal_intercept_ci'][1]:.4g}-{ext_cal['cal_intercept_ci'][2]:.4g})")

    pred_dir = os.path.join(output_dir, 'predictions')
    pd.DataFrame({
        'case_id': fy_case_ids,
        'true_label': [CLASSES[l] for l in all_true],
        'pred_label': [CLASSES[p] for p in all_pred],
        'prob_neg': all_prob[:, 0],
        'prob_pos': all_prob[:, 1],
    }).to_csv(os.path.join(pred_dir, 'main_model_external_predictions.csv'), index=False)

    full_state = {
        'ext_metrics': metrics,
        'ext_auc_ci': (auc_mean, auc_lower, auc_upper),
        'ext_brier_ci': ext_brier_ci,
        'ext_cat_ci': ext_cat_ci,
        'ext_f1_ci': ext_f1_ci,
        'ext_cal': ext_cal,
        # Store only the sklearn scaler (cross-module deserializable), not the
        # custom preprocessor object
        'final_stat_scaler': final_stat_pre.scaler,
        'stat_dim': final_stat_pre.output_dim,
        'fy_case_ids': fy_case_ids,
        'fy_stat_reduced': fy_stat_reduced,
        'fy_y_bin': fy_y_bin,
        'main_preds': {'prob': all_prob, 'class': all_pred, 'true': all_true},
        'threshold': main_threshold,
        'main_model': main_model,
        'main_model_name': main_model_name,
        'external_preproc': ext_preproc,
        'model_grids': {'lasso': list(LR_C_GRID),
                        'lightgbm': deepcopy(LIGHTGBM_GRID),
                        'xgboost': deepcopy(XGBOOST_GRID),
                        'elasticnet': deepcopy(ELASTICNET_GRID),
                        'randomforest': deepcopy(RANDOMFOREST_GRID)},
        'analysis_sig': get_analysis_signature(),
        'leak_fix_version': LR_LEAK_FIX_VERSION,
    }
    with open(os.path.join(pred_dir, 'external_test_state.pkl'), 'wb') as f:
        pickle.dump(full_state, f)
    return full_state


# Baseline models: K-fold cross-validation
#   - Patch-level features MLP only (CLAM attention pooling; inner 80/20 split
#     of the training fold used only for early stopping)
#   - Additional tabular models (mean-pooled features; inner nested CV for
#     hyperparameters)
#   - All models uniformly optimize a global threshold on their own internal OOF
def run_baselines_cv(folds, patch_list_all, labels_all, case_ids, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    print(f"\n{'='*60}")
    print(f"  Baseline {len(folds)}-fold cross-validation")
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
        y_train_bin = labels_all[train_indices]
        y_val_bin = labels_all[val_indices]

        val_patches = [patch_list_all[i] for i in val_indices]

        # Inner 80/20 split within the training fold
        inner_train_cases, inner_val_cases = split_inner_train_val(train_cases, y_train_bin)
        inner_train_indices = [case_to_idx[c] for c in inner_train_cases if c in case_to_idx]
        inner_val_indices = [case_to_idx[c] for c in inner_val_cases if c in case_to_idx]
        y_inner_train_bin = labels_all[inner_train_indices]
        y_inner_val_bin = labels_all[inner_val_indices]

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
        model, best_epoch, best_auc = train_model(
            model, inner_train_loader, inner_val_loader, DEVICE, fold_idx=fold_idx
        )

        criterion_eval = build_criterion(DEVICE)
        # Outer validation fold: accumulate OOF probabilities only; this model
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

    # Optimize a threshold on this model's internal OOF by THRESHOLD_RULE
    key = 'baseline_stat_only'
    display_name = 'Patch-level features MLP only'
    thr_res = optimize_model_threshold(key, oof_yt, oof_yp)
    model_thr = get_model_threshold(key)
    print(f"    [Threshold] {display_name}: internal-OOF optimized threshold="
          f"{model_thr:.4f} (Youden={thr_res['youden']:.4f}, "
          f"Sens={thr_res['sensitivity']:.4f}, Spec={thr_res['specificity']:.4f})")

    # Recompute per-fold classification metrics using this model's own threshold
    # (overrides fallback-threshold values computed during training)
    for i, r in enumerate(fold_results):
        yt_i, pr_i = fold_trues[i], fold_probs[i]
        pred_i = prob_to_class(pr_i, model_thr)
        tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
        r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        r['val_f1'] = f1_score(yt_i, pred_i)
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

    # --- Additional tabular models: mean-pooled features + inner nested CV ---
    print("\n  === Additional models (mean-pooled patch features + inner nested CV) ===")

    # Precompute per-fold mean-pooled features (identical to the LASSO main
    # model) for reuse by all additional models.
    #   internal_preproc='per_cohort' standardizes once with the full SL cohort
    _per_cohort_int = get_internal_preproc() == 'per_cohort'
    _gs_all = (new_feature_scaler().fit(compute_mean_pooled_features(patch_list_all))
               if _per_cohort_int else None)
    if _per_cohort_int:
        print("    Internal preprocessing: full-SL-cohort standardization (per_cohort)")
    fold_tab_data = []
    for fold_info in folds:
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        X_tr = compute_mean_pooled_features([patch_list_all[i] for i in train_indices])
        X_va = compute_mean_pooled_features([patch_list_all[i] for i in val_indices])
        if _per_cohort_int:
            X_tr = _gs_all.transform(X_tr)
            X_va = _gs_all.transform(X_va)
        fold_tab_data.append({
            'fold': fold_idx,
            'val_cases': val_cases,
            'y_train_bin': labels_all[train_indices],
            'y_val_bin': labels_all[val_indices],
            'X_train_mean': X_tr,
            'X_val_mean': X_va,
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

            # Inner 5-fold CV within the training fold for hyperparameter grid
            # search (same logic as LASSO)
            if _per_cohort_int:
                # per_cohort: features already standardized with full-SL stats;
                # inner CV does not refit a scaler
                X_tr_in, X_va_in = X_train_mean, X_val_mean
                best_param, _ = inner_cv_for_model(
                    factory, grid, X_tr_in, y_train_bin, STAT_INNER_CV_FOLDS,
                    seed=SEED + fold_idx, scale=False)
                model = factory(best_param, SEED)
                model.fit(X_tr_in, y_train_bin)
                y_val_prob = model.predict_proba(X_va_in)[:, 1]
            else:
                best_param, _ = inner_cv_for_model(
                    factory, grid, X_train_mean, y_train_bin, STAT_INNER_CV_FOLDS,
                    seed=SEED + fold_idx)
                # Fit scaler + model on the training fold
                scaler = new_feature_scaler().fit(X_train_mean)
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

        # Optimize a threshold on this model's internal OOF by THRESHOLD_RULE
        thr_res = optimize_model_threshold(key, oof_yt, oof_yp)
        model_thr = get_model_threshold(key)
        print(f"    [Threshold] {display_name}: internal-OOF optimized threshold="
              f"{model_thr:.4f} (Youden={thr_res['youden']:.4f}, "
              f"Sens={thr_res['sensitivity']:.4f}, Spec={thr_res['specificity']:.4f})")

        # Recompute per-fold classification metrics using this model's own
        # threshold
        for i, r in enumerate(fold_results):
            yt_i, pr_i = fold_trues[i], fold_probs[i]
            pred_i = prob_to_class(pr_i, model_thr)
            tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
            r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            r['val_f1'] = f1_score(yt_i, pred_i)
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
                     'analysis_sig': get_analysis_signature(),
                     'leak_fix_version': LR_LEAK_FIX_VERSION}, f)
    return all_results, oof_preds, oof_ci_results


# Comprehensive Excel summary
def _generate_comprehensive_summary(main_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=None,
                                     main_oof_metrics_ci=None,
                                     output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    import openpyxl

    def _fmt_ci(ci_tuple):
        if ci_tuple is None or np.isnan(ci_tuple[1]):
            return ''
        return f'{round_to_4sf(ci_tuple[1])}-{round_to_4sf(ci_tuple[2])}'

    def _build_oof_row(name, oof_dict, oof_metrics_ci=None):
        """Internal-OOF summary row (each model uses its own internal-OOF
        optimized global threshold)."""
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
            oof_metrics_ci = compute_oof_metrics_with_ci(
                oof_dict['y_true'], oof_dict['y_prob'], threshold=thr_for_f1)
        row['AUC_95CI'] = _fmt_ci(oof_metrics_ci.get('auc_ci'))
        row['Brier'] = round_to_4sf(oof_metrics_ci.get('brier', float('nan')))
        row['Brier_95CI'] = _fmt_ci(oof_metrics_ci.get('brier_ci'))
        row['Sensitivity_95CI'] = _fmt_ci(oof_metrics_ci.get('sensitivity_ci'))
        row['Specificity_95CI'] = _fmt_ci(oof_metrics_ci.get('specificity_ci'))
        row['Accuracy_95CI'] = _fmt_ci(oof_metrics_ci.get('accuracy_ci'))
        row['F1_95CI'] = _fmt_ci(oof_metrics_ci.get('f1_ci'))
        row['PPV_95CI'] = _fmt_ci(oof_metrics_ci.get('ppv_ci'))
        row['NPV_95CI'] = _fmt_ci(oof_metrics_ci.get('npv_ci'))
        row['Cal_Slope'] = round_to_4sf(oof_metrics_ci.get('cal_slope', float('nan')))
        row['Cal_Slope_95CI'] = _fmt_ci(oof_metrics_ci.get('cal_slope_ci'))
        row['Cal_Intercept'] = round_to_4sf(oof_metrics_ci.get('cal_intercept', float('nan')))
        row['Cal_Intercept_95CI'] = _fmt_ci(oof_metrics_ci.get('cal_intercept_ci'))
        return row

    _main_is_en = get_main_model() == 'elasticnet_lr'
    _lr_tag = 'candidate model' if _main_is_en else 'main model'

    def _disp(name, key):
        """Mark the ElasticNet row as the main model when it is the main model."""
        if _main_is_en and key == 'model_elasticnet':
            return f"{name.replace(' (baseline)', '')} (main model)"
        return name

    rows_internal = []
    rows_external = []

    # 1) LASSO candidate: per-fold mean (values from Steps 3-8 LASSO, not the
    #    ElasticNet main model)
    cv_csv_path = os.path.join(output_dir, 'predictions', f'{get_n_folds()}fold_cv_summary.csv')
    if os.path.exists(cv_csv_path):
        cv_table = pd.read_csv(cv_csv_path)
        numeric = cv_table[pd.to_numeric(cv_table['Fold'], errors='coerce').notna()]
        aucs = numeric['Val_AUC'].astype(float)
        sens = numeric['Sensitivity'].astype(float)
        specs = numeric['Specificity'].astype(float)
        f1s = numeric['F1'].astype(float)
        accs = numeric['Accuracy'].astype(float)
        rows_internal.append({
            'Model': f'LASSO(L1) Logistic Regression ({_lr_tag}, per-fold mean)',
            'Source': f'Internal {get_n_folds()}-fold CV',
            'AUC': round_to_4sf(aucs.mean()),
            'Sensitivity': round_to_4sf(sens.mean()),
            'Specificity': round_to_4sf(specs.mean()),
            'F1': round_to_4sf(f1s.mean()),
            'Accuracy': round_to_4sf(accs.mean()),
        })

    # 1b) LASSO OOF summary (threshold from the task main-model convention)
    if main_oof_metrics_ci is None:
        main_oof_metrics_ci = compute_oof_metrics_with_ci(
            main_oof['y_true'], main_oof['y_prob'])
    rows_internal.append(_build_oof_row(
        f'LASSO(L1) Logistic Regression ({_lr_tag}, OOF summary)',
        main_oof, main_oof_metrics_ci))

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
            'Model': _disp(name, key), 'Source': f'Internal {get_n_folds()}-fold CV',
            'AUC': round_to_4sf(np.nanmean(fold_aucs)),
            'Sensitivity': round_to_4sf(np.nanmean(fold_sens)) if not all(np.isnan(fold_sens)) else float('nan'),
            'Specificity': round_to_4sf(np.nanmean(fold_specs)) if not all(np.isnan(fold_specs)) else float('nan'),
            'F1': round_to_4sf(np.nanmean(fold_f1s)) if not all(np.isnan(fold_f1s)) else float('nan'),
        })

    # 3) Baseline OOF summary
    for key, name in baseline_order:
        if key in baseline_oof_preds:
            ci = baseline_oof_ci_results.get(key) if baseline_oof_ci_results else None
            rows_internal.append(_build_oof_row(_disp(name, key),
                                                baseline_oof_preds[key], ci))

    # External
    def _build_ext_row(name, ext_metrics, auc_ci, brier_ci=None, cat_ci=None,
                       f1_ci=None, cal=None):
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
    if ext_cal is None:
        # Backward compat: cache lacks calibration; recompute on the fly using
        # only info that does not leak external labels.
        _main_preds = ext_state.get('main_preds')
        if _main_preds is not None:
            ext_cal = calibration_metric_cis(_main_preds['true'], _main_preds['prob'][:, 1])
    rows_external.append(_build_ext_row(
        f"{ext_state.get('main_model_name', 'Image-only linear LR')} (main model)",
        ext_metrics, ext_auc_ci, ext_brier_ci, ext_cat_ci, ext_f1_ci, ext_cal))

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

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        if rows_internal:
            df_int = pd.DataFrame(rows_internal)
            df_int[[c for c in cols_int if c in df_int.columns]].to_excel(
                writer, sheet_name='Internal validation', index=False)
        if rows_external:
            df_ext = pd.DataFrame(rows_external)
            df_ext[[c for c in cols_ext if c in df_ext.columns]].to_excel(
                writer, sheet_name='External test', index=False)

    wb = openpyxl.load_workbook(xlsx_path)
    for ws in wb.worksheets:
        for col_letter in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                           'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF',
                           'AG', 'AH', 'AI', 'AJ']:
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


def plot_external_figures(ext_state, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')

    main_name = ext_state.get('main_model_name', 'Image-only linear LR')
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
    ax.set_title(f'ROC Curve on External Test Set ({main_name})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_roc_comparison.tiff'),
                dpi=600, bbox_inches='tight')
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
                dpi=600, bbox_inches='tight')
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
    ax.set_title(f'Decision Curve Analysis on External Test Set ({main_name})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.3, 1.0)
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='gray', fancybox=False)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_dca_curves.tiff'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  External DCA decision curve saved")


def plot_internal_oof_roc(main_oof, main_oof_metrics_ci, baseline_oof_preds,
                          baseline_oof_ci_results=None, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
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

    _main_is_en = get_main_model() == 'elasticnet_lr'
    oof_data = []
    main_thr = _model_threshold(main_oof)
    main_sens, main_spec = _operating_point(np.asarray(main_oof['y_true']),
                                            np.asarray(main_oof['y_prob']), main_thr)
    oof_data.append((
        'LASSO(L1) Logistic Regression (candidate model)' if _main_is_en
        else 'LASSO(L1) Logistic Regression (main model)',
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
        label = (f"{name.replace(' (baseline)', '')} (main model)"
                 if (_main_is_en and key == 'model_elasticnet') else name)
        oof_data.append((label, yt, yp, ci, thr, sens, spec))

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
                 '(per-model independent inner thresholds)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'internal_roc_comparison.tiff'),
                dpi=600, bbox_inches='tight')
    plt.close()
    print("  Internal OOF summary ROC figure saved (TIFF 600dpi)")


# Main pipeline (two tasks executed sequentially, each with independent cache
# and result directory)
def run_single_task(task):
    global GLOBAL_THRESHOLD
    set_active_task(task)
    # Each task independently resets random state and warning counts so task-2
    # does not inherit task-1's consumed global random state
    set_seed(SEED)
    CONVERGENCE_HITS.clear()
    t_start = datetime.now()
    print("\n" + "=" * 70)
    print(f"  Path features (MANEC cases) binary-classification multi-model "
          f"pipeline | task: {task['task_name']}")
    print(f"  Label files: train/internal-validation={os.path.basename(SL_LABEL_XLSX)} "
          f"external-test={os.path.basename(FY_LABEL_XLSX)} "
          f"(label_col='{CURRENT_LABEL_COL}')")
    print(f"  Result directory: {OUTPUT_DIR}")
    print(f"  Global random seed: {SEED} | Started at: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU parallel mode: {'fold-level parallel' if N_GPUS > 1 else 'single GPU/CPU'} "
          f"| Available GPUs: {N_GPUS}")
    print("=" * 70)

    # Step 1: load data (MANEC patch features only, aligned to label file, with
    # task-level population filter applied)
    step_start = datetime.now()
    _filt = get_include_filter()
    _filt_desc = f" | cohort filter: {_filt['col']}=={_filt['value']}" if _filt else ""
    print(f"\n[Step 1] Loading training patch features (MANEC only) and aligning "
          f"to the label file{_filt_desc} ...")
    sl_patch_dir = os.path.join(SL_DIR, 'SL-patch_features')
    case_ids, patch_list_all, labels_all = load_patch_features_aligned(
        sl_patch_dir, SL_LABEL_XLSX, CURRENT_LABEL_COL, include_filter=_filt)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 2: stratified fold split (fold count per task config)
    n_folds = get_n_folds()
    step_start = datetime.now()
    print(f"\n[Step 2] Generating stratified {n_folds}-fold split ...")
    folds = load_or_create_folds(case_ids, labels_all, n_folds=n_folds)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Remove stale per-fold CSVs beyond the current fold count (avoid residuals
    # when two tasks share a directory)
    _pred_dir = os.path.join(OUTPUT_DIR, 'predictions')
    for _stale in glob.glob(os.path.join(_pred_dir, 'val_prediction_fold_*.csv')):
        _m = re.search(r'fold_(\d+)\.csv$', os.path.basename(_stale))
        if _m and int(_m.group(1)) >= n_folds:
            os.remove(_stale)
            print(f"  Removed stale per-fold file: {os.path.basename(_stale)}")

    # Steps 3-8: main model = mean-pooled patch features + linear LASSO LR
    # (standardize on the full training fold + inner 5-fold CV for C +
    # internal-OOF global threshold)
    step_start = datetime.now()
    print(f"\n[Step 3-8] Main model (mean-pooled patch features + linear LASSO LR) "
          f"{n_folds}-fold CV ...")
    _per_cohort_main = get_internal_preproc() == 'per_cohort'
    _global_int_scaler = (new_feature_scaler().fit(compute_mean_pooled_features(patch_list_all))
                          if _per_cohort_main else None)
    if _per_cohort_main:
        print("  Internal preprocessing: full-SL-cohort mean/std fit "
              "(per_cohort; diagnostic setting)")
    main_fold_results = []
    main_oof_y_true = []
    main_oof_y_prob = []
    main_oof_case_ids = []
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    for fold_info in tqdm(
            folds,
            desc="  Main model (mean-pooled patch features + linear LASSO LR)",
            unit="fold"):
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        y_train_bin = labels_all[train_indices]
        y_val_bin = labels_all[val_indices]

        train_patches = [patch_list_all[i] for i in train_indices]
        val_patches = [patch_list_all[i] for i in val_indices]
        X_train_mean = compute_mean_pooled_features(train_patches)
        X_val_mean = compute_mean_pooled_features(val_patches)

        # Standardize on the whole training fold -> inner 5-fold CV for C
        #   per_cohort: use the full-SL-cohort scaler (diagnostic setting)
        if _per_cohort_main:
            X_train_stat = _global_int_scaler.transform(X_train_mean)
            X_val_stat = _global_int_scaler.transform(X_val_mean)
        else:
            scaler = new_feature_scaler()
            X_train_stat = scaler.fit_transform(X_train_mean)
            X_val_stat = scaler.transform(X_val_mean)
        best_C = select_lr_best_C(X_train_stat, y_train_bin)

        lr_main = LogisticRegression(penalty='l1', C=best_C, solver='saga',
                                     max_iter=LASSO_CV_MAX_ITER, random_state=SEED, class_weight=None)
        lr_main.fit(X_train_stat, y_train_bin)
        y_val_prob = lr_main.predict_proba(X_val_stat)
        # GLOBAL_THRESHOLD is still the fallback 0.5 here; pred_label will be
        # rewritten after main-model OOF threshold optimization
        y_val_class = prob_to_class(y_val_prob[:, 1], GLOBAL_THRESHOLD)
        val_auc = safe_auc(y_val_bin, y_val_prob[:, 1])

        pred_dir = os.path.join(OUTPUT_DIR, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({
            'case_id': val_cases,
            'true_label': [CLASSES[l] for l in y_val_bin],
            'pred_label': [CLASSES[p] for p in y_val_class],
            'prob_neg': 1.0 - y_val_prob[:, 1],
            'prob_pos': y_val_prob[:, 1],
        }).to_csv(os.path.join(pred_dir, f'val_prediction_fold_{fold_idx}.csv'),
                  index=False)

        main_fold_results.append({'fold': fold_idx, 'val_auc': val_auc, 'best_C': best_C})
        main_oof_y_true.extend(y_val_bin)
        main_oof_y_prob.extend(y_val_prob[:, 1])
        main_oof_case_ids.extend(val_cases)

    main_oof = {
        'case_ids': main_oof_case_ids,
        'y_true': np.array(main_oof_y_true),
        'y_prob': np.array(main_oof_y_prob),
        # This OOF is always from the Step 3-8 LASSO(L1) model, not the
        # ElasticNet main model
        'display_name': ('LASSO(L1) Logistic Regression (main model)'
                         if get_main_model() == 'lasso_lr'
                         else 'LASSO(L1) Logistic Regression (candidate model)')
    }

    print(f"\n  Main-model {n_folds}-fold CV summary:")
    auc_mean, ci_l, ci_u = auc_bca_ci(main_oof['y_true'], main_oof['y_prob'])
    print(f"    OOF AUC: {auc_mean:.4f} (95% BCa CI: {ci_l:.4f} - {ci_u:.4f})")
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 9: baseline CV (must run before threshold determination and external
    # test)
    #   Main-model OOF threshold is read uniformly from baselines_cv.pkl so that
    #   "internal report / external test / baseline table" share one source.
    #   If this step ran after the external test, the first run would go through
    #   the "compute on-the-fly" branch and disagree with the baseline-table
    #   threshold.
    step_start = datetime.now()
    print("\n[Step 9] Training/loading baseline models (MLP + additional models) ...")

    baseline_cv_path = os.path.join(OUTPUT_DIR, 'baselines', 'baselines_cv.pkl')
    baseline_cv_results = baseline_oof_preds = baseline_oof_ci_results = None
    if os.path.exists(baseline_cv_path):
        try:
            with open(baseline_cv_path, 'rb') as f:
                data = pickle.load(f)
            if data.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION \
                    and data.get('analysis_sig') == get_analysis_signature():
                baseline_cv_results = data['results']
                baseline_oof_preds = data['oof_preds']
                baseline_oof_ci_results = data.get('oof_ci_results', {})
                # Recompute OOF classification-metric CIs at each model's own
                # internal-OOF threshold
                for key, oof in baseline_oof_preds.items():
                    thr = oof.get('threshold', GLOBAL_THRESHOLD)
                    baseline_oof_ci_results[key] = compute_oof_metrics_with_ci(
                        oof['y_true'], oof['y_prob'], threshold=thr)
                print("    Loaded baseline cache; OOF classification-metric CIs "
                      "recomputed at each model's internal-OOF global threshold")
            else:
                print("    Baseline cache version/signature mismatch; recomputing ...")
        except Exception as _e:
            print(f"    Failed to read baseline cache ({_e}); recomputing ...")
            baseline_cv_results = None
    if baseline_cv_results is None:
        baseline_cv_results, baseline_oof_preds, baseline_oof_ci_results = run_baselines_cv(
            folds, patch_list_all, labels_all, case_ids
        )
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Threshold optimization: threshold convention follows the task's main model
    #   - main_model=LASSO: use LASSO's own internal OOF with the task rule
    #   - main_model=other (task-2 = ElasticNet): use that main model's own
    #     internal-OOF threshold so internal OOF report and external test share
    #     the same threshold convention
    threshold_config = None
    if ENABLE_THRESHOLD_OPTIMIZATION:
        if get_main_model() == 'lasso_lr':
            print(f"\n[Post-processing] Main-model internal-OOF threshold "
                  f"optimization ({get_threshold_rule()}) ...")
            thr_result = optimize_model_threshold('main', main_oof['y_true'], main_oof['y_prob'])
        else:
            print(f"\n[Post-processing] Internal OOF report threshold switched to "
                  f"main model ({get_main_model_name()})'s OOF threshold ...")
            _resolved = float(_resolve_main_oof_threshold(
                'model_elasticnet', _make_elasticnet, ELASTICNET_GRID))
            thr_result = metrics_at_threshold(main_oof['y_true'], main_oof['y_prob'], _resolved)
            thr_result['rule'] = get_threshold_rule()
        GLOBAL_THRESHOLD = float(thr_result['threshold'])
        main_oof['threshold'] = GLOBAL_THRESHOLD
        threshold_config = {
            'optimal_threshold': GLOBAL_THRESHOLD,
            'youden_index': thr_result['youden'],
            'sensitivity': thr_result['sensitivity'],
            'specificity': thr_result['specificity'],
            'rule': thr_result.get('rule', get_threshold_rule()),
            'accuracy': thr_result.get('accuracy', float('nan')),
            'composite_score': thr_result.get('composite_score', float('nan')),
            'abs_sens_spec_diff': thr_result.get('abs_sens_spec_diff', float('nan')),
            'threshold_source': ('main_model_oof' if get_main_model() != 'lasso_lr'
                                 else 'lasso_oof'),
        }
        print(f"    Threshold={GLOBAL_THRESHOLD:.4f} "
              f"(source={threshold_config['threshold_source']}, "
              f"rule={get_threshold_rule()}), "
              f"Sensitivity={thr_result['sensitivity']:.4f}, "
              f"Specificity={thr_result['specificity']:.4f}, "
              f"Accuracy={thr_result.get('accuracy', float('nan')):.4f}, "
              f"|Sens-Spec|={thr_result.get('abs_sens_spec_diff', float('nan')):.4f}, "
              f"Youden={thr_result['youden']:.4f}")
        with open(os.path.join(OUTPUT_DIR, 'models', 'threshold_config.pkl'), 'wb') as f:
            pickle.dump(threshold_config, f)
    else:
        main_oof['threshold'] = GLOBAL_THRESHOLD
        print(f"  [Post-processing] Threshold optimization disabled; main model uses "
              f"global default threshold {GLOBAL_THRESHOLD:.4f}")

    # Compute OOF classification-metric CIs at the main model's own OOF
    # threshold (only once, after the threshold is finalized)
    main_oof_ci = compute_oof_metrics_with_ci(main_oof['y_true'], main_oof['y_prob'],
                                              threshold=main_oof['threshold'])
    print(f"    OOF Brier: {main_oof_ci['brier']:.4g} "
          f"(95% BCa CI: {main_oof_ci['brier_ci'][1]:.4g}-{main_oof_ci['brier_ci'][2]:.4g})")
    print(f"    Cal. Slope: {main_oof_ci['cal_slope']:.4g} "
          f"(95% BCa CI: {main_oof_ci['cal_slope_ci'][1]:.4g}-{main_oof_ci['cal_slope_ci'][2]:.4g})")
    print(f"    Cal. Intercept: {main_oof_ci['cal_intercept']:.4g} "
          f"(95% BCa CI: {main_oof_ci['cal_intercept_ci'][1]:.4g}-{main_oof_ci['cal_intercept_ci'][2]:.4g})")

    # Export per-fold summary table (main model = mean-pooled patch features +
    # linear LASSO LR; classification metrics recomputed at the global threshold
    # from per-fold OOF probabilities). Also rewrite per-fold CSV pred_label to
    # the finalized OOF-optimized threshold (fold loop wrote fallback 0.5).
    fold_idx_to_res = {r['fold']: r for r in main_fold_results}
    # Task-2's main model is ElasticNet; this table records the LASSO candidate's
    # C value, so the column name must reflect that accurately
    _c_col = 'Best_C' if get_main_model() == 'lasso_lr' else 'LASSO_C (candidate)'
    cv_rows = []
    for fold_idx in range(n_folds):
        fold_csv_path = os.path.join(OUTPUT_DIR, 'predictions',
                                     f'val_prediction_fold_{fold_idx}.csv')
        df = pd.read_csv(fold_csv_path)
        yt = (df['true_label'] == CLASSES[1]).astype(int).values
        prob = df['prob_pos'].values
        pred = prob_to_class(prob, GLOBAL_THRESHOLD)
        df['pred_label'] = [CLASSES[p] for p in pred]
        df.to_csv(fold_csv_path, index=False)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        fr = fold_idx_to_res.get(fold_idx, {})
        cv_rows.append({
            'Fold': fold_idx,
            _c_col: fr.get('best_C', ''),
            'Val_AUC': safe_auc(yt, prob),
            'Accuracy': accuracy_score(yt, pred),
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'F1': f1_score(yt, pred),
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
    cv_summary_name = f'{n_folds}fold_cv_summary.csv'
    cv_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions', cv_summary_name),
                         index=False)
    print(f"    {n_folds}-fold summary saved: predictions/{cv_summary_name} "
          f"(global threshold={GLOBAL_THRESHOLD:.4f})")

    # Step 10: external test (threshold from main-model internal OOF; external
    # labels do not participate)
    step_start = datetime.now()
    print("\n[Step 10] External test inference (main model) ...")
    ext_state_path = os.path.join(OUTPUT_DIR, 'predictions', 'external_test_state.pkl')
    ext_state = None
    if os.path.exists(ext_state_path):
        try:
            with open(ext_state_path, 'rb') as f:
                ext_state = pickle.load(f)
            _ok = (
                ext_state.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION
                and 'main_preds' in ext_state
                and ext_state.get('analysis_sig') == get_analysis_signature()
                and abs(float(ext_state.get('threshold', -1))
                        - float(GLOBAL_THRESHOLD)) < 1e-9
            )
            if _ok:
                print("  Found cached external test results (signature/threshold "
                      "consistent); loading ...")
            else:
                print("  Cached external test results do not match current "
                      "signature/threshold; recomputing ...")
                ext_state = None
        except Exception as _e:
            print(f"  Failed to read external test cache ({_e}); recomputing ...")
            ext_state = None
    if ext_state is None:
        ext_state = process_external_test(
            threshold_config=threshold_config,
            main_threshold=GLOBAL_THRESHOLD)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 11: visualization
    step_start = datetime.now()
    print("\n[Step 11] Generating figures ...")
    plot_external_figures(ext_state)
    plot_internal_oof_roc(main_oof, main_oof_ci, baseline_oof_preds,
                          baseline_oof_ci_results=baseline_oof_ci_results)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 12: comprehensive Excel summary
    print("\n[Step 12] Generating comprehensive model comparison Excel ...")
    _generate_comprehensive_summary(
        main_oof, ext_state, baseline_cv_results, baseline_oof_preds,
        baseline_oof_ci_results=baseline_oof_ci_results,
        main_oof_metrics_ci=main_oof_ci)
    print("  Excel saved: predictions/all_models_comprehensive_summary.xlsx")

    # Convergence diagnostics (printed once per task)
    if CONVERGENCE_HITS:
        print(f"\n  [Convergence] {len(CONVERGENCE_HITS)} ConvergenceWarning(s) "
              f"captured (showing up to 10):")
        for _msg in CONVERGENCE_HITS[:10]:
            print(f"    {_msg}")

    t_end = datetime.now()
    print(f"\n{'='*70}")
    print(f"  Task completed: {task['task_name']} | Total elapsed: {t_end - t_start}")
    print(f"  Results directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


def main():
    for _task in TASKS:
        run_single_task(_task)


if __name__ == '__main__':
    main()
