"""
ewddbs_core.py
==============
Shared implementation of EW-DDBS, metrics, baselines and artefact logging.

Fixes applied relative to the original notebooks (see audit):
  D6  neighbour search is performed in the LATENT space on BOTH cohorts
  D7  the ablation grid genuinely isolates the entropy term
      (use_entropy and use_safe are independent switches, 2x2x2 = 8 variants)
  D8  jitter covariance follows Eq. (6): diagonal, eta * Var(minority-only)
  D9  a single G-Mean definition is used everywhere
  T1  y_proba is persisted so calibration (Brier / reliability) can be computed
      without re-running the experiment
  T2  H(x), S(z) and W_i are persisted so the weighting scheme can be shown to
      be non-degenerate
  T3  Baseline (no resampling), ClassWeight and PriorCorrection baselines added
  S2  every run is seeded and the seed is recorded with each result row
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from imblearn.under_sampling import TomekLinks
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# TensorFlow is imported lazily so that the post-hoc analysis script can use
# this module without pulling in the deep-learning stack.
tf = None


def _load_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    return tf

# ----------------------------------------------------------------------
# hyper-parameters (identical on both cohorts -- see Table 2 correction)
# ----------------------------------------------------------------------
T_TEMP = 0.5      # entropy temperature
TAU = 0.2         # safe-region threshold
K_NN = 5          # neighbourhood size for the Safe-Ratio
ETA_JITTER = 0.05 # jitter scale (eta in Eq. 6)


def set_seed(seed):
    np.random.seed(seed)
    _load_tf().random.set_seed(seed)


# ======================================================================
# 1. CNN topology guide
# ======================================================================
def cnn_topology_guide(X_train, y_train, seed=42, epochs=100, patience=5):
    """Train the 1D-CNN guide and return (entropy, latent_features).

    NOTE: entropy and latent codes are computed IN-SAMPLE, exactly as in the
    manuscript. The diagnostics returned by ew_ddbs_resample() are what reveal
    whether this makes the weighting degenerate.
    """
    _load_tf()
    from tensorflow.keras import callbacks, layers, models

    tf.keras.backend.clear_session()
    set_seed(seed)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    inputs = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    latent = layers.GlobalAveragePooling1D()(x)          # 32-D latent vector
    outputs = layers.Dense(n_classes, activation='softmax')(latent)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy')

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), cw)}

    X_cnn = X_train.reshape(-1, n_features, 1)
    es = callbacks.EarlyStopping(monitor='loss', patience=patience,
                                 restore_best_weights=True)
    model.fit(X_cnn, y_train, epochs=epochs, batch_size=32,
              class_weight=cw_dict, verbose=0, callbacks=[es])

    probs = model.predict(X_cnn, verbose=0)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    extractor = models.Model(inputs, latent)
    deep_feats = extractor.predict(X_cnn, verbose=0)
    return entropy.astype(np.float32), deep_feats.astype(np.float32)


# ======================================================================
# 2. EW-DDBS
# ======================================================================
def ew_ddbs_resample(X, y, entropy, latent,
                     use_entropy=True, use_safe=True, apply_tomek=True,
                     T=T_TEMP, tau=TAU, k=K_NN, eta=ETA_JITTER, rng=None):
    """Latent-guided safe-region oversampling.

    W_i = exp(H(x_i)/T) * S(z_i)      (Eq. 4)   -- both terms optional
    x_new = x_p + lam*(x_n - x_p) + d (Eq. 5)
    d ~ N(0, eta * diag(Var_minority)) (Eq. 6)

    Parents are drawn with P(i) ~ W_i; the neighbour x_n is the nearest
    same-class neighbour IN THE LATENT SPACE; synthesis happens in input space.

    Returns (X_res, y_res, diagnostics).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X_res, y_res = X.copy(), y.copy()
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    diag = {}

    for cls in classes:
        mask = y == cls
        n_cls = int(mask.sum())
        if n_cls >= max_count:
            continue
        num_add = int(max_count - n_cls)
        idx_cls = np.where(mask)[0]

        if n_cls < 2:
            chosen = rng.choice(idx_cls, size=num_add, replace=True)
            X_res = np.vstack([X_res, X[chosen]])
            y_res = np.hstack([y_res, np.full(num_add, cls)])
            continue

        # ---- Safe-Ratio in the latent manifold (Eq. 3) -------------------
        k_eff = int(min(k, len(latent) - 1))
        nn_lat_all = NearestNeighbors(n_neighbors=k_eff + 1).fit(latent)
        _, ind_all = nn_lat_all.kneighbors(latent[idx_cls])
        safe_ratio = np.mean(y[ind_all[:, 1:]] == cls, axis=1)

        # ---- weights (Eq. 4) --------------------------------------------
        w_ent = np.exp(entropy[idx_cls] / T) if use_entropy \
            else np.ones(n_cls, dtype=float)
        if use_safe:
            w_safe = safe_ratio.copy()
            w_safe[safe_ratio < tau] = 0.0
        else:
            w_safe = np.ones(n_cls, dtype=float)
        weights = w_ent * w_safe

        n_gated = int(np.sum(safe_ratio < tau)) if use_safe else 0
        if weights.sum() <= 0:
            weights = np.ones(n_cls, dtype=float)
        prob = weights / weights.sum()

        # ---- diagnostics (item T2) --------------------------------------
        diag[int(cls)] = {
            'n_minority': n_cls,
            'H_mean': float(np.mean(entropy[idx_cls])),
            'H_median': float(np.median(entropy[idx_cls])),
            'H_std': float(np.std(entropy[idx_cls])),
            'H_frac_below_0.01': float(np.mean(entropy[idx_cls] < 0.01)),
            'S_mean': float(np.mean(safe_ratio)),
            'S_median': float(np.median(safe_ratio)),
            'S_std': float(np.std(safe_ratio)),
            'S_frac_eq_1': float(np.mean(safe_ratio >= 0.999)),
            'S_frac_below_tau': float(np.mean(safe_ratio < tau)),
            'n_gated_by_tau': n_gated,
            'W_mean': float(np.mean(weights)),
            'W_std': float(np.std(weights)),
            'W_cv': float(np.std(weights) / (np.mean(weights) + 1e-12)),
            'W_max_over_min': float(weights.max() / (weights[weights > 0].min()
                                                     + 1e-12))
            if np.any(weights > 0) else 0.0,
            'P_effective_sample_size': float(1.0 / np.sum(prob ** 2)),
            'P_ess_ratio': float(1.0 / np.sum(prob ** 2) / n_cls),
        }

        # ---- latent-space neighbour, input-space synthesis (Eq. 5) ------
        k_loc = int(min(k, n_cls - 1))
        nn_lat_cls = NearestNeighbors(n_neighbors=k_loc + 1).fit(latent[idx_cls])
        parent_local = rng.choice(n_cls, size=num_add, p=prob)
        _, ind_loc = nn_lat_cls.kneighbors(latent[idx_cls][parent_local])
        # pick a random neighbour among the k nearest (excluding self)
        pick = rng.integers(1, ind_loc.shape[1], size=num_add)
        neighbour_local = ind_loc[np.arange(num_add), pick]

        # jitter: diagonal covariance from the MINORITY class only (Eq. 6)
        sigma_diag = np.var(X[idx_cls], axis=0) * eta
        sd = np.sqrt(np.maximum(sigma_diag, 0.0))

        xp = X[idx_cls][parent_local]
        xn = X[idx_cls][neighbour_local]
        lam = rng.uniform(0.1, 0.9, size=(num_add, 1))
        delta = rng.normal(0.0, 1.0, size=(num_add, X.shape[1])) * sd
        new_samples = xp + lam * (xn - xp) + delta

        X_res = np.vstack([X_res, new_samples])
        y_res = np.hstack([y_res, np.full(num_add, cls)])

    if apply_tomek:
        n_before = len(y_res)
        try:
            tl = TomekLinks(sampling_strategy='all')
            X_res, y_res = tl.fit_resample(X_res, y_res)
        except Exception:
            pass
        diag['tomek_removed'] = int(n_before - len(y_res))

    return X_res, y_res, diag


# ======================================================================
# 3. metrics -- ONE definition of each, used by both cohorts
# ======================================================================
def g_mean(y_true, y_pred):
    """Geometric mean of per-class sqrt(sensitivity * specificity)."""
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.diag(cm).astype(float)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - tp - fn - fp
        sens = np.where((tp + fn) > 0, tp / (tp + fn + 1e-12), np.nan)
        spec = np.where((tn + fp) > 0, tn / (tn + fp + 1e-12), np.nan)
        g = np.sqrt(sens * spec)
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(g + 1e-12))))   # geometric mean


def multiclass_brier(y_true, y_proba, classes):
    """Mean squared error between one-hot labels and predicted probabilities."""
    if y_proba is None:
        return np.nan
    Y = np.zeros_like(y_proba)
    for j, c in enumerate(classes):
        Y[:, j] = (y_true == c).astype(float)
    return float(np.mean(np.sum((y_proba - Y) ** 2, axis=1)))


def evaluate(y_true, y_pred, y_proba, classes):
    m = {
        'F1-Macro': f1_score(y_true, y_pred, average='macro'),
        'BalancedAccuracy': balanced_accuracy_score(y_true, y_pred),
        'G-Mean': g_mean(y_true, y_pred),
        # Degenerate-prediction guard. A model that emits a single class still
        # produces a finite F1, so it is silently averaged into the aggregates
        # unless it is flagged. PriorCorrection + AdaBoost does exactly this on
        # the UCI cohort (F1 = 0.05, G-Mean = 0.0).
        'n_pred_classes': int(len(np.unique(y_pred))),
        'degenerate': bool(len(np.unique(y_pred)) < len(classes)),
    }
    if y_proba is not None and y_proba.shape[1] == len(classes):
        try:
            if len(classes) == 2:
                m['AUC'] = roc_auc_score(y_true, y_proba[:, 1])
                m['AUPRC'] = average_precision_score(y_true, y_proba[:, 1])
                m['Brier'] = brier_score_loss(y_true, y_proba[:, 1])
            else:
                yb = label_binarize(y_true, classes=classes)
                m['AUC'] = roc_auc_score(yb, y_proba, multi_class='ovr',
                                         average='macro')
                m['AUPRC'] = average_precision_score(yb, y_proba,
                                                     average='macro')
                m['Brier'] = multiclass_brier(y_true, y_proba, classes)
        except Exception:
            m['AUC'] = m['AUPRC'] = m['Brier'] = np.nan
    else:
        m['AUC'] = m['AUPRC'] = m['Brier'] = np.nan
    return m


def predict_proba_safe(clf, X, n_classes):
    """Probabilities where available. Returns (proba, is_calibratable).

    LinearSVC/SGD-hinge have no predict_proba; a sigmoid/softmax of the
    decision function is returned for ranking metrics but flagged as NOT
    calibratable, so Brier scores from those models can be excluded.
    """
    if hasattr(clf, 'predict_proba'):
        try:
            p = clf.predict_proba(X)
            if p.shape[1] == n_classes:
                return p, True
        except Exception:
            pass
    if hasattr(clf, 'decision_function'):
        try:
            d = clf.decision_function(X)
            if d.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-d))
                return np.column_stack([1 - p1, p1]), False
            e = np.exp(d - d.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True), False
        except Exception:
            return None, False
    return None, False


# ======================================================================
# 4. classifiers
# ======================================================================
CLASSIFIERS = ["GBM", "RF", "Bagging", "XGBoost", "ExtraTrees", "AdaBoost",
               "DT", "SVM-RBF", "KNN", "SGD", "LinearSVM", "LogReg",
               "LDA", "QDA", "GNB"]


def build_classifier(name, seed, class_weight=None):
    cw = class_weight  # None or 'balanced'
    grids = {
        "GBM": (GradientBoostingClassifier(random_state=seed),
                {'n_estimators': [100, 200]}),
        "RF": (RandomForestClassifier(random_state=seed, n_jobs=-1,
                                      class_weight=cw),
               {'n_estimators': [100, 200]}),
        "Bagging": (BaggingClassifier(random_state=seed, n_jobs=-1),
                    {'n_estimators': [10, 30]}),
        "XGBoost": (XGBClassifier(eval_metric='mlogloss', random_state=seed,
                                  n_jobs=-1, verbosity=0),
                    {'n_estimators': [100, 200]}),
        "ExtraTrees": (ExtraTreesClassifier(random_state=seed, n_jobs=-1,
                                            class_weight=cw),
                       {'n_estimators': [100, 200]}),
        "AdaBoost": (AdaBoostClassifier(random_state=seed),
                     {'n_estimators': [50, 100]}),
        "DT": (DecisionTreeClassifier(random_state=seed, class_weight=cw),
               {'max_depth': [None, 10]}),
        "SVM-RBF": (SVC(kernel='rbf', probability=True, random_state=seed,
                        class_weight=cw), {'C': [1, 10]}),
        "KNN": (KNeighborsClassifier(n_jobs=-1), {'n_neighbors': [3, 5]}),
        "SGD": (SGDClassifier(random_state=seed, n_jobs=-1, loss='log_loss',
                              class_weight=cw), {'alpha': [1e-4, 1e-3]}),
        "LinearSVM": (LinearSVC(random_state=seed, max_iter=3000,
                                class_weight=cw), {'C': [1, 10]}),
        "LogReg": (LogisticRegression(random_state=seed, n_jobs=-1,
                                      max_iter=3000, class_weight=cw),
                   {'C': [1, 10]}),
        "LDA": (LinearDiscriminantAnalysis(), {}),
        "QDA": (QuadraticDiscriminantAnalysis(), {}),
        "GNB": (GaussianNB(), {}),
    }
    return grids[name]


def fit_classifier(name, X, y, seed, class_weight=None, cv=3):
    clf, grid = build_classifier(name, seed, class_weight)
    if grid:
        gs = GridSearchCV(clf, grid, cv=cv, n_jobs=-1, scoring='f1_macro')
        gs.fit(X, y)
        return gs.best_estimator_
    return clone(clf).fit(X, y)


# ======================================================================
# 5. resampling / baseline strategies
# ======================================================================
def ablation_grid(include_full_grid=True):
    """(use_entropy, use_safe, apply_tomek) keyed by display name.

    The 2x2x2 design makes the entropy axis identifiable, which the original
    4-variant grid did not (use_entropy was True in every variant).
    """
    base = {
        "EWDDBS (Uniform)":            (False, False, False),
        "EWDDBS (Uniform+Tomek)":      (False, False, True),
        "EWDDBS (Entropy)":            (True,  False, False),
        "EWDDBS (Entropy+Tomek)":      (True,  False, True),
        "EWDDBS (Safe)":               (False, True,  False),
        "EWDDBS (Safe+Tomek)":         (False, True,  True),
        "EWDDBS (Entropy+Safe)":       (True,  True,  False),
        "EWDDBS (Entropy+Safe+Tomek)": (True,  True,  True),
    }
    if include_full_grid:
        return base
    return {k: v for k, v in base.items()
            if k in ("EWDDBS (Entropy)", "EWDDBS (Entropy+Tomek)",
                     "EWDDBS (Safe)", "EWDDBS (Entropy+Safe+Tomek)")}


def prior_correction(y_proba, train_prior):
    """Threshold-moving baseline (Buda et al., 2018): divide by the training
    prior and renormalise, then take argmax."""
    p = y_proba / (train_prior[None, :] + 1e-12)
    return p / p.sum(axis=1, keepdims=True)


# ======================================================================
# 6. artefact logging
# ======================================================================
class ArtefactWriter:
    """Persists everything the post-hoc analysis needs.

    <outdir>/
        results.csv                     one row per (seed, fold, method, clf)
        diagnostics.csv                 H / S / W statistics per (seed, fold, variant)
        proba/seed{S}_fold{F}.npz       y_true + y_proba for every method x clf
        guide/seed{S}_fold{F}.npz       entropy, latent, y_train
    """

    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(os.path.join(outdir, 'proba'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'guide'), exist_ok=True)
        self.rows = []
        self.diags = []
        self._proba = {}

    def save_guide(self, seed, fold, entropy, latent, y_train):
        np.savez_compressed(
            os.path.join(self.outdir, 'guide', f'seed{seed}_fold{fold}.npz'),
            entropy=entropy, latent=latent, y_train=y_train)

    def add_diag(self, seed, fold, method, diag):
        for cls, d in diag.items():
            if cls == 'tomek_removed':
                continue
            row = {'Seed': seed, 'Fold': fold, 'Method': method, 'Class': cls}
            row.update(d)
            self.diags.append(row)

    def add_proba(self, key, y_proba):
        if y_proba is not None:
            self._proba[key] = np.asarray(y_proba, dtype=np.float32)

    def add_result(self, **kw):
        self.rows.append(kw)

    def flush_fold(self, seed, fold, y_true):
        path = os.path.join(self.outdir, 'proba',
                            f'seed{seed}_fold{fold}.npz')
        np.savez_compressed(path, y_true=np.asarray(y_true), **self._proba)
        self._proba = {}

    def close(self):
        df = pd.DataFrame(self.rows)
        df.to_csv(os.path.join(self.outdir, 'results.csv'), index=False)
        dd = pd.DataFrame(self.diags)
        dd.to_csv(os.path.join(self.outdir, 'diagnostics.csv'), index=False)
        with open(os.path.join(self.outdir, 'config.json'), 'w') as f:
            json.dump({'T': T_TEMP, 'tau': TAU, 'k': K_NN,
                       'eta': ETA_JITTER}, f, indent=2)
        with open(os.path.join(self.outdir, 'environment.json'), 'w') as f:
            json.dump(environment_record(), f, indent=2)
        return df, dd


def environment_record():
    """Package versions and accelerator state, recorded with every run.

    A result archive that does not carry its own environment cannot support a
    claim about which versions produced it. Recording this at write time costs
    nothing and removes the need to remember it afterwards.
    """
    import importlib
    import platform
    import sys

    packages = ['tensorflow', 'sklearn', 'imblearn', 'xgboost', 'numpy',
                'pandas', 'scipy', 'statsmodels', 'pywt', 'wfdb',
                'scikit_posthocs']
    versions = {}
    for name in packages:
        try:
            versions[name] = getattr(importlib.import_module(name),
                                     '__version__', 'unknown')
        except Exception:                                   # noqa: BLE001
            versions[name] = 'not installed'

    rec = {'python': sys.version.split()[0],
           'platform': platform.platform(),
           'packages': versions,
           'gpu': [], 'tf_deterministic_ops': os.environ.get(
               'TF_DETERMINISTIC_OPS', 'unset')}
    try:
        import tensorflow as tf
        rec['gpu'] = [d.name for d in tf.config.list_physical_devices('GPU')]
    except Exception:                                       # noqa: BLE001
        pass
    return rec
