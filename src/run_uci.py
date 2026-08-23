"""
run_uci.py
==========
EW-DDBS benchmark on the UCI Cardiotocography cohort (n = 2126, 3 classes).

Changes relative to the original notebook:
  * the silent synthetic-data fallback is REMOVED -- a failed download raises
  * "Baseline (None)" is included, so Table 3 can report the unsampled control
  * "ClassWeight" and "PriorCorrection" algorithm-level baselines are added
  * the FIGO feature ordering is made explicit (and can be permuted to test the
    inductive-bias claim in Sec. 2.1.1)
  * jitter, neighbour search and G-Mean follow ewddbs_core (audit D6/D8/D9)
  * entropy / latent / safe-ratio / weights / y_proba are all persisted

Usage
-----
    python run_uci.py --seeds 42 7 2024 --outdir results_uci
    python run_uci.py --permute-features --seeds 42 --outdir results_uci_perm
"""

import argparse
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE,
                                    KMeansSMOTE, RandomOverSampler, SVMSMOTE)

import ewddbs_core as C

# UCI CTG.xls, sheet "Data", columns 10:31 -- already grouped as FIGO prescribes
FIGO_ORDER = [
    # morphological
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP',
    # variability
    'ASTV', 'MSTV', 'ALTV', 'MLTV',
    # histogram
    'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median',
    'Variance', 'Tendency',
]


# The 21 feature columns carry these headers in the "Data" sheet. The
# deceleration/acceleration columns appear twice in CTG.xls: the plain names are
# raw episode COUNTS, the ".1" duplicates are the per-second RATES that the
# standard 21-feature UCI benchmark uses. We select by NAME so the mapping can
# never silently shift, and rename to the FIGO labels afterwards.
SOURCE_COLUMNS = [
    'LB', 'AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1',
    'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median',
    'Variance', 'Tendency',
]
TARGET_COLUMN = 'NSP'


def load_ctg(local_paths=('CTG.xls', 'data/CTG.xls')):
    """Load the UCI CTG dataset. Raises if it cannot be obtained.

    Two bugs inherited from the original notebook are fixed here:

    1. Row range. The data occupies rows 0-2125 of the "Data" sheet, not
       1-2126. The old slice `iloc[1:2127]` dropped the FIRST record and
       picked up an empty trailing row, which the NaN mask then removed --
       so every published UCI number was computed on 2125 records, not the
       2126 reported in the manuscript.
    2. Positional indexing. Columns are now selected by NAME, so a change in
       the sheet layout raises instead of silently shifting the features.

    The synthetic-data fallback of the original notebook is deliberately gone.
    """
    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls',
        'https://raw.githubusercontent.com/akmand/datasets/master/CTG.xls',
    ]
    df = None
    for p in local_paths:
        if os.path.exists(p):
            df = pd.read_excel(p, sheet_name='Data', skiprows=1)
            print(f'Loaded local file: {p}')
            break
    if df is None:
        for url in urls:
            try:
                print(f'Downloading {url} ...')
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                df = pd.read_excel(BytesIO(r.content), sheet_name='Data',
                                   skiprows=1)
                print('  -> ok')
                break
            except Exception as e:
                print(f'  -> failed: {type(e).__name__}: {str(e)[:90]}')
    if df is None:
        raise RuntimeError(
            'CTG dataset could not be obtained. Download CTG.xls from the UCI '
            'repository and place it next to this script. Refusing to fabricate '
            'data.')

    missing = [c for c in SOURCE_COLUMNS + [TARGET_COLUMN]
               if c not in df.columns]
    if missing:
        raise RuntimeError(
            f'CTG.xls is missing expected columns: {missing}. '
            f'Found: {list(df.columns)}')

    sub = df[SOURCE_COLUMNS + [TARGET_COLUMN]]
    # keep only complete records; this drops the blank trailing rows of the sheet
    ok = sub.notna().all(axis=1).values
    n_dropped = int((~ok).sum())
    sub = sub[ok]

    X = sub[SOURCE_COLUMNS].to_numpy(dtype=np.float32)
    y = sub[TARGET_COLUMN].to_numpy(dtype=int) - 1          # NSP 1,2,3 -> 0,1,2

    assert X.shape[1] == 21, f'expected 21 features, got {X.shape[1]}'
    assert set(np.unique(y)) == {0, 1, 2}, f'unexpected labels: {np.unique(y)}'

    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features '
          f'({n_dropped} incomplete rows dropped)')
    print(f'Class distribution (0=Normal, 1=Suspect, 2=Pathological): {counts}')

    # sanity check against the values reported in the manuscript
    if X.shape[0] != 2126 or counts != {0: 1655, 1: 295, 2: 176}:
        print('  WARNING: this does not match the 2126 / 1655-295-176 split '
              'reported in the manuscript. Verify the source file.')

    return X, y, list(FIGO_ORDER)


#: metric keys written when a (method, classifier) pair raises. Kept in one
#: place so it cannot drift away from what C.evaluate() actually returns.
FAILED_METRICS = ['F1-Macro', 'BalancedAccuracy', 'G-Mean', 'AUC', 'AUPRC',
                  'Brier', 'n_pred_classes', 'degenerate']


def run(X, y, feature_names, seeds, outdir, n_splits=5, permute=False,
        full_grid=True):
    os.makedirs(outdir, exist_ok=True)
    writer = C.ArtefactWriter(outdir)
    classes = np.unique(y)
    n_classes = len(classes)
    t0 = time.time()

    for seed in seeds:
        rng_master = np.random.default_rng(seed)
        order = np.arange(X.shape[1])
        if permute:
            order = rng_master.permutation(order)
            print(f'[seed {seed}] feature order permuted: '
                  f'{[feature_names[i] for i in order[:6]]} ...')
        Xs = X[:, order]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xs, y), start=1):
            print(f'\n=== seed {seed} | fold {fold}/{n_splits} ===')
            X_tr_raw, X_te_raw = Xs[tr], Xs[te]
            y_tr, y_te = y[tr], y[te]

            # leakage control: scaler fitted on the training fold only
            sc = StandardScaler().fit(X_tr_raw)
            X_tr, X_te = sc.transform(X_tr_raw), sc.transform(X_te_raw)

            print('  training CNN topology guide ...')
            ent, lat = C.cnn_topology_guide(X_tr, y_tr, seed=seed)
            writer.save_guide(seed, fold, ent, lat, y_tr)

            prior = np.array([np.mean(y_tr == c) for c in classes])

            methods = {}
            methods['Baseline'] = (X_tr, y_tr)
            for nm, smp in {
                'RandomOverSampler': RandomOverSampler(random_state=seed),
                'SMOTE': SMOTE(random_state=seed),
                'BorderlineSMOTE': BorderlineSMOTE(random_state=seed),
                'SVMSMOTE': SVMSMOTE(random_state=seed),
                'ADASYN': ADASYN(random_state=seed),
                'KMeansSMOTE': KMeansSMOTE(random_state=seed, k_neighbors=2,
                                           cluster_balance_threshold=0.01),
            }.items():
                try:
                    methods[nm] = smp.fit_resample(X_tr, y_tr)
                except Exception as e:
                    print(f'  {nm} failed: {str(e)[:60]}')

            for nm, (ue, us, tk) in C.ablation_grid(full_grid).items():
                Xr, yr, diag = C.ew_ddbs_resample(
                    X_tr, y_tr, ent, lat, use_entropy=ue, use_safe=us,
                    apply_tomek=tk, rng=np.random.default_rng(seed * 1000 + fold))
                methods[nm] = (Xr, yr)
                writer.add_diag(seed, fold, nm, diag)

            for m_name, (Xr, yr) in methods.items():
                for c_name in C.CLASSIFIERS:
                    try:
                        clf = C.fit_classifier(c_name, Xr, yr, seed)
                        y_pred = clf.predict(X_te)
                        proba, calib = C.predict_proba_safe(clf, X_te,
                                                            n_classes)
                        met = C.evaluate(y_te, y_pred, proba, classes)
                        writer.add_proba(f'{m_name}|{c_name}', proba)
                    except Exception as e:
                        met = {k: np.nan for k in FAILED_METRICS}
                        calib = False
                    writer.add_result(Seed=seed, Fold=fold, Oversampling=m_name,
                                      Classifier=c_name, Calibratable=calib,
                                      **met)

            # ---- algorithm-level baselines (no resampling) ----------------
            for c_name in C.CLASSIFIERS:
                try:
                    clf = C.fit_classifier(c_name, X_tr, y_tr, seed,
                                           class_weight='balanced')
                    y_pred = clf.predict(X_te)
                    proba, calib = C.predict_proba_safe(clf, X_te, n_classes)
                    met = C.evaluate(y_te, y_pred, proba, classes)
                    writer.add_proba(f'ClassWeight|{c_name}', proba)
                except Exception:
                    met = {k: np.nan for k in FAILED_METRICS}
                    calib = False
                writer.add_result(Seed=seed, Fold=fold,
                                  Oversampling='ClassWeight',
                                  Classifier=c_name, Calibratable=calib, **met)

                # prior correction / threshold moving on the unsampled model
                try:
                    clf = C.fit_classifier(c_name, X_tr, y_tr, seed)
                    proba, calib = C.predict_proba_safe(clf, X_te, n_classes)
                    if proba is not None:
                        pc = C.prior_correction(proba, prior)
                        y_pred = classes[np.argmax(pc, axis=1)]
                        met = C.evaluate(y_te, y_pred, pc, classes)
                        writer.add_proba(f'PriorCorrection|{c_name}', pc)
                    else:
                        raise ValueError
                except Exception:
                    met = {k: np.nan for k in FAILED_METRICS}
                    calib = False
                writer.add_result(Seed=seed, Fold=fold,
                                  Oversampling='PriorCorrection',
                                  Classifier=c_name, Calibratable=calib, **met)

            writer.flush_fold(seed, fold, y_te)
            print(f'  fold done ({(time.time() - t0) / 60:.1f} min elapsed)')

    df, dd = writer.close()
    print(f'\nSaved {len(df)} result rows and {len(dd)} diagnostic rows '
          f'to {outdir}/')
    print(f'Total runtime: {(time.time() - t0) / 60:.1f} min')
    return df, dd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--outdir', default='results_uci')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--permute-features', action='store_true',
                    help='randomly permute the input feature order (tests the '
                         'FIGO inductive-bias claim of Sec. 2.1.1)')
    ap.add_argument('--reduced-grid', action='store_true',
                    help='run 4 instead of 8 EW-DDBS ablation variants')
    a = ap.parse_args()

    X, y, names = load_ctg()
    run(X, y, names, a.seeds, a.outdir, n_splits=a.folds,
        permute=a.permute_features, full_grid=not a.reduced_grid)
