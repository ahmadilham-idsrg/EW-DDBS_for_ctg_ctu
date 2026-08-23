"""
run_ctu_uhb.py
==============
EW-DDBS benchmark on the CTU-UHB intrapartum cohort (PhysioNet, n <= 552).

Changes relative to the original notebook:
  * the 62-dimensional feature pipeline described in Contribution (2) is now
    ACTUALLY IMPLEMENTED (ctg_features.py): descriptive stats, db4 wavelet
    decomposition, VLF/LF/HF + spectral edge frequency, Sample/Approximate/
    Permutation entropy, CTG morphology, and FHR-UC coupling. The old notebook
    produced 30 plain statistics and none of these families.
  * PCA is OFF by default, so "synthesis occurs in the original input space"
    (Contribution 1) is true here as well. Set --pca to restore the old
    behaviour, but then that claim must be qualified in the manuscript.
  * the CNN guide, jitter, neighbour search, ablation grid and G-Mean are
    imported from ewddbs_core, so both cohorts are genuinely identical
    (Contribution 3 becomes true).
  * artefacts (entropy, latent, safe-ratio stats, y_proba) are persisted.

Usage
-----
    python run_ctu_uhb.py --seeds 42 7 2024 --outdir results_ctu
    python run_ctu_uhb.py --permute-features --seeds 42 --outdir results_ctu_perm
"""

import argparse
import os
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE,
                                    KMeansSMOTE, RandomOverSampler, SVMSMOTE)

import ewddbs_core as C
from ctg_features import FEATURE_NAMES, extract_62_features

PH_THRESHOLD = 7.15
CACHE = 'ctu_uhb_features62_cache.npz'


def parse_ph(header_comments):
    for line in header_comments:
        parts = line.strip().lstrip('#').strip().split()
        if len(parts) >= 2 and parts[0].lower() == 'ph':
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def prepare_ctu_uhb(max_records=552, cache=CACHE):
    """Download CTU-UHB and build the 62-D feature matrix (cached)."""
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y = d['X'], d['y']
        print(f'Loaded cache: {X.shape[0]} samples, {X.shape[1]} features')
        if X.shape[1] != 62:
            raise RuntimeError(
                f'Cache holds {X.shape[1]} features, expected 62. Delete '
                f'{cache} so it is rebuilt with the new extractor.')
        print(f'Class distribution: '
              f'{dict(zip(*np.unique(y, return_counts=True)))}')
        return X, y

    import wfdb
    print('Fetching record list from PhysioNet (ctu-uhb-ctgdb/1.0.0) ...')
    rec_ids = wfdb.get_record_list('ctu-uhb-ctgdb/1.0.0')
    print(f'  -> {len(rec_ids)} records')

    feats, labels, n_no_ph, n_err = [], [], 0, 0
    t0 = time.time()
    for i, rid in enumerate(rec_ids):
        if len(feats) >= max_records:
            break
        try:
            hdr = wfdb.rdheader(rid, pn_dir='ctu-uhb-ctgdb/1.0.0')
            ph = parse_ph(hdr.comments)
            if ph is None:
                n_no_ph += 1
                continue
            rec = wfdb.rdrecord(rid, pn_dir='ctu-uhb-ctgdb/1.0.0')
            fhr, uc = rec.p_signal[:, 0], rec.p_signal[:, 1]
            feats.append(extract_62_features(fhr, uc))
            labels.append(0 if ph >= PH_THRESHOLD else 1)
            if len(feats) % 25 == 0:
                print(f'  ...{len(feats)} processed '
                      f'({(time.time() - t0) / 60:.1f} min)')
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f'  skipped {rid}: {type(e).__name__}: {str(e)[:70]}')

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    if X.shape[0] == 0:
        raise RuntimeError('No CTU-UHB records could be processed.')
    np.savez_compressed(cache, X=X, y=y,
                        feature_names=np.array(FEATURE_NAMES))
    print(f'\nDataset: {X.shape[0]} samples, {X.shape[1]} features')
    print(f'Class distribution: '
          f'{dict(zip(*np.unique(y, return_counts=True)))}')
    if n_no_ph:
        print(f'  ({n_no_ph} records had no parseable pH)')
    if n_err:
        print(f'  ({n_err} records failed)')
    return X, y


#: metric keys written when a (method, classifier) pair raises. Kept in one
#: place so it cannot drift away from what C.evaluate() actually returns.
FAILED_METRICS = ['F1-Macro', 'BalancedAccuracy', 'G-Mean', 'AUC', 'AUPRC',
                  'Brier', 'n_pred_classes', 'degenerate']


def run(X, y, seeds, outdir, n_splits=10, use_pca=False, full_grid=True,
        permute=False, feature_names=None):
    os.makedirs(outdir, exist_ok=True)
    writer = C.ArtefactWriter(outdir)
    classes = np.unique(y)
    n_classes = len(classes)
    t0 = time.time()

    for seed in seeds:
        rng_master = np.random.default_rng(seed)
        order = np.arange(X.shape[1])
        if permute:
            # Negative control: destroy the feature ordering the CNN topology
            # guide is claimed to exploit. If the ranking survives unchanged,
            # the ordering carries no inductive bias.
            order = rng_master.permutation(order)
            shown = ([feature_names[i] for i in order[:6]] if feature_names
                     else order[:6].tolist())
            print(f'[seed {seed}] feature order permuted: {shown} ...')
        Xs = X[:, order]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xs, y), start=1):
            print(f'\n=== seed {seed} | fold {fold}/{n_splits} ===')
            X_tr_raw, X_te_raw = Xs[tr], Xs[te]
            y_tr, y_te = y[tr], y[te]

            imp = SimpleImputer(strategy='median').fit(X_tr_raw)
            X_tr, X_te = imp.transform(X_tr_raw), imp.transform(X_te_raw)
            sc = StandardScaler().fit(X_tr)
            X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
            if use_pca:
                pca = PCA(n_components=0.95, random_state=seed).fit(X_tr)
                X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)
                print(f'  PCA -> {X_tr.shape[1]} components '
                      f'(NOTE: input-space interpretability claim no longer holds)')

            print('  training CNN topology guide ...')
            ent, lat = C.cnn_topology_guide(X_tr, y_tr, seed=seed)
            writer.save_guide(seed, fold, ent, lat, y_tr)

            prior = np.array([np.mean(y_tr == c) for c in classes])

            methods = {'Baseline': (X_tr, y_tr)}
            for nm, smp in {
                'RandomOverSampler': RandomOverSampler(random_state=seed),
                'SMOTE': SMOTE(random_state=seed, k_neighbors=3),
                'BorderlineSMOTE': BorderlineSMOTE(random_state=seed,
                                                   k_neighbors=3),
                'SVMSMOTE': SVMSMOTE(random_state=seed, k_neighbors=3),
                'ADASYN': ADASYN(random_state=seed, n_neighbors=3),
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
                    apply_tomek=tk,
                    rng=np.random.default_rng(seed * 1000 + fold))
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
                    except Exception:
                        met = {k: np.nan for k in FAILED_METRICS}
                        calib = False
                    writer.add_result(Seed=seed, Fold=fold,
                                      Oversampling=m_name, Classifier=c_name,
                                      Calibratable=calib, **met)

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

                try:
                    clf = C.fit_classifier(c_name, X_tr, y_tr, seed)
                    proba, calib = C.predict_proba_safe(clf, X_te, n_classes)
                    if proba is None:
                        raise ValueError
                    pc = C.prior_correction(proba, prior)
                    y_pred = classes[np.argmax(pc, axis=1)]
                    met = C.evaluate(y_te, y_pred, pc, classes)
                    writer.add_proba(f'PriorCorrection|{c_name}', pc)
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
    ap.add_argument('--outdir', default='results_ctu')
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--max-records', type=int, default=552)
    ap.add_argument('--pca', action='store_true',
                    help='apply PCA(95%%) as in the original notebook; note '
                         'this invalidates the input-space interpretability claim')
    ap.add_argument('--reduced-grid', action='store_true')
    ap.add_argument('--permute-features', action='store_true',
                    help='randomly permute the input feature order (negative '
                         'control for the inductive-bias claim of Sec. 2.1.1; '
                         'mirrors run_uci.py so both cohorts have it)')
    a = ap.parse_args()

    X, y = prepare_ctu_uhb(max_records=a.max_records)
    run(X, y, a.seeds, a.outdir, n_splits=a.folds, use_pca=a.pca,
        full_grid=not a.reduced_grid, permute=a.permute_features,
        feature_names=FEATURE_NAMES)
