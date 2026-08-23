"""
verify_paper_numbers.py
=======================
Regenerate every quantitative claim in the manuscript directly from the archived
result files, so that a reader can check the paper without re-running the
benchmark.

This script is the operational form of the claim made in the manuscript that the
analysis "can be verified from the recorded outputs rather than by
re-execution". It reads only `results.csv` and `diagnostics.csv` from each
result directory and prints each number next to the table or section that cites
it.

Usage
-----
    python scripts/verify_paper_numbers.py --root results/

Expected layout under --root:

    results_uci_all/        UCI, seeds 42 + 7 + 2024   (3,825 rows)
    results_uci/            UCI, seed 42 only          (1,275 rows)
    results_uci_perm/       UCI, seed 42, permuted     (1,275 rows)
    results_ctu_all/        CTU-UHB, 3 seeds           (7,650 rows)
    results_ctu/            CTU-UHB, seed 42, first run (2,550 rows)
    results_ctu_run2/       CTU-UHB, seed 42, repeat run (2,550 rows)
    results_ctu_perm/       CTU-UHB, seed 42, permuted (2,550 rows)

Directories that are absent are skipped with a notice; the script never
fabricates a value it cannot compute.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

METRICS = ['F1-Macro', 'G-Mean', 'BalancedAccuracy', 'AUC', 'AUPRC', 'Brier']
FOLD_KEY = ['Seed', 'Fold', 'Classifier']
ROW_KEY = ['Seed', 'Fold', 'Oversampling', 'Classifier']
REFERENCE = 'EWDDBS (Entropy+Safe+Tomek)'
ORDER_INVARIANT = ['LogReg', 'LDA', 'GNB', 'KNN', 'SVM-RBF', 'LinearSVM', 'SGD']


def head(title):
    print('\n' + '=' * 74)
    print(title)
    print('=' * 74)


def load(root, name):
    path = os.path.join(root, name, 'results.csv')
    if not os.path.exists(path):
        print(f'  [skip] {name}/results.csv not found')
        return None
    df = pd.read_csv(path)
    print(f'  [ok]   {name:22s} {len(df):5d} rows, seeds {sorted(df.Seed.unique())}')
    return df


def aggregate(df, label, table):
    head(f'{table}  Aggregate performance, {label}')
    g = df.groupby('Oversampling')[METRICS].mean()
    sd = df.groupby('Oversampling')['F1-Macro'].std()
    g = g.reindex(g['F1-Macro'].sort_values(ascending=False).index)
    print(f'{"Strategy":30s}' + ''.join(f'{m:>12s}' for m in METRICS))
    for k, r in g.iterrows():
        print(f'{k:30s}' + ''.join(f'{r[m]:12.4f}' for m in METRICS)
              + f'   (SD F1 {sd[k]:.3f})')
    print('\n  best per metric:')
    for m in METRICS:
        b = g[m].idxmin() if m == 'Brier' else g[m].idxmax()
        print(f'    {m:18s} {b:30s} {g.loc[b, m]:.4f}')
    return g


def vs_baseline(df, label, table, metric='F1-Macro'):
    head(f'{table}  {metric} relative to the uncorrected baseline, {label}')
    base = df[df.Oversampling == 'Baseline'].set_index(FOLD_KEY)[metric]
    rows = []
    for m in df.Oversampling.unique():
        if m == 'Baseline':
            continue
        x = df[df.Oversampling == m].set_index(FOLD_KEY)[metric]
        j = pd.concat([base.rename('b'), x.rename('v')], axis=1).dropna()
        d = j['v'] - j['b']
        cliffs = np.mean(np.sign(j['v'].values[:, None] - j['b'].values[None, :]))
        rows.append(dict(Strategy=m, n=len(j), delta=d.mean(),
                         p_raw=wilcoxon(d)[1], cliffs=cliffs))
    r = pd.DataFrame(rows)
    r['p_holm'] = multipletests(r['p_raw'], method='holm')[1]
    r = r.sort_values('delta', ascending=False)
    print(r.round(5).to_string(index=False))
    n_sig = int((r.p_holm < 0.05).sum())
    n_neg = int((r.delta < 0).sum())
    print(f'\n  significant after Holm : {n_sig}/{len(r)}')
    print(f'  negative differences   : {n_neg}/{len(r)}')
    print(f'  largest |Cliff\'s delta|: {r.cliffs.abs().max():.3f}')
    return r


def calibration_contrast(df, label):
    head(f'Section 4.1  Calibration contrast for the improving oversamplers, {label}')
    base = {m: df[df.Oversampling == 'Baseline'].set_index(FOLD_KEY)[m]
            for m in ('AUPRC', 'Brier')}
    for strategy in ('SVMSMOTE', 'KMeansSMOTE'):
        if strategy not in set(df.Oversampling):
            continue
        for m in ('AUPRC', 'Brier'):
            x = df[df.Oversampling == strategy].set_index(FOLD_KEY)[m]
            j = pd.concat([base[m].rename('b'), x.rename('v')], axis=1).dropna()
            d = j['v'] - j['b']
            print(f'  {strategy:12s} {m:6s} delta = {d.mean():+.5f}   '
                  f'p = {wilcoxon(d)[1]:.5f}')


def entropy_axis(df, label):
    head(f'Section 4.5  Entropy axis against the uniform-weighted control, {label}')
    pairs = [('EWDDBS (Uniform)', 'EWDDBS (Entropy)'),
             ('EWDDBS (Safe)', 'EWDDBS (Entropy+Safe)'),
             ('EWDDBS (Uniform+Tomek)', 'EWDDBS (Entropy+Tomek)'),
             ('EWDDBS (Safe+Tomek)', 'EWDDBS (Entropy+Safe+Tomek)')]
    deltas, ps, labels = [], [], []
    for u, e in pairs:
        x = df[df.Oversampling == u].set_index(FOLD_KEY)['F1-Macro']
        y = df[df.Oversampling == e].set_index(FOLD_KEY)['F1-Macro']
        j = pd.concat([x.rename('u'), y.rename('e')], axis=1).dropna()
        d = j['e'] - j['u']
        deltas.append(d.mean()); ps.append(wilcoxon(d)[1]); labels.append((e, u))
    holm = multipletests(ps, method='holm')[1]
    for (e, u), dm, p, h in zip(labels, deltas, ps, holm):
        flag = ' *' if h < 0.05 else ''
        print(f'  {e:30s} - {u:24s} = {dm:+.5f}  p = {p:.4f}  '
              f'p_Holm = {h:.4f}{flag}')


def diagnostics(root, name, label):
    path = os.path.join(root, name, 'diagnostics.csv')
    if not os.path.exists(path):
        print(f'  [skip] {name}/diagnostics.csv not found')
        return
    d = pd.read_csv(path)
    weighted = d[~d.Method.str.contains('Uniform')]
    print(f'  {label}: CV(W) = {weighted.W_cv.mean():.4f}   '
          f'ESS ratio = {weighted.P_ess_ratio.mean():.4f}   '
          f'S = 1 fraction = {d["S_frac_eq_1"].mean():.4f}   '
          f'gated by tau = {d["S_frac_below_tau"].mean():.4f}   '
          f'entropy < 0.01 = {d["H_frac_below_0.01"].mean():.4f}')


def permutation(orig, perm, label, table):
    head(f'{table}  Feature-ordering permutation control, {label}')
    m = orig.merge(perm, on=ROW_KEY, suffixes=('_o', '_p'))
    a = orig.groupby('Oversampling')[METRICS].mean()
    b = perm.groupby('Oversampling')[METRICS].mean()
    print(f'  paired rows: {len(m)}')
    for c in METRICS:
        asc = (c == 'Brier')
        rho = spearmanr(a[c].rank(ascending=asc), b[c].rank(ascending=asc))[0]
        d = m[c + '_p'] - m[c + '_o']
        print(f'    {c:18s} rho = {rho:+.3f}   delta = {d.mean():+.5f}   '
              f'p = {wilcoxon(d)[1]:.4f}')
    ps = []
    for _, g in m.groupby('Oversampling'):
        d = g['F1-Macro_p'].values - g['F1-Macro_o'].values
        ps.append(wilcoxon(d)[1] if np.any(d != 0) else 1.0)
    holm = multipletests(ps, method='holm')[1]
    print(f'\n  per-strategy F1 after Holm: {int((holm < 0.05).sum())}/{len(holm)} '
          f'significant (minimum p_Holm = {holm.min():.4f})')

    m['same'] = (m['F1-Macro_p'] - m['F1-Macro_o']).abs() < 1e-12
    b_ = m[m.Oversampling == 'Baseline']
    inv = b_[b_.Classifier.isin(ORDER_INVARIANT)]
    tree = b_[~b_.Classifier.isin(ORDER_INVARIANT + ['QDA'])]
    print(f'  validity check, uncorrected baseline:')
    print(f'    order-invariant learners identical : {inv["same"].mean() * 100:.0f}%')
    print(f'    tree ensembles identical           : {tree["same"].mean() * 100:.0f}%')
    qda = b_[b_.Classifier == 'QDA']
    if len(qda):
        dmax = (qda['F1-Macro_p'] - qda['F1-Macro_o']).abs().max()
        print(f'    QDA maximum |delta| F1             : {dmax:.4f}')


def determinism(run1, run2):
    head('Table 7  Determinism audit, CTU-UHB seed 42, two executions')
    m = run1.merge(run2, on=ROW_KEY, suffixes=('_1', '_2'))
    m['same'] = (m['F1-Macro_2'] - m['F1-Macro_1']).abs() < 1e-12
    ew = m[m.Oversampling.str.startswith('EWDDBS')]
    other = m[~m.Oversampling.str.startswith('EWDDBS')]
    d = ew['F1-Macro_2'] - ew['F1-Macro_1']
    print(f'  comparator strategies : {other["same"].mean() * 100:5.1f}% identical '
          f'({len(other)} rows)')
    print(f'  EW-DDBS variants      : {ew["same"].mean() * 100:5.1f}% identical '
          f'({len(ew)} rows)')
    print(f'  row-level  mean |delta| = {d.abs().mean():.4f}   '
          f'SD = {d.std():.4f}   max = {d.abs().max():.4f}')
    a1 = run1.groupby('Oversampling')['F1-Macro'].mean()
    a2 = run2.groupby('Oversampling')['F1-Macro'].mean()
    shift = (a2 - a1)
    ew_idx = [i for i in shift.index if i.startswith('EWDDBS')]
    r1 = a1.rank(ascending=False); r2 = a2.rank(ascending=False)
    sd_run = shift[ew_idx].std(ddof=1) / np.sqrt(2)
    print(f'  aggregate  max |shift|  = {shift.abs().max():.4f}   '
          f'mean |shift| (EW-DDBS) = {shift[ew_idx].abs().mean():.4f}')
    print(f'  rank correlation between executions = {spearmanr(a1, a2)[0]:+.3f}')
    print(f'  largest rank displacement           = {int((r2 - r1).abs().max())} positions')
    print(f'  implied per-execution aggregate SD  = {sd_run:.4f}')
    return shift[ew_idx].abs().mean()


def calibrate_permutation(root, noise_mean):
    head('Section 4.4  Permutation effect calibrated against re-execution noise')
    o = load_quiet(root, 'results_ctu')
    p = load_quiet(root, 'results_ctu_perm')
    if o is None or p is None:
        print('  [skip] requires results_ctu and results_ctu_perm')
        return
    a = o.groupby('Oversampling')['F1-Macro'].mean()
    b = p.groupby('Oversampling')['F1-Macro'].mean()
    ew = [i for i in a.index if i.startswith('EWDDBS')]
    eff = (b[ew] - a[ew]).abs().mean()
    print(f'  mean |permutation effect|   = {eff:.4f}')
    print(f'  mean |re-execution noise|   = {noise_mean:.4f}')
    print(f'  ratio                       = {eff / noise_mean:.2f}')


def degenerate(df, label):
    head(f'Section 4.7  Degenerate predictions, {label}')
    if 'degenerate' not in df.columns:
        print('  [skip] this result set predates the degeneracy flag')
        return
    g = df.groupby(['Oversampling', 'Classifier'])['degenerate'].mean()
    print(f'  pairs with at least one degenerate fold : {int((g > 0).sum())}')
    print(f'  pairs degenerate in over half of folds  : {int((g > 0.5).sum())}')
    by_clf = (df.groupby('Classifier')['degenerate'].mean() * 100).sort_values(ascending=False)
    by_str = (df.groupby('Oversampling')['degenerate'].mean() * 100).sort_values(ascending=False)
    print(f'  highest by classifier : {by_clf.index[0]} ({by_clf.iloc[0]:.1f}%)')
    print('  by strategy (top 3 and the uncorrected baseline):')
    for k in list(by_str.index[:3]) + ['Baseline']:
        print(f'    {k:30s} {by_str[k]:5.1f}%')
    pc = df[df.Oversampling == 'PriorCorrection']
    if len(pc):
        print(f'  PriorCorrection aggregate F1        : {pc["F1-Macro"].mean():.4f}')
        print(f'  PriorCorrection without AdaBoost    : '
              f'{pc[pc.Classifier != "AdaBoost"]["F1-Macro"].mean():.4f}')


def load_quiet(root, name):
    path = os.path.join(root, name, 'results.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def main(root):
    head('Loading archived result sets')
    uci_all = load(root, 'results_uci_all')
    uci_42 = load(root, 'results_uci')
    uci_perm = load(root, 'results_uci_perm')
    ctu_all = load(root, 'results_ctu_all')
    ctu_r2 = load(root, 'results_ctu')
    ctu_r1 = load(root, 'results_ctu_run2')
    if ctu_r1 is None:                    # older archives used run1
        ctu_r1 = load(root, 'results_ctu_run1')
    ctu_perm = load(root, 'results_ctu_perm')

    if uci_all is not None:
        aggregate(uci_all, 'UCI, three seeds', 'Table 3')
        vs_baseline(uci_all, 'UCI, three seeds', 'Table 5')
        entropy_axis(uci_all, 'UCI, three seeds')
        degenerate(uci_all, 'UCI, three seeds')
    if ctu_all is not None:
        aggregate(ctu_all, 'CTU-UHB, three seeds', 'Table 4')
        vs_baseline(ctu_all, 'CTU-UHB, three seeds', 'Table 5')
        calibration_contrast(ctu_all, 'CTU-UHB, three seeds')
        entropy_axis(ctu_all, 'CTU-UHB, three seeds')
        degenerate(ctu_all, 'CTU-UHB, three seeds')

    head('Table 8  Weighting diagnostics')
    diagnostics(root, 'results_uci_all', 'UCI       ')
    diagnostics(root, 'results_ctu_all', 'CTU-UHB   ')

    if uci_42 is not None and uci_perm is not None:
        permutation(uci_42, uci_perm, 'UCI, seed 42', 'Table 6')
    if ctu_r2 is not None and ctu_perm is not None:
        permutation(ctu_r2, ctu_perm, 'CTU-UHB, seed 42', 'Table 6')

    if ctu_r1 is not None and ctu_r2 is not None:
        noise = determinism(ctu_r1, ctu_r2)
        calibrate_permutation(root, noise)
    else:
        print('\n  [skip] determinism audit requires results_ctu and a repeat '
              'execution in results_ctu_run2')

    head('Done')
    print('  Every value above was computed from the archived CSV files.')
    print('  No number in this report is hard-coded.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='results',
                    help='directory containing the extracted result folders')
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        sys.exit(f'error: {a.root} is not a directory')
    main(a.root)
