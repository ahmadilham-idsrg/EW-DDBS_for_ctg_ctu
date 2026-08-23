"""
analyze_results.py
==================
Post-hoc analysis of an EW-DDBS run. Produces everything the revision needs:

  T2  weighting diagnostics -- is W_i degenerate?          -> diag_summary.csv
  T1  calibration (Brier, reliability curves)              -> calibration.csv
                                                              reliability_*.png
  --  aggregate tables (mean +/- SD)                       -> table_methods.csv
                                                              table_by_classifier.csv
  --  Friedman (block=fold and block=fold x classifier)
      + Nemenyi post-hoc                                   -> nemenyi_*.csv
  T7  Wilcoxon ablation contrasts                          -> ablation_wilcoxon.csv
  T6  effect sizes (Cliff's delta, Cohen's d), CIs,
      Holm-Bonferroni correction                           -> included above
  --  LaTeX snippets for the manuscript                    -> latex_tables.tex

No TensorFlow required.

Usage
-----
    python analyze_results.py --dir results_uci --ref "EWDDBS (Entropy+Safe+Tomek)"
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ['F1-Macro', 'G-Mean', 'BalancedAccuracy', 'AUC', 'AUPRC', 'Brier']


# ----------------------------------------------------------------------
# effect sizes and corrections
# ----------------------------------------------------------------------
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def cohens_d_paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))


def ci_mean_diff(a, b, alpha=0.05):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = np.std(d, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(np.mean(d) - t * se), float(np.mean(d) + t * se)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def magnitude(delta):
    a = abs(delta)
    if a < 0.147:
        return 'negligible'
    if a < 0.33:
        return 'small'
    if a < 0.474:
        return 'medium'
    return 'large'


# ----------------------------------------------------------------------
# T2 -- weighting diagnostics
# ----------------------------------------------------------------------
def analyse_diagnostics(dd, outdir):
    if dd is None or len(dd) == 0:
        print('No diagnostics found.')
        return None
    cols = ['H_mean', 'H_median', 'H_frac_below_0.01', 'S_mean', 'S_median',
            'S_frac_eq_1', 'S_frac_below_tau', 'n_gated_by_tau', 'W_cv',
            'P_ess_ratio']
    cols = [c for c in cols if c in dd.columns]
    summ = (dd.groupby(['Method', 'Class'])[cols]
              .agg(['mean', 'std']).round(4))
    summ.to_csv(os.path.join(outdir, 'diag_summary.csv'))

    print('\n' + '=' * 78)
    print('T2  WEIGHTING DIAGNOSTICS  (is W_i degenerate?)')
    print('=' * 78)
    print(summ.to_string())

    full = dd[dd['Method'].str.contains('Entropy\\+Safe', regex=True)]
    if len(full) == 0:
        full = dd
    w_cv = full['W_cv'].mean() if 'W_cv' in full else np.nan
    ess = full['P_ess_ratio'].mean() if 'P_ess_ratio' in full else np.nan
    s1 = full['S_frac_eq_1'].mean() if 'S_frac_eq_1' in full else np.nan
    gated = full['S_frac_below_tau'].mean() if 'S_frac_below_tau' in full else np.nan
    h0 = full['H_frac_below_0.01'].mean() if 'H_frac_below_0.01' in full else np.nan

    print('\nVERDICT')
    print(f'  mean CV(W_i)                 = {w_cv:.4f}')
    print(f'  mean effective-sample ratio  = {ess:.4f}   (1.0 = uniform)')
    print(f'  mean fraction with S(z) = 1  = {s1:.4f}')
    print(f'  mean fraction gated by tau   = {gated:.4f}')
    print(f'  mean fraction with H < 0.01  = {h0:.4f}')
    if w_cv < 0.05 or ess > 0.98:
        print('  --> DEGENERATE: the weighting is effectively uniform, so')
        print('      EW-DDBS reduces to SMOTE with latent-space neighbours.')
        print('      Report this explicitly and consider out-of-fold H/z.')
    else:
        print('  --> NON-DEGENERATE: the weighting genuinely reshapes the')
        print('      parent distribution. Report the distributions in the paper.')
    return summ


# ----------------------------------------------------------------------
# T1 -- calibration
# ----------------------------------------------------------------------
def analyse_calibration(df, resdir, outdir, n_bins=10):
    files = sorted(glob.glob(os.path.join(resdir, 'proba', '*.npz')))
    if not files:
        print('\nNo probability files found; skipping reliability curves.')
        return None

    rows = []
    for f in files:
        d = np.load(f)
        y_true = d['y_true']
        classes = np.unique(y_true)
        for key in d.files:
            if key == 'y_true':
                continue
            p = d[key]
            if p.ndim != 2 or p.shape[0] != len(y_true):
                continue
            method, clf = key.split('|', 1)
            # positive class = last (minority in both cohorts)
            pos = p[:, -1]
            yb = (y_true == classes[-1]).astype(float)
            bins = np.clip(np.digitize(pos, np.linspace(0, 1, n_bins + 1)) - 1,
                           0, n_bins - 1)
            for b in range(n_bins):
                m = bins == b
                if m.sum() == 0:
                    continue
                rows.append({'File': os.path.basename(f), 'Method': method,
                             'Classifier': clf, 'bin': b,
                             'conf': float(pos[m].mean()),
                             'acc': float(yb[m].mean()), 'n': int(m.sum())})
    rel = pd.DataFrame(rows)
    rel.to_csv(os.path.join(outdir, 'reliability_bins.csv'), index=False)

    cal = df[df.get('Calibratable', True) == True] if 'Calibratable' in df \
        else df
    tab = (cal.groupby('Oversampling')['Brier']
              .agg(['mean', 'std', 'count']).round(4)
              .sort_values('mean'))
    tab.to_csv(os.path.join(outdir, 'calibration.csv'))

    print('\n' + '=' * 78)
    print('T1  CALIBRATION  (lower Brier is better; uncalibratable models excluded)')
    print('=' * 78)
    print(tab.to_string())

    # expected calibration error per method
    if rel.empty or 'Method' not in rel.columns:
        # No reliability bins were produced. This happens when no probability
        # file was written, for instance on a subset run with a single
        # uncalibratable classifier. Report the absence rather than crashing.
        print('\nExpected Calibration Error: not computed '
              '(no reliability bins were produced).')
        pd.DataFrame(columns=['Method', 'ECE']).to_csv(
            os.path.join(outdir, 'ece.csv'), index=False)
        return tab

    ece = (rel.groupby(['Method'])
              .apply(lambda g: np.average(np.abs(g['conf'] - g['acc']),
                                          weights=g['n']))
              .round(4).sort_values())
    ece.name = 'ECE'
    ece.to_csv(os.path.join(outdir, 'ece.csv'))
    print('\nExpected Calibration Error (minority class):')
    print(ece.to_string())
    return tab


# ----------------------------------------------------------------------
# aggregate tables + statistics
# ----------------------------------------------------------------------
def report_degenerate(df, outdir):
    """List (method, classifier) pairs that collapse to a single predicted
    class. These still yield a finite F1 and would otherwise be averaged into
    the aggregates unnoticed."""
    if 'degenerate' not in df.columns:
        return None
    g = (df.groupby(['Oversampling', 'Classifier'])['degenerate']
           .mean().reset_index())
    bad = g[g['degenerate'] > 0].sort_values('degenerate', ascending=False)
    bad.to_csv(os.path.join(outdir, 'degenerate_predictions.csv'), index=False)
    print('\n' + '=' * 78)
    print('DEGENERATE PREDICTIONS (fraction of folds collapsing to one class)')
    print('=' * 78)
    if len(bad) == 0:
        print('  none')
    else:
        print(bad.to_string(index=False))
        print('\n  Report these explicitly, or state that the aggregate is')
        print('  computed with them excluded -- do not average them in silently.')
    return bad


def aggregate(df, outdir):
    have = [m for m in METRICS if m in df.columns]
    t = (df.groupby('Oversampling')[have].agg(['mean', 'std']).round(4))
    t = t.sort_values(('F1-Macro', 'mean'), ascending=False)
    t.to_csv(os.path.join(outdir, 'table_methods.csv'))
    print('\n' + '=' * 78)
    print('AGGREGATE PERFORMANCE (mean +/- SD over seeds x folds x classifiers)')
    print('=' * 78)
    print(t.to_string())

    bc = (df.groupby(['Oversampling', 'Classifier'])[have].mean().round(4))
    bc.to_csv(os.path.join(outdir, 'table_by_classifier.csv'))
    return t


def friedman_nemenyi(df, outdir, metric='F1-Macro', ref=None):
    import scikit_posthocs as sp
    out = {}
    for scheme, keys in [('S1_fold', ['Seed', 'Fold']),
                         ('S2_fold_x_clf', ['Seed', 'Fold', 'Classifier'])]:
        piv = (df.groupby(keys + ['Oversampling'])[metric].mean()
                 .unstack().dropna())
        if piv.shape[0] < 3 or piv.shape[1] < 3:
            continue
        chi, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
        ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
        print(f'\nFriedman [{scheme}] N={piv.shape[0]}: '
              f'chi2={chi:.3f}, p={p:.3e}')
        print('  average ranks (lower = better):')
        for k, v in ranks.items():
            print(f'    {k:32s} {v:.3f}')
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = nem.columns = piv.columns
        nem.round(4).to_csv(os.path.join(outdir, f'nemenyi_{scheme}.csv'))
        if ref and ref in nem.columns:
            s = nem[ref].drop(ref).sort_values()
            print(f'  Nemenyi vs {ref}:')
            for k, v in s.items():
                flag = '*' if v < 0.05 else 'ns'
                print(f'    {k:32s} p={v:.4f} {flag}')
        out[scheme] = {'chi2': chi, 'p': p, 'ranks': ranks, 'nemenyi': nem}
    return out


def ablation_tests(df, outdir, metrics=('F1-Macro', 'AUPRC')):
    """Wilcoxon on every EW-DDBS pair, with effect size, CI and Holm."""
    variants = sorted(v for v in df['Oversampling'].unique()
                      if v.startswith('EWDDBS'))
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                a, b = variants[i], variants[j]
                sa = (df[df['Oversampling'] == a]
                      .set_index(['Seed', 'Fold', 'Classifier'])[m])
                sb = (df[df['Oversampling'] == b]
                      .set_index(['Seed', 'Fold', 'Classifier'])[m])
                idx = sa.index.intersection(sb.index)
                sa, sb = sa.loc[idx].dropna(), sb.loc[idx].dropna()
                idx = sa.index.intersection(sb.index)
                sa, sb = sa.loc[idx].values, sb.loc[idx].values
                if len(sa) < 5 or np.all(sa == sb):
                    continue
                try:
                    stat, p = stats.wilcoxon(sa, sb, zero_method='wilcox')
                except Exception:
                    continue
                lo, hi = ci_mean_diff(sb, sa)
                d = cliffs_delta(sb, sa)
                rows.append({'Metric': m, 'A': a, 'B': b, 'N': len(sa),
                             'mean_diff_B_minus_A': round(float(np.mean(sb - sa)), 5),
                             'CI95_low': round(lo, 5), 'CI95_high': round(hi, 5),
                             'cliffs_delta': round(d, 4),
                             'magnitude': magnitude(d),
                             'cohens_d': round(cohens_d_paired(sb, sa), 4),
                             'p_raw': round(float(p), 5)})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    # Holm correction WITHIN each metric family, aligned by index so the
    # adjusted value always lands on the row it belongs to.
    t['p_holm'] = np.nan
    for m, g in t.groupby('Metric'):
        t.loc[g.index, 'p_holm'] = holm(g['p_raw'].values)
    t['p_holm'] = t['p_holm'].round(5)
    t['sig_holm'] = np.where(t['p_holm'] < 0.05, '*', 'ns')
    assert (t['p_holm'] >= t['p_raw'] - 1e-9).all(), \
        'Holm-adjusted p must never be smaller than the raw p'
    t = t.sort_values(['Metric', 'p_raw'])
    t.to_csv(os.path.join(outdir, 'ablation_wilcoxon.csv'), index=False)
    print('\n' + '=' * 78)
    print('T6/T7  ABLATION CONTRASTS (Wilcoxon + effect size + Holm)')
    print('=' * 78)
    print(t.to_string(index=False))
    return t


def latex_tables(t_methods, outdir):
    path = os.path.join(outdir, 'latex_tables.tex')
    with open(path, 'w') as f:
        f.write('% auto-generated -- paste into the manuscript\n')
        f.write(t_methods.to_latex(escape=False))
    print(f'\nLaTeX snippets written to {path}')


# ----------------------------------------------------------------------
def main(resdir, ref):
    outdir = os.path.join(resdir, 'analysis')
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(os.path.join(resdir, 'results.csv'))
    dpath = os.path.join(resdir, 'diagnostics.csv')
    dd = pd.read_csv(dpath) if os.path.exists(dpath) else None

    analyse_diagnostics(dd, outdir)
    report_degenerate(df, outdir)
    t = aggregate(df, outdir)
    analyse_calibration(df, resdir, outdir)
    friedman_nemenyi(df, outdir, ref=ref)
    ablation_tests(df, outdir)
    latex_tables(t, outdir)
    print(f'\nAll analysis artefacts in {outdir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--ref', default='EWDDBS (Entropy+Safe+Tomek)')
    a = ap.parse_args()
    main(a.dir, a.ref)
