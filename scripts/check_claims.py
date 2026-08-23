"""
check_claims.py
===============
Recompute every quantitative claim listed in `paper_claims.json` from the
archived result files and fail loudly if any of them no longer holds.

`verify_paper_numbers.py` prints the numbers for a human to read. This script
asserts them, and exits non-zero on the first discrepancy, so it can run in
continuous integration and be cited as evidence rather than as an invitation to
squint at two documents side by side.

Usage
-----
    python scripts/check_claims.py --root results/
    python scripts/check_claims.py --root results/ --claims paper_claims.json

Exit codes
----------
    0   every claim that could be evaluated passed
    1   at least one claim failed
    2   the result directory or claims file is unusable
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

FOLD_KEY = ['Seed', 'Fold', 'Classifier']
ROW_KEY = ['Seed', 'Fold', 'Oversampling', 'Classifier']
DATASET_DIR = {'uci': 'results_uci_all', 'ctu': 'results_ctu_all'}
PERM_DIR = {'uci': ('results_uci', 'results_uci_perm'),
            'ctu': ('results_ctu', 'results_ctu_perm')}


class Skip(Exception):
    """Raised when a claim cannot be evaluated because its inputs are absent."""


def read(root, name):
    p = os.path.join(root, name, 'results.csv')
    if not os.path.exists(p):
        raise Skip(f'{name}/results.csv not found')
    return pd.read_csv(p)


def read_diag(root, name):
    p = os.path.join(root, name, 'diagnostics.csv')
    if not os.path.exists(p):
        raise Skip(f'{name}/diagnostics.csv not found')
    return pd.read_csv(p)


def contrasts(df, metric):
    base = df[df.Oversampling == 'Baseline'].set_index(FOLD_KEY)[metric]
    rows = []
    for m in df.Oversampling.unique():
        if m == 'Baseline':
            continue
        x = df[df.Oversampling == m].set_index(FOLD_KEY)[metric]
        j = pd.concat([base.rename('b'), x.rename('v')], axis=1).dropna()
        d = j['v'] - j['b']
        rows.append(dict(Strategy=m, delta=d.mean(), p=wilcoxon(d)[1]))
    r = pd.DataFrame(rows)
    r['p_holm'] = multipletests(r['p'], method='holm')[1]
    return r.set_index('Strategy')


def determinism_frames(root):
    """Pair the two independent executions of CTU-UHB seed 42.

    `reproduce_all.sh` writes the repeat execution to `results_ctu_run2`. Older
    archives named it `results_ctu_run1`, so both are accepted. Every quantity
    derived from this pair is symmetric in the two runs, so which directory is
    called first does not affect any reported value.
    """
    for repeat in ('results_ctu_run2', 'results_ctu_run1'):
        if os.path.exists(os.path.join(root, repeat, 'results.csv')):
            a = read(root, repeat)
            b = read(root, 'results_ctu')
            return a.merge(b, on=ROW_KEY, suffixes=('_1', '_2'))
    raise Skip('no repeat execution found '
               '(expected results_ctu_run2/ or results_ctu_run1/)')


def evaluate(claim, root, cache):
    kind = claim['kind']

    if kind in ('mean',):
        df = cache.setdefault(claim['dataset'],
                              read(root, DATASET_DIR[claim['dataset']]))
        return df.groupby('Oversampling')[claim['metric']].mean()[claim['strategy']]

    if kind == 'n_rows':
        return len(cache.setdefault(claim['dataset'],
                                    read(root, DATASET_DIR[claim['dataset']])))

    if kind == 'seeds':
        df = cache.setdefault(claim['dataset'],
                              read(root, DATASET_DIR[claim['dataset']]))
        return sorted(int(s) for s in df.Seed.unique())

    if kind in ('delta_vs_baseline', 'count_delta_negative_vs_baseline',
                'count_significant_vs_baseline'):
        df = cache.setdefault(claim['dataset'],
                              read(root, DATASET_DIR[claim['dataset']]))
        key = ('contrast', claim['dataset'], claim['metric'])
        r = cache.setdefault(key, contrasts(df, claim['metric']))
        if kind == 'delta_vs_baseline':
            return r.loc[claim['strategy'], 'delta']
        if kind == 'count_delta_negative_vs_baseline':
            return int((r.delta < 0).sum())
        return int((r.p_holm < 0.05).sum())

    if kind == 'count_beat_baseline':
        df = cache.setdefault(claim['dataset'],
                              read(root, DATASET_DIR[claim['dataset']]))
        g = df.groupby('Oversampling')[claim['metric']].mean()
        base, others = g['Baseline'], g.drop('Baseline')
        better = others < base if claim['metric'] == 'Brier' else others > base
        return int(better.sum())

    if kind == 'diagnostic':
        d = cache.setdefault(('diag', claim['dataset']),
                             read_diag(root, DATASET_DIR[claim['dataset']]))
        if claim['field'] in ('W_cv', 'P_ess_ratio'):
            d = d[~d.Method.str.contains('Uniform')]
        return d[claim['field']].mean()

    if kind in ('permutation_p', 'permutation_nsig'):
        o_name, p_name = PERM_DIR[claim['cohort']]
        m = cache.setdefault(('perm', claim['cohort']),
                             read(root, o_name).merge(read(root, p_name),
                                                      on=ROW_KEY,
                                                      suffixes=('_o', '_p')))
        if kind == 'permutation_p':
            d = m[claim['metric'] + '_p'] - m[claim['metric'] + '_o']
            return wilcoxon(d)[1]
        ps = []
        for _, g in m.groupby('Oversampling'):
            d = g['F1-Macro_p'].values - g['F1-Macro_o'].values
            ps.append(wilcoxon(d)[1] if np.any(d != 0) else 1.0)
        return int((multipletests(ps, method='holm')[1] < 0.05).sum())

    if kind == 'determinism_identical_pct':
        m = cache.setdefault('det', determinism_frames(root))
        same = (m['F1-Macro_2'] - m['F1-Macro_1']).abs() < 1e-12
        mask = m.Oversampling.str.startswith('EWDDBS')
        sub = same[mask] if claim['group'] == 'ewddbs' else same[~mask]
        return round(float(sub.mean() * 100), 1)

    if kind == 'determinism_per_run_sd':
        m = cache.setdefault('det', determinism_frames(root))
        a = m.groupby('Oversampling')['F1-Macro_1'].mean()
        b = m.groupby('Oversampling')['F1-Macro_2'].mean()
        ew = [i for i in a.index if i.startswith('EWDDBS')]
        return float((b[ew] - a[ew]).std(ddof=1) / np.sqrt(2))

    if kind == 'determinism_rank_shift':
        m = cache.setdefault('det', determinism_frames(root))
        a = m.groupby('Oversampling')['F1-Macro_1'].mean()
        b = m.groupby('Oversampling')['F1-Macro_2'].mean()
        return int((b.rank(ascending=False) - a.rank(ascending=False)).abs().max())

    if kind == 'permutation_over_noise':
        m = cache.setdefault('det', determinism_frames(root))
        a = m.groupby('Oversampling')['F1-Macro_1'].mean()
        b = m.groupby('Oversampling')['F1-Macro_2'].mean()
        ew = [i for i in a.index if i.startswith('EWDDBS')]
        noise = (b[ew] - a[ew]).abs().mean()
        o = read(root, 'results_ctu').groupby('Oversampling')['F1-Macro'].mean()
        p = read(root, 'results_ctu_perm').groupby('Oversampling')['F1-Macro'].mean()
        effect = (p[ew] - o[ew]).abs().mean()
        return float(effect / noise)

    raise Skip(f'unknown claim kind: {kind}')


def main(root, claims_path):
    with open(claims_path) as fh:
        spec = json.load(fh)
    claims = spec['claims']
    cache = {}
    passed = failed = skipped = 0
    failures = []

    print(f'Checking {len(claims)} claims against {root}\n')
    for c in claims:
        try:
            got = evaluate(c, root, cache)
        except Skip as exc:
            print(f'  SKIP  {c["id"]:26s} {exc}')
            skipped += 1
            continue
        except Exception as exc:            # noqa: BLE001
            print(f'  ERROR {c["id"]:26s} {type(exc).__name__}: {exc}')
            failed += 1
            failures.append(c['id'])
            continue

        exp = c['expected']
        if isinstance(exp, list):
            ok = list(got) == exp
            detail = f'{got} vs {exp}'
        else:
            tol = c.get('tol', 0)
            ok = abs(float(got) - float(exp)) <= tol
            detail = f'{float(got):.5f} vs {float(exp):.5f} (tol {tol})'

        if ok:
            print(f'  PASS  {c["id"]:26s} {c["where"]:22s} {detail}')
            passed += 1
        else:
            print(f'  FAIL  {c["id"]:26s} {c["where"]:22s} {detail}')
            failed += 1
            failures.append(c['id'])

    print(f'\n{passed} passed, {failed} failed, {skipped} skipped')
    if failed:
        print('\nFailed claims: ' + ', '.join(failures))
        print('The manuscript and the archived results no longer agree.')
        return 1
    if passed == 0:
        print('\nNo claim could be evaluated; check the --root layout.')
        return 2
    print('Every evaluated claim in the manuscript matches the archived results.')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='results')
    ap.add_argument('--claims', default=os.path.join(
        os.path.dirname(__file__), '..', 'paper_claims.json'))
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        sys.exit(f'error: {a.root} is not a directory')
    sys.exit(main(a.root, a.claims))
