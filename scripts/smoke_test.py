"""
smoke_test.py
=============
A two-minute end-to-end exercise of the pipeline on synthetic data, requiring
no dataset download and no GPU.

Its purpose is to let a reviewer confirm that the code runs and produces
well-formed artefacts before deciding whether to spend six hours reproducing
the published results. It exercises resampling, the ablation grid, the metric
layer, the artefact writer and the analysis script, but it uses a random
entropy vector in place of the neural topology guide, so its numbers carry no
scientific meaning.

Usage
-----
    python scripts/smoke_test.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

import ewddbs_core as C  # noqa: E402


def synthetic(n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [200, 40, 25][:n_classes]
    X = np.vstack([rng.normal(i * 1.4, 1.0, size=(s, 8))
                   for i, s in enumerate(sizes)])
    y = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    return X.astype(np.float64), y


def main():
    print('EW-DDBS smoke test (synthetic data, no GPU, no downloads)\n')
    X, y = synthetic()
    rng = np.random.default_rng(1)
    entropy = rng.uniform(0, 1, size=len(y))
    latent = rng.normal(size=(len(y), 6))
    classes = np.unique(y)
    print(f'  synthetic cohort: {X.shape[0]} samples, {X.shape[1]} features, '
          f'class counts {np.bincount(y).tolist()}')

    outdir = tempfile.mkdtemp(prefix='ewddbs_smoke_')
    try:
        writer = C.ArtefactWriter(outdir)
        grid = C.ablation_grid(True)
        print(f'  ablation grid: {len(grid)} variants')

        for name, (ue, us, tk) in grid.items():
            Xr, yr, diag = C.ew_ddbs_resample(
                X, y, entropy, latent, use_entropy=ue, use_safe=us,
                apply_tomek=tk, rng=np.random.default_rng(7))
            assert Xr.shape[1] == X.shape[1]
            writer.add_diag(1, 1, name, diag)

            # a single cheap classifier keeps the smoke test fast
            clf = C.fit_classifier('LogReg', Xr, yr, seed=1)
            pred = clf.predict(X)
            proba, calibratable = C.predict_proba_safe(clf, X, len(classes))
            met = C.evaluate(y, pred, proba, classes)
            writer.add_result(Seed=1, Fold=1, Oversampling=name,
                              Classifier='LogReg', Calibratable=calibratable,
                              **met)
            print(f'    {name:30s} n={len(yr):4d}  '
                  f'F1={met["F1-Macro"]:.3f}  degenerate={met["degenerate"]}')

        writer.flush_fold(1, 1, y)
        df, dd = writer.close()
        assert len(df) == len(grid), 'result rows do not match the grid size'
        assert os.path.exists(os.path.join(outdir, 'results.csv'))
        assert os.path.exists(os.path.join(outdir, 'diagnostics.csv'))
        print(f'\n  artefacts written: {len(df)} result rows, '
              f'{len(dd)} diagnostic rows')

        r = subprocess.run(
            [sys.executable, os.path.join(SRC, 'analyze_results.py'),
             '--dir', outdir, '--ref', 'EWDDBS (Entropy+Safe+Tomek)'],
            capture_output=True, text=True)
        if r.returncode != 0:
            print('\n  analyze_results.py failed:')
            print(r.stderr[-1500:])
            return 1
        print('  analyze_results.py completed without error')
        print('\nSmoke test passed. The pipeline runs end to end.')
        print('These numbers are meaningless: the topology guide was replaced '
              'by random noise.')
        return 0
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
