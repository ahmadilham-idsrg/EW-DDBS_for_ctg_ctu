# EW-DDBS: Class-imbalance correction in cardiotocographic decision support

Code and reproduction materials for the article *Class-Imbalance Correction in
Cardiotocographic Decision Support: A Two-Cohort Evaluation with
Feature-Ordering and Determinism Controls*.

The study asks whether class-imbalance correction benefits cardiotocographic
(CTG) decision support once an uncorrected model is included as a control and
probability calibration is reported alongside discrimination, and whether the
margins this literature customarily reports exceed the variability of the
pipelines that produce them.

## Headline findings

| | Finding |
|---|---|
| Antepartum cohort (UCI, n = 2,126) | All sixteen correction strategies performed worse than leaving the data unmodified; twelve significantly after Holm correction |
| Intrapartum cohort (CTU-UHB, n = 552) | SVMSMOTE and KMeansSMOTE improved F1 significantly, yet the same two significantly worsened AUPRC and Brier score on identical folds |
| Both cohorts | The uncorrected model attained the best AUPRC, Brier score and minority-class calibration error |
| Feature-ordering control | Randomising the column order changed no outcome (0/17 significant on either cohort) |
| Determinism audit | Re-running identical code shifted aggregate F1 by up to 0.011 and ranks by up to eight positions |

The method under examination, Entropy-Weighted Dynamic Density-Based Sampling
(EW-DDBS), is used here as an instrumented test vehicle rather than as a
proposed product. Its weighting function is logged in full so that its null
result can be attributed to the mechanism rather than to the implementation.

## Repository layout

```
src/
  ctg_features.py        62-dimensional feature extractor for raw FHR/UC signals
  ewddbs_core.py         EW-DDBS resampling, metrics, classifiers, artefact writer
  run_uci.py             benchmark driver, UCI cohort
  run_ctu_uhb.py         benchmark driver, CTU-UHB cohort
  analyze_results.py     Friedman, Nemenyi, Wilcoxon, Holm, calibration, diagnostics
  merge_runs.py          merge result directories across seeds
notebooks/
  01_UCI_cohort.ipynb    end-to-end Colab notebook, UCI
  02_CTU_UHB_cohort.ipynb end-to-end Colab notebook, CTU-UHB
tests/
  test_ewddbs.py         21 unit tests on the properties the article's argument uses
scripts/
  reproduce_all.sh       full reproduction from scratch
  verify_paper_numbers.py regenerate every number in the article from archived results
  check_claims.py        assert those numbers against paper_claims.json, exit non-zero
  check_notebook_sync.py prove the notebooks embed the same code as src/
  smoke_test.py          two-minute end-to-end run on synthetic data, no GPU
  record_environment.py  write requirements-lock.txt from the live interpreter
paper_claims.json        every quantitative claim in the article, machine-checkable
data/MANIFEST.md         expected input and output shapes, and what the code enforces
docs/
  data.md                how to obtain the two datasets
  environment.md         the software and hardware used for the published runs
  determinism.md         why determinism was not enabled, and how to enable it
```

## Checks you can run in under three minutes

None of these needs a GPU, a dataset or the result archive. They run in
continuous integration on every push, and a reviewer can watch them pass on a
fork.

```bash
pip install -r requirements.txt
pytest -q tests/                        # 21 unit tests
python scripts/check_notebook_sync.py   # notebooks and src/ agree
python scripts/smoke_test.py            # pipeline runs end to end
```

The unit tests exist because the article defends a null result by asserting that
the weighting mechanism worked as designed. That assertion is worth only as much
as the evidence behind it, so the tests check the stated properties one by one:
that every minority class reaches the majority count, that the uniform control
yields CV(W) = 0 and an effective-sample-size ratio of 1 by construction, that
lowering the temperature sharpens the weights monotonically, that raising the
gating threshold removes more parents, that total gating falls back to uniform
sampling rather than failing, that the jitter scale tracks eta, that Tomek
cleaning never adds samples, and that the ablation grid is a complete 2x2x2
factorial in which every entropy variant has an entropy-free counterpart.

The synchronisation check and the smoke test each found a real defect on first
run: a stale copy of `run_uci.py` embedded in the UCI notebook, and a crash in
`analyze_results.py` when no reliability bins were produced. Both are fixed.

## Verifying the article without re-running anything

The published runs take roughly six hours of GPU time. You do not need to repeat
them to check the article. Download the artefact archive from Zenodo
(DOI to be inserted), extract it, and run:

```bash
pip install -r requirements.txt
python scripts/verify_paper_numbers.py --root results/
```

The script reads only the archived CSV files and prints every quantitative claim
in the article next to the table or section that cites it: the two aggregate
tables, the contrasts against the uncorrected baseline with Holm correction and
Cliff's delta, the calibration contrasts for the two improving oversamplers, the
entropy-axis ablation, the weighting diagnostics, both permutation controls with
their internal validity check, the determinism audit, and the degeneracy
summary. No value in that report is hard-coded.

That script prints numbers for a human to compare. To have the comparison made
for you, run:

```bash
python scripts/check_claims.py --root results/
```

`paper_claims.json` lists every quantitative claim in the article together with
the tolerance implied by the precision at which the article quotes it, so a
value printed to four decimal places is checked to 5e-5. The checker recomputes
each one from the archived files and exits non-zero on the first disagreement.
If the data change, that file must change with them, which is the point.

Both scripts expect the extracted archive to hold these directories:

```
results/
  results_uci_all/    results_ctu_all/     three merged seeds per cohort
  results_uci/        results_ctu/         seed 42 alone, natural column order
  results_uci_perm/   results_ctu_perm/    seed 42, permuted column order
  results_ctu_run2/                        a second, independent execution of
                                           CTU-UHB seed 42, used only for the
                                           determinism audit
```

A directory that is absent produces `SKIP` rather than a false pass, so a
partial archive reports honestly on what it could not check.

## Reproducing from scratch

```bash
pip install -r requirements.txt
bash scripts/reproduce_all.sh
```

Each seed runs as a separate invocation, so a dropped session costs one run
rather than the whole sequence. The CTU-UHB driver caches the extracted feature
matrix in `ctu_uhb_features62_cache.npz`; keep that file to avoid re-downloading
552 recordings from PhysioNet on every run.

To run in Google Colab instead, open the notebooks in `notebooks/`. They write
the modules to disk, verify the feature extractor, run the benchmark, merge the
seeds and download the archives. Select a T4 GPU under **Runtime > Change
runtime type**.

## An important caveat about reproducibility

The results in `src/` are **not bitwise reproducible for EW-DDBS**, and this is
deliberate.

Every component of the pipeline except the neural topology guide is
deterministic given a fixed seed. The guide runs on GPU with TensorFlow's
deterministic-operations mode disabled, because the run-to-run variability that
this introduces is one of the objects of study. Re-running an identical seed
therefore reproduces all nine comparator strategies bitwise but only about
18.5% of EW-DDBS rows, with a per-execution standard deviation of roughly 0.0033
in aggregate macro-averaged F1.

Two consequences follow:

1. Reported aggregates absorb this variability through replication across three
   seeds, and the released artefacts allow the analysis to be checked from the
   recorded outputs.
2. If you need exact reproduction, enable determinism as described in
   `docs/determinism.md`. Doing so will change the EW-DDBS numbers slightly and
   will remove the variability that the determinism audit measures.

## Data

Neither dataset is redistributed here. `docs/data.md` gives the source for each
and the exact preprocessing applied, including a note on a row-range pitfall in
the UCI spreadsheet that silently yields 2,125 records instead of 2,126.

## Citing

See `CITATION.cff`. Please cite the article rather than this repository alone.

## Licence

MIT for the code in this repository. The datasets remain under the terms set by
their custodians.
