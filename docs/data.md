# Data

Neither dataset is redistributed in this repository. Both are publicly
available from their custodians under their own terms.

## UCI Cardiotocography

Obtain `CTG.xls` from the UCI Machine Learning Repository and place it beside
`src/run_uci.py`, or in `src/data/`. The loader also attempts two download URLs
if the file is absent, and raises rather than falling back to synthetic data.

The analysis uses the 21-feature benchmark set, selected **by column name**
rather than by position:

```
LB, AC.1, FM.1, UC.1, DL.1, DS.1, DP.1,
ASTV, MSTV, ALTV, MLTV,
Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency
```

The deceleration and acceleration columns appear twice in the spreadsheet. The
plain names are raw episode counts; the `.1` duplicates are the per-second rates
that the standard benchmark uses.

### A row-range pitfall worth knowing about

The data occupy rows 0 to 2125 of the `Data` sheet. A one-off range such as
`iloc[1:2127]` drops the first record and picks up an empty trailing row, which
the missing-value mask then removes. The result is 2,125 records instead of
2,126, and nothing about the run appears to fail. `run_uci.py` asserts a final
count of 2,126 with the class distribution 1,655 / 295 / 176. If you compare
your numbers against another study, check which count that study used.

## CTU-UHB Intrapartum Cardiotocography Database

Downloaded automatically from PhysioNet via `wfdb` on first run
(`ctu-uhb-ctgdb/1.0.0`). Recordings are labelled acidaemic when the umbilical
artery pH is below 7.15.

Feature extraction takes roughly 25 minutes for 552 recordings and is cached in
`ctu_uhb_features62_cache.npz`. The cache stores a 62-column matrix; a cache
with any other width is rejected rather than used, so that a stale 30-feature
cache from an earlier pipeline cannot silently contaminate a run.

The 62 features comprise 22 descriptive statistics across both channels, 8
Daubechies `db4` three-level wavelet coefficients, 10 spectral descriptors
(VLF, LF and HF band power plus the 95% spectral edge frequency), 6 nonlinear
entropy estimators (sample, approximate and permutation entropy on both
channels), 11 CTG morphological descriptors, and 5 cross-channel FHR/UC coupling
features. Running `python src/ctg_features.py` verifies on a synthetic signal
that the extractor returns exactly 62 finite values.

No dimensionality reduction is applied by default, so synthesis occurs in the
named feature space on both cohorts. The `--pca` flag restores an earlier
PCA-based behaviour, but using it invalidates the input-space interpretability
argument.
