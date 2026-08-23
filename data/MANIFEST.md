# Data manifest

This directory holds no data. Neither cohort is redistributed here, for the
reasons in `docs/data.md`. What follows is the description a reader needs in
order to confirm that the files they obtained are the files this study used.

## What the drivers enforce at run time

These are not documentation. They are assertions in the code, and a mismatch
stops or flags the run rather than producing a quiet near-miss.

| Cohort | Check | Enforced in |
|---|---|---|
| UCI | feature matrix has exactly 21 columns | `run_uci.py`, `assert` |
| UCI | labels are exactly `{0, 1, 2}` | `run_uci.py`, `assert` |
| UCI | 2,126 rows with class counts 1,655 / 295 / 176 | `run_uci.py`, warning |
| CTU-UHB | feature cache has exactly 62 columns | `run_ctu_uhb.py`, raises |

The UCI row count is a warning rather than an assertion because the loader
accepts a legitimately filtered subset; the warning names the expected split so
that an accidental 2,125-row read, described in `docs/data.md`, is visible in
the log instead of silent.

## Expected inputs

### UCI Cardiotocography

| | |
|---|---|
| File | `CTG.xls` |
| Source | UCI Machine Learning Repository, Cardiotocography data set |
| Sheet used | `Data` |
| Rows used | 0 to 2125 |
| Shape after preparation | (2126, 21) |
| Label column | `NSP`, remapped to 0 / 1 / 2 |
| Class counts | 1655 / 295 / 176 |
| Placement | beside `src/run_uci.py`, or in `src/data/` |

### CTU-UHB Intrapartum Cardiotocography Database

| | |
|---|---|
| Source | PhysioNet `ctu-uhb-ctgdb/1.0.0`, fetched by `wfdb` |
| Records | 552 |
| Cache file | `ctu_uhb_features62_cache.npz` |
| Cache contents | `X` of shape (552, 62), `y` of shape (552,) |
| Label rule | acidaemic when umbilical artery pH < 7.15 |
| Feature names | `ctg_features.FEATURE_NAMES`, 62 entries |

## Expected outputs

Each single-seed run writes one directory. The row counts below are exact and
are the ones `scripts/check_claims.py` asserts after merging.

| Directory | Rows | Composition |
|---|---|---|
| one UCI seed | 1,275 | 5 folds x 15 classifiers x 17 strategies |
| one CTU-UHB seed | 2,550 | 10 folds x 15 classifiers x 17 strategies |
| `results_uci_all` (3 seeds) | 3,825 | seeds 7, 42, 2024 |
| `results_ctu_all` (3 seeds) | 7,650 | seeds 7, 42, 2024 |

Alongside `results.csv`, every run directory contains `diagnostics.csv`,
`config.json` (the four EW-DDBS hyperparameters) and `environment.json`
(interpreter, platform, package versions, visible GPUs, and whether
deterministic operations were enabled).

## Checksums

No checksum is listed here, and inventing one would be worse than leaving it
blank. The two source files are fetched by the reader from their custodians,
who may repackage them, and no checksum was recorded when the published runs
executed. Verify instead against the shapes and class counts above, which the
drivers check on every run.

For the artefact archive released on Zenodo, the checksum published on the
Zenodo record is authoritative. Compute yours with:

```bash
sha256sum ewddbs-ctg-results.zip
```
