# Environment

## Hardware used for the published runs

Single NVIDIA Tesla T4 GPU (Google Colab). All seeds and both cohorts ran on the
same accelerator type, so that hardware variation does not confound the
multi-seed comparison.

Approximate wall-clock time per run: 40 minutes for one UCI seed (5 folds),
85 minutes for one CTU-UHB seed (10 folds) with the feature cache present, plus
25 minutes for the first CTU-UHB feature extraction.

## Software versions

Every run now records its own environment. `ewddbs_core.ResultWriter.close()`
writes `environment.json` into the result directory, holding the interpreter
version, the platform string, the version of each analysis package, the list of
visible GPU devices and the value of `TF_DETERMINISTIC_OPS`. A result archive
therefore carries the environment that produced it, and no one has to remember
it afterwards.

The runs published with the article predate that addition, so their archives do
not contain `environment.json`. Filling in the table below is an outstanding
task for the authors, and it must be done by measurement rather than by recall:

```bash
python scripts/record_environment.py
```

The script writes `requirements-lock.txt` and prints a Markdown table. Paste
that table here, replacing this paragraph and the placeholder below. Do not type
version numbers from memory. A version string nobody measured looks exactly like
evidence and is not.

```
python           x.y.z    <- not yet recorded
tensorflow       x.y.z
scikit-learn     x.y.z
imbalanced-learn x.y.z
xgboost          x.y.z
numpy            x.y.z
pandas           x.y.z
scipy            x.y.z
statsmodels      x.y.z
PyWavelets       x.y.z
wfdb             x.y.z
```

Until that table is filled in, the article should not claim that exact package
versions accompany the release. Either record them or drop the claim.

## A note on numpy 2

`ctg_features.py` uses `math.factorial` rather than `np.math.factorial`, which
numpy 2 removed. If you port this code to an older stack, that call is the one
place where the two differ.
