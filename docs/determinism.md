# Determinism

## Why it is disabled

The neural topology guide runs on GPU with TensorFlow's deterministic-operations
mode disabled. This is a deliberate choice, not an oversight: the resulting
run-to-run variability is one of the quantities the study measures.

The audit reported in the article compares two executions of an identical seed
and configuration on the CTU-UHB cohort:

| | Comparator strategies | EW-DDBS variants |
|---|---|---|
| Strategies compared | 9 | 8 |
| Rows reproducing bitwise | 100.0% | 18.5% |
| Mean row-level absolute difference in F1 | 0 | 0.0378 |
| Maximum aggregate shift | 0 | 0.0106 |
| Implied per-execution aggregate SD | 0 | 0.0033 |

Because every non-neural component is deterministic given a fixed seed, the
comparison localises the entire discrepancy to nondeterministic kernels in the
guide.

## How to enable it

Add the following at the top of `cnn_topology_guide()` in `src/ewddbs_core.py`,
before the model is built:

```python
import os, random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()
```

Expect a training slowdown of roughly 10 to 30 per cent.

## What changes if you do

Enabling determinism will shift the EW-DDBS numbers slightly, because it selects
a different set of kernels rather than averaging over them. Your results will
then be exactly reproducible on the same hardware and library versions, but they
will no longer match the published values, and the determinism audit will
measure nothing.

If your goal is to check the published analysis, use the archived artefacts and
`scripts/verify_paper_numbers.py` rather than re-running with determinism on.
