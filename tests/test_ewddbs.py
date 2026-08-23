"""
Unit tests for the EW-DDBS resampler and the metric layer.

The manuscript defends a null result by asserting that the weighting mechanism
"operated as designed". That assertion is only as good as the evidence behind
it, so these tests check the properties the manuscript relies on, one by one,
against the equations it states. They run in a few seconds and require no GPU,
because the topology guide is injected as an argument rather than trained.

Run with:  pytest -q tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import ewddbs_core as C  # noqa: E402


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def imbalanced():
    """A three-class problem with the UCI-like ratio 1655 : 295 : 176,
    scaled down so the tests stay fast."""
    rng = np.random.default_rng(0)
    n = [160, 30, 18]
    X = np.vstack([rng.normal(loc=i, scale=1.0, size=(n[i], 6)) for i in range(3)])
    y = np.concatenate([np.full(n[i], i) for i in range(3)])
    entropy = rng.uniform(0.0, 1.0, size=len(y))
    latent = rng.normal(size=(len(y), 4))
    return X.astype(np.float64), y, entropy, latent


# ----------------------------------------------------------------------
# balancing behaviour
# ----------------------------------------------------------------------
def test_every_minority_class_is_raised_to_the_majority_count(imbalanced):
    X, y, e, z = imbalanced
    Xr, yr, _ = C.ew_ddbs_resample(X, y, e, z, apply_tomek=False,
                                   rng=np.random.default_rng(1))
    counts = np.bincount(yr)
    assert counts.min() == counts.max(), 'classes are not balanced after resampling'
    assert counts.max() == np.bincount(y).max(), 'majority class was altered'


def test_original_samples_are_preserved(imbalanced):
    X, y, e, z = imbalanced
    Xr, yr, _ = C.ew_ddbs_resample(X, y, e, z, apply_tomek=False,
                                   rng=np.random.default_rng(1))
    assert np.allclose(Xr[:len(X)], X), 'original rows were modified'
    assert np.array_equal(yr[:len(y)], y), 'original labels were modified'


def test_dimensionality_is_unchanged(imbalanced):
    X, y, e, z = imbalanced
    Xr, _, _ = C.ew_ddbs_resample(X, y, e, z, apply_tomek=False,
                                  rng=np.random.default_rng(1))
    assert Xr.shape[1] == X.shape[1], 'synthesis did not occur in the input space'


# ----------------------------------------------------------------------
# the weighting function, Eq. (4)
# ----------------------------------------------------------------------
def test_uniform_variant_produces_degenerate_diagnostics(imbalanced):
    """With both weighting terms disabled, CV(W) must be 0 and the
    effective-sample-size ratio must be 1. Table 8 of the manuscript relies on
    this being true by construction."""
    X, y, e, z = imbalanced
    _, _, diag = C.ew_ddbs_resample(X, y, e, z, use_entropy=False, use_safe=False,
                                    apply_tomek=False, rng=np.random.default_rng(2))
    for cls, d in diag.items():
        if cls == 'tomek_removed':
            continue
        assert d['W_cv'] == pytest.approx(0.0, abs=1e-9)
        assert d['P_ess_ratio'] == pytest.approx(1.0, abs=1e-9)


def test_weighted_variants_are_not_degenerate(imbalanced):
    """The manuscript's defence against 'the implementation was inert' requires
    a measurably non-uniform weight distribution."""
    X, y, e, z = imbalanced
    _, _, diag = C.ew_ddbs_resample(X, y, e, z, use_entropy=True, use_safe=True,
                                    apply_tomek=False, rng=np.random.default_rng(2))
    cvs = [d['W_cv'] for k, d in diag.items() if k != 'tomek_removed']
    assert min(cvs) > 0.0, 'weighting collapsed to uniform'


def test_temperature_monotonically_sharpens_the_weights(imbalanced):
    """exp(H/T) must concentrate more as T falls."""
    X, y, e, z = imbalanced
    cv = {}
    for T in (2.0, 0.5, 0.1):
        _, _, d = C.ew_ddbs_resample(X, y, e, z, use_entropy=True, use_safe=False,
                                     apply_tomek=False, T=T,
                                     rng=np.random.default_rng(3))
        cv[T] = np.mean([v['W_cv'] for k, v in d.items() if k != 'tomek_removed'])
    assert cv[0.1] > cv[0.5] > cv[2.0], f'temperature has no monotone effect: {cv}'


def test_gating_threshold_removes_more_parents_as_tau_rises(imbalanced):
    X, y, e, z = imbalanced
    gated = {}
    for tau in (0.0, 0.4, 0.8):
        _, _, d = C.ew_ddbs_resample(X, y, e, z, use_entropy=False, use_safe=True,
                                     apply_tomek=False, tau=tau,
                                     rng=np.random.default_rng(4))
        gated[tau] = sum(v['n_gated_by_tau'] for k, v in d.items()
                         if k != 'tomek_removed')
    assert gated[0.0] <= gated[0.4] <= gated[0.8], f'gating is not monotone: {gated}'


def test_total_gating_falls_back_to_uniform_rather_than_failing(imbalanced):
    """With tau above 1 every candidate is gated out. The procedure must still
    return a balanced set, as Algorithm 1 line 16 specifies."""
    X, y, e, z = imbalanced
    Xr, yr, _ = C.ew_ddbs_resample(X, y, e, z, use_safe=True, apply_tomek=False,
                                   tau=1.5, rng=np.random.default_rng(5))
    counts = np.bincount(yr)
    assert counts.min() == counts.max(), 'fallback did not produce a balanced set'


# ----------------------------------------------------------------------
# synthesis, Eq. (5) and (6)
# ----------------------------------------------------------------------
def test_jitter_scale_follows_eta(imbalanced):
    """delta ~ N(0, eta * diag(Var_minority)); raising eta must widen the
    spread of the synthetic samples."""
    X, y, e, z = imbalanced
    spread = {}
    for eta in (0.0, 0.05, 1.0):
        Xr, yr, _ = C.ew_ddbs_resample(X, y, e, z, apply_tomek=False, eta=eta,
                                       rng=np.random.default_rng(6))
        new = Xr[len(X):][yr[len(y):] == 2]
        spread[eta] = new.std(axis=0).mean()
    assert spread[0.0] < spread[0.05] < spread[1.0], f'eta has no effect: {spread}'


def test_zero_jitter_keeps_synthetic_points_on_the_segment(imbalanced):
    """With eta = 0, every synthetic point must lie inside the bounding box of
    its own class, because it is a convex combination of two members of it."""
    X, y, e, z = imbalanced
    Xr, yr, _ = C.ew_ddbs_resample(X, y, e, z, apply_tomek=False, eta=0.0,
                                   rng=np.random.default_rng(7))
    for cls in (1, 2):
        orig = X[y == cls]
        new = Xr[len(X):][yr[len(y):] == cls]
        assert (new >= orig.min(axis=0) - 1e-9).all()
        assert (new <= orig.max(axis=0) + 1e-9).all()


def test_resampling_is_deterministic_for_a_fixed_generator(imbalanced):
    """Given the same entropy and latent inputs, the resampler itself contains
    no hidden randomness. This is what allows the manuscript to attribute
    run-to-run variability entirely to the neural guide."""
    X, y, e, z = imbalanced
    a = C.ew_ddbs_resample(X, y, e, z, rng=np.random.default_rng(11))[0]
    b = C.ew_ddbs_resample(X, y, e, z, rng=np.random.default_rng(11))[0]
    assert np.array_equal(a, b), 'resampler is not reproducible at fixed seed'


def test_different_generators_give_different_synthetic_points(imbalanced):
    X, y, e, z = imbalanced
    a = C.ew_ddbs_resample(X, y, e, z, rng=np.random.default_rng(11))[0]
    b = C.ew_ddbs_resample(X, y, e, z, rng=np.random.default_rng(12))[0]
    assert not np.array_equal(a, b), 'the generator argument is being ignored'


def test_tomek_cleaning_never_adds_samples(imbalanced):
    X, y, e, z = imbalanced
    n_off = len(C.ew_ddbs_resample(X, y, e, z, apply_tomek=False,
                                   rng=np.random.default_rng(8))[1])
    n_on = len(C.ew_ddbs_resample(X, y, e, z, apply_tomek=True,
                                  rng=np.random.default_rng(8))[1])
    assert n_on <= n_off, 'Tomek cleaning increased the sample count'


# ----------------------------------------------------------------------
# ablation design
# ----------------------------------------------------------------------
def test_ablation_grid_is_a_complete_two_by_two_by_two_design():
    """Table 1 of the manuscript claims a full factorial design. If it were not
    complete, the entropy axis would not be identifiable."""
    grid = C.ablation_grid(True)
    assert len(grid) == 8
    combos = set(grid.values())
    assert len(combos) == 8, 'the eight variants are not distinct'
    expected = {(a, b, c) for a in (False, True)
                for b in (False, True) for c in (False, True)}
    assert combos == expected, 'the factorial design has gaps'


def test_a_uniform_control_exists_for_every_entropy_variant():
    grid = C.ablation_grid(True)
    for name, (ent, safe, tomek) in grid.items():
        if ent:
            assert (False, safe, tomek) in set(grid.values()), (
                f'{name} has no entropy-free counterpart, '
                'so its entropy contribution is unidentifiable')


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_g_mean_vanishes_when_a_class_is_never_recalled():
    """The implementation adds 1e-12 inside the logarithm to keep the geometric
    mean finite, so a fully collapsed prediction yields 1e-12 rather than an
    exact zero. The manuscript reports this as 0 at four decimal places, which
    is correct; this test pins the underlying value so that the rounding stays
    honest if the epsilon is ever changed."""
    y = np.array([0, 0, 1, 1])
    value = C.g_mean(y, np.array([0, 0, 0, 0]))
    assert value < 1e-9, f'a collapsed prediction should vanish, got {value}'
    assert round(value, 4) == 0.0


def test_g_mean_is_one_for_a_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    assert C.g_mean(y, y) == pytest.approx(1.0)


def test_degeneracy_flag_fires_on_single_class_predictions():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.zeros_like(y)
    proba = np.full((6, 3), 1 / 3)
    m = C.evaluate(y, pred, proba, np.array([0, 1, 2]))
    assert m['degenerate'] is True
    assert m['n_pred_classes'] == 1


def test_degeneracy_flag_stays_off_for_a_healthy_prediction():
    y = np.array([0, 0, 1, 1, 2, 2])
    proba = np.full((6, 3), 1 / 3)
    m = C.evaluate(y, y, proba, np.array([0, 1, 2]))
    assert m['degenerate'] is False
    assert m['n_pred_classes'] == 3


def test_brier_score_bounds():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    worst = np.roll(np.eye(3), 1, axis=1)
    classes = np.array([0, 1, 2])
    assert C.multiclass_brier(y, perfect, classes) == pytest.approx(0.0, abs=1e-9)
    assert C.multiclass_brier(y, worst, classes) > 1.0


def test_prior_correction_renormalises_and_preserves_binary_ranking():
    """On a binary task, dividing by a constant prior is monotone, so AUC and
    AUPRC must be unchanged. The manuscript uses this equivalence as an
    implementation check in Table 4."""
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    prior = np.array([0.8, 0.2])
    out = C.prior_correction(proba, prior)
    assert np.allclose(out.sum(axis=1), 1.0), 'posteriors do not sum to one'
    assert np.array_equal(np.argsort(out[:, 1]), np.argsort(proba[:, 1])), \
        'prior correction changed the ranking on a binary task'
