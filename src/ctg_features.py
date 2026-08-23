"""
ctg_features.py
===============
62-dimensional feature extractor for raw CTU-UHB cardiotocography signals.

Implements EXACTLY the feature families described in Contribution (2) of the
manuscript:

  (a) basic descriptive statistics                       -> 22 (11 x 2 channels)
  (b) wavelet decomposition (Daubechies db4, 3 levels)   ->  8 ( 4 x 2 channels)
  (c) multi-band spectral analysis (VLF/LF/HF + SEF95)   -> 10 ( 5 x 2 channels)
  (d) nonlinear entropy (Sample, Approximate, Permutation)->  6 ( 3 x 2 channels)
  (e) CTG-specific morphological descriptors             -> 11 (8 FHR + 3 UC)
  (f) cross-channel FHR-UC coupling                      ->  5
                                                        -------
                                                    TOTAL   62

Signals are sampled at 4 Hz in CTU-UHB.

Author note: every returned value is finite; NaN/Inf are replaced by 0.0 so the
downstream imputer never has to guess.
"""

import math

import numpy as np
import pywt
from scipy import stats, signal as sps

FS = 4.0  # CTU-UHB sampling rate (Hz)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _clean(x, lo=None, hi=None, max_gap_s=15.0, fs=FS):
    """Remove non-physiological values and linearly interpolate short gaps."""
    x = np.asarray(x, dtype=float).copy()
    if lo is not None:
        x[x < lo] = np.nan
    if hi is not None:
        x[x > hi] = np.nan
    x[x == 0] = np.nan  # CTU-UHB codes signal loss as 0

    n = len(x)
    if n == 0:
        return x
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() < 10:
        return np.full(n, np.nan)

    # interpolate gaps shorter than max_gap_s, leave long gaps as NaN
    xi = np.interp(idx, idx[good], x[good])
    max_gap = int(max_gap_s * fs)
    isnan = np.isnan(x)
    if isnan.any():
        # find runs of NaN
        d = np.diff(np.concatenate([[0], isnan.view(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) > max_gap:
                xi[s:e] = np.nan
    return xi


def _finite(vals):
    out = []
    for v in vals:
        v = float(v) if v is not None else 0.0
        out.append(v if np.isfinite(v) else 0.0)
    return out


def _downsample(x, target_n=2000):
    """Decimate for O(N^2) entropy estimators."""
    x = x[np.isfinite(x)]
    if len(x) <= target_n:
        return x
    step = int(np.ceil(len(x) / target_n))
    return x[::step]


# ----------------------------------------------------------------------
# (a) descriptive statistics -- 11 per channel
# ----------------------------------------------------------------------
def descriptive_features(x):
    v = x[np.isfinite(x)]
    if len(v) < 10:
        return _finite([0] * 11)
    q25, q75 = np.percentile(v, [25, 75])
    hist, _ = np.histogram(v, bins=10, density=True)
    hist = hist[hist > 0]
    hist_ent = -np.sum(hist * np.log(hist + 1e-12))
    return _finite([
        np.mean(v), np.std(v), np.median(v), np.min(v), np.max(v),
        np.max(v) - np.min(v), q75 - q25, np.percentile(v, 90),
        stats.skew(v), stats.kurtosis(v), hist_ent,
    ])


# ----------------------------------------------------------------------
# (b) wavelet db4, 3 levels -- 4 per channel (log energy of cA3,cD3,cD2,cD1)
# ----------------------------------------------------------------------
def wavelet_features(x, wavelet='db4', level=3):
    v = x[np.isfinite(x)]
    if len(v) < 2 ** (level + 2):
        return _finite([0] * 4)
    try:
        coeffs = pywt.wavedec(v, wavelet, level=level)   # [cA3, cD3, cD2, cD1]
    except Exception:
        return _finite([0] * 4)
    return _finite([np.log10(np.sum(c ** 2) + 1e-12) for c in coeffs])


# ----------------------------------------------------------------------
# (c) multi-band spectral -- 5 per channel
# ----------------------------------------------------------------------
def spectral_features(x, fs=FS):
    v = x[np.isfinite(x)]
    if len(v) < 256:
        return _finite([0] * 5)
    nper = int(min(1024, len(v)))
    f, psd = sps.welch(v - np.mean(v), fs=fs, nperseg=nper)
    if len(f) < 2:
        return _finite([0] * 5)
    df = f[1] - f[0]

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.sum(psd[m]) * df)

    vlf = band(0.0, 0.03)     # very low frequency
    lf = band(0.03, 0.15)     # low frequency
    hf = band(0.15, 0.50)     # high frequency
    lf_hf = lf / (hf + 1e-12)

    # spectral edge frequency 95%
    total = np.cumsum(psd) * df
    if total[-1] <= 0:
        sef95 = 0.0
    else:
        sef95 = float(f[np.searchsorted(total, 0.95 * total[-1])])

    return _finite([np.log10(vlf + 1e-12), np.log10(lf + 1e-12),
                    np.log10(hf + 1e-12), lf_hf, sef95])


# ----------------------------------------------------------------------
# (d) nonlinear entropies -- 3 per channel
# ----------------------------------------------------------------------
def _phi(x, m, r):
    n = len(x)
    if n <= m + 1:
        return None
    emb = np.lib.stride_tricks.sliding_window_view(x, m)[: n - m + 1]
    # chunked Chebyshev distance to keep memory bounded
    counts = np.zeros(len(emb))
    chunk = 512
    for i in range(0, len(emb), chunk):
        d = np.max(np.abs(emb[i:i + chunk, None, :] - emb[None, :, :]), axis=2)
        counts[i:i + chunk] = np.sum(d <= r, axis=1)
    return counts


def approximate_entropy(x, m=2, r_factor=0.2):
    x = _downsample(x, 1200)
    if len(x) < 50:
        return 0.0
    r = r_factor * np.std(x)
    if r <= 0:
        return 0.0
    c_m = _phi(x, m, r)
    c_m1 = _phi(x, m + 1, r)
    if c_m is None or c_m1 is None:
        return 0.0
    phi_m = np.mean(np.log(c_m / len(c_m) + 1e-12))
    phi_m1 = np.mean(np.log(c_m1 / len(c_m1) + 1e-12))
    return float(phi_m - phi_m1)


def sample_entropy(x, m=2, r_factor=0.2):
    x = _downsample(x, 1200)
    if len(x) < 50:
        return 0.0
    r = r_factor * np.std(x)
    if r <= 0:
        return 0.0
    c_m = _phi(x, m, r)
    c_m1 = _phi(x, m + 1, r)
    if c_m is None or c_m1 is None:
        return 0.0
    # exclude self-matches
    A = np.sum(c_m1 - 1)
    B = np.sum(c_m[: len(c_m1)] - 1)
    if A <= 0 or B <= 0:
        return 0.0
    return float(-np.log(A / B))


def permutation_entropy(x, order=3, delay=1, normalise=True):
    x = _downsample(x, 5000)
    n = len(x)
    if n < order * delay + 1:
        return 0.0
    emb = np.array([x[i:i + (order - 1) * delay + 1:delay]
                    for i in range(n - (order - 1) * delay)])
    patterns = np.argsort(emb, axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    pe = -np.sum(p * np.log2(p))
    if normalise:
        pe /= np.log2(math.factorial(order))
    return float(pe)


def entropy_features(x):
    return _finite([sample_entropy(x), approximate_entropy(x),
                    permutation_entropy(x)])


# ----------------------------------------------------------------------
# (e) CTG-specific morphology
# ----------------------------------------------------------------------
def fhr_morphology(fhr, fs=FS):
    """8 features: baseline, STV, LTV, %abnormal STV, accel, decel,
    prolonged decel, baseline drift."""
    v = fhr.copy()
    good = np.isfinite(v)
    if good.sum() < int(60 * fs):
        return _finite([0] * 8)
    vv = v[good]

    # baseline = mode of 5-bpm-binned histogram (clinical convention)
    bins = np.arange(np.floor(vv.min()), np.ceil(vv.max()) + 5, 5)
    if len(bins) < 2:
        baseline = float(np.median(vv))
    else:
        h, edges = np.histogram(vv, bins=bins)
        baseline = float((edges[np.argmax(h)] + edges[np.argmax(h) + 1]) / 2)

    # short-term variation: mean |diff| over 3.75 s epochs (Dawes-Redman style)
    ep = int(3.75 * fs)
    n_ep = len(vv) // ep
    if n_ep >= 2:
        epochs = vv[:n_ep * ep].reshape(n_ep, ep).mean(axis=1)
        stv = float(np.mean(np.abs(np.diff(epochs))))
        # long-term variation: mean range per 1-min window
        per_min = int(60 * fs / ep)
        if n_ep >= per_min and per_min > 0:
            nw = n_ep // per_min
            w = epochs[:nw * per_min].reshape(nw, per_min)
            ltv = float(np.mean(w.max(axis=1) - w.min(axis=1)))
        else:
            ltv = float(np.ptp(epochs))
        pct_abn_stv = float(np.mean(np.abs(np.diff(epochs)) < 1.0) * 100)
    else:
        stv, ltv, pct_abn_stv = 0.0, 0.0, 0.0

    # accelerations: >= +15 bpm above baseline for >= 15 s
    # decelerations : <= -15 bpm below baseline for >= 15 s
    # prolonged dec.: <= -15 bpm for >= 120 s
    def _episodes(mask, min_s):
        m = mask.astype(np.int8)
        d = np.diff(np.concatenate([[0], m, [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        dur = (en - st) / fs
        return int(np.sum(dur >= min_s))

    above = np.nan_to_num(v, nan=baseline) >= baseline + 15
    below = np.nan_to_num(v, nan=baseline) <= baseline - 15
    accel = _episodes(above, 15)
    decel = _episodes(below, 15)
    prol = _episodes(below, 120)

    # baseline drift: slope of a linear fit over the whole trace (bpm/hour)
    t = np.arange(len(v))[good] / fs / 3600.0
    if len(t) > 2 and np.ptp(t) > 0:
        drift = float(np.polyfit(t, vv, 1)[0])
    else:
        drift = 0.0

    return _finite([baseline, stv, ltv, pct_abn_stv,
                    accel, decel, prol, drift])


def uc_morphology(uc, fs=FS):
    """3 features: contraction count, mean amplitude, mean duration."""
    v = uc[np.isfinite(uc)]
    if len(v) < int(60 * fs):
        return _finite([0] * 3)
    base = np.percentile(v, 25)
    thr = base + 0.5 * (np.percentile(v, 90) - base)
    m = (np.nan_to_num(uc, nan=base) > thr).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    dur = (en - st) / fs
    keep = dur >= 30.0                      # a contraction lasts >= 30 s
    if keep.sum() == 0:
        return _finite([0, 0, 0])
    amps = [np.nanmax(uc[s:e]) - base for s, e in zip(st[keep], en[keep])]
    return _finite([int(keep.sum()), np.mean(amps), np.mean(dur[keep])])


# ----------------------------------------------------------------------
# (f) cross-channel FHR-UC coupling -- 5
# ----------------------------------------------------------------------
def coupling_features(fhr, uc, fs=FS):
    a, b = fhr.copy(), uc.copy()
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < int(120 * fs):
        return _finite([0] * 5)
    a, b = a[good], b[good]
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    pearson = float(np.corrcoef(a, b)[0, 1])

    # cross-correlation within +/- 120 s
    maxlag = int(120 * fs)
    n = len(a)
    lags = np.arange(-maxlag, maxlag + 1)
    xc = np.correlate(a, b, mode='full') / n
    centre = n - 1
    seg = xc[centre - maxlag: centre + maxlag + 1] if n > maxlag else xc
    if len(seg) == 0:
        max_xc, lag_at_max = 0.0, 0.0
    else:
        k = int(np.argmax(np.abs(seg)))
        max_xc = float(seg[k])
        lag_at_max = float(lags[k] / fs) if len(seg) == len(lags) else 0.0

    # magnitude-squared coherence in the LF band
    try:
        f, cxy = sps.coherence(a, b, fs=fs, nperseg=int(min(1024, n)))
        m = (f >= 0.03) & (f < 0.15)
        lf_coh = float(np.mean(cxy[m])) if m.any() else 0.0
    except Exception:
        lf_coh = 0.0

    # fraction of FHR decelerations that overlap a contraction
    dec = a < -1.0
    con = b > 1.0
    overlap = float(np.sum(dec & con) / (np.sum(dec) + 1e-12))

    return _finite([pearson, max_xc, lag_at_max, lf_coh, overlap])


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
FEATURE_NAMES = (
    [f'FHR_{n}' for n in ['mean', 'std', 'median', 'min', 'max', 'range',
                          'iqr', 'p90', 'skew', 'kurt', 'hist_ent']] +
    [f'UC_{n}' for n in ['mean', 'std', 'median', 'min', 'max', 'range',
                         'iqr', 'p90', 'skew', 'kurt', 'hist_ent']] +
    [f'FHR_wav_{n}' for n in ['cA3', 'cD3', 'cD2', 'cD1']] +
    [f'UC_wav_{n}' for n in ['cA3', 'cD3', 'cD2', 'cD1']] +
    [f'FHR_{n}' for n in ['VLF', 'LF', 'HF', 'LFHF', 'SEF95']] +
    [f'UC_{n}' for n in ['VLF', 'LF', 'HF', 'LFHF', 'SEF95']] +
    [f'FHR_{n}' for n in ['SampEn', 'ApEn', 'PermEn']] +
    [f'UC_{n}' for n in ['SampEn', 'ApEn', 'PermEn']] +
    ['FHR_baseline', 'FHR_STV', 'FHR_LTV', 'FHR_pctAbnSTV',
     'FHR_accel_n', 'FHR_decel_n', 'FHR_prolDecel_n', 'FHR_drift'] +
    ['UC_contraction_n', 'UC_mean_amp', 'UC_mean_dur'] +
    ['XC_pearson', 'XC_maxcorr', 'XC_lag_s', 'XC_lf_coherence',
     'XC_decel_uc_overlap']
)
assert len(FEATURE_NAMES) == 62, f'expected 62 names, got {len(FEATURE_NAMES)}'


def extract_62_features(fhr_raw, uc_raw, fs=FS):
    """Return a 62-vector of features for one CTU-UHB recording."""
    fhr = _clean(fhr_raw, lo=50, hi=200, fs=fs)
    uc = _clean(uc_raw, lo=0, hi=100, fs=fs)

    feats = []
    feats += descriptive_features(fhr)      # 11
    feats += descriptive_features(uc)       # 11  -> 22
    feats += wavelet_features(fhr)          #  4
    feats += wavelet_features(uc)           #  4  -> 30
    feats += spectral_features(fhr, fs)     #  5
    feats += spectral_features(uc, fs)      #  5  -> 40
    feats += entropy_features(fhr)          #  3
    feats += entropy_features(uc)           #  3  -> 46
    feats += fhr_morphology(fhr, fs)        #  8  -> 54
    feats += uc_morphology(uc, fs)          #  3  -> 57
    feats += coupling_features(fhr, uc, fs) #  5  -> 62

    assert len(feats) == 62, f'expected 62 features, got {len(feats)}'
    return np.asarray(feats, dtype=np.float32)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 19200
    t = np.arange(n) / FS
    fhr = 140 + 8 * np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 3, n)
    fhr[5000:5200] -= 30          # a deceleration
    fhr[9000:9600] -= 25          # a prolonged deceleration
    uc = 20 + 30 * (np.sin(2 * np.pi * t / 180) > 0.7) + rng.normal(0, 2, n)
    fhr[100:400] = 0              # simulated signal loss

    f = extract_62_features(fhr, uc)
    print('n features :', len(f))
    print('all finite :', bool(np.all(np.isfinite(f))))
    print('n names    :', len(FEATURE_NAMES))
    for name, val in list(zip(FEATURE_NAMES, f))[:6]:
        print(f'  {name:24s} {val: .4f}')
    print('  ...')
    for name, val in list(zip(FEATURE_NAMES, f))[-8:]:
        print(f'  {name:24s} {val: .4f}')
