import numpy as np
from scipy import signal, spatial
from scipy.signal import find_peaks, peak_widths


def mse_per_sample(preds, targets):
    return np.mean((np.asarray(preds) - np.asarray(targets)) ** 2, axis=1)


def mae_per_sample(preds, targets):
    return np.mean(np.abs(np.asarray(preds) - np.asarray(targets)), axis=1)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asarray(a)[0:100]
    mx = np.amax(a)
    a = np.divide(a, mx)
    a = np.square(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def cosine_distances(preds, targets):
    return np.asarray([spatial.distance.cosine(t, p) for p, t in zip(preds, targets)])


def max_cross_correlations(preds, targets):
    return np.asarray([max(signal.correlate(t, p)) for p, t in zip(preds, targets)])


def width_error_ms(preds, targets, sample_rate_hz=5000, rel_heights=None):
    rel_heights = np.linspace(0.05, 0.8, 15) if rel_heights is None else rel_heights
    pred_widths, raw_widths = [], []
    for pred, raw in zip(preds, targets):
        pred_sample, raw_sample = [], []
        for rel_height in rel_heights:
            w, _, _, _ = peak_widths(pred, find_peaks(pred, distance=3000)[0], rel_height=rel_height)
            pred_sample.append(w)
            w, _, _, _ = peak_widths(raw, find_peaks(raw, distance=3000)[0], rel_height=rel_height)
            raw_sample.append(w)
        pred_widths.append(pred_sample)
        raw_widths.append(raw_sample)
    pred_widths = np.asarray(pred_widths)[:, :, 0]
    raw_widths = np.asarray(raw_widths)[:, :, 0]
    return np.abs(raw_widths - pred_widths) / (sample_rate_hz / 1000.0)


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
