import os

import numpy as np

try:
    from scipy.signal import butter, correlate, filtfilt, find_peaks, peak_widths
except ImportError:
    butter = correlate = filtfilt = find_peaks = peak_widths = None


DEFAULT_FS = 5000
DEFAULT_ETHAN_PAIRS = [[0, 12, 23], [1, 12, 23], [1, 33, 22], [3, 26, 17], [4, 44, 31], [5, 34, 24], [6, 21, 32]]


def require_scipy():
    if butter is None:
        raise ImportError("SciPy is required for raw signal preprocessing and APD-width calculations.")


def butter_bandpass(lowcut, highcut, fs, order=5):
    require_scipy()
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    return filtfilt(b, a, data, axis=1)


def find_poration_pulses(filtered_rec):
    require_scipy()
    largest_events = np.max(np.abs(filtered_rec), axis=0) * np.median(np.abs(filtered_rec), axis=0)
    thresh = 200 * np.median(largest_events)
    dist = len(largest_events) / 100.0
    p_idx, _ = find_peaks(
        largest_events,
        height=thresh,
        prominence=thresh,
        distance=dist,
        wlen=10000,
        width=(100, 5000),
        rel_height=0.8,
    )
    return p_idx


def find_rhythm(trace, shortest_beat_estimate=500, longest_beat_estimate=8000):
    require_scipy()
    corr = correlate(trace, trace, mode="same")
    corr = corr[len(corr) // 2 : len(corr) // 2 + longest_beat_estimate]
    return np.argmax(corr[shortest_beat_estimate:]) + shortest_beat_estimate


def get_peak_indices(filtered_rec, buffer, poration_splits, trace_idx):
    slices = poration_splits
    peak_indices = []
    trace = filtered_rec[trace_idx]

    if len(slices) > 0:
        periods = []
        for i in range(len(slices) - 1):
            if i == 0:
                sub_trace = filtered_rec[:, slices[i] : slices[i + 1] - buffer]
            elif i == len(slices) - 1:
                sub_trace = filtered_rec[:, slices[i] + buffer : slices[i + 1]]
            else:
                sub_trace = filtered_rec[:, slices[i] + buffer : slices[i + 1] - buffer]

            if slices[i + 1] - slices[i] > 2 * buffer + 1:
                periods.append(find_rhythm(np.mean(sub_trace, axis=0)))
            else:
                periods.append(0)

        for i in range(len(slices) - 1):
            if i == 0:
                sub_trace = trace[slices[i] : slices[i + 1] - buffer]
            elif i == len(slices) - 1:
                sub_trace = trace[slices[i] + buffer : slices[i + 1]]
            else:
                sub_trace = trace[slices[i] + buffer : slices[i + 1] - buffer]

            period = 0.6 * periods[i]
            if slices[i + 1] - slices[i] > 2 * buffer + 1:
                noise_level = np.std(butter_bandpass_filter(sub_trace, 2000.0, 2499.0, fs=DEFAULT_FS))
                peaks, _ = find_peaks(np.abs(sub_trace), height=12 * noise_level, distance=period)
                if i != 0:
                    peaks += slices[i] + buffer
                peak_indices.append(peaks)
    else:
        period = 0.6 * find_rhythm(np.mean(filtered_rec, axis=0))
        noise_level = np.std(butter_bandpass_filter(trace, 2000.0, 2499.0, fs=DEFAULT_FS))
        peaks, _ = find_peaks(np.abs(trace), height=12 * noise_level, distance=period)
        peak_indices.append(peaks)

    return np.concatenate(peak_indices) if peak_indices else np.array([], dtype=int)


def moving_filter(data, window_size):
    moving_averages = []
    i = 0
    while i < len(data) - window_size + 1:
        moving_averages.append(sum(data[i : i + window_size]) / window_size)
        i += 1
    for i in range(window_size - 1):
        moving_averages.append(data[len(data) - (window_size - i)])
    return moving_averages


def get_good_peak_pairs(peaks, c_idx, noise_levels, data, npoints_beforepeak=1000, npoints_afterpeak=2000):
    def span(x):
        return np.max(x) - np.min(x)

    pairs_0, pairs_1 = [], []
    count = 0
    for peak_0 in peaks[0]:
        count += 1
        if count % 2 != 0:
            continue
        if not ((peak_0 > npoints_beforepeak) & (peak_0 < len(data.T) - npoints_beforepeak - 1)):
            continue

        data_0 = data[c_idx[0]][peak_0 - npoints_beforepeak : peak_0 + npoints_afterpeak]
        data_1 = data[c_idx[1]][peak_0 - npoints_beforepeak : peak_0 + npoints_afterpeak]
        span_0 = span(data_0[0:1050])
        span_1 = span(data_1)
        std_1 = np.std(data_1[1050:1550])
        snr_0 = span_0 / noise_levels[0]
        snr_1 = span_1 / noise_levels[1]

        if (snr_0 > 50) & (snr_1 > 50):
            if (np.argmax(data_0) > npoints_beforepeak - 300) & (np.argmax(data_0) < npoints_beforepeak + 300):
                pairs_0.append((data_0 - data_0[0]) / span_0)
                pairs_1.append((data_1 - np.mean(data_1)) / (std_1 * 40))
    return pairs_0, pairs_1


def compute_apd_widths(intras, rel_heights=None, peak_distance=3000):
    require_scipy()
    rel_heights = np.linspace(0.05, 0.8, 15) if rel_heights is None else rel_heights
    apd_widths = []
    for intra in intras:
        widths = []
        peaks = find_peaks(intra, distance=peak_distance)[0]
        for rel_height in rel_heights:
            width, _, _, _ = peak_widths(intra, peaks, rel_height=rel_height)
            widths.append(width)
        apd_widths.append(widths)
    return np.asarray(apd_widths)[:, :, 0]


def extract_pairs_from_h5_directory(raw_dir, output_dir, ethan_pairs=None, fs=DEFAULT_FS):
    require_scipy()
    import h5py as h5

    raw_dir = os.fspath(raw_dir)
    output_dir = os.fspath(output_dir)
    ethan_pairs = ethan_pairs or DEFAULT_ETHAN_PAIRS
    recordings = sorted([el for el in os.listdir(raw_dir) if el.endswith(".h5")])
    if not recordings:
        raise FileNotFoundError(f"No .h5 recording files found in {raw_dir}")

    def get_paired_dat_and_info(ethan_obs):
        f_idx = ethan_pairs[ethan_obs][0]
        with h5.File(os.path.join(raw_dir, recordings[f_idx]), "r") as h5_file:
            stream = h5_file["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]
            channel_to_elec = np.asarray([(el[0], int(el[4])) for el in list(stream["InfoChannel"])])
            data = np.asarray(stream["ChannelData"])
        filtered_data = butter_bandpass_filter(data, 1, fs / 2.0 - 1, fs)
        p_idx = find_poration_pulses(filtered_data)
        poration_splits = np.hstack(([0], p_idx, len(filtered_data[0]))) if len(p_idx) > 0 else []
        return data, channel_to_elec, filtered_data, poration_splits

    def get_peaks_from_both_channels(ethan_obs, data, filtered_data, channel_to_elec, poration_splits):
        peaks, noise_levels, c_idx = [], [], []
        for idx in ethan_pairs[ethan_obs][1:]:
            i = channel_to_elec[np.asarray(channel_to_elec)[:, 1] == idx][0][0]
            noise_level = np.std(butter_bandpass_filter(filtered_data[i], 0.8 * fs / 2.0, fs / 2.0 - 1, fs))
            pks = get_peak_indices(filtered_data, poration_splits=poration_splits, buffer=50000, trace_idx=i)
            peaks.append(pks)
            c_idx.append(i)
            noise_levels.append(noise_level)
        return peaks, c_idx, noise_levels

    all_intras, all_extras = [], []
    for ethan_obs in range(len(ethan_pairs)):
        data, channel_to_elec, filtered_data, poration_splits = get_paired_dat_and_info(ethan_obs)
        peaks, c_idx, noise_levels = get_peaks_from_both_channels(
            ethan_obs, data, filtered_data, channel_to_elec, poration_splits
        )
        pairs_0, pairs_1 = get_good_peak_pairs(peaks, c_idx, noise_levels, data)
        width_0 = [peak_widths(p, peaks=[np.argmax(p)], rel_height=0.5)[0][0] for p in pairs_0]
        width_1 = [peak_widths(p, peaks=[np.argmax(p)], rel_height=0.5)[0][0] for p in pairs_1]

        if np.median(width_1) > np.median(width_0):
            intra, w_intra = pairs_1, np.asarray(width_1)
            extra, w_extra = pairs_0, np.asarray(width_0)
        else:
            extra, w_extra = pairs_1, np.asarray(width_1)
            intra, w_intra = pairs_0, np.asarray(width_0)

        intra, extra = np.asarray(intra), np.asarray(extra)
        keep = (w_intra > 100) & (w_extra > 0) & (w_extra < 30)
        all_intras.append(intra[keep])
        all_extras.append(extra[keep])

    os.makedirs(output_dir, exist_ok=True)
    all_intras = np.concatenate(all_intras)
    all_extras = np.concatenate(all_extras)
    apds = compute_apd_widths(all_intras)
    np.save(os.path.join(output_dir, "intras_new_normalization.npy"), all_intras)
    np.save(os.path.join(output_dir, "extras_new_normalization.npy"), all_extras)
    np.save(os.path.join(output_dir, "APDS_new_normalization.npy"), apds)
    return all_extras, all_intras, apds
