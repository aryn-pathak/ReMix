import numpy as np
from scipy.interpolate import PchipInterpolator
from visualisation import freqAxis as freq_axis
from visualisation import az_list
from visualisation import get_freq, smooth_fractional_octave

bark_edges = [
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
    1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
    9500, 12000, 15500, 20000
]

mask = freq_axis <= 20000
curves = []
for az in az_list:
    right_freq, _ = get_freq(az, 0)
    curves.append(smooth_fractional_octave(freq_axis, right_freq, fraction=3)[mask])

def get_peaks(curve, threshold=1.5, slope_tol=0.5):
    deltas = [curve[i] - curve[i-1] for i in range(1, len(curve))]

    significant = {0, len(curve)-1}

    for i in range(len(deltas)-1):
        idx = i + 1

        if deltas[i] * deltas[i+1] < 0:
            left_jump = abs(curve[idx] - curve[idx-1])
            right_jump = abs(curve[idx] - curve[idx+1])
            if left_jump >= threshold or right_jump >= threshold:
                significant.add(idx)

        elif abs(deltas[i+1] - deltas[i]) > slope_tol:
            significant.add(idx)

    return [(idx, curve[idx]) for idx in sorted(significant)]

def peak_similarity(curve_one, curve_two, f_tol=10, m_tol=4): # boolean function
    peaks_curve_one = get_peaks(curve_one)
    peaks_curve_two = get_peaks(curve_two)

    if len(peaks_curve_one) != len(peaks_curve_two):
        return False

    for (f1, m1), (f2, m2) in zip(peaks_curve_one, peaks_curve_two):
        if np.abs(f1-f2) > f_tol:
            return False
        if np.abs(m1-m2) > m_tol:
            return False

    return True

def simplify_freq_track(bins, bark_tol=0.2, sr=48000, fft_size=256):
    barks = np.interp(np.array(bins) * sr / fft_size,
                      bark_edges, np.arange(len(bark_edges)))

    simplified = [bins[0]]
    anchor = 0                      # frame index of last committed position

    for t in range(1, len(bins)):
        if abs(barks[t] - barks[anchor]) > bark_tol:
            simplified.append(bins[t])
            anchor = t
        else:
            simplified.append(simplified[-1])

    return [simplified[t] - simplified[0] for t in range(1, len(simplified))]

def simplify_mag_track(mags, mag_tol=1.0):

    simplified = [mags[0]]
    anchor = 0

    for t in range(1, len(mags)):
        if abs(mags[t] - mags[anchor]) > mag_tol:
            simplified.append(mags[t])
            anchor = t
        else:
            simplified.append(simplified[-1])

    return [simplified[t] - simplified[0] for t in range(1, len(simplified))]

def run_analysis(curves, sr, fft_size, bark_tol=0.2, mag_tol=1.0):
    peaks_per_frame = [get_peaks(c) for c in curves]

    phases = [[]]
    for c in range(len(curves)):
        phases[-1].append(peaks_per_frame[c])
        if c < len(curves) - 1 and not peak_similarity(curves[c], curves[c+1]):
            phases.append([])

    results = []
    for phase in phases:
        if not phase:
            continue

        initial = phase[0]
        n_points = len(initial)

        freq_changes = []
        mag_changes = []

        for k in range(n_points):
            bins_k = [frame[k][0] for frame in phase]
            mags_k = [frame[k][1] for frame in phase]

            freq_changes.append(simplify_freq_track(bins_k, sr, fft_size, bark_tol))
            mag_changes.append(simplify_mag_track(mags_k, mag_tol))

        results.append({
            'initial': initial,
            'freq_changes': freq_changes,
            'mag_changes': mag_changes,
        })

    return results

def reconstruct(n, deconstructed): # curve number
    c = 0
    for phase in deconstructed:
        length = len(phase['freq_changes'][0]) + 1
        if n < c + length:
            idx = n - c
            peaks = list(phase['initial'])

            if idx > 0:
                for k in range(len(peaks)):
                    f, m = peaks[k]
                    peaks[k] = (f + phase['freq_changes'][k][idx-1],
                                m + phase['mag_changes'][k][idx-1])
            break
        c += length

    peaks.sort(key=lambda p: p[0])
    freqs = [p[0] for p in peaks]
    mags = [p[1] for p in peaks]

    interpolator = PchipInterpolator(freqs, mags)
    return interpolator(freq_axis)

def choose_n(az, ear): # R or L
    if ear == 'R':
        return az_list.index(az)
    else:
        return az_list.index(360-az)