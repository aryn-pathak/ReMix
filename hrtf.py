import numpy as np

bark_edges = [
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
    1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
    9500, 12000, 15500, 20000
]


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

def run_analysis(curves):
    phases = [[]]

    for c in range(len(curves)):
        phases[-1].append(get_peaks(curves[c]))

        if c < len(curves) - 1 and not peak_similarity(curves[c], curves[c+1]):
            phases.append([])

    results = []
    for phase in phases:
        if not phase:
            continue

        initial = phase[0]
        n_points = len(initial)

        freq_changes = [[] for _ in range(n_points)]
        mag_changes = [[] for _ in range(n_points)]

        for t in range(1, len(phase)):
            prev = phase[t-1]
            curr = phase[t]
            for k in range(n_points):
                f_prev, m_prev = prev[k]
                f_curr, m_curr = curr[k]
                freq_changes[k].append(f_curr - f_prev)
                mag_changes[k].append(m_curr - m_prev)

        results.append({
            'initial': initial,
            'freq_changes': freq_changes,
            'mag_changes': mag_changes,
        })

    return results

def simplify_freq_track(bins, sr, fft_size, bark_tol=0.2):

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

    return [simplified[t] - simplified[t-1] for t in range(1, len(simplified))]

def simplify_mag_track(mags, mag_tol=1.0):

    simplified = [mags[0]]
    anchor = 0

    for t in range(1, len(mags)):
        if abs(mags[t] - mags[anchor]) > mag_tol:
            simplified.append(mags[t])
            anchor = t
        else:
            simplified.append(simplified[-1])

    return [simplified[t] - simplified[t-1] for t in range(1, len(simplified))]
