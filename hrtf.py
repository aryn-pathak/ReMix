import numpy as np

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

    return phases