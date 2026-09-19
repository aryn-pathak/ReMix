import numpy as np

def get_peaks(freqs, curve, threshold=1.5):
    deltas = [curve[i] - curve[i-1] for i in range(1, len(curve))]

    tp_index = []
    for i in range(len(deltas)-1):
        if deltas[i]*deltas[i+1] < 0:
            tp_index.append(i+1)

    filtered = []
    for idx in tp_index:
        left_jump = np.abs(curve[idx] - curve[idx-1])
        right_jump = np.abs(curve[idx] - curve[idx+1])
        if left_jump >= threshold or right_jump >= threshold:
            filtered.append((freqs[idx], curve[idx]))

    return filtered