import numpy as np

def get_spikes(curve, threshold=1.5):
    deltas = [curve[i] - curve[i-1] for i in range(1, len(curve))]

    tp_frequency = []
    for i in range(len(deltas)-1):
        if deltas[i]*deltas[i+1] < 0:
            tp_frequency.append(i+1)

    filtered = []
    for frequency in tp_frequency:
        left_jump = np.abs(curve[frequency] - curve[frequency-1])
        right_jump = np.abs(curve[frequency] - curve[frequency+1])
        if left_jump >= threshold or right_jump >= threshold:
            filtered.append((frequency, curve[frequency]))

    return filtered