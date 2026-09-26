import math
import numpy as np
import sofar as s
import soundfile as sf
import sounddevice as sd
import scipy.signal as signal

N = 256  # no of samples in a measurement
T = 256 / 48000 * 1000  # length of a measurement in ms (48kHz sampling rate)
r_ref = 1.0
SR=48000

sofa = s.read_sofa("SADIEII_KU100.sofa")
sources = sofa.SourcePosition[:, :2]

def getir(az, el):
    az_diff = (sources[:, 0] - az + 180) % 360 - 180
    el_diff = sources[:, 1] - el
    bestM = np.argmin(az_diff ** 2 + el_diff ** 2)
    left_IR = sofa.Data_IR[bestM, 0, :]
    right_IR = sofa.Data_IR[bestM, 1, :]

    return left_IR, right_IR

def apply_IR(az, el, wave):
    left_IR, right_IR = getir(az, el)
    left = signal.fftconvolve(wave, left_IR)
    right = signal.fftconvolve(wave, right_IR)
    return left, right

def apply_falloff(dist, wave):
    gain = r_ref / max(dist, r_ref)
    return wave * gain

absorption_dbm = {125.0: -0.0004, 250.0: -0.0013, 500.0: -0.0027, 1000.0: -0.0047, 1400.0: -0.0064, 2000.0: -0.0099, 2800.0: -0.0163, 4000.0: -0.0297, 5600.0: -0.0544, 8000.0: -0.1053, 11300.0: -0.1983, 16000.0: -0.3645}

def k(freq, db):
    return 10**(db/20) - (2*math.cos(2*math.pi*freq/SR))

def peaking_coeffs(freq, db, q):

    A = 10 ** (db / 40)
    w0 = 2 * math.pi * freq / SR
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

def apply_filter(wave, coeffs):
    b0, b1, b2, a1, a2 = coeffs
    result = []
    for n in range(len(wave)):
        x1 = wave[n - 1] if n - 1 >= 0 else 0
        x2 = wave[n - 2] if n - 2 >= 0 else 0
        y1 = result[n - 1] if n - 1 >= 0 else 0  # outputs already computed
        y2 = result[n - 2] if n - 2 >= 0 else 0
        filtered_sample = b0 * wave[n] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        result.append(filtered_sample)

    return result

def apply_eq(bands, wave, q=1.41):
    # bands: {frequency: dB}, +ve boosts, -ve cuts
    result = wave
    for freq in bands:
        result = apply_filter(result, peaking_coeffs(freq, bands[freq], q))

    return result