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

def q_from_bandwidth_octaves(bw_octaves):
    return 1 / (2 * math.sinh(math.log(2) / 2 * bw_octaves))

def band_q(freqs, i):
    if len(freqs) == 1:
        return 1.41  # fallback, nothing to reference
    if i == 0:
        bw = math.log2(freqs[1] / freqs[0])
    elif i == len(freqs) - 1:
        bw = math.log2(freqs[i] / freqs[i - 1])
    else:
        bw = 0.5 * math.log2(freqs[i + 1] / freqs[i - 1])
    return q_from_bandwidth_octaves(bw)

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

def apply_eq(bands, wave):
    # bands: {frequency: dB}, +ve boosts, -ve cuts
    # Q is now derived per-band from neighbor spacing instead of fixed
    freqs = sorted(bands)
    result = wave
    for i, freq in enumerate(freqs):
        q = band_q(freqs, i)
        result = apply_filter(result, peaking_coeffs(freq, bands[freq], q))

    return result

def apply_air_absorption(wave, dist): # meters
    bands = {k: v * dist for k, v in absorption_dbm.items()}
    return apply_eq(bands, wave)

def room_reverb(wave):
    v = ROOM_DIMENSIONS["height"]*ROOM_DIMENSIONS["width"]*ROOM_DIMENSIONS["length"]
    a = (ROOM_DIMENSIONS["height"]*ROOM_DIMENSIONS["width"] + ROOM_DIMENSIONS["height"]*ROOM_DIMENSIONS["length"])*A["walls"] + ROOM_DIMENSIONS["length"]*ROOM_DIMENSIONS["width"]*A["floor"] + ROOM_DIMENSIONS["length"]*ROOM_DIMENSIONS["width"]*A["ceiling"]
    rts = 0.161*v*a # RT60 reverberation (s)
    truncate_samples = rts*SR

    max_dist = math.sqrt(ROOM_DIMENSIONS["height"]**2 +ROOM_DIMENSIONS["width"]**2 +ROOM_DIMENSIONS["length"]**2)/2
    min_dist = ROOM_DIMENSIONS["width"]/2

    comb_filters = []

    for i in range(6):
        d = random.randint(min_dist, max_dist)
        t = 2*d/343
        g = 10^(-3*t/rts)
        M = t*SR

        y = []
        for n in range(len(wave)):
            y.append(wave[n] + g * y[n - M])

        comb_filters.append(y)
    return np.sum(comb_filters, axis=0)