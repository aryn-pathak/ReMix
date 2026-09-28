import math
import random

import librosa
import numpy as np
import sofar as s
import soundfile as sf
import sounddevice as sd
import scipy.signal as signal

N = 256  # no of samples in a measurement
r_ref = 1.0
SR=48000
T = 256 / SR * 1000  # length of a measurement in ms (48kHz sampling rate)

ROOM_DIMENSIONS = {
    "height" : 8,
    "length" : 12,
    "width" : 10
} # meters

A = {
    "ceiling" : 0.80,
    "floor" : 0.15,
    "walls" : 0.03
} # sound absorption constant

L_COORDS = (4.0, 5.0, 1.7)
S_COORDS = (6.0, 8.0, 1.7)

sofa = s.read_sofa("SADIEII_KU100.sofa")
sources = sofa.SourcePosition[:, :2]

hrirs = np.asarray(sofa.Data_IR, dtype=float)
df_energy = np.mean(np.sum(hrirs ** 2, axis=2), axis=0)   # one value per ear
hrirs /= np.sqrt(df_energy)[None, :, None]

def getir(az, el):
    az_diff = (sources[:, 0] - az + 180) % 360 - 180
    el_diff = sources[:, 1] - el
    bestM = np.argmin(az_diff ** 2 + el_diff ** 2)
    return hrirs[bestM, 0, :], hrirs[bestM, 1, :]

def apply_IR(az, el, wave):
    left_IR, right_IR = getir(az, el)
    left = signal.fftconvolve(wave, left_IR)
    right = signal.fftconvolve(wave, right_IR)
    return left, right

def direction(p):
    # listener faces +x (length), +y (width) is to the left, +z is up
    dx, dy, dz = (p[i] - L_COORDS[i] for i in range(3))
    az = math.degrees(math.atan2(dy, dx)) % 360
    el = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return az, el

def distance(p, q):
    return math.sqrt((q[0]-p[0])**2 + (q[1]-p[1])**2 + (q[2]-p[2])**2)

def apply_falloff(dist, wave):
    gain = r_ref / max(dist, r_ref)
    return wave * gain

absorption_dbm = {125.0: -0.0004, 250.0: -0.0013, 500.0: -0.0027, 1000.0: -0.0047, 1400.0: -0.0064, 2000.0: -0.0099, 2800.0: -0.0163, 4000.0: -0.0297, 5600.0: -0.0544, 8000.0: -0.1053, 11300.0: -0.1983, 16000.0: -0.3645}

def q_from_bandwidth_octaves(bw_octaves, freq):
    w0 = 2 * math.pi * freq / SR
    return 1 / (2 * math.sinh(math.log(2) / 2 * bw_octaves * w0 / math.sin(w0)))

def band_q(freqs, i):
    if len(freqs) == 1:
        return 1.41  # fallback, nothing to reference
    if i == 0:
        bw = math.log2(freqs[1] / freqs[0])
    elif i == len(freqs) - 1:
        bw = math.log2(freqs[i] / freqs[i - 1])
    else:
        bw = 0.5 * math.log2(freqs[i + 1] / freqs[i - 1])
    return q_from_bandwidth_octaves(bw, freqs[i])

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

def high_shelf_coeffs(freq, db, q):

    A = 10 ** (db / 40)
    w0 = 2 * math.pi * freq / SR
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    sqrt_A = math.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

def apply_filter(wave, coeffs):
    b0, b1, b2, a1, a2 = coeffs
    return signal.lfilter([b0, b1, b2], [1, a1, a2], wave)

def apply_eq(bands, wave):
    # bands: {frequency: dB}, +ve boosts, -ve cuts
    # Q is now derived per-band from neighbor spacing instead of fixed
    freqs = sorted(bands)
    target = np.array([bands[f] for f in freqs], dtype=float)

    def coeffs(i, db):
        # top band is a high shelf so the cut holds up to Nyquist
        if i == len(freqs) - 1 and len(freqs) > 1:
            return high_shelf_coeffs(freqs[i], db, 1 / math.sqrt(2))
        return peaking_coeffs(freqs[i], db, band_q(freqs, i))

    def band_response(i, db):  # dB of band i's filter at every band centre
        b0, b1, b2, a1, a2 = coeffs(i, db)
        _, h = signal.freqz([b0, b1, b2], [1, a1, a2], worN=2 * np.pi * np.array(freqs) / SR)
        return 20 * np.log10(np.abs(h))

    # bands overlap, so each band's gain != its target; solve for gains whose
    # combined response hits the target at every centre (Newton, dB isn't linear in gain)
    gains = target.copy()
    for _ in range(20):
        actual = np.sum([band_response(i, g) for i, g in enumerate(gains)], axis=0)
        interaction = np.column_stack([band_response(i, g + 0.01) - band_response(i, g)
                                       for i, g in enumerate(gains)]) / 0.01
        gains += np.linalg.solve(interaction, target - actual)

    result = wave
    for i, g in enumerate(gains):
        result = apply_filter(result, coeffs(i, g))

    return result

def apply_air_absorption(wave, dist): # meters
    bands = {k: v * dist for k, v in absorption_dbm.items()}
    return apply_eq(bands, wave)

def allpass(x, M, g=0.7):
    # Schroeder allpass: y[n] = -g*x[n] + x[n-M] + g*y[n-M]
    b = np.zeros(M + 1)
    b[0], b[M] = -g, 1
    a = np.zeros(M + 1)
    a[0], a[M] = 1, -g
    return signal.lfilter(b, a, x)

def room_reverb(wave):
    def is_prime(k):
        return k > 1 and all(k % p for p in range(2, int(k ** 0.5) + 1))

    def next_prime(k):
        while not is_prime(k):
            k += 1
        return k

    h = ROOM_DIMENSIONS["height"]
    w = ROOM_DIMENSIONS["width"]
    l = ROOM_DIMENSIONS["length"]

    v = h * w * l
    a = 2 * (h * w + h * l) * A["walls"] + l * w * A["floor"] + l * w * A["ceiling"]
    rts = 0.161 * v / a
    truncate_samples = int(rts * SR)

    max_dist = math.sqrt(h**2 + w**2 + l**2) / 2
    min_dist = min(h, w, l) / 2
    rng = random.Random(0)
    delays = set()
    while len(delays) < 6:
        d = rng.uniform(min_dist, max_dist)
        delays.add(next_prime(round(2 * d / 343 * SR)))

    # parallel combs: sparse, decaying spikes
    ir = np.zeros(truncate_samples)
    for M in sorted(delays):
        g = 10 ** (-3 * (M / SR) / rts)
        k = np.arange(1, (truncate_samples - 1) // M + 1)
        ir[k * M] += g ** (k - 1)

    # series allpasses: smear each spike into a dense cluster
    for t in (0.005, 0.0017):
        ir = allpass(ir, next_prime(round(t * SR)))

    ir /= math.sqrt(np.sum(ir ** 2))   # unit energy, keeps apply_drr calibrated
    return signal.fftconvolve(np.asarray(wave, dtype=float), ir)

def apply_drr(wave, dist):
    h = ROOM_DIMENSIONS["height"]
    w = ROOM_DIMENSIONS["width"]
    l = ROOM_DIMENSIONS["length"]

    a = 2 * (h * w + h * l) * A["walls"] + l * w * A["floor"] + l * w * A["ceiling"]

    falloff_gain = r_ref / max(dist, r_ref)
    rev_gain = 4 * math.sqrt(math.pi / a) / falloff_gain
    return room_reverb(wave) * rev_gain

def early_reflections(wave):
    size = (ROOM_DIMENSIONS["length"], ROOM_DIMENSIONS["width"], ROOM_DIMENSIONS["height"])

    def image(k):
        axis = k // 2
        wall = 0.0 if k % 2 == 0 else size[axis]
        p = list(S_COORDS)
        p[axis] = 2 * wall - p[axis]
        return tuple(p)

    SL = distance(S_COORDS, L_COORDS)

    def alpha(k):
        if k < 4:
            return A["walls"]
        return A["floor"] if k == 4 else A["ceiling"]

    def a(k):
        return math.sqrt(1 - alpha(k)) * max(SL, r_ref) / distance(image(k), L_COORDS)

    def M(k):
        return round((distance(image(k), L_COORDS) - SL) * SR / 343)

    # each reflection arrives from its image source's direction (the bounce point
    # lies on the listener -> image line), so it gets that direction's HRIR
    taps = [(M(k), a(k), *getir(*direction(image(k)))) for k in range(6)]
    max_delay = max(m for m, *_ in taps)

    b_l = np.zeros(max_delay + N)
    b_r = np.zeros(max_delay + N)
    for m, ak, left_IR, right_IR in taps:
        b_l[m:m + N] += ak * left_IR  # += in case two images share a delay
        b_r[m:m + N] += ak * right_IR

    x = np.asarray(wave, dtype=float)
    return signal.fftconvolve(x, b_l), signal.fftconvolve(x, b_r)

def decorrelate(wave, iacc = 0.4, d = 0.010):
    g = math.sqrt((1-iacc)/(1+iacc))
    d = int(round(d * SR))
    b_l = np.zeros(d + 1)
    b_l[0], b_l[d] = 1, g
    b_r = np.zeros(d + 1)
    b_r[0], b_r[d] = 1, -g
    x = np.concatenate([np.asarray(wave, dtype=float), np.zeros(d)])
    l = signal.lfilter(b_l, [1], x)
    r = signal.lfilter(b_r, [1], x)
    return l, r

def process_audio(audio_file):
    audio, sampling_rate = sf.read(audio_file, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sampling_rate != SR:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=SR, axis=0)

    dist = distance(S_COORDS, L_COORDS)
    az, el = direction(S_COORDS)

    audio = apply_air_absorption(audio, dist)      # unattenuated, air-absorbed
    direct = apply_falloff(dist, audio)            # distance-attenuated

    l_direct, r_direct = apply_IR(az, el, direct)
    l_early, r_early = early_reflections(direct)
    late_rev_L, late_rev_R = decorrelate(apply_drr(audio, dist))

    parts_l = [l_direct, l_early, late_rev_L]
    parts_r = [r_direct, r_early, late_rev_R]
    n = max(len(p) for p in parts_l + parts_r)

    def pad(p):
        return np.pad(np.asarray(p, dtype=float), (0, n - len(p)))

    left = np.sum([pad(p) for p in parts_l], axis=0)
    right = np.sum([pad(p) for p in parts_r], axis=0)

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 1:
        left, right = left / peak, right / peak

    return np.column_stack([left, right])

if __name__ == '__main__':
    audio_file = "test1.mp3"
    sd.play(process_audio(audio_file), SR)
    sd.wait()