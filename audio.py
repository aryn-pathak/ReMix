import numpy as np
import sofar as s
import soundfile as sf
import sounddevice as sd
import scipy.signal as signal

N = 256  # no of samples in a measurement
T = 256 / 48000 * 1000  # length of a measurement in ms (48kHz sampling rate)
r_ref = 1.0

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