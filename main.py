import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 384 # no of samples in a measurement
T = 8 # length of a measurement in ms (48kHz sampling rate)

sofa = sf.read_sofa("KU_SS2.sofa")
sources = sofa.SourcePosition[:, :2]

def getir(az, el):
    difference = sources - [az, el]
    bestM = np.argmin(np.sum(difference ** 2, axis=1))
    right_IR = sofa.Data_IR[bestM, 0, :]
    left_IR = sofa.Data_IR[bestM, 1, :]
    return right_IR, left_IR

def plot_ir(az, el):
    timeAxis = [i*(T/N) for i in range(384)]
    plt.plot(timeAxis, getir(az, el)[1], color="red", label="Left IR")
    plt.plot(timeAxis, getir(az, el)[0], color="blue", label="Right IR")
    plt.xlabel("time (ms)")
    plt.ylabel("values")
    plt.legend()
    plt.show()

def firstsample(wave, threshold=0.15):
    peak = np.max(np.abs(wave))
    threshold = threshold * peak
    return np.argmax(np.abs(wave) >= threshold)

def itd(right, left):
    rightTime = (T/N)*firstsample(right)
    leftTime = (T/N)*firstsample(left)
    return leftTime - rightTime # +ve means right, -ve means left

def plot_itd(el):
    elevation = sources[:, 1]
    idx = np.where(np.isclose(elevation, el, atol=1e-5))[0]
    azAxis = np.sort(sources[idx, 0])

    itdarr = []
    for az in azAxis:
        itdarr.append(itd(*getir(az, 0)))

    plt.xlabel("azimuth (degrees)")
    plt.ylabel("ITD (ms)")
    plt.plot(azAxis, itdarr)
    plt.show()


def get_freq(az, el):
    right_IR, left_IR = getir(az, el)
    rightNormal = right_IR[firstsample(right_IR):]
    leftNormal = left_IR[firstsample(left_IR):]

    max_len = max(len(rightNormal), len(leftNormal))

    rightNormal = np.pad(rightNormal, (0, max_len - len(rightNormal)), 'constant')
    leftNormal = np.pad(leftNormal, (0, max_len - len(leftNormal)), 'constant')

    right_freq = 20 * np.log10(np.abs(np.fft.rfft(rightNormal)) + 1e-9)
    left_freq = 20 * np.log10(np.abs(np.fft.rfft(leftNormal)) + 1e-9)

    freqAxis = np.fft.rfftfreq(n=max_len, d=1 / 48000)
    return freqAxis, right_freq, left_freq


def plot_freq(az, el):
    freqAxis, right_freq, left_freq = get_freq(az, el)
    freqDiff = left_freq - right_freq

    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude difference (dB)")
    plt.plot(freqAxis, freqDiff, color="red", label="Frequency Difference")
    plt.legend()
    plt.show()

plot_freq(0, 0)