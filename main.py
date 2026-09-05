import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 384
T = 8

sofa = sf.read_sofa("KU_SS2.sofa")
sources = sofa.SourcePosition[:, :2]

def getir(az,el):
    difference = sources - [az, el]
    bestM = np.argmin(np.sum(difference ** 2, axis=1))
    right_IR = sofa.Data_IR[bestM, 0, :]
    left_IR = sofa.Data_IR[bestM, 1, :]
    return right_IR, left_IR

def plot_ir():
    timeAxis = [i*(T/N) for i in range(384)]
    plt.plot(timeAxis, getir(0.0, 0.0)[1], color="red", label="Left IR")
    plt.plot(timeAxis, getir(0.0, 0.0)[0], color="blue", label="Right IR")
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
    return leftTime - rightTime

def plot_itd():
    el = sources[:, 1]
    idx = np.where(np.isclose(el, 0.0, atol=1e-5))[0]
    azAxis = np.sort(sources[idx, 0])

    itdarr = []
    for az in azAxis:
        itdarr.append(itd(*getir(az, 0)))

    plt.plot(azAxis, itdarr)
    plt.show()

plot_ir()
plot_itd()