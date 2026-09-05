import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 384 # no of samples for a measurement
T = 8 # milliseconds, time length of a measurement, at sampling rate of 48kHz

sofa = sf.read_sofa("KU_SS2.sofa")
sources = sofa.SourcePosition[:, :2]

def getir(az,el): # 0 right, 1 left
    difference = sources - [az, el]
    bestM = np.argmin(np.sum(difference ** 2, axis=1))
    right_IR = sofa.Data_IR[bestM, 0, :]
    left_IR = sofa.Data_IR[bestM, 1, :]
    return right_IR, left_IR

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
    return leftTime - rightTime # negative ITD = right sound, positive = left sound

el = sources[:, 1]
azAxis = [i for i in range(
    len(np.where(np.isclose(el, 0.0, atol=1e-5))[0])
)]

itdarr = []
for M in azAxis:
    az = sources[M, 0]
    itdarr.append(itd(az, 0))

plt.plot(azAxis, itdarr)
plt.show()