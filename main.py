import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 384 # no of samples for a measurement
T = 8 # milliseconds, time length of a measurement, at sampling rate of 48kHz

sofa = sf.read_sofa("KU_SS2.sofa")

target = [90.0, 0.0]
sources = sofa.SourcePosition[:, :2]

difference = sources - target
bestM = np.argmin(np.sum(difference ** 2, axis=1))

left_IR = sofa.Data_IR[bestM, 1, :]
right_IR = sofa.Data_IR[bestM, 0, :]

timeAxis = [i*(T/N) for i in range(384)]
plt.plot(timeAxis, left_IR, color="red", label="Left IR")
plt.plot(timeAxis, right_IR, color="blue", label="Right IR")
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
