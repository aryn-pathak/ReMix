import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 384 # no of samples for a measurement
T = 0.008 # seconds, time length of a measurement, at sampling rate of 48kHz

sofa = sf.read_sofa("KU_SS2.sofa")

target = [0.0, 0.0]
sources = sofa.SourcePosition[:, :2]

difference = sources - target
bestM = np.argmin(np.sum(difference ** 2, axis=1))

left_IR = sofa.Data_IR[bestM, 0, :]
right_IR = sofa.Data_IR[bestM, 1, :]

timeAxis = [i*(8/384) for i in range(384)]
plt.plot(timeAxis, left_IR, color="red", label="Left IR")
plt.plot(timeAxis, right_IR, color="blue", label="Right IR")
plt.xlabel("time (ms)")
plt.ylabel("values")
plt.show()