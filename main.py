import numpy as np
import sofar as sf
sofa = sf.read_sofa("KU_SS2.sofa")

target = [0.0, 0.0]
sources = sofa.SourcePosition[:, :2]

difference = sources - target
bestM = np.argmin(np.sum(difference ** 2, axis=1))

left_IR = sofa.Data_IR[bestM, 0, :]
right_IR = sofa.Data_IR[bestM, 1, :]

print(left_IR)
print(right_IR)