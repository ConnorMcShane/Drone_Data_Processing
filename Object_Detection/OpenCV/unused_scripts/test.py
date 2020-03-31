import numpy as np

nparray = np.array([np.array([np.array([x, x])]) for x in range(10)]) * 2
print(nparray)