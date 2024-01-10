import numpy as np
import cupy as cp


array = np.random.rand(100)
array2 = np.random.rand(100)

print(np.log10(array) - np.log10(array2))
