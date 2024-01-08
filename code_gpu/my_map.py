import numpy as np
import cupy as cp
import multiprocessing
import numpy as np
import multiprocessing as mp

def custom_map(func, iterable):
        # Convert the iterable to a CuPy array for GPU processing
        iterable_gpu = cp.array(iterable)

        # Apply the function in a vectorized manner
        output = func(iterable_gpu)

        # Convert back to a NumPy array if needed outside GPU context
        return output