import numpy as np
import cupy as cp
import multiprocessing

def custom_map(func, iterable):
    # Check if GPU is available
    if cp.cuda.is_available():
        # Get the number of elements in the iterable
        num_elements = len(iterable)

        # Create an output array to store the results
        output = cp.empty(num_elements)

        # Divide the work among the GPUs
        for i, item in enumerate(iterable):
            output[i] = func(item)

        return output.get()
    else:
        # Use multiprocessing library pool.map
        with multiprocessing.Pool() as pool:
            return np.asarray(pool.map(func, iterable))
        

