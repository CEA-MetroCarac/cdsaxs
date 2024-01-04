import numpy as np
import cupy as cp
import multiprocessing
import numpy as np
import multiprocessing as mp

def custom_map(func, iterable, use_gpu=False):
    if cp.cuda.is_available() and use_gpu:
        # Convert the iterable to a CuPy array for GPU processing
        iterable_gpu = cp.array(iterable)

        # Apply the function in a vectorized manner
        output = func(iterable_gpu)
        print(output)

        # Convert back to a NumPy array if needed outside GPU context
        return cp.asnumpy(output)
    else:
        # Use a persistent multiprocessing pool
        pool = multiprocessing.Pool()
        output = np.asarray(pool.map(func, iterable))
        pool.close()
        pool.join()
        return output