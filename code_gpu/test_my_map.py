import timeit
import numpy as np
import cupy as cp
import multiprocessing as mp
from multiprocessing import Pool
from my_map import custom_map

def test_custom_map(custom_map, func, iterable, use_gpu=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    start_time = timeit.default_timer()

    result = custom_map(func, iterable, use_gpu)

    end_time = timeit.default_timer()
    execution_time = end_time - start_time

    print(f"Execution time of custom_map (use_gpu={use_gpu}): {execution_time} seconds")
    return result

# Usage example
def my_function(x, use_gpu=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    arr = xp.array((2048**5))

    return xp.sin(arr) + xp.cos(arr)

if __name__ == '__main__':
    iterable = range(10000)

    # Test with use_gpu=True
    test_custom_map(custom_map, my_function, iterable, use_gpu=True)

    # Test with use_gpu=False
    test_custom_map(custom_map, my_function, iterable)