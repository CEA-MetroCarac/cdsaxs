import timeit
from multiprocessing import Pool
from my_map import custom_map
import numpy as np

def my_function(x):
    return np.sin(x) + np.cos(x)

if __name__ == '__main__':
    # Create a sample iterable
    iterable = range(1000000)

    # Measure the execution time of custom_map
    custom_map_time = timeit.timeit(lambda: custom_map(my_function, iterable), number=1)

    # Measure the execution time of pool.map
    with Pool() as pool:
        pool_map_time = timeit.timeit(lambda: pool.map(my_function, iterable), number=1)

    # Print the results
    print(f"Execution time of custom_map: {custom_map_time} seconds")
    print(f"Execution time of pool.map: {pool_map_time} seconds")
