from fit_parallel import cmaes as cmaes_parallel
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cupy as cp

# Define the path and load data from a file
path = '../data'
qxs = np.loadtxt(os.path.join(path, 'qx_exp.txt'))
qzs = np.loadtxt(os.path.join(path, 'qz_exp.txt'))
data = np.loadtxt(os.path.join(path, 'i_exp.txt'))

qxs = qxs.flatten()
qzs = qzs.flatten()
data = data.flatten()

# Define initial parameters and multiples
dwx = 0.1
dwz = 0.1
i0 = 0.203
bkg = 0.1
height = 23.48
bot_cd = 54.6
swa = [78, 90, 88, 84, 88, 85]

initial_guess = np.array([dwx, dwz, i0, bkg, height, bot_cd] + swa)
multiples = [1E-18, 1E-18, 1E-18, 1E-17, 1E-17, 1E-17] + len(swa) * [1E-17]

# Check if the number of initial guesses matches the number of multiples
assert len(initial_guess) == len(multiples), f'Number of adds ({len(initial_guess)}) is different from number of multiples ({len(multiples)})'

# Define data arrays
data = data
qxs = qxs
qzs = qzs

# Define a range of population sizes
nbpop = [10]#np.arange(10, 700, 50)
# nbpop = [10, 11]


use_gpu = True

if __name__ == '__main__':  # This is necessary for parallel execution
    # Iterate through different population sizes
    for i in nbpop:
        # gpu execution

        if use_gpu:
            start = time.time()
            data = cp.asarray(data)
            qxs = cp.asarray(qxs)
            qzs = cp.asarray(qzs)
            multiples = cp.asarray(multiples)
            initial_guess = cp.asarray(initial_guess)

            best_corr, best_fitness = cmaes_parallel(data=data, qxs=qxs, qzs=qzs, sigma=100, ngen=30, popsize=i, mu=10,
                                                        n_default=len(initial_guess), multiples=multiples, restarts=0, verbose=False, tolhistfun=5e-5,
                                                        initial_guess=initial_guess, ftarget=None, dir_save=None, use_gpu=use_gpu)
            print(best_corr, best_fitness)
            end = time.time()

            print(f'gpu execution time for {i} individuals: {end - start} seconds')

        # non-gpu execution
        # start = time.time()
        # best_corr, best_fitness = cmaes_parallel(data=data, qxs=qxs, qzs=qzs, sigma=100, ngen=30, popsize=i, mu=10,
        #                                             n_default=len(initial_guess), multiples=multiples, restarts=0, verbose=False, tolhistfun=5e-5,
        #                                             initial_guess=initial_guess, ftarget=None, dir_save=None, use_gpu=False)
        # end = time.time()

        # print(f'non-gpu execution time for {i} individuals: {end - start} seconds')

