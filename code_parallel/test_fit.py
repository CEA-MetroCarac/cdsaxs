from fit_parallel import cmaes as cmaes_parallel
from fit_parallel import PickeableResidual
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Define the path and load data from a file
path = '../data'
# qxs = np.loadtxt(os.path.join(path, 'qx_exp.txt'))
# qzs = np.loadtxt(os.path.join(path, 'qz_exp.txt'))
# data = np.loadtxt(os.path.join(path, 'i_exp.txt'))

pr = np.loadtxt(os.path.join(path, 'data_cdsaxs_test.txt'))
qxs = pr[:, 0]
qzs = pr[:, 1]
data = pr[:, 2]

# Define initial parameters and multiples
dwx = 0.1
dwz = 0.1
i0 = 0.203
bkg = 0.1
height = 23.48
bot_cd = 54.6
swa = [78, 90, 88, 84, 88, 85]

initial_guess = np.array([dwx, dwz, i0, bkg, height, bot_cd] + swa)
multiples = [1E-8, 1E-8, 1E-8, 1E-7, 1E-7, 1E-7] + len(swa) * [1E-7]

# Check if the number of initial guesses matches the number of multiples
assert len(initial_guess) == len(multiples), f'Number of adds ({len(initial_guess)}) is different from number of multiples ({len(multiples)})'

# Define data arrays
data = data
qxs = qxs
qzs = qzs


if __name__ == '__main__':
    best_corr, best_fitness = cmaes_parallel(data=data, qxs=qxs, qzs=qzs, sigma=100, ngen=11, popsize=10, mu=10,
                                            n_default=len(initial_guess), multiples=multiples, restarts=0, verbose=False, tolhistfun=5e-5,
                                            initial_guess=initial_guess, ftarget=None, dir_save=None)

    print(best_corr)

