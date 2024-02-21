from fit_parallel import mcmc
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

sigma = 1E-7 * np.asarray(initial_guess)

use_gpu = True

if __name__ == '__main__':  # This is necessary for parallel execution
    # Iterate through different population sizes
    
    best_corr = mcmc(data=data,
                        qxs=qxs,
                        qzs=qzs,
                        initial_guess=np.asarray(initial_guess),
                        multiples=np.asarray(multiples),
                        N=len(initial_guess),
                        sigma=sigma,
                        nsteps=50,
                        nwalkers=120,  # needs to be higher than 2 x N
                        gaussian_move=False,
                        parallel=False,
                        seed=None,
                        verbose=True,
                        test=True)

    print("Best correlation: ", best_corr)

