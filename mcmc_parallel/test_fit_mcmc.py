import os
import numpy as np
import fit


 #necessary for multiprocessing to work otherwise it will try to create process within process
path = '.'
pr = np.loadtxt(os.path.join(path, 'data_cdsaxs_test.txt'))
qxs=pr[:, 0]
qzs=pr[:, 1]
data=pr[:, 2]

initial_guess = np.array([0.1, 1, 1, 30, 73, 5] + [80, 80])
multiples = [1E-8, 1E-8, 1E-8, 1E-7, 1E-7, 1E-7] + 2 * [1E-4]

assert len(initial_guess) == len(multiples), f'Number of adds ({len(initial_guess)}) is different from number of multiples ({len(multiples)})'

data = data
qxs = qxs
qzs = qzs

if __name__ == '__main__':
     fit.mcmc(data=data, 
          qxs=qxs, 
          qzs=qzs, 
          initial_guess=np.asarray(initial_guess),
          N=8, 
          sigma = 1e-07, 
          nsteps = 8, 
          nwalkers = 50, 
          gaussian_move=True, 
          parallel=False, 
          seed=None, 
          verbose=True)