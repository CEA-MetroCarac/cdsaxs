import os
import numpy as np
import fit_parallel as fit


# Define the path and load data from a file
path = '../data'
qxs = np.loadtxt(os.path.join(path, 'qx_exp.txt'))
qzs = np.loadtxt(os.path.join(path, 'qz_exp.txt'))
data = np.loadtxt(os.path.join(path, 'i_exp.txt'))

# Define initial parameters and multiples
dwx = 0.1
dwz = 0.1
i0 = 0.203
bkg = 0.1
height = 23.48
bot_cd = 54.6
swa = [78, 90, 88, 84, 88, 85]

initial_guess = np.array([dwx, dwz, i0, bkg, height, bot_cd] + swa)
multiples = [1E-8, 1E-8, 1E-8, 1E-7, 1E-7, 1E-7] + len(swa) * [1E-5]
multiples = [1]*len(multiples)
sigma = 1E-7*np.asarray(initial_guess)


assert len(initial_guess) == len(multiples), f'Number of adds ({len(initial_guess)}) is different from number of multiples ({len(multiples)})'

data = data
qxs = qxs
qzs = qzs

if __name__ == '__main__':
     fit.mcmc(data=data, 
          qxs=qxs, 
          qzs=qzs, 
          initial_guess=np.asarray(initial_guess),
          multiples=np.asarray(multiples),
          N=12, 
          sigma = sigma, 
          nsteps = 50, 
          nwalkers = 100, #needs to be higher than 2 x N 
          gaussian_move=False, 
          parallel=False, 
          seed=None, 
          verbose=True)