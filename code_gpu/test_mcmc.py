import numpy as np
from fit_parallel_vect import mcmc, stacked_trapezoids, corrections_dwi0bk
import os
import cupy as cp
import matplotlib.pyplot as plt

use_gpu = False

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.8, 0.8, 120)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)

# Define initial parameters and multiples
dwx = [0.1]
dwz = [0.1]
i0 = 10
bkg = 0.1
height = [23.48, 23.45]
bot_cd = [54.6, 54.2]
swa = [[85],[87]]

multiples = [1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7] + len(swa[0]) * [1E-9]

if use_gpu:
    qxs = cp.array(qxs)
    qzs = cp.array(qzs)
    multiples = cp.array(multiples)

# Check if the number of initial guesses matches the number of multiples
# assert len(initial_guess) == len(multiples), f'Number of adds ({len(initial_guess)}) is different from number of multiples ({len(multiples)})'

# Generate data based on Fourier transform of arbitrary parameters using stacked_trapezoids
def generate_arbitrary_data(qxs, qzs):
    arbitrary_params = np.array([dwx[0], dwz[0], i0, bkg, height[0], bot_cd[0]] + swa[0])
    langle = np.deg2rad(np.asarray(swa))
    rangle = np.deg2rad(np.asarray(swa))

    # if (use_gpu):
    #     qxs = qxs.get()
    #     qzs = qzs.get()


    data = stacked_trapezoids(qxs, qzs, y1=np.asarray([0,0]), y2=np.asarray(bot_cd), height=np.asarray(height), langle=np.asarray(langle))


    data = corrections_dwi0bk(data, dwx, dwz, i0, bkg, qxs, qzs)

    return data, arbitrary_params



def test_mcmc_with_arbitrary_data():
    # Generate arbitrary data and parameters
    data, arbitrary_params = generate_arbitrary_data(qxs, qzs)

    if use_gpu:
        data = cp.asarray(data)
        arbitrary_params = cp.asarray(arbitrary_params)
        sigma = 100 * cp.asarray(arbitrary_params)
    else:
        data = np.asarray(data)
        arbitrary_params = np.asarray(arbitrary_params)
        sigma = 100 * np.asarray(arbitrary_params)
        
        best_corr, acceptance = mcmc(data=data[0],
                            qxs=qxs,
                            qzs=qzs,
                            initial_guess=arbitrary_params,
                            multiples=multiples,
                            N=len(arbitrary_params),
                            sigma=sigma,
                            nsteps=600,
                            nwalkers=200,  # needs to be higher than 2 x N
                            gaussian_move=False,
                            parallel=False,
                            seed=500,
                            verbose=True,
                            test=True,
                            use_gpu=use_gpu)
   
    print("arbitary_params:", arbitrary_params)
    print("best_params:", best_corr)
    tolerance = 1.0  # Adjust the tolerance as needed
    assert np.allclose(arbitrary_params,best_corr, atol=tolerance), "Test failed!"
    print("Test passed successfully!")

# Run the test
test_mcmc_with_arbitrary_data()
