import numpy as np
from fit_parallel_vect import cmaes, stacked_trapezoids, corrections_dwi0bk
import os
import cupy as cp

use_gpu = False

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 120)
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

    # if ~use_gpu:
    #     qxs = qxs.get()
    #     qzs = qzs.get()


    data = stacked_trapezoids(qxs, qzs, y1=np.asarray([0,0]), y2=np.asarray(bot_cd), height=np.asarray(height), langle=np.asarray(langle))


    data = corrections_dwi0bk(data, dwx, dwz, i0, bkg, qxs, qzs)
    return data, arbitrary_params

def test_cmaes_with_arbitrary_data():
    # Generate arbitrary data and parameters
    data, arbitrary_params = generate_arbitrary_data(qxs, qzs)
    

    if use_gpu:
        data = cp.array(data)
        arbitrary_params = cp.array(arbitrary_params)
    else:
        data = np.array(data)
        arbitrary_params = np.array(arbitrary_params)

    # Call the cmaes function with arbitrary data
    if __name__ == "__main__":
        
        best_corr, best_fitness = cmaes(data=data[0], qxs=qxs, qzs=qzs, sigma=100, ngen=1000, popsize=200, mu=10,
                                            n_default=len(arbitrary_params), multiples=multiples, restarts=0, verbose=False, tolhistfun=5e-5,
                                            initial_guess=arbitrary_params, ftarget=None, dir_save=None, use_gpu=use_gpu)
    
    print("arbitary_params:", arbitrary_params)
    print("best_params:", best_corr)
    tolerance = 1.0  # Adjust the tolerance as needed
    assert np.allclose(arbitrary_params,best_corr, atol=tolerance), "Test failed!"
    print("Test passed successfully!")

# Run the test
test_cmaes_with_arbitrary_data()
