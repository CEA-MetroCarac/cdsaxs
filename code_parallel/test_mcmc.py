import numpy as np
from fit import stacked_trapezoids, corrections_dwi0bk
from fit_parallel import mcmc
import os

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.8,0.8,120)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)

# Define initial parameters and multiples
dwx = 0.1
dwz = 0.1
i0 = 10
bkg = 0.1
height = 23.48
bot_cd = 54.6
swa = [85]

multiples = [1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7] + len(swa) * [1E-9]
# multiples = [1]*len(multiples)

# Generate data based on Fourier transform of arbitrary parameters using stacked_trapezoids
def generate_arbitrary_data():
    arbitrary_params = np.array([dwx, dwz, i0, bkg, height, bot_cd] + swa)
    
    langle = np.deg2rad(np.asarray(swa))
    data = stacked_trapezoids(qxs, qzs, y1=0, y2=bot_cd, height=height, langle=langle)
    data = corrections_dwi0bk(data, dwx, dwz, i0, bkg, qxs, qzs)

    return data, arbitrary_params

def test_mcmc_with_arbitrary_data():
    # Generate arbitrary data and parameters
    data, arbitrary_params = generate_arbitrary_data()
    sigma = 1E-7 * np.asarray(arbitrary_params)

    # Call the mcmc function with arbitrary data
    if (__name__ == "__main__"):
        
        for i in range(1):
            best_corr = mcmc(data=data,
                            qxs=qxs,
                            qzs=qzs,
                            initial_guess=np.asarray(arbitrary_params),
                            multiples=np.asarray(multiples),
                            N=len(arbitrary_params),
                            sigma=sigma,
                            nsteps=700,
                            nwalkers=25,  # needs to be higher than 2 x N
                            gaussian_move=False,
                            parallel=False,
                            seed=None,
                            verbose=True,
                            test=True)
 
    print("arbitary_params:", arbitrary_params)
    print("best_params:", best_corr)
    tolerance = 1.0  # Adjust the tolerance as needed
    assert np.allclose(arbitrary_params,best_corr, atol=tolerance), "Test failed!"
    print("Test passed successfully!")

# Run the test
test_mcmc_with_arbitrary_data()
