import numpy as np
from fit import stacked_trapezoids
from fit_parallel import mcmc
import os

# # Define the path and load data from a file
# path = '../data'
# qxs = np.loadtxt(os.path.join(path, 'qx_exp.txt'))
# qzs = np.loadtxt(os.path.join(path, 'qz_exp.txt'))

pitch = 100 * 10e-9 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.8,0.8,120)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)

# Define initial parameters and multiples
dwx = 0.1
dwz = 0.1
i0 = 0.203
bkg = 0.1
height = 23.48
bot_cd = 54.6
swa = [80, 80, 80, 80, 80, 80]

multiples = [1E-9, 1E-9, 1E-9, 1E-9, 1E-9, 1E-9] + len(swa) * [1E-9]
# multiples = [1]*len(multiples)

# Generate data based on Fourier transform of arbitrary parameters using stacked_trapezoids
def generate_arbitrary_data():
    arbitrary_params = np.array([dwx, dwz, i0, bkg, height, bot_cd] + swa)
    langle = np.deg2rad(np.asarray(swa))
    data = stacked_trapezoids(qxs, qzs, y1=0, y2=bot_cd, height=height, langle=langle)
    return data, arbitrary_params

def test_mcmc_with_arbitrary_data():
    # Generate arbitrary data and parameters
    data, arbitrary_params = generate_arbitrary_data()
    sigma = 1E-7 * np.asarray(arbitrary_params)

    # Call the mcmc function with arbitrary data
    if (__name__ == "__main__"):
        
        for i in range(5):
            best_corr = mcmc(data=data,
                            qxs=qxs,
                            qzs=qzs,
                            initial_guess=np.asarray(arbitrary_params),
                            multiples=np.asarray(multiples),
                            N=len(arbitrary_params),
                            sigma=sigma,
                            nsteps=50,
                            nwalkers=120,  # needs to be higher than 2 x N
                            gaussian_move=False,
                            parallel=False,
                            seed=None,
                            verbose=True,
                            test=True)

            print("Best correlation: ", best_corr)
            print("arbitrary_params: ", arbitrary_params)

        # Split the obtained parameters into non-angles and angles
        arbitary_non_angles = arbitrary_params[:6]
        best_non_angles = best_corr[:6]
        arbitary_angles = arbitrary_params[6:]
        best_angles = best_corr[6:]

        print("arbitary_non_angles:", arbitary_non_angles)
        print("best_non_angles:", best_non_angles)

        # Compare the obtained non-angle parameters with the arbitrary ones within a tolerance
        non_angle_tolerance = 1  # Adjust the tolerance as needed
        assert np.allclose(arbitary_non_angles, best_non_angles, atol=non_angle_tolerance), "Test failed!"

        print("arbitary_angles:", arbitary_angles)
        print("best_angles:", best_angles)

        # Compare the obtained angle parameters with the arbitrary ones within a tolerance
        angle_tolerance = 15  # Adjust the tolerance as needed
        assert np.allclose(arbitary_angles, best_angles, atol=angle_tolerance), "Test failed!"

        print("Test passed successfully!")

# Run the test
test_mcmc_with_arbitrary_data()
