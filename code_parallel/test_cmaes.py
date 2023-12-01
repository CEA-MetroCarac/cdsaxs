import numpy as np
from fit import stacked_trapezoids
from fit_parallel import cmaes
import os


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

multiples = [1E-8, 1E-8, 1E-8, 1E-7, 1E-7, 1E-7] + len(swa) * [1E-5]

# Check if the number of initial guesses matches the number of multiples
# assert len(initial_guess) == len(multiples), f'Number of adds ({len(initial_guess)}) is different from number of multiples ({len(multiples)})'

# Define data arrays
data = data
qxs = qxs
qzs = qzs

# Generate data based on Fourier transform of arbitrary parameters using stacked_trapezoids
def generate_arbitrary_data():
    arbitrary_params = np.array([dwx, dwz, i0, bkg, height, bot_cd] + swa)
    data = stacked_trapezoids(qxs, qzs, y1=0, y2=bot_cd, height=height, langle=np.asarray(swa), weight=None)
    return data, arbitrary_params

def test_cmaes_with_arbitrary_data():
    # Generate arbitrary data and parameters
    data, arbitrary_params = generate_arbitrary_data()



    # Call the cmaes function with arbitrary data
    if(__name__ == "__main__"):
        best_corr, best_fitness = cmaes(data=data, qxs=qxs, qzs=qzs, sigma=100, ngen=30, popsize=100, mu=10,
                                                    n_default=len(arbitrary_params), multiples=multiples, restarts=0, verbose=False, tolhistfun=5e-5,
                                                    initial_guess=arbitrary_params, ftarget=None, dir_save=None)

    # Compare the obtained parameters with the arbitrary ones within a tolerance
    tolerance = 0.02  # Adjust the tolerance as needed
    assert np.allclose(best_corr, arbitrary_params, rtol=tolerance), "Test failed!"

    print("Test passed successfully!")

# Run the test
test_cmaes_with_arbitrary_data()
