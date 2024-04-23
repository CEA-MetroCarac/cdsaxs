from code_parallel import fit_parallel as par
from code_gpu import fit_parallel_vect as vect
import os
import cupy as cp
import numpy as np
import pytest

use_gpu = False

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 1)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)

# Define initial parameters and multiples
dwx = [0.1]
dwz = [0.1]
i0 = 10
bkg = 0.1
height = [23.48, 23.45]
bot_cd = [54.6, 54.2]
swa = [[85, 83, 72],[85, 83, 72]]

multiples = [1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7] + len(swa[0]) * [1E-9]

if use_gpu:
    qxs = cp.array(qxs)
    qzs = cp.array(qzs)
    multiples = cp.array(multiples)

#params is initial parametre 
params = np.array([dwx[0], dwz[0], i0, bkg, height[0], bot_cd[0]] + swa[0])

simp = [[0.10000054, 0.09999963, 9.99999986, 0.10000013, 23.47999941, 54.59999988, 86.99999991, 85.00000006, 82.99999947, 72.00000049],
        [np.nan, 0.09999977, 9.99999986, 0.10000054, 23.47999941, 54.60000064, 86.99999995, 85.00000031, 83.00000013, 71.99999971],
        [0.10000061, 0.09999939, 9.99999912, 0.09999999, 23.47999898, 54.59999983, 87.00000028, 85.00000013, 82.99999965, 72.00000043],
        [0.10000034, 0.09999946, 9.99999919, 0.09999987, 23.47999883, 54.60000018, 87.00000067, 85.00000015, 82.99999919, 72.00000046],
        [0.10000053, 0.0999996, 9.99999935, 0.10000035, np.nan, 54.60000026, 87.00000064, 85.00000029, 82.99999934, 72.00000022],
        [0.10000044, 0.09999972, 9.99999932, 0.10000013, 23.47999836, 54.60000005, 87.00000044, 85.00000054, 82.99999949, 72.0000006],
        [0.10000016, 0.09999978, 9.99999943, 0.10000022, 23.47999868, 54.60000028, 86.99999997, 84.99999996, 82.99999972, 72.00000022],
        [0.10000012, 0.09999941, 9.99999934, 0.09999986, 23.47999907, 54.60000013, 87.00000005, 85.00000013, 82.99999948, 71.99999955]]

# Generate data based on Fourier transform of arbitrary parameters using stacked_trapezoids
def generate_arbitrary_data(qxs, qzs):
    arbitrary_params = np.array([dwx[0], dwz[0], i0, bkg, height[0], bot_cd[0]] + swa[0])
    langle = np.deg2rad(np.asarray(swa))
    rangle = np.deg2rad(np.asarray(swa))

    # if ~use_gpu:
    #     qxs = qxs.get()
    #     qzs = qzs.get()


    #generate data for vectorised version
    data_vect = vect.stacked_trapezoids(qxs, qzs, y1=np.asarray([0,0]), y2=np.asarray(bot_cd), height=np.asarray(height), langle=np.asarray(langle))
    data_vect = vect.corrections_dwi0bk(data_vect, dwx, dwz, i0, bkg, qxs, qzs)
    

    #generate data for parallel version
    data_par = par.stacked_trapezoids(qxs, qzs, y1=0, y2=bot_cd[0], height=height[0], langle=langle[0])
    data_par = par.corrections_dwi0bk(data_par, dwx[0], dwz[0], i0, bkg, qxs, qzs)

    return data_vect, data_par

def test_PickeableResidual():
    data_vect, data_par = generate_arbitrary_data(qxs, qzs)
    global simp
    simp = np.asarray(simp)
    
    data_vect = np.abs(np.asarray(data_vect))
    data_par = np.abs(np.asarray(data_par))

    residual_vect = vect.PickeableResidual( data= data_vect[0], qxs=qxs, qzs=qzs, multiples=multiples, initial_guess=params, fit_mode="mcmc", test=True)
    residual_par = par.PickeableResidual( data= data_par, qxs=qxs, qzs=qzs, multiples=multiples, initial_guess=params, fit_mode="mcmc", test=True)

    #call the residual functions 
    res_vect = residual_vect(simp)
    
    #the parallel version takes one set of parameters at a time and is not designed to deal with nans hence this approach
    res_par = [residual_par(x) for x in simp]
    
    np.testing.assert_allclose(res_vect, res_par, rtol=1e-3, atol=0.0001)
    print("test passed")
    # print("res_vect: ", res_vect, "res_par: ", res_par)

test_PickeableResidual()

