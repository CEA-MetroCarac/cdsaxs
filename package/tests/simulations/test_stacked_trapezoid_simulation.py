import sys
import os

import cdsaxs_fitting.simulations.stacked_trapezoid as simulation
import numpy as np
import pytest


def test_simulate_diffraction(stacked_trapezoid, get_params):
    expected_intensities = [2595837.42097937, 2983810.93619292, 3303845.79775438, 3531958.65370197,
                            3650634.41543888, 3650634.41543888, 3531958.65370197, 3303845.79775438,
                            2983810.93619292, 2595837.42097937]

    for param in get_params:
        calculated_intensities = stacked_trapezoid.simulate_diffraction(param)
        np.testing.assert_allclose(calculated_intensities, expected_intensities, rtol=1e-2)


def test_extract_params(stacked_trapezoid, get_params):

    dwx = 0.1
    dwz = 0.1
    i0 = 10.
    bkg = 0.1
    y1 = 0.
    height = [20.]
    bot_cd = 40.
    swa = [90]

    langle = np.deg2rad(np.asarray(swa))
    rangle = np.deg2rad(np.asarray(swa))

    expected_params = (np.asarray([height]), np.asarray([langle]), np.asarray([rangle]), np.asarray(y1), 
                       np.asarray(bot_cd), None, np.asarray(dwx), np.asarray(dwz), np.asarray(i0), np.asarray(bkg))
    

    extracted_params = stacked_trapezoid.TrapezoidGeometry.extract_params(params=get_params[1])
    
    assert extracted_params == expected_params




    
        

    
    


