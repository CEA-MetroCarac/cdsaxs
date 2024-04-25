import sys
import os

import cdsaxs_fitting.simulations.stacked_trapezoid_simulation as simulation
import cdsaxs_fitting.fitter as fitter
import numpy as np
import pytest

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 10)
qys = 2 * np.pi / pitch * np.ones_like(qzs)

# Initial parameters
dwx = 0.1
dwz = 0.1
i0 = 10
bkg = 0.1
y1 = 0.
height = [20.]
bot_cd = 40.
swa = [90.]

langle = np.deg2rad(np.asarray(swa))
rangle = np.deg2rad(np.asarray(swa))

# Make the parameters available to other tests
@pytest.fixture(scope="session")
def params():
    params = {
        'heights': height,
        'langles': langle,
        'rangles': rangle,
        'y1': y1,
        'bot_cd': bot_cd,
        'dwx': dwx,
        'dwz': dwz,
        'i0': i0,
        'bkg_cste': bkg
    }

    return params



@pytest.fixture
def simulate_intensities(params):
    
    stacked_trapezoid = simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys)
    
    return stacked_trapezoid.simulate_diffraction(params=params)


@pytest.fixture
def fitter_instance(simulate_intensities):
    initial_params = {'heights': {'value': height, 'variation': 10E-5},
                    'langles': {'value': langle, 'variation': 10E-5},
                    'rangles': {'value': rangle, 'variation': 10E-5},
                    'y1': {'value': y1, 'variation': 10E-5},
                    'bot_cd': {'value': bot_cd, 'variation': 10E-5},
                    'dwx': {'value': dwx, 'variation': 10E-5},
                    'dwz': {'value': dwz, 'variation': 10E-5},
                    'i0': {'value': i0, 'variation': 10E-5},
                    'bkg_cste': {'value': bkg, 'variation': 10E-5}
                    }
    
    stacked_trapezoid = simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys, initial_guess=initial_params)

    fitter_instance = fitter.Fitter(stacked_trapezoid, exp_data=simulate_intensities)

    return fitter_instance
