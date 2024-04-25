import sys
import os

import cdsaxs_fitting.simulations.stacked_trapezoid as simulation
import numpy as np
import pytest

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 10)
qys = 2 * np.pi / pitch * np.ones_like(qzs)


@pytest.fixture
def stacked_trapezoid():
    return simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys)

@pytest.fixture
def get_params():
    #Initial parameters
    dwx = 0.1
    dwz = 0.1
    i0 = 10
    bkg = 0.1
    y1 = 0
    height = [20.]
    bot_cd = 40
    swa = [90]

    langle = np.deg2rad(np.asarray(swa))
    rangle = np.deg2rad(np.asarray(swa))

    #simulation data
    params_height_constant = {'heights': height[0],
                'langles': langle,
                'rangles': rangle,
                'y1': y1,
                'bot_cd': bot_cd,
                'dwx': dwx,
                'dwz': dwz,
                'i0': i0,
                'bkg_cste': bkg
                }
    
    params_height_variable = {'heights': height,
                'langles': langle,
                'rangles': rangle,
                'y1': y1,
                'bot_cd': bot_cd,
                'dwx': dwx,
                'dwz': dwz,
                'i0': i0,
                'bkg_cste': bkg
                }
    
    params_without_rangles = {'heights': height,
                'langles': langle,
                'y1': y1,
                'bot_cd': bot_cd,
                'dwx': dwx,
                'dwz': dwz,
                'i0': i0,
                'bkg_cste': bkg
                }
    
    params = [params_height_constant, params_height_variable, params_without_rangles]
    
    return params