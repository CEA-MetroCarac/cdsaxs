import cdsaxs.simulations.stacked_trapezoid as simulation
import cdsaxs.fitter as fitter
import numpy as np
import pytest


@pytest.fixture(scope="session")
def qzs_qys():
    pitch = 100 #nm distance between two trapezoidal bars
    qzs = np.linspace(-0.1, 0.1, 10)
    qys = 2 * np.pi / pitch * np.ones_like(qzs)

    return qzs, qys

# Make the parameters available to other tests
@pytest.fixture(scope="session")
def params():

    # Initial parameters
    dwx = 0.1
    dwz = 0.1
    i0 = 10.
    bkg = 0.1
    y1 = 0.
    height = [20.]
    bot_cd = 40.
    swa = [90.]

    langle = np.deg2rad(np.asarray(swa))
    rangle = np.deg2rad(np.asarray(swa))

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
def multi_params(params):

    #simulation data
    params_height_constant = {
        'heights': params['heights'][0],
        'langles': params['langles'],
        'rangles': params['rangles'],
        'y1': params['y1'],
        'bot_cd': params['bot_cd'],
        'dwx': params['dwx'],
        'dwz': params['dwz'],
        'i0': params['i0'],
        'bkg_cste': params['bkg_cste']
    }

    params_height_variable = {
        'heights': params['heights'],
        'langles': params['langles'],
        'rangles': params['rangles'],
        'y1': params['y1'],
        'bot_cd': params['bot_cd'],
        'dwx': params['dwx'],
        'dwz': params['dwz'],
        'i0': params['i0'],
        'bkg_cste': params['bkg_cste']
    }
    
    params_without_rangles = {
        'heights': params['heights'][0],
        'langles': params['langles'],
        'y1': params['y1'],
        'bot_cd': params['bot_cd'],
        'dwx': params['dwx'],
        'dwz': params['dwz'],
        'i0': params['i0'],
        'bkg_cste': params['bkg_cste']
    }
    
    params = [params_height_constant, params_height_variable, params_without_rangles]
    
    return params

@pytest.fixture
def simulate_intensities(params, qzs_qys):
    qzs, qys = qzs_qys
    stacked_trapezoid = simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys)
    
    return stacked_trapezoid.simulate_diffraction(params=params)

@pytest.fixture
def initial_params(params):
    iparams = {'heights': {'value': params['heights'], 'variation': 10E-5},
                    'langles': {'value': params['langles'], 'variation': 10E-5},
                    'rangles': {'value': params['rangles'], 'variation': 10E-5},
                    'y1': {'value': params['y1'], 'variation': 10E-5},
                    'bot_cd': {'value': params['bot_cd'], 'variation': 10E-5},
                    'dwx': {'value': params['dwx'], 'variation': 10E-5},
                    'dwz': {'value': params['dwz'], 'variation': 10E-5},
                    'i0': {'value': params['i0'], 'variation': 10E-5},
                    'bkg_cste': {'value': params['bkg_cste'], 'variation': 10E-5}
                    }
    return iparams

@pytest.fixture
def fitter_instance(simulate_intensities, initial_params, qzs_qys):
    qzs, qys = qzs_qys

    stacked_trapezoid = simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys, initial_guess=initial_params)

    fitter_instance = fitter.Fitter(stacked_trapezoid, exp_data=simulate_intensities)

    return fitter_instance
