import sys
import os

path = os.path.dirname("/nobackup/nd276333/Workspace/cdsaxs/package/")
sys.path.append(path)

import cdsaxs.package.src.cdsaxs.fitter as fitter
import numpy as np

def test_cmaes(fitter_instance, params):

    calculated_fit, calculated_fitness  =fitter_instance.cmaes(sigma=100, ngen=100, popsize=20, mu=10, n_default=9, restarts=10, tolhistfun=10, ftarget=10, restart_from_best=True, verbose=False)

    #make an array from the dictionaries calculated_fit and params
    calculated = []
    expected = []
    for key in calculated_fit.keys():
        if key == 'heights'or key == 'langles' or key == 'rangles':  
            calculated.append(params[key])
        else:
            calculated.append([params[key]])
        
        expected.append(calculated_fit[key])

    np.testing.assert_allclose(calculated, expected, atol=0.1)


def test_mcmc(fitter_instance, params):

    fitter_instance.set_best_fit_cmaes(params)
    
    calculated_fit = fitter_instance.mcmc(N=9, sigma = np.asarray([100] * 9), nsteps=1000, nwalkers=18, test=True)

    #make an array from the dictionaries calculated_fit and params
    calculated = []
    expected = []
    for key in calculated_fit.keys():
        if key == 'heights'or key == 'langles' or key == 'rangles':  
            calculated.append(params[key])
        else:
            calculated.append([params[key]])
        
        expected.append(calculated_fit[key])

    np.testing.assert_allclose(calculated, expected, atol=1.0)
    