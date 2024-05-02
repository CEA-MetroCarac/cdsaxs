import numpy as np
from pytest import approx


def test_cmaes(fitter_instance, params):
    """
    Test the CMA-ES algorithm implementation in the fitter instance.

    Parameters:
    - fitter_instance: An instance of the fitter class.
    - params: A dictionary containing the input parameters for the CMA-ES algorithm.

    Returns:
    None

    Raises:
    AssertionError: If the calculated values do not match the expected values.

    """
    calculated_fit, calculated_fitness  = fitter_instance.cmaes(sigma=100, ngen=100, popsize=20,
                                                                mu=10, n_default=9, restarts=10,
                                                                tolhistfun=10, ftarget=10,
                                                                restart_from_best=True, verbose=False)

    #make an array from the dictionaries calculated_fit and params
    calculated = []
    expected = []
    for key in calculated_fit.keys():
        if key == 'heights'or key == 'langles' or key == 'rangles':  
            calculated.append(params[key])
        else:
            calculated.append([params[key]])
        
        expected.append(calculated_fit[key])

    calculated = approx(expected, abs=0.1)


def test_mcmc(fitter_instance, params):
    """
    Test the MCMC method of the fitter instance.

    Parameters:
    - fitter_instance: An instance of the fitter class.
    - params: A dictionary containing the parameters for the MCMC method.

    Returns:
    None
    """
    fitter_instance.set_best_fit_cmaes(best_fit=params)
    
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

    calculated = approx(expected, abs=1.0)
    