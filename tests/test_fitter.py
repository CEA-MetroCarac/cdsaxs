import pytest
import numpy as np
import pandas as pd
import os
import pandas.testing as pdt
from cdsaxs.simulations import stacked_trapezoid as simulation
from cdsaxs import fitter


def test_fitter_initialization(fitter_instance, simulate_intensities):
    assert isinstance(fitter_instance, fitter.Fitter)
    assert fitter_instance.exp_data is simulate_intensities
    assert fitter_instance.best_fit_cmaes is None

def average_dataframe(dfs):
    return sum(dfs) / len(dfs)

def test_cmaes(fitter_instance, params):
    calculated_fits = []
    calculated_fitnesses = []

    for _ in range(10):
        calculated_fit, calculated_fitness = fitter_instance.cmaes(
            sigma=100, ngen=100, popsize=20, mu=10, n_default=9, 
            restarts=10, tolhistfun=10, ftarget=10, 
            restart_from_best=True, verbose=False
        )
        calculated_fits.append(calculated_fit)
        calculated_fitnesses.append(calculated_fitness)
    
    # Calculate the average of the results
    avg_calculated_fit = average_dataframe(calculated_fits)
    avg_calculated_fitness = np.mean(calculated_fitnesses)
    
    geometry = simulation.StackedTrapezoidGeometry()
    expected_fit = geometry.convert_to_dataframe(params)
    
    pdt.assert_frame_equal(avg_calculated_fit, expected_fit, atol=1.0)
    assert isinstance(avg_calculated_fitness, float)


def test_mcmc(fitter_instance, params):
    geometry = simulation.StackedTrapezoidGeometry()
    best_fit = geometry.convert_to_dataframe(params)
    fitter_instance.set_best_fit_cmaes(best_fit=best_fit)

    calculated_fits = []
    calculated_fitnesses = []

    for _ in range(10):
        calculated_fit, calculated_fitness = fitter_instance.mcmc_bestfit_stats(
            N=9, sigma=np.asarray([100] * 9), nsteps=50, nwalkers=70, test=True
        )
        calculated_fits.append(calculated_fit)
        calculated_fitnesses.append(calculated_fitness)

    # Calculate the average of the results
    avg_calculated_fit = average_dataframe(calculated_fits)
    avg_calculated_fitness = np.mean(calculated_fitnesses)

    expected_fit = best_fit
    pdt.assert_frame_equal(avg_calculated_fit, expected_fit, atol=1.0)
    assert isinstance(avg_calculated_fitness, float)

def test_do_stats():
    data = {
        'param1': np.random.normal(0, 1, 1000),
        'param2': np.random.normal(5, 2, 1000)
    }
    df = pd.DataFrame(data)
    
    stats = fitter.Fitter.do_stats(df)
    
    assert isinstance(stats, pd.DataFrame)
    assert set(stats.columns) == {'mean', 'std', 'count', 'min', 'max', 'lower_ci', 'upper_ci', 'uncertainity'}
    assert np.allclose(stats.loc['param1', 'mean'], 0, atol=0.5)
    assert np.allclose(stats.loc['param2', 'mean'], 5, atol=0.5)