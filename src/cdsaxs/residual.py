from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .simulations.base import Simulation
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = np



class Residual:
    """
    A class to calculate the residual between experimental and simulated data, used for fitness evaluation in optimization algorithms.

    Attributes:
        mdata : numpy.ndarray
            Experimental intensity data.
        mfit_mode : str
            Method to calculate fitness, differentiating between 'cmaes' and 'mcmc'.
        xp : module
            NumPy or CuPy module.
        Simulation : Optional['Simulation']
            Class to simulate the diffraction pattern (for now only StackedTrapezoidSimulation).
        c : float
            Empirical factor to modify the MCMC acceptance rate.
        best_fit : list or None
            List containing the best fit parameters obtained from the optimization algorithm (optional).

    Methods:
        __call__: Calculate the residual between experimental and simulated data.
        log_error: Return the difference between experimental and simulated data using the log error.
        fix_fitness_mcmc: Fix the fitness for the MCMC algorithm using the Metropolis-Hastings criterion.
    """

    def __init__(self, data, fit_mode='cmaes', xp=np, Simulation: Optional['Simulation'] = None, c=1e-5, best_fit=None):
        """

        Parameters:
            data : numpy.ndarray
                Experimental intensity data.
            fit_mode : str, optional
                Method to calculate fitness, differentiating between 'cmaes' and 'mcmc'. Default is 'cmaes'.
            xp : module, optional
                NumPy or CuPy module. Default is numpy.
            Simulation : Optional['Simulation'], optional
                Class to simulate the diffraction pattern (for now only StackedTrapezoidSimulation). Default is None.
            c : float, optional
                Empirical factor to modify the MCMC acceptance rate. Default is 1e-5.
            best_fit : list or None, optional
                List containing the best fit parameters obtained from the optimization algorithm. Default is None.
        """

        self.mdata = data
        self.mfit_mode = fit_mode
        self.xp = xp
        self.Simulation = Simulation
        self.c = c
        self.best_fit = best_fit


    def __call__(self, fit_params):
        """
        Calculate the residual between experimental and simulated data.

        Parameters:
            fit_params : numpy.ndarray
                population of all the variation values generated by deap ranging from -sigma to +sigma.

        Returns:
            numpy.ndarray or float
                Residual value(s) between experimental and simulated data.
        """
        if not isinstance(fit_params, self.xp.ndarray):
            fit_params = self.xp.array(fit_params)
        if fit_params is not None and self.best_fit is not None:
            qxfit = self.Simulation.simulate_diffraction(fitparams=fit_params, fit_mode=self.mfit_mode, best_fit=self.best_fit)
        elif fit_params is not None:
            qxfit = self.Simulation.simulate_diffraction(fitparams=fit_params, fit_mode=self.mfit_mode)

        res = self.log_error(self.mdata, qxfit)
        
        try:
            if self.xp == cp:
                res = res.get()
        except:
            pass

        if self.mfit_mode == 'cmaes':
            return res
        
        elif self.mfit_mode == 'mcmc':
            return self.fix_fitness_mcmc(res)

        else:
            raise ValueError("Invalid fit mode. Please choose either 'cmaes' or 'mcmc'.")


    def log_error(self, exp_i_array, sim_i_array):
        """
        Return the difference between two set of values (experimental and
        simulated data), using the log error

        Parameters
        ----------
        exp_i_array: numpy.ndarray((n))
            Experimental intensities data
        sim_i_array: numpy.ndarray((n))
            Simulated intensities data

        Returns
        -------
        Returns:
            numpy.ndarray
                Difference between experimental and simulated data using the log error.
        """
        exp_i_array = self.xp.where(exp_i_array < 0, self.xp.nan, exp_i_array)
        
        exp_i_array = exp_i_array[self.xp.newaxis, ...]
        exp_i_array = exp_i_array * self.xp.ones_like(sim_i_array)

        error = self.xp.nansum(self.xp.abs((self.xp.log10(exp_i_array) - self.xp.log10(sim_i_array))), axis=1)

        #this is for normalization but we don't get the same results as the original code
        # error /= self.xp.count_nonzero(~self.xp.isnan(exp_i_array), axis=1)

        #replace the error of the population with inf if all the parameters are nan
        error = self.xp.where(self.xp.all(self.xp.isnan(sim_i_array), axis=1), self.xp.inf, error)

        return error

    def fix_fitness_mcmc(self, fitness):
        """
        Metropolis-Hastings criterion: acceptance probability equal to ratio between P(new)/P(old)
        where P is proportional to probability distribution we want to find
        for our case we assume that probability of our parameters being the best is proportional to a Gaussian centered at fitness=0
        where fitness can be log, abs, squared error, etc.
        emcee expects the fitness function to return ln(P(new)), P(old) is auto-calculated

        Parameters:
            fitness : float
                Fitness value to be fixed.

        Returns:
            float
                Fixed fitness value.
        """
        # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
        return -fitness / self.c