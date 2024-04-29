import os

try:
    import cupy as cp
except:
    import numpy as np

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .simulations.base import Simulation



class Residual:
    """
    Class to calculate the residual between the experimental data and the
    simulated data.
    """

    def __init__(self, data, fit_mode='cmaes', xp=np, Simulation: Optional['Simulation'] = None, c=1e-5, best_fit=None):
        """
        Parameters
        ----------
        data, qxs, qzs: np.arrays of floats
            List of experimental intensity data and qx, qz at which the form factor has to be simulated
        fit_mode: string
            Method to calculate the fitness, which is different between cmaes
            and mcmc
        xp: module
            Numpy or cupy
        Simulation: class
            Class to simulate the diffraction pattern (for now only StackedTrapezoidSimulation)
        c: float
            Empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual,
            higher c increases acceptance rate
        """
        self.mdata = data
        self.mfit_mode = fit_mode
        self.xp = xp
        self.Simulation = Simulation
        self.c = c
        self.best_fit = best_fit


    def __call__(self, fit_params):
        """
        Parameters
        ----------
        fit_params: numpy or cupy arrays of arrays of parametres of floats
            List of all the parameters value returned by CMAES/MCMC (list of
            Debye-Waller, I0, noise level, height, linewidth, [angles......])
            (not be given by the user if test=False, unexpected behavior occurs if not obtained from Fitter class)

        Returns
        -------

        """
        if not isinstance(fit_params, self.xp.ndarray):
            fit_params = self.xp.array(fit_params)
        if fit_params is not None and self.best_fit is not None:
            qxfit = self.Simulation.simulate_diffraction(fitparams=fit_params, fit_mode=self.mfit_mode, best_fit=self.best_fit)
        elif fit_params is not None:
            qxfit = self.Simulation.simulate_diffraction(fitparams=fit_params, fit_mode=self.mfit_mode)

        res = self.log_error(self.mdata, qxfit)
        
        if self.xp == cp:
            res = res.get()

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
        error: float
            Sum of difference of log data, normalized by the number of data
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
        """
        # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
        return -fitness / self.c