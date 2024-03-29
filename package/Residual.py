import numpy as np
import cupy as cp
import os
from cdsaxs.package.Simulation import TrapezoidGeometry



class PicklableResidual:
    """
    Factory created to call the residual function (which need to be pickled)
    in the MCMC andCMAES approach. This factory will allow to have the data,
    qx, qz, initial_guess, fit_mode to co;pare with the set of parameters
    returned by cmaes
    """

    def __init__(self, data, qxs, qzs, multiples, initial_guess, fit_mode='cmaes', test=False, xp=np, trapezoidGeometry=None):
        """
        Parameters
        ----------
        data, qxs, qzs: np.arrays of floats
            List of intensity/qx/qz at which the form factor has to be simulated
        initial_guess: list of floats
            List of the initial_guess of the user
        fit_mode: string
            Method to calculate the fitness, which is different between cmaes
            and mcmc
        """
        self.mdata = data
        self.mqz = qzs
        self.mqx = qxs
        self.multiples = multiples
        self.minitial_guess = initial_guess
        self.mfit_mode = fit_mode
        self.test = test
        self.xp = xp
        self.trapezoidGeometry = trapezoidGeometry

    def __call__(self, fit_params):
        """
        Parameters
        ----------
        fit_params: numpy or cupy arrays of arrays of parametres of floats
            List of all the parameters value returned by CMAES/MCMC (list of
            Debye-Waller, I0, noise level, height, linewidth, [angles......])

        Returns
        -------

        """
        if not isinstance(fit_params, self.xp.ndarray):
            fit_params = self.xp.array(fit_params)
        
        if self.test:

            simp = fit_params

        else:
            simp = TrapezoidGeometry.fittingp_to_simp(fit_params, initial_guess=self.minitial_guess, multiples=self.multiples)

        simp = self.xp.asarray(simp)

        dwx, dwz, intensity0, bkg, height, botcd, beta = simp[:,0], simp[:,1], simp[:,2], simp[:,3], simp[:,4], simp[:,5], self.xp.array(simp[:,6:])#modified for gpu so that each variable is a list of arrays
        
        langle = self.xp.deg2rad(self.xp.asarray(beta))
        rangle = self.xp.deg2rad(self.xp.asarray(beta))

        qxfit = TrapezoidGeometry.stacked_trapezoids(self.mqx, self.mqz, self.xp.zeros(botcd.shape), botcd, height, langle, rangle)

        qxfit = TrapezoidGeometry.corrections_dwi0bk(qxfit, dwx, dwz, intensity0, bkg, self.mqx, self.mqz)


        res = self.log_error(self.mdata, qxfit)

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
        # error /= xp.count_nonzero(~xp.isnan(exp_i_array), axis=1)

        #replace the error of the population with inf if all the parameters are nan
        error = self.xp.where(self.xp.all(self.xp.isnan(sim_i_array), axis=1), self.xp.inf, error)

        return error

    def fix_fitness_mcmc(fitness):
        """
        Metropolis-Hastings criterion: acceptance probability equal to ratio between P(new)/P(old)
        where P is proportional to probability distribution we want to find
        for our case we assume that probability of our parameters being the best is proportional to a Gaussian centered at fitness=0
        where fitness can be log, abs, squared error, etc.
        emcee expects the fitness function to return ln(P(new)), P(old) is auto-calculated
        """
        c = 1e-5  # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
        return -fitness / c


