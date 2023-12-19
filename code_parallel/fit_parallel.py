# -*- coding: utf-8 -*-

"""
Performs model inverse resolution from cut inside QxQzI reciprocal map to
obtained 3D line profile
Based on trapezoïds line model and literal fourier transformed (extracted from
XiCam an open source software

Code developed by Jerome Reche and Vincent Gagneur.
"""
import os
from collections import deque
import numpy as np
import pandas as pd
from random import randrange
import deap.base as dbase
from deap import creator, tools, cma
from scipy import stats
import scipy.interpolate
import emcee
import sys
import multiprocessing as mp
import fit as original_fit

def cmaes(data, qxs, qzs, initial_guess, multiples, sigma, ngen,
          popsize, mu, n_default, restarts, tolhistfun, ftarget,
          restart_from_best=True, verbose=True, dir_save=None):
    """
    Modified from deap/algorithms.py to return population_list instead of
    final population and use additional termination criteria (algorithm
    neuromorphic)
    Function extracted from XiCam (modified)

    Parameters
    ----------
    data: np.arrays of float32
        Intensities to fit
    qxs, qzs: list of floats
        List of qx/qz linked to intensities
    initial_guess: list of float32
        Values entered by the user as starting point for the fit (list of
        Debye-Waller, I0, noise level, height, linewidth, [angles......])
    sigma: float
        Initial standard deviation of the distribution
    ngen: int
        Number of generation maximum
    popsize: int
        Size of population used at each loop
    mu: ??
        TODO: investigate real impact
    n_default: int
        integer ussed to define size of default parameters
        TODO: investigate real impact
    restarts: int
        Number of time fitting must be restart
    tolhistfun: float
        Tolerance of error fit (allow to stop fit when difference between to
        successive fit is lower)
    ftarget: float
        Stop condition if error is below TODO: to confirm
    restart_from_best: bool, optional
        Next fitting restart with used of previous fitting values results
    verbose: bool, optional
        Verbose mode print more information during fitting
    dir_save: str, optional
        Directory pathname for population and fitness arrays saving in a
        'output.xlx' file. If None, saving is not done.

    Returns
    -------
    best_corr: ??
        Parameters of line obtain after the fit TODO: to confirm
    best_fitness: ??
        error obtain on fit TODO: to confirm
    """

    if dir_save is not None:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)

    if verbose:
        print("Start CMAES")
    toolbox = dbase.Toolbox()
    residual = PickeableResidual(data, qxs, qzs, initial_guess=initial_guess, multiples=multiples,
                                 fit_mode='cmaes')
    
    toolbox.register('evaluate', residual)
    
    #register parallel map function to toolbox
    parallel = 4#mp.cpu_count()
    pool = mp.Pool(parallel)
    toolbox.register('map', pool.map)

    halloffame = tools.HallOfFame(1)

    thestats = tools.Statistics(lambda ind: ind.fitness.values)
    thestats.register('avg', lambda x: np.mean(np.asarray(x)[np.isfinite(x)]) \
        if np.asarray(x)[np.isfinite(x)].size != 0 else None)
    thestats.register('std', lambda x: np.std(np.asarray(x)[np.isfinite(x)]) \
        if np.asarray(x)[np.isfinite(x)].size != 0 else None)
    thestats.register('min', lambda x: np.min(np.asarray(x)[np.isfinite(x)]) \
        if np.asarray(x)[np.isfinite(x)].size != 0 else None)
    thestats.register('max', lambda x: np.max(np.asarray(x)[np.isfinite(x)]) \
        if np.asarray(x)[np.isfinite(x)].size != 0 else None)
    thestats.register('fin', lambda x: np.sum(np.isfinite(x)) / np.size(x))

    # thestats.register('cumtime', lambda x: time.perf_counter() - last_time)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (thestats.fields if thestats else [])
    population_list = []
    popsize_default = int(4 + 3 * np.log(n_default))
    kwargs = {'lambda_': popsize if popsize is not None else popsize_default}
    if mu is not None:
        kwargs['mu'] = mu
    initial_individual = [0] * n_default

    morestats = {}
    morestats['sigma_gen'] = []
    morestats['axis_ratio'] = []  # ratio of min and max scaling at each gener.
    morestats['diagD'] = []  # scaling of each param. (eigenval of covar matrix)
    morestats['ps'] = []

    allbreak = False

    for restart in range(restarts + 1):
        if allbreak:
            break
        if restart != 0:
            kwargs['lambda_'] *= 2
            print('Doubled popsize')
            if restart_from_best:
                initial_individual = halloffame[0]

        # type of strategy: (parents, children) = (mu/mu_w, popsize), selection
        # takes place among offspring only
        strategy = cma.Strategy(centroid=initial_individual, sigma=sigma,
                                **kwargs)

        # The CMA-ES One Plus Lambda algo takes a initialized parent as argument
        #   parent = creator.Individual(initial_individual)
        #   parent.fitness.values = toolbox.evaluate(parent)
        #   strategy = cmaes.StrategyOnePlusLambda(parent=parent,
        #                                          sigma=sigma, lambda_=popsize)

        lambda_ = kwargs['lambda_']
        toolbox.register('generate', strategy.generate, creator.Individual)
        toolbox.register('update', strategy.update)
        maxlen = 10 + int(np.ceil(30 * n_default / lambda_))
        last_best_fitnesses = deque(maxlen=maxlen)
        cur_gen = 0
        # fewer generations when popsize is doubled
        # (unless fixed ngen is specified)

        ngen_default = int(100 + 50 * (n_default + 3) ** 2 / lambda_ ** 0.5)
        ngen_ = ngen if ngen is not None else ngen_default
        msg = "Iteration terminated due to {} criterion after {} gens"
        while cur_gen < ngen_:
            cur_gen += 1
            # Generate a new population
            population = toolbox.generate()
            population_list.append(population)

            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)

            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)  # tuple of length 1
            halloffame.update(population)
            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            record = thestats.compile(population) if stats is not None else {}
            logbook.record(gen=cur_gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            axis_ratio = max(strategy.diagD) ** 2 / min(strategy.diagD) ** 2
            morestats['sigma_gen'].append(strategy.sigma)
            morestats['axis_ratio'].append(axis_ratio)
            morestats['diagD'].append(strategy.diagD ** 2)
            morestats['ps'].append(strategy.ps)

            last_best_fitnesses.append(record['min'])
            if (ftarget is not None) and record['min'] <= ftarget:
                if verbose:
                    print(msg.format("ftarget", cur_gen))
                allbreak = True
                break
            if last_best_fitnesses[-1] is None:
                last_best_fitnesses.pop()
                pass
            # print(last_best_fitnesses)
            delta = max(last_best_fitnesses) - min(last_best_fitnesses)
            cond1 = tolhistfun is not None
            cond2 = len(last_best_fitnesses) == last_best_fitnesses.maxlen
            cond3 = delta < tolhistfun
            if cond1 and cond2 and cond3:
                print(msg.format("tolhistfun", cur_gen))
                break
        else:
            print(msg.format("ngen", cur_gen))

    best_uncorr = halloffame[0]  # np.abs(halloffame[0])
    best_fitness = halloffame[0].fitness.values[0]
    best_corr = fittingp_to_simp(best_uncorr, initial_guess=initial_guess, multiples=multiples)
    if verbose:
        print(('best', best_corr, best_fitness))
    # make population dataframe, order of rows is first generation for all
    # children, then second generation for all children...
    population_arr = np.array(
        [list(individual) for generation in population_list for individual in
         generation])

    population_arr = fittingp_to_simp1(population_arr, initial_guess=initial_guess, multiples=multiples)

    fitness_arr = np.array(
        [individual.fitness.values[0] for generation in population_list for
         individual in generation])

    population_fr = pd.DataFrame(np.column_stack((population_arr, fitness_arr)))
    if dir_save is not None:
        population_fr.to_excel(os.path.join(dir_save, "output.xlsx"))

    return best_corr, best_fitness

def mcmc(data, qxs, qzs, initial_guess, N, multiples, sigma, nsteps, nwalkers, gaussian_move=False, parallel=True, seed=None, verbose=True, test=False):
    """
    Fit data using the emcee package's implementation of the MCMC algorithm.

    Args:
        data (numpy.ndarray): Experimental data.
        qxs (numpy.ndarray): Q-values for x-direction.
        qzs (numpy.ndarray): Q-values for z-direction.
        initial_guess (numpy.ndarray): Initial parameter guess.
        N (int): Number of parameters.
        sigma (float or list): Initial standard deviation for each parameter.
        nsteps (int): Number of MCMC steps.
        nwalkers (int): Number of MCMC walkers.
        gaussian_move (bool, optional): Use Metropolis-Hastings gaussian proposal. Default is strech move.
        parallel (bool or int or str, optional): Set the parallel processing mode. Default is True.
        seed (int, optional): Seed for the random number generator.
        verbose (bool, optional): Print progress information. Default is True.

    Returns:
        None

    Attributes:
        best_uncorr (numpy.ndarray): Best uncorrected individual.
        best_fitness (float): Best fitness value.
        minfitness_each_gen (numpy.ndarray): Minimum fitness at each generation.
        Sampler (emcee.EnsembleSampler): Instance of emcee.Sampler with detailed output of the algorithm.
    """
    
    # Create a PickeableResidual instance for data fitting
    residual = PickeableResidual(data=data, qxs=qxs, qzs=qzs, initial_guess=initial_guess, multiples=multiples, fit_mode='mcmc')
    
    def do_verbose(Sampler):
        if hasattr(Sampler, 'acceptance_fraction'):
            print('Acceptance fraction: ' + str(np.mean(Sampler.acceptance_fraction)))
        else:
            print('Acceptance fraction: ' + str(np.mean([Sampler.acceptance_fraction for Sampler in Sampler])))
        sys.stdout.flush()
    
    # Empirical factor to modify MCMC acceptance rate
    c = 1e-1
    
    # Generate a random seed if none is provided
    if seed is None:
        seed = randrange(2 ** 32)
    seed = seed
    np.random.seed(seed)
    
    if hasattr(sigma, '__len__'):
        sigma = sigma
    else:
        sigma = [sigma] * N
        
        print('{} parameters'.format(N))
    
    with mp.Pool(processes=3) as pool:
        if gaussian_move:
            # Use Gaussian move for the proposal distribution
            individuals = np.asarray([np.random.uniform(0, 10e-10, N) for _ in range(nwalkers)])
            np.asarray(individuals)
            Sampler = emcee.EnsembleSampler(nwalkers, N, residual, moves=emcee.moves.GaussianMove(sigma), pool=pool)

            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
        else:
            # Use the default stretch move
            individuals = [[np.random.normal(loc=0, scale=s) for s in sigma] for _ in range(nwalkers)]
            Sampler = emcee.EnsembleSampler(nwalkers, N, residual, pool=pool)
            
            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
    
    pool.join()
    
    # Data processing and analysis
    s = Sampler.chain.shape
    flatchain = np.transpose(Sampler.chain, axes=[1, 0, 2]).reshape(s[0] * s[1], s[2])
    flatlnprobability = Sampler.lnprobability.transpose().flatten()
    minfitness_each_gen = np.min(-Sampler.lnprobability * c, axis=0)
    
    flatfitness = -flatlnprobability * c
    best_index = np.argmin(flatfitness)
    best_fitness = flatfitness[best_index]
    best_uncorr = flatchain[best_index]
    best_corr = fittingp_to_simp(best_uncorr, initial_guess=initial_guess, multiples=multiples)
    
    
    population_array = fittingp_to_simp1(flatchain, initial_guess=initial_guess, multiples=multiples)
    path = os.path.join('./', 'poparr.npy')
    np.save(os.path.join(path), population_array)

    population_frame = pd.DataFrame(np.column_stack((population_array, flatfitness)))
    gen_start = 0
    gen_stop = len(flatfitness)
    gen_step = 1
    popsize = int(population_frame.shape[0] / len(flatfitness))
    index = []
    for i in range(gen_start, gen_stop, gen_step):
        index.extend(list(range(i * popsize, (i + 1) * popsize)))
    resampled_frame = population_frame.iloc[index]
    stats = resampled_frame.describe()

    # to test we return at the mean values and make sure that they are close to the true values
    # and convert the pandas dataframe to a numpy array
    if(test):
        mean = (stats.mean()).to_numpy()
        return mean[:-1]#last value is the fitness don't need it for testing

    else:
        stats.to_csv(os.path.join('./', 'test.csv'))
        stats.to_csv(os.path.join('./', 'test.csv'))
    
        stats.to_csv(os.path.join('./', 'test.csv'))    
    
        print("CSV saved to {}".format(path))


def fix_fitness_mcmc(fitness):
    """
    Metropolis-Hastings criterion: acceptance probability equal to ratio between P(new)/P(old)
    where P is proportional to probability distribution we want to find
    for our case we assume that probability of our parameters being the best is proportional to a Gaussian centered at fitness=0
    where fitness can be log, abs, squared error, etc.
    emcee expects the fitness function to return ln(P(new)), P(old) is auto-calculated
    """
    c = 1e-1  # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
    return -fitness / c
    # return -0.5 * fitness ** 2 / c ** 2


class PickeableResidual():
    """
    Factory created to call the residual function (which need to be pickled)
    in the MCMC andCMAES approach. This factory will allow to have the data,
    qx, qz, initial_guess, fit_mode to co;pare with the set of parameters
    returned by cmaes
    """

    def __init__(self, data, qxs, qzs, multiples, initial_guess, fit_mode='cmaes'):
        """
        Parameters
        ----------
        data, qxs, qzs: np.arrays of float32
            List of intensity/qx/qz at which the form factor has to be simulated
        initial_guess: list of float32
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

    def __call__(self, fit_params):
        """
        Parameters
        ----------
        fit_params: list of float32
            List of all the parameters value returned by CMAES/MCMC (list of
            Debye-Waller, I0, noise level, height, linewidth, [angles......])

        Returns
        -------

        """
  #         print('test', fit_params, self.minitial_guess, self.multiples)
        simp = fittingp_to_simp(fit_params, initial_guess=self.minitial_guess, multiples=self.multiples)

        if simp is None:
            if self.mfit_mode == "cmaes":
                return np.inf
            else:
                return 10e7
        
        dwx, dwz, intensity0, bkg, height, botcd, beta = simp[0], simp[1], simp[2], simp[3], simp[4], simp[5], np.array(simp[6:])

        langle = np.deg2rad(np.asarray(beta))
        rangle = np.deg2rad(np.asarray(beta))
        qxfit = []
        for i in range(len(self.mqz)):
            ff_core = original_fit.stacked_trapezoids(self.mqx[i], self.mqz[i], 0, botcd, height, langle, rangle)
            qxfit.append(ff_core)
        qxfit = original_fit.corrections_dwi0bk(qxfit, dwx, dwz, intensity0, bkg, self.mqx, self.mqz)

        res = 0
        for i in range(0, len(self.mdata), 1):
            res += original_fit.log_error(self.mdata[i], qxfit[i])

        if self.mfit_mode == 'cmaes':
            return res
        
        elif self.mfit_mode == 'mcmc':
            return fix_fitness_mcmc(res)

        else:
            print("This mode does not exist")
            return -1
        


def fittingp_to_simp(fit_params, initial_guess, multiples):
    """
    Convert the parameters returned by CMAES/MCMC, centered at 0 and std.
    dev. of 100, to have a physical meaning.
    The idea is to have the parameters contained in an interval, with the
    mean is the initial value given by the user, and the standart deviation
    is a % of this value
    Function extracted from XiCam

    Parameters
    ----------
    fit_params: list of float32
        List of all the parameters value returned by CMAES/MCMC (list of
        Debye-Waller, I0, noise level, height, linewidth, [angles......])
    initial_guess: list of float32
        Values entered by the user as starting point for the fit (list of
        Debye-Waller, I0, noise level, height, linewidth, [angles......])

    Returns
    -------
    simp: list of float32
        List of all the parameters converted
    """
    nbc = len(initial_guess) - 6
    # multiples = multiples * nbc
    simp = np.asarray(multiples) * np.asarray(fit_params) + initial_guess
    if np.any(simp[:6] < 0):
        return None
    if np.any(simp[6:] < 0) or np.any(simp[6:] > 90):
        return None
    
    return simp

def fittingp_to_simp1(fit_params, initial_guess, multiples):
    """
    Same function as the previous one, but for all the set of parameters
    simulated (list of list: [nb of parameter to describe the trapezoid]
    repeated the number of different combination tried)
    Function extracted from XiCam

    Parameters
    ----------
    fit_params: list of float32
        List of all the parameters value returned by CMAES/MCMC (list of
        Debye-Waller, I0, noise level, height, linewidth, [angles......])
    initial_guess: list of float32
        Values entered by the user as starting point for the fit (list of
        Debye-Waller, I0, noise level, height, linewidth, [angles......])

    Returns
    -------
    simp: list of float32
        List of all the parameters converted
    """
    nbc = len(initial_guess) - 6
    simp = np.asarray(multiples) * np.asarray(fit_params) + initial_guess
    simp[np.where(simp < 0)[0], :] = None

    return simp