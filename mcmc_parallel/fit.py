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
import matplotlib.pyplot as plt
import deap.base as dbase
from deap import creator, tools, cma
from scipy import stats
import scipy.interpolate
import emcee
import sys
import multiprocessing as mp

creator.create('FitnessMin', dbase.Fitness, weights=(-1.,))  # to minim. fitness
creator.create('Individual', list, fitness=creator.FitnessMin)


def find_peaks(values, positions, threshold):
    """
    Gives peaks over threshold as list of list of each point
    TODO: check if this function already exist in numpy

    Parameters
    ----------
    values: numpy.ndarray((n))
        1D-data to analyse
    positions: numpy.ndarray((n))
        x-positions of data
    threshold: float
        threshold limit

    Returns
    -------
    peaks: list of tuple of floats
        list of peaks positions and related values
    """
    err_msg = "sizes of 'values' and 'positions' differ"
    assert (len(values) == len(positions)), err_msg

    first_point = 0
    inside_peak = False
    peaks = []
    for (i, val) in enumerate(values):
        # Detection of peak begining
        if not inside_peak and val > threshold:
            first_point = i
            inside_peak = True
        # Detection of end of peak
        if inside_peak and (val < threshold or np.isnan(val)):
            inside_peak = False
            # At least four points must be used to have point
            if (i - first_point) > 3:
                peak = (positions[first_point - 1:i], values[first_point - 1:i])
                peaks.append(peak)
            elif i - first_point == 1:
                peak = (positions[first_point], values[first_point])
                peaks.append(peak)

    return peaks


def fit_peaks(peaks):
    """
    Return centers position of each peak and their  maximum
    TODO: check why gauss fit was removed

    Parameters
    ----------
    peaks: list of tuple of floats
        list of peaks positions and related values

    Returns
    -------
    centers_pos, max_peak: list of floats
        All centers and values respectively obtained at this maximum
    """
    centers_pos = []
    max_peak = []
    for (pos, vals) in peaks:
        if isinstance(pos, float):
            centers_pos.append(pos)
            max_peak.append(vals)
        elif len(pos) > 1:
            center, value = maxleftright(pos, vals)
            centers_pos.append(center)
            max_peak.append(value)

    return centers_pos, max_peak


def maxleftright(positions, values):
    """
    Return the center of a peak and the associated value by "search around"
    TODO: check why gauss fit was removed

    Parameters
    ----------
    positions: list of floats
        Positions of data
    values: list of floats
        Values associated to positions

    Returns
    -------
    center, value: floats
        Center and associated value respectively
    """
    x0 = np.average(positions)

    ind0 = np.where((positions >= x0 - 0.02) & (positions <= x0))
    max0 = max(values[ind0])
    pos0 = (positions[ind0])[np.where(values[ind0] == max0)]
    ind1 = np.where((positions >= x0) & (positions <= x0 + 0.02))
    max1 = max(values[ind1])
    pos1 = (positions[ind1])[np.where(values[ind1] == max1)]

    center = (pos0[0] + pos1[0]) / 2
    value = (max0 + max1) / 2

    return center, value


def corrections_dwi0bk(intensities, dw_factorx, dw_factorz,
                       scaling, bkg_cste, qxs, qzs):
    """
    Return coorected intesnety from intensity simulated application of :
        - Debye waller factors
        - Intensity scalling
        - Constante background

    Parameters
    ----------
    intensities: list of floats
        Intensities obtained by simulation for each qx, qz
    dw_factorx: float
        Debye-Waller factor correction along x axis
    dw_factorz: float
        Debye-Waller factor correction along z axis
    scaling: float
        scaling factor applied to the intensity
    bkg_cste: float
        Constant background to add
    qxs: list of floats
        Qx values associated to each intensity
    qzs: list of floats
        Qz values associated to each intensity

    Returns
    -------
    intensities_corr: list of floats
        Corrected intensities
    """
    # TODO: use qxqzi data format as in other function

    intensities_corr = []
    for intensity, qxi, qzi in zip(intensities, qxs, qzs):
        dw_array = np.exp(-((np.asarray(qxi) * dw_factorx) ** 2 +
                            (np.asarray(qzi) * dw_factorz) ** 2))
        intensities_corr.append(np.asarray(intensity) * dw_array * scaling
                                + bkg_cste)
    return intensities_corr


def trapezoid_form_factor(qys, qzs, y1, y2, langle, rangle, height):
    """
    Simulation of the form factor of a trapezoid at all qx, qz position.
    Function extracted from XiCam

    Parameters
    ----------
    qys, qzs: list of floats
        List of qx/qz at which the form factor is simulated
    y1, y2: floats
        Values of the bottom right/left (y1/y2) position respectively of
        the trapezoids such as y2 - y1 = width of the bottom of the trapezoids
    langle, rangle: floats
        Left and right bottom angle of trapezoid
    height: float
        Height of the trapezoid

    Returns
    -------
    form_factor: list of float
        List of the values of the form factor
    """
    tan1 = np.tan(langle)
    tan2 = np.tan(np.pi - rangle)
    val1 = qys + tan1 * qzs
    val2 = qys + tan2 * qzs
    with np.errstate(divide='ignore'):
        form_factor = (tan1 * np.exp(-1j * qys * y1) *
                       (1 - np.exp(-1j * height / tan1 * val1)) / val1)
        form_factor -= (tan2 * np.exp(-1j * qys * y2) *
                        (1 - np.exp(-1j * height / tan2 * val2)) / val2)
        form_factor /= qys

    return form_factor


def stacked_trapezoids(qys, qzs, y1, y2, height, langle,
                       rangle=None, weight=None):
    """
    Simulation of the form factor of trapezoids at qx, qz position.
    Function extracted from XiCam (modified)

    Parameters
    ----------
    qys, qzs: list of floats
        List of qx/qz at which the form factor is simulated
    y1, y2: floats
        Values of the bottom right/left (y1/y2) position respectively of
        the trapezoid such as y2 - y1 = width of the bottom of the trapezoid
    height: float
        Height of the trapezoid
    langle, rangle: list of floats
        Each angle correspond to a trapezoid
    weight: list of floats
        To manage different material in the stack.

    Returns
    -------
    form_factor_intensity: list of floats
        Intensity of the form factor
    """
    if not isinstance(langle, np.ndarray):
        raise TypeError('angles should be array')

    if rangle is not None:
        if not langle.size == rangle.size:
            raise ValueError('both angle array are not of same size')
    else:
        rangle = langle

    form_factor = np.zeros(qzs.shape, dtype=complex)
    # loop over all the angles
    for i in range(langle.size):
        shift = height * i
        left, right = langle[i], rangle[i]
        coeff = np.exp(-1j * shift * qzs)
        if weight is not None:
            coeff *= weight[i] * (1. + 1j)
        form_factor += trapezoid_form_factor(qys, qzs, y1, y2,
                                             left, right, height) * coeff
        y1 += height / np.tan(left)
        y2 += height / np.tan(np.pi - right)

    form_factor_intensity = np.absolute(form_factor) ** 2

    return form_factor_intensity


def log_error(exp_i_array, sim_i_array):
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
    indice = exp_i_array > 0
    error = np.nansum(np.abs((np.log10(exp_i_array[indice]) -
                              np.log10(sim_i_array[indice]))))
    error /= np.count_nonzero(~np.isnan(exp_i_array))

    return error


def std_error(exp_i_array, sim_i_array):
    """
    Return the difference between two set of values (experimental and
    simulated data), using the std error

    Parameters
    ----------
    exp_i_array: numpy.ndarray((n))
        Experimental intensities data
    sim_i_array: numpy.ndarray((n))
        Simulated intensities data

    Returns
    -------
    error: float
        Error between the two set of data (normalized by number of data)
    """
    error = np.nansum(np.abs((exp_i_array - sim_i_array)))
    error /= np.count_nonzero(~np.isnan(exp_i_array))

    return error


def cmaes(data, qxs, qzs, initial_guess, sigma, ngen,
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
    residual = PickeableResidual(data, qxs, qzs, initial_guess,
                                 fit_mode='cmaes')
    
    toolbox.register('evaluate', residual)
    
    #register parallel map function to toolbox
    parallel = mp.cpu_count()
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

            #these two lines are added for parallel
            pool.close()
            pool.join()#close the pool and wait for the work to finish


            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)  # tuple of length 1
            halloffame.update(population)
            # print(fittingp_to_simp(halloffame[0], initial_guess))
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
    best_corr = fittingp_to_simp(best_uncorr, initial_guess)
    if verbose:
        print(('best', best_corr, best_fitness))
    # make population dataframe, order of rows is first generation for all
    # children, then second generation for all children...
    population_arr = np.array(
        [list(individual) for generation in population_list for individual in
         generation])

    population_arr = fittingp_to_simp1(population_arr, initial_guess)

    fitness_arr = np.array(
        [individual.fitness.values[0] for generation in population_list for
         individual in generation])

    population_fr = pd.DataFrame(np.column_stack((population_arr, fitness_arr)))
    if dir_save is not None:
        population_fr.to_excel(os.path.join(dir_save, "output.xlsx"))

    return best_corr, best_fitness


def fittingp_to_simp(fit_params, initial_guess):
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
    multiples = [0.0001, 0.0001, 0.001, 0.001, 0.01, 0.01] + [0.04] * nbc
    simp = np.asarray(multiples) * np.asarray(fit_params) + initial_guess
    if np.any(simp[:6] < 0):
        return None
    if np.any(simp[6:] < 0) or np.any(simp[6:] > 90):
        return None

    return simp


def fittingp_to_simp1(fit_params, initial_guess):
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
    multiples = [0.0001, 0.0001, 0.001, 0.001, 0.01, 0.01] + [0.04] * nbc
    simp = np.asarray(multiples) * np.asarray(fit_params) + initial_guess
    simp[np.where(simp < 0)[0], :] = None

    return simp

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

    def __init__(self, data, qxs, qzs, initial_guess, fit_mode='cmaes'):
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
        simp = fittingp_to_simp(fit_params, self.minitial_guess)
        if simp is None:
            return np.inf
        dwx, dwz, intensity0, bkg, height, botcd, beta = simp[0], simp[1], \
                                                         simp[2], simp[3], \
                                                         simp[4], simp[5], \
                                                         np.array(simp[6:])

        langle = np.deg2rad(np.asarray(beta))
        rangle = np.deg2rad(np.asarray(beta))
        qxfit = []
        for i in range(len(self.mqz)):
            ff_core = stacked_trapezoids(self.mqx[i], self.mqz[i],
                                         0, botcd, height, langle, rangle)
            qxfit.append(ff_core)
        qxfit = corrections_dwi0bk(qxfit, dwx, dwz, intensity0,
                                   bkg, self.mqx, self.mqz)

        res = 0
        for i in range(0, len(self.mdata), 1):
            # qxfit[i] -= min(qxfit[i])
            # qxfit[i] /= max(qxfit[i])
            # qxfit[i] += i + 1
            res += log_error(self.mdata[i], qxfit[i])

        if self.mfit_mode == 'cmaes':
            return res
        elif self.mfit_mode == 'mcmc':
            return fix_fitness_mcmc(res)
        else:
            print("This mode does not exist")
            return -1


def spectra_calcul(params, qxs, qzs):
    langle = rangle = np.deg2rad(np.array(params[6:]))

    intensities = []
    for i, qzi in enumerate(qzs):
        ff_core = stacked_trapezoids(qxs[i], qzi, 0, params[5], params[4],
                                     langle, rangle)

        intensities.append(ff_core)

    intensities_corr = corrections_dwi0bk(intensities, *params[:4], qxs, qzs)

    return intensities_corr


def fitting_cmaes(qxs, qzs, intensity,
                  height=100, cd_bot=100, swa=80,
                  nbtrap=5, dwx=0.11, dwz=0.11,
                  intensity0=0.4, bkg=10., verbose=True):
    """
    Manage fitting with cmaes function, allo default value and initiate
    guess data
    TODO: use format qxqzi data, to 1 argument
    TODO: create object which correspond to the line shape with associated
    parameters CD, height...

    Parameters
    ----------
    qxs, qzs: list of floats
        List of qx/qz to fit
    intensity: list of floats
        Intensities link to the qx, qz data
    height: int, optional
        Guess height of each trapezoid
    cd_bot: int, optional
        Guess bottom CD of trapezoid
    swa: int, optional
        guess global side-wall angle in [°]
    nbtrap: int, optional
        Number of tranpezoïd used to fit
    dwx: float, optional
        Guess debye-waller factor in x direction
    dwz: float, optional
        Guess debye-waller factor in z direction
    intensity0: float, optional
        Initial intensity scaling value
    bkg: float, optional
        initial background value
    verbose: bool, optional
        Activation key for verbosity

    Returns
    -------
    best_corr, best_fitness:
        Line parameters fitted and associated error
    """
    assert isinstance(height, int), "'height' should be integer"
    assert isinstance(cd_bot, int), "'cd_bot' should be integer"
    assert isinstance(swa, int), "'swa' should be integer"

    initial_vals = [dwx, dwz, intensity0, bkg, height, cd_bot] + [swa] * nbtrap

    best_corr, best_fitness = cmaes(data=intensity, qxs=qxs, qzs=qzs,
                                    initial_guess=np.asarray(initial_vals),
                                    sigma=200, ngen=1000, popsize=100,
                                    mu=10, n_default=len(initial_vals),
                                    restarts=0, tolhistfun=5e-5, ftarget=None,
                                    verbose=verbose)

    return best_corr, best_fitness


def bkg_removal(data, sigma, ampli, offset, savepath=None, show_plots=False):
    """
    Background removal

    Parameters
    ----------
    data :
        Data in columns with qx, qz and intensities
    sigma: float
        Sigma parameter of the gaussian
    ampli: float
        Amplitude of the gaussian
    offset:  float
        Offset of gaussian intensity
    show_plots: bool, optional
        Activation key for figure plotting

    Returns
    -------
    data_corr:
        Data minus the gaussian
    """
    data_corr = data.copy()

    alpha = sigma * np.sqrt(2 * np.pi)
    norm = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
    data_corr[:, 2] -= (ampli / (alpha * np.exp(0.5 * (norm / sigma) ** 2))
                        + offset)

    if show_plots:
        data_0 = data[data[:, 1] == 0, :]
        data_1 = data_corr[data_corr[:, 1] == 0, :]
        gauss = (ampli / (alpha * np.exp(0.5 * (data_0[:, 0] / sigma) ** 2))
                 + offset)

        plt.semilogy(data_0[:, 0], data_0[:, 2], label="raw")
        plt.semilogy(data_0[:, 0], data_1[:, 2], label="bkg_rmv")
        plt.semilogy(data_0[:, 0], gauss, label="gauss")
        plt.show()
        if savepath is not None:
            plt.savefig(savepath + '\\unfiltered_peaks.png')

    return data_corr


def qzflatten(data, bins=400):
    """
    Reduce data along qz axis with mean

    Parameters
    ----------
    data: list of floats
        Data in columns with qx, qz and intensities
    bins: int, optional
        Number of bin along qx

    Returns
    -------
    pitch: array of float
        Data in column qx, intensity
    """

    idata = []
    qxmax = np.max(data[:, 0])
    qxmin = np.min(data[:, 0])
    step = qxmax / bins
    qxrange = np.arange(qxmin, qxmax, step)
    for qxi in qxrange:
        ind = np.where((data[:, 0] >= qxi) & (data[:, 0] < qxi + step))
        if data[ind, 2].any():
            idata.append(np.mean(data[ind, 2]))
        else:
            idata.append(0)
    return np.array((qxrange, idata)).T


def pitch_determination(data, limit, nborder_max=-1,
                        bins=400, show_plots=False):
    """
    Calculate the pitch of line

    Parameters
    ----------
    data: list of floats
        Data in columns with qx, qz and intensities
    limit: float
        Intensity threshold used for determination of a peak
    nborder_max: int, optional
        Number of order to use for the fit (-1 all are used)
    bins: int, optional
        Number of bin along qx for qz flatten, must be adapted with
        resolution of acquisition
    show_plots: bool, optional
        Activation key for peak position plotting as function of order allow to
        check extraction

    Returns
    -------
    pitch: float
        The pitch of line sample

    """
    qxidata = np.array(qzflatten(data, bins=bins))
    peaks = find_peaks(qxidata[:, 1], qxidata[:, 0], limit)
    xypics = fit_peaks(peaks)
    if len(xypics[0]) < nborder_max:
        nborder_max = len(xypics[0])

    x = 2 * np.pi * np.arange(len(xypics[0][:nborder_max]))
    y = xypics[0][:nborder_max]
    res = stats.linregress(x, y)

    if show_plots:
        plt.plot(y, '+')
        plt.plot(x * res.slope + res.intercept)
        plt.show()

    # Verification of linearity
    if res.rvalue < 0.9999:
        print("WARNING, Linearity in pitch determination was not respected")

    pitch = 1. / res.slope

    return pitch


def interpolate_and_cutorder(qxqzi, qz_sampling, pitch,
                             nborder, startorder=1):
    """
    Intensity interpolation

    Parameters
    ----------
    qxqzi: list of floats
        Data in columns with qx, qz and intensities
    qz_sampling :
        Final sampling along qz
    pitch: float
        Pitch of sample (to extract specific qx part)
    nborder: int
        Number of order along qx to evaluate
    startorder: int, optional
        First order to treat

    Returns
    -------
    qxs, qzs, interp_intensity:
        Interpolated intensity and linked qx, qz values
    """

    step_qx = np.arcsin(2 * np.pi / pitch)
    interp_intensity = []
    qxs = []
    qzs = []
    for order in range(nborder):
        qx_ = np.ones(len(qz_sampling)) * step_qx * (order + startorder)
        data = scipy.interpolate.griddata(qxqzi[:, 0:2], qxqzi[:, 2],
                                          (qx_, qz_sampling),
                                          method='linear')
        interp_intensity.append(data[~np.isnan(data)])
        qxs.append(qx_[~np.isnan(data)])
        qzs.append(qz_sampling[~np.isnan(data)])

    return qxs, qzs, interp_intensity


def spectra_plot(params, qxs, qzs, data=None):
    """
    Plot spectra obtained with given line parameters.
    Could plot experimental data by superposition

    Parameters
    ----------
    params: list of floats and ints
        Parameters that describe line profile
        | format : (float Debye-Waller X factor, float Debye-waller Z factor,
        float Intensity scaling, float background constante,bint height of
        trapezoid, int CD of line , list of floats SWA)
    qxs, qzs: list of floats
        qx, qz to plot
    data: list of floats, optional
        intensity of experimental data to stack

    Returns
    -------
    fig: matplotlib figure
    """
    langle = rangle = np.deg2rad(np.array(params[6:]))

    intensities = []
    for i, qzi in enumerate(qzs):
        ff_core = stacked_trapezoids(qxs[i], qzi, 0, params[5], params[4],
                                     langle, rangle)
        intensities.append(ff_core)

    intensities_corr = corrections_dwi0bk(intensities, *params[:4], qxs, qzs)

    fig = plt.figure(figsize=(16, 10))
    for i, qzi in enumerate(qzs):
        coeff = 10 ** (-2 * i)  # offset add for visualisation
        plt.semilogy(qzi, intensities_corr[i] * coeff, label="fitO%d" % i)
        if data is not None:
            plt.scatter(qzi, data[i] * coeff, s=1, label="GenerateO%d" % i)
    plt.legend()

    return fig


def line_profile_plot(pitch, height, cd_bot, swa, savepath=None):
    """
    Plot the cross profil of line determined by morphological parameter
    can save the plot

    Parameters
    ----------
    pitch: float
        Pitch of line grating
    height: float
        Height of a trapezoïd
    cd_bot: float
        Bottom cd of first trapezoïd
    swa: list floats
        All side-wall angles
    savepath: str, optional
        Pathname to save the plot (if None, figure is not save)

    Returns
    -------
    fig: matplotlib figure
    """

    swa_rad = np.deg2rad(swa)
    nbrtrap = len(swa)
    x = np.zeros(2 * (len(swa) + 1), dtype=np.float_)
    y = np.zeros_like(x)
    dxl = np.cumsum(height / np.tan(swa_rad))
    dxr = np.cumsum(height / np.tan(swa_rad))[::-1]
    x[0] = -0.5 * cd_bot
    x[-1] = 0.5 * cd_bot
    x[1:nbrtrap + 1] = x[0] + dxl
    x[nbrtrap + 1:-1] = x[-1] - dxr
    y[1:nbrtrap + 1] = np.arange(1, nbrtrap + 1) * height
    y[nbrtrap + 1:-1] = np.arange(1, nbrtrap + 1)[::-1] * height

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    fig = plt.figure()
    plt.plot(x, y)
    # vertical line x<0
    plt.plot([-pitch / 2., -pitch / 2.], [0, height * nbrtrap], 'k')
    # vertical line x>0
    plt.plot([pitch / 2., pitch / 2.], [0, height * nbrtrap], 'k')
    plt.axis('equal')

    if savepath is not None:
        plt.savefig(savepath + "\\lineprofile.png")

    return fig


def mcmc(data, qxs, qzs, initial_guess, N, sigma, nsteps, nwalkers, use_mh=False, parallel=True, seed=None,
             verbose=True):
    """Fit with emcee package's implementation of MCMC algorithm and place into instance of self
    Calls fitness_individual many times, then calls make_population_frame_best

    Attributes:
        best_uncorr: best uncorrected individual
        best_fitness: scalar
        minfitness_each_gen: length ngen
        Sampler: instance of emcee.Sampler with detailed output of algorithm

    Args:
        self: instance of Run
        sigma: array or scalar, initial standard deviation for each parameter
        nsteps: number of steps
        nwalkers: number of walkers
        use_mh: True for Metropolis-Hastings proposal and ensemble Sampler, False for ensemble Sampler, 'MH' for Metropolis-Hastings proposal and Sampler
        parallel: False for no parallel, True for cpu_count() processes, or int to specify number of processes, or 'scoop' for cluster
        plot_on: whether to plot fitness, best trapezoids and sim and exp scattering
        seed: seed for random number generator
    """

    
    residual = PickeableResidual(data, qxs, qzs, initial_guess, fit_mode='mcmc')
        
    def do_verbose(Sampler):
            if hasattr(Sampler, 'acceptance_fraction'):
                print('Acceptance fraction: ' + str(np.mean(Sampler.acceptance_fraction)))
            else:
                print('Acceptance fraction: ' + str(np.mean([Sampler.acceptance_fraction for Sampler in Sampler])))
            sys.stdout.flush()
    
    c = 1e-1  # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate

    #generate random seed if none is given    
    if seed is None:
        seed = randrange(2 ** 32)
    seed = seed
    np.random.seed(seed)
    fix_fitness = fix_fitness_mcmc

    if hasattr(sigma, '__len__'):
        sigma = sigma
    else:
        sigma = [sigma] * N
        
        print('{} parameters'.format(N))


    with mp.Pool(processes=mp.cpu_count()) as pool:

        #if use_mh is true then use gaussian move for the proposal distribution else use strech move which is the default is moves is not specified
        if use_mh:
            
            individuals = [np.random.uniform(0,10e-10,N) for _ in range(nwalkers)]#original code used np.zeroes but emcee complains that they should be linearly independent so i choose values very close to zero but not exactly zero
            np.asarray(individuals)
            Sampler = emcee.EnsembleSampler(nwalkers, N, residual, moves=emcee.moves.GaussianMove(sigma), pool=pool)

            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)

        else:
            
            individuals = [[np.random.normal(loc=0, scale=s) for s in sigma] for _ in range(nwalkers)]
                        
            Sampler = emcee.EnsembleSampler(nwalkers, N, residual, pool=pool)
            
            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
    pool.join()
    

    s = Sampler.chain.shape
    flatchain = np.transpose(Sampler.chain, axes=[1, 0, 2]).reshape(s[0] * s[1], s[2])
    flatlnprobability = Sampler.lnprobability.transpose().flatten()
    minfitness_each_gen = np.min(-Sampler.lnprobability * c, axis=0)

    # flatchain has shape (nwalkers * nsteps, N)
    # flatlnprobability has shape (nwalkers * nsteps,)
    # flatchain and flatlnprobability list first step of all walkers, then second step of all walkers...

    # Sampler.flatchain and Sampler.flatlnprobability (made by package) list all steps of first walker, then all steps of second walker...
    # but we don't want that

    flatfitness = -flatlnprobability * c
    best_index = np.argmin(flatfitness)
    best_fitness = flatfitness[best_index]
    best_uncorr = flatchain[best_index]
    best_corr = fittingp_to_simp(best_uncorr, initial_guess)#, multiples)

    # Warning, it was a self.residual
    residual(best_uncorr)

    population_array = fittingp_to_simp(flatchain, initial_guess)#, multiples)
    path = os.path.join('./', 'poparr.npy')
    np.save(os.path.join(path), population_array)
    # self.population_array = flatchain

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
    stats.to_csv(os.path.join('./', 'test.csv'))

    print("CSV saved to {}".format(path))