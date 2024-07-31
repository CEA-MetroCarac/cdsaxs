""" This module contains the Fitter class, which is designed to fit the cdsaxs experimental data using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    and then do a statstical analysis of the best fit parameters using the MCMC (Markov Chain Monte Carlo) algorithm. 

    Classes:
        Fitter: A class that fits the cdsaxs experimental data using the CMA-ES and MCMC algorithms.
"""
import sys
import os
from typing import TYPE_CHECKING
from collections import deque
from random import randrange
import deap.base as dbase
from deap import creator, tools, cma
from scipy import stats
import emcee
import corner

import numpy as np
import pandas as pd
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = np

from .residual import Residual


if TYPE_CHECKING:
    from .simulations.base import Simulation

creator.create('FitnessMin', dbase.Fitness, weights=(-1.,))  # to minim. fitness
creator.create('Individual', list, fitness=creator.FitnessMin)


class Fitter:
    """
    This class is designed to fit the cdsaxs experimental data using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) and then do a statstical analysis
    of the best fit parameters using the MCMC (Markov Chain Monte Carlo) algorithm. It takes an instance of the Simulation class and fits this simulated data to the
    experimental data. 

    Attributes:
        Simulation (Simulation): An instance of the Simulation class representing the simulated diffraction pattern.
        exp_data (numpy.ndarray): Experimental diffraction data.
        xp (module): NumPy or CuPy module, depending on whether GPU acceleration is used.
        best_fit_cmaes (list or None): List containing the best fit parameters obtained using the CMA-ES algorithm.

    Methods:
        cmaes: Perform fitting using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
        mcmc: Give a set of statstical data on the best fit parameters using the MCMC (Markov Chain Monte Carlo) algorithm.

    """
    
    def __init__(self, Simulation: 'Simulation', exp_data):
        """
        Initialize the Fitter class with the Simulation instance and experimental data.
        
        Attributes:
            Simulation (Simulation): An instance of the Simulation class representing the simulated diffraction pattern.
            exp_data (numpy.ndarray): Experimental diffraction data.
        
        Returns: None
        """

        self.Simulation = Simulation
        self.exp_data = exp_data
        self.xp = Simulation.xp
        self.best_fit_cmaes = None #set the best fit obtained from cmaes to MCMC 
        
    def set_best_fit_cmaes(self, best_fit):
        """
        Set the best fit parameters obtained using the CMA-ES algorithm.

        Args:
            best_fit (pandas.DataFrame): The best fit parameters obtained using the CMA-ES algorithm.

        Returns:
            None
        """

        self.best_fit_cmaes = best_fit
        return None

    def cmaes(self, sigma, ngen, popsize, mu, n_default, restarts, tolhistfun, ftarget,
              restart_from_best=True, verbose=True, dir_save=None, test=False):
        """
        Fit experimental data using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm.

        This method utilizes a modified version of the CMA-ES algorithm to fit experimental data.

        Args:
            sigma (float): The initial standard deviation for each parameter.
            ngen (int): The number of generations to run the algorithm.
            popsize (int): The size of the population (number of candidate solutions) in each generation.
            mu (int): The number of parents/points for recombination.
            n_default (int): The number of parameters to be optimized.
            restarts (int): The number of restarts allowed during the optimization process.
            tolhistfun (float): The tolerance for the history of the best fitness value.
            ftarget (float): The target fitness value.
            restart_from_best (bool, optional): Determines whether to restart from the best individual found so far. Default is True.
            verbose (bool, optional): Controls whether to print progress information during optimization. Default is True.
            dir_save (str, optional): The directory to save the output. Default is None.
            test (bool, optional): Controls whether to test the function and return best value instead of performing the full optimization process. If True, the function returns best value. Default is False.

        Returns:
            tuple: A tuple containing the best fit parameters and the corresponding fitness value.

        Attributes:
            best_fit_cmaes (list or None): List containing the best fit parameters obtained using the CMA-ES algorithm.

        Notes:
            This method is modified from deap/algorithms.py to return a list of populations instead of the final population and to incorporate additional termination criteria based on neuromorphic algorithms. The function was originally extracted from XiCam and has been modified for specific use cases.

        """

        #cupy or numpy
        xp = self.xp


        if dir_save is not None:
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)

        if verbose:
            print("Start CMAES")
        toolbox = dbase.Toolbox()
        
                
        # Setting Simulation attribute to match the case
        self.Simulation.set_from_fitter(True)

        #declare Fitness function and register
        residual = Residual(self.exp_data, fit_mode='cmaes', xp=self.xp, Simulation=self.Simulation)
        toolbox.register('evaluate', residual)
        

        halloffame = tools.HallOfFame(1)

        thestats = tools.Statistics(lambda ind: ind.fitness.values)
        thestats.register('avg', lambda x: xp.mean(xp.asarray(x)[xp.isfinite(xp.asarray(x))]) \
            if xp.asarray(x)[xp.isfinite(xp.asarray(x))].size != 0 else None)
        thestats.register('std', lambda x: xp.std(xp.asarray(x)[xp.isfinite(xp.asarray(x))]) \
            if xp.asarray(x)[xp.isfinite(xp.asarray(x))].size != 0 else None)
        thestats.register('min', lambda x: xp.min(xp.asarray(x)[xp.isfinite(xp.asarray(x))]) \
            if xp.asarray(x)[xp.isfinite(xp.asarray(x))].size != 0 else None)
        thestats.register('max', lambda x: xp.max(xp.asarray(x)[xp.isfinite(xp.asarray(x))]) \
            if xp.asarray(x)[xp.isfinite(xp.asarray(x))].size != 0 else None)
        thestats.register('fin', lambda x: xp.sum(xp.isfinite(xp.asarray(x))) / xp.size(xp.asarray(x)))

        # thestats.register('cumtime', lambda x: time.perf_counter() - last_time)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (thestats.fields if thestats else [])
        population_list = []
        popsize_default = int(4 + 3 * xp.log(n_default))
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

        n_restarts = 0
        for restart in range(restarts + 1):
            if allbreak:
                break
            if restart != 0:
                kwargs['lambda_'] *= 2
                print('Doubled popsize')
                if restart_from_best:
                    initial_individual = halloffame[0]
                n_restarts = restart

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
            maxlen = 10 + int(xp.ceil(30 * n_default / lambda_))
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

                # save the population only if user wants to
                if dir_save is not None:
                    population_list.append(population)

                # Evaluate the individuals
                fitnesses = toolbox.evaluate(population)

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
        best_corr = self.Simulation.geometry.convert_to_dataframe([best_uncorr])

        if best_corr is not None:
            #update best fit attribute
            self.set_best_fit_cmaes(best_fit=best_corr)


        if verbose:
            print(('best', best_uncorr, best_fitness))
        # make population dataframe, order of rows is first generation for all
        # children, then second generation for all children...
        

        if test:
            return best_corr
        
        if dir_save is not None:
        
            #convert in to right format
            population_arr = self.xp.asarray(
            [list(individual) for generation in population_list for individual in
            generation])

            fitness_arr = self.xp.asarray(
            [individual.fitness.values[0] for generation in population_list for individual in
            generation])

            self.save_population(population_arr, fitness_arr, dir_save, fit_mode='cmaes')

        return best_corr, best_fitness
    
    def mcmc_bestfit_stats(self, N, sigma, nsteps, nwalkers, gaussian_move=False, seed=None, verbose=False, test=False, dir_save=None):
        """
        Generate a set of statstical data on the best fit parameters using the MCMC (Markov Chain Monte Carlo) algorithm. Two kinds of options for moves to explore solution space are provided gaussian and stretch move. Default is strech move and recommended.

        This method utilizes the emcee package's implementation of the MCMC algorithm and generates a csv file with the statistical data of the best fit parameters.

        Args:
            N (int): The number of parameters to be optimized.
            sigma (float or list): The initial standard deviation for each parameter. If a float is provided, it is applied to all parameters. If a list is provided, each parameter is initialized with the corresponding value.
            nsteps (int): The number of MCMC steps to perform.
            nwalkers (int): The number of MCMC walkers to use.
            gaussian_move (bool, optional): Determines whether to use Gaussian moves for proposal distribution. If True, Gaussian moves are used. If False, stretch moves are used. Default is False.
            seed (int, optional): The seed for the random number generator. If None, a random seed is generated. Default is None.
            verbose (bool, optional): Controls whether to print progress information during fitting. If True, progress information is printed. Default is False.
            test (bool, optional): Controls whether to test the function and return mean values instead of performing the full fitting process. If True, the function returns mean values. Default is True.

        Returns:
            None

        Attributes:
            best_uncorr (numpy.ndarray): The best uncorrected individual obtained from the MCMC fitting process.
            best_fitness (float): The fitness value of the best individual obtained from the MCMC fitting process.
            minfitness_each_gen (numpy.ndarray): The minimum fitness value at each generation during the MCMC fitting process.
            Sampler (emcee.EnsembleSampler): An instance of emcee.EnsembleSampler with detailed output of the MCMC algorithm.

        """
        #cupy or numpy
        xp = self.xp

        # Setting Simulation attribute to match the case
        self.Simulation.set_from_fitter(True, self.best_fit_cmaes)

        #declare Fitness function and register
        residual = Residual(self.exp_data, fit_mode='mcmc', xp=self.xp, Simulation=self.Simulation, best_fit=self.best_fit_cmaes)


        def do_verbose(Sampler):
            if hasattr(Sampler, 'acceptance_fraction'):
                print('Acceptance fraction: ' + str(xp.mean(Sampler.acceptance_fraction)))
            else:
                print('Acceptance fraction: ' + str(xp.mean([Sampler.acceptance_fraction for Sampler in Sampler])))
            sys.stdout.flush()
        
        # Empirical factor to modify MCMC acceptance rate
        c = residual.c
        
        # Generate a random seed if none is provided
        if seed is None:
            seed = randrange(2 ** 32)
        xp.random.seed(seed)
        
        if not hasattr(sigma, '__len__'):
            sigma = [sigma] * N
            
            print('{} parameters'.format(N))

        sigma = xp.array(sigma)

        try:
            if isinstance(sigma, cp.ndarray):
                sigma = sigma.get()
        except:
            pass

        if gaussian_move:
            # Use Gaussian move for the proposal distribution
            individuals = [np.random.uniform(-100, 100, N) for _ in range(nwalkers)]
            
            Sampler = emcee.EnsembleSampler(nwalkers, N, residual, moves=emcee.moves.GaussianMove(sigma), pool=None, vectorize=True)

            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
        else:

            individuals = [np.random.default_rng().normal(loc=0, scale=sigma, size=sigma.shape) for _ in range(nwalkers)]
            Sampler = emcee.EnsembleSampler(nwalkers, N, residual, pool=None, vectorize=True)
                
            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
            

        #Autocorelation time to find burnin steps
        try:
            tau = Sampler.get_autocorr_time(tol=5)
            burnin = int(2 * np.max(tau))
        except:
            burnin = int(1/3 * nsteps)
        
        
        # Data processing and analysis
        s = Sampler.get_chain(discard=burnin).shape

        flatchain = xp.transpose(Sampler.get_chain(discard=burnin), axes=[1, 0, 2]).reshape(s[0] * s[1], s[2])
        flatlnprobability = Sampler.get_log_prob(discard=burnin).transpose().flatten()
        minfitness_each_gen = xp.min(-1  * Sampler.get_log_prob(discard=burnin) * c, axis=0)
        
        #convert log probability to usual fitness
        flatfitness = -flatlnprobability * c

        #find the best individual and convert it to the correct form
        best_index = np.argmin(flatfitness)
        best_fitness = flatfitness[best_index]
        best_uncorr = flatchain[best_index]
        best_corr = self.Simulation.geometry.convert_to_dataframe([best_uncorr])

        population_df = self.Simulation.geometry.convert_to_dataframe(flatchain)


        mcmc_stats = self.do_stats(population_df.dropna())

        if(test):
            return best_corr, best_fitness
        
        elif dir_save is not None:
            self.save_population(flatchain, flatfitness, dir_save, fit_mode='mcmc')

            #save the stat data
            path_mcmc_stats = os.path.join(dir_save, 'mcmc_stats.csv')
            mcmc_stats.to_csv(path_mcmc_stats)
            print('Saved to ' + path_mcmc_stats)

        return mcmc_stats


    def plot_correlation(self, file, dir_save=None):
        """
            Generate a corner plot of the best fit parameters obtained from the MCMC fitting process.

            This method utilizes the corner package to generate a corner plot of the best fit parameters obtained from the MCMC fitting process.

            Args:
                file (str): The path to the file containing the data.
                dir_save (str, optional): The directory to save the output. If not provided, the plot will be displayed instead of being saved.

            Returns:
                None

        """
        data = np.genfromtxt(file, skip_header=1, delimiter=',')
        #read the headers for each column
        with open(file, 'r') as f:
            title = f.readline().strip().split(',')
        

        #remove the last column which is the fitness value
        data = data[:, :-1]
        title = title[:-1]

        # get rid of columns with nan values
        data = data[~np.isnan(data).any(axis=1)]

        figure = corner.corner(data, labels=title, quantiles=[0.16, 0.5, 0.84], show_titles=True)

        if dir_save is not None:
            figure.savefig(os.path.join(dir_save, "corner_plot.png"))
            print('Saved to ' + os.path.join(dir_save, 'corner_plot.png'))
        else:
            figure.show()

    def save_population(self, population_arr, fitness_arr, dir_save, fit_mode='cmaes'):
        """
        Save the population array to a csv file.

        Args:
            population (numpy.ndarray): The population array to save.
            dir_save (str): The directory to save the output.

        Returns:
            None

        """
        fitness_dataframe = pd.DataFrame(fitness_arr, columns=['fitness'])
        
        population_dataframe = self.Simulation.geometry.convert_to_dataframe(population_arr)
        
        frames = [population_dataframe, fitness_dataframe]
        result = pd.concat(frames, axis=1).dropna()

        if fit_mode == 'cmaes':
            name = 'population_cmaes.csv'
        else:
            name = 'population_mcmc.csv'
        
        result.to_csv(os.path.join(dir_save, name), index=False)
        print('Saved to ' + os.path.join(dir_save, name))

    @staticmethod
    def do_stats(df, cf=0.99):
        """
        Generate a set of statistical data on the best fit parameters obtained from the MCMC fitting process.

        This method generates a set of statistical data on the best fit parameters obtained from the MCMC fitting process.

        Args:
            df (pandas.DataFrame): The DataFrame containing the best fit parameters.

        Returns:
            pandas.DataFrame: A DataFrame containing the statistical data on the best fit parameters.

        """
        #calculate confidence interval for each parameter
        mean = df.mean()
        std = df.std()
        count = df.count()
        min = df.min()
        max = df.max()

        match cf:
            case 0.99:
                z = 2.576
            case 0.95:
                z = 1.96
            case 0.90:
                z = 1.645
            case _:
                z = 2.576

        uncertainity = z * std / np.sqrt(count)

        lower_ci = mean - uncertainity
        upper_ci = mean + uncertainity

        stat = pd.DataFrame({'mean': mean, 'std': std, 'count': count, 'min': min, 'max': max, 'lower_ci': lower_ci, 'upper_ci': upper_ci, 'uncertainity': uncertainity})

        return stat



        

