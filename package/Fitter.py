from cdsaxs.package.Residual import PicklableResidual
from cdsaxs.package.Simulation import TrapezoidGeometry
import os
from collections import deque
import numpy as np
import cupy as cp
import pandas as pd
from random import randrange
import deap.base as dbase
from deap import creator, tools, cma
from scipy import stats
import emcee
import sys


class Fitter():
    
    def __init__(self, TrapezoidGeometry):
        self.geometry = TrapezoidGeometry

    def cmaes(self, data, qxs, qzs, initial_guess, multiples, sigma, ngen,
            popsize, mu, n_default, restarts, tolhistfun, ftarget,
            restart_from_best=True, verbose=True, dir_save=None, use_gpu=False):
        """
        Modified from deap/algorithms.py to return population_list instead of
        final population and use additional termination criteria (algorithm
        neuromorphic)
        Function extracted from XiCam (modified)

        Parameters
        ----------
        data: np or cp arrays of floats
            Intensities to fit
        qxs, qzs: np or cp arrays of floats
            List of qx/qz linked to intensities
        initial_guess: np or cp arrays of floats
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
            integer used to define size of default parameters
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
        use_gpu: bool, optional
            user can choose to use the GPU for the computation

        Returns
        -------
        best_corr: Parameters of line obtain after the fit
        best_fitness: Fitness value of the best_corr
        """
        # declare a global variable xp that will be changed to cp or np depending on the use_gpu in the cmaes function
        global xp
        xp = np

        if use_gpu & cp.cuda.is_available():
            xp = cp

        if dir_save is not None:
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)

        if verbose:
            print("Start CMAES")
        toolbox = dbase.Toolbox()
        residual = PicklableResidual(data, qxs, qzs, initial_guess=initial_guess, multiples=multiples,
                                 fit_mode='cmaes', xp=xp)
        
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
                population_list.append(population)

                # Evaluate the individuals
                fitnesses = toolbox.evaluate(population)

                if use_gpu & isinstance(fitnesses, xp.ndarray):
                    fitnesses = fitnesses.get()

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
        best_corr = TrapezoidGeometry.fittingp_to_simp(best_uncorr, initial_guess=initial_guess, multiples=multiples)


        if verbose:
            print(('best', best_uncorr, best_fitness))
        # make population dataframe, order of rows is first generation for all
        # children, then second generation for all children...
        
        population_arr = xp.array(
            [list(individual) for generation in population_list for individual in
            generation])
        
        population_arr = TrapezoidGeometry.fittingp_to_simp1(population_arr, initial_guess=initial_guess, multiples=multiples)

        fitness_arr = xp.array(
            [individual.fitness.values[0] for generation in population_list for
            individual in generation])
        
        # convert to numpy arrays if using GPU
        if use_gpu:
            population_arr = population_arr.get()
            fitness_arr = fitness_arr.get()

        #create a new method to save the population and fitness arrays
        population_fr = pd.DataFrame(np.column_stack((population_arr, fitness_arr)))
        if dir_save is not None:
            population_fr.to_excel(os.path.join(dir_save, "output.xlsx"))

        return best_uncorr, best_fitness
    

    def mcmc(data, qxs, qzs, initial_guess, N, multiples, sigma, nsteps, nwalkers, gaussian_move=False, parallel=True, seed=None, verbose=True, test=False, use_gpu=False):
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
            test (bool, optional): Test the function and return the mean values. Default is False.
        
        Returns:
            None

        Attributes:
            best_uncorr (numpy.ndarray): Best uncorrected individual.
            best_fitness (float): Best fitness value.
            minfitness_each_gen (numpy.ndarray): Minimum fitness at each generation.
            Sampler (emcee.EnsembleSampler): Instance of emcee.Sampler with detailed output of the algorithm.
        """
        global xp
        xp = np

        if use_gpu & cp.cuda.is_available():
            xp = cp
        
        # Create a PickeableResidual instance for data fitting
        residual = PicklableResidual(data=data, qxs=qxs, qzs=qzs, initial_guess=initial_guess, multiples=multiples, fit_mode='mcmc', xp=xp)

        def do_verbose(Sampler):
            if hasattr(Sampler, 'acceptance_fraction'):
                print('Acceptance fraction: ' + str(xp.mean(Sampler.acceptance_fraction)))
            else:
                print('Acceptance fraction: ' + str(xp.mean([Sampler.acceptance_fraction for Sampler in Sampler])))
            sys.stdout.flush()
        
        # Empirical factor to modify MCMC acceptance rate
        c = 1e-5
        
        # Generate a random seed if none is provided
        if seed is None:
            seed = randrange(2 ** 32)
        seed = seed
        xp.random.seed(seed)
        
        if hasattr(sigma, '__len__'):
            sigma = sigma
        else:
            sigma = [sigma] * N
            
            print('{} parameters'.format(N))

        if isinstance(sigma, cp.ndarray):
            sigma = sigma.get()

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
        
        
        # Data processing and analysis
        s = Sampler.chain.shape

        flatchain = xp.transpose(Sampler.chain, axes=[1, 0, 2]).reshape(s[0] * s[1], s[2])
        flatlnprobability = Sampler.lnprobability.transpose().flatten()
        minfitness_each_gen = xp.min(-Sampler.lnprobability * c, axis=0)
        
        flatfitness = -flatlnprobability * c
        best_index = np.argmin(flatfitness)
        best_fitness = flatfitness[best_index]
        best_uncorr = flatchain[best_index]
        best_corr = fittingp_to_simp(best_uncorr, initial_guess=initial_guess, multiples=multiples)
        
        
        population_array = fittingp_to_simp1(flatchain, initial_guess=initial_guess, multiples=multiples)

        if use_gpu:
            population_array = population_array.get()

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

        # to test we return the mean values and make sure that they are close to the true values
        # and convert the pandas dataframe to a numpy array
        if(test):
            mean = stats.loc['mean'].to_numpy()
            return mean[:-1] #last value is the fitness don't need it for testing

        else:  
            #save the population array
            path = os.path.join('./', 'poparr.npy')
            np.save(os.path.join(path), population_array)

            #save the stat data
            stats.to_csv(os.path.join('./', 'test.csv')) 
            print("CSV saved to {}".format(path))

