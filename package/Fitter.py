from Residual import PicklableResidual
from StackedTrapezoidSimulation import StackedTrapezoidSimulation
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

creator.create('FitnessMin', dbase.Fitness, weights=(-1.,))  # to minim. fitness
creator.create('Individual', list, fitness=creator.FitnessMin)


class Fitter:
    
    def __init__(self, Simulation, exp_data):
        self.Simulation = Simulation
        self.exp_data = exp_data
        self.xp = Simulation.xp

    def cmaes(self, sigma, ngen,
            popsize, mu, n_default, restarts, tolhistfun, ftarget,
            restart_from_best=True, verbose=True, dir_save=None):
        """
        Modified from deap/algorithms.py to return population_list instead of
        final population and use additional termination criteria (algorithm
        neuromorphic)
        Function extracted from XiCam (modified)

        Parameters
        ----------
        sigma: float
            Initial standard deviation for each parameter.
        ngen: int
            Number of generations.
        popsize: int
            Population size.
        mu: int
            Number of parents/points for recombination.
        n_default: int
            Number of parameters.
        restarts: int
            Number of restarts.
        tolhistfun: float
            Tolerance for the history of the best fitness value.
        ftarget: float
            Target fitness value.
        restart_from_best: bool
            Restart from the best individual.
        verbose: bool
            Print progress information.
        dir_save: str
            Directory to save the output.
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
        Residual = PicklableResidual(self.exp_data, fit_mode='cmaes', xp=np, Simulation=self.Simulation)
        toolbox.register('evaluate', Residual)
        

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
        best_corr = self.Simulation.TrapezoidGeometry.extract_params([best_uncorr], for_best_fit=True)


        if verbose:
            print(('best', best_uncorr, best_fitness))
        # make population dataframe, order of rows is first generation for all
        # children, then second generation for all children...
        
        population_arr = np.array(
            [list(individual) for generation in population_list for individual in
            generation])
        
        population_arr = self.Simulation.TrapezoidGeometry.extract_params(population_arr, for_saving=True)

        fitness_arr = np.array(
            [ [individual.fitness.values[0]] for generation in population_list for
            individual in generation])
        
        # convert to numpy arrays if using GPU
        if xp == cp:
            population_arr = population_arr.get()
            fitness_arr = fitness_arr.get()

        # create a new method to save the population and fitness arrays
        population_with_fitness = np.asarray([np.append(population_arr[i], fitness_arr[i]) for i in range(len(population_arr))])
        
        population_with_fitness = population_with_fitness.reshape( ( popsize, n_default + 1 ) ) #plus one for fitness


        population_fr = pd.DataFrame( population_with_fitness )

        if dir_save is not None:
            population_fr.to_csv(os.path.join(dir_save, "output.csv"))

        return best_corr, best_fitness
    

    def mcmc(self, N, sigma, nsteps, nwalkers, gaussian_move=False, seed=None, verbose=False, test=True):
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
        #cupy or numpy
        xp = self.xp

        # Setting Simulation attribute to match the case
        self.Simulation.set_from_fitter(True)

        #declare Fitness function and register
        Residual = PicklableResidual(self.exp_data, fit_mode='mcmc', xp=np, Simulation=self.Simulation)


        def do_verbose(Sampler):
            if hasattr(Sampler, 'acceptance_fraction'):
                print('Acceptance fraction: ' + str(xp.mean(Sampler.acceptance_fraction)))
            else:
                print('Acceptance fraction: ' + str(xp.mean([Sampler.acceptance_fraction for Sampler in Sampler])))
            sys.stdout.flush()
        
        # Empirical factor to modify MCMC acceptance rate
        c = Residual.c
        
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

        sigma = xp.array(sigma)

        if isinstance(sigma, cp.ndarray):
            sigma = sigma.get()

        if gaussian_move:
            # Use Gaussian move for the proposal distribution
            individuals = [np.random.uniform(-100, 100, N) for _ in range(nwalkers)]
            
            Sampler = emcee.EnsembleSampler(nwalkers, N, Residual, moves=emcee.moves.GaussianMove(sigma), pool=None, vectorize=True)

            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
        else:

            individuals = [np.random.default_rng().normal(loc=0, scale=sigma, size=sigma.shape) for _ in range(nwalkers)]
            Sampler = emcee.EnsembleSampler(nwalkers, N, Residual, pool=None, vectorize=True)
                
            Sampler.run_mcmc(individuals, nsteps, progress=True)

            if verbose:
                do_verbose(Sampler)
            

        #Autocorelation time to find burnin steps
        tau = Sampler.get_autocorr_time(tol=5)
        burnin = int(2 * np.max(tau))
        
        
        # Data processing and analysis
        s = Sampler.get_chain(discard=burnin).shape

        flatchain = xp.transpose(Sampler.get_chain(discard=burnin), axes=[1, 0, 2]).reshape(s[0] * s[1], s[2])
        flatlnprobability = Sampler.get_log_prob(discard=burnin).transpose().flatten()
        minfitness_each_gen = xp.min(-Sampler.get_log_prob(discard=burnin) * c, axis=0)
        
        #convert log probability to usual fitness
        flatfitness = -flatlnprobability * c

        #find the best individual and convert it to the correct form
        best_index = np.argmin(flatfitness)
        best_fitness = flatfitness[best_index]
        best_uncorr = flatchain[best_index]
        best_corr = self.Simulation.TrapezoidGeometry.extract_params(fitparams=[best_uncorr], for_best_fit=True)
      
        
        population_array = self.Simulation.TrapezoidGeometry.extract_params(flatchain, for_saving=True)
    
        if xp == cp:
            population_array = population_array.get()

        # create a new method to save the population and fitness arrays
        population_with_fitness = np.asarray([np.append(population_array[i], flatfitness[i]) for i in range(len(population_array))])
        population_with_fitness = population_with_fitness.reshape( ( s[0]*s[1] , N+1 ) ) #plus one for fitness

        population_frame = pd.DataFrame( population_with_fitness )

        gen_start = 0
        gen_stop = len(flatfitness)
        gen_step = 1
        popsize = int(population_frame.shape[0] / len(flatfitness))
        index = []
        for i in range(gen_start, gen_stop, gen_step):
            index.extend(list(range(i * popsize, (i + 1) * popsize)))
        resampled_frame = population_frame.iloc[index]
        #remove nan and inf values
        resampled_frame = resampled_frame.dropna()
        stats = resampled_frame.describe()

        if(test):
            return best_corr

        else:  
            #save the population array
            # path = os.path.join('./', 'poparr.npy')
            # np.save(os.path.join(path), population_array)

            #save the stat data
            stats.to_csv(os.path.join('./', 'test.csv'))




#######################################TESTING############################################

pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 20)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)

# Define initial parameters and multiples

#Initial parameters
dwx = 0.1
dwz = 0.1
i0 = 10
bkg = 0.1
y1 = 0
height = [23.48]
bot_cd = 54.6
swa = [85]

langle = np.deg2rad(np.asarray(swa))
rangle = np.deg2rad(np.asarray(swa))

#simulation data
params = {'heights': height,
            'langles': langle,
            'rangles': rangle,
            'y1': y1,
            'bot_cd': bot_cd,
            'dwx': dwx,
            'dwz': dwz,
            'i0': i0,
            'bkg_cste': bkg
            }

Simulation1 = StackedTrapezoidSimulation(xp=np, qys=qxs, qzs=qzs)

intensity = Simulation1.simulate_diffraction(params=params)

initial_params = {'heights': {'value': height, 'variation': 10E-5},
                    'langles': {'value': langle, 'variation': 10E-5},
                    'rangles': {'value': rangle, 'variation': 10E-5},
                    'y1': {'value': y1, 'variation': 10E-5},
                    'bot_cd': {'value': bot_cd, 'variation': 10E-5},
                    'dwx': {'value': dwx, 'variation': 10E-5},
                    'dwz': {'value': dwz, 'variation': 10E-5},
                    'i0': {'value': i0, 'variation': 10E-5},
                    'bkg_cste': {'value': bkg, 'variation': 10E-5}
                    }

Simulation2 = StackedTrapezoidSimulation(xp=np, qys=qxs, qzs=qzs, initial_guess=initial_params)

Fitter1 = Fitter(Simulation=Simulation2, exp_data=intensity)

# print(Fitter1.mcmc(N=9, sigma = np.asarray([100] * 9), nsteps=1000, nwalkers=100, test=True))
Fitter1.cmaes(sigma=100, ngen=80, popsize=40, mu=10, n_default=9, restarts=10, tolhistfun=10, ftarget=10, restart_from_best=True, verbose=False, dir_save='.')