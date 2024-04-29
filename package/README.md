# How to Use cdsaxs Package for Diffraction Simulation and Fitting

Below are the instructions on how to perform diffraction simulation and fitting using the cdsaxs fitting package.

## Diffraction Simulation

### Prepare the Data

```python
from cdsaxs_fitting.simulations.stacked_trapezoid import StackedTrapezoidSimulation
import numpy as np

# Define parameters
pitch = 100  # nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 10)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)
dwx = 0.1
dwz = 0.1
i0 = 10
bkg = 0.1
y1 = 0.
height = [20.]
bot_cd = 40.
swa = [90.]

langle = np.deg2rad(np.asarray(swa))
rangle = np.deg2rad(np.asarray(swa))

# Simulation data
i_params = {
    'heights': np.asarray(height),
    'langles': langle,
    'rangles': rangle,
    'y1': y1,
    'bot_cd': bot_cd,
    'dwx': dwx,
    'dwz': dwz,
    'i0': i0,
    'bkg_cste': bkg
}

# Create instance of the Simulation class and call the simulation method
Simulation1 = StackedTrapezoidSimulation(qys=qxs, qzs=qzs)
intensity = Simulation1.simulate_diffraction(params=i_params)
```
Note: heights can be of format string in which case it will be supposed that all the trapezoids have the same height. If it is a list, then the heights of the trapezoids will be assigned according to the order of the list.

Also, if fit is for symmetric case rangles can be ommited from the dictionary above or set to ```None``` and the code will assume that rangles = langles.

## Data Fitting

### Prepare the Data

```python
from cdsaxs_fitting.fitter import Fitter

# Initial parameters
initial_params = {
    'heights': {'value': height, 'variation': 10E-5},
    'langles': {'value': langle, 'variation': 10E-5},
    'rangles': {'value': rangle, 'variation': 10E-5},
    'y1': {'value': y1, 'variation': 10E-5},
    'bot_cd': {'value': bot_cd, 'variation': 10E-5},
    'dwx': {'value': dwx, 'variation': 10E-5},
    'dwz': {'value': dwz, 'variation': 10E-5},
    'i0': {'value': i0, 'variation': 10E-5},
    'bkg_cste': {'value': bkg, 'variation': 10E-5}
}

# Create instance of the Simulation class and pass it to the Fitter class along with data to fit
Simulation2 = StackedTrapezoidSimulation(qys=qxs, qzs=qzs, initial_guess=initial_params)
Fitter1 = Fitter(Simulation=Simulation2, exp_data=intensity)
```

### Fit the Data

First you can do CMA-ES.
#### CMA-ES

```python
cmaes = Fitter1.cmaes(sigma=100, ngen=10, popsize=10, mu=10, n_default=9, restarts=10, tolhistfun=10E-5, ftarget=10, restart_from_best=True, verbose=False, dir_save="./")
```
Then use MCMC method to give you statistics of the best fit.
#### MCMC

```python
mcmc = Fitter1.mcmc(N=9, sigma=np.asarray([100] * 9), nsteps=1000, nwalkers=18)
```
