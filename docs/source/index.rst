.. cdsaxs documentation master file, created by
   sphinx-quickstart on Tue Jul 30 15:43:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Description
===========

The `cdsaxs` package provides a comprehensive framework for analyzing CD-SAXS data, focusing on the systematic workflow of candidate generation, evaluation, and uncertainty estimation. It is also flexible enough to accommodate user-defined models, making it a versatile tool for researchers working with diverse nanostructures.

1. **Candidate Generation and Evaluation**:
   
   - The core of the `cdsaxs` fitting process begins with generating a series of candidate parameters. Each set of parameters represents a possible nanostructure configuration, defined by a set of features (e.g., widths, heights, etc.).
   
   - These candidate models are then transformed into the reciprocal space through a Fourier Transform, allowing direct comparison with the experimental CD-SAXS data.
   
   - The package utilizes an optimization algorithm, specifically the Covariance Matrix Adaptation Evolutionary Strategy (CMAES), to iteratively refine the model parameters. This algorithm excels in high-dimensional optimization, rapidly converging on a solution that minimizes the error between the simulated and experimental scattering intensities.

2. **Simulation and Comparison**:
   
   - The simulation process can also function independently, generating CD-SAXS data based on user-defined parameters without the need for experimental data. This is particularly useful for testing and validating models in a controlled setting.
   
   - In addition to the two built-in models, users can define their own models, allowing for a wide range of nanostructure configurations to be tested.
   
   - When experimental data is available, the package simulates scattering profiles for each candidate model and calculates a goodness-of-fit metric by comparing the simulated data with the experimental measurements. The optimization algorithm adjusts the model parameters to minimize this metric, ensuring the best possible match.

3. **Uncertainty Estimation**:
   
   - After determining the best-fit model, `cdsaxs` employs a Monte Carlo Markov Chain (MCMC) algorithm to estimate the uncertainties associated with the model parameters. This step is crucial for understanding the robustness of the fit and identifying potential alternative structures that could produce similar scattering data.
   
   - The MCMC method generates a distribution of possible parameter sets, from which the package calculates confidence intervals, providing a quantitative measure of uncertainty for each parameter.

This workflow ensures that the `cdsaxs` package not only identifies the optimal model configuration but also quantifies the confidence in the results, making it a powerful tool for CD-SAXS data analysis in both research and industrial applications.


cdsaxs documentation
====================

.. toctree::
   :maxdepth: 3
   :caption: User Guide:

   introduction
   tutorials
   modules

.. toctree::
   :maxdepth: 2
   :caption: For Developers:

   for_developpers


