---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Miniaturizing transistors, the fundamental components of integrated circuits, poses significant challenges for the semiconductor industry. Accurate measurement of these features during production is essential to ensure the creation of high-quality chips. However, conventional in-line metrology techniques are approaching their limitations. To address these challenges, the industry is turning to advanced X-ray-based metrology.

CD-SAXS (Critical Dimension Small Angle X-ray Scattering) is an emerging and promising technique in this field. Studies<span style="color:red">TODO: Add reference</span> have demonstrated the effectiveness of CD-SAXS in accurately characterizing the shape and spacing of nanometer-scale patterns. The cdsaxs package is designed to offer comprehensive simulation and fitting tools for CD-SAXS synchrotron data, supporting researchers in advancing this innovative technology.


# Statement of need

CD-SAXS is a powerful yet emerging technique for characterization of nano-components in semiconductor industry, but its potential is currently limited by the lack of open-source software for comprehensive data analysis. Existing tools are either proprietary or insufficiently flexible, leaving researchers with the challenge of developing their own solutions for simulating and fitting CD-SAXS data. Moreover, the diversity of samples analyzed using CD-SAXS requires versatile software that can accommodate different types of models and experimental conditions.

The cdsaxs package is designed to address this critical gap by providing a modular, open-source solution tailored for CD-SAXS data analysis. It includes two robust models for simulating CD-SAXS data, while also allowing researchers to integrate their own models. This flexibility is crucial for testing and validating models against experimental data, making the development process more streamlined and accessible.

A key feature of cdsaxs is its separation of the simulation and fitting processes, enabling users to concentrate on model development and data analysis without being encumbered by technical complexities. The package is optimized for performance, with support for parallelized fitting on both CPUs and GPUs, significantly enhancing the speed and efficiency of data processing. The fitting process in cdsaxs is powered by the CMAES (Covariance Matrix Adaptation Evolutionary Strategy) algorithm, known for its rapid convergence. This efficiency allows for real-time data fitting during experiments, empowering researchers to dynamically adjust experimental parameters based on immediate feedback from the analysis.

Additionally, it incorporates uncertainty estimation in the fitted parameters using the MCMC (Monte Carlo Markov Chain) inverse algorithm, providing researchers with more reliable and nuanced results.

By filling the current void in CD-SAXS data analysis tools, cdsaxs not only accelerates research workflows but also democratizes access to advanced analytical techniques, fostering innovation and discovery in this promising field.

# Description

The `cdsaxs` package provides a comprehensive framework for analyzing CD-SAXS data, focusing on the systematic workflow of candidate generation, evaluation, and uncertainty estimation.

1. **Candidate Generation and Evaluation**:
    - The core of the `cdsaxs` fitting process begins with generating a series of candidate parameters. Each set of parameters represents a possible nanostructure configuration, defined by a set of features(e.g., widths, heights etc).
    - These candidate models are then transformed into the reciprocal space through a Fourier Transform, allowing direct comparison with the experimental CD-SAXS data.
    - The package utilizes an optimization algorithm, specifically the Covariance Matrix Adaptation Evolutionary Strategy (CMAES), to iteratively refine the model parameters. This algorithm excels in high-dimensional optimization, rapidly converging on a solution that minimizes the error between the simulated and experimental scattering intensities.

2. **Simulation and Comparison**:
    - The simulation process can also function independently, generating CD-SAXS data based on user-defined parameters without the need for experimental data. This is particularly useful for testing and validating models in a controlled setting.
    - When experimental data is available, the package simulates scattering profiles for each candidate model and calculates a goodness-of-fit metric by comparing the simulated data with the experimental measurements. The optimization algorithm adjusts the model parameters to minimize this metric, ensuring the best possible match.

3. **Uncertainty Estimation**:
    - After determining the best-fit model, `cdsaxs` employs a Monte Carlo Markov Chain (MCMC) algorithm to estimate the uncertainties associated with the model parameters. This step is crucial for understanding the robustness of the fit and identifying potential alternative structures that could produce similar scattering data.
    - The MCMC method generates a distribution of possible parameter sets, from which the package calculates confidence intervals, providing a quantitative measure of uncertainty for each parameter.

This workflow ensures that the `cdsaxs` package not only identifies the optimal model configuration but also quantifies the confidence in the results, making it a powerful tool for CD-SAXS data analysis in both research and industrial applications.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"


# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References