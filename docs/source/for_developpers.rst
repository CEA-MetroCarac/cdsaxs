Setting Up a Development Environment
====================================

When developing cdsaxs or running the latest version from GitHub, the setup process differs from a standard installation via PyPI. Start by following the steps to create a Python virtual environment as outlined in the :ref:`installation` section. Once your environment is ready, proceed with the instructions below to install from a git clone.

.. _`installing from a git clone`:

Installing from a Git Clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    There is a key difference between installing a released version and installing the latest development version. Both :ref:`installing from PyPI <installation>` and :ref:`installing from a git clone` involve using pip, but they serve different purposes. The command :code:`python -m pip install cdsaxs` pulls the latest stable release from PyPI.

    On the other hand, navigating to a git clone and running :code:`python -m pip install -e .` installs the package in "editable mode," linking the source directory to the Python environment. This setup is ideal for development since changes in the source code are immediately reflected in the environment.

    For development and using the :ref:`latest features from the development branch <continuous>`, installing from a git clone is recommended. For general use or new users, the PyPI installation is simpler and more stable.

To work with the latest development version, you need to clone the cdsaxs repository. Ensure that git is installed on your system. On Linux, install git using your package manager. On Windows, several clients are available, such as `Git for Windows <https://gitforwindows.org/>`_, `GitHub Desktop <https://desktop.github.com/>`_, `TortoiseGit <https://tortoisegit.org/>`_, or the git integration in your development environment.

Clone the repository using the following command:

.. code-block:: shell

    $ git clone https://github.com/cdsaxs/cdsaxs

If you intend to contribute, fork the repository on GitHub first:

#. Log into your `GitHub <https://github.com/>`_ account.

#. Visit the `cdsaxs GitHub <https://github.com/CEA-MetroCarac/cdsaxs>`_ page.

#. Click on the *fork* button:

    ..  figure:: ../../assets/images/fork.png

#. Clone your forked repository:

.. code-block:: shell

    $ git clone https://github.com/your-username/cdsaxs

For more guidance on `forking a repository <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ or to learn git basics, explore these resources:

* A `free course <https://www.udacity.com/course/version-control-with-git--ud123>`_ covering Git fundamentals.
* A `pull request practice <https://github.com/firstcontributions/first-contributions>`_ repository.
* A sample `development workflow <https://docs.astropy.org/en/latest/development/workflow/development_workflow.html>`_ for contributing code.

With the Python environment activated (using conda or virtualenv), navigate to the directory where you cloned cdsaxs. Start the installation with the following command (note the dot at the end, indicating the current directory):

.. code-block:: shell

    (cdsaxs) $ python -m pip install -e .

This command installs cdsaxs and its dependencies in your environment.

To install extra dependencies, use the same command format as you would when installing cdsaxs from PyPI:

.. code-block:: shell

    (cdsaxs) $ python -m pip install -e .[gpu]

Updating Your Git Clone
~~~~~~~~~~~~~~~~~~~~~~~~

If you've installed cdsaxs from a git clone, keeping it up-to-date is straightforward. Open a terminal in the cdsaxs directory and run:

.. code-block:: shell

    $ git pull

Since cdsaxs was installed in "editable mode," any changes pulled from the repository are immediately applied. However, if new dependencies are introduced, re-run the installation command to ensure all packages are up to date:

.. code-block:: shell

    $ python -m pip install -e .


Project Structure
=================

To gain a clear understanding of how the code works, the design of the algorithm is illustrated with the help of a UML diagram in :numref:`fig-UML`. This diagram offers a roadmap, highlighting the various components and their interactions. We will then delve deeper to explore each component's role in the algorithm.

Components
~~~~~~~~~~

**Fitter:**

This class is a crucial component of the design. It includes the *cmaes* function for estimating the best-fit parameters and the *mcmc* function for assessing the uncertainty in the fit. The class takes a simulation model and experimental data as input. When the *cmaes* function is called, it returns the best-fit parameters. Subsequently, the *mcmc* function can be invoked to provide statistical information about the best fit, including the uncertainties in the parameters.

**Residual:**

This class calculates the residuals between the experimental data and the model. Currently, we use the log-likelihood as the residual function, but it can be easily extended to other residual functions. The *Fitter* class calls this class and provides the relevant model. The *Residual* class then uses the model *simulate_diffraction* function to calculate the model diffraction pattern and compare it with the experimental data.

.. figure:: ../../assets/images/cdsaxs_UML.png
    :width: 70%
    :align: center

    UML diagram of the design for CD-SAXS simulation application.

**Interface Simulation:**

An interface defines a contract that classes must follow, specifying a set of methods that implementing classes should have. In Python, I have chosen to use Protocol to achieve a similar effect. This is the base class for all simulation models. The functions and classes used in it should be implemented in all simulation models. This interface simplifies future model building by ensuring that simulation functions are not geometry-dependent.

**Interface Geometry:**

Similar to the Interface Simulation, this is the base class for all geometry models. This interface simplifies future model building by ensuring that geometry functions are not dependent on the specific simulation details.

**Model:**

In this UML diagram (:numref:`fig-UML`), we present the implementation of the stacked trapezoid model. Central to this model is the *StackedTrapezoidSimulation* class, which is a composite class integrating *StackedTrapezoidGeometry* and *StackedTrapezoidDiffraction* classes. The *StackedTrapezoidGeometry* class handles all geometrical calculations and stores the relevant information, while the *StackedTrapezoidDiffraction* class is responsible for all diffraction-related calculations.

These classes work together to simulate the physics and generate data that can be compared with experimental results. Throughout the project, additional models are being developed and will be discussed later. The stacked trapezoid model serves as a prototype, illustrating how other models will be implemented. Each model will adhere to the base interfaces *Simulation* and *Geometry*.

Relationships between components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This structure is designed to simulate and analyze diffraction patterns using a modular approach. At its core, the system comprises several key components: the *Fitter*, *Residual*, and interfaces for *Simulation* and *Geometry*, along with specific model implementations like the *StackedTrapezoidSimulation*. The *Fitter* class orchestrates the fitting process by using the *cmaes* function to estimate the best-fit parameters and the *mcmc* function to assess the uncertainty in the fit. It takes in a model and experimental data, utilizing the *Residual* class to compute the difference between the experimental data and the model simulated data. The *Residual* class leverages the model *simulate_diffraction* function to generate the diffraction pattern, which it then compares with the experimental data.

The model, meaning specific implementations of the interfaces, is designed to operate independently, allowing users to utilize a particular model for simulations without needing to engage with the fitting component. This autonomous functionality ensures that users can easily perform simulations solely with the model of their choice. This design choice enhances flexibility and usability, as it decouples the simulation process from the fitting procedures, making it more accessible for users who may only need to run simulations. The usefulness and implications of this design choice will be discussed shortly.



Creating a New Simulation Model
============================

This section provides guidelines on how to develop a new simulation model that adheres to the established protocol. The protocol ensures compatibility between your simulation model and the fitter class, enabling seamless integration and functionality.

Overview
--------

The protocol consists of a set of methods and properties that your `Simulation` and `Geometry` classes must implement. These classes form the core of your simulation model, handling the setup, execution, and data management for your simulations.

Key Components
--------------

1. **Geometry Class**

   The `Geometry` class is responsible for defining the shape and structure of the system being simulated. It includes methods for converting simulation parameters into a structured format that can be used by the simulation engine.

   - **convert_to_dataframe(fitparams)**: This method takes a list of fitting parameters (fitparams) and converts them into a pandas DataFrame. This DataFrame should be structured based on the initial guess values provided by the user.

     - **Parameters**:
       - `fitparams (list)`: Array containing the parameters returned by the fitter.

     - **Returns**:
       - `pandas.DataFrame`: A DataFrame containing the parameters in a readable format.

2. **Simulation Class**

   The `Simulation` class orchestrates the simulation process. It uses the `Geometry` class to manage system structure and executes the core simulation logic.

   - **geometry (property)**: This property returns an instance of the `Geometry` class, providing access to the geometric data of the system.

   - **set_from_fitter(from_fitter)**: This method configures the simulation to recognize that the incoming data originates from a fitter object. It should also initialize necessary components, such as saving the initial guess to a DataFrame.

     - **Parameters**:
       - `from_fitter (bool)`: Indicates whether the simulation is being driven by a fitter.

   - **simulate_diffraction(fitparams=None, fit_mode='cmaes', best_fit=None)**: This method performs the actual simulation, generating the diffraction pattern or other relevant output based on the provided fitting parameters.

     - **Parameters**:
       - `fitparams (list, optional)`: Parameters for the simulation, typically obtained from a fitter.
       - `fit_mode (str, optional)`: Specifies the fitting method used ('cmaes' or 'mcmc').
       - `best_fit (array-like, optional)`: The best-fitting parameters obtained from the fitter.

     - **Raises**:
       - `NotImplementedError`: If the method is not implemented in the derived class.


To develop a new model, you need to create classes that inherit from the `Simulation` and `Geometry` protocols and implement all required methods. Here's how to approach this:

1. **Define the Geometry**

   Create a class that implements the `Geometry` protocol. This class should define the specific geometric properties of the system being simulated. For example, in a model of stacked trapezoids, this class would define the dimensions, angles, and positions of the trapezoids.

   Implement the `convert_to_dataframe` method to organize the input parameters into a structured format suitable for the simulation. This method is crucial for ensuring that the parameters can be easily interpreted and manipulated by the `Fitter` class.

2. **Create the Simulation Class**

   Your simulation class should inherit from the `Simulation` protocol. This class is where the core logic of your simulation resides. It should use the geometric data from the `Geometry` class to set up and run the simulation.

   Ensure that the `simulate_diffraction` method is implemented to perform the simulation. Depending on the complexity of your model, this method might involve extensive calculations or integrations.

3. **Integration with the Fitter**

   If your simulation will be used in conjunction with a fitter (e.g., for parameter optimization), make sure to implement the `set_from_fitter` method. This will enable your simulation to correctly handle data provided by the fitter and return results in a format that the fitter can use.

Conclusion
----------

By following these guidelines, you can develop a new simulation model that integrates smoothly with the existing framework. The key is to ensure that all required methods and properties are properly implemented, allowing your model to function seamlessly within the larger simulation and fitting ecosystem.
For getting concrete examples, you can refer to the existing models in the `cdsaxs` package, such as the Stacked Trapezoid and Strong Castle models. 


Testing
=======

Pytest is used for testing the codebase. The tests are located in the `tests` directory and are organized by module. When adding new features or modifying existing code, 
it is essential to write tests and check that the existing tests pass.

To run the tests, navigate to the root directory of the project and execute the following command:

.. code-block:: shell

    (cdsaxs) $ pytest