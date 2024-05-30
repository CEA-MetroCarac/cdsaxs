import copy
import numpy as np
import pandas as pd

from .base import Simulation, Geometry

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except:
    CUPY_AVAILABLE = False




class StackedTrapezoidSimulation(Simulation):
    """
    A class representing a simulation of stacked trapezoids for diffraction pattern calculation.

    Attributes:
    - qys (array-like): The q-values in the y-direction for diffraction calculation.
    - qzs (array-like): The q-values in the z-direction for diffraction calculation.
    - xp (module): The module used for numerical computations (numpy or cupy).
    - from_fitter (bool): Indicates if the simulation is called from the fitter.
    - TrapezoidGeometry (StackedTrapezoidGeometry): The geometry object for the stacked trapezoids.
    - TrapezoidDiffraction (StackedTrapezoidDiffraction): The diffraction object for the stacked trapezoids.

    Methods:
    - __init__(self, qys, qzs, from_fitter=False, use_gpu=False, initial_guess=None): Initializes the StackedTrapezoidSimulation object.
    - simulate_diffraction(self, params=None, fitparams=None, fit_mode='cmaes', best_fit=None): Simulates the diffraction pattern of the stacked trapezoids.
    - set_from_fitter(self, from_fitter): Sets the from_fitter attribute and updates the geometry object accordingly.
    - geometry(self): Returns the geometry object for the stacked trapezoids.
    """
    def __init__(self, qys, qzs, from_fitter=False, use_gpu=False, initial_guess=None):
        self.qys = qys
        self.qzs = qzs
        self.xp = cp if use_gpu and CUPY_AVAILABLE else np
        self.from_fitter = from_fitter
        self.TrapezoidGeometry = StackedTrapezoidGeometry(xp=self.xp, from_fitter=self.from_fitter, initial_guess=initial_guess)
        self.TrapezoidDiffraction = StackedTrapezoidDiffraction(TrapezoidGeometry=self.TrapezoidGeometry, xp=self.xp)

    def simulate_diffraction(self, params=None, fitparams=None, fit_mode='cmaes', best_fit=None):
        """
        Simulate the diffraction pattern of the stacked trapezoids.

        Parameters:
        - params (dict): A dictionary containing all the parameters needed to calculate the form factor.
        - fitparams (list): A list of floats coming from the fitter, not given by the user but by the fitter Class.
        - fit_mode (str): The fit mode to use for diffraction calculation ('cmaes' or 'mcmc').
        - best_fit (array-like): The best fit parameters obtained from the fitter.

        Returns:
        - corrected_intensity (array-like): A 2D array of floats containing the corrected intensity. The inner lists correspond to the simulated intensity obtained by varying the parameters using the Fitter Class.
        """

        if fitparams is not None:
            fitparams_dataframe = self.TrapezoidGeometry.convert_to_dataframe(fitparams=fitparams)
        else:
            fitparams_dataframe = self.TrapezoidGeometry.convert_to_dataframe(params)
        
        if fit_mode == 'cmaes':
            corrected_intensity = self.TrapezoidDiffraction.correct_form_factor_intensity(qys=self.qys, qzs=self.qzs, fit_mode=fit_mode, fitparams_dataframe = fitparams_dataframe)
        elif fit_mode == 'mcmc':
            corrected_intensity = self.TrapezoidDiffraction.correct_form_factor_intensity(qys=self.qys, qzs=self.qzs, fit_mode=fit_mode, best_fit=best_fit, fitparams_dataframe = fitparams_dataframe)

        if not self.from_fitter:
            return corrected_intensity[0]

        return corrected_intensity

    def set_from_fitter(self, from_fitter):
        """
        Set the from_fitter attribute and update the geometry object accordingly.

        Parameters:
        - from_fitter (bool): Indicates if the simulation is called from the fitter.
        """
        self.TrapezoidGeometry.set_initial_guess_dataframe()
        self.from_fitter = from_fitter
        self.TrapezoidGeometry.from_fitter = from_fitter


    @property
    def geometry(self):
        """
        Get the geometry object for the stacked trapezoids.

        Returns:
        - TrapezoidGeometry (StackedTrapezoidGeometry): The geometry object for the stacked trapezoids.
        """
        return self.TrapezoidGeometry


class StackedTrapezoidGeometry(Geometry):
    """
    A class representing the geometry of stacked trapezoids for simulations.

    Attributes:
    - xp (module): The numpy-like module to use for array operations.
    - from_fitter (bool): Flag indicating whether the geometry is created from a fitter.
    - initial_guess (dict): Dictionary containing the initial guess values for the geometry.
    - fitparams_indices (dict): Dictionary containing the indices to assign relevant values from fit_params.
    - variations (ndarray): Array containing the variations entered by the user from the initial_guess.
    - initial_guess_values (ndarray): Array containing the initial guess values entered by the user.

    Methods:
    - set_variations(): Set the variations in an array.
    - set_initial_guess_values(): Set the initial guess values in an array.
    - calculate_ycoords(height, langle, rangle, y1, y2): Calculate y1 and y2 coordinates for each trapezoid.
    - calculate_shift(height, langle): Calculate how much the trapezoids are shifted in the z direction.
    - check_params(params): Check if all the required parameters are in the dictionary.
    - set_fitparams_indices(): Calculate the indices to assign relevant values from fit_params.
    - extract_params(fitparams=None, params=None, for_best_fit=False, for_saving=False, best_fit=None, fit_mode='cmaes'): Extract the relevant values from the fitparams to calculate the form factor.
    """

    def __init__(self, xp=np, from_fitter=False, initial_guess=None):
        """
        Parameters:
        - xp (module): The numpy-like module to use for array operations.
        - from_fitter (bool): Flag indicating whether the geometry is created from a fitter.
        - initial_guess (dict): Dictionary containing the initial guess values for the geometry.
        """
        self.xp = xp
        self.from_fitter = from_fitter
        self.initial_guess = initial_guess
        self.fitparams_indices = None
        self.variations = None
        self.initial_guess_values = None
        self.initial_guess_dataframe = None
        self.symmetric = False
        self.ntrapezoid = None


    def set_initial_guess_dataframe(self):
        """
        Set the initial guess values in a dataframe

        """

        #handle symmetric case
        if 'langles' not in self.initial_guess.keys() or 'rangles' not in self.initial_guess.keys():
            self.symmetric = True
            if 'rangles' not in self.initial_guess.keys():
                self.initial_guess['rangles'] = self.initial_guess['langles']
            elif 'langles' not in self.initial_guess.keys():
                self.initial_guess['langles'] = self.initial_guess['rangles']
            else:
                raise ValueError('either langles or rangles should be provided')

        # New dictionary to hold the modified structure
        modified_params = {}

        # Iterate over the original dictionary
        for key, value in self.initial_guess.items():
            if key in ['heights', 'langles', 'rangles']:
                if not isinstance(value['value'], (list, tuple, np.ndarray)):
                    value['value'] = [value['value']]
                for idx, val in enumerate(value['value']):
                    new_key = f"{key[:-1]}{idx+1}"  # Create new key by removing the last character and adding the index
                    modified_params[new_key] = {'value': val, 'variation': value['variation']}
            else:
                modified_params[key] = value

        # Create a DataFrame from the modified dictionary
        self.initial_guess_dataframe = pd.DataFrame(modified_params)

    def convert_to_dataframe(self, fitparams):
        """
            Convert the fitparams to a DataFrame
        """

        #extract keys from the initial guess dataframe
        if self.from_fitter:
            keys = self.initial_guess_dataframe.columns

            #remove the keys that are not needed for the symmetric case
            if self.symmetric:
                keys = [key for key in keys if not key.startswith('langle')]

            pd_fitparams = pd.DataFrame(fitparams, columns=keys)

        else:
            
            #similar to set_initial_guess_dataframe look at how many values there are in the given key if it is an array split them
            modified_params = {}

            for key in fitparams.keys():
                if key == 'heights' or key == 'langles' or key == 'rangles':
                    if isinstance(fitparams[key], (list, tuple, np.ndarray)):
                        for idx, item in enumerate(fitparams[key]):
                            new_key = f"{key[:-1]}{idx+1}"
                            modified_params[new_key] = item
                    else:
                        modified_params[key] = fitparams[key]
                else:
                    modified_params[key] = fitparams[key]
            
            pd_fitparams = pd.DataFrame([modified_params])
                        

        return pd_fitparams

    def rescale_fitparams(self, fitparams_df):
        """
            Rescale the gaussian distributed values obtained from fitter to the actual values and check if the values make physical sense or not
        """
        if self.from_fitter:
            rescaled_fitparams_df = fitparams_df * self.initial_guess_dataframe.loc['variation'] + self.initial_guess_dataframe.loc['value']
        else:
            rescaled_fitparams_df = fitparams_df

        return self.check_physical_validity(rescaled_fitparams_df)
    
    def check_physical_validity(self, rescaled_fitparams_df):
        """
            Check if the values obtained from the fitter make physical sense or not
        """
        rescaled_df = rescaled_fitparams_df.copy()
        keys = rescaled_fitparams_df.columns
        for key in keys:
            if key.startswith('height') or key in ('y1','bot_cd','dwx','dwz','i0','bkg_cste'):
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif key.startswith('rangle') or key.startswith('langle'):
                rescaled_df.loc[(rescaled_df[key] < 0) | (rescaled_df[key] > np.pi/2), key] = np.nan

        return rescaled_df

    def calculate_ycoords(self, df):
        """
        Calculate y1 and y2 coordinates for each trapezoid.

        Parameters:
        - height (float or ndarray): Height for each layer of trapezoid or a float if all layers have the same height.
        - langle (ndarray): 2D array of left angles.
        - rangle (ndarray): 2D array of right angles.
        - y1 (ndarray): 1D array of y1 values.
        - y2 (ndarray): 1D array of y2 values.

        Returns:
        - y1 (ndarray): 2D array of y1 values.
        - y2 (ndarray): 2D array of y2 values.
        """

        heights = df.filter(like='height').values
        langles= df.filter(like='langle').values
        rangles = df.filter(like='rangle').values

        y1 = df.filter(like='y1').values * self.xp.ones_like(langles)
        y2 = df.filter(like='bot_cd').values * self.xp.ones_like(langles)

        #calculate y1 and y2 for each trapezoid cumilatively using cumsum but need to preserve the first values
        #/!\ for 90 degrees, tan(90deg) is infinite so height/tan(90deg) is equal to the zero upto the precision of 10E-14 only
        y1_cumsum = self.xp.cumsum(heights / self.xp.tan(langles), axis=1)
        y2_cumsum = self.xp.cumsum(heights / self.xp.tan(np.pi - rangles), axis=1)

        y1[:,1:] = y1[:,1:]  +  y1_cumsum[:,:-1]
        y2[:,1:] = y2[:,1:]  + y2_cumsum[:,:-1]


        return y1, y2

    def calculate_shift(self, df):
        """
        Calculate how much the trapezoids are shifted in the z direction.

        Parameters:
        - height (float or ndarray): Height for each layer of trapezoid or a float if all layers have the same height.
        - langle (ndarray): 2D array of left angles.

        Returns:
        - shift (ndarray): Array of calculated shift values.
        """

        heights = df.filter(like='height').values
        langles= df.filter(like='langle').values
        
        if (heights.shape[1] == 1):
            shift = heights[:] * self.xp.arange(langles.shape[1])
        elif (heights.shape == langles.shape):
            shift = self.xp.zeros_like(heights)
            height_cumsum = self.xp.cumsum(heights, axis=1)
            shift[:,1:] = height_cumsum[:,:-1]
        else:  
            raise ValueError('Height and langle should be compatible')

        return shift  

class StackedTrapezoidDiffraction():
    """
    Class for simulating diffraction from stacked trapezoids.
    """

    
    def __init__(self, TrapezoidGeometry, xp=np):
        """
        Initialize the StackedTrapezoidDiffraction object.

        Parameters:
        -----------
        TrapezoidGeometry: object
            Object containing the geometric properties of trapezoids.
        xp: module, optional
            Module to use for mathematical operations. Default is numpy.

        Returns:
        --------
        None
        """
        self.xp = xp
        self.TrapezoidGeometry = TrapezoidGeometry
  
    def calculate_coefficients(self, qzs, df=None):
        """
        Calculate the coefficients needed to simulate intensities which takes in account the material through weight parameter.
       
       Parameters:
        -----------
        qzs: array_like
            List of qz values.
        height: array_like
            Height for each layer of trapezoids.
        langle: array_like
            Left bottom angle of trapezoid.
        weight: array_like or None
            Material weight for each layer.

        Returns:
        --------
        coefficients: array_like
            Coefficients needed to calculate the form factor.
        """

        qzs = qzs[self.xp.newaxis, self.xp.newaxis, ...]

        shift = self.TrapezoidGeometry.calculate_shift(df=df)
        coeff = self.xp.exp(-1j * shift[:,:, self.xp.newaxis] * qzs)

        weight = df.get('weight', None)
        if weight is not None:
            coeff *= weight[:, self.xp.newaxis] * (1. + 1j)

        return coeff
    
    def trapezoid_form_factor(self, qys, qzs, y1, y2, df):

        """
        Simulation of the form factor of a trapezoid at all qx, qz position.
        Function modified and extracted from XiCam

        Parameters:
        -----------
        qys, qzs: array_like
            List of qx/qz at which the form factor is simulated.
        y1, y2: array_like
            Values of the bottom right and left position of the trapezoids.
        langle, rangle: array_like
            Left and right bottom angle of trapezoid.
        height: array_like
            Height of the trapezoid.

        Returns:
        --------
        form_factor: array_like
            Values of the form factor.
        """
        heights = df.filter(like='height').values
        langles = df.filter(like='langle').values
        rangles = df.filter(like='rangle').values


        tan1 = self.xp.tan(langles)[:,:, self.xp.newaxis]
        tan2 = self.xp.tan(np.pi - rangles)[:,:, self.xp.newaxis]
        val1 = qys + tan1 * qzs
        val2 = qys + tan2 * qzs

        with np.errstate(divide='ignore'):
            form_factor = (tan1 * self.xp.exp(-1j * qys * y1[:,:, self.xp.newaxis]) *
                        (1 - self.xp.exp(-1j * heights[:,:, self.xp.newaxis] / tan1 * val1)) / val1)
            form_factor -= (tan2 * self.xp.exp(-1j * qys * y2[:,:, self.xp.newaxis]) *
                            (1 - self.xp.exp(-1j * heights[:,:, self.xp.newaxis] / tan2 * val2)) / val2)
            form_factor /= qys

        return form_factor
    
    def corrections_dwi0bk(self, intensities, qys, qzs, df):
        """
        Apply corrections to the form factor intensities.

        Parameters:
        -----------
        intensities: array_like
            Intensity values.
        dw_factorx: float
            Factor for x-direction Debye-Waller correction.
        dw_factorz: float
            Factor for z-direction Debye-Waller correction.
        scaling: float
            Scaling factor.
        bkg_cste: float
            Background constant.
        qxs, qzs: array_like
            List of qx/qz at which the form factor is simulated.

        Returns:
        --------
        intensities_corr: array_like
            Corrected form factor intensities.
        """


        qys = qys[..., self.xp.newaxis]
        qzs = qzs[..., self.xp.newaxis]

        dw_factorx = df.filter(like='dwx').values.flatten()
        dw_factorz = df.filter(like='dwz').values.flatten()
        scaling = df.filter(like='i0').values.flatten()
        bkg_cste = df.filter(like='bkg_cste').values.flatten()

        dw_array = self.xp.exp(-(qys * dw_factorx) ** 2 +
                            (qzs * dw_factorz) ** 2)

        intensities_corr = (self.xp.asarray(intensities).T * dw_array * scaling
                                + bkg_cste)
        return intensities_corr.T
    
    def calculate_form_factor(self, qys, qzs, df):
        """
        Calculate the form factor of the stacked trapezoid at all qx, qz positions.

        Parameters:
        -----------
        qys, qzs: array_like
            List of qx/qz at which the form factor is simulated.
        height: array_like
            Height of the trapezoid.
        langle, rangle: array_like
            Left and right bottom angle of trapezoid.
        y1_initial, y2_initial: array_like
            Initial values of the bottom right/left position of the trapezoids.
        weight: array_like or None
            Material weight for each layer.

        Returns:
        --------
        form_factor: array_like
            Values of the form factor.
        """


        y1, y2 = self.TrapezoidGeometry.calculate_ycoords(df=df)

        form_factor = self.trapezoid_form_factor(qys=qys, qzs=qzs, y1=y1, y2=y2, df=df)

        return form_factor

    
    def correct_form_factor_intensity(self, qys, qzs, fit_mode='cmaes', best_fit=None, fitparams_dataframe=None):
        """
        Calculate the intensites using form factor and apply debye waller corrections

        Parameters:
        -----------
        qys, qzs: array_like
            List of qx/qz at which the form factor is simulated.
        params: dict, optional
            Dictionary containing parameters needed for correction.
        fitparams: array_like, optional
            Parameters returned by optimization.
        fit_mode: str, optional
            Method used for optimization. Default is 'cmaes'.
        best_fit: array_like, optional
            Best fit obtained from optimization.

        Returns:
        --------
        corrected_intensity: array_like
            Corrected form factor intensity.
        """

        if fitparams_dataframe is not None:
            fitparams_df = self.TrapezoidGeometry.rescale_fitparams(fitparams_df=fitparams_dataframe)


        try:
            if CUPY_AVAILABLE and self.xp == cp:
                qys = self.xp.asarray(qys)
                qzs = self.xp.asarray(qzs)
        except:
            pass
        
        
        #Actual Calculations

        coeff = self.calculate_coefficients(qzs=qzs, df = fitparams_df)

        form_factor = self.xp.sum(self.calculate_form_factor(qys=qys, qzs=qzs, df=fitparams_df) * coeff, axis=1)
        
        form_factor_intensity = self.xp.absolute(form_factor) ** 2

        corrected_intensity = self.corrections_dwi0bk(intensities=form_factor_intensity, qys=qys, qzs=qzs, df=fitparams_df)
        
        print(corrected_intensity)
        
        return corrected_intensity