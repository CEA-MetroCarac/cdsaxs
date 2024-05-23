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
        #deep copy of the params dictionary to avoid modifying the original dictionary
        params_deepcopy = copy.deepcopy(params)

        if fitparams is not None:
            fitparams_dataframe = self.TrapezoidGeometry.convert_to_dataframe(fitparams=fitparams)
        else:
            fitparams_dataframe = None
        
        if fit_mode == 'cmaes':
            corrected_intensity = self.TrapezoidDiffraction.correct_form_factor_intensity(qys=self.qys, qzs=self.qzs, params=params_deepcopy, fitparams=fitparams, fit_mode=fit_mode, fitparams_dataframe = fitparams_dataframe)
        elif fit_mode == 'mcmc':
            corrected_intensity = self.TrapezoidDiffraction.correct_form_factor_intensity(qys=self.qys, qzs=self.qzs, params=params_deepcopy, fitparams=fitparams, fit_mode=fit_mode, best_fit=best_fit, fitparams_dataframe = fitparams_dataframe)

        if not self.from_fitter:
            return corrected_intensity[0]

        return corrected_intensity

    def set_from_fitter(self, from_fitter):
        """
        Set the from_fitter attribute and update the geometry object accordingly.

        Parameters:
        - from_fitter (bool): Indicates if the simulation is called from the fitter.
        """
        self.from_fitter = from_fitter
        self.TrapezoidGeometry.from_fitter = from_fitter
        self.TrapezoidGeometry.set_variations()
        self.TrapezoidGeometry.set_initial_guess_values()
        self.TrapezoidGeometry.set_initial_guess_dataframe()

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

    def set_initial_guess_dataframe(self):
        """
        Set the initial guess values in a dataframe

        """

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
        keys = self.initial_guess_dataframe.columns

        pd_fitparams = pd.DataFrame(fitparams, columns=keys)

        return pd_fitparams

    def rescale_fitparams(self, fitparams_df):
        """
            Rescale the gaussian distributed values obtained from fitter to the actual values and check if the values make physical sense or not
        """
        rescaled_fitparams_df = fitparams_df * self.initial_guess_dataframe.loc['variation'] + self.initial_guess_dataframe.loc['value']

        return self.check_physical_validity(rescaled_fitparams_df)
    
    def check_physical_validity(self, rescaled_fitparams_df):
        """
            Check if the values obtained from the fitter make physical sense or not
        """
        rescaled_df = rescaled_fitparams_df.copy()
        keys = rescaled_fitparams_df.columns
        for key in keys:
            if 'height' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif 'angle' in key:
                rescaled_df.loc[(rescaled_df[key] < 0) | (rescaled_df[key] > 90), key] = np.nan
            elif 'y1' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif 'bot_cd' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif 'dwx' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif 'dwz' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif 'i0' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan
            elif 'bkg_cste' in key:
                rescaled_df.loc[rescaled_df[key] < 0, key] = np.nan

        return rescaled_df


        
   
    def set_variations(self):
        """
        Set variations for each parameter in an array based on the initial guesses.

        This method sets variations for each parameter based on the initial guesses provided. It ensures that the variations are compatible with the parameter shapes and existence of certain parameters. The variations are stored in an array accessible as `self.variations`.

        Raises:
            ValueError: If heights and left angles are not compatible, or if left and right angles are not compatible.

        Returns:
            None
        """

        initial_guess = self.initial_guess

        self.ntrapezoid = len(initial_guess['langles']['value'])

        height_is_list = isinstance(initial_guess['heights']['value'], list)#height constant or not
        rangles_exists = (initial_guess['rangles']['value'] is not None and 'rangles' in self.initial_guess.keys() )#rangles exists or not

        #check if values of heights and angles have the same shape or not
        if height_is_list:
            if not (self.xp.asarray(initial_guess['heights']['value']).shape ==  self.xp.asarray(initial_guess['langles']['value']).shape) :
                raise ValueError("Heights and left angles should be compatible")
        if rangles_exists:
            if not ( self.xp.asarray(initial_guess['langles']['value']).shape == self.xp.asarray(initial_guess['rangles']['value']).shape ):
                raise ValueError("Left and right angles should be compatible")

        if initial_guess is not None:
            #get all the variations entered by the user from the initial_guess in an array
            variations = self.xp.asarray([])
            for key in initial_guess.keys():
                if key == 'heights':
                    if not height_is_list:
                        variations = self.xp.concatenate( (variations, [ initial_guess[key]['variation'] ]) )
                    else:
                        variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ] * self.ntrapezoid) ) )
                elif key == 'langles':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ] * self.ntrapezoid) ) )
                elif key == 'rangles' and rangles_exists:
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ] * self.ntrapezoid) ) )
                elif key == 'y1':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ]) ) )
                elif key == 'bot_cd':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ]) ) )
                elif key == 'dwx':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ]) ) )
                elif key == 'dwz':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ]) ) )
                elif key == 'i0':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ]) ) )
                elif key == 'bkg_cste':
                    variations = self.xp.concatenate( (variations, self.xp.asarray([ initial_guess[key]['variation'] ]) ) )

        self.variations = variations

        return None

    def set_initial_guess_values(self):

        """
            Set the initial guess values in an array

        """

        height_is_list = isinstance(self.initial_guess['heights']['value'], list)#height constant or not
        rangles_exists = (self.initial_guess['rangles']['value'] is not None and 'rangles' in self.initial_guess.keys() )#rangles exists or not


        #get all the values in the initial guess entered by the user in an array
        initial_guess_values = self.xp.asarray([])
        initial_guess = self.initial_guess

        for key in initial_guess.keys():
            if key == 'heights':
                if height_is_list:
                    initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray(initial_guess[key]['value']) ) )
                else:
                    initial_guess_values = self.xp.concatenate( (initial_guess_values, [initial_guess[key]['value']]) )
            elif key == 'langles':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray(initial_guess[key]['value']) ) )
            elif key == 'rangles' and rangles_exists:
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray(initial_guess[key]['value']) ) )
            elif key == 'y1':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray([initial_guess[key]['value']]) ) )
            elif key == 'bot_cd':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray([initial_guess[key]['value']]) ) )
            elif key == 'dwx':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray([initial_guess[key]['value']]) ) )
            elif key == 'dwz':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray([initial_guess[key]['value']]) ) )
            elif key == 'i0':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray([initial_guess[key]['value']]) ) )
            elif key == 'bkg_cste':
                initial_guess_values = self.xp.concatenate( (initial_guess_values, self.xp.asarray([initial_guess[key]['value']]) ) )
            

        #set the collected values as an attribute
        self.initial_guess_values = initial_guess_values

        return None
            
    def calculate_ycoords(self, height, langle, rangle, y1, y2):
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
        
        #modify y1 and y2 to match the shape of langle and rangle
        y1 = y1[..., self.xp.newaxis] * self.xp.ones_like(langle)
        y2 = y2[..., self.xp.newaxis] * self.xp.ones_like(langle)

        #calculate y1 and y2 for each trapezoid cumilatively using cumsum but need to preserve the first values
        #/!\ for 90 degrees, tan(90deg) is infinite so height/tan(90deg) is equal to the zero upto the precision of 10E-14 only
        y1_cumsum = self.xp.cumsum(height / self.xp.tan(langle), axis=1)
        y2_cumsum = self.xp.cumsum(height / self.xp.tan(np.pi - rangle), axis=1)

        y1[:,1:] = y1[:,1:]  +  y1_cumsum[:,:-1]
        y2[:,1:] = y2[:,1:]  + y2_cumsum[:,:-1]

        return y1, y2

    def calculate_shift(self, height, langle):
        """
        Calculate how much the trapezoids are shifted in the z direction.

        Parameters:
        - height (float or ndarray): Height for each layer of trapezoid or a float if all layers have the same height.
        - langle (ndarray): 2D array of left angles.

        Returns:
        - shift (ndarray): Array of calculated shift values.
        """
        if (height.ndim == 1):
            shift = height[:, self.xp.newaxis] * self.xp.arange(langle.shape[1])

        elif (height.shape == langle.shape):
            shift = self.xp.zeros_like(height)
            height_cumsum = self.xp.cumsum(height, axis=1)
            shift[:,1:] = height_cumsum[:,:-1]

        else:  
            raise ValueError('Height and langle should be compatible')

        return shift

    def check_params(self, params):
        """
        The purpose of this method is to check if all the required parameters are in the dictionary and to make sure that the inputs are arrays of the right shape
        Parameters:
        params: a dictionary containing all the parameters needed to calculate the form factor
        Returns:
        params: a dictionary containing all the parameters needed to calculate the form factor
        """

        #check if all the required parameters are in the dictionary
        if 'heights' not in params:
            raise ValueError('Height is required')
        if 'langles' not in params:
            raise ValueError('Left angles are required')
        if 'y1' not in params:
            raise ValueError('Y1 is required')
        if 'bot_cd' not in params:
            raise ValueError('Bottom center distance is required')
        if 'dwx' not in params:
            raise ValueError('Dwx is required')
        if 'dwz' not in params:
            raise ValueError('Dwz is required')
        if 'i0' not in params:
            raise ValueError('I0 is required')
        if 'bkg_cste' not in params:
            raise ValueError('Background constant is required')
        
        #handle the non required parameters
        if 'rangles' not in params:
            params['rangles'] = params['langles']
        if 'weight' not in params:
            params['weight'] = None
        
        #making sure that height is a 1d or 2d array and angles are 2d arrays
        height = params['heights']
        langle = self.xp.asarray(params['langles'])
        rangle = self.xp.asarray(params['rangles'])

        if isinstance(height, float) or isinstance(height, int):
            params['heights'] =  self.xp.asarray([float(height)])
        elif isinstance(height, list) or isinstance(height, self.xp.ndarray):
            params['heights'] = self.xp.asarray([height])
        else:
            raise ValueError('Height should be a float or a list')

        if langle.ndim == 1 or rangle.ndim == 1:
            params['langles'] = [langle]
            params['rangles'] = [rangle]
        elif langle.ndim != 2 or rangle.ndim != 2:
            raise ValueError('Left angles and right angles should be a 1d list or a list of lists')
    
        return params
    
    def set_fitparams_indices(self):
        """
        Calculate the indices to assign relevant values from fit_params to calculation above

        Parameters
        ----------
        fit_params: list of floats
            List of the parameters returned by the optimizer

        Returns
        -------
        simp: list of floats
            List of the parameters in the simp format
        """

        #each time we will check if the height entered by user is constant or not and if the user wants symmetric case(langle=rangle)
        height_is_list = isinstance(self.initial_guess['heights']['value'], list)
        rangles_exists = ('rangles' in self.initial_guess.keys() and self.initial_guess['rangles']['value'] is not None)

        #initialize the indices dictionary with keys from self.initial_guess
        indices = {key: {} for key in self.initial_guess.keys()}

        for key in self.initial_guess.keys():

            if key == 'heights':
                if height_is_list:
                    indices[key]['start'] = 0
                    indices[key]['stop'] = self.ntrapezoid
                else:
                    indices[key]['start'] = 0
                    indices[key]['stop'] = None

            elif key == 'langles':
                if height_is_list:
                    indices[key]['start'] = self.ntrapezoid
                    indices[key]['stop'] = 2 * self.ntrapezoid
                else:
                    indices[key]['start'] = 1
                    indices[key]['stop'] = self.ntrapezoid + 1
            
            elif key == 'rangles' and rangles_exists:
                if height_is_list:
                    indices[key]['start'] = 2 * self.ntrapezoid
                    indices[key]['stop'] = 3 * self.ntrapezoid
                else:
                    indices[key]['start'] = self.ntrapezoid + 1
                    indices[key]['stop'] = 2 * self.ntrapezoid + 1

            elif key == 'y1':
                if height_is_list and rangles_exists:
                    indices[key]['start'] = 3 * self.ntrapezoid
                    indices[key]['stop'] = None
                elif height_is_list and not rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid
                    indices[key]['stop'] = None
                elif not height_is_list and rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 1
                    indices[key]['stop'] = None
                else:
                    indices[key]['start'] = self.ntrapezoid + 2

            elif key == 'bot_cd':
                if height_is_list and rangles_exists:
                    indices[key]['start'] = 3 * self.ntrapezoid + 1
                    indices[key]['stop'] = None
                elif height_is_list and not rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 1
                    indices[key]['stop'] = None
                elif not height_is_list and rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 2
                    indices[key]['stop'] = None
                else:
                    indices[key]['start'] = self.ntrapezoid + 2
            
            elif key == 'dwx':
                if height_is_list and rangles_exists:
                    indices[key]['start'] = 3 * self.ntrapezoid + 2
                elif height_is_list and not rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 2
                elif not height_is_list and rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 3
                else:
                    indices[key]['start'] = self.ntrapezoid + 3
            
            elif key == 'dwz':
                if height_is_list and rangles_exists:
                    indices[key]['start'] = 3 * self.ntrapezoid + 3
                elif height_is_list and not rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 3
                elif not height_is_list and rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 4
                else:
                    indices[key]['start'] = self.ntrapezoid + 4
            
            elif key == 'i0':
                if height_is_list and rangles_exists:
                    indices[key]['start'] = 3 * self.ntrapezoid + 4
                elif height_is_list and not rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 4
                elif not height_is_list and rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 5
                else:
                    indices[key]['start'] = self.ntrapezoid + 5

            elif key == 'bkg_cste':
                if height_is_list and rangles_exists:
                    indices[key]['start'] = 3 * self.ntrapezoid + 5
                elif height_is_list and not rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 5
                elif not height_is_list and rangles_exists:
                    indices[key]['start'] = 2 * self.ntrapezoid + 6
                else:
                    indices[key]['start'] = self.ntrapezoid + 6

        self.fitparams_indices = indices

        return None

    def extract_params(self, fitparams=None, params=None, for_best_fit=False, for_saving=False, best_fit=None, fit_mode='cmaes'):

        """
        Extract the relevant values from the fitparams to calculate the form factor
        
        Parameters:
        - fitparams (ndarray): 2D array of floats obtained from the fitter.
        - params (dict): Dictionary containing the required parameters.
        - for_best_fit (bool): Flag indicating whether to extract parameters for best fit.
        - for_saving (bool): Flag indicating whether to extract parameters for saving.
        - best_fit (ndarray): Array containing the best fit values.
        - fit_mode (str): The fit mode used.

        Returns:
        height, langle, rangle, y1_initial, y2_initial, weight, dwx, dwz, i0, bkg_cste : extracted 2d arrays of floats needed to calculate the form factor
        
        Note:
        Based on the flags for_best_fit and for_saving, the extracted values are returned in different formats
        """
        
        
        if not self.from_fitter and params is not None:

            params = self.check_params(params)

            height = params['heights']
            langle = params['langles']
            rangle = params.get('rangles', None)#check if rangles is in the dictionary if not set it to None
            y1_initial = params['y1']
            y2_initial = params['bot_cd']
            weight = params.get('weight', None)#check if weight is in the dictionary if not set it to None
            dwx = params['dwx']
            dwz = params['dwz']
            i0 = params['i0']
            bkg_cste = params['bkg_cste']

        elif fitparams is not None:
            
            #fitparams coming from fitter is an array, so pick the right values and put them in right variables
            fitparams = self.xp.asarray(fitparams)
            
            if best_fit is None and fit_mode == 'cmaes':
                fitparams = fitparams * self.variations + self.initial_guess_values #make sure fitparams values make physical sense
            else:
                fitparams = fitparams * self.variations + best_fit #for the mcmc case where the best fit obtain from cmaes is used

            #get the indices to assign relevant values from fit_params which is an array
            if self.fitparams_indices is None:
                self.set_fitparams_indices()
            
            if not self.fitparams_indices['heights']['stop'] == None:
                height = fitparams[:, self.fitparams_indices['heights']['start']:self.fitparams_indices['heights']['stop']]
            else:
                height = fitparams[:, self.fitparams_indices['heights']['start']]

            langle = fitparams[:, self.fitparams_indices['langles']['start']:self.fitparams_indices['langles']['stop']]

            if 'rangles' in self.fitparams_indices.keys() and self.fitparams_indices['rangles']:
                rangle = fitparams[:, self.fitparams_indices['rangles']['start']:self.fitparams_indices['rangles']['stop']]
            else:
                rangle = None
            
            y1_initial = fitparams[:, self.fitparams_indices['y1']['start']] 
            y2_initial = fitparams[:, self.fitparams_indices['bot_cd']['start']]
            
            dwx = fitparams[:, self.fitparams_indices['dwx']['start']]
            dwz = fitparams[:, self.fitparams_indices['dwz']['start']]
            i0 = fitparams[:, self.fitparams_indices['i0']['start']]
            bkg_cste = fitparams[:, self.fitparams_indices['bkg_cste']['start']]
            
            weight = self.initial_guess.get('weight', None)

            #making sure all the inputs are arrays
            if not isinstance(dwx, self.xp.ndarray):
                dwx = self.xp.asarray(dwx)
            if not isinstance(dwz, self.xp.ndarray):
                dwz = self.xp.asarray(dwz)
            if not isinstance(i0, self.xp.ndarray):
                i0 = self.xp.asarray(i0)
            if not isinstance(bkg_cste, self.xp.ndarray):
                bkg_cste = self.xp.asarray(bkg_cste)

        else:
            raise ValueError('Please provide the required parameters')
            

        #making sure all the inputs are arrays
        if isinstance(height, list) and not isinstance(height, self.xp.ndarray):
            height = self.xp.asarray(height)
        if not isinstance(langle, self.xp.ndarray):
            langle = self.xp.asarray(langle)
        if not isinstance(rangle, self.xp.ndarray) and rangle is not None:
            rangle = self.xp.asarray(rangle)
        if weight and not isinstance(weight, self.xp.ndarray):
            weight = self.xp.asarray(weight)

        #check if the values make physical sense
        height = self.xp.where(height < 0, self.xp.nan , height)
        langle = self.xp.where( (langle < 0) | (langle[:,] > 91) , self.xp.nan , langle)
        if rangle is not None:
            rangle = self.xp.where( (rangle < 0) | (rangle > 91) , self.xp.nan , rangle)
        
        condition_y1 = self.xp.asarray((y1_initial < 0) | (y1_initial > y2_initial))
        y1_initial = self.xp.where(condition_y1, self.xp.nan , y1_initial)

        condition_y2 = self.xp.asarray(y2_initial < 0) | (y2_initial < y1_initial)
        y2_initial = self.xp.where(condition_y2 < 0, self.xp.nan , y2_initial)

        condition_dwx = self.xp.asarray(dwx < 0) #the mask are converted to relevant cupy (or numpy) arrays to avoid error
        dwx = self.xp.where(condition_dwx < 0, self.xp.nan , dwx)

        condition_dwz = self.xp.asarray(dwz < 0)
        dwz = self.xp.where(condition_dwz < 0, self.xp.nan , dwz)

        condition_i0 = self.xp.asarray(i0 < 0)
        i0 = self.xp.where(condition_i0, self.xp.nan , i0)

        condition_bkg_cste = self.xp.asarray(bkg_cste < 0)
        bkg_cste = self.xp.where(condition_bkg_cste < 0, self.xp.nan , bkg_cste)

        if for_best_fit:
            return {'heights': height[0], 'langles': langle[0], 'rangles': rangle[0] if isinstance(rangle, self.xp.ndarray) else None, 'y1': y1_initial[0], 'bot_cd': y2_initial[0], 'dwx': dwx[0], 'dwz': dwz[0], 'i0': i0[0], 'bkg_cste': bkg_cste[0]}


        if for_saving:
            #bundle the parameter values of same population together in a list
            population_array  = []
            for i in range(self.xp.shape(height)[0]):
                if weight is None and rangle is None:
                    population_array.append( [ height[i], langle[i], [y1_initial[i]], [y2_initial[i]], [dwx[i]], [dwz[i]], [i0[i]], [bkg_cste[i]] ] )
                elif weight is None and rangle is not None:
                    population_array.append( [ height[i], langle[i], rangle[i], [y1_initial[i]], [y2_initial[i]], [dwx[i]], [dwz[i]], [i0[i]], [bkg_cste[i]] ] )
                elif weight is not None and rangle is None:
                    population_array.append( [ height[i], langle[i], [y1_initial[i]], [y2_initial[i]], weight[i], [dwx[i]], [dwz[i]], [i0[i]], [bkg_cste[i]] ] )
                else:
                    population_array.append( [ height[i], langle[i], rangle[i], [y1_initial[i]], [y2_initial[i]], weight[i], [dwx[i]], [dwz[i]], [i0[i]], [bkg_cste[i]] ] )
                    
            return self.xp.asarray(population_array)

        return height, langle, rangle, y1_initial, y2_initial, weight, dwx, dwz, i0, bkg_cste


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
  
    def calculate_coefficients(self, qzs, height, langle, weight):
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

        shift = self.TrapezoidGeometry.calculate_shift(height, langle)
        coeff = self.xp.exp(-1j * shift[:,:, self.xp.newaxis] * qzs)

        if weight is not None:
            coeff *= weight[:, self.xp.newaxis] * (1. + 1j)

        return coeff
    
    def trapezoid_form_factor(self, qys, qzs, y1, y2, langle, rangle, height):

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
        tan1 = self.xp.tan(langle)[:,:, self.xp.newaxis]
        tan2 = self.xp.tan(np.pi - rangle)[:,:, self.xp.newaxis]
        val1 = qys + tan1 * qzs
        val2 = qys + tan2 * qzs
        with np.errstate(divide='ignore'):
            form_factor = (tan1 * self.xp.exp(-1j * qys * y1[:,:, self.xp.newaxis]) *
                        (1 - self.xp.exp(-1j * height[:,:, self.xp.newaxis] / tan1 * val1)) / val1)
            form_factor -= (tan2 * self.xp.exp(-1j * qys * y2[:,:, self.xp.newaxis]) *
                            (1 - self.xp.exp(-1j * height[:,:, self.xp.newaxis] / tan2 * val2)) / val2)
            form_factor /= qys

        return form_factor
    
    def corrections_dwi0bk(self, intensities, dw_factorx, dw_factorz,
                       scaling, bkg_cste, qxs, qzs):
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


        qxs = qxs[..., self.xp.newaxis]
        qzs = qzs[..., self.xp.newaxis]

        dw_array = self.xp.exp(-(qxs * dw_factorx) ** 2 +
                            (qzs * dw_factorz) ** 2)

        intensities_corr = (self.xp.asarray(intensities).T * dw_array * scaling
                                + bkg_cste)
        return intensities_corr.T
    
    def calculate_form_factor(self, qys, qzs, height, langle, rangle, y1_initial, y2_initial, weight):
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


        #making sure all the inputs are arrays
        if not isinstance(height, self.xp.ndarray):
            height = self.xp.asarray(height)
        if not isinstance(qys, self.xp.ndarray):
            qys = self.xp.asarray(qys)
        if not isinstance(qzs, self.xp.ndarray):
            qzs = self.xp.asarray(qzs)
        if not isinstance(langle, self.xp.ndarray):
            langle = self.xp.asarray(langle)
        if not isinstance(rangle, self.xp.ndarray):
            rangle = self.xp.asarray(rangle)
        if not isinstance(y1_initial, self.xp.ndarray):
            y1_initial = self.xp.asarray(y1_initial)
        if not isinstance(y2_initial, self.xp.ndarray):
            y2_initial = self.xp.asarray(y2_initial)
        if weight and not isinstance(weight, self.xp.ndarray):
            weight = self.xp.asarray(weight)

        if not (height.shape == langle.shape):
            height = height[..., self.xp.newaxis]#needs to be of same dimension as langles
       

        y1, y2 = self.TrapezoidGeometry.calculate_ycoords(height=height, langle=langle, rangle=rangle, y1=y1_initial, y2=y2_initial)

        form_factor = self.trapezoid_form_factor(qys=qys, qzs=qzs, y1=y1, y2=y2, langle=langle, rangle=rangle, height=height)

        return form_factor

    
    def correct_form_factor_intensity(self, qys, qzs, params=None, fitparams=None, fit_mode='cmaes', best_fit=None, fitparams_dataframe=None):
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


        #Extract the parameters
        if fitparams is not None and self.TrapezoidGeometry.from_fitter:
            
            if fitparams_dataframe is not None:
                fitparams_df = self.TrapezoidGeometry.rescale_fitparams(fitparams_df=fitparams_dataframe)
                print(fitparams_df)

            if fit_mode == 'cmaes':
                height, langle, rangle, y1_initial, y2_initial, weight, dwx, dwz, i0, bkg_cste = self.TrapezoidGeometry.extract_params(fitparams=fitparams, fit_mode=fit_mode)
            elif fit_mode == 'mcmc':
                height, langle, rangle, y1_initial, y2_initial, weight, dwx, dwz, i0, bkg_cste = self.TrapezoidGeometry.extract_params(fitparams=fitparams, fit_mode=fit_mode, best_fit=best_fit)

        elif params is not None:
            height, langle, rangle, y1_initial, y2_initial, weight, dwx, dwz, i0, bkg_cste = self.TrapezoidGeometry.extract_params(params=params)
        else:
            raise ValueError('Please provide the required parameters')

        #symmetric case
        if rangle is None:
            rangle = langle

        try:
            if CUPY_AVAILABLE and self.xp == cp:
                height = self.xp.asarray(height)
                langle = self.xp.asarray(langle)
                rangle = self.xp.asarray(rangle)
                y1_initial = self.xp.asarray(y1_initial)
                y2_initial = self.xp.asarray(y2_initial)
                dwx = self.xp.asarray(dwx)
                dwz = self.xp.asarray(dwz)
                i0 =self.xp.asarray(i0)
                bkg_cste = self.xp.asarray(bkg_cste)
                qys = self.xp.asarray(qys)
                qzs = self.xp.asarray(qzs)
        except:
            pass
        
        
        #Actual Calculations

        coeff = self.calculate_coefficients(qzs=qzs, height=height, langle=langle, weight=weight)
        form_factor = self.xp.sum(self.calculate_form_factor(qys=qys, qzs=qzs, height=height, langle=langle, rangle=rangle, y1_initial=y1_initial, y2_initial=y2_initial, weight=weight) * coeff, axis=1)
        
        form_factor_intensity = self.xp.absolute(form_factor) ** 2

        corrected_intensity = self.corrections_dwi0bk(intensities=form_factor_intensity, dw_factorx=dwx, dw_factorz=dwz, scaling=i0, bkg_cste=bkg_cste, qxs=qys, qzs=qzs)
        
        return corrected_intensity