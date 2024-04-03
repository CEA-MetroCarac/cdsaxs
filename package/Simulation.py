import numpy as np
import cupy as cp


class TrapezoidGeometry:

    def __init__(self, multiples, xp=np, from_fitter=False, initial_guess= None):
        self.multiples = multiples
        self.xp = xp
        self.from_fitter = from_fitter
        self.initial_guess = initial_guess

        
        if initial_guess is not None:
            #get all the variations entered by the user from the initial_guess in an array
            variations = self.xp.asarray([])
            for key in initial_guess.keys():
                if key == 'heights':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'langles':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'rangles':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'botcd':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'dwx':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'dwz':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'i0':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
                elif key == 'bkg_cste':
                    variations = self.xp.concatenate((variations, initial_guess[key]['variation']))
            
            #set the collected variations as an attribute
            self.variations = variations


            #check if values of heights and angles have the same shape or not
            if not (self.xp.asarray(initial_guess['heights']['value']).shape ==  self.xp.asarray(initial_guess['langles']['value']).shape and self.xp.asarray(initial_guess['langles']['value']).shape == self.xp.asarray(initial_guess['rangles']['value']).shape) :
                raise ValueError("Heights and left and right angles should be compatible")
                
            
            #get all the values in the initial guess entered by the user in an array
            initial_guess_values = self.xp.asarray([])
            for key in initial_guess.keys():
                if key == 'heights':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'langles':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'rangles':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'botcd':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'dwx':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'dwz':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'i0':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                elif key == 'bkg_cste':
                    initial_guess_values = self.xp.concatenate((initial_guess_values, initial_guess[key]['value']))
                

            #set the collected values as an attribute
            self.initial_guess_values = initial_guess_values

    @staticmethod
    def calculate_shift(self, height, langle):
        """
        This method calculates how much the trapezoids are shifted in the z direction
        @param
        height: a list of floats containing height for each layer 
                of trapezoid or a float if all the layers have the same height
        langle: a 2d array of floats
        @return
        shift: a list with calculated shift values
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
    
    @staticmethod
    def calculate_coefficients(self, qzs, height, langle, weight):
        """
        @param
        qz: a list of floats
        height: a list of floats containing height for each layer 
                of trapezoid or a float if all the layers have the same height
        langle: a 2d array of floats
        weight: a list of floats or None
        @return
        coefficients: a 3d list of floats needed to calculate the form factor
        """

        qzs = qzs[self.xp.newaxis, self.xp.newaxis, ...]

        shift = self.calculate_shift(self, height, langle)
        coeff = self.xp.exp(-1j * shift[:,:, self.xp.newaxis] * qzs)

        if weight is not None:
            coeff *= weight[:, self.xp.newaxis] * (1. + 1j)

        return coeff
    
    @staticmethod
    def calculate_ycoords(self, height, langle, rangle, y1, y2):
        """
        @param
        height: a list of floats containing height for each layer 
                of trapezoid or a float if all the layers have the same height
        langle: a 2d array of floats
        rangle: a 2d array of floats
        y1: a list of floats
        y2: a list of floats
        @return
        y1, y2: a 2d array of floats
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

    @staticmethod
    def trapezoid_form_factor(self, qys, qzs, y1, y2, langle, rangle, height):

        """
        Simulation of the form factor of a trapezoid at all qx, qz position.
        Function extracted from XiCam

        Parameters
        ----------
        qys, qzs: numpy or cupy array of floats
            List of qx/qz at which the form factor is simulated
        y1, y2: numpy or cupy array of floats
            Values of the bottom right/left (y1/y2) position respectively of
            the trapezoids such as y2 - y1 = width of the bottom of the trapezoids
        langle, rangle: numpy or cupy 2d array of floats
            Left and right bottom angle of trapezoid
        height: numpy or cupy of floats
            Height of the trapezoid

        Returns
        -------
        form_factor: numpy or cupy 2d array of floats
            List of the values of the form factor
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
    
    @staticmethod
    def corrections_dwi0bk(self, intensities, dw_factorx, dw_factorz,
                       scaling, bkg_cste, qxs, qzs):
        """
        Apply corrections to the form factor
        @param
        intensities: a 2d array of floats
        dw_factorx: a float
        dw_factorz: a float
        """

        qxs = qxs[..., self.xp.newaxis]
        qzs = qzs[..., self.xp.newaxis]

        dw_array = self.xp.exp(-(qxs * dw_factorx) ** 2 +
                            (qzs * dw_factorz) ** 2)

        intensities_corr = (self.xp.asarray(intensities).T * dw_array * scaling
                                + bkg_cste)
        return intensities_corr.T
    
    def calculate_form_factor(self, qys, qzs, params):
        """
        @param
        params: a dictionary containing all the parameters needed to calculate the form factor
        @return
        form_factor: a 1d array of floats
        """
        height = params['heights']
        langle = params['langles']
        rangle = params.get('rangles', langle)#if rangle is not in the dictionary set it to langle
        y1_initial = params['y1']
        y2_initial = params['bot_cd']
        weight = params.get('weight', None)#check if weight is in the dictionary if not set it to None

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
       

        y1, y2 = self.calculate_ycoords(self, height=height, langle=langle, rangle=rangle, y1=y1_initial, y2=y2_initial)

        form_factor = self.trapezoid_form_factor(self, qys=qys, qzs=qzs, y1=y1, y2=y2, langle=langle, rangle=rangle, height=height)

        return form_factor

    
    def correct_form_factor_intensity(self, qys, qzs, params):
        """
        @param
        params: a dictionary containing all the parameters needed to calculate the form factor
        @return
        form_factor_intensity: a 2d array of floats
        """

        #if the parameters not from Fitter class check params and correct if needed
        if not self.from_fitter:
            params = self.check_params(params)

        height = params['heights']
        langle = params['langles']
        rangle = params.get('rangles', langle)#if rangle is not in the dictionary set it to langle
        weight = params.get('weight', None)#check if weight is in the dictionary if not set it to None

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
        if weight and not isinstance(weight, self.xp.ndarray):
            weight = self.xp.asarray(weight)

        coeff = self.calculate_coefficients(self, qzs=qzs, height=height, langle=langle, weight=weight)
        form_factor = self.xp.sum(self.calculate_form_factor(qys=qys, qzs=qzs, params=params)*coeff, axis=1)
        
        form_factor_intensity = self.xp.absolute(form_factor) ** 2

        dwx = params['dwx']
        dwz = params['dwz']
        i0 = params['i0']
        bkg_cste = params['bkg_cste']

        corrected_intensity = self.corrections_dwi0bk(self, intensities=form_factor_intensity, dw_factorx=dwx, dw_factorz=dwz, scaling=i0, bkg_cste=bkg_cste, qxs=qys, qzs=qzs)
        
        return corrected_intensity
    
    def check_params(self, params):
        """
        The purpose of this method is to check if all the required parameters are in the dictionary and to make sure that the inputs are arrays of the right shape
        @param
        params: a dictionary containing all the parameters needed to calculate the form factor
        @return
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

        if isinstance(height, float):
            params['heights'] =  self.xp.asarray([height])
        elif isinstance(height, list):
            params['heights'] = self.xp.asarray([height])
        else:
            raise ValueError('Height should be a float or a list')

        if langle.ndim == 1 or rangle.ndim == 1:
            params['langles'] = [langle]
            params['rangles'] = [rangle]
        elif langle.ndim != 2 or rangle.ndim != 2:
            raise ValueError('Left angles and right angles should be a 1d list or a list of lists')
    


        return params
    
    def convert_fitparams(self, fit_params):
        """
        Convert the fit_params from the optimizer to the simp format used above in __call__

        Parameters
        ----------
        fit_params: list of floats
            List of the parameters returned by the optimizer

        Returns
        -------
        simp: list of floats
            List of the parameters in the simp format
        """

        #convert fitparams to sensible data
        fit_params = fit_params * self.variations + self.initial_guess_values

        #determine the number of trapezoids
        ntrapezoid = len(self.initial_guess['langles']['value'])

        #create a dictionary to store the parametres obtained from the fitter
        simp = {}

        #each time we will check if the height entered by user is constant or not and if the user wants symmetric case(langle=rangle) or not
        height_is_list = isinstance(self.initial_guess[key]['value'], list)
        rangles_exists = 'rangles' in self.initial_guess.keys()

        for key in self.initial_guess.keys():

            if key == 'heights':
                if height_is_list:
                    simp[key] = fit_params[:,ntrapezoid+1]
                else:
                    simp[key] = fit_params[:,0]

            elif key == 'langles':
                if height_is_list:
                    simp[key] = fit_params[:,ntrapezoid+1:2*ntrapezoid+1]
                else:
                    simp[key] = fit_params[:,1:ntrapezoid+1]
            
            elif key == 'rangles':
                if height_is_list:
                    simp[key] = fit_params[:,2*ntrapezoid+1:3*ntrapezoid+1]
                else:
                    simp[key] = fit_params[:,ntrapezoid+1:2*ntrapezoid+1]

            elif key == 'botcd':
                if height_is_list and rangles_exists:
                    simp[key] = fit_params[:,3*ntrapezoid+1]
                elif height_is_list and not rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+1]
                elif not height_is_list and rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+1]
                else:
                    simp[key] = fit_params[:,ntrapezoid+1]
                
            
            elif key == 'dwx':
                if height_is_list and rangles_exists:
                    simp[key] = fit_params[:,3*ntrapezoid+2]
                elif height_is_list and not rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+2]
                elif not height_is_list and rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+2]
                else:
                    simp[key] = fit_params[:,ntrapezoid+2]

            elif key == 'dwz':
                if height_is_list and rangles_exists:
                    simp[key] = fit_params[:,3*ntrapezoid+3]
                elif height_is_list and not rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+3]
                elif not height_is_list and rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+3]
                else:
                    simp[key] = fit_params[:,ntrapezoid+3]

            elif key == 'i0':
                if height_is_list and rangles_exists:
                    simp[key] = fit_params[:,3*ntrapezoid+4]
                elif height_is_list and not rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+4]
                elif not height_is_list and rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+4]
                else:
                    simp[key] = fit_params[:,ntrapezoid+4]

            elif key == 'bkg_cste':
                if height_is_list and rangles_exists:
                    simp[key] = fit_params[:,3*ntrapezoid+5]
                elif height_is_list and not rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+5]
                elif not height_is_list and rangles_exists:
                    simp[key] = fit_params[:,2*ntrapezoid+5]
                else:
                    simp[key] = fit_params[:,ntrapezoid+5]

            

        
        return simp



##################################TESTING############################################
pitch = 100 #nm distance between two trapezoidal bars
qzs = np.linspace(-0.1, 0.1, 120)
qxs = 2 * np.pi / pitch * np.ones_like(qzs)

# Define initial parameters and multiples

#Example of data entered by the user
dwx = 0.1
dwz = 0.1
i0 = 10
bkg = 0.1
height = [23.48, 23.48]
bot_cd = [54.6]
swa = [85, 82]

langle = np.deg2rad(np.asarray(swa))
rangle = np.deg2rad(np.asarray(swa))

params1 = {'heights': height,
           'langles': langle, 
           'rangles': rangle,
           'y1': 0, 
           'bot_cd': bot_cd,  
           'dwx': dwx, 
           'dwz': dwz, 
           'i0': i0, 
           'bkg_cste': bkg}

#Example of data arriving from From fitter class
dwx2 = 0.1
dwz2 = 0.1
i02 = 10
bkg2 = 0.1
height2 = [23.48, 23.45]
bot_cd2 = [54.6, 54.2]
swa2 = [[85, 82, 83],[87, 85, 86]]

langle2 = np.deg2rad(np.asarray(swa2))
rangle2 = np.deg2rad(np.asarray(swa2))

params2 = {'heights': height2,
              'langles': langle2, 
              'rangles': rangle2,
              'y1': 0, 
              'bot_cd': bot_cd2,  
              'dwx': dwx2, 
              'dwz': dwz2, 
              'i0': i02, 
              'bkg_cste': bkg2}

Trapezoid1 = TrapezoidGeometry(np)


Trapezoid2 = TrapezoidGeometry(np, from_fitter=True)


form_factor1 = Trapezoid1.correct_form_factor_intensity(qys=qxs, qzs=qzs, params=params1)
form_factor2 = Trapezoid2.correct_form_factor_intensity(qys=qxs, qzs=qzs, params=params2)