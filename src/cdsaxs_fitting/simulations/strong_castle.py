import re
import numpy as np
import pandas as pd


from .stacked_trapezoid import StackedTrapezoidSimulation, StackedTrapezoidGeometry, StackedTrapezoidDiffraction

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except:
    CUPY_AVAILABLE = False



class StrongCastleSimulation(StackedTrapezoidSimulation):

    def __init__(self, qys, qzs, from_fitter=False, use_gpu=False, initial_guess=None):
        super().__init__(qys, qzs, from_fitter=from_fitter, use_gpu=use_gpu, initial_guess=initial_guess)

        self.TrapezoidGeometry = StrongCastleGeometry(xp=self.xp, from_fitter=self.from_fitter, initial_guess=initial_guess)
        self.TrapezoidDiffraction = StrongCastleDiffraction(self.TrapezoidGeometry, xp=self.xp)

    def simulate_diffraction(self, fitparams=None, fit_mode='cmaes', best_fit=None, params=None):
        if params is not None:
            fitparams = params

        if fit_mode == 'cmaes':
            corrected_intensity = self.TrapezoidDiffraction.correct_form_factor_intensity(qys=self.qys, qzs=self.qzs, fitparams = fitparams)
        elif fit_mode == 'mcmc':
            corrected_intensity = self.TrapezoidDiffraction.correct_form_factor_intensity(qys=self.qys, qzs=self.qzs, fitparams = fitparams)

        if not self.from_fitter:
            return corrected_intensity[0]

        return corrected_intensity



class StrongCastleGeometry(StackedTrapezoidGeometry):

    #there are two types of parameters coming in as dictionary ones that need to be fitted and the ones that are fixed i need to remove the fixed ones from the dictionary
    # and then do the regular fitting for the rest of the parameters
    #fixed params 

    def __init__(self, xp=np, from_fitter=False, initial_guess=None):
        
        """
            init basically same as parent class except that it removes the fixed parameters which are n1 and n2 from the initial_guess dictionary

            Parameters:
            -----------
            xp : numpy or cupy
                numpy or cupy module
            from_fitter : bool
                whether the object is created from the fitter or not
            initial_guess : dict
                dictionary containing the initial guess for the parameters

        """
        self.from_fitter = from_fitter
        self.n1 = 0
        self.n2 = 0
        if self.from_fitter:
            self.set_n1_n2(initial_guess["n1"], initial_guess["n2"])

            initial_guess_for_fit = self.remove_fixed_params(initial_guess)

            super().__init__(xp=xp, from_fitter=from_fitter, initial_guess=initial_guess_for_fit)
        else:
            super().__init__(xp=xp, from_fitter=from_fitter, initial_guess=None)
    
    def remove_fixed_params(self, initial_guess):

        """
            Remove the fixed parameters which are the number of trapezoids n1 and n2 from the initial_guess dictionary
        """
        #remove the parameters which are fixed and not be fitted from the initial_guess dictionary
        fixed_params = {"n1", "n2"}
        initial_guess_for_fit = {key: value for key, value in initial_guess.items() if key not in fixed_params}

        #check if the number of trapezoids is equal to the number of angles in the initial_guess dictionary²
        self.check_initial_guess(initial_guess_for_fit)

        return initial_guess_for_fit
    
    def set_n1_n2(self, n1, n2):
        """
            Set the number of trapezoids n1 and n2
        """
        self.n1 = n1
        self.n2 = n2

    def check_initial_guess(self, initial_guess):
        """
            Check if the number of trapezoids  n1+n2 is equal to the number of angles in the initial_guess dictionary

            Parameters:
            -----------
            initial_guess : dict
                dictionary containing the initial guess for the parameters

        """

        langles = initial_guess["langles"]
        rangles = initial_guess["rangles"]

        #from the initial_guess dictionary get the number of langles
        # Check the number of angles provided
        if langles is not None or rangles is not None:
            # Determine the number of angles in initial_guess
            n = langles.shape[0] if langles is not None else rangles.shape[0]
            
            # Verify that the number of angles matches the expected sum of trapezoids
            expected_n = self.n1 + self.n2
            if n != expected_n:
                raise ValueError(f"Number of angles should be same as the sum of the number of trapezoids. Expected {expected_n} but got {n}")


    def calculate_ycoords(self, df):
        """
            This is the modified version of the calculate_ycoords method in the parent class. The two trapezoids are separated into two groups and the y coordinates are calculated
            and the y coordinates of the second trapezoid are shifted by the overlay value


            Parameters:
            -----------
            df : pandas.DataFrame
                dataframe containing the trapezoid parameters

            Returns:
            --------
            y1 : numpy.ndarray
                y1 coordinates of the trapezoids
            y2 : numpy.ndarray
                y2 coordinates of the trapezoids
        """

        #first trapezoid
        first_trapezoid_columns = [
            col for col in df.columns
            if all(int(num) <= self.n1 for num in re.findall(r'\d+', col)) or not re.findall(r'\d+', col)
        ]

        first_trapezoid_df = df[first_trapezoid_columns]

        first_trapezoid_y1, first_trapezoid_y2  = super().calculate_ycoords(first_trapezoid_df)

        #Second trapezoid
        second_trapezoid_columns = [
            col for col in df.columns
            if all(int(num) > self.n1 for num in re.findall(r'\d+', col))
        ]

        #giving the second trapezoid correct parameters
        second_trapezoid_df = df[second_trapezoid_columns]
        second_trapezoid_df.loc[:, "bot_cd"] = second_trapezoid_df["top_cd"].values

        second_trapezoid_y1, second_trapezoid_y2 = super().calculate_ycoords(second_trapezoid_df)

        #shift the y coordinates of the second trapezoid by the overlay value
        midpoint_trapezoid1 = (first_trapezoid_df['y_start'].values + first_trapezoid_df['bot_cd'].values)/2
        midpoint_trapezoid2 = (second_trapezoid_df['y_start'].values + second_trapezoid_df['bot_cd'].values)/2
        translation = midpoint_trapezoid1 - midpoint_trapezoid2#overlay is defined as the difference between the midpoints of the two trapezoids so aligning the midpoints 
        translation_with_overlay = translation + df["overlay"].values

        #translate
        second_trapezoid_y1 = second_trapezoid_y1 + translation_with_overlay
        second_trapezoid_y2 = second_trapezoid_y2 + translation_with_overlay

        #combine
        y1 = np.hstack( (first_trapezoid_y1, second_trapezoid_y1) )
        y2 = np.hstack( (first_trapezoid_y2, second_trapezoid_y2) )

        print( "y1:", y1 )
        print( "y2:", y2 )

        return y1 , y2


class StrongCastleDiffraction(StackedTrapezoidDiffraction):

    def __init__(self, geometry, xp=np):
        self.geometry = geometry
        super().__init__(geometry, xp=xp)
        

    def correct_form_factor_intensity(self, qys, qzs, fitparams):

        self.geometry.set_n1_n2(fitparams["n1"], fitparams["n2"])
        fitparams_without_fixed = self.geometry.remove_fixed_params(fitparams)


        return super().correct_form_factor_intensity(qys, qzs, fitparams_without_fixed)