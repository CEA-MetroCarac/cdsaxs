"""This module is a child class of the StackedTrapezoidSimulation class. Its purpose is to implement the 
    Strong Castle model which calculates the overlay.

    Classes:
        StrongCastleSimulation: This class is a child class of the StackedTrapezoidSimulation class.
        StrongCastleGeometry: This class is a child class of the StackedTrapezoidGeometry class.
        StrongCastleDiffraction: This class is a child class of the StackedTrapezoidDiffraction class.
"""

import re
import numpy as np


from .stacked_trapezoid import StackedTrapezoidSimulation, StackedTrapezoidGeometry, StackedTrapezoidDiffraction




class StrongCastleSimulation(StackedTrapezoidSimulation):
    """ This class is a child class of the StackedTrapezoidSimulation class.
    """

    def __init__(self, qys, qzs, from_fitter=False, use_gpu=False, initial_guess=None):
        """ This init method modifies the parent class init method by adding the StrongCastleGeometry and StrongCastleDiffraction classes
            instead of the StackedTrapezoidGeometry and StackedTrapezoidDiffraction classes.

        Args:
            qys (np.ndarray): points in the y direction where form factor is calculated
            qzs (_type_): points in the z direction where form factor is calculated
            from_fitter (bool, optional): variable to identify if the simulation is coming from fitter or not. Defaults to False.
            use_gpu (bool, optional): wether to use gpu or not. Defaults to False.
            initial_guess (dictionary, optional): a dictionary containing all the parameters. Defaults to None.
        """
        
        super().__init__(qys, qzs, from_fitter=from_fitter, use_gpu=use_gpu, initial_guess=initial_guess)

        self.TrapezoidGeometry = StrongCastleGeometry(from_fitter=self.from_fitter, initial_guess=initial_guess)
        self.TrapezoidDiffraction = StrongCastleDiffraction(self.TrapezoidGeometry)





class StrongCastleGeometry(StackedTrapezoidGeometry):
    """ This class is a child class of the StackedTrapezoidGeometry class. Several methods are modified and added to implement the Strong Castle model(notably the most important
        calculate_ycoords).
    """

    def __init__(self, xp=np, from_fitter=False, initial_guess=None):
        
        """
        This init method is similar to the parent class, except it removes the fixed parameters `n1` and `n2`
        (supposed to represent the first layer and second layer of trapezoids) from the `initial_guess` dictionary.

        Args:
            xp (numpy or cupy): The numpy or cupy module.
            from_fitter (bool): Indicates whether the object is created from the fitter or not.
            initial_guess (dict): A dictionary containing the initial guess for the parameters.
        """

        self.from_fitter = from_fitter
        self.n1 = 0
        self.n2 = 0
        self.initial_guess = initial_guess

        if self.initial_guess is not None:
            self.set_n1_n2(self.initial_guess["n1"], self.initial_guess["n2"])

            initial_guess_for_fit = self.remove_fixed_params(self.initial_guess)

            super().__init__(xp=xp, from_fitter=from_fitter, initial_guess=initial_guess_for_fit)
        else:
            super().__init__(xp=xp, from_fitter=from_fitter, initial_guess=None)
    
    def remove_fixed_params(self, initial_guess):
        """Remove the fixed parameters which are the number of first and second layer of trapezoids n1 and n2 from the initial_guess dictionary so that they are not fitted.

        Args:
            initial_guess (dictionary): A dictionary containing the initial guess for the parameters provided by the user.

        Returns:
            initial_guess (dictionary): A dictionary containing the initial guess for the parameters without the fixed parameters.
        """
        #remove the parameters which are fixed and not be fitted from the initial_guess dictionary
        fixed_params = {"n1", "n2"}
        initial_guess_for_fit = {key: value for key, value in initial_guess.items() if key not in fixed_params}

        #check if the number of trapezoids is equal to the number of angles in the initial_guess dictionary²
        self.check_initial_guess(initial_guess_for_fit)

        return initial_guess_for_fit
    
    def set_n1_n2(self, n1, n2):
        """Set self.n1 and self.n2 to the provided values.

        Args:
            n1 (integer): number of trapezoids in first layer of trapezoids
            n2 (integer): number of trapezoids in second layer of trapezoids
        """
        self.n1 = n1
        self.n2 = n2

    def check_initial_guess(self, params):
        """
            Check if the number of trapezoids  n1+n2 is equal to the number of angles in the initial_guess dictionary

            Args:
            initial_guess (dict): dictionary containing the initial guess for the parameters

        """

        if self.initial_guess is None: 
            langles = params["langles"]
            rangles = params["rangles"]
        else:
            langles = self.initial_guess["langles"]["value"]
            rangles = self.initial_guess["rangles"]["value"]

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
        and the y coordinates of the second trapezoid are shifted by the overlay value.

        Args:
            df (pandas.DataFrame): dataframe containing the trapezoid parameters

        Returns:
            y1 (numpy.ndarray): y1 coordinates of the trapezoids
            y2 (numpy.ndarray): y2 coordinates of the trapezoids
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

        #if height is constant then add height1 to the second trapezoid columns
        if self.from_fitter and df.filter(like="height").shape[1] < 2:
            second_trapezoid_columns.append("height1")

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

        #if from fit convert in to array to avoid broadcasting error
        if self.from_fitter:
            translation_with_overlay = translation_with_overlay[..., np.newaxis] * self.xp.ones_like(second_trapezoid_y1)

        second_trapezoid_y1 = second_trapezoid_y1 + translation_with_overlay
        second_trapezoid_y2 = second_trapezoid_y2 + translation_with_overlay

        #combine
        y1 = np.hstack( (first_trapezoid_y1, second_trapezoid_y1) )
        y2 = np.hstack( (first_trapezoid_y2, second_trapezoid_y2) )

        return y1 , y2


class StrongCastleDiffraction(StackedTrapezoidDiffraction):
    """ This class is a child class of the StackedTrapezoidDiffraction class. It modifies the correct_form_factor_intensity method to account for the two 
        fixed params n1 and n2 which are the number of trapezoids in the first and second layer of trapezoids respectively.
    """

    def correct_form_factor_intensity(self, qys, qzs, fitparams):
        """remove the fixed parameters n1 and n2 from the fitparams dictionary and call the parent class correct_form_factor_intensity method.

        Args:
            qys (np.ndarray): points in the y direction where form factor is calculated
            qzs (np.ndarray): points in the z direction where form factor is calculated
            fitparams (np.ndarray): a dictionary containing the parameters

        Returns:
            corrected_intensity (np.ndarray): intensity of the strong castle model
        """
        #remove non fitted parameters before calling the parent class method
        if self.TrapezoidGeometry.initial_guess is None:
            self.TrapezoidGeometry.set_n1_n2(fitparams["n1"], fitparams["n2"])
            fitparams_without_fixed = self.TrapezoidGeometry.remove_fixed_params(fitparams)
            return super().correct_form_factor_intensity(qys, qzs, fitparams_without_fixed)
        
        return super().correct_form_factor_intensity(qys, qzs, fitparams)