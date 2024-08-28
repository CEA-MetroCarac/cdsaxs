""" 
    This is a protocol that defines the methods that should be implemented to ensure that the Simulation class 
    is compatible with the fitter class.
"""
from typing import Protocol


class Geometry(Protocol):
    """
        This is a protocol that defines the methods that should be implemented by the Geometry class.
    """
    def convert_to_dataframe(self, fitparams):
        """
            This is a obligatory method that should be implemented by the Geometry class.
            It's job is to take the arrays of fitparams and convert them into a pandas DataFrame
            based on the initial guess given by the user.
    
            Parameters:
            fitparams (list): an array returned by the fitter that contains the population to be evaluated.
            
            returns:
            df (pandas.DataFrame): a DataFrame containing the fitparams in a readable format.
        """
        ...
    
    def rescale_fitparams(self, fitparams):
        """
            This is a obligatory method that should be implemented by the Geometry class.
            The values incoming from the fitter are normalized in the range of -sigma to sigma they need to be rescaled.
            so this method's job is to take the arrays of fitparams and rescale them based on the initial guess given by the user.
            the formula used should be something like this:
            rescaled_fitparams = fitparams * multiplier + initial_guess
            multiplier and initial_guess are the values given by the user. if you find that a certain set of multiplier works
            better you can hardcode them as well but it is not recommended as it will make the code less flexible.
    
            Parameters:
            fitparams (list): an array returned by the fitter that contains the population to be evaluated.
            
            returns:
            df (dataframe): an array containing the rescaled fitparams.
        """
        ...


class Simulation(Protocol):

    """
        This is a protocol that defines the methods that should be implemented by the Simulation class.
    """
    @property
    def geometry(self) -> Geometry:
        ...

    def set_from_fitter(self, from_fitter):
        """
            This is a obligatory method that should be implemented by the Simulation class.
            It should tell the Simulation object that the incoming data is from a fitter object and it should also initialize necessary things
            like saving the initial guess given by user to a dataframe.

        """
        ...

    def simulate_diffraction(
        self, fitparams=None, best_fit=None
    ):
        
        """ This is a obligatory method that should be implemented by the Simulation class. It is used by Residual class.

        Raises:
            NotImplementedError: _description_
        """
        ...
