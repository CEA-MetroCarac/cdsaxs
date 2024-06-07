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
        raise NotImplementedError("Subclasses must implement this method")


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
        raise NotImplementedError("Subclasses must implement this method")

    def simulate_diffraction(
        self, fitparams=None, fit_mode='cmaes', best_fit=None
    ):
        raise NotImplementedError("Subclasses must implement this method")
