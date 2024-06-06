from typing import Protocol


class Geometry(Protocol):
    def convert_to_dataframe(self, fitparams):
        ...


class Simulation(Protocol):
    @property
    def geometry(self) -> Geometry:
        ...

    def set_from_fitter(self, from_fitter):
        ...

    def simulate_diffraction(
        self, fitparams=None, fit_mode='cmaes', best_fit=None
    ):
        ...
