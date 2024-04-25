from typing import Protocol


class Geometry(Protocol):
    def extract_params(
        self, fitparams=None, params=None, for_best_fit=False, for_saving=False, best_fit=None, fit_mode='cmaes',
    ):
        ...


class Simulation(Protocol):
    @property
    def geometry(self) -> Geometry:
        ...

    def set_from_fitter(self, from_fitter):
        ...

    def simulate_diffraction(
        self, params=None, fitparams=None, fit_mode='cmaes', best_fit=None
    ):
        ...
