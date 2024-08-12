import cdsaxs.residual as Residual
from cdsaxs.simulations.stacked_trapezoid import StackedTrapezoidSimulation
import numpy as np
import pytest


def test_residual(params, initial_params, qzs_qys ):
    qzs, qys = qzs_qys

    Simulation = StackedTrapezoidSimulation(qzs=qzs, qys=qys)
    intensity = Simulation.simulate_diffraction(params=params)
 
    Simulation1 = StackedTrapezoidSimulation(qzs=qzs, qys=qys, initial_guess=initial_params)
    Simulation1.set_from_fitter(True)

    residual = Residual.Residual(data=intensity, fit_mode='cmaes', Simulation=Simulation1)

    calculated_res = residual([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    expected_res = [0.0]

    assert calculated_res == expected_res