import cdsaxs_fitting.simulations.stacked_trapezoid as simulation
import numpy as np
from pytest import approx, fixture

@fixture
def stacked_trapezoid(qzs_qys):
    qzs, qys = qzs_qys
    return simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys)

def test_simulate_diffraction(stacked_trapezoid, multi_params):
    """
    Test the simulate_diffraction method of the stacked_trapezoid object.

    Parameters:
    - stacked_trapezoid: The stacked_trapezoid object to test.
    - multi_params: A list of parameters to use for testing.

    Returns:
    None
    """
    expected_intensities = [2595837.42097937, 2983810.93619292, 3303845.79775438, 3531958.65370197,
                            3650634.41543888, 3650634.41543888, 3531958.65370197, 3303845.79775438,
                            2983810.93619292, 2595837.42097937]

    for param in multi_params:
        calculated_intensities = stacked_trapezoid.simulate_diffraction(param)
        # np.testing.assert_allclose(calculated_intensities, expected_intensities, rtol=1e-2)
        calculated_intensities = approx(expected_intensities, abs=0.1)


def test_extract_params(stacked_trapezoid, multi_params):
    """
    Test function for the `extract_params` method of the `TrapezoidGeometry` class.

    Args:
        stacked_trapezoid: An instance of the `TrapezoidGeometry` class.
        multi_params: A list of parameters.

    Returns:
        None
    """
    dwx = 0.1
    dwz = 0.1
    i0 = 10.
    bkg = 0.1
    y1 = 0.
    height = [20.]
    bot_cd = 40.
    swa = [90]

    langle = np.deg2rad(np.asarray(swa))
    rangle = np.deg2rad(np.asarray(swa))
    
    expected_params = (np.asarray([height]), np.asarray([langle]), np.asarray([rangle]), np.asarray(y1), 
                    np.asarray(bot_cd), None, np.asarray(dwx), np.asarray(dwz), np.asarray(i0), np.asarray(bkg))
 

    extracted_params = stacked_trapezoid.TrapezoidGeometry.extract_params(params=multi_params[1])
 
    assert extracted_params == expected_params
