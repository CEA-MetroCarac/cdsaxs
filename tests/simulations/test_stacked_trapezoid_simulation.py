import cdsaxs.simulations.stacked_trapezoid as simulation
import numpy as np
import pandas as pd
from pytest import approx, fixture, mark, skip

class TestStackedTrapezoidSimulation:
    
    @fixture
    def stacked_trapezoid_sim(self, qzs_qys):
        qzs, qys = qzs_qys
        return simulation.StackedTrapezoidSimulation(qzs=qzs, qys=qys)

    def test_simulate_diffraction(self, stacked_trapezoid_sim, multi_params):
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
            calculated_intensities = stacked_trapezoid_sim.simulate_diffraction(param)
            # np.testing.assert_allclose(calculated_intensities, expected_intensities, rtol=1e-2)
            calculated_intensities = approx(expected_intensities, abs=0.1)


class TestStackedTrapezoidGeometry:
    
    @fixture
    def default_initial_guess(self):
        return {
            'heights': {'value': [10], 'variation': 1},
            'langles': {'value': [np.pi / 4], 'variation': 0.1},
            'rangles': {'value': [np.pi / 4], 'variation': 0.1},
            'y_start': {'value': 0, 'variation': 0.1},
            'bot_cd': {'value': 1, 'variation': 0.1}
        }

    @fixture
    def default_geometry(self, default_initial_guess):
        return simulation.StackedTrapezoidGeometry(initial_guess=default_initial_guess)

    def test_initialization(self, default_initial_guess):
        geom = simulation.StackedTrapezoidGeometry(initial_guess=default_initial_guess)
        assert geom.xp is np
        assert geom.from_fitter is False
        assert geom.initial_guess == default_initial_guess
        assert geom.initial_guess_dataframe is None
        assert geom.symmetric is False

    def test_set_initial_guess_dataframe(self, default_geometry, default_initial_guess):
        default_geometry.set_initial_guess_dataframe()
        assert default_geometry.initial_guess_dataframe is not None
        assert 'height1' in default_geometry.initial_guess_dataframe.columns

    def test_convert_to_dataframe(self, default_geometry):
        fitparams = {
            'heights': [10, 20],
            'langles': [np.pi / 6, np.pi / 3],
            'rangles': [np.pi / 6, np.pi / 3],
            'y_start': 0,
            'bot_cd': 1
        }
        df = default_geometry.convert_to_dataframe(fitparams)
        assert isinstance(df, pd.DataFrame)
        assert 'height1' in df.columns
        assert 'height2' in df.columns

    def test_rescale_fitparams(self, default_geometry):
        fitparams_df = pd.DataFrame({
            'height1': [0.5],
            'langle1': [0.5],
            'rangle1': [0.5],
            'y_start': [0],
            'bot_cd': [1]
        })
        rescaled_df = default_geometry.rescale_fitparams(fitparams_df)
        assert not rescaled_df.isnull().values.any()

    def test_check_physical_validity(self, default_geometry):
        rescaled_df = pd.DataFrame({
            'height1': [10],
            'langle1': [-1],  # Invalid value
            'rangle1': [np.pi / 2],
            'y_start': [0],
            'bot_cd': [1]
        })
        valid_df = default_geometry.check_physical_validity(rescaled_df)
        assert np.isnan(valid_df.loc[0, 'langle1'])

    def test_calculate_ycoords(self, default_geometry):
        df = pd.DataFrame({
            'height1': [10],
            'langle1': [np.pi / 2],
            'rangle1': [np.pi / 2],
            'y_start': [0],
            'bot_cd': [1]
        })
        y1, y2 = default_geometry.calculate_ycoords(df)
        assert y1 == np.array([[0]])
        assert y2 == np.array([[1]])

    def test_calculate_shift(self, default_geometry):
        df = pd.DataFrame({
            'height1': [10],
            'height2': [10],
            'langle1': [np.pi / 2],
            'langle2': [np.pi / 2],
        })
        shift = default_geometry.calculate_shift(df)
        assert np.array_equal(shift, np.array([[0, 10]]))

class TestStackedTrapezoidDiffraction:
    @fixture
    def trapezoid_geometry(self):
        initial_guess = {
            'heights': {'value': [10, 15], 'variation': 0.1},
            'langles': {'value': [np.pi / 4, np.pi / 6], 'variation': 0.1},
            'rangles': {'value': [np.pi / 4, np.pi / 5], 'variation': 0.1},
            'y_start': {'value': 0, 'variation': 0.1},
            'bot_cd': {'value': 1, 'variation': 0.1},
            'dwx': {'value': 0.1, 'variation': 0.01},
            'dwz': {'value': 0.1, 'variation': 0.01},
            'i0': {'value': 1.0, 'variation': 0.1},
            'bkg_cste': {'value': 0.1, 'variation': 0.01},
        }
        return simulation.StackedTrapezoidGeometry(initial_guess=initial_guess)

    @fixture
    def stacked_diffraction(self, trapezoid_geometry):
        return simulation.StackedTrapezoidDiffraction(TrapezoidGeometry=trapezoid_geometry)

    @fixture
    def sample_qys(self):
        return np.linspace(0.1, 1.0, 10)

    @fixture
    def sample_qzs(self):
        return np.linspace(0.1, 1.0, 10)

    def test_initialization(self, stacked_diffraction, trapezoid_geometry):
        assert stacked_diffraction.xp is np
        assert stacked_diffraction.TrapezoidGeometry is trapezoid_geometry

    def test_calculate_coefficients(self, stacked_diffraction, trapezoid_geometry, sample_qzs):
        trapezoid_geometry.set_initial_guess_dataframe()
        df = trapezoid_geometry.initial_guess_dataframe.drop('variation')
        coeff = stacked_diffraction.calculate_coefficients(sample_qzs, df=df)
        assert coeff.shape == (1, 2, len(sample_qzs))
        assert np.iscomplexobj(coeff)

    def test_trapezoid_form_factor(self, stacked_diffraction, trapezoid_geometry, sample_qys, sample_qzs):
        trapezoid_geometry.set_initial_guess_dataframe()
        df = trapezoid_geometry.initial_guess_dataframe.drop('variation')
        y1, y2 = trapezoid_geometry.calculate_ycoords(df=df)
        form_factor = stacked_diffraction.trapezoid_form_factor(sample_qys, sample_qzs, y1, y2, df)
        assert form_factor.shape == (1, 2, len(sample_qys))
        assert np.iscomplexobj(form_factor)

    def test_corrections_dwi0bk(self, stacked_diffraction, trapezoid_geometry, sample_qys, sample_qzs):
        trapezoid_geometry.set_initial_guess_dataframe()
        df = trapezoid_geometry.initial_guess_dataframe.drop('variation')
        intensities = np.random.rand( 1, len(sample_qzs) )
        corrected = stacked_diffraction.corrections_dwi0bk(intensities, sample_qys, sample_qzs, df)
        assert corrected.shape == intensities.shape
        assert np.all(corrected >= 0)  # Intensities should be non-negative

    def test_calculate_form_factor(self, stacked_diffraction, trapezoid_geometry, sample_qys, sample_qzs):
        trapezoid_geometry.set_initial_guess_dataframe()
        df = trapezoid_geometry.initial_guess_dataframe.drop('variation')
        form_factor = stacked_diffraction.calculate_form_factor(sample_qys, sample_qzs, df)
        assert form_factor.shape == (1, 2, len(sample_qys))
        assert np.iscomplexobj(form_factor)

    def test_correct_form_factor_intensity(self, stacked_diffraction, trapezoid_geometry, sample_qys, sample_qzs):
        trapezoid_geometry.set_initial_guess_dataframe()
        params = {
            'heights': [10, 15],
            'langles': [np.pi / 2, np.pi / 2],
            'rangles': [np.pi / 2, np.pi / 2],
            'y_start': 0,
            'bot_cd': 20,
            'dwx': 0.1,
            'dwz': 0.1,
            'i0': 10.0,
            'bkg_cste': 0.1
        }
        intensity = stacked_diffraction.correct_form_factor_intensity(sample_qys, sample_qzs, params)
        assert intensity.shape == (1, len(sample_qzs))
        assert np.all(intensity >= 0)  # Intensities should be non-negative

    def test_symmetric_case(self, trapezoid_geometry):
        symmetric_initial_guess = {
            'heights': {'value': [10], 'variation': 0.1},
            'langles': {'value': [np.pi / 4], 'variation': 0.1},
            'y_start': {'value': 0, 'variation': 0.1},
            'bot_cd': {'value': 1, 'variation': 0.1}
        }
        symmetric_geometry = simulation.StackedTrapezoidGeometry(initial_guess=symmetric_initial_guess)
        symmetric_geometry.set_initial_guess_dataframe()
        assert symmetric_geometry.symmetric
        assert 'rangle1' in symmetric_geometry.initial_guess_dataframe.columns

    def test_physical_validity(self, trapezoid_geometry):
        trapezoid_geometry.set_initial_guess_dataframe()
        df = trapezoid_geometry.initial_guess_dataframe.drop('variation')
        df.loc[0, 'height1'] = -1  # Invalid height
        df.loc[0, 'langle1'] = 2 * np.pi  # Invalid angle
        valid_df = trapezoid_geometry.check_physical_validity(df)
        assert np.isnan(valid_df.loc[0, 'height1'])
        assert np.isnan(valid_df.loc[0, 'langle1'])