from cdsaxs.simulations.strong_castle import StrongCastleSimulation, StrongCastleGeometry
from pytest import approx, fixture
import numpy as np

class TestStrongCastle:

    @fixture
    def qzs_qys(self):
        pitch = 100  # nm distance between two trapezoidal bars
        qzs = np.linspace(-0.1, 0.1, 10)
        qys = 2 * np.pi / pitch * np.ones_like(qzs)
        return qzs, qys
    
    @fixture
    def strong_castle_sim(self, qzs_qys):
        qzs, qys = qzs_qys
        return StrongCastleSimulation(qzs=qzs, qys=-qys)
    
    @fixture
    def strong_castle_geom(self):
        return StrongCastleGeometry()
    
    @fixture
    def overlay_params(self):
        # Initial parameters
        dwx = 0.1
        dwz = 0.1
        i0 = 10
        bkg = 0.1
        y1 = 0.
        height = 10.
        bot_cd = 40.
        top_cd = 20.
        swa = [90., 90.0, 90.0, 90.0, 90.0, 90.0]
        overlay = 0
        # Fixed parameters not to be fitted
        n1 = 3
        n2 = 3

        langle = np.deg2rad(np.asarray(swa))
        rangle = np.deg2rad(np.asarray(swa))

        overlay_params = {
            'heights': height,
            'langles': langle,
            'rangles': rangle,
            'y_start': y1,
            'bot_cd': bot_cd,
            'dwx': dwx,
            'dwz': dwz,
            'i0': i0,
            'bkg_cste': bkg,
            'overlay': overlay,
            'top_cd': top_cd,
            'n1': n1,
            'n2': n2,
        }
        
        return overlay_params
    
    def test_simulate_diffraction(self, strong_castle_sim, overlay_params):
        expected_intensities = [
            577316.24962421,  2705659.68913569,  8222621.93553933, 15543913.84497248,
            20837878.81639422, 20837878.81639424, 15543913.84497249,  8222621.93553934,
            2705659.68913569,   577316.24962421
        ]
        
        calculated_intensities = strong_castle_sim.simulate_diffraction(params=overlay_params)
        assert calculated_intensities == approx(expected_intensities, abs=0.1)

    def test_calculate_ycoords(self, strong_castle_geom, overlay_params):
        df = strong_castle_geom.convert_to_dataframe(overlay_params)
        strong_castle_geom.set_n1_n2(overlay_params['n1'], overlay_params['n2'])
        y1, y2 = strong_castle_geom.calculate_ycoords(df)

        print(y1)
        print(y2)

        assert y1[0] == approx([0., 0., 0., 10., 10., 10.], abs=0.1)
        assert y2[0] == approx([40., 40., 40., 30., 30., 30.], abs=0.1)

    def test_remove_fixed_params(self, strong_castle_geom, overlay_params):
        
        expected = overlay_params.copy()
        del expected['n1']
        del expected['n2']

        strong_castle_geom.set_n1_n2(overlay_params['n1'], overlay_params['n2'])
        calculated = strong_castle_geom.remove_fixed_params(overlay_params)

        assert calculated == expected