# usage example — identical call from cdsaxs/simulate.py and cdsem/prepare.py

import yaml
import numpy as np
from roughness.profile.generate import generate_profiles, validate_profile_psd, psd_lorentzian

with open("./config.yaml") as f:
    cfg = yaml.safe_load(f)["profile"]

profiles = generate_profiles(
    N=cfg["N"], dx=cfg["dx"],
    sigma=cfg["sigma"], xi=cfg["xi"], alpha=cfg["alpha"],
    N_realizations=cfg["N_realizations"],
    seed=cfg["seed"],
    sigma_noise=cfg["sigma_noise"], mu=cfg["mu"],
    psd_model=None,
    save_file=cfg["save_file"]
)
# profiles.shape == (100, 8000)

w_n = profiles[0]        # single realization, e.g. for one cdsaxs line stack
                          # or for one cdsem extrusion — same array, same seed

freq, psd_meas, psd_target = validate_profile_psd(
    profiles, dx=cfg["dx"],
    sigma=cfg["sigma"], xi=cfg["xi"], alpha=cfg["alpha"]
)

mask = freq > 0

psd_diff = np.abs(psd_meas[mask] - psd_target[mask])

print(f"Max PSD difference: {psd_diff.max():.3e}")

if psd_diff.max() > 1e-10:
    raise AssertionError("PSD arrays differ")