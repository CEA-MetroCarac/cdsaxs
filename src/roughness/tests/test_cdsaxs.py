import os
import yaml
import numpy as np

from roughness.profile.generate import generate_profiles, psd_lorentzian
from roughness.cdsaxs.simulate import simulate_line_intensity
from roughness.cdsaxs.extract import extract_psd


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_or_generate_profiles(prof_cfg, profiles_path=None):
    """
    Load profiles from disk if profiles_path is given, else generate them
    from prof_cfg.
    """
    if profiles_path:
        if not os.path.isfile(profiles_path):
            raise FileNotFoundError(f"profiles_path not found: {profiles_path}")
        profiles = np.load(profiles_path)
        print(f"loaded profiles: {profiles.shape} from {profiles_path}")
        return profiles

    psd_model_raw = prof_cfg.get("psd_model")
    psd_model = psd_lorentzian if psd_model_raw in (None, "None", "none", "") else psd_model_raw

    profiles = generate_profiles(
        N=prof_cfg["N"],
        dx=prof_cfg["dx"],
        sigma=prof_cfg["sigma"],
        xi=prof_cfg["xi"],
        alpha=prof_cfg["alpha"],
        N_realizations=prof_cfg["N_realizations"],
        seed=prof_cfg["seed"],
        sigma_noise=prof_cfg.get("sigma_noise", 0.0),
        mu=prof_cfg.get("mu", 0.0),
        psd_model=psd_model,
        save_file=prof_cfg.get("save_file"),
    )
    print(f"generated profiles: {profiles.shape}")
    return profiles


def build_q_grid(cdsaxs_cfg):
    """Build QX, QY, QZ meshgrids from config."""
    qy = np.linspace(cdsaxs_cfg["qy_min"], cdsaxs_cfg["qy_max"], cdsaxs_cfg["qy_points"])
    qx = np.linspace(cdsaxs_cfg["qx_min"], cdsaxs_cfg["qx_max"], cdsaxs_cfg["qx_points"])

    QX, QY = np.meshgrid(qx, qy)
    QZ = np.full_like(QX, cdsaxs_cfg["qz_fixed"])
    return QX, QY, QZ


def run_simulation(profiles, prof_cfg, cdsaxs_cfg, QX, QY, QZ):
    """Run simulate_line_intensity (cubes only) and return the intensity map."""
    intensity = simulate_line_intensity(
        qx=QX, qy=QY, qz=QZ,
        width=cdsaxs_cfg["width"], height=cdsaxs_cfg["height"],
        dy=prof_cfg["dx"], num_cubes=prof_cfg["N"],
        profiles=profiles,
    )
    print(f"intensity map shape: {intensity.shape}")
    return intensity


def run_extraction(intensity, prof_cfg, cdsaxs_cfg, QX, QY, QZ,
                    slice_cols=slice(8, 15), q_col=10, num_orders=1):
    """
    Average intensity over slice_cols (a diffuse-scattering band away from
    extinction points), then extract the PSD at column q_col.
    """
    I_slice = np.mean(intensity[:, slice_cols], axis=1)

    S_recov = extract_psd(
        I_slice=I_slice,
        qx=QX[:, q_col], qy=QY[:, q_col], qz=QZ[:, q_col],
        sigma=prof_cfg["sigma"], xi=prof_cfg["xi"], alpha=prof_cfg["alpha"],
        width=cdsaxs_cfg["width"], dy=prof_cfg["dx"],
        num_cubes=prof_cfg["N"], height=cdsaxs_cfg["height"],
        num_orders=num_orders,
    )
    return S_recov


def run_cdsaxs_pipeline(config_path, profiles_path=None):
    """
    Full CD-SAXS pipeline: load config -> get/generate profiles ->
    build q-grid -> simulate -> extract PSD.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (must contain a "profile" section, with a
        nested "cdsaxs" sub-section).
    profiles_path : str, optional
        If given, load profiles from this .npy instead of generating them.

    Returns
    -------
    dict with keys: profiles, intensity, S_recov, QX, QY, QZ
    """
    config = load_config(config_path)
    prof_cfg = config["profile"]
    cdsaxs_cfg = prof_cfg["cdsaxs"]

    profiles = get_or_generate_profiles(prof_cfg, profiles_path=profiles_path)
    QX, QY, QZ = build_q_grid(cdsaxs_cfg)
    intensity = run_simulation(profiles, prof_cfg, cdsaxs_cfg, QX, QY, QZ)
    S_recov = run_extraction(intensity, prof_cfg, cdsaxs_cfg, QX, QY, QZ)

    return {
        "profiles": profiles,
        "intensity": intensity,
        "S_recov": S_recov,
        "QX": QX, "QY": QY, "QZ": QZ,
    }


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        "cdsaxs/src/roughness/config.yaml"
    profiles_path = sys.argv[2] if len(sys.argv) > 2 else None

    results = run_cdsaxs_pipeline(config_path, profiles_path=profiles_path)
    print("S_recov shape:", results["S_recov"].shape)