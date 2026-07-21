import os
import yaml
import numpy as np

from roughness.cdsem.run_nebula import generate_primary_from_tri, run_nebula_pipeline


def run_full_pipeline(config_path):
    """
    Full CD-SEM pipeline:
      1. Generate primary electrons from the .tri mesh (if not already present).
      2. Run NEBULA.
      3. Generate the tiff image from detected electrons.

    Parameters
    ----------
    config_path : str
        Path to a YAML file with two top-level sections:
        'primary_electrons' and 'run_nebula' (see config.yaml).

    Returns
    -------
    tif_path : str
        Path to the generated tiff.
    """
    cfg = yaml.safe_load(open(config_path))
    pe_cfg = cfg["primary_electrons"]
    nb_cfg = cfg["run_nebula"]

    # ---- step 1: primary electrons (skip if already generated) ----
    primary_electrons_path = pe_cfg["file_name"]

    if not os.path.isfile(primary_electrons_path):
        os.makedirs(os.path.dirname(primary_electrons_path), exist_ok=True)

        generate_primary_from_tri(
            fname_out=pe_cfg["fname_out"],
            pixel_size=pe_cfg["pixel_size"],
            n_lines=pe_cfg["n_lines"],
            length_of_lines=pe_cfg["length_of_lines"],
            energy=pe_cfg["energy"],
            epx=pe_cfg["epx"],
            sigma=pe_cfg["sigma"],
            poisson=pe_cfg["poisson"],
            file_name=primary_electrons_path,
            interface_mat=tuple(pe_cfg["interface_mat"]),
        )
        print(f"Generated primary electrons -> {primary_electrons_path}")
    else:
        print(f"Primary electrons already exist, skipping generation: {primary_electrons_path}")

    # ---- step 2 + 3: run NEBULA, generate tiff ----
    tif_path = run_nebula_pipeline(
        mesh_path=nb_cfg["mesh_path"],
        primary_electrons_path=nb_cfg["primary_electrons_path"],
        mat_path=nb_cfg["mat_path"],
        detected_path=nb_cfg["detected_path"],
        save_tiff_path=nb_cfg["tif_path"],
        nebula_path=nb_cfg["nebula_path"],
    )

    return tif_path


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/homelocal/nd276333/Workspace/Alternance/cdsaxs/src/roughness/config.yaml"
    run_full_pipeline(config_path)