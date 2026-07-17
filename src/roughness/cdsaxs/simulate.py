# roughness/cdsaxs/simulate.py
"""
Functional CD-SAXS line diffraction simulation.

Single entry point: simulate_line_intensity().
Accepts pre-generated roughness profiles, or generates them internally
if none are supplied (using roughness.profile.generate.generate_profiles,
the same function used by the CD-SEM track, so both tracks can share
identical ground truth when called with the same seed).
"""

import numpy as np
import dask.array as da

from roughness.profile.generate import generate_profiles, psd_lorentzian
from roughness.cdsaxs.form_factors import _get_xp, cube_form_factor,  trapezoid_form_factor


# ------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------
def simulate_line_intensity(qx, qy, qz, width, height, dy, num_cubes,
                             profiles=None,
                             subdivide_factor=1,
                             beta_r=None, beta_l=None,
                             # used only if profiles is None:
                             sigma=None, xi=None, alpha=None,
                             N_realizations=None, seed=None,
                             sigma_noise=0., mu=0., psd_model=psd_lorentzian,
                             dask_chunks=(200, 2000)):
    """
    Compute ensemble-averaged CD-SAXS diffraction intensity for a rough line.

    Parameters
    ----------
    qx, qy, qz : array
        q-space grids (broadcastable, same shape).
    width, height : float
        Full line width / height (nm). width = bottom CD if trapezoid.
    dy : float
        Cuboid spacing along the line (nm).
    num_cubes : int
        Number of cuboids per line (must match profiles.shape[1] if profiles given).
    profiles : array, shape (N_realizations, num_cubes), optional
        Pre-generated roughness profiles (e.g. from roughness.profile.generate.generate_profiles,
        or loaded from disk). If None, generated internally using sigma/xi/alpha/seed below —
        pass the same seed used elsewhere to share ground truth with the CD-SEM track.
    subdivide_factor : int
        Optional further subdivision of each cuboid along y.
    beta_r, beta_l : float, optional
        Sidewall angles in degrees. Both required for trapezoid; omit both for rectangular.
    sigma, xi, alpha, N_realizations, seed : required only if profiles is None.
    sigma_noise, mu, psd_model : passed through to generate_profiles if profiles is None.
    dask_chunks : (chunk_repeat, chunk_cubes)
        Dask chunk sizes; tune for memory/core count.

    Returns
    -------
    array
        Ensemble-averaged diffraction intensity, shape matching qx (squeezed).
    """
    xp = _get_xp(qx)

    if profiles is None:
        if None in (sigma, xi, alpha, N_realizations, seed):
            raise ValueError(
                "profiles not given: sigma, xi, alpha, N_realizations, seed are required "
                "to generate them internally."
            )
        profiles = generate_profiles(
            N=num_cubes, dx=dy, sigma=sigma, xi=xi, alpha=alpha,
            N_realizations=N_realizations, seed=seed,
            sigma_noise=sigma_noise, mu=mu, psd_model=psd_model,
        )

    profiles = np.asarray(profiles)
    if profiles.shape[1] != num_cubes:
        raise ValueError(
            f"profiles.shape[1] ({profiles.shape[1]}) must equal num_cubes ({num_cubes})"
        )
    num_repeat = profiles.shape[0]

    half_w = width / 2.0
    half_z = height / 2.0
    dy2 = dy / subdivide_factor
    half_dy2 = dy2 / 2.0

    use_trapezoid = beta_r is not None and beta_l is not None

    if use_trapezoid:
        F0 = trapezoid_form_factor(qx, qy, qz, half_w, beta_r, beta_l, half_dy2, half_z)
    else:
        F0 = cube_form_factor(qx, qy, qz, half_w, half_dy2, half_z)

    qshape = qx.shape if hasattr(qx, "shape") else (qx.size,)
    q_ndim = len(qshape)

    F0_np = np.asarray(F0).reshape((1, 1, 1) + qshape)
    profiles_np = profiles.reshape((num_repeat, num_cubes, 1) + (1,) * q_ndim)

    i_idx = np.arange(num_cubes)[:, None]
    k_idx = np.arange(subdivide_factor)[None, :]
    y_pos = i_idx * dy + (k_idx + 0.5) * dy2  # (num_cubes, subdivide_factor)
    y_pos_np = y_pos.reshape((1, num_cubes, subdivide_factor) + (1,) * q_ndim)

    # broadcast profiles across subdivide_factor
    profiles_np = np.broadcast_to(
        profiles_np, (num_repeat, num_cubes, subdivide_factor) + (1,) * q_ndim
    )

    qx_np = np.asarray(qx).reshape((1, 1, 1) + qshape)
    qy_np = np.asarray(qy).reshape((1, 1, 1) + qshape)

    chunk_repeat = min(max(1, num_repeat // 8), dask_chunks[0])
    chunk_cubes = min(max(1, num_cubes // 8), dask_chunks[1])
    chunk_tuple = (chunk_repeat, chunk_cubes, subdivide_factor) + (1,) * q_ndim

    profiles_da = da.from_array(np.ascontiguousarray(profiles_np), chunks=chunk_tuple)
    y_pos_da = da.from_array(y_pos_np, chunks=(1, chunk_cubes, subdivide_factor) + (1,) * q_ndim)
    F0_da = da.from_array(F0_np, chunks=(1, 1, 1) + qshape)
    qx_da = da.from_array(qx_np, chunks=(1, 1, 1) + qshape)
    qy_da = da.from_array(qy_np, chunks=(1, 1, 1) + qshape)

    phase_x = da.exp(-1j * profiles_da * qx_da)
    phase_y = da.exp(-1j * y_pos_da * qy_da)
    amp_terms = F0_da * phase_x * phase_y

    amp = da.sum(amp_terms, axis=(1, 2))
    norm = float(num_cubes * subdivide_factor)
    intensity = da.absolute(amp / norm) ** 2

    result = intensity.mean(axis=0).squeeze().compute(scheduler='processes')
    return result