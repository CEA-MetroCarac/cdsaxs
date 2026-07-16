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
from contextlib import contextmanager

from roughness.profile.generate import generate_profiles, psd_lorentzian

try:
    import cupy as cp
except ImportError:
    cp = None


# ------------------------------------------------------------------
# array-module dispatch (numpy vs cupy), private
# ------------------------------------------------------------------
def _get_xp(qx):
    return cp if (cp is not None and isinstance(qx, cp.ndarray)) else np


@contextmanager
def _errstate(xp):
    if xp.__name__ == 'cupy':
        yield
    else:
        with xp.errstate(divide='ignore', invalid='ignore'):
            yield


# ------------------------------------------------------------------
# single-element form factors
# ------------------------------------------------------------------
def cube_form_factor(qx, qy, qz, half_width, half_dy, half_height):
    """Rectangular cuboid element form factor."""
    xp = _get_xp(qx)
    with _errstate(xp):
        sinc_qx = xp.sin(qx * half_width) / (qx * half_width)
        sinc_qy = xp.sin(qy * half_dy) / (qy * half_dy)
        sinc_qz = xp.sin(qz * half_height) / (qz * half_height)
        sinc_qx = xp.where(xp.isnan(sinc_qx), 1.0, sinc_qx)
        sinc_qy = xp.where(xp.isnan(sinc_qy), 1.0, sinc_qy)
        sinc_qz = xp.where(xp.isnan(sinc_qz), 1.0, sinc_qz)
        return half_width * half_dy * half_height * sinc_qx * sinc_qy * sinc_qz


def _centered_height_integral(xp, a, h):
    """I(a) = -i*h*exp(-i*a*h/2)*sinc(a*h/(2*pi)), stable at a=0."""
    return h * xp.exp(-1j * a * h / 2.0) * xp.sinc(a * h / (2.0 * xp.pi))


def trapezoid_form_factor(qx, qy, qz, half_width_bottom, beta_r, beta_l, half_dy, half_height):
    """
    Trapezoidal cross-section element form factor, centered at origin.
    beta_r, beta_l: sidewall angles in degrees. 90 = vertical (reduces to cube_form_factor).
    """
    xp = _get_xp(qx)
    height = 2 * half_dy

    if not (0.0 < beta_r < 180.0):
        raise ValueError(f"beta_r must be in (0, 180) degrees, got {beta_r}")
    if not (0.0 < beta_l < 180.0):
        raise ValueError(f"beta_l must be in (0, 180) degrees, got {beta_l}")

    tan_r = xp.tan(xp.deg2rad(beta_r))
    tan_l = xp.tan(xp.deg2rad(beta_l))

    half_w_top_r = half_width_bottom - height / tan_r
    half_w_top_l = half_width_bottom - height / tan_l
    if half_w_top_r <= 0 or half_w_top_l <= 0:
        raise ValueError(
            "Sidewall angles too steep for given height/width: derived top "
            f"half-widths (right={half_w_top_r}, left={half_w_top_l}) are "
            "non-positive; trapezoid pinches shut before reaching the top."
        )

    x0 = half_width_bottom
    a_r = qy - qx / tan_r
    a_l = qy + qx / tan_l

    I_r = _centered_height_integral(xp, a_r, height)
    I_l = _centered_height_integral(xp, a_l, height)

    bracket = xp.exp(-1j * qx * x0) * I_r - xp.exp(1j * qx * x0) * I_l
    qx_safe = xp.where(qx == 0, 1e-12, qx)
    Fxy = xp.exp(1j * qy * half_dy) * (-bracket / (-1j * qx_safe))
    Fz = 2 * half_height * xp.exp(-1j * qz * half_height) * xp.sinc(qz * half_height / xp.pi)

    return Fxy * Fz * (1 / 8)  # matches cube_form_factor normalization at beta=90


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