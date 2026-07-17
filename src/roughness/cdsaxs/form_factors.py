
import numpy as np
from contextlib import contextmanager

try:
    import cupy as cp
except ImportError:
    cp = None

@contextmanager
def _errstate(xp):
    if xp.__name__ == 'cupy':
        yield
    else:
        with xp.errstate(divide='ignore', invalid='ignore'):
            yield


# ------------------------------------------------------------------
# array-module dispatch (numpy vs cupy), private
# ------------------------------------------------------------------
def _get_xp(qx):
    return cp if (cp is not None and isinstance(qx, cp.ndarray)) else np


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

