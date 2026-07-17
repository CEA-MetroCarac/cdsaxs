# roughness/cdsaxs/extract.py

import numpy as np
import math
from roughness.cdsaxs.form_factors import cube_form_factor, trapezoid_form_factor
from roughness.profile.generate import psd_lorentzian, psd_azarnouche  # confirmed same as PSD_AN


# ------------------------------------------------------------------
# higher-order diffuse correction (private — only used by extract_psd)
# ------------------------------------------------------------------
def compute_I(ky, sigma, xi, alpha, L, Nr=20001, order=2, max_elements=5e7):
    """
    Compute I(ky) = ∫_{-L}^{L} [ (sigma^2 * exp(-(abs(r)/xi)^(2*alpha)))**order ] * exp(-1j*ky*r) dr
    using trapezoidal rule over r in [0, L] with symmetry -> cosine transform.
    Works for scalar, 1D array, and multi-D meshgrid ky (returns same shape as ky).
    """
    ky = np.asarray(ky, dtype=float)
    ky_shape = ky.shape
    scalar_in = (ky.ndim == 0)
    ky_flat = ky.ravel()

    # r-grid (0..L because integrand is even)
    r = np.linspace(0.0, L, Nr//2 + 1)
    dr = r[1] - r[0]

    # R_W(r)^order on r>=0
    RWn = (sigma**2)**order * np.exp(-order * (r / xi)**(2.0*alpha))

    # integrand prefactor (2 because integrating -L..L -> 0..L)
    prefactor = 2.0 * RWn

    # choose vectorized vs loop
    if r.size * ky_flat.size <= max_elements:
        cos_term = np.cos(np.outer(r, ky_flat))  # shape (r.size, ky_flat.size)
        integrand = prefactor[:, None] * cos_term
        Iky_flat = np.trapezoid(integrand, r, axis=0)
    else:
        Iky_flat = np.empty_like(ky_flat, dtype=np.complex128)
        for i, k in enumerate(ky_flat):
            Iky_flat[i] = np.trapezoid(prefactor * np.cos(k * r), r)

    Iky = Iky_flat.reshape(ky_shape)
    Iky = Iky + 0j
    if scalar_in:
        return Iky.item()
    return Iky


def add_orders_to_diffuse(
    nums_order: int,
    ky_array: np.ndarray,
    qx_array: np.ndarray,
    sigma: float,
    xi: float,
    alpha: float,
    L: float,
    compute_I_func=None,
    return_real: bool = True,
) -> np.ndarray:
    """
    Sum Taylor-order diffuse terms:
        sum_{i=0}^{nums_order-1} (qx**(2*i)) * I(ky; order=i) * i!
    where I(ky;order) is computed by compute_I_func (defaults to compute_I in scope).

    Parameters
    ----------
    nums_order : int
        Number of Taylor orders to include (0..nums_order-1).
    ky_array, qx_array : array-like (scalar, 1D, or meshgrid)
        Arrays specifying ky and qx values. Must be broadcastable to the same shape.
        Typically ky and qx are created with np.meshgrid(..., indexing='xy').
    sigma, xi, alpha, L : floats
        Parameters passed to compute_I.
    compute_I_func : callable or None
        Function to compute I(ky, ..., order). If None, uses `compute_I` from outer scope.
        Signature: compute_I_func(ky, sigma, xi, alpha, L, order=..., ...)
    return_real : bool
        If True, returns the real part of the accumulator (imag should be ~0).
        If False, returns complex dtype.

    Returns
    -------
    taylor_order_terms : np.ndarray
        Array of same shape as the broadcast result of (qx_array, ky_array), containing the sum.
    """

    if compute_I_func is None:
        # Expect compute_I to be in scope
        try:
            compute_I_func = compute_I
        except NameError:
            raise RuntimeError("compute_I_func is None and no compute_I is available in scope.")

    # convert to numpy arrays
    ky = np.asarray(ky_array)
    qx = np.asarray(qx_array)

    # check broadcasting compatibility
    try:
        out_shape = np.broadcast_shapes(ky.shape, qx.shape)
    except Exception as e:
        raise ValueError(f"qx and ky must be broadcastable to the same shape: {e}")

    # Prepare arrays broadcasted to out_shape (we will rely on broadcasting, so no full copies)
    # Initialize accumulator in complex (safe) with broadcasting shape
    taylor_order_terms = np.zeros(out_shape, dtype=np.complex128)

    # Loop orders
    for i in np.arange(2,2+nums_order,1):

        # always start from order 2 because 0th is 1 and 1st is psd already included in diffuse

        # compute I for this order. compute_I handles ky shaped arrays (including meshgrid)
        Iky = compute_I_func(ky, sigma=sigma, xi=xi, alpha=alpha, L=L, order=i)

        # Ensure Iky is broadcastable to out_shape (it should match ky shape)
        if np.shape(Iky) != ky.shape:
            # try to reshape or broadcast (if ky was scalar)
            Iky = np.asarray(Iky)
            if Iky.shape == ():  # scalar
                Iky = np.full(ky.shape, Iky, dtype=Iky.dtype)
            else:
                # let numpy handle broadcasting, but warn if incompatible
                try:
                    np.broadcast_shapes(Iky.shape, ky.shape)
                except Exception:
                    raise ValueError(f"compute_I returned shape {Iky.shape} which is not broadcastable with ky.shape {ky.shape}")

        # compute qx**(2*i) with broadcasting to out_shape (qx can be scalar/1D/2D)
        power = 2 * i
        # handle i=0 fast
        if power == 0:
            qx_factor = 1.0
        else:
            qx_factor = np.power(qx, power)

        # factorial (may grow fast)
        try:
            fact = math.factorial(i)
        except OverflowError:
            # fallback to float gamma if extremely large (rare)
            from math import gamma
            fact = gamma(i + 2)

        # term multiplication: rely on numpy broadcasting
        print("power:",power,"fact:",fact)
        term = (qx_factor * Iky) / fact

        # accumulate
        taylor_order_terms += term

    # final postprocessing
    if return_real:
        # integrand is real to numerical precision; drop tiny imaginary part
        taylor_order_terms = np.real_if_close(taylor_order_terms, tol=1000)
        # ensure dtype float if imaginary part negligible
        if np.isrealobj(taylor_order_terms):
            taylor_order_terms = taylor_order_terms.astype(np.float64)

    return taylor_order_terms

# ------------------------------------------------------------------
# single entry point — cube or trapezoid, matches simulate_line_intensity's
# beta_r/beta_l convention
# ------------------------------------------------------------------
def extract_psd(I_slice, qx, qy, qz,
                 sigma, xi, alpha,
                 width, dy, num_cubes, height,
                 num_orders=0,
                 beta_r=None, beta_l=None,
                 return_abs=False):
    """
    Recover 1D PSD S_W(qy) from a fixed-qx intensity slice, by algebraically
    inverting the same forward model used in simulate.simulate_line_intensity
    (shared form_factors.py guarantees forward/inverse consistency).
    """
    qy = np.asarray(qy)
    I_slice = np.asarray(I_slice)

    use_trapezoid = beta_r is not None and beta_l is not None
    if use_trapezoid:
        F = trapezoid_form_factor(qx, qy, qz, width / 2.0, beta_r, beta_l, dy / 2.0, height / 2.0)
    else:
        F = cube_form_factor(qx, qy, qz, width / 2.0, dy / 2.0, height / 2.0)

    R0 = sigma**2
    L = num_cubes * dy

    taylor_order_terms = add_orders_to_diffuse(
        nums_order=num_orders, ky_array=qy, qx_array=qx,
        sigma=sigma, xi=xi, alpha=alpha, L=L,
    ) if num_orders > 0 else np.zeros_like(qy)

    damping = np.exp(-(qx**2) * R0)
    denom = np.abs(F)**2 * damping

    I1 = (I_slice * L**2) / denom

    C = np.where(qy != 0, 2 * (1 - np.cos(qy * L)) / (qy**2), L**2)

    numerator = I1 - C - (taylor_order_terms * L)
    S_recov = numerator / (L * qx**2)

    if return_abs:
        S_recov = np.abs(S_recov)

    return S_recov / (2 * np.pi)