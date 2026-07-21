# ============================================================
# ALGORITHM (v2): Filter-free extraction of line-edge contours
# from a TIF image of vertical line gratings (|| || ||)
# Following Verduin, Kruit & Hagen (2014), incorporating the
# better structural aspects of the reference implementation
# (bounded coefficient-based fitting, unified fit routine,
#  explicit free/fixed parameter control, built-in QC plots)
# ============================================================
#
# Image convention:
#   I[row, col]  -> row = y position (pick top-to-bottom or
#                   bottom-to-top and stay consistent)
#                -> col = x position, left -> right
#   Each row is a horizontal scan crossing all M vertical lines.
#   Roughness = wobble of each edge's x-position as y varies.


# ---------- STEP 0: Load & orient the image -----------------
# 1. Load TIF into array I[y, x]
# 2. Orient consistently (flip if needed so array-row order
#    matches your intended physical y direction)
# 3. Crop out borders/scale bars/artifacts if present
#
# 4. Average all rows together along y to collapse roughness
#    and pixel noise simultaneously:
#       prof_mean[x] = mean over y of I[y, x]
#    (this is the "reference profile" -- Fig. 6 of the paper)
#
# 5. Auto-detect approximate edge locations in prof_mean
#    (peak/derivative/threshold-crossing search)
#    -> gives M initial index windows [imin_i, imax_i], one
#       per edge, wide enough to contain both Gaussian tails
#       of that edge but not overlap its neighbor
#
#    # TODO (parallelization): keep this detection step
#    # SEQUENTIAL and upfront, before any parallel dispatch.
#    # It needs the full averaged profile at once, is cheap
#    # (single pass over a 1D array), and produces the M
#    # edge windows that later get fanned out to workers.
#    # Do not attempt to parallelize this step itself.


# ---------- STEP 1: Build the reference edge profile ---------
# GOAL: a smooth, noise-free template of one edge transition,
#       built without ever blurring the raw image.
#
#
#
# 1. For each edge window, extract:
#       y_ref  = prof_mean[imin:imax]      (the crop to fit)
#       x_ref  = local index array (0 .. imax-imin)
#
# 2. Build an initial parameter guess directly FROM the data,
#    rather than hand-picked constants:
#       x0      = index of the peak/inflection in y_ref
#       ampli   = y_ref value at that peak
#       sigma_l = sigma_r = a reasonable starting width (e.g. ~5 px)
#       base_l  = min of y_ref to the left of x0
#       base_r  = min of y_ref to the right of x0
#    -> params_ref_init = (x0, ampli, sigma_l, base_l, sigma_r, base_r)
#
# 3. Fit the double-Gaussian model (paper Eq. 7) to y_ref using
#    a SINGLE unified fitting routine that will be reused later
#    for the row-by-row fits too. This routine:
#       - re-parametrizes each physical parameter as
#             param = coef * param_init
#         and fits the dimensionless coefficients instead of the
#         raw values directly
#       - bounds each coefficient to a safe multiplicative range
#         (e.g. 0.5x - 1.5x of its initial value) to keep the
#         optimizer well-scaled and prevent runaway/unphysical
#         solutions (negative widths, exploding amplitude, etc.)
#       - EXCEPTION: the position parameter (x0) must NOT use a
#         multiplicative bound (breaks down near zero / small
#         values). Bound it additively instead, e.g.
#             x0 in [x0_init - delta, x0_init + delta]
#         where delta reflects the expected roughness amplitude
#         (a few sigma of expected LER)
#       - solve with a proper bounded nonlinear least-squares
#         solver (trust-region-reflective), not a generic
#         minimizer on hand-rolled MSE
# 4. Store the fitted reference parameters params_ref[i] for
#    each edge i -- this is the fixed shape template used below.
#
# [Optional QC] plot y_ref vs. initial-guess curve vs. fitted
# curve to visually confirm the reference template is sane
# before moving to noisy row-by-row data.
#
#    # TODO (parallelization): Steps 1+2 together form the
#    # natural unit of parallel work. Refactor into a single
#    # callable, e.g. process_single_edge(image_slice, window)
#    # -> contour, that internally does both the reference-fit
#    # (this step) and the row-by-row fit (Step 2) for ONE edge.
#    # This function must be self-contained: no shared mutable
#    # state (shared plot lists, shared result dicts, shared
#    # RNG, etc.) across edges, so it's safe to dispatch to
#    # parallel workers independently.


# ---------- STEP 2: Row-by-row fitting on RAW data ------------
# GOAL: recover x_edge(y) per row, per edge, directly against
#       raw unfiltered pixel intensities -- no image smoothing.
#
# For each edge i in 1..M:
#
#   # TODO (parallelization): this outer loop over edges i is
#   # "embarrassingly parallel" -- each edge is fully independent
#   # of the others. Replace with a map over process_single_edge
#   # across a process pool (e.g. concurrent.futures.
#   # ProcessPoolExecutor, multiprocessing.Pool, or joblib.Parallel).
#   #   - Use PROCESSES, not threads: the bottleneck is repeated
#   #     scipy nonlinear least-squares calls (CPU-bound), and
#   #     the GIL prevents threads from giving real speedup here.
#   #   - Pass only the cropped column slice for that edge's
#   #     window to each worker, not the full image, to minimize
#   #     inter-process data transfer / pickling cost.
#   #   - Tag each dispatched task and returned result with its
#   #     edge index i, since parallel results may complete out
#   #     of order -- reassemble contours[i] by tag, not by
#   #     append order.
#
#   Explicitly declare which parameters are free vs. fixed for
#   the row-by-row fit (do NOT default to "all free"):
#       free  = {amplitude, x0 (position)}
#       fixed = {sigma_l, sigma_r, base_l, base_r}  <- locked to
#               params_ref[i] from Step 1, since row noise
#               should manifest as a POSITION shift, not as
#               a change in the reference edge's shape.
#       (this mask is a parameter of the fitting routine, kept
#        general enough to relax later for diagnostics, but
#        defaulted to the physically correct 2-free-parameter
#        case matching the paper's Eq. 8)
#
#   x_prev = the edge window's initial x0 from Step 1
#            (running "warm start" position)
#
#   contour_i = empty list of (y, x_edge) pairs
#
#   for each row y (in consistent order):
#
#       1. Extract the RAW pixel window around x_prev (same
#          crop width as used for the reference template)
#
#       2. Fit using the SAME unified fitting routine as Step 1,
#          but with only {amplitude, x0} free and everything
#          else fixed to params_ref[i]. Initialize the fit from
#          x_prev (NOT from the global reference position) so
#          consecutive rows warm-start each other -- roughness
#          is spatially correlated over the correlation length,
#          so this speeds convergence and avoids the fit
#          jumping to the wrong edge or a noise peak.
#
#          # TODO (parallelization): note that this row-to-row
#          # warm-start creates a SEQUENTIAL DEPENDENCY within
#          # a single edge's row loop (row y needs row y-1's
#          # result). This loop is fine to run sequentially
#          # within a worker (each edge is already its own
#          # parallel task). Only attempt to parallelize WITHIN
#          # an edge (e.g. splitting rows into chunks across
#          # workers) if edges are few and very tall (large
#          # N_rows) -- and if so, break the warm-start chaining
#          # at chunk boundaries by re-seeding each chunk from
#          # the Step 1 reference x0 instead of the previous
#          # row, trading some convergence robustness for
#          # finer-grained parallelism.
#
#       3. Recovered x0 for this row = x_edge(y)
#
#       4. Append (y, x_edge) to contour_i
#
#       5. Update running guess: x_prev = x_edge(y)
#
#       6. [QC] flag rows where:
#            - fitted coefficient hits its bound
#            - residual/cost is unusually high
#            - solver fails to converge
#          and exclude or mark these rows rather than silently
#          accepting a bad fit
#
#       7. [Optional] generate a diagnostic plot (raw row data
#          vs. reference-template curve vs. fitted curve) for
#          spot-checking a sample of rows, not necessarily all
#          of them (for performance)
#
#          # TODO (parallelization): plotting must NOT happen
#          # inside a parallel worker -- matplotlib is not
#          # reliably safe across threads/processes and figure
#          # objects don't serialize well. Have workers return
#          # only the numeric arrays needed for plotting (raw
#          # row, fitted params), and defer all actual plot
#          # calls to the main process AFTER results are
#          # collected back.
#
#   store contours[i] = contour_i


# ---------- STEP 3: Post-process contours ---------------------
# For each edge's contour x_i(y):
#   1. Subtract the mean position: x_i(y) -= mean(x_i)
#   2. Resample to uniform y-spacing if needed
#   3. Ready for PSD analysis: FFT along y, average |F_n|^2
#      across edges, fit Palasantzas model + white-noise term
#      to extract LER (sigma), correlation length (xi),
#      roughness exponent (alpha)


# ---------- OUTPUT ---------------------------------------------
# contours = {edge_index: [(y_1, x_1), ..., (y_N, x_N)]}
# -> one wiggly x(y) trace per vertical line edge, extracted
#    without any image blurring/filtering, with bounded and
#    warm-started per-row fits against raw pixel data.
#
#
# ============================================================
# TODO SUMMARY (future parallelization work):
#   1. Refactor Steps 1+2 into a single self-contained callable
#      process_single_edge(image_slice, window) -> contour
#   2. Keep edge-window detection (Step 1.2) sequential/upfront;
#      do not parallelize it
#   3. Parallelize the per-edge loop (Step 2) via a process pool
#      (not threads -- GIL blocks speedup for CPU-bound scipy fits)
#   4. Pass only cropped column slices to workers, not full image
#   5. Tag tasks/results with edge index; reassemble by tag, not
#      by completion/append order
#   6. Row-to-row warm-start within an edge is a sequential
#      dependency -- keep it sequential inside each worker;
#      only break the chain if parallelizing within an edge too
#   7. Move all QC plotting out of workers; defer to main process
#      using numeric results returned from workers
# ============================================================


import numpy as np
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks

def detect_edges(path: str | Path, axis: int = 0,
                  prominence: float = None,
                  window_frac: float = 0.4,
                  missing_thresh_frac: float = 0.10):
    """
    Detect vertical line-grating edges from the averaged 1D profile
    and return peak positions plus per-edge crop windows for
    downstream double-Gaussian fitting.

    Peak spacing is not assumed -- all peaks above the prominence
    threshold are detected first, then each edge's window is sized
    from its own actual neighbor distances so windows never overlap.

    Any detected peak whose profile value falls below
    missing_thresh_frac * mean(profile) is flagged as a missing
    contour and excluded from the returned peaks/windows -- it
    is reported separately, not silently dropped.

    Parameters
    ----------
    path : str or Path
        Path to the .tif image.
    axis : int
        Axis averaged over to build the profile (0 -> profile vs x).
    prominence : float, optional
        Minimum peak prominence. If None, estimated as 10% of profile
        dynamic range (max - min).
    window_frac : float
        Fraction of the local inter-peak spacing used as half-width
        of each edge's crop window. Must be < 0.5 to avoid overlap.
    missing_thresh_frac : float
        Fraction of mean(profile) below which a peak is considered
        a missing contour rather than a valid edge.

    Returns
    -------
    result : dict
        {
          'profile'        : 1D averaged profile,
          'peaks'          : valid peak indices (edge center guesses),
          'windows'        : list of (imin, imax) per valid edge,
          'missing_peaks'  : peak indices excluded as missing contours,
          'image'          : loaded 2D image (for plotting),
        }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no file at {path}")

    image = np.array(Image.open(path), dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got shape {image.shape}")

    profile = np.nanmean(image, axis=axis)
    n = profile.size

    if prominence is None:
        prominence = 0.1 * (np.nanmax(profile) - np.nanmin(profile))

    if window_frac >= 0.5:
        raise ValueError("window_frac must be < 0.5 to avoid overlapping edge windows")

    all_peaks, _ = find_peaks(profile, prominence=prominence)

    if all_peaks.size == 0:
        raise RuntimeError("no peaks detected; check prominence setting")
    if all_peaks.size == 1:
        raise RuntimeError("only one peak detected; cannot infer spacing for windows")

    # missing-contour filter: peak amplitude vs mean profile level,
    # computed on ALL detected peaks before windowing so excluded
    # peaks don't warp neighbor-gap calculations for valid ones
    mean_level = np.nanmean(profile[all_peaks])
    thresh = mean_level - missing_thresh_frac * mean_level

    valid_mask = profile[all_peaks] >=  thresh
    missing_peaks = all_peaks[~valid_mask]
    peaks = all_peaks[valid_mask]

    if peaks.size < 2:
        raise RuntimeError(
            f"fewer than 2 valid peaks after missing-contour filter "
            f"({peaks.size} valid, {missing_peaks.size} flagged missing)"
        )

    # window sizing uses only the VALID peaks' neighbor gaps --
    # a missing peak should not shrink its valid neighbors' windows
    windows = []
    for i, p in enumerate(peaks):
        left_gap = (p - peaks[i - 1]) if i > 0 else (peaks[i + 1] - p)
        right_gap = (peaks[i + 1] - p) if i < peaks.size - 1 else (p - peaks[i - 1])
        half_width = int(window_frac * min(left_gap, right_gap))
        imin = max(0, p - half_width)
        imax = min(n, p + half_width)
        windows.append((imin, imax))

    return {
        'profile': profile,
        'peaks': peaks,
        'windows': windows,
        'missing_peaks': missing_peaks,
        'image': image,
    }



#######################################################################Double gaussian Model fitting####################################################################
import numpy as np
from scipy.optimize import least_squares


# ---------- Double-Gaussian edge model (paper Eq. 7) ----------
def double_gaussian(x, x0, ampli, sigma_l, base_l, sigma_r, base_r):
    """
    Asymmetric double-Gaussian edge transition model.

    Left side (x < x0): decays from ampli down to base_l with sigma_l.
    Right side (x >= x0): decays from ampli down to base_r with sigma_r.

    Parameters
    ----------
    x : array
        Local index array.
    x0, ampli, sigma_l, base_l, sigma_r, base_r : float
        Physical model parameters.

    Returns
    -------
    y : array, same shape as x
    """
    y = np.empty_like(x, dtype=np.float64)
    left = x < x0
    right = ~left
    y[left] = base_l + (ampli - base_l) * np.exp(-(x[left] - x0) ** 2 / (2 * sigma_l ** 2))
    y[right] = base_r + (ampli - base_r) * np.exp(-(x[right] - x0) ** 2 / (2 * sigma_r ** 2))
    return y


def _initial_guess(x_ref, y_ref, default_sigma=None):
    peak_idx = int(np.nanargmax(y_ref))
    x0 = x_ref[peak_idx]
    ampli = y_ref[peak_idx]

    left_vals = y_ref[:peak_idx + 1]
    right_vals = y_ref[peak_idx:]

    base_l = np.nanmin(left_vals) if left_vals.size > 0 else y_ref[0]
    base_r = np.nanmin(right_vals) if right_vals.size > 0 else y_ref[-1]

    if default_sigma is None:
        half_max = base_l + 0.5 * (ampli - base_l)
        left_cross = np.where(left_vals <= half_max)[0]
        sigma_l_est = (x0 - x_ref[left_cross[-1]]) / 1.1774 if left_cross.size > 0 else 5.0

        half_max_r = base_r + 0.5 * (ampli - base_r)
        right_cross = np.where(right_vals <= half_max_r)[0]
        sigma_r_est = (x_ref[peak_idx + right_cross[0]] - x0) / 1.1774 if right_cross.size > 0 else 5.0

        sigma_l = max(1.0, sigma_l_est)
        sigma_r = max(1.0, sigma_r_est)
    else:
        sigma_l = sigma_r = default_sigma

    return {
        'x0': x0,
        'ampli': ampli,
        'sigma_l': sigma_l,
        'base_l': base_l,
        'sigma_r': sigma_r,
        'base_r': base_r,
    }


# ---------- Unified fitting routine (reused in Step 2) ----------
PARAM_ORDER = ['x0', 'ampli', 'sigma_l', 'base_l', 'sigma_r', 'base_r']


def fit_double_gaussian(x_data, y_data, params_init: dict,
                         free_params: set = None,
                         mult_bound_frac: float = 1.0,
                         x0_delta: float = None):
    """
    Fit the double-Gaussian model via coefficient reparametrization:
    each physical param = coef * param_init (except x0, additive).

    Parameters
    ----------
    x_data, y_data : array
        Cropped local index array and corresponding raw/averaged intensities.
    params_init : dict
        Initial physical parameter guess, keys matching PARAM_ORDER.
    free_params : set of str, optional
        Which physical parameters to actually fit. Defaults to all
        (used for Step 1 reference fit). Step 2 row fits should pass
        {'x0', 'ampli'} to lock shape params to the reference template.
    mult_bound_frac : float
        Multiplicative bound half-width for non-x0 params
        (e.g. 0.5 -> bounds are [0.5x, 1.5x] of init value).
    x0_delta : float, optional
        Additive bound half-width for x0. If None, defaults to
        max(3.0, 0.1 * window width) as a conservative fallback --
        this should be set explicitly from expected LER amplitude
        when known.

    Returns
    -------
    result : dict
        {
          'params'      : dict of fitted physical parameter values,
          'success'     : bool,
          'cost'        : float,
          'hit_bounds'  : list of param names that landed on a bound,
          'raw_result'  : scipy OptimizeResult,
        }
    """
    if free_params is None:
        free_params = set(PARAM_ORDER)

    fixed_params = set(PARAM_ORDER) - free_params
    for p in fixed_params:
        if p not in params_init:
            raise ValueError(f"fixed parameter '{p}' missing from params_init")

    if x0_delta is None:
        x0_delta = max(3.0, 0.1 * (x_data[-1] - x_data[0]))

    free_order = [p for p in PARAM_ORDER if p in free_params]
    if not free_order:
        raise ValueError("at least one parameter must be free")

    # coefficients start at 1.0 (since param = coef * param_init),
    # except x0 which is fit directly in additive space
    coef0 = []
    lower = []
    upper = []
    for p in free_order:
        init_val = params_init[p]
        if p == 'x0':
            coef0.append(init_val)
            lower.append(init_val - x0_delta)
            upper.append(init_val + x0_delta)
        else:
            if init_val == 0:
                raise ValueError(
                    f"param '{p}' has zero init value; multiplicative "
                    f"bound is ill-conditioned -- check params_init"
                )
            coef0.append(1.0)
            lo, hi = sorted([
                (1 - mult_bound_frac),
                (1 + mult_bound_frac),
            ])
            lower.append(lo)
            upper.append(hi)

    def _unpack(coefs):
        full = dict(params_init)  # start from fixed values
        for name, c in zip(free_order, coefs):
            if name == 'x0':
                full[name] = c  # x0 fit directly, not as a coefficient
            else:
                full[name] = c * params_init[name]
        return full

    def residuals(coefs):
        full = _unpack(coefs)
        model = double_gaussian(
            x_data, full['x0'], full['ampli'],
            full['sigma_l'], full['base_l'],
            full['sigma_r'], full['base_r'],
        )
        return model - y_data

    fit = least_squares(
        residuals, x0=coef0, bounds=(lower, upper),
        method='trf',  # trust-region-reflective
    )

    fitted_params = _unpack(fit.x)

    # flag coefficients that landed on (or essentially on) a bound
    hit_bounds = []
    tol = 1e-6
    for name, val, lo, hi in zip(free_order, fit.x, lower, upper):
        if abs(val - lo) < tol * max(1.0, abs(lo)) or abs(val - hi) < tol * max(1.0, abs(hi)):
            hit_bounds.append(name)

    return {
        'params': fitted_params,
        'success': fit.success,
        'cost': fit.cost,
        'hit_bounds': hit_bounds,
        'raw_result': fit,
    }


# ---------- Step 1 driver: build reference template per edge ----------
def build_reference_templates(profile: np.ndarray, windows: list,
                               default_sigma: float = 5.0,
                               x0_delta: float = None):
    """
    Fit the double-Gaussian model to each edge window in the averaged
    profile, producing the fixed shape template used in Step 2.

    Parameters
    ----------
    profile : np.ndarray
        1D averaged profile (output of average_profile / detect_edges).
    windows : list of (imin, imax)
        Per-edge crop windows (output of detect_edges).
    default_sigma : float
        Starting sigma guess for initial parameters.
    x0_delta : float, optional
        Additive x0 bound half-width, passed through to fit_double_gaussian.

    Returns
    -------
    templates : list of dict
        One entry per edge:
        {
          'params_init' : dict (initial guess),
          'fit'         : dict (fit_double_gaussian output),
          'x_ref'       : local index array used for the fit,
          'y_ref'       : cropped profile data used for the fit,
        }
    """
    templates = []
    for imin, imax in windows:
        y_ref = profile[imin:imax]
        x_ref = np.arange(y_ref.size, dtype=np.float64)

        if np.any(np.isnan(y_ref)):
            raise ValueError(
                f"NaN present in window [{imin}:{imax}]; "
                f"resolve masking before reference fit"
            )

        params_init = _initial_guess(x_ref, y_ref, default_sigma=default_sigma)
        fit_result = fit_double_gaussian(
            x_ref, y_ref, params_init,
            free_params=set(PARAM_ORDER),  # all free for reference fit
            x0_delta=x0_delta,
        )

        if not fit_result['success']:
            raise RuntimeError(f"reference fit failed to converge for window [{imin}:{imax}]")
        if fit_result['hit_bounds']:
            print(f"warning: window [{imin}:{imax}] fit hit bounds on: {fit_result['hit_bounds']}")

        templates.append({
            'params_init': params_init,
            'fit': fit_result,
            'x_ref': x_ref,
            'y_ref': y_ref,
        })

    return templates

####################################################################################################################################################################


###################################################################extract contour using previous functions##########################################################
def trace_edge_contour(image: np.ndarray, window_half_width: int,
                        params_ref: dict, x0_init: float,
                        cost_thresh: float = None):
    """
    Recover x_edge(y) for one edge by fitting row-by-row against
    RAW pixel data, warm-started from the previous row's position.

    Parameters
    ----------
    image : np.ndarray
        Full raw 2D image, I[y, x].
    window_half_width : int
        Half-width of the crop window around x_prev (same size
        used to build the Step 1 reference template).
    params_ref : dict
        Fitted reference parameters for this edge (from
        build_reference_templates()[i]['fit']['params']).
        sigma_l, base_l, sigma_r, base_r are locked to these values.
    x0_init : float
        Starting position estimate (this edge's Step 1 x0, in
        FULL IMAGE coordinates, not window-local).
    cost_thresh : float, optional
        Flag rows whose fit cost exceeds this value. If None,
        no cost-based flagging (only bound-hit / non-convergence flagged).

    Returns
    -------
    contour : list of (y, x_edge)
        Recovered edge position per row, in full image coordinates.
    flags : list of dict
        One entry per row: {'y', 'reason'} for any flagged row.
        Flagged rows are still included in contour (marked, not dropped) --
        decide exclude-vs-interpolate downstream in Step 3.
    """
    n_rows = image.shape[0]
    contour = []
    flags = []

    x_prev = x0_init  # full-image coordinate, running warm-start

    for y in range(n_rows):
        # 1. crop RAW row around x_prev
        imin = int(round(x_prev - window_half_width))
        imax = int(round(x_prev + window_half_width))
        imin_c = max(0, imin)
        imax_c = min(image.shape[1], imax)

        y_row = image[y, imin_c:imax_c].astype(np.float64)
        x_row = np.arange(imin_c, imax_c, dtype=np.float64)  # full-image coords

        if np.any(np.isnan(y_row)) or y_row.size < 4:
            flags.append({'y': y, 'reason': 'insufficient/NaN data in window'})
            contour.append((y, x_prev))  # hold last known position
            continue

        # 2. fit only {x0, ampli}, shape params locked to reference
        params_init = dict(params_ref)  # start from reference (shape) values
        params_init['x0'] = x_prev       # warm-start position from previous row

        fit_result = fit_double_gaussian(
            x_row, y_row, params_init,
            free_params={'x0', 'ampli'},
        )

        x_edge = fit_result['params']['x0']

        # 6. QC flags -- recorded, not silently accepted
        reasons = []
        if not fit_result['success']:
            reasons.append('did not converge')
        if fit_result['hit_bounds']:
            reasons.append(f"hit bounds: {fit_result['hit_bounds']}")
        if cost_thresh is not None and fit_result['cost'] > cost_thresh:
            reasons.append(f"high cost: {fit_result['cost']:.3g}")

        if reasons:
            flags.append({'y': y, 'reason': '; '.join(reasons)})

        # 3+4. record this row's edge position
        contour.append((y, x_edge))

        # 5. update running guess for next row
        x_prev = x_edge

    return contour, flags


def extract_all_contours(path: str | Path, axis: int = 0,
                          prominence: float = None,
                          window_frac: float = 0.4,
                          missing_thresh_frac: float = 0.10,
                          default_sigma: float = 5.0,
                          x0_delta: float = None,
                          cost_thresh: float = None):
    """
    End-to-end pipeline: load TIF -> detect edges -> build reference
    templates -> trace row-by-row contours for every edge.

    Parameters
    ----------
    path : str or Path
        Path to the .tif image.
    axis, prominence, window_frac, missing_thresh_frac :
        Passed to detect_edges().
    default_sigma, x0_delta :
        Passed to build_reference_templates().
    cost_thresh : float, optional
        Passed to trace_edge_contour() for QC flagging.

    Returns
    -------
    result : dict
        {
          'image'          : raw 2D image,
          'profile'        : averaged 1D profile,
          'peaks'          : valid edge peak indices,
          'missing_peaks'  : peaks excluded as missing contours,
          'windows'        : per-edge crop windows,
          'templates'      : per-edge Step 1 reference fit results,
          'contours'       : dict {edge_index: [(y, x_edge), ...]},
          'flags'          : dict {edge_index: [{'y','reason'}, ...]},
        }
    """
    det = detect_edges(
        path, axis=axis, prominence=prominence,
        window_frac=window_frac, missing_thresh_frac=missing_thresh_frac,
    )

    templates = build_reference_templates(
        profile=det['profile'], windows=det['windows'],
        default_sigma=default_sigma, x0_delta=x0_delta,
    )

    image = det['image']
    contours = {}
    flags = {}

    for i, (peak, window, tmpl) in enumerate(zip(det['peaks'], det['windows'], templates)):
        window_half_width = (window[1] - window[0]) // 2

        contour_i, flags_i = trace_edge_contour(
            image=image,
            window_half_width=window_half_width,
            params_ref=tmpl['fit']['params'],
            x0_init=float(peak),
            cost_thresh=cost_thresh,
        )

        contours[i] = contour_i
        flags[i] = flags_i

    return {
        'image': image,
        'profile': det['profile'],
        'peaks': det['peaks'],
        'missing_peaks': det['missing_peaks'],
        'windows': det['windows'],
        'templates': templates,
        'contours': contours,
        'flags': flags,
    }


####################################################################################################################################################################
