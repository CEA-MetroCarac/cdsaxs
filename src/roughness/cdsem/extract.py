# extract.py

import numpy as np
from roughness.cdsem.contour import extract_all_contours  # replace with actual module name


def subtract_mean(contour, pixel_size: float = 1.0):
    """
    Remove flat offset from a contour: x_i(y) -= mean(x_i).

    Parameters
    ----------
    contour : list of (y, x)
        Raw contour from extract_all_contours()['contours'][i].
    pixel_size : float
        Physical size per pixel. Applied to y only (position axis),
        x stays in pixel units until PSD frequency conversion.

    Returns
    -------
    y : np.ndarray (physical units, y * pixel_size)
    x_centered : np.ndarray (mean-subtracted, still in pixel units)
    """
    y = np.array([pt[0] for pt in contour], dtype=np.float64) * pixel_size
    x = np.array([pt[1] for pt in contour], dtype=np.float64)
    x_centered = x - np.nanmean(x)
    return y, x_centered


def detrend(y, x, method: str = 'linear'):
    """
    Remove a slow trend (e.g. stage drift/skew) from x(y), separate
    from mean-subtraction. Call only if a trend is actually present --
    inspect the contour plot first.

    Parameters
    ----------
    y, x : np.ndarray
        Position arrays (from subtract_mean or raw).
    method : str
        'linear' -- fit and subtract a first-order polynomial.
        'none'   -- return x unchanged (explicit no-op passthrough).

    Returns
    -------
    x_detrended : np.ndarray
    """
    if method == 'none':
        return x
    if method == 'linear':
        coeffs = np.polyfit(y, x, deg=1)
        trend = np.polyval(coeffs, y)
        return x - trend
    raise ValueError(f"unknown detrend method: {method}")


def resample_uniform(y, x, pixel_size: float = 1.0, flagged_ys: set = None,
                      n_points: int = None):
    """
    Ensure uniform y-spacing before FFT, on a FIXED-LENGTH grid shared
    across all edges (n_points), so per-edge exclusion counts never
    change the output array length.

    Parameters
    ----------
    y, x : np.ndarray
        Position arrays, y already in physical units (pixel_size applied).
    pixel_size : float
        Used to build the uniform target grid spacing.
    flagged_ys : set, optional
        Row y-values (ORIGINAL pixel index) to exclude before interpolation.
    n_points : int, optional
        Fixed number of output samples. If None, defaults to y.size --
        callers processing multiple edges MUST pass the same n_points
        for every edge so all edges return identical-length PSDs.

    Returns
    -------
    y_uniform : np.ndarray
    x_uniform : np.ndarray
    """
    if flagged_ys:
        flagged_phys = {fy * pixel_size for fy in flagged_ys}
        mask = np.array([yy not in flagged_phys for yy in y])
        y_valid, x_valid = y[mask], x[mask]
    else:
        y_valid, x_valid = y, x

    if y_valid.size < 2:
        raise ValueError("not enough valid rows left after excluding flagged rows")

    if n_points is None:
        n_points = y.size

    # fixed sample count spanning the ORIGINAL y range so every edge's
    # grid covers the same span regardless of how many rows were excluded
    y_uniform = np.linspace(y.min(), y.max(), n_points)
    x_uniform = np.interp(y_uniform, y_valid, x_valid)

    return y_uniform, x_uniform

def compute_psd(x_uniform, pixel_size: float = 1.0, window: str = None):
    n = x_uniform.size

    if window == 'hann':
        w = np.hanning(n)
    elif window == 'welch':
        m = (n - 1) / 2
        idx = np.arange(n)
        w = 1 - ((idx - m) / m) ** 2
    elif window == 'none':
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window: {window}")

    x_windowed = x_uniform * w

    F = np.fft.rfft(x_windowed)

    # normalize by window power (coherent gain loss) and sample count,
    # scale by pixel_size for physical units, and account for one-sided spectrum
    U = np.sum(w ** 2)  # window power normalization factor
    psd = (np.abs(F) ** 2) * pixel_size / U

    # one-sided correction: double all bins except DC and (if n even) Nyquist
    psd[1:-1 if n % 2 == 0 else None] *= 2

    freqs = np.fft.rfftfreq(n, d=pixel_size)

    return freqs, psd

def process_all_edges(contours: dict, flags: dict = None,
                       pixel_size: float = 1.0,
                       detrend_method: str = 'none',
                       window: str = 'hann',
                       max_missing_frac: float = 2/3):
    """
    Run Step 3 post-processing on every edge's contour and average
    the resulting PSDs.

    Edges with more than max_missing_frac of their rows flagged are
    skipped entirely (reported in skipped_edges), not included in the
    average, and do not raise an error.

    Parameters
    ----------
    contours : dict {edge_index: [(y, x), ...]}
    flags : dict {edge_index: [{'y','reason'}, ...]}, optional
    pixel_size : float
    detrend_method : str
    window : str
    max_missing_frac : float
        Fraction of flagged rows above which an edge is dropped
        entirely (default 2/3).

    Returns
    -------
    result : dict
        {
          'freqs'          : shared frequency axis,
          'psd_per_edge'   : dict {edge_index: psd array} (kept edges only),
          'psd_avg'        : array, mean |F_n|^2 across kept edges,
          'skipped_edges'  : dict {edge_index: missing_frac} (dropped edges),
        }
    """
    psd_per_edge = {}
    skipped_edges = {}
    freqs_ref = None

    # fixed n_points for ALL edges -- length of the longest contour --
    # so every edge's PSD comes out the same length regardless of
    # per-edge exclusion counts
    n_points = max(len(c) for c in contours.values())

    for i, contour in contours.items():
        n_total = len(contour)
        n_flagged = len(flags[i]) if flags is not None else 0
        missing_frac = n_flagged / n_total if n_total > 0 else 1.0

        if missing_frac > max_missing_frac:
            skipped_edges[i] = missing_frac
            continue

        y, x_centered = subtract_mean(contour, pixel_size=pixel_size)
        x_processed = detrend(y, x_centered, method=detrend_method)

        flagged_ys = {f['y'] for f in flags[i]} if flags is not None else None
        y_uniform, x_uniform = resample_uniform(
            y, x_processed, pixel_size=pixel_size,
            flagged_ys=flagged_ys, n_points=n_points,
        )

        freqs, psd = compute_psd(x_uniform, pixel_size=pixel_size, window=window)

        if freqs_ref is None:
            freqs_ref = freqs

        psd_per_edge[i] = psd

    if not psd_per_edge:
        raise RuntimeError("all edges skipped -- no usable contours remain")

    psd_avg = np.mean(np.stack(list(psd_per_edge.values())), axis=0)

    return {
        'freqs': freqs_ref,
        'psd_per_edge': psd_per_edge,
        'psd_avg': psd_avg,
        'skipped_edges': skipped_edges,
    }


def run_pipeline(path: str, pixel_size: float = 1.0,
                  detrend_method: str = 'none',
                  window: str = 'hann'):
    """
    Full pipeline: load TIF -> extract contours -> post-process -> PSD.

    Parameters
    ----------
    path : str
        Path to the .tif image.
    pixel_size : float
        Physical size per pixel, default 1 (pixel units).
    detrend_method : str
        'none' (default) or 'linear', passed to process_all_edges.
    window : str
        Window function for PSD computation.

    Returns
    -------
    result : dict
        Output of extract_all_contours (image, contours, flags, etc.)
    psd_result : dict
        {'freqs', 'psd_per_edge', 'psd_avg'} from process_all_edges.
    """
    result = extract_all_contours(path)

    psd_result = process_all_edges(
        contours=result['contours'],
        flags=result['flags'],
        pixel_size=pixel_size,
        detrend_method=detrend_method,
        window=window,
    )

    return result, psd_result