# ============================================================
# ALGORITHM (v3): Parallelizing the edge-contour extraction
# pipeline from contour.py
#
# Strategy: one task per edge, running processes (not threads),
# minimal data transfer, tagged/reassembled results, sequential
# row warm-start preserved inside each task, plotting deferred
# to main process.
# ============================================================


# ---------- STEP A: Sequential pre-work (unchanged) -----------
# 1. Call detect_edges(path, ...) exactly as today, in the main
#    process, BEFORE any parallel dispatch.
#    -> This is cheap (single pass over 1D averaged profile) and
#       needs the FULL profile at once. Do not parallelize it.
# 2. Result gives you:
#       image            (full 2D array)
#       profile          (1D averaged profile)
#       peaks[i]         (approx edge center, full-image coords)
#       windows[i]       (imin, imax) per edge, full-image coords
#       missing_peaks    (excluded, reported separately)


# ---------- STEP B: Build the self-contained per-edge task ----
# GOAL: refactor Steps 1+2 (+3) of the original algorithm into
# ONE function that a worker process can run independently,
# start to finish, for a single edge i.
#
# def process_single_edge(column_slice, window_local_offset,
#                          peak_full_coord, edge_index,
#                          default_sigma, x0_delta, cost_thresh):
#
#     # column_slice: ONLY the cropped columns [imin:imax] of the
#     #               FULL image for this edge -- not the whole image.
#     #               Sliced OUT in the main process before dispatch
#     #               (see Step C) to keep pickling cost low.
#
#     try:
#         # ---- Step 1 (reference template), scoped to this edge ----
#         a. y_ref = mean(column_slice, axis=rows)   # or reuse profile
#                     slice already computed in Step A -- either works,
#                     but slicing column_slice avoids re-touching the
#                     full profile array inside the worker
#         b. x_ref = local index array (0 .. width-1)
#         c. params_init = initial_guess(x_ref, y_ref, default_sigma)
#         d. fit_ref = fit_double_gaussian(x_ref, y_ref, params_init,
#                         free_params=ALL, x0_delta=x0_delta)
#         e. if not fit_ref.success: raise -> caught below, tagged
#         f. params_ref = fit_ref.params   # locked shape template
#
#         # ---- Step 2 (row-by-row raw trace), SEQUENTIAL within
#         #      this task -- warm-start dependency is preserved ----
#         g. x_prev = peak_full_coord   # full-image coord, running seed
#         h. contour_i = []
#            flags_i   = []
#         i. for y in range(n_rows):        # strictly sequential loop
#                - crop RAW row window around x_prev from column_slice
#                  (or from full image if column_slice doesn't cover
#                  drift beyond original window -- see Step C note)
#                - fit {x0, ampli} free, rest fixed to params_ref,
#                  warm-started from x_prev
#                - record (y, x_edge); flag bound-hits / non-convergence
#                  / high cost, same as current trace_edge_contour()
#                - x_prev = x_edge   # feeds next iteration
#
#         # ---- Step 3 (post-process), folded in here to avoid a
#         #      second round-trip through the pool ----
#         j. x_i(y) -= mean(x_i)             # de-mean
#         k. resample to uniform y-spacing if needed
#
#         return {
#             'edge_index': edge_index,       # TAG -- see Step D
#             'contour': contour_i,
#             'flags': flags_i,
#             'params_ref': params_ref,
#             'status': 'ok',
#             # numeric-only diagnostic payload for optional QC plots,
#             # e.g. a small sample of (raw_row, fitted_curve) pairs --
#             # NEVER a matplotlib figure object (doesn't pickle safely)
#             'plot_samples': [...],
#         }
#
#     except Exception as exc:
#         # ---- isolate failure to this edge only ----
#         return {
#             'edge_index': edge_index,
#             'status': 'failed',
#             'error': str(exc),
#         }
#
#     # NOTE: no shared mutable state referenced anywhere above --
#     # no shared plot lists, no shared result dict, no shared RNG.
#     # Everything this function touches is passed in as an argument
#     # or created locally. This is what makes it dispatch-safe.


# ---------- STEP C: Prepare per-edge payloads in main process ---
# For each edge i, BEFORE dispatch:
#   1. Slice column_slice_i = image[:, windows[i][0]:windows[i][1]]
#      -> only this narrow vertical strip crosses the process
#         boundary, not the full image.
#   2. NOTE on drift: if a row's true edge can wander outside the
#      original Step-1 window over the course of many rows, either:
#        (a) pad column_slice_i a bit wider than the Step-1 window
#            to give row-fits room to drift, or
#        (b) pass the full image reference (memmap) instead of a
#            copy, if using joblib.Parallel's automatic memmapping
#            for large images, and crop per-row inside the worker.
#      Pick (a) for typical LER amplitudes; fall back to (b) only
#      if drift routinely exceeds the window.
#   3. Bundle (column_slice_i, local_offset_i, peak_i, i, ...) as
#      one task's arguments.


# ---------- STEP D: Dispatch across a process pool --------------
# 1. pool_size = min(M, n_available_cpus)
# 2. Use concurrent.futures.ProcessPoolExecutor (or joblib.Parallel)
#    to map process_single_edge over the M per-edge payloads.
# 3. Do NOT rely on completion order. Each result dict carries its
#    own 'edge_index' tag (Step B). As results arrive:
#       contours[result['edge_index']] = result['contour']
#       flags[result['edge_index']]    = result['flags']
#    -> reassemble by tag, not by append/completion order.
# 4. If using Executor.map(), note order IS technically preserved,
#    but still tag+key by edge_index explicitly -- keeps failure
#    handling and future refactors safe regardless of API used.
# 5. Collect any 'status': 'failed' results separately; log/report
#    per-edge errors without aborting the whole run.


# ---------- STEP E: Post-collection work in MAIN process only ---
# 1. All matplotlib/QC plotting happens HERE, using the numeric
#    'plot_samples' returned by each worker -- never inside a
#    worker process.
# 2. Any cross-edge aggregation (e.g. averaging |F_n|^2 across
#    edges for the PSD -> Palasantzas fit) happens here too, since
#    it inherently needs all edges' results together.
# 3. Emit final output structure, same shape as today:
#       contours = {edge_index: [(y, x_edge), ...]}
#       flags    = {edge_index: [{'y', 'reason'}, ...]}
#    plus a separate 'failed_edges' list/dict for anything that
#    raised during processing.


# ---------- STEP F: Fallback for small-M / very-tall-image case -
# Only if M is small (few lines) AND n_rows is very large:
#   - Optionally chunk rows within a single edge's task across
#     sub-workers.
#   - Re-seed each chunk's warm-start from the Step-1 reference x0
#     rather than the true previous row at chunk boundaries --
#     this trades some convergence robustness for finer-grained
#     parallelism. Not the default path; only use if profiling
#     shows the one-task-per-edge split leaves cores idle.


# ============================================================
# SUMMARY OF CHANGES vs. current contour.py:
#   - extract_all_contours() becomes a thin orchestrator:
#       detect_edges() [sequential]
#       -> build per-edge payloads (sliced columns) [sequential]
#       -> pool.map(process_single_edge, payloads) [parallel]
#       -> reassemble by edge_index tag [sequential]
#       -> QC plotting + PSD aggregation [sequential, main process]
#   - build_reference_templates() + trace_edge_contour() logic
#     get merged INTO process_single_edge(), so each worker does
#     both steps for its one edge without a second dispatch round.
#   - fit_double_gaussian() and double_gaussian() stay as-is --
#     they're already pure functions with no shared state, safe
#     to call from inside a worker unchanged.
# ============================================================
"""
Parallel version of the edge-contour extraction pipeline.

Design (per the parallelization plan):
  - detect_edges() stays sequential/upfront (Step A)
  - process_single_edge() is the self-contained per-edge unit of work:
    reference fit (Step 1) + sequential row trace (Step 2) + de-mean
    post-processing (Step 3), all for ONE edge, with no shared mutable
    state -- safe to run in a separate process (Step B)
  - Only a padded column slice of the image (not the full image) is
    shipped to each worker (Step C)
  - Dispatch over a ProcessPoolExecutor, tagged/reassembled by
    edge_index, not by completion order (Step D)
  - Any plotting / cross-edge aggregation happens back in the main
    process after collection (Step E)
"""
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from roughness.cdsem.contour import (
    detect_edges,
    double_gaussian,
    _initial_guess,
    fit_double_gaussian,
    PARAM_ORDER,
)


def process_single_edge(column_slice: np.ndarray, col_offset: int,
                         peak_full: float, edge_index: int,
                         window_half_width: int,
                         orig_window: tuple,
                         default_sigma: float = 5.0,
                         x0_delta: float = None,
                         cost_thresh: float = None):
    """
    Self-contained per-edge task: reference fit + row-by-row trace +
    de-mean post-processing, for ONE edge. Runs in a worker process.

    column_slice : np.ndarray
        Padded column crop of the FULL image for this edge only,
        shape (n_rows, pad_width). This is the only image data that
        crosses the process boundary for this task. The padding gives
        row-fits room to drift; it is deliberately WIDER than the
        Step-1 detection window.
    col_offset : int
        Full-image column index corresponding to column_slice[:, 0],
        needed to convert between local and full-image x coordinates.
    peak_full : float
        Step-1 approximate peak position, in FULL IMAGE coordinates.
    edge_index : int
        Tag used to reassemble results regardless of completion order.
    window_half_width : int
        Half-width of the row-fit crop window (same as used to build
        the Step 1 reference template).
    orig_window : tuple
        (imin, imax) in FULL IMAGE coordinates -- the original,
        non-padded detect_edges() window for this edge. The Step-1
        reference template MUST be fit only within this tight window,
        not the padded slice, or it can pick up a neighboring line's
        peak when padding is wide.
    """
    try:
        # ---- Step 1: reference template, scoped to the ORIGINAL tight
        # window only (never the padded slice -- padding exists purely
        # to give Step 2's row fits room to drift, and can easily
        # contain a neighboring line's peak) ----
        orig_imin, orig_imax = orig_window
        local_imin = orig_imin - col_offset
        local_imax = orig_imax - col_offset
        ref_slice = column_slice[:, local_imin:local_imax]

        profile_local = np.nanmean(ref_slice, axis=0)
        x_ref = np.arange(profile_local.size, dtype=np.float64)
        y_ref = profile_local

        if np.any(np.isnan(y_ref)):
            raise ValueError(f"NaN in averaged profile for edge {edge_index}")

        params_init = _initial_guess(x_ref, y_ref, default_sigma=default_sigma)
        fit_ref = fit_double_gaussian(
            x_ref, y_ref, params_init,
            free_params=set(PARAM_ORDER),
            x0_delta=x0_delta,
        )
        if not fit_ref['success']:
            raise RuntimeError(f"reference fit failed to converge for edge {edge_index}")

        params_ref = fit_ref['params']

        # ---- Step 2: sequential row-by-row raw trace (warm-started) ----
        n_rows = column_slice.shape[0]
        slice_width = column_slice.shape[1]
        contour = []
        flags = []
        x_prev = peak_full  # full-image coordinate, running warm-start

        for y in range(n_rows):
            imin = int(round(x_prev - window_half_width))
            imax = int(round(x_prev + window_half_width))
            # clip to what this worker actually has available
            imin_c = max(col_offset, imin)
            imax_c = min(col_offset + slice_width, imax)
            local_imin = imin_c - col_offset
            local_imax = imax_c - col_offset

            y_row = column_slice[y, local_imin:local_imax].astype(np.float64)
            x_row = np.arange(imin_c, imax_c, dtype=np.float64)  # full-image coords

            if np.any(np.isnan(y_row)) or y_row.size < 4:
                flags.append({'y': y, 'reason': 'insufficient/NaN data in window'})
                contour.append((y, x_prev))
                continue

            params_init_row = dict(params_ref)
            params_init_row['x0'] = x_prev  # warm-start from previous row

            fit_result = fit_double_gaussian(
                x_row, y_row, params_init_row,
                free_params={'x0', 'ampli'},
            )

            x_edge = fit_result['params']['x0']

            reasons = []
            if not fit_result['success']:
                reasons.append('did not converge')
            if fit_result['hit_bounds']:
                reasons.append(f"hit bounds: {fit_result['hit_bounds']}")
            if cost_thresh is not None and fit_result['cost'] > cost_thresh:
                reasons.append(f"high cost: {fit_result['cost']:.3g}")
            if reasons:
                flags.append({'y': y, 'reason': '; '.join(reasons)})

            contour.append((y, x_edge))
            x_prev = x_edge  # sequential warm-start dependency, kept in-process

        # ---- Step 3: post-process (de-mean), folded into this task ----
        ys = np.array([c[0] for c in contour])
        xs = np.array([c[1] for c in contour])
        xs_demeaned = xs - np.mean(xs)
        contour_demeaned = list(zip(ys.tolist(), xs_demeaned.tolist()))

        # small numeric-only sample for optional QC plotting later,
        # NEVER a matplotlib figure -- plotting happens in main process
        sample_rows = np.linspace(0, n_rows - 1, min(3, n_rows)).astype(int)
        plot_samples = [
            {'y': int(y), 'x_row': column_slice[y].tolist(),
             'col_offset': col_offset, 'params_ref': params_ref}
            for y in sample_rows
        ]

        return {
            'edge_index': edge_index,
            'status': 'ok',
            'contour_raw': contour,
            'contour_demeaned': contour_demeaned,
            'flags': flags,
            'params_ref': params_ref,
            'plot_samples': plot_samples,
        }

    except Exception as exc:
        # isolate failure to this edge only -- don't kill the whole pool run
        return {
            'edge_index': edge_index,
            'status': 'failed',
            'error': f"{type(exc).__name__}: {exc}",
        }


def extract_all_contours_parallel(path, axis: int = 0,
                                   prominence: float = None,
                                   window_frac: float = 0.4,
                                   missing_thresh_frac: float = 0.10,
                                   default_sigma: float = 5.0,
                                   x0_delta: float = None,
                                   cost_thresh: float = None,
                                   pad_factor: float = 3.0,
                                   n_workers: int = None):
    """
    End-to-end parallel pipeline: sequential detect_edges(), then one
    process_single_edge() task per edge dispatched over a process pool,
    reassembled by edge_index tag.

    pad_factor : float
        How much extra column margin (in units of window_half_width)
        to give each worker beyond its Step-1 window, so row-fits have
        room to drift without needing the full image.
    """
    # ---- Step A: sequential pre-work ----
    det = detect_edges(
        path, axis=axis, prominence=prominence,
        window_frac=window_frac, missing_thresh_frac=missing_thresh_frac,
    )
    image = det['image']
    n_edges = len(det['peaks'])

    # ---- Step C: build per-edge payloads (padded column slices) ----
    payloads = []
    for i, (peak, window) in enumerate(zip(det['peaks'], det['windows'])):
        half = (window[1] - window[0]) // 2
        pad = int(half * pad_factor)
        col_min = max(0, window[0] - pad)
        col_max = min(image.shape[1], window[1] + pad)
        column_slice = image[:, col_min:col_max]
        payloads.append(dict(
            column_slice=column_slice, col_offset=col_min,
            peak_full=float(peak), edge_index=i,
            window_half_width=half, orig_window=(int(window[0]), int(window[1])),
            default_sigma=default_sigma,
            x0_delta=x0_delta, cost_thresh=cost_thresh,
        ))

    # ---- Step D: dispatch across a process pool, tagged by edge_index ----
    pool_size = min(n_edges, n_workers or n_edges)
    results_by_index = {}
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        futures = {
            executor.submit(process_single_edge, **payload): payload['edge_index']
            for payload in payloads
        }
        for future in as_completed(futures):
            result = future.result()
            results_by_index[result['edge_index']] = result  # tag, not order

    # ---- Step E: reassemble + surface failures, no plotting here (deferred) ----
    contours = {}
    contours_demeaned = {}
    flags = {}
    failed = {}
    for i in range(n_edges):
        r = results_by_index[i]
        if r['status'] == 'ok':
            contours[i] = r['contour_raw']
            contours_demeaned[i] = r['contour_demeaned']
            flags[i] = r['flags']
        else:
            failed[i] = r['error']

    return {
        'image': image,
        'profile': det['profile'],
        'peaks': det['peaks'],
        'missing_peaks': det['missing_peaks'],
        'windows': det['windows'],
        'contours': contours,
        'contours_demeaned': contours_demeaned,
        'flags': flags,
        'failed_edges': failed,
        'raw_results': results_by_index,  # includes plot_samples etc.
    }
