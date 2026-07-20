#Load the buld mesh and findout it's dimensions.
#users can choose the number of lines and length of line
# according to that dimension generate primary electrons that are inside this frame and save it in the given folder
# run with nebula input electrons, material files and output

import numpy as np
import os
import subprocess
import time
import tifffile

electron_dtype_global = np.dtype([
('x',  '=f'), ('y',  '=f'), ('z',  '=f'), # Position
('dx', '=f'), ('dy', '=f'), ('dz', '=f'), # Direction
('E',  '=f'),                             # Energy	
('px', '=i'), ('py', '=i')])              # Pixel index


#########################Generate Primary Electrons################################################
def detect_pitch_from_tri(fname, interface_mat=(0, -123), z_tol=1e-6, x_tol=0.6):
    """
    Auto-detect grating pitch from a NEBULA .tri mesh file.

    Parameters
    ----------
    fname : str
        Path to .tri file (output of build_cdsem_mesh / export_tri).
    interface_mat : tuple (mat_in, mat_out)
        Material pair identifying real interface geometry. Boundary/mirror
        faces (xmin, xmax, ymin, ymax, H_bottom, H_top) carry other tags
        and are excluded.
    z_tol : float
        Tolerance for treating a triangle as "flat" (max(z)-min(z) < z_tol).
        Rough sidewall triangles span both z=0 and z=height and are excluded.
    x_tol : float
        Clustering tolerance (nm) for merging near-duplicate x edge values.

    Returns
    -------
    pitch : float
        Median pitch estimated from same-type edge spacing.
    info : dict
        Diagnostic info: rising_edges, falling_edges (cluster centers),
        n_interface_rows, n_flat_triangles.
    """
    mat_in_target, mat_out_target = interface_mat

    verts_by_z = []  # list of (x_min_vert, x_max_vert_flag, z_level) not needed; do per-triangle

    rows_x = []
    rows_z = []

    with open(fname, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            mat_in, mat_out = int(float(parts[0])), int(float(parts[1]))
            if mat_in != mat_in_target or mat_out != mat_out_target:
                continue
            coords = np.array(parts[2:11], dtype=float).reshape(3, 3)  # 3 verts x (x,y,z)
            x = coords[:, 0]
            z = coords[:, 2]
            rows_x.append(x)
            rows_z.append(z)

    if not rows_x:
        raise ValueError(
            f"No rows matched interface_mat={interface_mat}. Check the .tri "
            "file's material tags before filtering."
        )

    n_interface_rows = len(rows_x)

    # ------------------------------------------------------------
    # keep only flat triangles (all 3 verts at ~same z), collect
    # per-vertex (x, z_level) pairs
    # ------------------------------------------------------------
    n_flat_triangles = sum(1 for x, z in zip(rows_x, rows_z) if (z.max() - z.min()) < z_tol)

    if n_flat_triangles == 0:
        raise ValueError(
            "No flat triangles found (all triangles span both z levels). "
            "Check z_tol or confirm the mesh actually contains flat "
            "top/bottom segments."
        )

    xs_all = []
    zlevels_all = []
    for x, z in zip(rows_x, rows_z):
        if (z.max() - z.min()) < z_tol:
            z_level = z.mean()
            xs_all.extend(x.tolist())
            zlevels_all.extend([z_level] * len(x))

    xs_all = np.asarray(xs_all)
    zlevels_all = np.asarray(zlevels_all)

    order = np.argsort(xs_all)
    xs_sorted = xs_all[order]
    z_sorted = zlevels_all[order]

    # ------------------------------------------------------------
    # cluster x-values within x_tol into edge groups
    # ------------------------------------------------------------
    edge_x = []
    edge_z = []
    cluster_x = [xs_sorted[0]]
    cluster_z = [z_sorted[0]]

    for xi, zi in zip(xs_sorted[1:], z_sorted[1:]):
        if xi - cluster_x[-1] <= x_tol:
            cluster_x.append(xi)
            cluster_z.append(zi)
        else:
            edge_x.append(np.mean(cluster_x))
            # dominant z-level in this cluster (rounds noisy near-0 / near-height values)
            edge_z.append(np.median(cluster_z))
            cluster_x = [xi]
            cluster_z = [zi]

    edge_x.append(np.mean(cluster_x))
    edge_z.append(np.median(cluster_z))

    edge_x = np.asarray(edge_x)
    edge_z = np.asarray(edge_z)

    # ------------------------------------------------------------
    # classify each edge by which z-level borders it on the right
    # (rising edge: z goes low->high moving in +x i.e. space->line;
    #  falling edge: z goes high->low i.e. line->space)
    # Determined by comparing each edge's z-level to the next edge's z-level.
    # ------------------------------------------------------------
    order2 = np.argsort(edge_x)
    edge_x = edge_x[order2]
    edge_z = edge_z[order2]

    z_height = edge_z.max()
    z_ground = edge_z.min()
    mid = (z_height + z_ground) / 2.0

    is_top = edge_z > mid  # this edge sits at top-of-line level
    is_bottom = ~is_top    # this edge sits at ground level

    rising_edges = edge_x[is_bottom]   # ground-level edge x's (space->line transitions start here)
    falling_edges = edge_x[is_top]     # top-level edge x's (line->space transitions)

    def median_diff(vals):
        vals = np.sort(vals)
        if len(vals) < 2:
            return np.nan
        return float(np.median(np.diff(vals)))

    pitch_candidates = []
    d_rising = median_diff(rising_edges)
    d_falling = median_diff(falling_edges)
    if not np.isnan(d_rising):
        pitch_candidates.append(d_rising)
    if not np.isnan(d_falling):
        pitch_candidates.append(d_falling)

    if not pitch_candidates:
        raise ValueError(
            "Could not compute pitch: fewer than 2 edges detected per class. "
            "Check x_tol/z_tol or confirm mesh has multiple lines."
        )

    pitch = float(np.median(pitch_candidates))

    info = {
        "rising_edges": rising_edges,
        "falling_edges": falling_edges,
        "n_interface_rows": n_interface_rows,
        "n_flat_triangles": n_flat_triangles,
        "pitch_from_rising": d_rising,
        "pitch_from_falling": d_falling,
    }

    return pitch, info

 
def get_beam_grid_from_tri(fname_out, pixel_size, n_lines, length_of_lines,
                            interface_mat=(0, -123)):
    """
    Determine electron-beam pixel grid (x_px_out, y_px_out) and the
    corresponding FOV / offset, anchored at the mesh's corner
    (x_min, y_min), from a NEBULA .tri file.
 
    Parameters
    ----------
    fname_out : str
        Path to .tri mesh file (output of build_cdsem_mesh).
    pixel_size : float
        Physical pixel size, nm/pixel.
    n_lines : int
        Number of grating lines to cover in x.
    length_of_lines : float
        Extent to cover in y (nm), i.e. FOV_y directly.
    interface_mat : tuple
        (mat_in, mat_out) tag identifying real interface geometry
        (same convention as detect_pitch_from_tri).
 
    Returns
    -------
    dict with keys:
        x_px_out, y_px_out : int
            Pixel counts for the electron grid.
        FOV_x, FOV_y : float
            Corrected FOV (nm), after pixel rounding.
        x_offset, y_offset : float
            Corner anchor position (mesh x_min, y_min) — start the
            electron/pixel grid here.
        pitch : float
            Detected grating pitch (nm).
    """
    pitch, _ = detect_pitch_from_tri(fname_out, interface_mat=interface_mat)
 
    # get mesh bbox from the same interface-only rows used for pitch detection
    mat_in_target, mat_out_target = interface_mat
    xs, ys, zs = [], [], []
    with open(fname_out, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            mat_in, mat_out = int(float(parts[0])), int(float(parts[1]))
            if mat_in != mat_in_target or mat_out != mat_out_target:
                continue
            coords = np.array(parts[2:11], dtype=float).reshape(3, 3)
            xs.extend(coords[:, 0].tolist())
            ys.extend(coords[:, 1].tolist())
            zs.extend(coords[:, 2].tolist())
 
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
 
    FOV_x_requested = n_lines * 2 * pitch
    FOV_y_requested = length_of_lines
 
    if FOV_x_requested > (x_max - x_min):
        raise ValueError(
            f"Requested FOV_x ({FOV_x_requested:.2f} nm from n_lines={n_lines} "
            f"x pitch={pitch:.2f}) exceeds mesh x-extent "
            f"({x_max - x_min:.2f} nm)."
        )
    if FOV_y_requested > (y_max - y_min):
        raise ValueError(
            f"Requested FOV_y ({FOV_y_requested:.2f} nm) exceeds mesh "
            f"y-extent ({y_max - y_min:.2f} nm)."
        )
 
    x_px_out = int(round(FOV_x_requested / pixel_size))
    y_px_out = int(round(FOV_y_requested / pixel_size))
 
    FOV_x = x_px_out * pixel_size
    FOV_y = y_px_out * pixel_size
 
    return {
        "x_px_out": x_px_out,
        "y_px_out": y_px_out,
        "FOV_x": FOV_x,
        "FOV_y": FOV_y,
        "x_offset": x_min,
        "y_offset": y_min,
        "z_beam": z_max+5,
        "pitch": pitch,
    }
 
def generate_primary(z, xpx, ypx, energy, epx, sigma, poisson, file_name, electron_dtype):
    """writes primary electrons to file_name."""

    with open(file_name, 'wb') as file:
        for i, xmid in enumerate(xpx):
            for j, ymid in enumerate(ypx):
                N_elec = np.random.poisson(epx) if poisson else epx
 
                buffer = np.empty(N_elec, dtype=electron_dtype)
 
                buffer['x'] = np.random.normal(xmid, sigma, N_elec)
                buffer['y'] = np.random.normal(ymid, sigma, N_elec)
                buffer['z'] = z
                buffer['dx'] = 0
                buffer['dy'] = 0
                buffer['dz'] = -1
                buffer['E'] = energy
                buffer['px'] = i
                buffer['py'] = j
 
                buffer.tofile(file)
 
 
def generate_primary_from_tri(fname_out, pixel_size, n_lines, length_of_lines,
                               energy, epx, sigma, poisson, file_name,
                               interface_mat=(0, -123)):
    """
    Derive beam grid (x/y pixel counts, offsets, z) from a .tri mesh, then
    generate primary electrons.
 
    Parameters
    ----------
    fname_out : str
        Path to .tri mesh file.
    pixel_size : float
        nm/pixel.
    n_lines : int
        Number of grating lines to cover in x.
    length_of_lines : float
        FOV_y, nm.
    energy, epx, sigma, poisson, file_name, electron_dtype :
        Passed straight through to generate_primary — same meaning as before.
    interface_mat : tuple
        (mat_in, mat_out) identifying real interface geometry.
 
    Returns
    -------
    grid_info : dict
        The dict returned by get_beam_grid_from_tri, for logging/reuse
        (x_px_out, y_px_out, FOV_x, FOV_y, x_offset, y_offset, z_beam, pitch).
    """
    grid_info = get_beam_grid_from_tri(
        fname_out, pixel_size, n_lines, length_of_lines, interface_mat=interface_mat
    )
 
    x_px_out = grid_info["x_px_out"]
    y_px_out = grid_info["y_px_out"]
    x_offset = grid_info["x_offset"]
    y_offset = grid_info["y_offset"]
    z_beam = grid_info["z_beam"]
 
    # pixel-center coordinate arrays, anchored at (x_offset, y_offset),
    # stepping by pixel_size, centered within each pixel (+0.5)
    xpx = x_offset + (np.arange(x_px_out) + 0.5) * pixel_size
    ypx = y_offset + (np.arange(y_px_out) + 0.5) * pixel_size

    generate_primary(
        z=z_beam, xpx=xpx, ypx=ypx,
        energy=energy, epx=epx, sigma=sigma, poisson=poisson,
        file_name=file_name, electron_dtype=electron_dtype_global,
    )
 
    return grid_info



#####################################################################################################


#######################################Run Nebula########################
# load nebula path, mesh file, primary electrons file, material file and the folder to save nebula
# execute nebula in the following manner:  os.system(f"{nebula_path} {mesh_path} {primary_electrons_path} {Mat_path} > {detected_path}") but maybe with a better code than this
# generate a tiff file using the histogram function

def generate_tiff(det_path, save_path):
    """
    Read NEBULA-detected electrons from det_path, build a 2D pixel-count
    histogram, and save as an 8-bit normalized TIFF at save_path.
    """
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"File {det_path} cannot be found")
 
    outdir = os.path.dirname(save_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
 
    data = np.fromfile(det_path, dtype=electron_dtype_global)
    print(f"Number of electrons detected: {len(data)}")
 
    xmin = data['px'].min()
    xmax = data['px'].max()
    ymin = data['py'].min()
    ymax = data['py'].max()
 
    H, xedges, yedges = np.histogram2d(
        data['px'], data['py'],
        bins=[
            np.linspace(xmin - 0.5, xmax + 0.5, xmax - xmin + 2),
            np.linspace(ymin - 0.5, ymax + 0.5, ymax - ymin + 2)
        ]
    )
 
    img = H.T
    img_to_save = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
 
    tifffile.imwrite(save_path, img_to_save)
    print("Saved TIFF to:", save_path)
    print("Output exists:", os.path.exists(save_path), "size (bytes):", os.path.getsize(save_path))
 
    return save_path
 
def run_nebula_pipeline(mesh_path, primary_electrons_path, mat_path,
                         detected_path, save_tiff_path,
                         nebula_path):
    """
    Run one NEBULA simulation end-to-end: validate inputs, execute NEBULA,
    generate the tiff image.
 
    Parameters
    ----------
    mesh_path, primary_electrons_path, mat_path : str
        Required input files. Existence is checked before running anything.
    detected_path : str
        Where NEBULA's stdout (.det) is written.
    save_tiff_path : str
        Where the final tiff is written.
    nebula_path : str
        Path to the nebula executable.
    Returns
    -------
    save_tiff_path : str
        Same path passed in, returned for convenience/chaining.
    """
    # ---- preflight: fail fast on missing inputs ----
    for label, path in [("mesh_path", mesh_path),
                         ("primary_electrons_path", primary_electrons_path),
                         ("mat_path", mat_path),
                         ("nebula_path", nebula_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
 
    # ---- run NEBULA ----
    os.makedirs(os.path.dirname(detected_path), exist_ok=True)
 
    t0 = time.time()
    with open(detected_path, "wb") as out_f:
        result = subprocess.run(
            [nebula_path, mesh_path, primary_electrons_path, mat_path],
            stdout=out_f, stderr=None,
        )
    elapsed = time.time() - t0
 
    if result.returncode != 0:
        raise RuntimeError(
            f"NEBULA failed (exit {result.returncode}) after {elapsed:.1f}s.\n"
            f"stderr: {result.stderr.decode(errors='replace')}"
        )
    print(f"NEBULA finished in {elapsed:.1f}s -> {detected_path}")
 
    # ---- generate tiff ----
    os.makedirs(os.path.dirname(save_tiff_path), exist_ok=True)
    generate_tiff(detected_path, save_tiff_path)
    print(f"tiff written -> {save_tiff_path}")
 
    return save_tiff_path
 

