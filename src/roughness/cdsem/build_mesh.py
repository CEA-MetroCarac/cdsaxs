import numpy as np
from scipy.spatial import Delaunay
import trimesh
from pathlib import Path

# helper functions################
AXES = ['x', 'y', 'z']
NORMALS = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}

def n_(L, dL):
    """Number of sample points spanning length L at step dL, minimum 2."""
    return int(max(round(L / dL), 2))

def reorient_faces(mesh, desired_normal):
    face_normals = mesh.face_normals
    dot = np.einsum('ij,j->i', face_normals, np.asarray(desired_normal))
    to_flip = np.where(dot < 0)[0]
    new_faces = mesh.faces.copy()
    new_faces[to_flip] = new_faces[to_flip][:, [0, 2, 1]]
    new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces, process=False)
    return new_mesh

def export_tri(meshes, materials, fname_out):
    assert len(materials) == len(meshes)

    with open(fname_out, 'w') as fid:
        for mesh, (mat_in, mat_out) in zip(meshes, materials):
            for face in mesh.faces:
                x, y, z = mesh.vertices[face]
                coords = np.hstack([x, y, z])
                line = f"{mat_in} {mat_out} " + " ".join(map(str, coords)) + "\n"
                fid.write(line)


def _lateral_face(bounds, axis, value):
    """Build a single flat rectangular face at bounds[axis] == value."""
    other_axes = [a for a in AXES if a != axis]

    coords = []
    for a1 in bounds[other_axes[0]]:
        for a2 in bounds[other_axes[1]]:
            point = {axis: value, other_axes[0]: a1, other_axes[1]: a2}
            coords.append([point['x'], point['y'], point['z']])
    coords = np.array(coords)

    faces = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = trimesh.Trimesh(vertices=coords, faces=faces)
    return reorient_faces(mesh, NORMALS[axis])

def rotate_plane(x, y, z, theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    Ry = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])
    coords = np.column_stack((x.flatten(), y.flatten(), z.flatten())) @ Ry.T
    return coords[:, 0], coords[:, 1], coords[:, 2]


#################################################################################################



def build_staircase_profile(cd, pitch, N_lines, height, start_with_space=True):
    """
    Trace the 2D outline (x-z cross-section) of a periodic array of
    rectangular lines, as a sequence of corner points.

    Parameters
    ----------
    cd : float
        Line width (top width of each tooth), nm.
    pitch : float
        Repeat distance between lines, nm. space = pitch - cd.
    N_lines : int
        Number of lines in the array.
    height : float
        Line height, nm.
    start_with_space : bool
        If True, the outline starts on the ground (space first, then line).
        If False, it starts at the top of a line (line first, then space).

    Returns
    -------
    x_prof : list of float
        x-coordinate of each corner, in walk order.
    z_prof : list of float
        z-coordinate of each corner, in walk order (only 0 or `height`).
    rough_indices : list of int
        Indices i such that the segment from corner i to corner i+1 is
        vertical (a sidewall, z changes) rather than flat (z constant).
    """
    space = pitch - cd

    x_prof = [0]
    z_prof = [0 if start_with_space else height]

    for i in range(N_lines):
        base = i * pitch
        if start_with_space:
            # ground -> up -> top -> down, ending back on the ground
            # at base + pitch (start of the next repeat)
            x_prof.extend([base + space, base + space, base + space + cd, base + space + cd])
            z_prof.extend([0, height, height, 0])
        else:
            # top -> down -> ground -> up, ending back at the top
            # at base + pitch
            x_prof.extend([base + cd, base + cd, base + pitch, base + pitch])
            z_prof.extend([height, 0, 0, height])

    rough_indices = [i for i in range(len(z_prof) - 1) if z_prof[i] != z_prof[i + 1]]

    return x_prof, z_prof, rough_indices


def assign_wall_profiles(profiles, rough_indices,
                         mode="per_line",
                         realization_index=0,
                         start_index=0):
    """
    Decide which 1D roughness realization goes on each sidewall.

    Parameters
    ----------
    profiles : ndarray, shape (N_realizations, N)
        Shared roughness profiles.
    rough_indices : list of int
        Wall segment indices returned by build_staircase_profile().
    mode : {"broadcast", "sequential", "per_line"}
        broadcast :
            Every wall receives the same realization.

        sequential :
            Every wall receives a different realization.

        per_line :
            The two sidewalls of each line receive the same realization.
            Consecutive lines receive consecutive realizations.

    realization_index : int
        Used for "broadcast".
    start_index : int
        First realization used for "sequential" and "per_line".

    Returns
    -------
    wall_profiles : dict
        Maps wall segment index -> 1D ndarray.
    """
    n_walls = len(rough_indices)

    if mode == "broadcast":
        row = profiles[realization_index]
        return {idx: row for idx in rough_indices}

    elif mode == "sequential":
        n_available = profiles.shape[0] - start_index
        if n_available < n_walls:
            raise ValueError(
                f"need {n_walls} realizations starting at row {start_index}, "
                f"but only {n_available} available "
                f"(profiles has {profiles.shape[0]} rows)"
            )

        return {
            idx: profiles[start_index + k]
            for k, idx in enumerate(rough_indices)
        }

    elif mode == "per_line":
        if n_walls % 2 != 0:
            raise ValueError(
                "per_line mode expects an even number of sidewalls "
                "(2 walls per line)."
            )

        n_lines = n_walls // 2
        n_available = profiles.shape[0] - start_index

        if n_available < n_lines:
            raise ValueError(
                f"need {n_lines} realizations starting at row {start_index}, "
                f"but only {n_available} available "
                f"(profiles has {profiles.shape[0]} rows)"
            )

        wall_profiles = {}

        for line in range(n_lines):
            row = profiles[start_index + line]

            left_wall = rough_indices[2 * line]
            right_wall = rough_indices[2 * line + 1]

            wall_profiles[left_wall] = row
            wall_profiles[right_wall] = row

        return wall_profiles

    else:
        raise ValueError(
            f"unknown mode: {mode!r}, "
            "expected 'broadcast', 'sequential', or 'per_line'"
        )
    


def extrude_profile_to_mesh(x_prof, z_prof, rough_indices, wall_profiles, Ly, dy):
    """
    Extrude a 2D staircase outline (x_prof, z_prof) along y into a 3D mesh,
    using externally supplied roughness profiles.
    """

    ny = n_(Ly, dy)
    y = np.linspace(0, Ly, ny)

    xs, ys, zs = [], [], []

    def add_contrib(x, y_, z):
        xs.extend(np.asarray(x).flatten().tolist())
        ys.extend(np.asarray(y_).flatten().tolist())
        zs.extend(np.asarray(z).flatten().tolist())

    for i in range(len(x_prof) - 1):

        x0, x1 = x_prof[i], x_prof[i + 1]
        z0, z1 = z_prof[i], z_prof[i + 1]

        # ------------------------------------------------------------
        # Rough sidewalls
        # ------------------------------------------------------------
        if i in rough_indices:

            w = np.asarray(wall_profiles[i])

            if len(w) != ny:
                raise ValueError(
                    f"wall profile length ({len(w)}) does not match ny ({ny})"
                )

            # identical to the old implementation except that
            # synthetic_line(...) is replaced by the imported profile.

            angle = -np.arctan2(z1 - z0, x1 - x0)

            # rotate the roughness profile into the wall orientation
            x, y_rot, z = rotate_plane(
                np.zeros_like(y),
                y,
                w,
                angle,
            )

            # first edge of the wall
            add_contrib(
                x + x0,
                y_rot,
                z + z0,
            )

            # second edge of the wall
            add_contrib(
                x + x1,
                y_rot,
                z + z1,
            )

        # ------------------------------------------------------------
        # Flat segments (unchanged)
        # ------------------------------------------------------------
        else:

            if i - 1 not in rough_indices:
                add_contrib(
                    x0 * np.ones_like(y),
                    y,
                    z0 * np.ones_like(y),
                )

            if i + 1 not in rough_indices:
                add_contrib(
                    x1 * np.ones_like(y),
                    y,
                    z1 * np.ones_like(y),
                )

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)

    # same triangulation as the original implementation
    nx = int(len(xs) / ny)

    x_, y_ = np.meshgrid(
        np.linspace(0, 100, nx),
        np.linspace(0, 100, ny),
        indexing="ij",
    )

    x_ = x_.flatten()
    y_ = y_.flatten()

    tri = Delaunay(np.column_stack((x_, y_)))

    flat_mesh = trimesh.Trimesh(
        vertices=np.column_stack((x_, y_, np.zeros_like(x_))),
        faces=tri.simplices,
    )

    flat_mesh = reorient_faces(flat_mesh, desired_normal=[0, 0, 1])

    mesh = trimesh.Trimesh(
        vertices=np.column_stack((xs, ys, zs)),
        faces=flat_mesh.faces,
    )

    return mesh


def build_nebula_meshes(mesh_interface, H_bottom, H_top):
    """
    Wrap mesh_interface with 6 domain-boundary faces for NEBULA.

    Returns
    -------
    list of trimesh.Trimesh
        [mesh_interface, xmin_face, xmax_face, ymin_face, ymax_face, zmin_face, zmax_face]
    """
    b = mesh_interface.bounds.T  # ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    zmin, zmax = b[2]

    if zmin < H_bottom:
        print("WARNING: mesh_interface has points below H_bottom")
    if zmax > H_top:
        print("WARNING: mesh_interface has points above H_top")

    bounds = {'x': b[0], 'y': b[1], 'z': (H_bottom, H_top)}

    meshes = [mesh_interface]
    for axis in AXES:
        for i in [0, 1]:
            meshes.append(_lateral_face(bounds, axis, bounds[axis][i]))

    return meshes

"""
Single entry point combining Steps 1-4: geometry -> roughness assignment
-> mesh extrusion -> NEBULA-ready mesh export.
"""

def build_cdsem_mesh(
    profiles_path,
    cd, pitch, N_lines, height,
    Ly, dy,
    H_bottom, H_top,
    fname_out,
    materials=None,
    start_with_space=True,
    wall_mode="per_line",
    realization_index=0,
    start_index=0,
):
    """
    Full CD-SEM mesh pipeline: staircase geometry -> roughness assignment
    -> extrusion -> NEBULA domain wrapping -> .tri export.

    Parameters
    ----------
    profiles_path : str
        Path to the shared .npy roughness profiles (N_realizations, N),
        the same file used by the CD-SAXS track for the same seed.
    cd : float
        Line width, nm.
    pitch : float
        Line repeat distance, nm.
    N_lines : int
        Number of lines in the array.
    height : float
        Line height, nm.
    Ly : float
        Extrusion length along y, nm.
    dy : float
        y-sampling step, nm. Must satisfy n_(Ly, dy) == profiles.shape[1],
        i.e. profile length must match the mesh's y-discretization exactly.
    H_bottom, H_top : float
        zmin/zmax of the NEBULA simulation domain.
    fname_out : str
        Output .tri path.
    materials : list of (mat_in, mat_out) tuples, optional
        One pair per mesh face: [interface, xmin, xmax, ymin, ymax, zmin, zmax].
        Defaults to the standard NEBULA convention if not given.
    start_with_space : bool
        Passed to build_staircase_profile.
    wall_mode : {"broadcast", "sequential"}
        Passed to assign_wall_profiles.
    realization_index : int
        Used when wall_mode="broadcast".
    start_index : int
        Used when wall_mode="sequential".

    Returns
    -------
    mesh_interface : trimesh.Trimesh
        The roughened line-array surface mesh (pre-NEBULA-wrapping),
        returned for inspection/plotting even though it's also exported.
    """
    profiles = np.load(profiles_path)

    ny_expected = n_(Ly, dy)
    if profiles.shape[1] != ny_expected:
        raise ValueError(
            f"profile length ({profiles.shape[1]}) does not match ny=n_(Ly,dy)={ny_expected}; "
            "adjust Ly/dy or regenerate profiles with matching N before calling this function."
        )

    x_prof, z_prof, rough_indices = build_staircase_profile(
        cd, pitch, N_lines, height, start_with_space=start_with_space
    )

    wall_profiles = assign_wall_profiles(
        profiles, rough_indices,
        mode=wall_mode, realization_index=realization_index, start_index=start_index
    )

    mesh_interface = extrude_profile_to_mesh(
        x_prof, z_prof, rough_indices, wall_profiles, Ly, dy
    )

    meshes = build_nebula_meshes(mesh_interface, H_bottom, H_top)

    if materials is None:
        materials = [
            (0, -123),      # interface: material/vacuum
            (-122, -122),   # xmin
            (-122, -122),   # xmax
            (-122, -122),   # ymin
            (-122, -122),   # ymax
            (-127, -127),   # H_bottom: terminator
            (-125, -125),   # H_top: SE detector
        ]

    export_tri(meshes, materials, fname_out)

    return mesh_interface
