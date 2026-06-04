"""
MHD post-processing utilities.

Compute derived GRMHD quantities (comoving magnetic field, Poynting flux,
magnetization, etc.) from primitive variables stored in grace output.

All public functions accept a :class:`grace_xmf_reader` (or any object that
exposes the same ``get_var(varname, time)`` interface) and a target time.
Coordinates and data arrays are returned in the same format as
``grace_xmf_reader.get_var``.

Variable name defaults follow the GRACE output convention.  Override via the
``varnames`` dict argument when the output uses different labels.
"""

import numpy as np
import h5py


# ── default variable-name mapping ──────────────────────────────────────────
DEFAULT_VARNAMES = {
    "rho":   "rho",        # rest-mass density                      (scalar)
    "press": "press",      # gas pressure                           (scalar)
    "eps":   "eps",        # specific internal energy                (scalar)
    "zvec":  "zvec",       # W_L * v^i  (Lorentz-boosted velocity)  (vector)
    "Bvec":  "Bvec",       # Eulerian magnetic field B^i            (vector)
    "alpha": "alpha",      # lapse                                  (scalar)
    "beta":  "beta",       # shift  beta^i                          (vector)
    "W_conf": "conf_fact",         # conformal factor W = Psi^{-2}          (scalar)
    "gt":    "gamma_tilde",         # conformal 3-metric tilde{gamma}_{ij}   (sym-tensor, 6-comp)
}


def _vn(varnames, key):
    """Look up a variable name, falling back to the default."""
    if varnames is not None and key in varnames:
        return varnames[key]
    return DEFAULT_VARNAMES[key]


# ── metric helpers ─────────────────────────────────────────────────────────

def _get_metric_diag(reader, time, varnames=None):
    """Return (coords, gamma_ij) as a (N, 3, 3) array from conformal metric.

    gamma_{ij} = W^{-2} * tilde{gamma}_{ij}

    The conformal metric components are stored as separate scalars:
    ``gamma_tilde[0,0]``, ``gamma_tilde[0,1]``, ..., ``gamma_tilde[2,2]``
    (6 independent components of the symmetric tensor).
    """
    coords, W = reader.get_var(_vn(varnames, "W_conf"), time)
    psi4 = W ** (-2)  # W = Psi^{-2}  =>  Psi^4 = W^{-2}

    gt_base = _vn(varnames, "gt")
    # component labels in storage order
    gt_comps = ["[0,0]", "[0,1]", "[0,2]", "[1,1]", "[1,2]", "[2,2]"]
    gt_first = gt_base + gt_comps[0]

    avail = reader.available_variables("cell")
    if gt_first in avail:
        gt = {}
        for c in gt_comps:
            _, gt[c] = reader.get_var(gt_base + c, time)
        gamma = np.zeros((len(psi4), 3, 3))
        gamma[:, 0, 0] = gt["[0,0]"]
        gamma[:, 0, 1] = gamma[:, 1, 0] = gt["[0,1]"]
        gamma[:, 0, 2] = gamma[:, 2, 0] = gt["[0,2]"]
        gamma[:, 1, 1] = gt["[1,1]"]
        gamma[:, 1, 2] = gamma[:, 2, 1] = gt["[1,2]"]
        gamma[:, 2, 2] = gt["[2,2]"]
        gamma *= psi4[:, None, None]
    else:
        # assume conformally flat: tilde{gamma} = delta
        gamma = np.zeros((len(psi4), 3, 3))
        gamma[:, 0, 0] = gamma[:, 1, 1] = gamma[:, 2, 2] = psi4
    return coords, gamma


def _lower_vector(gamma, v):
    """Lower a contravariant vector: v_i = gamma_{ij} v^j.

    Parameters
    ----------
    gamma : (N, 3, 3) array
    v     : (N, 3) array

    Returns
    -------
    v_lower : (N, 3) array
    """
    return np.einsum("nij,nj->ni", gamma, v)


def _dot(gamma, a, b):
    """Compute gamma_{ij} a^i b^j."""
    return np.einsum("nij,ni,nj->n", gamma, a, b)


# ── Lorentz factor ─────────────────────────────────────────────────────────

def lorentz_factor(reader, time, varnames=None):
    """Compute the Lorentz factor W_L = sqrt(1 + gamma_{ij} z^i z^j).

    GRACE stores z^i = W_L v^i, so the Lorentz factor is recovered
    directly from z^i without division.

    Returns
    -------
    coords : (N, 3) array
    W_L    : (N,) array
    """
    coords, gamma = _get_metric_diag(reader, time, varnames)
    _, zvec = reader.get_var(_vn(varnames, "zvec"), time)
    z2 = _dot(gamma, zvec, zvec)
    return coords, np.sqrt(1.0 + z2)


# ── comoving magnetic field ───────────────────────────────────────────────

def comoving_b_squared(reader, time, varnames=None):
    """Compute b^2 = b_mu b^mu (twice the magnetic pressure).

    With z^i = W_L v^i the identity becomes
    b^2 = B^2/W_L^2 + (B_k z^k)^2/W_L^4
    where B^2 = gamma_{ij} B^i B^j and W_L = sqrt(1 + gamma_{ij} z^i z^j).

    Returns
    -------
    coords : (N, 3) array
    b2     : (N,) array
    """
    coords, gamma = _get_metric_diag(reader, time, varnames)
    _, zvec = reader.get_var(_vn(varnames, "zvec"), time)
    _, Bvec = reader.get_var(_vn(varnames, "Bvec"), time)

    W_L2 = 1.0 + _dot(gamma, zvec, zvec)

    B2 = _dot(gamma, Bvec, Bvec)
    Bz = np.sum(_lower_vector(gamma, Bvec) * zvec, axis=1)

    b2 = B2 / W_L2 + Bz ** 2 / W_L2
    return coords, b2


def comoving_b(reader, time, varnames=None):
    """Compute the comoving magnetic field four-vector components.

    With z^i = W_L v^i the expressions simplify to
        b^0 = B_k z^k / alpha
        b^i = (B^i + (B_k z^k) z^i) / (alpha W_L)

    Returns
    -------
    coords : (N, 3) array
    b0     : (N,) array          — time component b^0
    bi     : (N, 3) array        — spatial contravariant b^i
    """
    coords, gamma = _get_metric_diag(reader, time, varnames)
    _, zvec = reader.get_var(_vn(varnames, "zvec"), time)
    _, Bvec = reader.get_var(_vn(varnames, "Bvec"), time)
    _, alpha = reader.get_var(_vn(varnames, "alpha"), time)

    W_L = np.sqrt(1.0 + _dot(gamma, zvec, zvec))

    Bz = np.sum(_lower_vector(gamma, Bvec) * zvec, axis=1)

    b0 = Bz / alpha
    bi = (Bvec + Bz[:, None] * zvec) / (alpha[:, None] * W_L[:, None])

    return coords, b0, bi


# ── magnetic pressure ─────────────────────────────────────────────────────

def magnetic_pressure(reader, time, varnames=None):
    """Compute the magnetic pressure P_mag = b^2 / 2.

    Returns
    -------
    coords : (N, 3) array
    Pmag   : (N,) array
    """
    coords, b2 = comoving_b_squared(reader, time, varnames)
    return coords, b2 / 2.0


# ── plasma beta ───────────────────────────────────────────────────────────

def plasma_beta(reader, time, varnames=None):
    """Compute the plasma beta = 2 P_gas / b^2.

    Returns
    -------
    coords : (N, 3) array
    beta_p : (N,) array
    """
    coords, b2 = comoving_b_squared(reader, time, varnames)
    _, press = reader.get_var(_vn(varnames, "press"), time)
    return coords, 2.0 * press / b2


# ── magnetization (sigma) ────────────────────────────────────────────────

def magnetization(reader, time, varnames=None):
    """Compute the magnetization sigma = b^2 / (rho h).

    The specific enthalpy is h = 1 + eps + P / rho.

    Returns
    -------
    coords : (N, 3) array
    sigma  : (N,) array
    """
    coords, b2 = comoving_b_squared(reader, time, varnames)
    _, rho = reader.get_var(_vn(varnames, "rho"), time)
    _, press = reader.get_var(_vn(varnames, "press"), time)
    _, eps = reader.get_var(_vn(varnames, "eps"), time)
    h = 1.0 + eps + press / rho
    return coords, b2 / (rho * h)


# ── Poynting flux ─────────────────────────────────────────────────────────

def poynting_flux(reader, time, varnames=None):
    """Compute the Poynting flux vector (EM energy flux in the Eulerian frame).

    The electromagnetic energy flux measured by the Eulerian observer is

        S^i_EM = alpha * (b^2 u^0 v^i - b^0 b^i)

    where u^0 = W_L / alpha and v^i = z^i / W_L.  This gives the energy per
    unit coordinate area per unit coordinate time carried by the EM field.

    Returns
    -------
    coords : (N, 3) array
    S_EM   : (N, 3) array   — contravariant Poynting flux vector
    """
    coords, gamma = _get_metric_diag(reader, time, varnames)
    _, zvec = reader.get_var(_vn(varnames, "zvec"), time)
    _, Bvec = reader.get_var(_vn(varnames, "Bvec"), time)
    _, alpha = reader.get_var(_vn(varnames, "alpha"), time)

    W_L2 = 1.0 + _dot(gamma, zvec, zvec)
    W_L = np.sqrt(W_L2)

    Bz = np.sum(_lower_vector(gamma, Bvec) * zvec, axis=1)

    b0 = Bz / alpha
    bi = (Bvec + Bz[:, None] * zvec) / (alpha[:, None] * W_L[:, None])

    B2 = _dot(gamma, Bvec, Bvec)
    b2_vals = B2 / W_L2 + Bz ** 2 / W_L2 ** 2

    # u^0 = W_L / alpha,  v^i = z^i / W_L
    # S^i = alpha (b^2 u^0 v^i - b^0 b^i) = b^2 z^i / W_L - alpha b^0 b^i
    # but keep the explicit form for clarity
    u0 = W_L / alpha
    vi = zvec / W_L[:, None]
    S_EM = alpha[:, None] * (b2_vals[:, None] * u0[:, None] * vi - b0[:, None] * bi)

    return coords, S_EM


def poynting_luminosity(reader, time, radius, varnames=None,
                        center=np.array([0.0, 0.0, 0.0])):
    """Integrate the Poynting flux over a coordinate sphere.

    Extracts data on a spherical surface at ``radius`` and computes
    L_EM = integral of sqrt{gamma} S^i n_i dOmega, where n_i is the
    outward unit normal.

    Parameters
    ----------
    reader : grace_xmf_reader
    time   : float
    radius : float
    varnames : dict, optional
    center : (3,) array, optional

    Returns
    -------
    L_EM : float   — integrated Poynting luminosity
    """
    coords_vol, S_EM = poynting_flux(reader, time, varnames)
    _, gamma = _get_metric_diag(reader, time, varnames)

    # Select cells near the sphere
    r_vec = coords_vol - center[None, :]
    r = np.linalg.norm(r_vec, axis=1)
    n_hat = r_vec / r[:, None]

    # Radial component: S^i n_i (with flat-space n_i for coordinate sphere)
    S_r = np.sum(S_EM * n_hat, axis=1)

    # For a proper integral we'd need the surface element.
    # This is a rough volume-shell estimate: pick cells in a thin shell
    dr = np.median(np.diff(np.sort(np.unique(r))))  # approximate grid spacing
    shell = np.abs(r - radius) < 1.5 * dr

    if not np.any(shell):
        raise ValueError(f"No grid cells found near radius {radius}. "
                         f"Grid radii range: [{r.min():.3f}, {r.max():.3f}]")

    # Approximate integral: sum S_r * r^2 * dV_shell / dr
    # where dV_shell ~ 4 pi r^2 dr for uniform shells
    # Here we just return sum(S_r * cell_volume) over the shell
    # as a first approximation. For accurate integrals use
    # get_var_spherical_slice and proper quadrature.
    det_gamma = np.linalg.det(gamma[shell])
    sqrt_gamma = np.sqrt(np.maximum(det_gamma, 0.0))
    L_EM = np.sum(S_r[shell] * sqrt_gamma * dr)
    return L_EM


# ── array-backed reader (for derived-quantity computation on resampled data)

class _array_reader:
    """Minimal reader interface backed by pre-computed numpy arrays.

    Allows the existing ``mhd_utils`` functions (which call
    ``reader.get_var(name, time)`` and ``reader.available_variables()``)
    to operate on data that has already been interpolated onto a uniform grid.
    """

    def __init__(self, coords, data_dict):
        self._coords = coords
        self._data = data_dict

    def get_var(self, varname, time=None):
        return self._coords, self._data[varname]

    def available_variables(self, vtype="cell"):
        return list(self._data.keys())


# ── registry of derived quantities ────────────────────────────────────────

# Pointwise quantities — computed from field values alone, no derivatives.
DERIVED_QUANTITIES = {
    "lorentz_factor":    lorentz_factor,
    "b_squared":         comoving_b_squared,
    "magnetic_pressure": magnetic_pressure,
    "plasma_beta":       plasma_beta,
    "magnetization":     magnetization,
    "poynting_flux":     poynting_flux,
}


# ── grid-level derived quantities (need spatial derivatives) ──────────────

def _compute_vorticity_on_grid(data, spacing, varnames=None):
    r"""Compute the vorticity on a uniform grid.

    Returns the curl of the transport velocity,

    .. math::
        \omega_i = \varepsilon_{ijk}\,\partial_j v^k ,
        \qquad v^k = z^k / W_L

    using second-order central finite differences (``np.gradient``).

    Parameters
    ----------
    data : dict
        Gridded arrays as produced by the probing step in ``export_uniform``.
        Must contain the zvec components and the metric variables needed
        to compute the Lorentz factor.
    spacing : tuple of float
        ``(dx, dy)`` for 2-D or ``(dx, dy, dz)`` for 3-D grids.
    varnames : dict, optional
        Variable name overrides.

    Returns
    -------
    result : dict
        ``{"vorticity": array}`` shaped ``(nx, ny, 3)`` or
        ``(nx, ny, nz, 3)``.
    """
    zvec_name = _vn(varnames, "zvec")
    W_conf_name = _vn(varnames, "W_conf")
    gt_base = _vn(varnames, "gt")
    gt_comps = ["[0,0]", "[0,1]", "[0,2]", "[1,1]", "[1,2]", "[2,2]"]

    zvec = data[zvec_name]                # (nx, ny, [nz,] 3)
    W_conf = data[W_conf_name]            # (nx, ny, [nz])
    psi4 = W_conf ** (-2)

    is_2d = zvec.ndim == 3                # (nx, ny, 3)
    spatial_dims = 2 if is_2d else 3

    # reconstruct gamma_{ij} on the grid to get W_L and lower z
    gt_first = gt_base + gt_comps[0]
    if gt_first in data:
        gt = {c: data[gt_base + c] for c in gt_comps}
        # gamma_{ij} z^j for each i — build inline to avoid (N,3,3) allocation
        # gamma z = psi4 * (gt . z)
        def gamma_dot_z():
            gz = np.zeros_like(zvec)
            gz[..., 0] = gt["[0,0]"] * zvec[..., 0] + gt["[0,1]"] * zvec[..., 1] + gt["[0,2]"] * zvec[..., 2]
            gz[..., 1] = gt["[0,1]"] * zvec[..., 0] + gt["[1,1]"] * zvec[..., 1] + gt["[1,2]"] * zvec[..., 2]
            gz[..., 2] = gt["[0,2]"] * zvec[..., 0] + gt["[1,2]"] * zvec[..., 1] + gt["[2,2]"] * zvec[..., 2]
            gz *= psi4[..., None]
            return gz
        gz = gamma_dot_z()
    else:
        gz = psi4[..., None] * zvec

    # W_L = sqrt(1 + gamma_{ij} z^i z^j)
    z2 = np.sum(gz * zvec, axis=-1)
    W_L = np.sqrt(1.0 + z2)

    # v^i = z^i / W_L
    vel = zvec / W_L[..., None]

    # curl(v) via finite differences
    # axes 0,1[,2] correspond to x,y[,z]
    if is_2d:
        dx, dy = spacing
        dvx_dy = np.gradient(vel[..., 0], dy, axis=1)
        dvy_dx = np.gradient(vel[..., 1], dx, axis=0)
        dvx_dx = np.gradient(vel[..., 0], dx, axis=0)  # unused but for completeness
        dvy_dy = np.gradient(vel[..., 1], dy, axis=1)
        dvz_dx = np.gradient(vel[..., 2], dx, axis=0)
        dvz_dy = np.gradient(vel[..., 2], dy, axis=1)
        omega = np.zeros_like(vel)
        # omega_x =  ∂_y v_z - ∂_z v_y  (∂_z = 0 in 2-D)
        omega[..., 0] = dvz_dy
        # omega_y = -(∂_x v_z - ∂_z v_x) (∂_z = 0 in 2-D)
        omega[..., 1] = -dvz_dx
        # omega_z =  ∂_x v_y - ∂_y v_x
        omega[..., 2] = dvy_dx - dvx_dy
    else:
        dx, dy, dz = spacing
        # ω_x = ∂_y v_z - ∂_z v_y
        omega_x = (np.gradient(vel[..., 2], dy, axis=1)
                   - np.gradient(vel[..., 1], dz, axis=2))
        # ω_y = ∂_z v_x - ∂_x v_z
        omega_y = (np.gradient(vel[..., 0], dz, axis=2)
                   - np.gradient(vel[..., 2], dx, axis=0))
        # ω_z = ∂_x v_y - ∂_y v_x
        omega_z = (np.gradient(vel[..., 1], dx, axis=0)
                   - np.gradient(vel[..., 0], dy, axis=1))
        omega = np.stack([omega_x, omega_y, omega_z], axis=-1)

    return {"vorticity": omega}


# Grid-level quantities — functions take (data_dict, spacing, varnames)
# and return a dict of {name: array} to merge into the export.
GRID_DERIVED_QUANTITIES = {
    "vorticity": _compute_vorticity_on_grid,
}


# ── uniform-grid export ──────────────────────────────────────────────────

def export_uniform(reader, time, outfile, extents, resolution,
                   derived=None, varnames=None, exclude=None):
    """Resample reader data onto a uniform grid and save to HDF5.

    Parameters
    ----------
    reader : grace_xmf_reader
        Volume or plane reader.
    time : float
        Simulation time to extract.
    outfile : str
        Path to the output HDF5 file.
    extents : tuple
        Grid extents.  For 3-D data: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        For 2-D data: ``(xmin, xmax, ymin, ymax)``.
    resolution : int or tuple of int
        Number of grid points in each direction.  A single int is used for all
        directions; a tuple ``(nx, ny)`` or ``(nx, ny, nz)`` sets them
        individually.
    derived : list of str, optional
        Derived MHD quantities to compute and include.  Valid names:
        ``"lorentz_factor"``, ``"b_squared"``, ``"magnetic_pressure"``,
        ``"plasma_beta"``, ``"magnetization"``, ``"poynting_flux"``,
        ``"vorticity"``.
        If ``None`` only the raw simulation variables are exported.
    varnames : dict, optional
        Variable name overrides (passed through to derived-quantity functions).
    exclude : list of str, optional
        Variable names to skip when exporting (e.g. internal bookkeeping
        fields like ``"Quad_ID"``).

    Returns
    -------
    None

    Notes
    -----
    The HDF5 file contains:

    * ``x``, ``y`` (and ``z`` for 3-D) — 1-D coordinate arrays.
    * One dataset per variable, shaped ``(nx, ny)`` or ``(nx, ny, nz)`` for
      scalars and ``(nx, ny, 3)`` or ``(nx, ny, nz, 3)`` for vectors.
    * Attributes ``time``, ``extents``, ``resolution`` on the root group.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    if exclude is None:
        exclude = []

    is_2d = reader.is_data_2D

    # ── parse extents & resolution ────────────────────────────────────────
    if is_2d:
        if len(extents) != 4:
            raise ValueError("2-D data requires extents = (xmin, xmax, ymin, ymax)")
        xmin, xmax, ymin, ymax = extents
        zmin = zmax = 0.0
    else:
        if len(extents) != 6:
            raise ValueError("3-D data requires extents = (xmin, xmax, ymin, ymax, zmin, zmax)")
        xmin, xmax, ymin, ymax, zmin, zmax = extents

    if isinstance(resolution, (int, np.integer)):
        nx = ny = nz = int(resolution)
    else:
        resolution = tuple(resolution)
        if is_2d:
            nx, ny = resolution[:2]
            nz = 1
        else:
            nx, ny, nz = resolution[:3]

    if is_2d:
        nz = 1

    # ── build VTK uniform grid ────────────────────────────────────────────
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz if not is_2d else 1)
    image.SetOrigin(xmin, ymin, zmin)
    dx = (xmax - xmin) / max(nx - 1, 1)
    dy = (ymax - ymin) / max(ny - 1, 1)
    dz = (zmax - zmin) / max(nz - 1, 1) if not is_2d else 1.0
    image.SetSpacing(dx, dy, dz)

    # ── probe all variables at once ───────────────────────────────────────
    reader.set_time(time)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(image)
    probe.SetSourceData(reader.get_output())
    probe.Update()
    probed = probe.GetOutput()

    # ── extract coordinates ───────────────────────────────────────────────
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    if not is_2d:
        z = np.linspace(zmin, zmax, nz)

    coords_flat = vtk_to_numpy(probed.GetPoints().GetData())

    # ── collect probed arrays into a dict ─────────────────────────────────
    point_data = probed.GetPointData()
    data = {}
    grid_shape = (nx, ny) if is_2d else (nx, ny, nz)
    npts = nx * ny * (1 if is_2d else nz)

    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        if name is None or name in exclude:
            continue
        arr = vtk_to_numpy(point_data.GetArray(i))
        if arr.ndim == 1:
            data[name] = arr.reshape(grid_shape, order="F")
        else:
            # vector (npts, ncomp)
            ncomp = arr.shape[1]
            data[name] = arr.reshape((*grid_shape, ncomp), order="F")

    # ── compute derived quantities ────────────────────────────────────────
    if derived:
        all_known = {**DERIVED_QUANTITIES, **GRID_DERIVED_QUANTITIES}
        for dname in derived:
            if dname not in all_known:
                raise ValueError(
                    f"Unknown derived quantity '{dname}'. "
                    f"Available: {sorted(all_known.keys())}")

        # --- pointwise quantities (algebraic, no derivatives) ---
        pointwise = [d for d in derived if d in DERIVED_QUANTITIES]
        if pointwise:
            flat_data = {}
            for i in range(point_data.GetNumberOfArrays()):
                name = point_data.GetArrayName(i)
                if name is None:
                    continue
                flat_data[name] = vtk_to_numpy(point_data.GetArray(i))
            areader = _array_reader(coords_flat, flat_data)

            for dname in pointwise:
                func = DERIVED_QUANTITIES[dname]
                result = func(areader, None, varnames)
                if dname == "poynting_flux":
                    _, vec = result
                    data[dname] = vec.reshape((*grid_shape, 3), order="F")
                else:
                    _, scalar = result
                    data[dname] = scalar.reshape(grid_shape, order="F")

        # --- grid-level quantities (need spatial derivatives) ---
        gridwise = [d for d in derived if d in GRID_DERIVED_QUANTITIES]
        if gridwise:
            spacing = (dx, dy) if is_2d else (dx, dy, dz)
            for dname in gridwise:
                func = GRID_DERIVED_QUANTITIES[dname]
                result = func(data, spacing, varnames)
                data.update(result)

    # ── write HDF5 ────────────────────────────────────────────────────────
    with h5py.File(outfile, "w") as f:
        f.attrs["time"] = time
        f.attrs["extents"] = list(extents)
        f.attrs["resolution"] = list(grid_shape)

        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        if not is_2d:
            f.create_dataset("z", data=z)

        for name, arr in data.items():
            f.create_dataset(name, data=arr)
