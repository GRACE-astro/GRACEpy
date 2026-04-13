"""Export and import merged scalar data to/from a single HDF5 file."""

import numpy as np
import re
import h5py

from grace_tools.timeseries_utils import grace_timeseries, grace_timeseries_array
from grace_tools.gw_reader_utils import grace_gw_mode, grace_gw_detector, grace_gw_data
from grace_tools.scalar_reader_utils import grace_scalars_reader
from grace_tools.detector_utils import grace_detector, grace_detector_set


def _write_timeseries(group, ts):
    """Write a grace_timeseries into an HDF5 group."""
    group.create_dataset("iteration", data=ts.iteration)
    if ts.time.size > 0:
        group.create_dataset("time", data=ts.time)
    if isinstance(ts.data, dict):
        for col, arr in ts.data.items():
            group.create_dataset(col, data=arr)
    else:
        group.create_dataset("data", data=ts.data)


def export_scalars_hdf5(scalars, gw, outfile, name=None, detectors=None):
    """Dump all scalar and GW data into a single HDF5 file.

    Args:
        scalars (grace_scalars_reader): Scalar data reader.
        gw (grace_gw_data): Gravitational wave data reader.
        outfile (str): Output HDF5 file path.
        name (str, optional): Simulation name (stored as file attribute).
        detectors (grace_detector_set, optional): Detector metadata to store.
    """
    with h5py.File(outfile, "w") as f:
        # File-level metadata
        if name is not None:
            f.attrs["name"] = name
        if gw.Madm is not None:
            f.attrs["Madm"] = gw.Madm
        if gw.omega0 is not None:
            f.attrs["omega0"] = gw.omega0
        # Reductions
        for red_name in ("maximum", "minimum", "norm2", "integral"):
            container = getattr(scalars, red_name)
            keys = list(container.available_vars())
            if not keys:
                continue
            grp = f.require_group(f"reductions/{red_name}")
            for var in sorted(keys):
                _write_timeseries(grp.create_group(var), container[var])

        # EM energy
        if scalars.em_energy is not None:
            _write_timeseries(f.create_group("em_energy"), scalars.em_energy)

        # Mass flux
        if scalars.mass_flux:
            mf_grp = f.create_group("mass_flux")
            for det in sorted(scalars.mass_flux):
                det_grp = mf_grp.create_group(det)
                container = scalars.mass_flux[det]
                for ftype in sorted(container.available_vars()):
                    _write_timeseries(det_grp.create_group(ftype), container[ftype])

        # Compact object locations
        if scalars.co_locations:
            co_grp = f.create_group("co_locations")
            for name in sorted(scalars.co_locations):
                _write_timeseries(co_grp.create_group(name), scalars.co_locations[name])

        # GW data
        for det_name in gw.available_detectors():
            det = gw[det_name]
            det_grp = f.require_group(f"gw/{det_name}")
            for l, m in det.available_modes():
                mode = det[(l, m)]
                mode_grp = det_grp.create_group(f"l{l}_m{m}")
                mode_grp.create_dataset("iteration", data=mode.iteration)
                mode_grp.create_dataset("time", data=mode.time)
                mode_grp.create_dataset("real", data=mode.data.real)
                mode_grp.create_dataset("imag", data=mode.data.imag)

        # Detector metadata
        if detectors is not None:
            det_grp = f.require_group("detectors")
            for det_name in detectors.available_detectors():
                det = detectors[det_name]
                dg = det_grp.create_group(det_name)
                if det.radius is not None:
                    dg.attrs["radius"] = det.radius
                if det.center is not None:
                    dg.attrs["center"] = np.array(det.center)
                if det.resolution is not None:
                    dg.attrs["resolution"] = det.resolution
                if det.sampling_policy is not None:
                    dg.attrs["sampling_policy"] = det.sampling_policy


def _read_timeseries(group, name):
    """Reconstruct a grace_timeseries from an HDF5 group."""
    ts = grace_timeseries.__new__(grace_timeseries)
    ts.name = name
    ts.iteration = group["iteration"][:].astype(int)
    ts.time = group["time"][:] if "time" in group else np.array([])

    # Identify data datasets (everything except iteration and time)
    data_keys = [k for k in group.keys() if k not in ("iteration", "time")]

    if len(data_keys) == 1 and data_keys[0] == "data":
        ts.columns = ["Value"]
        ts.data = group["data"][:]
    else:
        ts.columns = sorted(data_keys)
        ts.data = {k: group[k][:] for k in ts.columns}

    return ts


_MODE_RE = re.compile(r"^l(\d+)_m(-?\d+)$")


def import_scalars_hdf5(filepath):
    """Reconstruct scalar and GW data from an HDF5 file.

    This is the inverse of ``export_scalars_hdf5``. It returns the
    individual reader objects needed to build a ``grace_simulation``.

    Args:
        filepath (str): Path to the HDF5 file.

    Returns:
        dict: Keys ``"scalars"``, ``"gw"``, ``"detectors"``, ``"name"``.
    """
    with h5py.File(filepath, "r") as f:
        # File-level metadata
        sim_name = f.attrs.get("name", "grace")
        if isinstance(sim_name, bytes):
            sim_name = sim_name.decode()
        Madm = f.attrs.get("Madm")
        omega0 = f.attrs.get("omega0")

        # Reconstruct scalars reader (bypass __init__ which expects dirs)
        scalars = grace_scalars_reader.__new__(grace_scalars_reader)
        scalars.maximum = grace_timeseries_array()
        scalars.minimum = grace_timeseries_array()
        scalars.norm2 = grace_timeseries_array()
        scalars.integral = grace_timeseries_array()
        scalars.em_energy = None
        scalars.mass_flux = {}
        scalars.co_locations = {}

        # Reductions
        if "reductions" in f:
            for red_name in ("maximum", "minimum", "norm2", "integral"):
                if red_name not in f["reductions"]:
                    continue
                container = getattr(scalars, red_name)
                for var in f[f"reductions/{red_name}"]:
                    container[var] = _read_timeseries(
                        f[f"reductions/{red_name}/{var}"], var
                    )

        # EM energy
        if "em_energy" in f:
            scalars.em_energy = _read_timeseries(f["em_energy"], "E_em")

        # Mass flux
        if "mass_flux" in f:
            for det_name in f["mass_flux"]:
                scalars.mass_flux[det_name] = grace_timeseries_array()
                for ftype in f[f"mass_flux/{det_name}"]:
                    scalars.mass_flux[det_name][ftype] = _read_timeseries(
                        f[f"mass_flux/{det_name}/{ftype}"], ftype
                    )

        # CO locations
        if "co_locations" in f:
            for co_name in f["co_locations"]:
                scalars.co_locations[co_name] = _read_timeseries(
                    f[f"co_locations/{co_name}"], co_name
                )

        # GW data (bypass __init__ which expects dirs)
        gw = grace_gw_data.__new__(grace_gw_data)
        gw.Madm = float(Madm) if Madm is not None else None
        gw.omega0 = float(omega0) if omega0 is not None else None
        gw.detectors = {}

        if "gw" in f:
            for det_name in f["gw"]:
                gw_det = grace_gw_detector(det_name)
                for mode_key in f[f"gw/{det_name}"]:
                    match = _MODE_RE.match(mode_key)
                    if not match:
                        continue
                    l = int(match.group(1))
                    m = int(match.group(2))
                    mg = f[f"gw/{det_name}/{mode_key}"]
                    mode = grace_gw_mode(
                        l, m,
                        mg["iteration"][:],
                        mg["time"][:],
                        mg["real"][:],
                        mg["imag"][:],
                    )
                    gw_det[(l, m)] = mode
                gw.detectors[det_name] = gw_det

        # Detector metadata
        dset = grace_detector_set()
        if "detectors" in f:
            for det_name in f["detectors"]:
                dg = f[f"detectors/{det_name}"]
                center = tuple(dg.attrs["center"]) if "center" in dg.attrs else None
                sp = dg.attrs.get("sampling_policy")
                if isinstance(sp, bytes):
                    sp = sp.decode()
                det = grace_detector(
                    name=det_name,
                    radius=dg.attrs.get("radius"),
                    center=center,
                    resolution=dg.attrs.get("resolution"),
                    sampling_policy=sp,
                )
                dset[det_name] = det

        # Attach GW and mass flux data to detectors
        for det_name in gw.available_detectors():
            if det_name not in dset:
                dset[det_name] = grace_detector(det_name)
            dset[det_name].gw = gw[det_name]

        for det_name in scalars.mass_flux:
            if det_name not in dset:
                dset[det_name] = grace_detector(det_name)
            dset[det_name].mass_flux = scalars.mass_flux[det_name]

    return {
        "name": sim_name,
        "scalars": scalars,
        "gw": gw,
        "detectors": dset,
    }
