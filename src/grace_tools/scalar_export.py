"""Export merged scalar data to a single HDF5 file."""

import numpy as np
import h5py


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


def export_scalars_hdf5(scalars, gw, outfile):
    """Dump all scalar and GW data into a single HDF5 file.

    Args:
        scalars (grace_scalars_reader): Scalar data reader.
        gw (grace_gw_data): Gravitational wave data reader.
        outfile (str): Output HDF5 file path.
    """
    with h5py.File(outfile, "w") as f:
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
