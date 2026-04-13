"""Utilities for reading scalar output from grace."""

import numpy as np
import os
import glob
import re

from grace_tools.timeseries_utils import (
    grace_timeseries_array,
    load_scalar_file,
    merge_scalar_dirs,
)

REDUCTION_TYPES = {"max": "maximum", "min": "minimum", "norm2": "norm2", "integral": "integral"}
_REDUCTION_RE = re.compile(r"^(.+)_(max|min|norm2|integral)$")
_GW_RE = re.compile(r"^r?Psi\d+m?\d+_(re|im)_.+$")
_CO_LOC_RE = re.compile(r"^co_(.+)_loc$")

# Known flux name prefixes from GRACE diagnostics (longest first for greedy match)
_KNOWN_FLUX_PREFIXES = sorted([
    "Mdot_unbound_geo", "Mdot_unbound_bern", "Mdot_tot",
    "Mdot", "Edot", "Ldot", "Phi",
    "E_ADM", "Px_ADM", "Py_ADM", "Pz_ADM",
    "Jx_ADM", "Jy_ADM", "Jz_ADM",
], key=len, reverse=True)


class grace_scalars_reader:
    """Reader class for grace scalar data.

    This class reads all .dat files in one or more directories,
    categorizing them by type:

    - **Reductions** (max/min/norm2/integral): bucketed by reduction type
    - **EM energy** (E_em.dat): electromagnetic energy diagnostic
    - **Mass flux** (Mdot_{type}_{detector}.dat): per-detector mass flux
    - **Compact object locations** (co_{name}_loc.dat): tracked object positions

    GW files (rPsi4/Psi4) are skipped — use grace_gw_data for those.

    Attributes:
        maximum (grace_timeseries_array):
            Timeseries computed with 'maximum' reduction.
        minimum (grace_timeseries_array):
            Timeseries computed with 'minimum' reduction.
        norm2 (grace_timeseries_array):
            Timeseries computed with 'norm2' reduction.
        integral (grace_timeseries_array):
            Timeseries computed with 'integral' reduction.
        em_energy (grace_timeseries or None):
            Electromagnetic energy diagnostic (E_em.dat).
        mass_flux (dict):
            Mapping detector_name -> grace_timeseries_array keyed by flux type.
        co_locations (dict):
            Mapping CO name -> grace_timeseries with multi-column position data.
    """

    def __init__(self, dirs):
        """Create a grace_scalars_reader object.

        Args:
            dirs (str or list): Single directory path or list of directory
                paths (for restart merging) where scalar output is found.
        """
        if isinstance(dirs, str):
            dirs = [dirs]

        self.maximum      = grace_timeseries_array()
        self.minimum      = grace_timeseries_array()
        self.norm2        = grace_timeseries_array()
        self.integral     = grace_timeseries_array()
        self.em_energy    = None
        self.mass_flux    = {}
        self.co_locations = {}

        if len(dirs) == 1:
            self._load_single_dir(dirs[0])
        else:
            self._load_merged(dirs)

    def _categorize(self, varname, ts):
        """Route a timeseries into the appropriate category."""
        # Skip GW files (handled by grace_gw_data)
        if _GW_RE.match(varname):
            return

        # Reduction types (max, min, norm2, integral)
        match = _REDUCTION_RE.match(varname)
        if match:
            name = match.group(1)
            red_type = match.group(2)
            attr = REDUCTION_TYPES[red_type]
            getattr(self, attr)[name] = ts
            return

        # EM energy diagnostic
        if varname == "E_em":
            self.em_energy = ts
            return

        # Flux diagnostics: {flux_prefix}_{detector}
        # Flux prefixes are matched longest-first to handle e.g.
        # "Mdot_unbound_geo" before "Mdot"
        for prefix in _KNOWN_FLUX_PREFIXES:
            if varname.startswith(prefix + "_"):
                detector = varname[len(prefix) + 1:]
                if detector:
                    if detector not in self.mass_flux:
                        self.mass_flux[detector] = grace_timeseries_array()
                    self.mass_flux[detector][prefix] = ts
                    return
                break

        # Compact object locations: co_{name}_loc
        match = _CO_LOC_RE.match(varname)
        if match:
            co_name = match.group(1)
            self.co_locations[co_name] = ts
            return

        print(f"WARNING: unrecognized scalar file '{varname}.dat'")

    def _load_single_dir(self, directory):
        """Load scalar files from a single directory."""
        patt = os.path.join(directory, "*.dat")
        for ff in sorted(glob.glob(patt)):
            _, fname = os.path.split(ff)
            varname = os.path.splitext(fname)[0]
            ts = load_scalar_file(ff, name=varname)
            self._categorize(varname, ts)

    def _load_merged(self, dirs):
        """Load and merge scalar files from multiple directories."""
        merged = merge_scalar_dirs(dirs)
        for varname in merged.available_vars():
            self._categorize(varname, merged[varname])

    def __repr__(self):
        sections = []
        for label, container in [("maximum", self.maximum), ("minimum", self.minimum),
                                 ("norm2", self.norm2), ("integral", self.integral)]:
            keys = list(container.available_vars())
            if keys:
                sections.append(f"  {label} ({len(keys)}): {', '.join(sorted(keys))}")
        if self.em_energy is not None:
            cols = ", ".join(self.em_energy.columns)
            sections.append(f"  em_energy: {cols}")
        if self.mass_flux:
            for det in sorted(self.mass_flux):
                types = sorted(self.mass_flux[det].available_vars())
                sections.append(f"  mass_flux[{det}]: {', '.join(types)}")
        if self.co_locations:
            names = sorted(self.co_locations)
            sections.append(f"  co_locations ({len(names)}): {', '.join(names)}")
        if sections:
            return "grace_scalars_reader:\n" + "\n".join(sections)
        return "grace_scalars_reader: (empty)"
