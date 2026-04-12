"""Utilities for reading gravitational wave (rPsi4) scalar output from grace."""

import numpy as np
import os
import glob
import re

from grace_tools.timeseries_utils import load_scalar_file, merge_scalar_files

# Matches both old (Psi4) and new (rPsi4) naming conventions:
#   rPsi2m2_im_GW_1.dat  -> l=2, m=-2, im, GW_1
#   Psi22_re_GW_2.dat    -> l=2, m=2,  re, GW_2
_GW_RE = re.compile(r"^r?Psi(\d)(m?\d+)_(re|im)_(.+)\.dat$")


def _parse_m(m_str):
    """Parse the m quantum number from the filename encoding."""
    if m_str.startswith("m"):
        return -int(m_str[1:])
    return int(m_str)


class grace_gw_mode:
    """Complex timeseries for a single (l,m) mode of rPsi4.

    Attributes:
        l (int): Spherical harmonic degree.
        m (int): Spherical harmonic order.
        iteration (np.array): Iteration numbers.
        time (np.array): Coordinate times.
        data (np.array): Complex array of rPsi4_{lm}(t).
    """

    def __init__(self, l, m, iteration, time, re_data, im_data):
        self.l = l
        self.m = m
        self.iteration = iteration
        self.time = time
        self.data = re_data + 1j * im_data

    def __repr__(self):
        return f"grace_gw_mode(l={self.l}, m={self.m}, npoints={len(self.time)})"


class grace_gw_detector:
    """Container for all (l,m) modes extracted at one detector.

    Attributes:
        name (str): Detector name (e.g. 'GW_1').
        modes (dict): Mapping (l,m) -> grace_gw_mode.
    """

    def __init__(self, name):
        self.name = name
        self.modes = {}

    def __getitem__(self, lm):
        """Retrieve a mode by (l,m) tuple."""
        return self.modes[lm]

    def __setitem__(self, lm, mode):
        self.modes[lm] = mode

    def available_modes(self):
        """Return sorted list of available (l,m) tuples."""
        return sorted(self.modes.keys())

    def __repr__(self):
        modes_str = ", ".join(f"({l},{m})" for l, m in self.available_modes())
        return f"grace_gw_detector('{self.name}', modes=[{modes_str}])"


class grace_gw_data:
    """Reader for gravitational wave data from grace scalar output.

    Parses rPsi4 (or Psi4) files, groups them by detector and (l,m) mode,
    and stores rPsi4_{lm} as complex timeseries.

    Usage:
        gw = grace_gw_data("/path/to/output_scalar")
        mode22 = gw["GW_1"][2, 2]
        plt.plot(mode22.time, mode22.data.real)

    Attributes:
        detectors (dict): Mapping detector_name -> grace_gw_detector.
    """

    def __init__(self, dirs):
        """Create a grace_gw_data reader.

        Args:
            dirs (str or list): Single directory or list of directories
                (for restart merging) containing scalar output.
        """
        if isinstance(dirs, str):
            dirs = [dirs]

        self.detectors = {}
        self._load(dirs)

    def _load(self, dirs):
        """Scan directories for rPsi4/Psi4 files and build complex modes."""
        # Group files: (l, m, part, detector) -> [filepaths]
        file_groups = {}
        for d in dirs:
            for fpath in sorted(glob.glob(os.path.join(d, "*.dat"))):
                basename = os.path.basename(fpath)
                match = _GW_RE.match(basename)
                if not match:
                    continue
                l = int(match.group(1))
                m = _parse_m(match.group(2))
                part = match.group(3)
                det = match.group(4)
                key = (l, m, part, det)
                file_groups.setdefault(key, []).append(fpath)

        # Merge and pair re/im components
        # Intermediate: (l, m, det) -> {"re": timeseries, "im": timeseries}
        paired = {}
        for (l, m, part, det), fpaths in file_groups.items():
            if len(fpaths) == 1:
                ts = load_scalar_file(fpaths[0])
            else:
                ts = merge_scalar_files(fpaths)
            paired.setdefault((l, m, det), {})[part] = ts

        # Build detectors and modes
        for (l, m, det), parts in paired.items():
            if "re" not in parts or "im" not in parts:
                continue
            re_ts = parts["re"]
            im_ts = parts["im"]

            if det not in self.detectors:
                self.detectors[det] = grace_gw_detector(det)

            self.detectors[det][(l, m)] = grace_gw_mode(
                l, m, re_ts.iteration, re_ts.time, re_ts.data, im_ts.data
            )

    def __getitem__(self, detector_name):
        """Retrieve a detector by name."""
        return self.detectors[detector_name]

    def available_detectors(self):
        """Return list of available detector names."""
        return sorted(self.detectors.keys())

    def __repr__(self):
        parts = [repr(d) for d in self.detectors.values()]
        return "grace_gw_data:\n  " + "\n  ".join(parts) if parts else "grace_gw_data: (empty)"
