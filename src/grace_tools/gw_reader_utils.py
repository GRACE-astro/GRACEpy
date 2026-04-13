"""Utilities for reading gravitational wave (rPsi4) scalar output from grace."""

import numpy as np
import os
import glob
import re

from grace_tools.timeseries_utils import load_scalar_file, merge_scalar_files
from analysis.gw_utils import fixed_frequency_integration

# Matches both old (Psi4) and new (rPsi4) naming conventions:
#   rPsi2m2_im_GW_1.dat  -> l=2, m=-2, im, GW_1
#   Psi22_re_GW_2.dat    -> l=2, m=2,  re, GW_2
_GW_RE = re.compile(r"^r?Psi(\d)(m?\d+)_(re|im)_(.+)\.dat$")


def _parse_m(m_str):
    """Parse the m quantum number from the filename encoding."""
    if m_str.startswith("m"):
        return -int(m_str[1:])
    return int(m_str)


# ---------------------------------------------------------------------------
# Spin-weighted spherical harmonic coupling coefficients (s = -2)
#
# These arise from the recurrence relations:
#   cos(theta) * sY_{lm} = A * sY_{l+1,m}   + B * sY_{l-1,m}
#   sin(theta) e^{iphi} * sY_{lm} = -C * sY_{l+1,m+1} + D * sY_{l-1,m+1}
#
# with s = -2 for gravitational wave strain modes.
# ---------------------------------------------------------------------------

def _swsh_A(l, m):
    """cos(theta) coupling: (l,m) -> (l+1,m).  s=-2."""
    num = ((l + 1)**2 - m**2) * ((l - 1) * (l + 3))
    den = (l + 1)**2 * (2 * l + 1) * (2 * l + 3)
    return np.sqrt(num / den) if den > 0 else 0.0


def _swsh_C(l, m):
    """sin(theta)e^{iphi} coupling: (l,m) -> (l+1,m+1).  s=-2."""
    num = (l + m + 1) * (l + m + 2) * (l - 1) * (l + 3)
    den = (l + 1)**2 * (2 * l + 1) * (2 * l + 3)
    return np.sqrt(num / den) if den > 0 else 0.0


def _swsh_D(l, m):
    """sin(theta)e^{iphi} coupling: (l,m) -> (l-1,m+1).  s=-2."""
    if l == 0:
        return 0.0
    num = (l - m) * (l - m - 1) * (l - 2) * (l + 2)
    den = l**2 * (2 * l - 1) * (2 * l + 1)
    return np.sqrt(num / den) if den > 0 else 0.0


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

    def __init__(self, dirs, Madm=None, omega0=None):
        """Create a grace_gw_data reader.

        Args:
            dirs (str or list): Single directory or list of directories
                (for restart merging) containing scalar output.
            Madm (float, optional): Total ADM mass of the system.
            omega0 (float, optional): Initial orbital frequency.
        """
        if isinstance(dirs, str):
            dirs = [dirs]

        self.Madm = Madm
        self.omega0 = omega0
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

    def _resolve_f0(self, f0):
        """Resolve the cutoff frequency from f0 or omega0."""
        if f0 is not None:
            return f0
        if self.omega0 is None:
            raise ValueError("No cutoff frequency: pass f0 or set omega0")
        return self.omega0 / (2 * np.pi)

    def _integrate_all_modes(self, detector, f0, N, window, wpars):
        """Integrate rPsi4 for all modes at a detector.

        Args:
            detector (str): Detector name.
            f0 (float): Cutoff frequency.
            N (int): Number of integrations.
            window (str): Window function.
            wpars (list): Window parameters.

        Returns:
            dict: Mapping (l,m) -> complex np.array of integrated data.
        """
        det = self[detector]
        result = {}
        for lm in det.available_modes():
            mode = det[lm]
            result[lm] = fixed_frequency_integration(
                mode.time, mode.data, f0, N, window, wpars
            )
        return result

    def compute_strain(self, detector, lm, f0=None, N=2, window="tukey", wpars=[0.2]):
        """Compute rh from rPsi4 for a given mode via fixed-frequency integration.

        Since GRACE outputs :math:`r\\Psi_4`, the result is :math:`r \\cdot h`
        (not h), where r is the extraction radius in code units
        (:math:`M_\\odot`).

        Uses ``self.omega0 / (2*pi)`` as the default cutoff frequency
        if ``f0`` is not provided.

        Args:
            detector (str): Detector name (e.g. 'GW_1').
            lm (tuple): (l, m) mode indices.
            f0 (float, optional): Cutoff frequency. Defaults to omega0/(2*pi).
            N (int): Number of integrations (default 2).
            window (str): Window function ('tukey', 'blackman', or None).
            wpars (list): Window function parameters.

        Returns:
            tuple: (time, rh) — time array and complex :math:`r \\cdot h(t)`.
        """
        mode = self[detector][lm]
        f0 = self._resolve_f0(f0)
        h = fixed_frequency_integration(mode.time, mode.data, f0, N, window, wpars)
        return mode.time, h

    # ------------------------------------------------------------------
    # Radiated quantities
    #
    # All formulas work directly with r*Psi4 and its time integrals
    # (r*hdot, r*h).  The r^2 from the solid-angle integral cancels
    # with the r^2 in the mode products, so no explicit extraction
    # radius is needed.
    # ------------------------------------------------------------------

    def radiated_energy(self, detector, f0=None, window="tukey", wpars=[0.2]):
        """Compute radiated energy from all available modes at a detector.

        .. math::

            \\frac{dE}{dt} = \\frac{1}{16\\pi}
            \\sum_{l,m} |r\\dot{h}_{lm}|^2

        Args:
            detector (str): Detector name.
            f0 (float, optional): Cutoff frequency. Defaults to omega0/(2*pi).
            window (str): Window function.
            wpars (list): Window parameters.

        Returns:
            tuple: (time, dEdt, E) — time array, energy flux, and
            cumulative radiated energy.
        """
        f0 = self._resolve_f0(f0)
        rhdot = self._integrate_all_modes(detector, f0, 1, window, wpars)
        t = self[detector][self[detector].available_modes()[0]].time

        dEdt = np.zeros(len(t))
        for lm, hdot in rhdot.items():
            dEdt += np.abs(hdot)**2
        dEdt /= (16 * np.pi)

        dt = t[1] - t[0]
        E = np.cumsum(dEdt) * dt
        return t, dEdt, E

    def radiated_angular_momentum(self, detector, f0=None,
                                  window="tukey", wpars=[0.2]):
        """Compute radiated angular momentum (z-component).

        .. math::

            \\frac{dJ_z}{dt} = -\\frac{1}{16\\pi}
            \\sum_{l,m} m \\, \\mathrm{Im}\\!
            \\left[ r\\dot{h}_{lm}^{\\,*} \\; r h_{lm} \\right]

        Convention: dJz/dt < 0 for a system losing angular momentum.

        Args:
            detector (str): Detector name.
            f0 (float, optional): Cutoff frequency.
            window (str): Window function.
            wpars (list): Window parameters.

        Returns:
            tuple: (time, dJzdt, Jz) — time array, angular momentum flux,
            and cumulative radiated angular momentum.
        """
        f0 = self._resolve_f0(f0)
        rhdot = self._integrate_all_modes(detector, f0, 1, window, wpars)
        rh = self._integrate_all_modes(detector, f0, 2, window, wpars)
        t = self[detector][self[detector].available_modes()[0]].time

        dJzdt = np.zeros(len(t))
        for (l, m), hdot in rhdot.items():
            if (l, m) in rh:
                dJzdt -= m * np.imag(np.conj(hdot) * rh[(l, m)])
        dJzdt /= (16 * np.pi)

        dt = t[1] - t[0]
        Jz = np.cumsum(dJzdt) * dt
        return t, dJzdt, Jz

    def radiated_linear_momentum(self, detector, f0=None,
                                 window="tukey", wpars=[0.2]):
        """Compute radiated linear momentum (all three components).

        Uses spin-weighted spherical harmonic coupling coefficients to
        compute the momentum flux from inter-mode products:

        .. math::

            \\dot{P}_z = \\frac{1}{8\\pi} \\sum_{l,m}
            A_{lm} \\, \\mathrm{Re}\\!\\left[
            r\\dot{h}_{lm} \\, r\\dot{h}^*_{l+1,m} \\right]

        .. math::

            \\dot{P}_+ \\equiv \\dot{P}_x + i\\dot{P}_y =
            \\frac{1}{16\\pi} \\sum_{l,m} \\left[
            -C_{lm} \\, r\\dot{h}_{lm} \\, r\\dot{h}^*_{l+1,m+1}
            + D_{lm} \\, r\\dot{h}_{lm} \\, r\\dot{h}^*_{l-1,m+1}
            \\right]

        where :math:`A`, :math:`C`, :math:`D` are coupling coefficients
        from the spin-weight :math:`s=-2` spherical harmonic recurrence
        relations.

        Args:
            detector (str): Detector name.
            f0 (float, optional): Cutoff frequency.
            window (str): Window function.
            wpars (list): Window parameters.

        Returns:
            tuple: (time, dPdt, P) where dPdt and P are arrays of shape
            ``(3, npoints)`` for the x, y, z components.
        """
        f0 = self._resolve_f0(f0)
        rhdot = self._integrate_all_modes(detector, f0, 1, window, wpars)
        t = self[detector][self[detector].available_modes()[0]].time
        n = len(t)

        dPzdt = np.zeros(n)
        dPpdt = np.zeros(n, dtype=complex)  # P_x + i P_y

        available = set(rhdot.keys())

        for (l, m) in available:
            hdot_lm = rhdot[(l, m)]

            # P_z: coupling (l,m) <-> (l+1,m)
            if (l + 1, m) in available:
                A = _swsh_A(l, m)
                dPzdt += A * np.real(hdot_lm * np.conj(rhdot[(l + 1, m)]))

            # P_+: coupling (l,m) -> (l+1,m+1) with coefficient -C
            if (l + 1, m + 1) in available:
                C = _swsh_C(l, m)
                dPpdt -= C * hdot_lm * np.conj(rhdot[(l + 1, m + 1)])

            # P_+: coupling (l,m) -> (l-1,m+1) with coefficient +D
            if (l - 1, m + 1) in available:
                D = _swsh_D(l, m)
                dPpdt += D * hdot_lm * np.conj(rhdot[(l - 1, m + 1)])

        dPzdt /= (8 * np.pi)
        dPpdt /= (16 * np.pi)

        dPdt = np.zeros((3, n))
        dPdt[0] = np.real(dPpdt)  # P_x
        dPdt[1] = np.imag(dPpdt)  # P_y
        dPdt[2] = dPzdt

        dt = t[1] - t[0]
        P = np.cumsum(dPdt, axis=1) * dt
        return t, dPdt, P

    def __repr__(self):
        meta = []
        if self.Madm is not None:
            meta.append(f"Madm={self.Madm:.4f}")
        if self.omega0 is not None:
            meta.append(f"omega0={self.omega0:.6f}")
        header = "grace_gw_data"
        if meta:
            header += f" ({', '.join(meta)})"
        parts = [repr(d) for d in self.detectors.values()]
        if parts:
            return header + ":\n  " + "\n  ".join(parts)
        return header + ": (empty)"
