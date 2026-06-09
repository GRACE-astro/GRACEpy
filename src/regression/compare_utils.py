"""Compare GRACE simulation results against a regression baseline.

Flattens both result sets into a dict of named numpy arrays and diffs them
elementwise with ``rtol``/``atol``. Performance/timing metrics are excluded by
default because they are machine- and run-dependent.
"""

import os
import json

import numpy as np

from regression.artifact_utils import load_artifact_results


# Suffixes that identify coordinate (not physics) arrays; skipped under
# ``--ignore-coords``.
_COORD_SUFFIXES = ("/iteration", "/time", "/t_ret")

# Guard against division by zero in the relative-difference metric.
_TINY = 1e-300


def load_results_source(path, parfile=None, verbose=False):
    """Load results from either a simulation directory or an exported HDF5 file.

    Args:
        path (str): A simulation directory, or an exported scalars HDF5 file.
        parfile (str, optional): Parameter file (only used for a simdir).
        verbose (bool, optional): Print progress messages (only for a simdir).

    Returns:
        dict: ``{"name", "scalars", "gw", "detectors"}``.
    """
    if os.path.isdir(path):
        from grace_tools.simutils import grace_simulation
        sim = grace_simulation(path, parfile=parfile, verbose=verbose)
        return {
            "name": sim.name,
            "scalars": sim.scalars,
            "gw": sim.gw,
            "detectors": sim.detectors,
        }

    if os.path.isfile(path):
        from grace_tools.scalar_export import import_scalars_hdf5
        return import_scalars_hdf5(path)

    raise FileNotFoundError(f"Results source not found: {path}")


def _emit_timeseries(out, prefix, ts):
    """Add a grace_timeseries' coordinate and data arrays to ``out``."""
    out[f"{prefix}/iteration"] = np.asarray(ts.iteration)
    if getattr(ts, "time", None) is not None and np.size(ts.time) > 0:
        out[f"{prefix}/time"] = np.asarray(ts.time)
    if isinstance(ts.data, dict):
        for col, arr in ts.data.items():
            out[f"{prefix}/{col}"] = np.asarray(arr)
    else:
        out[f"{prefix}/Value"] = np.asarray(ts.data)


def flatten_results(data, include_performance=False):
    """Flatten a results dict into ``{qualified_key: np.ndarray}``.

    Args:
        data (dict): ``{"scalars", "gw", ...}`` as returned by
            :func:`load_results_source` or :func:`load_artifact_results`.
        include_performance (bool): Include ``performance/*`` arrays.

    Returns:
        dict[str, np.ndarray]: Flat mapping of all real-valued arrays. GW
        complex data is split into ``.../real`` and ``.../imag`` keys.
    """
    out = {}
    scalars = data["scalars"]
    gw = data["gw"]

    # Reductions
    for red_name in ("maximum", "minimum", "norm2", "integral"):
        container = getattr(scalars, red_name)
        for var in container.available_vars():
            _emit_timeseries(out, f"reductions/{red_name}/{var}", container[var])

    # EM energy
    if getattr(scalars, "em_energy", None) is not None:
        _emit_timeseries(out, "em_energy", scalars.em_energy)

    # Mass flux
    for det in getattr(scalars, "mass_flux", {}):
        container = scalars.mass_flux[det]
        for ftype in container.available_vars():
            _emit_timeseries(out, f"mass_flux/{det}/{ftype}", container[ftype])

    # Compact-object locations
    for name in getattr(scalars, "co_locations", {}):
        _emit_timeseries(out, f"co_locations/{name}", scalars.co_locations[name])

    # Performance (excluded by default)
    if include_performance:
        for metric in getattr(scalars, "performance", {}):
            _emit_timeseries(out, f"performance/{metric}", scalars.performance[metric])

    # GW data: split complex into real/imag
    for det_name in gw.available_detectors():
        det = gw[det_name]
        for l, m in det.available_modes():
            mode = det[(l, m)]
            prefix = f"gw/{det_name}/l{l}_m{m}"
            out[f"{prefix}/iteration"] = np.asarray(mode.iteration)
            out[f"{prefix}/time"] = np.asarray(mode.time)
            out[f"{prefix}/real"] = np.asarray(mode.data.real)
            out[f"{prefix}/imag"] = np.asarray(mode.data.imag)
            if getattr(mode, "t_ret", None) is not None:
                out[f"{prefix}/t_ret"] = np.asarray(mode.t_ret)

    return out


def compare_arrays(a, b, rtol, atol):
    """Compare two arrays elementwise.

    Returns:
        dict: ``status`` in ``{"pass", "fail", "shape_mismatch"}`` plus
        ``max_abs_diff``, ``max_rel_diff``, ``n``.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "status": "shape_mismatch",
            "max_abs_diff": float("nan"),
            "max_rel_diff": float("nan"),
            "n": int(a.size),
            "shape_a": a.shape,
            "shape_b": b.shape,
        }

    if a.size == 0:
        return {"status": "pass", "max_abs_diff": 0.0, "max_rel_diff": 0.0, "n": 0}

    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    diff = np.abs(a - b)
    max_abs = float(np.nanmax(diff))
    max_rel = float(np.nanmax(diff / (np.abs(b) + _TINY)))
    status = "pass" if bool(close.all()) else "fail"
    return {"status": status, "max_abs_diff": max_abs, "max_rel_diff": max_rel,
            "n": int(a.size)}


def compare_results(baseline, new, rtol=1e-9, atol=1e-12,
                    include_performance=False, ignore_coords=False):
    """Compare two flattened result sets and return a :class:`RegressionReport`.

    Args:
        baseline (dict): Baseline results dict (e.g. from the artifact).
        new (dict): New results dict.
        rtol, atol (float): Elementwise tolerances.
        include_performance (bool): Compare performance metrics too.
        ignore_coords (bool): Skip iteration/time/t_ret coordinate arrays.
    """
    base_flat = flatten_results(baseline, include_performance=include_performance)
    new_flat = flatten_results(new, include_performance=include_performance)

    if ignore_coords:
        base_flat = {k: v for k, v in base_flat.items() if not k.endswith(_COORD_SUFFIXES)}
        new_flat = {k: v for k, v in new_flat.items() if not k.endswith(_COORD_SUFFIXES)}

    base_keys = set(base_flat)
    new_keys = set(new_flat)
    common = sorted(base_keys & new_keys)
    missing = sorted(base_keys - new_keys)
    extra = sorted(new_keys - base_keys)

    entries = []
    for key in common:
        result = compare_arrays(base_flat[key], new_flat[key], rtol, atol)
        result["key"] = key
        entries.append(result)

    return RegressionReport(rtol, atol, entries, missing, extra)


class RegressionReport:
    """Outcome of a regression comparison.

    Attributes:
        rtol, atol (float): Tolerances used.
        entries (list[dict]): Per-key comparison results.
        missing (list[str]): Keys in baseline but absent from new (regression).
        extra (list[str]): Keys in new but absent from baseline.
    """

    def __init__(self, rtol, atol, entries, missing, extra):
        self.rtol = rtol
        self.atol = atol
        self.entries = entries
        self.missing = missing
        self.extra = extra

    def failures(self):
        """Entries whose status is not ``"pass"``."""
        return [e for e in self.entries if e["status"] != "pass"]

    def passed(self, strict=False):
        """Whether the comparison passes.

        Args:
            strict (bool): Treat extra keys as a failure.
        """
        if self.failures() or self.missing:
            return False
        if strict and self.extra:
            return False
        return True

    def summary(self, strict=False):
        """Return a human-readable report string."""
        lines = []
        n_pass = sum(1 for e in self.entries if e["status"] == "pass")
        lines.append(f"Regression comparison (rtol={self.rtol:g}, atol={self.atol:g})")
        lines.append(f"  compared : {len(self.entries)} quantities, {n_pass} passed")

        failures = self.failures()
        if failures:
            lines.append(f"  FAILED   : {len(failures)}")
            for e in failures:
                if e["status"] == "shape_mismatch":
                    lines.append(f"    [shape] {e['key']}: "
                                 f"{e.get('shape_a')} vs {e.get('shape_b')}")
                else:
                    lines.append(f"    [diff ] {e['key']}: "
                                 f"max_abs={e['max_abs_diff']:.3e} "
                                 f"max_rel={e['max_rel_diff']:.3e} (n={e['n']})")

        if self.missing:
            lines.append(f"  MISSING  : {len(self.missing)} (in baseline, absent in new)")
            for k in self.missing:
                lines.append(f"    - {k}")

        if self.extra:
            label = "EXTRA-FAIL" if strict else "extra"
            lines.append(f"  {label.upper():9}: {len(self.extra)} (in new, absent in baseline)")
            for k in self.extra:
                lines.append(f"    + {k}")

        lines.append(f"  RESULT   : {'PASS' if self.passed(strict=strict) else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self, strict=False):
        """Return a JSON-serializable dict of the report."""
        return {
            "rtol": self.rtol,
            "atol": self.atol,
            "passed": self.passed(strict=strict),
            "n_compared": len(self.entries),
            "n_failed": len(self.failures()),
            "entries": [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in e.items()}
                for e in self.entries
            ],
            "missing": self.missing,
            "extra": self.extra,
        }

    def to_json(self, strict=False):
        return json.dumps(self.to_dict(strict=strict), indent=2)


def compare_against_artifact(artifact_path, new_path, parfile=None,
                             rtol=1e-9, atol=1e-12, include_performance=False,
                             ignore_coords=False, verbose=False):
    """Convenience wrapper: load a baseline artifact and new results, compare.

    Returns:
        RegressionReport
    """
    baseline = load_artifact_results(artifact_path)
    new = load_results_source(new_path, parfile=parfile, verbose=verbose)
    return compare_results(baseline, new, rtol=rtol, atol=atol,
                           include_performance=include_performance,
                           ignore_coords=ignore_coords)
