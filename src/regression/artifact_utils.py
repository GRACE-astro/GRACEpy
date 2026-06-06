"""Build and read self-contained GRACE regression artifacts.

A regression artifact is a single HDF5 file with three non-colliding
namespaces::

    /            root attrs: provenance metadata (see metadata_utils)
    /source/     sealed source tree + git commit (source_seal)
    /results/    scalar + GW results (grace_tools.scalar_export)

plus, when available, a ``/config_summary`` dataset holding the build's
config-summary text recovered from ``<simdir>/config/config_summary``.
"""

import os
import h5py
import numpy as np

from source_seal.package_utils import seal_source_tree_into_group
from grace_tools.scalar_export import (
    export_scalars_to_group,
    import_scalars_from_group,
)
from regression.metadata_utils import capture_metadata, write_metadata


SCHEMA_VERSION = "1"
ARTIFACT_KIND = "grace_regression"

# Mirror the default exclusions used by the archive_source CLI.
DEFAULT_EXCLUDE_PATTERNS = ["*.git*", "*build*"]


def create_artifact(simdir, source_dir, output_file, parfile=None,
                    exclude_patterns=None, machine_config=None, verbose=False):
    """Build a self-contained regression artifact from a simulation.

    Args:
        simdir (str): Path to the simulation directory.
        source_dir (str): Source tree to seal (with its git commit).
        output_file (str): Output artifact HDF5 path.
        parfile (str, optional): Parameter file (auto-detected if omitted).
        exclude_patterns (list, optional): Glob patterns to exclude from the
            sealed source tree. Defaults to ``["*.git*", "*build*"]``.
        machine_config (str, optional): grace_pilot machine YAML for the logical
            machine name. Defaults to ``<simdir>/config/machine.yaml`` if present.
        verbose (bool, optional): Print progress messages.

    Returns:
        str: ``output_file``.
    """
    from grace_tools.simutils import grace_simulation

    if exclude_patterns is None:
        exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    sim = grace_simulation(simdir, parfile=parfile, verbose=verbose)

    # Default the logical machine name to the simulation's own machine config.
    if machine_config is None:
        candidate = os.path.join(simdir, "config", "machine.yaml")
        if os.path.isfile(candidate):
            machine_config = candidate

    meta = capture_metadata(source_dir, sim_name=sim.name, machine_config=machine_config)

    # Build provenance summary embedded by simpilot, if present.
    config_summary_path = os.path.join(simdir, "config", "config_summary")
    config_summary = None
    if os.path.isfile(config_summary_path):
        with open(config_summary_path, "r", errors="replace") as f:
            config_summary = f.read()

    with h5py.File(output_file, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["artifact_kind"] = ARTIFACT_KIND
        write_metadata(f, meta)

        if config_summary is not None:
            f.create_dataset("config_summary",
                             data=np.bytes_(config_summary.encode("utf-8")))

        seal_source_tree_into_group(f.create_group("source"), source_dir, exclude_patterns)
        export_scalars_to_group(f.create_group("results"), sim.scalars, sim.gw,
                                name=sim.name, detectors=sim.detectors)

    if verbose:
        print(f"Regression artifact written to {output_file}")
    return output_file


def load_artifact_results(artifact_path):
    """Load the scalar + GW results stored under ``/results`` in an artifact.

    Args:
        artifact_path (str): Path to a regression artifact HDF5 file.

    Returns:
        dict: ``{"name", "scalars", "gw", "detectors"}`` (see
        :func:`grace_tools.scalar_export.import_scalars_from_group`).
    """
    with h5py.File(artifact_path, "r") as f:
        if "results" not in f:
            raise ValueError(
                f"'{artifact_path}' has no /results group; not a regression artifact?"
            )
        return import_scalars_from_group(f["results"])


def load_artifact_metadata(artifact_path):
    """Return the provenance metadata of an artifact.

    Args:
        artifact_path (str): Path to a regression artifact HDF5 file.

    Returns:
        dict: Root attributes, the ``/source`` git attributes (under keys
        ``commit_hash`` / ``unstaged_changes``), and ``config_summary`` text
        if present.
    """
    meta = {}
    with h5py.File(artifact_path, "r") as f:
        for key, val in f.attrs.items():
            meta[key] = val.decode() if isinstance(val, bytes) else val
        if "source" in f:
            for key, val in f["source"].attrs.items():
                meta[key] = val.decode() if isinstance(val, bytes) else val
        if "config_summary" in f:
            data = f["config_summary"][()]
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            meta["config_summary"] = data
    return meta
