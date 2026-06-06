"""Capture run-provenance metadata for regression artifacts.

Collects hostname, user, date, platform, git commit, and (when running inside a
SLURM job) job information, so a regression artifact records where and how its
baseline results were produced.
"""

import os
import socket
import getpass
import platform
from datetime import datetime, timezone

from source_seal.package_utils import get_git_info


# SLURM environment variables captured at artifact-creation time. Only the ones
# present in the environment are written, so interactive (non-job) runs simply
# omit them.
SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_CLUSTER_NAME",
    "SLURM_SUBMIT_DIR",
    "SLURM_JOB_PARTITION",
)


def _machine_name_from_config(machine_config):
    """Return the logical machine name from a grace_pilot machine YAML, or None.

    Imported lazily so this module has no hard dependency on grace_pilot.
    """
    if machine_config is None or not os.path.isfile(machine_config):
        return None
    try:
        from grace_pilot.machine import machine
        return machine(machine_config).name
    except Exception:
        return None


def capture_metadata(source_dir, sim_name=None, machine_config=None):
    """Collect run-provenance metadata as a flat dict of HDF5-writable values.

    Args:
        source_dir (str): Source tree the commit hash is read from.
        sim_name (str, optional): Simulation name to record.
        machine_config (str, optional): Path to a grace_pilot machine YAML; its
            ``name`` is recorded as the logical machine name when available.

    Returns:
        dict: Flat ``{attr_name: value}`` mapping (no ``None`` values).
    """
    commit_hash, unstaged_changes = get_git_info(source_dir)
    has_unstaged = bool(
        unstaged_changes
        and unstaged_changes not in ("", "could not retrieve git information")
    )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "user": getpass.getuser(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "commit_hash": commit_hash,
        "has_unstaged_changes": has_unstaged,
    }

    if sim_name is not None:
        meta["sim_name"] = sim_name

    machine_name = _machine_name_from_config(machine_config)
    if machine_name is not None:
        meta["machine_name"] = machine_name

    # SLURM job info: only present variables are recorded. Keys are lowercased
    # to match the rest of the (lowercase) attribute namespace.
    for key in SLURM_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            meta[key.lower()] = val

    return meta


def write_metadata(group, meta):
    """Write each value of ``meta`` to ``group.attrs`` (skips ``None`` values).

    Args:
        group (h5py.Group): Open HDF5 group/file to write attributes on.
        meta (dict): Flat mapping produced by :func:`capture_metadata`.
    """
    for key, val in meta.items():
        if val is not None:
            group.attrs[key] = val
