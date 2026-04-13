"""Utilities for processing grace timeseries."""

import numpy as np
import os
import glob
import re
from collections import defaultdict


class grace_timeseries_array:
    """Array of grace timeseries

    This class is a container designed to hold
    timeseries data for different variables coming
    from grace.

    Methods:
        available_vars():
            Return a list of available vars in this container.
        __getitem__(key):
            Retrieve an item at the specified key.
        __setitem__(key, value):
            Set the item at the specified key to the given value.
    """

    def __init__(self):
        """Initialize an empty grace timeseries container."""
        self.__data = dict()

    def __getitem__(self, key: str):
        """Get a grace_timeseries corresponding to the requested variable.

        Args:
            key (str):
                Name of requested variable.

        Returns:
            grace_timeseries:
                The timeseries corresponding to the requested variable
        """
        return self.__data[key]

    def __setitem__(self, key: str, value):
        """Set a grace_timeseries corresponding to the requested variable.

        Args:
            key (str):
                Name of requested variable.
            value (grace_timeseries):
                The timeseries to be added to this array (or modified).
        """
        self.__data[key] = value

    def available_vars(self):
        """Query available variables in this container.

        Returns:
            list:
                A list of all available variable names.
        """
        return self.__data.keys()

    def __repr__(self):
        keys = sorted(self.__data.keys())
        if not keys:
            return "grace_timeseries_array: (empty)"
        return f"grace_timeseries_array ({len(keys)}): {', '.join(keys)}"


class grace_timeseries:
    """Class representing a timeseries of grace data.

    Supports multi-column scalar files. The header line is parsed
    to determine column names. Iteration and time are extracted as
    dedicated attributes; remaining columns are stored in the data dict.

    Attributes:
        iteration (np.array):
            Array containing iterations at which data is available.
        time (np.array):
            Array containing times at which data is available.
        data (dict or np.array):
            If the file has a single data column named 'Value', this is
            a plain numpy array (backward compatible). Otherwise it is
            a dict mapping column names to numpy arrays.
        columns (list):
            List of data column names (excludes iteration and time).
        name (str):
            Name of the variable.
    """

    def __init__(self, file, name):
        """Initialize a grace_timeseries from data in a file.

        Args:
            file (str): Path to the scalar data file.
            name (str): Name for this timeseries.
        """
        self.name = name
        if not os.path.isfile(file):
            raise ValueError(f"File {file} does not exist or is not readable.")

        self._load(file)

    def _load(self, file):
        """Parse a scalar file, extracting header and data."""
        with open(file, 'r') as f:
            header_line = f.readline().strip()

        col_names = header_line.split()

        raw = np.loadtxt(file, skiprows=1)

        # Map header names to columns
        iter_col = None
        time_col = None
        data_cols = {}

        for i, cname in enumerate(col_names):
            low = cname.lower()
            if low == "iteration":
                iter_col = i
            elif low == "time":
                time_col = i
            else:
                data_cols[cname] = i

        self.columns = list(data_cols.keys())

        # Handle empty files (header only, no data rows)
        if raw.size == 0:
            self.iteration = np.array([], dtype=int)
            self.time = np.array([])
            if len(data_cols) == 1 and self.columns[0] == "Value":
                self.data = np.array([])
            else:
                self.data = {cname: np.array([]) for cname in data_cols}
            return

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        self.iteration = raw[:, iter_col].astype(int) if iter_col is not None else np.array([], dtype=int)
        self.time = raw[:, time_col] if time_col is not None else np.array([])

        if len(data_cols) == 1 and self.columns[0] == "Value":
            self.data = raw[:, data_cols["Value"]]
        else:
            self.data = {cname: raw[:, idx] for cname, idx in data_cols.items()}

    @property
    def value(self):
        """Uniform accessor for single-column data.

        Returns the data array directly regardless of whether the
        underlying storage is a plain array or a dict with a 'Value' key.

        Returns:
            np.array: The data values.

        Raises:
            ValueError: If the timeseries has multiple data columns.
        """
        if isinstance(self.data, dict):
            if len(self.data) == 1:
                return next(iter(self.data.values()))
            raise ValueError(
                f"Timeseries '{self.name}' has multiple data columns "
                f"({self.columns}), use .data[column_name] instead.")
        return self.data

    def time_at_iteration(self, it):
        """Return the time corresponding to a given iteration number.

        Args:
            it (int): Iteration number.

        Returns:
            float: The corresponding time value.
        """
        idx = np.searchsorted(self.iteration, it)
        if idx >= len(self.iteration) or self.iteration[idx] != it:
            raise KeyError(f"Iteration {it} not found in timeseries '{self.name}'")
        return self.time[idx]

    def iteration_at_time(self, t):
        """Return the iteration number closest to a given time.

        Args:
            t (float): Target time.

        Returns:
            int: The iteration number of the closest time sample.
        """
        idx = np.argmin(np.abs(self.time - t))
        return self.iteration[idx]


def load_scalar_file(filepath, name=None):
    """Load a single grace scalar file.

    Args:
        filepath (str): Path to the .dat file.
        name (str, optional): Name for the timeseries. If None,
            derived from the filename (without extension).

    Returns:
        grace_timeseries: The loaded timeseries.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
    return grace_timeseries(filepath, name)


def merge_scalar_files(filepaths, name=None):
    """Merge multiple scalar files for the same variable (e.g. from restarts).

    Files are concatenated along the time/iteration axis. Duplicate
    iterations are removed, keeping the entry from the latest file
    (i.e. later files in the list take precedence).

    Args:
        filepaths (list): List of file paths to merge, ordered by restart
            sequence (earliest first).
        name (str, optional): Name for the resulting timeseries.

    Returns:
        grace_timeseries: Merged timeseries.
    """
    if not filepaths:
        raise ValueError("No files to merge.")

    series = [load_scalar_file(f, name=name) for f in filepaths]

    if name is None:
        name = series[0].name

    # Concatenate all arrays
    iterations = np.concatenate([s.iteration for s in series])
    times = np.concatenate([s.time for s in series]) if series[0].time.size > 0 else np.array([])

    multi_col = isinstance(series[0].data, dict)

    if multi_col:
        col_names = series[0].columns
        data_arrays = {c: np.concatenate([s.data[c] for s in series]) for c in col_names}
    else:
        data_concat = np.concatenate([s.data for s in series])

    # Remove duplicate iterations, keeping the last occurrence
    _, unique_idx = np.unique(iterations, return_index=True)

    # np.unique sorts by value, so unique_idx gives sorted-by-iteration indices.
    # For duplicates we want the *last* occurrence (latest restart takes precedence).
    # Reverse, unique, reverse back.
    _, last_idx = np.unique(iterations[::-1], return_index=True)
    last_idx = len(iterations) - 1 - last_idx
    last_idx = np.sort(last_idx)

    result = grace_timeseries.__new__(grace_timeseries)
    result.name = name
    result.iteration = iterations[last_idx].astype(int)
    result.time = times[last_idx] if times.size > 0 else np.array([])

    if multi_col:
        result.columns = col_names
        result.data = {c: data_arrays[c][last_idx] for c in col_names}
    else:
        result.columns = ["Value"]
        result.data = data_concat[last_idx]

    return result


def merge_scalar_dirs(dirs, pattern="*.dat"):
    """Merge scalar output from multiple directories (e.g. restart segments).

    Groups files by filename across directories, then merges each group.

    Args:
        dirs (list): List of directory paths containing scalar output,
            ordered by restart sequence (earliest first).
        pattern (str): Glob pattern for scalar files. Defaults to '*.dat'.

    Returns:
        grace_timeseries_array: Container with one merged timeseries per
            unique filename found across all directories.
    """
    # Group files by basename
    file_groups = defaultdict(list)
    for d in dirs:
        for fpath in sorted(glob.glob(os.path.join(d, pattern))):
            basename = os.path.basename(fpath)
            file_groups[basename].append(fpath)

    result = grace_timeseries_array()
    for basename, fpaths in file_groups.items():
        varname = os.path.splitext(basename)[0]
        result[varname] = merge_scalar_files(fpaths, name=varname)

    return result
