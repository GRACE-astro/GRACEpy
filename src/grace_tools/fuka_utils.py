"""Utilities for parsing FUKA initial data info files."""

import os
import re


def _parse_fuka_braces(text):
    """Parse FUKA's nested brace-delimited format into a dict.

    Each block is of the form:
        key value
    or:
        key { ... nested ... }
    """
    result = {}
    i = 0
    tokens = text.split()
    while i < len(tokens):
        token = tokens[i]
        if token == "}":
            break
        if i + 1 < len(tokens) and tokens[i + 1] == "{":
            # Nested block: consume everything between { and matching }
            depth = 0
            start = i + 2
            j = start
            while j < len(tokens):
                if tokens[j] == "{":
                    depth += 1
                elif tokens[j] == "}":
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            inner_text = " ".join(tokens[start:j])
            result[token] = _parse_fuka_braces(inner_text)
            i = j + 1
        else:
            # key value pair
            if i + 1 < len(tokens):
                result[token] = tokens[i + 1]
                i += 2
            else:
                i += 1
    return result


def parse_fuka_info(filepath):
    """Parse a FUKA .info file and return (Madm, omega0).

    Madm is the total ADM mass, computed as the sum of all component
    ``madm`` values found in the info file (e.g. ns1.madm + ns2.madm
    for a BNS, or bh.madm + ns.madm for a BHNS).

    omega0 is the initial orbital frequency (``global_omega``).

    Args:
        filepath (str): Path to the FUKA .info file.

    Returns:
        tuple: (Madm, omega0) as floats.

    Raises:
        FileNotFoundError: If the info file does not exist.
        ValueError: If required fields cannot be found.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"FUKA info file not found: {filepath}")

    with open(filepath, "r") as f:
        text = f.read()

    parsed = _parse_fuka_braces(text)

    # The top-level structure is: binary { ... }
    binary = parsed.get("binary")
    if binary is None:
        raise ValueError(f"No 'binary' block found in {filepath}")

    # Extract orbital frequency
    omega_str = binary.get("global_omega")
    if omega_str is None:
        raise ValueError(f"No 'global_omega' found in {filepath}")
    omega0 = float(omega_str)

    # Sum all component ADM masses (ns1, ns2, bh, etc.)
    # Component blocks are identified by having a 'madm' key.
    Madm = 0.0
    n_components = 0
    for key, value in binary.items():
        if isinstance(value, dict) and "madm" in value:
            Madm += float(value["madm"])
            n_components += 1

    if n_components == 0:
        raise ValueError(f"No component masses (madm) found in {filepath}")

    return Madm, omega0
