"""CLI tool to export merged scalar data from a GRACE simulation to HDF5."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Export merged scalar and GW data from a GRACE simulation to a single HDF5 file."
    )
    parser.add_argument("simdir", type=str,
                        help="Path to the simulation directory.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output HDF5 file path. Default: <simdir>/<simname>_scalars.h5")
    parser.add_argument("--parfile", type=str, default=None,
                        help="Parameter file (auto-detected if not specified).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress messages.")
    args = parser.parse_args()

    from grace_tools.simutils import grace_simulation

    sim = grace_simulation(args.simdir, parfile=args.parfile, verbose=args.verbose)
    sim.export_scalars(args.output)


if __name__ == "__main__":
    main()
