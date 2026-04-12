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
    args = parser.parse_args()

    from grace_tools.scalar_reader_utils import grace_scalars_reader
    from grace_tools.gw_reader_utils import grace_gw_data
    from grace_tools.scalar_export import export_scalars_hdf5
    from grace_tools.simutils import grace_simulation

    # Use grace_simulation for full auto-detection (parfile, restarts, etc.)
    # but skip XMF generation since we only need scalars
    sim = grace_simulation(args.simdir, parfile=args.parfile)

    outfile = args.output
    if outfile is None:
        outfile = os.path.join(args.simdir, f"{sim.name}_scalars.h5")

    export_scalars_hdf5(sim.scalars, sim.gw, outfile)
    print(f"Exported scalars to {outfile}")


if __name__ == "__main__":
    main()
