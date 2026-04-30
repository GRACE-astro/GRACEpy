import argparse
import numpy as np
from grace_tools.vtk_reader_utils import grace_xmf_reader
from grace_tools.mhd_utils import (export_uniform, DERIVED_QUANTITIES,
                                    GRID_DERIVED_QUANTITIES)


def main():
    parser = argparse.ArgumentParser(
        description="Resample GRACE data onto a uniform grid and export to HDF5.",
        epilog="Example:\n"
               "  export_uniform volume.xmf snapshot.h5 \\\n"
               "      -e -50 50 -50 50 -50 50 \\\n"
               "      -n 256 -t 500.0 -d b_squared poynting_flux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("descriptor", type=str,
                        help="Path to the XMF descriptor file.")
    parser.add_argument("output", type=str,
                        help="Output HDF5 file path.")
    parser.add_argument("-e", "--extents", type=float, nargs="+", required=True,
                        help="Grid extents: xmin xmax ymin ymax [zmin zmax].")
    parser.add_argument("-n", "--resolution", type=int, nargs="+", default=[256],
                        help="Number of grid points per direction. "
                             "A single value is used for all directions, "
                             "or provide one per direction (default: 256).")
    parser.add_argument("-t", "--time", type=float, default=None,
                        help="Simulation time to extract. "
                             "Defaults to last available time.")
    all_derived = sorted({**DERIVED_QUANTITIES, **GRID_DERIVED_QUANTITIES})
    parser.add_argument("-d", "--derived", type=str, nargs="+", default=None,
                        choices=all_derived,
                        help="Derived MHD quantities to compute.")
    parser.add_argument("--exclude", type=str, nargs="+", default=None,
                        help="Variable names to exclude from the export.")

    args = parser.parse_args()

    # ── validate extents ──────────────────────────────────────────────────
    if len(args.extents) not in (4, 6):
        parser.error("extents must be 4 values (2-D) or 6 values (3-D)")

    # ── resolve resolution ────────────────────────────────────────────────
    ndim = len(args.extents) // 2
    if len(args.resolution) == 1:
        res = args.resolution[0]
    elif len(args.resolution) == ndim:
        res = tuple(args.resolution)
    else:
        parser.error(f"--resolution expects 1 or {ndim} values, got {len(args.resolution)}")

    # ── open reader ───────────────────────────────────────────────────────
    reader = grace_xmf_reader(args.descriptor)

    # ── pick time ─────────────────────────────────────────────────────────
    if args.time is not None:
        time = args.time
    else:
        time = reader.available_times()[-1]
        print(f"No time specified, using last available: t = {time}")

    # ── export ────────────────────────────────────────────────────────────
    outfile = args.output
    if not outfile.endswith(".h5"):
        outfile += ".h5"

    print(f"Resampling {args.descriptor} at t = {time}")
    print(f"  extents:    {args.extents}")
    print(f"  resolution: {res}")
    if args.derived:
        print(f"  derived:    {args.derived}")

    export_uniform(
        reader, time, outfile,
        extents=tuple(args.extents),
        resolution=res,
        derived=args.derived,
        exclude=args.exclude,
    )

    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
