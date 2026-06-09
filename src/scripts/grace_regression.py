"""CLI to create and compare GRACE regression-test artifacts.

``create`` bundles a sealed source tree, the simulation's scalar + GW results,
and run provenance into a single self-contained HDF5 artifact. ``compare``
diffs new results (a simulation directory or an exported scalars HDF5) against
such a baseline and exits nonzero on regression.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Create and compare GRACE regression artifacts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- create ----
    c = sub.add_parser("create", help="Build a self-contained regression artifact.")
    c.add_argument("simdir", help="Path to the simulation directory.")
    c.add_argument("output_file", help="Output artifact HDF5 path.")
    c.add_argument("--source-dir", dest="source_dir", required=True,
                   help="Source tree to seal (with its git commit).")
    c.add_argument("--parfile", default=None,
                   help="Parameter file (auto-detected if not specified).")
    c.add_argument("--exclude", nargs="*", default=None,
                   help="Glob patterns to exclude from the sealed source tree.")
    c.add_argument("--include-git", action="store_true",
                   help="Include the .git directory in the sealed source tree.")
    c.add_argument("--machine", default=None,
                   help="grace_pilot machine YAML for the logical machine name.")
    c.add_argument("-v", "--verbose", action="store_true",
                   help="Print progress messages.")

    # ---- compare ----
    k = sub.add_parser("compare", help="Compare new results against a baseline artifact.")
    k.add_argument("baseline", help="Baseline regression artifact (.h5).")
    k.add_argument("new", help="New results: a simulation dir OR an exported scalars .h5.")
    k.add_argument("--parfile", default=None,
                   help="Parameter file for the new results (only used for a simdir).")
    k.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance.")
    k.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance.")
    k.add_argument("--include-performance", action="store_true",
                   help="Include performance/timing metrics in the comparison.")
    k.add_argument("--strict", action="store_true",
                   help="Treat extra keys (present only in new) as a failure.")
    k.add_argument("--ignore-coords", action="store_true",
                   help="Skip iteration/time/t_ret coordinate arrays.")
    k.add_argument("--json", action="store_true", help="Emit the report as JSON.")

    args = parser.parse_args()

    if args.command == "create":
        from regression.artifact_utils import create_artifact

        exclude_patterns = list(args.exclude) if args.exclude else []
        if not args.include_git:
            exclude_patterns += ["*.git*", "*build*"]
        else:
            exclude_patterns += ["*build*"]

        create_artifact(
            args.simdir, args.source_dir, args.output_file,
            parfile=args.parfile,
            exclude_patterns=exclude_patterns,
            machine_config=args.machine,
            verbose=args.verbose,
        )
        print(f"Regression artifact written to {args.output_file}")

    elif args.command == "compare":
        from regression.compare_utils import compare_against_artifact

        report = compare_against_artifact(
            args.baseline, args.new,
            parfile=args.parfile,
            rtol=args.rtol, atol=args.atol,
            include_performance=args.include_performance,
            ignore_coords=args.ignore_coords,
        )
        if args.json:
            print(report.to_json(strict=args.strict))
        else:
            print(report.summary(strict=args.strict))
        sys.exit(0 if report.passed(strict=args.strict) else 1)


if __name__ == "__main__":
    main()
