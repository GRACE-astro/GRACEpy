
import argparse
import source_seal.package_utils as sseal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seal source tree into HDF5 archive with exclusion patterns.')
    parser.add_argument('source_dir', type=str, help='Path to the source directory to package')
    parser.add_argument('output_file', type=str, help='Path to the output HDF5 file')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='Patterns to exclude from the archive')
    parser.add_argument('--include-git', action='store_true', help='Include .git directory in the archive')
    args = parser.parse_args()

    exclude_patterns = args.exclude
    if not args.include_git:
        exclude_patterns += ['*.git*', '*build*']
    else:
        exclude_patterns += ['*build*']
    print(f"Sealing source tree {args.source_dir}, into archive {args.output_file}, with exclusions: {', '.join(exclude_patterns)}...")
    sseal.seal_source_tree(args.source_dir, args.output_file, exclude_patterns)
    print(f"Source tree sealed into {args.output_file} with exclusions: {', '.join(exclude_patterns)}.")
