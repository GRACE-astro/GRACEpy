import argparse
import source_seal.unpackage_utils as sseal

def main():
    parser = argparse.ArgumentParser(description='Unseal source package in specified directory.')
    parser.add_argument('source_file', type=str, help='Path to the input HDF5 file.')
    parser.add_argument('output_dir', type=str, help='Path to the source directory where the source archive should be unsealed.')
    parser.add_argument('--force', action='store_true', help='Force unsealing even if directory is not empty')
    args = parser.parse_args()

    if args.force : 
        forced_str = "forced"
    else:
        forced_str = "not forced"
    print(f"Unsealing source tree {args.source_file}, at {args.output_dir} ({forced_str})...")
    sseal.unseal_source_tree(args.source_file, args.output_dir, args.force)
    print(f"Source tree unsealed from {args.source_file} to {args.output_dir}.")
    
if __name__ == "__main__":
    main()
