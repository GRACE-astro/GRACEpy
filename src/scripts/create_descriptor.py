import argparse
import grace_tools.xmf_utils as xmf 

def main():
    """
    Main function to generate an XMF descriptor file for GRACE HDF5 output.
    This function parses command-line arguments to get the input directory
    containing GRACE HDF5 output files and the output file path for the XMF file.
    It then calls the `write_xmf_file` function to generate the XMF file.
    Command-line arguments:
    input_dir (str): Path containing GRACE HDF5 output to be parsed.
    output_file (str): Output XMF file path.
    Example usage:
    python create_descriptor.py /path/to/input_dir /path/to/output_file
    """
    parser = argparse.ArgumentParser(description='Generate xmf descriptor file for grace hdf5 output.')
    parser.add_argument('input_dir', type=str, help='Path containing grace HDF5 output to be parsed.')
    parser.add_argument('output_file', type=str, help='Output xmf file.')
    parser.add_argument('--mode', 
                        type=str, default='auto', 
                        choices=['auto', 'temporal', 'spatial'],  # only allow these values
                        help="Can be 'auto' 'temporal' or 'spatial' depending on the kind of output processed.")
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output.')
    parser.add_argument('--filter', type=str, default="*.h5", help='Filter files in directory.')
    args = parser.parse_args()
    outfile = args.output_file
    if not outfile.endswith(".xmf"):
        outfile += ".xmf"
    if args.verbose:
        print(f"Generating XMF descriptor for {args.input_dir}")
        print(f"Output file: {outfile}")

    xmf.write_xmf_file(outfile,args.input_dir,args.mode,verbose=args.verbose,filter=args.filter)
    
if __name__=="__main__":
    main()