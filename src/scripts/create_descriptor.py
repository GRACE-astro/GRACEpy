import argparse
import grace_tools.xmf_utils as xmf 

def main():
    parser = argparse.ArgumentParser(description='Generate xmf descriptor file for grace hdf5 output.')
    parser.add_argument('input_dir', type=str, help='Path containing grace HDF5 output to be parsed.')
    parser.add_argument('output_file', type=str, help='Output xmf file.')
    args = parser.parse_args()
    xmf.write_xmf_file(args.output_file,args.input_dir)
    
if __name__=="__main__":
    main()