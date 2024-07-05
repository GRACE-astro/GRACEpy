import argparse
import os 
import glob 

import grace_tools.parsing_utils as gtp 

def main():
    parser = argparse.ArgumentParser(description='Convert binary output of kokkos kernels timing tool to ascii.')
    parser.add_argument('input_dir', type=str, help='Path containing the Kokkos kernels timings files to be converted.')
    args = parser.parse_args()
    for ffile in glob.iglob(os.path.join(args.input_dir,"gpu[0-9]*-[0-9]*.dat")):
        path, fname = os.path.split(ffile)
        name, extension = os.path.splitext(fname)
        with open(os.path.join(path,name+".asc"), "w") as f:
            f.write(gtp.execute_kp_reader(ffile))
    
if __name__=="__main__":
    main()