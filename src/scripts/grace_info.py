import argparse
import yaml 
import numpy as np


def parse_parfile(fname: str):
     with open(fname,'r') as f:
         return yaml.safe_load(f)
     
def is_power_of_two(n):
    """Returns True if n is a power of two, otherwise False."""
    return (n != 0) and (n & (n - 1)) == 0

def validate_params(pars):
    try:
        # Extract parameters
        nx = pars['amr']['npoints_block_x']
        ny = pars['amr']['npoints_block_y']  
        nz = pars['amr']['npoints_block_z']

        # Check if they are integers
        if not all(isinstance(i, int) for i in [nx, ny, nz]):
            raise ValueError("Parameters nx, ny, and nz must be integers.")

        # Check if they are powers of two
        if not all(is_power_of_two(i) for i in [nx, ny, nz]):
            print("WARNING: The number of points per block per direction"
                  " should really be powers of w for optimal performance.")

        # Check if they are in the range of 16 to 64
        if not all(16 <= i <= 64 for i in [nx, ny, nz]):
            print("WARNING: The number of points per block per direction should usually"
                  " be in the range 16 to 64 for optimal performance.")
    except KeyError as e:
        print(f"Missing parameter in YAML: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
    return nx,ny,nz 

def print_grid_information(fname: str):
    pars = parse_parfile(fname)
    
    nx,ny,nz = validate_params(pars)
    
    ncells_block = nx*ny*nz
    
    xmin = pars['amr']['xmin']
    xmax = pars['amr']['xmax']
    ymin = pars['amr']['ymin']
    ymax = pars['amr']['ymax']
    zmin = pars['amr']['zmin']
    zmax = pars['amr']['zmax']
    
    L = np.array(
        [xmax-xmin,ymax-ymin,zmax-zmin]
    )
    
    lmin = np.amin(L)
    
    Lnorm = np.array( [ int(np.floor(Li/lmin)) for Li in L ] ) 
    
    L = Lnorm * lmin 
    
    ntrees = np.prod(Lnorm)
    
    reflevel = pars['amr']['initial_refinement_level']
    
    nquads_tree = 8**reflevel 

    
    tot_ncells = nquads_tree * ncells_block * ntrees 
    
    ncells_dir = np.array([nx * 2**reflevel * Lnorm[0], 
                           ny * 2**reflevel * Lnorm[1], 
                           nz * 2**reflevel * Lnorm[2] ] )
    
    dx = L / ncells_dir 

    
    name = pars["name"]
    ngz = pars['amr']['n_ghostzones']
    print(f"""
          *********************************************************************************************
          Grid information for GRACE run "{name}":
          ---------------------------------------------------------------------------------------------
             Grid extent (code units):      [{L[0]:.3f}, {L[1]:.3f}, {L[2]:.3f}]
             Base resolution (code units):  [{dx[0]:.2g} {dx[1]:.2g} {dx[2]:.2g}]
             Number of cells per direction: [{ncells_dir[0]:d} {ncells_dir[1]:d} {ncells_dir[2]:d}]
             Total number of oct-trees:     {ntrees}
             Total number of quadrants:     {nquads_tree*ntrees}
             Total number of cells:         {np.prod(ncells_dir)}
             Number of ghostzones:          {ngz}
          *********************************************************************************************
          """)

    
    


def main():
    parser = argparse.ArgumentParser(description='Print information about a GRACE build or parfile.')
    parser.add_argument('--grid_info', action='store_true')
    parser.add_argument('--parfile', type=str)
    parser.add_argument('--twod',action="store_true")
    args = parser.parse_args()
    
    if args.grid_info:
        print_grid_information(args.parfile)
        return 
    
if __name__=="__main__":
    main()