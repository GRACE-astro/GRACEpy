"""This module contains utilities to handle simulations performed by grace."""

import numpy as np
import glob 
import re 
import os 
import yaml 

import grace_tools.vtk_reader_utils as gtv 
import grace_tools.xmf_utils as gtx 
import grace_tools.scalar_reader_utils as gts 
import grace_tools.profiling_reader_utils as gtp

class grace_simulation:
    """Class designed to aid the analysis of data from a grace simulation.
    
    This class parses the simulation parameter file and searches for hdf5 
    and scalar output.
    
    Attributes:
        name (str):
            The name of this simulation.
        xyz (grace_xmf_reader): 
            The volume output reader. Can be used 
            to query for available vars/times and to extract the data 
            or slice it for post-processing purposes. Refer to the class
            documentation for additional information.
        scalars (grace_scalars_reader):
            The scalar output reader. Contains all 
            available timeseries data. Refer to its documentation for  
            usage instructions.
        prof (grace_profiling_reader):
            The profiling output reader. Contains 
            all available profiling information for this simulation.  
            Refer to its documentation for usage instructions.
    """
    
    def __init__(self,simdir: str, parfile=None, ppdir="./plots"):
        """Create a grace_simulation object.
        
        This is used to handle output data from a grace simulation.

        Args:
            simdir (str): 
                Directory where the simulation is found.
            parfile (str, optional): 
                Parameter file of the simulation (full path or relative to where 
                this function is called). If None it will be looked for in the 
                simulation directory. In that case only one yaml file should be in 
                the directory and that will be interpreted to be the parameter file. 
                Defaults to None.
        """
        self.simdir = simdir 
        if not os.path.isdir(self.simdir):
            raise ValueError(f"Simulation directory {simdir} doesn't exist or cannot be read.")
        if parfile is None: 
            parfile = self.__find_parfile()
        self.__parse_parfile(parfile)
        self.bdir = ppdir 
        self.descdir = os.path.join(self.bdir,"descriptors")
        if not os.path.isdir(ppdir):
            os.mkdir(self.bdir)
        if not os.path.isdir(os.path.join(self.bdir,"descriptors")):
            os.mkdir(os.path.join(self.bdir,"descriptors"))
        gtx.write_xmf_file(os.path.join(self.descdir,"volume_descriptor.xmf"),
                           self.volume_out_dir)
        self.xyz     = gtv.grace_xmf_reader(os.path.join(self.descdir,"volume_descriptor.xmf"))
        self.scalars = gts.grace_scalars_reader(self.scalar_out_dir)
    
    
    def __find_parfile(self):
        patt = os.path.join(self.simdir,"*.yaml")
        flist = glob.glob(patt)
        if len(flist) == 0:
            print("WARNING: No parfile specified and the simulation directory"
                  f" {self.simdir} does not contain one, all paths set to default values.")
            return None 
        if len(flist) > 1: 
            print("WARNING: No parfile specified and the simulation directory"
                  f" contains more than one yaml file, all paths set to default values.")
            return None
        return flist[0]
    
    def __parse_parfile(self,parfile):
        if parfile is None:
            self.volume_out_dir = os.path.join(self.simdir,"output_volume")
            self.scalar_out_dir = os.path.join(self.simdir,"output_scalar")
            self.name = "grace"
            return
        with open(parfile,'r') as f:
            config = yaml.safe_load(f)
        # Find output directories in parameter file 
        try:
            self.volume_out_dir = os.path.join(self.simdir,config["IO"]["volume_output_base_directory"])
        except:
            self.volume_out_dir = os.path.join(self.simdir,"output_volume")
        try:
            self.scalar_out_dir = os.path.join(self.simdir,config["IO"]["scalar_output_base_directory"])
        except:
            self.scalar_out_dir = os.path.join(self.simdir,"output_scalar")
        try: 
            self.name = str(config["name"])
        except:
            self.name = "grace"
        return 