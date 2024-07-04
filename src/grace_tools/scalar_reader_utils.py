"""Utilities for reading scalar output from grace."""

import numpy as np
import os
import glob
import re 

from grace_tools.timeseries_utils import grace_timeseries_array, grace_timeseries

class grace_scalars_reader:
    """Reader class for grace scalar data.
    
    This class reads all files in a specified directory searching
    for grace scalar output. The output is separated according to 
    the type of reduction performed on the data, possible reductions 
    being 'maximum', 'minimum', 'norm2', 'integral'
    
    Attributes:
        maximum (grace_timeseries_array):
            Container of all timeseries computed with 'maximum' reduction.
        minimum (grace_timeseries_array):
            Container of all timeseries computed with 'minimum' reduction.
        norm2 (grace_timeseries_array):
            Container of all timeseries computed with 'norm2' reduction.
        integral (grace_timeseries_array):
            Container of all timeseries computed with 'integral' reduction.    
    """
    
    def __init__(self,dir):
        """Create a grace_scalars_reader object

        Args:
            dir (str): Directory where to search for scalar output.
        """
        patt = os.path.join(dir,"*.dat")
        files = glob.glob(patt)
        expr = re.compile("([\S]+)_([\S]+).dat")
        
        self.maximum  = grace_timeseries_array() 
        self.minimum  = grace_timeseries_array() 
        self.norm2    = grace_timeseries_array() 
        self.integral = grace_timeseries_array() 
        
        for ff in files:
            _,fname = os.path.split(ff)
            reg = expr.match(fname)
            red_type = reg.groups()[1]
            name = reg.groups()[0]
            if red_type == "max":
                self.maximum[name] = grace_timeseries(ff,name)
            elif red_type == "min":
                self.minimum[name] = grace_timeseries(ff,name)
            elif red_type == "norm2":
                self.norm2[name] = grace_timeseries(ff,name)
            elif red_type == "integral":
                self.integral[name] = grace_timeseries(ff,name)
            else:
                print(f"WARNING unrecognized reduction type {red_type} for variable {name}"
                      f" (file {ff})")
    
        
