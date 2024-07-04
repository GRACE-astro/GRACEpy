"""Utilities for processing grace timeseries."""

import numpy as np
import os
import glob
import re 

class grace_timeseries_array:
    """Array of grace timeseries
    
    This class is a container designed to hold 
    timeseries data for different variables coming 
    from grace.
    
    Methods:
        available_vars():
            Return a list of available vars in this container.
        __getitem__(key):
            Retrieve an item at the specified key.
        __setitem__(key, value):
            Set the item at the specified key to the given value.
    """
    
    def __init__(self):
        """Initialize an empty grace timeseries container."""
        self.__data = dict() 
    
    def __getitem__(self,key: str):
        """Get a grace_timeseries corresponding to the requested variable. 
        

        Args:
            key (str): 
                Name of requested variable.

        Returns:
            grace_timeseries: 
                The timeseries corresponding to the requested variable
        """
        return self.__data[key]
    
    def __setitem__(self,key: str, value):
        """Set a grace_timeseries corresponding to the requested variable. 
        

        Args:
            key (str): 
                Name of requested variable.
            value (grace_timeseries):
                The timeseries to be added to this array (or modified).
        """
        self.__data[key] = value
    
    def available_vars(self):
        """Query available variables in this container.
        
        Returns:
            list:
                A list of all available variable names.
        """
        return self.__data.keys()

class grace_timeseries:
    """Class representing a timeseries of grace data.
    
    Attributes:
        iteration (np.array):
            Array containing iterations at which data is available.
        time (np.array):
            Array containing times at which data is available.
        data (np.array):
            Array containing the timeseries data.
        name (str):
            Name of the variable.
    """
    
    def __init__(self,file,name):
        """Initialize a grace_timeseries from data in a file."""
        self.name = name
        if not os.path.isfile(file):
            raise ValueError(f"File {file} does not exist or is not readable.")
        self.iteration,self.time,self.data = np.loadtxt(file,unpack=True)
        