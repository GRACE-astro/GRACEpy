"""Utilities for reading profiling output from grace."""

import numpy as np
import os
import glob
import re 

from grace_tools.timeseries_utils import grace_timeseries_array, grace_timeseries
from grace_tools.parsing_utils import *

class grace_profiling_data:
    """Container for grace profiling data"""
    
    def __init__(self,profiling_dir,simdir):
        """_summary_

        Args:
            profiling_dir (_type_): _description_
            simdir (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.gpu = []
        for ff in glob.iglob(os.path.join(profiling_dir,"*gpu_counters_*.dat")):
            self.gpu.append(grace_gpu_profiling_data(*parse_profiling_file(ff)))
        
        self.cpu = grace_timeseries_array()
        patt = re.compile("([\S]+)_host_timers.dat")
        for ff in glob.iglob(os.path.join(profiling_dir,"*host_timers.dat")):
            _,fname = os.path.split(ff)
            name = patt.match(ff).groups()[0]
            self.cpu[name] = grace_timeseries(ff,name)
        # Search for kokkos kernel timings output in ascii format
        self.kokkos_timers = []
        for ff in glob.iglob(os.path.join(simdir,"gpu[0-9]*-[0-9]*.asc")):
            with open(ff,"r") as f:
                self.kokkos_timers.append(
                    kokkos_timings_data(parse_kp_reader_output(f.read()))
                )
        # TODO: what about memory high water and other tools? 
        return 
        

class grace_gpu_profiling_data:
    """Container for gpu low-level profiling data from grace.
    
    This contains one entry per profiling region. The entries are 
    stored in a list that can be accessed via normal indexing on
    the grace_gpu_profiling_data object. Since kernel names are 
    usually long and complicated they are stored separately. Each 
    entry in the list is a grace_gpu_profiling_entry_hip/cuda.
    
    Attributes:
        name:
            Name of the profiling region this data comes from.
    Methods:
        __getitem__(index):
            Get the data entry corresponding to the given index.
        __len__():
            Get the number of entries
        kernel_name(index):
            Get the kernel name corresponding to the entry at a given index.
    """
    
    def __init__(self,name,rank,prof_results):
        """Create a grace_gpu_profiling_data instance.

        Args:
            name (str): 
                Name of the profiling region.
            prof_results (list): 
                List containing the profiling results.
        """
        self.__data = list()
        self.__kernel_names = list()
        self.name = name 
        self.rank = rank
        self.__parse_profiling_data(prof_results)
    
    def __parse_profiling_data(self, prof_data):
        tmp_data = list()
        names = list() 
        for entry in prof_data:
            names.append(entry.kernel_name[0].strip('"'))
            tmp_data.append(grace_gpu_profiling_entry_hip(entry))
        names = np.unique(np.array(names))
        self.__kernel_names = names.tolist() 
        for name in names:
            tmplist = list() 
            for d in tmp_data:
                if d.kernel_name == name:
                    tmplist.append(d)
            self.__data.append(tmplist)
            
    def __getitem__(self,index: int):
        """Get a profiling entry at the provided index.
        
        Args:
            index (int):
                Index of requested entry.
        Returns:
            list of grace_gpu_profiling_entry_hip/cuda:
                The data entries pertaining to a specific kernel.
        """
        return self.__data[index]
    
    def __len__(self):
        """Get number of distinct kernels sampled in this session."""
        return len(self.__data)

    def kernel_name(self,index: int):
        """Get kernel name of entry at index."""
        return self.__kernel_names[index]
        
            
class grace_gpu_profiling_entry_hip:    
    """Profling entry representing one kernel in one GPU profiling session.
    
        Attribues:
            kernel_name (str):
                The name of the kernel being sampled.
            rank (int):
                The MPI rank collecting this output.
            dispatch (int):
                Number of this kernel dispatch.
            gpu_id (int):
                Unique GPU agent identifier.
            queue_id (int):
                Unique dispatch queue identifier.
            queue_index (int):
                Index in dispatch queue.
            tid (int):  
                Thread ID of the Host thread initiating the dispatch.
            kernel_properties (dict):
                Properties of the kernel:
                    grd: Grid size.
                    wgr: Workgroup size.
                    lds: Local data store size (per group).
                    scr: Scratch memory (per kernel).
                    arch_vgpr: Number of vector registers used.
                    accum_vgpr: Number of accumulated vector register used.
                    sgpr: Number of scalar registers used.
                    wave_size: Number of threads per wavefront.
                    sig: Signature of the kernel.
                    obj: Internal object identifier.
            timestamps (dict):
                Timestamps in nanoseconds:
                    time-begin: Time at which kernel execution began.
                    time-end  : Time at which kernel execution ended.
                    duration  : Duration of kernel execution in nanoseconds.
            counters (dict):
                Requested hardware counters. Dictionary contains counter_name: counter_value.
    """
    
    def __init__(self,entry):
        """Initialize a grace_gpu_profiling_entry_hip from a parsed profiling entry."""
        self.kernel_name = entry.kernel_name[0].strip('"')
        self.rank        = entry.rank 
        self.dispatch    = entry.dispatch 
        self.gpu_id      = entry.gpu_id 
        self.queue_id    = entry.queue_id 
        self.queue_index = entry.queue_index 
        self.tid         = entry.tid 
        self.kernel_properties = {prop[0]: prop[1] for prop in entry.kernel_properties}
        self.timestamps  = {ts[0]: ts[1] for ts in entry.timestamps}
        self.counters    = {cnt[0]: cnt[1] for cnt in entry.counters}
        
class kokkos_timings_data:
    """Container for parsed data from kokkos kernel timings tool. 
    
    All times are in seconds.
    
    Attributes:
        regions (list):
            A list of profiling regions containing coarse information.
            Each region is a dictionary whose key is the name and value 
            is a second dictionary containing the timings
        kernels (list):
            A list of all kernel instances during the simulation. Each 
            element of the list is a dictionary whose key is the name 
            and whose value is a dictionary containing timing data.
        summary (dict):
            A dictionary containing summary information about time spent 
            inside and outside Kokkos kernels.
    """
    
    def __init__(self,parsed_data):
        """Construct a kokkos_timings_data instance given a kokkos_timings parsed output."""            
        self.regions = []
        for region in parsed_data.regions:
            self.regions.append({region[0]: region[1].asDict()})
        self.kernels = []
        for kernel in parsed_data.kernels:
                self.kernels.append({kernel.name: kernel[1].asDict()})
        self.summary = dict() 
        print(parsed_data.summary[0])
        keys = ["total_exec_time", "total_time_kokkos_kernels","time_outside_kokkos_kernels","percentage_in_kokkos_kernels","total_calls_kokkos_kernels"]
        for i,val in enumerate(parsed_data.summary[0]):
            self.summary[keys[i]] = val     
        