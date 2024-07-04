"""Utilities for parsing profiling output from grace"""

import os 
import re 
import glob 
import numpy as np
from pyparsing import Word, alphas, alphanums, nums, Suppress, Combine, Optional, Group, OneOrMore, Regex, ZeroOrMore, ParseException, one_of


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
    
    def __init__(self,name,prof_results):
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
        self.__parse_profiling_data(prof_results)
    
    def __parse_profiling_data(self, prof_data):
        tmp_data = list()
        names = list() 
        for entry in prof_data:
            names.append(entry.kernel_name)
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
        self.kernel_name = entry.kernel_name
        self.rank        = entry.rank 
        self.dispatch    = entry.dispatch 
        self.gpu_id      = entry.gpu_id 
        self.queue_id    = entry.queue_id 
        self.queue_index = entry.queue_index 
        self.tid         = entry.tid 
        self.kernel_properties = {prop[0]: prop[1] for prop in entry.kernel_properties}
        self.timestamps  = {ts[0]: ts[1] for ts in entry.timestamps}
        self.counters    = {cnt[0]: cnt[1] for cnt in entry.counters}
        
        
def parse_profiling_file_body(body):
    """Parse the body of a gpu profiling output file.

    Args:
        body (str): 
            The body of the file being parsed

    Raises:
        ValueError: 
            If the file cannot be read or its format not recognized.

    Returns:
        pyparsing.ParseResults:
            Parsed file.
    """
    # Define the grammar
    identifier = Combine( Word(alphas) + Optional(one_of("_ -")) + Optional(Word(alphanums) + Optional(one_of("_ -"))))

    number = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
    number.setParseAction(lambda t: float(t[0]) if '.' in t[0] else int(t[0]))
        
    # Kernel info 
    kernel_name_data = Regex(r'".*?"')
    kernel_name = Group(Suppress(r'kernel-name(') + kernel_name_data + Suppress(r')')+Optional(Suppress(",")))
    rank = Group(Suppress("Rank[") + number + Suppress("],"))
    dispatch = Group(Suppress("dispatch[") + number + Suppress("],"))
    gpu_id = Group(Suppress("gpu_id(") + number + Suppress("),"))
    queue_id = Group(Suppress("queue_id(") + number + Suppress("),"))
    queue_index = Group(Suppress("queue_index(") + number + Suppress("),"))
    tid = Group(Suppress("tid(") + number + Suppress("),"))

    
    # Kernel properties
    kernel_properties_start = Suppress("Kernel Properties:")
    kernel_property = Group(identifier + Suppress('(') + number + Suppress(')') + Optional(Suppress(",")))

    # Timestamps
    timestamps_start = Suppress("Timestamps:")
    timestamp = Group(identifier + Suppress('(') + number + Suppress(')') + Optional(Suppress(",")))

    # Counters
    counters_start = Suppress("Counters:")
    counter = Group(identifier + Suppress('(') + number + Suppress(')') + Optional(Suppress(",")))

    # Single entry grammar
    entry = Group(
        kernel_name("kernel_name") +
        rank("rank") +
        dispatch("dispatch") +
        gpu_id("gpu_id") +
        queue_id("queue_id") +
        queue_index("queue_index") +
        tid("tid") +
        kernel_properties_start +
        OneOrMore(kernel_property)("kernel_properties") +
        timestamps_start +
        OneOrMore(timestamp)("timestamps") +
        counters_start +
        OneOrMore(counter)("counters")
    )
    
    # Grammar for multiple entries
    profiling_output = OneOrMore(entry)
    # Parsing the data
    try:
        result = profiling_output.parseString(data)
    except ParseException as pe:
        raise ValueError(f"Parsing error {pe}")
    return result 

def parse_profiling_file(fname):
    """Parse a gpu profiling output file.

    Args:
        fname (str):
            Path to the file being parsed.

    Raises:
        ValueError: 
            If the file cannot be read.

    Returns:
        str, pyparsing.ParseResults: 
            The name of the profiling region and the 
            parsed results contained in the file.
    """
    expr = re.compile("([\S]+)_gpu_counters_([\d]+).dat")
    rexpr = expr.match(fname)
    name = rexpr.groups()[0]    
    try:
        with open(fname,'r') as f:
            result = parse_profiling_file_body(f.read())
    except:
        raise ValueError(f"Can't read from file {fname}")
    
    return grace_gpu_profiling_data(name,result)

def is_file_binary()