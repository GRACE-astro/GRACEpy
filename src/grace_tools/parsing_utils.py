"""Utilities for parsing profiling output from grace"""

import os 
import re 
import glob 
import numpy as np
import subprocess
from pyparsing import Word, alphas, alphanums, nums, Suppress, Combine, Optional, Group, OneOrMore, Regex, ZeroOrMore, ParseException, oneOf, restOfLine, LineEnd

        
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
    identifier = Combine( Word(alphas) + Optional(oneOf("_ -")) + Optional(Word(alphanums) + Optional(oneOf("_ -"))))

    number = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
    number.setParseAction(lambda t: float(t[0]) if '.' in t[0] else int(t[0]))
        
    # Kernel info 
    iteration = Group(Suppress("iteration(") + number("iter") +  Suppress(r')')+Optional(Suppress(",")))
    kernel_name_data = Regex(r'".*?"')
    kernel_name = Group(Suppress(r'kernel-name(') + kernel_name_data + Suppress(r')')+Optional(Suppress(",")))
    dispatch = Group(Suppress("dispatch[") + number + Suppress("],"))
    gpu_id = Group(Suppress("gpu_id(") + number + Suppress("),"))
    queue_id = Group(Suppress("queue_id(") + number + Suppress("),"))
    queue_index = Group(Suppress("queue_index(") + number + Suppress("),"))
    tid = Group(Suppress("tid(") + number + Suppress("),"))

    
    # Kernel properties
    kernel_properties_start = Suppress("Kernel Properties:")
    kernel_property = Group(identifier + Suppress('(') + number + Suppress(')') + Optional(Suppress(",")))

    # Timestamps
    timestamps_start = Suppress("Timestamps (in nanoseconds):")
    timestamp = Group(identifier + Suppress('(') + number + Suppress(')') + Optional(Suppress(",")))

    # Counters
    counters_start = Suppress("Counters:")
    counter = Group(identifier + Suppress('(') + number + Suppress(')') + Optional(Suppress(",")))

    # Single entry grammar
    entry = Group(
        iteration("iteration") + 
        kernel_name("kernel_name") +
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
        result = profiling_output.parseString(body)
    except ParseException as pe:
        raise ValueError(f"Parsing error {pe}")
    return result 

def parse_profiling_file(fpath):
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
    _,fname = os.path.split(fpath)
    rexpr = expr.match(fname)
    name = rexpr.groups()[0] 
    rank = rexpr.groups()[1]  
    try:
        with open(fpath,'r') as f:
            result = parse_profiling_file_body(f.read())
    except:
        raise ValueError(f"Can't read from file {fpath}")
    
    return (name,rank,result)

def is_file_binary(filepath):
    """
    Check if a file is binary.

    Args:
        filepath (str): 
            The path to the file to check.

    Returns:
        bool: 
            True if the file is binary, False otherwise.
    """
    try:
        with open(filepath, 'rb') as file:
            chunk = file.read(1024)
            if b'\0' in chunk:  # Null byte is a good indicator of binary file
                return True
            text_characters = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
            return not all(byte in text_characters for byte in chunk)
    except Exception as e:
        print(f"Error checking if file is binary: {e}")
        return False
    
def execute_kp_reader(filepath):
    """
    Execute kp_reader on the binary file and parse the output.

    Args:
        filepath (str): 
            The path to the binary file.

    Returns:
        str: 
            The parsed text output from kp_reader.
    """
    try:
        result = subprocess.run(['kp_reader', filepath], capture_output=True, text=True)
        result.check_returncode()  # Raise an error if the command failed
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"kp_reader failed: {e}")
        return None
    except FileNotFoundError:
        print("kp_reader not found. Make sure it is installed and in your PATH.")
        return None

def parse_kp_reader_output(output):
    """Parse output from kp_reader regarding the kokkos_kernel_timer tool

    Args:
        output (str): 
            Output from kp_reader

    Returns:
        pyparsing.ParseResults: 
            The parsed output.
    """
    # Define basic elements
    number = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
    number.setParseAction(lambda t: float(t[0]) if '.' in t[0] else int(t[0]))
    word = Word(alphas + ":-_[]")
    identifier = Combine(Word(alphas + "_::[]") + Optional(restOfLine))

    # Define patterns for the output sections
    header = Suppress("(Type)") + Suppress("Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time")
    separator = Suppress("-" * 73)

    # Define patterns for regions
    region_name = Suppress("-") + restOfLine("name")
    region_data = Group(Suppress("(REGION)") + number("total_time") + number("call_count") + number("avg_time_per_call") +
                        number("total_time_in_kernels") + number("total_program_time"))

    region = Group(region_name + region_data)
    regions = Suppress("Regions:") + OneOrMore(region)

    # Define patterns for kernels
    kernel_name = Suppress("-") + restOfLine("name")
    kernel_data = Group(Suppress("(ParFor)") + number("total_time") + number("call_count") + number("avg_time_per_call") +
                        number("total_time_in_kernels") + number("total_program_time")) | \
                Group(Suppress("(ParRed)") + number("total_time") + number("call_count") + number("avg_time_per_call") +
                        number("total_time_in_kernels") + number("total_program_time"))

    kernel = Group(kernel_name + kernel_data)
    kernels = Suppress("Kernels:") + OneOrMore(kernel)

    # Define patterns for summary
    summary_header = Suppress("Summary:")
    summary_lines = Group(
        Suppress("Total Execution Time (incl. Kokkos + non-Kokkos):") + number("total_exec_time") + Suppress("seconds") + Suppress(LineEnd()) +
        Suppress("Total Time in Kokkos kernels:") + number("total_time_kokkos_kernels") + Suppress("seconds")+ Suppress(LineEnd()) +
        Suppress("-> Time outside Kokkos kernels:") + number("time_outside_kokkos_kernels") + Suppress("seconds")+ Suppress(LineEnd()) +
        Suppress("-> Percentage in Kokkos kernels:") + number("percentage_in_kokkos_kernels") + Suppress("%") + Suppress(LineEnd()) +
        Suppress("Total Calls to Kokkos Kernels:") + number("total_calls_kokkos_kernels")
    )
    summary = summary_header + summary_lines

    # Combine the grammar
    profiling_output = header + separator + regions("regions") + separator + kernels("kernels") + separator + summary("summary") + separator
    # Parsing the data
    try:
        result = profiling_output.parseString(output)
        #print(result.dump())
    except ParseException as pe:
        print("Parsing error:", pe)
    return result 