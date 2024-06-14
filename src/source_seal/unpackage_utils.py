import os
import h5py
import numpy as np
import fnmatch
import subprocess
from tqdm import tqdm

def is_directory_empty(directory):
    return not any(os.scandir(directory))

def read_data_from_hdf5(hdf5_file_path, relative_path):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        if relative_path in hdf5_file:
            data = hdf5_file[relative_path][()]
            # Convert np.void to bytes
            if isinstance(data, np.void):
                data = bytes(data)
            return data
        else:
            raise FileNotFoundError(f"Dataset '{relative_path}' not found in the HDF5 file")

def unseal_source_tree(hdf5_file_path, output_dir, force=False):
    """
    Unseal source tree from HDF5 into directory.
    """
    if os.path.isdir(output_dir) and not(is_directory_empty(output_dir)) and not(force) :
        raise(ValueError, f"Directory {output_dir} is not empty. If you want to still write there, pass the --force argument.")
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            
    def recursively_unpack(group, current_path):
        pbar = tqdm(total=len(group.items()), desc=f"Unsealing source tree from HDF5 group {os.path.join(current_path,group.name)}")
        for key, item in group.items():
            item_path = os.path.join(current_path, key)
            if isinstance(item, h5py.Group):
                # Create directories for groups
                os.makedirs(item_path, exist_ok=True)
                recursively_unpack(item, item_path)
                pbar.update(1)
            else:
                # Read the data from the dataset
                data = item[()]
                if isinstance(data, np.void):
                    data = bytes(data)
                
                # Write the data to the output file
                with open(item_path, 'wb') as f:
                    f.write(data)
                pbar.update(1)

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        recursively_unpack(hdf5_file, output_dir)