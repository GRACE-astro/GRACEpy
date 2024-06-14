import os
import h5py
import numpy as np
import fnmatch
import subprocess
from tqdm import tqdm


def should_exclude(path, exclude_patterns):
    """
    Check if a given path matches any of the exclude patterns.
    """
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False

def add_directory_to_hdf5(hdf5_file, source_dir, exclude_patterns):
    """
    Package a directory into the hdf5 database.
    """
    all_files = []
    n_excluded = 0 
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if should_exclude(file_path, exclude_patterns):
                n_excluded +=1 
                continue
            all_files.append(file_path)
    print(f"Total number of files {len(all_files)}, excluded {n_excluded}")
    pbar = tqdm(total=len(all_files), desc="Adding files to HDF5")
    for file_path in all_files:
        pbar.set_postfix(file=os.path.basename(file_path))
        with open(file_path, 'rb') as f:
            data = f.read()
        if len(data) == 0:
            pbar.update(1)
            continue
        relative_path = os.path.relpath(file_path, source_dir)
        
        hdf5_file.create_dataset(relative_path, data=np.void(data))
        pbar.update(1)
    pbar.close()


def get_git_info(source_dir):
    """
    Retrieve the current commit hash and unstaged changes.
    """
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source_dir).strip().decode('utf-8')
        unstaged_changes = subprocess.check_output(
            ['git', 'diff'], cwd=source_dir).strip().decode('utf-8')
    except subprocess.CalledProcessError as e:
        commit_hash = "unknown"
        unstaged_changes = "could not retrieve git information"

    return commit_hash, unstaged_changes

def seal_source_tree(source_dir, output_file, exclude_patterns):
    """
    Package a source tree into an HDF5 file with exclusion patterns and Git metadata.
    """
    commit_hash, unstaged_changes = get_git_info(source_dir)
    
    with h5py.File(output_file, 'w') as hdf5_file:
        # Store Git metadata
        hdf5_file.attrs['commit_hash'] = commit_hash
        hdf5_file.attrs['unstaged_changes'] = unstaged_changes

        # Add source files
        add_directory_to_hdf5(hdf5_file, source_dir, exclude_patterns)
        

