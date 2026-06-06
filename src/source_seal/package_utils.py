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

def seal_source_tree_into_group(group, source_dir, exclude_patterns):
    """
    Seal a source tree into an already-open HDF5 group (or file).

    Writes the Git metadata as attributes on ``group`` and the source files
    as datasets under it. Use this to nest a sealed source tree inside a larger
    archive; ``seal_source_tree`` is the standalone path-based wrapper.

    Returns:
        tuple[str, str]: ``(commit_hash, unstaged_changes)``.
    """
    commit_hash, unstaged_changes = get_git_info(source_dir)

    # Store Git metadata
    group.attrs['commit_hash'] = commit_hash
    group.attrs['unstaged_changes'] = unstaged_changes

    # Add source files
    add_directory_to_hdf5(group, source_dir, exclude_patterns)

    return commit_hash, unstaged_changes

def seal_source_tree(source_dir, output_file, exclude_patterns):
    """
    Package a source tree into an HDF5 file with exclusion patterns and Git metadata.
    """
    with h5py.File(output_file, 'w') as hdf5_file:
        seal_source_tree_into_group(hdf5_file, source_dir, exclude_patterns)
        

