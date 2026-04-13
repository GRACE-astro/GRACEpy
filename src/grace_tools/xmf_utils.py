import numpy as np
import os
import glob
import h5py
import re
from collections import defaultdict


# ---------------- Helper to group planes by iteration ----------------
def dtype_to_xdmf(dtype):
    """
    Converts a NumPy dtype to XDMF DataType and Precision.
    
    Returns:
        xdmf_type (str), precision (int)
    """
    if np.issubdtype(dtype, np.floating):
        return "Float", dtype.itemsize
    elif np.issubdtype(dtype, np.signedinteger):
        return "Int", dtype.itemsize
    elif np.issubdtype(dtype, np.unsignedinteger):
        return "UInt", dtype.itemsize
    else:
        raise ValueError(f"Unsupported dtype for XDMF: {dtype}")


def group_files__kind_iteration(files):
    """
    Groups HDF5 files by iteration and plane.
    Returns a dict: {iteration: {plane: filename, ...}, ...}
    """
    grouped = defaultdict(dict)
    for f in files:
        iter_num = extract_iteration(f)
        plane_match = re.search(r'plane_(xy|yz|xz)_\d+\.h5$', f)
        vol_match = re.search(r'volume_out_\d+\.h5$', f)
        if plane_match: 
            kind = plane_match.group(1) + "_plane"
        elif vol_match:
            kind = 'vol'
        else :
            print(f"WARNING could not determine kind of file {f}, assuming volume")
            kind = 'vol'
        grouped[iter_num][kind] = f
    return grouped


def write_xmf_scalar_attribute(name, staggering, dimensions,dtype,h5fname):
    """
    Generates an XMF (eXtensible Model Format) string for a scalar attribute.

    Parameters:
    name (str): The name of the attribute.
    staggering (str): The center of the attribute, typically "Node" or "Cell".
    dimensions (tuple): The dimensions of the data.
    h5fname (str): The filename of the HDF5 file containing the data.

    Returns:
    str: An XMF string representing the scalar attribute.
    """
    dstr,prec = dtype_to_xdmf(dtype)
    return '''<Attribute Center="{}" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{}" Type="Scalar">
    <DataItem DataType="{}" Dimensions="{}" Format="HDF" Precision="{}">{}:/{}</DataItem>
    </Attribute>\n'''.format(staggering,name,dstr,dimensions,prec,h5fname,name)

def write_xmf_vector_attribute(name,staggering,dimensions,dtype,h5fname):
    """
    Generates an XMF (eXtensible Model Format) string for a vector attribute.

    Parameters:
    name (str): The name of the attribute.
    staggering (str): The staggering type (e.g., "Node", "Cell").
    dimensions (tuple): A tuple representing the dimensions of the data (e.g., (100,)).
    h5fname (str): The filename of the HDF5 file containing the data.

    Returns:
    str: An XMF string defining the vector attribute.
    """
    dstr,prec = dtype_to_xdmf(dtype)
    return '''<Attribute Center="{}" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{}" Type="Vector">
    <DataItem DataType="{}" Dimensions="{} 3" Format="HDF" Precision="{}">{}:/{}</DataItem>
    </Attribute>\n'''.format(staggering,name,dstr,dimensions,prec,h5fname,name)
    
def write_xmf_symm_tensor_attribute(name,staggering,dimensions,h5fname):
    """
    Generates an XMF (eXtensible Model Format) string for a symmetric tensor attribute.

    Parameters:
    name (str): The name of the attribute.
    staggering (str): The staggering type (e.g., "Node", "Cell").
    dimensions (tuple): A tuple representing the dimensions of the data.
    h5fname (str): The filename of the HDF5 file containing the data.

    Returns:
    str: An XMF string defining the symmetric tensor attribute.
    """
    dstr,prec = dtype_to_xdmf(dtype)
    return '''<Attribute Center="{}" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{}" Type="Tensor6">
    <DataItem DataType="{}" Dimensions="{} 6" Format="HDF" Precision="{}">{}:/{}</DataItem>
    </Attribute>\n'''.format(staggering,name,dstr,dimensions,prec,h5fname,name)

def write_xmf_grid(iteration,time,points_dims,cells_dims,cells_type,h5name,attrs):
    """
    Generates an XMF (eXtensible Model Format) grid definition string for a given iteration and time step.

    Parameters:
    iteration (int): The iteration number of the grid.
    time (float): The time value associated with the grid.
    points_dims (tuple): A tuple containing the dimensions of the points (number of points, number of coordinates per point).
    cells_dims (tuple): A tuple containing the dimensions of the cells (number of cells, number of points per cell).
    cells_type (str): The type of the cells (e.g., "Hexahedron", "Tetrahedron").
    h5name (str): The name of the HDF5 file containing the data.
    attrs (list): A list of dictionaries, each containing attributes of the grid. Each dictionary should have the keys:
        - "name" (str): The name of the attribute.
        - "dtype" (str): The data type of the attribute ("Scalar", "Vector", or other).
        - "staggering" (str): The staggering type of the attribute.
        - "dimensions" (tuple): The dimensions of the attribute data.

    Returns:
    str: An XMF string defining the grid and its attributes.
    """
    output = '''<Grid Name="Grid_{}">
    <Time Value="{}"/>
    <Geometry Origin="" Type="XYZ">
    <DataItem DataType="Float" Dimensions="{} {}" Format="HDF" Precision="8">{}:/Points</DataItem>
    </Geometry>
    <Topology Dimensions="{}" Type="{}">
    <DataItem DataType="UInt" Dimensions="{} {}" Format="HDF" Precision="4">{}:/Cells</DataItem>
    </Topology>\n'''.format(iteration,time,points_dims[0],points_dims[1],h5name,cells_dims[0],cells_type,cells_dims[0],cells_dims[1],h5name)
    for attr in attrs:
        if( attr["dtype"] == "Scalar"):
            output += write_xmf_scalar_attribute(attr["name"],attr["staggering"],attr["dimensions"],attr["data_type"],h5name)
        elif attr["dtype"] == "Vector":
            output += write_xmf_vector_attribute(attr["name"],attr["staggering"],attr["dimensions"],attr["data_type"],h5name)
        else:
            output += write_xmf_symm_tensor_attribute(attr["name"],attr["staggering"],attr["dimensions"],attr["data_type"],h5name)
    return output + "</Grid>\n"

def write_xmf_point_grid(iteration, time, points_dims, h5name, attrs,
                         use_polyvertex=True):
    """
    Generates an XMF grid for point-only data (e.g., a spherical surface 
    sampled at points with nodal values). No cell connectivity is required.

    Parameters:
        iteration (int): Iteration index for the grid name.
        time (float): Physical time associated with this output.
        points_dims (tuple): (num_points, num_coordinates), usually (N, 3).
        h5name (str): Name of the HDF5 file containing the datasets.
        attrs (list): Attribute dictionaries with keys:
            - "name": attribute name (str)
            - "dtype": "Scalar", "Vector", or other
            - "staggering": typically "Node"
            - "dimensions": tuple defining the attribute size
            - "data_type": underlying type, e.g. "Float"
        use_polyvertex (bool): If True, write <Topology Type="Polyvertex">.
                               If False, omit Topology entirely.

    Returns:
        str: An XMF string representing a point-only grid.
    """

    # Header + Geometry
    output = f'''<Grid Name="PointGrid_{iteration}" GridType="Uniform">
    <Time Value="{time}"/>
    <Geometry Type="XYZ">
        <DataItem DataType="Float" Dimensions="{points_dims[0]} {points_dims[1]}"
                  Format="HDF" Precision="8">{h5name}:/Points</DataItem>
    </Geometry>
    '''

    # Optional Polyvertex topology (safe for all XDMF readers)
    if use_polyvertex:
        output += f'''    <Topology Type="Polyvertex" Dimensions="{points_dims[0]}"/>'''

    # Attributes
    for attr in attrs:
        if attr["dtype"] == "Scalar":
            output += write_xmf_scalar_attribute(
                attr["name"], attr["staggering"], attr["dimensions"],
                attr["data_type"], h5name
            )
        elif attr["dtype"] == "Vector":
            output += write_xmf_vector_attribute(
                attr["name"], attr["staggering"], attr["dimensions"],
                attr["data_type"], h5name
            )
        else:
            output += write_xmf_symm_tensor_attribute(
                attr["name"], attr["staggering"], attr["dimensions"],
                attr["data_type"], h5name
            )

    output += "</Grid>\n"
    return output



def write_xmf_grid_no_time(name,points_dims,cells_dims,cells_type,h5name,attrs):
    """
    Generates an XMF (eXtensible Model Format) grid definition string for a given iteration and time step.

    Parameters:
    iteration (int): The iteration number of the grid.
    time (float): The time value associated with the grid.
    points_dims (tuple): A tuple containing the dimensions of the points (number of points, number of coordinates per point).
    cells_dims (tuple): A tuple containing the dimensions of the cells (number of cells, number of points per cell).
    cells_type (str): The type of the cells (e.g., "Hexahedron", "Tetrahedron").
    h5name (str): The name of the HDF5 file containing the data.
    attrs (list): A list of dictionaries, each containing attributes of the grid. Each dictionary should have the keys:
        - "name" (str): The name of the attribute.
        - "dtype" (str): The data type of the attribute ("Scalar", "Vector", or other).
        - "staggering" (str): The staggering type of the attribute.
        - "dimensions" (tuple): The dimensions of the attribute data.

    Returns:
    str: An XMF string defining the grid and its attributes.
    """
    output = '''<Grid Name="Grid_{}">
    <Geometry Origin="" Type="XYZ">
    <DataItem DataType="Float" Dimensions="{} {}" Format="HDF" Precision="8">{}:/Points</DataItem>
    </Geometry>
    <Topology Dimensions="{}" Type="{}">
    <DataItem DataType="UInt" Dimensions="{} {}" Format="HDF" Precision="4">{}:/Cells</DataItem>
    </Topology>\n'''.format(name,points_dims[0],points_dims[1],h5name,cells_dims[0],cells_type,cells_dims[0],cells_dims[1],h5name)
    for attr in attrs:
        if( attr["dtype"] == "Scalar"):
            output += write_xmf_scalar_attribute(attr["name"],attr["staggering"],attr["dimensions"],attr["data_type"],h5name)
        elif attr["dtype"] == "Vector":
            output += write_xmf_vector_attribute(attr["name"],attr["staggering"],attr["dimensions"],attr["data_type"],h5name)
        else:
            output += write_xmf_symm_tensor_attribute(attr["name"],attr["staggering"],attr["dimensions"],attr["data_type"],h5name)
    return output + "</Grid>\n"

def write_xmf_spatial_collection(iteration,time,grids):
    """
    Generates an XMF (eXtensible Model Format) spatial collection string for a given set of grids.
    Parameters:
    name (str): The name of the XMF collection.
    grids (list of dict): A list of dictionaries, each representing a grid. Each dictionary should contain the following keys:
        - "iteration" (int): The iteration number of the grid.
        - "time" (float): The time value associated with the grid.
        - "points_dims" (tuple of int): The dimensions of the points in the grid.
        - "cells_dims" (tuple of int): The dimensions of the cells in the grid.
        - "cells_type" (str): The type of cells in the grid.
        - "h5name" (str): The name of the HDF5 file associated with the grid.
        - "attrs" (dict): A dictionary of attributes for the grid.
    Returns:
    str: An XMF collection string representing the provided grids.
    """
    output = '''<Grid CollectionType="Spatial" GridType="Collection" Name="Grids_{}">\n'''.format(iteration)
    output += '''<Time Value="{}"/>'''.format(time)
    for grid in grids:
        output += write_xmf_grid_no_time(grid["name"],grid["points_dims"],grid["cells_dims"],grid["cells_type"],grid["h5name"],grid["attrs"])
    return output + "</Grid>\n"

def write_xmf_temporal_collection(name, collections):
    """
    Generates an XMF (eXtensible Model Format) spatial collection string for a given set of grids.
    Parameters:
    name (str): The name of the XMF collection.
    grids (list of dict): A list of dictionaries, each representing a grid. Each dictionary should contain the following keys:
        - "iteration" (int): The iteration number of the grid.
        - "time" (float): The time value associated with the grid.
        - "points_dims" (tuple of int): The dimensions of the points in the grid.
        - "cells_dims" (tuple of int): The dimensions of the cells in the grid.
        - "cells_type" (str): The type of cells in the grid.
        - "h5name" (str): The name of the HDF5 file associated with the grid.
        - "attrs" (dict): A dictionary of attributes for the grid.
    Returns:
    str: An XMF collection string representing the provided grids.
    """
    output = '''<Grid CollectionType="Temporal" GridType="Collection" Name="{}">\n'''.format(name)
    for coll in collections:
        output += write_xmf_spatial_collection(coll['iteration'],coll['time'],coll['grids'])
    return output + "</Grid>\n"

def write_xmf_temporal_collection_spheres(name, grids):
    """
    Generates an XMF (eXtensible Model Format) spatial collection string for a given set of grids.
    Parameters:
    name (str): The name of the XMF collection.
    grids (list of dict): A list of dictionaries, each representing a grid. Each dictionary should contain the following keys:
        - "iteration" (int): The iteration number of the grid.
        - "time" (float): The time value associated with the grid.
        - "points_dims" (tuple of int): The dimensions of the points in the grid.
        - "cells_dims" (tuple of int): The dimensions of the cells in the grid.
        - "cells_type" (str): The type of cells in the grid.
        - "h5name" (str): The name of the HDF5 file associated with the grid.
        - "attrs" (dict): A dictionary of attributes for the grid.
    Returns:
    str: An XMF collection string representing the provided grids.
    """
    output = '''<Grid CollectionType="Temporal" GridType="Collection" Name="{}">\n'''.format(name)
    for grid in grids:
        output += write_xmf_point_grid(grid["iteration"],grid["time"],grid["points_dims"],grid["h5name"],grid["attrs"])
    return output + "</Grid>\n"

def write_xmf_collection(name,grids):
    """
    Generates an XMF (eXtensible Model Format) collection string for a given set of grids.
    Parameters:
    name (str): The name of the XMF collection.
    grids (list of dict): A list of dictionaries, each representing a grid. Each dictionary should contain the following keys:
        - "iteration" (int): The iteration number of the grid.
        - "time" (float): The time value associated with the grid.
        - "points_dims" (tuple of int): The dimensions of the points in the grid.
        - "cells_dims" (tuple of int): The dimensions of the cells in the grid.
        - "cells_type" (str): The type of cells in the grid.
        - "h5name" (str): The name of the HDF5 file associated with the grid.
        - "attrs" (dict): A dictionary of attributes for the grid.
    Returns:
    str: An XMF collection string representing the provided grids.
    """
    output = '''<Grid CollectionType="Temporal" GridType="Collection" Name="{}">\n'''.format(name)
    for grid in grids:
        output += write_xmf_grid(grid["iteration"],grid["time"],grid["points_dims"],grid["cells_dims"],grid["cells_type"],grid["h5name"],grid["attrs"])
    return output + "</Grid>\n"

def write_xmf_file_header(collection,spheres=False):
    """
    Generates the header for an XMF (eXtensible Model Format) file.
    Parameters:
    collection (dict): A dictionary containing the name and grids of the collection.
                       The dictionary should have the following keys:
                       - "name": A string representing the name of the collection.
                       - "grids": A list of grids to be included in the collection.
    Returns:
    str: A string containing the XMF file header with the collection information.
    """
    output = '''<?xml version="1.0" encoding="utf-8"?>
    <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
    <Domain>'''
    if spheres: output += write_xmf_temporal_collection_spheres(collection["name"], collection["grids"])
    else: output += write_xmf_collection(collection["name"],collection["grids"])
    return output + '''</Domain>
    </Xdmf>\n'''

def write_xmf_file_header_spatial_collection(collection):
    """
    Generates the header for an XMF (eXtensible Model Format) file.
    Parameters:
    collection (dict): A dictionary containing the name and grids of the collection.
                       The dictionary should have the following keys:
                       - "name": A string representing the name of the collection.
                       - "grids": A list of grids to be included in the collection.
    Returns:
    str: A string containing the XMF file header with the collection information.
    """
    output = '''<?xml version="1.0" encoding="utf-8"?>
    <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
    <Domain>'''
    output += write_xmf_temporal_collection(collection["name"],collection["collections"])
    return output + '''</Domain>
    </Xdmf>\n'''


def collect_hdf5_attributes(fname, has_cells=True):
    """
    Collects attributes from an HDF5 file, including dataset type and precision.
    
    Parameters:
    fname (str): The path to the HDF5 file.
    
    Returns:
    tuple: A tuple containing:
        - vnames (list): variable names in the HDF5 file.
        - vtypes (list): VariableType attribute for each variable.
        - vstags (list): VariableStaggering attribute for each variable.
        - vdatatypes (list): NumPy dtype for each variable.
        - vprecisions (list): Size in bytes for each variable's dtype.
        - cell_dims (tuple): shape of "Cells" dataset.
        - point_dims (tuple): shape of "Points" dataset.
        - var_dims (list): first dimension of the dataset depending on staggering.
        - time (float): Time attribute.
        - iteration (int): Iteration attribute.
        - topology (str): CellTopology attribute of "Cells".
    """
    vnames = []
    vtypes = []
    vstags = []
    vdatatypes = []
    var_dims = []
    
    with h5py.File(fname, "r") as f:
        if has_cells: 
            cell_dims = f["Cells"].shape
            topology = f["Cells"].attrs["CellTopology"]
        point_dims = f["Points"].shape

        time = f.attrs["Time"]
        iteration = f.attrs["Iteration"]
        

        for vname in f.keys():
            if vname in ("Cells", "Points"):
                continue
            
            ds = f[vname]
            vtypes.append(ds.attrs["VariableType"])
            vstags.append(ds.attrs["VariableStaggering"])
            
            # Determine dimensions based on staggering
            if vstags[-1] == "Cell":
                var_dims.append(cell_dims[0])
            elif vstags[-1] == "Node":
                var_dims.append(point_dims[0])
            else:
                raise ValueError(f"Unknown staggering type {vstags[-1]}")
            
            # Dataset dtype and precision
            vdatatypes.append(ds.dtype)
            
            vnames.append(vname)
    if has_cells:
        return (vnames, vtypes, vstags, vdatatypes,
                cell_dims, point_dims, var_dims, time, iteration, topology)
    else:
        return (vnames, vtypes, vstags, vdatatypes, point_dims, var_dims, time, iteration)


def find_iter_file(bdir,iteration):
    """
    Finds the file corresponding to a specific iteration in a given directory.
    Args:
        bdir (str): The base directory where the files are located.
        iteration (int): The iteration number to search for.
    Returns:
        str: The path to the file corresponding to the specified iteration.
    Raises:
        IndexError: If no file matching the iteration is found.
    """
    flist = glob.glob(bdir+"/*"+"{:06d}".format(iteration)+".h5")
    if len(flist)==0:
        raise IndexError("No file found for iteration {}".format(iteration))
    return flist[0]

def extract_iteration(filename):
    """
    Extracts the iteration number from a given filename.
    The function searches for a pattern in the filename that matches
    a sequence of non-whitespace characters followed by an underscore,
    a series of digits, and the '.h5' file extension. If such a pattern
    is found, it returns the iteration number as an integer. If no match
    is found, it returns -1.
    Args:
        filename (str): The filename from which to extract the iteration number.
    Returns:
        int: The extracted iteration number, or -1 if no match is found.
    """
    match = re.search(r'\S+_(\d+)\.h5', filename)
    return int(match.group(1)) if match else -1

def construct_grid(f, name="volume_grid"):
    ff = os.path.abspath(f)
    vnames,vtypes,vstags,vdtype,cell_dims,points_dims,var_dims,time,iteration,topology = collect_hdf5_attributes(ff)
    attrs = []
    for i,vname in enumerate(vnames):
        attrs.append({"name": vname, "dtype": vtypes[i], "data_type": vdtype[i], "dimensions": var_dims[i], "staggering": vstags[i]})
    return {"name": name, "iteration": iteration, "time": time, "points_dims": points_dims, "cells_dims": cell_dims, "h5name": ff, "cells_type": topology, "attrs": attrs}

def construct_spherical_grid(f, name="sphere"):
    ff = os.path.abspath(f)
    vnames,vtypes,vstags,vdtype,points_dims,var_dims,time,iteration = collect_hdf5_attributes(ff,False)
    attrs = []
    for i,vname in enumerate(vnames):
        attrs.append({"name": vname, "dtype": vtypes[i], "data_type": vdtype[i], "dimensions": var_dims[i], "staggering": vstags[i]})
    return {"name": name, "iteration": iteration, "time": time, "points_dims": points_dims, "h5name": ff, "attrs": attrs}


def write_xmf_file(outfile, bdir="./",mode="volume", verbose: bool = False, filter=None):
    """
    Writes an XMF (eXtensible Model Format) file that references a collection of HDF5 files.
    Parameters:
    outfile (str): The path to the output XMF file.
    bdir (str or list): The base directory (or list of directories) containing
        the HDF5 files. When a list is provided, files from all directories are
        collected (e.g. for merging output across restart segments). Defaults to "./".
    iterations (list or int, optional): A list of iteration numbers or a single iteration number to include in the XMF file.
                                        If None, all HDF5 files in the base directory are included. Defaults to None.
    Returns:
    None
    """
    outfile = os.path.abspath(outfile)
    if isinstance(bdir, str):
        bdir = [bdir]
    flist = []
    for d in bdir:
        if filter:
            flist.extend(glob.glob(os.path.join(d, filter)))
        else:
            flist.extend(glob.glob(os.path.join(d, "*.h5")))
    if not flist:
        return False
    grouped = group_files__kind_iteration(flist)
    iterations = sorted(grouped.keys())
    kinds_per_iter = {it: list(grouped[it].keys()) for it in iterations}
    
    use_spatial_collection = any(
        any(k != 'vol' and k != 'unknown' for k in kinds_per_iter[it])
        for it in iterations
    )

    if verbose:
        print("Iterations:", iterations)
        print("Kinds per iteration:", kinds_per_iter)
        print("Use spatial collection:", use_spatial_collection)
    
    if mode == "temporal" or (mode=="auto" and not use_spatial_collection):
        grids = []
        for it in iterations:
            if len(grouped[it].keys()) > 1:
                raise ValueError("Forced mode temporal but multiple outputs present.")
            key = list(grouped[it].keys())[0]
            grids.append(construct_grid(grouped[it][key]))
        with open(outfile,"w") as fout:
            fout.write(write_xmf_file_header({"name":"collection", "grids":grids}))
    elif mode == "spatial" or (mode=="auto" and use_spatial_collection):
        colls = [] 
        for it in iterations:
            grids = []
            for k in grouped[it].keys():
                grids.append(construct_grid(grouped[it][k], k))
            time = grids[-1]['time']
            colls.append({'iteration': it, 'time': time, 'grids': grids})
        with open(outfile,"w") as fout:
            fout.write(write_xmf_file_header_spatial_collection({"name":"collection", "collections":colls}))
    elif mode == "spherical":
        grids = [] 
        for it in iterations:
            key = list(grouped[it].keys())[0]
            grids.append(construct_spherical_grid(grouped[it][key]))
        with open(outfile,"w") as fout:
            fout.write(write_xmf_file_header({"name":"collection", "grids":grids}, True))
    return True

