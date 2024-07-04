import numpy as np
import os
import glob
import h5py
import re

def write_xmf_scalar_attribute(name,dimensions,h5fname):
    return '''<Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{}" Type="Scalar">
    <DataItem DataType="Float" Dimensions="{}" Format="HDF" Precision="8">{}:/{}</DataItem>
    </Attribute>\n'''.format(name,dimensions[0],h5fname,name)

def write_xmf_vector_attribute(name,dimensions,h5fname):
    return '''<Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{}" Type="Vector">
    <DataItem DataType="Float" Dimensions="{} 3" Format="HDF" Precision="8">{}:/{}</DataItem>
    </Attribute>\n'''.format(name,dimensions[0],h5fname,name)

def write_xmf_grid(iteration,time,points_dims,cells_dims,cells_type,h5name,attrs):
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
            output += write_xmf_scalar_attribute(attr["name"],attr["dimensions"],h5name)
        else:
            output += write_xmf_vector_attribute(attr["name"],attr["dimensions"],h5name)
    return output + "</Grid>\n"

def write_xmf_collection(name,grids):
    output = '''<Grid CollectionType="Temporal" GridType="Collection" Name="{}">\n'''.format(name)
    for grid in grids:
        output += write_xmf_grid(grid["iteration"],grid["time"],grid["points_dims"],grid["cells_dims"],grid["cells_type"],grid["h5name"],grid["attrs"])
    return output + "</Grid>\n"

def write_xmf_file_header(collection):
    output = '''<?xml version="1.0" encoding="utf-8"?>
    <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
    <Domain>'''
    output += write_xmf_collection(collection["name"],collection["grids"])
    return output + '''</Domain>
    </Xdmf>\n'''

def collect_hdf5_attributes(fname):
    vnames = []
    vtypes = [] 
    with h5py.File(fname,"r") as f:
        cell_dims = f["Cells"].shape
        point_dims = f["Points"].shape
        var_dims = [cell_dims[0]]
        time = f.attrs["Time"]
        iteration = f.attrs["Iteration"]
        topology = f["Cells"].attrs["CellTopology"]
        for vname in f.keys():
            if( vname=="Cells" or vname=="Points"):
                continue
            vtypes.append(f[vname].attrs["VariableType"])
            vnames.append(vname)
    return vnames,vtypes,cell_dims,point_dims,var_dims,time,iteration,topology

def find_iter_file(bdir,iteration):
    flist = glob.glob(bdir+"/*"+"{:06d}".format(iteration)+".h5")
    return flist[0]

def extract_iteration(filename):
    match = re.search(r'\S+_(\d+)\.h5', filename)
    return int(match.group(1)) if match else -1

def write_xmf_file(outfile, bdir="./",iterations=None):
    flist = []
    
    if iterations is None:
        flist = glob.glob(bdir+"/*.h5")
    elif type(iterations) is list:
        for it in iterations:
            flist.append(find_iter_file(bdir,it))
    else:
        flist.append(find_iter_file(bdir,iterations))
    sorted_file_list = sorted(flist, key=extract_iteration)
    grids = [] 
    for ff in sorted_file_list:
        vnames,vtypes,cell_dims,points_dims,var_dims,time,iteration,topology = collect_hdf5_attributes(ff)
        attrs = []
        for i,vname in enumerate(vnames):
            attrs.append({"name": vname, "dtype": vtypes[i], "dimensions": var_dims})
        grids.append({"iteration": iteration, "time": time, "points_dims": points_dims, "cells_dims": cell_dims, "h5name": ff, "cells_type": topology, "attrs": attrs})
    with open(outfile,"w") as fout:
        fout.write(write_xmf_file_header({"name":"collection", "grids":grids}))
    return 