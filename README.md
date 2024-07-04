## GACE utils ##
This repository contains several Python modules written to be used in conjunction with GRACE.

# Quick start guide

The first step is to install GRACEpy. This can be achieved very easily. 
Firstly, make sure to be inside a virtual environment where packages can be safely installed with `pip`. When this is done, it's enough to navigate to GRACEpy's directory and execute the following command
```bash
pip install -e .
```
This will install all of GRACEpy's dependencies as well as the packages contained in this repository and the associated command line tools. 

The most important tools in GRACEpy are those related to output analysis and post-processing. Most of the time when running a simulation with grace you will be dealing with hdf5 output. This output contains substantial amounts of human and machine readable metadata that make it compatible with several industry-standard visualization and analysis tools. 
All analysis tools need to access the metadata in order to know how data is stored inside grace hdf5 output. This information is usually provided through an Xdmf (or xmf) file. An xdmf file is a plain text file written in xml format containing information about timesteps in the output, available cell and vertex centered variables, and their types (whether they are scalars, vectors, tensors). For this reason, before analyzing grace output you will need to generate an xmf file describing it, descriptor file for short. Once you installed GRACEpy this can be very easily done, all that is needed is the following command:
```bash
create_descriptor <input_directory> <output_file>
```
Here, `<input_directory>` is the path to a directory containing grace hdf5 output and  `<output_file>` is the desired name of the descriptor file to be generated. The `.xmf` extension will be automatically added to the filename, ensuring native compatibility with VTK tools such as Paraview. Ensure that you don't add whitespaces or special characters to the filename. Should you still need help
```bash
create_descriptor --help
```
Will print a message with more details on how to use this command-line utility. Note that generating descriptors only reads light metadata, and will therefore be a very light task even for large datasets.
**NB** Make sure not to move the descriptor file once created since it generally contains relative paths pointing to the hdf5 files.

Once a descriptor file is created, you can analyze the data with a plethora of standard tools.
Below is a list of supported visualization and analysis tools supported by grace.

# Paraview

Paraview can read `.xmf` descriptor files natively, simply opening the descriptor with the default Paraview reader should suffice.

# Python 

GRACEpy comes with several packages to aid visualization of grace data in Python and to enable interoperability with standard scientific and visualization packages such as numpy, scipy or matplotlib. The first of these packages is `grace_tools`.

The `grace_tools` package defines the `grace_xmf_reader` class makes it easy to read and analyze multi-dimensional data from grace in Python. First, you will need a descriptor (if you don't know what it is, go back to the previous section). To read it in Python you can do the following:
```python
import grace_tools.vtk_reader_utils as gtv
reader = gtv.grace_xmf_reader("descriptor.xmf")
```
This loads the metadata in Python, and is a lightweight operation to perform even for very large datasets. The reader itself is a complicated object, but its extensive interface makes it easy to handle. Firstly, you can query the available output times and variables with
```python
reader.available_times()
# Output example: [0.0,1.0,2.0]
reader.available_variables()
# Output example: ["rho","vel"]
```
The reader has a state, meaning that at any moment it is either in its default state (no timestep loaded) or with a specific timestep in its pipeline. You can select a timestep as follows
```python
reader.set_time(0.0)
```
Most reader methods take a time as optional input. This means that if no time is specified, the operation will be performed on the time slice that is currently in the pipeline. The current reader time can also be set by selecting an index (with respect to the `available_times()` list)
```python
reader.set_time_index(0)
```
Note that if the requested time is not available, the reader will be set to the closest available time, and display a warning if this is far from what was requested.
The main purpose of the reader is to output the data in a format that can be plotted or used in post-processing. The easiest way to do this is simply
```python
xyz,rho = reader.get_var("rho",time=1.0)
# Output: xyz is an np.array with shape [ncells,3] containing cell-center coordinates 
# and rho is an np.array of size [ncells] containing datapoints.
```
This method takes many optional arguments, and you can refer to its docstring for more customization.
Data in grace is usually defined on a complicated grid, consisting of several quadrants at different refinement levels which might have a curvilinear geometry. For this reason, the output of `get_var` cannot simply be plotted with something like `matplotlib.pyplot.contourf`. However, by default data is returned by the reader as a numpy array, meaning that variables can be composed and post-processed using any tool supporting this data type. Plotting multi-dimensional grace data in Python can be achieved via triangulation-based plotting functions such as `matplotlib.pyplot.tricontourf`. For instance, for 2D data
```python
import matplotlib.pyplot as plt
import matplotlib.tri as tri

fig,ax = plt.subplots()
# Create a triangulation of the cell-center locations
triangulation = tri.Triangulation(xyz[:,0],xyz[:,1])
# Plot data over the triangulation
ax.tricontourf(triangulation,rho,cmap="inferno",levels=500)
```
This will produce a 2D filled contour plot much like `plt.contourf`. Note that cell-center coordinates always come as x,y,z with z simply being always 0 for 2D datasets.
Additionally, the `reader` provides methods for slicing datasets in order to obtain lower dimensional datasets in various ways.
- **1D slices**: Data can be sliced with an arbitrary line with the following method
```python
# Define origin and direction vector of an arbitrary line
# e.g. the x-axis
x_axis_direction = np.array([1,0,0])
x_axis_origin    = np.array([0,0,0])
# Slice a variable along the line
xyz1D,rho1D = reader.get_var_1D_slice("rho",
                                      time=1.0,
                                      line_direction=x_axis_direction,
                                      line_point=x_axis_origin) 
# The output can simply be plotted
fig,ax = plt.subplots()
ax.plot(xyz1D[:,0], rho1D)
# Note that we pick the 0th coordinate because
# in this example the slice is along the x-axis,
# adjust as needed.
```
- **2D slices**: Data can also be sliced with an arbitrary plane, given an origin and a normal vector
```python
# Define origin and normal vector of an arbitrary plane
# e.g. the xy-plane
xy_normal = np.array([0,0,1])
xy_origin    = np.array([0,0,0])
# Slice a variable with the plane
xyz2D,rho2D = reader.get_var_2D_slice("rho",
                                      time=1.0,
                                      plane_normal=xy_normal,
                                      plane_origin=xy_origin) 
# This output will need to be triangulated
```
- **Spherical slices**: The reader can also slice data using a sphere
```python
# Define origin and radius of a sphere
# e.g. the unit sphere
center = np.array([0,0,0])
radius = 1.
# Slice a variable with the sphere
xyz,rho = reader.get_var_spherical_slice("rho",
                                         time=1.,
                                         sphere_center=center,
                                         sphere_radius=radius) 
# This output will need to be triangulated
```
- **Cylindrical slices**: Or a cylinder
```python
# Define origin, radius and axis of a cylinder
# e.g. unit radius cylinder around the z-axis
center = np.array([0,0,0])
radius = 1.
axis   = np.array([0,0,1])
# Slice a variable with the cylinder
xyz,rho = reader.get_var_cylindrical_slice("rho",
                                          time=1.,
                                          cylinder_center=center,
                                          cylinder_axis=axis,
                                          cylinder_radius=radius) 
# This output will need to be triangulated
```
- **Conical slices**: Or even a cone
```python
# Define origin axis and opening angle of a cone
# e.g. a 45 degree cone around the z-axis
origin = np.array([0,0,0])
axis   = np.array([0,0,1])
angle  = 45.
# Slice a variable with the cone
xyz2D,rho2D = reader.get_var_conical_slice("rho",
                                           time=1.0,
                                           cone_center=center,
                                           cone_axis=axis,
                                           cone_angle=angle) 
# This output will need to be triangulated
```

Note that in all the output functions above, the time argument can be omitted and the current selected time of the reader will be used.
Another useful feature of the reader is that it can provide the coordinate of grid cell vertices, this is useful to plot the grid structure along with the data. Since grace typically runs on way too many cells to be plotted, and retrieving all the vertex coordinates can be expensive, only quadrant vertex coordinates are returned by the reader. The order of the vertices in the return is consistent with both VTK's convention (for vtkQuadrilateral cells in 2D and vtkHexahedron cells in 3D) as well as with matplotlib's `Polygon` convention, which makes it easy to plot the grid
```python
# Get quadrant vertex coordinates
vertices = reader.get_quadrant_vertices(time=1.,ncells=16)

# Plot grid using matplotlib
fig, ax = plt.subplots()

for i in range(vertices.shape[0]):
    polygon = plt.Polygon(vertices[i,:,:-1], edgecolor="black", linewidth=0.3, facecolor="None")
    ax.add_patch(polygon)
```
Both arguments are optional. As always, `time` is optional and if omitted defaults to the current timeslice of the reader.
`ncells` represents the number of cells per quadrant in each direction. It can be omitted, but then the grace output must have been generated with `output_extra: true`, since the output variable `Quad_ID` will be used to determine the number of cells per quadrant.

This completes the overview of basic methods and tools that can be used to visualize and analyze grace hdf5 output in Python. For more detailed information, refer to the sphinx generated documentation of the modules or to the docstrings that can be directly accessed from any python interpreter. Happy plotting!

