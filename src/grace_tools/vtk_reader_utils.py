"""
This module contains the grace_xmf_reader class.

The grace_xmf_reader class is the main entry point to
read xdmf files describing hdf5 output from grace. The 
class can parse an xmf file, extract variables at specific 
times and slice the grid for post-processing and plotting 
purposes.
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt
import grace_tools.geometric_utils as gu

class grace_xmf_reader:
    """
    Xmf descriptor file reader.

    This class is the backbone of data post-processing in grace.
    It reads an xmf file describing hdf5 output from grace.
    This class also provides methods to extract specific timesteps 
    from the output and slice datasets with planes, spheres, and lines.

    Methods:
        set_time(time):
            Set the current reader output to a specific time slice.
        
        available_times():
            Get the times where data is available.
        
        available_variables(varname, vartype):
            Get the available variables of a specific type (either "cell" or "point").
        
        get_var(varname, time, override_no_timestep_selection, convert_to_numpy, vartype):
            Get volume data and coordinates at a given time.
        
        get_var_1D_slice(varname, time, line_direction, line_point, override_no_timestep_selection, convert_to_numpy, vartype):
            Get a 1D slice of a variable at a given time.
        
        get_var_2D_slice(varname, time, plane_normal, plane_origin, override_no_timestep_selection, convert_to_numpy, vartype):
            Get a 2D plane slice of a variable at a given time.
    """

    
    def __init__(self,filename: str):
        """Construct a grace_xmf_reader given the descriptor filename.
        
        Xdmf descriptor files can be generated with python (see grace_tools.xmf_utils)
        or directly by grace. 

        Args:
            filename (str): Name of xdmf file this reader should parse for grace output.
        """
        self.reader = vtk.vtkXdmfReader()
        self.set_file(filename)
        self.update()
        self.available_times_list = self.reader.GetOutputInformation(0).Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
        output = self.reader.GetOutput()
        cell_data = output.GetCellData() 
        self.available_cell_vars_list  = [ cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays()) ]
        point_data = output.GetPointData() 
        self.available_point_vars_list  = [ point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays()) ]
        self.__query_data_dimensions()
        self.__bounds = output.GetBounds() 
    
    def __check_vtype(self,vtype):
        """
        Checks if the provided variable type is valid.

        Parameters:
        vtype (str): The variable type to check. Must be either 'cell' or 'point'.

        Returns:
        None

        Raises:
        ValueError: If the provided variable type is not 'cell' or 'point'.
        """
        if  vtype == "cell" or vtype == "point":
            return 
        else:
            raise ValueError(f"Unrecognized variable type {vtype}. Supported types are 'cell' or 'point'.")
    
    def __check_requested_var(self,varname,vtype=None):
        """
        Checks if the requested variable is present in the available variables list.

        Parameters:
        -----------
        varname : str
            The name of the variable to check.
        vtype : str, optional
            The type of the variable, either "cell" or "point". If not specified, the function will check both cell and point variables.

        Raises:
        -------
        ValueError
            If the requested variable is not present in the available variables list.
        """
        if vtype == "cell":
            if not (varname in self.available_cell_vars_list):
                raise ValueError(f"Requested cell variable {varname} not present in output.")
        elif vtype == "point":
            if not (varname in self.available_point_vars_list):
                raise ValueError(f"Requested point variable {varname} not present in output.")
        else:
            if not (varname in self.available_cell_vars_list) and not (varname in self.available_point_vars_list):
                raise ValueError(f"Requested variable {varname} not present in output.")
        return
    
    def __get_vartype(self, varname):
        if varname in self.available_cell_vars_list:
            return "cell"
        elif varname in self.available_point_vars_list:
            return "point"
        else:
            raise ValueError(f"Variable {var} is not present in simulation data.")    
    
    def __get_info(self,port=0):
        """
        Retrieve the output information from the reader.

        Args:
            port (int, optional): The port number to get the output information from. Defaults to 0.

        Returns:
            vtkInformation: The output information for the specified port.
        """
        return self.reader.GetOutputInformation(port)
    
    def __has_timestep_information(self):
        """
        Checks if the VTK data contains timestep information.

        Returns:
            bool: True if the VTK data has timestep information, False otherwise.
        """
        return self.__get_info().Has(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
    
    def __is_at_timestep(self):
        """
        Checks if the current VTK pipeline is at a specific timestep.

        Returns:
            bool: True if the current VTK pipeline is at the specified timestep, False otherwise.
        """
        return self.__get_info().Has(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
    
    def __get_current_timestep(self):
        """
        Retrieve the current timestep if the pipeline is at a timestep.

        This method checks if the pipeline is currently at a timestep and, if so,
        retrieves the current timestep information from the VTK pipeline.

        Returns:
            float: The current timestep value if the pipeline is at a timestep.
        """
        if self.__is_at_timestep():
            return self.__get_info().Get(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
    
    def __get_cell_centers(self,output):
        """
        Private method to compute the cell centers of a given VTK output.

        Parameters:
        output (vtk.vtkDataSet): The VTK dataset from which to compute cell centers.

        Returns:
        vtk.vtkDataArray: The data array containing the coordinates of the cell centers.
        """
        cell_centers = vtk.vtkCellCenters()
        cell_centers.SetInputData(output)
        cell_centers.Update() 
        return cell_centers.GetOutput().GetPoints().GetData() 
    
    def __query_data_dimensions(self):
        """
        Queries the dimensions of the data read by the VTK reader and sets the 
        `is_data_2D` attribute accordingly.

        This method sets the time index to 0 and retrieves the output from the 
        VTK reader. It then checks the cell types in the output to determine 
        if the data is 2D or 3D. If the cell type is quadrilateral (cell type 9), 
        the data is considered 2D. Otherwise, it is considered 3D. Finally, 
        the reader is reset to its initial state.

        Returns:
            None
        """
        self.set_time_index(0)
        output = self.reader.GetOutput()
        cell_types = vtk.vtkCellTypes()
        output.GetCellTypes(cell_types)
        # Check if the cell type is quadrilateral, else we are in 3D 
        self.is_data_2D = (cell_types.GetCellType(0) == 9)
        self.reset_reader()
        return
    
    def __find_ncells(self):
        """
        Private method to find the number of cells in the dataset.

        This method determines the number of cells by examining the "Quad_ID" variable.
        It counts the number of consecutive cells with the same Quad_ID to determine the total number of cells.
        Depending on whether the data is 2D or 3D, it returns the square root or cube root of the number of cells.

        Returns:
            int: The number of cells in the dataset. For 2D data, it returns the square root of the number of cells.
                 For 3D data, it returns the cube root of the number of cells.
        """
        _, quadid = self.get_var("Quad_ID",convert_to_numpy=True)
        ncells = 0
        qid0 = quadid[0]
        qid = qid0 
        while qid == qid0:
            ncells+=1
            qid = quadid[ncells]
        if self.is_data_2D:
            return int(np.sqrt(ncells)) 
        else:
            return int(np.cbrt(ncells))  

    def grid_bounds(self):
        """Get the grid boundary coordinates."""
        return self.__bounds 

    def spatial_dimensions(self):
        """Get the number of spatial dimensions of the output."""
        if self.is_data_2D:
            return 2 
        else:
            return 3
    
    def get_output(self):
        """Get output of reader in current state."""
        return self.reader.GetOutput() 
    
    def update(self):
        """Update reader."""
        self.reader.Update()
        return
    
    def get_cell_data(self):
        """Get cell data from reader in current state."""
        self.update()
        return self.get_output().GetCellData() 
    
    def get_point_data(self):
        """Get point data from reader in current state."""
        self.update()
        return self.get_output().GetPointData()

    def set_file(self,filename: str):
        """Set reader file and reset its state."""
        self.fname: str = filename 
        self.reader.SetFileName(filename)
        return
    
    def set_time(self,time):
        """Set the current time of the reader."""
        index = np.searchsorted(self.available_times_list,time)
        if abs(time - self.available_times_list[index]) > 1e-10:
            print(f"WARNING: requested time {time}, closest available {self.available_times_list[index]}")
        self.set_time_index(index)
        return
    
    def set_time_index(self,index):
        """Set current time of the reader given the index in the available times list."""
        self.reader.GetOutputInformation(0).Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(), self.available_times_list[index])
        self.update()
        return 
    
    def reset_reader(self):
        """Reset reader to default state."""
        self.set_file(self.fname)
        self.update()
        return
    
    def available_times(self):
        """Get available output times."""
        return self.available_times_list  
    
    def available_variables(self, vtype="cell"):
        """Get available variables in output.

        Args:
            vtype (str, optional): Type of variables requested ("cell" or "point"). Defaults to "cell".
        
        Returns:
            list of strings: List of available variables in output.
        """
        if vtype == "Cell" or vtype == "cell":
            return self.available_cell_vars_list
        elif vtype == "Point" or vtype == "point":
            return self.available_point_vars_list
        else:
            raise ValueError(f"Unrecognized variable type {vtype}. Supported types are 'cell' or 'point'.")
    
    def get_var(self,varname,time=None,override_no_timestep_selection=False,convert_to_numpy=True,vartype="cell"):
        """Get full output of a variable at a given time.

        Args:
            varname (str): Variable name.
            time (float, optional): Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None.
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
            convert_to_numpy (bool, optional): Return numpy arrays. Defaults to True.
            vartype (str, optional): Variable type. Defaults to "cell".

        Returns:
            np.array or vtk.vtkDataArray: The full variable output at specified time (codimension 0).
        """
        self.__check_vtype(vartype)
        self.__check_requested_var(varname, vtype=vartype)
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True")
            return None
        if time is not None:
            self.set_time(time)
        output = self.get_output()
        cell_data = output.GetCellData()
        point_data = output.GetPointData()
        if ( vartype == "cell" ) :
            vararray = cell_data.GetArray(varname)
            coords   = self.__get_cell_centers(output)
        else: 
            vararray = point_data.GetArray(varname)
            coords   = output.GetPoints().GetData()
        if convert_to_numpy:
            return (vtk_to_numpy(coords), vtk_to_numpy(vararray)[:])
        else:
            return (coords,vararray)

    def get_var_1D_slice(self,varname,time=None,
                         line_point_1=None,
                         line_point_2=None,
                         line_n_points=100,
                         override_no_timestep_selection=False,
                         convert_to_numpy=True):
        """Get 1D slice of variable at a specified time.

        Args:
            varname (str): Variable name.
            time (float, optional): Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None.
            line_point_1 (np.array, optional): Point where the line starts.
            line_point_2 (np.array, optional): Point where the line ends.
            line_n_points: Number of points to sample along the line.
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
            convert_to_numpy (bool, optional): Return numpy arrays. Defaults to True.

        Returns:
            np.array or vtk.vtkDataArray: 1D slice of the variable at the specified time.
        """
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True")
            return None
        if time is not None:
            self.set_time(time)
        if line_point_1 is None:
                line_point_1 = np.array( [self.__bounds[0], 0,0])
        if line_point_2 is None:
                line_point_2 = np.array( [self.__bounds[1], 0,0])
        line_point_1 = gu.pad_array_with_zeros(line_point_1)
        line_point_2 = gu.pad_array_with_zeros(line_point_2)
        
        return self.__get_1D_slice_impl(varname,line_point_1,line_point_2,line_n_points,convert_to_numpy)
    
    def get_var_2D_slice(self,varname,time=None,
                         plane_normal=np.array([0,0,1]),
                         plane_origin=np.array([0,0,0]),
                         override_no_timestep_selection=False,
                         convert_to_numpy=True,
                         vartype="cell"):
        """Get 2D slice of variable at a specified time.

        Args:
            varname (str): Variable name.
            time (float, optional): Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None.
            plane_normal (np.array, optional): Orthogonal vector to the plane. Defaults to np.array([0,0,1]).
            plane_origin (np.array, optional): Any within the plane where output is requested. Defaults to np.array([0,0,0]).
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
            convert_to_numpy (bool, optional): Return numpy arrays. Defaults to True.
            vartype (str, optional): Variable type. Defaults to "cell".

        Returns:
            np.array or vtk.vtkDataArray: 2D slice of the variable at the specified time.
        """
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True")
            return None
        plane_normal = gu.pad_array_with_zeros(plane_normal)
        plane_origin = gu.pad_array_with_zeros(plane_origin)
        if time is not None:
            self.set_time(time)
        if self.is_data_2D:
            return self.get_var(varname,None,override_no_timestep_selection,convert_to_numpy,vartype)
        else:
            return self.__get_2D_slice_impl(varname,plane_normal,plane_origin,convert_to_numpy,vartype)
        
    def get_var_spherical_slice(self,varname,time=None,
                                sphere_center=np.array([0,0,0]),
                                sphere_radius=1.0,
                                override_no_timestep_selection=False,
                                convert_to_numpy=True,
                                vartype="cell"):
        """Get spherical slice of variable at a specified time.

        Args:
            varname (str): Variable name.
            time (float, optional): Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None.
            sphere_center (np.array, optional): Coordinates of the sphere center. Defaults to np.array([0,0,0]).
            sphere_radius (float, optional): Radius of the sphere. Defaults to 1.0.
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
            convert_to_numpy (bool, optional): Return numpy arrays. Defaults to True.
            vartype (str, optional): Variable type. Defaults to "cell".

        Returns:
            np.array or vtk.vtkDataArray: Spherical slice of the variable at the specified time.
        """
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True")
            return None
        if time is not None:
            self.set_time(time)
        
        sphere_center = gu.pad_array_with_zeros(sphere_center)
        
        sphere = vtk.vtkSphere()
        sphere.SetCenter(sphere_center.tolist())
        sphere.SetRadius(sphere_radius)
        
        return self.__cut_dataset(varname,sphere,convert_to_numpy,vartype)
        
    def get_var_cylindrical_slice(self,varname,time=None,
                                  cylinder_center=np.array([0,0,0]),
                                  cylinder_axis=np.array([0,0,1]),
                                  cylinder_radius=1.0,
                                  override_no_timestep_selection=False,
                                  convert_to_numpy=True,
                                  vartype="cell"):
        """Get cylindrical slice of variable at a specified time.

        Args:
            varname (str): Variable name.
            time (float, optional): Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None.
            cylinder_center (np.array, optional): Coordinates of the cylinder center. Defaults to np.array([0,0,0]).
            cylinder_axis (np.array, optional): Direction of the cylinder axis. Defaults to np.array([0,0,1]).
            cylinder_radius (float, optional): Radius of the cylinder. Defaults to 1.0.
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
            convert_to_numpy (bool, optional): Return numpy arrays. Defaults to True.
            vartype (str, optional): Variable type. Defaults to "cell".

        Returns:
            np.array or vtk.vtkDataArray: Cylindrical slice of the variable at the specified time.
        """
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True")
            return None
        if time is not None:
            self.set_time(time)
        
        cylinder_center = gu.pad_array_with_zeros(cylinder_center)
        cylinder_axis = gu.pad_array_with_zeros(cylinder_axis)
        
        cylinder = vtk.vtkCylinder()
        cylinder.SetCenter(cylinder_center.tolist())
        cylinder.SetRadius(cylinder_radius)
        cylinder.SetAxis(cylinder_axis.tolist())
        
        return self.__cut_dataset(varname,cylinder,convert_to_numpy,vartype)
    
    
    def get_var_conical_slice(self,varname,time=None,
                              cone_center=np.array([0,0,0]),
                              cone_axis=np.array([0,0,1]),
                              cone_angle=45.0,
                              override_no_timestep_selection=False,
                              convert_to_numpy=True,
                              vartype="cell"):
        """Get conical slice of variable at a specified time.

        Args:
            varname (str): Variable name.
            time (float, optional): Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None.
            cone_center (np.array, optional): Coordinates of the cone center. Defaults to np.array([0,0,0]).
            cone_axis (np.array, optional): Direction of the cone axis. Defaults to np.array([0,0,1]).
            cone_angle (float, optional): Opening angle of the cone (in degrees). Defaults to 1.0.
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
            convert_to_numpy (bool, optional): Return numpy arrays. Defaults to True.
            vartype (str, optional): Variable type. Defaults to "cell".

        Returns:
            np.array or vtk.vtkDataArray: Conical slice of the variable at the specified time.
        """
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True")
            return None
        if time is not None:
            self.set_time(time)
        
        cone_center = gu.pad_array_with_zeros(cone_center)
        cone_axis = gu.pad_array_with_zeros(cone_axis)
        
        cone = vtk.vtkCone()
        cone.SetAngle(cone_angle)
        cone.SetCenter(cone_center.tolist())
        cone.SetDirection(cone_axis.tolist())
        
        return self.__cut_dataset(varname,cone,convert_to_numpy,vartype)
    
    
    def get_quadrant_vertices(self, time=None, ncells=None, override_no_timestep_selection=False):
        """Get coordinates of grid quadrant vertices.
        
        Can be used to plot the grid structure. 
        NB: this only returns quadrant vertices, not cell vertices. This is done to avoid 
        having too many points returned.

        Args:
            time (float, optional):Time of requested output. Actual output will be at closest available time.
                                    If None current reader time will be used. Defaults to None. 
            ncells (_type_, optional): Number of cells per quadrant in each direction. If None will attempt to 
                                       find this from Quad_ID. Defaults to None.
            override_no_timestep_selection (bool, optional): If True, reads even if no time is specified and the reader is 
                                                             not set at a specific time state. This means all timesteps will 
                                                             be read at once, which can be very slow. Defaults to False.
        Returns:
            np.array: Array containing quadrant vertices. The vertex ordering is consistent with both VTK (vtkQuad, vtkHex)
                      as well as with matplotlib.pyplot.Polygon.
        """
        if (time is None) and (not self.__is_at_timestep()) and (not override_no_timestep_selection):
            print("WARNING: Attempting to extract data from a reader"
                  " with no timestep selected. If you really want this, pass override_no_timestep_selection=True") 
        if time is not None:
            self.set_time(time)
        
        if "Quad_ID" in self.available_cell_vars_list:
            ncells = self.__find_ncells()
        elif ncells is None:
            raise ValueError("If the number of cells per quadrant is not provided, 'Quad_ID' must be present in the output")

        if self.is_data_2D:
            return self.__get_quadrant_vertices_2D(ncells)
        else:
            return self.__get_quadrant_vertices_3D(ncells)
        
        
    def __get_quadrant_vertices_2D(self,ncells: int):
        """
        Computes the vertices of quadrants in a 2D grid.

        This method retrieves the vertices of quadrants in a 2D grid from the output
        of a VTK object. The vertices are computed based on the number of cells in
        each dimension.

        Args:
            ncells (int): The number of cells in each dimension of the grid.

        Returns:
            np.ndarray: A numpy array containing the vertices of the quadrants.
        """
        output = self.get_output()
        vertices = []
        print("Ncells: {}".format(ncells))
        for i in range(output.GetNumberOfCells())[::int(ncells**2)]:
            stride = int(ncells**2)
            cell_idxs = [ i, i+ncells-1, i+stride-1, i+stride-ncells ]
            cell_vertices = [ output.GetCell(cell_idxs[ip]).GetPoints().GetPoint(ip) for ip in range(len(cell_idxs)) ]
            vertices.append(cell_vertices)
        return np.array(vertices)

    def __get_quadrant_vertices_3D(self,ncells: int):
        """
        Computes the vertices of the quadrants in a 3D grid.
        Args:
            ncells (int): The number of cells along one dimension of the grid.
        Returns:
            np.ndarray: A numpy array containing the vertices of the quadrants.
        """
        output = self.get_output()
        vertices = []
        
        for i in range(output.GetNumberOfCells())[::int(ncells**3)]:
            stridex = int(ncells)
            stridey = int(ncells**2)
            stridez = int(ncells**3)
            cell_idxs = [i,i+stridex-1,i+stridey-1,i+stridey-stridex,i+stridez-stridey,i+stridez-stridey+stridex-1,i+stridez-1,i+stridez-stridex]
            cell_vertices = [ output.GetCell(cell_idxs[ip]).GetPoints().GetPoint(ip) for ip in range(len(cell_idxs)) ]
            vertices.append(cell_vertices)
        return np.array(vertices)
    
    def __get_1D_slice_impl(self,varname,line_point_1,line_point_2,line_npoints,convert_to_numpy):
        """
        Extracts a 1D slice from the dataset along a specified line.
        Parameters:
        varname (str): The name of the variable to extract.
        line_point_1 (tuple): The starting point of the line (x, y, z).
        line_point_2 (tuple): The ending point of the line (x, y, z).
        line_npoints (int): The number of points along the line.
        convert_to_numpy (bool): If True, converts the result to a NumPy array.
        Returns:
        vtkDataArray or numpy.ndarray: The extracted data along the line, either as a VTK data array or a NumPy array.
        """
        self.__check_requested_var(varname)
        line = vtk.vtkLineSource()
        line.SetPoint1(*line_point_1)
        line.SetPoint2(*line_point_2)
        line.SetResolution(line_npoints)
        
        return self.__probe_dataset(varname,line,convert_to_numpy)
    
    
    def __get_2D_slice_impl(self,varname,plane_normal,plane_origin,convert_to_numpy,vartype):
        """
        Extracts a 2D slice from a dataset based on the specified plane parameters.
        Args:
            varname (str): The name of the variable to extract.
            plane_normal (tuple or list): The normal vector of the slicing plane.
            plane_origin (tuple or list): The origin point of the slicing plane.
            convert_to_numpy (bool): If True, converts the result to a NumPy array.
            vartype (str): The type of the variable (e.g., scalar, vector).
        Returns:
            The sliced dataset, optionally converted to a NumPy array.
        Raises:
            ValueError: If the variable type or requested variable is invalid.
        """
        
        self.__check_vtype(vartype)
        self.__check_requested_var(varname,vartype)
        cutter_plane = vtk.vtkPlane() 
        cutter_plane.SetNormal(list(plane_normal))
        cutter_plane.SetOrigin(list(plane_origin))
        return self.__cut_dataset(varname,cutter_plane,convert_to_numpy,vartype)
    
    def __probe_dataset(self, varname, probe_algo, convert_to_numpy):
        """
        Probes a dataset using the specified probe algorithm and retrieves the specified variable array.
        Args:
            varname (str): The name of the variable array to retrieve.
            probe_algo (vtkAlgorithm): The VTK algorithm used for probing the dataset.
            convert_to_numpy (bool): If True, converts the output coordinates and variable array to NumPy arrays.
        Returns:
            tuple: A tuple containing the coordinates and the variable array. If convert_to_numpy is True, 
                   the coordinates and variable array are returned as NumPy arrays. Otherwise, they are 
                   returned as VTK arrays.
        """
        probe_filter = vtk.vtkProbeFilter()
        probe_filter.SetInputConnection(probe_algo.GetOutputPort())
        probe_filter.SetSourceData(self.reader.GetOutput())
        probe_filter.Update()
        output = probe_filter.GetOutput()
        vararray = output.GetPointData().GetArray(varname)
        coords   = output.GetPoints().GetData()
        if convert_to_numpy:
            return (vtk_to_numpy(coords), vtk_to_numpy(vararray)[:])
        return (coords,vararray)
        
    
    def __cut_dataset(self,varname,cut_function,convert_to_numpy,vartype):
        """
        Cuts the dataset using the specified cutting function and extracts the variable data.
        Parameters:
        varname (str): The name of the variable to extract.
        cut_function (vtk.vtkImplicitFunction): The cutting function to use.
        convert_to_numpy (bool): If True, converts the output to numpy arrays.
        vartype (str): The type of data to extract, either "cell" or "point".
        Returns:
        tuple: A tuple containing the coordinates and the variable data. If convert_to_numpy is True, 
                both elements of the tuple are numpy arrays. If convert_to_numpy is False, both elements 
                are VTK arrays. If the cutter produces no output, returns (None, None).
        """
        cutter = vtk.vtkCutter() 
        cutter.SetCutFunction(cut_function)
        cutter.SetInputConnection(self.reader.GetOutputPort())
        cutter.Update()
        output = cutter.GetOutput()
        if output.GetNumberOfCells() == 0 :
            print("WARNING: Cutter produced no output. Check input parameters"
                  "and ensure  the cutting function intersects the grid.")
            return (None,None)
        cell_data  = output.GetCellData() 
        point_data = output.GetPointData()  

        if ( vartype == "cell" ) :
            vararray = cell_data.GetArray(varname)
            coords   = self.__get_cell_centers(output)
        else: 
            vararray = point_data.GetArray(varname)
            coords   = output.GetPoints().GetData()
        if convert_to_numpy:
            return (vtk_to_numpy(coords), vtk_to_numpy(vararray)[:])
        else:
            return (coords,vararray)
        
        
    
    
        
        

    


