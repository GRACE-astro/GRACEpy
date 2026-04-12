"""This module contains utilities to handle simulations performed by grace."""

import numpy as np
import glob
import re
import os
import yaml

import grace_tools.vtk_reader_utils as gtv
import grace_tools.xmf_utils as gtx
import grace_tools.scalar_reader_utils as gts
import grace_tools.profiling_reader_utils as gtp
import grace_tools.gw_reader_utils as gtg
from grace_tools.detector_utils import grace_detector_set
from grace_tools.scalar_export import export_scalars_hdf5

class grace_simulation:
    """Class designed to aid the analysis of data from a grace simulation.

    This class parses the simulation parameter file and searches for hdf5
    and scalar output. Supports both simpilot-managed directories (with
    config/parfile/ and restart_NNNN/ subdirectories) and flat layouts.

    Attributes:
        name (str):
            The name of this simulation.
        xyz (grace_xmf_reader):
            The volume output reader. Can be used
            to query for available vars/times and to extract the data
            or slice it for post-processing purposes. Refer to the class
            documentation for additional information.
        xy  (grace_xmf_reader):
            The xy plane reader.
        yz  (grace_xmf_reader):
            The yz plane reader.
        xz  (grace_xmf_reader):
            The xz plane reader.
        scalars (grace_scalars_reader):
            The scalar output reader. Contains all
            available timeseries data. Refer to its documentation for
            usage instructions.
        gw (grace_gw_data):
            Gravitational wave data reader. Contains rPsi4 modes
            organized by detector and (l,m).
        detectors (grace_detector_set):
            Unified per-detector view. Provides detector metadata
            (radius, center) from the parameter file and references
            to GW modes and mass flux data for each detector.
        prof (grace_profiling_reader):
            The profiling output reader. Contains
            all available profiling information for this simulation.
            Refer to its documentation for usage instructions.
    """

    def __init__(self, simdir: str, parfile=None, ppdir="./plots"):
        """Create a grace_simulation object.

        This is used to handle output data from a grace simulation.

        Args:
            simdir (str):
                Directory where the simulation is found.
            parfile (str, optional):
                Parameter file of the simulation (full path or relative to where
                this function is called). If None it will be looked for
                automatically: first in config/parfile/ (simpilot layout), then
                in the simulation directory root. Defaults to None.
            ppdir (str, optional):
                Directory for post-processing output (descriptors, plots).
                Defaults to "./plots".
        """
        self.simdir = simdir
        if not os.path.isdir(self.simdir):
            raise ValueError(f"Simulation directory {simdir} doesn't exist or cannot be read.")
        if parfile is None:
            parfile = self.__find_parfile()
        self.__parse_parfile(parfile)
        self.bdir = ppdir
        self.descdir = os.path.join(self.bdir, "descriptors")
        if not os.path.isdir(ppdir):
            os.mkdir(self.bdir)
        if not os.path.isdir(self.descdir):
            os.mkdir(self.descdir)

        # Build XMF descriptors from all restart dirs (or flat layout)
        volume_dirs = self.__find_output_dirs(self._volume_subdir)
        plane_dirs  = self.__find_output_dirs(self._plane_subdir)
        sphere_dirs = self.__find_output_dirs(self._sphere_subdir)
        scalar_dirs = self.__find_output_dirs(self._scalar_subdir)

        gtx.write_xmf_file(os.path.join(self.descdir, "volume_descriptor.xmf"),
                           volume_dirs, mode="temporal")
        gtx.write_xmf_file(os.path.join(self.descdir, "xy_plane_descriptor.xmf"),
                           plane_dirs, mode="temporal", filter="*xy*")
        gtx.write_xmf_file(os.path.join(self.descdir, "xz_plane_descriptor.xmf"),
                           plane_dirs, mode="temporal", filter="*xz*")
        gtx.write_xmf_file(os.path.join(self.descdir, "yz_plane_descriptor.xmf"),
                           plane_dirs, mode="temporal", filter="*yz*")
        gtx.write_xmf_file(os.path.join(self.descdir, "sphere_descriptor.xmf"),
                           sphere_dirs, mode="spherical")

        self.xyz     = gtv.grace_xmf_reader(os.path.join(self.descdir, "volume_descriptor.xmf"))
        self.xy      = gtv.grace_xmf_reader(os.path.join(self.descdir, "xy_plane_descriptor.xmf"))
        self.xz      = gtv.grace_xmf_reader(os.path.join(self.descdir, "xz_plane_descriptor.xmf"))
        self.yz      = gtv.grace_xmf_reader(os.path.join(self.descdir, "yz_plane_descriptor.xmf"))
        self.scalars = gts.grace_scalars_reader(scalar_dirs)
        self.gw      = gtg.grace_gw_data(scalar_dirs)

        # Assemble unified detector views
        self.detectors = self.__build_detectors()

    def __build_detectors(self):
        """Build the detector set from parfile config and attach data references."""
        dset = grace_detector_set.from_parfile_config(self._config)

        # Attach GW mode data
        for det_name in self.gw.available_detectors():
            if det_name not in dset:
                # Detector found in data but not in parfile — create without metadata
                from grace_tools.detector_utils import grace_detector
                dset[det_name] = grace_detector(det_name)
            dset[det_name].gw = self.gw[det_name]

        # Attach mass flux data
        for det_name in self.scalars.mass_flux:
            if det_name not in dset:
                from grace_tools.detector_utils import grace_detector
                dset[det_name] = grace_detector(det_name)
            dset[det_name].mass_flux = self.scalars.mass_flux[det_name]

        return dset

    def __find_output_dirs(self, subdir):
        """Collect output directories, including restart segments.

        Looks for restart_NNNN/ subdirectories under simdir. If any contain
        the requested subdirectory, returns all of them (sorted by restart id).
        Otherwise falls back to simdir/subdir.

        Args:
            subdir (str): Relative subdirectory name (e.g. 'output_scalar').

        Returns:
            list: List of directory paths, ordered by restart sequence.
        """
        restart_dirs = sorted(glob.glob(os.path.join(self.simdir, "restart_*")))
        if restart_dirs:
            candidates = [os.path.join(rd, subdir) for rd in restart_dirs
                          if os.path.isdir(os.path.join(rd, subdir))]
            if candidates:
                return candidates
        return [os.path.join(self.simdir, subdir)]

    def __find_parfile(self):
        """Search for a parameter file (YAML).

        First checks the simpilot layout (config/parfile/), then falls
        back to searching for YAML files in the simulation directory root.

        Returns:
            str or None: Path to the parameter file, or None if not found.
        """
        # Simpilot layout: config/parfile/<name>.yaml
        config_parfile_dir = os.path.join(self.simdir, "config", "parfile")
        if os.path.isdir(config_parfile_dir):
            flist = glob.glob(os.path.join(config_parfile_dir, "*.yaml"))
            if len(flist) == 1:
                return flist[0]
            if len(flist) > 1:
                print("WARNING: Multiple YAML files found in "
                      f"{config_parfile_dir}, all paths set to default values.")
                return None

        # Legacy/manual layout: simdir/*.yaml
        patt = os.path.join(self.simdir, "*.yaml")
        flist = glob.glob(patt)
        if len(flist) == 0:
            print("WARNING: No parfile specified and the simulation directory"
                  f" {self.simdir} does not contain one, all paths set to default values.")
            return None
        if len(flist) > 1:
            print("WARNING: No parfile specified and the simulation directory"
                  f" contains more than one yaml file, all paths set to default values.")
            return None
        return flist[0]

    def __parse_parfile(self, parfile):
        """Parse parameter file to extract output directory names and simulation name.

        Stores relative subdirectory names (not full paths) so they can be
        resolved per-restart via __find_output_dirs.
        """
        if parfile is None:
            self._config = None
            self._volume_subdir = "output_volume"
            self._plane_subdir  = "output_surface"
            self._sphere_subdir = "output_spheres"
            self._scalar_subdir = "output_scalar"
            self.name = "grace"
            return

        with open(parfile, 'r') as f:
            config = yaml.safe_load(f)
        self._config = config

        self._volume_subdir = config.get("IO", {}).get(
            "volume_output_base_directory", "output_volume")
        self._plane_subdir = config.get("IO", {}).get(
            "surface_output_base_directory", "output_surface")
        self._sphere_subdir = config.get("IO", {}).get(
            "sphere_surface_output_base_directory", "output_spheres")
        self._scalar_subdir = config.get("IO", {}).get(
            "scalar_output_base_directory", "output_scalar")
        self.name = str(config.get("name", "grace"))

    def export_scalars(self, outfile=None):
        """Export all merged scalar and GW data to a single HDF5 file.

        Args:
            outfile (str, optional): Output file path. Defaults to
                <ppdir>/<simname>_scalars.h5.
        """
        if outfile is None:
            outfile = os.path.join(self.bdir, f"{self.name}_scalars.h5")
        export_scalars_hdf5(self.scalars, self.gw, outfile)
        print(f"Scalars exported to {outfile}")
