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
from grace_tools.scalar_export import export_scalars_hdf5, import_scalars_hdf5
from grace_tools.fuka_utils import parse_fuka_info

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

    @classmethod
    def from_hdf5(cls, filepath, verbose=False):
        """Reconstruct a grace_simulation from an exported HDF5 file.

        This is the inverse of ``export_scalars``. The returned object
        has ``scalars``, ``gw``, and ``detectors`` fully populated.
        Volume/plane readers (``xyz``, ``xy``, ``xz``, ``yz``) and
        profiling (``prof``) are set to ``None``.

        Args:
            filepath (str): Path to the HDF5 file produced by
                ``export_scalars`` or ``export_scalars_hdf5``.
            verbose (bool, optional):
                Print progress messages. Defaults to False.

        Returns:
            grace_simulation: Reconstructed simulation object.
        """
        data = import_scalars_hdf5(filepath)

        sim = cls.__new__(cls)
        sim._verbose = verbose
        sim.simdir = None
        sim.name = data["name"]
        sim.bdir = None
        sim.descdir = None
        sim._config = None
        sim._id_Madm = data["gw"].Madm
        sim._id_omega0 = data["gw"].omega0

        sim.xyz = None
        sim.xy = None
        sim.xz = None
        sim.yz = None

        sim.scalars = data["scalars"]
        sim.gw = data["gw"]
        sim.detectors = data["detectors"]
        sim.prof = None

        sim._log(f"Reconstructed '{sim.name}' from {filepath}")
        sim._log(f"GW detectors: {sim.gw.available_detectors()}")
        return sim

    def __init__(self, simdir: str, parfile=None, ppdir="./plots", Madm=None, omega0=None, verbose=False):
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
            Madm (float, optional):
                Total ADM mass of the system. If None and the initial data
                type is FUKA, this is auto-detected from the info file.
            omega0 (float, optional):
                Initial orbital frequency. If None and the initial data
                type is FUKA, this is auto-detected from the info file.
            verbose (bool, optional):
                Print progress messages during construction. Defaults to False.
        """
        self._verbose = verbose
        self.simdir = simdir
        if not os.path.isdir(self.simdir):
            raise ValueError(f"Simulation directory {simdir} doesn't exist or cannot be read.")
        if parfile is None:
            parfile = self.__find_parfile()
        self.__parse_parfile(parfile)
        self._log(f"Simulation '{self.name}' in {self.simdir}")

        # Auto-detect ID metadata from FUKA info file if available
        self._id_Madm, self._id_omega0 = self.__detect_id_metadata()
        # Explicit constructor arguments take precedence
        if Madm is not None:
            self._id_Madm = Madm
        if omega0 is not None:
            self._id_omega0 = omega0
        if self._id_Madm is not None or self._id_omega0 is not None:
            self._log(f"ID metadata: Madm={self._id_Madm}, omega0={self._id_omega0}")

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
        self._log(f"Found {len(scalar_dirs)} scalar output dir(s)")

        self._log("Building XMF descriptors...")
        has_vol = gtx.write_xmf_file(os.path.join(self.descdir, "volume_descriptor.xmf"),
                                      volume_dirs, mode="temporal")
        has_xy  = gtx.write_xmf_file(os.path.join(self.descdir, "xy_plane_descriptor.xmf"),
                                      plane_dirs, mode="temporal", filter="*xy*")
        has_xz  = gtx.write_xmf_file(os.path.join(self.descdir, "xz_plane_descriptor.xmf"),
                                      plane_dirs, mode="temporal", filter="*xz*")
        has_yz  = gtx.write_xmf_file(os.path.join(self.descdir, "yz_plane_descriptor.xmf"),
                                      plane_dirs, mode="temporal", filter="*yz*")
        gtx.write_xmf_file(os.path.join(self.descdir, "sphere_descriptor.xmf"),
                           sphere_dirs, mode="spherical")

        self._log(f"VTK readers: vol={has_vol}, xy={has_xy}, xz={has_xz}, yz={has_yz}")
        self.xyz = gtv.grace_xmf_reader(os.path.join(self.descdir, "volume_descriptor.xmf")) if has_vol else None
        self.xy  = gtv.grace_xmf_reader(os.path.join(self.descdir, "xy_plane_descriptor.xmf")) if has_xy else None
        self.xz  = gtv.grace_xmf_reader(os.path.join(self.descdir, "xz_plane_descriptor.xmf")) if has_xz else None
        self.yz  = gtv.grace_xmf_reader(os.path.join(self.descdir, "yz_plane_descriptor.xmf")) if has_yz else None

        self._log("Reading scalar data...")
        self.scalars = gts.grace_scalars_reader(scalar_dirs)
        self._log("Reading GW data...")
        self.gw      = gtg.grace_gw_data(scalar_dirs, Madm=self._id_Madm, omega0=self._id_omega0)
        self._log(f"GW detectors: {self.gw.available_detectors()}")

        # Assemble unified detector views
        self.detectors = self.__build_detectors()
        self._log("Done.")

    def _log(self, msg):
        """Print a message if verbose mode is enabled."""
        if self._verbose:
            print(f"[grace_simulation] {msg}")

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

    def __detect_id_metadata(self):
        """Try to extract ADM mass and orbital frequency from initial data.

        Currently supports FUKA initial data: reads the .info file
        referenced in the parameter file under grmhd.fuka.

        Returns:
            tuple: (Madm, omega0) — both None if not available.
        """
        if self._config is None:
            return None, None

        id_type = self._config.get("grmhd", {}).get("id_type")
        if id_type != "fuka":
            return None, None

        fuka_cfg = self._config.get("grmhd", {}).get("fuka", {})
        id_dir = fuka_cfg.get("id_dir")
        filename = fuka_cfg.get("filename")
        if id_dir is None or filename is None:
            return None, None

        if not filename.endswith(".info"):
            filename = filename + ".info"
        info_path = os.path.join(id_dir, filename)

        try:
            Madm, omega0 = parse_fuka_info(info_path)
            return Madm, omega0
        except (FileNotFoundError, ValueError) as e:
            print(f"WARNING: Could not read FUKA info file: {e}")
            return None, None

    def __repr__(self):
        lines = [f"grace_simulation '{self.name}'"]
        if self.simdir is not None:
            lines.append(f"  directory: {self.simdir}")
        if self._id_Madm is not None:
            lines.append(f"  Madm: {self._id_Madm:.4f}")
        if self._id_omega0 is not None:
            lines.append(f"  omega0: {self._id_omega0:.6f}")

        readers = []
        if self.xyz is not None:
            readers.append("xyz")
        if self.xy is not None:
            readers.append("xy")
        if self.xz is not None:
            readers.append("xz")
        if self.yz is not None:
            readers.append("yz")
        lines.append(f"  VTK readers: {', '.join(readers) if readers else 'none'}")

        if self.scalars is not None:
            nred = sum(len(getattr(self.scalars, cat).available_vars())
                       for cat in ("maximum", "minimum", "norm2", "integral"))
            lines.append(f"  scalars: {nred} reduction var(s)")

        if self.gw is not None:
            dets = self.gw.available_detectors()
            if dets:
                parts = []
                for d in dets:
                    nmodes = len(self.gw[d].available_modes())
                    parts.append(f"{d} ({nmodes} modes)")
                lines.append(f"  GW: {', '.join(parts)}")
            else:
                lines.append("  GW: no detectors")

        if self.detectors is not None:
            det_names = self.detectors.available_detectors()
            if det_names:
                lines.append(f"  detectors: {', '.join(det_names)}")

        return "\n".join(lines)

    def export_scalars(self, outfile=None):
        """Export all merged scalar and GW data to a single HDF5 file.

        Args:
            outfile (str, optional): Output file path. Defaults to
                <ppdir>/<simname>_scalars.h5.
        """
        if outfile is None:
            outfile = os.path.join(self.bdir, f"{self.name}_scalars.h5")
        export_scalars_hdf5(self.scalars, self.gw, outfile,
                            name=self.name, detectors=self.detectors)
        print(f"Scalars exported to {outfile}")
