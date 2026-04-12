"""Utilities for unified per-detector views of grace data."""


class grace_detector:
    """Unified view of all data associated with a single spherical detector.

    This is a view class — it holds references to data loaded by standalone
    readers (grace_gw_data, grace_scalars_reader), not copies.

    Attributes:
        name (str): Detector name (e.g. 'GW_1').
        radius (float or None): Extraction radius in code units.
        center (tuple or None): Center coordinates (x, y, z).
        resolution (int or None): Angular resolution of the detector.
        sampling_policy (str or None): Discretization policy ('uniform', 'healpix').
        gw (grace_gw_detector or None): GW mode data for this detector.
        mass_flux (grace_timeseries_array or None): Mass flux data for this detector.
    """

    def __init__(self, name, radius=None, center=None, resolution=None,
                 sampling_policy=None):
        self.name = name
        self.radius = radius
        self.center = center
        self.resolution = resolution
        self.sampling_policy = sampling_policy
        self.gw = None
        self.mass_flux = None

    def __getitem__(self, lm):
        """Shortcut to access GW modes: det[2,2] -> det.gw[2,2]."""
        if self.gw is None:
            raise KeyError(f"No GW data attached to detector '{self.name}'")
        return self.gw[lm]

    def __repr__(self):
        parts = [f"grace_detector('{self.name}'"]
        if self.radius is not None:
            parts.append(f"r={self.radius}")
        if self.center is not None:
            parts.append(f"center={self.center}")
        attached = []
        if self.gw is not None:
            modes = self.gw.available_modes()
            attached.append(f"gw: {len(modes)} modes")
        if self.mass_flux is not None:
            types = sorted(self.mass_flux.available_vars())
            attached.append(f"mass_flux: {', '.join(types)}")
        if attached:
            parts.append(" | ".join(attached))
        return ", ".join(parts) + ")"


class grace_detector_set:
    """Dict-like container of grace_detector objects.

    Attributes:
        detectors (dict): Mapping name -> grace_detector.
    """

    def __init__(self):
        self._detectors = {}

    def __getitem__(self, name):
        return self._detectors[name]

    def __setitem__(self, name, detector):
        self._detectors[name] = detector

    def __contains__(self, name):
        return name in self._detectors

    def available_detectors(self):
        """Return sorted list of detector names."""
        return sorted(self._detectors.keys())

    def __repr__(self):
        if not self._detectors:
            return "grace_detector_set: (empty)"
        parts = [repr(self._detectors[n]) for n in self.available_detectors()]
        return "grace_detector_set:\n  " + "\n  ".join(parts)

    @staticmethod
    def from_parfile_config(config):
        """Build a detector set from the parsed simulation parameter file.

        Args:
            config (dict): Parsed YAML config (full parfile contents).

        Returns:
            grace_detector_set: Detectors with metadata populated from config.
        """
        dset = grace_detector_set()
        if config is None:
            return dset
        try:
            detectors = config["spherical_surfaces"]["spherical_detectors"]
        except (KeyError, TypeError):
            return dset

        for d in detectors:
            name = d.get("name", "unknown")
            det = grace_detector(
                name=name,
                radius=float(d["radius"]) if "radius" in d else None,
                center=(float(d.get("x_c", 0.0)),
                        float(d.get("y_c", 0.0)),
                        float(d.get("z_c", 0.0))),
                resolution=int(d["resolution"]) if "resolution" in d else None,
                sampling_policy=d.get("sampling_policy"),
            )
            dset[name] = det
        return dset
