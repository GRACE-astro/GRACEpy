"""Render a movie of a GRACE simulation from xy- and xz-plane descriptors.

Reads the cell-centered data referenced by two XMF descriptors (one for the
xy plane, one for the xz plane) and produces a two-panel movie, one frame per
available output time.  The default field is the rest-mass density ``rho``
shown on a base-10 log scale in g/cm^3.

The cell-centered output is plotted directly with ``matplotlib.tricontourf``
(Delaunay triangulation of the cell centers) -- no resampling onto a uniform
grid is required.

Example
-------
    make_movie xy_plane_descriptor.xmf xz_plane_descriptor.xmf -o rho.mp4
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from grace_tools.vtk_reader_utils import grace_xmf_reader
import grace_tools.mhd_utils as mhd
from eos.units_system import GEOM_UNIT_SYSTEM, CGS_UNIT_SYSTEM
from analysis.constants import CU_to_ms, CU_to_Gauss

# Conversion factor: rho[geometric] * RHO_GEOM_TO_CGS = rho[g/cm^3]
RHO_GEOM_TO_CGS = (GEOM_UNIT_SYSTEM / CGS_UNIT_SYSTEM).dens


# ── derived-field getters ─────────────────────────────────────────────────
# A getter takes (reader, time) and returns (coords, raw_field) in code units,
# mirroring grace_xmf_reader.get_var so it slots into the same plotting path.
def _get_bnorm(reader, time):
    """|b| = sqrt(b_mu b^mu): comoving magnetic field magnitude (code units)."""
    coords, b2 = mhd.comoving_b_squared(reader, time)
    return coords, np.sqrt(b2)

# Presets for known output fields.  Each entry is a dict with:
#   conv  : multiply the raw (code-unit) field by this to get physical units
#   log   : plot log10 of the (converted) field if True, else linear
#   floor : clip to this value before taking log (ignored when log=False)
#   label : colorbar label
# Anything not listed here is plotted linearly in code units (conv=1) with the
# variable name as the label.  Override any of these from the CLI.
#
# NOTE: GRACE stores temperature as k_B*T in MeV, so it needs no unit
# conversion (conv=1) and is shown linearly.  Check the exact field name in
# your output with reader.available_variables("cell") -- it may be "temp",
# "temperature", or "T"; pass it via --var and the temperature preset is keyed
# under each of those spellings below.
# Specific internal energy: dimensionless in code units (eps = u/rho), plotted
# linearly.  Pass --log if you prefer a log scale.
_EPS_SPEC = dict(conv=1.0, log=False, floor=0.0,
                 label=r"$\epsilon\ \mathrm{[code]}$")
_TEMP_SPEC = dict(conv=1.0, log=False, floor=0.0, label=r"$T\ \mathrm{[MeV]}$")
_B_SPEC = dict(conv=CU_to_Gauss, log=True, floor=1e-30,
               label=r"$\log_{10}\,|b|\ \mathrm{[G]}$", getter=_get_bnorm)
VAR_SPECS = {
    "rho":         dict(conv=RHO_GEOM_TO_CGS, log=True, floor=1.0,
                        label=r"$\log_{10}\,\rho\ \mathrm{[g\,cm^{-3}]}$"),
    "temperature": _TEMP_SPEC,
    "temp":        _TEMP_SPEC,
    "T":           _TEMP_SPEC,
    "press":       dict(conv=1.0, log=True, floor=1e-30,
                        label=r"$\log_{10}\,P\ \mathrm{[code]}$"),
    "eps":         _EPS_SPEC,
    "epsilon":     _EPS_SPEC,
    # derived: |b| = sqrt(b^2), comoving B-field magnitude, in Gauss on a
    # log scale.  Needs Bvec, zvec (+ metric) present in the plane output.
    "b":           _B_SPEC,
    "bnorm":       _B_SPEC,
    "sqrt_b2":     _B_SPEC,
}


def _resolve_spec(varname, args):
    """Build the plot spec for ``varname``, applying any CLI overrides."""
    spec = dict(VAR_SPECS.get(varname,
                              dict(conv=1.0, log=False, floor=0.0,
                                   label=varname)))
    if args.conv is not None:
        spec["conv"] = args.conv
    if args.log is not None:          # --log / --linear set this explicitly
        spec["log"] = args.log
    if args.floor is not None:
        spec["floor"] = args.floor
    if args.label is not None:
        spec["label"] = args.label
    return spec


def _plane_field(reader, varname, time, axes, extent, spec):
    """Return (h, v, field) for a plane at the given time, clipped to extent.

    ``axes`` selects which coordinate columns map to the horizontal/vertical
    plot axes, e.g. (0, 1) for xy and (0, 2) for xz.  The returned field is
    already converted to physical units and log10'd if ``spec["log"]``.
    A ``getter`` in the spec (for derived fields) overrides ``reader.get_var``.
    """
    getter = spec.get("getter")
    if getter is None:
        coords, field = reader.get_var(varname, time=time)
    else:
        coords, field = getter(reader, time)
    h = coords[:, axes[0]]
    v = coords[:, axes[1]]

    hmin, hmax, vmin, vmax = extent
    # small margin so triangles spanning the edge are kept
    mh = 0.02 * (hmax - hmin)
    mv = 0.02 * (vmax - vmin)
    keep = (
        (h >= hmin - mh) & (h <= hmax + mh) &
        (v >= vmin - mv) & (v <= vmax + mv)
    )
    h, v, field = h[keep], v[keep], field[keep]

    field = np.asarray(field, dtype=float) * spec["conv"]
    if spec["log"]:
        field = np.log10(np.maximum(field, spec["floor"]))
    return h, v, field


def _autoscale(readers_axes, varname, times, spec, pct):
    """Estimate (vmin, vmax) from a few sampled times across both panels."""
    sample_idx = np.unique(np.linspace(0, len(times) - 1, 5).astype(int))
    lo, hi = np.inf, -np.inf
    for i in sample_idx:
        for reader, axes, extent in readers_axes:
            _, _, f = _plane_field(reader, varname, times[i], axes, extent, spec)
            lo = min(lo, np.percentile(f, pct))
            hi = max(hi, np.percentile(f, 100 - pct))
    return float(lo), float(hi)


def _build_drawer(xy, xz, args, spec, vmin, vmax, levels):
    """Create the two-panel figure and a ``draw_frame(time)`` closure.

    Shared by the serial path and by each parallel worker so the rendering is
    byte-for-byte identical regardless of how the frames are distributed.
    Returns ``(fig, draw_frame)``.
    """
    readers_axes = [
        (xy, (0, 1), args.xy_extent),
        (xz, (0, 2), args.xz_extent),
    ]
    fig, (ax_xy, ax_xz) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")
    ax_xy.set(xlim=args.xy_extent[:2], ylim=args.xy_extent[2:],
              xlabel="x [M]", ylabel="y [M]", title="xy plane")
    ax_xz.set(xlim=args.xz_extent[:2], ylim=args.xz_extent[2:],
              xlabel="x [M]", ylabel="z [M]", title="xz plane")

    sm = plt.cm.ScalarMappable(
        cmap=args.cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=[ax_xy, ax_xz], fraction=0.046, pad=0.04)
    cbar.set_label(spec["label"])
    suptitle = fig.suptitle("")

    def draw_frame(time):
        for ax in (ax_xy, ax_xz):
            # clear only the contour collections, keep axes/labels/limits
            for coll in list(ax.collections):
                coll.remove()
        for ax, (reader, axes, extent) in zip((ax_xy, ax_xz), readers_axes):
            h, v, lf = _plane_field(reader, args.var, time, axes, extent, spec)
            ax.tricontourf(h, v, lf, levels=levels, cmap=args.cmap,
                           vmin=vmin, vmax=vmax, extend="both")
        suptitle.set_text(f"t = {time * CU_to_ms:.3f} ms")

    return fig, draw_frame


# ── parallel-worker state (one set per process) ───────────────────────────
_WORKER = {}


def _worker_init(xy_desc, xz_desc, args, spec, vmin, vmax, levels, frame_dir):
    """Pool initializer: build this process's own readers + figure once."""
    matplotlib.use("Agg")
    xy = grace_xmf_reader(xy_desc)
    xz = grace_xmf_reader(xz_desc)
    fig, draw = _build_drawer(xy, xz, args, spec, vmin, vmax, levels)
    _WORKER.update(fig=fig, draw=draw, dpi=args.dpi, frame_dir=frame_dir)


def _worker_render(item):
    """Render one frame (global index, time) to a PNG; return the index."""
    import os
    i, time = item
    _WORKER["draw"](time)
    _WORKER["fig"].savefig(
        os.path.join(_WORKER["frame_dir"], f"frame_{i:05d}.png"),
        dpi=_WORKER["dpi"])
    return i


def _assemble_movie(frame_dir, fps, output):
    """Stitch ``frame_%05d.png`` into ``output`` with ffmpeg. Returns success."""
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{frame_dir}/frame_%05d.png",
        # pad to even dimensions so libx264/yuv420p is happy
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", output,
    ]
    subprocess.run(cmd, check=True)
    return True


def _render_parallel(times, xy_desc, xz_desc, args, spec, vmin, vmax, levels):
    """Render frames across ``args.jobs`` processes, then assemble with ffmpeg."""
    import os, shutil
    from multiprocessing import Pool

    njobs = args.jobs
    if njobs <= 0:  # 0 / negative => use all available CPUs
        njobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1
    njobs = min(njobs, len(times))

    frame_dir = os.path.splitext(args.output)[0] + "_frames"
    os.makedirs(frame_dir, exist_ok=True)
    print(f"Rendering {len(times)} frames across {njobs} processes "
          f"-> {frame_dir}/")

    items = list(enumerate(times))
    init_args = (xy_desc, xz_desc, args, spec, vmin, vmax, levels, frame_dir)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with Pool(njobs, initializer=_worker_init, initargs=init_args) as pool:
        result = pool.imap_unordered(_worker_render, items)
        if tqdm is not None:
            result = tqdm(result, total=len(items), desc="rendering")
        for _ in result:
            pass

    if _assemble_movie(frame_dir, args.fps, args.output):
        if not args.keep_frames:
            shutil.rmtree(frame_dir)
        print(f"Wrote {args.output}")
    else:
        print(f"ffmpeg unavailable; kept {len(items)} PNG frames in {frame_dir}/.\n"
              f"Assemble with:\n"
              f"  ffmpeg -framerate {args.fps} -i "
              f"{frame_dir}/frame_%05d.png -pix_fmt yuv420p {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a two-panel (xy / xz) movie of a GRACE simulation.",
        epilog="Example:\n"
               "  make_movie xy_plane_descriptor.xmf xz_plane_descriptor.xmf \\\n"
               "      -o rho.mp4 --var rho --fps 20",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("xy_descriptor", type=str,
                        help="XMF descriptor for the xy-plane data.")
    parser.add_argument("xz_descriptor", type=str,
                        help="XMF descriptor for the xz-plane data.")
    parser.add_argument("-o", "--output", type=str, default="movie.mp4",
                        help="Output movie file (default: movie.mp4).")
    parser.add_argument("--var", type=str, default="rho",
                        help="Output field to plot (default: rho). Known "
                             "presets: rho, temperature/temp/T, press, "
                             "eps/epsilon, b/bnorm/sqrt_b2 (=sqrt(b^2), |b| in "
                             "Gauss). "
                             "Any other field name is plotted linearly in code "
                             "units; use --conv/--log/--label to customize.")
    parser.add_argument("--xy-extent", type=float, nargs=4,
                        default=[-64.0, 64.0, -64.0, 64.0],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="xy-plane extent (default: -64 64 -64 64).")
    parser.add_argument("--xz-extent", type=float, nargs=4,
                        default=[-64.0, 64.0, 0.0, 32.0],
                        metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
                        help="xz-plane extent (default: -64 64 0 32).")
    parser.add_argument("--levels", type=int, default=200,
                        help="Number of filled contour levels (default: 200).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Lower color limit (in plotted units). Default: auto.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Upper color limit (in plotted units). Default: auto.")
    # ── per-variable overrides (default to the preset for --var) ───────────
    parser.add_argument("--conv", type=float, default=None,
                        help="Override the code->physical unit conversion factor.")
    parser.add_argument("--label", type=str, default=None,
                        help="Override the colorbar label.")
    parser.add_argument("--floor", type=float, default=None,
                        help="Override the floor applied before log10.")
    log_grp = parser.add_mutually_exclusive_group()
    log_grp.add_argument("--log", dest="log", action="store_true", default=None,
                         help="Force a log10 color scale.")
    log_grp.add_argument("--linear", dest="log", action="store_false",
                         help="Force a linear color scale.")
    parser.add_argument("--cmap", type=str, default="inferno",
                        help="Matplotlib colormap (default: inferno).")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second (default: 15).")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output resolution in dpi (default: 120).")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Render frames in parallel across this many "
                             "processes (writes PNGs, then ffmpeg-assembles). "
                             "0 = all available CPUs (SLURM_CPUS_PER_TASK or "
                             "os.cpu_count). Default: 1 (serial streaming).")
    parser.add_argument("--keep-frames", action="store_true",
                        help="With --jobs, keep the intermediate PNG frames "
                             "instead of deleting them after assembly.")
    parser.add_argument("--tmin", type=float, default=None,
                        help="Only render times >= tmin (geometric units).")
    parser.add_argument("--tmax", type=float, default=None,
                        help="Only render times <= tmax (geometric units).")
    args = parser.parse_args()

    spec = _resolve_spec(args.var, args)

    xy = grace_xmf_reader(args.xy_descriptor)
    xz = grace_xmf_reader(args.xz_descriptor)

    # Use the xy reader's time axis as the master list; clip to [tmin, tmax].
    times = np.asarray(xy.available_times(), dtype=float)
    if args.tmin is not None:
        times = times[times >= args.tmin]
    if args.tmax is not None:
        times = times[times <= args.tmax]
    if len(times) == 0:
        parser.error("No output times to render in the requested range.")
    print(f"Rendering {len(times)} frames "
          f"[t = {times[0]:.2f} .. {times[-1]:.2f} M]")

    readers_axes = [
        (xy, (0, 1), args.xy_extent),
        (xz, (0, 2), args.xz_extent),
    ]

    # ── color scale ───────────────────────────────────────────────────────
    if args.vmin is None or args.vmax is None:
        vmin_a, vmax_a = _autoscale(readers_axes, args.var, times, spec, pct=1.0)
        vmin = args.vmin if args.vmin is not None else vmin_a
        vmax = args.vmax if args.vmax is not None else vmax_a
    else:
        vmin, vmax = args.vmin, args.vmax
    scale = "log10" if spec["log"] else "linear"
    print(f"Color limits: [{vmin:.3g}, {vmax:.3g}] ({scale})")
    levels = np.linspace(vmin, vmax, args.levels)

    # ── parallel path: render PNGs across processes, then assemble ─────────
    if args.jobs != 1:
        _render_parallel(times, args.xy_descriptor, args.xz_descriptor,
                         args, spec, vmin, vmax, levels)
        return

    # ── serial path: stream frames straight into ffmpeg ───────────────────
    fig, draw_frame = _build_drawer(xy, xz, args, spec, vmin, vmax, levels)

    try:
        from tqdm import tqdm
        frame_iter = tqdm(times, desc="rendering")
    except ImportError:
        frame_iter = times

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=args.fps, metadata={"artist": "GRACEpy"})
    try:
        with writer.saving(fig, args.output, dpi=args.dpi):
            for time in frame_iter:
                draw_frame(time)
                writer.grab_frame()
    except (FileNotFoundError, RuntimeError) as exc:
        # ffmpeg missing — fall back to a directory of PNG frames
        import os
        frame_dir = os.path.splitext(args.output)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        print(f"\nffmpeg unavailable ({exc}); writing PNG frames to {frame_dir}/")
        for i, time in enumerate(times):
            draw_frame(time)
            fig.savefig(os.path.join(frame_dir, f"frame_{i:05d}.png"),
                        dpi=args.dpi)
        print(f"Wrote {len(times)} frames. Assemble with e.g.:\n"
              f"  ffmpeg -framerate {args.fps} -i "
              f"{frame_dir}/frame_%05d.png -pix_fmt yuv420p {args.output}")
        return

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
