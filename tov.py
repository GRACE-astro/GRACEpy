#!/usr/bin/env python3

import argparse

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --------------------------------------------------
# GRACE cold-table header parsing
# --------------------------------------------------

GRACE_COLD_TABLE_HEADER_TAG_PREFIX = "GRACE cold EOS table v"
DEFAULT_COLUMNS = ["logrho", "logtemp", "ye", "logpress", "logeps", "cs2", "entropy"]


def parse_grace_cold_table_header(filename):
    """Return (metadata: dict[str,str], version: int) for a GRACE cold EOS table.

    v2 format (the only format supported as of the v1 GRACE release):
        # GRACE cold EOS table v2
        # key = value
        # ...
        <data>
    """
    meta = {}
    with open(filename, "r") as f:
        first = f.readline().lstrip("#").strip()

        if not first.startswith(GRACE_COLD_TABLE_HEADER_TAG_PREFIX):
            raise ValueError(
                f"{filename}: first line does not start with "
                f"{GRACE_COLD_TABLE_HEADER_TAG_PREFIX!r}. Legacy v1 free-form "
                f"headers are no longer supported — regenerate with current GRACEpy."
            )
        try:
            version = int(first[len(GRACE_COLD_TABLE_HEADER_TAG_PREFIX):])
        except ValueError:
            version = 2
        for line in f:
            if not line.startswith("#"):
                break
            s = line.lstrip("#").strip()
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.strip()] = v.strip()
        return meta, version


# --------------------------------------------------
# EOS class
# --------------------------------------------------

class TabulatedEOS:
    """Cold barotropic EOS read from a GRACE v2 cold table.

    Uses PCHIP for all interpolations so that phase-transition discontinuities
    in cs^2 / flat regions in P(rho) don't produce spurious oscillations.
    """

    def __init__(self, filename, verbose=True):
        self.filename = filename
        self.metadata, self.format_version = parse_grace_cold_table_header(filename)

        self.energy_shift = float(self.metadata.get("energy_shift", 0.0))
        self.baryon_mass  = (float(self.metadata["baryon_mass"])
                             if "baryon_mass" in self.metadata else None)
        cols_str = self.metadata.get("columns", " ".join(DEFAULT_COLUMNS))
        self.columns = cols_str.split()

        data = np.loadtxt(filename)
        idx = {c: i for i, c in enumerate(self.columns)}

        self.logrho   = data[:, idx["logrho"]]
        self.logpress = data[:, idx["logpress"]]
        self.logeps   = data[:, idx["logeps"]]
        self.cs2      = data[:, idx["cs2"]] if "cs2" in idx else None

        # Strict-monotonicity guard: PchipInterpolator requires strictly
        # increasing x.  For a sane v2 table this is already true.
        if not np.all(np.diff(self.logrho) > 0):
            raise ValueError(f"{filename}: logrho axis is not strictly increasing")

        eps     = np.exp(self.logeps) - self.energy_shift
        # Total energy density e = rho*(1+eps); store log e for the TOV ODE.
        self.logedens = np.log1p(eps) + self.logrho

        self._P_of_logrho = PchipInterpolator(self.logrho, self.logpress)
        self._e_of_logrho = PchipInterpolator(self.logrho, self.logedens)
        # Inversion P -> rho.  P-axis must be monotonic; warn if not (phase
        # transition can flatten it).  PCHIP only requires strictly increasing
        # x, so we pre-trim flat plateaus by averaging.
        lP = self.logpress.copy()
        lr = self.logrho.copy()
        nondec = np.diff(lP) > 0
        if not np.all(nondec):
            keep_mask = np.concatenate([[True], nondec])
            if verbose:
                n_drop = int(np.sum(~keep_mask))
                print(f"  WARN: {filename}: P(rho) flat/decreasing in {n_drop} points "
                      f"(phase transition?); collapsing to strictly increasing axis "
                      f"for the P->rho inverse.")
            lP, lr = lP[keep_mask], lr[keep_mask]
        self._logrho_of_logP = PchipInterpolator(lP, lr)

        if self.cs2 is not None:
            self._cs2_of_logrho = PchipInterpolator(self.logrho, self.cs2)
        else:
            self._cs2_of_logrho = None

        if verbose:
            self._print_summary()

    def _print_summary(self):
        rho_min, rho_max = np.exp(self.logrho[0]), np.exp(self.logrho[-1])
        P_min, P_max = np.exp(self.logpress[0]), np.exp(self.logpress[-1])
        bm = f"{self.baryon_mass:.6e}" if self.baryon_mass is not None else "n/a"
        print(f"loaded {self.filename}  (format v{self.format_version}, "
              f"{len(self.logrho)} points)")
        print(f"  log10 rho_geom in [{np.log10(rho_min):.3f}, {np.log10(rho_max):.3f}]")
        print(f"  log10 P_geom   in [{np.log10(P_min):.3f}, {np.log10(P_max):.3f}]")
        print(f"  energy_shift = {self.energy_shift:.6e}    baryon_mass = {bm}")

    # ---- thermodynamic accessors ---------------------------------------

    def pressure(self, rho):
        return np.exp(self._P_of_logrho(np.log(rho)))

    def energy_density(self, rho):
        return np.exp(self._e_of_logrho(np.log(rho)))

    def rho_from_P(self, P):
        return np.exp(self._logrho_of_logP(np.log(P)))

    def eps_from_P(self, P):
        # Returns total energy density e (NOT specific internal eps);
        # name is kept for back-compat with the old TOV driver.
        rho = self.rho_from_P(P)
        return self.energy_density(rho)

    def cs2_of(self, rho):
        if self._cs2_of_logrho is None:
            raise RuntimeError("cs2 column not present in this table")
        return float(self._cs2_of_logrho(np.log(rho)))


# --------------------------------------------------
# TOV system
# --------------------------------------------------

def tov_rhs(r, y, eos):
    m, P, Mb = y
    if (P <= 0) or (r < 1e-12):
        return [0, 0, 0]

    eps = eos.eps_from_P(P)
    rho = eos.rho_from_P(P)

    fac = 1.0 - 2.0 * m / r
    if fac <= 0:
        return [0, 0, 0]

    dmdr = 4.0 * np.pi * r**2 * eps
    dPdr = -(eps + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    dMbdr = 4.0 * np.pi * r**2 * rho / np.sqrt(fac)

    return [dmdr, dPdr, dMbdr]


# 1 g/cm^3 in GEOM (c=G=Msun=1).  Used as default for the matter-R
# threshold; converting the canonical 1e8 g/cm^3 gives ~1.62e-10.
_GEOM_PER_CGS_DENS = 1.0 / 6.1725855e17


def integrate_star(rho_c, eos, r_max=50.0, max_step=0.05, rtol=1e-9,
                   p_surface_rel=1e-10,
                   rho_matter_thresh=1e11 * _GEOM_PER_CGS_DENS):
    """Integrate a single TOV star.

    Returns (M, Mb, R_fluid, R_matter):
      - R_fluid: radius at which P drops below `p_surface_rel * P_c`
        (or the table floor, whichever is larger).  This is the
        "fluid surface" used to terminate the integration.
      - R_matter: radius at which rho drops below `rho_matter_thresh`
        (default 1e10 g/cm^3, i.e. above the neutron-drip transition
        and well above the fluid-surface termination for any P_c).
        NaN if the threshold is never crossed.

    The two often differ by O(0.5–1 km) for tables with a polytropic
    crust extension — the difference is the "ghost atmosphere" between
    a physical NS surface and where the table actually terminates in
    pressure.  Plot M(R_matter) for a smooth sequence.
    """
    Pc = eos.pressure(rho_c)
    y0 = [0.0, float(Pc), 0.0]

    P_floor_table = float(np.exp(eos.logpress[0])) * 1.001
    P_surface     = max(Pc * p_surface_rel, P_floor_table)

    def stop_surface(r, y):
        return y[1] - P_surface
    stop_surface.terminal  = True
    stop_surface.direction = -1

    def matter_crossing(r, y):
        # Non-terminal: just record the crossing.  Guard P>0 so we don't
        # call rho_from_P past the surface.
        if y[1] <= P_surface:
            return 1.0
        return float(eos.rho_from_P(y[1])) - rho_matter_thresh
    matter_crossing.terminal  = False
    matter_crossing.direction = -1

    # Per-component absolute tolerance: m and Mb are O(M_sun) ~ O(1),
    # P is O(P_c).  Without P_c-scaling, atol on P is many orders of
    # magnitude larger than the surface event value, so the integrator
    # detects termination with effectively no precision.
    atol_vec = [1e-12, max(Pc * 1e-12, 1e-30), 1e-12]

    sol = solve_ivp(
        lambda r, y: tov_rhs(r, y, eos),
        [0.0, r_max],
        y0,
        events=[stop_surface, matter_crossing],
        max_step=max_step,
        rtol=rtol,
        atol=atol_vec,
    )

    R_fluid  = sol.t[-1]
    M        = sol.y[0, -1]
    Mb       = sol.y[2, -1]
    R_matter = sol.t_events[1][0] if len(sol.t_events[1]) > 0 else np.nan
    return M, Mb, R_fluid, R_matter


# --------------------------------------------------
# Sequence + branch detection
# --------------------------------------------------

def tov_sequence(eos, n_points=80, log_rho_min=None, log_rho_max=None,
                 margin_lo=2.0, margin_hi=0.3, **kwargs):
    """Sweep central rest-mass density and integrate a TOV at each point.

    Parameters
    ----------
    eos : TabulatedEOS
    n_points : int
    log_rho_min, log_rho_max : float, optional
        Natural-log bounds on the central density. If None, defaults to
        (logrho.min() + margin_lo, logrho.max() - margin_hi) so the sweep
        skips the cold-table floor and any noisy high-density tail.
    margin_lo, margin_hi : float
        Default offsets from the table edges (in natural log of rho).

    Returns
    -------
    dict with keys: log_rho_c, rho_c, M, Mb, R, R_matter, stable (bool array).
    R is the fluid surface (P drops below `p_surface_rel * P_c`); R_matter
    is the radius at which rho crosses `rho_matter_thresh` (default
    ~1e8 g/cm^3).  R_matter is the smoother of the two for tables with a
    polytropic crust extension.
    """
    if log_rho_min is None:
        log_rho_min = eos.logrho[0]  + margin_lo
    if log_rho_max is None:
        log_rho_max = eos.logrho[-1] - margin_hi

    log_rho_c = np.linspace(log_rho_min, log_rho_max, n_points)
    rho_c = np.exp(log_rho_c)
    M  = np.full(n_points, np.nan)
    Mb = np.full(n_points, np.nan)
    R  = np.full(n_points, np.nan)
    R_matter = np.full(n_points, np.nan)

    r_max = kwargs.get("r_max", 50.0)
    for i, rc in enumerate(rho_c):
        try:
            Mi, Mbi, Ri, Rmi = integrate_star(rc, eos, **kwargs)
            # Reject runaways: if R hits the integration cap, the integrator
            # never found a surface — not a real star.
            if Ri > 0 and Mi > 0 and Ri < 0.999 * r_max:
                M[i], Mb[i], R[i], R_matter[i] = Mi, Mbi, Ri, Rmi
        except Exception as exc:
            print(f"  rho_c={rc:.3e} integration failed: {exc}")

    # Stability: dM/drho_c > 0.  NaN entries (failed/runaway) are not stable.
    dM = np.gradient(M, log_rho_c)
    stable = (dM > 0) & np.isfinite(M)

    return dict(
        log_rho_c=log_rho_c,
        rho_c=rho_c,
        M=M, Mb=Mb, R=R, R_matter=R_matter,
        dM_dlogrho=dM,
        stable=stable,
    )


def find_branches(seq, min_points=5, dM_tol_frac=1e-3, merge_frac=0.005):
    """Identify stable branches in a TOV sequence.

    Strategy: contiguous runs of `dM/dlogrho >= -tol` (where tol scales with
    M_max, so flat phase-transition plateaus count as stable rather than
    fragmenting branches), then suppress short noise runs (`min_points`)
    and merge adjacent branches whose M_max differ by less than
    `merge_frac * M_max_global` (collapses any residual spurious doublings).

    `min_points` typically 4–6 for N~100 sweeps.

    Returns a list of dicts ordered by rho_c.
    """
    M, R, rho_c, log_rho_c = seq["M"], seq["R"], seq["rho_c"], seq["log_rho_c"]
    R_matter = seq.get("R_matter")

    if not np.any(np.isfinite(M)):
        return []
    M_max_global = float(np.nanmax(M))
    dM_tol = M_max_global * dM_tol_frac
    # Forward differences: np.gradient's centered-diff smoothing averages
    # away the small instability dips that separate twin branches.
    dM_fwd = np.empty_like(M)
    dM_fwd[:-1] = np.diff(M)
    dM_fwd[-1]  = dM_fwd[-2]
    is_stable = (dM_fwd > -dM_tol) & np.isfinite(M)

    # Pass 1: contiguous stable regions, filtered by min_points
    raw = []
    i, n = 0, len(is_stable)
    while i < n:
        if not is_stable[i] or not np.isfinite(M[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and is_stable[j + 1] and np.isfinite(M[j + 1]):
            j += 1
        if (j - i + 1) >= min_points:
            kmax = i + int(np.argmax(M[i:j + 1]))
            raw.append((i, j, kmax))
        i = j + 1

    # Pass 2: merge adjacent branches with near-identical M_max
    merge_dM = M_max_global * merge_frac
    merged = []
    for entry in raw:
        if merged and abs(M[entry[2]] - M[merged[-1][2]]) < merge_dM:
            i_s, j_s, k_s = merged[-1]
            new_k = k_s if M[k_s] >= M[entry[2]] else entry[2]
            merged[-1] = (i_s, entry[1], new_k)
        else:
            merged.append(entry)

    branches = []
    for (i_s, j_s, k_s) in merged:
        br = dict(
            i_start=i_s, i_end=j_s,
            rho_c_start=rho_c[i_s], rho_c_end=rho_c[j_s],
            M_max=M[k_s], R_at_M_max=R[k_s], rho_c_at_M_max=rho_c[k_s],
        )
        if R_matter is not None:
            br["R_matter_at_M_max"] = R_matter[k_s]
        branches.append(br)
    return branches


# --------------------------------------------------
# Single-target inversion (kept for back-compat)
# --------------------------------------------------

def find_mass(M_target, mode, eos, branch=None):
    """Find rho_c giving the requested mass.

    For multi-valued M(rho_c) (twin stars), pass `branch` (dict from
    find_branches) to restrict the bracket to one stable branch.
    """
    if branch is None:
        logrho_min = float(np.min(eos.logrho))
        logrho_max = float(np.max(eos.logrho))
    else:
        logrho_min = float(np.log(branch["rho_c_start"]))
        logrho_max = float(np.log(branch["rho_c_at_M_max"]))

    def f(logrho_c):
        rho_c = np.exp(logrho_c)
        M, Mb, _, _ = integrate_star(rho_c, eos)
        if mode == "ADM":     return M  - M_target
        if mode == "Baryon":  return Mb - M_target
        raise ValueError(f"unknown mode {mode}")

    root  = brentq(f, logrho_min, logrho_max)
    rho_c = np.exp(root)
    e_c   = eos.energy_density(rho_c)
    p_c   = eos.pressure(rho_c)
    h_c   = (e_c + p_c) / rho_c
    M, Mb, R, R_matter = integrate_star(rho_c, eos)
    return M, Mb, R, rho_c, np.log(h_c), R_matter


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRACE cold-table TOV solver")
    parser.add_argument("eos_file", help="GRACE v2 cold EOS table")
    parser.add_argument("--M",        type=float, default=None,
                        help="Target mass [Msun] (single-star mode)")
    parser.add_argument("--mode",     type=str,   default="ADM",
                        help="ADM or Baryon (used with --M)")
    parser.add_argument("--sequence", action="store_true",
                        help="Build a M(rho_c), M(R) sequence and dump it")
    parser.add_argument("--n",        type=int,   default=120,
                        help="Number of sequence points")
    parser.add_argument("--log_rho_min", type=float, default=None)
    parser.add_argument("--log_rho_max", type=float, default=None)
    parser.add_argument("--out",      type=str,   default=None,
                        help="Path for sequence text output (default: <eos>.tov_seq)")
    args = parser.parse_args()

    eos = TabulatedEOS(args.eos_file)

    if args.sequence:
        seq = tov_sequence(eos, n_points=args.n,
                           log_rho_min=args.log_rho_min,
                           log_rho_max=args.log_rho_max)
        branches = find_branches(seq)
        print(f"\nFound {len(branches)} stable branch(es):")
        for b, br in enumerate(branches):
            rm = br.get("R_matter_at_M_max", float("nan"))
            print(f"  branch {b}: rho_c in [{br['rho_c_start']:.3e}, "
                  f"{br['rho_c_end']:.3e}]   "
                  f"M_max = {br['M_max']:.4f}  at rho_c = {br['rho_c_at_M_max']:.3e}, "
                  f"R = {br['R_at_M_max']:.4f}  R_matter = {rm:.4f}")

        out = args.out or (args.eos_file + ".tov_seq")
        arr = np.column_stack([seq["log_rho_c"], seq["rho_c"], seq["M"],
                               seq["Mb"], seq["R"], seq["R_matter"],
                               seq["stable"].astype(float)])
        header = "log_rho_c  rho_c  M_ADM  M_baryon  R_fluid  R_matter  stable"
        np.savetxt(out, arr, header=header)
        print(f"\nwrote sequence to {out}")
        return

    if args.M is None:
        parser.error("either --M or --sequence is required")

    M, Mb, R, rho_c, h_c, R_matter = find_mass(args.M, args.mode, eos)
    p_c = eos.pressure(rho_c)
    e_c = eos.energy_density(rho_c)
    print(f"M_ADM    = {M:.6f}")
    print(f"M_bary   = {Mb:.6f}")
    print(f"R_fluid  = {R:.6f}")
    print(f"R_matter = {R_matter:.6f}")
    print(f"rho_c    = {rho_c:.16e}")
    print(f"p_c      = {p_c:.16e}")
    print(f"e_c      = {e_c:.16e}")
    print(f"h_c      = {h_c:.16e}")


if __name__ == "__main__":
    main()
