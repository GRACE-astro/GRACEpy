#!/usr/bin/env python3

import numpy as np
import argparse
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --------------------------------------------------
# EOS handling
# --------------------------------------------------

import re

def extract_energy_shift(line: str) -> float:
    """
    Extract the energy shift value from a GRACEpy header line.

    Parameters
    ----------
    line : str
        Line containing 'energy shift <value>'.

    Returns
    -------
    float
        Extracted energy shift.

    Raises
    ------
    ValueError
        If the pattern is not found.
    """
    match = re.search(r'energy shift\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', line)
    if not match:
        raise ValueError("Energy shift not found in line")

    return float(match.group(1))


class TabulatedEOS:
    def __init__(self, filename):
        with open(filename, "r") as f:
            first_line = f.readline()
            self.energy_shift = extract_energy_shift(first_line)
        
        data = np.loadtxt(filename, skiprows=2)

        # assume columns: logrho + variables
        self.logrho = data[:, 0]
        self.logt   = data[:,1]
        self.ye     = data[:,2]
        self.press  = data[:, 3]   # log(P)
        eps         = np.exp(data[:, 4]) - self.energy_shift  # log(epsilon)
        self.edens  = np.log(1+eps) + self.logrho 


        # build splines in logrho
        self.P_of_logrho = CubicSpline(self.logrho, self.press)
        self.e_of_logrho = CubicSpline(self.logrho, self.edens)

        # inverse: logP -> logrho
        self.logP_grid = self.press
        self.logrho_of_logP = CubicSpline(self.logP_grid, self.logrho)

    def pressure(self, rho):
        return np.exp(self.P_of_logrho(np.log(rho)))

    def energy_density(self, rho):
        return np.exp(self.e_of_logrho(np.log(rho)))

    def rho_from_P(self, P):
        return np.exp(self.logrho_of_logP(np.log(P)))

    def eps_from_P(self, P):
        rho = self.rho_from_P(P)
        return self.energy_density(rho)

# --------------------------------------------------
# TOV system
# --------------------------------------------------

def tov_rhs(r, y, eos):
    m, P, Mb = y

    if (P <= 0) or (r < 1e-12):
        return [0, 0, 0]

    eps = eos.eps_from_P(P)
    rho = eos.rho_from_P(P)

    fac = 1.0 - 2.0*m/r
    if fac <= 0:
        return [0, 0, 0]

    dmdr = 4.0 * np.pi * r**2 * eps
    dPdr = -(eps + P) * (m + 4*np.pi*r**3*P) / (r * (r - 2*m))
    dMbdr = 4.0 * np.pi * r**2 * rho / np.sqrt(fac)

    return [dmdr, dPdr, dMbdr]

# --------------------------------------------------
# Single TOV integration
# --------------------------------------------------

def integrate_star(rho_c, eos):
    Pc = eos.pressure(rho_c)

    r0  = 0
    m0  = 0
    Mb0 = 0

    y0 = [m0, Pc, Mb0]

    def stop_surface(r, y):
        return y[1]
    stop_surface.terminal = True
    stop_surface.direction = -1

    sol = solve_ivp(
        lambda r, y: tov_rhs(r, y, eos),
        [r0, 50],
        y0,
        events=stop_surface,
        max_step=0.1,
        rtol=1e-6,
        atol=1e-8
    )

    R = sol.t[-1]
    M = sol.y[0, -1]
    Mb = sol.y[2, -1]

    return M, Mb, R

# --------------------------------------------------
# Invert M -> Mb
# --------------------------------------------------

def find_mass(M_target, mode, eos):

    def f_MADM(logrho_c):
        rho_c = np.exp(logrho_c)
        M, Mb, _ = integrate_star(rho_c, eos)
        return M - M_target

    def f_Mb(logrho_c):
        rho_c = np.exp(logrho_c)
        M, Mb, _ = integrate_star(rho_c, eos)
        return Mb - M_target

    # bracket in logrho
    logrho_min = np.min(eos.logrho)
    logrho_max = np.max(eos.logrho)

    if mode == "ADM": root = brentq(f_MADM, logrho_min, logrho_max)
    if mode == "Baryon": root = brentq(f_Mb, logrho_min, logrho_max)

    rho_c = np.exp(root)
    e_c = eos.energy_density(rho_c)
    p_c = eos.pressure(rho_c)
    h_c = e_c/rho_c + p_c/rho_c
    M, Mb, R = integrate_star(rho_c, eos)

    return M, Mb, R, rho_c, np.log(h_c)

# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eos_file", help="EOS table")
    parser.add_argument("--M", type=float, required=True,
                        help="Target mass (in Msun units)")
    parser.add_argument("--mode", type=str, required=False, default="ADM",
                        help="ADM or Baryon")

    args = parser.parse_args()

    eos = TabulatedEOS(args.eos_file)

    M, Mb, R, rho_c, h_c = find_mass(args.M, args.mode, eos)
    p_c = eos.pressure(rho_c)
    e_c = eos.energy_density(rho_c)
    print(f"M_ADM   = {M:.6f}")
    print(f"M_bary  = {Mb:.6f}")
    print(f"R       = {R:.6f}")
    print(f"rho_c   = {rho_c:.16e}")
    print(f"p_c     = {p_c:.16e}")
    print(f"e_c     = {e_c:.16e}")
    print(f"h_c     = {h_c:.16e}")

if __name__ == "__main__":
    main()
