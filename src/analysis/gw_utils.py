import numpy as np
import scipy.signal.windows as scw
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar

def fixed_frequency_integration(t, psi4, f0, N=2, window="tukey", wpars=[0.2]):
    """
    Double integrate Psi4 -> h using FFI.
    f0 = cutoff frequency
    """
    dt = t[1] - t[0]
    n = len(t)

    # FFT
    if window == "tukey":
        w = scw.tukey(n, *wpars)
    elif window == "blackman":
        w = scw.blackman(n)
    elif window == None: 
        w = np.ones_like(psi4)
    
    #w = blackman(n)
    psi4_f = np.fft.fft(psi4*w)
    freqs = np.fft.fftfreq(n, dt)

    omega = 2 * np.pi * freqs

    # Avoid division by zero / low freq
    omega0 = 2 * np.pi * f0

    denom = np.where(
        np.abs(omega) > omega0,
        -omega**N,
        -omega0**N
    )

    h_f = psi4_f / denom

    # Back to time domain
    h = np.fft.ifft(h_f)

    return h

def get_phase(h):
    return np.unwrap(np.angle(h))

def get_inst_frequency(t,h):
    phi = get_phase(h)
    return np.gradient(phi,t[1]-t[0]) / (2 * np.pi)

def rstar(r,M):
    R = ( 1 + 0.5 * M / r ) ** 2 * r 
    return R + 2 * M * np.log(r/(2*M)-1)

def retarded_time(t,r,M):
    rs = rstar(r,M)
    return t-rs 
 
def align_waveforms(t, psi1, psi2, t1, t2, n_samples=2000):
    '''
    Align psi2 to psi1 in the interval [t1,t2] using the
    procedure outlined in Boyle et al 2009 (10.1103/physrevd.78.104020).
    2D fit over (dt, dphi); see align_waveforms_1d for the (preferred)
    1D variant that absorbs dphi analytically.
    - t        Common time grid of the two signals
    - psi1     Reference signal
    - psi2     Signal to be aligned
    - t1, t2   Alignment window
    - n_samples Number of trapezoid points over [t1, t2]
    - return: aligned waveform, aligned phase, time shift, phase shift
    '''
    phi1 = CubicSpline(t, np.unwrap(np.angle(psi1)))
    phi2 = CubicSpline(t, np.unwrap(np.angle(psi2)))
    A2   = CubicSpline(t, np.abs(psi2))

    tt = np.linspace(t1, t2, n_samples)
    phi1_tt = phi1(tt)

    def mismatch(params):
        dt, dphi = params
        diff = phi1_tt - phi2(tt - dt) - dphi
        return np.trapz(diff * diff, tt)

    sol = minimize(mismatch, x0=(0.0, 0.0), method='Nelder-Mead')
    dt_opt, dphi_opt = sol.x

    phi2_aligned = phi2(t - dt_opt) + dphi_opt
    psi2_aligned = A2(t - dt_opt) * np.exp(1j * phi2_aligned)
    return psi2_aligned, phi2_aligned, dt_opt, dphi_opt


def align_waveforms_1d(t, psi1, psi2, t1, t2, dt_bound=None, n_samples=2000):
    '''
    1D Boyle alignment: at fixed dt the optimal dphi minimizing
        Xi(dt, dphi) = int_{t1}^{t2} [phi1(t) - phi2(t - dt) - dphi]^2 dt
    is the windowed mean
        dphi(dt) = < phi1 - phi2(t - dt) >,
    so the 2D problem collapses to a 1D minimization of the *variance*
    of the phase difference over dt alone. More robust than the 2D fit:
    one parameter, no (dt, dphi) degeneracy.
    - dt_bound  search range |dt| <= dt_bound (default: half the window).
    - return: aligned waveform, aligned phase, time shift, phase shift
    '''
    phi1 = CubicSpline(t, np.unwrap(np.angle(psi1)))
    phi2 = CubicSpline(t, np.unwrap(np.angle(psi2)))
    A2   = CubicSpline(t, np.abs(psi2))

    tt = np.linspace(t1, t2, n_samples)
    phi1_tt = phi1(tt)

    def cost(dt):
        return np.var(phi1_tt - phi2(tt - dt))

    if dt_bound is None:
        dt_bound = 0.5 * (t2 - t1)
    sol = minimize_scalar(cost, bounds=(-dt_bound, dt_bound), method='bounded')
    dt_opt = sol.x
    dphi_opt = np.mean(phi1_tt - phi2(tt - dt_opt))

    phi2_aligned = phi2(t - dt_opt) + dphi_opt
    psi2_aligned = A2(t - dt_opt) * np.exp(1j * phi2_aligned)
    return psi2_aligned, phi2_aligned, dt_opt, dphi_opt

def nakano_extrap(t, rpsilm, Madm, r, l, f0):
    psi_dt = fixed_frequency_integration(t,rpsilm,f0,1)

    return (1-2*Madm/r) * ( rpsilm - (l-1)*(l+2)/(2*r) * psi_dt )