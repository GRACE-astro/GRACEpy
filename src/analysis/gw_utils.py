import numpy as np 
import scipy.signal.windows as scw 
from scipy.interpolate import CubicSpline 
from scipy.integrate import quad
from scipy.optimize import minimize

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
 
def align_waveforms(t, psi1, psi2, t1, t2):
    '''
    Align psi2 to psi1 in the interval [t1,t2] using the 
    procedure outlined in Boyle et al 2009 (10.1103/physrevd.78.104020)
    - t Common time grid of the two signals 
    - psi1 Signal to which the other will be aligned 
    - psi2 Signal that will be aligned 
    - t1 Start of alignment window 
    - t2 End of alignment window 
    - return: aligned waveform, aligned waveform's phase, time shift, phase shift
    '''
    # Follows Boyle+ 2009, Eq 77-78

    phi1 = CubicSpline(t, np.unwrap(np.angle(psi1)))
    phi2 = CubicSpline(t, np.unwrap(np.angle(psi2)))
    A2   = CubicSpline(t, np.abs(psi2))

    def mismatch(params):
        dt, dphi = params
        integrand = lambda tt: (phi1(tt) - phi2(tt - dt) - dphi)**2
        return quad(integrand, t1, t2)[0]

    sol = minimize(mismatch, x0=(0.0, 0.0), method='Nelder-Mead')
    dt_opt, dphi_opt = sol.x

    # Eq 78: shift and rotate psi2
    phi2_aligned = (phi2(t - dt_opt) - dphi_opt)
    psi2_aligned = A2(t - dt_opt) * np.exp(-1j * phi2_aligned)
    return psi2_aligned, phi2_aligned, dt_opt, dphi_opt

def nakano_extrap(t, rpsilm, Madm, r, l, f0):
    psi_dt = fixed_frequency_integration(t,rpsilm,f0,1)

    return (1-2*Madm/r) * ( rpsilm - (l-1)*(l+2)/(2*r) * psi_dt )