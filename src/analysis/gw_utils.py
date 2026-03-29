import numpy as np 
import scipy.signal.windows as scw 

def fixed_frequency_integration(t, psi4, f0, window="tukey", wpars=(0.2)):
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
        -omega**2,
        -omega0**2
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