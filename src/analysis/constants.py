#src/analysis/constants.py
## Some useful constants
import numpy as np 


# Physical constants in SI 
G_si       = 6.6738e-11           # m^3/(kg s^2)
c_si       = 299792458.0          # m/s
Msun_si    = 1.9885e30            # kg
mu0_si     = 1.256637061435917e-6 # Newton/Ampere^2
Kb_si      = 1.3806488e-23        # Joule/K
Mparsec_si = 3.08567758e22        # m
parsec_si  = Mparsec_si * 1e-6    # m
fm_si      = 1e-15                # m
e_si       = 1.602176634e-19      # Coulomb
h_si       = 6.62607015e-34       # m^2 kg / s 

# CGS 
c_cgs     = c_si * 1e2      # cm/s
G_cgs     = 6.67430e-8      # cm^3/(g s^2) 
e_cgs     = 4.80320425e-10  # statC
sigma_cgs = 5.670374419e-5  # erg cmˆ-2 sˆ-1 Kˆ-4 
Msun_cgs  = 1.988475e33     # g 
rad_cgs   = 7.5646e-15      # 4 sigma / c [ erg / cm^3 / K ]
h_cgs     = h_si * 1e7      # cm^2 g / s 


# Weird units 
# Planck constant eV s 
h_eVs = 6.582119569e-16 
# Boltzmann constant 
k_evKm1 = 8.617333262e-5 


# Particle masses 
me_MeV = 0.51099895069 # electron
mp_MeV = 938.27208943  # proton 
mn_MeV = 939.56542194  # neutron 

# Fine structure constant 
alpha_fine = 1./137.

# Conversions 
erg_to_J     = 1e-7                  # Joule 
eV_to_J      = 1.6021766208e-19      # Joule
MeV_to_J     = 1.6021766208e-13      # Joule
eV_to_kg     = eV_to_J / c_si**2     # Kg 
MeV_to_kg    = MeV_to_J / c_si**2    # Kg 
eV_to_erg    = eV_to_J / erg_to_J    # erg 
MeV_to_erg   = MeV_to_J / erg_to_J   # erg 
eV_to_g      = eV_to_erg / c_cgs**2  # g 
MeV_to_g     = MeV_to_erg / c_cgs**2 # g 

# Particle masses 
me_si  = me_MeV * MeV_to_kg # kg 
me_cgs = me_MeV * MeV_to_g  # g 
mp_si  = me_MeV * MeV_to_kg # kg 
mp_cgs = me_MeV * MeV_to_g  # g 
mn_si  = me_MeV * MeV_to_kg # kg 
mn_cgs = me_MeV * MeV_to_g  # g 

# Convenience 
CU_to_m     = G_si * Msun_si / c_si**2 
CU_to_s     = CU_to_m / c_si 
CU_to_ms    = CU_to_s * 1e3 
CU_to_cm    = CU_to_m * 1e2 
CU_to_J     = Msun_si * c_si**2 
CU_to_erg   = Msun_cgs * c_cgs**2 
CU_to_Gauss = c_si**4 / Msun_si / G_si**(1.5)* mu0_si**(0.5) * 10000
CU_to_Tesla = c_si**4 / Msun_si / G_si**(1.5)* mu0_si**(0.5)
# 