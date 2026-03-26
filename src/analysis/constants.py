#src/analysis/constants.py
## Some useful constants
import numpy as np 



# Units are CGS unless specified 
# speed of light
c_cgs = 29979245800
# Planck constant eV s 
h_eVs = 6.582119569e-16 
# Boltzmann constant 
k_evKm1 = 8.617333262e-5 
# Charge constant (C)
e_charge = 1.602176634e-19 
# statC 
e_cgs = 4.80320425e-10 
# Fine structure constant 
alpha_fine = 1./137.
# Stefan Boltzmann constant 
sigma_cgs = 5.670374419e-5 # erg cmˆ-2 sˆ-1 Kˆ-4 
# Solar mass 
Msun_cgs = 1.988475e33 
# G constant 
G_cgs = 6.67430e-8 
G_pcMsunm1 = 4.3009172706e-3 # in parsec / Msun (km/s)ˆ2 
# Electron mass 
me_MeV = 0.51099895069
me_KeV = 510.99895069
# Magnetic
mu0 = 1.2566370614359173e-6 # Vacuum permeability
eps0 = 1.0/(mu0*c_cgs**2) 
# Proton mass 
mp_MeV = 938.27208943 
mn_MeV = 939.56542194

# SI units 
c_si = c_cgs * 1e-2
# Mass 
Msun_si = Msun_cgs * 1e-3 
# G_si 
G_si = G_cgs * 1e-3 

# conversions 

# Length
cm_to_m = 1e2 
fm_to_m = 1e-15
fm_to_cm = 1e-13 
m_to_cm = 1e-2 
cm_to_km = 1e5 
km_to_cm = 1e-5
mum_to_cm = 1e-4
cm_to_mum = 1e4 
pc_to_km = 3.0857e13 
AU_to_km = 149597870.7
ly_to_km = 9460730472580.8 
km_to_pc = 1/pc_to_km
angstrom_to_nm = 0.1 
nm_to_angstrom = 10 
# Assuming c = G = Msun = 1  
Msun_to_cm = G_cgs * Msun_cgs / c_cgs**2
cm_to_Msun = 1./Msun_to_cm 
Msun_to_pc = Msun_to_cm * cm_to_km * km_to_pc
# B field
CU_to_Gauss = (1.0/Msun_to_cm/np.sqrt(eps0*G_cgs/(c_cgs**2))) / 1e09 ;

# Time 
hour_to_s = 60 * 60 
day_to_s = 24*hour_to_s 
year_to_s = 365 * day_to_s 
s_to_year = 1/year_to_s
Msun_to_s = Msun_to_cm / c_cgs 
s_to_Msun = 1./Msun_to_s
ms_to_Msun = s_to_Msun * 1e-3 
Msun_to_ms = 1e03 * Msun_to_s

# Temperature 
eV_to_K = 1.0/k_evKm1
keV_to_K = eV_to_K * 1e03 
MeV_to_K = keV_to_K * 1e03 
# Energy 
J_to_erg = 1e07 
erg_to_J = 1/J_to_erg
eV_to_erg = e_charge * J_to_erg
erg_to_eV = 1/eV_to_erg
erg_to_keV = erg_to_eV * 1e-3 
erg_to_MeV = erg_to_eV * 1e-6
MeV_to_erg = 1/erg_to_MeV

# Mass 
eV_to_g = eV_to_erg/c_cgs**2
MeV_to_g = eV_to_g * 1e6 
MeV_to_kg = MeV_to_g * 1e-3

# Boltzmann constant in CGS 
k_cgs = k_evKm1 * eV_to_erg
# Planck constant in CGS 
h_cgs = h_eVs * eV_to_erg
# Electron mass revisited 
me_cgs = me_MeV * MeV_to_g
