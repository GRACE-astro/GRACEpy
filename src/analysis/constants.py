#src/analysis/constants.py
## Some useful constants
import numpy as np 


# Physical constants in SI
# Post-2019 SI redefinition: c, h, e, k_B, N_A are EXACT by definition.
# Other values are CODATA 2018 (latest published).
c_si       = 299792458.0          # m/s              (exact, SI definition)
h_si       = 6.62607015e-34       # J s = m^2 kg / s (exact, post-2019)
e_si       = 1.602176634e-19      # Coulomb          (exact, post-2019)
Kb_si      = 1.380649e-23         # Joule/K          (exact, post-2019)
NA_si      = 6.02214076e23        # 1/mol            (exact, post-2019)
G_si       = 6.67430e-11          # m^3/(kg s^2)     (CODATA 2018)
mu0_si     = 1.25663706212e-6     # Newton/Ampere^2  (CODATA 2018; was 4*pi*1e-7 pre-2019)
fm_si      = 1e-15                # m                (exact, definition)
Mparsec_si = 3.08567758e22        # m
parsec_si  = Mparsec_si * 1e-6    # m

# Solar mass derived from IAU 2015 nominal GMsun (exact) and our G,
# so that G_si * Msun_si == GMsun_IAU exactly. This avoids the 1e-5
# inconsistency between an independently chosen G and Msun.
GMsun_IAU  = 1.3271244e20         # m^3/s^2 (IAU 2015 nominal, exact)
Msun_si    = GMsun_IAU / G_si     # kg (derived)

# CGS — derived from SI by 1e3 / 1e2 scaling.
# Previously G_cgs was 6.67430e-8 (CODATA 2018) while G_si was 6.6738e-11
# (CODATA 2010) — inconsistent. Now both come from a single G_si.
c_cgs     = c_si * 1e2          # cm/s
G_cgs     = G_si * 1e3          # cm^3/(g s^2)
Msun_cgs  = Msun_si * 1e3       # g  (was independently set to 1.988475e33, inconsistent)
h_cgs     = h_si * 1e7          # cm^2 g / s
e_cgs     = 4.80320425e-10      # statC (Gaussian CGS — conversion factor differs)
sigma_cgs = 5.670374419e-5      # erg cm^-2 s^-1 K^-4 (Stefan-Boltzmann, CODATA 2018)
rad_cgs   = 4.0 * sigma_cgs / c_cgs  # erg / cm^3 / K^4 (radiation density constant a = 4σ/c)


# Weird units
# NOTE: the historic name "h_eVs" with value 6.5821...e-16 is actually
# hbar (reduced Planck constant) in eV·s, NOT h. Kept under both names
# for backwards compatibility; prefer hbar_eVs in new code.
hbar_eVs = h_si / (2.0 * np.pi) / e_si    # 6.582119569e-16 eV·s
h_eVs    = hbar_eVs                        # legacy alias (misnamed historically)
# Boltzmann constant in eV/K
k_evKm1  = Kb_si / e_si                    # 8.617333262e-5 eV/K


# Particle masses (CODATA 2018)
me_MeV = 0.51099895000  # electron
mp_MeV = 938.27208816   # proton  (was 938.27208943, slightly off CODATA 2018)
mn_MeV = 939.56542052   # neutron (was 939.56542194, slightly off CODATA 2018)
mu_MeV = 931.49410242   # atomic mass unit (CODATA 2018) — FUKA / LORENE / Margherita convention

# Fine structure constant (CODATA 2018)
alpha_fine = 7.2973525693e-3   # dimensionless (was 1./137, ~3e-4 too coarse)

# Conversions — derived from EXACT e_si (post-2019 SI) so eV_to_J etc.
# automatically match the SI definition. Previously eV_to_J was hardcoded
# to the pre-2019 CODATA value 1.6021766208e-19, inconsistent with e_si.
erg_to_J     = 1e-7                  # exact
eV_to_J      = e_si                  # exact, post-2019
MeV_to_J     = eV_to_J * 1e6         # exact
eV_to_kg     = eV_to_J / c_si**2     # kg
MeV_to_kg    = MeV_to_J / c_si**2    # kg
eV_to_erg    = eV_to_J / erg_to_J    # erg
MeV_to_erg   = MeV_to_J / erg_to_J   # erg
eV_to_g      = eV_to_kg * 1e3        # g
MeV_to_g     = MeV_to_kg * 1e3       # g

# Particle masses
# BUGFIX: mp_si/mp_cgs/mn_si/mn_cgs were all silently set to the ELECTRON
# mass times the unit conversion (copy-paste of me_MeV in every line).
# Now correctly use mp_MeV / mn_MeV for proton / neutron.
me_si  = me_MeV * MeV_to_kg  # kg
me_cgs = me_MeV * MeV_to_g   # g
mp_si  = mp_MeV * MeV_to_kg  # kg
mp_cgs = mp_MeV * MeV_to_g   # g
mn_si  = mn_MeV * MeV_to_kg  # kg
mn_cgs = mn_MeV * MeV_to_g   # g
mu_si  = mu_MeV * MeV_to_kg  # kg
mu_cgs = mu_MeV * MeV_to_g   # g

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