import numpy as np 
from analysis.constants import * 


class unit_system:

    def __init__(self, mass, length, time, Bfield):
        self.length = float(length)
        self.time   = float(time)
        self.mass   = float(mass)
        self.Bfield = float(Bfield)

        self.velocity = self.length / self.time 
        self.acceleration = self.velocity / self.time 
        self.force = self.acceleration * self.mass 
        self.surface = self.length**2
        self.volume = self.length**3 

        self.pressure = self.force / self.surface
        self.dens   = self.mass / self.volume
        self.energy = self.force * self.length 
        self.edens  = self.energy / self.volume 


    def __truediv__(self, other):
        return unit_system(self.mass/other.mass, self.length/other.length, self.time/other.time, self.Bfield/other.Bfield)
    
# SI, the base unit system
SI_UNIT_SYSTEM = unit_system(1,1,1,1)
# CGS 
CGS_UNIT_SYSTEM = unit_system(1e-3,1e-2,1,1e-4)
# c = G = Msun = 1 
GEOM_UNIT_SYSTEM = unit_system(Msun_si, G_si * Msun_si / c_si**2, G_si * Msun_si / c_si**3, c_si**4 / Msun_si / G_si**(1.5) * mu0_si**(0.5) )
# Compose units 
COMPOSE_UNIT_SYSTEM = unit_system(MeV_to_kg, fm_si, fm_si/c_si, np.nan)



