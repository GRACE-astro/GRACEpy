import numpy as np 

def cks_to_bl(xyz, a):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    rad = np.sqrt(x**2+y**2+z**2)
    r = np.sqrt(rad**2-a**2 + np.sqrt(rad**2-a**2)**2 + 4 * a**2*z**2)/np.sqrt(2)
    r[r<1e-6] = 0.5 * (1e-6 + r[r<1e-6]**2/1e-6 )
    r[np.isnan(r)] = 1 
    theta = np.where( (np.abs(z)/r < 1.), np.arccos(z/r), np.arccos(np.copysign(1.0,z)) )
    phi = np.atan2(r*y-a*x, a*y+r*x) - a*r/(r**2-a**2)
    return r,theta,phi

def get_alpha(xyz,a):
    z = xyz[:,2]
    r,theta,phi = cks_to_bl(xyz,a)
    H = r**3 / (r**4+a**2*z**2)
    return 1/np.sqrt(1+2*H)

def get_beta(xyz,a):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    r,theta,phi = cks_to_bl(xyz,a)
    l0 = (r*x+a*y)/(r**2+a**2)
    l1 = (r*y-a*x)/(r**2+a**2)
    l2 = z/r 
    H = r**3 / (r**4+a**2*z**2)
    betax=2*H/(1+2*H)*l0 
    betay=2*H/(1+2*H)*l1
    betaz=2*H/(1+2*H)*l2 
    return betax,betay,betaz

def get_gamma(xyz,a):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    r,theta,phi = cks_to_bl(xyz,a)
    l0 = (r*x+a*y)/(r**2+a**2)
    l1 = (r*y-a*x)/(r**2+a**2)
    l2 = z/r 
    H = r**3 / (r**4+a**2*z**2)
    gxx = 2 * H * l0**2 + 1 
    gxy = 2 * H * l0 * l1 
    gxz = 2 * H * l0 * l2 
    gyy = 2 * H * l1**2 + 1
    gyz = 2 * H * l1*l2 
    gzz = 2 * H * l2**2 + 1
    return gxx,gxy,gxz,gyy,gyz,gzz

def get_sqrtg(xyz,a):
    gxx,gxy,gxz,gyy,gyz,gzz = get_gamma(xyz,a)
    return np.sqrt(
        gxx*gyy*gzz - gxx*gyz**2 - gxy**2*gzz + 2 * gxy*gxz*gyz - gxz**2*gyy
    )

def get_poloidal_toroidal(vec, xyz, a):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    r,theta,phi = cks_to_bl(xyz,a)

    rad2 = x**2+y**2+z**2

    drdx = 
