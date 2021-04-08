import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy.integrate as inte
import scipy.misc as ms
import time
from class5 import CLASS
from scipy.interpolate import interp1d


z_i = np.linspace(0.9, 1.1, 2)  # GW bin

def integrate(x,y,ax=-1):
    return inte.simps(y,x,axis=ax)

def j(x,l=2):
    return spec.spherical_jn(l,x)

"""
x = np.linspace(0,100,1000)
f = j(x, l=5)
g = j(x, l=7)
plt.plot(x,f)
plt.plot(x,g)
plt.title("Spherical bessel functions")
plt.legend(["$j_5(x)$", "$j_7(x)$"])
plt.xlabel("x")
plt.savefig("spbes")
plt.show()
"""

def dj(x,l=2):
    """
    Returns the derivative of the Bessel function
    """
    return spec.spherical_jn(l,x,True)

def H(z):
    return 100*0.6763*np.sqrt(0.308*(1+z)**3 + 0.692)

def f(z):
    """
    Returns the dimensionless linear growth rate
    """
    omega_m  = 0.308
    omega_de = 0.692
    #omega = omega_m*(1+z)**3
    #return omega**0.6 + omega_de/70*(1+omega/2) # Dodelson approx


    omega = omega_m*(1+z)**3*H(0)**2/H(z)**2
    omega_de = omega_de*H(0)**2/H(z)**2
    return omega**(4/7) + omega_de/70*(1+omega/2) # Dodelson approx
    #return 5*omega/(2*(omega**(4/7) - omega_de + (1 + omega/2)*(1 + omega_de/70)))
    #return omega**0.55

def chi(z):
    c = 299792458/1000
    f = lambda zz: c/H(zz)
    zs = np.linspace(0,z,200)
    ch = integrate(zs,f(zs),ax=0)
    return ch

def chibar(z):
    """
    For now this just returns chi(z)
    """
    return chi(z)

def Hcal(z):
    return H(z)/(1 + z)

def omega_lambda(z):
    """
    Returns Omega_Lambda as a function
    of the redshift (unitless)
    """
    omega_de = 0.692
    h        = 0.6763
    H0       = 100*h
    omega = omega_de*H0**2/H(z)**2
    return omega

def alpha_M(z, alpha_M0 = 0):
    """
    Returns the alpha_M parameter (unitless)
    """
    alpha = alpha_M0*omega_lambda(z)/omega_lambda(0)
    return alpha

def mu(k, z):
    """
    Returns 1 for now
    """
    return 1

def gamma(k, z):
    """
    Returns 1 for now
    """
    return 1

def omega_matter(z):
    omega_m  = 0.308
    omega_de = 0.692
    omega = omega_m*(1 + z)**3/(omega_de + omega_m*(1 + z)**3)
    return omega

def Ws_i(z):
    I = CLASS(z_i)
    return I.Ws(z)

def Wt_i(z):
    I = CLASS(z_i)
    return I.Wt(z)

def Wg_j(z, z_j):
    J = CLASS(z_j)
    return J.Wg(z)

def Wkappa(z, zp):
    """
    For 1D-arrays z, zp and k
    the returned Wk should be
    of shape (k, z, zp)

    Unitless
    """
    c = 299792458/1000 # km/s
    A = np.zeros([len(z)] + list(np.shape(zp)))
    chiz = np.copy(A); np.transpose(chiz)[:] = chi(z)
    chifraction = (chiz - chi(zp))*chi(zp)/chiz
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*2
    W2 = 3/4*A*chifraction
    Wtransp = np.transpose(W2)
    W = np.transpose(Wtransp)
    W /= c # Unit correction for Wk to be unitless
    return W

def Wk(z, zp, k):
    """
    For 1D-arrays z, zp and k
    the returned Wk should be
    of shape (k, z, zp)

    Unitless
    """
    c = 299792458/1000 # km/s
    A = np.zeros((len(k), len(z), len(zp)))
    chiz = np.copy(A); np.transpose(chiz, (0,2,1))[:] = chi(z)
    chifraction = (chiz - chi(zp))*chi(zp)/chiz
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*mu(k,zp)*(gamma(k,zp) + 1)
    W2 = 3/4*A*chifraction
    Wtransp = np.transpose(W2)#/k**2 # If k is included, multiply by h
    W = np.transpose(Wtransp)
    W /= c # Unit correction for Wk to be unitless
    return W

def Wk2(z, zp, k):
    """
    For arrays zp, k of equal length,
    returns shape (z, zp)
    """
    c = 299792458/1000 # km/s
    A = np.zeros((len(z), len(zp)))
    chiz = np.copy(A); np.transpose(chiz)[:] = chi(z)
    chifraction = (chiz - chi(zp))*chi(zp)/chiz
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*mu(k,zp)*(gamma(k,zp) + 1)
    W2 = 3/4*A*chifraction
    W = W2#/k**2
    W /= c # Unit correction for Wk to be unitless
    return W

def Wv(z, k):
    """
    For 1D-arrays z and k
    the returned Wk should be
    of shape (k, z)

    Units Mpc
    """
    k *= 0.6763 # 1/Mpc
    c = 299792458/1000 # km/s
    A = np.zeros((len(k), len(z)))
    A[:] = -(1 - c/(Hcal(z)*chi(z)) + alpha_M(z)/2)*f(z)*H(z)/(1+z)
    Wtransp = np.transpose(A)*1/k**2*1/c
    W = np.transpose(Wtransp)
    return W

def Wv2(z, k):
    """
    For 1D-arrays z and k of equal
    length, the returned Wk is
    of shape (k)

    Units Mpc
    """
    k *= 0.6763 # 1/Mpc
    c = 299792458/1000 # km/s
    A = np.zeros((len(k)))
    A[:] = -(1 - c/(Hcal(z)*chi(z)) + alpha_M(z)/2)*f(z)*H(z)/(1+z)
    W = A*1/k**2*1/c
    return W


def b_g(z):
    I = CLASS(z_i)
    return I.b_g(z)

def b_GW(z):
    I = CLASS(z_i)
    return I.b_GW(z)


def savechi(zmin=1e-10, zmax=20, N=1000):
    z = np.geomspace(zmin,zmax,N)
    ch = chi(z)
    np.save("zchiarr.npy", z)
    np.save("chiarr.npy", ch)
#savechi()

def z_(chiarg, save=False):
    """
    The inverse of chi(z):
    for any value C, returns the corresponding z s.t. chi(z) = C
    """
    if save:
        savechi()
    z = np.load("zchiarr.npy")
    ch = np.load("chiarr.npy")
    f = interp1d(ch, z)
    return f(chiarg)














#
