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
    if type(z) == np.ndarray and 0 in z:
        ch[np.where(np.isnan(ch))] = 0
        print("Invalid value encountered OK")
        # zeros in z gives rise to nan in ch because of integration.
        # We but the zero back in as chi(0) = 0
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

def alpha_M(z, c_M = 0):
    """
    Returns the alpha_M parameter (unitless)
    """
    alpha = c_M*omega_lambda(z)/omega_lambda(0)
    return alpha

def delta_D(z, c_M = 0):
    """
    Returns the delta_D parameter (unitless)
    """
    n = 200
    zprime = np.linspace(1e-10, z, n)
    integrand = alpha_M(zprime, c_M)/(1 + zprime)
    delta = 0.5*integrate(zprime, integrand, ax=0)
    return delta

def M_star_sq(z, c_M=0):
    """
    Returns M_*^2 in units m_P
    """
    zint = np.linspace(z,20,200) # alpha_M is approx zero beyond z = 20
    integrand = alpha_M(zint, c_M)/(1+zint)
    logM = integrate(zint, integrand, ax=0)
    M = np.exp(logM)
    return M

def G_light(z, c_M=0):
    G = 1/M_star_sq(z, c_M)*(1 + alpha_M(z, c_M)/2)
    return G

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

def Wkappa(z, zp, c_M=0):
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
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*G_light(zp, c_M)
    W = 3/2*A*chifraction
    W /= c # Unit correction for Wk to be unitless
    return W

def Wk(z, zp, k, c_M=0):
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
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*G_light(zp, c_M)
    W2 = 3/2*A*chifraction
    Wtransp = np.transpose(W2)#/k**2 # If k is included, multiply by h
    W = np.transpose(Wtransp)
    W /= c # Unit correction for Wk to be unitless
    return W

def Wk2(z, zp, k, c_M=0):
    """
    For arrays zp, k of equal length,
    returns shape (z, zp)
    """
    c = 299792458/1000 # km/s
    A = np.zeros((len(z), len(zp)))
    chiz = np.copy(A); np.transpose(chiz)[:] = chi(z)
    chifraction = (chiz - chi(zp))*chi(zp)/chiz
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*G_light(zp, c_M)
    W = 3/2*A*chifraction
    W /= c # Unit correction for Wk to be unitless
    return W

def WvFFTlog(z):
    """
    For 1D-arrays z
    the returned Wk should be
    of shape z

    Units 1/Mpc
    """
    c = 299792458/1000 # km/s
    W = -(1 - c/(Hcal(z)*chi(z)) + alpha_M(z)/2)*f(z)*H(z)/(1+z)
    W *= 1/c
    return W


def Wv(z, k):
    """
    For 1D-arrays z and k
    the returned Wk should be
    of shape (k, z)
    k must be of units [h/Mpc]

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
    For use with the Limber approx.

    z, k must be of equal shape.
    k must be of units [h/Mpc]

    Units Mpc
    """
    k *= 0.6763 # 1/Mpc
    c = 299792458/1000 # km/s
    W = -(1 - c/(Hcal(z)*chi(z)) + alpha_M(z)/2)*f(z)*H(z)/(1+z)*1/k**2
    W *= 1/c
    return W

def W_MG(z, c_M=0):
    return Wt_i(z)*delta_D(z, c_M)


def b_g(z):
    I = CLASS(z_i)
    return I.b_g(z)

def b_GW(z):
    I = CLASS(z_i)
    return I.b_GW(z)


def savechi(zmin=0, zmax=10, N=10000):
    z = np.linspace(zmin, zmax, N)
    ch = chi(z)
    #plt.plot(z,ch,".")
    #plt.show()
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

"""
l = 100
print((l-10)/(chi(0.1)*0.6763))
x = np.linspace(0,150,1000)
plt.plot(x, j(x, l))
plt.show()
"""
#
