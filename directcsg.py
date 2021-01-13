import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy.integrate as inte
import scipy.misc as ms
import time
from class5 import CLASS
from MatterSpectrum import P_m_equaltime
from MatterSpectrum import intpol

"""
Problem:
When trying to create arrays of dimension (500,500,500,500),
terminal shuts down, saying "Killed: 9"
"""
# IDEA: CALCULATE z-prime integral in (20) by creating a standard
# zp = np.linspace(0.1,1.4,nzp) and then inserting a z-dependent
# Heaviside function to limit the integral at z.
# Then the possible z-matrix input of Pm won't be necessary
# and may be removed.

# PLOTS FOR Csg (maximum of order 1e-6) SEEMS TO BE ALMOST A FACTOR
# OF 10 OFF (TOO SMALL) COMPARED TO OGURI (maximum of order 1e-5).
#
# EXPLANATIONS?


# CHECK CONVERGENCE BY STARTING OFF
# WITH LOW n's, THEN INCREASE UNTIL
# THE RESULT STAYS THE SAME



# This program computes the C_sg (+ the other) power spectrum(s)
# by evaluating the bessel integrals directly

# PLAN: (we will let z, zt and zp denote the z, z tilda and z prime
# variables from Powspec-note, respectively)
# Make a 3D grid of k, z and zt values
# Evaluate all functions on this 3D grid
# Integrate over the dimensions in the grid
# Pay special attention to the matter power spectrum P_m

l = 100

z_i = np.linspace(0.9,1.1,100)  # GW bin
#z_j = np.linspace(0.9,1.0,100)  # Galaxy bin

z_imin = z_i[0]; z_imax = z_i[-1]
#z_jmin = z_j[0]; z_jmax = z_j[-1]

nz = int(2e2)
nzt = int(2e2)
nk = int(1e2)
nzp = int(1e2)

wobble = 0.3

k = np.geomspace(1.4e-5, 1.6, nk)
# Area where Pm(k) gives the main contribution
# Better to use geometric spacing as the range
# is over multiple orders of magnitude

z = np.linspace(z_imin - wobble, z_imax + wobble, nz)
# z-integral contains a GW-bin window function, so integrating
# far outside the range of this bin is pointless

zt = np.linspace(0.05,1.4,nzt)
# zt-integral contains a galaxy-bin window function, so integrating
# far outside the range of this bin is pointless

zp = np.linspace(0.05, z[-1], nzp)
# Should always be accompanied by a Heaviside function
# that limits the zp-integral to maximum z


def integrate(x,y,ax=0):
    return inte.simps(y,x,axis=ax)

def j(x,l=2):
    return spec.spherical_jn(l,x)

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
    omega = omega_m*(1+z)**3
    return omega**0.6 + omega_de/70*(1+omega/2) # Dodelson approx

def chi(z):
    c = 299792458/1000
    f = lambda zz: c/H(zz)
    zs = np.linspace(0,z,200)
    ch = integrate(zs,f(zs))
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

def Wk(z, zp, k):
    """
    For 1D-arrays z, zp and k
    the returned Wk should be
    of shape (k, z, zp)
    """
    c = 299792458/1000 # km/s
    A = np.zeros((len(k), len(z), len(zp)))
    chiz = np.copy(A); np.transpose(chiz, (0,2,1))[:] = chi(z)
    chifraction = (chiz - chi(zp))/(chiz*chi(zp))
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*mu(k,zp)*(gamma(k,zp) + 1)
    W2 = -3/4*A*chifraction
    Wtransp = np.transpose(W2)/k**2
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
    chifraction = (chiz - chi(zp))/(chiz*chi(zp))
    A[:] = omega_matter(zp)*H(zp)/(1 + zp)**2*mu(k,zp)*(gamma(k,zp) + 1)
    W2 = -3/4*A*chifraction
    W = W2/k**2
    W /= c # Unit correction for Wk to be unitless
    return W

def Wv(z, k):
    """
    For 1D-arrays z and k
    the returned Wk should be
    of shape (k, z)

    Units Mpc
    """
    c = 299792458/1000 # km/s
    A = np.zeros((len(k), len(z)))
    A[:] = (1 - c/(Hcal(z)*chi(z)) + alpha_M(z)/2)*f(z)*H(z)/(1+z)
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
    c = 299792458/1000 # km/s
    A = np.zeros((len(k)))
    A[:] = (1 - c/(Hcal(z)*chi(z)) + alpha_M(z)/2)*f(z)*H(z)/(1+z)
    W = A*1/k**2*1/c
    return W


def b_g(z):
    I = CLASS(z_i)
    return I.b_g(z)

def b_GW(z):
    I = CLASS(z_i)
    return I.b_GW(z)



def P_m(k, z, z_prime):
    """
    Returns the unequal time matter power spectrum
    by using the geometric approximation
    P(k, z, z')^2 = P(k, z) * P(k, z')

    The function only works for the following domain:
    1.1e-5 <      k     < 1.6   [h/Mpc]
    0.01   < z, z_prime < 1.5

    This function assumes that k, z and z_prime are 1D-arrays.
    Let the lengths of k, z, z_prime be denoted by
    nk, nz, nz_prime respectively. Then the shape
    of the returned power spectrum is (nk, nz, nz_prime).
    This will be the only relevant
    case for our computations.
    """
    try:
        nk = len(k)
        nz = len(z)
        nz_prime = len(z_prime)
        P = np.zeros((nk, nz, nz_prime))
        for i in range(nz):
            P[:, i, :] = np.sqrt(P_m_equaltime(k, z_prime))
        Ptransp = np.transpose(P, (2,0,1))*np.sqrt(P_m_equaltime(k, z))
        P = np.transpose(Ptransp, (1,2,0))
    except TypeError: # k, z or zp numbers instead of arrays
        P = np.sqrt(P_m_equaltime(k, z)*P_m_equaltime(k, z_prime))
    return P



"""
def Csgfunc(l, zj, Pinp=None):
    if Pinp is None:
        P = P_m(k, z, zt)
    else:
        P = Pinp

    z_j = zj

    def inner_integrand(l=2):
        A = np.ones((len(k),len(z),len(zt)))
        x1 = np.copy(A)
        x2 = np.copy(A)

        np.transpose(x1,(0,2,1))[:] = chi(z)
        kz = np.transpose(np.transpose(x1)*k)

        x2[:] = chi(zt)
        kzt = np.transpose(np.transpose(x2)*k)

        A[:] = Wg_j(zt, z_j)*b_g(zt)

        return A*P*j(kz,l)*j(kzt,l)

    def middle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),2)
        return Ws_i(z)*b_GW(z)*Integral

    def outer_integrand(l=2):
        Integral = integrate(z,middle_integrand(l),1)
        return Integral*k**2

    def Csg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi

    return Csg(l)




def Ctgfunc(l, zj, Pinp=None):
    if Pinp is None:
        P = P_m(k, zp, zt)
    else:
        P = Pinp

    z_j = zj

    def inner_integrand(l=2):
        A = np.ones((len(k),len(z),len(zp),len(zt)))
        PP = np.copy(A)
        x1 = np.copy(A)
        x2 = np.copy(A)

        np.transpose(x1,(0,1,3,2))[:] = chi(zp)
        kzp = np.transpose(np.transpose(x1)*k)

        x2[:] = chi(zt)
        kzt = np.transpose(np.transpose(x2)*k)

        np.transpose(PP, (1,0,2,3))[:] = P
        A[:] = Wg_j(zt, z_j)*b_g(zt)

        return A*PP*j(kzp,l)*j(kzt,l)

    def innermiddle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),3)
        H = np.ones((len(k), len(z), len(zp)))
        H[:] = zp
        difftransp = np.transpose(H, (0, 2, 1)) - z
        diff = np.transpose(difftransp, (0,2,1))
        Heaviside = np.heaviside(diff, 0)
        return Wk(z, zp, k)*Heaviside*Integral

    def outermiddle_integrand(l=2):
        Integral = integrate(zp,innermiddle_integrand(l),2)
        return Wt_i(z)*Integral

    def outer_integrand(l=2):
        Integral = integrate(z,outermiddle_integrand(l),1)
        return Integral*k**2

    def Ctg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi

    return Ctg(l)




def Cvgfunc(l, zj, Pinp=None):
    if Pinp is None:
        P = P_m(k, z, zt)
    else:
        P = Pinp

    z_j = zj

    def inner_integrand(l=2):
        A = np.ones((len(k),len(z),len(zt)))
        x1 = np.copy(A)
        x2 = np.copy(A)

        np.transpose(x1,(0,2,1))[:] = chi(z)
        kz = np.transpose(np.transpose(x1)*k)

        x2[:] = chi(zt)
        kzt = np.transpose(np.transpose(x2)*k)

        A[:] = Wg_j(zt, z_j)*b_g(zt)

        return A*P*dj(kz,l)*j(kzt,l)

    def middle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),2)
        return Wv(z, k)*Wt_i(z)*Integral

    def outer_integrand(l=2):
        Integral = integrate(z,middle_integrand(l),1)
        return Integral*k**3
        # The third factor of k comes from the chain rule
        # as we're dealing with the derivative of the Bessel j.

    def Cvg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi

    return Cvg(l)
"""

def Csgfunc(l, Csgfacs, Wg):
    innerfacs = Csgfacs[0]
    middlefacs = Csgfacs[1]
    outerfacs = Csgfacs[2]

    def inner_integrand(l=2):
        return innerfacs*Wg

    def middle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),2)
        return middlefacs*Integral

    def outer_integrand(l=2):
        Integral = integrate(z,middle_integrand(l),1)
        return outerfacs*Integral

    def Csg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi

    return Csg(l)



def Ctgfunc(l, Ctgfacs, Wg):
    innerfacs = Ctgfacs[0]
    innermiddlefacs = Ctgfacs[1]
    outermiddlefacs = Ctgfacs[2]
    outerfacs = Ctgfacs[3]

    def inner_integrand(l=2):
        return innerfacs*Wg

    def innermiddle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),3)
        return innermiddlefacs*Integral

    def outermiddle_integrand(l=2):
        Integral = integrate(zp,innermiddle_integrand(l),2)
        return outermiddlefacs*Integral

    def outer_integrand(l=2):
        Integral = integrate(z,outermiddle_integrand(l),1)
        return outerfacs*Integral

    def Ctg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi

    return Ctg(l)



def Cvgfunc(l, Cvgfacs, Wg):
    innerfacs = Cvgfacs[0]
    middlefacs = Cvgfacs[1]
    outerfacs = Cvgfacs[2]

    def inner_integrand(l=2):
        return innerfacs*Wg

    def middle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),2)
        return middlefacs*Integral

    def outer_integrand(l=2):
        Integral = integrate(z,middle_integrand(l),1)
        return outerfacs*Integral

    def Cvg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi

    return Cvg(l)

def get_facs(P, P2):
    Ws = Ws_i(z); Wt = Wt_i(z)
    Wkappa = Wk(z, zp, k)
    Wvv = Wv(z, k)
    bGW = b_GW(z); bg = b_g(zt)
    chiz = chi(z); chizp = chi(zp)
    chizt = chi(zt)

    A = np.ones((nk, nz, nzt))
    B = np.ones((nk, nz, nzp, nzt))
    x1 = np.copy(A); x2 = np.copy(A)
    y1 = np.copy(B); y2 = np.copy(B)
    PP = np.copy(B)

    np.transpose(x1,(0,2,1))[:] = chiz
    kz = np.transpose(np.transpose(x1)*k)
    x2[:] = chizt
    kzt = np.transpose(np.transpose(x2)*k)
    A[:] = bg

    np.transpose(y1,(0,1,3,2))[:] = chizp
    kzp = np.transpose(np.transpose(y1)*k)
    y2[:] = chizt
    kzt2 = np.transpose(np.transpose(y2)*k)
    np.transpose(PP, (1,0,2,3))[:] = P2
    B[:] = bg

    innerfac_Csg = A*P*j(kz,l)*j(kzt,l)
    innerfac_Ctg = B*PP*j(kzp,l)*j(kzt2,l)
    innerfac_Cvg = A*P*dj(kz,l)*j(kzt,l)

    H = np.ones((nk, nz, nzp))
    H[:] = zp
    difftransp = z - np.transpose(H, (0, 2, 1))
    diff = np.transpose(difftransp, (0,2,1))
    Heaviside = np.heaviside(diff, 0)

    middlefac_Csg = Ws*bGW
    innermiddlefac_Ctg = Wkappa*Heaviside
    outermiddlefac_Ctg = Wt
    middlefac_Cvg = Wvv*Wt

    outerfac_Cstg = k**2
    outerfac_Cvg = k**3

    Csgfacs = [innerfac_Csg, middlefac_Csg, outerfac_Cstg]
    Ctgfacs = [innerfac_Ctg, innermiddlefac_Ctg, \
               outermiddlefac_Ctg, outerfac_Cstg]
    Cvgfacs = [innerfac_Cvg, middlefac_Cvg, outerfac_Cvg]
    return Csgfacs, Ctgfacs, Cvgfacs

def Csgvg_get_facs(P):
    Ws = Ws_i(z); Wt = Wt_i(z)
    Wvv = Wv(z, k)
    bGW = b_GW(z); bg = b_g(zt)
    chiz = chi(z); chizp = chi(zp)
    chizt = chi(zt)

    A = np.ones((nk, nz, nzt))
    x1 = np.copy(A); x2 = np.copy(A)

    np.transpose(x1,(0,2,1))[:] = chiz
    kz = np.transpose(np.transpose(x1)*k)
    x2[:] = chizt
    kzt = np.transpose(np.transpose(x2)*k)
    A[:] = bg

    innerfac_Csg = A*P*j(kz,l)*j(kzt,l)
    innerfac_Cvg = A*P*dj(kz,l)*j(kzt,l)

    middlefac_Csg = Ws*bGW
    middlefac_Cvg = Wvv*Wt

    outerfac_Cstg = k**2
    outerfac_Cvg = k**3

    Csgfacs = [innerfac_Csg, middlefac_Csg, outerfac_Cstg]
    Cvgfacs = [innerfac_Cvg, middlefac_Cvg, outerfac_Cvg]
    return Csgfacs, Cvgfacs


def LimberCsg(l, Wg, P=None):
    c = 299792458
    h = 0.6763
    karg = (l+0.5)/chi(zt)
    if P is None:
        Pfunc = intpol()
        P = Pfunc(karg,zt)
    integrand = Ws_i(zt)*Wg*H(zt)/chi(zt)**2*b_GW(zt)*b_g(zt)*P
    integrand *= 1e3/c/h**3 # unitless
    Integral = integrate(zt,integrand)
    return Integral

def LimberCtg(l, Wg, P=None):
    c = 299792458
    h = 0.6763
    karg = (l+0.5)/chi(zt)
    if P is None:
        Pfunc = intpol()
        P = Pfunc(karg,zt)
    A = np.zeros((nz,nzt))
    A[:] = zt
    diff = np.transpose(z - np.transpose(A))
    Heaviside = np.heaviside(diff, 0)

    inner_integrand = Wg*Wk2(z,zt,karg)*H(zt)/chi(zt)**2*b_g(zt)*P*Heaviside
    integral1 = integrate(zt,inner_integrand,1)
    outer_integrand = Wt_i(z)*integral1
    integral2 = integrate(z,outer_integrand)
    return integral2*1e3/c/h**3

def LimberCvg(l, zj, P=None):
    c = 299792458/1000 # km/s
    h = 0.6763
    karg = (l+0.5)/chi(zt)
    if P is None:
        Pfunc = intpol()
        P = Pfunc(karg,zt)
    todiffanal = lambda zz: Wg_j(zz,zj)*b_g(zz)*H(zz)/chi(zz)**2
    deriv = ms.derivative(todiffanal, zt)
    integrand = H(zt)*Wt_i(zt)*Wv2(zt, karg)*P*deriv
    integral = integrate(zt,integrand)/c**2*1/h**3 # Unitless
    return integral

def PlotLimbers():
    delta_z = 0.1
    nzg = 20 # Number of data points
    zg = np.linspace(0.1,1.3,nzg) # Data points for final plot
    C_sgLimber = np.copy(zg)
    C_tgLimber = np.copy(zg)
    C_vgLimber = np.copy(zg)

    karg = (l+0.5)/chi(zt)
    Pfunc = intpol()
    P = Pfunc(karg,zt)

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wg = Wg_j(zt, zj)
        C_sgLimber[i] = LimberCsg(l, Wg, P)
        C_tgLimber[i] = LimberCtg(l, Wg, P)
        C_vgLimber[i] = LimberCvg(l, zj, P)
    plt.plot(zg, C_sgLimber,"r.-")
    plt.plot(zg, C_tgLimber,"b.-")
    plt.plot(zg, C_vgLimber,"g.-")
    plt.legend(["C_sgLimber(l=100)", "C_tgLimber(l=100)", "C_vgLimber(l=100)"])
    plt.yscale("log")
    #plt.axis([0.1,1.3, 1e-8,1e-4])
    plt.axis([0.01,1.4, 1e-12,1e-3])
    plt.savefig("directplots/Limbers/CsgCtgCvgLimber")
    plt.show()
    return None
PlotLimbers()

def Runall(reset_runindex=False):
    if reset_runindex:
        np.save("runindex.npy", 1)
    runindex = np.load("runindex.npy")
    np.save("runindex.npy", runindex + 1)

    delta_z = 0.1
    nzg = 20 # Number of data points
    zg = np.linspace(0.1,1.3,nzg) # Data points for final plot
    C_sg = np.copy(zg)
    C_tg = np.copy(zg)
    C_vg = np.copy(zg)
    C_sgLimber = np.copy(zg)

    t1 = time.time()
    print("Computing P_m ...")
    P = P_m(k, z, zt)
    P2 = P_m(k, zp, zt)
    print("... P_m found")
    t12 = time.time()

    print("Computing integrands ...")
    Csgfacs, Ctgfacs, Cvgfacs = get_facs(P, P2)
    print("... Integrands found")
    t22 = time.time()

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wg = Wg_j(zt, zj)
        C_sg[i] = Csgfunc(l, Csgfacs, Wg)
        C_tg[i] = Ctgfunc(l, Ctgfacs, Wg)
        C_vg[i] = Cvgfunc(l, Cvgfacs, Wg)
        C_sgLimber[i] = LimberCsg(l, Wg)

    t2 = time.time()
    print("\nTime spent computing P: {:.2f} minutes".format((t12 - t1)/60))
    print("Time spent computing integrands: {:.2f} minutes".format((t22 - t12)/60))
    print("Total time spent: {:.2f} minutes".format((t2-t1)/60))
    plt.plot(zg,C_sg,"r-.")
    plt.plot(zg,C_tg,"b-.")
    plt.plot(zg,C_vg,"k-.")
    plt.plot(zg,C_sgLimber,"g-.")
    plt.legend(["Csg(l={})".format(l), \
                "Ctg(l={})".format(l), \
                "Cvg(l={})".format(l), \
                "CsgLimber"])

    plt.title("nk, nz, nzt, nzp = ({}, {}, {}, {}), t = {:.2f} min"\
              .format(nk,nz,nzt,nzp,(t2-t1)/60),loc="right")
    #plt.yscale("log")
    #plt.axis([0.1,1.3, 1e-8,1e-4])
    plt.savefig("directplots/directCsgCtgCvg{}".format(runindex))
    plt.show()

    plt.plot(zg,C_sg,"r-.")
    plt.savefig("directplots/Csgs/plot{}".format(runindex))
    plt.close()

    plt.plot(zg,C_tg,"r-.")
    plt.savefig("directplots/Ctgs/plot{}".format(runindex))
    plt.close()

    plt.plot(zg,C_vg,"r-.")
    plt.savefig("directplots/Cvgs/plot{}".format(runindex))
    plt.close()

#Runall()



def Runsome(reset_runindex=False):
    if reset_runindex:
        np.save("runindex.npy", 1)
    runindex = np.load("runindex.npy")
    np.save("runindex.npy", runindex + 1)

    delta_z = 0.1
    nzg = 20 # Number of data points
    zg = np.linspace(0.1,1.3,nzg) # Data points for final plot
    C_sg = np.copy(zg)
    C_vg = np.copy(zg)
    C_sgLimber = np.copy(zg)

    t1 = time.time()
    print("Computing P_m ...")
    P = P_m(k, z, zt)
    print("... P_m found")
    t12 = time.time()

    print("Computing integrands ...")
    Csgfacs, Cvgfacs = Csgvg_get_facs(P)
    print("... Integrands found")
    t22 = time.time()

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wg = Wg_j(zt, zj)
        C_sg[i] = Csgfunc(l, Csgfacs, Wg)
        C_vg[i] = Cvgfunc(l, Cvgfacs, Wg)
        C_sgLimber[i] = LimberCsg(l, Wg)

    t2 = time.time()
    print("\nTime spent computing P: {:.2f} minutes".format((t12 - t1)/60))
    print("Time spent computing integrands: {:.2f} minutes".format((t22 - t12)/60))
    print("Total time spent: {:.2f} minutes".format((t2-t1)/60))
    plt.plot(zg,C_sg,"r-.")
    plt.plot(zg,C_vg,"k-.")
    plt.plot(zg,C_sgLimber,"g-.")
    plt.legend(["Csg(l={})".format(l), \
                "Cvg(l={})".format(l), \
                "CsgLimber"])

    plt.title("nk, nz, nzt, nzp = ({}, {}, {}, {}), t = {:.2f} min"\
              .format(nk,nz,nzt,nzp,(t2-t1)/60),loc="right")
    #plt.yscale("log")
    #plt.axis([0.1,1.3, 1e-8,1e-4])
    plt.savefig("directplots/directCsgCvg{}".format(runindex))
    plt.show()

    plt.plot(zg,C_sg,"r-.")
    plt.savefig("directplots/Csgs/plot{}".format(runindex))
    plt.close()

    plt.plot(zg,C_vg,"r-.")
    plt.savefig("directplots/Cvgs/plot{}".format(runindex))
    plt.close()

#Runsome()



"""
For Csg(l=100) with a 500x500x500-grid:

Time spent computing P: 6.04 minutes
Time spent computing integrands: 6.42 minutes
Total time spent: 14.08 minutes
"""

#
