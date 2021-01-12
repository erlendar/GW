import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy.integrate as inte
import scipy.misc as ms
import time
from class5 import CLASS
from MatterSpectrum import P_m_equaltime


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

l = 2

z_i = np.linspace(0.9,1.1,100)  # GW bin
#z_j = np.linspace(0.9,1.0,100)  # Galaxy bin

z_imin = z_i[0]; z_imax = z_i[-1]
#z_jmin = z_j[0]; z_jmax = z_j[-1]

nz = int(1e2)
nzt = int(1e2)
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

def H(z):
    return 100*0.6763*np.sqrt(0.308*(1+z)**3 + 0.692)

def chi(z):
    c = 299792458/1000
    f = lambda zz: c/H(zz)
    zs = np.linspace(0,z,200)
    ch = integrate(zs,f(zs))
    return ch

def Ws_i(z):
    I = CLASS(z_i)
    return I.Ws(z)

def Wt_i(z):
    I = CLASS(z_i)
    return I.Wt(z)

def Wg_j(z, z_j):
    J = CLASS(z_j)
    return J.Ws(z)

def Wk(z, zp, k):
    return 1 # Temporarily

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

    This function assumes that k and z_prime are 1D-arrays.
    Let the lengths of k, z, z_prime be denoted by
    nk, nz, nz_prime respectively. Then the shape
    of the returned power spectrum is as follows:

    (nk, nz, nz_prime) if z is a 1D-arrays.

    (nk, nzrow, nzcol, nz_prime) if z is a 2D-array.

    These will be the only relevant
    cases for our computations.
    """
    nk = len(k)
    nz_prime = len(z_prime)
    if len(np.shape(z)) == 1: # z is 1D
        nz = len(z)
        P = np.zeros((nk, nz, nz_prime))
        for i in range(nz):
            P[:, i, :] = np.sqrt(P_m_equaltime(k, z_prime))
        Ptransp = np.transpose(P, (2,0,1))*np.sqrt(P_m_equaltime(k, z))
        P = np.transpose(Ptransp, (1,2,0))
    elif len(np.shape(z)) == 2: # z is 2D
        # In this case P_m_equaltime(k, z) has
        # the shape (nk, nzrow, nzcol)
        nzrow = np.shape(z)[0]
        nzcol = np.shape(z)[1]
        P = np.zeros((nk, nzrow, nzcol, nz_prime))
        for i in range(nz_prime):
            P[:, :, :, i] = np.sqrt(P_m_equaltime(k, z))
        Ptransp = np.transpose(P, (1, 2, 0, 3))\
                * np.sqrt(P_m_equaltime(k, z_prime))
        P = np.transpose(Ptransp, (2, 0, 1, 3))
    else:
        P = np.sqrt(P_m_equaltime(k, z)*P_m_equaltime(k, z_prime))
    return P


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
        kz = np.transpose(np.transpose(x1)*k)

        x2[:] = chi(zt)
        kzt = np.transpose(np.transpose(x2)*k)

        np.transpose(PP, (1,0,2,3))[:] = P
        A[:] = Wg_j(zt, z_j)*b_g(zt)

        return A*PP*j(kz,l)*j(kzt,l)

    def innermiddle_integrand(l=2):
        Integral = integrate(zt,inner_integrand(l),2)
        """ASSUMING THAT Wk(z, zp, k) is of shape (k, z, zp)"""
        return Wk(z, zp, k)*np.heaviside(zp - z,0)*Integral
        """ HOW TO MAKE THE HEAVISIDE DIMS RIGHT? CHECK TOMORROW"""

    def outermiddle_integrand(l=2):
        Integral = integrate(zt,innermiddle_integrand(l),2)
        return Ws_i(z)*b_GW(z)*Integral



def Run(reset_runindex=False):
    if reset_runindex:
        np.save("runindex.npy", 1)
    runindex = np.load("runindex.npy")
    np.save("runindex.npy", runindex + 1)

    delta_z = 0.1
    nzg = 40 # Number of data points
    zg = np.linspace(0.1,1.3,nzg) # Data points for final plot
    C_sg = np.copy(zg)
    t1 = time.time()
    print("Computing P_m ...")
    P = P_m(k, z, zt)
    t12 = time.time()
    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zjj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        C_sg[i] = Csgfunc(l, zjj, P)
    t2 = time.time()
    print("\nTime spent computing P: {:.2f} minutes".format((t12 - t1)/60))
    print("Total time spent: {:.2f} minutes".format((t2-t1)/60))
    plt.plot(zg,C_sg,"r-.")
    plt.title("nk, nz, nzt, nzp = ({}, {}, {}, {}), t = {:.2f} min"\
              .format(nk,nz,nzt,nzp,(t2-t1)/60),loc="right")
    plt.legend(["Csg(l={})".format(l)])
    plt.savefig("directplots/directCsg{}".format(runindex))
    plt.show()
Run()





#
