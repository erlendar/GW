import numpy as np
import matplotlib.pyplot as plt
from genfuncs import *
from fourierpolexp import c_chi, nu_func, Pm, b
from besselint import Ibackwards, Ibackwardsmat
import time

# K-dim: LOOK INTO:
# https://github.com/lesgourg/class_public/tree/class_matter
# https://github.com/JoeMcEwen/FAST-PT

# CHECK OUT ANGPOW:
# https://arxiv.org/abs/1701.03592


#zrange =
zj = np.linspace(0.95,1.05,2)

def Wbar_si(chi):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    return Ws_i(z)*b_GW(z)*H(z)/(c*h)


def Wbar_gj(chi, zj):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    return Wg_j(z, zj)*b_g(z)*H(z)/(c*h)


def fcosm(c_n, nu_n, ch, Wsi, Wgj):
    """
    Returns the f-function ("cosmological part of Csg") for Csg.
    input should be t, n (from 0 to some N set in fourierpolexp.py),
    and cs - an array of Fourier coefficients (indexed from 0 to the same N).

    ch has units Mpc/h
    """
    integrand = Wsi*Wgj*c_n*ch**(1-nu_n)
    integral = integrate(ch, integrand)
    return integral


def Csg(l, N, zj):
    """
    Csg power spectrum
    """
    h = 0.6763
    eps = 1e-4
    t = np.linspace(0.1, 1-eps, 501)
    #t = np.linspace(0.975, 1-eps, 401)

    zrange = np.linspace(0.1,1.4,300)  # z-domain to integrate over
    ch = chi(zrange)*h                 # Corresponding chi-domain to
                                       # integrate over (in units Mpc/h)

    ch_t = np.zeros((len(t), len(ch)))
    np.transpose(ch_t)[:] = t
    ch_t *= ch

    print("Finding Fourier coefficients")
    cs = c_chi(ch, ch_t, N)
    csbar = c_chi(ch_t, ch, N)
    cs[np.where(np.isnan(cs))] = 0
    csbar[np.where(np.isnan(csbar))] = 0
    coeff = lambda n: cs[n] if n >= 0 else np.conj(cs[-n])
    barcoeff = lambda n: csbar[n] if n >= 0 else np.conj(csbar[-n])

    print("Computing window functions")
    Wsibar = Wbar_si(ch_t); Wgjbar = Wbar_gj(ch,zj)
    Wsi = Wbar_si(ch); Wgj = Wbar_gj(ch_t,zj)

    print("Summing terms")
    integrand = 0
    for n in range(-N,N+1):
        nu_n = nu_func(n) + b #DOUBLECHECK!!
        c_n = coeff(n)
        c_nbar = barcoeff(n)

        I = Ibackwards(l, nu_n, t)
        #plt.plot(t,np.real(I),"b.")
        #plt.plot(t,np.imag(I),"r.")
        #plt.show()
        F = fcosm(c_n, nu_n, ch, Wsi, Wgj) \
          + fcosm(c_nbar, nu_n, ch, Wsibar, Wgjbar)
        #plt.plot(t,np.real(F),"r.")
        #plt.plot(t,np.imag(F),"b.")
        #plt.show()
        integrand += I*F
        #plt.plot(t,np.real(integrand),"r.")
        #plt.plot(t,np.imag(integrand),"b.")
        #plt.show()

    #plt.plot(t,np.imag(integrand))
    #plt.plot(t,np.real(integrand),"r.")
    #plt.show()
    C = integrate(t,integrand)
    #print(np.imag(C))
    return 2/(4*np.pi**2)*np.real(C)

#zj = np.linspace(0.95,1.05,2)
#C = Csg(20,10,zj)
#print(C)

"""
delta_z = 0.1
nzg = 20 # Number of data points
zg = np.linspace(0.1,1.3,nzg) # Data points for final plot
C_sg = np.copy(zg)
for i in range(nzg):
    print("\nComputing datapoint {} out of {}".format(i+1,nzg))
    zg_elem = zg[i]
    zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
    C_sg[i] = Csg(2,10,zj)

plt.plot(zg,C_sg,".")
plt.title("$C^{sg}$ w/ FFTlog")
plt.xlabel("$z_g$")
plt.ylabel("$C^{sg}$")
plt.legend(["$C^{sg}(l=2)$"])
plt.savefig("fftplot13")
plt.show()
"""









def create_F(t, ch, Wsi, Wgj, nus, cs):
    F = np.zeros((len(t),len(ch), len(nus)), dtype=complex)
    Ffac = np.copy(F)
    np.transpose(Ffac, (2,0,1))[:] = Wsi*Wgj

    np.transpose(F, (0,2,1))[:] = ch
    F **= (1 - nus)
    F *= Ffac*np.transpose(cs, (1, 2, 0))
    Fintegral = integrate(ch, F, ax=1) # integrating ch-axis
    return Fintegral

def Csg2(t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, cbars, Is):
    """
    Csg power spectrum
    """

    F1 = create_F(t, ch, Wsi, Wgj, nus, cs)
    F2 = create_F(t, ch, Wsibar, Wgjbar, nus, cbars)
    F = F1 + F2

    integrand = np.sum(F*Is, axis=1)   # summing along nus-axis
    C = integrate(t, integrand)
    return 2/(4*np.pi**2)*np.real(C)


def make_oguriplot(l=2, N=10):
    h = 0.6763
    eps = 1e-4
    t = np.linspace(0.1, 1-eps, 301)
    #t = np.linspace(0.975, 1-eps, 401)

    zrange = np.linspace(0.1,1.4,200)  # z-domain to integrate over
    ch = chi(zrange)*h                 # Corresponding chi-domain to
                                       # integrate over (in units Mpc/h)

    ch_t = np.zeros((len(t), len(ch)))
    np.transpose(ch_t)[:] = t
    ch_t *= ch

    print("Finding Fourier coefficients")
    coeffs = c_chi(ch, ch_t, N)
    coeffsbar = c_chi(ch_t, ch, N)
    coeffs[np.where(np.isnan(coeffs))] = 0
    coeffsbar[np.where(np.isnan(coeffsbar))] = 0

    """
    def coeff(n):
        c1 = coeffs[n]
        c2 = np.conj(coeffs[-n])
        return np.where(n >= 0, c1, c2)

    def barcoeff(n):
        c1 = coeffsbar[n]
        c2 = np.conj(coeffsbar[-n])
        return np.where(n >= 0, c1, c2)
    """
    def coeff(n):
        nn = np.copy(coeffs[n])
        np.transpose(nn)[:] = n
        c1 = coeffs[n]
        c2 = np.conj(coeffs[-n])
        return np.where(nn >= 0, c1, c2) # The three arguments must be able to be broadcasted together

    def barcoeff(n):
        nn = np.copy(coeffsbar[n])
        np.transpose(nn)[:] = n
        c1 = coeffsbar[n]
        c2 = np.conj(coeffsbar[-n])
        return np.where(nn >= 0, c1, c2) # The three arguments must be able to be broadcasted together

    print("Computing window functions")
    Wsibar = Wbar_si(ch_t)
    Wsi = Wbar_si(ch)

    N_arr = np.array([i for i in range(-N, N+1)])
    nus = nu_func(N_arr) + b
    cs = coeff(N_arr)
    cbars = barcoeff(N_arr)

    print("Computing Bessel-integrals I:")
    Is = Ibackwardsmat(l, nus, t)

    delta_z = 0.1
    nzg = 20 # Number of data points
    zg = np.linspace(0.1,1.3,nzg) # Data points for final plot
    C_sg = np.copy(zg)
    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wgj = Wbar_gj(ch_t,zj)
        Wgjbar = Wbar_gj(ch,zj)
        C_sg[i] = Csg2(t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, cbars, Is)

    plt.plot(zg,C_sg,".")
    plt.title("$C^{sg}$ w/ FFTlog")
    plt.xlabel("$z_g$")
    plt.ylabel("$C^{sg}$")
    plt.legend(["$Csg(l={})$".format(l)])
    plt.savefig("fftplot13")
    plt.show()

make_oguriplot()








#
