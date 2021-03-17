import numpy as np
import matplotlib.pyplot as plt
from MatterSpectrum import P_m
from genfuncs import *


#kmin = 1e-4; kmax = 4.345e-1
kmin = 6.45e-3; kmax = 1e0
P = np.log(kmax/kmin) # Period of the function g
b = 1.9

"""
def Pm(k, z, zt):
    return P_m(k, z, zt)
"""


def Pm0(k, z, zt):
    #Returns the dimensionless unequal time matter power spectrum
    P = P_m(k, z, zt)
    k3arr = np.copy(P)
    np.transpose(k3arr)[:] = k**3
    return P*k3arr

def Pm(k, z, zt):
    #Returns the dimensionless unequal time matter power spectrum
    P = Pm0(k, z, zt)
    kbarr = np.copy(P)
    np.transpose(kbarr)[:] = k**(-b)
    return P*kbarr


def g(k, z, zt):
    """
    Function we find the Fourier series of
    g(k, z, zt) = Pm(exp(k), z, zt)
    We make this function periodic with period
    P = log(kmax) - log(kmin) = log(kmax/kmin).
    """
    if type(k) != np.ndarray:
        k = np.array([float(k)])
    kk = np.copy(k)
    while len(np.where(kk>np.log(kmax))[0]) > 0 or len(np.where(kk<np.log(kmin))[0]) > 0:
        kk[np.where(kk>np.log(kmax))] -= P
        kk[np.where(kk<np.log(kmin))] += P
    return Pm(np.exp(kk), z, zt)

def c(z, zt, N=int(1e2)):
    """
    Finds the coefficients of the Fourier series of g
    (corresponding to the polynomial series of Pm!)
    """
    f_sample = 2*N
    t = np.linspace(0, P, f_sample+2, endpoint=False)
    c = np.fft.rfft(g(t, z, zt),axis=0)/t.size
    return c

def c_chi(chi1, chi2, N=int(1e2)):
    """
    Takes chi1, chi2 as input, in units Mpc/h
    """
    h = 0.6763
    z = z_(chi1/h); zt = z_(chi2/h) # z_ reads chi in units Mpc
    return c(z, zt, N)

def nu_func(n):
    return 1j*2*np.pi*n/P

def PmFourier(k, z, zt):
    """
    Computes the polynomial series of Pm (mainly for comparison with Pm
    to validate the coefficients c)
    """
    N = int(1e2)
    series = 0
    cs = c(z, zt, N)
    nu = np.zeros((N+1), dtype=complex)
    for n in range(N+1):
        nu[n] = nu_func(n)
        series += cs[n]*k**nu[n]
        if n > 0:
            series += np.conj(cs[n])*k**(-nu[n])
    return np.real(series) # No imaginary contribution anyways


"""
N = 20
t = np.linspace(0, P, 2*N+2, endpoint=False)
chi = np.linspace(1000,2000,5)
chi2 = np.random.random((9,11))*4000
c = c_chi(chi,chi2,N)
print(np.shape(c))
"""
#k = np.linspace(1e-3,1e-2,2)
#z = np.linspace(0.4,0.9,3); z2 = np.ones((4,5))*0.9
#print(np.shape(Pm(k, z, z2)))

def plot_fg():
    #kmin = 1e-4; kmax = 4.345e-1
    #kmin = 6.45e-3; kmax = 1e0
    k = np.geomspace(kmin,kmax,200)
    k2 = np.linspace(np.log(kmin),np.log(kmax)+P,200)

    z = 0.47; zt = 1.056
    f = Pm(k, z, zt)
    gg = g(k2, z, zt)
    #plt.loglog(k,f,".")
    plt.plot(k2,gg,".")
    plt.show()
#plot_fg()

def compare_fourier():
    k = np.geomspace(kmin,kmax,200)

    z = 0.47; zt = 1.056
    I1 = integrate(k,Pm(k, z, zt))
    I2 = integrate(k,PmFourier(k, z, zt))
    print(I1/I2)

    plt.plot(k,Pm(k, z, zt),"k.")
    plt.plot(k,PmFourier(k, z, zt),"r--")
    #plt.loglog(k,g(np.log(k)),"r--")
    plt.legend(["Original", "Fourier series"])
    plt.show()
#compare_fourier()





"""
# COMPARING WITH FFTLOGCSG.py
# SAME SET UP AS THE COEFFICIENTS COMPUTED THERE

N = 100
eps = 1e-1
t = np.linspace(eps, 1-eps, 99)

k = np.geomspace(1e-4,4.345e-1,100)
zrange = np.linspace(0.1,1.4,101)  # z-domain to integrate over
ch = chi(zrange)               # Corresponding chi-domain to integrate over

ch_t = np.zeros((len(t), len(ch)))
np.transpose(ch_t)[:] = t
ch_t *= ch

cs = c_chi(ch, ch_t, N)

series=np.zeros((len(k), np.shape(cs[0])[-2], np.shape(cs[0])[-1]), dtype=complex)
temp=np.zeros((len(k), np.shape(cs[0])[-2], np.shape(cs[0])[-1]), dtype=complex)
for n in range(N+1):
    print(n)
    nu_n = nu_func(n)
    np.transpose(temp)[:] = k**nu_n
    temp *= cs[n]
    series += temp
    temp *= 0
    if n > 0:
        np.transpose(temp)[:] = k**(-nu_n)
        temp *= np.conj(cs[n])
        series += temp
        temp*=0
series = np.real(series)

z2 = np.zeros((len(t), len(ch)))
np.transpose(z2)[:] = t
z2 *= z_(ch)


plt.loglog(k,Pm(k, zrange[2], z2[3,2]),"b.")
plt.loglog(k,series[:,2,3],"r--")
plt.legend(["Original", "Fourier series"])
plt.show()
"""



#
