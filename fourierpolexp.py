import numpy as np
import matplotlib.pyplot as plt
from MatterSpectrum import P_m
from hi_MatterSpectrum import P_m_hi
from genfuncs import *


#kmin = 1e-4; kmax = 4.345e-1
kmin = 1e-4; kmax = 1.5
#kmin = 6.45e-3; kmax = 1e0
P = np.log(kmax/kmin) # Period of the function g
b = 1.5

"""
def Pm(k, z, zt):
    return P_m(k, z, zt)
"""

def W(k, k_left=kmin+1e-2, k_right=kmax-2e-1):
    l_fac = (k - kmin)/(k_left - kmin)
    r_fac = (kmax - k)/(kmax - k_right)
    Wleft = l_fac - 1/(2*np.pi)*np.sin(2*np.pi*l_fac)
    Wmid = 1
    Wright = r_fac - 1/(2*np.pi)*np.sin(2*np.pi*r_fac)
    Warr = np.copy(k)
    W = np.where(Warr < k_left, Wleft, Wmid)
    return np.where(Warr > k_right, Wright, W)


def Pm0(k, z, zt, samedim2=True, c_M=0, c_B=0):
    #Returns the dimensionless unequal time matter power spectrum
    """
    if c_M == 0 and c_B == 0:
        P = P_m(k, z, zt, samedim2=samedim2)
    else:
    """
    P = P_m_hi(k, z, zt, samedim2=samedim2, c_M=c_M, c_B=c_B)
    k3arr = np.copy(P)
    np.transpose(k3arr)[:] = k**3
    return P*k3arr

def Pm_no_W(k, z, zt):
    #Returns the dimensionless unequal time matter power spectrum
    #with the k^(-b)-factor.
    P = Pm0(k, z, zt)
    kbarr = np.copy(P)
    np.transpose(kbarr)[:] = k**(-b)
    return P*kbarr

def Pm(k, z, zt, samedim2=True, c_M=0, c_B=0):
    #Returns the dimensionless unequal time matter power spectrum
    #with the k^(-b)-factor and window. This is what we want to express in
    #polynomials of k.
    P = Pm0(k, z, zt, samedim2=samedim2, c_M=c_M, c_B=c_B)
    kbarr = np.copy(P)
    np.transpose(kbarr)[:] = k**(-b)*W(k) # W(k) is window to avoid ringing effects
    return P*kbarr

def g(k, z, zt, samedim2=True, c_M=0, c_B=0):
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
    return Pm(np.exp(kk), z, zt, samedim2=samedim2, c_M=c_M, c_B=c_B)

def c(z, zt, N=int(1e2), samedim2=True, c_M=0, c_B=0):
    """
    Finds the coefficients of the Fourier series of g
    (corresponding to the polynomial series of Pm!)
    """
    f_sample = 2*N
    t = np.linspace(0, P, f_sample+2, endpoint=False)
    c = np.fft.rfft(g(t, z, zt, samedim2=samedim2, c_M=c_M, c_B=c_B),axis=0)/t.size
    return c

def c_chi(chi1, chi2, N=int(1e2), samedim2=True, c_M=0, c_B=0):
    """
    Takes chi1, chi2 as input, in units Mpc/h

    Returns c of shape (N+2, shape(chi1), shape(chi2))
    If dim(chi1) = 1 and dim(chi2) = 2 and
    len(dim(chi1)) is equal to the length of one of chi2's 2nd axis,
    then the returned shape is just (N+2, shape(chi2)).

    Same vice versa
    """
    h = 0.6763
    z = z_(chi1/h); zt = z_(chi2/h) # z_ reads chi in units Mpc
    return c(z, zt, N, samedim2=samedim2, c_M=c_M, c_B=c_B)

def nu_func(n):
    return 1j*2*np.pi*n/P

def PmFourier(k, z, zt, c_M=0):
    """
    Computes the polynomial series of Pm (mainly for comparison with Pm
    to validate the coefficients c)
    """
    N = int(1e2)
    cs = c(z, zt, N, c_M=c_M)
    nu = np.zeros((N+1), dtype=complex)
    series = 0
    for n in range(N+1):
        nu[n] = nu_func(n)
        series += cs[n]*k**nu[n]
        if n > 0:
            series += np.conj(cs[n])*k**(-nu[n])
    return np.real(series) # No imaginary contribution anyways




def plot_coeffs():
    z1 = 0.45
    z2 = 0.93
    N = 100#1500
    n = [i for i in range(-N,N+1)]
    coe = np.zeros((2*N+1), dtype=complex)
    cc = c(z1, z2, N)
    for i in range(N+1):
        coe[N - i] = np.conj(cc[i])
        coe[N + i] = cc[i]
    #plt.plot(n, np.real(coe), ".")
    #plt.plot(n, np.imag(coe), ".")
    plt.plot(n, np.abs(coe), ".")
    plt.yscale("log")
    plt.show()
#plot_coeffs()



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

def plot_P_W():
    z = 0.5; zt = 0.9
    k = np.linspace(kmin, kmax, 800)
    plt.plot(k,Pm(k,z,zt),".")
    plt.plot(k,Pm_no_W(k,z,zt))
    plt.show()
    return None
#plot_P_W()

def plot_fg():
    #kmin = 1e-4; kmax = 4.345e-1
    #kmin = 6.45e-3; kmax = 1e0
    k = np.geomspace(kmin,kmax,800)
    k2 = np.linspace(np.log(kmin),np.log(kmax)+P,800)

    z = 0.47; zt = 1.056
    f = Pm(k, z, zt)
    gg = g(k2, z, zt)
    #plt.loglog(k,f,".")
    plt.plot(k2,gg,".")
    plt.show()
#plot_fg()

def compare_fourier():
    k = np.geomspace(kmin,kmax,400)

    z = 0.47; zt = 1.056
    I1 = integrate(k,Pm(k, z, zt, c_M=0))
    I2 = integrate(k,PmFourier(k, z, zt, c_M=0))
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
