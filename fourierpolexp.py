import numpy as np
import matplotlib.pyplot as plt
from MatterSpectrum import P_m
from genfuncs import *


kmin = 1e-4; kmax = 4.345e-1
P = np.log(kmax/kmin) # Period of the function g

def Pm(k, z, zt):
    """
    Unequal time matter power spectrum
    """
    return P_m(k, z,zt)

def g(k, z, zt):
    """
    Function we find the Fourier series of
    g(k, z, zt) = Pm(exp(k), z, zt)
    We make this function periodic with period
    P = log(kmax) - log(kmin) = log(kmax/kmin).
    """
    if type(k) != np.ndarray:
        k = np.array([float(k)])
    while len(np.where(k>np.log(kmax))[0]) > 0 or len(np.where(k<np.log(kmin))[0]) > 0:
        k[np.where(k>np.log(kmax))] -= P
        k[np.where(k<np.log(kmin))] += P
    return Pm(np.exp(k), z, zt)

def c(z, zt, N=int(1e2)):
    """
    Finds the coefficients of the Fourier series of g
    (corresponding to the polynomial series of Pm!)
    """
    f_sample = 2*N
    t = np.linspace(0, P, f_sample+2, endpoint=False)
    c = np.fft.rfft(g(t, z, zt))/t.size
    return c

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
        nu[n] = 1j*2*np.pi*n/P
        series += cs[n]*k**nu[n]
        if n > 0:
            series += np.conj(cs[n])*k**(-nu[n])
    return np.real(series) # No imaginary contribution anyways



if __name__ == "__main__":
    k = np.geomspace(kmin,kmax,50)

    z = 0.47; zt = 1.056
    I1 = integrate(k,Pm(k, z, zt))
    I2 = integrate(k,PmFourier(k, z, zt))
    print(I1/I2)

    plt.loglog(k,Pm(k, z, zt),"k.")
    plt.loglog(k,PmFourier(k, z, zt),"r--")
    #plt.loglog(k,g(np.log(k)),"r--")
    plt.legend(["Original", "Fourier series"])
    plt.show()






#
