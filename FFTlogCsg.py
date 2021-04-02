import numpy as np
import matplotlib.pyplot as plt
from genfuncs import *
from fourierpolexp import c_chi, nu_func, Pm, b
from besselint import Ibackwards, Ibackwardsmat, I
from directcsg import LimberCsg
import time

# CHECK OUT:
# https://arxiv.org/pdf/1809.03528.pdf



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


"""
nt = 200
nz = 1000
t = np.linspace(0.1, 0.99, nt)
z = np.linspace(0.1, 1.4, nz)

z_t = np.zeros((nt, nz))
np.transpose(z_t)[:] = t
z_t *= z

zg = np.linspace(0.5,1.3,20)
n = 10
zj = np.linspace(zg[n]-0.1/2, zg[n]+0.1/2, 2) # Galaxy bin
f = Wg_j(z_t, zj)

for i in range(0, nt, 40):
    integrand = f[i,:]
    nr_nonzero_points = len(np.where(integrand > 0)[0])
    #print(nr_nonzero_points)
    #range = z*t[i]
    print("Range of nonzero-interval for z: [{:.3f}, {:.3f}]".format(zj[0]/t[i], zj[-1]/t[i]))
    plt.plot(z,f[i,:],".--")
    # If number of nonzero points is unequal to the standard
    # Make a new array of the standard amount spread on the nonzero domain
    #print(integrate(z, integrand))
plt.show()


fintegral = integrate(z, f, ax=1)
plt.plot(t, fintegral,".--")
print(integrate(t,fintegral))
plt.show()
"""
"""
print(integrate(z, f[t_ind,:]))
print(integrate(z, f[t_ind+1,:]))
plt.show()
"""
"""
fintegral = integrate(z, f, ax=1)
plt.plot(t, fintegral,".--")
print(integrate(t,fintegral))
plt.show()
"""
"""
from scipy.ndimage import gaussian_filter1d
g = gaussian_filter1d(fintegral, 4)
plt.plot(t,g)
print(integrate(t,g))
plt.show()
"""

def test_coeffs(cs, nus, ch, ch_t):
    h = 0.6763
    kmin = 1e-4; kmax = 0.1
    kk = np.geomspace(kmin, kmax, 300)
    k = np.zeros((len(kk), np.shape(cs)[1], np.shape(cs)[2]))
    np.transpose(k)[:] = kk
    series1 = 0
    for n in range(len(nus)):
        print(n)
        series1 += cs[n]*k**(nus[n]-b)
    print(np.max(np.abs(np.imag(series1))))
    series1 = np.real(series1)
    f = Pm(kk, z_(ch/h), z_(ch_t/h))
    f[np.where(np.isnan(f))] = 0 # See comment on cs coeffs

    print("Maximum difference: {:.5f}".format(np.max(np.abs(f - series1))))
    print("Mean difference: {:.5f}".format(np.mean(np.abs(f - series1))))
    ind1 = 174; ind2 = 103
    plt.plot(kk, f[:, ind1, ind2],".")
    plt.plot(kk, series1[:, ind1, ind2],"--")
    plt.show()
    return None



def plot_some(t,nus,Is,show=False,save=False):
    for i in range(1,len(nus),15):
        plt.plot(t, np.real(Is)[:,i],".--")
        #plt.plot(t, np.imag(Is)[:,i],".")
    if save:
        np.savefig("Iplot")
    if show:
        plt.show()
    return None

def plotCsg(zg, C, l, N, nt, nch, timespent, fftlogindex):
    plt.plot(zg,C,".")
    plt.title("FFTlog, N={}, nt={}, nchi={}, t={:.1f}min".format(N, nt, nch, timespent))
    plt.xlabel("$z_g$")
    plt.ylabel("$C^{sg}$")
    #plt.legend(["$Csg(l={})$".format(l)])
    plt.legend(["Limber", "$Csg(l={})$".format(l)])
    plt.savefig("fftplots/fftplotgen{}".format(fftlogindex))
    plt.yscale("log")
    plt.axis([0.1,1.3, 1e-8,1e-4])
    plt.savefig("fftplots/fftplot{}".format(fftlogindex))
    plt.show()


def create_F(t, ch, Wsi, Wgj, nus, cs):
    F = np.zeros((len(t),len(ch), len(nus)), dtype=complex)
    Ffac = np.copy(F)
    np.transpose(Ffac, (2,0,1))[:] = Wsi*Wgj

    np.transpose(F, (0,2,1))[:] = ch
    F **= (1 - nus)
    F *= Ffac*np.transpose(cs, (1, 2, 0))
    Fintegral = integrate(ch, F, ax=1) # integrating ch-axis
    return Fintegral

def Csg(l, t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, Is, i=None):
    """
    Csg power spectrum
    """

    F1 = create_F(t, ch, Wsi, Wgj, nus, cs)
    F2 = create_F(t, ch, Wsibar, Wgjbar, nus, cs)
    F = F1 + F2
    #plot_some(t,nus,F,show=True)

    #Limber_I = 2*np.pi**2*(l+0.5)**(nus-3) # "evaluated at t=1"
    #Limber_F = F[-1,:]                     # "evaluated at t=1"

    integrand = np.sum(F*Is, axis=1)   # summing along nus-axis
    """
    plt.plot(t,np.real(integrand),".")
    plt.savefig("fftplots/Csg_integrand/ints{}".format(i))
    plt.close()
    """
    """
    # Smooth out integrand by applying a gaussian filter:
    from scipy.ndimage import gaussian_filter1d
    integrand = gaussian_filter1d(np.real(integrand), 4)
    """

    C = integrate(t, integrand)
    #print("Final imaginary part: {}".format(np.imag(C)))
    return 2/(4*np.pi**2)*np.real(C)


def make_oguriplot(reset_runindex=False, l=2, N=100):
    if reset_runindex:
        np.save("fftlogindex.npy", 1)
    fftlogindex = np.load("fftlogindex.npy")
    np.save("fftlogindex.npy", fftlogindex + 1)

    time1 = time.time()
    nt1 = 100
    nt2 = 100
    nt3 = 100
    nt = nt1 + nt2 + nt3
    nch = 200
    h = 0.6763
    eps = 1e-4
    #t = np.linspace(0.001, 1-eps, nt)

    t1 = np.linspace(0.01,0.85,nt1 + 1)
    t2 = np.linspace(0.85,0.95,nt2 + 1)
    t3 = np.geomspace(0.95,1-eps,nt3)
    t = np.concatenate((t1[:-1],t2[:-1],t3)) # No overlapping points

    zrange = np.linspace(0.1,1.4,nch)  # z-domain to integrate over
    ch = chi(zrange)*h                 # Corresponding chi-domain to
                                       # integrate over (in units Mpc/h)
    """
    You should sample from the chi-range!!
    As it is now, we're practically integrating over z as
    this is what we sample from.
    """


    ch_t = np.zeros((nt, nch))
    np.transpose(ch_t)[:] = t
    ch_t *= ch

    print("Finding Fourier coefficients")
    coeffs = c_chi(ch, ch_t, N)
    coeffs[np.where(np.isnan(coeffs))] = 0 # The matter power spectrum has not been defined
                                           # for some values of z (chi), so we'll set these to
                                           # zero for now. This must be fixed,
                                           # although contributions from this domain
                                           # should be negligible for Csg
                                           # (beware of the effect on the other contributions). Check this out.
    """
    ATTENTION COMMENT ABOVE;
    power spectrum should be defined for z-values down to z=0.01 at least,
    and up to z=1.4 (lower limit is the problem, check MatterSpectrum.py)
    """

    def coeff(n):
        nn = np.copy(coeffs[n])
        np.transpose(nn)[:] = n
        c1 = coeffs[n]
        c2 = np.conj(coeffs[-n])
        return np.where(nn >= 0, c1, c2) # The three arguments must be able to be broadcasted together

    print("Computing window functions")
    Wsibar = Wbar_si(ch_t)
    Wsi = Wbar_si(ch)

    N_arr = np.array([i for i in range(-N, N+1)])
    nus = nu_func(N_arr) + b
    cs = coeff(N_arr)
    """
    Recall that "cbars" = cs as we use the geometric approximation/midpoint approximation,
    i.e. c(x,y) is symmetric under x <---> y
    """

    #test_coeffs(cs, nus, ch, ch_t)
    #return None


    print("Computing Bessel-integrals I:")
    Is = np.zeros((len(t), len(nus)), dtype=complex)
    for i in range(len(nus)):
        if i in [n for n in range(0, len(nus), 22)]:
            print("{}%".format(int(i/len(nus)*100)))
        Is[:, i] = I(l, nus[i], t)[:,0]
    print("100%")
    #plot_some(t,nus,Is,show=True)

    """
    zj = np.linspace(0.95,1.05,2)
    Wgj = Wbar_gj(ch_t,zj)
    Wgjbar = Wbar_gj(ch,zj)
    return Csg(l, t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, Is, i)
    """

    delta_z = 0.1
    nzg1 = 6
    nzg2 = 24
    nzg = nzg1 + nzg2 # Number of data points
    zg1 = np.linspace(0.1, 0.7, nzg1 + 1)
    zg2 = np.linspace(0.7, 1.3, nzg2)
    zg = np.concatenate((zg1[:-1], zg2))
    C_sg = np.copy(zg)
    LimberC_sg = np.copy(zg)


    from MatterSpectrum import intpol
    zt = np.linspace(0.6,1.4,200)
    karg = (l+0.5)/chi(zt)
    Pfunc = intpol()
    P = Pfunc(karg,zt)

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wgj = Wbar_gj(ch_t,zj)
        Wgjbar = Wbar_gj(ch,zj)
        C_sg[i] = Csg(l, t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, Is, i)
        LimberC_sg[i] = LimberCsg(l, zj, P, zt)

    time2 = time.time()
    timespent = (time2-time1)/60
    print("Time spent: {:.1f} minutes".format(timespent))
    print(LimberC_sg/C_sg) # This value seems to be centered around ~ 1.45
    plt.plot(zg,LimberC_sg,"--")
    plotCsg(zg, C_sg, l, N, nt, nch, timespent, fftlogindex)


make_oguriplot(l=100)
"""
n = 15
C__sg = np.zeros(n)
ls = np.linspace(0,100,n)
for i in range(n):
    l = int(ls[i])
    ls[i] = l
    print("Working on datapoint {} out of {}".format(i+1,n))
    print("l = {}".format(l))
    C__sg[i] = make_oguriplot(l=l)
plt.plot(ls, C__sg, ".--")
plt.savefig("fftplots/Csg_of_l")
plt.show()
"""








#
