import numpy as np
import matplotlib.pyplot as plt
from genfuncs import *
from fourierpolexp import c_chi, nu_func, Pm, b
from besselint import Ibackwards, Ibackwardsmat, I
from directcsg import LimberCsg, LimberCtg, LimberCvg, LimberCvgprep
from scipy import interpolate
import scipy.misc as ms
import time

#plt.style.use("ggplot")

# CHECK OUT:
# https://arxiv.org/pdf/1809.03528.pdf



# K-dim: LOOK INTO:
# https://github.com/lesgourg/class_public/tree/class_matter
# https://github.com/JoeMcEwen/FAST-PT

# CHECK OUT ANGPOW:
# https://arxiv.org/abs/1701.03592


#zrange =
#zj = np.linspace(0.95,1.05,2)

def Wbar_si(chi, c_M=0):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    return Ws_i(z, c_M)*b_GW(z)*H(z)/(c*h)

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
def Wbar_gjdiff(ch, zj):
    h = 0.6763

    todiffanal = lambda chh: Wbar_gj(chh, zj)

    eps = 1e-4 # We want to avoid differentiating at the sharp cutoff of Wg_j
    zrange = np.linspace(zj[0]+eps,zj[-1]-eps,1000)
    chrange = chi(zrange)*h
    deriv = ms.derivative(todiffanal, chrange, 1e-5)
    myfunc = interpolate.interp1d(chrange, deriv, bounds_error=False, fill_value=0)
    W = myfunc(ch)
    return W
"""
def Wbar_ti(chi, c_M=0):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    return Wt_i(z, c_M)*H(z)/(c*h)

def Wbar_k(chi, chi_p, c_M=0):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    z_p = z_(chi_p/h)
    return Wkappa(z, z_p, c_M)*H(z_p)/(c*h)

"""
def Wbar_v(chi):
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    return WvFFTlog(z)/h
"""

def Wv1(chi, c_M=0):
    """
    Takes chi as input, with units [Mpc/h].

    Warning: chi must be a 1D-array! (for now)

    Returns output in units (h/Mpc)^2
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi/h) # z_ reads chi in units Mpc
    W = H(z)*Wt_i(z, c_M)*WvFFTlog(z, c_M)/(c*h**2)
    return W

def Wv2(chi_t, zj):
    """
    Takes chi as input, with units [Mpc/h].

    Returns output in units h/Mpc
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    z = z_(chi_t/h) # z_ reads chi in units Mpc
    W = H(z)*Wg_j(z, zj)*b_g(z)/(c*h)
    return W


def Big_W(ch, c_M=0):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    Returns array of same shape as ch
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    n_int = 203

    """
    z_int_var = np.linspace(0.6, 1.4, n_int) # limited by Wbar_ti
    chi_tilda = chi(z_int_var)*h
    """
    z_int_var = np.linspace(0.6, 1.4, 2) # limited by Wbar_ti
    chi_tilda_range = chi(z_int_var)*h
    chi_tilda = np.linspace(chi_tilda_range[0], chi_tilda_range[-1], n_int)

    shapelist = [n_int] + list(np.shape(ch))
    diff = np.zeros(shapelist)
    np.transpose(diff)[:] = chi_tilda
    diff -= ch

    W2 = Wbar_k(chi_tilda, ch, c_M)
    if len(np.shape(W2)) == 3:
        W2 = np.transpose(W2,(1,2,0))
        Heavi = np.transpose(np.heaviside(diff, 0),(1,2,0))
    else:
        W2 = np.transpose(W2)
        Heavi = np.transpose(np.heaviside(diff, 0))

    # Heavi is necessary as integration variable must be larger than chi
    integrand = Wbar_ti(chi_tilda, c_M)*W2*Heavi
    # z_int_var corresponds to chi_tilda, and z to ch
    # Index of 0 is to take out the k-component
    return integrate(chi_tilda, integrand, ax=-1)


def Wvdiff(tarr, chiarr, c_M=0, N=100):
    """
    Returns W in units [(h/Mpc)^(3-nu)]
    """
    nz1 = 70
    nt = 50
    nz2 = nz1
    h = 0.6763
    t = np.linspace(0.01, 1, nt)
    z1 = np.linspace(0.7, 1.3, nz1)
    z2 = np.linspace(0.7, 1.3, nz2)

    ch1 = chi(z1)*h
    ch2 = np.zeros((nt, nz2))
    np.transpose(ch2)[:] = t
    ch2 *= chi(z2)*h
    print("Finding coeffs")
    t1 = time.time()
    coeffs = c_chi(ch1, ch2, N, samedim2=False, c_M=c_M)
    t2 = time.time()
    print("Time spent doing so: {:.2f} min".format((t2-t1)/60))
    coeffs[np.where(np.isnan(coeffs))] = 0

    def coeff(n):
        nn = np.copy(coeffs[n])
        np.transpose(nn)[:] = n
        c1 = coeffs[n]
        c2 = np.conj(coeffs[-n])
        return np.where(nn >= 0, c1, c2)

    N_arr = np.array([i for i in range(-N, N+1)])
    cs = coeff(N_arr)

    print("Interpolating")
    f = interpolate.interp1d(ch1, cs, axis=1, bounds_error=False, fill_value=0)

    def toderiv(x):
        ftransp = Wv1(x, c_M)*np.transpose(f(x), (0, 3, 2, 1))
        return np.transpose(ftransp, (0, 3, 2, 1))
    #toderiv = lambda x: f(x)*Wv1(x)

    print("Differentiating")
    deriv = ms.derivative(toderiv, ch1, 1e-3)
    deriv[:,-1,:,:] = deriv[:,-2,:,:] # Final point is unstable
                                      # due to differentiation

    """
    fplot = toderiv(ch1)
    ind1 = 0
    ind2 = 7
    ind3 = 43
    plt.plot(ch1, np.real(fplot[ind1, :, ind2, ind3]), ".--")
    plt.show()
    plt.plot(ch1, np.real(deriv[ind1, :, ind2, ind3]), ".--")
    plt.show()
    """

    W = np.transpose(deriv, (0, 2, 1, 3))
    W = np.diagonal(W, axis1=2, axis2=3)

    print("Interpolating again")
    VAL = np.zeros((len(N_arr), len(tarr), len(chiarr)), dtype=complex)
    for i in range(len(N_arr)):
        f = interpolate.RectBivariateSpline(t, ch1, np.real(W[i]))
        g = interpolate.RectBivariateSpline(t, ch1, np.imag(W[i]))
        np.real(VAL)[i, :, :] = f(tarr, chiarr)
        np.imag(VAL)[i, :, :] = g(tarr, chiarr)


    # DO A GRID-INTERPOLATION of DERIV AND EVALUATE AT (chinp1, chinp2)
    # MAYBE CLASH TOGETHER CHI-AXES FIRST???!!
    return VAL

"""
nt = 400
nch = 301
h = 0.6763
t = np.linspace(0.01, 1-1e-4, nt)

zrange = np.linspace(0.05,1.35,2)  # z-domain to integrate over
chirange = chi(zrange)*h
ch = np.linspace(chirange[0], chirange[-1], nch)

ch_t = np.zeros((nt, nch))
np.transpose(ch_t)[:] = t
ch_t *= ch

f = Wvdiff(t, ch)
print(np.shape(f))
"""

"""
nz = 10
z = np.linspace(0.01,1.3,nz)
ch = chi(z)*0.6763

nt = 500
t = np.geomspace(1e-2,0.99,nt)
ch_t = np.zeros((nt, nz))
np.transpose(ch_t)[:] = t
ch_t *= ch

zj = np.linspace(0.05,0.15,2)
leg=[]
for i in range(nz):
    f = Big_W(ch)[i]*Wbar_gj(ch_t, zj)[:,i]
    plt.plot(t, f,".--")
    leg.append("{}".format(z[i]))
plt.legend(leg)
plt.show()
"""

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



def plot_some(t,nus,Is,show=False,save=False,j=0,name=""):
    for i in range(0,len(nus),15):
        plt.plot(t, np.real(Is)[:,i],".--")
        #plt.plot(t, np.imag(Is)[:,i],".")
    if save:
        plt.savefig("plotsome/" + name + "plot{}".format(j))
    if show:
        plt.show()
    else:
        plt.close()
    return None

def plotCs(zg, C, C2, C3, l, N, nt, nch, timespent, fftlogindex, c_M):
    zgGR = np.load("zgGR.npy")
    CsgGR = np.load("CsgGR.npy")
    CtgGR = np.load("CtgGR.npy")
    plt.plot(zgGR, CsgGR, "-")
    plt.plot(zgGR, CtgGR, "-")

    plt.plot(zg,C,".")
    plt.plot(zg,C2,".")
    #plt.plot(zg,C3,".")
    plt.title("FFTlog, N={}, nt={}, nchi={}, t={:.1f}min, c_M={}".format(N, nt, nch, timespent, c_M), fontsize=10)
    plt.xlabel("$z_g$")
    plt.ylabel("$C^{wg}$")
    #plt.legend(["$Csg(l={})$".format(l), "$Ctg(l={})$".format(l)], frameon=False)
    plt.legend(["c_M={}".format(c_M), "c_M={}".format(c_M), "$Csg(l={})$".format(l), "$Ctg(l={})$".format(l)], frameon=False)
    #plt.legend(["Limber", "Limber", "Limber", "$Csg(l={})$".format(l), "$Ctg(l={})$".format(l), "$Cvg(l={})$".format(l)], frameon=False)
    plt.savefig("fftplots/fftplotgen{}".format(fftlogindex))
    plt.yscale("log")
    plt.axis([zg[0],zg[-1], 1e-8,1e-4])
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

def Ctg(l, t, ch, W, Wgj, Wbar, Wgjbar, nus, cs, Is, i=None):
    """
    Ctg power spectrum
    """

    F1 = create_F(t, ch, W, Wgj, nus, cs)
    F2 = create_F(t, ch, Wbar, Wgjbar, nus, cs)
    F = F1 + F2
    #plot_some(t,nus,F,save=True,j=i,name="F")
    #Limber_I = 2*np.pi**2*(l+0.5)**(nus-3) # "evaluated at t=1"
    #Limber_F = F[-1,:]                     # "evaluated at t=1"

    #plot_some(t,nus,F*Is,save=True,j=i,name="FI")
    integrand = np.sum(F*Is, axis=1)   # summing along nus-axis
    """
    plt.plot(t,np.real(integrand),".")
    plt.savefig("fftplots/Ctg_integrand/ints{}".format(i))
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

def Ctg2(l, t, ch, W, Wgj, W2, Wgj2, nus, cs, Is, i=None):
    """
    Ctg power spectrum
    """
    F1 = create_F(t, ch, W, Wgj, nus, cs)
    F2 = create_F(t, ch, W2, Wgj2, nus, cs)
    F = F1 + F2
    integrand = np.sum(F*Is, axis=1)   # summing along nus-axis
    C = integrate(t, integrand)
    #print("Final imaginary part: {}".format(np.imag(C)))
    ctg = 2/(4*np.pi**2)*np.real(C)
    return ctg

def Cvg(l, t, ch, Wv_2, Wv_diff, nus, I_v, i=None):
    """
    Cvg power spectrum
    """
    Ffac = np.zeros((len(t), len(ch), len(nus)), dtype=complex)
    chis = np.zeros((len(ch), len(nus)), dtype=complex)
    np.transpose(chis)[:] = ch
    chis **= (3 - nus)

    np.transpose(Ffac, (2,0,1))[:] = Wv_2*Wv_diff

    F = Ffac*chis

    Fint = integrate(ch, F, ax=1)
    integrand = np.sum(Fint*I_v, axis=1)   # summing along nus-axis
    C = integrate(t, integrand)
    print("Final imaginary part: {}".format(np.imag(C)))
    cvg = -2/(4*np.pi**2)*np.real(C)
    return cvg



def make_oguriplot(reset_runindex=False, l=2, N=100, c_M=0):
    if reset_runindex:
        np.save("fftlogindex.npy", 1)
    fftlogindex = np.load("fftlogindex.npy")
    np.save("fftlogindex.npy", fftlogindex + 1)

    time1 = time.time()
    nt0 = 10
    nt1 = 10
    nt2 = 200
    nt3 = 200
    # nt0 and nt1 should be small for large l
    nt = nt0 + nt1 + nt2 + nt3
    nch = 301
    h = 0.6763
    eps = 1e-5
    #t = np.linspace(0.001, 1-eps, nt)
    """
    Remember that C_vg needs more t-values!
    """

    t0 = np.geomspace(1e-4,0.1,nt0 + 1)
    t1 = np.linspace(0.01,0.5,nt1 + 1)
    t2 = np.linspace(0.5,0.999,nt2 + 1)
    t3 = np.geomspace(0.999,1,nt3)
    t = np.concatenate((t0[:-1], t1[:-1],t2[:-1],t3)) # No overlapping points
    #t = np.linspace(0.5, 1.5, nt)

    """
    zrange = np.linspace(0.05,1.35,nch)  # z-domain to integrate over
    ch = chi(zrange)*h                 # Corresponding chi-domain to
                                       # integrate over (in units Mpc/h)
    """

    zrange = np.linspace(0.05,1.75,2)  # z-domain to integrate over
    # High-end must be larger than 1.35 for c_M < 0!
    chirange = chi(zrange)*h
    ch = np.linspace(chirange[0], chirange[-1], nch)

    """
    z_t = np.zeros((nt, nch))
    np.transpose(z_t)[:] = t
    z_t *= zrange
    ch_t = chi(z_t)*h
    """
    ch_t = np.zeros((nt, nch))
    np.transpose(ch_t)[:] = t
    ch_t *= ch


    print("Finding Fourier coefficients")
    coeffs = c_chi(ch, ch_t, N, c_M=c_M)
    #print(np.shape(np.where(np.isnan(coeffs))))
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
    Wsi = Wbar_si(ch, c_M)
    Wsibar = Wbar_si(ch_t, c_M)
    Wbig = Big_W(ch, c_M)
    Wbigbar = Big_W(ch_t, c_M)
    #Wti = Wbar_ti(ch)
    #Wv_diff = Wvdiff(t, ch, N)

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


    """
    print("Computing Bessel-integrals I, type 2:")
    I_v = np.zeros((len(t), len(nus)), dtype=complex)
    for i in range(len(nus)):
        if i in [n for n in range(0, len(nus), 22)]:
            print("{}%".format(int(i/len(nus)*100)))
        I_v[:, i] = I(l, nus[i] - 2, t)[:,0]
    print("100%")
    """
    #plot_some(t,nus,Is,save=True,name="I")
    #plot_some(t,nus,Is,show=True)
    #check_integrands(t, ch, ch_t, cs, Is,nus)
    #return None

    """
    zj = np.linspace(0.95,1.05,2)
    Wgj = Wbar_gj(ch_t,zj)
    Wgjbar = Wbar_gj(ch,zj)
    return Csg(l, t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, Is, i)
    """

    delta_z = 0.1
    nzg1 = 40
    nzg2 = 80
    nzg = nzg1 + nzg2 # Number of data points
    zg1 = np.linspace(0.1, 0.7, nzg1 + 1)
    zg2 = np.linspace(0.7, 1.7, nzg2)
    zg = np.concatenate((zg1[:-1], zg2))
    C_sg = np.zeros(nzg)
    C_sg[:] = None # Just to avoid plotting everything
    C_tg = np.copy(C_sg)
    C_vg = np.copy(C_sg)
    LimberC_sg = np.copy(C_sg)
    LimberC_tg = np.copy(C_sg)
    LimberC_vg = np.copy(C_sg)



    from hi_MatterSpectrum import intpol
    zt = np.linspace(0.6,1.7,400)
    z = np.linspace(0.6,1.7,400)
    #zt2 = np.linspace(0.01,1.4,200)
    zt2 = np.linspace(0.05,1.7,400)
    karg = (l+0.5)/chi(zt)*1/h
    karg2 = (l+0.5)/chi(zt2)*1/h
    Pfunc = intpol(fetchP=True, c_M=c_M)
    P = Pfunc(karg,zt)
    P2 = Pfunc(karg2,zt2)
    #print("Prepping for LimberCvg")
    #Cvderiv = LimberCvgprep(l, zt)

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wgj = Wbar_gj(ch_t, zj)
        Wgjbar = Wbar_gj(ch, zj)
        #Wv_2 = Wv2(ch_t, zj)
        #C_sg[i] = Csg(l, t, ch, Wsi, Wgj, Wsibar, Wgjbar, nus, cs, Is, i)
        #C_tg[i] = Ctg(l, t, ch, Wbig, Wgj, Wbigbar, Wgjbar, nus, cs, Is, i)
        #C_tg[i] = Ctg2(l, t, ch, Wbigbar, Wgjbar, Wbig, Wgj, nus, cs, Is, i)
        #C_vg[i] = Cvg(l, t, ch, Wv_2, Wv_diff, nus, I_v, i)
        LimberC_sg[i] = LimberCsg(l, zj, P, zt, c_M=c_M)
        LimberC_tg[i] = LimberCtg(l, zj, P2, z, zt2, c_M=c_M)
        #LimberC_vg[i] = LimberCvg(l, zj, zt, Cvderiv)

    time2 = time.time()
    timespent = (time2-time1)/60
    print("Time spent: {:.1f} minutes".format(timespent))

    np.save("zgalpha{}.npy".format(c_M), zg)
    np.save("LimberCsgalpha{}.npy".format(c_M), LimberC_sg)
    np.save("LimberCtgalpha{}.npy".format(c_M), LimberC_tg)


    plt.plot(zg,LimberC_sg,".")
    plt.plot(zg,LimberC_tg,".")
    #plt.plot(zg,LimberC_vg,"--")
    plt.yscale("log")
    plt.axis([zg[0],zg[-1], 1e-8,1e-4])
    plt.show()

    """
    if c_M == 0:
        np.save("zgGR.npy", zg)
        np.save("CsgGR.npy", C_sg)
        np.save("LimberCsgGR.npy", LimberC_sg)
        np.save("CtgGR.npy", C_tg)
        np.save("LimberCtgGR.npy", LimberC_tg)
    else:
        np.save("zgalpha{}.npy".format(c_M), zg)
        np.save("Csgalpha{}.npy".format(c_M), C_sg)
        np.save("Ctgalpha{}.npy".format(c_M), C_tg)
    """

    #plotCs(zg, C_sg, C_tg, C_vg, l, N, nt, nch, timespent, fftlogindex, c_M)


#make_oguriplot(l=100, c_M=0)



def Plot4Tessa():
    zgGR = np.load("zgGR.npy")
    CsgGR = np.load("CsgGR.npy")
    CtgGR = np.load("CtgGR.npy")

    zg1 = np.load("zgalpha1.npy")
    Csg1 = np.load("Csgalpha1.npy")
    Ctg1 = np.load("Ctgalpha1.npy")

    zg_1 = np.load("zgalpha-1.npy")
    Csg_1 = np.load("Csgalpha-1.npy")
    Ctg_1 = np.load("Ctgalpha-1.npy")

    plt.plot(zgGR, CsgGR, "r-")
    plt.plot(zg1, Csg1, "g-")
    plt.plot(zg_1, Csg_1, "b-")

    plt.plot(zgGR, CtgGR, "r--")
    plt.plot(zg1, Ctg1, "g--")
    plt.plot(zg_1, Ctg_1, "b--")

    plt.legend(["c_M=0", "c_M=1", "c_M=-1"], frameon=False)
    plt.yscale("log")
    plt.axis([zg_1[0],zg_1[-1], 1e-8,1e-4])
    plt.show()

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
