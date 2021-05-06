import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy.integrate as inte
import scipy.misc as ms
import time
from scipy import interpolate
from class5 import CLASS
from genfuncs import *
from MatterSpectrum import P_m_equaltime, P_m
from hi_MatterSpectrum import intpol

# Oguri-related papers:
# https://arxiv.org/pdf/2002.02466.pdf
# https://arxiv.org/pdf/1809.03528.pdf

# all n's equal to 4e2 OK, while nz_tg = 4e1

# Kopiere over (fra hjem paa pcn): rsync -rv erlendar@euclid.uio.no:/uio/hume/student-u20/erlendar/Documents/Master/Cs /Users/erlendaars/Documents/GWsummerjob/myprogs/data

# pwd at Master/Cluster: /uio/hume/student-u20/erlendar/Documents/Master
# pwd at Cs/Cluster:     /uio/hume/student-u20/erlendar/Documents/Master/Cs

# pwd at Cs/Mydata :     /Users/erlendaars/Documents/GWsummerjob/myprogs/data/Cs



"""
Problem:
When trying to create arrays of dimension (500,500,500,500),
terminal shuts down, saying "Killed: 9".

Same thing happens when running with (200, 200, 200, 200).
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



l = 100

z_i = np.linspace(0.9,1.1,2)  # GW bin
z_imin = z_i[0]; z_imax = z_i[-1]

if __name__ == "__main__":
    n = 5e2
    nz = int(n)
    nzt = int(n)
    nk = int(n)
    nzp = int(n)
    nz_tg = int(5e1)

    wobble = 0.3

    k = np.geomspace(1.4e-5, 1.6, nk)
    # Area where Pm(k) gives the main contribution
    # Better to use geometric spacing as the range
    # is over multiple orders of magnitude

    z = np.linspace(z_imin - wobble, z_imax + wobble, nz)
    # z-integral contains a GW-bin window function, so integrating
    # far outside the range of this bin is pointless

    z_tg = np.linspace(z_imin - wobble, z_imax + wobble, nz_tg)
    # For Ctg the z-variable is not evaluated by the Bessel functions
    # so we can use fewer values to integrate over

    zt = np.linspace(0.05,1.4,nzt)
    # zt-integral contains a galaxy-bin window function, so integrating
    # far outside the range of this bin is pointless

    zp = np.linspace(0.05, z[-1], nzp)
    # Should always be accompanied by a Heaviside function
    # that limits the zp-integral to maximum z

    split_n = 20 # Number of "nodes" to run program on
    partial_n = int(nz_tg/split_n)


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
#FOR TESTING!!!
def Ws_i(z):
    return np.heaviside(z-z_imin,0)*np.heaviside(z_imax-z,0)

def Wg_j(z, zj):
    return np.heaviside(z-zj[0],0)*np.heaviside(zj[-1]-z,0)

def b_g(z):
    x = np.ones(len(z))
    return x

def b_GW(z):
    x = np.ones(len(z))
    return x

def P_m(k, z, z_prime):
    nk = len(k)
    nz = len(z)
    nz_prime = len(z_prime)
    P = np.zeros((nk, nz, nz_prime))
    x = 1e-2
    gauss = np.exp(-(k-x)**2/1e-5)
    np.transpose(P)[:] = gauss
    return P


#TESTING DONE!!!
"""

"""
#def Csgfunc(l, Csgfacs, Wg, k=None, z=None, zt=None):
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


#def Ctgfunc(l, Ctgfacs, Wg, k=None, z_tg=None, zp=None, zt=None):
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
        Integral = integrate(z_tg,outermiddle_integrand(l),1)
        return outerfacs*Integral

    def Ctg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi
    return Ctg(l)
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
        Integral = integrate(z_tg,outermiddle_integrand(l),1)
        return outerfacs*Integral

    def Ctg(l=2):
        Integral = integrate(k,outer_integrand(l))
        return Integral*2/np.pi
    return Ctg(l)


#def Cvgfunc(l, Cvgfacs, Wg, k=None, z=None, zt=None):
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


#def compute_integrands(P, P2, k=None, z=None, zt=None, zp=None, z_tg=None, \
#                       sg=False, tg=False, vg=False):
def compute_integrands(P, P2, sg=False, tg=False, vg=False):
    nk = len(k)
    nz = len(z)
    nz_tg = len(z_tg)
    nzp = len(zp)
    nzt = len(zt)

    def svcommon():
        A = np.ones((nk, nz, nzt))
        x1 = np.copy(A); x2 = np.copy(A)

        np.transpose(x1,(0,2,1))[:] = chiz
        kz = np.transpose(np.transpose(x1)*k)
        x2[:] = chizt
        kzt = np.transpose(np.transpose(x2)*k)
        A[:] = bg
        return A, kz, kzt

    def sgintegrand():
        A, kz, kzt = svcommon()
        innerfac_Csg = A*P*j(kz,l)*j(kzt,l)
        middlefac_Csg = Ws*bGW
        outerfac_Cstg = k**2
        Csgfacs = [innerfac_Csg, middlefac_Csg, outerfac_Cstg]
        return Csgfacs

    def vgintegrand():
        A, kz, kzt = svcommon()
        innerfac_Cvg = A*P*dj(kz,l)*j(kzt,l)
        middlefac_Cvg = Wvv*Wt
        outerfac_Cvg = k**3
        Cvgfacs = [innerfac_Cvg, middlefac_Cvg, outerfac_Cvg]
        return Cvgfacs

    def tgintegrand():
        Wt = Wt_i(z_tg)
        B = np.ones((nk, nz_tg, nzp, nzt))
        y1 = np.copy(B); y2 = np.copy(B)
        PP = np.copy(B)

        np.transpose(y1,(0,1,3,2))[:] = chizp
        kzp = np.transpose(np.transpose(y1)*k)
        y2[:] = chizt
        kzt2 = np.transpose(np.transpose(y2)*k)
        np.transpose(PP, (1,0,2,3))[:] = P2
        B[:] = bg

        H = np.ones((nk, nz_tg, nzp))
        H[:] = zp
        difftransp = z_tg - np.transpose(H, (0, 2, 1))
        diff = np.transpose(difftransp, (0,2,1))
        Heaviside = np.heaviside(diff, 0)

        innerfac_Ctg = B*PP*j(kzp,l)*j(kzt2,l)
        innermiddlefac_Ctg = Wkappa*Heaviside
        outermiddlefac_Ctg = Wt
        outerfac_Cstg = k**2

        Ctgfacs = [innerfac_Ctg, innermiddlefac_Ctg, \
                   outermiddlefac_Ctg, outerfac_Cstg]
        return Ctgfacs

    Ws = Ws_i(z); Wt = Wt_i(z)
    Wvv = Wv(z, k)
    bGW = b_GW(z); bg = b_g(zt)
    chiz = chi(z); chizp = chi(zp)
    chizt = chi(zt)
    ints = []

    if sg:
        ints.append(sgintegrand())
    if tg:
        Wkappa = Wk(z_tg, zp, k)
        ints.append(tgintegrand())
    if vg:
        ints.append(vgintegrand())
    return ints
"""

def LimberCsg(l, zj, P=None, zt=None, c_M=0):
    c = 299792458
    h = 0.6763
    if zt is None:
        zt = np.linspace(0.6,1.4,200)
    if P is None:
        karg = (l+0.5)/chi(zt)*1/h
        Pfunc = intpol(fetchP=True, c_M=c_M)
        P = Pfunc(karg,zt)
    integrand = (Ws_i(zt) + W_MG(zt, c_M))*Wg_j(zt, zj)*H(zt)/chi(zt)**2*b_GW(zt)*b_g(zt)*P
    integrand *= 1e3/c/h**3 # unitless
    Integral = integrate(zt,integrand)
    return Integral

def LimberCtg(l, zj, P=None, z=None, zt=None, c_M=0):
    c = 299792458
    h = 0.6763
    if z is None:
        z = np.linspace(0.6,1.4,200)
    if zt is None:
        zt = np.linspace(0.1,1.4,200)
    karg = (l+0.5)/chi(zt)*1/h
    if P is None:
        Pfunc = intpol(fetchP=True, c_M=c_M)
        P = Pfunc(karg,zt)
    nz = len(z); nzt = len(zt)

    A = np.zeros((nz,nzt))
    A[:] = zt
    diff = np.transpose(z - np.transpose(A))
    Heaviside = np.heaviside(diff, 0)

    inner_integrand = Wg_j(zt, zj)*Wk2(z,zt,karg, c_M)*H(zt)/chi(zt)**2*b_g(zt)*P*Heaviside
    integral1 = integrate(zt,inner_integrand,1)
    outer_integrand = Wt_i(z)*integral1
    integral2 = integrate(z,outer_integrand)
    return integral2*1e3/c/h**3

def LimberCvgprep(l, z):
    """
    Returns the differentiated factor in Limber_Cvg
    in units [(Mpc/h)^3]
    """
    c = 299792458/1000 # km/s
    h = 0.6763
    karg = (l + 0.5)/chi(z)*1/h
    x = np.linspace(0.6, 1.4, len(z))
    """
    To simplify the differentiation of P_m(k,z,z2) we will
    split the power spectrum in the factors corresponding to the
    geometric approximation
    """

    toderiv = H(x)*Wt_i(x)*Wv(x, karg)*np.sqrt(P_m_equaltime(karg, x))
    f = interpolate.interp1d(x, toderiv, axis=-1, bounds_error=False, fill_value=0)

    deriv = ms.derivative(f, z, 1e-5)*np.sqrt(P_m_equaltime(karg, z))
    # Multiplying the final P_m factor according to the geometric approx.

    """
    ind = 2
    plt.plot(x, toderiv[ind, :],".--")
    plt.show()

    plt.plot(z, deriv[ind, :],".--")
    plt.show()
    """

    prep = deriv*1/c
    prep = np.diagonal(prep) # We want the diagonal because this corresponds
                             # to both arguments of P being evaluated at the
                             # same z!
    return prep

#z = np.linspace(0.6, 1.4, 200)
#LimberCvgprep(100, z)


def LimberCvg(l, zj, z=None, deriv=None):
    """
    Unitless
    """
    c = 299792458/1000 # km/s
    h = 0.6763
    if z is None:
        eps = 1e-4
        z = np.linspace(0.05 - eps, 1.35 + eps, 200)
    if deriv is None:
        deriv = LimberCvgprep(l, z)
    Fac = H(z)/chi(z)**2*Wg_j(z, zj)*b_g(z)
    integrand = Fac*deriv
    C = integrate(z, integrand)
    C *= -1/c*1/h**3
    return C



def PlotLimbers():
    runindex = np.load("runindex.npy")
    np.save("runindex.npy", runindex + 1)

    h = 0.6763
    delta_z = 0.1
    nzg1 = 24
    nzg2 = 27
    nzg = nzg1 + nzg2 # Number of data points
    zg1 = np.linspace(0.1, 0.8, nzg1 + 1)
    zg2 = np.linspace(0.8, 1.3, nzg2)
    zg = np.concatenate((zg1[:-1], zg2))
    #C_sgLimber = np.copy(zg)
    C_tgLimber = np.copy(zg)
    #C_vgLimber = np.copy(zg)

    zt = np.linspace(0.1,1.4,200)
    karg = (l+0.5)/chi(zt)*1/h
    Pfunc = intpol()
    P = Pfunc(karg,zt)

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wg = Wg_j(zt, zj)
        #C_sgLimber[i] = LimberCsg(l, zj, P, zt)
        C_tgLimber[i] = LimberCtg(l, zj, P)
        #C_vgLimber[i] = LimberCvg(l, zj, P)
    #np.save("data/Cs/Csglimber.npy", C_sgLimber)
    #np.save("data/Cs/Ctglimber.npy", C_tgLimber)
    #np.save("data/Cs/Cvglimber.npy", C_vgLimber)
    #np.save("data/Cs/zglimber.npy", zg)
    #plt.plot(zg, C_sgLimber,"r.-")
    plt.plot(zg, C_tgLimber,"b.-")
    #plt.plot(zg, C_vgLimber,"g.-")

    #x = np.load("zgdata.npy")
    #C = np.load("Csgdata.npy")
    #plt.plot(x, C, "k.-")

    #plt.legend(["C_sgLimber(l={})".format(l), "C_tgLimber(l={})".format(l)])#, \
                #"C_vgLimber(l={})".format(l)])
    plt.yscale("log")
    plt.axis([0.1,1.3, 1e-8,1e-4])
    #plt.axis([0.01,1.4, 1e-12,1e-3])
    #plt.savefig("data/Limbers{}".format(runindex))
    plt.show()
    return None
#PlotLimbers()


def plotC(x, functions, dt, sg=False, tg=False, vg=False):
    """
    x: array of points along the x-axis
    functions: list where each entry is an array
               corresponding to points along the y-axis
    legend: List of strings for the legend
    """
    n = len(functions)
    runindex = np.load("runindex.npy")
    legends=[]
    if n == 3:
        legends = ["Csg", "Ctg", "Cvg"]
    elif n == 2:
        if not sg:
            legends = ["Ctg", "Cvg"]
        if not tg:
            legends = ["Csg", "Cvg"]
        if not vg:
            legends = ["Csg", "Ctg"]
    else: # n == 1
        if sg:
            legends = ["Csg"]
        if tg:
            legends = ["Ctg"]
        if vg:
            legends = ["Cvg"]
    formats = ["r.-", "b.-", "g.-","k.-","y.-","m.-","c.-"]
    plots = []
    for i in range(n):
        plt.plot(x,functions[i],"r-.")
        plt.title("nk, nz, nzt, nzp = ({}, {}, {}, {}), t = {:.2f} min, l={}"\
                  .format(nk,nz,nzt,nzp,dt/60,l),loc="right")
        plt.legend([legends[i]])
        #plt.yscale("log")
        #plt.axis([0.1,1.3, 1e-8,1e-4])
        plt.savefig("directplots/" + legends[i] + "s/myplot{}".format(runindex))
        plt.close()
        plots.append(x)
        plots.append(functions[i])
        plots.append(formats[i])

    plt.plot(*plots)
    plt.title("nk, nz, nzt, nzp = ({}, {}, {}, {}), t = {:.2f} min, l={}"\
              .format(nk,nz,nzt,nzp,dt/60,l),loc="right")
    plt.legend(legends)
    #plt.yscale("log")
    #plt.axis([0.1,1.3, 1e-8,1e-4])
    plt.savefig("directplots/All/myplot{}".format(runindex))
    plt.show()
    return None


def run(reset_runindex=False, sg=False, tg=False, vg=False, z_tg=None):
    if reset_runindex:
        np.save("runindex.npy", 1)
    runindex = np.load("runindex.npy")
    np.save("runindex.npy", runindex + 1)

    delta_z = 0.1
    nzg = 30 # Number of data points
    zg = np.linspace(0.1,1.3,nzg) # Data points for final plot

    t1 = time.time()
    if sg:
        C_sg = np.copy(zg)
    if vg:
        C_vg = np.copy(zg)
    if tg:
        C_tg = np.copy(zg)
        print("Computing P2_m ...")
        P2 = P_m(k, zp, zt)
        print("... P2_m found")
    else:
        P2 = None
    if sg or vg:
        print("Computing P_m ...")
        P = P_m(k, z, zt)
        print("... P_m found")
    else:
        P = None
    t12 = time.time()

    print("Computing integrands ...")
    ints = compute_integrands(P=P, P2=P2, sg=sg, tg=tg, vg=vg)#, z_tg=z_tg)
    print("... Integrands found")
    t22 = time.time()

    if len(ints) == 3:
        Csgints, Ctgints, Cvgints = ints
    elif len(ints) == 2:
        if not sg:
            Ctgints, Cvgints = ints
        if not tg:
            Csgints, Cvgints = ints
        if not vg:
            Csgints, Ctgints = ints
    else: # len(ints) == 1
        if sg:
            Csgints = ints[0]
        if tg:
            Ctgints = ints[0]
        if vg:
            Cvgints = ints[0]

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin
        Wg = Wg_j(zt, zj)
        if sg:
            C_sg[i] = Csgfunc(l, Csgints, Wg)
        if tg:
            C_tg[i] = Ctgfunc(l, Ctgints, Wg)
        if vg:
            C_vg[i] = Cvgfunc(l, Cvgints, Wg)

    t2 = time.time()
    print("\nTime spent computing P: {:.2f} minutes".format((t12 - t1)/60))
    print("Time spent computing integrands: {:.2f} minutes".format((t22 - t12)/60))
    print("Total time spent: {:.2f} minutes".format((t2-t1)/60))

    plotlist = []
    if sg:
        np.save("Csgdata.npy", C_sg)
        plotlist.append(C_sg)
    if tg:
        np.save("Ctgdata.npy", C_tg)
        plotlist.append(C_tg)
    if vg:
        np.save("Cvgdata.npy", C_vg)
        plotlist.append(C_vg)
    np.save("zgdata.npy", zg)

    dt = t2-t1
    if sg or tg or vg:
        plotC(zg, plotlist, dt, sg, tg, vg)

    return None

#if __name__ == "__main__":
#    run(sg=True)
#    run(sg=True,tg=True,vg=True)


"""
For Csg(l=100) with a 500x500x500-grid:

Time spent computing P: 6.04 minutes
Time spent computing integrands: 6.42 minutes
Total time spent: 14.08 minutes

For all(l=2) with a 150x150x150x150-grid:
Time spent computing P: 3.52 minutes
Time spent computing integrands: 11.30 minutes
Total time spent: 41.44 minutes
"""



#
