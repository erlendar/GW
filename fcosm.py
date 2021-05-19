import numpy as np
import matplotlib.pyplot as plt
from genfuncs import *
from fourierpolexp import c_chi, nu_func, Pm, b
from besselint import Ibackwards, Ibackwardsmat, I
from directcsg import LimberCsg, LimberCtg, LimberCvg, LimberCvgprep
from scipy import interpolate
import scipy.misc as ms
import time


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


def Big_W(ch, c_M=0, zj=[0.5]):
    """
    Window function used to find fcosm
    In units h/Mpc

    Takes chi as input, with units [Mpc/h]
    Returns array of same shape as ch
    """
    h = 0.6763
    c = 299792458/1000 # km/s
    n_int = 303

    """
    z_int_var = np.linspace(0.6, 1.4, n_int) # limited by Wbar_ti
    chi_tilda = chi(z_int_var)*h
    """
    z_int_var = np.linspace(zj[0], 1.75, 2) # limited by Wbar_ti
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





def create_F(t, ch, Wsi, Wgj, nus, cs):
    F = np.zeros((len(t),len(ch), len(nus)), dtype=complex)
    Ffac = np.copy(F)
    np.transpose(Ffac, (2,0,1))[:] = Wsi*Wgj

    np.transpose(F, (0,2,1))[:] = ch
    F **= (1 - nus)
    F *= Ffac*np.transpose(cs, (1, 2, 0))
    Fintegral = integrate(ch, F, ax=1) # integrating ch-axis
    return Fintegral


def make_oguriplot(reset_runindex=False, N=100, c_M=0, savearrays=False):
    if reset_runindex:
        np.save("fftlogindex.npy", 1)
    fftlogindex = np.load("fftlogindex.npy")
    np.save("fftlogindex.npy", fftlogindex + 1)

    """
    INTERPOLATE f-cosm func FOR t-values!!!
    """

    time1 = time.time()
    nt1 = 300
    nt2 = 100
    nt = nt1 + nt2
    nch = 101
    nch_intp = 500
    h = 0.6763
    """
    Remember that C_vg needs more t-values!
    """
    t1 = np.linspace(0,4,nt1+1)[:-1]
    t2 = np.linspace(4,22,nt2)
    t = np.concatenate((t1, t2))

    zrange_intp = np.linspace(0.05, 1.75, 2)
    chirange_intp = chi(zrange_intp)*h
    ch_intp = np.linspace(chirange_intp[0], chirange_intp[-1], nch_intp)

    ch_t_intp = np.zeros((nt, nch_intp))
    np.transpose(ch_t_intp)[:] = t
    ch_t_intp *= ch_intp

    print("Finding Fourier coefficients")
    coeffs = c_chi(ch_intp, ch_t_intp, N, c_M=c_M)
    #print(np.shape(np.where(np.isnan(coeffs))))
    coeffs[np.where(np.isnan(coeffs))] = 0

    def coeff(n):
        nn = np.copy(coeffs[n])
        np.transpose(nn)[:] = n
        c1 = coeffs[n]
        c2 = np.conj(coeffs[-n])
        return np.where(nn >= 0, c1, c2) # The three arguments must be able to be broadcasted together

    N_arr = np.array([i for i in range(-N, N+1)])
    nus = nu_func(N_arr) + b
    cs = coeff(N_arr)

    print("Interpolating")
    cs_intp = interpolate.interp1d(ch_intp, cs, axis=2, bounds_error=False, fill_value=0)

    bigz = np.linspace(1e-12, 1.75,2)
    bigchirange=chi(bigz)*h
    bigchi = np.linspace(bigchirange[0], bigchirange[-1], 500)
    Wsi = Wbar_si(ch_intp, c_M)
    Wbig = Big_W(bigchi, c_M)
    Wsi_intp = interpolate.interp1d(ch_intp, Wsi, axis=0, bounds_error=False, fill_value=0)
    Wbig_intp = interpolate.interp1d(bigchi, Wbig, axis=0, bounds_error=False, fill_value=0)


    ch_t_temp = np.zeros((nt, nch))
    np.transpose(ch_t_temp)[:] = t
    def makech(zj):
        chirange = chi(zj)*h
        ch = np.linspace(chirange[0], chirange[-1], nch)
        ch_t = np.copy(ch_t_temp)
        ch_t *= ch
        return ch, ch_t

    delta_z = 0.1
    nzg = 200
    zg = np.linspace(0.1, 1.7, nzg)
    Fsg = np.zeros((nzg, len(t), len(nus)), dtype=complex)
    Ftg = np.zeros((nzg, len(t), len(nus)), dtype=complex)

    for i in range(nzg):
        print("\nComputing datapoint {} out of {}".format(i+1,nzg))
        zg_elem = zg[i]
        zj = np.linspace(zg_elem-delta_z/2, zg_elem+delta_z/2, 2) # Galaxy bin

        ch, ch_t = makech(zj)
        Wgj = Wbar_gj(ch, zj)
        cs = cs_intp(ch)
        Wsi = Wsi_intp(ch_t)
        Wbig = Wbig_intp(ch_t)

        Fsg[i] = create_F(t, ch, Wsi, Wgj, nus, cs)
        Ftg[i] = create_F(t, ch, Wbig, Wgj, nus, cs)

    time2 = time.time()
    timespent = (time2-time1)/60
    print("Time spent: {:.1f} minutes".format(timespent))

    if savearrays:
        np.save("zFcm{}.npy".format(c_M), zg)
        np.save("nus.npy", nus)
        np.save("t_intpol.npy", t) # for interpolation later
        np.save("Fsgcm{}.npy".format(c_M), Fsg)
        np.save("Ftgcm{}.npy".format(c_M), Ftg)



#make_oguriplot(c_M=0, savearrays=False)






#
