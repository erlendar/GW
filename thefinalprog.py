import numpy as np
import matplotlib.pyplot as plt
from genfuncs import *
from MatterSpectrum import P_m_equaltime as P_m
from fcosm import Wbar_si, Wbar_gj
from scipy.special import gamma
from scipy import interpolate
import time

"""
--------
chi should always be in units Mpc/h,
k should always be in units h/Mpc
--------
"""
h = 0.6763
c = 299792458

delta_zg = 0.1
delta_zw = 0.2

nk = 6001
nch_w = 1201
nch_g = 600
# These values should be set carefully
# nch_w and nch_g should be allowed to differ as delta_z does so
nz_intpol = int(1e3)

k = np.geomspace(1e-4, 1.5, nk)
k_grid_w = np.zeros((nk, nch_w))
k_grid_g = np.zeros((nk, nch_g))
np.transpose(k_grid_w)[:] = k
np.transpose(k_grid_g)[:] = k
z_intpol = np.linspace(0.1, 1.5, nz_intpol)

sqrt_Pm_intpol = np.sqrt(P_m(k, z_intpol))
sqrt_Pm_of_chi = interpolate.interp1d(chi(z_intpol)*h, sqrt_Pm_intpol, \
                                      bounds_error=False, fill_value=0)


def chi_integral(ch, l, W, sqrt_Pm, k_grid):
    """
    This function performs the chi-integral in
    the expression for the cross-correlations.

    Takes the following inputs:
    ch         -- (1D array) -- Chi-values to integrate over
    l          -- (Integer)  -- Multipole to evaluate at
    W          -- (1D array) -- Window function evaluated at ch
    k_grid     -- (2D array) -- k-array of shape (nk, nch)

    Returns:
    1D array with length nk
    """

    integrand = W * sqrt_Pm * j(k_grid * ch, l)
    I = integrate(ch, integrand, ax=-1)
    return I


def run_chi_integral(ls, z_bin, GW=False):
    """
    This function returns the chi-integral
    for several values of l.

    Takes the following inputs:
    ls         -- (1D array)     -- Multipoles to evaluate at
    z_bin      -- (1D array) -- z-bin represented as (z_min, z_max)
    GW         -- (boolean)  -- True if we're considering the GW-window func,
                                False if we're considering the galaxy-window.

    Returns:
    2D array of shape (nls, nk) that consists of the
    chi-integral for all the given values of l.
    """
    if GW:
        k_grid = k_grid_w
        eps = 1e-3
        chmin = chi(z_bin[0] - eps)*h
        chmax = chi(z_bin[-1] + eps)*h
        ch = np.linspace(chmin, chmax, nch_w)
        W = Wbar_si(ch, z_bin)
    else:
        k_grid = k_grid_g
        chmin = chi(z_bin[0])*h
        chmax = chi(z_bin[-1])*h
        ch = np.linspace(chmin, chmax, nch_g)
        W = Wbar_gj(ch, z_bin)

    sqrt_Pm = sqrt_Pm_of_chi(ch)

    nls = len(ls)
    Integrals = np.zeros((nls, nk))
    for i in range(nls):
        l = ls[i]
        #print("Working on multipole l={}".format(l))
        Integrals[i] = chi_integral(ch, l, W, sqrt_Pm, k_grid)
    return Integrals


def Correlation(ls, GW_auto=False, galaxy_auto=False, cross_corr=False,\
                z_bin_w=None, z_bin_g=None):
    """
    This program is currently assuming that we're not interested in auto-
    correlations for different redshift-bins.
    """
    if GW_auto:
        GW_Integrals = run_chi_integral(ls, z_bin_w, GW=True)
        Integral = GW_Integrals**2
    elif galaxy_auto:
        Galaxy_Integrals = run_chi_integral(ls, z_bin_g, GW=False)
        Integral = Galaxy_Integrals**2
    elif cross_corr:
        GW_Integrals = run_chi_integral(ls, z_bin_w, GW=True)
        Galaxy_Integrals = run_chi_integral(ls, z_bin_g, GW=False)
        Integral = GW_Integrals*Galaxy_Integrals
    Integrand = k**2 * Integral
    #for i in range(len(ls)):
    #    plt.plot(k, Integrand[i,:],".-")
    #plt.show()
    Cs = 2/np.pi * integrate(k, Integrand, ax=-1)
    return Cs


def Covariance(ls, string, z_bin_w=None, z_bin_g=None):
    from number_dens3 import Number_dens
    Iw = Number_dens(z_bin_w)
    Ig = Number_dens(z_bin_g)
    n_w = Iw.n_av()
    n_g = Iw.nav_g()
    delta_l = 1
    f_sky = 0.5
    shot_noise_g = 1/n_g
    shot_noise_w = 1/n_w

    if string.count("w") == 0:
    # Means that we're looking at the covariance Cgg x Cgg
        Cgg = Correlation(ls, galaxy_auto=True,\
                          z_bin_g=z_bin_g)
        Cov = 2*(Cgg + shot_noise_g)**2

    elif string.count("w") == 1:
    # Means that we're looking at the covariance Cwg x Cgg
        Cwg = Correlation(ls, cross_corr=True,\
                          z_bin_w=z_bin_w, z_bin_g=z_bin_g)
        Cgg = Correlation(ls, galaxy_auto=True,\
                          z_bin_g=z_bin_g)
        Cov = 2*Cwg*(Cgg + shot_noise_g)

    elif string.count("w") == 2:
        if string[:2].count("w") == 1:
        # Means that we're looking at the covariance Cwg x Cwg
            Cww = Correlation(ls, GW_auto=True,\
                              z_bin_w=z_bin_w)
            Cgg = Correlation(ls, galaxy_auto=True,\
                              z_bin_g=z_bin_g)
            Cwg = Correlation(ls, cross_corr=True,\
                              z_bin_w=z_bin_w, z_bin_g=z_bin_g)
            Cov = (Cww + shot_noise_w)*(Cgg + shot_noise_g) + Cwg**2
        else:
        # Means that we're looking at the covariance Cww x Cgg
            Cwg = Correlation(ls, cross_corr=True,\
                              z_bin_w=z_bin_w, z_bin_g=z_bin_g)
            Cov = 2*Cwg**2

    elif string.count("w") == 3:
    # Means that we're looking at the covariance Cww x Cwg
        Cww = Correlation(ls, GW_auto=True,\
                          z_bin_w=z_bin_w)
        Cwg = Correlation(ls, cross_corr=True,\
                          z_bin_w=z_bin_w, z_bin_g=z_bin_g)
        Cov = 2*(Cww + shot_noise_w)*Cwg

    elif string.count("w") == 4:
    # Means that we're looking at the covariance Cww x Cww
        Cww = Correlation(ls, GW_auto=True,\
                          z_bin_w=z_bin_w)
        Cov = 2*(Cww + shot_noise_w)**2

    Cov *= 1/((2*ls + 1)*delta_l*f_sky)
    return Cov

"""
t1 = time.time()
ls = np.array([5, 20, 40, 100])
z_w = np.linspace(0.9, 1.1, 2)
z_g = np.linspace(0.95, 1.05, 2)
string = "wgwg"
Covs = Covariance(ls, string, z_bin_w=z_w, z_bin_g=z_g)
print(time.time() - t1)
print(np.sqrt(Covs))
"""















#
