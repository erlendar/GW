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
"""
SHAPE OF GAUSSIAN RANDOM VECTOR C^{ij}:
(C_{w_1w_1}, ..., C_{w_Nw_N}, C_{g_1g_1}, ..., C_{g_Mg_M},
 C_{w_1g_1}, ..., C_{w_Ng1}, ..., C_{w_1g_M}, ..., C_{w_Ng_M})
i. e. autocorrelations first, starting with w,
then cross correlations with changing w-bins for
each galaxy bin.
"""


h = 0.6763
c = 299792458

redshiftrange = np.linspace(0.3, 1.5, 2)
zmin = redshiftrange[0]
zmax = redshiftrange[-1]
n_zw = 6 # Number of redshift bins
n_zg = 12
delta_zw = round((zmax - zmin)/n_zw, 10) # Size of redshift bins
delta_zg = round((zmax - zmin)/n_zg, 10)
zw = np.linspace(zmin + delta_zw/2, zmax - delta_zw/2, n_zw)
zg = np.linspace(zmin + delta_zg/2, zmax - delta_zg/2, n_zg)
w_bins = np.transpose(np.linspace(zw-delta_zw/2, zw+delta_zw/2, 2))
g_bins = np.transpose(np.linspace(zg-delta_zg/2, zg+delta_zg/2, 2))
bin_dict = {"w": w_bins, "g": g_bins}
# Transpose as we want the shape (what bin, elements in bin)

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


def run_chi_integral(ls, z_bin, GW=False, save=False, bin_no=None):
    """
    This function returns the chi-integral
    for several values of l.

    Takes the following inputs:
    ls         -- (1D array, dtype=int)     -- Multipoles to evaluate at
    z_bin      -- (1D array) -- z-bin represented as (z_min, z_max)
    GW         -- (boolean)  -- True if we're considering the GW-window func,
                                False if we're considering the galaxy-window.

    Returns:
    2D array of shape (nls, nk) that consists of the
    chi-integral for all the given values of l.
    """
    if GW:
        label = "s"
        tot = n_zw
        k_grid = k_grid_w
        eps = 1e-3
        chmin = chi(z_bin[0] - eps)*h
        chmax = chi(z_bin[-1] + eps)*h
        ch = np.linspace(chmin, chmax, nch_w)
        W = Wbar_si(ch, z_bin)
    else:
        label = "g"
        tot = n_zg
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
    if save:
        np.save("chiintegrals/"+label+"bin{}of{}_zfrom{}to{}_10"\
                .format(bin_no, tot, int(zmin*10), int(zmax*10)), Integrals)
        return None
    return Integrals


def save_chi_integrals():
    ls = np.linspace(0, 100, 101, dtype=int)
    #ls = np.linspace(10, 15, 6, dtype=int)
    i = 0
    for bin in w_bins:
        i += 1
        print("Working on GW-bin {} out of {}".format(i, n_zw))
        run_chi_integral(ls, bin, GW=True, save=True, bin_no=i)
    i = 0
    for bin in g_bins:
        i += 1
        print("Working on galaxy-bin {} out of {}".format(i, n_zg))
        run_chi_integral(ls, bin, GW=False, save=True, bin_no=i)
#save_chi_integrals()



def Correlation(ls=[5], GW_auto=False, galaxy_auto=False, cross_corr=False,\
                z_bin_w=None, z_bin_g=None,\
                GW_Integrals=None, Galaxy_Integrals=None):
    """
    This program is currently assuming that we're not interested in auto-
    correlations for different redshift-bins.
    """
    if GW_auto:
        if GW_Integrals is None:
            GW_Integrals = run_chi_integral(ls, z_bin_w, GW=True)
        Integral = GW_Integrals**2
    elif galaxy_auto:
        if Galaxy_Integrals is None:
            Galaxy_Integrals = run_chi_integral(ls, z_bin_g, GW=False)
        Integral = Galaxy_Integrals**2
    elif cross_corr:
        if GW_Integrals is None:
            GW_Integrals = run_chi_integral(ls, z_bin_w, GW=True)
        if Galaxy_Integrals is None:
            Galaxy_Integrals = run_chi_integral(ls, z_bin_g, GW=False)
        Integral = GW_Integrals*Galaxy_Integrals
    Integrand = k**2 * Integral
    #for i in range(len(ls)):
    #    plt.plot(k, Integrand[i,:],".-")
    #plt.show()
    Cs = 2/np.pi * integrate(k, Integrand, ax=-1)
    return Cs

def Correlations_load_save():
    nl = 101
    autocorr_w = np.zeros((n_zw, nl))
    autocorr_g = np.zeros((n_zg, nl))
    crosscorrs = np.zeros((n_zw, n_zg, nl))

    w_ints = np.zeros((n_zw, nl, nk))
    g_ints = np.zeros((n_zg, nl, nk))
    for i in range(n_zw):
        w_ints[i] = np.load("chiintegrals/sbin{}of{}_zfrom{}to{}_10.npy"\
                .format(i+1, n_zw, int(zmin*10), int(zmax*10)))
        autocorr_w[i] = Correlation(GW_auto=True, GW_Integrals=w_ints[i])

    for i in range(n_zg):
        g_ints[i] = np.load("chiintegrals/gbin{}of{}_zfrom{}to{}_10.npy"\
                .format(i+1, n_zg, int(zmin*10), int(zmax*10)))
        autocorr_g[i] = Correlation(galaxy_auto=True, Galaxy_Integrals=g_ints[i])

    for i in range(n_zw):
        for j in range(n_zg):
            crosscorrs[i, j] = Correlation(cross_corr=True, \
                                           GW_Integrals=w_ints[i], \
                                           Galaxy_Integrals=g_ints[j])
    np.save("correlations_final/w_autocorr_6w12g_z3to15_l0to100", autocorr_w)
    np.save("correlations_final/g_autocorr_6w12g_z3to15_l0to100", autocorr_g)
    np.save("correlations_final/crosscorr_6w12g_z3to15_l0to100", crosscorrs)
#Correlations_load_save()



def correlations_save2():
    nl = 101
    autocorr_w = np.zeros((n_zw, nl))
    autocorr_g = np.zeros((n_zg, nl))

    w_ints = np.zeros((n_zw, nl, nk))
    g_ints = np.zeros((n_zg, nl, nk))
    for i in range(n_zw):
        w_ints[i] = np.load("chiintegrals/sbin{}of{}_zfrom{}to{}_10.npy"\
                .format(i+1, n_zw, int(zmin*10), int(zmax*10)))
        autocorr_w[i] = Correlation(GW_auto=True, GW_Integrals=w_ints[i])
        np.save("correlations_final/w{}".format(i+1) \
              + "_w{}".format(i+1) \
              + "_z{}to{}_l{}to{}.npy".format(int(zmin*10), int(zmax*10), \
                0, 100), autocorr_w[i])


    for i in range(n_zg):
        g_ints[i] = np.load("chiintegrals/gbin{}of{}_zfrom{}to{}_10.npy"\
                .format(i+1, n_zg, int(zmin*10), int(zmax*10)))
        autocorr_g[i] = Correlation(galaxy_auto=True, Galaxy_Integrals=g_ints[i])
        np.save("correlations_final/g{}".format(i+1) \
              + "_g{}".format(i+1) \
              + "_z{}to{}_l{}to{}.npy".format(int(zmin*10), int(zmax*10), \
                0, 100), autocorr_g[i])

    for i in range(n_zw):
        for j in range(n_zg):
            crosscorrs = Correlation(cross_corr=True, \
                                           GW_Integrals=w_ints[i], \
                                           Galaxy_Integrals=g_ints[j])
            np.save("correlations_final/w{}".format(i+1) \
                  + "_g{}".format(j+1) \
                  + "_z{}to{}_l{}to{}.npy".format(int(zmin*10), int(zmax*10), \
                    0, 100), crosscorrs)
#correlations_save2()


def Covariance_optimal(string, ls=np.linspace(0, 100, 101, dtype=int)):
    """
    For a given string of shape "UiVjXmYn", where
    U, V, X, Y = w or g, and i, j, m, n are integers denoting
    the redshift bin of the quantity they are placed after. i.e.
    Ui = w3 means GW-bin number 3.

    For a string like the one above, this function
    should return the covariance between
    C_{U_i V_j} and C_{X_m Y_n}. This is proportional to the objects
    C_{U_i X_m}*C_{V_j Y_n} and C_{U_i Y_n}*C_{V_j X_m}, so these correlation
    should be saved in arrays such that we easily can load
    e.g. C_{U_i Y_n} for "any" U, Y, i, n. If these arrays are saved
    beforehand in labels such as e.g. "w3g2.npy", then this function will be quite simple.

    Function returns the covariance for all l-values given in ls if they are
    pre-saved, i.e. the returned covariance has shape (nl).
    """
    """
    Will assume for now that only the g-numbers can have two digits.
    """

    startindex = 0
    try:
        U, i, V, j, X, m, Y, n = string
    except ValueError:
        U = string[0]
        if string[1:3].isdigit():
            i = string[1:3]
            startindex += 3
        else:
            i = string[1]
            startindex += 2

        newstring = string[startindex:]
        try:
            V, j, X, m, Y, n = newstring
        except ValueError:
            V = newstring[0]
            if newstring[1:3].isdigit():
                j = newstring[1:3]
                startindex += 3
            else:
                j = newstring[1]
                startindex += 2

            newstring = string[startindex:]
            try:
                X, m, Y, n = newstring
            except ValueError:
                X = newstring[0]
                if newstring[1:3].isdigit():
                    m = newstring[1:3]
                    startindex += 3
                else:
                    m = newstring[1]
                    startindex += 2

                newstring = string[startindex:]
                try:
                    Y, n = newstring
                except ValueError:
                    Y = newstring[0]
                    if newstring[1:3].isdigit():
                        n = newstring[1:3]
                    else:
                        n = newstring[1]

    delta_l = 1
    f_sky = 0.5

    """
    Loading
    C_{U_i X_m}, C_{V_j Y_n}, C_{U_i Y_n} and C_{V_j X_m}
    """
    file = "correlations_final/{}{}_{}{}" \
         + "_z{}to{}_l{}to{}.npy".format(int(zmin*10), int(zmax*10), 0, 100)
    try:
        Cim = np.load(file.format(U, i, X, m))
    except FileNotFoundError:
        try:
            Cim = np.load(file.format(X, m, U, i))
        except FileNotFoundError:
            Cim = 0
    try:
        Cjn = np.load(file.format(V, j, Y, n))
    except FileNotFoundError:
        try:
            Cjn = np.load(file.format(Y, n, V, j))
        except FileNotFoundError:
            Cjn = 0
    try:
        Cin = np.load(file.format(U, i, Y, n))
    except FileNotFoundError:
        try:
            Cin = np.load(file.format(Y, n, U, i))
        except FileNotFoundError:
            Cin = 0
    try:
        Cjm = np.load(file.format(V, j, X, m))
    except FileNotFoundError:
        try:
            Cjm = np.load(file.format(X, m, V, j))
        except FileNotFoundError:
            Cjm = 0
    """
    The try-excepts either switches w and g due to symmetry,
    or sets autocorrelations to zero when evaluated at different bins.
    """

    letterpairs = [[U, X], [V, Y], [U, Y], [V, X]]
    numberpairs = [[i, m], [j, n], [i, n], [j, m]]
    cross_corrs = [Cim, Cjn, Cin, Cjm]

    from number_dens3 import Number_dens
    for k in range(len(letterpairs)):
        letters = letterpairs[k]
        numbers = numberpairs[k]
        corrs   = cross_corrs[k]
        if letters[0] == letters[1] and numbers[0] == numbers[1]:
            zbin = bin_dict[letters[0]][int(numbers[0])-1] #-1 as indexing starts at 0
            I = Number_dens(zbin)
            if letters[0] == "w":
                n_shot = I.n_av()
            else: # letters[0] == "g"
                n_shot = I.nav_g()
            shot_noise = 1/n_shot
            cross_corrs[k] += shot_noise
    cov = cross_corrs[0]*cross_corrs[1] + cross_corrs[2]*cross_corrs[3]
    cov *= 1/((2*ls + 1)*delta_l*f_sky)
    return cov
#print(np.shape(Covariance_optimal("w1g12g2g2")))


def Cov_matrix():
    nl = 101

    cont1 = ["w{}w{}".format(i, i) for i in range(1, n_zw+1)]
    cont2 = ["g{}g{}".format(i, i) for i in range(1, n_zg+1)]
    Cvector = cont1 + cont2
    for i in range(1, n_zg+1):
        for j in range(1, n_zw+1):
            Cvector.append("w{}g{}".format(j, i))
    Cvector = np.array(Cvector, dtype=object)
    Cvec1 = Cvector.reshape((90, 1))
    Cvec2 = Cvector.reshape((1, 90))
    Cmat = np.add(Cvec1, Cvec2)

    C_matrix = np.zeros((nl, 90, 90))
    for i in range(90):
        #print(i)
        for j in range(90):
            C_matrix[:, i, j] = Covariance_optimal(Cmat[i, j])
    #np.save("correlations_final/covariancematrix.npy", C_matrix)
    return C_matrix
#Cov_matrix()







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






"""
def Correlations_load_save():
    nl = 101
    autocorr_w = np.zeros((n_zw, nl))
    autocorr_g = np.zeros((n_zg, nl))
    crosscorrs = np.zeros((n_zw, n_zg, nl))

    w_ints = np.zeros((n_zw, nl, nk))
    g_ints = np.zeros((n_zg, nl, nk))
    for i in range(n_zw):
        w_ints[i] = np.load("chiintegrals/sbin{}of{}_zfrom{}to{}_10.npy"\
                .format(i+1, n_zw, int(zmin*10), int(zmax*10)))
        autocorr_w[i] = Correlation(GW_auto=True, GW_Integrals=w_ints[i])

    for i in range(n_zg):
        g_ints[i] = np.load("chiintegrals/gbin{}of{}_zfrom{}to{}_10.npy"\
                .format(i+1, n_zg, int(zmin*10), int(zmax*10)))
        autocorr_g[i] = Correlation(galaxy_auto=True, Galaxy_Integrals=g_ints[i])

    for i in range(n_zw):
        for j in range(n_zg):
            crosscorrs[i, j] = Correlation(cross_corr=True, \
                                           GW_Integrals=w_ints[i], \
                                           Galaxy_Integrals=g_ints[j])
    np.save("correlations_final/w_autocorr_6w12g_z3to15_l0to100", autocorr_w)
    np.save("correlations_final/g_autocorr_6w12g_z3to15_l0to100", autocorr_g)
    np.save("correlations_final/crosscorr_6w12g_z3to15_l0to100", crosscorrs)
#Correlations_load_save()
"""








#
