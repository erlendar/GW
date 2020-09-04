import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as spec
import os



class GW():
    def __init__(self):
        # 1 Mpc = 3.08567758*1e22 m
        self.omega_m  = 0.308
        self.omega_de = 0.692
        self.omega_b  = 0.04867
        self.h        = 0.6763
        self.H0       = 100*self.h                        # km/(s*Mpc)
        self.G        = 6.67408*1e-11                     # m**3*kg**(-1)*s**(-2)
        self.H0_SI = self.H0*1000/(3.08567758*1e22)       # 1/s
        self.rho_c0   = 3*self.H0_SI**2/(8*np.pi*self.G)  # kg/m**3
        self.c        = 299792458                         # m/s

        self.ng_av     = 10**(-3)*self.h**3               # 1/Mpc**3
        self.sigma_lnD = 0.05
        self.Tn_dot    = 3*10**(-6)*self.h**3             # 1/Mpc**3

        self.b_GW = 0.8   # Temporarily, should be evolving with D(z)
        self.b_g  = 0.95  # Temporarily, ---


    def H(self,z):
        """
        Returns the Hubble constant in km/(s*Mpc)
        """
        return self.H0*np.sqrt(self.omega_m*(1+z)**3 + self.omega_de)

    def chi(self,z):
        """
        Returns the comoving angular diameter distance in Mpc
        """
        c = self.c/1000 # km/s

        if len(np.shape(z)) >= 2:
            a = z.copy()
            for i in range(len(a)):
                for j in range(len(a[0])):
                    a[i,j] = integrate.quad(lambda zz: c/self.H(zz), 0, z[i,j])[0]
            return a

        if type(z) != np.ndarray:
            z = np.array([z])    # NEEDS FIXING, z may not be single number
        a = z.copy()
        for i in range(len(a)):
            a[i] = integrate.quad(lambda zz: c/self.H(zz), 0, z[i])[0]
        return a

    def D(self,z):
        """
        Returns the luminosity distance in Mpc
        """
        return self.chi(z)*(1+z)

    def D_A(self,z):
        """
        Returns the angular diameter distance in Mpc
        """
        return self.chi(z)/(1+z)

    def x(self, D_obs, z):
        """
        Evaluates x-function (unitless)
        """
        return np.log(D_obs/self.D(z))/(np.sqrt(2)*self.sigma_lnD)

    def nav_GW(self,z):
        """
        Returns the (angular-)average 3D number density field of GW sources
        in 1/Mpc**3
        """
        return self.Tn_dot/(1+z)

    def rhom_av(self,z):
        """
        Returns the mean physical density of the universe at redshift z
        in units kg/m**3
        """
        return self.omega_m*self.rho_c0*(1+z)**3

    def Sigma_crit(self,z,z_s):
        """
        Returns the physical critical surface density at redshift
        z for the source redshift z_s in units kg/(Mpc*m)
        """
        return self.c**2/(4*np.pi*self.G)*self.D_A(z_s)/\
              (self.D_A(z)*self.D_A(np.abs(z_s-z))*(1+z)**2)

    def Wk(self, z, zz):
        """
        Returns the unitless window function W_kappa
        """
        W = self.rhom_av(z)/(self.H(z)*(1+z)*self.Sigma_crit(z,zz))
        return W*(self.c/1000)*(3.08567758*1e22)**2



class GWbins(GW):
    """
    Takes in a bin z_i = (z_min, ..., z, ..., z_max)
    and computes the quantities associated to this
    i-th bin.

    So z_i corresponds to the bin and decides D_min and D_max,
    while the z-variable in the functions below is independent of this
    (simply represents the area over which to evaluate the functions)
    """
    def __init__(self, z_i):
        super().__init__()
        self.z_i = z_i

        self.z_min = z_i[0]
        self.z_max = z_i[-1]
        self.D_min = self.D(self.z_min)
        self.D_max = self.D(self.z_max)

        self.n_av = integrate.quad(self.n_av_integrand, 0, np.inf)[0]
        # Average projected number density of GW sources in i-th bin
        # Unitless

    def n_av_integrand(self,z):
        """
        Returns the (unitless?) integrand
        """
        return self.chi(z)**2/self.H(z)*self.S(z)*\
               self.nav_GW(z)*self.c/1000

    def S(self,z):
        """
        Returns the selection function S
        """
        return 0.5*(spec.erfc(self.x(self.D_min,z)) - \
                    spec.erfc(self.x(self.D_max,z)))

    def T(self,z):
        """
        Returns the selection function T
        """
        return (-np.exp(-self.x(self.D_min,z)**2) + \
                 np.exp(-self.x(self.D_max,z)**2))\
               /(np.sqrt(2*np.pi)*self.sigma_lnD)

    def Ws(self,z):
        """
        Returns the window function W_s (unitless)
        """
        return 1/self.n_av*self.chi(z)**2/self.H(z)*\
               self.S(z)*self.nav_GW(z)*self.c/1000

    def Wt(self,z):
        """
        Returns the window function W_t (unitless)
        """
        return 1/self.n_av*self.chi(z)**2/self.H(z)*\
               self.T(z)*self.nav_GW(z)*self.c/1000

    def Wg(self,z):
        """
        Returns the window function W_g (unitless)
        """
        return 1.0/self.nav_g()*self.chi(z)**2/self.H(z)*self.ng_av*\
                              np.heaviside(z - self.z_min, 0)*\
                              np.heaviside(self.z_max - z, 0)*self.c/1000

    def nav_g(self):
        """
        Returns the average projected number density of galaxies in
        the i-th bin (unitless)
        """
        integrand = lambda z: self.chi(z)**2/self.H(z)*self.ng_av*\
                              np.heaviside(z - self.z_min, 0)*\
                              np.heaviside(self.z_max - z, 0)*self.c/1000
        #return np.sqrt(integrate.quad(integrand, 0, np.inf)[0])
        return np.sqrt(integrate.quad(integrand, self.z_min, self.z_max)[0])
        # integrate-function dislikes the discontinuities
        # following from the Heaviside functions



class Class(GWbins):
    def __init__(self, z_i, z_j, zs):
        super().__init__(z_i)
        self.J = GWbins(z_j)
        self.zs = zs

    def Csg(self, l=100):
        """
        Returns the Csg power spectrum
        Unitless
        """
        zs = self.zs
        Ws_i = self.Ws
        Wg_j = self.J.Wg
        integrand = lambda z: Ws_i(z)*Wg_j(z)*self.H(z)/self.chi(z)**2*\
                              self.b_GW*self.b_g*self.P_m(l,z)*1e3/self.c*self.h**3
        return self.integration(zs, integrand(zs))

    def Ctg(self, l=100):
        """
        Returns the Ctg power spectrum
        Unitless
        """
        n = 20
        zs = self.zs
        Wt_i = self.Wt
        Wg_j = self.J.Wg
        Wk = self.Wk
        zint = lambda z: np.linspace(0.1,z,n)

        integrand1 = lambda z,zprime: Wg_j(zprime)*Wk(zprime,z)*self.H(zprime)/\
                                   self.chi(zprime)**2*self.b_g*self.P_m(l,zprime)*1e3/self.c*self.h**3
        integrand2 = lambda z: Wt_i(z)*self.integration(zint(z),integrand1(z,zint(z)))
        return self.integration(zs, integrand2(zs))

    def Cwg(self,l=100):
        """
        Returns the cross-correlation power spectrum
        Unitless
        """
        return self.Csg(l) + self.Ctg(l)


    def integration(self, x, y):
        """
        Numerically integrate a discrete function y(x) by using Simpson's rule
        """
        if len(np.shape(x)) >= 2:
            return integrate.simps(y,x,axis=0)
        return integrate.simps(y,x)

    def P_m(self, l, zs):
        """
        Read CLASS output file and return the matter power spectrum

        len(zs) should be less than 52 for now
        """
        self.get_Pm(zs)
        wanted_ks = (l+0.5)/self.chi(zs) # 1/Mpc
        wanted_ks *= self.h

        found_Pks = []


        if len(np.shape(zs)) >= 2:
            PPPs = zs.copy()
            for i in range(len(zs)):

                self.get_Pm(zs[i])
                wanted_ks = (l+0.5)/self.chi(zs[i]) # 1/Mpc
                wanted_ks *= self.h

                found_Pks = []

                for j in range(len(zs[0])):
                    ks = []
                    Pks = []
                    infile = open("../../Downloads/class_public-2.9.3/output/mydataz{}_pk.dat".format(i+1), "r")
                    # Indexing starting at 0
                    # Path to be generalized
                    infile.readline()
                    infile.readline()
                    infile.readline()
                    infile.readline()
                    for line in infile:
                        columns = line.strip().split(" ")
                        k = float(columns[0].strip())
                        Pk = float(columns[-1].strip())
                        ks.append(k)
                        Pks.append(Pk)
                    infile.close()

                    ks = np.array(ks)
                    Pks = np.array(Pks)

                    wanted_k = wanted_ks[j]
                    diff = np.abs(ks-wanted_k)
                    ind = np.where(diff == np.min(diff))[0][0]
                    found_Pk = Pks[ind]
                    found_Pks.append(found_Pk)
                    PPPs[i,j] = found_Pk

            return PPPs



        for i in range(len(zs)):
            ks = []
            Pks = []
            infile = open("../../Downloads/class_public-2.9.3/output/mydataz{}_pk.dat".format(i+1), "r")
            # Indexing starting at 0
            # Path to be generalized
            infile.readline()
            infile.readline()
            infile.readline()
            infile.readline()
            for line in infile:
                columns = line.strip().split(" ")
                k = float(columns[0].strip())
                Pk = float(columns[-1].strip())
                ks.append(k)
                Pks.append(Pk)
            infile.close()

            ks = np.array(ks)
            Pks = np.array(Pks)

            wanted_k = wanted_ks[i]
            diff = np.abs(ks-wanted_k)
            ind = np.where(diff == np.min(diff))[0][0]
            found_Pk = Pks[ind]
            found_Pks.append(found_Pk)

        np.array(found_Pks)
        return found_Pks

    def get_Pm(self, zs):
        """
        Run CLASS in terminal to get the matter power spectrum P(k,z)
        """
        s2 = len(zs)*"{}, "
        s2 = s2[:-2] + "\n"
        s1 = "h = {}\n".format(self.h)\
           + "omega_b = {} # baryon density\n".format(self.omega_b)\
           + "Omega_Lambda = {}\n".format(self.omega_de)\
           + "Omega_k = 0. #curvature\n"\
           + "output = mPk\n"\
           + "z_pk = "
        s3 = "root = output/mydata"

        s = s1 + s2 + s3
        s = s.format(*zs) # Example
        #print(s)
        outfile = open("../../Downloads/class_public-2.9.3/myrun.ini", "w")
        outfile.write(s)
        outfile.close()
        # Create input file

        os.system("cd; cd Downloads/class_public-2.9.3; ./class myrun.ini")
        # Run input file in Class from terminal
        # Paths to be generalized
        return None

# INTERPOLATION


def PlotCs():
    delta_z = 0.1
    l = 100
    n = 21 # must be odd number
    z_g = np.linspace(0.3,1.3,n) # central redshifts of spectroscopic galaxy sample

    z_i = np.arange(0.9,1.1,delta_z) # GW redshifts
    zs = np.linspace(z_g[0],z_g[-1],50) # Domain of integration
    # Our integrals will be zero outside of the galaxy readshifts in our bin
    # So zs might as well go from z_g[0] to z_g[-1]
    # Class does not run more than 52 z-arguments at once? - split up!

    P = []
    PP = []
    for zg in z_g:
        z_j = np.arange(zg - delta_z/2,zg + delta_z/2,delta_z) # Galaxy redshifts
        I = Class(z_i,z_j,zs)
        P.append(I.Ctg(l))
        PP.append(I.Csg(l))

    plt.plot(z_g,P,"r*")
    plt.plot(z_g,PP,"bo")
    plt.xlabel("z_g")
    plt.ylabel("C(l=100)")
    plt.yscale("log")
    #plt.axis([z_g[0],z_g[-1], 1e-3,1e2])
    plt.legend(["Ctg","Csg"])
    plt.savefig("Cs9th.pdf")
    plt.show()
    return None

PlotCs()




def PlotCsg():
    delta_z = 0.1
    l = 100
    n = 11 # must be odd number
    z_g = np.linspace(0.7,1.3,n) # central redshifts of spectroscopic galaxy sample

    z_i = np.arange(0.9,1.1,delta_z) # GW redshifts
    zs = np.linspace(z_g[0],z_g[-1],52) # Domain of integration
    # Our integrals will be zero outside of the galaxy readshifts in our bin
    # So zs might as well go from z_g[0] to z_g[-1]
    # Class does not run more than 52 z-arguments at once? - split up!

    P = []
    for zg in z_g:
        z_j = np.arange(zg - delta_z/2,zg + delta_z/2,delta_z) # Galaxy redshifts
        I = Class(z_i,z_j,zs)
        P.append(I.Csg(l))

    plt.plot(z_g,P,"o")
    plt.xlabel("z_g")
    plt.ylabel("Csg(l=100)")
    plt.yscale("log")
    #plt.axis([z_g[0],z_g[-1], 1e-8,1e1])
    plt.savefig("Csg6th.pdf")
    plt.show()
    return None

#PlotCsg()


def plotW():
    """
    See what the window functions look like
    """
    z_i = np.linspace(0.9,1.1,10)
    I = GWbins(z_i)
    z = np.linspace(0.8,1.2,100)
    f = I.Ws(z)
    g = I.Wt(z)
    h = I.Wg(z)

    plt.plot(z,f)
    plt.plot(z,g)
    plt.plot(z,h)
    plt.xlabel("z")
    #plt.ylabel("W")
    plt.legend(["Ws", "Wt", "Wg"])
    plt.show()
    return None

#plotW()


def test_S():
    """
    Check if S looks like a selection function
    """
    z_i = np.linspace(1.05,1.1,10)
    I = GWbins(z_i)
    z = np.linspace(0.8,1.2,100)
    f = I.S(z)
    #g = I.T(z)

    plt.plot(z,f)
    #plt.plot(z,g)
    plt.xlabel("z")
    plt.ylabel("S")
    plt.show()
    return None

#test_S()
