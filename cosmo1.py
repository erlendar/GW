import numpy as np
import scipy.integrate as integrate


class Cosmo():
    """
    Takes cosmological parameters as input, and contains different
    cosmological quantities (distance measures, Hubble parameter etc.)
    as functions of redshift z.

    All quantities dependent on our cosmological model.
    """
    def __init__(self):
        # 1 Mpc = 3.08567758*1e22 m
        self.omega_m  = 0.308
        self.omega_de = 0.692
        self.omega_b  = 0.04867
        self.h        = 0.6763
        self.H0       = 100*self.h                        # km/(s*Mpc)
        self.G        = 6.67408*1e-11                     # m**3*kg**(-1)*s**(-2)
        self.H0_SI    = self.H0*1000/(3.08567758*1e22)    # 1/s
        self.rho_c0   = 3*self.H0_SI**2/(8*np.pi*self.G)  # kg/m**3
        self.c        = 299792458                         # m/s

        self.b_g1 = 1; self.b_g2 = 1
        self.b_w1 = 1; self.b_w2 = 1

    def integration(self, x, y, ax=0):
        """
        Numerically integrate a discrete function y(x) (over
        first axis) by using Simpson's rule
        """
        val = integrate.simps(y,x,axis=ax)
        # axis=0 ensures that we integrate each of the
        # columns separately, if y and x are matrices
        # Correspondingly, axis=1 would integrate the rows

        # i.e. for axis=0 we treat each column as a vector
        # and integrate over that vector
        return val

    def H(self,z):
        """
        Returns the Hubble constant in km/(s*Mpc)
        """
        return self.H0*np.sqrt(self.omega_m*(1+z)**3 + self.omega_de)

    def chi(self,z,n=400):
        """
        Returns the comoving angular diameter distance ( = comoving distance) in Mpc
        """
        c = self.c/1000 # km/s

        zz = np.linspace(0,z,n)
        b = self.integration(zz, c/self.H(zz))
        return b

    def omega_lambda(self, z):
        """
        Returns Omega_Lambda as a function
        of the redshift (unitless)
        """
        return self.omega_de*self.H0**2/self.H(z)**2

    def alpha_M(self, z, c_M = 0):
        """
        Returns the alpha_M parameter (unitless)
        """
        alpha = c_M*self.omega_lambda(z)/self.omega_de
        return alpha

    def D_ratio(self, z, c_M = 0):
        """
        Returns the delta_D parameter (unitless)
        """
        n = 200
        zprime = np.linspace(1e-10, z, n)
        integrand = self.alpha_M(zprime, c_M)/(1 + zprime)
        delta = 0.5*self.integration(zprime, integrand, ax=0)
        return np.exp(delta)

    def D(self, z, c_M=0):
        """
        Returns the luminosity distance in Mpc.
        This is the standard FRW-D_L for c_M=0,
        else it returns the effective measured
        luminosity distance from a GW signal
        in a MG-theory.
        """
        return self.chi(z) * (1 + z) * self.D_ratio(z, c_M)

    def D_A(self,z):
        """
        Returns the angular diameter distance in Mpc
        """
        return self.chi(z) / (1 + z)

    def D_AA(self, z1, z2):
        """
        Returns the angular diameter distance between redshifts
        z1 and z2 in a flat universe (in Mpc)
        """
        val = (self.chi(z2) - self.chi(z1))/(1+z2)
        return val

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
        val = self.c**2/(4*np.pi*self.G)*self.D_A(z_s)/\
              (self.D_A(z)*self.D_AA(z,z_s))
        return val

    def f(self, z):
        """
        Returns the dimensionless linear growth rate
        """
        omega = self.omega_m#*(1+z)**3
        omega_de = self.omega_de
        return omega**0.6 + omega_de/70*(1+omega/2) # Dodelson approx
        #return omega**0.55

    def b_g(self, z):
        """
        Returns the bias parameter for spectroscopic galaxies
        """
        return self.b_g1 + self.b_g2/self.f(z)

    def b_GW(self, z):
        """
        Returns the bias parameter for GW sources
        """
        return self.b_w1 + self.b_w2/self.f(z)
