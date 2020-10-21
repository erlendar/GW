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

    def integration(self, x, y):
        """
        Numerically integrate a discrete function y(x) by using Simpson's rule
        """
        #y[np.where(np.isnan(y))] = 0 # NaN-values should not contribute
        val = integrate.simps(y,x,axis=0)
        return val
        # Assuming integration is over index 0 - axis

    def H(self,z):
        """
        Returns the Hubble constant in km/(s*Mpc)
        """
        return self.H0*np.sqrt(self.omega_m*(1+z)**3 + self.omega_de)

    def chi(self,z):
        """
        Returns the comoving angular diameter distance ( = comoving distance) in Mpc
        """
        c = self.c/1000 # km/s
        n = 200
        # Method slow if n gets much larger

        zz = np.linspace(0,z,n)
        b = self.integration(zz, c/self.H(zz))
        return b

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

    def D_AA(self,z1,z2):
        """
        Returns the angular diameter distance between redshifts
        z1 and z2 in a flat universe (in Mpc)
        """
        #DOUBLE CHECK
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
        Dzz = self.D_AA(z,z_s)
        #Dzz[np.where(Dzz==0)] = None
        # Avoid division by zero
        return self.c**2/(4*np.pi*self.G)*self.D_A(z_s)/\
              (self.D_A(z)*Dzz)

    def f(self, z):
        """
        Returns the dimensionless linear growth rate
        """
        omega = self.omega_m*(1+z)**3
        return omega**0.6 + self.omega_de/70*(1+omega/2) # Dodelson approx
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
