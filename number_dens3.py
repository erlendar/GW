from sel_funcs2 import Sel_funcs
import scipy.integrate as integrate
import numpy as np

class Number_dens(Sel_funcs):
    """
    Takes in a bin z_i = (z_min, ..., z, ..., z_max) and
    calculates the number densities associated to this bin.
    """
    def __init__(self, z_i):
        super().__init__(z_i)

    def ng_av(self):
        """
        Returns the 3D comoving number density of the
        spectroscopic galaxy sample in units 1/Mpc**3
        """
        n = 10**(-3)*self.h**3
        return n

    def nav_GW(self,z):
        """
        Returns the (angular-)average 3D number density field of GW sources
        in 1/Mpc**3
        """
        Tn_dot = 3*10**(-6)*self.h**3 # 1/Mpc**3
        # Assumed value of GW merger-rate times duration of observation
        n = Tn_dot/(1+z)
        return n

    def n_av(self, c_M=0):
        """
        Returns the average projected number density of GW sources in
        the i-th bin (unitless)
        """
        integrand = lambda z: self.chi(z)**2/self.H(z)*self.S(z, c_M)*\
                              self.nav_GW(z)*self.c/1000
        n = integrate.quad(integrand, 0, np.inf)[0]
        return n

    def nav_g(self):
        """
        Returns the average projected number density of galaxies in
        the i-th bin (unitless)
        """
        """
        integrand = lambda z: self.chi(z)**2/self.H(z)*self.ng_av()*\
                              np.heaviside(z - self.z_min, 0)*\
                              np.heaviside(self.z_max - z, 0)*self.c/1000
        n = np.sqrt(integrate.quad(integrand, self.z_min, self.z_max)[0])
        """
        zs = np.linspace(self.z_min, self.z_max, 200)
        integrand = lambda z: self.chi(z)**2/self.H(z)*self.ng_av()*self.c/1000
        n = (self.integration(zs, integrand(zs)))
        #Removed sqrt(n)
        #Should the sqrt be there? <-> do we interpret (16) in Oguri literally

        #return np.sqrt(integrate.quad(integrand, 0, np.inf)[0])

        # integrate-function dislikes the discontinuities
        # following from the Heaviside functions
        # (Integral really from 0 to inf)
        return n
