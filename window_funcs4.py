from number_dens3 import Number_dens
import numpy as np

class Window_funcs(Number_dens):
    """
    Takes in a bin z_i = (z_min, ..., z, ..., z_max) and
    calculates the window functions associated to this bin.
    """
    def __init__(self, z_i):
        super().__init__(z_i)

    def Ws(self,z):
        """
        Returns the window function W_s (unitless)
        """
        W = 1/self.n_av()*self.chi(z)**2/self.H(z)*\
            self.S(z)*self.nav_GW(z)*self.c/1000
        return W

    def Wt(self,z):
        """
        Returns the window function W_t (unitless)
        """
        W = 1/self.n_av()*self.chi(z)**2/self.H(z)*\
            self.T(z)*self.nav_GW(z)*self.c/1000
        return W

    def Wg(self,z):
        """
        Returns the window function W_g (unitless)
        """
        W = 1.0/self.nav_g()*self.chi(z)**2/self.H(z)*self.ng_av()*\
            np.heaviside(z - self.z_min, 0)*\
            np.heaviside(self.z_max - z, 0)*self.c/1000
        return W

    def Wk(self, z, zz):
        """
        Returns the unitless window function W_kappa (unitless)
        """
        W = self.rhom_av(z)/(self.H(z)*(1+z)*self.Sigma_crit(z,zz))
        W *= (self.c/1000)*(3.08567758*1e22)**2 # Get the units right
        return W
