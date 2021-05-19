from number_dens3 import Number_dens
import numpy as np

class Window_funcs(Number_dens):
    """
    Takes in a bin z_i = (z_min, ..., z, ..., z_max) and
    calculates the window functions associated to this bin.
    """
    def __init__(self, z_i):
        super().__init__(z_i)

    def Ws(self, z, c_M=0):
        """
        Returns the window function W_s (unitless)
        """
        W = 1/self.n_av(c_M)*self.chi(z)**2/self.H(z)*\
            self.S(z, c_M)*self.nav_GW(z)*self.c/1000
        return W

    def Wt(self, z, c_M=0):
        """
        Returns the window function W_t (unitless)
        """
        W = 1/self.n_av(c_M)*self.chi(z)**2/self.H(z)*\
            self.T(z, c_M)*self.nav_GW(z)*self.c/1000
        return W

    def Wg(self,z):
        """
        Returns the window function W_g (unitless)
        """
        W = 1.0/self.nav_g()*self.chi(z)**2/self.H(z)*self.ng_av()*\
            np.heaviside(z - self.z_min, 1)*\
            np.heaviside(self.z_max - z, 1)*self.c/1000
        return W

    def Wk(self, z, zz):
        """
        Returns the unitless window function W_kappa (unitless)
        """
        W = self.rhom_av(z)/(self.H(z)*(1+z)*self.Sigma_crit(z,zz))
        W *= (self.c/1000)*(3.08567758*1e22)**2 # Get the units right
        return W

    def Wu(self,z, c_M=0):
        """
        Returns the window function W_u (unitless)
        """
        W = 1/self.n_av(c_M)*self.chi(z)**2/self.H(z)*\
            self.U(z, c_M)*self.nav_GW(z)*self.c/1000
        return W
