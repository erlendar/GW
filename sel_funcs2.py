from cosmo1 import Cosmo
import scipy.special as spec
import numpy as np

class Sel_funcs(Cosmo):
    """
    Takes in a bin z_i = (z_min, ..., z, ..., z_max) and
    computes the selection functions associated to this bin.

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

        self.sigma_lnD = 0.05 # Dispersion of distance estimate

    def x(self, D_obs, z):
        """
        Evaluates x-function (unitless)
        """
        return np.log(D_obs/self.D(z))/(np.sqrt(2)*self.sigma_lnD)

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
