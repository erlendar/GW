from cosmo1 import Cosmo
import scipy.special as spec
import numpy as np

class Sel_funcs(Cosmo):
    """
    PascalCase
    camelCase
    snake_case

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
        self.D_min = self.D(self.z_min, c_M=0)
        self.D_max = self.D(self.z_max, c_M=0)
        # Luminosity bin corresponding to a redshift [z_min, z_max] in FRW-universe

        self.sigma_lnD = 0.05 # Dispersion of distance estimate

    def x(self, D_obs, z, c_M=0):
        """
        Evaluates x-function (unitless)
        """
        return np.log(D_obs/self.D(z, c_M))/(np.sqrt(2)*self.sigma_lnD)

    def S(self, z, c_M=0):
        """
        Returns the selection function S
        """
        return 0.5*(spec.erfc(self.x(self.D_min,z, c_M)) - \
                    spec.erfc(self.x(self.D_max,z, c_M)))

    def T(self, z, c_M=0):
        """
        Returns the selection function T
        """
        return (-np.exp(-self.x(self.D_min,z, c_M)**2) + \
                 np.exp(-self.x(self.D_max,z, c_M)**2))\
               /(np.sqrt(2*np.pi)*self.sigma_lnD)

    def U(self,z, c_M=0):
        """
        Returns the selection function U
        """
        return 1/2*1/(np.sqrt(2*np.pi)*self.sigma_lnD**2)*(self.sigma_lnD\
             * (np.exp(-self.x(self.D_max,z, c_M)**2) - np.exp(-self.x(self.D_min,z, c_M)**2)) \
             -  np.sqrt(2)*(self.x(self.D_max,z, c_M)*np.exp(-self.x(self.D_max,z, c_M)**2) \
             -  self.x(self.D_min,z, c_M)*np.exp(-self.x(self.D_min,z, c_M)**2)))
