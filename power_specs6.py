from class5 import CLASS
import numpy as np

class Power_specs(CLASS):
    def __init__(self, z_i, z_j, zs):
        super().__init__(z_i)
        self.J = CLASS(z_j)
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
                              self.b_GW(z)*self.b_g(z)*self.P_m(l,z)*1e3/self.c*self.h**3
        return self.integration(zs, integrand(zs))

    def Ctg(self, l=100, n=20):
        """
        Returns the Ctg power spectrum
        Unitless
        """

        eps = 1e-12
        zs = self.zs
        Wt_i = self.Wt
        Wg_j = self.J.Wg
        Wk = self.Wk
        zint = lambda z: np.linspace(eps,z-eps,n)
        # Add small epsilon to avoid z' = z and z' = 0

        integrand1 = lambda z,zprime: Wg_j(zprime)*Wk(zprime,z)*self.H(zprime)/\
                                   self.chi(zprime)**2*self.b_g(z)*self.P_m(l,zprime)*1e3/self.c*self.h**3
        integrand2 = lambda z: Wt_i(z)*self.integration(zint(z),integrand1(z,zint(z)))

        return self.integration(zs, integrand2(zs))

    def Cwg(self,l=100,n=20):
        """
        Returns the cross-correlation power spectrum
        Unitless
        """
        return self.Csg(l) + self.Ctg(l,n)
