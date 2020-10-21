import numpy as np
import matplotlib.pyplot as plt
import time
from power_specs6 import Power_specs



def PlotCtg():
    t1 = time.time()
    delta_z = 0.1
    l = 100
    n = 11   # Number of datapoints
    nn = 10  # Number of points in domain of integration
    nnn = 10 # Number of points for zprime integral in Ctg
    z_g = np.linspace(0.15,1.3,n) # central redshifts of spectroscopic galaxy sample

    z_i = np.arange(0.9,1.1,0.1) # GW redshifts
    #zs = np.linspace(z_g[0],z_g[-1],52) # Domain of integration
    zs = np.linspace(z_g[0] - delta_z/2,z_g[-1] + delta_z/2,nn) # Domain of integration
    # Our integrals will be zero outside of the galaxy readshifts in our bin
    # So zs might as well go from z_g[0] to z_g[-1]

    P = z_g.copy()
    i = 0
    for zg in z_g:
        print ("\nWorking on datapoint {} out of {} in total\n".format(i+1,n))
        #z_j = np.arange(zg - delta_z/2,zg + delta_z/2,delta_z) # Galaxy redshifts
        z_j = np.linspace(zg - delta_z/2,zg + delta_z/2,2) # Galaxy redshifts
        I = Power_specs(z_i,z_j,zs)
        P[i] = I.Ctg(l,nnn)
        i += 1

    print("\nTime spent: {} minutes\n".format((time.time()-t1)/60))
    print(z_g)
    print(P)
    plt.plot(z_g,P,"--ro")
    plt.xlabel("z_g")
    plt.ylabel("Ctg(l=100)")
    plt.yscale("log")
    #plt.axis([0.05,1.3, 1e-8,1e-4])
    plt.axis([0.05,1.3, 1e-12,1e-2])
    plt.savefig("Ctg10th.pdf")
    plt.show()
    return None

#PlotCtg()




def PlotCsg():
    t1 = time.time()
    delta_z = 0.1
    l = 100
    n = 41 # must be odd number, number of points in z_g
    nn = 100 # Number of points in domain of integration
    z_g = np.linspace(0.7,1.3,n) # central redshifts of spectroscopic galaxy sample

    z_i = np.arange(0.9,1.1,delta_z) # GW redshifts
    #zs = np.linspace(z_g[0],z_g[-1],52) # Domain of integration
    zs = np.linspace(z_g[0] - delta_z/2,z_g[-1] + delta_z/2,nn) # Domain of integration
    # Our integrals will be zero outside of the galaxy readshifts in our bin
    # So zs might as well go from z_g[0] to z_g[-1]
    # Class does not run more than 52 z-arguments at once? - split up!

    P = z_g.copy()
    i = 0
    for zg in z_g:
        print ("\nWorking on datapoint {} out of {} in total\n".format(i+1,n))
        z_j = np.linspace(zg - delta_z/2,zg + delta_z/2,2) # Galaxy redshifts
        I = Power_specs(z_i,z_j,zs)
        P[i] = I.Csg(l)
        i += 1

    print("\nTime spent: {} minutes\n".format((time.time()-t1)/60))
    print(z_g)
    print(P)
    plt.plot(z_g,P,"--o")
    plt.xlabel("z_g")
    plt.ylabel("Csg(l=100)")
    plt.yscale("log")
    #plt.axis([0.1,1.3, 1e-8,1e-4])
    plt.axis([0.05,1.3, 1e-12,1e-2])
    plt.savefig("Csg14th.pdf")
    plt.show()
    return None

#PlotCsg()


from sel_funcs2 import Sel_funcs
def test_S():
    """
    Check if S looks like a selection function
    """
    z_i = np.linspace(1.05,1.1,10)
    I = Sel_funcs(z_i)
    z = np.linspace(0.8,1.2,100)
    f = I.S(z)

    plt.plot(z,f)
    plt.xlabel("z")
    plt.ylabel("S")
    plt.savefig("Ssel.pdf")
    plt.show()
    return None

def test_T():
    """
    Check if S looks like a selection function
    """
    z_i = np.linspace(1.05,1.1,10)
    I = Sel_funcs(z_i)
    z = np.linspace(0.8,1.2,100)
    f = I.T(z)

    plt.plot(z,f)
    plt.xlabel("z")
    plt.ylabel("T")
    plt.savefig("Tsel.pdf")
    plt.show()
    return None

#test_S()
#test_T()
