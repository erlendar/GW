import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from class5 import CLASS

def GetPm(nz=int(1e2)):

    nk = 114 # This is determined by CLASS (?)
    #nz = int(2e3)

    k = np.geomspace(1e-6, 2, nk)
    z = np.linspace(0.01, 1.5, nz)
    # These quantities decides the points
    # we interpolate from

    def read_output(txt): # Copy from CLASS-class
        """
        Reads the CLASS Pk - output file "txt"
        and returns k values and P(k) values
        """
        k = []
        P = []
        infile = open(txt, "r")
        infile.readline()
        infile.readline()
        infile.readline()
        infile.readline()
        for line in infile:
            columns = line.strip().split(" ")
            k.append(float(columns[0].strip()))
            P.append(float(columns[-1].strip()))
        infile.close()
        return np.array(k), np.array(P)

    def RunCLASS(z):
        z_i = np.linspace(0.9,1.1,100)
        I = CLASS(z_i) # Bin not important for this matter
        I.get_Pm(z)
        return None

    def Pmsave(z):
        """
        For a given array z this function finds the
        corresponding (nk x nz) - matrix P(k, z)
        """
        max_z = 40
        runs = int(np.ceil(len(z)/max_z))
        # We will only run 40 values of z simultaneously in CLASS
        #P = z.copy()
        P = np.zeros((nk,len(z)))
        print("CALCULATING THE MATTER POWER SPECTRUM ...")
        for n in range(runs):
            print("Working on run {} out of {}".format(n+1,runs))
            low_ind = max_z*n
            if max_z*(n+1) >= len(z):
                high_ind = None # Slicing x[i:None] gives all elements except the i first
            else:
                high_ind = max_z*(n+1)
            z_ = z[low_ind:high_ind]
            RunCLASS(z_)
            Pk1 = np.zeros((len(z_),nk))
            for j in range(len(z_)):
                FileToRead = "../../../Downloads/class_public-2.9.3"\
                           + "/output/mydataz{}_pk.dat".format(j+1)
                # Indexing starting at 0
                # Path to be generalized
                ks, Pks = read_output(FileToRead)
                Pk1[j,:] = Pks
            np.transpose(P[:,low_ind:high_ind])[:] = Pk1
        np.save("karray.npy", ks)
        np.save("zarray.npy", z)
        np.save("Parray.npy", P)
        return None

    Pmsave(z)
    print("... MATTER POWER SPECTRUM SAVED")
    return None


def GetDat():
    k = np.load("karray.npy")
    z = np.load("zarray.npy")
    P = np.load("Parray.npy")
    return k, z, P

def intpol():
    k, z, P = GetDat()
    points = []
    vals = []
    for i_k in range(len(k)):
        for i_z in range(len(z)):
            points.append((k[i_k],z[i_z]))
            vals.append(P[i_k,i_z])
    points = np.array(points)
    vals = np.array(vals)
    Pmfunc = LinearNDInterpolator(points, vals)
    return Pmfunc


def P_m_equaltime(k, z):
    """
    Returns the matter power spectrum P_m(k, z)
    for a given wavenumber k and redshift z.

    If k = (k_1, ..., k_n) and z = (z_1, ..., z_m)
    are 1D - arrays, the returned power spectrum is
    a matrix with rows (P_m(k_i, z_1), ..., P_m(k_i, z_m))
    and columns (P_m(k_1, z_j), ..., P_m(k_n, z_j)).
    (i.e. shape is (len(k), len(z)).)

    The inputs z and k cannot have dimension larger than 1.
    The function also assumes that all input of
    dimension 1 are numpy arrays.
    """
    f = intpol()
    if type(k) == np.ndarray and type(z) == np.ndarray:
        nk = len(k)
        if len(np.shape(z)) == 1:    # z a 1D-array
            nz = len(z)
            P = np.zeros((nk, nz))
            for i in range(nk):
                P[i, :] = f(k[i], z)
        elif len(np.shape(z)) == 2:  # z a 2D-array
            nzrow = np.shape(z)[0]
            nzcol = np.shape(z)[1]
            P = np.zeros((nk, nzrow, nzcol))
            for i in range(nk):
                for j in range(nzrow):
                    P[i, j, :] = f(k[i], z[j, :])
        return P
    else:
        return f(k, z)





def plot_interpol():
    k, z, PP = GetDat()
    #plt.loglog(k,P_m_equaltime(k,z[-13]),"r--")
    #plt.loglog(k,PP[:,-13],"b.")
    plt.loglog(z,P_m_equaltime(k[-4],z),"r--")
    plt.loglog(z,PP[-4,:],"b.")
    #plt.xlabel("k [h/Mpc]")
    plt.xlabel("z")
    plt.ylabel("P [(Mpc/h)^3]")
    plt.savefig("Pm3")
    plt.show()


"""
def P_m(k, z, z_prime):

    z_dim = len(np.shape(z))
    z_prime_dim = len(np.shape(z_prime))
    # Dim is 0 for numbers, 1 for arrays and 2 for matrices
    if z_dim == 0 or z_prime_dim == 0:
        fac1 = np.transpose(P_m_equaltime(k, z))
        fac2 = np.transpose(P_m_equaltime(k, z_prime))
        P = np.transpose(np.sqrt(fac1*fac2))

    elif z_dim == 1 and z_prime_dim == 1:
        nk = len(k)
        nz = len(z)
        nz_prime = len(z_prime)
        P = np.zeros((nk, nz, nz_prime))
        for i in range(nz):
            P[:, i, :] = np.sqrt(P_m_equaltime(k, z_prime))
        Ptransp = np.transpose(P, (2,0,1))*np.sqrt(P_m_equaltime(k, z))
        P = np.transpose(Ptransp, (1,2,0))
    elif z_dim == 2 and z_prime_dim == 1:
        # In this case P_m_equaltime(k, z) has
        # the shape (nk, nzrow, nzcol)
        nk = len(k)
        nz_prime = len(z_prime)
        nzrow = np.shape(z)[0]
        nzcol = np.shape(z)[1]
        P = np.zeros((nk, nzrow, nzcol, nz_prime))
        for i in range(nz_prime):
            P[:, :, :, i] = np.sqrt(P_m_equaltime(k, z))
        Ptransp = np.transpose(P, (1, 2, 0, 3))\
                * np.sqrt(P_m_equaltime(k, z_prime))
        P = np.transpose(Ptransp, (2, 0, 1, 3))
    elif z_dim == 1 and z_prime_dim == 2:
        # In this case P_m_equaltime(k, z_prime) has
        # the shape (nk, nz_primerow, nz_primecol)
        nk = len(k)
        nz = len(z)
        nz_primerow = np.shape(z_prime)[0]
        nz_primecol = np.shape(z_prime)[1]
        P = np.zeros((nk, nz, nz_primerow, nz_primecol))
        for i in range(nz):
            P[:, i, :, :] = np.sqrt(P_m_equaltime(k, z_prime))
        Ptransp = np.transpose(P, (2, 3, 0, 1))\
                * np.sqrt(P_m_equaltime(k, z))
        P = np.transpose(Ptransp, (2, 3, 0, 1))
    else:
        P = np.sqrt(P_m_equaltime(k, z)*P_m_equaltime(k, z_prime))
    return P
"""




#if __name__ == "__main__":
    #GetPm(int(1e2))
    #plot_interpol()









def P_m(k, z, z_prime, same_dim_on=False):
    """
    Returns the unequal time matter power spectrum
    by using the geometric approximation
    P(k, z, z')^2 = P(k, z) * P(k, z')

    The function only works for the following domain:
    1.1e-5 <      k     < 1.6   [h/Mpc]
    0.01   < z, z_prime < 1.5

    This function assumes that k, z and z_prime are 1D-arrays.
    Let the lengths of k, z, z_prime be denoted by
    nk, nz, nz_prime respectively. Then the shape
    of the returned power spectrum is (nk, nz, nz_prime).
    This will be the only relevant
    case for our computations.
    """
    z_dim = len(np.shape(z))
    z_prime_dim = len(np.shape(z_prime))
    # Dim is 0 for numbers, 1 for arrays and 2 for matrices
    if z_dim == 0 or z_prime_dim == 0:
        fac1 = np.transpose(P_m_equaltime(k, z))
        fac2 = np.transpose(P_m_equaltime(k, z_prime))
        P = np.transpose(np.sqrt(fac1*fac2))
        """
        Either fac1 or fac2 will be of shape (len(k)), and the other will be
        of shape (len(k), len(z)) or (len(k), len(z_prime)), so we transpose
        the factors to multiply them (as we may multiply shapes (k) with (z, k))
        and then transpose the product back again to the shape (k, z(prime))
        """
    elif z_dim == 1 and z_prime_dim == 1:
        nk = len(k)
        nz = len(z)
        nz_prime = len(z_prime)
        if nz == nz_prime and same_dim_on:
            P = np.sqrt(P_m_equaltime(k, z))*np.sqrt(P_m_equaltime(k, z_prime))
        else:
            P = np.zeros((nk, nz, nz_prime))
            for i in range(nz):
                P[:, i, :] = np.sqrt(P_m_equaltime(k, z_prime))
            Ptransp = np.transpose(P, (2,0,1))*np.sqrt(P_m_equaltime(k, z))
            P = np.transpose(Ptransp, (1,2,0))
    elif z_dim == 2 and z_prime_dim == 1:
        """
        z has the shape (zrow, zcol), and we will assume that
        z_prime has the shape (zcol)!
        """
        P1 = np.sqrt(P_m_equaltime(k, z))       # Shape (nk, nzrow, nzcol)
        P2 = np.sqrt(P_m_equaltime(k, z_prime)) # Shape (nk, nzcol)
        Ptransp = np.transpose(P1, (1, 0, 2))*P2
        P = np.transpose(Ptransp, (1, 0, 2))
    elif z_dim == 1 and z_prime_dim == 2:
        """
        z_prime has the shape (z_primerow, z_primecol), and we will assume that
        z has the shape (z_primecol)!
        """
        P1 = np.sqrt(P_m_equaltime(k, z))       # Shape (nk, z_primecol)
        P2 = np.sqrt(P_m_equaltime(k, z_prime)) # Shape (nk, z_primerow, z_primecol)
        Ptransp = P1*np.transpose(P2, (1, 0, 2))
        P = np.transpose(Ptransp, (1, 0, 2))
    else:
        P = np.sqrt(P_m_equaltime(k, z)*P_m_equaltime(k, z_prime))
    return P
