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
        nz = len(z)
        P = np.zeros((nk, nz))
        for i in range(nk):
            P[i, :] = f(k[i], z)
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



def P_m(k, z, z_prime):
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
    try:
        nk = len(k)
        nz = len(z)
        nz_prime = len(z_prime)
        P = np.zeros((nk, nz, nz_prime))
        for i in range(nz):
            P[:, i, :] = np.sqrt(P_m_equaltime(k, z_prime))
        Ptransp = np.transpose(P, (2,0,1))*np.sqrt(P_m_equaltime(k, z))
        P = np.transpose(Ptransp, (1,2,0))
    except TypeError: # k, z or zp numbers instead of arrays
        P = np.sqrt(P_m_equaltime(k, z)*P_m_equaltime(k, z_prime))
    return P

#if __name__ == "__main__":
    #GetPm(int(1e2))
    #plot_interpol()
