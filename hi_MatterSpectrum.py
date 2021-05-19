import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from class5 import CLASS
from MatterSpectrum import P_m
import os

def GetPm_hi(nz=int(1e2), c_M=0, nk=1):

    # This is determined by CLASS (?)
    #nz = int(2e3)
    nk_tot = 114 + nk

    ks = np.geomspace(0.013*0.6763, 0.02*0.6763, nk)
    #ks = np.geomspace(1e-6, 2, nk)
    #z = np.linspace(0.01, 1.5, nz)
    z = np.geomspace(1e-5, 2, nz)
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

    def RunCLASS(zs):
        omega_m  = 0.308
        omega_de = 0.692
        omega_b  = 0.04867
        h        = 0.6763
        """
        Run CLASS in terminal to get the matter power spectrum P(k,z)

        NOTE:
        len(zs) must be smaller than 50,
        as CLASS cannot handle too many arguments
        """
        infile = open("myhiruntemplate.txt", "r")
        ss1 = infile.read() + "\n"
        infile.close()

        #ss1 = "gravity_model = propto_omega\n"\
        #    + "parameters_smg = 1., 0., 0., 0., 1.\n"


        sz = len(zs)*"{}, "
        sz = sz[:-2] + "\n"

        sk = len(ks)*"{}, "
        sk = sk[:-2] + "\n"

        s1 = "h = {}\n".format(h)\
           + "omega_b = {} # baryon density\n".format(omega_b*h**2)\
           + "omega_m = {}\n".format(omega_m)\
           + "Omega_Lambda = {}\n".format(omega_de)\
           + "Omega_k = 0. #curvature\n"

        s3 = "output = mPk\n"\
           + "z_pk = "

        s4 = "root = output/myhidata"


        s = s1 + ss1 + s3 + sz + s4


        S = "Omega_Lambda = 0\n"\
          + "Omega_fld = 0\n"\
          + "Omega_smg = -1\n"\
          + "gravity_model = propto_omega\n"\
          + "parameters_smg = 1., 0, {}, 0., 1.\n".format(c_M)\
          + "expansion_model = lcdm\n"\
          + "expansion_smg = 0.5\n"\
          + "k_output_values = " + sk.format(*ks) + "\n"

        SALT = S + s3 + sz + s4
        SALT = SALT.format(*zs)


        s = s.format(*zs)
        s=SALT

        outfile = open("../../../Downloads/hi_class_public-hi_class/myhirun.ini", "w")
        outfile.write(s)
        outfile.close()
        # Create input file

        os.system("cd; cd Downloads/hi_class_public-hi_class/; ./class myhirun.ini")
        # Subprocess
        # Run input file in Class from terminal
        # Paths to be generalized
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
        P = np.zeros((nk_tot,len(z)))
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
            Pk1 = np.zeros((len(z_),nk_tot))
            for j in range(len(z_)):
                FileToRead = "../../../Downloads/hi_class_public-hi_class/"\
                           + "/output/myhidataz{}_pk.dat".format(j+1)
                # Indexing starting at 0
                # Path to be generalized
                ks, Pks = read_output(FileToRead)
                Pk1[j,:] = Pks
            np.transpose(P[:,low_ind:high_ind])[:] = Pk1
        np.save("hi_karray.npy", ks)
        np.save("hi_zarray.npy", z)
        np.save("hi_Parray.npy", P)
        return None

    Pmsave(z)
    print("... MATTER POWER SPECTRUM SAVED")
    return None
#GetPm_hi()


def GetDat():
    k = np.load("hi_karray.npy")
    z = np.load("hi_zarray.npy")
    P = np.load("hi_Parray.npy")
    return k, z, P

def intpol(fetchP=False, c_M=0):
    if fetchP:
        GetPm_hi(c_M=c_M)
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


def P_m_equaltime_hi(k, z):
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
    plt.loglog(z,P_m_equaltime_hi(k[-4],z),"r--")
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









def P_m_hi(k, z, z_prime, same_dim_on=False, samedim2=True, c_M=0):
    GetPm_hi(c_M=c_M)
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
        fac1 = np.transpose(P_m_equaltime_hi(k, z))
        fac2 = np.transpose(P_m_equaltime_hi(k, z_prime))
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
            P = np.sqrt(P_m_equaltime_hi(k, z))*np.sqrt(P_m_equaltime_hi(k, z_prime))
        else:
            P = np.zeros((nk, nz, nz_prime))
            np.transpose(P, (1, 0, 2))[:] = np.sqrt(P_m_equaltime_hi(k, z_prime))
            Ptransp = np.transpose(P, (2,0,1))*np.sqrt(P_m_equaltime_hi(k, z))
            P = np.transpose(Ptransp, (1,2,0))
    elif z_dim == 2 and z_prime_dim == 1:
        """
        z has the shape (zrow, zcol), and we will assume that
        z_prime has the shape (zcol)!
        """
        P1 = np.sqrt(P_m_equaltime_hi(k, z))       # Shape (nk, nzrow, nzcol)
        P2 = np.sqrt(P_m_equaltime_hi(k, z_prime)) # Shape (nk, nzcol)
        Ptransp = np.transpose(P1, (1, 0, 2))*P2
        P = np.transpose(Ptransp, (1, 0, 2))
    elif z_dim == 1 and z_prime_dim == 2:
        """
        z_prime has the shape (z_primerow, z_primecol), and we will assume that
        z has the shape (z_primecol)!

        Not anymore:
        If else:
        P will have the shape (k, z, zprimerow, zprimecol)
        """
        P1 = np.sqrt(P_m_equaltime_hi(k, z))       # Shape (nk, z)
        P2 = np.sqrt(P_m_equaltime_hi(k, z_prime)) # Shape (nk, z_primerow, z_primecol)
        if not samedim2:
            nk = len(k)
            nz = len(z)
            nz_primerow = len(z_prime[:,0])
            nz_primecol = len(z_prime[0,:])
            P = np.zeros((nz, nk, nz_primerow, nz_primecol))
            np.transpose(P, (2, 3, 1, 0))[:] = P1
            P *= P2
            P = np.transpose(P, (1, 0, 2, 3))
        else:
            Ptransp = P1*np.transpose(P2, (1, 0, 2))
            P = np.transpose(Ptransp, (1, 0, 2))
    else:
        P = np.sqrt(P_m_equaltime_hi(k, z)*P_m_equaltime_hi(k, z_prime))
    return P




#z = np.linspace(0.01,1,1001)
#z = 0.9
#k = np.linspace(1e-4,0.2,1000)
#P = P_m_hi(k, z, 0.8, c_M=0)
#P2 = P_m(k, z, 0.8)
#plt.plot(k, P,".-")
#plt.plot(k,P2,".")
#plt.legend(["Modified Pm", "Original Pm"])
#plt.show()








#
