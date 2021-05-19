from window_funcs4 import Window_funcs
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import os

class CLASS(Window_funcs):
    def __init__(self, z_i):
        super().__init__(z_i)
        self.max_z_args = 40
        # Maximum number of z-arguments that you can have in
        # the CLASS input file

    def read_output(self, txt):
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

    def P_m(self, l, z):
        """
        Read CLASS output file and return the matter power spectrum

        l corresponds to the multipole, z the redshifts
        """
        max_z = self.max_z_args

        if len(np.shape(z)) == 1:   # i.e. if z is a vector
            P = self.P_m2(z,l)
        elif len(np.shape(z)) == 2: # i.e. if z is a matrix
                                    # Recall that it is the columns of z we
                                    # want to find the power spectrum over!
            P = z.copy()
            number_of_columns = len(z[0])
            for i in range(number_of_columns):
                zcol = z[:,i]
                Pcol = self.P_m2(zcol,l)
                P[:,i] = Pcol
        return P


    def P_m2(self,zcol,l):
        max_z = self.max_z_args
        runs = int(np.ceil(len(zcol)/max_z))
        Pcol = zcol.copy()
        for n in range(runs):
            # We can only run 50 values of z simultaneously in CLASS
            low_ind = max_z*n
            if max_z*(n+1) >= len(zcol):
                high_ind = None # Slicing x[i:None] gives all elements except the i first
            else:
                high_ind = max_z*(n+1)
            z_ = zcol[low_ind:high_ind]
            self.get_Pm(z_)
            wanted_ks = (l+0.5)/self.chi(z_) # 1/Mpc
            wanted_ks *= self.h

            found_Pks = np.copy(z_)
            for j in range(len(z_)):
                FileToRead = "../../../Downloads/class_public-2.9.3"\
                           + "/output/mydataz{}_pk.dat".format(j+1)
                # Indexing starting at 0
                # Path to be generalized
                ks, Pks = self.read_output(FileToRead)

                Pkfunc = interpolate.interp1d(ks,Pks)
                found_Pks[j] = Pkfunc(wanted_ks[j])

            Pcol[low_ind:high_ind] = found_Pks
        return Pcol



    def get_Pm(self, zs):
        """
        Run CLASS in terminal to get the matter power spectrum P(k,z)

        NOTE:
        len(zs) must be smaller than 50,
        as CLASS cannot handle too many arguments
        """
        s2 = len(zs)*"{}, "
        s2 = s2[:-2] + "\n"
        s1 = "h = {}\n".format(self.h)\
           + "omega_b = {} # baryon density\n".format(self.omega_b*self.h**2)\
           + "Omega_Lambda = {}\n".format(self.omega_de)\
           + "Omega_k = 0. #curvature\n"\
           + "output = mPk\n"\
           + "z_pk = "
           # Omega_cdm
        s3 = "root = output/mydata"

        """
        s1 = "\n".join([omega_b ..., omega_Lambda ...])
        with open(jfkgkdjgkfd, "w") as fp:
            fp.write(s)

        path = pathlib.Path("../../Downloads/class_public-2.9.3/myrun.ini")
        with path.open("w") as fp:
            fp.write(s)
        """

        s = s1 + s2 + s3
        s = s.format(*zs)
        outfile = open("../../../Downloads/class_public-2.9.3/myrun.ini", "w")
        outfile.write(s)
        outfile.close()
        # Create input file

        os.system("cd; cd Downloads/class_public-2.9.3; ./class myrun.ini")
        # Subprocess
        # Run input file in Class from terminal
        # Paths to be generalized
        return None
