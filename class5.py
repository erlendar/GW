from window_funcs4 import Window_funcs
import numpy as np
import os

class CLASS(Window_funcs):
    def __init__(self, z_i):
        super().__init__(z_i)
        self.max_z_args = 50
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

        if len(np.shape(z)) >= 2: # i.e. if z is a matrix
            P = z.copy()
            for i in range(len(z)):
                zi = z[i]
                runs = int(np.ceil(len(zi)/max_z))
                Pi = zi.copy()
                for n in range(runs):
                    # We can only run 50 values of z simultaneously in CLASS
                    low_ind = max_z*n
                    if max_z*(n+1) >= len(zi):
                        high_ind = -1
                    else:
                        high_ind = max_z*(n+1)

                    z_ = zi[low_ind:high_ind]
                    self.get_Pm(z_)
                    wanted_ks = (l+0.5)/self.chi(z_) # 1/Mpc
                    wanted_ks *= self.h

                    found_Pks = np.copy(z_)

                    for j in range(len(z_)):
                        FileToRead = "../../Downloads/class_public-2.9.3"\
                                   + "/output/mydataz{}_pk.dat".format(j+1)
                        # Indexing starting at 0
                        # Path to be generalized
                        ks, Pks = self.read_output(FileToRead)

                        diff = np.abs(ks-wanted_ks[j])
                        ind = np.where(diff == np.min(diff))[0][0]
                        found_Pks[j] = Pks[ind]

                    Pi[low_ind:high_ind] = found_Pks
                P[i] = Pi
            return P



        zs = self.zs
        runs = int(np.ceil(len(zs)/max_z))
        PPPPP = zs.copy()
        for n in range(runs):
            low_ind = max_z*n
            if max_z*(n+1) >= len(zs):
                high_ind = -1
                zss = zs[low_ind:high_ind]
            else:
                high_ind = max_z*(n+1)
                zss = zs[low_ind:high_ind]

            self.get_Pm(zss)
            wanted_ks = (l+0.5)/self.chi(zss) # 1/Mpc
            wanted_ks *= self.h

            found_Pks = []

            for i in range(len(zss)):
                ks = []
                Pks = []
                infile = open("../../Downloads/class_public-2.9.3/output/mydataz{}_pk.dat".format(i+1), "r")
                # Indexing starting at 0
                # Path to be generalized
                infile.readline()
                infile.readline()
                infile.readline()
                infile.readline()
                for line in infile:
                    columns = line.strip().split(" ")
                    k = float(columns[0].strip())
                    Pk = float(columns[-1].strip())
                    ks.append(k)
                    Pks.append(Pk)
                infile.close()

                ks = np.array(ks)
                Pks = np.array(Pks)

                wanted_k = wanted_ks[i]
                diff = np.abs(ks-wanted_k)
                ind = np.where(diff == np.min(diff))[0][0]
                found_Pk = Pks[ind]
                found_Pks.append(found_Pk)

            np.array(found_Pks)
            PPPPP[low_ind:high_ind] = found_Pks

        return PPPPP


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

        s = s1 + s2 + s3
        s = s.format(*zs) # Example
        #print(s)
        outfile = open("../../Downloads/class_public-2.9.3/myrun.ini", "w")
        outfile.write(s)
        outfile.close()
        # Create input file

        os.system("cd; cd Downloads/class_public-2.9.3; ./class myrun.ini")
        # Run input file in Class from terminal
        # Paths to be generalized
        return None
