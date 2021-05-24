import numpy as np
import matplotlib.pyplot as plt

l = 100
c_M = -0.4
c_B = 0.8

"""
LIMBERS:
"""

try:
    C_sgLimber = np.load("Csglimber_l{}_cm{}.npy".format(l, 0))
    C_sgLimber1 = np.load("Csglimber_l{}_cm{}.npy".format(l, 1))
    C_sgLimber2 = np.load("Csglimber_l{}_cm{}_cb{}.npy".format(l, c_M, c_B))
    C_tgLimber = np.load("Ctglimber_l{}_cm{}.npy".format(l, 0))
    C_tgLimber1 = np.load("Ctglimber_l{}_cm{}.npy".format(l, 1))
    C_tgLimber2 = np.load("Ctglimber_l{}_cm{}_cb{}.npy".format(l, c_M, c_B))
    Limberzg = np.load("Limberzg_l{}.npy".format(l))
except FileNotFoundError:
    None


"""
FFTLOG:
"""
def getstuff(l, c_M=0, c_B=0):
    try:
        zg = np.load("zg_l{}_cm{}_cb{}.npy".format(l, c_M, c_B))
        Csg = np.load("Csg_l{}_cm{}_cb{}.npy".format(l, c_M, c_B))
        Ctg = np.load("Ctg_l{}_cm{}_cb{}.npy".format(l, c_M, c_B))
    except FileNotFoundError:
        zg = np.load("zg_l{}_cm{}.npy".format(l, c_M))
        Csg = np.load("Csg_l{}_cm{}.npy".format(l, c_M))
        Ctg = np.load("Ctg_l{}_cm{}.npy".format(l, c_M))
    return zg, Csg, Ctg

zg100B, Csg100B, Ctg100B = getstuff(l=100, c_M=c_M, c_B=c_B)
zg100M, Csg100M, Ctg100M = getstuff(l=100, c_M=1)
zg100GR, Csg100GR, Ctg100GR = getstuff(l=100)

zg40B, Csg40B, Ctg40B = getstuff(l=40, c_M=c_M, c_B=c_B)
zg40M, Csg40M, Ctg40M = getstuff(l=40, c_M=1)
zg40GR, Csg40GR, Ctg40GR = getstuff(l=40)

zg20B, Csg20B, Ctg20B = getstuff(l=20, c_M=c_M, c_B=c_B)
zg20M, Csg20M, Ctg20M = getstuff(l=20, c_M=1)
zg20GR, Csg20GR, Ctg20GR = getstuff(l=20)

zg5B, Csg5B, Ctg5B = getstuff(l=5, c_M=c_M, c_B=c_B)
zg5M, Csg5M, Ctg5M = getstuff(l=5, c_M=1)
zg5GR, Csg5GR, Ctg5GR = getstuff(l=5)


def plotdiffl():
    plt.plot(zg100B, Csg100B+Ctg100B, "-g")
    plt.plot(zg100M, Csg100M+Ctg100M, "-b")
    plt.plot(zg100GR, Csg100GR+Ctg100GR, "-r")

    plt.title("$\ell = 100$")
    plt.legend(["$c_M = -0.4$, $c_M = 0.8$","$c_M = 1$, $c_B = 0$","$c_M = 0$, $c_B = 0$"])
    plt.yscale("log")
    plt.axis([zg100GR[0],zg100GR[-1], 7e-8,1e-4])
    plt.show()

    plt.plot(zg40B, Csg40B+Ctg40B, "-g")
    plt.plot(zg40M, Csg40M+Ctg40M, "-b")
    plt.plot(zg40GR, Csg40GR+Ctg40GR, "-r")

    plt.title("$\ell = 40$")
    plt.legend(["$c_M = -0.4$, $c_M = 0.8$","$c_M = 1$, $c_B = 0$","$c_M = 0$, $c_B = 0$"])
    plt.yscale("log")
    plt.axis([zg100GR[0],zg100GR[-1], 7e-8,1e-4])
    plt.show()

    plt.plot(zg20B, Csg20B+Ctg20B, "-g")
    plt.plot(zg20M, Csg20M+Ctg20M, "-b")
    plt.plot(zg20GR, Csg20GR+Ctg20GR, "-r")

    plt.title("$\ell = 20$")
    plt.legend(["$c_M = -0.4$, $c_M = 0.8$","$c_M = 1$, $c_B = 0$","$c_M = 0$, $c_B = 0$"])
    #plt.yscale("log")
    #plt.axis([zg100GR[0],zg100GR[-1], 5e-8,1e-4])
    plt.show()

    plt.plot(zg5B, Csg5B+Ctg5B, "-g")
    plt.plot(zg5M, Csg5M+Ctg5M, "-b")
    plt.plot(zg5GR, Csg5GR+Ctg5GR, "-r")

    plt.title("$\ell = 5$")
    plt.legend(["$c_M = -0.4$, $c_M = 0.8$","$c_M = 1$, $c_B = 0$","$c_M = 0$, $c_B = 0$"])
    #plt.yscale("log")
    #plt.axis([zg100GR[0],zg100GR[-1], 5e-8,1e-4])
    plt.show()

def plotdiffc():
    plt.plot(zg100B, Csg100B+Ctg100B, "-")
    plt.plot(zg40B, Csg40B+Ctg40B, "-")
    plt.plot(zg20B, Csg20B+Ctg20B, "-")
    plt.plot(zg5B, Csg5B+Ctg5B, "-")

    plt.title("$c_M = -0.4$, $c_B = 0.8$")
    plt.legend(["$\ell = 100$", "$\ell = 40$","$\ell = 20$","$\ell = 5$"])
    #plt.yscale("log")
    #plt.axis([zg100GR[0],zg100GR[-1], 5e-8,1e-4])
    plt.show()

    plt.plot(zg100M, Csg100M+Ctg100M, "-")
    plt.plot(zg40M, Csg40M+Ctg40M, "-")
    plt.plot(zg20M, Csg20M+Ctg20M, "-")
    plt.plot(zg5M, Csg5M+Ctg5M, "-")

    plt.title("$c_M = 1$, $c_B = 0$")
    plt.legend(["$\ell = 100$", "$\ell = 40$","$\ell = 20$","$\ell = 5$"])
    #plt.yscale("log")
    #plt.axis([zg100GR[0],zg100GR[-1], 5e-8,1e-4])
    plt.show()


    plt.plot(zg100GR, Csg100GR+Ctg100GR, "-")
    plt.plot(zg40GR, Csg40GR+Ctg40GR, "-")
    plt.plot(zg20GR, Csg20GR+Ctg20GR, "-")
    plt.plot(zg5GR, Csg5GR+Ctg5GR, "-")

    plt.title("$c_M = 0$, $c_B = 0$")
    plt.legend(["$\ell = 100$", "$\ell = 40$","$\ell = 20$","$\ell = 5$"])
    #plt.yscale("log")
    #plt.axis([zg100GR[0],zg100GR[-1], 5e-8,1e-4])
    plt.show()

#plotdiffl()
#plotdiffc()


"""
plt.plot(zg100B, Csg100B, "-g")
#plt.plot(Limberzg, C_sgLimber2, "-")
plt.plot(zg100B, Ctg100B, "-g")
#plt.plot(Limberzg, C_tgLimber2, "-")


plt.plot(zg100M, Csg100M, "-b")
#plt.plot(Limberzg, C_sgLimber1, "-")
plt.plot(zg100M, Ctg100M, "-b")
#plt.plot(Limberzg, C_tgLimber1, "-")

plt.plot(zg100GR, Csg100GR, "-r")
#plt.plot(Limberzg, C_sgLimber, "-")
plt.plot(zg100GR, Ctg100GR, "-r")
#plt.plot(Limberzg, C_tgLimber, "-")

plt.yscale("log")
plt.axis([zg100GR[0],zg100GR[-1], 1e-8,1e-4])
plt.show()
"""

"""
plt.plot(zg100, Csg100, "-")
plt.plot(zg40, Csg40, "-")
plt.plot(zg20, Csg20, "-")
plt.plot(zg5, Csg5, "-")
plt.title("Plotting $C^{sg}(\ell)$ for $c_M = 0$")
plt.xlabel("$z_g$")
plt.ylabel("$C^{sg}$")
plt.legend(["$\ell = 100$", "$\ell = 40$", "$\ell = 20$", "$\ell = 5$"])
plt.savefig("Csgscm{}".format(c_M), dpi=300)
plt.show()

plt.plot(zg100, Ctg100, "-")
plt.plot(zg40, Ctg40, "-")
plt.plot(zg20, Ctg20, "-")
plt.plot(zg5, Ctg5, "-")
plt.title("Plotting $C^{tg}(\ell)$ for $c_M = 0$")
plt.xlabel("$z_g$")
plt.ylabel("$C^{tg}$")
plt.legend(["$\ell = 100$", "$\ell = 40$", "$\ell = 20$", "$\ell = 5$"])
plt.savefig("Ctgscm{}".format(c_M), dpi=300)
plt.show()

plt.plot(zg100, Csg100 + Ctg100, "-")
plt.plot(zg40, Csg40 + Ctg40, "-")
plt.plot(zg20, Csg20 + Ctg20, "-")
plt.plot(zg5, Csg5 + Ctg5, "-")
plt.title("Plotting $C^{wg}(\ell) = C^{sg}(\ell) + C^{tg}(\ell)$ for $c_M = 0$")
plt.xlabel("$z_g$")
plt.ylabel("$C^{wg}$")
plt.legend(["$\ell = 100$", "$\ell = 40$", "$\ell = 20$", "$\ell = 5$"])
plt.savefig("Cwgs{}".format(c_M), dpi=300)
plt.show()
"""
"""
plt.plot(zg, Csg, ".--")
plt.plot(zg, Ctg, ".--")
#plt.plot(Limberzg, C_sgLimber, "-")
#plt.plot(Limberzg, C_tgLimber, "-")
#plt.legend(["FFTlog", "FFTlog", "Limber", "Limber"])
plt.legend(["FFTlog", "FFTlog"])
#plt.yscale("log")
#plt.axis([zg[0],zg[-1], 1e-8,1e-4])
plt.show()
"""






#
