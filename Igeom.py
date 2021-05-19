import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from fourierpolexp import nu_func, b
from besselint import I

l = 100
c_M = 0

try:
    nus = np.load("nus.npy")
except FileNotFoundError:
    N=100
    N_arr = np.array([i for i in range(-N, N+1)])
    nus = nu_func(N_arr) + b

zg = np.load("zFcm{}.npy".format(c_M))
Fsg = np.load("Fsgcm{}.npy".format(c_M))
Ftg = np.load("Ftgcm{}.npy".format(c_M))


nt0 = 2
nt1 = 2
nt2 = 50
nt3 = 350
nt4 = 150
nt5 = 950
nt = nt0 + nt1 + nt2 + nt3 + nt4

t0 = np.geomspace(1e-4,0.1,nt0 + 1)
t1 = np.linspace(0.1,0.6,nt1 + 1)
t2 = np.linspace(0.6,0.8,nt2 + 1)
t3 = np.geomspace(0.8, 0.99, nt3 + 1)
t4 = np.geomspace(0.99, 0.999, nt4 + 1)
t5 = np.geomspace(0.999,20.4,nt5)
t = np.concatenate((t0[:-1], t1[:-1],t2[:-1],t3[:-1], t4[:-1], t5))


print("Computing Bessel-integrals I:")
Is = np.zeros((len(t), len(nus)), dtype=complex)
for i in range(len(nus)):
    if i in [n for n in range(0, len(nus), 22)]:
        print("{}%".format(int(i/len(nus)*100)))
    Is[:, i] = I(l, nus[i], t)[:,0]
print("100%")











#
