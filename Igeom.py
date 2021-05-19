import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from fourierpolexp import nu_func, b
from besselint import I
from genfuncs import integrate
import time


l = 100
c_M = 0
starttime = time.time()

try:
    nus = np.load("nus.npy")
    t_intpol = np.load("t_intpol.npy")
except FileNotFoundError:
    N=100
    N_arr = np.array([i for i in range(-N, N+1)])
    nus = nu_func(N_arr) + b

    nt1 = 300
    nt2 = 100
    nt = nt1 + nt2
    t1 = np.linspace(0,4,nt1+1)[:-1]
    t2 = np.linspace(4,22,nt2)
    t_intpol = np.concatenate((t1, t2))

zg = np.load("zFcm{}.npy".format(c_M))
Fsg = np.load("Fsgcm{}.npy".format(c_M))
Ftg = np.load("Ftgcm{}.npy".format(c_M))

print("Interpolating F's")
Fsgfunc = interpolate.interp1d(t_intpol, Fsg, axis=1, bounds_error=False, fill_value=0)
Ftgfunc = interpolate.interp1d(t_intpol, Ftg, axis=1, bounds_error=False, fill_value=0)

nt0 = 2; nt1 = 2; nt2 = 150; nt3 = 350; nt4 = 150
nt5 = 200; nex = 300; nt6 = 300; nt7 = 300
nt = nt0 + nt1 + nt2 + nt3 + nt4 + nt5 + nt6 + nt7 + nex
t0 = np.geomspace(1e-4,0.1,nt0 + 1)
t1 = np.linspace(0.1,0.6,nt1 + 1)
t2 = np.linspace(0.6,0.8,nt2 + 1)
t3 = np.geomspace(0.8, 0.99, nt3 + 1)
t4 = np.geomspace(0.99, 0.999, nt4 + 1)
t5 = np.linspace(0.999,1,nt5)
tex = np.linspace(1,1.005,nex+1)
t6 = np.geomspace(1.005,1.06,nt6+1)
t7 = np.linspace(1.06,1.5,nt7)
t = np.concatenate((t0[:-1], t1[:-1],t2[:-1],t3[:-1], t4[:-1], t5, tex[1:-1], t6[:-1], t7))

print("Evaluating F's at lots of t's")
Fsg = Fsgfunc(t)
Ftg = Ftgfunc(t)

print("Computing Bessel-integrals I:")
Is = np.zeros((len(t), len(nus)), dtype=complex)
for i in range(len(nus)):
    if i in [n for n in range(0, len(nus), 22)]:
        print("{}%".format(int(i/len(nus)*100)))
    Is[:, i] = I(l, nus[i], t)[:,0]
print("100%")


Csg_integrand = np.sum(Fsg*Is, axis=-1)
Ctg_integrand = np.sum(Ftg*Is, axis=-1)
Csg_ = integrate(t, Csg_integrand, ax=-1)
Ctg_ = integrate(t, Ctg_integrand, ax=-1)
Csg = 2/(4*np.pi**2)*np.real(Csg_)
Ctg = 2/(4*np.pi**2)*np.real(Ctg_)

np.save("zg_l{}_cm{}.npy".format(l, c_M), zg)
np.save("Csg_l{}_cm{}.npy".format(l, c_M), Csg)
np.save("Ctg_l{}_cm{}.npy".format(l, c_M), Ctg)

endtime = time.time()
timespent = (endtime - starttime)/60
print("Time spent: {:.1f} minutes".format(timespent))

plt.plot(zg, Csg, ".--")
plt.plot(zg, Ctg, ".--")
plt.yscale("log")
plt.axis([zg[0],zg[-1], 1e-8,1e-4])
plt.show()




#
