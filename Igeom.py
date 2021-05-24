import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from fourierpolexp import nu_func, b
from besselint import I
from genfuncs import integrate
import time


l = 100
c_M = -0.4
c_B = 0.8
starttime = time.time()

try:
    nus = np.load("nuscm{}cb{}.npy".format(c_M, c_B))
    t_intpol = np.load("t_intpolcm{}cb{}.npy".format(c_M, c_B))
    zg = np.load("zFcm{}cb{}.npy".format(c_M, c_B))
    Fsg = np.load("Fsgcm{}cb{}.npy".format(c_M, c_B))
    Ftg = np.load("Ftgcm{}cb{}.npy".format(c_M, c_B))
except FileNotFoundError:
    nus = np.load("nuscm{}.npy".format(c_M))
    t_intpol = np.load("t_intpolcm{}.npy".format(c_M))
    zg = np.load("zFcm{}.npy".format(c_M))
    Fsg = np.load("Fsgcm{}.npy".format(c_M))
    Ftg = np.load("Ftgcm{}.npy".format(c_M))

print("Interpolating F's")
Fsgfunc = interpolate.interp1d(t_intpol, Fsg, axis=1, bounds_error=False, fill_value=0)
Ftgfunc = interpolate.interp1d(t_intpol, Ftg, axis=1, bounds_error=False, fill_value=0)

"""
nt0 = 2; nt1 = 2; nt2 = 150; nt3 = 350; nt4 = 150
nt5 = 200; nex = 400; nt6 = 300; nt7 = 300
nt = nt0 + nt1 + nt2 + nt3 + nt4 + nt5 + nt6 + nt7 + nex
t0 = np.geomspace(1e-4,0.1,nt0 + 1)
t1 = np.linspace(0.1,0.6,nt1 + 1)
t2 = np.linspace(0.6,0.8,nt2 + 1)
t3 = np.geomspace(0.8, 0.99, nt3 + 1)
t4 = np.geomspace(0.99, 0.999, nt4 + 1)
t5 = np.linspace(0.999,1,nt5)
tex = np.geomspace(1,1.005,nex+1)
t6 = np.geomspace(1.005,1.06,nt6+1)
t7 = np.linspace(1.06,1.5,nt7)
t = np.concatenate((t0[:-1], t1[:-1],t2[:-1],t3[:-1], t4[:-1], t5, tex[1:-1], t6[:-1], t7))
"""
nt0 = 100
nt1 = 100
nt2 = 150
nt3 = 350
nt4 = 150
nt5 = 200
nt52 = 100
nt53 = 100
nex = 400
nt6 = 300
nt7 = 500
nt = nt0 + nt1 + nt2 + nt3 + nt4 + nt5 + nt52 + nt53 + nt6 + nt7 + nex
t0 = np.linspace(0,0.1,nt0 + 1)
t1 = np.linspace(0.1,0.6,nt1 + 1)
t2 = np.linspace(0.6,0.8,nt2 + 1)
t3 = np.geomspace(0.8, 0.99, nt3 + 1)
t4 = np.geomspace(0.99, 0.999, nt4 + 1)
t5 = np.linspace(0.999,0.9999,nt5 + 1)
t52 = np.linspace(0.9999,1,nt52 +1)
t53 = np.linspace(1,1.0001, nt53+1)
tex = np.geomspace(1.0001,1.005,nex+1)
t6 = np.geomspace(1.005,1.06,nt6+1)
t7 = np.geomspace(1.06,5.4,nt7)
t = np.concatenate((t0[:-1], t1[:-1],t2[:-1],t3[:-1], t4[:-1], t5[:-1], t52[:-1], t53[:-1], tex[:-1], t6[:-1], t7))



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
#for i in range(len(Csg_)):
#    print(zg[i])
#    plt.plot(t, np.real(Csg_integrand[i]),".")
#    plt.show()
Csg = 2/(4*np.pi**2)*np.real(Csg_)
Ctg = 2/(4*np.pi**2)*np.real(Ctg_)

np.save("zg_l{}_cm{}_cb{}.npy".format(l, c_M, c_B), zg)
np.save("Csg_l{}_cm{}_cb{}.npy".format(l, c_M, c_B), Csg)
np.save("Ctg_l{}_cm{}_cb{}.npy".format(l, c_M, c_B), Ctg)

endtime = time.time()
timespent = (endtime - starttime)/60
print("Time spent: {:.1f} minutes".format(timespent))

plt.plot(zg, Csg, ".--")
plt.plot(zg, Ctg, ".--")
plt.yscale("log")
plt.axis([zg[0],zg[-1], 1e-8,1e-4])
plt.show()




#
