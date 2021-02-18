import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from genfuncs import integrate


def I_0(nu, t):
    I0 = 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu - 2)*1/t\
        * ((1 + t)**(2-nu) - (1 - t)**(2-nu))
    return I0

def I_1(nu, t):
    I1 = 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu - 2)/((4-nu)*t**2)\
        * ((1 + t)**(2-nu)*((1 + t)**2 + nu*t) - (1 - t)**(2-nu)\
        * ((1+t)**2 - nu*t))
    return I1

def Ibackwards(l, nu, t):
    """
    Using the recursion relation of I (and the function recursion of Python)
    to find I_{l}(nu, t)
    """
    if l == 0:
        BessInt = I_0(nu, t)
    elif l == 1:
        BessInt = I_1(nu, t)
    else:
        #I = np.zeros()
        BessInt = 1/(3 + l-2 - nu/2)*((1 + t**2)/t*(l-2 + 3/2)*Ibackwards(l-1, nu, t) \
          - (l-2 + nu/2)*Ibackwards(l-2, nu, t))
    return BessInt


"""
nu = 3 + 4.3j
eps = 1e-2
t = np.linspace(eps,1-eps,50)
print(integrate(t,Ibackwards(20,nu,t)))


test:
l=2
I2 = 1/(3 + l-2 - nu/2)*((1 + t**2)/t*(l-2 + 3/2)*I_1(nu, t) \
  - (l-2 + nu/2)*I_0(nu, t))
l = 3
I3 = 1/(3 + l-2 - nu/2)*((1 + t**2)/t*(l-2 + 3/2)*I2 \
  - (l-2 + nu/2)*I_1(nu, t))
print(I2)
print(I3)
"""

#print (I_0(nu,t))










#
