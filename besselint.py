import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from genfuncs import integrate


def I_0(nu, t):
    I0 = 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu - 2)*1/t\
        * ((1 + t)**(2-nu) - (1 - t)**(2-nu))
    return I0

def I_0mat(nu, t):
    I0 = np.zeros((len(t), len(nu)), dtype=complex)
    term1 = np.copy(I0); term2 = np.copy(I0)

    I0[:] = 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu - 2)
    Itransp = np.transpose(I0)
    Itransp *= 1/t

    np.transpose(term1)[:] = (1+t)
    np.transpose(term2)[:] = (1-t)
    term1 **= (2-nu)
    term2 **= (2-nu)

    return I0*(term1 - term2)


def I_1(nu, t):
    # FOUND A MISTAKE IN A SIGN IN EQ. 2.39 FOR THIS FUNCTION
    # BY COMPARING WITH https://arxiv.org/pdf/1705.05022.pdf (EQUATION B.9)
    I1 = 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu - 2)/((4-nu)*t**2)\
        * ((1 + t)**(2-nu)*((1 - t)**2 + nu*t) - (1 - t)**(2-nu)\
        * ((1 + t)**2 - nu*t))
    return I1


def I_1mat(nu, t):

    I1 = np.zeros((len(t), len(nu)), dtype=complex)
    nu_t = np.copy(I1);
    t_plus = np.copy(I1); t_minus = np.copy(I1)
    prefac1 = np.copy(I1); prefac2 = np.copy(I1)

    np.transpose(nu_t)[:] = t
    nu_t *= nu
    np.transpose(t_plus)[:]  = (1 + t)**2
    np.transpose(t_minus)[:] = (1 - t)**2
    np.transpose(prefac1)[:] = (1+t)
    np.transpose(prefac2)[:] = (1-t)
    prefac1 **= (2-nu)
    prefac2 **= (2-nu)

    np.transpose(I1)[:] = 1/t**2
    I1 *= 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu - 2)/(4-nu)

    return I1*(prefac1*(t_minus + nu_t) - prefac2*(t_plus - nu_t))


def Ibackwards(l, nu, t):
    """
    Using the recursion relation of I (and the function recursion of Python)
    to find I_{l}(nu, t). This is the backwards recursion method.
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


def Ibackwardsmat(l, nu, t):
    """
    Using the recursion relation of I (and the function recursion of Python)
    to find I_{l}(nu, t). This is the backwards recursion method.
    """
    if type(nu) != np.ndarray:
        nu = np.array([nu])
    if type(t) != np.ndarray:
        nu = np.array([t])
    # Assuming nu, t are floats/complex numbers if not arrays
    print("Currently working on l = {}".format(l))
    if l == 0:
        BessInt = I_0mat(nu, t)
    elif l == 1:
        BessInt = I_1mat(nu, t)
    else:
        Iterm1 = np.zeros((len(t), len(nu)), dtype=complex)
        Iterm2 = np.copy(Iterm1)
        prefac = np.copy(Iterm1)

        prefac[:] = 1/(3 + l-2 - nu/2)
        np.transpose(Iterm1)[:] = (1 + t**2)/t*(l-2 + 3/2)
        Iterm1 *= Ibackwardsmat(l-1, nu, t)

        Iterm2[:] = (l-2 + nu/2)
        Iterm2 *= Ibackwardsmat(l-2, nu, t)

        BessInt = prefac*(Iterm1 - Iterm2)
    return BessInt


def F_12_taylor(a,b,c,z):
    """
    Returns the Taylor series (to 3rd order in z)
    of the Hypergeometric function 2F1
    """
    return 1 + a*b/c*z + a*(a+1)*b*(b+1)/(c*(c+1))*z**2/2 + a*(a+1)*(a+2)*b*(b+1)*(b+2)/(c*(c+1)*(c+2))*z**3/(3*2)

def I_lim(l, nu, t):
    """
    Returns I_{l}(nu, t) in the limit t --> 1 by Taylor
    expanding the hypergeometric function
    """
    z = t**2
    prefac = np.pi**(3/2)*t**l*(2/(1+z))**(l+nu/2)/gamma((3-nu)/2)
    term1 = gamma((2*l+nu)/2)*gamma((2-nu)/2)/gamma((4+2*l-nu)/2)\
          * F_12_taylor((2*l+nu)/4, (2*l+nu+2)/4, nu/2, ((1-z)/(1+z))**2)
    term2 = gamma((nu-2)/2)*(2*(1+z)/(1-z))**(nu-2)\
          * F_12_taylor((2*l+6-nu)/4, (2*l+4-nu)/4, 2-nu/2, ((1-z)/(1+z))**2)
    return prefac*(term1 + term2)


def I_att1(l, nu):
    """
    Returns the value of I_{l}(nu, t-->1)
    """
    nom = np.pi**(3/2)*gamma(l + nu/2)*gamma(1 - nu/2)
    denom = gamma((3 - nu)/2)*gamma(l + 2 - nu/2)
    return nom/denom

def compare_I_with_approx(l=2, n=5):
    eps = 1e-12
    endpoint = 1-eps
    startpoint = 0.1
    t = np.linspace(startpoint, endpoint, 500)
    from fourierpolexp import nu_func
    nu_n = nu_func(n)
    Ib = Ibackwards(l, nu_n, t)
    Ilim = I_lim(l, nu_n, t)
    print("\nAnalytical value as t ---> 1:")
    print (I_att1(l, nu_n), "\n")
    plt.plot(t, np.real(Ib), "r.")
    plt.plot(t, np.imag(Ib), "b.")
    plt.plot(t, np.real(Ilim), "r--")
    plt.plot(t, np.imag(Ilim), "b--")
    plt.xlabel("t")
    plt.legend(["I_real", "I_imag", "Taylor_real", "Taylor_imag"])
    plt.show()
#compare_I_with_approx()




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














"""
def I_lim2(l, nu, t):
    #Returns I_{l}(nu, t) in the limit t --> 1 by Taylor
    #expanding the hypergeometric function
    tt = -(1-t**2)**2/(4*t**2)
    prefac = np.pi**(3/2)*t**(-nu/2)/gamma((3-nu)/2)
    term1 = gamma(l+nu/2)*gamma(1-nu/2)/(gamma(2+l-nu/2)*gamma(nu/2-1))\
          * F_12_taylor(l/2+nu/4, nu/4 - (l+1)/2, nu/2, tt)
    term2 = (-tt/4)**(1-nu/2)\
          * F_12_taylor(l/2-nu/4+1, 1/2-l/2-nu/4, 2-nu/2, tt)
    return prefac*(term1 + term2)
"""







#
