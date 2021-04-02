import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from genfuncs import integrate
import time


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


def Ibackwardsmat(l, nu, t, I_input=np.array([]), my_ls=np.array([])):
    """
    Using the recursion relation of I (and the function recursion of Python)
    to find I_{l}(nu, t). This is the backwards recursion method.

    Takes as input I_input, which is the value of I_9, I_10, I_19 etc (if they are available)
    given in a list
    """
    if len(my_ls) == 0 and len(I_input) != 0:
        n_input = len(I_input)
        gen_l_list = np.array([[9 + i, 10 + i] for i in range(0,90,10)]).flatten()
        my_ls = gen_l_list[np.where(gen_l_list <= l)][-n_input:]

    if type(nu) != np.ndarray:
        nu = np.array([nu])
    if type(t) != np.ndarray:
        t = np.array([t])
    # Assuming nu, t are floats/complex numbers if not arrays

    #print("Currently working on l = {}".format(l))
    if l == 0:
        BessInt = I_0mat(nu, t)
    elif l == 1:
        BessInt = I_1mat(nu, t)
    elif l in my_ls:
        ind = np.where(l == my_ls)[0][0]
        BessInt = I_input[ind]
    else:
        Iterm1 = np.zeros((len(t), len(nu)), dtype=complex)
        Iterm2 = np.copy(Iterm1)
        prefac = np.copy(Iterm1)

        prefac[:] = 1/(3 + l-2 - nu/2)
        np.transpose(Iterm1)[:] = (1 + t**2)/t*(l-2 + 3/2)
        Iterm1 *= Ibackwardsmat(l-1, nu, t, I_input, my_ls)

        Iterm2[:] = (l-2 + nu/2)
        Iterm2 *= Ibackwardsmat(l-2, nu, t, I_input, my_ls)

        BessInt = prefac*(Iterm1 - Iterm2)
    return BessInt

def I_bw(l, nu, t):
    l_list = np.array([[9 + i, 10 + i] for i in range(0,91,10)]).flatten()
    my_ls = np.concatenate((l_list[np.where(l_list < l)], np.array([l])))
    n_ls = len(my_ls)

    I_input_list = []
    l_prev = 0
    for i in range(n_ls):
        l = my_ls[i]
        n_I_input = len(I_input_list)
        current_ls = my_ls[np.where(l_list < l)]
        ls_sent = current_ls[-n_I_input:] if n_I_input > 0 else np.array([])
        I_temp = Ibackwardsmat(l, nu, t, np.array(I_input_list), ls_sent)
        I_input_list.append(I_temp)
        if (l-l_prev) == 1 and n_I_input > 2:
            I_input_list = I_input_list[-2:] # No need of dealing with a list with lots of large elements
        l_prev = l

    return I_temp




def timing_I():
    N = 100
    #n = N
    n = np.array([i for i in range(-N,N+1)],dtype=complex)
    b = 1.5
    kmin = 6.45e-3; kmax = 1e0
    P = np.log(kmax/kmin)
    nu = 1j*2*np.pi*n/P + b
    #t = np.array([0.9])
    t = np.linspace(0.3,0.9,800)
    l = 105

    """
    t1 = time.time()
    f = Ibackwardsmat(l, nu, t)#, l9=1, l10=1)
    t2 = time.time()
    print(t2-t1)
    """

    t1 = time.time()
    g = I_bw(l, nu, t)
    t2 = time.time()
    print(t2-t1)

    #print(np.max(np.abs(f-g)))

    """
    t21 = time.time()
    l9 = Ibackwardsmat(9, nu, t)
    l10 = Ibackwardsmat(10, nu, t, [l9])

    t22 = time.time()
    l19 = Ibackwardsmat(19, nu, t, [l9, l10])
    l20 = Ibackwardsmat(20, nu, t, [l9, l10, l19])
    t23 = time.time()
    g = Ibackwardsmat(l, nu, t, [l19, l20])
    t3 = time.time()
    print(t22-t21)
    print(t23-t22)
    print(t3-t23)

    print("Time of direct computation: {:.2f}s".format(t2-t1))
    print("Time of midstep computation: {:.2f}s".format(t3-t2))
    print("Max difference: {}".format(np.max(np.abs(g-f))))
    """
#timing_I()


def F_12_taylor(a,b,c,z):
    """
    Returns the Taylor series (to 3rd order in z)
    of the Hypergeometric function 2F1
    """
    term1 = 1
    term2 = a*b/c*z
    term3 = a*(a+1)*b*(b+1)/(c*(c+1))*z**2/2
    term4 = a*(a+1)*(a+2)*b*(b+1)*(b+2)/(c*(c+1)*(c+2))*z**3/(3*2)
    term5 = a*(a+1)*(a+2)*(a+3)*b*(b+1)*(b+2)*(b+3)/(c*(c+1)*(c+2)*(c+3))*z**4/(4*3*2)
    return term1 + term2 + term3 + term4 + term5

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

"""
def I_lim0(l, nu, t):

    grid = np.zeros((len(t), len(nu)))
    fac1 = np.copy(grid); fac2 = np.copy(grid)
    z = t**2

    np.transpose(fac1)[:] = (1+z)
    fac1 **= -(l + nu/2)
    fac1 = np.transpose(np.transpose(fac1)*t**l)
    fac1 **= 2**(nu - 1)*np.pi**2/gamma((3-nu)/2)

    fac2[:] = gamma((2*l + nu)/2)/gamma(l + 3/2)

    F = F_12_taylor((2*l + nu)/4, (2*l + nu + 2)/4, l + 3/2, 4*z/(1 + z)**2)
    return fac1*fac2*F
"""
def I_lim0(l, nu, t):
    """
    #Returns I_{l}(nu, t) in the limit t --> 0 by Taylor
    #expanding the hypergeometric function
    """
    if type(nu) != np.ndarray:
        nu = np.array([nu])
    if type(t) != np.ndarray:
        t = np.array([t])
    # Assuming nu, t are floats/complex numbers if not arrays

    grid = np.zeros((len(t), len(nu)), dtype=complex)
    tt = np.copy(grid)
    np.transpose(tt)[:] = t
    t = tt
    z = tt**2
    nuu = np.copy(grid)
    nuu[:] = nu
    nu = nuu

    fac1 = 2**(nu-1)*np.pi**2*t**l*(1+z)**(-(l+nu/2))/gamma((3-nu)/2)
    fac2 = gamma((2*l + nu)/2)/gamma(l + 3/2)
    F = F_12_taylor((2*l + nu)/4, (2*l + nu + 2)/4, l + 3/2, 4*z/(1 + z)**2)
    return fac1*fac2*F

def I_att0(nu):
    return 2*np.pi*np.cos(np.pi*nu/2)*gamma(nu-2)*2*(2-nu) #Last two factors found by lhopital

def I_att1(l, nu):
    """
    Returns the value of I_{l}(nu, t-->1)
    """
    nom = np.pi**(3/2)*gamma(l + nu/2)*gamma(1 - nu/2)
    denom = gamma((3 - nu)/2)*gamma(l + 2 - nu/2)
    return nom/denom

def I_taylor(l, nu, t):
    #I1 = # I at 0
    I2 = I_lim0(l, nu, t)
    I3 = I_lim(l, nu, t)
    #I4 =  #I at 1
    return np.where(t < 0.5, I2, I3)


"""
l=3
n = np.array([2,4])#np.array([3*i for i in range(21)])
b = 1.9
nt = 801
eps = 5e-8
t1 = np.linspace(0.01,0.9,nt)
t2 = np.geomspace(0.9,0.995,nt)
t3 = np.geomspace(0.995,0.9995,nt)
t4 = np.geomspace(0.9995,0.99998,nt)
t5 = np.geomspace(0.99998,1-eps,nt)
t = np.concatenate((t1,t2,t3,t4,t5))
#t = np.linspace(0.8, 1-eps, nt)
kmin = 6.45e-3; kmax = 1e0
P = np.log(kmax/kmin)
nu = 1j*2*np.pi*n/P + b
f = np.real(Ibackwardsmat(l,nu,t))
leg=[]
h = lambda tt, nuu: 1/t*((1+tt)**(2-nuu) + (1-tt)**(2-nuu))
for i in range(len(n)):
    #plt.plot(t,f[:,i],".")
    g = np.real(I_lim(l,nu[i],t))
    plt.plot(t,g,"--")
    #plt.plot(t,np.real(h(t,nu[i])))
    #plt.plot(t,np.imag(h(t,nu[i])))
    #leg.append("{:.2f}".format(np.real(I_att1(l, nu[i]))))
    #leg.append(None)
    #leg.append(None)
#plt.legend(leg)
plt.show()
"""

def compare_I_with_approx(l=2, n=5):
    eps = 1e-12
    endpoint = 1-eps
    startpoint = 0.01
    t = np.linspace(startpoint, endpoint, 500)
    from fourierpolexp import nu_func
    b = 1.5
    nu_n = nu_func(n) + b
    Ib = Ibackwards(l, nu_n, t)
    Ilim = I_taylor(l, nu_n, t)
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

def I(l, nu, t):
    """
    Using the best methods in the respective t-domains to return the I-integrals

    Cannot take nu-arrays as argument, only single values for now
    Can handle a maximum of n=+-200 corresponding to N = 200
    """
    kmin = 1e-4; kmax = 0.1
    P = np.log(kmax/kmin)
    fac = 2*np.pi/P
    N = np.imag(nu/(fac))
    """
    def tlim(n):
        n = np.abs(n)
        tmin = 0.3
        tmax = 0.8
        t = tmax*(1 - n/200) + tmin*(n/200)
        return t
    """
    def tlim(n, l):
        """
        The tlim depends on nu_n and l!
        This function is based very roughly on
        plotting and finding a simple way to
        paramtrize the effect of nu_n and l
        of the t_limit

        This t_limit basically says something about which t_values we
        need to evaluate in our C-integrals! In reality we only need
        to consider the interval [tlim(n,l), 1].
        """
        n = np.abs(n)
        tmin = 0.3
        tmax = 0.8
        t = tmax*(1 - n/200) + tmin*(n/200)
        # for n between 100 and 200:
        if n >= 50:
            t *= l/100
        elif n < 25 and l < 80:
            t *= (l+20)/100
        elif n < 50 and l < 90:
            t *= (l+10)/100
        return t

    #tlim = tlim(N)
    tlim = tlim(N, l)
    #plt.plot([tlim,tlim,], [-1,1],"-")
    t_Taylor = t[np.where(t < tlim)]
    t_bw = t[np.where(t >= tlim)]
    ITa = I_lim0(l, nu, t_Taylor)
    Ibw = I_bw(l, nu, t_bw)

    Ival = np.concatenate((ITa, Ibw))
    return Ival


def plot_Ibw():
    N = 20
    n = N
    eps = 1e-4
    #n = np.array([i for i in range(-N,N+1)],dtype=complex)
    b = 1.5
    kmin = 1e-4; kmax = 0.1
    P = np.log(kmax/kmin)
    nu = 1j*2*np.pi*n/P + b
    #t = np.array([0.9])
    t1 = np.linspace(0.01,0.85,200)
    t2 = np.linspace(0.85,0.95,201)
    t3 = np.geomspace(0.95,1-eps,201)
    t = np.concatenate((t1,t2,t3))
    l = 79

    f = I_lim0(l, nu, t)
    g = I_bw(l, nu, t)
    h = np.concatenate((I_lim0(l, nu, t[np.where(t<0.5)]), I_bw(l, nu, t[np.where(t>=0.5)])))
    i = I(l,nu,t)
    I0 = I_0(nu, t)

    print(I_att1(l,nu))
    tind=170
    #plt.plot(t,np.real(f),"r--")
    #plt.plot(t,np.imag(f),"b--")
    plt.plot(t[tind:],np.real(g)[tind:],"r")
    plt.plot(t[tind:],np.imag(g)[tind:],"b")
    #plt.plot(t,np.real(h),"r--")
    #plt.plot(t,np.imag(h),"b--")
    #plt.plot(t,np.abs(h),"k-")

    #plt.plot(t, np.real(I0),".")
    #plt.plot(t, np.imag(I0),".")
    plt.plot(t, np.real(i),"--")
    plt.plot(t, np.imag(i),"--")
    plt.show()
#plot_Ibw()




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

"""
def I_lim0(l, nu, t):



    fac1 = 2**(nu-1)*np.pi**2*t**l*(1+z)**(-(l+nu/2))/gamma((3-nu)/2)
    fac2 = gamma((2*l + nu)/2)/gamma(l + 3/2)
    F = F_12_taylor((2*l + nu)/4, (2*l + nu + 2)/4, l + 3/2, 4*z/(1 + z)**2)
    return fac1*fac2*F
"""


def check():
    def Ibackwards_from_start(l100,l99, nu, t, lstop=0):
        """
        Using the recursion relation of I (and the function recursion of Python)
        to find I_{l}(nu, t). This is the backwards recursion method.
        """
        l = 98
        Ilplus2 = l100
        Ilplus1 = l99
        while l >= lstop:
            print(l)
            Il = 1/(l + nu/2)*((1+t**2)/t*(l+3/2)*Ilplus1 - (3 + l -nu/2)*Ilplus2)
            Ilplus2 = Ilplus1
            Ilplus1 = Il
            l -= 1
        return Il


    kmin = 1e-4; kmax = 0.1
    P = np.log(kmax/kmin)
    n = 1
    b = 1.5
    nu = 1j*2*np.pi*n/P + b

    nt1 = 200; nt2 = 200; nt3 = 300; nt = nt1 + nt2 + nt3
    eps = 1e-4
    t1 = np.linspace(0.01,0.85,nt1 + 1)
    t2 = np.linspace(0.85,0.95,nt2 + 1)
    t3 = np.geomspace(0.95,1-eps,nt3)
    t = np.concatenate((t1[:-1],t2[:-1],t3))
    l100 = I(100, nu, t)[:,0]
    l99 = I(99, nu, t)[:,0]

    my_l = 0
    Il = I(my_l, nu, t)
    Il_b = Ibackwards_from_start(l100,l99, nu, t,lstop=my_l)
    I0 = I_0(nu, t)

    plt.plot(t, np.real(Il), ".")
    plt.plot(t, np.real(Il_b),"-")
    plt.plot(t, np.real(I0),"--")
    plt.show()
#check()
