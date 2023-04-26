# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:55:49 2022

"""

from scipy.special import factorial
from scipy.special import legendre
import scipy.special as ss
import sympy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt
from math import pi
from sympy import sin
from sympy import cos
from sympy import exp
from sympy.abc import x

# %%Question 1

# %% Q1(a)

# Getting the analytical derivatives -----------------------------//
# The p after f represents 'prime'
f = sp.exp(sp.sin(2*x))
fp = sp.diff(f, x)
fpp = sp.diff(fp, x)
fppp = sp.diff(fpp, x)
# ---------------------------------------------------------------//

# Setting the derivatives to lambda functions for ease of use ---//
# The p after f represents 'prime' and the l indicates that it's a lambda function


def fl(x): return exp(sin(2*x))
def fpl(x): return 2*exp(sin(2*x))*cos(2*x)
def fppl(x): return -4*exp(sin(2*x))*sin(2*x) + 4*exp(sin(2*x))*cos(2*x)**2
def fpppl(x): return -24*exp(sin(2*x))*sin(2*x)*cos(2*x) + 8 * \
    exp(sin(2*x))*cos(2*x)**3 - 8*exp(sin(2*x))*cos(2*x)
# ---------------------------------------------------------------//


xs = [i*(2*pi/199) for i in range(200)]  # x value array

print(xs[len(xs)-1])
print(len(xs), xs[0], xs[len(xs)-1] - 2 * m.pi)

# Plot the derivatives f' to f''' and the original function f -----//
plt.plot(xs, [sp.exp(sp.sin(2*x)) for x in xs], 'b', label='f(x)')
plt.plot(xs, [2*exp(sin(2*x))*cos(2*x) for x in xs], 'g', label="f'(x)")
plt.plot(xs, [-4*exp(sin(2*x))*sin(2*x) + 4*exp(sin(2*x))
         * cos(2*x)**2 for x in xs], 'r', label="f''(x)")
plt.plot(xs, [-24*exp(sin(2*x))*sin(2*x)*cos(2*x) + 8*exp(sin(2*x))*cos(2*x)
         ** 3 - 8*exp(sin(2*x))*cos(2*x) for x in xs], 'm', label="f'''(x)")
plt.xlabel('x', fontsize=10)
plt.ylabel("f(x), f'(x), f''(x), and f'''(x)", fontsize=10)
plt.title("Plot of f(x), f'(x), f''(x), and f'''(x)")
plt.xlim()
plt.ylim()
plt.legend(loc='upper right', prop={'size': 11})
plt.grid()
plt.show()
# ---------------------------------------------------------------//

# %% Q1(b)

h1 = 0.15
h2 = 0.5  # Step sizes

# Forward and Central Diff for h1 and h2 --------------------------//
fpfd1 = []
fpfd2 = []  # Initialize the lists for the fd for h1 and h2
for i in xs:
    fpfd1.append((fl(i + h1)-fl(i))/h1)  # fd for h1
for i in xs:
    fpfd2.append((fl(i + h2)-fl(i))/h2)  # fd for h2
# print(fpfd1)
fpcd1 = []
fpcd2 = []  # Initialize the lists for the cd for h1 and h2
for i in xs:
    fpcd1.append(((fl(i + h1/2)-fl(i - h1/2))/h1))  # fd for h1
for i in xs:
    fpcd2.append(((fl(i + h2/2)-fl(i - h2)/2)/h2))  # cd for h2
# ---------------------------------------------------------------//

# Plot the fd and cd for both step sizes and the Analytical solution of f'
plt.plot(xs, [fpl(x) for x in xs], 'k', label="Analytical f'(x)")
plt.plot(xs, fpfd1, 'b', linestyle='dashed',
         label="Forward Difference h1 f'(x)")
plt.plot(xs, fpcd1, 'r', linestyle='dashed',
         label="Central Difference h1 f'(x)")
plt.plot(xs, fpfd2, 'g', linestyle='dashed',
         label="Forward Difference h2 f'(x)")
plt.plot(xs, fpcd2, 'm', linestyle='dashed',
         label="Central Difference h2 f'(x)")
plt.ylabel("f'(x)", fontsize=10)
plt.xlabel('x', fontsize=10)
plt.title("Plot of f'(x)")
plt.xlim()
plt.ylim()
plt.legend(loc='upper right', prop={'size': 7})
plt.grid()
plt.show()
# ---------------------------------------------------------------//

# %% Q1(c)

eps = np.finfo(np.float64).eps  # Machine error
hs = np.geomspace(1e-16, 1, 16)  # Array of x values


def erfds(h): return (2*abs(fl(1))*(eps/h)) + ((h/2)*fppl(1))  # fd error
def ercds(h): return (2*abs(fl(1))*(eps/h)) + \
    (((h**2)/24)*(fpppl(1)))  # cd error


# Difference b/w analytical and fd
def fdifffd(h): return fpl(1) - ((fl(1+h)-fl(1))/h)
# Difference b/w analytical and cd
def fdiffcd(h): return fpl(1) - ((fl(1+(h/2))-fl(1-(h/2)))/h)


# Plots the absolute fd and cd error and the abolute difference b/w analytical and fd and the analytical and cd
plt.plot(hs, [abs(erfds(i)) for i in hs], 'b',
         linestyle='dashed', label="$\epsilon_{fd}$")
plt.plot(hs, [abs(ercds(i)) for i in hs], 'r',
         linestyle='dashed', label="$\epsilon_{cd}$")
plt.plot(hs, [abs(fdifffd(i)) for i in hs], 'g',
         linestyle='dashed', label="f-difference fd")
plt.plot(hs, [abs(fdiffcd(i)) for i in hs], 'm',
         linestyle='dashed', label="f-difference cd")
plt.ylabel("abs error and f-difference", fontsize=10)
plt.xlabel('h', fontsize=10)
plt.title("Plot of abs error and f-difference vs h")
plt.xlim()
plt.ylim()
plt.legend(loc='lower left')
plt.text(10**-15, 10**-5, '@x=1')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()
# ---------------------------------------------------------------//

# %% Q1(d)

# We used lambdify to avoid referencing issues with the maths functions
fa = sp.lambdify(x, f)  # The same as f as above
fpa = sp.lambdify(x, fp)  # The same as fp above


def fd(h): return (fa(1+h)-fa(1))/h  # Forward Difference Function
def cd(h): return (fl(1+(h/2))-fl(1-(h/2)))/h  # Central Difference Function

# Difference b/w analytical and fd; we used this b/c it avoided referencing issues


def ohfd(h):
    oh = fpa(1)-((fa(1+h)-fa(1))/h)
    return oh
# ---------------------------------------------------------------//


# Absolute fd Richardson error
def erfdr(h): return abs((2.*fd(h/2)-fd(h))-ohfd(h**2))
def ercdr(h): return abs(((4.*cd(h/2)-cd(h))/3.) -
                         fdiffcd(h**4))  # Absolute cd Richardson error


# Plots the absolute Richardson fd and cd error and the abolute difference b/w analytical and fd and the analytical and cd
plt.plot(hs, [erfdr(i) for i in hs], 'b', linestyle='dashed',
         label="$\epsilon_{fd}$-Richardson")
plt.plot(hs, [ercdr(i) for i in hs], 'r', linestyle='dashed',
         label="$\epsilon_{cd}$-Richardson")
plt.plot(hs, [abs(fdifffd(i)) for i in hs], 'g',
         linestyle='dashed', label="f-difference fd")
plt.plot(hs, [abs(fdiffcd(i)) for i in hs], 'm',
         linestyle='dashed', label="f-difference cd")
plt.ylabel("Richardson error and f-difference", fontsize=10)
plt.xlabel('h', fontsize=10)
plt.title("Plot of Richardson error and f-difference vs h")
plt.xlim()
plt.ylim()
plt.legend(loc='lower left')
plt.text(10**-15, 10**-5, '@x=1')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()
# ---------------------------------------------------------------//

# %%Question 2


# %% Q2(a)

# x value array
nsteps = 200
xsr = [i/nsteps for i in range(-nsteps+1, nsteps)]

h3 = 0.01  # step size

# C-d related functions ---------------------------------------//


# Function to be differentiated from the Rodrigues LP; with the variable x considered
def fr(n, x): return (x**2 - 1)**n


def cdcalc1(n, x, h):  # cd for f'
    cd = (fr(n, x+h/2) - fr(n, x-h/2))/h
    return cd


def cdcalc2(n, x, h):  # cd for f''
    cd = (cdcalc1(n, x+h/2, h) - cdcalc1(n, x-h/2, h))/h
    return cd


def cdcalc3(n, x, h):  # cd for f'''
    cd = (cdcalc2(n, x+h/2, h) - cdcalc2(n, x-h/2, h))/h
    return cd


def cdcalc4(n, x, h):  # cd for f''''
    cd = (cdcalc3(n, x+h/2, h) - cdcalc3(n, x-h/2, h))/h
    return cd
# ---------------------------------------------------------------//

# To use for the plots ------------------------------------------//

# Analytical LP's -----------------------------------------------


def aLP(n):
    Pn = legendre(n)
    y = Pn(xsr)
    return y
# ---------------------------------------------------------------

# Rodrigues LP's ------------------------------------------------


def rodLP(n, x):
    if n == 1:
        rlp = cdcalc1(n, x, h3)/((2**n)*factorial(n))
        return rlp
    elif n == 2:
        rlp = cdcalc2(n, x, h3)/((2**n)*factorial(n))
        return rlp
    elif n == 3:
        rlp = cdcalc3(n, x, h3)/((2**n)*factorial(n))
        return rlp
    elif n == 4:
        rlp = cdcalc4(n, x, h3)/((2**n)*factorial(n))
        return rlp
    else:
        print("Not in the scope of this section")
        return False
# ---------------------------------------------------------------//


# Plot the functs of Rodrigues and analytical LP from n=1 to n=4 ---//
for j in range(1, 5):
    plt.plot(xsr, [rodLP(j, i) for i in xsr], 'c', label="Rodrigues'")
    plt.plot(xsr, aLP(j), 'k', linestyle='dotted',
             linewidth=5, label="Analytical")
    plt.title(
        "Plot of the Rodrigues' and Analytical LP's at n = {0}".format(j))
    plt.xlabel("x")
    plt.ylabel("$P_{0}(x)$".format(j), fontsize=20)
    plt.xlim()
    plt.ylim()
    plt.legend(loc='upper center')
    plt.grid()
    plt.show()
# ---------------------------------------------------------------//

# %% Q2(b)


# The non-derivative term in the Rodrigues' LP
def constPL(n): return 1/((2**n)*factorial(n))
# Function to be differentiated taken from the Rodrigues'
def frpl(n): return (x**2 - 1)**n

# Recursive function that calculates a specified number of central differences


def cdcalcR(f, n, x, h, c):
    if c == n-1:
        cd = (fr(n, x+h/2) - fr(n, x-h/2))/h
        return cd
    cd = (cdcalcR(f, n, x+h/2, h, c+1)-cdcalcR(f, n, x-h/2, h, c+1))/h
    return cd
# ---------------------------------------------------------------//

# This function determines the LP of size n with the Rrodrigues'


def rodRPL(n, x, h):
    y = frpl(n)
    aaa = cdcalcR(y, n, x, h, 0)
    return constPL(n)*aaa
# ---------------------------------------------------------------//


# Plots the LP's from n=1 to n=8 ---------------------------------//
for j in range(1, 9):
    plt.plot(xsr, [rodRPL(j, i, h3) for i in xsr], 'b', label="Rodrigues'")
    plt.plot(xsr, aLP(j), 'darkorange', linestyle='dotted',
             linewidth=5, label="Analytical")
    plt.title(
        "Plot of the Rodrigues' and Analytical LP's at n = {0}".format(j))
    plt.xlabel("x")
    plt.ylabel("$P_{0}(x)$".format(j), fontsize=20)
    plt.xlim()
    plt.ylim()
    plt.legend(loc='upper center')
    plt.grid()
    plt.show()
# ---------------------------------------------------------------//
