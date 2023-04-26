# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 12:51:15 2022

"""

import sympy as sp
import numpy as np
import scipy.misc as sd
import scipy.misc as sd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as ax
from math import pi
from numpy import exp
from sympy.abc import x, y, z
from sympy import Matrix, N

# %%Question 1

# %%Q1(a)

# Function


def fa(x): return 1/(x-3)


def bisection(f, a, b, tol):
    mid = (a+b)/2
    if (abs(f(mid)-f(a)) <= tol or abs(f(mid)-f(b)) <= tol):
        return mid
    if ((f(mid)*f(a)) < 0):
        return bisection(f, a, mid, tol)
    else:
        return bisection(f, mid, b, tol)


# interval
a = 0
b = 5
# tolerance
tol = 1.e-8

# bisection(fa, a, b, tol)

'''
Justification:
    The answer is correct since we get "ZeroDivisionError: float division by zero", this means
    that the function is asymptotic, which it is, so it makse sense to get an error since the method won't
    stop until the tolerance condition is met, which for this function it won't happen and it will keep
    halfing until it approaches x = 3 which is where the vertical asymptote of the function is.
    The function also doesn't have roots.
'''

# %%Q1(b)

# Function


def fb(x): return exp(x - np.sqrt(x)) - x


# tolerance
tol = 1.e-8

# With the bisection code --------------------------------//
# interval
a = 0
b = 1.5

print("Root near x = 1 w/ the Bisection code:", bisection(fb, a, b, tol))

# ------------------------------------------------------//

# With the Newton solver code ---------------------------//
xog = 0.01  # initial guess

# Print the curve we want the root of
xr = np.array([i*0.01 for i in range(125)])
plt.plot(xr, fb(xr), 'b')
plt.plot([0, len(xr)*0.01], [0, 0], color='gray')

# the function to compute the numerical derivative


def fp(f, g): return sd.derivative(f, g, dx=1e-6)


gar = []

# this function finds the root with Newtons method and plots the tangent lines


def newtonandplot(f, g, tol):
    if (abs(f(g)) <= tol):
        return g

    # Print the tangenrt lines
    xr = np.linspace(g, g-(f(g)/fp(f, g)), 100)
    plt.plot(xr, fp(f, g)*(xr-g)+f(g), 'r', linestyle='dashed')
    plt.plot([g, g], [0, f(g)], color='k', linestyle='dotted')
    gar.append(g)

    return newtonandplot(f, g-(f(g)/fp(f, g)), tol)


print("Root near x = 1 w/ the Newton code:", newtonandplot(fb, xog, tol))

# Formatting the graph
for i in range(len(gar)-2):
    plt.text(gar[i], -0.07, "x$^($$^{0}$$^)$".format(i))
plt.ylim(-0.25, 1)
plt.grid()
plt.show()

# this is the pure Newton method w/out the plotting


def newton(f, g, tol):
    if (abs(f(g)) <= tol):
        return g
    return newton(f, g-(f(g)/fp(f, g)), tol)


# print("Newton code:", newton(fb,xog,tol))
print()
# ------------------------------------------------------//
print("-------------------------------------------------------------------//")
print()

# %%Q1(c)

# suppressed function


def u(x): return (exp(x - np.sqrt(x)) - x)/(x - 1)


# tolerance
tol = 1.e-8

# guess array
guess = [2.0, 0.5, 4.0, 0.1]

# printing the found root for each guess
for i in range(len(guess)):
    print("Root for guess x =", guess[i], ":", newton(u, guess[i], tol))

print()
print("-------------------------------------------------------------------//")
print()

# %%Q1(d)

# tolerance
tol = 1.e-8

# interval
a = 0
b = 2

# sphere density
rhos = 0.8

# sphere radius
rad = 1
# water density
rhow = 1
# shpere density (rhos) ranges 0<rhos<1

# Mass equation


def mass(rad, rho): return (4/3)*pi*(rad**3)*rho

# Volume equation


def vol(rad, h): return (1/3)*pi*(3*rad*h**2 - h**3)


# Height equation
def fh(h, rhos): return 3*h**2 - h**3 - 4*rhos

# Bisection method that takes a variable sphere density


def bisection2(f, a, b, rho, tol):
    mid = (a+b)/2
    if (abs(f(mid, rho)-f(a, rho)) <= tol or abs(f(mid, rho)-f(b, rho)) <= tol):
        return mid
    if ((f(mid, rho)*f(a, rho)) < 0):
        return bisection2(f, a, mid, rho, tol)
    else:
        return bisection2(f, mid, b, rho, tol)


print("h for", rhos, "sphere density:", bisection2(fh, a, b, rhos, tol))

# Printing the plot of the sphere and water surface in 3D
# -------------------------------------//
u, v = np.meshgrid(np.linspace(0, 2*pi, 100), np.linspace(0, pi, 100))
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax = plt.axes(projection='3d')
ax.plot_surface(xs, ys, zs+1-bisection2(fh, a, b, rhos, tol),
                cmap=plt.cm.YlGnBu_r, alpha=1)
xsr, ysr = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
ax.plot_surface(xsr, ysr, xsr*0, color='b', alpha=0.7)
# ax.view_init(0,90)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# -------------------------------------//

print()
print("-------------------------------------------------------------------//")
print()

# %%Question 2

# %%Q2(a)

# Set 1 of functions


def fs1(xs):
    x0, x1 = xs
    def f0(xs=xs): return x0**2 - 2*x0 + x1**4 - 2*x1**2 + x1
    def f1(xs=xs): return x0**2 + x0 + 2*x1**3 - 2*x1**2 - 1.5*x1 - 0.05
    return np.array([f0, f1])

# Set 2 of functions


def fs2(xs):
    x0, x1, x2 = xs
    def f0(xs=xs): return 2*x0 - x1*sp.cos(x2) - 3
    def f1(xs=xs): return x0**2 - 25*(x1-2)**2 + sp.sin(x2) - (pi/10)
    def f2(xs=xs): return 7*x0*sp.exp(x1) - 17*x2 + 8*pi
    return np.array([f0, f1, f2])


# %%Q2(b)

# Step Size
hs = 1.e-4

# tolerance
tol = 1.e-8

# sets of variables of each set
xs1 = np.array([x, y])
xs2 = np.array([x, y, z])

# Central differencing method


def cd(f, xs, h):
    e = np.identity(len(xs))
    cd = []
    for i in range(len(xs)):
        for j in range(len(xs)):
            cd.append((f(xs+e[j]*h)[i](xs+e[j]*h) -
                      f(xs-e[j]*h)[i](xs-e[j]*h))/(2*h))
    return cd

# jacobian matrix


def jacobian(f, xs, h):
    jm = []
    jm.append(cd(f, xs, h))
    return np.array(jm).reshape(len(xs), len(xs))

# print(jacobian(fs1,xs1,hs))
# print(jacobian(fs2,xs2,hs))


# %%Q2(c)

# Step Size
hs = 1.e-4

# tolerance
tol = 1.e-8

# sets of numbers
x1 = np.array([1., 1.])
x2 = np.array([1., 1., 1.])

# sets of variables of each set
xs1 = np.array([x, y])
xs2 = np.array([x, y, z])

print("Jacobian for fs1 for x = [1,1]:")
print(jacobian(fs1, x1, hs))
print()
print("Jacobian for fs2 for x = [1,1,1]:")
print(jacobian(fs2, x2, hs))
print()

# Newtons Method for the system of equations


def newtonND(jac, f, xv, h, tol):
    beta = []
    for i in range(len(xv)):
        beta.append(f(xv)[i](xv))
    beta = -1*np.array(beta)
    rts = np.linalg.solve(jac.astype(np.float64), beta.astype(np.float64))
    eps = abs(rts[0]/(xv[0]+rts[0]))
    if (eps <= tol):
        return xv
    return newtonND(jacobian(f, xv+rts, h), f, xv+rts, h, tol)


print("Roots for fs1 for x = [1,1]:")
print(newtonND(jacobian(fs1, x1, hs), fs1, x1, hs, tol))
print()
print("Putting the roots found back into fs1:")
print(fs1(newtonND(jacobian(fs1, x1, hs), fs1, x1, hs, tol))
      [0](newtonND(jacobian(fs1, x1, hs), fs1, x1, hs, tol)))
print(fs1(newtonND(jacobian(fs1, x1, hs), fs1, x1, hs, tol))
      [1](newtonND(jacobian(fs1, x1, hs), fs1, x1, hs, tol)))
print()
print("Roots for fs2 for x = [1,1,1]:")
print(newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol))
print()
print("Putting the roots found back into fs2:")
print(fs2(newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol))
      [0](newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol)))
print(fs2(newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol))
      [1](newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol)))
print(fs2(newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol))
      [2](newtonND(jacobian(fs2, x2, hs), fs2, x2, hs, tol)))

'''
jac11 = jacobian(fs1,x1,hs)
newtonND11 = newtonND(jac11, fs1, x1, hs, tol)
jac22 = jacobian(fs2,x2,hs)
newtonND22 = newtonND(jac22, fs2, x2, hs, tol)

print("Roots for fs1 for x = [1,1]:")
print(newtonND11)
print()
print("Putting the roots found back into fs1:")
print(fs1(newtonND11)[0](newtonND11))
print(fs1(newtonND11)[1](newtonND11))
print()
print("Roots for fs2 for x = [1,1,1]:")
print(newtonND22)
print()
print("Putting the roots found back into fs2:")
print(fs2(newtonND22)[0](newtonND22))
print(fs2(newtonND22)[1](newtonND22))
print(fs2(newtonND22)[2](newtonND22))
'''

print()
print("-------------------------------------------------------------------//")
print()

# %%Q2(d)

X = Matrix([2*x - y*sp.cos(z) - 3, x**2 - 25*(y-2)**2 +
           sp.sin(z) - (pi/10), 7*x*sp.exp(y) - 17*z + 8*pi])

Y = Matrix([x, y, z])

jac = X.jacobian(Y)

print("Analytical Solution for the Jacobian for fs2 at x = [1,1,1]:")
print(np.array(N(jac.subs([(x, 1), (y, 1), (z, 1)]))))

print()
print("-------------------------------------------------------------------//")
print()

# %% End

print("***END OF PROGRAM***")
