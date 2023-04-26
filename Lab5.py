# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 19:31:38 2022

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x

# %%Question 1

# %%Q1(a)

# Runge Function


def frug(x): return 1/(1+25*x**2)


# Spacing
n = 15

# Use "cheb" for the Chebyshev points and "eqd" equidistant spcaing


def gendata(n, f, atype):
    if (atype == "cheb"):
        x = []
        for i in range(1, n+1):
            x.append(np.cos(((2*i-1)/(2*n))*np.pi))
        x = np.array(x)
        y = f(x)
    if (atype == "eqd"):
        x = np.linspace(-1, 1, n)
        y = f(x)
    return x, y


# Cheb points for n = 15
chebx, cheby = gendata(n, frug, "cheb")

# Equidistant points for n = 15
eqx, eqy = gendata(n, frug, "eqd")

# Gets the coeffs for the monomial basis


def monobasiscoeff(n, xdt, ydt):
    A = np.zeros((n, n))
    for i in range(n):
        A[::, i] = np.power(np.transpose(xdt), i)
    return np.flip(np.linalg.solve(A, ydt), axis=0)


# Monomial Basis functuion for Equidistant points
fmonoeq = np.poly1d(monobasiscoeff(n, eqx, eqy))

# Monomial Basis functuion for Cheb points
fmonocheb = np.poly1d(monobasiscoeff(n, chebx, cheby))

# 100 points
xfull = np.linspace(-1, 1, 100)

# Lagrangian Basis


def lagbasis(n, xval, xdt, ydt):
    y = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if (i != j):
                p *= (xval - xdt[j])/(xdt[i] - xdt[j])
        y += ydt[i]*p
    return y


# Lagrangian Basis functuion for Equidistant points
flageq = lagbasis(n, xfull, eqx, eqy)

# Lagrangian Basis functuion for Cheb points
flagcheb = lagbasis(n, xfull, chebx, cheby)

# Plots the graphs side by side as in the hw instrictions --------------------//
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

# Cheb Plots
plt.scatter(chebx, cheby, color='m', label='point')
plt.plot(xfull, flagcheb, '-r', label='Lagrange')
plt.plot(xfull, np.polyval(fmonocheb, xfull), '--g', label='Monomial')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc=7)

plt.subplot(1, 2, 2)

# Equidistant Plots
plt.scatter(eqx, eqy, color='m', label='point')
plt.plot(xfull, flageq, '-r', label='Lagrange')
plt.plot(xfull, np.polyval(fmonoeq, xfull), '--g', label='Monomial')
plt.xlabel('x')
plt.legend(loc=9)
plt.show()

# ----------------------------------------------------------------------------//


# %%Q1(b)

# 500 points
xfull = np.linspace(-1, 1, 500)

# For the n = 101 spacing ----------------------------------------------------//

# Spacing
n = 101

# Cheb points for n = 101
chebx, cheby = gendata(n, frug, "cheb")

# Equidistant points for n = 101
eqx, eqy = gendata(n, frug, "eqd")

# Monomial Basis functuion for Equidistant points
fmonoeq = np.poly1d(monobasiscoeff(n, eqx, eqy))

# Monomial Basis functuion for Cheb points
fmonocheb = np.poly1d(monobasiscoeff(n, chebx, cheby))

# Lagrangian Basis functuion for Equidistant points
flageq = lagbasis(n, xfull, eqx, eqy)

# Lagrangian Basis functuion for Cheb points
flagcheb = lagbasis(n, xfull, chebx, cheby)

# Plots the graphs side by side as in the hw instrictions
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

plt.scatter(chebx, cheby, color='m', label='point')
plt.plot(xfull, flagcheb, '-r', label='Lagrange')
plt.plot(xfull, np.polyval(fmonocheb, xfull), '--g', label='Monomial')
plt.title('n = 101')
plt.ylabel('f')
plt.xlabel('x')
plt.legend(loc=7)

# ----------------------------------------------------------------------------//

# For the n = 91 spacing -----------------------------------------------------//

# Spacing
n = 91

# Cheb points for n = 91
chebx, cheby = gendata(n, frug, "cheb")

# Equidistant points for n = 91
eqx, eqy = gendata(n, frug, "eqd")

# Monomial Basis functuion for Equidistant points
fmonoeq = np.poly1d(monobasiscoeff(n, eqx, eqy))

# Monomial Basis functuion for Cheb points
fmonocheb = np.poly1d(monobasiscoeff(n, chebx, cheby))

# Lagrangian Basis functuion for Equidistant points
flageq = lagbasis(n, xfull, eqx, eqy)

# Lagrangian Basis functuion for Cheb points
flagcheb = lagbasis(n, xfull, chebx, cheby)

plt.subplot(1, 2, 2)

# Plots the graphs side by side as in the hw instrictions
plt.scatter(chebx, cheby, color='m', label='point')
plt.plot(xfull, flagcheb, '-r', label='Lagrange')
plt.plot(xfull, np.polyval(fmonocheb, xfull), '--g', label='Monomial')
plt.title('n = 91')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc=7)
plt.show()

# ----------------------------------------------------------------------------//

# Justification --------------------------------------------------------------//
print()
print("Justification:")
print("\tBoth of the n = 101 and n = 91 for the monomial and lagrangian appear the same")
print("\tthis is expected since at such large n values the interpolations will be rather close")
print("\tto the function we are trying to interpolate thus at a glance there will be no dicernable")
print("\tdifference; however, if we were to zoom in to the edges we would see more oscillations for")
print("\tn = 91 than for n= 101; furthermore, if we really look and zoom in we will see that n = 101 is")
print("\ta better approximation than n = 91")
# ----------------------------------------------------------------------------//

# %%Question 2

# %%Q2

# 100 points
xfull = np.linspace(-1, 1, 100)

# Cubic Spline Funuction


def cubes(n, xdt, ydt):
    m = n-2
    A = np.zeros((m, m))
    b = np.zeros(m)

    # This loop defines the b vector and the A matrix
    for i in range(m):
        b[i] = 6*(((ydt[i+2]-ydt[i+1])/(xdt[i+2]-xdt[i+1])) -
                  ((ydt[i+1]-ydt[i])/(xdt[i+1]-xdt[i])))
        if (i != m-1):
            A[i][i+1] = xdt[i+2]-xdt[i+1]
            A[i+1][i] = xdt[i+2]-xdt[i+1]
        A[i][i] = 2*(xdt[i+2]-xdt[i])

    # Coefficients of interest taken from the Ax=b
    coef = list(np.linalg.solve(A, b))
    coef.insert(0, 0)
    coef.append(0)
    coef = np.array(coef)

    xv = []
    yv = []
    # This loop gets s_k,k-1(x)
    for j in range(1, n):
        l = j-1
        def s(x): return ydt[l]*((xdt[j]-x)/(xdt[j]-xdt[l]))+ydt[j]*((x-xdt[l])/(xdt[j]-xdt[l]))-(coef[l]/6)*((xdt[j]-x)*(
            xdt[j]-xdt[l])-(((xdt[j]-x)**3)/(xdt[j]-xdt[l])))-(coef[j]/6)*((x-xdt[l])*(xdt[j]-xdt[l])-((x-xdt[l])**3)/(xdt[j]-xdt[l]))
        xs = np.linspace(xdt[l], xdt[j], int(len(xfull)/(len(xdt)-1)))
        # This loop puts the x and y values of the splines into a list
        for k in xs:
            yv.append(s(k))
            xv.append(k)

    return xv, yv

# For n = 7 --------------------------------------------//


# Cheb points for n = 7
chebx, cheby = gendata(7, frug, "cheb")

# Cubic Cheb x and y vals for n = 7
fcubcheb = cubes(7, chebx, cheby)

# Equidistant points for n = 7
eqx, eqy = gendata(7, frug, "eqd")

# Cubic Equidistant x and y vals for n = 7
fcubeq = cubes(7, eqx, eqy)

# ------------------------------------------------------//

# For n = 15 -------------------------------------------//

# Cheb points for n = 15
chebx2, cheby2 = gendata(15, frug, "cheb")

# Cubic Cheb x and y vals for n = 15
fcubcheb2 = cubes(15, chebx2, cheby2)

# Equidistant points for n = 15
eqx2, eqy2 = gendata(15, frug, "eqd")

# Cubic Equidistant x and y vals for n = 15
fcubeq2 = cubes(15, eqx2, eqy2)

# ------------------------------------------------------//

# Plot the interpolations ----------------------------------//

plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

# Plot the Chebyshev for n = 7 and n = 15 --------------//
plt.scatter(chebx, cheby, color='m', label='point')
plt.plot(fcubcheb[0], fcubcheb[1], '-b', label='Cheb n = 7')
plt.plot(fcubcheb2[0], fcubcheb2[1], '-r', label='Cheb n = 15')
plt.title('Chebyshev')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc=7)
# ------------------------------------------------------//

plt.subplot(1, 2, 2)

# Plot the Equidistant for n = 7 and n = 15 ------------//
plt.scatter(eqx, eqy, color='m', label='point')
plt.plot(fcubeq[0], fcubeq[1], '-b', label='Equidistant n = 7')
plt.plot(fcubeq2[0], fcubeq2[1], '-r', label='Equidistant n = 15')
plt.title('Equidistant')
plt.xlabel('x')
plt.legend(loc=7)
# ------------------------------------------------------//

# ----------------------------------------------------------//

# %%Question 3

# %%Q3

# The function to interpolate


def fun(x): return np.exp(np.sin(2*x))


# 500 points
xfull = np.linspace(0, 2*np.pi, 500)


def trigs(n, f):
    # Define variables
    m = (n-1)/2
    xj = (2*np.pi/n)*np.linspace(0, n, n, endpoint=False)
    kw = np.linspace(0, m, int(m+1))
    yj = f(xj)
    ak = np.zeros_like(kw)
    bk = np.zeros_like(kw)
    px = (2*np.pi/len(xfull))*np.linspace(0, len(xfull), len(xfull))
    py = np.zeros_like(px)

    # This loop find the ak and bk coeffs
    for i in range(int(m+1)):
        ak[i] = ((1/m)*(np.sum(yj*np.cos(kw[i]*xj))))
        bk[i] = ((1/m)*(np.sum(yj*np.sin(kw[i]*xj))))

    # This loop gets the actual interpolated function values
    for i in range(len(px)):
        py[i] = ((ak[0]/2)+np.sum(ak[1:]*np.cos(kw[1:]*px[i]) +
                 bk[1:]*np.sin(kw[1:]*px[i])))

    return px, py, xj, yj

# Plot graphs ----------------------------------------------------------------//


plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

# Plot for n = 11 --------------------------------------//
xvals, yvals, pointx, pointy = trigs(11, fun)
plt.plot(xvals, yvals, '-b', label='n = 11')
plt.scatter(pointx, pointy, color='m', label='points')
plt.title("n = 11")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# ------------------------------------------------------//

plt.subplot(1, 2, 2)

# Plot for n = 51 --------------------------------------//
xvals, yvals, pointx, pointy = trigs(51, fun)
plt.plot(xvals, yvals, '-b', label='n = 51')
plt.scatter(pointx, pointy, color='m', label='points')
plt.title("n = 51")
plt.xlabel("x")
plt.legend()
plt.show()
# ------------------------------------------------------//

# ----------------------------------------------------------------------------//

print()
print("---------------------------------------------------------------------------------------------------//")
print()

# %% End

print("***END OF PROGRAM***")
