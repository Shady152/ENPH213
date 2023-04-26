# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:45:30 2022

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import sin, cos, exp, pi

# %%Question 1

# %%Q1(a)

# Get the function data and the time data


def callf(n, a0, w0, a1, w1, a2, w2, l):
    t = np.arange(0, l, l/n)
    fun = a0*sin(w0*t)
    fun += a1*sin(w1*t) + a2*sin(w2*t)
    return (t, fun)

# Calculate the DFT and the corresponding frequencies


def dft(n, y):
    if (n % 2 != 0):
        sys.exit("ERROR: The number of points (n) is not even")
    # Here we get the DFT values
    N1 = len(y)
    n1 = np.arange(N1)
    k = n1.reshape((N1, 1))
    ex = exp((-2j*pi*k*n1)/N1)
    dft = np.dot(ex, y)
    # Below here we get the frequencies
    N2 = len(dft)
    n2 = np.arange(N2)
    frq = (n*n2)/N2
    return (frq, dft)

# Calculate the IDFT


def invdft(n, y):
    if (n % 2 != 0):
        sys.exit("ERROR: The number of points (n) is not even")
    N1 = len(y)
    n1 = np.arange(N1)
    k = n1.reshape((N1, 1))
    ex = exp((2j*pi*k*n1)/N1)
    return (1/N1)*(np.dot(ex, y))


tr1, fr1 = callf(30, 3., 1., 1., 4., 0.5, 7., 2*pi)
tb1, fb1 = callf(60, 3., 1., 1., 4., 0.5, 7., 2*pi)

# Plot the figures -----------------------------------------------------------//
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)
plt.plot(tb1, fb1, '-b', label='n = 60')
plt.plot(tr1, fr1, '--r', label='n = 30')
plt.ylabel('y')
plt.xlabel('t')
plt.legend(loc=1)

plt.subplot(1, 2, 2)

frqb1, yb1 = dft(60, fb1)
plt.stem(frqb1, abs(yb1), 'r', markerfmt=" ", basefmt="-r", label='n = 60')

frqr1, yr1 = dft(30, fr1)
plt.stem(frqr1, abs(yr1), 'b', markerfmt=" ", basefmt="-b", label='n = 30')

plt.ylabel(r'|$\tilde{y}$|')
plt.xlabel(r'$\omega$')
plt.legend(loc=9)

plt.show()

# ----------------------------------------------------------------------------//

# %%Q1(b)

# Calculate the DFT with np.linspace rather than np.arange


def dftlin(n, y):
    if (n % 2 != 0):
        sys.exit("ERROR: The number of points (n) is not even")
    # Here we get the DFT values
    n1 = np.linspace(0, n-1, n)
    n1 = n1.astype(int)
    k = n1.reshape((n, 1))
    ex = exp((-2j*pi*k*n1)/n)
    dft = np.dot(ex, y)
    # Below here we get the frequencies
    N2 = len(dft)
    n2 = np.linspace(0, n-1, n)
    frq = (n*n2)/N2
    return (frq, dft)


tr2, fr2 = callf(30, 3., 1., 1., 4., 0.5, 7., 2*pi)
tb2, fb2 = callf(60, 3., 1., 1., 4., 0.5, 7., 2*pi)

# Plot the figures -----------------------------------------------------------//
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)
plt.plot(tb2, fb2, '-b', label='n = 60')
plt.plot(tr2, fr2, '--r', label='n = 30')
plt.ylabel('y')
plt.xlabel('t')
plt.legend(loc=1)

plt.subplot(1, 2, 2)

frqb2, yb2 = dftlin(60, fb2)
plt.stem(frqb2, abs(yb2), 'r', markerfmt=" ", basefmt="-r", label='n = 60')

frqr2, yr2 = dftlin(30, fr2)
plt.stem(frqr2, abs(yr2), 'b', markerfmt=" ", basefmt="-b", label='n = 30')

plt.ylabel(r'|$\tilde{y}$|')
plt.xlabel(r'$\omega$')
plt.legend(loc=9)

plt.show()

# ----------------------------------------------------------------------------//

# %%Q1(c)

# For n = 60 plot the IDFT from the calculated DFT above
plt.figure(figsize=(16, 8), dpi=80)
plt.plot(tb1, np.real(invdft(60, yb1)), '-b', label='Inverse')
plt.plot(tb1, fb1, '--r', label='Original')
plt.ylabel('y')
plt.xlabel('t')
plt.legend(loc=1)
plt.show()

# %%Question 2

# %%Q2(a)

# The gauss pulse function


def gaussf(t, sig, w): return exp(-((t**2)/(sig**2)))*cos(w*t)


# Range from -pi to pi for n = 60
trng1 = np.linspace(-pi, pi, 60)

# Get the gaussian function data
ygauss1 = gaussf(trng1, 0.5, 0)

# Get the DFT
fqr1, dft1 = dft(60, ygauss1)

# Shifting the DFT
wshift1 = np.fft.fftfreq(len(fqr1), trng1[1]-trng1[0])*2.*pi
wshift1 = np.fft.fftshift(wshift1)
yshift1 = np.fft.fftshift(dft1)

# Plot the figures -----------------------------------------------------------//
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

plt.plot(trng1, ygauss1, '-b')
plt.xlim(-2.5, 2.5)
plt.xlabel('t')
plt.ylabel('y(t)')

plt.subplot(1, 2, 2)

plt.plot(fqr1, abs(dft1), '-b', label='No Shift')
plt.plot(wshift1, abs(yshift1), '--r', label='Shifted')
plt.ylabel(r'|$\tilde{y}$|')
plt.xlabel(r'$\omega$')
plt.legend(loc=6)
plt.show()

# ----------------------------------------------------------------------------//

# %%Q2(b)

trng2 = np.linspace(-pi, pi, 400)

# For w = 10 -----------------------------------------------------------------//

# Get the gaussian function data
ygauss2 = gaussf(trng2, 1, 10)

# Get the DFT
fqr2, dft2 = dft(400, ygauss2)

# Shifting the DFT
wshift2 = np.fft.fftfreq(len(fqr2), trng2[1]-trng2[0])*2.*pi
wshift2 = np.fft.fftshift(wshift2)
yshift2 = np.fft.fftshift(dft2)

# ----------------------------------------------------------------------------//

# For w = 20 -----------------------------------------------------------------//

# Get the gaussian function data
ygauss3 = gaussf(trng2, 1, 20)

# Get the DFT
fqr3, dft3 = dft(400, ygauss3)

# Shifting the DFT
wshift3 = np.fft.fftfreq(len(fqr3), trng2[1]-trng2[0])*2.*pi
wshift3 = np.fft.fftshift(wshift3)
yshift3 = np.fft.fftshift(dft3)

# ----------------------------------------------------------------------------//

# Plot the figures -----------------------------------------------------------//
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

plt.plot(trng2, ygauss2, '-b')
plt.plot(trng2, ygauss3, '-r')
plt.xlim(-3, 3)
plt.xlabel('t')
plt.ylabel('y(t)')

plt.subplot(1, 2, 2)

plt.plot(wshift2, abs(yshift2), '-ob', label='w = 10')
plt.plot(wshift3, abs(yshift3), '-or', label='w = 20')
plt.xlim(-40, 40)
plt.ylabel(r'|$\tilde{y}$|')
plt.xlabel(r'$\omega$')
plt.legend(loc=7)
plt.show()

# ----------------------------------------------------------------------------//

# %%Question 3

# %%Q3

'''
Note: The prof said that if we know how to fix it then to implement the fixed
        version and this is the fixed version
'''

# Range from 0 to 8pi for n = 200
trng3 = np.linspace(0, 8*pi, 200)

# Get Noise Data
nst, nsy = callf(200, 3, 1, 1, 10, 0, 0, 8*pi)

# Get the DFT
fqr4, dft4 = dft(200, nsy)


def clean(fqr, dft):
    mx = abs(dft).max()
    found = False
    for i in range(len(fqr)):
        if (round(abs(dft[i])) != round(mx) or found):
            dft[i] = 0
    return dft

# Plot the figures -----------------------------------------------------------//


plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

plt.plot(nst, nsy, '-b')
plt.plot(nst, nsy, '--r', label='unfiltered')
plt.xlabel('t')
plt.ylabel('y')

plt.subplot(1, 2, 2)

plt.plot(fqr4/4, abs(dft4), '--r', label='unfiltered')
plt.plot(fqr4/4, abs(clean(fqr4, dft4)), '-b', label='filtered')
plt.ylabel(r'|$\tilde{y}$|')
plt.xlabel(r'$\omega$')
plt.legend(loc=9)

plt.subplot(1, 2, 1)

nyclean = invdft(200, clean(fqr4/4, dft4))
plt.plot(nst, nyclean, '-b', label='filtered')
plt.legend(loc=9)

plt.show()

# ----------------------------------------------------------------------------//

# %% End

print("***END OF PROGRAM***")
