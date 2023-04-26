# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:49:41 2022

"""

import numpy as np
from numpy import linalg as lg
import sys
import time
import matplotlib.pyplot as plt

# %%Question 1

# %%Q1

# define values
alpha = 2.3E-4  # m^2/s
length = 1  # m
n = 100
dt = 0.1  # temporal change [s]
dx = length/n  # spacial change [m]
ue = 20  # hard wall boundry condition, for all times at the start and end of the rod

# time array; from 0 to 61 for dt
ts = np.arange(0, 61, dt)

# space array; for the length of the rod
sg = np.arange(0, length, dx)

# taken from the prev lab


def feval(funcName, *args): return eval(funcName)(*args)

# initial temp


# t = 0 and units in deg C
def initU(x): return 20 + 30*np.exp(-100*(x-0.5)**2)

# 1D heat equation


def heat(alpha, n, length, ts, dt, dx, bc, u0):
    k = alpha*dt/(dx**2)
    if (k > 0.5):
        sys.exit("Caurant condition not met, k > 0.5")
    u0 = feval(u0, sg)  # initial value
    u = np.zeros((len(ts), len(sg)))
    u[0, 1:-1] = u0[1:-1]  # populate with initial conditions
    u[0, 0] = 20
    u[0, len(sg)-1] = 20
    for i in range(1, len(ts)):
        u[i, 0] = 20
        u[i, len(sg)-1] = 20
        u[i, 1:-1] = u[i-1, 1:-1] + k * \
            (u[i-1, 2:] - 2*u[i-1, 1:-1] + u[i-1, :-2])
    return u


# Plot the graphs ------------------------------------------------------------//
heatdata = heat(alpha, n, length, ts, dt, dx, ue, 'initU')
snpashots = np.array([0, 5, 10, 20, 30, 60])
snpashots = snpashots/dt
plt.figure(figsize=(12, 6), dpi=180)
for i in snpashots:
    plt.plot(sg, heatdata[int(i), :], label="time = {}".format(i/10))
plt.xlabel("Rod Length (m)")
plt.ylabel(r"Temperature (C)")
plt.ylim(19,)
plt.legend()
plt.show()
# ----------------------------------------------------------------------------//

# %%Question 2

# %%Q2

# deifine values
xs = np.linspace(0, 2, 100)  # x array
ys = np.linspace(0, 1, 50)  # y array
tol = 10**-5
h = 1/50  # hx = hy
phi0 = 0  # boundry conditions

# forcing function


def ff(x, y): return np.cos(10*x) - np.sin(5*y - (np.pi/4))

# Relaxation Method: Jacobian Iteration


def jacrex(xs, ys, tol, h):
    phi = np.zeros((len(ys), len(xs)))
    X, Y = np.meshgrid(xs, ys)
    ffv = ff(X, Y)  # forcing matrix for the forcing function using the meshgrid
    dnorm = 1  # to enter loop
    # Initial guess will be a matrix of all zeros so in this case we dont need to
    # update the boundry conditions since they are already zero
    while (dnorm > tol):
        normprev = lg.norm(phi)
        phi[1:-1, 1:-1] = (1/4)*(phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:]
                                 + phi[1:-1, :-2]) - (1/4)*(h**2)*ffv[1:-1, 1:-1]
        normnew = lg.norm(phi)
        dnorm = (normnew - normprev)/normnew
    return phi


phidata = jacrex(xs, ys, tol, h)  # get the phi values

# Plot the graphs ------------------------------------------------------------//
plt.figure(figsize=(12, 6), dpi=180)
plt.imshow(phidata, cmap='hot', origin='lower', extent=[0, 2, 0, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='$\phi$(x,y)')
plt.show()
# ----------------------------------------------------------------------------//

# %%Question 3

# %%Q3

# define values
n = 800
xs2 = np.linspace(0, 2*np.pi, n)  # x array
ys2 = np.linspace(0, 2*np.pi, n)  # y array
h = 2*np.pi/n  # dx = dy = h

# forcing function


def ff2(x, y): return np.cos(3*x + 4*y) - np.cos(5*x - 2*y)

# equation for phi-tilde


def phitildefunc(f, k, l, n): return (1/2)*(((h**2)*f) /
                                            (np.cos(2*np.pi*k/n) + np.cos(2*np.pi*l/n) - 2))


def pdft(x, y, n):
    X2, Y2 = np.meshgrid(x, y)  # define the meshgrid
    ff2v = ff2(X2, Y2)  # get the values of the forcing function on the mesh
    ftilde = np.fft.fft2(ff2v)  # fourier transform the forcing function
    kl = np.linspace(0, n-1, n)
    kl[0] = 1  # making k[0] = 1 to get past zero division error
    K, L = np.meshgrid(kl, kl)  # the k and l index meshgrid
    # initialize the phitilde data matrix
    phitilde = np.zeros((len(X2), len(Y2)))
    phitilde = phitildefunc(ftilde, K, L, n)  # calc the phitilde data
    phitilde[0, 0] = 0
    phitilde[-1, -1] = 0  # make the (0,0) and (2pi,2pi) points zero
    phi = np.fft.ifft2(phitilde)  # inverse fourier transform
    return np.flip(phi, axis=0)


Q3start = time.time()  # Start counting time
phidata2 = pdft(xs2, ys2, n)  # get the data values of the phi data
Q3end = time.time()  # End counting
print("Function Run-time:", Q3end-Q3start)

# Plot the graphs ------------------------------------------------------------//
plt.figure(figsize=(12, 6), dpi=180)
plt.imshow(np.real(phidata2), cmap='hot', origin='lower', extent=[0, 2, 0, 2])
plt.xlabel('x($\pi$)')
plt.ylabel('y($\pi$)')
plt.colorbar(label='$\phi$(x,y)')
plt.show()
# ----------------------------------------------------------------------------//

# %% End
print("----------------------------------------------------------------------//")
print()
print("***END OF PROGRAM***")
