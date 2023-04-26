# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:02:03 2022

"""

import time
import numpy as np
import math as m
import matplotlib.pyplot as plt
import mpmath as mp
import scipy as sc
from scipy.integrate import dblquad
from numpy import pi
from numpy import exp
from numpy import sqrt
from numpy import sin
from numpy import cos
from mpmath import quad
import pandas as pd

mainstart = time.time()
# %%Question 1

# %%Q1(a)

# Code from the notes ----------------------------------//


def notes(a, b, n):
    h = (b-a)/(n-1)
    int = 0.  # Init the integral variable
    for i in range(0, n):
        xi = a+h*i  # Determine teh xi value for the loop
        int = int + (2/pi**0.5)*exp(-xi**2)*h
    return int
# ------------------------------------------------------//


# Now the fixed code -----------------------------------//
def erf(x): return (2/sqrt(pi))*(exp(-x**2))  # The error function


def rect(f, a, b, n):  # Rectangular approximation
    val = 0
    dx = (b-a)/n  # differential step dx
    for i in range(0, n):
        val = val + f(a+(i)*dx)*dx  # Calcs the integral
    return val
# ------------------------------------------------------//

# %% Q1(b)

# Calcs the percent relative error


def geterrorRel(v): return (abs((v - m.erf(1))/m.erf(1)))


print("Notes Rectangular Approx (n=100):", notes(0, 1, 100))
print("Notes Relative Error %:", geterrorRel(notes(0, 1, 100)), "%")
print("Rectangular Approx (n=101):", rect(erf, 0, 1, 101))
print("Relative Error % Rect(n=101):", geterrorRel(rect(erf, 0, 1, 101)), "%")
print("------------------------------------------------------------------------")


def trapez(f, a, b, n):
    valt = 0
    dx = (b-a)/n  # differential step dx
    for i in range(0, n):
        valt = valt + 2*f(a+(i+0.5)*dx)  # Calcs the integral
    return (dx/2)*valt


print("Trapezoidal Approx (n=100):", trapez(erf, 0, 1, 100))
print("Relative Error % Trapez(n=100):",
      geterrorRel(trapez(erf, 0, 1, 100)), "%")
print("Trapezoidal Approx (n=101):", trapez(erf, 0, 1, 101))
print("Relative Error % Trapez(n=101):",
      geterrorRel(trapez(erf, 0, 1, 101)), "%")
print("------------------------------------------------------------------------")
# ------------------------------------------------------//

# This function fills the weighting array/vector for the integration


def fillc(n):
    c = list(range(0, n))
    c[0] = 1
    for i in range(1, n, 2):
        c[i] = 4
    for i in range(2, n-1, 2):
        c[i] = 2
    c[n-1] = 1
    return c

# The simpson approx ------------------------------------//


def simpson(f, a, b, n):
    vals = 0
    dx = (b-a)/(n-1)  # differential step
    c = fillc(n)
    c = np.array(c)
    c = (dx/3)*c  # c is the weighted array/vector
    for i in range(0, n):
        vals = vals + (c[i])*(erf(a+i*dx))  # Calcs the integral
    return vals


print("Simpson's Approx (n=100):", simpson(erf, 0, 1, 100))
print("Relative Error % Simpson(n=100):",
      geterrorRel(simpson(erf, 0, 1, 100)), "%")
print("Simpson's Approx (n=101):", simpson(erf, 0, 1, 101))
print("Relative Error % Simpson(n=101):",
      geterrorRel(simpson(erf, 0, 1, 101)), "%")
print()
print("Actual:", m.erf(1))
print("------------------------------------------------------------------------")
# ------------------------------------------------------//

# Discussing the error ---------------------------------//
'''
For all Approximations:
    The error decreases as the size of n increase as expected which can be seen by comparing the error 
    of n=100 vs the error of n=101 and seeing that the error of n=101 is smaller than that of n=100.

Rectangular Approximation Error:
    Looking at the error from both the n=100 and the n=101 approx we can see that the approx is 
    an overestimate for this particular integral. This is the worst of the three approximations.
    
Trapezoidal Approximation Error:
    Looking at the error from both teh n=100 and the n=101 approx we can see that the approx is 
    an overestimate for this particular integral. This approximation is significantly
    better than the rectagular but not as good as simpsons
    
Simpsons Approximation Error:
    This is the best approximation compared to the others, but only for when n is odd if n is even then
    it gives a worse approximation, however, this can be remideied easily by adding an if statement
    and shifting n accordingly such that the number of windows isnt altered
'''
# ------------------------------------------------------//

# Making the sub routines more robust ------------------//

# Here we are making the subroutine better by altering definitions related to n so that the function
# returns a good approximation with low error for both even and odd step sizes


def simpsonImp(f, a, b, n):
    vals = 0
    # dx is the differential step
    if (n % 2 != 0):
        dx = (b-a)/(n-1)  # If odd
    else:  # If even
        dx = (b-a)/(n)
        n = n+1
    c = fillc(n)
    c = np.array(c)
    c = (dx/3)*c  # c is the weighted array/vector
    for i in range(0, n):
        vals = vals + (c[i])*(erf(a+i*dx))  # Calcs the integral
    return vals


print("Improved Simpson's Approx (n=100):", simpsonImp(erf, 0, 1, 100))
print("Relative Error % Simpson(n=100):",
      geterrorRel(simpsonImp(erf, 0, 1, 100)), "%")
print("Improved Simpson's Approx (n=101):", simpsonImp(erf, 0, 1, 101))
print("Relative Error % Simpson(n=101):",
      geterrorRel(simpsonImp(erf, 0, 1, 101)), "%")
print("------------------------------------------------------------------------")
# ------------------------------------------------------//

# %% Q1(c)


def epsN(f, fi, a, b, n):  # Calcs adaptive step error
    const = 1
    if f == simpsonImp:
        const = 15
    elif f == trapez:
        const = 3
    return abs((f(fi, a, b, 2*n-1) - f(fi, a, b, n))/const)


def adaptive_step(f, fi, a, b, n):
    itr = 1  # Declare iterator for counting
    if n % 2 == 0:
        n = n+1  # There is already a condition in simpsonImp that checks is a number is even or odd but this is just to guarantee that the counting and iterations are accurate
    while (epsN(f, fi, a, b, n) >= 10**-13):
        itr = itr + 1
        n = 2*n - 1  # Updtae n and itr until error is matched or lower
    # Return the threeplue of (area, step size, and iterations)
    return (f(fi, a, b, n), n, itr)


# defining ads and adt makes the program run faster since there are less calculations performed
# The calculations for ads and adt (specifically adt) take a few seconds, roughly 10ish seconds
ads = adaptive_step(simpsonImp, erf, 0, 1, 3)  # Adaptive Step Simpson threepul
adt = adaptive_step(trapez, erf, 0, 1, 3)  # Adaptive Step Trapezoidal threepul

print("Adaptive Steps Simpson:", ads[0])
print("Adaptive Steps Simpson Steps: n =", ads[1])
print("Adaptive Steps Simpson Iterations:", ads[2])
print("Adaptive Steps Simpson Error:", epsN(simpsonImp, erf, 0, 1, ads[1]))
print()
print("Adaptive Steps Trapezoidal:", adt[0])
print("Adaptive Steps Trapezoidal Steps: n =", adt[1])
print("Adaptive Steps Trapezoidal Iterations:", adt[2])
print("Adaptive Steps Trapezoidal Error:", epsN(trapez, erf, 0, 1, adt[1]))
print("------------------------------------------------------------------------")

# %%Question 2

# %% Q2(a)

# Calcs error difference


def geterror(v): return (m.erf(1) - v)


# Reading the data for an excel file
data = pd.read_csv("Hysteresis-Data.csv", usecols=['vx', 'vy'])
vx = np.array(data["vx"])
vy = np.array(data["vy"])

# Plot vy vs vx
plt.plot(vx, vy, 'b')
plt.ylabel("$V_y$", fontsize=10)
plt.xlabel("$V_x$", fontsize=10)
plt.title("Plot $V_y$ vs $V_x$")
plt.xlim()
plt.ylim()
plt.grid()
plt.show()

# %% Q2(b)

# Area enclosed in the curve


def integral(x, y): return np.sum(((y[1:] - y[:-1])*(x[1:] + x[:-1]))/2)


print("Area between Hysteresis Curves:", integral(vx, vy))
print("------------------------------------------------------------------------")

# %%Question 3

# %% Q3(a)

# The 2D simpsons subroutine


def simp2d(f, a, b, c, d, n, m):
    # if else to make sure that it takes both even and odd step sizes
    if (n % 2 and m % 2 != 0):  # If odd
        dx = (b-a)/(n-1)
        dy = (d-c)/(m-1)  # dx and dy are the differential steps
    else:  # if even
        dx = (b-a)/(n)
        dy = (d-c)/(m)
        n = n+1
        m = m+1
    x = np.linspace(a, b, n)
    y = np.linspace(c, d, m)  # generate the x and y for the given step sizes
    cx = fillc(n)
    cx = np.array(cx)
    cx = (dx/3)*cx  # generate the weigthted x and y arrays; fillc() above
    cy = fillc(m)
    cy = np.array(cy)
    cy = (dy/3)*cy  # ^^^^^
    cs = np.outer(cx, cy)  # Compute the outer product based on cx and cy
    # Compute the meshgrid based on x and y
    xm, ym = np.meshgrid(x, y, indexing='ij')
    return np.sum(cs*f(xm, ym))  # Returns the actual integartion

# %% Q3(b)


f2d = np.vectorize(lambda x, y: sqrt(x**2 + y)*sin(x) *
                   cos(y))  # The function to be integrated

print("For n = 101 and m = 101:", simp2d(f2d, 0, pi, 0, pi/2, 101, 101))
print("For n = 101 and m = 101:", simp2d(f2d, 0, pi, 0, pi/2, 1001, 1001))
print("For n = 51 and m = 101:", simp2d(f2d, 0, pi, 0, pi/2, 51, 101))
print("------------------------------------------------------------------------")

# %% Q3(c)


def f2d2(x, y): return mp.sqrt(x**2 + y)*mp.sin(x) * \
    mp.cos(y)  # Same as f2d just not vectorized


# Lambda function for the integral
def fq(f, a, b, c, d): return quad(f, [a, b], [c, d])


# The area returned from the quad function
print("Value calculated with quad():", fq(f2d2, 0, mp.pi, 0, mp.pi/2))
print("------------------------------------------------------------------------")

# %% Q3(d)

# Prints the tuple of the area and the error
print("Tuple of Area and Error Calcualted with dblquad():",
      dblquad(f2d2, 0, pi/2, 0, pi))
print("------------------------------------------------------------------------")

print("***END OF THE PROGRAM***")
print()
mainend = time.time()
print("Run-time:", mainend-mainstart)
