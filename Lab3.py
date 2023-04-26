# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:14:02 2022

"""

import time as t
import numpy as np
import timeit as ti
from timeit import default_timer


# %%Question 1

# %% Q1(a)

# matrix size so that you only change the number once
m = 4  # Will update throughout the code blocks as need

# Make a 2D array


def fillar(n): return np.arange(21, 21+n**2, 1).reshape(n, n)


print(fillar(m))
ar = np.array(fillar(m))
print()

# Lower and Upper diagonals --//
# Upper diagonal


def up(a, n):
    r = 1
    c = 0
    while r != n and c != n:
        a[r][c] = 0
        r += 1
        if (r == n):
            c += 1
            r = c+1
    return a


print("Upper Diagonal:")
print(up(ar, m))
print()

# Lower diagonal
ar = np.array(fillar(m))


def down(a, n):
    r = 0
    c = 1
    while r != n and c != n:
        a[r][c] = 0
        c += 1
        if (c == n):
            r += 1
            c = r+1
    return a


print("Lower Diagonal:")
print(down(ar, m))
print()
# -----------------------//

# Get the norm (ii) -----//
ar = np.array(fillar(m))
def norm(a, n): return np.sqrt(np.sum((np.ravel(a))**2))


print("My Norm Function:", norm(ar, m))

# Numpy Norm
ar = np.array(fillar(m))
print("Numpys Norm:", np.linalg.norm(ar))
print()
# -----------------------//

# Get the infinite norm (iii)


def infnorm(a, n): return np.sum(a[n-1])


print("My Infinite Norm Function:", infnorm(ar, m))

# Numpy Infinity Norm ----//
ar = np.array(fillar(m))
print("Numpys Infinite Norm:", np.linalg.norm(ar, np.inf))
print()
# -----------------------//

print("-------------------------------------------------------------------//")


# %% Q1(b)

# matrix size so that you only change the number once
m = 4

# The matrix with  a diagonal of ones and the upper diagoal is composed of -1's (i) --//


def uponeA(n):
    oar = np.ones((n, n))
    r = 0
    c = 0
    while r != n and c != n:
        oar[r][c] = -1*oar[r][c]
        r += 1
        c += 1
    return up(-1*oar, n)


print("1 and -1 matrix:")
print(uponeA(m))
print()
# -----------------------//

# Solve Ax = b (ii) -----//


def vectb(n):
    b = np.ones(n)
    b[1::2] = -1
    return b


b = vectb(m)
print("b vector:")
print(b)
print()

# Perturbes the matrix


def perturbe(a, n):
    a[n-1][0] = a[n-1][0] - 0.001
    return a

# Gets the condition number


def kappa(a, m): return norm(a, m)*norm(np.linalg.inv(a), m)


A2 = uponeA(m)  # No Perturbation
Ap2 = perturbe(uponeA(m), m)  # Perturbed

# This is x but I use soln so I don't get confused later
soln = np.linalg.solve(uponeA(m), b)  # No Perturbation
solnp = np.linalg.solve(perturbe(uponeA(m), m), b)  # Perturbed

print("For 4x4:")
print("x[0] w/ perturbation:", solnp[0])
print("x[0] no perturbation:", soln[0])
print("x[1] w/ perturbation:", solnp[1])
print("x[1] no perturbation:", soln[1])
print("x[2] w/ perturbation:", solnp[2])
print("x[2] no perturbation:", soln[2])
print("Condition Number Non-perturbed:", kappa(A2, m))
print("Condition Number Perturbed:", kappa(Ap2, m))
print()
# -----------------------//

# same as (ii) but for n=16 now (iii) --//

# matrix size so that you only change the number once
m = 16

A2 = uponeA(m)  # No Perturbation
Ap2 = perturbe(uponeA(m), m)  # Perturbed

# Gets the condition number


def kappa(a, m): return norm(a, m)*norm(np.linalg.inv(a), m)


b2 = vectb(m)
soln2 = np.linalg.solve(uponeA(m), b2)  # No Perturbation
solnp2 = np.linalg.solve(perturbe(uponeA(m), m), b2)  # Perturbed
print("For 16x16:")
print("x[0] w/ perturbation:", solnp2[0])
print("x[0] no perturbation:", soln2[0])
print("x[1] w/ perturbation:", solnp2[1])
print("x[1] no perturbation:", soln2[1])
print("x[2] w/ perturbation:", solnp2[2])
print("x[2] no perturbation:", soln2[2])
print("Condition Number Non-perturbed:", kappa(A2, m))
print("Condition Number Perturbed:", kappa(Ap2, m))
print()
# -----------------------//

print("-------------------------------------------------------------------//")


# %%Question2

# %%Q2(a)

# Note: The code takes longer to run on a laptop than on a tower due to undervolting on laptops

# matrix size so that you only change the number once
m = 5000
U = up(fillar(m), m)
bs = np.array([21+i for i in range(m)])

# Class backsub code


def backsub1(U, bs):
    n = bs.size
    xs = np.zeros(n)

    xs[n-1] = bs[n-1]/U[n-1, n-1]
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range(i+1, n):
            bb += U[i, j]*xs[j]
        xs[i] = (bs[i] - bb)/U[i, i]
    return xs


def testsolve(backsub1, Us, bs): return backsub1(Us, bs)


print(testsolve(backsub1, U, bs)[0:3])

timer_start = default_timer()
testsolve(backsub1, U, bs)
timer_end = default_timer()
time1 = timer_end-timer_start
print('time1:', time1)
print()

# This is being passed into the vectorized portion


def backsubsup(u, x, bb):
    bb = u*x
    return bb

# My backsub code


def backsub2(U, bs):
    n = bs.size
    xs = np.zeros(n)
    vect = np.vectorize(backsubsup)
    xs[n-1] = bs[n-1]/U[n-1, n-1]
    for i in range(n-2, -1, -1):
        bb = 0
        bb = np.sum(vect(U[i][n-1:i:-1], xs[n-1:i:-1], bb))
        xs[i] = (bs[i] - bb)/U[i, i]
    return xs


def mysolve(f, a, bs):
    xs = f(a, bs)
    print("My solution is:", xs[0:3])


def testsolve(backsub2, Us, bs): return backsub2(Us, bs)


mysolve(backsub2, U, bs)

timer_start = default_timer()
testsolve(backsub2, U, bs)
timer_end = default_timer()
time2 = timer_end-timer_start
print('time2:', time2)

print("-------------------------------------------------------------------//")


# %%Q2(b)

A3 = np.array([[2., 1., 1., 8.], [1., 1., -2., -2], [1., 2., 1., 2.]])
print(A3)
print()

# Inputting the concatonated matrix made it easier


def gauss1(a):
    n = len(a)
    for i in range(n):
        for r in range(i+1, n):
            const = a[r][i]/a[i][i]
            for c in range(n+1):  # If the matrix isn't preconcatonated then do n not n+1
                a[r][c] = a[r][c] - const*a[i][c]
    return a


print("Concatonated matrix found through gaussian elimination:")
print(gauss1(A3))
print()
print("Solution:")
print(backsub2(gauss1(A3)[0:, 0:-1], gauss1(A3)[0:, 3]))  # Solve w/ back solve

print("-------------------------------------------------------------------//")


# %%Q2(c)

A4 = np.array([[2., 1., 1.], [2., 1., -4.], [1., 2., 1.]])
b = np.array([8., -2., 2.])
print(A4)
print()


def gausspsol(a, b):
    n = len(b)
    for i in range(n-1):  # Main iterator
        # Everything in if statement is to search and swap elements
        if (np.abs(a[i][i]) < 1.0e-10):
            for l in range(i+1, n):
                if (np.abs(a[l][i]) > np.abs(a[i][i])):
                    a[[i, l]] = a[[l, i]]
                    b[[i, l]] = b[[l, i]]
                    break
        for r in range(i+1, n):  # Rows loop
            if (np.abs(a[r, i] == 0)):
                continue
            const = a[i][i]/a[r][i]
            for c in range(i, n):  # Cols loop
                a[r][c] = a[i][c] - const*a[r][c]
            b[r] = b[i] - const*b[r]
    return a, b


print("Matrix found through gaussian elimination:")
print(gausspsol(A4, b)[0])
print()
print("b vector found through gaussian elimination:")
print(gausspsol(A4, b)[1])
print()
print("Solution:")
# Solve w/ back solve
print(backsub2(gausspsol(A4, b)[0], gausspsol(A4, b)[1]))

print("-------------------------------------------------------------------//")


# %%Q2(d)

A5 = np.array([[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]])
print(A5)
print()


def gauss1(a):
    n = len(a)
    for i in range(n):
        for r in range(i+1, n):
            const = a[r][i]/a[i][i]
            for c in range(n):
                a[r][c] = a[r][c] - const*a[i][c]
    return a


# Getting the inverse matrix cols
b1 = np.array([1., 0., 0.])
b2 = np.array([0., 1., 0.])
b3 = np.array([0., 0., 1.])
# Cols of the identity matrix
c1 = backsub2(gauss1(A5), b1)
c2 = backsub2(gauss1(A5), b2)
c3 = backsub2(gauss1(A5), b3)
A5inv = np.vstack((c1, c2, c3)).T  # Putting the cols into a matrix
print("Inverted Matrix:")
print(A5inv)  # The inverted matrix
print()
print("Product of the original matrix and the found inverted matrix:")
print(np.dot(A5, A5inv))  # Shows that inverting the matrix was a success

print("-------------------------------------------------------------------//")
# %%
print()
print("***END OF PROGRAM***")
