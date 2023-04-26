# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:32:02 2022

"""

import numpy as np
import random as rn
import matplotlib.pyplot as plt
import matplotlib.colors as pc

# %%Question 1

# %%Q1

# number of random points
n = 1000

# function for a circle


def circle(r, n):
    theta = np.linspace(0, 2*np.pi, n)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return (x, y)

# returns the approx value of pi and the array of random points


def getpi(r, n):
    # generate an array of n random points on [-0.5,0.5]
    randpts = np.zeros((n, 2))
    for i in range(n):
        randpts[i, 0] = rn.uniform(-0.5, 0.5)  # x points
        randpts[i, 1] = rn.uniform(-0.5, 0.5)  # y points

    # array of the radial position of the random point
    radialpos = (randpts[:, 0]**2) + (randpts[:, 1]**2)
    incirclecount = 0
    # determine how many random points are in the w/in the bounds of the circle
    for i in range(n):
        if (radialpos[i] <= (r**2)):
            incirclecount = incirclecount + 1
    return (randpts, (4*(incirclecount/n)))


# Plot the graphs ------------------------------------------------------------//
fig, ax1 = plt.subplots(2, 2, figsize=(8, 8))

xc1, yc1 = circle(0.5, n)
rand1, mcpi1 = getpi(0.5, n)

ax1[0, 0].plot(xc1, yc1, 'b')  # plot the circle
ax1[0, 0].fill_between(xc1, yc1, color='b')
ax1[0, 0].plot(rand1[:, 0], rand1[:, 1], 'or', ms=3)  # plot the random points
ax1[0, 0].title.set_text(r'My pi = {}, n = {}'.format(mcpi1, n))

xc2, yc2 = circle(0.5, n)
rand2, mcpi2 = getpi(0.5, n)

ax1[1, 0].plot(xc2, yc2, 'b')  # plot the circle
ax1[1, 0].fill_between(xc2, yc2, color='b')
ax1[1, 0].plot(rand2[:, 0], rand2[:, 1], 'or', ms=3)  # plot the random points
ax1[1, 0].title.set_text(r'My pi = {}, n = {}'.format(mcpi2, n))

xc3, yc3 = circle(0.5, n)
rand3, mcpi3 = getpi(0.5, n)

ax1[0, 1].plot(xc3, yc3, 'b')  # plot the circle
ax1[0, 1].fill_between(xc3, yc3, color='b')
ax1[0, 1].plot(rand3[:, 0], rand3[:, 1], 'or', ms=3)  # plot the random points
ax1[0, 1].title.set_text(r'My pi = {}, n = {}'.format(mcpi3, n))

xc4, yc4 = circle(0.5, n)
rand4, mcpi4 = getpi(0.5, n)

ax1[1, 1].plot(xc4, yc4, 'b')  # plot the circle
ax1[1, 1].fill_between(xc4, yc4, color='b')
ax1[1, 1].plot(rand4[:, 0], rand4[:, 1], 'or', ms=3)  # plot the random points
ax1[1, 1].title.set_text(r'My pi = {}, n = {}'.format(mcpi4, n))
plt.show()
# ----------------------------------------------------------------------------//

# %%Question 2

# %%Q2

# total spin chain
n = 50

# Copyed this from the notes ------------------------------------------------//
# initialization


def initialize(n, p):
    spin = np.ones(n)
    E = 0.0
    M = 0.0
    for i in range(1, n):
        if (np.random.rand(1) < p):
            spin[i] = -1
        E = E - spin[i-1]*spin[i]  # Energy
        M = M + spin[i]  # Magnetization
    E = E - spin[n-1]*spin[0]
    M = M + spin[0]
    return spin, E, M

# update


def update(n, spin, kT, E, M):
    num = np.random.randint(0, n-1)
    flip = 0
    # periodic bc returns 0 if i + 1 == N , else no change :
    dE = 2*spin[num]*(spin[num-1]+spin[(num+1) % n])
    # if dE is negative, accept flip :
    if (dE < 0):
        flip = 1
    else:
        p = np.exp(-dE/kT)
        if np.random.rand(1) < p:
            flip = 1
        # otherwise, reject flip
    if (flip == 1):
        E += dE
        M -= 2*spin[num]
        spin[num] = -spin[num]
    return E, M, spin

# ----------------------------------------------------------------------------//

# metropolis algorithm


def metro(n, kT, itr, init, p):
    sd = np.zeros((itr, n))
    ed = np.zeros(itr)
    md = np.zeros(itr)
    sd[0, :], ed[0], md[0] = initialize(50, p)
    for i in range(1, itr):
        ed[i], md[i], sd[i, :] = update(n, sd[i-1], kT, ed[i-1], md[i-1])
    return sd, ed, md

# Plot the graphs ------------------------------------------------------------//


# colour scheme
yp = pc.ListedColormap(['yellow', 'purple'])

# prob array
prob = np.array([0.2, 0.6])
percentage = np.linspace(0, 100, n*800)

for i in range(2):

    init = initialize(n, prob[i])  # initial data for first series of plots

    fig2, ax2 = plt.subplots(3, 1, dpi=500, sharex='col', figsize=(24, 6))
    s1, e1, m1 = metro(n, 0.1, n*800, init, prob[i])
    ax2[0].imshow(s1.T, cmap=yp, origin='lower', extent=[0, 100, 0, 50])
    ax2[0].title.set_text('Spin Evolution with KT = {}'.format(0.1))

    s2, e2, m2 = metro(n, 0.5, n*800, init, prob[i])
    ax2[1].imshow(s2.T, cmap=yp, origin='lower', extent=[0, 100, 0, 50])
    ax2[1].title.set_text('Spin Evolution with KT = {}'.format(0.5))

    s3, e3, m3 = metro(n, 1, n*800, init, prob[i])
    ax2[2].imshow(s3.T, cmap=yp, origin='lower', extent=[0, 100, 0, 50])
    ax2[2].title.set_text('Spin Evolution with KT = {}'.format(1))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.8, hspace=0.8)
    plt.setp(ax2[-1], xlabel='Iteration/N')
    plt.setp(ax2[:], ylabel='N spins')

    plt.suptitle('P = {}'.format(prob[i]))

    plt.show()

    fig3, ax3 = plt.subplots(3, 1, dpi=200, sharex='col')

    ax3[0].plot(percentage, e1/n, 'r')
    ax3[0].plot(percentage, np.ones(n*800)*(sum(e1)/n)/len(e1), 'b')
    ax3[0].title.set_text(
        'Energy Evolution to Equilibrium to KT = {}'.format(0.1))

    ax3[1].plot(percentage, e2/n, 'r')
    ax3[1].plot(percentage, np.ones(n*800)*(sum(e2)/n)/len(e2), 'b')
    ax3[1].title.set_text(
        'Energy Evolution to Equilibrium to KT = {}'.format(0.5))

    ax3[2].plot(percentage, e3/n, 'r')
    ax3[2].plot(percentage, np.ones(n*800)*(sum(e3)/n)/len(e3), 'b')
    ax3[2].title.set_text(
        'Energy Evolution to Equilibrium to KT = {}'.format(1))

    plt.setp(ax3[:], ylabel='Energy/N$\epsilon$')
    plt.setp(ax3[-1], xlabel='Iteration/N')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.8, hspace=0.5)

    plt.suptitle("P = {}".format(prob[i]))

    plt.show()
# ----------------------------------------------------------------------------//

'''
Looking at both sets of graphs we can see that a cold start reaches an equilibrium
more quickly than compared to a hot start
'''

# %%Question 3

# %%Q3

n = 50
init = initialize(n, 0.6)
itr = n*800

# prediction function


def predict(kT): return np.tanh(1/kT)*-1


kbt = np.linspace(0.1, 6, 100)
avge = np.zeros(100)
avgm = np.zeros(100)

for i in range(100):
    e, m = (metro(n, kbt[i], itr, init, 0.6))[1:]
    avge[i] = (sum(e[20000:]))/(20000*n)
    avgm[i] = (sum(m[20000:]))/(20000*n)

# Plot the graphs ------------------------------------------------------------//
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(1, 2, 1)

prede = predict(kbt)
plt.scatter(kbt, avge, color='b')
plt.plot(kbt, prede, 'r')
plt.xlabel('kT/$\epsilon$')
plt.ylabel('<E>/N$\epsilon$')

plt.subplot(1, 2, 2)

predm = np.zeros_like(kbt)
plt.scatter(kbt, avgm, color='b')
plt.plot(kbt, predm, 'r')
plt.xlabel('kT/$\epsilon$')
plt.ylabel('<M>/N')
plt.show()
# ----------------------------------------------------------------------------//

# %% End
print("***END OF PROGRAM***")
