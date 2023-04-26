# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:49:19 2022

"""

import scipy.optimize as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# %%Question 1

# %%Q1(a)

plt.rcParams.update({'font.size': 18})  # keep those graph fonts readable!
plt.rcParams['figure.dpi'] = 120  # plot resolution

# similar to Matlab's fval function - allows one to pass a function


def feval(funcName, *args):
    return eval(funcName)(*args)

# vectorized forward Euler with 1d numpy arrays


def euler(f, y0, t, h):  # Vectorized forward Euler (so no need to loop)
    k1 = h*f(y0, t)
    y1 = y0+k1
    return y1

# stepper function for integrating ODE solver over some time array


def odestepper(odesolver, deriv, y0, t):
    # simple np array
    y0 = np.asarray(y0)  # convret just in case a numpy was not sent
    y = np.zeros((t.size, y0.size))
    y[0, :] = y0
    h = t[1]-t[0]
    y_next = y0  # initial conditions

    for i in range(1, len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1], h)
        y[i, :] = y_next
    return y

# vectorized backward Euler with 1d numpy arrays


def eulerback(f, y0, t, h): return sp.fsolve(lambda y1: y0-y1+h*f(y1, t), y0)


def fun(y, t):
    return -10.*y


def funexact(t):
    return np.exp(-10*t)


# Plot the graphs ------------------------------------------------------------//
a, b, n1, y0 = 0., 0.6, 10, 1.
ts1 = a+np.arange(n1)/(n1-1)*b
y1 = odestepper('euler', fun, y0, ts1)
y2 = funexact(ts1)
y3 = odestepper('eulerback', fun, y0, ts1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

plt.plot(ts1, y2, '-r', label='Exact', linewidth=3)
plt.plot(ts1, y1, 'gs', label='F-Euler $n={}$'.format(n1), markersize=4)
plt.plot(ts1, y3, 'bo', label='B-Euler $n={}$'.format(n1), markersize=4)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.xlim(0, b)
plt.ylim(0, 1.04)
plt.legend(loc='best')

plt.subplot(1, 2, 2)

n2 = 20
ts2 = a+np.arange(n2)/(n2-1)*b
y4 = odestepper('euler', fun, y0, ts2)
y5 = funexact(ts2)
y6 = odestepper('eulerback', fun, y0, ts2)

plt.plot(ts2, y5, '-r', label='Exact', linewidth=3)
plt.plot(ts2, y4, 'gs', label='F-Euler $n={}$'.format(n2), markersize=4)
plt.plot(ts2, y6, 'bo', label='B-Euler $n={}$'.format(n2), markersize=4)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.xlim(0, b)
plt.ylim(0, 1.04)
plt.legend(loc='best')

# ----------------------------------------------------------------------------//

# %%Q1(b)

# Derivative


def dy1(y, t): return np.array([y[1], -y[0]])

# The actual answer


def factual(t): return np.cos(t)


def rk4(f, y, t, h):
    k0 = h*f(y, t)
    k1 = h*f(y + k0/2, t+h/2)
    k2 = h*f(y + k1/2, t+h/2)
    k3 = h*f(y + k2, t+h)
    k = (k0 + 2*k1 + 2*k2 + k3)/6
    return y+k


# initial condition array
y0 = np.array([1, 0])

# time array
tsa = np.array([np.arange(0, 20*np.pi, 0.01), np.arange(0, 20*np.pi, 0.005)])

# Plot the graphs ------------------------------------------------------------//
for i in range(2):
    y1 = odestepper("rk4", dy1, y0, tsa[i])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    yf = odestepper("euler", dy1, y0, tsa[i])
    plt.plot(yf[:, 1], yf[:, 0], 'r--', label='F-Euler')

    plt.plot(y1[:, 1], y1[:, 0], '-b', label='RK4')
    plt.ylabel('x')
    plt.xlabel('v')
    if (i == 0):
        plt.title('dt = 0.01')
    else:
        plt.title('dt = 0.005')

    plt.legend(loc='best')

    plt.subplot(1, 2, 2)

    plt.plot(tsa[i], y1[:, 0], '-b', label='RK4')
    plt.plot(tsa[i], yf[:, 0], 'r--', label='F-Euler')
    plt.plot(tsa[i], factual(tsa[i]), 'y--', label='Exact')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.show()

# ----------------------------------------------------------------------------//

# %%Question 2

# %%Q2(a)

# Define variables
omega = 1
alpha = 0
beta = 1
gamma = 0.04
F = 0.2

# Derivative


def dy2(y, t): return np.array(
    [y[1], -2*gamma*y[1]-alpha*y[0]-beta*y[0]**3+F*np.cos(t)])


# initial condition array
y0 = np.array([-0.1, 0.1])

# time array
ts2 = np.arange(0, 80*np.pi, 0.01)

# Plot the graphs ------------------------------------------------------------//
y = odestepper("rk4", dy2, y0, ts2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

plt.plot(y[round(len(ts2)/4):, 1], y[round(len(ts2)/4):, 0], 'r-')
plt.xlabel('v')
plt.ylabel('x')
plt.title(r"D Osc: $\alpha, F, \omega = {},{},{}$".format(alpha, F, omega))
plt.plot(y[0, 1], y[0, 0], 'bo', ms=10)
plt.plot(y[-1, 1], y[-1, 0], 'go', ms=10)

plt.subplot(1, 2, 2)

plt.plot(ts2/2/np.pi, y[:, 0], 'b-')
plt.xlabel('t')
plt.show()

# ----------------------------------------------------------------------------//

# %% Q2(b)

# Derivative


def dy3(y, t): return np.array(
    [y[1], -0.08*y[1]-0.1*y[0]-y[0]**3+7.5*np.cos(t)])


# Define variables
alpha = 0.1
F = 7.5

# initial condition array
y0 = np.array([-0.1, 0.1])

# time array
ts3 = np.arange(0, 80*np.pi, 0.01)

# Plot the graphs ------------------------------------------------------------//
y = odestepper("rk4", dy3, y0, ts2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

plt.plot(y[round(len(ts3)/4):, 1], y[round(len(ts3)/4):, 0], 'r-')
plt.xlabel('v')
plt.ylabel('x')
plt.title(r"D Osc: $\alpha, F, \omega = {},{},{}$".format(alpha, F, omega))
plt.plot(y[0, 1], y[0, 0], 'bo', ms=10)
plt.plot(y[-1, 1], y[-1, 0], 'go', ms=10)

plt.subplot(1, 2, 2)

plt.plot(ts3/2/np.pi, y[:, 0], 'b-')
plt.xlabel('t')
plt.show()

# ----------------------------------------------------------------------------//

# %%Question 3

# %% Q3(a)

# Define variables
rho = 10
r = 28
b = 8/3

# lorenz attractor equation


def lorz(y, t): return np.array(
    [rho*(y[1]-y[0]), r*y[0]-y[1]-y[0]*y[2], y[0]*y[1]-b*y[2]])


# time array
ts3 = np.arange(0, 8*np.pi, 0.01)

# initial condition array
y01 = np.array([1, 1, 1])
y1 = odestepper("rk4", lorz, y01, ts3)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=20, elev=29)

ax.view_init(elev=29, azim=20)
ax.plot3D(y1[:, 0], y1[:, 1], y1[:, 2], '-',
          color='orange', label='y0 = [1,1,1]')

# initial condition array
# y02 = np.array([10, 10, 10])
y02 = np.array([1, 1, 1.001])
y2 = odestepper("rk4", lorz, y02, ts3)

ax.plot3D(y2[:, 0], y2[:, 1], y2[:, 2], 'b-', label='y0 = [1, 1, 1.001]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorentz Attractor')
plt.legend(loc='best')
plt.show()


# %% Q3(b)

x11 = np.array(y1[:, 0])
y11 = np.array(y1[:, 1])
z11 = np.array(y1[:, 2])
x22 = np.array(y2[:, 0])
y22 = np.array(y2[:, 1])
z22 = np.array(y2[:, 2])

time_vals = ts3
plt.rcParams.update({'font.size': 18})
fig = plt.figure(dpi=180)
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85], projection='3d')
line, = ax.plot3D(x11, y11, z11, 'r-', linewidth=0.8)
line2, = ax.plot3D(x22, y22, z22, 'b-', linewidth=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


def init():
    line.set_data(np.array([]), np.array([]))
    line.set_3d_properties([])
    line.axes.axis([-25, 25, -25, 25])
    line2.set_data(np.array([]), np.array([]))
    line2.set_3d_properties([])
    line2.axes.axis([-25, 25, -25, 25])
    return line, line2


def update(num):
    line.set_data(x11[:num], y11[:num])
    line.set_3d_properties(z11[:num])
    line2.set_data(x22[:num], y22[:num])
    line2.set_3d_properties(z22[:num])
    fig.canvas.draw()
    return line, line2


ani = animation.FuncAnimation(fig, update, init_func=init, interval=1, frames=len(
    time_vals), blit=True, repeat=True)
plt.show()


# %% End

print("***END OF PROGRAM***")
