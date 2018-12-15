# 测试函数
## from  https://www.sfu.ca/~ssurjano/optimization.html

import numpy as np
from numpy import pi, exp, cos, sin, abs, sqrt

## many local minimum
def schafferN2(x):
    """
    x_i in [-100, 100] for i = 1, 2
    f(x*) = 0, at x* = (0, 0)
    """
    a = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
    b = (1 + 0.001*(x[0]**2 + x[1]**2))**2
    return (0.5 + a / b)

def drop_wave(x):
    """
    x_i in [-5.12, 5.12] for i = 1, 2
    f(x*) = -1, at x* = (0, 0)
    """
    x1, x2 = x[0], x[1]
    a = 1 + cos(12 * sqrt(x1**2 + x2**2))
    b = 0.5 * (x1**2 + x2**2) + 2
    y = -a / b
    return y

def holder(x):
    """
    x_i in [-10, 10] for all i = 1, 2
    f(x*) = -19.2085, 
    at x* = (8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, 9.66459) and (-8.05502, -9.66459) 
    """
    x1, x2 = x[0], x[1]
    pp = abs(1 - (sqrt(x1**2 + x2**2)) / pi)
    y = -abs(sin(x1)*cos(x2)*exp(pp))
    return y

## bowl-shaped
def sumsquares(x):
    """
    x_i in [-10, 10] for all i = 1, ... , d
    f(x*) = 0, at x* = (0, ..., 0)
    """
    ll = len(x)
    coef = np.arange(1, ll + 1)
    y = np.sum(coef * (x**2))
    return y

## Plate-Shaped
def mccormick(x):
    """
    x1 in [-1.5, 4], x2 in [-3, 4]
    global minimum
    f(x*) = -1.9133, at x* = (-0.54719, -1.54719)
    """
    x1, x2 = x[0], x[1]
    y = sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1
    return y


## Steep Ridges/Drops
def easom(x):
    """
    x_i in [-100, 100] for i in 1, 2
    global minimum
    f(x*) = -1, at x* = (pi, pi)
    """
    x1, x2 = x[0], x[1]
    e12 = exp(-(x1 - pi)**2 - (x2 - pi)**2)
    return -cos(x1) * cos(x2) * e12
