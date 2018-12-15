w_start, w_end = 0.9, 0.4
T_max = 300

def func(k):
    return w_start * (w_start - w_end) * (T_max - k) / T_max

def func1(k):
    return w_start - (w_start - w_end) * (k/T_max)**2

def func2(k):
    return w_start + (w_start - w_end) * (2*k/T_max - (k/T_max)**2)

def func3(k):
    g = 1 / (1 + k/T_max)
    return w_end*(w_start / w_end) ** g


# import matplotlib.pyplot as plt

# plt.ion()
# ax = plt.figure().subplots(111)
# for f in [func, func1, func2, func3]:
#     x = list(range(300))
#     y = [f(xx) for xx in x]
#     ax.plot(x, y)

import numpy as np
from numpy import product, sin, pi

from pso import PSO

def  f(x):
    a = product(sin(x))
    b = sin(product(5*x))
    y = -5 * a - b + 8
    return y

bounds = np.array([[0, .9*pi]]*5)


"""
[1.61351085 1.56581549 1.57166843 1.5603979  1.5628975 ]
-2.00542260441725
[1.54417842 1.56347514 1.58807446 1.58898269 1.54758682]
-2.0049947856169528
[1.4612844  1.58294505 1.53470043 1.64235844 1.60295681]
-2.1130713295081787
[ 1.78605907  4.76624074  4.64407406 -1.6737616   4.75917089]
-2.17027418538433
"""
for vv in [0.2, 0.5, 0.8, 2.0, .9*pi]:
    pp = PSO(m=30, D=5, vmax=vv)
    pp.target = lambda x: -f(x)
    pp.initialize(bounds)
    x,y = pp.solve(max_iter=50, printInfo=True)
    print(x)
    print(y)