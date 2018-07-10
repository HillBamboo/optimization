import numpy as np
from testfucntions import schafferN2

fitness = lambda x: -x * (x - 30)



schaffer = lambda x: -schafferN2(x)

a, b, c = 1, 2.0, 2.0
max_iter = 100
vmax = 0.8

m = 50
D = 2

V = 2 * np.random.random((m, D))
X = 200 * np.random.random((m, D)) - 100

P, P_vals = X, np.array([schaffer(X[i, :]) for i in range(m)]).reshape((m,-1))
g, g_vals = P[np.argmax(P_vals, axis=1), :], np.max(P_vals)

for _ in range(max_iter):
    # 计算个体适应度
    temp = [0] * m
    for i in range(m):
        temp[i] = schaffer(X[i, :])

    # 更新个体的历史最优解
    for i in range(m):
        if temp[i] > P_vals[i]:
            P_vals[i] = temp[i]
            P[i, :] = X[i, :]
    
    # 更新全局最优
    g, g_vals = P[np.argmax(P_vals), :], np.max(P_vals)

    # 更新速度 v 和位置 x
    r1, r2 = np.random.random((2,1))
    V = a*V + b * r1 * (P - X) + c * r2 * (g - X)
    V[V > vmax] = vmax 
    V[V < -vmax] = -vmax
    X = X + V

    # if np.abs(g_vals) < 0.001: break 
 
print('the maximum value = {}'.format(g_vals))
print('the coordatnat value = {}'.format(g))
# print(g_vals - 15**2)