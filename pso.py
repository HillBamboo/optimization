import copy

import numpy as np
import matplotlib.pyplot as plt
from testfucntions import schafferN2, drop_wave, holder, mccormick

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

class PSO:
    def __init__(self, m=50, D=2, max_iter=500, vmax=0.8, a=1.0, b=2.0, c=2.0):
        self.m = m # 粒子数目
        self.D = D # 解空间维数
        self.max_iter = max_iter
        self.vmax = vmax
        self.a, self.b, self.c = a, b, c

        self.V = None # 粒子速度
        self.X = None # 粒子位置
        self.fitness = np.zeros((self.m, 1))
        self.trace = [] #保存历史全局最优解

    def initialize(self, bounds):
        """
        bounds = [[lhs_x1, rhs_x1], ..., [lhs_xD, rhs_xD]]
        bounds: ndarray, bounds.shape = (D, 2)
        """
        bounds.shape = (-1, 2)
        self.V = 2 * np.random.random((self.m, self.D))
        self.X = np.random.random((self.m, self.D)) * (bounds[:, 1] - bounds[:, 0]).T - bounds[:, 1]

        self.p, self.p_vals = self.X, np.array([self.func(self.X[i, :]) for i in range(self.m)]).reshape((self.m,-1))
        self.g, self.g_vals = self.p[np.argmax(self.p_vals, axis=1), :], np.max(self.p_vals) 

    def evaluation(self):
        for i in range(self.m):
            self.fitness[i] = self.func(self.X[i, :])
            if self.fitness[i, :] > self.p_vals[i]: # 若当前适应度比个体历史最佳适应度好
                self.p_vals[i] = copy.deepcopy(self.fitness[i])
                self.p[i, :] = copy.deepcopy(self.X[i, :])

    def update(self):
        # 更新速度 v 和位置 x
        r1, r2 = np.random.random((2,1))
        self.V = self.a*self.V + self.b*r1*(self.p - self.X) + self.c*r2*(self.g - self.X)
        self.V[self.V > self.vmax] = self.vmax
        self.V[self.V < -self.vmax] = -self.vmax
        self.X = self.X + self.V

    def solve(self, verbose=False, printInfo=False):
        """
        verbose: 输出每轮迭代的情况
        printInfo: 画出收敛曲线
        """
        # 0 - 种群初试化
        # self.initialize(np.array([-5.12, 5.12]))
        for i in range(self.max_iter):
            # 1 - 计算个体适应度
            self.evaluation()

            # 2 - 更新全局最优
            self.g, self.g_vals = copy.deepcopy(self.p[np.argmax(self.p_vals), :]), np.max(self.p_vals)
            self.trace.append(self.g_vals)

            # 3 - 更新粒子的速度和位置
            self.update()

            if verbose:
                print("\nITERATION {}".format(str(i+1)))
                print("    > optimal solution: ", self.g)
                print("    > optimal value: ", self.g_vals)

        if printInfo:
            self.printInfo()

    def printInfo(self):
        # plt.ion()
        x = range(self.max_iter)
        y = self.trace
        plt.plot(x, y)
        plt.title("PSO算法")
        plt.xlabel("迭代次数")
        plt.ylabel("最大值")
        plt.xticks(x)
        plt.grid()
        # plt.legend()

        plt.show()

    def func(self, xx):
        """适应度函数"""
        return -drop_wave(xx)

if __name__ == '__main__':
    pso = PSO(max_iter=300)
    bounds = np.array([[-1.5, 4], [-3, 4]])
    pso.initialize(bounds)
    pso.func = lambda x: -mccormick(x)
    # 很可能陷入局部最优了
    pso.solve(verbose=True, printInfo=True)