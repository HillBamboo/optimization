import copy

import numpy as np
import matplotlib.pyplot as plt
from test_fucntions import schafferN2, drop_wave, holder, mccormick

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

class PSO:
    def __init__(self, m=50, D=2, vmax=0.8, a=1.0, b=2.0, c=2.0):
        self.m = m # 粒子数目
        self.D = D # 解空间维数
        self.vmax = vmax
        self.a, self.b, self.c = a, b, c

        self.V = None # 粒子速度
        self.X = None # 粒子位置
        self.fitness = np.zeros((self.m, 1))
        self.trace = []

    def initialize(self, bounds):
        """
        bounds = [[lhs_x1, rhs_x1], ..., [lhs_xD, rhs_xD]]
        bounds: ndarray, bounds.shape = (D, 2)
        """
        np.random.seed(123)
        bounds.shape = (-1, 2)
        self.V = 2 * np.random.random((self.m, self.D))
        self.X = (bounds[:, 1] - bounds[:, 0]) * np.random.random((self.m, self.D)) + bounds[:, 0]
# (bounds[:, 1] - bounds[:, 0]) * np.random.random((5, 5)) + bounds[:, 0]
        # print("after random choice\n", self.X)
        self.p, self.p_vals = self.X, np.array([self.target(self.X[i, :]) for i in range(self.m)]).reshape((self.m, -1))
        # print("after p\n", self.X)
        self.optimalSolution, self.optimalValue = self.p[np.argmax(self.p_vals, axis=1), :], np.max(self.p_vals)
        # print("after optimal\n ", self.X)

    def evaluation(self):
        for i in range(self.m):
            self.fitness[i] = self.target(self.X[i, :])
            if self.fitness[i, :] > self.p_vals[i]: # 若当前适应度比个体历史最佳适应度好
                self.p_vals[i] = copy.deepcopy(self.fitness[i])
                self.p[i, :] = copy.deepcopy(self.X[i, :])

    # TODO: X的更新没有考虑到边界检查，从而可以找到最优值，但是最优点不对
    def update(self):
        # 更新速度 v 和位置 x
        r1, r2 = np.random.random((2,1))
        self.V = self.a*self.V + self.b*r1*(self.p - self.X) + self.c*r2*(self.optimalSolution - self.X)
        self.V[self.V > self.vmax] = self.vmax
        self.V[self.V < -self.vmax] = -self.vmax
        self.X = self.X + self.V

    def solve(self, max_iter=500, verbose=False, printInfo=False):
        """
        max_iter: 最大迭代次数
        verbose: 输出每轮迭代的情况
        printInfo: 画出收敛曲线
        """
        # 0 - 种群初试化
        # self.initialize(np.array([-5.12, 5.12]))
        # trace = []
        for i in range(max_iter):
            # 1 - 计算个体适应度
            self.evaluation()
            # 2 - 更新全局最优
            self.optimalSolution, self.optimalValue = copy.deepcopy(self.p[np.argmax(self.p_vals), :]), np.max(self.p_vals)
            self.trace.append(self.optimalValue)
            # 3 - 更新粒子的速度和位置
            self.update()

            if verbose:
                print("\nITERATION {}".format(str(i+1)))
                print("    > optimal solution: ", self.optimalSolution)
                print("    > optimal value: ", self.optimalValue)

        if printInfo:
            self.printInfo(max_iter)

        return self.optimalSolution, self.optimalValue

    def printInfo(self, x):
        # plt.ion()
        x = list(range(x))
        y = self.trace
        # fig = plt.figure()
        # ax = fig.subplots(111)
        # plt.ion()
        plt.plot(x, y)
        plt.title("PSO算法")
        plt.xlabel("迭代次数")
        plt.ylabel("最大值")
        plt.xticks(x, rotation='45')
        plt.grid()
        # plt.legend()
        plt.show()
        # plt.savefig(fname='', format='png')

    def target(self, x):
        """适应度函数"""
        pass