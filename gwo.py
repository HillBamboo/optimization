## 灰狼算法
import numpy as np
from numpy import e

def initilaztion(num_population, bounds):
    bounds.shape = (-1, 2)
    dims = bounds.shape[0]
    population = np.random.random((num_population, dims)) # 在[0, 1) 之间初始化种群
    population = population * (bounds[:, 1] - bounds[:, 0]).T - bounds[:, 1]

    return population

def target(x):
    pass

def main(num_population, bounds, max_iter=500):
    population = initilaztion(num_population, bounds)
    a = 2
    A = 2 * a * np.random.random((3, 1)) - a
    C = 2 * np.random.random((3, 1))

    # 1 - 计算个体适应度
    fitness = []
    for i in range(num_population):
        f = target(population[i, :])
        fitness.append(f)

    # 2 - 狼群分级

    x_p = population[np.argsort(fitness)[-3:]]

    for t in range(max_iter):
        for i in range(num_population):
            D = np.abs(C*x_p - population[i,:])
            xx = x_p - A*D
            population[i, :] = xx.sum(axis=0) / 3

        k = np.exp(t/max_iter) - 1
        a = 2 - 2*(1/(e-1) * k)
        A = 2 * a * np.random.random((3, 1)) - a
        C = 2 * np.random.random((3,1))

        # 1 - 计算个体适应度
        for i in range(num_population):
            fitness[i] = target(population[i, :])

        # 2 - 狼群分级
        x_p = population[np.argsort(fitness)[-3:]]

    return target(x_p[-1]), x_p[-1]

if __name__ == '__main__':
    target = lambda x: np.sum(x**2)

    max_iter = 500
    num_population = 30
    bounds = np.asarray([[-100,100]] * 30)

    optimal_value, optimal_solution = main(num_population, bounds, 300)
    # print("optimal value = ", optimal_value)
    # print("optimal solution = ", optimal_solution)

