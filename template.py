#encoding=utf8

# 本文件定义通用优化算法类的接口设计

class OptimizationAlgo:
    
    def __init__(self):
        pass

    def target(self, x):
        pass

    def initialize(self):
        pass

    def solve(self, max_iter):
        
        # do something
        for i in range(max_iter):
            pass

        return optimalSolution, optimalValue

    def print_convergence(self):
        """绘制收敛曲线""""
        pass