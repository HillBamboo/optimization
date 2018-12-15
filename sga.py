# simple genetic algorithm

# import numpy as np

#-*- coding:utf-8 -*-
 
import random
import math
from operator import itemgetter
 
from testfucntions import schafferN2

evaluate = lambda geneinfo: -schafferN2(geneinfo)

# class Gene:
#     '''
#     This is a class to represent individual(Gene) in GA algorithom
#     each object of this class have two attribute: data, size
#     '''
#     def __init__(self,**data):
#         self.__dict__.update(data)       
#         self.size = len(data['data'])#length of gene
                                
        
# class GA:
#     '''
#     This is a class of GA algorithm. 
#     '''

#     def __init__(self,parameter):
#         '''
#         Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value .
#         The data structure of pop is composed of several individuals which has the form like that:
        
#         {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
#         Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]
        
#         '''
#         #parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
#         self.parameter = parameter
 
#         low = self.parameter[4]
#         up = self.parameter[5]
        
#         self.bound = []
#         self.bound.append(low)
#         self.bound.append(up)
        
#         pop = []
#         for i in range(self.parameter[3]):
#             geneinfo = []
#             for pos in range(len(low)):
#                 geneinfo.append(random.uniform(self.bound[0][pos], self.bound[1][pos]))#initialise popluation
                
#             fitness = evaluate(geneinfo)#evaluate each chromosome
#             pop.append({'Gene':Gene(data = geneinfo), 'fitness':fitness})#store the chromosome and its fitness
            
#         self.pop = pop
#         self.bestindividual = self.selectBest(self.pop)#store the best chromosome in the population

#     def selectBest(self, pop):
#         '''
#         select the best individual from pop
#         '''
#         s_inds = sorted(pop, key = itemgetter("fitness"), reverse = False)
#         return s_inds[0]
        
#     def selection(self, individuals, k):
#         '''
#         select two individuals from pop
#         '''
#         s_inds = sorted(individuals, key = itemgetter("fitness"), reverse=True)#sort the pop by the reference of 1/fitness 
#         sum_fits = sum(1/ind['fitness'] for ind in individuals) #sum up the 1/fitness of the whole pop
        
#         chosen = []
#         for i in range(k):
#             u = random.random() * sum_fits#randomly produce a num in the range of [0, sum_fits]
#             sum_ = 0
#             for ind in s_inds:
#                 sum_ += 1/ind['fitness']#sum up the 1/fitness
#                 if sum_ > u:
#                     #when the sum of 1/fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
#                     chosen.append(ind)
#                     break
        
#         return chosen    
 
 
#     def crossoperate(self, offspring):
#         '''
#         cross operation
#         '''
#         dim = len(offspring[0]['Gene'].data)
 
#         geninfo1 = offspring[0]['Gene'].data#Gene's data of first offspring chosen from the selected pop
#         geninfo2 = offspring[1]['Gene'].data#Gene's data of second offspring chosen from the selected pop
        
#         pos1 = random.randrange(1,dim)#select a position in the range from 0 to dim-1, 
#         pos2 = random.randrange(1,dim)
 
#         newoff = Gene(data = [])#offspring produced by cross operation
#         temp = []
#         for i in range(dim):
#             if (i >= min(pos1,pos2) and i <= max(pos1,pos2)):
#                 temp.append(geninfo2[i])
#                 #the gene data of offspring produced by cross operation is from the second offspring in the range [min(pos1,pos2),max(pos1,pos2)]
#             else:
#                 temp.append(geninfo1[i])
#                 #the gene data of offspring produced by cross operation is from the frist offspring in the range [min(pos1,pos2),max(pos1,pos2)]
#         newoff.data = temp
       
#         return newoff
 
 
#     def mutation(self, crossoff, bound):
#         '''
#         mutation operation
#         '''
        
#         dim = len(crossoff.data)
 
#         pos = random.randrange(1,dim)#chose a position in crossoff to perform mutation.
 
#         crossoff.data[pos] = random.uniform(bound[0][pos],bound[1][pos])
#         return crossoff
    
#     def GA_main(self):
#         '''
#         main frame work of GA
#         '''
        
#         popsize = self.parameter[3]
        
#         print("Start of evolution")
        
#         # Begin the evolution
#         for g in range(NGEN):
            
#             print("-- Generation %i --" % g)      
                      
#             #Apply selection based on their converted fitness
#             selectpop = self.selection(self.pop, popsize)   
 
#             nextoff = []    
#             while len(nextoff) != popsize:      
#                 # Apply crossover and mutation on the offspring            
                                
#                 # Select two individuals
#                 offspring = [random.choice(selectpop) for i in range(2)]
                
#                 if random.random() < CXPB: # cross two individuals with probability CXPB
#                     crossoff = self.crossoperate(offspring)
#                     fit_crossoff = evaluate(self.xydata, crossoff.data)# Evaluate the individuals           
                    
#                     if random.random() < MUTPB: # mutate an individual with probability MUTPB
#                         muteoff = self.mutation(crossoff,self.bound)
#                         fit_muteoff = evaluate(self.xydata, muteoff.data)# Evaluate the individuals
#                         nextoff.append({'Gene':muteoff,'fitness':fit_muteoff})
                        
#             # The population is entirely replaced by the offspring
#             self.pop = nextoff
            
#             # Gather all the fitnesses in one list and print the stats
#             fits = [ind['fitness'] for ind in self.pop]
                
#             length = len(self.pop)
#             mean = sum(fits) / length
#             sum2 = sum(x*x for x in fits)
#             std = abs(sum2 / length - mean**2)**0.5
#             best_ind = self.selectBest(self.pop)
 
#             if best_ind['fitness'] < self.bestindividual['fitness']:
#                 self.bestindividual = best_ind
 
#             print("Best individual found is %s, %s" % (self.bestindividual['Gene'].data,self.bestindividual['fitness']))
#             print("  Min fitness of current pop: %s" % min(fits))
#             print("  Max fitness of current pop: %s" % max(fits))
#             print("  Avg fitness of current pop: %s" % mean)
#             print("  Std of currrent pop: %s" % std)
        
#         print("-- End of (successful) evolution --")    
 
# if __name__ == "__main__":
 
#     CXPB, MUTPB, NGEN, popsize = 0.8, 0.3, 50, 100#control parameters
    
#     # up = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]#upper range for variables
#     # low = [-64, -64, -64, -64, -64, -64, -64, -64, -64, -64]#lower range for variables
#     up = [64, 64]
#     low = [-64, -64]
#     parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
    
#     run = GA(parameter)
#     run.GA_main()

# !/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: wsw
# 简单实现SGA算法
import numpy as np
from scipy.optimize import fsolve, basinhopping
import random
import timeit


import matplotlib.pyplot as plt
 
 
# 根据解的精度确定染色体(chromosome)的长度
# 需要根据决策变量的上下边界来确定
def getEncodedLength(delta=0.0001, boundarylist=[]):
    # 每个变量的编码长度
    lengths = []
    for i in boundarylist:
        lower = i[0]
        upper = i[1]
        # lamnda 代表匿名函数f(x)=0,50代表搜索的初始解
        res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x - 1, 50)
        length = int(np.floor(res[0]))
        lengths.append(length)
    return lengths

  

 
 
# 染色体解码得到表现型的解
def decodedChromosome(encodelength, chromosomes, boundarylist, delta=0.0001):
    populations = chromosomes.shape[0]
    variables = len(encodelength)
    decodedvalues = np.zeros((populations, variables))
    for k, chromosome in enumerate(chromosomes):
        chromosome = chromosome.tolist()
        start = 0
        for index, length in enumerate(encodelength):
            # 将一个染色体进行拆分，得到染色体片段
            power = length - 1
            # 解码得到的10进制数字
            demical = 0
            for i in range(start, length + start):
                demical += chromosome[i] * (2 ** power)
                power -= 1
            lower = boundarylist[index][0]
            upper = boundarylist[index][1]
            decodedvalue = lower + demical * (upper - lower) / (2 ** length - 1)
            decodedvalues[k, index] = decodedvalue
            # 开始去下一段染色体的编码
            start = length
    return decodedvalues
 
 

 
 
# 新种群选择
def selectNewPopulation(chromosomes, cum_probability):
    m, n = chromosomes.shape
    newpopulation = np.zeros((m, n), dtype=np.uint8)
    # 随机产生M个概率值
    randoms = np.random.rand(m)
    for i, randoma in enumerate(randoms):
        logical = cum_probability >= randoma
        index = np.where(logical == 1)
        # index是tuple,tuple中元素是ndarray
        newpopulation[i, :] = chromosomes[index[0][0], :]
    return newpopulation
    pass
 
 
# 新种群交叉
def crossover(population, Pc=0.8):
    """
    :param population: 新种群
    :param Pc: 交叉概率默认是0.8
    :return: 交叉后得到的新种群
    """
    # 根据交叉概率计算需要进行交叉的个体个数
    m, n = population.shape
    numbers = np.uint8(m * Pc)
    # 确保进行交叉的染色体个数是偶数个
    if numbers % 2 != 0:
        numbers += 1
    # 交叉后得到的新种群
    updatepopulation = np.zeros((m, n), dtype=np.uint8)
    # 产生随机索引
    index = random.sample(range(m), numbers)
    # 不进行交叉的染色体进行复制
    for i in range(m):
        if not index.__contains__(i):
            updatepopulation[i, :] = population[i, :]
    # crossover
    while len(index) > 0:
        a = index.pop()
        b = index.pop()
        # 随机产生一个交叉点
        crossoverPoint = random.sample(range(1, n), 1)
        crossoverPoint = crossoverPoint[0]
        # one-single-point crossover
        updatepopulation[a, 0:crossoverPoint] = population[a, 0:crossoverPoint]
        updatepopulation[a, crossoverPoint:] = population[b, crossoverPoint:]
        updatepopulation[b, 0:crossoverPoint] = population[b, 0:crossoverPoint]
        updatepopulation[b, crossoverPoint:] = population[a, crossoverPoint:]
    return updatepopulation
    pass
 
 
# 染色体变异
def mutation(population, Pm=0.1):
    """
    :param population: 经交叉后得到的种群
    :param Pm: 变异概率默认是0.01
    :return: 经变异操作后的新种群
    """
    updatepopulation = np.copy(population)
    m, n = population.shape
    # 计算需要变异的基因个数
    gene_num = np.uint8(m * n * Pm)
    # 将所有的基因按照序号进行10进制编码，则共有m*n个基因
    # 随机抽取gene_num个基因进行基本位变异
    mutationGeneIndex = random.sample(range(0, m * n), gene_num)
    # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
    for gene in mutationGeneIndex:
        # 确定变异基因位于第几个染色体
        chromosomeIndex = gene // n
        # 确定变异基因位于当前染色体的第几个基因位
        geneIndex = gene % n
        # mutation
        if updatepopulation[chromosomeIndex, geneIndex] == 0:
            updatepopulation[chromosomeIndex, geneIndex] = 1
        else:
            updatepopulation[chromosomeIndex, geneIndex] = 0
    return updatepopulation
    pass
 

schaffer = lambda x: -schafferN2(x)
fitnessFunction = lambda : schaffer
 
def main(max_iter=500):
    pass

xx, yy, solution, value = main(500)

class GA:
    def __init__(self):
        pass

    def target(self, x):
        pass
        
    def initialize(self):
        pass
        # 随机生成初始编码种群

    def getIntialPopulation(encodelength, populationSize):
        # 随机化初始种群为0
        chromosomes = np.zeros((populationSize, sum(encodelength)), dtype=np.uint8)
        for i in range(populationSize):
            chromosomes[i, :] = np.random.randint(0, 2, sum(encodelength))
        # print('chromosomes shape:', chromosomes.shape)
        return chromosomes

    def solve(self, max_iter=300, verbose=False, printInfo=False):
        # 每次迭代得到的最优解
        optimal_solutions = []
        optimal_values = []
        # 决策变量的取值范围
        decisionVariables = [[-3.0, 12.1], [4.1, 5.8]]

        lengthEncode = getEncodedLength(boundarylist=decisionVariables)
        for iteration in range(max_iter):
            # 得到初始种群编码
            chromosomesEncoded = getIntialPopulation(lengthEncode, 10)

            # 种群解码
            decoded = decodedChromosome(lengthEncode, chromosomesEncoded, decisionVariables)
            evalvalues, cum_proba = getFitnessValue(fitnessFunction(), decoded)

            # 选择新的种群
            newpopulations = selectNewPopulation(chromosomesEncoded, cum_proba)
            crossoverpopulation = crossover(newpopulations)
            mutationpopulation = mutation(crossoverpopulation)

            # 将变异后的种群解码，得到每轮迭代最终的种群
            final_decoded = decodedChromosome(lengthEncode, mutationpopulation, decisionVariables)

            fitnessvalues, cum_individual_proba = getFitnessValue(fitnessFunction(), final_decoded)

            optimal_values.append(np.max(list(fitnessvalues)))
            index = np.where(fitnessvalues == max(list(fitnessvalues)))
            optimal_solutions.append(final_decoded[index[0][0], :])

        # 搜索最优解
        best = np.max(optimal_values)
        optimalIndex = np.where(optimal_values == best)
        optimal_solution = optimal_solutions[optimalIndex[0][0]]

        return optimalSolutions, optimalValues, optimalSolution, optimalValue

    def evaluation(self):
        for i in range(self.m):
            self.fitness[i] = self.func(self.X[i, :])
            if self.fitness[i, :] > self.p_vals[i]: # 若当前适应度比个体历史最佳适应度好
                self.p_vals[i] = copy.deepcopy(self.fitness[i])
                self.p[i, :] = copy.deepcopy(self.X[i, :])


    # 得到个体的适应度值及每个个体被选择的累积概率
    def getFitnessValue(self, chromosomesdecoded):
        # 得到种群规模和决策变量的个数
        population, nums = chromosomesdecoded.shape
        # 初始化种群的适应度值为0
        self.fitness = np.zeros((population, 1))

        for i in range(population):
            self.fitness[i, 0] = self.target(chromosomesdecoded[i, :])
        prob = self.fitness / np.sum(self.fitness)
        cum_prob = np.cumsum(prob)

        return cum_prob

    def encode(self):
        pass

    def decode(self):
        pass

    def mutate(self):
        pass

    def crossoperate(self):
        pass

    def select(self):
        pass

    def printInfo(self, x):
        # plt.ion()
        x = range(x)
        y = self.trace
        plt.plot(x, y)
        plt.title("PSO算法")
        plt.xlabel("迭代次数")
        plt.ylabel("最大值")
        plt.xticks(x)
        plt.grid()
        # plt.legend()

        plt.show()


    


print('最优解: x1, x2')
print(solution[0], solution[1])
print('最优目标函数值:', value)
# 测量运行时间
# elapsedtime = timeit.timeit(stmt=main, number=1)
# print('Searching Time Elapsed:(S)', elapsedtime)
 
 
