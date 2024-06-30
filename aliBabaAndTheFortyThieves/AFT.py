# Created by "FAKHRE-EDDINE" at 16:00, 17/03/2024 ----------%
#       Email: mohamedfakhreeddine2019@gmail.com            %
#       Github: https://github.com/aL0NEW0LF/               %
# ----------------------------------------------------------%

from random import random
from math import exp, log
import numpy as np
from numpy import array
from numpy.random import rand
import utils.functions as functions

class AFT:
    """ 
    Ali Baba and the forty thieves Optimization Algorithm (AFT)

    Links:
        https://link.springer.com/article/10.1007/s00521-021-06392-x

    Hyper-parameters: 
    """

    def __init__(self, objFunc: str, itemax: int = 1000, noThieves: int = 40) -> None:
        """ 
        Args:
            objFunc: str
                Name of the objective function, from F1 to F23
            itemax: int
                Maximum number of iterations, default is 1000
            noThieves: int
                Number of thieves, default is 40 
        """
        if type(objFunc) is not str:
            raise ValueError("objFunc must be a string")
        tmp = functions.GetFunctionsDetails(objFunc)
        if tmp == "Invalid function":
            raise ValueError("objFunc must be a valid function")

        self.noThieves = noThieves
        self.fobj, self.lb, self.ub, self.dim = tmp
        self.itemax = itemax
        self.Pp = 0
        self.Td = 0
        self.a = 0
        self.xth = None
        self.fit = None
        self.fitness = None
        self.Sorted_thieves = None
        self.Sorted_fitness = None
        self.gbest = None
        self.fit0 = None
        self.best = None
        self.xab = None
        self.ccurve = None
        self.bestThieves = None
        self.gbestSol = None
        
    def initialize_variables(self):
        if isinstance(self.lb, (int, float)) and isinstance(self.ub, (int, float)):
            self.xth = array([[self.lb - random() * (self.lb - self.ub) for j in range(self.dim)] for i in range(self.noThieves)])
        else:
            self.xth = array([[self.lb[j] - random() * (self.lb[j] - self.ub[j]) for j in range(self.dim)] for i in range(self.noThieves)])
        self.fit = array([self.fobj(self.xth[i]) for i in range(self.noThieves)])
        self.fitness = self.fit
        self.Sorted_thieves = self.xth[self.fit.argsort()]
        self.Sorted_fitness = self.fit[self.fit.argsort()]
        self.gbest = self.Sorted_thieves[0]
        self.fit0 = self.Sorted_fitness[0]
        self.best = self.xth
        self.xab = self.xth
        self.ccurve = np.zeros(self.itemax)
        
    def evolve(self):
        for ite in range(self.itemax):
            print(f"Iteration# {ite} Fitness = {self.fit0}")

            self.Pp = 0.1 * log(2.75 * ((ite + 1) / self.itemax) ** 0.1)

            self.Td = 2 * exp(-2 * ((ite + 1) / self.itemax) ** 2)
            self.a = np.ceil((self.noThieves - 1) * rand(self.noThieves))

            for i in range(self.noThieves):
                if random() >= 0.5:
                    if random() > self.Pp:
                        self.xth[i] = self.gbest + (self.Td * (self.best[i] - self.xab[i]) * rand() + self.Td * (self.xab[i] - self.best[int(self.a[i])]) * rand()) * np.sign(random() - 0.50)
                        self.xth[i] = np.maximum(self.xth[i], self.lb)
                        self.xth[i] = np.minimum(self.xth[i], self.ub)
                    else:
                        for j in range(self.dim):
                            if isinstance(self.lb, (int, float)) and isinstance(self.ub, (int, float)):
                                self.xth[i, j] = self.Td * ((self.ub - self.lb) * random() + self.lb)
                            else:
                                self.xth[i, j] = self.Td * ((self.ub[j] - self.lb[j]) * random() + self.lb[j])
                else:
                    for j in range(self.dim):
                        self.xth[i] = self.gbest - (self.Td * (self.best[i] - self.xab[i]) * rand() + self.Td * (self.xab[i] - self.best[int(self.a[i])]) * rand()) * np.sign(random() - 0.50)
                        self.xth[i] = np.maximum(self.xth[i], self.lb)
                        self.xth[i] = np.minimum(self.xth[i], self.ub)

            for i in range(self.noThieves):
                self.fit[i] = self.fobj(self.xth[i])
                if not ((self.xth[i] - self.lb) <= 0).all() and not ((self.xth[i] - self.ub) >= 0).all():
                    self.xab[i] = self.xth[i]
                    if self.fit[i] < self.fitness[i]:
                        self.best[i] = self.xth[i]
                        self.fitness[i] = self.fit[i]
                    if self.fitness[i] < self.fit0:
                        self.fit0 = self.fitness[i]
                        self.gbest = self.best[i]

            self.ccurve[ite] = self.fit0

        self.bestThieves = np.where(self.fitness == min(self.fitness))[0]
        self.gbestSol = self.best[self.bestThieves[0]]
        self.fitness = self.fobj(self.gbestSol)
        return self.fit0, self.gbest, self.ccurve
