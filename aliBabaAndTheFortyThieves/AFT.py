# Created by "FAKHRE-EDDINE" at 16:00, 17/03/2024 ----------%
#       Email: mohamedfakhreeddine2019@gmail.com            %
#       Github: https://github.com/aL0NEW0LF/               %
# ----------------------------------------------------------%

from random import random, randint
from math import exp, log, ceil
import numpy as np
from numpy import array
from numpy.random import rand
import matplotlib.pyplot as plt

class AFT:
    """ 
    Ali Baba and the forty thieves Optimization Algorithm (AFT)

    Links:
        https://link.springer.com/article/10.1007/s00521-021-06392-x

    Hyper-parameters: 
    """

    def __init__(self, lb: list, ub: list, dim: int, fobj, itemax: int = 1000, noThieves: int = 40) -> None:
        """ 
        Args:
            lb: list
                Lower bounds for the problem
            ub: list
                Upper bounds for the problem
            dim: int
                Dimension of the problem
            fobj: function
                Objective function to be minimized
            itemax: int
                Maximum number of iterations, default is 1000
            noThieves: int
                Number of thieves, default is 40 
        """
        self.noThieves = noThieves
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.fobj = fobj
        self.itemax = itemax
        self.Pp = 0
        self.Td = 0
        self.a = 0
        self.xth = None
        self.fit = None
        self.fitness = None
        self.Sorted_thieves = None
        self.gbest = None
        self.fit0 = None
        self.best = None
        self.xab = None
        self.ccurve = None
        self.bestThieves = None
        self.gbestSol = None
        
    def initialize_variables(self):
        self.xth = array([[self.lb[j] - random() * (self.lb[j] - self.ub[j]) for j in range(self.dim)] for i in range(self.noThieves)])
        self.fit = array([self.fobj(self.xth[i]) for i in range(self.noThieves)])
        self.fitness = self.fit
        self.Sorted_thieves = self.xth[self.fit.argsort()]
        self.gbest = self.Sorted_thieves[0]
        self.fit0 = self.fit[0]
        self.best = self.xth
        self.xab = self.xth
        self.ccurve = np.zeros(self.itemax)
        
    def evolve(self):
        for ite in range(self.itemax):
            print(f"Iteration# {ite}  Fitness= {self.fit0}")

            if (2.75 * (ite / self.itemax) ** 0.1) > 0:
                self.Pp = 0.1 * log(2.75 * (ite / self.itemax) ** 0.1)
            else:
                self.Pp = 0

            self.Td = exp(-2 * (ite / self.itemax) ** 2)
            self.a = np.ceil((self.noThieves - 1) * rand(self.noThieves))

            for i in range(self.noThieves):
                if random() >= 0.5:
                    if random() > self.Pp:
                        self.xth[i] = self.gbest + (self.Td * (self.best[i] - self.xab[i]) * rand() + self.Td * (self.xab[i] - self.best[int(self.a[i])]) * rand()) * np.sign(random() - 0.50)
                    else:
                        for j in range(self.dim):
                            self.xth[i, j] = self.Td * ((self.ub[j] - self.lb[j]) * random() + self.lb[j])
                else:
                    for j in range(self.dim):
                        self.xth[i, j] = self.gbest[j] - (self.Td * (self.best[i, j] - self.xab[i, j]) * rand() + self.Td * (self.xab[i, j] - self.best[int(self.a[i]), j]) * rand()) * np.sign(random() - 0.50)

            for i in range(self.noThieves):
                self.fit[i] = self.fobj(self.xth[i])
                if (self.xth[i] - self.lb <= 0).all() and (self.xth[i] - self.ub >= 0).all():
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
        return self.fitness, self.gbestSol, self.ccurve
