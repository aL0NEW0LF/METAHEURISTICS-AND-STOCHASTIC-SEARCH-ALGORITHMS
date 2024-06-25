# Created by "FAKHRE-EDDINE" at 18:00, 24/06/2024 ----------%
#       Email: mohamedfakhreeddine2019@gmail.com            %
#       Github: https://github.com/aL0NEW0LF/               %
# ----------------------------------------------------------%


from operator import ne
import random
from math import exp, log
import numpy as np
from numpy import array
from numpy.random import rand
import utils.functions as functions

class CHIO:
    """ 
    Coronavirus Herd Immunity Optimization Algorithm (CHIO)

    Links:
        https://link.springer.com/article/10.1007/s00521-020-05296-6
    """
    
    def __init__(self, objFunc: str = None, itemax: int = 1000, his: int = 100, c0: int = 50, brr: float = 0.5, maxAge: int = 10) -> None:
        """ 
        Args:
            objFunc: str
                Name of the objective function, from F1 to F23
            itemax: int
                Maximum number of iterations, default is 1000
            his: int
                Number of herd immunity seekers, default is 100
            c0: int
                number of initial infected cases, default is 50
            brr: float
                Basic reproduction rate (BRr) which controls the CHIO operators through spreading the virus pandemic between individuals.
            maxAge: int
                Maximum infected cases age (MaxAge): It determines the status of the infected cases where cases that reach MaxAge is either recovered or died.
        """
        self.itemax = itemax
        self.his = his
        if objFunc is not None:
            if isinstance(objFunc, tuple):
                self.fobj, self.lb, self.ub, self.dim = objFunc[0], int(objFunc[1]), int(objFunc[2]), int(objFunc[3])
            else:
                self.fobj, self.lb, self.ub, self.dim = functions.GetFunctionsDetails(objFunc)
        
        self.hip = None
        self.status = None
        self.c0 = c0
        self.brr = brr
        self.maxAge = maxAge
        self.is_corona = None
        self.fitness = None
        self.gbest = None
        self.bestFit = None
        self.bestHerd = None
        self.bestHerdFit = None

    def initialize_variables(self):
        if isinstance(self.lb, (int, float)) and isinstance(self.ub, (int, float)):
            self.hip = array([[self.lb - random.random() * (self.lb - self.ub) for j in range(self.dim)] for i in range(self.his)])
        else:
            self.hip = array([[self.lb - random.random() * (self.lb - self.ub) for j in range(self.dim)] for i in range(self.his)])

        # the status vector (S) of length HIS for all cases in HIP is also initiated by either zero (susceptible case) or one (infected case). Note that the number of ones in (S) is randomly initiated as many as c0
        # tmp_c0 = random.randrange(1, self.c0)
        self.status = [1 if i < self.c0 else 0 for i in range(self.his)]
        random.shuffle(self.status)
        self.agents = np.zeros(self.his)
        self.is_corona = np.zeros((self.itemax, self.his))
        self.is_corona[0] = np.copy(self.status)
        self.fitness = np.array([self.fobj(self.hip[i]) for i in range(self.his)])
        self.gbest = np.copy(self.hip[self.fitness.argmin()])
        self.bestFit = self.fitness.min()
        self.bestHerd = np.copy(self.hip)
        self.bestHerdFit = np.copy(self.fitness)

    def evolve(self):
        # Herd immunity evolution
        for ite in range(self.itemax):
            new_hip = np.zeros((self.his, self.dim))
            for j in range(self.his):
                self.is_corona[ite, j] = 0
                for i in range(self.dim):
                    r = rand()
                    if r < (1 / 3) * self.brr:
                        # hip[c, i](t) is randomly chosen from any infected case Xc based on the status vector (S) such that c ¼ fijSi ¼ 1g
                        infected_cases = np.array([i for i in range(self.his) if self.status[i] == 1])
                        if infected_cases.shape[0] == 0:
                            continue
                        c = np.random.choice(infected_cases.shape[0], 1, replace=False)
                        new_hip[j, i] = self.hip[j, i] + r * (self.hip[j, i] - self.hip[c, i])
                        self.is_corona[ite, j] = 1
                    elif r < (2 / 3) * self.brr:
                        susceptible_cases = np.array([i for i in range(self.his) if self.status[i] == 0])
                        m = np.random.choice(susceptible_cases.shape[0], 1, replace=False)
                        new_hip[j, i] = self.hip[j, i] + r * (self.hip[j, i] - self.hip[m, i])
                    elif r < self.brr:
                        new_hip[j, i] = self.hip[j, i] + r * (self.hip[j, i] - self.gbest[i])
                    else:
                        new_hip[j, i] = self.hip[j, i]

                # Update Herd immunity population
                if self.fobj(new_hip[j]) < self.fitness[j]:
                    self.hip[j] = new_hip[j]
                    self.fitness[j] = self.fobj(new_hip[j])
                else:
                    self.agents[j] += 1

                if self.fobj(new_hip[j]) < (self.fitness[j] / self.agents[j]) and self.status[j] == 0 and self.is_corona[ite, j] == 1:
                    self.status[j] = 1
                    self.agents[j] = 1
                elif self.fobj(new_hip[j]) > (self.fitness[j] / self.agents[j]) and self.status[j] == 1:
                    self.status[j] = 2
                    self.agents[j] = 0
                
                # Fatality condition
                if self.agents[j] >= self.maxAge and self.status[j] == 1:
                    self.hip[j] = array([self.lb - random.random() * (self.lb - self.ub) for i in range(self.dim)])
                    self.status[j] = 0
                    self.agents[j] = 0

            self.gbest = np.copy(self.hip[self.fitness.argmin()])
            self.bestFit = self.fitness.min()
            self.bestHerd = np.copy(self.hip)
            self.bestHerdFit = np.copy(self.fitness)
            print("Iteration: {} | Best fitness: {}".format(ite, self.bestFit))

        return self.bestFit, self.gbest