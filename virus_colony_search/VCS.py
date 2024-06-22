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

class VCS:
    """ 
    Virus Colony Search Optimization Algorithm (VCS)

    Links:
        https://link.springer.com/article/10.1007/s00521-021-06392-x

    Hyper-parameters: 
    """

    def __init__(self, objFunc: str, itemax: int = 1000, noViruses: int = 100, lamda: float = 0.5, sigma: float = 1.5) -> None:
        """ 
        Args:
            objFunc: str
                Name of the objective function, from F1 to F23
            itemax: int
                Maximum number of iterations, default is 1000
            noViruses: int
                Number of viruses, default is 100 
            lamda: float
                Infection rate, percentage of the best viruses that we will keep, default is 0.5
        """
        self.itemax = itemax
        self.noViruses = noViruses
        self.lamda = lamda
        self.sigma = sigma
        self.fobj, self.lb, self.ub, self.dim = functions.GetFunctionsDetails(objFunc)
        self.virus = None
        self.fitness = None
        self.gbest = None
        self.bestFit = None
        self.bestVirus = None
        self.n_best = int(self.lamda * self.noViruses)
    
    def initialize_variables(self):
        if isinstance(self.lb, (int, float)) and isinstance(self.ub, (int, float)):
            self.virus = array([[self.lb - random() * (self.lb - self.ub) for j in range(self.dim)] for i in range(self.noViruses)])
        else:
            self.virus = array([[self.lb[j] - random() * (self.lb[j] - self.ub[j]) for j in range(self.dim)] for i in range(self.noViruses)])
        self.fitness = array([self.fobj(self.virus[i]) for i in range(self.noViruses)])
        self.best = self.virus[self.fitness.argmin()]
        self.bestFit = self.fitness.min()
        self.bestVirus = self.best.copy()
        self.gbest = self.best.copy()

    def calculate_xmean__(self):
        ## Calculate the weighted mean of the λ best individuals by
        ## using the infection rate λ
        pop = self.virus[self.fitness.argsort()]
        factor_down = self.n_best * np.log1p(self.n_best + 1) - np.log1p(np.prod(range(1, self.n_best + 1)))
        w =  np.log1p(self.n_best + 1) / factor_down
        w = w / self.n_best
        xmean = w * np.sum(pop, axis=0)
        return xmean
    
    def evolve(self):
        # Viruses diffusion
        for ite in range(self.itemax):
            for i in range(self.noViruses):
                gauss = np.array([np.random.normal(self.gbest[j], self.sigma) for j in range(self.dim)])
                pos_new = gauss + np.random.uniform(0, 1) * self.gbest - np.random.uniform(0, 1) * self.virus[i]
                pos_new = np.maximum(pos_new, self.lb)
                pos_new = np.minimum(pos_new, self.ub)

            ## Evaluate the new position, FEs=FEs+N;
            fit_new = self.fobj(pos_new)
            ## Update the best virus
            if fit_new < self.bestFit:
                self.bestFit = fit_new
                self.bestVirus = pos_new.copy()
            ## Update the virus
            if fit_new < self.fitness[i]:
                self.virus[i] = pos_new.copy()
                self.fitness[i] = fit_new

            # Host cells infection
            x_mean = self.calculate_xmean__()

            sigma = self.sigma * (1 - ite / self.itemax)
            pop = []
            for i in range(self.noViruses):
                pos_new = x_mean + sigma * np.random.normal(0, 1, self.dim)
                pos_new = np.maximum(pos_new, self.lb)
                pos_new = np.minimum(pos_new, self.ub)
                pop.append(pos_new)

            pop = np.array(pop)

            ## Evaluate Hpop and updatetheVpop; FEs=FEs+N;
            fitness = np.array([self.fobj(pop[i]) for i in range(self.noViruses)])
            self.virus = pop.copy()
            self.fitness = fitness.copy()

            ## Update the best virus
            if fitness.min() < self.bestFit:
                self.bestFit = fitness.min()
                self.bestVirus = pop[fitness.argmin()].copy()

            ## Calculate the weighted mean of the λ best individuals by
            ## using the infection rate λ
            pop = pop[fitness.argsort()]
            x_mean = self.calculate_xmean__()

            # Immune response
            for i in range(self.noViruses):
                pr = (self.dim - i + 1) / self.dim
                for j in range(self.dim):
                    if rand() < pr:
                        self.virus[i, j] = x_mean[j]
                        if isinstance(self.lb, (int, float)) and isinstance(self.ub, (int, float)):
                            self.virus[i, j] = np.maximum(self.virus[i, j], self.lb)
                            self.virus[i, j] = np.minimum(self.virus[i, j], self.ub)
                        else:
                            self.virus[i, j] = np.maximum(self.virus[i, j], self.lb[j])
                            self.virus[i, j] = np.minimum(self.virus[i, j], self.ub[j])
                        self.fitness[i] = self.fobj(self.virus[i])
                        if self.fitness[i] < self.bestFit:
                            self.bestFit = self.fitness[i]
                            self.bestVirus = self.virus[i].copy()
            print("Iteration: {} | Best fitness: {}".format(ite, self.bestFit))
        return self.bestVirus, self.bestFit