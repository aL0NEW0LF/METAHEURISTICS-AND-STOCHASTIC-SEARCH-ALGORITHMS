from re import A
from aliBabaAndTheFortyThieves.AFT import AFT
from aliBabaAndTheFortyThieves.objFunction import Objfun
import utils.functions as functions
import utils.plot as plot
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    itemax = 1000
    noThieves = 30
    AFT = AFT("F17", itemax, noThieves)
    AFT.initialize_variables()

    AFT.evolve()

    plot.func_plot("F17", AFT.gbestSol, "AFT")
