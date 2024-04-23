from aliBabaAndTheFortyThieves.AFT import AFT
import utils.plot as plot

if __name__ == "__main__":
    itemax = 1000
    noThieves = 30
    AFT = AFT("F17", itemax, noThieves)
    AFT.initialize_variables()

    AFT.evolve()

    plot.func_plot("F17", AFT.gbestSol, "AFT")
