from aliBabaAndTheFortyThieves.AFT import AFT
from virus_colony_search.VCS import VCS
import utils.plot as plot

if __name__ == "__main__":
    itemax = 1000
    noThieves = 15
    VCS = VCS("F10", itemax, noThieves)
    VCS.initialize_variables()

    VCS.evolve()

    plot.func_plot("F10", VCS.bestVirus, "VCS")
