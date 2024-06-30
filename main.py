import numpy as np
from aliBabaAndTheFortyThieves.AFT import AFT
from virus_colony_search.VCS import VCS
from coronavirus_herd_immunity_optimizer.CHIO import CHIO
import utils.plot as plot
import tsplib95
import itertools

def generate_all_tours(num_cities):
    """
    Generate all possible tours for a given number of cities.
    
    :param num_cities: The number of cities.
    :return: A list of all possible tours, where each tour is a list of city indices.
    """
    # Generate a list of city indices
    city_indices = list(range(num_cities))
    
    # Use itertools.permutations to generate all possible orderings of city indices
    all_tours = list(itertools.permutations(city_indices))
    
    return all_tours

def tsp_objective_function(tour, distance_matrix):
    """
    Calculate the total distance of a tour based on a distance matrix.
    
    :param tour: A list of city indices representing the tour.
    :param distance_matrix: A 2D numpy array where element [i, j] is the distance from city i to city j.
    :return: Total distance of the tour.
    """
    total_distance = 0
    number_of_cities = len(tour)
    print(tour)
    for i in range(number_of_cities-1):
        # Ensure indices are integers
        total_distance += distance_matrix[i, i+1]
    
    return total_distance

if __name__ == "__main__":
    itemax = 1000
    pop = 30
    test_functions = ["F24", "F25", "F26", "F27", "F28", "F29", "F30", "F31", "F32", "F33", "F34", "F35", "F36", "F37"]
    test = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23"]
    
    for func in test_functions:
        try:
            print("Function: ", func)

            aft = AFT(func, itemax, pop)
            aft.initialize_variables()

            aft.evolve()


        except Exception as e:
            continue

    """ vcs = VCS(objFunc="F29", itemax=itemax, noViruses=pop)

    vcs.initialize_variables()
    vcs.evolve()

    plot.func_plot("F29", vcs.bestVirus, vcs.fitness_track, "VCS") """

    """ chio = CHIO(objFunc="F24", itemax=itemax, his=pop, c0=10, brr=0.5, maxAge=100)

    chio.initialize_variables()
    bestFit, bestHerd, fitness_track = chio.evolve()
    print("Best Fit: ", bestFit)
    print("Best Herd: ", bestHerd)
    plot.func_plot("F24", chio.gbest, fitness_track, "chio") """
    """ AFT = AFT("F24", itemax, pop)
    AFT.initialize_variables()

    AFT.evolve()

    plot.func_plot("F24", AFT.gbestSol, AFT.ccurve, "AFT") """