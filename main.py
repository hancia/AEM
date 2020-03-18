from collections import OrderedDict

import tsplib95
import seaborn as sns
import numpy as np
from tsplib95.distances import euclidean
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from cheapest_insertion import CheapestInsertion
from instance import Instance
from utils import pairwise

sns.set()

problem: tsplib95.models.Problem = tsplib95.load_problem('instances/kroA100.tsp')
coords: OrderedDict = problem.node_coords
d = problem.dimension
adjacency_matrix = np.zeros(shape=(d, d), dtype=np.int)
for i in range(d):
    for j in range(d):
        adjacency_matrix[i, j] = euclidean(coords[i + 1], coords[j + 1])

# TODO refactor class
instance = Instance(city_coords=coords, adjacency_matrix=adjacency_matrix)
solve_strategy: CheapestInsertion = CheapestInsertion(instance=instance)
solution = solve_strategy.run()


def draw_solution(instance: Instance, solution: list, title: str = None):
    ax = sns.scatterplot(instance.city_coords[:, 0], instance.city_coords[:, 1], color='black', zorder=5)

    for id_source, id_destination in pairwise(solution):
        ax.annotate('', xy=instance.city_coords[id_source], xytext=instance.city_coords[id_destination],
                    arrowprops=dict(arrowstyle='-|>', color='red', connectionstyle="arc3"))

    for i, coords in enumerate(instance.city_coords):
        ax.text(coords[0] - 35, coords[1] + 25, str(i), size=6)

    if title is not None:
        ax.set_title(title)

    ax.scatter(instance.city_coords[0, 0], instance.city_coords[0, 1], zorder=6)
    plt.show()


draw_solution(instance, solution, f'Cheapest Insertion, distance: {solve_strategy.solution_cost}')
