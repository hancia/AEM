from collections import OrderedDict

import tsplib95
import numpy as np
from tsplib95.distances import euclidean

problem: tsplib95.models.Problem = tsplib95.load_problem('kroA100.tsp')
coords: OrderedDict = problem.node_coords
d = problem.dimension
adjacency_matrix = np.zeros(shape=(d, d), dtype=np.int)
for i in range(d):
    for j in range(d):
        adjacency_matrix[i, j] = euclidean(coords[i + 1], coords[j + 1])
print(adjacency_matrix)
