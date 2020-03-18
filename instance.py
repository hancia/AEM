from typing import OrderedDict
from collections import OrderedDict
import numpy as np

from tsplib95 import Problem, load_problem
from tsplib95.distances import euclidean


class Instance:
    PATH = 'instances/{}.tsp'

    def __init__(self, name: str):
        self.adjacency_matrix: np.ndarray = None
        self.city_coords: np.ndarray = None
        self.load_instance(name)

    def load_instance(self, name: str) -> None:
        problem: Problem = load_problem(Instance.PATH.format(name))
        coords: OrderedDict = problem.node_coords
        d = problem.dimension

        self.adjacency_matrix = np.zeros(shape=(d, d), dtype=np.int)
        for i in range(d):
            for j in range(d):
                self.adjacency_matrix[i, j] = euclidean(coords[i + 1], coords[j + 1])

        self.city_coords: np.ndarray = np.array(list(coords.values()))

    @property
    def length(self):
        return len(self.city_coords)
