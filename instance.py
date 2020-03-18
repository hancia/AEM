from typing import OrderedDict
import numpy as np


class Instance:
    def __init__(self, city_coords: OrderedDict, adjacency_matrix: np.ndarray):
        self.city_coords: np.ndarray = np.array(list(city_coords.values()))
        self.adjacency_matrix = adjacency_matrix
