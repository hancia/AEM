import sys
from typing import OrderedDict
import numpy as np

import instance
from utils import pairwise


class Insertion:
    def __init__(self, city_id, place_in_solution, cost):
        self.city_id = city_id
        self.place_in_solution = place_in_solution  # 1 of should be between 0-1 in solution
        self.cost = cost

    def __str__(self):
        return f'Insertion(City {self.city_id}, In {self.place_in_solution}, Cost {self.cost})'

    def __repr__(self):
        return self.__str__()


class CheapestInsertion:
    def __init__(self, instance: instance.Instance):
        self.instance: instance = instance
        self._solution: list = []

    def run(self) -> list:
        self._solution = [0, 1, 2, 0]
        while len(set(self._solution)) < 50:
            cities_id_to_check = list(set(range(self.instance.length)) - set(self._solution))

            insertion_costs = [Insertion(city_id, i + 1, self._insertion_cost(*pair, city_id))
                               for city_id in cities_id_to_check for i, pair in enumerate(pairwise(self._solution))]
            best_city_insertion: Insertion = min(insertion_costs, key=lambda x: x.cost)  # min cost
            self._solution.insert(best_city_insertion.place_in_solution, best_city_insertion.city_id)
        return self.solution

    def _insertion_cost(self, from_city_id, to_city_id, city_id_to_insert):
        cost_before = self.instance.adjacency_matrix[from_city_id, to_city_id]
        cost_after = self.instance.adjacency_matrix[from_city_id, city_id_to_insert] + \
                     self.instance.adjacency_matrix[city_id_to_insert, to_city_id]
        cost_insertion = cost_after - cost_before
        return cost_insertion

    @property
    def solution(self) -> list:
        return self._solution

    @property
    def solution_cost(self):
        return sum([self.instance.adjacency_matrix[id_source, id_destination]
                    for id_source, id_destination in pairwise(self._solution)])
