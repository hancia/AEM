from collections import defaultdict
from copy import deepcopy
from random import sample

import instance

from insertion import Insertion
from utils import pairwise


class CheapestInsertion:
    def __init__(self, instance: instance.Instance, regret=0):
        self.instance: instance = instance
        self.regret: int = regret
        self._solution: list = []

    def run(self, run_times=50) -> None:
        solutions = [self._solve() for _ in range(run_times)]
        solution, best_cost = min(solutions, key=lambda x: x[1])
        self._solution = solution

    def _solve(self):
        solution = self._random_initial_solution()
        while len(set(solution)) < 50:

            cities_id_to_check = list(set(range(self.instance.length)) - set(solution))

            city_insertions = defaultdict(list)
            for city_id in cities_id_to_check:
                for i, pair in enumerate(pairwise(solution)):
                    insertion: Insertion = Insertion(city_id, i + 1, self._insertion_cost(*pair, city_id))
                    city_insertions[str(city_id)].append(insertion)

            city_insertion_cost = self._map_insertions_on_insertion_costs(city_insertions)

            best_city_insertion: Insertion = min(city_insertion_cost, key=lambda x: x.cost)  # min cost
            solution.insert(best_city_insertion.position_in_solution, best_city_insertion.city_id)
        return solution, self._get_solution_cost(solution)

    def _insertion_cost(self, from_city_id, to_city_id, city_id_to_insert) -> int:
        cost_before = self.instance.adjacency_matrix[from_city_id, to_city_id]
        cost_after = self.instance.adjacency_matrix[from_city_id, city_id_to_insert] + \
                     self.instance.adjacency_matrix[city_id_to_insert, to_city_id]
        cost_insertion = cost_after - cost_before
        return cost_insertion

    def _get_solution_cost(self, solution) -> int:
        # if we dont want random solutions, change it to yielding before generated list
        return sum([self.instance.adjacency_matrix[id_source, id_destination]
                    for id_source, id_destination in pairwise(solution)])

    def _random_initial_solution(self) -> list:
        ids = sample(range(self.instance.length), 2)
        return ids + [ids[0]]

    def _map_insertions_on_insertion_costs(self, city_insertions: defaultdict) -> list:
        """
        For each city aggregate its insertions based on regret and choose the best one
        :param city_insertions: 'city_id': [insertion(city_id,position,cost), insertion ...]
        :return: List of insertions
        """
        result = list()
        for city_id, insertions in city_insertions.items():
            sorted_insertions = sorted(insertions, key=lambda x: x.cost)
            cheapest_insertion = sorted_insertions[0]
            changed_insertion = deepcopy(cheapest_insertion)

            if self.regret != 0:
                for i in range(1, 1 + self.regret):
                    changed_insertion.cost += cheapest_insertion.cost - sorted_insertions[i].cost
                changed_insertion.cost *= -1  # dirty hack to avoid checking if there is regret or no during
                # choosing best insertion

            result.append(changed_insertion)
        return result

    @property
    def solution(self) -> list:
        return self._solution

    @property
    def solution_cost(self) -> int:
        return self._get_solution_cost(self._solution)
