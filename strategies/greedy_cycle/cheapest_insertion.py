from collections import defaultdict
from copy import deepcopy
from math import ceil
from operator import attrgetter
from random import sample

from api import instance

from api.insertion import Insertion
from strategies.abstract import AbstractStrategy
from utils.utils import pairwise


class CheapestInsertion(AbstractStrategy):
    def __init__(self, instance: instance.Instance, regret=0, path_length_percentage=50):
        self.instance: instance = instance
        self.regret: int = regret
        self.goal_length = ceil(instance.length * path_length_percentage / 100)
        self._solution: list = []
        self.solutions: list = []

    def run(self, run_times=50) -> None:
        solutions = [self._solve(first_city_id=i) for i in range(run_times)]
        solution, best_cost = min(solutions, key=lambda x: x[1])
        self._solution = solution
        self.solutions = solutions

    def _solve(self, first_city_id=0):
        solution = [first_city_id, first_city_id]
        while len(set(solution)) < self.goal_length:

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
            sorted_insertions = sorted(insertions, key=attrgetter('cost', 'city_id'))
            cheapest_insertion = sorted_insertions[0]
            changed_insertion = deepcopy(cheapest_insertion)

            if self.regret != 0 and len(insertions) >= 2:
                for i in range(1, 1 + self.regret):
                    changed_insertion.cost += cheapest_insertion.cost - sorted_insertions[i].cost
                changed_insertion.cost *= -1  # dirty hack to avoid checking if there is regret or no during
                # choosing best insertion

            result.append(changed_insertion)
        return result
