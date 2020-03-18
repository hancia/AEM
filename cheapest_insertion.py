from random import sample

import instance
from insertion import Insertion
from utils import pairwise


class CheapestInsertion:
    def __init__(self, instance: instance.Instance):
        self.instance: instance = instance
        self._solution: list = []

    def run(self, run_times=50) -> None:
        solutions = [self._solve() for _ in range(run_times)]
        solution, best_cost = min(solutions, key=lambda x: x[1])
        self._solution = solution

    def _solve(self):
        solution = self._random_initial_solution()
        while len(set(solution)) < 50:
            cities_id_to_check = list(set(range(self.instance.length)) - set(solution))

            insertion_costs = [Insertion(city_id, i + 1, self._insertion_cost(*pair, city_id))
                               for city_id in cities_id_to_check for i, pair in enumerate(pairwise(solution))]
            best_city_insertion: Insertion = min(insertion_costs, key=lambda x: x.cost)  # min cost
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

    @property
    def solution(self) -> list:
        return self._solution

    @property
    def solution_cost(self) -> int:
        return self._get_solution_cost(self._solution)
