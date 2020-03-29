# TODO add ABC and interfaces
from utils.utils import pairwise


class AbstractStrategy:

    def _get_solution_cost(self, solution) -> int:
        # if we dont want random solutions, change it to yielding before generated list
        return sum([self.instance.adjacency_matrix[id_source, id_destination]
                    for id_source, id_destination in pairwise(solution)])

    @property
    def solution(self) -> list:
        return self._solution

    @property
    def solution_cost(self) -> int:
        return self._get_solution_cost(self._solution)
