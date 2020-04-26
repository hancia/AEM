import time
from collections import namedtuple
from copy import deepcopy
from itertools import product
from random import sample, seed

from api.instance import Instance
from strategies.abstract import AbstractStrategy
from utils.utils import pairwise

import numpy as np


class LocalSearchWitchCandidatesMoves(AbstractStrategy):
    def __init__(self, instance: Instance, version: str = 'steepest', neighbourhood='edge'):
        self.instance = instance
        self.version = version
        self.neighbourhood = neighbourhood
        self._solution: list = []
        self.solutions: list = []

    def run(self, run_times=100):
        self.solutions = list()
        for i in range(run_times):
            self.solutions.append(self._solve_steepest(i))
        solution, _, _ = min(self.solutions, key=lambda x: x[1])
        self._solution = solution

    def _solve_steepest(self, s):
        start = time.time()
        np.random.seed(s)
        seed(s)
        # REMEMBER SOLUTION HERE DOESNT CONTAIN CYCLE!!!!!!! Append before return!
        solution: list = sample(list(range(self.instance.length)), int(self.instance.length / 2))
        improvement_out: bool = True
        Edge = namedtuple('Edge', 'a b')
        while improvement_out:
            improvement_out = False
            candidate = deepcopy(solution)
            sorted_neigh = np.argsort(self.instance.adjacency_matrix, axis=1)[:, 1:6]

            best_value = 0
            out_of_solution = list(set(range(self.instance.length)) - set(solution))
            for vertex in solution:
                neighbours = sorted_neigh[vertex]

                for neigh in neighbours:
                    if neigh in solution:
                        continue
                    diff = self.get_value_of_change_vertices(solution, out_of_solution, solution.index(vertex),
                                                             out_of_solution.index(neigh))
                    if diff < best_value:
                        candidate = deepcopy(solution)
                        candidate[solution.index(vertex)] = out_of_solution[out_of_solution.index(neigh)]
                        best_value = diff
                        improvement_out = True

            if improvement_out:
                solution = candidate

        solution += [solution[0]]
        return solution, self._get_solution_cost(solution), time.time() - start

    def get_value_of_change_vertices(self, s, o, r_id, i_id):
        # return difference in length of cycle, if > 0 bad, if < 0 good
        c = self.instance.adjacency_matrix

        now_length = c[s[r_id - 1], s[r_id]] + c[s[r_id], s[(r_id + 1) % int(self.instance.length / 2)]]
        new_length = c[s[r_id - 1], o[i_id]] + c[o[i_id], s[(r_id + 1) % int(self.instance.length / 2)]]
        return new_length - now_length

    def get_value_of_swap_edges(self, s, swap_a_id, swap_b_id):
        c = self.instance.adjacency_matrix

        from_e1_v, to_e1_v = s[swap_a_id], s[(swap_a_id + 1) % int(self.instance.length / 2)]
        from_e2_v, to_e2_v = s[swap_b_id], s[(swap_b_id + 1) % int(self.instance.length / 2)]

        diff = c[from_e1_v, from_e2_v] + c[to_e1_v, to_e2_v] - c[from_e1_v, to_e1_v] - c[from_e2_v, to_e2_v]
        return diff
