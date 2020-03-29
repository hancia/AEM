import time
from copy import deepcopy
from itertools import product
from random import sample, seed

from api.instance import Instance
from strategies.abstract import AbstractStrategy
from utils.utils import pairwise

import numpy as np


class LocalSearch(AbstractStrategy):
    def __init__(self, instance: Instance, version: str = 'greedy', neighbourhood='vertex'):
        assert version in ['greedy', 'steepest'], "Niedozwolona wersja kolego"
        assert neighbourhood in ['vertex', 'edge'], "Niedozwolone sąsiedztwo"
        self.instance = instance
        self.version = version
        self.neighbourhood = neighbourhood
        self._solution: list = []
        self.solutions: list = []

    def run(self, run_times=100):
        self.solutions = [self._solve(i) for i in range(run_times)]
        solution, _, _ = min(self.solutions, key=lambda x: x[1])
        self._solution = solution

    def _solve(self, s):
        start = time.time()
        np.random.seed(s)
        seed(s)

        # REMEMBER SOLUTION HERE DOESNT CONTAIN CYCLE!!!!!!! Append before return!
        solution: list = sample(list(range(100)), 50)
        improvement: bool = True
        while improvement:
            improvement = False
            candidate = None
            best_value = 0

            out_of_solution = list(set(range(100)) - set(solution))
            for remove_id, insert_id in product(range(50), repeat=2):
                diff = self.get_value_of_change_vertices(solution, out_of_solution, remove_id, insert_id)
                if diff < best_value:
                    candidate = deepcopy(solution)
                    candidate[remove_id] = out_of_solution[insert_id]
                    best_value = diff
                    if self.version == 'greedy':
                        break

            if candidate is None:
                break

            best_swap = (0, 0)
            best_value = 0
            for swap_a_id, swap_b_id in product(range(50), repeat=2):
                if swap_b_id <= swap_a_id:
                    continue

                if self.neighbourhood == 'vertex':
                    diff = self.get_value_of_swap_vertices(candidate, swap_a_id, swap_b_id)
                else:
                    diff = self.get_value_of_swap_edges(candidate, swap_a_id, swap_b_id)

                if diff < best_value:
                    best_swap = (swap_a_id, swap_b_id)
                    best_value = diff
                    improvement = True
                    if self.version == 'greedy':
                        break

            if improvement:
                if self.neighbourhood == 'vertex':
                    candidate[best_swap[0]], candidate[best_swap[1]] = candidate[best_swap[1]], candidate[best_swap[0]]
                else:
                    candidate = candidate[:best_swap[0] + 1] + candidate[best_swap[0] + 1:best_swap[1] + 1][::-1] + \
                                candidate[best_swap[1] + 1:]
                solution = candidate

        solution += [solution[0]]
        return solution, self._get_solution_cost(solution), time.time() - start

    def get_value_of_change_vertices(self, s, o, r_id, i_id):
        # return difference in length of cycle, if > 0 bad, if < 0 good
        c = self.instance.adjacency_matrix

        now_length = c[s[r_id - 1], s[r_id]] + c[s[r_id], s[(r_id + 1) % 50]]
        new_length = c[s[r_id - 1], o[i_id]] + c[o[i_id], s[(r_id + 1) % 50]]
        return new_length - now_length

    def get_value_of_swap_vertices(self, s, a_v_id, b_v_id):
        # 1 - A - 2 ... 3 - B - 4   -> 1 - B - 2 ... 3 - A - 4
        # in typical case but for      33-34 it is 32-33-34-35 so 1-A-B-4  1-B-A-4
        # but best case is 0-last   cycle-> last-1  -last-0-1 so three-last-A-2 xD
        # do not try understand it xDDDDDDDD

        one, two, three, four = a_v_id - 1, (a_v_id + 1) % 50, b_v_id - 1, (
                b_v_id + 1) % 50  # -1 is a correct idx in solution ;)
        c = self.instance.adjacency_matrix

        if a_v_id == 0 and b_v_id == 50 - 1:
            now_length = c[s[three], s[b_v_id]] + c[s[b_v_id], s[a_v_id]] + c[s[a_v_id], s[two]]
            new_length = c[s[three], s[a_v_id]] + c[s[a_v_id], s[b_v_id]] + c[s[b_v_id], s[two]]
        elif b_v_id - a_v_id == 1:
            now_length = c[s[one], s[a_v_id]] + c[s[a_v_id], s[b_v_id]] + c[s[b_v_id], s[four]]
            new_length = c[s[one], s[b_v_id]] + c[s[b_v_id], s[a_v_id]] + c[s[a_v_id], s[four]]
        else:
            now_length = c[s[one], s[a_v_id]] + c[s[a_v_id], s[two]] + c[s[three], s[b_v_id]] + c[s[b_v_id], s[four]]
            new_length = c[s[one], s[b_v_id]] + c[s[b_v_id], s[two]] + c[s[three], s[a_v_id]] + c[s[a_v_id], s[four]]

        return new_length - now_length

    def get_value_of_swap_edges(self, s, swap_a_id, swap_b_id):
        c = self.instance.adjacency_matrix

        from_e1_v, to_e1_v = s[swap_a_id], s[(swap_a_id + 1) % 50]
        from_e2_v, to_e2_v = s[swap_b_id], s[(swap_b_id + 1) % 50]

        diff = c[from_e1_v, from_e2_v] + c[to_e1_v, to_e2_v] - c[from_e1_v, to_e1_v] - c[from_e2_v, to_e2_v]
        return diff