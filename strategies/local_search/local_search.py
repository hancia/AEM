import time
from copy import deepcopy
from itertools import product
from random import sample, seed

from tqdm import tqdm

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
        self.solutions = list()
        for i in tqdm(range(run_times)):
            if self.version == 'greedy':
                self.solutions.append(self._solve_greedy(i))
            else:
                self.solutions.append(self._solve_steepest(i))
        solution, _, _ = min(self.solutions, key=lambda x: x[1])
        self._solution = solution

    def solve(self, solution, seed):
        solution, a, b = self._solve_steepest(seed, list(solution), cycle=False)
        return solution, a

    def _solve_steepest(self, s, sol=None, cycle=True):
        start = time.time()
        # np.random.seed(s)
        # seed(s)
        # REMEMBER SOLUTION HERE DOESNT CONTAIN CYCLE!!!!!!! Append before return!
        if sol is None:
            solution: list = sample(list(range(self.instance.length)), int(self.instance.length / 2))
        else:
            solution = sol
        improvement_out: bool = True
        improvement_in: bool = True
        while improvement_out or improvement_in:
            improvement_out = False
            candidate = None
            best_value = 0

            out_of_solution = list(set(range(self.instance.length)) - set(solution))
            for remove_id, insert_id in product(range(int(self.instance.length / 2)), repeat=2):
                diff = self.get_value_of_change_vertices(solution, out_of_solution, remove_id, insert_id)
                if diff < best_value:
                    candidate = deepcopy(solution)
                    candidate[remove_id] = out_of_solution[insert_id]
                    best_value = diff
                    improvement_out = True

            if improvement_out is False:
                candidate = deepcopy(solution)

            best_swap = (0, 0)
            best_value = 0
            improvement_in = False
            for swap_a_id, swap_b_id in product(range(int(self.instance.length / 2)), repeat=2):
                if swap_b_id <= swap_a_id:
                    continue

                if self.neighbourhood == 'vertex':
                    diff = self.get_value_of_swap_vertices(candidate, swap_a_id, swap_b_id)
                else:
                    diff = self.get_value_of_swap_edges(candidate, swap_a_id, swap_b_id)

                if diff < best_value:
                    best_swap = (swap_a_id, swap_b_id)
                    best_value = diff
                    improvement_in = True

            if improvement_in:
                if self.neighbourhood == 'vertex':
                    candidate[best_swap[0]], candidate[best_swap[1]] = candidate[best_swap[1]], candidate[best_swap[0]]
                else:
                    candidate = candidate[:best_swap[0] + 1] + candidate[best_swap[0] + 1:best_swap[1] + 1][::-1] + \
                                candidate[best_swap[1] + 1:]

            if improvement_in or improvement_out:
                solution = candidate

        full_sol = solution + [solution[0]]
        if cycle:
            solution=full_sol
        return solution, self._get_solution_cost(full_sol), time.time() - start

    def _solve_greedy(self, s):
        start = time.time()
        # np.random.seed(s)
        # seed(s)
        # REMEMBER SOLUTION HERE DOESNT CONTAIN CYCLE!!!!!!! Append before return!
        solution: list = sample(list(range(self.instance.length)), int(self.instance.length / 2))
        improvement_out: bool = True
        improvement_in: bool = True
        while improvement_out or improvement_in:
            order = sample(['out', 'in'], 2)
            for imp in order:
                candidate = deepcopy(solution)
                if imp == 'out':
                    improvement_out = False

                    out_of_solution = list(set(range(self.instance.length)) - set(solution))
                    for remove_id, insert_id in product(range(int(self.instance.length / 2)), repeat=2):
                        diff = self.get_value_of_change_vertices(solution, out_of_solution, remove_id, insert_id)
                        if diff < 0:
                            candidate = deepcopy(solution)
                            candidate[remove_id] = out_of_solution[insert_id]
                            improvement_out = True
                            break

                if imp == 'in':
                    improvement_in = False

                    for swap_a_id, swap_b_id in product(range(int(self.instance.length / 2)), repeat=2):
                        if swap_b_id <= swap_a_id:
                            continue

                        if self.neighbourhood == 'vertex':
                            diff = self.get_value_of_swap_vertices(candidate, swap_a_id, swap_b_id)
                        else:
                            diff = self.get_value_of_swap_edges(candidate, swap_a_id, swap_b_id)

                        if diff < 0:
                            improvement_in = True

                            if self.neighbourhood == 'vertex':
                                candidate[swap_a_id], candidate[swap_b_id] = candidate[swap_b_id], candidate[
                                    swap_a_id]
                            else:
                                candidate = candidate[:swap_a_id + 1] + candidate[swap_a_id + 1:swap_b_id + 1][::-1] + \
                                            candidate[swap_b_id + 1:]

            if improvement_in or improvement_out:
                solution = candidate

        solution += [solution[0]]
        return solution, self._get_solution_cost(solution), time.time() - start

    def get_value_of_change_vertices(self, s, o, r_id, i_id):
        # return difference in length of cycle, if > 0 bad, if < 0 good
        c = self.instance.adjacency_matrix

        now_length = c[s[r_id - 1], s[r_id]] + c[s[r_id], s[(r_id + 1) % int(self.instance.length / 2)]]
        new_length = c[s[r_id - 1], o[i_id]] + c[o[i_id], s[(r_id + 1) % int(self.instance.length / 2)]]
        return new_length - now_length

    def get_value_of_swap_vertices(self, s, a_v_id, b_v_id):
        # typical case: 1 - A - 2 ... 3 - B - 4   -> 1 - B - 2 ... 3 - A - 4
        # neighbours eg: 33-34 it is 32-33-34-35 -> 1-A-B-4  1-B-A-4
        # first and last: A(0)-B(50) [it is reversed!], 3-B-A-2 -> 3-A-B-2

        one, two, three, four = a_v_id - 1, (a_v_id + 1) % int(self.instance.length / 2), b_v_id - 1, (
                b_v_id + 1) % int(self.instance.length / 2)  # -1 is a correct idx in solution ;)
        c = self.instance.adjacency_matrix

        if a_v_id == 0 and b_v_id == int(self.instance.length / 2) - 1:
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

        from_e1_v, to_e1_v = s[swap_a_id], s[(swap_a_id + 1) % int(self.instance.length / 2)]
        from_e2_v, to_e2_v = s[swap_b_id], s[(swap_b_id + 1) % int(self.instance.length / 2)]

        diff = c[from_e1_v, from_e2_v] + c[to_e1_v, to_e2_v] - c[from_e1_v, to_e1_v] - c[from_e2_v, to_e2_v]
        return diff
