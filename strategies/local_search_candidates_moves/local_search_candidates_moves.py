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
        solution: list = sample(list(range(self.instance.length)), int(self.instance.length/2))
        improvement_out: bool = True
        improvement_in: bool = True
        Edge = namedtuple('Edge', 'a b')
        while improvement_out or improvement_in:
            improvement_out = False
            candidate = None
            best_value = 0

            out_of_solution = list(set(range(self.instance.length)) - set(solution))
            for remove_id, insert_id in product(range(int(self.instance.length/2)), repeat=2):
                diff = self.get_value_of_change_vertices(solution, out_of_solution, remove_id, insert_id)
                if diff < best_value:
                    candidate = deepcopy(solution)
                    candidate[remove_id] = out_of_solution[insert_id]
                    best_value = diff
                    improvement_out = True

            if improvement_out is False:
                candidate = deepcopy(solution)

            edges = list()
            sorted_neigh = np.argsort(self.instance.adjacency_matrix, axis=1)[:, 1:6]
            for i, vertex in enumerate(candidate):
                neighbours = sorted_neigh[vertex]
                true_neigh = [candidate[i - 1], candidate[(i + 1) % int(self.instance.length/2)]]
                a = candidate.index(vertex)
                for neigh in list(set(neighbours) & set(true_neigh)):
                    b = candidate.index(neigh)
                    # print(a)
                    e = None
                    if a < b:
                        e = Edge(a, b)
                    elif a > b:
                        e = Edge(b, a)
                    if e:
                        assert e.a < e.b, "tescik"
                        edges.append(e)

            best_swap = (0, 0)
            best_value = 0
            improvement_in = False

            for edge_first, edge_second in product(list(set(edges)), repeat=2):
                swap_a_id, swap_b_id = edge_first.a, edge_second.a
                if edge_first == edge_second or swap_a_id == swap_b_id:
                    continue

                if swap_b_id < swap_a_id:
                    swap_a_id, swap_b_id = swap_b_id, swap_a_id

                diff = self.get_value_of_swap_edges(candidate, swap_a_id, swap_b_id)

                if diff < best_value:
                    best_swap = (swap_a_id, swap_b_id)
                    best_value = diff
                    improvement_in = True

            if improvement_in:
                candidate = candidate[:best_swap[0] + 1] + candidate[best_swap[0] + 1:best_swap[1] + 1][::-1] + \
                            candidate[best_swap[1] + 1:]

            if improvement_in or improvement_out:
                solution = candidate

        solution += [solution[0]]
        return solution, self._get_solution_cost(solution), time.time() - start

    def get_value_of_change_vertices(self, s, o, r_id, i_id):
        # return difference in length of cycle, if > 0 bad, if < 0 good
        c = self.instance.adjacency_matrix

        now_length = c[s[r_id - 1], s[r_id]] + c[s[r_id], s[(r_id + 1) % int(self.instance.length/2)]]
        new_length = c[s[r_id - 1], o[i_id]] + c[o[i_id], s[(r_id + 1) % int(self.instance.length/2)]]
        return new_length - now_length

    def get_value_of_swap_edges(self, s, swap_a_id, swap_b_id):
        c = self.instance.adjacency_matrix

        from_e1_v, to_e1_v = s[swap_a_id], s[(swap_a_id + 1) % int(self.instance.length/2)]
        from_e2_v, to_e2_v = s[swap_b_id], s[(swap_b_id + 1) % int(self.instance.length/2)]

        diff = c[from_e1_v, from_e2_v] + c[to_e1_v, to_e2_v] - c[from_e1_v, to_e1_v] - c[from_e2_v, to_e2_v]
        return diff
