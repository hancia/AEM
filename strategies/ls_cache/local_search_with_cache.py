import time
from collections import namedtuple
from copy import deepcopy
from itertools import product
from random import sample, seed

from sortedcontainers import SortedList

from api.instance import Instance
from strategies.abstract import AbstractStrategy
from utils.utils import pairwise

import numpy as np

Insertion = namedtuple('Insertion', 'a b cost')


class LocalSearchWitchCache(AbstractStrategy):
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
        improvement_in: bool = True
        cache = list()

        for swap_a_id, swap_b_id in product(range(int(self.instance.length / 2)), repeat=2):
            if swap_b_id <= swap_a_id:
                continue
            diff = self.get_value_of_swap_edges(solution, swap_a_id, swap_b_id)
            if diff < 0:
                cache.append(Insertion(swap_a_id, swap_b_id, diff))
        cache = SortedList(cache, key=lambda x: x.cost)

        while improvement_out or improvement_in:
            improvement_out = False
            candidate = None
            best_value = 0

            out_of_solution = list(set(range(self.instance.length)) - set(solution))
            best_v_id = None
            for remove_id, insert_id in product(range(int(self.instance.length / 2)), repeat=2):
                diff = self.get_value_of_change_vertices(solution, out_of_solution, remove_id, insert_id)
                if diff < best_value:
                    candidate = deepcopy(solution)
                    candidate[remove_id] = out_of_solution[insert_id]
                    best_v_id = remove_id
                    best_value = diff
                    improvement_out = True

            if not improvement_out:
                candidate = deepcopy(solution)

            if improvement_out:
                cache = self.remove_vertex_from_cache(cache, [best_v_id])
                cache = self.cache_append_every_possible_pair_with_vertex(cache, candidate, [best_v_id])

            best_swap = (0, 0)
            improvement_in = False
            for i in range(len(cache)):
                insertion = cache[i]
                diff = self.get_value_of_swap_edges(candidate, insertion.a, insertion.b)
                if diff == insertion.cost:
                    x = self._get_solution_cost(candidate + [candidate[0]])
                    best_swap = [insertion.a, insertion.b]
                    candidate = candidate[:best_swap[0] + 1] + candidate[best_swap[0] + 1:best_swap[1] + 1][::-1] + \
                                candidate[best_swap[1] + 1:]
                    improvement_in = True
                    break

            for j, item in enumerate(cache):
                if j <= i:
                    cache.remove(cache[j])

            if improvement_in:
                indices_to_change = [best_swap[0], best_swap[1], (best_swap[0] + 1) % int(self.instance.length / 2),
                                     (best_swap[1] + 1) % int(self.instance.length / 2)]
                cache = self.remove_vertex_from_cache(cache, indices_to_change)
                cache = self.cache_append_every_possible_pair_with_vertex(cache, candidate, indices_to_change)

            if improvement_in or improvement_out:
                solution = candidate
            # break
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

    def cache_append_every_possible_pair_with_vertex(self, cache, solution, vertex_id):
        for swap_a_id, swap_b_id in product(vertex_id, range(int(self.instance.length/2))):
            if swap_a_id != swap_b_id:
                if swap_a_id > swap_b_id:
                    swap_a_id, swap_b_id = swap_b_id, swap_a_id
                diff = self.get_value_of_swap_edges(solution, swap_a_id, swap_b_id)
                if diff < 0:
                    cache.add(Insertion(swap_a_id, swap_b_id, diff))
        return cache

    @staticmethod
    def remove_vertex_from_cache(cache, vertex_ids):
        items_to_remove = list()
        for item in cache:
            if len(set(vertex_ids) & set([item.a, item.b])):
                items_to_remove.append(item)
        for item in items_to_remove:
            cache.remove(item)
        return cache
