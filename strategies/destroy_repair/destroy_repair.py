import time
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from operator import attrgetter
from random import sample, seed, shuffle

from tqdm import tqdm

from api.insertion import Insertion
from api.instance import Instance
from strategies.abstract import AbstractStrategy
from utils.utils import pairwise

import numpy as np


class DestroyRepairLocalSearch(AbstractStrategy):
    def __init__(self, instance: Instance, perturbation=40, neighbourhood='edge'):
        assert neighbourhood in ['vertex', 'edge'], "Niedozwolone sąsiedztwo"
        self.instance = instance
        self.neighbourhood = neighbourhood
        self.perturbation = perturbation
        ones_number = int(self.instance.length / 2 * perturbation / 100)
        self.mask = [1] * ones_number
        self.mask.extend([0] * (int(self.instance.length / 2) - ones_number))
        self._solution: list = []
        self.solutions: list = []
        self.regret = 0

    def run(self, run_times=10):
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
        init_solutions = list()
        solution: list = sample(list(range(self.instance.length)), int(self.instance.length / 2))
        results = dict()
        i=0
        while time.time() - start <= 240:
            i+=1
            solution = self.destroy_repair(solution)
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
                        candidate[best_swap[0]], candidate[best_swap[1]] = candidate[best_swap[1]], candidate[
                            best_swap[0]]
                    else:
                        candidate = candidate[:best_swap[0] + 1] + candidate[best_swap[0] + 1:best_swap[1] + 1][::-1] + \
                                    candidate[best_swap[1] + 1:]

                if improvement_in or improvement_out:
                    solution = candidate
            results[i] = {
                'path': deepcopy(solution + [solution[0]]),
                'cost': self._get_solution_cost(solution + [solution[0]]),
                'i': i
            }
        min_cost_id = min(results, key=lambda key: results[key]['cost'])
        # print(results[min_cost_id]['i'])
        return results[min_cost_id]['path'], results[min_cost_id]['cost'], time.time() - start

    def get_value_of_change_vertices(self, s, o, r_id, i_id):
        # return difference in length of cycle, if > 0 bad, if < 0 good
        c = self.instance.adjacency_matrix
        # print(r_id, i_id, s, o)
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

    def destroy_repair(self, solution):
        s, destroy_mask = deepcopy(solution), deepcopy(self.mask)
        shuffle(destroy_mask)
        for i, bit in enumerate(destroy_mask):
            if bit and i not in [0, len(s) - 1]:
                s[i] = None
        holes = self._construct_holes(s)
        for hole in holes:
            # print(hole, s)
            hole_cities = [s[hole[0]], s[hole[1]]]
            while len(set(hole_cities)) <= hole[1] - hole[0]:
                # print(set(hole_cities))
                cities_id_to_check = list(set(range(self.instance.length)) - set(hole_cities) - set(s))
                # print(cities_id_to_check)
                city_insertions = defaultdict(list)
                for city_id in cities_id_to_check:
                    for i, pair in enumerate(pairwise(hole_cities)):
                        insertion: Insertion = Insertion(city_id, i + 1, self._insertion_cost(*pair, city_id))
                        city_insertions[str(city_id)].append(insertion)
                city_insertion_cost = self._map_insertions_on_insertion_costs(city_insertions)
                best_city_insertion: Insertion = min(city_insertion_cost, key=lambda x: x.cost)  # min cost
                hole_cities.insert(best_city_insertion.position_in_solution, best_city_insertion.city_id)
            s[hole[0]:hole[1] + 1] = deepcopy(hole_cities)
        return s

    def _construct_holes(self, s):
        holes = list()
        i = 0
        while i <= int(self.instance.length / 2) - 2:
            if s[i] is not None:
                i += 1
                continue
            j = i + 1
            while True:
                if s[j] is None:
                    j += 1
                else:
                    break
            holes.append((i - 1, j))
            i += j - i
        return holes

    def _insertion_cost(self, from_city_id, to_city_id, city_id_to_insert) -> int:
        cost_before = self.instance.adjacency_matrix[from_city_id, to_city_id]
        cost_after = self.instance.adjacency_matrix[from_city_id, city_id_to_insert] + \
                     self.instance.adjacency_matrix[city_id_to_insert, to_city_id]
        cost_insertion = cost_after - cost_before
        return cost_insertion

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
