from copy import deepcopy
from itertools import product
from random import sample

from api.instance import Instance
from strategies.abstract import AbstractStrategy
from utils.utils import pairwise


class LocalSearch(AbstractStrategy):
    def __init__(self, instance: Instance, version: str = 'greedy', neighbourhood='vertice'):
        assert version in ['greedy', 'steepest'], "Niedozwolona wersja kolego"
        assert neighbourhood in ['vertice', 'edge'], "Niedozwolone sąsiedztwo"
        self.instance = instance
        self.version = version
        self.neighbourhood = neighbourhood
        self._solution: list = []
        self.solutions: list = []

    def run(self, run_times=100):
        solutions = [self._solve_greedy_vertex() for _ in range(run_times)]
        solution, best_cost = min(solutions, key=lambda x: x[1])
        self._solution = solution
        self.solutions = solutions

    def _solve_greedy_vertex(self):
        # REMEMBER SOLUTION HERE DOESNT CONTAIN CYCLE!!!!!!!
        solution: list = sample(list(range(100)), 50)
        improvement: bool = True
        while improvement:
            improvement = False

            out_of_solution = list(set(range(100)) - set(solution))
            for (remove_id, insert_id, swap_a_id, swap_b_id) in product(range(50), repeat=4):
                if swap_b_id == swap_a_id:
                    continue

                change1 = self.get_value_of_change_vertices(solution, out_of_solution, remove_id, insert_id)

                candidate = deepcopy(solution)
                candidate[remove_id] = out_of_solution[insert_id]
                change2 = self.get_value_of_swap_vertices(candidate, swap_a_id, swap_b_id)

                if change1 + change2 < 0:
                    candidate[swap_a_id], candidate[swap_b_id] = candidate[swap_b_id], candidate[swap_a_id]
                    solution = candidate
                    improvement = True
                    break

        return solution, self._get_solution_cost(solution)

    def get_value_of_change_vertices(self, s, o, r_id,
                                     i_id):  # keep in mind, value not id in solution, maybe not optimal but less confusing
        # return difference in length of cycle, if > 0 bad, if < 0 good
        c = self.instance.adjacency_matrix

        now_length = c[s[r_id - 1], s[r_id]] + c[s[r_id], s[(r_id + 1) % 50]]
        new_length = c[s[r_id - 1], o[i_id]] + c[o[i_id], s[(r_id + 1) % 50]]
        return new_length - now_length

    def get_value_of_swap_vertices(self, s, a_v_id, b_v_id):
        # 1 - A - 2 ... 3 - B - 4   -> 1 - B - 2 ... 3 - A - 4
        # in typical case but for      33-34 it is 32-33-34-35 so 1-A-B-4  1-B-A-3-4
        # but best case is 0-last   cycle-> last-1  -last-0-1 so three-last-A-2 xD
        # do not try understand it xDDDDDDDD

        a_id, b_id = min(a_v_id, b_v_id), max(a_v_id, b_v_id)

        a_v_id, b_v_id = a_id, b_id
        one, two, three, four = a_v_id - 1, (a_v_id + 1) % 50, b_v_id - 1, (b_v_id + 1) % 50
        c = self.instance.adjacency_matrix

        if a_v_id == 0 and b_v_id == 50-1:
            now_length = c[s[three], s[b_v_id]] + c[s[b_v_id], s[a_v_id]] + c[s[a_v_id], s[two]]
            new_length = c[s[three], s[a_v_id]] + c[s[a_v_id], s[b_v_id]] + c[s[b_v_id], s[two]]
        elif b_v_id-a_v_id == 1:
            now_length = c[s[one], s[a_v_id]] + c[s[a_v_id], s[b_v_id]] + c[s[b_v_id], s[four]]
            new_length = c[s[one], s[b_v_id]] + c[s[b_v_id], s[a_v_id]] + c[s[a_v_id], s[four]]
        else:
            now_length = c[s[one], s[a_v_id]] + c[s[a_v_id], s[two]] + c[s[three], s[b_v_id]] + c[s[b_v_id], s[four]]
            new_length = c[s[one], s[b_v_id]] + c[s[b_v_id], s[two]] + c[s[three], s[a_v_id]] + c[s[a_v_id], s[four]]

        return new_length - now_length
