import time
from itertools import product
from random import sample, randint
from api.instance import Instance
from strategies.abstract import AbstractStrategy
from strategies.local_search.local_search import LocalSearch

import numpy as np


class SteadyState(AbstractStrategy):
    def __init__(self, instance: Instance, neighbourhood='edge', pop_size=20, perturbation=6):
        assert neighbourhood in ['edge'], "Niedozwolone sąsiedztwo"
        self.instance = instance
        self._solution: list = []
        self.solutions: list = []
        self.pop_size = pop_size
        self.perturbation = perturbation

    def run(self, run_times=10):
        self.solutions = list()
        for i in range(run_times):
            self.solutions.append(self._solve_steady_state(i))
        solution, _, _ = min(self.solutions, key=lambda x: x[1])
        self._solution = solution

    def _solve_steady_state(self, seed):
        ls: LocalSearch = LocalSearch(instance=self.instance, version='steepest', neighbourhood='edge')
        # np.random.seed(seed)
        start = time.time()
        population = np.array([np.random.permutation(self.instance.length) \
                               for _ in range(self.pop_size)])[:, :self.instance.length // 2]
        costs = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            population[i], costs[i] = ls.solve(population[i], seed)

        while time.time() - start < 240:
            idx_1, idx_2 = np.random.randint(0, len(population), 2)
            parent_1, parent_2 = population[idx_1], population[idx_2]
            child = self.crossover(parent_1, parent_2)
            child = self.perturbate(child)
            child, cost = ls.solve(child, seed)
            in_list = False
            for p in range(len(population)):
                if len(set(child) & set(population[p])) == len(set(child)) and cost == costs[p]:
                    in_list = True
                    break
            if not in_list:
                max_id = np.argmax(costs)
                if costs[max_id] > cost:
                    population[max_id] = child
                    costs[max_id] = cost
                    # print(time.time() - start, np.min(costs), np.mean(costs), np.max(costs))

        best_id = np.argmin(costs)
        solution = list(population[best_id])
        solution += [solution[0]]
        cost= self._get_solution_cost(solution)
        print(cost)
        return solution, cost, time.time() - start

    @staticmethod
    def crossover(parent_1, parent_2):
        child = list()
        l = len(parent_1)
        h = l // 2
        start1, start2 = randint(0, l - 1), randint(0, l - 1)
        for i in range(h):
            child.append(parent_1[(start1 + i) % l])
        for i in range(l):
            if len(child) >= l:
                break
            id = (start2 + i) % l
            if parent_2[id] not in child:
                child.append(parent_2[id])
        return child

    def perturbate(self, solution):
        moves_number = self.perturbation
        half_moves = moves_number // 2
        moves = sample(list(product(range(int(self.instance.length / 2)), repeat=2)), moves_number)
        vert_moves, edge_moves = moves[:half_moves], moves[half_moves:]
        out_of_solution = list(set(range(self.instance.length)) - set(solution))
        for remove_id, insert_id in vert_moves:
            solution[remove_id], out_of_solution[insert_id] = out_of_solution[insert_id], solution[remove_id]
        for swap_a_id, swap_b_id in edge_moves:
            a, b = swap_a_id, swap_b_id
            if b < a:
                a, b = b, a
            if a == b:
                continue
            solution = solution[:a + 1] + solution[a + 1:b + 1][::-1] + \
                       solution[b + 1:]
        return solution
