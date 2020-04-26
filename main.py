from collections import defaultdict
from itertools import product
from random import seed

import seaborn as sns
import numpy as np
from tqdm import tqdm

from strategies.greedy_cycle.cheapest_insertion import CheapestInsertion
from api.instance import Instance
from strategies.local_search.local_search import LocalSearch
from strategies.local_search_candidates_moves.local_search_candidates_moves import LocalSearchWitchCandidatesMoves
from strategies.ls_cache.local_search_with_cache import LocalSearchWitchCache
from utils.utils import draw_solution
import pandas as pd

sns.set()
df = pd.DataFrame(columns=['strategy', 'instance', 'cost', 'time'])
for instance_name in ['kroA200', 'kroB200']:
    instance = Instance(name=instance_name)
    solve_strategy1: LocalSearch = LocalSearch(
        instance=instance,
        version="steepest",
        neighbourhood="edge",
    )
    solve_strategy2: LocalSearchWitchCandidatesMoves = LocalSearchWitchCandidatesMoves(instance=instance)
    solve_strategy3: LocalSearchWitchCache = LocalSearchWitchCache(instance=instance)

    solve_strategies = {}
    solve_strategies["Local_search"] = solve_strategy1
    solve_strategies["LM"] = solve_strategy3
    solve_strategies["Candidate_moves"] = solve_strategy2

    for str_id in solve_strategies.keys():
        solve_strategy = solve_strategies[str_id]
        solve_strategy.run(run_times=10)
        for s, cost, time in solve_strategy.solutions:
            df = df.append(
                pd.DataFrame([[str_id, instance_name, cost, time]], columns=['strategy', 'instance', 'cost', 'time']))
        costs = list(map(lambda x: x[1], solve_strategy.solutions))
        draw_solution(
            instance=instance,
            solution=solve_strategy.solution,
            title=f'{str_id}, {instance.name}, distance: {solve_strategy.solution_cost}, ',
            save_file_name=f'{instance.name}_{min(costs)}_{str_id}.png'
        )
print(df)