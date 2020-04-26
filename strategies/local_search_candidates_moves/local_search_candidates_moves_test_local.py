from itertools import product

import seaborn as sns
from tqdm import tqdm

from api.instance import Instance
from strategies.local_search.local_search import LocalSearch
from strategies.local_search_candidates_moves.local_search_candidates_moves import LocalSearchWitchCandidatesMoves
from utils.utils import draw_solution
import pandas as pd
import numpy as np

sns.set()
for instance_name in ['kroA100', 'kroB100']:
    instance = Instance(name=instance_name)
    solve_strategy: LocalSearchWitchCandidatesMoves = LocalSearchWitchCandidatesMoves(
        instance=instance,
    )
    solve_strategy.run(run_times=100)

    costs = list(map(lambda x: x[1], solve_strategy.solutions))
    times = list(map(lambda x: x[2], solve_strategy.solutions))
    print(instance_name, min(costs), np.mean(costs), np.mean(times))

    # for s, cost, time in solve_strategy.solutions:
    #     df = df.append(pd.DataFrame([[version, neighbourhood, cost, time]],columns=['version', 'neigbourhood', 'cost', 'time']))
    #
    # draw_solution(
    #     instance=instance,
    #     solution=solve_strategy.solution,
    #     title=f'Local search {version}, {instance.name}, distance: {solve_strategy.solution_cost}, {neighbourhood}',
    #     save_file_name=f'{instance.name}_{min(costs)}_{version}_{neighbourhood}.png'
    # )
