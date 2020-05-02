from collections import Counter
from heapq import nsmallest
from itertools import product

import seaborn as sns
from IPython.core.display import display
from tqdm import tqdm

from api.instance import Instance
from strategies.destroy_repair.destroy_repair import DestroyRepairLocalSearch
from strategies.iterated_local_search.iterated_local_search import IteratedLocalSearch
from strategies.local_search.local_search import LocalSearch
from strategies.ls_cache.local_search_with_cache import LocalSearchWitchCache
from strategies.multiple_local_search.multiple_local_search import MultipleStartLocalSearch
from utils.utils import draw_solution
import pandas as pd
import numpy as np

sns.set()
df = pd.DataFrame(columns=['version', 'instance', 'cost', 'time'])
for instance_name in ['kroA100', 'kroB100']:
    for p in [8,10,12,14,16]:
        instance = Instance(name=instance_name)
        solve_strategy: IteratedLocalSearch = IteratedLocalSearch(
            instance=instance,
            perturbation=p
        )
        solve_strategy.run(run_times=1)
        # raise KeyError
        costs = list(map(lambda x: x[1], solve_strategy.solutions))
        times = list(map(lambda x: x[2], solve_strategy.solutions))

        print(instance_name, min(costs), np.mean(times))

        for s, cost, time in solve_strategy.solutions:
            df = df.append(
                pd.DataFrame([[p, instance_name, cost, time]], columns=['version', 'instance', 'cost', 'time']))

        # draw_solution(
        #     instance=instance,
        #     solution=solve_strategy.solution,
        #     title=f'Local search {version}, {instance.name}, distance: {solve_strategy.solution_cost}, {neighbourhood}',
        #     save_file_name=f'{instance.name}_{min(costs)}_{version}_{neighbourhood}.png'
        # )
display(df)
