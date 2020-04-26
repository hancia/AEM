from itertools import product

import seaborn as sns
from tqdm import tqdm

from api.instance import Instance
from strategies.local_search.local_search import LocalSearch
from utils.utils import draw_solution
import pandas as pd

sns.set()
df = pd.DataFrame(columns=['version', 'neigbourhood', 'cost', 'time'])
for version, neighbourhood in tqdm(product(['greedy', 'steepest'], ['vertex', 'edge'])):
    for instance_name in ['kroA100', 'kroB100']:
        instance = Instance(name=instance_name)
        solve_strategy: LocalSearch = LocalSearch(
            instance=instance,
            version=version,
            neighbourhood=neighbourhood,
        )
        solve_strategy.run(run_times=1)

        costs = list(map(lambda x: x[1], solve_strategy.solutions))

        for s, cost, time in solve_strategy.solutions:
            df = df.append(pd.DataFrame([[version, neighbourhood, cost, time]],columns=['version', 'neigbourhood', 'cost', 'time']))
        print(" ")
        print(df)
        # draw_solution(
        #     instance=instance,
        #     solution=solve_strategy.solution,
        #     title=f'Local search {version}, {instance.name}, distance: {solve_strategy.solution_cost}, {neighbourhood}',
        #     save_file_name=f'{instance.name}_{min(costs)}_{version}_{neighbourhood}.png'
        # )

