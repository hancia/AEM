from random import seed

import seaborn as sns
import numpy as np
from tqdm import tqdm

from strategies.cheapest_insertion import CheapestInsertion
from api.instance import Instance
from strategies.local_search import LocalSearch
from utils.utils import draw_solution

sns.set()
np.random.seed(0)
seed(0)

for instance_name in tqdm(['kroA100']):
    instance = Instance(name=instance_name)
    solve_strategy: LocalSearch = LocalSearch(
        instance=instance,
        version='greedy',
        neighbourhood='vertice',
    )
    solve_strategy.run(run_times=1)

    costs = list(map(lambda x: x[1], solve_strategy.solutions))
    print(instance_name, min(costs), int(round(np.average(costs))), max(costs))

    draw_solution(
        instance=instance,
        solution=solve_strategy.solution,
        title=f'Local Search, {instance.name}, distance: {solve_strategy.solution_cost}',
        save_file_name=f'{instance.name}.png'
    )
