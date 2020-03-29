from random import seed

import seaborn as sns
import numpy as np
from tqdm import tqdm

from strategies.cheapest_insertion import CheapestInsertion
from api.instance import Instance
from utils.utils import draw_solution

sns.set()
np.random.seed(0)
seed(0)

for instance_name in tqdm(['kroA100', 'kroB100']):
    for regret in [0, 1]:
        instance = Instance(name=instance_name)
        solve_strategy: CheapestInsertion = CheapestInsertion(
            instance=instance,
            regret=regret,
            path_length_percentage=100,
        )
        solve_strategy.run(run_times=100)

        costs = list(map(lambda x: x[1], solve_strategy.solutions))
        print(instance_name, regret, min(costs), int(round(np.average(costs))), max(costs))

        draw_solution(
            instance=instance,
            solution=solve_strategy.solution,
            title=f'Greedy Cycle, {instance.name}, distance: {solve_strategy.solution_cost}, regret: {regret}',
            save_file_name=f'{instance.name}_{regret}.png'
        )
