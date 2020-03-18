from random import seed

import seaborn as sns
import numpy as np
from cheapest_insertion import CheapestInsertion
from instance import Instance
from utils import draw_solution

sns.set()
np.random.seed(0)
seed(0)

instance = Instance(name='kroA100')
solve_strategy: CheapestInsertion = CheapestInsertion(instance=instance, regret=1)
solve_strategy.run(run_times=50)

draw_solution(
    instance=instance,
    solution=solve_strategy.solution,
    title=f'Cheapest Insertion, distance: {solve_strategy.solution_cost}'
)
