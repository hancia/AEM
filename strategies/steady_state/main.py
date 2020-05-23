from collections import Counter
from heapq import nsmallest
from itertools import product

import seaborn as sns
from IPython.core.display import display

from api.instance import Instance
from strategies.steady_state.steady_state import SteadyState
from utils.utils import draw_solution
import pandas as pd
import numpy as np

sns.set()
df = pd.DataFrame(columns=['instance', 'cost', 'time'])
for instance_name in ['kroA200']:
    instance = Instance(name=instance_name)
    solve_strategy: SteadyState = SteadyState(
        instance=instance
    )
    solve_strategy.run(run_times=1)
    # raise KeyError
    costs = list(map(lambda x: x[1], solve_strategy.solutions))
    times = list(map(lambda x: x[2], solve_strategy.solutions))

    print(instance_name, min(costs), np.mean(times))

    for s, cost, time in solve_strategy.solutions:
        df = df.append(
            pd.DataFrame([[instance_name, cost, time]], columns=['instance', 'cost', 'time']))

    draw_solution(
        instance=instance,
        solution=solve_strategy.solution,
        title=f'Local search {instance.name}, distance: {solve_strategy.solution_cost}',
        save_file_name=f'{instance.name}_{min(costs)}_.png'
    )
display(df)
