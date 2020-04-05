from typing import Iterable

from api.instance import Instance
import seaborn as sns
import matplotlib.pyplot as plt


def pairwise(list_to_pair: Iterable):
    return zip(list_to_pair, list_to_pair[1:])


def draw_solution(instance: Instance, solution: list, title: str = None, save_file_name: str = None) -> None:
    ax = sns.scatterplot(instance.city_coords[:, 0], instance.city_coords[:, 1], color='black', zorder=5)

    for id_source, id_destination in pairwise(solution):
        ax.annotate('', xy=instance.city_coords[id_source], xytext=instance.city_coords[id_destination],
                    arrowprops=dict(arrowstyle='-', color='red', connectionstyle="arc3"))

    for i, coords in enumerate(instance.city_coords):
        ax.text(coords[0] - 35, coords[1] + 25, str(i), size=6)

    if title is not None:
        ax.set_title(title)

    first_vertex = solution[0]
    ax.scatter(instance.city_coords[first_vertex, 0], instance.city_coords[first_vertex, 1], zorder=6)
    if save_file_name is not None:
        plt.savefig(save_file_name)
    plt.show()
