from typing import Iterable


def pairwise(list_to_pair: Iterable):
    return zip(list_to_pair, list_to_pair[1:])
