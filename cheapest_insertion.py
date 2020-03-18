from typing import OrderedDict
import numpy as np

import instance


class CheapestInsertion:
    def __init__(self, instance: instance):
        self.instance: instance = instance
        self._solution: list = []

    def run(self) -> list:
        self._solution = [0, 1, 0]
        return self.solution

    @property
    def solution(self) -> list:
        return self._solution
