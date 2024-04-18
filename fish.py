import numpy as np
from itertools import pairwise


class Fish:
    def __init__(self, data, matrix) -> None:
        self._matrix = matrix
        self._fitness: float | None = None
        self.data = np.array(data)

    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = sum(self._matrix[i, j] for i, j in pairwise(self.data)) + self._matrix[self.data[-1], self.data[0]]
        return self._fitness

    def update(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]
        self._fitness = None

    def update_data(self, data):
        self.data = data
        self._fitness = None

    def __deepcopy__(self, memo=None) -> 'Fish':
        fish = Fish(self.data, self._matrix)
        fish._fitness = self._fitness
        return fish

    def __str__(self) -> str:
        return f'{self.data.tolist()} with fitness {self.fitness}'

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(tuple(self.data))

    def __eq__(self, other: 'Fish') -> bool:
        return tuple(self.data) == tuple(other.data)
