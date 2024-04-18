import numpy as np
from operator import attrgetter
from copy import deepcopy

from fish import Fish


class AFSA:

    def ordered_crossing(self, parent1: np.ndarray, parent2: np.ndarray):
        n = parent1.shape[0]
        rng = np.random.default_rng()
        idx1, idx2 = np.sort(rng.choice(range(1, n - 1), replace=False, size=2))

        child1 = np.empty(n, dtype=parent1.dtype)
        child1[idx1:idx2 + 1] = parent1[idx1:idx2 + 1]
        
        gene_pool = np.concatenate((parent2[idx2 + 1:], parent2[:idx2 + 1]))
        gene_pool_idx = 0
        for i in range(n):
            if i < idx1 or i > idx2:
                while gene_pool[gene_pool_idx] in child1[idx1:idx2 + 1]:
                    gene_pool_idx += 1
                child1[i] = gene_pool[gene_pool_idx]
                gene_pool_idx += 1

        return child1, None

    def init_population(self, n: int, N: int, *args) -> tuple[Fish]:
        population = np.tile(np.arange(n), (N, 1))
        for i in range(N):
            np.random.shuffle(population[i])
        return tuple(Fish(data, *args) for data in population)

    def is_finished(self, iteration: int, max_iter: int) -> bool:
        return iteration >= max_iter

    def find_in_diapason(self, fish: Fish, v: float, population: tuple[Fish]) -> tuple[Fish]:
        return [
            fish_ for fish_ in population
            if abs(fish.fitness - fish_.fitness) <= v and id(fish_) != id(fish)
        ]

    def random_move(self, fish: Fish) -> None:
        idx = np.random.choice(fish.data, replace=False, size=2)
        fish.update(idx[0], idx[1])

    def search_move(self, fish: Fish, fishes_in_diapason: tuple[Fish]) -> None:
        target_fish = fishes_in_diapason[np.random.randint(len(fishes_in_diapason))]
        y_data, _ = self.ordered_crossing(fish.data, target_fish.data)
        fish.update_data(y_data)

    def swarm_move(self, fish: Fish, fishes_in_diapason: tuple[Fish]) -> None:
        for fish_ in fishes_in_diapason:
            y_data, _ = self.ordered_crossing(fish.data, fish_.data)
            fish.update_data(y_data)

    def persecution_move(self, fish: Fish, target_fish: Fish) -> None:
        y_data, _ = self.ordered_crossing(fish.data, target_fish.data)
        fish.update_data(y_data)

    def jump_move(self, fish: Fish, k: float) -> None:
        for _ in range(len(fish.data)):
            if np.random.uniform() < k:
                idx = np.random.choice(fish.data, replace=False, size=2)
                fish.update(idx[0], idx[1])

    def get_centroid(fishes_in_diapason: tuple[Fish]) -> float:
        return sum(fish.fitness for fish in fishes_in_diapason) / len(fishes_in_diapason)

    def get_optimum_in_diapason(fishes_in_diapason: tuple[Fish]) -> Fish:
        return min(fishes_in_diapason, key=attrgetter('fitness'))

    def is_stagnate(self, eta):
        return np.random.uniform() < eta

    def resolve(
        self,
        adjacency_matrix,
        max_iter: int,
        N: int,
        v: float,
        theta: float,
        eta: float,
        k: float,
    ) -> list[Fish]:
        """
        Args:
            adjacency_matrix: Матрица путей
            N: количество рыб
            v: радиус визуального диапазона
            theta: параметр отношения заполненности [0, 1 - 1/N]
            eta: вероятность стагнации
            k: вероятность скачка
        """

        matrix = np.array(adjacency_matrix).astype(float)

        population = self.init_population(matrix.shape[0], N, matrix)

        iteration = 1

        while not self.is_finished(iteration, max_iter):
            y_population = deepcopy(population)

            for i, fish in enumerate(population):
                y_fish = y_population[i]
                fishes_in_diapason = self.find_in_diapason(fish, v, population) 

                if not fishes_in_diapason:
                    self.random_move(y_fish)
                elif len(fishes_in_diapason) > theta:
                    self.search_move(y_fish, fishes_in_diapason)
                else:
                    if self.get_centroid(fishes_in_diapason) < fish.fitness:
                        self.swarm_move(y_fish, fishes_in_diapason)
                    else:
                        self.search_move(y_fish, fishes_in_diapason)

                    optimum_in_diapason = self.get_optimum_in_diapason(fishes_in_diapason)

                    if optimum_in_diapason.fitness < fish.fitness:
                        self.persecution_move(y_fish, optimum_in_diapason)
                    else:
                        self.search_move(y_fish, fishes_in_diapason)

            population = [min(xy_fish, key=attrgetter('fitness')) for xy_fish in zip(population, y_population)]

            if self.is_stagnate(eta):
                fish = population[np.random.randint(N)]
                self.jump_move(fish, k)

            iteration += 1
        
        return sorted(set(population), key=attrgetter('fitness'))
