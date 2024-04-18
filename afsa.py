import numpy as np
from operator import attrgetter
from copy import deepcopy

from fish import Fish


class AFSA:

    @staticmethod
    def __ordered_crossing(parent1: np.ndarray, parent2: np.ndarray):
        n = parent1.shape[0]
        rng = np.random.default_rng()
        idx1, idx2 = np.sort(rng.choice(range(1, n - 1), replace=False, size=2))

        parents = np.vstack((parent1, parent2))
        children = np.full_like(parents, -1)
        
        children[:, idx1:idx2 + 1] = parents[::-1, idx1:idx2 + 1]
        genes = np.c_[parents[:, idx2 + 1:], parents[:, :idx2 + 1]]
        
        child1, _ = children
        
        for child, gene in zip((child1, ), genes):
            for i in range(idx2 + 1, n):
                for gen_idx in range(n):
                    if gene[gen_idx] not in child:
                        child[i] = gene[gen_idx]
                        break
            for i in range(idx1):
                for gen_idx in range(n):
                    if gene[gen_idx] not in child:
                        child[i] = gene[gen_idx]
                        break

        return child1, None

    @staticmethod
    def _init_population(n: int, N: int, *args) -> tuple[Fish]:
        population = np.tile(np.arange(n), (N, 1))
        for i in range(N):
            np.random.shuffle(population[i])
        return tuple(Fish(data, *args) for data in population)

    @staticmethod
    def _is_finished(iteration: int, max_iter: int) -> bool:
        return iteration >= max_iter

    @staticmethod
    def _find_in_diapason(fish: Fish, v: float, population: tuple[Fish]) -> tuple[Fish]:  # TODO: по совпадению городов в i местах
        return [
            fish_ for fish_ in population
            if abs(fish.fitness - fish_.fitness) <= v and id(fish_) != id(fish)
        ]

    @staticmethod
    def _random_move(fish: Fish) -> None:
        idx = np.random.choice(fish.data, replace=False, size=2)
        fish.update(idx[0], idx[1])

    @staticmethod
    def _search_move(fish: Fish, fishes_in_diapason: tuple[Fish]) -> None:
        target_fish = fishes_in_diapason[np.random.randint(len(fishes_in_diapason))]
        y_data, _ = AFSA.__ordered_crossing(fish.data, target_fish.data)
        fish.update_data(y_data)

    @staticmethod
    def _swarm_move(fish: Fish, fishes_in_diapason: tuple[Fish]) -> None:
        for fish_ in fishes_in_diapason:
            y_data, _ = AFSA.__ordered_crossing(fish.data, fish_.data)
            fish.update_data(y_data)

    @staticmethod
    def _persecution_move(fish: Fish, target_fish: Fish) -> None:
        y_data, _ = AFSA.__ordered_crossing(fish.data, target_fish.data)
        fish.update_data(y_data)

    @staticmethod
    def _jump_move(fish: Fish, k: float) -> None:
        for _ in range(len(fish.data)):
            if np.random.uniform() < k:
                idx = np.random.choice(fish.data, replace=False, size=2)
                fish.update(idx[0], idx[1])

    @staticmethod
    def _get_centroid(fishes_in_diapason: tuple[Fish]) -> float:
        return sum(fish.fitness for fish in fishes_in_diapason) / len(fishes_in_diapason)

    @staticmethod
    def _get_optimum_in_diapason(fishes_in_diapason: tuple[Fish]) -> Fish:
        return min(fishes_in_diapason, key=attrgetter('fitness'))

    @staticmethod
    def _is_stagnate(eta):
        return np.random.uniform() < eta

    @staticmethod
    def resolve(
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

        population = AFSA._init_population(matrix.shape[0], N, matrix)

        iteration = 1

        while not AFSA._is_finished(iteration, max_iter):
            y_population = deepcopy(population)

            for i, fish in enumerate(population):
                y_fish = y_population[i]
                fishes_in_diapason = AFSA._find_in_diapason(fish, v, population)  # TODO: on population, change y_population

                if not fishes_in_diapason:
                    AFSA._random_move(y_fish)
                elif len(fishes_in_diapason) > theta:
                    AFSA._search_move(y_fish, fishes_in_diapason)
                else:
                    if AFSA._get_centroid(fishes_in_diapason) < fish.fitness:
                        AFSA._swarm_move(y_fish, fishes_in_diapason)
                    else:
                        AFSA._search_move(y_fish, fishes_in_diapason)

                    optimum_in_diapason = AFSA._get_optimum_in_diapason(fishes_in_diapason)

                    if optimum_in_diapason.fitness < fish.fitness:
                        AFSA._persecution_move(y_fish, optimum_in_diapason)
                    else:
                        AFSA._search_move(y_fish, fishes_in_diapason)

            population = [min(xy_fish, key=attrgetter('fitness')) for xy_fish in zip(population, y_population)]

            if AFSA._is_stagnate(eta):
                fish = population[np.random.randint(N)]
                AFSA._jump_move(fish, k)

            iteration += 1
        
        return sorted(set(population), key=attrgetter('fitness'))
