from typing import Callable, Literal
import numpy as np

class Genetic:
    def __init__(
            self,
            fitness_function: Callable[[np.ndarray], np.float64],
            sequence_values: np.ndarray,
            crossover_type: Literal["one-point", "two-point", "uniform"] = "one-point",
            mixing_ratio: float = 0.5,
            mutation_chance: float = 0.1,
            rng: None | np.random.Generator = None
        ) -> None:
        self.fitness_function = fitness_function
        self.sequence_values = sequence_values
        self.crossover_type = crossover_type
        self.mixing_ratio = mixing_ratio
        self.mutation_chance = mutation_chance
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def mutation_op(self, sequence: np.ndarray) -> np.ndarray:
        mutation_mask = self.rng.random(sequence.shape) < self.mutation_chance
        random_values = self.rng.choice(self.sequence_values, size=sequence.shape)
        mutated = np.where(mutation_mask, random_values, sequence)
        return mutated

    def crossover_op(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if p1.shape != p2.shape:
            raise ValueError("Parents must have the same shape")

        n = p1.size

        if self.crossover_type == "one-point":
            point = self.rng.integers(1, n)
            flat1, flat2 = p1.ravel(), p2.ravel()
            child = np.concatenate([flat1[:point], flat2[point:]])
            return child.reshape(p1.shape)

        elif self.crossover_type == "two-point":
            point1, point2 = sorted(self.rng.integers(0, n, size=2))
            flat1, flat2 = p1.ravel(), p2.ravel()
            child = flat1.copy()
            child[point1:point2] = flat2[point1:point2]
            return child.reshape(p1.shape)

        elif self.crossover_type == "uniform":
            mask = self.rng.random(p1.shape) < self.mixing_ratio
            return np.where(mask, p1, p2)

        else:
            raise ValueError("Invalid crossover_type")
