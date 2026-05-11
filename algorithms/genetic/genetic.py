from typing import Callable, Literal
import numpy as np

from utils.utils import create_logger


logger = create_logger(__name__)

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

    def next_generation(
            self,
            population: np.ndarray,
            elite_count: int = 0,
        ) -> np.ndarray:
        """Create the next population by minimizing the fitness (cost) function.

        Args:
            population: Current generation with shape ``(population_size, ...)``.
            elite_count: Number of lowest-cost individuals copied directly to the next
                generation before creating offspring.

        Returns:
            A new population array with the same shape as ``population``.

        Raises:
            ValueError: If the population shape is invalid, the population is too
                small, or ``elite_count`` is outside the valid range.
        """
        if population.ndim < 2:
            raise ValueError("Population must have shape (population_size, ...)")

        population_size = population.shape[0]
        if population_size < 2:
            raise ValueError("Population size must be at least 2")

        if elite_count < 0 or elite_count >= population_size:
            raise ValueError("elite_count must be in range [0, population_size - 1]")

        fitness_scores = np.asarray(
            [self.fitness_function(individual) for individual in population],
            dtype=np.float64,
        ).reshape(population_size)

        # Convert costs into positive selection weights (lower cost => higher weight).
        finite_scores = fitness_scores[np.isfinite(fitness_scores)]
        if finite_scores.size == 0:
            logger.warning("All fitness scores are non-finite; using uniform parent selection")
            selection_probabilities = np.full(population_size, 1.0 / population_size)
            elite_indices = np.array([], dtype=int)
        else:
            safe_scores = np.where(np.isfinite(fitness_scores), fitness_scores, np.max(finite_scores))
            selection_weights = np.max(safe_scores) - safe_scores + 1e-12
            total_weight = float(np.sum(selection_weights))
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                logger.warning("Invalid GA selection weights sum=%s; using uniform parent selection", total_weight)
                selection_probabilities = np.full(population_size, 1.0 / population_size)
            else:
                selection_probabilities = selection_weights / total_weight
            elite_indices = np.argsort(safe_scores)[:elite_count] if elite_count > 0 else np.array([], dtype=int)

        next_population = np.empty_like(population)

        if elite_count > 0:
            next_population[:elite_count] = population[elite_indices]

        for i in range(elite_count, population_size):
            p1_idx, p2_idx = self.rng.choice(population_size, size=2, p=selection_probabilities, replace=True)
            child = self.crossover_op(population[p1_idx], population[p2_idx])
            child = self.mutation_op(child)
            next_population[i] = child

        return next_population
