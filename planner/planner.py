import numpy as np
from planner.exercise import Exercise

class Planner:
    def __init__(
            self,
            exercises: np.ndarray[Exercise],
            num_muscle_groups: int | None = None,
            muscle_group_weights: np.ndarray | None = None,
            intensity_weights: np.ndarray | None = None,
            balance_weight: float = 1.0,
            normalize_helper_muscles: bool = True,
        ) -> None:
        """
        Initializes the Planner class with the given parameters.

        Parameters:
            exercises (np.ndarray[Exercise]): Array of available exercises.
            num_muscle_groups (int | None): Optional explicit number of muscle
                groups. If omitted, it is inferred from unique muscle names
                across all exercises.
            muscle_group_weights (np.ndarray | None): Optional array of shape
                (num_muscle_groups, 1). If None, defaults to ones.
            intensity_weights (np.ndarray | None): Optional array of shape
                (3, 1) for target/synergist/stabilizer intensities. If None,
                defaults to ones.
            balance_weight (float): Weight for the balance term in the fitness
                function.
            normalize_helper_muscles (bool): If True, split synergist and
                stabilizer contribution equally across the muscles in each role.
        """
        self.exercises = exercises

        self.muscle_group2idx, self.idx2muscle_group = self._build_muscle_group_index()
        inferred_group_count = len(self.idx2muscle_group)
        self.num_muscle_groups = inferred_group_count if num_muscle_groups is None else num_muscle_groups
        if self.num_muscle_groups < inferred_group_count:
            raise ValueError(
                "num_muscle_groups is smaller than the number of unique muscle groups in exercises."
            )

        self.muscle_group_weights = muscle_group_weights if muscle_group_weights is not None else np.ones(shape=(self.num_muscle_groups, 1), dtype=np.float32)
        self.intensity_weights = intensity_weights if intensity_weights is not None else np.ones(shape=(3, 1), dtype=np.float32)
        self.balance_weight = balance_weight
        self.normalize_helper_muscles = normalize_helper_muscles
        self.fitness_evaluations = 0

    @staticmethod
    def _normalize_muscle_name(name: str) -> str:
        """Normalize muscle names so equivalent labels map to one index."""
        return " ".join(name.strip().lower().split())

    def _build_muscle_group_index(self) -> tuple[dict[str, int], list[str]]:
        """Build bidirectional mapping between muscle names and matrix indices."""
        muscle_group2idx: dict[str, int] = {}
        idx2muscle_group: list[str] = []

        for exercise in self.exercises:
            for muscles in (exercise.targets, exercise.synergists, exercise.stabilizers):
                for muscle_name in muscles:
                    key = self._normalize_muscle_name(muscle_name)
                    if key not in muscle_group2idx:
                        muscle_group2idx[key] = len(idx2muscle_group)
                        idx2muscle_group.append(key)

        return muscle_group2idx, idx2muscle_group

    def initialize_population(self, population_size: int, num_exercises_to_plan: int) -> np.ndarray:
        """
        Initializes a population of candidate solutions for a genetic algorithm.

        Parameters:
            population_size (int): The number of individuals in the population.
            num_exercises_to_plan (int): The number of exercises to be planned.

        Returns:
            np.ndarray: A 2D array representing the initialized population, where 0-dimension represents individuals, 1-dimension represents the exercise index.
        """
        population = np.random.randint(0, self.exercises.shape[0], size=(population_size, num_exercises_to_plan), dtype=np.int32)

        return population

    def evaluate_fitness(self, population: np.ndarray):
        """
        Evaluates the fitness of each individual in the population.

        Parameters:
            population (np.ndarray): A 2D array representing the population, where 0-dimension represents individuals, 1-dimension represents muscle type (target, synergist, stabilizer) and 2-dimension represents the exercise index.

        Returns:
            np.ndarray: A 1D array containing the fitness values for each individual in the population.
        """
        fitness = np.array([self.fitness_function(individual) for individual in population])

        return fitness

    def fitness_function(self, individual: np.ndarray):
        """
        Computes the fitness of a single individual based on the defined cost function.

        Parameters:
            individual (np.ndarray): A 2D array representing the genes of an individual, where 0-dimension represents muscle type (target, synergist, stabilizer) and 1-dimension represents the exercise index.

        Returns:
            float: The computed fitness value for the individual.
        """
        intensity_matrix = self.get_intensity_matrix(individual)
        weighted_groups = self.intensity_weights.T @ intensity_matrix
        weighted_groups_avg = np.mean(weighted_groups)
        fitness_value = self.balance_weight * np.linalg.norm(weighted_groups_avg - weighted_groups) - float((weighted_groups @ self.muscle_group_weights).sum())
        self.fitness_evaluations += 1

        return float(fitness_value)

    def get_intensity_matrix(self, individual: np.ndarray[int]):
        intensity_matrix = np.zeros(shape=(3, self.num_muscle_groups), dtype=np.float32)
        for exercise_idx in individual:
            exercise: Exercise = self.exercises[exercise_idx]

            for muscle_group in exercise.targets:
                muscle_idx = self.muscle_group2idx[self._normalize_muscle_name(muscle_group)]
                intensity_matrix[0, muscle_idx] += 1

            if exercise.synergists:
                synergist_share = 1.0 / len(exercise.synergists) if self.normalize_helper_muscles else 1.0
                for muscle_group in exercise.synergists:
                    muscle_idx = self.muscle_group2idx[self._normalize_muscle_name(muscle_group)]
                    intensity_matrix[1, muscle_idx] += synergist_share

            if exercise.stabilizers:
                stabilizer_share = 1.0 / len(exercise.stabilizers) if self.normalize_helper_muscles else 1.0
                for muscle_group in exercise.stabilizers:
                    muscle_idx = self.muscle_group2idx[self._normalize_muscle_name(muscle_group)]
                    intensity_matrix[2, muscle_idx] += stabilizer_share

        return intensity_matrix
