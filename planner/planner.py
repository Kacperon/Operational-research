import numpy as np
from planner.exercise import Exercise

class Planner:
    def __init__(
            self,
            exercises: np.ndarray[Exercise],
            num_muscle_groups: int,
            muscle_group_weights: np.ndarray | None = None,
            intensity_weights: np.ndarray | None = None,
            balance_weight: float = 1.0,
        ) -> None:
        """
        Initializes the Planner class with the given parameters.

        Parameters:
            exercises (np.ndarray[Exercises]): An array containing the details of the available exercises.
            num_muscle_groups (int): The number of muscle groups to consider.
            muscle_group_weights (np.ndarray, optional): An array of shape (num_muscle_groups, 1) containing the weights for each muscle group. If None, it defaults to an array of ones.
            intensity_weights (np.ndarray, optional): An array of shape (3, 1) containing the weights for each intensity level (target, synergist, stabilizer). If None, it defaults to an array of ones.
            balance_weight (float, optional): The weight for the balance component in the fitness function. If None, it defaults to 1.0.
        """
        self.exercises = exercises
        self.num_muscle_groups = num_muscle_groups
        self.muscle_group_weights = muscle_group_weights if muscle_group_weights is not None else np.ones(shape=(num_muscle_groups, 1), dtype=np.float32)
        self.intensity_weights = intensity_weights if intensity_weights is not None else np.ones(shape=(3, 1), dtype=np.float32)
        self.balance_weight = balance_weight

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
        fitness_value = self.balance_weight * np.linalg.norm(weighted_groups_avg - weighted_groups) - weighted_groups @ self.muscle_group_weights

        return fitness_value

    def get_intensity_matrix(self, individual: np.ndarray[int]):
        intensity_matrix = np.zeros(shape=(3, self.num_muscle_groups), dtype=np.int32)
        for exercise_idx in individual:
            exercise: Exercise = self.exercises[exercise_idx]
            for muscle_group in exercise.targets:
                intensity_matrix[0, muscle_group] += 1
            for muscle_group in exercise.synergists:
                intensity_matrix[1, muscle_group] += 1
            for muscle_group in exercise.stabilizers:
                intensity_matrix[2, muscle_group] += 1

        return intensity_matrix
