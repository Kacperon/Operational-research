import numpy as np

class DistributeExercisesMetaheuristic:
    def __init__(
            self,
            exercises: np.ndarray,
            days: int,
            rng: None | np.random.Generator = None
        ) -> None:
        self.exercises = exercises
        self.days = days
        self.rng = rng if rng is not None else np.random.default_rng()


    def split_exercises(self, exercises: np.ndarray) -> list[np.ndarray]:
        self.rng.shuffle(exercises)
        return np.array_split(exercises, self.days)
