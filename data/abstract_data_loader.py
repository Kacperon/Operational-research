import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from planner.exercise import Exercise

class AbstractDataLoader(ABC):
    @abstractmethod
    def raw(self) -> pd.DataFrame:
        """
        Returns the raw data loaded from the dataset, which may include exercise details and metadata.
        """
        pass

    @abstractmethod
    def exercises(self) -> np.ndarray[Exercise]:
        """
        Returns an array of Exercise objects loaded from the dataset.
        """
        pass

    @abstractmethod
    def muscle_group_count(self) -> int:
        """
        Returns the total number of unique muscle groups across all exercises.
        """
        pass
