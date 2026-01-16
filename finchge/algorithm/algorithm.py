from abc import ABC, abstractmethod
from typing import List


class BaseAlgorithm(ABC):
    """Abstract base class for Search Algorithm"""

    @abstractmethod
    def sort_population(self, individuals) -> None:
        pass

    @abstractmethod
    def get_best(self, individuals, fitness_functions) -> List:
        pass

    @abstractmethod
    def evolve_one_generation(self, population) -> 'Population':
        pass
