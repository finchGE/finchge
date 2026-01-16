import re
from abc import ABC, abstractmethod


class RepairStrategy(ABC):
    """Abstract base class for phenotype repair strategies."""

    @abstractmethod
    def repair(self, phenotype):
        """
        Repair the given phenotype.

        Args:
            phenotype (str): The phenotype string to repair.

        Returns:
            str: The repaired phenotype string.
        """
        pass

