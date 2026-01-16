from abc import ABC, abstractmethod

class BaseModelParser(ABC):
    """
    Base class for ModelParser.

    This abstract class defines the interface for model parser to be used in tasks involving ML models
    Subclasses must implement the `parse` method.

    Args:
        maximize (bool): Indicates whether the goal is to maximize (True)
                         or minimize (False) the fitness score.
    """

    def __init__(self):
        pass

    @abstractmethod
    def parse(self, phenotype):
        """
        Evaluate the fitness score using the provided context.

        Args:
            context (dict): Dictionary containing evaluation inputs,
                            such as 'y_pred', 'y_val', 'x_val', etc.

        Returns:
            float: The computed fitness score.
        """
        pass