from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score
from finchge.model.model import Model

class GEFitnessFunction(ABC):
    """
    Base class for a fitness function used in genetic algorithms.

    This abstract class defines the interface for evaluating the fitness
    of a given phenotype. Subclasses must implement the `evaluate` method.

    Args:
        maximize (bool): Indicates whether the goal is to maximize (True)
                         or minimize (False) the fitness score.
    """

    def __init__(self, maximize: bool = False):
        self.maximize = maximize
        self.default_fitness = np.nan  # Use if evaluation fails or is not computable

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> float:
        """
        Evaluate the fitness score using the provided context.

        Args:
            context (dict): Dictionary containing evaluation inputs,
                            such as 'y_pred', 'y_val', 'x_val', etc.

        Returns:
            float: The computed fitness score.
        """
        pass



class FitnessEvaluator:
    """
    Fitness evaluator is used to evaluate the fitness of the individuals.
    It is a wrapper for fitness function to allow flexible and convenient fitness evaluation especially for use cases where data and models are involved.
    For example in hyperparameter optimization or neural architecture search.
    The 'training_required' parameter is used to speficy whether the model training is required.
    The phenotype is converted to model using the 'phenotype_to_model' parameter.

    Supports both model-based evaluation (e.g., using train/validation data)
    and phenotype-only evaluation (e.g., string matching or symbolic tasks).

    Args:
        data_train (optional): pytorch Dataset or Tuple of X_train, y_train. Required if `training_required=True`.
        data_test (optional): pytorch Dataset or Tuple of X_test, y_test,  Validation feature data. Required if `training_required=True`.
        model_parser (optional): ModelPaser that converts a phenotype to a model instance. Required if `training_required=True`.
        fitness_functions (GEFitnessFunction or list): One or more fitness function instances.
        training_required (bool): Whether evaluation requires model training. Default is False.
    """

    def __init__(
        self,
        fitness_functions,
        data_train = None,
        data_test = None,
        model_parser = None,
        training_required=False,
    ):
        self.training_required = training_required

        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]
        self.fitness_functions = fitness_functions


        if self.training_required:
            # Data can be either pytorch Dataset instance or dataframes: eg data_train = (xtrain, y_train), or data_train = training_dataset

            self._data_types = self._infer_data_types(data_train)

            # Lazy imports - only when actually needed
            if self._data_types.get('X_type') == 'torch':
                try:
                    import torch
                    self.train_dataset = data_train
                    self.x_train = None
                    self.y_train = None
                    self.test_dataset = data_test
                    self.x_test = None
                    self.y_test = None

                except ImportError:
                    raise ImportError(
                        "PyTorch is required for evaluating PyTorch models. "
                        "Install with: pip install torch"
                    )
            elif self._data_types.get('X_type') == 'pandas':
                self.x_train, self.y_train = data_train
                self.train_dataset = None
                self.x_test, self.y_test = data_test
                self.test_dataset = None

            else:
                raise TypeError(
                    "data_train must be either a PyTorch Dataset or a tuple of (x_train, y_train) DataFrames"
                )

            if model_parser is None:
                raise ValueError(
                    "Model-based evaluation requires model_parser. Or set use_model to False if fitness does not depend on data.")


            self.model_parser = model_parser

    def _infer_data_types(self, data_train):
        """Infer data types from training data, handling both Dataset and tuple formats."""
        if data_train is None:
            return {'format': 'none', 'X_type': None}

        data_info = {}
        # id pytorch dataset
        if (hasattr(data_train, '__getitem__') and
                hasattr(data_train, '__len__') and
                hasattr(data_train, '__class__') and
                hasattr(data_train.__class__, '__name__') and
                ('Dataset' in data_train.__class__.__name__ or
                 'DataLoader' in data_train.__class__.__name__)):
            data_info['format'] = 'dataset'
            data_info['X_type'] = 'torch'

        if isinstance(data_train, tuple) and len(data_train) == 2:
            data_info['format'] = 'tuple'
            X_train, y_train = data_train

            # Infer the type of X_train
            if hasattr(X_train, 'shape') and hasattr(X_train, 'columns'):
                data_info['X_type'] = 'pandas'
            elif hasattr(X_train, 'shape') and hasattr(X_train, 'dtype'):
                data_info['X_type'] = 'numpy'
            else:
                data_info['X_type'] = 'unknown'

        else:
            raise ValueError(
                f"Unsupported data_train format. Expected Dataset or (X, y) tuple, got {type(data_train)}"
            )

        return data_info

    def is_multi_objective(self) -> bool:
        """
        Determines if the evaluation is multi-objective.

        Returns:
            bool: True if more than one fitness function is provided, False otherwise.
        """
        return len(self.fitness_functions) > 1

    def get_objective_names(self) -> list:
        """
        Retrieves human-readable names of the objectives.

        Returns:
            list: List of objective names derived from fitness function class names.
        """
        return [type(fitness).__name__.replace('Fitness', '') for fitness in self.fitness_functions]

    def get_fitness_functions(self) -> list:
        """
        Returns the list of fitness function instances.

        Returns:
            list: List of fitness function objects.
        """
        return self.fitness_functions

    def count_objectives(self) -> int:
        """
        Counts the number of objectives.

        Returns:
            int: Number of fitness functions (objectives).
        """
        return len(self.fitness_functions)

    def get_maximize_flags(self):
        """
        Returns the maximize flags for each objective
        :return:
        """
        return [fn.maximize for fn in self.fitness_functions]

    def evaluate(self, phenotype) -> list:
        """
        Evaluates a given phenotype by training the corresponding model and applying fitness functions.

        Args:
            phenotype: An individual configuration or representation used to generate a model.

        Returns:
            list: A list of fitness scores corresponding to each fitness function.
        """

        context = {
            'phenotype': phenotype
        }

        if self.training_required:
            if self._data_types.get('X_type') == 'torch':
                net_ = self.model_parser.parse(phenotype)
                model = Model(net=net_)
                model.fit(train_dataset=self.train_dataset, val_dataset=self.test_dataset)
                y_pred, y_true = model.predict(self.test_dataset)
                # Maybe later we may need to provide more context, for now just use predictions and labels
                context.update({
                    'y_pred': y_pred,
                    'y_test': y_true
                })
            else:
                model = self.model_parser.parse(phenotype)
                model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_test)
                context.update({
                    'model': model,
                    'y_pred': y_pred,
                    'x_train': self.x_train,
                    'y_train': self.y_train,
                    'x_test': self.x_test,
                    'y_test': self.y_test
                })

        scores = [func.evaluate(context) for func in self.fitness_functions]
        return scores



# Example Fitness functions


class AccuracyFitness(GEFitnessFunction):
    """
    Fitness function that evaluates model accuracy on a validation set.

    This metric computes the classification accuracy between the predicted labels
    and the ground-truth labels from the validation data. It is intended to be
    used in supervised classification tasks, and is a maximization objective.

    Inherits from:
        GEFitnessFunction (maximize=True)

    Methods:
        evaluate(context): Computes accuracy using 'y_pred' and 'y_val' from the context.
    """

    def __init__(self):
        super().__init__(maximize=True)

    def evaluate(self, context):
        """
        Evaluates the accuracy of the model's predictions.

        Args:
            context (dict): A dictionary containing evaluation context. Must include:
                - 'y_pred': Predicted labels from the model (array-like).
                - 'y_test': True labels for the test set (array-like).

        Returns:
            accuracy_score (float): Accuracy score between 0.0 and 1.0.
        """
        y_test = context['y_test']
        y_pred = context['y_pred']
        return accuracy_score(y_test, y_pred)

