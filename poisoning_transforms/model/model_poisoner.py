from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

class ModelPoisoner(ABC):
    """
    Base class for model poisoning that can fit and transform a list of NumPy arrays of weights.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, weights: List[np.ndarray]) -> None:
        """
        Fits the poisoner to the weights.

        Args:
            weights (List[np.ndarray]): List of weight arrays to fit the poisoner.
        """
        pass

    @abstractmethod
    def transform(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Transforms the weights by injecting poison.

        Args:
            weights (List[np.ndarray]): List of weight arrays to be transformed.

        Returns:
            List[np.ndarray]: List of poisoned weight arrays.
        """
        pass

class ModelPoisoningPipeline(ModelPoisoner):
    def __init__(self, poisoners: List[ModelPoisoner]):
        """
        Initializes the ModelPoisoningPipeline with a list of ModelPoisoners.

        Args:
            poisoners (List[ModelPoisoner]): A list of ModelPoisoner instances to apply sequentially.
        """
        super().__init__()
        self.poisoners = poisoners

    def fit(self, weights: List[np.ndarray]) -> None:
        """
        Fits each ModelPoisoner in the pipeline to the weights in sequence.

        Args:
            weights (List[np.ndarray]): List of weight arrays to fit each poisoner.
        """
        for poisoner in self.poisoners:
            poisoner.fit(weights)

    def transform(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies each ModelPoisoner in the pipeline to the weights in sequence.

        Args:
            weights (List[np.ndarray]): List of weight arrays to be transformed.

        Returns:
            List[np.ndarray]: List of poisoned weight arrays.
        """
        for poisoner in self.poisoners:
            weights = poisoner.transform(weights)
        return weights

class IdentityModelPoisoner(ModelPoisoner):
    def __init__(self):
        """
        Initializes an IdentityModelPoisoner that does not modify the weights.
        """
        super().__init__()

    def fit(self, weights: List[np.ndarray]) -> None:
        """
        This method is intentionally left blank as no fitting is required.
        """
        pass

    def transform(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Returns the weights unmodified.

        Args:
            weights (List[np.ndarray]): List of weight arrays to be transformed.

        Returns:
            List[np.ndarray]: List of unmodified weight arrays.
        """
        return weights
