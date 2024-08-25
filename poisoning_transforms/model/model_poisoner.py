from abc import ABC, abstractmethod
import numpy as np

class ModelPoisoner(ABC):
    """
    Base class for model poisoning that can fit and transform a NumPy array of weights.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, weights: np.ndarray) -> None:
        """
        Fits the poisoner to the weights.

        Args:
            weights (np.ndarray): Weight array to fit the poisoner.
        """
        pass

    @abstractmethod
    def transform(self, weights: np.ndarray) -> np.ndarray:
        """
        Transforms the weights by injecting poison.

        Args:
            weights (np.ndarray): Weight array to be transformed.

        Returns:
            np.ndarray: Poisoned weight array.
        """
        pass

class ModelPoisoningPipeline(ModelPoisoner):
    def __init__(self, poisoners: list[ModelPoisoner]):
        """
        Initializes the ModelPoisoningPipeline with a list of ModelPoisoners.

        Args:
            poisoners (list[ModelPoisoner]): A list of ModelPoisoner instances to apply sequentially.
        """
        super().__init__()
        self.poisoners = poisoners

    def fit(self, weights: np.ndarray) -> None:
        """
        Fits each ModelPoisoner in the pipeline to the weights in sequence.

        Args:
            weights (np.ndarray): Weight array to fit each poisoner.
        """
        for poisoner in self.poisoners:
            poisoner.fit(weights)

    def transform(self, weights: np.ndarray) -> np.ndarray:
        """
        Applies each ModelPoisoner in the pipeline to the weights in sequence.

        Args:
            weights (np.ndarray): Weight array to be transformed.

        Returns:
            np.ndarray: Poisoned weight array.
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

    def fit(self, weights: np.ndarray) -> None:
        """
        This method is intentionally left blank as no fitting is required.
        """
        pass

    def transform(self, weights: np.ndarray) -> np.ndarray:
        """
        Returns the weights unmodified.

        Args:
            weights (np.ndarray): Weight array to be transformed.

        Returns:
            np.ndarray: Unmodified weight array.
        """
        return weights
