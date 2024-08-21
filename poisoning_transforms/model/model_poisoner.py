from abc import ABC, abstractmethod
from typing import List, Any

import torch

class ModelPoisoner(ABC):
    """
    Base class for model poisoning that can fit and transform lists of weights.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, weights: List[torch.Tensor]) -> None:
        """
        Fits the poisoner to the weights.

        Args:
            weights (List[torch.Tensor]): List of weight tensors to fit the poisoner.
        """
        pass

    @abstractmethod
    def transform(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Transforms the weights by injecting poison.

        Args:
            weights (List[torch.Tensor]): List of weight tensors to be transformed.

        Returns:
            List[torch.Tensor]: List of poisoned weight tensors.
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

    def fit(self, weights: List[torch.Tensor]) -> None:
        """
        Fits each ModelPoisoner in the pipeline to the list of weights in sequence.

        Args:
            weights (List[torch.Tensor]): List of weight tensors to fit each poisoner.
        """
        for poisoner in self.poisoners:
            poisoner.fit(weights)

    def transform(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies each ModelPoisoner in the pipeline to the list of weights in sequence.

        Args:
            weights (List[torch.Tensor]): List of weight tensors to be transformed.

        Returns:
            List[torch.Tensor]: List of poisoned weight tensors.
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

    def transform(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Transforms the weights by injecting poison.

        Args:
            weights (List[torch.Tensor]): List of weight tensors to be transformed.

        Returns:
            List[torch.Tensor]: List of poisoned weight tensors.
        """
        return weights