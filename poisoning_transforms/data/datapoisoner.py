import torch
from torch.utils.data import DataLoader
from typing import Iterator, List, Optional, Any
from abc import ABC, abstractmethod

class DataPoisoner(ABC):
    """
    Base class for data poisoning that can fit and transform data.
    Can wrap around an iterator like a DataLoader to yield poisoned data batches.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: Any) -> None:
        """
        Fits the poisoner to the data.

        Args:
            data (Any): Data to fit the poisoner.
        """
        pass

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transforms the data by injecting poison.

        Args:
            data (Any): Data batch to be transformed.

        Returns:
            Any: Poisoned data batch.
        """
        pass

    def wrap_transform_iterator(self, dataloader: DataLoader) -> Iterator:
        """
        Wraps around a DataLoader (or any iterator) and applies the transform method to each batch.

        Args:
            dataloader (DataLoader): DataLoader to be wrapped and poisoned.

        Returns:
            Iterator: Iterator over poisoned batches.
        """
        for batch in dataloader:
            yield self.transform(batch)
            
    def wrap_fit_iterator(self, dataloader: DataLoader) -> Iterator:
        """
        Wraps around a DataLoader (or any iterator) and applies the fit method to each batch.

        Args:
            dataloader (DataLoader): DataLoader to be wrapped and poisoned.

        Returns:
            Iterator: Iterator over poisoned batches.
        """
        for batch in dataloader:
            yield self.fit(batch)
            

class PoisoningPipeline(DataPoisoner):
    def __init__(self, poisoners: List[DataPoisoner]):
        """
        Initializes the PoisoningPipeline with a list of DataPoisoners.

        Args:
            poisoners (List[DataPoisoner]): A list of DataPoisoner instances to apply sequentially.
        """
        super().__init__()
        self.poisoners = poisoners
    
    def fit(self, data: Any) -> Any:
        """
        Fits each DataPoisoner in the pipeline to the data in sequence.

        Args:
            data (Any): Data to fit each poisoner.

        Returns:
            Any: The data after fitting all poisoners.
        """
        for poisoner in self.poisoners:
            data = poisoner.fit(data)
        return data
    
    def transform(self, data: Any) -> Any:
        """
        Applies each DataPoisoner in the pipeline to the data in sequence.

        Args:
            data (Any): Data to be transformed.

        Returns:
            Any: The data after applying all poisoners.
        """
        for poisoner in self.poisoners:
            data = poisoner.transform(data)
        return data
    
class IdentityDataPoisoner(DataPoisoner):
    def __init__(self):
        """
        Initializes an IdentityDataPoisoner that does not modify the data.
        """
        super().__init__()
        
    def transform(self, data: Any) -> Any:
        """
        Transforms the data by injecting poison.

        Args:
            data (Any): Data batch to be transformed.

        Returns:
            Any: Poisoned data batch.
        """
        return data