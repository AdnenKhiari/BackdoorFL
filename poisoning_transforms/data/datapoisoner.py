import torch
from torch.utils.data import DataLoader
from typing import Dict, Iterator, List, Optional, Any, Union
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
            

class DataPoisoningPipeline(DataPoisoner):
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
   
class BatchPoisoner(DataPoisoner):
    def __init__(self, poisoner: DataPoisoner, k: int, label_replacement: Union[int, None] = None):
        """
        Initializes the BatchPoisoner with a specific poisoner and a batch size.

        Args:
            poisoner (DataPoisoner): The poisoner to apply to selected items.
            k (int): The number of items to select and poison from each batch. 
                     If k = -1, applies the poisoner to the entire batch.
            label_replacement (Union[int, None]): Optional label to replace the original labels for poisoned items.
        """
        super().__init__()
        self.poisoner = poisoner
        self.k = k
        self.label_replacement = label_replacement

    def fit(self, data: Dict[str, torch.Tensor]) -> None:
        """
        Fits the poisoner to the first K items in the data, or to the entire batch if K is -1.

        Args:
            data (Dict[str, torch.Tensor]): Dictionary containing 'label' and 'image' tensors.
        """
        # Extract images and determine the subset to fit on
        images = data['image']
        if self.k == -1:
            k_images = images
        else:
            k = min(self.k, images.size(0))
            k_images = images[:k]
        
        # Fit the poisoner to the selected items
        self.poisoner.fit({'image': k_images, 'label': data['label'][:k]})

    def transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies the poisoner to the first K items in the batch, or to the entire batch if K is -1.
        Also replaces the labels for poisoned items if a label replacement is specified.

        Args:
            data (Dict[str, torch.Tensor]): Dictionary containing 'label' and 'image' tensors.

        Returns:
            Dict[str, torch.Tensor]: Updated dictionary with poisoned items and optionally updated labels.
        """
        # Extract labels and images
        labels = data['label']
        images = data['image']
        
        if self.k == -1:
            # Apply poisoner to the entire batch
            poisoned_batch = self.poisoner.transform(data)
            poisoned_images = poisoned_batch['image']
            poisoned_labels = poisoned_batch['label']
            if self.label_replacement is not None:
                poisoned_labels[:] = self.label_replacement
        else:
            # Ensure K is less than the batch size
            k = min(self.k, images.size(0))
            poisoned_images = images.clone()
            poisoned_images[:k] = self.poisoner.transform({'image': images[:k], 'label': labels[:k]})['image']
            
            # Replace labels for the first K items if needed
            poisoned_labels = labels.clone()
            if self.label_replacement is not None:
                poisoned_labels[:k] = self.label_replacement
            
        # Return the updated data
        return {'label': poisoned_labels, 'image': poisoned_images}