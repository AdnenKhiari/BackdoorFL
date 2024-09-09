from matplotlib import pyplot as plt
import torch
import torch.utils
from poisoning_transforms.data.datapoisoner import DataPoisoner

class SimplePatchPoisoner(DataPoisoner):
    def __init__(self, patch_location: tuple, patch_dimension: tuple, patch_value: float):
        """
        Initializes the SimplePatchPoisoner with the location, dimension, and value of the patch.

        Args:
            patch_location (tuple): The (x, y) coordinates of the top-left corner of the patch.
            patch_dimension (tuple): The (width, height) of the patch.
            patch_value (float): The value to fill the patch with.
        """
        super().__init__()
        self.patch_location = patch_location
        self.patch_dimension = patch_dimension
        self.patch_value = patch_value
        
    def train(self) -> None:
        pass
    
    def transform(self, data: dict) -> dict:
        """
        Applies a patch to each image in the batch.

        Args:
            data (dict): A dictionary with "image" and "label". 
                         "image" contains a batch of images with shape (batch_size, channels, height, width).

        Returns:
            dict: The dictionary with the batch of images with the patch applied.
        """
        images = data['image']
        print("PPATCH",images.shape)
        x_start, y_start = self.patch_location
        width, height = self.patch_dimension

        # Apply the patch to each image in the batch
        images[:, :, y_start:y_start + height, x_start:x_start + width] = self.patch_value
        
        # Update the data dictionary with the poisoned images
        data['image'] = images
        return data