from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from poisoning_transforms.data.datapoisoner import DataPoisoner

class RandomizedSimplePatchPoisoner(DataPoisoner):
    def __init__(self, patch_location_range: tuple, patch_size_range: tuple, patch_value: float, seed: int = None):
        """
        Initializes the RandomizedSimplePatchPoisoner with ranges for the location and size of the patch, the patch value, and a seed.

        Args:
            patch_location_range (tuple): ((x_min, x_max), (y_min, y_max)) for the range of patch locations.
            patch_size_range (tuple): ((width_min, width_max), (height_min, height_max)) for the range of patch sizes.
            patch_value (float): The value to fill the patch with.
            seed (int, optional): Seed for random number generation. Defaults to None.
        """
        super().__init__()
        self.patch_location_range = patch_location_range
        self.patch_size_range = patch_size_range
        self.patch_value = patch_value
        self.seed = seed
        
        # if self.seed is not None:
        #     random.seed(self.seed)
        #     np.random.seed(self.seed)
        #     torch.manual_seed(self.seed)

    def _randomize_patch(self, image_height, image_width):
        """ Randomly determine patch location and size within the given ranges. """
        x_min, x_max = self.patch_location_range[0]
        y_min, y_max = self.patch_location_range[1]
        width_min, width_max = self.patch_size_range[0]
        height_min, height_max = self.patch_size_range[1]
        
        # Randomly choose the patch size
        patch_width = random.randint(width_min, width_max)
        patch_height = random.randint(height_min, height_max)
        
        # Randomly choose the patch location
        x_start = random.randint(x_min, min(x_max, image_width - patch_width))
        y_start = random.randint(y_min, min(y_max, image_height - patch_height))
        
        return x_start, y_start, patch_width, patch_height
    
    def train(self) -> None:
        pass

    def transform(self, data: dict) -> dict:
        """
        Applies a randomly-sized and randomly-located patch to each image in the batch.

        Args:
            data (dict): A dictionary with "image" and "label". 
                         "image" contains a batch of images with shape (batch_size, channels, height, width).

        Returns:
            dict: The dictionary with the batch of images with the patch applied.
        """
        images = data['image']
        batch_size, channels, image_height, image_width = images.shape
        
        for i in range(batch_size):
            x_start, y_start, patch_width, patch_height = self._randomize_patch(image_height, image_width)
            # Apply the patch to the current image
            images[i, :, y_start:y_start + patch_height, x_start:x_start + patch_width] = self.patch_value

        # Update the data dictionary with the poisoned images
        data['image'] = images
        return data
