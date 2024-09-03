from matplotlib import pyplot as plt
import numpy as np
from poisoning_transforms.data.datapoisoner import DataPoisoner
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner

class MultiPatchPoisoner(DataPoisoner):
    def __init__(self, num_patches: int, patch_dimension: tuple, value_range: tuple, image_size: tuple, seed: int = 0):
        """
        Initializes the MultiPatchPoisoner with a list of SimplePatchPoisoner instances.

        Args:
            num_patches (int): The number of patches to create.
            patch_dimension (tuple): The (width, height) of the patches.
            value_range (tuple): The range (min_value, max_value) for patch values.
            image_size (tuple): The (height, width) of the images.
            seed (int): The seed for random number generation.
        """
        super().__init__()
        self.patches = []
        self.patch_dimension = patch_dimension
        self.image_height, self.image_width = image_size
        self.value_range = value_range
        self.seed = seed
        
        self._generate_patches(num_patches)
    
    def _generate_patches(self, num_patches: int):
        """
        Generates a list of SimplePatchPoisoner instances with random locations and values.
        
        Args:
            num_patches (int): The number of patches to create.
        """
        np.random.seed(self.seed)  # Use the seed for reproducibility
        for _ in range(num_patches):
            x_start = np.random.randint(0, self.image_width - self.patch_dimension[0] + 1)
            y_start = np.random.randint(0, self.image_height - self.patch_dimension[1] + 1)
            patch_value = np.random.uniform(self.value_range[0], self.value_range[1])
            
            patch_poisoner = SimplePatchPoisoner(
                patch_location=(x_start, y_start),
                patch_dimension=self.patch_dimension,
                patch_value=patch_value
            )
            self.patches.append(patch_poisoner)
    
    def sample_patch(self, unique_id: int) -> SimplePatchPoisoner:
        """
        Samples a patch based on a unique numeric identifier.

        Args:
            unique_id (int): A unique numeric identifier for sampling a patch.

        Returns:
            SimplePatchPoisoner: The sampled patch.
        """
        num_patches = len(self.patches)
        sampled_index = (unique_id % num_patches + num_patches) % num_patches
        return self.patches[sampled_index]
        
    def transform(self, data: dict) -> dict:
        """
        Applies a specific patch to each image in the batch based on a unique numeric identifier.

        Args:
            data (dict): A dictionary with "image" and "label". 
                         "image" contains a batch of images with shape (batch_size, channels, height, width).
            unique_id (int): A unique numeric identifier to choose a patch.

        Returns:
            dict: The dictionary with the batch of images with the selected patch applied.
        """
        images = data['image']
        
        for patch in self.patches:
            x_start, y_start = patch['location']
            width, height = patch['dimension']
            patch_value = patch['value']
            
            # Apply the patch to each image in the batch
            images[:, :, y_start:y_start + height, x_start:x_start + width] = patch_value
        
        # Update the data dictionary with the poisoned images
        data['image'] = images
        return data