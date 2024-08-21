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
    
    def fit(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fit method. For this simple poisoner, it does nothing.

        Args:
            data (torch.Tensor): A batch of images.

        Returns:
            torch.Tensor: Unmodified data.
        """
        return data
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies a patch to each image in the batch.

        Args:
            data (torch.Tensor): A batch of images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The batch of images with the patch applied.
        """
        x_start, y_start = self.patch_location
        width, height = self.patch_dimension

        # Apply the patch to each image in the batch
        data[:, :, y_start:y_start + height, x_start:x_start + width] = self.patch_value
        
        return data

# Example usage
if __name__ == "__main__":
    # Create a simple batch of images (batch_size, channels, height, width)
    batch_size, channels, height, width = 8, 3, 32, 32
    images = (torch.randn(batch_size, channels, height, width) + 4) / 8
    
    # Initialize the poisoner
    patch_location = (10, 10)
    patch_dimension = (5, 5)
    patch_value = 1.0  # Set the patch to a bright value
    
    poisoner = SimplePatchPoisoner(patch_location, patch_dimension, patch_value)
    
    # Apply the patch to the batch of images
    poisoned_images = poisoner.transform(images)
    
    img = plt.imshow(poisoned_images[0].permute(1, 2, 0).numpy())
    # save image to disk
    plt.savefig('poisoned_image.png')