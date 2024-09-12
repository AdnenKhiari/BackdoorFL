import torch
import numpy as np
import cv2
from poisoning_transforms.data.datapoisoner import DataPoisoner

class FrequencyDomainPoisoner(DataPoisoner):
    def __init__(self, window_size: int, channel_list: list, pos_list: list, magnitude: float):
        """
        Initializes the FrequencyDomainPoisoner with the required parameters for poisoning in the frequency domain.

        Args:
            window_size (int): Size of the DCT window.
            channel_list (list): List of channels to apply the poison to.
            pos_list (list): List of positions in the frequency domain to modify.
            magnitude (float): Magnitude of the frequency modification.
        """
        super().__init__()
        self.window_size = window_size
        self.channel_list = channel_list
        self.pos_list = pos_list
        self.magnitude = magnitude

    def RGB2YUV(self, x_rgb):
        # Conversion from RGB to YUV using OpenCV
        x_yuv = np.zeros(x_rgb.shape, dtype=np.float32)
        for i in range(x_rgb.shape[0]):
            img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            x_yuv[i] = img
        return x_yuv

    def YUV2RGB(self, x_yuv):
        # Conversion from YUV back to RGB
        x_rgb = np.zeros(x_yuv.shape, dtype=np.float32)
        for i in range(x_yuv.shape[0]):
            img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
            x_rgb[i] = img
        return x_rgb

    def DCT(self, x_train):
        # Apply DCT to the image batch in windows
        x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float32)
        x_train = np.transpose(x_train, (0, 3, 1, 2))  # Reorder dimensions for easier processing

        for i in range(x_train.shape[0]):
            for ch in range(x_train.shape[1]):
                for w in range(0, x_train.shape[2], self.window_size):
                    for h in range(0, x_train.shape[3], self.window_size):
                        sub_dct = cv2.dct(x_train[i][ch][w:w+self.window_size, h:h+self.window_size].astype(np.float32))
                        x_dct[i][ch][w:w+self.window_size, h:h+self.window_size] = sub_dct
        return x_dct

    def IDCT(self, x_train):
        # Apply inverse DCT to the image batch
        x_idct = np.zeros(x_train.shape, dtype=np.float32)

        for i in range(x_train.shape[0]):
            for ch in range(x_train.shape[1]):
                for w in range(0, x_train.shape[2], self.window_size):
                    for h in range(0, x_train.shape[3], self.window_size):
                        sub_idct = cv2.idct(x_train[i][ch][w:w+self.window_size, h:h+self.window_size].astype(np.float32))
                        x_idct[i][ch][w:w+self.window_size, h:h+self.window_size] = sub_idct
        x_idct = np.transpose(x_idct, (0, 2, 3, 1))
        return x_idct
    
    def train(self):
        pass

    def transform(self, data: dict) -> dict:
        """
        Applies poisoning in the frequency domain to each image in the batch.

        Args:
            data (dict): A dictionary with "image" and "label". 
                         "image" contains a batch of images with shape (batch_size, channels, height, width).

        Returns:
            dict: The dictionary with the batch of images with the poison applied.
        """
        device = data['image'][0].device if len(data['image']) > 0 else torch.device('cpu')
        images = data['image'].cpu().permute(0, 2, 3, 1).numpy()  # Convert from PyTorch tensor to NumPy array
        images *= 255.  # Scale to [0, 255]

        # Convert to YUV color space
        images = self.RGB2YUV(images)

        # Apply DCT (Discrete Cosine Transform)
        images = self.DCT(images)

        # Plug the trigger frequency
        for i in range(images.shape[0]):  # Iterate over batch size
            for ch in self.channel_list:  # Iterate over specified channels
                for w in range(0, images.shape[2], self.window_size):  # Width loop
                    for h in range(0, images.shape[3], self.window_size):  # Height loop
                        for pos in self.pos_list:  # Position loop in the frequency domain
                            images[i][ch][w + pos[0]][h + pos[1]] += self.magnitude

        # Apply IDCT (Inverse Discrete Cosine Transform)
        images = self.IDCT(images)

        # Convert back to RGB
        images = self.YUV2RGB(images)

        # Normalize back to [0, 1] range
        images /= 255.
        images = np.clip(images, 0, 1)

        # Convert back to PyTorch tensor
        data['image'] = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        return data
