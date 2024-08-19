from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
import  torch

from dataset.dataset import Dataset

class Breast(Dataset):
    def __init__(self,partitioner):
        super(Breast,self).__init__(partitioner)
        
    def get_federated_dataset(self,partitioners):
        return FederatedDataset(dataset="dbzadnen/breast-histopathology-images",partitioners=partitioners)