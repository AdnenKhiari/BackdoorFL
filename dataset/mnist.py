from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
from hydra.utils import instantiate, call
import  torch

from dataset.dataset import Dataset

class Mnist(Dataset):
    def __init__(self,partitioner):
        super(Mnist,self).__init__(partitioner)
    def get_federated_dataset(self,partitioners):
        return FederatedDataset(dataset="mnist",partitioners=partitioners)