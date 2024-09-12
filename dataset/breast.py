from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
import torch
from torchvision import transforms

from dataset.dataset import Dataset

class Breast(Dataset):
    def __init__(self,partitioner,seed):
        super(Breast,self).__init__(partitioner)
        self.resizer = transforms.Resize((64,64),antialias=True,antialias=True)
        self.seed = seed
        
    def get_federated_dataset(self,partitioners):
        return FederatedDataset(dataset="dbzadnen/breast-histopathology-images",partitioners=partitioners,seed=self.seed)
    def collate(self):
        def col(batch):
            images = []
            labels = []
            for item in batch:
                if item["image"].shape == (3,50,50) :
                    images.append(item["image"])
                    labels.append(torch.tensor(item["label"]))
            return {
                "image": self.resizer(torch.stack(images)),
                "label": torch.stack(labels)
            }

        return col  