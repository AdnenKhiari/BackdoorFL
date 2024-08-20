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
    def collate(self):
        def col(batch):
            images = []
            labels = []
            for item in batch:
                if item["image"].shape == (3,50,50) :
                    print(item["image"].max())
                    print(item["image"].min())
                    images.append(item["image"])
                    labels.append(torch.tensor(item["label"]))
            return {
                "image": torch.stack(images),
                "label": torch.stack(labels)
            }

        return col  