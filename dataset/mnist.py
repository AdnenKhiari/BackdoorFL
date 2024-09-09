from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
from hydra.utils import instantiate, call
import  torch

from dataset.dataset import Dataset
from torchvision import transforms

class Mnist(Dataset):
    def __init__(self,partitioner):
        super(Mnist,self).__init__(partitioner)
        self.resizer = transforms.Resize((32,32))

    def get_federated_dataset(self,partitioners):
        return FederatedDataset(dataset="mnist",partitioners=partitioners)
    def collate(self):
        def col(batch):
            images = []
            labels = []
            for item in batch:
                images.append(torch.cat([item["image"],item["image"],item["image"]]))
                labels.append(torch.tensor(item["label"]))
            return {
                "image": self.resizer(torch.stack(images)),
                "label": torch.stack(labels)
            }
        return col