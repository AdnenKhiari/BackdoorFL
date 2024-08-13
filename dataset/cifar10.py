from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
import  torch

from dataset.dataset import Dataset

class Cifar10(Dataset):
    def __init__(self,partitioner):
        super(Cifar10,self).__init__(partitioner)
        
    def get_federated_dataset(self,partitioners):
        return FederatedDataset(dataset="cifar10",partitioners=partitioners)
    
    def apply_transforms(self):
        def tr(batch):
            pytorch_transforms = Compose(
                [ToTensor()]
            )
            batch["image"] = [pytorch_transforms(image) for image in batch["img"]]
            del batch["img"]
            return batch
        return tr
    def collate(self):
        def col(batch):
            images = []
            labels = []
            for item in batch:
                images.append(item["image"])
                labels.append(torch.tensor(item["label"]))
            return {
                "image": torch.stack(images),
                "label": torch.stack(labels)
            }
        return col