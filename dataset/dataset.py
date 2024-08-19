from abc import abstractmethod
from flwr_datasets import FederatedDataset
from hydra.utils import instantiate, call
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import  DataLoader
from flwr_datasets.partitioner import Partitioner
class Dataset():
    def __init__(self,partitioner):
        self.partitioner : Partitioner = partitioner
        
    def apply_transforms(self):
        def tr(batch):
            pytorch_transforms = Compose(
                [ToTensor()]
            )
            batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
            return batch
        return tr
    
    def collate(self):
        def col(batch):
            sizes = []
            try:
                images = []
                labels = []
                for item in batch:
                    images.append(item["image"])
                    labels.append(torch.tensor(item["label"]))
                    sizes.append({item["image"].shape})
                return {
                    "image": torch.stack(images),
                    "label": torch.stack(labels)
                }
            except Exception as e:
                print(f"Error in collate: {e}")
                print(f"Sizes: {list(frozenset(sizes))}")
                raise e
        return col    
    @abstractmethod
    def get_federated_dataset(self) -> FederatedDataset:
        raise NotImplementedError("Implement it !")
        
    def get_test_set(self,batch_size: int = 16):
        fds = self.get_federated_dataset({"test": 1})
        data = fds.load_split("test")

        data_with_transforms = data.with_transform(self.apply_transforms())
        return DataLoader(data_with_transforms,batch_size=batch_size,collate_fn=self.collate())

    def load_datasets(self,partition_id: int, batch_size: int = 16, val_ratio: float = 0.15,seed=42):
        fds = self.get_federated_dataset({"train": self.partitioner})
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=val_ratio, seed=seed)

        partition_train_test = partition_train_test.with_transform(self.apply_transforms())
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True,collate_fn=self.collate(),drop_last=True)
        valloader = DataLoader(partition_train_test["test"], batch_size=batch_size,collate_fn=self.collate())
        testset = fds.load_split("test").with_transform(self.apply_transforms())
        testloader = DataLoader(testset, batch_size=batch_size,collate_fn=self.collate())
        return trainloader, valloader, testloader
