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
    
    
    def load_datasets(self,partition_id: int, batch_size: int = 16, val_ratio: float = 0.15,seed=42):
        fds = self.get_federated_dataset({"train": self.partitioner})
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=val_ratio, seed=seed)

        partition_train_test = partition_train_test.with_transform(self.apply_transforms())
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True,collate_fn=self.collate(),drop_last=True)
        valloader = DataLoader(partition_train_test["validation"], batch_size=batch_size,collate_fn=self.collate())
        testset = fds.load_split("test").with_transform(self.apply_transforms())
        testloader = DataLoader(testset, batch_size=batch_size,collate_fn=self.collate())
        return trainloader, valloader, testloader
