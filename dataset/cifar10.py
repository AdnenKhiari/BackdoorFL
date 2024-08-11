from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
import  torch

class Cifar10():
    def __init__(self):
        pass
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
        
    def get_test_set(self,batch_size: int = 16):
        
        fds = FederatedDataset(dataset="cifar10",partitioners={"test": 1})
        data = fds.load_split("test")

        data_with_transforms = data.with_transform(self.apply_transforms())
        return DataLoader(data_with_transforms,batch_size=batch_size,collate_fn=self.collate())

    def load_datasets(self,partition_id: int, num_partitions: int, batch_size: int = 16, val_ratio: float = 0.15,seed=42):
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=val_ratio, seed=seed)

        partition_train_test = partition_train_test.with_transform(self.apply_transforms())
        trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True,collate_fn=self.collate(),drop_last=True)
        valloader = DataLoader(partition_train_test["test"], batch_size=32,collate_fn=self.collate())
        testset = fds.load_split("test").with_transform(self.apply_transforms())
        testloader = DataLoader(testset, batch_size=batch_size,collate_fn=self.collate())
        return trainloader, valloader, testloader
