from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset


def apply_transforms(batch):
    pytorch_transforms = Compose(
        # [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        [ToTensor()]
    )
    batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
    return batch
    
def get_test_set(batch_size: int = 16):
     
    fds = FederatedDataset(dataset="mnist",partitioners={"test": 1})
    data = fds.load_split("test")

    data_with_transforms = data.with_transform(apply_transforms)
    return DataLoader(data_with_transforms,batch_size=batch_size)

def load_datasets(partition_id: int, num_partitions: int, batch_size: int = 16, val_ratio: float = 0.15,seed=42):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=val_ratio, seed=seed)

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=32)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, valloader, testloader
