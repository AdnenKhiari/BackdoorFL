from torch.utils.data import  DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
import  torch

from dataset.dataset import Dataset

class Breast(Dataset):
    def __init__(self,partitioner):
        super(Breast,self).__init__(partitioner)
        
    def get_federated_dataset(self,partitioners):
        fd= FederatedDataset(dataset="EulerianKnight/breast-histopathology-images-train-test-valid-split",partitioners=partitioners)
        print("SP",fd._check_if_split_present("train"))
        print("SP",fd._check_if_split_present("test"))
    def apply_transforms(self):
        def tr(batch):
            print("BB",batch)
            pytorch_transforms = Compose(
                [ToTensor()]
            )
            batch["image"] = [pytorch_transforms(image) for image in batch["img"]]
            del batch["img"]
            return batch
        return tr
    def collate(self):
        def col(batch):
            print("BB",batch)

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