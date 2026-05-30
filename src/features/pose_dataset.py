import torch
from typing import Any
from datasets import Dataset as HFDataset
from torchvision.transforms.v2 import Compose
from torch.utils.data import Dataset as TorchDataset


class PoseDataset(TorchDataset):
    def __init__(
        self,
        dataset: HFDataset,
        transforms: Compose,
    ) -> None:
        self.dataset = dataset
        self.transforms = transforms
        self.num_videos = len(dataset)

    def __getitem__(self, index) -> Any:
        sample = self.dataset[index]
        data = self.transforms(sample["pose"])
        label = torch.Tensor([sample["gloss_id"]])
        return {"pose": data, "label": label}

    def __len__(self) -> int:
        return len(self.dataset)
