import torch
import dataset
from ... import task
from typing import List
from .mnist_fmnist_ff_attention import MnistFmnistFFAttention


@task()
class MnistFFAttention(MnistFmnistFFAttention):
    def get_raw_datasets(self) -> List[torch.utils.data.Dataset]:
        return [dataset.image.PermutedMNIST(0, "train")]

    def init_valid_sets(self):
        self.valid_sets["valid_mnist"] = dataset.image.PermutedMNIST(0, "valid")
        self.valid_sets["test_mnist"] = dataset.image.PermutedMNIST(0, "test")
        self.valid_sets["test_fmnist"] = dataset.image.FashionMNIST(0, "test")
