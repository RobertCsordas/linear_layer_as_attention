import torch
import dataset
from ... import task
from typing import List
from .mnist_fmnist_ff_attention import MnistFmnistFFAttention


@task()
class Cifar10FFAttention(MnistFmnistFFAttention):
    def get_raw_datasets(self) -> List[torch.utils.data.Dataset]:
        return [dataset.image.CIFAR10("train")]

    def init_valid_sets(self):
        self.valid_sets["test"] = dataset.image.CIFAR10("test")
