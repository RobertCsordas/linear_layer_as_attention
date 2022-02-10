import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Any
from .model_interface import ModelInterface
from .result import RecurrentResult
from framework.utils import U
import random


class LanguageModelInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, batch_dim: int = 1, drop_state_prob: float = 0):
        super().__init__()
        self.model = model
        self.state = None
        self.batch_dim = batch_dim
        self.drop_state_prob = drop_state_prob
        self.time_dim = 1 - self.batch_dim

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["data"].narrow(self.time_dim, 0, data["data"].shape[self.time_dim] - 1)

    def decode_outputs(self, outputs: RecurrentResult) -> Any:
        return outputs.outputs

    def reset_state(self):
        self.state = None

    def loss(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = data["data"].narrow(self.time_dim, 1, data["data"].shape[self.time_dim] - 1).contiguous()
        assert net_out.shape[:-1] == target.shape
        return F.cross_entropy(net_out.flatten(0, -2), target.flatten().long())

    def __call__(self, data: Dict[str, torch.Tensor]) -> RecurrentResult:
        if self.model.training and self.drop_state_prob > 0 and random.random() < self.drop_state_prob:
            self.state = None

        input = self.create_input(data)

        res, state = self.model(input, self.state)
        loss = self.loss(res, data)

        self.state = U.apply_to_tensors(state, lambda x: x.detach_())
        return RecurrentResult(res, loss)

    def state_dict(self) -> Dict[str, Any]:
        return {"state": self.state}

    def load_state_dict(self, state: Dict[str, Any]):
        self.state = state["state"]
