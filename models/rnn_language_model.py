import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Tuple

RNNLanguageModelState = Tuple[torch.Tensor, torch.Tensor]

class RNNLanguageModel(torch.nn.Module):
    def __init__(self, voc_size: int, embedding_size: Optional[int], state_size: int, dropout: float, tied_embedding: bool,
                 rnn: torch.nn.Module):
        super().__init__()

        self.embedding = torch.nn.Embedding(voc_size, embedding_size or state_size)
        with torch.no_grad():
            self.embedding.weight.uniform_(-0.1, 0.1)
        if embedding_size is None:
            self.embedding_adapter = lambda x: x
        else:
            self.embedding_adapter = torch.nn.Linear(embedding_size, state_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.rnn = rnn
        self.output_adapter = lambda x: x

        if tied_embedding:
            self.output = torch.nn.Linear(embedding_size or state_size, voc_size)
            self.output.weight = self.embedding.weight
            if embedding_size is not None:
                self.output_adapter = torch.nn.Linear(state_size, embedding_size)
        else:
            self.output = torch.nn.Linear(state_size, voc_size)

    def generate(self, prefix: torch.Tensor, n_steps: int, sample: bool = True, temperature: float = 1) -> Tuple[torch.Tensor, RNNLanguageModelState]:
        def gen_output(soft_out: torch.Tensor) -> torch.Tensor:
            if sample:
                probs = F.softmax(soft_out / temperature, -1)
                return torch.multinomial(probs.flatten(end_dim=-2), 1).view(*probs.shape[:-1])
            else:
                return soft_out.argmax(-1)

        output, state = self.forward(prefix, None)
        outputs = [gen_output(output)]
        for _ in range(n_steps):
            output, state = self.forward(outputs[-1][-1:], state)
            outputs.append(gen_output(output))

        return torch.cat(outputs, 0), state

    def forward(self, x: torch.Tensor, state: Optional[RNNLanguageModelState]) -> \
            Tuple[torch.Tensor, RNNLanguageModelState]:

        net = self.dropout(self.embedding(x.long()))
        net = self.embedding_adapter(net)
        net, state2 = self.rnn(net, state)

        net = self.output_adapter(net)

        net = self.output(net)
        return net, state2
