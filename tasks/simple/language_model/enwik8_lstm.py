import framework
import torch
import torch.nn
import torch.utils.data
import dataset
from typing import Tuple, Any, Dict, List, Union
from ..simple_task import SimpleTask
from models import RNNLanguageModel
from interfaces import LanguageModelInterface
from ... import task, args
import random


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.state_drop_probability", default=0.01)
    parser.add_argument("-lm.unroll", default=100)
    parser.add_argument("-lm.example_context", default=100)
    parser.add_argument("-lm.example_window", default=40)


@task()
class Enwik8Lstm(SimpleTask):
    VALID_NUM_WORKERS = 0
    TRAIN_NUM_WORKERS = 2

    def __init__(self, helper: framework.helpers.TrainingHelper):
        super().__init__(helper)

        self.rnd_valid = {k: self.pick_random_sentences(v, 3, self.helper.args.lm.example_context,
                             self.helper.args.lm.example_window) for k, v in self.valid_sets.items()}
        self.rnd_train = self.pick_random_sentences(self.train_set, 3, self.helper.args.lm.example_context,
                                                    self.helper.args.lm.example_window)

    def create_state(self):
        self.helper.state.epoch = 0

    def create_model(self) -> torch.nn.Module:
        rnn = torch.nn.LSTM(self.helper.args.state_size, self.helper.args.state_size,
                            self.helper.args.n_layers, dropout = self.helper.args.dropout)
        framework.utils.lstm_init_forget(rnn)

        model = RNNLanguageModel(len(self.train_set.vocabulary), self.helper.args.embedding_size,
                                 self.helper.args.state_size, self.helper.args.dropout,
                                 tied_embedding=self.helper.args.tied_embedding,
                                 rnn = rnn)
        
        self.n_weights = sum(p.numel() for p in model.parameters())
        return model

    def create_model_interface(self):
        self.model_interface = LanguageModelInterface(self.model, drop_state_prob=self.helper.args.lm.state_drop_probability)
        self.helper.saver["interface"] = self.model_interface

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        state = self.model_interface.state
        self.model_interface.reset_state()
        res = super().validate_on(set, loader)
        self.model_interface.state = state
        return res

    def log_epoch(self):
        self.helper.log({"epoch": self.helper.state.epoch})

    def start_next_epoch(self):
        self.model_interface.reset_state()
        self.helper.state.epoch += 1
        self.log_epoch()

    def get_train_batch(self) -> Dict[str, Any]:
        try:
            return next(self.data_iter)
        except StopIteration:
            self.start_next_epoch()
            self.data_iter = iter(self.train_loader)
            return next(self.data_iter)

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(vset,
                                   batch_sampler=framework.loader.sampler.MultibatchSequentialSampler(vset,
                                                    self.helper.args.batch_size),
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS)

    def create_train_loader(self, loader: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        sampler = framework.loader.sampler.MultibatchSequentialSampler(loader, self.helper.args.batch_size)
        self.helper.saver.register("sampler", sampler, replace=True)

        return torch.utils.data.DataLoader(loader, batch_sampler=sampler, num_workers=self.TRAIN_NUM_WORKERS,
                                           pin_memory=True, collate_fn=framework.loader.collate.VarLengthCollate(
                                           batch_dim=self.batch_dim))

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Enwik8("train", self.helper.args.lm.unroll)
        self.valid_sets.val = dataset.Enwik8("valid", self.helper.args.lm.unroll)
        self.valid_sets.test = dataset.Enwik8("test", self.helper.args.lm.unroll)

    def train(self):
        self.log_epoch()
        super().train()

    def pick_random_sentences(self, set, n: int, content_length: int, window: int) -> Tuple[torch.Tensor, List[str]]:
        res = []

        window_offset = max(0, window - content_length)
        for i in range(n):
            pos = random.randrange(window, set.linear_len() - max(content_length, window))
            res.append(set.get_linear(pos + window_offset, max(content_length, window) + window))

        data = torch.tensor(res, dtype=torch.long, device=self.helper.device).T
        input = data[abs(window_offset) : -window]

        to_display = data[abs(window_offset) + content_length - window:]
        to_display_str = []
        for b in range(to_display.shape[1]):
            to_display_str.append(set.vocabulary.to_string(to_display[:, b].cpu().numpy().tolist()))

        return input, to_display_str

    def mark_token(self, text: str, pos: int) -> str:
        # Pos is unaccurate for byte dataset because of the bytearray->string conversion
        return text

    def generate_example(self, input_data: torch.Tensor, inputs: List[str], sample: bool = True) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model.generate(input_data, self.helper.args.lm.example_window, sample=sample)
        self.model.train()

        res = {}
        for i, ins in enumerate(inputs):
            outs = self.mark_token(self.train_set.vocabulary.to_string(out[-2*self.helper.args.lm.example_window:, i].cpu().numpy().tolist()), self.helper.args.lm.example_window - 1)
            text = f"Input: \n\n {self.mark_token(inputs[i], self.helper.args.lm.example_window - 1)}\n\nOutput: {outs}"
            res[f"examples/{i}"] = framework.visualize.plot.Text(text)

        return res

    def validate(self) -> Dict[str, Any]:
        res = super().validate()
        for vname, vdata in self.rnd_valid.items():
            res.update({f"test/{vname}/{k}": v for k, v in self.generate_example(*vdata).items()})
        res.update({f"train/{k}": v for k, v in self.generate_example(*self.rnd_train).items()})
        res.update({f"train_argmax/{k}": v for k, v in self.generate_example(*self.rnd_train, sample=False).items()})
        return res
