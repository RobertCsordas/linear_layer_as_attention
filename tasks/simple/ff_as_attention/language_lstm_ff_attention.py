from dataclasses import dataclass
from ..language_model.wikitext2_lstm import Wikitext2Lstm
from ... import task, args
import torch
from models import RNNLanguageModel
from typing import Optional, Tuple, List, Dict, Any
from framework.utils import U
import os
import framework
import numpy as np
from interfaces import Result
import dataset
import pickle
from tqdm import tqdm
import random

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-ff_as_attention.lm.n_samples", default=3)

@dataclass
class AnalysisResult:
    scores: Dict[str, torch.Tensor]
    indices: torch.Tensor


class DatasetIndex(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, unroll_length: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.unroll_length = unroll_length or dataset.unroll_len

    def __getitem__(self, item: int) -> Dict[str, Any]:
        d = self.dataset.__getitem__(item)

        d["index"] = np.arange(item * self.unroll_length, item * self.unroll_length + d["data"].shape[0],
                               dtype=np.int64)
        return d

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)



@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-ff_as_attention.hugepath", default="./huge")


class LayerSeparatedLstm(torch.nn.Module):
    def __init__(self, input_size: int, state_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.state_size = state_size
        self.layers = torch.nn.ModuleList([torch.nn.LSTM(input_size if i == 0 else state_size, state_size) 
                                           for i in range(n_layers)])

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        new_states = []

        for i, l in enumerate(self.layers):
            if i != 0:
                input = self.dropout(input)

            if state is None:
                z = torch.zeros(1, input.shape[1], self.state_size, dtype=input.dtype, device=input.device)
                currstat = (z, z)
            else:
                currstat = state[i]

            out, state_u = l(input, currstat)
            new_states.append(state_u)

        return out, new_states

# torch.nn.LSTM(self.helper.args.state_size, self.helper.args.state_size,
#                                                      self.helper.args.n_layers, dropout = self.helper.args.dropout)

class TrainHistory:
    def __init__(self, name: str, savepath: str):
        self.layer_inputs = {}
        self.steps = {}
        self.savepath = os.path.join(savepath, name)
        self.at_pos = -1
        os.makedirs(self.savepath, exist_ok=True)
 
    def load(self):
        for f in os.listdir(self.savepath):
            self.layer_inputs[f] = self.open_file(f, "ab")

        # The counts are not saved, so check when do we run out of the file
        n = 0
        endpos = self.layer_inputs["index"].tell()
        self.seek_front()

        while self.layer_inputs["index"].tell() != endpos:
            pickle.load(self.layer_inputs["index"])
            n += 1

        self.seek_end()
        self.steps = {k: n for k in self.layer_inputs.keys()}

    def open_file(self, name: str, mode: str):
        return open(os.path.join(self.savepath, name), mode)

    def close_all(self):
        for v in self.layer_inputs.values():
            v.close()

    def reopen(self, mode: str):
        self.layer_inputs = {k: self.open_file(k, mode) for k in self.layer_inputs.keys()}

    def add(self, name: str, val):
        assert self.at_pos == -1
        # return
        l = self.layer_inputs.get(name)
        if l is None:
            self.layer_inputs[name] = l = self.open_file(name, "wb")

        self.steps[name] = self.steps.get(name, 0) + 1
        val = U.apply_to_tensors(val, lambda x: x.detach().cpu())
        val = U.apply_to_tensors(val, lambda x: x.half() if x.dtype == torch.float32 else x)
        pickle.dump(val, l, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return max(self.steps.values())

    def step(self):
        # return
        assert min(self.steps.values()) == max(self.steps.values())

    def seek_front(self):
        self.at_pos = 0
        self.reopen("rb")

    def seek_end(self):
        self.at_pos == -1
        self.reopen("ab")

    def __iter__(self):
        self.seek_front()
        return self

    def __next__(self) -> Dict[str, Any]:
        assert self.at_pos != -1

        if self.at_pos >= max(self.steps.values()):
            self.seek_end()
            raise StopIteration()

        self.at_pos += 1

        return {
            k: pickle.load(v) for k, v in self.layer_inputs.items()
        }

    def cleanup(self):
        self.close_all()

        for name in self.layer_inputs.keys():
            os.remove(os.path.join(self.savepath, name))

@task()
class LanguageLstmFfAttention(Wikitext2Lstm):
    @property
    def should_i_log(self) -> bool:
        return self.model.training or self.force_log

    def linear_hook(self, module, input, output):
        if self.should_i_log:
            self.history.add(self.module_map[id(module)], input[0].detach().cpu())

    def lstm_hook(self, module, input, output):
        if self.should_i_log:
            # The full input for the next step is the previous output fed back and concatenated with the input.
            # Also, it should be shifted and concatenated with the first input
            all_input = torch.cat([torch.cat([input[1][0].detach(), output[0][:-1].detach()],0), input[0].detach()], -1)
            self.history.add(self.module_map[id(module)], all_input.cpu())

    def run_model(self, data: Dict[str, torch.Tensor]) -> Tuple[Result, Dict[str, Any]]:
        if self.should_i_log:
            self.history.add("index", data["index"])

        res = self.model_interface(data)
        self.history.step()
        return res, {}

    def get_huge_path(self) -> str:
        return os.path.join(self.helper.args.ff_as_attention.hugepath, os.path.relpath(self.helper.dirs.base, "./"))

    def create_datasets(self):
        super().create_datasets()

        self.train_set = DatasetIndex(self.train_set)
        self.valid_sets.update({
            k: DatasetIndex(v) for k, v in self.valid_sets.items()
        })

    def get_scores(self, states: Dict[str, torch.Tensor], history: TrainHistory) -> AnalysisResult:
        device = self.helper.device
        # device = torch.device("cpu")
        states = {k: v.to(device) for k, v in states.items()}
        scores = {k: [] for k in states.keys()}
        indices = []

        print("Calculating matches...")
        with torch.no_grad():
            for h in tqdm(history):
                for k, s in states.items():
                    scores[k].append((s.flatten(end_dim=-2) @ h[k].flatten(end_dim=-2).T.to(device)).abs().cpu().view(*states[k].shape[:-1], -1))
                indices.append(h["index"][:-1].flatten())

        return AnalysisResult({k: torch.cat(v, -1) for k, v in scores.items()}, torch.cat(indices))

    def generate_and_get_state(self, input: torch.Tensor, n_gen: int) -> Dict[str, torch.Tensor]:
        train_history = self.history
        self.history = TrainHistory("tmp", self.get_huge_path())

        self.history.add("index", 0)
        self.model.eval()
        self.force_log = True
        with torch.no_grad():
            out, _ = self.model.generate(input, n_gen, sample=False)
        self.model.train()
        self.force_log  = False

        assert len(self.history) == 1 + n_gen
        for h in self.history:
            res = h
            break

        del res["index"]

        self.history.cleanup()
        self.history = train_history

        return res, out

    # def analyze(self, sentence: str):
    #     input = torch.tensor(self.train_set.vocabulary(sentence.split(" ")), dtype=torch.long, device=self.helper.device).unsqueeze(0)
    #     states = self.run_and_get_state(input)

    #     scores = self.get_scores(states, self.history)

    #     print(scores["rnn.layers.0"].shape)

    def analyze_set(self, dset: torch.utils.data.Dataset) -> Dict[str, Any]:
        window = self.helper.args.lm.example_window
        n_samples = self.helper.args.ff_as_attention.lm.n_samples
        K = 3
        inputs, samples = self.pick_random_sentences(dset, n_samples, self.helper.args.lm.example_context, window)
        states, out = self.generate_and_get_state(inputs, window)

        # Pick the element "window" steps before the end
        states = {k: v[-1:] for k, v in states.items()}
        scores = self.get_scores(states, self.history)

        # Mark the picked element
        for i, s in enumerate(samples):
            samples[i] = self.mark_token(s, window - 1)

        ssorted = {k: torch.topk(v.float(), 500, dim=-1) for k, v in scores.scores.items()}

        assert out.shape[0] >= 2 * window
        out = out[out.shape[0] - 2 * window:]

        ds_scores = {}
        for k, v in scores.scores.items():
            ds_scores[k] = torch.zeros(n_samples, self.train_set.linear_len(), dtype=torch.float32)
            ds_scores[k].index_add_(1, scores.indices, v[0].type_as(ds_scores[k]))

        ds_scores = {k: torch.topk(v.float(), 500, dim=-1) for k, v in ds_scores.items()}

        plots = {}
        for s in range(n_samples):
            out_str = self.train_set.vocabulary.to_string(out[:, s].cpu().numpy().tolist())
            out_str = self.mark_token(out_str, window - 1)

            for lname, sobj in ssorted.items():                
                t = f"REF: \n\n{samples[s]}\n\nNet next word prediction\n\n: {out_str}"
                t_offs = 0
                known_pos = set()

                for k in range(K):
                    #train_indices = scores.index[sobj.indices]
                    start_offs = t_offs
                    train_index = scores.indices[sobj.indices[0, s, t_offs]]
                    known_pos.add(train_index.item())

                    # Skip and count repeated matches
                    n = 0
                    while scores.indices[sobj.indices[0, s, t_offs]] == train_index:
                        t_offs += 1
                        n += 1

                    # Skip positions that are already seen
                    while scores.indices[sobj.indices[0, s, t_offs]].item() in known_pos:
                        t_offs += 1

                    offset = min(window - 1, train_index)
                    closest_match = self.train_set.get_linear(train_index - offset, offset + window)
                    closest_match = self.train_set.vocabulary.to_string(closest_match)
                    closest_match = self.mark_token(closest_match, offset)

                    t += f"\n\nTop {k+1}, index {train_index}, score {sobj.values[0, s, start_offs].item()}, {n} repeats\n\n {closest_match}"

                plots[f"closest_match_per_update/{s}/{lname}"] = framework.visualize.plot.Text(t)


                t = f"REF: \n\n{samples[s]}\n\nNet next word prediction\n\n: {out_str}"
                for k in range(K):
                    train_index = ds_scores[lname].indices[s, k]

                    offset = min(window - 1, train_index)
                    closest_match = self.train_set.get_linear(train_index - offset, offset + window)
                    closest_match = self.train_set.vocabulary.to_string(closest_match)
                    closest_match = self.mark_token(closest_match, offset)

                    t += f"\n\nTop {k+1}, score {ds_scores[lname].values[s, k].item()}\n\n {closest_match}"

                plots[f"closest_match_total/{s}/{lname}"] = framework.visualize.plot.Text(t)
        return plots


        # print(scores.scores["rnn.layers.0"].shape)
        # print(samples, inputs.shape)


    def analyze(self):
        plots = {}
        plots.update({f"test/{k}": v for k, v in self.analyze_set(self.valid_sets.test).items()})
        plots.update({f"train/{k}": v for k, v in self.analyze_set(self.train_set).items()})
        self.helper.log(plots)


    def create_model(self) -> torch.nn.Module:
        self.history = TrainHistory("train", self.get_huge_path())
        self.module_map = {}
        self.force_log = False

        model = RNNLanguageModel(len(self.train_set.vocabulary), self.helper.args.embedding_size,  
                                 self.helper.args.state_size, self.helper.args.dropout,
                                 tied_embedding=self.helper.args.tied_embedding,
                                 rnn = LayerSeparatedLstm(self.helper.args.state_size, self.helper.args.state_size,
                                                          self.helper.args.n_layers, dropout = self.helper.args.dropout))

        self.n_weights = sum(p.numel() for p in model.parameters())

        for lname, m in model.named_modules():
            self.module_map[id(m)] = lname
            if isinstance(m, torch.nn.Linear):
                print(f"Linear hook for {lname}: {m}")
                m.register_forward_hook(self.linear_hook)
            elif isinstance(m, torch.nn.LSTM):
                print(f"LSTM hook for {lname}: {m}")
                m.register_forward_hook(self.lstm_hook)

        return model

    def train(self):
        res = super().train()
        # self.history.load()
        self.analyze()
        return res
