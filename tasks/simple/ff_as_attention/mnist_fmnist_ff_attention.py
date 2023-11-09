from dataclasses import dataclass
import torch
import torch.nn.functional as F
import framework
import dataset
from ..simple_task import SimpleTask
from models import FeedforwardModel
from interfaces import FeedforwardImageClassifierInterface
from ... import task, args
from interfaces import Result
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm
import random


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-ff_as_attention.analyze_n_samples", default=1)
    parser.add_argument("-ff_as_attention.export", default=False)

class DatasetIndex(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        d = self.dataset[index]
        d["index"] = np.asarray(index, dtype=np.int64)
        return d

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class DatasetLimit(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, max_len: int):
        super().__init__()
        self.dataset = dataset
        self.l = min(len(dataset), max_len)
        self.indices = np.random.RandomState(0x123).permutation(len(dataset))[:self.l].tolist()

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return self.l

    def __getattr__(self, item):
        return getattr(self.dataset, item)


@dataclass
class TrainDataHistory:
    indices: torch.Tensor
    ds_indices: Optional[torch.Tensor]
    timesteps: torch.Tensor
    labels: torch.Tensor

    def save(self) -> Dict[str, Any]:
        return {
            "indices": self.indices.cpu(),
            "ds_indices": self.ds_indices.cpu() if self.ds_indices is not None else None,
            "timesteps": self.timesteps.cpu(),
            "labels": self.labels
        }

    def __getitem__(self, *args):
        return TrainDataHistory(
            self.indices.__getitem__(*args),
            self.ds_indices.__getitem__(*args) if self.ds_indices is not None else None,
            self.timesteps.__getitem__(*args),
            self.labels.__getitem__(*args)
        )


class MatchTracker:
    def __init__(self) -> None:
        self.vote_matches_pred = 0
        self.vote_matches_target = 0

    def update(self, vote: torch.Tensor, ref: torch.Tensor, out: torch.Tensor):
        vote = vote.to(out.device)
        self.vote_matches_pred = self.vote_matches_pred + \
                                 F.one_hot((vote == out).long(),2).T.float() @ F.one_hot((ref == out).long(),2).float()

        self.vote_matches_target = self.vote_matches_target + \
                                 F.one_hot((vote == ref).long(),2).T.float() @ F.one_hot((ref == out).long(),2).float()

    def plot(self) -> Dict[str, Any]:
        def get_normalized(m: torch.Tensor) -> torch.Tensor:
            m = torch.cat([m, m.sum(-1, keepdim=True)], -1)
            return m / m.sum(0, keepdim=True).clamp(min=1)

        v_m_p = get_normalized(self.vote_matches_pred)
        v_m_t = get_normalized(self.vote_matches_target)

        return {
            "vote_predicting_output": framework.visualize.plot.Heatmap(v_m_p, "prediction correct", "vote matches output", x_marks=["N", "Y", "*"], y_marks=["N", "Y"]),
            "vote_predicting_target": framework.visualize.plot.Heatmap(v_m_t, "prediction correct", "vote matches target", x_marks=["N", "Y", "*"], y_marks=["N", "Y"])
        }

    def save(self) -> Dict[str, torch.Tensor]:
        return {
            "vote_matches_pred": self.vote_matches_pred.cpu(),
            "vote_matches_target": self.vote_matches_target.cpu()
        }


@task()
class MnistFmnistFFAttention(SimpleTask):
    HISTORY_KEYS = ["history", "grad_history", "index_history", "ds_index_history", "label_history"]

    def create_train_set(self, raw_datasets: List[torch.utils.data.Dataset]) -> torch.utils.data.Dataset:
        return framework.loader.DatasetMerger([DatasetIndex(d) for d in raw_datasets], add_index=True)

    def get_raw_datasets(self) -> List[torch.utils.data.Dataset]:
        return [dataset.image.PermutedMNIST(0, "train"), dataset.image.FashionMNIST(0, "train")]

    def init_valid_sets(self):
        self.valid_sets["valid_mnist"] = dataset.image.PermutedMNIST(0, "valid")
        self.valid_sets["valid_fmnist"] = dataset.image.FashionMNIST(0, "valid")
        self.valid_sets["test_mnist"] = dataset.image.PermutedMNIST(0, "test")
        self.valid_sets["test_fmnist"] = dataset.image.FashionMNIST(0, "test")

    def create_datasets(self):
        self.batch_dim = 0
        self.raw_datasets = self.get_raw_datasets()
        self.train_set = self.create_train_set(self.raw_datasets)

        self.init_valid_sets()
        self.valid_sets = {k: DatasetIndex(v) for k, v in self.valid_sets.items()}
        self.clear_history()

        self.force_log = False
        # assert False

    def clear_history(self):
        self.history = {}
        self.grad_history = {}
        self.index_history = []
        self.ds_index_history = []
        self.label_history = []

    def save_history(self) -> Dict[str, Any]:
        return {k: self.__dict__[k] for k in self.HISTORY_KEYS}

    def restore_history(self, dict: Dict[str, Any]):
        self.__dict__.update(dict)

    def get_activations(self, data: torch.Tensor):
        history = self.save_history()
        self.clear_history()

        self.force_log = True

        self.model.eval()
        with torch.no_grad():
            self.model(data.to(self.helper.device))
        self.model.train()

        new_data = self.history

        self.restore_history(history)
        self.force_log = False

        return {k: v[0] for k, v in new_data.items()}

    def get_train_set(self, ts_id: int) -> torch.utils.data.Dataset:
        return self.raw_datasets[ts_id]

    def magic_plot(self, blocks: List[torch.Tensor], marker=None) -> framework.visualize.plot.XYChart:
        data = {}
        offset = 0
        for ds_id in range(len(self.raw_datasets)):
            n = blocks[ds_id].shape[0]
            data[self.raw_datasets[ds_id].__class__.__name__] = list(zip(range(offset, offset + n), blocks[ds_id]))
            offset += n

        return framework.visualize.plot.XYChart(data, point_markers=marker, legend=len(self.raw_datasets) > 1)

    def magic_histogram(self, blocks: List[torch.Tensor]) -> framework.visualize.plot.Barplots:
        data = {}
        offsets = {}
        offset = 0
        for ds_id in range(len(self.raw_datasets)):
            n = blocks[ds_id].shape[0]
            data[self.raw_datasets[ds_id].__class__.__name__] = blocks[ds_id]
            offsets[self.raw_datasets[ds_id].__class__.__name__] = offset
            offset += n

        return framework.visualize.plot.Barplots(data, offsets=offsets)

    def plot_per_update(self, all_scores: torch.Tensor, data_hist: TrainDataHistory) -> Dict[str, Any]:
        plots = {}

        assert (all_scores >= 0).all()
        all_scores_sorted, order = all_scores.sort(dim=0, descending=True)

        data_hist_s = data_hist[order]

        if data_hist_s.ds_indices is not None:
            if len(self.raw_datasets) > 1:
                plots["datasets_500"] = framework.visualize.plot.XYChart({"datasets": list(zip(range(500), data_hist_s.ds_indices[:500].cpu()))})

        plots["top_500"] = framework.visualize.plot.XYChart({"top500": list(zip(range(500), all_scores_sorted[:500].cpu()))})
        plots["labels_500"] = framework.visualize.plot.XYChart({"labels": list(zip(range(500), data_hist_s.labels[:500].cpu()))})
        plots["timesteps_500"] = framework.visualize.plot.XYChart({"timesteps": list(zip(range(500), data_hist_s.timesteps[:500].cpu()))})

        score_blocks = [[] for _ in range(len(self.raw_datasets))]
        timestep_blocks = [[] for _ in range(len(self.raw_datasets))]
        score_integrals = [[] for _ in range(len(self.raw_datasets))]
        for ds_id in range(len(self.raw_datasets)):
            for l in range(10):
                mask = (data_hist.labels == l) & (data_hist.ds_indices is None or data_hist.ds_indices == ds_id)
                s = all_scores[mask].sort(dim=0, descending=True)

                score_blocks[ds_id].append(s.values[:500].cpu())
                score_blocks[ds_id].append(torch.tensor([float("nan")]))
                timestep_blocks[ds_id].append(data_hist.timesteps[mask][s.indices[:500]].cpu())
                timestep_blocks[ds_id].append(torch.tensor([float("nan")]))
                # score_integrals[ds_id].append(all_scores[mask].sum())
                score_integrals[ds_id].append(s.values.sum().cpu())


        score_blocks = [torch.cat(b, 0) for b in score_blocks]
        timestep_blocks = [torch.cat(b, 0) for b in timestep_blocks]
        score_integrals = [torch.stack(b, 0) for b in score_integrals]

        plots["magic_class_plot"] = self.magic_plot(score_blocks, marker='x')
        plots["magic_timestep_plot"] = self.magic_plot(timestep_blocks)
        # plots["total_score_per_class"] = framework.visualize.plot.Histogram(torch.cat(score_integrals, 0))

        plots["total_score_per_class"] = self.magic_histogram(score_integrals)

        score_means = torch.stack([s.mean() for s in score_integrals])
        score_std = torch.stack([s.std() for s in score_integrals])
        normalized_score_std = score_means / score_std

        plots["mean_score_per_dataset"] = framework.visualize.plot.Barplot(score_means, [ds.__class__.__name__ for ds in self.raw_datasets], stds=score_std)
        plots["score_std_per_dataset"] = framework.visualize.plot.Barplot(score_std, [ds.__class__.__name__ for ds in self.raw_datasets])
        plots["score_normalized_std_per_dataset"] = framework.visualize.plot.Barplot(normalized_score_std, [ds.__class__.__name__ for ds in self.raw_datasets])

        if len(self.raw_datasets) > 1:
            total_score = sum(score_blocks)
            plots["magic_class_plot_total"] = framework.visualize.plot.XYChart({"total": list(zip(range(total_score.shape[0]), total_score))})
            plots["total_score_per_class_total"] = framework.visualize.plot.Barplot(sum(score_integrals))

        for i in range(3):
            ds = self.get_train_set(int(data_hist_s.ds_indices[i].item())) if data_hist_s.ds_indices is not None else self.train_set
            plots[f"closest_training_{i}"] = framework.visualize.plot.Image(ds.unnormalize(ds[data_hist_s.indices[i]]["image"])/255)

        return plots

    def get_all_scores(self, data: Dict[str, torch.Tensor], history: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        if history is None:
            history = self.save_history()

        device = "cpu"
        data = {k: v.to(device) for k, v in data.items()}

        total_scores = {k: [torch.zeros(len(tset), dtype=torch.float32, device=device) for tset in self.raw_datasets] for k in history["history"].keys()}
        all_scores = {k: [] for k in history["history"].keys()}

        for lname in history["history"].keys():
            for step, chunk in enumerate(history["history"][lname]):
                scores = (data[lname] @ chunk.T.to(device)).abs()
                all_scores[lname].append(scores)

                for i in range(len(self.raw_datasets)):
                    ds_mask = (history["ds_index_history"][step] == i).to(device)
                    total_scores[lname][i].index_add_(0, history["index_history"][step][ds_mask].to(device), scores[0][ds_mask])

        return {k: torch.cat(v, -1) for k, v in all_scores.items()}, total_scores

    def get_data_history(self) -> TrainDataHistory:
        indices = torch.cat(self.index_history, 0)
        labels = torch.cat(self.label_history, 0)
        ds_indices = torch.cat(self.ds_index_history, -1).long() if self.ds_index_history[0] is not None else None
        timesteps = torch.cat([torch.full_like(ind, i) for i, ind in enumerate(self.index_history)], -1)
        return TrainDataHistory(indices, ds_indices, timesteps, labels)


    def plot_total_scores(self, per_sample_scores: List[torch.Tensor]):
        ds_ids = torch.cat([torch.full_like(s, i, dtype=torch.long) for i, s in enumerate(per_sample_scores)], -1)
        all_scores = torch.cat(per_sample_scores, -1)

        all_scores = all_scores.sort(dim=-1, descending=True)
        ds_ids = ds_ids[all_scores.indices]
        all_scores = all_scores.values
        indices = torch.arange(all_scores.shape[-1])

        lnames = list(sorted([ts.__class__.__name__ for ts in self.raw_datasets], reverse=True))
        point_marker_sizes = {n: 2 + 1*i for i, n in enumerate(lnames)}

        return framework.visualize.plot.XYChart({
            ts.__class__.__name__: list(zip(indices[ds_ids==i], all_scores[ds_ids==i])) for i, ts in enumerate(self.raw_datasets)
        }, line_styles=" ", point_markers=".", point_marker_size=point_marker_sizes, alpha=0.7)


    def analyze(self, data: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = self.get_activations(data)

        # activation shape: N, d_model
        # history shape: N_train, d_model

        data_history = self.get_data_history()
        all_grad_norms = {k: torch.cat(v, -1) for k, v in self.grad_history.items()}

        all_scores, per_sample_scores = self.get_all_scores(data)
        plots = {}
        for lname in self.history.keys():
            plots.update({
                f"per_update/{lname}/{k}": v for k, v in self.plot_per_update(all_scores[lname][0], data_history).items()
            })
           
            plots.update({
                f"sample_total/{lname}/total_contribution": self.plot_total_scores(per_sample_scores[lname])
            })

        to_save = {
            "data_hist": data_history.save(),
            "all_scores": {k: v.cpu() for k, v in all_scores.items()},
            "all_grad_norms": {k: v.cpu() for k, v in all_grad_norms.items()}
        }

        return plots, to_save

    def add_to_dictlist(self, d: Dict, key: Any, data: Any):
        l = d.get(key)
        if l is None:
            l = []
            d[key] = l

        l.append(data)

    def linear_hook(self, module, input, output):
        if module.training or self.force_log:
            self.add_to_dictlist(self.history, self.module_map[id(module)], input[0].detach().cpu())

    def linear_bw_hook(self, module, grad_in, grad_out):
        if module.training or self.force_log:
            self.add_to_dictlist(self.grad_history, self.module_map[id(module)], grad_out[0].detach().norm(dim=-1).cpu())

    def register_hooks(self, model: torch.nn.Module):
        self.module_map = {}
        for k, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.module_map[id(m)] = k
                m.register_forward_hook(self.linear_hook)
                m.register_backward_hook(self.linear_bw_hook)

    def create_model(self) -> torch.nn.Module:
        n_inputs = self.train_set[0]["image"].size
        model = FeedforwardModel(n_inputs, 10, self.helper.args.layer_sizes)
        self.register_hooks(model)
        return model

    def create_model_interface(self):
        self.model_interface = FeedforwardImageClassifierInterface(self.model)

    def run_model(self, data: Dict[str, torch.Tensor]) -> Tuple[Result, Dict[str, Any]]:
        if self.model.training or self.force_log:
            self.index_history.append(data["index"].detach().cpu())
            self.label_history.append(data["label"].detach().cpu())
            self.ds_index_history.append(data["ds_index"].detach().cpu() if "ds_index" in data else None)
        res = self.model_interface(data)
        return res, {}

    def find_misclassified(self, name: str, loader: torch.utils.data.DataLoader) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
        self.model.eval()

        # all_indices = torch.cat(self.index_history, 0)
        # all_labels = torch.tensor([self.train_set[i]["label"] for i in all_indices], dtype=torch.int32)

        data_hist = self.get_data_history()
        #masks = {ds_id: [(data_hist.labels == l) & (data_hist.ds_indices is None or data_hist.ds_indices == ds_id) for l in range(10)] for ds_id in range(len(self.raw_datasets))}
        masks = [(data_hist.labels == l) for l in range(10)]
        history = self.save_history()
        self.force_log = True
        # all_norms = {k: torch.cat(v) for k, v in history["grad_history"].items()}

        state_trackers = {
            k: MatchTracker() for k in history["history"].keys()
        }

        # state_trackers_normalized = {
        #     k: MatchTracker() for k in history["history"].keys()
        # }

        bad = []
        good = []
        bad_model_outputs = []

        labelmasks = [(data_hist.labels == (i % 10)) & (data_hist.ds_indices == (i // 10)) for i in range(10 * len(self.raw_datasets))]
        score_per_label_sum_good = {k: 0 for k in history["history"].keys()}
        score_per_label_sqsum_good = {k: 0 for k in history["history"].keys()}
        score_good_count = 0

        score_per_label_sum_bad = {k: 0 for k in history["history"].keys()}
        score_per_label_sqsum_bad = {k: 0 for k in history["history"].keys()}
        score_bad_count = 0

        with torch.no_grad():
            for i, d in enumerate(tqdm(loader)):
                self.clear_history()

                d = self.prepare_data(d)
                res, _ = self.run_model(d)

                outputs = self.model_interface.decode_outputs(res)

                badmask = outputs != d["label"]
                bad.append(d["index"][badmask])
                good.append(d["index"][~badmask])
                bad_model_outputs.append(outputs[badmask])

                ref = {k: v[0] for k, v in self.history.items()}

                scores, _ = self.get_all_scores(ref, history)

                for lname, s in scores.items():
                    # Calculate the predictive powers
                    won = torch.stack([s[...,masks[l]].sum(-1) for l in range(10)], -1).argmax(-1)
                    state_trackers[lname].update(won, d["label"], outputs)

                    # Calculate the mean of the normalized variances over good and bad examples
                    score_per_label = torch.stack([s[:, lm].sum(-1) for lm in labelmasks], 0).view(10, len(self.raw_datasets), -1)
                    score_var_norm = score_per_label.mean(0) / score_per_label.std(0)

                    badmask = badmask.to(score_var_norm.device)

                    score_per_label_sum_good[lname] += score_var_norm[:, ~badmask].sum(-1)
                    score_per_label_sqsum_good[lname] += score_var_norm[:, ~badmask].pow(2).sum(-1)
                    score_per_label_sum_bad[lname] += score_var_norm[:, badmask].sum(-1)
                    score_per_label_sqsum_bad[lname] += score_var_norm[:, badmask].pow(2).sum(-1)                    

                n_bad = badmask.long().sum().item()
                score_good_count += badmask.shape[0] - n_bad
                score_bad_count += n_bad

            score_per_label_good = {k: v / score_good_count for k, v in score_per_label_sum_good.items()}
            score_per_label_bad = {k: v / score_bad_count for k, v in score_per_label_sum_bad.items()}

            score_per_label_good_std = {k: ((sqsum - score_good_count*score_per_label_good[k].pow(2)) / (score_good_count-1)).sqrt() for k, sqsum in score_per_label_sqsum_good.items()}
            score_per_label_bad_std = {k: ((sqsum - score_bad_count*score_per_label_bad[k].pow(2)) / (score_bad_count-1)).sqrt() for k, sqsum in score_per_label_sqsum_bad.items()}

        self.model.train()

        self.restore_history(history)

        plots = {}
        for lname, st in state_trackers.items():
            plots.update({f"analysis/vote_predictive_power/{name}/{lname}/{k}": v for k, v in st.plot().items()})
            self.helper.export_tensor(f"analysis/vote_predictive_power/{name}/{lname}", st.save())

            plots[f"analysis/mean_of_normalized_class_variances_per_dataset/{name}/{lname}/good"] = framework.visualize.plot.Barplot(score_per_label_good[lname], [d.__class__.__name__ for d in self.raw_datasets], stds=score_per_label_good_std[lname])
            plots[f"analysis/mean_of_normalized_class_variances_per_dataset/{name}/{lname}/bad"] = framework.visualize.plot.Barplot(score_per_label_bad[lname], [d.__class__.__name__ for d in self.raw_datasets], stds=score_per_label_bad_std[lname])

            plots[f"analysis/mean_of_normalized_class_variances_per_dataset/{name}/{lname}/good_std"] = framework.visualize.plot.Barplot(score_per_label_good_std[lname], [d.__class__.__name__ for d in self.raw_datasets])
            plots[f"analysis/mean_of_normalized_class_variances_per_dataset/{name}/{lname}/bad_std"] =  framework.visualize.plot.Barplot(score_per_label_bad_std[lname], [d.__class__.__name__ for d in self.raw_datasets])

        return torch.cat(good, 0).cpu().numpy(), torch.cat(bad, 0).cpu().numpy().tolist(),\
               torch.cat(bad_model_outputs, 0).cpu().numpy().tolist(), plots#, state_trackers_normalized

    def train(self):
        res = super().train()
        # VIS_INDEX = 1

        an_res = {}
        to_test = {k: (self.valid_loaders[k], self.valid_sets[k]) for k in self.valid_sets.keys() if "test" in k}

        train_sets = {f"train_{ds.__class__.__name__}": DatasetIndex(DatasetLimit(ds, 1000)) for ds in self.raw_datasets}
        to_test.update({k: (self.create_valid_loader(v), v) for k, v in train_sets.items()})

        for name, (loader, dset) in to_test.items():
            good, bad, bad_outputs, plots = self.find_misclassified(name, loader)
            
            bad_i = np.random.permutation(len(bad))[:self.helper.args.ff_as_attention.analyze_n_samples]
            good_i = np.random.permutation(len(good))[:self.helper.args.ff_as_attention.analyze_n_samples]

            an_res.update(plots)

            for i, vis_index in enumerate(bad_i):
                vis_index = random.randrange(len(bad))
                d = dset[bad[vis_index]]

                p, to_save = self.analyze(torch.tensor(d["image"]).unsqueeze(0).flatten(1))
                an_res.update({f"analysis/misclassified/{name}/{i}/{k}": v for k, v in p.items()})
                an_res[f"analysis/misclassified/{name}/{i}/input"] = framework.visualize.plot.Image(dset.unnormalize(d["image"])/255)
                an_res[f"analysis/misclassified/{name}/{i}/details"] = framework.visualize.plot.Text(f"dataset: {name}, sample index: {bad[vis_index]}, target label: {d['label']}, model output: {bad_outputs[vis_index]}")

                if self.helper.args.ff_as_attention.export:
                    self.helper.export_tensor(f"analysis/misclassified/{name}/{i}", to_save)

            for i, vis_index in enumerate(good_i):
                d = dset[good[vis_index]]

                p, to_save = self.analyze(torch.tensor(d["image"]).unsqueeze(0).flatten(1))
                an_res.update({f"analysis/correct/{name}/{i}/{k}": v for k, v in p.items()})
                an_res[f"analysis/correct/{name}/{i}/input"] = framework.visualize.plot.Image(dset.unnormalize(d["image"])/255)
                an_res[f"analysis/correct/{name}/{i}/details"] = framework.visualize.plot.Text(f"dataset: {name}, sample index: {good[vis_index]}, target label/model output: {d['label']}")

                if self.helper.args.ff_as_attention.export:
                    self.helper.export_tensor(f"analysis/correct/{name}/{i}", to_save)


        self.helper.log(an_res)
