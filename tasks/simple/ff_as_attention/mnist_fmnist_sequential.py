import dataset
import torch
import framework
from .mnist_fmnist_ff_attention import MnistFmnistFFAttention, DatasetIndex
from typing import Tuple, Dict, Any, List
from interfaces import Result
from ... import task,args


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-mnist_fmnist_seq.reverse", default=False)
    parser.add_argument("-mnist_fmnist_seq.switch_timesteps", default="", parser=parser.int_list_parser)


@task()
class MnistFmnistSequentialFFAttention(MnistFmnistFFAttention):
    def create_train_set(self, raw_datasets: List[torch.utils.data.Dataset]) -> torch.utils.data.Dataset:
        self.curr_train_set = 0
        self.all_training_sets = [DatasetIndex(ds) for ds in raw_datasets]

        if self.helper.args.mnist_fmnist_seq.reverse:
            self.all_training_sets = list(reversed(self.all_training_sets))

        return self.all_training_sets[self.curr_train_set]

    def run_model(self, data: Dict[str, torch.Tensor]) -> Tuple[Result, Dict[str, Any]]:
        data["ds_index"] = torch.full([data["image"].shape[0]], self.curr_train_set, dtype=torch.int32)
        res, plt = super().run_model(data)

        if self.helper.args.mnist_fmnist_seq.switch_timesteps:
            switch = self.curr_train_set < len(self.helper.args.mnist_fmnist_seq.switch_timesteps) and \
                     self.helper.state.iter == self.helper.args.mnist_fmnist_seq.switch_timesteps[self.curr_train_set]
        else:
            switch = self.helper.state.iter % (self.helper.args.stop_after // len(self.all_training_sets)) == 0
        if switch and self.helper.state.iter != 0 and (self.curr_train_set + 1) < len(self.all_training_sets):
            print(f"Iteration {self.helper.state.iter}. Switching to next dataset")
            self.curr_train_set += 1
            self.set_train_set(self.all_training_sets[self.curr_train_set])

        return res, plt
