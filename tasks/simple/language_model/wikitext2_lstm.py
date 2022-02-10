import dataset
import framework
from ... import task
from .enwik8_lstm import Enwik8Lstm
from typing import Tuple, List, Dict, Any, Union
import torch
import random


@task()
class Wikitext2Lstm(Enwik8Lstm):
    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Wikitext2WordLevel("train", self.helper.args.lm.unroll)
        self.valid_sets.val = dataset.Wikitext2WordLevel("valid", self.helper.args.lm.unroll)
        self.valid_sets.test = dataset.Wikitext2WordLevel("test", self.helper.args.lm.unroll)

    def mark_token(self, text: Union[str, List[str]], pos: int) -> str:
        if not isinstance(text, List):
            text = text.split(" ")

        text.insert(pos+1, "!!!!!")
        text.insert(pos, "!!!!!")
        return " ".join(text)
