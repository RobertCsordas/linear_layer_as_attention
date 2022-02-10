import dataset
from ... import task
from .enwik8_lstm import Enwik8Lstm


class AesopsFablesDataMixin:
    def create_datasets(self):
        self.batch_dim = 1
        # super().create_datasets()

        split_def = [
            dataset.LMFile("pad", "https://www.gutenberg.org/files/49010/49010-0.txt", 17563),
            dataset.LMFile("train", "https://www.gutenberg.org/files/49010/49010-0.txt", None),
            dataset.LMFile("test", "https://www.gutenberg.org/files/49010/49010-0.txt", 0.1),
            dataset.LMFile("pad", "https://www.gutenberg.org/files/49010/49010-0.txt", 22356)
        ]

        self.train_set = dataset.CharLanguageDataset(split_def, "train", self.helper.args.lm.unroll)
        self.valid_sets.test = dataset.CharLanguageDataset(split_def, "test", self.helper.args.lm.unroll)

    def mark_token(self, text: str, pos: int) -> str:
        return text[:pos] + " !!!!! " + text[pos] + " !!!!! " + text[pos + 1:]


@task()
class AesopsFablesLstm(AesopsFablesDataMixin, Enwik8Lstm):
    pass
