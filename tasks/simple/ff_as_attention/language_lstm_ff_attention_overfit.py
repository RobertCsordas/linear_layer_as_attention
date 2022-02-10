from ... import task
from .language_lstm_ff_attention import LanguageLstmFfAttention, DatasetIndex
from ..language_model.aeshops_fables_lstm import AesopsFablesDataMixin


@task()
class LanguageLstmFfAttentionOverfit(AesopsFablesDataMixin, LanguageLstmFfAttention):
    def create_datasets(self):
        super().create_datasets()

        self.train_set = DatasetIndex(self.train_set)
        self.valid_sets.update({
            k: DatasetIndex(v) for k, v in self.valid_sets.items()
        })
