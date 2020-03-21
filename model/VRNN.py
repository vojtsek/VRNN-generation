import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..dataset.embedding import Embeddings


class VRNN(pl.LightningModule):

    def __init__(self,
                 config: dict,
                 embeddings: Embeddings,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader):
        super(VRNN, self).__init__()
        self.config = config
        self.embeddings = Embeddings
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._test_loader = test_loader

        self.embeddings_matrix = torch.nn.Embedding(len(embeddings.w2id), embeddings.d)
        self.cell = torch.nn.LSTMCell(embeddings.d, config['hidden_size'])
        self.categorize = torch.nn.Linear(config['hidden_size'], 2)

    def forward(self, x):
        hidden = VRNN.zero_hidden((2, x.shape[0], self.config['hidden_size']))
        hidden = (hidden[0], hidden[1])
        x = self.embeddings_matrix(x)
        x = x.transpose(0, 1)
        for i in range(x.shape[2]):
            hidden = self.cell(x[0,:,i,:], hidden)
        logits = self.categorize(hidden[0])
        return F.log_softmax(logits, dim=-1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        user_dials, system_dials, user_lens, system_lens, dial_lens = train_batch
        zeros = torch.from_numpy(np.zeros((user_dials.shape[0]), dtype=np.int64))
        ones = torch.from_numpy(np.ones((system_dials.shape[0]), dtype=np.int64))
        logits = self.forward(user_dials)
        loss = self.cross_entropy_loss(logits, zeros)
        logits = self.forward(system_dials)
        loss += self.cross_entropy_loss(logits, ones)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        user_dials, system_dials, user_lens, system_lens, dial_lens = val_batch
        logits = self.forward(user_dials)
        zeros = torch.from_numpy(np.zeros((self.config['batch_size']), dtype=np.int64))
        print(logits.shape)
        loss = self.cross_entropy_loss(logits, zeros)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return [self._valid_loader]

    def test_dataloader(self):
        return [self._test_loader]

    @staticmethod
    def zero_hidden(sizes):
        return torch.randn(*sizes)
