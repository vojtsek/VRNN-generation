import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from . import VAECell
from ..dataset.embedding import Embeddings
from ..utils import zero_hidden


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

        # input is 'projected z sample' concat user and system encoded
        # hidden is configured
        self.vrnn_cell = torch.nn.LSTMCell(
            config['posterior_ff_sizes2'][-1] +
            config['input_encoder_hidden_size'] * 2 * (1 + int(config['bidirectional_encoder'])),
            config['vrnn_hidden_size'])
        self.embeddings_matrix = torch.nn.Embedding(len(embeddings.w2id), embeddings.d)
        self.vae_cell = VAECell(self.embeddings_matrix, self.vrnn_cell, self.config)

    def forward(self, user_dials, system_dials, user_lens, system_lens, dial_lens):
        batch_sort_perm = reversed(np.argsort(dial_lens))
        user_dials = user_dials[batch_sort_perm].transpose(1, 0)
        user_lens = user_lens[batch_sort_perm].transpose(1, 0)
        system_dials = system_dials[batch_sort_perm].transpose(1, 0)
        system_lens = system_lens[batch_sort_perm].transpose(1, 0)
        dial_lens = dial_lens[batch_sort_perm]
        
        user_dials_data, batch_sizes, sorted_indices, unsorted_indices =\
            pack_padded_sequence(user_dials, dial_lens, enforce_sorted=False)
        user_lens_data, _, __, ___ =\
            pack_padded_sequence(user_lens, dial_lens, enforce_sorted=False)
        system_dials_data, x, y, z = \
            pack_padded_sequence(system_dials, dial_lens, enforce_sorted=False)
        system_lens_data, _, __, ___ = \
            pack_padded_sequence(system_lens, dial_lens, enforce_sorted=False)

        initial_hidden = zero_hidden((2,
                                      self.config['batch_size'],
                                      self.config['vrnn_hidden_size']))
        initial_z = zero_hidden((self.config['batch_size'],
                                 self.config['z_logits_dim']))

        offset = 0
        vrnn_hidden_state = (initial_hidden[0], initial_hidden[1])
        z_latent = initial_z
        user_outputs = []
        q_zs, p_zs = [], []
        for bs in batch_sizes:
            decoded_user_outputs, vrnn_hidden_state, q_z, p_z =\
                self.vae_cell(user_dials_data[offset:offset+bs],
                              user_lens_data[offset:offset+bs],
                              system_dials_data[offset:offset+bs],
                              system_lens_data[offset:offset+bs],
                              (vrnn_hidden_state[0][:bs], vrnn_hidden_state[1][:bs]),
                              z_latent[:bs])
            offset += bs
            user_outputs.append(decoded_user_outputs)
            q_zs.extend(q_z)
            p_zs.extend(p_z)

        q_zs, lens = pad_packed_sequence(PackedSequence(
            torch.stack(q_zs), batch_sizes, sorted_indices, unsorted_indices))
        p_zs, lens = pad_packed_sequence(PackedSequence(
            torch.stack(p_zs), batch_sizes, sorted_indices, unsorted_indices))
        user_dials, lens = pad_packed_sequence(PackedSequence(
            user_dials_data, batch_sizes, sorted_indices, unsorted_indices))
        user_lens, lens = pad_packed_sequence(PackedSequence(
            user_lens_data, batch_sizes, sorted_indices, unsorted_indices))

        return user_dials, user_outputs, user_lens, q_zs, p_zs

    def cross_entropy_loss(self, logits, labels, reduction='mean'):
        return F.nll_loss(logits, labels, reduction=reduction)

    def training_step(self, train_batch, batch_idx):
        user_dials, system_dials, user_lens, system_lens, dial_lens = train_batch
        zeros = torch.from_numpy(np.zeros((user_dials.shape[0]), dtype=np.int64))
        user_dials, user_outputs, user_lens, q_zs, p_zs =\
            self.forward(user_dials, system_dials, user_lens, system_lens, dial_lens)

        total_decoder_ce_loss = 0
        total_count = 0
        for i, uo in enumerate(user_outputs):
            ud_reference = user_dials[i].transpose(0, 1)[:uo.shape[0]]
            batch_lens = user_lens[i, :uo.shape[1]]
            output_serialized, lens, sorted_indices, unsorted_indices =\
                pack_padded_sequence(uo, batch_lens, enforce_sorted=False)
            reference_serialized, lens, sorted_indices, unsorted_indices = \
                pack_padded_sequence(ud_reference, batch_lens, enforce_sorted=False)
            total_decoder_ce_loss += self.cross_entropy_loss(output_serialized, reference_serialized, reduction='sum')
            total_count += reference_serialized.shape[0]

        decoder_ce_loss = total_decoder_ce_loss / total_count
        logs = {'train_loss': decoder_ce_loss}
        return {'loss': decoder_ce_loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        user_dials, system_dials, user_lens, system_lens, dial_lens = val_batch
        logits = self.forward(user_dials, system_dials, user_lens, system_lens, dial_lens)
        zeros = torch.from_numpy(np.zeros((self.config['batch_size']), dtype=np.int64))
        # loss = self.cross_entropy_loss(logits, zeros)
        return {'val_loss': 0}

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
