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
        self.embeddings_matrix.weight = torch.nn.Parameter(torch.from_numpy(embeddings.matrix))
        self.vae_cell = VAECell(self.embeddings_matrix, self.vrnn_cell, self.config)
        self.epoch_number = 0
        self.k = config['k_loss_coeff']
        self.alpha = config['alpha_coeff_step']
        self.lmbd = config['default_lambda_loss']

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
        user_outputs, system_outputs = [], []
        q_zs, p_zs, z_samples = [], [], []
        bow_logits_list = []
        for bs in batch_sizes:
            decoded_user_outputs, decoded_system_outputs, vrnn_hidden_state, q_z, p_z, z_sample, bow_logits =\
                self.vae_cell(user_dials_data[offset:offset+bs],
                              user_lens_data[offset:offset+bs],
                              system_dials_data[offset:offset+bs],
                              system_lens_data[offset:offset+bs],
                              (vrnn_hidden_state[0][:bs], vrnn_hidden_state[1][:bs]),
                              z_latent[:bs])
            offset += bs
            user_outputs.append(decoded_user_outputs)
            system_outputs.append(decoded_system_outputs)
            q_zs.extend(q_z)
            p_zs.extend(p_z)
            if self.config['with_bow_loss']:
                bow_logits_list.extend(bow_logits)
            z_samples.extend(z_sample)

        q_zs, lens = pad_packed_sequence(PackedSequence(
            torch.stack(q_zs), batch_sizes, sorted_indices, unsorted_indices))
        p_zs, lens = pad_packed_sequence(PackedSequence(
            torch.stack(p_zs), batch_sizes, sorted_indices, unsorted_indices))
        z_samples, lens = pad_packed_sequence(PackedSequence(
            torch.stack(z_samples), batch_sizes, sorted_indices, unsorted_indices))
        if self.config['with_bow_loss']:
            bow_logits, lens = pad_packed_sequence(PackedSequence(
                torch.stack(bow_logits_list), batch_sizes, sorted_indices, unsorted_indices))
        else:
            bow_logits = None

        user_dials, lens = pad_packed_sequence(PackedSequence(
            user_dials_data, batch_sizes, sorted_indices, unsorted_indices))
        user_lens, lens = pad_packed_sequence(PackedSequence(
            user_lens_data, batch_sizes, sorted_indices, unsorted_indices))
        system_dials, lens = pad_packed_sequence(PackedSequence(
            system_dials_data, batch_sizes, sorted_indices, unsorted_indices))
        system_lens, lens = pad_packed_sequence(PackedSequence(
            system_lens_data, batch_sizes, sorted_indices, unsorted_indices))

        return user_dials, user_outputs, user_lens, system_dials, system_outputs,\
               system_lens, q_zs, p_zs, z_samples, bow_logits

    def cross_entropy_loss(self, logits, labels, reduction='mean'):
        return F.nll_loss(logits, labels, reduction=reduction)

    def _compute_decoder_loss(self, outputs, reference, output_lens):
        total_loss = 0
        total_count = 0
        for i, uo in enumerate(outputs):
            ud_reference = reference[i].transpose(0, 1)[:uo.shape[0]]
            batch_lens = output_lens[i, :uo.shape[1]]
            output_serialized, lens, sorted_indices, unsorted_indices = \
                pack_padded_sequence(uo, batch_lens, enforce_sorted=False)
            reference_serialized, lens, sorted_indices, unsorted_indices = \
                pack_padded_sequence(ud_reference, batch_lens, enforce_sorted=False)
            total_loss += self.cross_entropy_loss(output_serialized, reference_serialized,
                                                  reduction='sum')
            total_count += reference_serialized.shape[0]
        return total_loss / total_count

    def _compute_bow_loss(self, bow_logits, outputs, reference, output_lens):
        # todo: more effective
        total_loss = 0
        total_count = 0
        # go through decoded sequence step by step
        # (i.e. we know how long is the decoded sequence in terms of number of turns)
        for i, uo in enumerate(outputs):
            ud_reference = reference[i].transpose(0, 1)[:uo.shape[0]]
            batch_lens = output_lens[i, :uo.shape[1]]
            # serialize references at turn i w.r.t respective turn lens across batch
            reference_serialized, lens, sorted_indices, unsorted_indices = \
                pack_padded_sequence(ud_reference, batch_lens, enforce_sorted=False)

            off = 0
            # go through each turn in batch
            for k, l in enumerate(batch_lens):
                bow_turn_token_vector = torch.zeros((bow_logits.shape[-1]))
                turn_tokens_reference = reference_serialized[off:off+l]
                turn_predicted_bow = bow_logits[i, k]  # predicted BOW at turn i for k-th dialogue in batch
                bow_turn_token_vector[turn_tokens_reference] = 1  # create BOW vector using ground truth tokens
                # print(torch.sum(bow_turn_token_vector), torch.sum(turn_predicted_bow))
                total_loss += F.mse_loss(bow_turn_token_vector, turn_predicted_bow, reduction='sum')
                off += l
                total_count += l
        return total_loss / total_count

    def _compute_cvae_kl_loss(self, q_zs):
        total_kl_loss = 0
        count = 0
        for q_z in q_zs:
            mu = q_z[:, :int(q_z.shape[1] / 2)]
            logvar = q_z[:, int(q_z.shape[1] / 2):]
            total_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            count += 1
        return total_kl_loss / count

    def _step(self, batch):
        user_dials, system_dials, user_lens, system_lens, dial_lens = batch
        user_dials, user_outputs, user_lens, system_dials, system_outputs,\
            system_lens, q_zs, p_zs, z_samples, bow_logits =\
            self.forward(user_dials, system_dials, user_lens, system_lens, dial_lens)
        total_user_decoder_loss = self._compute_decoder_loss(user_outputs, user_dials, user_lens)
        total_system_decoder_loss = self._compute_decoder_loss(system_outputs, system_dials, system_lens)
        if self.config['with_bow_loss']:
            total_bow_loss = self._compute_bow_loss(bow_logits, user_outputs, user_dials, user_lens)
        else:
            total_bow_loss = 0

        decoder_loss = (total_system_decoder_loss + total_user_decoder_loss) / 2
        if self.config['z_type'] == 'cont':
            kl_loss = self._compute_cvae_kl_loss(q_zs)
        else:
            # todo DVRNN prior regularization
            kl_loss = 0

        lambda_i = min(self.lmbd, self.k + self.alpha * self.epoch_number)
        loss = decoder_loss + lambda_i * total_bow_loss + kl_loss
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch)
        logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': logs}

    def validation_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _post_process_forwarded_batch(self, outputs, reference_dials, predictions, ground_truths, inv_vocab):
        for i, uo in enumerate(outputs):
            ud_reference = reference_dials[i].transpose(1, 0)[:uo.shape[0], :uo.shape[1]].numpy()
            uo = torch.argmax(uo, dim=2).numpy()
            predictions.append([list(map(lambda x: inv_vocab[x], row))[0] for row in uo])
            ground_truths.append([list(map(lambda x: inv_vocab[x], row))[0] for row in ud_reference])

    def predict(self, batch, inv_vocab):
        user_dials, system_dials, user_lens, system_lens, dial_lens = batch
        user_dials, user_outputs, user_lens, system_dials, system_outputs,\
            system_lens, q_zs, p_zs, z_samples, bow_logits = \
            self.forward(user_dials, system_dials, user_lens, system_lens, dial_lens)

        all_user_predictions, all_user_gt = [], []
        all_system_predictions, all_system_gt = [], []
        self._post_process_forwarded_batch(user_outputs, user_dials, all_user_predictions, all_user_gt, inv_vocab)
        self._post_process_forwarded_batch(system_outputs, system_dials, all_system_predictions, all_system_gt, inv_vocab)
        all_samples = (torch.argmax(z_samples, dim=2).numpy())

        return all_user_predictions, all_user_gt, all_system_predictions, all_system_gt, all_samples

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return [self._valid_loader]

    def test_dataloader(self):
        return [self._test_loader]


class EpochEndCb(pl.Callback):

    def on_epoch_end(self, trainer, model):
        model.epoch_number += 1
