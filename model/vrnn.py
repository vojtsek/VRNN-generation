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
            # user input (possibly bidirectional) + system logits
            config['system_z_logits_dim'] +
            config['input_encoder_hidden_size'] * (1 + int(config['bidirectional_encoder'])),
            config['vrnn_hidden_size'])
        self.embeddings_matrix = torch.nn.Embedding(len(embeddings.w2id), embeddings.d)
        self.embeddings_matrix.weight = torch.nn.Parameter(torch.from_numpy(embeddings.matrix))
        self.vae_cell = VAECell(self.embeddings_matrix, self.vrnn_cell, self.config)
        self.epoch_number = 0
        self.k = config['k_loss_coeff']
        self.alpha = config['alpha_coeff_step']
        self.lmbd = config['default_lambda_loss']

    def forward(self, user_dials, system_dials, nlu_dials, user_lens, system_lens, nlu_lens, dial_lens):
        batch_sort_perm = reversed(np.argsort(dial_lens))
        user_dials = user_dials[batch_sort_perm].transpose(1, 0)
        user_lens = user_lens[batch_sort_perm].transpose(1, 0)
        nlu_dials = nlu_dials[batch_sort_perm].transpose(1, 0)
        nlu_lens = nlu_lens[batch_sort_perm].transpose(1, 0)
        system_dials = system_dials[batch_sort_perm].transpose(1, 0)
        system_lens = system_lens[batch_sort_perm].transpose(1, 0)
        dial_lens = dial_lens[batch_sort_perm]

        user_dials_data, batch_sizes, sorted_indices, unsorted_indices =\
            pack_padded_sequence(user_dials, dial_lens, enforce_sorted=False)
        user_lens_data, _, __, ___ =\
            pack_padded_sequence(user_lens, dial_lens, enforce_sorted=False)
        nlu_dials_data, batch_sizes, sorted_indices, unsorted_indices =\
            pack_padded_sequence(nlu_dials, dial_lens, enforce_sorted=False)
        nlu_lens_data, _, __, ___ =\
            pack_padded_sequence(nlu_lens, dial_lens, enforce_sorted=False)

        system_dials_data, x, y, z = \
            pack_padded_sequence(system_dials, dial_lens, enforce_sorted=False)
        system_lens_data, _, __, ___ = \
            pack_padded_sequence(system_lens, dial_lens, enforce_sorted=False)

        initial_hidden = zero_hidden((2,
                                      self.config['batch_size'],
                                      self.config['vrnn_hidden_size']))

        offset = 0
        vrnn_hidden_state = (initial_hidden[0], initial_hidden[1])
        user_z_previous = zero_hidden((self.config['batch_size'], self.config['user_z_logits_dim']))
        system_z_previous = zero_hidden((self.config['batch_size'], self.config['system_z_logits_dim']))
        user_outputs, system_outputs, nlu_outputs = [], [], []
        user_q_zs, user_p_zs, z_samples = [], [], []
        system_q_zs, system_p_zs = [], []
        z_samples_matrix = []
        bow_logits_list = []
        for bs in batch_sizes:
            vae_output = self.vae_cell(user_dials_data[offset:offset+bs],
                                       user_lens_data[offset:offset+bs],
                                       nlu_lens_data[offset:offset+bs],
                                       system_dials_data[offset:offset+bs],
                                       system_lens_data[offset:offset+bs],
                                       (vrnn_hidden_state[0][:bs], vrnn_hidden_state[1][:bs]),
                                       user_z_previous[:bs], system_z_previous[:bs])
            offset += bs
            user_z_previous = vae_output.user_turn_output.q_z if self.training else vae_output.user_turn_output.p_z
            system_z_previous = vae_output.system_turn_output.q_z if self.training else vae_output.system_turn_output.p_z
            user_outputs.append(vae_output.user_turn_output.decoded_outputs[0])
            nlu_outputs.append(vae_output.user_turn_output.decoded_outputs[1])
            system_outputs.append(vae_output.system_turn_output.decoded_outputs[0])
            user_q_zs.extend(vae_output.user_turn_output.q_z)
            z_samples_matrix.extend(vae_output.user_turn_output.z_samples_lst)
            user_p_zs.extend(vae_output.user_turn_output.p_z)
            system_q_zs.extend(vae_output.system_turn_output.q_z)
            system_p_zs.extend(vae_output.system_turn_output.p_z)
            if self.config['with_bow_loss']:
                bow_logits_list.extend(vae_output.user_turn_output.bow_logits)
            z_samples.extend(vae_output.system_turn_output.z_samples)

        def _pad_packed(seq):
            if isinstance(seq, list):
                seq = torch.stack(seq)
            seq_padded, _ = pad_packed_sequence(PackedSequence(
                seq, batch_sizes, sorted_indices, unsorted_indices))
            return seq_padded

        return VRNNStepOutput(user_dials=_pad_packed(user_dials_data),
                              user_outputs=user_outputs,
                              user_lens=_pad_packed(user_lens_data),
                              system_dials=_pad_packed(system_dials_data),
                              system_outputs=system_outputs,
                              system_lens=_pad_packed(system_lens_data),
                              nlu_dials=_pad_packed(nlu_dials_data),
                              nlu_outputs=nlu_outputs,
                              nlu_lens=_pad_packed(nlu_lens_data),
                              user_q_zs=_pad_packed(user_q_zs),
                              user_p_zs=_pad_packed(user_p_zs),
                              system_q_zs=_pad_packed(system_q_zs),
                              system_p_zs=_pad_packed(system_p_zs),
                              z_samples=_pad_packed(z_samples),
                              bow_logits=_pad_packed(bow_logits_list) if self.config['with_bow_loss'] else None,
                              z_samples_matrix=_pad_packed(z_samples_matrix))

    def _compute_decoder_ce_loss(self, outputs, reference, output_lens):
        total_loss = 0
        total_count = 0
        for i, uo in enumerate(outputs):
            batch_size = uo.shape[1]
            ud_reference = reference[i].transpose(0, 1)[:uo.shape[0]]
            batch_lens = output_lens[i, :batch_size]
            output_serialized, lens, sorted_indices, unsorted_indices = \
                pack_padded_sequence(uo, batch_lens, enforce_sorted=False)
            reference_serialized, lens, sorted_indices, unsorted_indices = \
                pack_padded_sequence(ud_reference, batch_lens, enforce_sorted=False)
            total_loss += F.nll_loss(output_serialized, reference_serialized, reduction='mean') * batch_size
            total_count += batch_size
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

    def _compute_vae_kl_loss(self, q_zs, dial_lens):
        q_z, lens, sorted_indices, unsorted_indices = \
            pack_padded_sequence(q_zs, dial_lens, enforce_sorted=False)

        mu = q_z[:, :int(q_z.shape[1] / 2)]
        logvar = q_z[:, int(q_z.shape[1] / 2):]
        total_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return total_kl_loss

    def _compute_discrete_vae_kl_loss(self, q_zs, p_zs, dial_lens):

        q_z, lens, sorted_indices, unsorted_indices = \
            pack_padded_sequence(q_zs, dial_lens, enforce_sorted=False)
        p_z, lens, sorted_indices, unsorted_indices = \
            pack_padded_sequence(p_zs, dial_lens, enforce_sorted=False)

        # todo: has to be regularized w.r.t. variable batch sizes
        # if self.config['with_BPR']:
        #     p_z = torch.mean(p_z, dim=0).unsqueeze(0)
        #     q_z = torch.mean(q_z, dim=0).unsqueeze(0)
        log_q_z = torch.log(q_z + 1e-20)
        log_p_z = torch.log(p_z + 1e-20)
        kl = (log_q_z - log_p_z) * q_z
        kl = torch.mean(torch.sum(kl, dim=-1), dim=0)
        return kl

    def _step(self, batch):
        user_dials, system_dials, nlu_dials, user_lens, system_lens, nlu_lens, dial_lens = batch
        step_output = self.forward(*batch)
        total_user_decoder_loss = self._compute_decoder_ce_loss(step_output.user_outputs,
                                                                step_output.user_dials,
                                                                step_output.user_lens)
        total_system_decoder_loss = self._compute_decoder_ce_loss(step_output.system_outputs,
                                                                  step_output.system_dials,
                                                                  step_output.system_lens)

        total_nlu_decoder_loss = self._compute_decoder_ce_loss(step_output.nlu_outputs,
                                                               step_output.nlu_dials,
                                                               step_output.nlu_lens)
        if self.config['with_bow_loss']:
            total_bow_loss = self._compute_bow_loss(step_output.bow_logits,
                                                    step_output.user_outputs,
                                                    step_output.user_dials,
                                                    step_output.user_lens)
        else:
            total_bow_loss = 0

        decoder_loss = (total_system_decoder_loss + total_user_decoder_loss + total_nlu_decoder_loss) / 3
        if self.config['user_z_type'] == 'cont':
            # KL loss from N(0, 1)
            usr_kl_loss = self._compute_vae_kl_loss(step_output.user_q_zs, dial_lens)
        else:
            # KL loss between q_z and p_z
            usr_kl_loss = self._compute_discrete_vae_kl_loss(step_output.user_q_zs, step_output.user_p_zs, dial_lens)
        if self.config['system_z_type'] == 'cont':
            system_kl_loss = self._compute_vae_kl_loss(step_output.system_q_zs, dial_lens)
        else:
            system_kl_loss = self._compute_discrete_vae_kl_loss(step_output.system_q_zs, step_output.system_p_zs, dial_lens)

        lambda_i = min(self.lmbd, self.k + self.alpha * self.epoch_number)
        min_epochs = 75
        final_kl_term = 1
        increase_start_epoch = 10
        # exponential decrease
        step = np.exp(np.log(final_kl_term / self.config['init_KL_term']) / min_epochs)
        lambda_kl = self.config['init_KL_term'] *\
                    step ** min(max(self.epoch_number - increase_start_epoch, 0), min_epochs)
        # step = (final_kl_term - self.config['init_KL_term']) / min_epochs
        # lambda_kl = self.config['init_KL_term'] +\
        #             step * min(max(self.epoch_number - increase_start_epoch, 0), min_epochs)
        kl_loss = (system_kl_loss + usr_kl_loss) / 2
        loss = decoder_loss + lambda_i * total_bow_loss + lambda_kl * kl_loss
        return loss, usr_kl_loss, total_user_decoder_loss, system_kl_loss, total_system_decoder_loss

    def training_step(self, train_batch, batch_idx):
        loss, usr_kl_loss, usr_decoder_loss, system_kl_loss, system_decoder_loss = self._step(train_batch)
        logs = {'train_total_loss': loss,
                'train_user_kl_loss': usr_kl_loss,
                'train_user_decoder_loss': usr_decoder_loss,
                'train_system_kl_loss': system_kl_loss,
                'train_system_decoder_loss': system_decoder_loss,
                'train_decoder_loss': (usr_decoder_loss + system_decoder_loss) / 2,
                'train_kl_loss': (usr_kl_loss + system_kl_loss) / 2
                }
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        loss, usr_kl_loss, usr_decoder_loss, system_kl_loss, system_decoder_loss= self._step(val_batch)
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
        # user_dials, system_dials, user_lens, system_lens, dial_lens = batch
        step_output = self.forward(*batch)

        all_user_predictions, all_user_gt = [], []
        all_system_predictions, all_system_gt = [], []
        all_nlu_predictions, all_nlu_gt = [], []
        self._post_process_forwarded_batch(step_output.user_outputs,
                                           step_output.user_dials,
                                           all_user_predictions,
                                           all_user_gt,
                                           inv_vocab)

        self._post_process_forwarded_batch(step_output.nlu_outputs,
                                           step_output.nlu_dials,
                                           all_nlu_predictions,
                                           all_nlu_gt,
                                           inv_vocab)

        self._post_process_forwarded_batch(step_output.system_outputs,
                                           step_output.system_dials,
                                           all_system_predictions,
                                           all_system_gt,
                                           inv_vocab)

        all_samples = torch.argmax(step_output.z_samples, dim=2).numpy()
        all_samples_matrix = torch.argmax(step_output.z_samples_matrix, dim=2).numpy()
        return PredictedOuputs(all_user_predictions,
                               all_user_gt,
                               all_system_predictions,
                               all_system_gt,
                               all_nlu_predictions,
                               all_nlu_gt,
                               all_samples,
                               all_samples_matrix)

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


def checkpoint_callback(filepath):
    return pl.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )


class VRNNStepOutput:
    def __init__(self,
                 user_dials=None,
                 user_outputs=None,
                 user_lens=None,
                 system_dials=None,
                 system_outputs=None,
                 system_lens=None,
                 nlu_dials=None,
                 nlu_outputs=None,
                 nlu_lens=None,
                 user_q_zs=None,
                 user_p_zs=None,
                 system_q_zs=None,
                 system_p_zs=None,
                 z_samples=None,
                 bow_logits=None,
                 z_samples_matrix=None):
        self.user_dials = user_dials
        self.user_outputs = user_outputs
        self.user_lens = user_lens
        self.system_dials = system_dials
        self.system_outputs = system_outputs
        self.system_lens = system_lens
        self.nlu_dials = nlu_dials
        self.nlu_outputs = nlu_outputs
        self.nlu_lens = nlu_lens
        self.user_q_zs = user_q_zs
        self.user_p_zs = user_p_zs
        self.system_q_zs = system_q_zs
        self.system_p_zs = system_p_zs
        self.z_samples = z_samples
        self.bow_logits = bow_logits
        self.z_samples_matrix = z_samples_matrix


class PredictedOuputs:
    def __init__(self,
                 all_user_predictions=None,
                 all_user_gt=None,
                 all_system_predictions=None,
                 all_system_gt=None,
                 all_nlu_predictions=None,
                 all_nlu_gt=None,
                 all_z_samples=None,
                 all_z_samples_matrix=None):
        self.all_user_predictions = all_user_predictions
        self.all_user_gt = all_user_gt
        self.all_system_predictions = all_system_predictions
        self.all_nlu_predictions = all_nlu_predictions
        self.all_nlu_gt = all_nlu_gt
        self.all_system_gt = all_system_gt
        self.all_z_samples = all_z_samples
        self.all_z_samples_matrix = all_z_samples_matrix
