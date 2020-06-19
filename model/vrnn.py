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
        self.embeddings = embeddings
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._test_loader = test_loader

        # input is 'projected z sample' concat user and system encoded
        # hidden is configured
        self.vrnn_cell = torch.nn.LSTMCell(
            # user input (possibly bidirectional) + system logits
            config['system_z_total_size'] +
            config['input_encoder_hidden_size'] * (1 + int(config['bidirectional_encoder'])),
            config['vrnn_hidden_size'])
        self.embeddings_matrix = torch.nn.Embedding(len(embeddings.w2id), embeddings.d).to(self.config['device'])
        self.embeddings_matrix.weight = torch.nn.Parameter(torch.from_numpy(embeddings.matrix))
        self.vae_cell = VAECell(self.embeddings_matrix,
                                self.vrnn_cell,
                                self.config,
                                embeddings)
        self.epoch_number = 0
        self.k = config['k_loss_coeff']
        self.alpha = config['alpha_coeff_step']
        self.lmbd = config['default_lambda_loss']

    def forward(self, user_dials, system_dials, usr_nlu_dials,
                sys_nlu_dials, user_lens, system_lens, usr_nlu_lens,
                sys_nlu_lens, db_res, dial_lens):
        batch_sort_perm = torch.LongTensor(list(reversed(np.argsort(dial_lens.cpu().numpy())))).to(self.config['device'])
        user_dials = user_dials[batch_sort_perm].transpose(1, 0)
        user_lens = user_lens[batch_sort_perm].transpose(1, 0)
        usr_nlu_dials = usr_nlu_dials[batch_sort_perm].transpose(1, 0)
        usr_nlu_lens = usr_nlu_lens[batch_sort_perm].transpose(1, 0)
        sys_nlu_dials = sys_nlu_dials[batch_sort_perm].transpose(1, 0)
        sys_nlu_lens = sys_nlu_lens[batch_sort_perm].transpose(1, 0)
        system_dials = system_dials[batch_sort_perm].transpose(1, 0)
        system_lens = system_lens[batch_sort_perm].transpose(1, 0)
        db_res = db_res[batch_sort_perm].transpose(1, 0)
        dial_lens = dial_lens[batch_sort_perm]

        user_dials_data, batch_sizes, sorted_indices, unsorted_indices =\
            pack_padded_sequence(user_dials, dial_lens, enforce_sorted=False)
        user_lens_data, _, __, ___ =\
            pack_padded_sequence(user_lens, dial_lens, enforce_sorted=False)
        usr_nlu_dials_data, batch_sizes, sorted_indices, unsorted_indices = \
            pack_padded_sequence(usr_nlu_dials, dial_lens, enforce_sorted=False)
        usr_nlu_lens_data, _, __, ___ = \
            pack_padded_sequence(usr_nlu_lens, dial_lens, enforce_sorted=False)
        db_res_data, _, __, ___ = \
            pack_padded_sequence(db_res, dial_lens, enforce_sorted=False)

        sys_nlu_dials_data, batch_sizes, sorted_indices, unsorted_indices = \
            pack_padded_sequence(sys_nlu_dials, dial_lens, enforce_sorted=False)
        sys_nlu_lens_data, _, __, ___ = \
            pack_padded_sequence(sys_nlu_lens, dial_lens, enforce_sorted=False)

        system_dials_data, x, y, z = \
            pack_padded_sequence(system_dials, dial_lens, enforce_sorted=False)
        system_lens_data, _, __, ___ = \
            pack_padded_sequence(system_lens, dial_lens, enforce_sorted=False)


        initial_hidden = zero_hidden((2,
                                      self.config['batch_size'],
                                      self.config['vrnn_hidden_size'])).to(self.config['device'])

        offset = 0
        vrnn_hidden_state = (initial_hidden[0], initial_hidden[1])
        user_z_previous = zero_hidden((self.config['batch_size'], self.config['user_z_total_size'])).to(self.config['device'])
        system_z_previous = zero_hidden((self.config['batch_size'], self.config['system_z_total_size'])).to(self.config['device'])
        to_del = [initial_hidden, user_z_previous, system_z_previous]
        user_outputs, system_outputs = [], []
        usr_nlu_outputs, sys_nlu_outputs = [], []
        user_q_zs, user_p_zs, z_samples = [], [], []
        system_q_zs, system_p_zs = [], []
        p_z_samples_matrix, q_z_samples_matrix = [], []
        user_z_samples_matrix = []
        db_data = []
        bow_logits_list = []
        for bs in batch_sizes:
            use_prior = not self.training or np.random.rand(1) > self.config['z_teacher_forcing_prob']
            vae_output = self.vae_cell(user_dials_data[offset:offset+bs],
                                       user_lens_data[offset:offset+bs],
                                       usr_nlu_lens_data[offset:offset+bs],
                                       sys_nlu_lens_data[offset:offset+bs],
                                       system_dials_data[offset:offset+bs],
                                       system_lens_data[offset:offset+bs],
                                       db_res_data[offset:offset+bs],
                                       (vrnn_hidden_state[0][:bs], vrnn_hidden_state[1][:bs]),
                                       user_z_previous[:bs], system_z_previous[:bs],
                                       use_prior)
            db_data.append(db_res_data[offset:offset+bs])
            offset += bs

            user_z_previous = self.vae_cell.aggregate(vae_output.user_turn_output.q_z.transpose(1, 0))
            system_z_previous = self.vae_cell.aggregate(vae_output.system_turn_output.p_z.transpose(1, 0)
                                                        if use_prior else
                                                        vae_output.system_turn_output.q_z.transpose(1, 0))
            user_outputs.append(vae_output.user_turn_output.decoded_outputs[0])
            usr_nlu_outputs.append(vae_output.user_turn_output.decoded_outputs[1])
            system_outputs.append(vae_output.system_turn_output.decoded_outputs[0])
            sys_nlu_outputs.append(vae_output.system_turn_output.decoded_outputs[1])
            user_q_zs.extend(vae_output.user_turn_output.q_z)
            user_p_zs.extend(vae_output.user_turn_output.p_z)
            p_z_samples_matrix.extend(vae_output.system_turn_output.p_z_samples_lst)
            q_z_samples_matrix.extend(vae_output.system_turn_output.q_z_samples_lst)
            user_z_samples_matrix.extend(vae_output.user_turn_output.q_z_samples_lst)
            system_q_zs.extend(vae_output.system_turn_output.q_z)
            system_p_zs.extend(vae_output.system_turn_output.p_z)
            if self.config['with_bow_loss']:
                bow_logits_list.extend(vae_output.user_turn_output.bow_logits)
            z_samples.extend(vae_output.system_turn_output.z_samples)

        for d in to_del:
            del d
        del batch_sort_perm
        # torch.cuda.empty_cache()
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
                              usr_nlu_dials=_pad_packed(usr_nlu_dials_data),
                              usr_nlu_outputs=usr_nlu_outputs,
                              usr_nlu_lens=_pad_packed(usr_nlu_lens_data),
                              sys_nlu_dials=_pad_packed(sys_nlu_dials_data),
                              sys_nlu_outputs=sys_nlu_outputs,
                              sys_nlu_lens=_pad_packed(sys_nlu_lens_data),
                              user_q_zs=_pad_packed(user_q_zs),
                              user_p_zs=_pad_packed(user_p_zs),
                              system_q_zs=_pad_packed(system_q_zs),
                              system_p_zs=_pad_packed(system_p_zs),
                              z_samples=_pad_packed(z_samples),
                              bow_logits=_pad_packed(bow_logits_list) if self.config['with_bow_loss'] else None,
                              p_z_samples_matrix=_pad_packed(p_z_samples_matrix),
                              q_z_samples_matrix=_pad_packed(q_z_samples_matrix),
                              user_z_samples_matrix=_pad_packed(user_z_samples_matrix),
                              db_data=db_data)

    def _compute_decoder_ce_loss(self, outputs, reference, output_lens, pr=False):
        total_loss = 0
        total_count = 0
        for i, uo in enumerate(outputs):
            batch_size = uo.shape[1]
            ud_reference = reference[i].transpose(0, 1)[:uo.shape[0]]
            batch_lens = output_lens[i, :batch_size]
            sort_perm = torch.LongTensor(list(reversed(np.argsort(batch_lens.cpu().numpy())))).to(self.config['device'])
            output_serialized, lens1, sorted_indices, unsorted_indices = \
                pack_padded_sequence(uo[:, sort_perm, :], batch_lens[sort_perm], enforce_sorted=True)
            reference_serialized, lens2, sorted_indices, unsorted_indices = \
                pack_padded_sequence(ud_reference[:, sort_perm], batch_lens[sort_perm], enforce_sorted=True)
                # print(torch.argmax(output_serialized, dim=-1))
                # print(reference_serialized)
                # print(batch_size)
            total_loss += F.nll_loss(output_serialized, reference_serialized, reduction='mean') * batch_size
            total_count += batch_size
            del sort_perm

        # torch.cuda.empty_cache()
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

        mu = q_z[..., :int(q_z.shape[-1] / 2)]
        logvar = q_z[..., int(q_z.shape[-1] / 2):]
        total_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=2), dim=0)
        return torch.sum(total_kl_loss)

    def _compute_discrete_vae_kl_loss(self, q_zs, p_zs, dial_lens):
        q_z, lens, sorted_indices, unsorted_indices = \
            pack_padded_sequence(q_zs, dial_lens, enforce_sorted=True)
        p_z, lens, sorted_indices, unsorted_indices = \
            pack_padded_sequence(p_zs, dial_lens, enforce_sorted=True)

        # todo: has to be regularized w.r.t. variable batch sizes
        offset = 0
        p_zs_norm, q_zs_norm = [], []
        if self.config['with_bpr']:
            for l in lens:
                p_norm = torch.mean(p_z[offset:offset + l], dim=0)
                p_zs_norm.append(p_norm)
                q_zs_norm.append(torch.mean(q_z[offset:offset+l], dim=0))
                offset += l
            q_z = torch.stack(q_zs_norm)
            p_z = torch.stack(p_zs_norm)
        q_z = q_z.detach()
        q_labels = torch.argmax(q_z, dim=-1).detach()
        p_labels = torch.argmax(p_z, dim=-1).detach()
        log_q_z = torch.log(q_z + 1e-20).detach()
        log_p_z = torch.log(p_z + 1e-20)
        kl = (log_q_z - log_p_z) * q_z
        q_max, _ = torch.max(q_z, dim=-1)
        p_max, _ = torch.max(p_z, dim=-1)
        q_diffs = torch.ones(*q_max.shape).to(self.config['device']) - q_max
        p_diffs = torch.ones(*p_max.shape).to(self.config['device']) - p_max
        # KL per each Z vector
        kl = torch.sum(torch.mean(torch.sum(kl, dim=-1), dim=0))
        ce = F.nll_loss(log_p_z.transpose(2, 1), q_labels, reduction='mean')
        # print('KL, CE', kl, ce)
        del p_diffs, q_diffs
        # torch.cuda.empty_cache()
        return ce + kl

    def _step(self, batch, optimizer_idx=0):
        user_dials, system_dials, usr_nlu_dials, sys_nlu_dials, user_lens,\
        system_lens, usr_nlu_lens, sys_nlu_lens, db_res, dial_lens = batch
        step_output = self.forward(*batch)
        total_user_decoder_loss = self._compute_decoder_ce_loss(step_output.user_outputs,
                                                                step_output.user_dials,
                                                                step_output.user_lens)
        total_system_decoder_loss = self._compute_decoder_ce_loss(step_output.system_outputs,
                                                                  step_output.system_dials,
                                                                  step_output.system_lens, pr=True)

        total_usr_nlu_decoder_loss = self._compute_decoder_ce_loss(step_output.usr_nlu_outputs,
                                                               step_output.usr_nlu_dials,
                                                               step_output.usr_nlu_lens)

        total_sys_nlu_decoder_loss = self._compute_decoder_ce_loss(step_output.sys_nlu_outputs,
                                                               step_output.sys_nlu_dials,
                                                               step_output.sys_nlu_lens)
        if self.config['with_bow_loss']:
            total_bow_loss = self._compute_bow_loss(step_output.bow_logits,
                                                    step_output.user_outputs,
                                                    step_output.user_dials,
                                                    step_output.user_lens)
        else:
            total_bow_loss = 0

        decoder_losses = [total_system_decoder_loss, total_user_decoder_loss,
                        total_usr_nlu_decoder_loss, total_sys_nlu_decoder_loss]
        decoder_loss = torch.mean(torch.stack(decoder_losses))
        batch_sort_perm = torch.LongTensor(list(reversed(np.argsort(dial_lens.cpu().numpy())))).to(self.config['device'])
        dial_lens = dial_lens[batch_sort_perm]
        if self.config['user_z_type'] == 'cont':
            # KL loss from N(0, 1)
            usr_kl_loss = self._compute_vae_kl_loss(step_output.user_q_zs, dial_lens)
        else:
            # KL loss between q_z and p_z
            usr_kl_loss = self._compute_discrete_vae_kl_loss(step_output.user_q_zs, step_output.user_p_zs, dial_lens)
        if self.config['system_z_type'] == 'cont':
            system_kl_loss = self._compute_vae_kl_loss(step_output.system_q_zs, dial_lens)
        else:
            system_kl_loss =\
                self._compute_discrete_vae_kl_loss(step_output.system_q_zs, step_output.system_p_zs, dial_lens)

        del step_output.system_q_zs, step_output.system_p_zs, step_output.user_q_zs, step_output.user_p_zs, batch_sort_perm
        lambda_i = min(self.lmbd, self.k + self.alpha * self.epoch_number)
        final_usr_kl_term = 1
        final_sys_kl_term = 10
        increase_start_epoch = 10
        # exponential decrease
        step_usr = np.exp(np.log(final_usr_kl_term / self.config['init_KL_term']) / self.config['min_epochs'])
        step_system = np.exp(np.log(final_sys_kl_term / self.config['init_KL_term']) / self.config['min_epochs'])
        lambda_usr_kl = self.config['init_KL_term'] *\
                        step_usr ** min(max(self.epoch_number - increase_start_epoch, 0), self.config['min_epochs'])
        lambda_sys_kl = self.config['init_KL_term'] *\
                        step_system ** min(max(self.epoch_number - increase_start_epoch, 0), self.config['min_epochs'])

        # step = (final_kl_term - self.config['init_KL_term']) / min_epochs
        # lambda_kl = self.config['init_KL_term'] +\
        #             step * min(max(self.epoch_number - increase_start_epoch, 0), min_epochs)
        kl_loss = (system_kl_loss + usr_kl_loss) / 2
        if optimizer_idx == 0:
            loss = decoder_loss + lambda_usr_kl * usr_kl_loss # + .1 * q_penalty
        else:
            loss = system_kl_loss + total_system_decoder_loss
        # loss = decoder_loss + lambda_usr_kl * usr_kl_loss + lambda_sys_kl * system_kl_loss
        return loss, usr_kl_loss, total_user_decoder_loss, system_kl_loss, total_system_decoder_loss

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        loss, usr_kl_loss, usr_decoder_loss, system_kl_loss, system_decoder_loss =\
            self._step(train_batch, optimizer_idx)
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
        loss, usr_kl_loss, usr_decoder_loss, system_kl_loss, system_decoder_loss = self._step(val_batch)
        logs = {'valid_total_loss': loss,
                'valid_user_kl_loss': usr_kl_loss,
                'valid_user_decoder_loss': usr_decoder_loss,
                'valid_system_kl_loss': system_kl_loss,
                'valid_system_decoder_loss': system_decoder_loss,
                'valid_decoder_loss': (usr_decoder_loss + system_decoder_loss) / 2,
                'valid_kl_loss': (usr_kl_loss + system_kl_loss) / 2
                }
        return {'val_loss': loss,
                'valid_user_kl_loss': usr_kl_loss,
                'valid_user_decoder_loss': usr_decoder_loss,
                'valid_system_kl_loss': system_kl_loss,
                'valid_system_decoder_loss': system_decoder_loss,
                'valid_decoder_loss': (usr_decoder_loss + system_decoder_loss) / 2,
                'valid_kl_loss': (usr_kl_loss + system_kl_loss) / 2,
                'log': logs}

    def validation_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        for o in outputs:
            tensorboard_logs = {k: torch.stack([x[k] for x in outputs]).mean()
                            for k in outputs[0].keys() if k != 'log'}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _post_process_forwarded_batch(self, outputs, reference_dials, predictions, ground_truths, inv_vocab):
        for i, uo in enumerate(outputs):
            ud_reference = reference_dials[i].transpose(1, 0)[:uo.shape[0], :uo.shape[1]].cpu().numpy()
            uo = torch.argmax(uo, dim=2).cpu().numpy()
            predictions.append([list(map(lambda x: inv_vocab[x], row))[0] for row in uo])
            ground_truths.append([list(map(lambda x: inv_vocab[x], row))[0] for row in ud_reference])

    def predict(self, batch, inv_vocab=None):
        # user_dials, system_dials, user_lens, system_lens, dial_lens = batch
        step_output = self.forward(*batch)
        if inv_vocab is None:
            inv_vocab = self.embeddings.id2w
        all_user_predictions, all_user_gt = [], []
        all_system_predictions, all_system_gt = [], []
        all_usr_nlu_predictions, all_usr_nlu_gt = [], []
        all_sys_nlu_predictions, all_sys_nlu_gt = [], []
        self._post_process_forwarded_batch(step_output.user_outputs,
                                           step_output.user_dials,
                                           all_user_predictions,
                                           all_user_gt,
                                           inv_vocab)

        self._post_process_forwarded_batch(step_output.usr_nlu_outputs,
                                           step_output.usr_nlu_dials,
                                           all_usr_nlu_predictions,
                                           all_usr_nlu_gt,
                                           inv_vocab)

        self._post_process_forwarded_batch(step_output.sys_nlu_outputs,
                                           step_output.sys_nlu_dials,
                                           all_sys_nlu_predictions,
                                           all_sys_nlu_gt,
                                           inv_vocab)

        self._post_process_forwarded_batch(step_output.system_outputs,
                                           step_output.system_dials,
                                           all_system_predictions,
                                           all_system_gt,
                                           inv_vocab)

        all_samples = torch.argmax(step_output.z_samples, dim=2).cpu().numpy()
        all_p_samples_matrix = torch.argmax(step_output.p_z_samples_matrix, dim=2).cpu().numpy()
        all_q_samples_matrix = torch.argmax(step_output.q_z_samples_matrix, dim=2).cpu().numpy()
        all_user_samples_matrix = torch.argmax(step_output.user_z_samples_matrix, dim=2).cpu().numpy()
        return PredictedOuputs(all_user_predictions,
                               all_user_gt,
                               all_system_predictions,
                               all_system_gt,
                               all_usr_nlu_predictions,
                               all_usr_nlu_gt,
                               all_sys_nlu_predictions,
                               all_sys_nlu_gt,
                               all_samples,
                               all_p_samples_matrix,
                               all_q_samples_matrix,
                               all_user_samples_matrix,
                               step_output.db_data)

    def configure_optimizers(self):
        prior_parameters = []
        supervised_parameters = []
        for z_net in self.vae_cell.system_z_nets:
            prior_parameters.extend(list(z_net.prior_net.parameters()))
            prior_parameters.extend(list(z_net.prior_projection.parameters()))

        for p in self.parameters():
            found = False
            for p2 in prior_parameters:
                if p.shape == p2.shape and torch.equal(p, p2):
                    found = True
            if not found:
                supervised_parameters.append(p)

        # prior_parameters.extend(self.vae_cell.system_dec.parameters())
        all_parameters = self.parameters()
        optimizer_network = torch.optim.Adam(supervised_parameters, lr=1e-3, betas=(0.9, 0.999),
                                             eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer_prior = torch.optim.Adam(prior_parameters, lr=1e-3, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler =\
            torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_network, gamma=self.config['lr_decay_rate'])

        opts = [optimizer_network]
        if not self.config['fake_prior']:
            opts.append(optimizer_prior)
        return opts

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        if optimizer_i == 0 and not self.config['retraining'] and \
                self.epoch_number % self.config['kl_to_ce'] == 0:
            optimizer.step()
            optimizer.zero_grad()
        #
        if optimizer_i == 1 and not self.config['fake_prior']:
            optimizer.step()
            optimizer.zero_grad()

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return [self._valid_loader]

    def test_dataloader(self):
        return [self._test_loader]


class EpochEndCb(pl.Callback):

    def on_epoch_end(self, trainer, model):
        model.epoch_number += 1
        model.vae_cell.epoch_number += 1
        init_tau = model.config['init_gumbel_softmax_tmp']
        gumbel_tmp_step = np.exp(np.log(0.001 / init_tau) / model.config['min_epochs'])
        model.config['gumbel_softmax_tmp'] *= \
            gumbel_tmp_step ** min(model.epoch_number, model.config['min_epochs'])
        # model.lr_scheduler.step()


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
                 usr_nlu_dials=None,
                 usr_nlu_outputs=None,
                 usr_nlu_lens=None,
                 sys_nlu_dials=None,
                 sys_nlu_outputs=None,
                 sys_nlu_lens=None,
                 user_q_zs=None,
                 user_p_zs=None,
                 system_q_zs=None,
                 system_p_zs=None,
                 z_samples=None,
                 bow_logits=None,
                 p_z_samples_matrix=None,
                 q_z_samples_matrix=None,
                 user_z_samples_matrix=None,
                 db_data=None):
        self.user_dials = user_dials
        self.user_outputs = user_outputs
        self.user_lens = user_lens
        self.system_dials = system_dials
        self.system_outputs = system_outputs
        self.system_lens = system_lens
        self.usr_nlu_dials = usr_nlu_dials
        self.usr_nlu_outputs = usr_nlu_outputs
        self.usr_nlu_lens = usr_nlu_lens
        self.sys_nlu_dials = sys_nlu_dials
        self.sys_nlu_outputs = sys_nlu_outputs
        self.sys_nlu_lens = sys_nlu_lens
        self.user_q_zs = user_q_zs
        self.user_p_zs = user_p_zs
        self.system_q_zs = system_q_zs
        self.system_p_zs = system_p_zs
        self.z_samples = z_samples
        self.bow_logits = bow_logits
        self.db_data = db_data
        self.p_z_samples_matrix = p_z_samples_matrix
        self.q_z_samples_matrix = q_z_samples_matrix
        self.user_z_samples_matrix = user_z_samples_matrix


class PredictedOuputs:
    def __init__(self,
                 all_user_predictions=None,
                 all_user_gt=None,
                 all_system_predictions=None,
                 all_system_gt=None,
                 all_usr_nlu_predictions=None,
                 all_usr_nlu_gt=None,
                 all_sys_nlu_predictions=None,
                 all_sys_nlu_gt=None,
                 all_z_samples=None,
                 all_p_z_samples_matrix=None,
                 all_q_z_samples_matrix=None,
                 all_user_z_samples_matrix=None,
                 db_data=None):
        self.all_user_predictions = all_user_predictions
        self.all_user_gt = all_user_gt
        self.all_system_predictions = all_system_predictions
        self.all_usr_nlu_predictions = all_usr_nlu_predictions
        self.all_usr_nlu_gt = all_usr_nlu_gt
        self.all_sys_nlu_predictions = all_sys_nlu_predictions
        self.all_sys_nlu_gt = all_sys_nlu_gt
        self.all_system_gt = all_system_gt
        self.all_z_samples = all_z_samples
        self.all_p_z_samples_matrix = all_p_z_samples_matrix
        self.all_q_z_samples_matrix = all_q_z_samples_matrix
        self.all_user_z_samples_matrix = all_user_z_samples_matrix
        self.db_data = db_data
