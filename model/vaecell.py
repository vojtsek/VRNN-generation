import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import FFNet, RNNDecoder
from ..utils import zero_hidden, gumbel_softmax_sample


class VAECell(torch.nn.Module):

    def __init__(self, embeddings, vrnn_cell, config):
        super(VAECell, self).__init__()
        self.config = config
        self.embeddings = embeddings
        embedding_dim = embeddings.embedding_dim
        self.vrnn_cell = vrnn_cell
        self.embedding_encoder = torch.nn.LSTM(embedding_dim,
                                               config['input_encoder_hidden_size'],
                                               bidirectional=config['bidirectional_encoder'])

        self.posterior_net1 = FFNet(config['input_encoder_hidden_size'] *
                                    2 * (1 + int(config['bidirectional_encoder'])) +
                                    config['vrnn_hidden_size'],
                                    config['posterior_ff_sizes1'],
                                    config['drop_prob'])
        self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'])
        self.posterior_net2 = FFNet(config['z_logits_dim'], config['posterior_ff_sizes2'], config['drop_prob'])

        self.user_dec = RNNDecoder(embeddings,
                                   config['posterior_ff_sizes2'][-1] + config['vrnn_hidden_size'],
                                   config['decoder_hidden_size'],
                                   config['teacher_forcing_prob'],
                                   config['drop_prob'])
        self.system_dec = RNNDecoder(embeddings,
                                     config['posterior_ff_sizes2'][-1] + config['vrnn_hidden_size'],
                                     config['decoder_hidden_size'],
                                     config['teacher_forcing_prob'],
                                     config['drop_prob'])

        self.state_cell = torch.nn.LSTMCell(config['posterior_ff_sizes2'][-1] + config['input_encoder_hidden_size'],
                                            config['vrnn_hidden_size'])

        self.prior_net = FFNet(config['z_logits_dim'], config['prior_ff_sizes'], config['drop_prob'])
        self.prior_projection = torch.nn.Linear(config['prior_ff_sizes'][-1], config['z_logits_dim'])

    #     todo: activation f?

    def forward(self,
                user_dials, user_lens,
                system_dials, system_lens,
                vrnn_hidden, z_previous):
        # encode user & system utterances
        user_dials = self.embeddings(user_dials).transpose(1, 0)
        user_dials_packed = pack_padded_sequence(user_dials, user_lens, enforce_sorted=False)
        system_dials = self.embeddings(system_dials).transpose(1, 0)
        system_dials_packed = pack_padded_sequence(system_dials, system_lens, enforce_sorted=False)
        encoder_init_state = zero_hidden((2,
                                          1 + int(self.config['bidirectional_encoder']),
                                          user_dials.shape[1],
                                          self.config['input_encoder_hidden_size']))
        user_encoder_hidden = (encoder_init_state[0], encoder_init_state[1])
        system_encoder_hidden = (encoder_init_state[0], encoder_init_state[1])

        user_encoder_out, user_encoder_hidden = self.embedding_encoder(user_dials_packed,
                                                                       user_encoder_hidden)
        system_encoder_out, system_encoder_hidden = self.embedding_encoder(system_dials_packed,
                                                                           system_encoder_hidden)

        # user_encoder_out, lens = pad_packed_sequence(user_encoder_out)
        # concat [fw, bw]
        last_user_hidden = user_encoder_hidden[0].transpose(1, 0).reshape(user_dials.shape[1], -1)
        last_system_hidden = system_encoder_hidden[0].transpose(1, 0).reshape(system_dials.shape[1], -1)
        input_concatenated = torch.cat([last_user_hidden, last_system_hidden], dim=1)

        # posterior network
        vrnn_hidden_cat_input = torch.cat([vrnn_hidden[0], input_concatenated], dim=1)
        z_posterior_logits = self.posterior_projection(self.posterior_net1(vrnn_hidden_cat_input))
        q_z = F.softmax(z_posterior_logits)
        log_q_z = torch.log(q_z + 1e-20)
        z_samples, z_samples_logits = gumbel_softmax_sample(z_posterior_logits, self.config['gumbel_softmax_tmp'])
        z_posterior_projection = self.posterior_net2(z_samples)

        # prior network
        z_prior_logits = self.prior_net(z_previous)
        p_z = F.softmax(z_prior_logits, dim=-1)
        log_p_z = torch.log(p_z)

        # decoder of user & system utterances
        decoder_init_hidden = torch.cat([vrnn_hidden[0], z_posterior_projection], dim=1)
        outputs, hidden, decoded_user_outputs = self.user_dec(user_dials, decoder_init_hidden, torch.max(user_lens))

        vrnn_input = torch.cat([z_posterior_projection, input_concatenated], dim=1)
        next_vrnn_hidden = self.vrnn_cell(vrnn_input, vrnn_hidden)

        return decoded_user_outputs, next_vrnn_hidden, q_z, p_z
