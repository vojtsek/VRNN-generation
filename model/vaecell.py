import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from . import FFNet, RNNDecoder
from .z_net import ZNet
from ..utils import zero_hidden


class VAECell(torch.nn.Module):

    def __init__(self, embeddings, vrnn_cell, config):
        super(VAECell, self).__init__()
        self.config = config
        self.embeddings = embeddings
        embedding_dim = embeddings.embedding_dim
        self.vrnn_cell = vrnn_cell
        self.aggregation_layer = torch.nn.Conv1d(in_channels=config['number_z_vectors'], out_channels=1, kernel_size=1)
        self.embedding_encoder = torch.nn.LSTM(embedding_dim,
                                               config['input_encoder_hidden_size'],
                                               bidirectional=config['bidirectional_encoder'])

        self.z_nets = torch.nn.ModuleList([ZNet(config) for _ in range(config['number_z_vectors'])])
        self.user_dec = RNNDecoder(embeddings,
                                   config['posterior_ff_sizes2'][-1] +
                                   config['vrnn_hidden_size'],
                                   # config['posterior_ff_sizes2'][-1],
                                   config['decoder_hidden_size'],
                                   config['teacher_forcing_prob'],
                                   drop_prob=config['drop_prob'])
        self.system_dec = RNNDecoder(embeddings,
                                     config['posterior_ff_sizes2'][-1] +
                                     config['vrnn_hidden_size'] +
                                     config['decoder_hidden_size'],
                                     # config['posterior_ff_sizes2'][-1],
                                     config['decoder_hidden_size'],
                                     config['teacher_forcing_prob'],
                                     config['drop_prob'])
        self.bow_projection = FFNet(config['posterior_ff_sizes2'][-1] + config['vrnn_hidden_size'],
        # self.bow_projection = FFNet(config['posterior_ff_sizes2'][-1],
                                    [config['bow_layer_size'], embeddings.num_embeddings],
                                    activations=[None, torch.relu],
                                    # activations=None,
                                    drop_prob=config['drop_prob']
                                    )

        self.state_cell = torch.nn.LSTMCell(config['posterior_ff_sizes2'][-1] + config['input_encoder_hidden_size'],
                                            config['vrnn_hidden_size'])

        self.prior_net = FFNet(config['z_logits_dim'], config['prior_ff_sizes'], drop_prob=config['drop_prob'])
        self.prior_projection = torch.nn.Linear(config['prior_ff_sizes'][-1], config['z_logits_dim'])

    #     todo: activation f?

    def forward(self,
                user_dials, user_lens,
                system_dials, system_lens,
                previous_vrnn_hidden, z_previous):
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
        vrnn_hidden_cat_input = torch.cat([previous_vrnn_hidden[0], input_concatenated], dim=1)
        z_projection_lst, q_z_lst, z_samples_lst = zip(*[z_net(vrnn_hidden_cat_input) for z_net in self.z_nets])

        # todo: weighted sum
        z_posterior_projection = self.aggregate(torch.stack(z_projection_lst))
        q_z = self.aggregate(torch.stack(q_z_lst))
        z_samples = self.aggregate(torch.stack(z_samples_lst))

        # prior network
        z_prior_logits = self.prior_net(z_previous)
        p_z = F.softmax(z_prior_logits, dim=-1)
        log_p_z = torch.log(p_z)

        # decoder of user & system utterances
        decoder_init_hidden = torch.cat([previous_vrnn_hidden[0], z_posterior_projection], dim=1)
        # decoder_init_hidden = torch.cat([z_posterior_projection], dim=1)
        outputs, last_user_decoder_hidden, decoded_user_outputs = self.user_dec(
            user_dials, decoder_init_hidden, torch.max(user_lens))
        if self.config['with_bow_loss']:
            bow_logits = self.bow_projection(decoder_init_hidden)
        else:
            bow_logits = None
        system_decoder_init_hidden = torch.cat([decoder_init_hidden, last_user_decoder_hidden[0].squeeze(0)], dim=1)
        outputs, hidden, decoded_system_outputs = self.system_dec(
            system_dials, system_decoder_init_hidden, torch.max(system_lens))

        vrnn_input = torch.cat([z_posterior_projection, input_concatenated], dim=1)
        next_vrnn_hidden = self.vrnn_cell(vrnn_input, previous_vrnn_hidden)

        z_samples_lst = torch.stack(z_samples_lst).transpose(1, 0).transpose(2, 1)

        return decoded_user_outputs, decoded_system_outputs,\
               next_vrnn_hidden, q_z, p_z, z_samples, bow_logits, z_samples_lst

    def aggregate(self, x):
        if x.shape[0] == 1:
            return x.squeeze(0)
        # concatenation
        # return x.transpose(1, 0).reshape(-1, self.config['number_z_vectors'] * x.shape[-1])
        x = x.transpose(1, 0)
        x = torch.sum(x, dim=1)
        return x.squeeze(1)
        # x = x.transpose(1, 0)
        # summed = torch.matmul(self.weights, x)
        # return summed.squeeze(1)
