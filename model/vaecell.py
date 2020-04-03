import torch
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

        self.user_z_nets = torch.nn.ModuleList([ZNet(config, config['user_z_type']) for _ in range(config['number_z_vectors'])])
        self.system_z_nets = torch.nn.ModuleList([ZNet(config, config['system_z_type']) for _ in range(config['number_z_vectors'])])
        self.user_dec = RNNDecoder(embeddings,
                                   config['z_logits_dim'] +
                                   # config['input_encoder_hidden_size'] *
                                   #  (1 + int(config['bidirectional_encoder'])) +
                                   config['vrnn_hidden_size'],
                                   config['decoder_hidden_size'],
                                   config['teacher_forcing_prob'],
                                   drop_prob=config['drop_prob'])
        self.nlu_dec = RNNDecoder(embeddings,
                                  config['z_logits_dim'] +
                                  # config['input_encoder_hidden_size'] *
                                  # (1 + int(config['bidirectional_encoder'])) +
                                  config['vrnn_hidden_size'],
                                  config['decoder_hidden_size'],
                                  config['teacher_forcing_prob'],
                                  drop_prob=config['drop_prob'])

        self.system_dec = RNNDecoder(embeddings,
                                     config['z_logits_dim'] * 2 +
                                     # config['z_logits_dim'] +
                                     # config['input_encoder_hidden_size'] *
                                     # (1 + int(config['bidirectional_encoder'])) +
                                     config['vrnn_hidden_size'],
                                     config['decoder_hidden_size'],
                                     config['teacher_forcing_prob'],
                                     config['drop_prob'])
        self.bow_projection = FFNet(config['z_logits_dim'] + config['vrnn_hidden_size'],
                                    [config['bow_layer_size'], embeddings.num_embeddings],
                                    activations=[None, torch.relu],
                                    # activations=None,
                                    drop_prob=config['drop_prob']
                                    )

        self.state_cell = torch.nn.LSTMCell(config['z_logits_dim'] + config['input_encoder_hidden_size'],
                                            config['vrnn_hidden_size'])

    #     todo: activation f?

    def _z_module(self, dials, lens, encoder_init_state, previous_vrnn_hidden,
                  z_nets, decoders, z_previous, prev_z_posterior_projection=None):
        # lens[0]: actual turns, lens[1:] possible further supervision; decoder corresponding list
        dials = self.embeddings(dials).transpose(1, 0)
        dials_packed = pack_padded_sequence(dials, lens[0], enforce_sorted=False)
        encoder_hidden = (encoder_init_state[0], encoder_init_state[1])
        encoder_outs, last_encoder_hidden = self.embedding_encoder(dials_packed, encoder_hidden)
        # concat fw+bw
        last_hidden = last_encoder_hidden[0].transpose(1, 0).reshape(dials.shape[1], -1)
        vrnn_hidden_cat_input = torch.cat([previous_vrnn_hidden[0], last_hidden], dim=1)
        z_projection_lst, q_z_lst, posterior_z_samples_lst, p_z_lst =\
            zip(*[z_net(vrnn_hidden_cat_input, z_previous) for z_net in z_nets])
        z_posterior_projection = self.aggregate(torch.stack(z_projection_lst))
        q_z = self.aggregate(torch.stack(q_z_lst))
        p_z = self.aggregate(torch.stack(p_z_lst))
        z_samples = self.aggregate(torch.stack(posterior_z_samples_lst))
        posterior_z_samples_lst = torch.stack(posterior_z_samples_lst).transpose(1, 0).transpose(2, 1)

        if prev_z_posterior_projection is not None:
            decoder_init_hidden = torch.cat(
                [previous_vrnn_hidden[0], z_posterior_projection, prev_z_posterior_projection], dim=1)
                # [previous_vrnn_hidden[0], last_hidden, prev_z_posterior_projection], dim=1)
        else:
            decoder_init_hidden = torch.cat(
                [previous_vrnn_hidden[0], z_posterior_projection], dim=1)
                # [previous_vrnn_hidden[0], last_hidden], dim=1)
        all_decoded_outputs = []
        for i, decoder in enumerate(decoders):
            outputs, last_decoder_hidden, decoded_outputs =\
                decoder(dials, decoder_init_hidden, torch.max(lens[i]))
            all_decoded_outputs.append(decoded_outputs)

        if self.config['with_bow_loss']:
            bow_logits = self.bow_projection(decoder_init_hidden)
        else:
            bow_logits = None

        return TurnOutput(decoded_outputs=all_decoded_outputs,
                          q_z=q_z,
                          p_z=p_z,
                          z_samples=z_samples,
                          z_samples_lst=posterior_z_samples_lst,
                          bow_logits=bow_logits,
                          z_posterior_projection=z_posterior_projection,
                          last_encoder_hidden=last_hidden)

    def forward(self,
                user_dials, user_lens, nlu_lens,
                system_dials, system_lens,
                previous_vrnn_hidden, z_previous):
        # encode user & system utterances
        encoder_init_state = zero_hidden((2,
                                          1 + int(self.config['bidirectional_encoder']),
                                          user_dials.shape[1],
                                          self.config['input_encoder_hidden_size']))

        user_turn_output = self._z_module(user_dials ,
                                          [user_lens, nlu_lens],
                                          encoder_init_state,
                                          previous_vrnn_hidden,
                                          self.user_z_nets,
                                          [self.user_dec, self.nlu_dec],
                                          z_previous)
        system_turn_output = self._z_module(system_dials,
                                            [system_lens],
                                            encoder_init_state,
                                            previous_vrnn_hidden,
                                            self.system_z_nets,
                                            [self.system_dec],
                                            z_previous,
                                            prev_z_posterior_projection=user_turn_output.z_posterior_projection)


        z_posterior_projection = torch.cat([user_turn_output.z_posterior_projection,
                                            system_turn_output.z_posterior_projection], dim=1)
        encoded_inputs = torch.cat([user_turn_output.last_encoder_hidden,
                                    system_turn_output.last_encoder_hidden], dim=1)
        vrnn_input = torch.cat([z_posterior_projection, encoded_inputs], dim=1)
        next_vrnn_hidden = self.vrnn_cell(vrnn_input, previous_vrnn_hidden)

        return VAECellOutput(next_vrnn_hidden=next_vrnn_hidden,
                             user_turn_output=user_turn_output,
                             system_turn_output=system_turn_output)

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


class TurnOutput:
    def __init__(self,
                 decoded_outputs=None,
                 q_z=None,
                 p_z=None,
                 z_samples=None,
                 z_samples_lst=None,
                 bow_logits=None,
                 z_posterior_projection=None,
                 last_encoder_hidden=None):
        self.decoded_outputs = decoded_outputs
        self.q_z = q_z
        self.p_z = p_z
        self.z_samples = z_samples
        self.z_samples_lst = z_samples_lst
        self.bow_logits = bow_logits
        self.z_posterior_projection = z_posterior_projection
        self.last_encoder_hidden = last_encoder_hidden


class VAECellOutput:
    def __init__(self,
                 next_vrnn_hidden=None,
                 user_turn_output: TurnOutput=None,
                 system_turn_output: TurnOutput=None):
        self.next_vrnn_hidden = next_vrnn_hidden
        self.user_turn_output = user_turn_output
        self.system_turn_output = system_turn_output
