import torch
import torch.nn.functional as F

from . import FFNet, RNNDecoder


class VAECell(torch.nn.Module):

    def __init__(self, embeddings, config):
        super(VAECell, self).__init__()
        self.config = config
        self.embeddings = embeddings
        embedding_dim = embeddings.embedding_dim
        self.embedding_encoder = torch.nn.LSTM(embedding_dim, config['input_encoder_hidden_size'])

        self.posterior_net1 = FFNet(config['input_encoder_hidden_size'] + config['vrnn_hidden_size'],
                                   config['posterior_ff_sizes1'],
                                   config['drop_prob'])
        self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'])
        self.posterior_net2 = FFNet(config['z_logits_dim'], config['posterior_ff_sizes2'], config['drop_prob'])

        self.user_dec = RNNDecoder(embedding_dim,
                                   config['posterior_ff_sizes2'][-1] + config['vrnn_hidden_size'])
        self.system_dec = RNNDecoder(embedding_dim,
                                   config['posterior_ff_sizes2'][-1] + config['vrnn_hidden_size'])

        self.state_cell = torch.nn.LSTMCell(config['posterior_ff_sizes2'][-1] + config['input_encoder_hidden_size'],
                                            config['vrnn_hidden_size'])

        self.prior_net = FFNet(config['z_logits_dim'], config['prior_ff_sizes'], config['drop_prob'])
        self.prior_projection = torch.nn.Linear(config['prior_ff_sizes'][-1], config['z_logits_dim'])

    #     todo: activation f?

    def forward(self,
                user_dials, user_lens,
                system_dials, system_lens,
                vrnn_hidden, z_previous):
        # b x d x w
        user = self.embeddings(user_dials)
        print(user.shape, vrnn_hidden[0].shape, z_previous.shape)
        return vrnn_hidden[0], vrnn_hidden, z_previous
