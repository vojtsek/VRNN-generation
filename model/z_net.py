import torch
import torch.nn.functional as F

from . import FFNet
from ..utils import gumbel_softmax_sample, normal_sample, zero_hidden


class ZNet(torch.nn.Module):

    def __init__(self, config, z_type, z_logits_dim, cond_z_logits_dim):
        super(ZNet, self).__init__()
        self.config = config
        self.z_logits_dim = z_logits_dim
        self.z_type = z_type
        z_input_size = config['input_encoder_hidden_size'] *\
                       (1 + int(config['bidirectional_encoder'])) +\
                       config['vrnn_hidden_size']
        self.posterior_net1 = FFNet(z_input_size,
                                    config['posterior_ff_sizes1'],
                                    drop_prob=config['drop_prob'])
        if self.z_type == 'gumbel':
            # self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'])
            self.posterior_projection = torch.nn.Linear(z_input_size, self.z_logits_dim)
        else:
            # self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'] * 2)
            self.posterior_projection = torch.nn.Linear(z_input_size, self.z_logits_dim * 2)

        self.posterior_net2 = FFNet(self.z_logits_dim,
                                    config['posterior_ff_sizes2'],
                                    drop_prob=config['drop_prob'])

        # self.prior_net = FFNet(z_logits_dim + config['vrnn_hidden_size'],
        if self.config['prior_module_recurrent']:
            self.prior_net = torch.nn.LSTMCell(cond_z_logits_dim, self.config['prior_lstm_size'])
            self.prior_projection = torch.nn.Linear(self.config['prior_lstm_size'], z_logits_dim)
            self._reset_hidden()
        else:
            self.prior_net = FFNet(cond_z_logits_dim,
                                   config['prior_ff_sizes'],
                                   activations=[torch.sigmoid],
                                   drop_prob=config['drop_prob'])
            self.prior_projection = torch.nn.Linear(config['prior_ff_sizes'][-1], z_logits_dim)


    def forward(self, x, z_previous, vrnn_hidden):
        # x = self.posterior_net1(x)
        z_posterior_logits = self.posterior_projection(x)
        z_previous = torch.cat([z_previous,], dim=-1)
        if self.config['prior_module_recurrent']:
            bs = vrnn_hidden.shape[0]
            if bs > self.last_prior_hidden[0].shape[0]:
                self._reset_hidden()
            self.last_prior_hidden =\
                self.prior_net(z_previous, (self.last_prior_hidden[0][:bs], self.last_prior_hidden[1][:bs]))
            z_prior_logits = self.last_prior_hidden[0]
        else:
            z_prior_logits = self.prior_net(z_previous)
        z_prior_logits = self.prior_projection(z_prior_logits)
        # z_prior_logits = self.prior_net(torch.cat([z_previous, vrnn_hidden], dim=-1))
        # z_prior_logits = self.prior_projection(z_prior_logits)
        p_z = F.softmax(z_prior_logits / self.config['gumbel_softmax_tmp'], dim=-1)

        if self.z_type == 'gumbel':
            q_z = F.softmax(z_posterior_logits / self.config['gumbel_softmax_tmp'], dim=-1)
            q_z_samples =\
                gumbel_softmax_sample(z_posterior_logits,
                                      self.config['gumbel_softmax_tmp'],
                                      hard=self.config['gumbel_hard'])
        else:
            mu = z_posterior_logits[:, :self.z_logits_dim]
            logvar = z_posterior_logits[:, self.z_logits_dim:]
            q_z = z_posterior_logits
            q_z_samples = normal_sample(mu, logvar)
        p_z_samples =\
            gumbel_softmax_sample(z_prior_logits,
                                  self.config['gumbel_softmax_tmp'],
                                  hard=self.config['gumbel_hard'])
        # z_posterior_projection = self.posterior_net2(q_z_samples)

        if self.z_type == 'cont':
            q_z = q_z_samples
        else:
            # logp = torch.log(p_z)
            # logq = torch.log(q_z)
            # kl = torch.sum((logq - logp) * q_z, dim=-1)
            print('q', torch.argmax(q_z, dim=-1))
            print('p', torch.argmax(p_z, dim=-1))
            # print('kl', kl)
            # print('meankl', torch.mean(kl))
        if self.config['fake_prior']:
            p_z_samples = q_z_samples
            p_z = q_z

        return q_z_samples, q_z, p_z_samples, p_z

    def _reset_hidden(self):
        zero_h = zero_hidden((self.config['batch_size'], self.lstm_size))
        self.last_prior_hidden = (zero_h, zero_h)