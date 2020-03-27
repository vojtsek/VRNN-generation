import torch
import torch.nn.functional as F

from . import FFNet
from ..utils import gumbel_softmax_sample, normal_sample


class ZNet(torch.nn.Module):

    def __init__(self, config):
        super(ZNet, self).__init__()
        self.config = config
        self.posterior_net1 = FFNet(config['input_encoder_hidden_size'] *
                                    2 * (1 + int(config['bidirectional_encoder'])) +
                                    config['vrnn_hidden_size'],
                                    config['posterior_ff_sizes1'],
                                    drop_prob=config['drop_prob'])
        if self.config['z_type'] == 'gumbel':
            self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'])
        else:
            self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'] * 2)
        self.posterior_net2 = FFNet(config['z_logits_dim'],
                                    config['posterior_ff_sizes2'],
                                    drop_prob=config['drop_prob'])

    def forward(self, x):
        z_posterior_logits = self.posterior_net1(x)
        z_posterior_logits = self.posterior_projection(z_posterior_logits)
        if self.config['z_type'] == 'gumbel':
            q_z = F.softmax(z_posterior_logits)
            log_q_z = torch.log(q_z + 1e-20)
            z_samples, z_samples_logits = gumbel_softmax_sample(z_posterior_logits, self.config['gumbel_softmax_tmp'])
        else:
            mu = z_posterior_logits[:, :self.config['z_logits_dim']]
            logvar = z_posterior_logits[:, self.config['z_logits_dim']:]
            q_z = torch.sigmoid(z_posterior_logits)
            z_samples = normal_sample(mu, logvar)
        z_posterior_projection = self.posterior_net2(z_samples)
        return z_posterior_projection, q_z, z_samples
