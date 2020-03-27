import torch
import torch.nn.functional as F

from . import FFNet
from ..utils import gumbel_softmax_sample


class ZNet(torch.nn.Module):

    def __init__(self, config):
        super(ZNet, self).__init__()
        self.config = config
        self.posterior_net1 = FFNet(config['input_encoder_hidden_size'] *
                                    2 * (1 + int(config['bidirectional_encoder'])) +
                                    config['vrnn_hidden_size'],
                                    config['posterior_ff_sizes1'],
                                    drop_prob=config['drop_prob'])
        self.posterior_projection = torch.nn.Linear(config['posterior_ff_sizes1'][-1], config['z_logits_dim'])
        self.posterior_net2 = FFNet(config['z_logits_dim'],
                                    config['posterior_ff_sizes2'],
                                    drop_prob=config['drop_prob'])

    def forward(self, x):
        z_posterior_logits = self.posterior_projection(self.posterior_net1(x))
        q_z = F.softmax(z_posterior_logits)
        log_q_z = torch.log(q_z + 1e-20)
        z_samples, z_samples_logits = gumbel_softmax_sample(z_posterior_logits, self.config['gumbel_softmax_tmp'])
        z_posterior_projection = self.posterior_net2(z_samples)
        return z_posterior_projection, q_z, z_samples
