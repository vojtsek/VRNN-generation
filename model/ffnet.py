import torch
import torch.nn.functional as F


class FFNet(torch.nn.Module):

    def __init__(self, input_size: int, layer_sizes: list, drop_prob: float = 1.0):
        super(FFNet, self).__init__()
        in_size = input_size
        self.layers = []
        for ls in layer_sizes:
            self.layers.append(torch.nn.Linear(in_size, ls))
            in_size = ls
    #     todo: activation f?
        self.dropout = torch.nn.Dropout(drop_prob)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.dropout(x)
