import torch
import torch.nn.functional as F


class RNNDecoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNNDecoder, self).__init__()
        self.cell = torch.nn.LSTMCell(input_size, hidden_size)

    def forward(self, x):
        return x
