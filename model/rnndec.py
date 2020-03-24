import torch
import torch.nn.functional as F


class RNNDecoder(torch.nn.Module):

    def __init__(self, embedding_dim, init_hidden_size,
                 hidden_size, encoder_size, vocab_size, drop_prob=0.0):
        super(RNNDecoder, self).__init__()
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_size, dropout=drop_prob)
        self.bridge = torch.nn.Linear(init_hidden_size, hidden_size)
        self.dropout_layer = torch.nn.Dropout(drop_prob)
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)

    def forward_step(self, prev_embed, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism

        # update rnn hidden state
        output, hidden = self.rnn(prev_embed, hidden)

        # pre_output = torch.cat([prev_embed, output], dim=2)
        projected_output = self.dropout_layer(output)
        projected_output = self.output_projection(projected_output)

        return output, hidden, projected_output

    def forward(self, trg_embed, init_hidden, max_len=None):
        """Unroll the decoder one step at a time."""

        hidden = self.get_init_hidden(init_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        outputs = []
        projected_outputs = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[i, :].unsqueeze(0)
            output, hidden, projected_output = self.forward_step(prev_embed, hidden)
            outputs.append(output)
            projected_outputs.append(projected_output)

        outputs = torch.cat(outputs, dim=0)
        projected_outputs = torch.cat(projected_outputs, dim=0)
        return outputs, hidden, projected_outputs  # [B, N, D]

    def get_init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros
        bridged = torch.tanh(self.bridge(encoder_final))
        return (bridged.unsqueeze(0), bridged.unsqueeze(0))
