import torch
import torch.nn.functional as F


class RNNDecoder(torch.nn.Module):

    def __init__(self, embeddings, init_hidden_size,
                 hidden_size, z_size=None, teacher_prob=0.7, drop_prob=0.0):
        super(RNNDecoder, self).__init__()
        self.embeddings = embeddings
        self.vocab_size = embeddings.num_embeddings
        rnn_input_dim = embeddings.embedding_dim + z_size if z_size is not None else embeddings.embedding_dim
        self.concat_z = z_size is not None
        self.rnn = torch.nn.LSTM(rnn_input_dim, hidden_size, dropout=drop_prob)
        self.bridge = torch.nn.Linear(init_hidden_size, hidden_size)
        self.dropout_layer = torch.nn.Dropout(drop_prob)
        self.output_projection = torch.nn.Linear(hidden_size, self.vocab_size)
        self.teacher_prob = teacher_prob

    def forward_step(self, prev_embed, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism

        # update rnn hidden state
        output, hidden = self.rnn(prev_embed, hidden)

        # pre_output = torch.cat([prev_embed, output], dim=2)
        projected_output = self.dropout_layer(output)
        projected_output = F.log_softmax(self.output_projection(projected_output), dim=2)

        return output, hidden, projected_output

    def forward(self, trg_embed, init_hidden, z=None, max_len=None):
        """Unroll the decoder one step at a time."""

        hidden = self.get_init_hidden(init_hidden)

        outputs = []
        projected_outputs = []

        prev_embed = torch.zeros((1, init_hidden.shape[0], self.embeddings.embedding_dim))
        for i in range(max_len):
            if self.concat_z:
                prev_embed = torch.cat([prev_embed, z.unsqueeze(0)], dim=-1)
            output, hidden, projected_output = self.forward_step(prev_embed, hidden)
            outputs.append(output)
            last_decoder = self.embeddings(torch.argmax(projected_output, dim=2))
            # teacher forcing
            if not self.training or torch.rand((1,)) > self.teacher_prob:
                prev_embed = last_decoder
            else:
                prev_embed = trg_embed[i, :].unsqueeze(0)

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
