import torch
torch.manual_seed(0)


class DotProjectionAttention(torch.nn.Module):

    def __init__(self, attn_size, enc_hidden_size, dec_hidden_size):
        super(DotProjectionAttention, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.attn_size = attn_size
        self.attn_w = torch.nn.Linear(self.enc_hidden_size + self.dec_hidden_size, self.attn_size)
        self.attn_v = torch.nn.Linear(self.attn_size, 1)

    def forward(self, hidden, vectors_to_attend):
        # vectors_to_attend N X B X H
        # hidden B X H
        max_len = vectors_to_attend.shape[0]
        vectors_to_attend = vectors_to_attend.transpose(0, 1)
        hidden = hidden.unsqueeze(1)

        # B x N x H
        hidden_expanded = hidden.expand(-1, max_len, -1)
        # B x 2N x 2H
        concated_hiddens = torch.cat([vectors_to_attend, hidden_expanded], dim=2)
        energies = torch.tanh(self.attn_w(concated_hiddens))
        energies = self.attn_v(energies)
        weights = torch.nn.functional.softmax(energies, dim=1).transpose(1, 2)
        context = torch.bmm(weights, vectors_to_attend)
        return context
