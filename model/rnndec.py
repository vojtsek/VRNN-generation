import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


class RNNDecoder(torch.nn.Module):

    def __init__(self, embeddings, init_hidden_size,
                 hidden_size, encoder_hidden_size=None, concat_size=None,
                 padding_idx=0, bos_idx=0, teacher_prob=0.7, drop_prob=0.0, use_copy=False, max_len=1000):
        super(RNNDecoder, self).__init__()
        self.embeddings = embeddings
        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.vocab_size = embeddings.num_embeddings
        rnn_input_dim = embeddings.embedding_dim + concat_size if concat_size is not None else embeddings.embedding_dim
        self.concat = concat_size is not None
        if use_copy:
            self.rnn = torch.nn.LSTM(rnn_input_dim + embeddings.embedding_dim, hidden_size, dropout=drop_prob)
        else:
            self.rnn = torch.nn.LSTM(rnn_input_dim, hidden_size, dropout=drop_prob)
        self.bridge = torch.nn.Linear(init_hidden_size, hidden_size)
        self.dropout_layer = torch.nn.Dropout(drop_prob)
        self.hidden_size = hidden_size
        self.output_projection = torch.nn.Linear(hidden_size, self.vocab_size)
        self.encoder_hidden_size = encoder_hidden_size
        self.use_copy = use_copy
        self.max_len = max_len
        if use_copy:
            self.attention_proj = torch.nn.Linear(hidden_size, encoder_hidden_size)
            self.attn_combine = torch.nn.Linear(2 * embeddings.embedding_dim, rnn_input_dim - concat_size)
            self.copy_weights = torch.nn.Linear(encoder_hidden_size, rnn_input_dim)
        else:
            self.copy_weights = None
        self.teacher_prob = teacher_prob

    def forward_step(self,
                     input_tk_idx,
                     hidden,
                     encoder_hidden_states=None,
                     encoder_idxs=None,
                     weighted_attention=None,
                     concat_z=None):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism

        # update rnn hidden state
        input_tk_embed = self.embeddings(input_tk_idx)
        if self.use_copy:
            h = self.attention_proj(hidden[0]).squeeze(0).unsqueeze(-1)
            attn_weights = encoder_hidden_states.transpose(1, 0).contiguous().bmm(h)
            attn_weights = F.softmax(attn_weights).squeeze(-1).unsqueeze(1)
                # self.attention_score(torch.cat([input_tk_embed, hidden[0]], dim=-1)), dim=1) \
                #                .transpose(1, 0)[..., :encoder_hidden_states.shape[0]]
            attn_applied = torch.bmm(attn_weights, encoder_hidden_states.transpose(1, 0).contiguous()).transpose(1, 0)
            rnn_input = torch.cat([input_tk_embed, attn_applied.contiguous()], dim=-1)
            # rnn_input = self.attn_combine(rnn_input)

        else:
            rnn_input = input_tk_embed
        if concat_z is not None:
            rnn_input = torch.cat([rnn_input, concat_z], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)

        # pre_output = torch.cat([input_tk_embed, output], dim=2)
        output_drop = self.dropout_layer(output)
        score_g = self.output_projection(output_drop)
        projected_output = F.log_softmax(score_g, dim=2)
        if self.copy_weights is None:
            return output, hidden, projected_output, None

        return output, hidden, projected_output, None
        assert encoder_hidden_states is not None, 'encoder hidden states need to be given if copy mechanism is there'
        assert encoder_idxs is not None, 'encoder indices need to be given if copy mechanism is there'
        encoder_hidden_states = encoder_hidden_states.transpose(1, 0)
        batch_size = encoder_hidden_states.shape[0]
        state = hidden[0].squeeze()
        # get scores score_c for copy mode
        score_c = F.tanh(
            self.copy_weights(
                encoder_hidden_states.contiguous().view(-1, self.encoder_hidden_size)))
        score_c = score_c.view(batch_size, -1, self.hidden_size)  # [b x seq x hidden_size]
        if state.dim() < 2:
            state = state.unsqueeze(0)
        score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze()  # [b x seq]

        # truncate unnecessary padding
        encoder_idxs = encoder_idxs[..., :encoder_hidden_states.shape[1]]
        encoder_padding_np = np.array(encoder_idxs == self.padding_idx, dtype='float32')
        encoded_mask = torch.from_numpy(encoder_padding_np * (-1000))  # [b x seq]
        score_c = score_c + encoded_mask  # padded parts will get close to 0 when applying softmax
        score_c = F.tanh(score_c)

        score = torch.cat([score_g.squeeze(0), score_c], 1)  # [b x (vocab+seq)]
        probs = F.log_softmax(score)
        prob_g = probs[:, :self.vocab_size]  # [b x vocab]
        prob_c = probs[:, self.vocab_size:]  # [b x seq]
        copy_probs_per_idx = torch.zeros(batch_size, self.vocab_size)
        for b_idx in range(batch_size):  # for each sequence in batch
            for s_idx in range(encoder_idxs.shape[1]):
                copy_probs_per_idx[b_idx, encoder_idxs[b_idx, s_idx]] =\
                    copy_probs_per_idx[b_idx, encoder_idxs[b_idx, s_idx]] + prob_c[b_idx, s_idx]
        out = prob_g + copy_probs_per_idx

        idx_from_input = []
        for i, j in enumerate(encoder_idxs):
            idx_from_input.append([int(k == input_tk_idx[:, i].data[0]) for k in j])
        idx_from_input = torch.from_numpy(np.array(idx_from_input, dtype='float32'))  # [b x seq]
        for i in range(batch_size):
            if idx_from_input.dim() > 0 and idx_from_input[i].sum().item() > 1:
                idx_from_input[i] = idx_from_input[i] / idx_from_input[i].sum().item()
        # 3-2) multiply with prob_c to get final weighted representation
        attn = prob_c * idx_from_input
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1)  # [b x 1 x seq]
        weighted = torch.bmm(attn, encoder_hidden_states).transpose(1, 0)

        return output, hidden, out.unsqueeze(0), weighted

    def forward(self, trg_embed, init_hidden, to_concat=None, max_len=None, encoder_hidden_states=None, encoder_idxs=None):
        """Unroll the decoder one step at a time."""

        hidden = self.get_init_hidden(init_hidden)

        outputs = []
        projected_outputs = []

        next_input_idx = torch.ones(1, init_hidden.shape[0], dtype=torch.int64) * self.bos_idx
        if self.use_copy:
            weighted_attention = torch.zeros(1, init_hidden.shape[0], self.encoder_hidden_size)
        else:
            weighted_attention = None
        for i in range(max_len):
            output, hidden, projected_output, weighted_attention =\
                self.forward_step(next_input_idx,
                                  hidden,
                                  encoder_hidden_states,
                                  encoder_idxs,
                                  weighted_attention,
                                  to_concat.unsqueeze(0) if self.concat else None)
            outputs.append(output)
            last_decoder = torch.argmax(projected_output, dim=2)
            # teacher forcing
            if not self.training or np.random.rand(1) < self.teacher_prob:
                next_input_idx = last_decoder
            else:
                next_input_idx = trg_embed[:, i].unsqueeze(0)

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
