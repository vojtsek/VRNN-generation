import torch
import torch.nn.functional as F
import numpy as np

from .attention import DotProjectionAttention
from ..utils import embed_oh, get_activation

torch.manual_seed(0)
np.random.seed(0)


class RNNDecoder(torch.nn.Module):

    def __init__(self, embeddings, init_hidden_size,
                 hidden_size, encoder_hidden_size=None, concat_size=None, use_attention=False,
                 padding_idx=0, bos_idx=0, teacher_prob=0.7,
                 drop_prob=0.0, use_copy=False, device=torch.device('cpu:0'),
                 activation='tanh'):
        super(RNNDecoder, self).__init__()
        self.embeddings = embeddings.to(device)
        self.device = device
        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.activation = get_activation(activation)
        self.vocab_size = embeddings.num_embeddings
        rnn_input_dim = embeddings.embedding_dim + concat_size if concat_size is not None else embeddings.embedding_dim
        self.concat = concat_size is not None
        if use_attention:
            self.rnn = torch.nn.LSTM(rnn_input_dim + encoder_hidden_size, hidden_size, dropout=drop_prob)
        else:
            self.rnn = torch.nn.LSTM(rnn_input_dim, hidden_size, dropout=drop_prob)
        self.bridge = torch.nn.Linear(init_hidden_size, hidden_size)
        self.dropout_layer = torch.nn.Dropout(drop_prob)
        self.hidden_size = hidden_size
        self.output_projection = torch.nn.Linear(hidden_size, self.vocab_size)
        self.encoder_hidden_size = encoder_hidden_size
        self.use_copy = use_copy
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = DotProjectionAttention(200, encoder_hidden_size, hidden_size)
            if concat_size is not None:
                rnn_input_dim -= concat_size
        if self.use_copy:
            self.copy_weights = torch.nn.Linear(encoder_hidden_size, hidden_size)
        self.teacher_prob = teacher_prob
        self.force = False

    def forward_step(self,
                     input_tk_idx,
                     hidden,
                     encoder_hidden_states=None,
                     encoder_idxs=None,
                     concat_z=None,
                     copy_coeff=0):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism

        # update rnn hidden state
        input_tk_embed = self.embeddings(input_tk_idx)
        if self.use_attention:
            attn_applied = self.attention(hidden[0].squeeze(0), encoder_hidden_states).transpose(1, 0).contiguous()
            rnn_input = torch.cat([input_tk_embed, attn_applied.contiguous()], dim=-1)
            # rnn_input = self.attn_combine(rnn_input)

        else:
            rnn_input = input_tk_embed
        if concat_z is not None:
            rnn_input = torch.cat([rnn_input, concat_z], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)

        # pre_output = torch.cat([input_tk_embed, output], dim=2)
        output_drop = self.dropout_layer(output)
        score_gen = self.output_projection(output_drop)
        output_proba = F.log_softmax(score_gen, dim=2)
        if not self.use_copy:
            return output, hidden, output_proba
        else:
            u_copy_score = self.activation(self.copy_weights(encoder_hidden_states.transpose(0, 1)))  # [B,T,H]
            # stable version of copynet
            u_copy_score = torch.matmul(u_copy_score, output_drop.squeeze(0).unsqueeze(2)).squeeze(2)
            u_copy_score_max, _ = torch.max(u_copy_score, dim=1, keepdim=True)
            u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
            encoder_idxs_emb = embed_oh(encoder_idxs, list(encoder_idxs.shape) + [self.vocab_size ], device=self.device)
            u_copy_score = torch.log(
                torch.bmm(u_copy_score.unsqueeze(1), encoder_idxs_emb).squeeze(1) + 1e-35) + u_copy_score_max  # [B,V]
            scores = F.softmax(torch.cat([score_gen.squeeze(0), u_copy_score], dim=1), dim=1)
            score_gen, u_copy_score = scores[:, :self.vocab_size], \
                                      scores[:, self.vocab_size:]
            output_proba = score_gen + copy_coeff * u_copy_score[:, :self.vocab_size]  # [B,V]
            del encoder_idxs_emb
            # torch.cuda().empty_cache()

            # gen = torch.argmax(score_gen, dim=-1).numpy()
            # cpy = torch.argmax(output_proba, dim=-1).numpy()
            # if any([g != c for g, c in zip(gen, cpy)]):
            #     print('COPY WINS', gen, cpy, torch.argmax(u_copy_score[:, :self.vocab_size], dim=-1).numpy())
            # else:
            #     print('GEN WINS')

        return output, hidden, torch.log(output_proba).unsqueeze(0)

    def forward(self,
                trg_embed,
                init_hidden,
                to_concat=None,
                max_len=None,
                encoder_hidden_states=None,
                encoder_idxs=None,
                copy_coeff=0):
        """Unroll the decoder one step at a time."""

        hidden = self.get_init_hidden(init_hidden)
        del_hidden = hidden


        outputs = []
        projected_outputs = []

        next_input_idx = torch.ones(1, init_hidden.shape[0], dtype=torch.int64) * self.bos_idx
        next_input_idx = next_input_idx.to(self.device)
        next_input_idx_to_del = next_input_idx
        for i in range(max_len):
            output, hidden, projected_output =\
                self.forward_step(next_input_idx,
                                  hidden,
                                  encoder_hidden_states,
                                  encoder_idxs,
                                  to_concat.unsqueeze(0) if self.concat else None,
                                  copy_coeff)
            outputs.append(output)
            last_decoder = torch.argmax(projected_output, dim=2)
            if i >= trg_embed.shape[1]:
                break
            # teacher forcing
            if not self.force and (not self.training or np.random.rand(1) < self.teacher_prob):
                next_input_idx = last_decoder
            else:
                next_input_idx = trg_embed[:, i].unsqueeze(0)

            projected_outputs.append(projected_output)

        outputs = torch.cat(outputs, dim=0)
        del next_input_idx_to_del, del_hidden
        # torch.cuda.empty_cache()
        projected_outputs = torch.cat(projected_outputs, dim=0)
        return outputs, hidden, projected_outputs  # [B, N, D]

    def get_init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros
        bridged = self.activation(self.bridge(encoder_final))
        return (bridged.unsqueeze(0), bridged.unsqueeze(0))
