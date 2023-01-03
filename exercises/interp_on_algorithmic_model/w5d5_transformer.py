import torch as t
from torch import nn
from einops import rearrange
from fancy_einsum import einsum
from typing import Optional
import math
from collections import OrderedDict
from typing import Optional
import torch
from torch import Tensor, nn
from dataclasses import dataclass

@dataclass
class TransformerConfig:

    vocab_size: int = 28996
    intermediate_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_size: int = 64
    max_position_embeddings: int = 512
    dropout: float = 0.1
    type_vocab_size: int = 2
    layer_norm_epsilon: float = 1e-12

class MultiheadAttention(nn.Module):
    W_Q: nn.Linear
    W_K: nn.Linear
    W_V: nn.Linear
    W_O: nn.Linear

    def __init__(self, config: TransformerConfig):
        """
        Adding option to override head_size (defaults to hidden_size / num_heads otherwise)
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        
        self.W_Q = nn.Linear(self.hidden_size, self.num_heads*self.head_size)
        self.W_K = nn.Linear(self.hidden_size, self.num_heads*self.head_size)
        self.W_V = nn.Linear(self.hidden_size, self.num_heads*self.head_size)
        self.W_O = nn.Linear(self.num_heads*self.head_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        attention_scores = self.attention_pattern_pre_softmax(x)
        if additive_attention_mask is not None:
            attention_scores = attention_scores + additive_attention_mask

        attention_probabilities = attention_scores.softmax(dim=-1)

        v = rearrange(self.W_V(x), "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=self.num_heads)
        attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

        attention_values = rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")

        output = self.W_O(attention_values)

        return self.dropout(output)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:

        q = rearrange(self.W_Q(x), "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=self.num_heads)
        k = rearrange(self.W_K(x), "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=self.num_heads)

        return einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (q.shape[-1] ** 0.5)


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.batch_first = batch_first
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[None, : x.size(1), :]
        else:
            x = x + self.pe[: x.size(0), None, :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_hid):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = MultiheadAttention(TransformerConfig(hidden_size=d_model, num_heads=nhead, head_size=d_model // nhead))

        self.linear1 = nn.Linear(d_model, d_hid)
        self.linear2 = nn.Linear(d_hid, d_model)
        self.activation = nn.ReLU()

    def forward(self, x, padding_mask):
        x = x + self.attn(x, padding_mask)
        x = x + self.mlp(x)
        return x

    def attn(self, x, padding_mask):
        x = self.norm1(x)
        additive_mask = torch.where(padding_mask, -10000, 0)[:, None, None, :]  # [batch, head=1, qpos=1, kpos]
        # print(additive_mask)
        x = self.self_attn(x, additive_attention_mask=additive_mask)
        return x

    def mlp(self, x):
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class ParenTransformer(nn.Module):
    def __init__(
        self, ntoken: int, nclasses: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.0
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, d_hid) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers

        self.decoder = nn.Linear(d_model, nclasses)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        padding_mask = x == SimpleTokenizer.PAD_TOKEN
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for l in self.layers:
            x = l(x, padding_mask)
        x = self.norm(x)
        x = self.decoder(x)
        return self.softmax(x[:, 0, :])

    def load_simple_transformer_state_dict(self, state_dict):
        new_dict = OrderedDict()
        for key, weight in state_dict.items():
            key = key.replace("transformer_encoder.", "").replace("out_proj", "W_O")
            if "in_proj_" in key:
                q, k, v = torch.tensor_split(weight, 3)
                # maps both in_proj_weight -> W_Q.weight and in_proj_bias -> W_Q.bias
                new_dict[key.replace("in_proj_", "W_Q.")] = q
                new_dict[key.replace("in_proj_", "W_K.")] = k
                new_dict[key.replace("in_proj_", "W_V.")] = v
            else:
                if key == "pos_encoder.pe":
                    weight = weight[:, 0, :]  # remove extra dimension from posencoder due to earlier architechture
                new_dict[key] = weight
        self.load_state_dict(new_dict)


class SimpleTokenizer:
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # the 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    def tokenize(self, strs: list[str], max_len = None) -> torch.Tensor:
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if isinstance(strs, str):
            strs = [strs]

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        ints = [
            [self.START_TOKEN] + [c_to_int(c) for c in s] + [self.END_TOKEN] + [self.PAD_TOKEN] * (max_len - len(s))
            for s in strs
        ]
        return torch.tensor(ints)

    def decode(self, tokens) -> list[str]:
        assert tokens.ndim >= 2, "Need to have a batch dimension"
        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        return [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN)
            for seq in tokens
        ]