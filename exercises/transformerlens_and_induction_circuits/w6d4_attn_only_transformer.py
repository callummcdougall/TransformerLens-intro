import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# from utils import StaticModuleList

# Define network architecture

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["d_vocab"]))
        nn.init.kaiming_uniform_(self.W_E, a=np.sqrt(5))

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return einops.rearrange(self.W_E[:, tokens], "d_model batch pos -> batch pos d_model")


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg["d_vocab"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_U, a=np.sqrt(5))

    def forward(self, tokens):
        return torch.einsum("vm,bpm->bpv", self.W_U, tokens)  # [batch, pos, d_vocab]


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["n_ctx"]))
        nn.init.kaiming_uniform_(self.W_pos, a=np.sqrt(5))

    def forward(self, x):
        # Output shape [pos, d_model] - will be broadcast along batch dim
        return self.W_pos[:, : x.size(-1)].T  # [pos, d_model]


# Attention
class Attention(nn.Module):
    def __init__(self, cfg, attn_type="global"):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_Q, a=np.sqrt(5))
        self.W_K = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_K, a=np.sqrt(5))
        self.W_V = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_V, a=np.sqrt(5))
        self.W_O = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_model"], self.cfg["d_head"]))
        nn.init.kaiming_uniform_(self.W_O, a=np.sqrt(5))

        self.attn_type = attn_type
        # Create a query_pos x key_pos mask, with True iff that query position
        # can attend to that key position
        causal_mask = torch.tril(torch.ones((self.cfg["n_ctx"], self.cfg["n_ctx"])).bool())
        self.register_buffer("mask", causal_mask)

        self.register_buffer("IGNORE", torch.tensor(-1e5))
        self.attn_scale = np.sqrt(self.cfg["d_head"])

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_qk_input = HookPoint()  # [batch, pos, d_model] - The residual stream + positional embeddings
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]

    def forward(self, x, pos_embed):
        # We add in the positional embeddings to the residual stream to create qk_input
        # This is the input to JUST the keys and querys, not the values
        qk_input = self.hook_qk_input(x + pos_embed)  # [batch, pos, d_model]
        q = self.hook_q(torch.einsum("ihm,bpm->bpih", self.W_Q, qk_input))  # [batch, pos, head_index, d_head]
        k = self.hook_k(torch.einsum("ihm,bpm->bpih", self.W_K, qk_input))  # [batch, pos, head_index, d_head]
        v = self.hook_v(torch.einsum("ihm,bpm->bpih", self.W_V, x))  # [batch, pos, head_index, d_head]
        attn_scores = torch.einsum("bpih,bqih->bipq", q, k) / self.attn_scale  # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(
            self.apply_causal_mask(attn_scores)
        )  # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(F.softmax(attn_scores, dim=-1))  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(torch.einsum("bpih,biqp->bqih", v, attn_matrix))  # [batch, pos, head_index, d_head]

        if self.cfg["use_attn_result"]:
            result = self.hook_result(torch.einsum("imh,bqih->bqim", self.W_O, z))  # [batch, pos, head_index, d_model]
            out = einops.reduce(
                result, "batch position index model->batch position model", "sum"
            )  # [batch, pos, d_model]
        else:
            out = torch.einsum("imh,bqih->bqm", self.W_O, z)  # [batch, pos, head_index, d_model]
        return out

    def apply_causal_mask(self, attn_scores):
        return torch.where(self.mask[: attn_scores.size(-2), : attn_scores.size(-1)], attn_scores, self.IGNORE)  # type: ignore


# Transformer Block
class AttnOnlyBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        # Note that resid_pre of layer k+1 is resid_post of layer k - given for convenience
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, x, pos_embed):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        attn_out = self.hook_attn_out(self.attn(x, pos_embed))  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_pre + attn_out)  # [batch, pos, d_model]
        return resid_post


# Full transformer
class AttnOnlyTransformer(HookedRootModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d_model]

        self.blocks: nn.ModuleList[AttnOnlyBlock] = nn.ModuleList([AttnOnlyBlock(self.cfg, block_index) for block_index in range(self.cfg["n_layers"])])
        self.unembed = Unembed(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup_hooks()

    def forward(self, tokens):
        # Input x is either a batch of tokens ([batch, pos]) or a text string
        if type(tokens) == str:
            # If text, convert to tokens (batch_size=1)
            tokens = self.to_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens))  # [batch, pos, d_model]
        # We do NOT add positional embeddings into the residual stream
        # Instead they are added into the input of just the query and
        # key matrices, and not the values or unembed.
        residual = embed  # [batch, pos, d_model]
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual, pos_embed)  # [batch, pos, d_model]
        logits = self.unembed(residual)  # [batch, pos, d_vocab]
        return logits

    def to_tokens(self, text):
        return self.tokenizer(
            self.tokenizer.bos_token + text,
            return_tensors="pt",
        )["input_ids"]

    def set_attn_result(self, use_attn_result: bool):
        self.cfg["use_attn_result"] = use_attn_result