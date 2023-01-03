# %%

# !pip install plotly
# !pip install einops
# !pip install fancy_einsum
# !pip install torchtyping
# !pip install typeguard
# !pip install git+https://github.com/neelnanda-io/TransformerLens.git@new-demo
# !pip install circuitsvis
# !pip3 install --upgrade protobuf==3.20.0

# %%

import os
import sys
import requests
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import w6d4_tests
from dataclasses import dataclass
from fancy_einsum import einsum
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from torchtyping import TensorType as TT
from typeguard import typechecked
from typing import Tuple, List, Dict, Union, Callable, Any, Optional

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
    # return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
    # return px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
    # return px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

# %%

# pio.renderers.default = "notebook"
device = "cuda" if t.cuda.is_available() else "cpu"
MAIN = __name__ == "__main__"

# download weights from https://drive.google.com/u/0/uc?id=19FQ4UQ-5vw8d-duXR9haks5xyKYPzvS8&export=download
# and save them in your directory, with the following path:

# WEIGHT_PATH = "attn_only_2L_half.pth"
WEIGHT_PATH = "attn_only_2L.pth"

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal", # defaults to "bidirectional"
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. use layernorm with weights and biases
    positional_embedding_type="shortformer" # this makes it so positional embeddings are used differently (makes induction heads cleaner to study)
)

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)


if MAIN:
    head_index = 4
    layer = 1
    W_U = model.unembed.W_U
    W_O_all = model.blocks[1].attn.W_O
    W_V_all = model.blocks[1].attn.W_V
    W_E = model.embed.W_E
    OV_circuit = einsum("emb1 voc1, d_k emb1, emb2 d_k, voc2 emb2 -> voc1 voc2", W_U, W_O_all[4], W_V_all[4], W_E)

# %%

def to_numpy(tensor):
    """Helper function to convert things to numpy before plotting with Plotly."""
    return tensor.detach().cpu().numpy()

if MAIN:
    rand_indices = t.randperm(model.cfg.d_vocab)[:200]
    # imshow(to_numpy(OV_circuit[rand_indices][:, rand_indices]))
# %%

# %%

def top_1_acc(OV_circuit):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    argmaxes = OV_circuit.argmax(dim=0)
    diag_indices = t.arange(OV_circuit.shape[0]).to(argmaxes.device)

    return (argmaxes == diag_indices).float().mean()
    

# def top_5_acc(OV_circuit):
#     '''
#     This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
#     '''
#     argmaxes = t.topk(OV_circuit, k=5, dim=0).indices
#     diag_indices = t.arange(OV_circuit.shape[0]).to(argmaxes.device)

#     return (argmaxes == diag_indices).any(dim=0).float().mean()

from tqdm import tqdm
# def get_diff_from_identity(OV_circuit, num_batches=200):
#     total_diff = 0
#     width = OV_circuit.shape[1] // num_batches
#     for col in tqdm(range(num_batches)):
#         cols_softmax = OV_circuit[:, slice(col*width, (col+1)*width)].softmax(0).half()
#         cols_target = t.eye(cols_softmax.shape[0], device=cols_softmax.device)[:, slice(col*width, (col+1)*width)].half()
#         total_diff += (cols_softmax - cols_target).pow(2).sum(0).sqrt().sum()
#     return total_diff / OV_circuit.shape[1]

# def get_diff_from_identity(OV_circuit):
#     total_diff = 0
#     for col in tqdm(range(OV_circuit.shape[0])):
#         cols_softmax = OV_circuit[:, col].softmax(0).half()
#         cols_softmax[col] -= 1.0
#         a = cols_softmax.pow(2).sum().sqrt()
#         total_diff += a
#         if col > 10:
#             break
#     return total_diff / OV_circuit.shape[0]

def get_avg_diag_value(OV_circuit):
    total = 0
    total_sq_deviation_from_1 = 0
    checksum = 0
    for col in tqdm(range(OV_circuit.shape[0])):
        column = OV_circuit[:, col].softmax(0)
        total += column[col]
        total_sq_deviation_from_1 += (column[col] - 1) ** 2
        checksum += column.sum()
    return total / OV_circuit.shape[0]


if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc(OV_circuit))
    # print("Fraction of the time that one of the best five logits is on the diagonal:")
    # print(top_5_acc(OV_circuit))
    # print("Average vector norm deviation between softmax of logits and identity:")
    # print(get_diff_from_identity(OV_circuit))
    # print("Average diagonal value of probabilities:")
    # print(get_avg_diag_value(OV_circuit))

# %%

if MAIN:
    try:
        del OV_circuit
    except:
        pass
    W_OV_full = einsum("d_k emb1, emb2 d_k -> emb1 emb2", W_O_all[4], W_V_all[4]) + einsum("d_k emb1, emb2 d_k -> emb1 emb2", W_O_all[10], W_V_all[10])
    OV_circuit_full = einsum("emb1 voc1, emb1 emb2, voc2 emb2 -> voc1 voc2", W_U, W_OV_full, W_E)
    print("Top 1 accuracy for the full OV Circuit:", top_1_acc(OV_circuit_full))
    # print("Average vector norm deviation between softmax of logits and identity:", get_diff_from_identity(OV_circuit_full))
    print("Average diagonal value of probabilities:", get_avg_diag_value(OV_circuit_full))
    try:
        del OV_circuit_full
    except:
        pass

# TODO - THIS IS THE IDENTITY, BUT WHY SHOULD IT BE? IT'S NOT EXPLAINED. ISN'T THIS JUST BIGRAMS?

# Answer - no, and this is exactly the point with induction heads! We argued that the W_OV circuit is just doing copying!
# I can put this stuff into my animated explainer! i.e. go from animations to code!

# %%

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, 
    seq_len: int, 
    batch=1
) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Add a prefix token, since the model was always trained to have one.

    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: {} The cache of the model run on rep_tokens
    """
    prefix = t.ones((batch, 1), dtype=t.int64, device=model.cfg.device) * model.tokenizer.bos_token_id
    rand_tokens = t.randint(0, model.tokenizer.vocab_size, (batch, seq_len)).to(model.cfg.device)
    rep_tokens = t.cat([prefix, rand_tokens, rand_tokens], dim=1)

    rep_logits, rep_cache = model.run_with_cache(rep_tokens, remove_batch_dim=False)

    return rep_logits, rep_tokens, rep_cache

if MAIN:
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()

# %%

def mask_scores(
    attn_scores: TT["query_d_model", "key_d_model"]
):
    """Mask the attention scores so that tokens don't attend to previous tokens."""
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

if MAIN:
    W_pos = model.pos_embed.W_pos
    W_Q_all = model.blocks[0].attn.W_Q
    W_K_all = model.blocks[0].attn.W_K
    pos_by_pos_scores = einsum(
        "ctx1 emb1, emb1 d_k, emb2 d_k, ctx2 emb2 -> ctx1 ctx2",
        W_pos,
        W_Q_all[7],
        W_K_all[7],
        W_pos
    )
    pos_by_pos_pattern = mask_scores(pos_by_pos_scores / model.cfg.d_head ** 0.5).softmax(-1)
    imshow(to_numpy(pos_by_pos_pattern[:200, :200]), xaxis="Key", yaxis="Query")

# %%

seq_len = rep_tokens.shape[1]
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
d_head = model.cfg.d_head

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

@typechecked
def decompose_qk_input(cache: ActivationCache) -> TT[2+n_heads, seq_len, d_model]:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, position, d_model]
    '''
    emb: TT[seq_len, d_model] = cache["embed"][0]
    pos_embed: TT[seq_len, d_model] = cache["pos_embed"][0]
    layer0_head_output: TT[n_heads, seq_len, d_model] = rearrange(cache["result", 0][0], "pos nhead d_model -> nhead pos d_model")

    return t.concat([emb.unsqueeze(0), pos_embed.squeeze(0).unsqueeze(0), layer0_head_output], dim=0)

@typechecked
def decompose_q(decomposed_qk_input: TT[2+n_heads, seq_len, d_model], ind_head_index: int) -> TT[2+n_heads, seq_len, d_head]:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just q)
    '''
    W_Q: TT[d_model, d_head] = model.blocks[1].attn.W_Q[ind_head_index]
    return einsum("component seq d_model, d_model d_k -> component seq d_k", decomposed_qk_input, W_Q)

@typechecked
def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> TT[2+n_heads, seq_len, d_head]:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just k) - exactly analogous as for q
    '''
    return einsum("component seq emb, emb d_k -> component seq d_k", decomposed_qk_input, model.blocks[1].attn.W_K[ind_head_index])


if MAIN:
    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1][0] + rep_cache["pos_embed"][0], rtol=0.01, atol=1e-05)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["blocks.1.attn.hook_q"][0, :, ind_head_index], rtol=0.01, atol=0.001)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["blocks.1.attn.hook_k"][0, :, ind_head_index], rtol=0.01, atol=0.01)
    component_labels = ["Embed", "PosEmbed"] + [f"L0H{h}" for h in range(model.cfg.n_heads)]
    imshow(to_numpy(decomposed_q.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of query")
    imshow(to_numpy(decomposed_k.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of key")

# %%

import circuitsvis as cv
from IPython.display import display

def decompose_attn_scores(
    decomposed_q: TT["q_component": 2+n_heads, "q_pos": seq_len, "d_k": d_head], 
    decomposed_k: TT["k_component": 2+n_heads, "k_pos": seq_len, "d_k": d_head], 
) -> TT["q_component": 2+n_heads, "k_component": 2+n_heads, "q_pos": seq_len, "k_pos": seq_len]:
    '''
    Output is decomposed_attn_scores with shape [2+num_heads, position, position]
    '''
    return einsum("q_component q_pos d_k, k_component k_pos d_k -> q_component k_component q_pos k_pos", decomposed_q, decomposed_k)

if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )
    imshow(to_numpy(decomposed_stds), xaxis="Key Component", yaxis="Query Component", title="Standard deviations of components of scores", x=component_labels, y=component_labels)
    # title="Attention Scores for component from Q=Embed and K=Prev Token Head",
    # html = cv.attention.attention_heads(
    #     tokens=rep_str, 
    #     attention=t.tril(decomposed_scores[0, 9])
    # )
    # with open("attn.html", "w") as f:
    #     f.write(str(html))
    imshow(
        to_numpy(t.tril(decomposed_scores[0, 9]))
    )
    # Might have to make these actual probabilities?

# %%

# %%

def find_K_comp_full_circuit(prev_token_head_index, ind_head_index):
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.embed.W_E.half()

    W_Q = model.blocks[1].attn.W_Q[ind_head_index].half()
    W_K = model.blocks[1].attn.W_K[ind_head_index].half()
    QK_circuit = einsum("emb_Q d_model, emb_K d_model -> emb_Q emb_K", W_Q, W_K)
    
    W_V = model.blocks[0].attn.W_V[prev_token_head_index].half()
    W_O = model.blocks[0].attn.W_O[prev_token_head_index].half()
    OV_circuit = einsum("d_model emb_O, emb_V d_model -> emb_O emb_V", W_O, W_V)
    
    return einsum(
        "voc1 emb1, emb1 emb2, emb2 emb3, voc2 emb3 -> voc1 voc2",
        W_E, QK_circuit, OV_circuit, W_E
    )


if MAIN:
    ind_head_index = 4
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_circuit(prev_token_head_index, ind_head_index)
    print("Fraction of tokens where the highest activating key is the same token", top_1_acc(K_comp_circuit.T).item())
    del K_comp_circuit
# %%

def find_K_comp_full_full_circuit(prev_token_head_index, ind_head_indices):
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.embed.W_E.half()

    W_Qs = [model.blocks[1].attn.W_Q[i].half() for i in ind_head_indices]
    W_Ks = [model.blocks[1].attn.W_K[i].half() for i in ind_head_indices]
    QK_circuits = [einsum("emb_Q d_model, emb_K d_model -> emb_Q emb_K", W_Q, W_K) for (W_Q, W_K) in zip(W_Qs, W_Ks)]
    QK_circuit = sum(QK_circuits)
    
    W_V = model.blocks[0].attn.W_V[prev_token_head_index].half()
    W_O = model.blocks[0].attn.W_O[prev_token_head_index].half()
    OV_circuit = einsum("d_model emb_O, emb_V d_model -> emb_O emb_V", W_O, W_V)
    
    return einsum(
        "voc1 emb1, emb1 emb2, emb2 emb3, voc2 emb3 -> voc1 voc2",
        W_E, QK_circuit, OV_circuit, W_E
    )

if MAIN:
    ind_head_indices = [4, 10]
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_full_circuit(prev_token_head_index, ind_head_indices)
    print("Fraction of tokens where the highest activating key is the same token", top_1_acc(K_comp_circuit.T).item())
    del K_comp_circuit