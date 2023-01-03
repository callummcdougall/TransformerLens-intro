# %%
import os
import sys
sys.path.append(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v2-exercises\chapter6_interpretability")
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import w6d4_tests
# from w2d4_attn_only_transformer import AttnOnlyTransformer
from dataclasses import dataclass
from fancy_einsum import einsum

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from torchtyping import TensorType as TT
from typeguard import typechecked

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

pio.renderers.default = "notebook"
device = "cuda" if t.cuda.is_available() else "cpu"
MAIN = __name__ == "__main__"

# download weights from https://drive.google.com/u/0/uc?id=19FQ4UQ-5vw8d-duXR9haks5xyKYPzvS8&export=download
# and save them in your directory, with the following path:

# WEIGHT_PATH = "./data/attn_only_2L.pth"
WEIGHT_PATH = "./data/attn_only_2L.pth"

# TASK 1 - define an attention-only transformer with the following params:

from typing import Tuple


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
    # if setting from config, set tokenizer this way rather than passing it in explicitly
    # model initialises via AutoTokenizer.from_pretrained(tokenizer_name)

    seed=398,
    # dataset_name="the_pile",
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. use layernorm with weights and biases
    positional_embedding_type="shortformer" # this makes it so positional embeddings are used differently (makes induction heads cleaner to study)
)

# def delete_biases(model: HookedTransformer) -> None:
#     for layer in range(model.cfg.n_layers):
#         for m in "Q", "K", "V", "O":
#             assert getattr(model.blocks[layer].attn, f"b_{m}").data.pow(2).sum() == 0
#     model.unembed.b_U = None

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    # delete_biases(model)
    # for (k, v) in pretrained_weights.items():
    #     if "attn.W_" in k:
    #         pretrained_weights[k] = rearrange(v, "i j k -> i k j")
    #     elif "embed" in k:
    #         pretrained_weights[k] = v.T
    # for (k, v) in raw_weights.items():
    #     if ".b_" in k and k not in pretrained_weights:
    #         pretrained_weights[k] = t.zeros_like(v)
    model.load_state_dict(pretrained_weights)


# %%

if MAIN:
    pretrained_weights_edited = {k: v.half() for k, v in pretrained_weights.items()}
    t.save(pretrained_weights_edited, "./data/attn_only_2L_half.pth")


# %%


# if MAIN:
#     example_text = "IT WAS A BRIGHT cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
#     example_tokens = model.to_tokens(example_text)
#     logits, cache = model.run_with_cache(example_tokens, remove_batch_dim=True)
#     print(f"There are {example_tokens.shape[-1]} tokens\n")
#     logits = model(example_tokens)
#     model.reset_hooks()
#     for activation_name in cache:
#         activation = cache[activation_name]
#         print(f"Activation: {activation_name:30} Shape: {tuple(activation.shape)}")
# # %%

# import importlib
# importlib.reload(w6d4_tests)

# def mask_scores(
#     attn_scores: TT["query_d_model", "key_d_model"]
# ):
#     """Mask the attention scores so that tokens don't attend to previous tokens."""
#     mask = t.tril(t.ones_like(attn_scores)).bool()
#     neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
#     masked_attn_scores = t.where(mask, attn_scores, neg_inf)
#     return masked_attn_scores

# @t.inference_mode()
# def QK_attn(
#     W_QK: TT["query_d_model", "key_d_model"], 
#     qk_input: TT["seq_pos", "d_model"]
# ) -> TT["seq_pos", "seq_pos"]:
#     """Calculate attention scores, using formula from the Mathematical Frameworks paper."""
#     attn_scores = einsum("seq1 headsize1, headsize1 headsize2, headsize2 seq2 -> seq1 seq2", qk_input, W_QK, qk_input.T)
#     attn_scores_masked = mask_scores(attn_scores / model.cfg.d_head ** 0.5)
#     attn_pattern = attn_scores_masked.softmax(-1)
#     return attn_pattern
    

# if MAIN:
#     layer = 0
#     head_index = 0
#     W_Q = model.blocks[layer].attn.W_Q[head_index]
#     W_K = model.blocks[layer].attn.W_K[head_index]
#     qk_input = cache[f"blocks.{layer}.hook_resid_pre"]
#     original_attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][head_index, :, :]
#     W_QK = W_Q @ W_K.T
#     QK_attn_pattern = QK_attn(W_QK, qk_input)

#     t.testing.assert_close(QK_attn_pattern, original_attn_pattern, atol=1e-3, rtol=0)
# # %%

# # def OV_result_mix_before(
# #     W_OV: TT["d_model", "d_model"], 
# #     residual_stream_pre: TT["seqQ", "d_model"], 
# #     attn_pattern: TT["seqQ", "seqK"]
# # ) -> TT["seqQ", "d_model"]:
# #     """
# #     Apply attention to the residual stream, and THEN apply W_OV.
# #     """
# #     pass

# # # @t.inference_mode()

# # if MAIN:
# #     layer = 0
# #     head_index = 0
# #     batch_index = 0
# #     W_O = model.blocks[layer].attn.W_O[head_index].detach().clone()
# #     W_V = model.blocks[layer].attn.W_V[head_index].detach().clone()
# #     W_OV = W_O @ W_V
# #     residual_stream_pre = cache[f"blocks.{layer}.hook_resid_pre"][batch_index].detach().clone()
# #     original_head_results = (
# #         cache[f"blocks.{layer}.attn.hook_result"][batch_index, :, head_index].detach().clone()
# #     )
# #     expected_attn_pattern = cache[f"blocks.{layer}.attn.hook_attn_scores"][batch_index, head_index, :, :].detach().clone()

# #     computed_head_out = OV_result_mix_before(W_OV, residual_stream_pre, expected_attn_pattern)
# #     expected_head_out = cache[f"blocks.{layer}.attn_score_out"]

# #     t.testing.assert_close(computed_head_out, expected_attn_pattern, atol=1e-1, rtol=0)








# %%  

# ====================================
# BUILDING INTERPRETABILITY TOOLS
# ====================================

# %%

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)
    tokens = tokens.to(device)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    model.reset_hooks()
# %%

def to_numpy(tensor):
    """Helper function to convert things to numpy before plotting with Plotly."""
    return tensor.detach().cpu().numpy()


def convert_tokens_to_string(tokens, batch_index=0):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]


def plot_logit_attribution(logit_attr: TT["seq", "path"], tokens: TT["seq"]):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    px.imshow(
        to_numpy(logit_attr),
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        height=25*len(tokens),
    ).show()



# Help - I'm confused about the number of dimensions / I can't get them to line up.

# Remember that the nth sequnce position in our model is used to predict the (n+1)th token. We passed in tokens[0, 1:] into W_U because we wanted to get the direction of correct predictions, but we should use embed[:-1] and l1_results[:-1], l2_results[:-1] because these represent the predictions for those tokens.

n_components = model.cfg.n_layers * model.cfg.n_heads + 1

# Add a note about `typechecked` here!

seq_len = tokens.shape[-1]


@typechecked
def logit_attribution(
    embed: TT["seq_len", "d_model"],
    l1_results: TT["seq_len", "n_heads", "d_model"],
    l2_results: TT["seq_len", "n_heads", "d_model"],
    W_U: TT["d_model", "d_vocab"],
    tokens: TT["seq_len"],
) -> TT["seq_len_less1", "n_components"]:
    """
    We have provided 'W_U_to_logits' which is a (d_model, seq_next) tensor where each row is the unembed for the correct NEXT token at the current position.
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
    Returns:
        Tensor representing the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
    """
    W_U_to_logits = W_U[:, tokens[1:]]

    direct_path_logits = einsum("emb seq_next, seq_next emb -> seq_next", W_U_to_logits, embed[:-1]).unsqueeze(1)
    l1_logits = einsum("emb seq_next, seq_next n_heads emb -> seq_next n_heads", W_U_to_logits, l1_results[:-1])
    l2_logits = einsum("emb seq_next, seq_next n_heads emb -> seq_next n_heads", W_U_to_logits, l2_results[:-1])
    logit_attribution = t.concat([direct_path_logits, l1_logits, l2_logits], dim=-1)
    return logit_attribution
    


if MAIN:
    with t.inference_mode():
        batch_index = 0
        embed = cache["hook_embed"]
        l1_results = cache["result", 0] # same as cache["blocks.0.attn.hook_result"]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[batch_index, t.arange(len(tokens[0]) - 1), tokens[batch_index, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-2, rtol=0)

if MAIN:
    embed = cache["hook_embed"]
    l1_results = cache["blocks.0.attn.hook_result"]
    l2_results = cache["blocks.1.attn.hook_result"]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
    plot_logit_attribution(logit_attr, tokens)

# %%

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import torch as t

patch_typeguard()  # use before @typechecked

nbatch = 3
@typechecked
def func(
    x: TensorType["nbatch"],
    y: TensorType["nbatch"]
) -> TensorType["nbatch_doubled": nbatch * 3]:
    return t.concat([x, y])

# func(t.rand(3), t.rand(3))  # works
# TypeError: Dimension 'nbatch' of inconsistent size. Got both 1 and 3.

# %%


# =============================
# VISUALISING ATTENTION PATTERNS

import circuitsvis as cv
# cv.examples.hello("Bob")

for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
    with open(f"layer_{layer}_attention.html", "w") as f:
        f.write(str(html))

# =============================
# SUMMARIZING ATTENTION PATTERNS

# Give a hint for what is required for this

seq_len = len(tokens[0])

from typing import List, Dict, Tuple, Optional, Callable, Any, Union

def current_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    current_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of diagonal elements
            current_attn_score = attention_pattern[t.arange(seq_len), t.arange(seq_len)].mean()
            if current_attn_score > 0.4:
                current_attn_heads.append(f"{layer}.{head}")
    return current_attn_heads



def prev_attn_detector(cache: ActivationCache):
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    prev_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of subdiagonal elements
            prev_attn_score = attention_pattern[t.arange(seq_len-1)+1, t.arange(seq_len-1)].mean()
            if prev_attn_score > 0.4:
                prev_attn_heads.append(f"{layer}.{head}")
    return prev_attn_heads


def first_attn_detector(cache: ActivationCache):
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    first_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of first column
            first_attn_score = attention_pattern[:, 0].mean()
            if first_attn_score > 0.4:
                first_attn_heads.append(f"{layer}.{head}")
    return first_attn_heads


def plot_head_scores(scores_tensor, title=""):
    px.imshow(
        to_numpy(scores_tensor),
        labels={"y": "Layer", "x": "Head"},
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).show()


if MAIN:
    # Compare this printout with your attention pattern visualisations. Do they make sense?
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%

# =============================
# ABLATIONS


# I think this example is just unnecessary. Remind people of how to ablate w/ reference to prev section, then do exercise.

# In fact, we've seen an example of ablations already, so skip it altogether! Go straight to finding induction heads.

# Question - one of the values actually increases when you ablate. Can you guess why?
# Answer - it seems to be the value corresponding to the "deceptive" prediction. This is because "deceptive" is quite an unlikely word.

if MAIN:
    print("As a reminder, here's the name and shape of each hook-able activation")
    print(f"The batch size is {tokens.size(0)} and the context length is {tokens.size(1)}")
    for activation_name in cache:
        activation = cache[activation_name]
        print(f"Activation: {activation_name} Shape: {activation.shape}")

def ablate_residual_stream_hook(resid_post, hook):
    resid_post[:, 3:] = 0.0
    return resid_post


def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]


if MAIN:
    corrupted_logits = model.run_with_hooks(
        tokens, fwd_hooks=[("blocks.1.hook_resid_post", ablate_residual_stream_hook)]
    )
    clean_per_token_losses = per_token_losses(logits, tokens)
    corrupted_per_token_losses = per_token_losses(corrupted_logits, tokens)
    px.line(
        to_numpy(corrupted_per_token_losses - clean_per_token_losses), 
        labels={"index": "Token", "y": "Per token loss"},
        title="Difference in per token loss after ablating residual stream" #, template="ggplot2"
    ).update_layout(showlegend=False).show()

# %%

# Exercise: ablate each of the induction heads in turn, and make a plot of the difference in cross entropy loss

# This is a good exercise because it forces you to understand how the head patching code worked.

# Rather than swapping out corrupted for clean, you're just zeroing.

def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


# @typechecked
def head_ablation(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_no: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    attn_result[:, :, head_no, :] = 0.0
    return attn_result



def get_ablation_scores(model: HookedTransformer, tokens: t.Tensor):

    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    model.reset_hooks()
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head_no in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = partial(head_ablation, head_no=head_no)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer, "attn"), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(patched_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head_no] = loss - loss_no_ablation

    return ablation_scores

from tqdm import tqdm
from functools import partial


ablation_scores = get_ablation_scores(model, tokens)

imshow(ablation_scores, xaxis="Head", yaxis="Layer", title="Logit Difference After Ablating Heads", text_auto=".2f")

# Note - remember to run `model.reset_hooks()` !

# %%

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, 
    seq_len: int, 
    batch=1
) -> tuple[t.Tensor, t.Tensor, ActivationCache]:
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

def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]
if MAIN:
    "\n    These are small numbers, since the results are very obvious and this makes it easier to visualise - in practice we'd obviously use larger ones on more subtle tasks. But it's often easiest to iterate and debug on small tasks.\n"
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print(f"Performance on the first half: {ptl[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {ptl[seq_len:].mean():.3f}")
    fig = px.line(
        to_numpy(ptl),
        hover_name=rep_str[1:],
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
        labels={"index": "Sequence position", "value": "Loss"},
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=49.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=49.5, x1=99, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()

# %%

# Note - remember that this cache has a batch dim; you should remove this before running cv.attention.attention_heads

for layer in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", layer][0]
    html = cv.attention.attention_heads(tokens=rep_str, attention=attention_pattern)
    with open(f"layer_{layer}_attention.html", "w") as f:
        f.write(str(html))

# %%

def induction_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember:
        The tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    induction_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][0, head]
            # queries are all the tokens T which have been repeated before
            q_indices = t.arange(seq_len+1, 2*seq_len+1) # quer
            # keys are the tokens AFTER the FIRST occurrence of T
            k_indices = t.arange(2, seq_len+2)
            # get induction score
            induction_attn_score = attention_pattern[q_indices, k_indices].mean()
            if induction_attn_score > 0.4:
                induction_attn_heads.append(f"{layer}.{head}")
    return induction_attn_heads


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%

if MAIN:
    seq_len = 50
    embed = rep_cache["hook_embed"][0]
    l1_results = rep_cache["blocks.0.attn.hook_result"][0]
    l2_results = rep_cache["blocks.1.attn.hook_result"][0]
    first_half_tokens = rep_tokens[0, :seq_len+1]
    second_half_tokens = rep_tokens[0, seq_len:]

    first_half_logit_attr = logit_attribution(embed[:1+seq_len], l1_results[:1+seq_len], l2_results[:1+seq_len], model.unembed.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.unembed.W_U, second_half_tokens)

    plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    plot_logit_attribution(second_half_logit_attr, second_half_tokens)

# %%

if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    imshow(ablation_scores, xaxis="Head", yaxis="Layer", title="Logit Difference After Ablating Heads (detecting induction heads)", text_auto=".2f")

# Question from this - which head in layer 0 is part of the induction circuit? Can you describe the role it plays?
# Answer - 0.7 is a prev-token head

# %%







# %%




# %%