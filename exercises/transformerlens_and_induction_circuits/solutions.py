# %%
import circuitsvis as cv
cv.examples.hello("Bob")

# %%
import plotly.io as pio
pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import repeat, rearrange, reduce
from fancy_einsum import einsum
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked
from typing import List, Union, Optional, Tuple
import functools
from tqdm import tqdm
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, display

import transformer_lens
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# import sys, os
# f = r"C:\Users\calsm\Documents\AI Alignment\ARENA\TRANSFORMERLENS_AND_MI\exercises"
# sys.path.append(f)
# os.chdir(f)

def imshow(tensor, renderer=None, xaxis="", yaxis="", caxis="", **kwargs):
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs) #.show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    return px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs) #.show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    return px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs) #.show(renderer)

# %%

device = "cuda" if t.cuda.is_available() else "cpu"
gpt2_model = HookedTransformer.from_pretrained("gpt2-small", device=device)
# gpt2_model.use_attn_result = True

# %%

model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

loss = gpt2_model(model_description_text, return_type="loss")
print("Model loss:", loss)

# %%

logits = gpt2_model(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
true_tokens = gpt2_model.to_tokens(model_description_text).squeeze()[1:]

num_correct = (prediction == true_tokens).sum()

print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
print(f"Correct words: {gpt2_model.to_str_tokens(prediction[prediction == true_tokens])}")

# %%

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_model.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = gpt2_model.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_model.to_str_tokens(gpt2_text)

html = cv.attention.attention_heads(tokens=gpt2_str_tokens, attention=attention_pattern)
with open("layer0_head_attn_patterns.html", "w") as f:
    f.write(str(html))

# %%

html = cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=attention_pattern)
with open("layer0_head_attn_patterns_2.html", "w") as f:
    f.write(str(html))


# %%

layer0_pattern_from_cache = gpt2_cache["pattern", 0]

seq, nhead, headsize = gpt2_cache["q", 0].shape
layer0_attn_scores = einsum("seqQ n h, seqK n h -> n seqQ seqK", gpt2_cache["q", 0], gpt2_cache["k", 0])
mask = t.tril(t.ones((seq, seq), device=device, dtype=bool))
layer0_attn_scores = t.where(mask, layer0_attn_scores, -1e9)
layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)

# %%

device = t.device("cuda" if t.cuda.is_available() else "cpu")

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
    positional_embedding_type="shortformer"
)

# %%

WEIGHT_PATH = "attn_only_2L_half.pth"

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)
# %%

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)
tokens = tokens.to(device)
logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
model.reset_hooks()
# %%

for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
    with open(f"layer_{layer}_attention.html", "w") as f:
        f.write(str(html))
# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of diagonal elements
            score = attention_pattern.diagonal().mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of sub-diagonal elements
            score = attention_pattern.diagonal(-1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of 0th elements
            score = attention_pattern[:, 0].mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
# %%

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, 
    seq_len: int, 
    batch: int = 1
) -> tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: The cache of the model run on rep_tokens
    '''
    prefix = t.ones((batch, 1), dtype=t.int64) * model.tokenizer.bos_token_id
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=1).to(device)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens, remove_batch_dim=True)
    return rep_logits, rep_tokens, rep_cache

def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]

if MAIN:
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print(f"Performance on the first half: {ptl[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {ptl[seq_len:].mean():.3f}")
    fig = px.line(
        utils.to_numpy(ptl), hover_name=rep_str[1:],
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
        labels={"index": "Sequence position", "value": "Loss"}
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()
# %%
def write_to_html(fig, filename):
    with open(f"{filename}.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
# write_to_html(fig, "repeated_tokens")

# %%

for layer in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", layer]
    html = cv.attention.attention_patterns(tokens=rep_str, attention=attention_pattern)
    with open(f"layer_{layer}_rep_attention.html", "w") as f:
        f.write(str(html))

# %%

def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len+1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%

def generate_rep_tokens(model: HookedTransformer, seq_len: int, batch: int) -> TT[batch, 2*seq_len+1]:
    prefix = t.ones((batch, 1), dtype=t.int64) * model.tokenizer.bos_token_id
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=1).to(device)
    return rep_tokens

seq_len = 50
batch = 10
rep_tokens = generate_rep_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

model.run_with_hooks(
    rep_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

fig = imshow(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head", text_auto=".2f")

write_to_html(fig, "induction_scores")

# %%

induction_head_layer = 5

seq_len = 50
batch = 10
rep_tokens = generate_rep_tokens(gpt2_model, seq_len, batch)

def visualize_pattern_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_model.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )

induction_score_store = t.zeros((gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads), device=gpt2_model.cfg.device)

gpt2_model.run_with_hooks(
    rep_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        utils.get_act_name("pattern", induction_head_layer),
        visualize_pattern_hook
    )]
)

# %%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
## 


def logit_attribution(embed, l1_results, l2_results, W_U, tokens) -> t.Tensor:
    '''
    We have provided 'W_U_correct_tokens' which is a (d_model, seq_next) tensor where each row is the unembed for the correct NEXT token at the current position.
    Inputs:
        embed (seq_len, d_model): the embeddings of the tokens (i.e. token + position embeddings)
        l1_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results (seq_len, n_heads, d_model): the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U (d_model, d_vocab): the unembedding matrix
    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = einsum("emb seq, seq emb -> seq", W_U_correct_tokens, embed[:-1])
    l1_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l1_results[:-1])
    l2_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l2_results[:-1])
    return t.cat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

if MAIN:
    text = "This must be Thursday. I never could get the hang of Thursdays."
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text).to(device)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    model.reset_hooks()

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-4, rtol=0)

# %%

def convert_tokens_to_string(tokens, batch_index=0):
    '''Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

def plot_logit_attribution(logit_attr: t.Tensor, tokens: t.Tensor):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    return imshow(utils.to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", height=25*len(tokens))

if MAIN:
    embed = cache["hook_embed"]
    l1_results = cache["blocks.0.attn.hook_result"]
    l2_results = cache["blocks.1.attn.hook_result"]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
    fig = plot_logit_attribution(logit_attr, tokens)

# fig.show()

# write_to_html(fig, "logit_attribution")

# %%

if MAIN:
    embed = rep_cache["hook_embed"]
    l1_results = rep_cache["blocks.0.attn.hook_result"]
    l2_results = rep_cache["blocks.1.attn.hook_result"]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]

    first_half_logit_attr = logit_attribution(embed[:1+seq_len], l1_results[:1+seq_len], l2_results[:1+seq_len], model.unembed.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.unembed.W_U, second_half_tokens)

    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    
    fig1 = plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    fig2 = plot_logit_attribution(second_half_logit_attr, second_half_tokens)

    write_to_html(fig2, "rep_logit_attribution")

# %%

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    str_tokens = gpt2_model.to_str_tokens(text)
    tokens = gpt2_model.to_tokens(text)
    tokens = tokens.to(device)
    logits, cache = gpt2_model.run_with_cache(tokens, remove_batch_dim=True)
    gpt2_model.reset_hooks()
# %%


def head_ablation_hook(
    value: TT["batch", "pos", "head_index", "d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> TT["batch", "pos", "head_index", "d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.0
    return value

if MAIN:
    layer_to_ablate = 0
    head_index_to_ablate = 7

    original_loss = gpt2_model(gpt2_tokens, return_type="loss")
    ablated_loss = gpt2_model.run_with_hooks(
        gpt2_tokens, 
        return_type="loss", 
        fwd_hooks=[(
            utils.get_act_name("v", layer_to_ablate), 
            functools.partial(head_ablation_hook, head_index_to_ablate=head_index_to_ablate)
        )]
    )
    print(f"Original Loss: {original_loss.item():.3f}")
    print(f"Ablated Loss: {ablated_loss.item():.3f}")

# %%

clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

clean_tokens = gpt2_model.to_tokens(clean_prompt)
corrupted_tokens = gpt2_model.to_tokens(corrupted_prompt)

def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
    # model.to_single_token maps a string value of a single token to the token index for that token
    # If the string is not a single token, it raises an error.
    correct_index = gpt2_model.to_single_token(correct_answer)
    incorrect_index = gpt2_model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# We run on the clean prompt with the cache so we store activations to patch in later.
clean_logits, clean_cache = gpt2_model.run_with_cache(clean_tokens, remove_batch_dim=True)
clean_logit_diff = logits_to_logit_diff(clean_logits)
print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

# We don't need to cache on the corrupted prompt.
corrupted_logits = gpt2_model(corrupted_tokens)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")

# %%

# We define a residual stream patching hook
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# The type annotations are a guide to the reader and are not necessary
def residual_stream_patching_hook(
    resid_pre: TT["batch", "pos", "d_model"],
    hook: HookPoint,
    position: int,
    clean_cache: ActivationCache
) -> TT["batch", "pos", "d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[position, :]
    return resid_pre

# We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
num_positions = len(clean_tokens[0])
ioi_patching_result = t.zeros((gpt2_model.cfg.n_layers, num_positions), device=gpt2_model.cfg.device)
gpt2_model.reset_hooks()

for layer in tqdm(range(gpt2_model.cfg.n_layers)):
    for position in range(num_positions):
        # Use functools.partial to create a temporary hook function with the position fixed
        temp_hook_fn = functools.partial(residual_stream_patching_hook, position=position, clean_cache=clean_cache)
        # Run the model with the patching hook
        patched_logits = gpt2_model.run_with_hooks(corrupted_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        ])
        # Calculate the logit difference
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        # Store the result, normalizing by the clean and corrupted logit difference so it's between 0 and 1 (ish)
        ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)
# %%

# Add the index to the end of the label, because plotly doesn't like duplicate labels
token_labels = [f"{token}_{index}" for index, token in enumerate(gpt2_model.to_str_tokens(clean_tokens))]
imshow(ioi_patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")
# %%


















# %%

" LOGIT ATTRIBUTION FOR INDUCTION HEADS "

if MAIN:
    seq_len = 50

    embed = rep_cache["hook_embed"]
    l1_results = rep_cache["blocks.0.attn.hook_result"]
    l2_results = rep_cache["blocks.1.attn.hook_result"]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]
    
    "YOUR CODE HERE"
    "Define `first_half_logit_attr` and `second_half_logit_attr`"
    first_half_logit_attr = logit_attribution(embed[:1+seq_len], l1_results[:1+seq_len], l2_results[:1+seq_len], model.unembed.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.unembed.W_U, second_half_tokens)

    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    
    plot_logit_attribution(first_half_logit_attr, first_half_tokens).show()
    plot_logit_attribution(second_half_logit_attr, second_half_tokens).show()

# EXERCISE - EXPLAIN THESE RESULTS

# Hint - the first plot is meaningless (why?)

# Solution
# The first plot is meaningless because the tokens are random! There's no order yet for the model to observe.
# Previously, we observed that heads 1.4 and 1.10 were acting as induction heads.
# This plot gives further evidence that this is the case, as these two heads have a large logit attribution score **on sequences in which the only way to get accurate predictions is to use induction**.

# %%

" ABLATION FOR INDUCTION HEADS "



def head_ablation_hook(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    attn_result[:, :, head_index_to_ablate] = 0.0
    return attn_result

def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def get_ablation_scores(
    model: HookedTransformer, 
    tokens: TT["batch", "seq"]
) -> TT["n_layers", "n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(patched_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores

if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    fig = imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads", text_auto=".2f")

# Question - why do you think it's valuable to ablate, when we've already done logit attribution?
# Answer - logit attribution might tell you something is important, but it won't tell you that it's necessary. (actually I don't like how this is phrased; pick something else!)





















# %%

# TODO - speak to Neel about adding this as a method for FactoredMatrix

def get_sample(M: FactoredMatrix, k=3, seed=None):
    if seed is not None: t.manual_seed(seed)
    indices = t.randint(0, M.A.shape[-2], (k,))
    return utils.get_corner(M.A[..., indices, :] @ M.B[..., :, indices], k)

if MAIN:
    W_O_4 = model.W_O[0, 4]
    W_V_4 = model.W_V[0, 4]
    W_E = model.W_E
    W_U = model.W_U

    OV_circuit = FactoredMatrix(W_V_4, W_O_4)
    full_OV_circuit = W_E @ OV_circuit @ W_U

    full_OV_circuit_sample = get_sample(full_OV_circuit, k=200)
    imshow(full_OV_circuit_sample).show()

# Trace = 209271.5625
# 

# %%

def top_1_acc(full_OV_circuit: FactoredMatrix):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B

    correct = []
    for i in tqdm(range(full_OV_circuit.shape[-1])):
        correct.append(t.argmax(A[i, :] @ B) == i)
    
    return t.tensor(correct).float().mean()

def top_5_acc(full_OV_circuit: FactoredMatrix):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B

    correct = []
    for i in tqdm(range(full_OV_circuit.shape[-1])):
        top5 = t.topk(A[i, :] @ B, k=5).indices
        correct.append(i in top5)
    
    return t.tensor(correct).float().mean()

if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    # print(top_1_acc(full_OV_circuit))
    print("Fraction of the time that the five best logits include the one on the diagonal:")
    print(top_5_acc(full_OV_circuit))


# %%

def top_1_acc_effective_circuit(full_OV_circuit_list: List[FactoredMatrix]):
    '''
    Returns top_1_acc for more than one OV_circuit
    '''
    num_circuits = len(full_OV_circuit_list)
    A_shape = full_OV_circuit_list[0].A.shape

    all_A = t.stack([full_OV_circuit.A for full_OV_circuit in full_OV_circuit_list])
    all_B = t.stack([full_OV_circuit.B for full_OV_circuit in full_OV_circuit_list])

    assert all_A.shape == (num_circuits, *A_shape)

    correct = []
    for i in tqdm(range(A_shape[0])):
        row = t.einsum("nj,njk->k", all_A[:, i, :], all_B)
        correct.append(row.argmax() == i)
    
    return t.tensor(correct).float().mean()

if MAIN:
    W_O_4, W_O_10 = model.W_O[1, [4, 10]]
    W_V_4, W_V_10 = model.W_V[1, [4, 10]]
    OV_circuit_4_full = W_E @ FactoredMatrix(W_V_4, W_O_4) @ W_U
    OV_circuit_10_full = W_E @ FactoredMatrix(W_V_10, W_O_10) @ W_U
    full_OV_circuit_list = [OV_circuit_4_full, OV_circuit_10_full]

    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc_effective_circuit(full_OV_circuit_list))
# %%

def mask_scores(attn_scores: TT["query_d_model", "key_d_model"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

if MAIN:
    "TODO: YOUR CODE HERE"
    W_pos = model.W_pos
    W_QK = model.W_Q[0, 7] @ model.W_K[0, 7].T
    pos_by_pos_scores = W_pos @ W_QK @ W_pos.T
    masked_scaled = mask_scores(pos_by_pos_scores / model.cfg.d_head ** 0.5)
    pos_by_pos_pattern = t.softmax(masked_scaled, dim=-1)

    imshow(utils.to_numpy(pos_by_pos_pattern[:200, :200]), xaxis="Key", yaxis="Query").show()

    print(f"Average lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
# %%

def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, pos, d_model]

    The [i, 0, 0]th element is y_i (from notation above)
    '''
    y0 = cache["embed"].unsqueeze(0) # shape (1, pos, d_model)
    y1 = cache["pos_embed"].unsqueeze(0) # shape (1, pos, d_model)
    y_rest = cache["result", 0].transpose(0, 1) # shape (12, pos, d_model)

    return t.cat([y0, y1, y_rest], dim=0)


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_Q (so the sum along axis i is just the q-values)
    '''
    W_Q = model.W_Q[1, ind_head_index]

    return einsum(
        "n pos d_head, d_head d_model -> n pos d_model",
        decomposed_qk_input, W_Q
    )


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_K(so the sum along axis i is just the k-values)
    '''
    W_K = model.W_K[1, ind_head_index]
    
    return einsum(
        "n pos d_head, d_head d_model -> n pos d_model",
        decomposed_qk_input, W_K
    )


if MAIN:
    # Compute decomposed input and output, test they are correct
    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)

    # Plot importance results
    component_labels = ["Embed", "PosEmbed"] + [f"L0H{h}" for h in range(model.cfg.n_heads)]
    fig1 = imshow(utils.to_numpy(decomposed_q.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of query", y=component_labels)
    fig2 = imshow(utils.to_numpy(decomposed_k.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of key", y=component_labels)
    write_to_html(fig1, "norms_of_query_components")
    write_to_html(fig2, "norms_of_key_components")

# %%

def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]
    
    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    return einsum(
        "q_comp q_pos d_model, k_comp k_pos d_model -> q_comp k_comp q_pos k_pos",
        decomposed_q, decomposed_k
    )


if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )
    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    fig1 = imshow(utils.to_numpy(t.tril(decomposed_scores[0, 9])), title="Attention Scores for component from Q=Embed and K=Prev Token Head")
    # Second plot: std dev over query and key positions, shown by component
    fig2 = imshow(utils.to_numpy(decomposed_stds), xaxis="Key Component", yaxis="Query Component", title="Standard deviations of components of scores", x=component_labels, y=component_labels)

    write_to_html(fig1, "attn_scores_for_component")
    write_to_html(fig2, "attn_scores_std_devs")
# %%

def find_K_comp_full_circuit(prev_token_head_index: int, ind_head_index: int) -> FactoredMatrix:
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]
    
    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K
    return FactoredMatrix(Q, K.T)


if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(prev_token_head_index, ind_head_index)
    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}", )

# %%

def find_K_comp_effective_circuit(prev_token_head_index: int, ind_head_indices: List[int]) -> List[FactoredMatrix]:
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.W_E
    W_Qs = model.W_Q[1, ind_head_indices]
    W_Ks = model.W_K[1, ind_head_indices]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]
    
    Qs = t.stack([W_E @ W_Q for W_Q in W_Qs])
    Ks = t.stack([W_E @ W_V @ W_O @ W_K for W_K in W_Ks])
    return [FactoredMatrix(Q, K.T).T for Q, K in zip(Qs, Ks)]


if MAIN:
    prev_token_head_index = 7
    ind_head_indices = [4, 10]
    K_comp_effective_circuit = find_K_comp_effective_circuit(prev_token_head_index, ind_head_indices)
    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc_effective_circuit(K_comp_effective_circuit):.4f}", )

    
# %%

# def outer_product_of_FM(
#     X: FactoredMatrix, 
#     Y: FactoredMatrix
# ) -> FactoredMatrix:
#     """Returns a large batch of factored matrices, constructed from their outer products.

#     Args:
#         X (*X_idx, XA, XB; Xmid): left-matrix
#         Y (*Y_idx, YA, YB; Ymid): right-matrix
#         with XB == YA

#     Returns:
#         FactoredMatrix (*X_idx, *Y_idx, XA, YB; Ymid)
#         = outer product of fm1 and fm2

#     Explanation:
#         This is useful when (X1, ..., Xm) and (Y1, ..., Yn) are indices for factored matrices (e.g. indexing on layer and head), and we want to do a batch of multiplications to e.g. calculate composition scores.
#     """
#     # Store indices of X and Y as strings, to be used in einops operations
#     X_index_values = " ".join(str(idx) for idx in X.shape[:-2])
#     Y_index_values = " ".join(str(idx) for idx in Y.shape[:-2])

#     # Extend dimensions of X and Y, so they can be multiplied together like normal matrices
#     X = FactoredMatrix(
#         repeat(X.A, f"... XA Xmid -> ... {Y_index_values} XA Xmid"),
#         repeat(X.B, f"... Xmid XB -> ... {Y_index_values} Xmid XB")
#     )
#     Y = FactoredMatrix(
#         repeat(Y.A, f"... YA Ymid -> {X_index_values} ... YA Ymid"),
#         repeat(Y.B, f"... Ymid YB -> {X_index_values} ... Ymid YB")
#     )

#     # Some assertions to check X and Y are suitable, and that this function worked as intended
#     assert len(X.shape) == len(Y.shape), "Unsuitable dims for multiplication"
#     assert X.B.shape[-1] == Y.A.shape[-2], "Unsuitable shapes for multiplication"

#     # Return matrix
#     return X @ Y


# def get_comp_scores(
#     W_As: FactoredMatrix, 
#     W_Bs: FactoredMatrix
# ) -> t.Tensor:
#     '''Returns the compositional scores from indexed tensors W_As and W_Bs.

#     Args:
#         W_As (FactoredMatrix): shape (*A_idx, A_in, A_out)
#         W_Bs (FactoredMatrix): shape (*B_idx, B_in, B_out), where A_out == B_in

#     Returns:
#         t.Tensor: shape (*A_idx, *B_idx)
#         the [*a, *b]th element is the compositional score from W_As[*a] to W_Bs[*b]

#     Example application:
#         W_As (nlayers-1, nhead, d_model, d_model):
#             The W_OV matrices from all but the last layer of a transformer
#         W_Bs (nhead, d_model, d_model):
#             The W_QK matrices from the last layer of a transformer
#         Output is tensor of shape (nlayers-1, nhead, nhead), where the [i, j, k]th entry is the compositional score from the output of head `i.j` to the input of head `k` in the last layer.
#     '''
#     W_ABs = outer_product_of_FM(W_As, W_Bs)

#     return W_ABs.norm() / t.outer(W_As.norm(), W_Bs.norm())

def batched_outer_matmul(
    X: FactoredMatrix,
    Y: FactoredMatrix,
    bottleneck_right: bool = True,
) -> FactoredMatrix:
    '''
    X.shape == (*X_idx, X_in, X_out)
    Y.shape == (*Y_idx, Y_in==X_out, Y_out)
    bottleneck_right: if True, then the bottleneck dimension is on the right, otherwise it's on the left

    Returns FactoredMatrix with shape (*X_idx, *Y_idx, X_in, Y_out), where the [*x_idx, *y_idx]th element is FactoredMatrix(X[*x_idx], Y[*y_idx]).

    Use-case? Maybe X and Y are actually input and output weights W_A and W_B, and we want to multiply them to get composition scores. Or we just want to make a bunch of circuits!
    '''
    # Reshape X and Y to only have one index dimension
    # Note, we include a dummy index dimension of size 1, so we can broadcast when we multiply X and Y
    X = FactoredMatrix(
        X.A.reshape(-1, 1, *X.A.shape[-2:]),
        X.B.reshape(-1, 1, *X.B.shape[-2:]),
    )
    Y = FactoredMatrix(
        Y.A.reshape(1, -1, *Y.A.shape[-2:]),
        Y.B.reshape(1, -1, *Y.B.shape[-2:]),
    )

    # Return the product
    if bottleneck_right:
        return FactoredMatrix(X @ Y.A, Y.B)
    else:
        return FactoredMatrix(X.A, Y @ X.B)


# %%

def get_comp_score(
    W_A: TT["in_A", "out_A"], 
    W_B: TT["out_A", "out_B"]
) -> float:
    '''
    Return the composition score between W_A and W_B.
    '''
    W_A_norm = W_A.pow(2).sum()
    W_B_norm = W_B.pow(2).sum()
    W_AB_norm = (W_A @ W_B).pow(2).sum()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item() ** 0.5

# %%

if MAIN:
    # Get all QK and OV matrices
    W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
    W_OV = model.W_V @ model.W_O

    # Define tensors to hold the composition scores
    q_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads)
    k_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads)
    v_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads)
    
    # Fill in the tensors
    "YOUR CODE HERE!"
    for i in tqdm(range(model.cfg.n_heads)):
        for j in range(model.cfg.n_heads):
            q_comp_scores[i, j] = get_comp_score(W_OV[0][i], W_QK[1][j])
            k_comp_scores[i, j] = get_comp_score(W_OV[0][i], W_QK[1][j].T)
            v_comp_scores[i, j] = get_comp_score(W_OV[0][i], W_OV[1][j])

    px.imshow(
        utils.to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        utils.to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        utils.to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()

# %%

def generate_single_random_comp_score() -> float:

    W_A_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_A_right = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_right = t.empty(model.cfg.d_model, model.cfg.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)

if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()
    print("Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()

# %%

def get_batched_comp_scores(
    W_As: FactoredMatrix,
    W_Bs: FactoredMatrix
) -> t.Tensor:
    '''Returns the compositional scores from indexed tensors W_As and W_Bs.

    Each of W_As and W_Bs is a FactoredMatrix object which is indexed by all but its last 2 dimensions, i.e. W_As.shape == (*A_idx, A_in, A_out) and W_Bs.shape == (*B_idx, B_in, B_out).

    Return: tensor of shape (*A_idx, *B_idx) where the [*a_idx, *b_idx]th element is the compositional score from W_As[*a_idx] to W_Bs[*b_idx].
    '''
    # Reshape W_As and W_Bs to only have one index dimension
    # Note, we include a dummy index dimension of size 1, so we can broadcast when we multiply W_As and W_Bs
    W_As = FactoredMatrix(
        W_As.A.reshape(-1, 1, *W_As.A.shape[-2:]),
        W_As.B.reshape(-1, 1, *W_As.B.shape[-2:]),
    )
    W_Bs = FactoredMatrix(
        W_Bs.A.reshape(1, -1, *W_Bs.A.shape[-2:]),
        W_Bs.B.reshape(1, -1, *W_Bs.B.shape[-2:]),
    )

    # Compute the product
    W_ABs = W_As @ W_Bs

    # Compute the norms, and return the metric
    return W_ABs.norm() / (W_As.norm() * W_Bs.norm())

# %%

if MAIN:
    W_V = model.W_V
    W_O = model.W_O
    W_Q = model.W_Q
    W_K = model.W_K

    W_QK = FactoredMatrix(W_Q, W_K.transpose(-1, -2))
    W_OV = FactoredMatrix(W_V, W_O)

    q_comp_scores = get_batched_comp_scores(W_OV[0], W_QK[1])
    k_comp_scores = get_batched_comp_scores(W_OV[0], W_QK[1].T) # Factored matrix: .T is interpreted as transpose of the last two axes
    v_comp_scores = get_batched_comp_scores(W_OV[0], W_OV[1])

# %%

if MAIN:
    figQ = px.imshow(
        utils.to_numpy(q_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    )
    figK = px.imshow(
        utils.to_numpy(k_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    )
    figV = px.imshow(
        utils.to_numpy(v_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    )
    write_to_html(figQ, "q_comp_scores")
    write_to_html(figK, "k_comp_scores")
    write_to_html(figV, "v_comp_scores")

# %%


def ablation_induction_score(prev_head_index: Optional[int], ind_head_index: int) -> float:
    '''
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    '''

    def ablation_hook(v, hook):
        if prev_head_index is not None:
            v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            (utils.get_act_name("v", 0), ablation_hook),
            (utils.get_act_name("pattern", 1), induction_pattern_hook)
        ],
    )
    return model.blocks[1].attn.hook_pattern.ctx[prev_head_index].item()


if MAIN:
    baseline_induction_score = ablation_induction_score(None, 4)
    print(f"Induction score for no ablations: {baseline_induction_score}\n")
    for i in range(model.cfg.n_heads):
        new_induction_score = ablation_induction_score(i, 4)
        induction_score_change = new_induction_score - baseline_induction_score
        print(f"Ablation score change for head {i}:", induction_score_change)

# %%

