# %%
MAIN = __name__ == "__main__"

if MAIN:

    f = r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v2-exercises\chapter6_interpretability"
    import sys
    sys.path.append(f)

    from IPython import get_ipython
    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    import plotly.io as pio
    # pio.renderers.default = "notebook_connected"
    pio.renderers.default = "browser"

    # import CircuitsVis.python.circuitsvis as cv
    import circuitsvis as cv
    # Testing that the library works
    cv.examples.hello("Bob")

    # Import stuff
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import einops
    from fancy_einsum import einsum
    import tqdm.auto as tqdm
    import random
    from pathlib import Path
    import plotly.express as px
    from torch.utils.data import DataLoader

    from torchtyping import TensorType as TT
    from typing import List, Union, Optional
    from functools import partial
    import copy

    import itertools
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    import dataclasses
    import datasets
    from IPython.display import HTML, display

    import transformer_lens
    import transformer_lens.utils as utils
    from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
    from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

    torch.set_grad_enabled(False)

    def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
        px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
        return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)

    def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
        px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)
        return px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)

    def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
        x = utils.to_numpy(x)
        y = utils.to_numpy(y)
        px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
        return px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%

example_text = "The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph."
example_text_str_tokens = model.to_str_tokens(example_text)
print(example_text_str_tokens)
# %%

example_text_tokens = model.to_tokens(example_text)
print(example_text_tokens)

# %%

example_multi_text = ["The cat sat on the mat.", "The cat sat on the mat really hard."]
example_multi_text_tokens = model.to_tokens(example_multi_text)
print(example_multi_text_tokens)

# %%

cat_text = "The cat sat on the mat."
cat_logits = model(cat_text)
cat_probs = cat_logits.softmax(dim=-1)
print(f"Probability tensor shape [batch, position, d_vocab] == {cat_probs.shape}")

capital_the_token_index = model.to_single_token(" The")
print(f"| The| probability: {cat_probs[0, -1, capital_the_token_index].item():.2%}")
# %%

print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")

# %%

ioi_logits_with_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=True)
mary_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference with BOS: {(claire_logit_with_bos - mary_logit_with_bos):.3f}")

ioi_logits_without_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=False)
mary_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference without BOS: {(claire_logit_without_bos - mary_logit_without_bos):.3f}")
# %%

OV_circuit_all_heads = model.OV
print(OV_circuit_all_heads)

OV_circuit_all_heads_eigenvalues = OV_circuit_all_heads.eigenvalues 
print(OV_circuit_all_heads_eigenvalues.shape)
print(OV_circuit_all_heads_eigenvalues.dtype)

OV_copying_score = OV_circuit_all_heads_eigenvalues.sum(dim=-1).real / OV_circuit_all_heads_eigenvalues.abs().sum(dim=-1)
fig = imshow(utils.to_numpy(OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)


# %%
# with open("ov_copying.html", "w") as f:
#     f.write(str(fig))

def write_to_html(fig, filename):
    with open(f"{filename}.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
# write_to_html(fig, "ov_copying")
# # %%
# write_to_html(scatter(x=OV_circuit_all_heads_eigenvalues[-1, -1, :].real, y=OV_circuit_all_heads_eigenvalues[-1, -1, :].imag, title="Eigenvalues of Head L11H11 of GPT-2 Small", xaxis="Real", yaxis="Imaginary"), "scatter_evals")

# %%

full_OV_circuit = (model.embed.W_E.to("cpu") @ OV_circuit_all_heads.to("cpu") @ model.unembed.W_U.to("cpu")).to("cuda")
print(full_OV_circuit)
full_OV_circuit_eigenvalues = full_OV_circuit.eigenvalues
print(full_OV_circuit_eigenvalues.shape)
print(full_OV_circuit_eigenvalues.dtype)
full_OV_copying_score = full_OV_circuit_eigenvalues.sum(dim=-1).real / full_OV_circuit_eigenvalues.abs().sum(dim=-1)
imshow(utils.to_numpy(full_OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)

write_to_html(scatter(x=full_OV_copying_score.flatten(), y=OV_copying_score.flatten(), hover_name=[f"L{layer}H{head}" for layer in range(12) for head in range(12)], title="OV Copying Score for each head in GPT-2 Small", xaxis="Full OV Copying Score", yaxis="OV Copying Score"), "scatter_ov")
# %%

model.generate("(CNN) President Barack Obama caught in embarrassing new scandal\n", max_new_tokens=50, temperature=0.7, prepend_bos=True)

# %%

model.to_str_tokens("The cat sat on the mat.")

# %%

example_text_tokens = model.to_tokens(example_text)
print(example_text_tokens)

# %%
example_multi_text = ["The cat sat on the mat.", "The cat sat on the mat really hard."]
example_multi_text_tokens = model.to_tokens(example_multi_text)
print(example_multi_text_tokens.shape)

# %%

cat_text = "The cat sat on the mat."
cat_logits = model(cat_text)
cat_probs = cat_logits.softmax(dim=-1)
print(f"Probability tensor shape [batch, position, d_vocab] == {cat_probs.shape}")

capital_the_token_index = model.to_single_token(" The")
print(f"| The| probability: {cat_probs[0, -1, capital_the_token_index].item():.2%}")

# %%

print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")

# %%

clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
tokens = model.to_tokens(clean_prompt)
logits, cache = model.run_with_cache(clean_prompt, remove_batch_dim=True)


# %%
cache
# %%
dir(cache)
# %%
len(cache.items())

# %%
cache["v", 3, "attn"].shape