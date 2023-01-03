# %%
f = r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v2-exercises\chapter6_interpretability"
import sys
sys.path.append(f)

# %%

from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# %%

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
        px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %%

import plotly.io as pio
# pio.renderers.default = "notebook_connected"
pio.renderers.default = "browser"


# %%
# import CircuitsVis.python.circuitsvis as cv
import circuitsvis as cv
# Testing that the library works
cv.examples.hello("Bob")

# %%

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

# %%

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# %%

torch.set_grad_enabled(False)

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%



def visualize_transformer(model: HookedTransformer, hooks=False, **kwargs):
    """
    Genrates the string for a mermaid plot of the transformer architecture.

    Also includes the positions of hook points, if hooks=True.

    Args:
        model (HookedTransformer): A transformer model from Neel's library.
        hooks (bool, optional): Whether to include positions and names of the hooks. Defaults to False.
    """

    # We make a list of all the layers
    layers = []
    for layer in range(model.cfg.n_layers):
        # We make a list of all the heads
        heads = []
        for head in range(model.cfg.n_heads):
            # We make a list of all the attention patterns
            patterns = []
            for pattern in ["pattern", "query", "key", "value"]:
                # We make a list of all the attention patterns
                patterns.append(f"{pattern}_{layer}_{head}")
            # We make a list of all the attention patterns
            heads.append(f"head_{layer}_{head}({', '.join(patterns)})")
        # We make a list of all the attention patterns
        layers.append(f"layer_{layer}({', '.join(heads)})")

    # We make a list of all the attention patterns
    layers_str = ", ".join(layers)

    # We make a list of all the attention patterns
    mermaid_str = f"graph LR\n{layers_str}"

    if hooks:
        # We make a list of all the attention patterns
        hooks_str = "\n".join(
            f"hook_{hook.layer()}_{hook.head()}({hook.name()}) --> layer_{hook.layer()}"
            for hook in model.hooks
        )
        mermaid_str += f"\n{hooks_str}"

    return mermaid_str


# %%

s = visualize_transformer(model, hooks=False)
# %%

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = model.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%
import torchinfo
torchinfo.summary(model, input_data=gpt2_tokens, depth=torch.inf)
# %%

model("testing the model")
# %%

import torch as t

A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")
# %%

print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)

# %%

C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")

# %%

AB_unfactored = AB_factor.AB
print(t.isclose(AB_unfactored, AB).all())

# %%

OV_circuit_all_heads = model.OV
print(OV_circuit_all_heads)

OV_circuit_all_heads_eigenvalues = OV_circuit_all_heads.eigenvalues 
print(OV_circuit_all_heads_eigenvalues.shape)
print(OV_circuit_all_heads_eigenvalues.dtype)

OV_copying_score = OV_circuit_all_heads_eigenvalues.sum(dim=-1).real / OV_circuit_all_heads_eigenvalues.abs().sum(dim=-1)
imshow(utils.to_numpy(OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)
# %%

scatter(x=OV_circuit_all_heads_eigenvalues[-1, -1, :].real, y=OV_circuit_all_heads_eigenvalues[-1, -1, :].imag, title="Eigenvalues of Head L11H11 of GPT-2 Small", xaxis="Real", yaxis="Imaginary")


# %%

full_OV_circuit = model.embed.W_E @ OV_circuit_all_heads @ model.unembed.W_U
print(full_OV_circuit)

# %%

full_OV_circuit_eigenvalues = full_OV_circuit.eigenvalues
print(full_OV_circuit_eigenvalues.shape)
print(full_OV_circuit_eigenvalues.dtype)

full_OV_copying_score = full_OV_circuit_eigenvalues.sum(dim=-1).real / full_OV_circuit_eigenvalues.abs().sum(dim=-1)
imshow(utils.to_numpy(full_OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)

# %%
