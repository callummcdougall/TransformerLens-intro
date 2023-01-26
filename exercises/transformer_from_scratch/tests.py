# %%

import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import math
from transformer_lens import EasyTransformer
import tqdm.auto as tqdm

import os; os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\TRANSFORMERLENS_AND_MI\exercises\transformer_from_scratch")

from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# %%

@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()

# %%

def test_layernorm(LayerNorm: nn.Module):
    import solutions

    ln: solutions.LayerNorm = LayerNorm(cfg)
    ln_soln = solutions.LayerNorm(cfg)

    residual = torch.randn(3, 4, cfg.d_model)
    ln_res = ln(residual)
    ln_soln_res = ln_soln(residual)
    torch.testing.assert_close(ln_res, ln_soln_res)
    print("Normalization tests passed")

    rand_w = torch.randn_like(ln.w.data)
    ln.w.data = rand_w
    ln_soln.w.data = rand_w
    rand_b = torch.randn_like(ln.b.data)
    ln.b.data = rand_b
    ln_soln.b.data = rand_b

    ln_res = ln(residual)
    ln_soln_res = ln_soln(residual)
    torch.testing.assert_close(ln_res, ln_soln_res)
    print("Learned parameters tests passed")

