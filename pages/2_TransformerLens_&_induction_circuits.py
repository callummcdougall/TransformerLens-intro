import os
if not os.path.exists("./images"):
    os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

def img_to_html(img_path, width):
    with open("images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;st-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

def read_from_html(filename):
    filename = f"images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

NAMES = ["attribution_fig", "attribution_fig_2", "failure_types_fig", "failure_types_fig_2", "logit_diff_from_patching", "line", "attn_induction_score","distil_plot", "ov_copying", "scatter_evals"]

NAMES.extend(["repeated_tokens", "induction_scores", "logit_attribution", "rep_logit_attribution", "logit_diff_rep", "norms_of_query_components", "norms_of_key_components", "attn_scores_for_component", "attn_scores_std_devs", "q_comp_scores", "k_comp_scores", "v_comp_scores"])

def complete_fig_dict(fig_dict):
    for name in NAMES:
        if name not in fig_dict:
            fig_dict[name] = read_from_html(name)
    return fig_dict
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
fig_dict_old = st.session_state["fig_dict"]
fig_dict = complete_fig_dict(fig_dict_old)
if len(fig_dict) > len(fig_dict_old):
    st.session_state["fig_dict"] = fig_dict

with open("images/layer0_head_attn_patterns.html") as f:
    layer0_head_attn_patterns = f.read()

def section_home():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#setup">Setup</a></li>
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# TransformerLens & induction circuits

These pages are designed to get you introduced to Neel's **TransformerLens** library.

Most of the sections are constructed in the following way:

1. A particular feature of TransformerLens is introduced. 
2. You are given an exercise, in which you have to apply the feature. 

The running theme of the exercises is **induction circuits**. Induction circuits are a particular type of circuit in a transformer, which can perform basic in-context learning. You should read the [corresponding section of Neel's glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_Jzi6YHRHKP1JziwdE02qdYZ), before continuing. This [LessWrong post](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated) might also help; it contains some diagrams (like the one below) which walk through the induction mechanism step by step.""")

    st.markdown("")
    st_image("kcomp_diagram.png", 900)
    st.markdown("")
    st.markdown(r"""

""")

#     st.info(r"""
# Today's material is transcribed directly from . If you prefer to go through the original Colab, you can find it [here](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/new-demo/New_Demo.ipynb#scrollTo=cfdePPmEkR8y) (or alternatively you can pull it from Colab into a VSCode notebook).

# .""")

    st.markdown(r"""
## Setup

If you haven't done initial setup, then you should navigate to the **Home** page from the left hand sidebar, and follow the instructions there.

Once you have, you can run the following code at the top of your file before starting:

```python
import plotly.io as pio
import plotly.express as px
pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser

import torch as t
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import einops
from fancy_einsum import einsum
import random
from pathlib import Path
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
from dataclasses import dataclass
import datasets
from IPython.display import HTML, display

import transformer_lens
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

import circuitsvis as cv

def imshow(tensor, renderer=None, xaxis="", yaxis="", caxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Learning Objectives

We've included the learning objectives for each section, so you can get a sense of what the general flow of the material is. You can also return here to check that you've understood all the material.
""")

    st.info(r"""
## 1️⃣ TransformerLens: Introduction

* Load and run a `HookedTransformer` model.
* Understand the basic architecture of these models.
* Use the model's tokenizer to convert text to tokens, and vice versa.
* Know how to cache activations, and to access activations from the cache.
* Use `circuitsvis` to visualise attention heads.

""")
    st.info(r"""
## 2️⃣ Finding induction heads

* Understand what induction heads are, and the algorithm they are implementing.
* Inspect activation patterns to identify basic attention head patterns, and write your own functions to detect attention heads for you.
* Identify induction heads by looking at the attention patterns produced from a repeating random sequence.

""")
    st.info(r"""
## 3️⃣ TransformerLens: Hooks

* Understand what hooks are, and how TransformerLens uses them
* Use hooks to access activations, process the results, and write them to an external tensor.
* Build tools to perform attribution, i.e. detecting which components of your model are responsible for performance on a given task.
* Use hooks to intervene on activations, specifically to perform **ablation** and **activation patching**.
    * Understand the importance of both these approaches for interpretability.
""")
    st.info(r"""
## 4️⃣ Reverse-engineering induction circuits

* Understand the difference between investigating a circuit by looking at activtion patterns, and reverse-engineering a circuit by looking directly at the weights.
* Use the factored matrix class to inspect the QK and OV circuits within an induction circuit.
* Perform further exploration of induction circuits: composition scores, and targeted ablations.
""")
    st.info(r"""
## 5️⃣ TransformerLens: Other Features

* Learn about the different models available in the TransformerLens library.
* Read about some transformer implementation details, such as folding layernorm.
* Understand exactly how tokenization works, including subtle details such as the beginning of sequence (BOS) token.
* Use transformers to generate text.
* Build your own custom transformer, with its own hook points.
* Load models at pre-trained checkpoints, and study these models to identify the induction head phase transition.
""")

def section_intro():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#loading-and-running-models">Loading and Running Models</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#hookedtransformerconfig">HookedTransformerConfig</a></li>
        <li><a class="contents-el" href="#running-your-model">Running your model</a></li>
    </ul></li>
    <li><a class="contents-el" href="#transformer-architecture">Transformer architecture</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#parameters-and-activations">Parameters and Activations</a></li>
    </ul></li>
    <li><a class="contents-el" href="#tokenization">Tokenization</a></li>
    <li><a class="contents-el" href="#caching-all-activations">Caching all Activations</a></li>
    <li><a class="contents-el" href="#visualising-attention-heads">Visualising Attention Heads</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# TransformerLens: Introduction""")
    st.markdown(r"")

#    <li><ul class="contents">
#        <li><a class="contents-el" href="#activation-hook-names">Activation + Hook Names</a></li>
#        <li><a class="contents-el" href="#folding-layernorm-for-the-curious">Folding LayerNorm (For the Curious)</a></li>
#     </ul></li>
    st.info(r"""
## Learning objectives

* Load and run a `HookedTransformer` model.
* Understand the basic architecture of these models.
* Use the model's tokenizer to convert text to tokens, and vice versa.
* Know how to cache activations, and to access activations from the cache.
* Use `circuitsvis` to visualise attention heads.

""")

    st.markdown(r"""

## Introduction

This is a demo notebook for [TransformerLens](https://github.com/neelnanda-io/TransformerLens), **a library I ([Neel Nanda](neelnanda.io)) wrote for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models.** The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this! Mechanistic interpretability is a very young and small field, and there are a *lot* of open problems - if you would like to help, please try working on one! **Check out my [list of concrete open problems](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit#) to figure out where to start.**

I wrote this library because after I left the Anthropic interpretability team and started doing independent research, I got extremely frustrated by the state of open source tooling. There's a lot of excellent infrastructure like HuggingFace and DeepSpeed to *use* or *train* models, but very little to dig into their internals and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get into the field even if you don't work at an industry org with real infrastructure! The core features were heavily inspired by [Anthropic's excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson Elhage and Chris Olah for building Garcon and showing me the value of good infrastructure for accelerating exploratory research!

The core design principle I've followed is to enable exploratory analysis - one of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. This notebook demonstrates how the library works and how to use it, but if you want to see how well it works for exploratory research, check out [my notebook analysing Indirect Objection Identification](https://github.com/neelnanda-io/TransformerLens/blob/main/Exploratory_Analysis_Demo.ipynb) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!""")

    st.markdown(r"""
## Loading and Running Models

TransformerLens comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. For this demo notebook we'll look at GPT-2 Small, an 80M parameter model, see the Available Models section for info on the rest.

```python
device = "cuda" if t.cuda.is_available() else "cpu"

gpt2_small = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

### HookedTransformerConfig

Alternatively, you can define a config object, then call `HookedTransformer.from_config(cfg)` to define your model. This is particularly useful when you want to have finer control over the architecture of your model. We'll see an example of this in the next section, when we define an attention-only model to study induction heads.

Even if you don't define your model in this way, you can still access the config object through the `cfg` attribute of the model.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - inspect your model

Use `model.cfg` to find the following, for your GPT-2 Small model:

* Number of layers
* Number of heads per layer
* Maximum context window

You might have to check out the documentation page for some of these. You can reach it by right-clicking on `HookedTransformerConfig` in the sidebar, and choosing "Go to definition".
""")
        with st.expander("Solution"):
            st.markdown(r"""
The following parameters in the config object give you the answers:

```
cfg.n_layers == 2
cfg.n_heads == 12
cfg.n_ctx == 2048
```
""")
    st.markdown(r"""
### Running your model

To try the model the model out, let's find the loss on this text!

Models can be run on a single string or a tensor of tokens (shape: `[batch, position]`, all integers). The possible return types are: 

* `"logits"` (shape `[batch, position, d_vocab]`, floats), 
* `"loss"` (the cross-entropy loss when predicting the next token), 
* `"both"` (a tuple of `(logits, loss)`) 
* `None` (run the model, but don't calculate the logits - this is faster when we only want to use intermediate activations)

```python
model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)
```

## Transformer architecture

HookedTransformer is a somewhat adapted GPT-2 architecture, but is computationally identical. The most significant changes are to the internal structure of the attention heads:

* The weights `W_K`, `W_Q`, `W_V` mapping the residual stream to queries, keys and values are 3 separate matrices, rather than big concatenated one.
* The weight matrices `W_K`, `W_Q`, `W_V`, `W_O` and activations have separate head_index and d_head axes, rather than flattening them into one big axis.
    * The activations all have shape `[batch, position, head_index, d_head]`.
    * `W_K`, `W_Q`, `W_V` have shape `[head_index, d_model, d_head]` and `W_O` has shape `[head_index, d_head, d_model]`""")

    st.info(r"""
* **Important - we generally follow the convention that weight matrices multiply on the right rather than the left.** In other words, they have shape `[input, output]`, and we have `new_activation = old_activation @ weights + bias`.
    * Click the dropdown below for examples of this, if it seems unintuitive.
""")
    with st.expander("Examples of matrix multiplication in our model"):
        st.markdown(r"""
* **Query matrices**
    * Each query matrix `W_Q` for a particular layer and head has shape `[d_model, d_head]`. 
    * So if a vector `x` in the residual stream has length `d_model`, then the corresponding query vector is `x @ W_Q`, which has length `d_head`.
* **Embedding matrix**
    * The embedding matrix `W_E` has shape `[d_vocab, d_model]`. 
    * So if `A` is a one-hot-encoded vector of length `d_vocab` corresponding to a particular token, then the embedding vector for this token is `A @ W_E`, which has length `d_model`.
""")
    st.markdown(r"""

The actual code is a bit of a mess, as there's a variety of Boolean flags to make it consistent with the various different model families in TransformerLens - to understand it and the internal structure, I instead recommend reading the code in [CleanTransformerDemo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb).

### Parameters and Activations

It's important to distinguish between parameters and activations in the model.

* **Parameters** are the weights and biases that are learned during training.
    * These don't change when the model input changes.
    * They can be accessed direction fromm the model, e.g. `model.W_E` for the embedding matrix.
* **Activations** are temporary numbers calculated during a forward pass, that are functions of the input.
    * We can think of these values as only existing for the duration of a single forward pass, and disappearing afterwards.
    * We can use hooks to access these values during a forward pass (more on hooks later), but it doesn't make sense to talk about a model's activations outside the context of some particular input.
    * Attention scores and patterns are activations (this is slightly non-intuitve because they're used in a matrix multiplication with another activation).

The dropdown below contains a diagram of a single layer (called a `TransformerBlock`) for an attention-only model with no biases. Each box corresponds to an **activation** (and also tells you the name of the corresponding hook point, which we will eventually use to access those activations). The red text below each box tells you the shape of the activation (ignoring the batch dimension). Each arrow corresponds to an operation on an activation; where there are **parameters** involved these are labelled on the arrows.
""")

    with st.expander("Attention-only diagram"):
        st.write("""<figure style="max-width:620px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrNVsFu2zAM_RVBh7UDYnQLdnKyHIasOfQwFCu2Q10EikXHhmVJkWQ3Sd1_H-U4dR2kQLFDEx0kUibtR_pR1BONFQca0qVhOiV300gSHLZc7DYiOraaSWLdRsD3i0RJF9hsC-Hwm15fTO4MkzZRpgDzQ6g4J5fMOUmUFJvP4yvvOYno7pV-xIJZO4WEQKHdhjxm3KXhF70epJAtU-fF0RFrDxEnoUxogKNFZ2PAZnyuDdxH9EUe86zaI14ow8EETulwqNfEKpFxshAszkcFM8tMNo-aYMZX6DcZN19rvbuPdpF_HXrjSwurAeHzAq0Fxuq9MNaHIAim9-QhDMMmxiCY1IzzusWmrOuwd9J7090k9xMUVo_6y7G9dyz_Nx_5sX5UfXXVV_O-6iOZ21hhXvoPrvuqRkMwsr8566vbvoqvLMWrNIPkR_jif8zf-Z-6QuJUH0cYmQLjnjd-7dNmdkCbx6YmgBNWLYlKSMVECaSC2Clj6y3i3p4eN0GkBBP5q96lfVeGKJwA2mEpemy1LQuiKjDEW9m64Z0qPc69eOrT4k1y3tYrhLk6C3JeH5CTK0e0UbyMHWFCyWXrNSA2ZgIIk5wUzOb1q0Lf53ynfUhYbUQY3W0z3_SiQnaoxBVsXbfHDCJspTNAF8zeZMZNnSPU_DyYcayhde2u68uonLLUDmFiW6ADiheXgmUcL0BPfjuiLoUCIhqiyCFhvpNgn3tG01Jz5uAnz_D8pWHChIUBZaVTvzcypqEzJeyNphnDfl60Vs__ANh9IJM" /></figure>""", unsafe_allow_html=True)

    st.markdown(r"""
The next dropdown contains a diagram of a `TransformerBlock` with full features (including biases, layernorms, and MLPs). Don't worry if not all of this makes sense at first - we'll return to some of the details later. As we work with these transformers, we'll get more comfortable with their architecture.
""")



    with st.expander("Full diagram"):
        st.write("""<figure style="max-width:680px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrdV1FP2zAQ_itWpI1tasSIeApdJaYWJqGNIRA8UBS5sdNadeLUdkJbwn_fOUkJ6ZoCm9R2y4N955yT786fz-cHyxeEWq41lDgeoatuP0LwqGRQDPSttopxhJSecfplLxCRthWbU9c5jKd7nSuJIxUIGVL5lQt_3N431p2-VXzGPD7HSnVpgGgY6xm6Z0SP3M_xtDWibDjSRjxaYW1gQcOFdCUlYFHZSKoY8WJJb_vWk9wmLF2gHAhJqLS1iF0nniIlOCNowLE_PgqxHLIof5U70N6HeZ12_rdydvXTytsDxxh_UHTSQsQLwZp_bO-bWeDrnW3b3Vt057pu7qNtdzJMSFZgCxmB973G97FQunIElG16UgW5kl7LBR4doPc0VPFR2T1v0ZruJev17QrK5ah9zGl9KAKeYg6ASTVOI7KCWAiWCGXguJbY1yikOIJosZRBbAczCADJ8u_DuuX95pbs4NliFShvPA7gBtBmlYMArFL-VUJhrSP0Flqgt3Z_1jYwLq2rk7o6rqvGN0_5AihXf3FSV2MwpDKqD57W1XldhU8mXDdQvGKFyUI33rWhznWWAmHSzfFkRDHxGJkaxhi5nktPl3LlfX5QUIJwOkQiQCnmCUUp9bWQKpsD9PlOQF_sx_OsWIIiq4OwHXTLW9HAg5wWIpFSmVuqFoJzCA0YVsCC8ywnpUgM8IW47WO1mbkXhrkX2QTATnaFuSdLzCVCo1gKksAhgrmIhuWsVnE8IRwRFGI1zp6lg0XwC20jnlVOgY8XeXtWcwx4IwId4mlW5iMAWUq7AdA-bSbKmSHKWTYGzOOdIcrfFVoOevHYW1cWOU11kbO2MIJK9qlQBXmbqeG1BZqzsxWa81-UaCGPX1tfNRByJfnyykcule_mbvRiVeMsQs7ykLMoK-6JG70hHqJPjZwFunoBoCpufZu97zXjgoDBYW8iBl0Gq1qWAaW05a1uo17JzXzVC_HdOwQAfxx_720E3eW345-933bNlkFYLSukQH1GLNd6MJD6lh7RkPYtF0RCA2yuArDlHsE0iQnWtEcY1M2WG2CuaMvCiRaXs8i3XC0TujDqMgz7PyytHn8BSKkJUQ" /></figure>""", unsafe_allow_html=True)
        # graph TD
        #     subgraph "<span style='font-size:24px'>TransformerBlock</span>"
        #         classDef empty width:0px,height:0px;
        #         classDef code color:red;

        #         resid_pre["resid_pre<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]---D[ ]:::empty-->|add|resid_mid---E[ ]:::empty-->|add|resid_post["resid_post<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]
                
        #         subgraph "<span style='font-size:24px'>ln1 &emsp; &emsp;&emsp;&emsp; &emsp; &emsp; &emsp; &emsp;&emsp;&emsp;&emsp; &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>"
        #             scale
        #             normalized
        #         end
        #         resid_pre --> |subtract mean, divide by std|scale["scale<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, 1)</code>"] --> |W_ln, b_ln|normalized["normalized<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]
                
        #         subgraph "<span style='font-size:24px'>attn &emsp;&emsp; &emsp; &emsp;&emsp;&emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>"
        #             v
        #             q
        #             k
        #             attn_scores
        #             F
        #             pattern
        #             G
        #             z
        #             result
        #         end
        #         normalized-->|W_V, b_V|v["v<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, head_idx, d_head)</code>"]---G[ ]:::empty-->|weighted avg of value vectors|z["z<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, head_idx, d_head)</code>"] --> |W_O|result["result<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, head_idx, d_model)</code>"] -->|sum over heads, add bias b_O|attn_out["attn_out<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]---D
        #         normalized-->|W_Q, b_Q|q["q<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, head_idx, d_head)</code>"]---F[ ]:::empty-->|dot product along d_head, scale and mask|attn_scores["attn_scores<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(head_idx, seqQ, seqK)</code>"]-->|softmax|pattern["pattern<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(head_idx, seqQ, seqK)</code>"]---G
        #         normalized-->|W_K, b_K|k["k<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, head_idx, d_head)</code>"]---F
                
        #         subgraph "<span style='font-size:24px'>ln2 &emsp;&emsp; &emsp; &emsp; &emsp;</span>"
        #             scale2
        #             normalized2
        #         end
        #         resid_mid["resid_mid<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"] --> |subtract mean, divide by std|scale2["scale<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, 1)</code>"] --> |W_ln, b_ln|normalized2["normalized<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]
                
        #         subgraph "<span style='font-size:24px'>mlp &emsp; &emsp; &emsp; &emsp;&emsp;&emsp;</span>"
        #             normalized2
        #             pre
        #             post
        #         end
        #         normalized2 --> |W_in, b_in|pre["pre<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, 4 * d_model)</code>"] --> |act_fn|post["post<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, 4 * d_model)</code>"] -->|W_out, b_out|mlp_out["mlp_out<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"] --- E 
        #     end

        #     %% ["NAME<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(SHAPE)</code>"]
        #     %% ["NAME<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(SHAPE)</code>"]

    st.markdown(r"""
A few shortctus to make your lives easier when using these models:

* You can index weights like `W_Q` directly from the model via e.g. `model.blocks[0].attn.W_Q` (which gives you the `[nheads, d_model, d_head]` query weights for all heads in layer 0).
    * But an easier way is just to index with `model.W_Q`, which gives you the `[nlayers, nheads, d_model, d_head]` tensor containing **every** query weight in the model.
* Similarly, there exist shortcuts `model.W_E`, `model.W_U` and `model.W_pos` for the embeddings, unembeddings and positional embeddings respectively.

## Tokenization

The tokenizer is stored inside the model, and you can access it using `model.tokenizer`. There are also a few helper methods that call the tokenizer under the hood, for instance:

* `model.to_str_tokens(text)` converts a string into a tensor of tokens-as-strings.
* `model.to_tokens(text)` converts a string into a tensor of tokens.
* `model.to_string(tokens)` converts a tensor of tokens into a string.

Examples of use:

```python
gpt2_small.to_str_tokens("gpt2")             # --> ['<|endoftext|>', 'g', 'pt', '2']
gpt2_small.to_tokens("gpt2")                 # --> tensor([[50256,    70,   457,    17]], device='cuda:0')
gpt2_small.to_string([50256, 70, 457, 17])   # --> '<|endoftext|>gpt2'
```

Later sections will go into more detail on how to use the tokenizer, but for now this will suffice.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - how many words does your model guess correctly?

Consider the `model_description_text` you fed into your model above. How many words did your model guess correctly? Which words were correct? 

""")

        with st.expander("Hint"):
            st.markdown(r"""
Use `return_type="logits"` to get the model's predictions, then take argmax across the vocab dimension. Then, compare these predictions with the actual tokens, derived from the `model_description_text`.

Remember, you should be comparing the `[:-1]`th elements of this tensor of predictions with the `[1:]`th elements of the input tokens (because your model's output represents a probability distribution over the *next* token, not the current one).
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
logits = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
num_correct = (prediction == true_tokens).sum()

print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
print(f"Correct words: {gpt2_small.to_str_tokens(prediction[prediction == true_tokens])}")
```

The output from this code is:

```
Model accuracy: 21/87
Correct words: ['\n', '\n', 'former', ' with', ' models', '.', ' can', ' of', 'ooked', 'Trans', 'former', '_', 'NAME', '`.', '\n', ' at', 'PT', '-', ',', "'s", ' the']
```

So the model got 21 out of 87 words correct. Not bad!""")

    st.markdown(r"""
**Induction heads** are a special kind of attention head which we'll examine a lot more in coming exercises. They allow a model to perform in-context learning of a specific form: generalising from one observation that token `B` follows token `A`, to predict that token `B` will follow `A` in future occurrences of `A`, even if these two tokens had never appeared together in the model's training data. **Can you see evidence of any induction heads at work?**""")

    with st.expander("Evidence of induction heads"):
        st.markdown(r"""

The evidence for induction heads comes from the fact that the model successfully predicted `'ooked', 'Trans', 'former'` following the token `'H'`. This is because it's the second time that `HookedTransformer` had appeared in this text string, and the model predicted it the second time but not the first. (The model did predict `former` the first time, but we can reasonably assume that `Transformer` so this prediction wouldn't require the induction capability, unlike `HookedTransformer`.)
""")

    st.markdown(r"""

## Caching all Activations

The first basic operation when doing mechanistic interpretability is to break open the black box of the model and look at all of the internal activations of a model. This can be done with `logits, cache = model.run_with_cache(tokens)`. Let's try this out, on the first sentence from the GPT-2 paper.""")

    with st.expander("A note on `remove_batch_dim`"):
        st.markdown(r"""
Every activation inside the model begins with a batch dimension. Here, because we only entered a single batch dimension, that dimension is always length 1 and kinda annoying, so passing in the `remove_batch_dim=True` keyword removes it. `gpt2_cache_no_batch_dim = gpt2_cache.remove_batch_dim()` would have achieved the same effect.
""")

    st.markdown(r"""
```python
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
```

If you inspect the `gpt2_cache` object, you should see that it contains a very large number of keys, each one corresponding to a different activation in the model. You can access the keys by indexing the cache directly, or by a more convenient indexing shorthand. For instance, the code:

```python
gpt2_cache["pattern", 0]
```

returns the same thing as:

```python
gpt2_cache["blocks.0.attn.hook_pattern"]
```
""")

    with st.expander("Aside:"):
        st.markdown(r"""
The reason these are the same is that, under the hood, the first example actually indexes by `utils.get_act_name("pattern", 0)`, which evaluates to `"blocks.0.attn.hook_pattern"`.

In general, `utils.get_act_name` is a useful function for getting the full name of an activation, given its short name and layer number.

You can use the diagram from the **Transformer Architecture** section to help you find activation names.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - verify activations

Verify that `hook_q`, `hook_k` and `hook_pattern` are related to each other in the way implied by the diagram. Do this by computing `layer0_pattern_from_cache` (the attention pattern taken directly from the cache, for layer 0) and `layer0_pattern_from_q_and_k` (the attention pattern calculated from `hook_q` and `hook_k`, for layer 0). Remember that attention pattern is the probabilities, so you'll need to scale and softmax appropriately.""")

        st.info(r"""
Note - if you're already comfortable implementing things like attention calculations (e.g. having gone through Neel's transformer walkthrough) you can skip this exercise. However, it might serve as a useful refresher.
""")
        st.markdown(r"""

```python
layer0_pattern_from_cache = None    # You should replace this!
layer0_pattern_from_q_and_k = None  # You should replace this!

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

seq, nhead, headsize = gpt2_cache["q", 0].shape
layer0_attn_scores = einsum("seqQ n h, seqK n h -> n seqQ seqK", gpt2_cache["q", 0], gpt2_cache["k", 0])
mask = t.tril(t.ones((seq, seq), device=device, dtype=bool))
layer0_attn_scores = t.where(mask, layer0_attn_scores, -1e9)
layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
```
""")
    st.markdown(r"""

## Visualising Attention Heads

A key insight from the Mathematical Frameworks paper is that we should focus on interpreting the parts of the model that are intrinsically interpretable - the input tokens, the output logits and the attention patterns. Everything else (the residual stream, keys, queries, values, etc) are compressed intermediate states when calculating meaningful things. So a natural place to start is classifying heads by their attention patterns on various texts.

When doing interpretability, it's always good to begin by visualising your data, rather than taking summary statistics. Summary statistics can be super misleading! But now that we have visualised the attention patterns, we can create some basic summary statistics and use our visualisations to validate them! (Accordingly, being good at web dev/data visualisation is a surprisingly useful skillset! Neural networks are very high-dimensional object.)

Let's visualize the attention pattern of all the heads in layer 0, using [Alan Cooney's CircuitsVis library](https://github.com/alan-cooney/CircuitsVis) (based on Anthropic's PySvelte library). We will use the function `cv.attention.attention_heads`, which takes two arguments:

* `attention`: Attention head activations. 
    * This should be a tensor of shape `[nhead, seq_dest, seq_src]`, i.e. the `[i, :, :]`th element is the grid of attention patterns (probabilities) for the `i`th attention head.
    * We get this by indexing our `gpt2_cache` object.
* `tokens`: List of tokens (e.g. `["A", "person"]`). 
    * Sequence length must match that inferred from `attention`.
    * This is used to label the grid.
    * We get this by using the `model.to_str_tokens` method.
""")

    st.markdown(r"""

This visualization is interactive! Try hovering over a token or head, and click to lock. The grid on the top left and for each head is the attention pattern as a destination position by source position grid. It's lower triangular because GPT-2 has **causal attention**, attention can only look backwards, so information can only move forwards in the network.

```python
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0, "attn"]
print(attention_pattern.shape)
gpt2_str_tokens = model.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_heads(tokens=gpt2_str_tokens, attention=attention_pattern)
```""")
    st.info(r"""
Note - this library is currently under development, and still might have a few bugs. In particular, the attention diagrams sometimes shrink when you hover over them, depending on the IDE you're using. One solution is to write the diagrams to an HTML file instead:

```python
html = cv.attention.attention_heads(tokens=gpt2_str_tokens, attention=attention_pattern)
with open("layer0_attn_patterns.html", "w") as f:
    f.write(str(html))
```

Then the file should pop up in your explorer on the left of VSCode. Right click on it and select "Open in Default Browser" to view it in your browser.
""")
    # with open("images/cv_attn.html") as f:
    #     text = f.read()
    # st.components.v1.html(text, height=400)
    st.components.v1.html(layer0_head_attn_patterns, height=1200)
    # st.markdown(layer0_head_attn_patterns, unsafe_allow_html=True)

    st.info(r"""
Note - this graphic was produced by the function `cv.attention.attention_heads`. You can also produce a slightly different graphic with `cv.attention.attention_pattern` (same arguments `tokens` and `attention`), which presents basically the same information in a slightly different way, shown below:
""")
    with open("images/layer0_head_attn_patterns_2.html") as f:
        text = f.read()
    st.components.v1.html(text, height=600)


def section_finding_induction_heads():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#learning-objectives">Learning objectives</a></li>
   <li><a class="contents-el" href="#introducing-our-toy-attention-only-model">Introducing Our Toy Attention-Only Model</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#summarising-attention-patterns">Summarising attention patterns</a></li>
   </ul></li>
   <li><a class="contents-el" href="#what-are-induction-heads">What are induction heads?</a></li>
   <li><a class="contents-el" href="#checking-for-the-induction-capability">Checking for the induction capability</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#looking-for-induction-attention-patterns">Looking for Induction Attention Patterns</a></li>
</ul>
""", unsafe_allow_html=True)    
    st.markdown(r"""
# Finding Induction Heads""")
    st.markdown(r"")

    st.info(r"""
## Learning objectives

* Understand what induction heads are, and the algorithm they are implementing.
* Inspect activation patterns to identify basic attention head patterns, and write your own functions to detect attention heads for you.
* Identify induction heads by looking at the attention patterns produced from a repeating random sequence.

""")
    st.markdown(r"""

## Introducing Our Toy Attention-Only Model

Here we introduce a toy 2L attention-only transformer trained specifically for today. Some changes to make them easier to interpret:
- It has only attention blocks.
- The positional embeddings are only added to each key and query vector in the attention layers as opposed to the token embeddings (meaning that the residual stream can't directly encode positional information).
    - This turns out to make it *way* easier for induction heads to form, it happens 2-3x times earlier - [see the comparison of two training runs](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-83---VmlldzoyNTI0MDMz?accessToken=8ap8ir6y072uqa4f9uinotdtrwmoa8d8k2je4ec0lyasf1jcm3mtdh37ouijgdbm) here. (The bump in each curve is the formation of induction heads.)
    - The argument that does this below is `positional_embedding_type="shortformer"`.
- It has no MLP layers, no LayerNorms, and no biases.
- There are separate embed and unembed matrices (i.e. the weights are not tied).

As mentioned previously, we are now defining our model with a `HookedTransformerConfig` object:

```python
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
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)
```""")

    with st.expander("An aside about tokenizers"):
        st.markdown(r"""
In the last section, we defined a tokenizer explicitly, and passed it into our model. But here, we just pass a tokenizer name. The model automatically creates a tokenizer for us (under the hood, it calls `AutoTokenizer.from_pretrained(tokenizer_name)`).
""")
    st.markdown(r"""

You should download your model weights from [this Google Drive link](https://drive.google.com/drive/folders/1LRkK3tqNuZ6Y_UwgQKxML3980DtM0KUe), and save them under `WEIGHT_PATH` as given below.

```python
WEIGHT_PATH = "attn_only_2L_half.pth"

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)
```

""")

    with st.expander("Click here to remind yourself of the relevant hook names."):
        st.markdown(r"""
This is for a model with just attention layers, and no MLPs, LayerNorms, or biases (like the one we're using).
""")
        st.write("""<figure style="max-width:620px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrNVsFu2zAM_RVBh7UDYnQLdnKyHIasOfQwFCu2Q10EikXHhmVJkWQ3Sd1_H-U4dR2kQLFDEx0kUibtR_pR1BONFQca0qVhOiV300gSHLZc7DYiOraaSWLdRsD3i0RJF9hsC-Hwm15fTO4MkzZRpgDzQ6g4J5fMOUmUFJvP4yvvOYno7pV-xIJZO4WEQKHdhjxm3KXhF70epJAtU-fF0RFrDxEnoUxogKNFZ2PAZnyuDdxH9EUe86zaI14ow8EETulwqNfEKpFxshAszkcFM8tMNo-aYMZX6DcZN19rvbuPdpF_HXrjSwurAeHzAq0Fxuq9MNaHIAim9-QhDMMmxiCY1IzzusWmrOuwd9J7090k9xMUVo_6y7G9dyz_Nx_5sX5UfXXVV_O-6iOZ21hhXvoPrvuqRkMwsr8566vbvoqvLMWrNIPkR_jif8zf-Z-6QuJUH0cYmQLjnjd-7dNmdkCbx6YmgBNWLYlKSMVECaSC2Clj6y3i3p4eN0GkBBP5q96lfVeGKJwA2mEpemy1LQuiKjDEW9m64Z0qPc69eOrT4k1y3tYrhLk6C3JeH5CTK0e0UbyMHWFCyWXrNSA2ZgIIk5wUzOb1q0Lf53ynfUhYbUQY3W0z3_SiQnaoxBVsXbfHDCJspTNAF8zeZMZNnSPU_DyYcayhde2u68uonLLUDmFiW6ADiheXgmUcL0BPfjuiLoUCIhqiyCFhvpNgn3tG01Jz5uAnz_D8pWHChIUBZaVTvzcypqEzJeyNphnDfl60Vs__ANh9IJM" /></figure>""", unsafe_allow_html=True)
        # graph TD
        #     subgraph "<span style='font-size:24px'>TransformerBlock (attn only)</span>"
        #         classDef empty width:0px,height:0px;
        #         classDef code color:red;

        #         resid_pre["resid_pre<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]---D[ ]:::empty-->|add|resid_post
                
        #         subgraph "<span style='font-size:24px'>attn &emsp; &emsp; &emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>"
        #             v
        #             q
        #             k
        #             attn_scores
        #             F
        #             pattern
        #             G
        #             z
        #             result
        #         end
        #         resid_pre-->|W_V|v["v<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, nhead, d_head)</code>"]---G[ ]:::empty-->|weighted avg of value vectors|z["z<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, nhead, d_head)</code>"] --> |W_O|result["result<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, nhead, d_model)</code>"] -->|sum over heads|attn_out["attn_out<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]---D
        #         resid_pre-->|W_Q|q["q<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, nhead, d_head)</code>"]---F[ ]:::empty-->|dot product along d_head, scale and mask|attn_scores["attn_scores<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(nhead, seqQ, seqK)</code>"]-->|softmax|pattern["pattern<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(nhead, seqQ, seqK)</code>"]---G
        #         resid_pre-->|W_K|k["k<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, nhead, d_head)</code>"]---F
                
        #         resid_post["resid_post<div style='border-top:2px solid black;margin-top:4px'></div><code style='color:red;font-size:12px'>(seq, d_model)</code>"]
                
        #     end

    with st.columns(1)[0]:
        st.markdown(r"""

#### Exercise - visualise attention patterns

Visualise the attention patterns for both layers of your model, on the following prompt:

```python
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
```

*(Note that we've run the model on the string `text`, rather than on tokens like we did previously when creating a cache - this is something that `HookedTransformer` allows.)*

Inspect the attention patterns. What do you notice about the attention heads? 

You should spot three relatively distinctive basic patterns, which occur in multiple heads. What are these patterns, and can you guess why they might be present?
""")
        st.info(r"""Reminder: rather than plotting inline, you can do the following, and then open in your browser from the left-hand file explorer menu of VSCode:

```python
html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
with open("attn_patterns.html", "w") as f:
    f.write(str(html))
```""")

        with st.expander("Solution"):
            st.markdown(r"""
We visualise attention patterns with the following code:

```python
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
    with open(f"layer_{layer}_attention.html", "w") as f:
        f.write(str(html))
```

We notice that there are three basic patterns which repeat quite frequently:

* `prev_token_heads`, which attend mainly to the previous token (e.g. head `0.7`)
* `current_token_heads`, which attend mainly to the current token (e.g. head `1.6`)
* `first_token_heads`, which attend mainly to the first token (e.g. head `0.9`, although this is a bit less clear-cut than the other two)

The `prev_token_heads` and `current_token_heads` are perhaps unsurprising, because words that are close together in a sequence probably have a lot more mutual information (i.e. we could get quite far using bigram or trigram prediction). 

The `first_token_heads` are a bit more surprising. The basic intuition here is that the first token in a sequence is often used as a resting or null position for heads that only sometimes activate.
""")
        st.info(r"""
Terminology note - we refer to particular attention heads using the syntax `layer.head`, e.g. `0.7` refers to the 7th head in the 0th layer. The other syntax you'll sometimes see is `L0H7`.
""")

    st.markdown(r"""

### Summarising attention patterns

Now that we've observed our three basic attention patterns, it's time to make detectors for those patterns!
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - write your own detectors

You should fill in the functions below, which act as detectors for particular types of heads. Validate your detectors by comparing these results to the visual attention patterns above - summary statistics on their own can be dodgy, but are much more reliable if you can validate it by directly playing with the data.

Tasks like this are useful, because we need to be able to take our observations / intuitions about what a model is doing, and translate these into quantitative measures. As the exercises proceed, we'll be creating some much more interesting tools and detectors!

Note - there's no objectively correct answer for which heads are doing which tasks, and which detectors can spot them. You should just try and come up with something plausible-seeming, which identifies the kind of behaviour you're looking for.
""")

        with st.expander("Hint"):
            st.markdown(r"""
Try and compute the average attention probability along the relevant tokens. For instance, you can get the tokens just below the diagonal by using `t.diagonal` with appropriate `offset` parameter:

```python
>>> arr = t.arange(9).reshape(3, 3)
>>> arr
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])

>>> arr.diagonal()
tensor([0, 4, 8])

>>> arr.diagonal(-1)
tensor([3, 7])
```

Remember that you should be using `cache["pattern", layer]` to get all the attention probabilities for a given layer, and then indexing on the 0th dimension to get the correct head.
""")
        with st.expander("Example solution for current_attn_detector (read if you're stuck)"):
            st.markdown(r"""
```python
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    current_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of diagonal elements
            current_attn_score = attention_pattern.diagonal().mean()
            if current_attn_score > 0.4:
                current_attn_heads.append(f"{layer}.{head}")
    return current_attn_heads
```

Note - choosing `0.4` as a threshold is a bit arbitrary, but it seems to work well enough. In this particular case, a threshold of `0.5` results in no head being classified as a current-token head.
""")

        st.markdown(r"""
```python
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    pass

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    pass

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    pass

if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
```

Compare the printouts to your attention visualisations above. Do they seem to make sense?

#### Bonus exercise - try different text

Try inputting different text, and see how stable your results are. Do you always get the same classifications for heads?
""")

    st.markdown(r"""
Now, it's time to turn our attention to induction heads.

## What are induction heads?

(Note: I use induction **head** to refer to the head in the second layer which attends to the 'token immediately after the copy of the current token', and induction **circuit** to refer to the circuit consisting of the composition of a **previous token head** in layer 0 and an **induction head** in layer 1)

[Induction heads](https://transformer-circuits.pub/2021/framework/index.html#induction-heads) are the first sophisticated circuit we see in transformers! And are sufficiently interesting that we wrote [another paper just about them](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
""")
    with st.expander("An aside on why induction heads are a big deal"):
        st.markdown(r"""

There's a few particularly striking things about induction heads:

* They develop fairly suddenly in a phase change - from about 2B to 4B tokens we go from no induction heads to pretty well developed ones. This is a striking divergence from a 1L model [see the training curves for this model vs a 1L one](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-65---VmlldzoyNTI0MDQx?accessToken=extt248d3qoxbqw1zy05kplylztjmx2uqaui3ctqb0zruem0tkpwrssq2ao1su3j) and can be observed in much larger models (eg a 13B one)
    * Phase changes are particularly interesting (and depressing) from an alignment perspective, because the prospect of a sharp left turn, or emergent capabilities like deception or situational awareness seems like worlds where alignment may be harder, and we get caught by surprise without warning shots or simpler but analogous models to test our techniques on.
* They are responsible for a significant loss decrease - so much so that there's a visible bump in the loss curve when they develop (this change in loss can be pretty comparable to the increase in loss from major increases in model size, though this is hard to make an apples-to-apples comparison)
* They seem to be responsible for the vast majority of in-context learning - the ability to use far back tokens in the context to predict the next token. This is a significant way in which transformers outperform older architectures like RNNs or LSTMs, and induction heads seem to be a big part of this.
* The same core circuit seems to be used in a bunch of more sophisticated settings, such as translation or few-shot learning - there are heads that seem clearly responsible for those *and* which double as induction heads
""")
        # st.markdown("")
    st.markdown(r"""
Again, you are strongly recommended to read the [corresponding section of the glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_Jzi6YHRHKP1JziwdE02qdYZ), before continuing (or [this LessWrong post](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated)). In brief, however, the induction circuit consists of a previous token head in layer 0 and an induction head in layer 1, where the induction head learns to attend to the token immediately *after* copies of the current token via K-Composition with the previous token head.
""")
    st.markdown("#### Question - why couldn't an induction head form in a 1L model?")
    with st.expander("Solution"):
        st.markdown(r"""
Because this would require a head which attends a key position based on the *value of the token before it*. Attention scores are just a function of the key token and the query token, and are not a function of other tokens.

(The attention pattern *does* in fact include effects from other tokens because of softmax - if another key token has a high attention score, softmax inhibits this pair. But this inhibition is symmetric across positions, so can't systematically favour the token *next* to the relevant one.)

Note that a key detail is that the value of adjacent tokens are (approximately) unrelated - if the model wanted to attend based on relative *position* this is easy.""")

    st.markdown(r"""
## Checking for the induction capability

A striking thing about models with induction heads is that, given a repeated sequence of random tokens, they can predict the repeated half of the sequence. This is nothing like it's training data, so this is kind of wild! The ability to predict this kind of out of distribution generalisation is a strong point of evidence that you've really understood a circuit.

To check that this model has induction heads, we're going to run it on exactly that, and compare performance on the two halves - you should see a striking difference in the per token losses.

Note - we're using small sequences (and just one sequence), since the results are very obvious and this makes it easier to visualise. In practice we'd obviously use larger ones on more subtle tasks. But it's often easiest to iterate and debug on small tasks.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - plot per-token loss on repeated sequence

You should fill in the function below (where it says `pass`). We've given you the first line of the function, where the prefix is defined (we need a prefix token, since the model was always trained to have one). See the next section for a deeper dive into tokenisation.

```python
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
```
""")
        with st.expander("Hint"):
            pass
        with st.expander("Click here to see the output you should be getting:"):
            st.plotly_chart(fig_dict["repeated_tokens"], use_container_width=False)
        with st.expander("Solution"):
            st.markdown(r"""
```python
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
```
""")

    st.markdown(r"""
### Looking for Induction Attention Patterns

The next natural thing to check for is the induction attention pattern.

First, go back to the attention patterns visualisation code from earlier (i.e. `cv.attention.attention_heads` or `attention_patterns`) and manually check for likely heads in the second layer. Which ones do you think might be serving as induction heads?

Note - we've defined `rep_str` for you, so you can use it in your `circuitsvis` functions.
""")
    with st.expander("What you should see (only click after you've made your own observations):"):
        st.markdown(r"""
The characteristic pattern of attention heads is a diagonal stripe, with the diagonal offset as `seq_len - 1` (because the destination token attends to the token *after* the destimation token's previous occurrence).

You should see that heads 4 and 10 are strongly induction-y, head 6 is very weakly induction-y, and the rest aren't.

For instance, here is `cv.attention.attention_patterns` for head 4. We can see how the `unfamiliar` token attends strongly to the `celebration` token, which follows the first occurrence of `unfamiliar`. 
""")
        st_image("attn_rep_pattern.png", 800)
        st.markdown("")
        st.markdown("")
        st.markdown(r"""
(Note - this is a good example of why `attention_patterns` can sometimes be a more helpful visualisation than `attention_heads` - it's easier to see precisely which token attends to which, whereas this information takes slightly longer to parse when you're looking at the grid.)""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - induction head detector

Now, you should make an induction pattern score function, which looks for the average attention paid to the offset diagonal. Do this in the same style as our earlier head scorers.

Remember, the offset in your diagonal should be `- (seq_len - 1)`, because the second instance of random token `T` will attend to the token **after** the first instance of `T`.

```python
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember:
        The tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    pass

if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
```

If this function works as expected, then you should see output that matches your observations from `circuitsvis`.""")

        with st.expander("Help - I'm getting basically every head classified as an induction head."):
            st.markdown(r"""
This might be because you're defining `seq_len` directly from the shape of the corresponding cache activations. Remember that the attention pattern activations will actually have shape `[2*seq_len+1, 2*seq_len+1]` (where `seq_len` is the length of the random tokens sequence).
""")

        with st.expander("Example solution"):
            st.markdown(r"""
```python
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
```""")

def section_hooks():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
<li><a class="contents-el" href="#what-are-hooks">What are hooks?</a></li>
<li><ul class="contents">
    <li><a class="contents-el" href="#hook-functions">Hook functions</a></li>
    <li><a class="contents-el" href="#running-with-hooks">Running with hooks</a></li>
    <li><a class="contents-el" href="#a-few-extra-notes-on-hooks-before-we-start-using-them">A few extra notes on hooks, before we start using them</a></li>
</ul></li>
<li><a class="contents-el" href="#hooks-accessing-activations">Hooks: Accessing Activations</a></li>
<li><a class="contents-el" href="#building-interpretability-tools">Building interpretability tools</a></li>
<li><ul class="contents">
    <li><a class="contents-el" href="#direct-logit-attribution">Direct Logit attribution</a></li>
    <li><a class="contents-el" href="#aside-typechecking">Aside - typechecking</a></li>
</ul></li>
<li><a class="contents-el" href="#hooks-intervening-on-activations">Hooks: Intervening on Activations</a></li>
<li><ul class="contents">
    <li><a class="contents-el" href="#ablations">Ablations</a></li>
    <li><a class="contents-el" href="#activation-patching-on-the-indirect-object-identification-task">Activation Patching on the Indirect Object Identification Task</a></li>
</ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Reverse-engineering induction circuits""")
    st.markdown(r"")
    st.info(r"""
## Learning Objectives

* Understand what hooks are, and how TransformerLens uses them
* Use hooks to access activations, process the results, and write them to an external tensor.
* Build tools to perform attribution, i.e. detecting which components of your model are responsible for performance on a given task.
* Use hooks to intervene on activations, specifically to perform **ablation** and **activation patching**.
    * Understand the importance of both these approaches for interpretability.
""")
    st.markdown(r"""
## What are hooks?

One of the great things about interpreting neural networks is that we have *full control* over our system. From a computational perspective, we know exactly what operations are going on inside (even if we don't know what they mean!). And we can make precise, surgical edits and see how the model's behaviour and other internals change. This is an extremely powerful tool, because it can let us e.g. set up careful counterfactuals and causal intervention to easily understand model behaviour. 

Accordingly, being able to do this is a pretty core operation, and this is one of the main things TransformerLens supports! The key feature here is **hook points**. Every activation inside the transformer is surrounded by a hook point, which allows us to edit or intervene on it. 

We do this by adding a **hook function** to that activation, and then calling `model.run_with_hooks`.

### Hook functions

Hook functions take two arguments: `activation_value` and `hook_point`. The `activation_value` is a tensor representing some activation in the model, just like the values in our `ActivationCache`. The `hook_point` is an object which gives us methods like `hook.layer()` or attributes like `hook.name` that are sometimes useful to call within the function.

If we're using hooks to edit activations, then the hook function should return a tensor of the same shape as the activation value. But we can also just have our hook function access the activation, do some processing, and write the results to some external variable (in which case our hook function should just not return anything).

An example hook function for changing the attention patterns at a particular layer might look like:

```python
def hook_function(
    attn_pattern: TT["batch", "heads", "seq_len", "seq_len"],
    hook: HookPoint
) -> TT["batch", "heads", "seq_len", "seq_len"]:

    # modify attn_pattern inplace
    return attn_pattern
```

### Running with hooks

Once you've defined a hook function (or functions), you should call `model.run_with_hooks`. A typical call to this function might look like:

```python
loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=[
        ('blocks.1.attn.hook_pattern', hook_function)
    ]
)
```

Let's break this code down.
* `tokens` represents our model's input.
* `return_type="loss"` is used here because we're modifying our activations and seeing how this affects the loss.
    * We could also return the logits, or just use `return_type=None` if we only want to access the intermediate activations and we don't care about the output.
* `fwd_hooks` is a list of 2-tuples of (hook name, hook function).
    * The hook name is a string that specifies which activation we want to hook. 
    * The hook function gets run with the corresponding activation as its first argument.

### A few extra notes on hooks, before we start using them""")

    st.markdown(r"""

Here are a few extra notes for how to squeeze even more functionality out of hooks. If you'd prefer, you can [jump ahead](#hooks-accessing-activations) to see an actual example of hooks being used, and come back to this section later.""", unsafe_allow_html=True)

    with st.expander("Resetting hooks"):
        st.markdown(r"""
`model.run_with_hooks` has the default parameter `reset_hooks_end=True` which resets all hooks at the end of the run (including both those that were added before and during the run).

Despite this, it's possible to shoot yourself in the foot with hooks, e.g. if there's an error in one of your hooks so the function never finishes. In this case, you can use `model.reset_hooks()` to reset all hooks.

Further, if you *do* want to keep the hooks that you add, you can do this by calling `add_hook` on the relevant `HookPoint`.
""")

    with st.expander("Adding multiple hooks at once"):
        st.markdown(r"""
Including more than one tuple in the `fwd_hooks` list is one way to add multiple hooks:

```python
loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=[
        ('blocks.0.attn.hook_pattern', hook_function),
        ('blocks.1.attn.hook_pattern', hook_function)
    ]
)
```

Another way is to use a **name filter** rather than a single name:

```python
loss = model.run_with_hooks(
    tokens,
    return_type="loss",
    fwd_hooks=[
        (lambda name: name.endswith("pattern"), hook_function)
    ]
)
```
""")
    with st.expander("utils.get_act_name"):
        st.markdown(r"""
When we were indexing the cache in the previous section, we found we could use strings like `cache['blocks.0.attn.hook_pattern']`, or use the shorthand of `cache['pattern', 0]`. The reason the second one works is that it calls the function `utils.get_act_name` under the hood, i.e. we have:

```python
utils.get_act_name('pattern', 0) == 'blocks.0.attn.hook_pattern'
```

Using `utils.get_act_name` in your forward hooks is often easier than using the full string, since the only thing you need to remember is the activation name (you can refer back to the diagram in the previous section for this).
""")

    with st.expander("Using functools.partial to create variations on hooks"):
        st.markdown(r"""
A useful trick is to define a hook function with more arguments than it needs, and then use `functools.partial` to fill in the extra arguments. For instance, if you want a hook function which only modifies a particular head, but you want to run it on all heads separately (rather than just adding all the hooks and having them all run on the next forward pass), then you can do something like:

```python
def hook_all_attention_patterns(
    attn_pattern: TT["batch", "heads", "seq_len", "seq_len"],
    hook: HookPoint,
    head_idx: int
) -> TT["batch", "heads", "seq_len", "seq_len"]:
    # modify attn_pattern inplace, at head_idx
    return attn_pattern

for head_idx in range(12):
    temp_hook_fn = functools.partial(hook_all_attention_patterns, head_idx=head_idx)
    model.run_with_hooks(tokens, fwd_hooks=[('blocks.1.attn.hook_pattern', temp_hook_fn)])
```""")

    with st.expander("Relationship to PyTorch hooks"):
        st.markdown(r"""
[PyTorch hooks](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/) are a great and underrated, yet incredibly janky, feature. They can act on a layer, and edit the input or output of that layer, or the gradient when applying autodiff. The key difference is that **Hook points** act on *activations* not layers. This means that you can intervene within a layer on each activation, and don't need to care about the precise layer structure of the transformer. And it's immediately clear exactly how the hook's effect is applied. This adjustment was shamelessly inspired by [Garcon's use of ProbePoints](https://transformer-circuits.pub/2021/garcon/index.html).

They also come with a range of other quality of life improvements. PyTorch's hooks are global state, which can be a massive pain if you accidentally leave a hook on a model. TransformerLens hooks are also global state, but `run_with_hooks` tries tries to create an abstraction where these are local state by removing all hooks at the end of the function (and they come with a helpful `model.reset_hooks()` method to remove all hooks).""")

    st.markdown(r"""
## Hooks: Accessing Activations

In later sections, we'll write some code to intervene on hooks, which is really the core feature that makes them so useful for interpretability. But for now, let's just look at how to access them without changing their value. This can be achieved by having the hook function write to a global variable, and return nothing (rather than modifying the activation in place).

Why might we want to do this? It turns out to be useful for things like:

* Extracting activations for a specific task
* Doing some long-running calculation across many inputs, e.g. finding the text that most activates a specific neuron

Note that, in theory, this could all be done using the `run_with_cache` function we used in the previous section, combined with post-processing of the cache result. But using hooks can be more intuitive and memory efficient.""")

    with st.columns(1)[0]:

        st.markdown(r"""
#### Exercise - calculate induction scores using hooks

To start with, we'll look at how hooks can be used to get the same results as from the previous section (where we ran our induction head detector functions on the values in the cache).

Most of the code has already been provided for you below; the only thing you need to do is **implement the `induction_score_hook` function**. As mentioned, this function takes two arguments: the activation value (which in this case will be our attention pattern) and the hook object (which gives us some useful methods and attributes that we can access in the function, e.g. `hook.layer()` to return the layer, or `hook.name` to return the name, which is the same as the name in the cache). 

Your function should do the following:

* Calculate the induction score for the attention pattern `pattern`, using the same methodology as you used in the previous section when you wrote your induction head detectors.
    * Note that this time, the batch dimension is greater than 1, so you should compute the average attention score over the batch dimension.
    * Also note that you are computing the induction score for all heads at once, rather than one at a time. You might find the arguments `dim1` and `dim2` of the `torch.diagonal` function useful.
* Write this score to the tensor `induction_score_store`, which is a global variable that we've provided for you. The `[i, j]`th element of this tensor should be the induction score for the `j`th head in the `i`th layer.

```python
# We make a tensor to store the induction score for each head. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score for each head in the corresponding layer, and stores it in the global induction_score_store tensor.
    '''
    pass

if MAIN:
    # Generate repeated tokens (no cache needed)
    seq_len = 50
    batch = 10
    prefix = t.ones((batch, 1), dtype=t.int64) * model.tokenizer.bos_token_id
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=1).to(device)

    # Make a boolean filter on activation names, that's true only on attention pattern names.
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[
            (pattern_hook_names_filter, induction_score_hook)
        ]
    )

    # Plot the induction score for each head in each layer
    imshow(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head", text_auto=".2f")
```

If this function has been implemented correctly, you should see a result matching your observations from the previous section: a high induction score for all the heads which you identified as induction heads, and a low score for all others.""")

        with st.expander("Help - I'm not sure how to implement this function."):
            st.markdown(r"""
To get the induction stripe, you can use:

```python
torch.diagonal(pattern, dim1=-2, dim2=-1, offset=1-seq_len)
```

since this returns the diagonal of each attention scores matrix, for every element in the batch and every attention head.

Once you have this, you can then take the mean over the batch and diagonal dimensions, giving you a tensor of length `n_heads`. You can then write this to the global `induction_score_store` tensor, using the `hook.layer()` method to get the correct row number.
    """)


        with st.expander("Solution"):
            st.markdown(r"""
```python
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score
```
""")
        with st.expander("Click to see what your result should look like (although the numbers will be slightly different due to randomness):"):
            st.plotly_chart(fig_dict["induction_scores"], use_container_width=True)
    st.markdown("")

    with st.columns(1)[0]:
        st.markdown(r"""        
#### Exercise - find induction heads in GPT2-small

Perform the same analysis on your `gpt2_small`. You should observe that some heads, particularly in a couple of the middle layers, have high induction scores. Use CircuitsVis to plot the attention patterns for these heads when run on the repeated token sequences, and verify that they look like induction heads.

Note - you can make CircuitsVis plots (and other visualisations) using hooks rather than plotting directly from the cache. For example, to plot the average attention over the batch dimension, you can use `model.run_with_hooks` with the following hook function:

```python
def visualize_pattern_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )
```
""")

    st.markdown(r"""

## Building interpretability tools

In order to develop a mechanistic understanding for how transformers perform certain tasks, we need to be able to answer questions like:

> *How much of the model's performance on some particular task is attributable to each component of the model?*

where "component" here might mean, for example, a specific head in a layer.

There are many ways to approach a question like this. For example, we might look at how a head interacts with other heads in different layers, or we might perform a causal intervention by seeing how well the model performs if we remove the effect of this head. However, we'll keep things simple for now, and ask the question: **what are the direct contributions of this head to the output logits?**

### Direct Logit attribution

A consequence of the residual stream is that the output logits are the sum of the contributions of each layer, and thus the sum of the results of each head. This means we can decompose the output logits into a term coming from each head and directly do attribution like this!
""")

    with st.expander("A concrete example"):

        st.markdown(r"""
Let's say that our model knows that the token Harry is followed by the token Potter, and we want to figure out how it does this. The logits on Harry are `W_U @ residual`. But this is a linear map, and the residual stream is the sum of all previous layers `residual = embed + attn_out_0 + attn_out_1`. So `logits = (W_U @ embed) + (W_U @ attn_out_0) + (W_U @ attn_out_1)`

We can be even more specific, and *just* look at the logit of the Potter token - this corresponds to a row of `W_U`, and so a direction in the residual stream - our logit is now a single number that is the sum of `(potter_U @ embed) + (potter_U @ attn_out_0) + (potter_U @ attn_out_1)`. Even better, we can decompose each attention layer output into the sum of the result of each head, and use this to get many terms.
""")

    st.markdown(r"""
Your mission here is to write a function to look at how much each component contributes to the correct logit. Your components are:

* The direct path (i.e. the residual connections from the embedding to unembedding),
* Each layer 0 head (via the residual connection and skipping layer 1)
* Each layer 1 head

To emphasise, these are not paths from the start to the end of the model, these are paths from the output of some component directly to the logits - we make no assumptions about how each path was calculated!

A few important notes for this exercise:

* Here we are just looking at the DIRECT effect on the logits, i.e. the thing that this component writes / embeds into the residual stream - if heads compose with other heads and affect logits like that, or inhibit logits for other tokens to boost the correct one we will not pick up on this!
* By looking at just the logits corresponding to the correct token, our data is much lower dimensional because we can ignore all other tokens other than the correct next one (Dealing with a 50K vocab size is a pain!). But this comes at the cost of missing out on more subtle effects, like a head suppressing other plausible logits, to increase the log prob of the correct one.
    * There are other situations where our job might be easier. For instance, in the IOI task (which we'll discuss shortly) we're just comparing the logits of the indirect object to the logits of the direct object, meaning we can use the **difference between these logits**, and ignore all the other logits.
* When calculating correct output logits, we will get tensors with a dimension `(position - 1,)`, not `(position,)` - we remove the final element of the output (logits), and the first element of labels (tokens). This is because we're predicting the *next* token, and we don't know the token after the final token, so we ignore it.
""")

    with st.expander("Aside:"):
        st.markdown(r"""
While we won't worry about this for this exercise, logit attribution is often more meaningful if we first center `W_U` - ie, ensure the mean of each row writing to the output logits is zero. Log softmax is invariant when we add a constant to all the logits, so we want to control for a head that just increases all logits by the same amount. We won't do this here for ease of testing.""")

    with st.expander("Question - why don't we do this to the log probs instead?"):
        st.markdown(r"""
Because log probs aren't linear, they go through `log_softmax`, a non-linear function.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - build logit attribution tool

You should implement the `logit_attribution` function below. This should return the contribution of each component in the "correct direction". We've already given you the unembedding vectors for the correct direction, `W_U_correct_tokens` (note that we take the `[1:]` slice of tokens, for reasons discussed above).

```python
def logit_attribution(embed, l1_results, l2_results, W_U, tokens) -> t.Tensor:
    '''
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
    pass
```

The code below will check your logit attribution function is working correctly, by taking the sum of logit attributions and comparing it to the actual values in the residual stream at the end of your model.

```python
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
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=5e-4, rtol=0)
```

If you're stuck, you can look at the solution below.""")

        with st.expander("Solution"):
            st.markdown(r"""
```python
def logit_attribution(embed, l1_results, l2_results, W_U, tokens) -> t.Tensor:
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = einsum("emb seq, seq emb -> seq", W_U_correct_tokens, embed[:-1])
    l1_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l1_results[:-1])
    l2_attributions = einsum("emb seq, seq nhead emb -> seq nhead", W_U_correct_tokens, l2_results[:-1])
    return t.cat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)
```
""")
    st.markdown(r"""

Once you've got the tests working, you can visualise the logit attributions for each path through the model.

```python
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
    imshow(utils.to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", height=25*len(tokens))

if MAIN:
    embed = cache["hook_embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
    plot_logit_attribution(logit_attr, tokens)
```""")

    with st.expander("Click here to see the output you should be getting."):
        st.plotly_chart(fig_dict["logit_attribution"], use_container_width=True)

    with st.columns(1)[0]:
        st.markdown(r"""

#### Exercise - interpret the results of this plot

You should find that the most variation in the logit attribution comes from the direct path. In particular, some of the tokens in the direct path have a very high logit attribution (e.g. tokens 12, 24 and 46). Can you guess what gives them in particular such a high logit attribution? 
""")
        with st.expander("Solution"):
            st.markdown(r"""
The tokens with very high logit attribution are the ones which "offer very probable bigrams". For instance, the highest contribution on the direct path comes from `| manip|`, because this is very likely to be followed by `|ulative|` (or presumably a different stem like `| ulation|`). `| super|` -> `|human|` is another example of a bigram formed when the tokenizer splits one word into multiple tokens.

There are also examples that come from two different words, rather than a single word split by the tokenizer. These include:

* `| more|` -> `| likely|`
* `| machine|` -> `| learning|`
* `| by|` -> `| default|`
* `| how|` -> `| to|`

See later for a discussion of all the ~infuriating~ fun quirks of tokenization!
""")

        st.markdown(r"""
Another feature of the plot - the heads in the second layer seem to have much higher contributions than the heads in the first layer. Why do you think this might be?""")
        with st.expander("Hint"):
            st.markdown(r"""
Think about what this graph actually represents, in terms of paths through the transformer.""")
        with st.expander("Solution"):
            st.markdown(r"""
This is because of a point we discussed earlier - this plot doesn't pick up on things like a head's effect in composition with another head. So the attribution for layer-0 heads won't involve any composition, whereas the attributions for layer-1 heads will involve not only the single-head paths through those attention heads, but also the 2-layer compositional paths through heads in layer 0 and layer 1.""")

    st.markdown("")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - logit attribution for the induction heads

Perform logit attribution for your attention-only model `model`, on the `rep_cache`. What do you expect to see?

Remember, you'll need to split the sequence in two, with one overlapping token (since predicting the next token involves removing the final token with no label) - your `logit_attr` should both have shape `[seq_len, 2*n_heads + 1]` (ie `[50, 25]` here).""")

        with st.expander("Note - the first plot will be pretty meaningless. Can you see why?"):
            st.markdown(r"""
Because the first plot shows the logit attribution for the first half of the sequence, i.e. the first occurrence of each of the tokens. Since there is no structure to this sequence (it is purely random), there is no reason to expect the heads to be doing meaningful computation. The structure lies in the second half of the sequence, when the tokens are repeated, and the heads with high logit attributions will be the ones that can perform induction.
""")

        st.markdown(r"""
```python
if MAIN:
    seq_len = 50

    embed = rep_cache["hook_embed"]
    l1_results = rep_cache["blocks.0.attn.hook_result"]
    l2_results = rep_cache["blocks.1.attn.hook_result"]
    first_half_tokens = rep_tokens[0, :seq_len+1]
    second_half_tokens = rep_tokens[0, seq_len:]
    
    "YOUR CODE HERE:"
    "Define `first_half_logit_attr` and `second_half_logit_attr`"

    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    
    plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    plot_logit_attribution(second_half_logit_attr, second_half_tokens)
```""")

        with st.expander("Solution"):
            st.markdown(r"""
```python
first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)
```
""")
        with st.expander("Click here to see the output you should be getting."):
            st.markdown(r"""
We only show the second plot (since as discussed, the first plot is meaningless).
""")
            st.plotly_chart(fig_dict["rep_logit_attribution"], use_container_width=True)

        st.markdown(r"""What is the interpretation of this plot, in the context of our induction head circuit?""")
        with st.expander("Answer"):
            st.markdown(r"""
Previously, we observed that heads `1.4` and `1.10` seemed to be acting as induction heads.

This plot gives further evidence that this is the case, since these two heads have a large logit attribution score **on sequences in which the only way to get accurate predictions is to use the induction mechanism**.

This also agrees with our attention scores result, in showing tht `1.10` is a stronger induction head than `1.4`.""")

    st.markdown(r"""

### Aside - typechecking

Typechecking is a useful habit to get into. It's not strictly necessary, but it can be a great help when you're debugging.

One good way to typecheck in PyTorch is with the `torchtyping`. The most important object in this library is the `TensorType` object, which can be used to specify things like the shape and dtype of a tensor.

You can also use the `typeguard.typechecked` to strictly enforce the type signatures of your inputs and outputs. For instance, if you replaced the `plot_logit_attribution` function with the following:

```python
from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def plot_logit_attribution(logit_attr: TT["seq", "path"], tokens: TT["seq"]):
    ...
```

then you would get an error when running this function, if the 0th dimension of `logit_attr` didn't match the length of `tokens`. (Note, it's necessary to call `patch_typeguard()` once before you use the `typechecked` decorator, but you don't have to call it any more after that.)

You can do other things with `TorchTyping`, such as:
* Specify values for dimensions, e.g. `TT["batch", 512, "embed"]`
* Specify values and names, e.g. `TT["batch", 512, "embed": 768]`
* Specify dtypes, e.g. `TT["batch", 512, t.float32]` checks the tensor has shape `(?, 512)` and dtype `torch.float32`.

You can read more [here](https://github.com/patrick-kidger/torchtyping).
""")

    st.markdown(r"""
## Hooks: Intervening on Activations

Now that we've built some tools to decompose our model's output, it's time to start making causal interventions.
""")

    st.markdown(r"""
### Ablations

Let's start with a simple example: **ablation**. An ablation is a simple causal intervention on a model - we pick some part of it and set it to zero. This is a crude proxy for how much that part matters. Further, if we have some story about how a specific circuit in the model enables some capability, showing that ablating *other* parts does nothing can be strong evidence of this.

As mentioned in [the glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=fh-HJyz1CgUVrXuoiban6bYx), there are many ways to do ablation. We'll focus on the simplest: zero-ablation (even though it's somewhat unprincipled).""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - zero-ablation for induction heads

The code below provides a template for performing zero-ablation on the value vectors at a particular head (i.e. the vectors we get when applying the weight matrices `W_V` to the residual stream). If you're confused about what different activations mean, you can refer back to the diagram:""")

        with st.expander("Attention-only diagram"):
            st.write("""<figure style="max-width:620px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrNVsFu2zAM_RVBh7UDYnQLdnKyHIasOfQwFCu2Q10EikXHhmVJkWQ3Sd1_H-U4dR2kQLFDEx0kUibtR_pR1BONFQca0qVhOiV300gSHLZc7DYiOraaSWLdRsD3i0RJF9hsC-Hwm15fTO4MkzZRpgDzQ6g4J5fMOUmUFJvP4yvvOYno7pV-xIJZO4WEQKHdhjxm3KXhF70epJAtU-fF0RFrDxEnoUxogKNFZ2PAZnyuDdxH9EUe86zaI14ow8EETulwqNfEKpFxshAszkcFM8tMNo-aYMZX6DcZN19rvbuPdpF_HXrjSwurAeHzAq0Fxuq9MNaHIAim9-QhDMMmxiCY1IzzusWmrOuwd9J7090k9xMUVo_6y7G9dyz_Nx_5sX5UfXXVV_O-6iOZ21hhXvoPrvuqRkMwsr8566vbvoqvLMWrNIPkR_jif8zf-Z-6QuJUH0cYmQLjnjd-7dNmdkCbx6YmgBNWLYlKSMVECaSC2Clj6y3i3p4eN0GkBBP5q96lfVeGKJwA2mEpemy1LQuiKjDEW9m64Z0qPc69eOrT4k1y3tYrhLk6C3JeH5CTK0e0UbyMHWFCyWXrNSA2ZgIIk5wUzOb1q0Lf53ynfUhYbUQY3W0z3_SiQnaoxBVsXbfHDCJspTNAF8zeZMZNnSPU_DyYcayhde2u68uonLLUDmFiW6ADiheXgmUcL0BPfjuiLoUCIhqiyCFhvpNgn3tG01Jz5uAnz_D8pWHChIUBZaVTvzcypqEzJeyNphnDfl60Vs__ANh9IJM" /></figure>""", unsafe_allow_html=True)

        st.markdown(r"""

The only thing left for you to do is fill in the function `head_ablation_hook` so that it performs zero-ablation on the head given by `head_index_to_ablate`. In other words, your function should return a modified version of `value` with this head ablated. (Technically you don't have to return any tensor if you're modifying `value` in-place; this is just convention.)

A few notes to help explain the code below:

* In the `get_ablation_scores` function, we run our hook function in a for loop: once for each layer and each head. Each time, we write a single value to the tensor `ablation_scores` that stores the results.
* We use `cross_entropy_loss` as a metric for model performance. If the head didn't have any effect on the output, then the ablation score would be zero (since the loss doesn't increase when we ablate). If the head was very important for making correct predictions, then we should see a very large ablation score.
* We use `functools.partial` to create a temporary hook function with the head number fixed. This is a nice way to avoid having to write a separate hook function for each head.
* We use `utils.get_act_name` to get the name of our activation. This is a nice shorthand way of getting the full name.

```python
def head_ablation_hook(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    pass

def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).
    '''
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
    imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads", text_auto=".2f")
```
""")

        with st.expander("Click here to see the output you should be getting."):
            st.plotly_chart(fig_dict["logit_diff_rep"], use_container_width=True)
        
        st.markdown("What is your interpretation of these results?")
        with st.expander("Interpretation:"):
            st.markdown(r"""
This tells us not just which heads are responsible for writing output to the residual stream that gets us the correct result, but **which heads play an important role in the induction circuit**. 

This chart tells us that head `0.7` is by far the most important in layer 0 (which makes sense, since we observed it to be the strongest "previous token head"), and heads `1.4`, `1.10` are the most important in layer 1 (which makes sense, since we observed these to be the most induction-y).

This is a good illustration of the kind of result which we can get from ablation, but **wouldn't be able to get from something like direct logit attribution**.
""")

    st.success(r"""
#### Bonus Exercise

Try ablating *every* head apart from the previous token head and the two induction heads. What does this do to performance?

What if you mean ablate it, rather than zero ablating it?""")

    st.markdown(r"""

### Activation Patching on the Indirect Object Identification Task

For a somewhat more involved example, let's use hooks to apply **activation patching** on the **Indirect Object Identification** (IOI) task. 

The IOI task is the task of identifying that a sentence like "After John and Mary went to the store, Mary gave a bottle of milk to" with " John" rather than " Mary" (ie, finding the indirect object), and Redwood Research have [an excellent paper studying the underlying circuit in GPT-2 Small](https://arxiv.org/abs/2211.00593).

**Activation patching** is a technique from [Kevin Meng and David Bau's excellent ROME paper](https://rome.baulab.info/). The goal is to identify which model activations are important for completing a task. We do this by setting up a **clean prompt** and a **corrupted prompt** and a **metric** for performance on the task. We then pick a specific model activation, run the model on the corrupted prompt, but then *intervene* on that activation and patch in its value when run on the clean prompt. We then apply the metric, and see how much this patch has recovered the clean performance. 
(See [a more detailed explanation of activation patching here](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Exploratory_Analysis_Demo.ipynb#scrollTo=5nUXG6zqmd0f))

We're returning to our `gpt2_small` for this example.

```python
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

clean_tokens = gpt2_small.to_tokens(clean_prompt)
corrupted_tokens = gpt2_small.to_tokens(corrupted_prompt)

def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
    # model.to_single_token maps a string value of a single token to the token index for that token
    # If the string is not a single token, it raises an error.
    correct_index = gpt2_small.to_single_token(correct_answer)
    incorrect_index = gpt2_small.to_single_token(incorrect_answer)
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]
```
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - what does this function do?

Explain in your own words what the function `logits_to_logit_diff` does.

How do you expect the output of this function to differ if the `logits` are from the clean prompt vs the corrupted prompt?
""")

        with st.expander("Explanation"):
            st.markdown(r"""
`logits[0, -1]` represents the logit output for the final token in the prompt, i.e. the model's prediction for what token comes after `to`. For the clean prompt, the correct answer is `" John"` and the incorrect answer is `" Mary"`.

So this function is taking the logit difference (equivalent to the log likelihood ratio) between `" John"` and `" Mary"`. **This is our metric for the model's performance on the task.**

If the logits are from the clean prompt, we expect the output to be positive, since the model should think that `" John"` is more likely than `" Mary"`. If the logits are from the corrupted prompt, we expect the output to be negative.
""")

        st.markdown(r"""
Now run the code below, and verify that your hypothesis is correct.

```python
# We run on the clean prompt with the cache so we store activations to patch in later.
clean_logits, clean_cache = gpt2_small.run_with_cache(clean_tokens, remove_batch_dim=True)
clean_logit_diff = logits_to_logit_diff(clean_logits)
print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

# We don't need to cache on the corrupted prompt.
corrupted_logits = gpt2_small(corrupted_tokens)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
```
""")

    st.markdown(r"""

We now setup the hook function to do **activation patching**. Here, we'll patch in the residual stream at the start of a specific layer and at a specific position. This will let us see how much the model is using the residual stream at that layer and position to represent the key information for the task. 

A few notes to help explain the code below:

* Our patching works by running the model on the **corrupted prompt**, but then replacing the activations at the specified layer and position with the activations from the **clean prompt**.
* The function `residual_stream_patching_hook` acts on the residual stream at the **start** of the layer, which is why we've called it `resid_pre`. The residual stream at the end of the layer is called `resid_post`.
* We use `hook.name` to access the activation name, so we know which activation we're patching in from the clean cache.
* `clean_resid_pre` has no batch dimension (because we specified `remove_batch_dim=True` when creating our cache), but `resid_pre` does (because by default activations in a model do have a batch dimension).
* Like before, when we actually run our model with hooks, we use `functools.partial` to construct a hook function to act on a specfic position.
* At the end, we store the result in the tensor `ioi_patching_result`. We've normalized the result, so that:
    * If the patching has no effect, then the result is zero (since the patched logit diff is just the same as the corrupted logit diff).
    * If the patching has "full effect" (i.e. making the output the same as it would have been on the clean prompt), then the result is 1.
    * Note that this doesn't actually guarantee that the results will always be in the range `[0, 1]`, but in this case it's very likely, because any patching will "move us from the corrupted logit output towards the clean logit output".
""")
    st.markdown("")
    st.markdown(r"""
```python
def residual_stream_patching_hook(
    resid_pre: TT["batch", "pos", "d_model"],
    hook: HookPoint,
    position: int,
) -> TT["batch", "pos", "d_model"]:
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[position, :]
    return resid_pre

num_positions = len(clean_tokens[0])
ioi_patching_result = t.zeros((gpt2_small.cfg.n_layers, num_positions), device=gpt2_small.cfg.device)

for layer in tqdm(range(gpt2_small.cfg.n_layers)):
    for position in range(num_positions):
        temp_hook_fn = functools.partial(residual_stream_patching_hook, position=position)
        # Run the model with the patching hook
        patched_logits = gpt2_small.run_with_hooks(corrupted_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        ])
        # Calculate the logit difference
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        # Store the normalized result
        ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)
```

We can now visualize the results, and see that this computation is extremely localised within the model. Initially, the second subject (Mary) token is all that matters (naturally, as it's the only different token), and all relevant information remains here until heads in layer 7 and 8 move this to the final token where it's used to predict the indirect object.

(Note - the heads are in layer 7 and 8, not 8 and 9, because we patched in the residual stream at the *start* of each layer).

```python
# Add the index to the end of the label, because plotly doesn't like duplicate labels
token_labels = [f"{token}_{index}" for index, token in enumerate(gpt2_small.to_str_tokens(clean_tokens))]
imshow(ioi_patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")
```""")

    st.plotly_chart(fig_dict["logit_diff_from_patching"], use_container_width=False)

    with st.expander("Help - I'm still confused about this diagram."):
        st.markdown(r"""
The interpretation of the $n$th **row** of the heatmap is as follows:

> If we run the **corrupted prompt** through the model (i.e. the one where `Mary` is the correct answer), but intervene by replacing the residual stream **before** the start of the $n$th layer with the values produced by the **clean prompt** (where `John` is the correct answer), then how far does this move the output towards predicting `John` instead? 
>
> (where a value of 0 in the heatmap indicates no movement of the output towards `John`, and a value of 1 indicates we now predict `John` as strongly as we would have on the clean prompt).

So the fact that the first row is zero everywhere except for the `Mary` token is trivial, because intervening at this point is just equivalent to feeding the clean prompt into the model instead.

Similarly, we expect that near the end the rows will be zero everywhere except for the `to` token, because the `to` token holds the information which will eventually result in a prediction of either John or Mary. So there must come a point where this token holds all the information it needs to make this prediction, and no other tokens will affect it.

The more interesting parts are the middle rows. They tells us that:

1. **The computation is very localised.**
    * In other words, basically only one sequence position holds the "John vs Mary" information at any one time. 
    * The exception is in the middle rows, where it takes two layers for most of the information to be transferred from the `Mary` token to the `to` token.
2. **The information path is very direct.**
    * The "John vs Mary" information moves from the `Mary` token to the `to` token without any detours.
    * In principle, we could imagine a situation where the information path is less direct, e.g. the majority of the information passes from `Mary` to `gave`, and then from `gave` to `to`. But that's not what's happening here.

One other note - all the values to the left of Mary are zero, whereas the values to the right aren't. This is because we're using causal attention, which means information can only flow forwards. So the residual streams for the clean and corrupted prompt can only differ on or after the first token which is different between the two prompts.
""")
        st.markdown("")
#     st.markdown(r"""
# A few extra questions about this diagram, to make sure you understand it:""")
#     with st.expander("All the values to the left of Mary are zero, whereas the values to the right aren't. Why is this?"):
#         st.markdown(r"""
# The attention is causal, or to put this another way, information in the model only ever flows forwards. So the output at token `T` can't be different for the clean and corrupted prompt, if `T` comes before the first token which is different between the two prompts.
# """)
    

def section_other_features():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#available-models">Available Models</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#an-overview-of-the-important-open-source-models-in-the-library">An overview of the important open source models in the library</a></li>
        <li><a class="contents-el" href="#an-overview-of-some-interpretability-friendly-models-i-ve-trained-and-included">An overview of some interpretability-friendly models I've trained and included</a></li>
    </ul></li>
    <li><a class="contents-el" href="#other-resources">Other Resources:</a></li>
    <li><a class="contents-el" href="#transformer-architecture">Transformer architecture</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#parameter-names">Parameter Names</a></li>
        <li><a class="contents-el" href="#activation--hook-names">Activation + Hook Names</a></li>
        <li><a class="contents-el" href="#folding-layernorm-for-the-curious">Folding LayerNorm (For the Curious)</a></li>
    </ul></li>
   <li><a class="contents-el" href="#dealing-with-tokens">Dealing with tokens</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#gotcha-prepend-bos">Gotcha: <code>prepend_bos</code></a></li>
   </ul></li>
   <li><a class="contents-el" href="#factored-matrix-class">Factored Matrix Class</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#basic-examples">Basic Examples</a></li>
       <li><a class="contents-el" href="#medium-example-eigenvalue-copying-scores">Medium Example: Eigenvalue Copying Scores</a></li>
   </ul></li>
   <li><a class="contents-el" href="#generating-text">Generating Text</a></li>
   <li><a class="contents-el" href="#hook-points">Hook Points</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#toy-example">Toy Example</a></li>
   </ul></li>
   <li><a class="contents-el" href="#loading-pre-trained-checkpoints">Loading Pre-Trained Checkpoints</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#example-induction-head-phase-transition">Example: Induction Head Phase Transition</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# TransformerLens: Other Features

## Available Models

TransformerLens comes with over 40 open source models available, all of which can be loaded into a consistent(-ish) architecture by just changing the name in `from_pretrained`. You can see [a table of the available models here](https://github.com/neelnanda-io/TransformerLens/blob/main/easy_transformer/model_properties_table.md), including the hyper-parameters and the alias used to load the model in TransformerLens.

**Note:** Though TransformerLens can load in some large transformers (eg OPT-66B), it does not currently support loading a single model across multiple GPUs, so in practice loading a model (eg >7B parameters) will be impractical, depending on your GPU memory. Feel free to reach out if you need to use larger models for your research.

Notably, this means that analysis can be near immediately re-run on a different model by just changing the name.

#### Exercise - re-run your analysis on a different model

Load in DistilGPT-2 (a distilled version of GPT-2, with half as many layers), with the following code: 

```python
distilgpt2 = HookedTransformer.from_pretrained("distilgpt2")
```

Take some analysis you did for GPT-2 small (e.g. locating induction heads), and re-run the exact same code on DistilGPT-2. Verify that it works without any further modification needed.""")


    st.markdown(r"""
### An overview of the important open source models in the library

* **GPT-2** - the classic generative pre-trained models from OpenAI
    * Sizes Small (85M), Medium (300M), Large (700M) and XL (1.5B).
    * Trained on ~22B tokens of internet text. ([Open source replication](https://huggingface.co/datasets/openwebtext))
* **GPT-Neo** - Eleuther's replication of GPT-2
    * Sizes 125M, 1.3B, 2.7B
    * Trained on 300B(ish?) tokens of [the Pile](https://pile.eleuther.ai/) a large and diverse dataset including a bunch of code (and weird stuff)
* **[OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)** - Meta AI's series of open source models
    * Trained on 180B tokens of diverse text.
    * 125M, 1.3B, 2.7B, 6.7B, 13B, 30B, 66B
* **GPT-J** - Eleuther's 6B parameter model, trained on the Pile
* **GPT-NeoX** - Eleuther's 20B parameter model, trained on the Pile
* **Stanford CRFM models** - a replication of GPT-2 Small and GPT-2 Medium, trained on 5 different random seeds.
    * Notably, 600 checkpoints were taken during training per model, and these are available in the library with eg `HookedTransformer.from_pretrained("stanford-gpt2-small-a", checkpoint_index=265)`.


### An overview of some interpretability-friendly models I've trained and included

(Feel free to [reach out](mailto:neelnanda27@gmail.com) if you want more details on any of these models)

Each of these models has about ~200 checkpoints taken during training that can also be loaded from TransformerLens, with the `checkpoint_index` argument to `from_pretrained`.

Note that all models are trained with a Beginning of Sequence token, and will likely break if given inputs without that! 

* **Toy Models**: Inspired by [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html), I've trained 12 tiny language models, of 1-4L and each of width 512. I think that interpreting these is likely to be far more tractable than larger models, and both serve as good practice and will likely contain motifs and circuits that generalise to far larger models (like induction heads):
    * Attention-Only models (ie without MLPs): attn-only-1l, attn-only-2l, attn-only-3l, attn-only-4l
    * GELU models (ie with MLP, and the standard GELU activations): gelu-1l, gelu-2l, gelu-3l, gelu-4l
    * SoLU models (ie with MLP, and [Anthropic's SoLU activation](https://transformer-circuits.pub/2022/solu/index.html), designed to make MLP neurons more interpretable): solu-1l, solu-2l, solu-3l, solu-4l
    * All models are trained on 22B tokens of data, 80% from C4 (web text) and 20% from Python Code
    * Models of the same layer size were trained with the same weight initialization and data shuffle, to more directly compare the effect of different activation functions.
* **SoLU** models: A larger scan of models trained with [Anthropic's SoLU activation](https://transformer-circuits.pub/2022/solu/index.html), in the hopes that it makes the MLP neuron interpretability easier. 
    * A scan up to GPT-2 Medium size, trained on 30B tokens of the same data as toy models, 80% from C4 and 20% from Python code. 
        * solu-6l (40M), solu-8l (100M), solu-10l (200M), solu-12l (340M)
    * An older scan up to GPT-2 Medium size, trained on 15B tokens of [the Pile](https://pile.eleuther.ai/)
        * solu-1l-pile (13M), solu-2l-pile (13M), solu-4l-pile (13M), solu-6l-pile (40M), solu-8l-pile (100M), solu-10l-pile (200M), solu-12l-pile (340M)

## Other Resources:

* [Concrete Open Problems in Mechanistic Interpretability](https://docs.google.com/document/d/1WONBzNqfKIxERejrrPlQMyKqg7jSFW92x5UMXNrMdPo/edit), a doc I wrote giving a long list of open problems in mechanistic interpretability, and thoughts on how to get started on trying to work on them. 
    * There's a lot of low-hanging fruit in the field, and I expect that many people reading this could use TransformerLens to usefully make progress on some of these!
* Other demos:
    * **[Exploratory Analysis Demo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/Exploratory_Analysis_Demo.ipynb)**, a demonstration of my standard toolkit for how to use TransformerLens to explore a mysterious behaviour in a language model.
    * [Interpretability in the Wild](https://github.com/redwoodresearch/Easy-Transformer) a codebase from Arthur Conmy and Alex Variengien at Redwood research using this library to do a detailed and rigorous reverse engineering of the Indirect Object Identification circuit, to accompany their paper
        * Note - this was based on an earlier version of this library, called EasyTransformer. It's pretty similar, but several breaking changes have been made since. 
    * A [recorded walkthrough](https://www.youtube.com/watch?v=yo4QvDn-vsU) of me doing research with TransformerLens on whether a tiny model can re-derive positional information, with [an accompanying Colab](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/No_Position_Experiment.ipynb)
* [Neuroscope](https://neuroscope.io), a website showing the text in the dataset that most activates each neuron in some selected models. Good to explore to get a sense for what kind of features the model tends to represent, and as a "wiki" to get some info
    * A tutorial on how to make an [Interactive Neuroscope](https://github.com/neelnanda-io/TransformerLens/blob/main/Hacky-Interactive-Lexoscope.ipynb), where you type in text and see the neuron activations over the text update live.

### Folding LayerNorm (For the Curious)

(For the curious - this is an important technical detail that's worth understanding, especially if you have preconceptions about how transformers work, but not necessary to use TransformerLens)

LayerNorm is a normalization technique used by transformers, analogous to BatchNorm but more friendly to massive parallelisation. No one *really* knows why it works, but it seems to improve model numerical stability. Unlike BatchNorm, LayerNorm actually changes the functional form of the model, which makes it a massive pain for interpretability! 

Folding LayerNorm is a technique to make it lower overhead to deal with, and the flags `center_writing_weights` and `fold_ln` in `HookedTransformer.from_pretrained` apply this automatically (they default to True). These simplify the internal structure without changing the weights.

Intuitively, LayerNorm acts on each residual stream vector (ie for each batch element and token position) independently, sets their mean to 0 (centering) and standard deviation to 1 (normalizing) (*across* the residual stream dimension - very weird!), and then applies a learned elementwise scaling and translation to each vector.

Mathematically, centering is a linear map, normalizing is *not* a linear map, and scaling and translation are linear maps. 
* **Centering:** LayerNorm is applied every time a layer reads from the residual stream, so the mean of any residual stream vector can never matter - `center_writing_weights` set every weight matrix writing to the residual to have zero mean. 
* **Normalizing:** Normalizing is not a linear map, and cannot be factored out. The `hook_scale` hook point lets you access and control for this.
* **Scaling and Translation:** Scaling and translation are linear maps, and are always followed by another linear map. The composition of two linear maps is another linear map, so we can *fold* the scaling and translation weights into the weights of the subsequent layer, and simplify things without changing the underlying computation. 

[See the docs for more details](https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#what-is-layernorm-folding-fold_ln)

A fun consequence of LayerNorm folding is that it creates a bias across the unembed, a `d_vocab` length vector that is added to the output logits - GPT-2 is not trained with this, but it *is* trained with a final LayerNorm that contains a bias. 

Turns out, this LayerNorm bias learns structure of the data that we can only see after folding! In particular, it essentially learns **unigram statistics** - rare tokens get suppressed, common tokens get boosted, by pretty dramatic degrees! Let's list the top and bottom 20 - at the top we see common punctuation and words like " the" and " and", at the bottom we see weird-ass tokens like " RandomRedditor":

```python
unembed_bias = model.unembed.b_U
bias_values, bias_indices = unembed_bias.sort(descending=True)
```

```python
top_k = 20
print(f"Top {top_k} values")
for i in range(top_k):
    print(f"{bias_values[i].item():.2f} {repr(model.to_string(bias_indices[i]))}")

print("...")
print(f"Bottom {top_k} values")
for i in range(top_k, 0, -1):
    print(f"{bias_values[-i].item():.2f} {repr(model.to_string(bias_indices[-i]))}")
```

This can have real consequences for interpretability - for example, this bias favours " John" over " Mary" by about 1.2, about 1/3 of the effect size of the Indirect Object Identification Circuit! All other things being the same, this makes the John token 3.6x times more likely than the Mary token.

```python
john_bias = model.unembed.b_U[model.to_single_token(' John')]
mary_bias = model.unembed.b_U[model.to_single_token(' Mary')]

print(f"John bias: {john_bias.item():.4f}")
print(f"Mary bias: {mary_bias.item():.4f}")
print(f"Prob ratio bias: {t.exp(john_bias - mary_bias).item():.4f}x")
```

## Dealing with tokens

**Tokenization** is one of the most annoying features of studying language models. We want language models to be able to take in arbitrary text as input, but the transformer architecture needs the inputs to be elements of a fixed, finite vocabulary. The solution to this is **tokens**, a fixed vocabulary of "sub-words", that any natural language can be broken down into with a **tokenizer**. This is invertible, and we can recover the original text, called **de-tokenization**. 

TransformerLens comes with a range of utility functions to deal with tokenization. Different models can have different tokenizers, so these are all methods on the model.

`get_token_position`, `to_tokens`, `to_string`, `to_str_tokens`, `prepend_bos`, `to_single_token`

The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph.

Some observations - there are a lot of arbitrary-ish details in here!
* The tokenizer splits on spaces, so no token contains two words.
* Tokens include the preceding space, and whether the first token is a capital letter. `'how'` and `' how'` are different tokens!
* Common words are single tokens, even if fairly long (` paragraph`) while uncommon words are split into multiple tokens (` token|ized`).
* Tokens *mostly* split on punctuation characters (eg `*` and `.`), but eg `'s` is a single token.

```python
example_text = "The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph."
example_text_str_tokens = model.to_str_tokens(example_text)
print(example_text_str_tokens)
```

The transformer needs to take in a sequence of integers, not strings, so we need to convert these tokens into integers. `model.to_tokens` does this, and returns a tensor of integers on the model's device (shape `[batch, position]`). It maps a string to a batch of size 1.

```python
example_text_tokens = model.to_tokens(example_text)
print(example_text_tokens)
```

`to_tokens` can also take in a list of strings, and return a batch of size `len(strings)`. If the strings are different numbers of tokens, it adds a PAD token to the end of the shorter strings to make them the same length.

(Note: In GPT-2, 50256 signifies both the beginning of sequence, end of sequence and padding token - see the `prepend_bos` section for details)

```python
example_multi_text = ["The cat sat on the mat.", "The cat sat on the mat really hard."]
example_multi_text_tokens = model.to_tokens(example_multi_text)
print(example_multi_text_tokens)
```

`model.to_single_token` is a convenience function that takes in a string corresponding to a *single* token and returns the corresponding integer. This is useful for eg looking up the logit corresponding to a single token. 

For example, let's input `The cat sat on the mat.` to GPT-2, and look at the log prob predicting that the next token is ` The`. 
""")
    with st.expander("Technical notes"):
        st.markdown(r"""
Note that if we input a string to the model, it's implicitly converted to a string with `to_tokens`. 

Note further that the log probs have shape `[batch, position, d_vocab]==[1, 8, 50257]`, with a vector of log probs predicting the next token for *every* token position. GPT-2 uses causal attention which means heads can only look backwards (equivalently, information can only move forwards in the model.), so the log probs at position k are only a function of the first k tokens, and it can't just cheat and look at the k+1 th token. This structure lets it generate text more efficiently, and lets it treat every *token* as a training example, rather than every *sequence*.""")

    st.markdown(r"""
```python
cat_text = "The cat sat on the mat."
cat_logits = model(cat_text)
cat_probs = cat_logits.softmax(dim=-1)
print(f"Probability tensor shape [batch, position, d_vocab] == {cat_probs.shape}")

capital_the_token_index = model.to_single_token(" The")
print(f"| The| probability: {cat_probs[0, -1, capital_the_token_index].item():.2%}")
```

`model.to_string` is the inverse of `to_tokens` and maps a tensor of integers to a string or list of strings. It also works on integers and lists of integers.

For example, let's look up token 256 (due to technical details of tokenization, this will be the most common pair of ASCII characters!), and also verify that our tokens above map back to a string.

```python
print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")
```

A related annoyance of tokenization is that it's hard to figure out how many tokens a string will break into. `model.get_token_position(single_token, tokens)` returns the position of `single_token` in `tokens`. `tokens` can be either a string or a tensor of tokens. 

Note that position is zero-indexed, it's two (ie third) because there's a beginning of sequence token automatically prepended (see the next section for details)

```python
print("With BOS:", model.get_token_position(" cat", "The cat sat on the mat"))
print("Without BOS:", model.get_token_position(" cat", "The cat sat on the mat", prepend_bos=False))
```

If there are multiple copies of the token, we can set `mode="first"` to find the first occurence's position and `mode="last"` to find the last

```python
print("First occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="first"))
print("Final occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="last"))
```

In general, tokenization is a pain, and full of gotchas. I highly recommend just playing around with different inputs and their tokenization and getting a feel for it. As another "fun" example, let's look at the tokenization of arithmetic expressions - tokens do *not* contain consistent numbers of digits. (This makes it even more impressive that GPT-3 can do arithmetic!)

```python
print(model.to_str_tokens("2342+2017=21445"))
print(model.to_str_tokens("1000+1000000=999999"))
```

I also *highly* recommend investigating prompts with easy tokenization when starting out - ideally key words should form a single token, be in the same position in different prompts, have the same total length, etc. Eg study Indirect Object Identification with common English names like ` Tim` rather than ` Ne|el`. Transformers need to spend some parameters in early layers converting multi-token words to a single feature, and then de-converting this in the late layers, and unless this is what you're explicitly investigating, this will make the behaviour you're investigating be messier.

### Gotcha: `prepend_bos`

##### Key Takeaway: **If you get weird off-by-one errors, check whether there's an unexpected `prepend_bos`!**

A weirdness you may have noticed in the above is that `to_tokens` and `to_str_tokens` added a weird `<|endoftext|>` to the start of each prompt. TransformerLens does this by default, and it can easily trip up new users. Notably, **this includes `model.forward`** (which is what's implicitly used when you do eg `model("Hello World")`). This is called a **Beginning of Sequence (BOS)** token, and it's a special token used to mark the beginning of the sequence. Confusingly, in GPT-2, the End of Sequence (EOS), Beginning of Sequence (BOS) and Padding (PAD) tokens are all the same, `<|endoftext|>` with index `50256`.

You can disable this behaviour by setting the flag `prepend_bos=False` in `to_tokens`, `to_str_tokens`, `model.forward` and any other function that converts strings to multi-token tensors. 

**Gotcha:** You only want to do this at the *start* of a prompt. If you, eg, want to input a question followed by an answer, and want to tokenize these separately, you do *not* want to prepend_bos on the answer.

```python
print("Logits shape by default (with BOS)", model("Hello World").shape)
print("Logits shape with BOS", model("Hello World", prepend_bos=True).shape)
print("Logits shape without BOS - only 2 positions!", model("Hello World", prepend_bos=False).shape)
```

`prepend_bos` is a bit of a hack, and I've gone back and forth on what the correct default here is. The reason I do this is that transformers tend to treat the first token weirdly - this doesn't really matter in training (where all inputs are >1000 tokens), but this can be a big issue when investigating short prompts! The reason for this is that attention patterns are a probability distribution and so need to add up to one, so to simulate being "off" they normally look at the first token. Giving them a BOS token lets the heads rest by looking at that, preserving the information in the first "real" token.

Further, *some* models are trained to need a BOS token (OPT and my interpretability-friendly models are, GPT-2 and GPT-Neo are not). But despite GPT-2 not being trained with this, empirically it seems to make interpretability easier.

For example, the model can get much worse at Indirect Object Identification without a BOS (and with a name as the first token):

```python
ioi_logits_with_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=True)
mary_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference with BOS: {(claire_logit_with_bos - mary_logit_with_bos):.3f}")

ioi_logits_without_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=False)
mary_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference without BOS: {(claire_logit_without_bos - mary_logit_without_bos):.3f}")
```

Though, note that this also illustrates another gotcha - when `Claire` is at the start of a sentence (no preceding space), it's actually *two* tokens, not one, which probably confuses the relevant circuit. (Note - in this test we put `prepend_bos=False`, because we want to analyse the tokenization of a specific string, not to give an input to the model!)

```
print(f"| Claire| -> {model.to_str_tokens(' Claire', prepend_bos=False)}")
print(f"|Claire| -> {model.to_str_tokens('Claire', prepend_bos=False)}")
```

## Generating Text

TransformerLens also has basic text generation functionality, which can be useful for generally exploring what the model is capable of (thanks to Ansh Radhakrishnan for adding this!). This is pretty rough functionality, and where possible I recommend using more established libraries like HuggingFace for this.

```python
model.generate("(CNN) President Barack Obama caught in embarrassing new scandal\n", max_new_tokens=50, temperature=0.7, prepend_bos=True)
```

## Hook Points

The key part of TransformerLens that lets us access and edit intermediate activations are the HookPoints around every model activation. Importantly, this technique will work for *any* model architecture, not just transformers, so long as you're able to edit the model code to add in HookPoints! This is essentially a lightweight library bundled with TransformerLens that should let you take an arbitrary model and make it easier to study. 

This is implemented by having a HookPoint layer. Each transformer component has a HookPoint for every activation, which wraps around that activation. The HookPoint acts as an identity function, but has a variety of helper functions that allows us to put PyTorch hooks in to edit and access the relevant activation. 

There is also a `HookedRootModule` class - this is a utility class that the root module should inherit from (root module = the model we run) - it has several utility functions for using hooks well, notably `reset_hooks`, `run_with_cache` and `run_with_hooks`. 

The default interface is the `run_with_hooks` function on the root module, which lets us run a forwards pass on the model, and pass on a list of hooks paired with layer names to run on that pass. 

The syntax for a hook is `function(activation, hook)` where `activation` is the activation the hook is wrapped around, and `hook` is the `HookPoint` class the function is attached to. If the function returns a new activation or edits the activation in-place, that replaces the old one, if it returns None then the activation remains as is.

### Toy Example

Here's a simple example of defining a small network with HookPoints:

We define a basic network with two layers that each take a scalar input $x$, square it, and add a constant:
$x_0=x$, $x_1=x_0^2+3$, $x_2=x_1^2-4$.

We wrap the input, each layer's output, and the intermediate value of each layer (the square) in a hook point.

```python
from transformer_lens.hook_points import HookedRootModule, HookPoint

class SquareThenAdd(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = nn.Parameter(t.tensor(offset))
        self.hook_square = HookPoint()

    def forward(self, x):
        # The hook_square doesn't change the value, but lets us access it
        square = self.hook_square(x * x)
        return self.offset + square

class TwoLayerModel(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.layer1 = SquareThenAdd(3.0)
        self.layer2 = SquareThenAdd(-4.0)
        self.hook_in = HookPoint()
        self.hook_mid = HookPoint()
        self.hook_out = HookPoint()

        # We need to call the setup function of HookedRootModule to build an
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, x):
        # We wrap the input and each layer's output in a hook - they leave the
        # value unchanged (unless there's a hook added to explicitly change it),
        # but allow us to access it.
        x_in = self.hook_in(x)
        x_mid = self.hook_mid(self.layer1(x_in))
        x_out = self.hook_out(self.layer2(x_mid))
        return x_out

model = TwoLayerModel()
```

We can add a cache, to save the activation at each hook point

(There's a custom `run_with_cache` function on the root module as a convenience, which is a wrapper around model.forward that return model_out, cache_object - we could also manually add hooks with `run_with_hooks` that store activations in a global caching dictionary. This is often useful if we only want to store, eg, subsets or functions of some activations.)

```python
out, cache = model.run_with_cache(t.tensor(5.0))
print("Model output:", out.item())
for key in cache:
    print(f"Value cached at hook {key}", cache[key].item())
```

We can also use hooks to intervene on activations - eg, we can set the intermediate value in layer 2 to zero to change the output to -5

```python
def set_to_zero_hook(tensor, hook):
    print(hook.name)
    return t.tensor(0.0)

print(
    "Output after intervening on layer2.hook_scaled",
    model.run_with_hooks(
        t.tensor(5.0), fwd_hooks=[("layer2.hook_square", set_to_zero_hook)]
    ).item(),
)
```

## Loading Pre-Trained Checkpoints

There are a lot of interesting questions combining mechanistic interpretability and training dynamics - analysing model capabilities and the underlying circuits that make them possible, and how these change as we train the model. 

TransformerLens supports these by having several model families with checkpoints throughout training. `HookedTransformer.from_pretrained` can load a checkpoint of a model with the `checkpoint_index` (the label 0 to `num_checkpoints-1`) or `checkpoint_value` (the step or token number, depending on how the checkpoints were labelled).


Available models:
* All of my interpretability-friendly models have checkpoints available, including:
    * The toy models - `attn-only`, `solu`, `gelu` 1L to 4L
        * These have ~200 checkpoints, taken on a piecewise linear schedule (more checkpoints near the start of training), up to 22B tokens. Labelled by number of tokens seen.
    * The SoLU models trained on 80% Web Text and 20% Python Code (`solu-6l` to `solu-12l`)
        * Same checkpoint schedule as the toy models, this time up to 30B tokens
    * The SoLU models trained on the pile (`solu-1l-pile` to `solu-12l-pile`)
        * These have ~100 checkpoints, taken on a linear schedule, up to 15B tokens. Labelled by number of steps.
        * The 12L training crashed around 11B tokens, so is truncated.
* The Stanford Centre for Research of Foundation Models trained 5 GPT-2 Small sized and 5 GPT-2 Medium sized models (`stanford-gpt2-small-a` to `e` and `stanford-gpt2-medium-a` to `e`)
    * 600 checkpoints, taken on a piecewise linear schedule, labelled by the number of steps.

The checkpoint structure and labels is somewhat messy and ad-hoc, so I mostly recommend using the `checkpoint_index` syntax (where you can just count from 0 to the number of checkpoints) rather than `checkpoint_value` syntax (where you need to know the checkpoint schedule, and whether it was labelled with the number of tokens or steps). The helper function `get_checkpoint_labels` tells you the checkpoint schedule for a given model - ie what point was each checkpoint taken at, and what type of label was used.

Here are graphs of the schedules for several checkpointed models: (note that the first 3 use a log scale, latter 2 use a linear scale)

```python
from transformer_lens.loading_from_pretrained import get_checkpoint_labels
for model_name in ["attn-only-2l", "solu-12l", "stanford-gpt2-small-a"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(checkpoint_labels, xaxis="Checkpoint Index", yaxis=f"Checkpoint Value ({checkpoint_label_type})", title=f"Checkpoint Values for {model_name} (Log scale)", log_y=True, markers=True)
for model_name in ["solu-1l-pile", "solu-6l-pile"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(checkpoint_labels, xaxis="Checkpoint Index", yaxis=f"Checkpoint Value ({checkpoint_label_type})", title=f"Checkpoint Values for {model_name} (Linear scale)", log_y=False, markers=True)
```

### Example: Induction Head Phase Transition

One of the more interesting results analysing circuit formation during training is the [induction head phase transition](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). They find a pretty dramatic shift in models during training - there's a brief period where models go from not having induction heads to having them, which leads to the models suddenly becoming much better at in-context learning (using far back tokens to predict the next token, eg over 500 words back). This is enough of a big deal that it leads to a visible *bump* in the loss curve, where the model's rate of improvement briefly increases. 

As a brief demonstration of the existence of the phase transition, let's load some checkpoints of a two layer model, and see whether they have induction heads. An easy test, as we used above, is to give the model a repeated sequence of random tokens, and to check how good its loss is on the second half. `evals.induction_loss` is a rough util that runs this test on a model.
(Note - this is deliberately a rough, non-rigorous test for the purposes of demonstration, eg `evals.induction_loss` by default just runs it on 4 sequences of 384 tokens repeated twice. These results totally don't do the paper justice - go check it out if you want to see the full results!)

In the interests of time and memory, let's look at a handful of checkpoints (chosen to be around the phase change), indices `[10, 25, 35, 60, -1]`. These are roughly 22M, 200M, 500M, 1.6B and 21.8B tokens through training, respectively. (I generally recommend looking things up based on indices, rather than checkpoint value!). 

```python
from transformer_lens import evals
# We use the two layer model with SoLU activations, chosen fairly arbitrarily as being both small (so fast to download and keep in memory) and pretty good at the induction task.
model_name = "solu-2l"
# We can load a model from a checkpoint by specifying the checkpoint_index, -1 means the final checkpoint
checkpoint_indices = [10, 25, 35, 60, -1]
checkpointed_models = []
tokens_trained_on = []
induction_losses = []
```

We load the models, cache them in a list, and 

```python
for index in checkpoint_indices:
    # Load the model from the relevant checkpoint by index
    model_for_this_checkpoint = HookedTransformer.from_pretrained(model_name, checkpoint_index=index)
    checkpointed_models.append(model_for_this_checkpoint)

    tokens_seen_for_this_checkpoint = model_for_this_checkpoint.cfg.checkpoint_value
    tokens_trained_on.append(tokens_seen_for_this_checkpoint)

    induction_loss_for_this_checkpoint = evals.induction_loss(model_for_this_checkpoint).item()
    induction_losses.append(induction_loss_for_this_checkpoint)
```

We can plot this, and see there's a sharp shift from ~200-500M tokens trained on (note the log scale on the x axis). Interestingly, this is notably earlier than the phase transition in the paper, I'm not sure what's up with that.

(To contextualise the numbers, the tokens in the random sequence are uniformly chosen from the first 20,000 tokens (out of ~48,000 total), so random performance is at least $\ln(20000)\approx 10$. A naive strategy like "randomly choose a token that's already appeared in the first half of the sequence (384 elements)" would get $\ln(384)\approx 5.95$, so the model is doing pretty well here.)

```python
line(induction_losses, x=tokens_trained_on, xaxis="Tokens Trained On", yaxis="Induction Loss", title="Induction Loss over training: solu-2l", markers=True, log_x=True)
```
""")

def section_reverse_engineering():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#refresher-do-you-understand-the-circuit">Refresher - do you understand the induction circuit?</a></li>
    <li><a class="contents-el" href="#refresher-qk-and-ov-circuits-and-some-terminology">Refresher - QK and OV circuits</a></li>
    <li><a class="contents-el" href="#factored-matrix-class">Factored Matrix class</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#basic-examples">Basic Examples</a></li>
    </ul></li>
    <li><a class="contents-el" href="#reverse-engineering-circuits">Reverse-engineering circuits</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#ov-copying-circuit">OV copying circuit</a></li>
        <li><a class="contents-el" href="#qk-prev-token-circuit">QK prev-token circuit</a></li>
        <li><a class="contents-el" href="#k-composition">K-composition</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#splitting-activations">Splitting activations</a></li>
            <li><a class="contents-el" href="#composition-analysis">Composition Analysis</a></li>
            <li><a class="contents-el" href="#interpreting-the-full-circuit">Interpreting the full circuit</a></li>
        </ul></li>
    </ul></li>
    <li><a class="contents-el" href="#further-exploration-of-induction-circuits">Further Exploration of Induction Circuits</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#composition-scores">Composition scores</a></li>
    </ul></li>
    <li><a class="contents-el" href="#bonus">Bonus</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#looking-for-circuits-in-real-llms">Looking for Circuits in Real LLMs</a></li>
        <li><a class="contents-el" href="#training-your-own-toy-models">Training Your Own Toy Models</a></li>
        <li><a class="contents-el" href="#interpreting-induction-heads-during-training">Interpreting Induction Heads During Training</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Reverse-engineering induction circuits""")
    st.markdown(r"")
    st.info(r"""
## Learning Objectives

* Understand the difference between investigating a circuit by looking at activtion patterns, and reverse-engineering a circuit by looking directly at the weights.
* Use the factored matrix class to inspect the QK and OV circuits within an induction circuit.
* Perform further exploration of induction circuits: composition scores, and targeted ablations.
* Optional bonus exercises to investigate induction heads still further (e.g. looking for them in larger models).
""")

    st.markdown(r"""
In previous exercises, we looked at the attention patterns and attributions of attention heads to try and identify which ones were important in the induction circuit. This might be a good way to get a feel for the circuit, but it's not a very rigorous way to understand it. It would be better described as **feature analysis**, where we observe *that* a particular head seems to be performing some task on a certain class of inputs, without identifying *why* it does so.

Now we're going to do some more rigorous mechanistic analysis - digging into the weights and using them to reverse engineer the induction head algorithm and verify that it is really doing what we think it is.

## Refresher - do you understand the induction circuit?

Before we get into the meat of this section, let's refresh the results we've gotten so far from investigating induction heads. We've found:

* When fed repeated sequences of tokens, heads `1.4` and `1.10` have the characteristic induction head attention pattern of a diagonal stripe with offset `seq_len - 1`. 
    * We saw this both from the CircuitsVis results, and from the fact that these heads had high induction scores by our chosen metric (with all other heads having much lower scores).
* We also saw that head `0.7` strongly attends to the previous token in the sequence (even on non-repeated sequences).
* We performed **logit attribution** on the model, and found that the values written to the residual stream by heads `1.4` and `1.10` were both important for getting us correct predictions in the second half of the sequence.
* We performed **zero-ablation** on the model, and found that heads `0.7`, `1.4` and `1.10` all resulted in a large accuracy degradation on the repeated sequence task when they were ablated.

Based on all these observations, try and summarise the induction circuit and how it works, in your own words. You should try and link your explanation to the QK and OV circuits for particular heads, and describe what type (or types) of attention head composition are taking place.

You can use the dropdown below to check your understanding.
""")

    with st.expander("My summary of the algorithm"):
        st.markdown(r"""
* Head `0.7` is a previous token head (the QK-circuit ensures it always attends to the previous token).
* The OV circuit of head `0.7` writes a copy of the previous token in a *different* subspace to the one used by the embedding.
* The output of head `0.7` is used by the *key* input of head `1.4` via K-Composition to attend to 'the source token whose previous token is the destination token'.
* The OV-circuit of head `1.4` copies the *value* of the source token to the same output logit.
    * Note that this is copying from the embedding subspace, *not* the `0.7` output subspace - it is not using V-Composition at all.

To emphasise - the sophisticated hard part is computing the *attention* pattern of the induction head - this takes careful composition. The previous token and copying parts are fairly easy. This is a good illustrative example of how the QK circuits and OV circuits act semi-independently, and are often best thought of somewhat separately. And that computing the attention patterns can involve real and sophisticated computation!")""")

    st.markdown(r"""
## Refresher - QK and OV circuits (and some terminology)

Before we start, a brief terminology note. I'll refer to weight matrices for a particular layer and head using superscript notation, e.g. $W_Q^{1.4}$ is the query matrix for the 4th head in layer 1, and it has shape `[d_model, d_head]` (remember that we multiply with weight matrices on the right). Similarly, attention patterns will be denoted $A^{1.4}$ (remember that these are **activations**, not parameters, since they're given by the formula $A^h = x_i^T W_{QK}^h x_j$, where $x_i$ and $x_j$ are vectors in the residual stream). 

As a shorthand, I'll often have $A$ denote the one-hot encoding of token `A` (i.e. the vector with zeros everywhere except a one at the index of `A`), and $\color{red}A\color{black} := A^T W_E$ will denote the embedding vector for `A`.

Lastly, I'll refer to special matrix products as follows:

* $W_{OV}^{h} := W_V^{h}W_O^{h}$ is the **OV circuit** for head $h$, and $W_E W_{OV}^h W_U$ is the **full OV circuit**. 
* $W_{QK}^h := W_Q^h {W_K^{hT}}$ is the **QK circuit** for head $h$, and $W_E W_{QK}^h W_E^T$ is the **full QK circuit**. 

Note that the order of these matrices are slightly different from the **Mathematical Frameworks** paper - this is a consequence of the way TransformerLens stores its weight matrices.
""")
    st.markdown("")

    st.markdown(r"""

#### Question - what is the interpretation of each of the following four matrices?

In your answers, you should describe the type of input it takes, and what the outputs represent.

#### $W_{OV}^{h}$
""")

    with st.expander("Answer"):
        st.markdown(r"""
$W_{OV}^{h}$ has size $(d_\text{model}, d_\text{model})$, it is a linear map describing **what information gets moved from source to destination, in the residual stream.**

In other words, if $x_j$ is the embedding vector at position $j$, then $x_j^T W_{OV}^{h}$ is the vector written to the residual stream at the destination position, if the destination token only pays attention to $x_j$.
""")
    st.markdown(r"""
#### $W_E W_{OV}^h W_U$
""")

    with st.expander("Answer"):
        st.markdown(r"""
$W_E W_{OV}^h W_U$ has size $(d_\text{vocab}, d_\text{vocab})$, it is a linear map describing **what information gets moved from source to destination, in a start-to-end sense.**

If $A$ is the one-hot encoding for token `A`, then $A^T W_E W_{OV}^h W_U$ is the vector of logits output by head $h$ at any token which pays attention to `A`.
""")

    st.markdown(r"""
#### $W_{QK}^{h}$
""")
    with st.expander("Answer"):
        st.markdown(r"""
$W_{QK}^{h}$ has size $(d_\text{model}, d_\text{model})$, it is a bilinear form describing **where information is moved to and from** in the residual stream (i.e. which residual stream vectors attend to which others).

$x_i^T W_{QK}^h x_j$ is the attention score paid by token $i$ to token $j$.
""")
    st.markdown(r"""
#### $W_E W_{QK}^h W_E^T$
""")

    with st.expander("Answer"):
        st.markdown(r"""
$W_E W_{QK}^h W_E^T$ has size $(d_\text{vocab}, d_\text{vocab})$, it is a bilinear form describing **where information is moved to and from**, among words in our vocabulary (i.e. which tokens pay attention to which others).

If $A$ and $B$ are one-hot encodings for tokens `A` and `B`, then $A^T W_E W_{QK}^h W_E^T B$ is the attention score paid by token `A` to token `B`.
""")

    st.markdown(r"""



Before we start, there's a problem that we might run into when calculating all these matrices. Some of them are massive, and might not fit on our GPU. For instance, both full circuit matrices have shape $(d_\text{vocab}, d_\text{vocab})$, which in our case means $50278\times 50278 \approx 2.5\times 10^{9}$ elements. If you're doing these exercises in Colab or using a powerful GPU then you might manage, but anything short of that will probably result in a `RuntimeError: CUDA error: out of memory` error. How can we get around this problem?

## Factored Matrix class

In transformer interpretability, we often need to analyse low rank factorized matrices - a matrix $M = AB$, where M is `[large, large]`, but A is `[large, small]` and B is `[small, large]`. This is a common structure in transformers. 

For instance, we can factorise the OV circuit above as $W_{OV}^h = W_V^h W_O^h$, where $W_V^h$ has shape `[768, 64]` and $W_O^h$ has shape `[64, 768]`. Even more extreme - the full OV circuit can be written as $(W_E W_V^h) (W_O^h W_U)$, where these two matrices have shape `[50278, 64]` and `[64, 50278]` respectively. 

The `FactoredMatrix` class is a convenient way to work with these. It implements efficient algorithms for various operations on these, such as computing the trace, eigenvalues, Frobenius norm, singular value decomposition, and products with other matrices. It can (approximately) act as a drop-in replacement for the original matrix.

This is all possible because knowing the factorisation of a matrix gives us a much easier way of computing its important properties. Intuitively, since $M=AB$ is a very large matrix that operates on very small subspaces, we shouldn't expect knowing the actual values $M_{ij}$ to be the most efficient way of storing it!
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - deriving properties of a factored matrix

*Note - if you're less interested in the maths, you can skip these exercises.*

To give you an idea of what kinds of properties you can easily compute if you have a factored matrix, let's try and derive some ourselves.

Suppose we have $M=AB$, where $A$ has shape $(m, n)$, $B$ has shape $(n, m)$, and $m > n$. So $M$ is a size-$(m, m)$ matrix with rank at most $n$.

**Question - how can you easily compute the trace of $M$?**
""")
        with st.expander("Answer"):
            st.markdown(r"""
The solution lies in the fact that trace is cyclic:

$$
\text{Tr}(M) = \text{Tr}(AB)
= \sum_{i,j=1}^n A_{ij} B_{ji}
= \sum_{i,j=1}^n B_{ji} A_{ij}
= \text{Tr}(BA)
$$

$AB$ is an $(m, m)$-matrix, but $BA$ is $(n, n)$ (much smaller). So we can just find the trace of $BA$ instead.
""")

        st.markdown(r"""
**Question - how can you easily compute the eigenvalues of $M$?**
    """)
        with st.expander("Hint"):
            st.markdown(r"""
It's computationally cheaper to find the eigenvalues of $BA$ rather than $AB$.

How are the eigenvalues of $AB$ and $BA$ related?
""")
        with st.expander("Answer"):
            st.markdown(r"""
The eigenvalues of $AB$ and $BA$ are related as follows: if $\mathbf{v}$ is an eigenvector of $AB$ with $ABv = \lambda \mathbf{v}$, then $B\mathbf{v}$ is an eigenvector of $BA$ with the same eigenvalue:

$$
BA(B\mathbf{v}) = B (AB\mathbf{v}) = B (\lambda \mathbf{v}) = \lambda (B\mathbf{v})
$$

This only fails when $B\mathbf{v} = \mathbf{0}$, but in this case $AB\mathbf{v} = \mathbf{0}$ so $\lambda = 0$. Thus, we can conclude that any non-zero eigenvalues of $AB$ are also eigenvalues of $BA$.

It's much computationally cheaper to compute the eigenvalues of $BA$ (since it's a much smaller matrix), and this gives us all the non-zero eigenvalues of $AB$.
""")
        st.markdown(r"""
**Question (hard) - how can you easily compute the SVD of $M$?**
    """)

        with st.expander("Hint"):
            st.markdown(r"""
For a size-$(m, n)$ matrix with $m > n$, the [algorithmic complexity of finding SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition#Numerical_approach) is $O(mn^2)$. So it's relatively cheap to find the SVD of $A$ and $B$ (complexity $mn^2$ vs $m^3$). Can you use that to find the SVD of $M$?
    """)
        with st.expander("Solution"):
            st.markdown(r"""
It's much cheaper to compute the SVD of the small matrices $A$ and $B$. Denote these SVDs by:

$$
\begin{aligned}
A &= U_A S_A V_A^T \\
B &= U_B S_B V_B^T
\end{aligned}
$$

where $U_A$ and $V_B$ are $(m, n)$, and the other matrices are $(n, n)$.

Then we have:

$$
\begin{aligned}
M &= AB \\
&= U_A (S_A V_A^T U_B S_B) V_B^T
\end{aligned}
$$

Note that the matrix in the middle has size $(n, n)$ (i.e. small), so we can compute its SVD easily:

$$
\begin{aligned}
S_A V_A^T U_B S_B &= U' S' {V'}^T
\end{aligned}
$$

and finally, this gives us the SVD of $M$:

$$
\begin{aligned}
M &= U_A U' S' {V'}^T V_B^T \\
&= U S {V'}^T
\end{aligned}
$$

where $U = U_A U'$, $V = V_B V'$, and $S = S' S_B$.

All our SVD calculations and matrix multiplications had complexity at most $O(mn^2)$, which is much better than $O(m^3)$.
""")
        st.markdown(r"""
If you're curious, you can go to the `FactoredMatrix` documentation to see the implementation of the SVD calculation, as well as other properties and operations.""")
    st.markdown(r"""
Now that we've discussed some of the motivations behind having a `FactoredMatrix` class, let's see it in action.

### Basic Examples

We can use the basic class directly - let's make a factored matrix directly and look at the basic operations:

```python
A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")
```

We can also look at the eigenvalues and singular values of the matrix. Note that, because the matrix is rank 2 but 5 by 5, the final 3 eigenvalues and singular values are zero - the factored class omits the zeros.

```python
print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)
```

We can multiply with other matrices - it automatically chooses the smallest possible dimension to factor along (here it's 2, rather than 5).

```python
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
```

If we want to collapse this back to an unfactored matrix, we can use the `AB` property to get the product:

```python
AB_unfactored = AB_factor.AB
t.testing.assert_close(AB_unfactored, AB)
```
""")

#     st.error("TODO - add this section back in at the end.")
#     with st.expander("Medium Example: Eigenvalue Copying Scores"):
#         st.markdown(r"""
# ### Medium Example: Eigenvalue Copying Scores

# (This is a more involved example of how to use the factored matrix class, skip it if you aren't following)

# For a more involved example, let's look at the eigenvalue copying score from [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html) of the OV circuit for various heads. The OV Circuit for a head (the factorised matrix $W_OV = W_V W_O$) is a linear map that determines what information is moved from the source position to the destination position. Because this is low rank, it can be thought of as *reading in* some low rank subspace of the source residual stream and *writing to* some low rank subspace of the destination residual stream (with maybe some processing happening in the middle).

# A common operation for this will just be to *copy*, ie to have the same reading and writing subspace, and to do minimal processing in the middle. Empirically, this tends to coincide with the OV Circuit having (approximately) positive real eigenvalues. I mostly assert this as an empirical fact, but intuitively, operations that involve mapping eigenvectors to different directions (eg rotations) tend to have complex eigenvalues. And operations that preserve eigenvector direction but negate it tend to have negative real eigenvalues. And "what happens to the eigenvectors" is a decent proxy for what happens to an arbitrary vector.

# We can get a score for "how positive real the OV circuit eigenvalues are" with $\frac{\sum \lambda_i}{\sum |\lambda_i|}$, where $\lambda_i$ are the eigenvalues of the OV circuit. This is a bit of a hack, but it seems to work well in practice.

# Let's use FactoredMatrix to compute this for every head in the model! We use the helper `model.OV` to get the concatenated OV circuits for all heads across all layers in the model. This has the shape `[n_layers, n_heads, d_model, d_model]`, where `n_layers` and `n_heads` are batch dimensions and the final two dimensions are factorised as `[n_layers, n_heads, d_model, d_head]` and `[n_layers, n_heads, d_head, d_model]` matrices.

# We can then get the eigenvalues for this, where there are separate eigenvalues for each element of the batch (a `[n_layers, n_heads, d_head]` tensor of complex numbers), and calculate the copying score.

# ```python
# OV_circuit_all_heads = model.OV
# print(OV_circuit_all_heads)

# OV_circuit_all_heads_eigenvalues = OV_circuit_all_heads.eigenvalues 
# print(OV_circuit_all_heads_eigenvalues.shape)
# print(OV_circuit_all_heads_eigenvalues.dtype)

# OV_copying_score = OV_circuit_all_heads_eigenvalues.sum(dim=-1).real / OV_circuit_all_heads_eigenvalues.abs().sum(dim=-1)
# imshow(utils.to_numpy(OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)
# ```""")

#         st.plotly_chart(fig_dict["ov_copying"], use_container_width=True)
#         st.markdown(r"""

# Head 11 in Layer 11 (L11H11) has a high copying score, and if we plot the eigenvalues they look approximately as expected.

# ```python
# scatter(x=OV_circuit_all_heads_eigenvalues[-1, -1, :].real, y=OV_circuit_all_heads_eigenvalues[-1, -1, :].imag, title="Eigenvalues of Head L11H11 of GPT-2 Small", xaxis="Real", yaxis="Imaginary")
# ```""")

#         st.plotly_chart(fig_dict["scatter_evals"], use_container_width=True)
#         st.markdown(r"""

# We can even look at the full OV circuit, from the input tokens to output tokens: $W_E W_V W_O W_U$. This is a `[d_vocab, d_vocab]==[50257, 50257]` matrix, so absolutely enormous, even for a single head. But with the FactoredMatrix class, we can compute the full eigenvalue copying score of every head in a few seconds.""")

#         st.error("This code gives a CUDA error - it will be fixed shortly.")
#         st.markdown(r"""

# ```python
# full_OV_circuit = model.embed.W_E @ OV_circuit_all_heads @ model.unembed.W_U
# print(full_OV_circuit)

# full_OV_circuit_eigenvalues = full_OV_circuit.eigenvalues
# print(full_OV_circuit_eigenvalues.shape)
# print(full_OV_circuit_eigenvalues.dtype)

# full_OV_copying_score = full_OV_circuit_eigenvalues.sum(dim=-1).real / full_OV_circuit_eigenvalues.abs().sum(dim=-1)
# imshow(utils.to_numpy(full_OV_copying_score), xaxis="Head", yaxis="Layer", title="OV Copying Score for each head in GPT-2 Small", zmax=1.0, zmin=-1.0)
# ```

# Interestingly, these are highly (but not perfectly!) correlated. I'm not sure what to read from this, or what's up with the weird outlier heads!

# ```python
# scatter(x=full_OV_copying_score.flatten(), y=OV_copying_score.flatten(), hover_name=[f"L{layer}H{head}" for layer in range(12) for head in range(12)], title="OV Copying Score for each head in GPT-2 Small", xaxis="Full OV Copying Score", yaxis="OV Copying Score")
# ```

# ```python
# print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# # Squeeze means to remove dimensions of length 1. 
# # Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# # Rank 2 tensors map to a list of strings
# print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")
# ```""")

    st.markdown(r"""

## Reverse-engineering circuits

Within our induction circuit, we have four individual circuits: the OV and QK circuits in our previous token head, and the OV and QK circuits in our attention head. They are all shown in the diagram below (each of the boxes with dashed borders explains a different one of the four). 
""")

    st_image("kcomp_diagram.png", 900)
    st.markdown("")

    st.markdown(r"""
The eagle-eyed among you may have noticed that the contents page only has three subsections: the OV copying circuit, the QK prev-token circuit, and K-composition. Why three sections, but four circuits?

The reason is because we can't actually analyse the layer-0 OV circuit or the layer-1 QK circuit in isolation. The OV circuit writes information to a subspace (named the "prev token subspace" below) which is then used by the QK circuit to form the attention patterns in the induction head. So we have to analyse the circuit formed by composing these two heads.

As an analogy, pretend each of the attention heads is a person reading coded messages from the airwaves, and sending coded messages forwards to people further down the line. We might not be able to crack the code that Alice and Bob are using to communicate with each other, but as long as we can (1) show that they are communicating with each other more than with anyone else, and (2) deduce how they function as a single unit (i.e. how our coded message changes when it passes through Alice *and* Bob), then we can consider their joint function as having been fully reverse-engineered.

Here is a summary of how each of the three sections will proceed:

* **OV copying circuit**
    * We show that $W_{OV}$ in layer 1 is indeed mapping token embeddings to their unembeddings (i.e. sending the embedding of token `T` to a vector which will be decoded as `T` by the unembedding matrix with high probability).
* **QK prev-token circuit**
    * We show that $W_{QK}$ in layer 0 is indeed making tokens attend backwards by one position.
* **K-composition**
    * We show that the layer-0 OV circuit and layer-1 QK circuit are composing together to form the characteristic induction patterns we observed in the previous section (and that all other inputs into the layer-1 QK circuit are not important for determining the attention pattern), when we feed the model repeating sequences.
    * Now that composition has been established, we evaluate the end-to-end circuit formed by composing these two, and show that it is performing the role of "make token attend back to the token *after* a previous instance of itself."

After this, we'll have a look at composition scores, which are a more mathematically justified way of showing that two attention heads are composing (without having to look at their behaviour on any particular class of inputs, since it is a property of the actual model weights).

### OV copying circuit

Let's start with an easy parts of the circuit - the copying OV circuit of `1.4` and `1.10`. Let's start with head 4. The only interpretable (read: **privileged basis**) things here are the input tokens and output logits, so we want to study the matrix $W_E W_{OV}^{1.4} W_U$ (and same for `1.10`). This is the $(d_\text{vocab}, d_\text{vocab})$-shape matrix that combines with the attention pattern to get us from input to output.

We want to calculate this matrix, and inspect it. We should find that its diagonal values are very high, and its non-diagonal values are much lower.

**Question - why should we expect this observation? (Try and come up with your own answer before reading.)**
""")

    with st.expander("Hint"):
        st.markdown(r"""
Take a repeated pattern `A B ... A B`. Your induction head is responsible for making sure the prediction at the second `A` token is `B`.

Try and work through each step of the matrix multiplication $B^T (W_E W_{OV}^{1.4} W_U)$, where $B$ is a one-hot encoding of the token `B` (i.e. it's a vector with zeros everywhere, except for a one at position `B`). In other words, first apply $W_E$, then $W_{OV}^{1.4}$, then $W_U$. What should you be left with at the end?
""")

    with st.expander("Answer"):
        st.markdown(r"""
The OV circuit $W_{OV}^{1.4}$ is meant to be a **copy circuit**. In other words, it takes the embedding of some token, and returns a vector which will result in that same token being predicted.

Let's make this a bit more concrete. Suppose our repeated pattern is `A B ... A B`, and let $B$ be a one-hot encoding of the token `B`. Then we have:

* $B^T W_E$ is the token-embedding of `B`. It is a vector of length $d_\text{model}$.

* $B^T W_E W_{OV}^{1.4}$ is the vector which gets moved from the first `B` token to the second `A` token, by the operation of attention head `1.4`. It is a vector of length $d_\text{model}$.

* $B^T W_E W_{OV}^{1.4} W_U$ is the logit distribution corresponding to our prediction for the token following the second `A`. It is a vector of length $d_\text{vocab}$.

We want this prediction to be `B`, in other words we want $(B^T W_E W_{OV}^{1.4} W_U)_B$ to be the largest element of this vector (the subscript here means we're taking the $B$th element of the vector). 

Since $B$ is a one-hot encoding, $(B^T W_E W_{OV}^{1.4} W_U)$ is the $B$th row of the $(d_\text{vocab}, d_\text{vocab})$-size matrix $W_E W_{OV}^{1.4} W_U$. So we want the largest element of the $B$th row to be the $B$th element, i.e. the one lying on the diagonal of the matrix $W_E W_{OV}^{1.4} W_U$. 

Since $B$ was arbitrary here (we expect this to work for any repeated sequence), we conclude that each row of $W_U W_{OV}^{1.4} W_E$ should have its largest element be on the diagonal.

---

Note a small subtlety here - we don't expect this matrix to be the identity. If the induction head always resulted in a 100% probability that the repeated sequence would continue, then we *would* expect the matrix to be the identity **after applying softmax over the columns**. But in reality this doesn't happen, and it turns out to be much more helpful studying the matrix before applying softmax.

Sometimes, we'll describe a logit matrix like this one as the identity, as a shorthand. You should interpret statements like this as *"the logit distribution implied by this matrix is (approximately) the identity"*.
""")

#     st.info(r"""
# Tip: If you start running out of CUDA memory, cast everything to float16 (`tensor` -> `tensor.half()`) before multiplying - 50K x 50K matrices are large! Alternately, do the multiply on CPU if you have enough CPU memory. This should take less than a minute.

# Note: on some machines like M1 Macs, half precision can be much slower on CPU - try doing a `%timeit` on a small matrix before doing a huge multiplication!

# If none of this works, you might have to use LambdaLabs for these exercises (I had to!). Here are a list of `pip install`'s that you'll need to run, to save you time:

# ```python
# !pip install git+https://github.com/neelnanda-io/TransformerLens.git@new-demo
# !pip install circuitsvis
# !pip install fancy_einsum
# !pip install einops
# !pip install plotly
# !pip install torchtyping
# !pip install typeguard
# ```
# """)

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - compute the full OV circuit for head `1.4`

You should compute it as a `FactoredMatrix` object.

```python
if MAIN:
    head_index = 4
    layer = 1
    full_OV_circuit = None # YOUR CODE HERE - calculate matrix
```
""")

        with st.expander("Help - I'm not sure how to use this class to compute a product of more than 2 matrices."):
            st.markdown(r"""
You can compute it directly, as:

```python
full_OV_circuit = FactoredMatrix(W_E @ W_V, W_O @ W_U)
```

Alternatively, another nice feature about the `FactoredMatrix` class is that you can chain together matrix multiplications:

```python
OV_circuit = FactoredMatrix(W_V, W_O)
full_OV_circuit = W_E @ OV_circuit @ W_U # this is also a FactoredMatrix
```
""")
    with st.columns(1)[0]:

        st.markdown(r"""
#### Exercise - verify this matrix is the identity

Now we want to check that this matrix is the identity. Since it's in factored matrix form, this is a bit tricky, but there are still things we can do.

First, to validate that it looks diagonal-ish, let's pick 200 random rows and columns and visualise that - it should at least look identity-ish here! The `FactoredMatrix` class comes with a `get_sample` method, which allows you to compute a small randomly-chosen submatrix.


```python
if MAIN:
    # YOUR CODE HERE - compute random 200x200 sample
    imshow(full_OV_circuit_sample)
```

You should observe a pretty distinct diagonal pattern here, which is a good sign. However, the matrix is pretty noisy so it probably won't be exactly the identity. Instead, we should come up with a summary statistic to capture a rough sense of "closeness to the identity".

**Accuracy** is a good summary statistic - what fraction of the time is the largest logit in a row on the diagonal? Even if there's lots of noise, you'd probably still expect the largest logit to be on the diagonal a good deal of the time.

If you're on a Colab or have a powerful GPU, you should be able to compute the full matrix and perform this test. If not, the most efficient method is to pick some sample of the columns and work on them individually. Hint - if `M = AB` is a factored matrix, then the $i$th row of `M` is `A[i, :] @ B` (which should fit on your GPU!).

Bonus exercise: Top-5 accuracy is also a good metric (use `t.topk`, take the indices output).

When I run this I get about 30.79% - pretty underwhelming. It goes up to 47.73% for top-5. What's up with that?

```python
def top_1_acc(full_OV_circuit):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    pass

if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc(full_OV_circuit))
```""")

        with st.expander("Solution:"):
            st.markdown(r"""
```python
def top_1_acc(full_OV_circuit: FactoredMatrix):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B

    correct = []
    for i in tqdm(range(full_OV_circuit.shape[-1])):
        correct.append(t.argmax(A[i, :] @ B) == i)
    
    return t.tensor(correct).float().mean()
```

And for top-5:

```python
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
```
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - compute circuit for both induction heads

Now we return to why we have *two* induction heads. If both have the same attention pattern, the effective OV circuit is actually $W_U(W_O^{[1,4]}W_V^{[1,4]}+W_O^{[1,10]}W_V^{[1,10]})W_E$, and this is what matters. So let's re-run our analysis on this!""")

        with st.expander("Question - why might the model want to split the circuit across two heads?"):
            st.markdown(r"""
Because $W_O W_V$ is a rank 64 matrix. The sum of two is a rank 128 matrix. This can be a significantly better approximation to the desired 50K x 50K matrix!""")

        st.markdown(r"""
```python
if MAIN:
    print("Fraction of the time that the best logit is on the diagonal, for circuit 1.4 + 1.10:")
    
    'YOUR CODE HERE - compute top-1 accuracy for the effective OV circuit'
```
""")

        with st.expander("Hint"):
            st.markdown(r"""
The easiest way is to store them as two separate factored matrices, i.e. `full_OV_circuit_4` and `full_OV_circuit_10`. Then you can loop through them, calculating the rows and adding them together.

You can use this function as a template

```python
def top_1_acc_effective_circuit(full_OV_circuit_list: List[FactoredMatrix]):
    '''
    Inputs: list of factored matrices W_U @ W_OV^h @ W_E, for different heads h

    Returns: top_1_acc for more than one OV_circuit
    '''
    pass
```
""")
        with st.expander("Solution (and expected output)"):
            st.markdown(r"""
```python
def top_1_acc_effective_circuit(full_OV_circuit_list: List[FactoredMatrix]):
    '''
    Returns top_1_acc for more than one OV circuit
    '''
    num_circuits = len(full_OV_circuit_list)
    A_shape = full_OV_circuit_list[0].A.shape

    all_A = t.stack([full_OV_circuit.A for full_OV_circuit in full_OV_circuit_list])
    all_B = t.stack([full_OV_circuit.B for full_OV_circuit in full_OV_circuit_list])

    assert all_A.shape == (num_circuits, *A_shape)

    correct = []
    for i in tqdm(range(A_shape[0] // 30)):
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
```

You should get an accuracy of greater than 95%!
""")

    st.markdown(r"""

### QK prev-token circuit

The other easy circuit is the QK-circuit of L0H7 - how does it know to be a previous token circuit?

We can multiply out the full QK circuit via the positional embeddings: $W_\text{pos} W_Q^{0.7} (W_K^{0.7})^T W_\text{pos}^T$ to get a matrix `pos_by_pos` of shape `[max_ctx, max_ctx]` (max ctx = max context length, i.e. maximum length of a sequence we're allowing, which is set by our choice of dimensions in $W_\text{pos}$).

Note that in this case, our max context window is 2048 (we can check this via `model.cfg.n_ctx`). This is much smaller than the 50k-size matrices we were working with in the previous section, so we shouldn't need to use the factored matrix class here.

Once we calculate it, we can then mask it and apply a softmax, and should get a clear stripe on the lower diagonal (Tip: Click and drag to zoom in, hover over cells to see their values and indices!)

**Question - why should we expect this matrix to have a lower-diagonal stripe?**
""")

    with st.expander("Answer"):
        st.markdown(r"""
The full QK circuit $W_\text{pos} W_{QK}^{0.7} W_\text{pos}^T$ has shape `[n_ctx, n_ctx]`. After masking and scaling, the $(i, j)$th element of the matrix is the **attention score** paid by the token with position $i$ to the token with position $j$ (ignoring token encoding). We expect this to be very large when $j = i - 1$, because this is a **previous head token**.

After applying softmax, the $(i, j)$th element is the **attention probability** token $i$ pays to $j$, in a sequence of length 2048 (again ignoring token encodings). We expect this to be close to 1 when $j = i - 1$, and others to be close to zero.

---

#### Optional bonus exercise

Why is it justified to ignore token encodings? In this case, it turns out that the positional encodings have a much larger effect on the attention scores than the token encodings. You can verify this for yourself - after going through the next section (reverse-engineering K-composition), you'll have a better sense of how to perform attribution on the inputs to attention heads, and assess their importance).
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - compute and plot the full QK-circuit

Now, you should compute and plot the matrix.

Remember, you're calculating the attention pattern (i.e. probabilites) not the scores. You'll need to mask the scores (you can use the `mask_scores` function we've provided you with), and scale them.""")

        st.info(r"""If you're using VSCode, it might help to periodically clear the plotly graphs from your screen; they can slow down your performance by quite a bit!""")

        st.markdown(r"""
```python
def mask_scores(attn_scores: TT["query_d_model", "key_d_model"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

if MAIN:
    "YOUR CODE HERE - define pos_by_pos_pattern"

    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
    imshow(utils.to_numpy(pos_by_pos_pattern[:200, :200]), xaxis="Key", yaxis="Query")
```""")

    st.markdown(r"""

### K-composition circuit

We now dig into the hard part of the circuit - demonstrating the K-Composition between the previous token head and the induction head.

#### Splitting activations

We can repeat the trick from the logit attribution scores. The QK-input for layer 1 is the sum of 14 terms (2+n_heads) - the token embedding, the positional embedding, and the results of each layer 0 head. So for each head $\text{H}$ in layer 1, the query tensor (ditto key) corresponding to sequence position $i$ is:

$$
\begin{align*}
x W^\text{1.H}_Q &= (e + pe + \sum_{h=0}^{11} x^\text{0.h}) W^\text{1.H}_Q \\
&= e W^\text{1.H}_Q + pe W^\text{1.H}_Q + \sum_{h=0}^{11} x^\text{0.h} W^\text{1.H}_Q
\end{align*}
$$

where $e$ stands for the token embedding, $pe$ for the positional embedding, and $x^\text{0.h}$ for the output of head $h$ in layer 0 (and the sum of these tensors equals the residual stream $x$). All these tensors have shape `[seq, d_model]`. So we can treat the expression above as a sum of matrix multiplications `[seq, d_model] @ [d_model, d_head] -> [seq, d_head]`. 

For ease of notation, I'll refer to the 14 inputs as $(y_0, y_1, ..., y_{13})$ rather than $(e, pe, x^\text{0.h}, ..., x^{11.h})$. So we have:

$$
x W^h_Q = \sum_{i=0}^{13} y_i W^h_Q
$$

""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - analyse the relative importance

We can now analyse the relative importance of these 14 terms! A very crude measure is to take the norm of each term (by component and position).

Note that this is a pretty dodgy metric - q and k are not inherently interpretable! But it can be a good and easy-to-compute proxy.""")

        with st.expander("Question - why are Q and K not inherently interpretable? Why might the norm be a good metric in spite of this?"):
            st.markdown(r"""
They are not inherently interpretable because they operate on the residual stream, which doesn't have a **privileged basis**. You could stick a rotation matrix $R$ after all of the Q, K and V weights (and stick a rotation matrix before everything that writes to the residual stream), and the model would still behave exactly the same.

The reason taking the norm is still a reasonable thing to do is that, despite the individual elements of these vectors not being inherently interpretable, it's still a safe bet that if they are larger than they will have a greater overall effect on the residual stream. So looking at the norm doesn't tell us how they work, but it does indicate which ones are more important.
""")
        st.markdown(r"""
Fill in the functions below:

```python
def decompose_qk_input(cache: dict) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, 0, 0]th element is y_i (from notation above)
    '''
    pass


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    pass


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is W_K @ y_i^T (so the sum along axis 0 is just the k-values)
    '''
    pass


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
    imshow(utils.to_numpy(decomposed_q.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of query")
    imshow(utils.to_numpy(decomposed_k.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of key")
```""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, 0, 0]th element is y_i (from notation above)
    '''
    y0 = cache["embed"].unsqueeze(0) # shape (1, seq, d_model)
    y1 = cache["pos_embed"].unsqueeze(0) # shape (1, seq, d_model)
    y_rest = cache["result", 0].transpose(0, 1) # shape (12, seq, d_model)

    return t.cat([y0, y1, y_rest], dim=0)


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    W_Q = model.W_Q[1, ind_head_index]

    return einsum(
        "n seq d_head, d_head d_model -> n seq d_model",
        decomposed_qk_input, W_Q
    )


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]
    
    The [i, 0, 0]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    '''
    W_K = model.W_K[1, ind_head_index]
    
    return einsum(
        "n seq d_head, d_head d_model -> n seq d_model",
        decomposed_qk_input, W_K
    )
```
""")
        with st.expander("Click to see the output you should be getting."):
            st.markdown(r"""
You should see that the most important query components are the token and positional embeddings. The most important key components are those from $y_9$, which is $x_7$, i.e. from head `0.7`.
""")
            st.plotly_chart(fig_dict["norms_of_query_components"], use_container_width=True)
            st.plotly_chart(fig_dict["norms_of_key_components"], use_container_width=True)

        with st.expander("A technical note on the positional embeddings - optional, feel free to skip this."):
            st.markdown(r"""
You might be wondering why the tests compare the decomposed qk sum with the sum of the `resid_pre + pos_embed`, rather than just `resid_pre`. The answer lies in how we defined the transformer, specifically in this line from the config:

```python
positional_embedding_type="shortformer"
```

The result of this is that the positional embedding isn't added to the residual stream. Instead, it's added as inputs to the Q and K calculation (i.e. we calculate `(resid_pre + pos_embed) @ W_Q` and same for `W_K`), but **not** as inputs to the V calculation (i.e. we just calculate `resid_pre @ W_V`). This isn't actually how attention works in general, but for our purposes it makes the analysis of induction heads cleaner because we don't have positional embeddings interfering with the OV circuit.

**Question - this type of embedding actually makes it impossible for attention heads to form via $Q$-composition. Can you see why?**
""")

    st.markdown(r"""
This tells us which heads are probably important, but we can do better than that. Rather than looking at the query and key components separately, we can see how they combine together - i.e. take the decomposed attention scores.

This is a bilinear function of q and k, and so we will end up with a `decomposed_scores` tensor with shape `[query_component, key_component, query_pos, key_pos]`, where summing along BOTH of the first axes will give us the original attention scores (pre-mask).""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - decompose attention scores

Implement the function giving the decomposed scores (remember to scale by `sqrt(d_head)`!) For now, don't mask it.
""")
        with st.expander("Question - why do I focus on the attention scores, not the attention pattern? (i.e. pre softmax not post softmax)"):
            st.markdown(r"""

Because the decomposition trick *only* works for things that are linear - softmax isn't linear and so we can no longer consider each component independently.
""")
        with st.expander("Help - I'm confused about what we're doing / why we're doing it."):
            st.markdown(r"""
Remember that each of our components writes to the residual stream separately. So after layer 1, we have:
""")
            st_image("components.png", 550)
            # st_excalidraw("components", 550)
            st.markdown("")
            st.markdown(r"""
We're particularly interested in the attention scores computed in head `1.4`, and how they depend on the inputs into that head. We've already decomposed the residual stream value $x$ into its terms $e$, $pe$, and $x^ 0$ through $x^{11}$ (which we've labelled $y_0, ..., y_{13}$ for simplicity), and we've done the same for key and query terms. We can picture these terms being passed into head `1.4` as:""")
            st_image("components-2.png", 680)
            # st_excalidraw("components-2", 800)
            st.markdown("")
            st.markdown(r"""
So when we expand `attn_scores` out in full, they are a sum of $14^2 = 196$ terms - one for each combination of `(query_component, key_component)`.

---

##### Why is this decomposition useful?

We have a theory about a particular circuit in our model. We think that head `1.4` is an attention head, and the most important components that feed into this head are the prev token head `0.7` (as key) and the token embedding (as query). This is already supported by the evidence of our magnitude plots above (because we saw that `0.7` as key and token embeddings as query were large), but we still don't know how this particular key and query work **together**; we've only looked at them separately.

By decomposing `attn_scores` like this, we can check whether the contribution from combination `(query=tok_emb, key=0.7)` is indeed producing the characteristic induction head pattern which we've observed (and the other 195 terms don't really matter).
""")
        st.markdown(r"""

```python
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]
    
    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    pass


if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = einops.reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )
    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    imshow(to_numpy(t.tril(decomposed_scores[0, 9])), title="Attention Scores for component from Q=Embed and K=Prev Token Head")
    # Second plot: std dev over query and key positions, shown by component
    imshow(to_numpy(decomposed_stds), xaxis="Key Component", yaxis="Query Component", title="Standard deviations of components of scores", x=component_labels, y=component_labels)

```""")

        with st.expander("Click here to see the output you should be getting"):
            st.plotly_chart(fig_dict["attn_scores_for_component"], use_container_width=True)
            st.plotly_chart(fig_dict["attn_scores_std_devs"], use_container_width=True)

        with st.expander("Question - what is the interpretation of these plots?"):
            st.markdown(r"""
The first plot tells you that the term $e W_{QK}^{1.4} (x^{0.7})^T$ (i.e. the component of the attention scores for head `1.4` where the query is supplied by the token embeddings and the key is supplied by the output of head `0.7`) produces the distinctive attention pattern we see in the induction head: a strong diagonal stripe.

Although this tells us that this this component would probably be sufficient to implement the induction mechanism, it doesn't tell us the whole story. Ideally, we'd like to show that the other 195 terms are unimportant. Taking the standard deviation across the attention scores for a particular pair of components is a decent proxy for how important this term is in the overall attention pattern. The second plot shows us that the standard deviation is very small for all the other components, so we can be confident that the other components are unimportant.

To summarise:

* The first plot tells us that the pair `(q_component=tok_emb, k_component=0.7)` produces the characteristic induction-head pattern we see in attention head `1.4`.
* The second plot confirms that this pair is the only important one for influencing the attention pattern in `1.4`; all other pairs have very small contributions.
""")

    st.markdown(r"""
#### Interpreting the full circuit

Now we know that head `1.4` is composing with head `0.7` via K composition, we can multiply through to create a full circuit:

$$
W_E\, W_{QK}^{1.4}\, (W_{OV}^{0.7})^T\, W_E^T
$$

and verify that it's the identity. (Note, when we say identity here, we're again thinking about it as a distribution over logits, so this should be taken to mean "high diagonal values", and we'll be using our previous metric of `top_1_acc`.)
""")


    with st.expander("Question - why should this be the identity?"):
        st.markdown(r"""
Like before, we'll consider the repeating sequence `A B ... A B`, denoting $A$ and $B$ as the one-hot encoded vectors. Take any arbitrary token $X$. Then the $(A, X)$th element of this matrix is:

$$
A^T \, W_E\, W_{QK}^{1.4}\, (W_{OV}^{0.7})^T\, W_E^T \, X = (\color{red}A\color{black}^T \, W_{Q}^{1.4}) \; (\color{red}X\color{black}^T \, W_{OV}^{0.7} \, W_K^{1.4})^T
$$

where the red letters denote embedding vectors.

The left term $(\color{red}A\color{black}^T \, W_{Q}^{1.4})$ is the query vector for token $A$. This query vector is essentially **"looking for a token which followed $A$".**

$\color{red}X\color{black}^T W_{OV}^{0.7}$ is the vector which was moved one position forwards by our prev attention head `0.7`. When projected by $W_K^{1.4}$, we get a key vector $(\color{red}X\color{black}^T \, W_{OV}^{0.7} \, W_K^{1.4})$ which essentially contains the information **"this token followed $X$".**

So, to verbally describe this attention pattern:

$$
\begin{aligned}
A^T \, W_E\, W_{QK}^{1.4}\, W_{OV}^{0.7}\, W_E^T \, X &= \underbrace{(\text{I'm looking for a token which followed A})}_\text{query} \boldsymbol{\cdot} \underbrace{(\text{I am a token which followed X})}_{\text{key}} \\
\end{aligned}
$$

If $X=A$, then the key is a good match for the query (since it's exactly what the query was "looking for"), so the dot product is large. If $X\neq A$, then the key is a poor match for the query, and the inner product is small.

---

To summarize, we've argued that the $(A, X)$th element of the matrix is large when $X=A$, and small when $X \neq A$. This is equivalent to saying that the largest element in each **column** is the diagonal element (note this is different from the previous case, where we were looking at **rows**, so you'll have to transpose your matrix - the `FactoredMatrix` class has a transpose method).
""")

# Note - this actually shows why the largest element of each **row** of the matrix should be the diagonal one, rather than the largest element on each column. It's important to get this the right way round - remember that logits are invariant to the addition of a constant, so it's meaningless to compare across two different logit distributions! That's why, in the code below, we've applied the test to the transpose of your function's output.
# """)

# We can now reuse our `top_1_acc` code from before to check that it's identity-like, we see that half the time the diagonal is the top (goes up to 89% with top 5 accuracy) (We transpose first, because we want the argmax over the key dimension)

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - compute the K-composition circuit

Calculate the matrix above, as a `FactoredMatrix` object.""")

        
        with st.expander("Aside about multiplying FactoredMatrix objects together."):
            st.markdown(r"""
If  `M1 = A1 @ B1` and `M2 = A2 @ B2` are factored matrices, then `M = M1 @ M2` returns a new factored matrix. This might be:

```python
FactoredMatrix(M1.AB @ M2.A, M2.B)
```

or it might be:

```python
FactoredMatrix(M1.A, M1.B @ M2.AB)
```

with these two objects corresponding to the factorisations $M = (A_1 B_1) (A_2 B_2)$ and $M = A_1 (B_1 A_2 B_2)$ respectively.

Which one gets returned depends on the size of the hidden dimension, e.g. `M1.mdim < M2.mdim` then the factorisation used will be $M = (A_1 B_1) (A_2 B_2)$.
""")
        st.markdown(r"""

```python
def find_K_comp_full_circuit(
    prev_token_head_index: int,
    ind_head_index: int
) -> FactoredMatrix:
    '''
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    pass

if MAIN:
    ind_head_index = 4
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_circuit(prev_token_head_index, ind_head_index)
    print("Fraction of tokens where the highest activating key is the same token", top_1_acc(K_comp_circuit.T).item())
    del K_comp_circuit
```

You can also try this out for `ind_head_index = 10`. Do you get a better result?

Note - unlike last time, it doesn't make sense to consider the "effective circuit" formed by adding together the weight matrices for heads `1.4` and `1.10`. Why not?
""")
        with st.expander("Answer"):
            st.markdown(r"""
Because the weight matrices we're dealing with here are from the QK circuit, not the OV circuit. These don't get combined in a linear way; instead we take softmax over each head's QK-circuit output individually.
""")

    st.markdown(r"""


## Further Exploration of Induction Circuits

I now consider us to have fully reverse engineered an induction circuit - by both interpreting the features and by reverse engineering the circuit from the weights. But there's a bunch more ideas that we can apply for finding circuits in networks that are fun to practice on induction heads, so here's some bonus content - feel free to skip to the later bonus ideas though.

### Composition scores

A particularly cool idea in the paper is the idea of [virtual weights](https://transformer-circuits.pub/2021/framework/index.html#residual-comms), or compositional scores. (Though I came up with it, so I'm deeply biased!). This is used [to identify induction heads](https://transformer-circuits.pub/2021/framework/index.html#analyzing-a-two-layer-model).

The key idea of compositional scores is that the residual stream is a large space, and each head is reading and writing from small subspaces. By default, any two heads will have little overlap between their subspaces (in the same way that any two random vectors have almost zero dot product in a large vector space). But if two heads are deliberately composing, then they will likely want to ensure they write and read from similar subspaces, so that minimal information is lost. As a result, we can just directly look at "how much overlap there is" between the output space of the earlier head and the K, Q, or V input space of the later head. 

We represent the **output space** with $W_{OV}=W_V W_O$. Call matrices like this $W_A$.

We represent the **input space** with $W_{QK}=W_Q W_K^T$ (for Q-composition), $W_{QK}^T=W_K  W_Q^T$ (for K-Composition) or $W_{OV}=W_V W_O$ (for V-Composition, of the later head). Call matrices like these $W_B$ (we've used this notation so that $W_B$ refers to a later head, and $W_A$ to an earlier head).""")

    with st.expander("Help - I don't understand what motivates these definitions."):
        st.markdown(r"""
Before we discuss the definitions, a quick reminder - we're using the convention that matrices multiply **on the right**. So when we use language like *"the output of matrix $M$ when fed input $x$"*, we mean the vector $x^T M$. And *"the output space of matrix $M$"* is all the vectors which can be written as $x^T M$ for some $x$.

---

First, let's consider Q-composition. 

Let $x_i, x_j$ be the residual stream vectors. The attention paid by the $i$th token to the $j$th token is:

$$
x_i^T W_Q W_K^T x_j
$$

Suppose we think K-composition might be happening in this head, i.e. the query input $x_i$ comes from the output of an earlier head. We can write $x_i^T = y^T (W_V' W_O')$ (where $y$ is some mix of values earlier in the residual stream, produced by the attention matrix $A'$). So our attention score is:

$$
y^T (W_V' W_O') (W_Q W_K^T) x_j
$$

If we multiply from left to right, we can see that the **output** of $W_V' W_O'$ is used as the **input** for $W_Q W_K^T$. This is why we call the former the output space, and the latter the input space. If there is lots of overlap between these two spaces, it probably means that meaningful computation is being done via their composition. In contrast, if there is very little overlap between these spaces, then the composition of these two matrices will mostly just send vectors to zero, and no information will be transmitted.

---

K-composition is very similar. This time it's the key-side vector $x_j$ which was produced from a previous attention head, so we have:

$$
\begin{aligned}
x_i^T W_Q W_K^T x_j &= x_j^T W_K W_Q^T x_i \\
&= y^T (W_V' W_O') (W_K W_Q^T) x_i \\
\end{aligned}
$$

i.e. the output of $W_V' W_O'$ is being used as input for $W_K W_Q^T$.

---

Finally, let's consider V-composition.

The complexity here isn't in how the attention scores are computed, but in which vectors get moved from the source to the destination. We suppose $x_j$ (the key-side vector) was itself produced by an **earlier** attention head, i.e. $x_j^T = y^T (W_V' W_O')$. So the vector that gets moved from source position $j$ (when it is attended to) in the **later** attention head is:

$$
x_j^T (W_V W_O) = y^T (W_V' W_O') (W_V W_O)
$$

so again, by examining the right hand side expression, we see that the output of $W_V' W_O'$ is used as input for $W_V W_O$.
""")

    st.markdown(r"""

How do we formalise overlap? This is basically an open question, but a surprisingly good metric is $\frac{|W_AW_B|}{|W_B||W_A|}$ where $|W|=\sum_{i,j}W_{i,j}^2$ is the Frobenius norm, the sum of squared elements. (If you're dying of curiosity as to what makes this a good metric, you can jump to the section immediately after the exercises below.)""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Excercise - calculate composition scores

Let's calculate this metric for all pairs of heads in layer 0 and layer 1 for each of K, Q and V composition and plot it.

We'll start by implementing this using plain old tensors (later on we'll see how this can be sped up using the `FactoredMatrix` class). We also won't worry about batching our calculations yet; we'll just do one matrix at a time.

We've given you tensors `q_comp_scores` etc. to hold the composition scores for each of Q, K and V composition (i.e. the `[i, j]`th element of `q_comp_scores` is the Q-composition score between the output from the `i`th head in layer 0 and the input to the `j`th head in layer 1). You should complete the function `get_comp_score`, and then fill in each of these tensors.
""")

# def get_comp_score(
#     W_A: TT["idx_A", "in_A", "out_A"], 
#     W_B: TT["idx_B", "out_A", "out_B]
# ) -> TT["idx_A", "idx_B"]:
#     '''
#     W_A is a batch of output matrices, indexed by first dimension.
#     W_B is a batch of input matrices, indexed by first dimension.

#     Return the composition scores between each matrix in W_A and W_B.
#     '''
        st.markdown(r"""

```python
def get_comp_score(
    W_A: TT["in_A", "out_A"], 
    W_B: TT["out_A", "out_B"]
) -> float:
    '''
    Return the composition score between W_A and W_B.
    '''
    pass

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

    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
```""")

        with st.expander("Solution"):
            st.markdown(r"""
```python
def get_comp_score(
    W_A: TT["in_A", "out_A"], 
    W_B: TT["out_A", "out_B"]
) -> float:
    '''
    Return the composition score between W_A and W_B.
    '''
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item()

if MAIN:
    # Get all QK and OV matrices
    W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
    W_OV = model.W_V @ model.W_O

    # Define tensors to hold the composition scores
    q_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads)
    k_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads)
    v_comp_scores = t.zeros(model.cfg.n_heads, model.cfg.n_heads)
    
    # Fill in the tensors
    for i in tqdm(range(model.cfg.n_heads)):
        for j in range(model.cfg.n_heads):
            q_comp_scores[i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
            k_comp_scores[i, j] = get_comp_score(W_OV[0, i], W_QK[1, j].T)
            v_comp_scores[i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
```""")
        st.markdown(r"""

#### Setting a Baseline

To interpret the above graphs we need a baseline! A good one is what the scores look like at initialisation. Make a function that randomly generates a composition score 200 times and tries this. Remember to generate 4 [d_head, d_model] matrices, not 2 [d_model, d_model] matrices! This model was initialised with Kaiming Uniform Initialisation:

```python
W = t.empty(shape)
nn.init.kaiming_uniform_(W, a=np.sqrt(5))
```

(Ideally we'd do a more efficient generation involving batching, and more samples, but we won't worry about that here)


```python
def generate_single_random_comp_score() -> float:
    '''
    Write a function which generates a single composition score for random matrices
    '''
    pass


if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()
    print("Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()
```

We can re-plot our above graphs with this baseline set to white. Look for interesting things in this graph!

```python
if MAIN:
    px.imshow(
        utils.to_numpy(q_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        utils.to_numpy(k_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        utils.to_numpy(v_comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
```""")
        with st.expander("Click to see the output you should be getting"):
            st.markdown(r"The plots for Q, K and V composition respectively:")
            st.plotly_chart(fig_dict["q_comp_scores"], use_container_width=True)
            st.plotly_chart(fig_dict["k_comp_scores"], use_container_width=True)
            st.plotly_chart(fig_dict["v_comp_scores"], use_container_width=True)
        with st.expander("Some interesting things to observe:"):
            st.markdown(r"""
The most obvious thing that jumps out (when considered in the context of all the analysis we've done so far) is the K-composition scores. `0.7` (the prev token head) is strongly composing with `1.4` and `1.10` (the two attention heads). This is what we expect, and is a good indication that our composition scores are working as intended.

Another interesting thing to note is that the V-composition scores for heads `1.4` and `1.10` with all other heads in layer 0 are very low. In the context of the induction circuit, this is a good thing - the OV circuits of our induction heads should be operating on the **embeddings**, rather than the outputs of the layer-0 heads. (If our repeating sequence is `A B ... A B`, then it's the QK circuit's job to make sure the second `A` attends to the first `B`, and it's the OV circuit's job to project the residual vector at that position onto the **embedding space** in order to extract the `B`-information, while hopefully ignoring anything else that has been written to that position by the heads in layer 0). So once again, this is a good sign for our composition scores.

""")

    st.markdown(r"""

#### Theory + Efficient Implementation

So, what's up with that metric? The key is a cute linear algebra result that the squared Frobenius norm is equal to the sum of the squared singular values.
""")
    with st.expander("Proof"):
        st.markdown(r"""We'll give three different proofs:

---

##### Short, not-fully-rigorous proof:

Clearly the Frobenius norm equals the sum of squared singular values when $M$ is diagonal. The singular values of $M$ don't change when we multiply it by an orthogonal matrix (only the matrices $U$ and $V$ will change, not $S$), so it remains to show that the Frobenius norm also won't change when we multiply $M$ by an orthogonal matrix. But this follows from the fact that the Frobenius norm is the sum of the squared $l_2$ norms of the column vectors of $M$, and orthogonal matrices preserve $l_2$ norms.

---

##### Ugly proof:

$$
\begin{aligned}
|M|^2 &= \sum_{ij}M_{ij}^2 \\
&= \sum_{ij}((USV^T)_{ij})^2 \\
&= \sum_{ij}\bigg(\sum_k U_{ik}S_{kk}V_{jk}\bigg)^2 \\
&= \sum_{ijk_1 k_2}S_{k_1 k_1} S_{k_2 k_2} U_{i k_1} U_{i k_2} V_{j k_2} V_{j k_2} \\
&= \sum_{k_1 k_2}S_{k_1 k_1} S_{k_2 k_2} \bigg(\sum_i U_{i k_1} U_{i k_2}\bigg)\bigg(\sum_j V_{j k_2} V_{j k_2}\bigg) \\
\end{aligned}
$$

Each of the terms in large brackets is actually the dot product of columns of $U$ and $V$ respectively. Since these are orthogonal matrices, these terms evaluate to 1 when $k_1=k_2$ and 0 otherwise. So we are left with:

$$
|M|^2 = \sum_{k}S_{k k}^2
$$

---

##### Cute proof which uses the fact that the squared Frobenius norm $|M|^2$ is the same as the trace of $MM^T$:

$$
|M|^2 = \text{Tr}(MM^T) = \text{Tr}(USV^TVSU^T) = \text{Tr}(US^2U^T) = \text{Tr}(S^2 U^T U) = \text{Tr}(S^2) = |S|^2
$$

where we used the cyclicity of trace, and the fact that $U$ is orthogonal so $U^TU=I$ (and same for $V$). We finish by observing that $|S|^2$ is precisely the sum of the squared singular values.
""")
    st.markdown(r"""
So if $W_A=U_AS_AV_A^T$, $W_B=U_BS_BV_B^T$, then $|W_A|=|S_A|$, $|W_B|=|S_B|$ and $|W_AW_B|=|S_AV_A^TU_BS_B|$. In some sense, $V_A^TU_B$ represents how aligned the subspaces written to and read from are, and the $S_A$ and $S_B$ terms weights by the importance of those subspaces.""")

    with st.expander("Click here, if this explanation still seems confusing."):
        st.markdown(r"""
$U_B$ is a matrix of shape `[d_model, d_head]`. It represents **the subspace being read from**, i.e. our later head reads from the residual stream by projecting it onto the `d_head` columns of this matrix.

$V_A$ is a matrix of shape `[d_model, d_head]`. It represents **the subspace being written to**, i.e. the thing written to the residual stream by our earlier head is a linear combination of the `d_head` column-vectors of $V_A$.

$V_A^T U_B$ is a matrix of shape `[d_head, d_head]`. Each element of this matrix is formed by taking the dot product of two vectors of length `d_model`:

* $v_i^A$, a column of $V_A$ (one of the vectors our earlier head embeds into the residual stream)
* $u_j^B$, a column of $U_B$ (one of the vectors our later head projects the residual stream onto)

Let the singular values of $S_A$ be $\sigma_1^A, ..., \sigma_k^A$ and similarly for $S_B$. Then:

$$
|S_A V_A^T U_B S_B| = \sum_{i,j=1}^k (\sigma_i^A \sigma_j^B)^2 |v^A_i \cdot u^B_j|^2
$$

This is a weighted sum of the squared cosine similarity of the columns of $V_A$ and $U_B$ (i.e. the output directions of the earlier head and the input directions of the later head). The weights are given by the singular values of both $S_A$ and $S_B$ - i.e. if $v^A_i$ is an important output direction, and $u_B^i$ is an important input direction, then the composition score will be much higher when these two directions are aligned with each other.

---

To build intuition, let's consider some extreme examples.

* If there was no overlap between the spaces being written to and read from, then $V_A^T U_B$ would be a matrix of zeros (since every $v_i^A \cdot u_j^B$ would be zero). This would mean that the composition score would be zero.

* If there was perfect overlap, i.e. the column vectors of $U_B$ and $V_A$ are the same up to a permutation, then $V_A^T U_B$ would just be a permutation matrix, and $|S_AV_A^TU_BS_B|$ would be some sumproduct of the squared singular values from $S_A$ and $S_B$. This is largest when the singular values are sorted in the same way, i.e. when .the "most important output vectors" of the earlier head line up with the "most important input vectors" of the later head.

* If our output $W_A$ actually had rank 1, then $|S_AV_A^TU_BS_B| = \sigma_1^A |  v_1^A \cdot \sum_{j=1}^k \sigma_j^B u_j^B|^2$. This is largest when $v_1^A$ is aligned with the most important singular vectors of $S_B$ (i.e. the most important input directions).
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - batching, and using the `FactoredMatrix` class

We can also use this insight to write a more efficient way to calculate composition scores - this is extremely useful if you want to do this analysis at scale! The key is that we know that our matrices have a low rank factorisation, and it's much cheaper to calculate the SVD of a narrow matrix than one that's large in both dimensions. See the [algorithm described at the end of the paper](https://transformer-circuits.pub/2021/framework/index.html#induction-heads:~:text=Working%20with%20Low%2DRank%20Matrices) (search for SVD).

So we can work with the `FactoredMatrix` class. This also provides the method `.norm()` which returns the Frobenium norm. This is also a good opportunity to bring back baching - this will sometimes be useful in our analysis, e.g. if we thought one of the final-layer heads was being used in Q-composition and the number of layers was more than two, then we might want to efficiently compute a 3D array with `[i, j, k]`th element equal to the Q-composition score between head `i.j` and the `k`th head in the final layer.

```python
def get_batched_comp_scores(
    W_As: FactoredMatrix,
    W_Bs: FactoredMatrix
) -> t.Tensor:
    '''Computes the compositional scores from indexed factored matrices W_As and W_Bs.

    Each of W_As and W_Bs is a FactoredMatrix object which is indexed by all but its last 2 dimensions, i.e.:
        W_As.shape == (*A_idx, A_in, A_out)
        W_Bs.shape == (*B_idx, B_in, B_out)
        A_out == B_in

    Return: tensor of shape (*A_idx, *B_idx) where the [*a_idx, *b_idx]th element is the compositional score from W_As[*a_idx] to W_Bs[*b_idx].
    '''
    pass

if MAIN:
    W_QK = FactoredMatrix(model.W_Q, model.W_K.transpose(-1, -2))
    W_OV = FactoredMatrix(model.W_V, model.W_O)

    q_comp_scores = get_batched_comp_scores(W_OV[0], W_QK[1])
    k_comp_scores = get_batched_comp_scores(W_OV[0], W_QK[1].T) 
    # Note - for FactoredMatrix, .T is interpreted as transpose of last two dims
    v_comp_scores = get_batched_comp_scores(W_OV[0], W_OV[1])
```""")

        with st.expander("Hint"):
            st.markdown(r"""
Suppose `W_As` has shape `(A1, A2, ..., Am, A_in, A_out)` and `W_Bs` has shape `(B1, B2, ..., Bn, B_in, B_out)` (where `A_out == B_in`).

It will be helpful to reshape these two tensors so that:

```python
W_As.shape == (A1*A2*...*Am, 1, A_in, A_out)
W_Bs.shape == (1, B1*B2*...*Bn, B_in, B_out)
```

since we can then multiply them together as `W_As @ W_Bs` (broadcasting will take care of this for us!).

To do the reshaping, the easiest way is to reshape `W_As.A` and `W_As.B`, and define a new `FactoredMatrix` from these reshaped tensors (and same for `W_Bs`).
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def get_batched_comp_scores(
    W_As: FactoredMatrix,
    W_Bs: FactoredMatrix
) -> t.Tensor:
    '''Computes the compositional scores from indexed factored matrices W_As and W_Bs.

    Each of W_As and W_Bs is a FactoredMatrix object which is indexed by all but its last 2 dimensions, i.e.:
        W_As.shape == (*A_idx, A_in, A_out)
        W_Bs.shape == (*B_idx, B_in, B_out)
        A_out == B_in

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
```""")

    st.markdown(r"""

#### Targeted Ablations

We can refine the ablation technique to detect composition by looking at the effect of the ablation on the attention pattern of an induction head, rather than the loss. Let's implement this!

Gotcha - by default, `run_with_hooks` removes any existing hooks when it runs. If you want to use caching, set the `reset_hooks_start` flag to False.

```python
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
```""")

    with st.expander("Question - what is the interpretation of the results you're getting?"):
        st.markdown(r"""
You should have found that the induction score without any ablations is about 0.68, and that most other heads don't change the induction score by much when they are ablated, except for head 7 which reduces the induction score to nearly zero.

This is another strong piece of evidence that head `0.7` is the prev token head in this induction circuit.
""")
    st.markdown(r"""

## Bonus

### Looking for Circuits in Real LLMs

A particularly cool application of these techniques is looking for real examples of circuits in large language models. Fortunately, there's a bunch of open source ones you can play around with in the `TransformerLens` library! Many of the techniques we've been using for our 2L transformer carry over to ones with more layers.

This library should make it moderately easy to play around with these models - I recommend going wild and looking for interesting circuits!

Some fun things you might want to try:

- Look for induction heads - try repeating all of the steps from above. Do they follow the same algorithm?
- Look for neurons that erase info
    - i.e. having a high negative cosine similarity between the input and output weights
- Try to interpret a position embedding.
""")
    with st.expander("Positional Embedding Hint"):
        st.markdown(r"""
Look at the singular value decomposition `t.svd` and plot the principal components over position space. High ones tend to be sine and cosine waves of different frequencies.
""")
    st.markdown(r"""

- Look for heads with interpretable attention patterns: e.g. heads that attend to the same word (or subsequent word) when given text in different languages, or the most recent proper noun, or the most recent full-stop, or the subject of the sentence, etc.
    - Pick a head, ablate it, and run the model on a load of text with and without the head. Look for tokens with the largest difference in loss, and try to interpret what the head is doing.
- Try replicating some of Kevin's work on indirect object identification.
- Inspired by the [ROME paper](https://rome.baulab.info/), use the causal tracing technique of patching in the residual stream - can you analyse how the network answers different facts?

Note: I apply several simplifications to the resulting transformer - these leave the model mathematically equivalent and doesn't change the output log probs, but does somewhat change the structure of the model and one change translates the output logits by a constant.""")

    with st.expander("Model simplifications"):
        st.markdown(r"""
#### Centering $W_U$

The output of $W_U$ is a $d_{vocab}$ vector (or tensor with that as the final dimension) which is fed into a softmax

#### LayerNorm Folding

LayerNorm is only applied at the start of a linear layer reading from the residual stream (eg query, key, value, mlp_in or unembed calculations)

Each LayerNorm has the functional form $LN:\mathbb{R}^n\to\mathbb{R}^n$, 
$LN(x)=s(x) * w_{ln} + b_{ln}$, where $*$ is element-wise multiply and $s(x)=\frac{x-\bar{x}}{|x-\bar{x}|}$, and $w_{ln},b_{ln}$ are both vectors in $\mathbb{R}^n$

The linear layer has form $l:\mathbb{R}^n\to\mathbb{R}^m$, $l(y)=Wy+b$ where $W\in \mathbb{R}^{m\times n},b\in \mathbb{R}^m,y\in\mathbb{R}^n$

So $f(LN(x))=W(w_{ln} * s(x)+b_{ln})+b=(W * w_{ln})s(x)+(Wb_{ln}+b)=W_{eff}s(x)+b_{eff}$, where $W_{eff}$ is the elementwise product of $W$ and $w_{ln}$ (showing that elementwise multiplication commutes like this is left as an exercise) and $b_{eff}=Wb_{ln}+b\in \mathbb{R}^m$.

From the perspective of interpretability, it's much nicer to interpret the folded layer $W_{eff},b_{eff}$ - fundamentally, this is the computation being done, and there's no reason to expect $W$ or $w_{ln}$ to be meaningful on their own. 
""")
    st.markdown(r"""
### Training Your Own Toy Models

A fun exercise is training models on the minimal task that'll produce induction heads - predicting the next token in a sequence of random tokens with repeated subsequences. You can get a small 2L Attention-Only model to do this.
""")
    with st.expander("Tips"):
        st.markdown(r"""
* Make sure to randomise the positions that are repeated! Otherwise the model can just learn the boring algorithm of attending to fixed positions
* It works better if you *only* evaluate loss on the repeated tokens, this makes the task less noisy.
* It works best with several repeats of the same sequence rather than just one.
* If you do things right, and give it finite data + weight decay, you *should* be able to get it to grok - this may take some hyper-parameter tuning though.
* When I've done this I get weird franken-induction heads, where each head has 1/3 of an induction stripe, and together cover all tokens.
* It'll work better if you only let the queries and keys access the positional embeddings, but *should* work either way
""")
    st.markdown(r"""

### Interpreting Induction Heads During Training

A particularly striking result about induction heads is that they consistently [form very abruptly in training as a phase change](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#argument-phase-change), and are such an important capability that there is a [visible non-convex bump in the loss curve](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-22-08-00---VmlldzoyNTI2MDM0?accessToken=r6v951q0e1l4q4o70wb2q67wopdyo3v69kz54siuw7lwb4jz6u732vo56h6dr7c2) (in this model, approx 2B to 4B tokens). I have a bunch of checkpoints for this model, you can try re-running the induction head detection techniques on intermediate checkpoints and see what happens. (Bonus points if you have good ideas for how to efficiently send a bunch of 300MB checkpoints from Wandb lol)

""")


func_page_list = [
    (section_home, "🏠 Home"), 
    (section_intro, "1️⃣ TransformerLens: Introduction"), 
    (section_finding_induction_heads, "2️⃣ Finding induction heads"), 
    (section_hooks, "3️⃣ TransformerLens: Hooks"), 
    (section_reverse_engineering, "4️⃣ Reverse-engineering induction circuits"),
    (section_other_features, "5️⃣ TransformerLens: Other Features"), 
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = {page: idx for idx, (func, page) in enumerate(func_page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()
