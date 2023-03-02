import os
# if not os.path.exists("./images"):
#     os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

def img_to_html(img_path, width):
    with open("images/page_images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;st-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

def read_from_html(filename):
    filename = f"images/{filename}.html" if "written_images" in filename else f"images/page_images/{filename}.html"
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

NAMES = []

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

def section_home():
    st.sidebar.markdown(r"""
## Table of contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
Links to Colab: [**exercises**](https://colab.research.google.com/drive/1LpDxWwL2Fx0xq3lLgDQvHKM5tnqRFeRM?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1ND38oNmvI702tu32M74G26v-mO5lkByM?usp=share_link)
""")
    st_image("sampling.png", 350)
    st.markdown(r"""
# Training and Sampling

## Introduction

. . .

## Imports

```python
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch as t
import torch.nn as nn
import math
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm
import datasets
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
```
""")

    with st.expander("Help - I get error `ImportError: DLL load failed while importing lib` when I try and import things."):
        st.markdown(r"""
To fix this problem, run the following code in your terminal:

```
conda install libboost boost-cpp -c conda-forge
```
 
then restart your IDE. Hopefully this fixes the problem.
""")
    st.markdown(r"""
## Learning objectives

Here are the learning objectives for each section of the tutorial. At the end of each section, you should refer back here to check that you've understood everything.
""")

    st.info(r"""
## 1️⃣ Training

* Review the interpretation of a transformer's output, and learn how it's trained by minimizing cross-entropy loss between predicted and actual next tokens
* Construct datasets and dataloaders for the corpus of Shakespeare text
* Implement a transformer training loop
""")
    st.info(r"""
## 2️⃣ Sampling and Caching

* Learn how to sample from a transformer
* Learn how to cache the output of a transformer, so that it can be used to generate text more efficiently
""")

    
def section_training():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#what-is-the-point-of-a-transformer">What is the point of a transformer?</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#how-is-the-model-trained">How is the model trained?</a></li>
   </ul></li>
   <li><a class="contents-el" href="#tokens-transformer-inputs">Tokens - Transformer Inputs</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#how-do-we-convert-language-to-vectors">How do we convert language to vectors?</a></li>
       <li><a class="contents-el" href="#idea-integers-to-vectors">Idea: integers to vectors</a></li>
       <li><a class="contents-el" href="#tokens-language-to-sequence-of-integers">Tokens: language to sequence of integers</a></li>
   </ul></li>
   <li><a class="contents-el" href="#logits-transformer-outputs">Logits - Transformer Outputs</a></li>
   <li><a class="contents-el" href="#generation">Generation!</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Training
""")
    st.info(r"""
## Learning Objectives

* Understand what a transformer is used for
* Understand causal attention, and what a transformer's output represents
* Learn what tokenization is, and how models do it
* Understand what logits are, and how to use them to derive a probability distribution over the vocabulary
""")
    st.markdown(r"""
### Setup

You should run the following at the top of your notebook / Python file:

```python
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import t
import t.nn as nn
import numpy as np
import math
from transformer_lens import EasyTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm

reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
```
""")

def section_sampling():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#high-level-architecture">High-Level architecture</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#summary">Summary</a></li>
        <li><a class="contents-el" href="#residual-stream">Residual stream</a></li>
        <li><a class="contents-el" href="#transformer-blocks">Transformer blocks</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#attention">Attention</a></li>
            <li><a class="contents-el" href="#mlps">MLPs</a></li>
        </ul></li>
        <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
        <li><a class="contents-el" href="#bonus-things-less-conceptually-important-but-key-technical-details">Bonus things</a></li>
    </ul></li>
    <li><a class="contents-el" href="#actual-code">Actual Code!</a></li>
    <li><ul class="contents">
    <li><a class="contents-el" href="#parameters-and-activations">Parameters vs Activations</a></li>
    <li><a class="contents-el" href="#config">Config</a></li>
    <li><a class="contents-el" href="#tests">Tests</a></li>
    <li><a class="contents-el" href="#layernorm">LayerNorm</a></li>
    <li><a class="contents-el" href="#embedding">Embedding</a></li>
    <li><a class="contents-el" href="#positional-embedding">Positional Embedding</a></li>
    <li><a class="contents-el" href="#attention-layer">Attention Layer</a></li>
    <li><a class="contents-el" href="#mlp">MLP</a></li>
    <li><a class="contents-el" href="#transformer-block">Transformer Block</a></li>
    <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
    <li><a class="contents-el" href="#full-transformer">Full Transformer</a></li>
    </ul></li>
    <li><a class="contents-el" href="#try-it-out">Try it out!</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Sampling
""")
    st.info(r"""
## Learning Objectives

* Understand that a transformer is composed of attention heads and MLPs, with each one performing operations on the residual stream
* Understand that the attention heads in a single layer operate independently, and that they have the role of calculating attention patterns (which determine where information is moved to & from in the residual stream)
* Implement the following transformer modules:
    * LayerNorm (transforming the input to have zero mean and unit variance)
    * Positional embedding (a lookup table from position indices to residual stream vectors)
    * Attention (the method of computing attention patterns for residual stream vectors)
    * MLP (the collection of linear and nonlinear transformations which operate on each residual stream vector in the same way)
    * Embedding (a lookup table from tokens to residual stream vectors)
    * Unembedding (a matrix for converting residual stream vectors into a distribution over tokens)
* Combine these first four modules to form a transformer block, then combine these with an embedding and unembedding to create a full transformer
* Load in weights to your transformer, and demo it on a sample input
""")



func_page_list = [
    (section_home, "🏠 Home"), 
    (section_training, "1️⃣ Training"),
    (section_sampling, "2️⃣ Sampling and Caching"),
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = {page: idx for idx, (func, page) in enumerate(func_page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

page()
