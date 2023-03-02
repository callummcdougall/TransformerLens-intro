import os
if not os.path.exists("./images"):
    os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

# st_image("fourier.png", 350)

def section_home():
    st_image("wheel3.png", 350)
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#problem-setup">Problem Setup</a></li>
    <li><a class="contents-el" href="#summary-of-the-algorithm">Summary of the algorithm</a></li>
    <li><a class="contents-el" href="#notation">Notation</a></li>
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#overview-of-content">Overview of content</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
## Introduction

Our goal for today is to reverse-engineer a one-layer transformer trained on modular addition! It turns out that the circuit responsible for this involves discrete Fourier transforms and trigonometric identities. This is perhaps the most interesting circuit for solving an algorithmic task that has been fully reverse-engineered thus far.

These exercises are adapted from the [original notebook](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20) by Neel Nanda and Tom Lierberum (and to a lesser extent the [accompanying paper](https://arxiv.org/abs/2301.05217)). We'll mainly be focusing on mechanistic analysis of this toy model, rather than replicating the grokking results (these may come in later exercises).

## Problem Setup

The model we will be reverse-engineering today is a one-layer transformer, with no layer norm and learned positional embeddings. $d_{model} = 128$, $n_{heads} = 4$, $d_{head}=32$, $d_{mlp}=512$. 

The task this model was trained on is addition modulo the prime $p = 113$. The input format is a sequence of three tokens `[x, y, =]`, with $d_{vocab}=114$ (integers from $0$ to $p - 1$ and $=$). The prediction for the next token after `=` should be the token corresponding to $x + y \pmod{p}$.
""")
    st_image("basic_schematic.png", 480)
    st.markdown(r"""

It was trained with full batch training, with 0.3 of the total data as training data. It is trained with AdamW, with $lr=10^{-3}$ and very high weight decay ($wd=1$).

## Summary of the algorithm

Broadly, the algorithm works as follows:
""")
    st.success(r"""
* Given two one-hot encoded tokens $x, y \in \{0, 1, \ldots, p - 1\}$, map these to $\sin(\omega x)$, $\cos(\omega x)$, $\sin(\omega y)$, $\cos(\omega y)$, where $\omega = \omega_k = \frac{2k\pi}{p}, k \in \mathbb{N}$.
    * In other words, we throw away most frequencies, and only keep a handful of **key frequencies** corresponding to specific values of $k$.
* Calcuates the quadratic terms $\cos(\omega x) \cos(\omega y)$, $\sin(\omega x) \sin(\omega y)$, $\cos(\omega x) \sin(\omega y)$, $\sin(\omega x) \cos(\omega y)$ by multiplying pairs of waves in $x$ and $y$.
* For each output logit $z$, compute $\cos(\omega (x + y - z))$ using the trig identity:
    $$
    \cos(\omega (x + y - z)) = \cos(\omega (x+y)) \cos(\omega z) + \sin(\omega (x+y)) \sin(\omega z)
    $$
    since this is a linear function of our input frequencies. 
* These values (for different $k$) will be added together to get our final output.
    * There is constructive interference at $z^* = x + y \mod p$, and destructive interference everywhere else - hence we get accurate predictions.
""")
    st.markdown(r"""
## Notation
    
A few words about notation we'll use in these exercises, to help remove ambiguity:

* $x$ and $y$ will always refer to the two inputs to the model. We'll also sometimes use the terminology $t_0$ and $t_1$, to refer to these two inputs as one-hot encoded tokens.
* The third input token, `=` will always be referred to as $t_2$. Unlike $t_0$ and $t_1$, this token is always the same in every input sequence.
* $t$ will refer to the matrix of all three one-hot encoded tokens, i.e. it has size $(3, d_{vocab})$. Here, we have $d_{vocab} = p + 1$ (since we have all the numbers from $0$ to $p - 1$, and the token `=`.)
* $z$ will always refer to the output of the model. For instance, when we talk about the model "computing $\cos(\omega (x + y - z))$", this means that the vector of output logits is the sequence:

$$
(\cos(\omega (x + y - z)))_{z = 0, 1, ..., p-1}
$$

* We are keeping TransformerLens' convention of left-multiplying matrices. For instance:
    * the embedding matrix $W_E$ has shape $(d_{vocab}, d_{model})$,
    * $t_0 ^T W_E$ is the embedding of the first token,
    * and $t W_E$ is the embedding of all tokens.
""")
    st.markdown(r"""
## Imports

```python
import torch as t
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import os

import plotly.express as px
import plotly.graph_objects as go

from functools import *

from typing import List, Tuple, Union, Optional
from fancy_einsum import einsum
import einops
from torchtyping import TensorType as TT

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

from my_utils import *
import tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

root = Path('Grokking/saved_runs')
large_root = Path('Grokking/large_files')
```

## Overview of content
    
These exercises will be structured as follows:

### 1️⃣ Periodicity & Fourier basis

This section gets you acquainted with the toy model. You'll do some initial investigations, and see that the activations are highly periodic. You'll also learn how to use the Fourier basis to represent periodic functions.
""")
    
    st.info(r"""
#### Learning Objectives

* Understand the problem statement, the model architecture, and the corresponding and functional form of any possible solutions.
* Learn about the Fourier basis (1D and 2D), and how it can be used to represent arbitrary functions.
* Understand that periodic functions are sparse in the Fourier basis, and how this relates to the model's weights.
""")
    
    st.markdown(r"""
### 2️⃣ Circuits & Feature Analysis

In this section, you'll apply your understanding of the Fourier basis and the periodicity of the model's weights to break down the exact algorithm used by the model to solve the task. You'll verify your hypotheses in several different ways.

""")
    st.info(r"""
#### Learning Objectives

* Apply your understanding of the 1D and 2D Fourier bases to show that the activtions / effective weights of your model are highly sparse in the Fourier basis.
* Turn these observations into concrete hypotheses about the model's algorithm.
* Verify these hypotheses using statistical methods, and interventions like ablation.
* Fully understand the model's algorithm, and how it solves the task.
""")
    st.markdown(r"""
### 3️⃣ Analysis During Training

In this section, you'll have a look at how the model evolves during the course of training. This section is optional, and the observations we make are more speculative than the rest of the material.
""")
    st.info(r"""
#### Learning Objectives

* Understand the idea of tracking metrics over time, and how this can inform when certain circuits are forming.
* Investigate and interpret the evolution over time of the singular values of the model's weight matrices.
* Investigate the formation of other capabilities in the model, like commutativity.
""")
    st.markdown(r"""
### 4️⃣ Discussion & Future Directions

Finally, we conclude with a discussion of these exercises, and some thoughts on future directions it could be taken.
""")
    
def section_intro():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#model-architecture">Model architecture</a></li>
    <li><a class="contents-el" href="#helper-variables">Helper variables</a></li>
    <li><a class="contents-el" href="#functional-form">Functional form</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#simplifying-assumptions">Simplifying Assumptions</a></li>
        <li><a class="contents-el" href="#mathematical-analysis">Mathematical Analysis</a></li>
        <li><a class="contents-el" href="#everything-is-periodic">Everything is periodic</a></li>
    </ul></li>
    <li><a class="contents-el" href="#fourier-transforms">Fourier Transforms</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#1d-fourier-basis">1D Fourier Basis</a></li>
        <li><a class="contents-el" href="#2d-fourier-basis">2D Fourier Basis</a></li>
    </ul></li>
    <li><a class="contents-el" href="#analysing-our-model-fourier-transforms">Analysing our model Fourier Transforms</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#plotting-activations-in-the-fourier-basis">Plotting activations in the Fourier basis</a></li>
        <li><a class="contents-el" href="#plotting-effective-weights-in-the-fourier-basis">Plotting effective weights in the Fourier basis</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Periodicity & Fourier Basis

## Model architecture

First, let's define our model, and some useful activations and weights as shorthand.

To review the information given in the previous page:

> *The model we will be reverse-engineering today is a one-layer transformer, with no layer norm and learned positional embeddings. $d_{model} = 128$, $n_{heads} = 4$, $d_{head}=32$, $d_{mlp}=512$.*
> 
> *The task this model was trained on is addition modulo the prime $p = 113$. The input format is a sequence of three tokens `[x, y, =]`, with $d_{vocab}=114$ (integers from $0$ to $p - 1$ and $=$). The prediction for the next token after `=` should be the token corresponding to $x + y \pmod{p}$.*

Run the code below to define your model:

```python
p = 113

cfg = HookedTransformerConfig(
    n_layers = 1,
    d_vocab = p+1,
    d_model = 128,
    d_mlp = 4 * 128,
    n_heads = 4,
    d_head = 128 // 4,
    n_ctx = 3,
    act_fn = "relu",
    normalization_type = None,
    device = device
)

model = HookedTransformer(cfg)
```

Next, run the following code to get the relevant data into your directory:

```
!git clone https://github.com/neelnanda-io/Grokking.git
os.mkdir(large_root)
!pip install gdown
!gdown "1OtbM0OGQCtGHjvSz-7q-7FkL2ReXIpmH&confirm=t" -O Grokking/large_files/full_run_data.pth
```

Once this has finished, you can load in your weights, using these helper functions:

```python
full_run_data = t.load(large_root / 'full_run_data.pth')
state_dict = full_run_data["state_dicts"][400]

model = load_in_state_dict(model, state_dict)
model = fix_order_of_attn_calc(model)
```

## Helper variables

Let's define some useful variables, and print out their shape to verify they are what we expect: 

```python
# Helper variables
W_O = model.W_O[0]
W_K = model.W_K[0]
W_Q = model.W_Q[0]
W_V = model.W_V[0]
W_in = model.W_in[0]
W_out = model.W_out[0]
W_pos = model.W_pos
W_E = model.W_E[:-1]
final_pos_resid_initial = model.W_E[-1] + W_pos[2]
W_U = model.W_U[:, :-1]

print('W_O  ', tuple(W_O.shape))
print('W_K  ', tuple(W_K.shape))
print('W_Q  ', tuple(W_Q.shape))
print('W_V  ', tuple(W_V.shape))
print('W_in ', tuple(W_in.shape))
print('W_out', tuple(W_out.shape))
print('W_pos', tuple(W_pos.shape))
print('W_E  ', tuple(W_E.shape))
print('W_U  ', tuple(W_U.shape))
```

Note here - we've taken slices of the embedding and unembedding matrices, to remove the final row (which corresponds to the `=` token). We've done this so that we can peform a Fourier transform on these weights later on. From now on, when we refer to $W_E$ and $W_U$, we'll usually be referring to these smaller matrices. We've explicitly defined `final_pos_resid_initial` because this will be needed later.

Also note we've indexed many of these matrices by `[0]`, this is because the first dimension is the layer dimension and we only have one layer.

Next, we'll run our model on all data. It's worth being clear on what we're doing here - we're taking every single one of the $p^2 = 12769$ possible sequences, stacking them into a single batch, and running the model on them. This only works because, in this particular problem, our universe is pretty small. We'll use the `run_with_cache` method to store all the intermediate activations.

```python
all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
labels = t.tensor([fn(i, j) for i, j, _ in all_data]).to(device)
original_logits, cache = model.run_with_cache(all_data)
# Final position only, also remove the logits for `=`
original_logits = original_logits[:, -1, :-1]
original_loss = cross_entropy_high_precision(original_logits, labels)
print(f"Original loss: {original_loss.item()}")
```

You should get an extremely small loss!
""")
    
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - extract key activations

Some important activations which we'll be investigating later are the attention matrices and neuron activations. In the code below, you should define the following:

* `attn_mat`: the attention patterns for query token `=`, over all sequences. This should have shape `(batch, head, key_posn)`, which in this case is `(12769, 4, 3)`.
    * Note that we only care when `=` is the query token, because this is the position we get our classifications from.
* `neuron_acts_post`: the neuron activations **for the last sequence position**, after applying our ReLU function. This should have shape `(batch, d_mlp)`, which in this case is `(12769, 512)`.
    * Note again that we only care about the last sequence position - can you see why?
* `neuron_acts_pre`: same as above, but before applying ReLU.

You can check your results by printing the tensor shapes.

```python
# Your code here

print('Attention Patterns:', attn_mat.shape)
print('Neuron Activations:', neuron_acts_post.shape)
print('Neuron Activations Pre:', neuron_acts_pre.shape)
```
""")
        
        with st.expander("Solution"):
            st.markdown(r"""
```python
attn_mat: TT["batch", "head", "key_posn"] = cache['pattern', 0][:, :, 2]
neuron_acts_post: TT["batch", "d_mlp"] = cache['post', 0][:, -1]
neuron_acts_pre: TT["batch", "d_mlp"] = cache['pre', 0][:, -1]
```
""")
    st.markdown(r"""

## Functional form

Next, let's think about the functional form of our model's solution. 

### Simplifying Assumptions

Here are a few questions, designed to get you thinking about the problem and how it relates to the model's internals. You can find answers to all of them (as well as more thorough discussion of other points) in the dropdown below.

* Of the six distinct pieces of information fed into the model (three token embeddings and three positional embeddings), which ones are relevant for solving the modular addition task?
* What does this imply about the role of position embeddings?
* What should the attention pattern look like? Which parts of the attention pattern will even matter?
* What will the role of the direct path (i.e. embeddings -> unembeddings, without any MLP or attention) be? How about the path that goes through the MLP layer but not the attention layer?
* What kinds of symmetries to you expect to see in the model? 
""")

    with st.expander("Solution"):
        st.markdown(r"""
The position embeddings are irrelevant, since addition is commutative. In fact, this results in these position embeddings being approximately symmetric. Only the token embeddings of the first two tokens are relevant, since the last token is always `=`. The attention pattern should be such that position `2` pays attention only to positions `0` and `1`, since position `2` has constant embeddings and provides no relevant information. (Note that it could act as a bias term; however, this is discouraged by the use of heavy weight decay during training and does not occur empirically.)

The direct path provides no relevant information and hence only acts as a bias term. Empirically, ablating the residual stream to zero before applying the unembedding matrix does not hurt performance very much. The same goes for the path through the MLP layer but not the attention layer (because information can't move from the `x`, `y` tokens to the token we'll use for prediction). 

As mentioned, addition is commutative, so we expect to see symmetries betweeen how the model deals with the first two tokens. Evidence for this:

* We can look at the difference between the position embeddings for pos 0 and pos 1 and see that they are close together and have high cosine similarity.
* We look at the difference between the neuron activations and the transpose of the neuron activations (i.e. compare $N(x, y)$ and $N(y, x)$) and see that they are close together.

```python
# Get the first three positional embedding vectors
W_pos_x, W_pos_y, W_pos_equals = W_pos

# Look at the difference between positional embeddings; show they are symmetric
def compare_tensors(v, w):
    return ((v-w).pow(2).sum()/v.pow(2).sum().sqrt()/w.pow(2).sum().sqrt()).item()
print('Difference in position embeddings', compare_tensors(W_pos_x, W_pos_y))
print('Cosine similarity of position embeddings', t.cosine_similarity(W_pos_x, W_pos_y, dim=0).item())

# Compare N(x, y) and N(y, x)
neuron_acts_square = neuron_acts.reshape(p, p, d_mlp)
print('Difference in neuron activations for (x,y) and (y,x): {.2f}'.format(
    compare_tensors(
        neuron_acts_square, 
        einops.rearrange(neuron_acts_square, "x y d_mlp -> y x d_mlp")
    )
))
```

This makes sense, because addition is commutative! Positions 0 and 1 *should* be symmetric.

Evidence that attention from position 2 to itself is negligible - I plot the average attention to each position for each head across all data points, and see that $2\to 2$ averages to near zero (and so is almost always near zero, as attention is always positive), and $2\to 0$ and $2 \to 1$ both average to zero, as we'd expect from symmetry.

```python
imshow(attn_mat.mean(0), xaxis='Position', yaxis='Head', title='Average Attention by source position and head', text_auto=".3f")
```

""")
        st_image("half_attn_2.png", 800)
        st.markdown(r"""

(Note that we could use circuitsvis to plot these attention patterns, but here we don't lose anything by using Plotly, since our analysis of attention patterns isn't too complicated.)
""")

    st.markdown(r"""
### Mathematical Analysis
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - think about the functional form
""")
        st.error(r"""
*Note, this exercise is quite tricky. You're recommended to spend at least 15 minutes working through this, because it's good practice to think about the functional form of a transformer. But don't be afraid to look at the solution if you get stuck, because there are quite a few steps involved!*
""")
        st.markdown(r"""
There exists a [comprehensive mathematical framework for understanding transformer circuits](https://transformer-circuits.pub/2021/framework/index.html), as you may have encountered in previous exercises. However, we will not need the full power of that framework today to understand the circuit responsible for modular additon, both because our model only has a single layer and because our task has some special structure that we can exploit.

Consider the following simplifying assumptions about our model:

* The position embeddings are irrelevant and can be zero-ablated without loss of performance;
* The residual stream is irrelevant and can be mean-ablated without loss of performance;
* For every head, position `2` only pays attention to positions `0` and `1`.

Write down the function $\ell = f(t)$ computed by the model, where $\ell \in \mathbb{R}^p$ is a vector of logits for each token and $t \in \mathbb{R}^{2 \times p}$ is a one-hot vector representing the input integers $m$ and $n$. Simplify the expression obtaind as far as possible. What can be said about it?
""")
        with st.expander("Hint - diagram"):
            st.markdown(r"""
Here is a diagram making the different computational stages more explicit. Can you use this to write down a closed-form expression for $f(t)$?
""")
            st_image("functional_form.png", 800)
        with st.expander("Solution"):
            st.markdown(r"""
Let's work through the model step-by-step. Let $n_\mathrm{seq} = 3$ denote the sequence length, so our (one-hot encoded) input tokens are $t \in \mathbb{R}^{n_\mathrm{seq} \times p}$. This contains the one-hot encoded integers $t_0$ and $t_1$, as well as the one-hot encoded equals sign $t_2$ (which is the same for all inputs). After applying the embedding matrix $W_E \in \mathbb{R}^{p \times d_\mathrm{model}}$, we get the embeddings:

$$
x = t W_E \in \mathbb{R}^{n_\mathrm{seq} \times d_\mathrm{model}}.
$$

Our function will look something like:

$$
f(t) = \operatorname{MLP}(\operatorname{Attn}(tW_E))W_U
$$

where $\operatorname{MLP}$ denotes the MLP layer, and $\operatorname{Attn}$ denotes the part of the attention layer determining what information gets moved to **position 2** (since this is where we take our predictions from). Note, this ignores all other residual stream terms. The only other paths which might be important are those going through the attention layer but not the MLP, but we can guess that by far the most significant ones will be those going through both.

Let's first address the MLP, because it's simpler. The functional form is just:

$$
\operatorname{MLP}(v) = \operatorname{ReLU}\left(v W_{in}\right)W_{out}
$$

where $v \in \mathbb{R}^{n_\mathrm{seq} \times d_\mathrm{model}}$ is the residual stream (after applying attn). 

Now let's think about the attention. We have:

$$
\begin{aligned}
\operatorname{Attn}(v)&=\sum_h \operatorname{softmax}\left(\frac{v_2^{\top} W_Q^h (W_K^h)^T\left[v_0 \, v_1\right]}{\sqrt{d_{head}}}\right)\left[v_0\, v_1\right]^T W_V^h W_O^h \\
&= \sum_h (\alpha^h v_0 + (1 - \alpha^h) v_1)^T W_V^h W_O^h \\
&\in \mathbb{R}^{d_{model}}
\end{aligned}
$$

where $v_0, v_1, v_2 \in \mathbb{R}^{d_{model}}$ are the the three embedding vectors in the residual stream, and $\alpha^h$ is the attention probability that token 2 pays to token 0 in head $h$. Note that we've ignored the attention paid by token 2 to itself, because we've seen that this is near zero. So the softmax is over the keys $v_0$ and $v_1$.

Can we simplify the formula for $\alpha^h$? As it turns out, yes. We're softmaxing over 2 dimensions, which is equivalent to sigmoid of the difference between logits:

$$
\operatorname{softmax}\left(\begin{array}{c}
\alpha \\
\beta
\end{array}\right)=\left(\begin{array}{c}
e^\alpha / (e^\alpha+e^\beta) \\
e^\beta / (e^\alpha+e^\beta)
\end{array}\right) = \left(\begin{array}{c}
\sigma(\alpha-\beta) \\
1-\sigma(\alpha-\beta)
\end{array}\right)
$$

so we can write:

$$
\begin{aligned}
\alpha^h &= \sigma\left(\frac{v_2^{\top} W_Q^h (W_K^h)^Tv_0}{\sqrt{d_{head}}} - \frac{v_2^{\top} W_Q^h (W_K^h)^Tv_1}{\sqrt{d_{head}}}\right) \\
&= \sigma\left(\frac{v_2^{\top} W_Q^h (W_K^h)^T(v_0 - v_1)}{\sqrt{d_{head}}}\right) \\
\end{aligned}
$$

Now, let's put both of these two together. We have the functional form as:

$$
f(t)=\operatorname{ReLU}\left(\sum_n\left(\alpha^h t_x+\left(1-\alpha^h\right) t_y\right)^T W_E W_V^h W_O^h W_{in}\right) W_{out} W_U
$$

---

Now that we have the funcional form, we can observe that the model's behaviour is fully determined by a handful of matrices, which we call **effective weight matrices**. They are:

* $W_{logit} := W_{out} W_U$, which has size $(d_{mlp}, d_{vocab}-1)$, and tells us how to get from the output of the nonlinear activation function to our final logits.

* $W_{neur} := W_E W_V W_O W_{in}$, which has size $(n_{heads}, d_{vocab}-1, d_{mlp})$ (we're stacking the OV matrices for each head along the zeroth dimension). This tells us how to get from a weighted sum of initial embeddings, to our neuron activations. 

* $W_{attn} := (t_2^T W_E) W_Q^h (W_K^h)^T W_E^T / \sqrt{d_{head}}$, which has size $(n_{heads}, d_{vocab}-1)$. This is the set of row (one vector per head) which we  we dot with $(t_0 - t_1)$, to give us our attention scores.

We can see how they all act in the transformer:

$$
f(t)=\operatorname{ReLU}\Bigg(\sum_h\underbrace{\bigg(\alpha^h t_0\;+\;\left(1\;-\;\alpha^h\right) t_1 \bigg)^T}_{\textstyle{\alpha^h = \sigma(W_{attn}^h(t_0 - t_1))}}  \underbrace{W_E W_V^h W_O^h W_{in}}_{\textstyle{W_{neur}^h}}\Bigg) \;\underbrace{W_{out} W_U}_{\textstyle{W_{logit}}}
$$

Note - the $W_E$ and $W_U$ above mostly refer to the reduced matrices (hence the sizes being $d_{vocab}-1$). This is because $t_0$ and $t_1$ can only ever be the integers $0, 1, ..., p-1$, and the only logit output we care about are those corresponding to integers. The only exception is when we define $W_{attn}$, because the $t_2^T W_E$ term is equal to the **last row of the *full* embedding matrix.**
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - define the effective weight matrices

In the solutions above, we identified three effective weight matrices which collectively determine the behaviour of the transformer. Below, you should calculate these three matrices directly from the model. Don't worry about using any kind of factored matrices; the model isn't large enough for this to be necessary.

*Gotcha - indexing via e.g. `model.W_Q` will return a 4D tensor, since the first dimension is the layer index (even though our model only has one layer). Make sure you remove this dimension.*

```python
# Your code here: define these matrices
W_logit = None
W_neur = None
W_attn = None

assert W_logit.shape == (cfg.d_mlp, cfg.d_vocab - 1)
assert W_neur.shape == (cfg.n_heads, cfg.d_vocab - 1, cfg.d_mlp)
assert W_attn.shape == (cfg.n_heads, cfg.d_vocab - 1)
""")
        with st.expander("Solution"):
            st.markdown(r"""
All these examples use the `@` operator. You might be more comfortable with `einsum` since it's more explicit and harder to make errors, but there's nothing wrong with using `@` if you're already comfortable with the sizes of the matrices in question, and how they multiply. 

```python
W_logit = W_out @ W_U

# Note - W_OV is 3D becuase first dim is head_idx, but the `@` operator detects this 
# and treats W_OV as a batch of matrices, which is exactly # what we want.
W_OV = W_V @ W_O
W_neur = W_E @ W_OV @ W_in

W_QK = W_Q @ W_K.transpose(-1, -2)
W_attn = final_pos_resid_initial @ W_QK @ W_E.T / (cfg.d_head ** 0.5)
```
""")
    st.markdown(r"""
### Everything is periodic

Any initial investigation and visualisation of activations and of the above effective weight matrices shows that things in the vocab basis are obviously periodic. Run the cells below, and demonstrate this for yourself.

#### Activations

**Attention patterns:**

The heatmap generated from the code below is a $p\times p$ image, where the cell $(x, y)$ represents some activation (ie a real number at a hidden layer of the network) on the input $x$ and $y$.

**Note:** Animation sliders are used to represent the different heads, not as a time dimension.

**Note:** $A_{2\to 2}^h\approx 0$, so $A_{2\to 1}^h = 1-A_{2\to 0}^h$. For this reason, the first thing we do below is redefine `attn_mat` to only refer to the attention paid to the first two tokens.

**Note:** We start by rearranging the attention matrix, so that the first two dimensions represent the (x, y) coordinates in the modular arithmetic equation. This is the meaning of the plots' axes.

```python
# Ignore attn from 2 -> 2
attn_mat = attn_mat[:, :, :2]

# Rearrange attn_mat, so the first two dims represent (x, y) in modular arithmetic equation
attn_mat_sq = einops.rearrange(attn_mat, "(x y) head seq -> x y head seq", x=p)

inputs_heatmap(
    attn_mat_sq[..., 0], 
    title=f'Attention score for heads at position 0',
    animation_frame=2,
    animation_name='head'
)
```

**Neuron activations:**

```python
# Rearrange activations, so the first two dims represent (x, y) in modular arithmetic equation
neuron_acts_post_sq = einops.rearrange(neuron_acts_post, "(x y) d_mlp -> x y d_mlp", x=p)
neuron_acts_pre_sq = einops.rearrange(neuron_acts_pre, "(x y) d_mlp -> x y d_mlp", x=p)

top_k = 3
inputs_heatmap(
    neuron_acts_post_sq[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
```

**Effective weights:**

**W_neur:**

```python
top_k = 5
animate_multi_lines(
    W_neur[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Input token', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attention)'
)
```

**W_attn:**

```python
lines(
    W_attn,
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token',
    yaxis='Contribution to attn score',
    title=f'Contribution to attention score (pre-softmax) for each head'
)
```

All this periodicity might make us think that the vocabulary basis isn't the most natural one to be operating in. The question is - what is the appropriate basis?

## Fourier Transforms
""")
    
    st.info(r"""
TL;DR: We can define a Fourier basis of cosine and sine waves with period dividing $p$ (i.e. frequency that's a multiple of $2 \pi / p$). We can apply a change of basis of the vocab space into the Fourier basis, and periodic functions are sparse in the Fourier basis. For activations that are a function of just one input we use the 1D Fourier transform; for activations that are a function of both inputs we use the 2D Fourier transform.""")
    
    st.markdown(r"""

A natural way to understand what's going on is using Fourier transforms. This represents any function as a sum of sine and cosine waves. Everything here is discrete, which means our functions are just $p$ or $p^2$ dimensional vectors, and the Fourier transform is just a change of basis. All functions have *some* representation in the Fourier basis, but "this function looks periodic" can be operationalised as "this function is sparse in the Fourier basis".

Note that we are applying a change of basis to $\mathbb{R}^p$, corresponding to the vocabulary space of one-hot encoded input vectors. (We are abusing notation by pretending that `=` is not in our vocabulary, so $d_\mathrm{vocab} = p$, allowing us to take Fourier transforms over the input space.)

### 1D Fourier Basis

We define the 1D Fourier basis as a list of sine and cosine waves. **We begin with the constant wave (and then add cosine and sine waves of different frequencies.** The waves need to have period dividing $p$, so they have frequencies that are integer multiples of $2 \pi / k$. We'll use the convention that $\cos k$ means the wave $\cos \left(\frac{2 \pi k}{p} x\right)$ for $x = 0, ..., p-1$, and that $F$ is the $p \times p$ matrix where each row is one such wave (either cos or sin).

Each term is scaled to have norm 1. Note that the constant term is scaled by $\frac{1}{p}$, the rest by $\frac{\sqrt{2}}{p}$. You can scale by these factors directly, or just divide the vectors by their norm.

Note also that the waves (esp. at high frequencies) look jagged, not smooth. This is because we discretise the inputs to just be integers, rather than all reals.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - create the 1D Fourier basis

Complete the function below. Don't worry about computational efficiency; using a for loop is fine.

```python
from typing import List, Tuple

def make_fourier_basis(p: int) -> Tuple[torch.Tensor, List[str]]:
    '''
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g. 
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    '''
    pass
```

Once you've done this (and passed the tests), you can run the cell below to visualise your Fourier components. 

```python
fourier_basis, fourier_basis_names = make_fourier_basis(p)

animate_lines(
    fourier_basis, 
    snapshot_index=fourier_basis_names, 
    snapshot='Fourier Component', 
    title='Graphs of Fourier Components (Use Slider)'
)
```

*Note - from this point onwards, the `fourier_basis` and `fourier_basis_names` variables are global, so you'll be using them in other functions. We won't be changing the value of `p`; this is also global.*
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def make_fourier_basis(p: int) -> Tuple[t.Tensor, List[str]]:
    '''
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g. 
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    '''
    # Define a grid for the Fourier basis vecs (we'll normalize them all at the end)
    # Note, the first vector is just the constant wave
    fourier_basis = t.ones(p, p)
    fourier_basis_names = ['Const']
    for i in range(1, p // 2 + 1):
        # Define each of the cos and sin terms
        fourier_basis[2*i-1] = t.cos(2*t.pi*t.arange(p)*i/p)
        fourier_basis[2*i] = t.sin(2*t.pi*t.arange(p)*i/p)
        fourier_basis_names.extend([f'cos {i}', f'sin {i}'])
    # Normalize vectors, and return them
    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis.to(device), fourier_basis_names
```
""")
        st.markdown(r"""
Now, you can prove the fourier basis is orthonormal by showing that the inner product of any two vectors is one if they are the same vector, and zero otherwise. Run the following cell to see for yourself:

```python
imshow(fourier_basis @ fourier_basis.T)
```

Remember that each of the **rows** of the matrix `fourier_basis` is a Fourier basis vector, so the (i, j)th element of the matrix above is the dot product of the ith and jth Fourier basis vectors.
""")

    st.markdown(r"""
Now that we've shown the Fourier transform is indeed an orthonormal basis, we can write any $p$-dimensional vector in terms of this basis. The **1D Fourier transform** is just the transformation taking the components of a vector in the standard basis to its components in the Fourier basis (in other words we project the vector along each of the Fourier basis vectors).
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - Fourier transform

You should now write a function to compute the Fourier transform of a vector. Remember that the **rows** of `fourier_basis` are the Fourier basis vectors.

```python
def fft1d(x: t.Tensor) -> t.Tensor:
    '''
    Returns the 1D Fourier transform of `x`,
    which can be a vector or a batch of vectors.
    '''
    pass
```
""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
def fft1d(x: t.Tensor) -> t.Tensor:
    '''
    Returns the 1D Fourier transform of `x`,
    which can be a vector or a batch of vectors.
    '''
    return x @ fourier_basis.T
```

Note - if `x` was a vector, then returning `fourier_basis @ x` would be perfectly fine. But if `x` is a batch of vectors, then we want to make sure the multiplication happens along the last dimension of `x`.
""")
    st.markdown(r"""
We can demonstrate this transformation on an example function which looks periodic. The key intuition is that function looks periodic in the original basis' $\implies$ 'function is sparse in the Fourier basis'. Note that functions over the integers $[0, p-1]$ are equivalent to vectors in $\mathbb{R}^p$, since we can associate any such function $f$ with the vector: 

$$
\begin{bmatrix}
f(0) \\
f(1) \\
\vdots \\
f(p-1)
\end{bmatrix}
$$

```python
v = sum([
    fourier_basis[4],
    fourier_basis[15]/5,
    fourier_basis[67]/10
])

line(v, xaxis='Vocab basis', title='Example periodic function')
line(fft1d(v), xaxis='Fourier Basis', title='Fourier Transform of example function', hover=fourier_basis_names)
```

You should observe a jagged but approximately periodic function in the first plot, and a very sparse function in the second plot (with only three non-zero coefficients).

### 2D Fourier Basis

**All of the above ideas can be naturally extended to a 2D Fourier basis on $\mathbb{R}^{p \times p}$, ie $p \times p$ images. Each term in the 2D Fourier basis is the outer product $v w^T$ of two terms $v, w$ in the 1D Fourier basis.**

This yields:

* a constant term, 
* linear terms of the form $\,\cos(wx),\,\sin(wx),\,\cos(wy)$, and $\sin(wy)$, 
* and quadratic terms of the form:

$$
\begin{aligned}
& \cos(w_ix)\cos(w_jy) \\
& \sin(w_ix)\cos(w_jy) \\
& \cos(w_ix)\sin(w_jy) \\
& \sin(w_ix)\sin(w_jy)
\end{aligned}
$$

Although we can think of these as vectors of length $p^2$, it makes much more sense to think of them as matrices of size $(p, p)$.
""")
    st.info(r"""
Notation - $\cos(w_i x)\cos(w_j y)$ should be understood as the $(p, p)$-size matrix constructed from the outer product of 1D vectors $\cos i$ and $\cos j$.

(Recall our notation for the 1D Fourier basis: $\cos i$ is the vector with elements $\cos(\omega_i x)$ for $x=0, \ldots, p-1$, where $\omega_i = \frac{2\pi i}{p}$.)
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - create the 2D Fourier basis

Complete the following function. Note that (unlike the function we wrote for the 1D Fourier basis) this only returns a single basis term, rather than the entire basis.

```python
def fourier_2d_basis_term(i: int, j: int) -> TT[p, p]:
    '''
    Returns the 2D Fourier basis term corresponding to the outer product of the
    `i`-th component of the 1D Fourier basis in the `x` direction and the `j`-th
    component of the 1D Fourier basis in the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    '''
    pass
```

Once you've defined this function, you can visualize the 2D Fourier basis by running the following code. Verify that they do indeed look periodic.

```python
x_term = 4
y_term = 6

inputs_heatmap(
    fourier_2d_basis_term(x_term, y_term).T,
    title=f"2D Fourier Basis term {fourier_basis_names[x_term]}x {fourier_basis_names[y_term]}y"
)
```

It's worth being clear on what this matrix represents. It is a 2D representation of a $p^2$-dimensional vector, one of $p^2$ basis vectors for $\mathbb{R}^{p^2}$. For an image of size $(p, p)$, we would extract the component of frequency $w_i$ in the $x$-direction and $w_j$ in the $y$-direction by taking the inner product of that image with this matrix.

Hence, you should observe periodicity matching `x_term` and `y_term`. Specifically, you should observe an x-period of 2 and a y-period of 3. You can play around with different values of `x_term` and `y_term` to get a feel for how this works. (Note that we've transposed the matrix in the code above, because the first dimension of a 2D matrix is generally treated as the vertical dimension.)
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def fourier_2d_basis_term(i: int, j: int) -> TT[p, p]:
    '''
    Returns the 2D Fourier basis term corresponding to the outer product of the
    `i`-th component of the 1D Fourier basis in the `x` direction and the `j`-th
    component of the 1D Fourier basis in the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    '''
    return (fourier_basis[i][:, None] * fourier_basis[j][None, :])
```

Note, indexing with `None` is one of many ways to write this function. A few others are:

* `torch.outer`
* Using einsum: `torch.einsum('i,j->ij', ...)`
* Using the `unsqueeze` method to add dummy dimensions to the vectors before multiplying them together.
""")
    st.markdown(r"""
What benefit do we get from thinking about $(p, p)$ images? Well, the batch dimension of all our data is of size $p^2$, since we're dealing with every possible value of inputs `x` and `y`. So we might think of reshaping this batch dimension to $(p, p)$, then applying a 2D Fourier transform to it.

Let's implement this transform now!
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - Implementing the 2D Fourier Transform

This exercise should be pretty familiar, since you've already done this in 1D. 

```python
def fft2d(tensor: t.Tensor) -> t.Tensor:
    '''
    Retuns the components of `tensor` in the 2D Fourier basis.

    Asumes that the input has shape `(p, p, ...)`.
    Output has the same shape as the input.
    '''
    pass
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def fft2d(tensor: t.Tensor) -> t.Tensor:
    '''
    Retuns the components of `tensor` in the 2D Fourier basis.

    Asumes that the input has shape `(p, p, ...)`.
    Output has the same shape as the input.
    '''
    # fourier_basis[i] is the i-th basis vector, which we want to multiply along
    return einops.einsum(
        tensor, fourier_basis, fourier_basis,
        "px py ..., i px, j py -> i j ..."
    )
```
""")
    st.markdown(r"""
While working with the 1D Fourier transform, we defined simple periodic functions which were linear combinations of the Fourier basis vectors, then showed that they were sparse when we expressed them in terms of the Fourier basis. That's exactly what we'll do here, but with functions of 2 inputs rather than 1.

Below is some code to plot a simple 2D periodic function (which is a linear combination of 2D Fourier basis terms). Note that we call our matrix `example_fn`, because we're thinking of it as a function of its two inputs (in the x and y directions). 

```python
example_fn = sum([
    fourier_2d_basis_term(4, 6), 
    fourier_2d_basis_term(14, 46)/3,
    fourier_2d_basis_term(97, 100)/6
])

inputs_heatmap(example_fn.T, title=f"Example periodic function")
```

Code to show this function is sparse in the 2D Fourier basis:

```python
imshow_fourier(
    fft2d(example_fn),
    title='Example period function in 2D Fourier basis'
)
```

You can run this code, and check that the non-zero components exactly match the basis terms we used to construct the function.

## Analysing our model Fourier Transforms

So far, we've made two observations:

* Many of our model's activations appear periodic
* Periodic functions appear sparse in the Fourier basis

So let's take the obvious next step, and apply a 2D Fourier transformation to our activations! Remember that the batch dimension of our activations is $p^2$, which can be rearranged into $(p, p)$, with these two dimensions representing the `x` and `y` inputs to our modular arithmetic equation. These are the dimensions over which we'll take our Fourier transform.

### Plotting activations in the Fourier basis

Recall our previous code, to plot the heatmap for the attention scores token 2 pays to token 0, for each head:

```python
inputs_heatmap(
    attn_mat[..., 0], 
    title=f'Attention score for heads at position 0',
    animation_frame=2,
    animation_name='head'
)
```

In that plot, the x and y axes represented the different values of inputs `x` and `y` in the modular arithmetic equation.

The code below takes the 2D Fourier transform of the attention matrix, and plots the heatmap for the attention scores token 2 pays to token 0, for each head, in the Fourier basis:

```python
# Apply Fourier transformation
attn_mat_fourier_basis = fft2d(attn_mat_sq)

# Plot results
imshow_fourier(
    attn_mat_fourier_basis[..., 0], 
    title=f'Attention score for heads at position 0, in Fourier basis',
    animation_frame=2,
    animation_name='head'
)
```

You should find that the result is extremely sparse - there will only be a few cells (mostly on the zeroth rows or columns, i.e. corresponding to the constant or linear terms) which aren't zero. This suggests that we're on the right track using the 2D Fourier basis!

Now, we'll do ths same for the neuron activations. Recall our previous code:

```python
top_k = 3
inputs_heatmap(
    neuron_acts_post[:, :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
```

We'll do the exact same here, and plot the activations in the Fourier basis:

```python
neuron_acts_post_fourier_basis = fft2d(neuron_acts_post_sq)

top_k = 3
imshow_fourier(
    neuron_acts_post_fourier_basis[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
```

Again, you should see sparsity, although this time some quadratic terms should be visible too.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - spot patterns in the activations

Increase `top_k` from 3 to a larger number, and look at different neurons. What do you notice about the patterns in the activations?
""")
        with st.expander("Answer"):
            st.markdown(r"""
There are 2 distinct patterns worth commenting on:

* Each neuron has the same pattern of non-zero terms: for some value of $k$, the non-zero terms are the constant term plus all linear and quadratic terms involving $k$ (i.e. $\sin(kx)$, $\cos(kx)$, $\sin(ky)$ and $\cos(ky)$).
* There are only a handful of different values of $k$ across all the neurons, so many of them end up having very similar-looking activation patterns.
""")
            st_image("activation-patterns.png", 600)
    st.markdown(r"""
### Plotting effective weights in the Fourier basis

As well as plotting our activations, we can also look at the weight matrices directly. 

*Note - this section isn't essential to understanding the rest of the notebook, so feel free to skip it if you're short on time.*

Recall our code to plot the contribution to neuron activations (pre-ReLU) via the OV circuit:

```python
top_k = 5
animate_multi_lines(
    W_neur[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Input token', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attention)'
)
```

Here, each line shows the activations for some neuron as a function of the input token `x` (if the `=` token at position 2 only paid attention to a token `x`), with the neuron index determined by the slider value. If this seems a bit confusing, you can use the dropdown below to remind yourself of the functional form of this transformer, and the role of $W_{neur}$.
""")
    with st.expander("Functional form"):
        st.markdown(r"""
$$
f(t)=\operatorname{ReLU}\Bigg(\sum_h{\bigg(\alpha^h t_0\;+\;\left(1\;-\;\alpha^h\right) t_1 \bigg)^T}{W_{neur}^h}\Bigg) \; W_{logit}
$$

From this, we can see clearly the role of $W_{neur}^h$. It is a matrix of shape $(d_{vocab}, d_{mlp})$, and its rows are the vectors we take a weighted average of to get our MLP activations (pre-ReLU).
""")
    st.markdown(r"""
The code below makes the same plot, but while the previous one was in the standard basis (with the x-axis representing the input token), in this plot the x-axis is the component of the input token in each Fourier basis direction. 

Note that we've provided you with the helper function `fft1d_given_dim`, which performs the 1D Fourier transform over a given dimension. This is necessary for `W_neur`, since it has shape `(n_heads, d_vocab, d_mlp)`, and we want to transform over the `d_vocab` dimension.

```python
def fft1d_given_dim(tensor: t.Tensor, dim: int) -> t.Tensor:
    '''
    Performs 1D FFT along the given dimension (not necessarily the last one).
    '''
    tensor = tensor.transpose(dim, -1)
    tensor = fft1d(tensor)
    tensor = tensor.transpose(dim, -1)
    return tensor

W_neur_fourier = fft1d_given_dim(W_neur, dim=1)

top_k = 5
animate_multi_lines(
    W_neur_fourier[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Fourier component', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    hover=fourier_basis_names,
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn), in Fourier basis'
)
```

Note that each line plot generally has $\sin k$ and $\cos k$ terms non-zero, rather than having one but not the other.

Lastly, we'll do the same with `W_attn`. Recall our previous code:

```python
lines(
    W_attn,
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token',
    yaxis='Contribution to attn score',
    title=f'Contribution to attention score (pre-softmax) for each head'
)
```

We amend this to:

```python
lines(
    fft1d(W_attn), 
    xaxis='Input token', 
    yaxis = 'Contribution to attn score',
    title=f'Contribution to attention score (pre-softmax) for each head, in Fourier Basis', 
    hover=fourier_basis_names
)
```

You may have noticed that the non-zero frequencies in both these last two line charts exactly match the important frequencies we read off the attention patterns!
""")
#     st.markdown(r"""
# ### Aside: Change of basis on the batch dimension

# It turns out that the natural basis to interpret the vocab space in is the Fourier basis, rather than the vocab basis (i.e. the basis where each basis vector is the one hot encoding of a different input). When we input the large batch of all possible data (in lexicographic order) the batch dimension, of size $p^2$, corresponds to pairs of inputs, and so we can apply the 2D Fourier transform to the batch dimension.

# A change of basis on the batch dimension is a pretty weird thing to do, and it's worth thinking carefully about what happens here (note that this is a significant deviation to the prior Transformer Circuits work, and only really makes sense here because this is such a toy problem that we can enter the entire universe as one batch).

# There are *four* operations that are not linear with respect to the batch dimension. As above, the attention softmax, ReLU and final softmax. But also the elementwise multiplication with the attention pattern. 

# In particular, ReLU becomes super weird - it goes from an elementwise operation to the operation 'rotate by the inverse of the Fourier basis, apply ReLU elementwise in the *new* basis, rotate back to the Fourier basis'
    st.markdown(r"""
---

Let's review what we've learned in this section. We found that:
""")
    st.success(r"""
- The simple architecture of our 1-layer model heavily constrains the functional form of any learned solutions.
    - In particular, we can define a handful of matrices which fully describe the model's behaviour (after making some simplifying assumptions). 
- Many of our model's internal activations appear periodic in the inputs `x`, `y` (e.g. the attention patterns and neuron activations).
- The natural way to represent a periodic function is in the Fourier basis. Periodic functions appear sparse in this basis.
- This suggests our model might only be using a handful of frequencies (i.e. projecting the inputs onto a few different Fourier basis vectors), and discarding the rest.
- We confirmed this hypothesis by looking at:
    - The model's activations (i.e. attention patterns and neuron activations)
    - The model's effective weight matrices (i.e. $W_{attn}$ and $W_{neur}$)
- Both these observations confirmed that we have sparsity in the Fourier basis. Furthermore, the same small handful of frequencies seemed to be appearing in all cases.
""")

def section_circuits():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#understanding-the-embedding">Understanding the embedding</a></li>
   <li><a class="contents-el" href="#understanding-neuron-activations">Understanding neuron activations</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#neurons-produce-quadratic-terms">Neurons produce quadratic terms</a></li>
    </li></ul>
    <li><a class="contents-el" href="#understanding-logit-computation">Understanding Logit Computation</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Circuit and Feature Analysis

Understanding a transformer breaks down into two high-level parts - interpreting the features represented by non-linear activations (output probabilities, attention patterns, neuron activations), and interpreting the circuits that calculate each feature (the weights representing the computation done to convert earlier features into later features). In this section we interpret the embedding, the neuron activations (a feature) and the logit computation (a circuit). These are the most important bits to interpet.

Let's start with the embedding.

## Understanding the embedding

Below is some code to plot the embedding in the Fourier basis. You should run this code, and interpret the output.

```python
line(
    (fourier_basis @ W_E).pow(2).sum(1), 
    hover=fourier_basis_names,
    title='Norm of embedding of each Fourier Component',
    xaxis='Fourier Component',
    yaxis='Norm'
)
```
""")
    with st.expander("Answer"):
        st.markdown(r"""
You should find that the embedding is sparse in the Fourier basis, and throws away all Fourier components apart from a handful of frequencies (the number of frequencies and their values are arbitrary, and vary between training runs).
""")
        st_image("fourier_norms.png", 500)
        st.markdown(r"""

The Fourier basis vector for component $2k$ is:

$$
\cos\left(\frac{2 \pi k}{p}x\right)_{x = 0, ..., p-1}
$$

and same for $2k-1$, but with $\cos$ replaced with $\sin$.

So this result tells us that, for the input `x`, we're keeping the information $\cos\left(\frac{2 \pi k}{p}x\right)$ for each of the key frequencies $k$, and throwing away this information for all other frequencies $\omega$.

Let us term the frequencies with non-trivial norm the key frequencies (here, 14, 31, 35, 41, 42, 52).
""")
    with st.expander("Another perspective (from singular value decomposition)"):
        st.markdown(r"""
Recall that we can write any matrix as the product of an orthogonal matrix, a diagonal matrix, and another orthogonal matrix:

$$
\begin{aligned}
A &= U S V^T \\
    &= \sum_{i=1}^k \sigma_i u_i v_i^T
\end{aligned}
$$

where $u_i$, $v_i$ are the column vectors for the orthogonal matrices $U$, $V$, and $\sigma_1, ..., \sigma_k$ are the non-zero singular values of $A$. If this isn't familiar, you might want to go through the induction heads exercises, which discusses SVD in more detail.

This is often a natural way to represent low-rank matrices.

Denore the matrix `fourier_basis` as $F$. Here, we've found that the matrix $F W_E$ is very sparse (i.e. most of its row vectors are zero). From this, we can deduce that $W_E$ is a low-rank matrix with SVD approximately:

$$
W_E \approx F^T S V^T
$$

because then left-multiplying by $F$ gives us a matrix with zeros everywhere except for the rows corresponding to non-zero singular values:

$$
F W_E \approx F^T F S V^T = S V^T
$$

To better visualise this, run the following:

```python
imshow_div(fourier_basis @ W_E)
```

To write $W_E$ as a sum, we have:

$$
W_E = \sum_{i=1}^k \sigma_i f_i v_i^T
$$

where $f_i$ are the row of $F$ (i.e. Fourier basis vectors), and $v_i$ are the corresponding **output directions** along which we write the information corresponding to these key frequencies:

$$
t_0^T W_E = \sum_{i=1}^k \sigma_i (t_0^T f_i) v_i^T
$$
""")

    st.markdown(r"""
## Understanding neuron activations

Now we've established that $W_E$ is only preserving the information corresponding to a few key frequencies, let's look at what those frequencies are actually being used for.
""")
    st.info(r"""
**TL;DR: Each neuron's activations are a linear combination of the constant term and the linear and quadratic terms of a specific frequency. The neurons clearly cluster according to the key frequencies.**
""")
    st.markdown(r"""
### Neurons produce quadratic terms

First, we recall the diagrams of the neuron activations in the input basis and the 2D Fourier basis, which we made in the previous section:

```python
top_k = 5
inputs_heatmap(
    neuron_acts_post_sq[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
imshow_fourier(
    neuron_acts_post_fourier_basis[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
```

We found that the first plot looked periodic, and the second clustered around some key frequencies (with each neuron having an associated frequency), from one of the key frequencies we observered earlier.
""")

# #### What do these plots mean?

# It's important to have a clear sense of what these plots actually represent. 

# The first plot is fairly straightforwards - the $(m, n)$-th value is the activation of that neuron on input `m n =`.

# The second plot is more complicated. We take the image corresponding to the first plot, and change the basis to the 2D Fourier Basis. The $(i, j)$-th value is the coefficient of the $(i, j)$-th Fourier term for the neuron's activation. 

# ##### Another framing

# In the last section, we established that the embedding matrix $W_E$ is low-rank, and for tokens `x`, it only preserves the information $\cos\left(\frac{2 \pi \omega}{p}x\right)$ for each of the key frequencies $\omega$. Another way of saying this:

# > If we projected the one-hot encoded tokens $t \in \mathbb{R}^p$ onto the 1D Fourier basis (i.e. $Ft$) and used this as our input instead, then the embedding matrix would be sparse (i.e. it would ignore most elements of this vector).

# The result here is analogous. It tells us that, if we project the one-hot encoded tokens for $x$ **and** $y$ onto the **2D Fourier basis**, then the linear map which takes us from this input to the activation patterns of any particular neuron is sparse. In other words, it is only a function of a handful of the elements of this $p \times p$ input vector.

# ---

    st.markdown(r"""
For instance, look at the first neuron. We can see that the only frequencies which matter in the 2D Fourier basis are the constant terms and the frequencies corresponding to $\omega = 42$ (we get both $\sin$ and $\cos$ terms). In total, this gives us nine terms, which are (up to scale factors):

$$
\begin{bmatrix}
1 & \cos(\omega_k x) & \sin(\omega_k x) \\
\cos(\omega_k y) & \cos(\omega_k x)\cos(\omega_k y) & \sin(\omega_k x)\cos(\omega_k y) \\
\sin(\omega_k y) & \cos(\omega_k x)\sin(\omega_k y) & \sin(\omega_k x)\sin(\omega_k y) 
\end{bmatrix}
$$

where $\omega_k = 2 \pi k / p$, and in this case $k = 42$. These include the constant term, four linear terms, and four quadratic terms.

What is the significance of this? Importantly, we have the following trig formulas:

$$
\begin{aligned}
\cos(\omega_k(x + y)) = \cos(\omega_k x) \cos(\omega_k y) - \sin(\omega_k x) \sin(\omega_k y) \\
\sin(\omega_k(x + y)) = \sin(\omega_k x) \cos(\omega_k y) + \cos(\omega_k x) \sin(\omega_k y)
\end{aligned}
$$

We know that some of the terms on the right of these equations (i.e. the quadratic terms) are the ones being detected by our neurons. So we might deduce that our neurons are in some sense storing the information $\cos(\omega_k(x + y))$ and $\sin(\omega_k(x + y))$, for different values of $\omega_k$.

Let's have a closer look at some of the coefficients for these 2D Fourier basis terms.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - calculate the mean squared coefficient

We might speculate that the neuron activation (which is a function of the inputs $x$, $y$) is in some sense trying to approximate the function $\sum_i \sum_j B_{i, j} F_{i, x} F_{j, y}$, where:

* $F$ is the Fourier change-of-basis matrix.
    * Note that, for example, $F_{1,17}= \cos(\frac{2\pi}{p} 17)$.
* $B$ is a matrix of coefficients, which we suspect is highly sparse.
    * Specifically, we suspect that the "true function" our model is trying to approximate looks something like a linear combination of the nine terms in the matrix above:
    
$$
\sum_{i\in \{0, 2\omega-1, 2\omega\}} \sum_{j\in \{0, 2\omega-1, 2\omega\}} B_{i,j} F_{i,x} F_{j,y}
$$

Create a heatmap of the mean squared coefficient for each neuron (in other words, the `(i, j)`-th value in your heatmap is the mean of $B_{i, j}^2$ across all neurons).

Your code should involve three steps:

* Centering the neuron activations, by subtracting the mean over all batches (because this essentially removes the bias term). All ReLU neurons always have non-negative activations, and the constant term should usually be considered separately.
* Taking the 2D Fourier transform of the centered neuron activations.
* Use the `imshow_fourier` function to plot the mean squared coefficient for each neuron (i.e. taking mean over all neurons). If `imshow_fourier` is fed a single 2D array as its first argument, it will plot this as a heatmap (with no sliders). 

```python
# Your code here
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
# Center activations
neuron_acts_centered = neuron_acts_post_sq - neuron_acts_post_sq.mean((0, 1), keepdim=True)

# Take 2D Fourier transform
neuron_acts_centered_fourier = fft2d(neuron_acts_centered)

# Plot
imshow_fourier(
    neuron_acts_centered_fourier.pow(2).mean(-1),
    title=f"Norms of 2D Fourier components of centered neuron activations",
)
```
""")
    st.markdown(r"""
Your plot of the average $B_{i, j}^2$ values should show the kind of sparsity you saw earlier: the only non-zero terms will be the const, linear and quadratic terms corresponding to a few select frequencies.
""")
    st.info(r"""
#### Aside - how do we get the quadratic terms?
    
Exactly how these quadratic terms are calculated is a bit convoluted. A good mental model for neural networks is that they are really good at matrix multiplication and addition, and anything else takes a lot of effort. So we should expect taking the product of two different parts of the input to be pretty hard!

You can see the [original notebook](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20#scrollTo=ByvnsimvZXGC) for some more details on this calculation. The short version - the model uses both ReLU activations and element-wise products with attention to multiply terms. The attention pattern products work because attention involves taking the product of value vectors and attention probabilities, which are each functions of the input. The ReLU activations are a bit more surprising - it turns out we can approximate quadratic terms:

$$
\operatorname{ReLU}(0.5 + \cos(\omega x) + \cos(\omega y)) \approx 0.71 + 0.695\cos(\omega x) + 0.695\cos(\omega y) + 0.43\cos(\omega x)\cos(\omega y)
$$

For the rest of these exercises, you should take it as an empirical fact that the neuron activations are able to compute these quadratic terms in the way you can see in your plots.
""")
    st.markdown(r"""
### Neurons cluster by frequency

Now that we've established that the neurons each seem to have some single frequency that they're most sensitive to (and ignore all others), let's try and sort the neurons by this frequency, and see how effective each of these frequencies are at explaining the neurons' behaviour. 
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - find neuron clusters

For each neuron, you should find the frequency such that the Fourier components containing that frequency explain the largest amount of the variance of that neuron's activation (in other words, which frequency is such that the sum of squares of the const, linear and quadratic terms of that frequency for this particular is largest, as a fraction of the sum of squares of all the Fourier coefficients for this neuron).

We've provided you with the helper function `arrange_by_2d_freqs`. This takes in a tensor of coefficients in the 2D Fourier basis (with shape `(p, p, ...)`), and returns a tensor of shape `(p//2, 3, 3, ...)`, representing the Fourier coefficients sorted into each frequency. In other words, the `[k-1, ...]`-th slice of this tensor will be the `(3, 3, ...)`-shape tensor containing the Fourier coefficients for the following (normalized) 2D Fourier basis vectors:

$$
\begin{bmatrix}
1 & \cos(\omega_k x) & \sin(\omega_k x) \\
\cos(\omega_k y) & \cos(\omega_k x)\cos(\omega_k y) & \sin(\omega_k x)\cos(\omega_k y) \\
\sin(\omega_k y) & \cos(\omega_k x)\sin(\omega_k y) & \sin(\omega_k x)\sin(\omega_k y) 
\end{bmatrix}
$$

Think of this as just a fancy rearranging of the tensor, so that you're more easily able to access all the (const, linear, quadratic) terms for any particular frequency.

```python
def arrange_by_2d_freqs(tensor):
    idx_2d_y_all = []
    idx_2d_x_all = []
    for freq in range(1, p//2):
        idx_1d = [0, 2*freq-1, 2*freq]
        idx_2d_x_all.append([idx_1d for _ in range(3)])
        idx_2d_y_all.append([[i]*3 for i in idx_1d])
    return tensor[idx_2d_y_all, idx_2d_x_all]


def find_neuron_freqs(
    fourier_neuron_acts: TT[p, p, d_mlp]
) -> Tuple[TT[d_mlp], TT[d_mlp]]:
    '''
    Returns the tensors `neuron_freqs` and `neuron_frac_explained`, 
    containing the frequencies that explain the most variance of each 
    neuron and the fraction  of variance explained, respectively.
    '''
    pass
    

neuron_freqs, neuron_frac_explained = find_neuron_freqs(neuron_acts_centered_fourier)
key_freqs, neuron_freq_counts = t.unique(neuron_freqs, return_counts=True)

assert key_freqs.tolist() == [14, 35, 41, 42, 52]
```
""")
        with st.expander("Help - all my key frequencies are off by one from the true answer."):
            st.markdown(r"""
Remember that the 0th slice of the `arrange_by_2d_freqs` function are actually the frequencies for $k=1$, not $k=0$ (and so on). If you get the key frequencies by argmaxing, then make sure you add one to these indices!
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def find_neuron_freqs(
    fourier_neuron_acts: TT[p, p, d_mlp]
) -> Tuple[TT[d_mlp], TT[d_mlp]]:
    '''
    Returns the tensors `neuron_freqs` and `neuron_frac_explained`, 
    containing the frequencies that explain the most variance of each 
    neuron and the fraction  of variance explained, respectively.
    '''
    fourier_neuron_acts_by_freq = arrange_by_2d_freqs(fourier_neuron_acts)
    assert fourier_neuron_acts_by_freq.shape == (p//2, 3, 3, d_mlp)

    # Sum squares of all frequency coeffs, for each neuron
    square_of_all_terms = einops.reduce(
        fourier_neuron_acts.pow(2),
        "x_coeff y_coeff neuron -> neuron",
        "sum"
    )

    # Sum squares just corresponding to const+linear+quadratic terms,
    # for each frequency, for each neuron
    square_of_each_freq = einops.reduce(
        fourier_neuron_acts_by_freq.pow(2),
        "freq x_coeff y_coeff neuron -> freq neuron",
        "sum"
    )

    # Find the freq explaining most variance for each neuron
    # (and the fraction of variance explained)
    neuron_variance_explained, neuron_freqs = square_of_each_freq.max(0)
    neuron_frac_explained = neuron_variance_explained / square_of_all_terms

    # The actual frequencies count up from k=1, not 0!
    neuron_freqs += 1

    return neuron_freqs, neuron_frac_explained
```

Note the use of `einops.reduce` here, rather than just using e.g. `fourier_neuron_acts.pow(2).sum((0, 1)). Like most of the situations where `einops` is helpful, this has the advantage of making your code more explicit, readable, and reduces the chance of mistakes.
""")
        st.markdown(r"""
Once you've written this function and passed the tests, you can plot the fraction of variance explained. 

```python
fraction_of_activations_positive_at_posn2 = (cache['pre', 0][:, -1] > 0).float().mean(0)

scatter(
    x=neuron_freqs, 
    y=neuron_frac_explained, 
    labels={'x':'Neuron frequency', 'y':'Frac explained', 'color': 'Frac positive'},
    color=utils.to_numpy(fraction_of_activations_positive_at_posn2)
)
```

We color the neurons according to the fraction of data points for which they are active. We see that there are 5 distinct clusters of neurons that are well explained (frac > 0.85) by one frequency.

There is a sixth, diffuse cluster of neurons that always fire. They are not well-explained by any particular frequency. This makes sense, because since ReLU acts as an identity on this cluster, there's no reason to privilege the neuron basis (i.e. there's no reason to expect that the specific value of this neuron's activations has any particular meaning in relation to the Fourier components of the input, since we could just as easily apply rotations to the always-firing neurons). 

```python
# To represent that they are in a special sixth cluster, we set the 
# frequency of these neurons to -1
neuron_freqs[neuron_frac_explained < 0.85] = -1.
key_freqs_plus = t.concatenate([key_freqs, -key_freqs.new_ones((1,))])

for i, k in enumerate(key_freqs_plus):
    print(f'Cluster {i}: freq k={k}, {(neuron_freqs==k).sum()} neurons')
```
""")

    st.markdown(r"""
### Further investigation of neuron clusters
    
We can separately view the norms of the Fourier Components of the neuron activations for each cluster. The following code should do the same thing as your plot of the average $B_{i, j}^2$ values earlier, except it sorts the neurons into clusters by their frequency before taking the mean. 

*(Note, we're using the argument `facet_col` rather than `animation_frame`, so we can see all the plots at once.)*

```python
fourier_norms_in_each_cluster = []
for freq in key_freqs:
    fourier_norms_in_each_cluster.append(
        einops.reduce(
            neuron_acts_centered_fourier.pow(2)[..., neuron_freqs==freq], 
            'batch_y batch_x neuron -> batch_y batch_x', 
            'mean'
        )
    )

imshow_fourier(
    t.stack(fourier_norms_in_each_cluster), 
    title=f'Norm of 2D Fourier components of neuron activations in each cluster',
    facet_col=0,
    facet_labels=[f"Freq={freq}" for freq in key_freqs]
)
```

Now that we've found what appear to be neuron clusters, it's time to validate our observations. We'll do this by showing that, for each neuron cluster, we can set terms for any other frequency to zero.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - validate neuron clusters

We want to do the following:

* Take `neuron_acts_post`, which is a tensor of shape `(p*p, p)`, with the `[i, j]`-th element being the activation of neuron `j` on the `i`-th input sequence in our `all_data` batch.
* Treating this tensor as `p` separate vectors in $\mathbb{R}^{p^2}$ (with the last dimension being the batch dimension), we will project each of these vectors onto the subspace of $\mathbb{R}^{p^2}$ spanned by the 2D Fourier basis vectors for the associated frequency of that particular neuron (i.e. the constant, linear, and quadratic terms).
* Take these projected `neuron_acts_post` vectors, and apply `W_logit` to give us new logits. Compare the cross entropy loss with these logits to our original logits.

If our hypothesis is correct (i.e. that for each cluster of neurons associated with a particular frequency, that frequency is the only one that matters), then our loss shouldn't decrease by much when we project out the other frequencies in this way.

First, you'll need to write the function `project_onto_direction`. This takes two inputs: `batch_vecs` (a batch of vectors, with batch dimension at the end) and `v` (a single vector), and returns the projection of each vector in `batch_vecs` onto the direction `v`.

```python
def project_onto_direction(batch_vecs: t.Tensor, v: t.Tensor) -> t.Tensor:
    '''
    Returns the component of each vector in `batch_vecs` in the
    direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    '''
    pass

    
tests.test_project_onto_direction(project_onto_direction)
```
""")
        with st.expander("Hint"):
            st.markdown(r"""
Recall that the projection of $w$ onto $v$ (for a normalized vector $v$) is given by:

$$
w_{proj} = (w \cdot v) v
$$

You might find it easier to do this in two steps: first calculate the components in the $v$-direction by taking the inner product along the `L`-dimension, then create a batch of multiples of $v$, scaled by these components.
""")
        with st.expander("Solution"):
            st.markdown(r"""
Two example solutions are given below. One is a single line, and uses clever indexing to add dummy dimensions to tensors. The other is a bit longer, but more explicit / readable.

```python
def project_onto_direction(batch_vecs: t.Tensor, v: t.Tensor) -> t.Tensor:
    '''
    Returns the component of each vector in `batch_vecs` in the
    direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    '''
    return v[:, None] @ (v @ batch_vecs)[None, ...]


def project_onto_direction(batch_vecs: t.Tensor, v: t.Tensor) -> t.Tensor:
    '''
    Returns the component of each vector in `batch_vecs` in the
    direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    '''

    # Get tensor of components of each vector in v-direction
    components_in_v_dir = einops.einsum(
        batch_vecs, v,
        "n ..., n -> ..."
    )

    # Use these components as coefficients of v in our projections
    return einops.einsum(
        components_in_v_dir, v,
        "..., n -> n ..."
    )
```
""")
        st.markdown(r"""
Next, you should write the function `project_onto_frequency`. This takes a batch of vectors with shape `(p**2, batch)`, and a frequency `freq`, and returns the projection of each vector onto the subspace spanned by the nine 2D Fourier basis vectors for that frequency (i.e. one constant term, four linear terms, and four quadratic terms).

```python
def project_onto_frequency(batch_vecs: t.Tensor, freq: int) -> t.Tensor:
    '''
    Returns the projection of each vector in `batch_vecs` onto the
    2D Fourier basis directions corresponding to frequency `freq`.
    
    batch_vecs.shape = (p**2, ...)
    '''
    assert batch_vecs.shape[0] == p**2

    pass
```
""")
        with st.expander("Hint"):
            st.markdown(r"""
This will just involve summing nine calls to the `project_onto_direction` function you wrote above (one for each basis vector), since your basis vectors are orthogonal. 

You should use the function `fourier_2d_basis_term` to get the vectors you'll be projecting onto. Remember to flatten these vectors, because you're working with vectors of length `p**2` rather than of size `(p, p)`!
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def project_onto_frequency(batch_vecs: t.Tensor, freq: int) -> t.Tensor:
    '''
    Returns the projection of each vector in `batch_vecs` onto the
    2D Fourier basis directions corresponding to frequency `freq`.
    
    batch_vecs.shape = (p**2, ...)
    '''
    assert batch_vecs.shape[0] == p**2

    return sum([
        project_onto_direction(
            batch_vecs,
            fourier_2d_basis_term(i, j).flatten(),
        )
        for i in [0, 2*freq-1, 2*freq] for j in [0, 2*freq-1, 2*freq]
    ])
```
""")
        st.markdown(r"""
Finally, run the following code to project out the other frequencies from the neuron activations, and compare the new loss. You should make sure you understand what this code is doing.

```python
logits_in_freqs = []

for freq in key_freqs:

    # Get all neuron activations corresponding to this frequency
    filtered_neuron_acts = neuron_acts_post[:, neuron_freqs==freq]

    # Project onto const/linear/quadratic terms in 2D Fourier basis
    filtered_neuron_acts_in_freq = project_onto_frequency(filtered_neuron_acts, freq)

    # Calcluate new logits, from these filtered neuron activations
    logits_in_freq = filtered_neuron_acts_in_freq @ W_logit[neuron_freqs==freq]

    logits_in_freqs.append(logits_in_freq)

# We add on neurons in the always firing cluster, unfiltered
logits_always_firing = neuron_acts_post[:, neuron_freqs==-1] @ W_logit[neuron_freqs==-1]
logits_in_freqs.append(logits_always_firing)

# Print new losses
print('Loss with neuron activations ONLY in key freq (incl always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Loss with neuron activations ONLY in key freq (excl always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs[:-1]), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Original loss\n{:.6e}'.format(original_loss))
```
""")
    st.markdown(r"""
We can also compare the importance of each cluster of neurons by ablating it (while continuing to restrict each cluster to its frequency). We see from this that `freq=52` is the most important cluster (because the loss increases a lot when this is removed), although clearly all clusters are important (because the loss is still very small if any one of them is ablated). We also see that ablating the always firing cluster has a very small effect, so clearly this cluster isn't very helpful for the task.

```python
    print('Loss with neuron activations excluding none:     {:.9f}'.format(original_loss.item()))
    for c, freq in enumerate(key_freqs_plus):
        print('Loss with neuron activations excluding freq={}:  {:.9f}'.format(
            freq, 
            test_logits(
                sum(logits_in_freqs) - logits_in_freqs[c], 
                bias_correction=True, 
                original_logits=original_logits
            )
        ))
```

## Understanding logit computation
""")
    st.info(r"""
TLDR: The network uses $W_{logit}=W_{out}W_U$ to cancel out all 2D Fourier components other than the directions corresponding to $\cos(w(x+y)),\sin(w(x+y))$, and then multiplies these directions by $\cos(wz),\sin(wz)$ respectively and sums to get the output logits.""")

    st.markdown(r"""
Recall that each neuron's activations are a linear combination of const, linear and quadratic terms of our key frequency:

$$
\begin{bmatrix}
1 & \cos(\omega_k x) & \sin(\omega_k x) \\
\cos(\omega_k y) & \cos(\omega_k x)\cos(\omega_k y) & \sin(\omega_k x)\cos(\omega_k y) \\
\sin(\omega_k y) & \cos(\omega_k x)\sin(\omega_k y) & \sin(\omega_k x)\sin(\omega_k y) 
\end{bmatrix}
$$

for $\omega_k = 2\pi k / p$, for our key frequencies $k$.

To calculate the logits, the network cancels out all directions apart from:

$$
\begin{aligned}
\cos(\omega_k (x+y)) &= \cos(\omega_k x)\cos(\omega_k y)-\sin(\omega_k x)\sin(\omega_k y) \\
\sin(\omega_k (x+y)) &= \sin(\omega_k x)\cos(\omega_k y)+\cos(\omega_k x)\sin(\omega_k y)
\end{aligned}
$$

The network then multiplies these by $\cos(wz),\sin(wz)$ and sums (i.e. the logit for value $z$ will be the product of these with $\cos(wz),\sin(wz)$, summed).

""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - can you explain why this algorithm works?
""")
        with st.expander("Hint"):
            st.markdown(r"""
Think about the following expression:

$$
\cos(w(x+y))\cos(wz)+\sin(w(x+y))\sin(wz)
$$
""")
        with st.expander("Answer"):
            st.markdown(r"""
The reason is again thanks to our trig formulas! Each neuron will add to the final logits a vector which looks like:

$$
\cos(w(x+y))\cos(wz)+\sin(w(x+y))\sin(wz)
$$

which we know from our trig formulas equals:

$$
\cos(w(x+y-z))
$$

This is large when $z = x+y$, and small if not.\*\* So the result is that the logits for $x+y$ will be large, and all others will be small. This is exactly what we want to happen, to solve our problem!

\*\**Technically this is an oversimplification; the value might be large for values of $z$ other than $x+y$. But remember that we have several different key frequencies $\omega_k$, and they combine destructively when we add together the neurons' outputs at the end wherever $z \neq x+y$.*
""")
    st.markdown(r"""
### 
    
To see that it cancels out other directions, we can transform the neuron activations and logits to the 2D Fourier Basis, and show the norm of the vector corresponding to each Fourier component - **we see that the quadratic terms have *much* higher norm in the logits than neuron activations, and linear terms are close to zero.** Remember that, to get from activations to logits, we apply the linear map $W_{logit}=W_{out}W_U$, which involves summing the outputs of all neurons.

Below is some code to visualise this. Note the use of `einops.reduce` rather than the `mean` method in the code below. Like most other uses of einops, this is useful because it's explicit and readable.

```python
imshow_fourier(
    einops.reduce(neuron_acts_centered_fourier.pow(2), 'y x neuron -> y x', 'mean'), 
    title='Norm of Fourier Components of Neuron Acts'
)

# Rearrange logits, so the first two dims represent (x, y) in modular arithmetic equation
original_logits = einops.rearrange(original_logits, "(x y) z -> x y z", x=p)
original_logits_fourier = fft2d(original_logits)

imshow_fourier(
    einops.reduce(original_logits_fourier.pow(2), 'y x z -> y x', 'mean'), 
    title='Norm of Fourier Components of Logits'
)
```

You should find that the linear and constant terms have more or less vanished relative to the quadratic terms, and that the quadratic terms are much larger in the logits than the neuron activations. Furthermore, the 2x2 squares in the logits plot (these are 2x2 squares, not large 1x1 squares!) show us that the model is indeed calculating this linear combination of quadratic terms.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - validate by only taking quadratic terms

Here, you will validate your results still further by *just* taking the components of the logits corresponding to $\cos(\omega_k(x+y))$ and $\sin(\omega_k(x+y))$ for each of our key frequencies $k$, and showing this increases performance.

First, you should write a function `get_trig_sum_directions`, which takes in a frequency `k` and returns the `(p, p)`-size vectors in 2D Fourier space corresponding to the directions:

$$
\begin{aligned}
\cos(\omega_k (x+y)) &= \cos(\omega_k x)\cos(\omega_k y)-\sin(\omega_k x)\sin(\omega_k y) \\
\sin(\omega_k (x+y)) &= \sin(\omega_k x)\cos(\omega_k y)+\cos(\omega_k x)\sin(\omega_k y)
\end{aligned}
$$

respectively.

```python
def get_trig_sum_directions(k: int) -> Tuple[TT[p, p], TT[p, p]]:
    '''
    Given frequency k, returns the normalized vectors in the 2D Fourier basis 
    representing the directions:

        cos(omega_k * (x + y))
        sin(omega_k * (x + y))

    respectively.
    '''

tests.test_get_trig_sum_directions(get_trig_sum_directions)
```

Once you've passed these tests, you can run the code to project the logits onto these directions, and see how the loss changes. Note the use of the `project_onto_direction` function which you wrote earlier.

```python
trig_logits = []

for k in key_freqs:

    cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(k)

    cos_xplusy_projection = project_onto_direction(
        original_logits,
        cos_xplusy_direction.flatten()
    )

    sin_xplusy_projection = project_onto_direction(
        original_logits,
        sin_xplusy_direction.flatten()
    )

    trig_logits.extend([cos_xplusy_projection, sin_xplusy_projection])

trig_logits = sum(trig_logits)

print('Loss with just x+y components:', test_logits(trig_logits, True, original_logits_sq).item())
print("Original Loss:", original_loss.item())
```
""")
    st.markdown(r"""
Okay, so we know that the model is mainly working with the terms $\cos(\omega_k(x+y))$ and $\sin(\omega_k(x+y))$ for each of our key frequencies $k$. But how can we show that it is indeed calculating the quantity:

$$
\cos(\omega_k (x + y))\cos(\omega_k z) + \sin(\omega_k (x + y))\sin(\omega_k z)
$$

?

The answer: **we examine $W_{logit}$ in the Fourier basis.** If we think that $W_{logit}$ is mainly projecting onto the directions $\cos(\omega_k z)$ and $\sin(\omega_k z)$, then we expect to find:

$$
W_{logit} \approx U S F = \sum_{i=1}^p \sigma_i u_i f_i^T
$$

where the singular values $\sigma_i$ are zero except those corresponding to Fourier basis vectors $f_i = \cos(\omega_k z), \sin(\omega_k z)$ for key frequencies $k$.

Thus, if we right-multiply $W_{logit}$ by $F^T$, we should get a matrix $US$ which has all rows zero except those corresponding to these Fourier basis vectors. Verify this for yourself:

```python
US: TT["neuron", "z"] = W_logit @ fourier_basis.T

imshow_div(
    US.T,
    aspect='auto', 
    y=fourier_basis_names, 
    xaxis='Neuron',
    title='W_logit in the Fourier Basis'
)
```
""")
    st.success(r"""
*Note the contrast between what we just did and what we've done before. In previous sections, we've taken the 2D Fourier transform of our activations / effective weights with respect to the input space. Now, we're taking the 1D Fourier tranformation with respect to the output space. It's mildly surprising that this worked!*""")
    st.markdown(r"""
We can further see the columns of $W_{logit}$ corresponding to each neuron cluster just gets terms corresponding to that frequency.

```python
for freq in key_freqs:
    imshow_div(
        US[neuron_freqs==freq].T,
        aspect='auto', 
        y=fourier_basis_names, 
        xaxis='Neurons in cluster', 
        title=f'W_logit in the Fourier Basis for neurons of freq {freq}'
    )
```

But we want to show more than just that this matrix is sparse. We want to show that, for instance, the input direction $u_i$ corresponding to the output direction $f_i = \cos(\omega_k z)$ is projecting the input onto the direction $\cos(\omega(x+y))$ (and equivalent for $\sin$). We'll do this now.

```python
cos_components = []
sin_components = []

for k in key_freqs:
    input_dir_cos = US[:, 2*k-1]
    input_dir_sin = US[:, 2*k]

    cos_components_in_std_basis = neuron_acts_post_sq @ input_dir_cos
    cos_components_in_fourier_basis = fft2d(cos_components_in_std_basis)
    cos_components.append(cos_components_in_fourier_basis)
    
    sin_components_in_std_basis = neuron_acts_post_sq @ input_dir_sin
    sin_components_in_fourier_basis = fft2d(sin_components_in_std_basis)
    sin_components.append(sin_components_in_fourier_basis)

for title, components in zip(['Cosine', 'Sine'], [cos_components, sin_components]):
    fig = imshow_fourier(
        t.stack(components),
        title=f'{title} components of neuron activations in Fourier basis',
        animation_frame=0,
        animation_name="Frequency",
        animation_labels=key_freqs.tolist()
    )
```

Let's just recap what this plot means, because it's a bit confusing. The matrix $US$ is sparse, because its main job is to project onto the directions $f_i = \cos(\omega_k z)$ and $\sin(\omega_k z)$ for our key frequencies. Furthermore, if we look at the input directions $\sigma_i u_i$ corresponding to these output directions $f_i$, we see that they are indeed projecting our input onto the directions:

$$
\begin{aligned}
\cos(\omega_k(x+y)) &= \cos(\omega_k x) \cos(\omega_k y) - \sin(\omega_k x) \sin(\omega_k y) \\
\sin(\omega_k(x+y)) &= \sin(\omega_k x) \cos(\omega_k y) + \cos(\omega_k x) \sin(\omega_k y)
\end{aligned}
$$

respectively. This is reflected in our plots, where each of the cosine plots basically has only the terms $\cos(\omega_k x) \cos(\omega_k y)$ and $\sin(\omega_k x) \sin(\omega_k y)$ non-zero (they are both large and of opposite sign), and the sine plots have only the terms $\sin(\omega_k x) \cos(\omega_k y)$ and $\cos(\omega_k x) \sin(\omega_k y)$ non-zero (they are both large and of the same sign).

An example:
""")
    st_image("cosine-components.png", 600)

def section_training():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#setup">Setup</a></li>
   <li><a class="contents-el" href="#excluded-loss">Excluded Loss</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#discussion
       ">Discussion</a></li>
   </ul></li>
   <li><a class="contents-el" href="#development-of-the-embedding">Development of the embedding</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#embedding-in-fourier-basis">Embedding in Fourier basis</a></li>
       <li><a class="contents-el" href="#svd-of-embedding">SVD of embedding</a></li>
   </ul></li>
   <li><a class="contents-el" href="#development-of-computing-trig-components">Development of computing trig components</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#discussion">Discussion</a></li>
   </ul></li>
   <li><a class="contents-el" href="#development-of-neuron-activations">Development of neuron activations</a></li>
   <li><a class="contents-el" href="#development-of-commutativity">Development of commutativity</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Analysis During Training

In this section, we analyse the modular addition transformer during training. In the data we'll be using, checkpoints were taken every 100 epochs, from epoch 0 to 50K (the model we used in previous exercises was taken at 40K).

Usability note: I often use animations in this section. I recommend using the slider manually, not pressing play - Plotly smooths animations in a confusing and misleading way (and I haven't figured out how to fix it).

Usability note 2: To get plots to display, they can't have too many data points, so often plots will have different epoch intervals between data points, or different final epochs

Notation: I use "trig components" to refer to the components $\cos(\omega(x+y))$, $\sin(\omega(x+y))$.
""")
    st.info(r"""
### Overview

* The model starts to learn the generalised algorithm well before the phase change, and this develops fairly smoothly
    * We can see this with the metric of excluded loss, which seems to indicate that the model learns to "memorise more efficiently" by using the $\cos(w(x+y))$ directions.
    * This is a clear disproof of the 'grokking is a random walk in the loss landscape that eventually gets lucky' hypothesis.
* We also examine more qualitatively each circuit in the model and how it develops
    * We see that all circuits somewhat develop pre-grokking, but at different rates and some have a more pronounced phase change than others
    * We examine the embedding circuit, the 'calculating trig dimensions' circuit and the development of commutativity.
    * We also explore the development of neuron activations and how it varies by cluster
* There's a small but noticeable lag between 'the model learns the generalisable algorithm' and 'the model cleans up all memorised noise'.
* There are indications of several smaller phase changes, beyond the main grokking one.
    * In particular, a phase change at 43K-44K, well after grokking (I have not yet interpreted what's going on here).
""")
    st.markdown(r"""
## Setup

First, we'll define some useful functions. In particular, the `get_metrics` function is designed to return a dictionary of metrics over the training period. The argument `metric_fn` is itself a function which takes in a model, and returns a metric (e.g. we use `metric_fn=test_loss`, to return the model's loss on the test set).

```python
    epochs = full_run_data['epochs']

    metric_cache = {}

    def get_metrics(model: HookedTransformer, metric_cache, metric_fn, name, reset=True):
        if reset or (name not in metric_cache) or (len(metric_cache[name])==0):
            metric_cache[name]=[]
            for c, sd in enumerate(tqdm((full_run_data['state_dicts']))):
                model = load_in_state_dict(model, sd)
                out = metric_fn(model)
                if type(out)==t.Tensor:
                    out = utils.to_numpy(out)
                metric_cache[name].append(out)
            model = load_in_state_dict(model, full_run_data['state_dicts'][400])
            try:
                metric_cache[name] = t.tensor(metric_cache[name])
            except:
                metric_cache[name] = t.tensor(np.array(metric_cache[name]))
    plot_metric = partial(lines, x=epochs, xaxis='Epoch', log_y=True)

    def test_loss(model):
        logits = model(all_data)[:, -1, :-1]
        return test_logits(logits, False, mode='test')
    get_metrics(model, metric_cache, test_loss, 'test_loss')

    def train_loss(model):
        logits = model(all_data)[:, -1, :-1]
        return test_logits(logits, False, mode='train')
    get_metrics(model, metric_cache, train_loss, 'train_loss')
```

## Excluded Loss

**Excluded Loss** for frequency $w$ is the loss on the training set where we delete the components of the logits corresponding to $\cos(w(x+y))$ and $sin(w(x+y))$. We get a separate metric for each $w$ in the key frequencies.

**Key observation:** The excluded loss (especially for frequency 14) starts to go up *well* before the point of grokking.

(Note: this performance decrease is way more than you'd get for deleting a random direction.)
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement excluded loss

You should fill in the function below to implement excluded loss. You'll need to use the `get_trig_sum_directions` and `project_onto_direction` functions from previous exercises.

Note - when calculating the loss, you should use the `test_logits` function. Don't perform bias correction, and use `mode='train'`.

```python
def excl_loss(model: HookedTransformer, key_freqs: list) -> float:
    '''
    Returns the excluded loss (i.e. subtracting the components of logits
    corresponding to cos(w_k(x+y)) and sin(w_k(x+y)), for each frequency
    k in key_freqs.
    '''
    pass
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def excl_loss(model: HookedTransformer, key_freqs: list) -> float:
    '''
    Returns the excluded loss (i.e. subtracting the components of logits
    corresponding to cos(w_k(x+y)) and sin(w_k(x+y)), for each frequency
    k in key_freqs.
    '''

    excl_loss_list = []
    logits = model(all_data)[:, -1, :-1]

    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)

        logits_cos_xplusy = project_onto_direction(
            logits,
            cos_xplusy_direction.flatten()
        )

        logits_sin_xplusy = project_onto_direction(
            logits,
            sin_xplusy_direction.flatten()
        )

        logits_excl = logits - logits_cos_xplusy - logits_sin_xplusy

        loss = test_logits(logits_excl, bias_correction=False, mode='train').item()

        excl_loss_list.append(loss)

    return excl_loss_list
```
""")
        st.markdown(r"""
Once you've completed this function, you can run the following code to plot the excluded loss for each of the key frequencies (as well as the training and testing loss as a baseline).

```python
excl_loss = partial(excl_loss, key_freqs=key_freqs)
get_metrics(model, metric_cache, excl_loss, 'excl_loss')

lines(
    t.concat([
        metric_cache['excl_loss'].T, 
        metric_cache['train_loss'][None, :],  
        metric_cache['test_loss'][None, :]
    ], axis=0), 
    labels=[f'excl {freq}' for freq in key_freqs]+['train', 'test'], 
    title='Excluded Loss for each trig component',
    log_y=True,
    x=full_run_data['epochs'],
    xaxis='Epoch',
    yaxis='Loss'
)
""")
    st.markdown(r"""
### Discussion

I find excluded loss particularly exciting because it's a clear metric that shows the develop of a capability well before the capability can be clearly seen from the model's output on unseen data. (We can see curves like `excl 14` increase above the baseline training loss curve quite early on, showing that the model is learning an algorithm which makes heavy use of this frequency.) This kind of metric seems a proof of concept of interpretability as a way to notice the development of capabilities before they happen.

It also seems weak evidence for the possibility of interpretability-inspired metrics that could be included in the loss function while training to influence the model's capabilities, though excluded loss is a fairly weird metric for this, as calculating it involves running the model on the training *and* test data.

## Development of the embedding

### Embedding in Fourier basis

We can plot the norms of the embedding of each 1D Fourier component at each epoch. Pre-grokking, the model is learning the representation of prioritising a few components, but most components still have non-trivial value, presumably because these directions are doing some work in memorising. Then, during the grokking period, the other components get set to near zero - the model no longer needs other directions to memorise things, it's learned a general algorithm.

(As a reminder, we found that the embedding was well-approximated by $W_E \approx F^T S V^T$ where $F$ is the Fourier basis vector, hence $F W_E \approx S V^T$ is a sparse matrix, with most rows equal to zero. So when we calculate the norm of the rows of $F W_E$ for intermediate points in training, we're seeing how the input space of the embedding learns to only contain these select few frequencies.)

```python
animate_lines(
    metric_cache['fourier_embed'][::2],
    snapshot_index = epochs[::2],
    snapshot='Epoch',
    hover=fourier_basis_names,
    animation_group='x',
    title='Norm of Fourier Components in the Embedding Over Training'
)
```

### SVD of embedding

We've discussed SVD, but now let's be more explicit about it and actually calculate the SVD!
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - Examine the SVD of $W_E$

Fill in the following function, which returns the singular values from the singular value decomposition of $W_E$ at an intermediate point in training. (Remember to remove the last row of $W_E$, which corresponds to the bias term.) PyTorch has an SVD function (`torch.svd`) which you should use for this.

```python
def embed_SVD(model):
    '''
    Returns vector S, where W_E = U @ diag(S) @ V.T in singular value decomp.
    '''
    pass
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def embed_SVD(model):
    '''
    Returns vector S, where W_E = U @ diag(S) @ V.T in singular value decomp.
    '''
    _, S, _ = t.svd(model.embed.W_E[:, :-1])
    return S
```
""")
        st.markdown(r"""
Can you interpret what's going on in this plot?
""")
        with st.expander("Interpretation"):
            st.markdown(r"""
At first, our SVD values are essentially random. Throughout training, the smaller singular values tend to zero, while the singular values corresponding to the key frequencies increase. Eventually, the graph demonstrates a sparse matrix, with all singular values zero except for those corresponding to the key frequencies.
""")
            
    st.markdown(r"""
## Development of computing trig components

In previous exercises, we've projected our logits or neuron activations onto 2D Fourier basis directions corresponding to $\cos(\omega_k(x+y))$ and $\sin(\omega_k(x+y))$ for each of the key frequencies $k$. We found that these directions explained basically all of the model's performance.

Here, we'll do the same thing, but over time, to see how the model learns to compute these trig components. We'll use the same functions you wrote in previous exercises, but all the code is provided to you.

The activations are first centered, then their sum of squares is taken, and then the Fourier components are extracted and we see what fraction of variance they explain. This is then averaged across the "output dimension" (which is neurons in the case of the neuron activations, or output classes in the case of the logits).

```python
def tensor_trig_ratio(model, mode):
    '''
    Returns the fraction of variance of the (centered) activations which is explained by the Fourier
    directions corresponding to cos(w(x+y)) and sin(w(x+y)) for all the key frequencies.
    '''
    logits, cache = model.run_with_cache(all_data)
    logits = model(all_data)[:, -1, :-1]
    if mode=='neuron_pre':
        tensor = cache['pre', 0][:, -1]
    elif mode=='neuron_post':
        tensor = cache['post', 0][:, -1]
    elif mode=='logit':
        tensor = logits
    else:
        raise ValueError(f"{mode} is not a valid mode")
    tensor_centered = tensor - einops.reduce(tensor, 'xy index -> 1 index', 'mean')
    tensor_var = einops.reduce(tensor_centered.pow(2), 'xy index -> index', 'sum')

    tensor_trig_vars = []
    
    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)
        cos_xplusy_projection_var = project_onto_direction(tensor_centered, cos_xplusy_direction.flatten()).pow(2).sum(0)
        sin_xplusy_projection_var = project_onto_direction(tensor_centered, sin_xplusy_direction.flatten()).pow(2).sum(0)
    
        tensor_trig_vars.extend([cos_xplusy_projection_var, sin_xplusy_projection_var])

    return utils.to_numpy(sum(tensor_trig_vars)/tensor_var)

    
for mode in ['neuron_pre', 'neuron_post', 'logit']:
    get_metrics(
        model, 
        metric_cache, 
        partial(tensor_trig_ratio, mode=mode), 
        f"{mode}_trig_ratio", 
        reset=True
    )

lines_list = []
line_labels = []
for mode in ['neuron_pre', 'neuron_post', 'logit']:
    tensor = metric_cache[f"{mode}_trig_ratio"]
    lines_list.append(einops.reduce(tensor, 'epoch index->epoch', 'mean'))
    line_labels.append(f"{mode}_trig_frac")
plot_metric(
    lines_list, 
    labels=line_labels, 
    log_y=False,
    yaxis='Ratio',
    title='Fraction of logits and neurons explained by trig terms',
)
```

By plotting on a log scale, we can more clearly see that all 3 are having a higher proportion of trig components over training, but that the logits are smoother while the neurons exhibit more of a phase change.

### Discussion

In the fully trained model, there are two key components to the algorithm that results in the model being able to meaningfully use trig directions in the logits - firstly that the neuron activations have significant quadratic terms, and secondly that $W_{logit}$ can cancel out all of the non-trig terms, and then map the trig terms to $(x+y)\% p$.

A natural question is whether one of these comes first, or if both evolve in tandem - as far as I'm aware, "how do circuits with multiple moving parts form over training" is not at all understood.

In this case, the logits develop the capability to cancel out everything but the trig directions early on, and the neurons don't develop significant quadratic or trig components until close to the grokking point.

I vaguely speculate that it makes more sense for circuits to develop in "reverse-order" - if we need two layers working together to produce a nice output, then if the second layer is randomly initialised the first layer can do nothing. But if the first layer is randomly initialised, the second layer can learn to just extract the components of the output corresponding to the "correct" output, and use them to badly approximate the output solution. And *now* the network has a training incentive to build up both parts of the circuit.

(This maybe has something to do with the Lottery Ticket hypothesis?)

## Development of neuron activations

Coming soon!

## Development of commutativity

Coming soon!
""")
    
def section_discussion():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#why-phase-changes">Why Phase Changes?</a></li>
   <li><a class="contents-el" href="#limitations">Limitations</a></li>
   <li><a class="contents-el" href="#relevance-to-alignment">Relevance to Alignment</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#model-training-dynamics">Model Training Dynamics</a></li>
       <li><a class="contents-el" href="#other-relevance">Other Relevance</a></li>
   </ul></li>
   <li><a class="contents-el" href="#training-dynamics">Training Dynamics</a></li>
   <li><a class="contents-el" href="#future-directions">Future Directions</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Discussion & Future Directions

The key claim I want to make is that grokking happens when the process of learning a general algorithm exhibits a phase change, *and* the model is given the minimal amount of data such that the phase change still happens. 

## Why Phase Changes?

I further speculate that phase changes happen when a model learns *any* single generalising circuit that requires different parts of the model to work together in a non-linear way. Learning a sophisticated generalising circuit is hard and requires different parts of the model to line up correctly (ie coordinate on a shared representation). It doesn't seem super surprising if the circuit is in some sense either working or not, rather than smoothly increasing in performance

The natural next question is why the model learns the general algorithm at all - if performance is constant pre-phase change, there should be no gradient. Here I think the key is that it is hard to generalise to *unseen data*. The circuits pre-phase change are still fairly helpful for predicting seen data, as can be seen from [excluded loss](#scrollTo=Excluded_Loss). (though, also, test loss tends to decrease even pre-phase change, it's just slower - the question is of how large the gradient needs to be, not whether it's zero or not)

My current model is that every gradient update of the model is partially an incentive to memorise the data in the batch and partially to pick up patterns within the data. The gradient updates to memorise point in arbitrary directions and tend to cancel out (both within a batch and between batches) if the model doesn't have enough capacity to memorise fully, while the gradient updates to identify patterns reinforce each other. Regularisation is something that dampens the memorising gradients over the pattern recognition gradients. We don't observe this dynamic in the infinite data setting because we never re-run the model on past data points, but I expect it has at least slightly memorised previous training data. (See, eg, [Does GPT-2 Know Your Phone Number?](https://bair.berkeley.edu/blog/2020/12/20/lmmem/))

I speculate that this is because the model is incentivised to memorise the data as efficiently as possible, ie is biased towards simplicity. This comes both due to both explicit regularisation such as weight decay and implicit regularisation like inherent model capacity and SGD. The model learns to pick up patterns between data points, because these regularities allow it to memorise more efficiently. But this needs enough training data to make the memorisation circuit more complex than the generalising circuit(s). 

This further requires there to be a few crisp circuits for the model to learn - if it fuzzily learns many different circuits, then lowering the amount of data will fairly smoothly decrease test performance, as it learns fewer and fewer circuits.

## Limitations

There are several limitations to this work and how confidently I can generalise from it to a general understanding of deep learning/grokking. In particular:
* The modular addition transformer is a toy model, which only needs a few circuits to solve the problem fully (unlike eg an LLM or an image model, which needs many circuits)
* The model is over-parametrised, as is shown by eg it learning many redundant neurons doing roughly the same task. It has many more parameters than it needs for near-perfect performance on its task, unlike real models.
* I only study weight decay as the form of regularisation
* My explanations of what's going on rely heavily on fairly fuzzy and intuitive notions that I have not explicitly defined: crisp circuits, simplicity, phase changes, what it means to memorise vs generalise, model capacity, etc.

## Relevance to Alignment

### Model Training Dynamics

The main thing which seems relevant from an alignment point of view is a better understanding of model training dynamics.

In particular, phase changes seem pretty bad from an alignment perspective, because they suggest that rapid capability gain as models train or are scaled up is likely. This makes it less likely that we get warning shots (ie systems exhibiting non-sophisticated unaligned behaviour) before we get dangerous unaligned systems, and makes it less likely that systems near to AGI are good empirical test beds for alignment work. This work exhibits a bunch more examples of phase changes, and marginally updates me towards alignment being hard.

This work makes me slightly more hopeful, since it shows that phase changes are clearly foreshadowed by the model learning the circuits beforehand, and that we can predict capability gain with interpretability inspired metrics (though obviously this is only exhibited in a toy case, and only after I understood the circuit well - I'd rather not have to train and interpret an unaligned AGI before we can identify misalignment!). 

More speculatively, it suggests that we may be able to use interpretability tools to shape training dynamics. A natural concern here is that a model may learn to obfuscate our tools, and perform the same algorithms but without them showing up on our tools. This could happen both from gradient descent directly learning to obfuscate things (because the unaligned solution performs much better than an aligned solution), or because the system itself has [learned to be deceptive and alter itself to avoid our tools](https://www.lesswrong.com/posts/nbq2bWLcYmSGup9aF/a-transparency-and-interpretability-tech-tree). The fact that we observe the circuits developing well before they are able to generalise suggests that we might be able to disincentivise deception before the model gets good enough at deception to be able to generalise and evade deveption detectors (though doesn't do much for evading gradient descent).

The natural future direction here is to explore training on interpretability inspired metrics, and to see how much gradient descent learns to Goodhart them vs shifting its inductive bias to learn a different algorithm. Eg, can we incentive generalisation and get grokking with less data? Can we incentivise memorisation and change the algorithm it learns? What happens if we disincentivise learning to add with certain frequencies?

### Other Relevance

This also seems relevant as more evidence of the [circuits hypothesis](https://distill.pub/2020/circuits/zoom-in/), that networks *can* be mechanistically understood and are learning interpretable algorithms. And also as one of the first examples of interpreting a transformer's neurons (though on a wildly different task to language).

It also seems relevant as furthering the science of deep learning by better understanding how models generalise, and the weird phenomena of grokking. Understanding whether a model will be aligned or not requires us to understand how it will generalise, and to predict future model behaviour, and what different training setups will and will not generalise. This motivation is somewhat more diffuse and less directed, but seems analogous to how statistical learning theory allows us to predict things about models in classical statistics (though is clearly insufficient for deep learning). 

(Though honestly I mostly worked on this because it was fun, I was on holiday, and I got pretty nerd-sniped by the problem. So all of this is somewhat ad-hoc and backwards looking, rather than this being purely alignment motivated work)

## Training Dynamics

One interesting thing this research gives us insight into is the training dynamics of circuits - which parts of the circuit develop first, at what rates, and why? 

My speculative guess is that, when a circuit has several components interacting between layers, the component closest to the logits is easier to form and will tend to form first. 

**Intuition:** Imagine a circuit involving two components interacting with a non-linearity in the middle, which are both randomly initialised. They want to converge on a shared representation, involving a few directions in the hidden space. Initially, the random first component will likely have some output corresponding to the correct features, but with small coefficients and among a lot of noise. The second component can learn to focus on these correct features and cancel out the noise, reinforcing the incentive for the first component to focus on these features. On the other hand, if the second component is random, it's difficult to see how the first component can produce reasonable features that are productively used by the second component.

Qualitatively, we observe that all circuits are forming in parallel pre-grokking, but it roughly seems that the order is logit circuit > embedding circuit > neuron circuit > attention circuit (ie logit is fastest, attention is slowest)

This seems like an interesting direction of future research by giving a concrete example of crisp circuits that are easy to study during training. Possible initial experiments would be fixing parts of the network to be random, initialising parts of the network to be fully trained and frozen, or giving different parts of the network different learning rates.

## Future Directions

Some thoughts on future directions I'd be excited to see - if you're interested in working on any of these, I'd love to chat!
* Modular addition
    * Interpreting the memorisation circuit, and figuring out *how* models memorise
    * Training on interpretability inspired metrics
        * Note that excluded loss is a somewhat dodgy metric to train on, as it involves computation over both the train and test data
* Interpreting the five digit addition or predicting repeated subsequencies examples
    * In particular, trying to map the many phase changes in 5 digit addition to circuits
* Looking for other examples of phase changes
    * Toy problems
        * Something incentivising skip trigrams
        * Something incentivising virtual attention heads
    * Looking for [curve detectors](https://distill.pub/2020/circuits/curve-circuits) in a ConvNet
        * A dumb way to try this would be to train a model to imitate the actual curve detectors in Inception (eg minimising OLS loss between the model's output and curve detector activations)
    * Looking at the formation of interpretable neurons in a [SoLU transformer](https://transformer-circuits.pub/2022/solu/index.html)
    * Looking inside a LLM with many checkpoints
        * Eleuther have many checkpoints of GPT-J and GPT-Neo, and will share if you ask
        * [Mistral](https://nlp.stanford.edu/mistral/getting_started/download.html) have public versions of GPT-2 small and medium, with 5 runs and many checkpoints
        * Possible capabilities to look for
            * Performance on benchmarks, or specific questions from benchmarks
            * Simple algorithmic tasks like addition, or sorting words into alphabetical order, or matching open and close brackets
            * Soft induction heads, eg [translation](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#performing-translation)
            * Look at attention heads on various text and see if any have recognisable attention patterns (eg start of word, adjective describing current word, syntactic features of code like indents or variable definitions, most recent open bracket, etc).
""")


func_page_list = [
    (section_home, "🏠 Home"), 
    (section_intro, "1️⃣ Periodicity & Fourier Basis"), 
    (section_circuits, "2️⃣ Circuit & Feature Analysis"), 
    (section_training, "3️⃣ Analysis During Training"), 
    (section_discussion, "4️⃣ Discussion & Future Directions"), 
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