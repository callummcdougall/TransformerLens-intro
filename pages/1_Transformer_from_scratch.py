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
    <li><a class="contents-el" href="#setup">Setup</a></li>
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
Links to Colab: [**exercises**](https://colab.research.google.com/drive/1LpDxWwL2Fx0xq3lLgDQvHKM5tnqRFeRM?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1ND38oNmvI702tu32M74G26v-mO5lkByM?usp=share_link)

# Introduction

This is a clean, first principles implementation of GPT-2 in PyTorch. This is an accompaniment to [my video tutorial on implementing GPT-2](https://neelnanda.io/transformer-tutorial-2). If you want to properly understand how to implement GPT-2, you'll need to do it yourself! There's a [template version of this notebook here](https://neelnanda.io/transformer-template), go and fill in the blanks (no copying and pasting!) and see if you can pass the tests. **I recommend filling out the template *as* you watch the video, and seeing how far you can get with each section before watching me do it**.

If you enjoyed this, I expect you'd enjoy learning more about what's actually going on inside these models and how to reverse engineer them! This is a fascinating young research field, with a lot of low-hanging fruit and open problems! **I recommend starting with my post [Concrete Steps for Getting Started in Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started).**

This notebook was written to accompany my [TransformerLens library](https://github.com/neelnanda-io/TransformerLens) for doing mechanistic interpretability research on GPT-2 style language models, and is a clean implementation of the underlying transformer architecture in the library.

Further Resources:
* [A Comprehensive Mechanistic Interpretability Explainer & Glossary](https://www.neelnanda.io/glossary) - an overview
    * Expecially [the transformers section](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=pndoEIqJ6GPvC1yENQkEfZYR)
* My [200 Concrete Open Problems in Mechanistic Interpretability](https://www.neelnanda.io/concrete-open-problems) sequence - a map
* My walkthrough of [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), for a deeper dive into how to think about transformers:.

Check out these other intros to transformers for another perspective:
* Jay Alammar's [illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
* [Andrej Karpathy's MinGPT](https://github.com/karpathy/minGPT)

**Sharing Guidelines:** This tutorial is still a bit of a work in progress! I think it's usable, but please don't post it anywhere publicly without checking with me first! Sharing with friends is fine. 

If you've found this useful, I'd love to hear about it! Positive and negative feedback also very welcome. You can reach me via [email](mailto:neelnanda27@gmail.com)

## Instructions

You have two options for how to go through these exercises:

1. Use this streamlit page, and your own choice of IDE (e.g. VSCode). The code you'll need to run can all be copied from the page, and the solutions will be available in dropdowns.
2. Use the [template Google Colab](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo_Template.ipynb#scrollTo=JZ-9yQIulqJr), and compare your solutions to the [filled-in Colab](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=bEYvLhXVAfRP).

In either case, you will want to follow along with Neel's [video walkthrough](https://www.youtube.com/watch?v=VMvpQhNkm8w&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz&index=1). The advantage of using Colab is that you'll have to spend less time setting up your environment, and it may be easier to follow along with his video. The advantages of using your own IDE are that you'll have more control over your environment, and future exercises may be easier to set up (since not all future exercises have a corresponding notebook yet).

As you go through the material, there will be exercises for you to try, and tests you can run to verify your solutions are correct. You get bonus points if you can do the exercises without looking at the solutions, and before I do it in the video!

## Learning objectives

Here are the learning objectives for each section of the tutorial. At the end of each section, you should refer back here to check that you've understood everything.
""")

    st.info(r"""
## 1️⃣ Understanding Inputs & Outputs of a Transformer

* Understand what a transformer is used for
* Understand causal attention, and what a transformer's output represents
* Learn what tokenization is, and how models do it
* Understand what logits are, and how to use them to derive a probability distribution over the vocabulary
""")
    st.info(r"""
## 2️⃣ Clean Transformer Implementation

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
    st.info(r"""
## 3️⃣ Training a model

* Use the `Adam` optimizer to train your transformer
* Run a training loop on a very small dataset, and verify that your model's loss is going down
""")

#     st.markdown(r"""
# ## Setup

# If you're using the Colab rather than the Streamlit page, then you can follow the instructions in the Colab (from the "Instructions" section onwards). If you're using your own IDE, then you'll need to install the following packages:

# ```python
# %pip install git+https://github.com/neelnanda-io/TransformerLens.git@new-demo
# %pip install git+https://github.com/neelnanda-io/PySvelte.git
# %pip install fancy_einsum
# %pip install einops
# ```

# and then run the following:

# ```python
# import einops
# from fancy_einsum import einsum
# from dataclasses import dataclass
# import torch
# import torch.nn as nn
# import numpy as np
# import math
# from transformer_lens import utils, HookedTransformer
# import tqdm.auto as tqdm
# ```

# """)
    
def section_intro():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#what-is-the-point-of-a-transformer">What is the point of a transformer?</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#how-is-the-model-trained">How is the model trained?</a></li>
       <li><a class="contents-el" href="#key-takeaway">Key takeaway:</a></li>
   </ul></li>
   <li><a class="contents-el" href="#tokens-transformer-inputs">Tokens - Transformer Inputs</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#how-do-we-convert-language-to-vectors">How do we convert language to vectors?</a></li>
       <li><a class="contents-el" href="#tokens-language-to-sequence-of-integers">Tokens: Language to sequence of integers</a></li>
       <li><a class="contents-el" href="#rant-tokenization-is-a-headache">Rant: Tokenization is a Headache</a></li>
       <li><a class="contents-el" href="#key-takeaway">Key Takeaway:</a></li>
   </ul></li>
   <li><a class="contents-el" href="#logits-transformer-outputs">Logits - Transformer Outputs</a></li>
   <li><a class="contents-el" href="#generation">Generation!</a></li>
   <li><a class="contents-el" href="#key-takeaways">Key takeaways:</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Understanding Inputs & Outputs of a Transformer
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
import torch
import torch.nn as nn
import numpy as np
import math
from transformer_lens import EasyTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm

reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
```

## What is the point of a transformer?

**Transformers exist to model text!**

We're going to focus GPT-2 style transformers. Key feature: They generate text! You feed in language, and the model generates a probability distribution over tokens. And you can repeatedly sample from this to generate text! 

(To explain this in more detail - you feed in a sequence of length $N$, then sample from the probability distribution over the $N+1$-th word, use this to construct a new sequence of length $N+1$, then feed this new sequence into the model to get a probability distribution over the $N+2$-th word, and so on.)

### How is the model trained?

You give it a bunch of text, and train it to predict the next token.

Importantly, if you give a model 100 tokens in a sequence, it predicts the next token for *each* prefix, ie it produces 100 predictions. This is kinda weird but it's much easier to make one that does this. And it also makes training more efficient, because you can 100 bits of feedback rather than just one.

#### Objection: Isn't this trivial for all except the last prediction, since the transformer can just "look at the next one"?

No! We make the transformer have *causal attention*. The core thing is that it can only move information forwards in the sequence. The prediction of what comes after token 50 is only a function of the first 50 tokens, *not* of token 51. We say the transformer is **autoregressive**, because it only predicts future words based on past data.""")

    st_image("transformer-overview.png", 1000)
    st.markdown("")
    st.markdown(r"""

### Key takeaway:

Transformers are *sequence modelling engines*. They the same processing in parallel at each sequence position, can move information between positions with attention, and conceptually can take a sequence of arbitrary length (not actually true, see later)

## Tokens - Transformer Inputs

Core point: Input is language (ie a sequence of characters, strings, etc)

### How do we convert language to vectors?

ML models take in vectors, not weird stuff like language. How do we convert between the two?

#### Idea: integers to vectors

We basically make a massive lookup table, which is called an **embedding**. It has one vector for each word in our vocabulary. We label every word in our vocabulary with an integer (this labelling never changes), and we use this integer to index into the embedding.

This is usually represented as a matrix $W_E$ (e.g. with shape `[vocab_size, d_model]`). We can represent words with a **one-hot encoding**, i.e. a vector of length `vocab_size` with zeros everywhere except for a 1 in the position corresponding to the word. This means that if $v$ is a one-hot encoding of a word, then $v^T W_E$ is the embedding of that word (because this is just the v-th row of $W_E$). This framing is exactly equivalent to thinking of $W_E$ as a lookup table.

A key intuition is that one-hot encodings let you think about each integer independently. We don't bake in any relation between words when we perform our embedding, because every word has a completely separate embedding vector.

### Tokens: Language to sequence of integers

Core idea: We need a model that can deal with arbitrary text. We want to convert this into integers, *and* we want these integers to be in a bounded range. 

* **Idea:** Form a vocabulary!
    * **Idea 1:** Get a dictionary! 
        * **Problem:** It can't cope with arbitrary text (e.g. URLs, punctuation, etc), also can't cope with mispellings.
    * **Idea 2:** Vocab = 256 ASCII characters. Fixed vocab size, can do arbitrary text, etc.
        * **Problem:** Loses structure of language - some sequences of characters are more meaningful than others
            * e.g. "language" is a lot more meaningful than "hjksdfiu" - we want the first to be a single token, second to not be. It's a more efficient use of our vocab.

#### What Actually Happens?

The most common strategy is called **Byte-Pair encodings**.

We begin with the 256 ASCII characters as our tokens, and then find the most common pair of tokens, and merge that into a new token. Note that we do have a space character as one of our 256 tokens, and merges using space are very common. For instance, here are the five first merges for the tokenizer used by GPT-2:

```
" t"
" a"
"he"
"in"
"re"
```
""")
    with st.expander("Fun (totally optional) exercise - can you guess what the first-formed 3/4/5/6/7-letter encodings in GPT-2's vocabulary are?"):
        st.markdown(r"""
They are:

```
3 -> "ing"
4 -> " and"
5 -> " that"
6 -> " their"
7 -> " people"
```
""")

    st.markdown(r"""
Note - you might see the character `Ġ` in front of some tokens. This is a special token that indicates that the token begins with a space. Tokens with a leading space vs not are different.

You can run the code below to see some more of GPT-2's tokenizer's vocabulary:

```python
sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n:n[1])
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
```

As you get to the end of the vocabulary, you'll be producing some pretty weird-looking esoteric tokens (because you'll already have exhausted all of the short frequently-occurring ones):

```python
sorted_vocab[-20:]
```

Transformers in the `transformer_lens` library have a `to_tokens` method that converts text to numbers. It also prepends them with a special token called `bos` (beginning of sequence) to indicate the start of a sequence (we'll learn more about this later). You can disable this with the `prepend_bos` argument.

Prepends with a special token to give attention a resting position, disable with `prepend_bos=False`

### Some tokenization annoyances

There are a few funky and frustrating things about tokenization, which causes it to behave differently than you might expect. For instance:

##### Whether a word begins with a capital or space matters!

```python
print(reference_gpt2.to_str_tokens("Ralph"))
print(reference_gpt2.to_str_tokens(" Ralph"))
print(reference_gpt2.to_str_tokens(" ralph"))
print(reference_gpt2.to_str_tokens("ralph"))
```

##### Arithmetic is a mess.

Length is inconsistent, common numbers bundle together.

```python
reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000")
```
""")

    st.success(r"""
### Key Takeaways

* We learn a dictionary of vocab of tokens (sub-words).
* We (approx) losslessly convert language to integers via tokenizing it.
* We convert integers to vectors via a lookup table.
* Note: input to the transformer is a sequence of *tokens* (ie integers), not vectors
""")
    st.markdown(r"""
## Logits - Transformer Outputs

**Goal:** Probability distribution over next tokens. (For every *prefix* of the sequence - given n tokens, we make n next token predictions)

**Problem:** How to convert a vector (where some values may be more than one, or negative) to a probability distribution? 

**Answer:** Use a softmax ($x_i \to \frac{e^{x_i}}{\sum e^{x_j}}$). Exponential makes everything positive, normalization makes it add to one.

So the model outputs a tensor of logits, one vector of size $d_{vocab}$ for each input token.

(Note - we call something a logit if it represents a probability distribution, and it is related to the actual probabilities via the softmax function. Logits and probabilities are both equally valid ways to represent a distribution.)

## Text generation

#### **Step 1:** Convert text to tokens

The sequence gets tokenized, so it has shape `[batch, seq_len]`. Here, the batch dimension is just one (because we only have one sequence).

```python
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and"
tokens = reference_gpt2.to_tokens(reference_text)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))
```

#### **Step 2:** Map tokens to logits

(`run_with_cache` tells the model to cache all intermediate activations. This isn't important right now; we'll look at it in more detail later.)

From our input of shape `[batch, seq_len]`, we get output of shape `[batch, seq_len, vocab_size]`. The `[i, j, :]`-th element of our output is a vector of logits representing our prediction for the `j+1`-th token in the `i`-th sequence.

```python
tokens = tokens.cuda()
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)
```

#### **Step 3:** Convert the logits to a distribution with a softmax

This doesn't change the shape, it is still `[batch, seq_len, vocab_size]`.

```python
log_probs = logits.log_softmax(dim=-1)
probs = logits.log_softmax(dim=-1)
print(log_probs.shape)
print(probs.shape)
```

#### **Bonus step:** What is the most likely next token at each position?

```python
list(zip(reference_gpt2.to_str_tokens(reference_text), reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])))
```

#### **Step 4:** Map distribution to a token

```python
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))
```

### **Step 5:** Add this to the end of the input, re-run

There are more efficient ways to do this (e.g. where we cache some of the values each time we run our input, so we don't have to do as much calculation each time we generate a new value), but this doesn't matter conceptually right now.

```python
next_tokens = torch.cat([tokens, torch.tensor(next_token, device='cuda', dtype=torch.int64)[None, None]], dim=-1)
new_logits = reference_gpt2(next_tokens)
print("New Input:", next_tokens)
print(next_tokens.shape)
print("New Input:", reference_gpt2.tokenizer.decode(next_tokens[0]))

print(new_logits.shape)
print(new_logits[-1, -1].argmax(-1))

print(reference_gpt2.tokenizer.decode(new_logits[-1, -1].argmax(-1)))
```
""")

    st.success(r"""
## Key takeaways:

* Takes in language, predicts next token (for *each* token in a causal way)
* We convert language to a sequence of integers with a tokenizer.
* We convert integers to vectors with a lookup table.
* Output is a vector of logits (one for each input token), we convert to a probability distn with a softmax, and can then convert this to a token (eg taking the largest logit, or sampling).
* We append this to the input + run again to generate more text (Jargon: *autoregressive*)
* Meta level point: Transformers are sequence operation models, they take in a sequence, do processing in parallel at each position, and use attention to move information between positions!
""")

def section_code():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#high-level-architecture">High-Level architecture</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#bonus-things-less-conceptually-important-but-key-technical-details">Bonus things - less conceptually important but key technical details</a></li>
   </ul></li>
   <li><a class="contents-el" href="#actual-code">Actual Code!</a></li>
   <li><ul class="contents">
    <li><a class="contents-el" href="#parameters-and-activations">Parameters and Activations</a></li>
    <li><a class="contents-el" href="#config">Config</a></li>
    <li><a class="contents-el" href="#tests">Tests</a></li>
    <li><a class="contents-el" href="#layernorm">LayerNorm</a></li>
    <li><a class="contents-el" href="#embedding">Embedding</a></li>
    <li><a class="contents-el" href="#positional-embedding">Positional Embedding</a></li>
    <li><a class="contents-el" href="#attention">Attention</a></li>
    <li><a class="contents-el" href="#mlp">MLP</a></li>
    <li><a class="contents-el" href="#transformer-block">Transformer Block</a></li>
    <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
    <li><a class="contents-el" href="#full-transformer">Full Transformer</a></li>
    </ul></li>
   <li><a class="contents-el" href="#try-it-out">Try it out!</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Clean Transformer Implementation
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


    st.markdown(r"""


This diagram shows the high-level transformer architecture. It can be thought of in terms of a sequence of **attention heads** (denoted $h_1, h_2, ...$) and MLPs (denoted $m$), with each one performing operations on the residual stream (which is the central object of the transformer).
""")
    st.markdown(r"")
    st_image("transformer.png", 900)
    st.markdown(r"")
    st.markdown(r"""

## High-Level architecture

Go watch my [Transformer Circuits walkthrough](https://www.youtube.com/watch?v=KV5gbOmHbjU) if you want more intuitions!

(Diagram is bottom to top)

### Summary

The input tokens $t$ are integers. We get them from taking a sequence, and tokenizing it (like we saw in the previous section).

The token embedding is a lookup table mapping tokens to vectors, which is implemented as a matrix $W_E$. The matrix consists of a stack of token embedding vectors (one for each token).

### Residual stream

The residual stream is the sum of all previous outputs of layers of the model, is the input to each new layer. It has shape `[batch, seq_len, d_model]` (where `d_model` is the length of a single embedding vector). 

The initial value of the residual stream is denoted $x_0$ in the diagram, and $x_i$ are later values of the residual stream (after more attention and MLP layers have been applied to the residual stream).

The residual stream is *Really* fundamental. It's the central object of the transformer. It's how model remembers things, moves information between layers for composition, and it's the medium used to store the information that attention moves between positions.

### Transformer blocks

Then we have a series of `n_layers` **transformer blocks** (also sometimes called **residual blocks**).

Note - a block contains an attention layer *and* an MLP layer, but we say a transformer has $k$ layers if it has $k$ blocks (i.e. $2k$ total layers).

#### Attention

First we have attention. This moves information from prior positions in the sequence to the current token. 

We do this for *every* token in parallel using the same parameters. The only difference is that we look backwards only (to avoid "cheating"). This means later tokens have more of the sequence that they can look at.

Attention layers are the only bit of a transformer that moves information between positions (i.e. between vectors at different sequence positions in the residual stream).

Attention layers are made up of `n_heads` heads - each with their own parameters, own attention pattern, and own information how to copy things from source to destination. The heads act independently and additively, we just add their outputs together, and back to the stream.

Each head does the following:
* Produces an **attention pattern** for each destination token, a probability distribution of prior source tokens (including the current one) weighting how much information to copy.
* Moves information (via a linear map) in the same way from each source token to each destination token.

A few key points:

* What information we copy depends on the source token's *residual stream*, but this doesn't mean it only depends on the value of that token, because the residual stream can store more information than just the token identity (the purpose of the attention heads is to move information between vectors at different positions in the residual stream!)
* We can think of each attention head as consisting of two different **circuits**:
    * One circuit determines **where to move information to and from** (this is a function of the residual stream for the source and destination tokens)
    * The other circuit determines **what information to move** (this is a function of only the source token's residual stream)
    * For reasons which will become clear later, we refer to the first circuit as the **QK circuit**, and the second circuit as the **OV circuit**

Below is a schematic diagram of the attention layers; don't worry if you don't follow this right now, we'll go into more detail during implementation.
""")

    st_image("transformer-attn.png", 1300)
    st.markdown(r"")
    st.markdown(r"""
### MLP

The MLP layers are just a standard neural network, with a singular hidden layer and a nonlinear activation function. The exact activation isn't conceptually important ([GELU](https://paperswithcode.com/method/gelu) seems to perform best).

Our hidden dimension is normally `d_mlp = 4 * d_model`. Exactly why the ratios are what they are isn't super important (people basically cargo-cult what GPT did back in the day!).

Importantly, **the MLP operates on positions in the residual stream independently, and in exactly the same way**. It doesn't move information between positions.

Intuition - once attention has moved relevant information to a single position in the residual stream, MLPs can actually do computation, reasoning, lookup information, etc. *What the hell is going on inside MLPs* is a pretty big open problem in transformer mechanistic interpretability - see the [Toy Model of Superposition Paper](https://transformer-circuits.pub/2022/toy_model/index.html) for more on why this is hard.

Another important intuition - `linear map -> non-linearity -> linear map` basically [just works](https://xkcd.com/1838/), and can approximate arbitrary functions.
""")
    st_image("transformer-mlp.png", 720)
    st.markdown(r"""
### Unembedding

Finally, we unembed!

This just consists of applying a linear map $W_U$, going from final residual stream to a vector of logits - this is the output.
""")

    with st.expander("Aside - tied embeddings"):
        st.markdown(r"""
Note - sometimes we use something called a **tied embedding** - this is where we use the same weights for our $W_E$ and $W_U$ matrices. In other words, to get the logit score for a particular token at some sequence position, we just take the vector in the residual stream at that sequence position and take the inner product with the corresponding token embedding vector. This is more training-efficient (because there are fewer parameters in our model), and it might seem pricipled at first. After all, if two words have very similar meanings, shouldn't they have similar embedding vectors because the model will treat them the same, and similar unembedding vectors because they could both be substituted for each other in most output?

However, this is actually not very principled, for the following main reason: **the direct path involving the embedding and unembedding should approximate bigram frequencies**. 

Let's break down this claim. **Bigram frequencies** refers to the frequencies of pairs of words in the english language (e.g. the bigram frequency of "Barack Obama" is much higher than the product of the individual frequencies of the words "Barack" and "Obama"). If our model had no attention heads or MLP layers, then all we have is a linear map from our one-hot encoded token `T` to a probability distribution over the token following `T`. This map is represented by the linear transformation $t \to t^T W_E W_U$ (where $t$ is our one-hot encoded token vector). Since the output of this transformation can only be a function of the token `T` (and no earlier tokens), the best we can do is have this map approximate the true frequency of bigrams starting with `T`, which appear in the training data. Importantly, **this is not a symmetric map**. We want `T = "Barack"` to result in a high probability of the next token being `"Obama"`, but not the other way around!

Even in multi-layer models, a similar principle applies. There will be more paths through the model than just the "direct path" $W_E W_U$, but because of the residual connections there will always exist a direct path, so there will always be some incentive for $W_E W_U$ to approximate bigram frequencies.
""")
    st.markdown(r"""
### Bonus things - less conceptually important but key technical details

#### LayerNorm

* Simple normalization function applied at the start of each layer (i.e. before each MLP, attention layer, and before the unembedding)
* Converts each input vector (independently in parallel for each batch x position residual stream vector) to have mean zero and variance 1.
* Then applies an elementwise scaling and translation
* Cool maths tangent: The scale & translate is just a linear map. LayerNorm is only applied immediately before another linear map. Linear compose linear = linear, so we can just fold this into a single effective linear layer and ignore it.
    * `fold_ln=True` flag in `from_pretrained` does this for you.
* LayerNorm is annoying for interpertability - the scale part is not linear, so you can't think about different bits of the input independently. But it's *almost* linear - if you're changing a small part of the input it's linear, but if you're changing enough to alter the norm substantially it's not linear :(

#### Positional embeddings

* **Problem:** Attention operates over all pairs of positions. This means it's symmetric with regards to position - the attention calculation from token 5 to token 1 and token 5 to token 2 are the same by default
    * This is dumb because nearby tokens are more relevant.
* There's a lot of dumb hacks for this.
* We'll focus on **learned, absolute positional embeddings**. This means we learn a lookup table mapping the index of the position of each token to a residual stream vector, and add this to the embed.
    * Note that we *add* rather than concatenate. This is because the residual stream is shared memory, and likely under significant superposition (the model compresses more features in there than the model has dimensions)
    * We basically never concatenate inside a transformer, unless doing weird shit like generating text efficiently.
* One intuition: 
    * *Attention patterns are like generalised convolutions* (where the transformer learns which words in a sentence are relevant to each other, as opposed to convolutions which just imposes the fixed prior of "pixels close to each other are relevant, pixels far away are not.")
    * Positional information helps the model figure out that words are close to each other, which is helpful because this probably implies they are relevant to each other.

## Actual Code!

Key (for the results you get when running the code immediately below)

```
batch = 1
position = 35
d_model = 768
n_heads = 12
n_layers = 12
d_mlp = 3072 (4 * d_model)
d_head = 64 (d_model / n_heads)
```

### Parameters and Activations

It's important to distinguish between parameters and activations in the model.

* **Parameters** are the weights and biases that are learned during training.
    * These don't change when the model input changes.
    * They can be accessed direction fromm the model, e.g. `model.W_E` for the token embedding.
* **Activations** are temporary numbers calculated during a forward pass, that are functions of the input.
    * We can think of these values as only existing for the duration of a single forward pass, and disappearing afterwards.
    * We can use hooks to access these values during a forward pass (more on hooks later), but it doesn't make sense to talk about a model's activations outside the context of some particular input.
    * Attention scores and patterns are activations (this is slightly non-intuitve because they're used in a matrix multiplication with another activation).

The dropdown below contains a diagram of a single layer (called a `TransformerBlock`) for an attention-only model with no biases. Each box corresponds to an **activation** (and also tells you the name of the corresponding hook point, which we will eventually use to access those activations). The red text below each box tells you the shape of the activation (ignoring the batch dimension). Each arrow corresponds to an operation on an activation; where there are **parameters** involved these are labelled on the arrows.

#### Print All Activation Shapes of Reference Model

Run the following code to print all the activation shapes of the reference model:

```python
for activation_name, activation in cache.cache_dict.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(activation_name, activation.shape)
```

#### Print All Parameters Shapes of Reference Model

```python
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(name, param.shape)
```

The diagram below shows the name of all activations and parameters in a fully general transformer model from transformerlens (except for a few at the start and end, like the embedding and unembedding). Lots of this won't make sense at first, but you can return to this diagram later and check that you understand most/all parts of it.
""")

    with st.expander("Diagram"):
        st.write("""<figure style="max-width:680px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrdV1FP2zAQ_itWpI1tasSIeApdJaYWJqGNIRA8UBS5sdNadeLUdkJbwn_fOUkJ6ZoCm9R2y4N955yT786fz-cHyxeEWq41lDgeoatuP0LwqGRQDPSttopxhJSecfplLxCRthWbU9c5jKd7nSuJIxUIGVL5lQt_3N431p2-VXzGPD7HSnVpgGgY6xm6Z0SP3M_xtDWibDjSRjxaYW1gQcOFdCUlYFHZSKoY8WJJb_vWk9wmLF2gHAhJqLS1iF0nniIlOCNowLE_PgqxHLIof5U70N6HeZ12_rdydvXTytsDxxh_UHTSQsQLwZp_bO-bWeDrnW3b3Vt057pu7qNtdzJMSFZgCxmB973G97FQunIElG16UgW5kl7LBR4doPc0VPFR2T1v0ZruJev17QrK5ah9zGl9KAKeYg6ASTVOI7KCWAiWCGXguJbY1yikOIJosZRBbAczCADJ8u_DuuX95pbs4NliFShvPA7gBtBmlYMArFL-VUJhrSP0Flqgt3Z_1jYwLq2rk7o6rqvGN0_5AihXf3FSV2MwpDKqD57W1XldhU8mXDdQvGKFyUI33rWhznWWAmHSzfFkRDHxGJkaxhi5nktPl3LlfX5QUIJwOkQiQCnmCUUp9bWQKpsD9PlOQF_sx_OsWIIiq4OwHXTLW9HAg5wWIpFSmVuqFoJzCA0YVsCC8ywnpUgM8IW47WO1mbkXhrkX2QTATnaFuSdLzCVCo1gKksAhgrmIhuWsVnE8IRwRFGI1zp6lg0XwC20jnlVOgY8XeXtWcwx4IwId4mlW5iMAWUq7AdA-bSbKmSHKWTYGzOOdIcrfFVoOevHYW1cWOU11kbO2MIJK9qlQBXmbqeG1BZqzsxWa81-UaCGPX1tfNRByJfnyykcule_mbvRiVeMsQs7ykLMoK-6JG70hHqJPjZwFunoBoCpufZu97zXjgoDBYW8iBl0Gq1qWAaW05a1uo17JzXzVC_HdOwQAfxx_720E3eW345-933bNlkFYLSukQH1GLNd6MJD6lh7RkPYtF0RCA2yuArDlHsE0iQnWtEcY1M2WG2CuaMvCiRaXs8i3XC0TujDqMgz7PyytHn8BSKkJUQ" /></figure>""", unsafe_allow_html=True)

    st.markdown(r"""
### Config

The config object contains all the hyperparameters of the model. We can print the config of the reference model to see what it contains:

```python
# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
print(reference_gpt2.cfg)
```

We define a stripped down config for our model:

```python
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)
```

### Tests

Tests are great, write lightweight ones to use as you go!

**Naive test:** Generate random inputs of the right shape, input to your model, check whether there's an error and print the correct output.

```python
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randn(shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randint(100, 1000, shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(input)
    print("Reference output shape:", reference_output.shape, "\n")

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")
```

### LayerNorm

You should fill in the code below, and then run the tests to verify that your layer is working correctly.

Your LayerNorm should do the following:

* Make mean 0
* Normalize to have variance 1
* Scale with learned weights
* Translate with learned bias

You can use the PyTorch [LayerNorm documentation](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) as a reference. A few more notes:

* Your layernorm implementation always has `affine=True`, i.e. you do learn parameters `w` and `b` (which are represented as $\gamma$ and $\beta$ respectively in the PyTorch documentation).
* Remember that, after the centering and normalization, each vector of length `d_model` in your input should have mean 0 and variance 1.
* As the PyTorch documentation page says, your variance should be computed using `unbiased=False`.
* The `layer_norm_eps` argument in your config object corresponds to the $\epsilon$ term in the PyTorch documentation (it is included to avoid division-by-zero errors).
* We've given you a `debug` argument in your config. If `debug=True`, then you can print output like the shape of objects in your `forward` function to help you debug (this is a very useful trick to improve your coding speed).

```python
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual):
        # residual: [batch, position, d_model]
        "YOUR CODE HERE"
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b
```
""")
    st.markdown(r"""
```python
rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
```

### Embedding

Basically a lookup table from tokens to residual stream vectors.

(Hint - you can implement this in just one line!)

```python
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        return self.W_E[tokens]

rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)
    
    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :] # [batch, position, d_model]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed

rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)
```""")

    st.markdown(r"""

### Positional Embedding

```python
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)
    
    def forward(self, tokens):
        "YOUR CODE HERE"

rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
```""")
    st.markdown(r"""

### Attention

* **Step 1:** Produce an attention pattern - for each destination token, probability distribution over previous tokens (including current token)
    * Linear map from input -> query, key shape `[batch, seq_posn, head_index, d_head]`
    * Dot product every *pair* of queries and keys to get attn_scores `[batch, head_index, query_pos, key_pos]` (query = dest, key = source)
    * Scale and mask `attn_scores` to make it lower triangular, ie causal
    * Softmax along the `key_pos` dimension, to get a probability distribution for each query (destination) token - this is our attention pattern!
* **Step 2:** Move information from source tokens to destination token using attention pattern (move = apply linear map)
    * Linear map from input -> value `[batch, key_pos, head_index, d_head]`
    * Mix along the `key_pos` with attn pattern to get `z`, which is a weighted average of the value vectors `[batch, query_pos, head_index, d_head]`
    * Map to output, `[batch, position, d_model]` (position = query_pos, we've summed over all heads)

Below is a much larger, more detailed version of the attention head diagram from earlier. Note that this diagram assumes a batch size of 1 (in the general case, all terms except the model weights $W_Q$, $b_Q$ etc will have a batch dimension at the start). Also, this diagram shows how the attention layer works for all heads (whenever there are three dimensional tensors in the diagram, the depth axis represents different heads).

```
Key for dimensions in the diagram:

n = num heads / head index
s = sequence length / position (s_q and s_k are the same, but indexed to distinguish them)
e = embedding dimension (also called d_model)
h = head size (also called d_head, or d_k)
```
""")
    st_image("transformer-attn-2.png", 1200)
    with st.expander("A few extra notes on attention (optional)"):
        st.markdown(r"""
Usually we have the relation `e = n * h` (i.e. `d_model = num_heads * d_head`). There are some computational justifications for this, but mostly this is just done out of convention (just like how we usually have `d_mlp = 4 * d_model`!).

---

The names **keys**, **queries** and **values** come from their analogy to retrieval systems. Broadly speaking:

* The **queries** represent some information that a token is **"looking for"**
* The **keys** represent the information that a token **"contains"**
    * So the attention score being high basically means that the source (key) token contains the information which the destination (query) token **is looking for**
* The **values** represent the information that is actually taken from the source token, to be moved to the destination token

---

This diagram can better help us understand the difference between the **QK** and **OV** circuit. We'll discuss this just briefly here, and will go into much more detail later on.

The **QK** circuit consists of the operation of the $W_Q$ and $W_K$ matrices. In other words, it determines the attention pattern, i.e. where information is moved to and from in the residual stream. The functional form of the attention pattern $A$ is:

$$
A = \text{softmax}\left(\frac{x^T W_Q W_K^T x}{\sqrt{d_{head}}}\right)
$$

where $x$ is the residual stream (shape `[seq_len, d_model]`), and $W_Q$, $W_K$ are the weight matrices for a single head (i.e. shape `[d_model, d_head]`).

The **OV** circuit consists of the operation of the $W_V$ and $W_O$ matrices. Once attention patterns are fixed, these matrices operate on the residual stream at the source position, and their output is the thing which gets moved from source to destination position.

The functional form of an entire attention head is:

$$
\begin{aligned}
\text{output} &= \text{softmax}\left(\frac{x W_Q W_K^T x^T}{\sqrt{d_{head}}}\right) (x W_V W_O) \\
    &= Ax W_V W_O
\end{aligned}
$$

where $W_V$ has shape `[d_model, d_head]`, and $W_O$ has shape `[d_head, d_model]`.

Here, we can clearly see that the **QK circuit** and **OV circuit** are doing conceptually different things, and should be thought of as two distinct parts of the attention head.

Again, don't worry if you don't follow all of this right now - we'll go into **much** more detail on all of this in subsequent exercises. The purpose of the discussion here is just to give you a flavour of what's to come!
""")
    st.markdown(r"""
First, it's useful to visualize and play around with attention patterns - what exactly are we looking at here? (Click on a head to lock onto just showing that head's pattern, it'll make it easier to interpret)

```python
import circuitsvis as cv
from IPython.display import display

display(cv.attention.attention_patterns(tokens=reference_gpt2.to_str_tokens(reference_text), attention=cache["pattern", 0][0]))
```

---

Note - don't worry if you don't get 100% accuracy here; the tests are pretty stringent. Even things like having your `einsum` input arguments in a different order might result in the output being very slightly different. You should be getting at least 99% accuracy though, so if the value is lower then this it probably means you've made a mistake somewhere.

Also, this implementation will probably be the most challenging exercise on this page, so don't worry if it takes you some time! You should look at parts of the solution if you're stuck.

```python
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        "YOUR CODE HERE"

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        "YOUR CODE HERE"

rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
```""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)
        
        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"])
```""")

    st.markdown(r"""

### MLP

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
    
    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        "YOUR CODE HERE"

rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
    
    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out

rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])
```
""")

    st.markdown(r"""
### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        "YOUR CODE HERE"

rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attn_out
        
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post

rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
```""")

    st.markdown(r"""
### Unembedding

```python
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))
    
    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        "YOUR CODE HERE"

rand_float_test(Unembed, [2, 4, 768])
load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))
    
    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits

rand_float_test(Unembed, [2, 4, 768])
load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
```
""")

    st.markdown(r"""
### Full Transformer

```python
class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens):
        # tokens [batch, position]
        "YOUR CODE HERE"

rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits

rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
```""")

    st.markdown(r"""

## Try it out!

```python
demo_gpt2 = DemoTransformer(Config(debug=False))
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
demo_gpt2.cuda()
```

Let's take a test string, and calculate the loss!

```python
test_string = '''There is a theory which states that if ever anyone discovers exactly what the Universe is for and why it is here, it will instantly disappear and be replaced by something even more bizarre and inexplicable. There is another theory which states that this has already happened.'''
test_tokens = reference_gpt2.to_tokens(test_string).cuda()
demo_logits = demo_gpt2(test_tokens)
```

```python
def lm_cross_entropy_loss(logits, tokens):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()
loss = lm_cross_entropy_loss(demo_logits, test_tokens)
print(loss)
print("Loss as average prob", (-loss).exp())
print("Loss as 'uniform over this many variables'", (loss).exp())
print("Uniform loss over the vocab", math.log(demo_gpt2.cfg.d_vocab))
```

We can also greedily generate text:

```python
test_string = '''There is a theory which states that if ever anyone discovers exactly what the Universe is for and why it is here, it will instantly disappear and be replaced by something even more bizarre and inexplicable. There is another theory which states that'''
for i in tqdm.tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).cuda()
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
print(test_string)
```
""")

def section_training():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#config">Config</a></li>
   <li><a class="contents-el" href="#create-data">Create Data</a></li>
   <li><a class="contents-el" href="#create-model">Create Model</a></li>
   <li><a class="contents-el" href="#create-optimizer">Create Optimizer</a></li>
   <li><a class="contents-el" href="#run-training-loop">Run Training Loop</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Training a Model!
""")
    st.info(r"""
## Learning Objectives

* Use the `Adam` optimizer to train your transformer
* Run a training loop on a very small dataset, and verify that your model's loss is going down
""")
    st.markdown(r"""

This is a lightweight demonstration of how you can actually train your own GPT-2 with this code! Here we train a tiny model on a tiny dataset, but it's fundamentally the same code for training a larger/more real model (though you'll need beefier GPUs and data parallelism to do it remotely efficiently, and fancier parallelism for much bigger ones).

For our purposes, we'll train 2L 4 heads per layer model, with context length 256, for 1000 steps of batch size 8, just to show what it looks like.

""")

    st.error(r"""
You should use the Colab to run this code, since it will put quite a strain on your GPU otherwise! You can access a complete version of the Colab [here](https://colab.research.google.com/drive/1kP27XsoJsPeCyVtzeolNMlVAsLFEmxZ6?usp=sharing#scrollTo=QfeyG6NZm4SC).
""")

# ```python
# %pip install datasets
# %pip install transformers
# ```

# ```python
# import datasets
# import transformers
# import plotly.express as px
# ```

# ## Config

# ```python
# batch_size = 8
# num_epochs = 1
# max_steps = 1000
# log_every = 10
# lr = 1e-3
# weight_decay = 1e-2
# model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)
# ```

# ## Create Data

# We load in a tiny dataset I made, with the first 10K entries in the Pile (inspired by Stas' version for OpenWebText!)

# ```python
# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
# print(dataset)
# print(dataset[0]['text'][:100])
# tokens_dataset = utils.tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
# data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# ```

# ## Create Model

# ```python
# model = DemoTransformer(model_cfg)
# model.cuda()
# ```

# ```
# DemoTransformer(
#   (embed): Embed()
#   (pos_embed): PosEmbed()
#   (blocks): ModuleList(
#     (0): TransformerBlock(
#       (ln1): LayerNorm()
#       (attn): Attention()
#       (ln2): LayerNorm()
#       (mlp): MLP()
#     )
#     (1): TransformerBlock(
#       (ln1): LayerNorm()
#       (attn): Attention()
#       (ln2): LayerNorm()
#       (mlp): MLP()
#     )
#   )
#   (ln_final): LayerNorm()
#   (unembed): Unembed()
# )
# ```

# ## Create Optimizer

# We use AdamW - it's a pretty standard optimizer.

# ```python
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# ```

# ## Run Training Loop

# ```python
# losses = []
# print("Number of batches:", len(data_loader))
# for epoch in range(num_epochs):
#     for c, batch in tqdm.tqdm(enumerate(data_loader)):
#         tokens = batch['tokens'].cuda()
#         logits = model(tokens)
#         loss = lm_cross_entropy_loss(logits, tokens)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         losses.append(loss.item())
#         if c % log_every == 0:
#             print(f"Step: {c}, Loss: {loss.item():.4f}")
#         if c > max_steps:
#             break
# ```

# We can now plot a loss curve!

# ```python
# px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")
# ```
# """)

func_page_list = [
    (section_home, "🏠 Home"), 
    (section_intro, "1️⃣ Understanding Inputs & Outputs of a Transformer"), 
    (section_code, "2️⃣ Clean Transformer Implementation"), 
    (section_training, "3️⃣ Training a model"), 
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
