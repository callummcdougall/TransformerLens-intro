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

NAMES = ["attribution_fig", "attribution_fig_2", "failure_types_fig", "failure_types_fig_2", "true_images/attn_probs_red", "true_images/attn_qpos1"]
def get_fig_dict():
    return {name: read_from_html(name) for name in NAMES}
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
if NAMES[0] not in st.session_state["fig_dict"]:
    st.session_state["fig_dict"] |= get_fig_dict()
fig_dict = st.session_state["fig_dict"]

def section_home():
    st.markdown(r"""
## 1️⃣ Bracket classifier

We'll start by looking at our bracket classifier and dataset, and see how it works. We'll also write our own hand-coded solution to the balanced bracket problem (understanding what a closed-form solution looks like will be helpful as we discover how our transformer is solving the problem).

## 2️⃣ Moving backwards

If we want to investigate which heads cause the model to classify a bracket string as balanced or unbalanced, we need to work our way backwards from the input. Eventually, we can find things like the **resudiaul stream unbalanced directions**, which are the directions of vectors in the residual stream which contribute most to the model's decision to classify a string as unbalanced.

This will require learning how to use **hooks**, to capture inputs and outputs of intermediate layers in the model.

## 3️⃣ Total elevation circuit

In this section (which is the meat of the exercises), we'll hone in on a particular circuit and try to figure out what kinds of composition it is using.

## 4️⃣ Bonus exercises

Now that we have a first-pass understanding of the total elevation circuit, we can try to go a bit deeper by:

* Getting a first-pass understanding of the other important circuit in the model (the no negative failures circuit)
* Exploiting our understanding of how the model classifies brackets to construct **advexes** (adversarial examples)
""")

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#life-on-the-frontier">Life On The Frontier</a></li>
   <li><a class="contents-el" href="#today-s-toy-model">Today's Toy Model</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#model-architecture">Model architecture</a></li>
   </ul></li>
   <li><a class="contents-el" href="#tokenizer">Tokenizer</a></li>
   <li><a class="contents-el" href="#dataset">Dataset</a></li>
   <li><a class="contents-el" href="#hand-written-solution">Hand-Written Solution</a></li>
   <li><a class="contents-el" href="#hand-written-solution-vectorized">Hand-Written Solution - Vectorized</a></li>
   <li><a class="contents-el" href="#the-model-s-solution">The Model's Solution</a></li>
   <li><a class="contents-el" href="#running-the-model">Running the Model</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""

# Bracket classifier

One of the many behaviors that a large language model learns is the ability to tell if a sequence of nested parentheses is balanced. For example, `(())()`, `()()`, and `(()())` are balanced sequences, while `)()`, `())()`, and `((()((())))` are not.

In training, text containing balanced parentheses is much more common than text with imbalanced parentheses - particularly, source code scraped from GitHub is mostly valid syntactically. A pretraining objective like "predict the next token" thus incentivizes the model to learn that a close parenthesis is more likely when the sequence is unbalanced, and very unlikely if the sequence is currently balanced.

Some questions we'd like to be able to answer are:

- How robust is this behavior? On what inputs does it fail and why?
- How does this behavior generalize out of distribution? For example, can it handle nesting depths or sequence lengths not seen in training?

If we treat the model as a black box function and only consider the input/output pairs that it produces, then we're very limited in what we can guarantee about the behavior, even if we use a lot of compute to check many inputs. This motivates interpretibility: by digging into the internals, can we obtain insight into these questions? If the model is not robust, can we directly find adversarial examples that cause it to confidently predict the wrong thing? Let's find out!

## Life On The Frontier

Unlike many of the days in the curriculum which cover classic papers and well-trodden topics, today you're at the research frontier, covering current research at Redwood. This is pretty cool, but also means you should expect that things will be more confusing and complicated than other days. TAs might not know answers because in fact nobody knows the answer yet, or might be hard to explain because nobody knows how to explain it properly yet.

Feel free to go "off-road" and follow your curiosity - you might discover uncharted lands 🙂

## Today's Toy Model

Today we'll study a small transformer that is trained to only classify whether a sequence of parentheses is balanced or not. It's small so we can run experiments quickly, but big enough to perform well on the task. The weights and architecture are provided for you.

### Model architecture

#### Causal vs bidirectional attention

The key difference between this and the GPT-style models you will have implemented already is the attention mechanism. 

GPT uses **causal attention**, where the attention scores get masked wherever the source token comes after the destination token. This means that information can only flow forwards in a model, never backwards (which is how we can train our model in parallel - our model's output is a series of distributions over the next token, where each distribution is only able to use information from the tokens that came before). This model uses **bidirectional attention**, where the attention scores aren't masked based on the relative positions of the source and destination tokens. This means that information can flow in both directions, and the model can use information from the future to predict the past.

#### Using transformers for classification

GPT is trained via gradient descent on the cross-entropy loss between its predictions for the next token and the actual next tokens. Models designed to perform classification are trained in a very similar way, but instead of outputting probability distributions over the next token, they output a distribution over class labels. We do this by having an unembedding matrix of size `[d_model, num_classifications]`, and only using a single sequence position (usually the 0th position) to represent our classification probabilities.

Below is a schematic to compare the model architectures and how they're used (you might need to open the image in a new tab to see it clearly):
""")

    st_image("true_images/gpt-vs-bert.png", 1200)
    st.markdown("")

    st.markdown(r"""

#### Masking padding tokens

The image on the top-right is actually slightly incomplete. It doesn't show how our model handles sequences of differing lengths. After all, during training we need to have all sequences be of the same length so we can batch them together in a single tensor. The model manages this via two new tokens: the end token and the padding token.

The end token goes at the end of every bracket sequence, and then we add padding tokens to the end until the sequence is up to some fixed length. For instance, this model was trained on bracket sequences of up to length 40, so if we wanted to classify the bracket string `(())` then we would pad it to the length-42 sequence:

```
[start] + ( + ( + ) + ) + [end] + [pad] + [pad] + ... + [pad]
```
""")

    with st.expander("Aside on how this relates to BERT"):
        st.markdown(r"""
This is all very similar to how the bidirectional transformer **BERT** works:

* BERT has the `[CLS]` (classification) token rather than `[start]`; but it works exactly the same.
* BERT has the `[SEP]` (separation) token rather than `[end]`; this has a similar function but also serves a special purpose when it is used in **NSP** (next sentence prediction).

If you're interested in reading more on this, you can check out [this link](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/).
""")

    st_image("true_images/gpt-vs-bert-2.png", 800)
    st.markdown("")

    st.markdown(r"""
TransformerLens does not (yet) provide support for this type of masking, so we have implemented it for you using TransformerLens's **permanent hooks** feature. If you are interested then you can read the code below to see how it works, but it's more important that you understand what padding is happening than understanding exactly how it's implemented.

#### Other details

Here is a summary of all the relevant architectural details:

* Positional embeddings are sinusoidal (non-learned).
* It has `hidden_size` (aka `d_model`, aka `embed_dim`) of 56.
* It has bidirectional attention, like BERT.
* It has 3 attention layers and 3 MLPs.
* Each attention layer has two heads, and each head has `headsize` (aka `d_head`) of `hidden_size / num_heads = 28`.
* The MLP hidden layer has 56 neurons (i.e. its linear layers are square matrices).
* The input of each attention layer and each MLP is first layernormed, like in GPT.
* There's a LayerNorm on the residual stream after all the attention layers and MLPs have been added into it (this is also like GPT).
* Our embedding matrix `W_E` has five rows: one for each of the tokens `[start]`, `[pad]`, `[end]`, `(`, and `)` (in that order).
* Our unembedding matrix `W_U` has two columns: one for each of the classes `unbalanced` and `balanced` (in that order). 
    * When running our model, we get output of shape `[batch, seq_len, 2]`, and we then take the `[:, 0, :]` slice to get the output for the `[start]` token (i.e. the classification logits).
    * We can then softmax to get our classification probabilities.
* Activation function is `ReLU`.

To refer to attention heads, we'll again use the shorthand `layer.head` where both layer and head are zero-indexed. So `2.1` is the second attention head (index 1) in the third layer (index 2). With this shorthand, the network graph looks like:""")

    st_image("true_images/bracket-transformer-entire-model.png", 300)
    st.markdown("")
    # st.write("""<figure style="max-width:400px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp9k19vgjAUxb8K6bMulEey-DI1MWHORN9gWaq9ajPaklISjfjdV-j4YyHw1PvrObcnl_aBTpICCtFFkezqHZaJ8MyXF0cLPiTPpAChc7tRfQf5C8KbzxdeyYSGC6iyRit-BBrXy_ejWtQlZeLyXWsjcgflb0RMKO2Tz2g3gHhIxmTBkDiydbTtl-XtR0jFS2_NBElrx9bUcV3aDl4FWjWFajyqjJgAohqay7Pm5FZ6e7uo-Vehs0J3U9rJnGkmnUEZasfUbJN0YlZdt4bY7a0ft-Gtw3_z3Zn2zGN6PKHvYHOeKdwWBvkvv8xpgFs3dq24nxYP0o7o8YS-g81542nxy81xGgStO3CtQT9tMEg7oscT-g42542nDboLbN0gKJohDooTRs2LfVQ4QfoKHBIUmiWFMylSnaBEPI20yCjRsKJMS4XCM0lzmCFSaLm_ixMKtSqgES0ZMe-d_6uefzWyPj4" /></figure>""", unsafe_allow_html=True)

# ```mermaid
# graph TD
#     subgraph Components
#         Token --> |integer|TokenEmbed[Token<br>Embedding] --> Layer0In[add] --> Layer0MLPIn[add] --> Layer1In[add] --> Layer1MLPIn[add] --> Layer2In[add] --> Layer2MLPIn[add] --> FLNIn[add] --> |x_norm| FinalLayerNorm[Final Layer Norm] --> |x_decoder|Linear --> |x_softmax| Softmax --> Output
#         Position --> |integer|PosEmbed[Positional<br>Embedding] --> Layer0In
#         Layer0In --> LN0[LayerNorm] --> 0.0 --> Layer0MLPIn
#         LN0[LayerNorm] --> 0.1 --> Layer0MLPIn
#         Layer0MLPIn --> LN0MLP[LayerNorm] --> MLP0 --> Layer1In
#         Layer1In --> LN1[LayerNorm] --> 1.0 --> Layer1MLPIn
#         LN1[LayerNorm] --> 1.1 --> Layer1MLPIn
#         Layer1MLPIn --> LN1MLP[LayerNorm] --> MLP1 --> Layer2In
#         Layer2In --> LN2[LayerNorm] --> 2.0 --> Layer2MLPIn
#         LN2[LayerNorm] --> 2.1 --> Layer2MLPIn
#         Layer2MLPIn --> LN2MLP[LayerNorm] --> MLP2 --> FLNIn
#     end
# ```
    st.markdown(r"""
Again, here is a diagram showing the hook names in an attention block, to serve as a useful reference:""")

    with st.expander("Full diagram"):
        st.markdown(r"""
Note that you can access the layernorms using e.g. `utils.get_act_name("normalized", 3, "ln2")` (for the layernorm before MLPs in the 3rd layer).

Another note - make sure you don't get mixed up by `pre` and `post` for the MLPs. This refers to pre and post-activation function, not pre and post-MLP. The input to the MLP is `ln2.hook_normalized`, and the output is `mlp_out`.

The other hooks not shown in this diagram which are useful to know:

* `hook_embed` for token embeddings
* `hook_pos_embed` for positional embeddings
* `ln_final.hook_normalized` for the output of the final layernorm (i.e. just before the unembedding)
""")
        st.write("""<figure style="max-width:680px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNrdV1FP2zAQ_itWpI1tasSIeApdJaYWJqGNIRA8UBS5sdNadeLUdkJbwn_fOUkJ6ZoCm9R2y4N955yT786fz-cHyxeEWq41lDgeoatuP0LwqGRQDPSttopxhJSecfplLxCRthWbU9c5jKd7nSuJIxUIGVL5lQt_3N431p2-VXzGPD7HSnVpgGgY6xm6Z0SP3M_xtDWibDjSRjxaYW1gQcOFdCUlYFHZSKoY8WJJb_vWk9wmLF2gHAhJqLS1iF0nniIlOCNowLE_PgqxHLIof5U70N6HeZ12_rdydvXTytsDxxh_UHTSQsQLwZp_bO-bWeDrnW3b3Vt057pu7qNtdzJMSFZgCxmB973G97FQunIElG16UgW5kl7LBR4doPc0VPFR2T1v0ZruJev17QrK5ah9zGl9KAKeYg6ASTVOI7KCWAiWCGXguJbY1yikOIJosZRBbAczCADJ8u_DuuX95pbs4NliFShvPA7gBtBmlYMArFL-VUJhrSP0Flqgt3Z_1jYwLq2rk7o6rqvGN0_5AihXf3FSV2MwpDKqD57W1XldhU8mXDdQvGKFyUI33rWhznWWAmHSzfFkRDHxGJkaxhi5nktPl3LlfX5QUIJwOkQiQCnmCUUp9bWQKpsD9PlOQF_sx_OsWIIiq4OwHXTLW9HAg5wWIpFSmVuqFoJzCA0YVsCC8ywnpUgM8IW47WO1mbkXhrkX2QTATnaFuSdLzCVCo1gKksAhgrmIhuWsVnE8IRwRFGI1zp6lg0XwC20jnlVOgY8XeXtWcwx4IwId4mlW5iMAWUq7AdA-bSbKmSHKWTYGzOOdIcrfFVoOevHYW1cWOU11kbO2MIJK9qlQBXmbqeG1BZqzsxWa81-UaCGPX1tfNRByJfnyykcule_mbvRiVeMsQs7ykLMoK-6JG70hHqJPjZwFunoBoCpufZu97zXjgoDBYW8iBl0Gq1qWAaW05a1uo17JzXzVC_HdOwQAfxx_720E3eW345-933bNlkFYLSukQH1GLNd6MJD6lh7RkPYtF0RCA2yuArDlHsE0iQnWtEcY1M2WG2CuaMvCiRaXs8i3XC0TujDqMgz7PyytHn8BSKkJUQ" /></figure>""", unsafe_allow_html=True)


    st.markdown(r"""
## Imports

Run the code below for all necessary imports.

```python
import functools
import json
import os
import sys
from typing import Dict, List, Tuple, Union, Optional
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import einops
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

MAIN = __name__ == "__main__"

t.set_grad_enabled(False)
device = t.device("cpu")

import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig
from torchtyping import TensorType as TT

from brackets_datasets import SimpleTokenizer, BracketsDataset

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
```

## Defining the model

Here, we define the model according to the description we gave above.

```python
if MAIN:
    VOCAB = "()"

    cfg = HookedTransformerConfig(
        n_ctx=42,
        d_model=56,
        d_head=28,
        n_heads=2,
        d_mlp=56,
        n_layers=3,
        attention_dir="bidirectional", # defaults to "causal"
        act_fn="relu",
        d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
        d_vocab_out=2, # 2 because we're doing binary classification
        use_attn_result=True, 
        device="cpu",
        hook_tokens=True
    )

    model = HookedTransformer(cfg).eval()

    state_dict = t.load("state_dict.pt")
    model.load_state_dict(state_dict)
```

## Tokenizer

There are only five tokens in our vocabulary: `[start]`, `[pad]`, `[end]`, `(`, and `)` in that order.

You have been given a tokenizer `SimpleTokenizer("()")` which will give you some basic functions. Try running the following to see what they do:

```python
if MAIN:
    tokenizer = SimpleTokenizer("()")

    # Examples of tokenization
    # (the second one applies padding, since the sequences are of different lengths)
    print(tokenizer.tokenize("()"))
    print(tokenizer.tokenize(["()", "()()"]))

    # Dictionaries mapping indices to tokens and vice versa
    print(tokenizer.i_to_t)
    print(tokenizer.t_to_i)

    # Examples of decoding (all padding tokens are removed)
    print(tokenizer.decode(t.tensor([[0, 3, 4, 2, 1, 1]])))
```

## Masking padding tokens

Now that we have the tokenizer, we can use it to write hooks that mask the padding tokens. If you understand how the padding works, then don't worry if you don't follow all the implementational details of this code.

```python
def add_hooks_for_masking_PAD(model: HookedTransformer) -> model:

    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(
        tokens: TT["batch", "seq"],
        hook: HookPoint,
    ) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == tokenizer.PAD_TOKEN, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: TT["batch", "head", "seq_Q", "seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model

if MAIN:
    model.reset_hooks(including_permanent=True)
    model = add_hooks_for_masking_PAD(model)
```

## Dataset

Each training example consists of `[start]`, up to 40 parens, `[end]`, and then as many `[pad]` as necessary.

In the dataset we're using, half the sequences are balanced, and half are unbalanced. Having an equal distribution is on purpose to make it easier for the model.

```python
if MAIN:
    N_SAMPLES = 5000
    with open("brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)
    data_tuples = data_tuples[:N_SAMPLES]
    data = BracketsDataset(data_tuples)
```

You are encouraged to look at the code for `BracketsDataset` (right click -> "Go to Definition") to see what methods and properties the `data` object has.

As is good practice, examine the dataset and plot the distribution of sequence lengths (e.g. as a histogram). What do you notice?

```python
if MAIN:
    # YOUR CODE HERE: plot the distribution of sequence lengths
```
""")
    with st.expander("Example code to plot dataset"):
        st.markdown(r"""
```python
if MAIN:
    fig = go.Figure(
        go.Histogram(x=[len(x) for x, _ in data_tuples], nbinsx=data.seq_length),
        layout=dict(title="Sequence Lengths", xaxis_title="Sequence Length", yaxis_title="Count")
    )
    fig.show()
```
""")
    with st.expander("Features of dataset"):
        st.markdown(r"""
The most striking feature is that all bracket strings have even length. We constructed our dataset this way because if we had odd-length strings, the model would presumably have learned the heuristic "if the string is odd-length, it's unbalanced". This isn't hard to learn, and we want to focus on the more interesting question of how the transformer is learning the structure of bracket strings, rather than just their length.

**Bonus exercise - can you describe an algorithm involving a single attention head which the model could use to distinguish between even and odd-length bracket strings?**
""")

    st.markdown(r"""
Now that we have all the pieces in place, we can try running our model on the data and generating some predictions.
```python
if MAIN:
    # Define and tokenize examples
    examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, True, False, True, True, False, True]
    toks = tokenizer.tokenize(examples).to(device)

    # Get output logits for the 0th sequence position (i.e. the [start] token)
    logits = model(toks)[:, 0]

    # Get the probabilities via softmax, then get the balanced probability (which is the second element)
    prob_balanced = logits.softmax(-1)[:, 1]

    # Display output
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))
```

We can also run our model on the whole dataset, and see how many brackets are correctly classified.

```python
def run_model_on_data(model: HookedTransformer, data: BracketsDataset, batch_size: int = 200) -> TT["batch", 2]:
    '''Return probability that each example is balanced'''
    all_logits = []
    for i in tqdm(range(0, len(data.strs), batch_size)):
        toks = data.toks[i : i + batch_size]
        logits = model(toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits

if MAIN:
    test_set = data
    n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
```

## Hand-Written Solution

A nice property of using such a simple problem is we can write a correct solution by hand. Take a minute to implement this using a for loop and if statements.

```python
def is_balanced_forloop(parens: str) -> bool:
    '''Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    pass

if MAIN:
for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_forloop(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")
```
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
def is_balanced_forloop(parens: str) -> bool:

    cumsum = 0
    for paren in parens:
        cumsum += 1 if paren == "(" else -1
        if cumsum < 0:
            return False
    
    return cumsum == 0
```

## Hand-Written Solution - Vectorized

A transformer has an inductive bias towards vectorized operations, because at each sequence position the same weights "execute", just on different data. So if we want to "think like a transformer", we want to get away from procedural for/if statements and think about what sorts of solutions can be represented in a small number of transformer weights.

Being able to represent a solutions in matrix weights is necessary, but not sufficient to show that a transformer could learn that solution through running SGD on some input data. It could be the case that some simple solution exists, but a different solution is an attractor when you start from random initialization and use current optimizer algorithms.

```python
def is_balanced_vectorized(tokens: TT["seq"]) -> bool:
    '''
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    '''
    pass
```
""")
    with st.expander(r"""Hint - Vectorized"""):
        st.markdown(r"""You can do this by indexing into a lookup table of size `vocab_size` and a `t.cumsum`. The lookup table represents the change in "nesting depth""")

    with st.expander("Instructions - Vectorized"):
        st.markdown(r"""
One solution is to map begin, pad, and end tokens to zero, map open paren to 1 and close paren to -1. Then calculate the cumulative sum of the sequence. Your sequence is unbalanced if and only if:

- The last element of the cumulative sum is nonzero
- Any element of the cumulative sum is negative""")
    with st.expander("Solution - Vectorized"):
        st.markdown(r"""
```python
def is_balanced_vectorized(tokens: TT["seq"]) -> bool:
    '''
    tokens: sequence of tokens including begin, end and pad tokens - recall that 3 is '(' and 4 is ')'
    '''
    # Convert start/end/padding tokens to zero, and left/right brackets to +1/-1
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    # Get altitude by taking cumulative sum
    altitude = t.cumsum(change, -1)
    # Check that the total elevation is zero and that there are no negative altitudes
    no_total_elevation_failure = altitude[-1] == 0
    no_negative_failure = altitude.min() >= 0

    return no_total_elevation_failure & no_negative_failure
```""")
    st.markdown("")

    st.markdown(r"""
```python
def is_balanced_vectorized(tokens: t.Tensor) -> bool:
    '''
    tokens: sequence of tokens including begin, end and pad tokens - recall that 3 is '(' and 4 is ')'
    '''
    pass

if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")
```

## The Model's Solution

It turns out that the model solves the problem like this:

At each position `i`, the model looks at the slice starting at the current position and going to the end: `seq[i:]`. It then computes (count of closed parens minus count of open parens) for that slice to generate the output at that position.

We'll refer to this output as the "elevation" at `i`, or equivalently the elevation for each suffix `seq[i:]`.

The sequence is imbalanced if one or both of the following is true:

- `elevation[0]` is non-zero
- `any(elevation < 0)`

For English readers, it's natural to process the sequence from left to right and think about prefix slices `seq[:i]` instead of suffixes, but the model is bidirectional and has no idea what English is. This model happened to learn the equally valid solution of going right-to-left.

We'll spend today inspecting different parts of the network to try to get a first-pass understanding of how various layers implement this algorithm. However, we'll also see that neural networks are complicated, even those trained for simple tasks, and we'll only be able to explore a minority of the pieces of the puzzle.
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#moving-backward">Moving backward</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#stage-1-translating-through-softmax">Stage 1: Translating through softmax</a></li>
       <li><a class="contents-el" href="#stage-2-translating-through-linear">Stage 2: Translating through linear</a></li>
       <li><a class="contents-el" href="#step-3-translating-through-layernorm">Step 3: Translating through LayerNorm</a></li>
       <li><a class="contents-el" href="#introduction-to-hooks">Introduction to hooks</a></li>
   </ul></li>
   <li><a class="contents-el" href="#writing-the-residual-stream-as-a-sum-of-terms">Writing the residual stream as a sum of terms</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#output-by-head-hooks">Output-by-head hooks</a></li>
       <li><a class="contents-el" href="#breaking-down-the-residual-stream-by-component">Breaking down the residual stream by component</a></li>
       <li><a class="contents-el" href="#which-components-matter">Which components matter?</a></li>
       <li><a class="contents-el" href="#head-influence-by-type-of-failures">Head influence by type of failures</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Moving backwards

Suppose we run the model on some sequence and it outputs the classification probabilities `[0.99, 0.01]`, i.e. highly confident classification as "unbalanced".

We'd like to know _why_ the model had this output, and we'll do so by moving backwards through the network, and figuring out the correspondence between facts about earlier activations and facts about the final output. We want to build a chain of connections through different places in the computational graph of the model, repeatedly reducing our questions about later values to questions about earlier values.

Let's start with an easy one. Notice that the final classification probabilities only depend on the difference between the class logits, as softmax is invariant to constant additions. So rather than asking, "What led to this probability on balanced?", we can equivalently ask, "What led to this difference in logits?". Let's move another step backward. Since the logits each a linear function of the output of the final LayerNorm, their difference will be some linear function as well. In other words, we can find a vector in the space of LayerNorm outputs such that the logit difference will be the dot product of the LayerNorm's output with that vector.

We now want some way to tell which parts of the model are doing something meaningful. We will do this by identifying a single direction in the embedding space of the start token that we claim to be the "unbalanced direction": the direction that most indicates that the input string is unbalanced. It is important to note that it might be that other directions are important as well (in particular because of layer norm), but for a first approximation this works well.

We'll do this by starting from the model outputs and working backwards, finding the unbalanced direction at each stage.

The final part of the model is the classification head, which has three stages - the final layernorm, the unembedding, and softmax, at the end of which we get our probabilities.""")

    st_image("true_images/bracket-transformer-first-attr-0.png", 550)

    st.markdown(r"""
Note - for simplicity, we'll ignore the batch dimension in the following discussion.

Some notes on the shapes of the objects in the diagram:

* `x_2` is the vector in the residual stream after layer 2's attention heads and MLPs. It has shape `(seq_len, d_model)`.
* `final_ln_output` has shape `(seq_len, d_model)`.
* `W_U` has shape `(d_model, 2)`, and so `logits` has shape `(seq_len, 2)`.
* We get `P(unbalanced)` by taking the 0th element of the softmaxed logits, for sequence position 0.

### Stage 1: Translating through softmax

Let's get `P(unbalanced)` as a function of the logits. Luckily, this is easy. Since we're doing the softmax over two elements, it simplifies to the sigmoid of the difference of the two logits:

$$
\text{softmax}(\begin{bmatrix} \text{logit}_0 \\ \text{logit}_1 \end{bmatrix})_0 = \frac{e^{\text{logit}_0}}{e^{\text{logit}_0} + e^{\text{logit}_1}} = \frac{1}{1 + e^{\text{logit}_1 - \text{logit}_0}} = \text{sigmoid}(\text{logit}_0 - \text{logit}_1)
$$

Since sigmoid is monotonic, a large value of $\hat{y}_0$ follows from logits with a large $\text{logit}_0 - \text{logit}_1$. From now on, we'll only ask "What leads to a large difference in logits?"

### Stage 2: Translating through linear

The next step we encounter is the decoder: `logits = final_LN_output @ W_U`, where

* `W_U` has shape `(d_model, 2)`
* `final_LN_output` has shape `(seq_len, d_model)`

We can now put the difference in logits as a function of $W$ and $x_{\text{linear}}$ like this:

```
logit_diff = (final_LN_output @ W_U)[0, 0] - (final_LN_output @ W_U)[0, 1]

           = final_LN_output[0, :] @ (W_U[0, :] - W_U[1, :])
```

So a high difference in the logits follows from a high dot product of the output of the LayerNorm with the vector `W_U[0, :] - W_U[1, :]`. We can now ask, "What leads to LayerNorm's output having high dot product with this vector?".

Use the weights of the final linear layer (`model.decoder`) to identify the direction in the space that goes into the linear layer (and out of the LN) corresponding to an 'unbalanced' classification. Hint: this is a one line function.

```python
def get_post_final_ln_dir(model: HookedTransformer) -> TT["d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    pass
```

""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
def get_post_final_ln_dir(model: HookedTransformer) -> TT["d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    return model.W_U[:, 0] - model.W_U[:, 1]
```""")

    st.markdown(r"""

### Step 3: Translating through LayerNorm

We want to find the unbalanced direction before the final layer norm, since this is where we can write the residual stream as a sum of terms. LayerNorm messes with this sort of direction analysis, since it is nonlinear. For today, however, we will approximate it with a linear fit. This is good enough to allow for interesting analysis (see for yourself that the $R^2$ values are very high for the fit)!

With a linear approximation to LayerNorm, which I'll use the matrix `L_final` for, we can translate "What is the dot product of the output of the LayerNorm with the unbalanced-vector?" to a question about the input to the LN. We simply write:

```python
final_ln_output[0, :] = final_ln(x_linear[0, :])

                      ≈ L_final @ x_linear[0, :]
```

""")
    with st.expander("An aside on layernorm"):
        st.markdown(r"""
Layernorm isn't actually linear. It's a combination of a nonlinear function (subtracting mean and dividing by std dev) with a linear one (a learned affine transformation).

However, in this case it turns out to be a decent approximation to use a linear fit. The reason we've included layernorm in these exercises is to give you an idea of how nonlinear functions can complicate our analysis, and some simple hacky ways that we can deal with them. 

When applying this kind of analysis to LLMs, it's sometimes harder to abstract away layernorm as just a linear transformation. For instance, many large transformers use layernorm to "clear" parts of their residual stream, e.g. they learn a feature 100x as large as everything else and use it with layer norm to clear the residual stream of everything but that element. Clearly, this kind of behaviour is not well-modelled by a linear fit.
""")
#     with st.expander("A note on what this linear approximation is actually representing:"):
#         st.info(r"""

# To clarify - we are approximating the LayerNorm operation as a linear operation from `hidden_size -> hidden_size`, i.e.:
# $$
# x_\text{linear}[i,j,k] = \sum_{l=1}^{hidden\_size} L[k,l] \cdot x_\text{norm}[i,j,l]
# $$
# In reality this isn't the case, because LayerNorm transforms each vector in the embedding dimension by subtracting that vector's mean and dividing by its standard deviation:
# $$
# \begin{align*}
# x_\text{linear}[i,j,k] &= \frac{x_\text{norm}[i,j,k] - \mu(x_\text{norm}[i,j,:])}{\sigma(x_\text{norm}[i,j,:])}\\
# &= \frac{x_\text{norm}[i,j,k] - \frac{1}{hidden\_size} \displaystyle\sum_{l=1}^{hidden\_size}x_\text{norm}[i,j,l]}{\sigma(x_\text{norm}[i,j,:])}
# \end{align*}
# $$
# If the standard deviation of the vector in $j$th sequence position is approximately the same across all **sequences in a batch**, then we can see that the LayerNorm operation is approximately linear (if we use a different regression fit $L$ for each sequence position, which we will for most of today). Furthermore, if we assume that the standard deviation is also approximately the same across all **sequence positions**, then using the same $L$ for all sequence positions is a good approximation. By printing the $r^2$ of the linear regression fit (which is the proportion of variance explained by the model), we can see that this is indeed the case. 

# Note that plotting the coefficients of this regression probably won't show a pattern quite as precise as this one, because of the classic problem of **correlated features**. However, this doesn't really matter for the purposes of our analysis, because we want to answer the question "What leads to the input to the LayerNorm having a high dot-product with this new vector?". As an example, suppose two features in the LayerNorm inputs are always identical and their coefficients are $c_1$, $c_2$ respectively. Our model could just as easily learn the coefficients $c_1 + \delta$, $c_2 - \delta$, but this wouldn't change the dot product of LayerNorm's input with the unbalanced vector (assuming we are using inputs that also have the property wherein the two features are identical).
# """)

    st.markdown(r"""

Now, we can ask "What leads to the _input_ to the LayerNorm having a high dot-product with this new vector?""")

    st_image("true_images/bracket-transformer-first-attr.png", 600)
    st.markdown("")

    st.markdown(r"""

At this point, we'll have to start using hooks again, because in order to fit our linear function we'll have to extract the inputs and outputs to the final layernorm.

First, you should implement the function `get_activations` below. This should use `model.run_with_hooks` to return the activations corresponding to the `activation_names` parameter (recall that each activation has an associated hook).

```python
def get_activations(model: HookedTransformer, data: BracketsDataset, names: Union[str, List[str]]) -> Union[t.Tensor, Dict[str, t.Tensor]]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns a tensor of activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    pass
```
""")

    with st.expander("Hint"):
        st.markdown(r"""
To record the activations, define an empty dictionary `activations_dict`, and use the hook functions:

```python
def hook_fn(value, hook):
    activations_dict[hook.name] = value
```
""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
def get_activations(model: HookedTransformer, data: BracketsDataset, names: Union[str, List[str]]) -> Union[t.Tensor, Dict[str, t.Tensor]]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns a tensor of activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    activations_dict = {}
    names = [names] if isinstance(names, str) else names

    def hook_fn(value, hook):
        activations_dict[hook.name] = value

    model.run_with_hooks(
        data.toks,
        return_type=None,
        fwd_hooks=[(lambda hook_name: hook_name in names, hook_fn)]
    )

    if len(names) == 1:
        return activations_dict[names[0]]
    else:
        return activations_dict
```
""")

    st.markdown(r"""

Now, use these functions and the [sklearn LinearRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to find a linear fit to the inputs and outputs of your model's layernorms.

A few notes:

* We've provided you with the helper function `get_input_and_output_hook_names_for_layernorm`. This returns the names of the hooks immediately before and after a given layernorm (see the docstring for how to use it).
* The `get_ln_fit` function takes `seq_pos` as an input. If this is an integer, then we are fitting only for that sequence position. If `seq_pos = None`, then we are fitting for all sequence positions (we aggregate the sequence and batch dimensions before performing our regression).
    * The reason for including this parameter is that sometimes we care about how the layernorm operates on a particular sequence position (e.g. for the final layernorm, we only care about the 0th sequence position), but later on we'll also consider the behaviour of layernorm across all sequence positions.
* You should include a fit coefficient in your linear regression (this is the default for `LinearRegression`).

```python
def LN_hook_names(layer: int, layernorm: Optional[int] = None) -> Tuple[str, str]:
    '''
    Returns the names of the hooks immediately before and after a given layernorm.

    LN_hook_names(layer, 1) gives the names for the layernorm before the attention heads in the given layer.
    LN_hook_names(layer, 2) gives the names for the layernorm before the MLP in the given layer.
    LN_hook_names(-1)       gives the names for the final layernorm (before the unembedding).
    '''
    if layer == -1:
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        assert layernorm in (1, 2)
        input_hook_name = utils.get_act_name("resid_pre" if layernorm==1 else "resid_mid", layer)
        output_hook_name = utils.get_act_name('normalized', layer, f'ln{layernorm}')
    
    return input_hook_name, output_hook_name


def get_ln_fit(
    model: HookedTransformer, data: BracketsDataset, layer: int, layernorm: Optional[int] = None, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit for all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in torch.tensor() to make a dimensionless tensor)
    '''
    pass


if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, model.norm, seq_pos=0)
    print("r^2: ", r2)
    w5d5_tests.test_final_ln_fit(model, data, get_ln_fit)
```""")

    with st.expander("Help - I'm not sure how to fit the linear regression."):
        st.markdown(r"""
If `inputs` and `outputs` are both tensors of shape `(samples, d_model)`, then `LinearRegression().fit(inputs, outputs)` returns the fit object which should be the first output of your function.
""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
def get_ln_fit(
    model: HookedTransformer, data: BracketsDataset, layer: int, layernorm: Optional[int] = None, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in torch.tensor() to make a dimensionless tensor)
    '''

    input_hook_name, output_hook_name = LN_hook_names(layer, layernorm)

    activations_dict = get_activations(model, data, [input_hook_name, output_hook_name])
    inputs = utils.to_numpy(activations_dict[input_hook_name])
    outputs = utils.to_numpy(activations_dict[output_hook_name])

    if seq_pos is None:
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]
    
    final_ln_fit = LinearRegression().fit(inputs, outputs)

    r2 = t.tensor(final_ln_fit.score(inputs, outputs))

    return (final_ln_fit, r2)
```""")
    st.markdown(r"""

Armed with our linear fit, we can now identify the direction in the residual stream before the final layer norm that most points in the direction of unbalanced evidence.


```python
def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> TT["d_model"]:
    pass


if MAIN:
    w5d5_tests.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)
```""")


    st.markdown(r"""
If you're still confused by any of this, the diagram below might help.""")
    st_image("true_images/bracket-transformer-first-attr-soln.png", 1000)
    st.markdown("")
    with st.expander("Solution"):
        st.markdown(r"""
```python
def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> TT["d_model"]:
    
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layer=-1, seq_pos=0)[0]
    final_ln_coefs = t.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir
```""")

    st.markdown(r"""

## Writing the residual stream as a sum of terms

As we've seen in previous exercises, it's much more natural to think about the residual stream as a sum of terms, each one representing a different path through the model. Here, we have ten components which write to the residual stream: the direct path (i.e. the embeddings), and two attention heads and one MLP on each of the three layers. We can write the residual stream as a sum of these terms.

Once we do this, we can narrow in on the components who are making direct contributions to the classification, i.e. which are writing vectors to the residual stream which have a high dot produce with the `pre_final_ln_dir` for unbalanced brackets relative to balanced brackets.

In order to answer this question, we need the following tools:
 - A way to break down the input to the LN by component.
 - A tool to identify a direction in the embedding space that causes the network to output 'unbalanced' (we already have this)

### Breaking down the residual stream by component

Use your `get_activations` function to create a tensor of shape `[num_components, dataset_size, seq_pos]`, where the number of components = 10.

This is a termwise representation of the input to the final layer norm from each component (recall that we can see each head as writing something to the residual stream, which is eventually fed into the final layer norm). The order of the components in your function's output should be:

```
embeddings  -> sum of token and positional embeddings (corresponding to the direct path through the model)
0.0         -> output of the first head in layer 0
0.1         -> output of the second head in layer 0
MLP0        -> output of the MLP in layer 0
1.0         -> output of the first head in layer 1
1.1         -> output of the second head in layer 1
MLP1        -> output of the MLP in layer 1
2.0         -> output of the first head in layer 2
2.1         -> output of the second head in layer 2
MLP2        -> output of the MLP in layer 2
```

(The only term missing from the sum of these is the `W_O`-bias from each of the attention layers).

```python
def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> TT["component", "batch", "seq_pos", "emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, ..., mlp2]
    '''
    pass
```

Now, you can test your function by confirming that input to the final layer norm is the sum of the output of each component and the output projection biases.

```python
if MAIN:
    biases = model.b_O.sum(0)
    out_by_components = get_out_by_components(model, data)
    summed_terms = out_by_components.sum(dim=0) + biases

    final_ln_input_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    final_ln_input = get_activations(model, data, final_ln_input_name)

    t.testing.assert_close(summed_terms, final_ln_input)
```""")

    with st.expander("Hint"):
        st.markdown(r"""
Start by getting all the activation names in a list. You will need `utils.get_act_name("result", layer)` to get the activation names for the attention heads' output, and `utils.get_act_name("mlp_out", layer)` to get the activation names for the MLPs' output.

Once you've done this, and run the `get_activations` function, it's just a matter of doing some reshaping and stacking. Your embedding and mlp activations will have shape `(batch, seq_pos, d_model)`, while your attention activations will have shape `(batch, seq_pos, head_idx, d_model)`.
""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> TT["component", "batch", "seq_pos", "emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)]
    mlp_hook_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]
    
    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    activations = get_activations(model, data, all_hook_names)

    out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = t.concat([
            out, 
            einops.rearrange(
                activations[head_hook_name],
                "batch seq heads emb -> heads batch seq emb"
            ),
            activations[mlp_hook_name].unsqueeze(0)
        ])

    return out
```
""")

    st.markdown(r"""
### Which components matter?

To figure out which components are directly important for the the model's output being "unbalanced", we can see which components tend to output a vector to the position-0 residual stream with higher dot product in the unbalanced direction for actually unbalanced inputs.

The idea is that, if a component is important for correctly classifying unbalanced inputs, then its vector output when fed unbalanced bracket strings will have a higher dot product in the unbalanced direction than when it is fed balanced bracket strings.

In this section, we'll plot histograms of the dot product for each component. This will allow us to observe which components are significant.

For example, suppose that one of our components produced bimodal output like this:""")

    st_image("exampleplot.png", 750)

    st.markdown(r"""
This would be **strong evidence that this component is important for the model's output being unbalanced**, since it's pushing the unbalanced bracket inputs further in the unbalanced direction (i.e. the direction which ends up contributing to the inputs being classified as unbalanced) relative to the balanced inputs.

In the `MAIN` block below, you should compute a `(10, batch)`-size tensor called `magnitudes`. The `[i, j]`th element of this tensors should be the dot product of the `i`th component's output with the unbalanced direction, for the `j`th sequence in your dataset. 

You should normalize it by subtracting the mean of the dot product of this component's output with the unbalanced direction on balanced samples - this will make sure the histogram corresponding to the balanced samples is centered at 0 (like in the figure above), which will make it easier to interpret. Remember, it's only the **difference between the dot product on unbalanced and balanced samples** that we care about (since adding a constant to both logits doesn't change the model's probabilistic output).

The `hists_per_comp` function to plot these histograms has been written for you - all you need to do is calculate the `magnitudes` object and supply it to that function.


```python
def hists_per_comp(magnitudes: TT["component", "batch"], data, xaxis_range=(-1, 1)):
    '''
    Plots the contributions in the unbalanced direction, as supplied by the `magnitudes` tensor.
    '''
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0", (2, 2): "head 0.1", (2, 3): "mlp 0",
        (3, 1): "head 1.0", (3, 2): "head 1.1", (3, 3): "mlp 1",
        (4, 1): "head 2.0", (4, 2): "head 2.1", (4, 3): "mlp 2"
    }
    n_layers = magnitudes.shape[0] // 3
    fig = make_subplots(rows=n_layers+1, cols=3)
    for ((row, col), title), mag in zip(titles.items(), magnitudes):
        fig.add_trace(go.Histogram(x=mag[data.isbal], name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=mag[~data.isbal], name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(width=1200, height=250*(n_layers+1), barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()

if MAIN:
    "TODO: YOUR CODE HERE"

    hists_per_comp(magnitudes, data, xaxis_range=[-10, 20])
```""")

    with st.expander("Hint"):
        st.markdown(r"""
Start by defining these two objects:

* The output by components at sequence position zero, i.e. a tensor of shape `(component, batch, d_model)`
* The `pre_final_ln_dir` vector, which has length `d_model`

Then create magnitudes by calculating an appropriate dot product. 

Don't forget to subtract the mean for each component across all the balanced samples (you can use the boolean `data.isbal` as your index).
""")
    with st.expander("Solution"):
        st.markdown(r"""
```python
if MAIN:
    "TODO: YOUR CODE HERE"
    # Get output by components, at sequence position 0 (which is used for classification)
    out_by_components_seq0: TT["comp", "batch", "d_model"] = out_by_components[:, :, 0, :]
    # Get the unbalanced direction for tensors being fed into the final layernorm
    pre_final_ln_dir: TT["d_model"] = get_pre_final_ln_dir(model, data)
    # Get the size of the contributions for each component
    magnitudes = einsum(
        "comp batch d_model, d_model -> comp batch",
        out_by_components_seq0, 
        pre_final_ln_dir
    )
    # Subtract the mean
    magnitudes_mean_for_each_comp: TT["comp", 1] = magnitudes[:, data.isbal].mean(dim=1).unsqueeze(1)
    magnitudes -= magnitudes_mean_for_each_comp
    # Plot the histograms
    hists_per_comp(magnitudes, data, xaxis_range=[-10, 20])
```
""")

    with st.expander("Click here to see the output you should be getting."):
        st.plotly_chart(fig_dict["attribution_fig"])

    with st.expander("Which heads do you think are the most important, and can you guess why that might be?"):
        st.markdown(r"""
The heads in layer 2 (i.e. `2.0` and `2.1`) seem to be the most important, because the unbalanced brackets are being pushed much further to the right than the balanced brackets. 

We might guess that some kind of composition is going on here. The outputs of layer 0 heads can't be involved in composition because they in effect work like a one-layer transformer. But the later layers can participate in composition, because their inputs come from not just the embeddings, but also the outputs of the previous layer. This means they can perform more complex computations.""")

    st.markdown(r"""
### Head influence by type of failures

Those histograms showed us which heads were important, but it doesn't tell us what these heads are doing, however. In order to get some indication of that, let's focus in on the two heads in layer 2 and see how much they write in our chosen direction on different types of inputs. In particular, we can classify inputs by if they pass the 'overall elevation' and 'nowhere negative' tests.

We'll also ignore sentences that start with a close paren, as the behaviour is somewhat different on them (they can be classified as unbalanced immediately, so they don't require more complicated logic).

Define, so that the graphing works:
* **`negative_failure`**
    * This is an `(N_SAMPLES,)` boolean vector that is true for sequences whose elevation (when reading from right to left) ever dips negative, i.e. there's an open paren that is never closed.
* **`total_elevation_failure`**
    * This is an `(N_SAMPLES,)` boolean vector that is true for sequences whose total elevation is not exactly 0. In other words, for sentences with uneven numbers of open and close parens.
* **`h20_magnitudes`**
    * This is an `(N_SAMPLES,)` float vector equal to head 2.0's contribution to the position-0 residual stream in the unbalanced direction, normalized by subtracting its average unbalancedness contribution to this stream over _balanced sequences_.
* **`h21_magnitudes`**
    * Same as above but head 2.1

For the first two of these, you will find it helpful to refer back to your `is_balanced_vectorized` code (although remember you're reading **right to left** here - this will affect your `negative_failure` object). You can get the last two from your `magnitudes` tensor.

```python
if MAIN:
    negative_failure = None
    total_elevation_failure = None
    h20_magnitudes = None
    h21_magnitudes = None

    failure_types = np.full(len(h20_magnitudes), "", dtype=np.dtype("U32"))
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure
    }
    for name, mask in failure_types_dict.items():
        failure_types = np.where(mask, name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": h20_magnitudes,
        "Head 2.1 contribution": h21_magnitudes,
        "Failure type": failure_types
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, 
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4)
    fig.show()
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
def is_balanced_vectorized_return_both(tokens: TT["batch", "seq"]) -> TT["batch", t.bool]:
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens].flip(-1)
    altitude = t.cumsum(change, -1)
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = altitude.max(-1).values > 0
    return total_elevation_failure, negative_failure

if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)
    h20_magnitudes = magnitudes[7]
    h21_magnitudes = magnitudes[8]
```
""")
    
    with st.expander("Click to see the output you should be getting."):
        st.plotly_chart(fig_dict["failure_types_fig"])

    st.markdown(r"""
Look at the above graph and think about what the roles of the different heads are!""")

    with st.expander("Read after thinking for yourself"):
        st.markdown(r"""

The primary thing to take away is that 2.0 is responsible for checking the overall counts of open and close parentheses, and that 2.1 is responsible for making sure that the elevation never goes negative.

Aside: the actual story is a bit more complicated than that. Both heads will often pick up on failures that are not their responsibility, and output in the 'unbalanced' direction. This is in fact incentived by log-loss: the loss is slightly lower if both heads unanimously output 'unbalanced' on unbalanced sequences rather than if only the head 'responsible' for it does so. The heads in layer one do some logic that helps with this, although we'll not cover it today.

One way to think of it is that the heads specialized on being very reliable on their class of failures, and then sometimes will sucessfully pick up on the other type.
""")

    st.info(r"""
Note - in the code above (and several more times), we'll be using the `plotly` graphing library. This is great for interactive visualisation, but one major disadvantage is that having too many plots open tends to slow down your window. If you're having trouble with this, you can use **Clear All** if you're using VSCode's Python Interactive window, or **Clear Outputs of All Cells** if you're using Jupyter or Colab.

Alternatively, you can replace the `fig.show()` code with `fig.show(rendered="browser")`. This will open the graph in your default browser (and still allow you to interact with it), but will not slow down your window (in particular, plots with a lot of data will tend to be much more responsive than they would be in your interactive window).
""")

    st.markdown(r"""
In most of the rest of these exercises, we'll focus on the overall elevation circuit as implemented by head 2.0. As an additional way to get intuition about what head 2.0 is doing, let's graph its output against the overall proportion of the sequence that is an open-paren.

```python
if MAIN:
    fig = px.scatter(
        x=data.open_proportion, y=h20_magnitudes, color=failure_types, 
        title="Head 2.0 contribution vs proportion of open brackets '('", template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}, category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()
```""")

    with st.expander("Click to see the output you should be getting."):
        st.plotly_chart(fig_dict["failure_types_fig_2"])

    st.markdown(r"""
Think about how this fits in with your understanding of what 2.0 is doing.

---

Let's review what we've learned in this section. Click the expander below, when you've done all the exercises.
""")
    with st.expander("Review"):
        st.success(r"""
In order to understand what components of our model are causing our outputs to be correctly classified, we need to work backwards from the end of the model, and find the direction in the residual stream which leads to the largest logit difference between the unbalanced and balanced outputs. This was easy for linear layers; for layernorms we needed to approximate them as linear transforms (which it turns out is a very good approximation).

Once we've identified the direction in our residual stream which points in the "maximally unbalanced" direction, we can then look at the outputs from each of the 10 components that writes to the residual stream: our embedding (the direct path), and each of the three layers of attention heads and MLPs. We found that heads `2.0` and `2.1` were particularly important. 

We made a scatter plot of their contributions, color-coded by the type of bracket failure (there are two different ways a bracket sequence can be unbalanced). From this, we observed that head `2.0` seemed particularly effective at identifying bracket strings which had non-zero elevation (i.e. a different number of left and right brackets). In the next section, we'll dive a little deeper on how this **total elevation circuit** works.
""")

def section_3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#attention-pattern-of-the-responsible-head">Attention pattern of the responsible head</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#identifying-meaningful-direction-before-this-head">Identifying meaningful direction before this head</a></li>
       <li><a class="contents-el" href="#breaking-down-an-mlps-contribution-by-neuron">Breaking down an MLP's contribution by neuron</a></li>
   </ul></li>
   <li><a class="contents-el" href="#understanding-how-the-open-proportion-is-calculated---head-00">Understanding how the open-proportion is calculated - Head 0.0</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#00-attention-pattern">0.0 Attention Pattern</a></li>
       <li><a class="contents-el" href="#proposing-a-hypothesis">Proposing a hypothesis</a></li>
       <li><a class="contents-el" href="#the-00-ov-circuit">The 0.0 OV circuit</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Understanding the total elevation circuit

## Attention pattern of the responsible head

Which tokens is 2.0 paying attention to when the query is an open paren at token 0? Recall that we focus on sequences that start with an open paren because sequences that don't can be ruled out immediately, so more sophisticated behavior is unnecessary.

Write a function that extracts the attention patterns for a given head when run on a batch of inputs. Our code will show you the average attention pattern paid by the query for residual stream 0 when that position is an open paren.

Specifically:
* Use `get_inputs` from earlier, on the self-attention module in the layer in question.
* You can use the `attention_pattern_pre_softmax` function to get the pattern (see the `MultiheadAttention` class in `w5d5_transformer.py`), then mask the padding (elements of the batch might be different lengths, and thus be suffixed with padding).""")

    with st.expander("How do I find the padding?"):
        st.markdown(r"""
`data.toks == tokenizer.PAD_TOKEN` will give you a boolean matrix of which positions in which batch elements are padding and which aren't.""")

    st.markdown(r"""
```python
def get_attn_probs(model: ParenTransformer, tokenizer: SimpleTokenizer, data: DataSet, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    pass

if MAIN:
    attn_probs = get_attn_probs(model, tokenizer, data, 2, 0)
    attn_probs_open = attn_probs[data.starts_open].mean(0)[[0]]
    px.bar(
        y=attn_probs_open.squeeze().numpy(), labels={"y": "Probability", "x": "Key Position"},
        template="simple_white", height=500, width=600, title="Avg Attention Probabilities for '(' query from query 0"
    ).update_layout(showlegend=False, hovermode='x unified').show()
```

You should see an average attention of around 0.5 on position 1, and an average of about 0 for all other tokens. So `2.0` is just copying information from residual stream 1 to residual stream 0. In other words, `2.0` passes residual stream 1 through its `W_OV` circuit (after `LayerNorm`ing, of course), weighted by some amount which we'll pretend is constant. The plot thickens. Now we can ask, "What is the direction in residual stream 1 that, when passed through `2.0`'s `W_OV`, creates a vector in the unbalanced direction in residual stream 0?"

### Identifying meaningful direction before this head

Previously, we looked at the vector each component wrote to the residual stream at sequence position 0 (remember the diagram with ten dotted arrows, each one going directly from one of the components to the residual stream). Now that we've observed head `2.0` is mainly copying information from sequence position 1 to position 0, we want to understand how each of the 7 components **before** head `2.0` contributes to the unbalanced direction **via its path through head `2.0`**.

Here is an annotated diagram to help better explain exactly what we're doing.""")

    st_image("true_images/bracket_transformer-elevation-circuit-1.png", 1000)

    st.markdown(r""" 
Below, you'll be asked to calculate this `pre_20_dir`, which is the unbalanced direction for inputs into head 2.0 at sequence position 1 (based on the fact that vectors at this sequence position are copied to position 0 by head `2.0`, and then used in prediction).

First, you'll implement the function `get_WOV`, to get the OV matrix for a particular layer and head. Recall that this is the product of the `W_O` and `W_V` matrices. Then, you'll use this function to write `get_pre_20_dir`.

```python
def get_WOV(model: HookedTransformer, layer: int, head: int) -> TT["d_model", "d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    pass


def get_pre_20_dir(model, data) -> TT["d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    '''
    pass


if MAIN:
    w5d5_tests.test_get_pre_20_dir(model, data, get_pre_20_dir)
```""")

    with st.expander("Help - I can't remember what W_OV should be."):
            st.markdown(r"""
    Recall that we're adopting the left-multiply convention. So if `x` is our vector in the residual stream (with length `d_model`), then `x @ W_V` is the vector of values (with length `d_head`), and `x @ W_V @ W_O` is the vector that gets moved from source to destination if `x` is attended to.

    So we have `W_OV = W_V @ W_O`, and the vector that gets moved from position 1 to position 0 by head `2.0` is `x @ W_OV`.""")

    st.markdown(r"""
Now that you've got the `pre_20_dir`, you can calculate magnitudes for each of the components that came before

```python
if MAIN:
    # YOUR CODE HERE
    # Define `magnitudes` (as before, but now in the `pre_20_dir` direction, for all components before head 2.0)

    hists_per_comp(magnitudes, data, n_layers=2, xaxis_range=(-5, 12))
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
if MAIN:
    pre_layer2_outputs = get_out_by_components(model, data)[:-3]
    magnitudes = einsum(
        "comp batch emb, emb -> comp batch",
        pre_layer2_outputs[:, :, 1, :],
        get_pre_20_dir(model, data)
    )
    magnitudes_mean_for_each_comp: TT["comp", 1] = magnitudes[:, data.isbal].mean(-1, keepdim=True)
    magnitudes -= magnitudes_mean_for_each_comp
    hists_per_comp(magnitudes, data, xaxis_range=(-5, 12))
```""")

    with st.expander("Click here to see the output you should be getting."):
        st.plotly_chart(fig_dict["attribution_fig_2"])

    st.markdown(r"""
What do you observe?""")

    with st.expander("Some things to notice"):
        st.markdown(r"""
One obvious note - the embeddings graph shows an output of zero, in other words no effect on the classification. This is because the input for this path is just the embedding vector in the 0th sequence position - in other words the `[START]` token's embedding, which is the same for all inputs.

---

More interestingly, we can see that `mlp0` and especially `mlp1` are very important. This makes sense -- one thing that mlps are especially capable of doing is turning more continuous features ('what proportion of characters in this input are open parens?') into sharp discontinuous features ('is that proportion exactly 0.5?').

For example, the sum $\operatorname{ReLU}(x-0.5) + \operatorname{ReLU}(0.5-x)$ evaluates to the nonlinear function $2 \times |x-0.5|$, which is zero if and only if $x=0.5$. This is one way our model might be able to classify all bracket strings as unbalanced unless they had exactly 50% open parens.""")

        st_image("true_images/relu2-light.png", 550)

        # st.markdown(r"*We can even add together more ReLUs to get even sharper discontinuities or more complex functions. For instance:*")
        # st_excalidraw("relu", 600)

        st.markdown(r"""
---

Head `1.1` also has some importance, although we will not be able to dig into this today. It turns out that one of the main things it does is incorporate information about when there is a negative elevation failure into this overall elevation branch. This allows the heads to agree the prompt is unbalanced when it is obviously so, even if the overall count of opens and closes would allow it to be balanced.
""")

    st.markdown(r"""
In order to get a better look at what `mlp0` and `mlp1` are doing more thoughly, we can look at their output as a function of the overall open-proportion.

```python
def mlp_attribution_scatter(magnitudes, data, failure_types):
    for layer in range(2):
        fig = px.scatter(
            x=data.open_proportion[data.starts_open], y=magnitudes[3+layer*3, data.starts_open], 
            color=failure_types[data.starts_open], category_orders={"color": failure_types_dict.keys()},
            title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
            template="simple_white", height=500, width=800,
            labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
        ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
        fig.show()

if MAIN:
    mlp_attribution_scatter(magnitudes, data, failure_types)
```

### Breaking down an MLP's contribution by neuron

We've already learned that an attention layer can be broken down as a sum of separate contributions from each head. It turns out that we can do something similar with MLPs, breaking them down as a sum of per-neuron contributions.

Ignoring biases, let $MLP(\vec x) = f(\vec x^T W^{in})^T W^{out}$ for matrices $W^{in}, W^{out}$, and $f$ is our nonlinear activation function (in this case ReLU). Note that $f(\vec x^T W^{in})$ is what we refer to as the **neuron activations**, let $n$ be its length (the intermediate size of the MLP).

(Note - when I write $f(z)$ for a vector $z$, this means the vector with $f(z)_i = f(z_i)$, i.e. we're applying the activation function elementwise.)

**Exercise: write $MLP$ as a sum of $n$ functions of $\vec x$**.
""")

    with st.expander("Answer and discussion"):

        st.markdown(r"""
Firstly, remember that MLPs act exactly the same on each sequence position, so we can ignore the sequence dimension and treat the MLP as a map from vectors $\vec x$ of length `emb_dim` to vectors which also have length `emb_dim` (and which are written directly into the residual stream).

One way to conceptualize the matrix-vector multiplication $\vec y^T V$ is as a weighted sum of the rows of $V$:

$$
V = \left[\begin{array}{c}
V_{[0,:]} \\
\overline{\quad\quad\quad} \\
V_{[1,:]} \\
\overline{\quad\quad\quad} \\
\ldots \\
\overline{\quad\quad\quad} \\
V_{[n-1,:]}
\end{array}\right], \quad  \vec y^T V = y_0 V_{[0, :]} + ... + y_{n-1} V_{[n-1, :]}
$$

Taking $y$ to be our **neuron activations** $f(\vec x^T W^{in})$, and $V$ to be our matrix $W^{out}$, we can write:

$$
MLP(\vec x) = \sum_{i=0}^{n-1}f(\vec x^T W^{in})_i W^{out}_{[i,:]}
$$

where $f(\vec x^T W^{in})_i$ is a scalar, and $A_{[;,i]}$ is a vector.

But we can actually simplify further, as $f(\vec x^T W^{in})_i = f(\vec x^T W^{in}_{[:, i]})$; i.e. the dot product of $\vec x$ and the $i$-th column of $W_{in}$ (and not the rest of $W_{in}$!). This is because:

$$
V = \left[V_{[:,0]} \;\bigg|\; V_{[:,1]} \;\bigg|\; ... \;\bigg|\; V_{[:,n-1]}\right], \quad \vec y^T V = \left[\vec y^T V_{[:,0]} \;\bigg|\; \vec y^T V_{[:,1]} \;\bigg|\; ... \;\bigg|\; \vec y^T V_{[:,n-1]}\right]
$$

and because $f$ acts on each element of its input vector independently.

Thus, we can write:

$$
MLP(\vec x) = \sum_{i=0}^{n-1}f(\vec x^T W^{in}_{[:, i]}) W^{out}_{[i,:]}
$$
or if we include biases on the Linear layers:

$$
MLP(\vec x) = b^{out} + \sum_{i=0}^{n-1}f(\vec x^T W^{in}_{[:, i]} + b^{in}_i) W^{out}_{[i,:]}
$$

This is a neat-enough equation that Buck jokes he's going to get it tattooed on his arm!

Summary:
""")

        st.success(r"""
We can write an MLP as a collection of neurons, where each one writes a vector to the residual stream independently of the others.

We can view the $i$-th column of $W^{in}$ as being the **"in-direction"** of neuron $i$, as the activation of neuron $i$ depends on how high the dot product between $x$ and that row is. And then we can think of the $i$-th row of $W^{out}$ as the **"out-direction"** signifying neuron $i$'s special output vector, which it scales by its activation and then adds into the residual stream.""")

        st.markdown(r"""
Here are some examples of what this could look like (feel free to skip if you're satisfied with this section):

* If $\vec x$ is orthogonal to the column $W^{in}_{[:, i]}$, i.e. $\vec x^T W^{in}_{[:, i]} = 0$, then the $i$-th neuron's output is independent of the input $\vec x$, i.e. it doesn't move any information through the MLP.
    * If $\vec x$ is orthogonal to all the in-directions $W^{in}_i$, then all the activations are the same, and no information is moved through the MLP.
* If $\vec x$ is not orthogonal to an input direction $W^{in}_{[:, i]}$, then this neuron will write some scalar multiple of $W^{out}_{[i, :]}$ to the residual stream.
    * The scalar multiple of $W^{out}_{[i, :]}$ depends on the size of the component of $\vec x$ in the $W^{in}_{[:, i]}$-direction.

Interestingly, there is some evidence that certain neurons in MLPs perform memory management. For instance, we might find that the $i$-th neuron satisfies $W^{in}_{[:, i]} \approx - W^{out}_{[i, :]} \approx \vec v$ for some unit vector $\vec v$, meaning it may be responsible for erasing the component of vector $\vec x$ in the direction $\vec v$ (exercise - can you show why this is the case?). This can free up space in the residual stream for other components to write to.

""")

    st.markdown(r"""
```python
def out_by_neuron(model, data, layer):
    '''
    Return shape: [len(data), seq_len, neurons, out]
    '''
    pass

@functools.cache
def out_by_neuron_in_20_dir(model, data, layer):
    pass
```

Now, try to identify several individual neurons that are especially important to `2.0`.

For instance, you can do this by seeing which neurons have the largest difference between how much they write in our chosen direction on balanced and unbalanced sequences (especially unbalanced sequences beginning with an open paren).

Use the `plot_neurons` function to get a sense of what an individual neuron does on differen open-proportions.

One note: now that we are deep in the internals of the network, our assumption that a single direction captures most of the meaningful things going on in this overall-elevation circuit is highly questionable. This is especially true for using our `2.0` direction to analyize the output of `mlp0`, as one of the main ways this mlp has influence is through more indirect paths (such as `mlp0 -> mlp1 -> 2.0`) which are not the ones we chose our direction to capture. Thus, it is good to be aware that the intuitions you get about what different layers or neurons are doing are likely to be incomplete.

```python
def plot_neurons(model: HookedTransformer, data: BracketsDataset, failure_types: np.ndarray, layer: int):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)[data.starts_open, 1, :].detach()

    # Get data that can be turned into a dataframe (plotly express is sometimes easier to use with a dataframe)
    # Plot a scatter plot of all the neuron contributions, color-coded according to failure type, with slider to view neurons
    neuron_numbers = einops.repeat(t.arange(model.cfg.d_model), "n -> (s n)", s=data.starts_open.sum())
    failure_types = einops.repeat(failure_types[data.starts_open], "s -> (s n)", n=model.cfg.d_model)
    data_open_proportion = einops.repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.cfg.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": neurons_in_d.flatten(),
        "Neuron number": neuron_numbers,
        "Open-proportion": data_open_proportion,
        "Failure type": failure_types
    })
    px.scatter(
        df, 
        x="Open-proportion", y="Output in 2.0 direction", color="Failure type", animation_frame="Neuron number",
        title=f"Neuron contributions from layer {layer}", 
        template="simple_white", height=500, width=800
    ).update_traces(marker_size=3).update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5]).show(renderer="browser")

    # Work out the importance (average difference in unbalanced contribution between balanced and inbalanced dirs) for each neuron
    # Plot a bar chart of the per-neuron importances
    neuron_importances = neurons_in_d[~data.isbal[data.starts_open]].mean(0) - neurons_in_d[data.isbal[data.starts_open]].mean(0)
    px.bar(
        x=t.arange(model.cfg.d_model), 
        y=neuron_importances, 
        title=f"Importance of neurons in layer {layer}", 
        labels={"x": "Neuron number", "y": "Mean contribution in unbalanced dir"},
        template="simple_white", height=400, width=600
    ).update_layout(hovermode="x unified").show(renderer="browser")

if MAIN:
    for layer in range(2):
        plot_neurons(model, data, failure_types, layer)
```
""")

    with st.expander("Some observations:"):
        st.markdown(r"""
The important neurons in layer 1 can be put into three broad categories:

- Some neurons detect when the open-proprtion is greater than 1/2. As a few examples, look at neurons **`1.53`**, **`1.39`**, **`1.8`** in layer 1. There are some in layer 0 as well, such as **`0.33`** or **`0.43`**. Overall these seem more common in Layer 1.

- Some neurons detect when the open-proprtion is less than 1/2. For instance, neurons **`0.21`**, and **`0.7`**. These are much more rare in layer 1, but you can see some such as **`1.50`** and **`1.6`**.

- The network could just use these two types of neurons, and compose them to measure if the open-proportion exactly equals 1/2 by adding them together. But we also see in layer 1 that there are many neurons that output this composed property. As a few examples, look at **`1.10`** and **`1.3`**. 
    - It's much harder for a single neuron in layer 0 to do this by themselves, given that ReLU is monotonic and it requires the output to be a non-monotonic function of the open-paren proportion. It is possible, however, to take advantage of the layernorm before **`mlp0`** to approximate this -- **`0.19`** and **`0.34`** are good examples of this.

---

Below: plots of neurons **`1.53`** and **`0.21`**. You can observe the patterns described above.""")
        # cols = st.columns([1, 10, 1, 10, 1])
        # with cols[1]:
        st_image("n53.png", 550)
        st.markdown("")
        st_image("n21.png", 550)
        st.markdown("")
        # with cols[-2]:

    st.markdown(r"""
## Understanding how the open-proportion is calculated - Head 0.0

Up to this point we've been working backwards from the logits and through the internals of the network. We'll now change tactics somewhat, and start working from the input embeddings forwards. In particular, we want to understand how the network calcuates the open-proportion of the sequence in the first place!

The key will end up being head 0.0. Let's start by examining its attention pattern.

### 0.0 Attention Pattern

We want to play around with the attention patterns in our heads. For instance, we'd like to ask questions like "what do the attention patterns look like when the queries are always left-parens?". To do this, we'll write a function that takes in a parens string, and returns the `q` and `k` vectors (i.e. the values which we take the inner product of to get the attention scores).

*Note - this is another messing-around-with-hooks function, and is less conceptually valuable than the other exercises. If you want to skip it, you can use the solutions directly and move on to the code after this.*

```python
def get_q_and_k_for_given_input(
    model: HookedTransformer, parens: str, layer: int, head: int
) -> Tuple[TT["seq", "d_model"], TT[ "seq", "d_model"]]:
    '''
    Returns the queries and keys (both of shape [seq, d_model]) for the given parns input, in the attention head `layer.head`.
    '''
    pass
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
def get_q_and_k_for_given_input(
    model: HookedTransformer, parens: str, layer: int, head: int
) -> Tuple[TT["seq", "d_model"], TT[ "seq", "d_model"]]:
    '''
    Returns the queries and keys (both of shape [seq, d_model]) for the given parns input, in the attention head `layer.head`.
    '''
    # Create lists to store the queries and keys
    q_inputs = []
    k_inputs = []

    # Run the model with hooks to store the queries and keys
    model.run_with_hooks(
        tokenizer.tokenize(parens),
        model.run_with_hooks,
        fwd_hooks=[
            (utils.get_act_name("q", layer), lambda q, hook: q_inputs.append(q[:, :, head, :])),
            (utils.get_act_name("k", layer), lambda k, hook: k_inputs.append(k[:, :, head, :])),
        ]
    )

    # Return the queries and keys
    return q_inputs[0][0], k_inputs[0][0]
```
""")

    st.markdown(r"""
Now that we have this function, we will use it to find the attention pattern in head `0.0` when `q` is supplied by a sequence of all left-parens, and `k` is the average of its value with all left parens and all right parens. Note that in some sense this is dishonest, since `q` and `k` will always be determined by the same input sequence. But what we're doing here should serve as a reasonably good indicator for how left-parens attend to other parens in the sequence in head `0.0`.

```python
if MAIN:

    all_left_parens = "".join(["(" * 40])
    all_right_parens = "".join([")" * 40])
    model.reset_hooks()
    q00_all_left, k00_all_left = get_q_and_k_for_given_input(model, all_left_parens, 0, 0)
    q00_all_right, k00_all_right = get_q_and_k_for_given_input(model, all_right_parens, 0, 0)
    k00_avg = (k00_all_left + k00_all_right) / 2

    # Define hook function to patch in q or k vectors
    def hook_fn_patch_qk(
        value: TT["batch", "seq", "head", "d_head"], 
        hook: HookPoint, 
        new_value: TT[..., "seq", "d_head"],
        head_idx: int = 0
    ) -> None:
        value[..., head_idx, :] = new_value
    
    # Define hook function to display attention patterns (using plotly)
    def hook_fn_display_attn_patterns(
        pattern: TT["batch", "heads", "seqQ", "seqK"],
        hook: HookPoint,
        head_idx: int = 0
    ) -> None:
        px.imshow(
            pattern[0, head_idx], 
            title="Estimate for avg attn probabilities when query is from '('",
            labels={"x": "Key tokens", "y": "Query tokens"},
            height=1200, width=1200,
            color_continuous_scale="RdBu_r", range_color=[0, pattern[0, head_idx].max().item()]
        ).update_layout(
            xaxis = dict(
                tickmode = "array", ticktext = ["[start]", *["L+R/2" for i in range(40)], "[end]"],
                tickvals = list(range(42)), tickangle = 45,
            ),
            yaxis = dict(
                tickmode = "array", ticktext = ["[start]", *["L" for i in range(40)], "[end]"],
                tickvals = list(range(42)), 
            ),
        ).show(renderer="browser")
    
    # Run our model on left parens, but patch in the average key values for left vs right parens
    # This is to give us an idea of how the model behaves on average when the query is a left paren
    model.run_with_hooks(
        tokenizer.tokenize(all_left_parens),
        model.run_with_hooks,
        fwd_hooks=[
            (utils.get_act_name("k", 0), functools.partial(hook_fn_patch_qk, new_value=k00_avg)),
            (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns),
        ]
    )
```""")

    with st.expander("Click here to see the output you should be getting."):
        st.plotly_chart(fig_dict["true_images/attn_probs_red"], use_container_width=True)

    with st.expander("Question - what are the noteworthy features of this plot?"):
        st.markdown(r"""
The most noticeable feature is the diagonal pattern - each query token pays almost zero attention to all the tokens that come before it, but much greater attention to those that come after it. For most query token positions, this attention paid to tokens after itself is roughly uniform. However, there are a few patches (especially for later query positions) where the attention paid to tokens after itself is not uniform. We will see that these patches are important for generating adversarial examples.

Incidentally, we can also observe roughly the same pattern when the query is a right paren (try running the last bit of code above, but using `all_right_parens` instead of `all_left_parens`), but the pattern is less pronounced.
""")

    st.markdown(r"""
We are most interested in the attention pattern at query position 1, because this is the position we move information to that is eventually fed into attention head `2.0`, then moved to position 0 and used for prediction.

(Note - we've chosen to focus on the scenario when the first paren is an open paren, because the model actually deals with bracket strings that open with a right paren slightly differently - these are obviously unbalanced, so a complicated mechanism is unnecessary.)

Let's plot a bar chart of the attention probability paid by the the open-paren query at position 1 to all the other positions. Here, rather than making the query and key artificial, we're running the model on our entire dataset and patching in an artificial value for the query (all open parens). Both methods are reasonable in this case, since we're just looking for a general sense of how our query vector at position 1 behaves when it's an open paren.

```python
if MAIN:

    def hook_fn_display_attn_patterns_for_single_query(
        pattern: TT["batch", "heads", "seqQ", "seqK"],
        hook: HookPoint,
        head_idx: int = 0,
        query_idx: int = 1
    ):
        px.bar(
            pattern[:, head_idx, query_idx].mean(0), 
            title=f"Average attn probabilities on data at posn 1, with query token = '('",
            labels={"index": "Sequence position of key", "value": "Average attn over dataset"}, 
            template="simple_white", height=500, width=800
        ).update_layout(showlegend=False, margin_l=100, yaxis_range=[0, 0.1], hovermode="x unified").show()

    data_len_40 = BracketsDataset.with_length(data_tuples, 40)

    model.reset_hooks()
    model.run_with_hooks(
        data_len_40.toks[data_len_40.isbal],
        return_type=None,
        fwd_hooks=[
            (utils.get_act_name("q", 0), functools.partial(hook_fn_patch_qk, new_value=q00_all_left)),
            (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns_for_single_query),
        ]
    )
```""")

    with st.expander("Click here to see the output you should be getting."):
        st.plotly_chart(fig_dict["true_images/attn_qpos1"], use_container_width=True)

    with st.expander("Question - what is the interpretation of this attention pattern? (i.e. what is the nature of the computation happening in this attention head, at this query position?)"):
        st.markdown(r"""
This shows that the attention pattern is almost exactly uniform over all tokens. This means the vector written to sequence position 1 will be approximately some scalar multiple of the vectors at each source position, mapped through the matrix $W_{OV}^{0.0}$.

Note - you can also check this for `data_len_n = DataSet.with_length(data_tuples, n)` for values of `n` other than 40, and verify that this attention pattern still basically holds up.
""")

    st.markdown(r"""
### Proposing a hypothesis

Before we connect all the pieces together, let's list the facts that we know about our model so far (going chronologically from our observations):""")

    st.info(r"""
* Attention head `2.0` seems to be largely responsible for classifying brackets as unbalanced when they have non-zero net elevation (i.e. have a different number of left and right parens).
    * Attention head `2.0` attends strongly to the sequence position $i=1$, in other words it's pretty much just moving the residual stream vector from position 1 to position 0 (and applying matrix $W_{OV}$).
    * So there must be earlier components of the model which write to sequence position 1, in a way which influences the model to make correct classifications (via the path through head `2.0`).
* There are several neurons in **`MLP0`** and **`MLP1`** which seem to calculate a nonlinear function of the open parens proportion - some of them are strongly activating when the proportion is strictly greater than $1/2$, others when it is strictly smaller than $1/2$.
* If the query token in attention head `0.0` is an open paren, then it attends to all key positions **after** $i$ with roughly equal magnitude.
    * In particular, this holds for the sequence position $i=1$, which attends approximately uniformly to all sequence positions.
 
""")

    st.markdown(r"""
Based on all this, can you formulate a hypothesis for how the elevation circuit works, which ties all three of these observations together?""")

    with st.expander("Hypothesis"):
        st.markdown("The hypothesis might go something like this:")
        st.success(r"""

1. **In the attention calculation for head `0.0`, the position-1 query token is doing some kind of aggregation over brackets. It writes to the residual stream information representing the difference between the number of left and right brackets - in other words, the net elevation.**

Remember that one-layer attention heads can pretty much only do skip-trigrams, e.g. of the form `keep ... in -> mind`. They can't capture three-way interactions flexibly, in other words they can't compute functions like "whether the number of left and right brackets is equal". So aggregation over left and right brackets is pretty much all we can do.

2. **Now that sequence position 1 contains information about the elevation, the MLP reads this information, and some of its neurons perform nonlinear operations to give us a vector which conatains "boolean" information about whether the number of left and right brackets is equal.**

Recall the example given earlier of $\operatorname{ReLU}(x-0.5) + \operatorname{ReLU}(0.5-x)$; we could guess that some of the neurons are taking each of these roles (in fact we saw it! - should make this clearer; add a diagram adding two relus).**

3. **Finally, now that the 1st sequence position in the residual stream stores boolean information about whether the net elevation is zero, this information is read by head `2.0`, and the output of this head is used to classify the sequence as balanced or unbalanced.**

This is based on the fact that we already saw head `2.0` is strongly attending to the 1st sequence position, and that it seems to be implementing the elevation test.
""")
        st.markdown(r"""At this point, we've pretty much empirically verified all the observations above. One thing we haven't really proven yet is that **(1)** is working as we've described above. We want to verify that head `0.0` is calculating some kind of difference between the number of left and right brackets, and writing this information to the residual stream. In the next section, we'll find a way to test this hypothesis.""")

# h(x) &= \left(\left(\,A\, \otimes \,W_{OV}\,\right) \cdot L x\right)_1 \\
# &= \left(\frac{1}{n}, \frac{1}{n}, ..., \frac{1}{n}\right)^T \left(Lx\right)^T W_{OV} \\
# &= \frac{1}{n} \sum_{i=1}^n \left(Lx\right)_i^T W_{OV} \\
    st.markdown(r"""

### The 0.0 OV circuit

**We want to understand what the `0.0` head is writing to the residual stream. In particular, we are looking for evidence that it is writing information about the net elevation.**

We've already seen that query position 1 is attending approximately uniformly to all key positions. This means that (ignoring start and end tokens) the vector written to position 1 is approximately:

$$
\begin{aligned}
h(x) &\approx \frac{1}{n} \sum_{i=1}^n \left(\left(L x\right)^T W_{OV}^{0.0}\right)_i \\
&= \frac{1}{n} \sum_{i=1}^n {\color{orange}x}_i^T L^T W_{OV}^{0.0} \\
\end{aligned}
$$

where $L$ is the linear approximation for the layernorm before the first attention layer, and $x$ is the `(seq_len, d_model)`-size residual stream consisting of vectors ${\color{orange}x}_i$ for each sequence position $i$.

We can write $x_j = {\color{orange}pos}_j + {\color{orange}tok}_j$, where ${\color{orange}pos}_j$ and ${\color{orange}tok}_j$ stand for the positional and token embeddings respectively. So this gives us:

$$
\begin{aligned}
h(x) &\approx \frac{1}{n} \left( \sum_{i=1}^n {\color{orange}pos}_i^T L^T W_{OV}^{0.0} + \sum_{i=1}^n {\color{orange}tok}_i^T L^T W_{OV}^{0.0})\right) \\
&= \frac{1}{n} \left( \sum_{i=1}^n {\color{orange}pos}_i^T L^T W_{OV}^{0.0} + n_L \boldsymbol{\color{orange}\vec v_L} + n_R \boldsymbol{\color{orange}\vec v_R}\right)
\end{aligned}
$$

where $n_L$ and $n_R$ are the number of left and right brackets respectively, and $\boldsymbol{\color{orange}\vec v_L}, \boldsymbol{\color{orange}\vec v_R}$ are the images of the token embeddings for left and right parens respectively under the image of the layernorm and OV circuit:

$$
\begin{aligned}
\boldsymbol{\color{orange}\vec v_L} &= {\color{orange}LeftParen}^T L^T W_{OV}^{0.0} \\
\boldsymbol{\color{orange}\vec v_R} &= {\color{orange}RightParen}^T L^T W_{OV}^{0.0}
\end{aligned}
$$

where ${\color{orange}LeftParen}$ and ${\color{orange}RightParen}$ are the token embeddings for left and right parens respectively.

Finally, we have an ability to formulate a test for our hypothesis in terms of the expression above:""")

    st.info(r"""
If head `0.0` is performing some kind of aggregation, then **we should see that $\boldsymbol{\color{orange}\vec v_L}$ and $\boldsymbol{\color{orange}\vec v_R}$ are vectors pointing in opposite directions.** In other words, head `0.0` writes some scalar multiple of vector $v$ to the residual stream, and we can extract the information $n_L - n_R$ by projecting in the direction of this vector. The MLP can then take this information and process it in a nonlinear way, writing information about whether the sequence is balanced to the residual stream.
""")
    # Note - you may find that these two vectors don't have similar magnitudes, so rather than storing the information $n_L - n_R$, it would be more accurate to say the information being stored is $n_L - \alpha n_R$, where $\alpha$ is some scalar (not necessarily $1$). However, this isn't really an issue for our interpretation of the model, because:

    # 1. It's very unclear how the layernorm affects the input magnitudes.
    # 2. There are ways we could imagine the model getting around the magnitude problem (e.g. by using information about the total length of the bracket string, which it does in a sense have access to).
    st.markdown(r"""
**Exercise - show that $\boldsymbol{\color{orange}\vec v_L}$ and $\boldsymbol{\color{orange}\vec v_R}$ do indeed have opposite directions (i.e. cosine similarity close to -1), demonstrating that this head is "tallying" the open and close parens that come after it.**

You can fill in the function `embedding` (to return the token embedding vector corresponding to a particular character), which will help when computing these vectors.

```python
def embedding(model: HookedTransformer, tokenizer: SimpleTokenizer, char: str) -> TT["d_model"]:
    assert char in ("(", ")")
    pass

if MAIN:
    "YOUR CODE HERE: define v_L and v_R, as described above."
    
    print("Cosine similarity: ", t.cosine_similarity(v_L, v_R, dim=0).item())
```""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
def embedding(model: HookedTransformer, tokenizer: SimpleTokenizer, char: str) -> TT["d_model"]:
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model.W_E[idx]

if MAIN:

    W_OV = model.W_V[0, 0] @ model.W_O[0, 0]

    layer0_ln_fit = get_ln_fit(model, data, layer=0, layernorm=1, seq_pos=None)[0]
    layer0_ln_coefs = t.from_numpy(layer0_ln_fit.coef_).to(device)

    v_L = embedding(model, tokenizer, "(") @ layer0_ln_coefs.T @ W_OV
    v_R = embedding(model, tokenizer, ")") @ layer0_ln_coefs.T @ W_OV

    print("Cosine similarity: ", t.cosine_similarity(v_L, v_R, dim=0).item())
```
""")
    st.markdown(r"""

Another way we can get evidence for this hypothesis - recall in our discussion of MLP neurons that $W^{in}_{[:,i]}$ (the $i$th column of matrix $W^{in}$, where $W^{in}$ is the first linear layer of the MLP) is a vector representing the "in-direction" of the neuron. If these neurons are indeed measuring open/closed proportions in the way we think, then we should expect to see the vectors $v_R$, $v_L$ have high dot product with these vectors.

Investigate this by filling in the two functions below. `cos_sim_with_MLP_weights` returns the vector of cosine similarities between a vector and the columns of $W^{in}$ for a given layer, and `avg_squared_cos_sim` returns the average **squared cosine similarity** between a vector $v$ and a randomly chosen vector with the same size as $v$ (we can choose this vector in any sensible way, e.g. sampling it from the iid normal distribution then normalizing it). You should find that the average squared cosine similarity per neuron between $v_R$ and the in-directions for neurons in `MLP0` and `MLP1` is much higher than you would expect by chance.

```python
def cos_sim_with_MLP_weights(model: HookedTransformer, v: TT["d_model"], layer: int) -> TT["d_hidden"]:
    '''
    Returns a vector of length d_hidden, where the ith element is the
    cosine similarity between `v` and the ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    pass


def avg_squared_cos_sim(v: TT["d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between `v` and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    pass


if MAIN:
    print("Avg squared cosine similarity of v_R with ...\n")

    cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
    print(f"...MLP input directions in layer 0:  {cos_sim_mlp0.pow(2).mean():.6f}")
   
    cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
    print(f"...MLP input directions in layer 1:  {cos_sim_mlp1.pow(2).mean():.6f}")
    
    cos_sim_rand = avg_squared_cos_sim(v_R)
    print(f"...random vectors of len = d_model:  {cos_sim_rand:.6f}")
""")

    with st.expander("Solution"):
        st.markdown(r"""
```python
def cos_sim_with_MLP_weights(model: HookedTransformer, v: TT["d_model"], layer: int) -> TT["d_hidden"]:
    '''
    Returns a vector of length d_hidden, where the ith element is the
    cosine similarity between v and the ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    '''
    v_unit = v / v.norm()
    W_in_unit = model.blocks[layer].mlp.W_in / model.blocks[layer].mlp.W_in.norm(dim=0)

    return einsum("d_model, d_model d_hidden -> d_hidden", v_unit, W_in_unit)


def avg_squared_cos_sim(v: TT["d_model"], n_samples: int = 1000) -> float:
    '''
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    '''
    v2 = t.randn(n_samples, v.shape[0])
    v2 /= v2.norm(dim=1, keepdim=True)

    v1 = v / v.norm()

    return (v1 * v2).pow(2).sum(1).mean().item()
""")

    st.markdown(r"""

As a bonus, you can also compare the squared cosine similarities per neuron to your neuron contribution plots you made earlier (the ones with sliders). Do the neurons which have particularly high cosine similarity with $v_R$ correspond to the neurons which write to the unbalanced direction of head `2.0` in a big way whenever the proportion of open parens is not 0.5? (This would provide further evidence that the main source of information about total open proportion of brackets which is used in the net elevation circuit is provided by the multiples of $v_R$ and $v_L$ written to the residual stream by head `0.0`). You can go back to your old plots and check.

---

Great! Let's stop and take stock of what we've learned about this circuit. Head 0.0 pays attention uniformly to the suffix following each token, tallying up the amount of open and close parens that it sees and writing that value to the residual stream. This means that it writes a vector representing the total elevation to residual stream 1. The MLPs in residual stream 1 then operate nonlinearly on this tally, writing vectors to the residual stream that distinguish between the cases of zero and non-zero total elevation. Head 2.0 copies this signal to residual stream 0, where it then goes through the classifier and leads to a classification as unbalanced. Our first-pass understanding of this behavior is complete.

An illustration of this circuit is given below. It's pretty complicated with a lot of moving parts, so don't worry if you don't follow all of it!

Key: the thick black lines and orange dotted lines show the paths through our transformer constituting the elevation circuit. The orange dotted lines indicate the skip connections. Each of the important heads and MLP layers are coloured bold. The three important parts of our circuit (head `0.0`, the MLP layers, and head `2.0`) are all give annotations explaining what they're doing, and the evidence we found for this.
""")
    st_image("true_images/bracket-transformer-attribution.png", 1200)
def section_4():
    st.markdown(r"""
# Bonus exercises

## Dealing with early closing parens

We mentioned that our model deals with early closing parens differently. One of our components in particular is responsible for classifying any sequence that starts with a closed paren as unbalnced - can you find the component that does this? """)

    with st.expander("Hint"):
        st.markdown(r"""
It'll have to be one of the attention heads, since these are the only things which can move information from sequence position 1 to position 0. Which of your attention heads was previously observed to move information from position 1 to position 0?
""")

    st.markdown(r"""
Can you prove that this component is responsible for this behavior?

## Detecting anywhere-negative failures

When we looked at our grid of attention patterns, we saw that not only did the first query token pay approximately uniform attention to all tokens following it, but so did most of the other tokens (to lesser degrees). This means that we can write the vector written to position $i$ (for general $i\geq 1$) as:

$$
\begin{aligned}
h(x)_i &\approx \frac{1}{n-i+1} \sum_{j=i}^n x_j^T L^T W_{OV}^{0.0} \\
&= \frac{1}{n} \left( \sum_{i=1}^n {\color{red}pos}_i^T L^T W_{OV}^{0.0} + n_L^i \boldsymbol{\color{red}\vec v_L} + n_R^i \boldsymbol{\color{red}\vec v_R}\right)
\end{aligned}
$$

where $n_L^i$ and $n_R^i$ are the number of left and right brackets respectively in the substring $x_i \dots x_n$ (i.e. this matches our definition of $n_L$ and $n_R$ when $i=1$).

Given what we've seen so far (that sequence position 1 stores tally information for all the brackets in the sequence), we can guess that each sequence position stores a similar tally, and is used to determine whether the substring consisting of all brackets to the right of this one has any elevation failures (i.e. making sure the total number of ***right*** brackets is at least as great as the total number of ***left*** brackets - recall it's this way around because our model learned the equally valid right-to-left solution).

Recall that the destination token only determines how much to pay attention to the source; the vector that is moved from the source to destination conditional on attention being paid to it is the same for all destination tokens. So the result about left-paren and right-paren vectors having cosine similarity of -1 also holds for all later sequence positions.

**Head 2.1 turns out to be the head for detecting anywhere-negative failures** (i.e. it  detects whether any sequence $x_i, ..., x_n$ has strictly more right than left parentheses, and writes to the residual stream in the unbalanced direction if this is the case). Can you find evidence for this behaviour?

One way you could investigate this is to construct a parens string which "goes negative" at some points, and look at the attention probabilities for head 2.0 at destination position 0. Does it attend most strongly to those source tokens where the bracket goes negative, and is the corresponding vector written to the residual stream one which points in the unbalanced direction?

You could also look at the inputs to head 2.1, just like we did for head 2.0. Which components are most important, and can you guess why?
""")

    with st.expander("Spoiler"):
        st.markdown(r"""
You should find that the MLPs are important inputs into head 2.1. This makes sense, because the MLPs' job is to convert tally information ($n_L^i - n_R^i$) into boolean information ($n_L^i > n_R^i$). This is a convenient form for our head 2.1 to read (since it's detecting any negative elevations).
""")

    st.markdown(r"""
## Adversarial attacks

Our model gets around 1 in a ten thousand examples wrong on the dataset we've been using. Armed with our understanding of the model, can we find a misclassified input by hand? I recommend stopping reading now and trying your hand at applying what you've learned so far to find a misclassified sequence. If this doesn't work, look at a few hints.
""")

    with st.expander("Hint 1"):
        st.markdown(r"""
What's up with those weird patchy bits in the bottom-right corner of the attention patterns? Can we exploit this?

Read the next hint for some more specific directions.
""")

    with st.expander("Hint 2"):
        st.markdown(r"""
We observed that each left bracket attended approximately uniformly to each of the tokens to its right, and used this to detect elevation failures at any point. We also know that this approximately uniform pattern breaks down around query positions 27-31. 

With this in mind, what kind of "just barely" unbalanced bracket string could we construct that would get classified as balanced by the model? 

Read the next hint for a suggested type of bracket string.
""")

    with st.expander("Hint 3"):
        st.markdown(r"""
We want to construct a string that has a negative elevation at some point, but is balanced everywhere else. We can do this by using a sequence of the form `A)(B`, where `A` and `B` are balanced substrings. The positions of the open paren next to the `B` will thus be the only position in the whole sequence on which the elevation drops below zero, and it will drop just to -1.

Read the next hint to get ideas for what `A` and `B` should be (the clue is in the attention pattern plot!).
""")

    with st.expander("Hint 4"):
        st.markdown(r"""
From the attention pattern plot, we can see that left parens in the range 27-31 attend bizarrely strongly to the tokens at position 38-40. This means that, if there is a negative elevation in or after the range 27-31, then the left bracket that should be detecting this negative elevation might miscount. In particular, if `B = ((...))`, this left bracket might heavily count the right brackets at the end, and less heavily weight the left brackets at the start of `B`, thus this left bracket might "think" that the sequence is balanced when it actually isn't.
""")

    with st.expander("Solution"):
        st.markdown(r"""
Choose `A` and `B` to each be a sequence of `(((...)))` terms with length $i$ and $38-i$ respectively (it makes sense to choose `A` like this also, because want the sequence to have maximal positive elevation everywhere except the single position where it's negative). Then, maximize over $i = 2, 4, ...\,$. Unsurprisingly given the observations in the previous hint, we find that the best adversarial examples (all with balanced probability of above 98%) are $i=24, 26, 28, 30, 32$. The best of these is $i=30$, which gets 99.9856% balanced confidence.

```python
def simple_balanced_bracket(length: int) -> str:
        return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])
    
example = simple_balanced_bracket(15) + ")(" + simple_balanced_bracket(4)
```

Please message the group if you find a better advex!""")

        st_image("graph.png", 900)

    st.markdown(r"""
```python
if MAIN:
    print("Update the examples list below below find adversarial examples")
    examples = ["()", "(())", "))"]
    m = max(len(ex) for ex in examples)
    toks = tokenizer.tokenize(examples).to(DEVICE)
    probs = model(toks)[:, 0].softmax(-1)[:, 1]
    print("\n".join([f"{ex:{m}} -> {p:.4%} balanced confidence" for (ex, p) in zip(examples, probs)]))
```
""")

func_list = [section_home, section_1, section_2, section_3, section_4]

page_list = ["🏠 Home", "1️⃣ Bracket classifier", "2️⃣ Moving backwards", "3️⃣ Total elevation circuit", "4️⃣ Bonus exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()


