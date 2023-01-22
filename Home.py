import os
if not os.path.exists("./images"):
    os.chdir("./ch6")

from st_dependencies import *
st.set_page_config(layout="wide")

st.markdown("""
<style>
div[data-testid="column"] {
    background-color: #f9f5ff;
    padding: 15px;
}
.st-ae h2 {
    margin-top: -15px;
}
p {
    line-height:1.48em;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code {
    color:red;
    white-space: pre-wrap !important;
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#about-transformerlens">About TransformerLens</a></li>
    <li><a class="contents-el" href="#glossary">Glossary</a></li>
    <li><a class="contents-el" href="#prerequisites">Prerequisites</a></li>
    <li><a class="contents-el" href="#about-these-pages">About these pages</a></li>
</ul>
""", unsafe_allow_html=True)

def page():
    st_image("magnifying-glass.png", width=350)
    st.title("Mechanistic Interpretability & TransformerLens")

    st.markdown(r"""

This page contains a collation of resources and exercises on interpretability. The focus is on [`TransformerLens`](https://github.com/neelnanda-io/TransformerLens), a library maintained by Neel Nanda.

All the demos, exercises and solutions can be found in the 

### How you should use this material

The best way to use this material is to clone the [GitHub repo](https://github.com/callummcdougall/TransformerLens-intro), and run it on your local machine. The vast majority of the exercises will not require a particularly good GPU, and where there are exceptions we will give some advice for how to get the most out of the exercises regardless.

Full instructions for running the exercises in this way:

* Clone the [GitHub repo](https://github.com/callummcdougall/TransformerLens-intro) into your local directory.
* Open in your choice of IDE (we recommend VSCode).
* Make & activate a virtual environment
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
* Install requirements.
    * First, install PyTorch (the command to run in your terminal can be found at the top of the `requirements.txt` file in your repo).
    * Then install the rest of the requirements e.g. with `pip install requirements.txt`.
* Navigate to the directory, and run `streamlit run Home.py` (this should work since Streamlit is one of the libraries in `requirements.txt`).
    * This should open up a local copy of the page you're reading right now!

You can go through each batch of exercises (i.e. each section at the top of the left-hand sidebar) by going into the appropriate directory in the repo, and creating a file called `answers.py` (or `answers.ipynb` if you prefer using notebooks). For each exercise, you'll be given import instructions which should go at the top of the file. These imports will incude local imports (e.g. you will have `tests.py` which contain test functions, to validate that your solutions are correct, and `solutions.py` which contain the solutions to all exercises - although the solutions will also be viewable in dropdowns in the Streamlit page). The exercises will also occasionally have you generate output that gets displayed in your Streamlit page.

Alternatively, you can open the pages in Colab. This means GPU support is guaranteed, but you won't be able to use your chosen IDE, and there are certain features of these exercises which might not work as well.

## About TransformerLens

From Neel's [TransformerLens repo's README](https://github.com/neelnanda-io/TransformerLens):

> *This is a library for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this!*
> 
> *TransformerLens lets you load in an open source language model, like GPT-2, and exposes the internal activations of the model to you. You can cache any internal activation in the model, and add in functions to edit, remove or replace these activations as the model runs. The core design principle I've followed is to enable exploratory analysis. One of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. Part of what I aimed for is to make my experience of doing research easier and more fun, hopefully this transfers to you!*

## Glossary

Neel recently released a [Mechanistic Interpretability Glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J), which I highly recommend you check out. It's a great resource for understanding the terminology and concepts that are used in the following pages. 

## Prerequisites

The main prerequisite we assume in these pages is a working understanding of transformers. In particular, you are strongly recommended to go through Neel's [Colab & video tutorial](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo_Template.ipynb#scrollTo=SKVxRKXVsgO6).

You are also strongly recommended to read the paper [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), or watch Neel's walkthrough of the paper [here](https://www.youtube.com/watch?v=KV5gbOmHbjU). Much of this material will be re-covered in the following pages (e.g. as we take a deeper dive into induction heads in the first set of exercises), but it will still probably be useful.

Below are a collection of unstructured notes from Neel, to help better understand the Mathematical Frameworks paper. Several of these are also addressed in his glossary and video walkthrough. There's a lot here, so don't worry about understanding every last point!""")

    with st.expander("Tips & Insights for the Paper"):
        st.markdown(r"""

* The eigenvalue stuff is very cool, but doesn't generalise that much, it's not a priority to get your head around
* It's really useful to keep clear in your head the difference between parameters (learned numbers that are intrinsic to the network and independent of the inputs) and activations (temporary numbers calculated during a forward pass, that are functions of the input).
    * Attention is a slightly weird thing - it's an activation, but is also used in a matrix multiplication with another activation (z), which makes it parameter-y.
        * The idea of freezing attention patterns disentangles this, and lets us treat it as parameters.
* The residual stream is the fundamental object in a transformer - each layer just applies incremental updates to it - this is really useful to keep in mind throughout!
    * This is in contrast to a classic neural network, where each layer's output is the central object
    * To underscore this, a funky result about transformers is that the aspect ratio isn't *that* important - if you increase d_model/n_layer by a factor of 10 from optimal for a 1.5B transformer (ie controlling for the number of parameters), then loss decreases by <1%.
* The calculation of attention is a bilinear form (ie via the QK circuit) - for any pair of positions it takes an input vector from each and returns a scalar (so a ctx x ctx tensor for the entire sequence), while the calculation of the output of a head pre weighting by attention (ie via the OV circuit) is a linear map from the residual stream in to the residual stream out - the weights have the same shape, but are doing functions of completely different type signatures!
* How to think about attention: A framing I find surprisingly useful is that attention is the "wiring" of the neural network. If we hold the attention patterns fixed, they tell the model how to move information from place to place, and thus help it be effective at sequence prediction. But the key interesting thing about a transformer is that attention is *not* fixed - attention is computed and takes a substantial fraction of the network's parameters, allowing it to dynamically set the wiring. This can do pretty meaningful computation, as we see with induction heads, but is in some ways pretty limited. In particular, if the wiring is fixed, an attention only transformer is a purely linear map! Without the ability to intelligently compute attention, an attention-only transformer would be incredibly limited, and even with it it's highly limited in the functional forms it can represent.
    * Another angle - attention as generalised convolution. A naive transformer would use 1D convolutions on the sequence. This is basically just attention patterns that are hard coded to be uniform over the last few tokens - since information is often local, this is a decent enough default wiring. Attention allows the model to devote some parameters to compute more intelligent wiring, and thus for a big enough and good enough model will significantly outperform convolutions.
* One of the key insights of the framework is that there are only a few activations of the network that are intrinsically meaningful and interpretable - the input tokens, the output logits and attention patterns (and neuron activations in non-attention-only models). Everything else (the residual stream, queries, keys, values, etc) are just intermediate states on a calculation between two intrinsically meaningful things, and you should instead try to understand the start and the end. Our main goal is to decompose the network into many paths between interpretable start and end states
    * We can get away with this because transformers are really linear! The composition of many linear components is just one enormous matrix
* A really key thing to grok about attention heads is that the QK and OV circuits act semi-independently. The QK circuit determines which previous tokens to attend to, and the OV circuit determines what to do to tokens *if* they are attended to. In particular, the residual stream at the destination token *only* determines the query and thus what tokens to attend to - what the head does *if* it attends to a position is independent of the destination token residual stream (other than being scaled by the attention pattern).
    <p align="center">
        <img src="w2d4_Attn_Head_Pic.png" width="400" />
    </p>
* Skip trigram bugs are a great illustration of this - it's worth making sure you really understand them. The key idea is that the destination token can *only* choose what tokens to pay attention to, and otherwise not mediate what happens *if* they are attended to. So if multiple destination tokens want to attend to the same source token but do different things, this is impossible - the ability to choose the attention pattern is insufficient to mediate this.
    * Eg, keep...in -> mind is a legit skip trigram, as is keep...at -> bay, but keep...in -> bay is an inherent bug from this pair of skip trigrams
* The tensor product notation looks a lot more complicated than it is. $A \otimes W$ is shorthand for "the function $f_{A,W}$ st $f_{A,W}(x)=AxW$" - I recommend mentally substituting this in in your head everytime you read it.
* K, Q and V composition are really important and fairly different concepts! I think of each attention head as a circuit component with 3 input wires (Q,K,V) and a single output wire (O). Composition looks like connecting up wires, but each possible connection is a choice! The key, query and value do different things and so composition does pretty different things.
    * Q-Composition, intuitively, says that we want to be more intelligent in choosing our destination token - this looks like us wanting to move information to a token based on *that* token's context. A natural example might be the final token in a several token word or phrase, where earlier tokens are needed to disambiguate it, eg `E|iff|el| Tower|`
    * K-composition, intuitively, says that we want to be more intelligent in choosing our source token - this looks like us moving information *from* a token based on its context (or otherwise some computation at that token).
        * Induction heads are a clear example of this - the source token only matters because of what comes before it!
    * V-Composition, intuitively, says that we want to *route* information from an earlier source token *other than that token's value* via the current destination token. It's less obvious to me when this is relevant, but could imagine eg a network wanting to move information through several different places and collate and process it along the way
        * One example: In the ROME paper, we see that when models recall that "The Eiffel Tower is in" -> " Paris", it stores knowledge about the Eiffel Tower on the " Tower" token. When that information is routed to `| in|`, it must then map to the output logit for `| Paris|`, which seems likely due to V-Composition
* A surprisingly unintuitive concept is the notion of heads (or other layers) reading and writing from the residual stream. These operations are *not* inverses! A better phrasing might be projecting vs embedding.
    * Reading takes a vector from a high-dimensional space and *projects* it to a smaller one - (almost) any two pair of random vectors will have non-zero dot product, and so every read operation can pick up *somewhat* on everything in the residual stream. But the more a vector is aligned to the read subspace, the most that vector's norm (and intuitively, its information) is preserved, while other things are lower fidelity
        * A common reaction to these questions is to start reasoning about null spaces, but I think this is misleading - rank and nullity are discrete concepts, while neural networks are fuzzy, continuous objects - nothing ever actually lies in the null space or has non-full rank (unless it's explicitly factored). I recommend thinking in terms of "what fraction of information is lost". The null space is the special region with fraction lost = 1
    * Writing *embeds* a vector into a small dimensional subspace of a larger vector space. The overall residual stream is the sum of many vectors from many different small subspaces.
        * Every read operation can see into every writing subspace, but will see some with higher fidelity, while others are noise it would rather ignore.
    * It can be useful to reason about this by imagining that d_head=1, and that every vector is a random Gaussian vector - projecting a random Gaussian onto another in $\mathbb{R}^n$ will preserve $\frac{1}{n}$ of the variance, on average.
* A key framing of transformers (and neural networks in general) is that they engage in **lossy compression** - they have a limited number of dimensions and want to fit in more directions than they have dimensions. Each extra dimension introduces some interference, but has the benefit of having more expressibility. Neural networks will learn an optimal-ish solution, and so will push the compression as far as it can until the costs of interference dominate.
    * This is clearest in the case of QK and OV circuits - $W_QK=W_Q^TW_K$ is a d_model x d_model matrix with rank d_head. And to understand the attention circuit, it's normally best to understand $W_QK$ on its own. Often, the right mental move is to forget that $W_QK$ is low rank, to understand what the ideal matrix to learn here would be, and then to assume that the model learns the best low rank factorisation of that.
        * This is another reason to not try to interpret the keys and queries - the intermediate state of a low rank factorisations are often a bit of a mess because everything is so compressed (though if you can do SVD on $W_QK$ that may get you a meaningful basis?)
        * Rough heuristic for thinking about low rank factorisations and how good they can get - a good way to produce one is to take the SVD and zero out all but the first d_head singular values.
    * This is the key insight behind why polysemanticity (back from w1d5) is a thing and is a big deal - naturally the network would want to learn one feature per neuron, but it in fact can learn to compress more features than total neurons. It has some error introduced from interference, but this is likely worth the cost of more compression.
        * Just as we saw there, the sparsity of features is a big deal for the model deciding to compress things! Inteference cost goes down the more features are sparse (because unrelated features are unlikely to co-occur) while expressibility benefits don't really change that much.
    * The residual stream is the central example of this - every time two parts of the network compose, they will be communicating intermediate states via the residual stream. Bandwidth is limited, so these will likely try to each be low rank. And the directions within that intermediate product will *only* make sense in the context of what the writing and reading components care about. So interpreting the residual stream seems likely fucked - it's just specific lower-dimensional parts of the residual stream that we care about at each point, corresponding to the bits that get preserved by our $W_Q$ / $W_K$ / $W_V$ projection matrices, or embedded by our $W_O$ projection matrix. The entire residual stream will be a mix of a fuckton of different signals, only some of which will matter for each operation on it.
* The *'the residual stream is fundamentally uninterpretable'* claim is somewhat overblown - most models do dropout on the residual stream which somewhat privileges that basis
    * And there are [*weird*](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) results about funky directions in the residual stream.
* Getting your head around the idea of a privileged basis is very worthwhile! The key mental move is to flip between "a vector is a direction in a geometric space" and "a vector is a series of numbers in some meaningful basis, where each number is intrinsically meaningful". By default, it's easy to spend too much time in the second mode, because every vector is represented as a series of numbers within the GPU, but this is often less helpful!

### An aside on why we need the tensor product notation at all

Neural networks are functions, and are built up of several subcomponents (like attention heads) that are also functions - they are defined by how they take in an input and return an output. But when doing interpretability we want the ability to talk about the network as a function intrinsically and analyse the structure of this function, *not* in the context of taking in a specific input and getting a specific output. And this means we need a language that allows us to naturally talk about functions that are the sum (components acting in parallel) or composition (components acting in series) of other functions.

A simple case of this: We're analysing a network with several linear components acting in parallel - component $C_i$ is the function $x \rightarrow W_ix$, and can be represented intrinsically as $W_i$ (matrices are equivalent to linear maps). We can represent the layer with all acting in parallel as $x \rightarrow \sum_i W_ix=(\sum_i W_i)x$, and so intrinsically as $\sum_i W_i$ - this is easy because matrix notation is designed to make addition of.

Attention heads are harder because they map the input tensor $x$ (shape: `[position x d_model]`) to an output $Ax(W_OW_V)^T$ - this is a linear function, but now on a *tensor*, so we can't trivially represent addition and composition with matrix notation. The paper uses the notation $A\otimes W_OW_V$, but this is just different notation for the same underlying function. The output of the layer is the sum over the 12 heads: $\sum_i A^{(i)}x(W_O^{(i)}W_V^{(i)})^T$. And so we could represent the function of the entire layer as $\sum_i A^{(i)} x (W_O^{(i)}W_V^{(i)})$. There are natural extensions of this notation for composition, etc, though things get much more complicated when reasoning about attention patterns - this is now a bilinear function of a pair of inputs: the query and key residual streams. (Note that $A$ is also a function of $x$ here, in a way that isn't obvious from the notation.)

The key point to remember is that if you ever get confused about what a tensor product means, explicitly represent it as a function of some input and see if things feel clearer.
""")
    st.markdown(r"""

## About these pages

There are three main pages in this notebook. 

* **TransformerLens & induction circuits** shows you the core features of TransformerLens, and also includes exercises to help reinforce what you've learned.
* **Interpretability on an algorithmic model** takes you through the application of TransformerLens features to an algorithmic task - interpreting a model trained to classify bracket strings as balanced or unbalanced. This is a great way to get a feel for interpretability, since we can compare the solution our model finds to the algorithmic solution.
* Finally, **Interpretability in the wild** takes you on a tour through some of the results of the recent paper [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593). This is to date the largest circuit that has been discovered in a language model.

Each page is split into several sub-pages. Each subpage starts with a list of learning objectives. For example, here's the learning objectives for the first subpage of the TransformerLens & induction circuits page:""")

    st.info(r"""
## Learning objectives

* Load and run a `HookedTransformer` model.
* Understand the basic architecture of these models.
* Use the model's tokenizer to convert text to tokens, and vice versa.
* Know how to cache activations, and to access activations from the cache.
* Use `circuitsvis` to visualise attention heads.""")

    st.markdown(r"""
Additionally, each subpage will have exercises for you to complete. These are denoted by a purple box. For example, here's a short exercise from the first subpage of the TransformerLens & induction circuits page:""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - verify activations

Verify that `hook_q`, `hook_k` and hook_pattern are related to each other in the way implied by the diagram. Do this by computing `layer0_pattern_from_cache` (the attention pattern taken directly from the cache, for layer 0) and `layer0_pattern_from_q_and_k` (the attention pattern calculated from `hook_q` and `hook_k`, for layer 0).

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
Some of these exercises will be very directed, and will just require you to fill in a few lines of code. Others will be more challenging. All the exercises have solutions available to view, so you can use them if you get stuck.
""")

if is_local or check_password():
    page()
