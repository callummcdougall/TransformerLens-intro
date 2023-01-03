# %%
import functools
import json
import os
from typing import Any, List, Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
from torch import nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange, repeat
import pandas as pd
import numpy as np

import sys, os
# os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v2-exercises\chapter6_interpretability")
# sys.path.append(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v2-exercises\chapter6_interpretability")

import w5d5_tests
from w5d5_transformer import ParenTransformer, SimpleTokenizer

MAIN = __name__ == "__main__"
DEVICE = t.device("cpu")

# import plotly.io as pio
# pio.renderers.default = "browser"

# %%

if MAIN:
    model = ParenTransformer(ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3).to(DEVICE)
    state_dict = t.load("w5d5_balanced_brackets_state_dict.pt")
    model.to(DEVICE)
    model.load_simple_transformer_state_dict(state_dict)
    model.eval()
    tokenizer = SimpleTokenizer("()")
    with open("w5d5_brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)

# %%

# for me!

# def plot_elevation(paren_string, title):
#     """Plot the elevation of a paren string"""
#     elevation = 0
#     elevations = []
#     for c in paren_string:
#         if c == "(":
#             elevation += 1
#         else:
#             elevation -= 1
#         elevations.append(elevation)
#     elevations = [e - elevations[-1] for e in elevations]
#     fig = px.line(x=list(range(len(paren_string))), y=elevations, template="simple_white", title=title)
#     fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=5)
#     fig.show()

# plot_elevation("((()()()()))", "Balanced brackets")

# plot_elevation("(()()()(()(())()", "Unbalanced brackets '(()()()(()(())()': `elevation[0] != 0`")

# plot_elevation("()(()))())", "Unbalanced brackets: `any(elevation < 0)`")

# %%

class DataSet:
    """A dataset containing sequences, is_balanced labels, and tokenized sequences"""

    def __init__(self, data_tuples: list):
        """
        data_tuples is List[Tuple[str, bool]] signifying sequence and label
        """
        self.strs = [x[0] for x in data_tuples]
        self.isbal = t.tensor([x[1] for x in data_tuples]).to(device=DEVICE, dtype=t.bool)
        self.toks = tokenizer.tokenize(self.strs).to(DEVICE)
        self.open_proportion = t.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = t.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self) -> int:
        return len(self.strs)

    def __getitem__(self, idx) -> Union["DataSet", Tuple[str, t.Tensor, t.Tensor]]:
        if type(idx) == slice:
            return self.__class__(list(zip(self.strs[idx], self.isbal[idx])))
        return (self.strs[idx], self.isbal[idx], self.toks[idx])

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)

    @classmethod
    def with_length(cls, data_tuples: List[Tuple[str, bool]], selected_len: int) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if len(s) == selected_len])

    @classmethod
    def with_start_char(cls, data_tuples: List[Tuple[str, bool]], start_char: str) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if s[0] == start_char])


if MAIN:
    N_SAMPLES = 5000
    data_tuples = data_tuples[:N_SAMPLES]
    data = DataSet(data_tuples)
    data_mini = data[:500]

    bracket_lengths = [len(s[0]) for s in data]
    fig = px.histogram(
        title="Length of bracket strings in dataset",
        x=bracket_lengths, 
        nbins=max(bracket_lengths),
        template="simple_white"
    )
    fig.show()

# %%

def is_balanced_forloop(parens: str) -> bool:
    """Return True if the parens are balanced.
    Parens is just the ( and ) characters, no begin or end tokens.
    """
    i = 0
    for c in parens:
        if c == "(":
            i += 1
        elif c == ")":
            i -= 1
            if i < 0:
                return False
        else:
            raise ValueError(parens)
    return i == 0


if MAIN:
    examples = ["()", "))()()()()())()(())(()))(()(()(()(", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, False, True, False, True]
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")

# %%

def is_balanced_vectorized(tokens: t.Tensor) -> bool:
    """
    tokens: sequence of tokens including begin, end and pad tokens - recall that 3 is '(' and 4 is ')'
    """
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    altitude = t.cumsum(change, -1)
    no_total_elevation_failure = altitude[-1] == 0
    no_negative_failure = altitude.min(0).values >= 0

    return no_total_elevation_failure and no_negative_failure


if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")

# %%

if MAIN:
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    prob_balanced = out.exp()[:, 1]
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))


def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    """Return probability that each example is balanced"""
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out

if MAIN:
    test_set = data
    n_correct = t.sum((run_model_on_data(model, test_set).argmax(-1) == test_set.isbal).int())
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")

# %%

def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    return model.decoder.weight[0] - model.decoder.weight[1]

# %%

def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    acts = []
    fn = lambda module, inputs, output: acts.append(inputs[0].detach().clone())
    h = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    h.remove()
    out = t.concat(acts, dim=0)
    assert out.shape == (len(data), data.seq_length, model.d_model)
    return out.clone()


def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    acts = []
    fn = lambda module, inputs, output: acts.append(output.detach().clone())
    h = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    h.remove()
    out = t.concat(acts, dim=0)
    assert out.shape == (len(data), data.seq_length, model.d_model)
    return out.clone()


if MAIN:
    w5d5_tests.test_get_inputs(get_inputs, model, data)
    w5d5_tests.test_get_outputs(get_outputs, model, data)

# %%

def get_ln_fit(
    model: ParenTransformer, data: DataSet, ln_module: nn.LayerNorm, seq_pos: Union[None, int]
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in torch.tensor() to make a dimensionless tensor)
    '''
    inputs = get_inputs(model, data, ln_module)
    outputs = get_outputs(model, data, ln_module)

    if seq_pos is None:
        inputs = rearrange(inputs, "batch seq hidden -> (batch seq) hidden")
        outputs = rearrange(outputs, "batch seq hidden -> (batch seq) hidden")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]

    final_ln_fit = LinearRegression().fit(inputs, outputs)

    r2 = t.tensor(final_ln_fit.score(inputs, outputs))

    return (final_ln_fit, r2)


if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, model.norm, seq_pos=None)
    print("r^2: ", r2)
    w5d5_tests.test_final_ln_fit(model, data, get_ln_fit)

# %%

def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    post_final_ln_dir = get_post_final_ln_dir(model)
    L = t.from_numpy(get_ln_fit(model, data, model.norm, seq_pos=0)[0].coef_)

    return t.einsum("i,ij->j", post_final_ln_dir, L)


if MAIN:
    w5d5_tests.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)

# %%

def get_out_by_head(model: ParenTransformer, data: DataSet, layer: int) -> t.Tensor:
    '''

    Get the output of the heads in a particular layer when the model is run on the data.
    Returns a tensor of shape (batch, nheads, seq, emb)
    '''
    # Capture the inputs to this layer using the input hook function you wrote earlier - call the inputs `r`
    module = model.layers[layer].self_attn.W_O
    r = get_inputs(model, data, module)

    # Reshape the inputs so that heads go along their own dimension (see expander above)
    r = rearrange(r, "batch seq (nheads headsize) -> batch nheads seq headsize", nheads=model.nhead)

    # Extract the weights from the model directly, and reshape them so that heads go along their own dimension
    W_O = module.weight
    W_O = rearrange(W_O, "emb (nheads headsize) -> nheads emb headsize", nheads=model.nhead)

    # Perform the matrix multiplication shown in the expander above (but keeping the terms in the sum separate)
    out_by_heads = einsum("batch nheads seq headsize, nheads emb headsize -> batch nheads seq emb", r, W_O)
    return out_by_heads


if MAIN:
    w5d5_tests.test_get_out_by_head(get_out_by_head, model, data)

# %%

def get_out_by_components(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    # Get the outputs of each head, for each layer
    head_outputs = [get_out_by_head(model, data, layer) for layer in range(model.nlayers)]
    # Get the MLP outputs for each layer
    mlp_outputs = [get_outputs(model, data, model.layers[layer].linear2) for layer in range(model.nlayers)]
    # Get the embedding outputs (note that model.pos_encoder is applied after the token embedding)
    embedding_output = get_outputs(model, data, model.pos_encoder)

    # Start with [embeddings]
    out = [embedding_output]
    for layer in range(model.nlayers):
        # Append [head n.0, head n.1, mlp n] for n = 0,1,2
        # Note that the heads are in the second dimension of the head_outputs tensor (the first is batch)
        out.extend([head_outputs[layer][:, 0], head_outputs[layer][:, 1], mlp_outputs[layer]])

    return t.stack(out, dim=0)


if MAIN:
    w5d5_tests.test_get_out_by_component(get_out_by_components, model, data_mini)

# %%

if MAIN:
    biases = sum([model.layers[l].self_attn.W_O.bias for l in (0, 1, 2)]).clone()
    out_by_components = get_out_by_components(model, data_mini)
    summed_terms = out_by_components.sum(dim=0) + biases
    pre_final_ln = get_inputs(model, data_mini, model.norm)
    t.testing.assert_close(summed_terms, pre_final_ln)

# %%

def hists_per_comp(magnitudes, data, n_layers=3, xaxis_range=(-1, 1)):
    num_comps = magnitudes.shape[0]
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0",
        (2, 2): "head 0.1",
        (2, 3): "mlp 0",
        (3, 1): "head 1.0",
        (3, 2): "head 1.1",
        (3, 3): "mlp 1",
        (4, 1): "head 2.0",
        (4, 2): "head 2.1",
        (4, 3): "mlp 2"
    }
    assert num_comps == len(titles)

    fig = make_subplots(rows=n_layers+1, cols=3)
    for ((row, col), title), mag in zip(titles.items(), magnitudes):
        if row == n_layers+2: break
        fig.add_trace(go.Histogram(x=mag[data.isbal].numpy(), name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=mag[~data.isbal].numpy(), name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(width=1200, height=250*(n_layers+1), barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()
    return fig


if MAIN:
    # Get output by components at the 0th sequence position
    out_by_components = get_out_by_components(model, data)[:, :, 0, :].detach()
    # Get unbalanced directions for balanced and unbalanced respectively
    unbalanced_dir = get_pre_final_ln_dir(model, data).detach()
    # Get magnitudes, and plot them
    magnitudes = einsum("component sample emb, emb -> component sample", out_by_components, unbalanced_dir)
    # Subtract the mean of the balanced magnitudes from each component
    magnitudes = magnitudes - magnitudes[:, data.isbal].mean(-1, keepdim=True)
    hists_per_comp(magnitudes, data, xaxis_range=[-10, 20])

# %%

def is_balanced_vectorized_return_both(tokens: t.Tensor) -> bool:
    tokens = tokens.flip(0)
    tokens[tokens <= 2] = 0
    tokens[tokens == 3] = 1
    tokens[tokens == 4] = -1
    brackets_altitude = tokens.cumsum(0)
    total_elevation_failure = brackets_altitude[-1] != 0
    negative_failure = brackets_altitude.max(0).values > 0

    assert total_elevation_failure.shape == negative_failure.shape == t.Size([5000])

    return total_elevation_failure, negative_failure

if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks.T.clone())
    h20_in_d = magnitudes[7] - magnitudes[7, data.isbal].mean(0)
    h21_in_d = magnitudes[8] - magnitudes[8, data.isbal].mean(0)

    failure_types = np.full(len(h20_in_d), "", dtype=np.dtype("U32"))
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure
    }
    for name, mask in failure_types_dict.items():
        failure_types = np.where(mask, name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": h20_in_d,
        "Head 2.1 contribution": h21_in_d,
        "Failure type": failure_types
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, 
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4)
    fig.show()

# %%

if MAIN:
    fig = px.scatter(
        x=data.open_proportion, y=h20_in_d, color=failure_types, 
        title="Head 2.0 contribution vs proportion of open brackets '('", template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}, category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()
# %%

def get_attn_probs(model: ParenTransformer, tokenizer: SimpleTokenizer, data: DataSet, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    # Get additive attention mask as described in the instructions
    additive_attention_mask = t.where(data.toks == tokenizer.PAD_TOKEN, t.tensor(-1.0e9), 0.0)
    # Broadcast it across the nheads and seqQ dimensions
    additive_attention_mask = rearrange(additive_attention_mask, "batch seqK -> batch 1 1 seqK")

    # Get the attention module
    attention_module = model.layers[layer].self_attn
    # Get the inputs to this attention module
    inputs = get_inputs(model, data, attention_module)
    # Get the attention pattern
    attention_pattern_pre_softmax = attention_module.attention_pattern_pre_softmax(inputs)
    attention_pattern_post_softmax = (attention_pattern_pre_softmax + additive_attention_mask).softmax(-1)

    # Get the correct head
    attention_pattern_post_softmax = attention_pattern_post_softmax[:, head]

    return attention_pattern_post_softmax.detach().clone()

if MAIN:
    attn_probs = get_attn_probs(model, tokenizer, data, 2, 0)
    attn_probs_open = attn_probs[data.starts_open].mean(0)[[0]]
    fig = px.bar(
        y=attn_probs_open.squeeze().numpy(), labels={"y": "Probability", "x": "Key Position"},
        template="simple_white", height=500, width=600, title="Avg Attention Probabilities for '(' query from query 0"
    ).update_layout(showlegend=False, hovermode='x unified')
    fig.show()

# %%

def get_WV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_V matrix of a head. Should be a CPU tensor of size (d_model / num_heads, d_model)
    '''
    # Get W_V which has shape (nheads * headsize, embed_size)
    W_V = model.layers[layer].self_attn.W_V.weight
    # Reshape it to (nheads, headsize, embed_size) and get the correct head
    W_V_head = rearrange(W_V, "(nheads headsize) embed_size -> nheads headsize embed_size", nheads=model.nhead)[head]

    return W_V_head.detach().clone()

def get_WO(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_O matrix of a head. Should be a CPU tensor of size (d_model, d_model / num_heads)
    '''
    # Get W_O which has shape (embed_size, nheads * headsize)
    W_O = model.layers[layer].self_attn.W_O.weight
    # Reshape it to (embed_size, nheads, headsize) and get the correct head
    W_O_head = rearrange(W_O, "embed_size (nheads headsize) -> embed_size nheads headsize", nheads=model.nhead)[:, head]
    
    return W_O_head.detach().clone()

def get_WOV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    return get_WO(model, layer, head) @ get_WV(model, layer, head)

def get_pre_20_dir(model, data):
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    '''
    pre_final_ln_dir = get_pre_final_ln_dir(model, data)
    WOV = get_WOV(model, 2, 0)

    # Get the regression fit for the second layernorm in layer 2
    layer2_ln_fit, r2 = get_ln_fit(model, data, model.layers[2].norm1, seq_pos=1)
    layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_)

    # Propagate back through the layernorm
    pre_20_dir = t.einsum("i,ij,jk->k", pre_final_ln_dir, WOV, layer2_ln_coefs)

    return pre_20_dir


if MAIN:
    w5d5_tests.test_get_WV(model, get_WV)
    w5d5_tests.test_get_WO(model, get_WO)
    w5d5_tests.test_get_pre_20_dir(model, data_mini, get_pre_20_dir)

if MAIN:
    # Get output by components at the 1th sequence position
    out_by_components = get_out_by_components(model, data)[:, :, 1, :].detach()
    # Get unbalanced directions for balanced and unbalanced respectively
    unbalanced_dir = get_pre_20_dir(model, data).detach()
    # Get magnitudes, and plot them
    magnitudes = einsum("component sample emb, emb -> component sample", out_by_components, unbalanced_dir)
    # Subtract the mean of the balanced magnitudes from each component
    magnitudes = magnitudes - magnitudes[:, data.isbal].mean(-1, keepdim=True)

# %%

if MAIN:
    fig = hists_per_comp(magnitudes, data, n_layers=2, xaxis_range=(-7, 7))

# %%

def mlp_attribution_scatter(data, magnitudes, failure_types, layer):
    px.scatter(
        x=data.open_proportion[data.starts_open], y=magnitudes[3+layer*3, data.starts_open], 
        color=failure_types[data.starts_open], category_orders={"color": failure_types_dict.keys()},
        title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
        template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type').show()

if MAIN:
    for layer in range(2):
        mlp_attribution_scatter(data, magnitudes, failure_types, layer)

# %%

def out_by_neuron(model, data, layer):
    '''
    Return shape: (len(data), seq_len, neurons, out)
    '''
    f_B_x = get_outputs(model, data, model.layers[layer].activation)
    A = model.layers[layer].linear2.weight
    return einsum("sample seq neuron, out neuron -> sample seq neuron out", f_B_x, A)

@functools.cache
def out_by_neuron_in_20_dir(model, data, layer):
    by_neuron = out_by_neuron(model, data, layer)
    direction = get_pre_20_dir(model, data)
    return einsum("batch seq neuron out, out -> batch seq neuron", by_neuron, direction)

def plot_neurons(model, data, failure_types, layer):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)[data.starts_open, 1, :].detach()

    # Get data that can be turned into a dataframe (plotly express is sometimes easier to use with a dataframe)
    # Plot a scatter plot of all the neuron contributions, color-coded according to failure type, with slider to view neurons
    neuron_numbers = repeat(t.arange(model.d_model), "n -> (s n)", s=data.starts_open.sum())
    failure_types = repeat(failure_types[data.starts_open], "s -> (s n)", n=model.d_model)
    data_open_proportion = repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": neurons_in_d.flatten(), "Neuron number": neuron_numbers,
        "Open-proportion": data_open_proportion, "Failure type": failure_types
    })
    px.scatter(
        df, x="Open-proportion", y="Output in 2.0 direction", color="Failure type", animation_frame="Neuron number",
        title=f"Neuron contributions from layer {layer}", template="simple_white", height=500, width=800
    ).update_traces(marker_size=3).update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5]).show()

    # Work out the importance (average difference in unbalanced contribution between balanced and inbalanced dirs) for each neuron
    # Plot a bar chart of the per-neuron importances
    neuron_importances = neurons_in_d[~data.isbal[data.starts_open]].mean(0) - neurons_in_d[data.isbal[data.starts_open]].mean(0)
    fig_bar = px.bar(
        x=t.arange(model.d_model), y=neuron_importances, 
        title=f"Importance of neurons in layer {layer}", 
        labels={"x": "Neuron number", "y": "Mean contribution in unbalanced dir"},
        template="simple_white", height=400, width=600
    ).update_layout(hovermode="x unified")
    fig_bar.layout.updatemenus = []
    fig_bar.show(renderer="browser")

# %%

if MAIN:
    for layer in range(2):
        plot_neurons(model, data, failure_types, layer)

# %%

def get_Q_and_K(model: ParenTransformer, layer: int, head: int) -> Tuple[t.Tensor, t.Tensor]:
    '''
    Get the Q and K weight matrices for the attention head at the given indices.

    Return: Tuple of two tensors, both with shape (embedding_size, head_size)
    '''
    W_Q = rearrange(model.layers[layer].self_attn.W_Q.weight, "(nheads headsize) emb -> nheads emb headsize", nheads=model.nhead)
    W_K = rearrange(model.layers[layer].self_attn.W_K.weight, "(nheads headsize) emb -> nheads emb headsize", nheads=model.nhead)

    return W_Q[head], W_K[head]


def qk_calc_termwise(
    model: ParenTransformer, layer: int, head: int, q_embedding: t.Tensor, k_embedding: t.Tensor
) -> t.Tensor:
    '''
    Get the pre-softmax attention scores that would be calculated by the given attention head from the given embeddings.

    q_embedding: tensor of shape (seq_len, embedding_size)
    k_embedding: tensor of shape (seq_len, embedding_size)

    Returns: tensor of shape (seq_len, seq_len)
    '''
    Q, K = get_Q_and_K(model, layer, head)
    q = einsum("e h, s e -> h s", Q, q_embedding)
    k = einsum("e h, s e -> h s", K, k_embedding)
    attention_scores = einsum("h sQ, h sK -> sQ sK", q, k)
    return attention_scores / q.shape[0] ** 0.5


if MAIN:
    w5d5_tests.qk_test(model, get_Q_and_K)
    # w5d5_tests.test_qk_calc_termwise(model, tokenizer, qk_calc_termwise)


# %%

if MAIN:
    probs = get_attn_probs(model, tokenizer, data, layer=0, head=0)[data.starts_open]
    
    for paren in ["(", ")"]:
        tok = tokenizer.t_to_i[paren]
        mask_for_query = rearrange(t.where(data.toks[data.starts_open] == tok, 1, 0), "b sQ -> b sQ 1")
        probs_avg_for_query_type = (probs * mask_for_query).sum(0) / mask_for_query.sum(0)
        px.imshow(
            probs_avg_for_query_type, 
            color_continuous_scale="RdBu_r", height=500, width=550,
            labels={"x": "Key Position", "y": "Query Position", "color": "Attn prob"},
            title=f"Predicted Attention Probabilities for '{paren}' query", origin="lower"
        ).update_layout(margin=dict(l=60, r=60, t=80, b=40)).show()

# %%

def embedding(model: ParenTransformer, tokenizer: SimpleTokenizer, char: str) -> torch.Tensor:
    assert char in ("(", ")")
    tok = t.tensor([tokenizer.t_to_i[char]]) # Get token id, as a 1D tensor
    return model.encoder(tok) # Get embedding of token

if MAIN:
    w5d5_tests.embedding_test(model, tokenizer, embedding)

# %%

if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")
    pos_embeds = model.pos_encoder.pe
    open_emb_ln_per_seqpos = model.layers[0].norm1(open_emb.to(DEVICE) + pos_embeds[1:41])
    close_emb_ln_per_seqpos = model.layers[0].norm1(closed_emb.to(DEVICE) + pos_embeds[1:41])
    attn_score_open_avg = qk_calc_termwise(model, 0, 0, open_emb_ln_per_seqpos, 0.5 * (open_emb_ln_per_seqpos + close_emb_ln_per_seqpos))
    attn_prob_open = attn_score_open_avg.softmax(-1).detach().clone().numpy()
    px.imshow(
        attn_prob_open, 
        color_continuous_scale="RdBu_r", height=500, width=550,
        labels={"x": "Key Position", "y": "Query Position", "color": "Attn prob"},
        title="Predicted Attention Probabilities for '(' query", origin="upper"
    ).update_layout(margin=dict(l=60, r=60, t=80, b=40)).show()

# %%

def avg_attn_probs_0_0(
    model: ParenTransformer, data: DataSet, tokenizer: SimpleTokenizer, query_token: int
) -> t.Tensor:
    '''
    Calculate the average attention probs for the 0.0 attention head for the provided data when the query is the given query token.
    Returns a tensor of shape (seq, seq)
    '''
    attn_probs = get_attn_probs(model, tokenizer, data, 0, 0)
    is_open = data.toks == query_token
    attn_probs_masked = t.where(is_open[:, :, None], attn_probs.double(), t.nan)
    out = t.nanmean(attn_probs_masked, dim=0)
    return out

if MAIN:
    data_len_40 = DataSet.with_length(data_tuples, 40)
    for paren in ("(", ")"):
        tok = tokenizer.t_to_i[paren]
        attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
        px.imshow(
            attn_probs_mean, 
            color_continuous_scale="RdBu_r", range_color=[0, 0.23], height=500, width=550,
            labels={"x": "Key Position", "y": "Query Position", "color": "Attn prob"},
            title=f"Attention patterns with query = {paren}", origin="upper"
        ).update_layout(margin=dict(l=60, r=60, t=80, b=40)).show()
        px.bar(
            attn_probs_mean[1], title=f"Attention pattern for first query position, query token = {paren!r}",
            labels={"index": "Sequence position", "value": "Average attention"}, template="simple_white", height=450, width=600
        ).update_layout(showlegend=False, margin_l=100, yaxis_range=[0, 0.1], hovermode="x unified").show()

# %%

def embedding_OV_0_0(model, emb_in: t.Tensor) -> t.Tensor:
    return emb_in @ get_WOV(model, 0, 0).T.clone()

if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")
    final_ln_fit, r2 = get_ln_fit(model, data, ln_module=model.layers[0].norm1, seq_pos=None)
    L = t.tensor(final_ln_fit.coef_)
    pe1 = model.pos_encoder.pe[1].clone() * 0
    W_OV_layernorm_of_open = embedding_OV_0_0(model, L @ open_emb.squeeze())
    W_OV_layernorm_of_closed = embedding_OV_0_0(model, L @ closed_emb.squeeze())

    print(f"Magnitudes: open={W_OV_layernorm_of_open.norm()}, closed={W_OV_layernorm_of_closed.norm()}" )
    print(f"Cosine similarity: {t.cosine_similarity(W_OV_layernorm_of_open, W_OV_layernorm_of_closed, dim=0)}")

    v = W_OV_layernorm_of_open

    # Visual plot of their similarities, across all dimensions:
# %%


if MAIN:
    print("Update the examples list below below find adversarial examples")
    examples = ["()", "(())", "))"]

    lenA = 15
    A = "(" * lenA + ")" * lenA
    B = "(" * (19-lenA) + ")" * (19-lenA)
    examples.append(A + ")(" + B)

    m = max(len(ex) for ex in examples)
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    logit_diff = (out[:, 1] - out[:, 0]).detach().cpu().numpy()
    print("\n".join([f"{ex:{m}} -> {p:.4%} balanced confidence" for (ex, p) in zip(examples, out.exp()[:, 1])]))

# %%

def plot_neurons_vs_similarity_with_v_dir(model, data, layer):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)[data.starts_open, 1, :].detach()
    # Work out the importance (average difference in unbalanced contribution between balanced and inbalanced dirs) for each neuron
    # Plot a bar chart of the per-neuron importances
    neuron_importances = neurons_in_d[~data.isbal[data.starts_open]].mean(0) - neurons_in_d[data.isbal[data.starts_open]].mean(0)

    in_dirs_mlp = model.layers[layer].linear1.weight.detach()
    neuron_dotprod_with_in_dir = t.cosine_similarity(in_dirs_mlp.T, W_OV_layernorm_of_open.unsqueeze(-1), dim=0).squeeze().abs()

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = go.Figure()
    for name, y in zip(["in-dir dotproduct", "importance"], [neuron_dotprod_with_in_dir.detach(), neuron_importances.detach()]):
        fig.add_trace(go.Bar(x=t.arange(model.d_model), y=y, name=name)) #, secondary_y=i==1)
    fig.update_layout(hovermode="x unified", template="simple_white", height=400, width=600, barmode="group", title=f"Layer {layer} neuron importance vs. similarity with v_dir")
    fig.show()

    # print((neuron_dotprod_with_in_dir ** 2).mean())


if MAIN:
    for layer in range(2):
        plot_neurons_vs_similarity_with_v_dir(model, data, layer)
# %%