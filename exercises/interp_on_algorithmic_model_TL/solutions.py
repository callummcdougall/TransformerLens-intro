# %%

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
device = t.device("cuda")

t.set_grad_enabled(False)

import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig
from torchtyping import TensorType as TT

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

f = r"C:\Users\calsm\Documents\AI Alignment\ARENA\TRANSFORMERLENS_AND_MI\exercises\interp_on_algorithmic_model_TL\transcribed"
sys.path.append(f)
os.chdir(f)

from brackets_datasets import SimpleTokenizer, BracketsDataset

device = t.device("cpu")

# %%

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

# %%

if MAIN:
    tokenizer = SimpleTokenizer("()")
    N_SAMPLES = 5000
    with open("brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)
    data_tuples = data_tuples[:N_SAMPLES]
    data = BracketsDataset(data_tuples)

# %%

if MAIN:
    fig = go.Figure(
        go.Histogram(x=[len(x) for x, _ in data_tuples], nbinsx=data.seq_length),
        layout=dict(title="Sequence Lengths", xaxis_title="Sequence Length", yaxis_title="Count")
    )
    fig.show()

# %%

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

# %%

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

# %%

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

# %%

def is_balanced_forloop(parens: str) -> bool:

    cumsum = 0
    for paren in parens:
        cumsum += 1 if paren == "(" else -1
        if cumsum < 0:
            return False
    
    return cumsum == 0

if MAIN:
    for (tokens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")

# %%

def is_balanced_vectorized(tokens: TT["seq"]) -> bool:
    """
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    """
    # Convert start/end/padding tokens to zero, and left/right brackets to +1/-1
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    # Get altitude by taking cumulative sum
    altitude = t.cumsum(change, -1)
    # Check that the total elevation is zero and that there are no negative altitudes
    no_total_elevation_failure = altitude[-1] == 0
    no_negative_failure = altitude.min() >= 0

    return no_total_elevation_failure & no_negative_failure

if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")

# %%

def get_post_final_ln_dir(model: HookedTransformer) -> TT["d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    return model.W_U[:, 0] - model.W_U[:, 1]

# %%

# Solution using hooks:

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


if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, layer=-1, seq_pos=None)
    print("r^2: ", r2)
# %%

def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> TT["d_model"]:
    
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layer=-1, seq_pos=0)[0]
    final_ln_coefs = t.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir

# %%

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

# %%

if MAIN:
    biases = model.b_O.sum(0)
    out_by_components = get_out_by_components(model, data)
    summed_terms = out_by_components.sum(dim=0) + biases

    final_ln_input_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    final_ln_input = get_activations(model, data, final_ln_input_name)

    t.testing.assert_close(summed_terms, final_ln_input)

# %%

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

# %%

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

# %%

if MAIN:
    fig = px.scatter(
        x=data.open_proportion, y=h20_magnitudes, color=failure_types, 
        title="Head 2.0 contribution vs proportion of open brackets '('", template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}, category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()
# %%

def get_attn_probs(model: HookedTransformer, tokenizer: SimpleTokenizer, data: BracketsDataset, layer: int, head: int) -> t.Tensor:
    '''
    Returns: (batch, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    '''
    return get_activations(model, data, utils.get_act_name("pattern", layer))[:, head, :, :]

if MAIN:
    attn_probs = get_attn_probs(model, tokenizer, data, 2, 0)
    attn_probs_open = attn_probs[data.starts_open].mean(0)[0]
    px.bar(
        y=attn_probs_open, labels={"y": "Probability", "x": "Key Position"},
        template="simple_white", height=500, width=600, title="Avg Attention Probabilities for '(' query from query 0"
    ).update_layout(showlegend=False, hovermode='x unified').show()

# %%

def get_WOV(model: HookedTransformer, layer: int, head: int) -> TT["d_model", "d_model"]:
    '''
    Returns the W_OV matrix for a particular layer and head.
    '''
    return model.W_V[layer, head] @ model.W_O[layer, head]

def get_pre_20_dir(model, data) -> TT["d_model"]:
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    '''
    W_OV = get_WOV(model, 2, 0)

    layer2_ln_fit = get_ln_fit(model, data, layer=2, layernorm=1, seq_pos=1)[0]
    layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_).to(device)

    pre_final_ln_dir = get_pre_final_ln_dir(model, data)

    return layer2_ln_coefs.T @ W_OV @ pre_final_ln_dir

# %%

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

# %%

def mlp_attribution_scatter(magnitudes: TT["comp", "batch"], data: BracketsDataset, failure_types: np.ndarray):
    for layer in range(2):
        fig = px.scatter(
            x=data.open_proportion[data.starts_open], 
            y=magnitudes[3+layer*3, data.starts_open], 
            color=failure_types[data.starts_open], category_orders={"color": failure_types_dict.keys()},
            title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
            template="simple_white", height=500, width=800,
            labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
        ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
        fig.show()

if MAIN:
    mlp_attribution_scatter(magnitudes, data, failure_types)

# %%

def out_by_neuron(model: HookedTransformer, data: BracketsDataset, layer: int) -> TT["batch", "seq", "neurons", "d_model"]:
    '''
    [b, s, i]th element is the vector f(x.T @ W_in[:, i]) @ W_out[i, :] which is written to the residual stream by the ith neuron
    (where x is the input to the MLP for the b-th element in the batch, and the s-th sequence position).
    '''
    # Get the W_out matrix for this MLP
    W_out: TT["neurons", "d_model"] = model.blocks[layer].mlp.W_out

    # Get activations of the layer just after the activation function, i.e. this is f(x.T @ W_in)
    f_x_W_in: TT["batch", "seq", "neurons"] = get_activations(model, data, utils.get_act_name('post', layer))

    # Calculate the output by neuron (i.e. so summing over the `neurons` dimension gives the output of the MLP)
    out = einsum(
        "batch seq neurons, neurons d_model -> batch seq neurons d_model",
        f_x_W_in, W_out
    )
    return out

@functools.cache
def out_by_neuron_in_20_dir(model: HookedTransformer, data: BracketsDataset, layer: int) -> TT["batch", "seq", "neurons"]:
    '''
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the residual stream
    in the unbalanced direction (for the b-th element in the batch, and the s-th sequence position).
    
    In other words we need to take the vector produced by the `out_by_neuron` function, and project it onto the
    unbalanced direction for head 2.0.
    '''

    return einsum(
        "batch seq neurons d_model, d_model -> batch seq neurons",
        out_by_neuron(model, data, layer),
        get_pre_20_dir(model, data)
    )

# %%

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


# %%

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
        fwd_hooks=[
            (utils.get_act_name("q", layer), lambda q, hook: q_inputs.append(q[:, :, head, :])),
            (utils.get_act_name("k", layer), lambda k, hook: k_inputs.append(k[:, :, head, :])),
        ]
    )

    # Return the queries and keys
    return q_inputs[0][0], k_inputs[0][0]

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
            labels={"x": "Key tokens (avg of left & right parens)", "y": "Query tokens (all left parens)"},
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
        return_type=None,
        fwd_hooks=[
            (utils.get_act_name("k", 0), functools.partial(hook_fn_patch_qk, new_value=k00_avg)),
            (utils.get_act_name("pattern", 0), hook_fn_display_attn_patterns),
        ]
    )

# %%
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

    def write_to_html(fig, filename):
        with open(f"{filename}.html", "w") as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

# %%

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

# %%


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


if MAIN:
    print("Avg squared cosine similarity of v_R with ...\n")

    cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
    print(f"...MLP input directions in layer 0:  {cos_sim_mlp0.pow(2).mean():.6f}")
   
    cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
    print(f"...MLP input directions in layer 1:  {cos_sim_mlp1.pow(2).mean():.6f}")
    
    cos_sim_rand = avg_squared_cos_sim(v_R)
    print(f"...random vectors of len = d_model:  {cos_sim_rand:.6f}")


# %%

if MAIN:
    print("Update the examples list below below find adversarial examples")
    examples = ["()", "(())", "))"]

    def simple_balanced_bracket(length: int) -> str:
        return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])
    
    examples.append(simple_balanced_bracket(15) + ")(" + simple_balanced_bracket(4))

    m = max(len(ex) for ex in examples)
    toks = tokenizer.tokenize(examples).to(device)
    logits = model(toks)[:, 0]
    prob_balanced = t.softmax(logits, dim=1)[:, 1]
    print("\n".join([f"{ex:{m}} -> {p:.4%} balanced confidence" for (ex, p) in zip(examples, prob_balanced)]))
# %%
