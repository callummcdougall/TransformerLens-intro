import plotly.express as px
from torchtyping import TensorType as TT

def plot_attn(v: TT["seq_Q", "seq_K"]):
    px.imshow(
        v, 
        title="Estimate for avg attn probabilities when query is from '('",
        labels={"x": "Key tokens", "y": "Query tokens"},
        height=1200, width=1200,
        color_continuous_scale="RdBu_r", range_color=[0, v.max().item() + 0.01]
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