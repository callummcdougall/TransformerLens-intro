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
#     st.markdown(r"""
# Links to Colab: [**exercises**](https://colab.research.google.com/drive/1LpDxWwL2Fx0xq3lLgDQvHKM5tnqRFeRM?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1ND38oNmvI702tu32M74G26v-mO5lkByM?usp=share_link)
# """)
    st_image("sampling.png", 350)
    st.markdown(r"""
# Training and Sampling

Coming soon!
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
    st.markdown(r"""
# Training

Coming soon!
""")

def section_sampling():
    st.markdown(r"""
# Sampling and Caching

Coming soon!
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
