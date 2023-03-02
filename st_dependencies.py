import streamlit as st
import base64
import platform
# from st_on_hover_tabs import on_hover_tabs

is_local = (platform.processor() != "")

def st_image(name, width):
    with open("images/page_images/" + name, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
    st.markdown(img_html, unsafe_allow_html=True)

def st_excalidraw(name, width):
    img_html_full = ""
    for suffix in ["light", "dark"]:
        with open("images/" + name + "-" + suffix + ".png", "rb") as file:
            img_bytes = file.read()
        encoded = base64.b64encode(img_bytes).decode()
        img_html = f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' class='img-fluid {suffix}Excalidraw' src='data:image/png;base64,{encoded}'>"
        img_html_full += img_html
    st.markdown(img_html_full, unsafe_allow_html=True)

def styling():
    st.set_page_config(layout="wide", page_icon="🔬")
    st.markdown(r"""
<style>
div[data-testid="column"] {
    background-color: #f9f5ff;
    padding: 15px;
}
.stAlert h4 {
    padding-top: 0px;
}
.st-ae code {
    padding: 0px !important;
}
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
}
p {
    line-height:1.48em;
}
.st-ae h2 {
    margin-top: -15px;
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
code:not(pre code) {
    color: red !important;
}
pre code {
    white-space: pre-wrap !important;
}
.st-ae code {
    padding: 4px;
}
.css-ffhzg2 .st-ae code: not(stCodeBlock) {
    background-color: black;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange !important;
    background-color: rgb(26, 28, 36);
}
.css-ffhzg2 code:not(pre code) {
    color: orange !important;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
.css-fg4pbf .lightExcalidraw {
    height: initial
}
.css-fg4pbf .darkExcalidraw {
    height: 0
}
.css-ffhzg2 .lightExcalidraw {
    height: 0
}
.css-ffhzg2 .darkExcalidraw {
    height: initial
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:18px;
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
ul {
    margin-bottom: 15px !important;
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

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True