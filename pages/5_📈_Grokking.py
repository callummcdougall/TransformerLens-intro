import os
if not os.path.exists("./images"):
    os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

if is_local or check_password():
    st_image("wheel.png", 350)
    st.markdown("Coming soon!")
