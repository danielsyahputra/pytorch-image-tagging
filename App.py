import streamlit as st
import numpy as np

from PIL import Image
from utils import MyPredictor

st.set_page_config(
    page_title="PyTorch Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title('Fashion Classification')

uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'jpeg'])

example_output = """
- dress
- black
- sleeves
"""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="text-align: center;">Your Image</p><hr>', unsafe_allow_html=True)
        st.image(image, caption="Fashion Image")
    with col2:
        st.markdown('<p style="text-align: center;">Image Tags</p><hr>', unsafe_allow_html=True)
        st.markdown("**Tags from Image:**")
        st.markdown(example_output)