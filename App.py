import streamlit as st

from PIL import Image
from streamlit_tags import st_tags
from utils import MyPredictor

st.set_page_config(
    page_title="Fashion Tagging",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown('## Auto Tagging for Fashion Retail Using Multi-label Image Classification')

uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'jpeg'])

predictor = MyPredictor()

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="text-align: center;">Your Image</p><hr>', unsafe_allow_html=True)
        st.image(image, caption="Fashion Image")
    with col2:
        st.markdown('<p style="text-align: center;">Image Tags</p><hr>', unsafe_allow_html=True)
        tags = predictor.predict(image=image)
        tags = st_tags(label="Tags:", text="Add more", value=tags)