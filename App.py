import streamlit as st

st.set_page_config(
    page_title="PyTorch Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title('Fashion Classification')

uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'jpeg'])