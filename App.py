import streamlit as st

from PIL import Image
from streamlit_tags import st_tags
from utils import MyPredictor

def load_settings():
    # Page config
    st.set_page_config(
        page_title="Fashion Tagging",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.markdown('## Auto Tagging for Fashion Retail Using Multi-label Image Classification')
    
    # Sidebar
    sidebar = st.sidebar

    with open("assets/test.jpg", "rb") as file:
        btn = sidebar.download_button(
                label="Download sample image",
                data=file,
                file_name="test.jpg",
                mime="image/jpg"
            )

def main():
    load_settings()
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

if __name__=="__main__":
    main()