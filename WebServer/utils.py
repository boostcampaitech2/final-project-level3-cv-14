import streamlit as st
from PIL import Image


@st.cache
def load_img(uploaded_file):
    return Image.open(uploaded_file)