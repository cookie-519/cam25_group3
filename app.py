import streamlit as st
from PIL import Image
import torch
import os
from utils import load_model, cartoonize, load_style_images

st.title("🧑‍🎨 Face2Cartoon with Local Style Dataset")

@st.cache_resource
def get_model():
    return load_model("model/generator.pth")

@st.cache_data
def get_style_data():
    return load_style_images("dataset")

model = get_model()
style_images = get_style_data()  # 模型可能会使用这个风格信息

uploaded_file = st.file_uploader("上传一张人脸图片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始图片", use_column_width=True)

    if st.button("生成卡通图像"):
        with st.spinner("处理中..."):
            output_img = cartoonize(model, image, style_images)  # 传入风格数据
            st.image(output_img, caption="卡通图像", use_column_width=True)
