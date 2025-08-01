import streamlit as st
from PIL import Image
import torch
import os
import requests
from tqdm import tqdm
from utils import load_model, cartoonize, load_style_images

st.title("🧑‍🎨 Face2Cartoon with Local Style Dataset")

MODEL_URL = "https://github.com/cookie-519/cam25_group3/releases/download/v1.0/generator.pth"
MODEL_PATH = "model/generator.pth"

# 下载模型（如果不存在）
def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.write("📥 正在下载模型，请稍候...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("✅ 模型下载完成！")
    else:
        st.write("✅ 模型已存在。")

@st.cache_resource
def get_model():
    download_model()
    return load_model(MODEL_PATH)

@st.cache_data
def get_style_data():
    return load_style_images("dataset")

# 加载模型和风格图像
model = get_model()
style_images = get_style_data()

# 上传图像
uploaded_file = st.file_uploader("上传一张人脸图片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始图片", use_column_width=True)

    if st.button("生成卡通图像"):
        with st.spinner("处理中..."):
            output_img = cartoonize(model, image, style_images)
            st.image(output_img, caption="卡通图像", use_column_width=True)
