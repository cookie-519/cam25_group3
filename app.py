import os
import sys
# 将工作目录切换为当前脚本文件所在的目录（兼容 Streamlit 启动方式）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import torch

model = torch.load("model/generator.pth", map_location="cpu")
model.eval()
#torch.save(model, "model/generator_compressed.pth", _use_new_zipfile_serialization=True)


import streamlit as st
from PIL import Image
import torch
import os
from utils import load_model, cartoonize

st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")

#@st.cache_resource
def get_model():
    try:
        print("开始加载模型...")
        model = load_model('model/generator.pth', strict=False)
        print("模型加载成功")
        return model
    except Exception as e:
        print("模型加载失败:", e)
        st.error(f"模型加载失败: {e}")
        return None


model = get_model()

uploaded_file = st.file_uploader("上传人脸图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始图片", use_column_width=True)

    if st.button("生成卡通图像"):
        with st.spinner("正在生成，请稍候..."):
            output_img = cartoonize(model, image)
            st.image(output_img, caption="卡通图像", use_column_width=True)
            output_img.save("output.png")

            with open("output.png", "rb") as f:
                st.download_button(label="下载卡通图像",
                                   data=f,
                                   file_name="cartoon_output.png",
                                   mime="image/png")
