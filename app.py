import os
import streamlit as st

model_path = "model/generator.pth"
st.write("当前路径：", os.getcwd())
st.write("模型文件存在吗？", os.path.exists(model_path))

import streamlit as st
from PIL import Image
import torch
import os
from utils import load_model, cartoonize

import os
import streamlit as st
import torch

st.write("当前工作目录:", os.getcwd())

model_path = "model/generator.pth"
st.write("模型路径:", model_path)
st.write("模型文件是否存在:", os.path.exists(model_path))

# 下面加载模型，如果文件不存在会报错
if os.path.exists(model_path):
    # 这里的GeneratorModel需要你自己的模型定义
    model = GeneratorModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    st.write("模型加载成功")
else:
    st.error("模型文件不存在，无法加载")


st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")

@st.cache_resource
def get_model():
    return load_model('model/generator.pth')

model = get_model()

uploaded_file = st.file_uploader("上传人脸图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始图片", use_column_width=True)

    if st.button("生成卡通图像"):
        with st.spinner("正在生成，请稍候..."):
            output_img = cartoonize(model, image)
            st.image(output_img, caption="卡通图像", use_column_width=True)
            output_img.save("output.png")

            with open("output.png", "rb") as f:
                btn = st.download_button(label="下载卡通图像",
                                         data=f,
                                         file_name="cartoon_output.png",
                                         mime="image/png")
