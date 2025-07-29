import streamlit as st
from PIL import Image
import torch
import os
from utils import load_model, cartoonize

import os
import streamlit as st

st.write("当前工作目录：", os.getcwd())
st.write("模型文件存在吗？", os.path.exists("model/generator.pth"))
print("模型文件存在吗？", os.path.exists("model/generator.pth"))

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
