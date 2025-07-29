import os
st.write("当前工作目录:", os.getcwd())
MODEL_PATH = "model/generator.pth"
st.write("模型路径:", MODEL_PATH)
st.write("模型绝对路径:", os.path.abspath(MODEL_PATH))
st.write("模型文件存在吗？", os.path.exists(MODEL_PATH))


import streamlit as st
from PIL import Image
import torch
import os
from utils import load_model, cartoonize

st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")

MODEL_PATH = "model/generator.pth"

def check_model_file():
    st.write("当前工作目录:", os.getcwd())
    st.write("模型路径:", MODEL_PATH)
    exists = os.path.exists(MODEL_PATH)
    st.write("模型文件是否存在:", exists)
    if not exists:
        st.error("模型文件不存在，请确认已上传到该路径！")
    return exists

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = load_model(MODEL_PATH)
    return model

def main():
    if not check_model_file():
        st.stop()  # 文件没找到就停止运行后续代码

    uploaded_file = st.file_uploader("上传人脸图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="原始图片", use_column_width=True)

        if st.button("生成卡通图像"):
            with st.spinner("正在生成，请稍候..."):
                model = get_model()
                if model is None:
                    st.error("模型加载失败！")
                    return
                output_img = cartoonize(model, image)
                st.image(output_img, caption="卡通图像", use_column_width=True)
                output_img.save("output.png")
                with open("output.png", "rb") as f:
                    st.download_button(label="下载卡通图像",
                                       data=f,
                                       file_name="cartoon_output.png",
                                       mime="image/png")

if __name__ == "__main__":
    main()
