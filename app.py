import os
import sys
import subprocess
import torch
import urllib.request
from PIL import Image
import streamlit as st
from utils import load_model, cartoonize

# 设置工作目录为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

# 设置页面
st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")

# 创建 model 文件夹（如果不存在）
os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/generator3.pth"
MODEL_URL = "https://github.com/cookie-519/cam25_group3/releases/download/v1.0/generator.pth"

@st.cache_resource
def get_model():
    st.warning("🔽开始工作")
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning("🔽 模型文件未找到，正在从 GitHub 下载...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ 模型文件下载完成！")

        model = load_model(MODEL_PATH)
        st.success("✅ 模型加载成功！")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

# 显示模型文件大小和存在状态
with st.expander("🔍 模型文件状态"):
    st.code(subprocess.getoutput("ls -lh model/"))

model = get_model()

# 图片上传与处理
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
                st.download_button(
                    label="下载卡通图像",
                    data=f,
                    file_name="cartoon_output.png",
                    mime="image/png"
                )
