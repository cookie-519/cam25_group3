import os
import sys
import torch
import subprocess
from PIL import Image
import streamlit as st
from utils import load_model, cartoonize

# 设置工作目录为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

# 显示 LFS 状态
st.write("LFS 拉取结果：")
st.code(subprocess.getoutput("git lfs ls-files && ls -lh model/"))

# 页面设置
st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")

# 加载模型
@st.cache_resource
def get_model():
    try:
        model = load_model("model/generator.pth")
        model.load_state_dict(torch.load('model/generator.pth', map_location=torch.device('cpu')))
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = get_model()
if model:
    st.success("模型加载成功 ✅")

# 图片上传与显示
uploaded_file = st.file_uploader("上传人脸图片", type=["jpg", "jpeg", "png"])
image_placeholder = st.empty()
button_placeholder = st.empty()
output_placeholder = st.empty()

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_placeholder.image(image, caption="原始图片", use_column_width=True)

    if button_placeholder.button("生成卡通图像"):
        with st.spinner("正在生成，请稍候..."):
            output_img = cartoonize(model, image)
            output_placeholder.image(output_img, caption="卡通图像", use_column_width=True)

            output_img.save("output.png")
            with open("output.png", "rb") as f:
                st.download_button(
                    label="下载卡通图像",
                    data=f,
                    file_name="cartoon_output.png",
                    mime="image/png"
                )
