import os
import sys
import torch

import subprocess
import os

# 如果是云端环境（Streamlit Cloud），尝试拉取 LFS 文件
#if os.getenv("HOME") == "/home/adminuser":
 #st.info("⏳ 正在拉取 Git LFS 模型文件...")
  #  result = subprocess.getoutput("git lfs pull")
   # st.code(result)


# 将工作目录切换为当前脚本文件所在的目录（兼容 Streamlit 启动方式）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import streamlit as st
from PIL import Image
from utils import load_model, cartoonize

import os, subprocess, streamlit as st
st.write("LFS 拉取结果：")
st.code(subprocess.getoutput("git lfs ls-files && ls -lh model/"))

st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")
st.write("开始加载模型...")

#@st.cache_resource
def get_model():


    try:
        model = load_model("model/generator.pth")  # 加了strict=False
        model.load_state_dict(torch.load('model/generator.pth', map_location=torch.device('cpu')))
        st.success("模型加载成功 ✅")
        st.success("模型加载成功！")
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None
    return load_model("model/generator.pth")  # 加载新的模型文件

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
