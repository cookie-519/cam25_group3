import os
import sys
import torch
import streamlit as st
from PIL import Image
from utils import cartoonize
from model_def import Generator  # 替换为你自己的模型定义文件

# 设置工作目录为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

# 设置页面基本信息
st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")
st.write("上传一张人脸图片，我将为你生成卡通风格图像。")

# 加载模型函数
@st.cache_resource
def load_generator_model():
    model_path = os.path.join("model", "generator.pth")

    if not os.path.exists(model_path):
        st.error("❌ 模型文件未找到，请确认 model/generator.pth 是否存在。")
        return None

    model = Generator()
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ 加载模型时发生错误: {e}")
        return None

# 获取模型
model = load_generator_model()

# 图片上传 + 推理部分
uploaded_file = st.file_uploader("📤 上传人脸图片", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始图片", use_column_width=True)

    if st.button("🎨 生成卡通图像"):
        with st.spinner("⏳ 正在生成卡通图像，请稍候..."):
            cartoon_image = cartoonize(model, image)
            st.image(cartoon_image, caption="卡通图像", use_column_width=True)

            # 下载按钮
            cartoon_image.save("output.png")
            with open("output.png", "rb") as f:
                st.download_button("⬇️ 下载卡通图像", data=f, file_name="cartoon_output.png", mime="image/png")
