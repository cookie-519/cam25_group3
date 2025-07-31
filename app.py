import os
import sys
import torch
import streamlit as st
from PIL import Image
from utils import load_model, cartoonize
from model_def import Generator  # 替换为你的文件名
from model_loader import download_model
download_model()

# 然后加载模型
model.load_state_dict(torch.load("model/generator.pth", map_location="cpu"))


# 将工作目录切换为当前脚本文件所在的目录（兼容 Streamlit 启动方式）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

model = Generator()
state_dict = torch.load('model/generator.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")
st.write("开始加载模型...")

#@st.cache_resource
def get_model():
    state_dict = torch.load('model/generator.pth', map_location='cpu')
    for k in state_dict.keys():
        print(k)

    try:

        model = load_model('model/generator.pth', strict=False)  # 加了strict=False
        model.load_state_dict(torch.load('model/generator.pth', map_location=torch.device('cpu')))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
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
