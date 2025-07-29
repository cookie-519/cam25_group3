import os
import sys
<<<<<<< HEAD

=======
>>>>>>> a92851e2b71abc8225b858373517e45f3dc4a76c
# 将工作目录切换为当前脚本文件所在的目录（兼容 Streamlit 启动方式）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)
<<<<<<< HEAD

print(">>> 当前工作目录:", os.getcwd())
print(">>> 模型文件存在吗？", os.path.exists("model/generator.pth"))

print(">>> 正常启动 app >>>")
=======
>>>>>>> a92851e2b71abc8225b858373517e45f3dc4a76c

import streamlit as st
from PIL import Image
from utils import load_model, cartoonize

st.set_page_config(page_title="Face2Cartoon", layout="centered")
st.title("🧑‍🎨 Face2Cartoon - Pix2Pix GAN")

#@st.cache_resource
def get_model():
<<<<<<< HEAD
    try:
        model = load_model('model/generator.pth', strict=False)  # 加了strict=False
        st.success("模型加载成功！")
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None
=======
    return load_model("model/generator_clean.pth")  # 加载新的模型文件
>>>>>>> a92851e2b71abc8225b858373517e45f3dc4a76c

model = get_model()
model = torch.load("model/generator.pth", map_location="cpu")

torch.save(model.state_dict(), "model/generator_clean.pth")



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
