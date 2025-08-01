import torch
from torchvision import transforms
from PIL import Image
from model_def import Generator  # 替换为实际的Generator定义路径
# 在本地执行一次，保存为 state_dict

def load_model(model_path, strict=True):
    model = Generator()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model

from torchvision import transforms

import torch
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st  # 方便调试用

def cartoonize(model, image):
    # 确认image是PIL图片
    if not isinstance(image, Image.Image):
        raise TypeError(f"传入的image不是PIL.Image对象，而是{type(image)}")
    # 确认是RGB图像
    if image.mode != "RGB":
        st.warning(f"图片模式是{image.mode}，已强制转换为RGB")
        image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    return output_image
