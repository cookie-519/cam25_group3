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

import torchvision.transforms as transforms
import streamlit as st  # 方便调试用

import torchvision.transforms as T
import numpy as np
import cv2

# utils.py
import numpy as np

def cartoonize(model, image):
    # 保险：确保是 PIL.Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image or np.ndarray")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),           # 现在不会崩
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(image).unsqueeze(0)  # [1,3,H,W]

    # 推理 & 转回 PIL
    with torch.no_grad():
        out = model(tensor).clamp(-1, 1) * 0.5 + 0.5
    out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(out)
