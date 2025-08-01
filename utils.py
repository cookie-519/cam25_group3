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

from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

def cartoonize(model, image):
    """
    image: PIL.Image or np.ndarray (H,W,C, RGB uint8)
    return: PIL.Image (RGB uint8)
    """
    # —— 1. 统一转成 PIL.Image ——
    if isinstance(image, np.ndarray):
        # OpenCV 读进来的 BGR 转 RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image or np.ndarray")

    # —— 2. 预处理 ——
    transform = T.Compose([
        T.Resize(256),               # 按需修改尺寸
        T.CenterCrop(256),
        T.ToTensor(),                # 现在安全了
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(image).unsqueeze(0)  # [1,3,H,W]

    # —— 3. 推理 ——
    with torch.no_grad():
        output = model(tensor)          # 假设模型输出也是 [-1,1]
    output = (output.squeeze().clamp(-1, 1) * 0.5 + 0.5)  # 归到 [0,1]

    # —— 4. 转回 PIL ——
    output_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(output_np)
