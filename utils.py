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
import cv2

def cartoonize(model, image):
    """
    image: PIL.Image or np.ndarray (H,W,C, RGB uint8)
    return: PIL.Image (RGB uint8)
    """
    # 1. 确认类型
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            # OpenCV BGR转RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image or np.ndarray")

    # 2. 转成RGB模式确保没问题
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 3. 预处理
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(image).unsqueeze(0)  # [1,3,H,W]

    # 4. 推理
    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().clamp(-1, 1)  # [-1,1]
    output = (output * 0.5 + 0.5).clamp(0, 1)  # 归一化到[0,1]

    # 5. 转回PIL
    output_np = (output.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(output_np)
