import torch
from torchvision import transforms
from PIL import Image
from model_def import Generator  # 替换为实际的Generator定义路径

def load_model(model_path, strict=True):
    model = Generator()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model

def cartoonize(model, image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    return output_image
