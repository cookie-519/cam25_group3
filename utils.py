import torch
from torchvision import transforms
from PIL import Image
from model_def import Generator  # 替换为实际的Generator定义路径

def load_style_images(style_dir="dataset"):
    from torchvision import transforms
    images = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    for fname in os.listdir(style_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            img = Image.open(os.path.join(style_dir, fname)).convert("RGB")
            images.append(transform(img))
    if images:
        return torch.stack(images)
    else:
        return None

def load_model(model_path):
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
