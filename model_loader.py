import os
import requests

def download_model():
    url = "https://github.com/cookie-519/model-storage/raw/main/generator.pth"
    save_path = "model/generator.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        print("🔽 Downloading model...")
        r = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded.")
    else:
        print("✅ Model already exists.")
