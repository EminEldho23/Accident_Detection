import requests
import os
from ultralytics import YOLO
from PIL import Image

# 1. Download Image
# Using a collision image from Wikimedia
url = "https://upload.wikimedia.org/wikipedia/commons/a/a4/Car_crash_1.jpg"
img_dir = "data/test/images"
os.makedirs(img_dir, exist_ok=True)
img_path = os.path.join(img_dir, "testing1.jpg") 

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

print(f"Downloading sample image to {img_path}...")
try:
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    with open(img_path, 'wb') as f:
        f.write(r.content)
    print("Download success.")
except Exception as e:
    print(f"Failed to download image: {e}")
    print("Creating dummy image instead.")
    Image.new('RGB', (640, 640), color=(128, 128, 128)).save(img_path)

# 2. Validation
print("\n--- Running Validation ---")
try:
    model = YOLO("best.pt")
    # We use data_local.yaml which points to our downloaded image
    model.val(data="data_local.yaml")
except Exception as e:
    print(f"Validation error: {e}")

# 3. Prediction (Folder)
print("\n--- Running Prediction (Folder: data/test/images) ---")
try:
    model.predict(source=img_dir, conf=0.25, save=True)
except Exception as e:
    print(f"Prediction error: {e}")

# 4. Prediction (Single File)
print(f"\n--- Running Prediction (File: {img_path}) ---")
try:
    model.predict(source=img_path, save=True)
except Exception as e:
    print(f"Prediction error: {e}")
