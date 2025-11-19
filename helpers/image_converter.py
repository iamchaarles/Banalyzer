import os
from PIL import Image

input_folder = 'banana_images/green banana'
output_folder = 'Ripeness_png/unripe'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.webp', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGB')
        # Remove extension and add .png
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_folder, f"{base_name}.png")
        img.save(out_path, 'PNG')
        print(f"Converted {filename} to {out_path}")
