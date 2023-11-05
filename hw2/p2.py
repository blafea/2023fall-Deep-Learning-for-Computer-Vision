import os
import sys
import torch
from torchvision.utils import save_image

from UNet import UNet
from p2_diffusion import sample

device = "cuda" if torch.cuda.is_available() else "cpu"

noise_path = sys.argv[1]
output_path = sys.argv[2]
model_path = sys.argv[3]

model = UNet()
model.load_state_dict(torch.load(model_path))
model.to(device)

noise_names = [filename for filename in os.listdir(noise_path) if filename.endswith(".pt")]

for noise_name in noise_names:
    noise = torch.load(os.path.join(noise_path, noise_name))
    noise.to(device)
    with torch.no_grad():
        img = sample(model, noise)
        save_image(img, os.path.join(output_path, noise_name.replace(".pt", ".png")))
