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

for i in range(10):
    noise = torch.load(os.path.join(noise_path, f"{i:02d}.pt"))
    noise.to(device)
    with torch.no_grad():
        img = sample(model, noise)
        save_image(img, os.path.join(output_path, f"{i:02d}.png"))
