import torch
from torchvision.utils import save_image

from UNet import UNet
from p2_diffusion import sample, sample_interpolation

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet()
model.load_state_dict(torch.load("./hw2_data/face/UNet.pt"))
model.to(device)

for i in range(4):
    noise = torch.load(f"./hw2_data/face/noise/{i:02d}.pt")
    noise.to(device)
    for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with torch.no_grad():
            img = sample(model, noise, eta)
            save_image(img, f"./p2_plot/{i:02d}_{eta}.png")

noise1 = torch.load("./hw2_data/face/noise/00.pt")
noise2 = torch.load("./hw2_data/face/noise/01.pt")
imgs = sample_interpolation(model, noise1, noise2)

dim = 256 * 11 + 5 * 12
out = torch.zeros((3, 256 + 5, dim))
w = 5
for i in range(11):
    out[:, 5 : 256 + 5, w : w + 256] = imgs[i].cpu()
    w += 256 + 5
save_image(out, "./p2_plot/2.png")
