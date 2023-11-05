import os
import sys
import numpy as np
import torch

from p1_model import DDPM, ContextUnet
from torchvision.utils import save_image

myseed = 7777
device = "cuda" if torch.cuda.is_available() else "cpu"
n_feat = 128
n_T = 500
path = sys.argv[1]

np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

ddpm = DDPM(
    nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=10),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1,
)
ddpm.load_state_dict(torch.load("./models/p1_model.pth"))

ddpm.eval()
for i in range(10):
    count = 0
    with torch.no_grad():
        x_i, x_store = ddpm.sample(100, size=(3, 28, 28), device=device, c=i)
    for image in x_i:
        count += 1
        save_image(image, os.path.join(path, f"{i}_{count:03d}.png"))
    # if i == 0:
    #     x_store = torch.tensor(x_store)
    #     print(x_store.shape)
    #     for j in range(len(x_store)):
    #         save_image(x_store[j][0], f"{j}.png")
    print(f"class {i} done")
