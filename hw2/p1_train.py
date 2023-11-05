import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from p1_model import DDPM, ContextUnet

from p1_dataloader import p1_dataset

# https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
myseed = 7777
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 20
n_T = 500
n_feat = 128
batch_size = 256
lr = 1e-4

np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

train_set = p1_dataset(
    "./hw2_data/digits/mnistm/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    csv_path="./hw2_data/digits/mnistm/train.csv",
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

ddpm = DDPM(
    nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=10),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1,
)
ddpm.to(device)

# optionally load a model
# ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

for epoch in range(epochs):
    print(f"epoch {epoch+1}")
    ddpm.train()

    # linear lrate decay
    optimizer.param_groups[0]["lr"] = lr * (1 - epoch / epochs)

    pbar = tqdm(train_loader)
    loss_ema = None
    for x, c in pbar:
        optimizer.zero_grad()
        x = x.to(device)
        c = c.to(device)
        loss = ddpm(x, c)
        loss.backward()
        loss_ema = loss.item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optimizer.step()

torch.save(ddpm.state_dict(), "model.pth")
