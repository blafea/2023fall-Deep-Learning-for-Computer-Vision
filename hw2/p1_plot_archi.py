import torch
from torchview import draw_graph
from torch.utils.data import DataLoader
from torchvision import transforms

from p1_model import DDPM, ContextUnet
from p1_dataloader import p1_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
n_feat = 128
n_T = 400  # 500


train_set = p1_dataset(
    "./hw2_data/digits/mnistm/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    csv_path="./hw2_data/digits/mnistm/train.csv",
)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
ddpm = DDPM(
    nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=10),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1,
)
for x, c in train_loader:
    break
model_graph = draw_graph(ddpm, input_data=(x, c), device="cuda")
model_graph.visual_graph
