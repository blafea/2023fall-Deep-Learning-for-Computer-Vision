import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from p1_dataloader import p1_dataset

# config
myseed = 6666
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
n_epochs = 10
lr = 0.001
weight_decay = 5e-3
ckpt_path = "p1_B_model"

np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# dataset&transformation
mean = [0.5077, 0.4813, 0.4312]
std = [0.2000, 0.1986, 0.2034]
train_set = p1_dataset(
    "hw1_data/p1_data/train_50",
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=4)
valid_set = p1_dataset(
    "hw1_data/p1_data/val_50",
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
)
valid_loader = DataLoader(
    valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

# model architecture
model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 50),
)
# print(model)
model.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)


best_acc = 0

# start training
for epoch in range(1, n_epochs+1):
    # ---------- Training ----------
    model.train()
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    scheduler.step()
    print(
        f"[ Train | {epoch:02d}/{n_epochs:02d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}"
    )

    # ---------- Validation ----------
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(
        f"[ Valid | {epoch:02d}/{n_epochs:02d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
    )

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{ckpt_path}/best.ckpt")
        best_acc = valid_acc
