import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vgg16_bn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from p1_dataloader import p1_dataset

# config
myseed = 6666
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 512
n_epochs = 300
lr = 0.001
weight_decay = 1e-2
patience = 20
ckpt_path = "p1_A_model"
plot_epochs = {50 * i for i in range(1, 6)} | {1}

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
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=4)
valid_set = p1_dataset(
    "hw1_data/p1_data/val_50",
    transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
)
valid_loader = DataLoader(
    valid_set, batch_size=batch_size, shuffle=False, num_workers=4)


# model architecture
class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.backbone = vgg16_bn()
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 512),
        )
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 50)
        )

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        return self.fc2(out)

    def sec_last(self, x):
        out = self.backbone(x)
        out = out.view(out.size()[0], -1)
        return self.fc1(out)


model = Vgg()
# print(model)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)


stale = 0
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
        f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}"
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
        f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
    )

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{ckpt_path}/best.ckpt")
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(
                f"No improvment {patience} consecutive epochs, early stopping")
            break

    # plot PCA&tSNE
    model.train()
    if epoch in plot_epochs:
        with torch.no_grad():
            x_list = None
            y_list = None
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                if x_list is None:
                    x_list = model.sec_last(x).detach().cpu().numpy()
                    y_list = y.detach().cpu().numpy().flatten()
                else:
                    out = model.sec_last(x).detach().cpu().numpy()
                    y = y.detach().cpu().numpy().flatten()
                    x_list = np.vstack((x_list, out))
                    y_list = np.concatenate((y_list, y))

        # plot PCA
        pca = PCA(n_components=2)
        pca_x = pca.fit_transform(x_list)
        plt.figure()
        plt.scatter(pca_x[:, 0], pca_x[:, 1], c=y_list)
        plt.title(f"PCA of epoch {epoch}")
        plt.savefig(f"./p1_figure/PCA_{epoch}")

        # plot t-SNE
        tsne = TSNE(n_components=2)
        tsne_x = tsne.fit_transform(x_list)
        plt.figure()
        plt.scatter(tsne_x[:, 0], tsne_x[:, 1], c=y_list)
        plt.title(f"t-SNE of epoch {epoch}")
        plt.savefig(f"./p1_figure/tSNE_{epoch}")

print("best acc:", best_acc)
