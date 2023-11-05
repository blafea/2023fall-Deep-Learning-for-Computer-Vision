import os
import sys
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from p2_dataloader import p2_dataset

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
setting = "e"
batch_size = 32
epochs = 500
lr = 1e-3
weight_decay = 5e-3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
ckpt_path = "p2_model"
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

# dataset&transformation
train_dataset = p2_dataset(
    "./hw1_data/p2_data/office/train",
    transforms.Compose(
        [
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((128, 128)),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
    label="./hw1_data/p2_data/office/train.csv",
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=12
)

valid_dataset = p2_dataset(
    "./hw1_data/p2_data/office/val",
    transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
    label="./hw1_data/p2_data/office/val.csv",
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


# model architecture
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None)
        if setting == "a":
            print("setting a: no pretrain/train full model")
        elif setting == "b":
            print("setting b: use given pretrain model/train full model")
            self.backbone.load_state_dict(
                torch.load("./hw1_data/p2_data/pretrain_model_SL.pt")
            )
        elif setting == "c":
            print("setting c: use my pretrain model/train full model")
            self.backbone.load_state_dict(torch.load("./p2_model/500_model.pth"))
        elif setting == "d":
            print("setting d: use given pretrain model/train clf only")
            self.backbone.load_state_dict(
                torch.load("./hw1_data/p2_data/pretrain_model_SL.pt")
            )
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False
        elif setting == "e":
            print("setting e: use my pretrain model/train clf only")
            self.backbone.load_state_dict(torch.load("./p2_model/500_model.pth"))
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False

        self.clf = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 65),
        )

    def forward(self, img):
        embeds = self.backbone(img)
        logits = self.clf(embeds)
        return logits


model = Classifier().to(device)
# print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_acc = 0
# start training
for epoch in range(1, epochs + 1):
    model.train()
    train_acc = []
    for data, label in tqdm(train_loader):
        img = data.to(device)
        label = label.to(device)
        logits = model(img)
        loss = criterion(logits, label)

        y_pred = torch.argmax(logits, dim=1)
        train_acc.append(torch.mean((y_pred == label).type(torch.float)).item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_acc = sum(train_acc) / len(train_acc)
    model.eval()
    va_loss = []
    va_acc = []
    for data, label in valid_loader:
        img = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            logits = model(img)
            va_loss.append(torch.nn.functional.cross_entropy(logits, label).item())
        y_pred = torch.argmax(logits, dim=1)
        va_acc.append(torch.mean((y_pred == label).type(torch.float)).item())

    va_loss = sum(va_loss) / len(va_loss)
    va_acc = sum(va_acc) / len(va_acc)
    model.train()

    print(f"epoch {epoch}: train acc = {train_acc}, valid acc = {va_acc}")
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), os.path.join(ckpt_path, "best_classifier.pth"))
        print("Saved model")

print(f"best acc = {best_acc}")
