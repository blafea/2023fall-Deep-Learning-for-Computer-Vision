import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from torchvision.models.segmentation.fcn import FCNHead
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from p3_dataloader import p3_dataset

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
epochs = 100
lr = 3e-4
weight_decay = 5e-4
ckpt_path = "p3_B_model"
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)
mean = [0.4085, 0.3785, 0.2809]
std = [0.1155, 0.0895, 0.0772]

# dataset&transformation
train_dataset = p3_dataset(
    "./hw1_data/p3_data/train",
    transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
valid_dataset = p3_dataset(
    "./hw1_data/p3_data/validation",
    transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


# miou function
def mean_iou_score(pred, labels):
    """
    Compute mean IoU score over 6 classes
    """
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6

    return mean_iou


# model architecture
model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
model.classifier = FCNHead(2048, 7)
print(model)
model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


best_mIoU = 0.0

# start training
for epoch in range(1, epochs + 1):
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)["out"]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        loss = 0
        preds = []
        gts = []
        for x, y in tqdm(valid_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)["out"]
            pred = out.argmax(dim=1)
            loss += nn.functional.cross_entropy(out, y).item()

            pred = pred.detach().cpu().numpy().astype(np.int64)
            y = y.detach().cpu().numpy().astype(np.int64)
            preds.append(pred)
            gts.append(y)

        loss /= len(valid_loader)
        mIoU = mean_iou_score(
            np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)
        )
    print(f"epoch {epoch}, loss = {loss}, mIoU = {mIoU}")

    # save models
    if mIoU > best_mIoU:
        best_mIoU = mIoU
        torch.save(model.state_dict(), os.path.join(ckpt_path, "best_model.pth"))
        print(f"Best model found at epoch {epoch}, saving model")
    if (epoch % 10) == 0 or epoch == 1:
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"{epoch}_model.pth"))

print(f"best mIoU: {best_mIoU}")
