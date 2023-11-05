import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from p3_dataloader import all_dataset
from p3_model import Extractor, Classifier

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 200
batch_size = 512
lr = 0.01

# dataset&transformation
source_set = all_dataset(
    "./hw2_data/digits/usps/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    csv_path="./hw2_data/digits/usps/train.csv",
)
source_loader = DataLoader(source_set, batch_size, shuffle=True, num_workers=4)

target_set = all_dataset(
    "./hw2_data/digits/usps/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    csv_path="./hw2_data/digits/usps/val.csv",
)
target_loader = DataLoader(target_set, batch_size, shuffle=False, num_workers=4)

# model
encoder = Extractor().to(device)
classifier = Classifier().to(device)
optimizer = torch.optim.SGD(
    list(encoder.parameters()) + list(classifier.parameters()), lr=lr, momentum=0.9
)
criterion = nn.CrossEntropyLoss()

# start training
best_acc = 0
curr_steps = 0
total_steps = epochs * len(source_loader)
for epoch in range(epochs):
    encoder.train()
    classifier.train()

    for img, label in tqdm(source_loader):
        img = img.to(device)
        label = label.to(device)

        p = curr_steps / total_steps
        optimizer.param_groups[0]["lr"] = lr / (1.0 + 10 * p) ** 0.75

        source_feature = encoder(img)
        pred = classifier(source_feature)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        curr_steps += 1

    encoder.eval()
    classifier.eval()
    val_acc = 0
    for img, label in tqdm(target_loader):
        img = img.to(device)
        label = label.cpu().numpy()

        with torch.no_grad():
            target_feature = encoder(img)
            pred = classifier(target_feature)
        pred = pred.argmax(-1).cpu().numpy()
        val_acc += np.sum((pred == label).astype(int)) / len(pred)
    val_acc /= len(target_loader)
    best_acc = max(best_acc, val_acc)
    print(f"epoch: {epoch}, val acc: {val_acc}")
print(f"best_acc: {best_acc}")
