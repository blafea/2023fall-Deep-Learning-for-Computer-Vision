import os
import sys
import csv
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from p2_dataloader import p2_dataset

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

input_csv = sys.argv[1]
input_path = sys.argv[2]
output_csv = sys.argv[3]

# dataset&transformation
test_dataset = p2_dataset(
    input_path,
    transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


# model architecture
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None)
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
model.load_state_dict(torch.load("./p2_model/best_classifier_53627.pth"))
model.eval()

name2label = dict()
with torch.no_grad():
    for data, filenames in tqdm(test_loader):
        img = data.to(device)
        logits = model(img)
        y_pred = torch.argmax(logits, dim=1)
        for filename, pred in zip(filenames, y_pred):
            name2label[filename] = int(pred)

with open(input_csv, "r") as incsv:
    with open(output_csv, "w") as outcsv:
        reader = csv.reader(incsv)
        writer = csv.writer(outcsv)
        next(reader)
        writer.writerow(("id", "filename", "label"))
        for id, filename, _ in reader:
            writer.writerow((id, filename, name2label[filename]))
