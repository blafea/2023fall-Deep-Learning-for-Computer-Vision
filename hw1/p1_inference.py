import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class test_dataset(Dataset):
    def __init__(self, path, tfm):
        self.images = []
        self.labels = []
        self.tfm = tfm
        self.filenames = [
            filename for filename in os.listdir(path) if filename.endswith(".png")
        ]
        for filename in self.filenames:
            image = Image.open(os.path.join(path, filename))
            self.images.append(image.convert("RGB"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return self.tfm(self.images[index]), self.filenames[index]


path = sys.argv[1]
test_file = sys.argv[2]

mean = [0.5077, 0.4813, 0.4312]
std = [0.2000, 0.1986, 0.2034]
test_set = test_dataset(
    path,
    transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = efficientnet_v2_m()
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 50),
)
model.load_state_dict(torch.load("p1_B_model/best.ckpt"))
model.to(device)

filename, label = [], []
model.eval()
with torch.no_grad():
    for data, file in tqdm(test_loader):
        test_pred = model(data.to(device))
        test_label = torch.argmax(test_pred, axis=1)
        label += test_label.squeeze().tolist()
        filename += file

filename = np.array(filename, dtype=str)
label = np.array(label, dtype=int)
idx = np.argsort(filename)
filename = filename[idx]
label = label[idx]

df = pd.DataFrame()
df["filename"] = filename
df["label"] = label
df.to_csv(test_file, index=False)
