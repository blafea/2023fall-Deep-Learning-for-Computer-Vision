import os
import csv
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class p1_dataset(Dataset):
    def __init__(self, path, tfm, csv_path):
        self.labels = dict()
        self.path = path
        self.tfm = tfm
        self.filenames = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for name, label in reader:
                self.labels[name] = int(label)
                self.filenames.append(name)
        with open("./hw2_data/digits/mnistm/val.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for name, label in reader:
                self.labels[name] = int(label)
                self.filenames.append(name)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.filenames[index])).convert(
            "RGB"
        )
        return self.tfm(image), self.labels[self.filenames[index]]
