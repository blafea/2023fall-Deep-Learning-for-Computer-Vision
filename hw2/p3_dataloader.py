import os
import csv
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class all_dataset(Dataset):
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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.filenames[index])).convert(
            "RGB"
        )
        return self.tfm(image), self.labels[self.filenames[index]]


if __name__ == "__main__":
    dst = all_dataset(
        path="hw2_data/digits/mnistm/data",
        tfm=transforms.ToTensor(),
        csv_path="hw2_data/digits/mnistm/train.csv",
    )
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for img, _ in dst:
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    print(mean / len(dst), std / len(dst))

    dst = all_dataset(
        path="hw2_data/digits/svhn/data",
        tfm=transforms.ToTensor(),
        csv_path="hw2_data/digits/svhn/train.csv",
    )
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for img, _ in dst:
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    print(mean / len(dst), std / len(dst))

    dst = all_dataset(
        path="hw2_data/digits/usps/data",
        tfm=transforms.ToTensor(),
        csv_path="hw2_data/digits/usps/train.csv",
    )
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for img, _ in dst:
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    print(mean / len(dst), std / len(dst))
