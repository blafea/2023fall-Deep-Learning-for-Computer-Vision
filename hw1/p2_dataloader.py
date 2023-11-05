import os
import torch
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class p2_dataset(Dataset):
    def __init__(self, path, tfm, label=None):
        self.images = []
        self.label = None
        self.tfm = tfm
        self.path = path
        self.filenames = [
            filename for filename in os.listdir(path) if filename.endswith(".jpg")
        ]
        if label is not None:
            self.label = dict()
            with open(label, "r") as f:
                rows = csv.reader(f)
                next(rows)
                for row in rows:
                    self.label[row[1]] = int(row[2])
        # print(sorted(set(self.label.values())))
        # print(len(set(self.label.values())))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.filenames[index]))
        if self.label is not None:
            label = self.label[self.filenames[index]]
            return self.tfm(image), label
        else:
            return self.tfm(image), self.filenames[index]


if __name__ == "__main__":
    p2_dst = p2_dataset(
        "/disks/local/l.h2t_1/r12922147/hw1_data/p2_data/office/train",
        transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        ),
        "/disks/local/l.h2t_1/r12922147/hw1_data/p2_data/office/train.csv",
    )
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for img, _ in p2_dst:
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    print(mean / len(p2_dst), std / len(p2_dst))
