import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class p1_dataset(Dataset):
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
            label = int(filename.split("_")[0])
            self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return self.tfm(self.images[index]), self.labels[index]


if __name__ == "__main__":
    p1_dst = p1_dataset(path="hw1_data/p1_data/train_50",
                        tfm=transforms.ToTensor())
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for img, _ in p1_dst:
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    print(mean / len(p1_dst), std / len(p1_dst))
