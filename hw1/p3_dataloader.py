import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


class p3_dataset(Dataset):
    def __init__(self, path, tfm):
        self.tfm = tfm
        self.path = path
        self.sat_names = [
            filename for filename in os.listdir(path) if filename.endswith(".jpg")
        ]
        self.mask_names = [
            filename for filename in os.listdir(path) if filename.endswith(".png")
        ]
        self.sat_names = sorted(self.sat_names)
        self.mask_names = sorted(self.mask_names)

    def __len__(self):
        return len(self.sat_names)

    def __getitem__(self, index):
        image = Image.open(os.path.join(
            self.path, self.sat_names[index])).convert("RGB")
        mask = Image.open(os.path.join(
            self.path, self.mask_names[index])).convert("RGB")
        mask = np.array(mask)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

        raw_mask = deepcopy(mask)

        mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
        mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
        mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
        mask[raw_mask == 2] = 3  # (Green: 010) Forest land
        mask[raw_mask == 1] = 4  # (Blue: 001) Water
        mask[raw_mask == 7] = 5  # (White: 111) Barren land
        mask[raw_mask == 0] = 6  # (Black: 000) Unknown

        return self.tfm(image), torch.tensor(mask)


if __name__ == "__main__":
    p3_dst = p3_dataset(path="hw1_data/p3_data/train",
                        tfm=transforms.ToTensor())
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for img, _ in p3_dst:
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    print(mean / len(p3_dst), std / len(p3_dst))
