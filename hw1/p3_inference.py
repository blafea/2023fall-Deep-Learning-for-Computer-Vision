import os
import sys
import torch
import torch.nn as nn
import numpy as np
import imageio
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class test_dataset(Dataset):
    def __init__(self, path, tfm):
        self.sat_images = []
        self.tfm = tfm
        self.sat_names = [
            filename for filename in os.listdir(path) if filename.endswith(".jpg")
        ]
        self.sat_names = sorted(self.sat_names)
        for filename in self.sat_names:
            image = Image.open(os.path.join(path, filename))
            self.sat_images.append(image.convert("RGB"))

    def __len__(self):
        return len(self.sat_names)

    def __getitem__(self, index):
        return self.tfm(self.sat_images[index]), self.sat_names[index]


input_path = sys.argv[1]
output_path = sys.argv[2]

mean = [0.4085, 0.3785, 0.2809]
std = [0.1155, 0.0895, 0.0772]
test_set = test_dataset(
    input_path,
    transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Conv2d(2048, 512, 3, 1, 1, bias=False),
    nn.BatchNorm2d(512, 1e-5),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Conv2d(512, 7, 1),
)
model.load_state_dict(torch.load("p3_B_model/best_model.pth"))
model.to(device)

model.eval()
with torch.no_grad():
    for data, file in tqdm(test_loader):
        test_pred = model(data.to(device))["out"]
        test_pred = test_pred.argmax(dim=1)
        for i, file in zip(test_pred, file):
            pred = i.detach().cpu().numpy()
            masks = np.zeros((512, 512, 3), dtype=np.uint8)
            # (Cyan: 011) Urban land
            masks[np.where(pred == 0)] = [0, 255, 255]
            # (Yellow: 110) Agriculture land
            masks[np.where(pred == 1)] = [255, 255, 0]
            # (Purple: 101) Rangeland
            masks[np.where(pred == 2)] = [255, 0, 255]
            # (Green: 010) Forest land
            masks[np.where(pred == 3)] = [0, 255, 0]
            masks[np.where(pred == 4)] = [0, 0, 255]  # (Blue: 001) Water
            # (White: 111) Barren land
            masks[np.where(pred == 5)] = [255, 255, 255]
            masks[np.where(pred == 6)] = [0, 0, 0]  # (Black: 000) Unknown
            imageio.imwrite(
                os.path.join(output_path, file.replace("sat.jpg", "mask.png")), masks
            )
