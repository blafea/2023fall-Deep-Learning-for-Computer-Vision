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

voc_cls = {
    "urban": 0,
    "rangeland": 2,
    "forest": 3,
    "unknown": 6,
    "barreb land": 5,
    "Agriculture land": 1,
    "water": 4,
}
cls_color = {
    0: [0, 255, 255],
    1: [255, 255, 0],
    2: [255, 0, 255],
    3: [0, 255, 0],
    4: [0, 0, 255],
    5: [255, 255, 255],
    6: [0, 0, 0],
}
cmap = cls_color


def mask_edge_detection(mask, edge_width):
    h = mask.shape[0]
    w = mask.shape[1]

    edge_mask = np.zeros((h, w))

    for i in range(h):
        for j in range(1, w):
            j_prev = j - 1
            # horizontal #
            if not mask[i][j] == mask[i][j_prev]:  # horizontal
                if mask[i][j] == 1:  # 0 -> 1
                    edge_mask[i][j] = 1
                    for add in range(1, edge_width):
                        if j + add < w and mask[i][j + add] == 1:
                            edge_mask[i][j + add] = 1

                else:  # 1 -> 0
                    edge_mask[i][j_prev] = 1
                    for minus in range(1, edge_width):
                        if j_prev - minus >= 0 and mask[i][j_prev - minus] == 1:
                            edge_mask[i][j_prev - minus] = 1
            # vertical #
            if not i == 0:
                i_prev = i - 1
                if not mask[i][j] == mask[i_prev][j]:
                    if mask[i][j] == 1:  # 0 -> 1
                        edge_mask[i][j] = 1
                        for add in range(1, edge_width):
                            if i + add < h and mask[i + add][j] == 1:
                                edge_mask[i + add][j] = 1
                    else:  # 1 -> 0
                        edge_mask[i_prev][j] = 1
                        for minus in range(1, edge_width):
                            if i_prev - minus >= 0 and mask[i_prev - minus][j] == 1:
                                edge_mask[i_prev - minus][j] == 1
    return edge_mask


def viz_data(im, seg, color, inner_alpha=0.3, edge_alpha=1, edge_width=5):
    edge = mask_edge_detection(seg, edge_width)

    color_mask = np.zeros((edge.shape[0] * edge.shape[1], 3))
    l_loc = np.where(seg.flatten() == 1)[0]
    color_mask[l_loc, :] = color
    color_mask = np.reshape(color_mask, im.shape)
    mask = np.concatenate(
        (seg[:, :, np.newaxis], seg[:, :, np.newaxis], seg[:, :, np.newaxis]), axis=-1
    )

    color_edge = np.zeros((edge.shape[0] * edge.shape[1], 3))
    l_col = np.where(edge.flatten() == 1)[0]
    color_edge[l_col, :] = color
    color_edge = np.reshape(color_edge, im.shape)
    edge = np.concatenate(
        (edge[:, :, np.newaxis], edge[:, :, np.newaxis], edge[:, :, np.newaxis]),
        axis=-1,
    )

    im_new = im * (1 - mask) + im * mask * (1 - inner_alpha) + color_mask * inner_alpha
    im_new = (
        im_new * (1 - edge) + im_new * edge * (1 - edge_alpha) + color_edge * edge_alpha
    )

    return im_new


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


input_path = "./p3_plot_data"
output_path = ["./p3_plot_data/early", "./p3_plot_data/middle", "./p3_plot_data/final"]

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
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(path):
    model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 512, 3, 1, 1, bias=False),
        nn.BatchNorm2d(512, 1e-5),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(512, 7, 1),
    )
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


early_model = get_model("p3_B_model/1_model.pth")
middle_model = get_model("p3_B_model/40_model.pth")
final_model = get_model("p3_B_model/best_model.pth")
models = [early_model, middle_model, final_model]

for model, path in zip(models, output_path):
    model.eval()
    with torch.no_grad():
        for data, file in tqdm(test_loader):
            test_pred = model(data.to(device))["out"]
            test_pred = test_pred.argmax(dim=1)
            for i, file in zip(test_pred, file):
                masks = i.detach().cpu().numpy()
                img = imageio.imread(os.path.join(input_path, file))
                cs = np.unique(masks)

                for c in cs:
                    mask = np.zeros((img.shape[0], img.shape[1]))
                    ind = np.where(masks == c)
                    mask[ind[0], ind[1]] = 1
                    img = viz_data(img, mask, color=cmap[c])
                    imageio.imsave(
                        os.path.join(path, file.replace(".jpg", ".png")), np.uint8(img)
                    )
