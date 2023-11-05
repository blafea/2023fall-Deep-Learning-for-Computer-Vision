import os
import sys
import csv
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from p3_model import Extractor, Classifier, Discriminator


class all_dataset(Dataset):
    def __init__(self, path, tfm):
        self.labels = dict()
        self.path = path
        self.tfm = tfm
        self.filenames = [
            filename
            for filename in sorted(os.listdir(path))
            if filename.endswith(".png")
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.filenames[index])).convert(
            "RGB"
        )
        return self.tfm(image), self.filenames[index]


# config
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 512


path = sys.argv[1]
out = sys.argv[2]

if "svhn" in path:
    target = "svhn"
else:
    target = "usps"
print(f"predicting {target} dataset")
ckpt_path = f"./models/p3_model/{target}"
if target == "svhn":
    mean, std = [0.4413, 0.4458, 0.4715], [0.1169, 0.1206, 0.1042]
else:
    mean, std = [0.2573, 0.2573, 0.2573], [0.3373, 0.3373, 0.3373]

# dataset&transformation
target_test_set = all_dataset(
    path,
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
)
target_test_loader = DataLoader(
    target_test_set, batch_size, shuffle=False, num_workers=4
)

# model
encoder = Extractor()
encoder.load_state_dict(torch.load(os.path.join(ckpt_path, "best_ecd.pth")))
encoder.to(device)
classifier = Classifier()
classifier.load_state_dict(torch.load(os.path.join(ckpt_path, "best_cls.pth")))
classifier.to(device)
discriminator = Discriminator()
discriminator.load_state_dict(torch.load(os.path.join(ckpt_path, "best_dis.pth")))
discriminator.to(device)

# start predicting
encoder.eval()
classifier.eval()
discriminator.eval()

all_pred = []
all_name = []
for img, name in tqdm(target_test_loader):
    img = img.to(device)
    with torch.no_grad():
        target_feature = encoder(img)
        pred = classifier(target_feature)
    pred = pred.argmax(-1).cpu().numpy()
    all_pred.extend(pred)
    all_name.extend(name)

with open(out, "w") as f:
    writer = csv.writer(f)
    writer.writerow(("image_name", "label"))
    for name, pred in zip(all_name, all_pred):
        writer.writerow((name, pred))
