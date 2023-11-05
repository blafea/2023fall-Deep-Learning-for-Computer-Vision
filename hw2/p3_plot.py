import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from p3_model import Extractor
from p3_dataloader import all_dataset


# config
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 512

ckpt_path = "./models/p3_model/svhn"
encoder = Extractor()
encoder.load_state_dict(torch.load(os.path.join(ckpt_path, "best_ecd.pth")))
encoder.to(device)

all_feat = []
all_label = []
all_domain = []
for target in ["svhn", "mnistm"]:
    
    if target == "svhn":
        mean, std = [0.4413, 0.4458, 0.4715], [0.1169, 0.1206, 0.1042]
    elif target == "usps":
        mean, std = [0.2573, 0.2573, 0.2573], [0.3373, 0.3373, 0.3373]
    else:
        mean, std = [0.4631, 0.4666, 0.4195], [0.1979, 0.1845, 0.2083]

    # dataset&transformation
    target_test_set = all_dataset(
        f"./hw2_data/digits/{target}/data",
        tfm=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        csv_path=f"./hw2_data/digits/{target}/val.csv",
    )
    target_test_loader = DataLoader(
        target_test_set, batch_size, shuffle=False, num_workers=4
    )
    if target == "mnistm":
        all_domain.extend([0]*len(target_test_set))
    else:
        all_domain.extend([1]*len(target_test_set))

    # model
    
    encoder.eval()

    for img, label in tqdm(target_test_loader):
        img = img.to(device)
        with torch.no_grad():
            feature = encoder(img)
        all_feat.extend(feature.cpu().numpy())
        all_label.extend(label.cpu().numpy())
all_feat = np.array(all_feat)
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
all_feat = TSNE().fit_transform(all_feat)
scatter = axes[0].scatter(
    all_feat[..., 0], all_feat[..., 1],
    c=all_label, alpha=0.5, s=10
)
axes[0].legend(*scatter.legend_elements(), title='Classes')
axes[0].set_title("by Class", fontsize = 20)

scatter = axes[1].scatter(
    all_feat[..., 0], all_feat[..., 1],
    c=all_domain, alpha=0.5, s=10
)
axes[1].legend(handles=scatter.legend_elements()[0], labels=[
    'source', 'target'], title='Domains')
axes[1].set_title("by Domain", fontsize = 20)
fig.savefig('SVHN.png')