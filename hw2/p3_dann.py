import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from p3_dataloader import all_dataset
from p3_model import Extractor, Classifier, Discriminator

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 100
batch_size = 512
lr = 0.1
target = "svhn"
ckpt_path = f"./models/p3_model/{target}"
if target == "svhn":
    mean, std = [0.4413, 0.4458, 0.4715], [0.1169, 0.1206, 0.1042]
else:
    mean, std = [0.2573, 0.2573, 0.2573], [0.3373, 0.3373, 0.3373]

# dataset&transformation
source_set = all_dataset(
    "./hw2_data/digits/mnistm/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4631, 0.4666, 0.4195], [0.1979, 0.1845, 0.2083]),
        ]
    ),
    csv_path="./hw2_data/digits/mnistm/train.csv",
)
source_loader = DataLoader(source_set, batch_size, shuffle=True, num_workers=4)

target_train_set = all_dataset(
    f"./hw2_data/digits/{target}/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
    csv_path=f"./hw2_data/digits/{target}/train.csv",
)
target_train_loader = DataLoader(
    target_train_set, batch_size, shuffle=True, num_workers=4
)

target_valid_set = all_dataset(
    f"./hw2_data/digits/{target}/data",
    tfm=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
    csv_path=f"./hw2_data/digits/{target}/val.csv",
)
target_valid_loader = DataLoader(
    target_valid_set, batch_size, shuffle=False, num_workers=4
)

# model
encoder = Extractor().to(device)
classifier = Classifier().to(device)
discriminator = Discriminator().to(device)
optimizer = torch.optim.SGD(
    list(encoder.parameters())
    + list(classifier.parameters())
    + list(discriminator.parameters()),
    lr=lr,
    momentum=0.9,
)
cls_criterion = nn.CrossEntropyLoss()
dis_criterion = nn.CrossEntropyLoss()

# start training
best_acc = 0
curr_steps = 0
total_steps = epochs * len(source_loader)
for epoch in range(epochs):
    encoder.train()
    classifier.train()
    discriminator.train()

    for (src_img, src_label), (tar_img, tar_label) in tqdm(
        zip(source_loader, target_train_loader), total=len(source_loader)
    ):
        src_img = src_img.to(device)
        src_label = src_label.to(device)
        tar_img = tar_img.to(device)
        tar_label = tar_label.to(device)

        p = curr_steps / total_steps
        optimizer.param_groups[0]["lr"] = lr / (1.0 + 10 * p) ** 0.75
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        combined_image = torch.cat((src_img, tar_img), 0)
        combined_feature = encoder(combined_image)
        source_feature = encoder(src_img)

        # 1.Classification loss
        class_pred = classifier(source_feature)
        class_loss = cls_criterion(class_pred, src_label)

        # 2. Domain loss
        domain_pred = discriminator(combined_feature, alpha)

        domain_source_labels = torch.zeros(src_label.shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(tar_label.shape[0]).type(torch.LongTensor)
        domain_combined_label = torch.cat(
            (domain_source_labels, domain_target_labels), 0
        ).to(device)

        domain_loss = dis_criterion(domain_pred, domain_combined_label)
        total_loss = class_loss + domain_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        curr_steps += 1

    encoder.eval()
    classifier.eval()
    discriminator.eval()
    val_acc = 0
    for img, label in tqdm(target_valid_loader):
        img = img.to(device)
        label = label.cpu().numpy()

        with torch.no_grad():
            target_feature = encoder(img)
            pred = classifier(target_feature)
        pred = pred.argmax(-1).cpu().numpy()
        val_acc += np.sum((pred == label).astype(int)) / len(pred)
    val_acc /= len(target_valid_loader)
    print(f"epoch: {epoch+1}, val acc: {val_acc}")

    # save models
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(encoder.state_dict(), os.path.join(ckpt_path, "best_ecd.pth"))
        torch.save(classifier.state_dict(), os.path.join(ckpt_path, "best_cls.pth"))
        torch.save(discriminator.state_dict(), os.path.join(ckpt_path, "best_dis.pth"))
        print("model saved")

print(f"best_acc: {best_acc}")
