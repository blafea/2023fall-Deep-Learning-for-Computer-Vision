import os
import sys
import torch
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from p2_dataloader import p2_dataset
from byol_pytorch import BYOL


# config
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs = 500
lr = 3e-4
ckpt_path = "p2_model_false"
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

# dataset&transformation
dataset = p2_dataset(
    "./hw1_data/p2_data/mini/train",
    transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    ),
)
train_dataset, valid_dataset = random_split(
    dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))]
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# model architecture
model = resnet50(weights=None)
learner = BYOL(
    net=model,
    image_size=128,
    hidden_layer="avgpool",
).to(device)
optimizer = torch.optim.Adam(learner.parameters(), lr=lr)

# start training
for epoch in range(1, epochs + 1):
    for data, _ in tqdm(train_loader):
        img = data.to(device)
        loss = learner(img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()

    va_loss = []
    for data, _ in valid_loader:
        img = data.to(device)
        with torch.no_grad():
            loss = learner(img)
        loss = loss.item()
        va_loss.append(loss)
    va_loss = sum(va_loss) / len(va_loss)

    # save models
    print(f"Epoch {epoch}, validation loss: {va_loss}")
    if epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"{epoch}_model.pth"))
torch.save(model.state_dict(), os.path.join(ckpt_path, "final_model.pth"))
