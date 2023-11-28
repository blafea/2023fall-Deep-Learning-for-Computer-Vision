import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import loralib as lora

from tokenizer import BPETokenizer
from decoder import Transformer_encdec
from p2_dataloader import p2_dataset
from p2_evaluate import CIDERScore, CLIPScore

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-5
weight_decay = 1e-5
batch_size = 64
epochs = 10
model_name = "vit_large_patch14_clip_224.openai"
peft_type = "lora"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
model = Transformer_encdec(model_name, "./hw3_data/p2_data/decoder_model.bin", peft_type)
for param in model.enc.parameters():
    param.requires_grad = False
if peft_type == "lora":
    lora.mark_only_lora_as_trainable(model)
elif peft_type == "prefix":
    for param in model.dec.parameters():
        param.requires_grad = False
    for name, param in model.dec.named_parameters():
        if "prefix" in name:
            param.requires_grad = True
elif peft_type == "adapter":
    for param in model.dec.parameters():
        param.requires_grad = False
    for name, param in model.dec.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
for name, param in model.dec.named_parameters():
    if ("cro_attn" in name) or ("ln_3" in name):
        param.requires_grad = True

trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad]
print(trainable_weights)
print(f"PEFT type: {peft_type}, Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
model = model.to(device)

transform_config = resolve_data_config(
    model=timm.create_model(model_name, pretrained=True, num_classes=0),
)
augmentation_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
    ]
)
transform = create_transform(**transform_config)
train_set = p2_dataset(
    "./hw3_data/p2_data/images/train",
    "./hw3_data/p2_data/train.json",
    transforms.Compose([augmentation_transforms, transform]),
    tokenizer,
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=train_set.collate_fn,
)
valid_set = p2_dataset(
    "./hw3_data/p2_data/images/val",
    "./hw3_data/p2_data/val.json",
    transform,
    tokenizer,
)
valid_loader = DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs * len(train_loader)
)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
cider_eval = CIDERScore()
clip_eval = CLIPScore()
optimizer.zero_grad(set_to_none=True)
optimizer.step()

step = 0
best_cider, best_clip = 0, 0
for epoch in range(epochs):
    total_loss = []
    model.train()
    for data in tqdm(train_loader):
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        data["image"] = data["image"].to(device)
        data["token_ids"] = data["token_ids"].to(device)
        # print(data['token_ids'], data['image'])
        pred = model(data["token_ids"], data["image"])

        loss = criterion(pred, data["token_ids"][:, 1:])
        # print(pred.shape, data["token_ids"][:, 1:].shape)
        if step % 100 == 0:
            print(loss.item())
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        step += 1
    print(f"loss: {sum(total_loss)/len(total_loss)}")

    model.eval()
    all_ans = []
    all_preds = []
    all_names = []
    for data in tqdm(valid_loader):
        data["image"] = data["image"].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
                output_ids = model.greedy_search(data["image"])
        for i in range(len(output_ids)):
            pred = tokenizer.decode(output_ids[i])
            all_preds.append(pred)
        all_ans.extend(data["caption"])
        for img in range(len(data["image"])):
            all_names.append(data["filenames"][img][:-4])

    cider_score = cider_eval(
        dict(zip(all_names, all_preds)), dict(zip(all_names, all_ans))
    )
    clip_score = clip_eval(
        dict(zip(all_names, all_preds)), "./hw3_data/p2_data/images/val"
    )
    print(f"epoch {epoch+1}: CIDEr score={cider_score}, CLIP score={clip_score}")

    if clip_score > 0.71 and clip_score > best_clip:
        model.train()
        weights = dict()
        save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
        torch.save(
            save_weights,
            f"./models/{peft_type}_{epoch+1}_{cider_score:.4f}_{clip_score:.4f}.pth",
        )
        best_clip = clip_score
        best_cider = cider_score
