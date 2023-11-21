import loralib as lora
import timm
import os
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tokenizer import BPETokenizer
from decoder import Decoder, Config
from p2_dataloader import p2_dataset
from PIL import Image
from p2_evaluate import CIDERScore, CLIPScore

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
weight_decay = 1e-5
batch_size = 64
epochs = 20
model_name = "vit_large_patch14_clip_224.openai"
peft_type = "lora"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Transformer_encdec(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, tgt, x):
        x = self.enc.forward_features(x)
        x = self.dec(tgt, x, x)
        return x

    def greedy_search(self, img, max_length=30):
        self.eval()
        with torch.no_grad():
            memory = self.enc.forward_features(img)

        current_state = torch.tensor([50256] * img.shape[0]).reshape((-1, 1)).to(device)
        for _ in range(max_length):
            with torch.no_grad():
                x = current_state
                x = torch.narrow(x, 1, 0, min(x.size(1), self.dec.block_size))
                pos = torch.arange(
                    x.size()[1], dtype=torch.long, device=x.device
                ).unsqueeze(0)
                x = self.dec.transformer.wte(x) + self.dec.transformer.wpe(pos)
                # print(x.shape)
                x, key, val = self.dec.transformer.h((x, memory, memory))
                x = self.dec.lm_head(self.dec.transformer.ln_f(x)[:, -1])
            next_word = x.argmax(dim=-1).unsqueeze(1)
            current_state = torch.concat((current_state, next_word), dim=1)
        current_state = current_state.cpu().numpy()
        preds = []
        for sentence in current_state:
            count = 0
            for pos in range(len(sentence)):
                if sentence[pos] == 50256:
                    count += 1
                if count == 2:
                    break
            preds.append(sentence[1:pos])

        return preds


tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
encoder = timm.create_model(model_name, pretrained=True)
decoder = Decoder(Config("./hw3_data/p2_data/decoder_model.bin"))
model = Transformer_encdec(encoder, decoder)
for param in model.parameters():
    param.requires_grad = False
if peft_type == "lora":
    lora.mark_only_lora_as_trainable(model)
elif peft_type == "prefix":
    for name, param in model.dec.named_parameters():
        if "prefix" in name:
            param.requires_grad = True
elif peft_type == "adapter":
    for name, param in model.dec.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
for name, param in model.dec.named_parameters():
    if "cro_attn" in name:
        param.requires_grad = True

# for name, param in model.dec.named_parameters():
#     # if "cro_attn" in name:
#     #     print(name)
#     if "cro_attn" not in name:
#         param.requires_grad = False
print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
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
    collate_fn=valid_set.collate_fn,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs * len(train_loader)
)
scaler = torch.cuda.amp.GradScaler(enabled=True)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
cider_eval = CIDERScore()
clip_eval = CLIPScore()
optimizer.zero_grad(set_to_none=True)
optimizer.step()

step = 0
for epoch in range(epochs):
    total_loss = []
    model.train()
    for data in tqdm(train_loader):
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        data["image"] = data["image"].to(device)
        data["token_ids"] = data["token_ids"].to(device)
        # print(data['token_ids'], data['image'])
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            pred = model(data["token_ids"], data["image"])

        loss = criterion(pred, data["token_ids"][:, 1:])
        # print(pred.shape, data["token_ids"][:, 1:].shape)
        if step % 100 == 0:
            print(loss.item())
        total_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        step += 1
    print(f"loss: {sum(total_loss)/len(total_loss)}")

    model.eval()
    all_ans = []
    all_preds = []
    all_names = []
    for data in tqdm(valid_loader):
        data["image"] = data["image"].to(device)
        with torch.no_grad():
            output_ids = model.greedy_search(data["image"])
        for i in range(len(output_ids)):
            pred = tokenizer.decode(output_ids[i])
            all_preds.append(pred)
        all_ans.extend(data["caption"])
        for img in range(len(data["image"])):
            all_names.append(data["filenames"][img][:-4])

    # print(all_preds)
    # import json
    # dt = dict()
    # for pred, name in zip(all_ans, all_names):
    #     dt[name[:-4]] = pred
    # dt = json.dumps(dt, indent=2)
    # # print(dt)
    # with open("out.json", 'w') as f:
    #     f.write(dt)

    cider_score = cider_eval(
        dict(zip(all_names, all_preds)), dict(zip(all_names, all_ans))
    )
    clip_score = clip_eval(
        dict(zip(all_names, all_preds)), "./hw3_data/p2_data/images/val"
    )
    print(f"epoch {epoch+1}: CIDEr score={cider_score}, CLIP score={clip_score}")
