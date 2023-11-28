import os, sys
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize

from tokenizer import BPETokenizer
from decoder import Transformer_encdec
from p2_dataloader import test_dataset
from p2_evaluate import CLIPScore


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "vit_large_patch14_clip_224.openai"
peft_type = "lora"

tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
model = Transformer_encdec(
    model_name, "./hw3_data/p2_data/decoder_model.bin", peft_type
)
state = model.load_state_dict(
    torch.load("lora_4/lora_10_0.8408_0.7231.pth"), strict=False
)
print(state.unexpected_keys)
assert len(state.unexpected_keys) == 0
model = model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize(
            size=224,
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        ),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.4815, 0.4578, 0.4082]),
            std=torch.tensor([0.2686, 0.2613, 0.2758]),
        ),
    ]
)
test_set = test_dataset("./hw3_data/p3_data/images", transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
feature = []


def visual_att(att_maps, ids, img_name, path):
    print(att_maps.shape)
    nrows = len(ids) // 5 if len(ids) % 5 == 0 else len(ids) // 5 + 1
    ncols = 5
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 8))
    feature_size = (16, 16)  # H/patch size, W/patch size = feature size
    img = Image.open(os.path.join(path, f"{img_name}.jpg"))
    size = img.size
    ax[0][0].imshow(img)
    ax[0][0].set_title("<|endoftext|>")
    for i in range(1, len(ids)):
        r, c = i // 5, i % 5
        attn_vector = att_maps[i, :, 1:]
        att_map = torch.reshape(attn_vector, feature_size)
        att_map -= torch.min(att_map)
        att_map /= torch.max(att_map)
        mask = resize(att_map.unsqueeze(0), [size[1], size[0]]).squeeze(0)
        mask = np.uint8(mask * 255)
        ax[r][c].imshow(img)
        ax[r][c].set_title(ids[i])
        ax[r][c].imshow(mask, alpha=0.7, cmap="jet")
    for i in range(nrows * ncols):
        r, c = i // 5, i % 5
        ax[r][c].axis("off")
    plt.savefig(img_name)
    plt.close()


model.eval()
# for img, name in tqdm(test_loader):
#     img = img.to(device)
#     features = []

#     def hook(module, input, output):
#         features.append(output[:, :, -1].detach().cpu())

#     handle = model.dec.transformer.h[3].cro_attn.att_map.register_forward_hook(hook)
#     with torch.no_grad():
#         output_ids = model.greedy_search(img)
#     tokens = tokenizer.decode(output_ids[0])
#     tokens = "<|endoftext|> " + tokens + " <|endoftext|>"
#     print(tokens)
#     # print(features[-1].shape)
#     # features = features[-1]
#     # print(features)
#     features = torch.stack(features).mean(dim=2)

#     # print(features.shape)
#     visual_att(features, tokens.split(" "), name[0], "hw3_data/p3_data/images")
#     handle.remove()


# get clip score top 1 and last 1
# with open("output2.json", "r") as f:
#     data = json.load(f)

# clip_eval = CLIPScore()
# top1_score = 0.0
# last1_score = 1.0
# for k, v in data.items():
#     image_path = os.path.join("./hw3_data/p2_data/images/val", f"{k}.jpg")
#     image = Image.open(image_path).convert("RGB")

#     score = clip_eval.getCLIPScore(image, v)
#     if score > top1_score:
#         top1_score = score
#         top1_name = k
#     if score < last1_score:
#         last1_score = score
#         last1_name = k
# print(top1_score, last1_score)
# print(top1_name, last1_name)  # 000000539189 000000428039

names = ["000000539189", "000000428039"]
for name in names:
    image_path = os.path.join("./hw3_data/p2_data/images/val", f"{name}.jpg")
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    features = []

    def hook(module, input, output):
        features.append(output[:, :, -1].detach().cpu())

    handle = model.dec.transformer.h[3].cro_attn.att_map.register_forward_hook(hook)
    with torch.no_grad():
        output_ids = model.greedy_search(img)
    tokens = tokenizer.decode(output_ids[0])
    tokens = "<|endoftext|> " + tokens + " <|endoftext|>"
    print(tokens)
    features = torch.stack(features).mean(dim=2)

    visual_att(features, tokens.split(" "), name, "./hw3_data/p2_data/images/val")
    handle.remove()
