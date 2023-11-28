import os, sys
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tokenizer import BPETokenizer
from decoder import Transformer_encdec
from p2_dataloader import test_dataset

folder_path = sys.argv[1]
output_file = sys.argv[2]
decoder_weights = sys.argv[3]

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "vit_large_patch14_clip_224.openai"
peft_type = "lora"


tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
model = Transformer_encdec(model_name, decoder_weights, peft_type)
state = model.load_state_dict(torch.load("best_model.pth"), strict=False)
# assert len(state.unexpected_keys) == 0
model = model.to(device)

test_set = test_dataset(
    folder_path,
    transforms.Compose(
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
    ),
)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
preds = dict()

model.eval()
for img, name in tqdm(test_loader):
    img = img.to(device)
    with torch.no_grad():
        output_ids = model.beam_search(img, beams=3)
    # for i in range(len(output_ids)):
    pred = tokenizer.decode(output_ids)
    preds[name[0]] = pred
    # print(pred)

with open(output_file, "w") as f:
    json.dump(preds, f, indent=2)
