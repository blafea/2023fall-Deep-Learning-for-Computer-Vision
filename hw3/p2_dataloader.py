import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class p2_dataset(Dataset):
    def __init__(self, path, json_file, tfm, tokenizer):
        super().__init__()
        self.path = path
        self.tfm = tfm
        self.tokenizer = tokenizer

        with open(json_file, "r") as f:
            data = json.load(f)
            annotations = data["annotations"]
            images = data["images"]
        self.data = [
            {
                "caption": annotations[i]["caption"],
                "image_id": annotations[i]["image_id"],
            }
            for i in range(len(annotations))
        ]
        self.id2img = dict()
        for i in range(len(images)):
            self.id2img[images[i]["id"]] = images[i]["file_name"]

    def __getitem__(self, index):
        data = self.data[index]
        name = self.id2img[data["image_id"]]
        img = Image.open(os.path.join(self.path, name)).convert("RGB")
        img = self.tfm(img)

        return {"caption": data["caption"], "image": img, "filename": name}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        captions = []
        tokens = []
        filenames = []
        images = []
        for sample in samples:
            token = self.tokenizer.encode(sample["caption"])
            # print(token)
            tokens.append([50256] + token + [50256])
            # print(captions[-1])
            filenames.append(sample["filename"])
            images.append(sample["image"])
            captions.append(sample["caption"])
        max_len = max([len(t) for t in tokens])
        for i in range(len(tokens)):
            tokens[i].extend([-100] * (max_len - len(tokens[i])))
            # print(tokens[i])
        images = torch.stack(images, dim=0)
        return {
            "filenames": filenames,
            "token_ids": torch.tensor(tokens),
            "image": images,
            "caption": captions,
        }
