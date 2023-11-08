import os, sys
import torch
import clip
import json, csv
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


folder = sys.argv[1]
id2label = sys.argv[2]
output_csv = sys.argv[3]

img_names = [name for name in os.listdir(folder) if name.endswith(".png")]
with open(id2label, 'r') as f:
    id2label = json.load(f)
text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in id2label.values()]).to(device)

all_names = []
all_labels = []
for name in img_names:
    image = preprocess(Image.open(os.path.join(folder, name))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    _, indices = similarity[0].topk(1)
    all_names.append(name)
    all_labels.append(int(indices))

with open(output_csv, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    for name, label in zip(all_names, all_labels):
        writer.writerow([name, label])