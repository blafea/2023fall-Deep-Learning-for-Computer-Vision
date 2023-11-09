import os, sys
import torch
import clip
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


folder = sys.argv[1]
id2label = sys.argv[2]

img_names = [name for name in os.listdir(folder) if name.endswith(".png")]
with open(id2label, 'r') as f:
    id2label = json.load(f)
text = torch.cat([clip.tokenize(f"a photo of {c}") for c in id2label.values()]).to(device)


for i in range(3):
    name = np.random.choice(img_names)
    image = preprocess(Image.open(os.path.join(folder, name))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    values, indices = values.cpu().numpy(), indices.cpu().numpy()

    image = Image.open(os.path.join(folder, name))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(f"GT: {id2label[name.split('_')[0]]}")
    sns.barplot(x=values*100, y=[f"a photo of {c}" for c in [id2label[str(index)] for index in indices]], ax=axes[1])
    axes[1].tick_params(axis="y", direction="in", pad=-150)
    plt.savefig(f"{i+1}.png")





