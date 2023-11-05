import pandas as pd
import numpy as np

gt = pd.read_csv("./hw2_data/digits/svhn/val.csv")
gt = dict(zip(gt["image_name"], gt["label"]))
pred = pd.read_csv("out.csv")
pred = dict(zip(pred["image_name"], pred["label"]))
count = 0
for k, v in gt.items():
    if pred[k] == v:
        count += 1
count /= len(gt)
print(count)
