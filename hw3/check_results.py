import pandas as pd
import numpy as np

gt = pd.read_csv("out.csv")
filename = np.array([file.split("_")[0] for file in gt["filename"]], dtype=int)
label = gt["label"]

print(np.sum(filename == label) / len(label))