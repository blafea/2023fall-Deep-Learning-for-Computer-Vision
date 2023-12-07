from dataset import KlevrDataset
import matplotlib.pyplot as plt

dataset = KlevrDataset("./dataset/", split="val")

plt.imsave("aaa.png", dataset[0]["rgbs"].reshape((256, 256, 3)).numpy())
