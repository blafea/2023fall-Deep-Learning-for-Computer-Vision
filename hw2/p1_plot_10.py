import numpy as np
from PIL import Image
import imageio

dim = 10 * 28 + 11 * 2
out = np.zeros((dim, dim, 3), dtype=np.uint8)
h = 2
for i in range(10):
    w = 2
    for j in range(10):
        image = Image.open(f"./test_path/{i}_{j+1:03d}.png")
        out[h : h + 28, w : w + 28, :] = np.array(image)
        w += 30
    h += 30

imageio.imsave("aaa.png", out)
