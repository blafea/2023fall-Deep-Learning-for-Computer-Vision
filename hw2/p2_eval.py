import imageio
from skimage.metrics import mean_squared_error

loss = 0
for i in range(10):
    gt = imageio.imread(f"./hw2_data/face/GT/{i:02d}.png")
    out = imageio.imread(f"./face_out/{i:02d}.png")
    print(mean_squared_error(gt, out))
    loss += mean_squared_error(gt, out)
loss /= 10
print(loss)
