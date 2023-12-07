import os
import sys
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

from rendering import *
from nerf import *
from utils import *

from dataset import KlevrDataset


torch.backends.cudnn.benchmark = True

img_wh = (256, 256)

dataset = KlevrDataset("./dataset", split="val")

embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)

nerf_coarse = NeRF()
nerf_fine = NeRF()

ckpt_path = "best_model.ckpt"

load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

nerf_coarse.cuda().eval()
nerf_fine.cuda().eval()

models = [nerf_coarse, nerf_fine]
embeddings = [embedding_xyz, embedding_dir]

N_samples = 64
N_importance = 64
use_disp = False
chunk = 1024*32*4


@torch.no_grad()
def f(rays):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back=True,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


selected = []
for i in np.random.randint(0, len(dataset), 6):
    print(i)
    selected.append(dataset[i])
count = 0
for sample in selected:
    rays = sample['rays'].cuda()

    results = f(rays)
    torch.cuda.synchronize()
    img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
    alpha_pred = results['opacity_fine'].view(
        img_wh[1], img_wh[0]).cpu().numpy()
    depth_pred = results['depth_fine'].view(img_wh[1], img_wh[0])

    plt.imsave(os.path.join(
        "./plot", f"{count}.png"), visualize_depth(depth_pred).permute(1, 2, 0).numpy())
    count += 1
