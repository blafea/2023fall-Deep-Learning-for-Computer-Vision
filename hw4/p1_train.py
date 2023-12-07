import os
import sys
from opt import get_opts
import torch
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from dataset import KlevrDataset

# models
from nerf import *
from rendering import *
from utils import *

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams_ = hparams

        self.loss = MSELoss()
        self.validation_step_outputs = []
        self.embedding_xyz = Embedding(3, 10)  # 10 is the default number
        self.embedding_dir = Embedding(3, 4)  # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams_.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams_.chunk],
                            self.hparams_.N_samples,
                            self.hparams_.use_disp,
                            self.hparams_.perturb,
                            self.hparams_.noise_std,
                            self.hparams_.N_importance,
                            self.hparams_.chunk,  # chunk size is effective in val mode
                            white_back=False)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        self.train_dataset = KlevrDataset("./dataset/", split="train")
        self.val_dataset = KlevrDataset("./dataset/", split="val")

    def configure_optimizers(self):
        eps = 1e-8
        parameters = []
        for model in self.models:
            parameters += list(model.parameters())
        self.optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                              weight_decay=self.hparams_.weight_decay)
        scheduler = MultiStepLR(self.optimizer, milestones=self.hparams_.decay_step,
                                gamma=self.hparams_.decay_gamma)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams_.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
                }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            W, H = self.hparams_.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(
                results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        self.validation_step_outputs.append(log)
        return log

    def on_validation_epoch_end(self):
        mean_loss = torch.stack([x['val_loss']
                                for x in self.validation_step_outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr']
                                for x in self.validation_step_outputs]).mean()
        print(f"loss: {mean_loss.item()}, psnr: {mean_psnr.item()}")
        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
                }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      num_sanity_val_steps=1,
                      benchmark=True,)

    trainer.fit(system)
