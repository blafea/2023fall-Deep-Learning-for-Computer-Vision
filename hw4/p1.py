import os, sys
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
from nerf import Embedding, NeRF
from rendering import render_rays

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams_ = hparams

        self.loss = MSELoss()
        self.validation_step_outputs = []
        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
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
                            self.hparams_.chunk, # chunk size is effective in val mode
                            white_back=True)

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
                          batch_size=1, # validate one image (H*W rays) at a time
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
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        
        if batch_nb == 0:
            W, H = self.hparams_.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        self.validation_step_outputs.append(log)
        return log

    def on_validation_epoch_end(self):
        mean_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in self.validation_step_outputs]).mean()
        print(mean_loss.item(), mean_psnr.item())
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