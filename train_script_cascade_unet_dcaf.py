import os
import sys
import numpy as np
import h5py
import pylab as plt
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from k_space_reconstruction.nets.cdn import UnetDCAFModule, CascadeModule
from k_space_reconstruction.datasets.fastmri import FastMRITransform, FastMRIh5Dataset, RandomMaskFunc
from k_space_reconstruction.utils.metrics import pt_msssim, pt_ssim, ssim, nmse, psnr
from k_space_reconstruction.utils.loss import l1_loss, compund_mssim_l1_loss
from k_space_reconstruction.utils.kspace import spatial2kspace, kspace2spatial

# %env CUDA_VISIBLE_DEVICES=1
print('Available GPUs: ', torch.cuda.device_count())

torch.manual_seed(42)
np.random.seed(42)

path = 'cascade-x5-unet16-dcaf-noiseless.pth' #<------------------------Path-to-the-cascade-wegihts----------------------
batch_size = 1
accum_val = 32

model_kwargs = dict(
    unet_chans=16,
    unet_num_layers=4,
    criterion=compund_mssim_l1_loss,
    verbose_batch=50,
    optimizer='RMSprop',
    lr=3e-4,
    lr_step_size=5,
    lr_gamma=0.9,
    weight_decay=0.0
)

cascade = CascadeModule(net=torch.nn.ModuleList([UnetDCAFModule(**model_kwargs).net]), **model_kwargs)
def get_trainer():
    return pl.Trainer(
        gpus=1, max_epochs=1,
        accumulate_grad_batches=accum_val,
        terminate_on_nan=True,
        default_root_dir='logs/CascadeUNetAF',
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                save_top_k=4,
                monitor='val_loss',
                filename='{epoch}-{ssim:.4f}-{psnr:.4f}-{nmse:.5f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.GPUStatsMonitor(temperature=True)
        ]
    )
#---------------gaussian------------------------

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=100,
    noise_type='normal'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=12, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle = True)

cascade = CascadeModule(net=torch.nn.ModuleList([UnetDCAFModule(**model_kwargs).net for _ in range(5)]), **model_kwargs)
cascade.net.load_state_dict(torch.load(path))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=accum_val,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeUNetAF_gaussian',
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            save_last=True,
            save_top_k=4,
            monitor='val_loss',
            filename='{epoch}-{ssim:.4f}-{psnr:.4f}-{nmse:.5f}'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.GPUStatsMonitor(temperature=True)
    ]
)
for param in cascade.net.parameters():
    param.requires_grad = True
trainer.fit(cascade, train_dataloader=train_generator, val_dataloaders=val_generator)

torch.save(cascade.net.state_dict(), 'cascade-x5-unet16-dcaf-gaussian.pth')
print('Saved GAUSSIAN CASCADE model')

#---------------salt------------------------

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=5e4,
    noise_type='salt'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=12, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle = True)

cascade = CascadeModule(net=torch.nn.ModuleList([UnetDCAFModule(**model_kwargs).net for _ in range(5)]), **model_kwargs)
cascade.net.load_state_dict(torch.load(path))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=accum_val,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeUNetAF_salt',
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            save_last=True,
            save_top_k=4,
            monitor='val_loss',
            filename='{epoch}-{ssim:.4f}-{psnr:.4f}-{nmse:.5f}'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.GPUStatsMonitor(temperature=True)
    ]
)
for param in cascade.net.parameters():
    param.requires_grad = True
trainer.fit(cascade, train_dataloader=train_generator, val_dataloaders=val_generator)

torch.save(cascade.net.state_dict(), 'cascade-x5-unet16-dcaf-salt.pth')
print('Saved SALT CASCADE model')

#---------------both------------------------

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=5e4,
    noise_type='normal_and_salt'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=12, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle = True)

cascade = CascadeModule(net=torch.nn.ModuleList([UnetDCAFModule(**model_kwargs).net for _ in range(5)]), **model_kwargs)
cascade.net.load_state_dict(torch.load(path))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=accum_val,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeUNetAF_salt_normal_and_salt',
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            save_last=True,
            save_top_k=4,
            monitor='val_loss',
            filename='{epoch}-{ssim:.4f}-{psnr:.4f}-{nmse:.5f}'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.GPUStatsMonitor(temperature=True)
    ]
)
for param in cascade.net.parameters():
    param.requires_grad = True
trainer.fit(cascade, train_dataloader=train_generator, val_dataloaders=val_generator)

torch.save(cascade.net.state_dict(), 'cascade-x5-unet16-dcaf-normal-and-salt.pth')
print('Saved BOTH CASCADE model')