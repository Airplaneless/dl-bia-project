import os
import sys
import numpy as np
import h5py
import pylab as plt
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from k_space_reconstruction.nets.cdn_dncnn import PureDnCNNDCModule, CascadeModule
from k_space_reconstruction.datasets.fastmri import FastMRITransform, FastMRIh5Dataset, RandomMaskFunc
from k_space_reconstruction.utils.metrics import pt_msssim, pt_ssim, ssim, nmse, psnr
from k_space_reconstruction.utils.loss import l1_loss, compund_mssim_l1_loss
from k_space_reconstruction.utils.kspace import spatial2kspace, kspace2spatial

# %env CUDA_VISIBLE_DEVICES=1
print('Available GPUs: ', torch.cuda.device_count())

torch.manual_seed(42)
np.random.seed(42)

model_kwargs = dict(
    dncnn_chans=64,
    dncnn_depth=10,
    criterion=compund_mssim_l1_loss,
    verbose_batch=50,
    optimizer='Adam',
    lr=1e-4,
    lr_step_size=3,
    lr_gamma=0.2,
    weight_decay=0.0
)

cascade = CascadeModule(net=torch.nn.ModuleList([PureDnCNNDCModule(**model_kwargs).net]), **model_kwargs)
cascade.net[0].cascade[0].load_state_dict(torch.load('dncnn-noiseless.pth'))


def get_trainer():
    return pl.Trainer(
        gpus=1, max_epochs=10,
        accumulate_grad_batches=3,
        terminate_on_nan=True,
        default_root_dir='logs/CascadeDnCNN_pure',
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                save_top_k=7,
                monitor='val_loss',
                filename='{epoch}-{ssim:.4f}-{psnr:.4f}-{nmse:.5f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.GPUStatsMonitor(temperature=True)
        ]
    )
#----------------noiseless----------------------

torch.manual_seed(42)
np.random.seed(42)

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=1000,
    noise_type='none'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=42, num_workers=12, shuffle=True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12)

num_blocks = 5
for i in range(num_blocks):
    # Train cascade block
    trainer = get_trainer()
    trainer.fit(cascade, train_dataloader=train_generator, val_dataloaders=val_generator)
    # Freeze last cascade blocks
    for param in cascade.net.parameters():
        param.requires_grad = False
    # Add new block to cascade
    if i != num_blocks - 1:
        cascade = CascadeModule(net=cascade.net.append(PureDnCNNDCModule(**model_kwargs).net), **model_kwargs)
        # Load statedict for unet in last trainable block
        cascade.net[-1].cascade[0].load_state_dict(torch.load('dncnn-noiseless.pth'))


transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=1000,
    noise_type='none'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=12, shuffle=True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12)


cascade = CascadeModule\
.load_from_checkpoint('logs/CascadeDnCNN_pure/lightning_logs/version_4/checkpoints/last.ckpt',
                      net=torch.nn.ModuleList([PureDnCNNDCModule(**model_kwargs).net for _ in range(5)]))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=16,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeDnCNN_pure',
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            save_last=True,
            save_top_k=7,
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
torch.save(cascade.net.state_dict(), 'cascade-x5-dncnn_pure-dc-noiseless.pth')

#---------------gaussian------------------------

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=100,
    noise_type='normal'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=12, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle = True)

cascade = CascadeModule(net=torch.nn.ModuleList([PureDnCNNDCModule(**model_kwargs).net for _ in range(5)]), **model_kwargs)
cascade.net.load_state_dict(torch.load('cascade-x5-dncnn_pure-dc-noiseless.pth'))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=16,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeDnCNN_pure_gaussian',
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

torch.save(cascade.net.state_dict(), 'cascade-x5-dncnn_pure-dc-gaussian.pth')
print('Saved GAUSSIAN CASCADE model')

#---------------salt------------------------

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=5e4,
    noise_type='salt'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=12, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle = True)

cascade = CascadeModule(net=torch.nn.ModuleList([PureDnCNNDCModule(**model_kwargs).net for _ in range(5)]), **model_kwargs)
cascade.net.load_state_dict(torch.load('cascade-x5-dncnn_pure-dc-noiseless.pth'))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=16,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeDnCNN_pure_salt',
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

torch.save(cascade.net.state_dict(), 'cascade-x5-dncnn_pure-dc-salt.pth')
print('Saved SALT CASCADE model')

#---------------both------------------------

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=5e4,
    noise_type='normal_and_salt'
)

train_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/train.h5', transform)
val_dataset = FastMRIh5Dataset('small_fastmri_pd_3t/val.h5', transform)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=12, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=12, shuffle = True)

cascade = CascadeModule(net=torch.nn.ModuleList([PureDnCNNDCModule(**model_kwargs).net for _ in range(5)]), **model_kwargs)
cascade.net.load_state_dict(torch.load('cascade-x5-dncnn_pure-dc-noiseless.pth'))

trainer = pl.Trainer(
    gpus=1, max_epochs=20,
    accumulate_grad_batches=16,
    terminate_on_nan=True,
    default_root_dir='logs/CascadeDnCNN_pure_salt_normal_and_salt',
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

torch.save(cascade.net.state_dict(), 'cascade-x5-dncnn_pure-dc-normal-and-salt.pth')
print('Saved BOTH CASCADE model')