import fire
import torch
import torch.nn as nn
import json
import torch.multiprocessing
from typing import Tuple
from pytorch_sound.models import build_model
from torch.utils.data import DataLoader
import os

from trainer.base_trainer import BaseTrainer
from dataset import Dataset

from opts import opt

# TODO: 
# * load checkpoint if global checkpoint status exists
# * set train max_step as loaded_checkpoint_step + max_step
def main(train_path: str, preprocessed_path: str,
         save_dir: str, save_prefix: str,
         model_name: str, pretrained_path: str = '', num_workers: int = 16,
         batch_size: int = 8,
         pitch_feature: str = 'phoneme', energy_feature: str = 'phoneme',
         pitch_min: float = 0., energy_min: float = 0.,
         lr: float = 2e-4, weight_decay: float = 0.0001, betas=(0.9, 0.98),
         max_step: int = 400000, group_size: int = 4,
         save_interval: int = 10000, log_interval: int = 50, grad_clip: float = 0.0, grad_norm: float = 5.0,
         milestones: Tuple[int] = None, gamma: float = 0.2, sr: int = 22050, seed: int = 2021,
         is_reference: bool = False):
    # create model
    model = build_model(model_name).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    else:
        scheduler = None

    dataset = Dataset(train_path, preprocessed_path, pitch_min=pitch_min, energy_min=energy_min,
                      text_cleaners=['english_cleaners'],
                      batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers
    )

    # train
    BaseTrainer(
        model, optimizer,
        train_loader, None,
        max_step=max_step, valid_max_step=1, save_interval=save_interval,
        log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path, sr=sr,
        scheduler=scheduler, seed=seed, is_reference=is_reference
    ).run()


# def run_config(config_path: str):
#     with open(config_path, 'r') as r:
#         config = json.load(r)
#     main(**config)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # fire.Fire(run_config)
    main(train_path=os.path.join(opt.current_data_path, opt.train_path),
         preprocessed_path=os.path.join(opt.current_data_path, opt.preprocessed_path),
         save_dir=os.path.join(opt.current_data_path, opt.model_save_path),
         save_prefix=opt.model_save_prefix,
         model_name=opt.model_name,
         pretrained_path=opt.pretrained_path,
         num_workers=opt.num_workers,
         batch_size=opt.batch_size,
         pitch_feature=opt.pitch_feature,
         energy_feature=opt.energy_feature,
         pitch_min=opt.pitch_min,
         energy_min=opt.energy_min,
         lr=opt.lr,
         weight_decay=opt.weight_decay,
         max_step=opt.max_step,
         group_size=opt.group_size,
         save_interval=opt.save_interval,
         log_interval=opt.log_interval,
         gamma=opt.gamma,
         sr=opt.sr,
         seed=opt.seed,
         is_reference=opt.is_reference)
