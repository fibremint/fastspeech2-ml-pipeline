import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import torch
import torch.multiprocessing
import torch.nn as nn
from fastspeech2.dataset import Dataset
from pytorch_sound.models import build_model
from torch.utils.data import DataLoader

from opts import opt
from trainer.evaluator import Evaluator


def main(current_data_path: str, checkpoint_stat_path: str, optimal_checkpoint_stat_path: str,
         evaluate_path: str, preprocessed_path: str,
         save_dir: str, save_prefix: str,
         model_name: str, pretrained_path: str = '', num_workers: int = 16,
         batch_size: int = 8,
         pitch_feature: str = 'phoneme', energy_feature: str = 'phoneme',
         pitch_min: float = 0., energy_min: float = 0.,
         lr: float = 2e-4, weight_decay: float = 0.0001, betas=(0.9, 0.98),
         max_step: int = 400000, validate_max_step: int = 100, group_size: int = 4,
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

    dataset = Dataset(evaluate_path, preprocessed_path, pitch_min=pitch_min, energy_min=energy_min,
                      text_cleaners=['english_cleaners'],
                      batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers
    )

    # train
    checkpoint_stats = Evaluator(
        model, optimizer,
        None, test_loader,
        current_data_path=current_data_path,
        max_step=1, valid_max_step=validate_max_step, save_interval=save_interval,
        log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path, sr=sr,
        scheduler=scheduler, seed=seed, is_reference=is_reference
    ).run()

    import operator

    optimal_checkpoint_path, optimal_checkpoint_loss = sorted(checkpoint_stats.items(), key=operator.itemgetter(1))[0]

    optimal_checkpoint = {
        'path': optimal_checkpoint_path,
        'loss': optimal_checkpoint_loss
    }

    with open(os.path.join(current_data_path, checkpoint_stat_path), 'w') as f:
        json.dump(checkpoint_stats, f, indent=2)

    shutil.copy(os.path.join(current_data_path, checkpoint_stat_path), os.path.join('/tmp', checkpoint_stat_path))

    with open(os.path.join(current_data_path, optimal_checkpoint_stat_path), 'w') as f:
        json.dump(optimal_checkpoint, f, indent=2)

    shutil.copy(os.path.join(current_data_path, optimal_checkpoint_stat_path), os.path.join('/tmp', optimal_checkpoint_stat_path))

    current_data_path = Path(current_data_path)
    finished_data_path = current_data_path.parent / '-'.join(current_data_path.stem.split('-')[:-1])
    shutil.move(current_data_path, finished_data_path)

    with open('/tmp/finished-data-path.txt', 'w') as f:
        f.write(str(finished_data_path))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # fire.Fire(run_config)
    main(current_data_path=opt.current_data_path,
         checkpoint_stat_path=opt.checkpoint_stat_path,
         optimal_checkpoint_stat_path=opt.optimal_checkpoint_stat_path,
         evaluate_path=os.path.join(opt.current_data_path, opt.evaluate_path),
         preprocessed_path=os.path.join(opt.current_data_path, opt.preprocessed_path),
         save_dir=os.path.join(opt.current_data_path, opt.model_save_path),
         save_prefix=opt.model_save_prefix,
         model_name=opt.model_name,
         pretrained_path=os.path.join(opt.current_data_path, opt.model_save_path, opt.pretrained_path),
         num_workers=opt.num_workers,
         batch_size=opt.batch_size,
         pitch_feature=opt.pitch_feature,
         energy_feature=opt.energy_feature,
         pitch_min=opt.pitch_min,
         energy_min=opt.energy_min,
         lr=opt.lr,
         weight_decay=opt.weight_decay,
         max_step=opt.max_step,
         validate_max_step=opt.validate_max_step,
         group_size=opt.group_size,
         save_interval=opt.save_interval,
         log_interval=opt.log_interval,
         gamma=opt.gamma,
         sr=opt.sr,
         seed=opt.seed,
         is_reference=opt.is_reference)
