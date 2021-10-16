

def train(data_base_path: str, current_data_path: str) -> None:
    import json
    import os
    from pathlib import Path
    from typing import Tuple

    import torch
    import torch.multiprocessing
    import torch.nn as nn
    from fastspeech2.dataset import Dataset
    from fastspeech2.trainers.trainer import Trainer
    from fastspeech2.utils.tools import parse_kwargs
    from pytorch_sound.models import build_model
    from torch.utils.data import DataLoader


    def main(train_path: str, preprocessed_path: str,
            save_dir: str, save_prefix: str,
            model_name: str, pretrained_path: str = None, num_workers: int = 16,
            batch_size: int = 16,
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

        save_dir = current_data_path / save_dir

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
        Trainer(
            model, optimizer,
            train_loader, None,
            max_step=max_step, save_interval=save_interval,
            log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
            save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
            pretrained_path=pretrained_path, sr=sr,
            scheduler=scheduler, seed=seed, is_reference=is_reference
        ).run()


    config = {
        "fs2_base_path": "fs2-data",
        "train_path": "train.txt",
        "eval_path": "val.txt",
        "preprocessed_path": "./preprocessed",
        "save_dir": "./saved-models",
        "save_prefix": "fastspeech2_base",
        "model_name": "fast_speech2_vctk",

        "log_interval": 100,
        "pitch_min": -1.9287127187455897,
        "energy_min": -1.375638484954834,
        "batch_size": 8,
        "save_interval": 100,
        "num_workers": 4,
        "max_step": 201,

        "metadata_path": "./metadata",
        "global_optimal_checkpoint_stat_path": "./global-optimal-checkpoint-status.json"
    }

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    current_data_path = Path(current_data_path)
    config['preprocessed_path'] = current_data_path / config['preprocessed_path']
    config['save_dir'] = current_data_path / config['save_dir']
    
    pretrained_path = None
    global_optimal_checkpoint_path = Path(data_base_path) / config['fs2_base_path'] / config['metadata_path'] / config['global_optimal_checkpoint_stat_path']
    if global_optimal_checkpoint_path.exists():
        with open(f'{global_optimal_checkpoint_path}', 'r') as f:
            global_optimal_checkpoint = json.load(f)
            pretrained_path = os.path.join(global_optimal_checkpoint['base_path'], global_optimal_checkpoint['deployed_checkpoint']['path'])

    main(pretrained_path=pretrained_path, **parse_kwargs(main, **config))


if __name__ == '__main__':
    res = train('/local-storage', '/local-storage/fs2-data/data.2/20211010-105342-intermediate')
    print(res)
