from typing import List, NamedTuple


def evaluate(data_base_path: str, current_data_path: str, data_ref_paths: List, eval_max_step: int, batch_size: int) -> NamedTuple(
    'evaluate_outputs',
    [
        ('train_finished_data_path', str)
    ]
):
    import json
    import operator
    import os
    import shutil
    from pathlib import Path
    from typing import Tuple

    import torch
    import torch.multiprocessing
    import torch.nn as nn
    from fastspeech2.dataset import Dataset
    from fastspeech2.trainers.evaluator import Evaluator
    from fastspeech2.utils.tools import parse_kwargs
    from fs2_env import get_paths
    from pytorch_sound.models import build_model
    from torch.utils.data import DataLoader


    paths = get_paths(base_path=data_base_path, current_data_path=current_data_path)


    def main(eval_path: str, preprocessed_paths: str,
            train_output: str, save_prefix: str,
            model_name: str, pretrained_path: str = None, num_workers: int = 16,
            batch_size: int = 16,
            pitch_feature: str = 'phoneme', energy_feature: str = 'phoneme',
            pitch_min: float = 0., energy_min: float = 0.,
            lr: float = 2e-4, weight_decay: float = 0.0001, betas=(0.9, 0.98),
            max_step: int = 1000, group_size: int = 4,
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

        dataset = Dataset(eval_path, preprocessed_paths, pitch_min=pitch_min, energy_min=energy_min,
                        text_cleaners=['english_cleaners'],
                        batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)

        print(f'INFO: length of data: {len(dataset)}')

        eval_loader = DataLoader(
            dataset,
            batch_size=batch_size * group_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers
        )

        # evaluate
        evaluted_stats = Evaluator(
            model, optimizer,
            None, eval_loader,
            max_step=max_step, save_interval=save_interval,
            log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
            save_dir=train_output, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
            pretrained_path=pretrained_path, sr=sr,
            scheduler=scheduler, seed=seed, is_reference=is_reference
        ).run()

        return evaluted_stats


    config = {
        # "train_path": "train.txt",
        "eval_path": "val.txt",
        # "preprocessed_path": "./preprocessed",
        # "save_dir": "./saved-models",
        "save_prefix": "fastspeech2_base",
        "model_name": "fast_speech2_vctk",

        "log_interval": 100,
        "pitch_min": -1.9287127187455897,
        "energy_min": -1.375638484954834,
        # "batch_size": 8,
        "save_interval": 100,
        # "max_step": 10,

        # "checkpoint_stat_path": './checkpoint-status.json',
        # "optimal_checkpoint_stat_path": "./optimal-checkpoint-status.json"
    }

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    evaluated_stats = main(preprocessed_paths=data_ref_paths, max_step=eval_max_step, batch_size=batch_size,
                           train_output=paths['train_output'],
                           **parse_kwargs(main, **config))

    optimal_checkpoint_path, optimal_checkpoint_loss = sorted(evaluated_stats.items(), key=operator.itemgetter(1))[0]
    optimal_checkpoint = {
        'path': optimal_checkpoint_path,
        'loss': optimal_checkpoint_loss
    }

    with open(paths['checkpoint_status'], 'w') as f:
        json.dump(evaluated_stats, f, indent=2)

    with open(paths['optimal_checkpoint_status'], 'w') as f:
        json.dump(optimal_checkpoint, f, indent=2)

    current_data_path = Path(current_data_path)
    finished_data_path = current_data_path.parent / '-'.join(current_data_path.stem.split('-')[:-1])
    shutil.move(current_data_path, finished_data_path)

    from collections import namedtuple
    evaluate_outputs = namedtuple(
        'evaluate_outputs',
        ['train_finished_data_path']
    )

    return evaluate_outputs(str(finished_data_path))


if __name__ == '__main__':
    res = evaluate(data_base_path='/local-storage', 
                   current_data_path='/opt/storage/fs2-data/data/20211017-111713-intermediate', 
                   eval_max_step=10,
                   batch_size=8,
                   data_ref_paths=[

  "/opt/storage/fs2-data/data/20211017-111713-intermediate/preprocessed",
  "/opt/storage/fs2-data/data/20211016-192849/preprocessed",
  "/opt/storage/fs2-data/data/20211017-012441/preprocessed"

])
    print(res)
