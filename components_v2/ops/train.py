

from typing import List, NamedTuple

from torch.utils import data

# TODO: return preprocessed paths
def train(data_base_path: str, current_data_path: str, train_max_step: int, batch_size: int, model_save_interval: int) -> NamedTuple(
    'train_outputs',
    [
        ('data_ref_paths', List)
    ]):
    import re
    import json
    import os
    from pathlib import Path
    from typing import Tuple

    import torch
    import torch.multiprocessing
    import torch.nn as nn
    from fastspeech2.dataset import Dataset
    from fastspeech2.trainers.trainer import Trainer
    from fastspeech2.utils import parse_kwargs, get_rest_path_from
    from pytorch_sound.models import build_model
    from torch.utils.data import DataLoader


    def main(train_path: str, preprocessed_paths: List,
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

        dataset = Dataset(train_path, preprocessed_paths, pitch_min=pitch_min, energy_min=energy_min,
                        text_cleaners=['english_cleaners'],
                        batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)

        print(f'INFO: length of data: {len(dataset)}')

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
        "preprocessed_path": "preprocessed",
        "save_dir": "./saved-models",
        "save_prefix": "fastspeech2_base",
        "model_name": "fast_speech2_vctk",

        "log_interval": 100,
        "pitch_min": -1.9287127187455897,
        "energy_min": -1.375638484954834,
        # "batch_size": 8,
        # "save_interval": 100,
        "num_workers": 4,
        # "max_step": 201,

        "metadata_path": "./metadata",
        "global_optimal_checkpoint_stat_path": "./global-optimal-checkpoint-status.json",
        "data_refs_filename": "data_refs.json",

        "fs2_data_base_path": "./data",
        "data_intermediate_regex": "\d{8}-\d{6}-intermediate"
    }

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    data_base_path = Path(data_base_path)
    current_data_path = Path(current_data_path)
    current_preprocessed_path = current_data_path / config['preprocessed_path']

    config['save_dir'] = current_data_path / config['save_dir']
    
    previous_data_refs = []
    pretrained_checkpoint_path = None
    preprocessed_paths = [str(current_preprocessed_path)]
    current_preprocessed_finished_path = current_data_path.parent / '-'.join(current_data_path.stem.split('-')[:-1]) / config['preprocessed_path']
    finished_preprocessed_paths = [str(current_preprocessed_finished_path)]

    global_optimal_checkpoint_path = data_base_path / config['fs2_base_path'] / config['metadata_path'] / config['global_optimal_checkpoint_stat_path']
    if global_optimal_checkpoint_path.exists():
        with open(f'{global_optimal_checkpoint_path}', 'r') as f:
            global_optimal_checkpoint = json.load(f)
        
        previous_checkpoint_base_path = global_optimal_checkpoint['base_path']
        pretrained_checkpoint_path = os.path.join(previous_checkpoint_base_path, global_optimal_checkpoint['deployed_checkpoint']['path'])
        
        with open(os.path.join(previous_checkpoint_base_path, config['data_refs_filename']), 'r') as f:
            previous_data_refs = json.load(f)
            finished_preprocessed_paths.extend(previous_data_refs)
        
        search_path_pattern=f'{data_base_path / config["fs2_base_path"] / config["fs2_data_base_path"]}' + '/*/' + config["preprocessed_path"]
        search_path_filter_regex=re.compile(config["data_intermediate_regex"])
        rest_preprocessed_paths = get_rest_path_from(search_path_pattern=search_path_pattern,
                                                     exclude_paths=previous_data_refs,
                                                     search_path_filter_regex=search_path_filter_regex)

        if rest_preprocessed_paths:
            print(f'INFO: find {len(rest_preprocessed_paths)} rest of preprocessed path(s)')
            print(f'INFO: {rest_preprocessed_paths}')
            print(f'INFO: set these to additional load path')

            preprocessed_paths.extend(rest_preprocessed_paths)
            # finished_preprocessed_paths.extend(rest_preprocessed_paths)

    main(pretrained_path=pretrained_checkpoint_path, max_step=train_max_step, batch_size=batch_size, 
         save_interval=model_save_interval,
         preprocessed_paths=preprocessed_paths,
         **parse_kwargs(main, **config))

    # current_preprocessed_finished_path = current_data_path.parent / '-'.join(current_data_path.stem.split('-')[:-1]) / config['preprocessed_path']
    # finished_preprocessed_paths = [str(current_preprocessed_finished_path)] + rest_preprocessed_paths

    with open(f'{current_data_path / config["data_refs_filename"]}', 'w') as f:
        json.dump(finished_preprocessed_paths, f, indent=2)

    from collections import namedtuple
    train_outputs = namedtuple(
        'train_outputs',
        ['data_ref_paths']
    )

    return train_outputs(preprocessed_paths)


if __name__ == '__main__':
    res = train('/local-storage', '/local-storage/fs2-data/data/20211016-031309-intermediate')
    print(res)
