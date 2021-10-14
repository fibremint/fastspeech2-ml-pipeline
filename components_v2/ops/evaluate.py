import argparse
from typing import Tuple
from typing import NamedTuple

# def _parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--current-data-path', type=str, default='', required=True)
#     parser.add_argument('--checkpoint-stat-path', type=str, default='./checkpoint-status.json')
#     parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')

#     parser.add_argument('--evaluate-path', type=str, default='./preprocessed/val.txt')
#     parser.add_argument('--preprocessed_path', type=str, default='./preprocessed')
#     parser.add_argument('--model-save-path', type=str, default='./saved-models/fastspeech2-base')
#     parser.add_argument('--model-save-prefix', type=str, default='fastspeech2-base')
#     parser.add_argument('--model-name', type=str, default='fast_speech2_vctk')
#     parser.add_argument('--log-interval', type=int, default=100)
#     parser.add_argument('--pitch-min', type=float, default=-1.9287127187455897)
#     parser.add_argument('--energy-min', type=float, default=-1.375638484954834)

#     parser.add_argument('--pretrained-path', type=str, default='./models/fastspeech2-base/FastSpeech2')
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--batch-size', type=int, default=8)
#     parser.add_argument('--pitch-feature', type=str, default='phoneme')
#     parser.add_argument('--energy-feature', type=str, default='phoneme')
#     parser.add_argument('--pitch-mean', type=float, default=0.)
#     parser.add_argument('--energy-mean', type=float, default=0.)
#     parser.add_argument('--lr', type=float, default=2e-4)
#     parser.add_argument('--weight-decay', type=float, default=0.0001)
#     parser.add_argument('--group-size', type=int, default=4)
#     parser.add_argument('--save-interval', type=int, default=100)
#     parser.add_argument('--grad-clip', type=float, default=0.)
#     parser.add_argument('--grad-norm', type=float, default=5.0)
#     parser.add_argument('--gamma', type=float, default=0.2)
#     parser.add_argument('--sr', type=int, default=22050)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--is-reference', type=bool, default=False)
#     parser.add_argument('--max-step', type=int, default=500)
#     parser.add_argument('--validate-max-step', type=int, default=10)

#     return parser.parse_args()


# def _evaluate(current_data_path: str, checkpoint_stat_path: str, optimal_checkpoint_stat_path: str,
#          evaluate_path: str, preprocessed_path: str,
#          save_dir: str, save_prefix: str,
#          model_name: str, pretrained_path: str = '', num_workers: int = 16,
#          batch_size: int = 8,
#          pitch_feature: str = 'phoneme', energy_feature: str = 'phoneme',
#          pitch_min: float = 0., energy_min: float = 0.,
#          lr: float = 2e-4, weight_decay: float = 0.0001, betas=(0.9, 0.98),
#          max_step: int = 400000, validate_max_step: int = 100, group_size: int = 4,
#          save_interval: int = 10000, log_interval: int = 50, grad_clip: float = 0.0, grad_norm: float = 5.0,
#          milestones: Tuple[int] = None, gamma: float = 0.2, sr: int = 22050, seed: int = 2021,
#          is_reference: bool = False) -> str:
#     # create model
#     import json
#     import os
#     import shutil
#     from pathlib import Path
#     from typing import Tuple

#     import torch
#     import torch.multiprocessing
#     import torch.nn as nn
#     from app.trainer.evaluator import Evaluator
#     from fastspeech2.dataset import Dataset
#     from pytorch_sound.models import build_model
#     from torch.utils.data import DataLoader

#     model = build_model(model_name).cuda()

#     # multi-gpu
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)

#     # create optimizers
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
#     if milestones:
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
#     else:
#         scheduler = None

#     dataset = Dataset(evaluate_path, preprocessed_path, pitch_min=pitch_min, energy_min=energy_min,
#                       text_cleaners=['english_cleaners'],
#                       batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)
#     test_loader = DataLoader(
#         dataset,
#         batch_size=batch_size * group_size,
#         shuffle=True,
#         collate_fn=dataset.collate_fn,
#         num_workers=num_workers
#     )

#     # train
#     checkpoint_stats = Evaluator(
#         model, optimizer,
#         None, test_loader,
#         current_data_path=current_data_path,
#         max_step=1, valid_max_step=validate_max_step, save_interval=save_interval,
#         log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
#         save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
#         pretrained_path=pretrained_path, sr=sr,
#         scheduler=scheduler, seed=seed, is_reference=is_reference
#     ).run()

#     import operator

#     optimal_checkpoint_path, optimal_checkpoint_loss = sorted(checkpoint_stats.items(), key=operator.itemgetter(1))[0]

#     optimal_checkpoint = {
#         'path': optimal_checkpoint_path,
#         'loss': optimal_checkpoint_loss
#     }

#     with open(os.path.join(current_data_path, checkpoint_stat_path), 'w') as f:
#         json.dump(checkpoint_stats, f, indent=2)

#     shutil.copy(os.path.join(current_data_path, checkpoint_stat_path), os.path.join('/tmp', checkpoint_stat_path))

#     with open(os.path.join(current_data_path, optimal_checkpoint_stat_path), 'w') as f:
#         json.dump(optimal_checkpoint, f, indent=2)

#     shutil.copy(os.path.join(current_data_path, optimal_checkpoint_stat_path), os.path.join('/tmp', optimal_checkpoint_stat_path))

#     current_data_path = Path(current_data_path)
#     finished_data_path = current_data_path.parent / '-'.join(current_data_path.stem.split('-')[:-1])
#     shutil.move(current_data_path, finished_data_path)

#     # with open('/tmp/finished-data-path.txt', 'w') as f:
#     #     f.write(str(finished_data_path))

#     return finished_data_path


def evaluate(current_data_path: str) -> NamedTuple(
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
    from app.trainer.evaluator import Evaluator
    from fastspeech2.dataset import Dataset
    from pytorch_sound.models import build_model
    from torch.utils.data import DataLoader


    def _parse_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--current-data-path', type=str, default='', required=True)
        parser.add_argument('--checkpoint-stat-path', type=str, default='./checkpoint-status.json')
        parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')

        parser.add_argument('--evaluate-path', type=str, default='./preprocessed/val.txt')
        parser.add_argument('--preprocessed_path', type=str, default='./preprocessed')
        parser.add_argument('--model-save-path', type=str, default='./saved-models/fastspeech2-base')
        parser.add_argument('--model-save-prefix', type=str, default='fastspeech2-base')
        parser.add_argument('--model-name', type=str, default='fast_speech2_vctk')
        parser.add_argument('--log-interval', type=int, default=100)
        parser.add_argument('--pitch-min', type=float, default=-1.9287127187455897)
        parser.add_argument('--energy-min', type=float, default=-1.375638484954834)

        parser.add_argument('--pretrained-path', type=str, default='./models/fastspeech2-base/FastSpeech2')
        parser.add_argument('--num-workers', type=int, default=4)
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--pitch-feature', type=str, default='phoneme')
        parser.add_argument('--energy-feature', type=str, default='phoneme')
        parser.add_argument('--pitch-mean', type=float, default=0.)
        parser.add_argument('--energy-mean', type=float, default=0.)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--weight-decay', type=float, default=0.0001)
        parser.add_argument('--group-size', type=int, default=4)
        parser.add_argument('--save-interval', type=int, default=100)
        parser.add_argument('--grad-clip', type=float, default=0.)
        parser.add_argument('--grad-norm', type=float, default=5.0)
        parser.add_argument('--gamma', type=float, default=0.2)
        parser.add_argument('--sr', type=int, default=22050)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--is-reference', type=bool, default=False)
        parser.add_argument('--max-step', type=int, default=500)
        parser.add_argument('--validate-max-step', type=int, default=10)

        return parser.parse_args()


    def _evaluate(current_data_path: str, checkpoint_stat_path: str, optimal_checkpoint_stat_path: str,
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
            is_reference: bool = False) -> str:
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

        # with open('/tmp/finished-data-path.txt', 'w') as f:
        #     f.write(str(finished_data_path))

        return finished_data_path


    args = _parse_args()
    
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    finished_data_path = _evaluate(current_data_path=current_data_path,
         checkpoint_stat_path=args.checkpoint_stat_path,
         optimal_checkpoint_stat_path=args.optimal_checkpoint_stat_path,
         evaluate_path=os.path.join(current_data_path, args.evaluate_path),
         preprocessed_path=os.path.join(current_data_path, args.preprocessed_path),
         save_dir=os.path.join(current_data_path, args.model_save_path),
         save_prefix=args.model_save_prefix,
         model_name=args.model_name,
         pretrained_path=os.path.join(current_data_path, args.model_save_path, args.pretrained_path),
         num_workers=args.num_workers,
         batch_size=args.batch_size,
         pitch_feature=args.pitch_feature,
         energy_feature=args.energy_feature,
         pitch_min=args.pitch_min,
         energy_min=args.energy_min,
         lr=args.lr,
         weight_decay=args.weight_decay,
         max_step=args.max_step,
         validate_max_step=args.validate_max_step,
         group_size=args.group_size,
         save_interval=args.save_interval,
         log_interval=args.log_interval,
         gamma=args.gamma,
         sr=args.sr,
         seed=args.seed,
         is_reference=args.is_reference)

    from collections import namedtuple
    evaluate_outputs = namedtuple(
        'evaluate_outputs',
        ['train_finished_data_path']
    )

    return evaluate_outputs(finished_data_path)

    # return finished_data_path


# def create_component():
#     from kfp import components

#     components.create_component_from_func(evaluate, output_component_file='./components_v2/resources/evaluate.yaml')


# if __name__ == '__main__':
#     create_component()
