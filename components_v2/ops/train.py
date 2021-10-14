# import argparse
# import json
# import os
# from typing import Tuple

# import torch
# import torch.multiprocessing
# import torch.nn as nn
# from fastspeech2.dataset import Dataset
# from pytorch_sound.models import build_model
# from torch.utils.data import DataLoader
# from app.trainer.base_trainer import BaseTrainer



# def _parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--current-data-path', type=str, default='', required=True)
#     parser.add_argument('--train-path', type=str, default='./preprocessed/train.txt')
#     parser.add_argument('--preprocessed_path', type=str, default='./preprocessed')
#     parser.add_argument('--model-save-path', type=str, default='./saved-models/fastspeech2-base')
#     parser.add_argument('--model-save-prefix', type=str, default='fastspeech2-base')
#     parser.add_argument('--model-name', type=str, default='fast_speech2_vctk')
#     parser.add_argument('--log-interval', type=int, default=100)
#     parser.add_argument('--pitch-min', type=float, default=-1.9287127187455897)
#     parser.add_argument('--energy-min', type=float, default=-1.375638484954834)

#     parser.add_argument('--pretrained-path', type=str, default='')
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

#     return parser.parse_args()





# # TODO: 
# # * load checkpoint if global checkpoint status exists
# # * set train max_step as loaded_checkpoint_step + max_step
# def _train(train_path: str, preprocessed_path: str,
#          save_dir: str, save_prefix: str,
#          model_name: str, pretrained_path: str = '', num_workers: int = 16,
#          batch_size: int = 8,
#          pitch_feature: str = 'phoneme', energy_feature: str = 'phoneme',
#          pitch_min: float = 0., energy_min: float = 0.,
#          lr: float = 2e-4, weight_decay: float = 0.0001, betas=(0.9, 0.98),
#          max_step: int = 400000, group_size: int = 4,
#          save_interval: int = 10000, log_interval: int = 50, grad_clip: float = 0.0, grad_norm: float = 5.0,
#          milestones: Tuple[int] = None, gamma: float = 0.2, sr: int = 22050, seed: int = 2021,
#          is_reference: bool = False):
#     # create model
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

#     dataset = Dataset(train_path, preprocessed_path, pitch_min=pitch_min, energy_min=energy_min,
#                       text_cleaners=['english_cleaners'],
#                       batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)
#     train_loader = DataLoader(
#         dataset,
#         batch_size=batch_size * group_size,
#         shuffle=True,
#         collate_fn=dataset.collate_fn,
#         num_workers=num_workers
#     )

#     # train
#     BaseTrainer(
#         model, optimizer,
#         train_loader, None,
#         max_step=max_step, valid_max_step=1, save_interval=save_interval,
#         log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
#         save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
#         pretrained_path=pretrained_path, sr=sr,
#         scheduler=scheduler, seed=seed, is_reference=is_reference
#     ).run()


# def run_config(config_path: str):
#     with open(config_path, 'r') as r:
#         config = json.load(r)
#     main(**config)

def train(current_data_path: str) -> None:
    import argparse
    import json
    import os
    from typing import Tuple

    import torch
    import torch.multiprocessing
    import torch.nn as nn
    from app.trainer.base_trainer import BaseTrainer
    from fastspeech2.dataset import Dataset
    from pytorch_sound.models import build_model
    from torch.utils.data import DataLoader


    def _parse_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--current-data-path', type=str, default='', required=True)
        parser.add_argument('--train-path', type=str, default='./preprocessed/train.txt')
        parser.add_argument('--preprocessed_path', type=str, default='./preprocessed')
        parser.add_argument('--model-save-path', type=str, default='./saved-models/fastspeech2-base')
        parser.add_argument('--model-save-prefix', type=str, default='fastspeech2-base')
        parser.add_argument('--model-name', type=str, default='fast_speech2_vctk')
        parser.add_argument('--log-interval', type=int, default=100)
        parser.add_argument('--pitch-min', type=float, default=-1.9287127187455897)
        parser.add_argument('--energy-min', type=float, default=-1.375638484954834)

        parser.add_argument('--pretrained-path', type=str, default='')
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

        return parser.parse_args()


    # TODO: 
    # * load checkpoint if global checkpoint status exists
    # * set train max_step as loaded_checkpoint_step + max_step
    def _train(train_path: str, preprocessed_path: str,
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



    args = _parse_args()
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # fire.Fire(run_config)
    _train(train_path=os.path.join(args.current_data_path, args.train_path),
         preprocessed_path=os.path.join(args.current_data_path, args.preprocessed_path),
         save_dir=os.path.join(args.current_data_path, args.model_save_path),
         save_prefix=args.model_save_prefix,
         model_name=args.model_name,
         pretrained_path=args.pretrained_path,
         num_workers=args.num_workers,
         batch_size=args.batch_size,
         pitch_feature=args.pitch_feature,
         energy_feature=args.energy_feature,
         pitch_min=args.pitch_min,
         energy_min=args.energy_min,
         lr=args.lr,
         weight_decay=args.weight_decay,
         max_step=args.max_step,
         group_size=args.group_size,
         save_interval=args.save_interval,
         log_interval=args.log_interval,
         gamma=args.gamma,
         sr=args.sr,
         seed=args.seed,
         is_reference=args.is_reference)
