
'''
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
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--current-data-path', type=str, default='', required=True)
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

opt = parser.parse_args()
