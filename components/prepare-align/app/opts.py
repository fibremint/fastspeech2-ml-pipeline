# {
#   "out_dir": "/local-storage/data/fastspeech2/vctk",
#   "sample_rate": 22050,
#   "max_wav_value": 32767.0,
#   "cleaners": ["english_cleaners"],
#   "in_dir": "/local-storage/data/VCTK-Corpus",
#   "train_speakers": "configs/vctk_train_speakers.txt"
# }
import argparse

default_cleaners = ['english_cleaners']

parser = argparse.ArgumentParser()
# TODO: set required to True when release
parser.add_argument('--data-base-path', type=str, default='', required=False)
parser.add_argument('--data-input-path', type=str, default='./raw-data')
parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
parser.add_argument('--data-output-path', type=str, default='./before-align')
parser.add_argument('--sample-rate', type=int, default=22050)
parser.add_argument('--max-wav-value', type=float, default=32767.0)
# ref: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
# https://stackoverflow.com/questions/43660172/why-does-argparse-include-default-value-for-optional-argument-even-when-argument
parser.add_argument('-c', '--cleaners', action='append', type=str)

opt = parser.parse_args()
opt.cleaners = opt.cleaners if opt.cleaners else default_cleaners
