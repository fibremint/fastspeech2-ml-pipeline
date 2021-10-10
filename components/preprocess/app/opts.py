import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-base-path', type=str, default='', required=True)
parser.add_argument('--input-path', type=str, default='./before-align')
parser.add_argument('--input-textgrid-path', type=str, default='./aligned')
parser.add_argument('--preprocess-output-path', type=str, default='./preprocessed')
parser.add_argument('--config-path', type=str, default='./configs/vctk_preprocess.json')
opt = parser.parse_args()
