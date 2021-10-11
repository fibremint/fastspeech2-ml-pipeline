import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-base-path', type=str, default='/mnt')
parser.add_argument('--raw-data-path', type=str, default='./raw-data')
parser.add_argument('--fs2-base-path', type=str, default='./fs2-data')
parser.add_argument('--fs2-data-path', type=str, default='./data')
parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
parser.add_argument('--fs2-dupl-data-relpaths-filename', type=str, default='data-dupl-relpaths.json')

opt = parser.parse_args()