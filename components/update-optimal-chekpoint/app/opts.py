import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-base-path', type=str, default='', required=True)
parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
parser.add_argument('--metadata-path', type=str, default='./metadata')
parser.add_argument('--target-data-path', type=str, default='', required=True)
parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')
parser.add_argument('--deployed-checkpoint-stat-path', type=str, default='./deployed-checkpoint-status.json')

opt = parser.parse_args()
