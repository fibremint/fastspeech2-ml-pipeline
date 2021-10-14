import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data-base-path', type=str, default='', required=True)
parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
parser.add_argument('--model-base-path', type=str, default='./models')
parser.add_argument('--model-export-path', type=str, default='./model-store')
parser.add_argument('--model-name', type=str, default='fastspeech2')
# parser.add_argument('--model-version', type=float, required=True)
parser.add_argument('--model-handler', type=str, default='./app/model_handler.py')
parser.add_argument('--lexicon-path', type=str, default='./common/lexicons/librispeech-lexicon.txt')
parser.add_argument('--metadata-path', type=str, default='./metadata')
parser.add_argument('--global-optimal-checkpoint-stat-path', type=str, default='./deployed-checkpoint-status.json')

opt = parser.parse_args()

fs2_base_path = Path(opt.data_base_path) / opt.fs2_data_path

model_export_path = fs2_base_path / opt.model_base_path / opt.model_export_path
if not model_export_path.exists():
    os.makedirs(model_export_path)

global_optimal_checkpoint_stat_path = fs2_base_path / opt.metadata_path / opt.global_optimal_checkpoint_stat_path
with open(f'{global_optimal_checkpoint_stat_path}', 'r') as f:
    global_optimal_checkpoint_stat = json.load(f)

target_checkpoint_path = Path(global_optimal_checkpoint_stat['base_path']) \
    / global_optimal_checkpoint_stat['deployed_checkpoint']['path'] \

lexicon_path = fs2_base_path / opt.lexicon_path

model_version = datetime.datetime.now().strftime("%y%m%d-%H%M")

subprocess.run(['torch-model-archiver',
                '--model-name', opt.model_name + '-' + model_version,
                '--version', model_version,
                '--serialized-file', str(target_checkpoint_path),
                '--export-path', str(model_export_path),
                '--handler', opt.model_handler,
                '--extra-files', lexicon_path])

with open('/tmp/model-version.txt', 'w') as f:
    f.write(model_version)
