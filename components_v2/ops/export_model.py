from typing import NamedTuple


def export_model(data_base_path: str) -> NamedTuple(
    'export_model_outputs',
    [
        ('model_version', str)
    ]
):
    import datetime
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from fs2_env import get_paths
    paths = get_paths(base_path=data_base_path)


    configs = {
        'fs2_data_path': './fs2-data',
        'model_base_path': './models',
        'model_export_path': './model-store',
        'model_name': 'fastspeech2',
        'model_handler': './app/model_handler.py',
        'lexicon_path': './common/lexicons/librispeech-lexicon.txt',
        'metadata_path': './metadata',
        'global_optimal_checkpoint_stat_path': './global-optimal-checkpoint-status.json'
    }

    model_export_tmp_path = Path(paths['models']) / 'tmp'
    os.makedirs(model_export_tmp_path, exist_ok=True)

    os.makedirs(paths['deployed_models'], exist_ok=True)

    with open(paths['global_optimal_checkpoint_status'], 'r') as f:
        global_optimal_checkpoint_stat = json.load(f)

    target_checkpoint_path = Path(global_optimal_checkpoint_stat['base_path']) \
        / global_optimal_checkpoint_stat['deployed_checkpoint']['path'] \

    model_version = datetime.datetime.now().strftime("%y%m%d-%H%M")

    print(f'INFO: Export model from checkpoint "{target_checkpoint_path}".')
    subprocess.run(['torch-model-archiver',
                    '--model-name', configs["model_name"],
                    '--version', model_version,
                    '--serialized-file', str(target_checkpoint_path),
                    '--export-path', str(model_export_tmp_path),
                    '--handler', configs["model_handler"],
                    '--extra-files', paths['lexicon']])
    print(f'INFO: Successfully exported model as archive file.')

    shutil.move(model_export_tmp_path / (configs["model_name"] + ".mar"), 
                Path(paths['deployed_models']) / (configs["model_name"] + "-" + model_version + ".mar"))
    print(f'INFO: Successfully moved model archive file to model export path')

    shutil.rmtree(model_export_tmp_path)

    from collections import namedtuple
    
    export_model_outputs = namedtuple(
        'export_model_outputs',
        ['model_version']
    )

    return export_model_outputs(model_version)


if __name__ == '__main__':
    export_model('/local-storage')
