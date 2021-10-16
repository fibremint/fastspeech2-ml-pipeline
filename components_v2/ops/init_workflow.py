# import argparse
# import datetime
# import glob
# import json
# import os
# import shutil
# import sys
# from pathlib import Path


# def _parse_):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-base-path', type=str, default='/mnt')
#     parser.add_argument('--raw-data-path', type=str, default='./raw-data')
#     parser.add_argument('--fs2-base-path', type=str, default='./fs2-data')
#     parser.add_argument('--fs2-data-path', type=str, default='./data')
#     parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
#     parser.add_argument('--fs2-dupl-data-relpaths-filename', type=str, default='data-dupl-relpaths.json')

#     return parser.parse_)

from typing import NamedTuple


def init_workflow(data_base_path: str) -> NamedTuple(
    'prepare_preprocess_outputs',
    [
        ('current_data_path', str),
        ('is_new_data_exist', bool)
    ]):

    import datetime
    import glob
    import json
    import os
    import shutil
    import sys
    from pathlib import Path

    
    # CONFIG_PATH = './configs'
    # FS2_DATA_CONFIG_FILENAME = 'fastspeech2_data_path.json'
    FS2_ENV_FILENAME = 'fastspeech2_env.json'

    configs = {
        'fs2_config_path': './configs',
        'raw_data_path': './raw-data',
        'fs2_base_path': './fs2-data',
        'fs2_data_path': './data',
        'fs2_data_relpaths_filename': 'data-relpaths.json',
        'fs2_dupl_data_relpaths_filename': 'data-dupl-relpaths.json'
    }

    from collections import namedtuple
    prepare_preprocess_outputs = namedtuple(
        'prepare_preprocess_outputs',
        ['current_data_path', 'is_new_data_exist']
    )

    base_path = Path(data_base_path)
    if not base_path.exists():
        sys.exit('ERR: base path is not exists')

    fs2_base_path = base_path / configs['fs2_base_path']
    fs2_config_path = fs2_base_path / configs['fs2_config_path']
    if not fs2_config_path.exists():
        os.makedirs(fs2_config_path)

    fs2_default_configs = {os.path.basename(config_path): config_path 
                           for config_path in glob.glob(f'{configs["fs2_config_path"]}/*.json')}

    current_fs2_configs = [os.path.basename(config_path) 
                           for config_path in glob.glob(f'{fs2_config_path}/*.json')]

    # fs2_env_path = fs2_base_path

    # if not fs2_env_path.exists():
    #     print(f'INFO: copy FastSpeech2 environment file')
    #     shutil.copy(FS2_ENV_FILENAME, )

    for default_config_filename, default_config_path in fs2_default_configs.items():
        if default_config_filename not in current_fs2_configs:
            print(f'INFO: copy configuration file: "{default_config_filename}"')
            shutil.copy(default_config_path, f'{fs2_config_path}/{default_config_filename}')

    raw_data_path = base_path / configs['raw_data_path']
    is_new_data_exist = raw_data_path.exists()
    if not is_new_data_exist:            
        print('WARN: raw data path is not exists. aborted.')
        return prepare_preprocess_outputs(str(None), is_new_data_exist)

    else:
        print('INFO: found raw data path')

        # move raw data to intermediate data path
        fs2_data_path = fs2_base_path / configs['fs2_data_path']
        curr_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        curr_data_path = fs2_data_path / f'{curr_datetime}-intermediate'
        os.makedirs(curr_data_path)

        print('INFO: move raw data path to intermediate data path')
        shutil.move(str(raw_data_path), str(curr_data_path))

        # get relpaths of current data
        # ref: https://stackoverflow.com/questions/44994604/python-glob-os-relative-path-making-filenames-into-a-list
        curr_raw_data_path = curr_data_path / configs['raw_data_path']
        print(f'INFO: current raw data path: {curr_raw_data_path}')
        curr_data_wav_relpaths = [os.path.relpath(data_abspath, curr_raw_data_path)
                                for data_abspath in glob.glob(f'{curr_raw_data_path}/wav48/*/*.wav')]
        curr_data_txt_relpaths = [os.path.relpath(data_abspath, curr_raw_data_path)
                                for data_abspath in glob.glob(f'{curr_raw_data_path}/txt/*/*.txt')]

        if not curr_raw_data_path:
            print('WARN: empty raw data')
            print(f'current raw data path: {curr_raw_data_path}')
            

        # get all of the relpath of data in ./fs2-data/data path 
        relpaths = set()
        data_relpaths = glob.glob(f'{fs2_data_path}/*/{configs["fs2_data_relpaths_filename"]}')
        for data_relpath in data_relpaths:
            with open(data_relpath, 'r') as f:
                curr_relpaths = json.load(f)
                relpaths.update(set(curr_relpaths))
        
        # write current data relpaths
        curr_data_relpaths = set(curr_data_wav_relpaths) | set(curr_data_txt_relpaths)
        print(f'INFO: current data len: {len(curr_data_relpaths) // 2}')

        # write duplicated relpaths
        duplicated_file_relpaths = curr_data_relpaths.intersection(relpaths)
        if duplicated_file_relpaths:
            print('WARN: some of the data are duplicated')

        with open(f'{curr_data_path}/{configs["fs2_dupl_data_relpaths_filename"]}', 'w') as f:
            duplicated_file_relpaths = sorted(list(duplicated_file_relpaths))
            json.dump(duplicated_file_relpaths, f, indent=2)
        
        # shutil.copy(f'{curr_data_path}/{configs["fs2_dupl_data_relpaths_filename"]}', f'/tmp/{configs["fs2_dupl_data_relpaths_filename"]}')

        # write not duplicated relpaths
        preprocess_target_relpaths = curr_data_relpaths.difference(relpaths)
        with open(f'{curr_data_path}/{configs["fs2_data_relpaths_filename"]}', 'w') as f:
            preprocess_target_relpaths = sorted(list(preprocess_target_relpaths))
            json.dump(preprocess_target_relpaths, f, indent=2)

        # shutil.copy(f'{curr_data_path}/{configs["fs2_data_relpaths_filename"]}', f'/tmp/{configs["fs2_data_relpaths_filename"]}')

        if not preprocess_target_relpaths:
            print('WARN: all of the file names are duplicated. current workflow will be aborted.')
            finished_data_path = curr_data_path.parent / '-'.join(curr_data_path.stem.split('-')[:-1])
            shutil.move(curr_data_path, finished_data_path)
            
            is_new_data_exist = False

    # # save values to set output variable

    # from collections import namedtuple
    # prepare_preprocess_outputs = namedtuple(
    #     'prepare_preprocess_outputs',
    #     ['current_data_path', 'is_new_data_exist']
    # )

    return prepare_preprocess_outputs(str(curr_data_path), is_new_data_exist)


if __name__ == '__main__':
    from fire import Fire
    Fire(init_workflow)
