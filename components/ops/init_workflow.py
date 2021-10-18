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

    from fs2_env import get_paths
    
    configs = {
        'fs2_config_path': './configs',
        'raw_data_path': './raw-data',
        'fs2_base_path': './fs2-data',
        'fs2_data_path': './data',
        'fs2_data_relpaths_filename': 'data-relpaths.json',
        'fs2_dupl_data_relpaths_filename': 'data-dupl-relpaths.json',
        'container_default_data_path': '/workspace/default-data'
    }

    from collections import namedtuple
    prepare_preprocess_outputs = namedtuple(
        'prepare_preprocess_outputs',
        ['current_data_path', 'is_new_data_exist']
    )

    container_data_paths = get_paths(base_path=configs['container_default_data_path'])
    paths = get_paths(base_path=data_base_path)

    # if not Path(paths['fs2_base']).exists():
    #     sys.exit('ERR: base path is not exists')
    os.makedirs(paths['fs2_base'], exist_ok=True)

    if not Path(paths['lexicon']).exists():
        os.makedirs(paths['lexicons'], exist_ok=True)
        shutil.copy(container_data_paths['lexicon'], paths['lexicons'])

    raw_data_path = Path(data_base_path) / configs['raw_data_path']
    is_new_data_exist = raw_data_path.exists()
    if not is_new_data_exist:            
        print('WARN: raw data path is not exists. aborted.')
        return prepare_preprocess_outputs(str(None), is_new_data_exist)

    else:
        print('INFO: found raw data path')

        curr_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        curr_data_path = Path(paths['data']) / f'{curr_datetime}-intermediate'
        paths = get_paths(base_path=data_base_path, current_data_path=str(curr_data_path))

        os.makedirs(curr_data_path)

        print('INFO: move raw data path to intermediate data path')
        shutil.move(raw_data_path, paths['raw_data'])

        # get relpaths of current data
        # ref: https://stackoverflow.com/questions/44994604/python-glob-os-relative-path-making-filenames-into-a-list
        print(f'INFO: current raw data path: {paths["raw_data"]}')
        curr_data_wav_relpaths = [os.path.relpath(data_abspath, paths["raw_data"])
                                for data_abspath in glob.glob(f'{paths["raw_data"]}/wav48/*/*.wav')]
        curr_data_txt_relpaths = [os.path.relpath(data_abspath, paths["raw_data"])
                                for data_abspath in glob.glob(f'{paths["raw_data"]}/txt/*/*.txt')]  

        relpaths = set()
        data_relpaths = glob.glob(f'{Path(paths["data"])}/*/{configs["fs2_data_relpaths_filename"]}')
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

        with open(paths['data_duplicated_relpaths'], 'w') as f:
            duplicated_file_relpaths = sorted(list(duplicated_file_relpaths))
            json.dump(duplicated_file_relpaths, f, indent=2)
        
        # write not duplicated relpaths
        preprocess_target_relpaths = curr_data_relpaths.difference(relpaths)
        with open(paths['data_relpaths'], 'w') as f:
            preprocess_target_relpaths = sorted(list(preprocess_target_relpaths))
            json.dump(preprocess_target_relpaths, f, indent=2)

        if not preprocess_target_relpaths:
            print('WARN: all of the file names are duplicated. current workflow will be aborted.')
            current_data_path = Path(paths['current_data'])
            finished_data_path = current_data_path.parent / '-'.join(current_data_path.stem.split('-')[:-1])

            shutil.move(current_data_path, finished_data_path)
            
            is_new_data_exist = False

    return prepare_preprocess_outputs(paths['current_data'], is_new_data_exist)


if __name__ == '__main__':
    init_workflow('/local-storage')