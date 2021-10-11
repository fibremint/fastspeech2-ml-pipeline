import datetime
import glob
import json
import os
import shutil
import sys
from pathlib import Path

from opts import opt


def main(opt):
    # check path args
    if not opt.data_base_path:
        sys.exit('ERR: base path is not set')

    base_path = Path(opt.data_base_path)
    if not base_path.exists():
        sys.exit('ERR: base path is not exists')

    curr_data_path = str(None)
    raw_data_path = base_path / opt.raw_data_path
    is_new_data_exist = raw_data_path.exists()
    if not is_new_data_exist:
        # sys.exit('WARN: raw data path is not exists. aborted.')
        with open(f'/tmp/{opt.fs2_dupl_data_relpaths_filename}', 'w') as f:
            json.dump([], f)

        with open(f'/tmp/{opt.fs2_dupl_data_relpaths_filename}', 'w') as f:
            json.dump([], f)
            
        print('WARN: raw data path is not exists. aborted.')

    else:
        print('INFO: found raw data path')

        # move raw data to intermediate data path
        fs2_data_path = base_path / opt.fs2_base_path / opt.fs2_data_path
        curr_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        curr_data_path = fs2_data_path / f'{curr_datetime}-intermediate'
        os.makedirs(curr_data_path)

        print('INFO: move raw data path to intermediate data path')
        shutil.move(str(raw_data_path), str(curr_data_path))

        # get relpaths of current data
        # ref: https://stackoverflow.com/questions/44994604/python-glob-os-relative-path-making-filenames-into-a-list
        curr_rawdata_path = curr_data_path / opt.raw_data_path
        curr_data_wav_relpaths = [os.path.relpath(data_abspath, curr_rawdata_path)
                                for data_abspath in glob.glob(f'{curr_data_path}/{opt.raw_data_path}/wav48/*/*.wav')]
        curr_data_txt_relpaths = [os.path.relpath(data_abspath, curr_rawdata_path)
                                for data_abspath in glob.glob(f'{curr_data_path}/{opt.raw_data_path}/txt/*/*.txt')]

        # get all of the relpath of data in ./fs2-data/data path 
        relpaths = set()
        data_relpaths = glob.glob(f'{fs2_data_path}/*/{opt.fs2_data_relpaths_filename}')
        for data_relpath in data_relpaths:
            with open(data_relpath, 'r') as f:
                curr_relpaths = json.load(f)
                relpaths.update(set(curr_relpaths))
        
        # write current data relpaths
        curr_data_relpaths = set(curr_data_wav_relpaths) | set(curr_data_txt_relpaths)

        # write duplicated relpaths
        duplicated_file_relpaths = curr_data_relpaths.intersection(relpaths)
        if duplicated_file_relpaths:
            print('WARN: some of the data are duplicated')

        with open(f'{curr_data_path}/{opt.fs2_dupl_data_relpaths_filename}', 'w') as f:
            duplicated_file_relpaths = sorted(list(duplicated_file_relpaths))
            json.dump(duplicated_file_relpaths, f, indent=2)
        
        shutil.copy(f'{curr_data_path}/{opt.fs2_dupl_data_relpaths_filename}', f'/tmp/{opt.fs2_dupl_data_relpaths_filename}')

        # write not duplicated relpaths
        preprocess_target_relpaths = curr_data_relpaths.difference(relpaths)
        with open(f'{curr_data_path}/{opt.fs2_data_relpaths_filename}', 'w') as f:
            preprocess_target_relpaths = sorted(list(preprocess_target_relpaths))
            json.dump(preprocess_target_relpaths, f, indent=2)

        shutil.copy(f'{curr_data_path}/{opt.fs2_data_relpaths_filename}', f'/tmp/{opt.fs2_data_relpaths_filename}')

        if not preprocess_target_relpaths:
            print('WARN: all of the file names are duplicated. current workflow will be aborted.')
            finished_data_path = curr_data_path.parent / '-'.join(curr_data_path.stem.split('-')[:-1])
            shutil.move(curr_data_path, finished_data_path)
            
            is_new_data_exist = False

    # save values to set output variable
    with open('/tmp/curr-data-path', 'w') as f:
        f.write(str(curr_data_path))

    with open('/tmp/is-new-data-exist', 'w') as f:
        f.write(str(is_new_data_exist))


if __name__ == '__main__':
    main(opt)
