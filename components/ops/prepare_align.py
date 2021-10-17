def prepare_align(data_base_path: str, current_data_path: str, dataset_name: str) -> None:
    import json
    from fastspeech2.preprocessing import vctk
    from fastspeech2.utils.tools import parse_kwargs
    from pathlib import Path

    from fs2_env import get_paths
    paths = get_paths(base_path=data_base_path, current_data_path=current_data_path)

    config = {
        # "input_path": "./raw-data",
        # "output_path": "./before-align",
        "sample_rate": 22050,
        "max_wav_value": 32767.0,
        "cleaners": ["english_cleaners"],
        "train_speakers": "configs/vctk_train_speakers.txt",
        "data_relpaths_filename": "data-relpaths.json"
    }


    if dataset_name == 'vctk':
        with open(paths['data_relpaths'], 'r') as f:
            data_relpaths = json.load(f)

        config = parse_kwargs(vctk.prepare_align, **config)

        vctk.prepare_align(data_relpaths=data_relpaths, 
                           input_path=paths['raw_data'], 
                           output_path=paths['before_align'],
                           **config)
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented! you should choose in vctk or libri_tts')

if __name__ == '__main__':
    prepare_align('/local-storage', '/local-storage/fs2-data/data/20211017-012441', 'vctk')