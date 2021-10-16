# import argparse
# import json
# import os

# from app.preprocess import vctk


# def _parse_args():
#     default_cleaners = ['english_cleaners']

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-base-path', type=str, default='', required=True)
#     parser.add_argument('--data-input-path', type=str, default='./raw-data')
#     parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
#     parser.add_argument('--data-output-path', type=str, default='./before-align')
#     parser.add_argument('--sample-rate', type=int, default=22050)
#     parser.add_argument('--max-wav-value', type=float, default=32767.0)
#     # ref: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
#     # https://stackoverflow.com/questions/43660172/why-does-argparse-include-default-value-for-argsional-argument-even-when-argument
#     parser.add_argument('-c', '--cleaners', action='append', type=str)

#     args = parser.parse_args()
#     configs['cleaners = configs['cleaners if configs['cleaners else default_cleaners

#     return args


def prepare_align(current_data_path: str, dataset_name: str) -> None:
    # import argparse
    # import json
    # import os
# 
    # from app.preprocess import vctk

    # configs = {
    #     'data_input_path': './raw-data',
    #     'fs2_data_relpaths_filename': 'data-relpaths.json',
    #     'data_output_path': './before-align',
    #     'sample_rate': 22050,
    #     'max_wav_value': 32767.0,
    #     'cleaners': ['english_cleaners']
    # }


    # def _parse_args():
    #     default_cleaners = ['english_cleaners']

    #     parser = argparse.ArgumentParser()
    #     # parser.add_argument('--data-base-path', type=str, default='', required=True)
    #     parser.add_argument('--data-input-path', type=str, default='./raw-data')
    #     parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
    #     parser.add_argument('--data-output-path', type=str, default='./before-align')
    #     parser.add_argument('--sample-rate', type=int, default=22050)
    #     parser.add_argument('--max-wav-value', type=float, default=32767.0)
    #     # ref: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    #     # https://stackoverflow.com/questions/43660172/why-does-argparse-include-default-value-for-argsional-argument-even-when-argument
    #     parser.add_argument('-c', '--cleaners', action='append', type=str)

    #     args = parser.parse_args()
    #     configs['cleaners = configs['cleaners if configs['cleaners else default_cleaners

    #     return args


    # args  = _parse_args()

    # input_path = os.path.join(current_data_path, configs['data_input_path'])
    # with open(f'{current_data_path}/{configs["fs2_data_relpaths_filename"]}', 'r') as f:
    #     data_relpaths = json.load(f)

    # output_path = os.path.join(current_data_path, configs['data_output_path'])

    # vctk.prepare_align(data_relpaths=data_relpaths,
    #                    input_path=input_path,
    #                    out_dir=output_path,
    #                    sample_rate=configs['sample_rate'],
    #                    max_wav_value=configs['max_wav_value'],
    #                    cleaners=configs['cleaners'])
    import json
    from fastspeech2.preprocessing import vctk
    from fastspeech2.utils.tools import parse_kwargs
    from pathlib import Path

    # config = {
    #     'data_input_path': './raw-data',
    #     'fs2_data_relpaths_filename': 'data-relpaths.json',
    #     'data_output_path': './before-align',
    #     'sample_rate': 22050,
    #     'max_wav_value': 32767.0,
    #     'cleaners': ['english_cleaners']
    # }

    config = {
        "input_path": "./raw-data",
        "output_path": "./before-align",
        "sample_rate": 22050,
        "max_wav_value": 32767.0,
        "cleaners": ["english_cleaners"],
        "train_speakers": "configs/vctk_train_speakers.txt",
        "data_relpaths_filename": "data-relpaths.json"
    }


    if dataset_name == 'vctk':
        current_data_path = Path(current_data_path)
        with open(f'{current_data_path / config["data_relpaths_filename"]}', 'r') as f:
            data_relpaths = json.load(f)

        config = parse_kwargs(vctk.prepare_align, **config)
        config['input_path'] = current_data_path / config['input_path']
        config['output_path'] = current_data_path / config['output_path']

        vctk.prepare_align(data_relpaths=data_relpaths, **config)
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented! you should choose in vctk or libri_tts')
