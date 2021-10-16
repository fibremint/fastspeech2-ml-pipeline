'''
base_image: fs2-runtime-base
'''

# import argparse
# import json
# import os
# from pathlib import Path

# from fastspeech2.preprocessing.preprocessor import Preprocessor
# from fastspeech2.utils.tools import parse_kwargs


# def _parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-base-path', type=str, default='', required=True)
#     parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
#     parser.add_argument('--input-path', type=str, default='./before-align')
#     parser.add_argument('--input-textgrid-path', type=str, default='./aligned')
#     parser.add_argument('--preprocess-output-path', type=str, default='./preprocessed')
#     parser.add_argument('--config-path', type=str, default='./configs/vctk_preprocess.json')
    
#     return parser.parse_args()


def preprocess(data_base_path: str, current_data_path: str):
    # import argparse
    # import json
    # import os
    # from pathlib import Path

    # from app.preprocessing.preprocessor import Preprocessor
    # from fastspeech2.utils.tools import parse_kwargs

    # configs = {
    #     'fs2_data_path': './fs2-data',
    #     'input_path': './before-align',
    #     'input_textgrid_path': './aligned',
    #     'preprocess_output_path': './preprocessed',
    #     'config_path': './configs/vctk_preprocess.json'
    # }

    # def _parse_args():
    #     parser = argparse.ArgumentParser()
    #     # parser.add_argument('--data-base-path', type=str, default='', required=True)
    #     parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
    #     parser.add_argument('--input-path', type=str, default='./before-align')
    #     parser.add_argument('--input-textgrid-path', type=str, default='./aligned')
    #     parser.add_argument('--preprocess-output-path', type=str, default='./preprocessed')
    #     parser.add_argument('--config-path', type=str, default='./configs/vctk_preprocess.json')
        
    #     return parser.parse_args()


    # args = _parse_args()

    # fs2_data_path = Path(data_base_path) / configs['fs2_data_path']
    # config_path = fs2_data_path / configs['config_path']
    # with open(f'{config_path}', 'r') as f:
    #     config = json.load(f)

    # config = parse_kwargs(Preprocessor, **config)
    # Preprocessor(input_path=os.path.join(current_data_path, configs['input_path']),
    #              input_textgrid_path=os.path.join(current_data_path, configs['input_textgrid_path']),
    #              output_path=os.path.join(current_data_path, configs['preprocess_output_path']),
    #              **config).build_from_path()

    import os

    from fastspeech2.preprocessing.preprocessor import Preprocessor
    from fastspeech2.utils.tools import parse_kwargs


    config = {
        "input_path": "./before-align",
        "input_textgrid_path": "./aligned",
        "output_path": "./preprocessed",

        "audio_params": {
        "sample_rate": 22050,
        "n_fft": 1024,
        "window_size": 1024,
        "hop_size": 256,
        "num_mels": 80,
        "fmin": 0,
        "fmax": 8000,
        "is_center": True
        },

        "pitch_feature": "phoneme_level",
        "energy_feature": "phoneme_level",
        "pitch_norm": True,
        "energy_norm": True,

        "sample_rate": 22050,
        "max_wav_value": 32767.0,
        "validation_rate": 0.05,

        "cleaners": ["english_cleaner"]
    }

    config = parse_kwargs(Preprocessor, **config)

    config["input_path"] = os.path.join(current_data_path, config['input_path'])
    config["input_textgrid_path"] = os.path.join(current_data_path, config["input_textgrid_path"])
    config["output_path"] = os.path.join(current_data_path, config["output_path"]) 

    # make preprocessor
    preprocessor = Preprocessor(**config)
    preprocessor.build_from_path()
