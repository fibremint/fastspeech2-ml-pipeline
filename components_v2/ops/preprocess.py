def preprocess(data_base_path: str, current_data_path: str):
    import os

    from fastspeech2.preprocessing.preprocessor import Preprocessor
    from fastspeech2.utils.tools import parse_kwargs

    from fs2_env import get_paths

    paths = get_paths(base_path=data_base_path, current_data_path=current_data_path)

    config = {
        # "input_path": "./before-align",
        # "input_textgrid_path": "./aligned",
        # "output_path": "./preprocessed",

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

    preprocessor = Preprocessor(input_path=paths['before_align'],
                                input_textgrid_path=paths['aligned'], 
                                output_path=paths['preprocessed'],
                                **config)
    preprocessor.build_from_path()


if __name__ == '__main__':
    preprocess(data_base_path='/local-storage', current_data_path='/local-storage/fs2-data/data/20211017-012441')