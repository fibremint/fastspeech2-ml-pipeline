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
    import argparse
    import json
    import os
    from pathlib import Path

    from fastspeech2.preprocessing.preprocessor import Preprocessor
    from fastspeech2.utils.tools import parse_kwargs


    def _parse_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--data-base-path', type=str, default='', required=True)
        parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
        parser.add_argument('--input-path', type=str, default='./before-align')
        parser.add_argument('--input-textgrid-path', type=str, default='./aligned')
        parser.add_argument('--preprocess-output-path', type=str, default='./preprocessed')
        parser.add_argument('--config-path', type=str, default='./configs/vctk_preprocess.json')
        
        return parser.parse_args()


    args = _parse_args()

    fs2_data_path = Path(data_base_path) / args.fs2_data_path
    config_path = fs2_data_path / args.config_path
    with open(f'{config_path}', 'r') as f:
        config = json.load(f)

    config = parse_kwargs(Preprocessor, **config)
    Preprocessor(input_path=os.path.join(current_data_path, args.input_path),
                 input_textgrid_path=os.path.join(current_data_path, args.input_textgrid_path),
                 output_path=os.path.join(current_data_path, args.preprocess_output_path),
                 **config).build_from_path()
