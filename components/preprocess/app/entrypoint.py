from opts import opt
from utils.tools import parse_kwargs
import json
from preprocessor import Preprocessor
import os
import pathlib


def main(opt):
    app_path = pathlib.Path(__file__).parent.resolve()
    config_path = app_path / opt.config_path
    with open(f'{config_path}', 'r') as f:
        config = json.load(f)

    config = parse_kwargs(Preprocessor, **config)
    Preprocessor(input_path=os.path.join(opt.data_base_path, opt.input_path),
                 input_textgrid_path=os.path.join(opt.data_base_path, opt.input_textgrid_path),
                 output_path=os.path.join(opt.data_base_path, opt.preprocess_output_path),
                 **config).build_from_path()


if __name__ == '__main__':
    main(opt)