import json
import os

from opts import opt
from preprocess import vctk


def main(opt):            
    input_path = os.path.join(opt.data_base_path, opt.data_input_path)
    with open(f'{opt.data_base_path}/{opt.fs2_data_relpaths_filename}', 'r') as f:
        data_relpaths = json.load(f)

    output_path = os.path.join(opt.data_base_path, opt.data_output_path)

    vctk.prepare_align(data_relpaths=data_relpaths,
                       input_path=input_path,
                       out_dir=output_path,
                       sample_rate=opt.sample_rate,
                       max_wav_value=opt.max_wav_value,
                       cleaners=opt.cleaners)

if __name__ == '__main__':
    main(opt)
