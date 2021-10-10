import os
import re
from collections import defaultdict
from itertools import groupby
from typing import List

import librosa
import numpy as np
from joblib import Parallel, cpu_count, delayed
from scipy.io import wavfile
from text import _clean_text
from tqdm import tqdm


def work(input_path, input_wav_path, txt_file, out_dir, cleaners, sampling_rate, max_wav_value):
    # get base name
    
    base_name = os.path.basename(input_wav_path).split('.')[0]
    speaker = input_wav_path.split('/')[1]

    # clean text
    with open(os.path.join(input_path, txt_file), 'r') as r:
        text = r.read().strip()

    text = _clean_text(text, cleaners)
    if re.match(r'[\w\.\w]+', text):
        text = '. '.join(text.split('.'))

    # mkdir
    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

    # load wav
    wav, _ = librosa.load(os.path.join(input_path, input_wav_path), sampling_rate)

    # make int16 wavform data
    wav = wav / max(abs(wav)) * max_wav_value

    # write wavefile
    wavfile.write(
        os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
        sampling_rate,
        wav.astype(np.int16),
    )

    # write textfile
    with open(
        os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
        "w",
    ) as f1:
        f1.write(text)


def prepare_align(data_relpaths: List, input_path: str, out_dir: str,
                  sample_rate: int, max_wav_value: float, cleaners: List):
    # Match wav and text files
    data_relpaths_dict = defaultdict(dict)
    for key, group in groupby(data_relpaths, lambda item: os.path.basename(item).split('.')[0]):
        filepath = list(group)[0]
        filetype = os.path.basename(filepath).split('.')[1]
        data_relpaths_dict[key][filetype] = filepath

    # re-write
    input_wav_list = []
    input_txt_list = []
    for key, files in data_relpaths_dict.items():
        if len(files) == 1:
            print(f"WARN: data '{key}' is not sufficient. maybe lack of the 'wav' or 'txt' file. skipped.")
            continue
        wav, txt_file = files['wav'], files['txt']
        input_wav_list.append(wav)
        input_txt_list.append(txt_file)

    # do parallel
    Parallel(n_jobs=cpu_count() - 1)(
        delayed(work)
        (input_path, wav_path, txt_file, out_dir, cleaners, sample_rate, max_wav_value)
        for wav_path, txt_file in tqdm(zip(input_wav_list, input_txt_list), desc='prepare align')
    )
