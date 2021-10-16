# def _parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-base-path', type=str, default='', required=True)
#     parser.add_argument('--lexicon-path', type=str, default='./fs2-data/common/lexicons/librispeech-lexicon.txt')

#     parser.add_argument('--current-data-path', type=str, default='', required=True)
#     parser.add_argument('--mfa-input-path', type=str, default='./before-align')
#     parser.add_argument('--mfa-output-path', type=str, default='./aligned')
    
#     return parser.parse_args()


def mfa_align(data_base_path: str, current_data_path: str) -> None:
    import subprocess
    import os

    configs = {
        'lexicon_path': './fs2-data/common/lexicons/librispeech-lexicon.txt',
        'mfa_input_path': './before-align',
        'mfa_output_path': './aligned'
    }


    # def _parse_args():
    #     parser = argparse.ArgumentParser()
    #     # parser.add_argument('--data-base-path', type=str, default='', required=True)
    #     parser.add_argument('--lexicon-path', type=str, default='./fs2-data/common/lexicons/librispeech-lexicon.txt')

    #     # parser.add_argument('--current-data-path', type=str, default='', required=True)
    #     parser.add_argument('--mfa-input-path', type=str, default='./before-align')
    #     parser.add_argument('--mfa-output-path', type=str, default='./aligned')
        
    #     return parser.parse_args()


    # args = _parse_args()
    cpu_threads = os.cpu_count()

    # ref: https://stackoverflow.com/questions/89228/how-to-execute-a-program-or-call-a-system-command
    # subprocess.run(['conda', 'run', '-n', 'aligner', '/bin/bash', '-c', '"', 'mfa', 'align',
    #                 os.path.join(current_data_path, configs['mfa_input_path']),
    #                 os.path.join(data_base_path, configs['lexicon_path']),
    #                 'english',
    #                 os.path.join(current_data_path, configs['mfa_output_path']),
    #                 '-j',
    #                 cpu_threads, '"'])

    subprocess.run(f'conda run -n aligner /bin/bash -c "mfa align ' \
        f'{os.path.join(current_data_path, configs["mfa_input_path"])} ' \
        f'{os.path.join(data_base_path, configs["lexicon_path"])} '
        f'english ' \
        f'{os.path.join(current_data_path, configs["mfa_output_path"])} ' \
        f'-j ' \
        f'{cpu_threads}"', shell=True)
