def mfa_align(data_base_path: str, current_data_path: str) -> None:
    import subprocess
    import os

    from fs2_env import get_paths
    paths = get_paths(base_path=data_base_path, current_data_path=current_data_path)

    configs = {
        'lexicon_path': './fs2-data/common/lexicons/librispeech-lexicon.txt',
        'mfa_input_path': './before-align',
        'mfa_output_path': './aligned'
    }

    cpu_threads = os.cpu_count()

    # ref: https://stackoverflow.com/questions/89228/how-to-execute-a-program-or-call-a-system-command
    subprocess.run(f'conda run -n aligner /bin/bash -c "mfa align ' \
        f'{paths["before_align"]} ' \
        f'{paths["lexicon"]} '
        f'english ' \
        f'{paths["aligned"]} ' \
        f'-j ' \
        f'{cpu_threads}"', shell=True)
