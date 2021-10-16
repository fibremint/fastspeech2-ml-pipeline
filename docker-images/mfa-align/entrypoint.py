import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-base-path', type=str, default='', required=True)
parser.add_argument('--lexicon-path', type=str, default='./fs2-data/common/lexicons/librispeech-lexicon.txt')

parser.add_argument('--current-data-path', type=str, default='', required=True)
parser.add_argument('--mfa-input-path', type=str, default='./before-align')
parser.add_argument('--mfa-output-path', type=str, default='./aligned')
opt = parser.parse_args()

cpu_threads = str(os.cpu_count())

# ref: https://stackoverflow.com/questions/89228/how-to-execute-a-program-or-call-a-system-command
subprocess.run(['mfa',
                'align',
                os.path.join(opt.current_data_path, opt.mfa_input_path),
                os.path.join(opt.data_base_path, opt.lexicon_path),
                'english',
                os.path.join(opt.current_data_path, opt.mfa_output_path),
                '-j',
                cpu_threads])
