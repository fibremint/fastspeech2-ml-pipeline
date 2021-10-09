import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mfa-input-path', type=str, default='')
parser.add_argument('--lexicon-path', type=str, default='')
parser.add_argument('--mfa-output-path', type=str, default='')
opt = parser.parse_args()

cpu_threads = str(os.cpu_count())

# ref: https://stackoverflow.com/questions/89228/how-to-execute-a-program-or-call-a-system-command
subprocess.run(['mfa',
                'align',
                opt.mfa_input_path,
                opt.lexicon_path,
                'english',
                opt.mfa_output_path,
                '-j',
                cpu_threads])
