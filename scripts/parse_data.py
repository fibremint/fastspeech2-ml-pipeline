import argparse
import base64


parser = argparse.ArgumentParser()
parser.add_argument('--input-path', type=str, default='./')
parser.add_argument('--output-path', type=str, default='./')
args = parser.parse_args()

print('INFO: load output file')
with open(args.input_path, 'r') as f:
    data = f.read()

print('INFO: decode base64 encoded file')
data_parsed = base64.b64decode(data)

print('INFO: decoded as binary file')
with open(args.output_path, 'bw') as f:
    f.write(data_parsed)
