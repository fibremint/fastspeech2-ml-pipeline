import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--container-name', type=str, default='torchserve')
parser.add_argument('--container-image', type=str, default='pytorch/torchserve:0.4.2-cpu')
parser.add_argument('--model-store-path', type=str, default='./model-store')
# parser.add_argument('--pvc-name', type=str, default='', required=True)
parser.add_argument('--pvc-name', type=str, default='local-pvc')
parser.add_argument('--prediction-port', type=int, default=8080)
parser.add_argument('--management-port', type=int, default=8081)
parser.add_argument('--metric-port', type=int, default=8082)


opt = parser.parse_args()
