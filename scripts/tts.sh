#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INFERENCE_SERVER_ADDR="http://example.com"

echo "Texts to request TTS inference:"
cat $SCRIPT_DIR/data.txt
echo 
echo "Request inference to FastSpeech2 model deploy server"
curl $INFERENCE_SERVER_ADDR/predictions/fastspeech2?batch_size=8\&max_batch_delay=5000 -T $SCRIPT_DIR/data.txt > $SCRIPT_DIR/output
#curl $INFERENCE_SERVER_ADDR/models?model_name=fastspeech2\&batch_size=8\&max_batch_delay=5000 -T $SCRIPT_DIR/data.txt > $SCRIPT_DIR/output
echo "Converting output to WAV."
python $SCRIPT_DIR/parse_data.py --input-path $SCRIPT_DIR/output --output-path $SCRIPT_DIR/output.wav

read -n 1 -r -s -p $"Complete. press any key to exit."
