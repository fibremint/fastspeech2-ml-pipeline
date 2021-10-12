'''
 torch-model-archiver --model-name vctk-tts --version 0.1 --serialized-file /home/fibremint/repo/FastSpeech2/data/step_400000.chkpt --export-path model_store --handler model_handler.py --extra-files /local-storage/fs2-data/common/lexicons/librispeech-lexicon.txt

torchserve --start --ncs --model-store model_store --models vctk-tts.mar
'''

'''
--extra-files
'''