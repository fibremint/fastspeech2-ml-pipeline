# ref: https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/waveglow_handler.py
# https://github.com/pytorch/serve/blob/master/docs/custom_service.md
import base64
import os
import re
import uuid

import numpy as np
import soundfile as sf
import torch
from fastspeech2.models.fastspeech2 import FastSpeech2
from fastspeech2.models.fastspeech2 import \
    fast_speech2_vctk as fast_speech2_config
from fastspeech2.text import text_to_sequence
from g2p_en import G2p
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN
from ts.torch_handler.base_handler import BaseHandler

LEXICON_FILENAME = 'librispeech-lexicon.txt'


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


class FastSpeech2Handler(BaseHandler):
    def __init__(self):
        self.fastspeech2_model = None
        self.hifi_gan_model = None
        self.device = None
        self.initialized = False
        self.lexicon = None
        self.g2p = None
        self.cleaners = None


    def initialize(self, context):        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        lexicon_path = os.path.join(model_dir, LEXICON_FILENAME)
        self.lexicon = read_lexicon(lexicon_path)

        self.g2p = G2p()
        self.cleaners = ['english_cleaners']

        serialized_file_path = context.manifest['model']['serializedFile']
        model_checkpoint_path = os.path.join(model_dir, serialized_file_path)

        if not os.path.isfile(model_checkpoint_path):
            raise RuntimeError("model.pt is not exist")

        self.fastspeech2_model = FastSpeech2(**fast_speech2_config()).to(self.device)
        self.fastspeech2_model.eval()
        fastspeech2_checkpoint = torch.load(model_checkpoint_path)
        self.fastspeech2_model.load_state_dict(fastspeech2_checkpoint['model'])

        self.hifi_gan_model = InterfaceHifiGAN(model_name='hifi_gan_v1_universal', device='cuda')


    def inference(self, model_input):
        spk_tensor, txt_tensor, txt_len_tensor, txt_len, pitch_control, energy_control, duration_control = model_input

        _, post_mel, *_ = self.fastspeech2_model(
            spk_tensor,
            txt_tensor,
            txt_len_tensor,
            txt_len,
            p_control = pitch_control,
            e_control = energy_control,
            d_control = duration_control
        )

        pred_wav = self.hifi_gan_model.decode(post_mel.transpose(1, 2)).squeeze()

        return pred_wav
        

    def preprocess(self, inputs: str, speaker: int = 0, pitch_control: float = 1., energy_control: float = 1., duration_control: float = 1.):
        def _process_txt(text):
            # make phones
            phones = []
            words = re.split(r"([,;.\-\?\!\s+])", text)
            for w in words:
                if w.lower() in self.lexicon:
                    phones += self.lexicon[w.lower()]
                else:
                    phones += list(filter(lambda p: p != " ", self.g2p(w)))
            phones = "{" + "}{".join(phones) + "}"
            phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
            phones = phones.replace("}{", " ")

            # make as numpy array
            sequence = np.array(text_to_sequence(phones, self.cleaners))
            return sequence

        input_text = inputs[0]['body']
        
        input_text = input_text.decode()
        sequence = _process_txt(input_text)
        txt_len = len(sequence)

        spk_tensor = torch.LongTensor([speaker]).to(self.device)
        txt_tensor = torch.LongTensor([sequence]).to(self.device)
        txt_len_tensor = torch.LongTensor([txt_len]).to(self.device)

        return spk_tensor, txt_tensor, txt_len_tensor, txt_len, pitch_control, energy_control, duration_control


    def postprocess(self, inference_output, sample_rate: int = 22050):
        audio_data = inference_output.cpu().numpy()

        save_path = f'/tmp/{uuid.uuid4().hex}.wav'
        sf.write(save_path, audio_data, samplerate=sample_rate, format='WAV')

        with open(save_path, 'rb') as output:
            data = output.read()

        os.remove(save_path)

        data = base64.b64encode(data)

        return [data]
