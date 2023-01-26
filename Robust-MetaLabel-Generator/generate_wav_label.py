import numpy as np
import os
from data.audio import Audio
from model.factory import tts_ljspeech


SAVE_DIR = 'results'
OUTPUT_DIMS = 1000

os.makedirs(SAVE_DIR, exist_ok=True)
model = tts_ljspeech(remote_path=None)
audio = Audio.from_config(model.config)


def spectrogram2wav(output):
    """Convert spectrogram to wav (with griffin lim)"""
    return audio.reconstruct_waveform(output['mel'].numpy().T)

def create_robust_label(store_mel=True):
    with open('coco_label.txt') as f:
        for i, label in enumerate(f):
            out = model.predict(str(label))
            if store_mel:
                mel = out['mel'].numpy()
                mel = mel.reshape(1, -1)
                mel = np.resize(mel, (1, OUTPUT_DIMS))
                print(i, mel.shape, type(mel), mel.max(), mel.min())
                np.save(os.path.join(SAVE_DIR, 'mel{}'.format(i)), mel.flatten())
            else:
                wav = spectrogram2wav(out)
                wav = wav.reshape(1, -1)
                wav = np.resize(wav, (1, OUTPUT_DIMS))
                print(i, wav.shape, type(wav), wav.max(), wav.min())
                audio.save_wav(wav.flatten(), os.path.join(SAVE_DIR, 'wav{}.wav'.format(i)))
               
if __name__=='__main__':
    create_robust_label(store_mel=True)


