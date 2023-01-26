import numpy as np
from data.audio import Audio
from model.factory import tts_ljspeech


OUTPUT_DIMS = 1000
model = tts_ljspeech(remote_path=True)
audio = Audio.from_config(model.config)


def spectrogram2wav(output):
    """Convert spectrogram to wav (with griffin lim)"""
    return audio.reconstruct_waveform(output['mel'].numpy().T)
    
with open('coco_label.txt') as f:
    for i, label in enumerate(f):
       out = model.predict(label)
       wav = spectrogram2wav(out)
       wav = wav.reshape(1, -1)
       wav = np.resize(wav, (1, OUTPUT_DIMS))
       print(i, wav.shape, type(wav), wav.max(), wav.min())
       #if store_mel:
           #np.save(os.path.join(SAVE_DIR, 'wav{}'.format(i)), out['mel'].numpy())
       #else:
           #audio.save_wav(wav, os.path.join(SAVE_DIR, 'wav{}.wav'.format(i)))
               

