# Versions
```
- python 3.7.0
- tensorflow 2.3.0
- scipy 1.4.1
```

# Abstract 

generate robust (high-dimensional) label such as wav, Glove matrix.

It convert text to wav as label.
To improve classification task, using not just index label but robust (high-dimensional) label is better way.

Using models as follows:
- [Transfomer-TTS](https://github.com/as-ideas/TransformerTTS)

# How to start

## download pretrained weight
download from remote or [this link](https://drive.google.com/file/d/1D7IMwgTxTNmhAXoNPMv0ZMXsXxwn2cc9/view?usp=sharing)
```zsh
# download pretrained weight from remote to model foloder
./download.sh
```

## generate wav vector from text as label
```zsh
# sample predict text to wav
python3 predict_tts.py -f coco_label.txt

# sample generate wav label vector
python3 demo.py

# save label.txt to npy or wav file
python3 generate_wav_label.py 
```


# Roubust(high-dimensional) label effect

## accuracy 

<img width="891" alt="model-exp" src="https://user-images.githubusercontent.com/48679574/171634982-590b8e21-f6ba-48b3-a1f1-27918041e58b.png">

## models output expressions

<img width="836" alt="accu" src="https://user-images.githubusercontent.com/48679574/171634992-6a04accf-3356-47ae-a76c-19380c2f94ff.png">


# Demo 

```python
import numpy as np
from data.audio import Audio
from model.factory import tts_ljspeech

OUTPUT_DIMS = 1000
model = tts_ljspeech(remote_path=None)
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
```


# References
- [Tranformer-TTS](https://github.com/as-ideas/TransformerTTS)
- [Beyond Categorical Label Representations for Image Classification](https://arxiv.org/abs/2104.02226)
