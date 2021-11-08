import librosa
import torch
from nnAudio import Spectrogram
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec_layer = Spectrogram.STFT(n_fft=2048, hop_length=512, output_format="Magnitude")
spec_layer.to(device)

def nn_stft(x, n_fft=2048, hop_length=512, output_format="Magnitude"):
    x = torch.tensor(x, device=device).float()
    return np.array(spec_layer(x).cpu()).squeeze(0)

def stft(y, hparams):
    return librosa.stft(
        y=y,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
    )

def istft(y, hparams):
    return librosa.istft(
        y, hop_length=hparams.hop_length, win_length=hparams.win_length
    )

# returns a stft with the real and imag components stacked
# shape of (n_fft / 2 + 1, t, 2)
def stft_stacked(y, hparams):
    transformed = stft(y, hparams)
    real, imag = transformed.real, transformed.imag
    stacked = np.stack((real, imag), axis=-1)
    return stacked

# use this one when istft is fixed!
def istft_tf(stfts, hparams):
    return tf.signal.inverse_stft(
        stfts, hparams.win_length, hparams.hop_length, hparams.n_fft
    )

def stft_tf(x, frame_length, frame_step):
    pad_amount = 2 * (frame_length - frame_step)
    x = tf.pad(x, [[pad_amount // 2, pad_amount // 2]], 'REFLECT')
    
    f = tf.contrib.signal.frame(x, frame_length, frame_step, pad_end=False)
    w = tf.contrib.signal.hann_window(frame_length, periodic=True)
    spectrograms_T = tf.spectral.rfft(f * w, fft_length=[frame_length])
        
    return spectrograms_T
