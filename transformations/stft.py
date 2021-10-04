import librosa
import tensorflow as tf

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

def stft_tf(signals, hparams):
    return tf.signal.stft(
        signals,
        hparams.win_length,
        hparams.hop_length,
        hparams.n_fft,
        pad_end=True,
        window_fn=tf.signal.hann_window,
    )