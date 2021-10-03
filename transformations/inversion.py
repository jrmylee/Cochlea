import tensorflow as tf
import numpy as np
from stft import stft, istft

def griffin_lim_tf(S, hparams):
    """TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb and
  https://github.com/keithito/tacotron/blob/master/util/audio.py
  issue: https://github.com/tensorflow/tensorflow/issues/28444
  """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(stft(y, hparams)))
        y = istft(S_complex * angles, hparams)
    return y
