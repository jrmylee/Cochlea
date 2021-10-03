import tensorflow as tf
import librosa
import numpy as np
from librosa import mel_frequencies
from spectrogram import spectrogram_tf, inv_spectrogram_tf

# Returns a mel spectrogram from audio clip
def mel_spec_from_audio(audio, hparams):
    spectrogram = spectrogram_tf(audio, hparams)
    mel_spectrogram = mel_spec_from_spec(spectrogram, hparams)
    return mel_spectrogram
    
def mel_spec_from_spec(spectrogram, hparams):
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hparams.num_mel_bins,
        num_spectrogram_bins=int(hparams.n_fft/2)+1,
        sample_rate=22050,
        lower_edge_hertz=hparams.mel_lower_edge_hertz,
        upper_edge_hertz=hparams.mel_upper_edge_hertz,
        dtype=tf.dtypes.float32,
        name=None,
    )
    mel_f = mel_frequencies(
        n_mels=hparams.num_mel_bins + 2,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )
    enorm = tf.dtypes.cast(
        tf.expand_dims(tf.constant(2.0 / (mel_f[2 : hparams.num_mel_bins + 2] - mel_f[:hparams.num_mel_bins])), 0),
        tf.float32)
    mel_matrix = tf.multiply(mel_matrix, enorm)
    mel_matrix = tf.divide(mel_matrix, tf.reduce_sum(mel_matrix, axis=0))
    mel_spectrogram = tf.tensordot(spectrogram,mel_matrix, 1)
    return mel_spectrogram

def inv_mel_spec(mel_spectrogram, hparams):
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hparams.num_mel_bins,
        num_spectrogram_bins=int(hparams.n_fft/2)+1,
        sample_rate=22050,
        lower_edge_hertz=hparams.mel_lower_edge_hertz,
        upper_edge_hertz=hparams.mel_upper_edge_hertz,
        dtype=tf.dtypes.float32,
        name=None,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mel_inversion_matrix = tf.constant(
            np.nan_to_num(
                np.divide(mel_matrix.numpy().T, np.sum(mel_matrix.numpy(), axis=1))
            ).T
        )
    mel_spectrogram_inv = tf.tensordot(mel_spectrogram,tf.transpose(mel_inversion_matrix), 1)
    reconstructed_y_mel = inv_spectrogram_tf(np.transpose(mel_spectrogram_inv), hparams)
    
    return reconstructed_y_mel
