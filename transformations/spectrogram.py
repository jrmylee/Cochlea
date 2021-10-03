import tensorflow as tf
from normalize import normalize_spectrogram_tf, denormalize_spectrogram_tf
from inversion import griffin_lim_tf
from stft import stft_tf

def spectrogram_tf(y, hparams):
    D = stft_tf(y, hparams)
    S = _amp_to_db_tensorflow(tf.abs(D)) - hparams.ref_level_db
    return normalize_spectrogram_tf(S, hparams)

def inv_spectrogram_tf(spectrogram, hparams):
    """Converts spectrogram to waveform using librosa"""
    S = _db_to_amp_tensorflow(
        denormalize_spectrogram_tf(spectrogram, hparams) + hparams.ref_level_db
    )  # Convert back to linear
    return griffin_lim_tf(S ** hparams.power, hparams)  # Reconstruct phase

# Helper Functions

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def _amp_to_db_tensorflow(x):
    return 20 * _tf_log10(tf.clip_by_value(tf.abs(x), 1e-5, 1e100))
