import tensorflow as tf
import os 
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from transformations.spec_helpers import *
import librosa

tf.disable_v2_behavior()

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    arr = []
    for i in range(0, len(l), group_size):
        arr.append(np.array(l[i:i+group_size]))
    return np.array(arr)

def save_specgrams(specgrams, save_path, song_name):
    for i in range(specgrams.shape[0]):
        spec = specgrams[i, :, :]
        save_file_path = os.path.join(save_path, song_name + "-" + str(i) + ".npy")
        np.save(save_file_path, spec)

# Librosa Transformations
def librosa_transformation(file_path, augment_fn, hparams):
    print("Loading song")
    x, sr = librosa.load(file_path)
    print("Loaded!")

    chunk_length = 2048 * 20
    if len(x) % chunk_length != 0:
        multiple = np.ceil(len(x) / chunk_length)
        pad_amount = chunk_length * multiple - len(x)
        x = np.pad(x, (0, int(pad_amount)), 'constant', constant_values=(0, 0))

    # split in 2 second chunks and export to files 
    arr = []
    angles = []
    for i in range(0, len(x), 2048 * 20):
        index = i // (2048 * 20)
        y = x[i : i + 2048 * 20]
        stft = librosa.stft(y)
        mag = np.abs(stft)
        angle = np.angle(stft)
        arr.append(mag)
        angles.append(angle)
    
    return np.array(arr), np.array(angles)

# Splits a single audio clip into 2 second intervals, augments, and transforms it.  Saves resulting transformations to disk
# params
# file_path: full path to audio file
# save_path: path to save directory
# transform_fn: transformation function, STFT, Mel Spectrogram, etc...
# augment_fn: function to augment audio clip prior to transformation. Optional
# sr: sample rate
def audio_transformation(file_path, spec_helper, augment_fn, hparams):
    # extract song name
    print("Loading song")
    x, sr = librosa.load(file_path)
    print("Loaded!")
    
    # zero pad the file so we use the entire clip
    chunk_length = 2 * sr
    if len(x) % chunk_length != 0:
        multiple = np.ceil(len(x) / chunk_length)
        pad_amount = chunk_length * multiple - len(x)
        pad_amount = int(pad_amount)
        x = np.pad(x, (0, pad_amount), 'constant', constant_values=(0, 0))

    chunked_audio = group_list(x, chunk_length) # chunk audio into 2 second intervals

    if augment_fn:
        for i in range(chunked_audio.shape[0]):
            chunked_audio[i, :] = augment_fn(chunked_audio[i, :], hparams)

    chunked_audio = np.expand_dims(chunked_audio, 2)
    with tf.Session() as sess:
        input_tensor = tf.convert_to_tensor(chunked_audio)
        specgrams = spec_helper.waves_to_specgrams(input_tensor).eval(session=sess)
    return specgrams

# Transforms each audio file in MAESTRO dataset into transformed files
# params
# ds_path: path to dataset
# mapping_filename: filename for csv containing audio track data
# save_path: path to save directory
# augment_fn: function to augment audio clips prior to transform.  Optional
def generate_spectrograms_from_ds(ds_path, mapping_filename, save_path, transform_fn, augment_fn, hparams):
    csv_path = os.path.join(ds_path, mapping_filename)
    csv = pd.read_csv(csv_path)
    
    # helper = SpecgramsHelper(audio_length=44100, spec_shape=[128, 1024], overlap=0.75, sample_rate=22050, mel_downscale=1, ifreq=True, discard_dc=True)

    for index, row in csv.iterrows():
        full_audio_path, full_midi_path = os.path.join(ds_path, row["audio_filename"]), os.path.join(ds_path, row["midi_filename"])

        song_name = full_audio_path.split("/")
        song_name = song_name[len(song_name) - 1]
        first_song_path = os.path.join(save_path, song_name + "-0.npy")
        
        if not os.path.exists(first_song_path):
            specs, angles = librosa_transformation(full_audio_path, augment_fn, hparams)
            save_specgrams(specs, save_path, song_name)

