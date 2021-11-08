import tensorflow as tf
import os 
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from transformations.spec_helpers import *
from transformations.stft import nn_stft
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
def stft_transformation(file_path, augment_fn, hparams):
    print("Loading song")
    x, sr = librosa.load(file_path)
    x = librosa.util.normalize(x)
    print("Loaded!")

    chunk_length = 2048 * 22
    if len(x) % chunk_length != 0:
        multiple = np.ceil(len(x) / chunk_length)
        pad_amount = chunk_length * multiple - len(x)
        x = np.pad(x, (0, int(pad_amount)), 'constant', constant_values=(0, 0))

    # split in 2 second chunks and export to files 
    arr = []
    for i in range(0, len(x), 2048 * 22):
        y = x[i : i + 2048 * 22]

        if augment_fn:
            length = len(y)
            y = augment_fn(y)[:length]

        stft = nn_stft(y)
        arr.append(stft)
    
    return np.array(arr)

# Transforms each audio file in MAESTRO dataset into transformed files
# params
# ds_path: path to dataset
# mapping_filename: filename for csv containing audio track data
# save_path: path to save directory
# augment_fn: function to augment audio clips prior to transform.  Optional
def generate_spectrograms_from_ds(ds_path, mapping_filename, save_path, augment_fn, hparams):
    csv_path = os.path.join(ds_path, mapping_filename)
    csv = pd.read_csv(csv_path)
    
    for index, row in csv.iterrows():
        full_audio_path, full_midi_path = os.path.join(ds_path, row["audio_filename"]), os.path.join(ds_path, row["midi_filename"])

        song_name = full_audio_path.split("/")
        song_name = song_name[len(song_name) - 1]
        first_song_path = os.path.join(save_path, song_name + "-0.npy")
        
        if not os.path.exists(first_song_path):
            stfts = stft_transformation(full_audio_path, augment_fn, hparams)
            save_specgrams(stfts, save_path, song_name)

