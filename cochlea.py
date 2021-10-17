import tensorflow as tf
import os 
import librosa
import numpy as np
import pandas as pd

# Splits a single audio clip into 2 second intervals, augments, and transforms it.  Saves resulting transformations to disk
# params
# file_path: full path to audio file
# save_path: path to save directory
# transform_fn: transformation function, STFT, Mel Spectrogram, etc...
# augment_fn: function to augment audio clip prior to transformation. Optional
# sr: sample rate
def save_audio_transformation(file_path, save_path, transform_fn, augment_fn, hparams):
    # extract song name
    song_name = file_path.split("/")
    song_name = song_name[len(song_name) - 1]
    first_song_path = os.path.join(save_path, song_name + "-0.npy")
    
    if not os.path.exists(first_song_path):
        print("Loading ", song_name)
        x, sr = librosa.load(file_path)
        print("Loaded!")
        
        # zero pad the file so we use the entire clip
        chunk_length = 2 * sr
        if len(x) % chunk_length != 0:
            multiple = np.ceil(len(x) / chunk_length)
            pad_amount = chunk_length * multiple - len(x)
            pad_amount = int(pad_amount)
            x = np.pad(x, (0, pad_amount), 'constant', constant_values=(0, 0))

        # split in 2 second chunks and export to files 
        for i in range(0, len(x), 2 * sr):
            index = i // (2 * sr)
            new_file_name = os.path.join(save_path, song_name + "-" + str(index))
            if not os.path.exists(new_file_name):
                y = x[i : i + 2*sr]
                if augment_fn:
                    y = augment_fn(y, sr=sr)
                transformed_y = transform_fn(y, hparams)
                np.save(new_file_name, transformed_y)
        print(new_file_name)
    else:
        print(song_name, " exists!")

# Transforms each audio file in MAESTRO dataset into transformed files
# params
# ds_path: path to dataset
# mapping_filename: filename for csv containing audio track data
# save_path: path to save directory
# augment_fn: function to augment audio clips prior to transform.  Optional
def generate_spectrograms_from_ds(ds_path, mapping_filename, save_path, transform_fn, augment_fn, hparams):
    csv_path = os.path.join(ds_path, mapping_filename)
    csv = pd.read_csv(csv_path)
    
    for index, row in csv.iterrows():
        full_audio_path, full_midi_path = os.path.join(ds_path, row["audio_filename"]), os.path.join(ds_path, row["midi_filename"])
        save_audio_transformation(full_audio_path, save_path, transform_fn, augment_fn, hparams)
