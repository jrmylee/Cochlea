# IGNORE THIS CODE
# unless you are looking for Tensoflow Data pipeline example code :)

import pandas as pd
import os
import tensorflow as tf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from helpers import *
import numpy as np
import tensorflow_io as tfio
import librosa
import json
from types import SimpleNamespace


AUTOTUNE = tf.data.experimental.AUTOTUNE

default_ds_dir = "/home/jerms/data/maestro-v3.0.0"
ds_mapping_file = "maestro-v3.0.0.csv"
_SEED = 2021

with open("params.json") as file:
    hparams = json.load(file, object_hook=lambda d: SimpleNamespace(**d)

def get_feature(audio):
    spectrogram = spectrogram_tensorflow(audio, hparams)
    mel_spectrogram = mel_spec(spectrogram, hparams)
    return mel_spectrogram

def get_training_set(ds_dir, ds_mapping_file):
    csv_dir = os.path.join(ds_dir, ds_mapping_file)
    
    df = pd.read_csv(csv_dir)
    ds = get_files_from_csv(df, ds_dir)
    
    train_ds, test_ds = get_train_test_set(ds)
    
    return df, train_ds, test_ds

# Returns a Tensorflow Dataset from a dataset mapping CSV
# Returns files in TF dataset format (audio_filepaths, midi_filepaths)
def get_files_from_csv(csv, ds_dir):
    audio_filenames, midi_filenames = [], []
    for index, row in csv.iterrows():
        audio, midi = os.path.join(ds_dir, row["audio_filename"]), os.path.join(ds_dir, row["midi_filename"])
        audio_filenames.append(audio)
        midi_filenames.append(midi)
    
    audio, midi = tf.constant(audio_filenames), tf.constant(midi_filenames)
    dataset = tf.data.Dataset.from_tensor_slices((audio, midi))    
    return dataset

def load_audio(audio_filepath, midi_filepath):
    print("loading audio")
    audio = tf.io.read_file(audio_filepath)
    audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=44100 * 2)
    audio = tfio.audio.resample(audio, 44100, 22050)
    audio = tf.reshape(audio, (22050 * 2, ))
    
    spec_clean = get_feature(audio)
    paddings = tf.constant([[0, 3], [0,0]])
    spec_clean = tf.pad(spec_clean, paddings, "CONSTANT")
    spec_clean = tf.expand_dims(spec_clean, -1)
    
    return spec_clean, audio_filepath

def get_train_test_set(ds, shuffle_buffer_size=1024, batch_size=64):
    test_ds = ds.take(200) 
    train_ds = ds.skip(200)

    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)
    test_ds = test_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
            
    return train_ds, test_ds
