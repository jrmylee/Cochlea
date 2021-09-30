import tensorflow as tf
from loader import get_feature, inv_mel_spec,HParams
import os 
import librosa
import numpy as np
from IPython.display import Audio 
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

hparams = HParams(  
    # spectrogramming
    win_length = 2048,
    n_fft = 2048,
    hop_length= 256,
    ref_level_db = 50,
    min_level_db = -100,
    # mel scaling
    num_mel_bins = 256,
    mel_lower_edge_hertz = 0,
    mel_upper_edge_hertz = 10000,
    # inversion
    power = 1.5, # for spectral inversion
    griffin_lim_iters = 50,
    pad=True
)

def get_dataset(csv, ds_dir):
    audio_filenames, midi_filenames = [], []
    for index, row in csv.iterrows():
        audio, midi = os.path.join(ds_dir, row["audio_filename"]), os.path.join(ds_dir, row["midi_filename"])
        audio_filenames.append(audio)
        midi_filenames.append(midi)
    return audio_filenames, midi_filenames

def add_augmentations(x, delay=1000, sr=22050):
    h = np.zeros(delay)
    h[0] = 1
    h[-1] = 1
    x = np.convolve(x, h)
    noise = np.random.normal(0,.005,len(x))
    return x + noise

def process_audio_file(file_path, save_path, sr=22050, augment=False):
    x, sr = librosa.load(file_path)
    if augment:
        print("augmenting")
        x = add_augmentations(x, sr=sr).astype('float32')
    song_name = file_path.split("/")
    song_name = song_name[len(song_name) - 1]
    print("exporting stft for ", song_name)
    for i in range(0, len(x), 2 * sr):
        index = i // (2 * sr)
        new_file_name = os.path.join(save_path, song_name + "-" + str(index))
        if not os.path.exists(new_file_name):
            y = x[i : i + 2*sr]
            y = tf.convert_to_tensor(y)
            mel_spec = get_feature(y)
            shape = mel_spec.shape
            if shape[0] == 173 and shape[1] == 256:
                np.save(new_file_name, mel_spec.numpy())
    print(new_file_name)

def generate_spectrograms_from_ds(ds_path, ds_filename, save_path, augment=False):
    csv_file = os.path.join(ds_path, ds_filename)
    csv = pd.read_csv(csv_file)
    
    for index, row in csv.iterrows():
        audio, midi = os.path.join(ds_path, row["audio_filename"]), os.path.join(ds_path, row["midi_filename"])
        process_audio_file(audio, save_path, augment=augment)

def get_spectrogram_files(save_path):
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return onlyfiles

def load_spectrogram(file_path):
    mel_spec = np.load(file_path)
    mel_spec = tensor.convert_to_tensor(mel_spec)
    paddings = tf.constant([[0, 3], [0,0]])
    mel_spec = tf.pad(mel_spec, paddings, "CONSTANT")
    mel_spec = tf.expand_dims(mel_spec, -1)
    return mel_spec, ""

def pad_files(onlyfiles):
    for filename in onlyfiles:
        file = np.load(filename)
        shape = file.shape[0] * file.shape[1]
        if shape != 44288:
            print(file.shape)
            os.remove(filename)

generate_spectrograms_from_ds("/home/jerms/data/maestro-v3.0.0", "maestro-v3.0.0.csv", "/home/jerms/disk1/spectrograms/aug", augment=True)
