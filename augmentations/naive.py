import numpy as np
import random
import os
import librosa
from audiomentations import AddBackgroundNoise
import pandas as pd 

# impulse response lib 
irlib_path = "/global/scratch/users/jrmylee/datasets/EchoThiefImpulseResponseLibrary/Venues"

# Human Sounds
esc_path = "/global/scratch/users/jrmylee/datasets/ESC-50-master/audio"

background_noise_filter = AddBackgroundNoise(esc_path)

def random_augmentations(x, delay=1000, sr=22050):
    h = np.zeros(delay)
    h[0] = 1
    h[-1] = 1
    x = np.convolve(x, h)
    noise = np.random.normal(0,.005,len(x))
    x = x + noise
    return x.astype('float32')

def apply_room_impulse(x, h):
    return np.convolve(x, h)

def apply_noise(x):
    samples = len(x)
    max_value = random.uniform(.0001, .001)
    return x + np.random.normal(0, max_value, (samples))
    
def apply_background_sound(x, sounds, weight=0.1, sr=22050):
    random_sound = random.choice(sounds)
    x = librosa.util.normalize(x)
    noise = librosa.util.normalize(random_sound)
    
    return (1 - weight) * x + weight * noise

def get_noises():
    esc_path = "/global/scratch/users/jrmylee/datasets/ESC-50-master/audio"
    esc_csv = "/global/scratch/users/jrmylee/datasets/ESC-50-master/meta/esc50.csv"
    esc_data = pd.read_csv(esc_csv)
    
    categories_of_interest = ["clapping", "footsteps", "coughing", "sneezing", "crying_baby", "breathing"]
    rows_of_interest = esc_data[esc_data.category.isin(categories_of_interest)]
    
    noises = []
    
    for index, row in rows_of_interest.iterrows():
        x, sr = librosa.load(os.path.join(esc_path, row['filename']))
        x = x[0 : sr * 2]
        noises.append(x)
    return noises
def room_impulses(ir_path=irlib_path):    
    impulses = []
    for file in os.listdir(ir_path):
        if file.endswith(".wav"):
            x, sr = librosa.load(os.path.join(ir_path, file))
            impulses.append(x)
    return impulses
def get_random_room_impulse(ir_path=irlib_path):
    random_file = random.choice(os.listdir(ir_path)) 
    h, sr = librosa.load(os.path.join(ir_path, random_file))
    
    return h
