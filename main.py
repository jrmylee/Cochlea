from cochlea import generate_spectrograms_from_ds
from augmentations.naive import *
from transformations.stft import stft_stacked
import json
from types import SimpleNamespace
import random

print("reading impulses")
impulses = room_impulses()
print("reading background noises!  Might take a min")
background_noises = get_noises()

def composed_aug(x, sr):
    x = apply_noise(x)
    x = apply_background_sound(x, background_noises)
    h = random.choice(impulses)
    y = np.convolve(x, h)
    y = np.resize(y, len(x))
    return y

print("beginning preprocess")
with open("params.json") as file:
    hparams = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    generate_spectrograms_from_ds(
    ds_path="/home/jerms/data/maestro-v3.0.0", 
    mapping_filename="maestro-v3.0.0.csv", 
    save_path="/home/jerms/disk1/stft_augmented", 
    transform_fn=stft_stacked,
    augment_fn=composed_aug,
    hparams=hparams)
